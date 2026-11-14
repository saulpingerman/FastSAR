"""A quick look at a spotlight collection: a reduced-resolution picture by polar format, from the part of the file
that resolution needs.

    out = fastsar.quicklook('scene_CPHD.cphd', 'look.png')              # 2048 pixels on the longer side
    out = fastsar.quicklook('scene_CPHD.cphd', size=4096, correct=False)
    out = fastsar.quicklook('scene_CPHD.cphd', 'look.nitf', resolution=1.0)
    out['display']                                                      # uint8 [rows, cols], shadows down

Resolution first. The header and a few hundred of the per-pulse parameters give the collection geometry, the band
and the tasked area; no sample is read for it. The picture is the tasked area's bounding rectangle on a slant-plane
grid whose range axis is the mid-aperture line of sight, with pixels of one size on the ground along the ground
range direction and across it: size pixels along its longer side, or pixels of 0.96 times the resolution asked for.
That pixel is 0.96 of the reciprocal of the band on each axis, the library's sampling of a complex image, so the cut
is the central run of frequency samples and the central run of pulses with those bands, never more than the
collection holds. Contiguous runs keep the sample and pulse spacings, and with them the unambiguous extents; a
picture is cut to 0.92 of those extents where the tasked area is larger. The picture stays on the slant plane: it is
the ground sheared by the plane's twist (zero at broadside, tens of degrees for a squinted look at a steep grazing
angle), and the product gives that angle.

Only those pulses and samples are read (io.read_cphd's pulses= and samples=), and the cut history is formed by polar
format in four stages, each a pass whose cost follows the picture, not the collection:

  1. Range gate. Each pulse's inverse transform is cut to the range bins the picture spans (a raised-cosine edge
     beyond them) and transformed back, to a spectrum sampled 1.5 times finer than the gate needs; a 16-tap
     Kaiser-windowed sinc then reads it at the range wavenumbers of the rectangular raster (the keystone).
  2. Azimuth gate. Each range wavenumber's pulse sequence is transformed, cut to the picture's azimuth extent, and
     transformed back at the pulse spacing that extent needs; the same kernel reads it at the raster's azimuth
     wavenumbers, at the pulse positions of the collection's own, uneven, spacing in the tangent of the look angle.
  3. A two-dimensional FFT of the raster. Uncorrected (correct=False) this is the picture: polar format's planar
     wavefronts leave each scatterer displaced by a smooth field that grows with the square of its distance from
     the center (the product's notes give its size in pixels).
  4. Corrected (the default): the transform is taken on a raster padded to twice the sampling and read at each
     pixel's apparent position (pfa2.distortion's polynomial) with a 6-tap Lanczos kernel.

Devices. 'cpu': scipy's FFTs on all cores and the C++ kernels (pfa_cpu.cpp, quicklook_cpu.cpp). 'cuda': the same
stages in CuPy with the CUDA kernels of polar format (pfa2.py) and one more for the uneven pulse spacing; the history
is windowed and gated on the device in blocks of pulses. 'jax' and 'tpu': polar format's resampling steps are
gathers, slow on a TPU's vector unit, so the same cut history is formed on the same grid by the factorized former
(ImageFormer; precision 'single-pass' on a TPU), and a note says so. 'auto': tpu, else cuda, else cpu.

A moving-beam collection (stripmap, sliding spotlight) is refused: polar format assumes one scene reference point,
and a moving beam's azimuth resolution is set per pixel by the processed Doppler band, not by a run of pulses, so
the cut above does not apply. process_cphd forms those.
"""
import math
import os
import threading
import time

import numpy as np

from . import io

__all__ = ['quicklook', 'write_png', 'read_png']

C = 299792458.0
TAPS, KAISER = 16, 8.0        # the resampling kernel of the range and azimuth stages (polar format's)
PASS = 0.65                   # the part of a gated transform's period that the kernel reads to its accuracy
OVERSAMPLE, LANCZOS = 2, 6    # the corrected image: the raster's padding and the taps of the final kernel
PHASES = 2048                 # fractional positions per sample in the host kernels' weight tables
_LIB = []
_POOL = []
_LOCK = threading.Lock()


def _lib():
    """The C++ kernels of quicklook_cpu.cpp, compiled on first use as the other CPU kernels."""
    if TAPS != 16:
        raise RuntimeError('quicklook_cpu.cpp is written for 16 taps')
    if not _LIB:
        import ctypes
        from ._build import shared_object, default_flags
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quicklook_cpu.cpp')) as fh:
            src = fh.read()
        L = ctypes.CDLL(shared_object('quicklook_cpu', src, default_flags() + ['-std=c++17', '-fopenmp-simd']))
        i, n, d, fl, p = ctypes.c_int, ctypes.c_long, ctypes.c_double, ctypes.c_float, ctypes.c_void_p
        L.ql_resample_rows.argtypes = [p, n, n, p, p, n, p, i, p]
        L.ql_resample_cols.argtypes = [p, n, n, p, d, d, p, n, d, d, n, p, i, p]
        L.ql_gather.argtypes = [p, n, n, n, n, n, n, i, i, i, i, i, i, d, d, d, d, d, p, p, p, i, i, d, d, i, fl, i, p, p,
                                i, i, p, n]
        L.ql_weight_pad.argtypes = [p, n, n, p, p, n, n, p]
        L.ql_resample_rows.restype = L.ql_resample_cols.restype = L.ql_gather.restype = L.ql_weight_pad.restype = None
        _LIB.append(L)
    return _LIB[0]


def _workers():
    """Threads for the host stages: the CPUs this process may run on, no more than a control group's quota grants
    or OMP_NUM_THREADS allows."""
    from .terrain import _workers as quota
    n = quota()
    omp = os.environ.get('OMP_NUM_THREADS', '').strip()
    if omp.isdigit() and int(omp) > 0:
        n = min(n, int(omp))
    return max(1, n)


def _blocks(fn, n, chunk):
    """fn(slice) over blocks of n rows on a pool of _workers() threads kept between calls."""
    sl = [slice(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    w = _workers()
    if len(sl) == 1 or w == 1:
        for s in sl:
            fn(s)
        return
    with _LOCK:
        if not _POOL or _POOL[0][0] != w:
            from concurrent.futures import ThreadPoolExecutor
            _POOL[:] = [(w, ThreadPoolExecutor(w, thread_name_prefix='fastsar-quicklook'))]
        pool = _POOL[0][1]
    list(pool.map(fn, sl))


def _unit(v):
    return v / np.linalg.norm(v)


def _fft_size(n):
    from .pfa2 import _fft_size as f
    return int(f(int(max(n, 2))))


def _taylor(n, nbar=4, sll=35.0):
    """The Taylor window of n points as the formers apply it (scipy.signal.windows.taylor with norm=False), computed
    here: importing scipy.signal takes a quarter of a second, more than a small quick look."""
    A = np.arccosh(10 ** (sll / 20)) / np.pi
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)
    Fm = np.empty(nbar - 1)
    for i, m in enumerate(ma):
        num = (-1.0) ** i * np.prod(1 - m * m / s2 / (A ** 2 + (ma - 0.5) ** 2))
        Fm[i] = num / (2 * np.prod(1 - m * m / ma[:i] ** 2.0) * np.prod(1 - m * m / ma[i + 1:] ** 2.0))
    return 1 + 2 * Fm @ np.cos(2 * np.pi * ma[:, None] * (np.arange(n) - n / 2.0 + 0.5) / n)


_TABLES = {}


def _polyphase(kind, taps, q):
    """The weights of a resampling kernel for the host kernels (quicklook_cpu.cpp), float32 [q + 1, taps]: row r holds
    the taps for the fractional position r / q, at the offsets r / q - (t - taps / 2 + 1), normalised to unit sum.
    kind 'kaiser': polar format's Kaiser-windowed sinc (pfa2._kernel, beta KAISER); 'lanczos': a sinc times a sinc
    window, the final resampling's kernel (pfa2's gathers compute it)."""
    key = (kind, taps, q)
    if key not in _TABLES:
        from .pfa2 import _kernel
        d = np.arange(q + 1)[:, None] / q - (np.arange(taps)[None, :] - taps // 2 + 1)
        if kind == 'kaiser':
            w = _kernel(d, taps, KAISER)
        else:
            w = np.where(np.abs(d) < taps // 2, np.sinc(d) * np.sinc(d / (taps // 2)), 0.0)
        _TABLES[key] = np.ascontiguousarray(w / w.sum(1, keepdims=True), np.float32)
    return _TABLES[key]


# ------------------------------------------------------------------------------------------------------- the plan

def _plan(r, cphd, channel, size, resolution, extent, ref_freq, looks, notes):
    """The cut and the grid, from the header and a few hundred per-pulse parameters (no sample is read): dict of the
    pulses and samples to read, the picture's center, slant-plane axes and plane normal (ECF), its half extents and
    spacings along the axes (m), its pixel on the ground, and the collection's facts the notes and the product
    quote."""
    from .cphd import _image_area
    m = r.cphd_meta
    v03 = str(getattr(r, 'cphd_version', '') or '').startswith('0.3')
    channel, arrays, ids, pols = io._cphd_channel(r, m, v03, channel)
    ch = arrays[channel]
    P, K = int(ch.NumVectors), int(ch.NumSamples)
    if m.Global.DomainType != 'FX':
        raise ValueError('needs a frequency-domain (FX) CPHD; time-of-arrival (TOA) CPHD is not supported')
    # the per-pulse parameters of some 512 pulses across the collection, and the last one
    if hasattr(r, 'read_pvp_rows'):
        step = max(1, P // 512)
        idx = np.arange(0, P, step)
        vb = r.read_pvp_rows(channel, 0, P, step)
        if idx[-1] != P - 1:
            idx, vb = np.r_[idx, P - 1], np.concatenate([vb, r.read_pvp_rows(channel, P - 1, P)])
        get = lambda n: np.array(vb[n], np.float64) if n in vb.dtype.names else None
    else:                                          # a reader without row access: the whole block
        idx = np.arange(P)

        def get(n):
            a = r.read_pvp_variable(n, channel)
            return None if a is None else np.array(a, np.float64)
    tx, rcv, srp = get('TxPos'), get('RcvPos'), get('SRPPos')
    sc0, scss = (get('Fx0'), get('Fx_SS')) if v03 else (get('SC0'), get('SCSS'))
    fx1, fx2 = (get('Fx1'), get('Fx2')) if v03 else (get('FX1'), get('FX2'))
    if v03 and int(getattr(m.Global, 'RefFreqIndex', 0) or 0):
        if ref_freq is None:
            raise ValueError('CPHD 0.3 file with RefFreqIndex set: its frequencies are offsets from a reference the '
                             "file does not give; pass reader_options=dict(ref_freq=...) (io.read_cphd)")
        sc0 = sc0 + float(ref_freq)
    ok = np.isfinite(tx).all(1) & np.isfinite(rcv).all(1) & np.isfinite(srp).all(1) & np.isfinite(sc0) & (scss > 0)
    if ok.sum() < 3:
        raise ValueError(f'{int(ok.sum())} of the {len(idx)} pulses looked at have valid positions: nothing to image')
    idx, tx, rcv, srp, sc0, scss = (a[ok] for a in (idx, tx, rcv, srp, sc0, scss))
    apc = 0.5 * (tx + rcv)
    f0, df = float(np.median(sc0)), float(np.median(scss))
    k_lo, k_hi = 0, K
    if fx1 is not None and fx2 is not None:        # the band every pulse holds
        lo, hi = np.nanmax((fx1[ok] - sc0) / scss), np.nanmin((fx2[ok] - sc0) / scss)
        if np.isfinite(lo) and np.isfinite(hi) and hi - lo >= 2:
            k_lo, k_hi = max(0, int(np.ceil(lo - 1e-6))), min(K, int(np.floor(hi + 1e-6)) + 1)
    Kv = k_hi - k_lo
    moved = float(np.linalg.norm(srp.max(0) - srp.min(0)))
    if moved > C / (2 * K * df):
        raise ValueError(f'quicklook forms spotlight collections by polar format, and this collection has a moving '
                         f'beam (its scene reference point moves {moved:.0f} m over the pulses: stripmap or sliding '
                         'spotlight). Use fastsar.process_cphd (or form_cphd), which form moving beams')
    mid = len(idx) // 2
    s0 = srp[mid]
    lat, lon, _ = (np.radians(float(np.ravel(v)[0])) for v in io.ecf_to_geodetic(s0))
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    corners, refpt = _image_area(cphd, dict(R=np.eye(3), origin=np.zeros(3)), r)        # ECF
    if extent is None and corners is None:
        raise ValueError('the CPHD states no planar image area: give extent=(across, along) the line of sight, in '
                         'meters on the ground')
    c = (s0 if refpt is None else refpt) if extent is not None else corners.mean(0)
    c = c - ((c - (c if refpt is None else refpt)) @ up) * up          # on the plane through the reference point
    # the slant plane at the mid-aperture pulse: e2 the line of sight (away from the radar), e1 across it in the
    # plane of the line of sight and the velocity, their normal on the upward side
    r0 = float(np.linalg.norm(c - apc[mid]))
    e2 = (c - apc[mid]) / r0
    vel = np.gradient(apc, idx, axis=0)[mid]
    n = _unit(np.cross(e2, vel))
    if n @ up < 0:
        n = -n
    if n @ up < 0.02:
        raise ValueError('the slant plane is within a degree of vertical (a look at the nadir): its picture has no '
                         'ground scale in range. Use fastsar.process_cphd')
    e1 = np.cross(e2, n)
    # the ground under the plane. Carried to the ground along the plane's normal, a meter along e2 is s2 meters along
    # the ground range direction, and a meter along e1 is s1 meters across that direction and tan(twist) s1 along
    # it: the picture is the ground sheared by the twist (none at broadside)
    g1, g2 = (e - n * (e @ up) / (n @ up) for e in (e1, e2))
    s2 = float(np.linalg.norm(g2))
    s1 = float(np.linalg.norm(g1 - (g1 @ g2) / s2 ** 2 * g2))
    twist = float(np.degrees(np.arctan2((g1 @ g2) / s2, s1)))
    if extent is None:
        # the corners where the picture shows them: the points of the plane with their range and range rate
        vh = _unit(vel - (vel @ n) * n)
        v1, v2 = float(vh @ e1), float(vh @ e2)
        w = corners - apc[mid]
        al = w @ vh
        pe = np.sqrt(np.maximum(np.sum(w * w, 1) - al * al, 0.0)) * np.sign(v1)
        Hx, Hy = float(np.abs(al * v1 - pe * v2).max()), float(np.abs(al * v2 + pe * v1 - r0).max())
    else:
        Hx, Hy = 0.5 * float(extent[0]) / s1, 0.5 * float(extent[1]) / s2
    u = apc - c
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = (u @ e1) / -(u @ e2)                         # the tangent of the look angle in the plane
    sgn = 1.0 if t[-1] > t[0] else -1.0
    if not np.all(np.diff(t) * sgn > 0):
        raise ValueError('the look angle does not change monotonically along the pulses, which polar format needs; '
                         'use fastsar.process_cphd')
    dt = float(np.abs(np.gradient(t, idx)).max())    # per pulse
    f_hi = f0 + (k_hi - 1) * df
    fcv = f0 + 0.5 * (k_lo + k_hi - 1) * df
    # the unambiguous extents (m, in the plane): a picture keeps inside them, with room for the gates' edges
    Xr, Xa = C / (2 * df), C / (2 * f_hi * dt)
    for name, H, X, what in (('azimuth', Hx, Xa, 'pulse rate'), ('range', Hy, Xr, 'frequency step')):
        lim = 0.46 * X
        if H > lim:
            notes.append(f'the area spans {2 * H:.0f} m in {name} on the slant plane, more than the unambiguous {name} '
                         f'extent of the {what} ({X:.0f} m): the picture is cut to the central {2 * lim:.0f} m')
            Hx, Hy = (lim, Hy) if name == 'azimuth' else (Hx, lim)
    # the pixel on the ground, and the finest one the collection holds on both axes
    bx_max, by_max = 2 * fcv / C * float(abs(t[-1] - t[0])), 2 * Kv * df / C
    p_min = 0.96 * max(s1 / bx_max, s2 / by_max)
    if resolution == 'full':
        p = p_min
    elif resolution is not None:
        p = 0.96 * float(resolution)
    else:
        p = max(2 * Hx * s1, 2 * Hy * s2) / int(size)
    if p < p_min * (1 - 1e-9):
        notes.append(f'the collection holds a resolution of {p_min / 0.96:.3f} m on both ground axes; the picture has '
                     f'that, not the {p / 0.96:.3f} m asked for')
        p = p_min
    spx, spy = p / s1, p / s2
    nx, ny = (2 * int(np.ceil(H / sp - 1e-6)) for H, sp in ((Hx, spx), (Hy, spy)))
    # the cut: the central samples and pulses whose bands are 0.96 / spacing; for several looks, as many such runs
    # side by side as were asked for and the collection holds
    Kc = int(min(Kv, max(2, round(0.96 / spy * C / (2 * df)))))
    lr = int(max(1, min(looks[1], Kv // Kc)))
    k0 = (k_lo + (Kv - lr * Kc) // 2) // 16 * 16      # from a multiple of 16: the reader's rows as they are read
    k0 += 16 if k0 < k_lo and k0 + 16 + lr * Kc <= k_hi else 0
    if k0 + lr * Kc > K:
        lr, Kc = (lr - 1, Kc) if lr > 1 else (1, K - k0)
    fc = f0 + (k0 + 0.5 * (lr * Kc - 1)) * df
    need = 0.96 / spx * C / (2 * fc)                  # the span of t of one look
    la = int(max(1, min(looks[0], np.floor(abs(t[-1] - t[0]) / need + 1e-9))))
    order = np.argsort(t)
    ends = np.interp([t[mid] - la * need / 2, t[mid] + la * need / 2], t[order], idx[order].astype(np.float64))
    pa, pb = int(max(idx[0], np.floor(ends.min()))), int(min(idx[-1], np.ceil(ends.max()))) + 1
    if (pb - pa) // la < 2 * TAPS:
        raise ValueError(f'the picture asked for needs {(pb - pa) // la} pulses, fewer than the {2 * TAPS} polar '
                         'format resamples: ask for more pixels or a finer resolution')
    if (la, lr) != tuple(looks):
        notes.append(f'{la} by {lr} looks (sub-apertures by sub-bands), not the {looks[0]} by {looks[1]} asked for: '
                     'the collection holds no more at this resolution')
    graze = float(np.degrees(np.arcsin(np.clip(-(e2 @ up), -1, 1))))
    return dict(channel=channel, pulses=(pa, pb), samples=(k0, k0 + lr * Kc), looks=(la, lr), P=P, K=K,
                valid=(k_lo, k_hi), center=c, e1=e1,
                e2=e2, normal=n, up=up, Hx=Hx, Hy=Hy, spx=spx, spy=spy, nx=nx, ny=ny, pixel=p, s=(s1, s2),
                resolution=p / 0.96, finest=p_min / 0.96, extents=(Xa, Xr), graze=graze, twist=twist, f0=f0, df=df)


# --------------------------------------------------------------------------------------------- the cut collection

def _collection(cphd, size=2048, resolution=None, extent=None, channel=0, reader_options=None, looks=(1, 1)):
    """Plan the cut and read it: dict(S [P, K] (referenced to the grid center, unweighted), ant [P, 3] (about that
    center, local frame), fmin, df, nx, ny, spx, spy, e1, e2 (local): what form_image, ExactFormer and _pfa take;
    center [3] (local), meta (read_cphd's), plan, notes, timing (plan, read, reference: seconds)). looks: (sub-apertures,
    sub-bands): S then holds that many runs of pulses and of samples side by side (plan['looks'] of them)."""
    from ._xmlmeta import reading
    opts = dict(reader_options or {})
    bad = sorted(set(opts) - {'troposphere', 'phase_sign', 'ref_freq', 'drop_flagged', 'regrid_tol'})
    if bad:
        raise ValueError(f'reader_options: {", ".join(bad)} not among troposphere, phase_sign, ref_freq, '
                         'drop_flagged, regrid_tol (io.read_cphd)')
    timing, notes = {}, []
    t0 = time.perf_counter()
    with reading(cphd, 'CPHD file', wrap=False):
        r = io._open_cphd(cphd)
        plan = _plan(r, cphd, channel, size, resolution, extent, opts.get('ref_freq'), tuple(looks), notes)
        timing['plan'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        col, meta = io._read_cphd(cphd, None, plan['channel'], True, opts.get('regrid_tol', 1e-3),
                                  opts.get('drop_flagged', False), opts.get('troposphere'), opts.get('phase_sign'),
                                  opts.get('ref_freq'), plan['pulses'], plan['samples'], reader=r)
    timing['read'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not meta['fixed_ref']:
        raise ValueError('the scene reference point moves over the pulses read: a moving beam, which '
                         'fastsar.process_cphd forms')
    S, f0, df = col['S'], float(col['fmin']), float(col['df'])
    P, K = S.shape
    R = np.asarray(meta['R'], np.float64)
    c = io.ecf_to_local(plan['center'], meta)
    e1, e2 = plan['e1'] @ R.T, plan['e2'] @ R.T
    rel = np.asarray(col['ant'], np.float64) - c
    # referenced to the grid center, |ant - center| (as form_cphd moves a spotlight's reference)
    dref = np.asarray(meta['ref'], np.float64) - np.linalg.norm(rel, axis=1)
    fk = f0 + df * np.arange(K)

    def freq(sl, out):
        out[:] = fk
    io._phase_rows(S, freq, 2 * dref / C, -1, linear=(f0, df))
    timing['reference'] = time.perf_counter() - t0
    notes = list(meta.get('notes', [])) + notes
    return dict(S=S, ant=rel, fmin=f0, df=df, nx=plan['nx'], ny=plan['ny'], spx=plan['spx'], spy=plan['spy'], e1=e1,
                e2=e2, center=c, meta=meta, plan=plan, notes=notes, timing=timing)


# ------------------------------------------------------------------------------------------------ polar format

def _geometry(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, correct, ov=OVERSAMPLE):
    """The float64 setup of the quick look's polar format for a cut history: the raster, the two gates, the
    resampling positions and the distortion model."""
    from .pfa2 import distortion
    from .sim import Collect
    P = len(ant)
    r0 = np.linalg.norm(ant, axis=1)
    u = ant / r0[:, None]
    ur = -(u @ e2)
    if ur.min() <= 0:
        raise ValueError('the grid\'s range axis e2 must point away from the radar')
    t = (u @ e1) / ur
    sgn = 1.0 if t[-1] > t[0] else -1.0
    if not np.all(np.diff(t) * sgn > 0):
        raise ValueError('polar format needs the look angle in the image plane to change monotonically from pulse to '
                         'pulse, and it does not; use fastsar.process_cphd or form_image with algorithm=\'ffbp\'')
    dts = (t[-1] - t[0]) / (P - 1)                                    # signed, per pulse
    dtp = np.gradient(t)
    f = fmin + df * np.arange(K)
    fc = fmin + (K // 2) * df
    Hx, Hy = 0.5 * nx * spx, 0.5 * ny * spy
    # where polar format puts the scatterers (a few hundred pulses give the fit)
    sub = np.unique(np.round(np.linspace(0, P - 1, min(P, 257))).astype(int))
    dist = distortion(Collect(fmin=fmin, df=df, K=K, ant=ant[sub], res=0.5), nx, ny, spx, spy, e1, e2)
    shx, shy = (float(v) for v in (dist['max_shift_px'][0] * spx, dist['max_shift_px'][1] * spy))
    # the gates. Range: the slant delays of the picture's points, exactly, over the aperture, with the displacement
    # (an uncorrected picture shows points from that far outside) and a pixel; azimuth: the picture's extent with
    # the displacement and the spread of a point's azimuth bins that the uneven pulse spacing makes
    gx, gy = max(8 * spx, 0.03 * Hx), max(8 * spy, 0.03 * Hy)
    xs, ys = (Hx + shx) * np.array([-1, 0, 1]), (Hy + shy) * np.array([-1, 0, 1])
    pts = (xs[:, None, None] * e1 + ys[None, :, None] * e2).reshape(-1, 3)
    dR = np.linalg.norm(pts[:, None] - ant[sub][None], axis=2) - r0[sub][None]
    Er = float(np.abs(dR).max()) + 2 * spy
    Ex = (Hx + shx) * float(1 + np.abs(dtp / dts - 1).max()) + 2 * spx
    # the raster: its periods hold the picture, the displacement and a gate's far edge
    nfy = _fft_size(int(np.ceil((Hy + shy + Er + gy) / spy)) + 2)
    nfx = _fft_size(int(np.ceil((Hx + shx + Ex + gx) / spx)) + 2)
    dkr, dka = 2 * math.pi / (nfy * spy), 2 * math.pi / (nfx * spx)
    kr_min, kr_max = 4 * math.pi * f[0] / C * ur.min(), 4 * math.pi * f[-1] / C * ur.max()
    nkr = int((kr_max - kr_min) / dkr) + 1
    if nkr > nfy:                                   # a band wider than the spacing resolves: its center
        kr_min += ((nkr - nfy) // 2) * dkr
        nkr = nfy
    kr = kr_min + dkr * np.arange(nkr)
    kr_max = kr[-1]
    ka_lo, ka_hi = kr_max * t.min(), kr_max * t.max()
    nka = int((ka_hi - ka_lo) / dka) + 1
    if nka > nfx:
        ka_lo += ((nka - nfx) // 2) * dka
        nka = nfx
    # range gate: K samples at k0 of Kf, the profile's bins C / (2 df Kf), Nr bins kept (or the Kf there are, in a
    # longer array of zeros), the spectrum back on Nr samples
    Xr = C / (2 * df)
    rho = 2 * (Er + gy) / PASS / Xr
    k0 = int(np.ceil((TAPS // 2 + 1) / rho)) + 1
    Kf = _fft_size(K + 2 * k0)
    Nr = _fft_size(int(np.ceil(rho * Kf)))
    bin_r = Xr / Kf
    scale = C / (4 * math.pi * ur * df)             # frequency samples per unit of kr, per pulse
    alpha = ((kr_min * scale - fmin / df) + k0) * (Nr / Kf)
    beta = dkr * scale * (Nr / Kf)
    # azimuth gate: P pulses at p0 of Pf, the bins of row kr 2 pi / (kr |dts| Pf), Na kept, the pulses back on Na
    Xa = 2 * math.pi / (kr_max * abs(dts))          # the shortest period, the highest wavenumber's
    rho_a = 2 * (Ex + gx) / PASS / Xa
    p0 = int(np.ceil((TAPS // 2 + 1) / rho_a)) + 1
    Pf = _fft_size(P + 2 * p0)
    Na = _fft_size(int(np.ceil(rho_a * Pf)))
    bin_a = Xa / Pf
    # the row position (in the Na pulses) of a tangent t, from the pulses' own t: a table at four points per pulse
    # over the aperture and a few pulses beyond it (linear continuation)
    nt = 4 * (P + 2 * TAPS) + 1
    tg = (t[0] - TAPS * dts) + (dts / 4) * np.arange(nt)
    pp = np.arange(P, dtype=np.float64)
    if sgn > 0:
        ptab = np.interp(tg, t, pp)
    else:
        ptab = np.interp(tg[::-1], t[::-1], pp[::-1])[::-1]
    out_lo, out_hi = (tg - t[0]) / dts < 0, (tg - t[-1]) / dts > 0
    ptab[out_lo] = ((tg - t[0]) / dtp[0])[out_lo]
    ptab[out_hi] = (P - 1 + (tg - t[-1]) / dtp[-1])[out_hi]
    qtab = (ptab + p0) * (Na / Pf)
    # weights: the polar Jacobian (the raster's density of samples: backprojection sums the samples as they come)
    wk = fc / f
    wp = (abs(dts) / np.abs(dtp)) / ur ** 2
    kr_cw = 4 * math.pi * fc / C
    gain = (dkr * C / (4 * math.pi * df)) * (dka / (kr_cw * abs(dts)))
    return dict(P=P, K=K, nx=nx, ny=ny, spx=spx, spy=spy, nfx=nfx, nfy=nfy, dkr=dkr, dka=dka, kr_min=kr_min, nkr=nkr,
                ka_lo=ka_lo, nka=nka, kr=kr, k0=k0, Kf=Kf, Nr=Nr, bin_r=bin_r, Er=Er, gy=gy, alpha=alpha, beta=beta,
                p0=p0, Pf=Pf, Na=Na, bin_a=bin_a, Ex=Ex, gx=gx, qtab=qtab, tg0=float(tg[0]), tg_inv=float(4 / dts),
                wk=wk, wp=wp, gain=gain, dist=dist, shift=(shx, shy), ov=ov, t=t, dts=dts)


def _taper(n, edge, guard, binsize):
    """A gate over the n bins of a transform, in the transform's own (wrapped) order: 1 within `edge` meters of
    bin 0, a raised cosine to 0 over the next `guard` meters, 0 beyond (float32)."""
    k = np.arange(n)
    d = np.minimum(k, n - k) * binsize
    return np.where(d <= edge, 1.0, np.where(d >= edge + guard, 0.0,
                                             0.5 + 0.5 * np.cos((d - edge) * (math.pi / guard)))).astype(np.float32)


def _wrap_cut(x, n, axis):
    """The n bins about bin 0 of a transform x along an axis, in wrapped order (bin 0 first, the negative bins at
    the end): cut out of a longer transform, or set into a longer array of zeros."""
    N = x.shape[axis]
    if n == N:
        return x
    xp = np if isinstance(x, np.ndarray) else __import__('cupy')
    keep = min(n, N)
    pos, neg = keep - keep // 2, keep // 2
    sl = lambda a, b: tuple(slice(a, b) if k == axis else slice(None) for k in range(x.ndim))
    shape = tuple(n if k == axis else s for k, s in enumerate(x.shape))
    out = xp.empty(shape, x.dtype) if n < N else xp.zeros(shape, x.dtype)
    out[sl(0, pos)] = x[sl(0, pos)]
    out[sl(n - neg, n)] = x[sl(N - neg, N)]
    return out


def _pfa_cpu(S, g, window, correct, complex_image, timing):
    """The quick look's polar format on the host -> the image, range-major [ny, nx]: complex64 with backprojection's
    phase (complex_image) or the amplitude (float32). The passes over the arrays are fused and run on all cores:
    NumPy's products in row blocks on the pool, scipy's transforms with its workers, the kernels with their own
    threads."""
    import ctypes
    import scipy.fft as sf
    lib = _lib()
    W = _workers()
    P, K, Kf, Nr, k0 = g['P'], g['K'], g['Kf'], g['Nr'], g['k0']
    nkr, nka, nfy, nfx, nx, ny = g['nkr'], g['nka'], g['nfy'], g['nfx'], g['nx'], g['ny']
    tab = _polyphase('kaiser', TAPS, PHASES)
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    t0 = time.perf_counter()
    wk = np.ascontiguousarray(g['wk'] * (_taylor(K) if window else 1.0), np.float32)
    wp = np.ascontiguousarray(g['wp'] * (_taylor(P) if window else 1.0), np.float32)
    # 1. range gate, in blocks of pulses on the pool: weighted and padded (one pass), to range profiles, the gate,
    # and back to the spectrum on Nr samples
    tap_r = _taper(Nr, g['Er'], g['gy'], g['bin_r'])
    keep = min(Nr, Kf)
    pos, neg = keep - keep // 2, keep // 2
    G = np.empty((P, Nr), np.complex64)

    def rows(sl):
        n = sl.stop - sl.start
        buf = np.empty((n, Kf), np.complex64)
        lib.ql_weight_pad(ctypes.c_void_p(S.ctypes.data + sl.start * S.strides[0]), n, K, ptr(wk),
                          ctypes.c_void_p(wp.ctypes.data + 4 * sl.start), k0, Kf, ptr(buf))
        prof = sf.ifft(buf, axis=1, overwrite_x=True)
        gate = np.empty((n, Nr), np.complex64) if Nr <= Kf else np.zeros((n, Nr), np.complex64)
        np.multiply(prof[:, :pos], tap_r[:pos], out=gate[:, :pos])
        np.multiply(prof[:, Kf - neg:], tap_r[Nr - neg:], out=gate[:, Nr - neg:])
        G[sl] = sf.fft(gate, axis=1, overwrite_x=True)
    _blocks(rows, P, int(max(1, min(64, (1 << 21) // (8 * Kf)))))
    timing['range gate'] = timing.get('range gate', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    A = np.empty((P, nkr), np.complex64)
    alpha, beta = np.ascontiguousarray(g['alpha'], np.float64), np.ascontiguousarray(g['beta'], np.float64)
    lib.ql_resample_rows(ptr(G), P, Nr, ptr(alpha), ptr(beta), nkr, ptr(tab), PHASES, ptr(A))
    del G
    timing['range resampling'] = timing.get('range resampling', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # 2. azimuth gate: the pulses of every range wavenumber transformed (padded to Pf), cut to Na bins, and back on
    # Na pulses. The gate carries the image's scale and the phase that sets the pulses p0 rows into the Na
    Pf, Na, p0 = g['Pf'], g['Na'], g['p0']
    ov = g['ov']
    Nfft = ov * nfy if correct else nfy
    k = np.arange(Na)
    k = np.where(k < Na - Na // 2, k, k - Na)
    tap_a = (_taper(Na, g['Ex'], g['gx'], g['bin_a']) * (g['gain'] * Nfft * Na / Pf)
             * np.exp(-2j * np.pi * (k * p0 % Pf) / Pf)).astype(np.complex64)
    prof = sf.fft(A, n=Pf, axis=0, workers=W)
    del A
    keep = min(Na, Pf)
    pos, neg = keep - keep // 2, keep // 2
    gate = np.empty((Na, nkr), np.complex64)
    rb = int(max(1, (1 << 18) // nkr))

    def cut(sl):
        if sl.stop <= pos:                                             # the bins from 0 on
            np.multiply(prof[sl], tap_a[sl, None], out=gate[sl])
        elif sl.start >= Na - neg:                                     # the bins before 0, from the transform's end
            np.multiply(prof[sl.start - Na + Pf:sl.stop - Na + Pf], tap_a[sl, None], out=gate[sl])
        else:
            gate[sl] = 0                                               # a longer array than the transform: zeros
    edges = sorted({0, pos, Na - neg, Na})
    for a_, b_ in zip(edges, edges[1:]):
        _blocks(lambda sl: cut(slice(a_ + sl.start, a_ + sl.stop)), b_ - a_, rb)
    del prof
    A2 = np.ascontiguousarray(sf.ifft(gate, axis=0, workers=W, overwrite_x=True))
    del gate
    timing['azimuth gate'] = timing.get('azimuth gate', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    B = np.empty((nkr, nka), np.complex64)
    inv_kr = np.ascontiguousarray(1.0 / g['kr'])
    qtab = np.ascontiguousarray(g['qtab'])
    lib.ql_resample_cols(ptr(A2), Na, nkr, ptr(inv_kr), g['ka_lo'], g['dka'], ptr(qtab), len(qtab), g['tg0'],
                         g['tg_inv'], nka, ptr(tab), PHASES, ptr(B))
    del A2
    timing['azimuth resampling'] = timing.get('azimuth resampling', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    out = np.empty((ny, nx), np.complex64 if complex_image else np.float32)
    cb = int(max(1, (1 << 18) // nx))
    if not correct:
        # 3. the raster's transform, cut to the picture: of the range transform, the last ny / 2 outputs are the
        # pixels before the center and the first ny / 2 those from it on; likewise the azimuth transform's
        F = sf.ifft(B, n=nfy, axis=0, workers=W)
        del B
        if complex_image:
            y, x = (np.arange(ny) - ny / 2.0) * g['spy'], (np.arange(nx) - nx / 2.0) * g['spx']
            cy, cx = np.exp(1j * g['kr_min'] * y).astype(np.complex64), np.exp(-1j * g['ka_lo'] * x).astype(np.complex64)
        for r_out, r_in in ((0, nfy - ny // 2), (ny // 2, 0)):
            T = sf.fft(F[r_in:r_in + ny // 2], n=nfx, axis=1, workers=W)

            def put(sl, T=T, r_out=r_out):
                o = out[r_out + sl.start:r_out + sl.stop]
                for c_out, c_in in ((0, nfx - nx // 2), (nx // 2, 0)):
                    if complex_image:
                        np.multiply(T[sl, c_in:c_in + nx // 2], cx[c_out:c_out + nx // 2], out=o[:, c_out:c_out + nx // 2])
                    else:
                        np.abs(T[sl, c_in:c_in + nx // 2], out=o[:, c_out:c_out + nx // 2])
                if complex_image:
                    o *= cy[r_out + sl.start:r_out + sl.stop, None]
            _blocks(put, ny // 2, cb)
            del T
        timing['image'] = timing.get('image', 0.0) + time.perf_counter() - t0
        return out
    # 4. the same on a raster padded by ov, at baseband, in blocks of range pixels read at the apparent positions.
    # The spectrum is set about index 0 of the padded transforms with alternating signs, so that their outputs
    # count from the image's center
    dist = g['dist']
    Ny2, Nx2 = ov * nfy, ov * nfx
    hy, hx = nkr // 2, nka // 2
    sy = ((-1.0) ** (np.arange(nkr) - hy)).astype(np.float32)
    sx = ((-1.0) ** (np.arange(nka) - hx)).astype(np.float32)
    Z = np.empty((Ny2, nka), np.complex64)
    rb = int(max(1, (1 << 18) // nka))

    def fill(sl):
        if sl.stop <= nkr - hy:
            np.multiply(B[hy + sl.start:hy + sl.stop], sy[hy + sl.start:hy + sl.stop, None], out=Z[sl])
        elif sl.start >= Ny2 - hy:
            a_ = sl.start - (Ny2 - hy)
            np.multiply(B[a_:a_ + sl.stop - sl.start], sy[a_:a_ + sl.stop - sl.start, None], out=Z[sl])
        else:
            Z[sl] = 0
    edges = sorted({0, nkr - hy, Ny2 - hy, Ny2})
    for a_, b_ in zip(edges, edges[1:]):
        _blocks(lambda sl: fill(slice(a_ + sl.start, a_ + sl.stop)), b_ - a_, rb)
    del B
    F1 = sf.ifft(Z, axis=0, workers=W, overwrite_x=True)                   # [Ny2, nka]: range done
    del Z
    timing['range transform'] = timing.get('range transform', 0.0) + time.perf_counter() - t0
    ltab = _polyphase('lanczos', LANCZOS, PHASES)
    ltab16 = np.zeros((PHASES + 1, 16), np.float32)                         # each weight for a sample's two parts
    ltab16[:, 0:2 * LANCZOS:2] = ltab16[:, 1:2 * LANCZOS:2] = ltab
    cx, cy = np.ascontiguousarray(dist['cx'], np.float64), np.ascontiguousarray(dist['cy'], np.float64)
    pw = np.ascontiguousarray(np.asarray(dist['powers'], np.int32).ravel())
    deg = max(a + b for a, b in dist['powers'])
    kr_c, ka_c = g['kr_min'] + hy * g['dkr'], g['ka_lo'] + hx * g['dka']
    msy = int(math.ceil(g['shift'][1] / g['spy'])) * ov + LANCZOS + 2
    bw = int(max(16, min(ny, (128 << 20) // (8 * Nx2 * ov))))               # range pixels per block
    rb = int(max(1, (1 << 18) // Nx2))
    tt = [0.0, 0.0]
    for jb in range(0, ny, bw):
        n = min(bw, ny - jb)
        rb0 = max(0, Ny2 // 2 + ov * (jb - ny // 2) - msy)
        span = min(Ny2, Ny2 // 2 + ov * (jb + n - ny // 2) + msy) - rb0
        t1 = time.perf_counter()
        Z = np.empty((span, Nx2), np.complex64)

        def place(sl, Z=Z, rb0=rb0):
            src = F1[rb0 + sl.start:rb0 + sl.stop]
            np.multiply(src[:, hx:], sx[hx:], out=Z[sl, :nka - hx])
            Z[sl, nka - hx:Nx2 - hx] = 0
            np.multiply(src[:, :hx], sx[:hx], out=Z[sl, Nx2 - hx:])
        _blocks(place, span, rb)
        base = np.ascontiguousarray(sf.fft(Z, axis=1, workers=W, overwrite_x=True))
        del Z
        t2 = time.perf_counter()
        lib.ql_gather(ptr(base), span, Nx2, Nx2, Ny2, rb0, 0, nx, n, 0, jb, nx, ny, float(g['spx']),
                      float(g['spy']), float(ov), float(dist['sx']), float(dist['sy']), ptr(cx), ptr(cy), ptr(pw),
                      len(cx), deg, float(kr_c), float(ka_c), 1, 1.0, LANCZOS, ptr(ltab), ptr(ltab16), PHASES,
                      0 if complex_image else 1, ctypes.c_void_p(out.ctypes.data + jb * out.strides[0]), nx)
        del base
        tt[0] += t2 - t1
        tt[1] += time.perf_counter() - t2
    timing['azimuth transform'] = timing.get('azimuth transform', 0.0) + tt[0]
    timing['correction'] = timing.get('correction', 0.0) + tt[1]
    return out


def _wrap_shift(x, n, axis):
    """The n outputs of a transform along an axis that are the pixels -n / 2 .. n / 2 - 1 about the center (the last
    n / 2 outputs, then the first n / 2), as a new array."""
    N = x.shape[axis]
    xp = np if isinstance(x, np.ndarray) else __import__('cupy')
    sl = lambda a, b: tuple(slice(a, b) if k == axis else slice(None) for k in range(x.ndim))
    out = xp.empty(tuple(n if k == axis else s for k, s in enumerate(x.shape)), x.dtype)
    out[sl(0, n // 2)] = x[sl(N - n // 2, N)]
    out[sl(n // 2, n)] = x[sl(0, n - n // 2)]
    return out


def _abs(img, scale):
    """The amplitude of a complex host image times a scale (float32), in row blocks on the pool."""
    out = np.empty(img.shape, np.float32)

    def rows(sl):
        np.abs(img[sl], out=out[sl])
        out[sl] *= scale
    _blocks(rows, img.shape[0], int(max(1, (1 << 20) // max(img.shape[1], 1))))
    return out


def _pfa(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, *, correct=True, window=True, backend='cpu', complex_image=True,
         timing=None, info=None):
    """The quick look's polar format of a history S [P, K] referenced to |ant| on the grid of form_image (e2 along
    the mid-aperture line of sight, away from the radar) -> the image [nx, ny]: complex64 as backprojection gives
    it, or its amplitude (float32, complex_image False); a NumPy array, or a CuPy array for backend 'cuda'.
    window: the Taylor window (nbar 4, 35 dB) along frequency and along pulses. info: a dict that receives the
    geometry's facts (the displacement, the raster and gate sizes)."""
    timing = {} if timing is None else timing
    t0 = time.perf_counter()
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    ant = np.asarray(ant, np.float64)
    if nx % 2 or ny % 2:
        raise ValueError(f'the quick look takes even pixel counts, got {nx} by {ny}')
    g = _geometry(ant, float(fmin), float(df), S.shape[1], int(nx), int(ny), float(spx), float(spy), e1, e2, correct)
    timing['geometry'] = timing.get('geometry', 0.0) + time.perf_counter() - t0
    if info is not None:
        info.update(shift_px=g['dist']['max_shift_px'], fit_rms_px=g['dist']['rms_px'], raster=(g['nkr'], g['nka']),
                    fft=(g['nfy'], g['nfx']), range_gate=(g['Kf'], g['Nr']), azimuth_gate=(g['Pf'], g['Na']))
    if backend == 'cuda':
        img = _pfa_cuda(S, g, window, correct, complex_image, timing)
    else:
        img = _pfa_cpu(np.ascontiguousarray(S, np.complex64), g, window, correct, complex_image, timing)
    return img.T


# ------------------------------------------------------------------------------------------------------- CUDA

_CUDA_COLS_SRC = r"""
#define HALF_T (TAPS / 2)
__device__ __forceinline__ float tabled(float d, const float* tab) {
  const float ad = fabsf(d);
  if (ad >= HALF_T) return 0.f;
  const float u = ad * TABN;
  const int k = (int)u;
  return fmaf(tab[k + 1] - tab[k], u - k, tab[k]);
}
// out[m, j] = column m of A [R, M] (complex) at the row position q(t), t = (ka0 + j dka) inv_kr[m]: q by linear
// interpolation in qtab [nt] (quicklook_cpu.cpp's ql_resample_cols); threads along m read consecutive addresses
extern "C" __global__ void ql_resample_cols(const float2* A, int R, int M, const double* inv_kr, double ka0, double dka,
                                            const double* qtab, int nt, double t0, double inv_dt, int n_out,
                                            const float* tab, float2* out) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (m >= M || j >= n_out) return;
  float2 acc = make_float2(0.f, 0.f);
  const double u = ((ka0 + j * dka) * inv_kr[m] - t0) * inv_dt;
  if (u >= 0.0 && u < nt - 1) {
    const int k = (int)u;
    const double pos = qtab[k] + (u - k) * (qtab[k + 1] - qtab[k]);
    if (pos > -0.5 && pos < R - 0.5) {
      const double fl = floor(pos);
      const int i0 = (int)fl - HALF_T + 1;
      const float fr = (float)(pos - fl);
      float w[TAPS], tot = 0.f;
      #pragma unroll
      for (int t = 0; t < TAPS; ++t) {
        const int i = i0 + t;
        w[t] = (i >= 0 && i < R) ? tabled(fr - (float)(t - HALF_T + 1), tab) : 0.f;
        tot += w[t];
      }
      const float inv = 1.f / fmaxf(tot, 1e-12f);
      #pragma unroll
      for (int t = 0; t < TAPS; ++t) {
        const int i = min(max(i0 + t, 0), R - 1);
        const float2 v = A[(long)i * M + m];
        acc.x = fmaf(w[t] * inv, v.x, acc.x); acc.y = fmaf(w[t] * inv, v.y, acc.y);
      }
    }
  }
  out[(long)m * n_out + j] = acc;
}
"""
_CUDA = {}


def _cuda_cols():
    """The CUDA kernel of the azimuth resampling, compiled once."""
    if 'cols' not in _CUDA:
        import cupy as cp
        from .pfa2 import TABN
        _CUDA['cols'] = cp.RawModule(code=_CUDA_COLS_SRC, options=(f'-DTAPS={TAPS}', f'-DTABN={TABN}')).get_function(
            'ql_resample_cols')
    return _CUDA['cols']


def _pfa_cuda(S, g, window, correct, complex_image, timing):
    """_pfa_cpu's stages in CuPy, with the CUDA kernels of polar format (pfa2.py: the range resampling and the final
    gather) and the azimuth kernel above. S: the host history (uploaded in blocks of pulses, each windowed and gated
    on the device) or a CuPy array. -> a CuPy image, range-major [ny, nx]."""
    import cupy as cp
    import cupyx.scipy.fft as cf
    from . import pfa2, _xp
    P, K, Kf, Nr, k0 = g['P'], g['K'], g['Kf'], g['Nr'], g['k0']
    nkr, nka, nfy, nfx, nx, ny = g['nkr'], g['nka'], g['nfy'], g['nfx'], g['nx'], g['ny']
    i32, f64 = np.int32, np.float64

    def lap(name, t0):
        cp.cuda.get_current_stream().synchronize()
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()
    t0 = time.perf_counter()
    k_rows = pfa2._resample_kernels(TAPS)[0]
    k_cols = _cuda_cols()
    tab = cp.asarray(pfa2.kernel_table(TAPS, KAISER))
    wk = cp.asarray((g['wk'] * (_taylor(K) if window else 1.0)).astype(np.float32))
    wp = cp.asarray((g['wp'] * (_taylor(P) if window else 1.0)).astype(np.float32))
    tap_r = cp.asarray(_taper(Nr, g['Er'], g['gy'], g['bin_r']))
    alpha, beta = cp.asarray(g['alpha'].astype(np.float32)), cp.asarray(g['beta'].astype(np.float32))
    t0 = lap('kernels', t0)
    # 1. range gate and range resampling, a block of pulses at a time (three arrays of the padded block at once)
    A = cp.empty((P, nkr), cp.complex64)
    nb = int(max(1, min(P, 1024, _xp.cuda_block() // (24 * max(Kf, Nr)))))
    for r in range(0, P, nb):
        x = cp.asarray(S[r:r + nb])
        n = x.shape[0]
        buf = cp.zeros((n, Kf), cp.complex64)
        cp.multiply(x, wk, out=buf[:, k0:k0 + K])
        del x
        gate = _wrap_cut(cf.ifft(buf, axis=1, overwrite_x=True), Nr, 1)
        del buf
        gate *= tap_r
        gate *= wp[r:r + n, None]
        G = cp.ascontiguousarray(cf.fft(gate, axis=1, overwrite_x=True))
        del gate
        k_rows((-(-nkr // 128), n), (128,), (G, i32(n), i32(Nr), alpha[r:r + n], beta[r:r + n], i32(nkr), tab, A[r:r + n]))
        del G
    t0 = lap('range gate', t0)
    # 2. azimuth gate and azimuth resampling
    Pf, Na, p0 = g['Pf'], g['Na'], g['p0']
    pad = cp.zeros((Pf, nkr), cp.complex64)
    pad[p0:p0 + P] = A
    del A
    gate = _wrap_cut(cf.fft(pad, axis=0, overwrite_x=True), Na, 0)
    del pad
    gate *= cp.asarray((_taper(Na, g['Ex'], g['gx'], g['bin_a']) * (g['gain'] * (g['ov'] * nfy if correct else nfy) * Na / Pf)).astype(np.float32))[:, None]
    A2 = cp.ascontiguousarray(cf.ifft(gate, axis=0, overwrite_x=True))
    del gate
    t0 = lap('azimuth gate', t0)
    B = cp.empty((nkr, nka), cp.complex64)
    qtab = cp.asarray(g['qtab'])
    k_cols((-(-nkr // 128), nka), (128,), (A2, i32(Na), i32(nkr), cp.asarray(1.0 / g['kr']), f64(g['ka_lo']), f64(g['dka']),
                                          qtab, i32(len(g['qtab'])), f64(g['tg0']), f64(g['tg_inv']), i32(nka), tab, B))
    del A2
    t0 = lap('azimuth resampling', t0)
    if not correct:
        F = _wrap_shift(cf.ifft(B, n=nfy, axis=0), ny, 0)
        del B
        img = _wrap_shift(cf.fft(F, n=nfx, axis=1), nx, 1)
        del F
        if complex_image:
            y, x = (np.arange(ny) - ny / 2.0) * g['spy'], (np.arange(nx) - nx / 2.0) * g['spx']
            img *= cp.asarray(np.exp(1j * g['kr_min'] * y).astype(np.complex64))[:, None]
            img *= cp.asarray(np.exp(-1j * g['ka_lo'] * x).astype(np.complex64))[None, :]
        else:
            img = cp.abs(img)
        lap('image', t0)
        return img
    ov, dist = g['ov'], g['dist']
    Ny2, Nx2 = ov * nfy, ov * nfx
    hy, hx = nkr // 2, nka // 2
    B *= cp.asarray(((-1.0) ** (np.arange(nkr) - hy)).astype(np.float32))[:, None]
    B *= cp.asarray(((-1.0) ** (np.arange(nka) - hx)).astype(np.float32))[None, :]
    Z = cp.zeros((Ny2, nka), cp.complex64)
    Z[:nkr - hy], Z[Ny2 - hy:] = B[hy:], B[:hy]
    del B
    F1 = cf.ifft(Z, axis=0, overwrite_x=True)
    del Z
    t0 = lap('range transform', t0)
    powers = dist['powers']
    kern = pfa2._gather_kernel(LANCZOS, len(powers), max(a + b for a, b in powers))
    d_cx, d_cy = cp.asarray(np.asarray(dist['cx'], np.float64)), cp.asarray(np.asarray(dist['cy'], np.float64))
    d_pw = cp.asarray(np.asarray(powers, np.int32).ravel())
    kr_c, ka_c = g['kr_min'] + hy * g['dkr'], g['ka_lo'] + hx * g['dka']
    gain = cp.ones((1,), cp.float32)
    msy = int(math.ceil(g['shift'][1] / g['spy'])) * ov + LANCZOS + 2
    out = cp.empty((nx, ny), cp.complex64)                           # the kernel writes azimuth-major
    bw = int(max(16, min(ny, _xp.cuda_block() // (16 * Nx2 * ov))))
    for jb in range(0, ny, bw):
        n = min(bw, ny - jb)
        rb0 = max(0, Ny2 // 2 + ov * (jb - ny // 2) - msy)
        span = min(Ny2, Ny2 // 2 + ov * (jb + n - ny // 2) + msy) - rb0
        Z = cp.zeros((span, Nx2), cp.complex64)
        Z[:, :nka - hx], Z[:, Nx2 - hx:] = F1[rb0:rb0 + span, hx:], F1[rb0:rb0 + span, :hx]
        base = cp.ascontiguousarray(cf.fft(Z, axis=1, overwrite_x=True))
        del Z
        kern((-(-nx // 128), n), (128,),
             (base, i32(span), i32(Nx2), i32(Ny2), i32(rb0), i32(nx), i32(n), i32(0), i32(jb), i32(nx), i32(ny),
              f64(g['spx']), f64(g['spy']), f64(0.0), f64(0.0), f64(ov), f64(dist['sx']), f64(dist['sy']), d_cx, d_cy, d_pw,
              f64(kr_c), f64(ka_c), i32(1), i32(-1), np.float32(1.0), gain, out[:, jb:], np.int64(ny)))
        cp.cuda.get_current_stream().synchronize()
        del base
    del F1
    img = cp.ascontiguousarray(out.T) if complex_image else cp.ascontiguousarray(cp.abs(out).T)
    lap('correction', t0)
    return img


# ------------------------------------------------------------------------------------------- factorized former

def _ffbp(col, backend, window, timing):
    """The cut history formed on the quick look's grid by the factorized former (ImageFormer) on a JAX device or
    a TPU -> complex64 [nx, ny] on the host."""
    from .api import ImageFormer
    S = col['S']
    P, K = S.shape
    t0 = time.perf_counter()
    if window:
        wp, wk = _taylor(P).astype(np.float32), _taylor(K).astype(np.float32)

        def rows(sl):
            S[sl] *= wp[sl, None] * wk[None, :]
        _blocks(rows, P, int(max(1, (1 << 18) // K)))
    former = ImageFormer(col['ant'], col['fmin'], col['df'], K, col['nx'], col['ny'], col['spx'], col['spy'], col['e1'],
                         col['e2'], backend=backend, precision='single-pass' if backend == 'tpu' else 'float32',
                         window=False, range_gate=True)
    timing['plan and compile'] = timing.get('plan and compile', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    img = former(S)
    timing['formation'] = timing.get('formation', 0.0) + time.perf_counter() - t0
    return img, former


# ------------------------------------------------------------------------------------------------------- PNG

def write_png(path, image, level=1):
    """Write a uint8 image [rows, cols] (gray), [rows, cols, 3] (RGB) or [rows, cols, 4] (RGBA) as a PNG file, with the
    standard library alone. The rows are deflated in blocks on all cores, each block an independent run of the one
    zlib stream, so a 16 megapixel picture is written in a few hundredths of a second. level: zlib's, 0 to 9
    (speckle hardly compresses, and 1 is as small as 9 on it)."""
    import struct
    import zlib
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)) or a.size == 0:
        raise ValueError(f'write_png takes a uint8 image [rows, cols] or [rows, cols, 3 or 4], got {a.dtype} {a.shape}')
    rows, cols = a.shape[:2]
    raw = np.zeros((rows, 1 + cols * (1 if a.ndim == 2 else a.shape[2])), np.uint8)       # filter type 0 on every row
    raw[:, 1:] = a.reshape(rows, -1)
    raw = raw.reshape(-1)
    step = int(min(4 << 20, max(1 << 16, -(-raw.size // (4 * _workers())))))
    cuts = list(range(0, raw.size, step))
    parts = [None] * len(cuts)

    def deflate(i):
        co = zlib.compressobj(level, zlib.DEFLATED, -15)
        last = i == len(cuts) - 1
        parts[i] = co.compress(raw[cuts[i]:cuts[i] + step]) + co.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)
    _blocks(lambda sl: deflate(sl.start), len(cuts), 1)
    data = b'\x78\x01' + b''.join(parts) + struct.pack('>I', zlib.adler32(raw))

    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    color = {2: 0, 3: 2, 4: 6}[2 if a.ndim == 2 else a.shape[2]]
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', cols, rows, 8, color, 0, 0, 0)))
        fh.write(chunk(b'IDAT', data))
        fh.write(chunk(b'IEND', b''))


def read_png(path):
    """Read an 8-bit PNG file that is not interlaced (gray, RGB or RGBA: what write_png writes, and what most
    writers make of such images) -> uint8 [rows, cols] or [rows, cols, 3 or 4]."""
    import struct
    import zlib
    with open(path, 'rb') as fh:
        b = fh.read()
    if b[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'{path}: not a PNG file')
    pos, idat, head = 8, [], None
    while pos + 8 <= len(b):
        n, kind = struct.unpack('>I', b[pos:pos + 4])[0], b[pos + 4:pos + 8]
        body = b[pos + 8:pos + 8 + n]
        if kind == b'IHDR':
            head = struct.unpack('>IIBBBBB', body)
        elif kind == b'IDAT':
            idat.append(body)
        pos += 12 + n
    if head is None or not idat:
        raise ValueError(f'{path}: a PNG file without a header or without image data')
    cols, rows, depth, color, _, _, interlace = head
    if depth != 8 or color not in (0, 2, 6) or interlace:
        raise ValueError(f'{path}: read_png reads 8-bit gray, RGB and RGBA images that are not interlaced '
                         f'(bit depth {depth}, color type {color}, interlace {interlace})')
    ch = {0: 1, 2: 3, 6: 4}[color]
    raw = np.frombuffer(zlib.decompress(b''.join(idat)), np.uint8).reshape(rows, 1 + cols * ch)
    out = raw[:, 1:].astype(np.int32)
    if raw[:, 0].any():                                # the row filters: sub, up, average, Paeth
        prev = np.zeros(cols * ch, np.int32)
        for r in range(rows):
            f, line = int(raw[r, 0]), out[r]
            if f == 1:
                for k in range(ch, len(line)):
                    line[k] = (line[k] + line[k - ch]) & 255
            elif f == 2:
                line[:] = (line + prev) & 255
            elif f in (3, 4):
                for k in range(len(line)):
                    a, up, ul = (int(line[k - ch]), int(prev[k]), int(prev[k - ch])) if k >= ch else (0, int(prev[k]), 0)
                    if f == 3:
                        pred = (a + up) // 2
                    else:
                        q = a + up - ul
                        pred = min((abs(q - a), 0, a), (abs(q - up), 1, up), (abs(q - ul), 2, ul))[2]
                    line[k] = (line[k] + pred) & 255
            elif f:
                raise ValueError(f'{path}: row filter {f}')
            prev = line
    out = out.astype(np.uint8)
    return out.reshape(rows, cols) if ch == 1 else out.reshape(rows, cols, ch)


# ------------------------------------------------------------------------------------------------- the quick look

def _backend(backend):
    """backend checked against this machine, 'auto' resolved (tpu, else cuda, else cpu). JAX is asked for a TPU only
    where one may be present (JAX already imported, the libtpu package installed, or TPU_LIBRARY_PATH set): importing
    it takes longer than a quick look."""
    import importlib.util
    import sys
    from . import api
    if backend != 'auto' or 'jax' in sys.modules or os.environ.get('TPU_LIBRARY_PATH') \
            or importlib.util.find_spec('libtpu') is not None:
        b = api._backend(backend)
        return 'cpu' if b == 'metal' else b
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


_KINDS = {'.png': 'png', '.nitf': 'sicd', '.ntf': 'sicd'}


def quicklook(cphd, output=None, *, size=2048, resolution=None, correct=True, remap='nrl', backend='auto',
              weighting='taylor', looks=1, complex_image=False, channel=0, extent=None, reader_options=None):
    """A reduced-resolution picture of a spotlight CPHD collection, by polar format on the part of the file that
    resolution needs (see the module docstring).

    cphd: path (CPHD 1.0.1, 1.1.0 or 0.3, frequency domain). output: a file to write, by its extension: .png, the
        8-bit picture (standard library only); .nitf or .ntf, the complex image as a slant-plane SICD
        (products.write_sicd). None: nothing is written.
    size: pixels along the longer side of the picture, which is the tasked area's bounding rectangle on the
        slant-plane grid (default 2048). resolution: the ground resolution in meters in place of size (the
        reciprocal of the band used on both axes; the pixel is 0.96 of it), or 'full' for the finest the collection
        holds on both axes. A request finer than that gets that, with a note.
    correct: remove polar format's displacement of scatterers (the default). False is the fastest picture: the
        displacement, a smooth field that grows with the square of the distance from the center, stays, and
        displacement_px says how large it is at the corners.
    remap: the display remap, a name of fastsar.remap.REMAPS other than 'linear' or a function (sidd.to_display's).
    backend: 'auto', 'cpu', 'cuda', 'jax' or 'tpu'. cpu and cuda form by polar format; jax and tpu form the same
        cut history on the same grid by the factorized former (ImageFormer), where correct does not apply.
    weighting: 'taylor' (nbar 4, 35 dB, along frequency and along pulses: the default) or 'uniform'.
    looks: n, or (sub-apertures, sub-bands): the picture is the mean power of that many images of the same pixels
        and resolution, formed from runs of pulses and of samples side by side (n by n for one number), for less
        speckle: the standard deviation of the power falls as the square root of their count. As many as the
        collection holds at the resolution; they cost their count in bytes read and in formation time. The complex
        image is of one look.
    complex_image: also return the complex image [nx, ny] (complex64, with backprojection's phase when corrected).
    channel: index, identifier or polarization. extent: (across, along) the line of sight in meters on the ground,
        about the scene's reference point, in place of the file's image area. reader_options: troposphere,
        phase_sign, ref_freq, drop_flagged or regrid_tol for io.read_cphd.

    -> dict(display: uint8 [rows, cols], laid out shadows down (range away from the radar down the rows, seen from
    above); image: the complex image or None; origin, e1, e2, spx, spy: the grid of image [nx, ny], pixel (i, j) at
    origin + i spx e1 + j spy e2 in the local frame of meta (as form_cphd's products); plane 'slant', axes 'los',
    mode 'spotlight', center; display_origin, row, col, row_ss, col_ss, transform: the display's pixel (r, c) at
    display_origin + r row_ss row + c col_ss col (sidd.shadows_down's entries); corners: latitude and longitude
    [4, 2] of the display's corner pixels on the scene's plane, from the first row and column clockwise; pixel, the
    pixel on the ground (m), resolution (m) and finest, the finest resolution the collection holds; band (first
    and last frequency used, Hz), bandwidth (cycles/m along e1 and e2), window, graze and twist (rad: the grazing
    angle, and the angle by which the slant plane shears the ground in the picture), pulses (of those read) and cut:
    dict(pulses, of_pulses, samples, of_samples: the file's vectors and samples read; signal_bytes, of_bytes: their
    size and the signal array's); algorithm ('pfa' or 'ffbp'), backend, correct, displacement_px (polar format's
    largest displacement along e1 and e2 over the picture, in pixels; removed when correct), looks (sub-apertures,
    sub-bands), remap, meta
    (read_cphd's), notes, timing (seconds by stage, and total), and path when a file was written). A moving-beam
    collection raises ValueError; process_cphd forms those."""
    from . import _xp, planes, sidd
    from . import remap as _remap
    t_start = time.perf_counter()
    if resolution is None:
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 16:
            raise ValueError(f'size must be a whole number of pixels, 16 or more, got {size!r}')
    elif resolution != 'full' and not (isinstance(resolution, (int, float, np.integer, np.floating))
                                       and not isinstance(resolution, bool) and np.isfinite(resolution) and resolution > 0):
        raise ValueError(f"resolution must be a positive length in meters or 'full', got {resolution!r}")
    if weighting not in ('taylor', 'uniform'):
        raise ValueError(f"weighting must be 'taylor' or 'uniform', got {weighting!r}")
    if not callable(remap) and (remap not in _remap.REMAPS or remap == 'linear'):
        raise ValueError(f"remap {remap!r}: one of {sorted(k for k in _remap.REMAPS if k != 'linear')}, or a function")
    if extent is not None and (np.size(extent) != 2 or not np.all(np.isfinite(extent)) or np.min(extent) <= 0):
        raise ValueError(f'extent must be (across, along) the line of sight, two positive lengths in meters, got {extent!r}')
    kind = None
    if output is not None:
        kind = _KINDS.get(os.path.splitext(os.fspath(output))[1].lower())
        if kind is None:
            raise ValueError(f'output {os.fspath(output)!r}: a .png file (the picture) or a .nitf or .ntf file (the '
                             'complex image as a SICD)')
    try:
        looks = tuple(int(v) for v in np.broadcast_to(np.asarray(looks), (2,)))
        if min(looks) < 1 or not np.all(np.asarray(looks) == np.broadcast_to(np.asarray(looks), (2,))):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f'looks must be a count, or (sub-apertures, sub-bands), 1 or more, got {looks!r}') from None
    want = bool(complex_image) or kind == 'sicd'
    if want and looks != (1, 1):
        raise ValueError('the complex image (complex_image=True, a .nitf output) is of one look: leave looks=1')
    backend = _backend(backend)
    window = weighting == 'taylor'
    col = _collection(cphd, size, resolution, extent, channel, reader_options, looks)
    timing, notes, plan, meta = col['timing'], col['notes'], col['plan'], col['meta']
    nx, ny, spx, spy, e1, e2, c = col['nx'], col['ny'], col['spx'], col['spy'], col['e1'], col['e2'], col['center']
    S, ant, f0, df = col['S'], col['ant'], col['fmin'], col['df']
    P, K = S.shape
    (pa, pb), (k0, k1) = meta['pulses'], meta['samples']
    isz = meta['signal_bytes'] // (P * K) if P * K else 0
    cut = dict(pulses=(pa, pb), of_pulses=plan['P'], samples=(k0, k1), of_samples=plan['K'],
               signal_bytes=int(meta['signal_bytes']), of_bytes=int(plan['P'] * plan['K'] * isz))
    notes.append(f"quick look: a ground resolution of {plan['resolution']:.3f} m in pixels of {plan['pixel']:.3f} m, from "
                 f"{pb - pa} of {plan['P']} pulses and {k1 - k0} of {plan['K']} samples "
                 f"({cut['signal_bytes'] / 1e6:.1f} MB of the {cut['of_bytes'] / 1e9:.2f} GB signal array)")
    notes.append(f'slant-plane grid along the mid-aperture line of sight, {nx} by {ny} pixels of {spx:.3f} by {spy:.3f} m '
                 f"(0.96 of the reciprocal of the band on each axis), which are {plan['pixel']:.3f} m across and along "
                 f"the ground range direction (grazing angle {plan['graze']:.1f} degrees). The picture is not projected "
                 f"to the ground: it is the ground sheared by the slant plane's twist, {plan['twist']:.1f} degrees")
    la, lr = plan['looks']
    Pl, Kl = P // la, K // lr                                            # pulses and samples of a look

    def look(i, j):
        """The collection of look (i, j): its pulses and samples, on their own."""
        if (la, lr) == (1, 1):
            return col
        rows, cols = slice(i * Pl, (i + 1) * Pl), slice(j * Kl, (j + 1) * Kl)
        return dict(col, S=np.ascontiguousarray(S[rows, cols]), ant=ant[rows], fmin=f0 + j * Kl * df)
    info, shift, acc, cimg = {}, None, None, None
    xp = np
    for i in range(la):
        for j in range(lr):
            sub = look(i, j)
            if backend in ('jax', 'tpu'):
                img = _ffbp(sub, backend, window, timing)[0]
                cimg = img if want else None
                rm = np.ascontiguousarray(img.T)
                complex_rm = True
            else:
                out = _pfa(sub['S'], sub['ant'], sub['fmin'], df, nx, ny, spx, spy, e1, e2, correct=bool(correct),
                           window=window, backend=backend, complex_image=want, timing=timing, info=info)
                rm = out.T                                               # range-major, where it was formed
                xp = _xp.module(rm)
                complex_rm = want
                if want:
                    cimg = np.ascontiguousarray(_xp.host(rm).T)
            t0 = time.perf_counter()
            if complex_rm:
                rm = _abs(rm, np.float32(1.0)) if _xp.kind(rm) == 'numpy' else xp.abs(rm)
            if (la, lr) == (1, 1):
                amp = rm
            else:                                                        # the looks' mean power
                rm *= rm
                acc = rm if acc is None else acc + rm
            del sub, rm
            timing['detect'] = timing.get('detect', 0.0) + time.perf_counter() - t0
    if acc is not None:
        acc *= np.float32(1.0 / (la * lr))
        amp = xp.sqrt(acc)
        del acc
        notes.append(f'{la} by {lr} looks (sub-apertures of {Pl} pulses by sub-bands of {Kl} samples), their power '
                     'averaged')
    if backend in ('jax', 'tpu'):
        algorithm, corrected = 'ffbp', True
        notes.append(f"formed by factorized backprojection on the '{backend}' backend"
                     + (" (precision 'single-pass')" if backend == 'tpu' else '') + ': polar format\'s resampling '
                     'steps are gathers, which are slow on this device; correct does not apply')
    else:
        algorithm, corrected, shift = 'pfa', bool(correct), tuple(float(v) for v in info['shift_px'])
        notes.append(f'polar format: its displacement of scatterers, up to {shift[0]:.1f} by {shift[1]:.1f} pixels '
                     '(azimuth by range) over the picture, ' + ('removed' if corrected else 'left in (correct=False)'))
    del S, col
    # the picture: remapped where the image is, laid out shadows down
    t0 = time.perf_counter()
    origin = c - 0.5 * nx * spx * e1 - 0.5 * ny * spy * e2
    grid = dict(image=np.broadcast_to(np.float32(0), (nx, ny)), origin=origin, e1=e1, e2=e2, spx=spx, spy=spy, meta=meta)
    lay = sidd.shadows_down(grid)
    tr = lay['transform']
    d = sidd._display(dict(image=amp), remap, None, {})[0]             # uint8, range-major
    d = d if tr['transpose'] else d.T
    d = d[::-1] if tr['flip_rows'] else d
    d = d[:, ::-1] if tr['flip_cols'] else d
    display = np.ascontiguousarray(_xp.host(d))
    del amp, d
    rows, cols = display.shape
    rc = np.array([[0, 0], [0, cols - 1], [rows - 1, cols - 1], [rows - 1, 0]], np.float64)
    q = lay['origin'] + rc[:, :1] * lay['row_ss'] * lay['row'] + rc[:, 1:] * lay['col_ss'] * lay['col']
    fr = planes.slant_frame(meta, c)
    ground = planes.slant_to_ground(fr, *planes.ground_to_slant(fr, q), c[2])
    lat, lon, _ = io.ecf_to_geodetic(io.local_to_ecf(ground, meta))
    timing['display'] = time.perf_counter() - t0
    u = ant / np.linalg.norm(ant, axis=1, keepdims=True)
    tt = (u @ e1) / -(u @ e2)
    fc = f0 + 0.5 * (K - 1) * df
    product = dict(
        display=display, image=cimg, origin=origin, e1=e1, e2=e2, spx=float(spx), spy=float(spy), plane='slant',
        axes='los', mode='spotlight', center=c, display_origin=lay['origin'], row=lay['row'], col=lay['col'],
        row_ss=lay['row_ss'], col_ss=lay['col_ss'], transform=tr, corners=np.stack([lat, lon], 1),
        pixel=float(plan['pixel']), resolution=float(plan['resolution']), finest=float(plan['finest']),
        band=(f0, f0 + (K - 1) * df), bandwidth=(2 * fc / C * float(abs(tt[-1] - tt[0])), 2 * K * df / C),
        window=('taylor', 'taylor') if window else None, pulses=(0, P), cut=cut, algorithm=algorithm, backend=backend,
        correct=corrected, displacement_px=shift, looks=(la, lr), remap=remap, graze=float(np.radians(plan['graze'])),
        twist=float(np.radians(plan['twist'])), meta=meta,
        notes=notes, timing=timing)
    if kind is not None:
        t0 = time.perf_counter()
        if kind == 'png':
            write_png(output, display)
        else:
            from .products import write_sicd
            product['metadata'] = write_sicd(output, product)
            if not complex_image:
                product['image'] = None
        product['path'] = output
        timing['write'] = time.perf_counter() - t0
    timing['total'] = time.perf_counter() - t_start
    return product
