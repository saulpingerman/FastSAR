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
format in stages whose cost follows the picture, not the collection:

  1. Range. Each pulse is weighted, transformed to a range profile, cut to the range bins the picture spans (a
     raised-cosine edge beyond them) and transformed back, to a spectrum sampled 1.5 times finer than the gate
     needs; a 16-tap Kaiser-windowed sinc then reads it at the range wavenumbers of the rectangular raster.
  2. Azimuth. Each range wavenumber's pulse sequence is transformed, cut to the picture's azimuth extent (a gate
     set in meters for that wavenumber's own bins), and transformed back at the pulse spacing that extent needs;
     the same kernel reads it at the raster's azimuth wavenumbers, at the pulse positions of the collection's own,
     uneven, spacing in the tangent of the look angle. The pulses are not resampled separately.
  3. A two-dimensional FFT of the raster. Uncorrected (correct=False) this is the picture: polar format's planar
     wavefronts leave each scatterer displaced by a smooth field that grows with the square of its distance from
     the center (the product's notes give its size in pixels).
  4. Corrected (the default): the transform is taken on a raster padded to twice the sampling and read at each
     pixel's apparent position (pfa2.distortion's polynomial) with a 6-tap Lanczos kernel.

What the correction does not remove is polar format's defocus far from the center at fine resolution: the complex
image is within -40 dB of exact backprojection of the same history at the sizes a quick look is for, and at full
resolution on a scene of several kilometers it falls to -13 to -23 dB at the corners, as form_image(algorithm='pfa')
does. _corner_error estimates that from the geometry; the product carries it and a note says so past -30 dB.

Devices. 'cpu': scipy's FFTs on all cores, every one along the rows of its array (the arrays are turned between
the stages), and the C++ kernels of quicklook_cpu.cpp, which read their weights from polyphase tables. 'cuda': the
same stages in CuPy with the CUDA kernels of polar format (pfa2.py: the range resampling and the final gather) and
one more for the uneven pulse spacing; the history is windowed and gated on the device in blocks of pulses. 'jax'
and 'tpu': polar format's resampling steps are gathers, slow on a TPU's vector unit, so the same cut history is
formed on the same grid by the factorized former (ImageFormer; precision 'single-pass' on a TPU), and a note says
so. 'auto': the device that forms the cut soonest, by its size (the AUTO_ thresholds below).

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
# backend='auto' takes the device that forms a cut history soonest, by the history's size in samples (pulses times
# frequency samples). Measured on 16 threads of a Threadripper PRO 7965WX against an RTX PRO 6000 that another job
# kept busy, a 9.1 GB Capella file in the page cache, CPU / CUDA seconds of a process's later calls: 0.4e6 samples
# 0.03 / 0.05, 1.8e6 0.04 / 0.045, 7.0e6 0.055 / 0.053, 28e6 0.14 / 0.076, 112e6 0.49 / 0.15, 449e6 1.9 / 0.53: the
# GPU is the faster from about 1e7 samples, by 3 ns a sample. A process's first CUDA quick look also loads CuPy,
# makes the context, loads the kernels and plans the transforms: first calls of 0.10 / 0.82, 0.14 / 0.78, 0.24 /
# 0.84, 0.59 / 0.98 and 2.0 / 1.4 s at 0.4e6, 7e6, 28e6, 112e6 and 449e6 samples, a start of 0.7 s that the
# device repays from 2.5e8 samples. (tests/bench_quicklook.py measures both backends on another machine.)
AUTO_CUDA_SAMPLES = 10_000_000
AUTO_CUDA_FIRST_SAMPLES = 250_000_000
# On a TPU host the C++ polar format on the host's cores is taken below this size and the device from it on. Not
# measured: no TPU was at hand (tests/bench_tpu_blocks.py times both sides on one); set above the collections of
# this class until it is, since the device's program is compiled for each cut.
AUTO_TPU_SAMPLES = 4_000_000_000
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
        L.ql_resample_rows.argtypes = [p, n, n, p, p, n, p, i, p, i]
        L.ql_resample_az.argtypes = [p, n, n, p, d, d, p, n, d, d, n, p, i, p, n, i]
        L.ql_transpose.argtypes = [p, n, n, n, p, n, p, i]
        L.ql_azimuth_gate.argtypes = [p, n, n, n, p, p, d, d, p]
        L.ql_azimuth_gate.restype = None
        L.ql_gather.argtypes = [p, n, n, n, n, n, n, i, i, i, i, i, i, d, d, d, d, d, p, p, p, i, i, d, d, i, fl, i, p, p,
                                i, i, p, n, d]
        L.ql_weight_pad.argtypes = [p, n, n, p, p, n, n, p]
        L.ql_unshear.argtypes = [p, n, n, p, p, n, n, n, p, n]
        L.ql_resample_rows.restype = L.ql_resample_az.restype = L.ql_gather.restype = L.ql_weight_pad.restype = None
        L.ql_transpose.restype = L.ql_unshear.restype = None
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
    pulses and samples to read, the picture's center and its axes on the ground (ECF), its half extents and its
    pixel (m), and the collection's facts the notes and the product quote."""
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
    # the picture's axes, on the ground (form_cphd's axes='los'): e2 the mid-aperture line of sight projected on the
    # plane, away from the radar, e1 = e2 x up across it. Range then runs down a column of the picture whatever the
    # squint, and a tasked area that the collection states along these axes is the picture's rectangle
    r0 = float(np.linalg.norm(c - apc[mid]))
    los = (c - apc[mid]) / r0
    graze = float(np.degrees(np.arcsin(np.clip(-(los @ up), -1, 1))))
    if 1 - (los @ up) ** 2 < 0.02 ** 2:
        raise ValueError('the line of sight is within a degree of the vertical (a look at the nadir): its picture has '
                         'no ground scale in range. Use fastsar.process_cphd')
    e2 = _unit(los - (los @ up) * up)
    e1 = np.cross(e2, up)
    if extent is None:
        w = corners - c
        Hx, Hy = float(np.abs(w @ e1).max()), float(np.abs(w @ e2).max())
    else:
        Hx, Hy = 0.5 * float(extent[0]), 0.5 * float(extent[1])
    u = apc - c
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    ur = -(u @ e2)                                   # a meter along e2 is ur meters of range (cos of the grazing angle)
    t = (u @ e1) / ur                                # the tangent of the look angle on the ground
    sgn = 1.0 if t[-1] > t[0] else -1.0
    if ur.min() <= 0 or not np.all(np.diff(t) * sgn > 0):
        raise ValueError('the look angle does not change monotonically along the pulses, which polar format needs; '
                         'use fastsar.process_cphd')
    f_hi = f0 + (k_hi - 1) * df
    fcv = f0 + 0.5 * (k_lo + k_hi - 1) * df
    # the unambiguous extents (m, on the ground): a picture keeps inside them, with room for the gates' edges
    Xr = C / (2 * df * float(ur.max()))
    Xa = C / (2 * f_hi * float(np.abs(ur * np.gradient(t, idx)).max()))
    for name, H, X, what in (('across', Hx, Xa, 'pulse rate'), ('along', Hy, Xr, 'frequency step')):
        lim = 0.46 * X
        if H > lim:
            notes.append(f'the area spans {2 * H:.0f} m {name} the ground range direction, and the {what} leaves '
                         f'{X:.0f} m unambiguous: the picture is cut to the central {2 * lim:.0f} m (0.92 of that)')
            Hx, Hy = (lim, Hy) if name == 'across' else (Hx, lim)
    # the pixel, one size on both ground axes, and the finest one the collection holds on both: the band across the
    # ground range is that of the pulses' directions along e1, the band along it that of the samples times ur
    ux = u @ e1
    urm = float(ur[mid])
    bx_max, by_max = 2 * fcv / C * float(abs(ux[-1] - ux[0])), 2 * Kv * df / C * urm
    p_min = 0.96 / min(bx_max, by_max)
    if resolution == 'full':
        p = p_min
    elif resolution is not None:
        p = 0.96 * float(resolution)
    else:
        p = max(2 * Hx, 2 * Hy) / int(size)
    if p < p_min * (1 - 1e-9):
        notes.append(f'the collection holds a resolution of {p_min / 0.96:.3f} m on both ground axes; the picture has '
                     f'that, not the {p / 0.96:.3f} m asked for')
        p = p_min
    spx = spy = p
    nx, ny = (2 * int(np.ceil(H / p - 1e-6)) for H in (Hx, Hy))
    # the cut: the central samples and pulses whose bands are 0.96 / spacing; for several looks, as many such runs
    # side by side as were asked for and the collection holds
    Kc = int(min(Kv, max(2, round(0.96 / spy * C / (2 * df * urm)))))
    lr = int(max(1, min(looks[1], Kv // Kc)))
    k0 = (k_lo + (Kv - lr * Kc) // 2) // 16 * 16      # from a multiple of 16: the reader's rows as they are read
    k0 += 16 if k0 < k_lo else 0
    Kc = min(Kc, (k_hi - k0) // lr)                   # (the whole band: up to 15 samples short of it)
    fc = f0 + (k0 + 0.5 * (lr * Kc - 1)) * df
    need = 0.96 / spx * C / (2 * fc)                  # the span of the directions along e1 of one look
    la = int(max(1, min(looks[0], np.floor(abs(ux[-1] - ux[0]) / need + 1e-9))))
    order = np.argsort(ux)
    ends = np.interp([ux[mid] - la * need / 2, ux[mid] + la * need / 2], ux[order], idx[order].astype(np.float64))
    pa, pb = int(max(idx[0], np.floor(ends.min()))), int(min(idx[-1], np.ceil(ends.max()))) + 1
    if (pb - pa) // la < 2 * TAPS:
        raise ValueError(f'the picture asked for needs {(pb - pa) // la} pulses, fewer than the {2 * TAPS} polar '
                         'format resamples: ask for more pixels or a finer resolution')
    if (la, lr) != tuple(looks):
        notes.append(f'{la} by {lr} looks (sub-apertures by sub-bands), not the {looks[0]} by {looks[1]} asked for: '
                     'the collection holds no more at this resolution')
    return dict(channel=channel, pulses=(pa, pb), samples=(k0, k0 + lr * Kc), looks=(la, lr), P=P, K=K,
                valid=(k_lo, k_hi), center=c, e1=e1, e2=e2, up=up, Hx=Hx, Hy=Hy, spx=spx, spy=spy, nx=nx, ny=ny, pixel=p,
                resolution=p / 0.96, finest=p_min / 0.96, extents=(Xa, Xr), graze=graze, f0=f0, df=df)


# --------------------------------------------------------------------------------------------- the cut collection

def _collection(cphd, size=2048, resolution=None, extent=None, channel=0, reader_options=None, looks=(1, 1),
                backend='cpu'):
    """Plan the cut and read it: dict(S [P, K] (referenced to the grid center, unweighted), ant [P, 3] (about that
    center, local frame), fmin, df, nx, ny, spx, spy, e1, e2 (local): what form_image, ExactFormer and _pfa take;
    center [3] (local), meta (read_cphd's), plan, notes, timing (plan, read, reference: seconds)). looks: (sub-apertures,
    sub-bands): S then holds that many runs of pulses and of samples side by side (plan['looks'] of them). backend:
    'auto' is resolved once the cut is planned (_auto) and returned as 'backend'; for 'cuda' the reference to the
    grid center is left to the device, which applies it with the weights: S is then as the reader left it and
    'phase' holds the pulses' (ph0, ph1) for _pfa."""
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
        if backend == 'auto':
            backend = _auto((plan['pulses'][1] - plan['pulses'][0]) * (plan['samples'][1] - plan['samples'][0]))
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
    phase = None
    if backend == 'cuda' or (backend in ('jax', 'tpu') and _jax_algorithm() == 'pfa'):
        # S[p, k] exp(-2 pi j (f0 + k df) 2 dref[p] / c), on the device
        a = -f0 * (2 * dref / C)
        phase = (a - np.floor(a), -df * (2 * dref / C))
    else:
        fk = f0 + df * np.arange(K)

        def freq(sl, out):
            out[:] = fk
        io._phase_rows(S, freq, 2 * dref / C, -1, linear=(f0, df))
    timing['reference'] = time.perf_counter() - t0
    notes = list(meta.get('notes', [])) + notes
    return dict(S=S, ant=rel, fmin=f0, df=df, nx=plan['nx'], ny=plan['ny'], spx=plan['spx'], spy=plan['spy'], e1=e1,
                e2=e2, center=c, meta=meta, plan=plan, notes=notes, timing=timing, backend=backend, phase=phase)


# ------------------------------------------------------------------------------------------------ polar format

def _distortion(ant, nx, ny, spx, spy, e1, e2, weights=None, n=21, deg=3):
    """Where polar format puts a scatterer, for any pair of grid axes (a slant plane, the ground plane). With
    ur = -u . e2 and t = (u . e1) / ur of each pulse, the raster's wavenumbers are kr = 4 pi f ur / c and ka = kr t,
    and a scatterer at x e1 + y e2 whose range history is dR has the phase kr (-dR / ur): the weighted
    least-squares line -dR / ur = t X - Y over the aperture gives its apparent position (X, Y), the peak of its
    response, and leaves no mean phase there (weights: the pulses' window, so that the corrected image has
    backprojection's phase). The mapping is fitted by 2-D polynomials of degree deg on an n by n grid over the
    picture -> the dict of pfa2.distortion (which takes e2 along the line of sight, where ur is 1 at mid-aperture)."""
    ant = np.asarray(ant, np.float64)
    r0 = np.linalg.norm(ant, axis=1)
    u = ant / r0[:, None]
    ur = -(u @ e2)
    w = np.sqrt(np.ones(len(ant)) if weights is None else np.asarray(weights, np.float64))
    pinv = np.linalg.pinv(np.stack([(u @ e1) / ur, -np.ones(len(ant))], 1) * w[:, None]) * w        # [2, P]
    xs, ys = np.linspace(-nx / 2.0, nx / 2.0, n) * spx, np.linspace(-ny / 2.0, ny / 2.0, n) * spy
    X, Y = (v.ravel() for v in np.meshgrid(xs, ys, indexing='ij'))
    pts = X[:, None] * e1 + Y[:, None] * e2
    dR = np.linalg.norm(pts[:, None, :] - ant[None, :, :], axis=2) - r0[None, :]
    g = -dR / ur[None, :]
    xa, ya = g @ pinv[0], g @ pinv[1]
    powers = [(a, b) for a in range(deg + 1) for b in range(deg + 1 - a)]
    sx, sy = nx / 2.0 * spx, ny / 2.0 * spy
    B = np.stack([(X / sx) ** a * (Y / sy) ** b for a, b in powers], 1)
    cx, cy = np.linalg.lstsq(B, xa - X, rcond=None)[0], np.linalg.lstsq(B, ya - Y, rcond=None)[0]
    rms = (float(np.sqrt(np.mean((B @ cx - (xa - X)) ** 2)) / spx), float(np.sqrt(np.mean((B @ cy - (ya - Y)) ** 2)) / spy))
    return dict(cx=cx, cy=cy, powers=powers, sx=sx, sy=sy, rms_px=rms,
                max_shift_px=(float(np.abs(xa - X).max() / spx), float(np.abs(ya - Y).max() / spy)))


def _places(t):
    """The place of each pulse in the evenly timed sequence, from the tangents t of the look angle [P]: 0 .. P - 1
    where no pulse is missing; where the step to a pulse is a whole multiple (to a fifth) of the steps around it,
    the pulses between were not recorded and the places skip them. A sequence whose rate changes smoothly, or one
    with more gaps than a twentieth of its pulses, is taken as it is."""
    P = len(t)
    d = np.abs(np.diff(t))
    if P < 16:
        return np.arange(P)
    k = np.arange(P - 1)
    near = np.stack([d[np.clip(k + o, 0, P - 2)] for o in (-4, -3, -2, 2, 3, 4)])
    r = d / np.median(near, axis=0)
    m = np.rint(r)
    gap = (m >= 2) & (np.abs(r - m) < 0.2)
    if not gap.any() or (m[gap] - 1).sum() > P / 20:
        return np.arange(P)
    return np.concatenate([[0], np.cumsum(np.where(gap, m, 1.0))]).astype(np.int64)


def _geometry(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, correct, ov=OVERSAMPLE, window=True, warn=False,
              support='union'):
    """The float64 setup of the quick look's polar format for a cut history: the raster, the two gates, the
    resampling positions and the distortion model. e1, e2: the picture's axes, any pair that spans its plane with
    e2 away from the radar (a slant plane along the line of sight, the ground plane along the ground range): the
    raster's wavenumbers are the components of the samples' along them."""
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
    # the pulses' places in the evenly timed sequence: a collection may lack pulses (an Umbra collection lacked six
    # of 9,038), and along the pulses that are there the phase of a scatterer away from the center then steps. Each
    # pulse is set at its place, a missing one's left zero, as backprojection sums nothing for it
    place = _places(t)
    Pp = int(place[-1]) + 1
    dts = (t[-1] - t[0]) / (Pp - 1)                                   # signed, per place
    dtp = np.gradient(t, place.astype(np.float64)) if Pp > P else np.gradient(t)
    if warn and abs(float(t[P // 2])) > math.tan(math.radians(2.0)):   # form_image's caller chose the grid
        import warnings
        warnings.warn(f"polar format: the grid's range axis e2 is {abs(math.degrees(math.atan(float(t[P // 2])))):.1f} "
                      'degrees from the mid-aperture line of sight in the image plane; the polar raster is taken about '
                      'e2 and the image degrades quickly with this angle. Rotate the grid so that e2 points along the '
                      "line of sight (the vendors' image grids do), or use algorithm='ffbp' or 'bp'", UserWarning,
                      stacklevel=4)
    f = fmin + df * np.arange(K)
    fc = fmin + (K // 2) * df
    Hx, Hy = 0.5 * nx * spx, 0.5 * ny * spy
    # where polar format puts the scatterers (a few hundred pulses give the fit)
    sub = np.unique(np.round(np.linspace(0, P - 1, min(P, 257))).astype(int))
    dist = _distortion(ant[sub], nx, ny, spx, spy, e1, e2, _taylor(P)[sub] if window else None)
    shx, shy = (float(v) for v in (dist['max_shift_px'][0] * spx, dist['max_shift_px'][1] * spy))
    # the gates. Range: the slant delays of the picture's points, exactly, over the aperture, with the displacement
    # (an uncorrected picture shows points from that far outside) and a pixel; azimuth: the picture's extent with
    # the displacement and the spread of a point's azimuth bins that the uneven pulse spacing makes. A meter along
    # e2 is ur meters of delay (1 on a slant plane along the line of sight, the cosine of the grazing angle on the
    # ground), so the range gate, which is set in delay, takes its guard and its pixel times ur
    ur_lo, ur_hi = float(ur.min()), float(ur.max())
    gx, gy = max(8 * spx, 0.03 * Hx), max(8 * spy, 0.03 * Hy) * ur_lo
    xs, ys = (Hx + shx) * np.array([-1, 0, 1]), (Hy + shy) * np.array([-1, 0, 1])
    pts = (xs[:, None, None] * e1 + ys[None, :, None] * e2).reshape(-1, 3)
    dR = np.linalg.norm(pts[:, None] - ant[sub][None], axis=2) - r0[sub][None]
    Er = float(np.abs(dR).max()) + 2 * spy * ur_hi
    Ex = (Hx + shx) * float(1 + np.abs(dtp / dts - 1).max()) + 2 * spx
    # the raster: its periods hold the picture, the displacement and a gate's far edge
    nfy = _fft_size(int(np.ceil((Hy + shy + (Er + gy) / ur_lo) / spy)) + 2)
    nfx = _fft_size(int(np.ceil((Hx + shx + Ex + gx) / spx)) + 2)
    dkr, dka = 2 * math.pi / (nfy * spy), 2 * math.pi / (nfx * spx)
    kr_lo, kr_hi = 4 * math.pi * f[0] / C * ur, 4 * math.pi * f[-1] / C * ur     # each pulse's band
    kr_min, kr_max = float(kr_lo.min()), float(kr_hi.max())
    nkr = int((kr_max - kr_min) / dkr) + 1                                        # the rows the bands reach together
    nwin = int(float((kr_hi - kr_lo).max()) / dkr) + 1                           # and those of one pulse
    jlo, hy, shear = None, None, 0.0
    if support == 'inscribed':                      # form_image's pfa_support: the rows every pulse reaches
        kr_min, kr_max = float(kr_lo.max()), float(kr_hi.min())
        if kr_max <= kr_min:
            raise ValueError("polar format: no range band is common to all pulses (support='inscribed'); use "
                             "support='union'")
        nkr = min(int((kr_max - kr_min) / dkr) + 1, nfy)
    elif nkr - nwin > max(8, 0.05 * nwin):
        # The band drifts along the aperture: e2 is not the line of sight (a picture on the ground under a slant
        # plane twisted about it), ur changes from pulse to pulse and the samples fill a sheared parallelogram of
        # the raster. Each pulse is resampled on its own nwin rows, from row jlo; the rows are then taken across
        # the pulses that reach them, and they fold into the transform's period, which holds a pulse's band and
        # need not hold their union. shear: rows of drift per unit of azimuth wavenumber
        mid = 0.5 * (kr_lo + kr_hi)
        if nwin > nfy:                              # a band wider than the spacing resolves: each pulse's center
            nwin = nfy
            jlo = np.round((mid - kr_min) / dkr).astype(np.int64) - nwin // 2
        else:
            jlo = np.ceil((kr_lo - kr_min) / dkr - 1e-9).astype(np.int64)
        jlo = np.maximum(jlo, 0).astype(np.int32)
        nkr = int(jlo.max()) + nwin
        hy = int(jlo[P // 2]) + nwin // 2
        shear = float(np.polyfit(mid * t, mid, 1)[0])
        first, last = kr_min + dkr * jlo, kr_min + dkr * (jlo + nwin - 1)
        ka_lo, ka_hi = float(min((first * t).min(), (last * t).min())), float(max((first * t).max(), (last * t).max()))
    elif nkr > nfy:                                 # a band wider than the spacing resolves: its center
        if warn and nkr > 1.03 * nfy:
            import warnings
            warnings.warn(f'polar format: the range band ({nkr} samples) exceeds the {nfy}-sample raster of the grid; '
                          f'the central {nfy} are kept, so the range resolution is limited by the pixel spacing',
                          UserWarning, stacklevel=4)
        kr_min += ((nkr - nfy) // 2) * dkr
        nkr = nfy
    kr = kr_min + dkr * np.arange(nkr)
    kr_max = kr[-1]
    if jlo is None:
        nwin, hy = nkr, nkr // 2
        kref = kr_min if support == 'inscribed' else kr_max          # the azimuth band common to the rows, or the widest
        ka_lo, ka_hi = kref * t.min(), kref * t.max()
    nka = int((ka_hi - ka_lo) / dka) + 1
    if nka > nfx:
        if warn and nka > 1.03 * nfx:
            import warnings
            warnings.warn(f'polar format: the azimuth band of the aperture ({nka} samples) exceeds the {nfx}-sample '
                          'raster of the grid; the central ones are kept, so the azimuth resolution is limited by the '
                          'pixel spacing (a finer spacing or a shorter aperture keeps the whole band)', UserWarning,
                          stacklevel=4)
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
    if jlo is not None:
        alpha = alpha + jlo * beta
    # azimuth gate: P pulses at p0 of Pf, the bins of row kr 2 pi / (kr |dts| Pf), Na kept, the pulses back on Na
    Xa = 2 * math.pi / (kr_max * abs(dts))          # the shortest period, the highest wavenumber's
    rho_a = 2 * (Ex + gx) / PASS / Xa
    p0 = int(np.ceil((TAPS // 2 + 1) / rho_a)) + 1
    Pf = _fft_size(Pp + 2 * p0)
    Na = _fft_size(int(np.ceil(rho_a * Pf)))
    bin_a = Xa / Pf
    bins_a = 2 * math.pi / (kr * abs(dts) * Pf)     # the bin of each range wavenumber's transform (m): wider at a lower one
    # the row position (in the Na pulses) of a tangent t, from the pulses' own t: a table at four points per pulse
    # over the aperture and a few pulses beyond it (linear continuation)
    nt = 4 * (Pp + 2 * TAPS) + 1
    tg = (t[0] - TAPS * dts) + (dts / 4) * np.arange(nt)
    pp = place.astype(np.float64)
    if sgn > 0:
        ptab = np.interp(tg, t, pp)
    else:
        ptab = np.interp(tg[::-1], t[::-1], pp[::-1])[::-1]
    out_lo, out_hi = (tg - t[0]) / dts < 0, (tg - t[-1]) / dts > 0
    ptab[out_lo] = ((tg - t[0]) / dtp[0])[out_lo]
    ptab[out_hi] = (Pp - 1 + (tg - t[-1]) / dtp[-1])[out_hi]
    qtab = (ptab + p0) * (Na / Pf)
    # weights: the polar Jacobian (the raster's density of samples: backprojection sums the samples as they come)
    wk = fc / f
    wp = (abs(dts) / np.abs(dtp)) / ur ** 2
    kr_cw = 4 * math.pi * fc / C
    gain = (dkr * C / (4 * math.pi * df)) * (dka / (kr_cw * abs(dts)))
    return dict(P=P, K=K, nx=nx, ny=ny, spx=spx, spy=spy, nfx=nfx, nfy=nfy, dkr=dkr, dka=dka, kr_min=kr_min, nkr=nkr,
                ka_lo=ka_lo, nka=nka, kr=kr, k0=k0, Kf=Kf, Nr=Nr, bin_r=bin_r, Er=Er, gy=gy, alpha=alpha, beta=beta,
                p0=p0, Pf=Pf, Na=Na, bin_a=bin_a, bins_a=bins_a, Ex=Ex, gx=gx, qtab=qtab, tg0=float(tg[0]), tg_inv=float(4 / dts),
                wk=wk, wp=wp, gain=gain, dist=dist, shift=(shx, shy), ov=ov, t=t, dts=dts, nwin=nwin, jlo=jlo, hy=hy,
                shear=shear, Pp=Pp, place=place.astype(np.int32) if Pp > P else None)


def _corner_error(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, window=True):
    """Polar format's own error at the picture's corners, as (phase, dB). Its planar wavefronts leave a phase over
    the aperture and the band for a scatterer away from the center; the part of it that is linear in the two
    wavenumbers is the displacement, which the correction removes, and the rest defocuses the scatterer. phase is the
    most of that rest in radians, and dB the relative energy of the difference it makes to the weighted point
    response with one complex gain fitted, at the worst of the four corners. The phase is smooth, so 65 pulses by 9
    frequencies carry it."""
    P = len(ant)
    ip = np.unique(np.linspace(0, P - 1, min(P, 65)).round().astype(np.int64))
    ik = np.unique(np.linspace(0, K - 1, min(K, 9)).round().astype(np.int64))
    a = np.asarray(ant, np.float64)[ip]
    R = np.linalg.norm(a, axis=1)
    u = a / R[:, None]
    k = 4 * np.pi * (fmin + df * ik) / C
    w = np.ones((len(ip), len(ik)))
    if window:
        w = (_taylor(len(ip))[:, None] * _taylor(len(ik))[None, :]) ** 2
    w = (w / w.sum()).ravel()
    kx, ky = ((u @ e1)[:, None] * k).ravel(), ((u @ e2)[:, None] * k).ravel()
    A = np.stack([np.ones(w.size), (kx - w @ kx) / np.ptp(kx), (ky - w @ ky) / np.ptp(ky)], 1)
    fit = np.linalg.pinv(A * np.sqrt(w)[:, None]) * np.sqrt(w)          # the weighted least-squares linear part
    worst = (0.0, -120.0)
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            r = sx * nx * spx * e1 + sy * ny * spy * e2
            phi = ((np.linalg.norm(a - r, axis=1) - R + u @ r)[:, None] * k).ravel()
            left = phi - A @ (fit @ phi)
            err = 1.0 - abs(np.sum(w * np.exp(1j * left))) ** 2
            worst = max(worst, (float(np.abs(left).max()), float(10 * np.log10(max(err, 1e-12)))), key=lambda v: v[1])
    return worst


def _taper(n, edge, guard, binsize):
    """A gate over the n bins of a transform, in the transform's own (wrapped) order: 1 within `edge` meters of
    bin 0, a raised cosine to 0 over the next `guard` meters, 0 beyond (float32)."""
    k = np.arange(n)
    d = np.minimum(k, n - k) * binsize
    return np.where(d <= edge, 1.0, np.where(d >= edge + guard, 0.0,
                                             0.5 + 0.5 * np.cos((d - edge) * (math.pi / guard)))).astype(np.float32)


def _fold(B, n, origin):
    """The rows of a raster (the columns J of B [m, nkr]) summed into the n bins of a transform's period: column J
    into bin (J - origin) mod n. A transform of n points takes its samples at multiples of its period's reciprocal,
    where rows n apart are the same, so the fold changes nothing at the pixels."""
    nkr = B.shape[1]
    Z = np.zeros((B.shape[0], n), np.complex64)
    J = 0
    while J < nkr:
        c = (J - origin) % n
        w = min(nkr - J, n - c)

        def add(sl, J=J, c=c, w=w):
            Z[sl, c:c + w] += B[sl, J:J + w]
        _blocks(add, B.shape[0], int(max(1, (1 << 18) // max(w, 1))))
        J += w
    return Z


def _pfa_cpu(S, g, window, correct, complex_image, timing):
    """The quick look's polar format on the host -> the image, range-major [ny, nx]: complex64 with backprojection's
    phase (complex_image) or the amplitude (float32). The passes over the arrays are fused and run on all cores:
    NumPy's products in row blocks on the pool, scipy's transforms with its workers, the kernels with their own
    threads."""
    import ctypes
    import scipy.fft as sf
    lib = _lib()
    W = _workers()
    S = S.pop() if isinstance(S, list) else S          # handed over: freed below, once it is gated
    P, K, Kf, Nr, k0 = g['P'], g['K'], g['Kf'], g['Nr'], g['k0']
    nkr, nka, nfy, nfx, nx, ny = g['nkr'], g['nka'], g['nfy'], g['nfx'], g['nx'], g['ny']
    tab = _polyphase('kaiser', TAPS, PHASES)
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    t0 = time.perf_counter()
    at = lambda a, row=0, col=0: ctypes.c_void_p(a.ctypes.data + row * a.strides[0] + (col * a.strides[1] if col else 0))
    wk = np.ascontiguousarray(g['wk'] * (_taylor(K) if window else 1.0), np.float32)
    wp = np.ascontiguousarray(g['wp'] * (_taylor(P) if window else 1.0), np.float32)
    # 1. range, in blocks of pulses on the pool, each block on its thread from the history to the raster's range
    # wavenumbers: weighted and padded (one pass), to range profiles, the gate, back to the spectrum on Nr samples,
    # and the kernel. Only the blocks' own arrays are made, a few megabytes each
    tap_r = _taper(Nr, g['Er'], g['gy'], g['bin_r'])
    keep = min(Nr, Kf)
    pos, neg = keep - keep // 2, keep // 2
    alpha, beta = np.ascontiguousarray(g['alpha'], np.float64), np.ascontiguousarray(g['beta'], np.float64)
    nwin, jlo = g['nwin'], g['jlo']                       # a pulse's rows of the raster (all of them, or its own)
    A = np.empty((P, nwin), np.complex64)

    def rows(sl):
        n = sl.stop - sl.start
        buf = np.empty((n, Kf), np.complex64)
        lib.ql_weight_pad(at(S, sl.start), n, K, ptr(wk), at(wp, sl.start), k0, Kf, ptr(buf))
        prof = sf.ifft(buf, axis=1, overwrite_x=True)
        gate = np.empty((n, Nr), np.complex64) if Nr <= Kf else np.zeros((n, Nr), np.complex64)
        np.multiply(prof[:, :pos], tap_r[:pos], out=gate[:, :pos])
        np.multiply(prof[:, Kf - neg:], tap_r[Nr - neg:], out=gate[:, Nr - neg:])
        G = np.ascontiguousarray(sf.fft(gate, axis=1, overwrite_x=True))
        lib.ql_resample_rows(ptr(G), n, Nr, at(alpha, sl.start), at(beta, sl.start), nwin, ptr(tab), PHASES,
                             at(A, sl.start), 1)
    _blocks(rows, P, int(max(1, min(64, (1 << 21) // (8 * max(Kf, Nr))))))
    S = None
    timing['range'] = timing.get('range', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # 2. azimuth, in blocks of range wavenumbers on the pool. A block is turned so that the pulses of each
    # wavenumber lie along a row (every transform runs along rows), transformed (padded to Pf), cut to Na bins,
    # transformed back to Na pulses and read by the kernel at the raster's azimuth wavenumbers. The gate carries
    # the image's scale and the phase that sets the pulses p0 places into the Na. The raster is written
    # azimuth-wavenumber major, for the range transform that follows
    Pf, Na, p0, Pp, place = g['Pf'], g['Na'], g['p0'], g['Pp'], g['place']
    ov = g['ov']
    Nfft = ov * nfy if correct else nfy
    k = np.arange(Na)
    k = np.where(k < Na - Na // 2, k, k - Na)
    fac = np.ascontiguousarray((g['gain'] * Nfft * Na / Pf) * np.exp(-2j * np.pi * (k * p0 % Pf) / Pf), np.complex64)
    bins = np.ascontiguousarray(g['bins_a'])                              # a bin of each row's transform, in meters
    inv_kr = np.ascontiguousarray(1.0 / g['kr'])
    qtab = np.ascontiguousarray(g['qtab'])
    B = np.empty((nka, nkr), np.complex64)

    def cols(sl):
        n = sl.stop - sl.start
        buf = np.empty((n, Pf), np.complex64)
        if jlo is None and place is None:
            lib.ql_transpose(at(A, 0, sl.start), nkr, P, n, ptr(buf), Pf, None, 1)
        else:
            lib.ql_unshear(ptr(A), nwin, P, None if jlo is None else ptr(jlo), None if place is None else ptr(place), Pp,
                           sl.start, n, ptr(buf), Pf)
        buf[:, Pp:] = 0
        prof = np.ascontiguousarray(sf.fft(buf, axis=1, overwrite_x=True))
        gate = np.empty((n, Na), np.complex64)
        lib.ql_azimuth_gate(ptr(prof), n, Pf, Na, ptr(fac), at(bins, sl.start), g['Ex'], g['gx'], ptr(gate))
        A2 = np.ascontiguousarray(sf.ifft(gate, axis=1, overwrite_x=True))
        lib.ql_resample_az(ptr(A2), n, Na, at(inv_kr, sl.start), g['ka_lo'], g['dka'], ptr(qtab), len(qtab), g['tg0'],
                           g['tg_inv'], nka, ptr(tab), PHASES, at(B, 0, sl.start), nkr, 1)
    _blocks(cols, nkr, int(max(8, min(64, (1 << 22) // (8 * max(Pf, Na))))))
    del A
    timing['azimuth'] = timing.get('azimuth', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    out = np.empty((ny, nx), np.complex64 if complex_image else np.float32)
    cb = int(max(1, (1 << 18) // nx))
    if not correct:
        # 3. the raster's transform, cut to the picture: the range transform along the rows, whose last ny / 2
        # outputs are the pixels before the center and whose first ny / 2 are those from it on, turned to
        # range-major with those in their places; then the azimuth transform along the rows, cut likewise
        if nkr > nfy:                                                   # the rows fold into the transform's period
            B = _fold(B, nfy, 0)
        F = sf.ifft(B, n=nfy, axis=1, workers=W)                        # [nka, nfy]
        del B
        Z = np.empty((ny, nfx), np.complex64)
        lib.ql_transpose(at(F, 0, nfy - ny // 2), nfy, nka, ny // 2, ptr(Z), nfx, None, 0)
        lib.ql_transpose(ptr(F), nfy, nka, ny - ny // 2, at(Z, ny // 2), nfx, None, 0)
        del F
        rb = int(max(1, (1 << 18) // nfx))
        if nfx > nka:
            def pad(sl):
                Z[sl, nka:] = 0
            _blocks(pad, ny, rb)
        T = sf.fft(Z, axis=1, workers=W, overwrite_x=True)              # [ny, nfx]
        del Z
        if complex_image:
            y, x = (np.arange(ny) - ny / 2.0) * g['spy'], (np.arange(nx) - nx / 2.0) * g['spx']
            cy, cx = np.exp(1j * g['kr_min'] * y).astype(np.complex64), np.exp(-1j * g['ka_lo'] * x).astype(np.complex64)

        def put(sl):
            o = out[sl]
            for c_out, c_in in ((0, nfx - nx // 2), (nx // 2, 0)):
                if complex_image:
                    np.multiply(T[sl, c_in:c_in + nx // 2], cx[c_out:c_out + nx // 2], out=o[:, c_out:c_out + nx // 2])
                else:
                    np.abs(T[sl, c_in:c_in + nx // 2], out=o[:, c_out:c_out + nx // 2])
            if complex_image:
                o *= cy[sl, None]
        _blocks(put, ny, cb)
        del T
        timing['image'] = timing.get('image', 0.0) + time.perf_counter() - t0
        return out
    # 4. the same on a raster padded by ov, at baseband, in blocks of range pixels read at the apparent positions.
    # The spectrum is set about index 0 of the padded transforms with alternating signs, so that their outputs
    # count from the image's center
    dist = g['dist']
    Ny2, Nx2 = ov * nfy, ov * nfx
    hy, hx = g['hy'], nka // 2
    sy = ((-1.0) ** (np.arange(nkr) - hy)).astype(np.float32)
    sx = np.ascontiguousarray((-1.0) ** (np.arange(nka) - hx), np.float32)
    rb = int(max(1, (1 << 18) // Ny2))
    if nkr > Ny2:                                                          # the rows fold into the padded period
        B *= sy
        Z = _fold(B, Ny2, hy)
    else:
        Z = np.empty((nka, Ny2), np.complex64)

        def fill(sl):
            np.multiply(B[sl, hy:], sy[hy:], out=Z[sl, :nkr - hy])
            Z[sl, nkr - hy:Ny2 - hy] = 0
            np.multiply(B[sl, :hy], sy[:hy], out=Z[sl, Ny2 - hy:])
        _blocks(fill, nka, rb)
    del B
    F1 = sf.ifft(Z, axis=1, workers=W, overwrite_x=True)                   # [nka, Ny2]: range done
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
    block = np.empty((min(Ny2, ov * bw + 2 * msy), Nx2), np.complex64)     # one array for every block
    for jb in range(0, ny, bw):
        n = min(bw, ny - jb)
        rb0 = max(0, Ny2 // 2 + ov * (jb - ny // 2) - msy)
        span = min(Ny2, Ny2 // 2 + ov * (jb + n - ny // 2) + msy) - rb0
        t1 = time.perf_counter()
        # the block's range rows, turned to range-major about index 0 of the padded azimuth transform
        Z = block[:span]
        lib.ql_transpose(at(F1, hx, rb0), Ny2, nka - hx, span, ptr(Z), Nx2, at(sx, hx), 0)
        lib.ql_transpose(at(F1, 0, rb0), Ny2, hx, span, at(Z, 0, Nx2 - hx), Nx2, ptr(sx), 0)

        def pad(sl, Z=Z):
            Z[sl, nka - hx:Nx2 - hx] = 0
        _blocks(pad, span, rb)
        base = np.ascontiguousarray(sf.fft(Z, axis=1, workers=W, overwrite_x=True))
        del Z
        t2 = time.perf_counter()
        lib.ql_gather(ptr(base), span, Nx2, Nx2, Ny2, rb0, 0, nx, n, 0, jb, nx, ny, float(g['spx']),
                      float(g['spy']), float(ov), float(dist['sx']), float(dist['sy']), ptr(cx), ptr(cy), ptr(pw),
                      len(cx), deg, float(kr_c), float(ka_c), 1, 1.0, LANCZOS, ptr(ltab), ptr(ltab16), PHASES,
                      0 if complex_image else 1, ctypes.c_void_p(out.ctypes.data + jb * out.strides[0]), nx,
                      float(g['shear'] * g['spy'] / g['spx']))
        del base
        tt[0] += t2 - t1
        tt[1] += time.perf_counter() - t2
    timing['azimuth transform'] = timing.get('azimuth transform', 0.0) + tt[0]
    timing['correction'] = timing.get('correction', 0.0) + tt[1]
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
         timing=None, info=None, warn=False, support='union', scale=None, layout='range', phase=None):
    """The quick look's polar format of a history S [P, K] referenced to |ant| on the grid of form_image (e2 along
    the mid-aperture line of sight, away from the radar) -> the image [nx, ny]: complex64 as backprojection gives
    it, or its amplitude (float32, complex_image False); a NumPy array, or a CuPy array for backend 'cuda'.
    window: the Taylor window (nbar 4, 35 dB) along frequency and along pulses. info: a dict that receives the
    geometry's facts (the displacement, the raster and gate sizes). scale: a factor for the image, or a function of
    the geometry that gives one (form_image's polar format keeps its JAX program's scale). layout: 'range' returns
    the image as it is formed, range-major, as the transposed view [nx, ny]; 'azimuth' returns a contiguous
    [nx, ny] array (form_image's). phase: (ph0, ph1) [P] float64 in cycles, a phase exp(2 pi j (ph0 + k ph1)) still to
    be given to sample k of each pulse (the quick look's reference to the grid center): the device applies it with
    the weights, the host in a pass of its own."""
    timing = {} if timing is None else timing
    t0 = time.perf_counter()
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    ant = np.asarray(ant, np.float64)
    if (nx % 2 or ny % 2) and not correct:             # corrected, every pixel is read at its own position
        raise ValueError(f'the uncorrected image takes even pixel counts, got {nx} by {ny}')
    box = S if isinstance(S, list) else [S]            # a one-element list hands the history over: it is freed
    del S                                              # as soon as the range stage has read it
    g = _geometry(ant, float(fmin), float(df), box[0].shape[1], int(nx), int(ny), float(spx), float(spy), e1, e2, correct,
                  window=bool(window), warn=warn, support=support)
    timing['geometry'] = timing.get('geometry', 0.0) + time.perf_counter() - t0
    info = {} if info is None else info
    info.update(shift_px=g['dist']['max_shift_px'], fit_rms_px=g['dist']['rms_px'], raster=(g['nkr'], g['nka']),
                fft=(g['nfy'], g['nfx']), range_gate=(g['Kf'], g['Nr']), azimuth_gate=(g['Pf'], g['Na']), dts=g['dts'])
    if callable(scale):
        scale = scale(g)
    if scale is not None:
        g['gain'] = g['gain'] * float(scale)
    if backend == 'cuda':
        img = _pfa_cuda(box, g, window, correct, complex_image, timing, phase, info)
    elif backend in ('jax', 'tpu'):
        from . import quick_jax
        img = quick_jax.pfa(box, g, window, correct, complex_image, timing, phase, info)
    else:
        box[0] = np.ascontiguousarray(box[0], np.complex64)
        if phase is not None:
            _apply_phase(box[0], *phase)
        img = _pfa_cpu(box, g, window, correct, complex_image, timing)
    if layout == 'range':
        return img.T
    t0 = time.perf_counter()
    if type(img).__module__.startswith('cupy'):
        out = __import__('cupy').ascontiguousarray(img.T)
    elif img.dtype == np.complex64:                    # turned in tiles on all cores
        import ctypes
        out = np.empty(img.shape[::-1], np.complex64)
        _lib().ql_transpose(img.ctypes.data_as(ctypes.c_void_p), img.shape[1], img.shape[0], img.shape[1],
                            out.ctypes.data_as(ctypes.c_void_p), img.shape[0], None, 0)
    else:
        out = np.ascontiguousarray(img.T)
    timing['layout'] = timing.get('layout', 0.0) + time.perf_counter() - t0
    return out


def _host(d):
    """A CuPy array on the host; a large one through page-locked memory, several times faster than into pageable
    memory (the array returned then lies in that memory, which CuPy's pool takes back when it is released)."""
    from . import memory
    if d.nbytes < (16 << 20):
        return d.get()
    host = np.frombuffer(memory._pinned(d.nbytes), d.dtype, d.size).reshape(d.shape)
    d.get(out=host)
    return host


def _apply_phase(S, ph0, ph1):
    """S[p, k] *= exp(2 pi j (ph0[p] + k ph1[p])) in place, in blocks of pulses on the pool."""
    k = np.arange(S.shape[1])

    def rows(sl):
        c = ph0[sl, None] + k[None, :] * ph1[sl, None]
        S[sl] *= np.exp(2j * np.pi * (c - np.floor(c))).astype(np.complex64)
    _blocks(rows, S.shape[0], int(max(1, (1 << 17) // S.shape[1])))


# ------------------------------------------------------------------------------------------------------- CUDA
# The stages on a GPU, written for it: the history goes up in blocks of pulses through two page-locked buffers while
# the device works on the block before; each pass over an array that is not a transform is one kernel (the weights,
# the padding and the reference phase; the range gate and its cut; the azimuth gate, its cut and the scale; the
# raster's place in the padded transforms with its signs; the read at the apparent positions with the detection);
# the transforms keep their plans between calls (CuPy's plan cache) and the kernels and tables stay compiled and
# loaded (_CUDA).

_CUDA_SRC = r"""
#define HALF_T (TAPS / 2)
#define HALF_L (LTAPS / 2)
__device__ __forceinline__ float tabled(float d, const float* tab) {
  const float ad = fabsf(d);
  if (ad >= HALF_T) return 0.f;
  const float u = ad * TABN;
  const int k = (int)u;
  return fmaf(tab[k + 1] - tab[k], u - k, tab[k]);
}
// quicklook_cpu.cpp's ql_gather on the device: one thread per pixel, i along azimuth (fastest), c the range pixel of
// the block. The apparent position from the distortion polynomial in float64, the baseband image (the block's rows
// rb0 .. of [Ny2, Nx2], oversampled by ov) read there with LTAPS x LTAPS taps whose weights come from the polyphase
// table tab [Q + 1, LTAPS]; each row of taps at the column fx + shear (row - fy). The result times exp(j (kr_c Ya -
// ka_c Xa)), or its amplitude (detect), written range-major: out[c, i].
extern "C" __global__ void ql_gather(const float2* base, int span, int Nx2, int Ny2, int rb0, int n, int jb, int nx,
                                     int ny, double spx, double spy, double ov, double dsx, double dsy,
                                     const double* cx, const double* cy, const int* pw, double kr_c, double ka_c,
                                     double shear, const float* tab, int detect, float gain, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
  if (i >= nx || c >= n) return;
  const double x = (i - nx / 2.0) * spx, y = (jb + c - ny / 2.0) * spy;
  const double xn = x / dsx, yn = y / dsy;
  double Xa = x, Ya = y, px[DEG + 1], py[DEG + 1];
  px[0] = 1.0; py[0] = 1.0;
  #pragma unroll
  for (int k = 1; k <= DEG; ++k) { px[k] = px[k-1] * xn; py[k] = py[k-1] * yn; }
  #pragma unroll
  for (int k = 0; k < NPOW; ++k) {
    const double t = px[pw[2*k]] * py[pw[2*k+1]];
    Xa += cx[k] * t; Ya += cy[k] * t;
  }
  const double fx = Xa * (ov / spx) + Nx2 / 2.0, fy = Ya * (ov / spy) + Ny2 / 2.0;
  const double fiy = floor(fy);
  const int iy0 = (int)fiy - HALF_L + 1;
  const float* wy = tab + (int)((fy - fiy) * QTAB + 0.5) * LTAPS;
  float ar = 0.f, ai = 0.f;
  #pragma unroll
  for (int a = 0; a < LTAPS; ++a) {
    const int row = iy0 + a;
    const double fxa = fx + shear * ((double)row - fy), fl = floor(fxa);
    const int ix0 = (int)fl - HALF_L + 1;
    const float* wx = tab + (int)((fxa - fl) * QTAB + 0.5) * LTAPS;
    const float2* r = base + (long)min(max(row - rb0, 0), span - 1) * Nx2;
    float re = 0.f, im = 0.f;
    #pragma unroll
    for (int b = 0; b < LTAPS; ++b) {
      const float2 v = r[min(max(ix0 + b, 0), Nx2 - 1)];
      re = fmaf(wx[b], v.x, re); im = fmaf(wx[b], v.y, im);
    }
    ar = fmaf(wy[a], re, ar); ai = fmaf(wy[a], im, ai);
  }
  ar *= gain; ai *= gain;
  const long o = (long)c * nx + i;
  if (detect) { out[o] = sqrtf(ar * ar + ai * ai); return; }
  double sn, cs;
  sincos(kr_c * Ya - ka_c * Xa, &sn, &cs);
  const float cr = (float)cs, ci = (float)sn;
  out[2 * o] = ar * cr - ai * ci; out[2 * o + 1] = ar * ci + ai * cr;
}
// The passes that are not transforms, one kernel each: a thread per output value, x along a row, y the row of a
// block of at most 32,768 rows, whose first row is row r0 of the whole array (the arrays that hold a value per row
// are given from that row on).
// out [n, Kf] = S[r, k - k0] wk wp[r] exp(2 pi j (ph0[r] + (k - k0) ph1[r])) inside the band, 0 in the padding
extern "C" __global__ void ql_weight(const float2* S, const float* wk, const float* wp, const double* ph0,
                                     const double* ph1, int K, int k0, int Kf, int ref, int r0, int nr, float2* out) {
  const int kk = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (kk >= Kf || r >= nr) return;
  const int k = kk - k0;
  float2 v = make_float2(0.f, 0.f);
  if (k >= 0 && k < K) {
    const float2 s = S[(long)r * K + k];
    const float w = wk[k] * wp[r];
    if (ref) {
      double c = ph0[r] + k * ph1[r];
      c -= floor(c);
      float sn, cs;
      sincosf((float)(6.283185307179586 * c), &sn, &cs);
      v = make_float2(w * (s.x * cs - s.y * sn), w * (s.x * sn + s.y * cs));
    } else v = make_float2(w * s.x, w * s.y);
  }
  out[(long)r * Kf + kk] = v;
}
// out [n, Nr] = the profile's bins about bin 0 (cut out of Kf, or set among zeros), times the taper
extern "C" __global__ void ql_rgate(const float2* prof, const float* tap, int Kf, int Nr, int r0, int nr, float2* out) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (m >= Nr || r >= nr) return;
  const int keep = min(Nr, Kf), pos = keep - keep / 2, neg = keep / 2;
  float2 v = make_float2(0.f, 0.f);
  if (m < pos) v = prof[(long)r * Kf + m];
  else if (m >= Nr - neg) v = prof[(long)r * Kf + (Kf - (Nr - m))];
  const float w = tap[m];
  out[(long)r * Nr + m] = make_float2(v.x * w, v.y * w);
}
// out [nkr, Pf], the pulses of each of the raster's range wavenumbers along a row (every transform runs along the
// rows of its array: along the columns CuPy copies the array turned, there and back): out[J, q] = A[p, J - jlo[p]]
// for the pulse p at place q - p0 (src[q - p0], or q - p0 itself; none where src is negative), where the pulse's own
// W rows hold that wavenumber (all of them, from 0, or its rows from jlo[p] on), and 0 elsewhere
// (quicklook_cpu.cpp's ql_transpose and ql_unshear)
extern "C" __global__ void ql_pulses(const float2* A, const int* jlo, const int* src, int has_jlo, int has_src, int W,
                                     int P, int Pp, int p0, int Pf, int r0, int nr, float2* out) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x, J = r0 + blockIdx.y;
  if (q >= Pf || blockIdx.y >= nr) return;
  float2 v = make_float2(0.f, 0.f);
  const int pl = q - p0;
  if (pl >= 0 && pl < Pp) {
    const int p = has_src ? src[pl] : pl;
    if (p >= 0 && p < P) {
      const int c = J - (has_jlo ? jlo[p] : 0);
      if (c >= 0 && c < W) v = A[(long)p * W + c];
    }
  }
  out[(long)blockIdx.y * Pf + q] = v;
}
// out [nkr, Na] = the bins about bin 0 of each row's transform along the pulses (prof [nkr, Pf]), times the scale and
// the gate at the bin's distance |bin| binm[J] in meters (quicklook_cpu.cpp's ql_azimuth_gate)
extern "C" __global__ void ql_agate(const float2* prof, const float* binm, int Pf, int Na, float edge, float guard,
                                    float scale, int r0, int nr, float2* out) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x, J = blockIdx.y;
  if (k >= Na || J >= nr) return;
  const int keep = min(Na, Pf), pos = keep - keep / 2, neg = keep / 2;
  const int n = k < Na - Na / 2 ? k : k - Na;
  float w = 0.f;
  if (n < pos && n >= -neg) {
    const float d = fabsf((float)n) * binm[J];
    w = d <= edge ? 1.f : (d >= edge + guard ? 0.f : 0.5f + 0.5f * cosf((d - edge) * (3.14159265f / guard)));
  }
  float2 v = make_float2(0.f, 0.f);
  if (w != 0.f) {
    const float2 p = prof[(long)J * Pf + (n >= 0 ? n : Pf + n)];
    v = make_float2(p.x * w * scale, p.y * w * scale);
  }
  out[(long)J * Na + k] = v;
}
// out [m, j] = row m of A [M, R] (complex: the pulses of one range wavenumber) at the position q(t) for
// t = (ka0 + j dka) inv_kr[m]: q by linear interpolation in qtab [nt] (quicklook_cpu.cpp's ql_resample_az)
extern "C" __global__ void ql_resample_az(const float2* A, int R, const double* inv_kr, double ka0, double dka,
                                          const double* qtab, int nt, double t0, double inv_dt, int n_out,
                                          const float* tab, int r0, int nr, float2* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (j >= n_out || m >= nr) return;
  float2 acc = make_float2(0.f, 0.f);
  const double u = ((ka0 + j * dka) * inv_kr[m] - t0) * inv_dt;
  if (u >= 0.0 && u < nt - 1) {
    const int k = (int)u;
    const double pos = qtab[k] + (u - k) * (qtab[k + 1] - qtab[k]);
    if (pos > -0.5 && pos < R - 0.5) {
      const double fl = floor(pos);
      const int i0 = (int)fl - HALF_T + 1;
      const float fr = (float)(pos - fl);
      const float2* row = A + (long)m * R;
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
        const float2 v = row[min(max(i0 + t, 0), R - 1)];
        acc.x = fmaf(w[t] * inv, v.x, acc.x); acc.y = fmaf(w[t] * inv, v.y, acc.y);
      }
    }
  }
  out[(long)m * n_out + j] = acc;
}
// out [nka, Ny], the raster's azimuth wavenumbers down the rows and its range wavenumbers along them in the order a
// transform of Ny takes: out[a, c] = the sum of B[J, a] over the rows J = (c + hy) mod Ny, + Ny, ... (the rows of a
// sheared raster that meet in one bin), times (-1)^(J - hy + a - sx0) when signs (the padded transforms' outputs
// then count from the image's center)
extern "C" __global__ void ql_place(const float2* B, int nkr, int nka, int hy, int Ny, int signs, int sx0, int r0,
                                    int nr, float2* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x, a = r0 + blockIdx.y;
  if (c >= Ny || blockIdx.y >= nr) return;
  float2 v = make_float2(0.f, 0.f);
  for (int J = (c + hy) % Ny; J < nkr; J += Ny) {
    const float2 b = B[(long)J * nka + a];
    if (signs && ((J - hy + a - sx0) & 1)) { v.x -= b.x; v.y -= b.y; } else { v.x += b.x; v.y += b.y; }
  }
  out[(long)blockIdx.y * Ny + c] = v;
}
// out [rows, Nx] = range bins (rb0 + r) mod Ny of F [nka, Ny] (the range transform done), each along a row with
// azimuth wavenumber a at (a - hx) mod Nx and zeros between: what the azimuth transform of those range bins takes
extern "C" __global__ void ql_aplace(const float2* F, int nka, int Ny, int hx, int Nx, int rb0, int r0, int nr,
                                     float2* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (x >= Nx || r >= nr) return;
  const int a = (x + hx) % Nx;
  out[(long)r * Nx + x] = a < nka ? F[(long)a * Ny + (rb0 + r) % Ny] : make_float2(0.f, 0.f);
}
// the uncorrected picture [ny, nx] out of its rows' azimuth transforms T [ny, nfx]: pixel x from output
// (x - nx / 2) mod nfx, with its carriers, or its amplitude (detect)
extern "C" __global__ void ql_cut(const float2* T, const float2* cy, const float2* cx, int nfx, int nx, int detect,
                                  int r0, int nr, float* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
  if (x >= nx || y >= nr) return;
  const float2 t = T[(long)y * nfx + (x - nx / 2 + nfx) % nfx];
  const long o = (long)y * nx + x;
  if (detect) { out[o] = sqrtf(t.x * t.x + t.y * t.y); return; }
  const float2 a = cy[y], b = cx[x];
  const float cr = a.x * b.x - a.y * b.y, ci = a.x * b.y + a.y * b.x;
  out[2 * o] = t.x * cr - t.y * ci; out[2 * o + 1] = t.x * ci + t.y * cr;
}
"""
_CUDA = {}


def _cuda(npow=10, deg=3):
    """The CUDA kernels and tables of the quick look, compiled and loaded once a process (and once more for a
    distortion polynomial of another length or degree than the usual): dict of the kernels by name and the tables
    'kaiser' and 'lanczos'."""
    key = ('module', npow, deg)
    if key not in _CUDA:
        import cupy as cp
        from . import pfa2
        mod = cp.RawModule(code=_CUDA_SRC, options=(f'-DTAPS={TAPS}', f'-DTABN={pfa2.TABN}', f'-DLTAPS={LANCZOS}',
                                                    f'-DNPOW={npow}', f'-DDEG={deg}', f'-DQTAB={PHASES}'))
        k = {n: mod.get_function('ql_' + n) for n in ('weight', 'rgate', 'pulses', 'agate', 'resample_az', 'place',
                                                       'aplace', 'cut', 'gather')}
        if 'kaiser' not in _CUDA:
            _CUDA['kaiser'] = cp.asarray(pfa2.kernel_table(TAPS, KAISER))
            _CUDA['lanczos'] = cp.asarray(_polyphase('lanczos', LANCZOS, PHASES))
            _CUDA['rows'] = pfa2._resample_kernels(TAPS)[0]
        k.update(kaiser=_CUDA['kaiser'], lanczos=_CUDA['lanczos'], rows=_CUDA['rows'])
        _CUDA[key] = k
    return _CUDA[key]


def _cuda_plan(g, correct, complex_image, budget, block):
    """Where the arrays of the CUDA path live for a device with `budget` bytes to give and `block` bytes for a
    stage's temporaries: dict(A, B, F, out: True on the device, False on the host, from where blocks go up and
    come down; kept: the names of those on the host). The stages hold at most two of the four at a time (the
    range stage's output A [P, nwin] and the raster B [nkr, nka]; B and its range transform F [nka, Ny]; F and the
    picture), so the largest go to the host first until what is left fits."""
    P, nwin, nkr, nka, nx, ny = g['P'], g['nwin'], g['nkr'], g['nka'], g['nx'], g['ny']
    Ny = (g['ov'] if correct else 1) * g['nfy']
    size = dict(A=8 * P * nwin, B=8 * nkr * nka, F=8 * nka * Ny, out=(8 if complex_image else 4) * nx * ny)
    names = dict(A="the range stage's output", B='the raster', F="the raster's range transform", out='the picture')
    on = dict.fromkeys(size, True)

    def peak():
        s = {k: size[k] if on[k] else 0 for k in size}
        return max(s['A'] + s['B'], s['B'] + s['F'], s['F'] + s['out']) + 2 * block
    for k in sorted(size, key=lambda k: -size[k]):
        if peak() <= budget:
            break
        on[k] = False
    return dict(on, kept=[names[k] for k in size if not on[k]])


def _pfa_cuda(S, g, window, correct, complex_image, timing, phase=None, info=None):
    """The quick look's polar format on a GPU -> the image, range-major [ny, nx]: a CuPy array, or a NumPy array
    where the picture does not fit on the device beside the arrays that form it. S: the host history (it goes up in
    blocks of pulses through two page-locked buffers, each block weighted, gated and resampled while the next is
    copied) or a CuPy array. Every stage works in blocks sized from the device's free memory (_xp.cuda_budget), and
    of the four large arrays (_cuda_plan) those that do not fit there stay on the host, their blocks going up and
    coming down as the stages need them; info['stream'] then says which. phase: (ph0, ph1) [P] float64, a
    reference phase exp(2 pi j (ph0 + k ph1)) of pulse p's sample k that the device applies with the weights (the
    history is then as the reader left it). timing: the stages' seconds; with timing['detail'] a dict, the device
    is brought to completion after every step and the steps' seconds are added to it (the steps then do not overlap,
    so their sum is longer than the call)."""
    import ctypes
    import cupy as cp
    import cupyx.scipy.fft as cf
    from . import _xp, memory
    powers = g['dist']['powers']
    kern = _cuda(len(powers), max(a + b for a, b in powers))
    S = S.pop() if isinstance(S, list) else S
    P, K, Kf, Nr, k0 = g['P'], g['K'], g['Kf'], g['Nr'], g['k0']
    nkr, nka, nfy, nfx, nx, ny = g['nkr'], g['nka'], g['nfy'], g['nfx'], g['nx'], g['ny']
    i32, f32, f64, c64 = np.int32, np.float32, np.float64, np.complex64
    detail = timing.get('detail') if isinstance(timing.get('detail'), dict) else None
    sync = cp.cuda.get_current_stream().synchronize
    budget, blk = _xp.cuda_budget(), _xp.cuda_block()
    plan = _cuda_plan(g, correct, complex_image, budget, blk)
    if plan['kept'] and info is not None:
        info['stream'] = (f"the device has {budget / 1e9:.1f} GB to give: {', '.join(plan['kept'])} "
                          f"{'is' if len(plan['kept']) == 1 else 'are'} kept on the host and cross in blocks")
    def new(on, shape, dtype=c64):
        """An array on the device, or on the host in page-locked memory (pageable where that much cannot be locked):
        its blocks then cross several times faster."""
        if on:
            return cp.empty(shape, dtype)
        try:
            n = int(np.prod(shape))
            return np.frombuffer(memory._pinned(n * np.dtype(dtype).itemsize), dtype, n).reshape(shape)
        except Exception:
            return np.empty(shape, dtype)

    def up(hb):
        """A host block on the device, through page-locked memory."""
        try:
            pin = np.frombuffer(memory._pinned(hb.nbytes), hb.dtype, hb.size).reshape(hb.shape)
        except Exception:
            return cp.asarray(hb)

        def copy(sl):
            np.copyto(pin[sl], hb[sl])
        _blocks(copy, hb.shape[0], int(max(1, (1 << 18) // hb.shape[1])))
        d = cp.empty(hb.shape, hb.dtype)
        d.set(pin)
        return d
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    at = lambda a, row=0, col=0: ctypes.c_void_p(a.ctypes.data + row * a.strides[0] + col * a.strides[1])

    def run(name, width, rows, r0, args, out):
        """Kernel `name` over the rows of a block (at most 32,768 of them, r0 the first one's place in the whole)."""
        kern[name]((-(-width // 256), rows), (256,), args + (i32(r0), i32(rows), out))

    def rows_of(n, per_row, floor=1):
        return int(max(floor, min(n, 32768, blk // max(int(per_row), 1))))

    def lap(name, t0):
        sync()
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()

    def step(name, t0):
        if detail is None:
            return t0
        sync()
        detail[name] = detail.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()
    t0 = time.perf_counter()
    tab = kern['kaiser']
    wk = cp.asarray((g['wk'] * (_taylor(K) if window else 1.0)).astype(np.float32))
    wp = cp.asarray((g['wp'] * (_taylor(P) if window else 1.0)).astype(np.float32))
    tap_r = cp.asarray(_taper(Nr, g['Er'], g['gy'], g['bin_r']))
    alpha, beta = cp.asarray(g['alpha'].astype(np.float32)), cp.asarray(g['beta'].astype(np.float32))
    ref = phase is not None
    ph0, ph1 = (cp.asarray(np.ascontiguousarray(v, f64)) for v in (phase if ref else (np.zeros(1), np.zeros(1))))
    t0 = lap('kernels', t0)
    # 1. range, a block of pulses at a time: each pulse on its rows of the raster (all of them, or its own)
    nwin, jlo = g['nwin'], g['jlo']
    A = new(plan['A'], (P, nwin))
    on_host = isinstance(S, np.ndarray)
    nb = rows_of(P, 24 * max(Kf, Nr))
    if on_host:
        S = np.ascontiguousarray(S, c64)
        nb = int(max(1, min(nb, 1024, (128 << 20) // (8 * K))))
        bufs = [memory._pinned(nb * K * 8) for _ in range(2)]
        events = [None, None]
        stream = cp.cuda.get_current_stream()
        cb = int(max(1, (1 << 18) // K))
    for b, r in enumerate(range(0, P, nb)):
        n = min(nb, P - r)
        t1 = time.perf_counter()
        if on_host:
            if events[b % 2] is not None:
                events[b % 2].synchronize()                  # its last copy has left the buffer
            hb = np.frombuffer(bufs[b % 2], c64, nb * K).reshape(nb, K)[:n]

            def copy(sl, r=r, hb=hb):
                np.copyto(hb[sl], S[r + sl.start:r + sl.stop])
            _blocks(copy, n, cb)
            x = cp.empty((n, K), c64)
            x.set(hb, stream=stream)
            events[b % 2] = cp.cuda.Event()
            events[b % 2].record(stream)
        else:
            x = S[r:r + n]
        t1 = step('upload', t1)
        buf = cp.empty((n, Kf), c64)
        run('weight', Kf, n, 0, (x, wk, wp[r:r + n], ph0[r:r + n] if ref else ph0, ph1[r:r + n] if ref else ph1, i32(K),
                                 i32(k0), i32(Kf), i32(ref)), buf)
        del x
        t1 = step('weights and padding', t1)
        prof = cf.ifft(buf, axis=1, overwrite_x=True)
        del buf
        t1 = step('range transforms', t1)
        gate = cp.empty((n, Nr), c64)
        run('rgate', Nr, n, 0, (prof, tap_r, i32(Kf), i32(Nr)), gate)
        del prof
        t1 = step('range gate', t1)
        G = cf.fft(gate, axis=1, overwrite_x=True)
        del gate
        t1 = step('range transforms', t1)
        Ab = A[r:r + n] if plan['A'] else cp.empty((n, nwin), c64)
        kern['rows']((-(-nwin // 128), n), (128,), (G, i32(n), i32(Nr), alpha[r:r + n], beta[r:r + n], i32(nwin), tab, Ab))
        del G
        t1 = step('range resampling', t1)
        if not plan['A']:
            Ab.get(out=A[r:r + n])
            step('blocks to the host', t1)
        del Ab
    if on_host:
        sync()
        del bufs
    S = None
    t0 = lap('range', t0)
    # 2. azimuth, in blocks of range wavenumbers, the pulses of each along a row: every transform of this path runs
    # along the rows of its array, and the kernels that pass over an array turn it where the next transform needs
    # it turned. The transform along the pulses, the gate in meters with its cut and the scale, the transform back
    # on Na pulses, the read at the raster's azimuth wavenumbers
    Pf, Na, p0, Pp, place = g['Pf'], g['Na'], g['p0'], g['Pp'], g['place']
    src = None
    if place is not None:
        src = np.full(Pp, -1, np.int32)
        src[place] = np.arange(P, dtype=np.int32)
    dummy = cp.zeros(1, cp.int32)
    d_jlo, d_src = dummy if jlo is None else cp.asarray(jlo), dummy if src is None else cp.asarray(src)
    scale = f32(g['gain'] * (g['ov'] * nfy if correct else nfy) * Na / Pf)     # the pulses lie at p0 here: no phase
    binm, inv_kr, qtab = cp.asarray(g['bins_a'].astype(np.float32)), cp.asarray(1.0 / g['kr']), cp.asarray(g['qtab'])
    B = new(plan['B'], (nkr, nka))
    nJ = rows_of(nkr, 24 * max(Pf, Na))
    lib = None if plan['A'] and plan['B'] and plan['F'] else _lib()
    for J0 in range(0, nkr, nJ):
        n = min(nJ, nkr - J0)
        t1 = time.perf_counter()
        if plan['A']:
            pad = cp.empty((n, Pf), c64)
            run('pulses', Pf, n, J0, (A, d_jlo, d_src, i32(jlo is not None), i32(src is not None), i32(nwin), i32(P), i32(Pp),
                                      i32(p0), i32(Pf)), pad)
            t1 = step('azimuth layout', t1)
        else:                                         # the block's rows from the host's array, the host's kernels
            hb = np.empty((n, Pf), c64)
            hb[:, :p0] = 0
            if jlo is None and place is None:
                lib.ql_transpose(at(A, 0, J0), nwin, P, n, at(hb, 0, p0), Pf, None, 0)
            else:
                lib.ql_unshear(ptr(A), nwin, P, None if jlo is None else ptr(jlo), None if place is None else ptr(place),
                               Pp, J0, n, at(hb, 0, p0), Pf)
            hb[:, p0 + Pp:] = 0
            pad = up(hb)
            del hb
            t1 = step('blocks to the device', t1)
        prof = cf.fft(pad, axis=1, overwrite_x=True)
        del pad
        t1 = step('azimuth transforms', t1)
        gate = cp.empty((n, Na), c64)
        run('agate', Na, n, J0, (prof, binm[J0:J0 + n], i32(Pf), i32(Na), f32(g['Ex']), f32(g['gx']), scale), gate)
        del prof
        t1 = step('azimuth gate', t1)
        A2 = cf.ifft(gate, axis=1, overwrite_x=True)
        del gate
        t1 = step('azimuth transforms', t1)
        Bb = B[J0:J0 + n] if plan['B'] else cp.empty((n, nka), c64)
        run('resample_az', nka, n, J0, (A2, i32(Na), inv_kr[J0:J0 + n], f64(g['ka_lo']), f64(g['dka']), qtab,
                                        i32(len(g['qtab'])), f64(g['tg0']), f64(g['tg_inv']), i32(nka), tab), Bb)
        del A2
        t1 = step('azimuth resampling', t1)
        if not plan['B']:
            Bb.get(out=B[J0:J0 + n])
            step('blocks to the host', t1)
        del Bb
    del A
    t0 = lap('azimuth', t0)
    # 3. the raster's range transform, in blocks of azimuth wavenumbers (each along a row, its range wavenumbers in
    # the order the transform takes, the rows of a sheared raster that meet in one bin summed)
    ov = g['ov'] if correct else 1
    Ny, hy, hx = ov * nfy, (g['hy'] if correct else 0), (nka // 2 if correct else 0)
    F = new(plan['F'], (nka, Ny))
    na = rows_of(nka, 16 * Ny)
    if not plan['B'] and correct:
        sy = ((-1.0) ** (np.arange(nkr) - hy)).astype(np.float32)
        sx = ((-1.0) ** (np.arange(nka) - hx)).astype(np.float32)
    for a0 in range(0, nka, na):
        n = min(na, nka - a0)
        t1 = time.perf_counter()
        if plan['B']:
            Z = cp.empty((n, Ny), c64)
            run('place', Ny, n, a0, (B, i32(nkr), i32(nka), i32(hy), i32(Ny), i32(1 if correct else 0), i32(hx)), Z)
        else:
            M = np.ascontiguousarray(B[:, a0:a0 + n].T)
            if correct:
                M *= sy[None, :] * sx[a0:a0 + n, None]
            Z = up(_fold(M, Ny, hy))
            del M
            t1 = step('blocks to the device', t1)
        Fb = cf.ifft(Z, axis=1, overwrite_x=True)
        del Z
        t1 = step('image transforms', t1)
        if plan['F']:
            F[a0:a0 + n] = Fb
        else:
            Fb.get(out=F[a0:a0 + n])
            step('blocks to the host', t1)
        del Fb
    del B
    t0 = lap('range transform', t0)
    # 4. the azimuth transform and the picture, in blocks of range pixels: uncorrected, the picture's range bins
    # transformed and cut; corrected, the padded transform of the bins a block's pixels read, and the read at their
    # apparent positions
    out = new(plan['out'], (ny, nx), c64 if complex_image else np.float32)
    Nx = ov * nfx
    if correct:
        dist = g['dist']
        d_cx, d_cy = cp.asarray(np.asarray(dist['cx'], f64)), cp.asarray(np.asarray(dist['cy'], f64))
        d_pw = cp.asarray(np.asarray(powers, np.int32).ravel())
        kr_c, ka_c = g['kr_min'] + hy * g['dkr'], g['ka_lo'] + hx * g['dka']
        shear = float(g['shear'] * g['spy'] / g['spx'])
        msy = int(math.ceil(g['shift'][1] / g['spy'])) * ov + LANCZOS + 2
        bw = rows_of(ny, 16 * Nx * ov, 16)
        blocks = [(jb, min(bw, ny - jb)) for jb in range(0, ny, bw)]
    else:
        y, x = (np.arange(ny) - ny / 2.0) * g['spy'], (np.arange(nx) - nx / 2.0) * g['spx']
        cy = cp.asarray(np.exp(1j * g['kr_min'] * y).astype(c64))
        cx = cp.asarray(np.exp(-1j * g['ka_lo'] * x).astype(c64))
        bw = rows_of(ny, 16 * Nx, 16)                 # the blocks end where the bins wrap: at the picture's center
        blocks = [(jb, min(bw, e - jb)) for s, e in ((0, ny // 2), (ny // 2, ny)) for jb in range(s, e, bw)]
    for jb, n in blocks:
        t1 = time.perf_counter()
        if correct:
            rb0 = max(0, Ny // 2 + ov * (jb - ny // 2) - msy)
            span = min(Ny, Ny // 2 + ov * (jb + n - ny // 2) + msy) - rb0
        else:
            rb0, span = (nfy - ny // 2 + jb) % nfy, n
        if plan['F']:
            Z = cp.empty((span, Nx), c64)
            run('aplace', Nx, span, 0, (F, i32(nka), i32(Ny), i32(hx), i32(Nx), i32(rb0)), Z)
        else:
            hb = np.empty((span, Nx), c64)
            lib.ql_transpose(at(F, hx, rb0), Ny, nka - hx, span, ptr(hb), Nx, None, 0)
            if hx:
                lib.ql_transpose(at(F, 0, rb0), Ny, hx, span, at(hb, 0, Nx - hx), Nx, None, 0)
            hb[:, nka - hx:Nx - hx] = 0
            Z = up(hb)
            del hb
            t1 = step('blocks to the device', t1)
        base = cf.fft(Z, axis=1, overwrite_x=True)
        del Z
        t1 = step('image transforms', t1)
        ob = out[jb:jb + n] if plan['out'] else cp.empty((n, nx), out.dtype)
        if correct:
            kern['gather']((-(-nx // 128), n), (128,),
                           (base, i32(span), i32(Nx), i32(Ny), i32(rb0), i32(n), i32(jb), i32(nx), i32(ny), f64(g['spx']),
                            f64(g['spy']), f64(ov), f64(dist['sx']), f64(dist['sy']), d_cx, d_cy, d_pw, f64(kr_c), f64(ka_c),
                            f64(shear), kern['lanczos'], i32(0 if complex_image else 1), np.float32(1.0), ob))
        else:
            run('cut', nx, n, jb, (base, cy[jb:jb + n], cx, i32(Nx), i32(nx), i32(0 if complex_image else 1)), ob)
        del base
        t1 = step('read at the apparent positions' if correct else 'picture', t1)
        if not plan['out']:
            ob.get(out=out[jb:jb + n])
            step('blocks to the host', t1)
        del ob
    del F
    lap('correction' if correct else 'image', t0)
    return out


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
    standard library alone. Blocks of rows are deflated on all cores, each an independent run of the one zlib stream
    and an IDAT chunk of its own, with its checksums made beside it, so a 16 megapixel picture is written in a few
    hundredths of a second. level: zlib's, 0 to 9 (speckle hardly compresses, and 1 is as small as 9 on it)."""
    import struct
    import zlib
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)) or a.size == 0:
        raise ValueError(f'write_png takes a uint8 image [rows, cols] or [rows, cols, 3 or 4], got {a.dtype} {a.shape}')
    rows, cols = a.shape[:2]
    a = a.reshape(rows, -1)
    line = 1 + a.shape[1]                                       # a row and its filter byte (type 0: none)
    per = int(max(1, min(rows, max(1 << 16, min(4 << 20, rows * line // (4 * _workers()))) // line)))
    starts = list(range(0, rows, per))
    parts = [None] * len(starts)

    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))

    def deflate(sl):
        i = sl.start
        raw = np.zeros((min(per, rows - starts[i]), line), np.uint8)
        raw[:, 1:] = a[starts[i]:starts[i] + per]
        co = zlib.compressobj(level, zlib.DEFLATED, -15)
        last = i == len(starts) - 1
        body = co.compress(raw) + co.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)
        parts[i] = [(b'\x78\x01' if i == 0 else b'') + body, zlib.adler32(raw), raw.size]
        if not last:
            parts[i][0] = chunk(b'IDAT', parts[i][0])
    _blocks(deflate, len(starts), 1)
    # the stream's Adler-32 from the blocks' own (zlib's adler32_combine)
    adler = 1
    for _, a2, n2 in parts:
        rem = n2 % 65521
        s1 = adler & 0xffff
        s2 = (rem * s1) % 65521
        s1 += (a2 & 0xffff) + 65521 - 1
        s2 += ((adler >> 16) & 0xffff) + ((a2 >> 16) & 0xffff) + 65521 - rem
        adler = (s1 % 65521) | ((s2 % 65521) << 16)
    color = {1: 0, 3: 2, 4: 6}[1 if np.ndim(image) == 2 else np.shape(image)[2]]
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', cols, rows, 8, color, 0, 0, 0)))
        for body, _, _ in parts[:-1]:
            fh.write(body)
        fh.write(chunk(b'IDAT', parts[-1][0] + struct.pack('>I', adler)))
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

def _jax_algorithm():
    """What forms a quick look on the jax and tpu backends: 'pfa', polar format's stages on the device
    (quick_jax.py), or, with FASTSAR_QUICKLOOK_JAX=ffbp, the factorized former on the same cut and grid."""
    v = os.environ.get('FASTSAR_QUICKLOOK_JAX', '').strip().lower() or 'pfa'
    if v not in ('pfa', 'ffbp'):
        raise ValueError(f"FASTSAR_QUICKLOOK_JAX must be 'pfa' or 'ffbp', got {v!r}")
    return v


def _backend(backend):
    """A backend named by the caller, checked against this machine ('auto' is resolved by _auto, once the cut is
    known)."""
    from . import api
    b = api._backend(backend)
    return 'cpu' if b == 'metal' else b


def _auto(samples):
    """backend='auto' for a cut history of `samples` samples: the device that forms its picture soonest, by the
    AUTO_ thresholds above. No device is asked for below them: importing CuPy or JAX takes longer than a small
    quick look. JAX is asked for a TPU only where one may be present (JAX already imported, the libtpu package
    installed, or TPU_LIBRARY_PATH set)."""
    import importlib.util
    import sys
    if samples >= AUTO_TPU_SAMPLES and ('jax' in sys.modules or os.environ.get('TPU_LIBRARY_PATH')
                                        or importlib.util.find_spec('libtpu') is not None):
        from . import api
        try:
            if 'tpu' in api.available_backends():
                return 'tpu'
        except Exception:
            pass
    if samples >= (AUTO_CUDA_SAMPLES if _CUDA else AUTO_CUDA_FIRST_SAMPLES):
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
        8-bit picture (standard library only); .nitf or .ntf, the complex image as a ground-plane SICD
        (products.write_sicd). None: nothing is written.
    size: pixels along the longer side of the picture, which is the tasked area's bounding rectangle along and
        across the ground range direction (default 2048). resolution: the ground resolution in meters in place of
        size (the reciprocal of the band used on both axes; the pixel is 0.96 of it), or 'full' for the finest the
        collection holds on both axes. A request finer than that gets that, with a note.
    correct: remove polar format's displacement of scatterers (the default). False is the fastest picture: the
        displacement, a smooth field that grows with the square of the distance from the center, stays, and
        displacement_px says how large it is at the corners.
    remap: the display remap, a name of fastsar.remap.REMAPS other than 'linear' or a function (sidd.to_display's).
    backend: 'auto', 'cpu', 'cuda', 'jax' or 'tpu', each with polar format's stages written for it (the jax and
        tpu backends form by the factorized former instead with FASTSAR_QUICKLOOK_JAX=ffbp, where correct does not
        apply). 'auto'
        takes the device that forms the cut soonest, by its size: the CPU for a small one (no device is started),
        a GPU or a TPU from the sizes of AUTO_CUDA_SAMPLES, AUTO_CUDA_FIRST_SAMPLES and AUTO_TPU_SAMPLES on.
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
    origin + i spx e1 + j spy e2 in the local frame of meta (the grid of form_cphd(axes='los')); plane 'ground', axes
    'los',
    mode 'spotlight', center; display_origin, row, col, row_ss, col_ss, transform: the display's pixel (r, c) at
    display_origin + r row_ss row + c col_ss col (sidd.shadows_down's entries); corners: latitude and longitude
    [4, 2] of the display's corner pixels on the scene's plane, from the first row and column clockwise; pixel, the
    pixel on the ground (m), resolution (m) and finest, the finest resolution the collection holds; band (first
    and last frequency used, Hz), bandwidth (cycles/m along e1 and e2), window, graze (rad), pulses (of those read)
    and cut:
    dict(pulses, of_pulses, samples, of_samples: the file's vectors and samples read; signal_bytes, of_bytes: their
    size and the signal array's); algorithm ('pfa' or 'ffbp'), backend, correct, displacement_px (polar format's
    largest displacement along e1 and e2 over the picture, in pixels; removed when correct), corner_error (polar
    format's own defocus at the worst corner: the phase its planar wavefronts leave there beyond the displacement,
    rad, and the error that makes against exact backprojection, dB; None for ffbp), looks (sub-apertures,
    sub-bands), remap, meta
    (read_cphd's), notes, timing (seconds by stage, and total), and path when a file was written). A moving-beam
    collection raises ValueError; process_cphd forms those."""
    from . import _xp, sidd
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
    backend = backend if backend == 'auto' else _backend(backend)
    window = weighting == 'taylor'
    col = _collection(cphd, size, resolution, extent, channel, reader_options, looks, backend)
    timing, notes, plan, meta, backend = col['timing'], col['notes'], col['plan'], col['meta'], col['backend']
    nx, ny, spx, spy, e1, e2, c = col['nx'], col['ny'], col['spx'], col['spy'], col['e1'], col['e2'], col['center']
    S, ant, f0, df, phase = col['S'], col['ant'], col['fmin'], col['df'], col['phase']
    P, K = S.shape
    (pa, pb), (k0, k1) = meta['pulses'], meta['samples']
    isz = meta['signal_bytes'] // (P * K) if P * K else 0
    cut = dict(pulses=(pa, pb), of_pulses=plan['P'], samples=(k0, k1), of_samples=plan['K'],
               signal_bytes=int(meta['signal_bytes']), of_bytes=int(plan['P'] * plan['K'] * isz))
    notes.append(f"quick look: a ground resolution of {plan['resolution']:.3f} m in pixels of {plan['pixel']:.3f} m, from "
                 f"{pb - pa} of {plan['P']} pulses and {k1 - k0} of {plan['K']} samples "
                 f"({cut['signal_bytes'] / 1e6:.1f} MB of the {cut['of_bytes'] / 1e9:.2f} GB signal array)")
    notes.append(f'ground-plane grid along the mid-aperture line of sight, {nx} by {ny} pixels of {spx:.3f} m (0.96 of '
                 f"the reciprocal of the band on each axis; grazing angle {plan['graze']:.1f} degrees)")
    la, lr = plan['looks']
    Pl, Kl = P // la, K // lr                                            # pulses and samples of a look

    def look(i, j):
        """The collection of look (i, j): its pulses and samples, on their own."""
        if (la, lr) == (1, 1):
            return col
        rows, cols = slice(i * Pl, (i + 1) * Pl), slice(j * Kl, (j + 1) * Kl)
        ph = None if phase is None else (phase[0][rows] + j * Kl * phase[1][rows], phase[1][rows])
        return dict(col, S=np.ascontiguousarray(S[rows, cols]), ant=ant[rows], fmin=f0 + j * Kl * df, phase=ph)
    info, shift, acc, cimg, defocus = {}, None, None, None, None
    xp = np
    for i in range(la):
        for j in range(lr):
            sub = look(i, j)
            if backend in ('jax', 'tpu') and _jax_algorithm() == 'ffbp':
                img = _ffbp(sub, backend, window, timing)[0]
                cimg = img if want else None
                rm = np.ascontiguousarray(img.T)
                complex_rm = True
            else:
                box = [sub.pop('S')]                                     # handed over: freed once it is gated
                if (la, lr) == (1, 1):
                    S = None
                out = _pfa(box, sub['ant'], sub['fmin'], df, nx, ny, spx, spy, e1, e2, correct=bool(correct),
                           window=window, backend=backend, complex_image=want, timing=timing, info=info,
                           phase=sub['phase'])
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
    if backend in ('jax', 'tpu') and _jax_algorithm() == 'ffbp':
        algorithm, corrected = 'ffbp', True
        notes.append(f"formed by factorized backprojection on the '{backend}' backend"
                     + (" (precision 'single-pass')" if backend == 'tpu' else '') + ' (FASTSAR_QUICKLOOK_JAX=ffbp); '
                     'correct does not apply')
    else:
        if info.get('stream'):
            notes.append(info['stream'])
        algorithm, corrected, shift = 'pfa', bool(correct), tuple(float(v) for v in info['shift_px'])
        notes.append(f'polar format: its displacement of scatterers, up to {shift[0]:.1f} by {shift[1]:.1f} pixels '
                     '(azimuth by range) over the picture, ' + ('removed' if corrected else 'left in (correct=False)'))
        defocus = _corner_error(ant[:Pl], f0, df, Kl, nx, ny, spx, spy, e1, e2, window)
        if defocus[1] > -30.0:
            notes.append(f"polar format's planar wavefronts leave up to {defocus[0]:.2f} rad of phase over the aperture "
                         f"at the picture's corners, beyond the displacement: at this resolution the corners are "
                         f'defocused, about {defocus[1]:.0f} dB from exact backprojection there. form_cphd forms the '
                         'whole area exactly')
    del S, col
    # the picture: remapped where the image is, laid out shadows down
    t0 = time.perf_counter()
    origin = c - 0.5 * nx * spx * e1 - 0.5 * ny * spy * e2
    grid = dict(image=np.broadcast_to(np.float32(0), (nx, ny)), origin=origin, e1=e1, e2=e2, spx=spx, spy=spy, meta=meta)
    lay = sidd.shadows_down(grid)
    tr = lay['transform']
    d = sidd._display(dict(image=amp), remap, None, {})[0]             # uint8, range-major, where the image is
    d = d if tr['transpose'] else d.T
    d = d[::-1] if tr['flip_rows'] else d
    d = d[:, ::-1] if tr['flip_cols'] else d
    if _xp.kind(d) == 'cupy':                                          # laid out there, brought over once
        d = xp.ascontiguousarray(d)
        xp.cuda.get_current_stream().synchronize()
        timing['remap'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        display = _host(d)
    else:
        display = np.ascontiguousarray(_xp.host(d))
    del amp, d
    rows, cols = display.shape
    rc = np.array([[0, 0], [0, cols - 1], [rows - 1, cols - 1], [rows - 1, 0]], np.float64)
    q = lay['origin'] + rc[:, :1] * lay['row_ss'] * lay['row'] + rc[:, 1:] * lay['col_ss'] * lay['col']
    lat, lon, _ = io.ecf_to_geodetic(io.local_to_ecf(q, meta))
    timing['display'] = time.perf_counter() - t0
    u = ant / np.linalg.norm(ant, axis=1, keepdims=True)
    ux, urm = u @ e1, float(-(u[P // 2] @ e2))
    fc = f0 + 0.5 * (K - 1) * df
    product = dict(
        display=display, image=cimg, origin=origin, e1=e1, e2=e2, spx=float(spx), spy=float(spy), plane='ground',
        axes='los', mode='spotlight', center=c, display_origin=lay['origin'], row=lay['row'], col=lay['col'],
        row_ss=lay['row_ss'], col_ss=lay['col_ss'], transform=tr, corners=np.stack([lat, lon], 1),
        pixel=float(plan['pixel']), resolution=float(plan['resolution']), finest=float(plan['finest']),
        band=(f0, f0 + (K - 1) * df), bandwidth=(2 * fc / C * float(abs(ux[-1] - ux[0])) / la, 2 * K * df / C * urm / lr),
        window=('taylor', 'taylor') if window else None, pulses=(0, P), cut=cut, algorithm=algorithm, backend=backend,
        correct=corrected, displacement_px=shift, corner_error=defocus, looks=(la, lr), remap=remap,
        graze=float(np.radians(plan['graze'])),
        meta=meta,
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
