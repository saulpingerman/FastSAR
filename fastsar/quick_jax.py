"""The quick look's polar format on a JAX device (a TPU, or whatever JAX runs on): the stages of fastsar.quick in
float32, with every complex array as two real planes outside the transforms.

The device is a block processor. Each stage is one compiled program that takes a block of rows of one shape (the
last block is moved back to end at the last row, so a stage compiles once) and the blocks are sized from the
device's memory (_xp.jax_block); the four large arrays (the range stage's output, the raster, its range transform
and the picture) live on the host and their blocks cross as the stages need them, so a full image forms on a
device that holds a small part of it. Which of them could stay on a device with room is a question of its transfer
rates, which tests/bench_tpu_blocks.py measures.

Positions. A resampling position reaches tens of thousands of samples and must hold to a part in 2048 of a sample,
more than float32 carries, and a TPU has no fast float64. The host therefore forms the positions in float64 at the
first, middle and last outputs of blocks of BLOCK outputs and sends, for each row and block, the integer part at
the block's first output (int32) and the remainder, the step and the curvature (float32); the device forms the
block's positions from those. Along a row
the range positions are linear and the azimuth positions and the apparent positions of the final read are smooth,
so a parabola over 32 outputs holds them to a millionth of a sample. The carrier of the complex image goes the
same way, in cycles.

The reads are indexed loads of each tap (jnp.take_along_axis), with the weights a row of the host kernels' own
polyphase tables: right on every device and fast on a CPU or a GPU. A TPU does indexed loads slowly; its reads
(window rows, dense products, separable passes for the final read) are to be chosen from its measured block times
and set in _read and _read2 alone, which the stages call.
"""
import math
import time

import numpy as np

BLOCK = 32                    # outputs whose positions are one parabola
_PROGRAMS = {}


def _program(key, make):
    """The compiled program of a stage for its static sizes, kept between calls (the newest 32)."""
    if key not in _PROGRAMS:
        import jax
        if len(_PROGRAMS) >= 32:
            _PROGRAMS.pop(next(iter(_PROGRAMS)))
        _PROGRAMS[key] = jax.jit(make())
    return _PROGRAMS[key]


def _starts(n, per):
    """Blocks of `per` rows over n rows, the last moved back to end at the last row: (per, starts)."""
    per = int(max(1, min(per, n)))
    starts = list(range(0, n - per + 1, per))
    if starts[-1] + per < n:
        starts.append(n - per)
    return per, starts


def _knots(pos):
    """Positions at the blocks' first outputs, middles and ends, [R, 2 nblk + 1] (float64) -> each block's parabola
    through its three: the integer part at its first output (int32 [R, nblk]), the remainder there, the step and
    the curvature per output (float32): output i of the block lies at base + lo + i s + i^2 c."""
    p0, pm, pe = pos[:, 0:-1:2], pos[:, 1::2], pos[:, 2::2]
    c = 2.0 * (pe - 2.0 * pm + p0) / BLOCK ** 2
    base = np.floor(p0)
    return (base.astype(np.int32), (p0 - base).astype(np.float32), ((pe - p0) / BLOCK - c * BLOCK).astype(np.float32),
            c.astype(np.float32))


def _positions(jnp, base, lo, s, c):
    """A block's positions as the sample below (int32) and the fraction above it, [R, nblk * BLOCK]."""
    i = jnp.arange(BLOCK, dtype=jnp.float32)
    f = lo[..., None] + i * s[..., None] + (i * i) * c[..., None]
    fl = jnp.floor(f)
    R = base.shape[0]
    return (base[..., None] + fl.astype(jnp.int32)).reshape(R, -1), (f - fl).reshape(R, -1)


def _read(jnp, xr, xi, ib, fr, tab, phases):
    """Rows of x [R, n] (two planes) at the positions ib + fr [R, m], with the taps of tab [phases + 1, taps] (the row
    of the fraction's nearest phase); a tap outside the row counts as zero and the rest are normalised again; zero
    where the position lies outside (-0.5, n - 0.5): quicklook_cpu.cpp's ql_resample_rows."""
    n, taps = xr.shape[1], tab.shape[1]
    w = jnp.take(tab, jnp.rint(fr * phases).astype(jnp.int32), axis=0)                    # [R, m, taps]
    idx = ib[..., None] - (taps // 2 - 1) + jnp.arange(taps, dtype=jnp.int32)
    w = jnp.where((idx >= 0) & (idx < n), w, 0.0)
    w = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-12)
    idx = jnp.clip(idx, 0, n - 1).reshape(idx.shape[0], -1)
    inside = ((ib >= 0) | ((ib == -1) & (fr > 0.5))) & ((ib < n - 1) | ((ib == n - 1) & (fr < 0.5)))
    get = lambda p: jnp.where(inside, (jnp.take_along_axis(p, idx, axis=1).reshape(w.shape) * w).sum(-1), 0.0)
    return get(xr), get(xi)


def _read2(jnp, br, bi, ix, fx, iy, fy, shear, tab, phases):
    """The baseband block (two planes [span, Nx]) at the positions (iy + fy, ix + fx) [n, m] with taps x taps taps
    of tab, each row of taps at the column where the line of the raster's shear through the position meets it:
    quicklook_cpu.cpp's ql_gather."""
    span, Nx = br.shape
    taps = tab.shape[1]
    half = taps // 2
    wy = jnp.take(tab, jnp.rint(fy * phases).astype(jnp.int32), axis=0)                    # [n, m, taps]
    fr_, fi_ = br.reshape(-1), bi.reshape(-1)
    ar = ai = 0.0
    for a in range(taps):
        t = fx + shear * ((a - half + 1) - fy)
        fl = jnp.floor(t)
        wx = jnp.take(tab, jnp.rint((t - fl) * phases).astype(jnp.int32), axis=0)
        row = jnp.clip(iy - half + 1 + a, 0, span - 1) * Nx
        cols = jnp.clip((ix + fl.astype(jnp.int32))[..., None] - half + 1 + jnp.arange(taps, dtype=jnp.int32), 0, Nx - 1)
        at = row[..., None] + cols
        ar = ar + wy[..., a] * (jnp.take(fr_, at) * wx).sum(-1)
        ai = ai + wy[..., a] * (jnp.take(fi_, at) * wx).sum(-1)
    return ar, ai


def pfa(S, g, window, correct, complex_image, timing, phase=None, info=None):
    """fastsar.quick's polar format on JAX's device -> the image on the host, range-major [ny, nx] (complex64, or
    its amplitude, float32). S [P, K] complex64 on the host; g: quick._geometry's; phase: (ph0, ph1) [P] in cycles,
    a reference phase the device applies with the weights."""
    import ctypes
    import jax
    import jax.numpy as jnp
    from jax import lax
    from . import _xp
    from . import quick as Q
    f32, i32, c64 = np.float32, np.int32, np.complex64
    lib = Q._lib()
    S = S.pop() if isinstance(S, list) else S
    S = np.ascontiguousarray(S, c64)
    P, K, Kf, Nr, k0 = g['P'], g['K'], g['Kf'], g['Nr'], g['k0']
    nkr, nka, nfy, nfx, nx, ny = g['nkr'], g['nka'], g['nfy'], g['nfx'], g['nx'], g['ny']
    nwin, jlo, Pf, Na, p0, Pp, place = g['nwin'], g['jlo'], g['Pf'], g['Na'], g['p0'], g['Pp'], g['place']
    blk = _xp.jax_block()
    PH = Q.PHASES
    tab = jnp.asarray(Q._polyphase('kaiser', Q.TAPS, PH))
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    at = lambda a, row=0, col=0: ctypes.c_void_p(a.ctypes.data + row * a.strides[0] + col * a.strides[1])
    planes = lambda a: (jnp.asarray(np.ascontiguousarray(a.real)), jnp.asarray(np.ascontiguousarray(a.imag)))

    def to_host(dst, re, im):
        v = dst.view(np.float32).reshape(dst.shape + (2,))
        v[..., 0], v[..., 1] = np.asarray(re), np.asarray(im)
    if info is not None:
        info['stream'] = ("the range stage's output, the raster, its range transform and the picture are kept on the "
                          'host and cross to the JAX device in blocks')
    t0 = time.perf_counter()
    # 1. range: a block of pulses weighted, padded, gated in range and read at the raster's range wavenumbers
    wk = jnp.asarray((g['wk'] * (Q._taylor(K) if window else 1.0)).astype(f32))
    wp = (g['wp'] * (Q._taylor(P) if window else 1.0)).astype(f32)
    tap_r = jnp.asarray(Q._taper(Nr, g['Er'], g['gy'], g['bin_r']))
    nblk = -(-nwin // BLOCK)
    keep = min(Nr, Kf)
    pos_, neg_ = keep - keep // 2, keep // 2
    ref = phase is not None

    def make1():
        def run(sr, si, wk, tap_r, tab, wpb, a0, b0, base, lo, s, c2):
            if ref:
                c = a0[:, None] + jnp.arange(K, dtype=jnp.float32) * b0[:, None]
                c = (c - jnp.floor(c)) * (2.0 * math.pi)
                cs, sn = jnp.cos(c), jnp.sin(c)
                sr, si = sr * cs - si * sn, sr * sn + si * cs
            w = wk[None, :] * wpb[:, None]
            z = jnp.pad(lax.complex(sr * w, si * w), ((0, 0), (k0, Kf - K - k0)))
            prof = jnp.fft.ifft(z, axis=1)
            gate = jnp.concatenate([prof[:, :pos_], jnp.zeros((prof.shape[0], Nr - keep), prof.dtype), prof[:, Kf - neg_:]], 1) * tap_r
            G = jnp.fft.fft(gate, axis=1)
            ib, fr = _positions(jnp, base, lo, s, c2)
            return _read(jnp, jnp.real(G), jnp.imag(G), ib, fr, tab, PH)
        return run
    nb, starts = _starts(P, blk // (24 * max(Kf, Nr) + 300 * nblk * BLOCK))
    step1 = _program(('range', nb, K, k0, Kf, Nr, nblk, ref), make1)
    j = np.arange(2 * nblk + 1, dtype=np.float64) * (BLOCK / 2)
    A = np.empty((P, nwin), c64)
    ph0, ph1 = (np.asarray(v, np.float64) for v in (phase if ref else (np.zeros(P), np.zeros(P))))
    for r in starts:
        sl = slice(r, r + nb)
        kn = _knots(g['alpha'][sl, None] + j[None, :] * g['beta'][sl, None])
        o = step1(*planes(S[sl]), wk, tap_r, tab, jnp.asarray(wp[sl]), jnp.asarray(ph0[sl].astype(f32)),
                  jnp.asarray(ph1[sl].astype(f32)), *(jnp.asarray(v) for v in kn))
        to_host(A[sl], o[0][:, :nwin], o[1][:, :nwin])
    del S
    timing['range'] = timing.get('range', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # 2. azimuth: a block of range wavenumbers, the pulses of each along a row, gated in azimuth (in meters, for
    # each wavenumber's own bins) and read at the raster's azimuth wavenumbers at the pulses' own spacing
    Nfft = (g['ov'] if correct else 1) * nfy
    scale = jnp.float32(g['gain'] * Nfft * Na / Pf)
    k = np.arange(Na)
    nbin = np.where(k < Na - Na // 2, k, k - Na)
    keep = min(Na, Pf)
    pa, na_ = keep - keep // 2, keep // 2
    dist_bins = jnp.asarray(np.abs(nbin).astype(f32))
    live = jnp.asarray(((nbin < pa) & (nbin >= -na_)).astype(f32))
    edge, guard = jnp.float32(g['Ex']), jnp.float32(g['gx'])
    nblk2 = -(-nka // BLOCK)

    def make2():
        def run(pr, pi, tab, dist_bins, live, edge, guard, scale, binm, base, lo, s, c2):
            prof = jnp.fft.fft(lax.complex(pr, pi), axis=1)
            cut = jnp.concatenate([prof[:, :pa], jnp.zeros((prof.shape[0], Na - keep), prof.dtype), prof[:, Pf - na_:]], 1)
            d = dist_bins[None, :] * binm[:, None]
            w = jnp.where(d <= edge, 1.0, jnp.where(d >= edge + guard, 0.0, 0.5 + 0.5 * jnp.cos((d - edge) * (math.pi / guard))))
            A2 = jnp.fft.ifft(cut * (w * live[None, :] * scale), axis=1)
            ib, fr = _positions(jnp, base, lo, s, c2)
            return _read(jnp, jnp.real(A2), jnp.imag(A2), ib, fr, tab, PH)
        return run
    nJ, starts = _starts(nkr, blk // (24 * max(Pf, Na) + 300 * nblk2 * BLOCK))
    step2 = _program(('azimuth', nJ, Pf, Na, nblk2), make2)
    ja = np.arange(2 * nblk2 + 1, dtype=np.float64) * (BLOCK / 2)
    B = np.empty((nkr, nka), c64)
    qtab, ntab = g['qtab'], len(g['qtab'])
    for J0 in starts:
        hb = np.empty((nJ, Pf), c64)
        hb[:, :p0] = 0
        if jlo is None and place is None:
            lib.ql_transpose(at(A, 0, J0), nwin, P, nJ, at(hb, 0, p0), Pf, None, 0)
        else:
            lib.ql_unshear(ptr(A), nwin, P, None if jlo is None else ptr(jlo), None if place is None else ptr(place), Pp,
                           J0, nJ, at(hb, 0, p0), Pf)
        hb[:, p0 + Pp:] = 0
        # the pulse position of each knot's tangent, from the pulses' own (quicklook_cpu.cpp's ql_resample_az)
        # (beyond the table the line of its last two points: such positions lie outside the row and read zero)
        u = ((g['ka_lo'] + ja[None, :] * g['dka']) / g['kr'][J0:J0 + nJ, None] - g['tg0']) * g['tg_inv']
        kq = np.clip(np.floor(u), 0, ntab - 2).astype(np.int64)
        kn = _knots(qtab[kq] + (u - kq) * (qtab[kq + 1] - qtab[kq]))
        o = step2(*planes(hb), tab, dist_bins, live, edge, guard, scale, jnp.asarray(g['bins_a'][J0:J0 + nJ].astype(f32)),
                  *(jnp.asarray(v) for v in kn))
        to_host(B[J0:J0 + nJ], o[0][:, :nka], o[1][:, :nka])
    del A
    timing['azimuth'] = timing.get('azimuth', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # 3. the raster's range transform, in blocks of azimuth wavenumbers: the rows of a sheared raster that meet in
    # one bin summed (a reshape and a sum), the signs that make the padded transforms count from the image's center
    ov = g['ov'] if correct else 1
    Ny, hy, hx = ov * nfy, (g['hy'] if correct else 0), (nka // 2 if correct else 0)
    m = -(-nkr // Ny)
    sy = jnp.asarray(((-1.0) ** (np.arange(nkr) - hy)).astype(f32) if correct else np.ones(nkr, f32))
    sxh = ((-1.0) ** (np.arange(nka) - hx)).astype(f32) if correct else np.ones(nka, f32)

    def make3():
        def run(br, bi, sy, sxb):
            z = lax.complex(br, bi) * (sy[None, :] * sxb[:, None])
            z = jnp.pad(z, ((0, 0), (0, m * Ny - nkr))).reshape(z.shape[0], m, Ny).sum(1)
            f = jnp.fft.ifft(jnp.roll(z, -hy, axis=1), axis=1)
            return jnp.real(f), jnp.imag(f)
        return run
    na, starts = _starts(nka, blk // (40 * max(Ny, nkr)))
    step3 = _program(('transform', na, nkr, Ny, hy, bool(correct)), make3)
    F = np.empty((nka, Ny), c64)
    for a0 in starts:
        o = step3(*planes(np.ascontiguousarray(B[:, a0:a0 + na].T)), sy, jnp.asarray(sxh[a0:a0 + na]))
        to_host(F[a0:a0 + na], *o)
    del B
    timing['range transform'] = timing.get('range transform', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # 4. the azimuth transform and the picture, in blocks of range pixels
    out = np.empty((ny, nx), c64 if complex_image else np.float32)
    Nx = ov * nfx
    if not correct:
        cols = jnp.asarray(((np.arange(nx) - nx // 2) % nfx).astype(i32))
        y, x = (np.arange(ny) - ny / 2.0) * g['spy'], (np.arange(nx) - nx / 2.0) * g['spx']
        cyh = np.exp(1j * g['kr_min'] * y).astype(c64)
        cxd = planes(np.exp(-1j * g['ka_lo'] * x).astype(c64))

        def make4():
            def run(zr, zi, cols, cxr, cxi, cyr, cyi):
                t = jnp.take(jnp.fft.fft(lax.complex(zr, zi), axis=1), cols, axis=1)
                if not complex_image:
                    return jnp.abs(t), jnp.zeros((1, 1), jnp.float32)
                t = t * lax.complex(cyr, cyi)[:, None] * lax.complex(cxr, cxi)[None, :]
                return jnp.real(t), jnp.imag(t)
            return run
        bw = int(max(1, min(ny // 2, blk // (40 * Nx))))
        step4 = _program(('picture', bw, Nx, nx, bool(complex_image)), make4)
        for s_, e_ in ((0, ny // 2), (ny // 2, ny)):                     # the bins wrap at the picture's center
            for jb in _starts(e_ - s_, bw)[1]:
                jb += s_
                rb = (nfy - ny // 2 + jb) % nfy
                hb = np.zeros((bw, Nx), c64)
                lib.ql_transpose(at(F, 0, rb), Ny, nka, bw, ptr(hb), Nx, None, 0)
                o = step4(*planes(hb), cols, *cxd, *planes(cyh[jb:jb + bw]))
                if complex_image:
                    to_host(out[jb:jb + bw], *o)
                else:
                    out[jb:jb + bw] = np.asarray(o[0])
        timing['image'] = timing.get('image', 0.0) + time.perf_counter() - t0
        return out
    dist = g['dist']
    ltab = jnp.asarray(Q._polyphase('lanczos', Q.LANCZOS, PH))
    shear = jnp.float32(g['shear'] * g['spy'] / g['spx'])
    kr_c, ka_c = g['kr_min'] + hy * g['dkr'], g['ka_lo'] + hx * g['dka']
    msy = int(math.ceil(g['shift'][1] / g['spy'])) * ov + Q.LANCZOS + 2
    nxb = -(-nx // BLOCK)
    bw = int(max(1, min(ny, blk // (40 * Nx * ov + 700 * nxb * BLOCK))))
    span = min(Ny, ov * bw + 2 * msy)
    xk = (np.arange(2 * nxb + 1, dtype=np.float64) * (BLOCK / 2) - nx / 2.0) * g['spx']
    powers = dist['powers']

    def apparent(yk):
        """The apparent positions (m) of the knots [n, nxb + 1] of the range pixels at yk [n]."""
        X, Y = np.broadcast_to(xk[None, :], (len(yk), len(xk))), np.broadcast_to(yk[:, None], (len(yk), len(xk)))
        dx = sum(c * (X / dist['sx']) ** a * (Y / dist['sy']) ** b for c, (a, b) in zip(dist['cx'], powers))
        dy = sum(c * (X / dist['sx']) ** a * (Y / dist['sy']) ** b for c, (a, b) in zip(dist['cy'], powers))
        return X + dx, Y + dy

    def make5():
        def run(zr, zi, ltab, shear, bx, lx, sx_, cx_, by, ly, sy_, cy_, c0, dc, ddc):
            base = jnp.fft.fft(lax.complex(zr, zi), axis=1)
            ix, fx = _positions(jnp, bx, lx, sx_, cx_)
            iy, fy = _positions(jnp, by, ly, sy_, cy_)
            ar, ai = _read2(jnp, jnp.real(base), jnp.imag(base), ix, fx, iy, fy, shear, ltab, PH)
            if not complex_image:
                return jnp.sqrt(ar * ar + ai * ai), jnp.zeros((1, 1), jnp.float32)
            i = jnp.arange(BLOCK, dtype=jnp.float32)
            c = (c0[..., None] + i * dc[..., None] + (i * i) * ddc[..., None]).reshape(ar.shape)
            c = (c - jnp.floor(c)) * (2.0 * math.pi)
            cs, sn = jnp.cos(c), jnp.sin(c)
            return ar * cs - ai * sn, ar * sn + ai * cs
        return run
    step5 = _program(('read', bw, span, Nx, nxb, bool(complex_image)), make5)
    for jb in _starts(ny, bw)[1]:
        rb0 = int(min(max(0, Ny // 2 + ov * (jb - ny // 2) - msy), Ny - span))
        hb = np.empty((span, Nx), c64)
        lib.ql_transpose(at(F, hx, rb0), Ny, nka - hx, span, ptr(hb), Nx, None, 0)
        if hx:
            lib.ql_transpose(at(F, 0, rb0), Ny, hx, span, at(hb, 0, Nx - hx), Nx, None, 0)
        hb[:, nka - hx:Nx - hx] = 0
        Xa, Ya = apparent((jb + np.arange(bw) - ny / 2.0) * g['spy'])
        kx = _knots(Xa * (ov / g['spx']) + Nx / 2.0)
        ky = _knots(Ya * (ov / g['spy']) + Ny / 2.0 - rb0)
        _, c0, dc, ddc = _knots((kr_c * Ya - ka_c * Xa) / (2.0 * math.pi))        # the carrier, in cycles
        o = step5(*planes(hb), ltab, shear, *(jnp.asarray(v) for v in kx + ky + (c0, dc, ddc)))
        if complex_image:
            to_host(out[jb:jb + bw], o[0][:, :nx], o[1][:, :nx])
        else:
            out[jb:jb + bw] = np.asarray(o[0][:, :nx])
    timing['correction'] = timing.get('correction', 0.0) + time.perf_counter() - t0
    return out
