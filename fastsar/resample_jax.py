"""resample and the per-pixel work of planes in JAX (a TPU, or whatever device JAX runs on): the image stays on
the device, where everything is float32, complex64 or int32 (a TPU has no fast float64).

Positions. A position reaches tens of thousands of samples and must hold to a small part of a table phase (1/2048
sample), more than float32 carries. The host therefore evaluates a knot map along the lines in float64 (a small
array, one row of knots per line) and sends, for each line and knot interval, the integer part of the position at
the lower knot (int32) and the two knot values less it (float32). The device interpolates the two float32 values
and carries the integer part in int32: the error is about 1e-7 of the map's change across a knot interval.

Blocks. Lines run in blocks of one size through one jitted step per pass, and the blocks are joined at the end;
the last block is moved back to end at the last line, so that every block has the same shape and a pass compiles
once.

Two interpolations. 'gather' reads each tap of each output sample with an indexed load, as the CPU and CUDA kernels
do. A TPU does such loads slowly (about 1e8 a second, and a slice of a line at a row and a column index slower
still), but it looks up whole rows of a table quickly and its elementwise work is fast. 'blocks' therefore works
on blocks of B consecutive output samples of a line, for positions given as a knot map: the oversampled samples a
block needs are one window of the line; the windows that start at every 8th sample are laid out as the rows of a
table and each block reads its row (the host sends where it starts); each sample then moves its taps to the front
of the window by one select per bit of its offset in it. The weights are a polynomial in the table's phase (fitted
to the table, which it reproduces to 1e-7), evaluated at the phase the table would be read at. Nothing is rounded
that the gather path does not round. FASTSAR_RESAMPLE_JAX=blocks|gather chooses; the default is blocks on a TPU
and gather on JAX's other devices, where it is the faster. Positions given as arrays, a knot map with more than
one knot inside a block, and maps that are not smooth (a window above WINDOW samples) take the gather path.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import _xp, resample

BLOCK = 1 << 24            # samples of an oversampled block of lines (complex64: 128 MB, and a few times it in flight)
BLOCK_ROWS = 1 << 27       # float32 values of a block of lines in flight in the blocks path (window table, selects)
WINDOW = 120               # the widest window (oversampled samples) of an output block the blocks path takes
STRIDE = 8                 # samples between the starts of the windows in the table
SPAN = 40                  # oversampled samples a block of output samples may span: sets the block size
PHASES = 2048              # of resample.kernel_table
SNAP = 1.5e-5              # samples: a position this near a sample is on it (blocks path)


def _path():
    """'blocks' or 'gather': FASTSAR_RESAMPLE_JAX, else blocks on a TPU and gather on JAX's other devices."""
    v = os.environ.get('FASTSAR_RESAMPLE_JAX', 'auto').strip().lower() or 'auto'
    if v not in ('auto', 'blocks', 'gather'):
        raise ValueError(f"FASTSAR_RESAMPLE_JAX must be 'blocks', 'gather' or 'auto', got {v!r}")
    return ('blocks' if jax.devices()[0].platform == 'tpu' else 'gather') if v == 'auto' else v


def _samples(limit, share):
    """The samples of a block of lines: `limit` (BLOCK, BLOCK_ROWS), less where the device's memory is small
    (_xp.jax_block() bytes at `share` bytes a sample)."""
    return int(max(1, min(limit, _xp.jax_block() // share)))


def _split(rows):
    """Positions at the output knots [nl, b] (float64) -> the integer part at each interval's lower knot [nl, b - 1]
    (int32) and the two knot values less it (float32). NaN and positions beyond 2^30 become far outside."""
    r = np.clip(np.nan_to_num(rows, nan=-2.0 ** 30), -2.0 ** 30, 2.0 ** 30)
    base = np.floor(r[:, :-1])
    return base.astype(np.int32), (r[:, :-1] - base).astype(np.float32), (r[:, 1:] - base).astype(np.float32)


def _floor(i, f):
    """An integer part and a float32 remainder, the remainder brought to [0, 1)."""
    fl = jnp.floor(f)
    return i + fl.astype(jnp.int32), f - fl


def _knot(P, J):
    """The positions [nb, m] of a block, as integer part and remainder in [0, 1): P = (base, lo, hi) [nb, b - 1] of
    _split, J = (j0, 1 - wj, wj) [m] the knot interval and weights of each output sample."""
    base, lo, hi = P
    j0, uj, wj = J
    return _floor(jnp.take(base, j0, axis=1), jnp.take(lo, j0, axis=1) * uj + jnp.take(hi, j0, axis=1) * wj)


def _within(i, f, lo, hi):
    """lo <= i + f <= hi for integers lo, hi and f in [0, 1)."""
    return (i >= lo) & ((i < hi) | ((i == hi) & (f == 0)))


def _rint(i, f):
    """i + f rounded to the nearest integer, halves to even (numpy.rint)."""
    return i + ((f > 0.5) | ((f == 0.5) & (i % 2 == 1))).astype(jnp.int32)


def _oversample(z, pad, n2, real):
    """Lines z [nb, n2] oversampled `pad` times by a zero-padded FFT, the Nyquist bin split between the two halves
    (real lines by the real transforms)."""
    nb, h, N = z.shape[0], n2 // 2, pad * n2
    if real:
        F = jnp.fft.rfft(z, axis=1)
        head = [F[:, :h + 1]] if n2 % 2 else [F[:, :h], 0.5 * F[:, h:h + 1]]
        G = jnp.concatenate(head + [jnp.zeros((nb, N // 2 - h), F.dtype)], axis=1)
        return jnp.fft.irfft(G, n=N, axis=1)
    F = jnp.fft.fft(z, axis=1)
    if n2 % 2:
        G = jnp.concatenate([F[:, :h + 1], jnp.zeros((nb, N - n2), F.dtype), F[:, h + 1:]], axis=1)
    else:
        ny = 0.5 * F[:, h:h + 1]
        G = jnp.concatenate([F[:, :h], ny, jnp.zeros((nb, N - n2 - 1), F.dtype), ny, F[:, h + 1:]], axis=1)
    return jnp.fft.ifft(G, axis=1)


def _fast_len(n):
    """The least length of at least n that is 32 times 1, 3, 5, 7, 9 or 15 times a power of two: a TPU transforms
    a length that 32 does not divide some ten times slower."""
    return min(next(v for v in (32 * o << k for k in range(32)) if v >= n) for o in (1, 3, 5, 7, 9, 15))


_KERNELS = {}


def _kernels(n2, pad, real):
    """The transforms [pad - 1, Lf or Lf // 2 + 1] (two float32 arrays: real and imaginary part), at a fast length
    Lf >= 2 n2, of the kernels that give the samples of _oversample between the given ones: sample pad k + r of the
    zero-padded transform of length n2 is the circular convolution of the line with the inverse transform of
    exp(2 pi j f r / (pad n2)) over the signed frequencies f (cos(pi r / pad) at the Nyquist bin), a real kernel of
    period n2, laid out here over (-n2, n2) so that a transform of length Lf convolves with it. Divided by pad, as
    _oversample's values are."""
    key = (n2, pad, real)
    if key not in _KERNELS:
        if len(_KERNELS) > 16:
            _KERNELS.clear()
        Lf, f = _fast_len(2 * n2), np.fft.fftfreq(n2) * n2
        t = np.arange(-(n2 - 1), n2)
        K = []
        for r in range(1, pad):
            D = np.exp(2j * np.pi * f * r / (pad * n2))
            if n2 % 2 == 0:
                D[n2 // 2] = np.cos(np.pi * r / pad)
            ext = np.zeros(Lf)
            ext[t % Lf] = np.fft.ifft(D).real[t % n2]
            K.append((np.fft.rfft(ext) if real else np.fft.fft(ext)) / pad)
        K = np.stack(K)
        _KERNELS[key] = (K.real.astype(np.float32), K.imag.astype(np.float32))
    return tuple(jnp.asarray(v) for v in _KERNELS[key])


def _oversampled(u, pad, n2, real, K):
    """_oversample(u, pad, n2, real), by the transforms of the line itself (K None) or, with K = _kernels(n2, pad,
    real), by convolutions through transforms of K's length: the same samples to rounding."""
    if K is None:
        return _oversample(u, pad, n2, real)
    Kc = lax.complex(K[0], K[1])
    if real:
        Lf = 2 * (Kc.shape[1] - 1)
        F = jnp.fft.rfft(u, n=Lf, axis=1)
        parts = [jnp.fft.irfft(F * Kc[r], n=Lf, axis=1)[:, :n2] for r in range(pad - 1)]
    else:
        F = jnp.fft.fft(u, n=Kc.shape[1], axis=1)
        parts = [jnp.fft.ifft(F * Kc[r], axis=1)[:, :n2] for r in range(pad - 1)]
    return jnp.stack([u * (1.0 / pad)] + parts, -1).reshape(u.shape[0], pad * n2)


def _step(src, a, P, J, M, tab, K, *, axis, mode, nb, pad, guard, n2):
    """Lines a ... a + nb of a pass (resample_cuda._pass's conventions) -> [nb, m] ([m, nb] along axis 0). mode
    'knots': P, J as
    _knot's; 'split': P = (integer part, remainder) [nb, m] from the host; 'device': P = (pos [L, m],) on the device.
    M: None, or (Px, limits [xlo, xhi, ylo, yhi], valid [nx, ny] or None), warp's mask with Px as P."""
    real = not jnp.iscomplexobj(src)
    if axis == 1:
        n = src.shape[1]
        z = lax.dynamic_slice(src, (a, 0), (nb, n))
    else:
        n = src.shape[0]
        z = lax.dynamic_slice(src, (0, a), (n, nb)).T
    if mode == 'knots':
        ib, ff = _knot(P, J)
    elif mode == 'split':
        ib, ff = P
    else:
        p = lax.dynamic_slice(P[0], (a, 0), (nb, P[0].shape[1])).astype(jnp.float32)
        ib, ff = _floor(jnp.zeros(p.shape, jnp.int32), p)
    ok = _within(ib, ff, 0, n - 1)
    if M is not None:
        Px, lim, valid = M
        ix, fx = _knot(Px, J)
        ok = ok & _within(ix, fx, lim[0], lim[1]) & _within(ib, ff, lim[2], lim[3])
        if valid is not None:
            nx, ny = valid.shape
            at = jnp.clip(_rint(ix, fx), 0, nx - 1) * ny + jnp.clip(_rint(ib, ff), 0, ny - 1)
            ok = ok & jnp.take(valid.reshape(-1), at, mode='clip')
    u = jnp.pad(z, ((0, 0), (guard, n2 - n - guard)))
    if pad > 1:
        u = _oversampled(u, pad, n2, real, K)
    taps, nphase = tab.shape[1], tab.shape[0] - 1
    q = jnp.where(ok, ff, 0) * pad
    fl = jnp.floor(q)
    i0 = (jnp.where(ok, ib, 0) + guard) * pad + fl.astype(jnp.int32) - (taps // 2 - 1)
    k = jnp.rint((q - fl) * nphase).astype(jnp.int32)
    o = jnp.zeros(ib.shape, u.dtype)
    for t in range(taps):
        o = o + jnp.take_along_axis(u, i0 + t, axis=1, mode='clip') * jnp.take(tab[:, t], k, mode='clip')
    o = jnp.where(ok, o, 0)
    return o if axis == 1 else o.T


def _poly(tab):
    """A kernel table [PHASES + 1, taps] as polynomials: Horner coefficients [degree + 1, taps] (float32, the highest
    power first) in t = 2 f - 1 of each tap's weight at the fractional part f, of the lowest odd degree up to 11 that
    returns the table to 3e-7 (1 for 'linear', 3 for 'cubic', 9 for the 8-tap sinc)."""
    from numpy.polynomial import polynomial as npoly
    x, t = np.linspace(-1.0, 1.0, tab.shape[0]), tab.astype(np.float64)
    for deg in (1, 3, 5, 7, 9, 11):
        c = npoly.polyfit(x, t, deg)                                             # [deg + 1, taps], lowest first
        if np.abs(npoly.polyval(x, c).T - t).max() < 3e-7:
            break
    return np.ascontiguousarray(c[::-1], np.float32)


def _span(pos, B):
    """The output blocks of B samples of a knot map, for every line: (first sample ja, the knot interval ka of it,
    its weight wa, the interval ke of the block's last sample, kink: the offset of the knot inside the block, B
    when there is none) each [nblk]; None when a block holds more than one knot."""
    ja = np.arange(0, pos.m, B)
    je = np.minimum(ja + B - 1, pos.m - 1)
    ka, ke = pos.j0[ja], pos.j0[je]
    if (ke - ka > 1).any():
        return None
    return ja, ka, pos.wj[ja], ke, np.where(ke > ka, pos.ok[ka + 1] - ja, float(B)), je - ja


def _blocks(pos, sl, span):
    """Lines sl of a knot map by output blocks (span: _span's): base (int32), lo, s, ds (float32) [nl, nblk], sample
    i of block b lying at base + lo + i s + max(i - kink, 0) ds, the map's own piecewise-linear value; and the least
    and the greatest position of each block [nl, nblk] (float64; of its samples near the source when some are NaN or
    beyond 2^28)."""
    ja, ka, wa, ke, kink, last = span
    r = np.clip(np.nan_to_num(pos.rows(sl), nan=-2.0 ** 30), -2.0 ** 30, 2.0 ** 30)
    slope = np.diff(r, axis=1) / np.diff(pos.ok)
    p0 = r[:, ka] * (1 - wa) + r[:, ka + 1] * wa
    s = slope[:, ka]
    ds = slope[:, ke] - s
    c = np.stack([p0, p0 + last * s + np.maximum(last - kink, 0) * ds, p0 + np.minimum(kink, last) * s])
    far = np.abs(c) > 2.0 ** 28
    near = np.where(far[0], np.where(far[1], np.where(far[2], 0.0, c[2]), c[1]), c[0])
    c = np.where(far, near, c)
    base = np.floor(p0)
    return (base.astype(np.int32), (p0 - base).astype(np.float32), s.astype(np.float32), ds.astype(np.float32)), \
        c.min(0), c.max(0)


def _windows(u, w0, R):
    """u [nb, N] (N a multiple of STRIDE), w0 [nb, nblk] (int32, multiples of STRIDE, w0 + R <= N; R a multiple of
    STRIDE) -> [nb, nblk, R]: u[l, w0[l, b] : w0[l, b] + R], each read as one row of the table of the windows
    that start at every STRIDE-th sample."""
    nb, N = u.shape
    c = u.reshape(nb, N // STRIDE, STRIDE)
    nrow = (N - R) // STRIDE + 1
    tab = jnp.concatenate([c[:, i:nrow + i] for i in range(R // STRIDE)], -1).reshape(nb * nrow, R)
    return tab.at[jnp.arange(nb, dtype=jnp.int32)[:, None] * nrow + w0 // STRIDE].get(mode='promise_in_bounds')


def _front(g, rel, W, taps):
    """g [..., 1, R] (windows), rel [..., B, 1] (int32, 0 <= rel < W, rel + taps <= R) -> [..., B, taps]:
    g[..., rel : rel + taps], by one select per bit of rel: after the select of a bit the offset left is below it."""
    bit = 1 << max(W - 1, 1).bit_length() - 1
    while bit:
        nw = bit + taps - 1
        lo, hi = g[..., :nw], g[..., bit:bit + nw]
        lo, hi = (jnp.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, nw - v.shape[-1])]) if v.shape[-1] < nw else v
                  for v in (lo, hi))
        g = jnp.where(rel & bit != 0, hi, lo)
        bit >>= 1
    return g


def _at(D, kink, B):
    """The positions [nb, nblk, B] of a block of lines from _blocks' arrays, as integer part and remainder in [0, 1).
    A remainder within SNAP of an integer is that integer: the sum below holds to a few 1e-6 of a sample, and a
    position that is the last sample of a line exactly (an output grid that ends on it) must not fall outside."""
    base, lo, s, ds = D
    i = jnp.arange(B, dtype=jnp.float32)
    f = lo[..., None] + i * s[..., None] + jnp.maximum(i - kink[:, None], 0) * ds[..., None]
    ib, ff = _floor(base[..., None], jnp.clip(f, -2.0 ** 24, 2.0 ** 24))
    return ib + (ff > 1 - SNAP).astype(jnp.int32), jnp.where((ff < SNAP) | (ff > 1 - SNAP), 0, ff)


def _step_blocks(src, a, D, w0, kink, M, coef, K, *, axis, nb, m, pad, guard, n2, B, W, R):
    """Lines a ... a + nb of a pass by blocks of B output samples -> [nb, m] ([m, nb] along axis 0). D, kink:
    _blocks' and _span's;
    w0 [nb, nblk]: the first oversampled sample of each block's window of R, the first taps of its samples starting
    less than W after it; coef: _poly's, times pad; M: None or (Dx, limits, valid), warp's mask; K: _oversampled's."""
    real = not jnp.iscomplexobj(src)
    if axis == 1:
        n = src.shape[1]
        z = lax.dynamic_slice(src, (a, 0), (nb, n))
    else:
        n = src.shape[0]
        z = lax.dynamic_slice(src, (0, a), (n, nb)).T
    taps, nblk = coef.shape[1], w0.shape[1]
    ib, ff = _at(D, kink, B)
    ok = _within(ib, ff, 0, n - 1)
    if M is not None:
        Dx, lim, valid = M
        ix, fx = _at(Dx, kink, B)
        ok = ok & _within(ix, fx, lim[0], lim[1]) & _within(ib, ff, lim[2], lim[3])
        if valid is not None:
            nx, ny = valid.shape
            at = jnp.clip(_rint(ix, fx), 0, nx - 1) * ny + jnp.clip(_rint(ib, ff), 0, ny - 1)
            ok = ok & jnp.take(valid.reshape(-1), at, mode='clip')
    u = jnp.pad(z, ((0, 0), (guard, n2 - n - guard)))
    if pad > 1:
        u = _oversampled(u, pad, n2, real, K)
    N = max(-(-u.shape[1] // STRIDE) * STRIDE, R)
    u = jnp.pad(u, ((0, 0), (0, N - u.shape[1])))
    q = jnp.where(ok, ff, 0) * pad
    fl = jnp.floor(q)
    i0 = (jnp.where(ok, ib, 0) + guard) * pad + fl.astype(jnp.int32) - (taps // 2 - 1)
    rel = jnp.clip(i0 - w0[..., None], 0, W - 1)[..., None]                            # [nb, nblk, B, 1]
    t = jnp.rint((q - fl) * PHASES) * (2.0 / PHASES) - 1.0
    w = jnp.broadcast_to(coef[0], t.shape + (taps,))
    for c in coef[1:]:
        w = w * t[..., None] + c
    o = [(_front(_windows(ch, w0, R)[:, :, None, :], rel, W, taps) * w).sum(-1)
         for ch in ((u,) if real else (jnp.real(u), jnp.imag(u)))]
    o = o[0] if real else lax.complex(o[0], o[1])
    o = jnp.where(ok, o, 0).reshape(nb, nblk * B)[:, :m]
    return o if axis == 1 else o.T


def _plan_blocks(pos, posx, L, n, tab, pad, guard, n2, real):
    """The blocks path's plan of a pass on a knot map, or None when the map does not suit it: dict(B, W, R
    (_step_blocks'), nb (lines a step), D, w0, kink (the host arrays of _step_blocks for all the lines), Dx (the
    mask's map, when posx is given))."""
    taps = tab.shape[1]
    with np.errstate(invalid='ignore', divide='ignore'):
        sp = np.nanmedian(np.abs(np.diff(pos.v, axis=1) / np.diff(pos.ok))) * pad
    if not np.isfinite(sp):
        return None
    span = None
    for B in (32, 16, 8, 4):
        if B * sp + 4 <= SPAN or B == 4:
            span = _span(pos, B)
            if span is not None:
                break
    if span is None:
        return None
    nblk, off = len(span[0]), guard * pad - (taps // 2 - 1) - 1
    # the whole pass on the host first: the window is one size for every block of lines
    from .patches import _rows
    got = {}

    def work(sl):
        d, lo, hi = _blocks(pos, sl, span)
        a0, a1 = (np.floor(np.clip(v, 0, n - 1) * pad) for v in (lo, hi))
        start = (a0 + off) // STRIDE * STRIDE                       # where the window starts: on the table's stride
        got[sl.start] = (d, start.astype(np.int32), int((a1 + off + 3 - start).max()),
                         _blocks(posx, sl, span)[0] if posx is not None else None)
    _rows(work, L, 256)
    got = [got[k] for k in sorted(got)]
    W = max(g[2] for g in got)
    if W > WINDOW:
        return None
    D, w0, Dx = [g[0] for g in got], [g[1] for g in got], [g[3] for g in got]
    cat = lambda parts: np.concatenate(parts) if len(parts) > 1 else parts[0]
    W = -(-W // 4) * 4
    R = -(-(W + taps - 1) // STRIDE) * STRIDE
    N = max(-(-pad * n2 // STRIDE) * STRIDE, R)
    per = (1 if real else 2) * (N // STRIDE * R + nblk * B * (W + 2 * taps))
    nb = int(max(1, min(L, _samples(BLOCK_ROWS, 8) // per, _samples(BLOCK, 64) // max(pad * n2, pos.m))))
    return dict(B=B, W=W, R=R, nb=nb, D=tuple(cat([d[k] for d in D]) for k in range(4)), w0=np.clip(cat(w0), 0, N - R),
                Dx=tuple(cat([d[k] for d in Dx]) for k in range(4)) if posx is not None else None,
                kink=span[4].astype(np.float32))


_JIT = {}


def _jit(fn, static):
    if fn not in _JIT:
        _JIT[fn] = jax.jit(fn, static_argnames=static)
    return _JIT[fn]


def _joined(parts, L, nb, axis=0):
    """The blocks of lines of _starts(L, nb), in its order, as one array along `axis`: the last block, moved back,
    gives only the lines the others have not. (A block written into a buffer of the whole image costs a copy of
    that buffer on a TPU, one for every block.)"""
    if len(parts) == 1:
        return parts[0]
    if L % nb:
        cut = [slice(None)] * parts[-1].ndim
        cut[axis] = slice(nb - L % nb, None)
        parts = parts[:-1] + [parts[-1][tuple(cut)]]
    return jnp.concatenate(parts, axis)


def _starts(L, nb):
    """Block starts: steps of nb, the last block moved back to end at L."""
    return list(range(0, L - nb + 1, nb)) + ([L - nb] if L % nb else [])


def _pass(src, pos, m, axis, kernel, taps, pad, mask=None):
    """One pass along `axis` of src, as resample_cuda._pass, returning the new array."""
    real = not jnp.iscomplexobj(src)
    dt = jnp.float32 if real else jnp.complex64
    src = src.astype(dt)
    L, n = (src.shape[0], src.shape[1]) if axis == 1 else (src.shape[1], src.shape[0])
    tab, pad, guard, n2 = resample._plan(n, kernel, taps, pad)
    tabd = jnp.asarray(tab * np.float32(pad))
    nb = int(max(1, min(L, _samples(BLOCK, 64) // max(pad * n2, m))))
    knots = isinstance(pos, resample.KnotMap)
    mode = 'knots' if knots else 'device' if isinstance(pos, jax.Array) else 'split'
    J = M = None
    if knots:
        J = (jnp.asarray(pos.j0.astype(np.int32)), jnp.asarray((1 - pos.wj).astype(np.float32)),
             jnp.asarray(pos.wj.astype(np.float32)))
    if mask is not None:
        posx, (xlo, xhi), (ylo, yhi), valid = mask
        if not (knots and np.array_equal(posx.lk, pos.lk) and np.array_equal(posx.ok, pos.ok)):
            raise ValueError('the maps of a mask must share their knots')
        lim = jnp.asarray(np.array([xlo, xhi, ylo, yhi], np.int32))
        valid = None if valid is None else jnp.asarray(valid).astype(bool)
    out, plan = [], None
    # on a TPU (wherever the blocks path is chosen) a length that 32 does not divide is transformed at a fast one
    K = _kernels(n2, pad, real) if pad > 1 and n2 % 32 and _path() == 'blocks' else None
    if knots and _path() == 'blocks':
        plan = _plan_blocks(pos, posx if mask is not None else None, L, n, tab, pad, guard, n2, real)
    if plan is not None:
        step = _jit(_step_blocks, ('axis', 'nb', 'm', 'pad', 'guard', 'n2', 'B', 'W', 'R'))
        coef, kink, nb = jnp.asarray(_poly(tab) * np.float32(pad)), jnp.asarray(plan['kink']), plan['nb']
        for a in _starts(L, nb):
            sl = slice(a, a + nb)
            D = tuple(jnp.asarray(v[sl]) for v in plan['D'])
            if mask is not None:
                M = (tuple(jnp.asarray(v[sl]) for v in plan['Dx']), lim, valid)
            out.append(step(src, np.int32(a), D, jnp.asarray(plan['w0'][sl]), kink, M, coef, K, axis=axis, nb=nb, m=m,
                            pad=pad, guard=guard, n2=n2, B=plan['B'], W=plan['W'], R=plan['R']))
        return _joined(out, L, nb, 1 - axis)
    step = _jit(_step, ('axis', 'mode', 'nb', 'pad', 'guard', 'n2'))
    for a in _starts(L, nb):
        sl = slice(a, a + nb)
        if knots:
            P = tuple(jnp.asarray(v) for v in _split(pos.rows(sl)))
        elif mode == 'device':
            P = (pos,)
        else:
            q = np.clip(np.nan_to_num(np.asarray(pos(sl) if callable(pos) else pos[sl], np.float64), nan=-2.0 ** 30),
                        -2.0 ** 30, 2.0 ** 30)
            ib = np.floor(q)
            P = (jnp.asarray(ib.astype(np.int32)), jnp.asarray((q - ib).astype(np.float32)))
        if mask is not None:
            M = (tuple(jnp.asarray(v) for v in _split(posx.rows(sl))), lim, valid)
        out.append(step(src, np.int32(a), P, J, M, tabd, K, axis=axis, mode=mode, nb=nb, pad=pad, guard=guard, n2=n2))
    return _joined(out, L, nb, 1 - axis)


def lines(src, pos, m, *, kernel='sinc', taps=8, pad=None, out=None):
    """resample.lines for a JAX array (out must be None: JAX arrays are not written in place)."""
    if out is not None:
        raise ValueError('out is not available for JAX arrays')
    return _pass(src, pos, m, 1, kernel, taps, pad)


def warp(src, pos0, pos1, shape, *, kernel='sinc', taps=8, pad=None, transposed=False, mask=None):
    """resample.warp for a JAX array; transposed and mask as resample_cuda.warp's (the mask's bounds integers)."""
    m0, m1 = shape
    if transposed:
        return _pass(_pass(src, pos0, m0, 1, kernel, taps, pad), pos1, m1, 0, kernel, taps, pad)
    return _pass(_pass(src, pos0, m0, 0, kernel, taps, pad), pos1, m1, 1, kernel, taps, pad, mask)


def _deramp_step(img, a, rows, cols, r0, k, *, nb):
    """Rows a ... a + nb of deramp."""
    px, pi, q1i, q2i, li, hi = (lax.dynamic_slice(v, (a,), (nb,))[:, None] for v in rows)
    py, pj, q1j, q2j, jr = (v[None, :] for v in cols)
    p, q = pi + pj, (q1i + q1j) ** 2 + (q2i + q2j) ** 2
    s = r0 + p
    Q = q / (s + jnp.sqrt(s * s + q))                                       # R - R0 - p
    rest = -(q / (2 * r0)) * (2 * p + Q) / (2 * r0 + 2 * p + Q)                # Q less q / (2 R0)
    turns = ((hi * jr) & 4095).astype(jnp.float32) * (1 / 4096) + li * jr.astype(jnp.float32)
    ang = px + py + (2 * np.pi) * turns + k * rest
    z = lax.dynamic_slice(img, (a, 0), (nb, img.shape[1]))
    return z * lax.complex(jnp.cos(ang), jnp.sin(ang))


def deramp(img, d0, a, b, u0, r0, k):
    """planes.deramp's multiplication in float32: img [nx, ny] times exp(j k (R - r0)), R the range of pixel (i, j)
    at scp + d0 + i a + j b (m) from the antenna at scp - r0 u0. -> the new array (complex64).

    With d the pixel less the scene center, p = u0 . d and q1, q2 the components of d across u0, the range is
    R - r0 = p + Q, Q = (q1^2 + q2^2) / (R + r0 + p), with no cancellation. d is linear in (i, j), so k p and the
    parts of k (q1^2 + q2^2) / (2 r0) that depend on i alone or on j alone are phases of one index: the host
    reduces them modulo 2 pi in float64 and sends them as float32 vectors. The part in i j is theta i j turns with
    one constant theta: the host sends, per row, theta i modulo 1 as an integer number of 1/4096 turns and a small
    remainder, and the device takes the integer part times j modulo 4096 in int32. The device computes only what
    is left of Q, less than a part in a hundred of it, in float32. Columns must number under 2^20."""
    nx, ny = img.shape
    d0, a, b, u0 = (np.asarray(v, np.float64) for v in (d0, a, b, u0))
    ic, jc = nx // 2, ny // 2
    i, j = np.arange(nx, dtype=np.float64) - ic, np.arange(ny, dtype=np.float64) - jc
    dc = d0 + ic * a + jc * b
    t1 = np.cross(u0, np.eye(3)[np.argmin(np.abs(u0))])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(u0, t1)
    pi, pj = u0 @ dc + i * (u0 @ a), j * (u0 @ b)
    q1i, q2i, q1j, q2j = t1 @ dc + i * (t1 @ a), t2 @ dc + i * (t2 @ a), j * (t1 @ b), j * (t2 @ b)
    px = k * pi + k / (2 * r0) * (q1i ** 2 + q2i ** 2)
    py = k * pj + k / (2 * r0) * (2 * (t1 @ dc) * q1j + 2 * (t2 @ dc) * q2j + q1j ** 2 + q2j ** 2)
    ti = k * ((t1 @ a) * (t1 @ b) + (t2 @ a) * (t2 @ b)) / r0 / (2 * np.pi) * i        # the i j part: turns per column
    ti -= np.rint(ti)
    hi = np.rint(ti * 4096)
    wrap = lambda v: (np.remainder(v + np.pi, 2 * np.pi) - np.pi).astype(np.float32)
    f32 = lambda v: v.astype(np.float32)
    rows = (wrap(px), f32(pi), f32(q1i), f32(q2i), f32(ti - hi / 4096), hi.astype(np.int32))
    cols = (wrap(py), f32(pj), f32(q1j), f32(q2j), j.astype(np.int32))
    rows, cols = tuple(jnp.asarray(v) for v in rows), tuple(jnp.asarray(v) for v in cols)
    img = img.astype(jnp.complex64)
    nb = int(max(1, min(nx, _samples(BLOCK, 64) // ny)))
    step = _jit(_deramp_step, ('nb',))
    return _joined([step(img, np.int32(s), rows, cols, np.float32(r0), np.float32(k), nb=nb) for s in _starts(nx, nb)],
                   nx, nb)
