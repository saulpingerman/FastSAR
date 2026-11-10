"""resample and the per-pixel work of planes in JAX (a TPU, or whatever device JAX runs on): the image stays on
the device, where everything is float32, complex64 or int32 (a TPU has no fast float64).

Positions. A position reaches tens of thousands of samples and must hold to a small part of a table phase (1/2048
sample), more than float32 carries. The host therefore evaluates a knot map along the lines in float64 (a small
array, one row of knots per line) and sends, for each line and knot interval, the integer part of the position at
the lower knot (int32) and the two knot values less it (float32). The device interpolates the two float32 values
and carries the integer part in int32: the error is about 1e-7 of the map's change across a knot interval.

Blocks. Lines run in blocks of one size through one jitted step per pass, which writes its block into the output
(a donated buffer); the last block is moved back to end at the last line, so that every block has the same shape
and a pass compiles once.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import resample

BLOCK = 1 << 24            # samples of an oversampled block of lines (complex64: 128 MB, and a few times it in flight)


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


def _step(out, src, a, P, J, M, tab, *, axis, mode, nb, pad, guard, n2):
    """Lines a ... a + nb of a pass (resample_cuda._pass's conventions), written into out. mode 'knots': P, J as
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
        p = resample._snap(lax.dynamic_slice(P[0], (a, 0), (nb, P[0].shape[1])).astype(jnp.float32), n, jnp)
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
        u = _oversample(u, pad, n2, real)
    taps, nphase = tab.shape[1], tab.shape[0] - 1
    q = jnp.where(ok, ff, 0) * pad
    fl = jnp.floor(q)
    i0 = (jnp.where(ok, ib, 0) + guard) * pad + fl.astype(jnp.int32) - (taps // 2 - 1)
    k = jnp.rint((q - fl) * nphase).astype(jnp.int32)
    o = jnp.zeros(ib.shape, u.dtype)
    for t in range(taps):
        o = o + jnp.take_along_axis(u, i0 + t, axis=1, mode='clip') * jnp.take(tab[:, t], k, mode='clip')
    o = jnp.where(ok, o, 0)
    return lax.dynamic_update_slice(out, o, (a, 0)) if axis == 1 else lax.dynamic_update_slice(out, o.T, (0, a))


_JIT = {}


def _jit(fn, static):
    if fn not in _JIT:
        _JIT[fn] = jax.jit(fn, static_argnames=static, donate_argnums=(0,))
    return _JIT[fn]


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
    nb = int(max(1, min(L, BLOCK // max(pad * n2, m))))
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
    step = _jit(_step, ('axis', 'mode', 'nb', 'pad', 'guard', 'n2'))
    out = jnp.zeros((L, m) if axis == 1 else (m, L), dt)
    for a in _starts(L, nb):
        sl = slice(a, a + nb)
        if knots:
            P = tuple(jnp.asarray(v) for v in _split(pos.rows(sl)))
        elif mode == 'device':
            P = (pos,)
        else:
            q = np.clip(np.nan_to_num(np.asarray(pos(sl) if callable(pos) else pos[sl], np.float64), nan=-2.0 ** 30),
                        -2.0 ** 30, 2.0 ** 30)
            q = resample._snap(q, n)
            ib = np.floor(q)
            P = (jnp.asarray(ib.astype(np.int32)), jnp.asarray((q - ib).astype(np.float32)))
        if mask is not None:
            M = (tuple(jnp.asarray(v) for v in _split(posx.rows(sl))), lim, valid)
        out = step(out, src, np.int32(a), P, J, M, tabd, axis=axis, mode=mode, nb=nb, pad=pad, guard=guard, n2=n2)
    return out


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


def _deramp_step(out, img, a, rows, cols, r0, k, *, nb):
    """Rows a ... a + nb of deramp, written into out."""
    px, pi, q1i, q2i, li, hi = (lax.dynamic_slice(v, (a,), (nb,))[:, None] for v in rows)
    py, pj, q1j, q2j, jr = (v[None, :] for v in cols)
    p, q = pi + pj, (q1i + q1j) ** 2 + (q2i + q2j) ** 2
    s = r0 + p
    Q = q / (s + jnp.sqrt(s * s + q))                                       # R - R0 - p
    rest = -(q / (2 * r0)) * (2 * p + Q) / (2 * r0 + 2 * p + Q)                # Q less q / (2 R0)
    turns = ((hi * jr) & 4095).astype(jnp.float32) * (1 / 4096) + li * jr.astype(jnp.float32)
    ang = px + py + (2 * np.pi) * turns + k * rest
    z = lax.dynamic_slice(img, (a, 0), (nb, img.shape[1]))
    return lax.dynamic_update_slice(out, z * lax.complex(jnp.cos(ang), jnp.sin(ang)), (a, 0))


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
    nb = int(max(1, min(nx, BLOCK // ny)))
    step = _jit(_deramp_step, ('nb',))
    out = jnp.zeros((nx, ny), jnp.complex64)
    for s in _starts(nx, nb):
        out = step(out, img, np.int32(s), rows, cols, np.float32(r0), np.float32(k), nb=nb)
    return out
