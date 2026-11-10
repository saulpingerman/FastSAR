"""Band-limited resampling of images along one axis at a time, and two-pass warps built from it.

A complex SAR image sampled a few percent above its Nyquist rate cannot be interpolated by a short local kernel: a
windowed sinc that holds -50 dB at 4% oversampling needs about 70 taps. Each line is therefore first oversampled
exactly by a zero-padded FFT (twice the rate by default) and then interpolated by a short Kaiser-windowed sinc read
from a table of fractional phases: 8 taps hold about -58 dB, 12 about -85 dB. A warp whose two source coordinates
are smooth functions of the output position runs as two such passes (Catmull and Smith, 1980), the first along
axis 0 at each source column, the second along axis 1 at each output row.

Samples outside the source are zero, and an output position outside the source gives zero.
"""
import numpy as np

__all__ = ['kernel_table', 'lines', 'warp', 'KnotMap']


def kernel_table(kernel='sinc', taps=8, nphase=2048):
    """Interpolation weights [nphase + 1, taps] (float32): row k holds the weights of the taps at offsets
    -(taps/2 - 1) ... taps/2 from the sample left of a position with fractional part k / nphase.
    'sinc': a Kaiser-windowed sinc of cutoff half the sample rate, for lines oversampled twice (pass band a
    quarter of the rate, stop band from three quarters; ripple and rejection 8 + 7.18 (taps - 1) dB).
    'cubic': Keys' cubic convolution (a = -1/2), 4 taps, for real images that are already smooth at the sample
    scale. 'linear': 2 taps. Each row sums to one."""
    if kernel == 'linear':
        taps = 2
    elif kernel == 'cubic':
        taps = 4
    elif kernel != 'sinc':
        raise ValueError(f"kernel must be 'sinc', 'cubic' or 'linear', got {kernel!r}")
    if taps < 2 or taps % 2:
        raise ValueError(f'taps must be an even number of at least 2, got {taps!r}')
    f = np.arange(nphase + 1, dtype=np.float64)[:, None] / nphase
    t = f - (np.arange(taps, dtype=np.float64)[None, :] - (taps // 2 - 1))          # position minus tap location
    if kernel == 'linear':
        w = 1.0 - np.abs(t)
    elif kernel == 'cubic':
        a, s = -0.5, np.abs(t)
        w = np.where(s <= 1, (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1, a * s ** 3 - 5 * a * s ** 2 + 8 * a * s - 4 * a)
    else:
        att = 8.0 + 2.285 * (taps - 1) * np.pi
        beta = 0.1102 * (att - 8.7) if att > 50 else 0.5842 * (att - 21) ** 0.4 + 0.07886 * (att - 21)
        w = np.sinc(t) * np.i0(beta * np.sqrt(np.clip(1 - (2 * t / taps) ** 2, 0, None))) / np.i0(beta)
    return np.ascontiguousarray(w / w.sum(1, keepdims=True), np.float32)


class KnotMap:
    """Source positions of a block of lines, bilinear between exact values on a coarse grid of knots:
    KnotMap(line_knots [a], out_knots [b], values [a, b])(sl) -> [len(sl), m] float64, for lines sl of n and the m
    output samples of each. The knots are indices (not necessarily integers) and must cover [0, n - 1] and
    [0, m - 1]. A warp map of curvature 1/R sampled every h metres is bilinear to h^2 / (8 R)."""

    def __init__(self, line_knots, out_knots, values, m):
        self.lk, self.ok = np.ascontiguousarray(line_knots, np.float64), np.ascontiguousarray(out_knots, np.float64)
        self.v, self.m = np.ascontiguousarray(values, np.float64), int(m)
        if self.v.shape != (len(self.lk), len(self.ok)):
            raise ValueError(f'values must be [{len(self.lk)}, {len(self.ok)}], got {self.v.shape}')
        j = np.arange(self.m, dtype=np.float64)
        self.j0 = np.clip(np.searchsorted(self.ok, j, 'right') - 1, 0, len(self.ok) - 2)
        self.wj = (j - self.ok[self.j0]) / (self.ok[self.j0 + 1] - self.ok[self.j0])

    def __call__(self, sl):
        i = np.arange(sl.start, sl.stop, dtype=np.float64)
        i0 = np.clip(np.searchsorted(self.lk, i, 'right') - 1, 0, len(self.lk) - 2)
        wi = ((i - self.lk[i0]) / (self.lk[i0 + 1] - self.lk[i0]))[:, None]
        rows = self.v[i0] * (1 - wi) + self.v[i0 + 1] * wi                          # [L, b]
        return rows[:, self.j0] * (1 - self.wj) + rows[:, self.j0 + 1] * self.wj


def _oversample(z, pad, n2, workers=1):
    """Lines z [L, n2] oversampled `pad` times by a zero-padded FFT (the Nyquist bin split between the two halves)."""
    from scipy import fft
    if pad == 1:
        return z
    F = fft.fft(z, axis=1, workers=workers)
    G = np.zeros((z.shape[0], pad * n2), F.dtype)
    h = n2 // 2
    G[:, :h] = F[:, :h]
    G[:, -h:] = F[:, n2 - h:]
    if n2 % 2:
        G[:, h] = F[:, h]
    else:
        G[:, -h] *= 0.5
        G[:, h] = G[:, -h]
    G *= pad
    return fft.ifft(G, axis=1, overwrite_x=True, workers=workers)


def _block(src, pos, tab, pad, guard, n2, real):
    """Lines src [L, n] at positions pos [L, m] (source samples): oversample, then the table-driven interpolation."""
    L, n = src.shape
    taps, nphase = tab.shape[1], tab.shape[0] - 1
    z = np.zeros((L, n2), np.float32 if real and pad == 1 else np.complex64)
    z[:, guard:guard + n] = src
    u = _oversample(z, pad, n2)
    if real and pad > 1:
        u = u.real
    u = np.ascontiguousarray(u, np.float32 if real else np.complex64).reshape(-1)
    ok = (pos >= 0) & (pos <= n - 1)
    p = (np.where(ok, pos, 0.0) + guard) * pad
    i0 = np.floor(p)
    k = np.rint((p - i0) * nphase).astype(np.intp)
    idx = i0.astype(np.intp) - (taps // 2 - 1) + (np.arange(L, dtype=np.intp) * (pad * n2))[:, None]
    out = np.zeros(pos.shape, u.dtype)
    tmp, w = np.empty(pos.shape, u.dtype), np.empty(pos.shape, np.float32)
    for t in range(taps):
        np.take(u, idx + t, out=tmp, mode='clip')
        np.take(tab[:, t], k, out=w)
        tmp *= w
        out += tmp
    out[~ok] = 0
    return out


def lines(src, pos, m=None, *, kernel='sinc', taps=8, pad=None, out=None):
    """src [L, n] resampled along its last axis: out[l, j] = src[l, pos[l, j]] by band-limited interpolation.
    pos: [L, m] source positions in samples, or a function of a slice of lines returning [len, m] (KnotMap), with
    m then given. Complex input gives complex64, real input float32. pad: FFT oversampling before the kernel
    (default 2 for 'sinc', 1 for 'cubic' and 'linear'); guard zeros at both ends keep the periodic FFT from wrapping
    one end of a line into the other. Lines run in blocks on all cores."""
    src = np.asarray(src)
    if src.ndim != 2:
        raise ValueError(f'src must be 2-D, got shape {src.shape}')
    real = not np.iscomplexobj(src)
    if not callable(pos):
        pos = np.asarray(pos)
        if pos.ndim != 2 or pos.shape[0] != src.shape[0]:
            raise ValueError(f'pos must be [{src.shape[0]}, m], got shape {pos.shape}')
        m, arr, pos = pos.shape[1], pos, (lambda sl: np.asarray(arr[sl], np.float64))
    if m is None:
        raise ValueError('m (output samples per line) is required when pos is a function')
    if out is None:
        out = np.empty((src.shape[0], int(m)), np.float32 if real else np.complex64)
    _run(src.__getitem__, out.__setitem__, src.shape[0], src.shape[1], real, pos, int(m), kernel, taps, pad)
    return out


_LIB = []


def _lib():
    """The C++ kernel (resample_cpu.cpp), compiled on first use as the other CPU kernels; None where there is no
    compiler or FASTSAR_RESAMPLE_KERNEL=0, and the NumPy path then runs."""
    if not _LIB:
        import ctypes
        import os
        L = None
        if os.environ.get('FASTSAR_RESAMPLE_KERNEL', '1') != '0':
            try:
                from ._build import shared_object, default_flags
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resample_cpu.cpp')) as fh:
                    src = fh.read()
                L = ctypes.CDLL(shared_object('resample_cpu', src, default_flags() + ['-std=c++17']))
                f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
                i, n, d, p = ctypes.c_int, ctypes.c_long, ctypes.c_double, ctypes.c_void_p
                L.resample_lines.argtypes = [f32, n, n, i, p, n, p, i, p, i, p, d, d, d, n, f32, i, i, f32]
                L.resample_lines.restype = None
            except (RuntimeError, OSError):
                L = None
        _LIB.append(L)
    return _LIB[0]


def _run(get, put, L, n, real, pos, m, kernel, taps, pad, block_bytes=64 << 20):
    """lines() on line blocks read by get(slice) -> [len, n] and written by put(slice, [len, m])."""
    from scipy import fft
    from .patches import _rows, _ncpu
    tab = kernel_table(kernel, taps)
    pad = (2 if kernel == 'sinc' else 1) if pad is None else int(pad)
    guard = tab.shape[1] if pad == 1 else 16
    n2 = fft.next_fast_len(n + 2 * guard) if pad > 1 else n + 2 * guard
    lib = _lib()
    if lib is None:
        chunk = int(max(1, min(256, (1 << 20) // (8 * max(pad * n2, m)))))

        def work(sl):
            put(sl, _block(get(sl), pos(sl), tab, pad, guard, n2, real))
        _rows(work, L, chunk)
        return
    ch, dt = (1, np.float32) if real else (2, np.complex64)
    nb = int(max(1, min(L, block_bytes // (8 * pad * n2))))
    knots = isinstance(pos, KnotMap)
    for a in range(0, L, nb):
        sl = slice(a, min(L, a + nb))
        z = np.zeros((sl.stop - a, n2), dt)
        z[:, guard:guard + n] = get(sl)
        if pad > 1:
            z = _oversample(z.astype(np.complex64, copy=False), pad, n2, _ncpu())
            z = np.ascontiguousarray(z.real if real else z, dt)
        o = np.empty((sl.stop - a, m), dt)
        if knots:
            args = (None, a, pos.lk.ctypes.data, len(pos.lk), pos.ok.ctypes.data, len(pos.ok), pos.v.ctypes.data)
        else:
            q = np.ascontiguousarray(pos(sl), np.float64)
            args = (q.ctypes.data, a, None, 0, None, 0, None)
        lib.resample_lines(z.view(np.float32).reshape(sl.stop - a, -1), sl.stop - a, pad * n2, ch, *args,
                           float(pad), float(guard * pad), float(n), m, tab, tab.shape[0] - 1, tab.shape[1],
                           o.view(np.float32).reshape(sl.stop - a, -1))
        put(sl, o)


def warp(src, pos0, pos1, shape, *, kernel='sinc', taps=8, pad=None):
    """Two-pass warp of src [n0, n1] to `shape` [m0, m1].
    Pass 1, along axis 0 at each source column j: mid[i, j] = src[pos0[j, i], j], with pos0 [n1, m0] (or a function
    of a slice of columns): the axis-0 source position of output row i on source column j.
    Pass 2, along axis 1 at each output row i: out[i, k] = mid[i, pos1[i, k]], with pos1 [m0, m1] (or a function of
    a slice of rows): the axis-1 source position of output pixel (i, k).
    The source must be band limited along axis 0 at each column and the intermediate along axis 1; a spectrum
    that wraps along axis 1 before pass 1 (a sheared support on a critically sampled grid) is resolved by it."""
    src = np.asarray(src)
    if src.ndim != 2:
        raise ValueError(f'src must be 2-D, got shape {src.shape}')
    n0, n1 = src.shape
    m0, m1 = int(shape[0]), int(shape[1])
    real = not np.iscomplexobj(src)
    f0 = pos0 if callable(pos0) else (lambda sl: np.asarray(pos0[sl], np.float64))
    mid = np.empty((m0, n1), np.float32 if real else np.complex64)

    def put(sl, v):
        mid[:, sl] = v.T
    _run(lambda sl: np.ascontiguousarray(src[:, sl].T), put, n1, n0, real, f0, m0, kernel, taps, pad)
    return lines(mid, pos1, m1, kernel=kernel, taps=taps, pad=pad)
