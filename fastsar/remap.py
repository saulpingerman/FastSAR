"""Remaps from complex or amplitude pixels to display values, matching the remap functions of the NGA MATLAB SAR
Toolbox (Visualization/remap/remap_funcs): density, brighter, darker, highcontrast, pedf, nrl, log and linear.

    img8 = fastsar.remap.nrl(image)                     # uint8, as nrlremap.m
    img8 = fastsar.remap.remap(image, 'density')        # by name
    img8 = fastsar.remap.nrl(image, valid=mask)         # statistics over the valid pixels only

Each function follows its .m file line for line: the arithmetic is single precision where MATLAB's is, and the
conversion to uint8 rounds halves away from zero and saturates, as MATLAB's uint8() does. Run in Octave on the same
pixels, the toolbox's functions give the same counts, except where a value sits on a rounding boundary (a few pixels
in 100,000 for the density family, up to one in 1,000 for log, never by more than one count): the toolbox sums
millions of single-precision values for its means, and such a sum depends on the order of its terms in the last
digits. The sums here are accumulated in double precision.

The toolbox computes its statistics (mean, minimum, maximum, 99th percentile) over every pixel. An image that is
zero outside its processed area would pull those statistics down, so each remap takes `valid`, a boolean mask of the
pixels that enter the statistics; the others come out 0. Without it the result is the toolbox's.

    density        log amplitude between 0.8 and 32 times the mean amplitude, on 30..255 (amplitudetodensity)
    brighter       the same on 60..255
    darker         the same on 0..255
    highcontrast   log amplitude between 0.8 and 3.2 times the mean amplitude, on 30..255
    pedf           density with the upper half (above 128) compressed by two (piecewise extended density format)
    nrl            linear up to the 99th percentile (output 0..220), logarithmic from there to the maximum (220..255)
    log            decibels over a 50 dB span centered on the mean power
    linear         the amplitude itself (floating point, not scaled)

Where the image lives. The image may be a NumPy array, a CuPy array or a JAX array, and the result comes back in the
same place (fastsar._xp); from a device only scalars, partial sums and histogram counts cross to the host. On the
host the image is taken in row blocks on the thread pool of fastsar.patches. The statistics are the host's on every
device: the 99th percentile from the same order statistics, which two histogram passes over the bit patterns of the
amplitudes find without a sort (on a TPU, which adds into indexed bins slowly, a search by counts of the pixels
below thresholds: FASTSAR_REMAP_JAX=count|histogram), and the means from float64 sums on a GPU. JAX works in single
precision throughout,
as a TPU must: its sums are float32 sums over blocks of 128 pixels, and of 128 of those, whose few thousand partial
sums are added in float64 on the host. A device's amplitude and logarithm differ from NumPy's in the last bit, so a
display value on a rounding boundary can come out one count away: a few pixels in a million (one in 100,000 for
log, measured on 64 million pixels of speckle), never by more than one count. `linear` of a complex JAX image is
float32.
"""
import functools

import numpy as np

from . import _xp

__all__ = ['density', 'brighter', 'darker', 'highcontrast', 'pedf', 'nrl', 'log', 'linear', 'remap', 'REMAPS',
           'amplitude_to_density', 'percentile']

_f = np.float32
_TOP = 1 << 16                                                    # bins of a histogram pass: 16 bits of the key
_SLOTS = 4                                                        # first-pass bins a second pass resolves


def _uint8(x):
    """MATLAB's uint8(): round to nearest with halves away from zero, saturate to 0..255, NaN to 0."""
    x = np.asarray(x)
    out = np.zeros(x.shape, np.uint8)
    ok = ~np.isnan(x)
    np.floor(np.clip(x, 0.0, 255.0, where=ok, out=np.zeros(x.shape, x.dtype)) + x.dtype.type(0.5), out=out, casting='unsafe')
    return out


# ---------------------------------------------------------------------------------------------------- where it runs
# A step is a function f(xp, x, valid, *scalars) of array operations that NumPy, CuPy and jax.numpy share. On the
# host it runs on row blocks, on a JAX device as one jit-compiled function (compiled once per image shape; the
# scalars are arguments, not constants), on a GPU as it is or as the fused kernel of _cupy_steps.

_JIT = {}


def _jit(step, *static):
    key = (step,) + static
    if key not in _JIT:
        import jax
        import jax.numpy as jnp
        _JIT[key] = jax.jit(functools.partial(step, jnp, *static))
    return _JIT[key]


def _blocked(step, x, valid, *static):
    """f(*args) -> the results of step(jnp, *static, block of x, block of valid, *args) for the blocks of a JAX image
    taken as one line of pixels, or None when the image is one block. A block holds _xp.jax_block() bytes at 48 a
    pixel and a multiple of 2^14 pixels, so that _psum's partial sums are the whole image's; the last holds the
    rest (one compilation for each of the two sizes)."""
    n = x.size
    c = max(1, _xp.jax_block() // 48 // 16384) * 16384
    if n <= c:
        return None
    key = (step, 'blocks') + static
    if key not in _JIT:
        import jax
        import jax.numpy as jnp
        from jax import lax

        def block(x, valid, a, *args, size):
            cut = lambda v: lax.dynamic_slice(v.reshape(-1), (a,), (size,))
            return step(jnp, *static, cut(x), None if valid is None else cut(valid), *args)
        _JIT[key] = jax.jit(block, static_argnames='size')
    if valid is not None and valid.shape != x.shape:
        valid = _xp.module(x).broadcast_to(valid.reshape(valid.shape + (1,) * (x.ndim - valid.ndim)), x.shape)
    return lambda *args: [_JIT[key](x, valid, np.int32(a), *args, size=min(c, n - a)) for a in range(0, n, c)]


def _args(x, valid):
    """x and the mask as arrays where x lives (the mask boolean, of x's shape)."""
    if _xp.kind(x) != 'numpy':
        return x, (None if valid is None else _xp.module(x).asarray(valid, bool))
    x = np.asarray(x)
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != x.shape:                                 # a mask of the leading axes, as indexing takes it
            valid = np.broadcast_to(valid.reshape(valid.shape + (1,) * (x.ndim - valid.ndim)), x.shape)
    return x, valid


def _each(x, work):
    """work(slice) over row blocks of x of about 2^18 pixels, on the thread pool."""
    from .patches import _rows
    _rows(work, x.shape[0], max(1, (1 << 18) // max(x[:1].size, 1)))


def _on_device(step, x):
    if _xp.kind(x) == 'jax':
        return _jit(step)
    import cupy
    return _cupy_steps().get(step) or functools.partial(step, cupy)


def _apply(step, dtype, x, valid, *args):
    """The array a step gives, where x lives. dtype: of the result (for the host's output array)."""
    if _xp.kind(x) == 'jax':
        parts = _blocked(step, x, valid)
        if parts is not None:
            import jax.numpy as jnp
            return jnp.concatenate(parts(*args)).reshape(x.shape)
    if _xp.kind(x) != 'numpy':
        return _on_device(step, x)(x, valid, *args)
    if x.ndim == 0:
        return _apply(step, dtype, x[None], None if valid is None else valid[None], *args).reshape(())
    out = np.empty(x.shape, dtype)

    def work(sl):
        out[sl] = step(np, x[sl], None if valid is None else valid[sl], *args)
    _each(x, work)
    return out


def _reduce(step, x, valid, *args):
    """The statistics of a step that returns (partial sums, minimum, maximum): the sums as Python floats (added in
    float64 on the host), the minimum and maximum as float32 (None where the step has none)."""
    if _xp.kind(x) != 'numpy':
        parts = _blocked(step, x, valid) if _xp.kind(x) == 'jax' else None
        parts = [_on_device(step, x)(x, valid, *args)] if parts is None else parts(*args)
        sums = [float(np.concatenate([np.ravel(_xp.host(p[0][i])) for p in parts]).astype(np.float64).sum())
                for i in range(len(parts[0][0]))]
        return sums, *(None if parts[0][j] is None else _f(fn([_xp.host(p[j]) for p in parts]))
                       for j, fn in ((1, np.min), (2, np.max)))
    x, valid = np.atleast_1d(x), (None if valid is None else np.atleast_1d(valid))
    parts = {}

    def work(sl):
        parts[sl.start] = step(np, x[sl], None if valid is None else valid[sl], *args)
    _each(x, work)
    parts = [parts[i] for i in sorted(parts)]
    sums = [float(np.sum([p[0][i] for p in parts])) for i in range(len(parts[0][0]))]
    return sums, *(None if parts[0][j] is None else fn([p[j] for p in parts]) for j, fn in ((1, np.min), (2, np.max)))


def _count(x, valid):
    return x.size if valid is None else int(_xp.module(valid).count_nonzero(valid))


def _amp(xp, x):
    return xp.abs(x).astype(xp.float32, copy=False)


def _psum(xp, a):
    """Sums of parts of a whose total, added in float64 on the host, is the sum of a: the float64 sum itself with
    NumPy and CuPy; with JAX, float32 sums of blocks of 128 values, then of blocks of 128 of those (a float32 sum
    of 128 terms loses its last 3 or 4 bits at most, and the errors of many blocks average out). Each block is
    added by halves, pair by pair: a reduction's order of additions is the compiler's and changes with the shape
    of the array, and the partial sums of an image taken in blocks (_blocked) must be the whole image's."""
    if xp.__name__ != 'jax.numpy':
        return a.sum(dtype=xp.float64)
    a = a.reshape(-1)
    for _ in range(2):
        a = xp.pad(a, (0, -a.size % 128)).reshape(-1, 128)
        while a.shape[1] > 1:
            a = a[:, 0::2] + a[:, 1::2]
        a = a[:, 0]
    return a


def _u8(xp, y):
    if xp is np:
        return _uint8(y)
    return xp.where(xp.isnan(y), 0, xp.floor(xp.clip(y, 0, 255) + _f(0.5))).astype(xp.uint8)


def _zero(xp, out, valid):
    if valid is None:
        return out
    if xp is np:
        out[~valid] = 0
        return out
    return xp.where(valid, out, xp.zeros((), out.dtype))


_CUPY = {}
_CUDA = '''static __device__ unsigned char u8(float d) {
    return d != d ? 0 : (unsigned char)floorf((d < 0.0f ? 0.0f : d > 255.0f ? 255.0f : d) + 0.5f); }
static __device__ unsigned int key(float a) {
    unsigned int b = __float_as_uint(a);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }'''


def _cupy_steps():
    """Fused CUDA kernels for the steps of the density family and of nrl, by step (compiled at their first call)."""
    if _density_u8 not in _CUPY:
        import cupy as cp
        dens = cp.ElementwiseKernel(
            'T x, bool v, float32 m, float32 b, bool pedf', 'uint8 y',
            '''float a = (float)abs(x);
            float d = m * log10f(a < 1e-5f ? 1e-5f : a) + b;
            if (pedf && d > 128.0f) d = 0.5f * (d + 128.0f);
            y = v ? u8(d) : 0;''', 'fastsar_density', preamble=_CUDA)
        nrl_ = cp.ElementwiseKernel(
            'T x, bool v, float32 amin, float32 thr, float32 knee, float32 b, float32 c', 'uint8 y',
            '''float a = (float)abs(x);
            y = v ? u8(a <= thr ? (a - amin) * c / knee : b * log10f((a - amin) / knee) + c) : 0;''',
            'fastsar_nrl', preamble=_CUDA)
        # the sum and the count of the finite valid amplitudes, of 1024 pixels a thread (a ReductionKernel takes
        # ten times as long)
        mean = cp.ElementwiseKernel(
            'raw T x, raw bool v, bool all, int64 n', 'float64 s, float64 c',
            '''double t = 0.0;
            long long k = 0, j = (long long)i * 1024, end = j + 1024 < n ? j + 1024 : n;
            for (; j < end; ++j) {
                float a = (float)abs(x[j]);
                if ((all || v[j]) && isfinite(a)) { t += (double)a; ++k; }
            }
            s = t; c = (double)k;''', 'fastsar_mean')

        def parts(x, v):
            return mean(x, cp.zeros(1, bool) if v is None else v, v is None, x.size,
                        size=-(-x.size // 1024)), None, None
        _CUPY.update({_density_u8: lambda x, v, m, b: dens(x, v is None or v, m, b, False),
                      _pedf_u8: lambda x, v, m, b: dens(x, v is None or v, m, b, True),
                      _nrl_u8: lambda x, v, *a: nrl_(x, v is None or v, *a), _mean_parts: parts})
    return _CUPY


# ------------------------------------------------------------------------------------------------- order statistics

def _key(xp, a):
    """uint32 keys of float32 values, in the order of the values (NaN last)."""
    if xp is np:
        b = a.view(np.uint32)
    else:
        from jax import lax
        b = lax.bitcast_convert_type(a, xp.uint32)
    return xp.where(b >> 31 != 0, ~b, b | xp.uint32(1 << 31))


def _hist_jax(xp, amp, x, valid, lut):
    u = _key(xp, (_amp(xp, x) if amp else x.astype(xp.float32)).reshape(-1))
    idx, nb = (u >> 16).astype(xp.int32), _TOP
    if lut is not None:
        slot, nb = lut[idx], _SLOTS * _TOP
        idx = xp.where(slot >= 0, slot * _TOP + (u & 0xffff).astype(xp.int32), nb)
    if valid is not None:
        idx = xp.where(valid.reshape(-1), idx, nb)
    return xp.zeros(nb + 1, xp.int32).at[idx].add(1)[:nb]


def _hist(x, valid, lut, amp=True):
    """Counts (int64, on the host) of the float32 amplitudes of x (amp False: of its real values) over the valid
    pixels, by key (_key): by the 2^16 values of the key's high half; or, given lut [2^16] (int32: a slot
    0.._SLOTS - 1 for some of those values, -1 for the rest), by slot * 2^16 + the key's low half."""
    k, nb = _xp.kind(x), _TOP if lut is None else _SLOTS * _TOP
    if k == 'jax':
        import jax.numpy as jnp
        lut = None if lut is None else jnp.asarray(lut)
        parts = _blocked(_hist_jax, x, valid, bool(amp))
        if parts is not None:
            return sum(np.asarray(h, np.int64) for h in parts(lut))
        return np.asarray(_jit(_hist_jax, bool(amp))(x, valid, lut), np.int64)
    if k == 'cupy':
        import cupy as cp
        name = ('amp' if amp else 'val') + ('1' if lut is None else '2')
        if name not in _CUPY:
            _CUPY[name] = cp.ElementwiseKernel(
                'T x, bool v' + ('' if lut is None else ', raw int32 lut'), 'raw int32 h',
                'if (v) { unsigned int u = key((float)%s); %s }' % (
                    'abs(x)' if amp else 'x', 'atomicAdd(&h[u >> 16], 1);' if lut is None else
                    f'int s = lut[u >> 16]; if (s >= 0) atomicAdd(&h[s * {_TOP} + (u & 0xffffu)], 1);'),
                'fastsar_hist_' + name, preamble=_CUDA)
        h = cp.zeros(nb, cp.int32)
        _CUPY[name](x, valid is None or valid, *(() if lut is None else (cp.asarray(lut),)), h)
        return h.get().astype(np.int64)
    import threading
    x, valid = np.atleast_1d(x), (None if valid is None else np.atleast_1d(valid))
    h, lock = np.zeros(nb, np.int64), threading.Lock()

    def work(sl):
        a = _amp(np, x[sl]) if amp else x[sl].astype(np.float32)
        u = _key(np, a.ravel() if valid is None else a[valid[sl]])
        if lut is None:
            c = np.bincount(u >> 16, minlength=nb)
        else:
            slot = lut[u >> 16]
            pick = slot >= 0
            if not pick.any():
                return
            c = np.bincount(slot[pick].astype(np.int64) * _TOP + (u[pick] & 0xffff), minlength=nb)
        with lock:
            h[...] += c
    _each(x, work)
    return h


def _below_jax(xp, amp, x, valid, thr):
    """The number (int32 [len(thr)]) of valid pixels whose key (_key, of the float32 amplitude, or of the value when
    amp is False) is below each key of thr (uint32)."""
    u = _key(xp, _amp(xp, x) if amp else x.astype(xp.float32))
    return xp.stack([((u < thr[i]) if valid is None else (u < thr[i]) & valid).sum(dtype=xp.int32)
                     for i in range(thr.shape[0])])


def _jax_statistics():
    """'count' or 'histogram', the way a JAX image's order statistics are found: FASTSAR_REMAP_JAX, else count on a
    TPU (which adds into indexed bins slowly, about 1e8 pixels a second, and compares and sums fast) and histogram
    on JAX's other devices."""
    import os
    v = os.environ.get('FASTSAR_REMAP_JAX', 'auto').strip().lower() or 'auto'
    if v not in ('auto', 'count', 'histogram'):
        raise ValueError(f"FASTSAR_REMAP_JAX must be 'count', 'histogram' or 'auto', got {v!r}")
    if v == 'auto':
        import jax
        v = 'count' if jax.devices()[0].platform == 'tpu' else 'histogram'
    return v


def _order(x, valid, amp=True):
    """n and take, as _ranks gives them, for the float32 amplitudes of x (amp False: its values) over the valid
    pixels, where x lives."""
    if _xp.kind(x) == 'jax' and _jax_statistics() == 'count':
        import jax.numpy as jnp
        parts, whole = _blocked(_below_jax, x, valid, bool(amp)), _jit(_below_jax, bool(amp))
        below = (lambda thr: [whole(x, valid, thr)]) if parts is None else parts
        return _ranks_count(lambda thr: sum(np.asarray(c, np.int64) for c in below(jnp.asarray(thr))), _count(x, valid))
    return _ranks(lambda lut: _hist(x, valid, lut, amp))


def _ranks_count(below, n, radix=16):
    """_ranks without histograms, from counts alone: below(thr) -> the number of values whose key is under each key
    of thr (uint32 [16] or [48]); n: the number of values. The keys are searched from the whole range down: a
    range of keys that holds a rank asked for is cut in `radix` parts, the counts below the cuts say which parts
    hold such ranks, and so on until a part is one key (8 passes for one group of neighboring ranks).
    -> n, take(*ranges), as _ranks'."""
    def take(*ranges):
        cells, found = [(0, 1 << 32, 0, n)], []                    # keys a ... b - 1 hold the ranks ca ... cb - 1
        while cells:
            edges = [[a + (b - a) * i // min(radix, b - a) for i in range(1, min(radix, b - a))]
                     for a, b, _, _ in cells]
            flat = np.array([t for e in edges for t in e], np.uint32)
            size = 16 if len(flat) <= 16 else 48                   # two compiled sizes, a short call filled by repeats
            cnt = np.concatenate([below(np.resize(flat[i:i + size], size))[:len(flat[i:i + size])]
                                  for i in range(0, len(flat), size)])
            cells, at, last = [], 0, cells
            for (a, b, ca, cb), e in zip(last, edges):
                es, cs = [a, *e, b], [ca, *cnt[at:at + len(e)].tolist(), cb]
                at += len(e)
                for i in range(len(es) - 1):
                    if cs[i + 1] > cs[i] and any(cs[i] < hi and cs[i + 1] > lo for lo, hi in ranges):
                        (found if es[i + 1] - es[i] == 1 else cells).append((es[i], es[i + 1], cs[i], cs[i + 1]))
        found.sort()
        keys, end = np.array([c[0] for c in found], np.uint32), np.array([c[3] for c in found], np.int64)
        out = []
        for lo, hi in ranges:
            u = keys[np.searchsorted(end, np.arange(lo, hi), 'right')]
            out.append(np.where(u >> 31 != 0, u & np.uint32(0x7fffffff), ~u).view(np.float32))
        return out
    return n, take


def _ranks(hist):
    """Exact order statistics of float32 values from two passes of histograms, without a sort and with only the
    counts on the host. hist(lut) -> counts as _hist gives them. The first pass counts the values by the high half
    of their keys; a second resolves the low half inside the first-pass bins that hold the ranks asked for
    (_SLOTS of them a pass: more passes only when the ranks spread over more bins than that).
    -> n (the number of values), take(*ranges): for each range (lo, hi), the sorted values of ranks lo..hi - 1
    (float32)."""
    c = hist(None)
    cum = np.cumsum(c)
    n = int(cum[-1])

    def take(*ranges):
        span = [tuple(int(b) for b in np.searchsorted(cum, (lo, hi - 1), 'right')) for lo, hi in ranges]
        need = sorted({b for b0, b1 in span for b in range(b0, b1 + 1) if c[b]})
        fine = {}
        for i in range(0, len(need), _SLOTS):
            lut = np.full(_TOP, -1, np.int32)
            lut[need[i:i + _SLOTS]] = np.arange(len(need[i:i + _SLOTS]))
            fine.update(zip(need[i:i + _SLOTS], hist(lut).reshape(_SLOTS, _TOP)))
        out = []
        for (lo, hi), (b0, b1) in zip(ranges, span):
            bins = [b for b in range(b0, b1 + 1) if c[b]]
            cnt = np.concatenate([fine[b] for b in bins])
            u = np.concatenate([np.uint32(b << 16) + np.arange(_TOP, dtype=np.uint32) for b in bins])[cnt > 0]
            last = np.cumsum(cnt[cnt > 0]) + (cum[b0 - 1] if b0 else 0)      # one past the last rank of each value
            u = u[np.searchsorted(last, np.arange(lo, hi), 'right')]
            out.append(np.where(u >> 31 != 0, u & np.uint32(0x7fffffff), ~u).view(np.float32))
        return out
    return n, take


def _window(n, p, halfwidth):
    """The fractional index of the p-th percentile among n sorted values, and the ranks lo..hi - 1 its spline is
    fitted to."""
    t = p * 0.01 * n - 0.5
    lo = int(np.clip(np.floor(t) - halfwidth, 0, max(n - 2 * halfwidth, 0)))
    return t, lo, min(n, lo + 2 * halfwidth)


def _spline(s, lo, t, dtype):
    from scipy.interpolate import CubicSpline
    s = s.astype(np.float64)
    if s.size < 2 or np.all(s == s[0]):
        return dtype(s[0])
    return dtype(CubicSpline(np.arange(lo, lo + s.size), s, bc_type='not-a-knot', extrapolate=True)(t))


def percentile(v, p, halfwidth=32):
    """The toolbox's percentile (pctl in nrlremap.m): a not-a-knot cubic spline through the sorted values, the i-th
    of n placed at (i - 0.5) / n, evaluated at p / 100. The spline is fitted to the 2 halfwidth order statistics
    around the query, which reproduces the spline through all of them (a spline's dependence on distant knots falls
    by a factor of about 3.7 per knot). v on a device (a CuPy or JAX array): taken as float32, its order statistics
    found there from histograms (they are the ones a sort gives); the result is a float32 scalar on the host."""
    if _xp.kind(v) != 'numpy':
        n, take = _order(v, None, amp=False)
        if n == 0:
            raise ValueError('percentile of no values')
        t, lo, hi = _window(n, p, halfwidth)
        return _spline(take((lo, hi))[0], lo, t, _f)
    v = np.ravel(np.asarray(v))
    n = v.size
    if n == 0:
        raise ValueError('percentile of no values')
    if n == 1:
        return v[0]
    t, lo, hi = _window(n, p, halfwidth)
    s = np.sort(np.partition(v, (lo, hi - 1))[lo:hi]) if n > 2 * halfwidth else np.sort(v)
    return _spline(s, lo, t, v.dtype.type)


# ------------------------------------------------------------------------------------------------------- the remaps

def _mean_parts(xp, x, valid):
    """The sum and the count of the finite amplitudes."""
    a = _amp(xp, x)
    ok = xp.isfinite(a) if valid is None else xp.isfinite(a) & valid
    return (_psum(xp, xp.where(ok, a, _f(0))), _psum(xp, ok.astype(xp.float32))), None, None


def _density_d(xp, x, valid, m, b):
    return m * xp.log10(xp.maximum(_amp(xp, x), _f(1e-5))) + b


def _density_u8(xp, x, valid, m, b):
    return _zero(xp, _u8(xp, _density_d(xp, x, valid, m, b)), valid)


def _pedf_u8(xp, x, valid, m, b):
    d = _density_d(xp, x, valid, m, b)
    return _zero(xp, _u8(xp, xp.where(d > 128, _f(0.5) * (d + _f(128)), d)), valid)


def _scale(x, valid, dmin, mmult, mean):
    """m and b of the density m log10(amplitude) + b."""
    f = _f
    if mean is None:
        s, n = _reduce(_mean_parts, x, valid)[0] if x.size else (0.0, 0.0)
        mean = s / n if n else 1.0
    cl = f(0.8) * f(mean)
    ch = f(mmult) * cl
    m = f(255.0 - dmin) / np.log10(ch / cl, dtype=f)
    return m, f(dmin) - m * np.log10(cl, dtype=f)


def amplitude_to_density(x, dmin=30.0, mmult=40.0, mean=None, valid=None):
    """amplitudetodensity.m: density values (float32, not clipped) of the amplitude of x. The amplitude 0.8 times
    the mean maps to dmin and mmult times that to 255, logarithmically. mean: the amplitude's mean, when it is
    known (default: over the finite pixels, or the finite valid ones)."""
    x, valid = _args(x, valid)
    return _apply(_density_d, np.float32, x, valid, *_scale(x, valid, dmin, mmult, mean))


def _density(dmin, mmult, doc):
    def fn(x, valid=None, mean=None):
        x, valid = _args(x, valid)
        return _apply(_density_u8, np.uint8, x, valid, *_scale(x, valid, dmin, mmult, mean))
    fn.__doc__ = doc
    return fn


density = _density(30.0, 40.0, 'densityremap.m: uint8 density, 30 at 0.8 times the mean amplitude, 255 at 32 times it.')
brighter = _density(60.0, 40.0, 'brighterremap.m: as density, on 60..255.')
darker = _density(0.0, 40.0, 'darkerremap.m: as density, on 0..255.')
highcontrast = _density(30.0, 4.0, 'highcontrastremap.m: uint8 density, 30 at 0.8 times the mean amplitude, 255 at 3.2 times it.')


def pedf(x, valid=None, mean=None):
    """pedfremap.m: piecewise extended density format, density with the values above 128 halved toward 128."""
    x, valid = _args(x, valid)
    return _apply(_pedf_u8, np.uint8, x, valid, *_scale(x, valid, 30.0, 40.0, mean))


def _linear(xp, x, valid):
    if xp.iscomplexobj(x):
        out = xp.abs(x) if xp.__name__ == 'jax.numpy' else xp.abs(x.astype(xp.complex128))
    else:
        out = x.copy()
    return _zero(xp, out, valid)


def linear(x, valid=None):
    """linearremap.m: the amplitude of a complex image, a real image unchanged. Floating point, not scaled (float64
    for a complex image; float32 for a complex JAX image, a TPU having no float64)."""
    x, valid = _args(x, valid)
    return _apply(_linear, np.float64 if np.iscomplexobj(x) else x.dtype, x, valid)


def _log_parts(xp, x, valid):
    """The sum, the minimum and the maximum of the amplitude."""
    a = _amp(xp, x)
    if valid is None:
        return (_psum(xp, a),), a.min(), a.max()
    return (_psum(xp, xp.where(valid, a, _f(0))),), xp.where(valid, a, _f(np.inf)).min(), \
        xp.where(valid, a, _f(-np.inf)).max()


def _log_db(xp, x, mean, off):
    a = _f(10) * _amp(xp, x) / mean
    return a, _f(20) * xp.log10(a + off)


def _log_parts2(xp, x, valid, mean, off):
    """The sum of the squares of the scaled amplitude, and the minimum and the maximum of its decibels."""
    a, db = _log_db(xp, x, mean, off)
    sq = (a if xp.__name__ == 'jax.numpy' else a.astype(xp.float64)) ** 2
    if valid is None:
        return (_psum(xp, sq),), db.min(), db.max()
    return (_psum(xp, xp.where(valid, sq, 0)),), xp.where(valid, db, _f(np.inf)).min(), \
        xp.where(valid, db, _f(-np.inf)).max()


def _log_u8(xp, x, valid, mean, off, lo, hi):
    return _zero(xp, _u8(xp, _f(255) * (_log_db(xp, x, mean, off)[1] - lo) / (hi - lo)), valid)


def log(x, valid=None, span_db=50.0):
    """logremap.m: decibels on a span of span_db centered on the mean power, 0..255."""
    f = _f
    x, valid = _args(x, valid)
    n = _count(x, valid)
    if n == 0:
        raise ValueError('log remap of no pixels')
    (s,), amin, _ = _reduce(_log_parts, x, valid)
    mean = f(s / n)
    # the least scaled amplitude: scaling keeps the order, and one infinite pixel makes it NaN (inf / inf)
    off = max(f(1) - (f(10) * amin / mean if np.isfinite(mean) else f(np.nan)), f(0))
    (s,), lo, hi = _reduce(_log_parts2, x, valid, mean, off)
    rcent = f(10) * np.log10(f(s / n))
    lo = max(lo, rcent - f(span_db / 2))
    hi = min(hi, rcent + f(span_db / 2))
    return _apply(_log_u8, np.uint8, x, valid, mean, off, lo, hi)


def _nrl_u8(xp, x, valid, amin, thr, knee, b, c):
    A = _amp(xp, x)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = xp.where(A <= thr, (A - amin) * c / knee, b * xp.log10((A - amin) / knee) + c)
    return _zero(xp, _u8(xp, y), valid)


def nrl(x, a=1.0, c=220.0, valid=None):
    """nrlremap.m: linear from the minimum (0) to a times the 99th percentile (c), logarithmic from there to the
    maximum (255). a: scale of the 99th percentile; c: the output value where the logarithmic part starts."""
    f = _f
    x, valid = _args(x, valid)
    n, take = _order(x, valid)
    if n == 0:
        raise ValueError('nrl remap of no pixels')
    t, lo, hi = _window(n, 99, 32)
    (amin,), s, (amax,) = take((0, 1), (lo, hi), (n - 1, n))
    if np.isnan(amax):                                             # a NaN pixel: the minimum is NaN as well
        amin = amax
    thr = f(a) * (amin if n == 1 else _spline(s, lo, t, f))
    knee = thr - amin
    with np.errstate(divide='ignore', invalid='ignore'):
        b = f(255 - c) / np.log10((amax - amin) / knee)
    return _apply(_nrl_u8, np.uint8, x, valid, amin, thr, knee, b, f(c))


REMAPS = dict(density=density, brighter=brighter, darker=darker, highcontrast=highcontrast, pedf=pedf, nrl=nrl,
              log=log, linear=linear)


def remap(x, name='nrl', **kw):
    """The remap called `name` (a key of REMAPS, the toolbox's names without 'remap') applied to x."""
    try:
        fn = REMAPS[str(name).lower().removesuffix('remap')]
    except KeyError:
        raise ValueError(f'remap must be one of {sorted(REMAPS)}, got {name!r}') from None
    return fn(x, **kw)
