"""Sidelobe control of a spotlight image after it is formed: spatially variant apodization, and the trim of the
phase history that gives the image the rectangular spectrum apodization needs.

    weighting.keystone(S, ant, fmin, df, scp, slant_e1)       # before the image is formed, unweighted
    ...                                                        # form, bring to the slant plane at twice Nyquist
    img = weighting.sva(slant['image'])

Spatially variant apodization (Stankwitz, Dallaire and Fienup, 1995) picks, at every pixel, the cosine-on-pedestal
weighting of the aperture that gives the least output: between the unweighted aperture, which keeps the full
resolution, and the Hann window, whose sidelobes are low. For an image sampled at an integer multiple N of its
Nyquist rate, every such weighting is the three-point filter y = x + w (x[-N] + x[+N]) with 0 <= w <= 1/2, so the
choice is closed form. Main lobes keep the unweighted width and sidelobes go to zero where the model holds: an
unweighted aperture whose spectrum is a rectangle on the image's axes, filling exactly 1/N of the sampled band
on each. The slant plane is where a spotlight image's spectrum is nearly that (planes.to_slant at spacings of
1 / (N band)), after keystone has trimmed the history: the azimuth band of a pulse fan grows with frequency, and
keystone drops, at each frequency, the pulses outside the azimuth band the lowest frequency reaches.

The result is not band limited and not a linear function of the scene: it is a product for detection and display.

sva takes the image where it is (a NumPy array, a CuPy array on a GPU, a JAX array on a TPU or another JAX device)
and returns the result there (fastsar._xp).
"""
import numpy as np

from . import _xp

__all__ = ['sva', 'keystone']

C = 299792458.0


def _pass(src, dst, axis, N, joint):
    """One apodization pass along `axis`, row blocks of src [n0, n1] into dst (along axis 0: another array)."""
    from .patches import _rows
    n0, n1 = src.shape

    def weigh(x, b):
        if joint:                                                   # one weight for the complex sample
            den = b.real * b.real + b.imag * b.imag
            num = x.real * b.real + x.imag * b.imag
            with np.errstate(divide='ignore', invalid='ignore'):
                w = np.where(den > 0, -num / den, 0.0)
            np.clip(w, 0.0, 0.5, out=w)
            return x + w * b
        y = np.empty_like(x)                                       # a weight for each of I and Q
        for xr, br, yr in ((x.real, b.real, y.real), (x.imag, b.imag, y.imag)):
            with np.errstate(divide='ignore', invalid='ignore'):
                w = np.where(br != 0, -xr / br, 0.0)
            np.clip(w, 0.0, 0.5, out=w)
            yr[...] = xr + w * br
        return y

    def work(sl):
        x = src[sl]
        b = np.zeros_like(x)
        if axis == 1:
            b[:, N:] += x[:, :-N]
            b[:, :-N] += x[:, N:]
        else:
            lo, hi = sl.start, sl.stop
            a0, a1 = max(lo - N, 0), min(hi - N, n0)                 # rows i - N
            if a1 > a0:
                b[a0 + N - lo:a1 + N - lo] += src[a0:a1]
            a0, a1 = min(lo + N, n0), min(hi + N, n0)                # rows i + N
            if a1 > a0:
                b[a0 - N - lo:a1 - N - lo] += src[a0:a1]
        dst[sl] = weigh(x, b)
    _rows(work, n0, int(max(1, min(256, (1 << 18) // max(n1, 1)))))


_DEVICE = {}                                          # the CUDA kernel and the jit-compiled JAX function, once made
_CUDA = '''
int r = i / n1, c = i - (ptrdiff_t)r * n1;
complex<float> v = x[i], b(0.0f, 0.0f);
if (axis == 1) {
    if (c >= N) b += x[i - N];
    if (c + N < n1) b += x[i + N];
} else {
    if (r >= N) b += x[i - (ptrdiff_t)N * n1];
    if (r + N < n0) b += x[i + (ptrdiff_t)N * n1];
}
float xr = v.real(), xi = v.imag(), br = b.real(), bi = b.imag(), wr, wi;
// products and sums rounded one by one (no fused multiply-add): the host's result to the bit
if (joint) {
    float den = __fadd_rn(__fmul_rn(br, br), __fmul_rn(bi, bi));
    float num = __fadd_rn(__fmul_rn(xr, br), __fmul_rn(xi, bi));
    wr = den > 0.0f ? -num / den : 0.0f;
    wi = wr = wr < 0.0f ? 0.0f : wr > 0.5f ? 0.5f : wr;
} else {
    wr = br != 0.0f ? -xr / br : 0.0f;
    wi = bi != 0.0f ? -xi / bi : 0.0f;
    wr = wr < 0.0f ? 0.0f : wr > 0.5f ? 0.5f : wr;
    wi = wi < 0.0f ? 0.0f : wi > 0.5f ? 0.5f : wi;
}
y = complex<float>(__fadd_rn(xr, __fmul_rn(wr, br)), __fadd_rn(xi, __fmul_rn(wi, bi)));
'''


def _sva_cupy(img, N, axes, joint, inplace):
    """sva on a GPU: one kernel call a pass, each into another array (the result and one scratch array; in place,
    the image and one scratch array)."""
    import cupy as cp
    if 'cupy' not in _DEVICE:
        _DEVICE['cupy'] = cp.ElementwiseKernel('raw complex64 x, int32 n0, int32 n1, int32 N, int32 axis, bool joint',
                                               'complex64 y', _CUDA, 'fastsar_sva')
    x = img if img.dtype == cp.complex64 else img.astype(cp.complex64)
    inplace = inplace or x is not img
    src = cp.ascontiguousarray(x)
    if not axes:
        return x if inplace else x.copy()
    cur, free = src, [src] if inplace or src is not x else []      # free: the arrays a pass may write to
    for ax in axes:
        dst = next((a for a in free if a is not cur), None)
        if dst is None:
            dst = cp.empty_like(src)
            free.append(dst)
        _DEVICE['cupy'](cur, src.shape[0], src.shape[1], N, ax, joint, dst)
        cur = dst
    if not inplace:
        return cur
    if cur is not x:
        x[...] = cur
    return x


def _sva_jax(jnp, N, axes, joint, x):
    def weigh(x, b):
        if joint:
            den = b.real * b.real + b.imag * b.imag
            num = x.real * b.real + x.imag * b.imag
            return x + jnp.clip(jnp.where(den > 0, -num / den, 0.0), 0.0, 0.5) * b
        from jax import lax
        return lax.complex(*(xr + jnp.clip(jnp.where(br != 0, -xr / br, 0.0), 0.0, 0.5) * br
                             for xr, br in ((x.real, b.real), (x.imag, b.imag))))
    for ax in axes:
        if N >= x.shape[ax]:                                        # no neighbor inside the image: unchanged
            continue
        lo, hi = ((slice(None), slice(None, -N)), (slice(None), slice(N, None)))
        pad = ((0, 0), (N, 0)), ((0, 0), (0, N))
        if ax == 0:
            lo, hi, pad = lo[::-1], hi[::-1], (pad[0][::-1], pad[1][::-1])
        x = weigh(x, jnp.pad(x[lo], pad[0]) + jnp.pad(x[hi], pad[1]))
    return x


def sva(img, oversample=2, *, axes=(1, 0), joint=True, inplace=False):
    """Spatially variant apodization of a complex image sampled at `oversample` times its Nyquist rate on the axes
    treated (an integer: 2 for an image at spacings of half the reciprocal of its bandwidth), one axis after the
    other in the order of `axes` (default range, then azimuth, of an image [azimuth, range]).

    joint: one weight per complex sample (the I and Q channels apodized together, which keeps the phase of a
    main lobe); False: I and Q each on their own, the original form, with lower sidelobes in clutter and a result
    whose phase is not the image's. Samples beyond the edges count as 0, and a pixel that is 0 stays 0.
    inplace: the result is written into img (a pass along axis 0 still takes a scratch array of the image's size).
    img: a NumPy array, a CuPy array or a JAX array; the result is of the same kind, on the same device. A JAX
    array cannot be written to: inplace does nothing there and the result is a new array.
    -> complex64 [n0, n1]."""
    N = int(oversample)
    if N < 1 or N != oversample:
        raise ValueError(f'oversample must be a positive integer (the image at that multiple of Nyquist), got {oversample!r}')
    kind = _xp.kind(img)
    x = np.asarray(img) if kind == 'numpy' else img
    if x.ndim != 2 or not _xp.module(x).iscomplexobj(x):
        raise ValueError('sva takes a complex image [n0, n1]')
    axes = tuple(int(a) % 2 for a in np.atleast_1d(axes))
    if kind == 'cupy':
        return _sva_cupy(x, N, axes, bool(joint), inplace)
    if kind == 'jax':
        if 'jax' not in _DEVICE:
            import functools
            import jax
            import jax.numpy as jnp
            _DEVICE['jax'] = jax.jit(functools.partial(_sva_jax, jnp), static_argnums=(0, 1, 2))
        return _DEVICE['jax'](N, axes, bool(joint), x.astype('complex64'))
    if x.dtype != np.complex64:
        x, inplace = x.astype(np.complex64), True
    if not axes:
        return x if inplace else x.copy()
    cur, spare = x, None                              # a pass along axis 1 can run in place; one along axis 0 cannot
    for ax in axes:
        if ax == 1 and (inplace or cur is not x):
            _pass(cur, cur, 1, N, joint)
            continue
        if cur is x:
            dst = np.empty_like(x) if spare is None else spare
        else:
            dst = x if inplace else (np.empty_like(x) if spare is None else spare)
        _pass(cur, dst, ax, N, joint)
        cur, spare = dst, (cur if cur is not x else spare)
    if inplace and cur is not x:
        x[...] = cur
        return x
    return cur


def keystone(S, ant, fmin, df, scp, e1, symmetric=True):
    """Trim a spotlight phase history S [P, K] (in place) to a rectangle of the slant plane's spatial frequencies.

    Pulse p at frequency f lies at the azimuth spatial frequency (2 f / c) s_p, s_p the component along the slant
    plane's azimuth axis e1 [3] of the unit vector from the antenna (ant [P, 3]) to the scene center scp [3]. The
    samples kept are those inside the azimuth band the first frequency spans, fmin * [min s, max s]: at frequency f
    the pulses with f s_p in it. symmetric: the band is cut to the same width on both sides of zero (the smaller
    of the two), which centers it as apodization assumes.
    -> the fraction of the samples set to zero, and the azimuth band kept (least, greatest; cycles/m)."""
    P, K = S.shape
    u = np.asarray(scp, np.float64)[None] - np.asarray(ant, np.float64)
    s = (u / np.linalg.norm(u, axis=1, keepdims=True)) @ np.asarray(e1, np.float64)
    hi, lo = s.max(), s.min()
    if symmetric:
        hi = min(hi, -lo)
        lo = -hi
    lim = np.where(s > 0, hi, lo)
    with np.errstate(divide='ignore', invalid='ignore'):
        fcut = np.where(s != 0, fmin * lim / s, np.inf)              # the last frequency kept of each pulse
    fcut = np.where(fcut < fmin, -np.inf, fcut)                       # a pulse outside the band at every frequency
    kcut = np.where(np.isfinite(fcut), np.floor((np.minimum(fcut, fmin + K * df) - fmin) / df + 1e-9), K - 1)
    kcut = np.where(np.isneginf(fcut), -1, kcut)
    kcut = np.clip(kcut.astype(np.int64) + 1, 0, K)
    for p in np.nonzero(kcut < K)[0]:
        S[p, kcut[p]:] = 0
    return float((K - kcut).sum() / (P * K)), (2 * fmin / C * lo, 2 * fmin / C * hi)
