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
"""
import numpy as np

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


def sva(img, oversample=2, *, axes=(1, 0), joint=True, inplace=False):
    """Spatially variant apodization of a complex image sampled at `oversample` times its Nyquist rate on the axes
    treated (an integer: 2 for an image at spacings of half the reciprocal of its bandwidth), one axis after the
    other in the order of `axes` (default range, then azimuth, of an image [azimuth, range]).

    joint: one weight per complex sample (the I and Q channels apodized together, which keeps the phase of a
    main lobe); False: I and Q each on their own, the original form, with lower sidelobes in clutter and a result
    whose phase is not the image's. Samples beyond the edges count as 0, and a pixel that is 0 stays 0.
    inplace: the result is written into img (a pass along axis 0 still takes a scratch array of the image's size).
    -> complex64 [n0, n1]."""
    N = int(oversample)
    if N < 1 or N != oversample:
        raise ValueError(f'oversample must be a positive integer (the image at that multiple of Nyquist), got {oversample!r}')
    x = np.asarray(img)
    if x.ndim != 2 or not np.iscomplexobj(x):
        raise ValueError('sva takes a complex image [n0, n1]')
    axes = tuple(int(a) % 2 for a in np.atleast_1d(axes))
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
