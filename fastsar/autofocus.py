"""Phase gradient autofocus (PGA) for spotlight images, and autofocus of a phase history through it.

    img, phi_p = fastsar.autofocus.autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, backend='cpu')
    img2, phi_b, hist = fastsar.autofocus.pga(img * np.exp(-1j * psi))      # image domain only; psi from deramp_phase

PGA (Wahl, Eichel, Ghiglia and Jakowatz 1994; Jakowatz et al. 1996, ch. 4) models the azimuth phase error as one
function of azimuth spatial frequency shared by every range line. Each iteration moves the brightest pixel of
every range line to the center of the line, keeps the range lines with the highest peak-to-mean intensity, windows
them, transforms along azimuth (axis 0 of a form_image image) and estimates the phase difference between adjacent
frequency bins from all kept lines at once. The integrated estimate, less its constant and linear terms (a phase
and a shift), is removed from the image, and the window narrows as the image focuses.

Pulses and bins. In the planar-wavefront (polar format) picture, pulse p (antenna a_p) adds to the image a plane
wave exp(-j 2 pi u_p y) in azimuth position y = x . e1, with u_p = 2 f (a_p . e1) / (|a_p| c), the same for every
scatterer. Under the numpy FFT along axis 0 that is frequency bin  b_p = -u_p spx nx  (bins in centered order, 0 at
DC), close to linear in p. A backprojection image keeps the spherical wavefront instead: a scatterer at azimuth
offset y sees the aperture from a direction turned by about y/R, which moves its whole azimuth spectrum by
2 fc y / (c R) cycles/m (16 bins for a scatterer 20 m off center at 5 km range on the test grid), so one phase
error per pulse lands on different bins in different parts of the scene. deramp_phase gives the phase
psi(x) = 4 pi fc / c (|x - a_c| - |a_c| + x . a_c / |a_c|), the difference between the spherical and the planar
wavefront of the aperture-center antenna a_c, and img * exp(-j psi) puts every pulse back on its bin b_p everywhere.
autofocus forms its images by backprojection only; form_image's polar format, which resamples its image to remove
the planar-wavefront displacement, has not been validated with it.

autofocus deramps, runs PGA, interpolates the estimate on bins at b_p and multiplies S[p] by exp(-j phi_p). The
approximations: u_p, b_p and psi are taken at the center frequency fc (over the band, of relative size B/fc, the
bin of a pulse spreads, which blurs the estimate, while the correction applied to S stays exact per pulse); the
deramp is that of the aperture-center pulse; and the aperture must fit inside the image's azimuth band,
|b_p| < nx/2, which needs an azimuth pixel spacing finer than the resolution. Pulses outside the band take the
value of the nearest edge bin, and the bin spacing 1/(nx spx) limits how fast the phase error can vary across the
aperture and still be followed. Each further round re-forms the image with the correction so far and adds PGA's
estimate of what is left.

Image domain. quadratic, spatially_variant and image_autofocus (below) work on a finished baseband image whose
axis 0 is azimuth, with transforms along axis 0 only, on the device that holds it (NumPy, CuPy or JAX array).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as _sfft

from . import _xp
from .sim import C

__all__ = ['pga', 'pulse_bins', 'deramp_phase', 'autofocus', 'quadratic', 'spatially_variant', 'image_autofocus']


def _detrend(phi, x, w=None):
    """phi less its (weighted) least-squares constant and linear terms in x."""
    A = np.stack([np.ones_like(x), x], 1)
    s = np.ones_like(x) if w is None else np.sqrt(w)
    return phi - A @ np.linalg.lstsq(A * s[:, None], phi * s, rcond=None)[0]


def pga(img, iterations=20, tol=0.005, method='ml', win_db=10.0, min_win=8, win_scale=1.5, shrink=0.7, lines=0.3, band_db=30.0):
    """Phase gradient autofocus along axis 0 (azimuth) of a complex image [nx, ny] in the polar-format convention
    (a backprojection image deramped first; see deramp_phase).

    method: 'ml' (angle of the sum over range lines of conj(G[m]) G[m+1], the maximum-likelihood phase-difference
    estimator) or 'lumv' (sum Im(conj(G) dG) / sum |G|^2, the linear unbiased minimum-variance one). lines: the
    fraction of range lines kept, by peak-to-mean intensity. The window width is win_scale times the width of the
    center-shifted, line-summed intensity above -win_db relative to its peak, but shrinks by at most a factor
    `shrink` per iteration, never grows and never falls below min_win pixels. Bins more than band_db below the
    strongest carry no gradient, and the detrend and the rms are weighted by the image's energy per bin. Iterates
    until the rms of a correction is below tol (rad).

    -> (focused image complex64 [nx, ny], phase error estimate [nx] (rad) on azimuth frequency bins in centered
    order, bin m - nx//2 at index m, constant and linear terms removed; history: one dict per iteration with the
    rms correction and the window width)."""
    g = np.asarray(img).astype(np.complex128)
    nx = g.shape[0]
    m = (np.arange(nx) - nx // 2).astype(np.float64)
    wb = np.fft.fftshift((np.abs(np.fft.fft(g, axis=0)) ** 2).sum(1))        # energy per bin; unchanged by PGA
    wb = wb / wb.max()
    band = (wb[:-1] > 10 ** (-band_db / 10)) & (wb[1:] > 10 ** (-band_db / 10))
    phi, W, hist = np.zeros(nx), nx, []
    for _ in range(iterations):
        pk = np.abs(g).argmax(0)
        h = np.take_along_axis(g, (np.arange(nx)[:, None] + pk[None, :] - nx // 2) % nx, 0)
        if lines < 1:
            h = h[:, np.argsort(np.abs(h[nx // 2]) ** 2 / np.mean(np.abs(h) ** 2, 0))[::-1][:max(1, int(lines * h.shape[1]))]]
        prof = (np.abs(h) ** 2).sum(1)
        low = np.nonzero(prof < prof[nx // 2] * 10 ** (-win_db / 10))[0] - nx // 2
        r = min(-low[low < 0].max() if (low < 0).any() else nx // 2, low[low > 0].min() if (low > 0).any() else nx // 2)
        W = max(min_win, int(np.ceil(shrink * W)), min(W, int(np.ceil(win_scale * (2 * r - 1)))))
        h = h * (np.abs(m) <= W / 2)[:, None]
        G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(h, 0), axis=0), 0)
        a, b = G[:-1], G[1:]
        if method == 'ml':
            d = np.angle(np.sum(np.conj(a) * b, 1))
        elif method == 'lumv':
            d = np.sum(np.imag(np.conj(a) * b), 1) / np.maximum(np.sum(np.abs(a + b) ** 2, 1) / 4, 1e-300)
        else:
            raise ValueError("method must be 'ml' or 'lumv'")
        dphi = _detrend(np.concatenate([[0.0], np.cumsum(d * band)]), m, wb)
        g = np.fft.ifft(np.fft.fft(g, axis=0) * np.exp(-1j * np.fft.ifftshift(dphi))[:, None], axis=0)
        phi += dphi
        hist.append(dict(rms=float(np.sqrt(np.sum(wb * dphi ** 2) / wb.sum())), window=W))
        if hist[-1]['rms'] < tol:
            break
    return g.astype(np.complex64), phi, hist


def pulse_bins(ant, fmin, df, K, nx, spx, e1=(1.0, 0.0, 0.0)):
    """Azimuth frequency bin b_p (centered order, fractional) of each pulse at the center frequency; see the module
    docstring."""
    a = np.asarray(ant, np.float64)
    fc = float(fmin) + (int(K) // 2) * float(df)
    return -2 * fc / C * (a @ np.asarray(e1, np.float64)) / np.linalg.norm(a, axis=1) * spx * nx


def deramp_phase(ant, fmin, df, K, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0)):
    """Phase psi [nx, ny] (rad) that takes a backprojection image to the polar-format convention PGA assumes:
    img * exp(-1j * psi) puts every pulse at the same azimuth frequency bin anywhere in the scene. See the module
    docstring."""
    a = np.asarray(ant, np.float64)[len(ant) // 2]
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    x = ((np.arange(nx) - nx / 2) * spx)[:, None, None] * e1 + ((np.arange(ny) - ny / 2) * spy)[None, :, None] * e2
    r = np.linalg.norm(a)
    return 4 * np.pi * (float(fmin) + (int(K) // 2) * float(df)) / C * (np.linalg.norm(x - a, axis=-1) - r + x @ a / r)


def autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), rounds=2, pga_kwargs=None,
              **form_kwargs):
    """Autofocus a phase history: form the image (factorized backprojection), deramp it, run pga, map its estimate
    onto pulses, multiply S by exp(-j phi_p) and re-form; `rounds` times, each adding to phi_p. form_kwargs go to
    ImageFormer (backend, precision, T, levels, ...), pga_kwargs to pga.

    -> (focused image complex64 [nx, ny], phi_p [P] (rad), the phase error estimate per pulse with its constant and
    linear terms in b_p removed; S * exp(-1j * phi_p)[:, None] is the corrected phase history)."""
    from .api import ImageFormer
    if form_kwargs.pop('algorithm', 'ffbp') != 'ffbp':
        raise ValueError('autofocus forms its images by factorized backprojection')
    form_kwargs.pop('pfa_guard', None)
    S = np.asarray(S)
    P, K = S.shape
    former = ImageFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, **form_kwargs)
    b = pulse_bins(ant, fmin, df, K, nx, spx, e1)
    m = np.arange(nx) - nx // 2
    ramp = np.exp(-1j * deramp_phase(ant, fmin, df, K, nx, ny, spx, spy, e1, e2))
    phi = np.zeros(P)

    def corrected(phi):          # S exp(-j phi) as one complex64 array, in row blocks (no complex128 temporary)
        out = np.empty(S.shape, np.complex64)
        for p0 in range(0, P, 1024):
            sl = slice(p0, min(P, p0 + 1024))
            np.multiply(S[sl], np.exp(-1j * phi[sl])[:, None].astype(np.complex64), out=out[sl])
        return out

    for _ in range(rounds):
        _, pb, _ = pga(former(corrected(phi)) * ramp, **(pga_kwargs or {}))
        phi = _detrend(phi + np.interp(b, m, pb), b)
    return former(corrected(phi)), phi


# ------------------------------------------------------------------------------------- image-domain autofocus
# Convention for everything below: img [nx, ny] complex, at baseband, axis 0 azimuth; the phase error multiplies
# the FFT along axis 0 of every column by exp(j phi(k)). Only axis-0 transforms are taken. v is the offset of an
# azimuth frequency bin from the center of the occupied band in half band widths (|v| <= 1 inside the band).
_MODES = ('quadratic', 'pga', 'spatially_variant', 'spatially_variant_pga')
_JIT = {}
_BLOCK = dict(numpy=1 << 19, cupy=1 << 27, jax=1 << 28)     # samples per column block when a correction is applied
MIN_SIG = 10.0                                              # least significance of a correlation peak (_peaks)


def _ft(a, kind='fft', workers=-1):
    """fft, ifft or rfft of a along axis 0 on the device of a (on the host with `workers` threads, -1: all cores)."""
    fn = getattr(_xp.fft(a), kind)
    return fn(a, axis=0, workers=workers) if _xp.kind(a) == 'numpy' else fn(a, axis=0)


def _offsets(n, f0, hf):
    """v of each bin of an n-point FFT (FFT order) for a band centered at f0 with half width hf (cycles/pixel)."""
    return (((np.arange(n) / n - f0 + 0.5) % 1.0) - 0.5) / hf


def _edge(e, r, w):
    """Fractional index at which the energies e, read outward, fall to half their level just inside index r (the
    outermost bin above the detection threshold; the level is the median of w bins ending three bins inside r).
    r + 0.5 when no such crossing lies within w bins of r."""
    half = 0.5 * np.median(e[max(0, r - 3 - w):max(1, r - 2)])
    for i in range(max(1, r - 2), min(len(e), r + w + 1)):
        if e[i] < half:
            if e[i - 1] <= half:
                return i - 0.5
            return i - 1 + np.log(e[i - 1] / half) / np.log(e[i - 1] / max(e[i], 1e-300))
    return r + 0.5


def _band(e, band_db=30.0):
    """The occupied band of the energies per azimuth FFT bin e [nx] (FFT order) -> (f0, hf, sep): center and half
    width in cycles per pixel, and the distance in v between the amplitude-weighted centers of its two halves.

    The band is searched from the emptiest part of the spectrum, so it may wrap. Its edges are first the outermost
    bins within band_db of the largest (5-bin mean) energy, then, to a fraction of a bin, the points where the
    energy falls to half its level just inside them. A spectrum with no bin below the threshold is full: hf = 0.5,
    centered at zero."""
    nx = len(e)
    e3 = np.convolve(np.concatenate([e[-1:], e, e[:1]]), np.ones(3) / 3, 'valid')
    e5 = np.convolve(np.concatenate([e[-2:], e, e[:2]]), np.ones(5) / 5, 'valid')
    k = max(1, nx // 16)                                                    # the emptiest run of 2k + 1 bins
    gap = int(np.argmin(np.convolve(np.concatenate([e[-k:], e, e[:k]]), np.ones(2 * k + 1), 'valid')))
    occ = np.flatnonzero(np.roll(e3, -(gap + 1)) > e5.max() * 10 ** (-band_db / 10))   # bins counted from the gap
    if len(occ) == nx:
        c, hf = 0.0, 0.5                                                    # baseband: centered at zero
    else:
        eo = np.roll(e5, -(gap + 1))
        w = max(4, int(round(0.03 * (occ[-1] - occ[0] + 1))))
        hi = _edge(eo, occ[-1], w)
        lo = nx - 1 - _edge(eo[::-1], nx - 1 - occ[0], w)
        c, hf = gap + 1 + (lo + hi) / 2, min(0.5, (hi - lo) / (2 * nx))
    f0 = (c / nx + 0.5) % 1.0 - 0.5
    v, a = _offsets(nx, f0, hf), np.sqrt(e)
    up, dn = (v >= 0) & (v <= 1), (v < 0) & (v >= -1)
    sep = (a[up] * v[up]).sum() / a[up].sum() - (a[dn] * v[dn]).sum() / a[dn].sum()
    return float(f0), float(hf), float(sep)


def _check(img):
    if getattr(img, 'ndim', 0) != 2 or 'complex' not in str(img.dtype) or img.shape[0] < 16:
        raise ValueError('image autofocus expects a complex image [nx, ny] with nx >= 16')


def _select(img, lines, max_lines, band_db):
    """The columns the global estimates use, and the occupied band. Up to max_lines evenly spaced columns are read;
    those with fewer than half as many nonzero samples as the fullest are dropped, the band comes from the rest
    (each tapered along azimuth so that the cut at the image ends does not leak across the band edges), and the
    fraction `lines` of them (at least 16) with the highest peak-to-mean intensity is kept.
    -> (columns [nx, m] complex64 on the device of img, (f0, hf, sep)), or None for an image without data."""
    xp = _xp.module(img)
    nx, ny = img.shape
    idx = np.unique(np.round(np.linspace(0, ny - 1, min(ny, int(max_lines)))).astype(np.int64))
    sub = img[:, idx].astype(xp.complex64)
    if _VIA and xp is np:                                           # a host image, its estimate on JAX's device
        import jax.numpy as xp
        sub = xp.asarray(sub)
    p = xp.abs(sub) ** 2
    cnt, s, mx = (_xp.host(t).astype(np.float64) for t in ((p > 0).sum(0), p.sum(0), p.max(0)))
    ok = np.flatnonzero((cnt > 0) & (cnt >= 0.5 * cnt.max()) & (s > 0))
    if len(ok) == 0:
        return None
    sub = sub[:, ok]
    tap = xp.asarray((np.sin(np.pi * (np.arange(nx) + 0.5) / nx) ** 2).astype(np.float32))[:, None]
    e = _xp.host((xp.abs(_ft(sub * tap)) ** 2).sum(1)).astype(np.float64)
    if lines < 1:
        con = mx[ok] * cnt[ok] / s[ok]
        keep = min(len(ok), max(16, int(round(lines * len(ok)))))
        sub = sub[:, np.sort(np.argsort(con, kind='stable')[len(ok) - keep:])]
    return sub, _band(e, band_db)


def _centroid(a):
    """Centroid along axes 0 and 2 of the squared intensity of each set of a [n, nb, m] -> host [nb, 2] (samples;
    the middle for an empty set). The bright scatterers that dominate an estimate sit there."""
    xp = _xp.module(a)
    p = xp.abs(a) ** 2
    p = p * p
    n, _, m = a.shape
    t = _xp.host(p.sum((0, 2))).astype(np.float64)
    c0 = _xp.host((p.sum(2) * xp.arange(n, dtype=xp.float32)[:, None]).sum(0)).astype(np.float64)
    c1 = _xp.host((p.sum(0) * xp.arange(m, dtype=xp.float32)[None, :]).sum(1)).astype(np.float64)
    ok, t = t > 0, np.where(t > 0, t, 1.0)
    return np.stack([np.where(ok, c0 / t, (n - 1) / 2), np.where(ok, c1 / t, (m - 1) / 2)], 1)


def _contrast(a, axes=None):
    """Contrast of the intensity of a, std / mean over the axes (all pixels, zeros included) -> host float64."""
    xp = _xp.module(a)
    if _xp.kind(a) == 'jax':                                        # one compiled program, not one an operation
        m1, m2 = (_xp.host(v).astype(np.float64) for v in _jit('contrast')(a, axes))
    else:
        p = xp.abs(a) ** 2
        m1 = _xp.host(p.mean(axes)).astype(np.float64)
        m2 = _xp.host((p * p).mean(axes)).astype(np.float64)
    return np.where(m1 > 0, np.sqrt(np.maximum(m2 - m1 * m1, 0.0)) / np.where(m1 > 0, m1, 1.0), 0.0)


def _peaks(X, n, up=8, workers=1):
    """Lag (samples, fractional), height and significance of the largest value of each circular cross-correlation
    whose rfft is X [nb, n // 2 + 1], within n / 4 of zero lag: a grid `up` times finer, then a parabola through its
    top three. The significance is the height above the median of the searched lags (whole samples) in units of
    their robust standard deviation (1.4826 times the median absolute deviation). workers: host FFT threads."""
    L = n * up
    xc = _sfft.irfft(X, L, axis=1, workers=workers) * up
    h = (n // 4) * up
    idx = np.concatenate([np.arange(0, h + 1), np.arange(L - h, L)])
    i = idx[np.argmax(xc[:, idx], axis=1)]
    r = np.arange(len(X))
    y0, y1, y2 = xc[r, (i - 1) % L], xc[r, i], xc[r, (i + 1) % L]
    den = y0 - 2 * y1 + y2
    frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, 1.0), 0.0)
    w = xc[:, idx[::up]]
    med = np.median(w, axis=1)
    sd = 1.4826 * np.median(np.abs(w - med[:, None]), axis=1)
    sig = np.where(sd > 0, (y1 - med) / np.where(sd > 0, sd, 1.0), 0.0)
    return (np.where(i >= L // 2, i - L, i) + frac) / up, y1, sig


def _map_drift(A, band, iterations, tol, workers=-1, c0=0.0, pool=False, G=None, contrast=True, taper=None):
    """Iterated map drift on A [n, nb, m] complex64 (on its device): nb independent sets of m columns, one quadratic
    coefficient each. Each iteration removes the estimate so far, forms the detected images of the lower and upper
    half bands, measures the shift d (samples) of the upper against the lower at the peak of their cross-correlation
    summed over the m columns, and adds  -pi hf d / sep  to the coefficient (a quadratic c v^2 moves the two looks
    apart by c sep / (pi hf) samples). Stops when every set's update is below tol. workers: host FFT threads; c0:
    the coefficient every set starts from (rad); pool: one coefficient for all sets together (the correlations
    summed over the sets too; the results then have one entry); G: fft(A, axis=0) if at hand; contrast: False to
    skip before and after; taper [n]: a weight per azimuth bin (FFT order) on both looks, with band's sep measured
    under it (moving.autofocus's: the looks of the estimate, not of the image).
    -> dict(c [nb] rad at the band edge, corr [nb] the correlation coefficient at the peak, sig [nb] the peak's
    significance (_peaks), step [nb] the last update, iterations, G, before, after [nb]: the contrast of each set
    as given and with its estimate removed)."""
    xp = _xp.module(A)
    f0, hf, sep = band
    n, nb = A.shape[:2]
    v = _offsets(n, f0, hf)
    dn, up = (v < 0) & (v >= -1), (v >= 0) & (v <= 1)
    if taper is not None:
        dn, up = dn * np.asarray(taper, np.float64), up * np.asarray(taper, np.float64)
    dn, up = (xp.asarray(k.astype(np.float32))[:, None, None] for k in (dn, up))
    jitted = _xp.kind(A) == 'jax'                                   # an iteration as one compiled program
    if G is None:
        G = _jit('fft0')(A) if jitted else _ft(A, 'fft', workers)
    quad = lambda c: xp.asarray(np.exp(-1j * c[None, :] * v[:, None] ** 2).astype(np.complex64))[:, :, None]
    c = np.full(nb, float(c0))
    for it in range(max(1, int(iterations))):
        if jitted:
            xr, xi, ea, eb = (_xp.host(t) for t in _jit('drift')(G, quad(c), dn, up))
            X = (xr + 1j * xi).T.astype(np.complex128)
        else:
            Gq = G * quad(c)
            a, b = xp.abs(_ft(Gq * dn, 'ifft', workers)) ** 2, xp.abs(_ft(Gq * up, 'ifft', workers)) ** 2
            a, b = a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)
            X = _xp.host((xp.conj(_ft(a, 'rfft', workers)) * _ft(b, 'rfft', workers)).sum(-1)).T.astype(np.complex128)
            ea, eb = _xp.host((a * a).sum((0, 2))), _xp.host((b * b).sum((0, 2)))
        ea, eb = ea.astype(np.float64), eb.astype(np.float64)
        if pool:
            X, ea, eb = X.sum(0, keepdims=True), ea.sum(keepdims=True), eb.sum(keepdims=True)
        ee = ea * eb
        d, pk, sig = _peaks(X, n, workers=workers)
        corr = np.where(ee > 0, pk / np.sqrt(np.where(ee > 0, ee, 1.0)), 0.0)
        step = np.where(ee > 0, -np.pi * hf / sep * d, 0.0)
        c = c + step
        if np.all(np.abs(step) < tol):
            break
    out = dict(c=c[:len(step)], corr=corr, sig=np.where(ee > 0, sig, 0.0), step=step, iterations=it + 1, G=G)
    if contrast:
        after = _jit('undo')(G, quad(c)) if jitted else _ft(G * quad(c), 'ifft', workers)
        out.update(before=_contrast(A, (0, 2)), after=_contrast(after, (0, 2)))
    return out


def _jit(name):
    """The jax.jit kernels that apply a correction (built on first use; one compilation per array shape)."""
    if not _JIT:
        import jax
        import jax.numpy as jnp

        def fixed(blk, q):                          # one phase per bin for every column
            blk = blk.astype(jnp.complex64)
            t = jnp.fft.ifft(jnp.fft.fft(blk, axis=0) * q[:, None], axis=0)
            return jnp.where(blk != 0, t, 0).astype(jnp.complex64)

        def strip(blk, V, Cf, w):                   # phase sum_n Cf[n, j] V[n, b] per bin b and column j, weight w[b]
            blk = blk.astype(jnp.complex64)
            ph = sum(V[n][:, None] * Cf[n][None, :] for n in range(V.shape[0]))    # not a matrix product: exact float32
            t = jnp.fft.ifft(jnp.fft.fft(blk, axis=0) * (jnp.cos(ph) - 1j * jnp.sin(ph)), axis=0)
            return (jnp.where(blk != 0, t, 0) * w[:, None]).astype(jnp.complex64)

        def drift(G, q, dn, up):                    # _map_drift's iteration: the looks' cross-spectrum and energies
            Gq = G * q
            a, b = jnp.abs(jnp.fft.ifft(Gq * dn, axis=0)) ** 2, jnp.abs(jnp.fft.ifft(Gq * up, axis=0)) ** 2
            a, b = a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)
            X = (jnp.conj(jnp.fft.rfft(a, axis=0)) * jnp.fft.rfft(b, axis=0)).sum(-1)
            return jnp.real(X), jnp.imag(X), (a * a).sum((0, 2)), (b * b).sum((0, 2))

        def contrast(a, axes):                      # _contrast's two means
            p = jnp.abs(a) ** 2
            return p.mean(axes), (p * p).mean(axes)

        def colsums(blk):                           # _colsums of a block
            p = jnp.real(blk) ** 2 + jnp.imag(blk) ** 2
            return (p > 0).sum(0), p.sum(0), (p * p).sum(0)
        _JIT.update(fixed=jax.jit(fixed), strip=jax.jit(strip), drift=jax.jit(drift), colsums=jax.jit(colsums),
                    contrast=jax.jit(contrast, static_argnums=1), fft0=jax.jit(lambda a: jnp.fft.fft(a, axis=0)),
                    undo=jax.jit(lambda G, q: jnp.fft.ifft(G * q, axis=0)))
    return _JIT[name]


def _threads(work, items):
    with ThreadPoolExecutor(min(len(items), os.cpu_count() or 1) or 1) as ex:
        list(ex.map(work, items))


_VIA = []                # not empty: a NumPy image's corrections and column sums run on JAX's device (_via_jax)


class _via_jax:
    """with _via_jax(): the global estimates and the correction of an image on the host run on JAX's device, the
    image crossing it in column blocks (process_cphd's, for an image the device cannot hold). The result is on the
    host."""

    def __enter__(self):
        _VIA.append(1)

    def __exit__(self, *exc):
        _VIA.pop()


def _columns(img):
    """block(j, w) -> columns j ... j + w of an image on the device or on the host as a JAX array [nx, w]
    (resample_jax._source: a device image is read through its real planes, a host one crosses block by block)."""
    from .resample_jax import _source
    lines = _source(img, 0)
    return lambda j, w: lines(j, w).T


def _jax_width(nx, ny):
    """Columns per block of a JAX image [nx, ny] (see _apply). MemoryError when the device cannot hold one."""
    return int(min(_xp.jax_rows(ny, 64 * nx, 'the autofocus correction on the JAX device'), max(1, _BLOCK['jax'] // nx)))


def _colsums(img):
    """Per column of img [nx, ny]: the number of nonzero samples, the sum of their intensity and the sum of its
    square -> host float64 [3, ny]. Computed on the device of img in the column blocks of _apply; only the sums
    come to the host."""
    nx, ny = img.shape
    out, k = np.zeros((3, ny)), _xp.kind(img)
    if k == 'jax' or _VIA:
        f, w, block = _jit('colsums'), _jax_width(nx, ny), _columns(img)
        for j in list(range(0, ny - w + 1, w)) + ([ny - w] if ny % w else []):
            out[:, j:j + w] = [_xp.host(t) for t in f(block(j, w))]
        return out
    xp = _xp.module(img)
    w = int(max(1 if k == 'cupy' else 4, _BLOCK[k] // nx))

    def work(j):
        blk = img[:, j:j + w]
        p = blk.real ** 2 + blk.imag ** 2
        out[0, j:j + w] = _xp.host(xp.count_nonzero(p, axis=0))
        out[1, j:j + w] = _xp.host(p.sum(0, dtype=xp.float64))
        p *= p
        out[2, j:j + w] = _xp.host(p.sum(0, dtype=xp.float64))
    if k == 'cupy':
        for j in range(0, ny, w):
            work(j)
    else:
        _threads(work, range(0, ny, w))
    return out


def _gain(sb, sa, sets=16):
    """The guard's measure, from the column sums (_colsums) of an image before (sb) and after (sa) a correction.
    The contrast is std / mean of the intensity over the samples that are nonzero before, of the whole image and
    of each of `sets` contiguous sets of columns (parts of the scene).
    -> (contrast of the image before, after; the standard error of the log of their ratio, by the jackknife that
    leaves out one set at a time; the median over the sets of log(after / before) of their own contrasts; its
    standard error, 1.2533 * 1.4826 * the median absolute deviation / sqrt(sets). With fewer than four sets that
    hold data the last three are 0, the log ratio of the whole image and 0)."""
    def con(n, s1, s2):
        n = np.maximum(n, 1.0)
        m1, m2 = s1 / n, s2 / n
        return np.where(m1 > 0, np.sqrt(np.maximum(m2 - m1 * m1, 0.0)) / np.where(m1 > 0, m1, 1.0), 0.0)
    cuts = np.unique(np.round(np.linspace(0, sb.shape[1], min(int(sets), sb.shape[1]) + 1)).astype(np.int64))[:-1]
    n, b1, b2, a1, a2 = (np.add.reduceat(t, cuts) for t in (sb[0], sb[1], sb[2], sa[1], sa[2]))
    cb, ca = float(con(n.sum(), b1.sum(), b2.sum())), float(con(n.sum(), a1.sum(), a2.sum()))
    kb, ka = con(n, b1, b2), con(n, a1, a2)
    ok = (kb > 0) & (ka > 0)
    if ok.sum() < 4:
        return cb, ca, 0.0, (float(np.log(ca / cb)) if cb > 0 and ca > 0 else 0.0), 0.0
    rest = lambda t: t.sum() - t[ok]                                # the image without one set
    jb, ja = con(rest(n), rest(b1), rest(b2)), con(rest(n), rest(a1), rest(a2))
    th = np.log(np.maximum(ja, 1e-300) / np.maximum(jb, 1e-300))
    k = int(ok.sum())
    g = np.log(ka[ok] / kb[ok])
    med = float(np.median(g))
    return (cb, ca, float(np.sqrt((k - 1) / k * ((th - th.mean()) ** 2).sum())), med,
            float(1.2533 * 1.4826 * np.median(np.abs(g - med)) / np.sqrt(k)))


def _keep(img, out, info, guard, sb=None, z=2.0):
    """Measure a corrected image `out` against `img` (_gain; sb: the column sums of img if at hand), record in info
    contrast_before, contrast_after, gain (the log of their ratio), gain_std, part_gain (the median gain of the
    sets of columns) and part_gain_std, and decide. With guard the correction is kept only if the contrast of the
    whole image rose, the median gain of the parts is not below -z times its standard error, and one of the two
    gains is at least z times its standard error. -> out, or img with info['reason'] set."""
    cb, ca, se, med, mse = _gain(_colsums(img) if sb is None else sb, _colsums(out))
    d = float(np.log(ca / cb)) if cb > 0 and ca > 0 else 0.0
    info.update(contrast_before=cb, contrast_after=ca, gain=d, gain_std=se, part_gain=med, part_gain_std=mse)
    if guard and not ca > cb:
        return _skip(img, info, 'the correction lowers the contrast of the image')
    if guard and med < -z * mse:
        return _skip(img, info, 'the correction lowers the contrast of most parts of the image')
    if guard and not (d >= z * se or (med > 0 and med >= z * mse)):
        return _skip(img, info, 'the contrast gain is within its scatter over the parts of the image')
    info.update(applied=True, reason=None)
    return out


def _apply(img, phi, q=None):
    """img with the phase phi [nx] (rad, FFT order) removed from its azimuth spectrum, complex64, on the device of
    img: ifft(fft(img, axis=0) * exp(-1j * phi)[:, None], axis=0), exact zeros of img kept (q [nx]: any factor per
    bin in place of exp(-1j * phi), as moving.band_limit gives). Column blocks: on the
    host one thread per block of about 4 MB, on a GPU blocks of 1 GB, in JAX equal blocks of at most 2 GB (less on
    a device with little memory: _xp.jax_block) through one jitted kernel (the last block overlaps its neighbor
    so that every block has the same shape)."""
    nx, ny = img.shape
    q = np.exp(-1j * np.asarray(phi, np.float64)).astype(np.complex64) if q is None else np.asarray(q, np.complex64)
    k = _xp.kind(img)
    if k == 'numpy' and _VIA:                                       # through JAX's device, to a new host image
        f, w, out, block = _jit('fixed'), _jax_width(nx, ny), np.empty((nx, ny), np.complex64), _columns(img)
        last = None
        for j in list(range(0, ny - w + 1, w)) + ([ny - w] if ny % w else []):
            last, cur = (j, f(block(j, w), q)), last
            if cur is not None:                                     # written while the next block is under way
                _xp.to_host(out, (slice(None), slice(cur[0], cur[0] + w)), cur[1])
        _xp.to_host(out, (slice(None), slice(last[0], last[0] + w)), last[1])
        return out
    if k == 'jax':
        import jax.numpy as jnp
        f, w, block = _jit('fixed'), _jax_width(nx, ny), _columns(img)
        parts = [f(block(j, w), q) for j in range(0, ny - w + 1, w)]
        if ny % w:
            parts.append(f(block(ny - w, w), q)[:, w - ny % w:])
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
    if k == 'cupy':
        import cupy as cp
        import cupyx.scipy.fft as cf
        out, qd, w = cp.empty((nx, ny), cp.complex64), cp.asarray(q)[:, None], int(max(1, _BLOCK['cupy'] // nx))
        for j in range(0, ny, w):
            blk = img[:, j:j + w]
            t = cf.fft(blk.astype(cp.complex64, copy=False), axis=0)
            t *= qd
            t = cf.ifft(t, axis=0, overwrite_x=True)
            cp.multiply(t, blk != 0, out=out[:, j:j + w])
        return out
    out, w = np.empty((nx, ny), np.complex64), int(max(4, _BLOCK['numpy'] // nx))

    def work(j):
        blk = img[:, j:j + w]
        t = _sfft.fft(blk.astype(np.complex64, copy=False), axis=0)
        t *= q[:, None]
        t = _sfft.ifft(t, axis=0, overwrite_x=True)
        np.multiply(t, blk != 0, out=out[:, j:j + w])
    _threads(work, range(0, ny, w))
    return out


def _skip(img, info, reason, **kw):
    info.update(applied=False, reason=reason, **kw)
    return img


def quadratic(img, *, iterations=6, lines=0.3, tol=0.02, guard=True, max_lines=1024, band_db=30.0, subsets=8,
              min_significance=MIN_SIG):
    """Estimate and remove a quadratic azimuth phase error by map drift.

    img [nx, ny]: complex baseband image, axis 0 azimuth (a slant-plane image, or a ground-plane image on axes along
    the line of sight after planes.deramp), as a NumPy, CuPy or JAX array; it is not modified. The error model is
    fft(img, axis=0) = fft(focused, axis=0) * exp(1j * c * v ** 2)[:, None], the same for every column, v being the
    offset of an azimuth frequency bin from the center of the occupied band in half band widths. The band (center
    f0, half width hf, cycles per pixel) is detected from the data to band_db below its strongest part, so an
    oversampled image is handled, and nothing outside it is changed. Only transforms along axis 0 are taken: the
    columns are independent looks and may carry any phase or spectrum along axis 1. Exact zeros stay exact zeros.

    Estimate: up to max_lines evenly spaced columns are read, mostly-zero ones dropped and the fraction `lines`
    with the highest peak-to-mean intensity kept. The band is split into halves, the two detected sub-look images
    are cross-correlated along azimuth (sum over the kept columns, peak to a fraction of a sample), the shift is
    converted to c and removed, and this repeats until an update is below tol (rad at the band edge) or
    `iterations` is reached. The transforms of the estimate and of the correction run on the device of img.

    guard: the estimate is applied only if it passes two tests, and otherwise the input is returned as it came with
    info['applied'] = False and info['reason'].
    Consistency, before anything is applied: the kept columns are cut into `subsets` equal contiguous sets (parts
    of the scene; the estimate uses the columns of the sets) and each gives its own estimate. The sets whose
    correlation peak stands min_significance robust standard deviations above the other lags count; at least half
    of them (and four) must, and so must the peak of all columns together. With s the robust scatter of their
    estimates (1.4826 times the median absolute deviation about the median) and sigma = s / sqrt(sets) the
    uncertainty of c, the estimate must lie within 3 sigma + tol of their median, |c| must be at least 3 sigma,
    and at most a quarter of the sets may lie more than 4.5 s + tol from the median: a scene whose parts ask for
    different corrections (relief, moving surfaces, one dominant reflector) gets none.
    Result: the contrast (std / mean of the intensity over the nonzero samples) is measured on every column of
    the image as given and as corrected, for the whole image and for 16 contiguous sets of columns. The contrast of
    the whole image must rise; the median over the sets of the log ratio of their contrasts must not be below
    minus twice its standard error (from the scatter of the sets); and either that median or the log ratio of the
    whole image must be at least twice its standard error (for the whole image: the jackknife over the sets).
    Without guard an estimate of at least tol is applied as it is. An estimate below tol is never applied.

    -> (img2 complex64 [nx, ny] in the same place as img; phi [nx] float64 on the host: the phase removed (rad) per
    bin of the axis-0 FFT in fftfreq order, c v^2 inside the band and zero outside (all zero when nothing was
    applied), so that img2 = ifft(fft(img, axis=0) * exp(-1j * phi)[:, None], axis=0) with the zeros of img put
    back; info: dict(edge_rad: c, the estimate in radians at the band edge, whether applied or not; edge_rad_std:
    sigma (None when the sets were not estimated); subsets: the sets' estimates (rad) and subset_significance
    theirs; significance and correlation: of the peak of all columns (correlation coefficient of the two looks);
    band: (f0, hf); iterations; lines: columns used; contrast_before, contrast_after: of the whole image, as given
    and with the estimate removed (equal when no correction was evaluated); gain, gain_std: the log of their ratio
    and its standard error; part_gain, part_gain_std: the median log contrast ratio of the sets of columns and its
    standard error; applied; reason: why not, or None)).

    Phase history. A phase error phi_p on pulse p (row p of the phase history multiplied by exp(1j * phi_p)) shows
    up with the same sign and scale at the pulse's azimuth frequency bin b_p (pulse_bins): phi[b_p mod nx] estimates
    +phi_p, less a constant and a linear term in b_p, which only move the image and change its phase. For
    phi_p = a t^2, t being the pulse's azimuth frequency scaled to -1 .. 1 across the aperture, edge_rad estimates
    +a (within 3% on the simulated collection of the tests, ground and slant plane). t is the scaled pulse index
    only when the pulses are evenly spaced in azimuth frequency: with squint or a curved track a quadratic in the
    pulse index has cubic and higher terms in t, which this function leaves and image_autofocus(img, 'pga') removes."""
    _check(img)
    nx = img.shape[0]
    info = dict(edge_rad=0.0, edge_rad_std=None, subsets=None, subset_significance=None, significance=0.0, band=None,
                iterations=0, correlation=0.0, lines=0, contrast_before=0.0, contrast_after=0.0, gain=0.0,
                gain_std=0.0, part_gain=0.0, part_gain_std=0.0, applied=False, reason=None)
    sel = _select(img, lines, max_lines, band_db)
    if sel is None:
        return _skip(img, info, 'the image holds no data'), np.zeros(nx), info
    cols, band = sel
    k = min(int(subsets), cols.shape[1] // 4)
    if k >= 4:                                                      # the sets: equal, contiguous, in column order
        A = cols[:, :k * (cols.shape[1] // k)].reshape(nx, k, cols.shape[1] // k)
    else:
        A = cols[:, None, :]
    m = int(A.shape[1] * A.shape[2])
    est = _map_drift(A, band, iterations, tol, pool=True, contrast=False)
    c = float(est['c'][0])
    sb = _colsums(img)
    info.update(edge_rad=c, band=band[:2], iterations=est['iterations'], correlation=float(est['corr'][0]),
                significance=float(est['sig'][0]), lines=m)
    info['contrast_before'] = info['contrast_after'] = _gain(sb, sb)[0]
    if abs(c) < tol:
        return _skip(img, info, 'the estimate is below tol'), np.zeros(nx), info
    if guard:
        if k < 4:
            return _skip(img, info, 'too few columns to check the estimate'), np.zeros(nx), info
        sub = _map_drift(A, band, iterations, tol, c0=c, G=est['G'], contrast=False)
        ok = np.isfinite(sub['c']) & (sub['sig'] >= min_significance)
        info.update(subsets=sub['c'], subset_significance=sub['sig'])
        if est['sig'][0] < min_significance or ok.sum() < max(4, (k + 1) // 2):
            why = 'the correlation peak is not significant in enough parts of the image'
            return _skip(img, info, why), np.zeros(nx), info
        med = float(np.median(sub['c'][ok]))
        dev = np.abs(sub['c'][ok] - med)
        s = 1.4826 * float(np.median(dev))
        sigma = s / np.sqrt(ok.sum())
        info['edge_rad_std'] = sigma
        far = np.count_nonzero(dev > 4.5 * s + tol)
        if abs(c - med) > 3 * sigma + tol or abs(c) < 3 * sigma or far > ok.sum() // 4:
            return _skip(img, info, 'the parts of the image disagree on the estimate'), np.zeros(nx), info
    v = _offsets(nx, band[0], band[1])
    phi = np.where(np.abs(v) <= 1, c * v ** 2, 0.0)
    out = _keep(img, _apply(img, phi), info, guard, sb)
    return out, (phi if info['applied'] else np.zeros(nx)), info


def _pga_columns(cols, band, kw):
    """pga on host columns [n, m] whose band is first moved to the center of the spectrum (pga integrates its
    gradient across centered bins) -> (phi [n] in FFT order, zero outside the band; contrast before, after;
    iterations run)."""
    n = cols.shape[0]
    k0 = int(round(band[0] * n))
    ramp = np.exp(-2j * np.pi * k0 * np.arange(n) / n)[:, None]
    g, ph, hist = pga(cols * ramp, **kw)
    phi = np.roll(np.fft.ifftshift(ph), k0)
    phi = np.where(np.abs(_offsets(n, band[0], band[1])) <= 1, phi, 0.0)
    return phi, float(_contrast(cols)), float(_contrast(g)), len(hist)


def _pga_global(img, *, lines=0.3, guard=True, max_lines=1024, band_db=30.0, **pga_kwargs):
    """Global phase error of any shape: the existing pga on the columns _select keeps (copied to the host), applied
    like quadratic's phase. pga_kwargs go to pga, which uses every column it is given.
    guard: as quadratic's. Consistency: pga is run again on each contiguous half of the columns (as many iterations
    as the estimate took, at least three), and the rms over the band (|v| <= 0.9, constant and linear terms
    removed) of the estimate must be at least three times rms_rad_std, half the rms of the difference of the two
    halves' estimates. Result: quadratic's test on the contrast of the whole image. -> (img2, info)"""
    _check(img)
    nx = img.shape[0]
    info = dict(band=None, lines=0, rms_rad=0.0, rms_rad_std=None, contrast_before=0.0, contrast_after=0.0, gain=0.0,
                gain_std=0.0, part_gain=0.0, part_gain_std=0.0, applied=False, reason=None, phi=np.zeros(nx))
    sel = _select(img, lines, max_lines, band_db)
    if sel is None:
        return _skip(img, info, 'the image holds no data'), info
    cols, band = sel
    cols, kw = _xp.host(cols), dict(pga_kwargs, lines=1.0)
    m = cols.shape[1]
    phi, _, _, its = _pga_columns(cols, band, kw)
    v = _offsets(nx, band[0], band[1])
    inb, mid = np.abs(v) <= 1, np.abs(v) <= 0.9
    sb = _colsums(img)
    info.update(band=band[:2], lines=int(m), rms_rad=float(np.sqrt(np.mean(phi[inb] ** 2))))
    info['contrast_before'] = info['contrast_after'] = _gain(sb, sb)[0]
    if guard:
        if m < 4:
            return _skip(img, info, 'too few columns to check the estimate'), info
        kw['iterations'] = max(3, its)                              # halves that disagree need not converge
        half = [_pga_columns(cols[:, sl], band, kw)[0] for sl in (slice(0, m // 2), slice(m // 2, m))]
        info['rms_rad_std'] = float(0.5 * _detrend((half[0] - half[1])[mid], v[mid]).std())
        if _detrend(phi[mid], v[mid]).std() < 3 * info['rms_rad_std']:
            return _skip(img, info, 'the parts of the image disagree on the estimate'), info
    out = _keep(img, _apply(img, phi), info, guard, sb)
    if info['applied']:
        info['phi'] = phi
    return out, info


def _starts(n, size, step):
    """Starts of blocks of `size` every `step` samples that cover n samples; the last one ends at n."""
    s = list(range(0, n - size + 1, step))
    return np.array(s if s[-1] + size == n else s + [n - size])


def _polyfit2(x, y, z, w, order):
    """Weighted robust fit of z [nb, nc] at points (x, y) by polynomials of total degree `order`: iteratively
    reweighted least squares, Tukey's biweight on the residuals scaled by their median absolute deviation (at
    least 0.02), a point taking the smallest of its weights over the nc columns. The degree drops until there are
    three points with weight per term; a coordinate that does not vary gets no terms; and a small ridge (1e-3 of
    the total weight, on every term but the constant; x and y are of order one) keeps terms the points do not
    determine near zero.
    -> (terms [(a, b)] for x^a y^b, coefficients [nt, nc], robust weights [nb])."""
    use = w > 0
    vx, vy = np.ptp(x[use]) > 1e-9, np.ptp(y[use]) > 1e-9
    for o in range(int(order), -1, -1):
        terms = [(a, b) for a in range(o + 1 if vx else 1) for b in range((o - a if vy else 0) + 1)]
        A = np.stack([x ** a * y ** b for a, b in terms], 1)
        if use.sum() >= 3 * len(terms) or o == 0:
            break
    rw, ridge = np.ones(len(x)), np.diag([0.0] + [1.0] * (len(terms) - 1))
    for _ in range(12):
        s = np.sqrt(w * rw)[:, None]
        lhs = np.concatenate([A * s, np.sqrt(1e-3 * (w * rw).sum()) * ridge])
        coef = np.linalg.lstsq(lhs, np.concatenate([z * s, np.zeros((len(terms), z.shape[1]))]), rcond=None)[0]
        res = z - A @ coef
        scale = np.maximum(1.4826 * np.median(np.abs(res[use]), axis=0), 0.02)
        rw = np.clip(1 - (np.abs(res) / (4.685 * scale)).max(1) ** 2, 0, None) ** 2
    return terms, coef, rw


def _polyval2(terms, coef, x, y):
    """The fit of _polyfit2 at (x, y) (broadcast) -> [nc, ...]"""
    return sum(c[(slice(None),) + (None,) * np.ndim(x * y)] * (x ** a * y ** b) for (a, b), c in zip(terms, coef))


def _apply_strips(img, st, B, pad, win, V, coef):
    """The space-variant correction. Strip k is rows st[k] - pad to st[k] + B + pad of img (n = B + 2 pad rows; zeros
    beyond the image). The phase sum_n coef(k)[n, j] V[n, b] is removed from azimuth bin b of column j of its
    n-point transform, the zeros of img are put back, and its middle B rows are added to the result with the
    weights win[k] [B], which sum to one over the strips at every row. The padding rows take what the circular
    transform wraps around the strip's ends. coef(k) -> [nc, ny] float32."""
    nx, ny = img.shape
    n, k = B + 2 * pad, _xp.kind(img)
    xp = _xp.module(img)

    def rows(i, cols=slice(None)):                                  # strip i, complex64 [n, columns]
        r0, r1 = max(0, st[i] - pad), min(nx, st[i] + B + pad)
        blk = img[r0:r1, cols].astype(xp.complex64)
        if r1 - r0 == n:
            return blk
        return xp.pad(blk, ((r0 - (st[i] - pad), st[i] + B + pad - r1), (0, 0)))
    if k == 'jax':
        f, Vd = _jit('strip'), xp.asarray(V)
        wp = np.zeros((len(st), n), np.float32)
        wp[:, pad:pad + B] = win
        cuts = np.unique(np.concatenate([st, st + B]))
        live, parts = {}, []
        for r0, r1 in zip(cuts[:-1], cuts[1:]):                     # rows between cuts: the same strips throughout
            for i in [i for i in live if st[i] + B <= r0]:
                del live[i]
            for i in np.flatnonzero((st <= r0) & (st + B >= r1)):
                if i not in live:
                    live[i] = f(rows(i), Vd, xp.asarray(coef(i)), xp.asarray(wp[i]))
            parts.append(sum(live[i][pad + r0 - st[i]:pad + r1 - st[i]] for i in sorted(live)))
        return xp.concatenate(parts, axis=0)
    if k == 'cupy':
        import cupyx.scipy.fft as cf
        out, Vd = xp.zeros((nx, ny), xp.complex64), xp.asarray(V)
        for i, i0 in enumerate(st):
            blk = rows(i)
            Cd = xp.asarray(coef(i))
            ph = sum(Vd[n][:, None] * Cd[n][None, :] for n in range(len(Vd)))
            t = cf.fft(blk, axis=0)
            t *= xp.cos(ph) - 1j * xp.sin(ph)
            t = cf.ifft(t, axis=0, overwrite_x=True)[pad:pad + B]
            t *= xp.asarray(win[i])[:, None] * (blk[pad:pad + B] != 0)
            out[i0:i0 + B] += t
        return out
    out, w = np.zeros((nx, ny), np.complex64), int(max(4, _BLOCK['numpy'] // (2 * n)))
    Cf = [coef(i) for i in range(len(st))]

    def work(j):
        q = np.empty((n, min(w, ny - j)), np.complex64)
        for i, i0 in enumerate(st):
            blk = rows(i, slice(j, j + w))
            ph = sum(V[n][:, None] * Cf[i][n, j:j + w][None, :] for n in range(len(V)))
            np.cos(ph, out=q.real)
            np.sin(-ph, out=q.imag)
            t = _sfft.fft(blk, axis=0)
            t *= q
            t = _sfft.ifft(t, axis=0, overwrite_x=True)[pad:pad + B]
            t *= win[i][:, None] * (blk[pad:pad + B] != 0)
            out[i0:i0 + B, j:j + w] += t
    _threads(work, range(0, ny, w))
    return out


def spatially_variant(img, *, block=512, overlap=0.5, method='quadratic', order=2, guard=True, iterations=6,
                      tol=0.02, poly=4, block_lines=64, min_fill=0.5, min_corr=0.1, min_contrast=1.5, band_db=30.0,
                      max_lines=1024, pga_kwargs=None):
    """Estimate and remove an azimuth phase error that varies over the scene.

    img: as for quadratic (complex baseband [nx, ny], axis 0 azimuth, NumPy, CuPy or JAX array, not modified).
    The image is cut into blocks of `block` pixels (an int or (azimuth, range); no larger than the image) that
    overlap by the fraction `overlap`, and each block's phase error is estimated as coefficients of powers of v
    (the offset from the center of the occupied band, detected once for the image, in half band widths):
    method 'quadratic': the map-drift coefficient of v^2 (see quadratic; `iterations`, `tol`), on the device;
    method 'pga': the existing pga on the block (host; pga_kwargs), its estimate fitted by a polynomial of degree
    `poly` in v, of which the coefficients of v^2 .. v^poly are kept.
    Each block uses about block_lines evenly spaced columns (the same columns for overlapping blocks). A block is
    dropped when less than min_fill of its pixels are nonzero, when its own correction lowers its contrast, when
    the correlation of its two looks is below min_corr ('quadratic') or when its contrast after pga is below
    min_contrast ('pga'; 1 is speckle). Each coefficient is then fitted over the scene by a polynomial of total
    degree `order` in the block positions (weights: r^2 / (1.01 - r^2) for the correlation r, or the contrast;
    robust to outlying blocks; a lower degree when there are fewer than three blocks per term).
    A block's position is the centroid of its squared intensity, where the scatterers that set its estimate are.

    Correction: strips of `block` rows that overlap along azimuth, each extended at both ends by the length of the
    blur (about |c| / (pi hf) pixels for c rad at the band edge; zeros beyond the image), are transformed along
    axis 0, every column of a strip has the phase of the fit at (strip center, that column) removed, so the
    correction follows range exactly, and the strips are cross-faded along azimuth with weights that sum to one.
    Outside the occupied band, where the image has no signal, the phase is not cut to zero as quadratic's is but
    rolls off smoothly to zero at the bin opposite the band center: a jump at the band edge would give the
    correction a long impulse response, which a strip cuts. Exact zeros stay exact zeros. The blur must be short
    compared with the block.

    guard: keep the result only if it passes quadratic's test on the contrast of the whole image and of its parts
    (every column is measured, before and after); otherwise, or when no block gives an estimate or the fitted
    phase stays below tol everywhere, the input is returned as it came with info['applied'] = False and
    info['reason'].

    -> (img2 complex64 [nx, ny] in the same place as img; info: dict(method; band: (f0, hf) cycles per pixel;
    block: (rows, columns); powers: the power of v of each coefficient; positions [nb, 2] (pixels);
    estimates [nb, nc] (rad at the band edge; nan for blocks without one); weights [nb]: weight in the fit after
    the robust iterations, zero for dropped blocks; terms: [(a, b)] for the monomials x^a y^b, x and y being the
    pixel coordinates along axes 0 and 1 scaled to -1 .. 1 across the image; coefficients [nt, nc]; rms [nc]:
    weighted rms of the fit residual (rad); edge_rad: (min, max) over the scene of the fitted v^2 coefficient;
    contrast_before, contrast_after, gain, gain_std, part_gain, part_gain_std: as quadratic's; applied; reason))."""
    _check(img)
    if method not in ('quadratic', 'pga'):
        raise ValueError("method must be 'quadratic' or 'pga'")
    if not 0 <= overlap < 1:
        raise ValueError('overlap must be in [0, 1)')
    xp = _xp.module(img)
    nx, ny = img.shape
    B0, B1 = (int(b) for b in (block if np.ndim(block) else (block, block)))
    B0, B1 = min(B0, nx), min(B1, ny)
    if B0 < 16:
        raise ValueError('blocks need at least 16 rows')
    powers = (2,) if method == 'quadratic' else tuple(range(2, max(2, int(poly)) + 1))
    info = dict(method=method, band=None, block=(B0, B1), powers=powers, positions=np.zeros((0, 2)),
                estimates=np.zeros((0, len(powers))), weights=np.zeros(0), terms=[], coefficients=None, rms=None,
                edge_rad=(0.0, 0.0), contrast_before=0.0, contrast_after=0.0, gain=0.0, gain_std=0.0, part_gain=0.0,
                part_gain_std=0.0, applied=False, reason=None)
    sel = _select(img, 1.0, max_lines, band_db)
    if sel is None:
        return _skip(img, info, 'the image holds no data'), info
    band = sel[1]
    f0, hf, _ = band
    info['band'] = band[:2]
    # block estimates, from every s-th column
    s = max(1, B1 // max(1, int(block_lines)))
    Bm = max(1, B1 // s)
    st0 = _starts(nx, B0, max(1, int(round(B0 * (1 - overlap)))))
    st1 = _starts(len(range(0, ny, s)), Bm, max(1, int(round(Bm * (1 - overlap)))))
    idx = st1[:, None] + np.arange(Bm)[None, :]
    cen = np.array([(i0, j0 * s) for i0 in st0 for j0 in st1], np.float64)                 # block corners for now
    est, wt = np.full((len(cen), len(powers)), np.nan), np.zeros(len(cen))
    if method == 'quadratic':
        host = _xp.kind(img) == 'numpy'

        def row(i):                                                 # the blocks of one strip of rows together
            A = img[st0[i]:st0[i] + B0, ::s].astype(xp.complex64)[:, idx]                # [B0, blocks, Bm]
            fill = _xp.host((A != 0).mean((0, 2)))
            r = _map_drift(A, band, iterations, tol, 1 if host else -1)
            ok = (fill >= min_fill) & (r['corr'] >= min_corr) & (r['after'] >= r['before']) & np.isfinite(r['c'])
            sl = slice(i * len(st1), (i + 1) * len(st1))
            r2 = np.clip(r['corr'], 0, 1) ** 2
            est[sl, 0], wt[sl] = np.where(ok, r['c'], np.nan), np.where(ok, r2 / (1.01 - r2), 0.0)
            cen[sl] += _centroid(A) * (1, s)
        if host:
            _threads(row, range(len(st0)))
        else:
            for i in range(len(st0)):
                row(i)
    else:
        sub = _xp.host(img[:, ::s]).astype(np.complex64, copy=False)
        vb = _offsets(B0, f0, hf)
        inb = np.abs(vb) <= 0.98
        Pv = np.linalg.pinv(np.vander(vb[inb], powers[-1] + 1, increasing=True))
        kw = dict(pga_kwargs or {})

        def one(n):
            i0, j0 = st0[n // len(st1)], st1[n % len(st1)]
            blk = sub[i0:i0 + B0, j0:j0 + Bm]
            cen[n] += _centroid(blk[:, None, :])[0] * (1, s)
            if np.count_nonzero(blk) < max(1, min_fill * blk.size):
                return
            blk = blk[:, np.any(blk != 0, axis=0)]
            phi, before, after, _ = _pga_columns(blk, band, kw)
            if after >= max(before, min_contrast) and np.all(np.isfinite(phi)):
                est[n], wt[n] = (Pv @ phi[inb])[2:], after
        for n in range(len(cen)):                                   # pga holds the interpreter lock: no threads
            one(n)
    info.update(positions=cen, estimates=est)
    if not np.any(wt > 0):
        return _skip(img, info, 'no block gave an estimate', weights=wt), info
    # fit over the scene
    xn, yn = (cen[:, 0] - (nx - 1) / 2) / (nx / 2), (cen[:, 1] - (ny - 1) / 2) / (ny / 2)
    terms, coef, rw = _polyfit2(xn, yn, np.nan_to_num(est), wt / wt.max(), order)
    wf = wt / wt.max() * rw
    res = np.nan_to_num(est) - np.stack([xn ** a * yn ** b for a, b in terms], 1) @ coef
    yc = (np.arange(ny) - (ny - 1) / 2) / (ny / 2)
    fit = lambda i: _polyval2(terms, coef, (st0[i] + (B0 - 1) / 2 - (nx - 1) / 2) / (nx / 2), yc)   # [nc, ny]
    c2 = np.stack([fit(i)[0] for i in range(len(st0))])
    info.update(weights=wf, terms=terms, coefficients=coef, edge_rad=(float(c2.min()), float(c2.max())),
                rms=np.sqrt((wf[:, None] * res ** 2).sum(0) / max(wf.sum(), 1e-300)))
    sb = _colsums(img)
    info['contrast_before'] = info['contrast_after'] = _gain(sb, sb)[0]
    if max(np.abs(fit(i)).sum(0).max() for i in range(len(st0))) < tol:
        return _skip(img, info, 'the fitted phase is below tol'), info
    # correction: strips along azimuth, cross-faded
    raw = np.sin(np.pi * (np.arange(B0) + 0.5) / B0) ** 2
    tot = np.zeros(nx)
    for i0 in st0:
        tot[i0:i0 + B0] += raw
    win = np.stack([raw / tot[i0:i0 + B0] for i0 in st0]).astype(np.float32)
    blur = max((np.abs(fit(i)) * np.array(powers)[:, None]).sum(0).max() for i in range(len(st0))) / (2 * np.pi * hf)
    n = _sfft.next_fast_len(B0 + 2 * min(B0, int(np.ceil(blur)) + 8), real=False)
    n += (n - B0) % 2
    vb = _offsets(n, f0, hf)
    roll = 0.5 * (1 + np.cos(np.pi * np.clip((np.abs(vb) - 1) / max(0.5 / hf - 1, 1e-9), 0, 1)))
    V = np.stack([vb ** p * roll for p in powers]).astype(np.float32)
    out = _apply_strips(img, st0, B0, (n - B0) // 2, win, V, lambda i: fit(i).astype(np.float32))
    return _keep(img, out, info, guard, sb), info


def image_autofocus(img, mode, **kw):
    """Autofocus of a finished complex baseband image [nx, ny] (axis 0 azimuth; NumPy, CuPy or JAX array, result in
    the same place) -> (img2, info), info['mode'] being the mode. mode:
    'quadratic': quadratic(img, **kw); info also holds phi.
    'pga': one phase error of any shape for the image, estimated by pga on the columns quadratic would use (kw:
    lines, guard, max_lines, band_db, and pga's own keywords) and applied like quadratic's (axis-0 transform,
    occupied band only, zeros kept). info: dict(phi [nx] (rad, fftfreq order, as quadratic's), band, lines, rms_rad,
    rms_rad_std (None without guard), contrast_before, contrast_after, gain, gain_std, part_gain, part_gain_std,
    applied, reason).
    'spatially_variant': spatially_variant(img, **kw).   'spatially_variant_pga': the same with method='pga'.
    None or False: img itself and dict(mode=None, applied=False)."""
    if mode is None or mode is False:
        return img, dict(mode=None, applied=False)
    if mode == 'quadratic':
        out, phi, info = quadratic(img, **kw)
        info['phi'] = phi
    elif mode == 'pga':
        out, info = _pga_global(img, **kw)
    elif mode == 'spatially_variant':
        out, info = spatially_variant(img, **kw)
    elif mode == 'spatially_variant_pga':
        out, info = spatially_variant(img, **dict(kw, method='pga'))
    else:
        raise ValueError(f'image_autofocus: unknown mode {mode!r}; valid modes are {", ".join(_MODES)}, None')
    info['mode'] = mode
    return out, info
