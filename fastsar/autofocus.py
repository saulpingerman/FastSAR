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
"""
import numpy as np

from .sim import C


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
    for _ in range(rounds):
        _, pb, _ = pga(former(S * np.exp(-1j * phi)[:, None]) * ramp, **(pga_kwargs or {}))
        phi = _detrend(phi + np.interp(b, m, pb), b)
    return former(S * np.exp(-1j * phi)[:, None]), phi
