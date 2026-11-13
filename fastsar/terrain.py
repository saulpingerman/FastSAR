"""Terrain-informed focusing: the defocus of scatterers off the focus plane, computed from the collection geometry
and a DEM and removed from a ground-plane spotlight image. Nothing is estimated from the image.

    out = fastsar.process_cphd('scene_CPHD.cphd', dem='Copernicus_DSM_COG_10_N19_00_W156_00_DEM.tif', geoid=22.4)
    out = fastsar.form_cphd('scene_CPHD.cphd', axes='los')                 # or, on a form_cphd image:
    info = fastsar.terrain.focus(out, fastsar.terrain.as_dem('dem.tif', geoid=22.4))

Model. Backprojection onto a plane focuses the scatterers of the plane. A scatterer at x, at height h above it,
appears at the point y of the plane that has its range and range rate at the center of aperture (COA; the layover
point: planes.ground_to_slant, the projection a SICD states), and its echo differs from that of a scatterer at y
by the phase
    phi_p = 4 pi fc / c (|y - a_p| - |x - a_p|)
on pulse p (antenna a_p). phi_p and its rate vanish at the COA by the choice of y; what is left is quadratic and
higher across the aperture. To first order in h, with w the unit normal of the slant plane (w_z its vertical
component) and A_w the component along w of the antenna's acceleration relative to the scene,
    phi(t) = 2 pi h A_w t^2 / (lambda R w_z),
t from the COA: the track's curvature out of the slant plane times the height. For a straight track every point of
the circle about the track through y has the range history of y, and phi is zero exactly. At the ends of an
aperture of length L flown at speed v (broadside: L / R is the angle it subtends, lambda / (2 L / R) the azimuth
resolution) the phase is
    pi / 2 (h / lambda) (L / R)^2 (R / rho) / w_z,        rho = v^2 / A_w,
rho being the radius of curvature of the track out of the slant plane: proportional to the height, to the square
of the aperture at a given wavelength, and to R / rho, which from a circular orbit of radius r is about
R R_e / r^2 times w_z at broadside (R_e the Earth's radius): a tenth from low Earth orbit. On two spotlights from
orbit the closed form is within 1.1% of the exact phase, which coefficients computes from the positions
themselves, with no expansion.

Image. On the baseband ground image on axes along the line of sight (planes.deramp; axis 0 azimuth), pulse p of a
scatterer imaged at y lies at the azimuth frequency xi_p = spx e1 . (u_p - u_COA) 2 fc / c cycles per pixel, u the
unit vector from the antenna to y, and the image's spectrum along axis 0 carries exp(1j * phi_p) there
(autofocus.quadratic's convention and sign). phi is fitted by a polynomial in v, the offset of xi from the center
of the band of the aperture in half band widths (terms 0 to `order`: the constant and the linear term put the
scatterer at y with the phase of its range at the COA, which is where and how SICD's projection expects it). The
quadratic term carries the phase from orbit, where the cubic is a hundredth of it or less; a short range and a
track with a jerk need more terms (the simulated collection of the tests: order 6 to 8, the cubic the largest).

Range walk. phi is the phase at the center frequency. At frequency f the phase is f / fc phi_p, and pulse p lies at
another azimuth frequency (each pulse's spectral line is a ray from the origin of the wavenumber plane), so that
over the image's 2-D spectrum the phase is a function psi(v, w) of the azimuth frequency and of w = eta / eta_c,
the range frequency (eta, cycles per pixel of axis 1) as a fraction of the center frequency's (eta_c). From orbit
the part that varies with w, the range walk, is about -w c_2 v^2, B / (2 fc) of the phase at the band's corners:
a correction along axis 0 alone leaves 0.009 of the phase in rms, -34 dB at 3 rad. _Model fits
psi = sum_n c_n v^n + w sum_n d_n v^n + w^2 sum_n e_n v^n to the exact phase at every pulse and five frequencies.

Correction (focus). The height of the surface each pixel shows is found on a lattice by solving the layover
relation (heights); the coefficients, exact functions of position and height, follow from it. The image is
corrected in strips of `block` rows along azimuth, each read with as many rows again on either side as the phase
displaces, on the device that holds the image, with its exact zeros kept. Within a strip, per column, each pixel
is corrected for its own height: the strip is corrected for a few heights that span the column's heights over the
strip's rows (Chebyshev nodes, as many as keep the interpolation in height within tol / 2 at the band edge) and
each pixel takes the interpolation between them at its height, which is what exact backprojection onto the
surface does pixel by pixel. The range walk, proportional to the height, is removed by a series in w (the terms
of exp(-1j w h D) - 1, h the pixel's height, D the walk per meter), each term's azimuth part applied with the
strip's and its w part through each row's spectrum. A strip takes its middle row's geometry; from orbit the phase
at a fixed height changes by under a thousandth along 64 m, at a short range strips shorten until the change over
half a strip is under tol / 8. The image stays on its plane: pixels, geolocation and the SICD's projection are
unchanged, and heights still enter geolocation as they did (products.geolocate, geocode_image). It is not an
orthorectification.

Limits. One correction serves the heights that share a pixel: in layover (several surface points at one range and
Doppler) their mean is used, weighted by the ground area each contributes, as surface_points does. The walk's
series stops at eight terms: beyond about 4.5 rad of walk at the band's corners the walk is left as it is (the
simulated 10 km collection of the tests at 300 m: 13 to 19 rad, two range cells; Etna's deepest ground: 0.25 rad). The interpolation in height stops at eight heights, enough for 4.3 rad of
phase across one column's heights within a strip (Etna: 1.6 rad at most). The correction treats the image's
spectrum near a pixel as plane waves; at a short range the wavefronts' curvature over the correction's kernel
limits it (a 300 m cliff at 10 km: -35 dB against exact backprojection onto the surface), from orbit it does not
(Etna: -49 to -52 dB, the plane former's own agreement with exact backprojection). The DEM's own errors remain,
for the data-driven autofocus that runs after it. Moving-beam collections are not handled: the model holds for them
pixel by pixel with the pulses that illuminate the pixel and their own center of aperture, and the correction
would run per patch of the mosaic on its track axes.

Formation onto the surface itself is the exact kernel's work: fastsar.backproject at surface_points.
"""
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft

from . import _xp, planes
from . import autofocus as af

C = 299792458.0

__all__ = ['TerrainWarning', 'as_dem', 'phase_error', 'coefficients', 'heights', 'surface_points', 'focus']


class TerrainWarning(UserWarning):
    """A DEM that does not agree with the collection's own scene height, or does not cover the scene."""


def as_dem(dem, geoid=None):
    """The surface as the function dem(lat, lon) -> height above the WGS-84 ellipsoid (m; arrays of degrees) that
    focus, products.geolocate and products.geocode_image take.

    dem: a GeoTIFF path, or a list of them (tiles: those on one grid, as adjacent Copernicus tiles are, joined
    into one raster so that heights are interpolated across their seams; else the first that covers a point gives
    its height); a function of (lat, lon); or a number, a surface of that constant height.
    geoid: the height of the geoid above the ellipsoid at the scene (the undulation N, m), added to every height:
    h_ellipsoid = H_geoid + N. Copernicus GLO-30 and GLO-90 give heights above EGM2008, SRTM above EGM96; N is
    +22 m in Hawaii and -26 m in North Dakota, and reaches -106 and +85 m. A file needs it stated: 0.0 for a DEM
    of ellipsoid heights. For a function or a number None means 0."""
    if isinstance(dem, (list, tuple)) or hasattr(dem, '__fspath__') or isinstance(dem, str):
        if geoid is None:
            raise ValueError('a DEM file needs geoid=: the height of the geoid above the ellipsoid at the scene in '
                             'meters (Copernicus DEM: EGM2008; SRTM: EGM96), or 0.0 for a DEM of ellipsoid heights')
        from .products import read_dem
        return read_dem(list(dem) if isinstance(dem, (list, tuple)) else dem, offset=float(geoid))
    n = 0.0 if geoid is None else float(geoid)
    if callable(dem):
        return dem if n == 0 else (lambda lat, lon: dem(lat, lon) + n)
    try:
        h = float(dem) + n
    except (TypeError, ValueError):
        raise ValueError('dem must be a GeoTIFF path, a function of (lat, lon) or a height in meters, '
                         f'got {dem!r}') from None
    if not np.isfinite(h):
        raise ValueError(f'dem must be a finite height in meters, got {dem!r}')
    return lambda lat, lon: np.full(np.broadcast(lat, lon).shape, h)


def _scene(out):
    """The grid, the slant frame, the antenna positions of the pulses used and the center frequency of a
    ground-plane spotlight image on axes along the line of sight."""
    if out.get('plane', 'ground') != 'ground' or out.get('mode', 'spotlight') != 'spotlight' or \
            out.get('axes', 'los') != 'los':
        raise ValueError("terrain focusing takes a ground-plane spotlight image on axes along the line of sight "
                         "(form_cphd(axes='los')); a moving beam is not handled")
    nx, ny, o, e1, e2, spx, spy = planes._grid(out)
    if abs(e1[2]) > 1e-9 or abs(e2[2]) > 1e-9:
        raise ValueError('terrain focusing takes an image on a horizontal plane of the local frame')
    fc = 0.5 * (out['band'][0] + out['band'][1])
    return (nx, ny, o, e1, e2, spx, spy), planes._frame(out), planes._used(out), fc


def _some(ant, n):
    return ant[np.unique(np.round(np.linspace(0, len(ant) - 1, int(n))).astype(int))]


def _phase(grid, fr, ant, fc, y, h):
    """xi [..., p] (cycles per pixel) and phi [..., p] (rad) of plane points y [..., 3] showing heights h [...]."""
    o, e1, spx = grid[2], grid[3], grid[5]
    a, r = planes.ground_to_slant(fr, y)
    x = planes.slant_to_ground(fr, a, r, o[2] + h)
    d = y[..., None, :] - ant
    R = np.linalg.norm(d, axis=-1)
    phi = 4 * np.pi * fc / C * (R - np.linalg.norm(x[..., None, :] - ant, axis=-1))
    uc = y - fr['arp']
    xi = 2 * fc / C * spx * ((d @ e1) / R - ((uc @ e1) / np.linalg.norm(uc, axis=-1))[..., None])
    return xi, phi


def phase_error(out, i, j, h, pulses=257):
    """The exact model at pixels (i, j) (fractional, any shape) that show a surface h meters above the image plane
    (local z): -> (xi [..., p], the azimuth frequency of each of `pulses` evenly spaced pulses of the aperture in
    the baseband image, cycles per pixel of axis 0; phi [..., p], the phase (rad) the scatterer's echo has there
    over that of a scatterer of the plane at the pixel: the image's axis-0 spectrum carries exp(1j * phi))."""
    grid, fr, ant, fc = _scene(out)
    i, j, h = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (i, j, h)))
    y = grid[2] + i[..., None] * grid[5] * grid[3] + j[..., None] * grid[6] * grid[4]
    return _phase(grid, fr, _some(ant, pulses), fc, y, h)


def _band(grid, fr, ant, fc):
    """(f0, hf): center and half width (cycles per pixel) of the azimuth band of the aperture over the image."""
    nx, ny, o, e1, e2, spx, spy = grid
    c = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1], [(nx - 1) / 2, (ny - 1) / 2]])
    xi = _phase(grid, fr, _some(ant, 33), fc, o + c[:, :1] * spx * e1 + c[:, 1:] * spy * e2, np.zeros(5))[0]
    return float(0.5 * (xi.min() + xi.max())), float(0.5 * (xi.max() - xi.min()))


def _fit(xi, phi, band, order):
    """Least-squares coefficients [..., order + 1] of phi in powers of v = (xi - f0) / hf; the largest residual."""
    v = (xi - band[0]) / band[1]
    A = v[..., None] ** np.arange(order + 1)
    c = (np.linalg.pinv(A) @ phi[..., None])[..., 0]
    return c, float(np.abs((A @ c[..., None])[..., 0] - phi).max()) if phi.size else 0.0


def coefficients(out, i, j, h, *, order=4, band=None, pulses=257):
    """The polynomial of the model at pixels (i, j) showing heights h (as phase_error's).
    -> (c [..., order + 1]: phi = sum_n c[n] v^n (rad), v = (xi - f0) / hf, so that c[2] is the quadratic phase at
    the band edge in autofocus.quadratic's sense; band: (f0, hf) in cycles per pixel, as given or the band of the
    aperture over the image; the largest residual of the fits (rad))."""
    grid, fr, ant, fc = _scene(out)
    band = _band(grid, fr, ant, fc) if band is None else (float(band[0]), float(band[1]))
    c, res = _fit(*phase_error(out, i, j, h, pulses), band, int(order))
    return c, band, res


def _spectrum(grid, fr, ant, fc, bw, eta_c, band, y, h, nf=5):
    """The exact phase over the 2-D spectrum: for plane points y [..., 3] showing heights h [...], at pulse p and
    frequency f (nf of them over the band bw about fc), (v, w, psi) [..., p, f]: v the azimuth frequency as
    coefficients takes it, w = eta / eta_c the range frequency (eta in cycles per pixel of axis 1), and
    psi = f / fc phi_p, the phase the image's spectrum carries there."""
    o, e1, e2, spx, spy = grid[2], grid[3], grid[4], grid[5], grid[6]
    a, r = planes.ground_to_slant(fr, y)
    x = planes.slant_to_ground(fr, a, r, o[2] + h)
    d = y[..., None, :] - ant
    R = np.linalg.norm(d, axis=-1)
    dR = R - np.linalg.norm(x[..., None, :] - ant, axis=-1)
    u = d / R[..., None]
    uc = y - fr['arp']
    uc = uc / np.linalg.norm(uc, axis=-1, keepdims=True)
    f = fc + 0.5 * bw * np.linspace(-1.0, 1.0, nf)
    k = lambda e, sp: 2 / C * sp * (f * (u @ e)[..., None] - fc * (uc @ e)[..., None, None])
    return (k(e1, spx) - band[0]) / band[1], k(e2, spy) / eta_c, 4 * np.pi / C * f * dR[..., None]


class _Model:
    """The phase over the image's 2-D spectrum as a function of pixel and height, fitted to the exact model at 5 x 5
    points of the image and four heights, over the pulses and five frequencies of the band:
        psi = sum_n c_n v^n + w sum_n d_n v^n + w^2 sum_n e_n v^n,
    v the azimuth frequency as coefficients takes it, w = eta / eta_c the range frequency relative to the center
    frequency (eta in cycles per pixel of axis 1, eta_c its value at the center frequency on the line of sight of
    the scene center). c is the phase along axis 0 at the center of the range band; d the range walk, the part of
    the phase that scales with frequency (with the frequency, each pulse's azimuth frequency moves too); e its
    curvature over the band, a few thousandths of the phase at the band's corners from orbit. Each
    coefficient is sum over m = 1..3, a, b = 0..2 of K h^m s^a t^b (s, t, h scaled to -1 .. 1; no term without h,
    which is the plane). self.res: the largest residual of the azimuth fits at the center frequency (which sets
    the order), res2 of the 2-D fits; self.err: the largest misfit at the nodes (of d at the band's edge, w =
    bw / (2 fc)); self.e: the largest w^2 part (rad at w = 1)."""

    def __init__(self, grid, fr, ant, fc, band, order, hmin, hmax, bw, eta_c):
        nx, ny, o, e1, e2, spx, spy = grid
        self.n, self.H, self.order = (nx, ny), max(abs(hmin), abs(hmax), 1e-3), order
        lo, hi = min(hmin, 0.0), max(hmax, 0.0)
        hs = np.array([lo + (hi - lo) * q for q in (0.0, 1 / 3, 2 / 3, 1.0)])
        hs = hs[np.abs(hs) > 1e-6 * self.H]
        if len(hs) < 3:
            hs = self.H * np.array([-1.0, -0.5, 0.5, 1.0])
        s = np.linspace(-1, 1, 5)
        S, T, Hh = np.meshgrid(s, s, hs, indexing='ij')
        y = o + ((S + 1) * (nx - 1) / 2 * spx)[..., None] * e1 + ((T + 1) * (ny - 1) / 2 * spy)[..., None] * e2
        a = _some(ant, 129)
        self.res = _fit(*_phase(grid, fr, a, fc, y, Hh), band, order)[1]       # the azimuth fit at the center frequency
        v, w, psi = (q.reshape(q.shape[:3] + (-1,)) for q in _spectrum(grid, fr, a, fc, bw, eta_c, band, y, Hh))
        A = np.concatenate([w[..., None] ** k * v[..., None] ** np.arange(order + 1) for k in (0, 1, 2)], -1)
        c = (np.linalg.pinv(A) @ psi[..., None])[..., 0]
        self.res2 = float(np.abs((A @ c[..., None])[..., 0] - psi).max())      # the 2-D fit
        self.e = float(np.abs(c[..., 2 * (order + 1):]).sum(-1).max())          # the w^2 part (rad at w = 1)
        c = c.reshape(-1, 3 * (order + 1))
        A = self._basis(S.ravel(), T.ravel(), Hh.ravel() / self.H)
        self.K = np.linalg.lstsq(A, c, rcond=None)[0]                            # [27, 3 (order + 1)]
        miss, wb = np.abs(A @ self.K - c), 0.5 * bw / fc
        self.err = float(max(miss[:, :order + 1].max(), wb * miss[:, order + 1:2 * order + 2].max(),
                             wb * wb * miss[:, 2 * order + 2:].max()))

    @staticmethod
    def _basis(s, t, h):
        return np.stack([h ** m * s ** a * t ** b for m in (1, 2, 3) for a in range(3) for b in range(3)], -1)

    def __call__(self, i, j, h):
        """-> (c, d, e) stacked [3 (order + 1), ...] at pixels (i, j) showing heights h (broadcast)."""
        i, j, h = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (i, j, h)))
        A = self._basis(2 * i / max(self.n[0] - 1, 1) - 1, 2 * j / max(self.n[1] - 1, 1) - 1, h / self.H)
        return np.moveaxis(A @ self.K, -1, 0)

    def parts(self, i, j):
        """The polynomial in height at row i and columns j [...]: P [3, 3 (order + 1), ...], the coefficients being
        sum_m P[m - 1] (h / H)^m."""
        s = 2 * float(i) / max(self.n[0] - 1, 1) - 1
        t = 2 * np.asarray(j, np.float64) / max(self.n[1] - 1, 1) - 1
        st = np.stack([s ** a * t ** b for a in range(3) for b in range(3)])           # [9, ...]
        return np.stack([np.tensordot(self.K[9 * m:9 * m + 9], st, axes=(0, 0)) for m in range(3)])

    def at(self, P, h, deriv=False):
        """The coefficients (or their derivatives in height, per meter) [3 (order + 1), ...] of parts P at h."""
        q = np.asarray(h, np.float64) / self.H
        if deriv:
            return (P[0] + 2 * q * P[1] + 3 * q * q * P[2]) / self.H
        return q * (P[0] + q * (P[1] + q * P[2]))


def _lattice(n, sp, post):
    """Pixel coordinates of lattice nodes about `post` meters apart that span 0 .. n - 1."""
    return np.linspace(0.0, n - 1.0, max(2, int(np.ceil((n - 1) * sp / post)) + 1))


def heights(out, dem, *, post=None):
    """The height of the surface each pixel shows, through layover: for nodes of a lattice over the image, the
    point of the DEM that has the node's range and range rate at the COA.

    dem: dem(lat, lon) -> height above the ellipsoid (as_dem). post: the lattice spacing in meters (default 10 m,
    more for an image wider than 400 nodes). The DEM is sampled once on a lattice of that spacing over the image
    and as far beyond it as its heights displace, and the relation is solved by a scan over height: where the
    circle of constant range and Doppler through a node meets the surface once, that crossing; where it meets it
    several times (layover), the mean of the crossings weighted by the ground area per pixel each contributes;
    where it does not (outside the DEM), NaN.

    -> dict(h [ni, nj]: height above the image plane (m, local z); i [ni], j [nj]: the nodes' pixel coordinates;
    post; layover: the fraction of nodes with several crossings; outside: the fraction without a height;
    plane: the heights of the DEM above the plane at the nodes' own positions [ni, nj] (no layover))."""
    from .io import local_to_ecf, ecf_to_geodetic
    from .products import sample
    grid, fr, _, _ = _scene(out)
    nx, ny, o, e1, e2, spx, spy = grid
    if post is None:
        post = max(10.0, (nx - 1) * spx / 400, (ny - 1) * spy / 400)
    post = float(post)
    li, lj = _lattice(nx, spx, post), _lattice(ny, spy, post)
    Y = o + (li * spx)[:, None, None] * e1 + (lj * spy)[None, :, None] * e2
    a, r = planes.ground_to_slant(fr, Y)
    at = lambda h: planes.slant_to_ground(fr, a, r, o[2] + h)

    def above(s, t):                                    # the DEM above the plane on the lattice s x t (m on e1, e2)
        lat, lon, hp = ecf_to_geodetic(local_to_ecf(o + s[:, None, None] * e1 + t[None, :, None] * e2, out['meta']))
        return np.asarray(dem(lat, lon), np.float64) - hp
    s0, s1, t0, t1 = 0.0, (nx - 1) * spx, 0.0, (ny - 1) * spy
    own = above(li * spx, lj * spy)
    if not np.isfinite(own).any():
        raise ValueError('the DEM does not cover the image')
    lo, hi = float(np.nanmin(own)), float(np.nanmax(own))
    for _ in range(6):                                  # the lattice grows until it holds what its heights displace
        pad = 0.25 * (hi - lo) + post
        d = np.concatenate([(at(h)[::max(1, len(li) - 1), ::max(1, len(lj) - 1)] - o).reshape(-1, 3)
                            for h in (lo - pad, hi + pad)])
        b = (min(s0, (d @ e1).min()), max(s1, (d @ e1).max()), min(t0, (d @ e2).min()), max(t1, (d @ e2).max()))
        s = b[0] - post + post * np.arange(int(np.ceil((b[1] - b[0]) / post)) + 3)
        t = b[2] - post + post * np.arange(int(np.ceil((b[3] - b[2]) / post)) + 3)
        H = above(s, t)
        l2, h2 = float(np.nanmin(H)), float(np.nanmax(H))
        done = l2 >= lo - pad and h2 <= hi + pad
        lo, hi = min(lo, l2), max(hi, h2)
        if done:
            break
    # crossings of H(position at height h) - h along the circle, by a scan over h
    step = min(max(post / 2, (hi - lo) / 160), max((hi - lo) / 8, 1e-3))
    hs = lo - step + step * np.arange(int(np.ceil((hi - lo) / step)) + 3)
    num, den, cnt = np.zeros(Y.shape[:2]), np.zeros(Y.shape[:2]), np.zeros(Y.shape[:2], np.int32)
    g0 = None
    for h in hs:
        p = at(h) - o
        g1 = sample(H, np.stack([(p @ e1 - s[0]) / post, (p @ e2 - t[0]) / post], -1)) - h
        if g0 is not None:
            hit = (g0 * g1 <= 0) & (g0 != g1)
            dg = np.where(hit, g0 - g1, 1.0)
            w = np.where(hit, 1.0 / np.maximum(np.abs(dg) / step, 0.05), 0.0)     # ground area per pixel, at most 20
            num += np.where(hit, w * (h - step + step * g0 / dg), 0.0)
            den += w
            cnt += hit & (g1 != 0)                                         # a crossing at a scan height counts once
        g0 = g1
    return dict(h=np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), i=li, j=lj, post=post,
                layover=float((cnt > 1).mean()), outside=float((den == 0).mean()), plane=own)


def _at(hm, i, j):
    """The lattice heights at pixels (i, j): bilinear, the nearest node's beyond the image, 0 where there is none."""
    from .products import sample
    h = np.nan_to_num(hm['h'])
    f = lambda x, l: np.clip(np.asarray(x, np.float64), l[0], l[-1]) / (l[1] - l[0])
    i, j = np.broadcast_arrays(f(i, hm['i']), f(j, hm['j']))
    return sample(h, np.stack([i, j], -1))


def surface_points(out, dem, i, j, *, post=None):
    """The points [..., 3] (local frame) of the surface dem (as_dem; or heights' output) that pixels (i, j)
    (fractional, any shape) show: exact backprojection at them, fastsar.backproject(S, ant, fmin, df, points - c)
    for a history referenced to c, forms the image on the surface itself, pixel for pixel on the image's grid."""
    grid, fr, _, _ = _scene(out)
    nx, ny, o, e1, e2, spx, spy = grid
    hm = dem if isinstance(dem, dict) else heights(out, dem, post=post)
    i, j = np.broadcast_arrays(np.asarray(i, np.float64), np.asarray(j, np.float64))
    a, r = planes.ground_to_slant(fr, o + i[..., None] * spx * e1 + j[..., None] * spy * e2)
    return planes.slant_to_ground(fr, a, r, o[2] + _at(hm, i, j))


def focus(out, dem, *, order=None, block=None, post=None, tol=0.05, check=True):
    """Remove the terrain's defocus from a ground-plane spotlight image (see the module docstring).

    out: form_cphd(axes='los')'s output, or process_cphd's ground image: brought to baseband in place if it is not
    (planes.deramp), and its image (a NumPy, CuPy or JAX array) replaced by the corrected one, on the same device.
    dem: dem(lat, lon) -> height above the WGS-84 ellipsoid in meters (as_dem makes one of a GeoTIFF or a number),
        or heights' output for this image (its h may be edited: heights from another source, on its lattice).
    order: the degree of the polynomial in azimuth frequency, 2 to 8; None: the lowest that fits the model within
        tol / 5 at the heights of the scene (2 from orbit; more for a short range or a track with a jerk). Terms
        that stay under tol / 10 are dropped.
    block: rows of a strip of the correction (default: about 64 m, at least 32 rows and eight times the blur, and
        fewer, down to 16, where the geometry changes along azimuth by more than tol / 8 over half a strip).
    post: heights' lattice spacing (m). tol: nothing is applied when the phase stays below it everywhere (rad).
    check: warn (TerrainWarning, and a note) when the median of the DEM over the image is further from the image
    plane, the collection's stated scene height, than 10 m plus three times the DEM's standard deviation there:
    a wrong geoid offset on flat ground, a wrong tile, heights in other units.

    -> info, also out['terrain']: dict(applied; reason (why not, or None); band: (f0, hf), the azimuth band of the
    aperture in cycles per pixel; powers: of v, the terms applied; edge_rad: (least, greatest) over the image of
    each term's coefficient [len(powers), 2] (rad at the band edge); quadratic_rad: the same of the quadratic term;
    height: (least, median, greatest) of the surface above the image plane (m); offset, relief: the median and
    the standard deviation of the DEM above the plane at the pixels' own positions (m); layover, outside: heights';
    block, post; order; fit_rad: the largest misfit of the polynomial and of its model over the image (rad); and,
    when applied, levels: the most heights a strip column was corrected for; height_rad: the largest phase across
    one column's heights within a strip, half of it (rad at the band edge); walk_rad: the largest range walk (rad
    at the band's corners); walk_terms: the most terms of its series; walk_left_rad: the walk left where it
    is beyond the series, 0 elsewhere)."""
    if order is not None and not 2 <= int(order) <= 8:
        raise ValueError(f'order must be 2 to 8, got {order!r}')
    grid, fr, ant, fc = _scene(out)
    planes.deramp(out)
    img = out['image']
    nx, ny, o, e1, e2, spx, spy = grid
    hm = dem if isinstance(dem, dict) else heights(out, dem, post=post)
    h = np.nan_to_num(hm['h'])
    own = hm['plane'][np.isfinite(hm['plane'])]
    band = _band(grid, fr, ant, fc)
    info = dict(applied=False, reason=None, band=band, powers=(), edge_rad=np.zeros((0, 2)), quadratic_rad=(0.0, 0.0),
                height=(float(h.min()), float(np.median(h)), float(h.max())), offset=float(np.median(own)),
                relief=float(own.std()), layover=hm['layover'], outside=hm['outside'], block=None, post=hm['post'],
                order=None, fit_rad=0.0)
    out['terrain'] = info
    notes = out.setdefault('notes', [])
    if check and abs(info['offset']) > 10.0 + 3 * info['relief']:
        msg = (f"the DEM's median height over the image is {info['offset']:+.0f} m from the image plane (the scene "
               f"height the collection states), with a relief of {info['relief']:.0f} m: check the geoid offset "
               '(as_dem(geoid=): ellipsoid height = geoid height + undulation), the tile and the units')
        warnings.warn(msg, TerrainWarning, stacklevel=2)
        notes.append('terrain: ' + msg)
    if hm['outside'] > 0:
        notes.append(f"terrain: {100 * hm['outside']:.1f}% of the image is outside the DEM and is left as it is")
    bw = float(out['band'][1] - out['band'][0])
    eta_c = 2 / C * fc * spy * float(fr['e2'] @ e2)                  # cycles per pixel of axis 1 at w = 1
    for n in range(2, 9) if order is None else [int(order)]:
        model = _Model(grid, fr, ant, fc, band, n, float(h.min()), float(h.max()), bw, eta_c)
        if model.res <= tol / 5:
            break
    cde = model(hm['i'][:, None], hm['j'][None, :], h)                         # [3 (order + 1), ni, nj]
    c, d, e = cde[:n + 1], cde[n + 1:2 * n + 2], cde[2 * n + 2:]
    info.update(order=n, fit_rad=max(model.res, model.err))
    top = np.abs(c).reshape(len(c), -1).max(1)
    info['quadratic_rad'] = (float(c[2].min()), float(c[2].max()))
    if top.sum() < tol:
        info['reason'] = 'the terrain phase is below tol everywhere'
        return info
    keep = np.flatnonzero(top >= tol / 10)
    hf = min(band[1], 0.5)
    # the defocus in pixels: the steepest phase over the band and across the gap between its edges, where the
    # phase returns from one edge's value to the other's
    vs = _bins(4096, 0.0, hf, keep)
    slope = np.abs(np.diff(vs, axis=1)) * (4096 * hf)                          # d v^p / dv, per power
    blur = float((np.abs(c[keep]).reshape(len(keep), -1).max(1)[:, None] * slope).sum(0).max()) / (2 * np.pi * hf)
    if block is not None:
        B = int(block)
    else:                                       # about 64 m, and short enough that the geometry holds over a strip
        B = max(32, 2 * int(round(32.0 / spx)), 16 * int(np.ceil(blur / 2)))
        B = min(B, max(16, _rows(model, keep, float(h.min()), float(h.max()), nx, ny, 0.125 * tol)))
    B = max(1, min(B, nx))
    pad = int(np.ceil(blur)) + _PAD
    nb = sfft.next_fast_len(B + 2 * pad, real=False)
    nb += (nb - B) % 2
    pad = (nb - B) // 2
    # the range walk: along axis 1 through each row's spectrum, w = eta / eta_c in the band, continued smoothly past it
    eb = min(0.5, float(np.abs(np.asarray(planes.support(fr, ant, out['band'], e1, e2)['k2'])).max() * spy))
    nr = sfft.next_fast_len(ny + 64, real=False)
    Wq = (eb / eta_c * _bins(nr, 0.0, eb, [1])[0]).astype(np.float32)
    wmax = float(np.abs(Wq).max())
    big = lambda q, s: np.flatnonzero(np.abs(q).reshape(len(q), -1).max(1) * s >= tol / 10)
    sets = (keep, big(d, wmax), big(e, wmax ** 2))
    bins = [_bins(nb, band[0], hf, k) for k in sets]                           # [terms, nb] for c, d and e
    plan = _plan(model, hm, sets, bins, B, nx, ny, wmax, 0.5 * tol)
    out['image'] = _apply(img, plan, bins, Wq, B, pad, nr)
    info.update(applied=True, powers=tuple(int(p) for p in keep), block=B,
                edge_rad=np.stack([c[keep].reshape(len(keep), -1).min(1), c[keep].reshape(len(keep), -1).max(1)], 1),
                levels=max(len(p['um']) for p in plan), walk_terms=max(len(p['terms']) for p in plan),
                walk_rad=max(p['xw'] for p in plan), height_rad=max(p['x'] for p in plan),
                walk_left_rad=max(p['left'] for p in plan))
    q, hh = info['quadratic_rad'], info['height']
    notes.append(f'terrain: focused with the DEM, quadratic phase {q[0]:+.2f} to {q[1]:+.2f} rad at the band edge for '
                 f'surface heights {hh[0]:+.0f} to {hh[2]:+.0f} m about the image plane')
    return info


def _rows(model, keep, hmin, hmax, nx, ny, tol):
    """The most rows over which a strip may take its middle row's geometry: at the extreme heights and across the
    image, twice the rows over which the phase at a fixed height changes by tol at the band edge."""
    i, j = np.linspace(0, nx - 1, 17), np.linspace(0, ny - 1, 5)
    rate = 0.0
    for hh in (hmin, hmax):
        c = model(i[:, None], j[None, :], hh)[keep]                                 # [terms, 17, 5]
        rate = max(rate, float(np.abs(np.diff(c, axis=1)).sum(0).max()) / (i[1] - i[0]))
    return int(2 * tol / rate) if rate > 0 else nx


def _bins(n, f0, hf, powers):
    """v^p [len(powers), n] on the bins of an n-point FFT (v as autofocus._offsets: |v| <= 1 in the band f0 +- hf),
    continued across the gap between the band's two edges (|v| > 1, out to the bin where they meet) by the cubic
    that joins the values and slopes of v^p at the edges: a phase built of them is smooth around the circle, so its
    impulse response stays short, and blocks cut from the image need no more padding than the defocus itself."""
    v = af._offsets(n, f0, hf)
    L = max(1.0 / hf - 2.0, 1e-9)                                     # the gap's length in units of v
    g = np.clip(np.where(v > 1, v - 1, v + 1 / hf - 1) / L, 0.0, 1.0)  # 0 at the upper edge, 1 at the lower
    h00, h10, h01, h11 = 2 * g ** 3 - 3 * g ** 2 + 1, g ** 3 - 2 * g ** 2 + g, 3 * g ** 2 - 2 * g ** 3, g ** 3 - g ** 2
    rows = []
    for p in powers:
        p = int(p)
        gap = h00 + h10 * p * L + h01 * (-1.0) ** p + h11 * p * (-1.0) ** (p - 1) * L
        rows.append(np.where(np.abs(v) <= 1, v ** p, gap))
    return np.array(rows, np.float32).reshape(len(rows), n)


_PAD = 16                                       # rows beyond the defocus that a strip reads on either side
_MOST = 8                                       # Chebyshev nodes in height, and terms of the walk, at most per strip


def _nodes(x, tol):
    """The fewest Chebyshev nodes in u at which interpolation of exp(-1j x u), |u| <= 1, errs by at most tol
    (the bound x^M / (2^(M - 1) M!)), up to _MOST."""
    for m in range(1, _MOST + 1):
        if x ** m / (2.0 ** (m - 1) * math.factorial(m)) <= tol:
            return m
    return _MOST


def _series(x, tol):
    """The terms q = 1 .. Q of the series of exp(-1j x) - 1 that leave at most tol (x^(Q + 1) / (Q + 1)!), up to
    _MOST; 0 when x is under tol, and None when _MOST terms would leave more than a radian."""
    if x <= tol:
        return 0
    if x ** (_MOST + 1) / math.factorial(_MOST + 1) > 1.0:
        return None
    q = 1
    while q < _MOST and x ** (q + 1) / math.factorial(q + 1) > tol:
        q += 1
    return q


def _plan(model, hm, sets, bins, B, nx, ny, wmax, tol):
    """Per strip of B rows (the last one ends at the image's last row), on the host, float32 for the device: the
    heights its pixels show, as lattice rows Hr [r, ny] interpolated along columns and each row's weights Wr [B, r];
    per column the middle hb and half range dl of the heights over the strip; Chebyshev nodes um in
    u = (h - hb) / dl, as many as keep the interpolation in height within tol at the band edge in the strip's worst
    column (x: that column's half range in radians), and the coefficients C [len(um), terms, ny] at the nodes'
    heights. The range walk: the derivatives in height of d and e at hb, D1 and E1 [terms, ny], so that the walk of
    a pixel showing h is w h D1 + w^2 h E1 (d and e are proportional to the height to a part in a thousand); the
    terms (a, b) of the series of exp(-1j (w h D1 + w^2 h E1)) - 1 that leave tol (xw, xe: the largest of each part
    at the band's corners); and the exponent's coefficients C0 at hb when no node lies there."""
    keep, dkeep, ekeep = sets
    li, lj, h = hm['i'], hm['j'], np.nan_to_num(hm['h'])
    jc = np.arange(ny, dtype=np.float64)
    st = list(range(0, nx - B + 1, B))
    if st[-1] + B < nx:
        st.append(nx - B)
    step = li[1] - li[0] if len(li) > 1 else 1.0
    r = int(np.ceil(B / step)) + 2
    vmax = [np.abs(v).max(1) if len(v) else np.zeros(0) for v in bins]
    n1 = model.order + 1
    plan = []
    for s0 in st:
        f = np.clip((np.arange(s0, s0 + B) - li[0]) / step, 0, max(len(li) - 1 - 1e-9, 0))
        k0 = f.astype(np.intp)
        lo = int(k0.min())
        Hr = np.stack([np.interp(jc, lj, h[min(q, len(li) - 1)]) for q in range(lo, lo + r)])     # [r, ny]
        Wr = np.zeros((B, r))
        Wr[np.arange(B), k0 - lo] = 1 - (f - k0)
        Wr[np.arange(B), np.minimum(k0 - lo + 1, r - 1)] += f - k0
        span = np.concatenate([Wr[[0, -1]] @ Hr, Hr[1 + k0[0] - lo:k0[-1] - lo + 1]])
        hlo, hhi = span.min(0), span.max(0)
        hb, dl = 0.5 * (hlo + hhi), 0.5 * (hhi - hlo)
        P = model.parts(s0 + 0.5 * (B - 1), jc)                               # [3, 3 n1, ny]
        dh = model.at(P, hb, deriv=True)
        x = float((dl * (np.abs(dh[keep]) * vmax[0][:, None]).sum(0)).max())
        M = _nodes(x, tol)
        um = np.cos(np.pi * (2 * np.arange(M) + 1) / (2 * M)) if M > 1 else np.zeros(1)
        Cm = np.stack([model.at(P, hb + dl * u)[keep] for u in um]).astype(np.float32)
        D1, E1 = dh[n1 + dkeep].astype(np.float32), dh[2 * n1 + ekeep].astype(np.float32)
        top = np.abs(hb) + dl
        part = lambda Q, k, s: s * float((top * (np.abs(Q) * vmax[k][:, None]).sum(0)).max()) if len(Q) else 0.0
        xw, xe = part(D1, 1, wmax), part(E1, 2, wmax ** 2)
        qa, qb = _series(xw, tol), _series(xe, tol)
        if qa is None or qb is None:                                # beyond the series: the walk is left
            qa = qb = 0
            left = xw + xe
        else:
            left = 0.0
        terms = [(i, j) for i in range(qa + 1) for j in range(qb + 1) if i or j]
        C0 = model.at(P, hb)[keep].astype(np.float32) if terms and M % 2 == 0 else None
        plan.append(dict(s0=s0, Hr=Hr.astype(np.float32), Wr=Wr.astype(np.float32), hb=hb.astype(np.float32),
                         dl=dl.astype(np.float32), um=um, C=Cm, C0=C0, D1=D1, E1=E1, terms=terms, x=x, xw=xw, xe=xe,
                         left=left))
    return plan


def _strip(xp, ft, blk, bins, C, C0, D1, E1, Hr, Wr, hb, dl, um, terms, B, pad, kw):
    """The azimuth part of one strip's correction on the device of blk [B + 2 pad, columns] (xp: its array module,
    ft: its FFT module, kw: the FFT's keyword arguments): the image corrected for the height of each pixel, by
    interpolation between the corrections at the Chebyshev nodes of the strip's heights, column by column,
    -> (z [B, columns], the heights h [B, columns], and per walk term (a, b) the image whose rows, multiplied by
    w^(a + 2 b) in their spectrum and by (-1j h)^(a + b) / (a! b!), add that term of the walk)."""
    V, Vd, Ve = bins
    F = ft.fft(blk, axis=0, **kw)
    h = sum(Wr[:, q][:, None] * Hr[q][None, :] for q in range(Wr.shape[1]))     # [B, columns]: not a matrix product
    M = len(um)
    if M > 1:
        u = xp.clip((h - hb[None, :]) / xp.where(dl > 0, dl, 1)[None, :], -1, 1)
    z, Emid = 0, None
    for m in range(M):
        ph = sum(V[k][:, None] * C[m][k][None, :] for k in range(V.shape[0]))
        E = xp.cos(ph) - 1j * xp.sin(ph)
        Z = ft.ifft(F * E, axis=0, **kw)[pad:pad + B]
        if M > 1:
            L = 1.0
            for l in range(M):
                if l != m:
                    L = L * ((u - float(um[l])) / float(um[m] - um[l]))
            Z = Z * L
        z = z + Z
        if 2 * m + 1 == M:
            Emid = E
    T = []
    if terms:
        if Emid is None:
            ph = sum(V[k][:, None] * C0[k][None, :] for k in range(V.shape[0]))
            Emid = xp.cos(ph) - 1j * xp.sin(ph)
        PD = sum(Vd[k][:, None] * D1[k][None, :] for k in range(Vd.shape[0])) if len(Vd) else 0
        PE = sum(Ve[k][:, None] * E1[k][None, :] for k in range(Ve.shape[0])) if len(Ve) else 0
        G, qa, qb = F * Emid, max(t[0] for t in terms), max(t[1] for t in terms)
        for a in range(qa + 1):
            Gb = G
            for b in range(qb + 1):
                if a or b:
                    T.append(ft.ifft(Gb, axis=0, **kw)[pad:pad + B])
                if b < qb:
                    Gb = Gb * PE
            if a < qa:
                G = G * PD
    return z, h, T


def _walk(xp, ft, z, h, T, terms, Wq, nr, kw):
    """Add the walk's terms to z [B, ny] through each row's spectrum (nr points, the image's columns at its start)."""
    ny = z.shape[1]
    for (a, b), t in zip(terms, T):
        R = ft.ifft(ft.fft(t, n=nr, axis=1, **kw) * Wq ** (a + 2 * b), axis=1, **kw)[:, :ny]
        z = z + ((-1j) ** (a + b) / (math.factorial(a) * math.factorial(b))) * h ** (a + b) * R
    return z


_JIT = {}


def _apply(img, plan, bins, Wq, B, pad, nr):
    """The strips of plan on the device of img (NumPy: all cores; CuPy; JAX, one compilation per number of nodes
    and walk terms) -> the corrected image, complex64, on the same device, its exact zeros kept. The strips do not
    overlap; the last one, which ends at the last row, gives only the rows the others leave."""
    nx, ny = img.shape
    k, xp = _xp.kind(img), _xp.module(img)

    def rows(s0, cols=slice(None)):                                  # strip rows s0 - pad .. s0 + B + pad, zeros beyond
        r0, r1 = max(0, s0 - pad), min(nx, s0 + B + pad)
        blk = img[r0:r1, cols].astype(xp.complex64)
        if r1 - r0 == B + 2 * pad:
            return blk
        return xp.pad(blk, ((r0 - (s0 - pad), s0 + B + pad - r1), (0, 0)))
    done = [max(p['s0'], plan[i - 1]['s0'] + B) if i else 0 for i, p in enumerate(plan)]   # first row each one gives
    args = lambda p, cs=slice(None): (p['C'][:, :, cs], None if p['C0'] is None else p['C0'][:, cs], p['D1'][:, cs],
                                      p['E1'][:, cs], p['Hr'][:, cs], p['Wr'], p['hb'][cs], p['dl'][cs])
    keep0 = lambda z, blk: xp.where(blk[pad:pad + B] != 0, z, 0).astype(xp.complex64)
    if k in ('jax', 'cupy'):
        if k == 'jax':
            import jax
            import jax.numpy as jnp
            ft, put = jnp.fft, jnp.asarray
        else:
            import cupyx.scipy.fft as ft
            put = xp.asarray
        dev = [put(v) for v in bins] + [put(Wq)]
        parts, out = [], (xp.empty((nx, ny), xp.complex64) if k == 'cupy' else None)
        for p, r0 in zip(plan, done):
            a = [None if v is None else put(v) for v in args(p)]
            um, terms = p['um'], p['terms']

            def one(blk, bins, Wq, C, C0, D1, E1, Hr, Wr, hb, dl, um=um, terms=terms):
                z, h, T = _strip(xp, ft, blk.astype(xp.complex64), bins, C, C0, D1, E1, Hr, Wr, hb, dl, um, terms,
                                 B, pad, {})
                return keep0(_walk(xp, ft, z, h, T, terms, Wq, nr, {}), blk)
            if k == 'jax':
                key = (len(um), tuple(terms), B, pad, nr)
                if key not in _JIT:
                    _JIT[key] = jax.jit(one)
                parts.append(_JIT[key](rows(p['s0']), dev[:3], dev[3], *a)[r0 - p['s0']:])
            else:
                out[r0:p['s0'] + B] = one(rows(p['s0']), dev[:3], dev[3], *a)[r0 - p['s0']:]
        return jnp.concatenate(parts, axis=0) if k == 'jax' else out
    out = np.empty((nx, ny), np.complex64)
    w = int(max(64, af._BLOCK['numpy'] // (2 * (B + 2 * pad))))
    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        for p, r0 in zip(plan, done):
            _host_strip(ex, p, r0, rows, args, keep0, bins, Wq, B, pad, nr, ny, w, out)
    return out


def _host_strip(ex, p, r0, rows, args, keep0, bins, Wq, B, pad, nr, ny, w, out):
    """One strip on the host: its column blocks, then the walk's row transforms, on the threads of ex."""
    terms, z0, off = p['terms'], out[r0:p['s0'] + B], r0 - p['s0']
    if terms:
        hs, zs, Ts = (np.empty((B, ny), np.float32), np.empty((B, ny), np.complex64),
                      np.empty((len(terms), B, ny), np.complex64))
        nz = np.empty((B, ny), bool)

    def work(j):                                                # column blocks on all cores
        cs = slice(j, j + w)
        blk = rows(p['s0'], cs)
        z, h, T = _strip(np, sfft, blk, bins, *args(p, cs), p['um'], terms, B, pad, {})
        if terms:
            hs[:, cs], zs[:, cs], nz[:, cs] = h, z, blk[pad:pad + B] != 0
            for t, Tt in zip(Ts, T):
                t[:, cs] = Tt
        else:
            z0[:, cs] = keep0(z, blk)[off:]
    list(ex.map(work, range(0, ny, w)))
    if terms:                                                   # the walk's transforms along whole rows

        def walk(i):
            rs = slice(max(i, off), min(i + 8, B))
            z = _walk(np, sfft, zs[rs], hs[rs], Ts[:, rs], terms, Wq, nr, dict(workers=1))
            z0[rs.start - off:rs.stop - off] = np.where(nz[rs], z, 0)
        list(ex.map(walk, [i for i in range(0, B, 8) if i + 8 > off]))


def _stage(out, dem):
    """process_cphd's stage: focus where the image is, or, on a device without room for the corrected image beside
    it (CuPy: one more image; JAX, which joins its blocks: two and a half), on the host and back, with a note."""
    from . import memory
    img, notes = out['image'], out['notes']
    k, n0 = _xp.kind(img), len(out['notes'])
    if k == 'cupy':
        room = 1.25 * img.nbytes + (1 << 26) <= _xp.cuda_budget()
    else:
        room = k == 'numpy' or _xp.jax_budget() is None or 2.5 * img.nbytes <= _xp.jax_budget()
    del img
    if room:
        try:
            return focus(out, dem)
        except Exception as e:                                       # the device held the image, not the correction
            if k != 'cupy' or type(e).__name__ != 'OutOfMemoryError':
                raise
            del notes[n0:]
    host = dict(out, image=_xp.host(out['image']))
    out['image'] = None
    if k == 'cupy':
        memory.cuda_release()
    out['terrain'] = focus(host, dem)
    out['image'] = _xp.device(host['image'], 'cuda' if k == 'cupy' else 'jax')
    notes.append('terrain focusing ran on the host: the device had no memory for its correction')
    return out['terrain']


def _record(sm, info):
    """ImageFormation.Processing of a SICD for a product focused with a DEM: the step and its range, nothing about
    the grid, which is the plane's still."""
    from ._sicdxml import el
    q, h = info['quadratic_rad'], info['height']
    par = dict(MinQuadraticPhase=q[0], MaxQuadraticPhase=q[1], MinHeight=h[0], MaxHeight=h[2])
    sm.ImageFormation.element.append(el('Processing', [el('Type', 'TERRAIN PHASE CORRECTION'), el('Applied', True)]
                                        + [el('Parameter', float(v), name=n) for n, v in par.items()]))
