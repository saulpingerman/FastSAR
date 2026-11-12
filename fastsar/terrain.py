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

Correction (focus). The height of the surface each pixel shows is found on a lattice by solving the layover
relation (heights); the polynomial's coefficients, exact functions of position and height, follow from it; and the
phase is removed by the strips of autofocus.spatially_variant: per column, so that it follows range from pixel to
pixel, and cross-faded along azimuth over `block` rows, on the device that holds the image, with its exact zeros
kept. The image stays on its plane: pixels, geolocation and the SICD's projection are unchanged, and heights still
enter geolocation as they did (products.geolocate, geocode_image). It is not an orthorectification.

Limits. One correction serves the heights that share a pixel: in layover (several surface points at one range and
Doppler) their mean is used, weighted by the ground area each contributes, and within a strip the height is
averaged along azimuth over half a block, so that cliffs keep a residual. The phase is taken at the center
frequency: over the band it varies by B / (2 fc) of itself (the range error behind it, phi lambda / (4 pi), walks
the response across range), which a correction along axis 0 leaves: negligible while |phi| B / (2 fc) stays under
a quarter of a radian, 8 rad of phase at a 3% band. The DEM's own errors remain, for the data-driven autofocus
that runs after it. Moving-beam collections are not handled: the model holds for them pixel by pixel with the
pulses that illuminate the pixel and their own center of aperture, and the correction would run per patch of the
mosaic on its track axes.

Formation onto the surface itself is the exact kernel's work: fastsar.backproject at surface_points.
"""
import warnings

import numpy as np

from . import _xp, planes

C = 299792458.0

__all__ = ['TerrainWarning', 'as_dem', 'phase_error', 'coefficients', 'heights', 'surface_points', 'focus']


class TerrainWarning(UserWarning):
    """A DEM that does not agree with the collection's own scene height, or does not cover the scene."""


def as_dem(dem, geoid=None):
    """The surface as the function dem(lat, lon) -> height above the WGS-84 ellipsoid (m; arrays of degrees) that
    focus, products.geolocate and products.geocode_image take.

    dem: a GeoTIFF path, or a list of them (tiles; the first that covers a point gives its height); a function of
    (lat, lon); or a number, a surface of that constant height.
    geoid: the height of the geoid above the ellipsoid at the scene (the undulation N, m), added to every height:
    h_ellipsoid = H_geoid + N. Copernicus GLO-30 and GLO-90 give heights above EGM2008, SRTM above EGM96; N is
    +22 m in Hawaii and -26 m in North Dakota, and reaches -106 and +85 m. A file needs it stated: 0.0 for a DEM
    of ellipsoid heights. For a function or a number None means 0."""
    if isinstance(dem, (list, tuple)) or hasattr(dem, '__fspath__') or isinstance(dem, str):
        if geoid is None:
            raise ValueError('a DEM file needs geoid=: the height of the geoid above the ellipsoid at the scene in '
                             'meters (Copernicus DEM: EGM2008; SRTM: EGM96), or 0.0 for a DEM of ellipsoid heights')
        from .products import read_dem
        tiles = [read_dem(p, offset=float(geoid)) for p in (dem if isinstance(dem, (list, tuple)) else [dem])]
        if len(tiles) == 1:
            return tiles[0]

        def mosaic(lat, lon):
            h = tiles[0](lat, lon)
            for t in tiles[1:]:
                h = np.where(np.isfinite(h), h, t(lat, lon))
            return h
        return mosaic
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


class _Model:
    """The coefficients as functions of pixel and height, fitted to the exact model at 5 x 5 points of the image
    and four heights: sum over m = 1..3, a, b = 0..2 of K h^m s^a t^b (s, t, h scaled to -1 .. 1; no term without
    h, which is the plane). self.err: the largest misfit at the nodes (rad)."""

    def __init__(self, grid, fr, ant, fc, band, order, hmin, hmax):
        nx, ny, o, e1, e2, spx, spy = grid
        self.n, self.H = (nx, ny), max(abs(hmin), abs(hmax), 1e-3)
        lo, hi = min(hmin, 0.0), max(hmax, 0.0)
        hs = np.array([lo + (hi - lo) * q for q in (0.0, 1 / 3, 2 / 3, 1.0)])
        hs = hs[np.abs(hs) > 1e-6 * self.H]
        if len(hs) < 3:
            hs = self.H * np.array([-1.0, -0.5, 0.5, 1.0])
        s = np.linspace(-1, 1, 5)
        S, T, Hh = np.meshgrid(s, s, hs, indexing='ij')
        y = o + ((S + 1) * (nx - 1) / 2 * spx)[..., None] * e1 + ((T + 1) * (ny - 1) / 2 * spy)[..., None] * e2
        c, self.res = _fit(*_phase(grid, fr, _some(ant, 129), fc, y, Hh), band, order)
        A = self._basis(S.ravel(), T.ravel(), Hh.ravel() / self.H)
        self.K = np.linalg.lstsq(A, c.reshape(-1, order + 1), rcond=None)[0]              # [27, order + 1]
        self.err = float(np.abs(A @ self.K - c.reshape(-1, order + 1)).max())

    @staticmethod
    def _basis(s, t, h):
        return np.stack([h ** m * s ** a * t ** b for m in (1, 2, 3) for a in range(3) for b in range(3)], -1)

    def __call__(self, i, j, h):
        """-> c [order + 1, ...] at pixels (i, j) showing heights h (broadcast)."""
        i, j, h = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (i, j, h)))
        A = self._basis(2 * i / max(self.n[0] - 1, 1) - 1, 2 * j / max(self.n[1] - 1, 1) - 1, h / self.H)
        return np.moveaxis(A @ self.K, -1, 0)


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
    block: rows of a strip of the correction (default: about 64 m, at least 32 rows and eight times the blur).
    post: heights' lattice spacing (m). tol: nothing is applied when the phase stays below it everywhere (rad).
    check: warn (TerrainWarning, and a note) when the median of the DEM over the image is further from the image
    plane, the collection's stated scene height, than 10 m plus three times the DEM's standard deviation there:
    a wrong geoid offset on flat ground, a wrong tile, heights in other units.

    -> info, also out['terrain']: dict(applied; reason (why not, or None); band: (f0, hf), the azimuth band of the
    aperture in cycles per pixel; powers: of v, the terms applied; edge_rad: (least, greatest) over the image of
    each term's coefficient [len(powers), 2] (rad at the band edge); quadratic_rad: the same of the quadratic term;
    height: (least, median, greatest) of the surface above the image plane (m); offset, relief: the median and
    the standard deviation of the DEM above the plane at the pixels' own positions (m); layover, outside: heights';
    block, post; order; fit_rad: the largest misfit of the polynomial and of its model over the image (rad))."""
    from . import autofocus as af
    import scipy.fft as sfft
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
    for n in range(2, 9) if order is None else [int(order)]:
        model = _Model(grid, fr, ant, fc, band, n, float(h.min()), float(h.max()))
        if model.res <= tol / 5:
            break
    c = model(hm['i'][:, None], hm['j'][None, :], h)                           # [order + 1, ni, nj]
    info.update(order=n, fit_rad=max(model.res, model.err))
    top = np.abs(c).reshape(len(c), -1).max(1)
    info['quadratic_rad'] = (float(c[2].min()), float(c[2].max()))
    if top.sum() < tol:
        info['reason'] = 'the terrain phase is below tol everywhere'
        return info
    keep = np.flatnonzero(top >= tol / 10)
    c = c[keep]
    # strips along azimuth: each column's coefficients are the lattice's, averaged over the strip's rows
    hf = min(band[1], 0.5)
    blur = (np.abs(c) * keep[:, None, None]).sum(0).max() / (2 * np.pi * hf)
    B = int(block) if block is not None else max(32, 2 * int(round(32.0 / spx)), 16 * int(np.ceil(blur / 2)))
    B = max(1, min(B, nx))
    st = af._starts(nx, B, max(1, B // 2))
    raw = np.sin(np.pi * (np.arange(B) + 0.5) / B) ** 2
    tot = np.zeros(nx)
    for i0 in st:
        tot[i0:i0 + B] += raw
    win = np.stack([raw / tot[i0:i0 + B] for i0 in st])
    li, lj = hm['i'], hm['j']
    f = np.clip((np.arange(nx) - li[0]) / (li[1] - li[0]), 0, len(li) - 1 - 1e-9)
    k0, fr0 = f.astype(np.intp), f % 1.0
    jc = np.arange(ny)

    def coef(k):
        w = np.zeros(len(li))
        sl = slice(st[k], st[k] + B)
        np.add.at(w, k0[sl], win[k] * (1 - fr0[sl]))
        np.add.at(w, k0[sl] + 1, win[k] * fr0[sl])
        row = np.tensordot(w / w.sum(), c, axes=(0, 1))                                    # [terms, nj]
        return np.stack([np.interp(jc, lj, r) for r in row]).astype(np.float32)
    n = sfft.next_fast_len(B + 2 * min(B, int(np.ceil(blur)) + 8), real=False)
    n += (n - B) % 2
    vb = af._offsets(n, band[0], hf)
    roll = 0.5 * (1 + np.cos(np.pi * np.clip((np.abs(vb) - 1) / max(0.5 / hf - 1, 1e-9), 0, 1)))
    V = np.stack([vb ** p * roll for p in keep]).astype(np.float32)
    out['image'] = af._apply_strips(img, st, B, (n - B) // 2, win.astype(np.float32), V, coef)
    info.update(applied=True, powers=tuple(int(p) for p in keep), block=B,
                edge_rad=np.stack([c.reshape(len(c), -1).min(1), c.reshape(len(c), -1).max(1)], 1))
    q, hh = info['quadratic_rad'], info['height']
    notes.append(f'terrain: focused with the DEM, quadratic phase {q[0]:+.2f} to {q[1]:+.2f} rad at the band edge for '
                 f'surface heights {hh[0]:+.0f} to {hh[2]:+.0f} m about the image plane')
    return info


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
