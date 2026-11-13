"""Several collections of one scene: a common ground grid, registration, and the products of a stack.

    fastsar.process_stack(['a_CPHD.cphd', 'b_CPHD.cphd', 'c_CPHD.cphd'], 'mean_SIDD.nitf')      # speckle-reduced display
    fastsar.process_stack(['a_CPHD.cphd', 'b_CPHD.cphd'], 'coherence.tif', product='coherence')
    fastsar.process_stack(['a_CPHD.cphd', 'b_CPHD.cphd'], 'phase.tif', product='interferogram')

    grid = stack.ground_grid((46.9043, -96.8288, 247.7), extent=2000.0, spacing=0.65, azimuth=247.4)
    outs = [fastsar.form_cphd(p, grid=grid) for p in paths]             # every collection on the same pixels
    outs = stack.register(outs)                                         # one translation each against the first
    avg = stack.mean(outs, equalize='mean')                             # incoherent average, with its looks
    coh = stack.coherence(outs[0], outs[1], window=7)                   # of a pair inside the critical baseline

Grid. An Earth-fixed ground grid is a plane with two orthonormal axes, a spacing along each and a shape: pixel
(i, j) lies at origin + i spx e1 + j spy e2 in ECEF, e1 x e2 upward. ground_grid builds one that is horizontal at
its center (tangent to the ellipsoid's surface of that height) from a latitude, longitude and height, an extent, a
spacing and the azimuth of e2; as_grid takes that dict, the same things under other names, or a product of
form_cphd or process_cphd (the same pixels as that product). form_cphd(grid=) and process_cphd(grid=) form a
collection directly on the grid, in the collection's own local frame. The plane is flat: a distance d from the
center it stands d^2 / (2 R) above the surface of the center's height (0.7 m at 3 km, 8 m at 10 km, 71 m at 30
km), and ground at the center's height there appears displaced toward the radar by that height over the tangent
of the grazing angle, differently for each look direction. Over tens of kilometers a stack is therefore better
cut into grids of a few kilometers each.

Registration. Collections formed on one grid agree only as well as their orbits and timing are known, which for
the small satellites measured is 9 to 22 m (docs/stack.md). register finds one translation per image
against a reference from the cross-correlation of the detected images (the peak of the correlation of their
box-filtered, clipped decibel images, refined on the upsampled correlation), reports how significant the peak is
and what tiles of the image still disagree by afterwards, and applies it: by a phase ramp on the spectrum of the
baseband image (shift_image), or by forming the collection again on the displaced grid (form_cphd(grid=,
shift=)). Either way the registered product is recorded on the grid itself with its frame moved by the
translation, so every image of the stack has the reference's geolocation. What one translation leaves between
different look directions is height: a scatterer above the grid plane lays over toward each radar.

Products. mean is the incoherent average, with the equivalent number of looks measured on homogeneous tiles;
coherence is the sample coherence of a registered pair on the common part of the two spectra with a linear fringe
removed, beside the floor the same estimator gives for an incoherent pair at the image's actual sampling;
change gives that coherence and the ratio of the two intensities; baseline tells from the geometry alone whether
a pair has a common spectrum at all. process_stack does it all from the files, one collection in memory at a time.

Coherent pairs. Two images are compared in phase only over the part of the ground spectrum both collections
have. common_support finds it from the geometry: the look directions both apertures cover and, in each
direction, the ground-range wavenumbers both bands hold. Each collection is cut to it in its phase history, the
pulses by form_cphd(aperture=) and the frequencies of every pulse by form_cphd(band=), under one weighting of
the common part; that is how process_stack forms a pair. interferogram is the phase of such a pair, with its
linear fringe (and a surface's topography) removed and the height of ambiguity that reads it; coherent_sum adds
the two images in phase and measures what the sum gained; register(coherent=True) refines the translation on
the complex images.

Devices. Every function works where the images are (NumPy on all cores, CuPy, JAX) and returns arrays of the
same kind.
"""
import time

import numpy as np

from . import _xp, io, planes

C = 299792458.0

__all__ = ['ground_grid', 'as_grid', 'geometry', 'baseline', 'common_support', 'register', 'shift_image', 'mean',
           'coherence', 'interferogram', 'coherent_sum', 'change', 'process_stack']


# ---------------------------------------------------------------------------------------------- the grid
def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _enu(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return (np.array([-np.sin(lo), np.cos(lo), 0.0]),
            np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)]),
            np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]))


def _pair(v, name):
    a = np.ravel(np.asarray(v, np.float64))
    if a.size not in (1, 2) or not np.all(np.isfinite(a)) or a.min() <= 0:
        raise ValueError(f'{name} must be one or two positive numbers, got {v!r}')
    return float(a[0]), float(a[-1])


def ground_grid(center, extent, spacing, azimuth=0.0):
    """An Earth-fixed ground grid, horizontal at its center.

    center: (latitude, longitude) or (latitude, longitude, height): degrees, and meters above the WGS-84 ellipsoid
    (default 0). extent: the sides in meters along e1 and e2 (one number: a square). spacing: the pixel spacings
    in meters along e1 and e2 (one number: square pixels). azimuth: of e2, degrees clockwise from north; e1 = e2 x
    up, 90 degrees clockwise of it (azimuth 0: e2 north, e1 east; a radar looking along e2 has its shadows down
    the rows of the display, sidd.shadows_down).

    -> dict(origin [3] (ECEF, m: pixel (0, 0)), e1, e2 [3] (ECEF unit vectors), spx, spy (m), shape (nx, ny),
    center [3] (ECEF: pixel (nx / 2, ny / 2)), lat, lon, height, azimuth). Pixel (i, j) lies at
    origin + i spx e1 + j spy e2. The plane is tangent at the center to the surface of the center's height and
    leaves it by d^2 / (2 R) at distance d (see the module docstring)."""
    c = np.ravel(np.asarray(center, np.float64))
    if c.size not in (2, 3) or not np.all(np.isfinite(c)) or abs(c[0]) > 90:
        raise ValueError(f'center must be (latitude, longitude[, height]) in degrees and meters, got {center!r}')
    lat, lon, h = float(c[0]), float(c[1]), float(c[2]) if c.size == 3 else 0.0
    (L1, L2), (spx, spy) = _pair(extent, 'extent'), _pair(spacing, 'spacing')
    east, north, up = _enu(lat, lon)
    a = np.radians(float(azimuth))
    e2 = np.sin(a) * east + np.cos(a) * north
    e1 = np.cross(e2, up)
    nx, ny = int(np.ceil(L1 / spx)) + 1, int(np.ceil(L2 / spy)) + 1
    cen = np.ravel(io.geodetic_to_ecf(lat, lon, h))
    return dict(origin=cen - nx / 2.0 * spx * e1 - ny / 2.0 * spy * e2, e1=e1, e2=e2, spx=spx, spy=spy,
                shape=(nx, ny), center=cen, lat=lat, lon=lon, height=h, azimuth=float(azimuth) % 360.0)


def as_grid(grid):
    """A grid in ground_grid's form, from any of
    - ground_grid's dict;
    - a product of form_cphd, process_cphd or this module on the ground plane: the same pixels as that product;
    - a dict naming the same things otherwise: origin (ECEF of pixel (0, 0)), or center (ECEF) or lat, lon and
      height (of the center pixel (nx / 2, ny / 2)); e1 and e2 (ECEF, orthonormal, e1 x e2 upward) or azimuth (of
      e2, degrees clockwise from north, for a plane horizontal at the center); spx and spy or spacing; shape (nx,
      ny) or extent (m)."""
    if not isinstance(grid, dict):
        raise TypeError(f'grid must be a dict (fastsar.stack.ground_grid) or a product, got {type(grid).__name__}')
    if 'image' in grid and 'meta' in grid:                           # a product: its own pixels
        if grid.get('plane', 'ground') != 'ground':
            raise ValueError('grid: a slant-plane product is not a ground grid; give a ground-plane product')
        m = grid['meta']
        R = np.asarray(m['R'], np.float64)
        g = dict(origin=io.local_to_ecf(np.asarray(grid['origin'], np.float64), m),
                 e1=np.asarray(grid['e1'], np.float64) @ R, e2=np.asarray(grid['e2'], np.float64) @ R,
                 spx=float(grid['spx']), spy=float(grid['spy']), shape=tuple(int(v) for v in np.shape(grid['image'])[:2]))
        # a product formed on a grid records it. Those numbers are returned where they name the pixels the product
        # has now: the way through the product's local frame gives them to rounding only, and forming on them again
        # would then give the same image to float32 rounding instead of the same bits
        own = grid.get('grid')
        if isinstance(own, dict) and all(k in own for k in g):
            own = dict(origin=np.ravel(np.asarray(own['origin'], np.float64)), e1=np.ravel(np.asarray(own['e1'], np.float64)),
                       e2=np.ravel(np.asarray(own['e2'], np.float64)), spx=float(own['spx']), spy=float(own['spy']),
                       shape=tuple(int(v) for v in np.ravel(own['shape'])))
            if own['origin'].shape == own['e1'].shape == own['e2'].shape == (3,) and _same_grid(own, g, tol=1e-6):
                g = own
    else:
        g = dict(grid)
        if 'spacing' in g:
            g['spx'], g['spy'] = _pair(g.pop('spacing'), 'spacing')
        if 'spx' not in g or 'spy' not in g:
            raise ValueError('grid: give spacing, or spx and spy (m)')
        g['spx'], g['spy'] = float(g['spx']), float(g['spy'])
        cen = None
        if 'origin' not in g:
            if 'center' in g:
                cen = np.ravel(np.asarray(g['center'], np.float64))
            elif 'lat' in g and 'lon' in g:
                cen = np.ravel(io.geodetic_to_ecf(float(g['lat']), float(g['lon']), float(g.get('height', 0.0))))
            else:
                raise ValueError('grid: give origin (ECEF of pixel (0, 0)), center (ECEF), or lat and lon')
        if 'e1' not in g or 'e2' not in g:
            if 'azimuth' not in g:
                raise ValueError('grid: give e1 and e2 (ECEF), or azimuth (of e2, degrees from north)')
            p = cen if cen is not None else np.asarray(g['origin'], np.float64)
            east, north, up = _enu(*(float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(p)[:2]))
            a = np.radians(float(g['azimuth']))
            g['e2'] = np.sin(a) * east + np.cos(a) * north
            g['e1'] = np.cross(g['e2'], up)
        if 'shape' not in g:
            if 'extent' not in g:
                raise ValueError('grid: give shape (nx, ny) or extent (m)')
            L1, L2 = _pair(g['extent'], 'extent')
            g['shape'] = (int(np.ceil(L1 / g['spx'])) + 1, int(np.ceil(L2 / g['spy'])) + 1)
        if cen is not None:
            g['origin'] = cen - g['shape'][0] / 2.0 * g['spx'] * np.asarray(g['e1'], np.float64) \
                - g['shape'][1] / 2.0 * g['spy'] * np.asarray(g['e2'], np.float64)
    o, e1, e2 = (np.ravel(np.asarray(g[k], np.float64)) for k in ('origin', 'e1', 'e2'))
    if o.shape != (3,) or e1.shape != (3,) or e2.shape != (3,) or not np.all(np.isfinite(np.r_[o, e1, e2])):
        raise ValueError('grid: origin, e1 and e2 must be finite 3-vectors (ECEF)')
    if abs(e1 @ e1 - 1) > 1e-6 or abs(e2 @ e2 - 1) > 1e-6 or abs(e1 @ e2) > 1e-6:
        raise ValueError(f'grid: e1 and e2 must be orthonormal, got {e1.tolist()} and {e2.tolist()}')
    shape = tuple(int(v) for v in np.ravel(g['shape']))
    if len(shape) != 2 or min(shape) < 2 or not (g['spx'] > 0 and g['spy'] > 0):
        raise ValueError("grid: shape must be two pixel counts of at least 2 and the spacings positive, got "
                         f"{g['shape']!r}, {g['spx']!r}, {g['spy']!r}")
    cen = o + shape[0] / 2.0 * g['spx'] * e1 + shape[1] / 2.0 * g['spy'] * e2
    lat, lon, h = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(cen))
    east, north, up = _enu(lat, lon)
    if np.cross(e1, e2) @ up < 0.5:
        raise ValueError('grid: e1 x e2 must point upward (a ground plane seen from above); swap or reverse an axis')
    return dict(origin=o, e1=e1, e2=e2, spx=g['spx'], spy=g['spy'], shape=shape, center=cen, lat=lat, lon=lon,
                height=h, azimuth=float(np.degrees(np.arctan2(e2 @ east, e2 @ north)) % 360.0))


def _same_grid(a, b, tol=1e-3):
    """Whether two grids (as_grid) name the same pixels to tol of a pixel."""
    if a['shape'] != b['shape'] or abs(a['spx'] - b['spx']) > 1e-9 or abs(a['spy'] - b['spy']) > 1e-9:
        return False
    n = max(a['shape'])
    return (np.linalg.norm(a['origin'] - b['origin']) < tol * min(a['spx'], a['spy'])
            and np.linalg.norm(a['e1'] - b['e1']) * n < tol and np.linalg.norm(a['e2'] - b['e2']) * n < tol)


def _moved(meta, s):
    """meta with its local frame translated by -s (ECEF, m): the collection's positions, and with them where it
    places an image, move by -s."""
    return dict(meta, origin=np.asarray(meta['origin'], np.float64) - s, srp=np.asarray(meta['srp'], np.float64) - s)


def _enter(grid, shift, meta):
    """form_cphd's side of grid= and shift=: -> (meta, with the frame moved by -shift; the grid in that frame:
    dict(origin, e1, e2, normal, center (local), spx, spy, shape, grid (as_grid's), shift (d1, d2)))."""
    g = as_grid(grid)
    d = (0.0, 0.0) if shift is None else tuple(float(v) for v in np.ravel(shift))
    if len(d) != 2 or not np.all(np.isfinite(d)):
        raise ValueError(f'shift must be (d1, d2) in meters along the grid axes, got {shift!r}')
    if d[0] or d[1]:
        meta = _moved(meta, d[0] * g['e1'] + d[1] * g['e2'])
        meta['notes'] = list(meta.get('notes', [])) + [
            f'formed on the grid displaced by {d[0]:.3f} m along e1 and {d[1]:.3f} m along e2 and recorded on the grid: '
            "the collection's frame is moved by as much (registration)"]
    R = np.asarray(meta['R'], np.float64)
    o, e1, e2 = io.ecf_to_local(g['origin'], meta), g['e1'] @ R.T, g['e2'] @ R.T
    nx, ny = g['shape']
    return meta, dict(origin=o, e1=e1, e2=e2, normal=np.cross(e1, e2), spx=g['spx'], spy=g['spy'], shape=(nx, ny),
                      center=o + nx / 2.0 * g['spx'] * e1 + ny / 2.0 * g['spy'] * e2, grid=g, shift=d)


def _grid_of(out):
    """The grid a product's pixels lie on now (from its own origin, axes and frame, whatever it records)."""
    return as_grid(out)


# ---------------------------------------------------------------------------------------------- geometry of a pair
def geometry(source, channel=0):
    """The collection geometry a stack is planned from, without the phase history: from a CPHD file (its
    per-pulse parameters alone are read) or a product of form_cphd.

    -> dict(ant [P, 3] (antenna phase centers, ECEF: the mean of the transmit and receive positions), srp [3] (the
    mid-aperture scene reference point, ECEF), band (first, last frequency, Hz), times [P] (transmit, s from the
    start), start (the collection start as the file gives it), name, collector, path)."""
    if isinstance(source, dict):
        m = source['meta']
        ant = 0.5 * (np.asarray(m['tx'], np.float64) + np.asarray(m['rcv'], np.float64))
        return dict(ant=io.local_to_ecf(ant, m), srp=np.asarray(m['srp'], np.float64),
                    band=tuple(float(v) for v in source['band']), times=np.asarray(m['tx_time'], np.float64),
                    start=str(m.get('start')), name=m.get('core_name'), collector=m.get('collector'), path=None,
                    pulses=tuple(source['pulses']) if source.get('pulses') is not None else None)
    r = io._open_cphd(source)
    m = r.cphd_meta
    v03 = str(getattr(r, 'cphd_version', '') or '').startswith('0.3')
    vb = r.read_pvp_array(channel)
    tx, rcv, srp = (np.asarray(vb[k], np.float64) for k in ('TxPos', 'RcvPos', 'SRPPos'))
    f0, fs = (np.asarray(vb[k], np.float64) for k in (('Fx0', 'Fx_SS') if v03 else ('SC0', 'SCSS')))
    K = int((m.Data.ArraySize if v03 else m.Data.Channels)[channel].NumSamples)
    P = len(tx)
    coll = getattr(m, 'CollectionID', None) or getattr(m, 'CollectionInfo', None)
    start = io._opt(m, 'Global.Timeline.CollectionStart') or io._opt(m, 'Global.CollectStart')
    return dict(ant=0.5 * (tx + rcv), srp=srp[P // 2], band=(float(f0[P // 2]), float(f0[P // 2] + (K - 1) * fs[P // 2])),
                times=np.asarray(vb['TxTime'], np.float64), start=None if start is None else io._start(start),
                name=getattr(coll, 'CoreName', None), collector=getattr(coll, 'CollectorName', None),
                path=str(source), pulses=None)


def _look(geo, point, aperture='square'):
    """How a collection sees a ground point (ECEF): the pulses an image of square resolution uses (form_cphd's
    rule; aperture='full' or a product's own pulses otherwise), and over them the azimuth of the radar from the
    point (degrees clockwise from north) and the grazing angle (degrees) at mid-aperture, the azimuth interval,
    and the ground-plane wavenumbers' polar description: phi [n] (rad, the direction from the radar to the point
    in the horizontal plane at the point, from north toward east) and cosg [n] (the cosine of the grazing angle)
    of evenly spaced pulses."""
    lat, lon, _ = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(point))
    east, north, up = _enu(lat, lon)
    ant = geo['ant']
    P = len(ant)
    lo, hi = geo.get('pulses') or (0, P)
    f0, f1 = geo['band']
    u = point[None] - ant
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    if geo.get('pulses') is None and aperture == 'square':
        mid = P // 2
        graze = np.arcsin(np.clip(-u[mid] @ up, 0.05, 1.0))
        ang = np.arccos(np.clip(u @ u[mid], -1, 1)) * np.sign(np.arange(P) - mid)
        want = (C / (0.5 * (f0 + f1))) / (2 * (C / (2 * (f1 - f0))) / np.cos(graze))
        if np.ptp(ang) > 1.15 * want:
            keep = np.nonzero(np.abs(ang) <= want / 2)[0]
            lo, hi = int(keep[0]), int(keep[-1]) + 1
    idx = np.unique(np.linspace(lo, hi - 1, min(hi - lo, 129)).astype(np.int64))
    uu = u[idx]
    phi = np.unwrap(np.arctan2(uu @ east, uu @ north))
    cosg = np.hypot(uu @ east, uu @ north)
    m = len(idx) // 2
    return dict(pulses=(int(lo), int(hi)), azimuth=float(np.degrees(phi[m] + np.pi) % 360.0),
                graze=float(np.degrees(np.arccos(cosg[m]))), phi=phi, cosg=cosg, band=(f0, f1),
                azimuth_span=float(np.degrees(abs(phi[-1] - phi[0]))))


def _overlap(la, lb):
    """The common part of two ground-plane spectra (_look): -> dict(range, azimuth: the fractions of the first
    one's radial and angular extent that the second shares; rho (lo, hi): the common radial interval at the
    first's mid-aperture, cycles/m; phi (lo, hi): the common interval of direction, rad)."""
    dphi = (lb['phi'][len(lb['phi']) // 2] - la['phi'][len(la['phi']) // 2] + np.pi) % (2 * np.pi) - np.pi
    pb = lb['phi'] - lb['phi'][len(lb['phi']) // 2] + la['phi'][len(la['phi']) // 2] + dphi      # on a's branch
    a0, a1 = sorted((la['phi'][0], la['phi'][-1]))
    b0, b1 = sorted((pb[0], pb[-1]))
    p0, p1 = max(a0, b0), min(a1, b1)
    ca, cb = la['cosg'][len(la['cosg']) // 2], lb['cosg'][len(lb['cosg']) // 2]
    ra = (2 * la['band'][0] * ca / C, 2 * la['band'][1] * ca / C)
    rb = (2 * lb['band'][0] * cb / C, 2 * lb['band'][1] * cb / C)
    r0, r1 = max(ra[0], rb[0]), min(ra[1], rb[1])
    return dict(range=float(max(0.0, r1 - r0) / (ra[1] - ra[0])), azimuth=float(max(0.0, p1 - p0) / (a1 - a0)),
                rho=(float(r0), float(r1)), phi=(float(p0), float(p1)))


def _point(point, geo):
    """A ground point as ECEF: None (the collection's scene reference point), ECEF, or (latitude, longitude[, h])."""
    if point is None:
        return np.asarray(geo['srp'], np.float64)
    p = np.ravel(np.asarray(point, np.float64))
    return p if np.linalg.norm(p) > 1e6 else np.ravel(io.geodetic_to_ecf(p[0], p[1], p[2] if p.size == 3 else 0.0))


def _directions(geo, point):
    """Every pulse of a collection as it looks at a ground point (ECEF): phi [P] (rad, unwrapped: the direction
    from the radar to the point in the horizontal plane there, from north toward east) and cosg [P] (the cosine of
    the grazing angle). A pulse at frequency f samples the ground wavenumber of radius 2 f cosg / c in the
    direction phi."""
    lat, lon, _ = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(point))
    east, north, _ = _enu(lat, lon)
    u = point[None] - geo['ant']
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return np.unwrap(np.arctan2(u @ east, u @ north)), np.hypot(u @ east, u @ north)


def common_support(a, b, point=None):
    """The part of the ground spectrum that two collections of one scene both have, and the selection of each
    collection's phase history that covers it: what a coherent pair is formed from.

    a, b: CPHD files (their per-pulse parameters alone are read) or products of form_cphd (the whole collection
    each came from is considered, not the pulses it used). point: the ground point (ECEF, or (latitude, longitude,
    height)) the geometry is taken at, default a's scene reference point.

    A pulse that looks at the point from the direction phi samples, at frequency f, the ground wavenumber of
    radius 2 f cos(graze) / c in that direction. The common support is the interval of directions both apertures
    cover, cut to its central part where it is wider than a square resolution cell needs (the common radial band
    over its center radius), and in each direction the radii both bands reach: the grazing angle changes along a
    squinted aperture, so the radial limits follow the direction, and a difference in grazing angle or in band
    between the collections moves them. Each collection covers it with the pulses in the interval and, on every
    pulse, the frequencies between the limits over its own cos(graze).

    -> dict(inside (bool: there is a common support), pulses ((first, last + 1) of each collection), bands (for
    each, [pulses, 2]: the first and last frequency of every pulse, Hz): the aperture= and band= of form_cphd;
    phi (the interval, rad), azimuth_deg (its width), azimuth_available_deg (the whole common interval),
    azimuth_needed_deg (what square resolution takes), rho (the radial limits at the interval's center,
    cycles/m), overlap_range (for each collection, the fraction of its band inside the common one, at equal
    direction), overlap_azimuth (the fraction that the apertures form_cphd would use for each alone share),
    d_graze (degrees, b less a at equal direction), critical_graze (degrees: the difference at which the bands
    part), height_of_ambiguity (m: the height above the grid plane that turns the phase of a b* by one cycle,
    signed; None for equal grazing angles), resolution (ground range, azimuth; m, unweighted))."""
    ga, gb = (v if isinstance(v, dict) and 'ant' in v else geometry(v) for v in (a, b))
    p = _point(point, ga)
    (pa, ca), (pb, cb) = _directions(ga, p), _directions(gb, p)
    pb = pb + np.round((pa[len(pa) // 2] - pb[len(pb) // 2]) / (2 * np.pi)) * 2 * np.pi
    phi, cosg, bands = (pa, pb), (ca, cb), (ga['band'], gb['band'])
    own = [_look(dict(g, pulses=None), p)['pulses'] for g in (ga, gb)]
    oi = [np.sort(ph[[lo, hi - 1]]) for ph, (lo, hi) in zip(phi, own)]
    own_az = (max(0.0, min(oi[0][1], oi[1][1]) - max(oi[0][0], oi[1][0]))
              / min(oi[0][1] - oi[0][0], oi[1][1] - oi[1][0]))
    lo, hi = max(pa.min(), pb.min()), min(pa.max(), pb.max())
    none = dict(inside=False, pulses=None, bands=None, phi=None, azimuth_deg=0.0, overlap_range=(0.0, 0.0),
                azimuth_available_deg=float(np.degrees(max(hi - lo, 0.0))), overlap_azimuth=float(own_az),
                height_of_ambiguity=None, d_graze=None, critical_graze=None)
    if hi <= lo:
        return none
    order = [np.argsort(ph) for ph in phi]

    def limits(lo, hi):
        pg = np.linspace(lo, hi, 257)
        cw = [np.interp(pg, ph[o], c[o]) for ph, c, o in zip(phi, cosg, order)]
        rl = (2 / C) * np.maximum(bands[0][0] * cw[0], bands[1][0] * cw[1])
        rh = (2 / C) * np.minimum(bands[0][1] * cw[0], bands[1][1] * cw[1])
        return pg, cw, rl, rh
    pg, cw, rl, rh = limits(lo, hi)
    if np.median(rh - rl) <= 0:
        return none
    available = hi - lo
    need = float(np.median((rh - rl) / (0.5 * (rl + rh))))
    if available > need:
        lo, hi = 0.5 * (lo + hi) - need / 2, 0.5 * (lo + hi) + need / 2
        pg, cw, rl, rh = limits(lo, hi)
    pulses, fb, frac = [], [], []
    for ph, c, band, w in zip(phi, cosg, bands, cw):
        idx = np.nonzero((ph >= lo) & (ph <= hi))[0]
        if len(idx) < 2:
            return none
        sl = slice(int(idx[0]), int(idx[-1]) + 1)
        pulses.append((sl.start, sl.stop))
        f = np.stack([np.interp(ph[sl], pg, rl), np.interp(ph[sl], pg, rh)], 1) * (C / 2) / c[sl, None]
        fb.append(np.clip(f, band[0], band[1]))
        frac.append(float(np.mean((rh - rl) / ((2 / C) * (band[1] - band[0]) * w))))
    psi = [np.arccos(np.clip(w, -1, 1)) for w in cw]
    rho_c = float(np.mean(0.5 * (rl + rh)))
    dtan = float(np.median(np.tan(psi[0]) - np.tan(psi[1])))
    fc = 0.25 * (bands[0][0] + bands[0][1] + bands[1][0] + bands[1][1])
    bw = min(bands[0][1] - bands[0][0], bands[1][1] - bands[1][0])
    return dict(inside=True, pulses=tuple(pulses), bands=tuple(fb), phi=(float(lo), float(hi)),
                azimuth_deg=float(np.degrees(hi - lo)), azimuth_available_deg=float(np.degrees(available)),
                azimuth_needed_deg=float(np.degrees(need)), rho=(float(np.median(rl)), float(np.median(rh))),
                overlap_range=tuple(frac), overlap_azimuth=float(own_az),
                d_graze=float(np.degrees(np.median(psi[1] - psi[0]))),
                critical_graze=float(np.degrees(bw / fc / np.tan(np.median(0.5 * (psi[0] + psi[1]))))),
                height_of_ambiguity=None if abs(dtan) < 1e-12 else float(1.0 / (rho_c * dtan)),
                resolution=(float(1.0 / np.median(rh - rl)), float(1.0 / (rho_c * (hi - lo)))))


def baseline(a, b, point=None, aperture='square'):
    """Whether two collections of one scene can be coherent, and how much of each the other shares, from their
    geometry alone.

    a, b: CPHD files (only their per-pulse parameters are read) or products of form_cphd. point: the ground point
    (ECEF, or (latitude, longitude, height)) the geometry is taken at; default the first's scene reference point.
    aperture: 'square' (the part of each aperture form_cphd uses without a spacing) or 'full', for the apertures'
    own figures; a product's own pulses are used as they are.

    Two images share spectrum only where their ground wavenumbers coincide (common_support). In range these are
    2 f cos(graze) / c, so a difference in grazing angle moves one band against the other and the bands part at
    the critical difference (B / fc) / tan(graze); in azimuth the directions the apertures look from must
    overlap, so they part when the difference in azimuth reaches the aperture's own angular extent on the ground.
    The same difference in grazing angle sets the height of ambiguity, 1 / (rho (tan(graze a) - tan(graze b)))
    with rho the center radius of the common band: the height above the grid plane that turns the
    interferometric phase by a cycle. Under a meter of it, relief decorrelates the pair within a resolution cell
    of tall structure; hundreds of meters of it leave the phase to motion and propagation.

    -> dict(inside (a common support exists), overlap_range (the fraction of the narrower band that is common, at
    equal look direction), overlap_azimuth (the fraction that the apertures each image would use alone share),
    overlap (their product), azimuth_available (the common interval of the whole apertures over what square
    resolution needs: 1 or more means the pulses can be cut to full overlap), d_graze (degrees, b less a at equal
    look direction; at mid-aperture where no direction is common), d_azimuth (degrees, at mid-aperture),
    critical_graze, critical_azimuth (degrees), height_of_ambiguity (m, signed, None for equal grazing angles),
    graze, azimuth (of each, at mid-aperture), days (b's start less a's, where both state one))."""
    ga, gb = (v if isinstance(v, dict) and 'ant' in v else geometry(v) for v in (a, b))
    p = _point(point, ga)
    la, lb = _look(ga, p, aperture), _look(gb, p, aperture)
    ov = _overlap(la, lb)
    cs = common_support(ga, gb, p)
    fc = 0.25 * (la['band'][0] + la['band'][1] + lb['band'][0] + lb['band'][1])
    bw = min(la['band'][1] - la['band'][0], lb['band'][1] - lb['band'][0])
    psi = np.radians(0.5 * (la['graze'] + lb['graze']))
    days = None
    try:
        days = float((np.datetime64(str(gb['start']).rstrip('Z')) - np.datetime64(str(ga['start']).rstrip('Z')))
                     / np.timedelta64(1, 's')) / 86400.0
    except (ValueError, TypeError):
        pass
    # in range: at equal look direction where the apertures share one, else each at its own mid-aperture
    rg = min(cs['overlap_range']) if cs['azimuth_available_deg'] > 0 else ov['range']
    return dict(inside=bool(cs['inside'] and ov['range'] > 0 and ov['azimuth'] > 0), overlap_range=float(rg),
                overlap_azimuth=ov['azimuth'], overlap=float(rg * ov['azimuth']),
                azimuth_available=(float(cs['azimuth_available_deg'] / cs['azimuth_needed_deg'])
                                   if cs['inside'] else 0.0),
                d_graze=cs['d_graze'] if cs['inside'] else lb['graze'] - la['graze'],
                d_azimuth=(lb['azimuth'] - la['azimuth'] + 180.0) % 360.0 - 180.0,
                critical_graze=float(np.degrees(bw / fc / np.tan(psi))),
                critical_azimuth=min(la['azimuth_span'], lb['azimuth_span']),
                height_of_ambiguity=cs['height_of_ambiguity'], graze=(la['graze'], lb['graze']),
                azimuth=(la['azimuth'], lb['azimuth']), days=days)


# ---------------------------------------------------------------------------------------------- arrays, where they are
def _fft(a, name, **kw):
    f = getattr(_xp.fft(a), name)
    return f(a, workers=-1, **kw) if _xp.kind(a) == 'numpy' else f(a, **kw)


def _rowwise(f, *arrs, dtype=None):
    """f(*arrs), an elementwise function of arrays with equal first dimensions: at once on a device, in blocks of
    rows on all cores on the host."""
    a0 = arrs[0]
    if _xp.kind(a0) != 'numpy' or a0.shape[0] < 64:
        return f(*arrs)
    from .patches import _rows
    first = f(*(a[:1] for a in arrs))
    out = np.empty((a0.shape[0],) + first.shape[1:], dtype or first.dtype)

    def blk(sl):
        out[sl] = f(*(a[sl] for a in arrs))
    _rows(blk, a0.shape[0], int(max(1, min(256, (1 << 20) // max(a0[0].size, 1)))))
    return out


def _power(img):
    """|img|^2 as float32, where the image is."""
    xp = _xp.module(img)
    if 'complex' in str(img.dtype):
        return _rowwise(lambda z: (z.real * z.real + z.imag * z.imag).astype(xp.float32), img)
    return _rowwise(lambda z: (z * z).astype(xp.float32), img)


def _box(x, w):
    """The mean over w x w pixels about each pixel (edges repeat the nearest pixel), where x is."""
    w = int(w)
    if w <= 1:
        return x
    xp = _xp.module(x)
    lo, hi = (w - 1) // 2, w // 2

    def one(v, axis):
        pad = [(0, 0)] * v.ndim
        pad[axis] = (lo, hi)
        p = xp.pad(v, pad, mode='edge')
        n = v.shape[axis]
        acc = None
        for k in range(w):
            s = p[k:k + n] if axis == 0 else p[:, k:k + n]
            acc = s if acc is None else acc + s
        return acc / w
    if _xp.kind(x) == 'numpy' and x.shape[0] >= 256:                 # rows in blocks with a halo, on all cores
        from .patches import _rows
        y = np.empty_like(x)
        n = x.shape[0]
        step = int(max(w, min(256, (1 << 20) // max(x[0].size, 1))))

        def blk(sl):
            a, b = max(sl.start - lo, 0), min(sl.stop + hi, n)
            v = x[a:b]
            if sl.start - lo < 0 or sl.stop + hi > n:
                v = np.pad(v, [(a - (sl.start - lo), sl.stop + hi - b)] + [(0, 0)] * (x.ndim - 1), mode='edge')
            acc = v[0:sl.stop - sl.start]
            for k in range(1, w):
                acc = acc + v[k:k + sl.stop - sl.start]
            y[sl] = one(acc / w, 1)
        _rows(blk, n, step)
        return y
    return one(one(x, 0), 1)


def _scalar(v):
    return float(_xp.host(v))


def _percentile(x, q, step=4):
    """Percentiles q of x from every step-th pixel of each axis, as floats."""
    xp = _xp.module(x)
    v = xp.percentile(x[::step, ::step].ravel(), xp.asarray(q, dtype=xp.float32))
    return [float(t) for t in np.atleast_1d(_xp.host(v))]


# ---------------------------------------------------------------------------------------------- registration
def _detected(img, box=5):
    """The image a registration correlates: the intensity averaged over box x box pixels, in decibels, clipped at
    its 1st and 99.5th percentiles (float32, where the image is). Unprocessed pixels (0) take the lower clip."""
    xp = _xp.module(img)
    db = _rowwise(lambda p: 10.0 * xp.log10(xp.maximum(p, 1e-30)), _box(_power(img), box))
    lo, hi = _percentile(db[db.shape[0] // 16:-(db.shape[0] // 16) or None], [1.0, 99.5])
    return xp.clip(db, lo, max(hi, lo + 1e-6)).astype(xp.float32)


def _hann(n, xp):
    return xp.asarray(np.hanning(n).astype(np.float32))


def _peak(a, b, max_shift, around=(0, 0), narrow=None, up=16, highpass=32.0):
    """The translation d (pixels, along both axes) at which b(x + d) best matches a(x), from the peak of the
    cross-correlation of the two (mean removed, Hann weighted, structure larger than `highpass` pixels left out:
    it sharpens nothing and fills the correlation surface) within max_shift pixels of `around`.
    -> (d [2] float, z: the peak over the rms of the correlation surface, rho: the normalized correlation at the
    peak, clear: the peak is not on the border of the search and, with narrow (pixels), lies within it of
    `around`). The peak is refined on the correlation upsampled `up` times about it."""
    from scipy.fft import next_fast_len
    xp = _xp.module(a)
    n1, n2 = a.shape
    w1, w2 = _hann(n1, xp)[:, None], _hann(n2, xp)[None, :]
    aw, bw = (a - a.mean()) * w1 * w2, (b - b.mean()) * w1 * w2
    fs = (next_fast_len(n1, real=True), next_fast_len(n2, real=True))
    A, B = _fft(aw, 'rfft2', s=fs), _fft(bw, 'rfft2', s=fs)
    X = xp.conj(A) * B
    hp = None
    if highpass:
        k1, k2 = np.fft.fftfreq(fs[0])[:, None], np.fft.rfftfreq(fs[1])[None, :]
        hp = xp.asarray((1.0 - np.exp(-0.5 * (k1 * k1 + k2 * k2) * (2 * np.pi * highpass / 2.355) ** 2)).astype(np.float32))
        X = X * hp
    cc = _fft(X, 'irfft2', s=fs)
    m1, m2 = min(int(max_shift), fs[0] // 2 - 1), min(int(max_shift), fs[1] // 2 - 1)
    i1 = (int(around[0]) + np.arange(-m1, m1 + 1)) % fs[0]
    i2 = (int(around[1]) + np.arange(-m2, m2 + 1)) % fs[1]
    sub = _xp.host(xp.take(xp.take(cc, xp.asarray(i1), axis=0), xp.asarray(i2), axis=1)).astype(np.float64)
    i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak = sub[i, j]
    z = float((peak - _scalar(cc.mean())) / max(_scalar(cc.std()), 1e-30))
    if hp is None:
        rho = float(peak / max(np.sqrt(_scalar((aw * aw).sum()) * _scalar((bw * bw).sum())), 1e-30))
    else:                                                            # of the filtered images (Parseval, half plane)
        m = np.arange(fs[1] // 2 + 1)
        wk = xp.asarray(np.where((m == 0) | ((fs[1] % 2 == 0) & (m == fs[1] // 2)), 1.0, 2.0).astype(np.float32))[None, :] * hp
        ea, eb = _scalar((xp.abs(A) ** 2 * wk).sum()), _scalar((xp.abs(B) ** 2 * wk).sum())
        rho = float(peak * fs[0] * fs[1] / max(np.sqrt(ea * eb), 1e-30))
    d = np.array([i - m1 + int(around[0]), j - m2 + int(around[1])], np.float64)
    clear = 0 < i < 2 * m1 and 0 < j < 2 * m2
    if narrow is not None:
        clear = clear and abs(i - m1) <= narrow and abs(j - m2) <= narrow
    if not clear or up <= 1:
        return d, z, rho, bool(clear)
    # the correlation about the peak at 1 / up of a pixel, from its spectrum (the real transform's half plane), in
    # double precision: on a CUDA device, else on the host
    t = np.arange(-up, up + 1) / up
    k1 = np.fft.fftfreq(fs[0])
    k2 = np.fft.rfftfreq(fs[1])
    E1 = np.exp(2j * np.pi * np.outer(d[0] + t, k1))
    wgt = np.where((np.arange(len(k2)) == 0) | ((fs[1] % 2 == 0) & (np.arange(len(k2)) == len(k2) - 1)), 1.0, 2.0)
    E2 = np.exp(2j * np.pi * np.outer(k2, d[1] + t)) * wgt[:, None]
    if _xp.kind(X) == 'cupy':
        fine = _xp.host(xp.real(xp.asarray(E1) @ X.astype(xp.complex128) @ xp.asarray(E2)))
    else:
        fine = np.real(E1 @ _xp.host(X).astype(np.complex128) @ E2)
    i, j = np.unravel_index(int(np.argmax(fine)), fine.shape)
    off = np.array([t[i], t[j]])
    if 0 < i < 2 * up and 0 < j < 2 * up:                            # a parabola through the three samples about it
        di = 0.5 * (fine[i - 1, j] - fine[i + 1, j]) / (fine[i - 1, j] - 2 * fine[i, j] + fine[i + 1, j])
        dj = 0.5 * (fine[i, j - 1] - fine[i, j + 1]) / (fine[i, j - 1] - 2 * fine[i, j] + fine[i, j + 1])
        off += np.array([di, dj]) / up
    return d + off, z, rho, True


def _estimate(ref, det, spx, spy, max_shift, tiles, highpass=32.0):
    """One translation between two detected images (_detected) and what tiles of them say after it: the dict
    register reports (shifts in pixels and meters along the grid axes). A peak counts as significant at 5 times the
    rms of its correlation surface."""
    d, z, rho, clear = _peak(ref, det, max_shift, highpass=highpass)
    sp = np.array([spx, spy])
    n1, n2 = ref.shape
    nt = int(tiles)
    T1, T2 = n1 // nt, n2 // nt
    res, zs, used = [], [], []
    ar = (int(round(d[0])), int(round(d[1])))
    for ti in range(nt if min(T1, T2) >= 48 else 0):
        for tj in range(nt):
            sl = (slice(ti * T1, (ti + 1) * T1), slice(tj * T2, (tj + 1) * T2))
            dt, zt, _, ok = _peak(ref[sl], det[sl], max_shift, around=ar, narrow=8, highpass=highpass)
            res.append((dt - d) * sp)
            zs.append(zt)
            used.append(bool(ok and zt > 5.0))
    res, used = np.array(res).reshape(-1, 2), np.array(used, bool)
    rr = np.linalg.norm(res[used], axis=1) if used.any() else np.array([np.nan])
    return dict(shift_px=(float(d[0]), float(d[1])), shift=(float(d[0] * spx), float(d[1] * spy)), significance=z,
                correlation=rho, found=bool(clear and z > 5.0), tiles=nt * nt if len(res) else 0,
                tiles_used=int(used.sum()), tile_residuals=res, tile_used=used, tile_significance=np.array(zs),
                residual_rms=float(np.sqrt(np.mean(rr ** 2))), residual_median=float(np.median(rr)),
                residual_worst=float(rr.max()))


def _support(out):
    """The baseband spectrum of a ground image at its scene center on its own axes, from the collection: the
    limits (k1, k2) along e1 and e2 (cycles/m), and the directions and half widths of the parallelogram the
    pulses and the band span (for a spectrum that wraps around the sampling rate)."""
    fr = planes._frame(out)
    ant = planes._used(out)
    ant = ant[np.unique(np.linspace(0, len(ant) - 1, min(len(ant), 129)).astype(np.int64))]
    e1, e2 = np.asarray(out['e1'], np.float64), np.asarray(out['e2'], np.float64)
    u = fr['scp'] - ant
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    f = np.array([out['band'][0], 0.5 * (out['band'][0] + out['band'][1]), out['band'][1]])
    K = (2 / C) * (f[None, :, None] * u[:, None, :] - f[1] * fr['e2'])              # [p, 3, 3]
    k = np.stack([K @ e1, K @ e2], -1)                                              # [p, 3, 2]
    m = len(ant) // 2
    r, t = k[m, 2] - k[m, 0], k[-1, 1] - k[0, 1]                                    # along frequency, along pulses
    M = np.linalg.inv(np.stack([r / np.linalg.norm(r), t / np.linalg.norm(t)], 1))
    ab = k.reshape(-1, 2) @ M.T
    return dict(k1=(float(k[..., 0].min()), float(k[..., 0].max())), k2=(float(k[..., 1].min()), float(k[..., 1].max())),
                M=M, half=np.abs(ab).max(0), center=(2 / C) * f[1] * np.array([fr['e2'] @ e1, fr['e2'] @ e2]))


def _ramp(n, sp, frac, lim, xp):
    """The spectrum's factor [n] (complex64) for a shift by frac pixels (|frac| <= 1/2) along an axis of n samples
    at spacing sp: exp(2 pi j k frac sp) over the band lim = (lo, hi) (cycles/m, about zero), with the phase
    carried smoothly from one band edge to the other across the part of the sampled band the image leaves empty,
    which keeps the interpolation's reach to a few pixels over the gap's width instead of the whole line."""
    k = np.fft.fftfreq(n, sp)
    half = 0.5 / sp
    lo, hi = max(lim[0], -half), min(lim[1], half)
    ph = k * sp * frac
    gap = (1.0 / sp) - (hi - lo)
    if lim[0] >= -half and lim[1] <= half and gap > 0.02 / sp:
        # from hi up through the Nyquist frequency to lo + 1 / sp the linear phase would arrive a whole `frac` of
        # a cycle off: that step is taken smoothly over the gap (two continuous derivatives at both edges)
        t = np.where(k > hi, (k - hi) / gap, np.where(k < lo, (k + 1.0 / sp - hi) / gap, -1.0))
        s = t * t * t * (t * (6 * t - 15) + 10)
        ph = np.where(t >= 0, (hi + gap * t) * sp * frac - s * frac, ph)
    return xp.asarray(np.exp(2j * np.pi * ph).astype(np.complex64))


def _translate(img, d, lim=None, wrap=None):
    """img(x + d) for d = (d1, d2) pixels: the whole pixels by moving the array, the fraction by a phase ramp on
    the spectrum along each axis (_ramp; lim: the band limits along the two axes, cycles per pixel), or, for a
    spectrum that wraps around the sampling rate (wrap: (spx, spy, _support's dict)), by a ramp over the plane with
    every sample at its own alias. Pixels whose source is outside the image, or exactly 0, are 0."""
    from scipy.fft import next_fast_len
    xp = _xp.module(img)
    n1, n2 = img.shape
    whole = np.rint(np.asarray(d, np.float64)).astype(int)
    frac = np.asarray(d, np.float64) - whole
    cplx = 'complex' in str(img.dtype)
    zero = img == 0
    has_zero = bool(_xp.host(zero.any()))
    out = img
    if np.abs(frac).max() > 1e-6:
        fs = (next_fast_len(n1 + 16), next_fast_len(n2 + 16))        # zeros beyond the edges, not the far side
        F = _fft(img if cplx else img.astype(xp.complex64), 'fft2', s=fs)
        if wrap is None:
            lim = lim or ((-0.5, 0.5), (-0.5, 0.5))
            F = _rowwise(lambda f, r: f * r, F, _ramp(fs[0], 1.0, frac[0], lim[0], xp)[:, None]
                         * _ramp(fs[1], 1.0, frac[1], lim[1], xp)[None, :])
        else:
            F = F * _alias_ramp(fs, frac, wrap, xp)
        out = _fft(F, 'ifft2')[:n1, :n2]
        del F
        if not cplx:
            out = xp.maximum(out.real, 0).astype(img.dtype)
    # the whole pixels: result[i] = out[i + whole]
    def moved(a, fill):
        sl, pad = [], []
        for n, w in zip(a.shape, whole):
            lo, hi = max(w, 0), min(n + w, n)
            if hi <= lo:
                return xp.full(a.shape, fill, a.dtype)
            sl.append(slice(lo, hi))
            pad.append((lo - w, n - (hi - w)))
        return xp.pad(a[tuple(sl)], pad, mode='constant', constant_values=fill) if any(whole) else a
    out = moved(out, 0)
    if has_zero:
        out = xp.where(moved(zero, True), xp.zeros((), out.dtype), out)
    return out


def _wraps(sup, spx, spy):
    """Whether a baseband spectrum (_support) reaches past half the sampling rate along an axis."""
    return (sup['k1'][0] < -0.5 / spx or sup['k1'][1] > 0.5 / spx or sup['k2'][0] < -0.5 / spy or sup['k2'][1] > 0.5 / spy)


def _bins(k1, k2, spx, spy, sup):
    """The baseband spatial frequency (cycles/m along e1 and e2) that samples of an image's transform hold, for
    their own frequencies k1 [rows, 1] and k2 [1, n2] (arrays of any kind): -> (k1, k2), broadcastable to
    [rows, n2]. The frequencies themselves where the spectrum lies within half the sampling rate; for a spectrum
    that wraps around it (sup: _support's, the parallelogram the pulses and the band span), the alias that lies in
    the support, found among the nine nearest to zero by its distance in the support's own oblique coordinates."""
    if not _wraps(sup, spx, spy):
        return k1, k2
    xp = _xp.module(k1)
    M, half = sup['M'], sup['half']
    best = ba = bb = None
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            p, q = k1 + m / spx, k2 + n / spy
            r = xp.maximum(xp.abs(M[0, 0] * p + M[0, 1] * q) / half[0], xp.abs(M[1, 0] * p + M[1, 1] * q) / half[1])
            if best is None:
                best, ba, bb = r, p + 0 * q, q + 0 * p
            else:
                t = r < best
                best, ba, bb = xp.where(t, r, best), xp.where(t, p, ba), xp.where(t, q, bb)
    return ba, bb


def _alias_ramp(fs, frac, wrap, xp):
    """The factor [n1, n2] (complex64) of a shift by frac pixels for a spectrum that wraps around the sampling
    rate (wrap: (spx, spy, _support's dict)): each sample of the transform takes the phase of its frequency's alias
    in the support (_bins). Where the image is: on its device, or on all cores."""
    spx, spy, sup = wrap
    k2 = xp.asarray(np.fft.fftfreq(fs[1], spy))[None, :]

    def rows(k1):
        a, b = _bins(k1, k2, spx, spy, sup)
        ph = (2 * np.pi) * (a * (spx * frac[0]) + b * (spy * frac[1]))
        return (xp.cos(ph) + 1j * xp.sin(ph)).astype(xp.complex64)
    return _rowwise(rows, xp.asarray(np.fft.fftfreq(fs[0], spx))[:, None])


def shift_image(out, shift, *, inplace=False):
    """A ground-plane product translated on its grid: pixel (i, j) of the result shows what the product has
    at (i, j) displaced by shift = (d1, d2) meters along e1 and e2, as if the collection had been formed on the
    grid displaced by it (form_cphd(grid=, shift=), which this agrees with to the formers' accuracy away from the
    edges).

    A complex image is translated at baseband, where its spectrum is centered at zero whatever the squint and
    wherever the pixel (planes.deramp, exact at every pixel; an image on its carrier is brought there and back
    onto the carrier of the displaced grid): the whole pixels by moving the array, the fraction of a pixel by a
    phase ramp on the spectrum along each axis, which is exact for the band of the image. The sampling leaves part
    of the band empty, across which the ramp returns smoothly, so a pixel depends on its neighbors within some
    tens of pixels only. Pixels shifted in from outside the image and pixels that were exactly 0 (unprocessed)
    are 0; within that reach of them, and of the image's edges, the result is approximate. A real (detected)
    image is translated likewise as it is; its band is twice the complex image's, so unless it is sampled for
    that the fraction of a pixel is approximate.

    The result is recorded on the same grid: its frame (meta['origin']) is moved by the translation, so that its
    pixels keep their ECEF positions and the collection's own geometry of them stays what formed them. shift
    (the product's entry) accumulates. inplace: the input's image may be overwritten.

    -> the product, with image, origin, meta, shift, notes."""
    d1, d2 = (float(v) for v in shift)
    res = dict(out)
    img = out['image']
    xp = _xp.module(img)
    cplx = 'complex' in str(img.dtype)
    if out.get('plane', 'ground') != 'ground':
        raise ValueError('shift_image takes a ground-plane product')
    e1, e2 = np.asarray(out['e1'], np.float64), np.asarray(out['e2'], np.float64)
    spx, spy = float(out['spx']), float(out['spy'])
    lim = wrap = None
    based = bool(out.get('baseband'))
    if cplx:
        if not based:                                                # to baseband, on a copy unless given up
            work = dict(out, image=img if inplace or _xp.kind(img) == 'jax' else img.copy())
            planes.deramp(work)
            img = work['image']
        sup = _support(dict(out, image=img))
        lim = (tuple(v * spx for v in sup['k1']), tuple(v * spy for v in sup['k2']))
        if _wraps(sup, spx, spy):
            wrap = (spx, spy, sup)
    img = _translate(img, (d1 / spx, d2 / spy), lim, wrap)
    s_l = d1 * e1 + d2 * e2
    meta = _moved(out['meta'], s_l @ np.asarray(out['meta']['R'], np.float64))
    old = out.get('shift') or (0.0, 0.0)
    res.update(image=img, origin=np.asarray(out['origin'], np.float64) + s_l, meta=meta,
               shift=(old[0] + d1, old[1] + d2), notes=list(out.get('notes', [])) + [
                   f'translated by {d1:.3f} m along e1 and {d2:.3f} m along e2 on its grid (registration); '
                   "the collection's frame is moved by as much"])
    if out.get('scp') is not None:
        res['scp'] = np.asarray(out['scp'], np.float64) + s_l
    if out.get('center') is not None:
        res['center'] = np.asarray(out['center'], np.float64) + s_l
    if cplx:
        # the scene center's range changed with the displaced grid: the baseband image's phase reference
        k = 4 * np.pi * 0.5 * (out['band'][0] + out['band'][1]) / C
        c0 = np.exp(1j * k * (planes._frame(res)['r0'] - planes._frame(dict(out, image=img))['r0'])).astype(np.complex64)
        res['image'] = _rowwise(lambda z: z * c0, img)
        res['baseband'] = True
        if not based:
            planes.deramp(res, sign=+1)
    return res


def _baseband_part(out, sl):
    """A part of a product's complex image (sl: two slices) at baseband, as the whole image there would have it;
    a new array unless the product is at baseband already."""
    t = out['image'][sl]
    if out.get('baseband'):
        return t
    o = (np.asarray(out['origin'], np.float64) + sl[0].start * float(out['spx']) * np.asarray(out['e1'], np.float64)
         + sl[1].start * float(out['spy']) * np.asarray(out['e2'], np.float64))
    w = dict(out, image=t if _xp.kind(t) == 'jax' else t.copy(), origin=o, scp=planes._scp(out))
    planes.deramp(w)
    return w['image']


def _coherent_offset(a, b, around=(0.0, 0.0), tile=256, count=8):
    """The translation d (pixels) at which b(x + d) is most coherent with a(x), sought within half a pixel of
    `around` (the translation the detected images gave).

    On up to count x count tiles of tile x tile pixels spread over the image, both at baseband, the tile's linear
    fringe removed where it has one: the magnitude of the complex correlation sum(conj(a) b(x + d)), evaluated
    from the two spectra on a lattice of quarter pixels about `around` and refined by a parabola. The tiles whose
    peak lies inside the lattice and whose normalized correlation is at least 0.15 and half of the tiles' 90th
    percentile count; d is their mean weighted by q^2 / (1 - q^2), q that correlation, without those further from
    the median than three times the median distance (and a tenth of a pixel).

    -> dict(found (four tiles or more counted), shift_px (d), delta_px (d less `around`), tiles, tiles_used, quality
    (the median q of the tiles used), residual_px (the median distance of their own peaks from d), gain (the tiles'
    summed correlation at d over that at `around`: what the refinement added to the coherence))."""
    xp = _xp.module(a['image'])
    n1, n2 = a['image'].shape
    around = np.asarray(around, np.float64)
    whole = np.rint(around).astype(int)
    frac0 = around - whole
    m = int(np.abs(whole).max()) + 8
    T = int(min(tile, (min(n1, n2) - 2 * m) // 2))
    none = dict(found=False, shift_px=(float(around[0]), float(around[1])), delta_px=(0.0, 0.0), tiles=0, tiles_used=0,
                quality=None, residual_px=None, gain=1.0)
    if T < 48:
        return none
    spx, spy = float(b['spx']), float(b['spy'])
    sup = _support(b)
    lim = (tuple(v * spx for v in sup['k1']), tuple(v * spy for v in sup['k2']))
    wrap = (spx, spy, sup) if _wraps(sup, spx, spy) else None

    def ramp(d):
        if wrap is not None:
            return _alias_ramp((T, T), d, wrap, xp)
        return _ramp(T, 1.0, d[0], lim[0], xp)[:, None] * _ramp(T, 1.0, d[1], lim[1], xp)[None, :]
    q4 = 0.25 * np.arange(-2, 3)
    R = xp.stack([ramp((frac0[0] + p, frac0[1] + q)) for p in q4 for q in q4])
    win = _hann(T, xp)[:, None] * _hann(T, xp)[None, :]
    r1, r2 = np.arange(T)[:, None], np.arange(T)[None, :]
    rows, spectra = [], []
    for i in np.unique(np.linspace(m, n1 - m - T, min(int(count), (n1 - 2 * m) // T)).astype(int)):
        for j in np.unique(np.linspace(m, n2 - m - T, min(int(count), (n2 - 2 * m) // T)).astype(int)):
            at = _baseband_part(a, (slice(i, i + T), slice(j, j + T)))
            bt = _baseband_part(b, (slice(i + whole[0], i + whole[0] + T), slice(j + whole[1], j + whole[1] + T)))
            if not bool(_xp.host((at != 0).all() & (bt != 0).all())):
                continue
            f, z, thr = _fringe(at * xp.conj(bt))
            if z > thr:
                bt = bt * xp.asarray(np.exp(2j * np.pi * (f[0] * r1 + f[1] * r2)).astype(np.complex64))
            at, bt = at * win, bt * win
            X = xp.conj(_fft(at, 'fft2')) * _fft(bt, 'fft2')
            v = np.abs(_xp.host((X[None] * R).sum((1, 2)))).astype(np.float64).reshape(5, 5)
            nrm = T * T * np.sqrt(_scalar((xp.abs(at) ** 2).sum()) * _scalar((xp.abs(bt) ** 2).sum()))
            p, q = np.unravel_index(int(np.argmax(v)), v.shape)
            clear = 0 < p < 4 and 0 < q < 4
            d = np.array([q4[p], q4[q]])
            if clear:
                d[0] += 0.25 * 0.5 * (v[p - 1, q] - v[p + 1, q]) / (v[p - 1, q] - 2 * v[p, q] + v[p + 1, q])
                d[1] += 0.25 * 0.5 * (v[p, q - 1] - v[p, q + 1]) / (v[p, q - 1] - 2 * v[p, q] + v[p, q + 1])
            rows.append((frac0[0] + d[0], frac0[1] + d[1], v[p, q] / max(nrm, 1e-30), v[2, 2], float(clear)))
            spectra.append(X)
    if not rows:
        return none
    t = np.array(rows)
    none['tiles'] = len(t)
    good = (t[:, 4] > 0) & (t[:, 2] > max(0.15, 0.5 * np.percentile(t[:, 2], 90)))
    if good.sum() >= 4:
        med = np.median(t[good, :2], 0)
        dist = np.hypot(*(t[:, :2] - med).T)
        good &= dist <= max(3 * np.median(dist[good]), 0.1)
    if good.sum() < 4:
        return none
    wq = t[good, 2] ** 2 / np.maximum(1 - t[good, 2] ** 2, 1e-3)
    d = (t[good, :2] * wq[:, None]).sum(0) / wq.sum()
    Rd = ramp((d[0], d[1]))
    at_d = sum(abs(complex(_xp.host((spectra[k] * Rd).sum()))) for k in np.nonzero(good)[0])
    return dict(found=True, shift_px=(float(whole[0] + d[0]), float(whole[1] + d[1])),
                delta_px=(float(d[0] - frac0[0]), float(d[1] - frac0[1])), tiles=len(t), tiles_used=int(good.sum()),
                quality=float(np.median(t[good, 2])),
                residual_px=float(np.median(np.hypot(*(t[good, :2] - d).T))),
                gain=float(at_d / max(t[good, 3].sum(), 1e-30)))


def register(products, reference=0, *, apply='fourier', max_shift=64, tiles=4, box=5, highpass=32.0, coherent=False,
             form=None):
    """Register images formed on one grid: one translation each against a reference.

    products: a list of ground-plane products on the same grid (form_cphd(grid=)); complex or detected images, on
    the host or a device. reference: its index. apply: 'fourier' (shift_image: a phase ramp on the spectrum),
    'reform' (the collection formed again on the displaced grid: form(k, shift) -> product, default
    fastsar.form_cphd(product['source'], grid=, shift=) with the product's own options where it records them), or
    None (estimate only). max_shift: the largest translation searched, pixels. tiles: the image is cut into tiles x
    tiles parts (of 48 pixels at least), each registered again after the translation. box: the side of the averaging
    before the correlation, pixels. highpass: structure of the detected images larger than this many pixels is left
    out of the correlation (None: kept). coherent: refine each translation on the complex images, to the one at
    which the pair is most coherent (for the products of a coherent pair: interferogram, coherent_sum, coherence).

    Each image gets the translation (d1, d2) at which it best matches the reference, image(x + d) ~ reference(x),
    from the peak of the cross-correlation of the detected images, refined to a few hundredths of a pixel. Two
    figures say how far to trust it. significance is the peak over the rms of the correlation surface (70 to 130 on
    six spotlights of 9 megapixels; correlation is the normalized value of the peak), and found is False when it is
    under 5 or the peak lies on the border of the search (the translation is then not applied). residual_median
    and residual_rms (m; residual_worst, tile_residuals [tiles^2, 2] along e1 and e2) are what the tiles still
    disagree by after the translation, over the tiles whose own peak is significant and lies within 8 pixels of
    the image's (tiles_used of tiles): a tenth of a pixel or two when one translation registers the pair; a pixel
    or more when it does not, which between different look directions is the layover of whatever stands above the
    grid plane and no warp of the ground removes (single tiles can set the rms: read it beside the median).

    coherent. The detected images register a pair to a few hundredths of a pixel, which already costs a coherent
    pair little; the complex images say where the rest is. Within half a pixel of the detected translation, the
    magnitude of the complex correlation of tiles of the two images is maximized (their linear fringe removed
    first), and the mean of the coherent tiles' peaks replaces the translation. The registration's coherent entry
    reports it: delta_px (what it changed), tiles_used, quality (the tiles' normalized correlation), residual_px
    (how far single tiles' peaks lie from it) and gain, the tiles' correlation at the refined translation over
    that at the detected one (1.002 and 1.008 on the pairs measured: there is little left to gain, and shifts of
    single tiles add 0.2 to 0.5%, so one translation is kept). It needs four coherent tiles; a pair without
    them keeps the detected translation.

    -> the list of products, registered (apply) or as given, each with registration (that dict; the reference:
    shift (0, 0)) and, when applied, shift. The stack then has the reference's geolocation."""
    if apply not in ('fourier', 'reform', None, False):
        raise ValueError(f"apply must be 'fourier', 'reform' or None, got {apply!r}")
    products = list(products)
    if not 0 <= reference < len(products):
        raise ValueError(f'reference {reference} is not one of the {len(products)} products')
    g0 = _grid_of(products[reference])
    for k, p in enumerate(products):
        if not _same_grid(g0, _grid_of(p)):
            raise ValueError(f'product {k} is not on the grid of the reference: form them with form_cphd(grid=)')
    ref = _detected(products[reference]['image'], box)
    out = []
    for k, p in enumerate(products):
        if k == reference:
            q = dict(p, registration=dict(shift=(0.0, 0.0), shift_px=(0.0, 0.0), reference=True))
            q.setdefault('shift', (0.0, 0.0))
            out.append(q)
            continue
        est = _estimate(ref, _xp.module(ref).asarray(_detected(p['image'], box)), float(p['spx']), float(p['spy']),
                        max_shift, tiles, highpass)
        est['reference'] = False
        if coherent and est['found']:
            _refine(est, products[reference], p)
        q = dict(p)
        if apply and est['found']:
            if apply == 'fourier':
                q = shift_image(p, est['shift'])
            else:
                old = p.get('shift') or (0.0, 0.0)
                tot = (old[0] + est['shift'][0], old[1] + est['shift'][1])
                q = form(k, tot) if form is not None else _reform(p, g0, tot)
        elif apply:
            q['notes'] = list(p.get('notes', [])) + [
                f"registration: no significant correlation peak (significance {est['significance']:.1f}): not translated"]
            q.setdefault('shift', (0.0, 0.0))
        q['registration'] = est
        out.append(q)
    return out


def _refine(est, ref, p):
    """A registration (_estimate's dict) refined on the complex images, in place: coherent (_coherent_offset's
    dict) and, where it found one, the translation replaced (detected, detected_px keep the first)."""
    if any('complex' not in str(q['image'].dtype) for q in (ref, p)):
        return est
    co = _coherent_offset(ref, p, est['shift_px'])
    est['coherent'] = co
    if co['found']:
        est.update(detected=est['shift'], detected_px=est['shift_px'], shift_px=co['shift_px'],
                   shift=(co['shift_px'][0] * float(p['spx']), co['shift_px'][1] * float(p['spy'])))
    return est


def _reform(p, grid, shift):
    from .cphd import form_cphd
    src = p.get('source')
    if src is None:
        raise ValueError("apply='reform' forms the collection again: the product records no source file; give form=")
    return form_cphd(src, grid=grid, shift=shift, **dict(p.get('form_options') or {}))


# ---------------------------------------------------------------------------------------------- the average
def _tiles(x, T):
    n1, n2 = x.shape[0] // T, x.shape[1] // T
    return _xp.module(x).swapaxes(x[:n1 * T, :n2 * T].reshape(n1, T, n2, T), 1, 2).reshape(n1, n2, T * T)


def _tile_stats(x, T):
    t = _tiles(x, T)
    return t.mean(-1), t.var(-1)


def _level(p, how):
    """The brightness of an intensity image over its processed pixels: the mean of the pixels under their 99th
    percentile ('mean': bright points do not set it) or the median."""
    xp = _xp.module(p)
    v = p[::2, ::2].ravel()
    v = v[_xp.host(v > 0)] if _xp.kind(p) == 'jax' else v[v > 0]
    if v.size == 0:
        return 0.0
    if how == 'median':
        return _scalar(xp.median(v))
    t = xp.percentile(v, 99.0)
    return _scalar(xp.where(v <= t, v, 0).sum() / xp.maximum((v <= t).sum(), 1))


class _Mean:
    """The running incoherent average of images on one grid, with what the looks are measured from."""

    def __init__(self, equalize, tile):
        self.equalize, self.T = equalize, int(tile)
        self.sum = self.wsum = self.first = None
        self.n, self.level0 = 0, None
        self.gains, self.single, self.running, self.inputs, self.weights = [], [], [], [], []

    def add(self, out, weight=1.0):
        p = _power(out['image'])
        xp = _xp.module(p)
        g = 1.0
        if self.equalize:
            lv = _level(p, self.equalize)
            if self.level0 is None:
                self.level0 = lv
            g = self.level0 / lv if lv > 0 else 1.0
        w = float(weight) * g
        valid = p > 0
        if self.sum is None:
            self.first = out
            self.sum = p * xp.float32(w)
            self.wsum = valid.astype(xp.float32) * xp.float32(weight)
        else:
            if p.shape != self.sum.shape:
                raise ValueError(f'image {self.n} is {p.shape}, the first {self.sum.shape}: the images of a mean '
                                 'share a grid')
            self.sum = self.sum + p * xp.float32(w)
            self.wsum = self.wsum + valid.astype(xp.float32) * xp.float32(weight)
        self.n += 1
        self.gains.append(g)
        self.weights.append(float(weight))
        if min(p.shape) >= 2 * self.T:
            self.single.append(tuple(_xp.host(v) for v in _tile_stats(p * xp.float32(g), self.T)))
            self.running.append(tuple(_xp.host(v) for v in _tile_stats(self.image(), self.T)))
        m = out.get('meta') or {}
        self.inputs.append(dict(name=m.get('core_name'), collector=m.get('collector'), start=str(m.get('start')),
                                shift=tuple(out.get('shift') or (0.0, 0.0)), gain=g, weight=float(weight),
                                graze=out.get('graze'), pulses=out.get('pulses'), band=out.get('band')))

    def image(self):
        xp = _xp.module(self.sum)
        return xp.where(self.wsum > 0, self.sum / xp.maximum(self.wsum, 1e-30), 0).astype(xp.float32)

    def looks(self, M):
        """The equivalent number of looks on homogeneous tiles (see mean)."""
        if not self.single:
            return dict(looks=None, tiles_selected=0)
        xp = _xp.module(M)
        T = self.T
        tm = _tiles(M, T)
        mu = tm.mean(-1)
        whole = _xp.host(_tiles(self.wsum, T).min(-1) >= sum(self.weights) * (1 - 1e-6))
        ts = _tiles(_box(M, 9), T)
        cv = _xp.host(ts.std(-1) / xp.maximum(ts.mean(-1), 1e-30))
        ok = _xp.host(tm.max(-1) < 6 * mu) & whole & (_xp.host(mu) > 0)
        if ok.sum() < 8:
            return dict(looks=None, tiles_selected=0)
        hom = ok & (cv <= np.percentile(cv[ok], 15))

        def enl(s):
            return float(np.median(s[0][hom].astype(np.float64) ** 2 / np.maximum(s[1][hom].astype(np.float64), 1e-300)))
        single = [enl(s) for s in self.single]
        by_n = [enl(s) for s in self.running]
        # what independent speckle would give with these tile means: (sum of the means)^2 / (sum of the variances)
        mus = np.stack([s[0][hom] for s in self.single]).astype(np.float64)
        var = np.stack([s[1][hom] for s in self.single]).astype(np.float64)
        w = np.asarray(self.weights)[:, None]
        pred = float(np.median((w * mus).sum(0) ** 2 / np.maximum((w * w * var).sum(0), 1e-300)))
        spread = float(np.median(10 * np.log10(mus.max(0) / np.maximum(mus.min(0), 1e-300))))
        return dict(looks=by_n[-1], looks_single=single, looks_by_n=by_n, looks_ideal=self.n * float(np.mean(single)),
                    looks_independent=pred, brightness_spread_db=spread, tiles_selected=int(hom.sum()),
                    tiles=int(hom.size), tile=T, homogeneous=hom)


def mean(products, *, equalize=None, weights=None, tile=32):
    """The incoherent average of registered images of one scene: the mean of their intensities at full resolution.

    products: ground-plane products on the same grid (register's), complex or detected, as a list or any iterable
    (a generator forms and gives up one image at a time); on the host or a device. weights: one number per image
    (default equal). equalize: None, or 'mean' or 'median': each image's intensity is first scaled by one factor
    that makes its brightness the first image's (the mean of the pixels under their 99th percentile, or the
    median, over the processed pixels). Speckle falls fastest when the images are equally bright: looks of
    intensities with means m_k add to (sum m_k)^2 / sum m_k^2 of one image's, N only when the m_k are equal.
    Radiometry: without equalize the average is the plain mean of the images in their own units (a CPHD carries no
    calibration; products.py); with it the average is in the first image's units and a difference in brightness
    between dates, whether of the radar's gain or of the ground (moisture, crops, snow), is removed with the gain.
    A pixel that is 0 in an image (unprocessed, or shifted in from outside) is left out of that pixel's average.

    Looks. The equivalent number of looks, mean^2 / variance of the intensity, is measured where the scene is
    homogeneous: on tiles of tile x tile pixels, without the tiles whose brightest pixel exceeds 6 times their
    mean, the 15% whose average smoothed over 9 x 9 pixels varies least; the median over those tiles.

    -> a product dict with the first image's geometry (origin, e1, e2, spx, spy, meta, grid, ...): image [nx, ny]
    float32, the amplitude of the average (its square is the mean intensity; what sidd.write_sidd takes),
    product 'mean', count (N), looks (of the average; None for an image too small to measure), looks_single [N]
    (of each image, about 1 for full-resolution images), looks_by_n [N] (of the average of the first n), looks_ideal
    (N times the mean of looks_single), looks_independent (what uncorrelated speckle gives with the tiles' own
    brightness in each image), brightness_spread_db (the median over the tiles of the brightest over the faintest
    image), gains [N] (the equalizing power factors), inputs [N] (name, collector, start, shift, gain, weight of
    each), tiles_selected, homogeneous (the tiles' mask), timing."""
    if equalize not in (None, False, 'mean', 'median'):
        raise ValueError(f"equalize must be None, 'mean' or 'median', got {equalize!r}")
    t0 = time.perf_counter()
    acc = _Mean(equalize or None, tile)
    g0 = None
    for k, p in enumerate(products):
        if p.get('plane', 'ground') != 'ground':
            raise ValueError('mean takes ground-plane products')
        g = _grid_of(p)
        if g0 is None:
            g0 = g
        elif not _same_grid(g0, g):
            raise ValueError(f'product {k} is not on the grid of the first: form them with form_cphd(grid=)')
        w = 1.0 if weights is None else float(weights[k])
        if not w > 0:
            raise ValueError(f'weights must be positive, got {w!r}')
        acc.add(p, w)
    if acc.n == 0:
        raise ValueError('mean needs at least one product')
    return _mean_product(acc, t0)


def _mean_product(acc, t0):
    xp = _xp.module(acc.sum)
    M = acc.image()
    first = acc.first
    res = {k: v for k, v in first.items() if k not in ('image', 'phase_error', 'registration', 'baseband', 'valid')}
    res.update(acc.looks(M))
    how = {None: 'as they are', 'mean': 'each scaled to the brightness of the first (mean under the 99th percentile)',
           'median': 'each scaled to the brightness of the first (median)'}[acc.equalize]
    res.update(image=xp.sqrt(M), product='mean', count=acc.n, gains=list(acc.gains), inputs=acc.inputs,
               equalize=acc.equalize, grid=_grid_of(first),
               notes=list(first.get('notes', [])) + [
                   f'incoherent average of {acc.n} images, {how}; geometry and geolocation of the first'])
    dt = time.perf_counter() - t0
    res['timing'] = dict(mean=dt, megapixels_per_second=acc.n * M.size / 1e6 / max(dt, 1e-9))
    return res


# ---------------------------------------------------------------------------------------------- coherence
def _carrier(out, on=True, copy=True):
    """The product's complex image on its carrier (on) or at baseband, a new array unless copy is False."""
    if bool(out.get('baseband')) != on:
        return out['image']
    w = dict(out, image=out['image'].copy() if copy and _xp.kind(out['image']) != 'jax' else out['image'])
    planes.deramp(w, sign=+1 if on else -1)
    return w['image']


def _band_mask(out, other, shape, xp=np):
    """The part of a baseband image's spectrum that another collection shares (bool [n1, n2] over the transform's
    samples, an array of xp): the ground wavenumber of each sample, its baseband frequency (_bins) plus the
    carrier's at the scene center, must lie in the other's ground spectrum, the sector its pulses' directions and
    its band span. -> (mask, the fractions (range, azimuth) of this image's support that are common)."""
    sup = _support(out)
    spx, spy = float(out['spx']), float(out['spy'])
    scp_ecf = io.local_to_ecf(planes._scp(out), out['meta'])
    la, lb = _look(geometry(out), scp_ecf), _look(geometry(other), scp_ecf)
    ov = _overlap(la, lb)
    # this image's axes in the horizontal plane at the scene center: the east and north components of e1 and e2
    lat, lon, _ = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(scp_ecf))
    east, north, _ = _enu(lat, lon)
    R = np.asarray(out['meta']['R'], np.float64)
    a1, a2 = np.asarray(out['e1'], np.float64) @ R, np.asarray(out['e2'], np.float64) @ R
    (e11, e12), (e21, e22) = (float(a1 @ east), float(a1 @ north)), (float(a2 @ east), float(a2 @ north))
    c1, c2 = (float(v) for v in sup['center'])
    order = np.argsort(lb['phi'])
    pb, cb = xp.asarray(lb['phi'][order]), xp.asarray(lb['cosg'][order])
    p0, p1, pmid = float(lb['phi'][order][0]), float(lb['phi'][order][-1]), float(lb['phi'][len(order) // 2])
    f0, f1 = (2 * f / C for f in lb['band'])
    k2 = xp.asarray(np.fft.fftfreq(shape[1], spy))[None, :]

    def rows(k1):
        a, b = _bins(k1, k2, spx, spy, sup)
        a, b = a + c1, b + c2
        ke, kn = a * e11 + b * e21, a * e12 + b * e22
        phi = (xp.arctan2(ke, kn) - pmid + np.pi) % (2 * np.pi) - np.pi + pmid
        rho = xp.hypot(ke, kn)
        cg = xp.interp(phi, pb, cb)
        return (phi >= p0) & (phi <= p1) & (rho >= f0 * cg) & (rho <= f1 * cg)
    return _rowwise(rows, xp.asarray(np.fft.fftfreq(shape[0], spx))[:, None]), (ov['range'], ov['azimuth'])


def _sel(v, mask):
    """The elements of v where mask (arrays of one kind)."""
    return v[_xp.host(mask)] if _xp.kind(v) == 'jax' else v[mask]


def _fringe(x, step=8):
    """The linear fringe of an interferogram x [n1, n2] (complex), and whether there is one to find.

    -> (f [2]: cycles per pixel along the two axes, anywhere within half a cycle per pixel; z: the power of the
    spectrum's peak over the level of the spectrum about it; threshold: what z must exceed for the peak to be a
    fringe). A spectrum of N samples without a fringe has its highest sample near log(N) times its level, and
    the threshold log(N) + 10 is one that such a spectrum seldom reaches (none of sixty simulated tiles of each
    of 64, 128, 256 and 384 pixels did); a fringe of coherence g over N samples gives of the order of N g^2 (a
    common part of coherence 0.1 was found on most tiles of 64 pixels, one of 0.05 on all of 256). An image of up to 512 pixels a side is transformed whole and the peak refined on
    the transform itself to a few hundredths of its sample. A larger one takes the fringe of up to sixteen parts
    of 512 pixels (the median of those that have one) for a first value, which keeps a steep fringe from
    aliasing, and then the peak of the spectrum of the whole, with that removed, averaged over step x step
    pixels."""
    xp = _xp.module(x)
    f0, D = np.zeros(2), 1
    if max(x.shape) > 512:
        D = int(step)
        T = min(512, min(x.shape))
        own = []
        for i in np.unique(np.linspace(0, x.shape[0] - T, min(4, x.shape[0] // T)).astype(int)):
            for j in np.unique(np.linspace(0, x.shape[1] - T, min(4, x.shape[1] // T)).astype(int)):
                ft, zt, tt = _fringe(x[i:i + T, j:j + T])
                if zt > tt:
                    own.append(ft)
        if own:
            f0 = np.median(np.array(own), 0)
            x = _unfringe(x, f0)
    n1, n2 = x.shape[0] // D, x.shape[1] // D
    xm = x[:n1 * D, :n2 * D].reshape(n1, D, n2, D).mean((1, 3)) if D > 1 else x
    xm = xm * xp.float32(1.0 / max(_scalar(xp.abs(xm).max()), 1e-30))     # the squares below stay in float32's range
    m = tuple(max(512, 1 << int(np.ceil(np.log2(v)) + (D > 1))) for v in (n1, n2))
    F = xp.abs(_fft(xm, 'fft2', s=m)) ** 2
    i, j = divmod(int(_xp.host(xp.argmax(F))), m[1])
    # the level the peak stands on: the spectrum of a product of two band-limited images is not flat, so the
    # peak is measured against the spectrum about it, the median (log 2 of an exponential's mean; a fringe
    # that a slowly varying phase spreads over a few samples does not move it) over the samples within an eighth
    # of the unpadded transform's width, at most 32 of its spacings
    hw = [int(max(2, min(32, nn // 8)) * mm / nn) for mm, nn in zip(m, (n1, n2))]
    st = [max(1, mm // nn) for mm, nn in zip(m, (n1, n2))]
    box = xp.take(xp.take(F, xp.asarray((i + np.arange(-hw[0], hw[0] + 1, st[0])) % m[0]), axis=0),
                  xp.asarray((j + np.arange(-hw[1], hw[1] + 1, st[1])) % m[1]), axis=1)
    z = _scalar(F[i, j]) / max(_scalar(xp.median(box)) / np.log(2.0), 1e-30)
    f = np.array([np.fft.fftfreq(m[0])[i], np.fft.fftfreq(m[1])[j]])
    r1, r2 = np.arange(n1), np.arange(n2)
    for h in (0.5, 0.125):                                           # in samples of the padded transform
        t = np.array([-h, 0.0, h])
        E1 = xp.asarray(np.exp(-2j * np.pi * np.outer(f[0] + t / m[0], r1)).astype(np.complex64))
        E2 = xp.asarray(np.exp(-2j * np.pi * np.outer(r2, f[1] + t / m[1])).astype(np.complex64))
        v = np.abs(_xp.host(E1 @ xm @ E2)).astype(np.float64)
        for ax, (lo, mid, hi) in enumerate(((v[0, 1], v[1, 1], v[2, 1]), (v[1, 0], v[1, 1], v[1, 2]))):
            den = lo - 2 * mid + hi
            if den < 0:
                f[ax] += float(np.clip(0.5 * (lo - hi) / den, -1.0, 1.0)) * h / m[ax]
    return f0 + f / D, float(z), float(np.log(n1 * n2) + 10.0)


def _unfringe(x, f, sign=-1):
    """x exp(sign 2 pi j (f1 i + f2 j)) over its pixels (i, j)."""
    xp = _xp.module(x)
    if not (f[0] or f[1]):
        return x
    r1 = xp.asarray(np.exp(sign * 2j * np.pi * f[0] * np.arange(x.shape[0])).astype(np.complex64))[:, None]
    r2 = xp.asarray(np.exp(sign * 2j * np.pi * f[1] * np.arange(x.shape[1])).astype(np.complex64))[None, :]
    return _rowwise(lambda u, a: u * a * r2, x, r1)


def _coherence_map(a, b, w, fringe, tile):
    """The sample coherence of two complex images over w x w windows with a linear fringe removed: the image's own
    ('global'), or in tiles each tile's own (found on the tile and applied to it and a halo of half a window)
    where the tile has a fringe to find (_fringe's test) and the image's elsewhere, or none.
    -> (map float32, fringes [n, 2] (cycles per pixel, as removed), found [n] (bool: the tile's own), the image's
    fringe [2] (zero when the image has none to find))."""
    xp = _xp.module(a)
    n1, n2 = a.shape

    def one(x, pa, pb):
        num = xp.abs(_box(x, w))
        den = xp.sqrt(_box(pa, w)) * xp.sqrt(_box(pb, w))              # a product of the two could leave float32's range
        return xp.where(den > 0, num / xp.maximum(den, 1e-30), 0).astype(xp.float32)
    pa, pb = _power(a), _power(b)
    x = _rowwise(lambda u, v: u * xp.conj(v), a, b)
    if not fringe:
        return one(x, pa, pb), np.zeros((0, 2)), np.zeros(0, bool), np.zeros(2)
    fg, z, thr = _fringe(x)
    if z <= thr:
        fg = np.zeros(2)
    if fringe == 'global' or max(n1, n2) <= tile:
        return one(_unfringe(x, fg), pa, pb), np.array([fg]), np.array([z > thr]), fg
    h = w // 2 + 1
    rows, fr, found = [], [], []
    for i in range(0, n1, tile):
        row = []
        for j in range(0, n2, tile):
            i1, j1 = min(i + tile, n1), min(j + tile, n2)
            a0, b0, a1, b1 = max(i - h, 0), max(j - h, 0), min(i1 + h, n1), min(j1 + h, n2)
            f, own = fg, False
            if min(i1 - i, j1 - j) >= 48:
                ft, zt, tt = _fringe(x[i:i1, j:j1])
                if zt > tt:
                    f, own = ft, True
            fr.append(f)
            found.append(own)
            sl = (slice(a0, a1), slice(b0, b1))
            g = one(_unfringe(x[sl], f), pa[sl], pb[sl])
            row.append(g[i - a0:i - a0 + i1 - i, j - b0:j - b0 + j1 - j])
        rows.append(xp.concatenate(row, axis=1))
    return xp.concatenate(rows, axis=0), np.array(fr), np.array(found, bool), fg


def _stats(g, valid=None):
    xp = _xp.module(g)
    v = g.ravel() if valid is None else _sel(g, valid)
    if v.size == 0:
        return dict(median=None, mean=None, p90=None, above_0p5=None, above_0p7=None)
    med, p90 = (float(t) for t in _xp.host(xp.percentile(v, xp.asarray([50.0, 90.0], dtype=xp.float32))))
    return dict(median=med, mean=_scalar(v.mean()), p90=p90, above_0p5=_scalar((v > 0.5).mean()),
                above_0p7=_scalar((v > 0.7).mean()))


def _check_pair(name, a, b):
    for nm, p in (('a', a), ('b', b)):
        if 'complex' not in str(p['image'].dtype) or p.get('plane', 'ground') != 'ground':
            raise ValueError(f'{name} takes complex ground-plane products; {nm} is not one')
    if not _same_grid(_grid_of(a), _grid_of(b)):
        raise ValueError(f'{name}: the two products are not on one grid: form them with form_cphd(grid=)')


def _shared(a, b):
    """The least fraction of either product's ground spectrum that the other shares, in range or in azimuth, at
    a's scene center (_overlap of the pulses and the bands each image was formed from), and the fractions (range,
    azimuth) of a's that b shares."""
    p = io.local_to_ecf(planes._scp(a), a['meta'])
    la, lb = _look(geometry(a), p), _look(geometry(b), p)
    ab, ba = _overlap(la, lb), _overlap(lb, la)
    return min(ab['range'], ab['azimuth'], ba['range'], ba['azimuth']), (ab['range'], ab['azimuth'])


def _pair_baseline(a, b):
    try:
        return baseline(a, b, point=io.local_to_ecf(planes._scp(a), a['meta']))
    except (KeyError, TypeError, ValueError):
        return None


def coherence(a, b, window=7, *, common_band=True, fringe='tiles', tile=384, floor=True):
    """The coherence of a registered pair of complex images on one grid.

    a, b: ground-plane complex products on the same grid, registered (register); on the host or a device. window:
    the side of the estimation window, pixels. common_band: keep of each image only the part of its spectrum the
    other shares, since what one has alone can only lower the coherence: on the ground the two spectra are the
    sectors their pulses' directions and their bands span, displaced in range by a difference in grazing angle
    (the spectral shift) and in azimuth by a difference in look direction; each image is cut at baseband to the
    other's sector. A pair whose spectra coincide to 2% has nothing to cut and is taken as it is: a pair formed
    on its common support (form_cphd(aperture=, band=) with common_support's selection, as process_stack forms
    it), or an exact repeat. Forming on the common support is the better way, because there the two images also
    carry one weighting of the common part, where a cut of two images each weighted over its own aperture leaves
    them weighted differently (measured with the apertures displaced by 30%: 0.63 by this cut against 0.81 by the
    selection, under Taylor weighting; docs/stack.md).
    fringe: 'tiles' (a linear fringe removed in each tile x tile pixels: the tile's own where it has one to find,
    the image's elsewhere), 'global' (the image's) or None. A residual baseline or registration error and a tilt
    of the ground against the grid plane leave a fringe that a window would otherwise average down. A tile's own
    fringe is taken at the peak of the tile's interferogram spectrum only when that peak stands clear of what a
    spectrum without a fringe gives (_fringe): a tile of water or of vegetation has none, and the peak of its
    noise is not fitted.

    The estimator. |<a b*>| / sqrt(<|a|^2> <|b|^2>) over the window is biased upward: an incoherent pair gives
    about 0.89 / sqrt(n) for n independent samples, and a window of w x w pixels holds fewer than w^2 of those,
    because the image is sampled finer than its resolution. floor: the same estimate, fringe removal included,
    with the second image displaced by 4 windows and a pixel along each axis, where nothing is coherent: its
    statistics are the floor the pair's are to be read against (measured on six spotlights at 0.7 of a resolution
    cell per pixel: a median of 0.25 for 7 x 7, where 49 independent samples would give 0.13).

    -> a product dict with a's geometry: image [nx, ny] float32 (the coherence, 0 where either image is 0),
    product 'coherence', window, median, mean, p90, above_0p5, above_0p7 (over the processed pixels), floor (the
    same statistics of the displaced pair, or None), samples (the independent samples per window the floor's mean
    implies), overlap (range, azimuth: the fractions of a's spectrum that b shares; None without common_band),
    fringes [n, 2] (cycles per pixel, as removed), fringes_found [n] (bool: the tile had its own), fringe (the
    image's, cycles per pixel), baseline (baseline(a, b)), timing."""
    t0 = time.perf_counter()
    _check_pair('coherence', a, b)
    if fringe not in ('tiles', 'global', None, False):
        raise ValueError(f"fringe must be 'tiles', 'global' or None, got {fringe!r}")
    xp = _xp.module(a['image'])
    notes = list(a.get('notes', []))
    overlap = None
    cut = same = False
    if common_band:
        try:
            least, overlap = _shared(a, b)
            same = least >= 0.98
        except (KeyError, TypeError, ValueError):
            pass
    if common_band and not same:
        za, zb = _carrier(a, on=False), _carrier(b, on=False)
        own = [za is not a['image'], zb is not b['image']]
        shape = za.shape
        ma, overlap = _band_mask(a, b, shape, xp)
        mb, _ = _band_mask(b, a, shape, xp)
        if min(overlap) <= 0.02:
            notes.append(f'coherence: the spectra share {100 * overlap[0]:.0f}% in range and {100 * overlap[1]:.0f}% in '
                         'azimuth: outside the critical baseline, the whole bands were used')
        else:
            za = _fft(_fft(za, 'fft2') * ma, 'ifft2')
            zb = _fft(_fft(zb, 'fft2') * mb, 'ifft2')
            own = [True, True]
            cut = True
        za = _carrier(dict(a, image=za, baseband=True), on=True, copy=not own[0])
        zb = _carrier(dict(b, image=zb, baseband=True), on=True, copy=not own[1])
    else:
        za, zb = _carrier(a, on=True), _carrier(b, on=True)
    valid = (a['image'] != 0) & (b['image'] != 0)
    g, fr, found, fg = _coherence_map(za, zb, int(window), fringe, int(tile))
    g = xp.where(valid, g, 0).astype(xp.float32)
    all_valid = bool(_xp.host(valid.all()))
    st = _stats(g, None if all_valid else valid)
    fl = samples = None
    if floor:
        s = 4 * int(window) + 1
        gf = _coherence_map(za[s:, s:], zb[:-s, :-s], int(window), fringe, int(tile))[0]
        fl = _stats(gf, None if all_valid else valid[s:, s:] & valid[:-s, :-s])
        samples = None if not fl['mean'] else float(np.pi / (4 * fl['mean'] ** 2))
        del gf
    res = {k: v for k, v in a.items() if k not in ('image', 'phase_error', 'registration', 'baseband', 'valid')}
    dt = time.perf_counter() - t0
    how = ', whose spectra coincide' if same else ', on their common band (cut in the images)' if cut else ''
    res.update(image=g, product='coherence', window=int(window), floor=fl, samples=samples, overlap=overlap,
               fringes=fr, fringes_found=found, fringe=(float(fg[0]), float(fg[1])), baseline=_pair_baseline(a, b),
               grid=_grid_of(a),
               notes=notes + [f'coherence of two images over {int(window)} x {int(window)} pixels' + how],
               timing=dict(coherence=dt, megapixels_per_second=g.size / 1e6 / max(dt, 1e-9)), **st)
    return res


def _noise(p, box=15):
    """The noise level of an intensity image, from above: the 0.5th percentile of the intensity averaged over
    box x box pixels (the darkest ground still scatters, so the noise is at most this)."""
    s = _box(p, box)
    v = s[::4, ::4].ravel()
    v = _sel(v, v > 0)
    return _percentile(v[:, None], [0.5], step=1)[0] if v.size else 0.0


def _allowed(pa, pb, window):
    """The coherence that the signal-to-noise ratios of two intensity images allow over a window,
    1 / sqrt((1 + 1 / snr_a) (1 + 1 / snr_b)), with each image's noise taken from its darkest ground (_noise: an
    upper bound on the noise, so this is a lower bound). -> (map float32, (noise of a, noise of b))."""
    xp = _xp.module(pa)
    na, nb = max(_noise(pa), 1e-30), max(_noise(pb), 1e-30)
    sa, sb = _box(pa, window), _box(pb, window)

    def f(u, v):
        ra, rb = xp.maximum(u - na, 1e-6 * na) / na, xp.maximum(v - nb, 1e-6 * nb) / nb
        return (1.0 / xp.sqrt((1 + 1 / ra) * (1 + 1 / rb))).astype(xp.float32)
    return _rowwise(f, sa, sb), (na, nb)


def change(a, b, window=7, *, capable=0.9, lost=0.4, **kw):
    """What changed between two registered images of one scene, two ways.

    a, b, window and the other keywords: coherence's; the pair is best formed on its common support
    (process_stack does; common_support). Coherent change is a loss of coherence: the scatterers within a
    resolution cell moved by a fraction of the wavelength, or were replaced. Incoherent change is a change of
    brightness: the ratio of the intensities averaged over the window, in decibels, less its median over the image
    (the difference in gain between the dates).

    What decorrelates is not only what changed. Vegetation, water, snow and tilled or wet soil lose coherence
    within hours to days, and so does any pair outside the critical baseline (baseline) or less than well
    registered. Noise does too: ground with a signal-to-noise ratio s in both images cannot exceed s / (1 + s).
    Low coherence therefore means change only on surfaces that stay coherent, such as buildings, roads and bare
    rock, only where the brightness allows a high coherence at all, and only against the floor of the estimator
    (coherence): on the repeat-pass spotlights measured, fields and water were at the floor and only built
    surfaces above it. allowed is the coherence the two brightnesses allow at each pixel, from each image's
    noise level as its darkest ground bounds it; capable (default 0.9) and lost (0.4) set the summary: of the
    pixels whose brightness allows more than `capable`, the fraction whose coherence is under `lost`. The
    intensity ratio does not need coherence, but its speckle is that of the window's looks: 10 log10 of a ratio
    of two averages over n independent samples scatters by about 6.1 / sqrt(n) dB.

    -> coherence's product dict with product 'change', image [nx, ny] float32 (the intensity ratio b over a, dB),
    coherence [nx, ny] float32, allowed [nx, ny] float32, capable_fraction (of the pixels), changed_fraction (of
    the capable pixels: coherence under `lost`), coherent_fraction (of them: coherence above 0.8), noise (the two
    levels, in the images' units), ratio_sigma (dB: the scatter of the ratio the window's independent samples give
    for unchanged ground, where the floor was measured), and coherence's entries."""
    res = coherence(a, b, window, **kw)
    t0 = time.perf_counter()
    xp = _xp.module(res['image'])
    ia, ib = _power(a['image']), _power(b['image'])
    pa, pb = _box(ia, window), _box(ib, window)
    valid = (pa > 0) & (pb > 0)
    r = _rowwise(lambda u, v: 10.0 * (xp.log10(xp.maximum(v, 1e-30)) - xp.log10(xp.maximum(u, 1e-30))), pa, pb)
    med = _percentile(xp.where(valid, r, 0), [50.0])[0]
    r = xp.where(valid, r - med, 0).astype(xp.float32)
    al, noise = _allowed(ia, ib, window)
    cap = (al > capable) & valid
    ncap = max(_scalar(cap.sum()), 1.0)
    res.update(coherence=res['image'], image=r, product='change', ratio_median_db=med, allowed=al, noise=noise,
               capable_fraction=_scalar(cap.mean()),
               changed_fraction=_scalar((cap & (res['image'] < lost)).sum()) / ncap,
               coherent_fraction=_scalar((cap & (res['image'] > 0.8)).sum()) / ncap,
               ratio_sigma=None if not res.get('samples') else float(10 / np.log(10) * np.sqrt(2.0 / res['samples'])))
    res['notes'] = res['notes'] + ['change: the intensity ratio of the second image over the first (dB, median removed) '
                                   'and the coherence; low coherence is change only where the surface stays coherent '
                                   'and its brightness allows coherence']
    res['timing']['change'] = time.perf_counter() - t0
    return res


def _sensitivity(a, b, point):
    """The interferometric sensitivity to height of a pair of products at a ground point (ECEF), cycles per
    meter: rho (tan(graze a) - tan(graze b)), rho the radius of the ground wavenumber at the middle of the band
    each image was formed from and the grazing angles those of one look direction, the middle of the two
    apertures used. A scatterer h above the grid plane lies in each image where its range puts it, displaced
    toward that radar by h tan(graze), and the pixel of a that shows it sees b's response to it
    h (tan(graze a) - tan(graze b)) of ground range away from its peak, hence with that many cycles of b's ground
    carrier in the phase of a b*. The reciprocal is the height of ambiguity."""
    d = []
    for q in (a, b):
        g = geometry(q)
        if q.get('shift') is not None and any(q['shift']):           # as the file states its orbit (see interferogram)
            R = np.asarray(q['meta']['R'], np.float64)
            g = dict(g, ant=g['ant'] + q['shift'][0] * (np.asarray(q['e1'], np.float64) @ R)
                     + q['shift'][1] * (np.asarray(q['e2'], np.float64) @ R))
        phi, cg = _directions(g, point)
        lo, hi = g.get('pulses') or (0, len(phi))
        d.append([phi[lo:hi], cg[lo:hi], 0.5 * (g['band'][0] + g['band'][1])])
    d[1][0] = d[1][0] + np.round((np.median(d[0][0]) - np.median(d[1][0])) / (2 * np.pi)) * 2 * np.pi
    mid = 0.25 * (d[0][0].min() + d[0][0].max() + d[1][0].min() + d[1][0].max())
    cg = [float(np.interp(mid, ph[np.argsort(ph)], c[np.argsort(ph)])) for ph, c, _ in d]
    rho = (d[0][2] * cg[0] + d[1][2] * cg[1]) / C
    return float(rho * (np.tan(np.arccos(cg[0])) - np.tan(np.arccos(cg[1]))))


def _plane_fit(a, b):
    """_sensitivity over the grid of a: -> (c0, c1, c2) with the sensitivity at pixel (i, j) c0 + c1 i + c2 j."""
    g = _grid_of(a)
    nx, ny = g['shape']
    pts, val = [], []
    for i in (0.0, nx / 2.0, nx - 1.0):
        for j in (0.0, ny / 2.0, ny - 1.0):
            pts.append((1.0, i, j))
            val.append(_sensitivity(a, b, g['origin'] + i * g['spx'] * g['e1'] + j * g['spy'] * g['e2']))
    return np.linalg.lstsq(np.array(pts), np.array(val), rcond=None)[0]


def _upsample(c, shape, step, xp):
    """A lattice c [ni, nj] of values at every step-th pixel, bilinearly at every pixel of shape (an array of xp)."""
    c = xp.asarray(np.asarray(c, np.float32))

    def along(v, n, axis):
        t = np.arange(n) / step
        i0 = np.minimum(t.astype(np.int64), v.shape[axis] - 2)
        w = xp.asarray((t - i0).astype(np.float32))
        w = w[:, None] if axis == 0 else w[None, :]
        lo, hi = xp.take(v, xp.asarray(i0), axis=axis), xp.take(v, xp.asarray(i0 + 1), axis=axis)
        return lo * (1 - w) + hi * w
    return along(along(c, shape[1], 1), shape[0], 0)


def _height(out, dem, step=4, iterations=3):
    """The height (m) above a product's grid plane of the surface dem(lat, lon) -> height (m above the WGS-84
    ellipsoid; arrays of degrees) where each pixel shows it, at every step-th pixel [ni, nj] (the lattice reaches
    past the last pixel). A point h above the plane appears displaced toward the radar by h tan(graze) of ground
    range, so the pixel's height is the surface's that far behind it, found by repeated substitution (a surface
    steep enough to lay over itself has several such points; one is returned)."""
    g = _grid_of(out)
    nx, ny = g['shape']
    n = np.cross(g['e1'], g['e2'])
    i, j = np.arange(0, nx + step, step), np.arange(0, ny + step, step)
    P = g['origin'] + (i * g['spx'])[:, None, None] * g['e1'] + (j * g['spy'])[None, :, None] * g['e2']
    geo = geometry(out)
    lo, hi = geo.get('pulses') or (0, len(geo['ant']))
    u = g['center'] - geo['ant'][(lo + hi) // 2]
    ug = u - (u @ n) * n
    back = ug / np.linalg.norm(ug) * (-(u @ n) / np.linalg.norm(ug))             # tan(graze), away from the radar
    h = np.zeros(P.shape[:2])
    for _ in range(int(iterations)):
        lat, lon, _h = io.ecf_to_geodetic((P + h[..., None] * back).reshape(-1, 3))
        H = np.asarray(dem(np.asarray(lat), np.asarray(lon)), np.float64)
        h = ((io.geodetic_to_ecf(lat, lon, H) - g['origin']) @ n).reshape(P.shape[:2])
    return h


def interferogram(a, b, window=7, *, fringe=True, dem=None, dem_step=4, floor=True):
    """The interferogram of a registered pair formed on its common support: the phase of a b* and its coherence.

    a, b: ground-plane complex products on one grid, registered (register, to a few hundredths of a pixel:
    coherent=True), formed from the selection common_support gives (form_cphd(aperture=, band=)); on the host or
    a device. window: the side of the averaging window, pixels. fringe: remove one linear fringe, the image's own
    (the peak of the interferogram's spectrum, when it stands clear of noise): a difference between the two
    orbits' errors leaves a tilt of the phase across the scene that carries no information about it. dem: a
    surface dem(lat, lon) -> height (m above the WGS-84 ellipsoid, arrays of degrees; products.read_dem), whose
    topographic phase is removed, evaluated at every dem_step-th pixel.

    Phase. Both images are formed on one plane, so ground in that plane has no phase in a b* whatever the two
    look directions (there is no flat-earth fringe to remove). A scatterer h above the plane turns the phase by
    2 pi h / height_of_ambiguity, with 1 / height_of_ambiguity = rho (tan(graze a) - tan(graze b)), rho the radius
    of the common ground wavenumber band (signed: positive when a looks down more steeply). A displacement d of
    the ground toward the radar between the two collections turns it by 4 pi d / wavelength (b after a: positive
    phase for motion toward the radar). The phase is that of a b* averaged over the window, after the fringe and
    the topography, less its value over the most coherent tenth of the scene (an orbit error of a fraction of the
    wavelength makes the absolute phase meaningless). Earth curvature counts as height: the plane stands d^2 /
    (2 R) above the surface a distance d from the grid center, which a dem accounts for and a small height of
    ambiguity makes visible.

    -> a product dict with a's geometry: image [nx, ny] float32 (the phase, radians in (-pi, pi]; 0 where either
    image is 0), coherence [nx, ny] float32 (over the same window, same fringe), product 'interferogram', window,
    fringe (cycles per pixel along the axes, as removed), fringe_per_km (cycles), fringe_found, phase_offset (rad,
    removed), height_of_ambiguity (m, at the grid center; None for equal grazing angles), height_per_radian,
    displacement_per_radian (m of line-of-sight motion), wavelength, topography (whether a dem was removed),
    phase_std (rad: of the phase where the coherence exceeds 0.6), flatness (dict: the phase of the complex mean
    over blocks of 128 pixels at least 5% coherent: std, p5, p95 in rad, blocks), median, mean, p90, above_0p5,
    above_0p7 (of the coherence), floor and samples (coherence's), baseline, timing."""
    t0 = time.perf_counter()
    _check_pair('interferogram', a, b)
    xp = _xp.module(a['image'])
    w = int(window)
    za, zb = _carrier(a, on=True), _carrier(b, on=True)
    valid = (a['image'] != 0) & (b['image'] != 0)
    x = _rowwise(lambda u, v: u * xp.conj(v), za, zb)
    g0 = _grid_of(a)
    kappa = _sensitivity(a, b, g0['center'])
    notes = list(a.get('notes', []))
    if dem is not None:
        step = int(max(1, dem_step))
        h = _height(a, dem, step)
        kc = _plane_fit(a, b)
        ii, jj = np.arange(h.shape[0]) * step, np.arange(h.shape[1]) * step
        cyc = _upsample(h * (kc[0] + kc[1] * ii[:, None] + kc[2] * jj[None, :]), x.shape, step, xp)
        turn = lambda u, c: u * (xp.cos(2 * np.pi * c) - 1j * xp.sin(2 * np.pi * c)).astype(xp.complex64)
        x = _rowwise(turn, x, cyc)
        notes.append(f'interferogram: the topographic phase of the surface given removed ({np.ptp(h):.1f} m of relief '
                     'above the grid plane)')
        del cyc
    f, found = np.zeros(2), False
    if fringe:
        f, z, thr = _fringe(x)
        found = z > thr
        f = f if found else np.zeros(2)
        x = _unfringe(x, f)
    pa, pb = _power(za), _power(zb)
    xs = _box(x, w)
    den = xp.sqrt(_box(pa, w)) * xp.sqrt(_box(pb, w))
    g = xp.where(valid & (den > 0), xp.abs(xs) / xp.maximum(den, 1e-30), 0).astype(xp.float32)
    all_valid = bool(_xp.host(valid.all()))
    st = _stats(g, None if all_valid else valid)
    top = g >= max(st['p90'] or 0.0, 1e-6)
    off = float(np.angle(complex(_xp.host(xp.where(top, xs, 0).sum()))))
    rot = np.complex64(np.exp(-1j * off))
    phase = xp.where(valid, xp.angle(xs * rot), 0).astype(xp.float32)
    solid = g > 0.6
    nsolid = _scalar(solid.sum())
    pstd = None if nsolid < 2 else float(np.sqrt(max(_scalar(xp.where(solid, phase * phase, 0).sum()) / nsolid
                                                     - (_scalar(xp.where(solid, phase, 0).sum()) / nsolid) ** 2, 0.0)))
    flat = None
    T = 128
    t1, t2 = x.shape[0] // T, x.shape[1] // T
    if min(t1, t2) >= 2:
        xb = _xp.host(xp.where(solid, x, 0)[:t1 * T, :t2 * T].reshape(t1, T, t2, T).sum((1, 3)))
        cnt = _xp.host(solid[:t1 * T, :t2 * T].reshape(t1, T, t2, T).sum((1, 3)))
        blk = np.angle(xb * rot)[cnt > 0.05 * T * T]
        if blk.size:
            flat = dict(std=float(np.std(blk)), p5=float(np.percentile(blk, 5)), p95=float(np.percentile(blk, 95)),
                        blocks=int(blk.size), block=T)
    fl = samples = None
    if floor:
        s = 4 * w + 1
        gf = _coherence_map(za[s:, s:], zb[:-s, :-s], w, 'global' if fringe else None, 1 << 30)[0]
        fl = _stats(gf, None if all_valid else valid[s:, s:] & valid[:-s, :-s])
        samples = None if not fl['mean'] else float(np.pi / (4 * fl['mean'] ** 2))
        del gf
    lam = 4 * C / (a['band'][0] + a['band'][1] + b['band'][0] + b['band'][1])
    hamb = None if abs(kappa) < 1e-9 else 1.0 / kappa
    res = {k: v for k, v in a.items() if k not in ('image', 'phase_error', 'registration', 'baseband', 'valid')}
    dt = time.perf_counter() - t0
    res.update(image=phase, coherence=g, product='interferogram', window=w, fringe=(float(f[0]), float(f[1])),
               fringe_per_km=(float(f[0] / a['spx'] * 1e3), float(f[1] / a['spy'] * 1e3)), fringe_found=bool(found),
               phase_offset=off, height_of_ambiguity=hamb,
               height_per_radian=None if hamb is None else hamb / (2 * np.pi),
               displacement_per_radian=lam / (4 * np.pi), wavelength=lam, topography=dem is not None, phase_std=pstd,
               flatness=flat, floor=fl, samples=samples, baseline=_pair_baseline(a, b), grid=g0,
               notes=notes + [f'interferogram of two images over {w} x {w} pixels'
                              + (', one linear fringe removed' if found else '')],
               timing=dict(interferogram=dt, megapixels_per_second=g.size / 1e6 / max(dt, 1e-9)), **st)
    return res


def coherent_sum(a, b, window=15, *, equalize=True, coherence_window=7, dark=0.01):
    """The coherent sum of a registered pair formed on its common support, and what it gained.

    a, b: as interferogram's. The second image is brought to the first's phase, the linear fringe of the pair and
    what remains of the interferogram's phase averaged over window x window pixels (propagation, orbit, relief:
    whatever varies slowly), and to its amplitude (equalize: one factor, from the most coherent tenth of the
    scene), and added. Where the pair is coherent the amplitudes add and the intensity is 2 (1 + g) of one
    image's for a coherence g; noise adds in power, 2 of one image's, so the signal-to-noise ratio gains 1 + g: 3
    dB for a perfectly coherent pair, nothing for an incoherent one, and the image keeps its resolution, where the
    incoherent average (mean) trades none of the noise for its smoother speckle. The phase a pixel of b is turned
    by comes from its surroundings, the window without its central 5 x 5 pixels: a phase estimated from the pixel
    itself would line up the two images' noise there, and the sum's noise would exceed twice one image's. The
    gain is measured, not assumed.

    -> a product dict with a's geometry: image [nx, ny] complex64 (a + b as brought to it, on its carrier; 0
    where either is 0), product 'coherent_sum', phase_window, fringe (cycles per pixel), amplitude_factor (applied
    to b), gain (dict, dB, against the first image: signal, the intensity of the sum over a's on the most
    coherent tenth of the scene; noise, the same on the darkest incoherent ground, as many pixels as the fraction
    `dark` of the scene, by the two images' intensity averaged over 15 x 15, among those whose coherence over 31 x
    31 pixels is at most twice that estimator's floor on the pair (the second image displaced by 64 pixels): dark
    ground the two images share adds coherently and is not noise (noise_reference 'incoherent'; 'darkest', the
    darkest ground of all, where under 1% of the scene is incoherent); snr, their difference; second, the same against the second
    image, which differs where the two images' signal-to-noise ratios do (the sum gains most against the
    noisier); expected, 10 log10(1 + g) for the mean coherence g of that tenth and equal noise; incoherent, the
    signal gain of the sum of the intensities; coherence, that g), timing."""
    t0 = time.perf_counter()
    _check_pair('coherent_sum', a, b)
    xp = _xp.module(a['image'])
    za, zb = _carrier(a, on=True), _carrier(b, on=True)
    valid = (a['image'] != 0) & (b['image'] != 0)
    x = _rowwise(lambda u, v: u * xp.conj(v), za, zb)
    f, z, thr = _fringe(x)
    f = f if z > thr else np.zeros(2)
    x = _unfringe(x, f)
    pa, pb = _power(za), _power(zb)
    w7, w = int(coherence_window), int(window)
    den = xp.sqrt(_box(pa, w7)) * xp.sqrt(_box(pb, w7))
    g = xp.where(valid & (den > 0), xp.abs(_box(x, w7)) / xp.maximum(den, 1e-30), 0).astype(xp.float32)
    del den
    p90 = _percentile(g, [90.0], step=3)[0]
    stable = (g >= max(p90, 1e-6)) & valid
    ns = max(_scalar(stable.sum()), 1.0)
    sa, sb = _scalar(xp.where(stable, pa, 0).sum()) / ns, _scalar(xp.where(stable, pb, 0).sum()) / ns
    amp = float(np.sqrt(sa / max(sb, 1e-30))) if equalize else 1.0
    ref = _box(x, w) * xp.float32(w * w) - (_box(x, 5) * xp.float32(25.0) if w > 7 else 0)
    del x
    zb = _unfringe(zb, f, sign=+1)
    # the ground the noise is read on must be incoherent: dark ground that the two images share (a smooth playa, a
    # calm water surface near the noise level) adds coherently and would read as noise that gained more than 3 dB.
    # Coherence over 31 x 31 pixels, against its floor on the same images with the second displaced by 64 pixels
    gw, sh = 31, 64
    def coh31(v, pv, ok):
        d = xp.sqrt(_box(pa, gw)) * xp.sqrt(_box(pv, gw))
        return xp.where(ok & (d > 0), xp.abs(_box(_rowwise(lambda u, t: u * xp.conj(t), za, v), gw)) / xp.maximum(d, 1e-30), 0)
    g31 = coh31(zb, pb, valid).astype(xp.float32)
    okf = valid & xp.roll(valid, (sh, sh), axis=(0, 1))
    gf = coh31(xp.roll(zb, (sh, sh), axis=(0, 1)), xp.roll(pb, (sh, sh), axis=(0, 1)), okf)
    floor31 = float(_percentile(xp.where(okf, gf, np.inf), [50.0 * _scalar(okf.mean())], step=3)[0])
    del gf, okf
    quiet = valid & (g31 <= 2.0 * floor31)
    del g31
    tot = _rowwise(lambda u, v, r: u + xp.complex64(amp) * v * (r / xp.maximum(xp.abs(r), 1e-30)), za, zb, ref)
    del ref, zb
    tot = xp.where(valid, tot, 0).astype(xp.complex64)
    ps = _power(tot)
    one = 0.5 * (pa + xp.float32(amp * amp) * pb)
    lev = _box(one, 15)
    nq, nv = _scalar(quiet.sum()), max(_scalar(valid.sum()), 1.0)
    rule = 'incoherent' if nq >= 0.01 * nv else 'darkest'
    pool = quiet if rule == 'incoherent' else valid           # too little incoherent ground: the darkest, as it is
    frac = float(dark) * nv / max(_scalar(pool.sum()), 1.0) if rule == 'incoherent' else float(dark)
    if frac >= 1.0:                                          # the incoherent ground is all the noise there is
        dk = pool
    else:
        dk = (lev <= _percentile(xp.where(pool, lev, np.inf), [100.0 * frac * _scalar(pool.mean())], step=2)[0]) & pool
    sig = _scalar(xp.where(stable, ps, 0).sum()) / ns
    nsum, na, nb = (_scalar(xp.where(dk, v, 0).sum()) for v in (ps, pa, pb))
    first = (sig / max(sa, 1e-30)) / (nsum / max(na, 1e-30))
    second = (sig / max(sb, 1e-30)) / (nsum / max(nb, 1e-30))
    gs = _scalar(xp.where(stable, g, 0).sum()) / ns
    db = lambda v: float(10 * np.log10(max(v, 1e-30)))
    res = {k: v for k, v in a.items() if k not in ('image', 'phase_error', 'registration', 'valid')}
    dt = time.perf_counter() - t0
    res.update(image=tot, baseband=False, product='coherent_sum', phase_window=w, amplitude_factor=amp,
               fringe=(float(f[0]), float(f[1])), grid=_grid_of(a),
               gain=dict(signal=db(sig / max(sa, 1e-30)), noise=db(nsum / max(na, 1e-30)), snr=db(first),
                         second=db(second), expected=db(1 + gs),
                         incoherent=db((sa + amp * amp * sb) / max(sa, 1e-30)), coherence=gs,
                         noise_reference=rule, noise_pixels=int(_scalar(dk.sum())), floor31=floor31),
               notes=list(a.get('notes', [])) + ['coherent sum of two images, the second brought to the phase and '
                                                 'amplitude of the first'],
               timing=dict(coherent_sum=dt, megapixels_per_second=tot.size / 1e6 / max(dt, 1e-9)))
    return res


# ---------------------------------------------------------------------------------------------- from the files
def _default_grid(path, geos, reference, spacing, extent):
    """The grid of process_stack when none is given: the reference collection's tasked area on axes along its line
    of sight (e2 away from the radar on the ground, so that its shadows fall down the rows), horizontal at the
    area's center at the height of the collection's reference point, with square pixels of 0.7 of the coarsest
    collection's ground range resolution c / (2 B cos(graze)), finer where some collection's spectrum on these
    axes asks for it."""
    from .cphd import _image_area
    corners, ref = _image_area(path, dict(origin=np.zeros(3), R=np.eye(3)))       # ECEF
    if corners is None and (extent is None or ref is None):
        raise ValueError('the reference collection states no image area: give extent= (m) or grid=')
    cen = ref if corners is None else corners.mean(0)
    lat, lon, h = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(cen))
    if ref is not None:
        h = float(np.ravel(io.ecf_to_geodetic(ref)[2])[0])
    cen = np.ravel(io.geodetic_to_ecf(lat, lon, h))
    east, north, up = _enu(lat, lon)
    g = geos[reference]
    los = cen - g['ant'][len(g['ant']) // 2]
    e2 = _unit(los - (los @ up) * up)
    e1 = np.cross(e2, up)
    if extent is not None:
        L1, L2 = _pair(extent, 'extent')
    else:
        d = corners - cen
        L1, L2 = 2 * np.abs(d @ e1).max(), 2 * np.abs(d @ e2).max()
    if spacing is None:
        res, b1, b2 = [], [], []
        for q in geos:                                               # each collection's baseband spectrum on e1, e2
            lk = _look(q, cen)
            m = len(lk['phi']) // 2
            f = np.array([lk['band'][0], lk['band'][1]])
            k = (2 / C) * f[None, :, None] * (lk['cosg'][:, None, None] * (
                np.sin(lk['phi'])[:, None, None] * east + np.cos(lk['phi'])[:, None, None] * north))
            k = k - (2 / C) * f.mean() * lk['cosg'][m] * (np.sin(lk['phi'][m]) * east + np.cos(lk['phi'][m]) * north)
            res.append(C / (2 * (f[1] - f[0]) * lk['cosg'][m]))
            b1.append(np.ptp(k @ e1))
            b2.append(np.ptp(k @ e2))
        spacing = min(0.7 * max(res), 0.96 / max(b1), 0.96 / max(b2))
    return ground_grid((lat, lon, h), (L1, L2), spacing, float(np.degrees(np.arctan2(e2 @ east, e2 @ north))))


def _on_device(image, backend, notes):
    """image where the stack's stages run for a backend: on its device when the images and transforms of a stage
    (about six times the complex image) fit what is free there (memory.cuda_plan on _xp.cuda_budget(), or the JAX
    device's), else on the host, with a note."""
    from . import memory
    need = 6 * 8 * image.size
    if backend == 'cuda':
        if not memory.cuda_plan(dict(ground=need), _xp.cuda_budget())['fits']:
            notes.append(f'the GPU has no room for the stack stages of a {image.shape[0]} x {image.shape[1]} image '
                         f'({need / 1e9:.1f} GB): they ran on the host')
            return image
        return _xp.device(image, 'cuda')                             # complete when it returns: the image is given up
    if backend in ('jax', 'tpu'):
        budget = _xp.jax_budget()
        if budget is not None and not memory.cuda_plan(dict(ground=need), budget, joined=True)['fits']:
            notes.append(f'the JAX device has no room for the stack stages of a {image.shape[0]} x {image.shape[1]} '
                         f'image ({need / 1e9:.1f} GB): they ran on the host')
            return image
        return _xp.device(image, backend)
    return image


def write_geotiff(path, out, bands=None):
    """A stack product on its ground grid as a GeoTIFF in the UTM zone of its center (float32; needs rasterio).

    bands: the arrays [nx, ny] to write (default the product's image; a change product: its coherence and its
    intensity ratio; an interferogram: its phase and its coherence). The grid is written as it is, without
    resampling, with the affine transform that places its pixels on the map (rotated by the grid's azimuth; rows
    along e1, columns along e2), when that places every pixel within a tenth of a pixel of its true position, as
    it does for grids of a few kilometers; a larger grid is resampled north up (products.geocode_image).
    -> dict(crs, transform, resampled)."""
    from . import products, _deps
    _deps.require('rasterio')
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp
    if bands is None:
        keys = dict(change=['coherence', 'image'], interferogram=['image', 'coherence'])
        bands = [out[k] for k in keys.get(out.get('product'), ['image'])]
    bands = [np.asarray(_xp.host(b), np.float32) for b in bands]
    nx, ny = bands[0].shape
    g = _grid_of(out)
    crs = products._utm(g['lat'], g['lon'])
    ti, tj = np.meshgrid(np.linspace(-0.5, nx - 0.5, 5), np.linspace(-0.5, ny - 0.5, 5), indexing='ij')
    ecf = g['origin'] + ti[..., None] * g['spx'] * g['e1'] + tj[..., None] * g['spy'] * g['e2']
    lat, lon, _ = io.ecf_to_geodetic(ecf)
    x, y = (np.asarray(v).reshape(ti.shape) for v in warp('EPSG:4326', crs, lon.ravel(), lat.ravel()))
    # x = a col + b row + c, with the pixel centers at half-integer raster coordinates
    A = np.stack([tj.ravel() + 0.5, ti.ravel() + 0.5, np.ones(ti.size)], 1)
    cx, cy = np.linalg.lstsq(A, x.ravel(), rcond=None)[0], np.linalg.lstsq(A, y.ravel(), rcond=None)[0]
    err = np.hypot(A @ cx - x.ravel(), A @ cy - y.ravel()).max()
    if err <= 0.1 * min(g['spx'], g['spy']):
        tr = Affine(cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
        products.write_geotiff(path, np.stack(bands), tr, crs)
        return dict(crs=crs, transform=tr, resampled=False)
    geo = [products.geocode_image(dict(out, image=bands[0]), b, crs=crs) for b in bands]
    products.write_geotiff(path, np.stack([q['data'] for q in geo]), geo[0]['transform'], crs)
    return dict(crs=crs, transform=geo[0]['transform'], resampled=True)


def process_stack(cphds, output=None, *, product='mean', grid=None, reference=0, spacing=None, extent=None,
                  register='fourier', equalize='mean', weights=None, window=7, weighting='taylor', aperture='auto',
                  dem=None, remap='nrl', remap_args=None, shadows_down=True, backend='auto', precision='float32',
                  max_shift=64, range_gate=True, channel=0, troposphere=None, reader_options=None, target_db=-40.0):
    """Several collections of one scene, from the CPHD files to one product (see the module docstring).

    cphds: the files, two or more. output: the file to write, or None to return the product without writing it.
    product: 'mean' (the incoherent average of all of them, written as a SIDD: detected, remapped, shadows down;
    the speckle-reduced display product), or a product of two collections as a coherent pair: 'coherence' (written
    as a GeoTIFF), 'change' (the coherence and the intensity ratio in decibels: a GeoTIFF of two bands),
    'interferogram' (the phase and the coherence: a GeoTIFF of two bands) or 'coherent_sum' (the sum of the two
    complex images, written as a SIDD like the mean).
    grid: the common ground grid (stack.ground_grid, or a product: its pixels). None: the reference collection's
        tasked area on axes along its line of sight, so that its shadows fall down the rows, with square pixels of
        `spacing` (default: 0.7 of the coarsest collection's ground range resolution, finer where a collection's
        spectrum on those axes needs it) and sides `extent` (m, across and along that line of sight; default the
        tasked area's).
    reference: the index of the collection the others are registered to; the product has its geolocation.
    register: 'fourier' (one translation per image from the detected images' cross-correlation, applied on the
        spectrum: stack.register, stack.shift_image), 'reform' (applied by forming the collection again on the
        displaced grid, from the file: exact at the edges too, at the cost of a second read and formation) or None
        (the collections' own geolocation is trusted). A coherent pair's translation is refined on the complex
        images (stack.register's coherent). max_shift: the largest translation searched, pixels.
    equalize, weights: stack.mean's (default: each image scaled to the reference's brightness; None keeps the
        images' own units). window: the window of the coherence and of the interferogram, pixels.
    weighting: 'taylor' (nbar 4, 35 dB, on the phase history) or 'uniform'; for a moving-beam collection, as in
        process_cphd, the Taylor window along frequency and over each pixel's processed azimuth band, or a
        rectangle over that band and no frequency window. aperture: form_cphd's; with 'auto' the
        two collections of a coherent pair are formed from their common support (stack.common_support): the
        pulses that look from the interval of azimuth both apertures cover and, on each pulse, the frequencies
        whose ground-range wavenumbers both bands hold, under one weighting of that support. What one collection
        has alone only lowers the coherence, and it is removed from the phase history, not from the image.
    dem: an interferogram's surface (stack.interferogram's), whose topographic phase is removed. It does not
        focus anything: terrain-informed focusing (fastsar.terrain, process_cphd(dem=)) corrects an image on a
        collection's own axes along its line of sight, and the collections of a stack share one grid.
    remap, remap_args, shadows_down: the SIDD's (sidd.write_sidd).
    range_gate: form_cphd's; True by default here: a stack's grid is often small against the collections' swaths,
        and a gated first level is several times faster.
    backend, precision, channel, troposphere, reader_options, target_db: form_cphd's.

    The collections are read and formed one at a time (the reference first), so that one phase history is in
    memory at a time, and each image goes to the backend's device for registration and the product, or stays on
    the host when the device lacks the room (a note says so).

    -> the product dict of stack.mean, stack.coherence, stack.change, stack.interferogram or stack.coherent_sum
    (arrays on the host), with registrations (one dict per collection, in the order given: stack.register's),
    inputs (per collection: path, name, collector, start, graze, azimuth, pulses, band, shift, notes), timing
    (seconds: read, form, register, combine, write, total) and, when written, path and metadata (the SIDD's, or
    the GeoTIFF's crs and transform). A pair's product has support (stack.common_support's figures for the two
    files, the reference first, without the per-pulse bands; trimmed: whether the pair was formed from it) and
    baseline (stack.baseline's, of the files). gains and looks_single are per collection in the order given;
    looks_by_n counts the images as processed, the reference first.

    The SIDD of a mean states what it is: ProductClass 'Detected Image', ProductType 'Multi-image incoherent
    average', the number of collections, the equivalent number of looks and the registration as
    ProductCreationExtension parameters, one ExploitationFeatures Collection per input with its own date, time and
    geometry, and the reference collection's aperture and time for the projection (Measurement). A coherent sum's
    likewise, as ProductType 'Multi-image coherent sum' with the measured gain."""
    from .api import _backend
    from .cphd import form_cphd
    from . import products as _products
    pairs = dict(coherence=coherence, change=change, interferogram=interferogram, coherent_sum=coherent_sum)
    if product != 'mean' and product not in pairs:
        raise ValueError("product must be 'mean', 'coherence', 'change', 'interferogram' or 'coherent_sum', "
                         f'got {product!r}')
    if register not in ('fourier', 'reform', None, False):
        raise ValueError(f"register must be 'fourier', 'reform' or None, got {register!r}")
    if weighting not in ('taylor', 'uniform'):
        raise ValueError(f"weighting must be 'taylor' or 'uniform', got {weighting!r}")
    paths = [cphds] if isinstance(cphds, (str, bytes)) or hasattr(cphds, '__fspath__') else list(cphds)
    n = len(paths)
    if n < 2:
        raise ValueError(f'a stack is two collections or more, got {n}')
    if product != 'mean' and n != 2:
        raise ValueError(f"product={product!r} takes two collections, got {n}")
    if not 0 <= reference < n:
        raise ValueError(f'reference {reference} is not one of the {n} collections')
    if grid is not None and (spacing is not None or extent is not None):
        raise ValueError('grid= gives the spacing and the shape: leave spacing and extent unset')
    if weights is not None and len(weights) != n:
        raise ValueError(f'weights: one per collection ({n}), got {len(weights)}')
    if dem is not None and product != 'interferogram':
        raise ValueError("dem= is the surface of product='interferogram'")
    backend = _backend(backend)
    t_start = time.perf_counter()
    timing = dict(read=0.0, form=0.0, register=0.0, combine=0.0, write=0.0)
    notes = []
    geos = None
    try:
        geos = [geometry(p, channel) for p in paths]
    except Exception as e:                                           # not CPHD files: no plan from their pulses
        if grid is None:
            raise ValueError(f'the default grid is planned from the per-pulse parameters of CPHD files ({e}): '
                             'give grid= (fastsar.stack.ground_grid)') from e
    g = _default_grid(paths[reference], geos, reference, spacing, extent) if grid is None else as_grid(grid)
    mpx = g['shape'][0] * g['shape'][1] / 1e6

    def sync(image=None):
        if backend == 'cuda':
            import cupy
            cupy.cuda.Stream.null.synchronize()
        elif _xp.kind(image) == 'jax':
            image.block_until_ready()

    order = [reference] + [k for k in range(n) if k != reference]
    select, support, bl = {}, None, None
    if product != 'mean' and geos is not None:                       # the pair on its common support
        cs = common_support(geos[order[0]], geos[order[1]], g['center'])
        bl = baseline(geos[order[0]], geos[order[1]], g['center'])
        support = {k: v for k, v in cs.items() if k != 'bands'}
        # a fifth of a square cell's aperture, or a tenth of a band, is too little to form an image from
        enough = (cs['inside'] and cs['azimuth_deg'] >= 0.2 * cs['azimuth_needed_deg']
                  and min(cs['overlap_range']) >= 0.1)
        support['trimmed'] = bool(enough and aperture == 'auto')
        if cs['inside'] and not enough:
            notes.append(f"the two collections share {cs['azimuth_deg']:.2f} of the {cs['azimuth_needed_deg']:.2f} "
                         f"degrees of azimuth a square cell takes and {100 * min(cs['overlap_range']):.0f}% of a "
                         'band: too little to form from, each was formed as it would be alone')
        if support['trimmed']:
            select = {order[0]: (cs['pulses'][0], cs['bands'][0]), order[1]: (cs['pulses'][1], cs['bands'][1])}
            notes.append(f"the two collections were formed from their common support: {cs['azimuth_deg']:.2f} degrees "
                         f"of azimuth ({cs['azimuth_available_deg']:.2f} common), "
                         f"{100 * cs['overlap_range'][0]:.0f}% and {100 * cs['overlap_range'][1]:.0f}% of their bands")
        elif not cs['inside']:
            notes.append('the two collections have no common support (outside the critical baseline): each was '
                         'formed as it would be alone')
    timing['plan'] = time.perf_counter() - t_start

    def form(k, shift=None):
        t0 = time.perf_counter()
        mark = {}
        ap, fb = select.get(k, (aperture, None))
        out = form_cphd(paths[k], grid=g, shift=shift, backend=backend, window=weighting == 'taylor', aperture=ap,
                        azimuth_window=weighting,                    # a moving beam's, over each pixel's azimuth band
                        band=fb, precision=precision, channel=channel, troposphere=troposphere,
                        reader_options=reader_options, target_db=target_db, range_gate=range_gate,
                        _plan=lambda q: mark.update(read=time.perf_counter()))
        t1 = time.perf_counter()
        timing['read'] += mark['read'] - t0
        timing['form'] += t1 - mark['read']
        out['image'] = _on_device(out['image'], backend, notes)
        return out

    acc = _Mean(equalize or None, 32) if product == 'mean' else None
    regs, inputs, sms, kept = [None] * n, [None] * n, [None] * n, {}
    ref_det = None
    for k in order:
        out = form(k)
        t0 = time.perf_counter()
        if k == reference:
            if register:
                ref_det = _detected(out['image'])
            est = dict(shift=(0.0, 0.0), shift_px=(0.0, 0.0), reference=True)
        elif register:
            est = _estimate(ref_det, _detected(out['image']), g['spx'], g['spy'], max_shift, 4)
            est['reference'] = False
            if est['found'] and product != 'mean':
                _refine(est, kept[reference], out)
            if not est['found']:
                notes.append(f"collection {k}: no significant correlation peak against the reference (significance "
                             f"{est['significance']:.1f}): not translated")
            elif register == 'fourier':
                out = shift_image(out, est['shift'], inplace=True)
            else:
                sync(out['image'])
                timing['register'] += time.perf_counter() - t0
                out = form(k, est['shift'])
                t0 = time.perf_counter()
        else:
            est = dict(shift=(0.0, 0.0), shift_px=(0.0, 0.0), reference=False)
        out.setdefault('shift', (0.0, 0.0))
        sync(out['image'])
        timing['register'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        regs[k] = est
        lk = _look(geometry(out), g['center'])
        inputs[k] = dict(path=str(paths[k]), name=out['meta'].get('core_name'), collector=out['meta'].get('collector'),
                         start=str(out['meta'].get('start')), graze=lk['graze'], azimuth=lk['azimuth'],
                         pulses=tuple(out['pulses']), band=tuple(out['band']), shift=tuple(out['shift']),
                         notes=list(out['notes']))
        if output is not None and product in ('mean', 'coherent_sum'):   # each input's own collection metadata
            sms[k] = _products.sicd_meta(dict(out, image=np.broadcast_to(np.complex64(0), g['shape'])))[0]
        if product == 'mean':
            acc.add(out, 1.0 if weights is None else float(weights[k]))
            sync(acc.sum)
        else:
            kept[k] = out
        del out
        timing['combine'] += time.perf_counter() - t0
    t0 = time.perf_counter()
    if product == 'mean':
        res = _mean_product(acc, t0)
        res['inputs'] = [dict(inputs[k], **{q: v for q, v in i.items() if q in ('gain', 'weight')})
                         for k, i in zip(order, acc.inputs)]
        res['inputs'] = [res['inputs'][order.index(k)] for k in range(n)]
        for key in ('gains', 'looks_single'):                        # per collection, in the order given
            if res.get(key) is not None:
                res[key] = [res[key][order.index(k)] for k in range(n)]
    else:
        a, b = kept[reference], kept[order[1]]
        if product == 'interferogram':
            res = interferogram(a, b, window, dem=dem)
        elif product == 'coherent_sum':
            res = coherent_sum(a, b, coherence_window=window)
        else:
            res = pairs[product](a, b, window)
        res['inputs'] = inputs
        if bl is not None:
            res.update(baseline=bl, support=support)
        del a, b, kept
    sync(res['image'])
    timing['combine'] += time.perf_counter() - t0
    res.update(registrations=regs, reference=reference, notes=list(res.get('notes', [])) + notes)
    sidd = product in ('mean', 'coherent_sum')
    if sidd:
        res.update(remap=remap, remap_args=dict(remap_args or {}), shadows_down=bool(shadows_down))
    t0 = time.perf_counter()
    if output is not None and sidd:
        from .sidd import write_sidd
        shifts = '; '.join(f'{i["shift"][0]:.2f} {i["shift"][1]:.2f}' for i in res['inputs'])
        ext = {'NumberOfCollections': n, 'ReferenceCollection': reference + 1,
               'Combination': 'mean of the detected intensities at full resolution'
                              + (f', each scaled to the brightness of the reference ({equalize})' if equalize else ''),
               'Registration': ('none' if not register else
                                'one translation per collection against the reference, m along the grid axes: ' + shifts),
               'Geolocation': "the reference collection's"}
        if res.get('looks'):
            ext['EquivalentNumberOfLooks'] = f"{res['looks']:.2f}"
        kind = 'Multi-image incoherent average'
        if product == 'coherent_sum':
            kind = 'Multi-image coherent sum'
            ext['Combination'] = ('sum of the complex images, the second brought to the phase and amplitude of the '
                                  'first' + (', both formed from their common support' if select else ''))
            ext['SignalToNoiseGain'] = (f"{res['gain']['snr']:.2f} dB measured, {res['gain']['expected']:.2f} dB for "
                                        'the coherence')
        res['metadata'] = write_sidd(output, res, remap=remap, remap_args=remap_args, shadows_down=shadows_down,
                                     collections=[sms[k] for k in range(n)], product_type=kind, extensions=ext)
    elif output is not None:
        res['metadata'] = write_geotiff(output, res)
    for key in ('image', 'coherence', 'allowed'):
        if res.get(key) is not None:
            res[key] = _xp.host(res[key])
    if output is not None:
        res['path'] = output
        timing['write'] = time.perf_counter() - t0
    timing['total'] = time.perf_counter() - t_start
    timing['megapixels'] = mpx
    timing['megapixels_per_second'] = n * mpx / timing['total']
    res['timing'] = timing
    return res
