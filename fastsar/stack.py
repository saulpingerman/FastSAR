"""Several collections of one scene: a common ground grid, registration, and the products of a stack.

    fastsar.process_stack(['a_CPHD.cphd', 'b_CPHD.cphd', 'c_CPHD.cphd'], 'mean_SIDD.nitf')      # speckle-reduced display
    fastsar.process_stack(['a_CPHD.cphd', 'b_CPHD.cphd'], 'coherence.tif', product='coherence')

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

Devices. Every function works where the images are (NumPy on all cores, CuPy, JAX) and returns arrays of the
same kind.
"""
import time

import numpy as np

from . import _xp, io, planes

C = 299792458.0

__all__ = ['ground_grid', 'as_grid', 'geometry', 'baseline', 'register', 'shift_image', 'mean', 'coherence',
           'change', 'process_stack']


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


def _common_aperture(ga, gb, point):
    """The pulses of two collections that look at a ground point (ECEF) from one interval of azimuth: the mean of
    the two mid-aperture directions, as wide as the narrower of the apertures an image of square resolution uses
    (_look). -> ((first, last + 1), (first, last + 1)), or None when either collection has less than half of its
    own such aperture inside the interval (the pair is then formed as each collection would be alone)."""
    lat, lon, _ = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(point))
    east, north, _ = _enu(lat, lon)
    phi, own = [], []
    for g in (ga, gb):
        u = point[None] - g['ant']
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        phi.append(np.unwrap(np.arctan2(u @ east, u @ north)))
        own.append(_look(dict(g, pulses=None), point)['pulses'])
    mid = [p[(lo + hi) // 2] for p, (lo, hi) in zip(phi, own)]
    phi[1] = phi[1] + np.round((mid[0] - mid[1]) / (2 * np.pi)) * 2 * np.pi
    mid[1] = phi[1][(own[1][0] + own[1][1]) // 2]
    c = 0.5 * (mid[0] + mid[1])
    w = min(abs(p[hi - 1] - p[lo]) for p, (lo, hi) in zip(phi, own))
    out = []
    for p, (lo, hi) in zip(phi, own):
        idx = np.nonzero(np.abs(p - c) <= w / 2)[0]
        if len(idx) < 0.5 * (hi - lo):
            return None
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return tuple(out)


def baseline(a, b, point=None, aperture='square'):
    """Whether two collections of one scene can be coherent, from their geometry alone.

    a, b: CPHD files (only their per-pulse parameters are read) or products of form_cphd. point: the ground point
    (ECEF, or (latitude, longitude, height)) the geometry is taken at; default the first's scene reference point.
    aperture: 'square' (the part of each aperture form_cphd uses without a spacing) or 'full'; a product's own
    pulses are used as they are.

    Two images share spectrum only where their ground wavenumbers coincide. In range these are 2 f cos(graze) / c,
    so a difference in grazing angle moves one band against the other and the bands part at the critical
    difference (B / fc) / tan(graze); in azimuth the directions the apertures look from must overlap, so they part
    when the difference in azimuth reaches the aperture's own angular extent on the ground.

    -> dict(d_graze, d_azimuth (degrees, b less a at mid-aperture), critical_graze, critical_azimuth (degrees),
    overlap_range, overlap_azimuth (the fraction of a's band that b shares, each way), overlap (their product),
    inside (both overlaps positive), graze, azimuth (of each), days (b's start less a's, where both state one))."""
    ga, gb = geometry(a), geometry(b)
    if point is None:
        p = ga['srp']
    else:
        p = np.ravel(np.asarray(point, np.float64))
        p = p if np.linalg.norm(p) > 1e6 else np.ravel(io.geodetic_to_ecf(p[0], p[1], p[2] if p.size == 3 else 0.0))
    la, lb = _look(ga, p, aperture), _look(gb, p, aperture)
    ov = _overlap(la, lb)
    fc = 0.25 * (la['band'][0] + la['band'][1] + lb['band'][0] + lb['band'][1])
    bw = min(la['band'][1] - la['band'][0], lb['band'][1] - lb['band'][0])
    psi = np.radians(0.5 * (la['graze'] + lb['graze']))
    days = None
    try:
        days = float((np.datetime64(str(gb['start']).rstrip('Z')) - np.datetime64(str(ga['start']).rstrip('Z')))
                     / np.timedelta64(1, 's')) / 86400.0
    except (ValueError, TypeError):
        pass
    return dict(d_graze=lb['graze'] - la['graze'], d_azimuth=(lb['azimuth'] - la['azimuth'] + 180.0) % 360.0 - 180.0,
                critical_graze=float(np.degrees(bw / fc / np.tan(psi))),
                critical_azimuth=min(la['azimuth_span'], lb['azimuth_span']),
                overlap_range=ov['range'], overlap_azimuth=ov['azimuth'], overlap=ov['range'] * ov['azimuth'],
                inside=bool(ov['range'] > 0 and ov['azimuth'] > 0), graze=(la['graze'], lb['graze']),
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


def register(products, reference=0, *, apply='fourier', max_shift=64, tiles=4, box=5, highpass=32.0, form=None):
    """Register images formed on one grid: one translation each against a reference.

    products: a list of ground-plane products on the same grid (form_cphd(grid=)); complex or detected images, on
    the host or a device. reference: its index. apply: 'fourier' (shift_image: a phase ramp on the spectrum),
    'reform' (the collection formed again on the displaced grid: form(k, shift) -> product, default
    fastsar.form_cphd(product['source'], grid=, shift=) with the product's own options where it records them), or
    None (estimate only). max_shift: the largest translation searched, pixels. tiles: the image is cut into tiles x
    tiles parts (of 48 pixels at least), each registered again after the translation. box: the side of the averaging
    before the correlation, pixels. highpass: structure of the detected images larger than this many pixels is left
    out of the correlation (None: kept).

    Each image gets the translation (d1, d2) at which it best matches the reference, image(x + d) ~ reference(x),
    from the peak of the cross-correlation of the detected images, refined to a few hundredths of a pixel. Two
    figures say how far to trust it. significance is the peak over the rms of the correlation surface (70 to 130 on
    six spotlights of 9 megapixels; correlation is the normalized value of the peak), and found is False when it is
    under 5 or the peak lies on the border of the search (the translation is then not applied). residual_median
    and residual_rms (m; residual_worst, tile_residuals [tiles^2, 2] along e1 and e2) are what the tiles still
    disagree by after the translation, over the tiles whose own peak is significant and lies within 8 pixels of
    the image's (tiles_used of tiles): a tenth of a pixel or two when one translation registers the pair; a pixel
    or more when it does not, which between different look directions is the layover of whatever stands above the
    grid plane and no warp of the ground removes (a tile of tall trees or buildings stands out in the rms).

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


def _coherence_map(a, b, w, fringe, tile):
    """The sample coherence of two complex images over w x w windows with a linear fringe removed: in tiles (each
    with the fringe at the peak of its interferogram's spectrum, estimated on the tile and applied to it and a
    halo of half a window), over the whole image ('global'), or not at all. -> (map float32, fringes [n, 2]
    cycles per pixel)."""
    xp = _xp.module(a)
    n1, n2 = a.shape

    def one(x, pa, pb):
        num = xp.abs(_box(x, w))
        den = xp.sqrt(_box(pa, w) * _box(pb, w))
        return xp.where(den > 0, num / xp.maximum(den, 1e-30), 0).astype(xp.float32)

    def peak(x):
        m = (512, 512) if max(x.shape) <= 512 else tuple(1 << int(np.ceil(np.log2(v))) for v in x.shape)
        F = xp.abs(_fft(x[:m[0], :m[1]], 'fft2', s=m))
        i, j = np.unravel_index(int(_xp.host(xp.argmax(F))), m)
        return np.fft.fftfreq(m[0])[i], np.fft.fftfreq(m[1])[j]

    def flat(x, f, i0, j0):
        r1 = xp.asarray(np.exp(-2j * np.pi * f[0] * (np.arange(x.shape[0]) + i0)).astype(np.complex64))[:, None]
        r2 = xp.asarray(np.exp(-2j * np.pi * f[1] * (np.arange(x.shape[1]) + j0)).astype(np.complex64))[None, :]
        return x * r1 * r2
    pa, pb = _power(a), _power(b)
    x = _rowwise(lambda u, v: u * xp.conj(v), a, b)
    if not fringe:
        return one(x, pa, pb), np.zeros((0, 2))
    if fringe == 'global' or max(n1, n2) <= tile:
        f = peak(x)
        return one(flat(x, f, 0, 0), pa, pb), np.array([f])
    h = w // 2 + 1
    rows, fr = [], []
    for i in range(0, n1, tile):
        row = []
        for j in range(0, n2, tile):
            i1, j1 = min(i + tile, n1), min(j + tile, n2)
            a0, b0, a1, b1 = max(i - h, 0), max(j - h, 0), min(i1 + h, n1), min(j1 + h, n2)
            f = peak(x[i:i1, j:j1])
            fr.append(f)
            sl = (slice(a0, a1), slice(b0, b1))
            g = one(flat(x[sl], f, 0, 0), pa[sl], pb[sl])
            row.append(g[i - a0:i - a0 + i1 - i, j - b0:j - b0 + j1 - j])
        rows.append(xp.concatenate(row, axis=1))
    return xp.concatenate(rows, axis=0), np.array(fr)


def _stats(g, valid=None):
    xp = _xp.module(g)
    v = g.ravel() if valid is None else g[_xp.host(valid)] if _xp.kind(g) == 'jax' else g[valid]
    if v.size == 0:
        return dict(median=None, mean=None, p90=None, above_0p5=None, above_0p7=None)
    med, p90 = (float(t) for t in _xp.host(xp.percentile(v, xp.asarray([50.0, 90.0], dtype=xp.float32))))
    return dict(median=med, mean=_scalar(v.mean()), p90=p90, above_0p5=_scalar((v > 0.5).mean()),
                above_0p7=_scalar((v > 0.7).mean()))


def coherence(a, b, window=7, *, common_band=True, fringe='tiles', tile=384, floor=True):
    """The coherence of a registered pair of complex images on one grid.

    a, b: ground-plane complex products on the same grid, registered (register); on the host or a device. window:
    the side of the estimation window, pixels. common_band: keep of each image only the part of its spectrum the
    other shares, since what one has alone can only lower the coherence: on the ground the two spectra are the
    sectors their pulses' directions and their bands span, displaced in range by a difference in grazing angle
    (the spectral shift) and in azimuth by a difference in look direction; each image is cut at baseband to the
    other's sector. fringe: 'tiles' (a linear fringe removed in each tile x tile pixels, at the peak of the tile's
    interferogram spectrum), 'global' (one for the image) or None. A residual baseline or registration error and a
    tilt of the ground against the grid plane leave a fringe that a window would otherwise average down.

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
    fringes [n, 2] (cycles per pixel), baseline (baseline(a, b)), timing."""
    t0 = time.perf_counter()
    for name, p in (('a', a), ('b', b)):
        if 'complex' not in str(p['image'].dtype) or p.get('plane', 'ground') != 'ground':
            raise ValueError(f'coherence takes complex ground-plane products; {name} is not one')
    if not _same_grid(_grid_of(a), _grid_of(b)):
        raise ValueError('coherence: the two products are not on one grid: form them with form_cphd(grid=)')
    if fringe not in ('tiles', 'global', None, False):
        raise ValueError(f"fringe must be 'tiles', 'global' or None, got {fringe!r}")
    xp = _xp.module(a['image'])
    notes = list(a.get('notes', []))
    overlap = None
    if common_band:
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
        za = _carrier(dict(a, image=za, baseband=True), on=True, copy=not own[0])
        zb = _carrier(dict(b, image=zb, baseband=True), on=True, copy=not own[1])
    else:
        za, zb = _carrier(a, on=True), _carrier(b, on=True)
    valid = (a['image'] != 0) & (b['image'] != 0)
    g, fr = _coherence_map(za, zb, int(window), fringe, int(tile))
    g = xp.where(valid, g, 0).astype(xp.float32)
    all_valid = bool(_xp.host(valid.all()))
    st = _stats(g, None if all_valid else valid)
    fl = samples = None
    if floor:
        s = 4 * int(window) + 1
        gf, _ = _coherence_map(za[s:, s:], zb[:-s, :-s], int(window), fringe, int(tile))
        fl = _stats(gf, None if all_valid else valid[s:, s:] & valid[:-s, :-s])
        samples = None if not fl['mean'] else float(np.pi / (4 * fl['mean'] ** 2))
        del gf
    res = {k: v for k, v in a.items() if k not in ('image', 'phase_error', 'registration', 'baseband', 'valid')}
    try:
        bl = baseline(a, b, point=io.local_to_ecf(planes._scp(a), a['meta']))
    except (KeyError, TypeError, ValueError):
        bl = None
    dt = time.perf_counter() - t0
    res.update(image=g, product='coherence', window=int(window), floor=fl, samples=samples, overlap=overlap,
               fringes=fr, baseline=bl, grid=_grid_of(a), notes=notes + [
                   f'coherence of two images over {int(window)} x {int(window)} pixels'
                   + (', on their common band' if overlap is not None and min(overlap) > 0.02 else '')],
               timing=dict(coherence=dt, megapixels_per_second=g.size / 1e6 / max(dt, 1e-9)), **st)
    return res


def change(a, b, window=7, **kw):
    """What changed between two registered images of one scene, two ways.

    a, b, window and the keywords: coherence's. Coherent change is a loss of coherence: the scatterers within a
    resolution cell moved by a fraction of the wavelength, or were replaced. Incoherent change is a change of
    brightness: the ratio of the intensities averaged over the window, in decibels, less its median over the image
    (the difference in gain between the dates).

    What decorrelates is not only what changed. Vegetation, water, snow and tilled or wet soil lose coherence
    within hours to days, and so does any pair outside the critical baseline (baseline) or less than well
    registered. Low coherence therefore means change only on surfaces that stay coherent, such as buildings,
    roads and bare rock, and only against the floor of the estimator (coherence): on the repeat-pass spotlights
    measured two weeks apart, fields were at the floor and only built surfaces above it. The intensity ratio does
    not need coherence, but its speckle is that of the window's looks: 10 log10 of a ratio of two averages over n
    independent samples scatters by about 6.1 / sqrt(n) dB.

    -> coherence's product dict with product 'change', image [nx, ny] float32 (the intensity ratio b over a, dB),
    coherence [nx, ny] float32, ratio_sigma (dB: the scatter of the ratio the window's independent samples give
    for unchanged ground, where the floor was measured), and coherence's entries."""
    res = coherence(a, b, window, **kw)
    t0 = time.perf_counter()
    xp = _xp.module(res['image'])
    pa, pb = _box(_power(a['image']), window), _box(_power(b['image']), window)
    valid = (pa > 0) & (pb > 0)
    r = _rowwise(lambda u, v: 10.0 * xp.log10(xp.maximum(v, 1e-30) / xp.maximum(u, 1e-30)), pa, pb)
    med = _percentile(xp.where(valid, r, 0), [50.0])[0]
    r = xp.where(valid, r - med, 0).astype(xp.float32)
    res.update(coherence=res['image'], image=r, product='change', ratio_median_db=med,
               ratio_sigma=None if not res.get('samples') else float(10 / np.log(10) * np.sqrt(2.0 / res['samples'])))
    res['notes'] = res['notes'] + ['change: the intensity ratio of the second image over the first (dB, median removed) '
                                   'and the coherence; low coherence is change only where the surface stays coherent']
    res['timing']['change'] = time.perf_counter() - t0
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
        import cupy
        dev = cupy.asarray(image)
        cupy.cuda.Stream.null.synchronize()                          # the former's page-locked image may be given up now
        return dev
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
    intensity ratio). The grid is written as it is, without resampling, with the affine transform that places its
    pixels on the map (rotated by the grid's azimuth; rows along e1, columns along e2), when that places every
    pixel within a tenth of a pixel of its true position, as it does for grids of a few kilometers; a larger grid
    is resampled north up (products.geocode_image). -> dict(crs, transform, resampled)."""
    from . import products, _deps
    _deps.require('rasterio')
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp
    if bands is None:
        bands = [out['coherence'], out['image']] if out.get('product') == 'change' else [out['image']]
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
                  remap='nrl', remap_args=None, shadows_down=True, backend='auto', precision='float32', max_shift=64,
                  range_gate=True, channel=0, troposphere=None, reader_options=None, target_db=-40.0):
    """Several collections of one scene, from the CPHD files to one product (see the module docstring).

    cphds: the files, two or more. output: the file to write, or None to return the product without writing it.
    product: 'mean' (the incoherent average of all of them, written as a SIDD: detected, remapped, shadows down;
    the speckle-reduced display product), 'coherence' or 'change' (of two collections, written as a GeoTIFF: the
    coherence; the coherence and the intensity ratio in decibels).
    grid: the common ground grid (stack.ground_grid, or a product: its pixels). None: the reference collection's
        tasked area on axes along its line of sight, so that its shadows fall down the rows, with square pixels of
        `spacing` (default: 0.7 of the coarsest collection's ground range resolution, finer where a collection's
        spectrum on those axes needs it) and sides `extent` (m, across and along that line of sight; default the
        tasked area's).
    reference: the index of the collection the others are registered to; the product has its geolocation.
    register: 'fourier' (one translation per image from the detected images' cross-correlation, applied on the
        spectrum: stack.register, stack.shift_image), 'reform' (applied by forming the collection again on the
        displaced grid, from the file: exact at the edges too, at the cost of a second read and formation) or None
        (the collections' own geolocation is trusted). max_shift: the largest translation searched, pixels.
    equalize, weights: stack.mean's (default: each image scaled to the reference's brightness; None keeps the
        images' own units). window: the coherence window, pixels.
    weighting: 'taylor' (nbar 4, 35 dB, on the phase history) or 'uniform'. aperture: form_cphd's; with 'auto' the
        two collections of a coherence or change product are formed from the pulses that look from one interval of
        azimuth (the mean of their directions, as wide as the narrower aperture of square resolution), where each
        has at least half of it: what one aperture has alone only lowers the coherence.
    remap, remap_args, shadows_down: the SIDD's (sidd.write_sidd).
    range_gate: form_cphd's; True by default here: a stack's grid is often small against the collections' swaths,
        and a gated first level is several times faster.
    backend, precision, channel, troposphere, reader_options, target_db: form_cphd's.

    The collections are read and formed one at a time (the reference first), so that one phase history is in
    memory at a time, and each image goes to the backend's device for registration and the product, or stays on
    the host when the device lacks the room (a note says so).

    -> the product dict of stack.mean, stack.coherence or stack.change (image on the host), with registrations (one
    dict per collection, in the order given: stack.register's), inputs (per collection: path, name, collector,
    start, graze, azimuth, pulses, shift, notes), timing (seconds: read, form, register, combine, write, total) and,
    when written, path and metadata (the SIDD's, or the GeoTIFF's crs and transform). gains and looks_single are
    per collection in the order given; looks_by_n counts the images as processed, the reference first.

    The SIDD of a mean states what it is: ProductClass 'Detected Image', ProductType 'Multi-image incoherent
    average', the number of collections, the equivalent number of looks and the registration as
    ProductCreationExtension parameters, one ExploitationFeatures Collection per input with its own date, time and
    geometry, and the reference collection's aperture and time for the projection (Measurement)."""
    from .api import _backend
    from .cphd import form_cphd
    from . import products as _products
    if product not in ('mean', 'coherence', 'change'):
        raise ValueError(f"product must be 'mean', 'coherence' or 'change', got {product!r}")
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
    backend = _backend(backend)
    t_start = time.perf_counter()
    timing = dict(read=0.0, form=0.0, register=0.0, combine=0.0, write=0.0)
    notes = []
    if grid is None:
        try:
            geos = [geometry(p, channel) for p in paths]
        except Exception as e:
            raise ValueError(f'the default grid is planned from the per-pulse parameters of CPHD files ({e}): '
                             'give grid= (fastsar.stack.ground_grid)') from e
        g = _default_grid(paths[reference], geos, reference, spacing, extent)
    else:
        g = as_grid(grid)
    timing['plan'] = time.perf_counter() - t_start
    mpx = g['shape'][0] * g['shape'][1] / 1e6

    def sync(image=None):
        if backend == 'cuda':
            import cupy
            cupy.cuda.Stream.null.synchronize()
        elif _xp.kind(image) == 'jax':
            image.block_until_ready()

    pulses = {}
    if product != 'mean' and aperture == 'auto':                     # the pair on one interval of azimuth
        try:
            common = _common_aperture(geometry(paths[0], channel), geometry(paths[1], channel), g['center'])
        except Exception:                                            # not CPHD files: each collection's own aperture
            common = None
        if common is not None:
            pulses = dict(enumerate(common))
            notes.append('the two collections were formed from the pulses on one interval of azimuth')

    def form(k, shift=None):
        t0 = time.perf_counter()
        mark = {}
        out = form_cphd(paths[k], grid=g, shift=shift, backend=backend, window=weighting == 'taylor',
                        aperture=pulses.get(k, aperture), precision=precision, channel=channel, troposphere=troposphere,
                        reader_options=reader_options, target_db=target_db, range_gate=range_gate,
                        _plan=lambda q: mark.update(read=time.perf_counter()))
        t1 = time.perf_counter()
        timing['read'] += mark['read'] - t0
        timing['form'] += t1 - mark['read']
        out['image'] = _on_device(out['image'], backend, notes)
        return out

    order = [reference] + [k for k in range(n) if k != reference]
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
                         pulses=tuple(out['pulses']), shift=tuple(out['shift']), notes=list(out['notes']))
        if product == 'mean':
            if output is not None:                                   # each input's own collection metadata
                sms[k] = _products.sicd_meta(dict(out, image=np.broadcast_to(np.complex64(0), g['shape'])))[0]
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
        res = (coherence if product == 'coherence' else change)(a, b, window)
        res['inputs'] = inputs
        del a, b, kept
    sync(res['image'])
    timing['combine'] += time.perf_counter() - t0
    res.update(registrations=regs, reference=reference, notes=list(res.get('notes', [])) + notes)
    if product == 'mean':
        res.update(remap=remap, remap_args=dict(remap_args or {}), shadows_down=bool(shadows_down))
    t0 = time.perf_counter()
    if output is not None and product == 'mean':
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
        res['metadata'] = write_sidd(output, res, remap=remap, remap_args=remap_args, shadows_down=shadows_down,
                                     collections=[sms[k] for k in range(n)],
                                     product_type='Multi-image incoherent average', extensions=ext)
    elif output is not None:
        res['metadata'] = write_geotiff(output, res)
    for key in ('image', 'coherence'):
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
