"""The slant plane of a spotlight image, and resampling between it and the ground plane.

    out = fastsar.form_cphd('scene_CPHD.cphd', axes='los', window=False)     # ground plane, axes along the line of sight
    slant = fastsar.planes.to_slant(out, spacing=(0.25, 0.30))               # azimuth, range (m)

Planes. FastSAR forms a spotlight image on a ground plane, where backprojection is exact for any track. The slant
plane is the plane through the scene center point (SCP) that holds the antenna's position and velocity at the
center of aperture (COA, the mid-aperture pulse): e2 along the line of sight from the antenna to the SCP, e1 = e2 x
normal along azimuth, the normal on the upward side. A pixel at azimuth a and range r of the slant image shows the
ground point that has, at the COA, the range and the range rate of the point SCP + a e1 + r e2 of the slant plane.
That is the projection of a SICD with Grid.Type PLANE and ImagePlane SLANT, so a SICD reader's own projection of a
slant product is the map used here (ground_to_slant, slant_to_ground: closed forms).

Baseband. Backprojection leaves each scatterer's image on a carrier, exp(j 4 pi fc (R - R0) / c) in the range R to
the COA antenna position. deramp removes it, exactly at every pixel: the image's spectrum is then centered at zero
on both axes, and a scatterer's phase is -4 pi fc / c times its range less the SCP's (SICD's convention, Sgn -1).

Resampling. On ground axes along the line of sight (form_cphd(axes='los')) the baseband spectrum is a
parallelogram: its extent along e1 (across the line of sight) is the azimuth band's alone, and each line along e1
is therefore band limited however the spectrum wraps along e2. to_slant interpolates along e1 first, to the
azimuth positions of the slant grid, which leaves a spectrum that is a rectangle in azimuth and ground range, then
along e2 to the slant ranges (resample.warp: FFT oversampling by two and a Kaiser sinc). sampling gives the
spacings this takes from the exact spectrum of the collection at points of the scene. to_ground is the reverse,
for complex images or, with a cubic kernel, detected ones.

Devices. deramp, to_slant and to_ground work where the image is: a NumPy array on the host, a CuPy array on its
GPU (kernels in double: resample_cuda), a JAX array on JAX's device (float32 throughout, as a TPU needs:
resample_jax), and the product holds an array of the same kind. The maps are computed on the host at the knots.
"""
import numpy as np

from . import _xp, resample

C = 299792458.0

__all__ = ['coa', 'slant_frame', 'ground_to_slant', 'slant_to_ground', 'sampling', 'support', 'deramp', 'to_slant',
           'to_ground']


def _unit(v):
    return v / np.linalg.norm(v)


def coa(meta, pulses=None):
    """Center of aperture: the mid-aperture pulse's time (s), and the antenna's position and velocity then (local
    frame), from the polynomial fit of position against time that products.sicd_meta writes as ARPPoly. pulses:
    (first, last + 1) of the pulses the image used (form_cphd's output entry), default all of them."""
    ant = 0.5 * (np.asarray(meta['tx'], np.float64) + np.asarray(meta['rcv'], np.float64))
    t = np.asarray(meta['tx_time'], np.float64)
    lo, hi = (0, len(t)) if pulses is None else pulses
    m, tc = (int(lo) + int(hi)) // 2, 0.5 * (t[0] + t[-1])
    fit = [np.polynomial.Polynomial.fit(t - tc, ant[:, k], int(min(5, len(t) - 1))) for k in range(3)]
    return float(t[m]), np.array([f(t[m] - tc) for f in fit]), np.array([f.deriv()(t[m] - tc) for f in fit])


def slant_frame(meta, scp, pulses=None):
    """The slant plane at the scene center point scp [3] (local): dict(scp, arp, vel, t (COA), r0 (range of the
    SCP), e1 (azimuth), e2 (range, the line of sight), normal (e1 x e2, upward), v1, v2 (the velocity along e1, e2)).
    pulses: as coa's."""
    t, arp, vel = coa(meta, pulses)
    scp = np.asarray(scp, np.float64)
    r0 = float(np.linalg.norm(scp - arp))
    e2 = (scp - arp) / r0
    n = _unit(np.cross(e2, vel))
    if n[2] < 0:
        n = -n
    e1 = np.cross(e2, n)
    return dict(scp=scp, arp=arp, vel=vel, t=t, r0=r0, e1=e1, e2=e2, normal=n, v1=float(vel @ e1), v2=float(vel @ e2))


def ground_to_slant(fr, g):
    """Slant-plane coordinates (a, r) (m from the SCP along azimuth and range) of points g [..., 3] (local): the
    point of the slant plane with the same range and range rate at the COA."""
    w = np.asarray(g, np.float64) - fr['arp']
    R, c = np.linalg.norm(w, axis=-1), w @ fr['vel']
    v1, v2 = fr['v1'], fr['v2']
    v = np.hypot(v1, v2)
    al = c / v                                                    # along the velocity
    pe = np.sqrt(np.maximum(R * R - al * al, 0.0)) * np.sign(v1)   # across it, on the SCP's side
    return (al * v1 - pe * v2) / v, (al * v2 + pe * v1) / v - fr['r0']


def slant_to_ground(fr, a, r, hz):
    """The points [..., 3] (local) at height z = hz (one value, or one per point) that slant-plane coordinates
    (a, r) show."""
    a, r, hz = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(r, np.float64), np.asarray(hz, np.float64))
    arp, vel = fr['arp'], fr['vel']
    q2 = fr['r0'] + r
    c = a * fr['v1'] + q2 * fr['v2']
    wz = hz - arp[2]
    vh = np.hypot(vel[0], vel[1])
    al = (c - wz * vel[2]) / vh
    s = np.sign((fr['scp'][0] - arp[0]) * -vel[1] + (fr['scp'][1] - arp[1]) * vel[0])
    pe = s * np.sqrt(np.maximum(a * a + q2 * q2 - wz * wz - al * al, 0.0))
    return np.stack([arp[0] + (al * vel[0] - pe * vel[1]) / vh, arp[1] + (al * vel[1] + pe * vel[0]) / vh,
                     hz], -1)


def _grid(out):
    nx, ny = np.shape(out['image'])[:2]
    return (nx, ny, np.asarray(out['origin'], np.float64), np.asarray(out['e1'], np.float64),
            np.asarray(out['e2'], np.float64), float(out['spx']), float(out['spy']))


def _scp(out):
    if out.get('scp') is not None:
        return np.asarray(out['scp'], np.float64)
    nx, ny, o, e1, e2, spx, spy = _grid(out)
    return o + (nx // 2) * spx * e1 + (ny // 2) * spy * e2            # the center pixel, as products.sicd_meta


def sampling(fr, ant, band, points, e1, e2):
    """The spectrum of the baseband ground image at points [n, 3] of the scene, from the collection itself: pulse
    p at frequency f contributes the spatial frequency (2 f / c) u_p - (2 fc / c) u_coa, u the horizontal part of
    the unit vector from the antenna to the point. ant [P, 3]: the pulses used; band: first and last frequency.

    -> dict of the largest full widths (cycles/m; twice the largest offset from zero) over the points: bx along e1;
    by along e2 of the image resampled to the slant azimuths (to_slant's second pass); ba, br of the slant image
    along azimuth and range; ba_min: the azimuth width at the first frequency (the rectangle inscribed in the
    slant spectrum, whose azimuth width grows with frequency)."""
    pts = np.atleast_2d(np.asarray(points, np.float64))
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    ant = np.asarray(ant, np.float64)
    ant = ant[np.unique(np.round(np.linspace(0, len(ant) - 1, 129)).astype(int))]
    u = pts[:, None] - ant[None]
    u /= np.linalg.norm(u, axis=-1, keepdims=True)                    # [n, p, 3]
    uc = pts - fr['arp']
    uc /= np.linalg.norm(uc, axis=-1, keepdims=True)
    f = np.array([band[0], band[1]], np.float64)
    K = (2 / C) * (f[None, None, :, None] * u[:, :, None, :] - f.mean() * uc[:, None, None, :])     # [n, p, 2, 3]
    kx, ky = K @ e1, K @ e2
    a, r = ground_to_slant(fr, pts)
    hz, h = pts[:, 2], 1.0
    ga = (slant_to_ground(fr, a + h, r, hz) - slant_to_ground(fr, a - h, r, hz)) / (2 * h)       # dg/da [n, 3]
    gr = (slant_to_ground(fr, a, r + h, hz) - slant_to_ground(fr, a, r - h, hz)) / (2 * h)
    gax, gay, grx, gry = (v[:, None, None] for v in (ga @ e1, ga @ e2, gr @ e1, gr @ e2))
    ka, kr = kx * gax + ky * gay, kx * grx + ky * gry
    full = lambda k: float(2 * np.abs(k).max())
    return dict(bx=full(kx), by=full(kx * (grx / gry) + ky), ba=full(ka), br=full(kr), ba_min=full(ka[:, :, 0]))


def _frame(out):
    return slant_frame(out['meta'], _scp(out), out.get('pulses'))


def _used(out):
    """The antenna positions [P, 3] (local) of the pulses an image used."""
    ant = 0.5 * (np.asarray(out['meta']['tx'], np.float64) + np.asarray(out['meta']['rcv'], np.float64))
    lo, hi = out.get('pulses') or (0, len(ant))
    return ant[int(lo):int(hi)]


def support(fr, ant, band, d1, d2):
    """The baseband spectrum at the scene center along two directions of the ground d1, d2 [3] (an image's axes;
    for a slant image the ground displacement of a meter of azimuth and of range): pulse p at frequency f lies at
    (2 f / c) u_p - (2 fc / c) u_coa, u the unit vector from the antenna to the scene center.
    -> dict(k1, k2: (least, greatest) spatial frequency along each (cycles/m), over the pulses and the band;
    k1_mid, k1_min: the same along d1 at the center frequency and at the first frequency alone)."""
    u = fr['scp'] - np.asarray(ant, np.float64)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    f = np.array([band[0], 0.5 * (band[0] + band[1]), band[1]], np.float64)
    K = (2 / C) * (f[None, :, None] * u[:, None, :] - f[1] * fr['e2'])                     # [p, 3, 3]
    k1, k2 = K @ np.asarray(d1, np.float64), K @ np.asarray(d2, np.float64)
    lim = lambda k: (float(k.min()), float(k.max()))
    return dict(k1=lim(k1), k2=lim(k2), k1_mid=lim(k1[:, 1]), k1_min=lim(k1[:, 0]))


def _slant_axes(fr, hz, h=1.0):
    """The ground displacement [3] of a meter of azimuth and of a meter of range at the scene center."""
    return ((slant_to_ground(fr, h, 0.0, hz) - slant_to_ground(fr, -h, 0.0, hz)) / (2 * h),
            (slant_to_ground(fr, 0.0, h, hz) - slant_to_ground(fr, 0.0, -h, hz)) / (2 * h))


def deramp(out, sign=-1):
    """Bring a ground-plane spotlight image to baseband, in place: multiplied by exp(sign j 4 pi fc (R - R0) / c),
    R each pixel's range to the COA antenna position, R0 the scene center's, fc the center frequency. out['baseband']
    records it (products.sicd_meta reads it); sign=+1 puts the carrier back. A JAX image is replaced in out by the
    new array (computed in float32, to under 1e-3 rad: resample_jax.deramp)."""
    from .patches import _rows
    if bool(out.get('baseband')) == (sign < 0):
        return out
    nx, ny, o, e1, e2, spx, spy = _grid(out)
    img = out['image']
    fr = _frame(out)
    k = sign * 4 * np.pi * 0.5 * (out['band'][0] + out['band'][1]) / C
    dev = resample._device(img)
    if dev is not None:
        if _xp.kind(img) == 'cupy':
            dev.deramp(img, o - fr['arp'], spx * e1, spy * e2, fr['r0'], k)
        else:
            out['image'] = dev.deramp(img, o - fr['scp'], spx * e1, spy * e2, fr['e2'], fr['r0'], k)
        out['baseband'] = sign < 0
        return out
    w = (o - fr['arp'])[:, None] + (np.arange(ny) * spy)[None] * e2[:, None]          # [3, ny]: row 0 less the antenna

    def work(sl):
        s = (np.arange(sl.start, sl.stop) * spx)[:, None]
        d = (w[0][None] + s * e1[0]) ** 2
        d += (w[1][None] + s * e1[1]) ** 2
        d += (w[2][None] + s * e1[2]) ** 2
        np.sqrt(d, out=d)
        d -= fr['r0']
        d *= k
        img[sl] *= np.exp(1j * d)
    _rows(work, nx, int(max(1, min(64, (1 << 17) // max(ny, 1)))))
    out['baseband'] = sign < 0
    return out


def _knots(n, step):
    return np.unique(np.round(np.linspace(0, n - 1, max(2, int(np.ceil((n - 1) / step)) + 1))))


def _solve(f, x, iterations=8, h=1.0):
    """Newton's iteration for f(x) = 0 on arrays, the derivative by central differences."""
    for _ in range(iterations):
        x = x - f(x) * (2 * h) / (f(x + h) - f(x - h))
    return x


def to_slant(out, spacing, bounds=None, *, margin=0, valid=None, taps=8, knots=32):
    """A ground-plane spotlight image on axes along the line of sight, resampled to the slant plane.

    out: form_cphd(axes='los')'s output, sampled as sampling asks (spx <= 1 / bx, spy <= 1 / by, with a few
    percent to spare); brought to baseband in place if it is not (deramp). spacing: (azimuth, range) of the slant
    grid (m). bounds: (a0, a1, r0, r1), the slant grid's extent in meters from the SCP; default the bounding
    rectangle of the ground image. margin: pixels at each edge of the ground image that are used but not shown
    (the interpolation is poor within some twenty pixels of where the data end). valid: a mask [nx, ny] of the
    ground pixels to show. A slant pixel whose ground point is outside the image less its margin, or not valid, is 0.
    taps: of the interpolation kernel. knots: spacing (pixels) of the points where the maps are computed exactly
    (bilinear between them).

    -> a product dict as form_cphd's, on the slant plane: image [na, nr] complex64 (azimuth, range), origin, e1
    (azimuth), e2 (range), spx, spy (azimuth and range spacing), plane 'slant', scp (the scene center point, the
    ground image's), scp_pixel, baseband True, support (the spectrum's limits along azimuth and range at the scene
    center, cycles/m: planes.support), bandwidth (along azimuth at the center frequency, and along range), and the
    ground image's other entries."""
    from .patches import _rows
    if out.get('plane', 'ground') != 'ground':
        raise ValueError('to_slant takes a ground-plane image')
    deramp(out)
    nx, ny, o, e1, e2, spx, spy = _grid(out)
    scp = _scp(out)
    fr = _frame(out)
    hz = float(o[2])
    if abs(e1[2]) > 1e-9 or abs(e2[2]) > 1e-9:
        raise ValueError('to_slant takes an image on a horizontal plane of the local frame')
    sa, sr = (float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
    mg = int(margin)
    if bounds is None:                                               # the ground rectangle's outline in the slant plane
        ti, tj = np.linspace(mg, nx - 1 - mg, 65), np.linspace(mg, ny - 1 - mg, 65)
        ij = np.concatenate([np.stack([ti, np.full(65, v)], -1) for v in (mg, ny - 1 - mg)]
                            + [np.stack([np.full(65, v), tj], -1) for v in (mg, nx - 1 - mg)])
        a, r = ground_to_slant(fr, o + ij[:, :1] * spx * e1 + ij[:, 1:] * spy * e2)
        bounds = (a.min(), a.max(), r.min(), r.max())
    a0, a1, r0, r1 = (float(v) for v in bounds)
    i0, j0 = int(np.ceil(-a0 / sa)), int(np.ceil(-r0 / sr))            # the SCP's pixel
    na, nr = i0 + int(np.ceil(a1 / sa)) + 1, j0 + int(np.ceil(r1 / sr)) + 1
    if na < 2 or nr < 2:
        raise ValueError(f'slant bounds {bounds} hold no image')
    ak, rk, yk = _knots(na, knots), _knots(nr, knots), _knots(ny, knots)
    # pass 1, along e1 at each ground column: the x index where the azimuth is that of slant row i
    xs, ys = (scp - o) @ e1, (scp - o) @ e2
    h = 1.0
    al = (ground_to_slant(fr, scp + h * e1)[0] - ground_to_slant(fr, scp - h * e1)[0]) / (2 * h)
    be = (ground_to_slant(fr, scp + h * e2)[0] - ground_to_slant(fr, scp - h * e2)[0]) / (2 * h)
    if abs(al) < 0.05:
        raise ValueError('the azimuth of the slant plane does not vary across the line of sight on the ground '
                         f'(d azimuth / d x = {al:.3f}): the squint is too large for this resampling')
    A, Y = np.meshgrid((ak - i0) * sa, yk * spy, indexing='ij')        # [nak, nyk]
    X = _solve(lambda x: ground_to_slant(fr, o + x[..., None] * e1 + Y[..., None] * e2)[0] - A,
               xs + (A - be * (Y - ys)) / al)
    pos0 = resample.KnotMap(yk, ak, np.ascontiguousarray(X.T) / spx, na)
    # pass 2, along e2 at each slant row: the y index of the ground point of slant pixel (i, j)
    G = slant_to_ground(fr, ((ak - i0) * sa)[:, None], ((rk - j0) * sr)[None], hz) - o       # [nak, nrk, 3]
    pos1 = resample.KnotMap(ak, rk, (G @ e2) / spy, nr)
    dev = resample._device(out['image'])
    masked = mg > 0 or valid is not None
    if dev is not None:
        posx = resample.KnotMap(ak, rk, (G @ e1) / spx, nr)
        mask = (posx, (mg, nx - 1 - mg), (mg, ny - 1 - mg), valid) if masked else None
        img = dev.warp(out['image'], pos0, pos1, (na, nr), taps=taps, mask=mask)
    else:
        img = resample.warp(out['image'], pos0, pos1, (na, nr), taps=taps)
    if masked and dev is None:
        posx = resample.KnotMap(ak, rk, (G @ e1) / spx, nr)
        v = None if valid is None else np.asarray(_xp.host(valid), bool)

        def mask(sl):
            x, y = posx(sl), pos1(sl)
            ok = (x >= mg) & (x <= nx - 1 - mg) & (y >= mg) & (y <= ny - 1 - mg)
            if v is not None:
                ok &= v[np.clip(np.rint(x), 0, nx - 1).astype(np.intp), np.clip(np.rint(y), 0, ny - 1).astype(np.intp)]
            img[sl][~ok] = 0
        _rows(mask, na, int(max(1, min(64, (1 << 17) // nr))))
    sup = support(fr, _used(out), out['band'], *_slant_axes(fr, hz))
    res = {k: v for k, v in out.items() if k not in ('image', 'valid')}
    res.update(image=img, origin=scp - i0 * sa * fr['e1'] - j0 * sr * fr['e2'], e1=fr['e1'], e2=fr['e2'], spx=sa, spy=sr,
               plane='slant', scp=scp, scp_pixel=(i0, j0), baseband=True, support=(sup['k1'], sup['k2']),
               bandwidth=(sup['k1_mid'][1] - sup['k1_mid'][0], sup['k2'][1] - sup['k2'][0]),
               ground=dict(origin=o, e1=e1, e2=e2, spx=spx, spy=spy, shape=(nx, ny), margin=mg))
    return res


def to_ground(out, origin, e1, e2, spacing, shape, *, kernel=None, taps=8, pad=None, knots=32):
    """A slant-plane image (to_slant's output, or one derived from it on the same grid: detected, apodized)
    resampled to a ground-plane grid: pixel (i, j) at origin + i spacing[0] e1 + j spacing[1] e2 (local frame, e1
    and e2 horizontal and orthonormal, e1 x e2 up), shape (nx, ny).

    kernel: 'sinc' (default for a complex image, which the ground grid must then sample as sampling asks, on axes
    along the line of sight), 'cubic' (default for a real one) or 'linear'; taps, pad: resample.lines's. A detected
    image is band limited as power (twice the complex image's band, so from a slant image at half the spacing):
    the sinc kernel then places a point within a few hundredths of a pixel, the cubic one within a quarter. Ground
    pixels the slant image does not cover are 0.
    -> the product dict on the ground plane (plane 'ground')."""
    if out.get('plane') != 'slant':
        raise ValueError('to_ground takes a slant-plane image')
    dev = resample._device(out['image'])
    img = out['image'] if dev is not None else np.asarray(out['image'])
    na, nr = img.shape
    o, e1, e2 = (np.asarray(v, np.float64) for v in (origin, e1, e2))
    spx, spy = (float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
    nx, ny = (int(v) for v in shape)
    scp = np.asarray(out['scp'], np.float64)
    fr = _frame(out)
    (i0, j0), sa, sr, hz = out['scp_pixel'], float(out['spx']), float(out['spy']), float(o[2])
    if kernel is None:
        kernel = 'cubic' if np.isrealobj(img) else 'sinc'
    ak, yk, xk = _knots(na, knots), _knots(ny, knots), _knots(nx, knots)
    # pass 1, along range at each slant row: the range index that shows ground column j
    ys = (scp - o) @ e2
    h = 1.0
    gr = (slant_to_ground(fr, 0.0, h, hz) - slant_to_ground(fr, 0.0, -h, hz)) / (2 * h)
    A, Y = np.meshgrid((ak - i0) * sa, yk * spy, indexing='ij')        # [nak, nyk]
    R = _solve(lambda r: (slant_to_ground(fr, A, r, hz) - o) @ e2 - Y, (Y - ys) / (gr @ e2))
    pos0 = resample.KnotMap(ak, yk, R / sr + j0, ny)
    # pass 2, along azimuth at each ground column: the azimuth index of ground pixel (i, j)
    g = o + (xk * spx)[None, :, None] * e1 + (yk * spy)[:, None, None] * e2                  # [nyk, nxk, 3]
    pos1 = resample.KnotMap(yk, xk, ground_to_slant(fr, g)[0] / sa + i0, nx)
    kw = dict(kernel=kernel, pad=pad) if kernel != 'sinc' else dict(kernel='sinc', taps=taps, pad=pad)
    res = {k: v for k, v in out.items() if k not in ('image', 'scp_pixel', 'ground', 'support', 'bandwidth')}
    img = dev.warp(img, pos0, pos1, (ny, nx), transposed=True, **kw) if dev is not None else \
        resample.warp(img.T, pos0, pos1, (ny, nx), **kw).T
    res.update(image=img, origin=o, e1=e1, e2=e2, spx=spx, spy=spy,
               plane='ground', scp=scp)
    return res
