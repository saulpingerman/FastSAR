"""Products from formed images: multilooked detection, interferograms and coherence (interferometry and coherent change
detection), range-Doppler projection of 3-D points into the image (geocoding and terrain correction), latitude and
longitude of pixels, map-projected GeoTIFF and SICD output.

    amp_db = products.to_db(products.multilook(img, 2, 2))
    vals = products.geocode(img, grid, ant, points)           # image sampled at 3-D points (e.g. a map grid on a DEM)
    products.write_sicd('out.nitf', img, 'vendor_SICD.nitf')  # image formed on the vendor's grid by read_cphd

    out = fastsar.form_cphd('scene_CPHD.cphd')                # from form_cphd's output:
    lat, lon, h = products.geolocate(out, i, j, height=0.0)   # where pixels (i, j) are on the ground
    i, j = products.locate(out, lat, lon, h)                  # and where ground points appear in the image
    products.write_geotiff('pow.tif', **products.geocode_image(out, products.multilook(out['image'], 2, 2)))
    products.write_sicd('out.nitf', out)                      # complex image with its geometry
    dem = products.read_dem('dem.tif', offset=geoid)          # terrain for geolocate and geocode_image (height=dem)

grid is the dict of nx, ny, spx, spy, e1, e2 the image was formed on (read_cphd's output carries them), and ant the
antenna positions in the same local frame. A point x off the image plane appears where the plane has the same range
and the same Doppler cone angle at the aperture center: |p - a_c| = |x - a_c| and (p - a_c) . v = (x - a_c) . v,
with a_c and v the mid-aperture antenna position and the direction of motion. project solves the two conditions in
closed form (a line in the plane intersected with a circle), so terrain correction is exact for that model; for
apertures much longer than the image is wide it holds to second order in the angle the aperture subtends. For a
straight track the conditions do not depend on which point of the track is a_c, so the model also holds for the
moving-beam mosaics of form_cphd; for an orbit it holds as long as the track over the image stays close to a line.

Amplitude. form_image and form_cphd sum the phase history without normalization: a point scatterer whose samples
have amplitude a peaks at a sum(w), and noise of power s^2 per sample gives a mean pixel power s^2 sum(w^2), with w
the window over pulses times the window over frequency (both ones for window=False). CPHD carries no radiometric
calibration, so the pixels are in the phase history's units; beta0 or sigma0 need the vendor's calibration constant.
"""
import numpy as np


def multilook(img, la=1, lr=1):
    """Power |img|^2 averaged over la x lr pixel boxes (axis 0, axis 1) and decimated by them."""
    p = np.abs(np.asarray(img)) ** 2
    nx, ny = (p.shape[0] // la) * la, (p.shape[1] // lr) * lr
    return p[:nx, :ny].reshape(nx // la, la, ny // lr, lr).mean((1, 3))


def _box(x, w1, w2):
    """Moving average over w1 x w2 windows, same shape (edges average what is there)."""
    from scipy.ndimage import uniform_filter
    f = lambda v: uniform_filter(v, (w1, w2), mode='nearest')
    return f(x.real) + 1j * f(x.imag) if np.iscomplexobj(x) else f(x)


def interferogram(a, b, la=1, lr=1):
    """Multilooked interferogram: a conj(b) averaged over la x lr boxes and decimated, complex (np.angle is the
    interferometric phase). a and b must be formed on the same grid (coregistered), which backprojection onto one
    set of points or one planar grid gives directly."""
    x = np.asarray(a) * np.conj(np.asarray(b))
    nx, ny = (x.shape[0] // la) * la, (x.shape[1] // lr) * lr
    return x[:nx, :ny].reshape(nx // la, la, ny // lr, lr).mean((1, 3))


def coherence(a, b, w1=5, w2=5):
    """Sample coherence |<a b*>| / sqrt(<|a|^2> <|b|^2>) over moving w1 x w2 windows, same shape as the images:
    the coherent change detection statistic (low where the scene changed between passes). It is biased upward for
    small windows, to about 1/sqrt(w1 w2) for incoherent pairs."""
    a, b = np.asarray(a).astype(np.complex128), np.asarray(b).astype(np.complex128)
    num = np.abs(_box(a * np.conj(b), w1, w2))
    den = np.sqrt(_box(np.abs(a) ** 2, w1, w2) * _box(np.abs(b) ** 2, w1, w2))
    return num / np.maximum(den, 1e-300)


def pauli(hh, hv, vv, vh=None, la=1, lr=1):
    """Pauli decomposition of a quad-polarization image set (formed on one grid): the powers of
    (HH - VV)/sqrt(2) (double bounce), 2 HV/sqrt(2) with HV the mean of HV and VH when both are given (volume), and
    (HH + VV)/sqrt(2) (surface), multilooked over la x lr boxes; stacked [..., 3] in the usual red, green, blue order."""
    x = np.asarray(hv) if vh is None else 0.5 * (np.asarray(hv) + np.asarray(vh))
    k = [(np.asarray(hh) - np.asarray(vv)) / np.sqrt(2), np.sqrt(2) * x, (np.asarray(hh) + np.asarray(vv)) / np.sqrt(2)]
    return np.stack([multilook(c, la, lr) for c in k], -1)


def to_db(power, floor=1e-30):
    return 10 * np.log10(np.maximum(power, floor))


def _aperture(ant):
    ant = np.asarray(ant, np.float64)
    m = len(ant) // 2
    v = ant[min(m + 1, len(ant) - 1)] - ant[max(m - 1, 0)]
    return ant[m], v / np.linalg.norm(v)


def project(points, ant, nx, ny, spx, spy, e1, e2, **_):
    """Fractional pixel coordinates (i, j) [..., 2] at which points [..., 3] appear in an image formed on the grid
    (nx, ny, spx, spy, e1, e2); NaN where no plane point has the point's range and Doppler."""
    a, v = _aperture(ant)
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    x = np.asarray(points, np.float64)
    R2 = ((x - a) ** 2).sum(-1)
    D = (x - a) @ v
    # p = s e1 + t e2: s (e1.v) + t (e2.v) = D + a.v defines a line p0 + l d in the plane
    g1, g2 = e1 @ v, e2 @ v
    nrm = np.hypot(g1, g2)
    d = (-g2 * e1 + g1 * e2) / nrm
    p0 = ((D + a @ v) / nrm ** 2)[..., None] * (g1 * e1 + g2 * e2)
    # |p0 + l d - a|^2 = R2, with d a unit vector in the plane: l^2 + 2 l (p0 - a).d + |p0 - a|^2 - R2 = 0
    w = p0 - a
    b = w @ d
    c = (w * w).sum(-1) - R2
    disc = b * b - c
    sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
    # the root on the far side of the antenna's footprint on the plane, nearest to the point's own projection
    l1, l2 = -b - sq, -b + sq
    lx = (x - p0) @ d
    l = np.where(np.abs(l1 - lx) < np.abs(l2 - lx), l1, l2)
    p = p0 + l[..., None] * d
    return np.stack([p @ e1 / spx + nx / 2.0, p @ e2 / spy + ny / 2.0], -1)


def sample(img, ij, order=1):
    """img at fractional pixel coordinates ij [..., 2]: bilinear (order=1) or nearest (0); NaN outside."""
    img = np.asarray(img)
    i, j = ij[..., 0], ij[..., 1]
    ok = (i >= 0) & (j >= 0) & (i <= img.shape[0] - 1) & (j <= img.shape[1] - 1)
    i, j = np.where(ok, i, 0), np.where(ok, j, 0)
    if order == 0:
        out = img[np.rint(i).astype(int), np.rint(j).astype(int)]
    else:
        i0 = np.minimum(np.floor(i).astype(int), img.shape[0] - 2)
        j0 = np.minimum(np.floor(j).astype(int), img.shape[1] - 2)
        fi, fj = i - i0, j - j0
        out = ((1 - fi) * (1 - fj) * img[i0, j0] + fi * (1 - fj) * img[i0 + 1, j0]
               + (1 - fi) * fj * img[i0, j0 + 1] + fi * fj * img[i0 + 1, j0 + 1])
    return np.where(ok, out, np.nan)


def geocode(img, grid, ant, points, order=1):
    """img (any real or complex array on the grid, e.g. amplitude or multilooked power at full resolution) at the
    3-D points [..., 3] of a map grid (local frame of the image; see io.geodetic_to_ecf and io.ecf_to_local), with
    terrain correction through the points' heights."""
    return sample(img, project(points, ant, **grid), order)


def write_geotiff(path, data, transform, crs='EPSG:4326', nodata=np.nan):
    """GeoTIFF of data [rows, cols] or [bands, rows, cols], float32 or, for complex data, complex64 (GDAL CFloat32),
    with an affine transform (rasterio.Affine or its 6 coefficients a, b, c, d, e, f: x = a col + b row + c,
    y = d col + e row + f) in crs. write_geotiff(path, **geocode_image(...)) writes a geocoded image. Needs rasterio."""
    import rasterio
    from rasterio.transform import Affine
    tr = transform if isinstance(transform, Affine) else Affine(*transform)
    data = np.asarray(data)
    data = data.astype(np.complex64 if np.iscomplexobj(data) else np.float32)
    data = data[None] if data.ndim == 2 else data
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2], count=data.shape[0],
                       dtype=data.dtype.name, crs=crs, transform=tr, nodata=nodata, compress='deflate') as dst:
        dst.write(data)


def write_sicd(path, img, template=None, transpose=None):
    """Write a complex image as a SICD. Needs sarpy.

    write_sicd(path, out): form_cphd's output, with metadata built from it (sicd_meta).
    write_sicd(path, img, template): img formed on the grid of the template SICD (path or sarpy SICDType;
    read_cphd(..., sicd=template)), with the template's metadata; transpose as meta['sicd_transpose'] (default:
    when the template's rows run along range). ImageFormAlgo becomes OTHER and Grid.Type PLANE: the pixels are
    backprojection's on the template's image plane, which keep each scatterer's phase, not polar format's."""
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.complex.sicd import SICDWriter
    if template is None:
        sm, a = sicd_meta(img)
    else:
        sm = template if not isinstance(template, str) else open_complex(template).sicd_meta
        sm = sm.copy()
        rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
        a = np.asarray(img)
        if transpose is None:
            transpose = a.shape == (cols, rows) and rows != cols
        a = a.T if transpose else a
        if a.shape != (rows, cols):
            raise ValueError(f'image {a.shape} does not match the template grid {(rows, cols)}')
        sm.ImageFormation.ImageFormAlgo = 'OTHER'
        sm.PFA = None
        sm.Grid.Type = 'PLANE'          # pixels on a plane, linear in its coordinates: the grid backprojection used
        sm.ImageData.PixelType = 'RE32F_IM32F'
    with SICDWriter(path, sm, check_existence=False) as w:
        w.write_chip(np.asarray(a, np.complex64), start_indices=(0, 0))
    return sm


def sicd_meta(out):
    """SICD metadata (sarpy SICDType) for a form_cphd output, and its image arranged as the SICD array [rows, cols].

    Rows run along range away from the radar (e2 or -e2) and columns along e1 (the direction of motion); the
    scene center point is the grid's center pixel. Grid.Type PLANE on the ground plane of the grid, ImageFormAlgo
    OTHER, the aperture position as a polynomial in time fitted to the pulses (degree 5 at most), and one center of
    aperture time for the whole image (the mid-aperture pulse's): SICD's projection of a pixel is then the
    range-Doppler model of geolocate and locate. Impulse response bandwidths and widths are nominal (the data's
    spatial frequency support, and the 3 dB width that support and the window give), and the image keeps
    backprojection's phase: the band of each axis is centered at its spatial frequency, KCtr, modulo the sampling
    rate. There is no radiometric calibration (see the module docstring)."""
    from datetime import datetime, timezone
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType, RadarModeType
    from sarpy.io.complex.sicd_elements.ImageCreation import ImageCreationType
    from sarpy.io.complex.sicd_elements.ImageData import ImageDataType
    from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
    from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType, WgtTypeType
    from sarpy.io.complex.sicd_elements.Timeline import TimelineType
    from sarpy.io.complex.sicd_elements.Position import PositionType, XYZPolyType
    from sarpy.io.complex.sicd_elements.RadarCollection import RadarCollectionType, TxFrequencyType, ChanParametersType
    from sarpy.io.complex.sicd_elements.ImageFormation import ImageFormationType, RcvChanProcType, TxFrequencyProcType
    from .io import local_to_ecf
    from . import __version__
    C = 299792458.0
    meta = out['meta']
    img = np.asarray(out['image'])
    nx, ny = img.shape
    R = np.asarray(meta['R'], np.float64)
    e1, e2 = np.asarray(out['e1'], np.float64), np.asarray(out['e2'], np.float64)
    i0, j0 = nx // 2, ny // 2
    scp_l = np.asarray(out['origin'], np.float64) + i0 * out['spx'] * e1 + j0 * out['spy'] * e2
    ant = 0.5 * (np.asarray(meta['tx'], np.float64) + np.asarray(meta['rcv'], np.float64))
    t = np.asarray(meta['tx_time'], np.float64)
    m = len(t) // 2
    away = (scp_l - ant[m]) @ e2 > 0               # e2 points away from the radar: rows follow it
    a = img.T if away else img.T[::-1]
    srow = 1 if away else -1
    urow, ucol = srow * e2 @ R, e1 @ R
    scp = local_to_ecf(scp_l, meta)
    # aperture position against time
    deg = int(min(5, len(t) - 1))
    tc = 0.5 * (t[0] + t[-1])
    apc = local_to_ecf(ant, meta)
    coef = [np.polynomial.Polynomial.fit(t - tc, apc[:, k], deg).convert().coef for k in range(3)]
    shift = np.polynomial.Polynomial([-tc, 1.0])                      # poly(t - tc) -> poly in t
    arp = XYZPolyType(*[np.polynomial.Polynomial(cf)(shift).coef for cf in coef])
    # spatial frequencies: (2 / lambda) along the line of sight from the mid-aperture antenna, projected on the axes
    f0, f1 = out['band']
    los = (scp - apc[m]) / np.linalg.norm(scp - apc[m])
    kr, kc = (2 * 0.5 * (f0 + f1) / C * (los @ u) for u in (urow, ucol))
    bw_c, bw_r = out['bandwidth']
    win = out.get('window') or (None, None)
    width = {'taylor': 1.184, 'hann': 1.441, None: 0.886}           # 3 dB width times bandwidth

    def axis(u, ss, bw, kctr, w):
        bw = min(bw, 1.0 / ss)
        wgt = {None: WgtTypeType(WindowName='UNIFORM'), 'hann': WgtTypeType(WindowName='HANNING'),
               'taylor': WgtTypeType(WindowName='TAYLOR', Parameters={'NBAR': '4', 'SLL': '-35'})}[w]
        dk = (kctr * ss + 0.5) % 1.0 / ss - 0.5 / ss              # the band's center in the sampled image
        return DirParamType(UVectECF=u, SS=ss, ImpRespWid=width[w] / bw, Sgn=-1, ImpRespBW=bw, KCtr=kctr,
                            DeltaK1=dk - bw / 2, DeltaK2=dk + bw / 2, DeltaKCOAPoly=[[dk]], WgtType=wgt)
    pol = meta.get('polarization')
    pol = f'{pol[0]}:{pol[1]}' if pol else 'UNKNOWN'
    mode = str(meta.get('mode') or '').upper()
    if mode not in ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP'):
        mode = 'SPOTLIGHT' if out.get('mode') == 'spotlight' else 'DYNAMIC STRIPMAP'
    sm = SICDType(
        CollectionInfo=CollectionInfoType(CollectorName=meta.get('collector') or 'UNKNOWN',
                                          CoreName=meta.get('core_name') or 'UNKNOWN', CollectType='MONOSTATIC',
                                          RadarMode=RadarModeType(ModeType=mode), Classification='UNCLASSIFIED'),
        ImageCreation=ImageCreationType(Application=f'FastSAR {__version__}',
                                        DateTime=np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None), 'us')),
        ImageData=ImageDataType(PixelType='RE32F_IM32F', NumRows=ny, NumCols=nx, FirstRow=0, FirstCol=0,
                                FullImage=(ny, nx), SCPPixel=(j0 if away else ny - 1 - j0, i0)),
        GeoData=GeoDataType(EarthModel='WGS_84', SCP=SCPType(ECF=scp)),
        Grid=GridType(ImagePlane='GROUND', Type='PLANE', TimeCOAPoly=[[float(t[m])]],
                      Row=axis(urow, out['spy'], bw_r, kr, win[1]), Col=axis(ucol, out['spx'], bw_c, kc, win[0])),
        Timeline=TimelineType(CollectStart=np.datetime64(str(meta.get('start')).rstrip('Z'), 'us'),
                              CollectDuration=float(t[-1])),
        Position=PositionType(ARPPoly=arp),
        RadarCollection=RadarCollectionType(TxFrequency=TxFrequencyType(Min=f0, Max=f1),
                                            TxPolarization=pol[0] if pol != 'UNKNOWN' else 'OTHER',
                                            RcvChannels=[ChanParametersType(TxRcvPolarization=pol, index=1)]),
        ImageFormation=ImageFormationType(RcvChanProc=RcvChanProcType(NumChanProc=1, ChanIndices=[1]),
                                          TxRcvPolarizationProc=pol, TStartProc=float(t[0]), TEndProc=float(t[-1]),
                                          TxFrequencyProc=TxFrequencyProcType(MinProc=f0, MaxProc=f1),
                                          ImageFormAlgo='OTHER', STBeamComp='NO', ImageBeamComp='NO',
                                          AzAutofocus='GLOBAL' if out.get('phase_error') is not None else 'NO',
                                          RgAutofocus='NO'))
    sm.derive()
    sm.define_geo_image_corners(override=True)
    return sm, a


def _plane(out):
    """Grid center c (local), the grid dict project takes (centered on c) and the antenna phase centers relative to c,
    of a form_cphd output."""
    nx, ny = np.shape(out['image'])
    e1, e2 = np.asarray(out['e1'], np.float64), np.asarray(out['e2'], np.float64)
    c = np.asarray(out['origin'], np.float64) + nx / 2.0 * out['spx'] * e1 + ny / 2.0 * out['spy'] * e2
    ant = 0.5 * (np.asarray(out['meta']['tx'], np.float64) + np.asarray(out['meta']['rcv'], np.float64)) - c
    return c, dict(nx=nx, ny=ny, spx=out['spx'], spy=out['spy'], e1=e1, e2=e2), ant


def _llh(x, meta):
    from .io import local_to_ecf, ecf_to_geodetic
    return ecf_to_geodetic(local_to_ecf(x, meta))


def _height(height, lat, lon):
    return np.broadcast_to(height(lat, lon) if callable(height) else height, np.shape(lat)).astype(np.float64)


def locate(out, lat, lon, height=0.0):
    """Fractional pixel coordinates (i, j) [..., 2] at which the ground points (lat, lon in degrees, height above
    the WGS-84 ellipsoid in m) appear in a form_cphd image (range-Doppler projection onto its plane, layover
    included)."""
    from .io import geodetic_to_ecf, ecf_to_local
    c, grid, ant = _plane(out)
    lat, lon, h = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (lat, lon, height)))
    return project(ecf_to_local(geodetic_to_ecf(lat, lon, h), out['meta']) - c, ant, **grid)


def geolocate(out, i, j, height=None, iterations=10, tol=1e-4):
    """Latitude, longitude (degrees) and height above the WGS-84 ellipsoid (m) of pixels (i, j) (fractional, any
    shape) of a form_cphd image. height None: the pixel's own position on the image plane. A number, or a function
    dem(lat, lon) -> height (m) of arrays: the point on that surface which appears at the pixel, found on the circle
    of the pixel's range and Doppler cone angle at the aperture center (the inverse of locate) by Newton iteration
    on the circle's angle."""
    c, grid, ant = _plane(out)
    i, j = np.broadcast_arrays(np.asarray(i, np.float64), np.asarray(j, np.float64))
    p = (c + (i - grid['nx'] / 2.0)[..., None] * grid['spx'] * grid['e1']
         + (j - grid['ny'] / 2.0)[..., None] * grid['spy'] * grid['e2'])
    if height is None:
        return _llh(p, out['meta'])
    a, v = _aperture(ant)
    a = a + c
    q = a + ((p - a) @ v)[..., None] * v                       # circle center on the track's line, radius rho
    r = p - q
    rho = np.linalg.norm(r, axis=-1, keepdims=True)
    u = r / rho
    w = np.cross(v, u)
    x = lambda t: q + rho * (np.cos(t)[..., None] * u + np.sin(t)[..., None] * w)

    def f(t):
        la, lo, h = _llh(x(t), out['meta'])
        return h - _height(height, la, lo), h
    t, dt = np.zeros(i.shape), 1e-6
    for _ in range(iterations):
        (f0, h0), (f1, h1) = f(t), f(t + dt)
        # where the terrain's slope along the circle cancels the circle's own climb (steep walls facing the radar,
        # layover), Newton's derivative vanishes: step with the climb alone there, at most 100 m along the circle
        d, g = (f1 - f0) / dt, (h1 - h0) / dt
        d = np.where(np.isfinite(d) & (np.abs(d) > 0.1 * np.abs(g)), d, g)
        step = np.clip(f0 / d, -100.0 / rho[..., 0], 100.0 / rho[..., 0])
        t = t - step
        if np.max(np.abs(step) * rho[..., 0]) < tol:
            break
    return _llh(x(t), out['meta'])


def _utm(lat, lon):
    return f'EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) % 60 + 1}'


def geocode_image(out, data=None, spacing=None, crs=None, height=None, order=1):
    """Resample a form_cphd image, or data derived from it, onto a north-up map grid with terrain correction.

    data: an array on the image grid [nx, ny] (default |image|), or multilooked from it by integer factors
    (multilook(img, la, lr): its pixel (k, l) is the mean over pixels k la .. k la + la - 1 and l lr ...); real or
    complex. crs: the map's coordinate reference system, default the UTM zone of the scene center; 'EPSG:4326' for
    longitude and latitude. spacing: the map's pixel size in crs units, one number or (x, y); default the data's
    coarser pixel spacing (for EPSG:4326 converted to degrees of longitude and latitude at the scene). height:
    the surface (m above the WGS-84 ellipsoid) the map lies on: a number, a function dem(lat, lon) -> height
    (a DEM), or None for the ellipsoid height of the image center. order: 1 bilinear, 0 nearest. Every map pixel
    center is projected through its range and Doppler onto the image plane (locate), so scatterers on the surface
    land at their true map position; NaN where the image has no data. Other crs than EPSG:4326 need rasterio.

    -> dict(data [rows, cols] (row 0 north), transform (rasterio Affine, or its 6 coefficients without rasterio;
    pixel corners), crs), the arguments of write_geotiff."""
    img = np.abs(out['image']) if data is None else np.asarray(data)
    nx, ny = np.shape(out['image'])
    l1, l2 = nx // img.shape[0], ny // img.shape[1]
    if height is None:
        height = float(geolocate(out, nx / 2.0, ny / 2.0)[2])
    lat0, lon0, _ = (float(v) for v in geolocate(out, nx / 2.0, ny / 2.0))
    crs = _utm(lat0, lon0) if crs is None else crs
    geo = str(crs).upper() in ('EPSG:4326', 'OGC:CRS84')
    if geo:
        fwd = inv = lambda xs, ys: (np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    else:
        from rasterio.warp import transform as warp
        fwd = lambda lon, lat: tuple(np.asarray(v) for v in warp('EPSG:4326', crs, np.ravel(lon), np.ravel(lat)))
        inv = lambda xs, ys: tuple(np.asarray(v) for v in warp(crs, 'EPSG:4326', np.ravel(xs), np.ravel(ys)))
    # map extent: the image's edges, on the plane and on the surface
    a, b = np.linspace(-0.5, nx - 0.5, 33), np.linspace(-0.5, ny - 0.5, 33)
    I = np.concatenate([a, np.full(33, nx - 0.5), a, np.full(33, -0.5)])
    J = np.concatenate([np.full(33, -0.5), b, np.full(33, ny - 0.5), b])
    ll = [geolocate(out, I, J), geolocate(out, I, J, height=height)]
    xs, ys = (np.concatenate(v) for v in zip(*(fwd(g[1], g[0]) for g in ll)))
    ok = np.isfinite(xs) & np.isfinite(ys)                         # edge pixels off the DEM (NaN) do not set the extent
    xs, ys = xs[ok], ys[ok]
    if spacing is None:
        d = max(out['spx'] * l1, out['spy'] * l2)
        spacing = (d / (111320.0 * np.cos(np.radians(lat0))), d / 111132.0) if geo else d
    dx, dy = (spacing, spacing) if np.isscalar(spacing) else spacing
    x0, y1 = np.floor(xs.min() / dx) * dx, np.ceil(ys.max() / dy) * dy
    ncol, nrow = int(np.ceil((xs.max() - x0) / dx)), int(np.ceil((y1 - ys.min()) / dy))
    res = np.full((nrow, ncol), np.nan, np.result_type(img.dtype, np.float32))
    rows = max(1, (1 << 20) // ncol)                               # a million map pixels at a time
    for r0 in range(0, nrow, rows):
        X, Y = np.meshgrid(x0 + (np.arange(ncol) + 0.5) * dx, y1 - (np.arange(r0, min(nrow, r0 + rows)) + 0.5) * dy)
        lon, lat = (v.reshape(X.shape) for v in inv(X, Y))
        ij = locate(out, lat, lon, _height(height, lat, lon))
        ij = (ij - [(l1 - 1) / 2.0, (l2 - 1) / 2.0]) / [l1, l2]
        res[r0:r0 + len(X)] = sample(img, ij, order)
    try:
        from rasterio.transform import Affine
        tr = Affine(dx, 0.0, x0, 0.0, -dy, y1)
    except ImportError:
        tr = (dx, 0.0, x0, 0.0, -dy, y1)
    return dict(data=res, transform=tr, crs=crs)


def read_dem(path, offset=0.0):
    """A DEM GeoTIFF (any crs) as the function dem(lat, lon) -> height that geolocate and geocode_image take:
    bilinear between posts, NaN outside, plus offset. DEMs often give heights above the geoid (Copernicus DEM:
    EGM2008); offset, the geoid's height above the ellipsoid at the scene, makes them ellipsoid heights. The whole
    raster is read into memory. Needs rasterio."""
    import rasterio
    from rasterio.warp import transform as warp
    with rasterio.open(path) as src:
        z = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        tr, crs = src.transform, src.crs

    def dem(lat, lon):
        lat, lon = np.broadcast_arrays(np.asarray(lat, np.float64), np.asarray(lon, np.float64))
        x, y = lon, lat
        if not crs.is_geographic:
            x, y = (np.reshape(v, lat.shape) for v in warp('EPSG:4326', crs, lon.ravel(), lat.ravel()))
        col, row = ~tr * (x, y)
        return sample(z, np.stack([row - 0.5, col - 0.5], -1)) + offset
    return dem
