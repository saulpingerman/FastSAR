"""SIDD products and the display orientation, from a form_cphd output.

    disp = fastsar.sidd.to_display(out)                  # uint8, NRL remap, shadows down
    fastsar.sidd.write_sidd('image.nitf', out)           # the same pixels as a SIDD (MONO8I NITF, needs sarpy)
    view = fastsar.sidd.shadows_down(out)                # the reorientation alone, with its geometry

A detected image is displayed with the radar at the top: range increases down the rows, so that shadows, which fall
away from the radar, point down and relief reads as it would under light from above. The view is also the one from
above (not a mirror image): row x col points up. Of the eight ways of laying a pixel grid out without resampling
(a transpose, with or without a flip of either axis), four are views from above, and shadows_down takes the one
whose rows lie nearest the range direction. On a grid whose axes follow the line of sight, shadows then point
straight down; on any other grid they lean by the angle between the grid and the line of sight (key `angle`), at
most 45 degrees, and removing that lean would take a resampling.

The product dict is form_cphd's: image [n1, n2] with pixel (i, j) at origin + i spx e1 + j spy e2 in the local
frame of meta (io.local_to_ecf), e1 and e2 orthonormal, and the antenna track in meta. Pixels that are exactly 0
are taken as unprocessed (a tasked area inside a rectangular array): they stay 0, and they are left out of the
statistics a remap draws from the image.

The image may be on a device (a CuPy or JAX array, fastsar._xp): shadows_down, orient and to_display then work
there and return device arrays, and write_sidd remaps on the device and brings only the display values to the host.
"""
import os

import numpy as np

from . import _deps, _xp
from . import remap as _remap

__all__ = ['shadows_down', 'orient', 'to_display', 'write_sidd']

_UP = np.array([0.0, 0.0, 1.0])                                   # the local frame's vertical
_SPECIFICATION = {2: ('2.0', '2019-05-31T00:00:00Z'), 3: ('3.0', '2021-11-30T00:00:00Z')}   # SIDD version and date


def _unit(v):
    return v / np.linalg.norm(v)


def _array(a):
    """a itself when it is on a device, else as a NumPy array."""
    return np.asarray(a) if _xp.kind(a) == 'numpy' else a


def _frame(out):
    """e1, e2, the plane's normal on the upward side, and the unit vector in the plane along range away from the
    radar (the line of sight from the mid-aperture antenna position to the image center, projected on the plane)."""
    n1, n2 = np.shape(out['image'])[:2]
    e1, e2 = np.asarray(out['e1'], np.float64), np.asarray(out['e2'], np.float64)
    meta = out['meta']
    ant = 0.5 * (np.asarray(meta['tx'], np.float64) + np.asarray(meta['rcv'], np.float64))
    center = (np.asarray(out['origin'], np.float64) + 0.5 * (n1 - 1) * out['spx'] * e1
              + 0.5 * (n2 - 1) * out['spy'] * e2)
    n = _unit(np.cross(e1, e2))
    if n @ _UP < 0:
        n = -n
    los = center - ant[len(ant) // 2]
    g = los - (los @ n) * n
    if np.linalg.norm(g) < 1e-6 * np.linalg.norm(los):
        raise ValueError('the line of sight is normal to the image plane: there is no range direction in it')
    return e1, e2, n, _unit(g)


def _layout(out, transpose, flip_rows, flip_cols):
    """The image of a product laid out by a transform, with the geometry of the result."""
    img = _array(out['image'])
    e1, e2, n, g = _frame(out)
    tr = dict(transpose=bool(transpose), flip_rows=bool(flip_rows), flip_cols=bool(flip_cols))
    (row, rs, nr), (col, cs, nc) = (e1, out['spx'], img.shape[0]), (e2, out['spy'], img.shape[1])
    if transpose:
        (row, rs, nr), (col, cs, nc) = (col, cs, nc), (row, rs, nr)
    origin = np.asarray(out['origin'], np.float64)
    if flip_rows:
        origin, row = origin + (nr - 1) * rs * row, -row
    if flip_cols:
        origin, col = origin + (nc - 1) * cs * col, -col
    angle = float(np.degrees(np.arctan2(g @ col, g @ row)))
    return dict(image=orient(img, tr), origin=origin, row=row, col=col, row_ss=float(rs), col_ss=float(cs),
                transform=tr, angle=angle)


def _shadows_down(out):
    e1, e2, n, g = _frame(out)
    best = None
    for transpose, a, b in ((False, e1, e2), (True, e2, e1)):
        for s in (1.0, -1.0):
            row = s * a
            col = np.cross(n, row)                                 # row x col = n: the view from above
            if best is None or row @ g > best[0]:
                best = (row @ g, transpose, s < 0, col @ b < 0)
    return _layout(out, *best[1:])


def shadows_down(out):
    """A form_cphd product laid out for display: range away from the radar increasing down the rows, seen from above.

    Of the lossless layouts (transpose and flips, no interpolation) that are views from above (row x col along the
    upward normal of the image plane), the one whose rows lie nearest the range direction in the plane away from
    the radar, judged at the image center from the mid-aperture antenna position. Ground-plane products, and
    slant-plane ones (out['plane'] == 'slant', e1 and e2 spanning the slant plane) by the same rule.

    -> dict(image [rows, cols] (a view of out['image']), origin [3] (local position of its pixel (0, 0)),
    row, col (unit vectors along increasing row and column, local), row_ss, col_ss (m), transform (what was done to
    the array: dict(transpose, flip_rows, flip_cols), applied in that order; orient applies it to another array, a
    mask for example), angle (degrees from the rows to the range direction, positive toward the columns: 0 when
    the grid follows the line of sight))."""
    return _shadows_down(out)


def orient(array, transform):
    """An array on the product's grid [n1, n2, ...] laid out as shadows_down laid the image out (transform: its
    'transform' entry). A view (of a JAX array, a copy when an axis is flipped)."""
    a = _array(array)
    if transform['transpose']:
        a = _xp.module(a).swapaxes(a, 0, 1)
    if transform['flip_rows']:
        a = a[::-1]
    if transform['flip_cols']:
        a = a[:, ::-1]
    return a


def _valid(out, valid):
    """The mask of processed pixels [n1, n2]: the argument, else out['valid'], else the pixels that are not exactly
    0. None when every pixel is valid. Where the image is."""
    img = _array(out['image'])
    v = valid if valid is not None else out.get('valid')
    v = (img != 0) if v is None else _xp.module(img).asarray(v, bool)
    if v.shape != img.shape:
        raise ValueError(f'valid mask {v.shape} does not match the image {img.shape}')
    return None if bool(v.all()) else v


def _display(out, remap, valid, args):
    """uint8 display values on the product's own grid, and the valid mask (None: all), both where the image is."""
    img = _array(out['image'])
    v = _valid(out, valid)
    if img.dtype == np.uint8:
        return img, v
    if callable(remap):
        d = _array(remap(img, valid=v, **args))
    else:
        if remap not in _remap.REMAPS or remap == 'linear':
            names = sorted(k for k in _remap.REMAPS if k != 'linear')
            raise ValueError(f'remap {remap!r}: one of {names}, or a function f(image, valid=mask) -> uint8 '
                             f"('linear' gives amplitudes, not display values)")
        d = _remap.remap(img, remap, valid=v, **args)
    if d.dtype != np.uint8 or d.shape != img.shape:
        raise ValueError(f'the remap returned {d.dtype} {d.shape}, not uint8 {img.shape}')
    return d, v


def to_display(out, remap='nrl', valid=None, shadows_down=True, **remap_args):
    """Display values (uint8 [rows, cols]) of a form_cphd product: detected, remapped, laid out shadows down.

    remap: a name of fastsar.remap.REMAPS other than 'linear' ('nrl', 'density', 'brighter', 'darker',
    'highcontrast', 'pedf', 'log'; the remaps of the NGA MATLAB SAR Toolbox), with its parameters as keywords
    (to_display(out, 'nrl', a=1.0, c=220.0)), or a function f(image, valid=mask) -> uint8 (given the image and
    the mask where the image is, a device array for a device image). Complex pixels are
    detected by the remap; a uint8 image is taken as display values already. valid: the pixels [n1, n2] (the
    product's own grid) that enter the remap's statistics, the others coming out 0; default out['valid'] if the
    product has one, else every pixel that is not exactly 0. shadows_down False keeps the product's own layout
    (rows along e1, columns along e2). An image on a device (a CuPy or JAX array) is remapped there, and the
    display values are an array of the same kind (a remap's statistics as fastsar.remap computes them there, which
    can move a value on a rounding boundary by one count)."""
    d, _ = _display(out, remap, valid, remap_args)
    return orient(d, (_shadows_down(out) if shadows_down else _layout(out, False, False, False))['transform'])


def _hull(v):
    """Vertices [k, 2] (row, col) of the convex hull of the True pixels of a mask, clockwise as SIDD counts it
    (in the plane x = row, y = col) from the vertex of least row, then least column. The bounding rectangle of the
    True pixels when the hull has fewer than 3 or more than 100 vertices. The mask has a True pixel; it may be on a
    device, from where only its rows' first and last True columns come to the host."""
    rows, lo, hi = (_xp.host(a) for a in (v.any(1), v.argmax(1), v.shape[1] - 1 - v[:, ::-1].argmax(1)))
    rows = np.flatnonzero(rows)
    lo, hi = lo[rows], hi[rows]
    r0, r1, c0, c1 = rows[0], rows[-1], lo.min(), hi.max()
    pts = sorted(set(zip(rows.tolist(), lo.tolist())) | set(zip(rows.tolist(), hi.tolist())))

    def chain(p):
        h = []
        for q in p:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = chain(pts), chain(pts[::-1])
    hull = lower[:-1] + upper[:-1]                                  # counterclockwise from the least (row, col)
    if not 3 <= len(hull) <= 100:
        return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
    return np.array([hull[0]] + hull[:0:-1])


def _igeolo(latlon):
    """The NITF IGEOLO field of four corners [4, 2] (latitude, longitude in degrees): ddmmssXdddmmssY for each,
    rounded to the arcsecond."""
    def dms(value, digits, pos, neg):
        s = int(round(abs(float(value)) * 3600))
        return f'{s // 3600:0{digits}d}{s // 60 % 60:02d}{s % 60:02d}{pos if value >= 0 else neg}'
    return ''.join(dms(lat, 2, 'N', 'S') + dms(lon, 3, 'E', 'W') for lat, lon in np.asarray(latlon))


def write_sidd(path, out, *, remap='nrl', remap_args=None, valid=None, shadows_down=True, version=3,
               product_name=None, classification='UNCLASSIFIED', sicd=None):
    """Write a form_cphd product as a SIDD: a detected, remapped 8-bit image (MONO8I) in a NITF. Needs sarpy.

    The pixels are to_display's (remap, remap_args (a dict of the remap's parameters), valid and shadows_down as
    there). A uint8 image is written as it is, and remap then only names, in the metadata, the remap that made
    it. An image on a device (a CuPy or JAX array) is remapped there, and its display values alone are copied to the
    host. The product must lie on the ground plane: SIDD's plane projection could describe a slant-plane image, but
    a display product is a ground product, and a slant-plane one (out['plane'] == 'slant') is refused.

    Metadata. Measurement: a plane projection (reference point at the center pixel, the sample spacings, the row
    and column unit vectors in ECF), the aperture position polynomial and one center of aperture time, so that the
    SIDD's own projection of a pixel is the range-Doppler model of products.geolocate and products.locate (as for
    the SICD of products.sicd_meta); the pixel footprint, and ValidData, the convex hull of the valid pixels.
    GeoData: the corners of the image and that hull, in latitude and longitude. Display: MONO8I, the remap's name,
    and the shadow direction (DOWN when shadows point down the rows to within a degree, likewise UP, LEFT or RIGHT,
    else ARBITRARY). ExploitationFeatures: the collection geometry (azimuth, graze, slope, tilt, squint, Doppler
    cone angle), the shadow, layover, multipath and ground track directions and north in the pixel frame, and the
    resolution, all as sarpy derives them from the SICD metadata of the same product. ProductCreation: FastSAR and
    its version as the processor.

    version: SIDD 3 or 2 (3 falls back to 2 with a sarpy that cannot write it). product_name: default the
    collection's core name, else the file's name. classification: 'UNCLASSIFIED', or a sarpy
    ProductClassificationType carrying the markings of anything else. sicd: the SICD metadata (sarpy SICDType) of
    the complex image the product was made from; it supplies the collection information and geometry and is stored
    in the file beside the SIDD. Default: products.sicd_meta(out), not stored.

    -> the sarpy SIDDType written."""
    import importlib
    from datetime import datetime, timezone
    _deps.require('sarpy')
    from sarpy.io.product.sidd import SIDDWriter, SIDDWritingDetails
    from . import __version__, products
    from .io import local_to_ecf, ecf_to_geodetic
    if out.get('plane') == 'slant':
        raise ValueError('write_sidd writes ground-plane products; this one is on the slant plane '
                         "(out['plane'] == 'slant'): project it to the ground first")
    if version not in (2, 3):
        raise ValueError(f'SIDD version {version}: 2 or 3')
    try:
        importlib.import_module(f'sarpy.io.product.sidd{version}_elements.SIDD')
    except ImportError:
        if version == 2:
            raise
        version = 2
    S, D, B, G, M, E, P = (importlib.import_module(f'sarpy.io.product.sidd{version}_elements.{m}') for m in
                           ('SIDD', 'Display', 'blocks', 'GeoData', 'Measurement', 'ExploitationFeatures',
                            'ProductCreation'))
    meta = out['meta']
    data, v = _display(out, remap, valid, dict(remap_args or {}))
    lay = _shadows_down(out) if shadows_down else _layout(out, False, False, False)
    data = np.ascontiguousarray(_xp.host(orient(data, lay['transform'])))
    rows, cols = data.shape

    def ecf(rc):                                                   # pixel coordinates [..., 2] (row, col) -> ECF
        rc = np.asarray(rc, np.float64)
        return local_to_ecf(lay['origin'] + rc[..., :1] * lay['row_ss'] * lay['row']
                            + rc[..., 1:] * lay['col_ss'] * lay['col'], meta)

    def latlon(rc):
        return np.stack(ecf_to_geodetic(ecf(rc))[:2], -1)

    def outline(r0, r1, c0, c1):                                   # the outer corners of pixels [r0, r1) x [c0, c1)
        return [[r0 - 0.5, c0 - 0.5], [r0 - 0.5, c1 - 0.5], [r1 - 0.5, c1 - 0.5], [r1 - 0.5, c0 - 0.5]]
    urow, ucol = lay['row'] @ meta['R'], lay['col'] @ meta['R']
    if sicd is not None:
        sm = sicd
    elif _xp.kind(out['image']) == 'numpy':
        sm = products.sicd_meta(out)[0]
    else:                                                          # the metadata need the image's shape alone
        sm = products.sicd_meta(dict(out, image=np.broadcast_to(np.complex64(0), np.shape(out['image']))))[0]
    # Measurement
    ref = (rows // 2, cols // 2)
    t_coa = float(sm.SCPCOA.SCPTime) if sm.SCPCOA is not None and sm.SCPCOA.SCPTime is not None \
        else float(sm.Grid.TimeCOAPoly.get_array()[0, 0])
    poly = _hull(orient(v, lay['transform'])) if v is not None and v.any() else \
        np.array([[0, 0], [0, cols - 1], [rows - 1, cols - 1], [rows - 1, 0]])
    arp = sm.Position.ARPPoly
    measurement = M.MeasurementType(
        PixelFootprint=(rows, cols), ValidData=[tuple(int(k) for k in p) for p in poly],
        PlaneProjection=M.PlaneProjectionType(
            ReferencePoint=B.ReferencePointType(ECEF=ecf(ref), Point=(float(ref[0]), float(ref[1]))),
            SampleSpacing=(lay['row_ss'], lay['col_ss']), TimeCOAPoly=B.Poly2DType(Coefs=[[t_coa]]),
            ProductPlane=M.ProductPlaneType(RowUnitVector=urow, ColUnitVector=ucol)),
        ARPPoly=B.XYZPolyType(X=arp.X.get_array(), Y=arp.Y.get_array(), Z=arp.Z.get_array()))
    # GeoData: the outer corners of the corner pixels (first row first column, then clockwise) and the valid data
    geodata = G.GeoDataType(ImageCorners=latlon(outline(0, rows, 0, cols)), ValidData=latlon(poly))
    # Display
    name = (remap if isinstance(remap, str) else getattr(remap, '__name__', 'custom')).upper()
    g = _frame(out)[3]                                             # shadows fall along range, away from the radar
    along = {'DOWN': g @ lay['row'], 'UP': -(g @ lay['row']), 'RIGHT': g @ lay['col'], 'LEFT': -(g @ lay['col'])}
    shadow = max(along, key=along.get)
    if along[shadow] < np.cos(np.radians(1.0)):
        shadow = 'ARBITRARY'

    def bilinear(fname, op):
        return B.FilterType(FilterName=fname, Operation=op, FilterBank=B.FilterBankType(
            Predefined=B.PredefinedFilterType(DatabaseName='BILINEAR')))
    display = D.ProductDisplayType(
        PixelType='MONO8I', NumBands=1,
        NonInteractiveProcessing=[D.NonInteractiveProcessingType(
            ProductGenerationOptions=D.ProductGenerationOptionsType(DataRemapping=B.NewLookupTableType(
                LUTName=name, Predefined=B.PredefinedLookupType(DatabaseName=name))),
            RRDS=D.RRDSType(DownsamplingMethod='DECIMATE'), band=1)],
        InteractiveProcessing=[D.InteractiveProcessingType(
            GeometricTransform=D.GeometricTransformType(
                Scaling=D.ScalingType(AntiAlias=bilinear('AntiAlias', 'CONVOLUTION'),
                                      Interpolation=bilinear('Interpolation', 'CORRELATION')),
                Orientation=D.OrientationType(ShadowDirection=shadow)),
            SharpnessEnhancement=D.SharpnessEnhancementType(
                ModularTransferFunctionEnhancement=bilinear('ModularTransferFunctionEnhancement', 'CONVOLUTION')),
            DynamicRangeAdjustment=D.DynamicRangeAdjustmentType(AlgorithmType='NONE', BandStatsSource=1),
            band=1)])
    # ProductCreation
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if isinstance(classification, P.ProductClassificationType):
        marking = classification
    elif str(classification).upper() in ('UNCLASSIFIED', 'U'):
        marking = P.ProductClassificationType(classification='U', createDate=now.date().isoformat())
    else:
        raise ValueError(f'classification {classification!r}: UNCLASSIFIED, or a sarpy ProductClassificationType '
                         f'with the markings')
    core = sm.CollectionInfo.CoreName if sm.CollectionInfo is not None else None
    if product_name is None:
        product_name = core if core and core != 'UNKNOWN' else os.path.splitext(os.path.basename(str(path)))[0]
    creation = P.ProductCreationType(
        ProcessorInformation=P.ProcessorInformationType(Application=f'FastSAR {__version__}',
                                                        ProcessingDateTime=np.datetime64(now, 'us'), Site='Unknown'),
        Classification=marking, ProductName=str(product_name), ProductClass='Detected Image')
    sidd = S.SIDDType(ProductCreation=creation, GeoData=geodata, Display=display, Measurement=measurement,
                      ExploitationFeatures=E.ExploitationFeaturesType.from_sicd(sm, urow, ucol))
    # The NITF headers are sarpy's, but for two fields that sarpy 2.1.1 fills differently from the standard: the
    # version and date of the SIDD specification in the XML segment's header, and the corner coordinates of each
    # image segment, which sarpy truncates to the arcsecond where the standard rounds.
    details = SIDDWritingDetails(sidd, sicd_meta=sicd)
    header = details.des_managers[0].subheader.UserHeader
    header.DESSHSV, header.DESSHSD = _SPECIFICATION[version]
    for manager, box in zip(details.image_managers, details.image_segment_coordinates[0]):
        manager.subheader.IGEOLO = _igeolo(latlon(outline(*box)))
    with SIDDWriter(path, sidd_writing_details=details, check_existence=False) as w:
        w.write_chip(data, start_indices=(0, 0))
    return sidd
