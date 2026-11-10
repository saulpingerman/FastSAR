"""SIDD metadata without a SAR-format library: the XML of a detected, remapped 8-bit product on a plane.

    sd = build(fields, version=3)            # SIDD 3.0 (or 2.0) metadata of sidd.write_sidd's product
    sd.Measurement.PlaneProjection.SampleSpacing.Row, sd.Display.PixelType          # attribute access
    sd.Display.NonInteractiveProcessing[0].ProductGenerationOptions.DataRemapping.LUTName
    sd.ExploitationFeatures.Collections[0].Geometry.Graze
    sd.to_xml_bytes()

What implements what (NGA.STND.0025, SIDD Volumes 1 and 2, versions 2.0 and 3.0):
- build: the XML structure of Volume 1 (Design and Implementation Description) for a MONO8I product with a plane
  projection: ProductCreation, Display, GeoData, Measurement and ExploitationFeatures, in the SIDD namespace with the
  shared SICommon types and the classification attributes of the intelligence community's ISM schema.
- exploitation: Volume 1's ExploitationFeatures as functions of the collection geometry at the scene center point
  (taken from the SICD metadata of the complex image) and of the product plane: azimuth, slope, squint, graze, tilt
  and Doppler cone angle; shadow, layover, multipath, ground track and north in the pixel frame; the slant-plane
  resolution of the uniformly weighted aperture and the resolution in the product plane. Version 3.0 counts the
  in-image angles clockwise from the column direction in [0, 360); version 2.0 from the rows toward the columns.
- SPECIFICATION: the version and date Volume 2 (the NITF file format description) has the XML data extension
  segment's subheader state.

The object is a read-only view (fastsar._xmlmeta.Node over a namespace-free copy); to_xml_bytes returns the document
as built, with its namespace prefixes."""
import xml.etree.ElementTree as ET

import numpy as np

from ._xmlmeta import Node, Spec, parse, text
from . import _sicdxml

SPECIFICATION = {2: ('2.0', '2019-05-31T00:00:00Z', 'urn:SIDD:2.0.0'), 3: ('3.0', '2021-11-30T00:00:00Z', 'urn:SIDD:3.0.0')}
_NS = {'xmlns:sicommon': 'urn:SICommon:1.0', 'xmlns:sfa': 'urn:SFA:1.2.0', 'xmlns:ism': 'urn:us:gov:ic:ism:13'}

SPEC = Spec(
    text={'Application', 'ProcessingDateTime', 'Site', 'Profile', 'ProductName', 'ProductClass', 'ProductType',
          'PixelType', 'LUTName', 'DatabaseName', 'DownsamplingMethod', 'FilterName', 'Operation', 'ShadowDirection',
          'AlgorithmType', 'EarthModel', 'SensorName', 'ModeType', 'ModeID', 'CollectionDateTime', 'LocalDateTime',
          'TxPolarization', 'RcvPolarization', 'TxPolarizationProc', 'RcvPolarizationProc', 'Classification'},
    ints={'NumBands', 'BandStatsSource', 'DefaultBandDisplay', 'PixelFootprint/Row', 'PixelFootprint/Col',
          'Vertex/Row', 'Vertex/Col'},
    lists={('Display', 'NonInteractiveProcessing'), ('Display', 'InteractiveProcessing'), ('ValidData', 'Vertex'),
           ('ImageCorners', 'ICP')},
    aliases={('ExploitationFeatures', 'Collections'): ('', 'Collection', 'list'),
             ('ExploitationFeatures', 'Products'): ('', 'Product', 'list'),
             ('Information', 'Polarizations'): ('', 'Polarization', 'list'),
             ('Product', 'Polarizations'): ('', 'Polarization', 'list')})

class SIDD(Node):
    """SIDD metadata as written: attribute access by the XML's element names (ExploitationFeatures.Collections and
    .Products, Display.NonInteractiveProcessing and .InteractiveProcessing are lists; the classification markings
    are ProductCreation.Classification.attrib), version ('3.0.0'), to_xml_bytes(). A read-only view."""
    __slots__ = ('_xml',)

    def __init__(self, xml):
        n = parse(xml, SPEC)
        if n.tag != 'SIDD':
            raise ValueError(f'not SIDD metadata: the root element is {n.tag}')
        Node.__init__(self, n.element, SPEC)
        object.__setattr__(self, '_ns', n.namespace)
        object.__setattr__(self, '_xml', xml if isinstance(xml, bytes) else str(xml).encode())

    from_xml = classmethod(lambda cls, xml: cls(xml))

    def __setattr__(self, name, value):
        raise AttributeError('SIDD metadata are read-only once built')

    @property
    def version(self):
        return str(self._ns).rsplit(':', 1)[-1]

    def copy(self):
        return SIDD(self._xml)

    def __getstate__(self):
        return self._xml

    def __setstate__(self, s):
        SIDD.__init__(self, s)

    def to_xml_bytes(self, declaration=False, **kw):
        return (b'<?xml version="1.0" encoding="utf-8"?>\n' if declaration else b'') + self._xml


# ---------------------------------------------------------------------------------------------- geometry
def _angle(y, x, version):
    """The in-image angle of a direction with components x along the rows and y along the columns: version 2 from
    the rows toward the columns in (-180, 180]; version 3 from the columns toward the rows in [0, 360)."""
    if version == 2:
        return float(np.degrees(np.arctan2(y, x)))
    return float(np.degrees(np.arctan2(x, y)) % 360.0)


def exploitation(sm, urow, ucol, version=3):
    """The ExploitationFeatures values of a product on the plane with unit vectors urow, ucol [3] (ECF) made from
    the complex image with SICD metadata sm (its SCPCOA block gives the collection geometry) -> dict(geometry,
    phenomenology, resolution (range, azimuth; slant plane, m), product_resolution (row, col; m), ellipticity,
    north), angles in degrees."""
    unit = _sicdxml._unit
    scp = sm.GeoData.SCP.ECF.get_array()
    arp, vel = sm.SCPCOA.ARPPos.get_array(), sm.SCPCOA.ARPVel.get_array()
    row, col = unit(urow), unit(ucol)
    nrm = unit(np.cross(row, col))
    sx = unit(arp - scp)                               # slant plane axes: toward the radar, across, and the normal
    sz = unit(np.cross(sx, vel))
    if sz @ arp < 0:
        sz = -sz
    sy = np.cross(sz, sx)
    etp = _sicdxml._up(scp)
    c = _sicdxml.scpcoa(scp, 0.0, arp, vel, np.zeros(3))
    look = 1.0 if c['SideOfTrack'] == 'L' else -1.0
    ua = unit(arp)
    vh, lh = unit(unit(vel) - (unit(vel) @ ua) * ua), unit(-sx - (-sx @ ua) * ua)      # horizontal at the radar
    graze = float(np.degrees(np.arcsin(sx @ etp)))
    tilt = float(np.degrees(np.arctan((etp @ sy) / (etp @ sz))))
    geometry = dict(Azimuth=c['AzimAng'], Slope=float(np.degrees(np.arccos(sz @ etp))),
                    Squint=float(look * np.degrees(np.arccos(np.clip(lh @ vh, -1, 1)))), Graze=graze, Tilt=tilt,
                    DopplerConeAngle=c['DopplerConeAng'])

    def on_plane(v):                                   # carried to the product plane along the slant plane normal
        return v - (v @ nrm) / (sz @ nrm) * sz

    def ang(v):
        return _angle(col @ v, row @ v, version)
    shadow = on_plane(etp - sx / (sx @ etp))
    layover = nrm - sz / (sz @ nrm)
    lat, lon = (np.radians(v) for v in _sicdxml._geodetic(scp)[:2])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    phenomenology = dict(Shadow=(ang(shadow), float(np.linalg.norm(shadow))),
                         Layover=(ang(layover), float(np.linalg.norm(layover))),
                         MultiPath=ang(on_plane(sx)), GroundTrack=ang(vel - (vel @ nrm) * nrm))
    rng, az = slant_resolution(sm)
    # the product plane's resolution from the slant plane's (Volume 1): the range direction's angle in the image,
    # the graze and the tilt
    xg = sx - (sx @ nrm) * nrm
    th = -np.arctan2(col @ xg, row @ xg)
    g, tl = np.radians(graze), np.radians(tilt)
    tt = np.tan(g) * np.tan(tl)
    kr1 = (np.cos(th) / np.cos(g)) ** 2 + (np.sin(th) ** 2 * tt - np.sin(2 * th) / np.cos(g)) * tt
    kr2 = (np.sin(th) / np.cos(tl)) ** 2
    kc1 = (np.sin(th) ** 2 / np.cos(g) - np.sin(2 * th) * tt) / np.cos(g) + (np.cos(th) * tt) ** 2
    kc2 = (np.cos(th) / np.cos(tl)) ** 2
    rr, rcol = float(np.sqrt(kr1 * rng ** 2 + kr2 * az ** 2)), float(np.sqrt(kc1 * rng ** 2 + kc2 * az ** 2))
    return dict(geometry=geometry, phenomenology=phenomenology, resolution=(rng, az), product_resolution=(rr, rcol),
                ellipticity=max(rr, rcol) / min(rr, rcol), north=ang(on_plane(north)))


_HALF_POWER = 0.8858929413789047           # the half-power width of a uniform aperture's response, times its band


def slant_resolution(sm):
    """(range, azimuth) resolution (m) in the slant plane of a SICD's image for a uniformly weighted aperture: the
    half-power widths 0.886 / ImpRespBW along the rows and columns, each carried to the slant plane at the scene
    center point through the change of range and of Doppler cone angle it makes, the larger of the two taken."""
    m = _sicdxml._Model(sm)
    d = _HALF_POWER / np.array([float(sm.Grid.Row.ImpRespBW), float(sm.Grid.Col.ImpRespBW)])
    r, rdot, a, v = m.coa(m.pixel + np.array([[0.0, 0.0], [d[0] / m.ss[0], 0.0], [0.0, d[1] / m.ss[1]]]))
    vm = np.linalg.norm(v[0])
    uv = v[0] / vm
    sx = _sicdxml._unit(a[0] - m.scp)
    look = np.sign(np.cross(a[0], v[0]) @ (m.scp - a[0]))
    sz = _sicdxml._unit(look * np.cross(sx, uv))
    sy, vc = np.cross(sz, sx), np.cross(sz, uv)
    cos0 = -rdot[0] / vm
    da, dv = a[1:] - a[0], v[1:] - v[0]
    dcos = -(rdot[1:] - rdot[0] + (dv @ uv) * cos0) / vm
    dang = (dv @ vc) / vm - look * dcos / np.sqrt(1 - cos0 ** 2)
    return float(np.max(np.abs(da @ sx - (r[1:] - r[0])))), float(np.max(np.abs(da @ sy - r[0] * dang)))


# ---------------------------------------------------------------------------------------------- XML
def _e(tag, value=None, **attrib):
    e = ET.Element(tag, {k: str(v) for k, v in attrib.items()})
    if isinstance(value, list):
        e.extend(value)
    elif value is not None:
        e.text = text(value)
    return e


def _c(tag, names, values, **attrib):
    """An element of a shared SICommon type: children names (prefixed sicommon) with values."""
    return _e(tag, [_e(f'sicommon:{k}', v) for k, v in zip(names, values)], **attrib)


def _poly1(tag, coef):
    c = np.asarray(coef, np.float64)
    return _e(tag, [_e('sicommon:Coef', float(v), exponent1=i) for i, v in enumerate(c)], order1=len(c) - 1)


def classification(marking, date):
    """The ISM attributes of ProductCreation.Classification: 'UNCLASSIFIED' (or 'U'), or a dict of the markings
    (classification and ownerProducer at the least, as dict(classification='U', ownerProducer='USA',
    releasableTo='USA GBR'))."""
    base = {'DESVersion': '13', 'resourceElement': 'true', 'createDate': date, 'compliesWith': 'USGov',
            'ISMCATCESVersion': '201903'}
    if isinstance(marking, dict) and 'DESVersion' in marking:
        return marking
    if isinstance(marking, dict):
        if 'classification' not in marking or 'ownerProducer' not in marking:
            raise ValueError('classification markings need at least classification and ownerProducer')
        base.update({str(k): str(v) for k, v in marking.items()})
    elif str(marking).upper() in ('UNCLASSIFIED', 'U'):
        base.update(classification='U', ownerProducer='USA')
    else:
        raise ValueError(f'classification {marking!r}: UNCLASSIFIED, or a dict of the ISM markings '
                         "(classification, ownerProducer, ...)")
    return base


def build(f, version=3):
    """SIDD metadata of a MONO8I product on a plane. f: dict(application, created ('YYYY-MM-DDThh:mm:ss.ffffffZ'),
    classification (see classification), product_name, lut (the remap's name), shadow ('DOWN', ...), corners [4, 2]
    and valid_ll [n, 2] (latitude, longitude, degrees), ref_ecf [3], ref_pixel (row, col), spacing (row, col; m),
    t_coa (s), urow, ucol [3] (ECF), rows, cols, arp [3, n] (polynomial coefficients), valid_rc [n, 2] (pixels),
    sicd (the SICD metadata of the complex image, for the collection and its geometry)). version: 3 or 2."""
    sm = f['sicd']
    ex = exploitation(sm, f['urow'], f['ucol'], version)
    ism = classification(f['classification'], f['created'][:10])

    def bilinear(tag, op):
        return _e(tag, [_e('FilterName', tag), _e('FilterBank', [_e('Predefined', [_e('DatabaseName', 'BILINEAR')])]),
                        _e('Operation', op)])

    def points(tag, sub, pts, names, labels=None, size=False, cast=float):
        kids = [_c(sub, names, [cast(v) for v in p], index=f'{i + 1}:{labels[i]}' if labels else i + 1)
                for i, p in enumerate(pts)]
        return _e(tag, kids, **({'size': len(kids)} if size else {}))
    ci, tl = sm.CollectionInfo, sm.Timeline
    pol = str(sm.ImageFormation.TxRcvPolarizationProc or 'UNKNOWN')
    tx, rcv = pol.split(':') if ':' in pol else (pol, pol)
    if version == 2:                            # version 2 enumerates V, H, RHC, LHC, OTHER and UNKNOWN
        tx, rcv = (p if p in ('V', 'H', 'RHC', 'LHC', 'OTHER', 'UNKNOWN') else
                   'OTHER' if p in ('X', 'Y', 'S', 'E') or p.startswith('OTHER') else 'UNKNOWN' for p in (tx, rcv))
    core = str(ci.CoreName)
    g, ph = ex['geometry'], ex['phenomenology']
    root = _e('SIDD', [
        _e('ProductCreation', [
            _e('ProcessorInformation', [_e('Application', f['application']), _e('ProcessingDateTime', f['created']),
                                        _e('Site', 'Unknown')]),
            _e('Classification', **{f'ism:{k}': v for k, v in ism.items()}),
            _e('ProductName', f['product_name']), _e('ProductClass', 'Detected Image')]),
        _e('Display', [
            _e('PixelType', 'MONO8I'), _e('NumBands', 1),
            _e('NonInteractiveProcessing', [
                _e('ProductGenerationOptions', [_e('DataRemapping', [
                    _e('LUTName', f['lut']), _e('Predefined', [_e('DatabaseName', f['lut'])])])]),
                _e('RRDS', [_e('DownsamplingMethod', 'DECIMATE')])], band=1),
            _e('InteractiveProcessing', [
                _e('GeometricTransform', [
                    _e('Scaling', [bilinear('AntiAlias', 'CONVOLUTION'), bilinear('Interpolation', 'CORRELATION')]),
                    _e('Orientation', [_e('ShadowDirection', f['shadow'])])]),
                _e('SharpnessEnhancement', [bilinear('ModularTransferFunctionEnhancement', 'CONVOLUTION')]),
                _e('DynamicRangeAdjustment', [_e('AlgorithmType', 'NONE'), _e('BandStatsSource', 1)])], band=1)]),
        _e('GeoData', [_e('EarthModel', 'WGS_84'),
                       points('ImageCorners', 'ICP', f['corners'], ('Lat', 'Lon'), ('FRFC', 'FRLC', 'LRLC', 'LRFC')),
                       points('ValidData', 'Vertex', f['valid_ll'], ('Lat', 'Lon'), size=True)]),
        _e('Measurement', [
            _e('PlaneProjection', [
                _e('ReferencePoint', [_c('sicommon:ECEF', 'XYZ', [float(v) for v in f['ref_ecf']]),
                                      _c('sicommon:Point', ('Row', 'Col'), [float(v) for v in f['ref_pixel']])]),
                _c('SampleSpacing', ('Row', 'Col'), [float(v) for v in f['spacing']]),
                _e('TimeCOAPoly', [_e('sicommon:Coef', float(f['t_coa']), exponent1=0, exponent2=0)], order1=0,
                   order2=0),
                _e('ProductPlane', [_c('RowUnitVector', 'XYZ', [float(v) for v in f['urow']]),
                                    _c('ColUnitVector', 'XYZ', [float(v) for v in f['ucol']])])]),
            _c('PixelFootprint', ('Row', 'Col'), [int(f['rows']), int(f['cols'])]),
            _e('ARPPoly', [_poly1(f'sicommon:{k}', c) for k, c in zip('XYZ', f['arp'])]),
            points('ValidData', 'Vertex', f['valid_rc'], ('Row', 'Col'), size=True, cast=int)]),
        _e('ExploitationFeatures', [
            _e('Collection', [
                _e('Information', [
                    _e('SensorName', str(ci.CollectorName)),
                    _e('RadarMode', [_e('sicommon:ModeType', str(ci.RadarMode.ModeType))]),
                    _e('CollectionDateTime', str(tl.CollectStart)), _e('CollectionDuration', float(tl.CollectDuration)),
                    _c('Resolution', ('Range', 'Azimuth'), ex['resolution']),
                    _e('Polarization', [_e('TxPolarization', tx), _e('RcvPolarization', rcv)])]),
                _e('Geometry', [_e(k, g[k]) for k in ('Azimuth', 'Slope', 'Squint', 'Graze', 'Tilt', 'DopplerConeAngle')]),
                _e('Phenomenology', [_c('Shadow', ('Angle', 'Magnitude'), ph['Shadow']),
                                     _c('Layover', ('Angle', 'Magnitude'), ph['Layover']),
                                     _e('MultiPath', ph['MultiPath']), _e('GroundTrack', ph['GroundTrack'])])],
               identifier=core),
            _e('Product', [_c('Resolution', ('Row', 'Col'), ex['product_resolution']),
                           _e('Ellipticity', ex['ellipticity']),
                           _e('Polarization', [_e('TxPolarizationProc', tx), _e('RcvPolarizationProc', rcv)]),
                           _e('North', ex['north'])])])],
        xmlns=SPECIFICATION[version][2], **_NS)
    return SIDD(ET.tostring(root, encoding='utf-8'))
