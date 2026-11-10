"""SICD metadata without a SAR-format library: the XML of a product, a light object over it, and its projection.

    sm = build(fields)                       # SICD 1.3.0 metadata of a FastSAR product (products.sicd_meta)
    sm = read(path)                          # of a SICD NITF, a .xml file, or XML bytes (versions 1.1 to 1.4)
    sm.Grid.Row.SS, sm.GeoData.SCP.ECF.get_array(), sm.SCPCOA.GrazeAng       # attribute access (fastsar._xmlmeta)
    sm.project_image_to_ground(rc, projection_type='HAE', hae0=h)            # ECF [..., 3]
    sm.project_ground_to_image(ecf)                                          # (row, col) [..., 2], residual, iterations
    sm.to_xml_bytes()

What implements what (NGA.STND.0024, SICD Volumes 1 to 3, version 1.3.0):
- build and the element order tables: the XML structure of Volume 1 (Design and Implementation Description),
  for the blocks a backprojected product on a plane carries (CollectionInfo, ImageCreation, ImageData, GeoData, Grid,
  Timeline, Position, RadarCollection, ImageFormation, SCPCOA).
- derive: the fields Volume 1 defines as functions of others: GeoData.SCP.LLH, the SCPCOA block (the aperture
  reference point's position, velocity and acceleration at the scene center point's center of aperture time; slant
  and ground range; Doppler cone, graze, incidence, twist, slope, azimuth and layover angles; side of track), the
  weight function samples of a named window (Grid.Row/Col.WgtFunct), and GeoData.ImageCorners (the corner pixels
  projected to the scene center point's height).
- Projection: Volume 3 (Image Projections Description): the image grid to the center-of-aperture range and range
  rate for PLANE, XRGYCR and XCTYAT grids, RGAZIM grids formed by the polar format algorithm (PFA) or by range
  azimuth compression (RGAZCOMP), and RGZERO grids (INCA parameters); the range and range rate contour to a plane and
  to a surface of constant height above the WGS-84 ellipsoid; and the iterative scene to image projection.

Conventions: positions in ECF meters, angles in degrees, times in seconds from the collection start, pixel
coordinates (row, col) of the full image (ImageData.FirstRow and FirstCol are subtracted). Leaves read as float, int,
bool or str; dates are their XML text ('2026-01-02T03:04:05.250000Z')."""
import xml.etree.ElementTree as ET

import numpy as np

from ._xmlmeta import Node, Spec, parse, text

NAMESPACE = 'urn:SICD:1.3.0'
# (version, date) the NITF data extension segment's subheader states for a SICD of each namespace version
SPECIFICATION = {'1.1.0': '2014-09-30T00:00:00Z', '1.2.0': '2016-06-30T00:00:00Z', '1.2.1': '2018-12-13T00:00:00Z',
                 '1.3.0': '2021-11-30T00:00:00Z', '1.4.0': '2023-10-26T00:00:00Z'}

SPEC = Spec(
    text={'CollectorName', 'IlluminatorName', 'CoreName', 'CollectType', 'ModeType', 'ModeID', 'Classification',
          'CountryCode', 'Parameter', 'Application', 'DateTime', 'Site', 'Profile', 'PixelType', 'EarthModel',
          'ImagePlane', 'Type', 'WindowName', 'TxPolarization', 'RcvPolarization', 'TxRcvPolarization',
          'TxRcvPolarizationProc', 'ImageFormAlgo', 'STBeamComp', 'ImageBeamComp', 'AzAutofocus', 'RgAutofocus',
          'SideOfTrack', 'CollectStart', 'NoiseLevelType', 'Identifier', 'SegmentIdentifier', 'Desc', 'TypeID',
          'RcvDemodType', 'ImageType', 'RMAlgoType', 'Frame', 'SensorName', 'CollectorType', 'MatchType'},
    ints={'NumRows', 'NumCols', 'FirstRow', 'FirstCol', 'Row', 'Col', 'Sgn', 'NumChanProc', 'ChanIndex',
          'RefFreqIndex', 'IPPStart', 'IPPEnd', 'TxSequence', 'NumAcqs', 'NumLines', 'NumSamples', 'FirstLine',
          'FirstSample', 'CurrentIndex', 'NumMatchTypes', 'NumMatchCollections', 'MatchIndex'},
    lists={('IPP', 'Set'), ('WgtFunct', 'Wgt'), ('ValidData', 'Vertex'), ('ImageCorners', 'ICP'),
           ('Waveform', 'WFParameters'), ('TxSequence', 'TxStep'), ('ImageFormation', 'Processing'),
           ('AmpTable', 'Amplitude')},
    aliases={('RadarCollection', 'RcvChannels'): ('RcvChannels', 'ChanParameters', 'list'),
             ('RcvChanProc', 'ChanIndices'): ('', 'ChanIndex', 'list'),
             ('CollectionInfo', 'Parameters'): ('', 'Parameter', 'dict'),
             ('CollectionInfo', 'CountryCodes'): ('', 'CountryCode', 'list'),
             ('WgtType', 'Parameters'): ('', 'Parameter', 'dict'),
             ('RadarCollection', 'Parameters'): ('', 'Parameter', 'dict')})

# child order of the elements this module inserts into (Volume 1's sequence)
_ORDER = {
    'SICD': ('CollectionInfo', 'ImageCreation', 'ImageData', 'GeoData', 'Grid', 'Timeline', 'Position',
             'RadarCollection', 'ImageFormation', 'SCPCOA', 'Radiometric', 'Antenna', 'ErrorStatistics', 'MatchInfo',
             'RgAzComp', 'PFA', 'RMA'),
    'GeoData': ('EarthModel', 'SCP', 'ImageCorners', 'ValidData', 'GeoInfo'),
    'SCP': ('ECF', 'LLH'),
    'Row': ('UVectECF', 'SS', 'ImpRespWid', 'Sgn', 'ImpRespBW', 'KCtr', 'DeltaK1', 'DeltaK2', 'DeltaKCOAPoly',
            'WgtType', 'WgtFunct'),
}
_ORDER['Col'] = _ORDER['Row']
WEIGHT_SIZE = 512                       # samples of a derived weight function


# ---------------------------------------------------------------------------------------------- geodesy
def _geodetic(ecf):
    from .io import ecf_to_geodetic
    return np.stack(np.broadcast_arrays(*ecf_to_geodetic(ecf)), -1)


def _ecf(llh):
    from .io import geodetic_to_ecf
    llh = np.asarray(llh, np.float64)
    return geodetic_to_ecf(llh[..., 0], llh[..., 1], llh[..., 2])


def _up(ecf):
    """Unit normal of the WGS-84 ellipsoid (the geodetic vertical) at ECF points [..., 3]."""
    llh = _geodetic(ecf)
    la, lo = np.radians(llh[..., 0]), np.radians(llh[..., 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------- XML building
def el(tag, value=None, **attrib):
    """An element: value None (empty, to append children to), a 3-vector (X, Y, Z children), a list of elements
    (children), or a leaf value."""
    e = ET.Element(tag, {k: str(v) for k, v in attrib.items()})
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], ET.Element):
        e.extend(value)
    elif isinstance(value, np.ndarray) or (isinstance(value, (list, tuple)) and len(value) == 3):
        for k, v in zip('XYZ', np.asarray(value, np.float64)):
            ET.SubElement(e, k).text = text(float(v))
    elif value is not None:
        e.text = text(value)
    return e


def poly(tag, coef):
    """A polynomial element from its coefficient array [n1] or [n1, n2] (exponent attributes as SICD writes them)."""
    c = np.asarray(coef, np.float64)
    if c.ndim == 1:
        e = ET.Element(tag, order1=str(len(c) - 1))
        for i, v in enumerate(c):
            ET.SubElement(e, 'Coef', exponent1=str(i)).text = text(float(v))
        return e
    e = ET.Element(tag, order1=str(c.shape[0] - 1), order2=str(c.shape[1] - 1))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ET.SubElement(e, 'Coef', exponent1=str(i), exponent2=str(j)).text = text(float(c[i, j]))
    return e


def insert(parent, child, order=None):
    """Put child into the ElementTree element parent at its place in the element order (replacing one of its tag)."""
    order = order or _ORDER[parent.tag]
    for old in parent.findall(child.tag):
        parent.remove(old)
    rank = order.index(child.tag)
    at = 0
    for i, c in enumerate(parent):
        if c.tag in order and order.index(c.tag) <= rank:
            at = i + 1
    parent.insert(at, child)


class SICD(Node):
    """SICD metadata: a fastsar._xmlmeta.Node over the SICD XML tree (attribute access by the XML's element names,
    get_array() on vectors and polynomials, None for absent elements, assignment to leaves) with the SICD's own
    projection model. Lists: RadarCollection.RcvChannels, ImageFormation.RcvChanProc.ChanIndices,
    CollectionInfo.Parameters (a dict), GeoData.ImageCorners.get_array() ([4, 2] latitude, longitude)."""
    __slots__ = ()

    def __init__(self, element, namespace=NAMESPACE):
        Node.__init__(self, element, SPEC)
        object.__setattr__(self, '_ns', namespace)

    @classmethod
    def from_xml(cls, xml):
        n = parse(xml, SPEC)
        if n.tag != 'SICD':
            raise ValueError(f'not SICD metadata: the root element is {n.tag}')
        return cls(n.element, n.namespace or NAMESPACE)

    from_xml_string = from_xml

    @property
    def version(self):
        return str(self._ns).rsplit(':', 1)[-1]

    def copy(self):
        import copy
        return SICD(copy.deepcopy(self._e), self._ns)

    def __setstate__(self, s):
        Node.__setstate__(self, s)

    def to_xml_bytes(self, declaration=False, **kw):
        return Node.to_xml_bytes(self, declaration)

    # -------------------------------------------------------------------------------- projection
    def project_image_to_ground(self, im_points, projection_type='HAE', hae0=None, gref=None, ugpn=None,
                                tolerance=1e-6, max_iterations=20):
        """Pixels [..., 2] (row, col; fractional) -> ECF [..., 3] (m) of the points that appear there, on the
        surface at height hae0 above the WGS-84 ellipsoid (projection_type 'HAE'; default the scene center
        point's height) or on the plane through gref with unit normal ugpn ('PLANE'; default the scene center
        point and the ellipsoid normal there, or the focus plane normal of a PFA image). NaN where the pixel's
        range and range rate do not reach the surface."""
        return image_to_ground(self, im_points, projection_type, hae0, gref, ugpn, tolerance, max_iterations)

    def project_image_to_ground_geo(self, im_points, **kw):
        """As project_image_to_ground -> [..., 3] latitude, longitude (degrees), height (m)."""
        return _geodetic(self.project_image_to_ground(im_points, **kw))

    def project_ground_to_image(self, coords, tolerance=1e-4, max_iterations=20):
        """ECF points [..., 3] -> (pixels [..., 2] (row, col), the residual ground distance [...] (m), the
        iterations taken)."""
        return ground_to_image(self, coords, tolerance, max_iterations)

    def project_ground_to_image_geo(self, llh, **kw):
        return self.project_ground_to_image(_ecf(llh), **kw)


def read(sicd):
    """SICD metadata of: a SICD object (returned as it is), XML bytes or an XML string, the path of a .xml file
    (the metadata alone), or the path of a SICD NITF. An object with a to_xml_bytes method (another library's
    SICD type) is converted through its XML."""
    import os
    if isinstance(sicd, SICD):
        return sicd
    if isinstance(sicd, bytes) or (isinstance(sicd, str) and sicd.lstrip().startswith('<')):
        return SICD.from_xml(sicd)
    if isinstance(sicd, (str, os.PathLike)):
        path = os.fspath(sicd)
        if path.lower().endswith('.xml'):
            with open(path, 'rb') as fh:
                return SICD.from_xml(fh.read())
        from . import _nitf
        xml = _nitf.read(path).xml('SICD')
        if not xml:
            raise ValueError(f'{path}: no SICD XML data extension segment in the file')
        return SICD.from_xml(xml[0])
    if hasattr(sicd, 'to_xml_bytes'):
        try:
            return SICD.from_xml(sicd.to_xml_bytes(tag='SICD'))
        except TypeError:
            return SICD.from_xml(sicd.to_xml_bytes())
    raise TypeError(f'not a SICD: {type(sicd).__name__}')


# ---------------------------------------------------------------------------------------------- derived fields
def _polyval(c, t):
    return np.polynomial.polynomial.polyval(t, np.asarray(c, np.float64))


def _arp(sm, t, order=0):
    """The aperture reference point polynomial's value [..., 3] (ECF) at times t, or its order-th time derivative."""
    out = []
    for k in 'XYZ':
        c = getattr(sm.Position.ARPPoly, k).get_array()
        out.append(_polyval(np.polynomial.polynomial.polyder(c, order) if order else c, np.asarray(t, np.float64)))
    return np.stack(out, -1)


def weight_function(name, parameters=None, size=WEIGHT_SIZE):
    """Samples of a named aperture weighting (Grid.Row/Col.WgtType) across the processed band, peak about 1 (512;
    32 ones for UNIFORM), or None for a name without a closed form here. TAYLOR: NBAR and SLL (dB, either sign)."""
    from scipy.signal import windows
    name = str(name).upper()
    p = parameters or {}
    if name == 'TAYLOR':
        nbar, sll = int(float(p.get('NBAR', 4))), abs(float(p.get('SLL', -30)))
        return windows.taylor(size, nbar=nbar, sll=sll, norm=True, sym=True)
    if name == 'HANNING':
        return windows.general_hamming(size, 0.5, sym=True)
    if name == 'HAMMING':
        try:
            a = float(next(iter(p.values()))) if p else 0.54
        except ValueError:
            a = 0.54
        return windows.general_hamming(size, a, sym=True)
    if name == 'KAISER':
        try:
            beta = float(next(iter(p.values()))) if p else 14.0
        except ValueError:
            beta = 14.0
        return windows.kaiser(size, beta, sym=True)
    if name == 'UNIFORM':
        return np.ones(32)
    return None


def scpcoa(scp, t, arp, vel, acc):
    """The SCPCOA block's values (Volume 1): scp [3] ECF, t the center of aperture time, arp, vel, acc [3] the
    aperture reference point's position, velocity and acceleration then -> dict in the block's element order."""
    scp, arp, vel = (np.asarray(v, np.float64) for v in (scp, arp, vel))
    los = scp - arp
    rng = float(np.linalg.norm(los))
    ulos, uvel = los / rng, _unit(vel)
    left = np.cross(_unit(arp), uvel)
    look = 1.0 if left @ ulos > 0 else -1.0                # +1: the scene is left of the track
    etp = _up(scp)                                          # the vertical of the earth tangent plane at the SCP
    dca = np.degrees(np.arccos(np.clip(uvel @ ulos, -1, 1)))
    graze = np.degrees(np.arcsin(np.clip(-ulos @ etp, -1, 1)))
    ugpx = _unit(-ulos - (-ulos @ etp) * etp)               # ground range direction, from the SCP toward the radar
    ugpy = np.cross(etp, ugpx)
    spn = _unit(look * np.cross(uvel, ulos))                # slant plane normal, upward
    north = _unit(np.array([0.0, 0.0, 1.0]) - etp[2] * etp)
    east = np.cross(north, etp)
    lay = etp - spn / (etp @ spn)                           # layover direction in the tangent plane
    return dict(SCPTime=float(t), ARPPos=arp, ARPVel=vel, ARPAcc=np.asarray(acc, np.float64),
                SideOfTrack='L' if look > 0 else 'R', SlantRange=rng,
                GroundRange=float(np.linalg.norm(scp) * np.arccos(np.clip(_unit(arp) @ _unit(scp), -1, 1))),
                DopplerConeAng=float(dca), GrazeAng=float(graze), IncidenceAng=float(90.0 - graze),
                TwistAng=float(-np.degrees(np.arcsin(np.clip(ugpy @ spn, -1, 1)))),
                SlopeAng=float(np.degrees(np.arccos(np.clip(etp @ spn, -1, 1)))),
                AzimAng=float(np.degrees(np.arctan2(ugpx @ east, ugpx @ north)) % 360.0),
                LayoverAng=float(np.degrees(np.arctan2(lay @ east, lay @ north)) % 360.0))


def derive(sm, corners=True):
    """Fill, in place, the fields the standard defines from others (see the module docstring): GeoData.SCP.LLH,
    SCPCOA, Grid.Row/Col.WgtFunct where a named window has none, and (corners) GeoData.ImageCorners. -> sm."""
    root = sm.element
    scp = sm.GeoData.SCP.ECF.get_array()
    llh = _geodetic(scp)
    e = ET.Element('LLH')
    for k, v in zip(('Lat', 'Lon', 'HAE'), llh):
        ET.SubElement(e, k).text = text(float(v))
    insert(root.find('GeoData/SCP'), e)
    t = float(sm.Grid.TimeCOAPoly.get_array()[0, 0])
    vals = scpcoa(scp, t, _arp(sm, t), _arp(sm, t, 1), _arp(sm, t, 2))
    insert(root, el('SCPCOA', [el(k, v) for k, v in vals.items()]))
    for d in ('Row', 'Col'):
        ax = getattr(sm.Grid, d)
        if ax.WgtType is not None and ax.WgtFunct is None:
            w = weight_function(ax.WgtType.WindowName, ax.WgtType.Parameters)
            if w is not None:
                f = ET.Element('WgtFunct', size=str(len(w)))
                for i, v in enumerate(w):
                    ET.SubElement(f, 'Wgt', index=str(i + 1)).text = text(float(v))
                insert(root.find(f'Grid/{d}'), f)
    if corners:
        derive_corners(sm)
    return sm


def derive_corners(sm):
    """Set GeoData.ImageCorners, in place: the centers of the four corner pixels (first row first column, then
    clockwise in the image) projected to the scene center point's height."""
    nr, nc = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
    fr, fc = int(sm.ImageData.FirstRow or 0), int(sm.ImageData.FirstCol or 0)
    rc = np.array([[fr, fc], [fr, fc + nc - 1], [fr + nr - 1, fc + nc - 1], [fr + nr - 1, fc]], np.float64)
    ll = _geodetic(image_to_ground(sm, rc))
    c = ET.Element('ImageCorners')
    for i, (name, p) in enumerate(zip(('FRFC', 'FRLC', 'LRLC', 'LRFC'), ll)):
        icp = ET.SubElement(c, 'ICP', index=f'{i + 1}:{name}')
        ET.SubElement(icp, 'Lat').text = text(float(p[0]))
        ET.SubElement(icp, 'Lon').text = text(float(p[1]))
    insert(sm.element.find('GeoData'), c)
    return sm


def specification(sm):
    """(version, date, namespace) of the SICD specification the metadata follow, as the NITF data extension
    segment's subheader states them (DESSHSV, DESSHSD, DESSHTN)."""
    v = sm.version
    return v, SPECIFICATION.get(v, SPECIFICATION['1.3.0']), str(sm.namespace)


def build(f):
    """SICD 1.3.0 metadata of a product imaged on a plane, with its derived fields. f: dict(
    collector, core_name, mode ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP'), application, created (the creation
    time, 'YYYY-MM-DDThh:mm:ss.ffffffZ'), rows, cols, scp_pixel (row, col), scp [3] (ECF, m), plane ('GROUND' or
    'SLANT'), t_coa (s), row and col (each dict(u [3] ECF, ss (m), wid (m), bw, kctr, k1, k2, dk (cycles/m), window
    (None, 'hann', 'taylor' or 'sva'))), start (the collection start, as created), duration (s), arp [3, n] (the
    polynomial coefficients of X, Y and Z in t), f0, f1 (Hz), pol ('V:V' or 'UNKNOWN'), t0, t1 (the processed
    times, s), autofocus ('NO', 'GLOBAL' or 'SV'))."""
    wgt = {None: [el('WindowName', 'UNIFORM')], 'hann': [el('WindowName', 'HANNING')],
           'taylor': [el('WindowName', 'TAYLOR'), el('Parameter', '4', name='NBAR'), el('Parameter', '-35', name='SLL')],
           'sva': [el('WindowName', 'SVA')]}

    def axis(tag, a):
        return el(tag, [el('UVectECF', np.asarray(a['u'], np.float64)), el('SS', float(a['ss'])),
                        el('ImpRespWid', float(a['wid'])), el('Sgn', -1), el('ImpRespBW', float(a['bw'])),
                        el('KCtr', float(a['kctr'])), el('DeltaK1', float(a['k1'])), el('DeltaK2', float(a['k2'])),
                        poly('DeltaKCOAPoly', [[float(a['dk'])]]), el('WgtType', wgt[a['window']])])
    pol = f['pol']
    arp = np.asarray(f['arp'], np.float64)
    root = el('SICD', [
        el('CollectionInfo', [el('CollectorName', f['collector']), el('CoreName', f['core_name']),
                              el('CollectType', 'MONOSTATIC'), el('RadarMode', [el('ModeType', f['mode'])]),
                              el('Classification', 'UNCLASSIFIED')]),
        el('ImageCreation', [el('Application', f['application']), el('DateTime', f['created'])]),
        el('ImageData', [el('PixelType', 'RE32F_IM32F'), el('NumRows', int(f['rows'])), el('NumCols', int(f['cols'])),
                         el('FirstRow', 0), el('FirstCol', 0),
                         el('FullImage', [el('NumRows', int(f['rows'])), el('NumCols', int(f['cols']))]),
                         el('SCPPixel', [el('Row', int(f['scp_pixel'][0])), el('Col', int(f['scp_pixel'][1]))])]),
        el('GeoData', [el('EarthModel', 'WGS_84'), el('SCP', [el('ECF', np.asarray(f['scp'], np.float64))])]),
        el('Grid', [el('ImagePlane', f['plane']), el('Type', 'PLANE'), poly('TimeCOAPoly', [[float(f['t_coa'])]]),
                    axis('Row', f['row']), axis('Col', f['col'])]),
        el('Timeline', [el('CollectStart', f['start']), el('CollectDuration', float(f['duration']))]),
        el('Position', [el('ARPPoly', [poly(k, arp[i]) for i, k in enumerate('XYZ')])]),
        el('RadarCollection', [el('TxFrequency', [el('Min', float(f['f0'])), el('Max', float(f['f1']))]),
                               el('TxPolarization', pol[0] if pol != 'UNKNOWN' else 'OTHER'),
                               el('RcvChannels', [el('ChanParameters', [el('TxRcvPolarization', pol)], index=1)],
                                  size=1)]),
        el('ImageFormation', [el('RcvChanProc', [el('NumChanProc', 1), el('ChanIndex', 1)]),
                              el('TxRcvPolarizationProc', pol), el('TStartProc', float(f['t0'])),
                              el('TEndProc', float(f['t1'])),
                              el('TxFrequencyProc', [el('MinProc', float(f['f0'])), el('MaxProc', float(f['f1']))]),
                              el('ImageFormAlgo', 'OTHER'), el('STBeamComp', 'NO'), el('ImageBeamComp', 'NO'),
                              el('AzAutofocus', f['autofocus']), el('RgAutofocus', 'NO')])])
    return derive(SICD(root))


# ---------------------------------------------------------------------------------------------- projection
class _Model:
    """What the projection needs of a SICD, read once: the grid, the polynomials, and the grid type's map from
    image coordinates to range and range rate at the center of aperture (Volume 3, section 4)."""

    def __init__(self, sm):
        g = sm.Grid
        self.scp = sm.GeoData.SCP.ECF.get_array()
        self.pixel = np.array([float(sm.ImageData.SCPPixel.Row) - float(sm.ImageData.FirstRow or 0),
                               float(sm.ImageData.SCPPixel.Col) - float(sm.ImageData.FirstCol or 0)])
        self.ss = np.array([float(g.Row.SS), float(g.Col.SS)])
        self.urow, self.ucol = g.Row.UVectECF.get_array(), g.Col.UVectECF.get_array()
        self.tcoa = np.atleast_2d(g.TimeCOAPoly.get_array())
        self.arp = [getattr(sm.Position.ARPPoly, k).get_array() for k in 'XYZ']
        self.varp = [np.polynomial.polynomial.polyder(c) if len(c) > 1 else np.zeros(1) for c in self.arp]
        self.kind = str(g.Type)
        algo = str(sm.ImageFormation.ImageFormAlgo) if sm.ImageFormation is not None else ''
        self.fpn = None
        if self.kind == 'RGAZIM' and algo == 'PFA':
            self.kind = 'PFA'
            self.polar = sm.PFA.PolarAngPoly.get_array()
            self.ksf = sm.PFA.SpatialFreqSFPoly.get_array()
            self.fpn = sm.PFA.FPN.get_array()
        elif self.kind == 'RGAZIM' and algo == 'RGAZCOMP':
            self.kind = 'RGAZCOMP'
            self.azsf = float(sm.RgAzComp.AzSF)
        elif self.kind == 'RGZERO':
            inca = sm.RMA.INCA if sm.RMA is not None else None
            if inca is None:
                raise ValueError('an RGZERO grid needs the RMA.INCA parameters to be projected')
            self.r_ca, self.tca = float(inca.R_CA_SCP), inca.TimeCAPoly.get_array()
            self.drsf = np.atleast_2d(inca.DRateSFPoly.get_array())
        elif self.kind not in ('PLANE', 'XRGYCR', 'XCTYAT'):
            raise ValueError(f'the projection of a SICD grid of type {self.kind} '
                             f'(image formation {algo or "unknown"}) is not implemented')
        # the scene center point's slant plane normal at its center of aperture, for the scene to image iteration
        t0 = float(self.tcoa[0, 0])
        a0, v0 = self.pos(t0), self.vel(t0)
        look = np.sign(np.cross(a0, v0) @ (self.scp - a0))
        self.spn = _unit(look * np.cross(v0, self.scp - a0))

    def pos(self, t):
        return np.stack([_polyval(c, t) for c in self.arp], -1)

    def vel(self, t):
        return np.stack([_polyval(c, t) for c in self.varp], -1)

    def coa(self, rc):
        """Pixels [n, 2] -> (range [n], range rate [n], ARP position [n, 3], velocity [n, 3]) at each pixel's center
        of aperture time."""
        x = (rc[:, 0] - self.pixel[0]) * self.ss[0]
        y = (rc[:, 1] - self.pixel[1]) * self.ss[1]
        t = np.polynomial.polynomial.polyval2d(x, y, self.tcoa)
        a, v = self.pos(t), self.vel(t)
        if self.kind in ('PLANE', 'XRGYCR', 'XCTYAT'):
            d = a - (self.scp + x[:, None] * self.urow + y[:, None] * self.ucol)
            r = np.linalg.norm(d, axis=1)
            return r, np.einsum('ij,ij->i', v, d) / r, a, v
        if self.kind == 'RGZERO':
            r_ca, t_ca = self.r_ca + x, _polyval(self.tca, y)
            v2 = np.sum(self.vel(t_ca) ** 2, -1)
            drsf = np.polynomial.polynomial.polyval2d(x, y, self.drsf)
            dt = t - t_ca
            r = np.sqrt(r_ca ** 2 + drsf * v2 * dt ** 2)
            return r, drsf * v2 * dt / r, a, v
        d = a - self.scp
        r0 = np.linalg.norm(d, axis=1)
        rd0 = np.einsum('ij,ij->i', v, d) / r0
        if self.kind == 'RGAZCOMP':
            return r0 + x, rd0 - np.linalg.norm(v, axis=1) * self.azsf * y, a, v
        th = _polyval(self.polar, t)                                   # PFA: the polar angle and its scale factor
        dth = _polyval(np.polynomial.polynomial.polyder(self.polar), t)
        ksf = _polyval(self.ksf, th)
        dksf = _polyval(np.polynomial.polynomial.polyder(self.ksf), th) if len(self.ksf) > 1 else 0.0
        ka, kc = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        return r0 + ksf * ka, rd0 + (dksf * ka + ksf * kc) * dth, a, v


def _to_plane(r, rdot, arp, varp, gref, uz):
    """The point [n, 3] of the plane through gref (one point or one per pixel) with unit normal uz at range r and
    range rate rdot from the aperture (Volume 3: the precise projection along the contour to a plane)."""
    az = np.einsum('ij,ij->i', arp - gref, np.broadcast_to(uz, arp.shape))
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.sqrt(r * r - az * az)                   # ground distance from the aperture's nadir on the plane
        vz = varp @ uz if uz.ndim == 1 else np.einsum('ij,ij->i', varp, uz)
        vx = np.sqrt(np.sum(varp * varp, -1) - vz * vz)
        ux = (varp - vz[:, None] * uz) / vx[:, None]
        uy = np.cross(np.broadcast_to(uz, ux.shape), ux)
        cos_az = (-rdot + vz * az / r) / (vx * g / r)
        cos_az = np.where(np.abs(cos_az) > 1, np.nan, cos_az)
        look = np.sign(np.einsum('ij,ij->i', np.cross(arp - gref, varp), np.broadcast_to(uz, arp.shape)))
        sin_az = look * np.sqrt(1 - cos_az * cos_az)
    return arp - az[:, None] * uz + (g * cos_az)[:, None] * ux + (g * sin_az)[:, None] * uy


def _model(sm):
    return sm if isinstance(sm, _Model) else _Model(sm)


def image_to_ground(sm, im_points, projection_type='HAE', hae0=None, gref=None, ugpn=None, tolerance=1e-6,
                    max_iterations=20):
    """SICD.project_image_to_ground (see there). sm: a SICD (or a _Model of one)."""
    m = _model(sm)
    rc = np.asarray(im_points, np.float64)
    shape = rc.shape[:-1]
    r, rdot, a, v = m.coa(rc.reshape(-1, 2))
    kind = str(projection_type).upper()
    if kind == 'PLANE':
        gref = m.scp if gref is None else np.asarray(gref, np.float64).reshape(3)
        uz = (m.fpn if m.fpn is not None else _up(gref)) if ugpn is None else _unit(np.asarray(ugpn, np.float64).reshape(3))
        return _to_plane(r, rdot, a, v, gref, uz).reshape(shape + (3,))
    if kind != 'HAE':
        raise ValueError(f"projection_type {projection_type!r}: 'HAE' or 'PLANE'")
    ref_hae = float(_geodetic(m.scp)[2])
    hae0 = ref_hae if hae0 is None else float(hae0)
    uz = _up(m.scp)
    g = np.broadcast_to(m.scp - (ref_hae - hae0) * uz, a.shape).copy()
    # planes tangent to the height surface, each through the last point moved to the height (Volume 3: projection
    # to a surface of constant height); the plane's normal follows the point, so the iteration ends on the surface
    uzn = np.broadcast_to(uz, a.shape).copy()
    p = g
    for _ in range(int(max_iterations)):
        p = _to_plane(r, rdot, a, v, g, uzn)
        ok = np.isfinite(p).all(1)
        if not ok.any():
            break
        llh = _geodetic(p[ok])
        dh = llh[:, 2] - hae0
        uzn[ok] = _up(p[ok])
        g[ok] = p[ok] - dh[:, None] * uzn[ok]
        if np.max(np.abs(dh)) < tolerance:
            break
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p).all(1)
    if ok.any():
        llh = _geodetic(p[ok])
        llh[:, 2] = hae0
        out[ok] = _ecf(llh)
    return out.reshape(shape + (3,))


def ground_to_image(sm, coords, tolerance=1e-4, max_iterations=20):
    """SICD.project_ground_to_image (see there): Volume 3's scene to image iteration. The point is carried to the
    image plane along the scene center point's slant plane normal, the pixel found there is projected back to the
    plane through the point, and the miss is fed back until it is under tolerance (m)."""
    m = _model(sm)
    c = np.asarray(coords, np.float64)
    shape = c.shape[:-1]
    c = c.reshape(-1, 3)
    uipn = _unit(np.cross(m.urow, m.ucol))
    cs = float(m.urow @ m.ucol)
    inv = np.array([[1.0, -cs], [-cs, 1.0]]) / (1.0 - cs * cs)       # oblique axes -> coordinates along each
    sf = float(m.spn @ uipn)
    ugpn = m.fpn if m.fpn is not None else _up(m.scp)
    g = c.copy()
    rc = np.zeros((len(c), 2))
    miss = np.zeros(len(c))
    it = 0
    for it in range(1, int(max_iterations) + 1):
        dist = ((m.scp - g) @ uipn) / sf
        d = g + dist[:, None] * m.spn - m.scp
        xy = np.stack([d @ m.urow, d @ m.ucol], -1) @ inv
        rc = xy / m.ss + m.pixel
        r, rdot, a, v = m.coa(rc)
        diff = c - _to_plane(r, rdot, a, v, g, ugpn)
        miss = np.linalg.norm(diff, axis=1)
        g = g + diff
        left = miss[np.isfinite(miss)]
        if left.size == 0 or left.max() <= tolerance:
            break
    return rc.reshape(shape + (2,)), miss.reshape(shape), np.full(shape, it, np.int16) if shape else it
