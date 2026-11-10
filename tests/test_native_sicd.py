"""SICD metadata and projection without sarpy (fastsar._sicdxml) against sarpy as the independent reference: every
field of the metadata products.sicd_meta builds against sarpy's SICDType built from the same product and derived by
sarpy, for right and left looks, ground and slant planes; the projection (image to ground on a plane and at a height,
ground to image) against sarpy.geometry.point_projection on those products and on vendor-style SICDs with polar
format, range azimuth compression and zero-Doppler grids built with sarpy; and the metadata object's access patterns.
Needs sarpy."""
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fastsar import io, products, _sicdxml

pytestmark = pytest.mark.needs('sarpy')
C = 299792458.0


def product(side='R', plane='ground', nx=96, ny=80, lat=35.0, lon=-106.0, h=1500.0, heading=30.0, graze=35.0,
            squint=8.0, base=False, window=('taylor', 'taylor')):
    """A form_cphd-style product dict (a zero image) of a collection from 600 km on a curved track, looking to
    `side`, squinted, on the ground plane or the slant plane."""
    s0 = io.geodetic_to_ecf(lat, lon, h)
    phi, lam = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.cross(up, east)
    hd = np.radians(heading)
    along = np.sin(hd) * east + np.cos(hd) * north
    cross = np.cross(along, up) * (1 if side == 'R' else -1)          # from the track toward the scene
    P, R0, v, g = 400, 600e3, 7500.0, np.radians(graze)
    t = 1.0 + np.arange(P) / 2000.0
    s = (t - t.mean()) * v + R0 * np.tan(np.radians(squint))
    apc = (s0 - cross * R0 * np.cos(g) + up * R0 * np.sin(g) + s[:, None] * along
           - 4.0 * ((t - t.mean()) ** 2)[:, None] * up)
    mid = apc[P // 2] - s0                                            # the local frame of io.read_cphd
    los_h = mid - (mid @ up) * up
    yhat = -los_h / np.linalg.norm(los_h)
    R = np.stack([np.cross(yhat, up), yhat, up])
    meta = dict(R=R, origin=s0, tx=(apc - s0) @ R.T, rcv=(apc - s0) @ R.T, tx_time=t, rcv_time=t + 0.004,
                polarization='VV', mode='SPOTLIGHT', collector='SIM-1', core_name='CORE_123',
                start='2026-01-02T03:04:05.250000Z')
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if plane == 'slant':
        los = -mid @ R.T / np.linalg.norm(mid)
        vel = (apc[P // 2 + 1] - apc[P // 2 - 1]) @ R.T
        n = np.cross(los, vel)
        e2, e1 = los, np.cross(los, n / np.linalg.norm(n))
        e1 = e1 / np.linalg.norm(e1) * np.sign(e1 @ vel)
    spx, spy = 0.4, 0.5
    out = dict(image=np.zeros((nx, ny), np.complex64), meta=meta, e1=e1, e2=e2, spx=spx, spy=spy,
               origin=-(nx / 2) * spx * e1 - (ny / 2) * spy * e2 + np.array([3.0, -2.0, 0.0]), band=(9.4e9, 9.9e9),
               bandwidth=(1.9, 1.7), window=window, mode='spotlight', plane=plane)
    if base:
        out.update(baseband=True, support=((-0.8, 0.9), (-0.7, 0.75)))
    return out


def sarpy_sicd(out):
    """The reference: sarpy's SICDType built from a product with sarpy's classes and derived by sarpy."""
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
    from fastsar.io import local_to_ecf
    from fastsar import __version__
    C = 299792458.0
    meta = out['meta']
    img = np.asarray(out['image'])
    nx, ny = img.shape
    R = np.asarray(meta['R'], np.float64)
    e1, e2 = np.asarray(out['e1'], np.float64), np.asarray(out['e2'], np.float64)
    i0, j0 = (int(v) for v in out['scp_pixel']) if out.get('scp_pixel') is not None else (nx // 2, ny // 2)
    scp_l = np.asarray(out['origin'], np.float64) + i0 * out['spx'] * e1 + j0 * out['spy'] * e2
    ant = 0.5 * (np.asarray(meta['tx'], np.float64) + np.asarray(meta['rcv'], np.float64))
    t = np.asarray(meta['tx_time'], np.float64)
    p0, p1 = (int(v) for v in (out.get('pulses') or (0, len(t))))     # the pulses used
    m = (p0 + p1) // 2
    away = (scp_l - ant[m]) @ e2 > 0               # e2 points away from the radar: rows follow it
    # rows along range away from the radar, columns so that row x column is upward (e1 x e2 is)
    a = img.T[:, ::-1] if away else img.T[::-1]
    srow = 1 if away else -1
    urow, ucol = srow * e2 @ R, -srow * e1 @ R
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
    win = tuple(None if w == 'uniform' else w for w in (out.get('window') or (None, None)))
    width = {'taylor': 1.184, 'hann': 1.441, None: 0.886, 'sva': 0.886}           # 3 dB width times bandwidth
    base = bool(out.get('baseband'))
    sup = out.get('support') if base else None      # the spectrum's limits along e1 and e2, on the SICD's axes
    sup_r, sup_c = (None, None) if sup is None else (
        tuple(srow * v for v in sup[1])[::srow], tuple(-srow * v for v in sup[0])[::-srow])

    def axis(u, ss, bw, kctr, w, lim):
        bw = min(bw, 1.0 / ss)
        wgt = {None: WgtTypeType(WindowName='UNIFORM'), 'hann': WgtTypeType(WindowName='HANNING'),
               'taylor': WgtTypeType(WindowName='TAYLOR', Parameters={'NBAR': '4', 'SLL': '-35'}),
               'sva': WgtTypeType(WindowName='SVA')}[w]             # apodized after formation: no one aperture weighting
        if lim is not None:                                         # the band as measured from the collection
            k1, k2 = max(lim[0], -0.5 / ss), min(lim[1], 0.5 / ss)
            dk = 0.5 * (k1 + k2)
        else:                                                       # its center in the sampled image, nominal width
            dk = 0.0 if base else (kctr * ss + 0.5) % 1.0 / ss - 0.5 / ss
            k1, k2 = dk - bw / 2, dk + bw / 2
        return DirParamType(UVectECF=u, SS=ss, ImpRespWid=width[w] / bw, Sgn=-1, ImpRespBW=bw, KCtr=kctr,
                            DeltaK1=k1, DeltaK2=k2, DeltaKCOAPoly=[[dk]], WgtType=wgt)
    pol = meta.get('polarization')
    pol = f'{pol[0]}:{pol[1]}' if pol else 'UNKNOWN'
    mode = str(meta.get('mode') or '').upper()
    if mode not in ('SPOTLIGHT', 'STRIPMAP', 'DYNAMIC STRIPMAP'):
        mode = 'SPOTLIGHT' if out.get('mode') == 'spotlight' else 'DYNAMIC STRIPMAP'
    focus = out.get('autofocus')                  # the mode process_cphd ran, or form_cphd's phase-history estimate
    focus = ('SV' if 'variant' in str(focus) else 'GLOBAL') if focus else (
        'GLOBAL' if out.get('phase_error') is not None else 'NO')
    sm = SICDType(
        CollectionInfo=CollectionInfoType(CollectorName=meta.get('collector') or 'UNKNOWN',
                                          CoreName=meta.get('core_name') or 'UNKNOWN', CollectType='MONOSTATIC',
                                          RadarMode=RadarModeType(ModeType=mode), Classification='UNCLASSIFIED'),
        ImageCreation=ImageCreationType(Application=f'FastSAR {__version__}',
                                        DateTime=np.datetime64(datetime.now(timezone.utc).replace(tzinfo=None), 'us')),
        ImageData=ImageDataType(PixelType='RE32F_IM32F', NumRows=ny, NumCols=nx, FirstRow=0, FirstCol=0,
                                FullImage=(ny, nx),
                                SCPPixel=(j0, nx - 1 - i0) if away else (ny - 1 - j0, i0)),
        GeoData=GeoDataType(EarthModel='WGS_84', SCP=SCPType(ECF=scp)),
        Grid=GridType(ImagePlane='SLANT' if out.get('plane') == 'slant' else 'GROUND', Type='PLANE',
                      TimeCOAPoly=[[float(t[m])]], Row=axis(urow, out['spy'], bw_r, kr, win[1], sup_r),
                      Col=axis(ucol, out['spx'], bw_c, kc, win[0], sup_c)),
        Timeline=TimelineType(CollectStart=np.datetime64(str(meta.get('start')).rstrip('Z'), 'us'),
                              CollectDuration=float(t[-1])),
        Position=PositionType(ARPPoly=arp),
        RadarCollection=RadarCollectionType(TxFrequency=TxFrequencyType(Min=f0, Max=f1),
                                            TxPolarization=pol[0] if pol != 'UNKNOWN' else 'OTHER',
                                            RcvChannels=[ChanParametersType(TxRcvPolarization=pol, index=1)]),
        ImageFormation=ImageFormationType(RcvChanProc=RcvChanProcType(NumChanProc=1, ChanIndices=[1]),
                                          TxRcvPolarizationProc=pol, TStartProc=float(t[p0]), TEndProc=float(t[p1 - 1]),
                                          TxFrequencyProc=TxFrequencyProcType(MinProc=f0, MaxProc=f1),
                                          ImageFormAlgo='OTHER', STBeamComp='NO', ImageBeamComp='NO',
                                          AzAutofocus=focus,
                                          RgAutofocus='NO'))
    sm.derive()
    sm.define_geo_image_corners(override=True)
    return sm, a


def leaves(xml):
    """[(path, text)] of the leaf elements of an XML document, namespaces dropped, in document order."""
    out = []

    def walk(e, path):
        tag = e.tag.rsplit('}', 1)[-1]
        if len(e) == 0:
            out.append((f'{path}/{tag}', (e.text or '').strip()))
        for c in e:
            walk(c, f'{path}/{tag}')
    walk(ET.fromstring(xml), '')
    return out


# sarpy takes the vertical at the scene center point as the gradient of the ellipsoid's equation there, the standard
# as the geodetic vertical; at 1.5 km height the two differ by 2e-5 degrees, which the angles below inherit
ANGLES = ('GrazeAng', 'IncidenceAng', 'TwistAng', 'SlopeAng', 'AzimAng', 'LayoverAng', 'DopplerConeAng')


@pytest.mark.parametrize('side', ['R', 'L'])
@pytest.mark.parametrize('plane', ['ground', 'slant'])
@pytest.mark.parametrize('variant', ['formed', 'baseband', 'hann-uniform', 'sva'])
def test_metadata_against_sarpy(side, plane, variant):
    kw = {'formed': {}, 'baseband': dict(base=True), 'hann-uniform': dict(window=('hann', 'uniform')),
          'sva': dict(window=('sva', 'sva'))}[variant]
    out = product(side, plane, **kw)
    sm, a = products.sicd_meta(out)
    ref, b = sarpy_sicd(out)
    assert np.array_equal(a, b) and sm.SCPCOA.SideOfTrack == side
    mine, theirs = leaves(sm.to_xml_bytes()), leaves(ref.to_xml_bytes(tag='SICD'))
    assert [k for k, _ in mine] == [k for k, _ in theirs]             # the same elements in the same order
    worst = dict(angle=0.0, number=0.0)
    for (key, x), (_, y) in zip(mine, theirs):
        name = key.rsplit('/', 1)[-1]
        if key.endswith('ImageCreation/DateTime'):
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            assert x == y, (key, x, y)
            continue
        if name in ANGLES:
            worst['angle'] = max(worst['angle'], abs(fx - fy))
        else:
            worst['number'] = max(worst['number'], abs(fx - fy) / max(abs(fy), 1e-30) if fy else abs(fx))
            assert fx == pytest.approx(fy, rel=1e-9, abs=1e-12), (key, x, y)
    assert worst['angle'] < 1e-4, worst
    print(f'SICD metadata ({side}, {plane}, {variant}): {len(mine)} leaf elements, numbers within '
          f'{worst["number"]:.1e} (relative) of sarpy, angles within {worst["angle"]:.1e} degrees')


def test_metadata_object():
    """The access patterns of the metadata object, its XML round trip, copies, edits and pickling."""
    sm = products.sicd_meta(product())[0]
    assert sm.version == '1.3.0' and sm.Grid.Type == 'PLANE' and sm.Grid.Row.SS == 0.5
    assert isinstance(sm.ImageData.NumRows, int) and sm.ImageData.NumRows == 80 and sm.ImageData.SCPPixel.Row == 40
    assert sm.GeoData.SCP.ECF.get_array().shape == (3,) and sm.GeoData.ImageCorners.get_array().shape == (4, 2)
    assert sm.Grid.TimeCOAPoly.get_array().shape == (1, 1) and sm.Grid.TimeCOAPoly[0, 0] == sm.SCPCOA.SCPTime
    assert sm.Position.ARPPoly.X.get_array().shape == (6,) and sm.Grid.Row.WgtType.WindowName == 'TAYLOR'
    assert sm.Grid.Row.WgtType.Parameters == {'NBAR': '4', 'SLL': '-35'} and len(sm.Grid.Row.WgtFunct.Wgt) == 512
    assert sm.RadarCollection.RcvChannels[0].TxRcvPolarization == 'V:V' and sm.ImageFormation.RcvChanProc.ChanIndices == [1]
    assert sm.CollectionInfo.CoreName == 'CORE_123' and sm.CollectionInfo.Parameters == {}
    assert sm.Radiometric is None and sm.PFA is None and sm.Timeline.CollectStart == '2026-01-02T03:04:05.250000Z'
    again = _sicdxml.read(sm.to_xml_bytes())
    assert leaves(again.to_xml_bytes()) == leaves(sm.to_xml_bytes())
    c = sm.copy()
    c.Grid.Type, c.SCPCOA = 'RGZERO', None
    assert c.Grid.Type == 'RGZERO' and c.SCPCOA is None and sm.Grid.Type == 'PLANE' and sm.SCPCOA is not None
    p = pickle.loads(pickle.dumps(sm))
    assert type(p) is type(sm) and p.to_xml_bytes() == sm.to_xml_bytes()
    with pytest.raises(ValueError, match='root element'):
        _sicdxml.read(b'<CPHD/>')


def vendor(sm, kind):
    """A vendor-style SICD (sarpy SICDType) on the collection and the slant-plane grid of sm: a polar format image
    ('PFA'), a range azimuth compression image ('RGAZCOMP') or a zero-Doppler image ('RGZERO'), each with the
    parameters that make its range and range rate follow the grid, and a center of aperture time that varies over
    the image."""
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.PFA import PFAType
    from sarpy.io.complex.sicd_elements.RgAzComp import RgAzCompType
    from sarpy.io.complex.sicd_elements.RMA import RMAType, INCAType
    s = SICDType.from_xml_string(sm.to_xml_bytes().decode())
    t0 = float(sm.SCPCOA.SCPTime)
    arp, vel, scp = sm.SCPCOA.ARPPos.get_array(), sm.SCPCOA.ARPVel.get_array(), sm.GeoData.SCP.ECF.get_array()
    rng, v = np.linalg.norm(arp - scp), np.linalg.norm(vel)
    ucol = s.Grid.Col.UVectECF.get_array()
    along = float(vel @ ucol)                       # the velocity along the columns (cross range), signed
    if kind == 'PFA':
        s.Grid.Type, s.ImageFormation.ImageFormAlgo = 'RGAZIM', 'PFA'
        s.Grid.TimeCOAPoly = [[t0, 2e-5], [1e-5, 0.0]]
        ipn = np.cross(s.Grid.Row.UVectECF.get_array(), ucol)
        s.PFA = PFAType(FPN=scp / np.linalg.norm(scp), IPN=ipn, PolarAngRefTime=t0,
                        PolarAngPoly=[0.0, -along / rng, 1e-5], SpatialFreqSFPoly=[1.0, 2e-3, -0.4],
                        Krg1=60.0, Krg2=66.0, Kaz1=-1.0, Kaz2=1.0)
    elif kind == 'RGAZCOMP':
        s.Grid.Type, s.ImageFormation.ImageFormAlgo = 'RGAZIM', 'RGAZCOMP'
        s.Grid.TimeCOAPoly = [[t0, 3e-5], [0.0, 0.0]]
        s.RgAzComp = RgAzCompType(AzSF=along / (v * rng), KazPoly=[0.0, 1.0])
    else:                                           # the product is broadside: closest approach at t0, columns along track
        s.Grid.Type, s.ImageFormation.ImageFormAlgo = 'RGZERO', 'RMA'
        s.Grid.TimeCOAPoly = [[t0, along / v ** 2], [2e-7, 0.0]]
        s.RMA = RMAType(RMAlgoType='OMEGA_K', ImageType='INCA', INCA=INCAType(
            TimeCAPoly=[t0, along / v ** 2, 1e-12], R_CA_SCP=rng, FreqZero=9.65e9,
            DRateSFPoly=[[1.0, 1e-9], [-2e-9, 0.0]]))
    return s


@pytest.mark.parametrize('side', ['R', 'L'])
@pytest.mark.parametrize('kind', ['PLANE ground', 'PLANE slant', 'PFA', 'RGAZCOMP', 'RGZERO'])
def test_projection_against_sarpy(side, kind):
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.geometry import point_projection as pp
    out = product(side, 'ground' if kind == 'PLANE ground' else 'slant', squint=0.0 if kind == 'RGZERO' else 8.0)
    sm = products.sicd_meta(out)[0]
    if kind.startswith('PLANE'):
        ref = SICDType.from_xml_string(sm.to_xml_bytes().decode())
    else:
        ref = vendor(sm, kind)
        sm = _sicdxml.read(ref)                                       # through its XML, as a vendor's file is read
        assert sm.Grid.Type == ref.Grid.Type
    rng = np.random.default_rng(3)
    rc = rng.uniform(-200.0, 300.0, (64, 2))
    h0 = float(sm.GeoData.SCP.LLH.HAE)
    d = {}
    for name, h in (('at the scene height', None), ('300 m above', h0 + 300.0), ('200 m below', h0 - 200.0)):
        g = sm.project_image_to_ground(rc, projection_type='HAE', hae0=h)
        d[name] = np.abs(g - pp.image_to_ground(rc, ref, projection_type='HAE', hae0=h, tolerance=1e-7)).max()
        assert np.abs(io.ecf_to_geodetic(g)[2] - (h0 if h is None else h)).max() < 1e-6
    plane = sm.project_image_to_ground(rc, projection_type='PLANE')
    d['plane'] = np.abs(plane - pp.image_to_ground(rc, ref, projection_type='PLANE')).max()
    gref, n = sm.GeoData.SCP.ECF.get_array() + [10.0, -20.0, 5.0], np.array([0.1, -0.5, 0.86])
    mine = sm.project_image_to_ground(rc, projection_type='PLANE', gref=gref, ugpn=n)
    d['given plane'] = np.abs(mine - pp.image_to_ground(rc, ref, projection_type='PLANE', gref=gref, ugpn=n)).max()
    ground = sm.project_image_to_ground(rc) + rng.normal(0.0, 40.0, (64, 3))
    back, miss, _ = sm.project_ground_to_image(ground, tolerance=1e-7)
    d['ground to image (pixels)'] = np.abs(back - pp.ground_to_image(ground, ref, tolerance=1e-7)[0]).max()
    loop = np.abs(sm.project_ground_to_image(sm.project_image_to_ground(rc), tolerance=1e-7)[0] - rc).max()
    print(f'projection ({kind}, {side}): ' + ', '.join(f'{k} {v:.1e}' for k, v in d.items())
          + f'; image to ground to image {loop:.1e} pixels')
    # the default plane's normal is the geodetic vertical here and the ellipsoid's gradient in sarpy (2e-5 degrees
    # apart), which moves a point 150 m from the scene center by a tenth of a millimeter on that plane; a PFA image's
    # default plane is its focus plane in both
    assert max(d['at the scene height'], d['300 m above'], d['200 m below'], d['given plane']) < 1e-6, d
    assert d['plane'] < (1e-6 if kind == 'PFA' else 1e-3) and d['ground to image (pixels)'] < 1e-6 and loop < 1e-6, d
    assert miss.max() < 1e-6
    # one point, and a grid of points
    one = sm.project_image_to_ground(rc[0])
    assert one.shape == (3,) and np.allclose(one, sm.project_image_to_ground(rc)[0], atol=1e-9)
    assert sm.project_ground_to_image(ground.reshape(8, 8, 3))[0].shape == (8, 8, 2)
    assert sm.project_image_to_ground_geo(rc).shape == (64, 3)
