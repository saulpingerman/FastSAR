"""process_cphd: a simulated spotlight collection (squinted, on a curved track, flown to either side) written as
CPHD and processed to each kind of product, every file read back with sarpy: the targets where sarpy's own
projection of the product puts them, the plane, weighting and layout the metadata state, and the options' errors.
Needs sarpy."""
import dataclasses

import numpy as np
import pytest

import fastsar
from fastsar import quality, sim

pytestmark = pytest.mark.needs('sarpy')

LAT, LON, H0, HEADING = 40.79, -77.86, 351.0, 30.0
TG = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 15.0, 0.0], [-18.0, -22.0, 0.0], [12.0, -9.0, 0.0],
               [-40.0, 35.0, 0.0], [45.0, 28.0, 0.0]])


@pytest.fixture(scope='module', params=[(20.0, False), (-25.0, True)], ids=['right', 'left'])
def cphd(request, tmp_path_factory):
    pytest.importorskip('sarpy')
    squint, left = request.param
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    y = col.ant[:, 1] / np.ptp(col.ant[:, 1])
    ant = col.ant.copy()
    ant[:, 1] -= 10e3 * np.cos(np.radians(35)) * np.tan(np.radians(squint))
    ant[:, 0] += 160.0 * y ** 2 + 120.0 * y ** 3                      # a curved track
    ant[:, 2] += -100.0 * y ** 2 + 6.0 * y
    col = dataclasses.replace(col, ant=ant[::-1].copy() if left else ant)
    d = tmp_path_factory.mktemp('process')
    sim.write_cphd(str(d / 'c.cphd'), col, sim.simulate(col, TG, np.ones(len(TG))), LAT, LON, H0, heading=HEADING,
                   scene=140.0)
    return d, sim.to_ecf(TG, LAT, LON, H0, HEADING)


def read_sicd(path, ecf):
    """The SICD's metadata and pixels, and the point-target measurements at sarpy's projection of the targets."""
    from sarpy.io.complex.converter import open_complex
    from sarpy.geometry import point_projection
    rd = open_complex(str(path))
    sm, d = rd.sicd_meta, rd[:, :]
    rc, _, _ = point_projection.ground_to_image(ecf, sm)
    m = [quality.point_target(d, (round(r), round(c)), sm.Grid.Row.SS, sm.Grid.Col.SS, half=16) for r, c in rc]
    off = np.array([(k['i'] - r, k['j'] - c) for k, (r, c) in zip(m, rc)])
    return sm, d, off, np.array([(k['pslr_az'], k['pslr_rg']) for k in m])


def from_above(sm):
    """Whether the SICD's rows and columns are the view from above (row x column upward)."""
    return np.cross(sm.Grid.Row.UVectECF.get_array(), sm.Grid.Col.UVectECF.get_array()) @ sm.GeoData.SCP.ECF.get_array() > 0


def rows_along_range(sm):
    los = sm.GeoData.SCP.ECF.get_array() - sm.SCPCOA.ARPPos.get_array()
    return sm.Grid.Row.UVectECF.get_array() @ los / np.linalg.norm(los)


@pytest.mark.parametrize('plane', ['slant', 'ground'])
def test_sicd(cphd, plane):
    d, ecf = cphd
    f = d / f'{plane}.nitf'
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(f), plane=plane, backend='cpu')
    assert o['plane'] == plane and o['product'] == 'sicd' and o['weighting'] == 'taylor' and o['path'] == str(f)
    assert o['image'].dtype == np.complex64 and o['baseband'] and {'read', 'form', 'baseband', 'write'} <= set(o['timing'])
    sm, pix, off, pslr = read_sicd(f, ecf)
    assert sm.Grid.ImagePlane == plane.upper() and sm.Grid.Type == 'PLANE' and sm.Grid.Row.WgtType.WindowName == 'TAYLOR'
    assert np.abs(off).max() < 0.05                                  # pixels: sarpy's projection is the product's
    assert pslr.max() < -27 and from_above(sm)
    assert rows_along_range(sm) > (0.99 if plane == 'slant' else 0.7)   # rows run away from the radar
    assert pix.shape == (o['image'].shape[1], o['image'].shape[0])


def test_default_is_the_slant_plane_and_nothing_is_written_without_an_output(cphd):
    d, ecf = cphd
    o = fastsar.process_cphd(str(d / 'c.cphd'), backend='cpu')
    assert o['plane'] == 'slant' and 'path' not in o and 'metadata' not in o and o['area'] == 'tasked'
    e1, e2 = o['e1'], o['e2']
    assert abs(e1 @ e2) < 1e-12 and np.cross(e1, e2)[2] > 0 and (o['image'] == 0).mean() < 0.35


def test_sva_sicd(cphd):
    d, ecf = cphd
    f = d / 'sva.nitf'
    ref = fastsar.process_cphd(str(d / 'c.cphd'), weighting='uniform', backend='cpu')
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(f), weighting='sva', backend='cpu')
    sm, pix, off, pslr = read_sicd(f, ecf)
    assert sm.Grid.Row.WgtType.WindowName == 'SVA' and np.abs(off).max() < 0.1
    assert pslr.max() < -24                                          # the unweighted aperture's: -13 dB
    assert 0.45 < o['spx'] / ref['spx'] < 0.62 and abs(o['spy'] / ref['spy'] - 0.5 / 0.96) < 0.03   # twice Nyquist
    assert any('keystone' in n for n in o['notes'])
    with pytest.raises(ValueError, match='spacing'):
        fastsar.process_cphd(str(d / 'c.cphd'), weighting='sva', spacing=0.3, backend='cpu')


@pytest.mark.parametrize('weighting', [None, 'taylor'])
def test_sidd(cphd, weighting):
    from sarpy.io.product.converter import open_product
    from sarpy.geometry import point_projection
    d, ecf = cphd
    f = d / f'sidd_{weighting}.nitf'
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(f), product='sidd', weighting=weighting,
                             backend='cpu')
    assert o['plane'] == 'ground' and o['weighting'] == (weighting or 'sva') and o['image'].dtype == np.float32
    assert abs(o['spx'] - o['spy']) < 1e-12 and (o['image'] >= 0).all()
    rd = open_product(str(f))
    sd = rd.sidd_meta[0] if isinstance(rd.sidd_meta, (list, tuple)) else rd.sidd_meta
    pix = rd[:, :]
    assert pix.dtype == np.uint8 and pix.shape == o['image'].shape
    assert sd.Display.InteractiveProcessing[0].GeometricTransform.Orientation.ShadowDirection == 'DOWN'
    lut = sd.Display.NonInteractiveProcessing[0].ProductGenerationOptions.DataRemapping.LUTName
    assert lut == 'NRL'
    rc, _, _ = point_projection.ground_to_image(ecf, sd)
    for r, c in rc:                                                  # each target is the brightest thing near it
        r0, c0 = int(round(r)), int(round(c))
        w = pix[r0 - 6:r0 + 7, c0 - 6:c0 + 7]
        a, b = np.unravel_index(np.argmax(w), w.shape)
        assert abs(a - 6) <= 1 and abs(b - 6) <= 1 and w.max() > 200


def test_sidd_spacing_and_remap(cphd):
    d, ecf = cphd
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(d / 'log.nitf'), product='sidd', remap='log', spacing=0.5,
                             shadows_down=False, backend='cpu')
    assert o['spx'] == 0.5 and o['spy'] == 0.5
    assert o['metadata'].Display.NonInteractiveProcessing[0].ProductGenerationOptions.DataRemapping.LUTName == 'LOG'
    with pytest.raises(ValueError, match='square'):
        fastsar.process_cphd(str(d / 'c.cphd'), product='sidd', spacing=(0.5, 0.4), backend='cpu')


def test_quality_areas_need_an_antenna_pattern(cphd):
    d, ecf = cphd
    for area in ('tasked_quality', 4.0, 'black'):                    # the simulator's CPHD describes no antenna
        with pytest.raises(ValueError, match='antenna pattern'):
            fastsar.process_cphd(str(d / 'c.cphd'), area=area, nesz_db=-20.0, backend='cpu')


def test_autofocus_modes(cphd):
    d, ecf = cphd
    ref = fastsar.process_cphd(str(d / 'c.cphd'), autofocus=False, backend='cpu')
    assert ref['autofocus'] is None and ref['autofocus_info'] is None and 'autofocus' not in ref['timing']
    for mode in ('quadratic', 'pga', 'spatially_variant'):
        o = fastsar.process_cphd(str(d / 'c.cphd'), autofocus=mode, backend='cpu')
        assert o['autofocus_info']['mode'] == mode and 'autofocus' in o['timing']
        # a focused collection stays focused: the image is the one without autofocus to within a few percent
        e = np.abs(o['image'] - ref['image']).max() / np.abs(ref['image']).max()
        assert e < 0.05, (mode, e)


@pytest.mark.parametrize('backend', ['cuda', 'jax'])
def test_backends_agree(cphd, backend):
    if backend not in fastsar.available_backends() + ['jax']:
        pytest.skip(f'no {backend} backend on this machine')
    d, ecf = cphd
    for kw in (dict(), dict(product='sidd')):
        ref = fastsar.process_cphd(str(d / 'c.cphd'), backend='cpu', **kw)
        o = fastsar.process_cphd(str(d / 'c.cphd'), backend=backend, **kw)
        assert isinstance(o['image'], np.ndarray) and o['image'].shape == ref['image'].shape
        assert np.abs(o['image'] - ref['image']).max() < 2e-3 * np.abs(ref['image']).max()


def test_errors(cphd):
    d, ecf = cphd
    p = str(d / 'c.cphd')
    for kw, match in ((dict(product='png'), 'product'), (dict(plane='tilted'), 'plane'),
                      (dict(weighting='hamming'), 'weighting'), (dict(autofocus='sharpen'), 'autofocus'),
                      (dict(area='everything'), 'area'), (dict(area=True), 'area'), (dict(spacing=-1.0), 'spacing'),
                      (dict(product='sidd', plane='slant'), 'ground plane'),
                      (dict(plane='ground', weighting='sva'), 'slant plane')):
        with pytest.raises(ValueError, match=match):
            fastsar.process_cphd(p, backend='cpu', **kw)
