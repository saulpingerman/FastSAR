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
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(f), plane=plane, weighting='taylor', backend='cpu')
    assert o['plane'] == plane and o['product'] == 'sicd' and o['weighting'] == 'taylor' and o['path'] == str(f)
    assert o['image'].dtype == np.complex64 and o['baseband'] and {'read', 'form', 'baseband', 'write'} <= set(o['timing'])
    sm, pix, off, pslr = read_sicd(f, ecf)
    assert sm.Grid.ImagePlane == plane.upper() and sm.Grid.Type == 'PLANE' and sm.Grid.Row.WgtType.WindowName == 'TAYLOR'
    assert np.abs(off).max() < 0.05                                  # pixels: sarpy's projection is the product's
    assert pslr.max() < -27 and from_above(sm)
    assert rows_along_range(sm) > (0.99 if plane == 'slant' else 0.7)   # rows run away from the radar
    assert pix.shape == (o['image'].shape[1], o['image'].shape[0])


def test_defaults(cphd):
    """What process_cphd gives when nothing is asked for. A SICD: the ground plane, no weighting, quadratic
    autofocus, the tasked area. A SIDD: the ground plane, spatially variant apodization, quadratic autofocus, the
    NRL remap, shadows down. Each is the product of the same options spelled out, and the default SICD of 0.2.0 is
    plane='slant', weighting='taylor'."""
    d, ecf = cphd
    p, f = str(d / 'c.cphd'), d / 'default.nitf'
    o = fastsar.process_cphd(p, str(f), backend='cpu')
    assert (o['product'], o['plane'], o['weighting'], o['area']) == ('sicd', 'ground', 'uniform', 'tasked')
    assert o['autofocus_info']['mode'] == 'quadratic' and 'autofocus' in o['timing'] and 'to_slant' not in o['timing']
    sm, pix, off, pslr = read_sicd(f, ecf)
    assert sm.Grid.ImagePlane == 'GROUND' and sm.Grid.Type == 'PLANE'
    assert sm.Grid.Row.WgtType.WindowName == 'UNIFORM' and sm.Grid.Col.WgtType.WindowName == 'UNIFORM'
    up = sm.GeoData.SCP.ECF.get_array() / np.linalg.norm(sm.GeoData.SCP.ECF.get_array())
    urow, ucol = sm.Grid.Row.UVectECF.get_array(), sm.Grid.Col.UVectECF.get_array()
    assert abs(urow @ up) < 4e-3 and abs(ucol @ up) < 4e-3            # both axes on the ground (geocentric up: 0.2 deg)
    assert rows_along_range(sm) > 0.7 and from_above(sm)             # rows away from the radar, the view from above
    assert np.abs(off).max() < 0.05 and -14.5 < pslr.max() < -11.5   # the unweighted aperture's sidelobes
    same = fastsar.process_cphd(p, plane='ground', weighting='uniform', autofocus='quadratic', area='tasked', backend='cpu')
    assert np.array_equal(same['image'], o['image'])
    s = fastsar.process_cphd(p, product='sidd', backend='cpu')
    assert (s['product'], s['plane'], s['weighting'], s['area']) == ('sidd', 'ground', 'sva', 'tasked')
    assert s['autofocus_info']['mode'] == 'quadratic' and s['remap'] == 'nrl' and s['shadows_down'] is True
    same = fastsar.process_cphd(p, product='sidd', plane='ground', weighting='sva', autofocus='quadratic', remap='nrl',
                                backend='cpu')
    assert np.array_equal(same['image'], s['image'])
    old = fastsar.process_cphd(p, plane='slant', weighting='taylor', backend='cpu')
    assert (old['plane'], old['weighting'], old['autofocus_info']['mode']) == ('slant', 'taylor', 'quadratic')
    assert old['image'].shape != o['image'].shape and 'to_slant' in old['timing']


SICD_OPTIONS = [(pl, w) for pl in ('ground', 'slant') for w in ('uniform', 'taylor')] + [('slant', 'sva')]


@pytest.mark.parametrize('plane, weighting', SICD_OPTIONS, ids=[f'{pl}-{w}' for pl, w in SICD_OPTIONS])
@pytest.mark.parametrize('autofocus', ['quadratic', False])
def test_explicit_sicd_options_are_kept(cphd, plane, weighting, autofocus):
    """Every plane and weighting of a SICD, with and without autofocus, when asked for by name: the product and its
    metadata state what was asked, whatever the defaults are."""
    d, ecf = cphd
    f = d / 'explicit.nitf'
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(f), plane=plane, weighting=weighting, autofocus=autofocus,
                             backend='cpu')
    assert (o['product'], o['plane'], o['weighting']) == ('sicd', plane, weighting)
    assert (o['autofocus_info'] is not None) == bool(autofocus) and ('autofocus' in o['timing']) == bool(autofocus)
    sm, pix, off, pslr = read_sicd(f, ecf)
    assert sm.Grid.ImagePlane == plane.upper() and sm.Grid.Type == 'PLANE' and from_above(sm)
    assert sm.Grid.Row.WgtType.WindowName == sm.Grid.Col.WgtType.WindowName == weighting.upper()
    assert sm.ImageFormation.AzAutofocus == ('GLOBAL' if o['autofocus'] else 'NO')
    assert np.abs(off).max() < (0.1 if weighting == 'sva' else 0.05)
    assert pslr.max() < {'uniform': -11.5, 'taylor': -27, 'sva': -24}[weighting]
    assert weighting != 'uniform' or pslr.max() > -14.5
    assert rows_along_range(sm) > (0.99 if plane == 'slant' else 0.7)


def test_nothing_is_written_without_an_output(cphd):
    d, ecf = cphd
    for plane in ('slant', 'ground'):
        o = fastsar.process_cphd(str(d / 'c.cphd'), plane=plane, backend='cpu')
        assert o['plane'] == plane and 'path' not in o and 'metadata' not in o and o['area'] == 'tasked'
        e1, e2 = o['e1'], o['e2']
        assert abs(e1 @ e2) < 1e-12 and np.cross(e1, e2)[2] > 0 and (o['image'] == 0).mean() < 0.35


def test_sva_sicd(cphd):
    d, ecf = cphd
    f = d / 'sva.nitf'
    ref = fastsar.process_cphd(str(d / 'c.cphd'), plane='slant', weighting='uniform', backend='cpu')
    o = fastsar.process_cphd(str(d / 'c.cphd'), str(f), plane='slant', weighting='sva', backend='cpu')
    sm, pix, off, pslr = read_sicd(f, ecf)
    assert sm.Grid.Row.WgtType.WindowName == 'SVA' and np.abs(off).max() < 0.1
    assert pslr.max() < -24                                          # the unweighted aperture's: -13 dB
    assert 0.45 < o['spx'] / ref['spx'] < 0.62 and abs(o['spy'] / ref['spy'] - 0.5 / 0.96) < 0.03   # twice Nyquist
    assert any('keystone' in n for n in o['notes'])


@pytest.mark.parametrize('weighting', [None, 'taylor', 'uniform', 'sva'])
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
    one = fastsar.process_cphd(str(d / 'c.cphd'), product='sidd', spacing=[0.5], autofocus=False, backend='cpu')
    assert one['spx'] == 0.5 and one['image'].shape == o['image'].shape
    tiny = fastsar.process_cphd(str(d / 'c.cphd'), plane='ground', extent=(3.0, 3.0), backend='cpu')   # too small to focus
    assert tiny['autofocus'] is None and any('autofocus skipped' in n for n in tiny['notes'])


def test_quality_areas_need_an_antenna_pattern(cphd):
    d, ecf = cphd
    for area in ('tasked_quality', 4.0, 'black'):                    # the simulator's CPHD describes no antenna
        with pytest.raises(ValueError, match='antenna pattern'):
            fastsar.process_cphd(str(d / 'c.cphd'), area=area, nesz_db=-20.0, backend='cpu')


def test_ground_sicd_spectrum_fits_its_sampling(cphd):
    """The image as formed is the product: its spectrum must leave a guard band on both axes (sampled for the
    resampling to the slant plane alone, it wraps along the line of sight)."""
    d, ecf = cphd
    o = fastsar.process_cphd(str(d / 'c.cphd'), plane='ground', weighting='uniform', autofocus=False, backend='cpu')
    F = np.abs(np.fft.fft2(o['image'])) ** 2
    for axis in (0, 1):
        p = np.sort(F.sum(1 - axis))
        assert p[:max(2, len(p) // 100)].mean() < 1e-3 * p[len(p) // 2], axis      # the emptiest 1% of the bins
    wide = fastsar.process_cphd(str(d / 'c.cphd'), plane='ground', spacing=(0.3, 0.6), autofocus=False, backend='cpu')
    assert wide['spy'] == 0.6 and any('aliased' in n for n in wide['notes'])
    assert not any('aliased' in n for n in o['notes'])


def test_sidd_targets_keep_their_amplitude(cphd):
    """Detection at twice Nyquist, then the projection: equal targets come out equal whatever the weighting."""
    d, ecf = cphd
    for weighting, limit in (('sva', 1.5), ('taylor', 1.5), ('uniform', 1.5)):
        o = fastsar.process_cphd(str(d / 'c.cphd'), product='sidd', weighting=weighting, autofocus=False, backend='cpu')
        img, pk = o['image'].copy(), []
        for _ in TG:
            i, j = np.unravel_index(np.argmax(img), img.shape)
            pk.append(img[i, j])
            img[max(i - 8, 0):i + 9, max(j - 8, 0):j + 9] = 0
        assert 20 * np.log10(max(pk) / min(pk)) < limit, (weighting, pk)
        assert o['remap'] == 'nrl' and o['shadows_down'] is True


def test_quality_area_masks_the_product(cphd, monkeypatch):
    """A quality area end to end with a stand-in for the radiometric model (the simulated file has no antenna
    pattern): the grid is the area's bounding rectangle and the pixels outside the area are 0, on both planes."""
    from fastsar import rgiqe
    d, ecf = cphd
    seen = {}

    class Model:
        has_antenna, notes = True, ['stand-in model']

        def __init__(self, path, meta, **kw):
            seen['kw'] = kw

    def bounds(rad, kind, center, e1, e2, bw, n, tasked=None, **kw):
        seen.update(center=center, e1=e1, e2=e2, kind=kind, bounds=kw)
        return (-60.0, 60.0, -45.0, 45.0)

    def mask(rad, pts, kind, bw, n, tasked=None, **kw):
        seen['mask'] = kw
        q = pts - seen['center']
        level = (1.0 - np.hypot((q @ seen['e1']) / 60.0, (q @ seen['e2']) / 45.0)).astype(np.float32)   # an ellipse
        return level >= 0, dict(level=level)
    monkeypatch.setattr(rgiqe, 'Radiometry', Model)
    monkeypatch.setattr(rgiqe, 'area_bounds', bounds)
    monkeypatch.setattr(rgiqe, 'area_mask', mask)
    g = fastsar.process_cphd(str(d / 'c.cphd'), plane='ground', area='black', autofocus=False, backend='cpu')
    assert seen['kind'] == 'black' and 'stand-in model' in g['notes']
    assert seen['bounds'] == seen['mask'] == dict(black_sigma0_db=-10.0)             # the reference of the black edge
    nx, ny = g['image'].shape
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    q = g['origin'] + I[..., None] * g['spx'] * g['e1'] + J[..., None] * g['spy'] * g['e2'] - seen['center']
    r = np.hypot((q @ seen['e1']) / 60.0, (q @ seen['e2']) / 45.0)
    assert not g['image'][r > 1.03].any() and (g['image'][r < 0.97] != 0).mean() > 0.999
    assert abs((g['image'] != 0).mean() - np.pi / 4) < 0.03          # the ellipse inscribed in its rectangle
    s = fastsar.process_cphd(str(d / 'c.cphd'), plane='slant', area=4.5, autofocus=False, backend='cpu')
    assert seen['kind'] == 4.5 and s['plane'] == 'slant' and 0.2 < (s['image'] != 0).mean() < np.pi / 4 + 0.03
    fastsar.process_cphd(str(d / 'c.cphd'), area='black', black_sigma0_db=-3.0, autofocus=False, backend='cpu')
    assert seen['bounds'] == seen['mask'] == dict(black_sigma0_db=-3.0)


@pytest.mark.parametrize('kw', [dict(), dict(plane='slant', weighting='taylor')], ids=['default', 'slant-taylor'])
def test_autofocus_modes(cphd, kw):
    d, ecf = cphd
    ref = fastsar.process_cphd(str(d / 'c.cphd'), autofocus=False, backend='cpu', **kw)
    assert ref['autofocus'] is None and ref['autofocus_info'] is None and 'autofocus' not in ref['timing']
    for mode in ('quadratic', 'pga', 'spatially_variant'):
        o = fastsar.process_cphd(str(d / 'c.cphd'), autofocus=mode, backend='cpu', **kw)
        assert o['autofocus_info']['mode'] == mode and 'autofocus' in o['timing']
        # a focused collection stays focused: the image is the one without autofocus to within a few percent
        e = np.abs(o['image'] - ref['image']).max() / np.abs(ref['image']).max()
        assert e < 0.05, (mode, e)


@pytest.mark.parametrize('backend', ['cuda', 'jax'])
def test_backends_agree(cphd, backend):
    if backend not in fastsar.available_backends() + ['jax']:
        pytest.skip(f'no {backend} backend on this machine')
    d, ecf = cphd
    for kw in (dict(), dict(plane='slant', weighting='taylor'), dict(product='sidd')):
        ref = fastsar.process_cphd(str(d / 'c.cphd'), backend='cpu', **kw)
        o = fastsar.process_cphd(str(d / 'c.cphd'), backend=backend, **kw)
        assert isinstance(o['image'], np.ndarray) and o['image'].shape == ref['image'].shape
        # the formers agree to their accuracy target (-40 dB by default); 2e-4 on x86-64, 6e-3 on 64-bit ARM
        assert np.abs(o['image'] - ref['image']).max() < 2e-2 * np.abs(ref['image']).max()


# products of one image (the default SICD) and of several, for the tests of devices too small to hold them
PLACED = [dict(), dict(plane='slant', weighting='taylor'), dict(product='sidd'), dict(plane='slant', weighting='sva'),
          dict(plane='ground', weighting='taylor'), dict(product='sidd', autofocus=False, remap='log')]


@pytest.mark.cuda
@pytest.mark.parametrize('kw', PLACED, ids=['default', 'sicd', 'sidd', 'sva', 'ground', 'sidd-log'])
def test_small_gpu_gives_the_same_product(cphd, kw, monkeypatch, tmp_path):
    """A GPU with too little memory for the images of a product (FASTSAR_CUDA_BLOCK_MB: a megabyte to count on, or
    half the product's need) gives the product of a GPU with room: the images stay on the host and cross the
    device in blocks, a note says so, and the pixels are the same, the zeros exactly (autofocus then estimates
    with the host's transforms: float32 rounding)."""
    pytest.importorskip('cupy')
    from fastsar import memory
    d, ecf = cphd
    ext = 'sidd' if kw.get('product') == 'sidd' else 'sicd'
    ref = fastsar.process_cphd(str(d / 'c.cphd'), str(tmp_path / f'ref.{ext}'), backend='cuda', **kw)
    assert not any('GPU has' in n for n in ref['notes'])
    plans = []
    plan = memory.cuda_plan
    monkeypatch.setattr(memory, 'cuda_plan', lambda *a, **k: plans.append(plan(*a, **k)) or plans[-1])
    for mb in (1.0, None, None):
        if mb is None:                                               # some of the images on the device
            mb = 0.55 * plans[0]['need'] / 2 ** 20 if len(plans) == 1 else 0.8 * plans[0]['need'] / 2 ** 20
        monkeypatch.setenv('FASTSAR_CUDA_BLOCK_MB', str(mb))
        o = fastsar.process_cphd(str(d / 'c.cphd'), str(tmp_path / f'small.{ext}'), backend='cuda', **kw)
        assert not plans[-1]['fits'] and sum('GPU has' in n and 'kept on the host' in n for n in o['notes']) == 1
        assert [n for n in o['notes'] if 'GPU has' not in n] == ref['notes']
        assert o['image'].dtype == ref['image'].dtype and o['image'].shape == ref['image'].shape
        assert np.array_equal(o['image'] == 0, ref['image'] == 0)
        e = np.abs(o['image'] - ref['image']).max() / np.abs(ref['image']).max()
        assert e < 2e-6, (mb, e)
        a, b = (np.fromfile(str(tmp_path / f'{k}.{ext}'), np.uint8) for k in ('small', 'ref'))
        differ = np.count_nonzero(a != b) if a.size == b.size else a.size
        assert differ < (64 if ext == 'sidd' and kw.get('remap') != 'log' else 1e-2 * a.size), (mb, differ)
    # the budgets gave different placements, when the product has more than one image
    assert len({tuple(p['place'].values()) for p in plans}) >= min(2, len(plans[0]['place']))
    monkeypatch.setenv('FASTSAR_CUDA_BLOCK_MB', '0')
    with pytest.raises(ValueError, match='FASTSAR_CUDA_BLOCK_MB'):
        fastsar.process_cphd(str(d / 'c.cphd'), backend='cuda', **kw)


def test_errors(cphd):
    d, ecf = cphd
    p = str(d / 'c.cphd')
    for kw, match in ((dict(product='png'), 'product'), (dict(plane='tilted'), 'plane'),
                      (dict(product='sidd', remap='nosuch'), 'remap'), (dict(product='sidd', remap='linear'), 'remap'),
                      (dict(weighting='sva', spacing=0.3), 'spacing'),
                      (dict(product='sidd', spacing=(0.5, 0.4)), 'square'), (dict(extent=(10.0,)), 'extent'),
                      (dict(weighting='hamming'), 'weighting'), (dict(autofocus='sharpen'), 'autofocus'),
                      (dict(area='everything'), 'area'), (dict(area=True), 'area'), (dict(spacing=-1.0), 'spacing'),
                      (dict(area='black', black_sigma0_db='dark'), 'black_sigma0_db'),
                      (dict(product='sidd', plane='slant'), 'ground plane'),
                      (dict(plane='ground', weighting='sva'), 'slant plane'),
                      (dict(weighting='sva'), "plane='slant'")):       # the default plane of a SICD is the ground
        with pytest.raises(ValueError, match=match):
            fastsar.process_cphd(p, backend='cpu', **kw)


@pytest.mark.parametrize('kw', PLACED, ids=['default', 'sicd', 'sidd', 'sva', 'ground', 'sidd-log'])
def test_small_jax_device_gives_the_same_product(cphd, kw, monkeypatch, tmp_path):
    """A JAX device with too little memory for the images of a product (FASTSAR_JAX_BLOCK_MB: a megabyte to count
    on, or part of the product's need) gives the product of a device with room: the images stay on the host and
    cross the device in blocks, a note says so, and the pixels are the same to float32 rounding, with the zeros
    of the area."""
    pytest.importorskip('jax')
    from fastsar import memory
    d, ecf = cphd
    ext = 'sidd' if kw.get('product') == 'sidd' else 'sicd'
    monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB', raising=False)
    ref = fastsar.process_cphd(str(d / 'c.cphd'), str(tmp_path / f'ref.{ext}'), backend='jax', **kw)
    assert not any('JAX device has' in n for n in ref['notes'])
    plans = []
    plan = memory.cuda_plan
    monkeypatch.setattr(memory, 'cuda_plan', lambda *a, **k: plans.append(plan(*a, **k)) or plans[-1])
    need = None
    for mb in (1.0, 0.45, 0.8):
        if need is not None:                                         # some of the images on the device
            mb = mb * need / 2 ** 20
        monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', str(mb))
        o = fastsar.process_cphd(str(d / 'c.cphd'), str(tmp_path / f'small.{ext}'), backend='jax', **kw)
        need = plans[0]['need']
        assert plans[-1]['budget'] == int(mb * 2 ** 20) and not plans[-1]['fits']
        assert sum('JAX device has' in n and 'kept on the host' in n for n in o['notes']) == 1
        assert [n for n in o['notes'] if 'JAX device has' not in n] == ref['notes']
        assert o['image'].dtype == ref['image'].dtype and o['image'].shape == ref['image'].shape
        e = np.abs(o['image'] - ref['image']) / np.abs(ref['image']).max()
        assert e.max() < 1e-5, (mb, e.max())
        # the zeros of the area are the same; apodization also nulls single pixels to rounding (1e-11 of the peak),
        # which fall on zero or beside it
        differ = (o['image'] == 0) != (ref['image'] == 0)
        assert differ.sum() < 1e-4 * differ.size and (not differ.any() or e[differ].max() < 1e-8), (mb, differ.sum())
        a, b = (np.fromfile(str(tmp_path / f'{k}.{ext}'), np.uint8) for k in ('small', 'ref'))
        # the complex samples differ in their last bits (the transforms of other blocks of lines); display values
        # on a rounding boundary by one count
        assert a.size == b.size
        assert ext == 'sicd' or np.count_nonzero(a != b) < (1e-3 if kw.get('remap') != 'log' else 1e-2) * a.size, mb
    assert len({tuple(p['place'].values()) for p in plans}) >= min(2, len(plans[0]['place']))
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '0')
    with pytest.raises(ValueError, match='FASTSAR_JAX_BLOCK_MB'):
        fastsar.process_cphd(str(d / 'c.cphd'), backend='jax', **kw)
