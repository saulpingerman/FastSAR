"""The display layout and SIDD products (fastsar.sidd), on a simulated spotlight collection written as CPHD
(sim.write_cphd) and formed by form_cphd, flown both ways so that the scene lies to the right of the track and to
the left: the shadows-down layout against the simulator's own frame, the remapped display, a SIDD round trip
through sarpy whose own projection is compared with the targets' true positions, and the options of write_sidd.
Needs sarpy; the consistency check needs sarkit (Python 3.11 or later)."""
import itertools
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import io, products, quality, remap, sidd, sim

pytestmark = pytest.mark.needs('sarpy')

lat0, lon0, h0, heading, graze, spacing = 40.7934, -77.8600, 351.0, 30.0, 35.0, 0.4
# Point targets in the simulator's frame (x: ground range away from the radar, y: along the track of the pass that
# looks right, z: up). No rotation or reflection of the pattern maps it to itself, so a mirrored or turned display
# cannot pass for the right one. Seen from above with the radar at the top, x runs down the rows and y along the
# columns.
TG = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 15.0, 0.0], [-18.0, -22.0, 0.0], [12.0, -9.0, 0.0]])
LAYOUTS = [dict(transpose=t, flip_rows=r, flip_cols=c) for t, r, c in itertools.product((False, True), repeat=3)]


def find_peaks(a, guard=6):
    """The brightest separated peaks [len(TG), 2] (row, col) of an image's magnitude."""
    a = np.abs(a).astype(np.float64)
    out = []
    for _ in TG:
        i, j = np.unravel_index(np.argmax(a), a.shape)
        out.append((i, j))
        a[max(i - guard, 0):i + guard + 1, max(j - guard, 0):j + guard + 1] = 0
    return np.array(out, np.float64)


def by_target(p):
    """Peaks [n, 2] put in the targets' order, taking rows along +x and columns along +y, and the largest distance
    (pixels) left between a target and its peak once the two patterns are laid over each other."""
    e = TG[:, :2] / spacing
    d = np.linalg.norm((p - p.mean(0))[None] - (e - e.mean(0))[:, None], axis=-1)       # [target, peak]
    return p[d.argmin(1)], d.min(1).max()


def fine_peaks(lay):
    """Peak positions [n, 2] (fractional row, col) of the targets in a laid out complex image, in their order."""
    img = np.ascontiguousarray(lay['image'])
    p, miss = by_target(find_peaks(img))
    assert miss < 1.0
    m = [quality.point_target(img, ij, lay['row_ss'], lay['col_ss'], half=16) for ij in p]
    return np.array([(k['i'], k['j']) for k in m])


def read_sidd(path):
    from sarpy.io.product.converter import open_product
    rd = open_product(str(path))
    sd = rd.sidd_meta[0] if isinstance(rd.sidd_meta, (list, tuple)) else rd.sidd_meta
    return rd, sd, rd[:, :]


def fake(e1, e2, radar, shape=(5, 7), **kw):
    """A product dict with the given axes, seen from a radar at `radar` from the image's center (local frame)."""
    rng = np.random.default_rng(1)
    e1, e2, origin = np.asarray(e1, np.float64), np.asarray(e2, np.float64), np.array([3.0, -4.0, 0.0])
    center = origin + 0.5 * (shape[0] - 1) * 0.5 * e1 + 0.5 * (shape[1] - 1) * 0.3 * e2
    track = center + radar + np.linspace(-50.0, 50.0, 9)[:, None] * np.cross([0.0, 0.0, 1.0], radar)
    img = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    return dict(image=img, e1=e1, e2=e2, spx=0.5, spy=0.3, origin=origin, meta=dict(tx=track, rcv=track), **kw)


def positions(out, lay):
    """Local positions of the pixels: on the product's own grid, laid out as lay; and from lay's own geometry."""
    n1, n2 = out['image'].shape
    i, j = np.mgrid[:n1, :n2]
    own = out['origin'] + i[..., None] * out['spx'] * out['e1'] + j[..., None] * out['spy'] * out['e2']
    r, c = np.mgrid[:lay['image'].shape[0], :lay['image'].shape[1]]
    return sidd.orient(own, lay['transform']), \
        lay['origin'] + r[..., None] * lay['row_ss'] * lay['row'] + c[..., None] * lay['col_ss'] * lay['col']


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    """The collection flown both ways, written as CPHD and formed by form_cphd; the targets' ECF positions."""
    tmp = tmp_path_factory.mktemp('sidd')
    col = sim.make_collect(res=0.5, scene=100.0, r0=6e3, graze_deg=graze)
    S = sim.add_noise(sim.simulate_brute(col, TG, np.ones(len(TG))), 0.3, np.random.default_rng(0))
    S = S.astype(np.complex64)
    out = {}
    for side in ('right', 'left'):                                 # the same aperture flown the other way looks left
        c, s = (col, S) if side == 'right' else \
            (sim.Collect(fmin=col.fmin, df=col.df, K=col.K, ant=col.ant[::-1].copy(), res=col.res), S[::-1].copy())
        path = str(tmp / f'{side}.cphd')
        sim.write_cphd(path, c, s, lat0, lon0, h0, heading=heading)
        out[side] = fastsar.form_cphd(path, backend='cpu', spacing=spacing, height=h0)
    return SimpleNamespace(tmp=tmp, out=out, ecf=sim.to_ecf(TG, lat0, lon0, h0, heading))


def test_layout_rule():
    """Every lossless grid (either handedness, any quarter turn), seen from each side: rows along range away from
    the radar, row x col up, the geometry of the laid out pixels right."""
    x, y, z = np.eye(3)
    for (a, b), sa, sb in itertools.product(((x, y), (y, x)), (1, -1), (1, -1)):
        for radar in (-x, x, -y, y):
            out = fake(sa * a, sb * b, 5e3 * radar + 4e3 * z)
            lay = sidd.shadows_down(out)
            assert np.allclose(lay['row'], -radar) and np.allclose(np.cross(lay['row'], lay['col']), z)
            assert abs(lay['angle']) < 1e-9
            own, laid = positions(out, lay)
            assert np.allclose(own, laid, atol=1e-12)
            assert lay['image'].shape == ((7, 5) if lay['transform']['transpose'] else (5, 7))
            assert (lay['row_ss'], lay['col_ss']) == ((0.3, 0.5) if lay['transform']['transpose'] else (0.5, 0.3))
            assert np.shares_memory(lay['image'], out['image'])
            assert np.array_equal(lay['image'], sidd.orient(out['image'], lay['transform']))
    # a grid turned from the line of sight: the nearest of the four views, and the lean that is left
    for turn in (-30.0, 20.0, 40.0):
        t = np.radians(turn)
        lay = sidd.shadows_down(fake(x, y, -5e3 * (np.cos(t) * x + np.sin(t) * y) + 4e3 * z))
        assert np.allclose(lay['row'], x) and np.allclose(lay['col'], y) and abs(lay['angle'] - turn) < 1e-6
    # a slant-plane product: e1 along the track, e2 along the line of sight; "up" is the plane's upward normal
    g = np.radians(graze)
    out = fake(y, np.cos(g) * x - np.sin(g) * z, -5e3 * np.cos(g) * x + 5e3 * np.sin(g) * z, plane='slant')
    lay = sidd.shadows_down(out)
    assert lay['transform'] == dict(transpose=True, flip_rows=False, flip_cols=False)
    assert np.allclose(lay['row'], out['e2']) and np.allclose(lay['col'], y)
    assert np.cross(lay['row'], lay['col']) @ z > 0
    own, laid = positions(out, lay)
    assert np.allclose(own, laid, atol=1e-12)
    with pytest.raises(ValueError, match='normal to the image plane'):
        sidd.shadows_down(fake(x, y, 4e3 * z))


def test_shadows_down(scene):
    """The simulated images: the targets lie where the simulator's frame puts them (range down the rows, seen from
    above) whichever way the radar flew, and no other layout of the pixels does that."""
    used = {}
    for side, out in scene.out.items():
        lay = sidd.shadows_down(out)
        used[side] = lay['transform']
        assert abs(lay['angle']) < 0.05 and np.allclose(np.cross(lay['row'], lay['col']), [0.0, 0.0, 1.0])
        own, laid = positions(out, lay)
        assert np.allclose(own, laid, atol=1e-9)
        miss = {tuple(tr.values()): by_target(find_peaks(sidd.orient(out['image'], tr)))[1] for tr in LAYOUTS}
        assert miss.pop(tuple(lay['transform'].values())) < 0.5
        assert min(miss.values()) > 5.0, miss
        p = fine_peaks(lay)
        e = TG[:, :2] / spacing
        assert np.abs((p - p[0]) - (e - e[0])).max() < 0.05           # e.g. 20 m farther in range: 50 rows down
        # the peaks are where the laid out geometry puts the targets
        want = (io.ecf_to_local(scene.ecf, out['meta']) - lay['origin']) @ np.stack([lay['row'], lay['col']], 1)
        assert np.abs(p - want / [lay['row_ss'], lay['col_ss']]).max() < 0.05
    assert used['right'] != used['left']


def test_to_display(scene):
    out = scene.out['right']
    img, tr = out['image'], sidd.shadows_down(out)['transform']
    d = sidd.to_display(out)
    assert d.dtype == np.uint8 and np.array_equal(d, sidd.orient(remap.nrl(img), tr))
    assert by_target(find_peaks(d))[1] < 1.0
    assert np.array_equal(sidd.to_display(out, shadows_down=False), remap.nrl(img))
    assert np.array_equal(sidd.to_display(dict(out, image=np.abs(img))), d)             # a detected image
    assert np.array_equal(sidd.to_display(out, 'density'), sidd.orient(remap.density(img), tr))
    assert np.array_equal(sidd.to_display(out, 'nrl', a=0.8, c=200.0),
                          sidd.orient(remap.nrl(img, a=0.8, c=200.0), tr))
    assert np.array_equal(sidd.to_display(out, lambda x, valid=None: remap.log(x, valid=valid, span_db=40.0)),
                          sidd.orient(remap.log(img, span_db=40.0), tr))
    assert sidd.to_display(dict(out, image=d), shadows_down=False) is d                 # display values already
    for bad in ('linear', 'nonesuch'):
        with pytest.raises(ValueError, match='remap'):
            sidd.to_display(out, bad)
    # unprocessed pixels (exactly 0) stay 0 and stay out of the statistics; a mask does the same
    i, j = np.mgrid[:img.shape[0], :img.shape[1]]
    disk = (i - 150) ** 2 + (j - 170) ** 2 <= 120 ** 2
    cut = dict(out, image=np.where(disk, img, 0))
    want = sidd.orient(remap.nrl(cut['image'], valid=disk), tr)
    assert np.array_equal(sidd.to_display(cut), want)
    assert not np.array_equal(want, sidd.orient(remap.nrl(cut['image']), tr))
    assert not sidd.to_display(cut)[~sidd.orient(disk, tr)].any()
    assert np.array_equal(sidd.to_display(out, valid=disk), want)
    assert np.array_equal(sidd.to_display(dict(out, valid=disk)), want)
    assert np.array_equal(sidd.to_display(dict(out, valid=disk), valid=np.ones_like(disk)), d)
    with pytest.raises(ValueError, match='valid mask'):
        sidd.to_display(out, valid=disk[:-1])


@pytest.mark.parametrize('side', ['right', 'left'])
def test_write_sidd(scene, side):
    """A round trip through sarpy: the pixels, the metadata, and the SIDD's own projection at the targets."""
    from sarpy.geometry import point_projection
    out = scene.out[side]
    path = scene.tmp / f'{side}.nitf'
    written = sidd.write_sidd(str(path), out)
    rd, sd, a = read_sidd(path)
    lay = sidd.shadows_down(out)
    assert type(sd) is type(written) and type(sd).__module__.endswith('sidd3_elements.SIDD')
    assert a.dtype == np.uint8 and np.array_equal(a, sidd.to_display(out))
    # range runs down the rows and the view is from above, in the file as on the screen
    assert by_target(find_peaks(a))[1] < 1.0
    assert by_target(find_peaks(a[:, ::-1]))[1] > 5.0 and by_target(find_peaks(a[::-1]))[1] > 5.0
    # the SIDD's projection: the peaks project to the targets, and the targets to the peaks
    p = fine_peaks(lay)
    for kind in ('PLANE', 'HAE'):
        err = np.linalg.norm(point_projection.image_to_ground(p, sd, projection_type=kind) - scene.ecf, axis=1)
        assert err.max() < 0.1 * spacing, (kind, err)
    rc, _, _ = point_projection.ground_to_image(scene.ecf, sd)
    assert np.abs(rc - p).max() < 0.1, np.abs(rc - p).max()
    # the metadata: the plane, the footprint, the corners, the display, the geometry of the collection
    m, f = sd.Measurement, sd.ExploitationFeatures
    assert (m.PixelFootprint.Row, m.PixelFootprint.Col) == a.shape
    assert np.allclose(m.PlaneProjection.SampleSpacing.get_array(), [lay['row_ss'], lay['col_ss']])
    assert np.allclose(m.PlaneProjection.ProductPlane.RowUnitVector.get_array(), lay['row'] @ out['meta']['R'])
    assert np.allclose(m.PlaneProjection.ProductPlane.ColUnitVector.get_array(), lay['col'] @ out['meta']['R'])
    assert m.ValidData.get_array(dtype=np.int64).tolist() == [[0, 0], [0, a.shape[1] - 1],
                                                              [a.shape[0] - 1, a.shape[1] - 1], [a.shape[0] - 1, 0]]
    edge = np.array([[-0.5, -0.5], [-0.5, a.shape[1] - 0.5], [a.shape[0] - 0.5, a.shape[1] - 0.5],
                     [a.shape[0] - 0.5, -0.5]])
    corners = io.geodetic_to_ecf(*sd.GeoData.ImageCorners.get_array(dtype=np.float64).T, h0)
    proj = point_projection.image_to_ground(edge, sd, projection_type='PLANE')
    assert np.linalg.norm(corners - proj, axis=1).max() < 0.05
    assert sd.Display.PixelType == 'MONO8I'
    assert sd.Display.NonInteractiveProcessing[0].ProductGenerationOptions.DataRemapping.LUTName == 'NRL'
    assert sd.Display.InteractiveProcessing[0].GeometricTransform.Orientation.ShadowDirection == 'DOWN'
    assert sd.ProductCreation.ProcessorInformation.Application == f'FastSAR {fastsar.__version__}'
    assert sd.ProductCreation.ProductName == 'SIMULATED' and sd.ProductCreation.Classification.classification == 'U'
    g, ph = f.Collections[0].Geometry, f.Collections[0].Phenomenology
    # SIDD 3 counts angles in the image clockwise from the columns: 90 is down the rows, 270 up, 0 to the right
    track = 0.0 if side == 'right' else 180.0
    for name, got, want in (('azimuth', g.Azimuth, heading + 270.0), ('graze', g.Graze, graze),
                            ('slope', g.Slope, graze), ('shadow', ph.Shadow.Angle, 90.0),
                            ('layover', ph.Layover.Angle, 270.0), ('ground track', ph.GroundTrack, track),
                            ('north', f.Products[0].North, 360.0 - heading)):
        assert abs((got - want + 180.0) % 360.0 - 180.0) < 0.2, (name, got, want)
    assert abs(g.Squint) == pytest.approx(90.0, abs=0.5) and np.sign(g.Squint) == (-1 if side == 'right' else 1)
    res = f.Products[0].Resolution
    assert res.Row == pytest.approx(res.Col / np.cos(np.radians(graze)), rel=0.03)       # ground range, cross range
    assert 0.4 < res.Col < 0.5
    print(f'SIDD ({side}): projection of the peaks within {err.max() * 100:.2f} cm of the targets, '
          f'{np.abs(rc - p).max():.3f} px; resolution {res.Row:.3f} x {res.Col:.3f} m')


def test_write_sidd_options(scene):
    out = scene.out['right']
    img, tr = out['image'], sidd.shadows_down(out)['transform']
    path = scene.tmp / 'options.nitf'
    # version 2, and the product's own layout
    written = sidd.write_sidd(str(path), out, version=2, shadows_down=False, remap='density', product_name='a name')
    rd, sd, a = read_sidd(path)
    assert type(sd).__module__.endswith('sidd2_elements.SIDD') and type(sd) is type(written)
    assert np.array_equal(a, remap.density(img)) and sd.ProductCreation.ProductName == 'a name'
    assert sd.Display.NonInteractiveProcessing[0].ProductGenerationOptions.DataRemapping.LUTName == 'DENSITY'
    assert sd.Display.InteractiveProcessing[0].GeometricTransform.Orientation.ShadowDirection == 'LEFT'
    ph = sd.ExploitationFeatures.Collections[0].Phenomenology     # SIDD 2: from the rows toward the columns
    assert abs(abs(ph.Shadow.Angle) - 90.0) < 0.2 and np.sign(ph.Shadow.Angle) == -1
    # the SICD of the complex image stored beside the SIDD; remap parameters
    sm = products.sicd_meta(out)[0]
    sidd.write_sidd(str(path), out, sicd=sm, remap_args=dict(a=0.8, c=200.0))
    rd, sd, a = read_sidd(path)
    assert len(rd.sicd_meta) == 1 and rd.sicd_meta[0].CollectionInfo.CoreName == sm.CollectionInfo.CoreName
    assert np.array_equal(a, sidd.to_display(out, a=0.8, c=200.0))
    # a tasked area inside the rectangle: zeros outside, and the valid data polygon around it
    i, j = np.mgrid[:img.shape[0], :img.shape[1]]
    disk = (i - 150) ** 2 + (j - 170) ** 2 <= 120 ** 2
    for cut in (dict(out, image=np.where(disk, img, 0)), dict(out, valid=disk)):
        sidd.write_sidd(str(path), cut)
        rd, sd, a = read_sidd(path)
        inside = sidd.orient(disk, tr)
        assert not a[~inside].any() and np.array_equal(a, sidd.to_display(cut))
        poly = sd.Measurement.ValidData.get_array(dtype=np.int64)
        assert 8 <= len(poly) <= 100 and tuple(poly[0]) == min(map(tuple, poly))
        r, c = np.nonzero(inside)
        assert poly[:, 0].min() == r.min() and poly[:, 0].max() == r.max()
        assert poly[:, 1].min() == c.min() and poly[:, 1].max() == c.max()
        assert inside[poly[:, 0], poly[:, 1]].all()
        ll = sd.GeoData.ValidData.get_array(dtype=np.float64)
        assert len(ll) == len(poly)
    # display values are written as they are
    d = sidd.to_display(out, 'brighter', shadows_down=False)
    sidd.write_sidd(str(path), dict(out, image=d), remap='brighter')
    rd, sd, a = read_sidd(path)
    assert np.array_equal(a, sidd.orient(d, tr))
    assert sd.Display.NonInteractiveProcessing[0].ProductGenerationOptions.DataRemapping.LUTName == 'BRIGHTER'
    for kw, match in ((dict(version=1), 'version'), (dict(classification='SECRET'), 'classification'),
                      (dict(remap='linear'), 'remap')):
        with pytest.raises(ValueError, match=match):
            sidd.write_sidd(str(path), out, **kw)
    with pytest.raises(ValueError, match='slant'):
        sidd.write_sidd(str(path), dict(out, plane='slant'))


def test_polygon_and_corner_fields():
    v = np.zeros((9, 12), bool)
    v[2:7, 3:10] = True
    assert sidd._hull(v).tolist() == [[2, 3], [2, 9], [6, 9], [6, 3]]
    v[:] = np.tri(9, 12, dtype=bool)                               # a triangle: row >= col
    assert sidd._hull(v).tolist() == [[0, 0], [8, 8], [8, 0]]
    v[:] = False
    v[4, 2:6] = True                                               # a line has no area: its bounding rectangle
    assert sidd._hull(v).tolist() == [[4, 2], [4, 5], [4, 5], [4, 2]]
    # degrees, minutes and seconds rounded to the second, with the carries
    assert sidd._igeolo([[40.7933, -77.8611], [10.99999, 0.5], [-0.00001, 179.99999], [-33.5, -0.25]]) == \
        '404736N0775140W' '110000N0003000E' '000000S1800000E' '333000S0001500W'


def test_sarkit_consistency(scene, need):
    if sys.version_info < (3, 11):
        pytest.skip('sarkit needs Python 3.11 or later')
    skv = need('sarkit.verification')
    out = scene.out['left']
    i, j = np.mgrid[:out['image'].shape[0], :out['image'].shape[1]]
    cases = {'SIDD 3': dict(), 'SIDD 2': dict(version=2), 'product layout': dict(shadows_down=False),
             'with its SICD': dict(sicd=products.sicd_meta(out)[0]),
             'masked': dict(valid=(i - 150) ** 2 + (j - 170) ** 2 <= 120 ** 2)}
    for name, kw in cases.items():
        path = scene.tmp / 'sarkit.nitf'
        sidd.write_sidd(str(path), out, **kw)
        with open(path, 'rb') as fh:
            con = skv.SiddConsistency.from_file(fh)
        con.check()
        fail = con.failures(omit_passed_sub=True)
        err = sorted(k for k, v in fail.items() if any(d['severity'] == 'Error' for d in v['details']))
        warn = sorted(set(fail) - set(err))
        print(f'sarkit consistency of the SIDD ({name}): {len(con.all())} checks, {len(err)} errors'
              + (f', warnings {warn}' if warn else ''))
        assert not err, (name, err)
