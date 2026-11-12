"""Several collections of one scene (fastsar.stack): passes of one simulated scene from different tracks, written as
CPHD with a known geolocation error each (the whole collection displaced by meters), formed on one Earth-fixed
grid, registered, averaged and compared: where the grid puts the targets, the translations recovered, the two
ways of applying them, the looks of the average, the coherence of a coherent and of an incoherent pair against
their true values, process_stack on each device, and the options' refusals. Needs finufft (sim.simulate)."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import io, quality, sim, stack

pytestmark = pytest.mark.needs('finufft')

LAT, LON, H0 = 40.79, -77.86, 351.0
SCENE, RES = 110.0, 0.5
# heading (degrees), grazing angle (degrees), the geolocation error written into the file (east, north; m)
PASSES = [(30.0, 35.0, (0.0, 0.0)), (30.0, 35.0, (3.1, -2.2)), (42.0, 36.0, (-4.4, 1.7)), (56.0, 34.0, (1.9, 5.3))]
SNR = 8.0                                  # of the clutter in the noisy pass of the coherent pair


def frame(heading):
    """East, north, up -> the simulator's frame of a track with this heading."""
    h = np.radians(heading)
    return np.array([[np.cos(h), np.sin(h), 0.0], [-np.sin(h), np.cos(h), 0.0], [0.0, 0.0, 1.0]])


def write(path, col, S, heading, err=(0.0, 0.0), day=1):
    """The collection as a CPHD, displaced as a whole by err (east, north; m): what an orbit known to meters does."""
    E, N, U = sim.enu_axes(LAT, LON)
    c = io.geodetic_to_ecf(LAT, LON, H0) + err[0] * E + err[1] * N
    lat, lon, h = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(c))
    sim.write_cphd(str(path), col, S.astype(np.complex64), lat, lon, h, heading=heading, scene=90.0,
                   start=f'2026-01-{day:02d}T00:00:00.000000Z')
    return str(path)


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    """Four passes over forty points on the ground and fields of two brightnesses: the second on the first's track,
    the others from other directions, each file with its own geolocation error."""
    from scipy.ndimage import gaussian_filter
    d = tmp_path_factory.mktemp('stack')
    rng = np.random.default_rng(5)
    points = (rng.random((40, 2)) - 0.5) * 84
    n = int(2.0 * (SCENE / RES) ** 2)
    xy = (rng.random((n, 2)) - 0.5) * SCENE
    f = gaussian_filter(rng.standard_normal((128, 128)), 7.0)        # fields: the sign of smoothed noise, 1 m posts
    bright = f[np.clip((xy[:, 0] + 64).astype(int), 0, 127), np.clip((xy[:, 1] + 64).astype(int), 0, 127)] > 0
    amp = np.where(bright, 1.5, 0.5) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    en = np.concatenate([np.concatenate([points, xy]), np.zeros((len(points) + n, 1))], 1)
    amp = np.concatenate([np.full(len(points), 150.0 + 0j), amp])
    paths = []
    for k, (heading, graze, err) in enumerate(PASSES):
        col = sim.make_collect(res=RES, scene=SCENE, r0=10e3, graze_deg=graze)
        paths.append(write(d / f'pass{k}.cphd', col, sim.simulate(col, en @ frame(heading), amp), heading, err, 1 + 3 * k))
    grid = stack.ground_grid((LAT, LON, H0), 96.0, 0.4, azimuth=120.0)
    E, N, U = sim.enu_axes(LAT, LON)
    # east, north (m from the center) -> pixel offsets along e1, e2
    A = np.linalg.inv(np.array([[grid['e1'] @ E, grid['e2'] @ E], [grid['e1'] @ N, grid['e2'] @ N]])) / grid['spx']
    nx, ny = grid['shape']
    inside = [p for p in points if np.all(np.abs(p @ A.T) < nx / 2 - 30)
              and np.sort(np.linalg.norm(points - p, axis=1))[1] > 8.0]                 # isolated, away from the edge
    return SimpleNamespace(dir=d, paths=paths, grid=grid, A=A, points=np.array(inside),
                           errs=np.array([p[2] for p in PASSES]))


@pytest.fixture(scope='module')
def clutter(tmp_path_factory):
    """Uniform clutter, formed on one grid: a pass, the same track again with noise (a coherent pair), and three
    passes from other directions over clutter drawn anew (speckle independent of the first's)."""
    d = tmp_path_factory.mktemp('clutter')
    rng = np.random.default_rng(11)
    grid = stack.ground_grid((LAT, LON, H0), 96.0, 0.4, azimuth=120.0)
    outs, density = [], 3.0
    for k, (heading, graze) in enumerate(((30.0, 35.0), (30.0, 35.0), (42.0, 36.0), (56.0, 34.0), (70.0, 35.5))):
        col = sim.make_collect(res=RES, scene=SCENE, r0=10e3, graze_deg=graze)
        if k != 1:
            pos, a1, _ = sim.clutter_pair(SCENE, RES, rng, per_cell=density, n_bright=0)
        S = sim.simulate(col, pos @ frame(heading), a1)
        if k == 1:
            # noise per sample against the clutter of the unambiguous area: its density times that area
            area = (sim.C / (2 * col.df)) ** 2 / np.cos(np.radians(graze))
            S = sim.add_noise(S, np.sqrt(density / RES ** 2 * area / SNR), rng)
        outs.append(fastsar.form_cphd(write(d / f'clutter{k}.cphd', col, S, heading), grid=grid, backend='cpu'))
    return outs


@pytest.fixture(scope='module')
def formed(scene):
    return [fastsar.form_cphd(p, grid=scene.grid, backend='cpu') for p in scene.paths]


@pytest.fixture(scope='module')
def registered(scene, formed):
    return stack.register(formed)


def pixels(scene, en):
    nx, ny = scene.grid['shape']
    return np.atleast_2d(en) @ scene.A.T + np.array([nx / 2, ny / 2])


def test_grid_forms():
    g = stack.ground_grid((LAT, LON, H0), (96.0, 60.0), (0.4, 0.3), azimuth=120.0)
    E, N, U = sim.enu_axes(LAT, LON)
    assert g['shape'] == (241, 201) and abs(g['e1'] @ g['e2']) < 1e-12 and np.cross(g['e1'], g['e2']) @ U > 1 - 1e-12
    assert abs(np.degrees(np.arctan2(g['e2'] @ E, g['e2'] @ N)) - 120.0) < 1e-9
    assert np.allclose(g['center'], io.geodetic_to_ecf(LAT, LON, H0)) and stack.ground_grid((LAT, LON), 10, 1)['height'] == 0
    same = [dict(lat=LAT, lon=LON, height=H0, azimuth=120.0, spx=0.4, spy=0.3, extent=(96.0, 60.0)),
            dict(center=g['center'], e1=g['e1'], e2=g['e2'], spacing=(0.4, 0.3), shape=(241, 201)),
            dict(origin=g['origin'], e1=g['e1'], e2=g['e2'], spx=0.4, spy=0.3, shape=(241, 201)), g]
    for q in same:
        h = stack.as_grid(q)
        assert stack._same_grid(g, h) and abs(h['azimuth'] - 120.0) < 1e-6 and abs(h['height'] - H0) < 1e-6
    for bad, msg in ((dict(g, e2=g['e1']), 'orthonormal'), (dict(g, e1=g['e2'], e2=g['e1']), 'upward'),
                     (dict(origin=g['origin'], e1=g['e1'], e2=g['e2'], shape=(4, 4)), 'spacing'),
                     (dict(lat=LAT, lon=LON, spacing=1.0, shape=(4, 4)), 'azimuth'),
                     (dict(azimuth=0.0, spacing=1.0, shape=(4, 4)), 'origin'),
                     (dict(lat=LAT, lon=LON, azimuth=0.0, spacing=1.0), 'shape'), (dict(g, shape=(1, 5)), 'shape')):
        with pytest.raises(ValueError, match=msg):
            stack.as_grid(bad)
    with pytest.raises(TypeError):
        stack.as_grid('grid.json')
    with pytest.raises(ValueError, match='center'):
        stack.ground_grid((LAT,), 10, 1)


def test_common_grid_formation(scene, formed):
    """Every collection lands on the grid's own pixels, each target where the grid says, displaced by the error
    written into its file."""
    E, N, U = sim.enu_axes(LAT, LON)
    for k, o in enumerate(formed):
        assert o['image'].shape == scene.grid['shape'] and o['axes'] == 'grid' and o['plane'] == 'ground'
        assert stack._same_grid(o['grid'], scene.grid) and stack._same_grid(stack.as_grid(o), scene.grid)
        assert o['shift'] == (0.0, 0.0) and o['source'] == scene.paths[k]
        want = pixels(scene, scene.points + scene.errs[k])
        off = []
        for i, j in want:
            m = quality.point_target(o['image'], (int(round(i)), int(round(j))), o['spx'], o['spy'], half=12)
            off.append((m['i'] - i, m['j'] - j))
        print(f'pass {k}: targets from the grid position plus the injected error, pixels: max {np.abs(off).max():.3f}')
        assert np.abs(off).max() < 0.05
        # the plane is the grid's: horizontal at its center, not at this collection's reference point
        assert abs(np.cross(o['e1'], o['e2']) @ o['meta']['R'] @ U - 1) < 1e-9
    again = fastsar.form_cphd(scene.paths[2], grid=formed[0], backend='cpu')      # a product as the grid: its pixels
    assert np.array_equal(again['image'], formed[2]['image'])


def test_moving_beam_on_a_grid(tmp_path):
    """A moving-beam collection (stripmap: the scene reference point follows the antenna), formed as a mosaic on an
    Earth-fixed grid turned 45 degrees from its track: the targets where the grid says."""
    col = sim.make_collect(res=0.5, scene=60.0, r0=6e3, graze_deg=35.0)
    srp = np.zeros_like(col.ant)
    srp[:, 1] = col.ant[:, 1]
    tg = np.array([[0.0, 0.0, 0.0], [12.0, -15.0, 0.0], [-15.0, 18.0, 0.0], [10.0, 10.0, 0.0]])
    dr = np.linalg.norm(tg[None] - col.ant[:, None], axis=-1) - np.linalg.norm(srp - col.ant, axis=1)[:, None]
    S = np.exp(-4j * np.pi / sim.C * col.freqs[None, :, None] * dr[:, None, :]).sum(-1).astype(np.complex64)
    sim.write_cphd(str(tmp_path / 'strip.cphd'), col, S, LAT, LON, H0, heading=30.0, srp=srp)
    grid = stack.ground_grid((LAT, LON, H0), 70.0, 0.4, azimuth=75.0)
    o = fastsar.form_cphd(str(tmp_path / 'strip.cphd'), grid=grid, backend='cpu')
    assert o['mode'] == 'moving' and o['axes'] == 'grid' and o['image'].shape == grid['shape']
    assert stack._same_grid(stack.as_grid(o), grid)
    E, N, U = sim.enu_axes(LAT, LON)
    A = np.linalg.inv(np.array([[grid['e1'] @ E, grid['e2'] @ E], [grid['e1'] @ N, grid['e2'] @ N]])) / grid['spx']
    want = (tg @ frame(30.0).T)[:, :2] @ A.T + np.array(grid['shape']) / 2
    for i, j in want:
        m = quality.point_target(o['image'], (int(round(i)), int(round(j))), 0.4, 0.4, half=10)
        assert abs(m['i'] - i) < 0.1 and abs(m['j'] - j) < 0.1, (i, j, m['i'], m['j'])
    with pytest.raises(ValueError, match='spotlight'):                 # the moving beam's own refusals stand
        fastsar.process_cphd(str(tmp_path / 'strip.cphd'), grid=grid, autofocus='quadratic', backend='cpu')
    p = fastsar.process_cphd(str(tmp_path / 'strip.cphd'), grid=grid, backend='cpu')
    assert p['plane'] == 'ground' and np.array_equal(p['image'], o['image'])


def test_grid_refusals(scene, formed):
    p = scene.paths[0]
    for kw, msg in ((dict(spacing=0.5), 'spacing'), (dict(extent=(50, 50)), 'extent'), (dict(height=0.0), 'height')):
        with pytest.raises(ValueError, match='grid= gives'):
            fastsar.form_cphd(p, grid=scene.grid, backend='cpu', **kw)
    with pytest.raises(ValueError, match='give grid='):
        fastsar.form_cphd(p, shift=(1.0, 0.0), backend='cpu')
    with pytest.raises(ValueError, match='shift must be'):
        fastsar.form_cphd(p, grid=scene.grid, shift=(1.0, np.nan), backend='cpu')
    with pytest.raises(ValueError, match='line of sight'):             # the phase history's autofocus, likewise
        fastsar.form_cphd(scene.paths[2], grid=scene.grid, autofocus=True, backend='cpu')
    with pytest.raises(ValueError, match='aperture'):
        fastsar.form_cphd(p, grid=scene.grid, aperture=(10, 5), backend='cpu')
    part = fastsar.form_cphd(p, grid=scene.grid, aperture=(40, 350), backend='cpu')
    assert part['pulses'] == (40, 350) and any('pulses 40 to 349' in n for n in part['notes'])
    for kw, msg in ((dict(plane='slant'), 'ground plane'), (dict(area='black'), 'processed area'),
                    (dict(area=4.0), 'processed area'), (dict(weighting='sva'), 'slant plane'),
                    (dict(spacing=0.5), 'grid gives'), (dict(extent=(40, 40)), 'grid gives'), (dict(height=1.0), 'plane')):
        with pytest.raises(ValueError, match=msg):
            fastsar.process_cphd(p, grid=scene.grid, backend='cpu', **kw)
    with pytest.raises(ValueError, match='first axis'):                # the autofocus needs azimuth along e1
        fastsar.process_cphd(scene.paths[2], grid=scene.grid, autofocus='quadratic', backend='cpu')
    slant = fastsar.process_cphd(p, backend='cpu', autofocus=False)
    with pytest.raises(ValueError, match='slant-plane'):
        stack.as_grid(slant)


def test_process_cphd_on_a_grid(scene, formed, tmp_path):
    """process_cphd(grid=): the image on the grid as the product, weighted, at baseband, written as a SICD whose
    own projection puts the targets where they are; autofocus where the grid follows the line of sight."""
    sarpy = pytest.importorskip('sarpy')
    from sarpy.io.complex.converter import open_complex
    from sarpy.geometry import point_projection
    o = fastsar.process_cphd(scene.paths[2], str(tmp_path / 'g.nitf'), grid=scene.grid, backend='cpu')
    assert o['plane'] == 'ground' and o['weighting'] == 'taylor' and o['baseband'] and o['autofocus'] is None
    assert any('no autofocus' in n for n in o['notes']) and o['image'].shape == scene.grid['shape']
    assert np.allclose(np.abs(o['image']), np.abs(formed[2]['image']), atol=2e-3 * np.abs(o['image']).max())
    E, N, U = sim.enu_axes(LAT, LON)
    en = scene.points + scene.errs[2]
    ecf = io.geodetic_to_ecf(LAT, LON, H0) + en[:, :1] * E + en[:, 1:] * N
    rd = open_complex(str(tmp_path / 'g.nitf'))
    rc, _, _ = point_projection.ground_to_image(ecf, rd.sicd_meta)
    pix = rd[:, :]
    for r, c in rc:
        m = quality.point_target(pix, (int(round(r)), int(round(c))), 0.4, 0.4, half=10)
        assert abs(m['i'] - r) < 0.1 and abs(m['j'] - c) < 0.1
    # a grid along this collection's own line of sight (an earlier product of it): the autofocus runs
    own = fastsar.process_cphd(scene.paths[2], plane='ground', autofocus=False, backend='cpu')
    af = fastsar.process_cphd(scene.paths[2], grid=own, autofocus='quadratic', backend='cpu')
    assert af['image'].shape == own['image'].shape and af['autofocus_info'] is not None
    sd = fastsar.process_cphd(scene.paths[2], grid=own, product='sidd', autofocus=False, backend='cpu')
    assert sd['image'].dtype == np.float32 and sd['image'].shape == own['image'].shape and sd['image'].min() >= 0


def test_register_recovers_the_errors(scene, formed, registered):
    """One translation per image, to a few hundredths of a pixel; the phase ramp and the formation on the displaced
    grid give the same image and the same geolocation."""
    sp = scene.grid['spx']
    worst = 0.0
    for k, r in enumerate(registered):
        g = r['registration']
        if k == 0:
            assert g['reference'] and r['shift'] == (0.0, 0.0) and r['image'] is formed[0]['image']
            continue
        want = (scene.errs[k] - scene.errs[0]) @ scene.A.T * sp           # m along e1, e2
        err = (np.array(g['shift']) - want) / sp
        print(f"pass {k}: shift {np.round(g['shift'], 3)} m, error {np.round(err, 3)} pixels, significance "
              f"{g['significance']:.0f}, tiles {g['tiles_used']}/{g['tiles']}, residual rms {g['residual_rms'] / sp:.3f} pixels")
        worst = max(worst, np.abs(err).max())
        # the same track: the same speckle; another direction: other speckle on a 240 pixel image
        assert g['found'] and g['significance'] > 15 and np.abs(err).max() < (0.03 if k == 1 else 0.08)
        assert g['tiles'] == 16 and g['tiles_used'] >= 4 and g['residual_rms'] / sp < 0.6
        assert np.allclose(r['shift'], g['shift']) and stack._same_grid(stack.as_grid(r), scene.grid)
        # the target positions, now the reference's
        for i, j in pixels(scene, scene.points[:3] + scene.errs[0]):
            m = quality.point_target(r['image'], (int(round(i)), int(round(j))), sp, sp, half=12)
            assert abs(m['i'] - i) < 0.08 and abs(m['j'] - j) < 0.08
    print(f'largest error of a recovered shift: {worst:.3f} pixels')
    k = 2
    again = stack.register([formed[0], formed[k]], apply='reform')[1]
    a, b = registered[k]['image'], again['image']
    c = slice(24, -24)
    db = 20 * np.log10(np.linalg.norm(a[c, c] - b[c, c]) / np.linalg.norm(b[c, c]))
    print(f'phase ramp against formation on the displaced grid, away from the edges: {db:.1f} dB')
    assert db < -35 and np.allclose(again['shift'], registered[k]['shift'], atol=1e-9)
    assert np.allclose(io.local_to_ecf(again['origin'], again['meta']),
                       io.local_to_ecf(registered[k]['origin'], registered[k]['meta']), atol=1e-6)
    assert not again.get('baseband') and not registered[k].get('baseband')
    # a baseband product is translated as one, to the same pixels
    from fastsar import planes
    bb = planes.deramp(dict(formed[k], image=formed[k]['image'].copy()))
    sh = planes.deramp(stack.shift_image(bb, registered[k]['shift']), sign=+1)
    assert np.linalg.norm(sh['image'][c, c] - a[c, c]) < 1e-3 * np.linalg.norm(a[c, c])
    # estimate only; a product off the grid
    est = stack.register([formed[0], formed[k]], apply=None)[1]
    assert est['image'] is formed[k]['image'] and np.allclose(est['registration']['shift'], registered[k]['shift'])
    with pytest.raises(ValueError, match='not on the grid'):
        stack.register([formed[0], dict(formed[1], spx=0.41)])


def test_mean_looks(clutter):
    """Four images with independent speckle: the average's looks are four times one image's."""
    ind = [clutter[k] for k in (0, 2, 3, 4)]
    avg = stack.mean(ind)
    assert avg['product'] == 'mean' and avg['count'] == 4 and avg['image'].dtype == np.float32
    one = np.mean(avg['looks_single'])
    print(f"looks: one image {np.round(avg['looks_single'], 2)}, the average of the first n {np.round(avg['looks_by_n'], 2)}, "
          f"on {avg['tiles_selected']} of {avg['tiles']} tiles; independent speckle would give {avg['looks_independent']:.2f}")
    assert avg['tiles_selected'] >= 5 and 0.75 < one < 1.2
    for n in (2, 3, 4):
        assert abs(avg['looks_by_n'][n - 1] / (n * one) - 1) < 0.15
    assert abs(avg['looks'] / avg['looks_ideal'] - 1) < 0.15 and abs(avg['looks'] / avg['looks_independent'] - 1) < 0.1
    fine = stack.mean(ind, tile=16)                                    # more, smaller tiles: the same within their scatter
    assert fine['tiles_selected'] > 3 * avg['tiles_selected'] and abs(fine['looks'] / avg['looks'] - 1) < 0.15
    p = np.mean([np.abs(o['image']) ** 2 for o in ind], axis=0)
    assert avg['gains'] == [1.0] * 4 and np.allclose(avg['image'] ** 2, p, rtol=2e-4, atol=1e-6 * p.max())
    w = stack.mean(iter(ind[:2]), weights=[3.0, 1.0])                  # any iterable
    p = 0.75 * np.abs(ind[0]['image']) ** 2 + 0.25 * np.abs(ind[1]['image']) ** 2
    assert np.allclose(w['image'] ** 2, p, rtol=2e-4, atol=1e-6 * p.max())
    # an image 3 dB brighter: scaled back by equalize (the average is then in the first image's units), kept without
    twice = dict(ind[1], image=ind[1]['image'] * np.float32(np.sqrt(2.0)))
    for how in ('mean', 'median'):
        g0 = stack.mean([ind[0], ind[1]], equalize=how)['gains'][1]
        eq = stack.mean([ind[0], twice], equalize=how)
        assert abs(g0 - 1) < 0.1 and abs(eq['gains'][1] * 2 / g0 - 1) < 1e-3 and eq['gains'][0] == 1.0
    two = stack.mean([ind[0], ind[1]])['looks']
    assert abs(stack.mean([ind[0], twice])['looks'] / two - 0.9) < 0.05      # (1 + 2)^2 / (1 + 4) of two looks
    assert np.array_equal(avg['origin'], ind[0]['origin']) and avg['meta'] is ind[0]['meta']
    with pytest.raises(ValueError, match='equalize'):
        stack.mean(ind, equalize='max')
    with pytest.raises(ValueError, match='not on the grid'):
        stack.mean([ind[0], dict(ind[1], spy=0.5)])
    # a pixel that is 0 in one image is averaged over the others
    holed = ind[1]['image'].copy()
    holed[100:110, 50:60] = 0
    h = stack.mean([ind[0], dict(ind[1], image=holed)])
    assert np.allclose(h['image'][100:110, 50:60], np.abs(ind[0]['image'][100:110, 50:60]), rtol=1e-5)


def test_coherence(clutter):
    """The same track with noise: the coherence its signal-to-noise ratio gives. Another direction over other
    clutter: the floor of the estimator, which is above that of independent pixels."""
    a, b, other = clutter[0], clutter[1], clutter[2]
    nx, ny = a['image'].shape
    core = (slice(24, -24),) * 2
    coh = stack.coherence(a, b, window=7)
    true = 1 / np.sqrt(1 + 1 / SNR)
    got = float(np.median(coh['image'][core]))
    print(f"coherent pair: median {got:.3f} (true {true:.3f}); floor {coh['floor']['median']:.3f} "
          f"({coh['samples']:.1f} independent samples in 49 pixels); overlap {np.round(coh['overlap'], 2)}")
    assert coh['product'] == 'coherence' and coh['image'].shape == (nx, ny) and coh['image'].max() <= 1 + 1e-5
    assert abs(got - true) < 0.015 and abs(coh['median'] - true) < 0.02 and min(coh['overlap']) > 0.99
    assert coh['baseline']['inside'] and abs(coh['baseline']['d_azimuth']) < 0.01 and coh['window'] == 7
    # the floor: above the 0.13 of 49 independent pixels, since the image is sampled finer than its resolution
    assert 0.17 < coh['floor']['median'] < 0.35 and 6 < coh['samples'] < 30
    inc = stack.coherence(a, other, window=7)
    print(f"incoherent pair: median {inc['median']:.3f}, its floor {inc['floor']['median']:.3f}; "
          f"{inc['baseline']['d_azimuth']:.1f} degrees apart in azimuth against a critical {inc['baseline']['critical_azimuth']:.2f}")
    assert abs(inc['median'] - inc['floor']['median']) < 0.02 and not inc['baseline']['inside']
    assert any('critical baseline' in n for n in inc['notes']) and inc['above_0p7'] < 0.01
    for kw in (dict(common_band=False), dict(fringe=None), dict(fringe='global'), dict(window=11)):
        c = stack.coherence(a, b, **kw)
        assert abs(float(np.median(c['image'][core])) - true) < 0.015, kw
    wide, narrow = stack.coherence(a, other, window=11), stack.coherence(a, other, window=5, floor=False)
    assert wide['floor']['median'] < inc['floor']['median'] < narrow['median'] and narrow['floor'] is None
    # a fringe across the image: removed in tiles and globally, and lost on the window without
    i = np.arange(nx)[:, None].astype(np.float32)
    tilted = dict(b, image=(b['image'] * np.exp(2j * np.pi * 0.11 * i)).astype(np.complex64))
    assert abs(stack.coherence(a, tilted)['median'] - coh['median']) < 0.02
    assert abs(stack.coherence(a, tilted, fringe='global')['median'] - coh['median']) < 0.02
    assert stack.coherence(a, tilted, fringe=None, common_band=False)['median'] < 0.6
    ch = stack.change(a, b, window=7)
    assert ch['product'] == 'change' and np.array_equal(ch['coherence'], coh['image']) and ch['image'].dtype == np.float32
    assert abs(float(np.median(ch['image'][core]))) < 0.2 and 1.0 < ch['ratio_sigma'] < 3.0
    half = dict(b, image=b['image'].copy())
    half['image'][:, :ny // 2] *= np.float32(2.0)                      # 6 dB brighter over half of the image
    r = stack.change(a, half)['image']
    assert abs(np.median(r[core][:, :60]) - np.median(r[core][:, -60:]) - 6.02) < 0.3
    with pytest.raises(ValueError, match='complex'):
        stack.coherence(a, dict(b, image=np.abs(b['image'])))
    with pytest.raises(ValueError, match='fringe'):
        stack.coherence(a, b, fringe='every')


def test_baseline_from_the_files(scene):
    b = stack.baseline(scene.paths[0], scene.paths[1])
    assert b['inside'] and b['overlap'] > 0.9 and abs(b['d_graze']) < 0.03 and abs(b['days'] - 3.0) < 1e-6
    b = stack.baseline(scene.paths[0], scene.paths[2])
    assert not b['inside'] and b['overlap_azimuth'] == 0 and abs(b['d_azimuth'] - 12.0) < 0.2
    assert abs(b['d_graze'] - 1.0) < 0.05 and 0.5 < b['critical_azimuth'] < 5 and b['critical_graze'] > 1.0
    g = stack.geometry(scene.paths[3])
    assert g['ant'].shape[1] == 3 and g['band'][1] > g['band'][0] and g['start'].startswith('2026-01-10')


def read_sidd(path):
    from sarpy.io.product.converter import open_product
    rd = open_product(str(path))
    return (rd.sidd_meta[0] if isinstance(rd.sidd_meta, (list, tuple)) else rd.sidd_meta), rd[:, :]


def test_process_stack_mean(scene, registered, tmp_path, need):
    """From four files to the SIDD of the average: the registrations, the pixels, and metadata that say what the
    product is."""
    need('sarpy')
    f = tmp_path / 'mean.nitf'
    o = fastsar.process_stack(scene.paths, str(f), grid=scene.grid, backend='cpu', range_gate='auto')
    assert o['product'] == 'mean' and o['count'] == 4 and o['path'] == str(f) and o['image'].shape == scene.grid['shape']
    assert {'read', 'form', 'register', 'combine', 'write', 'total', 'megapixels_per_second'} <= set(o['timing'])
    for k in range(4):
        assert np.allclose(o['registrations'][k]['shift'], registered[k]['registration']['shift'], atol=1e-6)
        assert o['inputs'][k]['path'] == scene.paths[k] and np.allclose(o['inputs'][k]['shift'], registered[k]['shift'])
    ref = stack.mean(registered, equalize='mean')
    assert np.allclose(o['image'], ref['image'], rtol=1e-4, atol=1e-5 * ref['image'].max())
    assert o['looks'] is None or abs(o['looks'] - ref['looks']) < 1e-3
    sd, pix = read_sidd(f)
    assert pix.dtype == np.uint8 and sorted(pix.shape) == sorted(o['image'].shape)
    assert np.array_equal(pix, fastsar.sidd.to_display(o, 'nrl'))
    m = o['metadata']
    assert m.ProductCreation.ProductClass == 'Detected Image'
    assert m.ProductCreation.ProductType == 'Multi-image incoherent average'
    ext = {e.attrib['name']: e.text for e in m.element.iter() if e.tag.endswith('ProductCreationExtension')}
    assert ext['NumberOfCollections'] == '4' and ext['ReferenceCollection'] == '1' and 'reference' in ext['Geolocation']
    assert 'translation' in ext['Registration'] and 'brightness' in ext['Combination']
    cols = m.ExploitationFeatures.Collections
    assert len(cols) == 4 and len({c.attrib['identifier'] for c in cols}) == 4
    days = [str(c.Information.CollectionDateTime)[:10] for c in cols]
    assert days == ['2026-01-01', '2026-01-04', '2026-01-07', '2026-01-10']
    az = [float(c.Geometry.Azimuth) for c in cols]
    assert abs((az[2] - az[0]) % 360 - 12.0) < 0.5 and abs(float(cols[3].Geometry.Graze) - 34.0) < 0.2
    assert len(sd.ExploitationFeatures.Collections) == 4               # as sarpy reads it
    # the SIDD's projection is the reference's: the targets where the reference has them
    from sarpy.geometry import point_projection
    E, N, U = sim.enu_axes(LAT, LON)
    en = scene.points + scene.errs[0]
    rc, _, _ = point_projection.ground_to_image(io.geodetic_to_ecf(LAT, LON, H0) + en[:, :1] * E + en[:, 1:] * N, sd)
    for r, c in rc:
        w = pix[int(round(r)) - 5:int(round(r)) + 6, int(round(c)) - 5:int(round(c)) + 6]
        a, b = np.unravel_index(np.argmax(w), w.shape)
        assert abs(a - 5) <= 1 and abs(b - 5) <= 1
    import sys
    if sys.version_info >= (3, 11):                                    # the schema and sarkit's checks of the product
        skv = need('sarkit.verification')
        with open(f, 'rb') as fh:
            con = skv.SiddConsistency.from_file(fh)
        con.check()
        fail = con.failures(omit_passed_sub=True)
        err = sorted(k for k, v in fail.items() if any(q['severity'] == 'Error' for q in v['details']))
        print(f'sarkit consistency of the SIDD of a mean: {len(con.all())} checks, {len(err)} errors')
        assert not err, err
    # the reference chosen, the registration left out, the second way of applying it
    o2 = fastsar.process_stack(scene.paths[:3], grid=scene.grid, reference=2, backend='cpu')
    want = (scene.errs[0] - scene.errs[2]) @ scene.A.T
    assert o2['registrations'][2]['reference'] and np.abs(np.array(o2['registrations'][0]['shift_px']) - want).max() < 0.08
    o3 = fastsar.process_stack(scene.paths[:2], grid=scene.grid, register=None, backend='cpu')
    assert o3['registrations'][1]['shift'] == (0.0, 0.0) and o3['inputs'][1]['shift'] == (0.0, 0.0)
    o4 = fastsar.process_stack(scene.paths[:2], grid=scene.grid, register='reform', backend='cpu')
    o5 = fastsar.process_stack(scene.paths[:2], grid=scene.grid, backend='cpu')
    c = (slice(24, -24),) * 2
    assert np.linalg.norm(o4['image'][c] - o5['image'][c]) < 1e-2 * np.linalg.norm(o5['image'][c])


def test_process_stack_default_grid_and_pairs(scene, registered, tmp_path, need):
    """The default grid follows the reference's line of sight; coherence and change of a pair as GeoTIFF."""
    o = fastsar.process_stack(scene.paths[:2], backend='cpu', extent=(70.0, 70.0), equalize=None)
    g = o['grid']
    lk = stack._look(stack.geometry(scene.paths[0]), g['center'])
    assert abs((g['azimuth'] - lk['azimuth'] - 180.0 + 180.0) % 360.0 - 180.0) < 0.5      # e2 away from the radar
    assert abs(g['spx'] - g['spy']) < 1e-12 and 0.25 < g['spx'] < 0.6 and 'path' not in o and o['gains'] == [1.0, 1.0]
    assert abs(fastsar.sidd.shadows_down(o)['angle']) < 0.6            # the reference's shadows fall down the rows
    E, N, U = sim.enu_axes(LAT, LON)
    A = np.linalg.inv(np.array([[g['e1'] @ E, g['e2'] @ E], [g['e1'] @ N, g['e2'] @ N]])) / g['spx']
    assert np.abs(np.array(o['registrations'][1]['shift_px']) - (scene.errs[1] - scene.errs[0]) @ A.T).max() < 0.08
    need('rasterio')
    import rasterio
    from rasterio.warp import transform as warp
    c = fastsar.process_stack(scene.paths[:2], str(tmp_path / 'coh.tif'), product='coherence', grid=scene.grid,
                              backend='cpu')
    # the pair is formed from the pulses on one interval of azimuth; with each collection's own aperture it is
    # the coherence of the two images registered above
    own = fastsar.process_stack(scene.paths[:2], product='coherence', grid=scene.grid, aperture='square', backend='cpu')
    ref = stack.coherence(registered[0], registered[1], 7)
    assert c['product'] == 'coherence' and np.allclose(own['image'], ref['image'], atol=2e-3) and c['median'] > 0.9
    assert any('one interval of azimuth' in n for n in c['notes']) and min(c['overlap']) > 0.99
    assert c['median'] >= own['median'] - 0.005 and c['inputs'][0]['pulses'] != own['inputs'][0]['pulses'] or \
        c['inputs'][1]['pulses'] != own['inputs'][1]['pulses']
    with rasterio.open(tmp_path / 'coh.tif') as src:
        assert src.count == 1 and src.crs.to_epsg() == 32618 and not c['metadata']['resampled']
        assert np.array_equal(src.read(1), c['image'])
        i, j = 37, 181                                                 # a pixel's map position is the grid's
        t = src.transform
        x, y = t.a * (j + 0.5) + t.b * (i + 0.5) + t.c, t.d * (j + 0.5) + t.e * (i + 0.5) + t.f
        lon, lat = (v[0] for v in warp(src.crs, 'EPSG:4326', [x], [y]))
        la, lo, _ = io.ecf_to_geodetic(scene.grid['origin'] + i * 0.4 * scene.grid['e1'] + j * 0.4 * scene.grid['e2'])
        assert abs(lat - la) * 111e3 < 0.02 and abs(lon - lo) * 111e3 * np.cos(np.radians(LAT)) < 0.02
    ch = fastsar.process_stack(scene.paths[:2], str(tmp_path / 'ch.tif'), product='change', grid=scene.grid, backend='cpu')
    with rasterio.open(tmp_path / 'ch.tif') as src:
        assert src.count == 2 and np.array_equal(src.read(1), ch['coherence']) and np.array_equal(src.read(2), ch['image'])
    for kw, msg in ((dict(product='coherence'), 'two collections'), (dict(product='ratio'), 'product must be'),
                    (dict(reference=7), 'reference'), (dict(grid=scene.grid, spacing=0.5), 'grid= gives'),
                    (dict(register='affine'), 'register must be'), (dict(weights=[1.0]), 'weights')):
        with pytest.raises(ValueError, match=msg):
            fastsar.process_stack(scene.paths[:3], backend='cpu', **kw)
    with pytest.raises(ValueError, match='two collections or more'):
        fastsar.process_stack(scene.paths[:1], backend='cpu')


@pytest.mark.parametrize('backend', [pytest.param('cuda', marks=pytest.mark.cuda), 'jax'])
def test_process_stack_devices(scene, backend, request):
    """The same products where the images live on a device."""
    from fastsar import _xp
    if backend == 'jax':
        request.getfixturevalue('jax_cpu')
    g = stack.ground_grid((LAT, LON, H0), 64.0, 0.4, azimuth=120.0)
    use = [scene.paths[0], scene.paths[2]]
    ref = fastsar.process_stack(use, grid=g, backend='cpu')
    o = fastsar.process_stack(use, grid=g, backend=backend)
    assert isinstance(o['image'], np.ndarray) and o['image'].shape == ref['image'].shape
    assert np.allclose(o['registrations'][1]['shift'], ref['registrations'][1]['shift'], atol=0.02 * 0.4)
    c = (slice(16, -16),) * 2
    d = np.linalg.norm(o['image'][c] ** 2 - ref['image'][c] ** 2) / np.linalg.norm(ref['image'][c] ** 2)
    print(f"{backend}: mean intensity against the cpu's {20 * np.log10(d):.1f} dB")
    assert d < 0.02
    pair = scene.paths[:2]
    rc = fastsar.process_stack(pair, product='change', grid=g, backend='cpu')
    oc = fastsar.process_stack(pair, product='change', grid=g, backend=backend)
    assert np.abs(oc['coherence'][c] - rc['coherence'][c]).max() < 0.05 and abs(oc['median'] - rc['median']) < 0.01
    assert np.median(np.abs(oc['image'][c] - rc['image'][c])) < 0.05
    # the stack functions on device arrays: results of the same kind
    kind = {'cuda': 'cupy', 'jax': 'jax'}[backend]
    host = [fastsar.form_cphd(p, grid=g, backend='cpu') for p in pair]
    dev = stack.register([dict(q, image=_xp.device(q['image'], backend)) for q in host])
    cpu = stack.register(host)
    assert _xp.kind(dev[1]['image']) == kind and np.allclose(dev[1]['shift'], cpu[1]['shift'], atol=0.01 * 0.4)
    assert np.linalg.norm(_xp.host(dev[1]['image'])[c] - cpu[1]['image'][c]) < 2e-3 * np.linalg.norm(cpu[1]['image'][c])
    m = stack.mean(dev)
    assert _xp.kind(m['image']) == kind
    assert np.allclose(_xp.host(m['image']), stack.mean(cpu)['image'], rtol=5e-3, atol=1e-3 * float(m['image'].max()))
    assert _xp.kind(stack.coherence(dev[0], dev[1])['image']) == kind
