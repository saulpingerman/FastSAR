"""fastsar.terrain: the defocus of scatterers off the image plane, from the geometry alone. The model against the
closed form, a straight track and simulated targets from 300 m below to 300 m above the plane (the squinted
spotlight on a curved track of test_planes, flown to either side); the correction against the targets' on-plane
responses and against exact backprojection at their true heights; the layover map; the DEM's conventions; the
devices; and process_cphd(dem=) to a SICD."""
import dataclasses
import io as _io
import sys
import warnings

import numpy as np
import pytest

import fastsar
from fastsar import io, planes, products, quality, sim, terrain

C = 299792458.0
LAT, LON, H0, HEADING = 40.79, -77.86, 351.0, 30.0
HTS = np.array([-300.0, -100.0, -30.0, 0.0, 30.0, 100.0, 300.0])
# where the targets appear, m from the scene center across and along the line of sight: every one in its own
# columns, the high and low ones in the middle rows, where their smear of up to 50 m stays inside the image
SPOT = np.array([[10.0, -60.0], [-45.0, 20.0], [45.0, -20.0], [0.0, 0.0], [-45.0, 40.0], [45.0, -40.0],
                 [-10.0, 60.0]])


def _collect(squint, left, curved=True):
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    y = col.ant[:, 1] / np.ptp(col.ant[:, 1])
    ant = col.ant.copy()
    ant[:, 1] -= 10e3 * np.cos(np.radians(35)) * np.tan(np.radians(squint))
    if curved:
        ant[:, 0] += 160.0 * y ** 2 + 120.0 * y ** 3
        ant[:, 2] += -100.0 * y ** 2 + 6.0 * y
    return dataclasses.replace(col, ant=ant[::-1].copy() if left else ant)


def _los_axes(ant):
    e2 = -ant[len(ant) // 2] * [1.0, 1.0, 0.0]
    e2 /= np.linalg.norm(e2)
    return np.cross(e2, [0.0, 0.0, 1.0]), e2


def _product(col, shape=(480, 440), sp=(0.32, 0.38), image=None):
    """A ground-plane product on axes along the line of sight for a collection of the simulator, whose frame is
    then the local frame (the scene center at LAT, LON, H0), without forming anything."""
    ant = np.asarray(col.ant, np.float64)
    o0 = sim.to_ecf(np.zeros(3), LAT, LON, H0, HEADING)
    t = 1.0 + np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(ant, axis=0), axis=1)]) / 200.0
    e1, e2 = _los_axes(ant)
    nx, ny = shape
    return dict(meta=dict(tx=ant, rcv=ant, tx_time=t, R=sim.to_ecf(np.eye(3), LAT, LON, H0, HEADING) - o0, origin=o0),
                origin=-(nx // 2) * sp[0] * e1 - (ny // 2) * sp[1] * e2, e1=e1, e2=e2, spx=sp[0], spy=sp[1],
                band=(col.fmin, col.fmin + (col.K - 1) * col.df), plane='ground', mode='spotlight', axes='los',
                baseband=True, notes=[], image=np.zeros(shape, np.complex64) if image is None else image)


def _plane_height(out):
    """The ellipsoid height of the image plane at its center."""
    nx, ny = out['image'].shape
    c = out['origin'] + nx / 2 * out['spx'] * out['e1'] + ny / 2 * out['spy'] * out['e2']
    return float(io.ecf_to_geodetic(io.local_to_ecf(c, out['meta']))[2])


# ------------------------------------------------------------------------------------------------ the model
def test_straight_track_needs_no_correction():
    """On a straight track the points of the circle about the track share one range history: the phase is zero at
    any height and squint, and nothing is applied."""
    for squint in (0.0, 20.0):
        out = _product(_collect(squint, False, curved=False))
        c, band, res = terrain.coefficients(out, [240, 20, 460], [220, 400, 30], [300.0, -300.0, 150.0], order=4)
        assert np.abs(c).max() < 1e-6 and 0.3 < band[1] < 0.5
        xi, phi = terrain.phase_error(out, 240, 220, 300.0)
        assert np.abs(phi).max() < 1e-6 and np.all(np.diff(xi) != 0)
    rng = np.random.default_rng(0)
    img = (rng.normal(size=(480, 440)) + 1j * rng.normal(size=(480, 440))).astype(np.complex64)
    out = _product(_collect(20.0, False, curved=False), image=img)
    with pytest.warns(terrain.TerrainWarning):                       # 300 m from the stated scene height
        info = terrain.focus(out, terrain.as_dem(H0 + 300.0))
    assert not info['applied'] and 'below tol' in info['reason'] and out['image'] is img
    assert info['height'][0] == pytest.approx(300.0, abs=0.01) and abs(info['quadratic_rad'][1]) < 1e-6


def test_scaling_law():
    """The first-order law: at the band edge  pi / 2 (h / lambda) dtheta^2 (R / rho) / w_z  for a track that bends
    out of the slant plane with radius rho; linear in the height, quadratic in the aperture, and of the sign of
    the bend."""
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    lam = C / (col.fmin + (col.K - 1) / 2 * col.df)
    w = np.array([np.sin(np.radians(35)), 0.0, np.cos(np.radians(35))])       # the slant plane's normal
    y = col.ant[:, 1]

    def edge(h, rho, part=1.0):
        keep = np.abs(y) <= part * np.abs(y).max() + 1e-9
        ant = (col.ant + (0.5 * y ** 2 / rho)[:, None] * w)[keep]
        out = _product(dataclasses.replace(col, ant=ant))
        c, band, res = terrain.coefficients(out, 240, 220, h, order=4)
        dth = np.ptp(y[keep]) / 10e3
        return c[2], np.pi / 2 * h / lam * dth ** 2 * 10e3 / rho / w[2], res
    c, law, res = edge(20.0, 40e3)
    assert res < 1e-3 and 0.5 < law < 0.7 and c == pytest.approx(law, rel=0.02), (c, law)
    assert edge(-20.0, 40e3)[0] == pytest.approx(-c, rel=0.01) and edge(20.0, -40e3)[0] == pytest.approx(-c, rel=0.01)
    assert edge(40.0, 40e3)[0] == pytest.approx(2 * c, rel=0.01)
    assert edge(20.0, 80e3)[0] == pytest.approx(c / 2, rel=0.01)
    assert edge(20.0, 40e3, part=0.5)[0] == pytest.approx(c / 4, rel=0.02)


def test_model_over_the_image():
    """focus's model of the coefficients over pixels and heights reproduces the exact fit at points and heights
    it was not built from."""
    out = _product(_collect(20.0, False))
    grid, fr, ant, fc = terrain._scene(out)
    band = terrain._band(grid, fr, ant, fc)
    model = terrain._Model(grid, fr, ant, fc, band, 6, -300.0, 300.0)
    rng = np.random.default_rng(3)
    i, j, h = rng.uniform(0, 479, 40), rng.uniform(0, 439, 40), rng.uniform(-300, 300, 40)
    exact = terrain.coefficients(out, i, j, h, order=6, band=band)[0]
    v = np.linspace(-1, 1, 101)[:, None] ** np.arange(7)
    assert np.abs(v @ (model(i, j, h) - exact.T)).max() < 0.02       # rad, of phases that reach 150


# ------------------------------------------------------------------------------------------------ simulated targets
def _scene(squint, left, directory):
    """Point targets that appear at SPOT from the heights HTS, formed on the plane of the scene center."""
    col = _collect(squint, left)
    geo = _product(col)
    fr = planes._frame(geo)
    e1, e2 = _los_axes(col.ant)
    a, r = planes.ground_to_slant(fr, SPOT[:, :1] * e1 + SPOT[:, 1:] * e2)
    tg = planes.slant_to_ground(fr, a, r, HTS)                       # the simulator's frame: the plane is z = 0
    path = str(directory / 'c.cphd')
    sim.write_cphd(path, col, sim.simulate(col, tg, np.ones(len(tg))), LAT, LON, H0, heading=HEADING, scene=160.0)
    got = {}

    def plan(g):
        got.update(g)
        f = planes.slant_frame(g['meta'], g['center'])
        t = np.linspace(-85, 85, 5)
        pts = g['center'] + (t[:, None, None] * g['e1'] + t[None, :, None] * g['e2']).reshape(-1, 3)
        band = (g['fmin'], g['fmin'] + (g['S'].shape[1] - 1) * g['df'])
        b = planes.sampling(f, g['ant'], band, pts, g['e1'], g['e2'])
        return dict(spacing=(0.96 / b['bx'], 0.96 / b['by']), bounds=(-85, 85, -85, 85))
    out = fastsar.form_cphd(path, axes='los', window=False, backend='cpu', aperture='full', _plan=plan)
    planes.deramp(out)
    tgl = io.ecf_to_local(sim.to_ecf(tg, LAT, LON, H0, HEADING), out['meta'])
    fr = planes._frame(out)
    y = planes.slant_to_ground(fr, *planes.ground_to_slant(fr, tgl), out['origin'][2])    # where they lay over
    pix = np.stack([(y - out['origin']) @ out['e1'] / out['spx'], (y - out['origin']) @ out['e2'] / out['spy']], 1)
    assert np.abs(tgl[:, 2] - out['origin'][2] - HTS).max() < 1e-6
    return dict(out=out, pix=pix, path=path, tg=tg, hp=_plane_height(out),
                bp=lambda pts: fastsar.backproject(got['S'], got['ant'] - out['center'], got['fmin'], got['df'],
                                                   pts - out['center'], backend='cpu', window=False))


@pytest.fixture(scope='module', params=[(20.0, False), (-25.0, True)], ids=['right', 'left'])
def scene(request, tmp_path_factory):
    return _scene(*request.param, tmp_path_factory.mktemp('terrain'))


def _measure(img, out, p):
    return quality.point_target(img, (int(round(p[0])), int(round(p[1]))), out['spx'], out['spy'], half=16)


def test_defocus_is_the_models(scene):
    """Uncorrected, a target off the plane is smeared as the model says: the phase of its azimuth spectrum is the
    model's to hundredths of a radian, and at 30 m the on-plane target given the model's phase has the shape of
    the target's own azimuth cut, its widened main lobe and its sidelobes."""
    out, pix = scene['out'], scene['pix']
    img = out['image']
    nx, half = img.shape[0], 96
    f = np.fft.fftfreq(2 * half)
    # azimuth cuts summed over the columns of a response: its part at the center frequency, whatever its range walk
    cut = lambda k: np.roll(img[int(round(pix[k, 0])) - half:int(round(pix[k, 0])) + half,
                                int(round(pix[k, 1])) - 8:int(round(pix[k, 1])) + 9].sum(1), -half)
    c, band, res = terrain.coefficients(out, pix[:, 0], pix[:, 1], HTS, order=6)
    assert res < 0.12 and np.abs(c[3]).max() < 1e-6                   # the plane's own target: no phase
    v = (f - band[0]) / band[1]
    inb = np.abs(v) < 0.9
    vs, order = np.sort(v[inb]), np.argsort(v[inb])
    on = cut(3)
    width = {}
    for k in (1, 2, 4, 5):                                           # 30 and 100 m: phases of 13 and 45 rad
        got = np.fft.fft(cut(k))[inb][order] * np.conj(np.fft.fft(on)[inb][order])
        d = np.unwrap(np.angle(got * np.exp(-1j * (vs[:, None] ** np.arange(7)) @ c[k])))
        d -= np.polyval(np.polyfit(vs, d, 1), vs)                    # the cuts' own subpixel positions
        assert d.std() < (0.03 if abs(HTS[k]) < 50 else 0.08), (k, d.std())
        # the on-plane response with the model's phase, at the target's place in its pixel, against the target's own
        dx = pix[k, 0] - round(pix[k, 0]) - (pix[3, 0] - round(pix[3, 0]))
        pred = np.fft.ifft(np.fft.fft(on) * np.exp(1j * ((v[:, None] ** np.arange(7)) @ c[k] - 2 * np.pi * f * dx))
                           * (np.abs(v) <= 1.02))
        pred, act = (np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft(z), 16 * 2 * half))) for z in (pred, cut(k)))
        if abs(HTS[k]) < 50:
            assert np.abs(pred / pred.max() - act / act.max()).max() < 0.15, k

            def lobe(p):                                             # the highest lobe's width at -3 dB (m)
                m, lo = int(np.argmax(p)), p < p.max() / np.sqrt(2)
                return (np.argmax(lo[m:]) + np.argmax(lo[m::-1])) / 16 * out['spx']
            assert lobe(pred) == pytest.approx(lobe(act), abs=0.03), (k, lobe(pred), lobe(act))   # 1.5 samples
        width[k] = _measure(img, out, pix[k])['res_az']
    ref = _measure(img, out, pix[3])
    assert -14.5 < ref['pslr_az'] < -12.3
    assert all(w > 1.7 * ref['res_az'] for w in width.values()) and width[1] > 1.2 * width[2]    # worse with height
    assert all(_measure(img, out, pix[k])['pslr_az'] > -4 for k in (0, 1, 2, 4, 5, 6))


def test_correction_restores_the_targets(scene):
    """With the correction for its height each target has the on-plane response where its range and Doppler at
    the COA put it: within 1% in width and 0.3 dB in peak sidelobe at 30 m, where the phase reaches 12 rad at the
    band edge. The correction is the phase at the center frequency: over the band the phase varies by B / (2 fc)
    of itself (a range walk of the response across the aperture), which leaves 0.3 rad there, 0.9 rad at 100 m
    (40 rad: widths within 10%) and 2.7 rad at 300 m (120 rad, a walk of a range cell: widths within 1.6, from
    4.1 times the on-plane width uncorrected)."""
    out, pix, hp = scene['out'], scene['pix'], scene['hp']
    img = out['image']
    ref = _measure(img, out, pix[3])
    lim = {30.0: (1.01, 1.01, 0.3, 0.05, 0.97), 100.0: (1.1, 1.06, 1.0, 0.1, 0.85), 300.0: (1.6, 1.25, 1.5, 0.1, 0.6)}
    for k in range(7):
        if k == 3:
            continue
        wid, rg, pslr, pos, peak = lim[abs(HTS[k])]
        o = dict(out, image=img.copy(), notes=[])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', terrain.TerrainWarning)
            info = terrain.focus(o, terrain.as_dem(hp + HTS[k]))
        assert info['applied'] and info['fit_rad'] < 0.011 and info['height'][1] == pytest.approx(HTS[k], abs=0.01)
        assert 6 <= info['order'] <= 8 and np.abs(info['edge_rad']).max() > 10
        assert any('terrain: focused' in n for n in o['notes'])
        m = _measure(o['image'], out, pix[k])
        assert abs(m['i'] - pix[k, 0]) < pos and abs(m['j'] - pix[k, 1]) < pos, (k, m['i'], m['j'])
        assert 0.99 < m['res_az'] / ref['res_az'] < wid, (k, m['res_az'])
        assert 0.99 < m['res_rg'] / ref['res_rg'] < rg, (k, m['res_rg'])
        assert abs(m['pslr_az'] - ref['pslr_az']) < pslr and m['peak'] > peak * ref['peak'], (k, m['pslr_az'])
        before = _measure(img, out, pix[k])
        assert m['res_az'] < 0.6 * before['res_az'] and m['peak'] > 1.9 * before['peak']


def test_against_exact_backprojection_at_the_true_height(scene):
    """The corrected chip of a target against exact backprojection onto the surface at its height, at the points
    the chip's pixels show (surface_points), as complex images: before the correction they differ by about the
    peak itself, after it by -19 to -26 dB of the peak at 30 m and -11 to -14 dB at 100 m (the range walk)."""
    out, pix, hp, bp = scene['out'], scene['pix'], scene['hp'], scene['bp']
    img = out['image']
    fr, kc = planes._frame(out), 4 * np.pi * 0.5 * (out['band'][0] + out['band'][1]) / C
    for k, lim in ((2, -17.0), (4, -17.0), (1, -10.0), (5, -10.0)):
        dem = terrain.as_dem(hp + HTS[k])
        i0, j0 = int(round(pix[k, 0])), int(round(pix[k, 1]))
        I, J = np.meshgrid(np.arange(i0 - 12, i0 + 13), np.arange(j0 - 12, j0 + 13), indexing='ij')
        pts = terrain.surface_points(out, dem, I, J)
        assert np.abs(pts[..., 2] - out['origin'][2] - HTS[k]).max() < 0.02
        y = out['origin'] + I[..., None] * out['spx'] * out['e1'] + J[..., None] * out['spy'] * out['e2']
        Ry = np.linalg.norm(y - fr['arp'], axis=-1)
        assert np.abs(np.linalg.norm(pts - fr['arp'], axis=-1) - Ry).max() < 1e-6          # the pixels' own ranges
        exact = bp(pts) * np.exp(-1j * kc * (Ry - fr['r0']))
        o = dict(out, image=img.copy(), notes=[])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', terrain.TerrainWarning)
            terrain.focus(o, dem)
        db = lambda a: 20 * np.log10(np.abs(a[i0 - 12:i0 + 13, j0 - 12:j0 + 13] - exact).max() / np.abs(exact).max())
        assert db(img) > -6 and db(o['image']) < lim, (k, db(img), db(o['image']))


def test_exact_route(scene):
    """The documented route to an image on the surface itself: read_cphd's history backprojected at
    surface_points. The target 100 m above the plane, smeared over 40 pixels in the plane's image, has the
    on-plane response at its own pixel."""
    out, pix, hp = scene['out'], scene['pix'], scene['hp']
    col, meta = io.read_cphd(scene['path'], meta=True)
    ref = _measure(out['image'], out, pix[3])
    k = 5
    i0, j0 = int(round(pix[k, 0])), int(round(pix[k, 1]))
    I, J = np.meshgrid(np.arange(i0 - 20, i0 + 20), np.arange(j0 - 20, j0 + 20), indexing='ij')
    pts = terrain.surface_points(out, terrain.as_dem(hp + HTS[k]), I, J)
    chip = fastsar.backproject(col['S'], col['ant'], col['fmin'], col['df'], pts, backend='cpu', window=False)
    m = quality.point_target(chip, (20, 20), out['spx'], out['spy'], half=16)
    assert abs(m['i'] - (pix[k, 0] - i0 + 20)) < 0.05 and abs(m['j'] - (pix[k, 1] - j0 + 20)) < 0.05
    assert m['res_az'] == pytest.approx(ref['res_az'], rel=0.02)
    assert m['res_rg'] == pytest.approx(ref['res_rg'], rel=0.02)
    assert abs(m['pslr_az'] - ref['pslr_az']) < 0.3 and m['peak'] == pytest.approx(ref['peak'], rel=0.02)


def test_flat_dem_changes_nothing(scene):
    out = scene['out']
    img = out['image']
    for dem in (scene['hp'], lambda lat, lon: np.full(np.shape(lat), scene['hp'])):
        o = dict(out, notes=[])
        info = terrain.focus(o, terrain.as_dem(dem))
        assert not info['applied'] and o['image'] is img and o['terrain'] is info and not o['notes']
        assert abs(info['offset']) < 1e-2 and info['layover'] == 0 and info['outside'] == 0


# ------------------------------------------------------------------------------------------------ the layover map
def _ramp(out, slope, lim=400.0, axis='e2'):
    """A DEM that rises with ground range (or, axis='e1', across the line of sight) through the scene center:
    (dem, h(t), center), t the distance along the axis (m)."""
    from fastsar.io import geodetic_to_ecf, ecf_to_local
    nx, ny = out['image'].shape
    c = out['origin'] + (nx - 1) / 2 * out['spx'] * out['e1'] + (ny - 1) / 2 * out['spy'] * out['e2']
    hp = float(io.ecf_to_geodetic(io.local_to_ecf(c, out['meta']))[2])
    h = lambda t: np.clip(slope * t, -lim, lim)

    def dem(lat, lon):
        p = ecf_to_local(geodetic_to_ecf(lat, lon, np.full(np.shape(lat), hp)), out['meta'])
        return hp + h((p - c) @ out[axis])
    return dem, h, c


def test_heights_through_layover():
    """heights finds, for each pixel, the surface point with its range and Doppler: on a slope facing the radar,
    where that point lies up to 300 m of ground from the pixel's own, the height is the one a bisection along the
    pixel's circle finds; surface_points lie on the DEM and project back onto their pixels."""
    out = _product(_collect(20.0, False))
    nx, ny = out['image'].shape
    dem, h, c = _ramp(out, 1.0, 330.0)
    hm = terrain.heights(out, dem, post=4.0)
    assert hm['layover'] == 0 and hm['outside'] == 0 and hm['h'].shape == (len(hm['i']), len(hm['j']))
    assert np.nanmax(hm['h']) > 200 and np.nanmin(hm['h']) < -200 and np.abs(hm['plane']).max() < 90
    fr = planes._frame(out)
    I, J = np.meshgrid(hm['i'][::7], hm['j'][::5], indexing='ij')
    y = out['origin'] + I[..., None] * out['spx'] * out['e1'] + J[..., None] * out['spy'] * out['e2']
    a, r = planes.ground_to_slant(fr, y)
    g = lambda z: h((planes.slant_to_ground(fr, a, r, z) - c) @ out['e2']) - z      # the surface over the circle
    lo, hi = np.full(I.shape, -400.0), np.full(I.shape, 400.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        up = g(mid) > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    assert np.abs(hm['h'][::7, ::5] - lo).max() < 0.3
    pts = terrain.surface_points(out, hm, I, J)
    assert np.abs(pts[..., 2] - h((pts - c) @ out['e2'])).max() < 0.3
    a2, r2 = planes.ground_to_slant(fr, pts)
    assert np.abs(a2 - a).max() < 1e-6 and np.abs(r2 - r).max() < 1e-6
    # across the line of sight nothing is displaced at broadside: a surface that varies along e1 alone is seen
    # where it is, however steep
    side = _product(_collect(0.0, False, curved=False))
    hm2 = terrain.heights(side, _ramp(side, 3.0, axis='e1')[0], post=5.0)
    assert hm2['layover'] == 0 and np.ptp(hm2['h']) > 400 and np.abs(hm2['h'] - hm2['plane']).max() < 0.5


def test_layover_and_missing_dem():
    """A wall steeper than the line of sight is seen with the ground before and behind it: heights takes the mean
    of the crossings, weighted by the ground each covers, and counts them; outside the DEM there is no height,
    and focus says so."""
    out = _product(_collect(20.0, False))
    dem, h, c = _ramp(out, 4.0, 40.0)                                # 80 m in 20 m of ground range
    hm = terrain.heights(out, dem, post=3.0)
    assert 0.1 < hm['layover'] < 0.8 and hm['outside'] == 0
    assert np.nanmin(hm['h']) == pytest.approx(-40.0, abs=0.5) and np.nanmax(hm['h']) == pytest.approx(40.0, abs=0.5)
    row = hm['h'][len(hm['i']) // 2]
    mid = np.abs(row) < 39                                           # the foot, the wall and the top in one pixel
    assert 0.1 < mid.mean() < 0.3 and np.abs(row[mid]).max() < 15 and np.all(np.abs(row[~mid]) > 39.9)
    hole = lambda lat, lon: np.where(lon > LON, np.nan, dem(lat, lon))
    hm = terrain.heights(out, hole, post=6.0)
    assert 0.1 < hm['outside'] < 0.9 and np.isnan(hm['h']).mean() == pytest.approx(hm['outside'])
    rng = np.random.default_rng(1)
    img = (rng.normal(size=(480, 440)) + 1j * rng.normal(size=(480, 440))).astype(np.complex64)
    o = _product(_collect(20.0, False), image=img)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', terrain.TerrainWarning)
        info = terrain.focus(o, hole)
    assert info['applied'] and any('outside the DEM' in n for n in o['notes'])
    with pytest.raises(ValueError, match='does not cover'):
        terrain.heights(out, lambda lat, lon: np.full(np.shape(lat), np.nan))


# ------------------------------------------------------------------------------------------------ the DEM
def test_dem_conventions(tmp_path, need):
    need('rasterio')
    lat, lon = LAT + 0.01 * np.arange(-2, 3)[:, None], LON + 0.01 * np.arange(-2, 3)[None, :]
    surf = lambda la, lo: 300.0 + 2000.0 * (np.asarray(la) - LAT) + 1000.0 * (np.asarray(lo) - LON)
    # two tiles of heights above a geoid 30 m below the ellipsoid there, west and east of LON + 0.005
    for name, x0 in (('w', LON - 0.045), ('e', LON + 0.005)):
        xs, ys = x0 + 0.001 * (np.arange(50) + 0.5), LAT + 0.045 - 0.001 * (np.arange(90) + 0.5)
        products.write_geotiff(str(tmp_path / f'{name}.tif'), surf(ys[:, None], xs[None, :]) + 30.0,
                               (0.001, 0.0, x0, 0.0, -0.001, LAT + 0.045))
    with pytest.raises(ValueError, match='geoid='):                  # a file's heights need their datum stated
        terrain.as_dem(str(tmp_path / 'w.tif'))
    west = terrain.as_dem(str(tmp_path / 'w.tif'), geoid=-30.0)
    assert np.abs(west(lat[:, :3], lon[:, :3]) - surf(lat[:, :3], lon[:, :3])).max() < 1e-6
    assert np.isnan(west(LAT, LON + 0.02))
    both = terrain.as_dem([tmp_path / 'w.tif', tmp_path / 'e.tif'], geoid=-30.0)
    assert np.abs(both(lat, lon) - surf(lat, lon)).max() < 1e-6
    assert terrain.as_dem(120.0)(lat, lon).shape == (5, 5)
    assert np.all(terrain.as_dem(120, geoid=5.0)(lat, lon) == 125.0) and terrain.as_dem(surf) is surf
    assert np.all(terrain.as_dem(surf, geoid=-30.0)(lat, lon) == surf(lat, lon) - 30.0)
    for bad in ('nan', None, float('inf')):
        with pytest.raises(ValueError):
            terrain.as_dem(bad if bad != 'nan' else float('nan'))


def test_warning_for_a_dem_off_the_scene_height():
    """A DEM whose median over the image is further from the image plane than its relief explains (here the
    geoid's 30 m, forgotten on flat ground) warns and leaves a note; with the offset, or with relief of that
    size, it does not."""
    rng = np.random.default_rng(2)
    img = (rng.normal(size=(480, 440)) + 1j * rng.normal(size=(480, 440))).astype(np.complex64)
    base = _product(_collect(20.0, False), image=img)
    hp = _plane_height(base)
    flat = lambda lat, lon: hp + 30.0 + 2000.0 * (np.asarray(lat) - LAT)          # 2 m of relief
    o = dict(base, notes=[])
    with pytest.warns(terrain.TerrainWarning, match=r'\+30 m from the image plane'):
        info = terrain.focus(o, flat)
    assert info['offset'] == pytest.approx(30.0, abs=0.5) and info['relief'] < 2
    assert any('geoid' in n for n in o['notes'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', terrain.TerrainWarning)
        o = dict(base, notes=[])
        assert abs(terrain.focus(o, terrain.as_dem(flat, geoid=-30.0))['offset']) < 0.5
        assert not any('geoid' in n for n in o['notes'])
        hills = lambda lat, lon: hp + 30.0 + 2e5 * (np.asarray(lat) - LAT)        # 60 m of relief
        assert terrain.focus(dict(base, notes=[]), hills)['relief'] > 20
        terrain.focus(dict(base, notes=[]), flat, check=False)


def test_what_focus_takes():
    out = _product(_collect(20.0, False))
    for change in (dict(plane='slant'), dict(mode='moving'), dict(axes='track')):
        with pytest.raises(ValueError, match='line of sight'):
            terrain.focus(dict(out, **change), terrain.as_dem(H0))
    with pytest.raises(ValueError, match='order'):
        terrain.focus(dict(out), terrain.as_dem(H0), order=9)


# ------------------------------------------------------------------------------------------------ devices
DEVICES = [pytest.param('cupy', marks=pytest.mark.cuda), 'jax']


@pytest.mark.parametrize('lib', DEVICES)
def test_on_device(lib, need, request):
    """focus on a CuPy and on a JAX image gives the host's image, with its zeros, in the same place."""
    from fastsar import _xp
    if lib == 'jax':
        request.getfixturevalue('jax_cpu')
    xp = need('cupy') if lib == 'cupy' else need('jax.numpy')
    rng = np.random.default_rng(5)
    img = (rng.normal(size=(480, 440)) + 1j * rng.normal(size=(480, 440))).astype(np.complex64)
    img[:60, :200] = 0
    img[300:, 400:] = 0
    base = _product(_collect(20.0, False), image=img)
    dem, h, c = _ramp(base, 0.3)
    ref, dev = dict(base, notes=[]), dict(base, image=xp.asarray(img.copy()), notes=[])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', terrain.TerrainWarning)
        a, b = terrain.focus(ref, dem, block=64), terrain._stage(dev, dem)
        blk = dict(base, image=xp.asarray(img.copy()), notes=[])
        terrain.focus(blk, dem, block=64)
    assert a['applied'] and b['applied'] and a['powers'] == b['powers'] and _xp.kind(dev['image']) == lib
    assert np.abs(ref['image'] - img).max() > 0.5                    # a correction of tens of radians
    for got in (blk['image'], ):
        got = _xp.host(got)
        assert got.dtype == np.complex64 and np.array_equal(got == 0, img == 0)
        assert np.abs(got - ref['image']).max() < 2e-4 * np.abs(ref['image']).max()
    assert np.array_equal(_xp.host(dev['image']) == 0, img == 0)


@pytest.mark.cuda
def test_stage_without_device_memory(need, monkeypatch):
    """With no room on the GPU for a second image the stage runs on the host and returns the image to the device."""
    cp = need('cupy')
    rng = np.random.default_rng(6)
    img = (rng.normal(size=(480, 440)) + 1j * rng.normal(size=(480, 440))).astype(np.complex64)
    base = _product(_collect(20.0, False), image=img)
    dem, h, c = _ramp(base, 0.3)
    ref, dev = dict(base, image=cp.asarray(img), notes=[]), dict(base, image=cp.asarray(img), notes=[])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', terrain.TerrainWarning)
        terrain._stage(ref, dem)
        monkeypatch.setenv('FASTSAR_CUDA_BLOCK_MB', '1')
        info = terrain._stage(dev, dem)
    assert info['applied'] and dev['terrain'] is info and isinstance(dev['image'], cp.ndarray)
    assert any('ran on the host' in n for n in dev['notes']) and not any('ran on the host' in n for n in ref['notes'])
    assert float(abs(dev['image'] - ref['image']).max()) < 1e-5 * float(abs(ref['image']).max())


# ------------------------------------------------------------------------------------------------ process_cphd
def _sarkit_errors(sm):
    if sys.version_info < (3, 11):
        return None
    try:
        import lxml.etree
        import sarkit.verification as skv
    except ImportError:
        return None
    con = skv.SicdConsistency(lxml.etree.parse(_io.BytesIO(sm.to_xml_bytes())))
    con.check()
    fail = con.failures(omit_passed_sub=True)
    return sorted(k for k, v in fail.items() if any(d['severity'] == 'Error' for d in v['details']))


def test_process_cphd_with_a_dem(scene, tmp_path):
    """process_cphd(dem=): the stage runs before the autofocus, is recorded in the product, its notes and its SICD
    (which stays a plane's, and consistent), and a target at the DEM's height is focused where the SICD's
    projection of its true position puts it; a DEM at the plane's height changes nothing, to the last bit."""
    path, hp = scene['path'], scene['hp']
    kw = dict(plane='ground', weighting='uniform', backend='cpu')
    base = fastsar.process_cphd(path, **kw)
    assert 'terrain' not in base and 'terrain' not in base['timing']
    flat = fastsar.process_cphd(path, dem=hp, **kw)
    assert not flat['terrain']['applied'] and np.array_equal(flat['image'], base['image'])
    assert 'terrain' in flat['timing']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', terrain.TerrainWarning)
        o = fastsar.process_cphd(path, str(tmp_path / 't.nitf'), dem=lambda lat, lon: np.full(np.shape(lat), hp),
                                 geoid=30.0, **kw)
    t = o['terrain']
    assert t['applied'] and t['height'][1] == pytest.approx(30.0, abs=0.01) and 2 in t['powers'] and t['block'] >= 32
    assert o['timing']['terrain'] > 0 and any(n.startswith('terrain: focused with the DEM') for n in o['notes'])
    sm = o['metadata']
    assert sm.Grid.Type == 'PLANE' and sm.Grid.ImagePlane == 'GROUND' and sm.ImageFormation.ImageFormAlgo == 'OTHER'
    proc = sm.ImageFormation.Processing[0]
    assert proc.Type == 'TERRAIN PHASE CORRECTION' and proc.Applied is True
    assert base.get('metadata') is None and products.sicd_meta(base)[0].ImageFormation.Processing is None
    assert _sarkit_errors(sm) in (None, [])
    # the target 30 m above the plane: focused, and where the SICD's projection of its position puts it
    k = 4
    rc = sm.project_ground_to_image(sim.to_ecf(scene['tg'][k], LAT, LON, H0, HEADING)[None])[0][0]
    pixels = products.sicd_meta(o)[1]
    m1 = quality.point_target(pixels, (int(round(rc[0])), int(round(rc[1]))), sm.Grid.Row.SS, sm.Grid.Col.SS, half=16)
    m0 = quality.point_target(products.sicd_meta(base)[1], (int(round(rc[0])), int(round(rc[1]))), sm.Grid.Row.SS,
                              sm.Grid.Col.SS, half=16)
    assert abs(m1['i'] - rc[0]) < 0.1 and abs(m1['j'] - rc[1]) < 0.1, (m1['i'] - rc[0], m1['j'] - rc[1])
    # the SICD's rows run along range: axis 1 of its array is azimuth
    assert m1['pslr_rg'] < -12 and m1['pslr_az'] < -12 and m0['pslr_rg'] > -5 and m1['peak'] > 1.5 * m0['peak']
    # the slant-plane product carries it too, and a moving beam or a file without its datum is refused at once
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', terrain.TerrainWarning)
        s = fastsar.process_cphd(path, dem=hp + 30.0, backend='cpu')
    stages = list(s['timing'])
    assert s['plane'] == 'slant' and s['terrain']['applied'] and stages.index('terrain') < stages.index('autofocus')
    with pytest.raises(ValueError, match='geoid='):
        fastsar.process_cphd(path, dem='dem.tif', backend='cpu')
    with pytest.raises(ValueError, match='spotlight'):
        fastsar.process_cphd(path, dem=hp, mode='moving', backend='cpu')
