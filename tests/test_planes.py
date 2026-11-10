"""fastsar.planes: the slant plane's maps, the baseband image, and resampling between the planes, on a simulated
spotlight collection with squint and a curved track, for both look sides."""
import dataclasses

import numpy as np
import pytest

import fastsar
from fastsar import io, planes, quality, sim

C = 299792458.0
LAT, LON, H0, HEADING = 40.79, -77.86, 351.0, 30.0
TG = np.array([[0, 0, 0], [50, 35, 0], [-55, 45, 0], [40, -60, 0], [-45, -50, 0], [65, 0, 0], [0, -70, 0.]])
AMP = np.exp(1j * np.arange(len(TG)))


def test_maps():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 501)
    for sign in (1, -1):                                             # right and left looking
        ant = np.stack([-4e5 + 30 * t ** 2 - 0.5 * t ** 3, sign * (-3e4 + 7500 * t - 8 * t ** 2),
                        5e5 - 20 * t - 3 * t ** 2], 1)
        scp = np.array([120.0, -340.0, 15.0])
        fr = planes.slant_frame(dict(tx=ant, rcv=ant, tx_time=t), scp)
        assert np.allclose(np.cross(fr['e1'], fr['e2']), fr['normal']) and fr['normal'][2] > 0
        assert abs(fr['vel'] @ fr['normal']) < 1e-9 * np.linalg.norm(fr['vel'])
        assert np.sign(fr['v1']) == -sign                            # azimuth runs against the velocity when right looking
        assert np.allclose(planes.ground_to_slant(fr, scp), 0, atol=1e-9)
        g = np.c_[rng.uniform(-6e3, 6e3, (500, 2)) + scp[:2], rng.uniform(-50, 80, 500)]
        a, r = planes.ground_to_slant(fr, g)
        assert np.abs(planes.slant_to_ground(fr, a, r, g[:, 2]) - g).max() < 1e-6
        p = scp + a[:, None] * fr['e1'] + r[:, None] * fr['e2']       # same range and range rate at the COA
        R1, R2 = np.linalg.norm(p - fr['arp'], axis=1), np.linalg.norm(g - fr['arp'], axis=1)
        assert np.abs(R1 - R2).max() < 1e-6
        assert np.abs((p - fr['arp']) @ fr['vel'] / R1 - (g - fr['arp']) @ fr['vel'] / R2).max() < 1e-9


def _collect(squint, left):
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    y = col.ant[:, 1] / np.ptp(col.ant[:, 1])
    ant = col.ant.copy()
    ant[:, 1] -= 10e3 * np.cos(np.radians(35)) * np.tan(np.radians(squint))
    ant[:, 0] += 160.0 * y ** 2 + 120.0 * y ** 3                      # a curved track
    ant[:, 2] += -100.0 * y ** 2 + 6.0 * y
    return dataclasses.replace(col, ant=ant[::-1].copy() if left else ant)


@pytest.fixture(scope='module', params=[(20.0, False), (-25.0, True)], ids=['right', 'left'])
def scene(request, tmp_path_factory):
    squint, left = request.param
    col = _collect(squint, left)
    path = str(tmp_path_factory.mktemp('planes') / 'c.cphd')
    sim.write_cphd(path, col, sim.simulate(col, TG, AMP), LAT, LON, H0, heading=HEADING, scene=160.0)
    got = {}

    def plan(g):
        fr = planes.slant_frame(g['meta'], g['center'])
        t = np.linspace(-85, 85, 5)
        pts = g['center'] + (t[:, None, None] * g['e1'] + t[None, :, None] * g['e2']).reshape(-1, 3)
        got['b'] = planes.sampling(fr, g['ant'], (g['fmin'], g['fmin'] + (g['S'].shape[1] - 1) * g['df']), pts,
                                   g['e1'], g['e2'])
        return dict(spacing=(0.96 / got['b']['bx'], 0.96 / got['b']['by']), bounds=(-85, 85, -85, 85))
    out = fastsar.form_cphd(path, axes='los', window=False, backend='cpu', _plan=plan)
    tg = io.ecf_to_local(sim.to_ecf(TG, LAT, LON, H0, HEADING), out['meta'])
    return dict(out=out, b=got['b'], tg=tg, left=left, fr=planes._frame(out))


def test_los_axes(scene):
    out, fr = scene['out'], scene['fr']
    assert out['axes'] == 'los' and out['plane'] == 'ground'
    e1, e2, up = out['e1'], out['e2'], np.array([0.0, 0.0, 1.0])
    los = fr['scp'] - fr['arp']
    assert abs(e1[2]) < 1e-12 and abs(e2[2]) < 1e-12 and np.allclose(np.cross(e1, e2), up)
    assert e2 @ los > 0 and abs(e1 @ los) < 1e-3 * np.linalg.norm(los)     # e2: the line of sight on the ground
    nx, ny = out['image'].shape
    assert abs(nx * out['spx'] - 170) < 2 * out['spx'] and abs(ny * out['spy'] - 170) < 2 * out['spy']
    assert np.abs(out['center'] - fr['scp']).max() < max(out['spx'], out['spy'])


def test_deramp(scene):
    out = scene['out']
    raw = out['image'].copy()
    planes.deramp(out)
    assert out['baseband'] and np.array_equal(planes.deramp(out)['image'], out['image'])     # once only
    # along e1 the baseband image lies in the band sampling gives; on its carrier it does not
    w = np.hanning(raw.shape[0])[:, None] * np.hanning(raw.shape[1])[None]
    k = np.abs(np.fft.fftfreq(raw.shape[0])) > 0.5 * scene['b']['bx'] * out['spx'] + 0.01
    share = lambda im: (lambda p: p[k].sum() / p.sum())((np.abs(np.fft.fft2(im * w)) ** 2).sum(1))
    assert share(out['image']) < 2e-5 < 2e-4 < share(raw), (share(out['image']), share(raw))
    planes.deramp(out, sign=+1)
    assert not out['baseband'] and np.abs(out['image'] - raw).max() < 2e-5 * np.abs(raw).max()
    planes.deramp(out)


def _measure(img, ie, je, si, sj, half=16):
    m = quality.point_target(img, (int(round(ie)), int(round(je))), si, sj, half=half)
    ii, jj = int(round(ie)), int(round(je))
    U = quality.upsample(img[ii - half:ii + half, jj - half:jj + half], 16)
    return m, U.flat[np.argmax(np.abs(U))]


def test_to_slant(scene):
    out, b, tg, fr = scene['out'], scene['b'], scene['tg'], scene['fr']
    sl = planes.to_slant(out, (0.8 / b['ba'], 0.8 / b['br']), margin=24)
    assert sl['plane'] == 'slant' and sl['baseband'] and sl['image'].dtype == np.complex64
    i0, j0 = sl['scp_pixel']
    assert np.allclose(sl['origin'] + i0 * sl['spx'] * sl['e1'] + j0 * sl['spy'] * sl['e2'], sl['scp'])
    assert np.allclose(sl['e2'], fr['e2']) and np.allclose(np.cross(sl['e1'], sl['e2']), fr['normal'])
    fc = 0.5 * (out['band'][0] + out['band'][1])
    a, r = planes.ground_to_slant(fr, tg)
    res = []
    for k in range(len(tg)):
        ie, je = a[k] / sl['spx'] + i0, r[k] / sl['spy'] + j0
        m, peak = _measure(sl['image'], ie, je, sl['spx'], sl['spy'])
        assert abs(m['i'] - ie) < 0.03 and abs(m['j'] - je) < 0.03, (k, m['i'] - ie, m['j'] - je)
        # SICD's phase: -4 pi fc / c times the target's range less the scene center's, at the COA
        ph = np.angle(peak * np.conj(AMP[k]) * np.exp(4j * np.pi * fc / C * (np.linalg.norm(tg[k] - fr['arp']) - fr['r0'])))
        assert abs(ph) < 0.01, (k, ph)
        assert -14.0 < m['pslr_az'] < -12.0 and -14.0 < m['pslr_rg'] < -12.6, (k, m['pslr_az'], m['pslr_rg'])
        res.append((m['res_az'], m['res_rg']))
    res = np.array(res)
    assert np.ptp(res, axis=0).max() < 0.02 * res.mean()              # one impulse response across the image
    assert abs(res[:, 1].mean() - 0.886 * C / (2 * (out['band'][1] - out['band'][0]))) < 0.01
    # outside the ground image less its margin: exactly zero
    ii, jj = np.meshgrid(np.arange(sl['image'].shape[0]), np.arange(sl['image'].shape[1]), indexing='ij')
    g = planes.slant_to_ground(fr, (ii - i0) * sl['spx'], (jj - j0) * sl['spy'], out['origin'][2]) - out['origin']
    x, y = g @ out['e1'] / out['spx'], g @ out['e2'] / out['spy']
    nx, ny = out['image'].shape
    outside = (x < 23) | (x > nx - 24) | (y < 23) | (y > ny - 24)
    assert outside.any() and not sl['image'][outside].any()
    inside = (x > 26) & (x < nx - 27) & (y > 26) & (y < ny - 27)
    assert np.count_nonzero(sl['image'][inside]) > 0.999 * inside.sum()


def test_valid_mask(scene):
    out, b = scene['out'], scene['b']
    nx, ny = out['image'].shape
    valid = np.zeros((nx, ny), bool)
    valid[: nx // 2] = True
    sl = planes.to_slant(out, (0.8 / b['ba'], 0.8 / b['br']), valid=valid)
    i0, j0 = sl['scp_pixel']
    ii, jj = np.meshgrid(np.arange(sl['image'].shape[0]), np.arange(sl['image'].shape[1]), indexing='ij')
    g = planes.slant_to_ground(scene['fr'], (ii - i0) * sl['spx'], (jj - j0) * sl['spy'], out['origin'][2]) - out['origin']
    x = g @ out['e1'] / out['spx']
    y = g @ out['e2'] / out['spy']
    assert not sl['image'][x > nx // 2 + 1].any()
    assert sl['image'][(x > 30) & (x < nx // 2 - 2) & (y > 30) & (y < out['image'].shape[1] - 31)].all()


def test_bounds(scene):
    out, b = scene['out'], scene['b']
    sl = planes.to_slant(out, (0.8 / b['ba'], 0.8 / b['br']), bounds=(-30.0, 40.0, -20.0, 25.0))
    i0, j0 = sl['scp_pixel']
    na, nr = sl['image'].shape
    assert abs(i0 * sl['spx'] - 30) < sl['spx'] and abs((na - 1 - i0) * sl['spx'] - 40) < sl['spx']
    assert abs(j0 * sl['spy'] - 20) < sl['spy'] and abs((nr - 1 - j0) * sl['spy'] - 25) < sl['spy']
    with pytest.raises(ValueError):
        planes.to_slant(sl, (0.3, 0.3))


def test_to_ground(scene):
    out, b, tg = scene['out'], scene['b'], scene['tg']
    sl = planes.to_slant(out, (0.8 / b['ba'], 0.8 / b['br']))
    g = sl['ground']
    back = planes.to_ground(sl, g['origin'], g['e1'], g['e2'], (g['spx'], g['spy']), g['shape'])
    assert back['plane'] == 'ground' and back['image'].shape == out['image'].shape
    d = (back['image'] - out['image'])[40:-40, 40:-40]
    assert 20 * np.log10(np.abs(d).max() / np.abs(out['image']).max()) < -45
    # a detected image (power), from a slant image sampled for its doubled band
    s2 = planes.to_slant(out, (0.48 / b['ba'], 0.48 / b['br']))
    det = dict(s2, image=np.abs(s2['image']) ** 2)
    px = 0.15
    n = (int(g['shape'][0] * g['spx'] / px), int(g['shape'][1] * g['spy'] / px))
    for kernel, tol in (('cubic', 0.25), ('sinc', 0.05)):
        gd = planes.to_ground(det, g['origin'], g['e1'], g['e2'], px, n, kernel=kernel)
        assert gd['image'].dtype == np.float32 and gd['image'].shape == n
        for k in range(len(tg)):
            ie, je = (tg[k] - g['origin']) @ g['e1'] / px, (tg[k] - g['origin']) @ g['e2'] / px
            m = quality.point_target(gd['image'], (int(round(ie)), int(round(je))), px, px, half=16)
            assert abs(m['i'] - ie) < tol and abs(m['j'] - je) < tol, (kernel, k, m['i'] - ie, m['j'] - je)
    with pytest.raises(ValueError):
        planes.to_ground(out, g['origin'], g['e1'], g['e2'], px, n)


DEVICES = [pytest.param('cupy', marks=pytest.mark.cuda), 'jax']


def _close(got, ref, lib, tol, rms, flips=0):
    """A device result against the CPU's: the same kind of array as the input, the same zeros (but `flips` pixels at
    the edge of a mask), and differences under tol (largest) and rms of the largest sample."""
    from fastsar import _xp
    assert _xp.kind(got) == lib
    got = _xp.host(got)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    same = (got == 0) == (ref == 0)
    assert np.count_nonzero(~same) <= flips
    d, s = np.abs(got - ref)[same], np.abs(ref).max()
    assert d.max() < tol * s and np.sqrt((d ** 2).mean()) < rms * s, (d.max() / s, np.sqrt((d ** 2).mean()) / s)


def _product(nx, ny, spx, spy, r0, turn, squint, wavelength=0.031):
    """A ground-plane product without its image: a track at range r0 and 35 degrees of grazing, squinted, and axes
    along the line of sight turned by `turn` (deg), the scene center at pixel (nx // 2, 2 ny // 5)."""
    t = np.linspace(-1.5, 1.5, 301)
    gr, sq = np.radians(35.0), np.radians(squint)
    scp = np.array([30.0, -20.0, 5.0])
    los = np.array([np.cos(gr) * np.sin(sq), np.cos(gr) * np.cos(sq), -np.sin(gr)])
    ant = scp - r0 * los + t[:, None] * np.array([7500.0, 300.0, -20.0])
    c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    e2 = np.array([c * los[0] - s * los[1], s * los[0] + c * los[1], 0.0]) / np.hypot(los[0], los[1])
    e1 = np.cross(e2, [0.0, 0.0, 1.0])
    fc = C / wavelength
    return dict(meta=dict(tx=ant, rcv=ant, tx_time=t), origin=scp - (nx // 2) * spx * e1 - (2 * ny // 5) * spy * e2,
                e1=e1, e2=e2, spx=spx, spy=spy, band=(fc - 1e8, fc + 1e8), plane='ground', scp=scp)


@pytest.mark.parametrize('lib', DEVICES)
def test_on_device(lib, need):
    """deramp, to_slant (margin and mask) and to_ground on a CuPy and on a JAX image give the CPU's results. CuPy
    computes positions and phases in double as the CPU does; JAX in float32 about exact integers, which moves a
    sample in a hundred to the next row of the interpolation table (5e-4 of the largest sample)."""
    xp, tol, rms, flips = (need('cupy'), 5e-6, 5e-6, 0) if lib == 'cupy' else (need('jax.numpy'), 2e-3, 2e-4, 2)
    nx, ny = 520, 610
    rng = np.random.default_rng(6)
    ref = _product(nx, ny, 0.5, 0.35, 10e3, 0.0, 20.0)
    ref['image'] = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))).astype(np.complex64)
    dev = dict(ref, image=xp.asarray(ref['image']))
    planes.deramp(ref), planes.deramp(dev)
    assert dev['baseband']
    _close(dev['image'], ref['image'], lib, 5e-6, 5e-6)
    valid = np.ones((nx, ny), bool)
    valid[: nx // 3, : ny // 2] = False
    sl, sd = (planes.to_slant(o, (0.45, 0.25), margin=24, valid=valid) for o in (ref, dev))
    assert 0.1 < (sl['image'] == 0).mean() < 0.9
    _close(sd['image'], sl['image'], lib, tol, rms, flips)
    sl, sd = (planes.to_slant(o, (0.45, 0.25)) for o in (ref, dev))
    _close(sd['image'], sl['image'], lib, tol, rms, flips)
    g = sl['ground']
    bk, bd = (planes.to_ground(o, g['origin'], g['e1'], g['e2'], (g['spx'], g['spy']), g['shape']) for o in (sl, sd))
    assert bk['image'].shape == (nx, ny)
    _close(bd['image'], bk['image'], lib, tol, rms)
    det, dd = dict(sl, image=np.abs(sl['image']) ** 2), dict(sd, image=abs(sd['image']) ** 2)
    for kernel in ('cubic', 'sinc'):
        bk, bd = (planes.to_ground(o, g['origin'], g['e1'], g['e2'], 0.3, (800, 700), kernel=kernel) for o in (det, dd))
        assert bk['image'].dtype == np.float32
        _close(bd['image'], bk['image'], lib, tol, rms)
    planes.deramp(ref, sign=+1), planes.deramp(dev, sign=+1)
    assert not dev['baseband']
    _close(dev['image'], ref['image'], lib, 5e-6, 5e-6)


@pytest.mark.parametrize('lib', DEVICES)
@pytest.mark.parametrize('turn, squint', [(0.0, 12.0), (33.0, -20.0)])
def test_deramp_phase_on_device(lib, need, turn, squint):
    """The phase deramp applies on a device against the CPU's in float64, over 12 km at 600 km and 31 mm: JAX works
    in float32 (a TPU has no fast float64), where the range itself holds to 0.03 m, 12 rad."""
    from fastsar import _xp
    xp = need('cupy') if lib == 'cupy' else need('jax.numpy')
    nx, ny, spx, spy = 1200, 1500, 10.0, 8.0
    assert nx * spx == ny * spy == 12e3
    out = _product(nx, ny, spx, spy, 600e3, turn, squint)
    ref = planes.deramp(dict(out, image=np.ones((nx, ny), np.complex128)))
    dev = planes.deramp(dict(out, image=xp.ones((nx, ny), xp.complex64)))
    err = np.angle(_xp.host(dev['image']) * np.conj(ref['image']))
    assert np.abs(err).max() < (1e-6 if lib == 'cupy' else 1e-3), np.abs(err).max()


@pytest.mark.parametrize('lib', DEVICES)
def test_blocks_on_device(lib, need, monkeypatch):
    """Device memory that holds a few lines at a time gives the same image: many blocks, the last one short (CuPy)
    or moved back to keep its size (JAX), on an image that is a view."""
    if lib == 'cupy':
        xp, tol, rms, flips = need('cupy'), 5e-6, 5e-6, 0
        from fastsar import resample_cuda
        monkeypatch.setattr(resample_cuda, '_free', lambda: 12 << 20)
    else:
        xp, tol, rms, flips = need('jax.numpy'), 2e-3, 2e-4, 2
        from fastsar import resample_jax
        monkeypatch.setattr(resample_jax, 'BLOCK', 1 << 16)
    nx, ny = 310, 270
    rng = np.random.default_rng(7)
    ref = _product(nx, ny, 0.5, 0.35, 10e3, 0.0, -15.0)
    ref['image'] = (rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))).astype(np.complex64)
    dev = dict(ref, image=xp.asarray(np.ascontiguousarray(ref['image'].T)).T)
    planes.deramp(ref), planes.deramp(dev)
    _close(dev['image'], ref['image'], lib, 5e-6, 5e-6)
    valid = np.ones((nx, ny), bool)
    valid[: nx // 3, : ny // 2] = False
    sl, sd = (planes.to_slant(o, (0.45, 0.25), margin=10, valid=valid) for o in (ref, dev))
    _close(sd['image'], sl['image'], lib, tol, rms, flips)
    g = sl['ground']
    bk, bd = (planes.to_ground(o, g['origin'], g['e1'], g['e2'], (g['spx'], g['spy']), g['shape']) for o in (sl, sd))
    _close(bd['image'], bk['image'], lib, tol, rms)
