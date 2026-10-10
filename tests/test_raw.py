"""Raw (level 0) data: fastsar.raw's range compression and frequency-domain conversion on a synthetic echo; the
zero-Doppler ground point; a NISAR L0B look-alike written by the simulator read by io.read_nisar and formed by
form_cphd, with the targets where the truth puts them; the collection dispatcher."""
import os

import numpy as np
import pytest

import fastsar
from fastsar import sim, io, raw, products

pytestmark = pytest.mark.needs('h5py')
C = 299792458.0
lat0, lon0, h0, heading = 40.7934, -77.8600, 351.0, 30.0


def test_fx_history_delay():
    """A single echo at a known delay: the frequency-domain history has the phase exp(-j 2 pi f (tau - tau_ref))
    over the chirp's band after range compression."""
    fs, bw, fc, T = 120e6, 100e6, 9.6e9, 5e-6
    L = int(T * fs)
    u = (np.arange(L) - 0.5 * (L - 1)) / fs
    chirp = np.exp(1j * np.pi * (bw / T) * u ** 2).astype(np.complex64)
    n = 4096
    tau0 = 2 * 10e3 / C
    tau = tau0 + 1234 / fs                                    # the echo starts 1234 samples into the window
    tau_ref = tau0 + 1000.0 / fs
    line = np.zeros(n, np.complex128)
    line[1234:1234 + L] = np.exp(-2j * np.pi * fc * tau) * chirp
    rc = raw.range_compress(line[None].astype(np.complex64), chirp)
    assert np.argmax(np.abs(rc[0])) == 1234
    S, fmin, df = raw.fx_history(rc, fs, fc, tau0, np.array([tau_ref]), bw)
    K = S.shape[1]
    f = fmin + df * np.arange(K)
    expect = np.exp(-2j * np.pi * f * (tau - tau_ref))
    strong = np.abs(S[0]) > 0.2 * np.abs(S[0]).max()           # the band's edges, where the chirp's spectrum rolls off, are left out
    ph = np.angle(S[0] * np.conj(expect))
    ph = (ph - ph[K // 2])[strong]
    print(f'fx_history: {K} bins over {K * df / 1e6:.1f} MHz, {strong.mean() * 100:.0f}% strong, phase deviation from the model '
          f'{np.abs(ph).max():.3f} rad')
    assert K * df == pytest.approx(bw, rel=0.02)
    assert strong.mean() > 0.85 and np.abs(ph).max() < 0.05


def test_zero_doppler_points():
    """The point lies at the asked range, perpendicular to the velocity, on the ellipsoid at the asked height, and
    to the right or left of the track as asked."""
    pos = io.geodetic_to_ecf(np.array([45.0, 45.1]), np.array([10.0, 10.0]), np.array([700e3, 700e3]))
    vel = 7500.0 * np.stack([sim.enu_axes(45.0, 10.0)[1], sim.enu_axes(45.1, 10.0)[1]])      # heading north
    east = np.stack([sim.enu_axes(45.0, 10.0)[0], sim.enu_axes(45.1, 10.0)[0]])
    for side, sign in (('right', 1), ('left', -1)):
        p = raw.zero_doppler_points(pos, vel, 900e3, side, height=100.0)
        d = p - pos
        assert np.allclose(np.linalg.norm(d, axis=1), 900e3, rtol=1e-9)
        assert np.abs((d * vel).sum(1) / np.linalg.norm(d, axis=1) / 7500.0).max() < 1e-9
        assert np.allclose(io.ecf_to_geodetic(p)[2], 100.0, atol=1e-3)
        assert np.all(sign * (d * east).sum(1) > 0)                 # heading north, right is east


@pytest.fixture(scope='module')
def nisar(tmp_path_factory):
    """A 20 km simulated collection with eight scatterers written as a NISAR L0B look-alike."""
    rng = np.random.default_rng(11)
    col = sim.make_collect(res=0.5, scene=60.0, r0=20e3)
    gx, gy = np.meshgrid([-16.0, -2.0, 14.0], [-15.0, 1.0, 17.0], indexing='ij')          # well apart, off the grid lines
    tg = np.stack([gx.ravel()[:8], gy.ravel()[:8], np.zeros(8)], 1)
    amp = np.ones(8) * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    path = os.path.join(tmp_path_factory.mktemp('nisar'), 'sim_l0b.h5')
    info = sim.write_nisar(path, col, tg, amp, lat0, lon0, h0, heading=heading)
    truth = io.ecf_to_geodetic(sim.to_ecf(tg, lat0, lon0, h0, heading))
    return dict(path=path, col=col, tg=tg, truth=truth, info=info)


def test_read_nisar(nisar):
    col, meta = io.read_nisar(nisar['path'], meta=True)
    S = col['S']
    P, K = S.shape
    assert P == len(nisar['col'].ant) and np.isfinite(S).all()
    assert K * col['df'] == pytest.approx(nisar['col'].K * nisar['col'].df, rel=0.02)
    assert meta['collector'] == 'NISAR' and meta['polarization'] == 'HH' and meta['mode'] == 'STRIPMAP'
    assert meta['image_area'].shape == (4, 3) and any('NISAR L0B' in n for n in meta['notes'])
    # the transmitter positions come from the orbit: within a millimeter of the simulator's track
    assert np.abs(io.local_to_ecf(meta['tx'], meta) - nisar['info']['tx']).max() < 1e-3
    with pytest.raises(ValueError, match='polarization'):
        io.read_nisar(nisar['path'], polarization='VV')
    with pytest.raises(ValueError, match='frequency'):
        io.read_nisar(nisar['path'], frequency='B')


def test_form_nisar(nisar):
    """form_cphd on the look-alike: every scatterer peaks within a third of a meter of the truth on the ground."""
    out = fastsar.form_cphd(nisar['path'], backend='cpu', spacing=0.4, height=h0, extent=(60.0, 60.0))
    assert out['mode'] == 'moving' or out['mode'] == 'spotlight'
    img = np.abs(out['image'])
    assert np.isfinite(img).all() and img.max() > 0
    ij = products.locate(out, *nisar['truth'])
    err = []
    for (i, j) in ij:
        i0, j0 = int(round(i)), int(round(j))
        w = img[max(0, i0 - 6):i0 + 7, max(0, j0 - 6):j0 + 7]
        pi, pj = np.unravel_index(np.argmax(w), w.shape)
        err.append(np.hypot((max(0, i0 - 6) + pi - i) * out['spx'], (max(0, j0 - 6) + pj - j) * out['spy']))
    pk = img.max() / np.median(img)
    print(f'NISAR look-alike: {img.shape} image, peak to median {pk:.0f}, worst target offset {max(err):.2f} m')
    assert max(err) < 0.35 and pk > 50


def test_read_collection_dispatch(nisar):
    col, meta = io.read_collection(nisar['path'], channel=0, meta=True, troposphere='model')
    assert col['S'].shape[0] > 0 and any('not applicable' in n and 'troposphere' in n for n in meta['notes'])


@pytest.fixture(scope='module')
def palsar(tmp_path_factory):
    """The same scene written as an ALOS PALSAR level 1.0 look-alike (CEOS leader and image files)."""
    rng = np.random.default_rng(12)
    col = sim.make_collect(res=0.5, scene=60.0, r0=20e3)
    gx, gy = np.meshgrid([-16.0, -2.0, 14.0], [-15.0, 1.0, 17.0], indexing='ij')
    tg = np.stack([gx.ravel()[:8], gy.ravel()[:8], np.zeros(8)], 1)
    amp = np.ones(8) * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    d = str(tmp_path_factory.mktemp('palsar'))
    info = sim.write_palsar(d, col, tg, amp, lat0, lon0, h0, heading=heading)
    truth = io.ecf_to_geodetic(sim.to_ecf(tg, lat0, lon0, h0, heading))
    return dict(dir=d, col=col, truth=truth, info=info)


def test_read_palsar(palsar):
    col, meta = io.read_palsar(palsar['dir'], meta=True)
    S = col['S']
    assert S.shape[0] == len(palsar['col'].ant) and np.isfinite(S).all()
    assert S.shape[1] * col['df'] == pytest.approx(palsar['col'].K * palsar['col'].df, rel=0.02)
    assert meta['polarization'] == 'HH' and meta['mode'] == 'STRIPMAP' and any('ALOS PALSAR' in n for n in meta['notes'])
    assert np.abs(io.local_to_ecf(meta['tx'], meta) - palsar['info']['tx']).max() < 1e-2        # microsecond line times
    with pytest.raises(ValueError, match='polarization'):
        io.read_palsar(palsar['dir'], polarization='VV')
    # the image file alone finds its leader beside it; a zip of the product reads too
    import os, zipfile
    img = [f for f in os.listdir(palsar['dir']) if f.startswith('IMG-')][0]
    c2 = io.read_palsar(os.path.join(palsar['dir'], img))
    assert np.array_equal(c2['S'], S)
    zp = os.path.join(palsar['dir'], 'product.zip')
    with zipfile.ZipFile(zp, 'w') as z:
        for f in os.listdir(palsar['dir']):
            if f.startswith(('IMG-', 'LED-')):
                z.write(os.path.join(palsar['dir'], f), f'ALPSRP000000000-L1.0/{f}')
    c3 = io.read_palsar(zp)
    assert np.array_equal(c3['S'], S)


def test_form_palsar(palsar):
    out = fastsar.form_cphd(palsar['dir'], backend='cpu', spacing=0.4, height=h0, extent=(60.0, 60.0))
    img = np.abs(out['image'])
    ij = products.locate(out, *palsar['truth'])
    err = []
    for (i, j) in ij:
        i0, j0 = int(round(i)), int(round(j))
        w = img[max(0, i0 - 6):i0 + 7, max(0, j0 - 6):j0 + 7]
        pi, pj = np.unravel_index(np.argmax(w), w.shape)
        err.append(np.hypot((max(0, i0 - 6) + pi - i) * out['spx'], (max(0, j0 - 6) + pj - j) * out['spy']))
    pk = img.max() / np.median(img)
    print(f'PALSAR look-alike: {img.shape} image, peak to median {pk:.0f}, worst target offset {max(err):.2f} m')
    assert max(err) < 0.35 and pk > 50


@pytest.fixture(scope='module')
def sentinel1(tmp_path_factory):
    """The same scene at 2 m resolution written as a Sentinel-1 Level-0 measurement file look-alike."""
    rng = np.random.default_rng(13)
    col = sim.make_collect(res=2.0, scene=120.0, r0=20e3, fc=5.405e9)           # Sentinel-1's carrier, which the reader assumes
    gx, gy = np.meshgrid([-40.0, -5.0, 35.0], [-38.0, 2.0, 42.0], indexing='ij')
    tg = np.stack([gx.ravel()[:8], gy.ravel()[:8], np.zeros(8)], 1)
    amp = np.ones(8) * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    path = os.path.join(tmp_path_factory.mktemp('s1'), 's1z-s1-raw-s-hh-sim.dat')
    info = sim.write_sentinel1(path, col, tg, amp, lat0, lon0, h0, heading=heading)
    truth = io.ecf_to_geodetic(sim.to_ecf(tg, lat0, lon0, h0, heading))
    return dict(path=path, col=col, truth=truth, info=info)


def test_read_sentinel1(sentinel1):
    col, meta = io.read_sentinel1(sentinel1['path'], meta=True, height=h0)
    S = col['S']
    assert S.shape[0] == len(sentinel1['col'].ant) and np.isfinite(S).all()
    assert S.shape[1] * col['df'] == pytest.approx(sentinel1['col'].K * sentinel1['col'].df, rel=0.03)
    assert meta['polarization'] == 'HH' and meta['mode'] == 'STRIPMAP' and any('Sentinel-1' in n for n in meta['notes'])
    assert not any('ended before' in n for n in meta['notes'])
    assert np.abs(io.local_to_ecf(meta['tx'], meta) - sentinel1['info']['tx']).max() < 0.2       # 16-bit fine time: 15 us, 0.1 m
    with pytest.raises(ValueError, match='polarization'):
        io.read_sentinel1(os.path.dirname(sentinel1['path']), polarization='VV')
    c2, _ = io.read_collection(sentinel1['path'], meta=True, pulses=(0, 50), height=h0)
    assert c2['S'].shape[0] == 50


def test_form_sentinel1(sentinel1):
    out = fastsar.form_cphd(sentinel1['path'], backend='cpu', spacing=1.0, height=h0, extent=(120.0, 120.0))
    img = np.abs(out['image'])
    ij = products.locate(out, *sentinel1['truth'])
    err = []
    for (i, j) in ij:
        i0, j0 = int(round(i)), int(round(j))
        w = img[max(0, i0 - 5):i0 + 6, max(0, j0 - 5):j0 + 6]
        pi, pj = np.unravel_index(np.argmax(w), w.shape)
        err.append(np.hypot((max(0, i0 - 5) + pi - i) * out['spx'], (max(0, j0 - 5) + pj - j) * out['spy']))
    pk = img.max() / np.median(img)
    print(f'Sentinel-1 look-alike: {img.shape} image, peak to median {pk:.0f}, worst target offset {max(err):.2f} m')
    assert max(err) < 1.0 and pk > 50
