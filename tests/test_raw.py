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


def test_form_nisar_dithered(tmp_path):
    """Dithered pulse timing (NISAR's PRI dithering): the reader resamples the history to uniform pulse times and
    the targets still focus where they are, within half a meter."""
    rng = np.random.default_rng(21)
    col = sim.make_collect(res=0.5, scene=60.0, r0=20e3)
    gx, gy = np.meshgrid([-16.0, -2.0, 14.0], [-15.0, 1.0, 17.0], indexing='ij')
    tg = np.stack([gx.ravel()[:6], gy.ravel()[:6], np.zeros(6)], 1)
    amp = np.ones(6) * np.exp(1j * rng.uniform(0, 2 * np.pi, 6))
    path = os.path.join(tmp_path, 'dithered.h5')
    sim.write_nisar(path, col, tg, amp, lat0, lon0, h0, heading=heading, dither=0.08)
    col_, meta = io.read_nisar(path, meta=True)
    assert any('dithered' in n for n in meta['notes'])
    out = fastsar.form_cphd(path, backend='cpu', spacing=0.4, height=h0, extent=(60.0, 60.0))
    img = np.abs(out['image'])
    truth = io.ecf_to_geodetic(sim.to_ecf(tg, lat0, lon0, h0, heading))
    err = []
    for (i, j) in products.locate(out, *truth):
        i0, j0 = int(round(i)), int(round(j))
        w = img[max(0, i0 - 6):i0 + 7, max(0, j0 - 6):j0 + 7]
        pi, pj = np.unravel_index(np.argmax(w), w.shape)
        err.append(np.hypot((max(0, i0 - 6) + pi - i) * out['spx'], (max(0, j0 - 6) + pj - j) * out['spy']))
    pk = img.max() / np.median(img)
    print(f'dithered NISAR look-alike: worst target offset {max(err):.2f} m, peak to median {pk:.0f}')
    assert max(err) < 0.5 and pk > 50


def test_read_collection_dispatch(nisar):
    col, meta = io.read_collection(nisar['path'], channel=0, meta=True, troposphere='model', block=16)
    assert col['S'].shape[0] > 0 and any('troposphere delay removed' in n for n in meta['notes'])


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


def test_palsar_header_blank_samples():
    """JAXA's level 1.0 image descriptors leave the pixels-per-line field blank: the sample count comes from the data
    bytes per record."""
    from fastsar import raw
    rec = bytearray(b' ' * 720)
    rec[180:186] = b' 35421'; rec[216:220] = b'   8'; rec[224:228] = b'   2'; rec[236:244] = b'   35421'
    rec[276:280] = b' 412'; rec[280:288] = b'   20688'
    h = raw.palsar_image_header(bytes(rec))
    assert h['samples'] == 10344 and h['record_length'] == 21100


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


def test_nisar_product_options(nisar, tmp_path):
    """orbit= and range_delay= from an RSLC-like product file: the look-alike's own orbit written under a product's
    metadata group with another epoch, and its delay as JPL records it (sign reversed) give the same positions and
    the same history, with notes naming the file; a tuple orbit with its own epoch; an orbit that does not cover the
    pulses is refused; the reference range option is noted."""
    import h5py
    path = nisar['path']
    col0, meta0 = io.read_nisar(path, meta=True)
    with h5py.File(path, 'r') as f:
        orb = f['science/LSAR/RRSD/lowRateTelemetry/orbit']
        ot, op, ov = orb['time'][()], orb['position'][()], orb['velocity'][()]
    prod = os.path.join(tmp_path, 'rslc_like.h5')
    with h5py.File(prod, 'w') as f:
        o = f.create_group('science/LSAR/RSLC/metadata/orbit')
        d = o.create_dataset('time', data=ot - 3600.0)
        d.attrs['units'] = 'seconds since 2026-01-01T01:00:00'            # an hour after the look-alike's epoch
        o['position'], o['velocity'], o['orbitType'] = op, ov, np.bytes_(b'MOE')
        f.create_group('science/LSAR/RSLC/metadata/calibrationInformation/frequencyA')['commonDelay'] = -io.NISAR_RANGE_DELAY['A']
        f.create_group('science/LSAR/identification')
    info = io.nisar_product(prod)
    assert info['product'] == 'RSLC' and info['orbit_type'] == 'MOE' and info['epoch'] == '2026-01-01T01:00:00'
    assert info['range_delay'] == {'A': pytest.approx(io.NISAR_RANGE_DELAY['A'])}
    col1, meta1 = io.read_nisar(path, meta=True, orbit=prod, range_delay=prod)
    assert np.abs(meta1['tx'] - meta0['tx']).max() < 1e-6 and np.abs(meta1['ref'] - meta0['ref']).max() < 1e-6
    assert np.allclose(col1['S'], col0['S'])
    assert any('from the RSLC file' in n for n in meta1['notes']) and any('(the RSLC file)' in n for n in meta1['notes'])
    _, meta2 = io.read_nisar(path, meta=True, orbit=(ot + 60.0, op, ov, '2025-12-31T23:59:00'))
    assert np.abs(meta2['tx'] - meta0['tx']).max() < 1e-6
    with pytest.raises(ValueError, match='spans'):
        io.read_nisar(path, orbit=(ot + 60.0, op, ov))
    _, meta3 = io.read_nisar(path, meta=True, reference_range=float(meta0['ref'].mean()) + 500.0, troposphere=None)
    assert any('reference points at' in n for n in meta3['notes']) and not any('troposphere' in n for n in meta3['notes'])
    assert any('troposphere delay removed' in n for n in meta0['notes'])            # the model by default


def test_sentinel1_orbit_file(sentinel1, tmp_path):
    """orbit= for Sentinel-1: the look-alike's packet records written as an ESA orbit file (TAI stamps, 19 s ahead
    of GPS) read back through sentinel1.orbit_from_file give the same positions and the same image geometry."""
    from fastsar import sentinel1 as s1
    dat = safe = sentinel1['path']
    hdr = s1.parse_packets(np.memmap(dat, np.uint8, 'r'))
    ot, op, ov = s1.orbit_from_packets(hdr)
    gps0 = np.datetime64('1980-01-06T00:00:00', 'ns')
    eof = os.path.join(tmp_path, 'S1A_OPER_AUX_POEORB_OPOD_test.EOF')
    with open(eof, 'w') as f:
        f.write('<?xml version="1.0"?>\n<Earth_Explorer_File><Data_Block><List_of_OSVs>\n')
        for ti, pi, vi in zip(ot, op, ov):
            tai = gps0 + np.timedelta64(int(round((ti + 19.0) * 1e9)), 'ns')
            f.write(f'<OSV><TAI>TAI={str(tai)}</TAI><UTC>UTC=unused</UTC><X unit="m">{pi[0]:.6f}</X><Y unit="m">{pi[1]:.6f}</Y>'
                    f'<Z unit="m">{pi[2]:.6f}</Z><VX unit="m/s">{vi[0]:.6f}</VX><VY unit="m/s">{vi[1]:.6f}</VY><VZ unit="m/s">{vi[2]:.6f}</VZ>'
                    f'<Quality>NOMINAL</Quality></OSV>\n')
        f.write('</List_of_OSVs></Data_Block></Earth_Explorer_File>\n')
    t2, p2, v2 = s1.orbit_from_file(eof)
    assert np.abs(t2 - ot).max() < 1e-6 and np.abs(p2 - op).max() < 1e-5 and np.abs(v2 - ov).max() < 1e-5
    _, meta0 = io.read_sentinel1(safe, meta=True)
    _, meta1 = io.read_sentinel1(safe, meta=True, orbit=eof)
    assert np.abs(meta1['tx'] - meta0['tx']).max() < 1e-5 and any('orbit file' in n for n in meta1['notes'])
    with pytest.raises(ValueError, match='spans'):
        io.read_sentinel1(safe, orbit=(ot + 100.0, op, ov))
