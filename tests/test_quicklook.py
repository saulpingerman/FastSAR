"""fastsar.quicklook on simulated spotlight collections (sim.write_cphd): the reader's pulse and sample selection
against slicing the whole read; the cut (the central band and pulses of the resolution asked for, never more than the
file holds); the picture's geometry (point targets where the product's own geometry puts them, shadows down, the
corners' coordinates); the complex image against exact backprojection of the same cut history on the same grid;
the output files; the refusals; the factorized former on a JAX device; and the CUDA path under the `cuda` marker."""
import os
import zlib

import numpy as np
import pytest

import fastsar
from fastsar import io, quick, sim

C = 299792458.0
LAT, LON = 35.0, -106.0
# simulator frame: x across track toward the scene, y along track, z up (m)
TARGETS = np.array([[0.0, 0.0, 0.0], [50.0, 40.0, 0.0], [-50.0, 40.0, 0.0], [50.0, -40.0, 0.0], [-50.0, -40.0, 0.0]])


def look(*args, **kw):
    """fastsar.quicklook on the cpu backend unless a test names another ('auto' is cuda where CuPy has a GPU)."""
    kw.setdefault('backend', 'cpu')
    return fastsar.quicklook(*args, **kw)


def rel_db(a, b):
    a, b = a.astype(np.complex128).ravel(), b.astype(np.complex128).ravel()
    g = np.vdot(a, b) / np.vdot(a, a)
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


@pytest.fixture(scope='module')
def spot(tmp_path_factory):
    """A 0.3 m spotlight at 30 km and 35 degrees of grazing: five points, a little noise, a 150 m image area, and a
    troposphere delay that the file states and its samples hold (the reader removes it)."""
    rng = np.random.default_rng(5)
    col = sim.make_collect(res=0.3, scene=200.0, r0=30e3, graze_deg=35.0)
    S = sim.simulate_brute(col, TARGETS, np.ones(len(TARGETS)) + 0j)
    S += 0.02 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
    tropo = np.linspace(8e-9, 9e-9, col.Np)
    S = (S * np.exp(-2j * np.pi * col.freqs[None, :] * tropo[:, None])).astype(np.complex64)
    d = tmp_path_factory.mktemp('quicklook')
    path = str(d / 'spot.cphd')
    sim.write_cphd(path, col, S, LAT, LON, scene=150.0, tropo=tropo)
    return dict(path=path, col=col, S=S, dir=d)


# ---------------------------------------------------------------------------------------------------- the reader

def test_selection_equals_slicing_the_whole_read(spot):
    """read_cphd(pulses=, samples=) gives the rows and columns of the whole read, value for value: the samples, the
    frequency of the first one, the antenna positions (through ECF: the local frame is the selection's own)."""
    path, P, K = spot['path'], *spot['S'].shape
    full, mf = io.read_cphd(path, meta=True)
    assert 'removed the troposphere delay' in ' '.join(mf['notes'])            # the phase ramp is exercised
    for pulses, samples in (((10, P - 7), (37, K - 21)), (None, (16, 80)), ((5, 60), None), ((0, P), (0, K)),
                            ((3, 5), (K - 2, K))):
        part, mp = io.read_cphd(path, meta=True, pulses=pulses, samples=samples)
        (pa, pb), (k0, k1) = pulses or (0, P), samples or (0, K)
        assert part['S'].shape == (pb - pa, k1 - k0) and part['S'].flags.c_contiguous
        assert np.array_equal(part['S'], full['S'][pa:pb, k0:k1]), (pulses, samples)
        assert part['fmin'] == pytest.approx(full['fmin'] + k0 * full['df'], rel=1e-15) and part['df'] == full['df']
        assert mp['pulses'] == (pa, pb) and mp['samples'] == (k0, k1)
        assert mp['signal_bytes'] <= (pb - pa) * (k1 - k0 + 15) * 8            # a multiple of 16 samples before
        np.testing.assert_allclose(io.local_to_ecf(part['ant'], mp), io.local_to_ecf(full['ant'][pa:pb], mf), atol=1e-6)
        np.testing.assert_allclose(mp['tx_time'], mf['tx_time'][pa:pb])
    for bad in (dict(pulses=(5, 6)), dict(samples=(0, K + 1)), dict(pulses=(1.5, 9)), dict(samples=3), dict(pulses=(-1, 9))):
        with pytest.raises(ValueError, match='first, last'):
            io.read_cphd(path, **bad)


def test_phase_ramp_of_a_selection_has_the_whole_rows_values():
    """The compiled phase ramp on the samples first .. last - 1 of longer rows gives each sample the value it has in
    the pass over the whole rows, wherever the selection ends in a block of 16 samples (the vector part and the
    remainder of the kernel's loop round differently on some CPUs)."""
    rng = np.random.default_rng(3)
    P, K = 9, 203
    S = (rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K))).astype(np.complex64)
    f0, df, coef = 9.6e9 + rng.uniform(0, 1e6, P), 2.5e5 + rng.uniform(0, 10, P), rng.uniform(-3e-9, 3e-9, P)

    def freq(first, n):
        kk = np.arange(first, first + n)

        def f(sl, out):
            out[:] = f0[sl, None] + df[sl, None] * kk[None, :]
        return f
    full = io._phase_rows(S.copy(), freq(0, K), coef, +1, linear=(f0, df))
    assert not np.array_equal(full, S)
    for first in (0, 16, 96, 192):
        for last in range(first + 1, K + 1):
            part = io._phase_rows(S[:, first:last].copy(), freq(first, last - first), coef, +1, linear=(f0, df),
                                  first=first, total=K)
            assert np.array_equal(part, full[:, first:last]), (first, last)


def test_selection_without_the_compiled_pass(spot, monkeypatch):
    """The NumPy passes (no compiler, or FASTSAR_IO_KERNEL=0) select the same values."""
    path, P, K = spot['path'], *spot['S'].shape
    monkeypatch.setenv('FASTSAR_IO_KERNEL', '0')
    monkeypatch.setattr(io, '_LIB', [])
    full = io.read_cphd(path)
    part = io.read_cphd(path, pulses=(10, P - 7), samples=(37, K - 21))
    assert io._lib() is None
    assert np.array_equal(part['S'], full['S'][10:P - 7, 37:K - 21])


def test_selection_reads_its_bytes_alone(spot):
    """The compiled reader asks the file for the selection's bytes, in pulse blocks: read() calls counted by the
    process's own accounting (Linux)."""
    if not os.path.exists('/proc/self/io') or io._lib() is None:
        pytest.skip('needs /proc/self/io and the compiled reader')
    rchar = lambda: int(dict(line.split(': ') for line in open('/proc/self/io').read().strip().split('\n'))['rchar'])
    path, P, K = spot['path'], *spot['S'].shape
    io.read_cphd(path, pulses=(0, 4), samples=(0, 4))                          # the header's and the kernels' reads
    a = rchar()
    meta = io.read_cphd(path, meta=True, pulses=(100, 400), samples=(160, 480))[1]
    used = rchar() - a
    assert meta['signal_bytes'] == 300 * 320 * 8
    assert meta['signal_bytes'] <= used < meta['signal_bytes'] + (1 << 20), used   # with the header and the 300 vectors
    assert used < 0.2 * os.path.getsize(path)


def test_selection_on_per_pulse_frequency_grids(spot, tmp_path):
    """A file whose pulses start at different frequencies is resampled to one grid: a sample selection resamples
    from the samples its kernel reaches and equals the whole read's columns."""
    import shutil
    from fastsar import _cphdxml
    path = str(tmp_path / 'grids.cphd')
    shutil.copy(spot['path'], path)
    r = _cphdxml.open_cphd(path)
    m = np.memmap(path, dtype=r.pvp_dtype(), mode='r+', offset=r.pvp_map(0).offset, shape=(spot['col'].Np,))
    step = float(m['SCSS'][0])
    m['SC0'] = m['SC0'] + step * 0.4 * np.sin(np.arange(len(m)))                # up to 0.4 samples off the median
    m.flush()
    del m
    full, mf = io.read_cphd(path, meta=True)
    assert 'resampled per-pulse frequency grids' in ' '.join(mf['notes'])
    K = full['S'].shape[1]
    for samples in ((40, 200), (0, 64), (K - 100, K)):
        part = io.read_cphd(path, samples=samples)
        assert np.array_equal(part['S'], full['S'][:, samples[0]:samples[1]]), samples


@pytest.mark.needs('sarpy')
def test_selection_cphd03(spot, tmp_path):
    """The same selection of a CPHD 0.3 file."""
    from test_cphd03 import write_cphd03
    path = str(tmp_path / 'v03.cphd')
    write_cphd03(path, spot['path'])
    full, mf = io.read_cphd(path, meta=True)
    part, mp = io.read_cphd(path, meta=True, pulses=(20, 300), samples=(48, 500))
    assert mf['cphd_version'].startswith('0.3')
    assert np.array_equal(part['S'], full['S'][20:300, 48:500])
    assert mp['pulses'] == (20, 300) and mp['samples'] == (48, 500)
    out = look(path, size=128)
    assert out['display'].shape == (128, 128) and out['cut']['of_samples'] == full['S'].shape[1]


def test_selection_through_form_cphd(spot):
    """form_cphd's reader_options carry the selection: the image of a part of the band and of the aperture."""
    P, K = spot['S'].shape
    out = fastsar.form_cphd(spot['path'], backend='cpu', reader_options=dict(pulses=(200, P - 200), samples=(304, K - 300)))
    col = spot['col']
    assert out['meta']['pulses'] == (200, P - 200) and out['meta']['samples'] == (304, K - 300)
    assert out['band'][0] == pytest.approx(col.fmin + 304 * col.df, rel=1e-12)
    assert out['band'][1] == pytest.approx(col.fmin + (K - 301) * col.df, rel=1e-12)
    whole = fastsar.form_cphd(spot['path'], backend='cpu')
    assert out['spy'] == pytest.approx(whole['spy'] * K / (K - 604), rel=1e-6)   # the spacing follows the band


# ------------------------------------------------------------------------------------------------------- the cut

def test_cut_is_the_central_band_and_pulses(spot):
    """The cut is the central run of samples and pulses whose bands are 0.96 over the pixel spacings, read alone."""
    P, K = spot['S'].shape
    col = spot['col']
    out = look(spot['path'], size=256)
    cut = out['cut']
    (pa, pb), (k0, k1) = cut['pulses'], cut['samples']
    assert cut['of_pulses'] == P and cut['of_samples'] == K and out['pulses'] == (0, pb - pa)
    assert cut['signal_bytes'] == (pb - pa) * (k1 - k0) * 8 and cut['of_bytes'] == P * K * 8
    assert max(out['display'].shape) == 256 and out['algorithm'] == 'pfa' and out['correct'] is True
    # the band of the samples kept is 0.96 / spy, that of the pulses 0.96 / spx (to a sample and a pulse or two)
    assert (k1 - k0) * col.df * 2 / C * np.cos(out['graze']) == pytest.approx(0.96 / out['spy'], rel=2 / (k1 - k0))
    assert out['bandwidth'][0] == pytest.approx(0.96 / out['spx'], rel=3 / (pb - pa))
    assert out['bandwidth'][1] == pytest.approx(0.96 / out['spy'], rel=2 / (k1 - k0))
    # central: within the 16 samples the reader's alignment may move the band, and a pulse
    assert abs((k0 + k1) / 2 - K / 2) <= 16 and abs((pa + pb) / 2 - P / 2) <= 2
    assert out['band'][0] == pytest.approx(col.fmin + k0 * col.df, rel=1e-12)
    assert out['band'][1] == pytest.approx(col.fmin + (k1 - 1) * col.df, rel=1e-12)
    # the pixel is 0.96 of the resolution, the same on the ground along both axes
    assert out['pixel'] == pytest.approx(0.96 * out['resolution']) and out['resolution'] > out['finest']
    assert out['pixel'] == out['spx'] == out['spy'] and out['graze'] == pytest.approx(np.radians(35.0), abs=1e-3)
    assert out['plane'] == 'ground' and out['axes'] == 'los'
    # half the pixels: half the band on both axes
    small = look(spot['path'], size=128)
    assert small['cut']['samples'][1] - small['cut']['samples'][0] == pytest.approx((k1 - k0) / 2, abs=2)
    assert small['cut']['pulses'][1] - small['cut']['pulses'][0] == pytest.approx((pb - pa) / 2, abs=3)
    assert small['pixel'] == pytest.approx(2 * out['pixel'], rel=1e-6)


def test_never_more_than_the_collection_holds(spot):
    """resolution='full' uses the whole of the axis that limits the resolution; a finer request gets that, with a
    note; a resolution in meters sets the pixel."""
    P, K = spot['S'].shape
    full = look(spot['path'], resolution='full')
    assert full['cut']['samples'] == (0, K) and full['cut']['pulses'][1] - full['cut']['pulses'][0] <= P
    assert full['resolution'] == pytest.approx(full['finest'])
    assert full['finest'] == pytest.approx(C / (2 * K * spot['col'].df) / np.cos(np.radians(35.0)), rel=1e-3)
    fine = look(spot['path'], resolution=0.01)
    assert fine['cut'] == full['cut'] and fine['display'].shape == full['display'].shape
    assert any('the collection holds a resolution' in n for n in fine['notes'])
    one = look(spot['path'], resolution=1.0, weighting='uniform')
    assert one['pixel'] == pytest.approx(0.96) and one['window'] is None
    assert one['display'].shape == tuple(2 * int(np.ceil(75.0 / 0.96 - 1e-6)) for _ in range(2))


def test_extent(spot):
    """extent=(across, along) the line of sight on the ground replaces the file's image area."""
    out = look(spot['path'], size=128, extent=(120.0, 60.0))
    assert out['display'].shape == (64, 128)                                   # rows along range
    assert out['pixel'] == pytest.approx(120.0 / 128)


# ------------------------------------------------------------------------------------------------- the picture

def _expected_pixels(out):
    """Where the product's own geometry puts the simulated targets, which lie on the picture's plane, in the
    display: rows and columns."""
    q = io.ecf_to_local(sim.to_ecf(TARGETS, LAT, LON), out['meta']) - out['display_origin']
    return q @ out['row'] / out['row_ss'], q @ out['col'] / out['col_ss']


@pytest.mark.parametrize('correct', [True, False])
def test_points_where_they_should_be(spot, correct):
    """Five point targets appear at the pixels the product's geometry gives, the picture is laid out shadows down
    and seen from above, and its corners lie on the scene."""
    out = look(spot['path'], size=256, correct=correct)
    d = out['display']
    assert d.dtype == np.uint8 and d.shape == (256, 256) and d.flags.c_contiguous
    rows, cols = _expected_pixels(out)
    for r, c in zip(rows, cols):
        i, j = int(round(r)), int(round(c))
        box = d[i - 4:i + 5, j - 4:j + 5].astype(int)
        pi, pj = np.unravel_index(box.argmax(), box.shape)
        assert box.max() >= 250 and abs(pi - 4) <= 1 and abs(pj - 4) <= 1, (r, c, pi, pj, box.max())
    assert (d > 250).sum() < 200                                               # and nowhere else
    # shadows down, from above: rows along the ground range away from the radar, row x col upward
    meta = out['meta']
    ant = 0.5 * (meta['tx'] + meta['rcv'])
    los = (out['center'] - ant[len(ant) // 2]) * np.array([1.0, 1.0, 0.0])
    assert out['row'] @ los > 0.999 * np.linalg.norm(los) and np.cross(out['row'], out['col'])[2] > 0.999
    assert out['plane'] == 'ground' and abs(out['e2'] @ out['row']) > 0.999
    # the corner pixels on the ground: the 150 m area about the scene center (their centers' mean lies half a
    # pixel from it: pixel i of n sits at (i - n / 2) spacings)
    lat, lon = out['corners'].T
    assert lat.mean() == pytest.approx(LAT, abs=5e-6) and lon.mean() == pytest.approx(LON, abs=5e-6)
    ecf = io.geodetic_to_ecf(lat, lon, np.zeros(4))
    side = np.linalg.norm(ecf - np.roll(ecf, 1, axis=0), axis=1)
    np.testing.assert_allclose(side, 150.0 * 255 / 256, rtol=5e-3)
    assert out['displacement_px'][0] < 1 and out['displacement_px'][1] < 1
    assert any('displacement' in n and ('removed' if correct else 'left in') in n for n in out['notes'])


def test_complex_image_against_exact_backprojection(spot):
    """The corrected complex image equals exact backprojection of the same cut history on the same grid to polar
    format's accuracy, in amplitude and phase; the amplitude image is its modulus; the product's complex image is
    the same array."""
    col = quick._collection(spot['path'], size=256)
    S, ant, f0, df = col['S'], col['ant'], col['fmin'], col['df']
    grid = tuple(col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2'))
    ex = fastsar.ExactFormer(ant, f0, df, S.shape[1], *grid, backend='cpu', window=True)(S)
    info = {}
    img = quick._pfa(S, ant, f0, df, *grid, correct=True, window=True, info=info)
    e = rel_db(img, ex)
    g = np.vdot(img.ravel(), ex.ravel()) / np.vdot(img.ravel(), img.ravel())
    print(f'quick look against exact backprojection: {e:.1f} dB, gain {abs(g):.4f}, displacement {info["shift_px"]} px')
    assert e < -30 and abs(g) == pytest.approx(1.0, abs=0.01) and abs(np.angle(g)) < 0.01
    amp = quick._pfa(S, ant, f0, df, *grid, correct=True, window=True, complex_image=False)
    assert amp.dtype == np.float32 and np.abs(amp - np.abs(img)).max() < 1e-5 * np.abs(img).max()
    # uncorrected: the same amplitudes where the displacement is a small part of a pixel, as here
    raw = quick._pfa(S, ant, f0, df, *grid, correct=False, window=True, complex_image=False)
    assert rel_db(raw.astype(np.complex64), np.abs(ex).astype(np.complex64)) < -15
    # unweighted, against the unweighted exact image
    ex0 = fastsar.ExactFormer(ant, f0, df, S.shape[1], *grid, backend='cpu', window=False)(S)
    assert rel_db(quick._pfa(S, ant, f0, df, *grid, correct=True, window=False), ex0) < -25
    out = look(spot['path'], size=256, complex_image=True)
    assert out['image'].shape == grid[:2] and out['image'].dtype == np.complex64 and out['image'].flags.c_contiguous
    assert np.abs(out['image'] - img).max() <= 1e-5 * np.abs(img).max()
    assert look(spot['path'], size=256)['image'] is None
    # a picture polar format holds: its own error at the corners is far below the kernels', and no note speaks of it
    assert out['corner_error'][1] < -40 and not any('defocused' in n for n in out['notes'])


@pytest.fixture(scope='module')
def squinted(tmp_path_factory):
    """A 0.5 m spotlight at 60 degrees of grazing whose track passes 18 km to the side of broadside: the slant plane
    is twisted 42 degrees about the line of sight, so that on the ground the lines of equal range and of equal
    Doppler meet at an angle and the band of a pulse drifts along the aperture."""
    col = sim.make_collect(res=0.5, scene=160.0, r0=30e3, graze_deg=60.0, offset=(0.0, 18e3, 0.0))
    S = sim.simulate_brute(col, TARGETS, np.ones(len(TARGETS)) + 0j).astype(np.complex64)
    path = str(tmp_path_factory.mktemp('squinted') / 'squinted.cphd')
    sim.write_cphd(path, col, S, LAT, LON, scene=120.0)
    return dict(path=path, col=col, S=S)


def test_whole_scene_under_a_twisted_slant_plane(squinted):
    """The picture is the tasked area on the ground along the line of sight, whatever the twist of the slant plane:
    its points are where the product's geometry puts them, and the complex image is exact backprojection's of the
    same cut on the same grid, corrected and windowed or not. The raster is then sheared: each pulse holds its own
    rows of range wavenumbers, and they fold into the transform's period."""
    out = look(squinted['path'], size=128, complex_image=True)
    d = out['display']
    assert out['plane'] == 'ground' and max(d.shape) == 128 and not any('cut to the central' in n for n in out['notes'])
    rows, cols = _expected_pixels(out)
    for r, c in zip(rows, cols):
        i, j = int(round(r)), int(round(c))
        box = d[i - 4:i + 5, j - 4:j + 5].astype(int)
        pi, pj = np.unravel_index(box.argmax(), box.shape)
        assert box.max() >= 250 and abs(pi - 4) <= 1 and abs(pj - 4) <= 1, (r, c, pi, pj, box.max())
    # the tasked square, 120 m, seen along a line of sight 31 degrees from its sides
    lat, lon = out['corners'].T
    ecf = io.geodetic_to_ecf(lat, lon, np.zeros(4))
    side = np.linalg.norm(ecf - np.roll(ecf, 1, axis=0), axis=1)
    rot = np.arctan2(18e3, 30e3 * np.cos(np.radians(60.0)))
    assert side == pytest.approx(120.0 * (np.cos(rot) + np.sin(rot)) * 127 / 128, rel=0.02)
    col = quick._collection(squinted['path'], size=128)
    S, ant, f0, df = col['S'], col['ant'], col['fmin'], col['df']
    grid = tuple(col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2'))
    g = quick._geometry(ant, f0, df, S.shape[1], *grid, True)
    assert g['jlo'] is not None and g['nkr'] > 1.2 * g['nwin'] and abs(g['shear']) > 0.3
    for window, limit in ((True, -35), (False, -25)):
        ex = fastsar.ExactFormer(ant, f0, df, S.shape[1], *grid, backend='cpu', window=window)(S)
        e = rel_db(quick._pfa(S, ant, f0, df, *grid, window=window), ex)
        ea = rel_db(quick._pfa(S, ant, f0, df, *grid, window=window, correct=False, complex_image=False), np.abs(ex))
        print(f'twisted slant plane, window {window}: {e:.1f} dB against exact backprojection, uncorrected amplitude {ea:.1f} dB')
        assert e < limit and ea < -15
    assert rel_db(out['image'], fastsar.ExactFormer(ant, f0, df, S.shape[1], *grid, backend='cpu', window=True)(S)) < -35


def test_distortion_on_any_axes():
    """The apparent position is the weighted least-squares line of -dR / ur over the tangent of the look angle: on a
    slant plane along the line of sight it is pfa2.distortion's to a fraction of a pixel, and on the ground, where a
    meter of range is not a meter along the axis, the corrected image has backprojection's phase."""
    from fastsar import pfa2
    col = sim.make_collect(res=0.3, scene=200.0, r0=30e3, graze_deg=35.0)
    P = col.Np
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    e1 = np.cross(e2, n)
    own = quick._distortion(col.ant[::8], 256, 256, 0.5, 0.5, e1, e2)
    rel = pfa2.distortion(sim.Collect(fmin=col.fmin, df=col.df, K=col.K, ant=col.ant[::8], res=0.3), 256, 256, 0.5, 0.5, e1, e2)
    assert np.abs(np.array(own['cx']) - rel['cx']).max() < 0.01 and np.abs(np.array(own['cy']) - rel['cy']).max() < 0.01
    rng = np.random.default_rng(2)
    tg = np.stack([rng.uniform(-40, 40, 30), rng.uniform(-40, 40, 30), np.zeros(30)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    grid = (360, 290, 0.28, 0.35, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))      # the ground
    ex = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, *grid, backend='cpu', window=True)(S)
    img = quick._pfa(S, col.ant, col.fmin, col.df, *grid, window=True)
    g = np.vdot(img.ravel(), ex.ravel()) / np.vdot(img.ravel(), img.ravel())
    assert rel_db(img, ex) < -35 and abs(np.angle(g)) < 0.01 and abs(g) == pytest.approx(1.0, abs=0.01)


def test_missing_pulses_keep_their_places():
    """A collection that lacks pulses of an evenly timed sequence (their neighbors' positions are a double step
    apart): each pulse is set at its place and the missing ones' places stay zero, so that a scatterer at the edge
    of the picture, whose phase steps across a gap, is formed as backprojection forms it."""
    rng = np.random.default_rng(8)
    col = sim.make_collect(res=0.5, scene=260.0, r0=60e3)
    keep = np.ones(col.Np, bool)
    keep[[40, 41, 300, 700, 702]] = False
    P = col.Np
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    e1 = np.cross(e2, n)
    grid = (480, 200, 0.48, 0.48, e1, e2)
    tg = np.concatenate([rng.uniform(-110, -100, (15, 1)) * e1 + rng.uniform(-40, 40, (15, 1)) * e2,     # at the far edge
                         rng.uniform(100, 110, (15, 1)) * e1 + rng.uniform(-40, 40, (15, 1)) * e2])
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    ant, S = col.ant[keep], np.ascontiguousarray(S[keep])
    u = ant / np.linalg.norm(ant, axis=1)[:, None]
    place = quick._places((u @ e1) / -(u @ e2))
    assert place[-1] == P - 1 and np.array_equal(place, np.nonzero(keep)[0])
    g = quick._geometry(ant, col.fmin, col.df, col.K, *grid, True)
    assert g['Pp'] == P and g['place'] is not None
    ex = fastsar.ExactFormer(ant, col.fmin, col.df, col.K, *grid, backend='cpu', window=True)(S)
    e = rel_db(quick._pfa(S, ant, col.fmin, col.df, *grid, window=True), ex)
    print(f'five pulses missing: {e:.1f} dB against exact backprojection of the pulses there are')
    assert e < -38
    assert np.array_equal(quick._places(np.linspace(-1, 1, 500) ** 3 + np.linspace(-3, 3, 500)), np.arange(500))


def test_taylor_window_is_scipys():
    from scipy.signal.windows import taylor
    for n in (2, 3, 64, 1001):
        np.testing.assert_allclose(quick._taylor(n), taylor(n, nbar=4, sll=35.0, norm=False), rtol=0, atol=1e-13)


def test_uneven_pulse_spacing():
    """Pulses that are not evenly spaced along the track (a varying pulse rate) are resampled at their own
    positions: the image keeps its accuracy against exact backprojection."""
    rng = np.random.default_rng(3)
    col = sim.make_collect(res=0.5, scene=90.0, r0=20e3)
    P = col.Np
    y = col.ant[:, 1]
    col.ant[:, 1] = y + 0.02 * (y[-1] - y[0]) * np.sin(np.pi * np.arange(P) / (P - 1)) ** 2      # up to 6% in spacing
    tg = np.stack([rng.uniform(-30, 30, 40), rng.uniform(-30, 30, 40), np.zeros(40)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(40) + 1j * rng.standard_normal(40)).astype(np.complex64)
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    grid = (128, 96, 0.48, 0.48, np.cross(e2, n), e2)
    ex = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, *grid, backend='cpu', window=True)(S)
    e = rel_db(quick._pfa(S, col.ant, col.fmin, col.df, *grid, window=True), ex)
    print(f'uneven pulses: {e:.1f} dB')
    assert e < -30


def test_corner_error_is_polar_formats_own():
    """Fine resolution over a picture 880 m wide at 20 km: polar format defocuses its ends, by the phase its planar
    wavefronts leave beyond the displacement. The estimate the product carries (corner_error) is the error measured
    against exact backprojection on patches at the corners, and it is small on a picture polar format holds."""
    rng = np.random.default_rng(4)
    r0, wide, deep, res, fc, psi = 20e3, 880.0, 100.0, 0.3, 9.6e9, np.radians(30.0)
    df = C / (2 * 1.6 * deep)
    K = int(C / (2 * res) / df) // 2 * 2
    dy = (C / fc) * r0 / (2 * 1.6 * wide)
    P = int(r0 * (C / fc) / (2 * res) / dy) // 2 * 2
    ant = np.stack([np.full(P, -r0 * np.cos(psi)), (np.arange(P) - (P - 1) / 2.0) * dy, np.full(P, r0 * np.sin(psi))], 1)
    col = sim.Collect(fmin=fc - (K // 2) * df, df=df, K=K, ant=ant, res=res)
    a = ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, ant[P // 2 + 1] - ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    e1 = np.cross(e2, n)
    sp, h = 0.28, 32
    nx, ny = int(wide / sp) // 2 * 2, int(deep / sp) // 2 * 2
    grid = (nx, ny, sp, sp, e1, e2)
    phase, db = quick._corner_error(ant, col.fmin, df, K, *grid)
    assert phase > 0.8 and -16 < db < -12
    cens = [((i + h - nx / 2) * sp, (j + h - ny / 2) * sp) for i in (0, nx - 2 * h) for j in (0, ny - 2 * h)]
    tg = np.concatenate([(np.array([cx, cy]) + rng.uniform(-6, 6, (6, 2))) @ np.stack([e1, e2]) for cx, cy in cens])
    amp = rng.standard_normal(len(tg)) + 1j * rng.standard_normal(len(tg))
    S = sim.simulate_brute(col, tg, amp).astype(np.complex64)
    img = quick._pfa(S, ant, col.fmin, df, *grid, window=True)
    worst = -200.0
    for cx, cy in cens:
        ex = fastsar.ExactFormer(ant, col.fmin, df, K, 2 * h, 2 * h, sp, sp, e1, e2, backend='cpu', window=True,
                                 center=cx * e1 + cy * e2)(S)
        i, j = int(round(cx / sp + nx / 2 - h)), int(round(cy / sp + ny / 2 - h))
        worst = max(worst, rel_db(img[i:i + 2 * h, j:j + 2 * h], ex))
    print(f'corner error: estimated {db:.1f} dB ({phase:.2f} rad), measured {worst:.1f} dB')
    assert abs(worst - db) < 1.5
    assert quick._corner_error(ant, col.fmin, df, K, 256, 256, sp, sp, e1, e2)[1] < -50


def test_wide_band_keeps_the_far_edge_clean():
    """Over a wide band the bins of the azimuth gate's transform differ from one range wavenumber to the next (5%
    here). A strong point just beyond the gate must be rejected at every wavenumber: where the gate is set in bins,
    the low end of the band passes it and it wraps into the picture's far edge."""
    rng = np.random.default_rng(9)
    col = sim.make_collect(res=0.3, scene=220.0, r0=30e3)
    P = col.Np
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    e1 = np.cross(e2, n)
    grid = (256, 256, 0.28, 0.28, e1, e2)
    g = quick._geometry(col.ant, col.fmin, col.df, col.K, *grid, True)
    assert g['bins_a'].max() / g['bins_a'].min() > 1.04
    out = (g['Ex'] + 1.02 * g['gx']) * e1                        # just past the gate's zero, on the +x side
    tg = np.concatenate([np.stack([rng.uniform(-30, 30, 30), rng.uniform(-30, 30, 30), np.zeros(30)], 1) @ np.stack([e1, e2, n]),
                         out[None]])
    amp = np.concatenate([rng.standard_normal(30) + 1j * rng.standard_normal(30), [40.0]])
    S = sim.simulate_brute(col, tg, amp).astype(np.complex64)
    ex = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, *grid, backend='cpu', window=True)(S)
    img = quick._pfa(S, col.ant, col.fmin, col.df, *grid, window=True)
    e_far, e_all = rel_db(img[:48], ex[:48]), rel_db(img, ex)
    print(f'wide band: far edge {e_far:.1f} dB, whole {e_all:.1f} dB')
    assert e_far < -20 and e_all < -25


# ------------------------------------------------------------------------------------------------------ files

def test_png_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    for a in (rng.integers(0, 256, (37, 53), dtype=np.uint8), rng.integers(0, 256, (1300, 1700), dtype=np.uint8),
              np.zeros((5, 1), np.uint8), rng.integers(0, 256, (20, 30, 3), dtype=np.uint8),
              rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)):
        p = str(tmp_path / 'a.png')
        quick.write_png(p, a)
        assert np.array_equal(quick.read_png(p), a)
        with open(p, 'rb') as fh:                                              # valid chunks, one valid zlib stream
            b = fh.read()
        assert b[:8] == b'\x89PNG\r\n\x1a\n' and b[-12:-8] == b'\x00\x00\x00\x00' and b[-8:-4] == b'IEND'
        at, data = 33, b''
        while b[at + 4:at + 8] == b'IDAT':
            n = int.from_bytes(b[at:at + 4], 'big')
            assert zlib.crc32(b[at + 4:at + 8 + n]) == int.from_bytes(b[at + 8 + n:at + 12 + n], 'big')
            data, at = data + b[at + 8:at + 8 + n], at + 12 + n
        assert at == len(b) - 12 and len(zlib.decompress(data)) == a.shape[0] * (1 + a[0].size)
    for bad in (np.zeros((4, 4), np.float32), np.zeros((4,), np.uint8), np.zeros((4, 4, 2), np.uint8)):
        with pytest.raises(ValueError, match='uint8'):
            quick.write_png(str(tmp_path / 'b.png'), bad)
    with open(tmp_path / 'c.png', 'wb') as fh:
        fh.write(b'not a png')
    with pytest.raises(ValueError, match='not a PNG'):
        quick.read_png(str(tmp_path / 'c.png'))


def test_png_read_by_pillow(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    a = np.random.default_rng(1).integers(0, 256, (1100, 1500), dtype=np.uint8)
    quick.write_png(str(tmp_path / 'a.png'), a)
    assert np.array_equal(np.asarray(Image.open(tmp_path / 'a.png')), a)
    Image.fromarray(a[:64, :48]).save(tmp_path / 'b.png')                      # another writer's filters
    assert np.array_equal(quick.read_png(str(tmp_path / 'b.png')), a[:64, :48])


def test_png_blocks_join_into_one_stream(tmp_path):
    """The blocks, deflated apart on all cores, make one zlib stream in several IDAT chunks that another decoder
    reads (Pillow where it is installed, else zlib and the filter bytes), at the levels 0 (stored), 1 (run lengths
    and Huffman codes) and 6 (zlib's search), for speckle with a black border."""
    rng = np.random.default_rng(2)
    a = np.minimum(40 * rng.rayleigh(1.0, (1900, 2300)), 255).astype(np.uint8)
    a[:300] = 0
    a[:, -200:] = 0
    sizes = {}
    for level in (0, 1, 6):
        p = str(tmp_path / f'level{level}.png')
        quick.write_png(p, a, level=level)
        with open(p, 'rb') as fh:
            b = fh.read()
        sizes[level] = len(b)
        at, data, chunks = 33, [], 0
        while b[at + 4:at + 8] == b'IDAT':
            n = int.from_bytes(b[at:at + 4], 'big')
            assert zlib.crc32(b[at + 4:at + 8 + n]) == int.from_bytes(b[at + 8 + n:at + 12 + n], 'big')
            data.append(b[at + 8:at + 8 + n])
            at, chunks = at + 12 + n, chunks + 1
        assert chunks > 4 and b[at:] == b'\x00\x00\x00\x00IEND\xaeB`\x82'
        try:
            from PIL import Image
            got = np.asarray(Image.open(p))
        except ImportError:
            raw = np.frombuffer(zlib.decompress(b''.join(data)), np.uint8).reshape(a.shape[0], 1 + a.shape[1])
            assert not raw[:, 0].any()                                         # filter type 0 on every row
            got = raw[:, 1:]
        assert np.array_equal(got, a) and np.array_equal(quick.read_png(p), a)
    assert sizes[1] < 0.9 * sizes[0] and sizes[1] < 1.02 * sizes[6]            # as small as the search makes it
    with pytest.raises(ValueError, match='level'):
        quick.write_png(str(tmp_path / 'bad.png'), a, level=12)


def test_display_passes_on_all_cores(spot, monkeypatch):
    """The host's display passes by the compiled kernels (the histograms of the remap's order statistics, the
    linear part of nrl, the layout) give NumPy's values, value for value."""
    from fastsar import remap
    if remap._host_lib() is None:
        pytest.skip('needs the compiled kernels')
    rng = np.random.default_rng(4)
    x = rng.rayleigh(1.0, (301, 517)).astype(np.float32)
    x[::37, ::41] *= 200
    x[5, 7], x[9, 3] = np.nan, 0.0
    z = (x * np.exp(1j * rng.uniform(0, 6.3, x.shape))).astype(np.complex64)
    mask = rng.random(x.shape) > 0.2
    lut = np.full(1 << 16, -1, np.int32)
    lut[np.argsort(remap._hist(x, None, None))[-3:]] = [0, 1, 2]
    cases = [(x, None, None, True), (x, mask, None, True), (x, None, lut, True), (x, mask, lut, True), (-x, None, None, False),
             (-x, mask, lut, False), (z, None, None, True), (z, mask, lut, True), (x[:, ::2], mask[:, ::2], None, True),
             (x.astype(np.float64), None, lut, True)]
    got = [remap._hist(*c) for c in cases]
    maps = [remap.nrl(x[:, :400]), remap.nrl(x[10:], valid=mask[10:]), remap.nrl(x, a=0.2), remap.nrl(np.full((4, 5), 2.0, np.float32)),
            remap.nrl(z), remap.density(x)]
    look_c = look(spot['path'], size=128)['display']
    monkeypatch.setattr(remap, '_HOST', [None])                                # NumPy's steps
    for c, h in zip(cases, got):
        assert h.dtype == np.int64 and np.array_equal(h, remap._hist(*c)), c[1:]
    ref = [remap.nrl(x[:, :400]), remap.nrl(x[10:], valid=mask[10:]), remap.nrl(x, a=0.2), remap.nrl(np.full((4, 5), 2.0, np.float32)),
           remap.nrl(z), remap.density(x)]
    assert all(m.dtype == np.uint8 and np.array_equal(m, r) for m, r in zip(maps, ref))
    monkeypatch.setattr(quick, '_oriented', lambda d, t, fr, fc: np.ascontiguousarray((d.T if t else d)[::-1 if fr else 1, ::-1 if fc else 1]))
    assert np.array_equal(look(spot['path'], size=128)['display'], look_c)
    monkeypatch.undo()
    d = rng.integers(0, 256, (133, 517), dtype=np.uint8)
    for turn in (False, True):
        for fr in (False, True):
            for fc in (False, True):
                o = quick._oriented(d, turn, fr, fc)
                assert o.flags.c_contiguous and np.array_equal(o, (d.T if turn else d)[::-1 if fr else 1, ::-1 if fc else 1])
    assert quick._oriented(d[:, ::2], True, True, False).shape == (259, 133)   # a view: NumPy's copy


def test_output_files(spot, tmp_path):
    """.png: the display; .nitf: the complex image as a ground-plane SICD that the library's reader takes."""
    png = str(tmp_path / 'look.png')
    out = look(spot['path'], png, size=128)
    assert out['path'] == png and np.array_equal(quick.read_png(png), out['display']) and 'write' in out['timing']
    ntf = str(tmp_path / 'look.nitf')
    out = look(spot['path'], ntf, size=128)
    pix, sm = io.read_sicd(ntf, native=True)
    assert out['image'] is None and out['metadata'].Grid.ImagePlane == 'GROUND'
    assert pix.shape == (128, 128) and sm.Grid.ImagePlane == 'GROUND' and sm.Grid.Type == 'PLANE'
    assert sm.Grid.Row.SS == pytest.approx(out['spy']) and sm.Grid.Col.SS == pytest.approx(out['spx'])
    # the SICD's rows run along range as the display's do: the same picture, detected
    full = look(spot['path'], size=128, complex_image=True)
    lay = full['transform']
    a = np.abs(full['image']).T
    a = a[::-1] if lay['flip_rows'] else a
    a = a[:, ::-1] if lay['flip_cols'] else a
    assert np.abs(np.abs(pix) - a).max() < 1e-5 * a.max()
    assert set(out['timing']) >= {'plan', 'read', 'reference', 'geometry', 'range', 'azimuth', 'range transform',
                                  'azimuth transform', 'correction', 'display', 'write', 'total'}
    assert out['timing']['total'] >= sum(v for k, v in out['timing'].items() if k != 'total') * 0.5


# ---------------------------------------------------------------------------------------------------- refusals

def test_moving_beam_is_refused(spot, tmp_path):
    """A collection whose scene reference point moves is refused with the name of the function that forms it."""
    col = spot['col']
    srp = np.zeros((col.Np, 3))
    srp[:, 1] = np.linspace(-40.0, 40.0, col.Np)                               # the beam slides along the track
    path = str(tmp_path / 'moving.cphd')
    sim.write_cphd(path, col, spot['S'], LAT, LON, srp=srp, scene=150.0)
    with pytest.raises(ValueError, match=r'moving beam.*process_cphd'):
        look(path)
    assert not os.path.exists(tmp_path / 'x.png')


def test_arguments_are_checked_before_the_file_is_read(spot, tmp_path, monkeypatch):
    monkeypatch.setattr(io, '_read_cphd', lambda *a, **k: pytest.fail('the samples were read'))
    for kw, match in ((dict(size=8), 'size'), (dict(size=2.5), 'size'), (dict(resolution=-1.0), 'resolution'),
                      (dict(resolution='fine'), 'resolution'), (dict(weighting='hann'), 'weighting'),
                      (dict(remap='linear'), 'remap'), (dict(remap='nope'), 'remap'), (dict(extent=(10.0,)), 'extent'),
                      (dict(extent=(10.0, -1.0)), 'extent'), (dict(backend='gpu'), 'backend'),
                      (dict(output=str(tmp_path / 'x.tif')), r'\.png'), (dict(reader_options=dict(sicd='x')), 'reader_options')):
        with pytest.raises(ValueError, match=match):
            look(spot['path'], **kw)
    with pytest.raises(OSError):
        look(str(tmp_path / 'absent.cphd'))


def test_too_few_pulses(spot):
    with pytest.raises(ValueError, match='pulses'):
        look(spot['path'], resolution=40.0)


# ----------------------------------------------------------------------------------------------------- devices

def test_remap_and_function(spot):
    a = look(spot['path'], size=128, remap='density')['display']
    b = look(spot['path'], size=128)['display']
    assert a.shape == b.shape and not np.array_equal(a, b)
    c = look(spot['path'], size=128, remap=lambda x, valid=None: np.full(x.shape, 7, np.uint8))['display']
    assert (c == 7).all()


def test_looks(tmp_path):
    """looks: the mean power of images of the same pixels from sub-apertures and sub-bands side by side. On speckle
    the contrast of the power (standard deviation over mean) falls as the square root of their count, the mean
    stays, and no more looks are taken than the collection holds."""
    rng = np.random.default_rng(5)
    col = sim.make_collect(res=0.3, scene=200.0, r0=30e3, graze_deg=35.0)
    S = (rng.standard_normal((col.Np, col.K)) + 1j * rng.standard_normal((col.Np, col.K))).astype(np.complex64)
    path = str(tmp_path / 'speckle.cphd')
    sim.write_cphd(path, col, S, LAT, LON, scene=150.0)
    power, mean = {}, {}
    for looks in (1, 2, (3, 1), (1, 3), 8):
        got = []
        out = look(path, size=128, looks=looks, remap=lambda x, valid=None: (got.append(np.array(x)), np.zeros(x.shape, np.uint8))[1])
        la, lr = out['looks']
        pw = got[0].astype(np.float64) ** 2
        power[looks], mean[looks] = pw.std() / pw.mean(), pw.mean()
        print(f'looks {looks}: {la} by {lr}, contrast of the power {power[looks]:.3f}, expected {1 / np.sqrt(la * lr):.3f}')
        assert power[looks] == pytest.approx(1 / np.sqrt(la * lr), rel=0.05)
        assert mean[looks] == pytest.approx(mean[1], rel=0.03) and out['display'].shape == (128, 128)
        one = look(path, size=128)['cut']
        n = lambda c, k: c[k][1] - c[k][0]
        assert n(out['cut'], 'pulses') == pytest.approx(la * n(one, 'pulses'), abs=2 * la)
        assert n(out['cut'], 'samples') == lr * n(one, 'samples')
        assert (looks == 8) == any('holds no more' in v for v in out['notes'])
    assert out['looks'] == (4, 3)                                               # 8 by 8 asked for
    for bad in (0, (2, 0), 'two', (1, 2, 3)):
        with pytest.raises(ValueError, match='looks'):
            look(path, looks=bad)
    with pytest.raises(ValueError, match='one look'):
        look(path, looks=2, complex_image=True)


def _cut(f, size=128, drop=()):
    """The cut history of a quick look of a fixture's file, on its grid; drop: pulses left out of it."""
    col = quick._collection(f['path'], size=size)
    S, ant = col['S'], col['ant']
    if len(drop):
        keep = np.ones(len(ant), bool)
        keep[list(drop)] = False
        S, ant = np.ascontiguousarray(S[keep]), ant[keep]
    return S, ant, col['fmin'], col['df'], tuple(col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2'))


def _device_against_host(S, ant, f0, df, grid, backend='jax'):
    """quick._pfa on a JAX device against the C++ stages -> the errors (dB) of the four images: corrected and
    uncorrected, complex and detected, and the two corrected complex images."""
    err = {}
    for correct in (True, False):
        for cplx in (True, False):
            h = quick._pfa(S, ant, f0, df, *grid, correct=correct, complex_image=cplx)
            j = quick._pfa(S, ant, f0, df, *grid, correct=correct, complex_image=cplx, backend=backend)
            assert isinstance(j, np.ndarray) and j.dtype == h.dtype and j.shape == h.shape
            err[correct, cplx] = rel_db(j.astype(np.complex64), h.astype(np.complex64))
            if correct and cplx:
                pair = (j, h)
    return err, pair


@pytest.fixture(scope='module')
def steep(tmp_path_factory):
    """A 0.5 m spotlight at 72 degrees of grazing whose track passes 26 km to the side of broadside: on the ground
    the bands of the pulses together span 2.4 times one pulse's, more than the padded transform's period holds."""
    col = sim.make_collect(res=0.5, scene=160.0, r0=30e3, graze_deg=72.0, offset=(0.0, 26e3, 0.0))
    S = sim.simulate_brute(col, TARGETS, np.ones(len(TARGETS)) + 0j)
    rng = np.random.default_rng(6)
    S = (S + 0.02 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))).astype(np.complex64)
    path = str(tmp_path_factory.mktemp('steep') / 'steep.cphd')
    sim.write_cphd(path, col, S, LAT, LON, scene=120.0)
    return dict(path=path, col=col, S=S)


def test_device_stages_on_a_plain_raster(spot, request):
    """backend='jax': polar format's stages as a TPU runs them (window reads and products of blocks with their
    weights, float32 planes, no indexed load of a sample), on JAX's CPU device. On a plain raster the two passes of
    the read at the apparent positions take the C++ kernel's own taps, and the four images are the C++ stages'."""
    request.getfixturevalue('jax_cpu')
    from fastsar import quick_jax
    S, ant, f0, df, grid = _cut(spot)
    g = quick._geometry(ant, f0, df, S.shape[1], *grid, True)
    assert g['jlo'] is None and g['place'] is None and quick_jax._layout(g, 1 << 30)['fold'] == 1
    err, _ = _device_against_host(S, ant, f0, df, grid)
    print('plain raster, device against C++:', {k: round(v, 1) for k, v in err.items()})
    assert max(err.values()) < -70
    # an odd pixel count, and no window
    odd = (grid[0] - 1, grid[1] - 3) + grid[2:]
    assert rel_db(quick._pfa(S, ant, f0, df, *odd, backend='jax'), quick._pfa(S, ant, f0, df, *odd)) < -70
    assert rel_db(quick._pfa(S, ant, f0, df, *grid, window=False, backend='jax'), quick._pfa(S, ant, f0, df, *grid, window=False)) < -65


def test_device_stages_on_a_sheared_raster(squinted, request):
    """A picture on the ground under a twisted slant plane: each pulse is read on its own rows of the raster, and
    the raster and the uncorrected images are the C++ stages'. The corrected image is read from the baseband image
    shifted along its shear, whose samples are not the ones the C++ kernel takes its taps from: the two differ by
    the C++ kernel's error (6 taps), and the device's image, read with 10 taps, is the closer to exact
    backprojection."""
    request.getfixturevalue('jax_cpu')
    S, ant, f0, df, grid = _cut(squinted)
    g = quick._geometry(ant, f0, df, S.shape[1], *grid, True)
    assert g['jlo'] is not None and abs(g['shear']) > 0.3
    err, (dev, host) = _device_against_host(S, ant, f0, df, grid)
    ex = fastsar.ExactFormer(ant, f0, df, S.shape[1], *grid, backend='cpu', window=True)(S)
    e_dev, e_host = rel_db(dev, ex), rel_db(host, ex)
    print('sheared raster, device against C++:', {k: round(v, 1) for k, v in err.items()},
          f'; against exact backprojection: device {e_dev:.1f} dB, C++ {e_host:.1f} dB')
    assert err[False, True] < -70 and err[False, False] < -70 and err[True, True] < -45 and err[True, False] < -45
    assert e_dev < e_host + 0.2 and e_dev < -40


def test_device_stages_on_a_folded_raster_with_pulses_missing(steep, request):
    """A twist steep enough that the raster's rows exceed the padded transform's period and fold into it, from a
    history that lacks pulses: each pulse at its place, the fold a reshape and a sum."""
    request.getfixturevalue('jax_cpu')
    from fastsar import quick_jax
    S, ant, f0, df, grid = _cut(steep, 160, drop=(20, 21, 90, 200))
    g = quick._geometry(ant, f0, df, S.shape[1], *grid, True)
    assert g['place'] is not None and g['nkr'] > g['ov'] * g['nfy'] and quick_jax._layout(g, 1 << 30)['fold'] == 2
    err, (dev, host) = _device_against_host(S, ant, f0, df, grid)
    ex = fastsar.ExactFormer(ant, f0, df, S.shape[1], *grid, backend='cpu', window=True)(S)
    e_dev, e_host = rel_db(dev, ex), rel_db(host, ex)
    print('folded raster with gaps, device against C++:', {k: round(v, 1) for k, v in err.items()},
          f'; against exact backprojection: device {e_dev:.1f} dB, C++ {e_host:.1f} dB')
    assert err[False, True] < -70 and err[False, False] < -70 and err[True, True] < -45 and e_dev < e_host + 0.2


def test_device_stages_with_missing_and_uneven_pulses(request):
    """Pulses missing from a plain raster's history, and pulses unevenly spaced along the track (the positions of
    the azimuth read then curve within a block of outputs, and are sent as parabolas over fewer of them)."""
    request.getfixturevalue('jax_cpu')
    from fastsar import quick_jax
    rng = np.random.default_rng(8)
    col = sim.make_collect(res=0.5, scene=260.0, r0=60e3)
    keep = np.ones(col.Np, bool)
    keep[[40, 41, 300, 700, 702]] = False
    P = col.Np
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    e1 = np.cross(e2, n)
    grid = (480, 200, 0.48, 0.48, e1, e2)
    tg = np.concatenate([rng.uniform(-110, -100, (15, 1)) * e1 + rng.uniform(-40, 40, (15, 1)) * e2,
                         rng.uniform(100, 110, (15, 1)) * e1 + rng.uniform(-40, 40, (15, 1)) * e2])
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    S, ant = np.ascontiguousarray(S[keep]), col.ant[keep]
    assert quick._geometry(ant, col.fmin, col.df, col.K, *grid, True)['place'] is not None
    err, _ = _device_against_host(S, ant, col.fmin, col.df, grid)
    print('pulses missing, device against C++:', {k: round(v, 1) for k, v in err.items()})
    assert max(err.values()) < -65
    col = sim.make_collect(res=0.5, scene=90.0, r0=20e3)
    P = col.Np
    y = col.ant[:, 1]
    col.ant[:, 1] = y + 0.02 * (y[-1] - y[0]) * np.sin(np.pi * np.arange(P) / (P - 1)) ** 2      # up to 6% in spacing
    tg = np.stack([rng.uniform(-30, 30, 40), rng.uniform(-30, 30, 40), np.zeros(40)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(40) + 1j * rng.standard_normal(40)).astype(np.complex64)
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    grid = (128, 96, 0.48, 0.48, np.cross(e2, n), e2)
    g = quick._geometry(col.ant, col.fmin, col.df, col.K, *grid, True)
    assert quick_jax._layout(g, 1 << 30)['pb2'] < quick_jax.BLOCK
    err, _ = _device_against_host(S, col.ant, col.fmin, col.df, grid)
    print('uneven pulses, device against C++:', {k: round(v, 1) for k, v in err.items()})
    assert max(err.values()) < -65


@pytest.mark.parametrize('megabytes', ['0.5', '1.5'])
def test_device_streams_what_does_not_fit(spot, steep, request, monkeypatch, megabytes):
    """A JAX device with less memory to give than the arrays take (FASTSAR_JAX_BLOCK_MB): the stages work in
    smaller blocks, the arrays that do not fit live on the host and their blocks cross, a note names them, and the
    pictures are those of a device with room."""
    request.getfixturevalue('jax_cpu')
    from fastsar import quick_jax
    cases = [(_cut(spot), ()), (_cut(steep, 160, drop=(20, 21, 90)), ())]
    full = [{c: quick._pfa(*k[:4], *k[4], correct=c, backend='jax') for c in (True, False)} for k, _ in cases]
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', megabytes)
    for (k, _), ref in zip(cases, full):
        S, ant, f0, df, grid = k
        g = quick._geometry(ant, f0, df, S.shape[1], *grid, True)
        where = quick_jax.plan(g)
        assert where['kept'] and where['budget'] == int(float(megabytes) * (1 << 20))
        for correct in (True, False):
            info = {}
            img = quick._pfa(S, ant, f0, df, *grid, correct=correct, backend='jax', info=info)
            e = rel_db(img, ref[correct])
            print(f'{megabytes} MB of device memory, correct {correct}: {e:.1f} dB against the device with room; {info["stream"]}')
            assert e < -90 and 'kept on the host' in info['stream'] and where['kept'][0] in info['stream']
    out = look(spot['path'], size=128, backend='jax', complex_image=True)
    assert any('kept on the host and cross' in n for n in out['notes'])
    monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB')
    assert quick_jax.plan(g)['kept'] == [] and not any('kept on the host' in n for n in look(spot['path'], size=128, backend='jax')['notes'])
    sizes = dict(A=4 << 30, B=2 << 30, F=5 << 30)
    assert quick_jax._plan(sizes, None, 1 << 30)['kept'] == [] and quick_jax._plan(sizes, 16 << 30, 2 << 30)['kept'] == []
    assert quick_jax._plan(sizes, 15 << 30, 2 << 30)['kept'] == ["the raster's range transform"]
    assert quick_jax._plan(sizes, 13 << 30, 2 << 30)['kept'] == ["the range stage's output", "the raster's range transform"]
    assert len(quick_jax._plan(sizes, 1 << 30, 1 << 28)['kept']) == 3


def test_device_programs_are_shared(spot, squinted, request, monkeypatch):
    """The programs that take long to compile (the transforms and the reads) are compiled for block sizes of a
    short ladder and for a TPU's few transform lengths, so a second collection of nearly the same size compiles
    none of them, only the cuts and layouts that hold its own sizes, and a formation repeated compiles nothing.
    FASTSAR_JAX_PROGRAMS=fused, one program a stage, forms the same image. With a TPU's transform lengths (a power
    of two times an odd part up to 243) the device's images are the C++ stages' at the same lengths."""
    request.getfixturevalue('jax_cpu')
    from fastsar import quick_jax
    monkeypatch.setattr(quick, '_fft_size', quick._fft_tpu)
    assert [quick._fft_tpu(n) for n in (2, 1000, 1025, 29160, 30000, 31105)] == [2, 1000, 1080, 30720, 30720, 32000]
    for f in (spot, squinted):
        S, ant, f0, df, grid = _cut(f)
        e = rel_db(quick._pfa(S, ant, f0, df, *grid, backend='jax'), quick._pfa(S, ant, f0, df, *grid))
        print(f"a TPU's transform lengths, device against C++: {e:.1f} dB")
        assert e < (-45 if f is squinted else -70)
    S, ant, f0, df, grid = _cut(spot)
    ref = quick._pfa(S, ant, f0, df, *grid, backend='jax')
    n = len(quick_jax.COMPILED)
    quick._pfa(S, ant, f0, df, *grid, backend='jax')
    assert len(quick_jax.COMPILED) == n
    quick._pfa(S[:-2], ant[:-2], f0, df, *grid, backend='jax')                   # two pulses fewer
    new = [c[0] for c in quick_jax.COMPILED[n:]]
    print('compiled for the second collection:', new)
    assert new and not {'range', 'azimuth', 'inverse transform', 'transform', 'to rows', 'to columns'} & set(new)
    assert quick_jax._ladder(1000) == 1024 and quick_jax._ladder(1025) == 1088 and quick_jax._ladder(20) == 20
    assert quick_jax._ladder_down(1087) == 1024 and all(quick_jax._ladder(v) >= v for v in range(1, 5000, 7))
    monkeypatch.setenv('FASTSAR_JAX_PROGRAMS', 'fused')
    assert rel_db(quick._pfa(S, ant, f0, df, *grid, backend='jax'), ref) < -100
    monkeypatch.delenv('FASTSAR_JAX_PROGRAMS')
    monkeypatch.setattr(quick_jax, 'AT_ONCE', 0)              # every transform in a loop over chunks of rows
    for c in (True, False):
        e = rel_db(quick._pfa(S, ant, f0, df, *grid, correct=c, backend='jax'), quick._pfa(S, ant, f0, df, *grid, correct=c))
        print(f'transforms in chunks of rows, corrected {c}: {e:.1f} dB against C++')
        assert e < -70
    monkeypatch.setenv('FASTSAR_JAX_PROGRAMS', 'both')
    with pytest.raises(ValueError, match='FASTSAR_JAX_PROGRAMS'):
        quick._pfa(S, ant, f0, df, *grid, backend='jax')


_CACHE_SCRIPT = '''
import json, sys
import jax
hits = []
jax.monitoring.register_event_listener(lambda name, **kw: hits.append(name))
import fastsar
from fastsar import _xp, quick_jax
if sys.argv[2] == 'form':
    fastsar.quicklook(sys.argv[1], size=96, backend='jax')
print(json.dumps(dict(cache=_xp.jax_cache(), jax=jax.config.jax_compilation_cache_dir, compiled=len(quick_jax.COMPILED),
                      hits=hits.count('/jax/compilation_cache/cache_hits'))))
'''


def test_device_programs_persist_between_processes(spot, tmp_path):
    """JAX's persistent compilation cache: by default a directory beside the compiled CPU kernels, or
    FASTSAR_JAX_CACHE; a second process loads every program of the polar format's stages from it, the short ones
    too; a cache the user configured is kept, and FASTSAR_JAX_CACHE=off leaves JAX's settings alone."""
    import json
    import subprocess
    import sys
    script = tmp_path / 'cache_probe.py'
    script.write_text(_CACHE_SCRIPT)
    base = {k: v for k, v in os.environ.items() if not k.startswith(('JAX_COMPILATION_CACHE', 'FASTSAR_JAX_CACHE', 'FASTSAR_CACHE_DIR'))}
    base.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')

    def run(what, **env):
        out = subprocess.run([sys.executable, str(script), spot['path'], what], env=dict(base, **env), capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-3000:]
        return json.loads(out.stdout.strip().splitlines()[-1])
    kernels = tmp_path / 'kernels'
    d = run('none', FASTSAR_CACHE_DIR=str(kernels))
    assert d['cache'] == d['jax'] == str(kernels / 'jax')
    mine = tmp_path / 'mine'
    first = run('form', FASTSAR_JAX_CACHE=str(mine))
    files = sorted(p.name for p in mine.iterdir())
    assert first['cache'] == str(mine) and first['hits'] == 0 and first['compiled'] > 5 and len(files) >= first['compiled']
    second = run('form', FASTSAR_JAX_CACHE=str(mine))
    print('programs of a quick look:', first['compiled'], 'compiled and written,', second['hits'], 'loaded by a second process')
    assert second['compiled'] == first['compiled'] and second['hits'] >= first['compiled'] and sorted(p.name for p in mine.iterdir()) == files
    users = tmp_path / 'users'
    d = run('none', JAX_COMPILATION_CACHE_DIR=str(users), FASTSAR_JAX_CACHE=str(mine))
    assert d['cache'] == d['jax'] == str(users)
    d = run('none', FASTSAR_JAX_CACHE='off', FASTSAR_CACHE_DIR=str(kernels))
    assert d['cache'] is None and d['jax'] is None


def test_quicklook_on_jax(spot, squinted, request, monkeypatch):
    """quicklook(backend='jax'): the cut referenced to the grid center on the device (a phase in 32-bit integer
    arithmetic), the picture and the complex image of the cpu backend, on a plain and on a sheared raster; looks;
    and the picture remapped on the device (FASTSAR_QUICKLOOK_DISPLAY=device), where only its 8 bits come down."""
    request.getfixturevalue('jax_cpu')
    for f, limit in ((spot, -70), (squinted, -45)):
        ref = look(f['path'], size=128, complex_image=True)
        out = look(f['path'], size=128, backend='jax', complex_image=True)
        assert out['algorithm'] == 'pfa' and out['backend'] == 'jax' and out['cut'] == ref['cut'] and out['correct']
        e = rel_db(out['image'], ref['image'])
        print(f'quicklook on jax against cpu: {e:.1f} dB')
        assert e < limit and out['image'].flags.c_contiguous and out['display'].shape == ref['display'].shape
        assert np.mean(np.abs(out['display'].astype(int) - ref['display'].astype(int)) <= (1 if f is spot else 2)) > (0.999 if f is spot else 0.98)
        raw = look(f['path'], size=128, backend='jax', correct=False)
        assert np.mean(np.abs(raw['display'].astype(int) - look(f['path'], size=128, correct=False)['display'].astype(int)) <= 1) > 0.999
    lk = look(spot['path'], size=64, backend='jax', looks=2)
    assert lk['looks'] == (2, 2) and np.mean(np.abs(lk['display'].astype(int) - look(spot['path'], size=64, looks=2)['display'].astype(int)) <= 1) > 0.99
    host = {c: look(squinted['path'], size=128, backend='jax', correct=c)['display'] for c in (True, False)}
    assert quick._jax_display() == quick.JAX_DISPLAY
    monkeypatch.setenv('FASTSAR_QUICKLOOK_DISPLAY', 'device')
    for c in (True, False):
        dev = look(squinted['path'], size=128, backend='jax', correct=c)
        assert isinstance(dev['display'], np.ndarray) and np.abs(dev['display'].astype(int) - host[c].astype(int)).max() <= 1
    two = look(spot['path'], size=64, backend='jax', looks=2)
    assert np.abs(two['display'].astype(int) - lk['display'].astype(int)).max() <= 1
    monkeypatch.setenv('FASTSAR_QUICKLOOK_DISPLAY', 'gpu')
    with pytest.raises(ValueError, match='FASTSAR_QUICKLOOK_DISPLAY'):
        look(spot['path'], size=64, backend='jax')


def test_factorized_former_on_jax(spot, request, monkeypatch):
    """FASTSAR_QUICKLOOK_JAX=ffbp: the same cut history on the same grid by the factorized former, with a note; the
    picture agrees with polar format's."""
    request.getfixturevalue('jax_cpu')
    monkeypatch.setenv('FASTSAR_QUICKLOOK_JAX', 'ffbp')
    ref = look(spot['path'], size=128, complex_image=True)
    out = look(spot['path'], size=128, backend='jax', complex_image=True)
    assert out['algorithm'] == 'ffbp' and out['backend'] == 'jax' and out['displacement_px'] is None
    assert out['cut'] == ref['cut'] and out['display'].shape == ref['display'].shape
    assert any('factorized backprojection' in n for n in out['notes'])
    e = rel_db(out['image'], ref['image'])
    print(f'factorized former against polar format on the quick look grid: {e:.1f} dB')
    assert e < -25
    assert np.mean(np.abs(out['display'].astype(int) - ref['display'].astype(int)) <= 12) > 0.9
    rows, cols = _expected_pixels(out)
    for r, c in zip(rows, cols):
        assert out['display'][int(round(r)) - 2:int(round(r)) + 3, int(round(c)) - 2:int(round(c)) + 3].max() >= 250
    monkeypatch.setenv('FASTSAR_QUICKLOOK_JAX', 'gathers')
    with pytest.raises(ValueError, match='FASTSAR_QUICKLOOK_JAX'):
        look(spot['path'], size=128, backend='jax')


def test_auto_backend_by_the_size_of_the_cut(spot, monkeypatch):
    """'auto' takes the CPU for a small cut without asking for a device; from the thresholds on it takes a GPU where
    CuPy has one (a first look of a process from a larger size, which repays the start of the device), and it asks
    JAX for a TPU only where one may be present."""
    import importlib.util
    import sys
    import types
    from fastsar import api
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a: None if name == 'libtpu' else real(name, *a))
    monkeypatch.delenv('TPU_LIBRARY_PATH', raising=False)
    monkeypatch.setattr(api, 'available_backends', lambda: pytest.fail('the device probe ran'))
    asked = []
    rt = types.SimpleNamespace(getDeviceCount=lambda: asked.append(1) or 1)
    monkeypatch.setitem(sys.modules, 'cupy', types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=rt)))
    saved = sys.modules.pop('jax', None)
    try:
        monkeypatch.setattr(quick, '_CUDA', {'cols': None})                     # the device has formed a look here
        assert quick._auto(quick.AUTO_CUDA_SAMPLES - 1) == 'cpu' and not asked
        assert quick._auto(quick.AUTO_CUDA_SAMPLES) == 'cuda' and asked
        monkeypatch.setattr(quick, '_CUDA', {})                                 # a first look: the start is not repaid
        assert quick._auto(quick.AUTO_CUDA_FIRST_SAMPLES - 1) == 'cpu' and quick._auto(quick.AUTO_CUDA_FIRST_SAMPLES) == 'cuda'
        assert quick._auto(10 ** 12) == 'cuda'                                  # no TPU here: JAX is not asked
    finally:
        if saved is not None:
            sys.modules['jax'] = saved
    assert quick.AUTO_CUDA_SAMPLES <= quick.AUTO_CUDA_FIRST_SAMPLES
    out = fastsar.quicklook(spot['path'], size=64)                              # a few hundred thousand samples
    assert out['backend'] == 'cpu'
    with pytest.raises(ValueError):
        fastsar.quicklook(spot['path'], size=64, backend='gpu')


def test_auto_says_which_on_a_tpu_host(spot, monkeypatch):
    """On a machine that may have a TPU, 'auto' takes the device from AUTO_TPU_SAMPLES on and the host's cores
    below, and the product's notes say which and why."""
    from fastsar import api
    monkeypatch.setattr(quick, '_tpu_host', lambda: True)
    monkeypatch.setattr(api, 'available_backends', lambda: ['tpu', 'cpu'])
    monkeypatch.setattr(quick, 'AUTO_CUDA_SAMPLES', 10 ** 15)
    monkeypatch.setattr(quick, 'AUTO_CUDA_FIRST_SAMPLES', 10 ** 15)
    notes = []
    assert quick._auto(quick.AUTO_TPU_SAMPLES, notes) == 'tpu' and 'formed on the TPU' in notes[0] and 'AUTO_TPU_SAMPLES' in notes[0]
    notes = []
    assert quick._auto(quick.AUTO_TPU_SAMPLES - 1, notes) == 'cpu' and "formed on the host's cores" in notes[0]
    out = fastsar.quicklook(spot['path'], size=64)
    assert out['backend'] == 'cpu' and any("backend 'auto': formed on the host's cores" in n for n in out['notes'])
    monkeypatch.setattr(quick, '_tpu_host', lambda: False)
    notes = []
    assert quick._auto(10 ** 12, notes) == 'cpu' and notes == []              # no TPU here: nothing to say
    assert not any("backend 'auto'" in n for n in fastsar.quicklook(spot['path'], size=64)['notes'])


@pytest.mark.tpu
def test_tpu(spot, squinted, steep, monkeypatch):
    """On a TPU: quicklook(backend='tpu') forms the cpu backend's picture on a plain raster, on a sheared one and on
    a folded one, corrected and not; 'auto' picks the device from its threshold on and says so; the device's
    memory held small gives the same picture with a note; the factorized former (precision 'single-pass') stays
    selectable."""
    from fastsar import quick_jax
    for f, limit, size in ((spot, -60, 128), (squinted, -45, 128), (steep, -45, 160)):
        ref = look(f['path'], size=size, complex_image=True)
        out = look(f['path'], size=size, backend='tpu', complex_image=True)
        e = rel_db(out['image'], ref['image'])
        print(f'polar format on the TPU against the host: {e:.1f} dB')
        assert out['backend'] == 'tpu' and out['algorithm'] == 'pfa' and out['cut'] == ref['cut'] and e < limit
        assert np.mean(np.abs(out['display'].astype(int) - ref['display'].astype(int)) <= 2) > 0.98
        raw, raw_ref = look(f['path'], size=size, backend='tpu', correct=False, complex_image=True), look(f['path'], size=size, correct=False, complex_image=True)
        assert rel_db(raw['image'], raw_ref['image']) < -60
    S, ant, f0, df, grid = _cut(steep, 160, drop=(20, 21, 90, 200))
    assert rel_db(quick._pfa(S, ant, f0, df, *grid, correct=False, backend='tpu'), quick._pfa(S, ant, f0, df, *grid, correct=False)) < -60
    ref = look(spot['path'], size=128, complex_image=True)
    monkeypatch.setattr(quick, 'AUTO_TPU_SAMPLES', 0)
    out = fastsar.quicklook(spot['path'], size=128, complex_image=True)
    assert out['backend'] == 'tpu' and any("backend 'auto': formed on the TPU" in n for n in out['notes'])
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '1')
    small = look(spot['path'], size=128, backend='tpu', complex_image=True)
    assert rel_db(small['image'], out['image']) < -80 and any('kept on the host' in n for n in small['notes'])
    monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB')
    monkeypatch.setenv('FASTSAR_QUICKLOOK_JAX', 'ffbp')
    out = fastsar.quicklook(spot['path'], size=128, backend='tpu', complex_image=True)
    assert out['algorithm'] == 'ffbp' and any('single-pass' in n for n in out['notes'])
    assert rel_db(out['image'], ref['image']) < -20


@pytest.mark.tpu
def test_tpu_holds_to_its_budget(monkeypatch):
    """On a TPU held to less memory than a formation's arrays take (FASTSAR_JAX_BLOCK_MB), the device's peak stays
    within it: the arrays that do not fit are kept on the host and cross in blocks, a note says so, and the image is
    the C++ stages' and that of the device with room. (A TPU v5e, 16 GB, forms the 252-megapixel Capella image
    within its own budget, 15.2 GB, with every array on the device: a peak of 11.1 GB.)"""
    import math
    import jax
    from fastsar import _xp
    dev = jax.devices('tpu')[0]
    res, n = 0.5, 2048
    col = sim.make_collect(res=res, scene=0.96 * res * n / (1.25 * math.sqrt(2.0)) * 1.2, r0=600e3, graze_deg=45.0)
    P = col.Np
    e2 = -col.ant[P // 2] / np.linalg.norm(col.ant[P // 2])
    nrm = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    nrm *= np.sign(nrm[2]) / np.linalg.norm(nrm)
    S = np.random.default_rng(0).standard_normal((P, col.K, 2), np.float32).view(np.complex64)[..., 0]
    grid = (n, n, 0.96 * res, 0.96 * res, np.cross(e2, nrm), e2)
    form = lambda backend, info=None: quick._pfa(S.copy(), col.ant, col.fmin, col.df, *grid, backend=backend, info=info, layout='formed')
    ref = form('cpu')
    before = dev.memory_stats()
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '120')
    info = {}
    held, budget = form('tpu', info), _xp.jax_budget()
    peak = dev.memory_stats()['peak_bytes_in_use']
    monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB')
    free = form('tpu')
    print(f"held to {budget / 1e6:.0f} MB: peak {(peak - before['bytes_in_use']) / 1e6:.0f} MB beside what was in use, "
          f"{rel_db(held, ref):.1f} dB against C++; {info.get('stream')}")
    assert 'kept on the host' in info['stream'] and rel_db(held, ref) < -60 and rel_db(held, free) < -100
    if before['peak_bytes_in_use'] < budget:                  # an earlier test's peak does not hide this one's
        assert peak <= before['bytes_in_use'] + budget


@pytest.mark.cuda
@pytest.mark.parametrize('megabytes', ['0.4', '2'])
def test_cuda_streams_what_does_not_fit(spot, squinted, monkeypatch, megabytes):
    """A device with less memory to give than the arrays take: the stages work in smaller blocks and the arrays that
    do not fit stay on the host, their blocks crossing as the stages need them; the picture is the same and a note
    says what was kept on the host."""
    import cupy as cp
    full = {k: look(f['path'], size=256, backend='cuda', complex_image=True) for k, f in (('spot', spot), ('squinted', squinted))}
    monkeypatch.setenv('FASTSAR_CUDA_BLOCK_MB', megabytes)
    for k, f in (('spot', spot), ('squinted', squinted)):
        for correct in (True, False):
            out = look(f['path'], size=256, backend='cuda', correct=correct, complex_image=True)
            ref = full[k] if correct else None
            assert out['backend'] == 'cuda' and any('kept on the host' in n for n in out['notes'])
            if ref is not None:
                e = rel_db(out['image'], ref['image'])
                print(f'{k}, {megabytes} MB of device memory: {e:.1f} dB against the picture formed with room')
                assert e < -80 and np.array_equal(out['display'], ref['display'])
    col = quick._collection(spot['path'], size=256)
    g = quick._geometry(col['ant'], col['fmin'], col['df'], col['S'].shape[1], *(col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2')), True)
    assert quick._cuda_plan(g, True, False, 1 << 40, 1 << 20)['kept'] == []
    assert len(quick._cuda_plan(g, True, False, 1 << 10, 1 << 8)['kept']) == 4


@pytest.mark.cuda
def test_cuda_gives_the_device_back():
    """A second formation in a process plans with the memory the first one had. On a device that just holds the
    arrays (CuPy's pool limited to that, with transforms long enough for cuFFT to ask for work areas), both calls
    keep everything on the device and the pool holds no more after the second than after the first; work areas
    another caller left in CuPy's plan cache are released before an array would be kept on the host."""
    import cupy as cp
    rng = np.random.default_rng(11)
    col = sim.make_collect(res=0.3, scene=3600.0, r0=30e3)
    ant = np.ascontiguousarray(col.ant[::128])
    P, K = len(ant), col.K
    S = (rng.standard_normal((P, K), np.float32) + 1j * rng.standard_normal((P, K), np.float32)).astype(np.complex64)
    grid = (128, 1024, 0.3, 0.3, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    g = quick._geometry(ant, col.fmin, col.df, K, *grid, True)
    assert g['Kf'] > 20000                                                     # a length with a work area
    pool, cache = cp.get_default_memory_pool(), cp.fft.config.get_plan_cache()
    quick._cuda()                                                              # the kernels and tables, loaded
    cache.clear()
    limit = pool.get_limit()
    try:
        for correct in (True, False):
            need = next(b for b in range(1 << 20, 1 << 30, 1 << 16)
                        if not quick._cuda_plan(g, correct, False, b, b // 4)['kept'])
            pool.free_all_blocks()
            pool.set_limit(size=pool.used_bytes() + int(1.03 * need / 0.9))
            seen = []
            for call in range(3):
                if call == 2:                                                  # another caller's plan, with its work area
                    cp.fft.fft(cp.zeros((8, 13500), cp.complex64), axis=1)
                    assert cache.get_curr_memsize() > 0.1 * need
                info = {}
                img = quick._pfa(S, ant, col.fmin, col.df, *grid, correct=correct, complex_image=False, backend='cuda', info=info)
                cp.cuda.get_current_stream().synchronize()
                assert isinstance(img, cp.ndarray)
                del img
                seen.append((info.get('stream'), info['device'], pool.used_bytes(), cache.get_curr_memsize()))
            assert [s[0] for s in seen] == [None] * 3, seen                    # nothing kept on the host, any call
            assert seen[0][1]['on_host'] == [] and seen[1][1] == seen[0][1] and seen[2][1] == seen[0][1], seen
            assert seen[2][2] <= seen[1][2] <= seen[0][2], seen                # the pool holds no more
            assert [s[3] for s in seen] == [0] * 3                             # and CuPy's plan cache nothing
    finally:
        pool.set_limit(size=limit)


@pytest.mark.cuda
def test_cuda_sheared_raster(squinted):
    """The CUDA path on a ground grid under a twisted slant plane: the host's image."""
    import cupy as cp
    col = quick._collection(squinted['path'], size=128)
    S, ant, f0, df = col['S'], col['ant'], col['fmin'], col['df']
    grid = tuple(col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2'))
    assert quick._geometry(ant, f0, df, S.shape[1], *grid, True)['jlo'] is not None
    for correct in (True, False):
        for cplx in (True, False):
            h = quick._pfa(S, ant, f0, df, *grid, correct=correct, complex_image=cplx)
            d = quick._pfa(S, ant, f0, df, *grid, correct=correct, complex_image=cplx, backend='cuda')
            e = rel_db(cp.asnumpy(d).astype(np.complex64), h.astype(np.complex64))
            print(f'cuda against the host on a sheared raster, correct {correct}, complex {cplx}: {e:.1f} dB')
            assert d.shape == h.shape and e < -60


@pytest.mark.cuda
def test_cuda_missing_pulses():
    """The CUDA path on a history that lacks pulses: the host's image."""
    import cupy as cp
    rng = np.random.default_rng(8)
    col = sim.make_collect(res=0.5, scene=200.0, r0=60e3)
    keep = np.ones(col.Np, bool)
    keep[[40, 41, 300, 500]] = False
    tg = np.stack([rng.uniform(-60, 60, 20), rng.uniform(-60, 60, 20), np.zeros(20)], 1)
    S = np.ascontiguousarray(sim.simulate_brute(col, tg, np.ones(20) + 0j).astype(np.complex64)[keep])
    ant = col.ant[keep]
    grid = (300, 240, 0.5, 0.6, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert quick._geometry(ant, col.fmin, col.df, col.K, *grid, True)['place'] is not None
    for correct in (True, False):
        h = quick._pfa(S, ant, col.fmin, col.df, *grid, correct=correct)
        d = quick._pfa(S, ant, col.fmin, col.df, *grid, correct=correct, backend='cuda')
        e = rel_db(cp.asnumpy(d), h)
        print(f'cuda against the host with pulses missing, correct {correct}: {e:.1f} dB')
        assert e < -60


@pytest.mark.cuda
def test_cuda(spot):
    """The CUDA path (CuPy alone): the same image as the host's to rounding, corrected and uncorrected, complex and
    detected, and the same picture."""
    import cupy as cp
    col = quick._collection(spot['path'], size=256)
    S, ant, f0, df = col['S'], col['ant'], col['fmin'], col['df']
    grid = tuple(col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2'))
    for correct in (True, False):
        for cplx in (True, False):
            h = quick._pfa(S, ant, f0, df, *grid, correct=correct, complex_image=cplx)
            d = quick._pfa(S, ant, f0, df, *grid, correct=correct, complex_image=cplx, backend='cuda')
            assert isinstance(d, cp.ndarray) and d.dtype == h.dtype and d.shape == h.shape
            e = rel_db(cp.asnumpy(d).astype(np.complex64), h.astype(np.complex64))
            print(f'cuda against the host, correct {correct}, complex {cplx}: {e:.1f} dB')
            assert e < -60
    a = look(spot['path'], size=256, backend='cuda')
    b = look(spot['path'], size=256, backend='cpu')
    assert a['backend'] == 'cuda' and isinstance(a['display'], np.ndarray) and a['display'].shape == b['display'].shape
    assert np.mean(np.abs(a['display'].astype(int) - b['display'].astype(int)) <= 1) > 0.999
    assert a['cut'] == b['cut'] and {'range', 'azimuth', 'correction'} <= set(a['timing'])
    c = look(spot['path'], size=256, backend='cuda', complex_image=True)
    d = look(spot['path'], size=256, backend='cpu', complex_image=True)
    assert isinstance(c['image'], np.ndarray) and c['image'].flags.c_contiguous and rel_db(c['image'], d['image']) < -60
    # looks on the device: the remap is given a CuPy image
    got = []
    out = look(spot['path'], size=128, looks=2, backend='cuda', remap=lambda x, valid=None: (got.append(x), cp.zeros(x.shape, cp.uint8))[1])
    ref = []
    look(spot['path'], size=128, looks=2, remap=lambda x, valid=None: (ref.append(np.array(x)), np.zeros(x.shape, np.uint8))[1])
    assert out['looks'] == (2, 2) and isinstance(got[0], cp.ndarray)
    assert rel_db(cp.asnumpy(got[0]).astype(np.complex64), ref[0].astype(np.complex64)) < -60
