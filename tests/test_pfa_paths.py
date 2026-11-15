"""Polar format on a simulated scene. The JAX program of 0.3.0 (FASTSAR_PFA_PROGRAM=jax): the kernel gather of the
final resampling (C++ on the CPU, CUDA on a GPU) against the JAX gather, and the direct Kaiser-sinc resampling
(FASTSAR_PFA_CZT=0) against the chirp-z transforms. The stages that form on every backend against that program and
against exact backprojection; the stages on a JAX device against the C++ stages, with the device's memory held
small; what keeps a scatterer outside the grid out of the image. ExactFormer's float16 profiles against its float32
ones on cuda."""
import os

import numpy as np
import pytest

import fastsar
from fastsar import sim

GRID = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))


def rel_db(a, b):
    g = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(11)
    col = sim.make_collect(res=0.5, scene=70.0, r0=20e3)
    tg = np.stack([rng.uniform(-28, 28, 40), rng.uniform(-28, 28, 40), np.zeros(40)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(40) + 1j * rng.standard_normal(40)).astype(np.complex64)
    return col, S


def _pfa(scene, grid=GRID, guard=10.0, **env):
    """form_image's polar format by the JAX program, under the environment given."""
    col, S = scene
    env = dict(env, FASTSAR_PFA_PROGRAM='jax')
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        fastsar.api._PFA_PROGRAMS.clear()
        return fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, algorithm='pfa', pfa_guard=guard)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        fastsar.api._PFA_PROGRAMS.clear()


def test_kernel_gather_matches_jax(scene):
    """The C++ or CUDA gather (whichever the default backend uses) against the JAX gather: the same arithmetic."""
    a = _pfa(scene, FASTSAR_PFA_GATHER='1', FASTSAR_PFA_CZT='1')
    b = _pfa(scene, FASTSAR_PFA_GATHER='0', FASTSAR_PFA_CZT='1')
    e = rel_db(a, b)
    print(f'kernel gather vs jax gather: {e:.1f} dB')
    assert e < -100, e


def test_direct_resampling_matches_czt(scene):
    """Direct resampling against the chirp-z transforms: a band-limited interpolation of the same spectra."""
    a = _pfa(scene, FASTSAR_PFA_CZT='0', FASTSAR_PFA_GATHER='1')
    b = _pfa(scene, FASTSAR_PFA_CZT='1', FASTSAR_PFA_GATHER='1')
    e = rel_db(a, b)
    print(f'direct resampling vs chirp-z: {e:.1f} dB')
    assert e < -40, e


def _slant_grid(col, nx=120, ny=100, sp=0.48):
    P = col.Np
    a = col.ant[P // 2]
    e2 = -a / np.linalg.norm(a)
    n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
    n *= np.sign(n[2]) / np.linalg.norm(n)
    return dict(nx=nx, ny=ny, spx=sp, spy=sp, e1=tuple(np.cross(e2, n)), e2=tuple(e2))


@pytest.mark.parametrize('grid', ['slant', 'ground', 'odd'])
def test_stages_against_the_program_and_exact_backprojection(grid):
    """The cpu backend forms by the stages of fastsar.quick, FASTSAR_PFA_PROGRAM=jax by the JAX program. On a
    2,000-pulse aperture, whose ends the program's pulse resampling shortens by 1.6%, the two images agree to
    -30 dB on a slant-plane grid along the line of sight, at one scale (the program's); against exact
    backprojection the stages are the closer. On a ground grid, where a meter of range is not a meter along the
    range axis, the program's image carries a phase that backprojection's does not: the amplitudes agree and the
    stages have backprojection's phase. An odd pixel count forms as an even one does."""
    rng = np.random.default_rng(5)
    col = sim.make_collect(res=0.5, scene=560.0, r0=200e3)
    g = dict(slant=_slant_grid(col), ground=dict(GRID, nx=120, ny=100, spy=0.55), odd=_slant_grid(col, 121, 99))[grid]
    e1, e2 = np.array(g['e1']), np.array(g['e2'])
    tg = rng.uniform(-24, 24, (30, 1)) * e1 + rng.uniform(-20, 20, (30, 1)) * e2
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    new = fastsar.form_image(S, col.ant, col.fmin, col.df, **g, algorithm='pfa', pfa_guard=40.0, backend='cpu')
    old = _pfa((col, S), grid=g, guard=40.0)
    ex = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, **g, backend='cpu', window=True)(S)
    gain = np.vdot(new.ravel(), old.ravel()) / np.vdot(new.ravel(), new.ravel())
    e_new, e_old, e_pair = rel_db(new, ex), rel_db(old, ex), rel_db(new, old)
    a_pair = rel_db(np.abs(new).astype(np.complex64), np.abs(old).astype(np.complex64))
    print(f'{grid}: stages against the program {e_pair:.1f} dB (amplitudes {a_pair:.1f} dB, gain {abs(gain):.4f}); against '
          f'exact backprojection: stages {e_new:.1f} dB, program {e_old:.1f} dB')
    assert new.shape == old.shape == (g['nx'], g['ny']) and new.dtype == np.complex64
    assert e_new < -35 and e_new < e_old + 1 and a_pair < -28
    if grid != 'ground':
        assert e_pair < -30 and abs(gain) == pytest.approx(1.0, abs=0.02) and abs(np.angle(gain)) < 0.01


def test_program_choice(scene, monkeypatch, request):
    """form_image(algorithm='pfa') forms by the stages on every backend; FASTSAR_PFA_PROGRAM=jax takes the JAX
    program of 0.3.0 on every backend."""
    request.getfixturevalue('jax_cpu')
    from fastsar import api
    for b in ('cpu', 'jax'):
        assert api._pfa_stages(b, 'union') == b and api._pfa_stages(b, 'inscribed') == b
    monkeypatch.setenv('FASTSAR_PFA_PROGRAM', 'stages')
    assert api._pfa_stages('jax', 'union') == 'jax' and api._pfa_stages('cpu', 'union') == 'cpu'
    monkeypatch.setenv('FASTSAR_PFA_PROGRAM', 'jax')
    assert api._pfa_stages('cpu', 'union') is None and api._pfa_stages('jax', 'union') is None
    col, S = scene
    called = []
    real = api._pfa
    monkeypatch.setattr(api, '_pfa', lambda *a, **k: called.append(1) or real(*a, **k))
    old = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='jax')
    assert len(called) == 1 and np.array_equal(old, _pfa(scene))          # the released program, value for value
    monkeypatch.delenv('FASTSAR_PFA_PROGRAM')
    del called[:]
    new = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='jax')
    assert not called and new.shape == old.shape and new.dtype == np.complex64 and new.flags.c_contiguous
    e = rel_db(np.abs(new).astype(np.complex64), np.abs(old).astype(np.complex64))
    print(f'the stages on jax against the program, amplitudes (a ground grid): {e:.1f} dB')
    assert e < -20 and not np.array_equal(new, old)                       # the stages: the program's picture
    cpu = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='cpu')
    assert rel_db(new, cpu) < -65
    monkeypatch.setenv('FASTSAR_PFA_PROGRAM', 'fast')
    with pytest.raises(ValueError, match='FASTSAR_PFA_PROGRAM'):
        api._pfa_stages('cpu', 'union')


@pytest.mark.parametrize('grid', ['slant', 'ground', 'odd'])
def test_stages_on_a_jax_device(grid, request, monkeypatch):
    """backend='jax' (and 'tpu'): the stages as a TPU runs them, on JAX's CPU device. The image is the cpu
    backend's, at its scale, on a slant-plane grid, on a ground grid and for an odd pixel count; with
    pfa_support='inscribed'; and with the device's memory held to a megabyte, where the arrays live on the host and
    cross in blocks."""
    request.getfixturevalue('jax_cpu')
    rng = np.random.default_rng(5)
    col = sim.make_collect(res=0.5, scene=560.0, r0=200e3)
    g = dict(slant=_slant_grid(col), ground=dict(GRID, nx=120, ny=100, spy=0.55), odd=_slant_grid(col, 121, 99))[grid]
    e1, e2 = np.array(g['e1']), np.array(g['e2'])
    tg = rng.uniform(-24, 24, (30, 1)) * e1 + rng.uniform(-20, 20, (30, 1)) * e2
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    form = lambda backend, **kw: fastsar.form_image(S, col.ant, col.fmin, col.df, **g, algorithm='pfa', pfa_guard=40.0,
                                                    backend=backend, **kw)
    cpu, dev = form('cpu'), form('jax')
    gain = np.vdot(dev.ravel(), cpu.ravel()) / np.vdot(dev.ravel(), dev.ravel())
    e = rel_db(dev, cpu)
    print(f'{grid}: the stages on jax against cpu {e:.1f} dB, gain {abs(gain):.5f}')
    assert dev.shape == cpu.shape == (g['nx'], g['ny']) and dev.dtype == np.complex64 and dev.flags.c_contiguous
    assert e < -55 and abs(gain) == pytest.approx(1.0, abs=1e-3) and abs(np.angle(gain)) < 1e-3
    if grid == 'slant':
        assert rel_db(form('jax', pfa_support='inscribed'), form('cpu', pfa_support='inscribed')) < -55
        assert rel_db(form('jax', window=False), form('cpu', window=False)) < -55
        cal = np.linalg.norm(col.ant, axis=1) + 3.0                         # a reference range that is not |ant|
        assert rel_db(form('jax', ref=cal), form('cpu', ref=cal)) < -55
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '1')
    small = form('jax')
    assert rel_db(small, dev) < -90


def test_scatterer_outside_the_grid_stays_out():
    """What keeps wrap-around out of the stages' image is their two gates, not pfa_guard. Each pulse's range profile
    is cut to the picture's delays and each range wavenumber's azimuth profile to the picture's extent (a raised
    cosine beyond each, of the larger of 8 pixels and 3% of the half extent, in meters), and the raster's periods
    hold the picture, its displacement and a gate's far edge, so what a gate's edge passes wraps to beyond the
    opposite edge of the picture. A scatterer outside the grid, from 3 pixels beyond its edge to where an ungated
    raster would fold it onto the picture's middle, in azimuth and in range: what it adds to the picture, as a
    part of its own peak, is its sidelobes as exact backprojection has them (to 3 dB), and beyond the gates no
    more than -50 dB anywhere, where a fold would add 0 dB. Within the gate's edge its near sidelobes wrap into the
    far half of the picture at -50 dB (Taylor weights). pfa_guard changes the image's scale and nothing else."""
    rng = np.random.default_rng(12)
    col = sim.make_collect(res=0.3, scene=220.0, r0=30e3)
    g = _slant_grid(col, 256, 256, 0.28)
    e1, e2 = np.array(g['e1']), np.array(g['e2'])
    geo = fastsar.quick._geometry(col.ant, col.fmin, col.df, col.K, g['nx'], g['ny'], g['spx'], g['spy'], e1, e2, True)
    half = (0.5 * g['nx'] * g['spx'], 0.5 * g['ny'] * g['spy'])
    period = (geo['nfx'] * g['spx'], geo['nfy'] * g['spy'])
    shift = geo['shift']
    assert period[0] >= half[0] + shift[0] + geo['Ex'] + geo['gx'] and period[1] >= half[1] + shift[1] + geo['Er'] + geo['gy']
    assert geo['gx'] == pytest.approx(8 * g['spx']) and geo['Ex'] < half[0] + 1.0
    scene = rng.uniform(-30, 30, (30, 1)) * e1 + rng.uniform(-30, 30, (30, 1)) * e2
    S0 = sim.simulate_brute(col, scene, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    form = lambda S, guard=30.0: fastsar.form_image(S, col.ant, col.fmin, col.df, **g, algorithm='pfa', pfa_guard=guard,
                                                    backend='cpu')
    exact = lambda S: fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, **g, backend='cpu', window=True)(S)
    one = sim.simulate_brute(col, np.zeros((1, 3)), np.array([1.0 + 0j])).astype(np.complex64)
    img0, ex0, peak, peak_ex = form(S0), exact(S0), np.abs(form(one)).max(), np.abs(exact(one)).max()
    db = lambda v: 20 * np.log10(max(float(v), 1e-30))
    for axis, e, edge, guard in ((0, e1, geo['Ex'], geo['gx']), (1, e2, geo['Er'], geo['gy'])):
        for name, d, limit in (('3 pixels beyond the edge', half[axis] + 3 * g['spx'], None),
                               ("in the gate's edge", edge + 0.5 * guard, None), ('beyond the gate', edge + 1.5 * guard, -50),
                               ('half a period out', 0.5 * period[axis] + 5.0, -50), ('a period out', period[axis] - 10.0, -60)):
            for sign in ((1.0, -1.0) if limit is None else (1.0,)):
                Sx = 1000.0 * sim.simulate_brute(col, (sign * d * e)[None], np.array([1.0 + 0j])).astype(np.complex64)
                leak = np.abs(form(S0 + Sx) - img0) / (1000.0 * peak)
                far = leak[:128] if axis == 0 else leak[:, :128]
                far = far if sign > 0 else (leak[128:] if axis == 0 else leak[:, 128:])
                line = f"{'azimuth' if axis == 0 else 'range'}, {name} ({sign * d:+.1f} m): at most {db(leak.max()):.1f} dB of its peak, {db(far.max()):.1f} dB in the far half"
                if limit is None:                              # in the gate: its sidelobes, as exact backprojection's
                    true = db((np.abs(exact(S0 + Sx) - ex0) / (1000.0 * peak_ex)).max())
                    print(line + f'; exact backprojection {true:.1f} dB')
                    assert db(leak.max()) < true + 3.0 and db(far.max()) < -48
                else:                                          # beyond it: gated out
                    print(line)
                    assert db(leak.max()) < limit and db(far.max()) < -55
    # pfa_guard: the scale alone (the size of the JAX program's azimuth transform, which it set)
    S = S0 + 1000.0 * sim.simulate_brute(col, ((geo['Ex'] + 1.5 * geo['gx']) * e1)[None], np.array([1.0 + 0j])).astype(np.complex64)
    a, b = form(S, 10.0), form(S, 300.0)
    k = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    assert rel_db(a, b) < -100 and abs(k.imag) < 1e-6 * abs(k) and k.real == pytest.approx(
        fastsar.pfa2._fft_size(g['nx'] + int(600 / g['spx'])) / fastsar.pfa2._fft_size(g['nx'] + int(20 / g['spx'])), rel=1e-5)


@pytest.mark.cuda
def test_scatterer_outside_the_grid_stays_out_on_cuda():
    """The cuda stages take the gates of the same geometry: a scatterer beyond the gate, in azimuth and in range."""
    rng = np.random.default_rng(12)
    col = sim.make_collect(res=0.3, scene=220.0, r0=30e3)
    g = _slant_grid(col, 256, 256, 0.28)
    e1, e2 = np.array(g['e1']), np.array(g['e2'])
    geo = fastsar.quick._geometry(col.ant, col.fmin, col.df, col.K, g['nx'], g['ny'], g['spx'], g['spy'], e1, e2, True)
    scene = rng.uniform(-30, 30, (30, 1)) * e1 + rng.uniform(-30, 30, (30, 1)) * e2
    S0 = sim.simulate_brute(col, scene, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    for e, d in ((e1, geo['Ex'] + 1.5 * geo['gx']), (e2, geo['Er'] + 1.5 * geo['gy']), (e1, geo['nfx'] * g['spx'] - 10.0)):
        S = S0 + sim.simulate_brute(col, (d * e)[None], np.array([1000.0 + 0j])).astype(np.complex64)
        a = fastsar.form_image(S, col.ant, col.fmin, col.df, **g, algorithm='pfa', backend='cuda')
        b = fastsar.form_image(S, col.ant, col.fmin, col.df, **g, algorithm='pfa', backend='cpu')
        assert rel_db(a, b) < -50


@pytest.mark.tpu
def test_stages_on_a_tpu(scene):
    """On a TPU the stages are form_image's polar format, and its image is the cpu backend's."""
    col, S = scene
    a = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='tpu')
    b = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='cpu')
    e = rel_db(a, b)
    print(f'polar format, tpu against cpu: {e:.1f} dB')
    assert isinstance(a, np.ndarray) and a.dtype == np.complex64 and e < -55


@pytest.mark.cuda
def test_stages_on_cuda(scene):
    """The cuda backend's stages give the cpu backend's image."""
    col, S = scene
    a = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='cuda')
    b = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0, backend='cpu')
    e = rel_db(a, b)
    print(f'polar format, cuda against cpu: {e:.1f} dB')
    assert isinstance(a, np.ndarray) and a.dtype == np.complex64 and e < -60


@pytest.mark.cuda
@pytest.mark.parametrize('profiles', ['float16', 'float16-pair'])
def test_exact_float16_profiles(scene, profiles):
    """float16 range profiles against float32 ones (linear interpolation): the rounding is far below the
    interpolation error of the linear kernel."""
    col, S = scene
    K = S.shape[1]
    kw = dict(backend='cuda', interp='linear')
    ref = fastsar.ExactFormer(col.ant, col.fmin, col.df, K, **GRID, **kw)(S)
    img = fastsar.ExactFormer(col.ant, col.fmin, col.df, K, **GRID, profiles=profiles, **kw)(S)
    e = rel_db(img, ref)
    print(f'{profiles} vs float32 profiles: {e:.1f} dB')
    assert e < -60, e


def test_exact_profiles_validation(scene):
    col, S = scene
    with pytest.raises(ValueError, match='profiles'):
        fastsar.ExactFormer(col.ant, col.fmin, col.df, S.shape[1], **GRID, backend='cpu', profiles='float8')
    with pytest.raises(ValueError, match='float16-pair'):
        fastsar.ExactFormer(col.ant, col.fmin, col.df, S.shape[1], **GRID, backend='cpu', profiles='float16-pair')


def test_grid_rotated_from_line_of_sight_warns(scene):
    """The polar raster is taken about e2: a grid whose range axis is rotated from the mid-aperture line of sight
    gets a warning (2 degrees and beyond); the aligned grid does not."""
    import warnings
    col, S = scene
    th = np.radians(10.0)
    rot = dict(GRID, e1=(-np.sin(th), np.cos(th), 0.0), e2=(np.cos(th), np.sin(th), 0.0))
    with pytest.warns(UserWarning, match='line of sight'):
        fastsar.form_image(S, col.ant, col.fmin, col.df, **rot, algorithm='pfa', pfa_guard=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0)


def test_band_wider_than_grid():
    """A grid coarser than the aperture's azimuth resolution: the spectrum is cropped to the raster with a warning,
    and the image still forms (before 0.1.7 the padding to the raster failed with a negative width)."""
    import warnings
    rng = np.random.default_rng(3)
    col = sim.make_collect(res=0.5, scene=70.0, r0=20e3)
    tg = np.stack([rng.uniform(-20, 20, 10), rng.uniform(-20, 20, 10), np.zeros(10)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(10) + 1j * rng.standard_normal(10)).astype(np.complex64)
    grid = dict(nx=48, ny=48, spx=1.5, spy=1.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        img = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, algorithm='pfa', pfa_guard=10.0)
    assert img.shape == (48, 48) and np.isfinite(img).all()
    assert any('band' in str(x.message) for x in w), [str(x.message) for x in w]
