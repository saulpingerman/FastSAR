"""Polar format on a simulated scene. The JAX program's device paths (FASTSAR_PFA_PROGRAM=jax, the jax and tpu
backends): the kernel gather of the final resampling (C++ on the CPU, CUDA on a GPU) against the JAX gather, and
the direct Kaiser-sinc resampling (FASTSAR_PFA_CZT=0) against the chirp-z transforms. The stages that form on the
cpu and cuda backends against that program and against exact backprojection. ExactFormer's float16 profiles against
its float32 ones on cuda."""
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


def test_program_choice(scene, monkeypatch):
    """FASTSAR_PFA_PROGRAM: 'jax' keeps the JAX program on every backend; the jax backend keeps it too."""
    from fastsar import api
    assert api._pfa_stages('cpu', 'union') == 'cpu' and api._pfa_stages('cpu', 'inscribed') == 'cpu'
    assert api._pfa_stages('jax', 'union') is None
    monkeypatch.setenv('FASTSAR_PFA_PROGRAM', 'jax')
    assert api._pfa_stages('cpu', 'union') is None
    monkeypatch.setenv('FASTSAR_PFA_PROGRAM', 'fast')
    with pytest.raises(ValueError, match='FASTSAR_PFA_PROGRAM'):
        api._pfa_stages('cpu', 'union')


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
