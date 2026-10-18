"""Polar format's device paths against its JAX path on a simulated scene: the kernel gather of the final resampling
(C++ on the CPU, CUDA on a GPU) against the JAX gather, and the direct Kaiser-sinc resampling (FASTSAR_PFA_CZT=0)
against the chirp-z transforms; ExactFormer's float16 profiles against its float32 ones on cuda."""
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


def _pfa(scene, **env):
    col, S = scene
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        fastsar.api._PFA_PROGRAMS.clear()
        return fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, algorithm='pfa', pfa_guard=10.0)
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
