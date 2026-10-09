"""form_image on a simulated spotlight scene: every available backend against the plain JAX program, and polar
format against factorized backprojection (amplitude, since polar format's residual is larger)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import sim

# the backends fastsar.available_backends() can return; those this machine lacks are skipped (or fail with --require)
BACKENDS = [pytest.param('tpu', marks=pytest.mark.tpu), pytest.param('cuda', marks=pytest.mark.cuda), 'cpu']
GRID = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))   # the simulator's track runs along y, looking along +x


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(1)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
    amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
    print('backends here:', fastsar.available_backends())
    return SimpleNamespace(col=col, S=S)


@pytest.fixture(scope='module')
def ref(scene):
    c = scene.col
    return fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend='jax', T=16, levels=2)


@pytest.mark.parametrize('b', BACKENDS)
def test_backend_vs_jax(scene, ref, b):
    c = scene.col
    img = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend=b, T=16, levels=2)
    e = rel_db(img, ref)
    print(f'{b:5s} vs jax: {e:.1f} dB')
    assert e < -60, (b, e)


@pytest.mark.cuda
def test_cuda_float16(scene):
    c, S = scene.col, scene.S
    ref32 = fastsar.form_image(S, c.ant, c.fmin, c.df, **GRID, backend='jax', T=32, levels=2)
    img = fastsar.form_image(S, c.ant, c.fmin, c.df, **GRID, backend='cuda', precision='float16', T=32, levels=2)
    e = rel_db(img, ref32)
    print(f'cuda float16 vs jax: {e:.1f} dB')
    assert e < -45, e


def test_pfa(scene, ref):
    c = scene.col
    pf = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, algorithm='pfa', pfa_guard=10.0)
    cc = np.corrcoef(np.abs(pf).ravel(), np.abs(ref).ravel())[0, 1]
    print(f'pfa amplitude correlation with ffbp: {cc:.3f}')
    assert cc > 0.9, cc


def test_former_reuse(scene):
    """A reused former gives the same image as form_image, and the second call skips setup."""
    c, S = scene.col, scene.S
    b0 = fastsar.available_backends()[0]
    former = fastsar.ImageFormer(c.ant, c.fmin, c.df, S.shape[1], **GRID, backend=b0, T=16, levels=2)
    former(S)
    t = time.perf_counter()
    img2 = former(S)
    t2 = time.perf_counter() - t
    ref_b = fastsar.form_image(S, c.ant, c.fmin, c.df, **GRID, backend=b0, T=16, levels=2)
    print(f'ImageFormer ({b0}) vs form_image: {rel_db(img2, ref_b):.1f} dB; second call {t2:.3f} s')
    assert rel_db(img2, ref_b) < -100


@pytest.fixture(scope='module')
def ref4(scene):
    c = scene.col
    return fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend='jax', T=16, levels=4)


@pytest.mark.parametrize('b', BACKENDS)
def test_four_levels(scene, ref4, b):
    """Four levels (a grid that three cannot split falls back to more): every backend against the JAX program."""
    c = scene.col
    e = rel_db(fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend=b, T=16, levels=4), ref4)
    print(f'{b:5s} four levels vs jax: {e:.1f} dB')
    assert e < -60, (b, e)


def test_four_levels_chosen(scene):
    c = scene.col
    assert fastsar.ImageFormer(c.ant, c.fmin, c.df, scene.S.shape[1], 128, 128 * 600, 0.5, 0.5, GRID['e1'], GRID['e2'],
                               backend='cpu', T=16).levels == 4
