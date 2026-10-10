"""The Metal backend of ExactFormer (Apple GPUs): the same image as the CPU kernel, both interpolations; skipped
where Metal is not available."""
import numpy as np
import pytest

import fastsar
from fastsar import sim

try:
    from fastsar import metal
    HAVE = metal.available()
except Exception:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason='no Metal device')


@pytest.mark.parametrize('interp', ['linear', 'cubic'])
def test_metal_matches_cpu(interp):
    rng = np.random.default_rng(3)
    col = sim.make_collect(res=0.5, scene=120.0, r0=20e3)
    tg = np.array([[-40.0, -30.0, 0.0], [10.0, 20.0, 0.0], [45.0, -25.0, 0.0]])
    amp = np.ones(3) * np.exp(1j * rng.uniform(0, 2 * np.pi, 3))
    S = sim.simulate(col, tg, amp).astype(np.complex64)
    ant, f0, df = np.asarray(col.ant, np.float64), float(col.fmin), float(col.df)
    K = S.shape[1]
    cpu = fastsar.ExactFormer(ant, f0, df, K, 256, 256, 0.4, 0.4, backend='cpu', interp=interp)(S)
    gpu_former = fastsar.ExactFormer(ant, f0, df, K, 256, 256, 0.4, 0.4, backend='metal', interp=interp)
    gpu = gpu_former(S)
    assert gpu_former.backend == 'metal'
    assert np.isfinite(gpu).all()
    assert np.abs(gpu - cpu).max() / np.abs(cpu).max() < 1e-4
    assert np.abs(gpu).max() / np.median(np.abs(gpu)) > 1e3


def test_auto_picks_metal_for_exact():
    col = sim.make_collect(res=1.0, scene=60.0, r0=20e3)
    f = fastsar.ExactFormer(np.asarray(col.ant), float(col.fmin), float(col.df), int(col.K), 64, 64, 0.8, 0.8)
    assert f.backend == 'metal'
    g = fastsar.ImageFormer(np.asarray(col.ant), float(col.fmin), float(col.df), int(col.K), 64, 64, 0.8, 0.8, backend='metal')
    assert g.backend == 'cpu'                 # the factorized former has no Metal kernels
