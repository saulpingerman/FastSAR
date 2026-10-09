"""The C++ (CPU) factorized image against the JAX dense image (float32, highest precision) on the simulated scene."""
import numpy as np
import pytest

from fastsar import sim, ffbp2, ffbp_cpu

pytestmark = pytest.mark.usefixtures('jax_cpu')


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(1)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
    amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    return col, sim.simulate_brute(col, pos, amp).astype(np.complex64)


# thresholds about 5 dB above the values measured when they were set (-86, -79 dB)
@pytest.mark.parametrize('nlev', [2, 3])
def test_cpu_vs_dense(scene, nlev):
    col, S = scene
    n = 128
    plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
    print('levels', [(l['P'], l['K'], l['Dk'], l['Dp'], l['C']) for l in plan['levels']])
    coll = ffbp2.collection_arrays(plan, col.ant)
    static = ffbp2.static_arrays('fp32', plan, 'dense')
    arrs = ffbp2.device_arrays('fp32', plan, coll, static)
    hre, him, scale = ffbp2.prepare('fp32', S)
    fn = ffbp2.make_ffbp('fp32', plan, 'dense', 1 << 24, 'direct')
    re, im = fn(hre, him, arrs)
    ref = (np.asarray(re) + 1j * np.asarray(im)) * scale
    form = ffbp_cpu.make_ffbp_cpu(plan, coll)
    img = form(S, ng=2)
    d = img - ref
    e = 10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(ref) ** 2))
    print(f'nlev {nlev}: cpu vs dense float32: {e:.1f} dB, peak ratio {np.abs(img).max() / np.abs(ref).max():.4f}')
    assert not (e > -70 or abs(np.abs(img).max() / np.abs(ref).max() - 1) > 1e-3), f'nlev {nlev}: {e:.1f} dB'
