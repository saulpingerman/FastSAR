"""The CUDA factorized image against the JAX dense image (float32, highest precision) on the simulated scene."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax
jax.config.update('jax_platform_name', 'cpu')
from fastsar import sim, ffbp2, ffbp_cuda
import cupy as cp
rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
n = 128
for nlev in (2, 3):
    plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
    print('levels', [(l['P'], l['K'], l['Dk'], l['Dp'], l['C']) for l in plan['levels']])
    coll = ffbp2.collection_arrays(plan, col.ant)
    static = ffbp2.static_arrays('fp32', plan, 'dense')
    arrs = ffbp2.device_arrays('fp32', plan, coll, static)
    hre, him, scale = ffbp2.prepare('fp32', S)
    fn = ffbp2.make_ffbp('fp32', plan, 'dense', 1 << 24, 'direct')
    re, im = fn(hre, him, arrs)
    ref = (np.asarray(re) + 1j * np.asarray(im)) * scale
    form = ffbp_cuda.make_ffbp_cuda(plan, coll)
    img = cp.asnumpy(form(S, ng=2))
    d = img - ref
    print(f'nlev {nlev}: cuda vs dense float32: {10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(ref) ** 2)):.1f} dB, peak ratio {np.abs(img).max() / np.abs(ref).max():.4f}')
