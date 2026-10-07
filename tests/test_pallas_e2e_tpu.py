"""End-to-end check of the TPU kernels (level kernel and fused final stage) in interpret mode on the CPU: the pallas2
image against the dense image on the simulated scene."""
import os, sys
os.environ['FFBP_FORCE_TPU_KERNELS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
from fastsar import sim, ffbp2, pallas_ffbp
bad = []                  # thresholds about 5 dB above the values measured on the CPU (interpret mode) when they were set


def check(label, e, lim):
    if not e < lim:
        bad.append(f'{label}: {e:.1f} dB (limit {lim} dB)')
    return e


for name in ('fused_rotate_dec_k2', 'fused_rotate_dec_k3', 'fused_final', 'fused_final2', 'fused_final3'):
    _o = getattr(pallas_ffbp, name)
    setattr(pallas_ffbp, name, (lambda o: lambda *a, **k: o(*a, **{**k, 'interpret': True}))(_o))
rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
n = 128
import os as _os
nlev = int(_os.environ.get('NLEV', '2'))
plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
print('levels', nlev)
coll = ffbp2.collection_arrays(plan, col.ant)
for pol in ('fp32_fast', 'fp32_high'):
    hre, him, _ = ffbp2.prepare(pol, S)
    out = {}
    for filt, kw in (('dense', {}), ('pallas2', dict(pallas_pb=128, pallas_nc=4, pallas_ng=2))):
        static = ffbp2.static_arrays(pol, plan, filt)
        arrs = ffbp2.device_arrays(pol, plan, coll, static)
        fn = ffbp2.make_ffbp(pol, plan, filt, 1 << 24, 'direct', **kw)
        re, im = fn(hre, him, arrs)
        out[filt] = np.asarray(re) + 1j * np.asarray(im)
    d = out['pallas2'] - out['dense']
    print(pol, 'pallas2 (tpu kernels, interpret) vs dense: %.1f dB' % check(pol, 10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(out['dense']) ** 2)), -40 if pol == 'fp32_fast' else -90))

if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
