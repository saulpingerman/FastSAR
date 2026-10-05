"""form_image on a simulated spotlight scene: every available backend against the plain JAX program, and polar
format against factorized backprojection (amplitude, since polar format's residual is larger)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sarform
from sarform import sim

rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
grid = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))   # the simulator's track runs along y, looking along +x


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


ref = sarform.form_image(S, col.ant, col.fmin, col.df, **grid, backend='jax', T=16, levels=2)
print('backends here:', sarform.available_backends())
for b in sarform.available_backends():
    img = sarform.form_image(S, col.ant, col.fmin, col.df, **grid, backend=b, T=16, levels=2)
    e = rel_db(img, ref)
    print(f'{b:5s} vs jax: {e:.1f} dB')
    assert e < -60, (b, e)
if 'cuda' in sarform.available_backends():
    ref32 = sarform.form_image(S, col.ant, col.fmin, col.df, **grid, backend='jax', T=32, levels=2)
    img = sarform.form_image(S, col.ant, col.fmin, col.df, **grid, backend='cuda', precision='float16', T=32, levels=2)
    e = rel_db(img, ref32)
    print(f'cuda float16 vs jax: {e:.1f} dB')
    assert e < -45, e
pf = sarform.form_image(S, col.ant, col.fmin, col.df, **grid, algorithm='pfa', pfa_guard=10.0)
c = np.corrcoef(np.abs(pf).ravel(), np.abs(ref).ravel())[0, 1]
print(f'pfa amplitude correlation with ffbp: {c:.3f}')
assert c > 0.9, c
print('ok')
