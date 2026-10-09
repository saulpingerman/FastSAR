"""form_image on a simulated spotlight scene: every available backend against the plain JAX program, and polar
format against factorized backprojection (amplitude, since polar format's residual is larger)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim

rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
grid = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))   # the simulator's track runs along y, looking along +x


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


ref = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend='jax', T=16, levels=2)
print('backends here:', fastsar.available_backends())
for b in fastsar.available_backends():
    img = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend=b, T=16, levels=2)
    e = rel_db(img, ref)
    print(f'{b:5s} vs jax: {e:.1f} dB')
    assert e < -60, (b, e)
if 'cuda' in fastsar.available_backends():
    ref32 = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend='jax', T=32, levels=2)
    img = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend='cuda', precision='float16', T=32, levels=2)
    e = rel_db(img, ref32)
    print(f'cuda float16 vs jax: {e:.1f} dB')
    assert e < -45, e
pf = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, algorithm='pfa', pfa_guard=10.0)
c = np.corrcoef(np.abs(pf).ravel(), np.abs(ref).ravel())[0, 1]
print(f'pfa amplitude correlation with ffbp: {c:.3f}')
assert c > 0.9, c
print('ok')

# a reused former gives the same image as form_image, and the second call skips setup
import time
b0 = fastsar.available_backends()[0]
former = fastsar.ImageFormer(col.ant, col.fmin, col.df, S.shape[1], **grid, backend=b0, T=16, levels=2)
img1 = former(S); t = time.perf_counter(); img2 = former(S); t2 = time.perf_counter() - t
ref_b = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend=b0, T=16, levels=2)
print(f'ImageFormer ({b0}) vs form_image: {rel_db(img2, ref_b):.1f} dB; second call {t2:.3f} s')
assert rel_db(img2, ref_b) < -100
print('ok')

# four levels (a grid that three cannot split falls back to more): every backend against the JAX program
ref4 = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend='jax', T=16, levels=4)
for b in fastsar.available_backends():
    e = rel_db(fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend=b, T=16, levels=4), ref4)
    print(f'{b:5s} four levels vs jax: {e:.1f} dB')
    assert e < -60, (b, e)
assert fastsar.ImageFormer(col.ant, col.fmin, col.df, S.shape[1], 128, 128 * 600, 0.5, 0.5, grid['e1'], grid['e2'],
                           backend='cpu', T=16).levels == 4
print('ok')
