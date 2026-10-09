"""ExactFormer (exact backprojection onto the form_image grid) on each backend against a float64 backprojection
with 64 times oversampled profiles, cubic and linear, at airborne and orbital range; a grid off the origin
(center=) against fastsar.backproject on the same points; form_image(algorithm='bp') with its interp, upsample and
center; airborne ranges of 1 to 20 km, where the tile shrinks with range (every CUDA tile variant); a wide grid at
1 km whose nearest pixels lie mid-edge, not at a corner; the tile choice and its warning; input validation; and
memory()."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, bp


def rel_db(a, b):
    g = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


backends = [b for b in ('cpu', 'cuda', 'jax') if b == 'jax' or b in fastsar.available_backends()]
print('backends:', backends)
rng = np.random.default_rng(5)
for r0 in (20e3, 600e3):
    col = sim.make_collect(res=0.5, scene=60.0, r0=r0)
    tg = np.stack([rng.uniform(-25, 25, 30), rng.uniform(-25, 25, 30), np.zeros(30)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    grid = dict(nx=100, ny=120, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
    ref = bp.backproject(S, col.ant, col.fmin, col.df, fastsar.plane_points(**grid), backend='cpu', upsample=64)
    line = []
    for b in backends:
        for interp, limit in (('cubic', -65), ('linear', -52)):
            img = fastsar.ExactFormer(col.ant, col.fmin, col.df, S.shape[1], **grid, backend=b, interp=interp)(S)
            e = rel_db(img, ref)
            line.append(f'{b} {interp} {e:.1f} dB')
            assert img.shape == (100, 120) and e < limit, (r0, b, interp, e)
    print(f'range {r0/1e3:.0f} km: ' + ', '.join(line), flush=True)

# a grid centered off the origin against backproject on the same points
c = np.array([7.0, -4.0, 0.0])
grid = dict(nx=64, ny=48, spx=0.4, spy=0.6, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
pts = fastsar.plane_points(**grid) + c
ref = bp.backproject(S, col.ant, col.fmin, col.df, pts, backend='cpu', upsample=64)
for b in backends:
    e = rel_db(fastsar.ExactFormer(col.ant, col.fmin, col.df, S.shape[1], **grid, backend=b, center=c)(S), ref)
    print(f'{b}: off-center grid {e:.1f} dB')
    assert e < -65, (b, e)
e = rel_db(fastsar.form_image(S, col.ant, col.fmin, col.df, 100, 120, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                              algorithm='bp', backend='cpu'),
           bp.backproject(S, col.ant, col.fmin, col.df, fastsar.plane_points(100, 120, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)), backend='cpu', upsample=64))
print(f"form_image(algorithm='bp'): {e:.1f} dB")
assert e < -65, e

# form_image passes interp, upsample and center through to ExactFormer, and refuses them for the other algorithms
kw = dict(interp='linear', upsample=8, center=c)
a = fastsar.form_image(S, col.ant, col.fmin, col.df, 64, 48, 0.4, 0.6, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), algorithm='bp',
                       backend='cpu', **kw)
b = fastsar.ExactFormer(col.ant, col.fmin, col.df, S.shape[1], 64, 48, 0.4, 0.6, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                        backend='cpu', **kw)(S)
assert np.array_equal(a, b)
for k, v in kw.items():
    try:
        fastsar.form_image(S, col.ant, col.fmin, col.df, 64, 48, 0.4, 0.6, algorithm='ffbp', backend='cpu', **{k: v})
        raise AssertionError(f'{k} accepted with ffbp')
    except ValueError:
        pass
print('form_image(algorithm=\'bp\') passes interp, upsample, center')

# airborne ranges: X band, 128 by 128 pixels of 1 or 2 m; the tile shrinks at short range (cpu 32, 16, 8; cuda
# 32x32, 16x16, 8x16 and 8x8, each a compiled variant of the kernel)
expect_tiles = {(1e3, 1.0): {'cpu': (8, 8), 'cuda': (8, 8)}, (5e3, 2.0): {'cpu': (8, 8), 'cuda': (8, 16)},
                (10e3, 2.0): {'cpu': (16, 16), 'cuda': (16, 16)}, (20e3, 1.0): {'cpu': (32, 32), 'cuda': (32, 32)}}
for (r0, sp), tiles in expect_tiles.items():
    rng = np.random.default_rng(1)
    col = sim.make_collect(res=max(sp, 0.5), scene=128 * sp, r0=r0)
    tg = np.stack([rng.uniform(-58 * sp, 58 * sp, 40), rng.uniform(-58 * sp, 58 * sp, 40), np.zeros(40)], 1)
    S = sim.simulate_brute(col, tg, np.ones(40)).astype(np.complex64)
    g = dict(nx=128, ny=128, spx=sp, spy=sp)
    ref = bp.backproject(S, col.ant, col.fmin, col.df, fastsar.plane_points(**g), backend='cpu', upsample=64)
    line = []
    for b in backends:
        with warnings.catch_warnings():
            warnings.simplefilter('error')              # no tile misses the phase limit here
            f = fastsar.ExactFormer(col.ant, col.fmin, col.df, S.shape[1], **g, backend=b)
        e = rel_db(f(S), ref)
        line.append(f'{b} {e:.1f} dB (tile {f.tile})')
        assert e < -65, (r0, sp, b, e)
        assert b == 'jax' or f.tile == tiles[b], (r0, sp, b, f.tile)
    print(f'{r0 / 1e3:g} km, {sp:g} m pixels: ' + ', '.join(line), flush=True)

# the tile choice on cuda without a GPU (the selection alone)
import fastsar.exact as ex
for (tx, ty), (tb, px) in ex.CUDA_BLOCKS.items():
    assert tb * px // tx == ty and tb * px % tx == 0 and tuple(ex.TILES['cuda']).count((tx, ty)) == 1
col = sim.make_collect(res=1.0, scene=128.0, r0=1e3)
f = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, 128, 128, 1.0, 1.0, backend='jax')
f.backend = 'cuda'
f._choose_tile()
assert f.tile == (8, 8), f.tile

# a wide grid at 1 km (768 by 768 pixels of 0.5 m): the nearest pixel to the antenna lies on the near edge at
# mid-azimuth, not at a corner. Every real pixel's bins lie inside the window, and the near rows match backprojection
col = sim.make_collect(res=0.5, scene=384.0, r0=1e3)
ant = col.ant[::16]
n, sp = 768, 0.5
rng = np.random.default_rng(3)
S = (rng.standard_normal((len(ant), col.K)) + 1j * rng.standard_normal((len(ant), col.K))).astype(np.complex64)
f = fastsar.ExactFormer(ant, col.fmin, col.df, col.K, n, n, sp, sp, backend='cpu')
pts = fastsar.plane_points(n, n, sp, sp)
dist = np.sqrt(((pts.reshape(-1, 1, 3) - ant[None]) ** 2).sum(-1))           # [pixels, pulses]
t = (dist - f.ref) * f.inv_dr - f.lo                                        # window index of every pixel and pulse
print(f'wide grid at 1 km: window indices {t.min():.1f} to {t.max():.1f} of W={f.W} (tile {f.tile})')
assert t.min() >= 2 and t.max() <= f.W - 3, (t.min(), t.max(), f.W)
del dist, t
img = f(S)
imgs = {b: img if b == 'cpu' else fastsar.ExactFormer(ant, col.fmin, col.df, col.K, n, n, sp, sp, backend=b)(S)
        for b in backends if b != 'jax'}           # the JAX path does not tile or crop
for rows in (slice(0, 4), slice(n // 2, n // 2 + 4)):
    r = bp.backproject(S, ant, col.fmin, col.df, pts[rows], backend='cpu', upsample=64)
    for b, im in imgs.items():
        e = rel_db(im[rows], r)
        print(f'  {b} rows {rows.start}-{rows.stop - 1}: {e:.1f} dB')
        assert e < -65, (b, rows, e)

# the warning when even the smallest tile misses the phase limit, and the error at a few metres from the antenna
y = np.linspace(-5, 5, 64)
ant = np.stack([np.full_like(y, -20.0), y, np.full_like(y, 20.0)], 1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    f = fastsar.ExactFormer(ant, 9.6e9, 2e6, 128, 32, 32, 1.0, 1.0, backend='cpu')
assert f.tile == (4, 4) and len(w) == 1 and issubclass(w[0].category, UserWarning) and 'backproject' in str(w[0].message), w
print(f'close range warning: {w[0].message}')

# input validation: each raises ValueError
col = sim.make_collect(res=0.5, scene=40.0, r0=20e3)
base = dict(ant=col.ant, fmin=col.fmin, df=col.df, K=col.K, nx=32, ny=32, spx=0.5, spy=0.5)
bad = [('upsample=0', dict(upsample=0)), ('upsample=2.5', dict(upsample=2.5)), ('upsample=-2', dict(upsample=-2)),
       ('chunk=0', dict(chunk=0)), ('chunk=1.5', dict(chunk=1.5)), ('center 2-vector', dict(center=(1.0, 2.0))),
       ('center NaN', dict(center=(np.nan, 0.0, 0.0))), ('ref NaN', dict(ref=np.where(np.arange(len(col.ant)) == 3, np.nan, 1.0))),
       ('ref wrong length', dict(ref=np.ones(5))), ('df=0', dict(df=0.0)), ('fmin NaN', dict(fmin=np.nan)),
       ("fmin 'x'", dict(fmin='x')), ('K=1', dict(K=1)), ("interp 'nearest'", dict(interp='nearest')),
       ("backend 'gpu'", dict(backend='gpu')), ('grid wider than c / (2 df)', dict(nx=600, ny=600)),
       ('antenna in the grid', dict(ant=np.stack([np.zeros(8), np.linspace(-1, 1, 8), np.full(8, 0.5)], 1)))]
if 'tpu' not in fastsar.available_backends():
    bad.append(("backend 'tpu' without a TPU", dict(backend='tpu')))
for name, kw in bad:
    try:
        fastsar.ExactFormer(**{**base, 'backend': 'cpu', **kw})
    except ValueError as err:
        print(f'  {name}: ValueError: {str(err)[:110]}')
        continue
    raise AssertionError(f'{name}: accepted')
try:
    fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, 32, 32, 0.5, 0.5, (1.0, 0, 0), (0, 1.0, 0), 'cpu')
    raise AssertionError('backend accepted positionally')
except TypeError:
    pass

# memory(): the bytes of a call's buffers, as ImageFormer.memory()
for b in backends:
    m = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, 32, 32, 0.5, 0.5, backend=b).memory()
    assert m['backend'] == b and m['needed'] > 0 and m['available'] > 0 and set(m) == {'backend', 'needed', 'available', 'full_speed', 'parts'}, m
    print(f'  memory() {b}: {m["needed"] / 1e6:.1f} MB needed')

# a CuPy history on the cuda backend is read in place and gives the same image
if 'cuda' in backends:
    import cupy as cp
    S = sim.simulate_brute(col, np.array([[0.0, 0, 0], [3.0, -2, 0]]), np.ones(2)).astype(np.complex64)
    f = fastsar.ExactFormer(col.ant, col.fmin, col.df, col.K, 32, 32, 0.5, 0.5, backend='cuda')
    assert np.array_equal(f(cp.asarray(S)), f(S))
    print('cuda: a CuPy history gives the image of the host one')
print('ok')
