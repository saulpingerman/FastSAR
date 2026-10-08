"""ExactFormer (exact backprojection onto the form_image grid) on each backend against a float64 backprojection
with 64 times oversampled profiles, cubic and linear, at airborne and orbital range; a grid off the origin
(center=) against fastsar.backproject on the same points; and form_image(algorithm='bp')."""
import os, sys
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
print('ok')
