"""Accuracy against range and settings, the figures of docs/algorithms.md:

  final tile size   factorized images with T=16 and T=32 against float64 exact backprojection at 1, 4 and 16 km and
                    at orbital range (128 by 128 pixels of 0.5 m), beside ffbp2.final_phase_error's prediction
  oversampling      exact backprojection with upsample 2, 8 and 32 against upsample 128 at 4 km

Errors are 10 log10 of the energy of the difference over that of the reference, after one fitted complex gain.
Where the final stage's model sets the error (predicted above -55 dB) the script fails if a measurement differs
from the prediction by more than 2 dB or if T=16 is not more accurate than T=32; at orbital range other error
sources dominate and both tile sizes must reach -55 dB. It also fails if the oversampling error falls by less than
10 dB per doubling."""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, ffbp2

bad = []
t_start = time.perf_counter()


def err_db(img, ref):
    g = np.vdot(img.ravel(), ref.ravel()) / np.vdot(img.ravel(), img.ravel())
    return 10 * np.log10(np.sum(np.abs(g * img - ref) ** 2) / np.sum(np.abs(ref) ** 2))


n, sp = 128, 0.5
e1, e2 = (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)
pts = fastsar.plane_points(n, n, sp, sp, e1, e2)
print('final tile size: T=16 and T=32 against exact backprojection')
for r0 in (1e3, 4e3, 16e3, 600e3):
    rng = np.random.default_rng(7)
    col = sim.make_collect(res=0.5, scene=n * sp, r0=r0)
    tg = np.stack([rng.uniform(-25, 25, 30), rng.uniform(-25, 25, 30), np.zeros(30)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    ref = fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='cpu', upsample=16)
    row = []
    for T in (16, 32):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            img = fastsar.form_image(S, col.ant, col.fmin, col.df, n, n, sp, sp, e1, e2, backend='cpu', T=T)
        pred = ffbp2.final_phase_error(col.ant, col.fmin + col.K * col.df, n, n, sp, sp, np.array(e1), np.array(e2), T)
        meas = err_db(img, ref)
        row.append((T, pred, meas))
        if (pred > -55.0 and abs(meas - pred) > 2.0) or meas > max(pred + 2.0, -55.0):
            bad.append(f'{r0 / 1e3:.0f} km T={T}: measured {meas:.1f} dB, predicted {pred:.1f} dB')
    if row[1][1] > -55.0 and row[0][2] > row[1][2]:
        bad.append(f'{r0 / 1e3:.0f} km: T=16 less accurate than T=32')
    print(f'  {r0 / 1e3:5.0f} km ({len(S)} pulses): ' +
          '; '.join(f'T={T} measured {m:6.1f} dB, predicted {p:6.1f} dB' for T, p, m in row))

print('\nexact backprojection: oversampling against upsample=128 at 4 km')
rng = np.random.default_rng(3)
col = sim.make_collect(res=0.5, scene=40.0, r0=4e3)
tg = np.stack([rng.uniform(-15, 15, 12), rng.uniform(-15, 15, 12), np.zeros(12)], 1)
S = sim.simulate_brute(col, tg, rng.standard_normal(12) + 1j * rng.standard_normal(12))
pts = fastsar.plane_points(80, 80, 0.5, 0.5, e1, e2)
ref = fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='cpu', upsample=128)
prev = None
for u in (2, 8, 32):
    e = err_db(fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='cpu', upsample=u), ref)
    print(f'  upsample {u:3d}: {e:6.1f} dB')
    if prev is not None and e > prev - 20.0:          # two doublings
        bad.append(f'upsample {u}: {e:.1f} dB, less than 10 dB per doubling below {prev:.1f} dB')
    prev = e

print(f'\n{time.perf_counter() - t_start:.0f} s')
if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
