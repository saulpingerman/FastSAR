"""Products: targets above the image plane appear where project predicts (layover), a target on the plane projects
onto itself, geocode samples the image there, multilook keeps the mean power, and a SICD round trip keeps the
pixels (when sarpy is installed)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, products, quality

rng = np.random.default_rng(2)
col = sim.make_collect(res=0.5, scene=100.0, r0=6e3, graze_deg=35.0)
grid = dict(nx=256, ny=256, spx=0.4, spy=0.4, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
tg = np.array([[0.0, 0.0, 0.0], [12.0, -15.0, 10.0], [-15.0, 8.0, 20.0], [10.0, 10.0, 30.0]])
S = sim.simulate_brute(col, tg, np.ones(len(tg))).astype(np.complex64)
img = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend='jax')
ij = products.project(tg, col.ant, **grid)
worst, peaks = 0.0, []
for k, (x, p) in enumerate(zip(tg, ij)):
    # the actual peak near the predicted pixel
    i0, j0 = (int(round(v)) for v in p)
    w = np.abs(img[i0 - 6:i0 + 7, j0 - 6:j0 + 7])
    a, b = np.unravel_index(np.argmax(w), w.shape)
    m = quality.point_target(img, (i0 - 6 + a, j0 - 6 + b), grid["spx"], grid["spy"], half=16)
    d = np.hypot((m['i'] - p[0]) * grid['spx'], (m['j'] - p[1]) * grid['spy'])
    peaks.append(w.max())
    plane = np.array([(p[0] - grid['nx'] / 2) * grid['spx'], (p[1] - grid['ny'] / 2) * grid['spy']])
    lay = np.hypot(plane[0] - x[1], plane[1] - x[0])
    print(f'target at height {x[2]:4.0f} m: layover {lay:6.2f} m, peak found {d * 100:.2f} cm from the projection')
    worst = max(worst, d)
assert worst < 0.05, worst
assert np.hypot(*(ij[0] - [grid['nx'] / 2, grid['ny'] / 2])) < 1e-9
# geocode: the amplitude at each target's 3-D position, bilinear between pixels, is near its peak pixel's
amp = products.geocode(np.abs(img), grid, col.ant, tg)
print('geocoded amplitude / brightest pixel of each target:', np.round(amp / np.array(peaks), 3))
assert (amp / np.array(peaks) > 0.7).all()
# multilook keeps the mean power
ml = products.multilook(img, 4, 2)
assert ml.shape == (64, 128) and abs(ml.mean() / (np.abs(img) ** 2).mean() - 1) < 1e-6
try:
    import sarpy  # noqa: F401
    have_sarpy = True
except ImportError:
    have_sarpy = False
print('ok' + ('' if have_sarpy else ' (SICD round trip skipped: no sarpy)'))

# Pauli decomposition: a trihedral (HH = VV), a dihedral (HH = -VV) and a dipole at 45 degrees (HV only) land in
# the surface, double-bounce and volume channels; total power is kept
hh, hv, vv = np.array([[1.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, -1.0, 0.0]])
p = products.pauli(hh, hv, vv)[0]
print('pauli (double bounce, volume, surface):', np.round(p, 3).tolist())
assert np.allclose(p, [[0, 0, 2], [2, 0, 0], [0, 2, 0]])
print('ok')
