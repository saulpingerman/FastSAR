"""Wide-angle and circular apertures: factorized backprojection against exact backprojection for apertures of 10 to
360 degrees around a scene, each imaged at its own resolution (lambda / (4 sin(span / 2)), lambda / 4 for the full
circle), where polar format does not apply."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim

C = 299792458.0
rng = np.random.default_rng(4)
fc, B, r0, graze, n = 9.6e9, 1.5e9, 1500.0, np.deg2rad(40), 256
lam = C / fc
for span_deg in (10.0, 45.0, 120.0, 360.0):
    span = np.deg2rad(span_deg)
    res = lam / (4 * np.sin(min(span, np.pi) / 2))
    sp = 0.8 * res
    scene = n * sp
    df = C / (2 * 1.25 * np.sqrt(2) * scene)
    K = max(16, int(B / df) // 2 * 2)
    P = int(np.ceil(span * 2 * 1.25 * np.sqrt(2) * scene / lam)) // 2 * 2
    th = np.linspace(-span / 2, span / 2, P, endpoint=span_deg < 360)
    ant = np.stack([-r0 * np.cos(graze) * np.cos(th), r0 * np.cos(graze) * np.sin(th), np.full(P, r0 * np.sin(graze))], 1)
    col = sim.Collect(fmin=fc - B / 2, df=df, K=K, ant=ant, res=sp)
    tg = np.stack([rng.uniform(-0.35, 0.35, 10) * scene, rng.uniform(-0.35, 0.35, 10) * scene, np.zeros(10)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(10) + 1j)
    grid = dict(nx=n, ny=n, spx=sp, spy=sp, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
    t = time.perf_counter()
    ff = fastsar.form_image(S, ant, col.fmin, col.df, **grid, backend='cpu')
    tf = time.perf_counter() - t
    t = time.perf_counter()
    ex = fastsar.backproject(S, ant, col.fmin, col.df, fastsar.plane_points(**grid), backend='cpu', upsample=16)
    te = time.perf_counter() - t
    g = np.vdot(ff.ravel(), ex.ravel()) / np.vdot(ff.ravel(), ff.ravel())
    e = 10 * np.log10(np.sum(np.abs(g * ff - ex) ** 2) / np.sum(np.abs(ex) ** 2))
    print(f'aperture {span_deg:3.0f} deg: {P} pulses x {K} samples, {n}^2 pixels of {sp * 1000:.1f} mm; '
          f'ffbp vs exact {e:.1f} dB ({tf:.1f} s against {te:.1f} s)', flush=True)
    assert e < -45, (span_deg, e)
print('ok')
