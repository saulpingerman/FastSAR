"""Phase gradient autofocus on a simulated spotlight scene (point targets over clutter): a known phase error per
pulse (quadratic, higher-order polynomial, low-pass random) is injected into the phase history, and the image
after autofocus is compared with the image of the error-free history. Also checks that a focused image is left
alone."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, autofocus as af

backend = sys.argv[1] if len(sys.argv) > 1 else 'jax'
rng = np.random.default_rng(3)
col = sim.make_collect(res=0.6, scene=64.0, r0=5e3)            # 0.6 m resolution on 0.5 m pixels
n_pt, n_cl = 40, 3000
pos = np.concatenate([np.stack([rng.uniform(-26, 26, n_pt), rng.uniform(-26, 26, n_pt), np.zeros(n_pt)], 1),
                      np.stack([rng.uniform(-30, 30, n_cl), rng.uniform(-30, 30, n_cl), np.zeros(n_cl)], 1)])
amp = np.concatenate([rng.uniform(2, 10, n_pt) * np.exp(2j * np.pi * rng.random(n_pt)),
                      0.15 * (rng.standard_normal(n_cl) + 1j * rng.standard_normal(n_cl))])
S = sum(sim.simulate_brute(col, pos[i:i + 250], amp[i:i + 250]) for i in range(0, len(pos), 250)).astype(np.complex64)
grid = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))   # track along y, looking along +x
fk = dict(backend=backend, T=16, levels=2)
P = col.Np
p = np.arange(P, dtype=np.float64)
t = 2 * p / (P - 1) - 1


def detrend(x):
    return af._detrend(x, p)


def rel_db(a, b):
    g = np.vdot(a, b) / np.vdot(a, a)                                  # best complex gain
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


former = fastsar.ImageFormer(col.ant, col.fmin, col.df, col.K, **grid, **fk)
ref = former(S)
walk = np.convolve(np.cumsum(rng.standard_normal(P + 40)), np.hanning(41) / np.hanning(41).sum(), 'valid')[:P]
errs = {'quadratic': 8 * t ** 2, 'polynomial': 6 * t ** 3 + 5 * t ** 4 - 4 * t ** 5, 'random walk': walk}
print(f'{P} pulses, {col.K} samples, {n_pt} points over {n_cl} clutter scatterers, backend {backend}')
for name, e in errs.items():
    e = detrend(e)
    if name == 'random walk':
        e *= 3 / np.abs(e).max()
    t0 = time.perf_counter()
    img, phi = af.autofocus(S * np.exp(1j * e)[:, None].astype(np.complex64), col.ant, col.fmin, col.df, **grid, **fk)
    res = detrend(phi - e)
    before, after = rel_db(former(S * np.exp(1j * e)[:, None].astype(np.complex64)), ref), rel_db(img, ref)
    print(f'{name:12s} peak {np.abs(e).max():.1f} rad, rms {e.std():.2f} rad: image {before:6.1f} dB -> {after:6.1f} dB, '
          f'residual rms {res.std():.3f} rad ({time.perf_counter() - t0:.1f} s)')
    assert res.std() < 0.1 and after < -25, (name, res.std(), after)

# a focused image stays focused (what PGA does change is its estimation error on this scene)
ramp = np.exp(-1j * af.deramp_phase(col.ant, col.fmin, col.df, col.K, **grid))
img, _, hist = af.pga(ref * ramp)
img2, phi = af.autofocus(S, col.ant, col.fmin, col.df, **grid, **fk)
print(f'focused input: pga changes the image by {rel_db(img, ref * ramp):.1f} dB in {len(hist)} iterations; '
      f'autofocus by {rel_db(img2, ref):.1f} dB with phase rms {detrend(phi).std():.3f} rad')
assert rel_db(img, ref * ramp) < -25 and rel_db(img2, ref) < -25 and detrend(phi).std() < 0.1
pts = former(sim.simulate_brute(col, pos[:n_pt], amp[:n_pt]).astype(np.complex64)) * ramp
img, _, hist = af.pga(pts)
print(f'focused points without clutter: pga changes the image by {rel_db(img, pts):.1f} dB in {len(hist)} iterations')
assert rel_db(img, pts) < -30
print('ok')
