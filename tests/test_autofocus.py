"""Phase gradient autofocus on a simulated spotlight scene (point targets over clutter): a known phase error per
pulse (quadratic, higher-order polynomial, low-pass random) is injected into the phase history, and the image
after autofocus is compared with the image of the error-free history. Also checks that a focused image is left
alone. Runs on the JAX backend (the scripts' default) and on every other backend this machine has."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import sim, autofocus as af

GRID = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))   # track along y, looking along +x
N_PT, N_CL = 40, 3000


def rel_db(a, b):
    g = np.vdot(a, b) / np.vdot(a, a)                                  # best complex gain
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(3)
    col = sim.make_collect(res=0.6, scene=64.0, r0=5e3)            # 0.6 m resolution on 0.5 m pixels
    pos = np.concatenate([np.stack([rng.uniform(-26, 26, N_PT), rng.uniform(-26, 26, N_PT), np.zeros(N_PT)], 1),
                          np.stack([rng.uniform(-30, 30, N_CL), rng.uniform(-30, 30, N_CL), np.zeros(N_CL)], 1)])
    amp = np.concatenate([rng.uniform(2, 10, N_PT) * np.exp(2j * np.pi * rng.random(N_PT)),
                          0.15 * (rng.standard_normal(N_CL) + 1j * rng.standard_normal(N_CL))])
    S = sum(sim.simulate_brute(col, pos[i:i + 250], amp[i:i + 250]) for i in range(0, len(pos), 250)).astype(np.complex64)
    P = col.Np
    p = np.arange(P, dtype=np.float64)
    t = 2 * p / (P - 1) - 1
    walk = np.convolve(np.cumsum(rng.standard_normal(P + 40)), np.hanning(41) / np.hanning(41).sum(), 'valid')[:P]
    errs = {'quadratic': 8 * t ** 2, 'polynomial': 6 * t ** 3 + 5 * t ** 4 - 4 * t ** 5, 'random walk': walk}
    print(f'{P} pulses, {col.K} samples, {N_PT} points over {N_CL} clutter scatterers')
    return SimpleNamespace(col=col, pos=pos, amp=amp, S=S, p=p, errs=errs)


@pytest.fixture(scope='module', params=['jax', 'cpu', pytest.param('cuda', marks=pytest.mark.cuda),
                                        pytest.param('tpu', marks=pytest.mark.tpu)])
def focused(request, scene):
    """The backend, its keyword arguments, a former for the scene and the image of the error-free history."""
    fk = dict(backend=request.param, T=16, levels=2)
    c = scene.col
    former = fastsar.ImageFormer(c.ant, c.fmin, c.df, c.K, **GRID, **fk)
    return SimpleNamespace(fk=fk, former=former, ref=former(scene.S))


@pytest.mark.parametrize('name', ['quadratic', 'polynomial', 'random walk'])
def test_phase_error_removed(scene, focused, name):
    c, S, fk, former, ref = scene.col, scene.S, focused.fk, focused.former, focused.ref
    e = af._detrend(np.array(scene.errs[name]), scene.p)
    if name == 'random walk':
        e *= 3 / np.abs(e).max()
    t0 = time.perf_counter()
    img, phi = af.autofocus(S * np.exp(1j * e)[:, None].astype(np.complex64), c.ant, c.fmin, c.df, **GRID, **fk)
    res = af._detrend(phi - e, scene.p)
    before, after = rel_db(former(S * np.exp(1j * e)[:, None].astype(np.complex64)), ref), rel_db(img, ref)
    print(f'{fk["backend"]}: {name:12s} peak {np.abs(e).max():.1f} rad, rms {e.std():.2f} rad: image {before:6.1f} dB -> '
          f'{after:6.1f} dB, residual rms {res.std():.3f} rad ({time.perf_counter() - t0:.1f} s)')
    assert res.std() < 0.1 and after < -25, (name, res.std(), after)


def test_focused_image_unchanged(scene, focused):
    """A focused image stays focused (what PGA does change is its estimation error on this scene)."""
    c, S, fk, former, ref = scene.col, scene.S, focused.fk, focused.former, focused.ref
    ramp = np.exp(-1j * af.deramp_phase(c.ant, c.fmin, c.df, c.K, **GRID))
    img, _, hist = af.pga(ref * ramp)
    img2, phi = af.autofocus(S, c.ant, c.fmin, c.df, **GRID, **fk)
    print(f'{fk["backend"]}: focused input: pga changes the image by {rel_db(img, ref * ramp):.1f} dB in {len(hist)} '
          f'iterations; autofocus by {rel_db(img2, ref):.1f} dB with phase rms {af._detrend(phi, scene.p).std():.3f} rad')
    assert rel_db(img, ref * ramp) < -25 and rel_db(img2, ref) < -25 and af._detrend(phi, scene.p).std() < 0.1
    pts = former(sim.simulate_brute(c, scene.pos[:N_PT], scene.amp[:N_PT]).astype(np.complex64)) * ramp
    img, _, hist = af.pga(pts)
    print(f'focused points without clutter: pga changes the image by {rel_db(img, pts):.1f} dB in {len(hist)} iterations')
    assert rel_db(img, pts) < -30
