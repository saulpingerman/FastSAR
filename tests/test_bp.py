"""Exact backprojection onto points: each backend against a float64 NumPy reference that range compresses at 64x
and interpolates linearly, monostatic and bistatic, at airborne and orbital range; and agreement with factorized
backprojection on its own grid."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import sim, bp
from fastsar.api import _window

C = 299792458.0
BACKENDS = ['cpu', 'jax', pytest.param('cuda', marks=pytest.mark.cuda)]
GRID = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))


def rel_db(a, b):
    g = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


def reference(S, tx, rcv, fmin, df, pts, os_=64):
    P, K = S.shape
    nfft = 1 << int(np.ceil(np.log2(os_ * K)))
    rc = bp.range_compress(S.astype(np.complex128), nfft)
    out = np.zeros(len(pts), complex)
    for p in range(P):
        dR = np.linalg.norm(pts - tx[p], axis=1) - np.linalg.norm(tx[p])
        if rcv is not None:
            dR = 0.5 * (dR + np.linalg.norm(pts - rcv[p], axis=1) - np.linalg.norm(rcv[p]))
        t = dR * 2 * df * nfft / C + nfft // 2
        i = np.floor(t).astype(int); w = t - i
        out += (rc[p, i] * (1 - w) + rc[p, i + 1] * w) * np.exp(2j * np.pi * 2 * (fmin + (K // 2) * df) / C * dR)
    return out


@pytest.fixture(scope='module')
def draws():
    """The random scenes of every check, drawn in the order of the original script from one generator."""
    rng = np.random.default_rng(3)
    cases = []
    for r0 in (5e3, 600e3):
        tg = np.stack([rng.uniform(-15, 15, 12), rng.uniform(-15, 15, 12), rng.uniform(-3, 3, 12)], 1)
        amp = rng.standard_normal(12) + 1j * rng.standard_normal(12)
        for bistatic in (False, True):
            pts = np.concatenate([tg + rng.normal(0, 0.3, tg.shape), rng.uniform(-20, 20, (500, 3))])
            cases.append((r0, bistatic, tg, amp, pts))
    tg30 = np.stack([rng.uniform(-25, 25, 30), rng.uniform(-25, 25, 30), np.zeros(30)], 1)
    amp30 = rng.standard_normal(30) + 1j * rng.standard_normal(30)
    amp8 = rng.standard_normal(8) + 1j
    return SimpleNamespace(cases=cases, tg30=tg30, amp30=amp30, amp8=amp8)


@pytest.fixture(scope='module')
def point_cases(draws):
    """Monostatic and bistatic phase histories at 5 and 600 km with their float64 references."""
    out = []
    for r0, bistatic, tg, amp, pts in draws.cases:
        col = sim.make_collect(res=0.5, scene=40.0, r0=r0)
        tx = col.ant
        rcv = None
        if bistatic:
            rcv = col.ant + np.array([0.0, 0.0, 0.02 * r0]) + np.array([0.0, 1.0, 0.0]) * np.linspace(-0.01, 0.01, len(col.ant))[:, None] * r0
        # phase history for the actual geometry (bistatic delays in the CPHD convention)
        f = col.freqs
        S = np.zeros((col.Np, col.K), complex)
        for x, a in zip(tg, amp):
            dR = np.linalg.norm(x - tx, axis=1) - np.linalg.norm(tx, axis=1)
            if rcv is not None:
                dR = 0.5 * (dR + np.linalg.norm(x - rcv, axis=1) - np.linalg.norm(rcv, axis=1))
            S += a * np.exp(-4j * np.pi * f[None, :] / C * dR[:, None])
        wp, wk = _window(*S.shape)
        ref = reference(S * wp[:, None] * wk[None, :], tx, rcv, col.fmin, col.df, pts)
        out.append(SimpleNamespace(r0=r0, bistatic=bistatic, col=col, tx=tx, rcv=rcv, S=S, pts=pts, ref=ref))
    return out


@pytest.mark.parametrize('b', BACKENDS)
def test_against_reference(point_cases, b):
    bad = []
    for c in point_cases:
        img = fastsar.backproject(c.S, c.tx, c.col.fmin, c.col.df, c.pts, rcv=c.rcv, backend=b, upsample=16)
        e = rel_db(img, c.ref)
        print(f'{b}: range {c.r0 / 1e3:.0f} km, {"bistatic" if c.bistatic else "monostatic"}: {e:.1f} dB', flush=True)
        if not e < -50:
            bad.append((c.r0, c.bistatic, b, e))
    assert not bad, bad


@pytest.fixture(scope='module')
def ffbp_exact(draws):
    """Factorized and exact backprojection on the same grid."""
    col = sim.make_collect(res=0.5, scene=60.0, r0=20e3)
    S = sim.simulate_brute(col, draws.tg30, draws.amp30).astype(np.complex64)
    ff = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, backend='cpu')
    ex = fastsar.backproject(S, col.ant, col.fmin, col.df, fastsar.plane_points(**GRID), backend='cpu', upsample=16)
    return ff, ex


def test_ffbp_vs_exact(ffbp_exact):
    ff, ex = ffbp_exact
    e = rel_db(ff, ex)
    print(f'ffbp vs exact backprojection on the ffbp grid: {e:.1f} dB')
    assert e < -40, e


@pytest.fixture(scope='module')
def moving(draws):
    """A moving reference point: each pulse compensated to its own point along the track."""
    col = sim.make_collect(res=0.5, scene=60.0, r0=8e3)
    srp = np.zeros((col.Np, 3)); srp[:, 1] = np.linspace(-20, 20, col.Np)
    f = col.freqs
    S0 = np.zeros((col.Np, col.K), complex); S1 = np.zeros_like(S0)
    for x, a in zip(draws.tg30[:8], draws.amp8):
        r = np.linalg.norm(x - col.ant, axis=1)
        S0 += a * np.exp(-4j * np.pi * f[None, :] / C * (r - np.linalg.norm(col.ant, axis=1))[:, None])
        S1 += a * np.exp(-4j * np.pi * f[None, :] / C * (r - np.linalg.norm(col.ant - srp, axis=1))[:, None])
    pts = fastsar.plane_points(64, 64, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    fixed = fastsar.backproject(S0, col.ant, col.fmin, col.df, pts, backend='cpu')
    return SimpleNamespace(col=col, srp=srp, S1=S1, pts=pts, fixed=fixed)


@pytest.mark.parametrize('b', BACKENDS)
def test_moving_reference(moving, b):
    """Backprojection with the per-pulse reference ranges gives the image of the fixed-reference data."""
    col = moving.col
    e = rel_db(fastsar.backproject(moving.S1, col.ant, col.fmin, col.df, moving.pts,
                                   ref=np.linalg.norm(col.ant - moving.srp, axis=1), backend=b), moving.fixed)
    print(f'{b}: moving reference point vs fixed: {e:.1f} dB')
    assert e < -55, e


def test_gain(ffbp_exact):
    """backproject has the gain of form_image (the plain sum over pulses and samples)."""
    ff, ex = ffbp_exact
    g = np.vdot(ff.ravel(), ex.ravel()) / np.vdot(ff.ravel(), ff.ravel())
    print(f'backproject / form_image gain: {abs(g):.4f}')
    assert abs(abs(g) - 1) < 0.02, g


def test_range_periodic():
    """A frequency-domain phase history is periodic in range: a scatterer 0.7 of the unambiguous range c / (2 df)
    from the reference range focuses as well as one at the reference point."""
    col = sim.make_collect(res=1.0, scene=30.0, r0=5e3)
    L = C / (2 * col.df)
    for off in (0.0, 0.7 * L):
        x = np.array([[off * np.cos(np.deg2rad(30)), 0.0, 0.0]])     # along ground range; slant offset about 0.7 L
        S = sim.simulate_brute(col, x, np.ones(1))
        v = fastsar.backproject(S, col.ant, col.fmin, col.df, x, backend='cpu')[0]
        if off == 0.0:
            v0 = v
        print(f'scatterer {off:6.1f} m from the reference in ground range: |image| / |image at the reference| = {abs(v) / abs(v0):.4f}')
    assert abs(abs(v) / abs(v0) - 1) < 0.02
