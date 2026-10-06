"""Exact backprojection onto points: each backend against a float64 NumPy reference that range compresses at 64x
and interpolates linearly, monostatic and bistatic, at airborne and orbital range; and agreement with factorized
backprojection on its own grid."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, bp

C = 299792458.0


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


backends = [b for b in ('cpu', 'jax', 'cuda') if b == 'jax' or b in fastsar.available_backends()]
print('backends:', backends)
rng = np.random.default_rng(3)
for r0 in (5e3, 600e3):
    col = sim.make_collect(res=0.5, scene=40.0, r0=r0)
    tg = np.stack([rng.uniform(-15, 15, 12), rng.uniform(-15, 15, 12), rng.uniform(-3, 3, 12)], 1)
    amp = rng.standard_normal(12) + 1j * rng.standard_normal(12)
    for bistatic in (False, True):
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
        pts = np.concatenate([tg + rng.normal(0, 0.3, tg.shape), rng.uniform(-20, 20, (500, 3))])
        from fastsar.api import _window
        wp, wk = _window(*S.shape)
        ref = reference(S * wp[:, None] * wk[None, :], tx, rcv, col.fmin, col.df, pts)
        line = []
        for b in backends:
            t = time.perf_counter()
            img = fastsar.backproject(S, tx, col.fmin, col.df, pts, rcv=rcv, backend=b, upsample=16)
            e = rel_db(img, ref)
            line.append(f'{b} {e:.1f} dB')
            assert e < -50, (r0, bistatic, b, e)
        print(f'range {r0/1e3:.0f} km, {"bistatic" if bistatic else "monostatic"}: ' + ', '.join(line), flush=True)

# factorized backprojection against exact backprojection on the same grid
col = sim.make_collect(res=0.5, scene=60.0, r0=20e3)
tg = np.stack([rng.uniform(-25, 25, 30), rng.uniform(-25, 25, 30), np.zeros(30)], 1)
S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
grid = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
ff = fastsar.form_image(S, col.ant, col.fmin, col.df, **grid, backend='cpu')
ex = fastsar.backproject(S, col.ant, col.fmin, col.df, fastsar.plane_points(**grid), backend='cpu', upsample=16)
e = rel_db(ff, ex)
print(f'ffbp vs exact backprojection on the ffbp grid: {e:.1f} dB')
assert e < -40, e
print('ok')

# a moving reference point: each pulse compensated to its own point along the track; backprojection with the
# per-pulse reference ranges gives the image of the fixed-reference data
col = sim.make_collect(res=0.5, scene=60.0, r0=8e3)
srp = np.zeros((col.Np, 3)); srp[:, 1] = np.linspace(-20, 20, col.Np)
f = col.freqs
S0 = np.zeros((col.Np, col.K), complex); S1 = np.zeros_like(S0)
for x, a in zip(tg[:8], rng.standard_normal(8) + 1j):
    r = np.linalg.norm(x - col.ant, axis=1)
    S0 += a * np.exp(-4j * np.pi * f[None, :] / C * (r - np.linalg.norm(col.ant, axis=1))[:, None])
    S1 += a * np.exp(-4j * np.pi * f[None, :] / C * (r - np.linalg.norm(col.ant - srp, axis=1))[:, None])
pts = fastsar.plane_points(64, 64, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
for b in backends:
    e = rel_db(fastsar.backproject(S1, col.ant, col.fmin, col.df, pts, ref=np.linalg.norm(col.ant - srp, axis=1), backend=b),
               fastsar.backproject(S0, col.ant, col.fmin, col.df, pts, backend='cpu'))
    print(f'{b}: moving reference point vs fixed: {e:.1f} dB')
    assert e < -55, e
print('ok')

# gain: backproject has the gain of form_image (the plain sum over pulses and samples)
g = np.vdot(ff.ravel(), ex.ravel()) / np.vdot(ff.ravel(), ff.ravel())
print(f'backproject / form_image gain: {abs(g):.4f}')
assert abs(abs(g) - 1) < 0.02, g
print('ok')
