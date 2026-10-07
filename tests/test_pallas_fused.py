"""Toy-size check of the fused rotate-and-decimate kernel against the dense XLA path (interpret mode runs on CPU).

  uv run --with jax --with numpy python tests/test_pallas_fused.py
"""
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
from fastsar.pallas_ffbp import band_blocks, pad_columns, fused_rotate_dec_k

jax.config.update('jax_platform_name', 'cpu')
bad = []                  # thresholds about 5 dB above the values measured on the CPU (interpret mode) when they were set


def check(label, e, lim):
    if not e < lim:
        bad.append(f'{label}: {e:.1f} dB (limit {lim} dB)')
    return e


rng = np.random.default_rng(0)
P, K, D, L = 64, 1536, 6, 44
Ko = (K + D - 1) // D
# a banded decimation matrix like the plan's: Kaiser taps, output j from inputs D j + r - pl
taps = np.kaiser(L, 8.0) * np.sinc((np.arange(L) - (L - 1) / 2) / D) / D
pl_ = L // 2 - D // 2
Fk = np.zeros((K, Ko))
for j in range(Ko):
    for r in range(L):
        i = D * j + r - pl_
        if 0 <= i < K:
            Fk[i, j] = taps[r]
S = (rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K))).astype(np.complex64)
c0 = rng.uniform(-1e4, 1e4, P).astype(np.float32)
slope = rng.uniform(-0.3, 0.3, P).astype(np.float32)
kc = (K - 1) / 2.0
# reference: rotate in float64 then dense product
kk = np.arange(K) - kc
cyc = c0[:, None].astype(np.float64) + kk[None, :] * slope[:, None].astype(np.float64)
ang = (cyc - np.round(cyc)) * 2 * np.pi
Z = S.astype(np.complex128) * np.exp(1j * ang)
Yref = Z @ Fk
band = band_blocks(Fk, D, kob=256)
Sre = pad_columns(jnp.asarray(S.real), band)
Sim = pad_columns(jnp.asarray(S.imag), band)
for passes in (1, 3):
    yr, yi = fused_rotate_dec_k(Sre, Sim, jnp.asarray(c0), jnp.asarray(slope), band, kc, pb=32, chunk=512, passes=passes, interpret=True)
    Y = np.asarray(yr)[:, :Ko] + 1j * np.asarray(yi)[:, :Ko]
    err = 10 * np.log10(np.sum(np.abs(Y - Yref) ** 2) / np.sum(np.abs(Yref) ** 2))
    print(f'passes {passes}: error {check(f"kernel1 {passes}", err, -45 if passes == 1 else -55):.1f} dB relative to float64')

# second-generation kernel: two parents, three children each, against the same float64 reference per child
from fastsar.pallas_ffbp import band_blocks2, fused_rotate_dec_k2
band2 = band_blocks2(Fk, D)
print('band2', {k: band2[k] for k in ('kob', 'win', 'stride', 'nb', 'Kpad', 'chunk')})
N, nc = 2, 3
Ss = [(rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K))).astype(np.complex64) for _ in range(N)]
c0s = rng.uniform(-1e4, 1e4, (N, nc, P)).astype(np.float32)
sls = rng.uniform(-0.3, 0.3, (N, nc, P)).astype(np.float32)
Sre2 = jnp.stack([pad_columns(jnp.asarray(s.real), band2) for s in Ss]); Sim2 = jnp.stack([pad_columns(jnp.asarray(s.imag), band2) for s in Ss])
for passes in (1, 3):
    yr, yi = fused_rotate_dec_k2(Sre2, Sim2, jnp.asarray(c0s), jnp.asarray(sls), band2, kc, pb=32, passes=passes, interpret=True)
    num = den = 0.0
    for n in range(N):
        for c in range(nc):
            cyc = c0s[n, c][:, None].astype(np.float64) + kk[None, :] * sls[n, c][:, None].astype(np.float64)
            ref = (Ss[n].astype(np.complex128) * np.exp(1j * (cyc - np.round(cyc)) * 2 * np.pi)) @ Fk
            Y = np.asarray(yr)[n, c, :, :Ko] + 1j * np.asarray(yi)[n, c, :, :Ko]
            num += np.sum(np.abs(Y - ref) ** 2); den += np.sum(np.abs(ref) ** 2)
    print(f'kernel2 passes {passes}: error {check(f"kernel2 {passes}", 10 * np.log10(num / den), -45 if passes == 1 else -55):.1f} dB')

# multi-block form of the second kernel (forced small output blocks)
band3 = band_blocks2(Fk, D, kob=128)
print('band3', {k: band3[k] for k in ('kob', 'win', 'stride', 'nb', 'Kpad', 'chunk')})
Sre3 = jnp.stack([pad_columns(jnp.asarray(s.real), band3) for s in Ss]); Sim3 = jnp.stack([pad_columns(jnp.asarray(s.imag), band3) for s in Ss])
yr, yi = fused_rotate_dec_k2(Sre3, Sim3, jnp.asarray(c0s), jnp.asarray(sls), band3, kc, pb=32, passes=1, interpret=True)
num = den = 0.0
for n in range(N):
    for c in range(nc):
        cyc = c0s[n, c][:, None].astype(np.float64) + kk[None, :] * sls[n, c][:, None].astype(np.float64)
        ref = (Ss[n].astype(np.complex128) * np.exp(1j * (cyc - np.round(cyc)) * 2 * np.pi)) @ Fk
        Y = np.asarray(yr)[n, c, :, :Ko] + 1j * np.asarray(yi)[n, c, :, :Ko]
        num += np.sum(np.abs(Y - ref) ** 2); den += np.sum(np.abs(ref) ** 2)
print(f'kernel2 multi-block: error {check("kernel2 multi-block", 10 * np.log10(num / den), -45):.1f} dB')

# third kernel: precomputed coarse tables, with and without the fused pulse decimation
from fastsar.pallas_ffbp import fused_rotate_dec_k3
Dp, Po = 2, 40
tp_ = np.kaiser(16, 8.0) * np.sinc((np.arange(16) - 7.5) / Dp) / Dp
Fp = np.zeros((P, Po))
for j in range(Po):
    for r in range(16):
        i = Dp * j + r - 6
        if 0 <= i < P:
            Fp[i, j] = tp_[r]
for fused in (False, True):
    for passes in (1, 3):
        yr, yi = fused_rotate_dec_k3(Sre2, Sim2, jnp.asarray(c0s), jnp.asarray(sls), band2, kc, pb=32, passes=passes,
                                     FpT=jnp.asarray(Fp.T, jnp.float32) if fused else None, interpret=True)
        num = den = 0.0
        for n in range(N):
            for c in range(nc):
                cyc = c0s[n, c][:, None].astype(np.float64) + kk[None, :] * sls[n, c][:, None].astype(np.float64)
                ref = (Ss[n].astype(np.complex128) * np.exp(1j * (cyc - np.round(cyc)) * 2 * np.pi)) @ Fk
                if fused:
                    ref = Fp.T @ ref
                    Y = np.asarray(yr)[n, c, :Po, :Ko] + 1j * np.asarray(yi)[n, c, :Po, :Ko]
                else:
                    Y = np.asarray(yr)[n, c, :, :Ko] + 1j * np.asarray(yi)[n, c, :, :Ko]
                num += np.sum(np.abs(Y - ref) ** 2); den += np.sum(np.abs(ref) ** 2)
        print(f'kernel3 fused_p={fused} passes {passes}: error {check(f"kernel3 {fused} {passes}", 10 * np.log10(num / den), -44 if passes == 1 else -55):.1f} dB')

if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
