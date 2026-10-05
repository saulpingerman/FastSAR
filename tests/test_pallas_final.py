"""Toy check of the fused final-stage kernel against a float64 evaluation (interpret mode, CPU)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
from sarform.pallas_ffbp import fused_final
rng = np.random.default_rng(0)
B, Pf, Qf, T = 3, 16, 79, 32
Qpad = 128
a0, a1 = 2 * 9.6e9 / 3e8, 2 * 1.2e6 / 3e8                      # cycles per metre and per metre per sample
data = (rng.standard_normal((B, Pf, Qf)) + 1j * rng.standard_normal((B, Pf, Qf)))
ux = rng.uniform(-0.3, 0.3, (B, Pf)); uy = rng.uniform(0.5, 0.9, (B, Pf))
dl = (np.arange(T) - (T - 1) / 2) * 0.5
gx = dl[None, :, None] * ux[:, None, :]; gy = dl[None, :, None] * uy[:, None, :]          # [B, T, Pf]
q = np.arange(Qf)
qv = a0 + a1 * q
# reference
ref = np.zeros((B, T, T), np.complex128)
for b in range(B):
    A = data[b][None] * np.exp(-2j * np.pi * gx[b][:, :, None] * qv[None, None, :])        # [T, Pf, Qf]
    Bm = np.exp(-2j * np.pi * gy[b][:, :, None] * qv[None, None, :])
    ref[b] = A.reshape(T, -1) @ Bm.reshape(T, -1).T
tre = np.zeros((B, Pf, Qpad), np.float32); tim = np.zeros_like(tre)
tre[..., :Qf] = data.real; tim[..., :Qf] = data.imag
qvp = np.zeros((1, Qpad), np.float32); qvp[0, :Qf] = qv
for passes in (1, 3):
    cre, cim = fused_final(jnp.asarray(tre), jnp.asarray(tim), jnp.asarray(gx, jnp.float32), jnp.asarray(gy, jnp.float32), jnp.asarray(qvp), passes=passes, interpret=True)
    out = np.asarray(cre) + 1j * np.asarray(cim)
    print(f'final kernel passes {passes}: error {10 * np.log10(np.sum(np.abs(out - ref) ** 2) / np.sum(np.abs(ref) ** 2)):.1f} dB')

# the recurrence version: data transposed (q on rows, p on columns)
from sarform.pallas_ffbp import fused_final2
Pl = 128
Qp8 = 8 * -(-Qf // 8)
dT = np.zeros((B, Qp8, Pl), np.complex128); dT[:, :Qf, :Pf] = np.transpose(data, (0, 2, 1))
gxp = np.zeros((B, T, Pl), np.float32); gxp[..., :Pf] = gx
gyp = np.zeros((B, T, Pl), np.float32); gyp[..., :Pf] = gy
for passes in (1, 3):
    cre, cim = fused_final2(jnp.asarray(dT.real, jnp.float32), jnp.asarray(dT.imag, jnp.float32), jnp.asarray(gxp), jnp.asarray(gyp), float(a0), float(a1), Qf, passes=passes, interpret=True)
    out = np.asarray(cre) + 1j * np.asarray(cim)
    print(f'final kernel 2 (recurrence) passes {passes}: error {10 * np.log10(np.sum(np.abs(out - ref) ** 2) / np.sum(np.abs(ref) ** 2)):.1f} dB')

# the stacked-tile version (B = 3 tiles here -> pad to 4)
from sarform.pallas_ffbp import fused_final3
Bp = 4
pad = lambda x: np.concatenate([x, np.zeros((Bp - B,) + x.shape[1:], x.dtype)], 0)
for passes in (1, 3):
    cre, cim = fused_final3(jnp.asarray(pad(dT.real.astype(np.float32))), jnp.asarray(pad(dT.imag.astype(np.float32))), jnp.asarray(pad(gxp)), jnp.asarray(pad(gyp)), float(a0), float(a1), Qf, passes=passes, interpret=True)
    out = (np.asarray(cre) + 1j * np.asarray(cim))[:B]
    print(f'final kernel 3 (stacked tiles) passes {passes}: error {10 * np.log10(np.sum(np.abs(out - ref) ** 2) / np.sum(np.abs(ref) ** 2)):.1f} dB')
