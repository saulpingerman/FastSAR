"""Toy check of the fused final-stage kernel against a float64 evaluation (interpret mode, CPU)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures('jax_cpu')

B, Pf, Qf, T = 3, 16, 79, 32
Qpad = 128
a0, a1 = 2 * 9.6e9 / 3e8, 2 * 1.2e6 / 3e8                      # cycles per meter and per meter per sample


@pytest.fixture(scope='module')
def toy():
    """Random tiles and their float64 reference."""
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((B, Pf, Qf)) + 1j * rng.standard_normal((B, Pf, Qf)))
    ux = rng.uniform(-0.3, 0.3, (B, Pf)); uy = rng.uniform(0.5, 0.9, (B, Pf))
    dl = (np.arange(T) - (T - 1) / 2) * 0.5
    gx = dl[None, :, None] * ux[:, None, :]; gy = dl[None, :, None] * uy[:, None, :]          # [B, T, Pf]
    q = np.arange(Qf)
    qv = a0 + a1 * q
    ref = np.zeros((B, T, T), np.complex128)
    for b in range(B):
        A = data[b][None] * np.exp(-2j * np.pi * gx[b][:, :, None] * qv[None, None, :])        # [T, Pf, Qf]
        Bm = np.exp(-2j * np.pi * gy[b][:, :, None] * qv[None, None, :])
        ref[b] = A.reshape(T, -1) @ Bm.reshape(T, -1).T
    # the transposed layout of the recurrence versions (q on rows, p on columns)
    Pl = 128
    Qp8 = 8 * -(-Qf // 8)
    dT = np.zeros((B, Qp8, Pl), np.complex128); dT[:, :Qf, :Pf] = np.transpose(data, (0, 2, 1))
    gxp = np.zeros((B, T, Pl), np.float32); gxp[..., :Pf] = gx
    gyp = np.zeros((B, T, Pl), np.float32); gyp[..., :Pf] = gy
    return SimpleNamespace(data=data, gx=gx, gy=gy, qv=qv, ref=ref, dT=dT, gxp=gxp, gyp=gyp)


def err_db(out, ref):
    return 10 * np.log10(np.sum(np.abs(out - ref) ** 2) / np.sum(np.abs(ref) ** 2))


# limits about 5 dB above the values measured on the CPU (interpret mode) when they were set
@pytest.mark.parametrize('passes, lim', [(1, -48), (3, -78)])
def test_fused_final(toy, passes, lim):
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import fused_final
    tre = np.zeros((B, Pf, Qpad), np.float32); tim = np.zeros_like(tre)
    tre[..., :Qf] = toy.data.real; tim[..., :Qf] = toy.data.imag
    qvp = np.zeros((1, Qpad), np.float32); qvp[0, :Qf] = toy.qv
    cre, cim = fused_final(jnp.asarray(tre), jnp.asarray(tim), jnp.asarray(toy.gx, jnp.float32), jnp.asarray(toy.gy, jnp.float32), jnp.asarray(qvp), passes=passes, interpret=True)
    e = err_db(np.asarray(cre) + 1j * np.asarray(cim), toy.ref)
    print(f'final kernel passes {passes}: error {e:.1f} dB')
    assert e < lim, f'final {passes}: {e:.1f} dB (limit {lim} dB)'


@pytest.mark.parametrize('passes, lim', [(1, -48), (3, -84)])
def test_fused_final2(toy, passes, lim):
    """The recurrence version: data transposed (q on rows, p on columns)."""
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import fused_final2
    dT = toy.dT
    cre, cim = fused_final2(jnp.asarray(dT.real, jnp.float32), jnp.asarray(dT.imag, jnp.float32), jnp.asarray(toy.gxp), jnp.asarray(toy.gyp), float(a0), float(a1), Qf, passes=passes, interpret=True)
    e = err_db(np.asarray(cre) + 1j * np.asarray(cim), toy.ref)
    print(f'final kernel 2 (recurrence) passes {passes}: error {e:.1f} dB')
    assert e < lim, f'final2 {passes}: {e:.1f} dB (limit {lim} dB)'


@pytest.mark.parametrize('passes, lim', [(1, -48), (3, -84)])
def test_fused_final3(toy, passes, lim):
    """The stacked-tile version (B = 3 tiles here -> pad to 4)."""
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import fused_final3
    dT = toy.dT
    Bp = 4
    pad = lambda x: np.concatenate([x, np.zeros((Bp - B,) + x.shape[1:], x.dtype)], 0)
    cre, cim = fused_final3(jnp.asarray(pad(dT.real.astype(np.float32))), jnp.asarray(pad(dT.imag.astype(np.float32))), jnp.asarray(pad(toy.gxp)), jnp.asarray(pad(toy.gyp)), float(a0), float(a1), Qf, passes=passes, interpret=True)
    e = err_db((np.asarray(cre) + 1j * np.asarray(cim))[:B], toy.ref)
    print(f'final kernel 3 (stacked tiles) passes {passes}: error {e:.1f} dB')
    assert e < lim, f'final3 {passes}: {e:.1f} dB (limit {lim} dB)'
