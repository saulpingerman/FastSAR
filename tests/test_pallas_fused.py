"""Toy-size check of the fused rotate-and-decimate kernel against the dense XLA path (interpret mode runs on CPU)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures('jax_cpu')

P, K, D, L = 64, 1536, 6, 44
Ko = (K + D - 1) // D
N, nc = 2, 3
kc = (K - 1) / 2.0
kk = np.arange(K) - kc


def err_db(num, den):
    return 10 * np.log10(num / den)


@pytest.fixture(scope='module')
def toy():
    """A banded decimation matrix like the plan's (Kaiser taps, output j from inputs D j + r - pl) and the random
    histories and phases of every kernel, drawn in the original order from one generator."""
    rng = np.random.default_rng(0)
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
    Ss = [(rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K))).astype(np.complex64) for _ in range(N)]
    c0s = rng.uniform(-1e4, 1e4, (N, nc, P)).astype(np.float32)
    sls = rng.uniform(-0.3, 0.3, (N, nc, P)).astype(np.float32)
    return SimpleNamespace(Fk=Fk, S=S, c0=c0, slope=slope, Ss=Ss, c0s=c0s, sls=sls)


def child_refs(toy):
    """The float64 reference of each parent and child: rotate, then the dense product."""
    out = {}
    for n in range(N):
        for c in range(nc):
            cyc = toy.c0s[n, c][:, None].astype(np.float64) + kk[None, :] * toy.sls[n, c][:, None].astype(np.float64)
            out[n, c] = (toy.Ss[n].astype(np.complex128) * np.exp(1j * (cyc - np.round(cyc)) * 2 * np.pi)) @ toy.Fk
    return out


# limits about 5 dB above the values measured on the CPU (interpret mode) when they were set
@pytest.mark.parametrize('passes, lim', [(1, -45), (3, -55)])
def test_kernel1(toy, passes, lim):
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import band_blocks, pad_columns, fused_rotate_dec_k
    # reference: rotate in float64 then dense product
    cyc = toy.c0[:, None].astype(np.float64) + kk[None, :] * toy.slope[:, None].astype(np.float64)
    ang = (cyc - np.round(cyc)) * 2 * np.pi
    Yref = (toy.S.astype(np.complex128) * np.exp(1j * ang)) @ toy.Fk
    band = band_blocks(toy.Fk, D, kob=256)
    Sre = pad_columns(jnp.asarray(toy.S.real), band)
    Sim = pad_columns(jnp.asarray(toy.S.imag), band)
    yr, yi = fused_rotate_dec_k(Sre, Sim, jnp.asarray(toy.c0), jnp.asarray(toy.slope), band, kc, pb=32, chunk=512, passes=passes, interpret=True)
    Y = np.asarray(yr)[:, :Ko] + 1j * np.asarray(yi)[:, :Ko]
    e = 10 * np.log10(np.sum(np.abs(Y - Yref) ** 2) / np.sum(np.abs(Yref) ** 2))
    print(f'passes {passes}: error {e:.1f} dB relative to float64')
    assert e < lim, f'kernel1 {passes}: {e:.1f} dB (limit {lim} dB)'


@pytest.fixture(scope='module')
def band2(toy):
    """Second-generation kernel: two parents, three children each."""
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import band_blocks2, pad_columns
    band2 = band_blocks2(toy.Fk, D)
    print('band2', {k: band2[k] for k in ('kob', 'win', 'stride', 'nb', 'Kpad', 'chunk')})
    Sre2 = jnp.stack([pad_columns(jnp.asarray(s.real), band2) for s in toy.Ss]); Sim2 = jnp.stack([pad_columns(jnp.asarray(s.imag), band2) for s in toy.Ss])
    return band2, Sre2, Sim2


@pytest.mark.parametrize('passes, lim', [(1, -45), (3, -55)])
def test_kernel2(toy, band2, passes, lim):
    """Against the same float64 reference per child."""
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import fused_rotate_dec_k2
    band, Sre2, Sim2 = band2
    yr, yi = fused_rotate_dec_k2(Sre2, Sim2, jnp.asarray(toy.c0s), jnp.asarray(toy.sls), band, kc, pb=32, passes=passes, interpret=True)
    num = den = 0.0
    for (n, c), ref in child_refs(toy).items():
        Y = np.asarray(yr)[n, c, :, :Ko] + 1j * np.asarray(yi)[n, c, :, :Ko]
        num += np.sum(np.abs(Y - ref) ** 2); den += np.sum(np.abs(ref) ** 2)
    e = err_db(num, den)
    print(f'kernel2 passes {passes}: error {e:.1f} dB')
    assert e < lim, f'kernel2 {passes}: {e:.1f} dB (limit {lim} dB)'


def test_kernel2_multiblock(toy):
    """Multi-block form of the second kernel (forced small output blocks)."""
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import band_blocks2, pad_columns, fused_rotate_dec_k2
    band3 = band_blocks2(toy.Fk, D, kob=128)
    print('band3', {k: band3[k] for k in ('kob', 'win', 'stride', 'nb', 'Kpad', 'chunk')})
    Sre3 = jnp.stack([pad_columns(jnp.asarray(s.real), band3) for s in toy.Ss]); Sim3 = jnp.stack([pad_columns(jnp.asarray(s.imag), band3) for s in toy.Ss])
    yr, yi = fused_rotate_dec_k2(Sre3, Sim3, jnp.asarray(toy.c0s), jnp.asarray(toy.sls), band3, kc, pb=32, passes=1, interpret=True)
    num = den = 0.0
    for (n, c), ref in child_refs(toy).items():
        Y = np.asarray(yr)[n, c, :, :Ko] + 1j * np.asarray(yi)[n, c, :, :Ko]
        num += np.sum(np.abs(Y - ref) ** 2); den += np.sum(np.abs(ref) ** 2)
    e = err_db(num, den)
    print(f'kernel2 multi-block: error {e:.1f} dB')
    assert e < -45, f'kernel2 multi-block: {e:.1f} dB (limit -45 dB)'


@pytest.mark.parametrize('passes, lim', [(1, -44), (3, -55)])
@pytest.mark.parametrize('fused', [False, True])
def test_kernel3(toy, band2, fused, passes, lim):
    """Third kernel: precomputed coarse tables, with and without the fused pulse decimation."""
    import jax.numpy as jnp
    from fastsar.pallas_ffbp import fused_rotate_dec_k3
    band, Sre2, Sim2 = band2
    Dp, Po = 2, 40
    tp_ = np.kaiser(16, 8.0) * np.sinc((np.arange(16) - 7.5) / Dp) / Dp
    Fp = np.zeros((P, Po))
    for j in range(Po):
        for r in range(16):
            i = Dp * j + r - 6
            if 0 <= i < P:
                Fp[i, j] = tp_[r]
    yr, yi = fused_rotate_dec_k3(Sre2, Sim2, jnp.asarray(toy.c0s), jnp.asarray(toy.sls), band, kc, pb=32, passes=passes,
                                 FpT=jnp.asarray(Fp.T, jnp.float32) if fused else None, interpret=True)
    num = den = 0.0
    for (n, c), ref in child_refs(toy).items():
        if fused:
            ref = Fp.T @ ref
            Y = np.asarray(yr)[n, c, :Po, :Ko] + 1j * np.asarray(yi)[n, c, :Po, :Ko]
        else:
            Y = np.asarray(yr)[n, c, :, :Ko] + 1j * np.asarray(yi)[n, c, :, :Ko]
        num += np.sum(np.abs(Y - ref) ** 2); den += np.sum(np.abs(ref) ** 2)
    e = err_db(num, den)
    print(f'kernel3 fused_p={fused} passes {passes}: error {e:.1f} dB')
    assert e < lim, f'kernel3 {fused} {passes}: {e:.1f} dB (limit {lim} dB)'
