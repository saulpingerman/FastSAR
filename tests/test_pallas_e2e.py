"""End-to-end check of the TPU kernels (level kernel and fused final stage) in interpret mode on the CPU: the pallas2
image against the dense image on the simulated scene. FFBP_FORCE_TPU_KERNELS and the interpret-mode wrappers are
set for this module only; FASTSAR_TEST_NLEV (default 2) sets the number of levels."""
import os

import numpy as np
import pytest

from fastsar import sim, ffbp2, pallas_ffbp

pytestmark = pytest.mark.usefixtures('jax_cpu')


@pytest.fixture(scope='module', autouse=True)
def tpu_kernels_interpreted():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FFBP_FORCE_TPU_KERNELS', '1')
        for name in ('fused_rotate_dec_k2', 'fused_rotate_dec_k3', 'fused_final', 'fused_final2', 'fused_final3'):
            _o = getattr(pallas_ffbp, name)
            mp.setattr(pallas_ffbp, name, (lambda o: lambda *a, **k: o(*a, **{**k, 'interpret': True}))(_o))
        yield


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(1)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
    amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
    n = 128
    nlev = int(os.environ.get('FASTSAR_TEST_NLEV', '2'))
    plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
    print('levels', nlev)
    return S, plan, ffbp2.collection_arrays(plan, col.ant)


# thresholds about 5 dB above the values measured on the CPU (interpret mode) when they were set
@pytest.mark.parametrize('pol', ['fp32_fast', 'fp32_high'])
def test_tpu_kernels_vs_dense(scene, pol, monkeypatch):
    S, plan, coll = scene
    bad = []
    hre, him, _ = ffbp2.prepare(pol, S)
    out = {}
    dense_rule, tqp = ffbp2.dense_pulse_filter, ffbp2.FINAL_VMEM_TQP
    for key, filt, kw in (('dense', 'dense', {}), ('pallas2', 'pallas2', dict(pallas_pb=128, pallas_nc=4, pallas_ng=2)),
                          ('banded', 'pallas2', dict(pallas_pb=128, pallas_nc=4, pallas_ng=2)),
                          ('qchunks', 'pallas2', dict(pallas_pb=128, pallas_nc=4, pallas_ng=2))):
        with monkeypatch.context() as mp:
            mp.setattr(ffbp2, 'dense_pulse_filter', (lambda lv: False) if key == 'banded' else dense_rule)   # banded: the long-aperture form
            mp.setattr(ffbp2, 'FINAL_VMEM_TQP', 8 * 16 * 128 if key == 'qchunks' else tqp)                   # final stage in chunks of 8 rows
            static = ffbp2.static_arrays(pol, plan, filt)
            arrs = ffbp2.device_arrays(pol, plan, coll, static)
            fn = ffbp2.make_ffbp(pol, plan, filt, 1 << 24, 'direct', **kw)
            re, im = fn(*fn.pad(hre, him), arrs)          # the history padded once, as ImageFormer passes it
        out[key] = np.asarray(re) + 1j * np.asarray(im)
        if key == 'banded':
            assert any('Wp' in la for la in arrs['levels']), 'the banded pulse filter was not used'
    for key in ('pallas2', 'banded', 'qchunks'):
        d = out[key] - out['dense']
        e = 10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(out['dense']) ** 2))
        lim = -40 if pol == 'fp32_fast' else -90
        if not e < lim:
            bad.append(f'{pol} {key}: {e:.1f} dB (limit {lim} dB)')
        print(pol, '%s (tpu kernels, interpret) vs dense: %.1f dB' % (key, e))
    assert not bad, 'FAILED: ' + '; '.join(bad)
