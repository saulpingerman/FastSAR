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
                          ('qchunks', 'pallas2', dict(pallas_pb=128, pallas_nc=4, pallas_ng=2)),
                          ('capped', 'pallas2', dict(pallas_pb=128, pallas_nc=4, pallas_ng=2, pallas_win_cap=1))):
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
    for key in ('pallas2', 'banded', 'qchunks', 'capped'):
        d = out[key] - out['dense']
        e = 10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(out['dense']) ** 2))
        lim = -40 if pol == 'fp32_fast' else -90
        if not e < lim:
            bad.append(f'{pol} {key}: {e:.1f} dB (limit {lim} dB)')
        print(pol, '%s (tpu kernels, interpret) vs dense: %.1f dB' % (key, e))
    assert not bad, 'FAILED: ' + '; '.join(bad)


@pytest.mark.parametrize('pol', ['fp32_high', 'bf16'])
def test_prepare_into_layout(scene, pol, monkeypatch):
    """prepare's block-wise upload into the first-level kernel's padded planes (form.layout) gives the planes that
    form.pad makes from the unpadded ones, and form.pad passes them unchanged."""
    import warnings
    S, plan, coll = scene
    fn = ffbp2.make_ffbp(pol, plan, 'pallas2', 1 << 24, 'direct', pallas_pb=128, pallas_nc=4, pallas_ng=2)
    assert fn.layout is not None
    a = fn.pad(*ffbp2.prepare(pol, S)[:2])
    monkeypatch.setattr(ffbp2, '_blocked_upload', lambda: True)
    monkeypatch.setattr(ffbp2, 'UPLOAD_ROWS', 100)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Some donated buffers were not usable')
        hre, him, _ = ffbp2.prepare(pol, S, fn.layout)
    b = fn.pad(hre, him)
    assert b[0] is hre and b[1] is him
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype and np.array_equal(np.asarray(x), np.asarray(y))
    print(f'  prepare {pol}: planes uploaded into the padded layout equal form.pad\'s')


def test_vmem_retry(scene, monkeypatch):
    """ImageFormer(backend='tpu') whose level kernel runs out of vector memory (simulated for pulse blocks of more than
    a quarter of the first cap) retries with smaller pulse blocks, remembers the cap for the plan, and forms the image
    of the jax backend."""
    import jax
    import fastsar
    from fastsar import api, sim as fsim
    col = fsim.make_collect(res=0.5, scene=60.0, r0=5e3)
    S = scene[0]
    seen = []
    k3 = pallas_ffbp.fused_rotate_dec_k3

    def tight(*a, **k):
        pb, win = k['pb'], a[4]['win']
        seen.append(pb * win)
        if pb * win > lim[0]:
            seen.append('raised')
            raise jax.errors.JaxRuntimeError('RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space vmem.')
        return k3(*a, **k)

    lim = [None]
    monkeypatch.setattr(pallas_ffbp, 'fused_rotate_dec_k3', tight)
    monkeypatch.setattr(api, '_WIN_CAPS', {})
    f = fastsar.ImageFormer(col.ant, col.fmin, col.df, S.shape[1], 128, 128, 0.5, 0.5, backend='tpu')
    wmax = max(b['win'] for b in f._fn.stages['bands'].values())
    lim[0] = 32 * wmax                       # the third cap (128, 64, 32 times the widest window) fits
    with pytest.warns(UserWarning, match='vector memory'):
        img = f(S)
    assert list(api._WIN_CAPS.values()) == [32 * wmax] and f._fn.stages['win_cap'] == 32 * wmax
    last = seen[len(seen) - seen[::-1].index('raised'):]                 # the kernel calls of the attempt that formed
    assert last and max(last) <= 32 * wmax
    ref = fastsar.ImageFormer(col.ant, col.fmin, col.df, S.shape[1], 128, 128, 0.5, 0.5, backend='jax')(S)
    e = 10 * np.log10(np.sum(np.abs(img - ref) ** 2) / np.sum(np.abs(ref) ** 2))
    print(f'  vmem retry: cap {32 * wmax} samples, image vs jax backend {e:.1f} dB')
    assert e < -80


def test_upload_memory_error(scene, monkeypatch):
    """A device out of memory during the upload raises MemoryError whatever class the runtime used (a TPU buffer
    allocation that failed surfaced as a ValueError carrying RESOURCE_EXHAUSTED)."""
    import fastsar
    from fastsar import ffbp2 as f2, sim as fsim
    col = fsim.make_collect(res=0.5, scene=60.0, r0=5e3)
    S = scene[0]
    f = fastsar.ImageFormer(col.ant, col.fmin, col.df, S.shape[1], 128, 128, 0.5, 0.5, backend='tpu')
    for exc in (ValueError, RuntimeError):
        def boom(*a, exc=exc, **k):
            raise exc('RESOURCE_EXHAUSTED: E0100: RuntimeBufferAllocationFailure: Error allocating device buffer')
        monkeypatch.setattr(f2, 'prepare', boom)
        with pytest.raises(MemoryError, match='does not fit in device memory'):
            f(S)
    monkeypatch.setattr(f2, 'prepare', lambda *a, **k: (_ for _ in ()).throw(ValueError('something else')))
    with pytest.raises(ValueError, match='something else'):
        f(S)
    print('  upload: RESOURCE_EXHAUSTED as ValueError and RuntimeError -> MemoryError; other errors pass through')
