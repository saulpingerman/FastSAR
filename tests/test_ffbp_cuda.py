"""The CUDA factorized image against the JAX dense image (float32, highest precision) on the simulated scene; the
streamed first level, batched later levels, group sizes, out-of-memory retries and the moving window of mosaic
range profiles against the plain paths. Needs a GPU (marked cuda)."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from fastsar import sim, ffbp2, memory

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures('jax_cpu')]


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(1)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
    amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    return col, sim.simulate_brute(col, pos, amp).astype(np.complex64)


def _plan(col, nlev, T=16):
    n = 128
    plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=T, nlev=nlev, pmax=0.4)
    return plan, ffbp2.collection_arrays(plan, col.ant)


# thresholds about 5 dB above the values measured when they were set (-86, -79 dB)
@pytest.mark.parametrize('nlev', [2, 3])
def test_cuda_vs_dense(scene, nlev, monkeypatch):
    import cupy as cp
    from fastsar import ffbp_cuda
    col, S = scene
    bad = []
    plan, coll = _plan(col, nlev)
    print('levels', [(l['P'], l['K'], l['Dk'], l['Dp'], l['C']) for l in plan['levels']])
    static = ffbp2.static_arrays('fp32', plan, 'dense')
    arrs = ffbp2.device_arrays('fp32', plan, coll, static)
    hre, him, scale = ffbp2.prepare('fp32', S)
    fn = ffbp2.make_ffbp('fp32', plan, 'dense', 1 << 24, 'direct')
    re, im = fn(hre, him, arrs)
    ref = (np.asarray(re) + 1j * np.asarray(im)) * scale
    form = ffbp_cuda.make_ffbp_cuda(plan, coll)
    img = cp.asnumpy(form(S, ng=2))
    d = img - ref
    e = 10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(ref) ** 2))
    if e > -70 or abs(np.abs(img).max() / np.abs(ref).max() - 1) > 1e-3:
        bad.append(f'nlev {nlev}: {e:.1f} dB')
    print(f'nlev {nlev}: cuda vs dense float32: {e:.1f} dB, peak ratio {np.abs(img).max() / np.abs(ref).max():.4f}')
    # the first level streamed from the host in blocks of 97 pulses (as for a history larger than the device memory)
    with monkeypatch.context() as mp:
        mp.setenv('FASTSAR_CUDA_STREAM', '1')
        mp.setattr(memory, 'STREAM_PULSES', 97)
        img_s = cp.asnumpy(form(S, ng=2))
    es = 10 * np.log10(np.sum(np.abs(img_s - img) ** 2) / np.sum(np.abs(img) ** 2))
    if es > -100:
        bad.append(f'nlev {nlev} streamed: {es:.1f} dB')
    print(f'nlev {nlev}: streamed first level vs in memory: {es:.1f} dB')
    assert not bad, 'FAILED: ' + '; '.join(bad)


@pytest.fixture(scope='module')
def three(scene):
    """The three-level former of the scene, shared in order by the tests below as one former was in the script,
    and its image with the later levels batched over first-level children."""
    import cupy as cp
    from fastsar import ffbp_cuda
    col, S = scene
    plan, coll = _plan(col, 3)
    form = ffbp_cuda.make_ffbp_cuda(plan, coll)
    img_pt = cp.asnumpy(form(S, ng=2, batch=1))
    img_bt = cp.asnumpy(form(S, ng=2))
    return SimpleNamespace(plan=plan, form=form, S=S, img_pt=img_pt, img_bt=img_bt,
                           rel=lambda a: 10 * np.log10(np.sum(np.abs(a - img_bt) ** 2) / np.sum(np.abs(img_bt) ** 2) + 1e-300))


def test_batched_children(three):
    """The later levels carried through for a batch of first-level children together against one child at a time."""
    eb = 10 * np.log10(np.sum(np.abs(three.img_bt - three.img_pt) ** 2) / np.sum(np.abs(three.img_pt) ** 2))
    print(f'later levels batched over children vs one child at a time: {eb:.1f} dB')
    assert not eb > -100, f'batched children: {eb:.1f} dB'


def test_group_env(three, monkeypatch):
    """FASTSAR_CUDA_GROUP with a partial last group (3 does not divide the first level's children), no warning."""
    import cupy as cp
    G = three.plan['levels'][0]['C']
    monkeypatch.setenv('FASTSAR_CUDA_GROUP', '3')
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        img_g = cp.asnumpy(three.form(three.S))
    monkeypatch.delenv('FASTSAR_CUDA_GROUP')
    nw = sum(issubclass(w.category, memory.MemoryWarning) for w in wl)
    print(f'groups of 3 of {G} children (FASTSAR_CUDA_GROUP) vs 2: {three.rel(img_g):.1f} dB, {nw} MemoryWarning')
    assert not (three.rel(img_g) > -100 or nw), f'FASTSAR_CUDA_GROUP=3: {three.rel(img_g):.1f} dB, {nw} warnings'


def test_out_of_memory_retry(three, monkeypatch):
    """Out of GPU memory: a first level that fails for groups above 2 children halves the group (one
    MemoryWarning), a later level that fails for more than one child halves the batch (quietly), and the image is
    unchanged."""
    import cupy as cp
    from fastsar import ffbp_cuda
    lvs = three.plan['levels']
    real_children = ffbp_cuda._children

    def failing_children(kern, pre, pim, c0, sl, lv):
        if (lv is lvs[0] and c0.shape[1] > 2) or (lv is lvs[1] and pre.shape[0] > 1):
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)
        return real_children(kern, pre, pim, c0, sl, lv)

    with monkeypatch.context() as mp:
        mp.setattr(ffbp_cuda, '_children', failing_children)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            img_o = cp.asnumpy(three.form(three.S, ng=8))
    msg = [str(w.message) for w in wl if issubclass(w.category, memory.MemoryWarning)]
    print(f'out-of-memory retries in the first and later levels: {three.rel(img_o):.1f} dB, warnings: {msg}')
    assert not (three.rel(img_o) > -100 or not any('out of GPU memory' in m for m in msg)), \
        f'out-of-memory retry: {three.rel(img_o):.1f} dB, warnings {msg}'



def test_later_level_split(three, monkeypatch):
    """A later level whose intermediate of all output children does not fit is formed over parts of its children
    (a 12 GB UMBRA-10 spotlight on an L4): the image is unchanged."""
    import cupy as cp
    from fastsar import ffbp_cuda
    lvs = three.plan['levels']
    real_children = ffbp_cuda._children

    def failing_children(kern, pre, pim, c0, sl, lv):
        if lv is lvs[1] and c0.shape[1] > 1:
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)
        return real_children(kern, pre, pim, c0, sl, lv)

    with monkeypatch.context() as mp:
        mp.setattr(ffbp_cuda, '_children', failing_children)
        img_s = cp.asnumpy(three.form(three.S, ng=2))
    print(f'later level formed one output child at a time: {three.rel(img_s):.1f} dB')
    assert not three.rel(img_s) > -100, f'later-level split: {three.rel(img_s):.1f} dB'


def test_out_of_memory_message(three, monkeypatch):
    """A later level that fails even for one child of one first-level child: smaller groups first (a MemoryWarning),
    then a MemoryError that names what to change, not CuPy's error."""
    import cupy as cp
    from fastsar import ffbp_cuda
    lvs = three.plan['levels']
    real_children = ffbp_cuda._children

    def failing_children(kern, pre, pim, c0, sl, lv):
        if lv is lvs[1]:
            raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)
        return real_children(kern, pre, pim, c0, sl, lv)

    with monkeypatch.context() as mp:
        mp.setattr(ffbp_cuda, '_children', failing_children)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            with pytest.raises(MemoryError, match="precision='float16'") as ei:
                three.form(three.S, ng=2)
    msg = [str(w.message) for w in wl if issubclass(w.category, memory.MemoryWarning)]
    print(f'out of memory for one child: {ei.value}; warnings: {msg}')
    assert not isinstance(ei.value, cp.cuda.memory.OutOfMemoryError)
    assert any('later levels' in m for m in msg), msg

def test_memory_warning_before_forming(three, monkeypatch):
    """The sizing warns before forming when told the memory is short, and the image is unchanged."""
    import cupy as cp
    plan, S = three.plan, three.S
    child = memory.cuda_child(plan['levels'][0], 8, False)
    with monkeypatch.context() as mp:
        mp.setattr(memory, 'cuda_free', lambda: 8 * S.size + 8.0 * plan['Nx'] * plan['Ny'] + 0.5 * child)   # room for no full group
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            img_w = cp.asnumpy(three.form(S))
    msg = [str(w.message) for w in wl if issubclass(w.category, memory.MemoryWarning)]
    print(f'short of memory (patched free memory): {three.rel(img_w):.1f} dB, warnings: {msg}')
    assert not (three.rel(img_w) > -100 or not any('full speed needs' in m for m in msg)), \
        f'MemoryWarning before forming: {three.rel(img_w):.1f} dB, warnings {msg}'


def test_profile_window(monkeypatch):
    """Stripmap mosaic with the range profiles in a moving GPU window (as when they exceed the GPU memory) against
    all of them resident: the same image."""
    from fastsar import stripmap as sm, patches as pt
    bad = []
    p = sm.make_params(squint_deg=0.0)
    r, x = sm.axes(p)
    raw = sm.simulate(p, np.array([[x[len(x) // 2], r[len(r) // 2]]]))
    rows = (len(x) // 2 - 140, len(x) // 2 + 140)
    img_res, _, _ = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor')
    # a window of 1200 rows serves the patches; one of 16 rows is smaller than any patch, whose profiles are then
    # gathered on the host into pinned memory (on four threads at once with four prefetch workers)
    for wr in ('1200', '16'):
        monkeypatch.setenv('FASTSAR_PROFILE_WINDOW_ROWS', wr)
        for nw in ('1', '4'):          # patches prepared on one thread and on four sharing the window
            monkeypatch.setenv('FASTSAR_MOSAIC_PREFETCH', nw)
            img_win, _, _ = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor')
            ew = 10 * np.log10(np.sum(np.abs(img_win - img_res) ** 2) / np.sum(np.abs(img_res) ** 2))
            if ew > -100:
                bad.append(f'profile window of {wr} rows, {nw} prefetch workers: {ew:.1f} dB')
            print(f'mosaic, profiles in a {wr}-row GPU window vs resident, {nw} prefetch workers: {ew:.1f} dB')
    assert not bad, 'FAILED: ' + '; '.join(bad)


def test_staging_reuse(monkeypatch):
    """A mosaic's page-locked staging buffers are reused: a smaller request takes a free buffer, a larger one replaces
    it (the count stays bounded), and the patches of a host-gathered mosaic make at most one per gather in flight."""
    import threading
    import cupy as cp
    from fastsar import stripmap as sm, patches as pt
    stage = dict(lock=threading.Lock(), free=[])
    a, ba = pt._staging_take(stage, (100, 1000), np.complex64)
    a[:] = 1
    pt._staging_give(stage, ba)
    b, bb = pt._staging_take(stage, (90, 1000), np.complex64)
    assert bb is ba and b.shape == (90, 1000) and stage.get('made') == 1
    pt._staging_give(stage, bb)
    c, bc = pt._staging_take(stage, (400, 1000), np.complex64)
    assert bc is not ba and stage['made'] == 2 and not stage['free']
    pt._staging_give(stage, bc)
    p = sm.make_params(squint_deg=0.0)
    r, x = sm.axes(p)
    raw = sm.simulate(p, np.array([[x[len(x) // 2], r[len(r) // 2]]]))
    rows = (len(x) // 2 - 140, len(x) // 2 + 140)
    monkeypatch.setenv('FASTSAR_PROFILE_WINDOW_ROWS', '16')
    monkeypatch.setenv('FASTSAR_TIMING', '1')
    imgs = {}
    for nw in (1, 3):
        monkeypatch.setenv('FASTSAR_MOSAIC_PREFETCH', str(nw))
        pt._Timer.acc.clear()
        info = []
        imgs[nw] = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor', info=info)
        made = pt._Timer.acc.get('form_mosaic: page-locked staging buffers made (count)')
        kept = pt._Timer.acc.get('form_mosaic: free blocks in the pinned pool at the end (count)')
        print(f'prefetch {nw}: {len(info)} patches, {made} staging buffers, {kept} free pinned-pool blocks')
        assert len(info) > 2 and made is not None and 1 <= made <= 2 * (nw + 1)
        assert kept is not None and cp.get_default_pinned_memory_pool().n_free_blocks() <= pt._PINNED_KEEP
    # the same patches gathered into CuPy's pool instead of the staging buffers give the same image
    real = pt._staging_take
    monkeypatch.setattr(pt, '_staging_take', lambda stage, shape, dtype: (pt._pinned_empty(shape, dtype), None))
    plain = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor')
    monkeypatch.setattr(pt, '_staging_take', real)
    assert np.array_equal(plain, imgs[1]) and np.array_equal(plain, imgs[3])


def test_pinned_off(three, monkeypatch):
    """FASTSAR_PINNED=0 (a plain copy of the image) against the pinned copy: the same image; a bad value raises."""
    import cupy as cp
    from fastsar import api
    dev = three.form(three.S, ng=2)
    a = api._to_host(dev)
    monkeypatch.setenv('FASTSAR_PINNED', '0')
    b = api._to_host(dev)
    assert np.array_equal(a, b) and np.array_equal(a, cp.asnumpy(dev))
    monkeypatch.setenv('FASTSAR_PINNED', 'no')
    with pytest.raises(ValueError, match='FASTSAR_PINNED'):
        api._to_host(dev)


def test_float16_zero_first_block(scene, monkeypatch):
    """float16 storage scales by the first uploaded block's peak; a history whose first block is zero (dropped
    pulses, zeroed) falls back to the whole history's peak and forms the same image as float32 within float16's
    error, where a scale of 1 would overflow the half range."""
    import cupy as cp
    from fastsar import ffbp_cuda
    col, S = scene
    plan, coll = _plan(col, 2, T=32)                 # float16 on cuda: T = 32
    monkeypatch.setattr(memory, 'STREAM_PULSES', 32)          # upload blocks of 32 pulses
    S0 = S.copy()
    S0[:32] = 0
    S0 *= 1e4 / np.abs(S0).max()                   # a peak of 10^4: above float16's range times any block's share
    f32 = cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll)(S0))
    f16 = cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode='f16tc', store='f16')(S0))
    e = 10 * np.log10(np.sum(np.abs(f16 - f32) ** 2) / np.sum(np.abs(f32) ** 2))
    print(f'float16 with a zero first block vs float32: {e:.1f} dB, finite {np.isfinite(f16).all()}')
    assert np.isfinite(f16).all()
    assert e < -40, e


def test_env_validation(scene, monkeypatch):
    """The CUDA settings with fixed choices reject other values, naming the variable."""
    from fastsar import ffbp_cuda
    col, S = scene
    plan = ffbp2.make_plan(col, 128, 128, 0.5, 0.5, T=32, nlev=2, pmax=0.4)       # T = 32: the tensor-core stage
    coll = ffbp2.collection_arrays(plan, col.ant)
    for name, bad in (('FASTSAR_FINAL', 'foo'), ('FASTSAR_TC_PPS', '3'), ('FASTSAR_TC_MW', '16'), ('FASTSAR_FIRP', 'x')):
        with monkeypatch.context() as mp:
            mp.setenv(name, bad)
            with pytest.raises(ValueError, match=name):
                ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode='f16tc', store='f16')(S)


def test_no_nvcc(scene, monkeypatch):
    """Without nvcc the float32 final stage falls back to the CUDA cores with one warning (the same image as
    FASTSAR_FINAL=fp32), and float16 raises an error that names the remedy."""
    import cupy as cp
    from fastsar import ffbp_cuda
    col, S = scene
    plan, coll = _plan(col, 2, T=32)                 # T = 32: the tensor-core final stage applies
    monkeypatch.setenv('FASTSAR_FINAL', 'fp32')
    ref = cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll)(S))
    monkeypatch.delenv('FASTSAR_FINAL')
    monkeypatch.setattr(ffbp_cuda, '_NVCC', {'path': None})
    with pytest.warns(RuntimeWarning, match='nvcc not found'):
        img = cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll)(S))
    assert np.array_equal(img, ref)
    with pytest.raises(RuntimeError, match='nvcc'):
        ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode='f16tc', store='f16')

@pytest.mark.parametrize('weighted', [False, True])
def test_tensor_core_final(scene, weighted, monkeypatch):
    """The final stage on the tensor cores at T = 32: the three bfloat16 passes (the float32 default on Ampere and later)
    against the float32 CUDA-core kernel, and the double-buffered float16 kernel against the 0.1.2 one, with and
    without aperture weights in the kernel."""
    import cupy as cp
    from fastsar import ffbp_cuda
    from fastsar.api import final_weights
    col, S = scene
    plan = ffbp2.make_plan(col, 128, 128, 0.5, 0.5, T=32, nlev=2, pmax=0.4)
    coll = ffbp2.collection_arrays(plan, col.ant)
    P = len(col.ant)
    wf = None
    if weighted:
        def weight(q, pul):
            q, pul = np.asarray(q, np.float64), np.asarray(pul, np.float64)
            return (1.0 + 0.3 * np.cos(2 * np.pi * pul / P))[:, None] * (1.0 + 0.01 * q[None, :, 0] - 0.005 * q[None, :, 1])
        wf = final_weights(plan, weight, P, grad=True)

    def form(mode='fp32', store='fp32', **env):
        with monkeypatch.context() as mp:
            for k, v in env.items():
                mp.setenv(k, v)
            ffbp_cuda._mods_tc.clear()
            return cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode=mode, store=store, wf=wf)(S, ng=2))

    def rel(a, b):
        return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2) + 1e-300)
    bad = []
    cores = form(FASTSAR_FINAL='fp32')
    if int(cp.cuda.Device().compute_capability) >= 80:
        e = rel(form(), cores)
        print(f'weighted {weighted}: three bfloat16 passes vs float32 CUDA cores: {e:.1f} dB')
        if e > -80:                                     # measured -87 to -90 dB (L4, H100)
            bad.append(f'three passes {e:.1f} dB')
    f2 = form('f16tc', 'f16')
    f1 = form('f16tc', 'f16', FASTSAR_TC_KERNEL='1')
    e2, e1, e21 = rel(f2, cores), rel(f1, cores), rel(f2, f1)
    print(f'weighted {weighted}: float16 new kernel vs float32 {e2:.1f} dB, 0.1.2 kernel vs float32 {e1:.1f} dB, new vs 0.1.2 {e21:.1f} dB')
    if e2 > e1 + 1.0 or e2 > -63:                       # measured -68.8 to -69.1 dB; the 0.1.2 kernel -56.3 weighted
        bad.append(f'float16 new kernel {e2:.1f} dB against {e1:.1f} dB')
    assert not bad, 'FAILED: ' + '; '.join(bad)
