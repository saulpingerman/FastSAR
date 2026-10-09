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


def _plan(col, nlev):
    n = 128
    plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
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
    monkeypatch.setenv('FASTSAR_PROFILE_WINDOW_ROWS', '1200')
    for nw in ('1', '4'):              # patches prepared on one thread and on four sharing the window
        monkeypatch.setenv('FASTSAR_MOSAIC_PREFETCH', nw)
        img_win, _, _ = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor')
        ew = 10 * np.log10(np.sum(np.abs(img_win - img_res) ** 2) / np.sum(np.abs(img_res) ** 2))
        if ew > -100:
            bad.append(f'profile window, {nw} prefetch workers: {ew:.1f} dB')
        print(f'mosaic, profiles in a moving GPU window vs resident, {nw} prefetch workers: {ew:.1f} dB')
    assert not bad, 'FAILED: ' + '; '.join(bad)
