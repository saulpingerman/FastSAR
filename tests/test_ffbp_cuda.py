"""The CUDA factorized image against the JAX dense image (float32, highest precision) on the simulated scene."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax
jax.config.update('jax_platform_name', 'cpu')
import warnings
from fastsar import sim, ffbp2, ffbp_cuda, memory
import cupy as cp
rng = np.random.default_rng(1)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
n = 128
bad = []                       # thresholds about 5 dB above the values measured when they were set (-86, -79 dB)
for nlev in (2, 3):
    plan = ffbp2.make_plan(col, n, n, 0.5, 0.5, T=16, nlev=nlev, pmax=0.4)
    print('levels', [(l['P'], l['K'], l['Dk'], l['Dp'], l['C']) for l in plan['levels']])
    coll = ffbp2.collection_arrays(plan, col.ant)
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
    print(f'nlev {nlev}: cuda vs dense float32: {10 * np.log10(np.sum(np.abs(d) ** 2) / np.sum(np.abs(ref) ** 2)):.1f} dB, peak ratio {np.abs(img).max() / np.abs(ref).max():.4f}')
    # the first level streamed from the host in blocks of 97 pulses (as for a history larger than the device memory)
    os.environ['FASTSAR_CUDA_STREAM'] = '1'
    keep, memory.STREAM_PULSES = memory.STREAM_PULSES, 97
    img_s = cp.asnumpy(form(S, ng=2))
    os.environ.pop('FASTSAR_CUDA_STREAM')
    memory.STREAM_PULSES = keep
    es = 10 * np.log10(np.sum(np.abs(img_s - img) ** 2) / np.sum(np.abs(img) ** 2))
    if es > -100:
        bad.append(f'nlev {nlev} streamed: {es:.1f} dB')
    print(f'nlev {nlev}: streamed first level vs in memory: {es:.1f} dB')
# the later levels carried through for a batch of first-level children together against one child at a time
form = ffbp_cuda.make_ffbp_cuda(plan, coll)
img_pt = cp.asnumpy(form(S, ng=2, batch=1))
img_bt = cp.asnumpy(form(S, ng=2))
eb = 10 * np.log10(np.sum(np.abs(img_bt - img_pt) ** 2) / np.sum(np.abs(img_pt) ** 2))
if eb > -100:
    bad.append(f'batched children: {eb:.1f} dB')
print(f'later levels batched over children vs one child at a time: {eb:.1f} dB')


def rel(a):
    return 10 * np.log10(np.sum(np.abs(a - img_bt) ** 2) / np.sum(np.abs(img_bt) ** 2) + 1e-300)


# FASTSAR_CUDA_GROUP with a partial last group (3 does not divide the first level's children), no warning
G = plan['levels'][0]['C']
os.environ['FASTSAR_CUDA_GROUP'] = '3'
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter('always')
    img_g = cp.asnumpy(form(S))
os.environ.pop('FASTSAR_CUDA_GROUP')
nw = sum(issubclass(w.category, memory.MemoryWarning) for w in wl)
if rel(img_g) > -100 or nw:
    bad.append(f'FASTSAR_CUDA_GROUP=3: {rel(img_g):.1f} dB, {nw} warnings')
print(f'groups of 3 of {G} children (FASTSAR_CUDA_GROUP) vs 2: {rel(img_g):.1f} dB, {nw} MemoryWarning')

# out of GPU memory: a first level that fails for groups above 2 children halves the group (one MemoryWarning), a
# later level that fails for more than one child halves the batch (quietly), and the image is unchanged; the sizing
# then warns before forming when told the memory is short
lvs = plan['levels']
real_children = ffbp_cuda._children


def failing_children(kern, pre, pim, c0, sl, lv):
    if (lv is lvs[0] and c0.shape[1] > 2) or (lv is lvs[1] and pre.shape[0] > 1):
        raise cp.cuda.memory.OutOfMemoryError(1 << 40, 0)
    return real_children(kern, pre, pim, c0, sl, lv)


ffbp_cuda._children = failing_children
try:
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        img_o = cp.asnumpy(form(S, ng=8))
finally:
    ffbp_cuda._children = real_children
msg = [str(w.message) for w in wl if issubclass(w.category, memory.MemoryWarning)]
if rel(img_o) > -100 or not any('out of GPU memory' in m for m in msg):
    bad.append(f'out-of-memory retry: {rel(img_o):.1f} dB, warnings {msg}')
print(f'out-of-memory retries in the first and later levels: {rel(img_o):.1f} dB, warnings: {msg}')
child = memory.cuda_child(lvs[0], 8, False)
keep = memory.cuda_free
memory.cuda_free = lambda: 8 * S.size + 8.0 * plan['Nx'] * plan['Ny'] + 0.5 * child      # room for no full group
try:
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        img_w = cp.asnumpy(form(S))
finally:
    memory.cuda_free = keep
msg = [str(w.message) for w in wl if issubclass(w.category, memory.MemoryWarning)]
if rel(img_w) > -100 or not any('full speed needs' in m for m in msg):
    bad.append(f'MemoryWarning before forming: {rel(img_w):.1f} dB, warnings {msg}')
print(f'short of memory (patched free memory): {rel(img_w):.1f} dB, warnings: {msg}')

# stripmap mosaic with the range profiles in a moving GPU window (as when they exceed the GPU memory) against all of
# them resident: the same image
from fastsar import stripmap as sm, patches as pt
p = sm.make_params(squint_deg=0.0)
r, x = sm.axes(p)
raw = sm.simulate(p, np.array([[x[len(x) // 2], r[len(r) // 2]]]))
rows = (len(x) // 2 - 140, len(x) // 2 + 140)
img_res, _, _ = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor')
os.environ['FASTSAR_PROFILE_WINDOW_ROWS'] = '1200'
for nw in ('1', '4'):              # patches prepared on one thread and on four sharing the window
    os.environ['FASTSAR_MOSAIC_PREFETCH'] = nw
    img_win, _, _ = pt.form_stripmap(raw, p, rows=rows, backend='cuda', rwin='taylor', awin='taylor')
    ew = 10 * np.log10(np.sum(np.abs(img_win - img_res) ** 2) / np.sum(np.abs(img_res) ** 2))
    if ew > -100:
        bad.append(f'profile window, {nw} prefetch workers: {ew:.1f} dB')
    print(f'mosaic, profiles in a moving GPU window vs resident, {nw} prefetch workers: {ew:.1f} dB')
os.environ.pop('FASTSAR_PROFILE_WINDOW_ROWS'); os.environ.pop('FASTSAR_MOSAIC_PREFETCH')
if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
