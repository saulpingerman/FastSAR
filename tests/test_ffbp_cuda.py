"""The CUDA factorized image against the JAX dense image (float32, highest precision) on the simulated scene."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax
jax.config.update('jax_platform_name', 'cpu')
from fastsar import sim, ffbp2, ffbp_cuda
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
    keep, ffbp_cuda.STREAM_PULSES = ffbp_cuda.STREAM_PULSES, 97
    img_s = cp.asnumpy(form(S, ng=2))
    os.environ.pop('FASTSAR_CUDA_STREAM'); ffbp_cuda.STREAM_PULSES = keep
    es = 10 * np.log10(np.sum(np.abs(img_s - img) ** 2) / np.sum(np.abs(img) ** 2))
    if es > -100:
        bad.append(f'nlev {nlev} streamed: {es:.1f} dB')
    print(f'nlev {nlev}: streamed first level vs in memory: {es:.1f} dB')
# the later levels carried through for a batch of first-level children together against one child at a time
os.environ['FASTSAR_CUDA_PER_TILE'] = '1'
img_pt = cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll)(S, ng=2))
os.environ.pop('FASTSAR_CUDA_PER_TILE')
img_bt = cp.asnumpy(ffbp_cuda.make_ffbp_cuda(plan, coll)(S, ng=2))
eb = 10 * np.log10(np.sum(np.abs(img_bt - img_pt) ** 2) / np.sum(np.abs(img_pt) ** 2))
if eb > -100:
    bad.append(f'batched children: {eb:.1f} dB')
print(f'later levels batched over children vs one child at a time: {eb:.1f} dB')

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
