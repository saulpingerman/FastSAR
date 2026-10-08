"""Unit checks of pieces that the end-to-end scripts reach only indirectly, each against an independent or slower
path that computes the same thing:

  ffbp2.decimator_fir          against fir(ffbp.decimator(...)), bit for bit; LazyDecimator against the dense matrix
  ffbp2.plan_signature         equal plans share one compiled JAX program (api._JAX_PROGRAMS), and a program reused
                               for another antenna path gives the image of a fresh compile
  ffbp_cpu                     the complex64 input read in place (rot_fir_k_cplx) against split float32 planes
                               (rot_fir_k), and level-0 group sizes (ng, FASTSAR_CPU_GROUP_GB) against each other
  api.final_weights            against the mean weight over each final subaperture's pulses, and its gradient
  ImageFormer(aperture_weight) unit weights against none; pulse-only weights against a pre-weighted history; cpu
                               against jax with pixel-dependent weights
  patches.range_profiles       patch_history(prof=...) against the direct path, and form_mosaic with
                               FASTSAR_SHARED_PROFILES=0 against 1
  patches.beam_span            against the beam evaluated on every pulse
  patches.fill_gaps            inserted positions and the index of the recorded pulses
  form_mosaic weights          in the final stage against the separable terms (FASTSAR_WEIGHT_TERMS=1)
  form_mosaic prefetch         FASTSAR_MOSAIC_PREFETCH=1 against 0 (identical images), and an error in the prefetch
                               thread reaching the caller
  input checks                 bad shapes, dtypes, non-finite samples and backends raise ValueError or TypeError;
                               complex128 and non-contiguous histories give the complex64 image

Errors are 10 log10 of the energy of the difference over that of the reference. Limits sit about 5 dB above the
values measured when they were set, or at zero where the two paths must agree exactly."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, ffbp2, ffbp_cpu, patches as pt, stripmap as sm, api
from fastsar.ffbp import decimator

bad = []


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2) + 1e-300)


def check(name, value, limit, fmt='{:.1f} dB'):
    ok = value <= limit
    print(f'  {name:62s} {fmt.format(value):>12s}  (limit {fmt.format(limit)})' + ('' if ok else '  FAIL'))
    if not ok:
        bad.append(name)


def expect(name, fn, exc):
    """fn() must raise exc; prints its message."""
    try:
        fn()
    except exc as e:
        print(f'  {name:40s} {type(e).__name__}: {str(e)[:90]}')
        return
    except Exception as e:
        print(f'  {name:40s} FAIL: {type(e).__name__} instead of {exc.__name__}: {str(e)[:80]}')
        bad.append(name)
        return
    print(f'  {name:40s} FAIL: no error')
    bad.append(name)


class env:
    """Set environment variables for a block."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        os.environ.update({k: str(v) for k, v in self.kw.items()})

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


t_start = time.perf_counter()
rng = np.random.default_rng(3)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
tg = np.stack([rng.uniform(-25, 25, 30), rng.uniform(-25, 25, 30), np.zeros(30)], 1)
S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
grid = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
P, K = S.shape
print(f'spotlight scene: {P} pulses x {K} samples, 30 targets, grid 128 x 128 of 0.5 m')

# ---------------------------------------------------------------------------------------------- decimation filters
print('\ndecimation filters')
n, nbits = 0, 0
for n_in in (97, 301, 1000, 4097):
    for D in (2, 3, 5, 8):
        for pf in (0.3, 0.71875, 0.95):
            F, m = decimator(n_in, D, pf)
            if n_in < 2 * m * D + D:
                continue
            a, (b, mb) = ffbp2.fir(F, D, m), ffbp2.decimator_fir(n_in, D, pf)
            same = mb == m and all(a[k] == b[k] for k in ('pl', 'pr', 'n_out', 'L')) and np.array_equal(a['kern'], b['kern'])
            lz = ffbp2.LazyDecimator(n_in, D, pf, 70.0)
            same &= lz.m == m and lz.shape == F.shape and np.array_equal(np.asarray(lz), F) and np.array_equal(lz.T, F.T)
            same &= np.array_equal(lz.astype(np.float32), F.astype(np.float32))
            n, nbits = n + 1, nbits + (not same)
lz = ffbp2.LazyDecimator(50, 1, 0.95, 70.0)
nbits += not (lz.m == 0 and lz.shape == (50, 50) and np.array_equal(np.asarray(lz), np.eye(50)))
check(f'decimator_fir and LazyDecimator differing from decimator ({n} cases)', nbits, 0, '{:.0f}')

# ---------------------------------------------------------------------------------------------- JAX program cache
print('\nJAX program cache (plan_signature)')
api._JAX_PROGRAMS.clear()
f1 = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend='jax', T=16, levels=2, window=False)
f2 = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend='jax', T=16, levels=2, window=False)
check('programs after two formers of one geometry', len(api._JAX_PROGRAMS), 1, '{:.0f}')
check('second former not sharing the first one\'s program', float(f1._fn is not f2._fn), 0, '{:.0f}')
# another antenna path (the track moved 2 m and turned slightly) with an equal signature reuses the program
ant2 = col.ant + np.array([0.0, 2.0, 1.0]) + 1e-4 * np.c_[np.zeros(P), np.zeros(P), np.arange(P) - P / 2]
plan1 = ffbp2.make_plan(sim.Collect(col.fmin, col.df, K, col.ant, 0.5), 128, 128, 0.5, 0.5, T=16, nlev=2, e1=grid['e1'], e2=grid['e2'])
plan2 = ffbp2.make_plan(sim.Collect(col.fmin, col.df, K, ant2, 0.5), 128, 128, 0.5, 0.5, T=16, nlev=2, e1=grid['e1'], e2=grid['e2'])
plan3 = ffbp2.make_plan(sim.Collect(col.fmin, col.df, K, col.ant, 0.5), 128, 96, 0.5, 0.5, T=16, nlev=2, e1=grid['e1'], e2=grid['e2'])
hash(ffbp2.plan_signature(plan1))
check('signature of the moved track differing', float(ffbp2.plan_signature(plan1) != ffbp2.plan_signature(plan2)), 0, '{:.0f}')
check('signature of a narrower grid equal', float(ffbp2.plan_signature(plan1) == ffbp2.plan_signature(plan3)), 0, '{:.0f}')
S2 = sim.simulate_brute(sim.Collect(col.fmin, col.df, K, ant2, 0.5), tg, np.ones(30)).astype(np.complex64)
f3 = fastsar.ImageFormer(ant2, col.fmin, col.df, K, **grid, backend='jax', T=16, levels=2, window=False)
check('programs after the moved track', len(api._JAX_PROGRAMS), 1, '{:.0f}')
img_cached = f3(S2)
api._JAX_PROGRAMS.clear()
img_fresh = fastsar.ImageFormer(ant2, col.fmin, col.df, K, **grid, backend='jax', T=16, levels=2, window=False)(S2)
check('reused program against a fresh compile (max abs difference)', float(np.abs(img_cached - img_fresh).max()), 0, '{:.2g}')
img_cpu = fastsar.ImageFormer(ant2, col.fmin, col.df, K, **grid, backend='cpu', T=16, levels=2, window=False)(S2)
check('reused program against the cpu backend', rel_db(img_cached, img_cpu), -80)

# ---------------------------------------------------------------------------------------------- C++ input paths
print('\nC++ backend input paths')
col_cpu = sim.Collect(col.fmin, col.df, K, col.ant, 0.5)
plan = ffbp2.make_plan(col_cpu, 128, 128, 0.5, 0.5, T=16, nlev=2, e1=grid['e1'], e2=grid['e2'])
print(f'  level 0: Dk {plan["levels"][0]["Dk"]}, Dp {plan["levels"][0]["Dp"]}, {plan["levels"][0]["C"]} children')
assert plan['levels'][0]['Dk'] > 1                        # the complex64 path needs range decimation at level 0
form = ffbp_cpu.make_ffbp_cpu(plan, ffbp2.collection_arrays(plan, col.ant))
Sw = S * np.outer(*api._window(P, K))
img_c64 = form(np.ascontiguousarray(Sw, np.complex64))                 # read in place (rot_fir_k_cplx)
img_split = form(Sw.astype(np.complex128))                            # split float32 planes (rot_fir_k)
check('complex64 in place against split planes', rel_db(img_c64, img_split), -105)
img_ng1 = form(np.ascontiguousarray(Sw, np.complex64), ng=1)
check('level-0 groups of 1 against 8 (max abs difference)', float(np.abs(img_ng1 - img_c64).max()), 0, '{:.2g}')
with env(FASTSAR_CPU_GROUP_GB=1e-6):
    img_env = form(np.ascontiguousarray(Sw, np.complex64))
check('FASTSAR_CPU_GROUP_GB=1e-6 against groups of 8 (max abs diff)', float(np.abs(img_env - img_c64).max()), 0, '{:.2g}')
check('complex64 in place against the JAX program', rel_db(img_c64, fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid,
      backend='jax', T=16, levels=2, window=False)(Sw)), -80)

# ---------------------------------------------------------------------------------------------- final_weights
print('\naperture weights (api.final_weights, ImageFormer(aperture_weight=...))')
e1v, e2v = np.array(grid['e1']), np.array(grid['e2'])
hann = lambda p: 0.5 - 0.5 * np.cos(2 * np.pi * (np.asarray(p) + 0.5) / P)


def wfun(q, p):
    """A smooth weight over pulses that also varies across the scene: [len(p), len(q)]."""
    q = np.asarray(q)
    return hann(p)[:, None] * (1.0 + 0.004 * (q @ e1v) - 0.003 * (q @ e2v))[None, :]


plan3l = ffbp2.make_plan(col_cpu, 128, 128, 0.5, 0.5, T=16, nlev=3, e1=grid['e1'], e2=grid['e2'])
for nm, pl in (('2 levels', plan), ('3 levels', plan3l)):
    wf = api.final_weights(pl, wfun, P)
    idx, D = np.arange(pl['final']['P'], dtype=np.float64), 1.0
    for lv in reversed(pl['levels']):
        idx, D = lv['Dp'] * idx + float(lv['pidx'][0]), D * lv['Dp']
    cen = pl['final']['cen']
    ref = np.zeros_like(wf)
    for f in range(len(idx)):
        p = np.arange(int(np.ceil(idx[f] - D / 2)), int(np.floor(idx[f] + D / 2)) + 1)
        p = p[(p >= 0) & (p < P)]
        if not len(p):                    # centred beyond the collection: the nearest pulse's weight
            p = np.array([int(np.clip(np.rint(idx[f]), 0, P - 1))])
        ref[:, f] = wfun(cen, p).mean(0)
    check(f'final_weights against the mean over pulses ({nm}, D {D:.0f}, max abs)', float(np.abs(wf - ref).max()), 0.006, '{:.4f}')
    g = api.final_weights(pl, wfun, P, grad=True)
    gx = hann(np.clip(np.rint(idx), 0, P - 1))[None, :] * 0.004
    check(f'gradient along e1 against 0.004 x pulse weight ({nm}, max abs)',
          float(np.abs(g[1] - gx).max() / 0.004), 0.006, '{:.4f}')
    check(f'weight of the gradient call against the plain call ({nm})', float(np.abs(g[0] - wf).max()), 0, '{:.2g}')

for b in ('cpu', 'jax'):
    plain = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend=b, T=16, levels=2, window=False)
    ones = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend=b, T=16, levels=2, window=False,
                               aperture_weight=lambda q, p: np.ones((len(p), len(q))))
    check(f'{b}: unit aperture weight against none', rel_db(ones(S), plain(S)), -100)
    pw = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend=b, T=16, levels=2, window=False,
                             aperture_weight=lambda q, p: np.repeat(hann(p)[:, None], len(q), 1))
    check(f'{b}: Hann over pulses in the final stage against S x Hann', rel_db(pw(S), plain(S * hann(np.arange(P))[:, None])), -43)
img_w = {b: fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend=b, T=16, levels=2, window=False,
                                aperture_weight=wfun)(S) for b in ('cpu', 'jax')}
check('pixel-dependent weights: cpu against jax', rel_db(img_w['cpu'], img_w['jax']), -75)
with env(FASTSAR_WEIGHT_GRAD=0):
    img_ng = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **grid, backend='cpu', T=16, levels=2, window=False,
                                 aperture_weight=wfun)(S)
print(f'  (FASTSAR_WEIGHT_GRAD=0 against the gradient form: {rel_db(img_ng, img_w["cpu"]):.1f} dB)')

# ---------------------------------------------------------------------------------------------- beam_span, fill_gaps
print('\npulse spans and gaps')
nfail, ncase = 0, 0
for P_ in (1, 7, 64, 65, 1000, 4097):
    for center, half in ((0.3, 0.004), (0.5, 0.02), (0.0, 0.05), (1.0, 0.05), (0.97, 0.2), (1.6, 0.01), (-0.5, 0.2), (0.5, 3.0)):
        def beam(idx, pts, P_=P_, center=center, half=half):
            # pulse p sees point x at u = (p/P - center - x/1000)/half
            p = np.arange(P_)[idx] if isinstance(idx, slice) else np.asarray(idx)
            return (p[:, None] / P_ - center - np.asarray(pts)[None, :, 0] / 1000.0) / half
        pts = np.c_[np.linspace(-5, 5, 9), np.zeros(9), np.zeros(9)]
        hit = np.nonzero((np.abs(beam(np.arange(P_), pts)) <= 1).any(1))[0]
        want = None if len(hit) == 0 else (int(hit[0]), int(hit[-1]) + 1)
        for step in (1, 16, 64, 1000):
            ncase += 1
            got = pt.beam_span(beam, P_, pts, 1.0, step=step)
            if got != want:
                nfail += 1
                print(f'    P {P_} center {center} half {half} step {step}: {got} != {want}')
check(f'beam_span against every pulse ({ncase} cases)', nfail, 0, '{:.0f}')

a = np.c_[np.arange(40) * 0.5, np.zeros(40), np.full(40, 100.0)]
keep = np.ones(40, bool)
keep[[5, 6, 7, 20, 30, 31]] = False
af, idx = pt.fill_gaps(a[keep])
err = max(float(np.abs(af - a).max()), float(np.abs(idx - np.nonzero(keep)[0]).max())) if af.shape == a.shape else np.inf
check('fill_gaps: positions and index of a track with 3 gaps', err, 1e-12, '{:.2g}')
af, idx = pt.fill_gaps(a)
check('fill_gaps: no gap returns the input', float(np.abs(af - a).max() + np.abs(idx - np.arange(40)).max()), 0, '{:.2g}')
for nm, x in (('one position', a[:1]), ('a platform at rest', np.zeros((6, 3)))):
    af, idx = pt.fill_gaps(x)
    check(f'fill_gaps: {nm} returned unchanged', float(np.abs(af - x).max() + np.abs(idx - np.arange(len(x))).max()), 0, '{:.2g}')

# ---------------------------------------------------------------------------------------------- range profiles
print('\nshared range profiles (patches.range_profiles)')
fx = dict(S=S, fmin=col.fmin, df=col.df, ref=np.linalg.norm(col.ant, axis=1))
prof = pt.range_profiles(fx)
c = np.array([10.0, -5.0, 0.0])
pts9 = c + np.array([[i, j, 0.0] for i in np.linspace(-8, 8, 9) for j in np.linspace(-8, 8, 9)])
lo, hi = 20, P - 30
d, *rest_d = pt.patch_history(fx, col.ant, c, pts9, lo, hi)
s, *rest_s = pt.patch_history(fx, col.ant, c, pts9, lo, hi, prof=prof)
print(f'  patch history {d.shape} of {S.shape}: gated to {d.shape[1]} samples')
assert d.shape[1] < K and d.shape == s.shape
# they differ at the gate's edges (the fractional delay wraps on the gate, not on the full profile), which the
# margin keeps away from the patch: compare range profiles over the gate's central half, and the images below
qs, qd = np.fft.fftshift(np.fft.ifft(s, axis=1), 1), np.fft.fftshift(np.fft.ifft(d, axis=1), 1)
h4 = d.shape[1] // 4
print(f'  (whole gated history: {rel_db(s, d):.1f} dB)')
check('patch_history with profiles: central half of the gate', rel_db(qs[:, h4:-h4], qd[:, h4:-h4]), -90)
check('their antenna paths and frequency grids (max abs difference)',
      float(np.abs(rest_s[0] - rest_d[0]).max() + abs(rest_s[1] - rest_d[1]) + abs(rest_s[2] - rest_d[2])), 0, '{:.2g}')
o = -32.0 * e1v - 32.0 * e2v
mos = {}
for v in ('1', '0'):
    with env(FASTSAR_SHARED_PROFILES=v):
        mos[v] = pt.form_mosaic(fx, col.ant, o, 128, 128, 0.5, 0.5, e1v, e2v, patch=(64, 64), backend='cpu')
check('form_mosaic: FASTSAR_SHARED_PROFILES=1 against 0', rel_db(mos['1'], mos['0']), -90)

# ---------------------------------------------------------------------------------------------- mosaic weights, prefetch
print('\nmosaic: azimuth weights and prefetch (stripmap scene)')
p = sm.make_params()
tgs = np.array([[-24.3, p.r0 - 88.6], [0.37, p.r0 + 0.41], [22.6, p.r0 + 89.3]])
raw = sm.simulate(p, tgs)
r, x = sm.axes(p)
rows = (int(np.argmin(np.abs(x - tgs[:, 0].min()))) - 48, int(np.argmin(np.abs(x - tgs[:, 0].max()))) + 48)
kw = dict(rwin='taylor', awin='taylor', rows=rows, backend='cpu', patch=(96, 128))
t = time.perf_counter()
img_k = pt.form_stripmap(raw, p, **kw)[0]
print(f'  {p.na} pulses x {p.nr} samples, grid {img_k.shape}: {time.perf_counter() - t:.1f} s')
with env(FASTSAR_WEIGHT_TERMS=1):
    img_t = pt.form_stripmap(raw, p, **kw)[0]
check('azimuth window in the final stage against separable terms', rel_db(img_k, img_t), -50)
kw['awin'] = None
img_on = pt.form_stripmap(raw, p, **kw)[0]
with env(FASTSAR_MOSAIC_PREFETCH=0):
    img_off = pt.form_stripmap(raw, p, **kw)[0]
check('prefetch on against off (max abs difference)', float(np.abs(img_on - img_off).max()), 0, '{:.2g}')
calls = []


def beam_fails(idx, pts):
    calls.append(1)
    if len(calls) > 3:
        raise RuntimeError('beam failed on purpose')
    return pt.stripmap_beam(p, pt.straight_track(p))(idx, pts)


fxs = pt.echoes_to_fx(raw, p, rwin='taylor')
t = time.perf_counter()
expect('an error in the prefetch thread', lambda: pt.form_mosaic(fxs, pt.straight_track(p), (x[rows[0]], r[0], 0.0), 192, 256,
       x[1] - x[0], r[1] - r[0], patch=(96, 128), beam=beam_fails), RuntimeError)
check('time to report it (s)', time.perf_counter() - t, 30, '{:.1f}')

# ---------------------------------------------------------------------------------------------- input checks
print('\ninput checks')
g = (128, 128, 0.5, 0.5, grid['e1'], grid['e2'])
fi = lambda S_, ant=col.ant, **k: fastsar.form_image(S_, ant, col.fmin, col.df, *g, **{'backend': 'cpu', 'T': 16, 'levels': 2, **k})
ref_img = fi(S)
for nm, S_ in (('complex128', S.astype(np.complex128)), ('Fortran order', np.asfortranarray(S)),
               ('a strided view', np.repeat(S, 2, axis=1)[:, ::2])):
    for b in ('cpu', 'jax'):
        check(f'{b}: {nm} history against complex64', rel_db(fi(S_, backend=b), fi(S, backend=b)), -120)
former = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, *g, backend='cpu', T=16, levels=2, window=False)
check('cpu, window=False: Fortran order against C order (max abs)', float(np.abs(former(np.asfortranarray(S)) - former(S)).max()), 0, '{:.2g}')
check('zero history: largest pixel', float(np.abs(fi(np.zeros_like(S))).max()), 0, '{:.2g}')
check('zero history, jax: largest pixel', float(np.abs(fi(np.zeros_like(S), backend='jax')).max()), 0, '{:.2g}')
Sn = S.copy()
Sn[7, 3] = np.nan
expect('NaN sample', lambda: fi(Sn), ValueError)
expect('inf sample, backproject', lambda: fastsar.backproject(np.where(Sn == Sn, S, np.inf), col.ant, col.fmin, col.df, np.zeros((4, 3))), ValueError)
expect('real history', lambda: fi(S.real), TypeError)
expect('integer history', lambda: fi(np.ones(S.shape, np.int16)), TypeError)
expect('1-D history', lambda: fi(S[0]), ValueError)
expect('pulse count differing from ant', lambda: fi(S[:-1]), ValueError)
expect('ant [P, 2]', lambda: fi(S, col.ant[:, :2]), ValueError)
expect('one pulse', lambda: fi(S[:1], col.ant[:1]), ValueError)
expect('nx = 0', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 0, 128, 0.5, 0.5, backend='cpu'), ValueError)
expect('nx = 12.5', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 12.5, 128, 0.5, 0.5, backend='cpu'), ValueError)
expect('negative spacing', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 128, 128, -0.5, 0.5, backend='cpu'), ValueError)
expect('e1 not unit length', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 128, 128, 0.5, 0.5, (0, 2.0, 0), (1.0, 0, 0)), ValueError)
expect('unknown backend', lambda: fi(S, backend='gpu'), ValueError)
expect('unknown algorithm', lambda: fi(S, algorithm='rda'), ValueError)
if 'cuda' not in fastsar.available_backends():
    expect('cuda on a machine without one', lambda: fi(S, backend='cuda'), ValueError)
if 'tpu' not in fastsar.available_backends() and not os.environ.get('FFBP_FORCE_TPU_KERNELS'):
    expect('tpu on a machine without one', lambda: fi(S, backend='tpu'), ValueError)
expect('former called on a wrong shape', lambda: former(S[:, :-1]), ValueError)
expect('former called with NaN', lambda: former(Sn), ValueError)
small = fastsar.form_image(S, col.ant, col.fmin, col.df, 6, 4, 0.5, 0.5, grid['e1'], grid['e2'], backend='cpu')
check('6 x 4 grid (smaller than a tile) against the 128 x 128 image', rel_db(small, ref_img[61:67, 62:66]), -80)
pts = fastsar.plane_points(8, 8, 0.5, 0.5, grid['e1'], grid['e2'])
expect('backproject: points [..., 2]', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts[..., :2]), ValueError)
expect('backproject: ref of the wrong length', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts, ref=np.ones(3)), ValueError)
expect('backproject: rcv of the wrong length', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts, rcv=col.ant[1:]), ValueError)
expect('backproject: unknown backend', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='gpu'), ValueError)
check('backproject: Fortran-order complex128 against complex64 (max abs)', float(np.abs(
    fastsar.backproject(np.asfortranarray(S.astype(np.complex128)), col.ant, col.fmin, col.df, pts, backend='cpu')
    - fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='cpu')).max() / np.abs(S).sum()), 1e-6, '{:.2g}')
mos_args = (col.ant, o, 64, 64, 0.5, 0.5, e1v, e2v)
expect('form_mosaic: fx without ref', lambda: pt.form_mosaic({k: v for k, v in fx.items() if k != 'ref'}, *mos_args), ValueError)
expect('form_mosaic: ref of the wrong length', lambda: pt.form_mosaic({**fx, 'ref': fx['ref'][:5]}, *mos_args), ValueError)
expect('form_mosaic: fewer positions than pulses', lambda: pt.form_mosaic(fx, col.ant[:-2], *mos_args[1:]), ValueError)
expect('form_mosaic: NaN sample', lambda: pt.form_mosaic({**fx, 'S': Sn}, *mos_args), ValueError)
expect('form_mosaic: patch of 0 pixels', lambda: pt.form_mosaic(fx, *mos_args, patch=(0, 64)), ValueError)
expect('form_mosaic: unknown backend', lambda: pt.form_mosaic(fx, *mos_args, backend='gpu'), ValueError)
expect('form_cphd: unknown mode', lambda: fastsar.form_cphd('missing.cphd', mode='scan'), ValueError)
expect('form_cphd: unknown backend', lambda: fastsar.form_cphd('missing.cphd', backend='gpu'), ValueError)
expect('form_cphd: negative spacing', lambda: fastsar.form_cphd('missing.cphd', spacing=-1.0), ValueError)

# ImageFormer(ref=...): samples referenced to a range other than |ant| (a bistatic half path) form the same image
print('\nreference range')
rng_ = np.random.default_rng(3)
colr = sim.make_collect(res=0.3, scene=60.0, r0=5e3)
posr = np.stack([rng_.uniform(-25, 25, 30), rng_.uniform(-25, 25, 30), np.zeros(30)], 1)
Sr = sim.simulate_brute(colr, posr, rng_.standard_normal(30) + 1j * rng_.standard_normal(30)).astype(np.complex64)
r0r = np.linalg.norm(colr.ant, axis=1)
refr = r0r + 0.2e-3 + 0.05e-3 * np.sin(np.linspace(0, 3, len(r0r)))
fr = colr.fmin + colr.df * np.arange(colr.K)
Sr2 = (Sr * np.exp(-4j * np.pi * fr[None] / 299792458.0 * (r0r - refr)[:, None])).astype(np.complex64)
kwr = dict(nx=128, ny=128, spx=0.3, spy=0.3, backend='cpu', window=False)
ir = fastsar.form_image(Sr, colr.ant, colr.fmin, colr.df, **kwr)
ir2 = fastsar.form_image(Sr2, colr.ant, colr.fmin, colr.df, ref=refr, **kwr)
check('form_image with ref against the |ant|-referenced image', 10 * np.log10(np.sum(abs(ir2 - ir) ** 2) / np.sum(abs(ir) ** 2)), -90)

print(f'\n{time.perf_counter() - t_start:.0f} s')
if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
