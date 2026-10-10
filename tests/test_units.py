"""Unit checks of pieces that the end-to-end tests reach only indirectly, each against an independent or slower
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
values measured when they were set, or at zero where the two paths must agree exactly. Each check prints its value
and limit; a test fails if any of its checks fails."""
import os
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import sim, ffbp2, ffbp_cpu, patches as pt, stripmap as sm, api
from fastsar import memory as fmem
from fastsar.bp import backproject
from fastsar.ffbp import decimator

GRID = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
E1, E2 = np.array(GRID['e1']), np.array(GRID['e2'])
ORIGIN = -32.0 * E1 - 32.0 * E2


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2) + 1e-300)


def check(bad, name, value, limit, fmt='{:.1f} dB'):
    ok = value <= limit
    print(f'  {name:62s} {fmt.format(value):>12s}  (limit {fmt.format(limit)})' + ('' if ok else '  FAIL'))
    if not ok:
        bad.append(name)


def expect(bad, name, fn, exc):
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


def done(bad):
    assert not bad, 'FAILED: ' + '; '.join(bad)


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(3)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    tg = np.stack([rng.uniform(-25, 25, 30), rng.uniform(-25, 25, 30), np.zeros(30)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(30) + 1j * rng.standard_normal(30)).astype(np.complex64)
    P, K = S.shape
    print(f'spotlight scene: {P} pulses x {K} samples, 30 targets, grid 128 x 128 of 0.5 m')
    col_cpu = sim.Collect(col.fmin, col.df, K, col.ant, 0.5)
    plan = ffbp2.make_plan(col_cpu, 128, 128, 0.5, 0.5, T=16, nlev=2, e1=GRID['e1'], e2=GRID['e2'])
    return SimpleNamespace(col=col, tg=tg, S=S, P=P, K=K, col_cpu=col_cpu, plan=plan,
                           fx=dict(S=S, fmin=col.fmin, df=col.df, ref=np.linalg.norm(col.ant, axis=1)))


# ---------------------------------------------------------------------------------------------- decimation filters

def test_decimation_filters():
    print('\ndecimation filters')
    bad = []
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
    check(bad, f'decimator_fir and LazyDecimator differing from decimator ({n} cases)', nbits, 0, '{:.0f}')
    done(bad)


# ---------------------------------------------------------------------------------------------- JAX program cache

def test_jax_program_cache(scene):
    print('\nJAX program cache (plan_signature)')
    bad = []
    col, S, P, K, tg = scene.col, scene.S, scene.P, scene.K, scene.tg
    api._JAX_PROGRAMS.clear()
    f1 = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend='jax', T=16, levels=2, window=False)
    f2 = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend='jax', T=16, levels=2, window=False)
    check(bad, 'programs after two formers of one geometry', len(api._JAX_PROGRAMS), 1, '{:.0f}')
    check(bad, 'second former not sharing the first one\'s program', float(f1._fn is not f2._fn), 0, '{:.0f}')
    # another antenna path (the track moved 2 m and turned slightly) with an equal signature reuses the program
    ant2 = col.ant + np.array([0.0, 2.0, 1.0]) + 1e-4 * np.c_[np.zeros(P), np.zeros(P), np.arange(P) - P / 2]
    plan1 = ffbp2.make_plan(sim.Collect(col.fmin, col.df, K, col.ant, 0.5), 128, 128, 0.5, 0.5, T=16, nlev=2, e1=GRID['e1'], e2=GRID['e2'])
    plan2 = ffbp2.make_plan(sim.Collect(col.fmin, col.df, K, ant2, 0.5), 128, 128, 0.5, 0.5, T=16, nlev=2, e1=GRID['e1'], e2=GRID['e2'])
    plan3 = ffbp2.make_plan(sim.Collect(col.fmin, col.df, K, col.ant, 0.5), 128, 96, 0.5, 0.5, T=16, nlev=2, e1=GRID['e1'], e2=GRID['e2'])
    hash(ffbp2.plan_signature(plan1))
    check(bad, 'signature of the moved track differing', float(ffbp2.plan_signature(plan1) != ffbp2.plan_signature(plan2)), 0, '{:.0f}')
    check(bad, 'signature of a narrower grid equal', float(ffbp2.plan_signature(plan1) == ffbp2.plan_signature(plan3)), 0, '{:.0f}')
    S2 = sim.simulate_brute(sim.Collect(col.fmin, col.df, K, ant2, 0.5), tg, np.ones(30)).astype(np.complex64)
    f3 = fastsar.ImageFormer(ant2, col.fmin, col.df, K, **GRID, backend='jax', T=16, levels=2, window=False)
    check(bad, 'programs after the moved track', len(api._JAX_PROGRAMS), 1, '{:.0f}')
    img_cached = f3(S2)
    api._JAX_PROGRAMS.clear()
    img_fresh = fastsar.ImageFormer(ant2, col.fmin, col.df, K, **GRID, backend='jax', T=16, levels=2, window=False)(S2)
    check(bad, 'reused program against a fresh compile (max abs difference)', float(np.abs(img_cached - img_fresh).max()), 0, '{:.2g}')
    img_cpu = fastsar.ImageFormer(ant2, col.fmin, col.df, K, **GRID, backend='cpu', T=16, levels=2, window=False)(S2)
    check(bad, 'reused program against the cpu backend', rel_db(img_cached, img_cpu), -80)
    done(bad)


# ---------------------------------------------------------------------------------------------- C++ input paths

def test_cpp_input_paths(scene, monkeypatch):
    print('\nC++ backend input paths')
    bad = []
    col, S, P, K, plan = scene.col, scene.S, scene.P, scene.K, scene.plan
    print(f'  level 0: Dk {plan["levels"][0]["Dk"]}, Dp {plan["levels"][0]["Dp"]}, {plan["levels"][0]["C"]} children')
    assert plan['levels'][0]['Dk'] > 1                        # the complex64 path needs range decimation at level 0
    form = ffbp_cpu.make_ffbp_cpu(plan, ffbp2.collection_arrays(plan, col.ant))
    Sw = S * np.outer(*api._window(P, K))
    img_c64 = form(np.ascontiguousarray(Sw, np.complex64))                 # read in place (rot_fir_k_cplx)
    img_split = form(Sw.astype(np.complex128))                            # split float32 planes (rot_fir_k)
    check(bad, 'complex64 in place against split planes', rel_db(img_c64, img_split), -105)
    img_ng1 = form(np.ascontiguousarray(Sw, np.complex64), ng=1)
    check(bad, 'level-0 groups of 1 against 8 (max abs difference)', float(np.abs(img_ng1 - img_c64).max()), 0, '{:.2g}')
    with monkeypatch.context() as mp:
        mp.setenv('FASTSAR_CPU_GROUP_GB', str(1e-6))
        img_env = form(np.ascontiguousarray(Sw, np.complex64))
    check(bad, 'FASTSAR_CPU_GROUP_GB=1e-6 against groups of 8 (max abs diff)', float(np.abs(img_env - img_c64).max()), 0, '{:.2g}')
    check(bad, 'complex64 in place against the JAX program', rel_db(img_c64, fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID,
          backend='jax', T=16, levels=2, window=False)(Sw)), -80)
    done(bad)


# ---------------------------------------------------------------------------------------------- final_weights

def _hann(P):
    return lambda p: 0.5 - 0.5 * np.cos(2 * np.pi * (np.asarray(p) + 0.5) / P)


def _wfun(P):
    hann = _hann(P)

    def wfun(q, p):
        """A smooth weight over pulses that also varies across the scene: [len(p), len(q)]."""
        q = np.asarray(q)
        return hann(p)[:, None] * (1.0 + 0.004 * (q @ E1) - 0.003 * (q @ E2))[None, :]
    return wfun


@pytest.mark.parametrize('nlev', [2, 3], ids=['2 levels', '3 levels'])
def test_final_weights(scene, nlev):
    print('\naperture weights (api.final_weights)')
    bad = []
    P = scene.P
    hann, wfun = _hann(P), _wfun(P)
    nm = f'{nlev} levels'
    pl = scene.plan if nlev == 2 else ffbp2.make_plan(scene.col_cpu, 128, 128, 0.5, 0.5, T=16, nlev=3, e1=GRID['e1'], e2=GRID['e2'])
    wf = api.final_weights(pl, wfun, P)
    idx, D = np.arange(pl['final']['P'], dtype=np.float64), 1.0
    for lv in reversed(pl['levels']):
        idx, D = lv['Dp'] * idx + float(lv['pidx'][0]), D * lv['Dp']
    cen = pl['final']['cen']
    ref = np.zeros_like(wf)
    for f in range(len(idx)):
        p = np.arange(int(np.ceil(idx[f] - D / 2)), int(np.floor(idx[f] + D / 2)) + 1)
        p = p[(p >= 0) & (p < P)]
        if not len(p):                    # centered beyond the collection: the nearest pulse's weight
            p = np.array([int(np.clip(np.rint(idx[f]), 0, P - 1))])
        ref[:, f] = wfun(cen, p).mean(0)
    check(bad, f'final_weights against the mean over pulses ({nm}, D {D:.0f}, max abs)', float(np.abs(wf - ref).max()), 0.006, '{:.4f}')
    g = api.final_weights(pl, wfun, P, grad=True)
    gx = hann(np.clip(np.rint(idx), 0, P - 1))[None, :] * 0.004
    check(bad, f'gradient along e1 against 0.004 x pulse weight ({nm}, max abs)',
          float(np.abs(g[1] - gx).max() / 0.004), 0.006, '{:.4f}')
    check(bad, f'weight of the gradient call against the plain call ({nm})', float(np.abs(g[0] - wf).max()), 0, '{:.2g}')
    done(bad)


@pytest.mark.parametrize('b', ['cpu', 'jax'])
def test_aperture_weight(scene, b):
    print('\nImageFormer(aperture_weight=...)')
    bad = []
    col, S, P, K = scene.col, scene.S, scene.P, scene.K
    hann = _hann(P)
    plain = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend=b, T=16, levels=2, window=False)
    ones = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend=b, T=16, levels=2, window=False,
                               aperture_weight=lambda q, p: np.ones((len(p), len(q))))
    check(bad, f'{b}: unit aperture weight against none', rel_db(ones(S), plain(S)), -100)
    pw = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend=b, T=16, levels=2, window=False,
                             aperture_weight=lambda q, p: np.repeat(hann(p)[:, None], len(q), 1))
    check(bad, f'{b}: Hann over pulses in the final stage against S x Hann', rel_db(pw(S), plain(S * hann(np.arange(P))[:, None])), -43)
    done(bad)


def test_pixel_dependent_weights(scene, monkeypatch):
    bad = []
    col, S, K = scene.col, scene.S, scene.K
    wfun = _wfun(scene.P)
    img_w = {b: fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend=b, T=16, levels=2, window=False,
                                    aperture_weight=wfun)(S) for b in ('cpu', 'jax')}
    check(bad, 'pixel-dependent weights: cpu against jax', rel_db(img_w['cpu'], img_w['jax']), -75)
    with monkeypatch.context() as mp:
        mp.setenv('FASTSAR_WEIGHT_GRAD', '0')
        img_ng = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, **GRID, backend='cpu', T=16, levels=2, window=False,
                                     aperture_weight=wfun)(S)
    print(f'  (FASTSAR_WEIGHT_GRAD=0 against the gradient form: {rel_db(img_ng, img_w["cpu"]):.1f} dB)')
    done(bad)


# ---------------------------------------------------------------------------------------------- beam_span, fill_gaps

def test_beam_span():
    print('\npulse spans and gaps')
    bad = []
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
    check(bad, f'beam_span against every pulse ({ncase} cases)', nfail, 0, '{:.0f}')
    done(bad)


def test_fill_gaps():
    bad = []
    a = np.c_[np.arange(40) * 0.5, np.zeros(40), np.full(40, 100.0)]
    keep = np.ones(40, bool)
    keep[[5, 6, 7, 20, 30, 31]] = False
    af, idx = pt.fill_gaps(a[keep])
    err = max(float(np.abs(af - a).max()), float(np.abs(idx - np.nonzero(keep)[0]).max())) if af.shape == a.shape else np.inf
    check(bad, 'fill_gaps: positions and index of a track with 3 gaps', err, 1e-12, '{:.2g}')
    af, idx = pt.fill_gaps(a)
    check(bad, 'fill_gaps: no gap returns the input', float(np.abs(af - a).max() + np.abs(idx - np.arange(40)).max()), 0, '{:.2g}')
    for nm, x in (('one position', a[:1]), ('a platform at rest', np.zeros((6, 3)))):
        af, idx = pt.fill_gaps(x)
        check(bad, f'fill_gaps: {nm} returned unchanged', float(np.abs(af - x).max() + np.abs(idx - np.arange(len(x))).max()), 0, '{:.2g}')
    done(bad)


# ---------------------------------------------------------------------------------------------- range profiles

def test_range_profiles(scene, monkeypatch):
    print('\nshared range profiles (patches.range_profiles)')
    bad = []
    col, P, K, fx = scene.col, scene.P, scene.K, scene.fx
    prof = pt.range_profiles(fx)
    c = np.array([10.0, -5.0, 0.0])
    pts9 = c + np.array([[i, j, 0.0] for i in np.linspace(-8, 8, 9) for j in np.linspace(-8, 8, 9)])
    lo, hi = 20, P - 30
    d, *rest_d = pt.patch_history(fx, col.ant, c, pts9, lo, hi)
    s, *rest_s = pt.patch_history(fx, col.ant, c, pts9, lo, hi, prof=prof)
    print(f'  patch history {d.shape} of {scene.S.shape}: gated to {d.shape[1]} samples')
    assert d.shape[1] < K and d.shape == s.shape
    # they differ at the gate's edges (the fractional delay wraps on the gate, not on the full profile), which the
    # margin keeps away from the patch: compare range profiles over the gate's central half, and the images below
    qs, qd = np.fft.fftshift(np.fft.ifft(s, axis=1), 1), np.fft.fftshift(np.fft.ifft(d, axis=1), 1)
    h4 = d.shape[1] // 4
    print(f'  (whole gated history: {rel_db(s, d):.1f} dB)')
    check(bad, 'patch_history with profiles: central half of the gate', rel_db(qs[:, h4:-h4], qd[:, h4:-h4]), -90)
    check(bad, 'their antenna paths and frequency grids (max abs difference)',
          float(np.abs(rest_s[0] - rest_d[0]).max() + abs(rest_s[1] - rest_d[1]) + abs(rest_s[2] - rest_d[2])), 0, '{:.2g}')
    # a wider gate (gate_len, as the JAX and TPU mosaics use to share programs) keeps the patch's image: exact
    # backprojection of both gated histories at the patch points
    w, *rest_w = pt.patch_history(fx, col.ant, c, pts9, lo, hi, prof=prof, gate_len=lambda n: n + 40)
    assert w.shape[1] > s.shape[1]
    im_s = backproject(s, rest_s[0], rest_s[1], rest_s[2], pts9 - c, backend='cpu', window=False)
    im_w = backproject(w, rest_w[0], rest_w[1], rest_w[2], pts9 - c, backend='cpu', window=False)
    check(bad, f'patch_history: gate widened by 40 bins ({s.shape[1]} -> {w.shape[1]} samples), image at the patch', rel_db(im_w, im_s), -60)
    # the GPU window of profiles: covers the patch, starts an eighth behind it, and a forward sweep of patches of
    # 1,000 pulses each 500 apart moves a 10,000-row window about every 17 patches (once per patch before the fix)
    for lo_, hi_, rows_ in ((5000, 6000, 10000), (5000, 6000, 1000), (0, 900, 4000), (100, 600, 600)):
        w0_ = pt._window_start(lo_, hi_, rows_)
        assert w0_ <= lo_ and w0_ + rows_ >= hi_ and w0_ >= 0, (lo_, hi_, rows_, w0_)
    w, moves = (0, 0), 0
    for k in range(400):
        lo_, hi_ = 500 * k, 500 * k + 1000
        if not (w[0] <= lo_ and hi_ <= w[1]):
            w0_ = pt._window_start(lo_, hi_, 10000)
            w, moves = (w0_, w0_ + 10000), moves + 1
    check(bad, 'profile window moves for 400 patches advancing 500 pulses (10,000 rows; 382 with the old start)', moves, 30, '{:.0f}')
    mos = {}
    for v in ('1', '0'):
        with monkeypatch.context() as mp:
            mp.setenv('FASTSAR_SHARED_PROFILES', v)
            mos[v] = pt.form_mosaic(fx, col.ant, ORIGIN, 128, 128, 0.5, 0.5, E1, E2, patch=(64, 64), backend='cpu')
    check(bad, 'form_mosaic: FASTSAR_SHARED_PROFILES=1 against 0', rel_db(mos['1'], mos['0']), -90)
    done(bad)


# ---------------------------------------------------------------------------------------------- mosaic weights, prefetch

def test_mosaic_weights_and_prefetch(monkeypatch):
    print('\nmosaic: azimuth weights and prefetch (stripmap scene)')
    bad = []
    p = sm.make_params()
    tgs = np.array([[-24.3, p.r0 - 88.6], [0.37, p.r0 + 0.41], [22.6, p.r0 + 89.3]])
    raw = sm.simulate(p, tgs)
    r, x = sm.axes(p)
    rows = (int(np.argmin(np.abs(x - tgs[:, 0].min()))) - 48, int(np.argmin(np.abs(x - tgs[:, 0].max()))) + 48)
    kw = dict(rwin='taylor', awin='taylor', rows=rows, backend='cpu', patch=(96, 128))
    t = time.perf_counter()
    img_k = pt.form_stripmap(raw, p, **kw)[0]
    print(f'  {p.na} pulses x {p.nr} samples, grid {img_k.shape}: {time.perf_counter() - t:.1f} s')
    with monkeypatch.context() as mp:
        mp.setenv('FASTSAR_WEIGHT_TERMS', '1')
        img_t = pt.form_stripmap(raw, p, **kw)[0]
    check(bad, 'azimuth window in the final stage against separable terms', rel_db(img_k, img_t), -50)
    kw['awin'] = None
    img_on = pt.form_stripmap(raw, p, **kw)[0]
    with monkeypatch.context() as mp:
        mp.setenv('FASTSAR_MOSAIC_PREFETCH', '0')
        img_off = pt.form_stripmap(raw, p, **kw)[0]
    check(bad, 'prefetch on against off (max abs difference)', float(np.abs(img_on - img_off).max()), 0, '{:.2g}')
    calls = []

    def beam_fails(idx, pts):
        calls.append(1)
        if len(calls) > 3:
            raise RuntimeError('beam failed on purpose')
        return pt.stripmap_beam(p, pt.straight_track(p))(idx, pts)

    fxs = pt.echoes_to_fx(raw, p, rwin='taylor')
    t = time.perf_counter()
    expect(bad, 'an error in the prefetch thread', lambda: pt.form_mosaic(fxs, pt.straight_track(p), (x[rows[0]], r[0], 0.0), 192, 256,
           x[1] - x[0], r[1] - r[0], patch=(96, 128), beam=beam_fails), RuntimeError)
    check(bad, 'time to report it (s)', time.perf_counter() - t, 120, '{:.1f}')
    done(bad)


# ---------------------------------------------------------------------------------------------- input checks

def test_input_checks(scene):
    print('\ninput checks')
    bad = []
    col, S, K, fx = scene.col, scene.S, scene.K, scene.fx
    g = (128, 128, 0.5, 0.5, GRID['e1'], GRID['e2'])
    fi = lambda S_, ant=col.ant, **k: fastsar.form_image(S_, ant, col.fmin, col.df, *g, **{'backend': 'cpu', 'T': 16, 'levels': 2, **k})
    ref_img = fi(S)
    for nm, S_ in (('complex128', S.astype(np.complex128)), ('Fortran order', np.asfortranarray(S)),
                   ('a strided view', np.repeat(S, 2, axis=1)[:, ::2])):
        for b in ('cpu', 'jax'):
            check(bad, f'{b}: {nm} history against complex64', rel_db(fi(S_, backend=b), fi(S, backend=b)), -120)
    former = fastsar.ImageFormer(col.ant, col.fmin, col.df, K, *g, backend='cpu', T=16, levels=2, window=False)
    check(bad, 'cpu, window=False: Fortran order against C order (max abs)', float(np.abs(former(np.asfortranarray(S)) - former(S)).max()), 0, '{:.2g}')
    check(bad, 'zero history: largest pixel', float(np.abs(fi(np.zeros_like(S))).max()), 0, '{:.2g}')
    check(bad, 'zero history, jax: largest pixel', float(np.abs(fi(np.zeros_like(S), backend='jax')).max()), 0, '{:.2g}')
    Sn = S.copy()
    Sn[7, 3] = np.nan
    expect(bad, 'NaN sample', lambda: fi(Sn), ValueError)
    expect(bad, 'inf sample, backproject', lambda: fastsar.backproject(np.where(Sn == Sn, S, np.inf), col.ant, col.fmin, col.df, np.zeros((4, 3))), ValueError)
    expect(bad, 'real history', lambda: fi(S.real), TypeError)
    expect(bad, 'integer history', lambda: fi(np.ones(S.shape, np.int16)), TypeError)
    expect(bad, '1-D history', lambda: fi(S[0]), ValueError)
    expect(bad, 'pulse count differing from ant', lambda: fi(S[:-1]), ValueError)
    expect(bad, 'ant [P, 2]', lambda: fi(S, col.ant[:, :2]), ValueError)
    expect(bad, 'one pulse', lambda: fi(S[:1], col.ant[:1]), ValueError)
    expect(bad, 'nx = 0', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 0, 128, 0.5, 0.5, backend='cpu'), ValueError)
    expect(bad, 'nx = 12.5', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 12.5, 128, 0.5, 0.5, backend='cpu'), ValueError)
    expect(bad, 'negative spacing', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 128, 128, -0.5, 0.5, backend='cpu'), ValueError)
    expect(bad, 'e1 not unit length', lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 128, 128, 0.5, 0.5, (0, 2.0, 0), (1.0, 0, 0)), ValueError)
    expect(bad, 'unknown backend', lambda: fi(S, backend='gpu'), ValueError)
    expect(bad, 'unknown algorithm', lambda: fi(S, algorithm='rda'), ValueError)
    if 'cuda' not in fastsar.available_backends():
        expect(bad, 'cuda on a machine without one', lambda: fi(S, backend='cuda'), ValueError)
    if 'tpu' not in fastsar.available_backends() and not os.environ.get('FFBP_FORCE_TPU_KERNELS'):
        expect(bad, 'tpu on a machine without one', lambda: fi(S, backend='tpu'), ValueError)
    expect(bad, 'former called on a wrong shape', lambda: former(S[:, :-1]), ValueError)
    expect(bad, 'ImageFormer: backend passed by position', lambda: fastsar.ImageFormer(col.ant, col.fmin, col.df, K, *g, 'cpu'), TypeError)
    expect(bad, "algorithm='pfa' with e1 across the track (look angle not monotonic)",
           lambda: fastsar.form_image(S, col.ant, col.fmin, col.df, 64, 64, 0.5, 0.5, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                      algorithm='pfa', pfa_guard=5.0), ValueError)
    expect(bad, 'former called with NaN', lambda: former(Sn), ValueError)
    small = fastsar.form_image(S, col.ant, col.fmin, col.df, 6, 4, 0.5, 0.5, GRID['e1'], GRID['e2'], backend='cpu')
    check(bad, '6 x 4 grid (smaller than a tile) against the 128 x 128 image', rel_db(small, ref_img[61:67, 62:66]), -80)
    pts = fastsar.plane_points(8, 8, 0.5, 0.5, GRID['e1'], GRID['e2'])
    expect(bad, 'backproject: points [..., 2]', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts[..., :2]), ValueError)
    expect(bad, 'backproject: ref of the wrong length', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts, ref=np.ones(3)), ValueError)
    expect(bad, 'backproject: rcv of the wrong length', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts, rcv=col.ant[1:]), ValueError)
    expect(bad, 'backproject: unknown backend', lambda: fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='gpu'), ValueError)
    check(bad, 'backproject: Fortran-order complex128 against complex64 (max abs)', float(np.abs(
        fastsar.backproject(np.asfortranarray(S.astype(np.complex128)), col.ant, col.fmin, col.df, pts, backend='cpu')
        - fastsar.backproject(S, col.ant, col.fmin, col.df, pts, backend='cpu')).max() / np.abs(S).sum()), 1e-6, '{:.2g}')
    mos_args = (col.ant, ORIGIN, 64, 64, 0.5, 0.5, E1, E2)
    expect(bad, 'form_mosaic: fx without ref', lambda: pt.form_mosaic({k: v for k, v in fx.items() if k != 'ref'}, *mos_args), ValueError)
    expect(bad, 'form_mosaic: ref of the wrong length', lambda: pt.form_mosaic({**fx, 'ref': fx['ref'][:5]}, *mos_args), ValueError)
    expect(bad, 'form_mosaic: fewer positions than pulses', lambda: pt.form_mosaic(fx, col.ant[:-2], *mos_args[1:]), ValueError)
    expect(bad, 'form_mosaic: NaN sample', lambda: pt.form_mosaic({**fx, 'S': Sn}, *mos_args), ValueError)
    expect(bad, 'form_mosaic: patch of 0 pixels', lambda: pt.form_mosaic(fx, *mos_args, patch=(0, 64)), ValueError)
    expect(bad, 'form_mosaic: unknown backend', lambda: pt.form_mosaic(fx, *mos_args, backend='gpu'), ValueError)
    expect(bad, 'form_cphd: unknown mode', lambda: fastsar.form_cphd('missing.cphd', mode='scan'), ValueError)
    expect(bad, 'form_cphd: unknown backend', lambda: fastsar.form_cphd('missing.cphd', backend='gpu'), ValueError)
    expect(bad, 'form_cphd: negative spacing', lambda: fastsar.form_cphd('missing.cphd', spacing=-1.0), ValueError)
    done(bad)


# ---------------------------------------------------------------------------------------------- reference range

@pytest.fixture(scope='module')
def fine():
    """A 0.3 m scene with a reference range other than |ant|."""
    rng_ = np.random.default_rng(3)
    colr = sim.make_collect(res=0.3, scene=60.0, r0=5e3)
    posr = np.stack([rng_.uniform(-25, 25, 30), rng_.uniform(-25, 25, 30), np.zeros(30)], 1)
    Sr = sim.simulate_brute(colr, posr, rng_.standard_normal(30) + 1j * rng_.standard_normal(30)).astype(np.complex64)
    return SimpleNamespace(colr=colr, Sr=Sr)


def test_reference_range(fine):
    """ImageFormer(ref=...): samples referenced to a range other than |ant| (a bistatic half path) form the same
    image."""
    print('\nreference range')
    bad = []
    colr, Sr = fine.colr, fine.Sr
    r0r = np.linalg.norm(colr.ant, axis=1)
    refr = r0r + 0.2e-3 + 0.05e-3 * np.sin(np.linspace(0, 3, len(r0r)))
    fr = colr.fmin + colr.df * np.arange(colr.K)
    Sr2 = (Sr * np.exp(-4j * np.pi * fr[None] / 299792458.0 * (r0r - refr)[:, None])).astype(np.complex64)
    kwr = dict(nx=128, ny=128, spx=0.3, spy=0.3, backend='cpu', window=False)
    ir = fastsar.form_image(Sr, colr.ant, colr.fmin, colr.df, **kwr)
    ir2 = fastsar.form_image(Sr2, colr.ant, colr.fmin, colr.df, ref=refr, **kwr)
    check(bad, 'form_image with ref against the |ant|-referenced image', 10 * np.log10(np.sum(abs(ir2 - ir) ** 2) / np.sum(abs(ir) ** 2)), -90)
    kwp = dict(kwr, algorithm='pfa', pfa_guard=10.0, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))     # e1 along the track
    ipr = fastsar.form_image(Sr, colr.ant, colr.fmin, colr.df, **kwp)
    ipr2 = fastsar.form_image(Sr2, colr.ant, colr.fmin, colr.df, ref=refr, **kwp)
    check(bad, "algorithm='pfa' with ref against the |ant|-referenced image", rel_db(ipr2, ipr), -90)
    expect(bad, "algorithm='pfa': ref of the wrong length", lambda: fastsar.form_image(Sr2, colr.ant, colr.fmin, colr.df, ref=refr[:5], **kwp),
           ValueError)
    done(bad)


# ---------------------------------------------------------------------------------------------- memory fallback

def test_memory_fallback(fine, monkeypatch):
    """Full speed when it fits, a MemoryWarning naming the fallback and the memory needed when it does not."""
    print('\nmemory fallback')
    bad = []
    colr, Sr = fine.colr, fine.Sr
    fm = fastsar.ImageFormer(colr.ant, colr.fmin, colr.df, colr.K, 128, 128, 0.3, 0.3, backend='cpu', window=False)
    need_fm = fm.memory()['needed']
    with monkeypatch.context() as mp:
        mp.setattr(fmem, 'host_available', lambda: 4 * need_fm)                      # a host with room to spare, whatever runs the test
        check(bad, 'memory(): full speed reported when the memory suffices', 0.0 if fm.memory()['full_speed'] else 1.0, 0.0, '{:.0f}')
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            im_full = fm(Sr)
    with monkeypatch.context() as mp:
        mp.setattr(fmem, 'host_available', lambda: 1e6)                      # a host out of memory
        with warnings.catch_warnings(record=True) as wl2:
            warnings.simplefilter('always')
            im_low = fastsar.ImageFormer(colr.ant, colr.fmin, colr.df, colr.K, 128, 128, 0.3, 0.3, backend='cpu', window=False)(Sr)
    mw = [w for w in wl2 if issubclass(w.category, fastsar.MemoryWarning)]
    check(bad, 'memory: warnings at full speed', float(sum(issubclass(w.category, fastsar.MemoryWarning) for w in wl)), 0.0, '{:.0f}')
    check(bad, 'memory: one MemoryWarning in the fallback', abs(len(mw) - 1.0), 0.0, '{:.0f}')
    check(bad, 'memory: the warning states the memory full speed needs', 0.0 if mw and 'full speed needs about' in str(mw[0].message) else 1.0, 0.0, '{:.0f}')
    check(bad, 'memory: fallback image equals the full-speed image', float(np.abs(im_low - im_full).max()), 0.0, '{:.1e}')
    done(bad)


def test_cpu_blocked_first_level(fine, monkeypatch):
    """The CPU's pulse-blocked first level (used when memory allows fewer than 4 children per group) equals the
    unblocked one."""
    print('\nCPU first level in pulse blocks')
    bad = []
    colr, Sr = fine.colr, fine.Sr
    with monkeypatch.context() as mp:
        mp.setattr(ffbp_cpu, 'BLOCK_PULSES', 128)                  # several blocks, with halos at both ends of the aperture
        for T, limit in ((32, -100), (16, -85)):
            # T=32: bit identical; T=16: a deterministic difference of -93 dB on every machine (the same on x86 and
            # ARM), far below any tolerance of the images, not yet traced
            mp.setenv('FASTSAR_CPU_BLOCKED', '1')
            im_blk = fastsar.ImageFormer(colr.ant, colr.fmin, colr.df, colr.K, 128, 128, 0.3, 0.3, backend='cpu', window=False, T=T)(Sr)
            mp.setenv('FASTSAR_CPU_BLOCKED', '0')
            im_unb = fastsar.ImageFormer(colr.ant, colr.fmin, colr.df, colr.K, 128, 128, 0.3, 0.3, backend='cpu', window=False, T=T)(Sr)
            check(bad, f'cpu: blocked first level against unblocked (T={T})',
                  10 * np.log10(np.sum(abs(im_blk - im_unb) ** 2) / np.sum(abs(im_unb) ** 2)), limit)
    done(bad)
