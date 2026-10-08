"""Patch-wise factorized backprojection of long apertures (fastsar.patches) on the X-band scene of test_stripmap
(five point targets across a 200 m swath, Taylor range window).

(a) Straight track: the patch mosaic on the zero-Doppler grid against float64 time-domain backprojection
    (stripmap.backproject), broadside and at 5 degrees of squint, without and with a Taylor azimuth window, from raw
    echoes and from range-compressed ones; the same patches formed by exact backprojection separate the error of
    the patching (gate, pulse selection, weights) from that of FFBP. Then a frequency-domain phase history cut to
    the band and referenced to a point that moves with the platform, which exercises the per-pulse reference ranges
    and the zero padding of the frequency axis.
(b) A track with smooth cross-track and vertical motion of about a meter, from 3 km altitude onto flat ground:
    omega-k on the nominal straight track against exact backprojection with the true positions, and the patch
    mosaic on a ground-plane grid with the true positions against the same.
(c) Dropped pulses: a simulated spotlight with 34 of 2122 pulses missing in gaps of 5 to 10 pulses. The mosaic, which
    fills the gaps with zero pulses (patches.fill_gaps), against exact backprojection of the remaining pulses.

Errors are complex, on 64 by 64 pixel patches around the targets, after the best complex gain. Resolution, PSLR and
position come from fastsar.quality."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fastsar import stripmap as sm, patches as pt, io
from fastsar.quality import point_target

H = 32                                                       # half-size of the comparison patches (pixels)
RWIN = 'taylor'


def rel_db(a, b):
    g = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    return 10 * np.log10(np.sum(np.abs(g * a - b) ** 2) / np.sum(np.abs(b) ** 2))


def targets(p):
    r0 = p.r0
    return np.array([[-24.3, r0 - 88.6], [11.7, r0 - 41.2], [0.37, r0 + 0.41], [-9.8, r0 + 46.9], [22.6, r0 + 89.3]])


def compare(img, refs, ij, d_az, d_rg, name, off=(0, 0), metrics=True):
    """Print per-target metrics of img against the reference patches; -> list of (error dB, metrics, ref metrics).
    metrics=False prints only the error and the peak amplitude (of the patch, over the reference's peak)."""
    out = []
    for n, ((i, j), ref) in enumerate(zip(ij, refs)):
        pa = img[i - H - off[0]:i + H - off[0], j - H - off[1]:j + H - off[1]]
        e = rel_db(pa, ref)
        if not metrics:
            out.append((e, None, None))
            print(f'  {name:16s} target {n}: peak {np.abs(pa).max() / np.abs(ref).max():.4f} | error {e:6.1f} dB')
            continue
        m, mr = point_target(pa, (H, H), d_az, d_rg, half=24), point_target(ref, (H, H), d_az, d_rg, half=24)
        out.append((e, m, mr))
        print(f'  {name:16s} target {n}: res az {m["res_az"]:.3f} ({mr["res_az"]:.3f}) rg {m["res_rg"]:.3f} '
              f'({mr["res_rg"]:.3f}) m | PSLR az {m["pslr_az"]:6.1f} ({mr["pslr_az"]:6.1f}) rg {m["pslr_rg"]:6.1f} '
              f'({mr["pslr_rg"]:6.1f}) dB | shift az {(m["i"] - mr["i"]) * d_az:+.4f} rg {(m["j"] - mr["j"]) * d_rg:+.4f} m '
              f'| peak {m["peak"] / mr["peak"]:.4f} | error {e:6.1f} dB')
    return out


def check(res, bound, name):
    """Error below bound; resolution within 1 %, position within 0.01 pixel, and PSLR within 0.5 dB or, for low
    sidelobes, within -54 dB of the peak in amplitude."""
    for e, m, mr in res:
        assert e < bound, (name, e)
        assert abs(m['res_az'] / mr['res_az'] - 1) < 0.01 and abs(m['res_rg'] / mr['res_rg'] - 1) < 0.01, (name, m, mr)
        for k in ('pslr_az', 'pslr_rg'):
            assert abs(m[k] - mr[k]) < 0.5 or abs(10 ** (m[k] / 20) - 10 ** (mr[k] / 20)) < 0.002, (name, k, m, mr)
        assert abs(m['i'] - mr['i']) < 0.01 and abs(m['j'] - mr['j']) < 0.01, (name, m, mr)


# ------------------------------------------------------------------ (a) straight track

def straight(squint, awin, compressed=False, backends=('cpu',)):
    p = sm.make_params(squint_deg=squint)
    tg = targets(p)
    raw = sm.simulate(p, tg)
    data = sm.range_compress(raw, p, RWIN, dtype='float64') if compressed else raw
    kw = dict(compressed=compressed, rwin=None if compressed else RWIN, awin=awin)
    r, x = sm.axes(p)
    dr, dx = r[1] - r[0], x[1] - x[0]
    ij = [(int(round((xt - x[0]) / dx)), int(round((rt - r[0]) / dr))) for xt, rt in tg]
    rows = (min(i for i, _ in ij) - 2 * H, max(i for i, _ in ij) + 2 * H)
    print(f'\nsquint {squint:.0f} deg, azimuth window {awin}, {"range-compressed" if compressed else "raw"} input: '
          f'{p.na} pulses x {p.nr} samples, grid {rows[1] - rows[0]} x {p.nr} pixels of {dx:.3f} x {dr:.3f} m')
    t = time.perf_counter()
    refs = [sm.backproject(data, p, x[i - H:i + H, None], r[None, j - H:j + H], **kw) for i, j in ij]
    print(f'  stripmap.backproject on {len(ij)} patches of {2 * H}x{2 * H}: {time.perf_counter() - t:.1f} s')
    out = {}
    for name, be, exact in [('patches exact', 'cpu', True)] + [(f'patches {b}', b, False) for b in backends]:
        info = []
        t = time.perf_counter()
        img, _, _ = pt.form_stripmap(data, p, rows=rows, backend=be, exact=exact, info=info, **kw)
        dt = time.perf_counter() - t
        if not exact:
            print(f'  {name}: {len(info)} patches of 128x128 in {dt:.1f} s; pulses per patch '
                  f'{min(d["pulses"][1] - d["pulses"][0] for d in info)}-{max(d["pulses"][1] - d["pulses"][0] for d in info)}, '
                  f'K {info[0]["K"]}, (T, sub) {sorted({(d["T"], d["sub"]) for d in info})}, levels {info[0]["levels"]}, '
                  f'weight terms {sorted({d["terms"] for d in info})}, predicted error '
                  f'{max(d["predicted_error_db"] for d in info):.1f} dB at worst')
        out[name] = compare(img, refs, ij, dx, dr, name, (rows[0], 0))
    return p, out


for squint, awin in ((0.0, None), (5.0, None), (0.0, 'taylor')):
    p, out = straight(squint, awin, backends=('cpu', 'jax') if squint == 0.0 else ('cpu',))
    check(out['patches exact'], -50, 'exact')
    for name, res in out.items():
        check(res, -45, name)
p, out = straight(0.0, None, compressed=True)
for name, res in out.items():
    check(res, -45, name)
print('ok')

# a phase history in the frequency domain: the band only (no room for the guard, so the frequency axis is padded),
# compensated per pulse to the point at the platform's along-track position and the swath center's range
p = sm.make_params()
raw = sm.simulate(p, targets(p))
fx = pt.echoes_to_fx(raw, p, rwin=RWIN)
f = fx['fmin'] + fx['df'] * np.arange(fx['S'].shape[1])
k = np.nonzero((f >= fx['band'][0]) & (f <= fx['band'][1]))[0]
ant = pt.straight_track(p)
mov = ant + np.array([0.0, p.r0, 0.0])
ref = np.linalg.norm(ant - mov, axis=1)
fx2 = dict(S=io.rereference(fx['S'][:, k], f[k[0]], fx['df'], fx['ref'] - ref), fmin=f[k[0]], df=fx['df'], ref=ref)
r, x = sm.axes(p)
grid = ((x[384], r[0], 0.0), 256, p.nr, x[1] - x[0], r[1] - r[0])
kw = dict(beam=pt.stripmap_beam(p, ant), backend='cpu')
a, b = pt.form_mosaic(fx, ant, *grid, **kw), pt.form_mosaic(fx2, ant, *grid, **kw)
e = rel_db(b, a)
print(f'\nFX input cut to the band, moving reference point: {fx2["S"].shape[1]} of {fx["S"].shape[1]} samples; '
      f'mosaic against the echo input: {e:.1f} dB over {grid[1]} x {grid[2]} pixels')
assert e < -60, e
print('ok')

# ------------------------------------------------------------------ (b) non-linear track

h = 3000.0
p = sm.make_params()
eta = p.eta
nom = pt.straight_track(p, h)
dev = np.stack([np.zeros_like(eta), 0.8 * np.sin(2 * np.pi * eta / 1.3 + 0.4) + 0.25 * np.sin(2 * np.pi * eta / 0.47 + 1.1),
                0.6 * np.cos(2 * np.pi * eta / 1.7 + 0.2) - 0.2 * np.sin(2 * np.pi * eta / 0.61)], 1)
true = nom + dev
tg = targets(p)
tg3 = np.c_[tg[:, 0], np.sqrt(tg[:, 1] ** 2 - h * h), np.zeros(len(tg))]
print(f'\nnon-linear track: {h:.0f} m altitude, cross-track motion {np.ptp(dev[:, 1]):.2f} m and vertical '
      f'{np.ptp(dev[:, 2]):.2f} m peak to peak over the {p.na / p.prf:.2f} s of the scene; range change at the swath '
      f'center up to {np.abs(np.linalg.norm(true - [0, tg3[2, 1], 0], axis=1) - np.linalg.norm(nom - [0, tg3[2, 1], 0], axis=1)).max():.2f} m')
a, b = pt.simulate(p, nom, tg3), sm.simulate(p, tg)
d = np.abs(a - b).max() / np.abs(b).max()
print(f'echoes on the nominal track (3-D, ground targets) against stripmap.simulate: max difference {d:.1e} of the peak')
assert d < 1e-8
raw = pt.simulate(p, true, tg3)
beam = pt.stripmap_beam(p, true)

# omega-k on the nominal track, on its zero-Doppler grid, against exact backprojection with the true positions
img, r, x = sm.focus_stripmap(raw, p, 'omegak', rwin=RWIN)
dr, dx = r[1] - r[0], x[1] - x[0]
ij = [(int(round((xt - x[0]) / dx)), int(round((rt - r[0]) / dr))) for xt, rt in tg]


def ground(xs, rs):
    X, R = np.meshgrid(xs, rs, indexing='ij')
    return np.stack([X, np.sqrt(R * R - h * h), 0 * X], -1)


t = time.perf_counter()
refs = [pt.backproject(raw, p, true, ground(x[i - H:i + H], r[j - H:j + H]), rwin=RWIN, beam=beam) for i, j in ij]
print(f'exact backprojection with the true positions, zero-Doppler grid: {time.perf_counter() - t:.1f} s')
ok = compare(img, refs, ij, dx, dr, 'omega-k nominal', metrics=False)
for e, m, mr in ok:
    assert e > -10, e
# the control: echoes from the straight track, omega-k against exact backprojection in the same 3-D geometry
raw0 = pt.simulate(p, nom, tg3)
img0 = sm.focus_stripmap(raw0, p, 'omegak', rwin=RWIN)[0]
refs0 = [pt.backproject(raw0, p, nom, ground(x[i - H:i + H], r[j - H:j + H]), rwin=RWIN) for i, j in ij]
for e, m, mr in compare(img0, refs0, ij, dx, dr, 'omega-k straight', metrics=False):
    assert e < -55, e

# patch mosaic on a ground-plane grid with the true positions
spy = 1.5
y0 = tg3[:, 1].min() - 2 * H * spy
i0 = min(i for i, _ in ij) - 2 * H
grid = ((x[i0], y0, 0.0), max(i for i, _ in ij) + 2 * H - i0, int(np.ceil((tg3[:, 1].max() - y0) / spy)) + 2 * H, dx, spy)
ijg = [(int(round((xt - x[i0]) / dx)), int(round((yt - y0) / spy))) for xt, yt, _ in tg3]
pts = [ground(x[i0] + np.arange(i - H, i + H) * dx, np.hypot(y0 + np.arange(j - H, j + H) * spy, h)) for i, j in ijg]
t = time.perf_counter()
refg = [pt.backproject(raw, p, true, q, rwin=RWIN, beam=beam) for q in pts]
print(f'exact backprojection with the true positions, ground grid of {dx:.3f} x {spy:.3f} m: {time.perf_counter() - t:.1f} s')
fx = pt.echoes_to_fx(raw, p, rwin=RWIN)
res = {}
for name, track, kw in (('patches true', true, dict(crop=8)), ('patches exact', true, dict(crop=8, exact=True)),
                        ('patches nominal', nom, dict())):
    info = []
    t = time.perf_counter()
    img = pt.form_mosaic(fx, track, *grid, beam=pt.stripmap_beam(p, track), backend='cpu', info=info, **kw)
    print(f'{name}: {len(info)} patches in {time.perf_counter() - t:.1f} s' + ('' if kw.get('exact') else
          f', (T, sub) {sorted({(d["T"], d["sub"]) for d in info})}, predicted error '
          f'{max(d["predicted_error_db"] for d in info):.1f} dB at worst'))
    res[name] = compare(img, refg, ijg, dx, spy, name, metrics=track is true)
check(res['patches true'], -45, 'patches true')
check(res['patches exact'], -50, 'patches exact')
for e, m, mr in res['patches nominal']:
    assert e > -10, e

# (c) dropped pulses
from fastsar import sim
from fastsar.bp import backproject
rng = np.random.default_rng(2)
col = sim.make_collect(res=0.5, scene=600.0, r0=5e3)
tg = np.stack([rng.uniform(-25, 25, 80), rng.uniform(-25, 25, 80), np.zeros(80)], 1)
S = sim.simulate_brute(col, tg, rng.standard_normal(80) + 1j * rng.standard_normal(80)).astype(np.complex64)
P = len(col.ant)
keep = np.ones(P, bool)
for c in (P // 5, P // 2, 4 * P // 5):
    keep[c:c + rng.integers(5, 11)] = False
    keep[c + 30:c + 40:2] = False
e1, e2 = np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
o = -32.0 * e1 - 32.0 * e2
X, Y = np.meshgrid(np.arange(128) * 0.5, np.arange(128) * 0.5, indexing='ij')
ex = backproject(S[keep], col.ant[keep], col.fmin, col.df, o + X[..., None] * e1 + Y[..., None] * e2, backend='cpu', window=False, upsample=16)
fx = dict(S=S[keep], fmin=col.fmin, df=col.df, ref=np.linalg.norm(col.ant[keep], axis=1))
img = pt.form_mosaic(fx, col.ant[keep], o, 128, 128, 0.5, 0.5, e1, e2, patch=(64, 64), backend='cpu')
e = 10 * np.log10(np.sum(np.abs(img - ex) ** 2) / np.sum(np.abs(ex) ** 2))
print(f'dropped pulses ({(~keep).sum()} of {P}): {e:.1f} dB')
assert e < -45, e                # -46.4 dB when the limit was set; -43.5 dB without the zero pulses

# (d) the image does not depend on how many threads prepare the patches (FASTSAR_MOSAIC_PREFETCH)
imgs = {}
for nw in ('0', '1', '4'):
    os.environ['FASTSAR_MOSAIC_PREFETCH'] = nw
    imgs[nw] = pt.form_mosaic(fx, col.ant[keep], o, 128, 128, 0.5, 0.5, e1, e2, patch=(32, 32), backend='cpu')
os.environ.pop('FASTSAR_MOSAIC_PREFETCH')
for nw in ('1', '4'):
    assert np.array_equal(imgs[nw], imgs['0']), f'{nw} prefetch workers changed the image'
print('prefetch workers 0, 1, 4: identical images')

# (e) the JAX backend (histories staged on the worker threads, pulse counts and gates padded onto shared programs)
# against the CPU backend on the same mosaic, with the dropped pulses
jx = pt.form_mosaic(fx, col.ant[keep], o, 128, 128, 0.5, 0.5, e1, e2, patch=(64, 64), backend='jax')
e = 10 * np.log10(np.sum(np.abs(jx - ex) ** 2) / np.sum(np.abs(ex) ** 2))
print(f'jax mosaic against exact backprojection: {e:.1f} dB')
assert e < -45, e
print('ok')
