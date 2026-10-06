"""Stripmap focusing on a simulated X-band scene of five point targets (swath edges, center, two between), broadside
and at 5 degrees of squint: omega-k and RDA (with and without secondary range compression) against float64
time-domain backprojection. Prints, per target, resolution against the expected value, PSLR, ISLR, position error
and the complex image error against backprojection on a 64 by 64 pixel patch around the target."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fastsar import stripmap as sm
from fastsar.quality import point_target

H = 32                                                       # patch half-size (pixels)
RWIN, AWIN = 'taylor', None                                  # azimuth weighting: the two-way sinc^2 pattern alone


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


def run(squint):
    p = sm.make_params(squint_deg=squint)
    r0 = p.r0
    tg = np.array([[-24.3, r0 - 88.6], [11.7, r0 - 41.2], [0.37, r0 + 0.41], [-9.8, r0 + 46.9], [22.6, r0 + 89.3]])
    t = time.perf_counter()
    raw = sm.simulate(p, tg)
    print(f'\nsquint {squint:.0f} deg: {p.na} pulses x {p.nr} samples, f_dc {p.fdc:.0f} Hz, prf {p.prf:.0f} Hz, '
          f'simulated in {time.perf_counter() - t:.2f} s')
    rr, ra = sm.resolution(p, RWIN, AWIN)
    print(f'expected resolution: range {rr:.3f} m (c/2B = {sm.C / 2 / p.B:.3f} m, Taylor), '
          f'azimuth {ra:.3f} m (La/2 = {p.La / 2:.3f} m, two-way pattern over |u| <= 1)')
    r, x = sm.axes(p)
    dr, dx = r[1] - r[0], x[1] - x[0]
    ij = [(int(round((xt - x[0]) / dx)), int(round((rt - r[0]) / dr))) for xt, rt in tg]
    t = time.perf_counter()
    bp = [sm.backproject(raw, p, x[i - H:i + H, None], r[None, j - H:j + H], rwin=RWIN, awin=AWIN) for i, j in ij]
    print(f'backprojection on {len(tg)} patches of {2 * H}x{2 * H}: {time.perf_counter() - t:.1f} s')
    imgs = {}
    for name, alg, kw in (('omega-k', 'omegak', {}), ('omega-k f64', 'omegak', dict(dtype='float64')),
                          ('RDA+SRC', 'rda', dict(src=True)), ('RDA', 'rda', dict(src=False))):
        sm.focus_stripmap(raw, p, alg, rwin=RWIN, awin=AWIN, **kw)            # compile
        t = time.perf_counter()
        imgs[name] = sm.focus_stripmap(raw, p, alg, rwin=RWIN, awin=AWIN, **kw)[0]
        print(f'{name:12s} {time.perf_counter() - t:.2f} s')
    e64 = rel_db(imgs['omega-k'], imgs['omega-k f64'])
    print(f'omega-k float32 against float64, whole image: {e64:.1f} dB')
    assert e64 < -70, e64
    i, j = ij[2]
    b, _, _ = sm.focus_stripmap(raw, p, 'bp', rows=np.arange(i - 2, i + 2), cols=np.arange(j - 2, j + 2), rwin=RWIN, awin=AWIN)
    assert np.allclose(b, bp[2][H - 2:H + 2, H - 2:H + 2], rtol=1e-12, atol=0)
    out = {}
    print('target (x, r - r0) m | algorithm | res az, rg (m) | PSLR az, rg (dB) | ISLR az, rg, 2-D (dB) | '
          'position error az, rg (m) | error vs BP (dB)')
    for n, ((xt, rt), (i, j)) in enumerate(zip(tg, ij)):
        for name, im in [('BP', None)] + list(imgs.items()):
            patch = bp[n] if im is None else im[i - H:i + H, j - H:j + H]
            m = point_target(patch, (H, H), dx, dr, half=24)
            ex = x[i - H] + m['i'] * dx - xt
            er = r[j - H] + m['j'] * dr - rt
            e = None if im is None else rel_db(patch, bp[n])
            out[n, name] = dict(m, ex=ex, er=er, err=e)
            print(f'({xt:6.2f}, {rt - r0:6.2f}) {name:12s} {m["res_az"]:.3f} {m["res_rg"]:.3f} | '
                  f'{m["pslr_az"]:6.1f} {m["pslr_rg"]:6.1f} | {m["islr_az"]:6.1f} {m["islr_rg"]:6.1f} {m["islr_2d"]:6.1f} | '
                  f'{ex:+.4f} {er:+.4f}' + ('' if e is None else f' | {e:6.1f}'))
    return p, (rr, ra), out


for squint in (0.0, 5.0):
    p, (rr, ra), out = run(squint)
    for (n, name), m in out.items():
        assert abs(m['ex']) < 0.05 * ra and abs(m['er']) < 0.05 * rr, (squint, n, name, m['ex'], m['er'])
        assert abs(m['res_az'] / out[n, 'BP']['res_az'] - 1) < 0.03, (squint, n, name, m['res_az'])
        assert abs(m['res_rg'] / out[n, 'BP']['res_rg'] - 1) < 0.03, (squint, n, name, m['res_rg'])
        if name in ('omega-k', 'omega-k f64'):
            assert m['err'] < -30, (squint, n, name, m['err'])
        if name == 'RDA+SRC':
            assert m['err'] < -30, (squint, n, name, m['err'])
    if squint == 0:
        for n in range(5):
            assert abs(out[n, 'BP']['res_az'] / ra - 1) < 0.03 and abs(out[n, 'BP']['res_rg'] / rr - 1) < 0.03, out[n, 'BP']
print('ok')
