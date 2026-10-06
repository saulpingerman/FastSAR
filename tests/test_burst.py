"""Burst modes on the X-band geometry of tests/test_stripmap.py (9.6 GHz, 100 MHz, PRF 650 Hz, 200 m/s, 1.5 m antenna).
TOPS: one burst of 512 pulses steered at 0.12 rad/s (alpha = 1/4 at 5 km, Doppler span 2.7 PRF). ScanSAR: two
subswaths at 5.0 and 5.25 km, alternating bursts of 128 pulses. Targets at two ranges are illuminated at the
beginning, middle and end of a TOPS burst; ScanSAR targets cross the beam center 0.25 s before the burst, at its
first, middle and last pulse and 0.25 s after it, and are simulated and focused one at a time, since the
unweighted burst response has sinc sidelobes that reach the neighboring targets' patches. Prints the sign checks
of the steering, then per target resolution against the expected value, PSLR, ISLR, position error and the complex
error of each focused burst image against float64 backprojection of the same burst's pulses on a 64 by 64 pixel
patch; and the ScanSAR mosaic of one subswath."""
import os, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fastsar import burst as bm, stripmap as sm
from fastsar.quality import point_target

H = 32                                                       # patch half-size for the comparison (pixels)
RWIN = 'taylor'


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


def wrap(f, prf):
    return np.mod(f + prf / 2, prf) - prf / 2


def x_at(b, eta, r):
    """Along-track position of the target at zero-Doppler range r that crosses the beam center at slow time eta."""
    return b.v * eta + r * np.tan(b.psi(eta))


def signs(b, tg):
    """For each target alone: the energy centroid of its echoes in slow time is the beam-center crossing, and the
    pulse-pair Doppler there is the steered centroid f_dc = 2 v sin(psi)/lambda (modulo the PRF), at baseband after
    the deramp."""
    r, _ = sm.axes(b)
    for (xt, rt), eb in tg:
        rc = sm.range_compress(bm.simulate(b, [(xt, rt)]), b, dtype='float64')
        dr = np.exp(-1j * bm._ramp(b, b.eta))[:, None] * rc
        j = int(round((rt / np.sqrt(1 - np.sin(b.psi(eb)) ** 2) - r[0]) / (r[1] - r[0])))
        E = np.sum(np.abs(rc[:, j - 8:j + 9]) ** 2, 1)
        ec = np.sum(E * b.eta) / E.sum()
        k = int(round((ec - b.eta0) * b.prf))
        ff = [np.angle(np.sum(np.conj(d[k - 4:k + 4, j - 2:j + 3]) * d[k - 3:k + 5, j - 2:j + 3])) * b.prf / (2 * np.pi)
              for d in (rc, dr)]
        fdc = b.fdc_t(b.eta[k] + 0.5 / b.prf)
        print(f'  target crossing at {eb:+.4f} s: echo centroid {ec:+.4f} s, f_dc {fdc:+7.1f} Hz, '
              f'pulse-pair {ff[0]:+7.1f} Hz (f_dc mod prf {wrap(fdc, b.prf):+7.1f}), deramped {ff[1]:+6.1f} Hz')
        assert abs(ec - eb) < 1 / b.prf, (eb, ec)
        assert abs(wrap(ff[0] - fdc, b.prf)) < 10 and abs(ff[1]) < 10, (ff, fdc)
        if abs(wrap(2 * fdc, b.prf)) > 50:
            assert abs(wrap(ff[0] + fdc, b.prf)) > 40                  # the opposite sign would be detected


def check(b, raws, tg, algs, half, label):
    """Focus one burst with each algorithm, backproject the patches and measure. raws: one raw burst holding all
    targets, or one per target (the processing is linear)."""
    out, imgs = {}, {}
    for name, alg, kw in algs:
        bm.focus_burst(raws[0], b, alg, rwin=RWIN, **kw)                   # compile
        t = time.perf_counter()
        imgs[name] = [bm.focus_burst(w, b, alg, rwin=RWIN, **kw) for w in raws]
        print(f'  {name:12s} {(time.perf_counter() - t) / len(raws):.2f} s per image of '
              f'{imgs[name][0][0].shape[0]} x {imgs[name][0][0].shape[1]}')
    _, r, x = imgs[algs[0][0]][0]
    dr, dx = r[1] - r[0], x[1] - x[0]
    rr = sm.resolution(b, RWIN)[0]
    print(f'  expected range resolution {rr:.3f} m; azimuth from each target\'s illumination (closed form)')
    print('  target (x, r - r0) m | algorithm | res az (expected, closed form), rg (m) | PSLR az, rg | '
          'ISLR az, rg, 2-D (dB) | position error az, rg (mm) | error vs BP (dB)')
    t = time.perf_counter()
    for n, ((xt, rt), eb) in enumerate(tg):
        k = n if len(raws) > 1 else 0
        i, j = int(round((xt - x[0]) / dx)), int(round((rt - r[0]) / dr))
        bp = bm.backproject(raws[k], b, x[i - H:i + H, None], r[None, j - H:j + H], rwin=RWIN)
        ra = bm.resolution(b, xt, rt)
        if b.kpsi:
            cf = sm.resolution(b)[1] / b.alpha(rt)
        else:                                                          # uniform illumination over the burst
            cf = 0.886 * b.lam * rt / (2 * b.v * b.na / b.prf)
        for name, im in imgs.items():
            img = im[k][0]
            m = point_target(img, (i, j), dx, dr, half=half)
            ex, er = x[0] + m['i'] * dx - xt, r[0] + m['j'] * dr - rt
            e = rel_db(img[i - H:i + H, j - H:j + H], bp)
            out[n, name] = dict(m, ex=ex, er=er, err=e, ra=ra, rr=rr)
            print(f'  ({xt:7.2f}, {rt - b.r0:6.2f}) {name:12s} {m["res_az"]:.3f} ({ra:.3f}, {cf:.3f}) {m["res_rg"]:.3f} | '
                  f'{m["pslr_az"]:6.1f} {m["pslr_rg"]:6.1f} | {m["islr_az"]:6.1f} {m["islr_rg"]:6.1f} '
                  f'{m["islr_2d"]:6.1f} | {1e3 * ex:+5.2f} {1e3 * er:+5.2f} | {e:6.1f}')
    print(f'  {label}: backprojection of {len(tg)} patches {time.perf_counter() - t:.1f} s')
    return out, imgs


def verify(out, ra_tol=0.02):
    for (n, name), m in out.items():
        assert m['err'] < -40, (n, name, m['err'])
        assert abs(m['ex']) < 5e-3 and abs(m['er']) < 5e-3, (n, name, m['ex'], m['er'])
        assert abs(m['res_az'] / m['ra'] - 1) < ra_tol, (n, name, m['res_az'], m['ra'])
        assert abs(m['res_rg'] / m['rr'] - 1) < 0.03, (n, name, m['res_rg'], m['rr'])


# ------------------------------------------------------------------ TOPS
b = bm.make_bursts('tops')[0]
T = (b.na - 1) / b.prf
dwell = 2 * b.lam * b.r0 / b.La / (b.v + b.r0 * b.kpsi)                    # main lobe, |u| <= 1
print(f'TOPS: {b.na} pulses ({T:.3f} s) x {b.nr} samples, kpsi {b.kpsi:.3f} rad/s, steering '
      f'{np.rad2deg(b.psi(b.eta[0])):+.2f} to {np.rad2deg(b.psi(b.eta[-1])):+.2f} deg, centroid rate {b.kt:.0f} Hz/s, '
      f'f_dc {b.fdc_t(b.eta[0]):+.0f} to {b.fdc_t(b.eta[-1]):+.0f} Hz (span with the beam '
      f'{np.ptp(b.fdc_t(b.eta[[0, -1]])) + 4 * b.v / b.La:.0f} Hz = {(np.ptp(b.fdc_t(b.eta[[0, -1]])) + 4 * b.v / b.La) / b.prf:.2f} PRF), '
      f'alpha {b.alpha(b.r0):.3f}, target dwell {dwell:.3f} s, upsampling {bm.upsampling(b)}')
etas = (b.eta[0] + dwell / 2, 0.013, b.eta[-1] - dwell / 2)                # beginning, middle, end of the burst
tg = [((x_at(b, e, rt) + dxo, rt), e) for rt in (b.r0 - 61.3, b.r0 + 47.9) for e, dxo in zip(etas, (0.11, 0.37, -0.23))]
t = time.perf_counter()
raw = bm.simulate(b, [p for p, _ in tg])
print(f'simulated in {time.perf_counter() - t:.2f} s; sign checks:')
signs(b, tg[:3])
out, imgs = check(b, [raw], tg, (('omega-k', 'omegak', {}), ('omega-k f64', 'omegak', dict(dtype='float64')),
                               ('RDA+SRC', 'rda', {})), 24, 'TOPS')
e64 = rel_db(imgs['omega-k'][0][0], imgs['omega-k f64'][0][0])
print(f'TOPS omega-k float32 against float64, whole image: {e64:.1f} dB')
assert e64 < -70, e64
verify(out)

# the chain depends on the sign of the steering: focusing with -kpsi leaves no point target
bad = bm.focus_burst(raw, replace(b, kpsi=-b.kpsi), rwin=RWIN)[0]
i, j = (int(round(v)) for v in ((tg[1][0][0] - imgs['omega-k'][0][2][0]) / (b.v / b.prf), (tg[1][0][1] - imgs['omega-k'][0][1][0]) / (sm.C / 2 / b.fs)))
pk = np.abs(imgs['omega-k'][0][0][i - 3:i + 4, j - 3:j + 4]).max()
pb = np.abs(bad[i - 3:i + 4, j - 3:j + 4]).max()
print(f'TOPS focused with the steering reversed: peak {20 * np.log10(pb / pk):.1f} dB')
assert pb < 0.3 * pk

# ------------------------------------------------------------------ ScanSAR, two subswaths
bs = bm.make_bursts('scansar', nburst=4, subswaths=[(5e3, 200.0), (5.25e3, 200.0)])
b, b2 = bs[0], bs[2]
dwell = 2 * b.lam * b.r0 / b.La / b.v
print(f'\nScanSAR: subswaths at {bs[0].r0:.0f} and {bs[1].r0:.0f} m, bursts of {b.na} pulses '
      f'({b.na / b.prf:.3f} s), burst cycle {(bs[2].eta0 - bs[0].eta0):.3f} s per subswath, stripmap dwell '
      f'{dwell:.2f} s, fast-time windows {bs[0].nr} and {bs[1].nr} samples')
etas = (b.eta[0] - 0.25, b.eta[0], 0.013, b.eta[-1], b.eta[-1] + 0.25)     # beam-center crossings
tg = [((x_at(b, e, rt) + dxo, rt), e) for rt in (b.r0 - 61.3, b.r0 + 47.9)
      for e, dxo in zip(etas, (0.11, -0.07, 0.37, 0.19, -0.23))]
tg2 = [((x_at(bs[1], bs[1].eta_mid, 5.25e3 + 20.2) + 0.07, 5.25e3 + 20.2), bs[1].eta_mid)]
for w, q in zip([bm.simulate(q, [p for p, _ in tg + tg2]) for q in bs], bs):
    print(f'  burst at {q.eta0:+.3f} s, subswath at {q.r0:.0f} m: energy {np.sum(np.abs(w) ** 2):.4g}')
raws = [bm.simulate(b, [p]) for p, _ in tg]                              # one target per image: unweighted burst
algs = (('omega-k', 'omegak', {}), ('RDA+SRC', 'rda', {}))               # images have far sidelobes
out, imgs = check(b, raws, tg, algs, 88, 'ScanSAR burst 1')
verify(out)
out2, _ = check(bs[1], [bm.simulate(bs[1], [p for p, _ in tg2])], tg2, algs, 88, 'ScanSAR subswath 2')
verify(out2)

print('mosaic of subswath 1 from bursts 1 and 3, one target at a time:')
for n, ((xt, rt), eb) in enumerate(tg):
    im2 = bm.focus_burst(bm.simulate(b2, [(xt, rt)]), b2, rwin=RWIN)
    mos, r, x, sel = bm.mosaic([imgs['omega-k'][n], im2], [b, b2])
    dx, dr = x[1] - x[0], r[1] - r[0]
    i, j = int(round((xt - x[0]) / dx)), int(round((rt - r[0]) / dr))
    m = point_target(mos, (i, j), dx, dr, half=88)
    e = [np.sum(bm.coverage(q, xt, rt)[0] ** 2) for q in (b, b2)]
    ex, er = x[0] + m['i'] * dx - xt, r[0] + m['j'] * dr - rt
    print(f'  ({xt:7.2f}, {rt - b.r0:6.2f}) from burst {2 * sel[i] + 1} (illumination energy {e[0]:6.1f}, {e[1]:6.1f}): '
          f'res az {m["res_az"]:.3f} m (expected {bm.resolution([b, b2][sel[i]], xt, rt):.3f}), '
          f'position error {1e3 * ex:+.2f} {1e3 * er:+.2f} mm')
    assert sel[i] == int(np.argmax(e)) and abs(ex) < 5e-3 and abs(er) < 5e-3
print('ok')
