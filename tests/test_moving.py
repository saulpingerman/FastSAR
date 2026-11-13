"""process_cphd on moving-beam collections (stripmap, sliding spotlight): the weightings, autofocus along the track
and the quality areas in range, on the simulated collections of moving_sim.py read through its stand-in for the
CPHD reader.

(a) Weighting. The impulse responses of point targets against theory, in units of the reciprocal bands: Taylor
    (nbar 4, 35 dB) 1.184 wide with a -35 dB peak sidelobe, uniform 0.886 and -13.26 dB, on a broadside stripmap, a
    squinted one and a sliding spotlight; the product at baseband on both axes. Apodization: no sidelobe above
    -30 dB of its target around the main lobe, which keeps the unweighted width; refused where the squint shears
    the spectrum. The defaults, which are a spotlight's: an unweighted SICD and an apodized SIDD, the SIDD Taylor
    weighted with a note where apodization is ruled out and was not asked for by name.
(b) Autofocus. A quadratic phase error over slow time injected into the phase history (2, 5 and 15 rad at the edge
    of a synthetic aperture), an error that changes along the track, and targets above the focus plane seen from a
    curved track (one height, and a height that changes across the swath): the impulse response returns to the
    error-free one. A focused collection comes back unchanged. These are measured on Taylor-weighted images, whose
    main lobe the product's own sampling resolves; the unweighted default is tested for the same recovery.
(c) Areas. With an antenna block in the file, the area in ground range against the two-way gain computed here
    from the beam's formula, the edge of 'black' at the sigma zero asked for (black_sigma0_db), the receive window
    and its extension, a window that holds nothing beyond the tasked swath, a file without an antenna pattern, and
    the check of the antenna block against the beam's footprint in the phase history: the pointing it chooses for
    a beam whose reference point moves, and the refusal of the areas when no pattern is trusted.
(d) The backends give the same products, and under a window with a step at the band's edge (the rectangle, a
    Taylor window's pedestal) JAX's former cuts each pixel's band at the pulses the cpu former cuts it at. On JAX's
    device, where everything is float32: the baseband phase of an orbit at 600 km over tens of kilometers of ground
    against the phase in double, and the stages with a device budget of a fifth of a megabyte (blocks of a few rows
    or columns, an image kept on the host) against the unblocked ones, with no device array larger than the budget.

Limits are about twice the departures measured when they were set."""
import gc

import numpy as np
import pytest

import fastsar
from fastsar import io, moving, quality

import moving_sim as ms

C = ms.C
TG = np.array([[-12.0, -15.3, 0.0], [3.7, 0.4, 0.0], [10.2, 16.1, 0.0]])
EXT = (60.0, 50.0)


@pytest.fixture(scope='module', autouse=True)
def stand_in_reader():
    """io._open_cphd replaced by the simulator's reader while this module runs."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(io, '_open_cphd', ms.Reader)
        yield
    ms.FILES.clear()


def responses(o, c, half=16, search=0):
    """Per target: azimuth and range width over the reciprocal bands, the two peak sidelobe ratios (dB), the peak
    amplitude over the target's own, and the offset of the peak from the target (pixels). search: rows on either
    side of the target's place in which its peak is looked for first (a phase error over slow time that grows
    along the strip also moves the targets along it)."""
    r = []
    for (i, j), a in zip(ms.to_pixels(o, o['meta'], c.tg), c.amp):
        if search:
            w = np.abs(o['image'][round(i) - search:round(i) + search + 1, round(j) - 2:round(j) + 3])
            i = round(i) - search + np.unravel_index(np.argmax(w), w.shape)[0]
        m = quality.point_target(o['image'], (round(i), round(j)), o['spx'], o['spy'], half=half)
        r.append((m['res_az'] / c.ares, m['res_rg'] / (c.rres / np.cos(c.graze)), m['pslr_az'], m['pslr_rg'],
                  m['peak'] / abs(a), np.hypot(m['i'] - i, m['j'] - j)))
    return np.array(r)


def sidelobes(o, c, box=3, reach=24):
    """Per target: the largest pixel within `reach` pixels of it but outside the (2 box + 1)^2 pixels around its
    peak, over the peak (dB)."""
    a, r = np.abs(o['image']), []
    for i, j in np.round(ms.to_pixels(o, o['meta'], c.tg)).astype(int):
        w = a[i - reach:i + reach + 1, j - reach:j + reach + 1].copy()
        pi, pj = np.unravel_index(np.argmax(w), w.shape)
        pk = w[pi, pj]
        w[max(pi - box, 0):pi + box + 1, max(pj - box, 0):pj + box + 1] = 0
        r.append(20 * np.log10(w.max() / pk))
    return np.array(r)


def centroids(o):
    """The centroid of the image's spectrum along each axis over the product's bandwidth along it."""
    F = np.abs(np.fft.fft2(o['image'])) ** 2
    out = []
    for ax, sp in ((0, o['spx']), (1, o['spy'])):
        p = F.sum(1 - ax)
        out.append((p * np.fft.fftfreq(len(p), sp)).sum() / p.sum() / o['bandwidth'][ax])
    return np.abs(out)


# ---------------------------------------------------------------------------------------------- (a) weighting
CASES = dict(broadside={}, squinted=dict(squint=np.radians(3.0), P=1800), sliding=dict(slide=2.0, P=2400))


@pytest.fixture(scope='module', params=list(CASES))
def strip(request):
    return request.param, ms.collection(request.param + '.cphd', TG, np.array([1.0, 0.8j, -0.6]), **CASES[request.param])


def test_weighting_impulse_responses(strip):
    name, c = strip
    fine = (0.5 * c.ares, 0.5 * c.rres / np.cos(c.graze))            # twice Nyquist: the unweighted response is measurable
    for wt, width, pslr in (('taylor', 1.184, -35.0), ('uniform', 0.886, -13.26)):
        o = fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, weighting=wt, spacing=fine, autofocus=False)
        r = responses(o, c)
        print(f'  {name} {wt}: azimuth {r[:, 0].round(3)} {r[:, 2].round(2)} dB, range {r[:, 1].round(3)} {r[:, 3].round(2)} dB')
        assert o['mode'] == 'moving' and o['plane'] == 'ground' and o['window'] == (wt, wt) and o['baseband']
        assert np.abs(r[:, :2] / width - 1).max() < 0.015, (wt, r[:, :2])
        assert np.abs(r[:, 2:4] - pslr).max() < (1.5 if wt == 'taylor' else 0.3), (wt, r[:, 2:4])
        assert r[:, 5].max() < 0.1 and np.ptp(r[:, 4]) < 0.03 * r[:, 4].mean()
        assert centroids(o).max() < 0.02                             # at baseband on both axes, whatever the squint


def test_default_product(strip):
    """The defaults are a spotlight's. A SICD: the ground plane, no weighting (a rectangle over each pixel's
    processed azimuth band and no frequency window) at form_cphd's own spacing (finer where a sliding beam widens
    the band), quadratic autofocus, the tasked area, no Radiometric block. A SIDD: spatially variant apodization,
    quadratic autofocus, the NRL remap, shadows down. Each is the product of the same options spelled out, and
    weighting='taylor' is the Taylor window on both axes."""
    name, c = strip
    run = lambda **kw: fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, **kw)
    o = run()
    assert (o['product'], o['plane'], o['weighting'], o['area']) == ('sicd', 'ground', 'uniform', 'tasked')
    assert o['mode'] == 'moving' and o['autofocus_info']['mode'] == 'quadratic' and 'autofocus' in o['timing']
    assert o['radiometric'] is None and sum('No Radiometric block' in n and 'moving beam' in n for n in o['notes']) == 1
    assert not any(n.startswith('weighting:') for n in o['notes'])
    same = run(plane='ground', weighting='uniform', autofocus='quadratic', area='tasked')
    assert np.array_equal(same['image'], o['image'])
    plain = run(autofocus=False)
    for wt, p in (('uniform', plain), ('taylor', run(weighting='taylor'))):
        ref = fastsar.form_cphd(name + '.cphd', backend='cpu', extent=EXT, azimuth_window=wt, window=wt == 'taylor',
                                spacing=(p['spx'], p['spy']),
                                _plan=lambda g: dict(fit_patch=True))    # process_cphd's patch, for the same rounding
        assert p['weighting'] == wt and p['window'] == (wt, wt) and p['autofocus'] is None and p['baseband']
        assert p['image'].shape == ref['image'].shape == o['image'].shape
        assert np.abs(np.abs(p['image']) - np.abs(ref['image'])).max() < 2e-3 * np.abs(ref['image']).max()
        # the band of the image along the track: the processed Doppler band, widened by a slide or a squint
        assert any('azimuth band of the image' in n for n in p['notes']) == (name != 'broadside')
        assert 0.9 < p['spx'] / (0.96 * c.ares) < (0.98 if name != 'broadside' else 1.02)
        if wt == 'taylor':                 # a focused collection comes back as it came, and its main lobe is Taylor's
            assert p['autofocus_info']['applied'] is False
            r = responses(p, c, half=12)
            assert np.abs(r[:, 0] / 1.184 - 1).max() < 0.03, r[:, 0]
    # the unweighted default with autofocus: the focused collection stays focused (the estimate of a band that ends
    # in a step, at 0.96 of the sampling rate, is a few hundredths of a radian, and its correction changes little)
    info = o['autofocus_info']
    assert np.abs(info['estimates'][np.isfinite(info['estimates'])]).max() < 0.1, info['estimates']
    assert np.abs(o['image'] - plain['image']).max() < 0.03 * np.abs(plain['image']).max()


def test_default_sidd(strip):
    """The default SIDD of a moving beam is apodized, as a spotlight's is. Where apodization is ruled out (a squint
    of 3 degrees, which shears the range band by more than a tenth of its width; a spacing given, where it sets
    its own) and was not asked for by name, the image is Taylor weighted and a note says why; weighting='sva'
    raises there (test_apodization, test_errors)."""
    name, c = strip
    run = lambda **kw: fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, product='sidd', **kw)
    s = run()
    said = [n for n in s['notes'] if n.startswith('weighting:')]
    assert (s['product'], s['plane'], s['area'], s['remap'], s['shadows_down']) == ('sidd', 'ground', 'tasked', 'nrl', True)
    assert s['autofocus_info']['mode'] == 'quadratic'
    if name == 'squinted':
        assert s['weighting'] == 'taylor' and s['window'] == ('taylor', 'taylor') and len(said) == 1, s['notes']
        assert 'shears the range band' in said[0] and "'sva'" in said[0] and "weighting='taylor'" in said[0]
        assert not any('apodization: formed' in n for n in s['notes'])
    else:
        assert s['weighting'] == 'sva' and s['window'] == ('sva', 'sva') and not said, s['notes']
        assert any('apodization: formed' in n for n in s['notes'])
    same = run(weighting=s['weighting'])                             # the same product, asked for by name
    assert np.array_equal(same['image'], s['image']) and same['spx'] == s['spx']
    assert [n for n in s['notes'] if n not in said] == same['notes']
    from fastsar import sidd
    pix = sidd.to_display(s)
    assert pix.dtype == np.uint8 and pix.max() > 200
    # a spacing: apodization sets its own, so the default gives way to Taylor at the spacing asked for
    g = run(spacing=0.5)
    said = [n for n in g['notes'] if n.startswith('weighting:')]
    assert g['weighting'] == 'taylor' and g['spx'] == g['spy'] == 0.5 and len(said) == 1 and 'a spacing was given' in said[0]


def test_apodization(strip):
    name, c = strip
    if name == 'squinted':                                           # 3 degrees: beyond the limit, with the reason
        with pytest.raises(ValueError, match='shears the range band'):
            fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, weighting='sva')
        return
    o = fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, weighting='sva', autofocus=False)
    u = fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, weighting='uniform', autofocus=False,
                             spacing=(o['spx'], o['spy']))
    # exactly twice Nyquist of the processed azimuth band and of the ground range band, rows a multiple of four
    assert abs(o['spx'] * 2 * (2 * c.dsin / c.lam) * CASES[name].get('slide', 1.0) - 1) < 0.01
    assert abs(o['spy'] * 2 * (2 * c.K * c.df / C * np.cos(c.graze)) - 1) < 0.005
    assert o['window'] == ('sva', 'sva') and any('apodization' in n for n in o['notes'])
    sl, su = sidelobes(o, c), sidelobes(u, c)
    r, ru = responses(o, c, half=32), responses(u, c, half=32)
    print(f'  {name}: sidelobes {sl.round(1)} dB (unweighted {su.round(1)}), width {r[:, :2].round(3)}')
    assert sl.max() < -30 and su.min() > -20, (sl, su)
    assert np.abs(r[:, :2] / 0.886 - 1).max() < 0.08, r[:, :2]       # the unweighted main lobe
    assert np.abs(r[:, 4] / ru[:, 4] - 1).max() < 0.05 and r[:, 5].max() < 0.15


def test_small_squint_is_apodized():
    c = ms.collection('squint05.cphd', TG, squint=np.radians(0.5), P=1500)
    o = fastsar.process_cphd('squint05.cphd', backend='cpu', extent=EXT, weighting='sva', autofocus=False)
    sl = sidelobes(o, c)
    print(f'  0.5 degree squint: sidelobes {sl.round(1)} dB')
    assert sl.max() < -30, sl


def test_errors(strip):
    name, c = strip
    for kw, match in ((dict(plane='slant'), 'moving beam'), (dict(weighting='sva', product='sidd', spacing=0.5), 'spacing'),
                      (dict(autofocus='pga'), 'phase gradient'), (dict(autofocus='spatially_variant_pga'), 'phase gradient'),
                      (dict(area='black', nesz_db=-10.0), 'antenna pattern')):
        with pytest.raises(ValueError, match=match):
            fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT, **kw)


# ---------------------------------------------------------------------------------------------- (b) autofocus
LONG = (240.0, 50.0)
PL = 2600
TAYLOR = dict(weighting='taylor')  # the weighting these are measured with: the product's sampling resolves its lobe


def scene():
    """36 targets over 210 x 26 m, one every 6 m along the track at a ground range drawn over the swath: each alone
    in the window its response is measured in."""
    rng = np.random.default_rng(3)
    tg = np.stack([rng.uniform(-13, 13, 36), np.linspace(-105, 105, 36) + rng.uniform(-0.3, 0.3, 36), np.zeros(36)], 1)
    return tg, rng.uniform(0.5, 1.0, len(tg)) * np.exp(2j * np.pi * rng.uniform(size=len(tg)))


@pytest.fixture(scope='module')
def focused():
    tg, amp = scene()
    c = ms.collection('long.cphd', tg, amp, P=PL)
    ref = fastsar.process_cphd('long.cphd', backend='cpu', extent=LONG, autofocus=False, **TAYLOR)
    return c, ref, responses(ref, c, half=12)


def test_autofocus_leaves_a_focused_collection(focused):
    c, ref, r0 = focused
    for mode in ('quadratic', 'spatially_variant'):
        o = fastsar.process_cphd('long.cphd', backend='cpu', extent=LONG, autofocus=mode, **TAYLOR)
        info = o['autofocus_info']
        assert o['autofocus'] is None and not info['applied'] and info['mode'] == mode and 'autofocus' in o['timing']
        assert len(info['starts']) == 3 and info['counted'].all() and np.abs(info['estimates']).max() < 0.05
        assert info['reason'] == 'the fitted phase is below tol'
        assert np.array_equal(o['image'], ref['image'])


@pytest.mark.parametrize('rad', [2.0, 5.0, 15.0])
def test_autofocus_recovers_a_slow_time_error(focused, rad):
    c, ref, r0 = focused
    tg, amp = scene()
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)                  # slow time in half apertures
    ms.collection('quad.cphd', tg, amp, P=PL, phase=rad * s ** 2)
    blurred = responses(fastsar.process_cphd('quad.cphd', backend='cpu', extent=LONG, autofocus=False, **TAYLOR), c, 12, 10)
    o = fastsar.process_cphd('quad.cphd', backend='cpu', extent=LONG, **TAYLOR)
    info, r = o['autofocus_info'], responses(o, c, 12, 10)
    print(f'  {rad} rad: estimates {info["estimates"].round(2)}, fit {info["fit"].round(2)}, width '
          f'{np.median(blurred[:, 0]):.3f} -> {np.median(r[:, 0]):.3f} (focused {np.median(r0[:, 0]):.3f}), sidelobe '
          f'{np.median(blurred[:, 2]):.1f} -> {np.median(r[:, 2]):.1f} dB ({np.median(r0[:, 2]):.1f})')
    assert o['autofocus'] == 'quadratic' and info['applied'] and info['applied_blocks'].all()
    assert np.abs(info['fit'] / rad - 1).max() < 0.05                # rad at the edge of the synthetic aperture
    assert np.median(blurred[:, 0]) > 1.1 * np.median(r0[:, 0])
    assert abs(np.median(r[:, 0]) / np.median(r0[:, 0]) - 1) < 0.015 and np.abs(r[:, 0] / r0[:, 0] - 1).max() < 0.06
    assert abs(np.median(r[:, 2]) - np.median(r0[:, 2])) < 1.5 and np.abs(r[:, 2] - r0[:, 2]).max() < 4.0
    assert np.abs(r[:, 4] / r0[:, 4] - 1).max() < 0.03
    assert info['contrast_after'] > info['contrast_before'] and (o['image'] == 0).sum() == (ref['image'] == 0).sum()


@pytest.mark.parametrize('rad', [2.0, 15.0])
def test_autofocus_recovers_an_error_on_the_default_product(rad):
    """The default SICD is unweighted: the same slow-time error is estimated and removed from it. Its main lobe is
    0.886 of the reciprocal band on pixels of 0.96, so the width of one target is measured to a tenth and the
    median over the strip to a percent."""
    tg, amp = scene()
    c = ms.collection('long.cphd', tg, amp, P=PL)
    r0 = responses(fastsar.process_cphd('long.cphd', backend='cpu', extent=LONG, autofocus=False), c, 12)
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)
    ms.collection('quad_u.cphd', tg, amp, P=PL, phase=rad * s ** 2)
    blurred = responses(fastsar.process_cphd('quad_u.cphd', backend='cpu', extent=LONG, autofocus=False), c, 12, 10)
    o = fastsar.process_cphd('quad_u.cphd', backend='cpu', extent=LONG)
    info, r = o['autofocus_info'], responses(o, c, 12, 10)
    print(f'  unweighted, {rad} rad: fit {info["fit"].round(2)}, width {np.median(blurred[:, 0]):.3f} -> '
          f'{np.median(r[:, 0]):.3f} (focused {np.median(r0[:, 0]):.3f}), sidelobe {np.median(r[:, 2]):.1f} dB '
          f'({np.median(r0[:, 2]):.1f})')
    assert o['weighting'] == 'uniform' and o['autofocus'] == 'quadratic' and info['applied'] and info['applied_blocks'].all()
    assert np.abs(info['fit'] / rad - 1).max() < 0.05
    assert np.median(blurred[:, 0]) > 1.08 * np.median(r0[:, 0])
    assert abs(np.median(r[:, 0]) / np.median(r0[:, 0]) - 1) < 0.015 and np.abs(r[:, 0] / r0[:, 0] - 1).max() < 0.2
    # the peak sample of a lobe narrower than a pixel changes with the target's place between the pixels, which a
    # slow-time error that grows along the strip moves: by a tenth for one target, by a percent in the median
    assert abs(np.median(r[:, 2]) - np.median(r0[:, 2])) < 1.0
    assert abs(np.median(r[:, 4] / r0[:, 4]) - 1) < 0.02 and np.abs(r[:, 4] / r0[:, 4] - 1).max() < 0.2


def test_autofocus_follows_an_error_along_the_track(focused):
    c, ref, r0 = focused
    tg, amp = scene()
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)
    ms.collection('vary.cphd', tg, amp, P=PL, phase=4.0 * s ** 2 + 0.3 * s ** 3)     # 4 rad +- 0.9 along the strip
    o = fastsar.process_cphd('vary.cphd', backend='cpu', extent=LONG, **TAYLOR)
    info, r = o['autofocus_info'], responses(o, c, 12, 10)
    print(f'  along track: estimates {info["estimates"].round(2)}, fit {info["fit"].round(2)}, width {r[:, 0].max():.3f}')
    assert info['applied'] and info['fit'][0] < 3.7 and info['fit'][-1] > 4.3 and abs(info['fit'][1] - 4.0) < 0.2
    assert np.abs(r[:, 0] / r0[:, 0] - 1).max() < 0.06 and np.abs(r[:, 4] / r0[:, 4] - 1).max() < 0.05


@pytest.mark.parametrize('heights', ['level', 'tilted'])
def test_autofocus_recovers_a_height_offset(heights):
    """Targets above the focus plane, seen from a track that curves across the line of sight: a quadratic error in
    proportion to the height. 'level': all 40 m up. 'tilted': the height grows across the swath, from -6 to 46 m."""
    tg, amp = scene()
    kw = dict(P=PL, curve=1.7e-4)
    c = ms.collection('curved.cphd', tg, amp, **kw)
    r0 = responses(fastsar.process_cphd('curved.cphd', backend='cpu', extent=LONG, autofocus=False, **TAYLOR), c, 12)
    h = np.full(len(tg), 40.0) if heights == 'level' else 20.0 + 2.0 * tg[:, 0]
    up = tg + np.stack([h * np.tan(c.graze), 0 * h, h], 1)           # the same image points, on the ground plane
    ms.collection('raised.cphd', up, amp, **kw)
    # the layover of a raised target follows the local direction across the curved track: a few rows along it
    blurred = responses(fastsar.process_cphd('raised.cphd', backend='cpu', extent=LONG, autofocus=False, **TAYLOR), c, 12, 4)
    o = fastsar.process_cphd('raised.cphd', backend='cpu', extent=LONG, **TAYLOR)
    info, r = o['autofocus_info'], responses(o, c, 12, 4)
    print(f'  {heights}: fit {info["fit"].round(2)}, slope {info["slope"].round(2)}, width {np.median(blurred[:, 0]):.3f} '
          f'(max {blurred[:, 0].max():.3f}) -> {np.median(r[:, 0]):.3f} (max {r[:, 0].max():.3f}), sidelobe '
          f'{np.median(r[:, 2]):.1f} dB ({np.median(r0[:, 2]):.1f})')
    assert info['applied'] and blurred[:, 0].max() > 2 * r0[:, 0].max()
    assert (np.abs(info['slope']) > 3).all() == (heights == 'tilted')
    assert np.abs(r[:, 0] / r0[:, 0] - 1).max() < 0.06 and np.abs(np.median(r[:, 2]) - np.median(r0[:, 2])) < 1.5


def test_products_on_disk(tmp_path):
    """The SICD of a moving beam states the ground plane, a band centered at zero with the widths the product
    measured from the collection, the weighting, and autofocus that changes over the scene; the apodized SIDD is
    written from the device or the host."""
    from fastsar import _sicdxml
    tg, amp = scene()
    c = ms.collection('disk.cphd', tg, amp, P=PL, phase=5.0 * ((np.arange(PL) - PL / 2) / 500.0) ** 2)
    for wt, kw in (('uniform', {}), ('taylor', TAYLOR)):                # the default, and the Taylor window
        o = fastsar.process_cphd('disk.cphd', str(tmp_path / 'm.nitf'), backend='cpu', extent=LONG, **kw)
        sm = _sicdxml.read(str(tmp_path / 'm.nitf'))
        assert o['path'] == str(tmp_path / 'm.nitf') and o['autofocus'] == 'quadratic' and o['weighting'] == wt
        assert sm.Grid.ImagePlane == 'GROUND' and sm.Grid.Type == 'PLANE' and sm.ImageFormation.AzAutofocus == 'SV'
        assert sm.Grid.Row.WgtType.WindowName == wt.upper() and sm.Grid.Col.WgtType.WindowName == wt.upper()
        assert sm.Grid.Row.DeltaKCOAPoly[0, 0] == 0 and sm.Grid.Col.DeltaKCOAPoly[0, 0] == 0
        assert abs(sm.Grid.Col.ImpRespBW / (2 * c.dsin / c.lam) - 1) < 0.01          # columns along the track
        assert abs(sm.Grid.Row.ImpRespBW / (2 * c.K * c.df / C * np.cos(c.graze)) - 1) < 0.01
        assert (int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)) == o['image'].shape[::-1]
        assert sm.Radiometric is None and o['radiometric'] is None  # the radiometric model is a spotlight's
    for kw in (dict(), dict(weighting='sva')):                       # the default SIDD is the apodized one
        d = fastsar.process_cphd('disk.cphd', str(tmp_path / 'd.nitf'), backend='cpu', extent=LONG, product='sidd', **kw)
        assert d['weighting'] == 'sva' and d['autofocus'] == 'quadratic'
        assert (tmp_path / 'd.nitf').stat().st_size > d['image'].size


# ---------------------------------------------------------------------------------------------- (c) areas
ANT = (9600.0, 1.0e5)              # one-way dB per squared direction cosine along and across the track: 6 dB down
#                                    at the edge of the processed band, so that the azimuth ambiguities stay small
FAR = np.array([[-12.0, -15.3, 0.0], [3.7, 0.4, 0.0], [10.2, 16.1, 0.0], [45.0, 5.0, 0.0], [-50.0, -8.0, 0.0]])


def gain_db(c, x):
    """The two-way gain (dB, relative to the beam center's) of the ground points (x, 0, 0) from the beam's formula:
    the square of the amplitude gain averaged over the pulses of the processed band."""
    out = []
    zb = ms.unit(c.srp - c.ant)
    xa = ms.unit(c.d - np.einsum('ij,ij->i', c.d, zb)[:, None] * zb)
    for v in np.atleast_1d(x):
        w = np.array([v, 0.0, 0.0])[None] - c.ant
        R = np.linalg.norm(w, axis=1)
        on = np.abs(np.einsum('ij,ij->i', w, c.d) / R - c.s_srp) <= c.dsin / 2
        dcx, dcy = np.einsum('ij,ij->i', w, xa) / R, np.einsum('ij,ij->i', w, c.ya) / R
        out.append(20 * np.log10((10.0 ** (-0.1 * (ANT[0] * dcx ** 2 + ANT[1] * dcy ** 2)) * (c.ref / R) ** 2)[on].mean()))
    return np.array(out)


def swath(o):
    """The ground range (simulator x, m) of the first and last nonzero pixel of three rows of the product."""
    nz, out = o['image'] != 0, []
    for i in (2, nz.shape[0] // 2, nz.shape[0] - 3):
        j = np.flatnonzero(nz[i])[[0, -1]]
        p = io.local_to_ecf(o['origin'] + i * o['spx'] * o['e1'] + j[:, None] * o['spy'] * o['e2'], o['meta']) - ms.SRP0
        out.append(np.sort(p @ ms.N))
    return np.array(out)


def antenna_strip(name, toa_m, toae_m, **kw):
    """A stripmap with an antenna block and receive windows of +-toa_m (extended: +-toae_m) meters of ground range."""
    k = 2 * np.cos(np.radians(30.0)) / C
    return ms.collection(name, FAR, antenna=ANT, toa=k * toa_m, toae=None if toae_m is None else k * toae_m,
                         df=0.75e6, **kw)


def test_areas_follow_the_gain_in_range():
    """A wide receive window: 'black' ends where terrain of black_sigma0_db (-10 dB) has a signal-to-noise ratio of
    1, that is, where the two-way gain equals the noise-equivalent sigma zero over that sigma zero; an RNIIRS
    where the gain gives the ratio its reference scene of 0 dB needs, whatever black_sigma0_db is; and
    'tasked_quality' where the gain falls below its least value in the tasked swath; each to one step of the
    model's lattice across the track (3 m, 0.8 dB of this beam)."""
    from fastsar import rgiqe
    c = antenna_strip('ant.cphd', 95.0, 120.0)
    t = fastsar.process_cphd('ant.cphd', backend='cpu', extent=EXT, autofocus=False)
    bw = t['bandwidth'][0] * t['bandwidth'][1]
    rniirs = -10.0 + 10 * np.log10(rgiqe.rniirs_to_snr(bw, 4.8))
    width = {}
    for area, kw, level in (('black', dict(nesz_db=-20.0), -10.0),                       # the default: -10 dB
                            ('black', dict(nesz_db=-20.0, black_sigma0_db=-12.0), -8.0),
                            ('black', dict(nesz_db=-10.0, black_sigma0_db=0.0), -10.0),  # the RGIQE's reference scene
                            (4.8, dict(nesz_db=-10.0), rniirs), (4.8, dict(nesz_db=-10.0, black_sigma0_db=-30.0), rniirs),
                            ('tasked_quality', dict(nesz_db=-10.0), gain_db(c, [-25.0, 25.0]).min())):
        o = fastsar.process_cphd('ant.cphd', backend='cpu', extent=EXT, area=area, autofocus=False, **kw)
        sw = swath(o)
        g = gain_db(c, sw.mean(0))
        print(f'  area {area} {kw}: ground range {sw.mean(0).round(1)} m, gain there {g.round(2)} dB (criterion {level:.2f} dB)')
        assert o['area'] == area and o['image'].shape[0] == t['image'].shape[0] and np.ptp(sw, 0).max() < 2 * o['spy']
        assert sw[:, 0].max() <= -25 + o['spy'] and sw[:, 1].min() >= 25 - o['spy']          # never short of the tasked swath
        assert np.abs(g - level).max() < 1.0, (area, sw, g, level)
        # a few point targets show no footprint to check the antenna block against: the pattern as the file has it
        assert sum('could not be checked' in n and 'unverified' in n for n in o['notes']) == 1, o['notes']
        if area == 'tasked_quality':                                 # the tasked swath is where its own quality ends
            assert o['image'].shape == t['image'].shape and any('No ground range beyond' in n for n in o['notes'])
            continue
        assert (o['image'] == 0).any() and any('ground range with' in n for n in o['notes'])
        if area == 'black':
            assert sum(f"sigma zero {kw.get('black_sigma0_db', -10.0):.1f} dB" in n for n in o['notes']) == 1
        width[area, tuple(kw.values())] = np.ptp(sw.mean(0))
        j = ms.to_pixels(o, o['meta'], c.tg)
        inside = (c.tg[:, 0] > sw[1, 0] + 2) & (c.tg[:, 0] < sw[1, 1] - 2)
        assert all(abs(o['image'][round(a), round(b)]) > 0 for (a, b), ok in zip(j, inside) if ok)
    # a brighter reference ends sooner; the same ratio of noise to reference is the same area; an RNIIRS ignores it
    assert width['black', (-20.0, -12.0)] < width['black', (-20.0,)] - 3
    assert abs(width['black', (-10.0, 0.0)] - width['black', (-20.0,)]) < 1e-6
    assert abs(width[4.8, (-10.0, -30.0)] - width[4.8, (-10.0,)]) < 1e-6
    # at the default, a noise level of -10 dB leaves nothing beyond the tasked swath, where the gain is 4.6 dB down
    o = fastsar.process_cphd('ant.cphd', backend='cpu', extent=EXT, area='black', nesz_db=-10.0, autofocus=False)
    assert o['image'].shape == t['image'].shape and sum('No ground range beyond the tasked swath' in n for n in o['notes']) == 1


def pointed(monkeypatch, pointing):
    """rgiqe.Radiometry's check of the antenna block replaced by one that finds the given pointing, as the data of
    a scene with clutter would decide it; the frames are turned onto the reference points for 'reference point' as
    the check does. -> the list of the models that ran the check."""
    from fastsar import rgiqe
    ran = []

    def check(self, fp):
        ran.append(self)
        self.footprint = dict(status='consistent' if pointing else 'inconsistent')
        self.pointing = pointing
        if pointing == 'reference point':
            self._repoint()
    monkeypatch.setattr(rgiqe.Radiometry, '_check_footprint', check)
    return ran


def test_areas_follow_the_pointing_the_data_chose(monkeypatch):
    """The antenna block's pointing is checked once, on the pulses of the middle of the strip, and the models of
    the other positions take it over. 'reference point': each boresight on its pulse's own scene reference point,
    which is where this simulator's beam is, so the area of a file whose frames are turned 8 m off in ground range
    is the one of the frames as simulated; with the frames as that file states them it lies 8 m to the side. None
    (the data contradict the pattern either way): every area but the tasked one is refused, with the reason of a
    spotlight."""
    from fastsar import rgiqe
    c = antenna_strip('point.cphd', 95.0, 120.0)
    kw = dict(backend='cpu', extent=EXT, area='black', nesz_db=-20.0, autofocus=False)
    ran = pointed(monkeypatch, 'file')
    ref = swath(fastsar.process_cphd('point.cphd', **kw)).mean(0)
    assert len(ran) == 1 and ran[0].moving and ran[0].pulses_used < c.P      # once, on a stretch of the pulses
    # the same echoes in a file whose antenna frames look 8 m farther in ground range than the beam does
    d = ms.FILES['point.cphd']
    turn = 8.0 * np.sin(c.graze) / c.r0                              # the direction cosine across the track
    zb = ms.unit(c.srp - c.ant)
    xa = ms.unit(c.d - np.einsum('ij,ij->i', c.d, zb)[:, None] * zb)
    ya = ms.unit(c.ya - turn * zb)                                   # the boresight xa x ya, turned by `turn`
    rot = np.stack([ms.N, ms.E, ms.U], 1)
    ms.FILES['turned.cphd'] = dict(d, pvp=dict(d['pvp'], TxACY=ya @ rot.T, RcvACY=ya @ rot.T))
    off = swath(fastsar.process_cphd('turned.cphd', **kw)).mean(0)
    print(f'  ground range of the black area: {ref.round(1)} m; frames turned, as the file points them {off.round(1)} m')
    assert np.abs(np.abs(off - ref) - 8.0).max() < 3.5 and np.ptp(off - ref) < 3.5
    del ran[:]
    pointed(monkeypatch, 'reference point')
    o = fastsar.process_cphd('turned.cphd', **kw)
    on = swath(o).mean(0)
    print(f'  with each boresight on its scene reference point {on.round(1)} m')
    assert np.abs(on - ref).max() < 0.5 * o['spy'] + 1e-6
    pointed(monkeypatch, None)
    for area in ('black', 4.8, 'tasked_quality'):
        with pytest.raises(ValueError) as e:
            fastsar.process_cphd('turned.cphd', **dict(kw, area=area))
        assert str(e.value) == rgiqe._untrusted(area) and 'does not describe the beam the data show' in str(e.value)
    t = fastsar.process_cphd('turned.cphd', **dict(kw, area='tasked'))               # the tasked area needs no pattern
    assert t['area'] == 'tasked' and not (t['image'] == 0).all()


def lit_strip(name, finufft, n=24000, seed=11):
    """antenna_strip's collection with clutter in place of its point targets: n scatterers over 220 m of ground
    range and 440 m along the track, seen through the beam the file's antenna block states on both of its axes
    (the boresight of each pulse on that pulse's scene reference point), with a little noise. The echoes are
    referred to each pulse's own reference point, as a moving beam's are."""
    c = antenna_strip(name, 95.0, 120.0)
    rng = np.random.default_rng(seed)
    pos = np.stack([rng.uniform(-110, 110, n), rng.uniform(-220, 220, n), np.zeros(n)], 1)
    amp = np.exp(2j * np.pi * rng.uniform(size=n))
    zb = ms.unit(c.srp - c.ant)
    xa = ms.unit(c.d - np.einsum('ij,ij->i', c.d, zb)[:, None] * zb)
    S = np.empty((c.P, c.K), np.complex128)
    fref = c.f0 + (c.K // 2) * c.df
    for p in range(c.P):
        w = pos - c.ant[p]
        R = np.linalg.norm(w, axis=1)
        gain = 10.0 ** (-0.1 * (ANT[0] * (w @ xa[p] / R) ** 2 + ANT[1] * (w @ c.ya[p] / R) ** 2)) * (c.ref[p] / R) ** 2
        dr = R - c.ref[p]
        S[p] = finufft.nufft1d1(4 * np.pi * c.df / C * dr, amp * gain * np.exp(-4j * np.pi * fref / C * dr), c.K,
                                eps=1e-9, isign=-1, nthreads=1)
    S += 0.02 * np.abs(S).std() * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
    ms.FILES[name] = dict(ms.FILES[name], S=S.astype(np.complex64))
    return c


def test_the_data_decide_the_pointing_of_a_moving_beam(need, monkeypatch):
    """The check of the antenna block against the beam's footprint in the phase history, on a stripmap whose
    scene shows one. The footprint of a moving beam travels with the pulses, so the model takes the antenna's
    frame at the pulses of the measured block and the ground around their reference point: the frames as
    simulated pass with the fitted center on the modeled one. With the frame of the middle of the stretch, as for
    a beam held on one point, the fitted center of the same data lies the distance flown in between away, a
    whole half-power radius here.
    A file whose frames are turned across the track fails as stated and passes with each boresight on its pulse's
    reference point; process_cphd says which pointing its areas used, and gives the area of the frames as
    simulated either way."""
    from fastsar import rgiqe
    finufft = need('finufft')
    c = lit_strip('lit.cphd', finufft)
    col, meta = io.read_cphd('lit.cphd', meta=True)
    lo, hi = 200, 1200                                               # the pulses of a pixel in the middle of the strip
    model = lambda path: rgiqe.Radiometry(path, meta, pulses=(lo, hi), nesz_db=-20.0, history=col['S'][lo:hi])
    rad = model('lit.cphd')
    fp = rad.footprint
    print(f"  moving beam: {fp['status']}, {fp['ratio']:.2f} radii ({fp['offset_m']:.1f} m), correlation {fp['correlation']:.2f}")
    assert rad.moving and rad.pointing == 'file' and fp['status'] == 'consistent' and fp['ratio'] < 0.25, fp
    assert fp['correlation'] > 0.55 and sum('agrees with the antenna block as the file points it' in n for n in rad.notes) == 1
    # the model of a beam held on one point, on the same data
    with monkeypatch.context() as mp:
        mp.setattr(rgiqe, '_MOVING_SRP', np.inf)
        held = model('lit.cphd')
    flown = 145 * c.dy                                               # from the stretch's middle pulse to the block's
    print(f"  as a beam held on one point: {held.footprint['status']}, {held.footprint['ratio']:.2f} radii "
          f"({held.footprint['offset_m']:.1f} m; {flown:.1f} m flown)")
    assert not held.moving and held.footprint['ratio'] > 0.6 and abs(held.footprint['offset_m'] - flown) < 0.35 * flown
    # frames turned 1.2 half-power radii across the track in the file, the beam where it was
    turn = 1.2 * fp['radius']
    zb = ms.unit(c.srp - c.ant)
    xa = ms.unit(c.d - np.einsum('ij,ij->i', c.d, zb)[:, None] * zb)
    ya = ms.unit(c.ya - turn * zb)
    rot = np.stack([ms.N, ms.E, ms.U], 1)
    d = ms.FILES['lit.cphd']
    ms.FILES['lit_turned.cphd'] = dict(d, pvp=dict(d['pvp'], TxACY=ya @ rot.T, RcvACY=ya @ rot.T))
    turned = model('lit_turned.cphd')
    ft = turned.footprint
    print(f"  frames turned: as the file points them {ft['file']['ratio']:.2f} radii, on the reference points "
          f"{ft['reference_point']['ratio']:.2f}")
    assert turned.pointing == 'reference point' and ft['file']['status'] == 'inconsistent'
    assert abs(ft['file']['ratio'] - 1.2) < 0.3 and ft['reference_point']['ratio'] < 0.25
    kw = dict(backend='cpu', extent=EXT, area='black', nesz_db=-20.0, autofocus=False)
    a, b = fastsar.process_cphd('lit.cphd', **kw), fastsar.process_cphd('lit_turned.cphd', **kw)
    assert sum('agrees with the antenna block as the file points it' in n for n in a['notes']) == 1
    assert sum('With each boresight on its scene reference point' in n and 'is used with that pointing' in n
               for n in b['notes']) == 1
    assert a['image'].shape == b['image'].shape and np.array_equal(a['image'] == 0, b['image'] == 0)
    assert np.abs(swath(a).mean(0) - swath(b).mean(0)).max() < 1e-6 and (a['image'] == 0).any()


def test_a_beam_that_turns_is_not_checked():
    """A sliding spotlight steers its beam along the track. The blocks of pulses of the footprint measurement are
    added into one map in the echoes' own coordinates, where such a beam's footprint does not stay in place, so the
    check is not made: the file's pointing is kept, the note says why, and the areas are processed. A stripmap,
    whose beam center keeps its direction from the platform, is checked (the tests above)."""
    from fastsar import rgiqe
    c = antenna_strip('slides.cphd', 95.0, 120.0, slide=2.0, P=2400)
    col, meta = io.read_cphd('slides.cphd', meta=True)
    rad = rgiqe.Radiometry('slides.cphd', meta, pulses=(400, 2000), nesz_db=-20.0, history=col['S'][400:2000])
    fp = rad.footprint
    assert rad.moving and rad.pointing == 'file' and fp['status'] == 'unchecked', fp
    assert 'turns by' in fp['reason'] and 'sliding spotlight' in fp['reason'] and fp['ratio'] is None
    assert sum('could not be checked' in n and 'turns by' in n and 'unverified' in n for n in rad.notes) == 1
    straight = antenna_strip('straight.cphd', 95.0, 120.0)
    _, m2 = io.read_cphd('straight.cphd', meta=True)
    fixed = rgiqe.Radiometry('straight.cphd', m2, pulses=(200, 1200), nesz_db=-20.0)
    assert fixed.moving and 'turns by' not in fixed.footprint['reason']      # point targets: no footprint to see
    o = fastsar.process_cphd('slides.cphd', backend='cpu', extent=EXT, area='black', nesz_db=-20.0, autofocus=False)
    assert sum('could not be checked' in n and 'turns by' in n for n in o['notes']) == 1
    assert (o['image'] == 0).any() and any('ground range with' in n for n in o['notes'])


def test_areas_end_at_the_receive_window():
    c = antenna_strip('win.cphd', 26.0, 45.0)                        # the extension reaches 45 m, the gain farther
    o = fastsar.process_cphd('win.cphd', backend='cpu', extent=EXT, area='black', nesz_db=-20.0, autofocus=False)
    sw = swath(o)
    print(f'  black, extension +-45 m: ground range {sw.mean(0).round(1)} m')
    assert np.abs(np.abs(sw) - 45).max() < 6 and any('TOAE1' in n for n in o['notes'])
    # the full-bandwidth window ends at the tasked swath: nothing beyond it, and a note that says so
    antenna_strip('tight.cphd', 25.0, None)
    t = fastsar.process_cphd('tight.cphd', backend='cpu', extent=EXT, autofocus=False)
    for area in ('black', 4.0, 'tasked_quality'):
        o = fastsar.process_cphd('tight.cphd', backend='cpu', extent=EXT, area=area, autofocus=False,
                                 nesz_db=-20.0 if area == 'black' else -10.0)
        assert sum('holds no ground range beyond the tasked swath' in n for n in o['notes']) == 1, o['notes']
        assert o['image'].shape == t['image'].shape and np.allclose(o['origin'], t['origin'])
        assert np.abs(np.abs(o['image']) - np.abs(t['image'])).max() < 2e-3 * np.abs(t['image']).max()


# ---------------------------------------------------------------------------------------------- (d) backends
@pytest.mark.parametrize('backend', ['cuda', 'jax'])
def test_backends_agree(backend):
    if backend not in fastsar.available_backends() + ['jax']:
        pytest.skip(f'no {backend} backend on this machine')
    tg, amp = scene()
    c = ms.collection('be.cphd', tg, amp, P=PL, phase=5.0 * ((np.arange(PL) - PL / 2) / 500.0) ** 2)
    antenna_strip('be_ant.cphd', 95.0, 120.0)
    for name, kw in (('be.cphd', dict(extent=LONG)), ('be.cphd', dict(extent=LONG, weighting='sva')),
                     ('be.cphd', dict(extent=LONG, weighting='uniform', autofocus='spatially_variant', product='sidd')),
                     ('be_ant.cphd', dict(extent=EXT, area='black', nesz_db=-20.0))):
        if backend == 'jax' and 'product' in kw:                     # each grid compiles its own programs
            continue
        ref = fastsar.process_cphd(name, backend='cpu', **kw)
        o = fastsar.process_cphd(name, backend=backend, **kw)
        assert isinstance(o['image'], np.ndarray) and o['image'].shape == ref['image'].shape
        assert o['autofocus'] == ref['autofocus'] and (name != 'be.cphd' or o['autofocus'] is not None)
        # the formers agree to their accuracy target (-40 dB by default)
        e = np.abs(o['image'] - ref['image']).max() / np.abs(ref['image']).max()
        print(f'  {backend} {kw}: {e:.2e}')
        assert e < 2e-2, (kw, e)
        if 'area' in kw:
            assert np.array_equal(o['image'] == 0, ref['image'] == 0) and (o['image'] == 0).any()


def test_jax_cuts_each_band_where_the_cpu_does(strip):
    """A weighting with a step at the edge of each pixel's azimuth band (the rectangle of the default product, the
    pedestal of a Taylor window) enters the former per final subaperture, so the band ends where the pulses are cut
    into subapertures. The JAX former pads a patch's pulses and its range gate to the sizes its compiled programs
    are shared at, and must still cut the pulses where the cpu former does: a mosaic of patches with their own
    pulse counts under form_cphd's rectangle and Taylor window, then the default product and the Taylor-weighted
    one (one patch fitted to the grid). With subapertures of its own the JAX image was 2e-3 to 3e-2 of the peak and
    -23 to -41 dB of the image away from the cpu's; here they agree to 5e-4 and -69 dB or better."""
    pytest.importorskip('jax')
    name, c = strip

    def departure(a, b):
        d = np.abs(b - a).astype(np.float64)
        return d.max() / np.abs(a).max(), 10 * np.log10((d ** 2).sum() / (np.abs(a).astype(np.float64) ** 2).sum())
    runs = [(f'form_cphd {aw}', lambda be, aw=aw: fastsar.form_cphd(name + '.cphd', backend=be, extent=EXT, patch=64,
                                                                    azimuth_window=aw, window=aw == 'taylor'))
            for aw in ('uniform', 'taylor')]
    runs += [(f'process_cphd {kw}', lambda be, kw=kw: fastsar.process_cphd(name + '.cphd', backend=be, extent=EXT, **kw))
             for kw in ({}, dict(weighting='taylor', autofocus=False))]
    worst = []
    for what, run in runs:
        ref, o = run('cpu'), run('jax')
        assert o['image'].shape == ref['image'].shape and o['window'] == ref['window']
        pk, db = departure(ref['image'], np.asarray(o['image']))
        print(f'  {name} {what}: {pk:.1e} of the peak, {db:.1f} dB')
        worst.append((what, float(pk), round(float(db), 1)))
    assert max(w[1] for w in worst) < 2e-3 and max(w[2] for w in worst) < -60, worst


def test_tpu_programs_cut_each_band_where_the_cpu_does(strip, monkeypatch):
    """The same through the tpu backend's programs, whose final stage is a Pallas kernel that takes the two factors
    of the weight (pallas_ffbp.fused_final2's weights=). On a TPU the kernels are the device's; elsewhere they run
    in interpret mode on the CPU (as in tests/test_pallas_e2e.py), which checks what the programs compute and not
    that a TPU compiles them."""
    pytest.importorskip('jax')
    from fastsar import api, pallas_ffbp
    name, c = strip
    weighted = []
    real = 'tpu' in fastsar.available_backends()
    monkeypatch.setattr(api, '_JAX_PROGRAMS', {})                   # the programs made here are not kept
    monkeypatch.setenv('FFBP_FORCE_TPU_KERNELS', '1')
    for k in ('fused_rotate_dec_k2', 'fused_rotate_dec_k3', 'fused_final', 'fused_final2', 'fused_final3'):
        def kernel(*a, _f=getattr(pallas_ffbp, k), _k=k, **kw):
            weighted.append((_k, kw.get('weights') is not None))
            return _f(*a, **(kw if real else {**kw, 'interpret': True}))
        monkeypatch.setattr(pallas_ffbp, k, kernel)
    worst = []
    for what, run in (
            [(f'form_cphd {aw}', lambda be, aw=aw: fastsar.form_cphd(name + '.cphd', backend=be, extent=EXT, patch=64,
                                                                     azimuth_window=aw, window=aw == 'taylor'))
             for aw in ('uniform', 'taylor')]
            + [('process_cphd', lambda be: fastsar.process_cphd(name + '.cphd', backend=be, extent=EXT))]):
        ref, o = run('cpu'), run('tpu')
        d = np.abs(np.asarray(o['image']) - ref['image']).astype(np.float64)
        pk, db = d.max() / np.abs(ref['image']).max(), 10 * np.log10((d ** 2).sum() / (np.abs(ref['image']).astype(np.float64) ** 2).sum())
        print(f'  {name} {what} (tpu programs{"" if real else ", interpreted"}): {pk:.1e} of the peak, {db:.1f} dB')
        worst.append((what, float(pk), round(float(db), 1)))
    assert ('fused_final2', True) in weighted and ('fused_final2', False) not in weighted, set(weighted)
    assert max(w[1] for w in worst) < 2e-3 and max(w[2] for w in worst) < -60, worst


@pytest.mark.cuda
def test_small_gpu_gives_the_same_product(monkeypatch):
    """A GPU without room for the image and a second one of its size (FASTSAR_CUDA_BLOCK_MB: a megabyte to count
    on): the stages after formation run on the host, a note says so, and the product is the device's to float32
    rounding."""
    pytest.importorskip('cupy')
    tg, amp = scene()
    ms.collection('small.cphd', tg, amp, P=PL, phase=5.0 * ((np.arange(PL) - PL / 2) / 500.0) ** 2)
    for kw in (dict(), dict(weighting='sva'), dict(product='sidd')):
        monkeypatch.delenv('FASTSAR_CUDA_BLOCK_MB', raising=False)
        ref = fastsar.process_cphd('small.cphd', backend='cuda', extent=LONG, **kw)
        monkeypatch.setenv('FASTSAR_CUDA_BLOCK_MB', '1')
        o = fastsar.process_cphd('small.cphd', backend='cuda', extent=LONG, **kw)
        assert sum('they ran on the host' in n for n in o['notes']) == 1 and not any('host' in n for n in ref['notes'])
        assert o['autofocus'] == ref['autofocus'] == 'quadratic'
        assert np.abs(o['image'] - ref['image']).max() < 2e-6 * np.abs(ref['image']).max()


@pytest.mark.parametrize('kind', [pytest.param('cupy', marks=pytest.mark.cuda), 'jax'])
def test_stages_on_a_device_array(kind, focused):
    """The stages themselves on a device array against the host's: baseband to rounding of the phase, the band
    limit and the autofocus correction to float32 rounding, the result where the image is."""
    xp = pytest.importorskip('cupy' if kind == 'cupy' else 'jax.numpy')
    from fastsar import _xp
    c = focused[0]
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)
    tg, amp = scene()
    ms.collection('dev.cphd', tg, amp, P=PL, phase=5.0 * s ** 2)
    o = fastsar.form_cphd('dev.cphd', backend='cpu', extent=LONG, azimuth_window='taylor',
                          _plan=lambda g: dict(fit_patch=True))
    # the library's own copy to the device: on its CPU device an older JAX keeps the host buffer of xp.asarray, and
    # the host image is written in place below
    d = dict(o, image=_xp.device(o['image'], 'cuda' if kind == 'cupy' else 'jax'))
    moving.baseband(o)
    moving.baseband(d)
    top = np.abs(o['image']).max()
    assert _xp.kind(d['image']) == kind and np.abs(_xp.host(d['image']) - o['image']).max() < 2e-3 * top
    kw = dict(band=moving.band(dict(o, bandwidth=(2 * c.dsin / c.lam, 1.0))), footprint=c.aperture * c.dy / o['spx'])
    a, ia = moving.autofocus(o['image'], 'quadratic', **kw)
    b, ib = moving.autofocus(d['image'], 'quadratic', **kw)
    assert ia['applied'] and ib['applied'] and _xp.kind(b) == kind
    assert np.abs(ia['fit'] - ib['fit']).max() < 0.02 and np.abs(_xp.host(b) - a).max() < 5e-3 * top
    nx = o['image'].shape[0] // 4 * 4
    keep = (np.abs(np.fft.fftfreq(nx)) < 0.25).astype(np.float32)
    la, lb = moving.band_limit(o['image'][:nx], keep), moving.band_limit(d['image'][:nx], keep)
    assert _xp.kind(lb) == kind and np.abs(_xp.host(lb) - la).max() < 1e-4 * top


def orbit(T, prf, squint=0.0, slide=1.0, drift=0.0, R0=600e3, alt=520e3, v=7600.0):
    """A circular orbit past a flat scene at slant range R0 at closest approach: antenna and SRP positions [P, 3]
    in the scene's tangent frame (x along the track) and times [P]. The beam center moves along the ground at the
    footprint's speed over `slide`, at a squint that drifts by `drift` rad over the collection."""
    Re = 6371e3
    r = Re + alt
    t = (np.arange(int(T * prf)) - int(T * prf) / 2) / prf
    th = v * t / r
    gam = np.arccos((r * r + Re * Re - R0 * R0) / (2 * r * Re))      # the earth angle from the nadir track to the scene
    ant = np.stack([r * np.sin(th), -r * np.cos(th) * np.sin(gam), r * np.cos(th) * np.cos(gam) - Re], 1)
    sq = squint + drift * t / T
    srp = np.stack([v * Re / r * np.cos(gam) / slide * t + R0 * np.tan(sq), np.zeros(len(t)), np.zeros(len(t))], 1)
    return ant, srp, t


@pytest.mark.parametrize('name, kw, grid', [
    ('stripmap', dict(T=8.0), (39000, 5000, 0.77, 1.2)),
    ('long strip, drifting squint', dict(T=14.0, drift=np.radians(0.2)), (104000, 5000, 0.77, 1.2)),
    ('squinted', dict(T=8.0, squint=np.radians(3.0)), (39000, 5000, 0.77, 1.2)),
    ('sliding spotlight', dict(T=11.0, slide=5.8), (34000, 5000, 0.146, 1.0))], ids=lambda v: v if isinstance(v, str) else '')
def test_baseband_phase_in_float32(name, kw, grid):
    """The baseband phase as JAX's device computes it, in float32 and int32, against the phase in double at every
    pixel of three blocks of rows (the first, the middle and the last) of a grid of 30 by 6 km (80 km for the long
    strip, 5 by 5 km for the sliding spotlight) at 600 km: under 1e-3 rad (measured: 2e-5)."""
    jnp = pytest.importorskip('jax.numpy')
    ant, srp, t = orbit(prf=2000.0, **kw)
    lam = 0.031
    speed = np.linalg.norm(np.diff(ant, axis=0), axis=1).mean() * 2000.0
    bm = moving.beam_model(ant, srp, t, 0.8 * lam * 2000.0 / (2 * speed))
    nx, ny, spx, spy = grid
    e1 = bm['flight'] * [1.0, 1.0, 0.0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross([0.0, 0.0, 1.0], e1)
    e2 = e2 if e2 @ (srp[len(srp) // 2] - ant[len(ant) // 2]) > 0 else -e2
    a, b = spx * e1, spy * e2
    o = srp[len(srp) // 2] - nx / 2 * a - ny / 2 * b
    k = -4 * np.pi / lam
    F0 = float(moving._exact(bm, (o + (nx // 2) * a + (ny // 2) * b)[None])[0])
    rows, cols = moving._separable(bm, o, a, b, nx, ny, k, F0)
    assert all(v.dtype in (np.float32, np.int32) for v in rows + cols) and rows[3].shape[1] <= 4
    rows, cols = tuple(jnp.asarray(v) for v in rows), tuple(jnp.asarray(v) for v in cols)
    nb, worst = 24, 0.0
    for r0 in (0, nx // 2 - nb // 2, nx - nb):
        got = np.asarray(moving._baseband_step(jnp.ones((nb, ny), jnp.complex64), np.int32(r0), rows, cols))
        i, j = np.arange(r0, r0 + nb, dtype=np.float64), np.arange(ny, dtype=np.float64)
        want = k * (moving._exact(bm, o + i[:, None, None] * a + j[None, :, None] * b) - F0)
        worst = max(worst, np.abs(np.angle(got * np.exp(-1j * want))).max())
    print(f'  {name}: {nx} x {ny} pixels, {rows[3].shape[1]} terms, largest error {worst:.1e} rad')
    assert worst < 1e-3, worst


def device_arrays(monkeypatch, sizes):
    """Record in `sizes` the bytes of the largest JAX array alive whenever a block comes to the host or an estimate
    starts."""
    import jax
    from fastsar import _xp, autofocus as af

    def seen(fn):
        def wrapped(*a, **k):
            sizes.append(max((x.nbytes for x in jax.live_arrays()), default=0))
            return fn(*a, **k)
        return wrapped
    monkeypatch.setattr(_xp, 'to_host', seen(_xp.to_host))
    monkeypatch.setattr(af, '_map_drift', seen(af._map_drift))


def test_jax_stages_in_blocks(focused, monkeypatch):
    """The stages on JAX's device with a fifth of a megabyte to count on: baseband in blocks of 8 rows, the autofocus
    correction column by column and the guard's sums cell by cell give the image of the unblocked stages to
    float32 rounding, with its zeros; an image on the host crosses the device in the same blocks, written in
    place, and nothing larger than the budget is ever on the device."""
    jnp = pytest.importorskip('jax.numpy')
    from fastsar import _xp, autofocus as af
    c = focused[0]
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)
    tg, amp = scene()
    ms.collection('blk.cphd', tg, amp, P=PL, phase=5.0 * s ** 2)
    o = fastsar.form_cphd('blk.cphd', backend='cpu', extent=LONG, azimuth_window='taylor',
                          _plan=lambda g: dict(fit_patch=True))
    o['image'][:90, :30] = 0
    nx, ny = o['image'].shape
    kw = dict(band=moving.band(dict(o, bandwidth=(2 * c.dsin / c.lam, 1.0))), block=192)
    monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB', raising=False)
    ref = moving.baseband(dict(o, image=jnp.asarray(o['image'])))
    top = np.abs(np.asarray(ref['image'])).max()
    ra, ria = moving.autofocus(ref['image'], 'quadratic', **kw)
    assert ria['applied'] and ria['applied_blocks'].all()
    ref, ra = np.array(ref['image']), np.array(ra)                   # copies: a view would keep the device arrays
    host = moving.baseband(dict(o, image=o['image'].copy()))['image']                      # in double
    assert np.abs(ref - host).max() < 2e-5 * top
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '0.2')
    assert _xp.jax_budget() < o['image'].nbytes / 2
    sizes = []
    device_arrays(monkeypatch, sizes)

    def same(got, want, kind):
        assert _xp.kind(got) == kind
        got = _xp.host(got)
        assert got.dtype == np.complex64 and np.array_equal(got == 0, want == 0) and not got[:90, :30].any()
        assert np.abs(got - want).max() < 1e-5 * top
    dev = moving.baseband(dict(o, image=jnp.asarray(o['image'])))
    same(dev['image'], ref, 'jax')
    b, ib = moving.autofocus(dev['image'], 'quadratic', **kw)
    same(b, ra, 'jax')
    assert np.abs(ib['fit'] - ria['fit']).max() < 1e-3 and abs(ib['part_gain'] - ria['part_gain']) < 1e-5
    del dev, b
    gc.collect()
    sizes.clear()                                                    # from here on the image is on the host
    img = o['image'].copy()
    via = moving.baseband(dict(o, image=img), _jax=True)
    assert via['image'] is img
    same(img, ref, 'numpy')
    with af._via_jax():
        b, ib = moving.autofocus(img, 'quadratic', **kw)
    same(b, ra, 'numpy')
    assert np.abs(ib['fit'] - ria['fit']).max() < 1e-3 and ib['applied_blocks'].all()
    assert len(sizes) > 100 and max(sizes) <= _xp.jax_budget(), (len(sizes), max(sizes))


def test_small_jax_device_gives_the_same_product(monkeypatch):
    """A JAX device without room for the image and what the stages make beside it (FASTSAR_JAX_BLOCK_MB: a fifth
    of a megabyte to count on): the image stays on the host and crosses the device in blocks, a note says so, no array
    on the device is as large as the image, and the product is the one of a device with room to float32
    rounding, with the zeros of its area."""
    pytest.importorskip('jax')
    from fastsar import _xp
    tg, amp = scene()
    ms.collection('sj.cphd', tg, amp, P=PL, phase=5.0 * ((np.arange(PL) - PL / 2) / 500.0) ** 2)
    antenna_strip('sj_ant.cphd', 95.0, 120.0)
    for name, kw in (('sj.cphd', dict(extent=LONG)), ('sj.cphd', dict(extent=LONG, weighting='sva')),
                     ('sj_ant.cphd', dict(extent=EXT, area='black', nesz_db=-20.0))):
        monkeypatch.undo()
        monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB', raising=False)
        ref = fastsar.process_cphd(name, backend='jax', **kw)
        ref = dict(ref, image=np.array(ref['image']))                # a copy: a view would keep the device array
        assert not any('JAX device' in n for n in ref['notes']), ref['notes']
        gc.collect()
        monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '0.2')
        sizes = []
        device_arrays(monkeypatch, sizes)
        o = fastsar.process_cphd(name, backend='jax', **kw)
        assert sum('kept on the host and passed through the device in blocks' in n for n in o['notes']) == 1
        assert [n for n in o['notes'] if 'JAX device' not in n] == ref['notes']
        assert o['autofocus'] == ref['autofocus'] and (name != 'sj.cphd' or o['autofocus'] == 'quadratic')
        assert o['image'].shape == ref['image'].shape
        # the zeros of the area are zeros in both; a null of the apodization at 1e-11 of the peak lands on zero or
        # beside it with the compiled program's rounding, which differs between block shapes and machines
        flips = np.count_nonzero((o['image'] == 0) != (ref['image'] == 0))
        assert flips <= 1e-4 * o['image'].size, (kw, flips)
        e = np.abs(o['image'] - ref['image']).max() / np.abs(ref['image']).max()
        print(f'  {kw}: {e:.1e}, largest device array {max(sizes)} bytes of {ref["image"].nbytes}')
        assert e < 1e-5, (kw, e)
        assert ('area' in kw) == bool((o['image'] == 0).mean() > 1e-3)
        # nothing larger than the budget but the columns of one block's estimate, which stay together (half of the
        # long image; the short one is a single block)
        block = 8 * o['autofocus_info']['block'] * o['image'].shape[1]
        assert max(sizes) <= max(_xp.jax_budget(), block), (kw, max(sizes), block)
        assert 'area' in kw or block < 0.6 * o['image'].nbytes

