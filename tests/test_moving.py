"""process_cphd on moving-beam collections (stripmap, sliding spotlight): the weightings, autofocus along the track
and the quality areas in range, on the simulated collections of moving_sim.py read through its stand-in for the
CPHD reader.

(a) Weighting. The impulse responses of point targets against theory, in units of the reciprocal bands: Taylor
    (nbar 4, 35 dB) 1.184 wide with a -35 dB peak sidelobe, uniform 0.886 and -13.26 dB, on a broadside stripmap, a
    squinted one and a sliding spotlight; the product at baseband on both axes. Apodization: no sidelobe above
    -30 dB of its target around the main lobe, which keeps the unweighted width; refused where the squint shears
    the spectrum.
(b) Autofocus. A quadratic phase error over slow time injected into the phase history (2, 5 and 15 rad at the edge
    of a synthetic aperture), an error that changes along the track, and targets above the focus plane seen from a
    curved track (one height, and a height that changes across the swath): the impulse response returns to the
    error-free one. A focused collection comes back unchanged.
(c) Areas. With an antenna block in the file, the area in ground range against the two-way gain computed here
    from the beam's formula, the receive window and its extension, a window that holds nothing beyond the tasked
    swath, and a file without an antenna pattern.
(d) The backends give the same products.

Limits are about twice the departures measured when they were set."""
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
    """The default: Taylor on both axes at form_cphd's own spacing (finer where a sliding beam widens the band),
    quadratic autofocus that leaves a focused collection as it came."""
    name, c = strip
    o = fastsar.process_cphd(name + '.cphd', backend='cpu', extent=EXT)
    ref = fastsar.form_cphd(name + '.cphd', backend='cpu', extent=EXT, azimuth_window='taylor', spacing=(o['spx'], o['spy']))
    assert o['weighting'] == 'taylor' and o['autofocus'] is None and o['autofocus_info']['applied'] is False
    assert o['image'].shape == ref['image'].shape
    assert np.abs(np.abs(o['image']) - np.abs(ref['image'])).max() < 2e-3 * np.abs(ref['image']).max()
    # the band of the image along the track: the processed Doppler band, widened by a slide or a squint
    assert any('azimuth band of the image' in n for n in o['notes']) == (name != 'broadside')
    assert 0.9 < o['spx'] / (0.96 * c.ares) < (0.98 if name != 'broadside' else 1.02)
    r = responses(o, c, half=12)
    assert np.abs(r[:, 0] / 1.184 - 1).max() < 0.03, r[:, 0]


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
    ref = fastsar.process_cphd('long.cphd', backend='cpu', extent=LONG, autofocus=False)
    return c, ref, responses(ref, c, half=12)


def test_autofocus_leaves_a_focused_collection(focused):
    c, ref, r0 = focused
    for mode in ('quadratic', 'spatially_variant'):
        o = fastsar.process_cphd('long.cphd', backend='cpu', extent=LONG, autofocus=mode)
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
    blurred = responses(fastsar.process_cphd('quad.cphd', backend='cpu', extent=LONG, autofocus=False), c, 12, 10)
    o = fastsar.process_cphd('quad.cphd', backend='cpu', extent=LONG)
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


def test_autofocus_follows_an_error_along_the_track(focused):
    c, ref, r0 = focused
    tg, amp = scene()
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)
    ms.collection('vary.cphd', tg, amp, P=PL, phase=4.0 * s ** 2 + 0.3 * s ** 3)     # 4 rad +- 0.9 along the strip
    o = fastsar.process_cphd('vary.cphd', backend='cpu', extent=LONG)
    info, r = o['autofocus_info'], responses(o, c, 12, 10)
    print(f'  along track: estimates {info["estimates"].round(2)}, fit {info["fit"].round(2)}, width {r[:, 0].max():.3f}')
    assert info['applied'] and info['fit'][0] < 3.7 and info['fit'][-1] > 4.3 and abs(info['fit'][1] - 4.0) < 0.2
    assert np.abs(r[:, 0] / r0[:, 0] - 1).max() < 0.06 and np.abs(r[:, 4] / r0[:, 4] - 1).max() < 0.05


@pytest.mark.parametrize('heights', ['level', 'tilted'])
def test_autofocus_recovers_a_height_offset(heights):
    """Targets above the focus plane, seen from a track that curves across the line of sight: a quadratic error in
    proportion to the height. 'level': all 40 m up. 'tilted': the height grows across the swath, from -10 to 50 m."""
    tg, amp = scene()
    kw = dict(P=PL, curve=1.7e-4)
    c = ms.collection('curved.cphd', tg, amp, **kw)
    r0 = responses(fastsar.process_cphd('curved.cphd', backend='cpu', extent=LONG, autofocus=False), c, 12)
    h = np.full(len(tg), 40.0) if heights == 'level' else 20.0 + 2.0 * tg[:, 0]
    up = tg + np.stack([h * np.tan(c.graze), 0 * h, h], 1)           # the same image points, on the ground plane
    ms.collection('raised.cphd', up, amp, **kw)
    # the layover of a raised target follows the local direction across the curved track: a few rows along it
    blurred = responses(fastsar.process_cphd('raised.cphd', backend='cpu', extent=LONG, autofocus=False), c, 12, 4)
    o = fastsar.process_cphd('raised.cphd', backend='cpu', extent=LONG)
    info, r = o['autofocus_info'], responses(o, c, 12, 4)
    print(f'  {heights}: fit {info["fit"].round(2)}, slope {info["slope"].round(2)}, width {np.median(blurred[:, 0]):.3f} '
          f'(max {blurred[:, 0].max():.3f}) -> {np.median(r[:, 0]):.3f} (max {r[:, 0].max():.3f}), sidelobe '
          f'{np.median(r[:, 2]):.1f} dB ({np.median(r0[:, 2]):.1f})')
    assert info['applied'] and blurred[:, 0].max() > 2 * r0[:, 0].max()
    assert (np.abs(info['slope']) > 3).all() == (heights == 'tilted')
    assert np.abs(r[:, 0] / r0[:, 0] - 1).max() < 0.03 and np.abs(np.median(r[:, 2]) - np.median(r0[:, 2])) < 1.5


def test_products_on_disk(tmp_path):
    """The SICD of a moving beam states the ground plane, a band centered at zero with the widths the product
    measured from the collection, the weighting, and autofocus that changes over the scene; the apodized SIDD is
    written from the device or the host."""
    from fastsar import _sicdxml
    tg, amp = scene()
    c = ms.collection('disk.cphd', tg, amp, P=PL, phase=5.0 * ((np.arange(PL) - PL / 2) / 500.0) ** 2)
    o = fastsar.process_cphd('disk.cphd', str(tmp_path / 'm.nitf'), backend='cpu', extent=LONG)
    sm = _sicdxml.read(str(tmp_path / 'm.nitf'))
    assert o['path'] == str(tmp_path / 'm.nitf') and o['autofocus'] == 'quadratic'
    assert sm.Grid.ImagePlane == 'GROUND' and sm.Grid.Type == 'PLANE' and sm.ImageFormation.AzAutofocus == 'SV'
    assert sm.Grid.Row.WgtType.WindowName == 'TAYLOR' and sm.Grid.Col.WgtType.WindowName == 'TAYLOR'
    assert sm.Grid.Row.DeltaKCOAPoly[0, 0] == 0 and sm.Grid.Col.DeltaKCOAPoly[0, 0] == 0
    assert abs(sm.Grid.Col.ImpRespBW / (2 * c.dsin / c.lam) - 1) < 0.01          # columns along the track
    assert abs(sm.Grid.Row.ImpRespBW / (2 * c.K * c.df / C * np.cos(c.graze)) - 1) < 0.01
    assert (int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)) == o['image'].shape[::-1]
    d = fastsar.process_cphd('disk.cphd', str(tmp_path / 'd.nitf'), backend='cpu', extent=LONG, product='sidd',
                             weighting='sva')
    assert d['weighting'] == 'sva' and d['autofocus'] == 'quadratic' and (tmp_path / 'd.nitf').stat().st_size > d['image'].size


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


def antenna_strip(name, toa_m, toae_m):
    """A stripmap with an antenna block and receive windows of +-toa_m (extended: +-toae_m) meters of ground range."""
    k = 2 * np.cos(np.radians(30.0)) / C
    return ms.collection(name, FAR, antenna=ANT, toa=k * toa_m, toae=None if toae_m is None else k * toae_m,
                         df=0.75e6)


def test_areas_follow_the_gain_in_range():
    """A wide receive window: 'black' ends where the two-way gain equals the noise-equivalent sigma zero, an RNIIRS
    where the gain gives the ratio it needs, and 'tasked_quality' where the gain falls below its least value in
    the tasked swath; each to one step of the model's lattice across the track (3 m, 0.8 dB of this beam)."""
    from fastsar import rgiqe
    c = antenna_strip('ant.cphd', 95.0, 120.0)
    t = fastsar.process_cphd('ant.cphd', backend='cpu', extent=EXT, autofocus=False)
    bw = t['bandwidth'][0] * t['bandwidth'][1]
    for area, level in (('black', -10.0), (4.8, -10.0 + 10 * np.log10(rgiqe.rniirs_to_snr(bw, 4.8))),
                        ('tasked_quality', gain_db(c, [-25.0, 25.0]).min())):
        o = fastsar.process_cphd('ant.cphd', backend='cpu', extent=EXT, area=area, nesz_db=-10.0, autofocus=False)
        sw = swath(o)
        g = gain_db(c, sw.mean(0))
        print(f'  area {area}: ground range {sw.mean(0).round(1)} m, gain there {g.round(2)} dB (criterion {level:.2f} dB)')
        assert o['area'] == area and o['image'].shape[0] == t['image'].shape[0] and np.ptp(sw, 0).max() < 2 * o['spy']
        assert sw[:, 0].max() <= -25 + o['spy'] and sw[:, 1].min() >= 25 - o['spy']          # never short of the tasked swath
        assert np.abs(g - level).max() < 1.0, (area, sw, g, level)
        assert (o['image'] == 0).any() and any('ground range' in n for n in o['notes'])
        j = ms.to_pixels(o, o['meta'], c.tg)
        inside = (c.tg[:, 0] > sw[1, 0] + 2) & (c.tg[:, 0] < sw[1, 1] - 2)
        assert all(abs(o['image'][round(a), round(b)]) > 0 for (a, b), ok in zip(j, inside) if ok)


def test_areas_end_at_the_receive_window():
    c = antenna_strip('win.cphd', 26.0, 45.0)                        # the extension reaches 45 m, the gain farther
    o = fastsar.process_cphd('win.cphd', backend='cpu', extent=EXT, area='black', nesz_db=-10.0, autofocus=False)
    sw = swath(o)
    print(f'  black, extension +-45 m: ground range {sw.mean(0).round(1)} m')
    assert np.abs(np.abs(sw) - 45).max() < 6 and any('TOAE1' in n for n in o['notes'])
    # the full-bandwidth window ends at the tasked swath: nothing beyond it, and a note that says so
    antenna_strip('tight.cphd', 25.0, None)
    t = fastsar.process_cphd('tight.cphd', backend='cpu', extent=EXT, autofocus=False)
    for area in ('black', 4.0, 'tasked_quality'):
        o = fastsar.process_cphd('tight.cphd', backend='cpu', extent=EXT, area=area, nesz_db=-10.0, autofocus=False)
        assert sum('holds no ground range beyond the tasked swath' in n for n in o['notes']) == 1, o['notes']
        sw = swath(o)
        assert np.abs(np.abs(sw) - 25).max() < 4, sw
        # the tasked swath's pixels are those of the tasked product: the wider grid lies on its lattice
        j0 = ms.to_pixels(o, o['meta'], [[0.0, 0.0, 0.0]])[0, 1] - ms.to_pixels(t, t['meta'], [[0.0, 0.0, 0.0]])[0, 1]
        assert abs(j0 - round(j0)) < 1e-6
        a, b = o['image'][:, round(j0) + 8:round(j0) + t['image'].shape[1] - 8], t['image'][:, 8:-8]
        assert np.abs(np.abs(a) - np.abs(b)).max() < 2e-2 * np.abs(b).max()       # another tiling: the former's accuracy


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
                     ('be_ant.cphd', dict(extent=EXT, area='black', nesz_db=-10.0))):
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


@pytest.mark.parametrize('kind', ['cupy', 'jax'])
def test_stages_on_a_device_array(kind, focused):
    """The stages themselves on a device array against the host's: baseband to rounding of the phase, the band
    limit and the autofocus correction to float32 rounding, the result where the image is."""
    xp = pytest.importorskip('cupy' if kind == 'cupy' else 'jax.numpy')
    from fastsar import _xp
    c = focused[0]
    s = (np.arange(PL) - PL / 2) / (c.aperture / 2)
    tg, amp = scene()
    ms.collection('dev.cphd', tg, amp, P=PL, phase=5.0 * s ** 2)
    o = fastsar.form_cphd('dev.cphd', backend='cpu', extent=LONG, azimuth_window='taylor')
    d = dict(o, image=xp.asarray(o['image']))
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
