"""Radiometric calibration of the SICD (fastsar.rgiqe.calibration, the Radiometric block process_cphd writes): a
simulated spotlight collection with an antenna beam that peaks off the scene center, written as CPHD with PTRef and
PNRef, and processed on both planes with both linear weightings. Point targets of known RCS, clutter of known sigma
zero and noise of known power per sample come back through the written file's own polynomials, read and evaluated
by sarpy, within 0.1 dB. Then the levels of support: PTRef with a measured noise level, no PTRef, apodized
products (a moving beam is in test_cphd.py). Needs sarpy and finufft."""
import dataclasses

import numpy as np
import pytest

import fastsar
from fastsar import io, products, quality, rgiqe, sim

pytestmark = pytest.mark.needs('sarpy', 'finufft')

LAT, LON, H0, HEADING = 40.79, -77.86, 351.0, 30.0
PT_REF, PN_REF = 3.0e-4, 2.0           # power per sample of a 1 m^2 point at the scene center; noise power per sample
BEAM = (3.0e4, 2.0e-3, -3.0e-3)        # one-way dB per squared direction cosine; where the beam peaks
TG = np.array([[0.0, 0.0, 0.0], [31.0, -22.0, 0.0], [-40.0, 36.0, 0.0], [44.0, 41.0, 0.0], [-47.0, -44.0, 0.0],
               [12.0, 49.0, 0.0]])
RCS = np.array([1.0, 4.0, 0.5, 2.0, 1.0, 8.0])
SIGMA0, DENSITY = 0.05, 1.2            # clutter: sigma zero (-13 dB) as scatterers per m^2 of RCS SIGMA0 / DENSITY
MID = 55.0                             # radius (m) about the scene center over which clutter and noise are averaged
CASES = [(plane, weighting) for plane in ('ground', 'slant') for weighting in ('uniform', 'taylor')]
C = 299792458.0


def echoes(col, pos, amp):
    """Phase history [Np, K] of scatterers pos [N, 3] whose amplitude amp [Np, N] changes from pulse to pulse."""
    import finufft
    out = np.empty((col.Np, col.K), np.complex128)
    r0 = np.linalg.norm(col.ant, axis=1)
    for p in range(col.Np):
        dr = np.linalg.norm(pos - col.ant[p], axis=1) - r0[p]
        c = amp[p] * np.exp(-1j * (4.0 * np.pi * col.fref / C) * dr)
        out[p] = finufft.nufft1d1((4.0 * np.pi * col.df / C) * dr, c, col.K, eps=1e-12, isign=-1)
    return out


@pytest.fixture(scope='module', params=[(20.0, False), (-25.0, True)], ids=['right', 'left'])
def scene(request, tmp_path_factory):
    """Three files of one collection: point targets, clutter, and noise alone."""
    squint, left = request.param
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    y = col.ant[:, 1] / np.ptp(col.ant[:, 1])
    ant = col.ant.copy()
    ant[:, 1] -= 10e3 * np.cos(np.radians(35)) * np.tan(np.radians(squint))
    ant[:, 0] += 160.0 * y ** 2 + 120.0 * y ** 3                      # a curved track
    ant[:, 2] += -100.0 * y ** 2 + 6.0 * y
    col = dataclasses.replace(col, ant=ant[::-1].copy() if left else ant)
    d = tmp_path_factory.mktemp('calibration')
    rs = np.random.default_rng(7)
    kw = dict(heading=HEADING, scene=140.0, pt_ref=PT_REF, pn_ref=PN_REF, beam=BEAM)
    S = echoes(col, TG, np.sqrt(RCS * PT_REF) * sim.beam_gain(col, TG, BEAM))
    sim.write_cphd(str(d / 'targets.cphd'), col, S, LAT, LON, H0, **kw)
    n = int(DENSITY * 190.0 ** 2)
    pos = np.column_stack([rs.uniform(-95.0, 95.0, (n, 2)), np.zeros(n)])
    amp = np.sqrt(SIGMA0 / DENSITY * PT_REF) * np.exp(2j * np.pi * rs.uniform(size=n))
    gain = np.concatenate([sim.beam_gain(col, pos[a:a + 4096], BEAM) for a in range(0, n, 4096)], axis=1)
    sim.write_cphd(str(d / 'clutter.cphd'), col, echoes(col, pos, amp * gain), LAT, LON, H0, **kw)
    noise = (rs.standard_normal(S.shape) + 1j * rs.standard_normal(S.shape)) * np.sqrt(PN_REF / 2.0)
    sim.write_cphd(str(d / 'noise.cphd'), col, noise, LAT, LON, H0, **kw)
    sim.write_cphd(str(d / 'unmeasured.cphd'), col, S + noise, LAT, LON, H0, **dict(kw, pn_ref=None))
    sim.write_cphd(str(d / 'uncalibrated.cphd'), col, S, LAT, LON, H0, **dict(kw, pt_ref=None, pn_ref=None))
    # the sigma zero the scatterers drawn inside the averaging disc amount to, and the noise power drawn
    sigma0 = float((np.linalg.norm(pos, axis=1) < MID).sum() * SIGMA0 / DENSITY / (np.pi * MID ** 2))
    return d, col, sim.to_ecf(TG, LAT, LON, H0, HEADING), float(np.mean(np.abs(noise) ** 2)), sigma0


def sarpy_sicd(path):
    from sarpy.io.complex.converter import open_complex
    rd = open_complex(str(path))
    return rd.sicd_meta, rd[:, :]


def coordinates(sm, rows, cols):
    """SICD's image coordinates (m from the scene center point) of pixels."""
    return ((np.asarray(rows, np.float64) - sm.ImageData.SCPPixel.Row) * sm.Grid.Row.SS,
            (np.asarray(cols, np.float64) - sm.ImageData.SCPPixel.Col) * sm.Grid.Col.SS)


def db(x):
    return 10 * np.log10(x)


@pytest.mark.parametrize('plane, weighting', CASES, ids=[f'{p}-{w}' for p, w in CASES])
def test_known_levels_come_back_from_the_written_factors(scene, plane, weighting):
    """RCS of the point targets from their peak power and RCSSFPoly; sigma zero, beta zero and gamma zero of the
    clutter from its mean power and their polynomials; the noise power from NoisePoly, and the noise-equivalent
    sigma zero from both."""
    from sarpy.geometry import point_projection
    d, col, ecf, pn, sigma0 = scene
    kw = dict(plane=plane, weighting=weighting, backend='cpu')
    f = d / 'p.nitf'
    o = fastsar.process_cphd(str(d / 'targets.cphd'), str(f), **kw)
    cal = o['radiometric']
    assert cal['source'] == 'cphd' and cal['residual_db'] < 0.1 and 1 <= cal['order'] <= 6 and cal['gain_span_db'] > 3.0
    assert any('Radiometric block' in n and 'PTRef' in n and 'PNRef' in n for n in o['notes'])
    sm, pix = sarpy_sicd(f)
    rad = sm.Radiometric
    assert sm.Grid.ImagePlane == plane.upper() and rad.NoiseLevel.NoiseLevelType == 'ABSOLUTE'
    for name in ('RCSSFPoly', 'SigmaZeroSFPoly', 'BetaZeroSFPoly', 'GammaZeroSFPoly'):   # as written, not derived
        np.testing.assert_allclose(getattr(rad, name).get_array(), cal[{'RCSSFPoly': 'rcs', 'SigmaZeroSFPoly': 'sigma0',
                                   'BetaZeroSFPoly': 'beta0', 'GammaZeroSFPoly': 'gamma0'}[name]], rtol=1e-12)
    # point targets
    rc, _, _ = point_projection.ground_to_image(ecf, sm)
    m = [quality.point_target(pix, (round(r), round(c)), half=32, up=16) for r, c in rc]
    got = np.array([k['peak'] ** 2 for k in m]) * rad.RCSSFPoly(*coordinates(sm, [k['i'] for k in m], [k['j'] for k in m]))
    err = db(got / RCS)
    assert np.abs(err).max() < 0.1, (plane, weighting, err)
    sf = db(rad.RCSSFPoly(*coordinates(sm, rc[:, 0], rc[:, 1])))
    assert np.ptp(sf) > 2.0                                          # the factor follows the beam over the targets
    # SICD's relations between the factors, with the angles of the scene center point
    slope, graze = np.radians(sm.SCPCOA.SlopeAng), np.radians(sm.SCPCOA.GrazeAng)
    s0, b0, g0 = (getattr(rad, k).get_array() for k in ('SigmaZeroSFPoly', 'BetaZeroSFPoly', 'GammaZeroSFPoly'))
    np.testing.assert_allclose(s0, b0 * np.cos(slope), rtol=1e-12)
    np.testing.assert_allclose(g0, s0 / np.sin(graze), rtol=1e-12)
    # clutter: over the middle of the scene, each pixel's power times its own factor
    c = fastsar.process_cphd(str(d / 'clutter.cphd'), str(f), **kw)
    sm, pix = sarpy_sicd(f)
    rad = sm.Radiometric
    R, Cc = np.meshgrid(np.arange(pix.shape[0]), np.arange(pix.shape[1]), indexing='ij')
    x, y = coordinates(sm, R, Cc)
    g = np.asarray(sm.project_image_to_ground(np.stack([R, Cc], -1).reshape(-1, 2).astype(np.float64)))
    q = (g - sim.to_ecf(np.zeros(3), LAT, LON, H0, HEADING)).reshape(pix.shape + (3,))
    mid = np.linalg.norm(q, axis=-1) < MID                            # ground points within MID of the scene center
    power = np.abs(pix) ** 2
    est = float(np.mean((power * rad.SigmaZeroSFPoly(x, y))[mid]))
    assert abs(db(sigma0 / SIGMA0)) < 0.2 and abs(db(est / sigma0)) < 0.1, (plane, weighting, db(est / sigma0))
    assert abs(db(np.mean((power * rad.BetaZeroSFPoly(x, y))[mid]) / (sigma0 / np.cos(slope)))) < 0.1
    assert abs(db(np.mean((power * rad.GammaZeroSFPoly(x, y))[mid]) / (sigma0 / np.sin(graze)))) < 0.1
    # noise: the pixel power NoisePoly states, and the sigma zero it is equivalent to
    n = fastsar.process_cphd(str(d / 'noise.cphd'), str(f), **kw)
    sm, pix = sarpy_sicd(f)
    rad = sm.Radiometric
    power = np.abs(pix) ** 2
    level = float(rad.NoiseLevel.NoisePoly(0.0, 0.0))
    assert rad.NoiseLevel.NoisePoly.get_array().shape == (1, 1)
    assert abs(db(power[mid].mean()) - level - db(pn / PN_REF)) < 0.1, (plane, weighting, db(power[mid].mean()) - level)
    nesz = float(np.mean((power * rad.SigmaZeroSFPoly(x, y))[mid]))
    want = float(np.mean((10 ** (0.1 * level) * rad.SigmaZeroSFPoly(x, y))[mid]))
    assert abs(db(nesz / want) - db(pn / PN_REF)) < 0.1
    assert abs(n['radiometric']['nesz_db'] - (level + db(rad.SigmaZeroSFPoly(0.0, 0.0)))) < 1e-9
    print(f'{plane}, {weighting}: RCS {err.min():+.3f} to {err.max():+.3f} dB, sigma zero {db(est / sigma0):+.3f} dB, noise '
          f'{db(power[mid].mean()) - level - db(pn / PN_REF):+.3f} dB; order {cal["order"]}, fit {cal["residual_db"]:.3f} dB over '
          f'{cal["gain_span_db"]:.1f} dB of gain; NESZ {n["radiometric"]["nesz_db"]:.2f} dB')


def test_factors_are_the_model_at_each_pixel(scene):
    """The polynomials against the model itself at pixels over the whole image, through sarpy's projection: the
    orientation of both image coordinates, on both planes."""
    from sarpy.geometry import point_projection
    d, col, ecf, pn, _ = scene
    _, meta = io.read_cphd(str(d / 'targets.cphd'), meta=True)
    spans = {}
    for plane in ('ground', 'slant'):
        f = d / 'm.nitf'
        o = fastsar.process_cphd(str(d / 'targets.cphd'), str(f), plane=plane, backend='cpu')
        sm, pix = sarpy_sicd(f)
        model = rgiqe.Radiometry(str(d / 'targets.cphd'), meta, pulses=o['pulses'])
        assert model.level_source == 'cphd' and model.has_antenna
        r, c = np.meshgrid(np.linspace(0, pix.shape[0] - 1, 15), np.linspace(0, pix.shape[1] - 1, 15), indexing='ij')
        r, c = np.rint(r).astype(int).ravel(), np.rint(c).astype(int).ravel()
        data = pix[r, c] != 0                                        # a slant-plane image is zero beyond the ground formed
        assert data.all() if plane == 'ground' else 0.5 < data.mean() < 1.0
        r, c = r[data], c[data]
        g = point_projection.image_to_ground(np.stack([r, c], -1).astype(np.float64), sm)
        gain = model.gain_db(io.ecf_to_local(g, meta))
        n = (o['pulses'][1] - o['pulses'][0]) * col.K
        x, y = coordinates(sm, r, c)
        got = db(sm.Radiometric.RCSSFPoly(x, y))
        np.testing.assert_allclose(got, -db(n ** 2 * PT_REF) - gain, atol=0.1)
        assert np.ptp(gain) > 3.0 and abs(np.corrcoef(got, -gain)[0, 1] - 1) < 1e-3
        # the beam peaks off the scene center, on one side in each coordinate: mirrored, the factors would not fit
        for flip in ((-x, y), (x, -y)):
            assert np.abs(db(sm.Radiometric.RCSSFPoly(*flip)) - got).max() > 0.5
        spans[plane] = o['radiometric']['gain_span_db']
    assert abs(spans['slant'] - spans['ground']) < 0.5               # fitted over the same ground, the pixels with data


def test_levels_of_support(scene, tmp_path):
    d, col, ecf, pn, _ = scene
    # PTRef and no PNRef: a calibrated signal, the noise measured from the phase history, and said so
    f = tmp_path / 'a.nitf'
    full = fastsar.process_cphd(str(d / 'targets.cphd'), backend='cpu')
    o = fastsar.process_cphd(str(d / 'unmeasured.cphd'), str(f), backend='cpu')
    cal = o['radiometric']
    assert cal['source'] == 'cphd+measured' and any('measured from the phase history' in n for n in o['notes'])
    for k in ('rcs', 'sigma0', 'beta0', 'gamma0'):
        np.testing.assert_allclose(cal[k], full['radiometric'][k], rtol=1e-12)
    sm, _ = sarpy_sicd(f)
    assert sm.Radiometric.NoiseLevel.NoiseLevelType == 'ABSOLUTE'
    assert cal['noise_db'] != full['radiometric']['noise_db'] and abs(cal['noise_db'] - full['radiometric']['noise_db']) < 0.6
    assert float(sm.Radiometric.NoiseLevel.NoisePoly(0.0, 0.0)) == pytest.approx(cal['noise_db'])
    # no PTRef: no block, and the note says why
    o = fastsar.process_cphd(str(d / 'uncalibrated.cphd'), str(f), backend='cpu')
    assert o['radiometric'] is None and sarpy_sicd(f)[0].Radiometric is None
    assert sum('No Radiometric block' in n and 'PTRef' in n for n in o['notes']) == 1
    # an apodized product is not linear in the scene: none either
    o = fastsar.process_cphd(str(d / 'targets.cphd'), str(f), plane='slant', weighting='sva', backend='cpu')
    assert o['radiometric'] is None and sarpy_sicd(f)[0].Radiometric is None
    assert sum('No Radiometric block' in n and 'apodization' in n for n in o['notes']) == 1
    # a SIDD has no such block, and no note about one
    o = fastsar.process_cphd(str(d / 'targets.cphd'), product='sidd', backend='cpu')
    assert 'radiometric' not in o and not any('Radiometric' in n for n in o['notes'])
    # the calibration belongs to the image it was made for: with another scene center point it is not written
    moved = dict(full, scp_pixel=(full['image'].shape[0] // 2 + 40, full['image'].shape[1] // 2))
    assert products.sicd_meta(full)[0].Radiometric is not None and products.sicd_meta(moved)[0].Radiometric is None
    assert rgiqe.reference_levels(str(d / 'targets.cphd'), full['meta']) == (PT_REF, PN_REF)
    assert rgiqe.reference_levels(str(d / 'unmeasured.cphd'), full['meta']) == (PT_REF, None)
    assert rgiqe.reference_levels(str(d / 'uncalibrated.cphd'), full['meta']) == (None, None)


def test_support_area_and_weights(scene):
    """The area the samples fill in the ground plane's spatial frequencies, against the energy of a point target's
    response (its integral over the ground is the peak power times sum(w^2) samples / (sum(w)^2 area))."""
    d, col, ecf, pn, _ = scene
    for weighting in ('uniform', 'taylor'):
        o = fastsar.process_cphd(str(d / 'targets.cphd'), weighting=weighting, autofocus=False, backend='cpu')
        n = (o['pulses'][1] - o['pulses'][0]) * col.K
        ant = 0.5 * (o['meta']['tx'] + o['meta']['rcv'])[o['pulses'][0]:o['pulses'][1]]
        area = rgiqe.support_area(ant, o['band'], o['center'])
        cal = o['radiometric']
        wf = float(cal['rcs'][0, 0] / cal['sigma0'][0, 0] * area)   # sum(w^2) samples / sum(w)^2
        assert abs(wf - (1.0 if weighting == 'uniform' else 1.5234)) < 2e-3
        img = o['image']
        nx, ny = img.shape                                          # the target at the scene center, the most isolated
        w = np.abs(img[nx // 2 - 8:nx // 2 + 9, ny // 2 - 8:ny // 2 + 9])
        i, j = (int(v) for v in np.add(np.unravel_index(np.argmax(w), w.shape), (nx // 2 - 8, ny // 2 - 8)))
        h = 40
        energy = float((np.abs(img[i - h:i + h, j - h:j + h]) ** 2).sum() * o['spx'] * o['spy'])
        peak = quality.point_target(img, (i, j), half=32, up=16)['peak'] ** 2
        print(f'{weighting}: support area {area:.4f}, wf {wf:.4f}, energy / peak against wf / area {db(energy / peak * area / wf):+.3f} dB')
        assert abs(db(energy / peak * area / wf)) < 0.2, (weighting, db(energy / peak * area / wf))


def test_fit_follows_a_steep_gain_and_says_what_it_reaches():
    """rgiqe.calibration on stand-ins: the order rises with the span of the gain over the image, the residual is
    the fit's error against the model between its nodes, and a span no polynomial of the highest order follows is
    reported, not hidden."""
    from types import SimpleNamespace as NS

    class Poly:
        def __init__(self, a):
            self.a = np.asarray(a, np.float64)

        def get_array(self):
            return self.a

    def sicd(rows=801, cols=601, ss=1.0):
        s = NS(ImageData=NS(NumRows=rows, NumCols=cols, SCPPixel=NS(Row=rows // 2, Col=cols // 2)),
               Grid=NS(Row=NS(SS=ss, UVectECF=Poly([1.0, 0.0, 0.0])), Col=NS(SS=ss, UVectECF=Poly([0.0, 1.0, 0.0]))),
               SCPCOA=NS(SlopeAng=40.0, GrazeAng=35.0), GeoData=NS(SCP=NS(ECF=Poly([0.0, 0.0, 0.0]))))
        s.project_image_to_ground = lambda rc: np.stack([(rc[..., 0] - rows // 2) * ss, (rc[..., 1] - cols // 2) * ss,
                                                         np.zeros(rc.shape[:-1])], -1)
        return s

    class Model:
        level_source, pt_ref, noise_per_sample = 'cphd', 2.0e-3, 5.0
        meta = dict(R=np.eye(3), origin=np.zeros(3))

        def __init__(self, depth_db):
            self.k = depth_db / 400.0 ** 2

        def gain(self, p):
            return 10.0 ** (-0.1 * self.k * ((p[..., 0] - 60.0) ** 2 + 1.5 * (p[..., 1] + 40.0) ** 2))

        def _evaluate(self, points, clamp=False):
            g = self.gain(points) / self.gain(np.zeros(3))
            return dict(gain=g, clamped=np.zeros(g.shape, bool))

    n, area, last = 1.0e6, 2.0, -1
    for depth, reached in ((0.0, True), (0.5, True), (2.0, True), (6.0, True), (60.0, False)):
        m = Model(depth)
        cal = rgiqe.calibration(m, sicd(), n, area)
        assert (cal['residual_db'] <= 0.1) == reached, (depth, cal['order'], cal['residual_db'])
        assert cal['noise_db'] == pytest.approx(10 * np.log10(n * 5.0))
        assert cal['sigma0'][0, 0] / cal['rcs'][0, 0] == pytest.approx(area)            # unweighted
        if not reached:                                              # 60 dB over the image: beyond any such fit
            assert cal['order'] <= 8 and cal['gain_span_db'] > 60.0
            continue
        assert cal['order'] >= last and (cal['order'] == 0) == (depth == 0.0)
        last = cal['order']
        # the constant terms are the factors at the scene center point, where the gain is 1
        assert cal['rcs'][0, 0] == pytest.approx(1 / (n ** 2 * 2.0e-3), rel=0.03)
        assert cal['nesz_db'] == pytest.approx(10 * np.log10(5.0 * area / (n * 2.0e-3)), abs=0.11)
        # against the model at pixels that are not nodes of the fit
        rs = np.random.default_rng(1)
        rc = np.stack([rs.uniform(0, 800, 500), rs.uniform(0, 600, 500)], -1)
        x, y = rc[:, 0] - 400, rc[:, 1] - 300
        p = np.polynomial.polynomial.polyval2d(x, y, cal['rcs']) * n ** 2 * 2.0e-3
        e = np.abs(10 * np.log10(p * m._evaluate(sicd().project_image_to_ground(rc))['gain']))
        assert e.max() <= 1.2 * cal['residual_db'] + 0.01, (depth, e.max(), cal['residual_db'])
    weighted = rgiqe.calibration(Model(1.0), sicd(), n, area, coherent=0.8 * n, noise=0.9 * n)
    plain = rgiqe.calibration(Model(1.0), sicd(), n, area)
    np.testing.assert_allclose(weighted['rcs'], plain['rcs'] / 0.64, rtol=1e-12)
    np.testing.assert_allclose(weighted['sigma0'], plain['sigma0'] / 0.9, rtol=1e-12)
    assert weighted['noise_db'] == pytest.approx(plain['noise_db'] + 10 * np.log10(0.9))
    assert weighted['nesz_db'] == pytest.approx(plain['nesz_db'])                    # the weighting cancels in it
    np.testing.assert_allclose(plain['beta0'], plain['sigma0'] / np.cos(np.radians(40.0)), rtol=1e-12)
    np.testing.assert_allclose(plain['gamma0'], plain['sigma0'] / np.sin(np.radians(35.0)), rtol=1e-12)
    # a region: only its points are fitted and judged
    half, whole = (rgiqe.calibration(Model(6.0), sicd(), n, area, region=r) for r in (lambda pts: pts[..., 0] > 0, None))
    assert half['order'] <= whole['order'] and half['gain_span_db'] < whole['gain_span_db'] - 3.0
    tasked = np.array([[-100.0, -100.0, 0.0], [100.0, -100.0, 0.0], [100.0, 100.0, 0.0], [-100.0, 100.0, 0.0]])
    t = rgiqe.calibration(Model(60.0), sicd(), n, area, tasked=tasked)
    assert t['tasked_residual_db'] <= t['residual_db'] and rgiqe.calibration(m, sicd(), n, area)['tasked_residual_db'] is None
    with pytest.raises(ValueError, match='PTRef'):
        none = Model(1.0)
        none.pt_ref = None
        rgiqe.calibration(none, sicd(), n, area)
