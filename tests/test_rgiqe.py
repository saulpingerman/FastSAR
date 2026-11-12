"""The radiometric quality model and the processing areas (fastsar.rgiqe): the RGIQE functions against sarpy, the
antenna gain from every description a CPHD file can carry, the noise level from each source in its order of
preference, noise and clutter measured from simulated phase histories, azimuth ambiguities, and the area masks with
their bounds. The collections are small stand-ins for a sarpy reader, built in memory, and one file written by
sim.write_cphd for the path through sarpy itself."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from fastsar import sim, io, rgiqe

pytestmark = pytest.mark.needs('sarpy')

C = 299792458.0
A_DB = 1.09e5                      # the test beam: one-way gain -A_DB (dcx^2 + dcy^2) dB, 3 dB wide at 0.6 degrees
ROT = np.linalg.qr(np.array([[0.3, -0.8, 0.5], [0.7, 0.1, -0.6], [0.2, 0.9, 0.4]]))[0]   # local axes as ECF rows
ORIGIN = np.array([1.2e6, -4.7e6, 4.1e6])
REF = np.array([300.0, -200.0, 0.0])                                # the scene reference point, local frame
BA, N = 2.0, 1.0e6


class Poly:
    """Stands in for sarpy's polynomial types."""

    def __init__(self, c):
        self.c = np.asarray(c, np.float64)

    def get_array(self, dtype=None):
        return self.c

    def __getitem__(self, i):
        return self.c[i]


class Reader:
    """Stands in for sarpy's CPHD reader: the four members fastsar.rgiqe uses."""

    def __init__(self, meta, pvp, signal=None, support=None):
        self.cphd_meta, self.pvp, self.signal, self.support = meta, pvp, signal, support or {}

    def read_pvp_array(self, index=0):
        return self.pvp

    def read_chip(self, pulses, samples, index=0):
        return self.signal[pulses[0]:pulses[1], samples[0]:samples[1]]

    def read_support_array(self, identifier):
        return self.support[identifier]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def collection(antenna='poly', P=64, K=256, prf=6000.0, toa=15e-6, toae=None, period=400e-6, pt_ref=None, pn_ref=None,
               bn_ref=None, pol='HH', steer=(0.0, 0.0), signal=None, sgn=-1, peak=(0.0, 0.0), collector=None):
    """A spotlight collection from 640 km: the reader stand-in, the meta dict, and what the model should see.
    antenna: 'poly' (axis polynomials and a gain polynomial), 'pvp' (per-pulse axes and steering, gain polynomial),
    'array' (axis polynomials and a gain array) or None. The antenna follows CPHD 1.x: X and Y axes with X x Y the
    mechanical boresight toward the scene, direction cosines of the unit vector from the antenna to the point, the
    array pattern a function of the direction cosines minus the steering `steer`. peak: the direction cosines,
    relative to the steering, at which the array pattern peaks."""
    t = 10.0 + np.arange(P) / prf
    apc = np.stack([7500.0 * (t - t.mean()), np.full(P, -400e3), np.full(P, 500e3)], -1)
    srp = np.tile(REF, (P, 1))
    zb = unit(srp - apc)                                            # the boresight follows the reference point
    xa = unit(np.array([1.0, 0.0, 0.0]) - zb[:, :1] * zb)
    ya = np.cross(zb, xa)
    fc, df = 9.6e9, 1.0 / period
    fields = [('TxTime', 'f8'), ('TOA1', 'f8'), ('TOA2', 'f8'), ('SCSS', 'f8'), ('FX1', 'f8'), ('FX2', 'f8')]
    if toae is not None:
        fields += [('TOAE1', 'f8'), ('TOAE2', 'f8')]
    if antenna == 'pvp':
        fields += [(k + a, 'f8', (3,)) for k in ('Tx', 'Rcv') for a in ('ACX', 'ACY')] + [('TxEB', 'f8', (2,)), ('RcvEB', 'f8', (2,))]
    pv = np.zeros(P, fields)
    pv['TxTime'], pv['TOA1'], pv['TOA2'], pv['SCSS'] = t, -toa, toa, df
    pv['FX1'], pv['FX2'] = fc - 0.5 * K * df, fc + 0.5 * K * df
    if toae is not None:
        pv['TOAE1'], pv['TOAE2'] = -toae, toae
    par = NS(Identifier='CH', FxC=fc, Polarization=NS(TxPol=pol[0], RcvPol=pol[1]),
             TgtRefLevel=None if pt_ref is None else NS(PTRef=pt_ref),
             NoiseLevel=None if pn_ref is None else NS(PNRef=pn_ref, BNRef=bn_ref), Antenna=None)
    m = NS(Global=NS(DomainType='FX', SGN=sgn), Data=NS(Channels=[NS(NumSamples=K)]), Channel=NS(Parameters=[par]),
           Antenna=None, SupportArray=None, CollectionID=NS(CollectorName=collector))
    t_rcv = t + 2 * np.linalg.norm(srp - apc, axis=1) / C
    support, ax, ay, axr, ayr = {}, xa, ya, xa, ya
    if antenna is not None:
        par.Antenna = NS(TxAPCId='APC', TxAPATId='PAT', RcvAPCId='APC', RcvAPATId='PAT')
        gain = Poly([[0.0, 2 * A_DB * peak[1], -A_DB], [2 * A_DB * peak[0], 0.0, 0.0], [-A_DB, 0.0, 0.0]])
        pat = NS(Identifier='PAT', EB=NS(DCXPoly=Poly([0.0]), DCYPoly=Poly([0.0])),
                 Array=NS(GainPoly=gain, AntGPId=None), Element=NS(GainPoly=Poly([[0.0]]), AntGPId=None))
        acf = NS(Identifier='ACF')
        m.Antenna = NS(AntCoordFrame=[acf], AntPhaseCenter=[NS(Identifier='APC', ACFId='ACF')], AntPattern=[pat])
        if antenna == 'pvp':
            for k in ('Tx', 'Rcv'):
                pv[k + 'ACX'], pv[k + 'ACY'], pv[k + 'EB'] = xa @ ROT, ya @ ROT, steer
            acf.XAxisPoly = acf.YAxisPoly = None
        else:                                                       # the axes as straight lines in time, in ECF
            fit = [np.polynomial.polynomial.polyfit(t, a @ ROT, 1) for a in (xa, ya)]
            acf.XAxisPoly, acf.YAxisPoly = (NS(X=Poly(c[:, 0]), Y=Poly(c[:, 1]), Z=Poly(c[:, 2])) for c in fit)
            ax, ay = (unit(np.polynomial.polynomial.polyval(t, c).T @ ROT.T) for c in fit)
            axr, ayr = (unit(np.polynomial.polynomial.polyval(t_rcv, c).T @ ROT.T) for c in fit)   # on reception
            pat.EB = NS(DCXPoly=Poly([steer[0]]), DCYPoly=Poly([steer[1]]))
        if antenna == 'array':
            x = -0.03 + 1e-4 * np.arange(601)
            a = np.zeros((601, 601), [('Gain', 'f4'), ('Phase', 'f4')])
            a['Gain'] = -A_DB * ((x[:, None] - peak[0]) ** 2 + (x[None, :] - peak[1]) ** 2)     # rows DCX, columns DCY
            support['AGP'] = a
            pat.Array = NS(GainPoly=Poly([[0.0]]), AntGPId='AGP')
            m.SupportArray = NS(AntGainPhase=[NS(Identifier='AGP', X0=-0.03, Y0=-0.03, XSS=1e-4, YSS=1e-4)])
    meta = dict(tx=apc, rcv=apc, srp_pulses=srp, tx_time=t, rcv_time=t_rcv,
                R=ROT, origin=ORIGIN, srp=ORIGIN + REF @ ROT, pulses=(0, P), channel_index=0, channel='CH')
    return NS(reader=Reader(m, pv, signal, support), meta=meta, apc=apc, ax=ax, ay=ay, axr=axr, ayr=ayr,
              steer=np.add(steer, peak), toa=toa, P=P, K=K, prf=prf, df=df, wavelength=C / fc)


def expected(col, X, power=False):
    """Two-way gain at the ground points X relative to the reference point, pulse by pulse from the beam's formula
    (the transmit axes at the transmit time, the receive axes at the receive time): (mean amplitude gain)^2, or the
    mean power gain. Also whether every pulse has the echo inside TOA1..TOA2."""
    X = np.asarray(X, np.float64)
    acc, inside = 0.0, np.ones(X.shape[:-1], bool)
    for p in range(col.P):
        def one(Y):
            d = Y - col.apc[p]
            r = np.linalg.norm(d, axis=-1)
            u = d / r[..., None]
            return sum(-A_DB * ((u @ x[p] - col.steer[0]) ** 2 + (u @ y[p] - col.steer[1]) ** 2)
                       for x, y in ((col.ax, col.ay), (col.axr, col.ayr))), r
        (g, r), (g0, r0) = one(X), one(REF)
        a = 10.0 ** (0.05 * (g - g0)) * (r0 / r) ** 2
        acc = acc + (a * a if power else a)
        inside &= np.abs(2 * (r - r0) / C) <= col.toa
    return (acc / col.P if power else (acc / col.P) ** 2), inside


def ground(rad, rng, cross):
    """Ground points at the given offsets (m) from the reference point along and across the line of sight."""
    rng, cross = np.asarray(rng, np.float64), np.asarray(cross, np.float64)
    return rad.ref_point + rng[..., None] * rad.range_dir + cross[..., None] * rad.cross_dir


def swath(col, rad, toa):
    """The ground range offsets (m, to 1 m) between which every pulse has the echo within toa of the reference delay."""
    x = np.arange(-12000.0, 12000.0, 1.0)
    r = np.linalg.norm(ground(rad, x, np.zeros(len(x)))[None] - col.apc[:, None], axis=-1)
    ok = (np.abs(2 * (r - np.linalg.norm(REF - col.apc, axis=-1)[:, None]) / C) <= toa).all(0)
    return x[ok].min(), x[ok].max()


def in_window(col, X, toa):
    """Whether every pulse has the echo of the ground points X [n1, n2, 3] within toa of the reference delay."""
    r = np.linalg.norm(X[None] - col.apc[:, None, None], axis=-1)
    return (np.abs(2 * (r - np.linalg.norm(REF - col.apc, axis=-1)[:, None, None]) / C) <= toa).all(0)


def square(rad, half):
    return ground(rad, half * np.array([-1.0, -1.0, 1.0, 1.0]), half * np.array([-1.0, 1.0, 1.0, -1.0]))


def sicd_meta(noise_db=-30.0, sf=0.5, kind='ABSOLUTE', parameters=None):
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
    from sarpy.io.complex.sicd_elements.CollectionInfo import CollectionInfoType
    return SICDType(Radiometric=RadiometricType(NoiseLevel=NoiseLevelType_(NoiseLevelType=kind, NoisePoly=[[noise_db]]),
                                                SigmaZeroSFPoly=[[sf]]),
                    CollectionInfo=CollectionInfoType(Parameters=parameters or {}))


# ---------------------------------------------------------------------------------------------------- the RGIQE

def test_rgiqe_matches_sarpy():
    from sarpy.processing.sicd import rgiqe as sr
    d = np.r_[0.0, np.logspace(-6, 4, 401)]
    assert np.array_equal(rgiqe.rniirs(d), sr.get_rniirs(d))
    assert rgiqe.rniirs(3.0) == sr.get_rniirs(3.0) and rgiqe.rniirs(1e-3) == sr.get_rniirs(1e-3)
    r = np.linspace(0.0, 9.0, 361)
    assert np.array_equal(rgiqe.information_density(r), sr.get_information_density_for_rniirs(r))
    assert rgiqe.information_density(0.2) == sr.get_information_density_for_rniirs(0.2)
    t = rgiqe._transition()[0]                                      # both branches meet there, and both were covered
    assert d.min() < t < d.max() and r.min() < rgiqe.rniirs(t) < r.max()
    snr = np.logspace(-3, 5, 97)
    for ba in (0.05, 0.94, 12.6):
        assert np.array_equal(rgiqe.snr_to_rniirs(ba, snr), sr.snr_to_rniirs(ba, snr, 1.0)[1])
        assert np.array_equal(rgiqe.snr_to_rniirs(ba, snr), sr.snr_to_rniirs(ba, 0.25 * snr, 0.25)[1])
        np.testing.assert_allclose(rgiqe.rniirs_to_snr(ba, rgiqe.snr_to_rniirs(ba, snr)), snr, rtol=1e-9)
    np.testing.assert_allclose(rgiqe.information_density(rgiqe.rniirs(d)), d, rtol=1e-12, atol=1e-300)


def test_sarpy_names():
    """The metadata members the model reads exist under those names in sarpy's CPHD and SICD classes."""
    from sarpy.io.phase_history.cphd1_elements import Antenna as A, Channel as Ch, SupportArray as Su
    from sarpy.io.complex.sicd_elements.Radiometric import RadiometricType, NoiseLevelType_
    assert {'AntCoordFrame', 'AntPhaseCenter', 'AntPattern'} <= set(A.AntennaType._fields)
    assert {'Identifier', 'XAxisPoly', 'YAxisPoly'} <= set(A.AntCoordFrameType._fields)
    assert {'Identifier', 'ACFId'} <= set(A.AntPhaseCenterType._fields)
    assert {'Identifier', 'EB', 'Array', 'Element'} <= set(A.AntPatternType._fields)
    assert {'DCXPoly', 'DCYPoly'} <= set(A.EBType._fields)
    assert {'GainPoly', 'AntGPId'} <= set(A.GainPhasePolyType._fields)
    assert {'TxAPCId', 'TxAPATId', 'RcvAPCId', 'RcvAPATId'} <= set(Ch.AntennaType._fields)
    assert {'FxC', 'Polarization', 'Antenna', 'TgtRefLevel', 'NoiseLevel'} <= set(Ch.ChannelParametersType._fields)
    assert 'PTRef' in Ch.TgtRefLevelType._fields and {'PNRef', 'BNRef'} <= set(Ch.NoiseLevelType._fields)
    assert {'Identifier', 'X0', 'Y0', 'XSS', 'YSS'} <= set(Su.AntGainPhaseType._fields)
    assert {'NoiseLevel', 'SigmaZeroSFPoly'} <= set(RadiometricType._fields)
    assert {'NoiseLevelType', 'NoisePoly'} <= set(NoiseLevelType_._fields)


# ------------------------------------------------------------------------------------------------- antenna gain

@pytest.mark.parametrize('antenna, steer', [('poly', (0.0, 0.0)), ('poly', (2e-3, -1e-3)), ('pvp', (0.0, 0.0)),
                                            ('pvp', (-1.5e-3, 2.5e-3)), ('array', (0.0, 0.0))])
def test_gain(antenna, steer):
    col = collection(antenna, steer=steer)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    np.testing.assert_allclose(rad.ref_point, REF, atol=1e-6)
    assert abs(rad.prf - col.prf) < 1e-6 and abs(rad.wavelength - col.wavelength) < 1e-12
    assert abs(np.degrees(rad.graze) - np.degrees(np.arctan2(500e3, 400e3 - 200.0))) < 1e-3
    g = np.linspace(-4000.0, 4000.0, 17)
    X = ground(rad, g[:, None], g[None, :])
    got = rad.gain_db(X)
    assert got.shape == (17, 17) and abs(rad.gain_db(rad.ref_point)) < 1e-9
    np.testing.assert_allclose(got, 10 * np.log10(expected(col, X)[0]), atol=0.01)
    assert np.ptp(got) > 10.0
    words = {'poly': 'antenna axis polynomials and gain polynomials', 'pvp': 'gain polynomials', 'array': 'gain arrays'}
    assert any(words[antenna] in n for n in rad.notes)
    assert ('per-pulse antenna axes' in ' '.join(rad.notes)) == (antenna == 'pvp')


def beam_point(col, dc):
    """Where the ray from the mid-aperture antenna with the direction cosines dc meets the ground z = 0, from the
    definition alone: dc[0] X + dc[1] Y + sqrt(1 - |dc|^2) (X x Y)."""
    p = col.P // 2
    u = dc[0] * col.ax[p] + dc[1] * col.ay[p] + np.sqrt(1 - dc[0] ** 2 - dc[1] ** 2) * np.cross(col.ax[p], col.ay[p])
    return col.apc[p] - col.apc[p][2] / u[2] * u


@pytest.mark.parametrize('antenna', ['poly', 'pvp', 'array'])
@pytest.mark.parametrize('steer, peak', [((3e-3, 0.0), (0.0, 0.0)), ((0.0, -4e-3), (0.0, 0.0)), ((0.0, 0.0), (-2e-3, 3e-3)),
                                         ((2e-3, 3e-3), (1e-3, -5e-3))])
def test_beam_points_where_the_standard_says(antenna, steer, peak):
    """A beam steered off the reference point, or a pattern that peaks off its own center: the gain peaks where the
    ray with those direction cosines meets the ground, on the side the antenna axes say. Here +X is the flight
    direction and +Y points below the line of sight, toward the radar's side of the reference point."""
    col = collection(antenna, steer=steer, peak=peak)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    p = col.P // 2
    assert np.cross(col.ax[p], col.ay[p]) @ unit(REF - col.apc[p]) > 0.999999     # X x Y looks at the scene
    assert col.ax[p] @ [1.0, 0.0, 0.0] > 0.99 and col.ay[p] @ rad.range_dir < -0.5
    dc = np.add(steer, peak)
    want = beam_point(col, dc)
    off = np.array([(want - REF) @ rad.range_dir, (want - REF) @ rad.cross_dir])
    assert np.hypot(*off) > 1500.0                                  # well off the reference point
    # the sides, independently: +DCX is along the flight direction (the -cross side here), +DCY toward the radar
    assert np.sign(off[1]) == -np.sign(dc[0]) or dc[0] == 0
    assert np.sign(off[0]) == -np.sign(dc[1]) or dc[1] == 0
    g = np.arange(-6000.0, 6001.0, 50.0)
    got = rad.gain_db(ground(rad, g[:, None], g[None, :]))
    i, j = np.unravel_index(np.argmax(got), got.shape)
    assert abs(g[i] - off[0]) <= 100.0 and abs(g[j] - off[1]) <= 100.0 and got[i, j] > 1.5
    mirrored = ground(rad, np.array([-off[0]]), np.array([-off[1]]))
    assert rad.gain_db(want[None])[0] > got[i, j] - 0.05 and rad.gain_db(mirrored)[0] < got[i, j] - 6.0


def test_gain_polynomial_is_held_beyond_its_lobe():
    """A polynomial is trusted to 15 dB one-way below its peak and holds that value beyond, flagged as unknown; an
    array is read as it is."""
    poly, arr = (rgiqe.Radiometry(c.reader, c.meta, nesz_db=-20.0) for c in (collection('poly'), collection('array')))
    far = ground(poly, np.zeros(3), np.array([6000.0, 12000.0, 16000.0]))
    gp, ga = poly.gain_db(far), arr.gain_db(far)
    assert abs(gp[0] - ga[0]) < 0.02 and ga[1] < -70 and np.all(np.abs(gp[1:] + 30.0) < 0.5)
    assert list(poly._evaluate(far, clamp=True)['clamped']) == [False, True, True]
    assert not arr._evaluate(far, clamp=True)['clamped'].any()
    assert np.all(poly.snr(far, BA, N, ambiguities=False) > [1.0, 0.05, 0.05])   # the SNR itself stays defined there
    mask, _ = rgiqe.area_mask(poly, far[None], 'black', BA, N, ambiguities=False, black_sigma0_db=0.0)
    assert list(mask[0]) == [True, False, False]                    # but no area criterion is met
    assert rgiqe._Pattern.from_poly([[3.0]]) is None                # a constant is no pattern
    up = rgiqe._Pattern.from_poly([[0.0, 0.0, -2e4, 0.0, 4e8], [0.0] * 5, [-2e4, 0.0, 0.0, 0.0, 0.0]])
    x = np.array([0.0, 0.004, 0.02])                                # this one turns upward at 0.005 along its y axis
    v, cl = up(0 * x, x, True)
    assert abs(v[1] - (-2e4 * 0.004 ** 2 + 4e8 * 0.004 ** 4)) < 1e-3 and list(cl) == [False, False, True]
    assert abs(v[2] - (-2e4 * 0.005 ** 2 + 4e8 * 0.005 ** 4)) < 0.02


# --------------------------------------------------------------------------------------------------- noise level

def test_level_sources_in_order():
    kw = dict(pt_ref=2e-3, pn_ref=4.0, bn_ref=0.8)
    col = collection('poly', **kw)
    ref = lambda rad: float(rad.nesz_db(rad.ref_point, BA, N))
    rad = rgiqe.Radiometry(col.reader, col.meta)
    assert rad.level_source == 'cphd' and abs(ref(rad) - 10 * np.log10(4.0 / 0.8 * BA / (N * 2e-3))) < 1e-9
    X = ground(rad, np.array([500.0, -1500.0]), np.array([-2500.0, 1000.0]))
    np.testing.assert_allclose(rad.nesz_db(X, BA, N), ref(rad) - rad.gain_db(X), atol=1e-9)
    np.testing.assert_allclose(rad.nesz_db(X, 2 * BA, 4 * N), rad.nesz_db(X, BA, N) - 10 * np.log10(2.0), atol=1e-9)
    sicd = sicd_meta(parameters={'PREDICTED_RNIIRS': '4.7'})
    rad = rgiqe.Radiometry(col.reader, col.meta, sicd=sicd)
    assert rad.level_source == 'sicd' and abs(ref(rad) - (-30.0 + 10 * np.log10(0.5))) < 1e-9 and rad.sicd_rniirs == 4.7
    rad = rgiqe.Radiometry(col.reader, col.meta, sicd=NS(sicd_meta=sicd), nesz_db=-18.0)
    assert rad.level_source == 'argument' and abs(ref(rad) + 18.0) < 1e-9 and rad.sicd_rniirs == 4.7
    for bad, word in ((sicd_meta(noise_db=-80.0), 'plausible'), (sicd_meta(kind='RELATIVE'), 'ABSOLUTE'),
                      (NS(Radiometric=None, CollectionInfo=None), 'no Radiometric block')):
        rad = rgiqe.Radiometry(col.reader, col.meta, sicd=bad)
        assert rad.level_source == 'cphd' and rad.sicd_rniirs is None and any(word in n for n in rad.notes)
    # the reference signal: sarpy's default by polarization, or the caller's
    snr = lambda rad: float(rad.snr(rad.ref_point[None], BA, N, ambiguities=False)[0])
    co = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    cross = collection('poly', pol='HV')
    cross = rgiqe.Radiometry(cross.reader, cross.meta, nesz_db=-20.0)
    given = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0, sigma0_db=-10.0)
    assert abs(snr(co) - 100.0) < 1e-9 and abs(snr(cross) - 25.0) < 1e-9 and abs(snr(given) - 10.0) < 1e-9
    assert co.sigma0_db == 0.0 and cross.sigma0_db == -6.02 and given.sigma0_db == -10.0
    assert abs(float(co.rniirs(co.ref_point[None], BA, N, False)[0]) - rgiqe.snr_to_rniirs(BA, 100.0)) < 1e-12


def phase_history(P, K, df, toa, noise, clutter, sgn, rs, doppler=1.0, window=None):
    """White noise of the given power per delay bin (a number or a profile over the K bins) and clutter of the given
    power in the bins of the swath |delay| <= toa (or window[0] <= delay <= window[1]), confined to the fraction
    `doppler` of the Doppler band, written as frequency-domain pulses [P, K] whose phase follows sgn. Returns it
    and the clutter power per sample."""
    cn = lambda: (rs.standard_normal((P, K)) + 1j * rs.standard_normal((P, K))) / np.sqrt(2.0)
    tau = np.fft.fftfreq(K, df)
    sw = np.abs(tau) <= toa if window is None else (tau >= window[0]) & (tau <= window[1])
    cl = cn() * np.sqrt(clutter)
    if doppler < 1.0:
        F = np.fft.fft(cl, axis=0)
        F[np.abs(np.fft.fftfreq(P)) > 0.5 * doppler] = 0.0
        cl = np.fft.ifft(F, axis=0)
    rc = np.sqrt(noise) * cn() + cl * sw
    S = np.fft.fft(rc, axis=1) / np.sqrt(K) if sgn < 0 else np.fft.ifft(rc, axis=1) * np.sqrt(K)
    return S.astype(np.complex64), float(np.mean(np.abs(cl * sw) ** 2))


@pytest.mark.parametrize('sgn', [-1, 1])
def test_noise_from_the_doppler_band(sgn):
    """No extended swath: the noise is the floor of the range-Doppler map beside the clutter band."""
    P, K, period, toa = 512, 1024, 400e-6, 60e-6
    S, clutter = phase_history(P, K, 1.0 / period, toa, 3.0, 40.0, sgn, np.random.default_rng(11), doppler=0.4)
    col = collection('array', P=P, K=K, toa=toa, period=period, pt_ref=2e-3, signal=S, sgn=sgn)
    rad = rgiqe.Radiometry(col.reader, col.meta)
    m = rad.measurement
    assert rad.level_source == 'cphd+measured' and m['method'] == 'range-Doppler minimum' and m['pulses'] == 512
    assert abs(rad.noise_per_sample / 3.0 - 1) < 0.1 and abs(rad.clutter_per_sample / clutter - 1) < 0.05
    assert abs(float(rad.nesz_db(rad.ref_point, BA, N)) - 10 * np.log10(rad.noise_per_sample * BA / (N * 2e-3))) < 1e-9
    assert any('measured from 512 pulses' in n for n in rad.notes) and any('mean scene sigma zero' in n for n in rad.notes)
    with pytest.raises(ValueError, match='frequency-domain'):
        col.reader.cphd_meta.Global.DomainType = 'TOA'
        rgiqe.Radiometry(col.reader, col.meta)


@pytest.mark.parametrize('collector, sgn, phase', [(None, -1, -1), (None, 1, 1), ('Capella-12', 1, -1), ('capella', -1, -1)])
def test_noise_uses_the_phase_sign_of_the_reader(collector, sgn, phase):
    """The delay axis of the measurement follows the phase sign fastsar.io reads the file with (Capella files
    declare +1 and follow -1): a swath to one side of the reference delay is found on that side, not mirrored."""
    P, K, period, toa = 512, 1024, 400e-6, 60e-6
    S, clutter = phase_history(P, K, 1.0 / period, toa, 3.0, 40.0, phase, np.random.default_rng(13), doppler=0.4,
                               window=(0.2 * toa, toa))
    col = collection('array', P=P, K=K, toa=toa, period=period, pt_ref=2e-3, signal=S, sgn=sgn, collector=collector)
    col.reader.pvp['TOA1'] = 0.2 * toa
    assert rgiqe._phase_sign(col.reader.cphd_meta) == phase
    rad = rgiqe.Radiometry(col.reader, col.meta)
    assert abs(rad.noise_per_sample / 3.0 - 1) < 0.1 and abs(rad.clutter_per_sample * K / (clutter * K) - 1) < 0.05
    col.reader.cphd_meta.Global.SGN = -sgn                          # read the other way, the swath holds only noise
    if collector is None:
        assert rgiqe.Radiometry(col.reader, col.meta).clutter_per_sample < 0.05 * clutter


def test_noise_from_the_extended_swath():
    """Clutter fills the Doppler band, so the map has no noise-only cell; the noise floor, which falls linearly
    toward the ends of the saved swath, is continued under the scene from either side."""
    P, K, period = 512, 2048, 400e-6
    half, toa = 0.5 * period, 0.03 * period
    tau = np.fft.fftfreq(K, 1.0 / period)
    floor = 5.0 * (1 - np.abs(tau) / (1.1 * half))
    skirt = 60.0 * np.exp(-np.maximum(np.abs(tau) - toa, 0) / (0.2 * toa)) * (np.abs(tau) > toa)   # echoes past the swath
    S, clutter = phase_history(P, K, 1.0 / period, toa, floor + skirt, 60.0, -1, np.random.default_rng(12))
    col = collection('array', P=P, K=K, toa=toa, toae=0.9 * half, period=period, signal=S)
    rad = rgiqe.Radiometry(col.reader, col.meta)
    m = rad.measurement
    assert rad.has_extended and m['method'] == 'extended swath' and m['rd_min'] > 10 * rad.noise_per_sample
    assert abs(rad.noise_per_sample / floor[np.abs(tau) <= toa].mean() - 1) < 0.03
    assert abs(rad.clutter_per_sample / clutter - 1) < 0.05
    # no PTRef: the level rests on the clutter-to-noise ratio and the assumed scene, through the effective area
    assert rad.level_source == 'uncalibrated' and any('not radiometrically calibrated' in n for n in rad.notes)
    g = np.linspace(-9000.0, 9000.0, 361)
    X = ground(rad, g[:, None], g[None, :])
    pw, inside = expected(col, X, power=True)
    area = float((pw * inside).sum() * (g[1] - g[0]) ** 2)
    assert abs(rad.area_eff / area - 1) < 0.02
    nesz = 0.1 * BA * rad.area_eff * rad.noise_per_sample / (N * rad.clutter_per_sample)
    assert abs(float(rad.nesz_db(rad.ref_point, BA, N)) - 10 * np.log10(nesz)) < 1e-9
    assert abs(rad.cnr_db(BA, N) - (-10.0 - 10 * np.log10(nesz))) < 1e-9
    low = rgiqe.Radiometry(col.reader, col.meta, scene_sigma0_db=-13.0)
    assert abs(float(low.nesz_db(low.ref_point, BA, N)) - (10 * np.log10(nesz) - 3.0)) < 1e-9


def test_file_through_sarpy(tmp_path):
    """A file from sim.write_cphd (no antenna, no reference levels) read by sarpy: the path argument, the pulses
    sarpy hands out, and what the model says it could not find."""
    from sarpy.io.phase_history.converter import open_phase_history
    col = sim.make_collect(res=0.5, scene=150.0)
    pos = np.array([[0.0, 0.0, 0.0], [20.0, -30.0, 0.0], [-40.0, 10.0, 0.0]])
    S = sim.add_noise(sim.simulate(col, pos, np.array([3.0, 2.0, 2.0])), 1.5, np.random.default_rng(5))
    path = str(tmp_path / 'sim.cphd')
    sim.write_cphd(path, col, S, 40.7934, -77.86, 351.0, heading=30.0)
    _, meta = io.read_cphd(path, meta=True)
    r = open_phase_history(path)
    m = rgiqe._measure_noise(r, 0, r.read_pvp_array(0), np.arange(col.Np), col.K)
    assert m['method'] == 'range-Doppler minimum' and abs(m['noise'] / 2.25 - 1) < 0.12 and abs(m['clutter'] / 17.0 - 1) < 0.05
    with pytest.raises(ValueError, match='no reference level'):
        rgiqe.Radiometry(path, meta)
    rad = rgiqe.Radiometry(path, meta, nesz_db=-20.0)
    assert rad.level_source == 'argument' and rad.measurement is None
    assert any('no antenna description' in n for n in rad.notes)
    X = ground(rad, np.array([0.0, 60.0, -60.0]), np.array([0.0, 40.0, -40.0]))
    rr = np.linalg.norm(meta['tx'][:, None] - X[None], axis=-1)
    np.testing.assert_allclose(rad.gain_db(X), 20 * np.log10(np.mean((rr[:, :1] / rr) ** 2, axis=0)), atol=1e-6)
    assert np.all(rad.snr(X, BA, N, ambiguities=False) > 0)        # all three lie in the file's delay window
    assert float(rad.snr(ground(rad, np.array([5000.0]), np.array([0.0])), BA, N)[0]) == 0.0


# ------------------------------------------------------------------------------------- ambiguities, delay window

def test_ambiguities():
    col = collection('array')
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    rs = np.random.default_rng(2)
    X = ground(rad, rs.uniform(-3000, 3000, 40), rs.uniform(-3000, 3000, 40))
    apc, vel = col.apc[col.P // 2], np.array([7500.0, 0.0, 0.0])
    fd = lambda Y: 2.0 / col.wavelength * (unit(Y - apc) @ vel)
    den = 0.01
    for k in (-2, -1, 1, 2):
        Y = rad._ambiguity(X, k)
        np.testing.assert_allclose(np.linalg.norm(Y - apc, axis=1), np.linalg.norm(X - apc, axis=1), atol=1e-5)
        np.testing.assert_allclose(fd(Y) - fd(X), k * col.prf, atol=1e-6)
        assert np.all(Y[:, 2] == X[:, 2]) and np.all(Y[:, 1] > apc[1])
        spacing = col.wavelength * np.linalg.norm(REF - apc) * col.prf / (2 * 7500.0)
        assert np.all(np.abs(np.abs(Y[:, 0] - X[:, 0]) / (abs(k) * spacing) - 1) < 0.02)
        den = den + expected(col, Y, power=True)[0]
    assert np.isnan(rad._ambiguity(X, 400)).all()                   # no such Doppler frequency on the ground
    g = expected(col, X)[0]
    with_amb, without = rad.snr(X, BA, N), rad.snr(X, BA, N, ambiguities=False)
    np.testing.assert_allclose(without, g / 0.01, rtol=2e-3)
    np.testing.assert_allclose(with_amb, g / den, rtol=2e-3)
    assert np.all(with_amb < without) and (with_amb / without).min() < 0.5
    np.testing.assert_allclose(rad.rniirs(X, BA, N), rgiqe.snr_to_rniirs(BA, with_amb), atol=1e-12)


def test_delay_window():
    col = collection('array', toa=15e-6, toae=30e-6)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    edge = 0.5 * C * 15e-6 / np.cos(rad.graze)                      # ground range of the window's ends, about 3.6 km
    X = ground(rad, np.array([-2 * edge - 300, -edge - 100, -edge + 100, 0.0, edge - 100, edge + 100, 2 * edge - 100,
                              2 * edge + 300]), np.zeros(8))
    assert list(rad.snr(X, BA, N) > 0) == [False, False, True, True, True, False, False, False]
    assert list(rad.snr(X, BA, N, extended=True) > 0) == [False, True, True, True, True, True, True, False]
    assert list(rad.rniirs(X, BA, N) > 0) == [False, False, True, True, True, False, False, False]
    assert np.isfinite(rad.gain_db(X)).all() and rad.gain_db(X)[-1] < rad.gain_db(X)[3]   # the gain is not cut


# ------------------------------------------------------------------------------------------------------- areas

class Field:
    """Stands in for a Radiometry with a given SNR over the ground."""
    level_source, sicd_rniirs, has_extended, ref_point = 'argument', None, False, np.zeros(3)

    def __init__(self, f):
        self.f = f

    def _nesz_ref(self, bandwidth_area, samples):
        return 0.01

    def _fields(self, points, bandwidth_area, samples, ambiguities=True, sigma0_db=None):
        s = self.f(points[..., 0], points[..., 1])                  # the SNR of whichever scene the criterion asks
        return s, np.ones(s.shape), np.ones(s.shape), np.zeros(s.shape, bool)

    def snr(self, points, bandwidth_area, samples, ambiguities=True, *, extended=False, sigma0_db=None):
        return self._fields(points, bandwidth_area, samples)[0]


def grid(half, step):
    g = np.arange(-half, half + step / 2, step)
    return np.stack(np.broadcast_arrays(g[:, None], g[None, :], 0.0), -1), g


def test_area_is_connected_to_the_scene():
    """Of the places that meet the criterion, only the region around the reference point and those the tasked area
    touches are kept, and the tasked area always is."""
    blob = lambda x, y, x0, s: 100.0 * np.exp(-((x - x0) ** 2 + y * y) / (2 * s * s))
    f = Field(lambda x, y: blob(x, y, 0.0, 800.0) + blob(x, y, 6000.0, 500.0))
    pts, g = grid(9000.0, 100.0)
    x, y = pts[..., 0], pts[..., 1]
    center, island = np.hypot(x, y) <= 2400.0, np.hypot(x - 6000.0, y) <= 1500.0
    mask, info = rgiqe.area_mask(f, pts, 'black', BA, N)
    assert info['criterion'] == 'snr' and info['threshold'] == 1.0 and info['snr_threshold'] == 1.0
    assert mask[center].all() and not mask[np.hypot(x, y) > 2500.0].any() and info['border'] == (False,) * 4
    rect = lambda x1, x2, y1, y2: np.array([[x1, y1, 0.0], [x2, y1, 0.0], [x2, y2, 0.0], [x1, y2, 0.0]])
    apart = rect(-4750.0, -3450.0, 4950.0, 6050.0)                  # corners between the grid lines
    mask, _ = rgiqe.area_mask(f, pts, 'black', BA, N, tasked=apart)
    inside = (x > -4750) & (x < -3450) & (y > 4950) & (y < 6050)
    assert mask[center].all() and mask[inside].all() and not mask[island].any() and not mask[~inside & (np.hypot(x, y) > 2500)].any()
    touching = rect(3450.0, 4750.0, -450.0, 450.0)                  # reaches into the island, not the center
    mask, _ = rgiqe.area_mask(f, pts, 'black', BA, N, tasked=touching)
    assert mask[center].all() and mask[island].all() and mask[(x > 3450) & (x < 4750) & (np.abs(y) < 450)].all()
    assert not mask[(np.hypot(x, y) > 2500) & (np.hypot(x - 6000, y) > 1600) & ~((x > 3450) & (x < 4750) & (np.abs(y) < 450))].any()
    # a criterion the reference point does not meet leaves the tasked area alone, and says so
    mask, info = rgiqe.area_mask(f, pts, 9.0, BA, N, tasked=apart)
    assert np.array_equal(mask, inside) and any('no area beyond the tasked one' in n for n in info['notes'])
    with pytest.raises(ValueError, match="kind must be"):
        rgiqe.area_mask(f, pts, 'everything', BA, N)
    with pytest.raises(ValueError, match="needs the tasked area"):
        rgiqe.area_mask(f, pts, 'tasked', BA, N)
    with pytest.raises(ValueError, match='tasked_rniirs'):
        rgiqe.area_mask(f, pts, 'tasked_quality', BA, N)


def test_level_reproduces_the_mask_and_draws_a_finer_boundary():
    """info['level'] is >= 0 exactly on the mask for every kind and on both evaluation paths, is the margin of the
    criterion inside, stays negative on a region that meets the criterion apart from the scene, and interpolated
    from a coarse grid gives the boundary to a fraction of the grid step."""
    blob = lambda x, y, x0, s: 100.0 * np.exp(-((x - x0) ** 2 + y * y) / (2 * s * s))
    f = Field(lambda x, y: blob(x, y, 0.0, 800.0) + blob(x, y, 6000.0, 500.0))
    pts, g = grid(9000.0, 100.0)
    x, y = pts[..., 0], pts[..., 1]
    apart = np.array([[-4750.0, 4950.0, 0.0], [-3450.0, 4950.0, 0.0], [-3450.0, 6050.0, 0.0], [-4750.0, 6050.0, 0.0]])
    for kind, tasked in (('black', None), ('black', apart), (4.2, apart), ('tasked', apart), (9.0, apart)):
        mask, info = rgiqe.area_mask(f, pts, kind, BA, N, tasked=tasked)
        lev = info['level']
        assert lev.dtype == np.float32 and lev.shape == mask.shape and np.array_equal(lev >= 0, mask)
        assert lev[~mask].max() <= -1e-3 + 1e-9
    mask, info = rgiqe.area_mask(f, pts, 'black', BA, N, tasked=apart)
    lev = info['level']
    snr_db = 10 * np.log10(f.f(x, y))
    center, island = np.hypot(x, y) <= 2300.0, np.hypot(x - 6000.0, y) <= 1400.0
    np.testing.assert_allclose(lev[center], snr_db[center], atol=1e-4)          # the SNR in dB over the threshold of 1
    assert (snr_db[island] > 0).all() and (lev[island] <= -1e-3 + 1e-9).all()   # met, but apart from the scene
    inside = (x > -4750) & (x < -3450) & (y > 4950) & (y < 6050)
    assert (lev[inside] == 0).all()                                 # the tasked area, where the criterion fails
    mask, info = rgiqe.area_mask(f, pts, 4.2, BA, N)
    r = rgiqe.snr_to_rniirs(BA, f.f(x, y))
    np.testing.assert_allclose(info['level'][center & mask], (r - 4.2)[center & mask], atol=1e-5)
    # the lattice path of a large grid
    one = Field(lambda x, y: 100.0 * np.exp(-(x * x / (2 * 2000.0 ** 2) + y * y / (2 * 1000.0 ** 2))))
    big, _ = grid(8000.0, 20.0)
    mask, info = rgiqe.area_mask(one, big, 4.2, BA, N, tasked=apart)
    assert big.shape[0] * big.shape[1] > 300000 and np.array_equal(info['level'] >= 0, mask) and mask.any() and not mask.all()
    # a 400 m grid interpolated to 20 m: the boundary lands within 40 m of the true contour, not within 400 m
    from scipy.ndimage import map_coordinates
    coarse, gc = grid(8000.0, 400.0)
    mask, info = rgiqe.area_mask(one, coarse, 'black', BA, N)
    ij = np.stack(np.meshgrid(np.arange(0, len(gc) - 1 + 1e-9, 0.05), np.arange(0, len(gc) - 1 + 1e-9, 0.05), indexing='ij'))
    fine = map_coordinates(info['level'].astype(np.float64), ij, order=1) >= 0
    exact = one.f(big[..., 0], big[..., 1]) >= 1.0
    assert fine.shape == exact.shape
    from scipy.ndimage import binary_dilation, binary_erosion
    assert not (fine & ~binary_dilation(exact, iterations=2)).any() and not (binary_erosion(exact, iterations=2) & ~fine).any()
    blocky = np.kron(mask, np.ones((20, 20), bool))                 # the mask alone, a block per node
    assert (fine != exact).sum() < 0.1 * (blocky[:exact.shape[0], :exact.shape[1]] != exact[:blocky.shape[0], :blocky.shape[1]]).sum()


def test_area_on_a_large_grid_matches_the_criterion():
    """Above 300,000 points the criterion is evaluated on a lattice and interpolated."""
    f = Field(lambda x, y: 100.0 * np.exp(-(x * x / (2 * 2000.0 ** 2) + y * y / (2 * 1000.0 ** 2))))
    pts, g = grid(8000.0, 20.0)
    assert pts.shape[0] * pts.shape[1] > 300000
    thr = float(rgiqe.rniirs_to_snr(BA, 4.2))
    exact = f.f(pts[..., 0], pts[..., 1]) >= thr
    mask, info = rgiqe.area_mask(f, pts, '4.2', BA, N)
    assert info['criterion'] == 'rniirs' and info['threshold'] == 4.2 and abs(info['snr_threshold'] - thr) < 1e-12
    assert (mask != exact).sum() < 0.01 * exact.sum()
    coarse, _ = rgiqe.area_mask(f, pts[::4, ::4], 4.2, BA, N)      # few enough points to evaluate directly
    assert np.array_equal(coarse, exact[::4, ::4])


def test_area_kinds():
    col = collection('array', toa=15e-6, toae=30e-6)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    tasked = square(rad, 1450.0)                                    # its sides between the grid lines
    g = np.arange(-9000.0, 9001.0, 100.0)
    pts = ground(rad, g[:, None], g[None, :])
    inside = (np.abs(g)[:, None] < 1450.0) & (np.abs(g)[None, :] < 1450.0)
    m_t, info = rgiqe.area_mask(rad, pts, 'tasked', BA, N, tasked=tasked)
    assert np.array_equal(m_t, inside) and info['criterion'] == 'polygon' and info['threshold'] is None
    assert abs(info['center_rniirs'] - float(rad.rniirs(rad.ref_point[None], BA, N)[0])) < 1e-12
    assert abs(info['nesz_ref_db'] + 20.0) < 1e-9 and info['level_source'] == 'argument'
    assert abs(info['center_snr_db'] - 10 * np.log10(float(rad.snr(rad.ref_point[None], BA, N)[0]))) < 1e-9
    gain, window = expected(col, pts)
    snr = gain / 0.01                                               # of the reference scene, without ambiguities
    dark = gain / 0.1                                               # of a scene of -10 dB, which 'black' asks about
    masks = {}
    for kind in (4.8, 4.4, 4.0, 'black'):
        m, info = rgiqe.area_mask(rad, pts, kind, BA, N, tasked=tasked, ambiguities=False)
        thr = 1.0 if kind == 'black' else float(rgiqe.rniirs_to_snr(BA, kind))
        of = dark if kind == 'black' else snr
        want = (of >= thr) & (window if kind != 'black' else in_window(col, pts, 30e-6)) | inside
        edge = np.abs(of / thr - 1) < 0.02                          # leave the points on the contour to rounding
        assert np.array_equal(m[~edge], want[~edge])
        assert m[inside].all() and info['snr_threshold'] == thr and m.sum() > inside.sum()
        assert info['black_sigma0_db'] == (-10.0 if kind == 'black' else None)
        masks[kind] = m
    assert (masks[4.8] <= masks[4.4]).all() and (masks[4.4] <= masks[4.0]).all()
    assert masks[4.8].sum() < masks[4.4].sum() < masks[4.0].sum()
    # 'black': an SNR of at least 1 for a scene of -10 dB, with the range swath opened to the extended delay window
    m, info = rgiqe.area_mask(rad, pts, 'black', BA, N, tasked=tasked, ambiguities=False)
    assert info['criterion'] == 'snr' and any('TOAE1..TOAE2' in n for n in info['notes'])
    rows = np.flatnonzero(m.any(1))
    near, far = swath(col, rad, 15e-6)                              # about 3.6 km either way
    assert g[rows[0]] < near - 500 and g[rows[-1]] > far + 500 and (dark[m & ~inside] >= 0.98).all()
    near_e, far_e = swath(col, rad, 30e-6)
    assert near_e <= g[rows[0]] and g[rows[-1]] <= far_e
    rows = np.flatnonzero(masks[4.0].any(1))
    assert near <= g[rows[0]] < near + 100 and far - 100 < g[rows[-1]] <= far
    # ambiguities take area away
    m_a, _ = rgiqe.area_mask(rad, pts, 4.0, BA, N, tasked=tasked)
    assert (m_a <= masks[4.0]).all() and m_a.sum() < masks[4.0].sum()


def test_tasked_quality():
    col = collection('array', toa=15e-6)
    tasked_of = lambda rad, half: square(rad, half)
    g = np.arange(-9000.0, 9001.0, 100.0)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    pts = ground(rad, g[:, None], g[None, :])
    tasked = tasked_of(rad, 1500.0)
    m_num, _ = rgiqe.area_mask(rad, pts, 4.5, BA, N, tasked=tasked)
    m, info = rgiqe.area_mask(rad, pts, 'tasked_quality', BA, N, tasked=tasked, tasked_rniirs=4.5)
    assert np.array_equal(m, m_num) and (info['tasked_rniirs'], info['tasked_rniirs_source']) == (4.5, 'argument')
    assert info['threshold'] == 4.5 and info['criterion'] == 'rniirs'
    with_sicd = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0, sicd=sicd_meta(parameters={'PREDICTED_RNIIRS': '4.5'}))
    m, info = rgiqe.area_mask(with_sicd, pts, 'tasked_quality', BA, N, tasked=tasked)
    assert np.array_equal(m, m_num) and (info['tasked_rniirs'], info['tasked_rniirs_source']) == (4.5, 'sicd')
    # neither given: the lowest RNIIRS the model predicts in the tasked area, here at one of its corners
    m, info = rgiqe.area_mask(rad, pts, 'tasked_quality', BA, N, tasked=tasked)
    low = float(rad.rniirs(tasked, BA, N).min())
    assert info['tasked_rniirs_source'] == 'model' and abs(info['tasked_rniirs'] - low) < 1e-3
    assert any('lowest RNIIRS the model predicts' in n for n in info['notes']) and len(info['notes']) == 1
    same, _ = rgiqe.area_mask(rad, pts, low - 1e-6, BA, N, tasked=tasked)
    # the corners of the tasked area are grid nodes and sit on the threshold itself, where the last bit of the SNR
    # (which varies with the size of the batch the linear algebra library is handed) decides: compare away from it
    edge = np.abs(rad.rniirs(pts, BA, N) - low) < 1e-4
    assert edge.sum() < 0.01 * m.sum() and np.array_equal(m[~edge], same[~edge])
    assert m.sum() > 1.1 * (np.abs(g) < 1500).sum() ** 2
    # a tasked area that leaves the full-bandwidth delay window still gets a quality, from the SNR alone
    wide = tasked_of(rad, 4500.0)
    m, info = rgiqe.area_mask(rad, pts, 'tasked_quality', BA, N, tasked=wide)
    assert 1.0 < info['tasked_rniirs'] < info['center_rniirs'] and any('outside the delay window' in n for n in info['notes'])
    assert m[(np.abs(g)[:, None] < 4500) & (np.abs(g)[None, :] < 4500)].all() and not m.all()


def test_area_bounds():
    f = Field(lambda x, y: 100.0 * np.exp(-(x * x / (2 * 2000.0 ** 2) + y * y / (2 * 1000.0 ** 2))))
    a, b = 2000.0 * np.sqrt(2 * np.log(100.0)), 1000.0 * np.sqrt(2 * np.log(100.0))     # the SNR 1 ellipse: 6070 by 3035 m
    e1, e2, o = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.zeros(3)
    lo1, hi1, lo2, hi2 = rgiqe.area_bounds(f, 'black', o, e1, e2, BA, N)
    for got, want in ((-lo1, a), (hi1, a), (-lo2, b), (hi2, b)):
        assert want <= got <= want + 100.0                          # the last grid line inside, and a step of padding
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    r1, r2 = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    out = rgiqe.area_bounds(f, 'black', o, r1, r2, BA, N, step=25.0)
    w1, w2 = np.hypot(a * c, b * s), np.hypot(a * s, b * c)
    for got, want in zip(out, (-w1, w1, -w2, w2)):
        assert abs(want) <= abs(got) <= abs(want) + 50.0 and np.sign(got) == np.sign(want)
    assert rgiqe.area_bounds(f, 'black', o, e1, e2, BA, N, limit=4000.0)[:2] == (-4000.0, 4000.0)
    off = np.array([1000.0, -500.0, 0.0])                           # bounds are measured from the given center
    np.testing.assert_allclose(rgiqe.area_bounds(f, 'black', off, e1, e2, BA, N), (lo1 - 1000, hi1 - 1000, lo2 + 500, hi2 + 500))
    tasked = np.array([[5000.0, 2000.0, 0.0], [7000.0, 2000.0, 0.0], [7000.0, 4000.0, 0.0], [5000.0, 4000.0, 0.0]])
    assert rgiqe.area_bounds(f, 'tasked', o, e1, e2, BA, N, tasked=tasked) == (5000.0, 7000.0, 2000.0, 4000.0)
    got = rgiqe.area_bounds(f, 'black', o, e1, e2, BA, N, tasked=tasked)
    assert got == (lo1, 7000.0, lo2, 4000.0)                        # the union with a tasked area off to one side
    with pytest.raises(ValueError, match='needs the tasked area'):
        rgiqe.area_bounds(f, 'tasked', o, e1, e2, BA, N)
    # the model itself: across the line of sight the beam sets the extent, along it the delay window
    col = collection('array', toa=15e-6, toae=30e-6)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    tasked = square(rad, 1500.0)
    args = (rad.ref_point, rad.range_dir, rad.cross_dir, BA, N)
    assert rgiqe.area_bounds(rad, 'tasked', *args, tasked=tasked) == pytest.approx((-1500.0, 1500.0, -1500.0, 1500.0))
    near, far = swath(col, rad, 15e-6)
    q = rgiqe.area_bounds(rad, 4.0, *args, tasked=tasked, ambiguities=False)
    x = np.arange(-9000.0, 9000.0, 10.0)
    snr = expected(col, ground(rad, np.zeros(len(x)), x))[0] / 0.01
    x = x[snr >= rgiqe.rniirs_to_snr(BA, 4.0)]                      # where RNIIRS 4.0 holds across the line of sight
    assert x.max() > 4000 and x.max() <= q[3] <= x.max() + 120 and x.min() - 120 <= q[2] <= x.min()
    assert near - 60 <= q[0] <= near and far <= q[1] <= far + 60
    k = rgiqe.area_bounds(rad, 'black', *args, tasked=tasked, ambiguities=False, black_sigma0_db=0.0)
    assert k[0] < q[0] - 500 and k[1] > q[1] + 500 and k[2] <= q[2] and k[3] >= q[3]
    d = rgiqe.area_bounds(rad, 'black', *args, tasked=tasked, ambiguities=False)     # -10 dB: inside the 0 dB one
    assert all(a <= b for a, b in zip((k[0], d[1], k[2], d[3]), (d[0], k[1], d[2], k[3])))
    assert d[3] < k[3] - 500 and d[2] > k[2] + 500 and d[0] < q[0] - 500 and d[1] > q[1] + 500
    x = np.arange(-9000.0, 9000.0, 10.0)
    x = x[rad.nesz_db(ground(rad, np.zeros(len(x)), x), BA, N) <= -10.0]         # the -10 dB NESZ contour, across
    assert x.max() <= d[3] <= x.max() + 120 and x.min() - 120 <= d[2] <= x.min()
    amb = rgiqe.area_bounds(rad, 4.0, *args, tasked=tasked)
    assert amb[3] < q[3] - 200 and amb[2] > q[2] + 200 and amb[3] >= 1500.0


def test_areas_nest_and_bounds_enclose_them():
    """Tasked, tasked quality and falling RNIIRS thresholds each contain the one before, and so does 'black' when
    its reference is the RGIQE's own (0 dB: the thresholds here need an SNR above 1). With its default of -10 dB,
    'black' contains the RNIIRS areas whose thresholds need an SNR of at least 10 of the reference scene and not
    those that need less. area_bounds is the extent of area_mask with a step of padding."""
    col = collection('array', toa=15e-6, toae=30e-6)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    tasked = square(rad, 1450.0)
    g = np.arange(-12000.0, 12001.0, 200.0)
    pts = ground(rad, g[:, None], g[None, :])
    assert rgiqe.rniirs_to_snr(BA, 4.0) > 1.0
    last, masks = None, {}
    for kind in ('tasked', 'tasked_quality', 4.4, 4.0, 'black'):
        m, info = rgiqe.area_mask(rad, pts, kind, BA, N, tasked=tasked, black_sigma0_db=0.0)
        assert m.any() and info['border'] == (False,) * 4
        assert np.array_equal(info['level'] >= 0, m) and info['level'].dtype == np.float32
        if last is not None:
            assert (last <= m).all() and last.sum() < m.sum()
        last = masks[kind] = m
        b = rgiqe.area_bounds(rad, kind, rad.ref_point, rad.range_dir, rad.cross_dir, BA, N, tasked=tasked, step=200.0,
                              black_sigma0_db=0.0)
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        for lo, hi, first, end in ((b[0], b[1], g[rows[0]], g[rows[-1]]), (b[2], b[3], g[cols[0]], g[cols[-1]])):
            assert first - 200.001 <= lo <= first and end <= hi <= end + 200.001
    dark, info = rgiqe.area_mask(rad, pts, 'black', BA, N, tasked=tasked)
    assert info['black_sigma0_db'] == -10.0 and (dark <= masks['black']).all() and dark.sum() < masks['black'].sum()
    assert (masks['tasked'] <= dark).all() and np.array_equal(info['level'] >= 0, dark)
    high = float(rgiqe.snr_to_rniirs(BA, 10.5))                     # a threshold that needs a little over 10 dB
    m, _ = rgiqe.area_mask(rad, pts, high, BA, N, tasked=tasked)
    assert (m <= dark).all() and m.sum() > masks['tasked'].sum()
    assert (masks[4.0] & ~dark).any() and (masks[4.4] & ~dark).any()            # these need an SNR of 1.2 and 2.0
    b = rgiqe.area_bounds(rad, 'black', rad.ref_point, rad.range_dir, rad.cross_dir, BA, N, tasked=tasked, step=200.0)
    rows, cols = np.flatnonzero(dark.any(1)), np.flatnonzero(dark.any(0))
    for lo, hi, first, end in ((b[0], b[1], g[rows[0]], g[rows[-1]]), (b[2], b[3], g[cols[0]], g[cols[-1]])):
        assert first - 200.001 <= lo <= first and end <= hi <= end + 200.001


def test_black_is_the_nesz_contour_of_its_reference():
    """'black' ends where a scene of black_sigma0_db has an SNR of 1: without ambiguities, on the contour where the
    noise-equivalent sigma zero equals it; with them, where the noise-equivalent sigma zero plus the ambiguity level
    of such a scene does. The default is -10 dB, a lower reference gives a larger area, and the RNIIRS kinds, which
    keep the RGIQE's reference, do not move with it."""
    col = collection('array', toa=15e-6, toae=30e-6)
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-20.0)
    g = np.arange(-9000.0, 9001.0, 100.0)
    pts = ground(rad, g[:, None], g[None, :])
    window = in_window(col, pts, 30e-6)
    gain = expected(col, pts)[0]
    nesz = 0.01 / gain                                              # -20 dB at the reference point
    np.testing.assert_allclose(rad.nesz_db(pts, BA, N), 10 * np.log10(nesz), atol=0.01)
    amb = sum(np.nan_to_num(expected(col, rad._ambiguity(pts, k), power=True)[0]) for k in (-2, -1, 1, 2)) / gain
    assert amb[window].max() > 1.0 and amb.min() < 1e-3             # from negligible to above the signal
    masks = {}
    for ref in (-10.0, -5.0, 0.0):
        sigma = 10.0 ** (0.1 * ref)
        for ambiguities in (False, True):
            level = nesz + sigma * amb if ambiguities else nesz     # as a sigma zero
            m, info = rgiqe.area_mask(rad, pts, 'black', BA, N, ambiguities=ambiguities, black_sigma0_db=ref)
            edge = np.abs(10 * np.log10(level / sigma)) < 0.05      # leave the points on the contour to rounding
            assert np.array_equal(m[~edge], ((level <= sigma) & window)[~edge]), (ref, ambiguities)
            assert info['black_sigma0_db'] == ref and info['snr_threshold'] == 1.0 and info['criterion'] == 'snr'
            s = rad.snr(pts, BA, N, ambiguities, extended=True, sigma0_db=ref)
            np.testing.assert_allclose(s[window], (sigma / level)[window], rtol=5e-3)
            masks[ref, ambiguities] = m
    for ambiguities in (False, True):
        a, b, c = (masks[ref, ambiguities] for ref in (-10.0, -5.0, 0.0))
        assert (a <= b).all() and (b <= c).all() and a.sum() < b.sum() < c.sum() and a.any()
        assert np.array_equal(rgiqe.area_mask(rad, pts, 'black', BA, N, ambiguities=ambiguities)[0], a)    # the default
    assert (masks[-10.0, True] <= masks[-10.0, False]).all() and masks[-10.0, True].sum() < masks[-10.0, False].sum()
    # the reference scene of the RGIQE is another matter: its SNR, and the areas rated by it, stay where they were
    np.testing.assert_allclose(rad.snr(pts, BA, N, extended=True), rad.snr(pts, BA, N, extended=True, sigma0_db=0.0))
    for kind in (4.4, 'tasked_quality'):
        a, b = (rgiqe.area_mask(rad, pts, kind, BA, N, tasked=square(rad, 1450.0), black_sigma0_db=v)[0] for v in (-10.0, 0.0))
        assert np.array_equal(a, b)
    with pytest.raises(ValueError, match='black_sigma0_db'):
        rgiqe.area_mask(rad, pts, 'black', BA, N, black_sigma0_db=np.nan)


def test_given_noise_level_and_arguments():
    col = collection('poly')
    rad = rgiqe.Radiometry(col.reader, col.meta, nesz_db=-17.5, sigma0_db=-10.0)
    X = ground(rad, np.array([0.0, 800.0, -2500.0]), np.array([0.0, -3000.0, 1500.0]))
    gain = expected(col, X)[0]
    np.testing.assert_allclose(rad.nesz_db(X, BA, N), -17.5 - 10 * np.log10(gain), atol=0.01)
    np.testing.assert_allclose(rad.nesz_db(X, 7 * BA, N / 3), rad.nesz_db(X, BA, N), atol=1e-12)   # the image's level
    np.testing.assert_allclose(rad.snr(X, BA, N, ambiguities=False), 10.0 ** 0.75 * gain, rtol=3e-3)
    assert rad.level_source == 'argument' and any('-17.50 dB' in n for n in rad.notes)
    # a part of the aperture, as a slice or as (first, last + 1)
    a, b = (rgiqe.Radiometry(col.reader, col.meta, nesz_db=-17.5, pulses=p) for p in ((8, 40), slice(8, 40)))
    assert a.pulses_used == b.pulses_used == 32 and np.array_equal(a.gain_db(X), b.gain_db(X))
    sub = collection('poly', P=32)
    assert abs(a.prf - sub.prf) < 1e-6 and np.abs(a.gain_db(X) - rad.gain_db(X)).max() < 0.05
    with pytest.raises(ValueError, match='at least two pulses'):
        rgiqe.Radiometry(col.reader, col.meta, nesz_db=-17.5, pulses=(5, 6))
    with pytest.raises(ValueError, match='ground grid'):
        rgiqe.area_mask(rad, X, 'black', BA, N)
    assert set(rgiqe.__all__) == {'rniirs', 'information_density', 'snr_to_rniirs', 'rniirs_to_snr', 'Radiometry',
                                  'area_mask', 'area_bounds'}
