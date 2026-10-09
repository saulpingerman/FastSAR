"""io.read_cphd and form_cphd end to end on simulated FX-domain collections, through a stand-in for sarpy's CPHD
reader (no CPHD writer is a dependency, and the tests write no files): open_phase_history returns the metadata
fields, per-vector parameters and samples that read_cphd and form_cphd read, built from a simulated collection placed
on the WGS-84 ellipsoid at the equator. The stand-in replaces sarpy's modules in sys.modules for this module's
tests only.

(a) Spotlight: a fixed scene reference point (SRP), the grid from the image area. read_cphd returns the samples
    and the antenna path in its local frame; SGN = +1 data are conjugated back; non-finite samples are zeroed and
    pulses without positions at the ends are trimmed. form_cphd against exact backprojection of the same windowed
    samples onto its output grid.
(b) Moving SRP (stripmap): the SRP of each pulse is the ground point broadside of the antenna, and the echoes carry
    an azimuth pattern inside the Doppler band the PRF samples. form_cphd (a mosaic with a Hann window over the
    azimuth band about each pulse's SRP) against a direct float64 sum with that window, around each target.

Errors are 10 log10 of the energy of the difference over that of the reference; limits about 5 dB above the
values measured when they were set."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import io, sim, stripmap as sm

C = 299792458.0


def check(bad, name, value, limit, fmt='{:.1f} dB'):
    ok = value <= limit
    print(f'  {name:60s} {fmt.format(value):>10s}  (limit {fmt.format(limit)})' + ('' if ok else '  FAIL'))
    if not ok:
        bad.append(name)


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2) + 1e-300)


# ---------------------------------------------------------------------------------------------- the stand-in reader
NS = types.SimpleNamespace
FILES = {}


class XYZ:
    def __init__(self, v):
        self.v = np.asarray(v, np.float64)

    def get_array(self):
        return self.v.copy()


class Reader:
    def __init__(self, path):
        self.d = FILES[path]
        self.cphd_meta = self.d['meta']

    def read_pvp_variable(self, name, index):
        v = self.d['pvp'].get(name)
        return None if v is None else np.array(v)

    def read_chip(self, rows, cols, index=0):
        return self.d['S'][rows[0]:rows[1], cols[0]:cols[1]].copy()


@pytest.fixture(scope='module', autouse=True)
def stand_in_reader():
    """sarpy's CPHD reader replaced by the stand-in while this module runs; the real modules, if loaded, return
    afterwards."""
    with pytest.MonkeyPatch.context() as mp:
        for name in ('sarpy', 'sarpy.io', 'sarpy.io.phase_history', 'sarpy.io.phase_history.converter'):
            mp.setitem(sys.modules, name, types.ModuleType(name))
        sys.modules['sarpy.io.phase_history.converter'].open_phase_history = Reader
        yield
    FILES.clear()


# a local east-north-up frame at the equator (geodetic and geocentric up agree there)
lat, lon = 0.0, 30.0
srp0 = io.geodetic_to_ecf(lat, lon, 0.0)
E = np.array([-np.sin(np.radians(lon)), np.cos(np.radians(lon)), 0.0])
N = np.array([0.0, 0.0, 1.0])
U = srp0 / np.linalg.norm(srp0)


def ecf(local):
    """Simulator coordinates (x north, y east, z up, origin the SRP) -> ECF."""
    local = np.asarray(local, np.float64)
    return srp0 + local[..., :1] * N + local[..., 1:2] * E + local[..., 2:3] * U


def cphd(path, S, ant, srp, f0, df, prf, sgn=-1, image_area=None, mode='SPOTLIGHT'):
    """Register a collection: S [P, K] in the CPHD convention for SGN = -1 (conjugated for +1), antenna and SRP
    positions [P, 3] (ECF)."""
    P, K = S.shape
    t = np.arange(P) / prf
    r = np.linalg.norm(ant - srp, axis=1)
    pvp = dict(TxPos=ant, RcvPos=ant, SRPPos=srp, SC0=np.full(P, f0), SCSS=np.full(P, df), TxTime=t, RcvTime=t + 2 * r / C)
    sc = NS(IARP=NS(ECF=XYZ(srp[P // 2])), ImageArea=None, ReferenceSurface=None)
    if image_area is not None:
        sc.ImageArea = NS(X1Y1=NS(X=-image_area[0] / 2, Y=-image_area[1] / 2), X2Y2=NS(X=image_area[0] / 2, Y=image_area[1] / 2))
        sc.ReferenceSurface = NS(Planar=NS(uIAX=XYZ(N), uIAY=XYZ(E)))
    meta = NS(Data=NS(Channels=[NS(NumVectors=P, NumSamples=K, Identifier='VV')]),
              Channel=NS(Parameters=[NS(Polarization=NS(TxPol='V', RcvPol='V'))]),
              Global=NS(DomainType='FX', SGN=sgn), CollectionID=NS(CollectorName='SIMULATED', RadarMode=NS(ModeType=mode)),
              SceneCoordinates=sc)
    FILES[path] = dict(meta=meta, pvp=pvp, S=np.conj(S) if sgn > 0 else S.copy())


def taylor(n):
    from scipy.signal.windows import taylor as t
    return t(n, nbar=4, sll=35.0, norm=False)


# ---------------------------------------------------------------------------------------------- (a) spotlight

@pytest.fixture(scope='module')
def spot():
    print('(a) spotlight, fixed SRP')
    rng = np.random.default_rng(5)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    tg = np.stack([rng.uniform(-22, 22, 12), rng.uniform(-22, 22, 12), np.zeros(12)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(12) + 1j * rng.standard_normal(12)).astype(np.complex64)
    P, K = S.shape
    ant = ecf(col.ant)
    prf = 1000.0
    cphd('spot.cphd', S, ant, np.repeat(srp0[None], P, 0), col.fmin, col.df, prf, image_area=(50.0, 50.0))
    return SimpleNamespace(col=col, tg=tg, S=S, P=P, K=K, ant=ant, prf=prf)


def test_read_cphd(spot):
    bad = []
    col, S, P, K, ant, prf = spot.col, spot.S, spot.P, spot.K, spot.ant, spot.prf
    c, meta = io.read_cphd('spot.cphd', meta=True)
    check(bad, 'read_cphd: samples (max abs difference)', float(np.abs(c['S'] - S).max()), 0, '{:.2g}')
    check(bad, 'read_cphd: antenna path against the local frame (m)', float(np.abs(c['ant'] - io.ecf_to_local(ant, meta)).max()), 1e-6, '{:.2g}')
    check(bad, 'read_cphd: ranges from the antenna path kept (m)', float(np.abs(np.linalg.norm(c['ant'], axis=1) - np.linalg.norm(col.ant, axis=1)).max()), 1e-6, '{:.2g}')
    cphd('spot_sgn.cphd', S, ant, np.repeat(srp0[None], P, 0), col.fmin, col.df, prf, sgn=+1)
    check(bad, 'read_cphd: SGN = +1 conjugated back (max abs difference)', float(np.abs(io.read_cphd('spot_sgn.cphd')['S'] - S).max()), 0, '{:.2g}')
    # non-finite samples in one pulse, and no positions for the first two pulses
    Sb, ab = S.copy(), ant.copy()
    Sb[50, 7] = np.nan
    ab[:2] = np.nan
    cphd('spot_bad.cphd', Sb, ab, np.repeat(srp0[None], P, 0), col.fmin, col.df, prf)
    cb, mb = io.read_cphd('spot_bad.cphd', meta=True)
    ok = cb['S'].shape == (P - 2, K) and cb['S'][48, 7] == 0 and np.isfinite(cb['S']).all() and mb['pulses'] == (2, P)
    check(bad, 'read_cphd: NaN sample zeroed, 2 pulses without positions trimmed', float(not ok), 0, '{:.0f}')
    print('  notes:', '; '.join(mb['notes']))
    try:
        cphd('spot_empty.cphd', S * 0, ant, np.repeat(srp0[None], P, 0), col.fmin, col.df, prf)
        io.read_cphd('spot_empty.cphd')
        bad.append('read_cphd of an empty file did not raise')
    except ValueError as e:
        print(f'  read_cphd, no valid pulse: ValueError: {e}')
    try:
        io.read_cphd('spot.cphd', channel=3)
        bad.append('read_cphd of a missing channel did not raise')
    except ValueError as e:
        print(f'  read_cphd, channel 3: ValueError: {e}')
    assert not bad, 'FAILED: ' + '; '.join(bad)


def test_form_cphd_spotlight(spot):
    bad = []
    col, S, P, K, ant = spot.col, spot.S, spot.P, spot.K, spot.ant
    meta = io.read_cphd('spot.cphd', meta=True)[1]
    out = fastsar.form_cphd('spot.cphd', backend='cpu')
    img, o, e1, e2 = out['image'], out['origin'], out['e1'], out['e2']
    print(f'  form_cphd: mode {out["mode"]}, grid {img.shape} of {out["spx"]:.3f} x {out["spy"]:.3f} m; ' + '; '.join(out['notes']))
    assert out['mode'] == 'spotlight'
    X, Y = np.meshgrid(np.arange(img.shape[0]) * out['spx'], np.arange(img.shape[1]) * out['spy'], indexing='ij')
    pts = o + X[..., None] * e1 + Y[..., None] * e2
    ref = fastsar.backproject(S * np.outer(taylor(P), taylor(K)), io.ecf_to_local(ant, meta), col.fmin, col.df, pts,
                              ref=np.linalg.norm(ant - srp0, axis=1), backend='cpu', window=False, upsample=16)
    check(bad, 'form_cphd against exact backprojection on its grid', rel_db(img, ref), -45)
    tl = io.ecf_to_local(ecf(spot.tg), meta)
    inside = [((t - o) @ e1 / out['spx'], (t - o) @ e2 / out['spy']) for t in tl]
    check(bad, 'targets inside the image area grid', float(sum(not (0 <= i < img.shape[0] and 0 <= j < img.shape[1]) for i, j in inside)), 0, '{:.0f}')
    assert not bad, 'FAILED: ' + '; '.join(bad)


@pytest.mark.parametrize('b', ['jax'])
def test_form_cphd_spacing_extent(spot, b):
    bad = []
    col, S, P, K, ant = spot.col, spot.S, spot.P, spot.K, spot.ant
    meta = io.read_cphd('spot.cphd', meta=True)[1]
    o2 = fastsar.form_cphd('spot.cphd', backend=b, spacing=0.4, extent=(30.0, 20.0))
    X, Y = np.meshgrid(np.arange(o2['image'].shape[0]) * 0.4, np.arange(o2['image'].shape[1]) * 0.4, indexing='ij')
    pts = o2['origin'] + X[..., None] * o2['e1'] + Y[..., None] * o2['e2']
    ref = fastsar.backproject(S * np.outer(taylor(P), taylor(K)), io.ecf_to_local(ant, meta), col.fmin, col.df, pts,
                              ref=np.linalg.norm(ant - srp0, axis=1), backend='cpu', window=False, upsample=16)
    check(bad, f'form_cphd ({b}, spacing 0.4 m, extent 30 x 20 m) against exact backprojection', rel_db(o2['image'], ref), -43)
    assert not bad, 'FAILED: ' + '; '.join(bad)


# ---------------------------------------------------------------------------------------------- (b) moving SRP

def test_form_cphd_moving_srp():
    print('\n(b) stripmap, SRP moving with the beam')
    bad = []
    lam0 = 0.03125
    fc, B, df = C / lam0, 300e6, 1.5e6
    K = int(round(B / df))
    f0 = fc - K // 2 * df
    dy, r0, graze = 0.25, 5e3, np.radians(30.0)
    P = 1400
    y = (np.arange(P) - P / 2) * dy
    ant_l = np.stack([np.full(P, -r0 * np.cos(graze)), y, np.full(P, r0 * np.sin(graze))], 1)
    srp_l = np.stack([np.zeros(P), y, np.zeros(P)], 1)
    tg = np.array([[-12.0, -15.3, 0.0], [3.7, 0.4, 0.0], [10.2, 16.1, 0.0]])
    amp = np.array([1.0, 0.8j, -0.6])
    f = f0 + df * np.arange(K)
    lam = C / (f0 + K / 2 * df)
    ref_p = np.linalg.norm(ant_l - srp_l, axis=1)
    d_at = np.array([0.0, 1.0, 0.0])

    def sin_look(x, a):
        w = x - a
        return (w @ d_at) / np.linalg.norm(w, axis=-1)

    s_srp = sin_look(srp_l, ant_l)
    S = np.zeros((P, K), np.complex128)
    for x, a in zip(tg, amp):
        u = (sin_look(x[None], ant_l) - s_srp) / (0.92 * lam / (4 * dy))     # pattern inside the band the PRF samples
        g = np.where(np.abs(u) < 1, np.cos(np.pi * u / 2) ** 2, 0.0)
        R = np.linalg.norm(x - ant_l, axis=1)
        S += (a * g)[:, None] * np.exp(-4j * np.pi * f[None, :] / C * (R - ref_p)[:, None])
    S = S.astype(np.complex64)
    v = 100.0
    cphd('strip.cphd', S, ecf(ant_l), ecf(srp_l), f0, df, v / dy, mode='STRIPMAP')
    out = fastsar.form_cphd('strip.cphd', backend='cpu', extent=(60.0, 50.0), patch=64)
    img, o, e1, e2, spx, spy, meta = (out[k] for k in ('image', 'origin', 'e1', 'e2', 'spx', 'spy', 'meta'))
    print(f'  form_cphd: mode {out["mode"]}, grid {img.shape} of {spx:.3f} x {spy:.3f} m; ' + '; '.join(out['notes']))
    assert out['mode'] == 'moving'
    # the reference: direct sum over pulses and frequencies, Taylor along frequency, Hann over the azimuth band
    # 0.8 lambda PRF / (2 v) about each pulse's SRP, at the pixels around each target (float64)
    dsin = 0.8 * lam * (v / dy) / (2 * v)
    wk = taylor(K)
    H = 12
    for n, x in enumerate(io.ecf_to_local(ecf(tg), meta)):
        i, j = int(round((x - o) @ e1 / spx)), int(round((x - o) @ e2 / spy))
        ii, jj = np.meshgrid(np.arange(i - H, i + H), np.arange(j - H, j + H), indexing='ij')
        q = io.local_to_ecf(o + ii[..., None] * spx * e1 + jj[..., None] * spy * e2, meta).reshape(-1, 3) - srp0
        q = np.stack([q @ N, q @ E, q @ U], 1)                                      # the simulator's frame
        acc = np.zeros(len(q), np.complex128)
        for p0 in range(0, P, 100):
            a = ant_l[p0:p0 + 100]
            w = q[None] - a[:, None]
            R = np.linalg.norm(w, axis=-1)                                          # [p, m]
            wa = sm.window('hann')(((w @ d_at) / R - s_srp[p0:p0 + 100, None]) / (dsin / 2))
            ph = np.exp(4j * np.pi * f[None, None, :] / C * (R - ref_p[p0:p0 + 100, None])[..., None])   # [p, m, K]
            acc += np.einsum('pk,pmk,pm->m', S[p0:p0 + 100] * wk[None], ph, wa)
        refp = acc.reshape(ii.shape)
        check(bad, f'target {n}: form_cphd against the direct sum ({2 * H} x {2 * H} pixels)', rel_db(img[i - H:i + H, j - H:j + H], refp), -46)
    assert not bad, 'FAILED: ' + '; '.join(bad)
