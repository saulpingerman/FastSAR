"""A simulated moving-beam collection for the tests of process_cphd on stripmap and sliding-spotlight data, read
through a stand-in for the CPHD reader (no file is written): the phase history of point targets seen through a beam
whose center, each pulse's scene reference point (SRP), moves along the ground.

Simulator frame: x north (ground range from the swath center line, the radar to the south), y east (along track),
z up, origin on the ellipsoid at the equator. The track runs along y at slant range r0 and grazing angle `graze`
from the line x = 0; `curve` bends it across track (x += curve y^2, in 1/m). The SRP of pulse p lies on x = 0 at
y_srp = y_p / slide + r0 tan(squint): slide = 1 is a stripmap, slide > 1 a sliding spotlight (the beam's footprint
moves at 1 / slide of the platform's speed), squint the look angle of the beam center from broadside. The two-way
azimuth pattern is flat over 0.88 of the Doppler band the pulse spacing samples and falls to zero at 0.98 of it,
so that the weighting of the processed band (0.8 of it by default) is the processor's alone. With `antenna` the
file carries a CPHD 1.x antenna block (per-pulse axes with the boresight on the SRP, or on the beam's own center
where `beam` gives it one, a gain polynomial of
-A (dcx^2 + dcy^2) dB one-way) and the per-pulse receive windows TOA1..TOA2 and TOAE1..TOAE2; the echoes are then
weighted by that elevation pattern too.
"""
import types

import numpy as np

from fastsar import io

C = 299792458.0
NS = types.SimpleNamespace
FILES = {}

# a local east-north-up frame at the equator (geodetic and geocentric up agree there)
LON = 30.0
SRP0 = io.geodetic_to_ecf(0.0, LON, 0.0)
E = np.array([-np.sin(np.radians(LON)), np.cos(np.radians(LON)), 0.0])
N = np.array([0.0, 0.0, 1.0])
U = SRP0 / np.linalg.norm(SRP0)


def ecf(local):
    """Simulator coordinates -> ECF."""
    local = np.asarray(local, np.float64)
    return SRP0 + local[..., :1] * N + local[..., 1:2] * E + local[..., 2:3] * U


class XYZ:
    def __init__(self, v):
        self.v = np.asarray(v, np.float64)

    def get_array(self, dtype=None):
        return self.v.copy()

    def __getitem__(self, i):
        return self.v[i]


class Reader:
    """Stands in for the CPHD reader: what io.read_cphd, form_cphd and rgiqe.Radiometry read."""

    def __init__(self, path):
        self.d = FILES[path]
        self.cphd_meta = self.d['meta']
        self.cphd_details = NS(get_cphd_bytes=lambda: b'<CPHD/>')

    def read_pvp_variable(self, name, index):
        v = self.d['pvp'].get(name)
        return None if v is None else np.array(v)

    def read_pvp_array(self, index=0):
        pv = self.d['pvp']
        out = np.zeros(len(pv['TxTime']), [(k, 'f8', np.shape(v)[1:]) for k, v in pv.items()])
        for k, v in pv.items():
            out[k] = v
        return out

    def read_chip(self, rows, cols, index=0):
        return self.d['S'][rows[0]:rows[1], cols[0]:cols[1]].copy()


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def collection(path, targets, amp=None, *, P=1400, dy=0.25, r0=5e3, graze_deg=30.0, lam0=0.03125, B=300e6, df=1.5e6,
               squint=0.0, slide=1.0, curve=0.0, phase=None, v=100.0, antenna=None, toa=None, toae=None, noise=0.0,
               seed=0, beam=None, aspect=None):
    """Register the collection under `path` and return what the tests compare with: a namespace of the antenna and
    SRP positions [P, 3] (simulator frame), the targets, lam (center wavelength), fc, K, f0, df, dsin (the processed
    spread of sin(look) at 0.8 of the sampled band), ares, rres (the unweighted -3 dB to -3 dB reciprocal bands:
    lam / (2 dsin slide) and c / (2 K df)), aperture (pulses in a pixel's band).
    targets [n, 3] (simulator frame, from the mid-aperture SRP), amp [n]. phase [P]: a phase error per pulse (rad)
    on the history.
    antenna: (A_az, A_el), the one-way gain's fall in dB per squared direction cosine along and across track; toa,
    toae: half widths (s) of the receive window and of its extension about the SRP's delay. noise: the standard
    deviation of complex white noise added to each sample. beam: the beam's own slide where it is not the SRPs'
    (its footprint moves at 1 / beam of the platform's speed while the SRPs move at 1 / slide): the azimuth
    pattern and the antenna block's boresight are then centered on it, and the SRPs only reference the echoes.
    aspect [n]: None, or (lo, hi, q) for a target that is not a point but a specular return off a curved surface,
    seen only while its sine of look from the beam center lies between lo and hi half processed spreads (dsin / 2:
    v from -1 to 1 over a pixel's band), with a sin^2 lobe over that interval, and whose echo carries q v^2 rad more
    than a point's (the specular point moves over the surface, so its range curves as that of the surface's center
    of curvature does)."""
    tg = np.atleast_2d(np.asarray(targets, np.float64)) + [0.0, r0 * np.tan(squint), 0.0]
    amp = np.ones(len(tg), np.complex128) if amp is None else np.asarray(amp, np.complex128)
    fc = C / lam0
    K = int(round(B / df))
    f0 = fc - K // 2 * df
    f = f0 + df * np.arange(K)
    lam = C / (f0 + K / 2 * df)
    g = np.radians(graze_deg)
    y = (np.arange(P) - P / 2) * dy
    ant = np.stack([-r0 * np.cos(g) + curve * y ** 2, y, np.full(P, r0 * np.sin(g))], 1)
    srp = np.stack([np.zeros(P), y / slide + r0 * np.tan(squint), np.zeros(P)], 1)
    d = unit(np.gradient(ant, axis=0))
    ref = np.linalg.norm(ant - srp, axis=1)
    s_srp = np.einsum('ij,ij->i', srp - ant, d) / ref
    full = lam / (2 * dy)                                           # the spread of sin(look) the pulse spacing samples
    bc, s_beam = srp, s_srp                                         # the beam's center: the SRP, or its own track
    if beam is not None:
        bc = np.stack([np.zeros(P), y / beam + r0 * np.tan(squint), np.zeros(P)], 1)
        s_beam = np.einsum('ij,ij->i', bc - ant, d) / np.linalg.norm(bc - ant, axis=1)
    zb = unit(bc - ant)                                             # antenna axes: boresight on the beam's center
    xa = unit(d - np.einsum('ij,ij->i', d, zb)[:, None] * zb)
    ya = np.cross(zb, xa)
    S = np.zeros((P, K), np.complex128)
    for n, (x, a) in enumerate(zip(tg, amp)):
        w = x[None] - ant
        R = np.linalg.norm(w, axis=1)
        u = np.abs(np.einsum('ij,ij->i', w, d) / R - s_beam) / (full / 2)
        pat = np.where(u <= 0.88, 1.0, np.where(u < 0.98, np.cos(np.pi / 2 * (u - 0.88) / 0.10) ** 2, 0.0))
        own = np.ones(len(ant), np.complex128)
        if aspect is not None and aspect[n] is not None:
            lo, hi, q = aspect[n]
            vb = (np.einsum('ij,ij->i', w, d) / R - s_beam) / (0.4 * full)          # dsin / 2 = 0.4 full
            pat = pat * np.where((vb > lo) & (vb < hi), np.sin(np.pi * (vb - lo) / (hi - lo)) ** 2, 0.0)
            own = np.exp(1j * q * vb ** 2)
        if antenna is not None:                                     # two-way, relative to the SRP's
            dcy = np.einsum('ij,ij->i', w, ya) / R
            pat = pat * 10.0 ** (-0.1 * antenna[1] * dcy ** 2) * (ref / R) ** 2
        on = pat > 0
        S[on] += (a * pat[on] * own[on])[:, None] * np.exp(-4j * np.pi * f[None, :] / C * (R[on] - ref[on])[:, None])
    if phase is not None:
        S *= np.exp(1j * np.asarray(phase, np.float64))[:, None]
    if noise:
        rng = np.random.default_rng(seed)
        S += noise * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)) / np.sqrt(2)
    prf = v / dy
    t = np.arange(P) / prf
    pvp = dict(TxPos=ecf(ant), RcvPos=ecf(ant), SRPPos=ecf(srp), SC0=np.full(P, f0), SCSS=np.full(P, df), TxTime=t,
               RcvTime=t + 2 * ref / C, FX1=np.full(P, f0), FX2=np.full(P, f0 + K * df))
    par = NS(Identifier='VV', FxC=fc, Polarization=NS(TxPol='V', RcvPol='V'), TgtRefLevel=None, NoiseLevel=None,
             Antenna=None)
    meta = NS(Data=NS(Channels=[NS(NumVectors=P, NumSamples=K, Identifier='VV')]), Channel=NS(Parameters=[par]),
              Global=NS(DomainType='FX', SGN=-1, Timeline=NS(CollectionStart='2026-01-01T00:00:00.000000Z')),
              CollectionID=NS(CollectorName='SIMULATED', CoreName='SIMULATED', RadarMode=NS(ModeType='STRIPMAP')),
              SceneCoordinates=NS(IARP=NS(ECF=XYZ(ecf(srp[P // 2]))), ImageArea=None, ReferenceSurface=None),
              Antenna=None, SupportArray=None)
    if toa is not None:
        pvp.update(TOA1=np.full(P, -toa), TOA2=np.full(P, toa))
        if toae is not None:
            pvp.update(TOAE1=np.full(P, -toae), TOAE2=np.full(P, toae))
    if antenna is not None:
        rot = np.stack([N, E, U], 1)                                # simulator axes -> ECF
        zero = np.zeros((P, 2))
        pvp.update(TxACX=xa @ rot.T, TxACY=ya @ rot.T, RcvACX=xa @ rot.T, RcvACY=ya @ rot.T, TxEB=zero, RcvEB=zero)
        par.Antenna = NS(TxAPCId='APC', TxAPATId='PAT', RcvAPCId='APC', RcvAPATId='PAT')
        gain = XYZ([[0.0, 0.0, -antenna[1]], [0.0, 0.0, 0.0], [-antenna[0], 0.0, 0.0]])
        pat = NS(Identifier='PAT', EB=None, Array=NS(GainPoly=gain, AntGPId=None),
                 Element=NS(GainPoly=XYZ([[0.0]]), AntGPId=None))
        meta.Antenna = NS(AntCoordFrame=[NS(Identifier='ACF', XAxisPoly=None, YAxisPoly=None)],
                          AntPhaseCenter=[NS(Identifier='APC', ACFId='ACF')], AntPattern=[pat])
    FILES[path] = dict(meta=meta, pvp=pvp, S=S.astype(np.complex64))
    dsin = 0.8 * lam * prf / (2 * v)
    return NS(ant=ant, srp=srp, tg=tg, amp=amp, lam=lam, fc=f0 + K / 2 * df, K=K, f0=f0, df=df, P=P, dy=dy, r0=r0,
              graze=g, dsin=dsin, ares=lam / (2 * dsin * slide), rres=C / (2 * K * df), ya=ya, ref=ref, prf=prf, v=v,
              aperture=r0 * dsin * slide / dy, s_srp=s_srp, d=d, s_beam=s_beam,
              beam_ares=lam / (2 * dsin * (slide if beam is None else beam)))


def to_pixels(out, meta, pts):
    """Fractional pixel (i, j) of simulator-frame points [n, 3] in a product."""
    q = io.ecf_to_local(ecf(np.atleast_2d(pts)), meta) - out['origin']
    return np.stack([q @ out['e1'] / out['spx'], q @ out['e2'] / out['spy']], 1)
