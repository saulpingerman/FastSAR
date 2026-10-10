"""Synthetic spotlight SAR collects in float64: geometry, scenes, phase history.

The phase history follows the CPHD convention of motion compensation to the
scene center:  S[p, k] = sum_n sigma_n exp(-j 4 pi f_k dR_pn / c)  with
dR_pn = |x_n - a_p| - |a_p|.  It is synthesised with a type-1 NUFFT per pulse
(finufft), which is exact to the requested tolerance and makes million-scatterer
clutter scenes cheap.
"""
from dataclasses import dataclass

import numpy as np
from . import _deps

C = 299792458.0


@dataclass
class Collect:
    fmin: float          # Hz, frequency of sample k=0
    df: float            # Hz, frequency step
    K: int               # frequency samples per pulse
    ant: np.ndarray      # [Np, 3] antenna phase center, scene-centered meters
    res: float           # nominal slant-plane resolution, meters

    @property
    def Np(self):
        return self.ant.shape[0]

    @property
    def fref(self):
        """Frequency of sample K//2, used as the carrier after range compression."""
        return self.fmin + (self.K // 2) * self.df

    @property
    def freqs(self):
        return self.fmin + self.df * np.arange(self.K)


def _even(n):
    n = int(np.ceil(n))
    return n + (n % 2)


def make_collect(res=0.3, scene=150.0, r0=10e3, graze_deg=30.0, fc=9.6e9,
                 margin=1.25, offset=(0.0, 0.0, 0.0)):
    """Broadside linear-track spotlight collect looking along +x.

    `scene` is the side of the square ground patch that must stay unaliased.
    `offset` shifts the whole track (repeat-pass baseline).
    """
    lam = C / fc
    bw = C / (2.0 * res)
    extent = margin * np.sqrt(2.0) * scene
    df = C / (2.0 * extent)
    K = _even(bw / df)
    psi = np.deg2rad(graze_deg)
    L = r0 * lam / (2.0 * res)                 # aperture length
    dy = lam * r0 / (2.0 * extent)             # pulse spacing along track
    Np = _even(L / dy)
    y = (np.arange(Np) - (Np - 1) / 2.0) * dy
    ant = np.stack([np.full(Np, -r0 * np.cos(psi)), y,
                    np.full(Np, r0 * np.sin(psi))], axis=1)
    ant = ant + np.asarray(offset, dtype=np.float64)[None, :]
    return Collect(fmin=fc - (K // 2) * df, df=df, K=K, ant=ant, res=res)


def delta_range(ant, pos):
    """dR[p, n] in float64 for antenna rows `ant` and scatterer rows `pos`."""
    d = pos[None, :, :] - ant[:, None, :]
    return np.sqrt(np.einsum('pnk,pnk->pn', d, d)) - np.linalg.norm(ant, axis=1)[:, None]


def simulate(col, pos, amp, eps=1e-12, nthreads=0):
    """Phase history [Np, K] complex128 via one type-1 NUFFT per pulse."""
    import finufft
    out = np.empty((col.Np, col.K), np.complex128)
    kw = dict(eps=eps, isign=-1)
    if nthreads:
        kw['nthreads'] = nthreads
    r0 = np.linalg.norm(col.ant, axis=1)
    for p in range(col.Np):
        d = pos - col.ant[p]
        dr = np.sqrt(np.einsum('nk,nk->n', d, d)) - r0[p]
        x = (4.0 * np.pi * col.df / C) * dr
        c = amp * np.exp(-1j * (4.0 * np.pi * col.fref / C) * dr)
        out[p] = finufft.nufft1d1(x, c, col.K, **kw)
    return out


def simulate_brute(col, pos, amp):
    """Direct O(Np K N) sum, for validating `simulate` on small cases; in blocks of scatterers, so that the
    temporaries stay near 64 MB (one [Np, K, N] array is 5.8 GB at 2122 x 2122 x 80)."""
    dr = delta_range(col.ant, np.asarray(pos, np.float64))
    amp = np.asarray(amp)
    out = np.zeros((col.Np, col.K), np.complex128)
    nb = max(1, (1 << 22) // (col.Np * col.K))
    for n0 in range(0, dr.shape[1], nb):
        ph = -4.0 * np.pi / C * col.freqs[None, :, None] * dr[:, None, n0:n0 + nb]
        out += (amp[None, None, n0:n0 + nb] * np.exp(1j * ph)).sum(-1)
    return out


def add_noise(S, sigma, rng):
    n = rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)
    return S + (sigma / np.sqrt(2.0)) * n



def enu_axes(lat, lon):
    """East, north and up unit vectors (rows, ECF) of the ellipsoid at a geodetic latitude and longitude (degrees)."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0.0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def to_ecf(points, lat, lon, height=0.0, heading=0.0):
    """Points [..., 3] of the simulator's frame (x across track toward the scene, y along track, z up, origin at the
    scene center) -> ECF, for a scene center at (lat, lon, height) and a track heading in degrees clockwise from
    north. The frame's ground plane is the ellipsoid's tangent plane there; the radar looks right."""
    from .io import geodetic_to_ecf
    h = np.radians(heading)
    M = np.array([[np.cos(h), np.sin(h), 0.0], [-np.sin(h), np.cos(h), 0.0], [0.0, 0.0, 1.0]])   # columns: x, y in ENU
    return np.asarray(points, np.float64) @ M.T @ enu_axes(lat, lon) + geodetic_to_ecf(lat, lon, height)


def _reference_geometry(arp, varp, srp):
    """CPHD ReferenceGeometry/Monostatic angles (CPHD 1.0.1, section 6.5.2) of an aperture position and velocity
    seen from a scene reference point (ECF)."""
    from .io import ecf_to_geodetic
    E, N, U = enu_axes(*(float(np.ravel(v)[0]) for v in ecf_to_geodetic(srp)[:2]))
    r = np.linalg.norm(arp - srp)
    ua, uv = (arp - srp) / r, varp / np.linalg.norm(varp)
    look = 1 if np.cross(arp / np.linalg.norm(arp), uv) @ ua < 0 else -1
    gpy = np.cross(U, ua)
    gpy /= np.linalg.norm(gpy)
    gpx = np.cross(gpy, U)
    spn = look * np.cross(ua, uv)
    spn /= np.linalg.norm(spn)
    graze = np.degrees(np.arccos(ua @ gpx))
    deg = lambda y, x: np.degrees(np.arctan2(y, x)) % 360
    return dict(SideOfTrack='L' if look == 1 else 'R', SlantRange=r,
                GroundRange=np.linalg.norm(srp) * np.arccos(arp @ srp / np.linalg.norm(arp) / np.linalg.norm(srp)),
                DopplerConeAngle=np.degrees(np.arccos(-ua @ uv)), GrazeAngle=graze, IncidenceAngle=90 - graze,
                AzimuthAngle=deg(gpx @ E, gpx @ N), TwistAngle=-np.degrees(np.arcsin(spn @ gpy)),
                SlopeAngle=np.degrees(np.arccos(U @ spn)), LayoverAngle=deg(-spn @ E, -spn @ N))


def write_cphd(path, col, S, lat, lon, height=0.0, heading=0.0, speed=200.0, srp=None, scene=None, pol='VV',
               start='2026-01-01T00:00:00.000000Z', tropo=0.0):
    """Write a simulated phase history S [Np, K] (this module's convention: compensated to the scene center, phase
    exp(-j 4 pi f dR / c)) as a monostatic frequency-domain CPHD 1.0.1 file, placed by to_ecf, with the pulses
    `speed` m/s apart in time along the track. srp: per-pulse scene reference points [Np, 3] in the simulator's
    frame for a moving beam (S compensated to them), default the scene center. scene: side (m) of the square image
    area, default the extent the collection's frequency step leaves unambiguous. tropo: the two-way troposphere
    delay (s, one value or one per pulse) written as TDTropoSRP; S should already hold it (exp(-j 2 pi f tropo)).
    Needs sarpy."""
    _deps.require('sarpy')
    from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType
    from sarpy.io.phase_history.cphd import CPHDWriter1
    from .io import ecf_to_geodetic
    P, K = S.shape
    ant = np.asarray(col.ant, np.float64)
    srp = np.zeros_like(ant) if srp is None else np.asarray(srp, np.float64)
    tx, sr = to_ecf(ant, lat, lon, height, heading), to_ecf(srp, lat, lon, height, heading)
    s0 = to_ecf(np.zeros(3), lat, lon, height, heading)
    t = 1.0 + (np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(ant, axis=0), axis=1)]) / speed)
    vel = np.gradient(tx, t, axis=0)
    rng = np.linalg.norm(tx - sr, axis=1)
    half = 0.5 * (C / (2 * col.df) / np.sqrt(2) if scene is None else scene)
    E, N, U = enu_axes(lat, lon)
    h = np.radians(heading)
    ux, uy = np.cos(h) * E - np.sin(h) * N, np.sin(h) * E + np.cos(h) * N      # across and along track
    corners = [s0 + a * ux + b * uy for a, b in ((-half, -half), (-half, half), (half, half), (half, -half))]
    ll = lambda x: tuple(float(np.ravel(v)[0]) for v in ecf_to_geodetic(x))
    g = lambda v: repr(float(v))
    xyz = lambda tag, v: f'<{tag}><X>{g(v[0])}</X><Y>{g(v[1])}</Y><Z>{g(v[2])}</Z></{tag}>'
    m = P // 2
    geom = _reference_geometry(tx[m], vel[m], sr[m])
    names = ['TxTime', 'TxPos', 'TxVel', 'RcvTime', 'RcvPos', 'RcvVel', 'SRPPos', 'aFDOP', 'aFRR1', 'aFRR2', 'FX1',
             'FX2', 'TOA1', 'TOA2', 'TDTropoSRP', 'SC0', 'SCSS']
    pvp, off = [], 0
    for n in names:
        size = 3 if n.endswith(('Pos', 'Vel')) else 1
        pvp.append(f'<{n}><Offset>{off}</Offset><Size>{size}</Size><Format>{"X=F8;Y=F8;Z=F8;" if size == 3 else "F8"}</Format></{n}>')
        off += size
    fmin, fmax = col.fmin, col.fmin + (K - 1) * col.df
    poly = lambda tag, v: f'<{tag} order1="0" order2="0"><Coef exponent1="0" exponent2="0">{g(v)}</Coef></{tag}>'
    xml = f"""<CPHD xmlns="http://api.nsgreg.nga.mil/schema/cphd/1.0.1">
<CollectionID><CollectorName>FastSAR simulator</CollectorName><CoreName>SIMULATED</CoreName><CollectType>MONOSTATIC</CollectType>
<RadarMode><ModeType>{'SPOTLIGHT' if not np.any(srp) else 'STRIPMAP'}</ModeType></RadarMode><Classification>UNCLASSIFIED</Classification><ReleaseInfo>UNRESTRICTED</ReleaseInfo></CollectionID>
<Global><DomainType>FX</DomainType><SGN>-1</SGN><Timeline><CollectionStart>{start}</CollectionStart><TxTime1>{g(t[0])}</TxTime1><TxTime2>{g(t[-1])}</TxTime2></Timeline>
<FxBand><FxMin>{g(fmin)}</FxMin><FxMax>{g(fmax)}</FxMax></FxBand><TOASwath><TOAMin>{g(-0.45 / col.df)}</TOAMin><TOAMax>{g(0.45 / col.df)}</TOAMax></TOASwath></Global>
<SceneCoordinates><EarthModel>WGS_84</EarthModel><IARP>{xyz('ECF', s0)}<LLH><Lat>{g(lat)}</Lat><Lon>{g(lon)}</Lon><HAE>{g(height)}</HAE></LLH></IARP>
<ReferenceSurface><Planar>{xyz('uIAX', ux)}{xyz('uIAY', uy)}</Planar></ReferenceSurface>
<ImageArea><X1Y1><X>{g(-half)}</X><Y>{g(-half)}</Y></X1Y1><X2Y2><X>{g(half)}</X><Y>{g(half)}</Y></X2Y2></ImageArea>
<ImageAreaCornerPoints>{''.join(f'<IACP index="{k + 1}"><Lat>{g(ll(c)[0])}</Lat><Lon>{g(ll(c)[1])}</Lon></IACP>' for k, c in enumerate(corners))}</ImageAreaCornerPoints></SceneCoordinates>
<Data><SignalArrayFormat>CF8</SignalArrayFormat><NumBytesPVP>{8 * off}</NumBytesPVP><NumCPHDChannels>1</NumCPHDChannels>
<Channel><Identifier>{pol}</Identifier><NumVectors>{P}</NumVectors><NumSamples>{K}</NumSamples><SignalArrayByteOffset>0</SignalArrayByteOffset><PVPArrayByteOffset>0</PVPArrayByteOffset></Channel><NumSupportArrays>0</NumSupportArrays></Data>
<Channel><RefChId>{pol}</RefChId><FXFixedCPHD>true</FXFixedCPHD><TOAFixedCPHD>true</TOAFixedCPHD><SRPFixedCPHD>{'false' if np.any(srp) else 'true'}</SRPFixedCPHD>
<Parameters><Identifier>{pol}</Identifier><RefVectorIndex>{m}</RefVectorIndex><FXFixed>true</FXFixed><TOAFixed>true</TOAFixed><SRPFixed>{'false' if np.any(srp) else 'true'}</SRPFixed>
<Polarization><TxPol>{pol[0]}</TxPol><RcvPol>{pol[1]}</RcvPol></Polarization><FxC>{g(0.5 * (fmin + fmax))}</FxC><FxBW>{g(fmax - fmin)}</FxBW><TOASaved>{g(0.9 / col.df)}</TOASaved>
<DwellTimes><CODId>COD</CODId><DwellId>DWELL</DwellId></DwellTimes></Parameters></Channel>
<PVP>{''.join(pvp)}</PVP>
<Dwell><NumCODTimes>1</NumCODTimes><CODTime><Identifier>COD</Identifier>{poly('CODTimePoly', 0.5 * (t[0] + t[-1]))}</CODTime>
<NumDwellTimes>1</NumDwellTimes><DwellTime><Identifier>DWELL</Identifier>{poly('DwellTimePoly', t[-1] - t[0])}</DwellTime></Dwell>
<ReferenceGeometry><SRP>{xyz('ECF', sr[m])}{xyz('IAC', (sr[m] - s0) @ np.stack([ux, uy, U]).T)}</SRP><ReferenceTime>{g(t[m] + rng[m] / C)}</ReferenceTime>
<SRPCODTime>{g(0.5 * (t[0] + t[-1]))}</SRPCODTime><SRPDwellTime>{g(t[-1] - t[0])}</SRPDwellTime><Monostatic>{xyz('ARPPos', tx[m])}{xyz('ARPVel', vel[m])}
<SideOfTrack>{geom.pop('SideOfTrack')}</SideOfTrack>{''.join(f'<{k}>{g(v)}</{k}>' for k, v in geom.items())}</Monostatic></ReferenceGeometry>
</CPHD>"""
    meta = CPHDType.from_xml_string(xml)
    v = np.zeros(P, meta.PVP.get_vector_dtype())
    v['TxTime'], v['TxPos'], v['TxVel'] = t, tx, vel
    v['RcvTime'], v['RcvPos'], v['RcvVel'] = t + 2 * rng / C, tx, vel
    v['SRPPos'], v['FX1'], v['FX2'], v['SC0'], v['SCSS'] = sr, fmin, fmax, fmin, col.df
    v['TOA1'], v['TOA2'] = -0.45 / col.df, 0.45 / col.df
    v['TDTropoSRP'] = tropo
    with CPHDWriter1(path, meta, check_existence=False) as w:
        w.write_file({pol: v}, {pol: np.asarray(S, np.complex64)})



def raw_echoes(col, pos, amp, fs, chirp, r_near, n, lat, lon, height=0.0, heading=0.0, vel=None):
    """Baseband raw echo lines [Np, n] (complex64) of scatterers pos [N, 3] (simulator frame) with amplitudes amp
    [N], for a collection placed by to_ecf: the receive window starts at the slant range r_near (m) and holds n
    samples at the rate fs (Hz); each echo is the chirp [L] (baseband replica at fs) delayed by the two-way time
    and carrying the carrier phase exp(-j 2 pi fc tau) with fc the collection's center frequency. With vel [Np, 3]
    (ECF, m/s) the receiver has moved on by the echo's travel time, as a satellite does. The sum is exact (direct
    O(Np N L))."""
    ant = to_ecf(np.asarray(col.ant, np.float64), lat, lon, height, heading)
    pts = to_ecf(np.asarray(pos, np.float64), lat, lon, height, heading)
    fc = col.fmin + (col.K // 2) * col.df
    chirp = np.asarray(chirp, np.complex64)
    L = len(chirp)
    out = np.zeros((len(ant), n), np.complex64)
    tau0 = 2.0 * r_near / C
    spec = np.fft.fft(chirp, 4 * L)                        # the replica, shifted by fractions of a sample through its spectrum
    fq = np.fft.fftfreq(4 * L)
    for i, a in enumerate(ant):
        tau = 2.0 * np.linalg.norm(pts - a, axis=1) / C
        if vel is not None:                     # the receiver at the echo's arrival: one refinement of the delay
            tau = (np.linalg.norm(pts - a, axis=1) + np.linalg.norm(pts - (a[None] + vel[i][None] * tau[:, None]), axis=1)) / C
        line = np.zeros(n, np.complex128)
        for d, A in zip(tau, np.asarray(amp)):
            m = (d - tau0) * fs                            # fractional start sample of this echo
            m0 = int(np.floor(m))
            frac = m - m0
            shifted = np.fft.ifft(spec * np.exp(-2j * np.pi * fq * frac))[:L]
            lo, hi = max(0, m0), min(n, m0 + L)
            if hi > lo:
                line[lo:hi] += A * np.exp(-2j * np.pi * fc * d) * shifted[lo - m0:hi - m0]
        out[i] = line
    return out


def write_nisar(path, col, pos, amp, lat, lon, height=0.0, heading=0.0, speed=7500.0, bandwidth=None, fs=None,
                chirp_duration=10e-6, r_margin=200.0, pol='HH', frequency='A', epoch='2026-01-01T00:00:00', range_delay=None,
                dither=0.0):
    """Write a NISAR L0B RRSD look-alike (HDF5) of scatterers pos [N, 3] (simulator frame) with amplitudes amp:
    the raw echoes of a linear-FM chirp (bandwidth default the collection's, sampled at fs, default 1.2 times the
    bandwidth) in a receive window from r_margin m before the nearest scatterer to r_margin m past the farthest,
    block-floating-point encoded; the pulse times speed m/s apart along the track; the orbit as nine state vectors
    along the (straight) track; the datasets and attributes io.read_nisar reads. The radar looks right (to_ecf).
    range_delay: the instrument's range delay (m, default io.NISAR_RANGE_DELAY for the frequency): the echoes
    arrive late by it, so each sample's slant range labels a target that far nearer. dither: the pulse timing varied
    at random by up to this fraction of the interval (NISAR dithers its PRI), the antenna positions following the
    times along the track. Needs h5py."""
    _deps.require('h5py')
    import h5py
    ant_l = np.asarray(col.ant, np.float64)
    ant = to_ecf(ant_l, lat, lon, height, heading)
    pts = to_ecf(np.asarray(pos, np.float64), lat, lon, height, heading)
    P = len(ant)
    bw = float(col.K * col.df if bandwidth is None else bandwidth)
    fs = float(1.2 * bw if fs is None else fs)
    fc = col.fmin + (col.K // 2) * col.df
    L = int(round(chirp_duration * fs))
    u = (np.arange(L) - 0.5 * (L - 1)) / fs
    slope = bw / chirp_duration
    chirp = np.exp(1j * np.pi * slope * u ** 2).astype(np.complex64)
    from .io import NISAR_RANGE_DELAY
    delay = float(NISAR_RANGE_DELAY.get(frequency, 0.0) if range_delay is None else range_delay)
    r = np.linalg.norm(pts[None, :, :] - ant[:, None, :], axis=2)
    r_near = r.min() - r_margin + delay              # the labeled slant range of sample 0 (the echoes arrive late)
    n = int(np.ceil(2.0 * (r.max() + r_margin - (r_near - delay)) / C * fs)) + L     # the physical window
    t = 1.0 + np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(ant_l, axis=0), axis=1)]) / speed
    if dither:
        rng = np.random.default_rng(7)
        dt = float(np.median(np.diff(t)))
        tj = t + rng.uniform(-dither, dither, len(t)) * dt           # the pulses fire at jittered times
        tj[0], tj[-1] = t[0], t[-1]
        from scipy.interpolate import CubicSpline
        ant = CubicSpline(t, ant, axis=0)(tj)                         # on the same straight track, where the platform is then
        ant_l = CubicSpline(t, ant_l, axis=0)(tj)
        t = tj
        col = Collect(col.fmin, col.df, col.K, ant_l, col.res)
    vel = np.gradient(ant, t, axis=0)
    z = raw_echoes(col, pos, amp, fs, chirp, r_near - delay, n, lat, lon, height, heading, vel=vel)
    # nine state vectors on the straight track, one second apart around the aperture
    ts = np.linspace(t[0] - 4.0, t[-1] + 4.0, 9)
    v0 = vel[P // 2]
    sv_pos = ant[P // 2][None] + (ts - t[P // 2])[:, None] * v0[None]
    sv_vel = np.repeat(v0[None], 9, 0)
    scale = 32000.0 / max(np.abs(z.real).max(), np.abs(z.imag).max(), 1e-30)
    lut = ((np.arange(65536) - 32768) / scale).astype(np.float32)
    enc = np.empty(z.shape, dtype=[('r', '<u2'), ('i', '<u2')])
    enc['r'] = np.clip(np.round(z.real * scale) + 32768, 0, 65535).astype(np.uint16)
    enc['i'] = np.clip(np.round(z.imag * scale) + 32768, 0, 65535).astype(np.uint16)
    with h5py.File(path, 'w') as f:
        ident = f.create_group('science/LSAR/identification')
        ident['listOfFrequencies'] = np.array([frequency.encode()], dtype='S1')
        ident['lookDirection'] = np.bytes_(b'Right')
        ident['granuleId'] = np.bytes_(b'SIMULATED')
        ident['isDithered'] = np.bytes_(b'False')
        g = f.create_group(f'science/LSAR/RRSD/swaths/frequency{frequency}/tx{pol[0]}')
        g['listOfRxPolarizations'] = np.array([pol[1].encode()], dtype='S1')
        g['chirpWaveform'] = chirp
        g['chirpDuration'] = chirp_duration
        g['chirpSlope'] = slope
        g['centerFrequency'] = fc
        g['rangeBandwidth'] = bw
        g['slantRangeSpacing'] = C / (2.0 * fs)
        g['slantRange'] = r_near + np.arange(n) * C / (2.0 * fs)
        d = g.create_dataset('UTCtime', data=t)
        d.attrs['units'] = f'seconds since {epoch}'
        g['numberOfSubSwaths'] = np.uint8(1)
        g['validSamplesSubSwath1'] = np.tile(np.array([[0, n]], np.uint32), (P, 1))
        h = g.create_group(f'rx{pol[1]}')
        h['BFPQLUT'] = lut
        h.create_dataset(pol, data=enc)
        h['basebandPhaseCorrection'] = np.ones(P, np.complex64)
        orb = f.create_group('science/LSAR/RRSD/lowRateTelemetry/orbit')
        d = orb.create_dataset('time', data=ts)
        d.attrs['units'] = f'seconds since {epoch}'
        orb['position'] = sv_pos
        orb['velocity'] = sv_vel
        orb['orbitType'] = np.bytes_(b'POE')
        orb['interpMethod'] = np.bytes_(b'Hermite')
    return dict(fs=fs, chirp=chirp, r_near=r_near, n=n, tx=ant, times=t)



def write_palsar(directory, col, pos, amp, lat, lon, height=0.0, heading=0.0, speed=7500.0, bandwidth=None, fs=None,
                 chirp_duration=27e-6, r_margin=200.0, pol='HH', scene_id='SIMULATED'):
    """Write an ALOS PALSAR level 1.0 look-alike (CEOS: LED- leader and IMG- image files in the directory) of
    scatterers pos [N, 3] (simulator frame): the raw echoes of a linear FM chirp (bandwidth default the
    collection's; sampled at fs, default 1.2 times the bandwidth; negative chirp rate as PALSAR's), 8-bit I and Q
    with a DC bias of 16; the per-line prefix fields (time, PRF, slant range to the first sample) and the leader's
    data set summary and platform position records that io.read_palsar reads. The radar looks right (to_ecf)."""
    import datetime
    import os
    ant_l = np.asarray(col.ant, np.float64)
    ant = to_ecf(ant_l, lat, lon, height, heading)
    pts = to_ecf(np.asarray(pos, np.float64), lat, lon, height, heading)
    P = len(ant)
    bw = float(col.K * col.df if bandwidth is None else bandwidth)
    fs = float(1.2 * bw if fs is None else fs)
    fc = col.fmin + (col.K // 2) * col.df
    L = int(round(chirp_duration * fs))
    u = (np.arange(L) - 0.5 * (L - 1)) / fs
    k = -bw / chirp_duration
    chirp = np.exp(1j * np.pi * k * u ** 2).astype(np.complex64)
    r = np.linalg.norm(pts[None, :, :] - ant[:, None, :], axis=2)
    r_near = r.min() - r_margin
    n = int(np.ceil(2.0 * (r.max() + r_margin - r_near) / C * fs)) + L
    t = 1.0 + np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(ant_l, axis=0), axis=1)]) / speed
    vel = np.gradient(ant, t, axis=0)
    z = raw_echoes(col, pos, amp, fs, chirp, r_near, n, lat, lon, height, heading, vel=vel)
    scale = 15.0 / max(np.abs(z.real).max(), np.abs(z.imag).max(), 1e-30)
    iq = np.empty((P, n, 2), np.uint8)
    iq[:, :, 0] = np.clip(np.round(z.real * scale) + 16, 0, 255)
    iq[:, :, 1] = np.clip(np.round(z.imag * scale) + 16, 0, 255)
    prf = 1.0 / np.median(np.diff(t))
    day = datetime.date(2026, 1, 1)

    def header(seq, codes, length):
        return seq.to_bytes(4, 'big') + bytes(codes) + length.to_bytes(4, 'big')

    def put(buf, a, s):
        s = s.encode('ascii') if isinstance(s, str) else s
        buf[a - 1:a - 1 + len(s)] = s

    def num(v, w, prec=7):
        return f'{v:{w}.{prec}f}'[:w].rjust(w)

    def sci(v, w=22):
        return f'{v:.15E}'.rjust(w)

    # leader: file descriptor (720), data set summary (4096), platform position (4680)
    fd = bytearray(720); fd[:12] = header(1, (11, 192, 18, 18), 720); put(fd, 17, 'CEOS-SAR-CCT')
    ds = bytearray(4096); ds[:12] = header(2, (18, 10, 18, 20), 4096)
    put(ds, 13, '   1'); put(ds, 21, scene_id.ljust(32)); put(ds, 117, num(lat, 16)); put(ds, 133, num(lon, 16))
    put(ds, 149, num(heading, 16)); put(ds, 309, num(height / 1e3, 16)); put(ds, 397, 'ALOS'.ljust(16))
    put(ds, 501, num(C / fc, 16)); put(ds, 519, 'LINEAR FM CHIRP '); put(ds, 535, f'{0.0:16.7E}'); put(ds, 551, f'{abs(k):16.7E}')     # JAXA writes the magnitude
    put(ds, 711, num(fs / 1e6, 16)); put(ds, 727, num(2 * r_near / C * 1e6, 16)); put(ds, 743, num(chirp_duration * 1e6, 16))
    put(ds, 759, 'YES '); put(ds, 763, 'NOT '); put(ds, 799, '       5'); put(ds, 819, num(16.0, 16)); put(ds, 835, num(16.0, 16))
    put(ds, 935, num(prf * 1e3, 16)); put(ds, 1095, '1.0'.ljust(16))
    pp = bytearray(4680); pp[:12] = header(3, (18, 30, 18, 20), 4680)
    ts = np.linspace(t[0] - 4.0, t[-1] + 4.0, 9)
    v0 = vel[P // 2]
    sv_pos = ant[P // 2][None] + (ts - t[P // 2])[:, None] * v0[None]
    put(pp, 141, f'{9:4d}'); put(pp, 145, f'{day.year:4d}'); put(pp, 149, f'{day.month:4d}'); put(pp, 153, f'{day.day:4d}')
    put(pp, 157, f'{day.timetuple().tm_yday:4d}'); put(pp, 161, sci(ts[0])); put(pp, 183, sci(ts[1] - ts[0])); put(pp, 205, 'ECR'.ljust(64))
    for j in range(9):
        for c in range(3):
            put(pp, 387 + 22 * (6 * j + c), sci(sv_pos[j, c]))
            put(pp, 387 + 22 * (6 * j + 3 + c), sci(v0[c]))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f'LED-{scene_id}'), 'wb') as fh:
        fh.write(bytes(fd) + bytes(ds) + bytes(pp))
    # image file: descriptor (720) then one record per line, 412 bytes of prefix and 2 n bytes of samples
    rl = 412 + 2 * n
    idf = bytearray(720); idf[:12] = header(1, (50, 192, 18, 18), 720); put(idf, 17, 'CEOS-SAR-CCT')
    put(idf, 181, f'{P:6d}'); put(idf, 217, f'{8:4d}'); put(idf, 221, f'{2:4d}'); put(idf, 225, f'{2:4d}'); put(idf, 237, f'{P:8d}')
    put(idf, 249, f'{n:8d}'); put(idf, 273, ' 1'); put(idf, 275, ' 1'); put(idf, 277, f'{412:4d}'); put(idf, 281, f'{2 * n:8d}')
    put(idf, 289, f'{0:4d}')
    recs = np.zeros((P, rl), np.uint8)
    secs = t + 2.0 * r_near / C                     # the lines' times are those of their windows (seconds of the day, the
                                                    # scale of the platform position record), one window delay after the pulse
    for i in range(P):
        pre = bytearray(412)
        pre[:12] = header(i + 2, (50, 10, 18, 20), rl)
        for a, v in ((13, i + 1), (17, 1), (25, n), (33, 1), (37, day.year), (41, day.timetuple().tm_yday),
                     (45, int(round(secs[i] * 1e3))), (49, 1 << 16), (57, int(round(prf * 1e3))), (69, int(round(chirp_duration * 1e9))),
                     (117, int(round(r_near))), (121, int(round(2 * r_near / C * 1e9)))):
            pre[a - 1:a + 3] = int(v).to_bytes(4, 'big')
        pre[49:51] = (1).to_bytes(2, 'big')          # SAR channel indicator
        pre[296:300] = (int(round((secs[i] - np.floor(secs[i])) * 1e6)) << 8).to_bytes(4, 'big')      # 1Mpps counter (aux item 2)
        recs[i, :412] = np.frombuffer(bytes(pre), np.uint8)
        recs[i, 412:] = iq[i].reshape(-1)
    with open(os.path.join(directory, f'IMG-{pol}-{scene_id}'), 'wb') as fh:
        fh.write(bytes(idf)); fh.write(recs.tobytes())
    return dict(fs=fs, chirp=chirp, r_near=r_near, n=n, tx=ant, times=t, sv_time=ts)



def write_sentinel1(path, col, pos, amp, lat, lon, height=0.0, heading=0.0, speed=7500.0, bandwidth=None, range_decimation=0,
                    chirp_duration=20e-6, r_margin=200.0, rank=None, pol='HH', swath=1, gps_start=1.4e9, internal_delay=433e-9,
                    cal_packets=4):
    """Write a Sentinel-1 Level-0 measurement file look-alike of scatterers pos [N, 3] (simulator frame): one
    space packet per pulse with the primary and secondary headers io.read_sentinel1 reads (datation, radar
    configuration with the chirp encoded as start frequency, ramp rate and length, rank, PRI, SWST, SWL, SES
    message with the echo signal type, number of quads) and the echoes as 10-bit decimation-only user data (BAQ mode
    0); the position and velocity records sub-commutated over the first 22 packets of every 64. The sampling rate
    follows the range decimation code (default 0: 112.6 MHz) and the chirp bandwidth is the collection's unless
    given; internal_delay (s) the instrument's internal delay, which delays the echoes and positions the chirp in the
    cal_packets calibration packets of each type written ahead of the echoes (io.read_sentinel1 estimates it from
    them); the data of each packet are the echo of the pulse sent `rank` intervals earlier (default: as many as the
    near range spans). The radar looks right."""
    from . import sentinel1 as s1
    ant_l = np.asarray(col.ant, np.float64)
    ant = to_ecf(ant_l, lat, lon, height, heading)
    pts = to_ecf(np.asarray(pos, np.float64), lat, lon, height, heading)
    P = len(ant)
    fs = s1.sampling_frequency(range_decimation)
    bw = float(col.K * col.df if bandwidth is None else bandwidth)
    if bw > 0.9 * fs:
        raise ValueError(f'bandwidth {bw / 1e6:.1f} MHz exceeds the sampling rate of decimation code {range_decimation} ({fs / 1e6:.1f} MHz)')
    # the chirp as the headers encode it: ramp rate and start frequency quantized like the instrument's fields
    prr_code = int(round(bw / chirp_duration * 2 ** 21 / s1.F_REF ** 2))
    txprr = prr_code * s1.F_REF ** 2 / 2 ** 21
    psf_code = int(round((bw / 2 - txprr / (4 * s1.F_REF)) * 2 ** 14 / s1.F_REF))
    txpsf = txprr / (4 * s1.F_REF) - psf_code * s1.F_REF / 2 ** 14
    pl_code = int(round(chirp_duration * s1.F_REF))
    txpl = pl_code / s1.F_REF
    chirp = s1.chirp(txpsf, txprr, txpl, fs)
    L = len(chirp)
    t = gps_start + np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(ant_l, axis=0), axis=1)]) / speed
    pri = float(np.median(np.diff(t)))
    pri_code = int(round(pri * s1.F_REF)); pri = pri_code / s1.F_REF
    t = gps_start + np.arange(P) * pri                          # the instrument's own uniform timing
    vel = np.gradient(ant, t, axis=0)
    r = np.linalg.norm(pts[None, :, :] - ant[:, None, :], axis=2)
    r_near = r.min() - r_margin
    # the window starts SWST + the suppressed transient after the pulse sent `rank` intervals before the packet's own
    tau_first = 2.0 * r_near / C + internal_delay          # the window's label: the echoes arrive late by the delay
    if rank is None:                        # the pulse intervals the echo spans (default: as many as fit)
        rank = max(0, int((tau_first - s1.T_SUPPRESSED) // pri))
    swst_code = int(round((tau_first - rank * pri - s1.T_SUPPRESSED) * s1.F_REF))
    if swst_code < 0:
        raise ValueError('the near range is too short for this rank and PRI')
    tau0 = rank * pri + swst_code / s1.F_REF + s1.T_SUPPRESSED
    n = int(np.ceil(2.0 * (r.max() + r_margin) / C * fs - (tau0 - internal_delay) * fs)) + L
    n += n % 2
    nq = n // 2
    # the instrument's internal delay: echoes arrive late by it, and the calibration packets (the chirp through the
    # internal paths, the packet data starting at the nominal transmit time) show it as the chirp's position
    z = raw_echoes(col, pos, amp, fs, chirp, C * (tau0 - internal_delay) / 2.0, n, lat, lon, height, heading, vel=vel)
    scale = 400.0 / max(np.abs(z.real).max(), np.abs(z.imag).max(), 1e-30)
    zi = np.clip(np.round(z.real * scale), -511, 511).astype(int); zq = np.clip(np.round(z.imag * scale), -511, 511).astype(int)
    swl_code = int(round(n / fs * s1.F_REF))
    n_cal = L + 2 * int(round(1.5e-6 * fs)); n_cal += n_cal % 2
    tc_ = np.arange(n_cal) / fs - internal_delay
    zc = np.where((tc_ >= 0) & (tc_ < txpl), np.exp(2j * np.pi * (txpsf * tc_ + 0.5 * txprr * tc_ ** 2)), 0) * 400.0
    zci = np.clip(np.round(zc.real), -511, 511).astype(int); zcq = np.clip(np.round(zc.imag), -511, 511).astype(int)
    # sub-commutated position and velocity records: the state vector at the start of each 64-packet cycle
    words = {}
    for c0 in range(0, P, 64):
        tc = t[c0] - rank * pri
        pos_c, vel_c = ant[c0], vel[c0]
        raw = pos_c.astype('>f8').tobytes() + vel_c.astype('>f4').tobytes() + bytes([0]) + int(np.floor(tc)).to_bytes(4, 'big') + int(round((tc - np.floor(tc)) * 2 ** 24)).to_bytes(3, 'big')
        for i in range(22):
            words[c0 + i] = int.from_bytes(raw[2 * i:2 * i + 2], 'big')

    def codes(vals):
        return ''.join(('1' if v < 0 else '0') + format(abs(int(v)), '09b') for v in vals)

    def align(b):
        return b + '0' * (-len(b) % 16)

    def packet(seq, time, zr, zq_, signal_type, nq_, subcom=0):
        user_bits = align(codes(zr[0::2])) + align(codes(zr[1::2])) + align(codes(zq_[0::2])) + align(codes(zq_[1::2]))
        user_bits += '0' * (-len(user_bits) % 32)
        user = bytes(int(user_bits[k:k + 8], 2) for k in range(0, len(user_bits), 8))
        h = bytearray(68)
        pdl = 62 + len(user) - 1
        h[0:2] = (0x0800 | 1052).to_bytes(2, 'big'); h[2:4] = (0xC000 | (seq & 0x3FFF)).to_bytes(2, 'big'); h[4:6] = pdl.to_bytes(2, 'big')
        coarse = int(np.floor(time)); fine = int(round((time - coarse) * 65536)) & 0xFFFF
        h[6:10] = coarse.to_bytes(4, 'big'); h[10:12] = fine.to_bytes(2, 'big'); h[12:16] = (0x352EF853).to_bytes(4, 'big')
        h[16:20] = (1).to_bytes(4, 'big'); h[20] = 11; h[21] = 1 if pol[1] == 'H' else 0
        h[26] = subcom; h[27:29] = words.get(seq, 0).to_bytes(2, 'big') if subcom else b'\0\0'
        h[29:33] = seq.to_bytes(4, 'big'); h[33:37] = seq.to_bytes(4, 'big')
        h[37] = 0; h[38] = 31; h[40] = range_decimation; h[41] = 0
        h[42:44] = (0x8000 | prr_code).to_bytes(2, 'big'); h[44:46] = (psf_code & 0x7FFF).to_bytes(2, 'big'); h[46:49] = pl_code.to_bytes(3, 'big')
        h[49] = rank; h[50:53] = pri_code.to_bytes(3, 'big'); h[53:56] = swst_code.to_bytes(3, 'big'); h[56:59] = swl_code.to_bytes(3, 'big')
        h[59] = ({'HH': 1, 'HV': 2, 'VH': 5, 'VV': 6}[pol.upper()] << 4); h[62] = 0; h[63] = signal_type << 4; h[64] = swath
        h[65:67] = nq_.to_bytes(2, 'big')
        return bytes(h) + user

    out = bytearray()
    # the calibration sequence ahead of the echoes: tx, rx, epdn, ta and apdn calibration packets (bypass encoded)
    ncal = 0
    for st in (8, 9, 10, 11, 12):
        for _ in range(cal_packets):
            out += packet(0, t[0] - (5 * cal_packets - ncal) * pri, zci, zcq, st, n_cal // 2)
            ncal += 1
    for i in range(P):
        out += packet(i, t[i], zi[i], zq[i], 0, nq, (i % 64) + 1)
    with open(path, 'wb') as fh:
        fh.write(bytes(out))
    return dict(fs=fs, chirp=chirp, n=n, tx=ant, times=t - rank * pri, tau0=tau0)


# ---------------------------------------------------------------- scenes

def change_mask(x, y, scene, part='all'):
    """True where the ground was disturbed between passes.

    Two rectangles plus a 1 m wide diagonal track, all expressed as fractions
    of the scene so the layout scales.
    """
    s = scene
    r1 = (np.abs(x - 0.22 * s) < 0.06 * s) & (np.abs(y + 0.20 * s) < 0.04 * s)
    r2 = (np.abs(x + 0.25 * s) < 0.03 * s) & (np.abs(y - 0.15 * s) < 0.09 * s)
    d = (x + y) / np.sqrt(2.0)
    along = (x - y) / np.sqrt(2.0)
    track = (np.abs(d + 0.05 * s) < 0.5) & (np.abs(along) < 0.3 * s)
    return track if part == 'track' else r1 | r2 | track


def clutter_pair(scene, res, rng, per_cell=4, gamma_t=0.98, n_bright=3,
                 bright_db=30.0):
    """Scatterers for a repeat-pass pair: positions, pass-1 and pass-2 amplitudes.

    Unchanged clutter keeps coherence `gamma_t`; clutter inside the change mask
    is redrawn. A few bright stable reflectors are added to load the dynamic
    range the way corner reflectors and buildings do in real scenes.
    """
    n = int(per_cell * (scene / res) ** 2)
    xy = (rng.random((n, 2)) - 0.5) * scene
    pos = np.concatenate([xy, np.zeros((n, 1))], axis=1)

    def cn(m):
        return (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2.0)

    a1 = cn(n)
    a2 = gamma_t * a1 + np.sqrt(1.0 - gamma_t ** 2) * cn(n)
    ch = change_mask(xy[:, 0], xy[:, 1], scene)
    a2[ch] = cn(int(ch.sum()))

    bxy = (rng.random((n_bright, 2)) - 0.5) * 0.7 * scene
    bpos = np.concatenate([bxy, np.zeros((n_bright, 1))], axis=1)
    # amplitude such that the focused peak sits bright_db above mean clutter
    bamp = np.full(n_bright, np.sqrt(per_cell) * 10 ** (bright_db / 20.0), np.complex128)
    pos = np.concatenate([pos, bpos])
    return pos, np.concatenate([a1, bamp]), np.concatenate([a2, bamp])


def point_scene(scene):
    """Isolated unit point targets at off-grid positions (center, mid, corner)."""
    s = scene
    xy = np.array([[0.013, -0.021], [0.21 * s + 0.07, 0.19 * s - 0.04],
                   [-0.36 * s + 0.03, -0.37 * s + 0.11], [0.35 * s, -0.1 * s + 0.05],
                   [-0.12 * s + 0.09, 0.33 * s]])
    pos = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
    return pos, np.ones(len(xy), np.complex128)


def taylor_2d(Np, K, nbar=4, sll=35.0):
    from scipy.signal.windows import taylor
    return taylor(Np, nbar=nbar, sll=sll, norm=False)[:, None] * \
        taylor(K, nbar=nbar, sll=sll, norm=False)[None, :]


def ground_grid(n, spacing):
    """Flattened ground-plane pixel coordinates for an n x n image, x = axis 0."""
    ax = (np.arange(n) - n / 2.0) * spacing
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return X.ravel(), Y.ravel(), np.zeros(n * n)


# ------------------------------------------------- harder change-detection scene

STRATA_DB = (0.0, -10.0, -20.0, -30.0)
CLASSES = (0.9, 0.7, 0.5)          # pass-to-pass coherence of the partially changed patches


def hdr_layout(scene):
    """Geometry of the high-dynamic-range scene: four reflectivity strata as
    bands in y, a grid of 4 m patches with reduced coherence in every stratum,
    and a 1 m wide strip that crosses all strata."""
    s = scene
    edges = np.linspace(-0.44 * s, 0.44 * s, len(STRATA_DB) + 1)
    patches = []                                    # (x0, x1, y0, y1, class index)
    for b in range(len(STRATA_DB)):
        yc = 0.5 * (edges[b] + edges[b + 1])
        for i in range(9):
            xc = (-0.36 + 0.09 * i) * s
            patches.append((xc - 2.0, xc + 2.0, yc - 2.0, yc + 2.0, i % len(CLASSES)))
    return edges, patches


def hdr_stratum(y, scene):
    edges, _ = hdr_layout(scene)
    return np.clip(np.searchsorted(edges, y) - 1, 0, len(STRATA_DB) - 1)


def hdr_change_class(x, y, scene):
    """-1 unchanged, 0..len(CLASSES)-1 partial-change class, len(CLASSES) the strip."""
    _, patches = hdr_layout(scene)
    cls = np.full(np.shape(x), -1)
    for x0, x1, y0, y1, c in patches:
        cls = np.where((x >= x0) & (x < x1) & (y >= y0) & (y < y1), c, cls)
    strip = (np.abs(x - 0.40 * scene) < 0.5) & (np.abs(y) < 0.42 * scene)
    return np.where(strip, len(CLASSES), cls)


def clutter_pair_hdr(scene, res, rng, per_cell=4, gamma_t=0.98, bright_db=50.0):
    """Repeat-pass scatterers with reflectivity strata 0 to -30 dB, patches of
    partial decorrelation, a fully changed 1 m strip, and two reflectors
    `bright_db` above the brightest stratum's clutter power per resolution cell."""
    n = int(per_cell * (scene / res) ** 2)
    xy = (rng.random((n, 2)) - 0.5) * scene
    pos = np.concatenate([xy, np.zeros((n, 1))], axis=1)

    def cn(m):
        return (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2.0)

    amp = 10.0 ** (np.asarray(STRATA_DB)[hdr_stratum(xy[:, 1], scene)] / 20.0)
    cls = hdr_change_class(xy[:, 0], xy[:, 1], scene)
    gam = np.full(n, gamma_t)
    for c, g in enumerate(CLASSES):
        gam[cls == c] = g
    gam[cls == len(CLASSES)] = 0.0
    e1 = cn(n)
    e2 = gam * e1 + np.sqrt(1.0 - gam ** 2) * cn(n)
    bxy = np.array([[-0.41 * scene, -0.30 * scene], [-0.41 * scene, 0.31 * scene]])
    bpos = np.concatenate([bxy, np.zeros((2, 1))], axis=1)
    bamp = np.full(2, np.sqrt(per_cell) * 10 ** (bright_db / 20.0), np.complex128)
    return (np.concatenate([pos, bpos]), np.concatenate([amp * e1, bamp]), np.concatenate([amp * e2, bamp]),
            float(n))                               # n = clutter power per sample of a uniformly bright scene
