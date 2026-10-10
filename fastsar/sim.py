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
