"""Read a collection from CPHD (frequency domain) and, optionally, the image grid of the matching SICD, into the
arrays form_image takes. Needs sarpy.

    col = fastsar.io.read_cphd('x_CPHD.cphd', sicd='x_SICD.nitf')
    img = fastsar.form_image(**col)

    col, meta = fastsar.io.read_cphd('x_CPHD.cphd', channel='HV', meta=True)  # a channel by polarization or identifier
    img = fastsar.backproject(col['S'], meta['tx'], col['fmin'], col['df'], points, rcv=meta['rcv'], ref=meta['ref'])

The local frame has x along track, y along ground range away from the radar and z up (the ellipsoid normal), with its
origin at the scene reference point (the mid-aperture one when it moves). With a SICD the output grid is the vendor's
own (pixel counts, spacings and image-plane axes) and the origin moves to the point under the grid's center pixel, so
that pixel (i, j) of the image is the SICD's column i and row j (when range runs along rows): img.T is the SICD
array. Without one, pass nx, ny, spx, spy, e1, e2 to form_image yourself.

Per-pulse frequency grids (FXFixed false: SC0 and SCSS vary from pulse to pulse) are resampled onto the common
grid of the median start frequency and spacing with a 16-tap Kaiser-windowed sinc, when the largest offset exceeds
regrid_tol samples. A moving scene reference point (SRPFixed false) is re-referenced to the mid-aperture point when
the change of range stays within a quarter of the unambiguous range; otherwise the data keep their per-pulse
reference, meta['ref'] gives it, and the collection must be imaged with backproject (form_image would refuse).

phase_sign overrides the file's SGN (-1: the phase of a scatterer falls with frequency, the convention of
form_image; +1: the data are conjugated on reading). By default the file's SGN is used, except for Capella
collections, whose files declare +1 while their phase follows -1 (checked against the vendor's SICD).

troposphere (default None: when the file gives a nonzero delay) removes the per-pulse troposphere delay at the
scene reference point (PVP TDTropoSRP); False keeps it, True notes its absence; 'model' removes the delay of a
standard atmosphere (troposphere_delay) for files that give none, such as ICEYE's, whose own images include the
correction and lie about 6 m in ground range from an uncorrected image at 26 degrees of incidence. An uncorrected
delay displaces scatterers in range, by 3.8 m on the Umbra Panama collection (25.1 ns mean), 1.4 m at Silver Peak,
Nevada (9.1 ns), where the correction moves FastSAR's geocoded image 2 m toward its position in Sentinel-2 and NAIP
imagery. Removing it also
sharpens 1024 x 1024 Panama crops by 1 to 7 percent (fourth moment of the amplitude). Capella's SICD images include
the correction (a stripmap registers to within a pixel of Capella's SICD with it, 6 pixels or 3.7 m off without);
Umbra's do not. The ICEYE file checked (X38, 2026) gives a zero delay, so its image keeps the delay: 2.6 m in range,
which places it 5.9 m from ICEYE's SICD in ground range.
"""
import numpy as np
from . import _deps

C = 299792458.0


def _sinc_regrid(S, u, taps=16, beta=8.0, nbins=4096, out=None):
    """Resample each row of S [P, K] at fractional sample positions u [P, K] (Kaiser-windowed sinc from a table of
    nbins fractional offsets; zero outside the row). u may be a function of a row slice returning its rows of u, so
    that the full array is never built; out=S resamples in place (rows are independent)."""
    P, K = S.shape
    h = taps // 2
    fr = np.arange(nbins + 1) / nbins
    x = fr[:, None] - np.arange(-h + 1, h + 1)[None, :]
    W = (np.sinc(x) * np.i0(beta * np.sqrt(np.clip(1 - (x / h) ** 2, 0, None))) / np.i0(beta)).astype(np.float32)
    out = np.zeros_like(S) if out is None else out
    for p0 in range(0, P, 256):
        sl = slice(p0, min(P, p0 + 256))
        us = u(sl) if callable(u) else u[sl]
        i0 = np.floor(us).astype(np.int64)
        b = np.rint((us - i0) * nbins).astype(np.int64)
        acc = np.zeros(i0.shape, S.dtype)
        for j, t in enumerate(range(-h + 1, h + 1)):
            i = i0 + t
            ok = (i >= 0) & (i < K)
            acc += np.where(ok, W[b, j] * np.take_along_axis(S[sl], np.clip(i, 0, K - 1), axis=1), 0)
        out[sl] = acc
    return out


def troposphere_delay(tx, rcv, srp):
    """Two-way troposphere delay (s) [P] at the scene reference point from a standard atmosphere, for the transmitter
    and receiver positions tx, rcv [P, 3] and the point srp [P, 3] (ECF, m): the Saastamoinen hydrostatic zenith
    delay at the point's latitude and height (about 2.3 m at sea level, from the 1976 standard atmosphere's pressure),
    mapped by the cosecant of the elevation of each antenna above the point's horizon. The wet delay (0.05 to 0.3 m
    at the zenith, from the water vapor of the day) is not modeled. For a file whose TDTropoSRP is zero
    (ICEYE's), read_cphd(troposphere='model') removes this delay."""
    tx, rcv, srp = (np.asarray(a, np.float64) for a in (tx, rcv, srp))
    lat, lon, h = ecf_to_geodetic(srp)
    lat, lon = np.radians(lat), np.radians(lon)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    hpa = 1013.25 * np.clip(1.0 - 2.25577e-5 * h, 0.0, None) ** 5.25588
    zenith = 0.0022768 * hpa / (1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00028 * h / 1e3)      # m, one way
    out = np.zeros(len(srp))
    for pos in (tx, rcv):
        d = pos - srp
        sin_el = np.clip(np.einsum('ij,ij->i', d, up) / np.linalg.norm(d, axis=1), np.sin(np.radians(5.0)), 1.0)
        out += zenith / sin_el
    return out / C


def _mode_type(r, m, notes):
    """CollectionID/RadarMode/ModeType: sarpy's value, or the XML's own text when the value is outside the CPHD
    enumeration (sarpy then reports None; ICEYE's CPHD 1.1.0 files write EXPERIMENTAL)."""
    coll = getattr(m, 'CollectionID', None) or getattr(m, 'CollectionInfo', None)       # 1.x, 0.3
    mode = getattr(getattr(coll, 'RadarMode', None), 'ModeType', None)
    if mode is not None:
        return mode
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.cphd_details.get_cphd_bytes())
        for e in root.iter():
            if e.tag.split('}')[-1] == 'ModeType' and e.text and e.text.strip():
                mode = e.text.strip()
                notes.append(f'radar mode {mode!r} is outside the CPHD enumeration (SPOTLIGHT, STRIPMAP, DYNAMIC STRIPMAP); '
                             'taken from the XML')
                return mode
    except Exception:
        pass
    return None


def _cphd03_channel(r, m, arrays, channel):
    """CPHD 0.3 per-vector parameters and samples of one channel, read at the offsets the 0.3 layout gives:
    channel c's vector block starts NumBytesVBP times the vectors of the channels before it into the vector-based
    block, and its samples after those channels' samples. (sarpy's 0.3 reader offsets channel c's vector block by c
    vectors only, so for any channel but the first it returns the first channel's parameters shifted by c pulses, and
    scales the samples with them.) -> (pv(name) -> float64 array or None, read_rows(p0, p1) -> complex64 [rows, K])."""
    h = r.cphd_header
    fname = r.cphd_details.file_name
    dt = m.VectorParameters.get_vector_dtype()
    nv = [int(a.NumVectors) for a in arrays]
    P, K = nv[channel], int(arrays[channel].NumSamples)
    vb = np.memmap(fname, dtype=dt, mode='r', offset=int(h.VB_BYTE_OFFSET) + int(m.Data.NumBytesVBP) * sum(nv[:channel]),
                   shape=(P,))
    raw = {'RE32F_IM32F': '>f4', 'RE16I_IM16I': '>i2', 'RE08I_IM08I': '>i1'}[m.Data.SampleType]
    isz = np.dtype(raw).itemsize
    off = int(h.CPHD_BYTE_OFFSET) + sum(int(a.NumVectors) * int(a.NumSamples) * 2 * isz for a in arrays[:channel])
    sig = np.memmap(fname, dtype=raw, mode='r', offset=off, shape=(P, K, 2))
    amp = np.asarray(vb['AmpSF'], np.float64) if 'AmpSF' in dt.names else None

    def pv(name):
        return np.array(vb[name], np.float64) if name in dt.names else None

    def read_rows(p0, p1):
        b = sig[p0:p1]
        out = np.empty((p1 - p0, K), np.complex64)
        out.real, out.imag = b[..., 0], b[..., 1]
        if amp is not None:
            out *= amp[p0:p1, None].astype(np.float32)
        return out
    return pv, read_rows


def _phase_rows(S, f, coef, sign):
    """In place, in row blocks on all cores: S[p, k] *= exp(sign 2j pi f[p, k] coef[p]) with f(slice, out) writing the
    frequencies (Hz) of a row slice into out [rows, K] (no full-size temporary). The phase is formed in float64 and its
    cosine and sine taken straight into complex64 (on one core a 17 GB history took 100 s, the most of a read)."""
    from .patches import _rows_with
    c = sign * 2 * np.pi * np.asarray(coef, np.float64)
    P, K = S.shape
    rows = 128

    def blk(sl, scratch):
        ph, e = scratch[0][:sl.stop - sl.start], scratch[1][:sl.stop - sl.start]
        f(sl, ph)
        ph *= c[sl, None]
        np.cos(ph, out=e.real, casting='same_kind')
        np.sin(ph, out=e.imag, casting='same_kind')
        S[sl] *= e
    _rows_with(blk, P, lambda: (np.empty((min(rows, P), K)), np.empty((min(rows, P), K), np.complex64)), chunk=rows)
    return S


def rereference(S, fmin, df, dref):
    """Move the motion-compensation point of each pulse: S [P, K] compensated to ranges ref_old becomes compensated
    to ref_new, given dref = ref_old - ref_new [P] (one way, or the mean of the two legs). Returns a new complex64
    array (S is not modified), built in row blocks without a full-size temporary."""
    S = np.asarray(S)
    P, K = S.shape
    f = fmin + df * np.arange(K)
    dref = np.asarray(dref, np.float64)
    out = np.empty((P, K), np.complex64)
    for p0 in range(0, P, 1024):
        sl = slice(p0, min(P, p0 + 1024))
        np.multiply(S[sl], np.exp(-4j * np.pi * f[None, :] / C * dref[sl, None]).astype(np.complex64), out=out[sl])
    return out


def _opt(m, path, conv=None):
    """An optional metadata field by dotted path, or None when any part is missing."""
    for k in path.split('.'):
        m = getattr(m, k, None)
        if m is None:
            return None
    return conv(m) if conv else m


def _assemble(S, tx, rcv, srp, f0, df, notes, sicd, meta, info):
    """The tail shared by the readers: the local frame at the mid-aperture scene reference point, the reference
    range of each pulse (re-referenced to one point when the scene reference point barely moves), the SICD grid
    when one is given, and the (col, meta) outputs. S [P, K] is the frequency-domain phase history on the grid
    f0 + k df compensated to each pulse's srp [P, 3] (ECF), tx and rcv [P, 3] the antenna positions (ECF), info
    the source-specific metadata fields."""
    P, K = S.shape
    # local frame at the mid-aperture scene reference point, z along the ellipsoid normal (not the geocentric radial,
    # which leans up to 0.19 degrees from it: 3 m of height across 1 km)
    s0 = srp[P // 2]
    phi, lam, _ = (np.radians(float(v)) for v in ecf_to_geodetic(s0))
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    apc = 0.5 * (tx + rcv)
    mid = apc[P // 2] - s0
    los_h = mid - (mid @ up) * up
    yhat = -los_h / np.linalg.norm(los_h)
    xhat = np.cross(yhat, up)
    R = np.stack([xhat, yhat, up])
    ref = 0.5 * (np.linalg.norm(tx - srp, axis=1) + np.linalg.norm(rcv - srp, axis=1))
    ref0 = 0.5 * (np.linalg.norm(tx - s0, axis=1) + np.linalg.norm(rcv - s0, axis=1))
    moved = float(np.abs(ref - ref0).max())
    fixed_ref = True
    if moved > 1e-6:
        if moved < 0.25 * C / (2 * df):
            # exp(-j 4 pi f (|x - a| - ref)) -> exp(-j 4 pi f (|x - a| - ref0))
            fk = f0 + df * np.arange(K)

            def freq(sl, out):
                out[:] = fk
            _phase_rows(S, freq, 2 * (ref - ref0) / C, -1)
            notes.append(f're-referenced to the mid-aperture scene reference point (range change up to {moved:.1f} m)')
        else:
            fixed_ref = False
            notes.append(f'scene reference point moves (range change up to {moved:.0f} m): image with backproject and ref')
    moving_note = notes[-1] if not fixed_ref else None
    cur_ref = ref if not fixed_ref else ref0           # the range each pulse is compensated to from here on
    origin = s0
    if sicd is not None:
        _deps.require('sarpy')
        from sarpy.io.complex.converter import open_complex
        sm = open_complex(sicd).sicd_meta
        if sm.Grid.Type not in ('RGAZIM', 'PLANE', 'XRGYCR', 'XCTYAT'):
            raise ValueError(f'the SICD grid is {sm.Grid.Type}, not a plane: read without sicd and form its pixels with '
                             'fastsar.backproject(S, ..., io.sicd_points(sicd, rows, cols, meta))')
        rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
        row_ss, col_ss = float(sm.Grid.Row.SS), float(sm.Grid.Col.SS)
        row_u = np.array(sm.Grid.Row.UVectECF.get_array()) @ R.T
        col_u = np.array(sm.Grid.Col.UVectECF.get_array()) @ R.T
        range_is_row = abs(row_u[1]) > abs(row_u[0])
        e2, e1 = (row_u, col_u) if range_is_row else (col_u, row_u)
        grid = dict(nx=cols if range_is_row else rows, ny=rows if range_is_row else cols,
                    spx=col_ss if range_is_row else row_ss, spy=row_ss if range_is_row else col_ss,
                    e1=e1 / np.linalg.norm(e1), e2=e2 / np.linalg.norm(e2))
        # the grid's center pixel (nx / 2, ny / 2) on the SICD's pixel grid: SICD pixel (r, c) lies at
        # SCP + (r - SCPRow) row_ss row_u + (c - SCPCol) col_ss col_u (first row and column of the full image)
        scp = np.array(sm.GeoData.SCP.ECF.get_array())
        r0 = float(sm.ImageData.SCPPixel.Row - sm.ImageData.FirstRow)
        c0 = float(sm.ImageData.SCPPixel.Col - sm.ImageData.FirstCol)
        origin = (scp + (rows / 2.0 - r0) * row_ss * np.array(sm.Grid.Row.UVectECF.get_array())
                  + (cols / 2.0 - c0) * col_ss * np.array(sm.Grid.Col.UVectECF.get_array()))
        grid['transpose'] = range_is_row
    shift = (origin - s0) @ R.T
    if np.abs(shift).max() > 0:
        ref1 = 0.5 * (np.linalg.norm(tx - origin, axis=1) + np.linalg.norm(rcv - origin, axis=1))
        S = rereference(S, f0, df, cur_ref - ref1)
        cur_ref = ref1
        notes.append(f'grid origin moved {np.linalg.norm(shift):.2f} m from the scene reference point onto the SICD pixel grid')
    out = dict(S=S, ant=(apc - origin) @ R.T, fmin=f0, df=df)
    if sicd is not None:
        out.update({k: v for k, v in grid.items() if k != 'transpose'})
    if not meta:
        if not fixed_ref:
            raise ValueError(moving_note + '; call read_cphd(..., meta=True)')
        return out
    info = dict(info, tx=(tx - origin) @ R.T, rcv=(rcv - origin) @ R.T, ref=cur_ref, fixed_ref=fixed_ref,
                R=R, origin=origin, srp=s0, srp_pulses=(srp - origin) @ R.T,
                sicd_transpose=None if sicd is None else grid['transpose'], notes=notes)
    return out, info


def read_cphd(cphd, sicd=None, channel=0, meta=False, regrid_tol=1e-3, drop_flagged=False, troposphere=None, phase_sign=None):
    """-> dict(S, ant, fmin, df[, nx, ny, spx, spy, e1, e2]) ready for form_image(**d), and with meta=True also a
    dict: tx, rcv [P, 3] (local frame, m), ref [P] (the one-way range each pulse is compensated to after this
    function's corrections, m), fixed_ref (False when the scene reference point moves and the image needs
    backproject with ref), R (local axes as ECF rows), origin and srp (ECF, m), srp_pulses [P, 3] (each pulse's
    scene reference point, local frame, positions interpolated where the file lacks them), tx_time and rcv_time
    [P] (s), pulses (the (first, last + 1) kept of the file's), channel_index, sicd_transpose (with a SICD: whether
    its rows run along range), the channel's polarization and identifier, the collector, core name and radar mode,
    the file's CPHD version, and notes on what was done to the data. Reads CPHD 1.0.x, 1.1.0 and 0.3 (frequency
    domain); a 0.3 file's channels are identified by index or polarization, and its TropoSRP field, whose units 0.3
    does not define, is reported but not applied."""
    _deps.require('sarpy')
    from sarpy.io.phase_history.converter import open_phase_history
    r = open_phase_history(cphd)
    m = r.cphd_meta
    # CPHD 0.3 (the pre-standard NGA format) keeps the per-vector positions and times under the 1.x names but sizes
    # its channels in Data.ArraySize, gives the frequency grid as Fx0 and Fx_SS, the phase sign as Global.PhaseSGN and
    # the collection and polarization in SICD-style CollectionInfo and RadarCollection blocks
    v03 = str(getattr(r, 'cphd_version', '') or '').startswith('0.3')
    if v03:
        arrays = list(m.Data.ArraySize)
        ids = [str(i) for i in range(len(arrays))]
        # 0.3 does not tie RadarCollection's receive channels to its arrays, so they are taken as one per array, in
        # order, only when the counts agree; otherwise the polarizations are left unknown
        rcvch = list(_opt(m, 'RadarCollection.RcvChannels') or [])
        pols = [None] * len(arrays)
        if len(rcvch) == len(arrays):
            for i, c in enumerate(rcvch):
                tr = getattr(c, 'TxRcvPolarization', None)
                pols[i] = None if tr is None or ':' not in str(tr) else str(tr).replace(':', '')
    else:
        arrays = list(m.Data.Channels)
        ids = [c.Identifier for c in arrays]
        pols = [f'{p.Polarization.TxPol}{p.Polarization.RcvPol}' if getattr(p, 'Polarization', None) is not None else None
                for p in m.Channel.Parameters]
    if isinstance(channel, str):
        if channel in ids:
            channel = ids.index(channel)
        elif channel.upper() in pols:
            channel = pols.index(channel.upper())
        else:
            raise ValueError(f'channel {channel!r} not among identifiers {ids} or polarizations {pols}')
    if not 0 <= channel < len(arrays):
        raise ValueError(f'channel {channel} out of range: the file has {len(arrays)} channels')
    ch = arrays[channel]
    P, K = ch.NumVectors, ch.NumSamples
    if m.Global.DomainType != 'FX':
        raise ValueError('needs a frequency-domain (FX) CPHD; time-of-arrival (TOA) CPHD is not supported')
    if v03:
        if troposphere is True:
            raise ValueError("troposphere=True: CPHD 0.3's TropoSRP does not define its units, so it cannot be applied; "
                             "use troposphere='model' or None")
        pv, read_rows = _cphd03_channel(r, m, arrays, channel)
    else:
        pv = lambda n: r.read_pvp_variable(n, channel)
        read_rows = lambda p0, p1: r.read_chip((p0, p1), (0, K), index=channel)
    tx, rcv, srp = pv('TxPos'), pv('RcvPos'), pv('SRPPos')
    if v03:
        if int(getattr(m.Global, 'RefFreqIndex', 0) or 0):
            raise ValueError('CPHD 0.3 file with RefFreqIndex set: its frequencies are offsets from a reference '
                             'frequency the file does not give, so the absolute grid is unknown')
        sc0, scss = pv('Fx0'), pv('Fx_SS')
    else:
        sc0, scss = pv('SC0'), pv('SCSS')
    # read in pulse blocks into one complex64 array; every later step works in place on blocks of rows, so the
    # reader needs about the history's complex64 size (a 16.5 GB ICEYE file needed over 128 GB with whole-array steps)
    S = np.empty((P, K), np.complex64)
    for p0 in range(0, P, 2048):
        p1 = min(P, p0 + 2048)
        S[p0:p1] = read_rows(p0, p1)
    notes = ['CPHD 0.3 file'] if v03 else []
    coll = m.CollectionInfo if v03 else m.CollectionID
    sgn = phase_sign
    if sgn is None:
        sgn = int(m.Global.PhaseSGN if v03 else m.Global.SGN)
        collector = str(getattr(coll, 'CollectorName', '') or '')
        if collector.lower().startswith('capella') and sgn > 0:
            # Capella's open-data CPHDs declare SGN = +1, but their phase follows -1: a 2021 stripmap and a 2024 spotlight
            # collection focus and match the vendor's SICD (amplitude correlation 0.96 and 0.78) only without conjugation
            sgn = -1
            notes.append('Capella collection: SGN = +1 declared, phase taken as SGN = -1')
    if sgn > 0:
        np.conjugate(S, out=S)
    # pulses without valid positions or samples at the ends of the aperture are trimmed; an invalid interior
    # position is interpolated from its neighbors, and non-finite interior samples are zeroed
    posbad = ~np.isfinite(tx).all(1) | ~np.isfinite(rcv).all(1) | ~np.isfinite(srp).all(1)
    empty = np.zeros(P, bool)
    for p0 in range(0, P, 2048):
        blk = S[p0:p0 + 2048]
        blk[~np.isfinite(blk)] = 0
        empty[p0:p0 + 2048] = np.abs(blk).sum(1) == 0
    bad = posbad | empty
    sig = pv('SIGNAL')
    flagged = np.zeros(P, bool) if sig is None else np.asarray(sig).ravel() == 0
    good = np.nonzero(~bad)[0]
    if len(good) < 2:
        raise ValueError(f'{len(good)} of {P} pulses have valid positions and nonzero samples: nothing to image')
    lo, hi = int(good[0]), int(good[-1]) + 1
    if lo or hi < P:
        notes.append('trimmed pulses ' + ' and '.join(f'[{a}, {b})' for a, b in ((0, lo), (hi, P)) if b > a))
    S, tx, rcv, srp, sc0, scss, posbad, empty, flagged = (a[lo:hi] for a in (S, tx, rcv, srp, sc0, scss, posbad, empty, flagged))
    if posbad.any():
        idx = np.arange(len(tx))
        for a in (tx, rcv, srp, sc0[:, None], scss[:, None]):
            for c in range(a.shape[1]):
                a[posbad, c] = np.interp(idx[posbad], idx[~posbad], a[~posbad, c])
        notes.append(f'interpolated the positions of {int(posbad.sum())} interior pulses')
    if empty.any():
        notes.append(f'{int(empty.sum())} interior pulses have no samples')
    P = len(tx)
    if flagged.any():
        if drop_flagged:
            S[flagged] = 0
        notes.append(f'{int(flagged.sum())} pulses flagged SIGNAL=0 (not normal): ' + ('zeroed' if drop_flagged else 'kept'))
    if troposphere not in (None, True, False, 'model'):
        raise ValueError(f"troposphere must be None, True, False or 'model', not {troposphere!r}")
    td = pv('TDTropoSRP') if troposphere is not False and not v03 else None
    td = None if td is None else np.asarray(td, np.float64)[lo:hi]
    if v03 and troposphere is not False:
        # 0.3's TropoSRP has no published definition of its units or of one-way against two-way, so it is not applied
        t3 = pv('TropoSRP')
        if t3 is not None and np.any(np.asarray(t3)[lo:hi]):
            t3 = np.asarray(t3, np.float64)[lo:hi]
            notes.append(f'the file gives TropoSRP (mean {t3.mean():.4g}, units not defined by CPHD 0.3): not applied; '
                         "troposphere='model' applies the modeled delay")
    if troposphere == 'model':
        if td is not None and np.any(td):
            notes.append(f'the file gives a troposphere delay (mean {td.mean() * 1e9:.2f} ns); the modeled delay is applied '
                         'instead')
        td = troposphere_delay(tx, rcv, srp)
        what = 'a modeled troposphere delay (standard atmosphere, hydrostatic)'
    else:
        what = 'the troposphere delay'
    if td is None or not np.any(td):
        if troposphere:
            notes.append('no troposphere delay in the file (TDTropoSRP absent or zero): not applied')
    else:
        # the data hold the SRP's echo at its tropospheric delay td beyond the geometric one: exp(+j 2 pi f td)
        # moves it back, f from each pulse's own grid
        kk = np.arange(K)

        def freq(sl, out):                      # sc0 + scss k, formed in out
            np.multiply(scss[sl, None], kk[None, :], out=out)
            out += sc0[sl, None]
        _phase_rows(S, freq, td, +1)
        notes.append(f'removed {what} at the SRP (mean {td.mean() * 1e9:.2f} ns, span {np.ptp(td) * 1e9:.3f} ns)')
    f0, df = float(np.median(sc0)), float(np.median(scss))
    t1, t2 = pv('TOA1'), pv('TOA2')
    if t1 is not None and t2 is not None:
        span = float(np.nanmax(np.asarray(t2, np.float64)[lo:hi] - np.asarray(t1, np.float64)[lo:hi])) * C / 2
        if span > C / (2 * df) * 1.001:
            import warnings
            msg = (f'valid delay window (TOA1 to TOA2) of {span:.0f} m exceeds the unambiguous range c/(2 df) = '
                   f'{C / (2 * df):.0f} m: the samples are not a plain frequency-domain phase history and images will alias')
            warnings.warn(msg)
            notes.append(msg)
    # u - k is linear in k for each pulse, so its largest magnitude is at the first or the last sample
    a_, b_ = (f0 - sc0) / scss, df / scss - 1.0
    shift = float(np.maximum(np.abs(a_), np.abs(a_ + (K - 1) * b_)).max())
    if shift > regrid_tol:
        _sinc_regrid(S, lambda sl: (f0 + np.arange(K)[None, :] * df - sc0[sl, None]) / scss[sl, None], out=S)
        notes.append(f'resampled per-pulse frequency grids (largest offset {shift:.3g} samples)')
    cpath = 'CollectionInfo' if v03 else 'CollectionID'
    info = dict(tx_time=pv('TxTime')[lo:hi], rcv_time=pv('RcvTime')[lo:hi], pulses=(lo, hi),
                polarization=pols[channel], channel=ids[channel], channel_index=channel,
                mode=_mode_type(r, m, notes), cphd_version=str(getattr(r, 'cphd_version', '') or '') or None,
                start=_opt(m, 'Global.CollectStart' if v03 else 'Global.Timeline.CollectionStart', str),
                collector=_opt(m, f'{cpath}.CollectorName'), core_name=_opt(m, f'{cpath}.CoreName'))
    return _assemble(S, tx, rcv, srp, f0, df, notes, sicd, meta, info)


def read_collection(path, **kw):
    """read_nisar for a NISAR L0B (HDF5, .h5) file, read_sentinel1 for a Sentinel-1 Level-0 product (a name starting
    with s1 that is raw or .SAFE), read_palsar for a zip, a directory or an IMG-/LED- file, read_cphd otherwise; with
    the keyword arguments the reader called accepts; the others are dropped, with a note in meta when they were set."""
    import inspect
    import os
    s = str(path).lower()
    base = os.path.basename(s.rstrip('/'))
    if s.endswith(('.h5', '.hdf5')):
        reader, kind = read_nisar, 'a NISAR file'
    elif base.startswith('s1') and ('raw' in base or base.endswith('.safe')):
        reader, kind = read_sentinel1, 'a Sentinel-1 Level-0 product'
    elif s.endswith('.zip') or os.path.isdir(path) or base.startswith(('img-', 'led-')):
        reader, kind = read_palsar, 'an ALOS PALSAR product'
    else:
        reader, kind = read_cphd, 'a CPHD file'
    allowed = set(inspect.signature(reader).parameters)
    dropped = sorted(k for k in kw if k not in allowed and kw[k] not in (None, 0, False))
    out = reader(path, **{k: v for k, v in kw.items() if k in allowed})
    if dropped and kw.get('meta'):
        out[1]['notes'].append(f'not applicable to {kind}: {", ".join(dropped)}')
    return out


def _tropo(troposphere, tx, rcv, srp, notes):
    """The two-way troposphere delay [P] (s) a raw reader subtracts from its echoes' delays: 'model' the standard
    atmosphere of troposphere_delay, None or False nothing."""
    if troposphere in (None, False):
        return 0.0
    if troposphere != 'model':
        raise ValueError(f"troposphere must be 'model' or None for a raw collection, not {troposphere!r}")
    td = troposphere_delay(tx, rcv, srp)
    notes.append(f'modeled troposphere delay removed ({td.mean() * C / 2:.2f} m one way, standard atmosphere)')
    return td


NISAR_RANGE_DELAY = {'A': 57.125, 'B': 19.65}
# m, JPL's common delay for L-band frequencies A and B as read_nisar subtracts it: the commonDelay of the calibration
# metadata of RSLC and GCOV products with the sign reversed (the products record the correction added to the ranges).
# A calibration JPL revises: products processed in August 2026 carry 53.24 m for frequency A, those processed from
# September 2026 carry 57.125 m. nisar_product(path) reads the values of any RSLC or GCOV file.


def nisar_product(path):
    """Ancillary data of a NISAR level 1 or 2 product (RSLC, GSLC, GCOV; HDF5) for read_nisar: dict(orbit=(time [n],
    position [n, 3], velocity [n, 3]) (ECF, m, m/s), epoch (the times' reference, 'YYYY-MM-DDThh:mm:ss'), orbit_type
    ('MOE', 'POE', ...), range_delay ({'A': m, 'B': m}, JPL's common delay per frequency as read_nisar subtracts it),
    product ('RSLC', ...)). The products carry JPL's medium or precise orbit where an L0B carries a forecast, and the
    delay calibration of their processing date. Needs h5py."""
    from . import _deps
    h5py = _deps.require('h5py')
    with h5py.File(path, 'r') as f:
        prods = [k for k in f['science/LSAR'] if k != 'identification' and 'metadata' in f['science/LSAR'][k]]
        if not prods:
            raise ValueError(f'{path}: no NISAR product with metadata under science/LSAR')
        m = f['science/LSAR'][prods[0]]['metadata']
        orb = m['orbit']
        out = dict(orbit=(orb['time'][()].astype(np.float64), orb['position'][()].astype(np.float64), orb['velocity'][()].astype(np.float64)),
                   epoch=_units_epoch(orb['time']), orbit_type=orbit_type(orb), range_delay={}, product=prods[0])
        cal = m.get('calibrationInformation')
        if cal is not None:
            for k in cal:
                if k.startswith('frequency') and 'commonDelay' in cal[k]:
                    out['range_delay'][k[-1]] = -float(cal[k]['commonDelay'][()])
    return out


def _units_epoch(ds):
    """The epoch of an HDF5 time dataset whose units attribute reads 'seconds since <epoch>'."""
    u = ds.attrs.get('units', b'')
    u = u.decode() if isinstance(u, bytes) else str(u)
    return u.replace('seconds since', '').strip()


def _epoch_offset(a, b):
    """Seconds from epoch b to epoch a (ISO date-times): times counted since a, plus the offset, count since b."""
    a, b = (s.strip().rstrip('Z').replace(' ', 'T') for s in (a, b))
    if a == b:
        return 0.0
    try:
        return float((np.datetime64(a, 'ns') - np.datetime64(b, 'ns')) / np.timedelta64(1, 's'))
    except ValueError as e:
        raise ValueError(f'cannot relate the epochs {a!r} and {b!r}: {e}') from None


def _orbit_given(orbit, epoch):
    """read_nisar's orbit argument -> (time, position, velocity, epoch, orbit type, source)."""
    if isinstance(orbit, (str, bytes)) or hasattr(orbit, '__fspath__'):
        prod = nisar_product(orbit)
        t, p, v = prod['orbit']
        return t, p, v, prod['epoch'], prod['orbit_type'], f'the {prod["product"]} file'
    parts = list(orbit)
    if len(parts) not in (3, 4):
        raise ValueError('orbit must be the path of an RSLC or GCOV product or (time, position, velocity[, epoch])')
    t, p, v = (np.asarray(a, np.float64) for a in parts[:3])
    if t.ndim != 1 or p.shape != (len(t), 3) or v.shape != (len(t), 3):
        raise ValueError(f'orbit arrays of shapes {t.shape}, {p.shape}, {v.shape}: need [n], [n, 3], [n, 3]')
    return t, p, v, (str(parts[3]) if len(parts) == 4 else epoch), 'given', 'the orbit argument'


def read_nisar(path, frequency=None, polarization=None, meta=False, height=None, band_margin=1.0, block=1024, range_delay=None,
               troposphere='model', orbit=None, reference_range=None):
    """A NISAR L0B RRSD granule (HDF5) as read_cphd returns a CPHD: dict(S, ant, fmin, df) with the raw echoes of one
    frequency ('A' or 'B', default the file's first) and polarization ('HH', 'HV', 'VH', 'VV', default the first)
    range compressed with the file's chirp replica and taken to the frequency domain (fastsar.raw), compensated to
    each pulse's zero-Doppler ground point at mid swath at the given height above the ellipsoid (default 0; a moving scene
    reference point, so the collection forms with form_cphd's moving mode), the antenna positions interpolated
    from the file's orbit state vectors at the transmit times (Hermite). band_margin: the fraction of the range
    bandwidth kept around the carrier (1.0: the chirp's band). With meta=True also the dict read_cphd gives, plus
    image_area (the swath's corners, local frame) and refpt. Transmit gaps (the file's valid-sample intervals) are
    zeroed; the per-receiver calibration (caltone, attenuation, TRM phases) is not applied. range_delay: the
    instrument's range delay in meters, subtracted from the file's slant ranges as JPL's processors subtract their
    calibrated common delay: a number, the path of an RSLC or GCOV product to take the value from (nisar_product),
    or by default NISAR_RANGE_DELAY for the frequency (57.125 m for A); 0 removes none. troposphere: 'model' (the
    default, as JPL's processor) removes the standard atmosphere's delay (troposphere_delay); None keeps it.
    orbit: state vectors in place of the L0B's, which are a forecast (10 m off radially on a September 2026 pass):
    the path of an RSLC or GCOV product of the pass (JPL's medium or precise orbit), or (time, position, velocity[,
    epoch]) in ECF with the times in seconds since the epoch ('YYYY-MM-DDThh:mm:ss', default the L0B's).
    reference_range: the slant range (m) of the zero-Doppler reference points, default mid swath; the receiver
    positions are taken at the reference points' echo times, which displaces targets along track by v (r -
    reference) / c, 2.5 m per 100 km of range from the reference. Needs h5py (pip install "fastsar[raw]")."""
    from . import _deps, raw
    h5py = _deps.require('h5py')
    f = h5py.File(path, 'r')
    try:
        ident = f['science/LSAR/identification']
        freqs = [x.decode() if isinstance(x, bytes) else str(x) for x in ident['listOfFrequencies'][()]]
        if frequency is None:
            frequency = freqs[0]
        if frequency not in freqs:
            raise ValueError(f'frequency {frequency!r} not in the file ({freqs})')
        sw = f[f'science/LSAR/RRSD/swaths/frequency{frequency}']
        txs = [k for k in sw if k.startswith('tx')]
        pols = []
        for tx in txs:
            for rx in [k for k in sw[tx] if k.startswith('rx')]:
                pols.append((tx[2:] + rx[2:], tx, rx))
        if polarization is None:
            polarization = pols[0][0]
        match = [p for p in pols if p[0] == polarization.upper()]
        if not match:
            raise ValueError(f'polarization {polarization!r} not in the file ({[p[0] for p in pols]})')
        pol, tx, rx = match[0]
        g, h = sw[tx], sw[tx][rx]
        ds = h[pol]
        P, n = ds.shape
        lut = h['BFPQLUT'][()]
        chirp = g['chirpWaveform'][()]
        sr = g['slantRange'][()].astype(np.float64)
        dr = float(g['slantRangeSpacing'][()])
        fs = C / (2.0 * dr)
        fc = float(g['centerFrequency'][()])
        bw = float(g['rangeBandwidth'][()])
        ut = g['UTCtime'][()].astype(np.float64)
        epoch = g['UTCtime'].attrs.get('units', b'')
        epoch = (epoch.decode() if isinstance(epoch, bytes) else str(epoch)).replace('seconds since ', '')
        bpc = h['basebandPhaseCorrection'][()].astype(np.complex64) if 'basebandPhaseCorrection' in h else None
        nsub = int(g['numberOfSubSwaths'][()]) if 'numberOfSubSwaths' in g else 1
        valid = [g[f'validSamplesSubSwath{k}'][()] for k in range(1, nsub + 1) if f'validSamplesSubSwath{k}' in g]
        orb = f['science/LSAR/RRSD/lowRateTelemetry/orbit']
        ot, op, ov = orb['time'][()].astype(np.float64), orb['position'][()], orb['velocity'][()]
        oepoch, otype, osrc = _units_epoch(orb['time']), orbit_type(orb), 'the file'
        if orbit is not None:
            ot, op, ov, oepoch, otype, osrc = _orbit_given(orbit, oepoch)
        ot = ot + _epoch_offset(oepoch, epoch)
        if ut.min() < ot[0] or ut.max() > ot[-1]:
            raise ValueError(f'the orbit from {osrc} spans {ot[0]:.1f} to {ot[-1]:.1f} s, the pulses {ut.min():.1f} to {ut.max():.1f} s')
        ephem = raw.hermite_orbit(ot, op, ov)
        side = ident['lookDirection'][()]
        side = (side.decode() if isinstance(side, bytes) else str(side)).lower()
        core = ident['granuleId'][()] if 'granuleId' in ident else b''
        core = core.decode() if isinstance(core, bytes) else str(core)
        notes = [f'NISAR L0B frequency {frequency}, {pol}, {P} pulses of {n} samples at {fs / 1e6:.1f} MHz, band '
                 f'{bw / 1e6:.0f} MHz at {fc / 1e9:.4f} GHz, {side}-looking; orbit {otype} with {len(ot)} state vectors'
                 + (f' from {osrc}' if orbit is not None else '')]
        if nsub > 1:
            notes.append(f'{nsub} sub-swaths: samples in the transmit gaps zeroed')
        # the instrument's range delay: the file's slant ranges count from the transmit event; the echo of a point at
        # range r arrives as if from r + delay, and JPL's processors subtract the calibrated delay
        dsrc = ''
        if range_delay is None:
            range_delay = NISAR_RANGE_DELAY.get(frequency, 0.0)
        elif isinstance(range_delay, (str, bytes)) or hasattr(range_delay, '__fspath__'):
            prod = nisar_product(range_delay)
            if frequency not in prod['range_delay']:
                raise ValueError(f'{range_delay}: no common delay for frequency {frequency} in its calibration metadata')
            dsrc, range_delay = f' (the {prod["product"]} file)', prod['range_delay'][frequency]
        range_delay = float(range_delay)
        if range_delay:
            sr = sr - range_delay
            notes.append(f'instrument range delay of {range_delay:.3f} m removed{dsrc}')
        else:
            notes.append('no instrument range delay removed')
        # geometry: transmitter at the pulse time, the zero-Doppler point at mid swath, the receiver at the echo time
        tx_pos, tx_vel = ephem(ut)
        # the last chirp length of the window holds partial correlations: the far edge of the imaged swath is before it
        n_far = max(1, n - len(chirp))
        r_far = sr[n_far - 1]
        r_mid = 0.5 * (sr[0] + r_far) if reference_range is None else float(reference_range)
        if reference_range is not None:
            notes.append(f'zero-Doppler reference points at {r_mid / 1e3:.1f} km slant range')
        height = 0.0 if height is None else float(height)
        srp = raw.zero_doppler_points(tx_pos, tx_vel, r_mid, side, height=height)
        rcv_pos, _ = ephem(ut + 2.0 * np.linalg.norm(srp - tx_pos, axis=1) / C)
        tau_ref = (np.linalg.norm(tx_pos - srp, axis=1) + np.linalg.norm(rcv_pos - srp, axis=1)) / C
        tau0 = np.broadcast_to(2.0 * sr[0] / C - _tropo(troposphere, tx_pos, rcv_pos, srp, notes), (P,))
        S = None
        for p0 in range(0, P, block):
            p1 = min(P, p0 + block)
            z = raw.decode_bfpq(ds[p0:p1], lut)
            if bpc is not None:
                z *= bpc[p0:p1, None]
            if valid:
                keep = np.zeros((p1 - p0, n), bool)
                cols = np.arange(n)[None, :]
                for v in valid:
                    a, b = v[p0:p1, 0][:, None], v[p0:p1, 1][:, None]
                    ok = (a < n) & (b <= n) & (a < b)
                    keep |= ok & (cols >= a) & (cols < b)
                z[~keep] = 0
            blk, fmin, df = raw.fx_history(raw.range_compress(z, chirp), fs, fc, tau0[p0:p1], tau_ref[p0:p1], bw, band_margin)
            if S is None:
                S = np.empty((P, blk.shape[1]), np.complex64)
            S[p0:p1] = blk
        # dithered pulse timing (NISAR varies the PRI by a few percent): the factorized former's pulse filters assume
        # pulses evenly spaced, so the referenced history is resampled onto uniform pulse times (cubic splines across
        # pulses, the positions and reference delays with it); exact backprojection needs none of this
        dt = np.diff(ut)
        if dt.size > 1 and np.ptp(dt) > 1e-3 * dt.mean():
            uu = np.linspace(ut[0], ut[-1], P)
            S, (tx_pos, rcv_pos, srp, tau_ref) = raw.resample_pulses(S, ut, uu, (tx_pos, rcv_pos, srp, tau_ref))
            notes.append(f'pulse timing dithered ({dt.min() * 1e6:.0f} to {dt.max() * 1e6:.0f} us): history resampled to uniform pulse times')
            ut = uu
            tx_vel = ephem(ut)[1]
        # the swath's corners: near and far range at zero Doppler on the first and last pulses
        near_far = [raw.zero_doppler_points(tx_pos[[0, -1]], tx_vel[[0, -1]], r, side, height=height) for r in (sr[0], r_far)]
        corners = np.stack([near_far[0][0], near_far[1][0], near_far[1][1], near_far[0][1]])
        start = f'{epoch.strip()}T00:00:00' if 'T' not in epoch else epoch.strip()
        info = dict(tx_time=ut, rcv_time=ut + tau_ref, pulses=(0, P), polarization=pol, channel=f'{frequency}{pol}',
                    channel_index=[p[0] for p in pols].index(pol), mode='STRIPMAP', start=f'{start} + {ut[0]:.6f} s',
                    collector='NISAR', core_name=core)
        out = _assemble(S, tx_pos, rcv_pos, srp, fmin, df, notes, None, meta, info)
        if meta:
            col, info = out
            info['image_area'] = (corners - info['origin']) @ info['R'].T
            info['refpt'] = np.zeros(3)
            return col, info
        return out
    finally:
        f.close()


def read_palsar(path, polarization=None, meta=False, height=None, band_margin=1.0, block=512, troposphere='model', reference_range=None):
    """An ALOS PALSAR level 1.0 product (raw signal data in CEOS format: a directory or zip holding the LED- leader
    and IMG- image files, or one IMG- file beside its leader) as read_cphd returns a CPHD: the 8-bit I and Q samples
    of one polarization (default the first image file), DC bias removed, range compressed with a linear FM replica
    built from the leader's chirp rate and pulse length, taken to the frequency domain over the chirp's band
    (fastsar.raw) and compensated to each pulse's zero-Doppler ground point at mid swath at the given height
    (default the leader's terrain height); the antenna positions interpolated (Hermite) from the leader's
    platform position record at each line's time. Stripmap (FBS, FBD) products; ScanSAR is not handled."""
    import os
    import zipfile
    from . import raw
    members = {}
    if str(path).lower().endswith('.zip'):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if os.path.basename(n).upper().startswith(('LED-', 'IMG-'))]
        get = lambda n: zf.read(n)
    else:
        d = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        names = [os.path.join(d, n) for n in os.listdir(d) if n.upper().startswith(('LED-', 'IMG-'))]
        get = lambda n: open(n, 'rb').read()
    for n in names:
        members[os.path.basename(n).upper()] = n
    leds = [k for k in members if k.startswith('LED-')]
    imgs = sorted(k for k in members if k.startswith('IMG-'))
    if not leds or not imgs:
        raise ValueError(f'no LED- leader and IMG- image file in {path}')
    if polarization is not None:
        want = [k for k in imgs if k.startswith(f'IMG-{polarization.upper()}-')]
        if not want:
            raise ValueError(f'polarization {polarization!r} not in the product ({[k.split("-")[1] for k in imgs]})')
        imgs = want
    pol = imgs[0].split('-')[1]
    lead = raw.palsar_leader(get(members[leds[0]]))
    data = get(members[imgs[0]])
    hdr = raw.palsar_image_header(data)
    P, n = hdr['records'], hdr['samples']
    if hdr['bytes_per_group'] != 2 or hdr['bits_per_sample'] != 8:
        raise ValueError(f"expected 8-bit I and Q samples (2 bytes per pixel), the file has {hdr['bits_per_sample']} bits and "
                         f"{hdr['bytes_per_group']} bytes per pixel")
    fs, fc = lead['sampling_rate'], C / lead['wavelength']
    T = lead['pulse_length']
    k = -abs(lead['chirp_rate'])                     # PALSAR's chirp sweeps downward; the leader gives the rate's magnitude
    bw = abs(k) * T
    L = int(round(T * fs))
    u = (np.arange(L) - 0.5 * (L - 1)) / fs
    chirp = np.exp(1j * np.pi * k * u ** 2).astype(np.complex64)
    dr = C / (2.0 * fs)
    orbit = raw.hermite_orbit(lead['sv_time'], lead['sv_pos'], lead['sv_vel'])
    # line times as seconds of the platform position record's day
    import datetime
    y, mo, d = lead['sv_day']
    doy0 = datetime.date(y, mo, d).timetuple().tm_yday
    # the geometry fields of every line first (small), the samples in blocks
    fields_all = []
    for p0 in range(0, P, 4096):
        f, _ = raw.palsar_lines(data, hdr, p0, min(P, p0 + 4096))
        fields_all.append(f)
    fields = {k_: np.concatenate([f[k_] for f in fields_all]) for k_ in fields_all[0]}
    day_shift = (fields['year'] - y) * 365 + (fields['doy'] - doy0)       # same day in practice
    ut = fields['msec'] * 1e-3 + 86400.0 * day_shift
    # the millisecond field is too coarse (a millisecond is 7.5 m along track): the microsecond counter since the
    # 1PPS pulse refines it when present, else the line number and PRF
    us = fields['pps_us'] * 1e-6
    if np.any(us) and np.all(us < 1.0):
        fine = np.floor(ut) + us
        fine += np.where(fine - ut > 0.5, -1.0, np.where(ut - fine > 0.5, 1.0, 0.0))
        ut = fine
        notes_time = 'line times from the 1PPS microsecond counter'
    else:
        ut = ut[0] + (fields['line'] - fields['line'][0]) / fields['prf']
        notes_time = 'line times from the first line and the PRF (no microsecond counter)'
    sr0 = fields['slant_range']
    # the line's time counts to the PRI in which its window was sampled (the format's 1Mpps counter 'to this PRI');
    # the echo in it is of the pulse sent one window delay (2 r0 / c, about 12 PRIs) earlier
    ut = ut - 2.0 * sr0 / C
    height = float(lead['terrain_height'] if height is None else height)
    notes = [f'ALOS PALSAR level 1.0 {lead["scene_id"]}, {pol}, {P} lines of {n} samples at {fs / 1e6:.0f} MHz, chirp '
             f'{bw / 1e6:.0f} MHz over {T * 1e6:.0f} us, wavelength {lead["wavelength"]:.4f} m; orbit {len(lead["sv_time"])} '
             f'state vectors ({lead["sv_frame"].split()[0] if lead["sv_frame"] else "?"})']
    notes.append(notes_time)
    if np.ptp(fields['prf']) > 1e-3:
        notes.append(f'PRF changes within the scene ({fields["prf"].min():.1f} to {fields["prf"].max():.1f} Hz)')
    tx_pos, tx_vel = orbit(ut)
    n_far = max(1, n - L)
    r_mid = sr0 + 0.5 * n_far * dr if reference_range is None else float(reference_range)
    if reference_range is not None:
        notes.append(f'zero-Doppler reference points at {r_mid / 1e3:.1f} km slant range')
    srp = raw.zero_doppler_points(tx_pos, tx_vel, r_mid, 'right', height=height)
    rcv_pos, _ = orbit(ut + 2.0 * np.linalg.norm(srp - tx_pos, axis=1) / C)
    tau_ref = (np.linalg.norm(tx_pos - srp, axis=1) + np.linalg.norm(rcv_pos - srp, axis=1)) / C
    tau0 = 2.0 * sr0 / C - _tropo(troposphere, tx_pos, rcv_pos, srp, notes)
    bias = complex(lead['dc_bias_i'] or 0.0, lead['dc_bias_q'] or 0.0)
    S = None
    for p0 in range(0, P, block):
        p1 = min(P, p0 + block)
        _, z = raw.palsar_lines(data, hdr, p0, p1)
        if bias == 0:
            z -= z.mean()                               # no bias in the leader: the block's own mean
        else:
            z -= np.complex64(bias)
        blk, fmin, df = raw.fx_history(raw.range_compress(z, chirp), fs, fc, tau0[p0:p1], tau_ref[p0:p1], bw, band_margin)
        if S is None:
            S = np.empty((P, blk.shape[1]), np.complex64)
        S[p0:p1] = blk
    near_far = [raw.zero_doppler_points(tx_pos[[0, -1]], tx_vel[[0, -1]], r, 'right', height=height)
                for r in (sr0[[0, -1]], sr0[[0, -1]] + n_far * dr)]
    corners = np.stack([near_far[0][0], near_far[1][0], near_far[1][1], near_far[0][1]])
    info = dict(tx_time=ut, rcv_time=ut + tau_ref, pulses=(0, P), polarization=pol, channel=pol, channel_index=0,
                mode='STRIPMAP', start=f'{y:04d}-{mo:02d}-{d:02d}T00:00:00 + {ut[0]:.6f} s', collector=lead['mission'] or 'ALOS',
                core_name=lead['scene_id'])
    out = _assemble(S, tx_pos, rcv_pos, srp, fmin, df, notes, None, meta, info)
    if meta:
        col, info = out
        info['image_area'] = (corners - info['origin']) @ info['R'].T
        info['refpt'] = np.zeros(3)
        return col, info
    return out


def read_sentinel1(path, polarization=None, meta=False, height=None, band_margin=1.0, block=512, pulses=None, swath=None,
                   internal_delay=None, troposphere='model', reference_range=None, orbit=None):
    """A Sentinel-1 Level-0 product (a .SAFE directory or its zip, or one measurement .dat file) as read_cphd returns
    a CPHD: the echo packets of one polarization (default the first measurement file; 'HH', 'HV', 'VV' or 'VH'
    picks the file), decoded (fastsar.sentinel1: FDBAQ, BAQ or bypass), range compressed with the replica of the
    transmitted pulse built from the packet headers (start frequency, ramp rate, length), taken to the frequency
    domain over the chirp's band and compensated to each pulse's zero-Doppler ground point at mid swath at the
    given height, the antenna positions interpolated (Hermite) from the position and velocity records
    sub-commutated in the packets, at the transmit time of each echo (the packet time, latched 1.439 us before the
    packet's own transmit pulse, less RANK pulse intervals).
    pulses: (first, last) packet indices among the echo packets to read a part of the collection. swath: the swath
    number to keep when the file holds several (default the most frequent). internal_delay: the instrument's
    internal time delay (s, two way) subtracted from the echoes' delays as ESA's processor subtracts it; default
    estimated from the product's calibration packets (fastsar.sentinel1.internal_delay, about 0.43 us), 0 none.
    troposphere: 'model' (default) removes the standard atmosphere's delay (troposphere_delay), None keeps it (ESA's
    products do). reference_range: as read_nisar. orbit: state vectors in place of the packets' own records (which
    agree with ESA's precise orbit to 1.4 m on a 2014 datatake): the path of an ESA orbit file (AUX_POEORB or
    AUX_RESORB .EOF, fastsar.sentinel1.orbit_from_file) or (time, position, velocity) in GPS seconds and ECEF.
    Stripmap (S1 to S6) and wave products; the bursts of IW and EW products need fastsar.burst."""
    import os
    import zipfile
    from . import raw, sentinel1 as s1
    s = str(path)
    if s.lower().endswith('.zip'):
        zf = zipfile.ZipFile(s)
        names = [n for n in zf.namelist() if n.lower().endswith('.dat') and '-annot' not in n and '-index' not in n]
        pick = _pick_measurement(names, polarization)
        data = np.frombuffer(zf.read(pick), np.uint8)
    elif os.path.isdir(s):
        names = [os.path.join(s, n) for n in os.listdir(s) if n.lower().endswith('.dat') and '-annot' not in n and '-index' not in n]
        pick = _pick_measurement(names, polarization)
        data = np.memmap(pick, np.uint8, 'r')
    else:
        pick = s
        data = np.memmap(s, np.uint8, 'r')
    base = os.path.basename(pick)
    pol = base.split('-')[4].upper() if base.count('-') >= 5 else '?'       # s1a-s3-raw-s-hh-...
    hdr = s1.parse_packets(data)
    echo = np.nonzero((hdr['signal_type'] == 0) & (hdr['error_flag'] == 0) & (hdr['quads'] > 0))[0]
    if echo.size == 0:
        raise ValueError(f'{base}: no echo packets')
    sw = hdr['swath'][echo]
    if swath is None:
        vals, counts = np.unique(sw, return_counts=True)
        swath = int(vals[np.argmax(counts)])
    echo = echo[sw == swath]
    if pulses is not None:
        lo, hi = int(pulses[0]), int(pulses[1])
        echo = echo[lo:hi]
    P = len(echo)
    if P < 2:
        raise ValueError(f'{base}: {P} echo packets of swath {swath} selected')
    # constant radar configuration within a swath: the first echo packet's values
    k = int(echo[0])
    fs = s1.sampling_frequency(hdr['range_decimation'][k])
    txprr, txpsf, txpl = float(hdr['tx_ramp_rate'][k]), float(hdr['tx_start_frequency'][k]), float(hdr['tx_pulse_length'][k])
    bw = abs(txprr) * txpl
    chirp = s1.chirp(txpsf, txprr, txpl, fs)
    rank, pri = hdr['rank'][echo], hdr['pri'][echo]
    swst = hdr['swst'][echo]
    nq = hdr['quads'][echo]
    n = int(2 * nq.max())
    t_packet = hdr['time'][echo]
    # the stamp is latched T_LATCH before the packet's own transmit pulse, and the echo is of the pulse sent RANK
    # intervals before that one (ESA's Level 1 Detailed Algorithm Definition, 9.23.2)
    tx_time = t_packet + s1.T_LATCH - rank * pri
    if orbit is None:
        ot, op, ov = s1.orbit_from_packets(hdr)
        osrc = 'position records'
    elif isinstance(orbit, (str, bytes)) or hasattr(orbit, '__fspath__'):
        ot, op, ov = s1.orbit_from_file(orbit)
        osrc = 'state vectors of the orbit file'
    else:
        ot, op, ov = (np.asarray(a, np.float64) for a in orbit)
        osrc = 'state vectors given'
    if orbit is not None and (tx_time.min() < ot[0] - np.median(np.diff(ot)) or tx_time.max() > ot[-1] + np.median(np.diff(ot))):
        raise ValueError(f'the orbit given spans GPS seconds {ot[0]:.1f} to {ot[-1]:.1f}, the pulses {tx_time.min():.1f} to {tx_time.max():.1f}')
    ephem = raw.hermite_orbit(ot, op, ov)
    notes = [f'Sentinel-1 Level-0 {base.split("-")[0].upper()} {pol}, swath {swath}, {P} echo packets of {n} samples at {fs / 1e6:.2f} MHz, '
             f'chirp {bw / 1e6:.1f} MHz over {txpl * 1e6:.1f} us, PRI {pri[0] * 1e6:.1f} us, rank {int(rank[0])}; '
             f'{len(ot)} {osrc} at {np.diff(ot).mean():.1f} s']
    if internal_delay is None:
        internal_delay, ncal = s1.internal_delay(data, hdr, k)
        if internal_delay is None:
            internal_delay = 0.0
            notes.append('no calibration packets: the internal time delay is not removed')
        else:
            notes.append(f'internal time delay {internal_delay * 1e9:.1f} ns from {ncal} calibration packets removed')
    else:
        internal_delay = float(internal_delay)
        notes.append(f'internal time delay {internal_delay * 1e9:.1f} ns removed' if internal_delay else 'no internal time delay removed')
    # two-way delay of sample 0 after that pulse: the window opens SWST after the transmit event, the decimation
    # filter's transient is dropped, and the instrument's internal delay makes echoes arrive late
    tau0 = rank * pri + swst + s1.T_SUPPRESSED - internal_delay
    if np.ptp(nq) > 0:
        notes.append(f'number of samples varies ({2 * nq.min()} to {2 * nq.max()}); shorter packets zero padded')
    if np.ptp(swst) > 1e-9:
        notes.append(f'sampling window start varies by {np.ptp(swst) * 1e6:.2f} us')
    height = 0.0 if height is None else float(height)
    tx_pos, tx_vel = ephem(tx_time)
    dr = C / (2.0 * fs)
    n_far = max(1, n - len(chirp))
    r0 = C * tau0 / 2.0
    r_mid = r0 + 0.5 * n_far * dr if reference_range is None else float(reference_range)
    if reference_range is not None:
        notes.append(f'zero-Doppler reference points at {r_mid / 1e3:.1f} km slant range')
    srp = raw.zero_doppler_points(tx_pos, tx_vel, r_mid, 'right', height=height)
    rcv_pos, _ = ephem(tx_time + 2.0 * np.linalg.norm(srp - tx_pos, axis=1) / C)
    tau_ref = (np.linalg.norm(tx_pos - srp, axis=1) + np.linalg.norm(rcv_pos - srp, axis=1)) / C
    tau0 = tau0 - _tropo(troposphere, tx_pos, rcv_pos, srp, notes)
    S = None
    bad = 0
    for p0 in range(0, P, block):
        p1 = min(P, p0 + block)
        z, b = s1.decode_user_data(data, hdr, echo[p0:p1])
        bad += b
        if z.shape[1] < n:
            z = np.pad(z, ((0, 0), (0, n - z.shape[1])))
        blk, fmin, df = raw.fx_history(raw.range_compress(z, chirp), fs, s1.F_CARRIER, tau0[p0:p1], tau_ref[p0:p1], bw, band_margin)
        if S is None:
            S = np.empty((P, blk.shape[1]), np.complex64)
        S[p0:p1] = blk
    if bad:
        notes.append(f'{bad} packets whose user data ended before their last code (samples partial)')
    near_far = [raw.zero_doppler_points(tx_pos[[0, -1]], tx_vel[[0, -1]], r, 'right', height=height) for r in (r0[[0, -1]], r0[[0, -1]] + n_far * dr)]
    corners = np.stack([near_far[0][0], near_far[1][0], near_far[1][1], near_far[0][1]])
    info = dict(tx_time=tx_time, rcv_time=tx_time + tau_ref, pulses=(int(echo[0]), int(echo[-1]) + 1), polarization=pol, channel=pol,
                channel_index=0, mode='STRIPMAP', start=f'GPS {tx_time[0]:.6f} s', collector=base.split('-')[0].upper(),
                core_name=base.rsplit('.', 1)[0])
    out = _assemble(S, tx_pos, rcv_pos, srp, fmin, df, notes, None, meta, info)
    if meta:
        col, info = out
        info['image_area'] = (corners - info['origin']) @ info['R'].T
        info['refpt'] = np.zeros(3)
        return col, info
    return out


def _pick_measurement(names, polarization):
    import os
    """The measurement file of a polarization among a Level-0 product's .dat files (names like
    s1a-s3-raw-s-hh-...dat), default the first in name order."""
    names = sorted(names)
    if not names:
        raise ValueError('no measurement .dat file in the product')
    if polarization is None:
        return names[0]
    want = [n for n in names if f'-{polarization.lower()}-' in n.lower()]
    if not want:
        raise ValueError(f'polarization {polarization!r} not in the product ({[os.path.basename(n).split("-")[4] for n in names]})')
    return want[0]


def orbit_type(orb):
    v = orb['orbitType'][()] if 'orbitType' in orb else b'?'
    return v.decode() if isinstance(v, bytes) else str(v)


def sicd_points(sicd, rows, cols, meta, hae=None):
    """Positions [..., 3] in read_cphd's local frame of the SICD pixels (rows, cols) (arrays of one shape), projected
    by the SICD's own model onto the surface at height hae above the ellipsoid (default: the SCP's). Any grid type,
    including range / zero-Doppler grids, which are not planes; backproject onto these points forms the image on the
    vendor's pixels."""
    _deps.require('sarpy')
    from sarpy.io.complex.converter import open_complex
    sm = sicd if not isinstance(sicd, str) else open_complex(sicd).sicd_meta
    rows, cols = np.broadcast_arrays(np.asarray(rows, np.float64), np.asarray(cols, np.float64))
    h = float(sm.GeoData.SCP.LLH.HAE) if hae is None else float(hae)
    ecf = np.asarray(sm.project_image_to_ground(np.stack([rows.ravel(), cols.ravel()], 1), projection_type='HAE', hae0=h))
    return ecf_to_local(ecf.reshape(rows.shape + (3,)), meta)


def local_to_ecf(points, meta):
    """Points [..., 3] in the local frame of read_cphd -> ECF [..., 3] (m)."""
    return np.asarray(points, np.float64) @ meta['R'] + meta['origin']


def ecf_to_local(points, meta):
    return (np.asarray(points, np.float64) - meta['origin']) @ meta['R'].T


def ecf_to_geodetic(ecf):
    """ECF [..., 3] -> latitude, longitude (degrees), height above the WGS-84 ellipsoid (m)."""
    a, f = 6378137.0, 1 / 298.257223563
    b, e2 = a * (1 - f), f * (2 - f)
    ep2 = (a * a - b * b) / (b * b)
    x, y, z = np.moveaxis(np.asarray(ecf, np.float64), -1, 0)
    p = np.hypot(x, y)
    th = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), p / np.cos(lat) - n


def geodetic_to_ecf(lat, lon, h):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.radians(lat), np.radians(lon)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - e2) + h) * np.sin(lat)], -1)
