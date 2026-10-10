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


def _phase_rows(S, f, coef, sign):
    """In place, in row blocks: S[p, k] *= exp(sign 2j pi f[p, k] coef[p]) with f a function of a row slice giving
    its frequencies (Hz) [rows, K] (no full-size temporary)."""
    P = S.shape[0]
    for p0 in range(0, P, 1024):
        sl = slice(p0, min(P, p0 + 1024))
        S[sl] *= np.exp(sign * 2j * np.pi * f(sl) * np.asarray(coef, np.float64)[sl, None]).astype(np.complex64)
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


def read_cphd(cphd, sicd=None, channel=0, meta=False, regrid_tol=1e-3, drop_flagged=False, troposphere=None, phase_sign=None):
    """-> dict(S, ant, fmin, df[, nx, ny, spx, spy, e1, e2]) ready for form_image(**d), and with meta=True also a
    dict: tx, rcv [P, 3] (local frame, m), ref [P] (the one-way range each pulse is compensated to after this
    function's corrections, m), fixed_ref (False when the scene reference point moves and the image needs
    backproject with ref), R (local axes as ECF rows), origin and srp (ECF, m), srp_pulses [P, 3] (each pulse's
    scene reference point, local frame, positions interpolated where the file lacks them), tx_time and rcv_time
    [P] (s), pulses (the (first, last + 1) kept of the file's), channel_index, sicd_transpose (with a SICD: whether
    its rows run along range), the channel's polarization and identifier, the collector, core name and radar mode,
    and notes on what was done to the data."""
    _deps.require('sarpy')
    from sarpy.io.phase_history.converter import open_phase_history
    r = open_phase_history(cphd)
    m = r.cphd_meta
    if isinstance(channel, str):
        ids = [c.Identifier for c in m.Data.Channels]
        pols = [f'{p.Polarization.TxPol}{p.Polarization.RcvPol}' if getattr(p, 'Polarization', None) is not None else None
                for p in m.Channel.Parameters]
        if channel in ids:
            channel = ids.index(channel)
        elif channel.upper() in pols:
            channel = pols.index(channel.upper())
        else:
            raise ValueError(f'channel {channel!r} not among identifiers {ids} or polarizations {pols}')
    if not 0 <= channel < len(m.Data.Channels):
        raise ValueError(f'channel {channel} out of range: the file has {len(m.Data.Channels)} channels')
    ch = m.Data.Channels[channel]
    par = m.Channel.Parameters[channel]
    P, K = ch.NumVectors, ch.NumSamples
    if m.Global.DomainType != 'FX':
        raise ValueError('needs a frequency-domain (FX) CPHD; time-of-arrival (TOA) CPHD is not supported')
    pv = lambda n: r.read_pvp_variable(n, channel)
    tx, rcv, srp = pv('TxPos'), pv('RcvPos'), pv('SRPPos')
    sc0, scss = pv('SC0'), pv('SCSS')
    # read in pulse blocks into one complex64 array; every later step works in place on blocks of rows, so the
    # reader needs about the history's complex64 size (a 16.5 GB ICEYE file needed over 128 GB with whole-array steps)
    S = np.empty((P, K), np.complex64)
    for p0 in range(0, P, 2048):
        p1 = min(P, p0 + 2048)
        S[p0:p1] = r.read_chip((p0, p1), (0, K), index=channel)
    notes = []
    sgn = phase_sign
    if sgn is None:
        sgn = int(m.Global.SGN)
        collector = str(getattr(m.CollectionID, 'CollectorName', '') or '')
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
    td = pv('TDTropoSRP') if troposphere is not False else None
    td = None if td is None else np.asarray(td, np.float64)[lo:hi]
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
        _phase_rows(S, lambda sl: sc0[sl, None] + scss[sl, None] * np.arange(K)[None, :], td, +1)
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
            _phase_rows(S, lambda sl: np.broadcast_to(f0 + df * np.arange(K), (len(range(*sl.indices(len(S)))), K)), 2 * (ref - ref0) / C, -1)
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
    pol = getattr(par, 'Polarization', None)
    info = dict(tx=(tx - origin) @ R.T, rcv=(rcv - origin) @ R.T, ref=cur_ref, fixed_ref=fixed_ref,
                R=R, origin=origin, srp=s0, srp_pulses=(srp - origin) @ R.T,
                sicd_transpose=None if sicd is None else grid['transpose'], tx_time=pv('TxTime')[lo:hi], rcv_time=pv('RcvTime')[lo:hi], pulses=(lo, hi),
                polarization=None if pol is None else f'{pol.TxPol}{pol.RcvPol}', channel=ch.Identifier, channel_index=channel,
                mode=getattr(m.CollectionID.RadarMode, 'ModeType', None),   # None for modes outside the CPHD enumeration (ICEYE: EXPERIMENTAL)
                notes=notes, start=_opt(m, 'Global.Timeline.CollectionStart', str),
                collector=_opt(m, 'CollectionID.CollectorName'), core_name=_opt(m, 'CollectionID.CoreName'))
    return out, info


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
