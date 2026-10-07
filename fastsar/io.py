"""Read a collection from CPHD (frequency domain) and, optionally, the image grid of the matching SICD, into the
arrays form_image takes. Needs sarpy.

    col = fastsar.io.read_cphd('x_CPHD.cphd', sicd='x_SICD.nitf')
    img = fastsar.form_image(**col)

    col, meta = fastsar.io.read_cphd('x_CPHD.cphd', channel='HV', meta=True)  # a channel by polarization or identifier
    img = fastsar.backproject(col['S'], meta['tx'], col['fmin'], col['df'], points, rcv=meta['rcv'], ref=meta['ref'])

The local frame has x along track, y along ground range away from the radar and z up, with its origin at the scene
reference point (the mid-aperture one when it moves). With a SICD the output grid is the vendor's own (pixel counts,
spacings and image-plane axes) and the origin moves to the point under the grid's center pixel, so that pixel (i, j)
of the image is the SICD's column i and row j (when range runs along rows): img.T is the SICD array. Without one,
pass nx, ny, spx, spy, e1, e2 to form_image yourself.

Per-pulse frequency grids (FXFixed false: SC0 and SCSS vary from pulse to pulse) are resampled onto the common
grid of the median start frequency and spacing with a 16-tap Kaiser-windowed sinc, when the largest offset exceeds
regrid_tol samples. A moving scene reference point (SRPFixed false) is re-referenced to the mid-aperture point when
the change of range stays within a quarter of the unambiguous range; otherwise the data keep their per-pulse
reference, meta['ref'] gives it, and the collection must be imaged with backproject (form_image would refuse).

phase_sign overrides the file's SGN (-1: the phase of a scatterer falls with frequency, the convention of
form_image; +1: the data are conjugated on reading). By default the file's SGN is used, except for Capella
collections, whose files declare +1 while their phase follows -1 (checked against the vendor's SICD).

troposphere=True removes the per-pulse troposphere delay at the scene reference point (PVP TDTropoSRP). On the
Umbra Panama collection (25.1 ns mean, 0.48 ns span) it sharpens 1024 x 1024 crops by 1 to 7 percent (fourth
moment of the amplitude) and shifts the image 3.8 m in range; without it the image lands on the vendor's SICD
pixel grid to a quarter pixel near the scene center, so it is off by default.
"""
import numpy as np

C = 299792458.0


def _sinc_regrid(S, u, taps=16, beta=8.0, nbins=4096):
    """Resample each row of S [P, K] at fractional sample positions u [P, K] (Kaiser-windowed sinc from a table of
    nbins fractional offsets; zero outside the row)."""
    P, K = S.shape
    h = taps // 2
    fr = np.arange(nbins + 1) / nbins
    x = fr[:, None] - np.arange(-h + 1, h + 1)[None, :]
    W = (np.sinc(x) * np.i0(beta * np.sqrt(np.clip(1 - (x / h) ** 2, 0, None))) / np.i0(beta)).astype(np.float32)
    out = np.zeros_like(S)
    for p0 in range(0, P, 256):
        sl = slice(p0, min(P, p0 + 256))
        i0 = np.floor(u[sl]).astype(np.int64)
        b = np.rint((u[sl] - i0) * nbins).astype(np.int64)
        acc = np.zeros(i0.shape, S.dtype)
        for j, t in enumerate(range(-h + 1, h + 1)):
            i = i0 + t
            ok = (i >= 0) & (i < K)
            acc += np.where(ok, W[b, j] * np.take_along_axis(S[sl], np.clip(i, 0, K - 1), axis=1), 0)
        out[sl] = acc
    return out


def rereference(S, fmin, df, dref):
    """Move the motion-compensation point of each pulse: S [P, K] compensated to ranges ref_old becomes compensated
    to ref_new, given dref = ref_old - ref_new [P] (one way, or the mean of the two legs)."""
    K = S.shape[1]
    f = fmin + df * np.arange(K)
    return (S * np.exp(-4j * np.pi * f[None, :] / C * np.asarray(dref, np.float64)[:, None])).astype(S.dtype)


def read_cphd(cphd, sicd=None, channel=0, meta=False, regrid_tol=1e-3, drop_flagged=False, troposphere=False, phase_sign=None):
    """-> dict(S, ant, fmin, df[, nx, ny, spx, spy, e1, e2]) ready for form_image(**d), and with meta=True also a
    dict of tx, rcv [P, 3] and ref [P] (local frame), R (local axes in ECF rows), srp (ECF origin), times, the
    channel's polarization and identifier, the radar mode, and what was done to the data."""
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
    ch = m.Data.Channels[channel]
    par = m.Channel.Parameters[channel]
    P, K = ch.NumVectors, ch.NumSamples
    if m.Global.DomainType != 'FX':
        raise ValueError('needs a frequency-domain (FX) CPHD; time-of-arrival (TOA) CPHD is not supported')
    pv = lambda n: r.read_pvp_variable(n, channel)
    tx, rcv, srp = pv('TxPos'), pv('RcvPos'), pv('SRPPos')
    sc0, scss = pv('SC0'), pv('SCSS')
    S = r.read_chip((0, P), (0, K), index=channel).astype(np.complex64)
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
        S = np.conj(S)
    # pulses without valid positions or samples at the ends of the aperture are trimmed; an invalid interior
    # position is interpolated from its neighbors, and non-finite interior samples are zeroed
    posbad = ~np.isfinite(tx).all(1) | ~np.isfinite(rcv).all(1) | ~np.isfinite(srp).all(1)
    S[~np.isfinite(S)] = 0
    empty = np.abs(S).sum(1) == 0
    bad = posbad | empty
    sig = pv('SIGNAL')
    flagged = np.zeros(P, bool) if sig is None else np.asarray(sig).ravel() == 0
    good = np.nonzero(~bad)[0]
    lo, hi = int(good[0]), int(good[-1]) + 1
    if lo or hi < P:
        notes.append(f'trimmed pulses [0, {lo}) and [{hi}, {P})')
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
    if troposphere:
        td = pv('TDTropoSRP')
        if td is None:
            notes.append('no TDTropoSRP in the file: troposphere delay not applied')
        else:
            td = np.asarray(td, np.float64)[lo:hi]
            # the data hold the SRP's echo at its tropospheric delay td beyond the geometric one: exp(+j 2 pi f td)
            # moves it back, f from each pulse's own grid
            S = (S * np.exp(2j * np.pi * (sc0[:, None] + scss[:, None] * np.arange(K)[None, :]) * td[:, None])).astype(np.complex64)
            notes.append(f'removed the troposphere delay at the SRP (mean {td.mean() * 1e9:.2f} ns, span {np.ptp(td) * 1e9:.3f} ns)')
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
    u = (f0 + np.arange(K)[None, :] * df - sc0[:, None]) / scss[:, None]
    shift = float(np.abs(u - np.arange(K)[None, :]).max())
    if shift > regrid_tol:
        S = _sinc_regrid(S, u)
        notes.append(f'resampled per-pulse frequency grids (largest offset {shift:.3g} samples)')
    # local frame at the mid-aperture scene reference point
    s0 = srp[P // 2]
    up = s0 / np.linalg.norm(s0)
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
            S = rereference(S, f0, df, ref - ref0)
            notes.append(f're-referenced to the mid-aperture scene reference point (range change up to {moved:.1f} m)')
        else:
            fixed_ref = False
            notes.append(f'scene reference point moves (range change up to {moved:.0f} m): image with backproject and ref')
    origin = s0
    if sicd is not None:
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
        S = rereference(S, f0, df, (ref if not fixed_ref else ref0) - ref1)
        if fixed_ref:
            ref0 = ref1
        notes.append(f'grid origin moved {np.linalg.norm(shift):.2f} m from the scene reference point onto the SICD pixel grid')
    out = dict(S=S, ant=(apc - origin) @ R.T, fmin=f0, df=df)
    if sicd is not None:
        out.update({k: v for k, v in grid.items() if k != 'transpose'})
    if not meta:
        if not fixed_ref:
            raise ValueError(notes[-1] + '; call read_cphd(..., meta=True)')
        return out
    pol = getattr(par, 'Polarization', None)
    info = dict(tx=(tx - origin) @ R.T, rcv=(rcv - origin) @ R.T, ref=ref if not fixed_ref else ref0, fixed_ref=fixed_ref,
                R=R, origin=origin, srp=s0, sicd_transpose=None if sicd is None else grid['transpose'], tx_time=pv('TxTime')[lo:hi], rcv_time=pv('RcvTime')[lo:hi], pulses=(lo, hi),
                polarization=None if pol is None else f'{pol.TxPol}{pol.RcvPol}', channel=ch.Identifier,
                mode=getattr(m.CollectionID.RadarMode, 'ModeType', None), notes=notes)   # None for modes outside the CPHD enumeration (ICEYE: EXPERIMENTAL)
    return out, info


def sicd_points(sicd, rows, cols, meta, hae=None):
    """Positions [..., 3] in read_cphd's local frame of the SICD pixels (rows, cols) (arrays of one shape), projected
    by the SICD's own model onto the surface at height hae above the ellipsoid (default: the SCP's). Any grid type,
    including range / zero-Doppler grids, which are not planes; backproject onto these points forms the image on the
    vendor's pixels."""
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
