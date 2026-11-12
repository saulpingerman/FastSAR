"""One call from a CPHD file to an image, for any collection mode FastSAR handles:

    out = fastsar.form_cphd('scene_CPHD.cphd')                      # mode, grid and window chosen from the file
    out = fastsar.form_cphd('scene_CPHD.cphd', sicd='scene_SICD.ntf', backend='cuda')
    img, origin, e1, e2 = out['image'], out['origin'], out['e1'], out['e2']

Mode. A spotlight collection keeps one scene reference point (SRP) for every pulse; stripmap, sliding spotlight and
dynamic stripmap collections move it with the beam. mode='auto' tells them apart by how far the SRP moves (more than
one range resolution: a moving beam). A spotlight is formed as one image by factorized backprojection over the full
aperture (ImageFormer, with the phase reference moved to the grid center); a moving beam is formed as a mosaic of
range-gated patches (patches.form_mosaic), each pixel weighted by a window over the azimuth band the beam
illuminates around it (the beam center of each pulse is its own SRP): the pulses p for which the sine of the look
angle of the pixel, d_p . (x - a_p) / |x - a_p| with d_p the direction of flight, lies within half the processed
spread of the SRP's. The window is a Hann window over that spread by default; azimuth_window selects a Taylor
window (nbar 4, 35 dB) or a rectangle, both zero outside the spread.

Grid. A ground-plane grid (axes along track and across track, horizontal at the scene) covering the vendor image's
footprint when a SICD is given, else the CPHD's image area (SceneCoordinates.ImageArea; for CPHD 0.3, the corners of Global.ImageArea). Pixel spacing: the SICD's,
widened to 0.8 of its impulse response width, or 0.8 of the resolution the data support (range: c / 2B over the
cosine of the grazing angle; azimuth: lambda / 2 over the angular span of the aperture, or of the processed band).

Window. Taylor (nbar 4, 35 dB) along frequency, and along pulses for a spotlight (window=False: none; a moving
beam then weights every pulse of a patch by one, whatever the pixel, unless azimuth_window is given).

Azimuth band of a moving beam: the SICD's processed bandwidth (Grid.Col.ImpRespBW) when a SICD is given, else
`azimuth_fraction` (default 0.8) of the Doppler band the PRF samples, 2 v sin(theta) / lambda in [-PRF/2, PRF/2].
"""
import numpy as np

from . import io

C = 299792458.0


def _taylor(n):
    from scipy.signal.windows import taylor
    return taylor(n, nbar=4, sll=35.0, norm=False).astype(np.float32)


def _sicd_meta(sicd):
    """The vendor SICD's metadata (fastsar._sicdxml.SICD) from a path (NITF, or .xml: the metadata alone, as ICEYE
    publishes it beside the image), XML, or a metadata object; None for None."""
    if sicd is None:
        return None
    from . import _sicdxml
    return _sicdxml.read(sicd)


def _image_area(cphd, meta):
    """Corners [4, 3] of the CPHD's image area rectangle and its reference point [3] (local frame), or (None, IARP or
    None)."""
    if 'image_area' in meta:                       # a reader that supplies the footprint itself (read_nisar)
        return meta['image_area'], meta.get('refpt')
    m = io._open_cphd(cphd).cphd_meta
    sc = getattr(m, 'SceneCoordinates', None)
    if sc is None:                                 # CPHD 0.3: the image area is four corners (latitude, longitude, height)
        ia3 = getattr(getattr(m, 'Global', None), 'ImageArea', None)
        corners = getattr(ia3, 'Corner', None)
        rp = getattr(getattr(getattr(ia3, 'Plane', None), 'RefPt', None), 'ECF', None)
        ref = None if rp is None else io.ecf_to_local(np.array(rp.get_array(), np.float64)[None], meta)[0]
        if corners is None or len(corners) != 4:
            return None, ref
        ecf = np.array([np.ravel(io.geodetic_to_ecf(float(c.Lat), float(c.Lon), float(c.HAE))) for c in corners], np.float64)
        if np.ptp(ecf, axis=0).max() < 1.0:        # placeholder corners (SARCASTIC writes all four at 0, 0, 0)
            return None, ref
        return io.ecf_to_local(ecf, meta), ref
    ia, iarp = sc.ImageArea, sc.IARP
    ref = None if iarp is None else io.ecf_to_local(np.array(iarp.ECF.get_array(), np.float64)[None], meta)[0]
    if ia is None or iarp is None or sc.ReferenceSurface is None or sc.ReferenceSurface.Planar is None:
        return None, ref
    # the corner points (latitude, longitude) are unambiguous; ImageArea is in meters by the standard, but some
    # producers write it in ImageGrid lines and samples (Capella: 49,837 lines of 0.2 m read as meters would make a
    # 10 km footprint 50 km), so it is used only without corner points, and rescaled when it matches the grid's indices
    iacp = getattr(sc, 'ImageAreaCornerPoints', None)
    h = float(iarp.LLH.HAE) if getattr(iarp, 'LLH', None) is not None else 0.0
    if iacp is not None and len(iacp) == 4:
        ecf = np.array([np.ravel(io.geodetic_to_ecf(float(c.Lat), float(c.Lon), h)) for c in iacp], np.float64)
        return io.ecf_to_local(ecf, meta), ref
    p0 = np.array(iarp.ECF.get_array(), np.float64)
    ux = np.array(sc.ReferenceSurface.Planar.uIAX.get_array(), np.float64)
    uy = np.array(sc.ReferenceSurface.Planar.uIAY.get_array(), np.float64)
    x1, y1, x2, y2 = ia.X1Y1.X, ia.X1Y1.Y, ia.X2Y2.X, ia.X2Y2.Y
    g = getattr(sc, 'ImageGrid', None)
    if g is not None and g.IAXExtent is not None and g.IAYExtent is not None:
        fx, nx_ = g.IAXExtent.FirstLine, g.IAXExtent.NumLines
        fy, ny_ = g.IAYExtent.FirstSample, g.IAYExtent.NumSamples
        if abs(x1 - fx) <= 1 and abs(x2 - (fx + nx_ - 1)) <= 1 and abs(y1 - fy) <= 1 and abs(y2 - (fy + ny_ - 1)) <= 1:
            sx, sy = float(g.IAXExtent.LineSpacing), float(g.IAYExtent.SampleSpacing)
            x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
    ecf = np.array([p0 + x * ux + y * uy for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2))])
    return io.ecf_to_local(ecf, meta), ref


def form_cphd(cphd, sicd=None, *, mode='auto', backend='auto', window=True, spacing=None, channel=0, patch=1024,
              azimuth_fraction=0.8, extent=None, height=None, precision='float32', target_db=-40.0, info=None,
              autofocus=False, troposphere=None, reader_options=None, axes='track', aperture='auto',
              azimuth_window=None, band=None, grid=None, shift=None, range_gate='auto', _plan=None):
    """Form the image of a CPHD collection (see the module docstring).

    cphd: path. sicd: the vendor's SICD (path, .xml metadata, or SICD metadata) for the footprint, spacing and
    processed azimuth band, or None. mode: 'auto', 'spotlight' or 'moving' (stripmap, sliding spotlight, dynamic
    stripmap). backend: 'auto', 'cpu', 'cuda', 'tpu' or 'jax'. spacing: (along track, across track) in meters, or
    None. extent: (along track, across track) in meters around the scene center, overriding the footprint. height:
    the grid plane's height above the ellipsoid at the scene center (default: the SICD's scene center point, else the
    CPHD's image area reference point; a scatterer at another height appears displaced in range). patch:
    mosaic patch size in pixels. info: a list that receives one dict per mosaic patch. autofocus: phase gradient
    autofocus (fastsar.autofocus.autofocus, two rounds; spotlight only), which forms the image three times.
    troposphere: remove the per-pulse troposphere delay the file gives (io.read_cphd; default: when nonzero; a raw
    collection's readers remove the standard atmosphere's by default, None keeps it);
        'model' removes a standard-atmosphere delay instead, for files that give none (ICEYE).
    reader_options: a dict of further keyword arguments for the reader of this file type (io.read_collection), such as
        pulses=(first, last) or polarization for the raw-data readers, or ref_freq for a CPHD 0.3 file with
        RefFreqIndex set (io.read_cphd).
    axes: 'track' (e1 along the track, e2 across it) or, for a spotlight, 'los': e2 along the line of sight from the
        mid-aperture antenna position to the scene center, projected on the ground and pointing away from the radar,
        e1 = e2 x up across it (the axes fastsar.planes resamples from, and the ones on which range runs along a
        grid axis whatever the squint); extent is then (across, along) the line of sight. 'auto': 'los' for a
        spotlight, 'track' for a moving beam.
    aperture: 'auto' (a spotlight without a sicd or a spacing: the central part of the aperture that makes the
        azimuth resolution the ground range resolution), 'square' (that part whatever else is given), 'full', or
        the pulses to use, (first, last + 1) of those read.
    azimuth_window: of a moving beam, each pixel's weighting of the pulses across its processed azimuth band: 'hann',
        'taylor' (nbar 4, 35 dB) or 'uniform'; None: 'hann' with window, else no per-pixel weighting.
    band: the frequencies to use, (first, last) in Hz, or one such pair per pulse of the aperture [pulses, 2]: the
        samples outside are given no weight, and the Taylor window (window=False: none) spans the band of each
        pulse instead of the whole band. aperture=(first, last + 1) and band= are the selection a coherent pair is
        formed with: each collection cut, before formation, to the look directions and the ground-range
        wavenumbers both have, under one weighting (fastsar.stack.common_support gives both for both).
    grid: an Earth-fixed ground grid (fastsar.stack.ground_grid, or a product of an earlier call: the same pixels)
        in place of the collection's own: its plane, axes, spacing and shape; spacing, extent, height and axes
        then do not apply, and the product records it (grid, and axes 'grid'). shift: (d1, d2) in meters along the
        grid's e1 and e2: the image is formed on the grid displaced by it and recorded on the grid itself, which
        registers a collection whose orbit knowledge is off by that much (fastsar.stack.register; the product's
        frame is moved with it, so its geolocation is the grid's).
    range_gate: 'auto' gates the phase history in range to the image's swath before formation where the cuda
        backend's first level could not hold it otherwise (a note says so), True wherever the swath is under half
        of the unambiguous range, False never (fastsar.ImageFormer).
    -> dict(image [nx, ny] complex64, origin [3] (local, pixel (0, 0)), e1, e2 (unit axes, local), spx, spy, mode,
    band (first and last frequency, Hz), bandwidth (spatial frequency support along e1 and e2, cycles/m), window,
    phase_error (autofocus: per pulse, rad, else None), meta (read_cphd's), notes, and the geometry of the grid:
    plane ('ground'), axes, center [3] (local), height (the plane's, local z), graze (rad, at the center), pulses
    (first, last + 1 of the pulses read that were used), dsin (a moving beam's processed spread of the sine of the
    look angle about each pulse's SRP, else None); with grid=: grid (fastsar.stack.as_grid's dict, ECEF), shift,
    source (the file) and form_options (the arguments that form it again)). Pixel (i, j) lies at
    origin + i spx e1 + j spy e2 in the local frame (io.local_to_ecf for ECF; products.geolocate for latitude and
    longitude)."""
    from .api import ImageFormer, _backend
    from . import patches
    if mode not in ('auto', 'spotlight', 'moving'):
        raise ValueError("mode must be 'auto', 'spotlight' or 'moving'")
    if axes not in ('track', 'los', 'auto'):
        raise ValueError("axes must be 'track', 'los' or 'auto'")
    if isinstance(aperture, str) and aperture not in ('auto', 'square', 'full') or \
            not isinstance(aperture, str) and np.shape(aperture) != (2,):
        raise ValueError("aperture must be 'auto', 'square', 'full' or the pulses (first, last + 1)")
    if azimuth_window not in (None, 'hann', 'taylor', 'uniform'):
        raise ValueError("azimuth_window must be 'hann', 'taylor', 'uniform' or None")
    backend = _backend(backend)                       # before the file is read
    for n, v in (('spacing', spacing), ('extent', extent)):
        if v is not None and (np.size(v) not in (1, 2) or not np.all(np.isfinite(v)) or np.min(v) <= 0):
            raise ValueError(f'{n} must be one or two positive lengths in meters, got {v!r}')
    if extent is not None and np.size(extent) != 2:
        raise ValueError(f'extent must be (along track, across track) in meters, got {extent!r}')
    kw = dict(channel=channel, meta=True, height=height)
    if troposphere is not None:                   # else each reader's default: the file's delay (CPHD), the model (raw)
        kw['troposphere'] = troposphere
    kw.update(reader_options or {})
    if band is not None and (np.ndim(band) not in (1, 2) or np.shape(band)[-1] != 2 or not np.all(np.isfinite(band))
                             or np.min(band) <= 0 or np.any(np.diff(np.asarray(band, np.float64), axis=-1) <= 0)):
        raise ValueError('band must be (first, last) frequencies in Hz, first < last, or one such pair per pulse')
    if grid is not None and any(v is not None for v in (spacing, extent, height)):
        raise ValueError('grid= gives the plane, the spacing and the shape: leave spacing, extent and height unset')
    if shift is not None and grid is None:
        raise ValueError('shift= displaces a grid: give grid=')
    col, meta = io.read_collection(cphd, **kw)
    gl = None
    if grid is not None:                          # the grid in this collection's frame (moved by -shift)
        from . import stack
        meta, gl = stack._enter(grid, shift, meta)
    S = col['S']
    P, K = S.shape
    f0, df = float(col['fmin']), float(col['df'])
    lam = C / (f0 + K / 2 * df)
    ant = np.asarray(col['ant'], np.float64)
    sm = _sicd_meta(sicd)
    notes = list(meta.get('notes', []))
    # the SRP of every pulse (local frame, positions interpolated where the file lacks them), for the mode and a
    # moving beam's center
    channel = meta.get('channel_index', channel)
    srp = np.asarray(meta['srp_pulses'], np.float64)
    rres = C / (2 * K * df)
    if mode == 'auto':
        mode = 'moving' if np.linalg.norm(srp.max(0) - srp.min(0)) > rres else 'spotlight'
    spot = mode == 'spotlight'
    if autofocus and not spot:
        raise ValueError('autofocus is implemented for spotlight collections only')
    if axes == 'auto':
        axes = 'los' if spot else 'track'
    if axes == 'los' and not spot:
        raise ValueError("axes='los' is for spotlight collections: a moving beam has no one line of sight")
    # ground-plane axes at the scene: e1 along track, e2 across, horizontal
    d = np.gradient(ant, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    up = np.array([0.0, 0.0, 1.0])
    e1 = d[P // 2] - (d[P // 2] @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    # footprint corners and the scene's reference point (local)
    corners, refpt = _image_area(cphd, meta)
    if sm is not None:
        refpt = io.ecf_to_local(np.array(sm.GeoData.SCP.ECF.get_array(), np.float64)[None], meta)[0]
        if extent is None:
            rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
            R_, C_ = np.meshgrid([0, rows - 1], [0, cols - 1], indexing='ij')
            corners = io.sicd_points(sm, R_.ravel(), C_.ravel(), meta)
    if extent is not None or gl is not None:
        corners = None
    elif corners is None:
        raise ValueError('the CPHD has no planar image area: give a sicd or an extent')
    center = (srp[len(srp) // 2] if refpt is None else refpt) if corners is None else corners.mean(0)
    # the grid plane through the reference point (not the corners' mean: over a footprint of tens of km the corners
    # lie meters below the tangent plane), or at the height asked for
    if height is not None:
        lat, lon, _ = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(io.local_to_ecf(center[None], meta)[0]))
        hz = float(io.ecf_to_local(io.geodetic_to_ecf(lat, lon, float(height))[None], meta)[0, 2])
    else:
        hz = (refpt if refpt is not None else center)[2]
    center = np.array([center[0], center[1], hz])
    graze = np.arcsin(np.clip(-((center - ant[P // 2]) / np.linalg.norm(center - ant[P // 2])) @ up, 0.05, 1.0))
    if axes == 'los':
        e2 = (center - ant[P // 2]) - ((center - ant[P // 2]) @ up) * up
        e2 /= np.linalg.norm(e2)
        e1 = np.cross(e2, up)
    if gl is not None:                            # the grid's own center, plane (horizontal at the grid) and axes
        center, e1, e2, up, axes = gl['center'], gl['e1'], gl['e2'], gl['normal'], 'grid'
        hz = float(center[2])
        graze = np.arcsin(np.clip(-((center - ant[P // 2]) / np.linalg.norm(center - ant[P // 2])) @ up, 0.05, 1.0))
        los = center - ant[P // 2]
        if autofocus and abs(los @ e1) > np.tan(np.radians(5.0)) * (los @ e2):
            raise ValueError("autofocus estimates the phase error along the grid's first axis, which must be this "
                             "collection's azimuth: the grid's second axis is "
                             f'{np.degrees(np.arctan2(abs(los @ e1), los @ e2)):.0f} degrees from its line of sight')
    ref_r = np.asarray(meta['ref'], np.float64)
    used = (0, P)
    if not isinstance(aperture, str):                 # the pulses given (a pair on one azimuth interval: fastsar.stack)
        lo_, hi_ = (int(v) for v in aperture)
        if not spot or not 0 <= lo_ < hi_ - 1 < P:
            raise ValueError(f'aperture={aperture!r}: pulses (first, last + 1) of the {P} of a spotlight collection')
        S, ant, srp, ref_r = S[lo_:hi_], ant[lo_:hi_], srp[lo_:hi_], ref_r[lo_:hi_]
        used = (lo_, hi_)
        notes.append(f'aperture: pulses {lo_} to {hi_ - 1} of {P}')
        P = hi_ - lo_
    elif spot and (aperture == 'square' or (aperture == 'auto' and sm is None and spacing is None)):
        # without the vendor's product or a spacing: square resolution, the azimuth resolution matched to the ground
        # range resolution by the central part of a longer aperture (the vendor's products do likewise); spacing=
        # forms the whole aperture
        u = (center[None] - ant) / np.linalg.norm(center[None] - ant, axis=1, keepdims=True)
        ang = np.arccos(np.clip(u @ u[P // 2], -1, 1)) * np.sign(np.arange(P) - P // 2)
        want = lam / (2 * rres / np.cos(graze))              # the angular span for azimuth resolution = ground range resolution
        if np.ptp(ang) > 1.15 * want:
            keep = np.nonzero(np.abs(ang) <= want / 2)[0]
            lo_, hi_ = int(keep[0]), int(keep[-1]) + 1
            S, ant, srp, ref_r = S[lo_:hi_], ant[lo_:hi_], srp[lo_:hi_], ref_r[lo_:hi_]
            used = (lo_, hi_)
            notes.append(f'aperture: the central {hi_ - lo_} of {P} pulses ({np.degrees(want):.2f} of {np.degrees(np.ptp(ang)):.2f} '
                         'degrees) for square resolution; give spacing= or a sicd for the whole aperture')
            P = hi_ - lo_
    # azimuth band of a moving beam (spread of sin(look angle) about each pulse's SRP)
    dsin = None
    if not spot:
        if sm is not None:
            dsin = float(sm.Grid.Col.ImpRespBW) * lam / 2
        else:
            prf = (P - 1) / float(meta['tx_time'][-1] - meta['tx_time'][0])
            vel = np.linalg.norm(np.gradient(ant, axis=0), axis=1).mean() * prf
            dsin = azimuth_fraction * lam * prf / (2 * vel)
            notes.append(f'azimuth band {azimuth_fraction:.2f} of the PRF ({prf:.0f} Hz): spread of sin(look) {dsin:.4f}')
    # a caller's own grid (process_cphd): spacing and bounds (m from the center along e1 and e2) from the geometry
    plan = {} if _plan is None else (_plan(dict(S=S, ant=ant, fmin=f0, df=df, center=center, e1=e1, e2=e2, graze=graze,
                                                corners=corners, mode=mode, meta=meta, sicd=sm, notes=notes,
                                                pulses=used, srp=srp, dsin=dsin, grid=gl)) or {})
    if plan.get('window') is not None:
        window = bool(plan['window'])
    if plan.get('azimuth_window') is not None:
        azimuth_window = plan['azimuth_window']
    if plan.get('axes') is not None:                  # any plane through the center (a slant-plane grid)
        e1, e2 = (np.asarray(v, np.float64) for v in plan['axes'])
    bd, bwid = (f0, f0 + (K - 1) * df), K * df
    if band is not None:
        # the frequencies used: on each block of 64 pulses the window spans the block's own band, 0 outside it,
        # and along the aperture it is a function of the look direction on the ground, not of the pulse's number:
        # two collections cut to one support (fastsar.stack.common_support) then carry one weighting of it
        fb = np.asarray(band, np.float64)
        if fb.shape not in ((2,), (P, 2)):
            raise ValueError(f'band: one (first, last) pair, or one per pulse of the aperture ({P}), got shape '
                             f'{fb.shape}')
        fb = np.clip(np.broadcast_to(fb, (P, 2)), f0, f0 + (K - 1) * df)
        if np.any(fb[:, 1] - fb[:, 0] < 2 * df):
            raise ValueError(f'band: no samples of this collection ({f0:.6g} to {bd[1]:.6g} Hz) on some pulse')
        from .patches import _rows_with
        fk, xg = f0 + df * np.arange(K), np.linspace(0.0, 1.0, 4097)
        tw = _taylor(4097) if window else np.ones(4097, np.float32)
        wp = np.ones(P, np.float32)
        if spot and window:
            u = center[None] - ant
            ph = np.unwrap(np.arctan2(u @ e1, u @ e2))
            wp = np.interp((ph - ph[0]) / (ph[-1] - ph[0]), xg, tw).astype(np.float32)

        def _cut(sl, w):
            w = w[:sl.stop - sl.start]
            lo_, hi_ = fb[sl].mean(0)
            wk = np.interp((fk - lo_) / (hi_ - lo_), xg, tw, left=0.0, right=0.0).astype(np.float32)
            np.multiply(wp[sl, None], wk[None, :], out=w)
            S[sl] *= w
        _rows_with(_cut, P, lambda: np.empty((min(64, P), K), np.float32), chunk=64)
        bd = (float(np.median(fb[:, 0])), float(np.median(fb[:, 1])))
        bwid = bd[1] - bd[0]
        notes.append(f'band: {bd[0] / 1e6:.1f} to {bd[1] / 1e6:.1f} MHz used of {f0 / 1e6:.1f} to '
                     f'{(f0 + (K - 1) * df) / 1e6:.1f} MHz ({100 * bwid / ((K - 1) * df):.0f}%)')
    elif window:
        wk = _taylor(K)
        wp = _taylor(P) if spot else np.ones(P, np.float32)
        from .patches import _rows_with

        def _win(sl, w):                        # in place, row blocks on all cores
            w = w[:sl.stop - sl.start]
            np.multiply(wp[sl, None], wk[None, :], out=w)
            S[sl] *= w
        _rows_with(_win, P, lambda: np.empty((min(256, P), K), np.result_type(wp, wk)), chunk=256)
    # resolution the data support (before windowing)
    if spot:
        u = (center[None] - ant) / np.linalg.norm(center[None] - ant, axis=1, keepdims=True)
        span = float(np.arccos(np.clip(u[0] @ u[-1], -1, 1)))
        ares = lam / (2 * max(span, 1e-6))
    else:
        ares = lam / (2 * dsin)
    # spacing
    if gl is not None:
        spx, spy = gl['spx'], gl['spy']
    elif plan.get('spacing') is not None:
        spx, spy = (float(v) for v in plan['spacing'])
    elif spacing is not None:
        sp_ = np.ravel(np.asarray(spacing, np.float64))
        spx, spy = (float(sp_[0]), float(sp_[0])) if sp_.size == 1 else (float(sp_[0]), float(sp_[1]))
    elif sm is not None:
        spx = max(float(sm.Grid.Col.SS), 0.8 * float(sm.Grid.Col.ImpRespWid))
        spy = max(float(sm.Grid.Row.SS), 0.8 * float(sm.Grid.Row.ImpRespWid))
        if getattr(sm.Grid, 'ImagePlane', 'SLANT') != 'GROUND':      # slant-plane rows: project onto the ground
            spy /= np.cos(np.radians(float(sm.SCPCOA.GrazeAng)))
    else:
        spx, spy = 0.8 * 1.2 * ares, 0.8 * 1.2 * rres / np.cos(graze)          # 1.2: the Taylor window's broadening
    # grid
    if gl is not None:                                # its own shape and origin, set below
        a1 = a2 = np.zeros(2)
    elif plan.get('bounds') is not None:
        b = np.asarray(plan['bounds'], np.float64)
        a1, a2 = center @ e1 + b[:2], center @ e2 + b[2:]
    elif corners is None:
        L1, L2 = map(float, extent)
        a1, a2 = center @ e1 + np.array([-L1, L1]) / 2, center @ e2 + np.array([-L2, L2]) / 2
    else:
        a1, a2 = corners @ e1, corners @ e2
    h = hz
    nx = int(np.ceil((a1.max() - a1.min()) / spx)) + 1
    ny = int(np.ceil((a2.max() - a2.min()) / spy)) + 1
    origin = a1.min() * e1 + a2.min() * e2 + h * up
    if gl is not None:
        (nx, ny), origin = gl['shape'], gl['origin']
    elif plan.get('bounds') is not None:
        origin = center + b[0] * e1 + b[2] * e2
    # host memory: the history, the image and the mosaic's accumulation; past what is available, a clear error
    from . import memory as _mem
    need = S.nbytes + 2 * 8 * nx * ny
    avail = _mem.host_available()
    if need > 0.9 * avail:
        raise MemoryError(f'the {nx} x {ny} pixel grid ({spx:.2f} x {spy:.2f} m) and the {S.nbytes / 2**30:.1f} GiB phase '
                          f'history need about {need / 2**30:.0f} GiB of host memory, {avail / 2**30:.0f} GiB is available: '
                          'give a smaller extent= or a coarser spacing=')
    fx = dict(S=S, fmin=f0, df=df, ref=ref_r, band=(f0, f0 + K * df))
    if spot:
        # phase reference to the grid center, a grid centered on it (as ImageFormer takes it)
        c = origin + (nx / 2.0) * spx * e1 + (ny / 2.0) * spy * e2
        # referenced to |ant - c|, the range the factorized former assumes (not the bistatic half path, which differs
        # from it by a near constant 0.19 mm on ICEYE and limited the image to -52 dB at 7 cm resolution)
        ref_new = np.linalg.norm(ant - c, axis=1)
        dref = ref_r - ref_new
        f = f0 + df * np.arange(K)

        def freq(sl, out):
            out[:] = f
        io._phase_rows(S, freq, 2 * dref / C, -1, linear=(f0, df))     # in place, on all cores
        notes.append(f'phase reference moved {np.linalg.norm(c):.1f} m to the grid center')
        kw = dict(backend=backend, precision=precision, window=False, target_db=target_db)
        if autofocus:
            from .autofocus import autofocus as pga_autofocus
            img, phi = pga_autofocus(S, ant - c, f0, df, nx, ny, spx, spy, e1, e2, **kw)
            notes.append(f'autofocus: removed a phase error of {np.std(phi):.2f} rad rms (constant and linear terms excluded)')
        else:
            former = ImageFormer(ant - c, f0, df, K, nx, ny, spx, spy, e1, e2, range_gate=range_gate, **kw)
            img = former(S)
            if former.range_gate is not None:
                notes.append(former.range_gate['note'])
        origin = c - (nx / 2.0) * spx * e1 - (ny / 2.0) * spy * e2
    else:
        sp = ((srp - ant) * d).sum(1) / np.linalg.norm(srp - ant, axis=1)        # beam center: each pulse's SRP

        def beam(idx, pts):
            w = np.asarray(pts)[None] - ant[idx][:, None]
            return ((w * d[idx][:, None]).sum(-1) / np.linalg.norm(w, axis=-1) - sp[idx][:, None]) / (dsin / 2)

        # a window that does not end at zero is cut at the band's edge; azimuth_pad (process_cphd's) widens the
        # rectangle beyond the band, for a cut made on the image's spectrum afterwards
        awin = azimuth_window or ('hann' if window else None)
        pp = (patch, patch)
        if plan.get('fit_patch'):                     # a grid smaller than a patch as one patch, in whole tiles of 32
            pp = tuple(int(min(patch, 32 * -(-n // 32))) for n in (nx, ny))
        img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=pp, beam=beam, precision=precision,
                                  awin='none' if awin == 'uniform' else awin, amask=awin in ('taylor', 'uniform'),
                                  umax=1.0 + float(plan.get('azimuth_pad') or 0.0), backend=backend,
                                  target_db=target_db, info=info)
    # on a grid: the grid, and what forms the collection again on it displaced (stack.register)
    on_grid = {} if gl is None else dict(grid=gl['grid'], shift=gl['shift'], source=cphd, form_options=dict(
        sicd=sicd, mode=mode, backend=backend, window=window, channel=channel, patch=patch,
        azimuth_fraction=azimuth_fraction, precision=precision, target_db=target_db, autofocus=autofocus,
        troposphere=troposphere, reader_options=reader_options, aperture=aperture, azimuth_window=azimuth_window,
        band=band, range_gate=range_gate))
    return dict(image=img, origin=origin, e1=e1, e2=e2, spx=float(spx), spy=float(spy), mode=mode,
                band=bd, bandwidth=(1.0 / ares, 2 * bwid * np.cos(graze) / C),
                window=(('taylor', 'taylor') if window else None) if spot else (
                    (awin, 'taylor' if window else 'uniform') if awin else None),
                phase_error=phi if autofocus else None, meta=meta, notes=notes, plane='ground', axes=axes,
                center=origin + (nx / 2.0) * spx * e1 + (ny / 2.0) * spy * e2, height=float(hz), graze=float(graze),
                pulses=used, dsin=dsin, **on_grid)
