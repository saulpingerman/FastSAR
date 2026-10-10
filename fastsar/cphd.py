"""One call from a CPHD file to an image, for any collection mode FastSAR handles:

    out = fastsar.form_cphd('scene_CPHD.cphd')                      # mode, grid and window chosen from the file
    out = fastsar.form_cphd('scene_CPHD.cphd', sicd='scene_SICD.ntf', backend='cuda')
    img, origin, e1, e2 = out['image'], out['origin'], out['e1'], out['e2']

Mode. A spotlight collection keeps one scene reference point (SRP) for every pulse; stripmap, sliding spotlight and
dynamic stripmap collections move it with the beam. mode='auto' tells them apart by how far the SRP moves (more than
one range resolution: a moving beam). A spotlight is formed as one image by factorized backprojection over the full
aperture (ImageFormer, with the phase reference moved to the grid center); a moving beam is formed as a mosaic of
range-gated patches (patches.form_mosaic), each pixel weighted by a Hann window over the azimuth band the beam
illuminates around it (the beam center of each pulse is its own SRP).

Grid. A ground-plane grid (axes along track and across track, horizontal at the scene) covering the vendor image's
footprint when a SICD is given, else the CPHD's image area (SceneCoordinates.ImageArea). Pixel spacing: the SICD's,
widened to 0.8 of its impulse response width, or 0.8 of the resolution the data support (range: c / 2B over the
cosine of the grazing angle; azimuth: lambda / 2 over the angular span of the aperture, or of the processed band).

Window. Taylor (nbar 4, 35 dB) along frequency, and along pulses for a spotlight (window=False: none).

Azimuth band of a moving beam: the SICD's processed bandwidth (Grid.Col.ImpRespBW) when a SICD is given, else
`azimuth_fraction` (default 0.8) of the Doppler band the PRF samples, 2 v sin(theta) / lambda in [-PRF/2, PRF/2].
"""
import numpy as np
from . import _deps

from . import io

C = 299792458.0


def _taylor(n):
    from scipy.signal.windows import taylor
    return taylor(n, nbar=4, sll=35.0, norm=False).astype(np.float32)


def _sicd_meta(sicd):
    if sicd is None:
        return None
    if not isinstance(sicd, str):
        return sicd
    _deps.require('sarpy')
    if sicd.endswith('.xml'):                 # the SICD's metadata alone (ICEYE publishes it beside the image)
        from sarpy.io.complex.sicd_elements.SICD import SICDType
        with open(sicd) as fh:
            return SICDType.from_xml_string(fh.read())
    from sarpy.io.complex.converter import open_complex
    return open_complex(sicd).sicd_meta


def _image_area(cphd, meta):
    """Corners [4, 3] of the CPHD's image area rectangle and its reference point [3] (local frame), or (None, IARP or
    None)."""
    if 'image_area' in meta:                       # a reader that supplies the footprint itself (read_nisar)
        return meta['image_area'], meta.get('refpt')
    _deps.require('sarpy')
    from sarpy.io.phase_history.converter import open_phase_history
    sc = open_phase_history(cphd).cphd_meta.SceneCoordinates
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
              autofocus=False, troposphere=None):
    """Form the image of a CPHD collection (see the module docstring).

    cphd: path. sicd: the vendor's SICD (path, .xml metadata, or sarpy SICDType) for the footprint, spacing and
    processed azimuth band, or None. mode: 'auto', 'spotlight' or 'moving' (stripmap, sliding spotlight, dynamic
    stripmap). backend: 'auto', 'cpu', 'cuda', 'tpu' or 'jax'. spacing: (along track, across track) in meters, or
    None. extent: (along track, across track) in meters around the scene center, overriding the footprint. height:
    the grid plane's height above the ellipsoid at the scene center (default: the SICD's scene center point, else the
    CPHD's image area reference point; a scatterer at another height appears displaced in range). patch:
    mosaic patch size in pixels. info: a list that receives one dict per mosaic patch. autofocus: phase gradient
    autofocus (fastsar.autofocus.autofocus, two rounds; spotlight only), which forms the image three times.
    troposphere: remove the per-pulse troposphere delay the file gives (io.read_cphd; default: when nonzero);
        'model' removes a standard-atmosphere delay instead, for files that give none (ICEYE).
    -> dict(image [nx, ny] complex64, origin [3] (local, pixel (0, 0)), e1, e2 (unit axes, local), spx, spy, mode,
    band (first and last frequency, Hz), bandwidth (spatial frequency support along e1 and e2, cycles/m), window,
    phase_error (autofocus: per pulse, rad, else None), meta (read_cphd's), notes). Pixel (i, j) lies at
    origin + i spx e1 + j spy e2 in the local frame (io.local_to_ecf for ECF; products.geolocate for latitude and
    longitude)."""
    from .api import ImageFormer, _backend
    from . import patches
    if mode not in ('auto', 'spotlight', 'moving'):
        raise ValueError("mode must be 'auto', 'spotlight' or 'moving'")
    backend = _backend(backend)                       # before the file is read
    for n, v in (('spacing', spacing), ('extent', extent)):
        if v is not None and (np.size(v) not in (1, 2) or not np.all(np.isfinite(v)) or np.min(v) <= 0):
            raise ValueError(f'{n} must be one or two positive lengths in meters, got {v!r}')
    if extent is not None and np.size(extent) != 2:
        raise ValueError(f'extent must be (along track, across track) in meters, got {extent!r}')
    col, meta = io.read_collection(cphd, channel=channel, meta=True, troposphere=troposphere, height=height)
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
    if extent is not None:
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
    ref_r = np.asarray(meta['ref'], np.float64)
    if spot and sm is None and spacing is None:
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
            notes.append(f'aperture: the central {hi_ - lo_} of {P} pulses ({np.degrees(want):.2f} of {np.degrees(np.ptp(ang)):.2f} '
                         'degrees) for square resolution; give spacing= or a sicd for the whole aperture')
            P = hi_ - lo_
    if window:
        wk = _taylor(K)
        wp = _taylor(P) if spot else np.ones(P, np.float32)
        for p0 in range(0, P, 1024):            # in place, row blocks
            S[p0:p0 + 1024] *= wp[p0:p0 + 1024, None] * wk[None, :]
    # azimuth band of a moving beam (spread of sin(look angle) about each pulse's SRP)
    if not spot:
        if sm is not None:
            dsin = float(sm.Grid.Col.ImpRespBW) * lam / 2
        else:
            prf = (P - 1) / float(meta['tx_time'][-1] - meta['tx_time'][0])
            vel = np.linalg.norm(np.gradient(ant, axis=0), axis=1).mean() * prf
            dsin = azimuth_fraction * lam * prf / (2 * vel)
            notes.append(f'azimuth band {azimuth_fraction:.2f} of the PRF ({prf:.0f} Hz): spread of sin(look) {dsin:.4f}')
    # resolution the data support (before windowing)
    if spot:
        u = (center[None] - ant) / np.linalg.norm(center[None] - ant, axis=1, keepdims=True)
        span = float(np.arccos(np.clip(u[0] @ u[-1], -1, 1)))
        ares = lam / (2 * max(span, 1e-6))
    else:
        ares = lam / (2 * dsin)
    # spacing
    if spacing is not None:
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
    if corners is None:
        L1, L2 = map(float, extent)
        a1, a2 = center @ e1 + np.array([-L1, L1]) / 2, center @ e2 + np.array([-L2, L2]) / 2
    else:
        a1, a2 = corners @ e1, corners @ e2
    h = hz
    nx = int(np.ceil((a1.max() - a1.min()) / spx)) + 1
    ny = int(np.ceil((a2.max() - a2.min()) / spy)) + 1
    origin = a1.min() * e1 + a2.min() * e2 + h * up
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
        for p0 in range(0, P, 1024):
            sl = slice(p0, min(P, p0 + 1024))
            S[sl] *= np.exp(-4j * np.pi * f[None, :] / C * dref[sl, None]).astype(np.complex64)
        notes.append(f'phase reference moved {np.linalg.norm(c):.1f} m to the grid center')
        kw = dict(backend=backend, precision=precision, window=False, target_db=target_db)
        if autofocus:
            from .autofocus import autofocus as pga_autofocus
            img, phi = pga_autofocus(S, ant - c, f0, df, nx, ny, spx, spy, e1, e2, **kw)
            notes.append(f'autofocus: removed a phase error of {np.std(phi):.2f} rad rms (constant and linear terms excluded)')
        else:
            img = ImageFormer(ant - c, f0, df, K, nx, ny, spx, spy, e1, e2, **kw)(S)
        origin = c - (nx / 2.0) * spx * e1 - (ny / 2.0) * spy * e2
    else:
        sp = ((srp - ant) * d).sum(1) / np.linalg.norm(srp - ant, axis=1)        # beam center: each pulse's SRP

        def beam(idx, pts):
            w = np.asarray(pts)[None] - ant[idx][:, None]
            return ((w * d[idx][:, None]).sum(-1) / np.linalg.norm(w, axis=-1) - sp[idx][:, None]) / (dsin / 2)

        img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(patch, patch), beam=beam, precision=precision,
                                  awin='hann' if window else None, backend=backend, target_db=target_db, info=info)
    return dict(image=img, origin=origin, e1=e1, e2=e2, spx=float(spx), spy=float(spy), mode=mode,
                band=(f0, f0 + (K - 1) * df), bandwidth=(1.0 / ares, 2 * K * df * np.cos(graze) / C),
                window=('taylor' if spot else 'hann', 'taylor') if window else None,
                phase_error=phi if autofocus else None, meta=meta, notes=notes)
