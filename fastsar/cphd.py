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
    if sicd.endswith('.xml'):                 # the SICD's metadata alone (ICEYE publishes it beside the image)
        from sarpy.io.complex.sicd_elements.SICD import SICDType
        return SICDType.from_xml_string(open(sicd).read())
    from sarpy.io.complex.converter import open_complex
    return open_complex(sicd).sicd_meta


def _image_area(cphd, meta):
    """Corners [4, 3] of the CPHD's image area rectangle and its reference point [3] (local frame), or (None, IARP or
    None)."""
    from sarpy.io.phase_history.converter import open_phase_history
    sc = open_phase_history(cphd).cphd_meta.SceneCoordinates
    ia, iarp = sc.ImageArea, sc.IARP
    ref = None if iarp is None else io.ecf_to_local(np.array(iarp.ECF.get_array(), np.float64)[None], meta)[0]
    if ia is None or iarp is None or sc.ReferenceSurface is None or sc.ReferenceSurface.Planar is None:
        return None, ref
    p0 = np.array(iarp.ECF.get_array(), np.float64)
    ux = np.array(sc.ReferenceSurface.Planar.uIAX.get_array(), np.float64)
    uy = np.array(sc.ReferenceSurface.Planar.uIAY.get_array(), np.float64)
    x1, y1, x2, y2 = ia.X1Y1.X, ia.X1Y1.Y, ia.X2Y2.X, ia.X2Y2.Y
    ecf = np.array([p0 + x * ux + y * uy for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2))])
    return io.ecf_to_local(ecf, meta), ref


def form_cphd(cphd, sicd=None, mode='auto', backend='auto', window=True, spacing=None, channel=0, patch=1024,
              azimuth_fraction=0.8, extent=None, height=None, precision='float32', target_db=-40.0, info=None):
    """Form the image of a CPHD collection (see the module docstring).

    cphd: path. sicd: the vendor's SICD (path, .xml metadata, or sarpy SICDType) for the footprint, spacing and
    processed azimuth band, or None. mode: 'auto', 'spotlight' or 'moving' (stripmap, sliding spotlight, dynamic
    stripmap). backend: 'auto', 'cpu', 'cuda', 'tpu' or 'jax'. spacing: (along track, across track) in metres, or
    None. extent: (along track, across track) in metres around the scene center, overriding the footprint. height:
    the grid plane's height above the ellipsoid at the scene center (default: the SICD's scene center point, else the
    CPHD's image area reference point; a scatterer at another height appears displaced in range). patch:
    mosaic patch size in pixels. info: a list that receives one dict per mosaic patch.
    -> dict(image [nx, ny] complex64, origin [3] (local, pixel (0, 0)), e1, e2 (unit axes, local), spx, spy, mode,
    meta (read_cphd's), notes). Pixel (i, j) lies at origin + i spx e1 + j spy e2 in the local frame
    (io.local_to_ecf for ECF)."""
    from .api import ImageFormer, _backend
    from . import patches
    if mode not in ('auto', 'spotlight', 'moving'):
        raise ValueError("mode must be 'auto', 'spotlight' or 'moving'")
    backend = _backend(backend)                       # before the file is read
    for n, v in (('spacing', spacing), ('extent', extent)):
        if v is not None and (np.size(v) not in (1, 2) or not np.all(np.isfinite(v)) or np.min(v) <= 0):
            raise ValueError(f'{n} must be one or two positive lengths in metres, got {v!r}')
    if extent is not None and np.size(extent) != 2:
        raise ValueError(f'extent must be (along track, across track) in metres, got {extent!r}')
    col, meta = io.read_cphd(cphd, channel=channel, meta=True)
    S = col['S']
    P, K = S.shape
    f0, df = float(col['fmin']), float(col['df'])
    lam = C / (f0 + K / 2 * df)
    ant = np.asarray(col['ant'], np.float64)
    sm = _sicd_meta(sicd)
    notes = list(meta.get('notes', []))
    # the SRP of every pulse (local frame), for the mode and a moving beam's center
    from sarpy.io.phase_history.converter import open_phase_history
    lo, hi = meta.get('pulses', (0, P))
    srp = io.ecf_to_local(open_phase_history(cphd).read_pvp_variable('SRPPos', channel)[lo:hi], meta)
    rres = C / (2 * K * df)
    if mode == 'auto':
        mode = 'moving' if np.linalg.norm(srp.max(0) - srp.min(0)) > rres else 'spotlight'
    spot = mode == 'spotlight'
    if window:
        wk = _taylor(K)
        wp = _taylor(P) if spot else np.ones(P, np.float32)
        for p0 in range(0, P, 1024):            # in place, row blocks
            S[p0:p0 + 1024] *= wp[p0:p0 + 1024, None] * wk[None, :]
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
    # lie metres below the tangent plane), or at the height asked for
    if height is not None:
        lat, lon, _ = (float(np.ravel(v)[0]) for v in io.ecf_to_geodetic(io.local_to_ecf(center[None], meta)[0]))
        hz = float(io.ecf_to_local(io.geodetic_to_ecf(lat, lon, float(height))[None], meta)[0, 2])
    else:
        hz = (refpt if refpt is not None else center)[2]
    center = np.array([center[0], center[1], hz])
    graze = np.arcsin(np.clip(-((center - ant[P // 2]) / np.linalg.norm(center - ant[P // 2])) @ up, 0.05, 1.0))
    # azimuth band of a moving beam (spread of sin(look angle) about each pulse's SRP)
    if not spot:
        if sm is not None:
            dsin = float(sm.Grid.Col.ImpRespBW) * lam / 2
        else:
            prf = (P - 1) / float(meta['tx_time'][-1] - meta['tx_time'][0])
            vel = np.linalg.norm(np.gradient(ant, axis=0), axis=1).mean() * prf
            dsin = azimuth_fraction * lam * prf / (2 * vel)
            notes.append(f'azimuth band {azimuth_fraction:.2f} of the PRF ({prf:.0f} Hz): spread of sin(look) {dsin:.4f}')
    # spacing
    if spacing is not None:
        spx, spy = (float(spacing), float(spacing)) if np.isscalar(spacing) else map(float, spacing)
    elif sm is not None:
        spx = max(float(sm.Grid.Col.SS), 0.8 * float(sm.Grid.Col.ImpRespWid))
        spy = max(float(sm.Grid.Row.SS), 0.8 * float(sm.Grid.Row.ImpRespWid)) / np.cos(np.radians(float(sm.SCPCOA.GrazeAng)))
    else:
        if spot:
            u = (center[None] - ant) / np.linalg.norm(center[None] - ant, axis=1, keepdims=True)
            span = float(np.arccos(np.clip(u[0] @ u[-1], -1, 1)))
            ares = lam / (2 * max(span, 1e-6))
        else:
            ares = lam / (2 * dsin)
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
    fx = dict(S=S, fmin=f0, df=df, ref=np.asarray(meta['ref'], np.float64), band=(f0, f0 + K * df))
    if spot:
        # phase reference to the grid center, a grid centered on it (as ImageFormer takes it)
        c = origin + (nx / 2.0) * spx * e1 + (ny / 2.0) * spy * e2
        tx, rcv = np.asarray(meta['tx']) - c, np.asarray(meta['rcv']) - c
        ref_new = 0.5 * (np.linalg.norm(tx, axis=1) + np.linalg.norm(rcv, axis=1))
        dref = np.asarray(meta['ref'], np.float64) - ref_new
        f = f0 + df * np.arange(K)
        for p0 in range(0, P, 1024):
            sl = slice(p0, min(P, p0 + 1024))
            S[sl] *= np.exp(-4j * np.pi * f[None, :] / C * dref[sl, None]).astype(np.complex64)
        notes.append(f'phase reference moved {np.linalg.norm(c):.1f} m to the grid center')
        former = ImageFormer(ant - c, f0, df, K, nx, ny, spx, spy, e1, e2, backend=backend, precision=precision,
                             window=False, target_db=target_db)
        img = former(S)
        origin = c - (nx / 2.0) * spx * e1 - (ny / 2.0) * spy * e2
    else:
        sp = ((srp - ant) * d).sum(1) / np.linalg.norm(srp - ant, axis=1)        # beam center: each pulse's SRP

        def beam(idx, pts):
            w = np.asarray(pts)[None] - ant[idx][:, None]
            return ((w * d[idx][:, None]).sum(-1) / np.linalg.norm(w, axis=-1) - sp[idx][:, None]) / (dsin / 2)

        img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(patch, patch), beam=beam,
                                  awin='hann' if window else None, backend=backend, target_db=target_db, info=info)
    return dict(image=img, origin=origin, e1=e1, e2=e2, spx=float(spx), spy=float(spy), mode=mode, meta=meta, notes=notes)
