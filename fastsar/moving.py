"""The stages after formation for a moving-beam image (stripmap, sliding spotlight): baseband, autofocus along the
track, apodization, and the quality areas in range.

    out = fastsar.form_cphd('strip_CPHD.cphd', azimuth_window='uniform', window=False)   # ground plane, track axes
    moving.baseband(out)                                              # exact at every pixel, in place
    img, info = moving.autofocus(out['image'], 'quadratic', band=moving.band(out), footprint=...)

Spectrum. form_cphd forms a moving beam on a ground grid with e1 along the track. Pulse p at frequency f adds to
the image around a ground point x the spatial frequency (2 f / c) u_p(x), u the horizontal part of the unit vector
from the antenna to x. The pulses a pixel takes are those whose look angle lies within the processed spread of the
beam center's, |sin_p(x) - s_p| <= dsin / 2, with sin_p(x) = d_p . (x - a_p) / |x - a_p|, d_p the direction of
flight and s_p the same for the pulse's scene reference point (SRP). Along e1 the band is therefore
(2 f / c) [s - dsin / 2, s + dsin / 2] about the beam center's s at the time the beam crosses the pixel: its
center follows the squint, which under a sliding or drifting beam changes along the track, and its width at the
center frequency is 2 dsin / lambda times the ratio of the platform's speed to the footprint's (1 for a stripmap).
The beam center is each pulse's SRP, unless the SRPs move so much slower than the antenna's footprint that their
processed spread takes a pixel through pulses outside the Doppler band the PRF samples about the antenna's beam:
process_cphd then centers it on the antenna's boresight (plan, _steer).
Along e2 it is the range band times the horizontal part of the line of sight, (2 B / c) cos(graze), on the carrier
(2 fc / c) cos(graze), both of which change across the swath.

Baseband. Let F(x, t) = |x - a(t)| + G(t) with G the integral of s |da/dt| over time. F is stationary in t where
the pixel is at the beam center, sin_t(x) = s(t), and psi(x) = (4 pi fc / c) F(x, t_c(x)) at that time has the
gradient (4 pi fc / c) u_{t_c}(x): the center of the pixel's spectrum on both axes. baseband multiplies the image
by exp(-j psi), which centers the spectrum at zero everywhere. a(t) and s(t) |da/dt| are polynomials fitted to the
pulses (a smooth reference: the pulses' own positions stay in the image formation), t_c is solved at knots and read
bilinearly between them, and F is evaluated at every pixel in double; an error dt of t_c changes F only in second
order. For a broadside stripmap psi is 4 pi fc / c times the range of closest approach.

Rectangle. The band along e1 grows with frequency (a pulse's spatial frequency is proportional to f) and, under
squint, its center moves with frequency: on the grid's axes the support is a trapezoid, sheared by the squint.
Apodization needs a rectangle sampled at twice Nyquist. process_cphd forms the image for it with a rectangular
per-pixel weighting wider than the processed band by what the trapezoid and the shear take, and band_limit then
keeps the same band about zero at every pixel, the processed band at the center frequency, with one transform
along e1. What remains off the grid's axes is the shear of the range band, tan(squint) of the azimuth band, and the
change of cos(graze) across the swath; plan refuses apodization beyond the limits SHEAR and RANGE.

Devices. baseband, band_limit and autofocus take the image as a NumPy, CuPy or JAX array and return it in the same
place. The baseband phase is computed in double on the host's cores or in a CUDA kernel. JAX's device works in
float32 (a TPU has no fast float64): the host solves psi in double along the center row and the center column of
the grid, which the device adds as two vectors, and the device computes only what is left, a few radians that
change slowly over the grid (_separable). Every stage crosses JAX's device in blocks sized to its memory
(_xp.jax_block), one compiled program per block shape, and takes an image on the host through the device the same
way (process_cphd's, for an image the device cannot hold).
"""
import numpy as np

from . import _xp

__all__ = ['beam_model', 'center_time', 'spectrum', 'band', 'baseband', 'band_limit', 'autofocus']

C = 299792458.0
SHEAR = 0.10       # largest shear of the range band, as a fraction of it, apodization accepts
RANGE = 0.05       # largest change of the ground range band from its value at the swath center
VARY = 0.10        # largest spread of the processed azimuth band over the scene
ALONG = 5.0        # degrees between a caller's grid's first axis and the track within which the stages apply
PAD = 0.06         # the rectangular weighting formed beyond the band kept: the soft edge of the per-pixel weights
_KNOTS = 32        # pixels between the knots of the beam-center time


def _poly(c, x):
    """Polynomials with coefficients c [..., n] (lowest first) at x -> [..., *x.shape]"""
    c = np.asarray(c, np.float64)
    out = np.zeros(c.shape[:-1] + np.shape(x)) + c[..., -1].reshape(c.shape[:-1] + (1,) * np.ndim(x))
    for k in range(c.shape[-1] - 2, -1, -1):
        out = out * x + c[..., k].reshape(c.shape[:-1] + (1,) * np.ndim(x))
    return out


def beam_model(ant, srp, t, dsin, sine=None):
    """A smooth model of a moving beam, from the pulses: antenna positions ant [P, 3] and scene reference points
    srp [P, 3] (local frame, m), times t [P] (s), dsin the processed spread of sin(look) about the beam center.
    sine [P]: the beam center's sine of look from broadside at each pulse in place of the SRP's (antenna_sine's,
    where the antenna's boresight is the beam center: plan).

    -> dict(t0, th: the time x = (t - t0) / th runs from -1 to 1 over the pulses; A [3, n]: the position as
    polynomials in x (lowest power first); dA, ddA: their derivatives; Q: s |da/dx| as a polynomial, s the sine of
    the look angle of the beam center (the SRP) from broadside, forward positive; dQ, G: its derivative and
    integral; d [P, 3] and s [P]: the pulses' own
    direction of flight and sine, as form_cphd's weighting takes them; dsin; fit: the largest distance of a pulse
    from the fitted track (m); drift: the largest departure of s from the model after a running mean over an
    eighth of an aperture, as a fraction of dsin)."""
    ant, srp = np.asarray(ant, np.float64), np.asarray(srp, np.float64)
    t = np.asarray(t, np.float64)
    P = len(ant)
    t0, th = 0.5 * (t[0] + t[-1]), max(0.5 * (t[-1] - t[0]), 1e-12)
    x = (t - t0) / th
    pf = np.polynomial.polynomial
    d = np.gradient(ant, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = srp - ant
    R = np.linalg.norm(w, axis=1)
    s = np.einsum('ij,ij->i', w, d) / R if sine is None else np.asarray(sine, np.float64)
    reach = float(np.median(R)) * dsin                               # the length of a synthetic aperture
    for deg in range(min(5, P - 1), min(11, P - 1) + 1, 2):
        A = pf.polyfit(x, ant, deg).T                                # [3, deg + 1]
        fit = float(np.linalg.norm(_poly(A, x).T - ant, axis=1).max())
        if fit <= 1e-3 * reach:
            break
    dA = pf.polyder(A, axis=1)
    speed = np.linalg.norm(_poly(dA, x), axis=0)
    n = max(1, int(reach / max(np.median(np.linalg.norm(np.diff(ant, axis=0), axis=1)), 1e-9) / 8))
    box = lambda v: np.convolve(np.pad(v, n, mode='edge'), np.ones(2 * n + 1) / (2 * n + 1), 'valid')
    for deg in range(min(5, P - 1), min(11, P - 1) + 1, 2):
        Q = pf.polyfit(x, s * speed, deg)
        drift = float(np.abs(box(s - _poly(Q, x) / speed)).max() / dsin)
        if drift <= 0.005:
            break
    return dict(t0=float(t0), th=float(th), A=A, dA=dA, ddA=pf.polyder(dA, axis=1), Q=Q, dQ=pf.polyder(Q),
                G=pf.polyint(Q), d=d, s=s, dsin=float(dsin), fit=fit, drift=drift, x=x, flight=d[P // 2])


def _slope(bm, X, x):
    """dF/dx and d2F/dx2 of F(x) = |X - a(x)| + G(x) at the points X [n, 3] and times x [n]."""
    w = X - _poly(bm['A'], x).T
    v, acc = _poly(bm['dA'], x).T, _poly(bm['ddA'], x).T
    r = np.linalg.norm(w, axis=1)
    wv = np.einsum('ij,ij->i', w, v)
    return (-wv / r + _poly(bm['Q'], x),
            (np.einsum('ij,ij->i', v, v) - np.einsum('ij,ij->i', w, acc)) / r - wv * wv / r ** 3 + _poly(bm['dQ'], x))


def center_time(bm, X, iterations=12, reach=1.3):
    """The time x (the model's, -1 to 1 over the pulses) at which the beam center crosses each of the points X
    [..., 3]: the stationary point of F, by Newton's iteration from the time of closest approach, held within
    +-reach (a pixel the beam center never crosses takes the nearest end)."""
    X = np.asarray(X, np.float64)
    q = X.reshape(-1, 3)
    P = len(bm['x'])
    k = np.unique(np.round(np.linspace(0, P - 1, min(P, 4097))).astype(int))
    along = _poly(bm['A'], bm['x'][k]).T @ bm['flight']
    x = np.interp(q @ bm['flight'], along, bm['x'][k])
    for _ in range(iterations):
        f, df = _slope(bm, q, x)
        new = np.clip(x - f / np.where(np.abs(df) > 1e-12, df, 1e-12), -reach, reach)
        done = np.abs(new - x).max() < 1e-13
        x = new
        if done:
            break
    return x.reshape(X.shape[:-1])


def spectrum(bm, ant, band, K, points, e1, e2, umax=1.0):
    """The spectrum of the image as formed at ground points [n, 3], from the pulses each takes (|u| <= umax of
    form_cphd's weighting): dict of arrays [n]: pulses (their number), first (the first of them), full (the band is
    not cut by an end of the collection), k1c, k2c (the center along e1 and e2 at the center frequency: the
    baseband phase's gradient, cycles/m), b1 (the width along e1 at the center frequency), lo, hi (the least and
    greatest offset along e1
    from the center over the band's frequencies that every frequency holds: the rectangle inside the trapezoid),
    b2 (the width along e2: K frequency steps times the horizontal part of the line of sight), shear (the slope
    of the range band's center against the azimuth offset, tan of the squint on the ground), speed (the distance
    along e1 the beam center moves per pulse, m).
    ant [P, 3]: the pulses' positions; band: first and last frequency (Hz); K: frequency samples."""
    pts = np.atleast_2d(np.asarray(points, np.float64))
    ant = np.asarray(ant, np.float64)
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    f = np.array([band[0], 0.5 * (band[0] + band[1]), band[1]])
    df = (band[1] - band[0]) / max(K - 1, 1)
    out = {k: np.zeros(len(pts)) for k in ('pulses', 'first', 'k1c', 'k2c', 'b1', 'lo', 'hi', 'b2', 'shear', 'speed')}
    out['full'] = np.zeros(len(pts), bool)
    xc = center_time(bm, pts)
    ac = _poly(bm['A'], xc).T
    uc = pts - ac
    uc /= np.linalg.norm(uc, axis=1, keepdims=True)
    out['k1c'], out['k2c'] = 2 * f[1] / C * (uc @ e1), 2 * f[1] / C * (uc @ e2)
    out['b2'] = 2 * K * df / C * np.abs(uc @ e2)
    out['shear'] = (uc @ e1) / (uc @ e2)
    h = 1.0
    dx = (center_time(bm, pts + h * e1) - center_time(bm, pts - h * e1)) / (2 * h)       # model time per meter
    out['speed'] = 1.0 / np.maximum(np.abs(dx) * (len(ant) - 1) / 2.0, 1e-12)
    for n, x in enumerate(pts):
        w = x[None] - ant
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        on = np.flatnonzero(np.abs((np.einsum('ij,ij->i', w, bm['d']) - bm['s']) / (bm['dsin'] / 2)) <= umax)
        if len(on) < 2:
            continue
        lo, hi = on[0], on[-1]
        k1 = np.sort(2 / C * np.outer(f, w[[lo, hi]] @ e1), axis=1) - out['k1c'][n]              # [3, 2]
        out['pulses'][n], out['first'][n], out['full'][n] = hi - lo + 1, lo, lo > 0 and hi < len(ant) - 1
        out['b1'][n], out['lo'][n], out['hi'][n] = k1[1, 1] - k1[1, 0], k1[:, 0].max(), k1[:, 1].min()
    return out


def band(out):
    """(f0, hf) of a baseband moving-beam product for autofocus: center 0 and half width (cycles/pixel) of the
    processed azimuth band, from the product's bandwidth entry."""
    return 0.0, min(0.5, 0.5 * float(out['bandwidth'][0]) * float(out['spx']))


def antenna_sine(cphd, meta, ant, pulses=None, channel=0):
    """The sine of the look angle from broadside of the antenna's two-way electrical boresight at each pulse, as
    form_cphd's weighting takes the SRP's (d . u, d the pulse's direction of flight, u the mean of the transmit and
    receive boresights), from the antenna block of a CPHD 1.x file (rgiqe.antenna_frames), or None where the
    file has none or its boresight does not point at the scene. meta: read_cphd's; ant [P, 3]: the positions of
    the pulses used (local frame), pulses (first, last + 1) of those read (None: all)."""
    from . import rgiqe
    from .io import _open_cphd
    if 'cphd_version' not in meta:                                  # a raw collection's reader: no antenna block
        return None
    r = _open_cphd(str(cphd))
    m = r.cphd_meta
    ch = int(meta.get('channel_index', channel))
    try:
        par = rgiqe._parameters(m, meta, ch)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):    # an antenna block that cannot be read
        return None
    if par is None or getattr(m, 'Antenna', None) is None:
        return None
    pv = r.read_pvp_array(ch)
    lo, hi = meta.get('pulses', (0, len(meta['tx'])))
    sel = slice(*(pulses if pulses is not None else (0, hi - lo)))
    fi = np.arange(lo, hi)[sel]
    t_tx = np.asarray(meta['tx_time'], np.float64)[sel]
    t_rcv = np.asarray(meta['rcv_time'], np.float64)[sel] if 'rcv_time' in meta else t_tx
    ant = np.asarray(ant, np.float64)
    if len(fi) != len(ant):
        return None
    f = rgiqe.antenna_frames(m, par, pv, fi, t_tx, t_rcv, meta['R'])
    if f is None:
        return None
    w = np.asarray(meta['srp_pulses'], np.float64)[sel] - ant
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    u = 0.0
    for k in ('tx', 'rcv'):
        _, ax, ay, eb, _ = f[k]
        z = np.cross(ax, ay)
        z *= np.where(np.einsum('ij,ij->i', z, w) < 0, -1.0, 1.0)[:, None]      # toward the scene, either hand
        b = eb[:, :1] * ax + eb[:, 1:] * ay + np.sqrt(np.clip(1.0 - (eb ** 2).sum(1), 0.0, 1.0))[:, None] * z
        u = u + b / np.linalg.norm(b, axis=1, keepdims=True)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    if not np.all(np.isfinite(u)) or np.median(np.einsum('ij,ij->i', u, w)) < 0.99:
        return None                                                 # not toward the scene: frames read otherwise
    d = np.gradient(ant, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.einsum('ij,ij->i', u, d)


def _steer(g, bm, b, opt, lam):
    """The beam center of a process_cphd product: the SRP of each pulse (bm, beam_model's), or the antenna's
    boresight (antenna_sine) where the SRPs' processed spread reaches pulses on which a point of the tasked
    rectangle b lies outside the Doppler band the PRF samples about the antenna's beam, sin(look) within
    +-lambda PRF / (4 v) of the boresight's. Such pulses hold no echo of the point that is not ambiguous with
    others: under a sliding spotlight whose SRPs move slower than the antenna's footprint (a Capella C10 sliding
    spotlight: 0.09 of the speed a fixed point's look moves at, against the boresight's 0.16) the spread of the
    SRPs takes a pixel through pulses well outside the beam, which add noise and no band. The antenna's track is
    taken only where it reaches every point and is as smooth as the beam model needs, and only for a processed
    spread the PRF samples (not a vendor's band wider than that). -> (beam model, the boresight's sine [P] or
    None, note or None)."""
    ant, dsin = np.asarray(g['ant'], np.float64), float(g['dsin'])
    if g.get('grid') is not None or 'cphd_version' not in g['meta']:
        return bm, None, None
    half = lam / (4 * np.linalg.norm(np.gradient(ant, axis=0), axis=1).mean())      # lambda PRF / (4 v)
    if dsin / 2 > half:
        return bm, None, None
    bs = antenna_sine(opt['cphd'], g['meta'], ant, g['pulses'], opt['channel'])
    if bs is None:
        return bm, None, None
    c, e1, e2 = g['center'], g['e1'], g['e2']
    pts = c + (np.linspace(b[0], b[1], 7)[:, None, None] * e1 + np.linspace(b[2], b[3], 5)[None, :, None] * e2
               ).reshape(-1, 3)
    reach, seen = 0.0, True
    for x in pts:
        w = x[None] - ant
        sx = np.einsum('ij,ij->i', w, bm['d']) / np.linalg.norm(w, axis=1)
        on = np.abs(sx - bm['s']) <= dsin / 2
        if on.any():
            reach = max(reach, float(np.abs(sx[on] - bs[on]).max()))
        seen = seen and int((np.abs(sx - bs) <= dsin / 2).sum()) > 1
    if reach <= half:
        return bm, None, None
    if not seen:
        return bm, None, ('beam center: the SRPs of the pulses take the tasked area beyond the Doppler band the PRF '
                          "samples about the antenna's boresight, which does not reach all of it: the SRPs are kept")
    am = beam_model(ant, g['srp'], g['meta']['tx_time'], dsin, sine=bs)
    if am['drift'] > bm['drift'] and am['drift'] > 0.05:
        return bm, None, ("beam center: the SRPs of the pulses take the tasked area beyond the Doppler band the PRF "
                          "samples about the antenna's boresight, which does not move smoothly: the SRPs are kept")
    k = lambda m: 2.0 / max(abs(float(center_time(m, (c + e1)[None])[0] - center_time(m, (c - e1)[None])[0]))
                            * m['th'], 1e-12)                       # the beam center's speed along e1 (m/s)
    note = (f"beam center: the antenna's boresight (the file's antenna block), not the SRPs of the pulses. The "
            f'SRPs move {k(bm):.0f} m/s along the track at the scene center and the boresight {k(am):.0f} m/s, '
            f'so the processed spread about the SRPs reaches pulses on which the scene lies up to {reach:.4f} '
            f'from the boresight in sin(look), beyond the {half:.4f} the PRF samples: they hold no echo of it')
    return am, bs, note


def _model(out):
    """The beam model of a form_cphd product (kept in the product once made)."""
    if out.get('beam') is None:
        if out.get('dsin') is None:
            raise ValueError('baseband of a moving beam takes a form_cphd product of a moving-beam collection')
        meta = out['meta']
        ant = 0.5 * (np.asarray(meta['tx'], np.float64) + np.asarray(meta['rcv'], np.float64))
        out['beam'] = beam_model(ant, meta['srp_pulses'], meta['tx_time'], out['dsin'])
    return out['beam']


def _lattice(bm, o, a, b, nx, ny):
    """The beam-center time at knots every _KNOTS pixels of the grid o + i a + j b: (times [ni, nj], pixels per
    knot interval along i, along j)."""
    ni, nj = (max(2, int(np.ceil((n - 1) / _KNOTS)) + 1) for n in (nx, ny))
    si, sj = max(nx - 1, 1) / (ni - 1), max(ny - 1, 1) / (nj - 1)
    X = o + (np.arange(ni) * si)[:, None, None] * a + (np.arange(nj) * sj)[None, :, None] * b
    return center_time(bm, X), si, sj


def _path(bm, tk, si, sj, o, a, b, rows, ny):
    """F = |x - a(t)| + G(t) at the pixels of rows `rows` (a slice): [len, ny] float64, t read bilinearly from the
    knots tk."""
    i = np.arange(rows.start, rows.stop, dtype=np.float64)
    j = np.arange(ny, dtype=np.float64)
    i0 = np.clip((i / si).astype(np.intp), 0, tk.shape[0] - 2)
    j0 = np.clip((j / sj).astype(np.intp), 0, tk.shape[1] - 2)
    wi, wj = (i / si - i0)[:, None], (j / sj - j0)[None, :]
    r0 = tk[i0] * (1 - wi) + tk[i0 + 1] * wi                         # [len, nj]
    t = r0[:, j0] * (1 - wj) + r0[:, j0 + 1] * wj
    A, F = bm['A'], _poly(bm['G'], t)
    d2 = 0.0
    for k in range(3):
        w = (o[k] + i[:, None] * a[k] + j[None, :] * b[k]) - _poly(A[k], t)
        d2 = d2 + w * w
    F += np.sqrt(d2)
    return F


_KERNEL = []


def _cuda_phase(img, row0, bm, tk, si, sj, o, a, b, k, F0):
    """img [n, ny] (a CuPy array: rows row0 ... of the image) *= exp(j k (F - F0)), F as _path's, in double."""
    import cupy as cp
    if not _KERNEL:
        _KERNEL.append(cp.ElementwiseKernel(
            'int64 ny, int64 row0, int32 ni, int32 nj, float64 si, float64 sj, raw float64 tk, int32 na, int32 ng, '
            'raw float64 pc, raw float64 geo, float64 k, float64 f0', 'T v',
            '''const double r = (double)(i / ny + row0), c = (double)(i % ny);
            const double fi = r / si, fj = c / sj;
            const int i0 = min(max((int)fi, 0), ni - 2), j0 = min(max((int)fj, 0), nj - 2);
            const double wi = fi - i0, wj = fj - j0;
            const long long q0 = (long long)i0 * nj + j0, q1 = q0 + nj;
            const double t = (tk[q0] * (1.0 - wi) + tk[q1] * wi) * (1.0 - wj)
                             + (tk[q0 + 1] * (1.0 - wi) + tk[q1 + 1] * wi) * wj;
            double px = 0.0, py = 0.0, pz = 0.0, g = 0.0;
            for (int m = na - 1; m >= 0; --m) {
                px = px * t + pc[m]; py = py * t + pc[na + m]; pz = pz * t + pc[2 * na + m];
            }
            for (int m = ng - 1; m >= 0; --m) g = g * t + pc[3 * na + m];
            const double x = geo[0] + r * geo[3] + c * geo[6] - px, y = geo[1] + r * geo[4] + c * geo[7] - py,
                         z = geo[2] + r * geo[5] + c * geo[8] - pz;
            double sn, cs;
            sincos(k * (sqrt(x * x + y * y + z * z) + g - f0), &sn, &cs);
            const double re = v.real(), im = v.imag();
            v = T(re * cs - im * sn, re * sn + im * cs);''', 'fastsar_moving_baseband'))
    A, G = bm['A'], bm['G']
    _KERNEL[0](np.int64(img.shape[1]), np.int64(row0), np.int32(tk.shape[0]), np.int32(tk.shape[1]), float(si),
               float(sj), cp.asarray(tk), np.int32(A.shape[1]), np.int32(len(G)),
               cp.asarray(np.concatenate([A.ravel(), G])), cp.asarray(np.concatenate([o, a, b])), float(k), float(F0),
               img)


def _exact(bm, X):
    """F = |x - a(t)| + G(t) at the points X [..., 3], t the beam-center time solved at each one (double)."""
    t = center_time(bm, X)
    w = np.asarray(X, np.float64).reshape(-1, 3) - _poly(bm['A'], t.ravel()).T
    return np.linalg.norm(w, axis=1).reshape(t.shape) + _poly(bm['G'], t)


_TERMS = 24                   # most separable terms of the remainder that a JAX device adds (_separable)


def _separable(bm, o, a, b, nx, ny, k, F0, tol=2e-4):
    """psi(i, j) = k (F - F0) on the grid o + i a + j b in parts that float32 holds, for JAX's device:

        psi = p1[i] + p2[j] + 2 pi theta (i - ic) (j - jc) + sum_m U[i, m] V[m, j].

    p1 and p2 are psi along the center column and the center row, solved at every pixel in double and reduced
    modulo 2 pi. What is left vanishes on both and is small and smooth (zero for a straight track and a constant
    squint; some hundred radians over 80 km of an orbit): its part in i j is theta turns per row and column, which
    the device takes in integers as resample_jax.deramp does, and the rest is solved at knots, split into its
    singular vectors and carried to the pixels by a cubic spline of each. The knots are halved (twice at most)
    until the sum gives psi at the pixels between them to tol rad.
    -> (rows: p1 [nx] float32, theta (i - ic) turns less their nearest 1/4096 [nx] float32, that 1/4096 as an
    integer [nx] int32, U [nx, m] float32; cols: p2 [ny] float32, j - jc [ny] int32, V [m, ny] float32), or None
    where the rest is not smooth (pixels the beam center never crosses, whose time is held at its limit)."""
    from scipy.interpolate import make_interp_spline
    if min(nx, ny) < 4:
        return None
    ic, jc = nx // 2, ny // 2
    i, j = np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64)
    psi = lambda fi, fj: k * (_exact(bm, o + np.asarray(fi, np.float64)[..., None] * a
                                     + np.asarray(fj, np.float64)[..., None] * b) - F0)
    p1, p2 = psi(i, float(jc)), psi(float(ic), j)
    for step in (_KNOTS, _KNOTS // 2, _KNOTS // 4):
        ki, kj = (np.linspace(0, n - 1, max(4, int(np.ceil((n - 1) / step)) + 1)) for n in (nx, ny))
        r = psi(ki[:, None], kj[None, :]) - psi(ki, float(jc))[:, None] - psi(float(ic), kj)[None, :]
        di, dj = ki - ic, kj - jc
        theta = float(di @ r @ dj / ((di @ di) * (dj @ dj))) if ny < 1 << 20 else 0.0
        r -= theta * di[:, None] * dj[None, :]
        U, sv, Vt = np.linalg.svd(r, full_matrices=False)
        left = np.cumsum((sv * np.abs(U).max(0) * np.abs(Vt).max(1))[::-1])[::-1]       # the terms from each one on
        m = max(1, int((left > 0.05 * tol).sum()))
        if m > _TERMS:
            return None
        Ui = make_interp_spline(ki, U[:, :m] * sv[:m], k=3, axis=0)(i)
        Vj = make_interp_spline(kj, Vt[:m].T, k=3, axis=0)(j).T
        qi, qj = (np.unique(np.round(0.5 * (v[:-1] + v[1:])).astype(np.int64)) for v in (ki, kj))
        got = p1[qi][:, None] + p2[qj][None, :] + theta * np.outer(qi - ic, qj - jc) + Ui[qi] @ Vj[:, qj]
        if np.abs(psi(qi[:, None], qj[None, :]) - got).max() <= tol:
            break
    else:
        return None
    wrap = lambda v: (np.remainder(v + np.pi, 2 * np.pi) - np.pi).astype(np.float32)
    ti = theta / (2 * np.pi) * (i - ic)
    ti -= np.rint(ti)
    hi = np.rint(ti * 4096)
    return ((wrap(p1), (ti - hi / 4096).astype(np.float32), hi.astype(np.int32), Ui.astype(np.float32)),
            (wrap(p2), (j - jc).astype(np.int32), Vj.astype(np.float32)))


def _baseband_step(z, a, rows, cols):
    """Rows a ... a + len(z) of an image times exp(j psi), psi in _separable's parts, all float32 and int32."""
    import jax.numpy as jnp
    from jax import lax
    nb = z.shape[0]
    p1, li, hi = (lax.dynamic_slice(v, (a,), (nb,))[:, None] for v in rows[:3])
    U = lax.dynamic_slice(rows[3], (a, 0), (nb, rows[3].shape[1]))
    p2, jr, V = cols[0][None, :], cols[1][None, :], cols[2]
    turns = ((hi * jr) & 4095).astype(jnp.float32) * (1 / 4096) + li * jr.astype(jnp.float32)
    ang = p1 + p2 + (2 * np.pi) * turns
    for m in range(V.shape[0]):                                      # not a matrix product: exact float32
        ang = ang + U[:, m][:, None] * V[m][None, :]
    return z * lax.complex(jnp.cos(ang), jnp.sin(ang))


def _baseband_jax(img, rows, cols):
    """img [nx, ny] times exp(j psi) on JAX's device, in blocks of rows sized to its memory through one compiled
    program: a device image -> the new array; an image on the host (complex64) crosses block by block and is
    written in place. img: a list that holds the image alone (resample_jax._source's: a device image is freed once
    its planes exist)."""
    import jax.numpy as jnp
    from . import resample_jax as rj
    nx, ny = img[0].shape
    host = _xp.kind(img[0]) != 'jax'
    rows, cols = tuple(jnp.asarray(v) for v in rows), tuple(jnp.asarray(v) for v in cols)
    rj._fits(64 * ny, 'the baseband step on the JAX device')
    nb = int(max(1, min(nx, rj._samples(rj.BLOCK, 64) // ny)))
    sink = rj._Sink('device', nx, ny, nb, 1, np.complex64, into=img[0] if host else None)
    step, lines = rj._jit(_baseband_step, ()), rj._source(img)
    # a host image is written where it is read: no block moved back over rows already done
    for s0 in (range(0, nx, nb) if host else rj._starts(nx, nb)):
        sink.put(s0, step(lines(s0, min(nb, nx - s0)), np.int32(s0), rows, cols))
    return sink.done()


def baseband(out, sign=-1, *, _jax=False):
    """Bring a moving-beam image (form_cphd's product: ground plane, axes on the track) to baseband, in place:
    multiplied by exp(sign j psi), psi the module docstring's phase less its value at the grid's center pixel.
    out['baseband'] records it; sign=+1 puts the carrier back. The image may be a NumPy, CuPy or JAX array; a JAX
    image is replaced in out by the new array (float32 on the device, to under 1e-3 rad: _separable). _jax: an
    image on the host crosses JAX's device in blocks (process_cphd's, for an image the device cannot hold)."""
    from .patches import _rows
    if bool(out.get('baseband')) == (sign < 0):
        return out
    bm = _model(out)
    img = out['image']
    nx, ny = img.shape
    o, e1, e2 = (np.asarray(out[k], np.float64) for k in ('origin', 'e1', 'e2'))
    a, b = float(out['spx']) * e1, float(out['spy']) * e2
    k = sign * 4 * np.pi * 0.5 * (out['band'][0] + out['band'][1]) / C
    c = o + (nx // 2) * a + (ny // 2) * b
    F0 = float(_exact(bm, c[None])[0])
    kind = _xp.kind(img)
    sep = None
    if kind == 'jax' or (_jax and kind == 'numpy' and img.dtype == np.complex64):
        sep = _separable(bm, o, a, b, nx, ny, k, F0)
    if sep is not None:                                              # the image is given up: freed as blocks are made
        out['image'], img = None, [img]
        out['image'] = _baseband_jax(img, *sep)
        out['baseband'] = sign < 0
        return out
    tk, si, sj = _lattice(bm, o, a, b, nx, ny)
    if kind == 'cupy':
        _cuda_phase(img, 0, bm, tk, si, sj, o, a, b, k, F0)
    elif kind == 'jax':                                              # no smooth remainder: the phasor from the host
        import jax
        import jax.numpy as jnp
        from jax import lax
        from . import resample_jax as rj
        nb = int(max(1, min(nx, _xp.jax_rows(nx, 64 * ny, 'the baseband step on the JAX device'))))
        lines, sink = rj._source([img]), rj._Sink('device', nx, ny, nb, 1, np.complex64)
        out['image'] = img = None
        if baseband not in rj._JIT:
            rj._JIT[baseband] = jax.jit(lambda z, cs, sn: z * lax.complex(cs, sn))
        for s in rj._starts(nx, nb):
            cs, sn = np.empty((nb, ny), np.float32), np.empty((nb, ny), np.float32)

            def work(sl):
                ph = k * (_path(bm, tk, si, sj, o, a, b, slice(s + sl.start, s + sl.stop), ny) - F0)
                cs[sl], sn[sl] = np.cos(ph), np.sin(ph)
            _rows(work, nb, int(max(1, min(64, (1 << 17) // max(ny, 1)))))
            sink.put(s, rj._JIT[baseband](lines(s, nb), jnp.asarray(cs), jnp.asarray(sn)))
        out['image'] = sink.done()
    else:
        def work(sl):
            img[sl] *= np.exp(1j * (k * (_path(bm, tk, si, sj, o, a, b, sl, ny) - F0)))
        _rows(work, nx, int(max(1, min(64, (1 << 17) // max(ny, 1)))))
    out['baseband'] = sign < 0
    return out


def band_limit(img, keep):
    """img [nx, ny] with only the azimuth frequencies kept that `keep` [nx] (FFT order along axis 0; 0 or 1, or any
    weights) holds: ifft(fft(img, axis=0) * keep[:, None], axis=0) in blocks of columns on the device of img, exact
    zeros kept. With nx a multiple of four and keep the nx / 2 bins about zero, a baseband image at twice Nyquist
    of that band becomes one whose azimuth spectrum is exactly that rectangle. -> complex64, where img is."""
    from .autofocus import _apply
    return _apply(img, None, np.asarray(keep, np.complex64))


# --------------------------------------------------------------------------------------------- autofocus along track
def _local_fit(x, y, z, x0, h, tol, across):
    """Robust local fit of z at points (x along track, y across) around x0: tricube weights of half width h in
    x, a line in x where x0 lies among the points (else a constant: no extrapolation), powers of y up to `across`
    (0, 1 or 2, as far as the points' distinct y allow; 'auto': a linear term kept when it is three times its
    standard error and at least tol across the swath), and Tukey's biweight on the residuals scaled by their
    median absolute deviation (at least tol). -> (value at (x0, y = 0), coefficient of y, of y^2)."""
    k = np.clip(1 - np.abs((x - x0) / h) ** 3, 0, None) ** 3
    use = k > 0
    x, y, z, k = (x[use] - x0) / h, y[use], z[use], k[use]
    cols = [np.ones_like(x)]
    if x.min() <= 0 <= x.max() and len(np.unique(x)) >= 2:
        cols.append(x)
    ny = len(np.unique(np.round(y, 6)))
    order = 1 if across == 'auto' else int(across)
    order = min(order, 2 if ny >= 5 else 1 if ny >= 3 else 0)
    for o in ([order, 0] if across == 'auto' and order else [order]):
        A = np.stack(cols + [y ** n for n in range(1, o + 1)], 1)
        rw = np.ones(len(z))
        for _ in range(8):
            sw = np.sqrt(k * rw)
            coef = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)[0]
            res = z - A @ coef
            scale = max(1.4826 * float(np.median(np.abs(res))), tol)
            rw = np.clip(1 - (res / (4.685 * scale)) ** 2, 0, None) ** 2
        yc = np.concatenate([coef[len(cols):], np.zeros(2)])[:2]
        if across != 'auto' or o == 0:
            return float(coef[0]), float(yc[0]), float(yc[1])
        wt = k * rw / max((k * rw).max(), 1e-300)
        var = float((wt * res ** 2).sum() / max(wt.sum() - A.shape[1], 1.0))
        se = np.sqrt(var * np.linalg.pinv((A * wt[:, None]).T @ A)[-1, -1])
        if abs(coef[-1]) >= 3 * se and abs(coef[-1]) >= tol:
            return float(coef[0]), float(yc[0]), 0.0


def _modes(mode):
    if mode in ('pga', 'spatially_variant_pga'):
        raise ValueError(f'autofocus={mode!r} is for spotlight collections: phase gradient autofocus estimates one '
                         'phase function from an image formed from one aperture, and every block of a moving-beam '
                         "image has its own; use 'quadratic' or 'spatially_variant'")
    if mode not in ('quadratic', 'spatially_variant'):
        raise ValueError(f"autofocus of a moving beam: mode must be 'quadratic' or 'spatially_variant', got {mode!r}")


_JIT = {}
_CELL = (128, 32)             # least rows and columns of a part of the image the guard measures the contrast of
_TAYLOR = (4, 35.0)           # the Taylor window (nbar, sidelobe level in dB) of the copy the estimate is made on
_SETTLED = 10.0               # a block's iteration has settled when its last update is under this many tol


def _cellsums(img, h, cuts):
    """The guard's sums over cells of img [nx, ny], h rows by the sets of columns that start at cuts [c]: the number
    of nonzero samples, the sum of their intensity and the sum of its square -> host float64 [nx // h, 3, c] (rows
    beyond the last whole cell are left out). Computed where the image is, in blocks of whole cells: on the host
    one thread a cell, on a GPU blocks of 2^27 samples, on JAX's device blocks sized to its memory through one
    compiled program (an image on the host under autofocus._via_jax crosses it block by block)."""
    from . import autofocus as af
    nx, ny = img.shape
    R, kind = nx // h, _xp.kind(img)
    out = np.zeros((R, 3, len(cuts)))
    if kind == 'jax' or af._VIA:
        import jax
        import jax.numpy as jnp
        from . import resample_jax as rj
        if 'cells' not in _JIT:
            def cells(blk, h):
                p = jnp.real(blk) ** 2 + jnp.imag(blk) ** 2
                p = p.reshape(blk.shape[0] // h, h, blk.shape[1])
                return (p > 0).sum(1), p.sum(1), (p * p).sum(1)
            _JIT['cells'] = jax.jit(cells, static_argnums=1)
        g = int(max(1, min(R, _xp.jax_rows(R * h, 24 * ny, 'the guard of autofocus on the JAX device') // h)))
        lines = rj._source(img)
        for r0 in rj._starts(R, g):
            t = [np.asarray(v, np.float64) for v in _JIT['cells'](lines(r0 * h, g * h), h)]
            out[r0:r0 + g] = np.stack([np.add.reduceat(v, cuts, axis=1) for v in t], 1)
        return out
    xp = _xp.module(img)
    g = int(max(1, (1 << 27) // max(h * ny, 1))) if kind == 'cupy' else 1

    def work(r0):
        blk = img[r0 * h:min(R, r0 + g) * h]
        p = (blk.real ** 2 + blk.imag ** 2).reshape(-1, h, ny)
        t = (xp.count_nonzero(p, axis=1), p.sum(1, dtype=xp.float64), (p * p).sum(1, dtype=xp.float64))
        out[r0:r0 + g] = np.stack([np.add.reduceat(_xp.host(v).astype(np.float64), cuts, axis=1) for v in t], 1)
    if kind == 'cupy':
        for r0 in range(0, R, g):
            work(r0)
    else:
        af._threads(work, range(R))
    return out


def _strips_jax(img, st, B, pad, win, V, coef):
    """autofocus._apply_strips for a moving beam on JAX's device, where a strip's phase is coef[k, j] V[b] in
    azimuth bin b of column j: strip k is rows st[k] - pad to st[k] + B + pad (zeros beyond the image), its
    transform along the track loses that phase, the zeros of img are put back, and its middle B rows are added to
    the result with the weights win[k] [B]. The image crosses the device in blocks of columns sized to its memory
    (_xp.jax_block), each through one compiled program that loops over the strips: an image on the device -> a
    device array (the blocks joined); one on the host (under autofocus._via_jax) -> a host array, written block by
    block. st [ns], win [ns, B], V [B + 2 pad], coef [ns, ny] (float32)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from . import autofocus as af
    nx, ny = img.shape
    n = B + 2 * pad
    if 'strips' not in _JIT:
        def strips(blk, st, coef, win, V, pad):
            blk = blk.astype(jnp.complex64)
            nx, n, B = blk.shape[0], V.shape[0], win.shape[1]
            rows = jnp.arange(n, dtype=jnp.int32) - pad

            def body(k, acc):
                idx = st[k] + rows                                   # the strip's rows; beyond the image: zeros
                s = jnp.take(blk, jnp.clip(idx, 0, nx - 1), axis=0)
                s = jnp.where(((idx >= 0) & (idx < nx))[:, None], s, 0)
                ph = V[:, None] * coef[k][None, :]
                t = jnp.fft.ifft(jnp.fft.fft(s, axis=0) * lax.complex(jnp.cos(ph), -jnp.sin(ph)), axis=0)
                t = lax.dynamic_slice(jnp.where(s != 0, t, 0), (pad, 0), (B, blk.shape[1])) * win[k][:, None]
                cur = lax.dynamic_slice(acc, (st[k], 0), (B, blk.shape[1]))
                return lax.dynamic_update_slice(acc, (cur + t).astype(jnp.complex64), (st[k], 0))
            return lax.fori_loop(0, st.shape[0], body, jnp.zeros_like(blk))
        _JIT['strips'] = jax.jit(strips, static_argnums=5)
    f = _JIT['strips']
    w = int(min(_xp.jax_rows(ny, 24 * nx + 64 * n, 'the autofocus correction on the JAX device'),
                max(1, af._BLOCK['jax'] // (nx + 2 * n))))
    std, wind, Vd = jnp.asarray(np.asarray(st, np.int32)), jnp.asarray(win), jnp.asarray(V)
    block = af._columns(img)
    step = lambda j: f(block(j, w), std, jnp.asarray(np.ascontiguousarray(coef[:, j:j + w])), wind, Vd, int(pad))
    starts = list(range(0, ny - w + 1, w)) + ([ny - w] if ny % w else [])
    if _xp.kind(img) != 'jax':                                      # written while the next block is under way
        out, last = np.empty((nx, ny), np.complex64), None
        for j in starts:
            last, cur = (j, step(j)), last
            if cur is not None:
                _xp.to_host(out, (slice(None), slice(cur[0], cur[0] + w)), cur[1])
        _xp.to_host(out, (slice(None), slice(last[0], last[0] + w)), last[1])
        return out
    parts = [step(j) for j in starts]
    del block
    if ny % w and len(parts) > 1:
        parts[-1] = parts[-1][:, w - ny % w:]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)


def autofocus(img, mode='quadratic', *, band=None, footprint=None, block=None, overlap=0.5, guard=True,
              iterations=6, tol=0.02, lines=0.3, max_lines=1024, subsets=8, span=4, band_db=30.0,
              min_significance=None, taper=True, range_band=None):
    """Estimate and remove the azimuth phase error of a moving-beam image, which changes along the track: each
    along-track position is formed from its own stretch of pulses, so an error over slow time defocuses each part
    of the strip by its own amount, and a height above the focus plane adds a term that changes with the terrain.

    img [nx, ny]: the complex image at baseband (moving.baseband), axis 0 along the track; a NumPy, CuPy or JAX
    array, not modified. band: (f0, hf), the center and half width of the processed azimuth band in cycles per
    pixel (moving.band; None: detected from the middle of the image). The error model in a block of rows is
    autofocus.quadratic's: the transform along axis 0 of every column times exp(1j * c * v ** 2), v the offset from
    the band center in half band widths, so c is the error in radians at the edge of the synthetic aperture.

    Blocks. The pulses of a pixel change completely over `footprint` pixels along the track (the synthetic
    aperture's length on the ground over the pixel spacing), so pixels farther apart than that share no error.
    The blocks are `block` rows, by default half the footprint, held between 256 and 4096 rows and no more than
    the image, and overlap by `overlap`: the pixels of a block then share at least half of their pulses.
    Estimate. Each block gives quadratic's estimate by map drift on the fraction `lines` of up to max_lines
    columns, and the columns are cut into `subsets` contiguous sets across the swath that each give their own.
    With `taper` the estimate reads a Taylor-weighted copy of those columns, so that it does not depend on the
    weighting the image was formed with, and the correction goes to the image as it came. Across range the copy
    is each column with its neighbors one resolution cell to either side, at 0.203, 0.594 and 0.203: the Taylor
    window (nbar 4, 35 dB) to its first cosine term. range_band, the half width of the range band in cycles per
    pixel, gives the cell as 0.5 / range_band columns (None: one column, an image sampled at its band). Along the
    track it is the same window over the processed band on the two looks. A column of an unweighted image is one
    range cell and the sidelobes of its neighbors, and a bright scatterer that crosses or straddles cells over
    its aperture leaves unequal parts of its band in two columns: the looks of either column are then different
    parts of it, their correlation has two peaks and the iteration alternates between them. taper=False reads the
    columns as they are: for an image that is Taylor weighted on both axes already (process_cphd's
    weighting='taylor', whose own columns are that copy).
    Capture range. An error of c rad at the band edge moves the two looks apart by c sep / (pi hf) rows, sep being
    the distance between the looks' centers in half bands (0.72 under the Taylor window, less under an antenna
    pattern that falls toward the band's edges, never more than 1), and the peak of their correlation is sought
    within a quarter of a block: an estimate needs |c| <= pi hf block / (4 sep), which every error up to
    info['capture'] = pi hf block / 4 rad meets (96 rad for 256 rows of an image sampled at 0.96 of its band,
    1,500 rad for 4096 rows). Beyond that range another peak is found, and the second test of the guard returns
    the image as it came. Inside it the scene decides: each look of a scatterer is spread over about half of
    that lag and has to stand out of what surrounds it.
    Guard, the first test, quadratic's on every block. The iteration of its estimate must have settled, its last
    update under ten times tol (info['settled']: an estimate that still alternates between two peaks is that of
    neither). The correlation peak of all its columns and of at least
    half of its sets (and four) must stand min_significance robust deviations above the other lags; the estimate
    of all columns must lie within three standard errors (from the sets' scatter about their median) of that
    median; and no more than a quarter of the sets may lie far from it (4.5 times the scatter and twice tol, the
    iteration's own precision). A block with a small estimate counts: it says that its part of the strip is
    focused. Without the guard every block that has settled with significant peaks counts.
    Fit. Across the swath, the departures of the sets that agree from their block's median are fitted against
    their offset from the block's middle, locally along the track (tricube weights over `span` block steps,
    Tukey's biweight against outliers): a linear trend where it is three times its standard error
    ('spatially_variant': a parabola, always). The trend is held at its end values beyond the columns the sets
    cover. Along the track, the estimates of all columns of the blocks that count, made again with the trend
    removed column by column wherever there is one (the columns in best focus would otherwise carry the
    estimate), are tied by a local line with the same weights. Blocks that do not count take the fit of their
    neighbors.
    Correction. autofocus.spatially_variant's: strips of `block` rows, each extended by the length of the blur,
    the phase of the fit at the strip's center removed column by column, cross-faded along the track with weights
    that sum to one; exact zeros stay exact zeros. Then the second test of the guard (autofocus._keep's rule) on
    the rows of the blocks that were corrected: their contrast must rise, the median gain of the parts must not be
    below minus twice its standard error, and one of the two gains must be twice its standard error; else the
    image is returned as it came. The parts are the cells of _CELL rows by _CELL columns that those rows hold (16
    sets of columns at least, as for a spotlight; the rows in cells too, because the blocks along the track are
    separate measurements): the standard errors are measured over as many parts as the image has, so that a narrow
    image is not declined for the few sets of columns it holds. A single block whose own contrast does not rise,
    or whose cells lose, is given back as it came, cross-faded with its neighbors.

    mode: 'quadratic' or 'spatially_variant'. The phase gradient modes are refused: they estimate one phase
    function of any shape from the brightest scatterers of an image formed from one aperture, and here every
    block has its own.

    img on JAX's device: the estimates, the sums of the guard and the correction run in blocks sized to its memory
    (_xp.jax_block), one compiled program per block shape; an image on the host under autofocus._via_jax crosses
    the device the same way (process_cphd's, for an image the device cannot hold) and the result is on the host.

    -> (img2 complex64 [nx, ny] where img is; info: dict(mode, band (f0, hf), block, starts [n] (first row of
    each block), estimates [n] (rad at the band edge, all columns; nan without one), estimates_std [n], subsets
    [n, sets], subset_significance, subset_position [n, sets] (-1 to 1 across the swath), settled [n], capture
    (rad: the capture range), counted [n] (the first
    test), fit [n], slope [n], curve [n], offset [n], reach (the correction at a block's center is fit + slope d +
    curve d^2, d = y - offset with y from -1 to 1 across the swath, held within reach = (least, greatest y of the
    sets)), applied_blocks [n], block_gain [n] (log of the contrast ratio), edge_rad (least,
    greatest of the fit), contrast_before, contrast_after (the whole image), gain, gain_std, part_gain,
    part_gain_std (of the corrected blocks, as quadratic's), applied, reason))."""
    from . import autofocus as af
    import scipy.fft as sfft
    af._check(img)
    _modes(mode)
    if not 0 <= overlap < 1:
        raise ValueError('overlap must be in [0, 1)')
    kind = _xp.kind(img)
    jx = kind == 'jax' or (kind == 'numpy' and bool(af._VIA))        # estimates and correction on JAX's device
    host = kind == 'numpy' and not jx
    if jx:
        import jax.numpy as xp
        from . import resample_jax as rj
    else:
        xp = _xp.module(img)
    nx, ny = img.shape
    sig0 = af.MIN_SIG if min_significance is None else float(min_significance)
    B0 = int(np.clip((footprint if footprint else 2048) / 2, 256, 4096)) if block is None else int(block)
    B0 = min(B0, nx)
    if B0 < 16:
        raise ValueError('blocks need at least 16 rows')
    step = max(1, int(round(B0 * (1 - overlap))))
    st0 = af._starts(nx, B0, step)
    nb, ks = len(st0), int(subsets)
    idx = np.unique(np.round(np.linspace(0, ny - 1, min(ny, int(max_lines)))).astype(np.int64))
    if band is None:
        sel = af._select(img[st0[nb // 2]:st0[nb // 2] + B0], 1.0, max_lines, band_db)
        if sel is None:
            sel = af._select(img, 1.0, max_lines, band_db)
        band = (0.0, 0.5) if sel is None else sel[1][:2]
    f0, hf = float(band[0]), float(band[1])
    v = af._offsets(B0, f0, hf)
    up, dn = (v >= 0) & (v <= 1), (v < 0) & (v >= -1)
    tap = xp.asarray((np.sin(np.pi * (np.arange(B0) + 0.5) / B0) ** 2).astype(np.float32))[:, None]
    tw = None
    if taper:                              # the Taylor window of the estimate's copy: on the looks, and across range
        from scipy.signal.windows import taylor
        tay = taylor(1025, nbar=_TAYLOR[0], sll=_TAYLOR[1], norm=False)
        tw = np.where(np.abs(v) <= 1, np.interp(v, np.linspace(-1.0, 1.0, 1025), tay), 0.0)
        f1 = float(np.mean(tay * np.cos(2 * np.pi * (np.arange(1025) - 512) / 1025)))  # its first cosine term
        side = np.float32(f1 / (1 + 2 * f1))
        js = 0.5 / max(0.5 if range_band is None else float(range_band), 1e-6)      # columns in a range cell
        js = int(np.clip(round(js), 1, max(1, ny // 2)))
        lo, hi = np.clip(idx - js, 0, ny - 1), np.clip(idx + js, 0, ny - 1)
        three = np.stack([lo, idx, hi], 1).ravel()                   # a column between its two neighbors: one read
        w3 = np.array([side, 1 - 2 * side, side], np.float32)
    info = dict(mode=mode, band=(f0, hf), block=B0, starts=st0, estimates=np.full(nb, np.nan),
                estimates_std=np.full(nb, np.nan), subsets=np.full((nb, ks), np.nan),
                subset_significance=np.zeros((nb, ks)), subset_position=np.full((nb, ks), np.nan),
                settled=np.zeros(nb, bool), capture=float(np.pi * hf * B0 / 4),
                counted=np.zeros(nb, bool), fit=np.zeros(nb), slope=np.zeros(nb), curve=np.zeros(nb),
                offset=np.zeros(nb), reach=(-1.0, 1.0),
                applied_blocks=np.zeros(nb, bool), block_gain=np.zeros(nb), edge_rad=(0.0, 0.0), contrast_before=0.0,
                contrast_after=0.0, gain=0.0, gain_std=0.0, part_gain=0.0, part_gain_std=0.0, applied=False,
                reason=None)
    cand, kept, ends = np.zeros(nb, bool), {}, np.zeros((2, nb, ks))        # ends: the sets' first and last columns
    if jx:                                 # the columns of the estimates, cut by rows through one compiled program
        if kind == 'numpy' and taper:
            cols = np.empty((nx, len(idx)), np.complex64)
            for r0 in range(0, nx, 4096):
                cols[r0:r0 + 4096] = np.take(img[r0:r0 + 4096], three, axis=1).reshape(-1, len(idx), 3) @ w3
        elif kind == 'numpy':
            cols = np.ascontiguousarray(img[:, idx], np.complex64)
        else:
            cols = xp.take(img, xp.asarray(idx), axis=1)
            if taper:
                cols = cols * (1 - 2 * side)
                for j in (lo, hi):
                    cols = cols + xp.take(img, xp.asarray(j), axis=1) * side
        cut = rj._source([cols])
        del cols
        grab = lambda i: cut(int(st0[i]), B0)
    elif taper and host:
        grab = lambda i: (np.take(img[st0[i]:st0[i] + B0], three, axis=1).reshape(B0, len(idx), 3) @ w3
                          ).astype(np.complex64, copy=False)
    elif taper:
        def grab(i):
            blk = img[st0[i]:st0[i] + B0][:, three]
            return ((1 - 2 * side) * blk[:, 1::3] + side * (blk[:, 0::3] + blk[:, 2::3])).astype(xp.complex64)
    else:
        grab = lambda i: img[st0[i]:st0[i] + B0][:, idx].astype(xp.complex64)

    def estimate(i):
        sub = grab(i)
        p = xp.abs(sub) ** 2
        cnt, s, mx = (_xp.host(t).astype(np.float64) for t in ((p > 0).sum(0), p.sum(0), p.max(0)))
        ok = np.flatnonzero((cnt >= 0.5 * B0) & (s > 0))
        keep = min(len(ok), max(16, int(round(lines * len(ok)))))
        k = min(ks, keep // 4)
        if k < 4:
            return
        sel = np.sort(ok[np.argsort((mx * cnt / np.where(s > 0, s, 1.0))[ok], kind='stable')[len(ok) - keep:]])
        m = keep // k
        cols = sub[:, sel[:k * m]]
        w = 1 if host else -1
        a = np.sqrt(_xp.host((xp.abs(af._ft(cols * tap, workers=w)) ** 2).sum(1)).astype(np.float64))
        if taper:
            a = a * tw
        if not (a[up].sum() > 0 and a[dn].sum() > 0):
            return
        sep = float((a[up] * v[up]).sum() / a[up].sum() - (a[dn] * v[dn]).sum() / a[dn].sum())
        A = cols.reshape(B0, k, m)
        est = af._map_drift(A, (f0, hf, sep), iterations, tol, w, pool=True, contrast=False, taper=tw)
        sets = af._map_drift(A, (f0, hf, sep), iterations, tol, w, c0=float(est['c'][0]), G=est['G'], contrast=False,
                             taper=tw)
        ycol = (idx[sel[:k * m]].reshape(k, m) - (ny - 1) / 2) / (ny / 2)
        info['estimates'][i], info['subsets'][i, :k] = float(est['c'][0]), sets['c']
        info['subset_significance'][i, :k] = np.where(np.isfinite(sets['c']), sets['sig'], 0.0)
        info['subset_position'][i, :k], ends[0, i, :k], ends[1, i, :k] = ycol.mean(1), ycol.min(1), ycol.max(1)
        info['settled'][i] = abs(float(est['step'][0])) < _SETTLED * tol
        kept[i] = (sel[:k * m], k, m, (f0, hf, sep), ycol)
        cand[i] = (info['settled'][i] and est['sig'][0] >= sig0
                   and (info['subset_significance'][i] >= sig0).sum() >= max(4, (k + 1) // 2))
    if host:
        af._threads(estimate, range(nb))
    else:
        for i in range(nb):
            estimate(i)
    sb = af._colsums(img)
    info['contrast_before'] = info['contrast_after'] = af._gain(sb, sb)[0]
    if not cand.any():
        return af._skip(img, info, 'the correlation peak is not significant in enough parts of any block'), info
    cen = st0 + (B0 - 1) / 2
    ypos, sets, sig = info['subset_position'], info['subsets'], info['subset_significance']
    across = 2 if mode == 'spatially_variant' else 'auto'
    # the first test, quadratic's, block by block: the estimate of all columns against its sets
    use = np.zeros((nb, ks), bool)                                  # the sets that agree: the trend's data
    for i in np.flatnonzero(cand):
        g = sig[i] >= sig0
        med = float(np.median(sets[i, g]))
        dev = np.abs(sets[i] - med)
        sd = 1.4826 * float(np.median(dev[g]))
        info['estimates_std'][i] = sd / np.sqrt(g.sum())
        use[i] = g & (dev <= 4.5 * sd + 2 * tol)
        info['counted'][i] = (abs(info['estimates'][i] - med) <= 3 * info['estimates_std'][i] + tol
                              and (g & ~use[i]).sum() <= g.sum() // 4)
    blocks = info['counted'] if guard else cand
    if not blocks.any():
        return af._skip(img, info, 'the parts of the blocks disagree on the estimate'), info
    # across the swath: the sets' departures from their block's median against their offset from its middle, by a
    # local fit of which the trend alone is kept, and no farther than the sets reach
    use &= blocks[:, None]
    ybar = np.array([ypos[i, use[i]].mean() if use[i].any() else 0.0 for i in range(nb)])
    ref = np.array([np.median(sets[i, use[i]]) if use[i].any() else 0.0 for i in range(nb)])
    X, dY, R = np.broadcast_to(cen[:, None], (nb, ks))[use], (ypos - ybar[:, None])[use], (sets - ref[:, None])[use]
    reach = (float(ends[0][use].min()), float(ends[1][use].max()))
    xb = cen[blocks]
    where = []
    trend = np.zeros((nb, 2))
    for i in range(nb):                                              # beyond the blocks that count: the nearest's fit
        x0 = float(np.clip(cen[i], xb.min(), xb.max()))
        near = np.sort(np.abs(xb - x0))
        where.append((x0, max(span * step, near[min(1, len(near) - 1)] + step)))
        trend[i] = _local_fit(X, dY, R, *where[i], tol, across)[1:]
    # along the track: the blocks' estimates of all columns, made again with the trend removed column by column
    # (the columns in best focus would otherwise carry the estimate), tied by a local line
    level = info['estimates'].copy()
    vq = af._offsets(B0, f0, hf) ** 2
    for i in np.flatnonzero(blocks):
        if np.abs(trend[i]).max() >= tol:
            sel, kk, mm, bandi, ycol = kept[i]
            A = grab(i)[:, sel].reshape(B0, kk, mm)
            G = af._jit('fft0')(A) if jx else af._ft(A, 'fft', -1)
            d = ycol - ybar[i]
            q = np.exp(-1j * (trend[i, 0] * d + trend[i, 1] * d * d)[None] * vq[:, None, None]).astype(np.complex64)
            G = G * xp.asarray(q)
            level[i] = float(af._map_drift(G, bandi, iterations, tol, -1, c0=float(level[i]), pool=True, G=G,
                                           contrast=False, taper=tw)['c'][0])
    kept.clear()
    mid = np.zeros(nb)
    for i in range(nb):
        mid[i] = float(np.interp(where[i][0], xb, ybar[blocks]))
        info['fit'][i] = _local_fit(xb, np.zeros(len(xb)), level[blocks], *where[i], tol, 0)[0]
    info['slope'], info['curve'], info['offset'], info['reach'] = trend[:, 0], trend[:, 1], mid, reach

    def model(i, y):
        d = np.clip(y, *reach) - mid[i]
        return info['fit'][i] + trend[i, 0] * d + trend[i, 1] * d * d
    yc = (np.arange(ny) - (ny - 1) / 2) / (ny / 2)
    most = np.array([np.abs(model(i, yc)).max() for i in range(nb)])
    act = most >= tol                                                # blocks whose correction is not nothing
    info['edge_rad'] = (float(min(model(i, yc).min() for i in range(nb))),
                        float(max(model(i, yc).max() for i in range(nb))))
    if not act.any():
        return af._skip(img, info, 'the fitted phase is below tol'), info
    raw = np.sin(np.pi * (np.arange(B0) + 0.5) / B0) ** 2
    tot = np.zeros(nx)
    for i0 in st0:
        tot[i0:i0 + B0] += raw
    win = np.stack([raw / tot[i0:i0 + B0] for i0 in st0]).astype(np.float32)

    def correct(on):
        blur = 2 * most[on].max() / (2 * np.pi * hf)
        n = sfft.next_fast_len(B0 + 2 * min(B0, int(np.ceil(blur)) + 8), real=False)
        n += (n - B0) % 2
        vb = af._offsets(n, f0, hf)
        roll = 0.5 * (1 + np.cos(np.pi * np.clip((np.abs(vb) - 1) / max(0.5 / hf - 1, 1e-9), 0, 1)))
        V, coef = (vb ** 2 * roll).astype(np.float32), lambda i: (model(i, yc) * on[i]).astype(np.float32)
        if jx:
            return _strips_jax(img, st0, B0, (n - B0) // 2, win, V, np.stack([coef(i) for i in range(nb)]))
        return af._apply_strips(img, st0, B0, (n - B0) // 2, win, V[None], lambda i: coef(i)[None, :])
    out = correct(act)
    if not guard:
        info['applied_blocks'] = act
    else:
        # the second test, on cells of h rows by the sets of columns: [cells along the track, 3, sets]
        h = int(max(8, min(_CELL[0], B0 // 2)))
        sets = int(min(ny, max(16, ny // _CELL[1])))
        cuts = np.unique(np.round(np.linspace(0, ny, sets + 1)).astype(np.int64))[:-1]
        qb, qa = _cellsums(img, h, cuts), _cellsums(out, h, cuts)
        mid = (np.arange(len(qb)) + 0.5) * h
        rows = lambda i: (mid >= st0[i]) & (mid < st0[i] + B0)
        flat = lambda q, sel: q[sel].transpose(1, 0, 2).reshape(3, -1)   # every cell a set of its own for _gain
        gain = lambda sel: af._gain(flat(qb, sel), flat(qa, sel), sets=int(sel.sum()) * len(cuts))
        cb, ca, se, med, mse = gain(np.any([rows(i) for i in np.flatnonzero(act)], axis=0))
        d = float(np.log(ca / cb)) if cb > 0 and ca > 0 else 0.0
        info.update(gain=d, gain_std=se, part_gain=med, part_gain_std=mse)
        if not ca > cb:
            return af._skip(img, info, 'the correction lowers the contrast of the image'), info
        if med < -2 * mse:
            return af._skip(img, info, 'the correction lowers the contrast of most parts of the image'), info
        if not (d >= 2 * se or (med > 0 and med >= 2 * mse)):
            return af._skip(img, info, 'the contrast gain is within its scatter over the parts of the image'), info
        for i in np.flatnonzero(act):
            cb, ca, se, med, mse = gain(rows(i))
            info['block_gain'][i] = float(np.log(ca / cb)) if cb > 0 and ca > 0 else 0.0
            info['applied_blocks'][i] = ca > cb and med >= -2 * mse
        if not info['applied_blocks'].any():
            return af._skip(img, info, 'the correction does not raise the contrast of any block'), info
        if not info['applied_blocks'][act].all():
            del out
            out = correct(info['applied_blocks'])
    info.update(contrast_after=af._gain(sb, af._colsums(out))[1], applied=True)
    return out, info


# ------------------------------------------------------------------------------------------ quality areas in range
_LINES = 7                    # along-track positions at which the quality model is evaluated against ground range


def _range_area(g, bm, b, opt, step=50.0, limit=30e3):
    """The area of a moving-beam product by image quality (process_cphd's area): the tasked extent along the
    track, and across it the ground range around the tasked swath in which the criterion of rgiqe.area_mask holds.

    The model is rgiqe.Radiometry's, evaluated for the moving beam: at _LINES positions along the track, on the
    line across the track through each, with the pulses that the pixels there are formed from (|u| <= 1 of
    form_cphd's weighting at the swath center). Each pulse's reference is its own SRP, so the gain of a ground
    point is the two-way pattern in elevation and in azimuth, averaged in amplitude over those pulses, with the
    spreading loss, and the delay windows are the pulses' own. The noise level at the beam center comes once, from
    the middle position's pulses, by Radiometry's order of sources, and so does the check of the antenna block
    against the beam's footprint in the phase history (Radiometry.pointing: the frames as the file states them,
    each boresight on its pulse's SRP, or no trusted pattern, with which every area but 'tasked' is refused as for
    a spotlight). 'black' ends where a scene of sigma zero opt['black_sigma0_db'] (-10 dB) has a signal-to-noise
    ratio of 1, ambiguities counted as noise (rgiqe.area_mask). b: the tasked rectangle (lo1, hi1, lo2, hi2), m
    from g['center'] along e1 and e2.
    -> (the area's bounding rectangle, valid: dict(along, across (m from the center), level [lines, across]: the
    margin of rgiqe.area_mask, >= 0 inside the area); the tasked rectangle and None when the area reaches less
    than one step of the model's lattice beyond the tasked swath, with a note that says which of the receive
    window and the criterion ends it there)."""
    from . import rgiqe
    area, c, e1, e2 = opt['area'], g['center'], g['e1'], g['e2']
    K, f0, df = g['S'].shape[1], float(g['fmin']), float(g['df'])
    t1 = np.linspace(b[0], b[1], _LINES)
    mid = 0.5 * (b[2] + b[3])
    sp = spectrum(bm, g['ant'], (f0, f0 + (K - 1) * df), K, c + t1[:, None] * e1 + mid * e2, e1, e2)
    if not (sp['pulses'] > 1).all():
        raise ValueError(f'area={area!r}: part of the tasked extent along the track is not illuminated')
    n0 = _LINES // 2
    bw, ns = sp['b1'] * sp['b2'], sp['pulses'] * K               # the support's area on the ground, samples summed
    span = lambda n: (int(sp['first'][n]), int(sp['first'][n] + sp['pulses'][n]))
    kw = dict(channel=opt['channel'], sicd=g['sicd'], scene_sigma0_db=opt['scene_sigma0_db'])
    black = float(opt.get('black_sigma0_db', -10.0))
    # the middle position's pulses, from the history in memory: the noise level and the check of the antenna block
    # against the beam's footprint (Radiometry.pointing), which the other positions take over
    rad = rgiqe.Radiometry(opt['cphd'], g['meta'], pulses=span(n0), nesz_db=opt['nesz_db'],
                           history=g['S'][slice(*span(n0))], **kw)
    if not rad.has_antenna:
        raise ValueError(f'area={area!r}: the file describes no antenna pattern, so the signal has no edge '
                         "to follow; only area='tasked' can be processed")
    if rad.pointing is None:                                         # the data contradict the antenna block
        raise ValueError(rgiqe._untrusted(area))
    level = 10 * np.log10(rad._nesz_ref(bw[n0], ns[n0]))
    rads = [rad if n == n0 else rgiqe.Radiometry(opt['cphd'], g['meta'], pulses=span(n), nesz_db=level, adopt=rad,
                                                 **kw) for n in range(_LINES)]
    step = float(min(step, (b[3] - b[2]) / 16))
    line = lambda n, y: c + t1[n] * e1 + y[:, None] * e2
    quality = None
    notes = list(rad.notes)
    if area == 'tasked_quality' and rad.sicd_rniirs is None:        # the least the model predicts in the tasked swath
        y = np.linspace(b[2], b[3], 17)
        q = []
        for n in range(_LINES):
            snr, _, _, cl = rads[n]._fields(line(n, y), bw[n], ns[n])
            q.append(rgiqe.snr_to_rniirs(bw[n], snr[~cl].min()) if not cl.all() else np.inf)
        if not np.isfinite(min(q)):
            raise ValueError('the antenna pattern is not known anywhere in the tasked swath; give an RNIIRS')
        quality = float(min(q))
        notes.append('No tasked RNIIRS is known, so the tasked quality is the lowest RNIIRS the model predicts in '
                     f'the tasked swath, {quality:.2f}.')
    ext = np.array([mid - b[2], b[3] - mid]) + 2 * step             # reach below and above the swath center
    while True:
        y = mid + np.arange(-np.ceil(ext[0] / step), np.ceil(ext[1] / step) + 1) * step
        lev, edge = np.empty((_LINES, len(y)), np.float32), np.zeros(2, bool)
        for n in range(_LINES):
            tasked = c + (t1[n] + step * np.array([-1.0, 1.0, 1.0, -1.0]))[:, None] * e1 + np.array(
                [b[2], b[2], b[3], b[3]])[:, None] * e2
            m, info = rgiqe.area_mask(rads[n], line(n, y)[None], area, bw[n], ns[n], tasked, quality,
                                      black_sigma0_db=black)
            lev[n] = info['level'][0]
            edge |= (m[0, 0], m[0, -1])
        grow = edge & (ext < limit)
        if not grow.any():
            break
        ext = np.where(grow, np.minimum(np.maximum(1.5 * ext, ext + 4 * step), limit), ext)
    inside = (lev >= 0).any(0)
    lo, hi = y[inside][0], y[inside][-1]
    more = max(b[2] - lo, 0.0) + max(hi - b[3], 0.0)
    src = dict(argument='the noise level given', sicd="the SICD's noise level", cphd="the file's PTRef and PNRef",
               uncalibrated='the clutter-to-noise ratio measured from the data and the assumed scene sigma zero')
    what = (f'RNIIRS at least {info["threshold"]:.2f}' if info['criterion'] == 'rniirs'
            else f'a signal-to-noise ratio of at least 1 for a scene of sigma zero {black:.1f} dB')
    what += f' ({src.get(rad.level_source, "PTRef and the noise measured from the data")})'
    if more < step:                                                  # why not: the delay window, or the criterion
        y2 = np.array([b[2] - 2 * step, b[3] + 2 * step])
        _, mg, mge, _ = rads[n0]._fields(line(n0, y2), bw[n0], ns[n0])
        if ((mge if area == 'black' and rad.has_extended else mg) < 0).all():
            notes.append('The receive window of the pulses holds no ground range beyond the tasked swath: the '
                         f'area is the tasked area ({what} was asked).')
        else:
            notes.append(f'No ground range beyond the tasked swath has {what}: the area is the tasked area.')
        g['notes'].extend(notes)
        return np.asarray(b, np.float64), None
    else:
        notes.append(f'The area is the tasked extent along the track and the ground range with {what}: '
                     f'{lo - mid:+.0f} to {hi - mid:+.0f} m from the swath center (tasked {b[2] - mid:+.0f} to '
                     f'{b[3] - mid:+.0f} m).')
        if area == 'black' and rad.has_extended:
            notes.append('The range swath is opened to the extended delay window TOAE1..TOAE2 of the file, in '
                         'which the echoes do not have the full bandwidth everywhere.')
    g['notes'].extend(notes)
    out = np.array([b[0], b[1], min(lo - step, b[2]), max(hi + step, b[3])])
    return out, dict(along=t1, across=y, level=lev, center=c)


def _range_mask(out, valid):
    """The pixels [nx, ny] of a product inside the area of _range_area: its level read bilinearly between the
    lines along the track and the samples across it, >= 0."""
    from .patches import _rows
    from .resample import KnotMap
    nx, ny = np.shape(out['image'])
    q = np.asarray(out['origin'], np.float64) - valid['center']
    ki = (valid['along'] - q @ out['e1']) / out['spx']
    kj = (valid['across'] - q @ out['e2']) / out['spy']
    km = KnotMap(ki, kj, valid['level'].astype(np.float64), ny)
    mask = np.empty((nx, ny), bool)

    def work(sl):
        mask[sl] = km(sl) >= 0
    _rows(work, nx, 256)
    return mask


# ------------------------------------------------------------------------------------------- process_cphd's stages
_MARGIN = 24                  # rows formed beyond the area at each end for apodization: the band limit is circular


def plan(g, st, opt):
    """process_cphd's plan for a moving-beam collection: the weighting form_cphd applies, the grid, and what the
    stages after formation need, kept in st. g: form_cphd's geometry; opt: dict(product, weighting, autofocus,
    area, spacing, extent, cphd, channel, nesz_db, scene_sigma0_db, and optionally black_sigma0_db (the sigma zero
    of area='black', dB; -10), fallback (the weighting to use, with a note, where 'sva' cannot be applied: the
    caller did not ask for 'sva' by name; None: raise), grid (form_cphd's grid= in this collection's frame: the
    area and the sampling are then the grid's) and autofocus_given (whether the caller named the autofocus)).

    On a caller's grid. Where the grid's first axis lies within ALONG degrees of the track, the stages run as on
    the collection's own grid, at the grid's spacing (a note says so when it is coarser than the band). On another
    grid the product is the image as formed there with its weighting: the stages work along the track, so it is
    not brought to baseband, an autofocus left to the default is not run, with a note, and one asked for by name
    raises a ValueError.

    Spacing. 'taylor' and 'uniform' keep form_cphd's own (0.96 of the reciprocal of the nominal bands) unless the
    azimuth band of the image is wider than that samples (a footprint slower than the platform: an orbit, a
    sliding spotlight; a squint), and a given spacing is used as it is. 'sva': exactly half the reciprocal of
    the processed azimuth band and of the ground range band at the swath center, on a grid whose rows number a
    multiple of four. An area beyond the tasked swath is formed on the tasked grid's own pixel lattice.
    Apodization is refused where the rectangle of the module docstring does not hold: the processed azimuth band
    differs over the scene by more than VARY, the squint shears the range band by more than SHEAR of it, or the
    ground range band changes across the swath by more than RANGE; and where a spacing is given, since it sets its
    own. With opt['fallback'] the product then takes that weighting and a note says why; without it a ValueError
    does."""
    wt, af, area, spacing = opt['weighting'], opt['autofocus'], opt['area'], opt['spacing']
    if af:
        _modes(af)
    S, ant, dsin = g['S'], np.asarray(g['ant'], np.float64), float(g['dsin'])
    K, f0, df = S.shape[1], float(g['fmin']), float(g['df'])
    fband = (f0, f0 + (K - 1) * df)
    fc = 0.5 * (fband[0] + fband[1])
    c, e1, e2 = g['center'], g['e1'], g['e2']
    bm = beam_model(ant, g['srp'], g['meta']['tx_time'], dsin)
    gs = opt.get('grid')
    if gs is not None:                                               # the grid is the area and the sampling
        b = np.array([-gs['shape'][0] * gs['spx'], gs['shape'][0] * gs['spx'],
                      -gs['shape'][1] * gs['spy'], gs['shape'][1] * gs['spy']], np.float64) / 2
        along = bm['flight'] - (bm['flight'] @ gs['normal']) * np.asarray(gs['normal'], np.float64)
        off = float(np.degrees(np.arccos(np.clip(abs(along @ e1) / np.linalg.norm(along), 0.0, 1.0))))
        if off > ALONG:
            if af and opt.get('autofocus_given'):
                raise ValueError(f"autofocus={af!r} works along the grid's first axis, which is {off:.0f} degrees from "
                                 "this collection's track: give autofocus=False, or a grid along the track")
            g['notes'].append(f"the grid's first axis is {off:.0f} degrees from the track: the product is the image as "
                              'formed on the grid with its weighting, not at baseband and without autofocus (the '
                              'stages of a moving beam work along the track)')
            st.update(spot=False, plane='ground', weighting=wt, autofocus=False, beam=bm, margin=0, valid=None,
                      as_formed=True)
            return dict(window=wt == 'taylor', azimuth_window='taylor' if wt == 'taylor' else 'uniform')
    elif opt['extent'] is not None:
        b = np.array([-opt['extent'][0], opt['extent'][0], -opt['extent'][1], opt['extent'][1]], np.float64) / 2
    else:
        d = g['corners'] - c
        b = np.array([(d @ e1).min(), (d @ e1).max(), (d @ e2).min(), (d @ e2).max()])
    notes, valid, tasked = g['notes'], None, b
    lam = C / (f0 + K / 2 * df)
    bm, steer, why = _steer(g, bm, b, opt, lam)                      # the SRPs, or the antenna's boresight
    if why:
        notes.append(why)
    if area != 'tasked':
        import time
        tr = time.perf_counter()
        b, valid = _range_area(g, bm, b, opt)
        st['model_s'] = time.perf_counter() - tr                     # process_cphd's 'radiometry' stage
    smooth = bm['drift'] <= 0.05
    own = (0.96 * lam / (2 * dsin), 0.96 * C / (2 * K * df) / np.cos(g['graze']))        # form_cphd's spacing

    def bands(over):
        """The spectrum over the rectangle that sets the sampling -> (spectrum at 7 x 5 points, the points that
        count, the processed azimuth band (cycles/m), the ground range band at the swath center, the widest
        azimuth band of the image)."""
        t1, t2 = np.linspace(over[0], over[1], 7), np.linspace(over[2], over[3], 5)
        pts = c + (t1[:, None, None] * e1 + t2[None, :, None] * e2).reshape(-1, 3)
        sp = spectrum(bm, ant, fband, K, pts, e1, e2)
        use = sp['full'] if sp['full'].any() else sp['pulses'] > 1
        if not use.any():
            raise ValueError('no pulse of the collection illuminates the area: check the extent')
        b2 = float(np.interp(0.5 * (over[2] + over[3]), t2, sp['b2'].reshape(len(t1), len(t2)).mean(0)))
        wide = float((sp['b1'] * fband[1] / fc + 2 * np.abs(sp['k1c']) * (fband[1] - fc) / fc)[use].max())
        return sp, use, float(np.median(sp['b1'][use])), b2, wide
    # the area's rectangle for apodization, whose band every pixel must hold, else the tasked one, so that a wider
    # area lies on the tasked product's own pixels
    sp, use, b1, b2, wide = bands(b if wt == 'sva' else tasked)
    pad, margin, out = 0.0, 0, {}
    if wt == 'sva':
        why = None
        vary = float(np.ptp(sp['b1'][use]) / b1)
        shear = float(np.abs(sp['shear'][use]).max() * b1 / b2)
        rng = float(np.abs(sp['b2'] / b2 - 1).max())
        if spacing is not None:
            if not opt.get('fallback'):
                raise ValueError("weighting='sva' sets the spacing itself (twice Nyquist): leave spacing=None")
            why = 'it sets the spacing itself (twice Nyquist), and a spacing was given'
        elif not smooth:
            why = 'the beam center does not move smoothly along the track, so the image has no one baseband'
        elif vary > VARY and not sp['full'].any():
            why = (f'no pixel is seen across the whole processed band (the collection is shorter than a synthetic '
                   f'aperture and the scene), and the azimuth band the ends leave differs by {100 * vary:.0f}% over '
                   f'the scene; the limit is {100 * VARY:.0f}%')
        elif vary > VARY:
            why = (f'the processed azimuth band differs by {100 * vary:.0f}% over the scene (a beam whose speed over '
                   f'the ground changes); the limit is {100 * VARY:.0f}%')
        elif shear > SHEAR:
            why = (f'the squint ({np.degrees(np.arctan(np.abs(sp["shear"][use]).max())):.1f} degrees on the ground) '
                   f'shears the range band by {100 * shear:.1f}% of its width across the azimuth band; the limit is '
                   f'{100 * SHEAR:.0f}%')
        elif rng > RANGE:
            why = (f'the ground range band changes by {100 * rng:.1f}% across the swath (cos of the grazing angle); '
                   f'the limit is {100 * RANGE:.0f}%')
        if why and not opt.get('fallback'):
            raise ValueError("weighting='sva' needs a spectrum that is a rectangle on the grid's axes: " + why)
        if why:                                                      # 'sva' was not asked for by name
            wt = opt['fallback']
            if spacing is None:
                why = "it needs a spectrum that is a rectangle on the grid's axes, and " + why
            notes.append(f"weighting: this product's default, spatially variant apodization ('sva'), is not applied: "
                         f'{why}. The image has weighting={wt!r} instead.')
            if area != 'tasked':
                sp, use, b1, b2, wide = bands(tasked)
    if wt == 'sva':
        bt = float(sp['b1'][use].min())
        # half the band kept, its soft edge and the squint's shift at the first frequency, in units of the half band
        # the pulses of |u| <= 1 give there
        shift = np.abs(sp['k1c'][use]) * (fc - fband[0]) / fc
        need = (bt / 2 * (1 + PAD) + shift) / (sp['b1'][use] / 2 * fband[0] / fc)
        pad = float(need.max() - 1)
        spx, spy = 0.5 / bt, 0.5 / b2
        margin = _MARGIN
        n = 4 * int(np.ceil((int(np.ceil((b[1] - b[0]) / spx)) + 1 + 2 * margin) / 4))
        out['bounds'] = (b[0] - margin * spx, b[0] + (n - margin - 1.5) * spx, b[2], b[3])
        out['spacing'] = (spx, spy)
        notes.append(f'apodization: formed with a rectangular azimuth weighting {100 * pad:.1f}% wider than the '
                     f'processed band, of which the band {bt:.4f} cycles/m is kept at every pixel')
        b1 = bt
    else:
        if gs is not None:
            if gs['spx'] > 1.0001 / wide or gs['spy'] > 1.0001 / sp['b2'].max():
                notes.append(f"the grid's spacing {gs['spx']:.3f} x {gs['spy']:.3f} m is coarser than the band of the "
                             f"data allows on its axes ({1 / wide:.3f} x {1 / sp['b2'].max():.3f} m): the image is "
                             'aliased')
        elif spacing is not None:
            out['spacing'] = tuple(float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
            if out['spacing'][0] > 1.0001 / wide or out['spacing'][1] > 1.0001 / sp['b2'].max():
                notes.append(f"spacing {out['spacing'][0]:.3f} x {out['spacing'][1]:.3f} m is coarser than the band "
                             f"of the data allows ({1 / wide:.3f} x {1 / sp['b2'].max():.3f} m): the image is "
                             'aliased')
        elif own[0] > 1 / (1.002 * wide):                            # a wider band than the beam's own Doppler band
            out['spacing'] = (0.96 / wide, own[1])
            notes.append(f'azimuth band of the image {wide:.3f} cycles/m, {wide * lam / (2 * dsin):.2f} times the '
                         'processed Doppler band over the platform\'s speed (the footprint moves slower than the '
                         f'platform, or a squint moves the band with frequency): spacing {out["spacing"][0]:.3f} m '
                         'along the track')
        if area != 'tasked':                                         # on the tasked grid's own pixels
            spy = (out.get('spacing') or own)[1]
            out['bounds'] = (b[0], b[1], tasked[2] - np.ceil((tasked[2] - b[2]) / spy) * spy, b[3])
    spx = gs['spx'] if gs is not None else (out.get('spacing') or own)[0]
    if af and not smooth:
        notes.append('autofocus skipped: the beam center does not move smoothly along the track')
        af = False
    elif af and spx > 1.0001 / wide:
        notes.append('autofocus skipped: the spacing along the track is coarser than the azimuth band')
        af = False
    st.update(spot=False, plane='ground', weighting=wt, autofocus=af, beam=bm, margin=margin, valid=valid,
              band1=b1, band2=b2, pad=pad, wide=wide,
              footprint=float(np.median((sp['pulses'] * sp['speed'])[use])))
    out.update(window=wt == 'taylor', azimuth_window='taylor' if wt == 'taylor' else 'uniform', azimuth_pad=pad,
               fit_patch=True)
    if steer is not None:                                            # form_cphd's weighting about the boresight
        out['beam_sine'] = steer
    return out


def stages(out, st, product, lap, backend):
    """process_cphd's stages after formation for a moving beam: baseband, autofocus, apodization, the area's mask.
    The image goes to the backend's device when its memory plan (memory.cuda_plan) holds it with the arrays the
    stages make beside it, and the stages run where it is. An image a CUDA device cannot hold is processed on the
    host; one a JAX device cannot hold stays on the host and crosses the device in blocks, stage by stage. A note
    says so. -> (out, tight: the CUDA plan takes more than half of the device)."""
    import contextlib
    from . import autofocus as afm, memory, weighting
    wt, af, mg = st['weighting'], st['autofocus'], st['margin']
    out['beam'], out['window'] = st['beam'], (wt, wt)
    if st.get('as_formed'):                                          # a caller's grid off the track: the image as formed
        return out, False
    nx, ny = out['image'].shape
    sizes = dict(ground=8 * nx * ny, valid=nx * ny if st['valid'] is not None else 0,
                 display=nx * ny if product == 'sidd' else 0)
    jaxdev = backend in ('jax', 'tpu')
    tight = via = False
    if backend == 'cuda':
        p = memory.cuda_plan(sizes, _xp.cuda_budget(), autofocus=bool(af) or wt == 'sva')
        if p['place']['ground'] == 'device':
            tight = p['need'] > 0.5 * p['budget']
            if tight:
                memory.cuda_release()
            from .patches import _to_device
            out['image'] = _to_device(out['image'])                  # through two page-locked blocks
        else:
            out['notes'].append(f"the device has {memory._gb(p['budget'])} for the stages after formation, which "
                                f"take {memory._gb(p['need'])} with the image on it: they ran on the host")
    elif jaxdev:
        budget = _xp.jax_budget()                                    # None: a device that does not report its memory
        p = budget and memory.cuda_plan(sizes, budget, autofocus=bool(af) or wt == 'sva', joined=True)
        if not p or p['place']['ground'] == 'device':
            out['image'] = _xp.device(out['image'], backend)
        else:                                # too large for the device, or small enough to cross it for nothing
            via = True
            if not p['fits']:
                out['notes'].append(f"the JAX device has {memory._gb(p['budget'])} for the stages after formation, "
                                    f"which take {memory._gb(p['need'])} with the image on it: kept on the host and "
                                    'passed through the device in blocks')
    release = memory.cuda_release if tight else (lambda: None)
    crossing = afm._via_jax if via else contextlib.nullcontext      # a host image's autofocus and band limit

    def run(stage):
        """A stage that returns a new image, where the image is; on the host when the GPU has no memory for it."""
        try:
            return stage(out['image'])
        except Exception as e:
            if _xp.kind(out['image']) != 'cupy' or type(e).__name__ != 'OutOfMemoryError':
                raise
            out['image'] = _xp.host(out['image'])
            memory.cuda_release()
            out['notes'].append('the stages after formation ran on the host: the GPU had no memory for them')
            return stage(out['image'])
    baseband(out, _jax=via)
    lap('baseband', out['image'])
    hf = min(0.5, 0.5 * st['band1'] * out['spx'])                    # the processed band: radians at its edge
    if af:
        try:
            fp = st['footprint'] / out['spx']
            hr = min(0.5, 0.5 * st['band2'] * out['spy'])            # the range band, for the estimate's taper
            with crossing():
                out['image'], info = run(lambda a: autofocus(a, af, band=(0.0, hf), footprint=fp, range_band=hr,
                                                             taper=wt != 'taylor'))
        except ValueError as e:                                      # an image too small for the estimate
            info = dict(mode=af, applied=False, reason=str(e))
            out['notes'].append(f'autofocus skipped: {e}')
        out.update(autofocus=af if info.get('applied') else None, autofocus_info=info)
        release()
        lap('autofocus', out['image'])
    if wt == 'sva':
        m = (np.fft.fftfreq(nx) * nx)
        with crossing():
            out['image'] = run(lambda a: band_limit(a, ((m >= -nx // 4) & (m < nx // 4)).astype(np.float32)))
        release()
        if jaxdev:                                                   # in slabs of rows; the image is given up
            img, out['image'] = [out['image']], None
            out['image'] = weighting.sva(img, 2, _jax=(None, None, 'host' if via else 'device'))
        else:
            out['image'] = run(lambda a: weighting.sva(a, 2, inplace=True))
        out['image'] = out['image'][mg:nx - mg]
        out['origin'] = out['origin'] + mg * out['spx'] * np.asarray(out['e1'], np.float64)
        out['center'] = (out['origin'] + ((nx - 2 * mg) / 2.0) * out['spx'] * out['e1']
                         + (ny / 2.0) * out['spy'] * out['e2'])
        release()
        lap('sva', out['image'])
    if st['valid'] is not None:
        valid = _range_mask(out, st['valid'])
        if _xp.kind(out['image']) == 'cupy':
            import cupy
            cupy.putmask(out['image'], cupy.asarray(~valid), 0)
        else:
            xp = _xp.module(out['image'])
            out['image'] = xp.where(xp.asarray(valid), out['image'], 0)
        lap('mask', out['image'])
    b1, b2 = st['band1'], st['band2']
    out.update(bandwidth=(b1, b2), support=((-b1 / 2, b1 / 2), (-b2 / 2, b2 / 2)))
    return out, tight
