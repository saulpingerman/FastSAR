"""Raw (level 0) radar data: the steps that turn recorded echoes into the frequency-domain phase history the rest
of fastsar consumes, shared by the level-0 readers in fastsar.io (read_nisar).

A raw line holds the baseband echo samples of one pulse at the sampling rate fs, sample 0 at the two-way delay
tau0 of the receive window. range_compress correlates each line with the transmitted chirp; fx_history takes the
spectrum of the compressed line over the chirp's band and references it to a delay per pulse, which gives the
CPHD convention (SGN = -1): a scatterer at delay tau contributes exp(-j 2 pi f (tau - tau_ref)) at the frequency f.
Referenced to each pulse's zero-Doppler ground point at mid swath (zero_doppler_points), a stripmap collection
reads like a CPHD with a moving scene reference point and forms with form_cphd's moving mode.

Orbits come as state vectors: hermite_orbit interpolates positions and velocities between them (a cubic Hermite
spline through the positions with the velocities as slopes, the interpolation NISAR's own products name).
"""
import numpy as np
import scipy.fft

C = 299792458.0


def range_compress(z, chirp, workers=-1):
    """z [P, n] complex baseband samples of P pulses, chirp [L] the replica at the same sampling rate: the matched
    filter output, complex64 [P, n], aligned so that an echo whose chirp starts at sample m peaks at sample m
    (the last L - 1 samples of each line hold partial correlations)."""
    z = np.asarray(z)
    P, n = z.shape
    L = len(chirp)
    nfft = scipy.fft.next_fast_len(n + L - 1)
    H = np.conj(scipy.fft.fft(np.asarray(chirp, np.complex64), nfft))
    X = scipy.fft.fft(z, nfft, axis=1, workers=workers)
    X *= H[None, :]
    return scipy.fft.ifft(X, axis=1, workers=workers, overwrite_x=True)[:, :n].astype(np.complex64)


def fx_history(rc, fs, fc, tau0, tau_ref, bandwidth, margin=1.0, workers=-1):
    """rc [P, n] range-compressed baseband lines sampled at fs (Hz), the carrier fc (Hz), sample 0 at the two-way
    delay tau0 (s, one value or [P]), referenced to the two-way delays tau_ref [P] (s): the frequency-domain phase
    history S [P, K] (complex64) on the grid fmin + k df, df = fs / n, over |f - fc| <= margin * bandwidth / 2,
    with the phase exp(-j 2 pi f (tau - tau_ref)) for an echo at delay tau. -> S, fmin, df. The unambiguous range
    c / (2 df) is the receive window's length."""
    rc = np.asarray(rc)
    P, n = rc.shape
    X = scipy.fft.fftshift(scipy.fft.fft(rc, axis=1, workers=workers), axes=1)
    fb = scipy.fft.fftshift(scipy.fft.fftfreq(n, 1.0 / fs))
    idx = np.nonzero(np.abs(fb) <= margin * bandwidth / 2)[0]
    k0, k1 = int(idx[0]), int(idx[-1]) + 1
    fb = fb[k0:k1]
    tau0 = np.broadcast_to(np.asarray(tau0, np.float64), (P,))
    tau_ref = np.asarray(tau_ref, np.float64)
    S = X[:, k0:k1]
    S *= np.exp(2j * np.pi * (fc * tau_ref[:, None] + fb[None, :] * (tau_ref - tau0)[:, None]))
    return S.astype(np.complex64), float(fc + fb[0]), float(fs / n)


def hermite_orbit(t_sv, pos, vel):
    """State vectors at times t_sv [n] (s) with positions pos [n, 3] and velocities vel [n, 3] (ECF, m and m/s)
    -> a function of times t [...] returning (positions [..., 3], velocities [..., 3]) by a cubic Hermite spline
    per axis (the positions as values, the velocities as slopes)."""
    from scipy.interpolate import CubicHermiteSpline
    t_sv, pos, vel = (np.asarray(a, np.float64) for a in (t_sv, pos, vel))
    spl = [CubicHermiteSpline(t_sv, pos[:, i], vel[:, i]) for i in range(3)]
    der = [s.derivative() for s in spl]

    def at(t):
        t = np.asarray(t, np.float64)
        return np.stack([s(t) for s in spl], -1), np.stack([d(t) for d in der], -1)
    return at


def zero_doppler_points(pos, vel, r, side, height=0.0, iterations=60):
    """Ground points [P, 3] (ECF) at slant range r (m, one value or [P]) from the antenna positions pos [P, 3], in
    the plane perpendicular to the velocity vel [P, 3] (zero Doppler), on the given side of the track ('left' or
    'right' looking along the velocity), at the given height above the WGS-84 ellipsoid; by bisection on the look
    angle from nadir."""
    from .io import ecf_to_geodetic
    pos, vel = np.asarray(pos, np.float64), np.asarray(vel, np.float64)
    P = len(pos)
    r = np.broadcast_to(np.asarray(r, np.float64), (P,))
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    v = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    lat, lon, _ = ecf_to_geodetic(pos)
    lat, lon = np.radians(lat), np.radians(lon)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    down = -(up - np.sum(up * v, 1, keepdims=True) * v)
    down /= np.linalg.norm(down, axis=1, keepdims=True)
    right = np.cross(v, up)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    sign = 1.0 if side == 'right' else -1.0
    lo, hi = np.zeros(P), np.full(P, 0.5 * np.pi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        p = pos + r[:, None] * (np.cos(mid)[:, None] * down + sign * np.sin(mid)[:, None] * right)
        h = ecf_to_geodetic(p)[2]
        below = h < height                 # the point rises with the look angle
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    mid = 0.5 * (lo + hi)
    return pos + r[:, None] * (np.cos(mid)[:, None] * down + sign * np.sin(mid)[:, None] * right)


def decode_bfpq(block, lut):
    """NISAR's block floating point quantization: a compound array with uint16 fields 'r' and 'i' indexing the
    lookup table lut [65536] -> complex64."""
    lut = np.asarray(lut, np.float32)
    return (lut[block['r']] + 1j * lut[block['i']]).astype(np.complex64)
