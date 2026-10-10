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


def resample_pulses(S, t, tu, arrays=(), block=2048):
    """The phase history S [P, K] (complex64, referenced so that it varies slowly from pulse to pulse) at the pulse
    times t [P] resampled onto the times tu [P'] by cubic splines across pulses, in blocks of frequency samples;
    the arrays [P, ...] (positions, delays) interpolated the same way. -> S' [P', K], tuple of interpolated arrays."""
    from scipy.interpolate import CubicSpline
    t, tu = np.asarray(t, np.float64), np.asarray(tu, np.float64)
    P, K = S.shape
    out = np.empty((len(tu), K), np.complex64)
    for k0 in range(0, K, block):
        k1 = min(K, k0 + block)
        blk = S[:, k0:k1]
        out[:, k0:k1] = CubicSpline(t, blk.real, axis=0)(tu) + 1j * CubicSpline(t, blk.imag, axis=0)(tu)
    return out, tuple(CubicSpline(t, np.asarray(a, np.float64), axis=0)(tu) for a in arrays)


def decode_bfpq(block, lut):
    """NISAR's block floating point quantization: a compound array with uint16 fields 'r' and 'i' indexing the
    lookup table lut [65536] -> complex64."""
    lut = np.asarray(lut, np.float32)
    return (lut[block['r']] + 1j * lut[block['i']]).astype(np.complex64)


# ---------------------------------------------------------------------------------------------- CEOS (ALOS PALSAR 1.0)

def ceos_records(data):
    """The records of a CEOS file (bytes): (sequence number, subtype1, type, subtype2, subtype3, start, length) from
    each 12-byte record header (big-endian sequence number and length, four code bytes)."""
    out, pos, n = [], 0, len(data)
    while pos + 12 <= n:
        seq = int.from_bytes(data[pos:pos + 4], 'big')
        codes = tuple(data[pos + 4:pos + 8])
        length = int.from_bytes(data[pos + 8:pos + 12], 'big')
        if length < 12:
            break
        out.append((seq, codes[0], codes[1], codes[2], codes[3], pos, length))
        pos += length
    return out


def _ascii(rec, a, b):
    """The ASCII field at 1-based byte positions a to b of a record."""
    return rec[a - 1:b].decode('ascii', 'replace').strip()


def _num(rec, a, b, default=None):
    s = _ascii(rec, a, b).replace('D', 'E')
    try:
        return float(s)
    except ValueError:
        return default


def palsar_leader(data):
    """The fields of an ALOS PALSAR level 1.0 SAR leader file (bytes) that a reader needs: from the data set summary
    record the scene center (latitude, longitude, terrain height), the wavelength, the chirp (constant and linear
    terms, pulse length), the sampling rate, the range gate, the nominal PRF, the bits per sample; from the platform
    position record the state vectors (times as seconds of the record's day, positions and velocities, ECR) and the
    day. Positions follow JAXA's format description (Tables 3.3-5 and 3.3-6)."""
    recs = ceos_records(data)
    out = {}
    for seq, s1, typ, s2, s3, start, length in recs:
        rec = data[start:start + length]
        if typ == 10 and s1 == 18:                       # data set summary
            out.update(scene_lat=_num(rec, 117, 132), scene_lon=_num(rec, 133, 148), scene_heading=_num(rec, 149, 164),
                       terrain_height=_num(rec, 309, 324, 0.0) * 1e3, wavelength=_num(rec, 501, 516),
                       chirp_constant=_num(rec, 535, 550, 0.0), chirp_rate=_num(rec, 551, 566),
                       sampling_rate=_num(rec, 711, 726) * 1e6, range_gate=_num(rec, 727, 742) * 1e-6,
                       pulse_length=_num(rec, 743, 758) * 1e-6, bits_per_sample=_num(rec, 799, 806),
                       dc_bias_i=_num(rec, 819, 834), dc_bias_q=_num(rec, 835, 850), prf=_num(rec, 935, 950) * 1e-3,
                       scene_id=_ascii(rec, 21, 52), mission=_ascii(rec, 397, 412))
        elif typ == 30:                                  # platform position
            npts = int(_num(rec, 141, 144))
            year, month, day = int(_num(rec, 145, 148)), int(_num(rec, 149, 152)), int(_num(rec, 153, 156))
            t0, dt = _num(rec, 161, 182), _num(rec, 183, 204)
            frame = _ascii(rec, 205, 268)
            pv = np.array([_num(rec, 387 + 22 * k, 408 + 22 * k) for k in range(6 * npts)], np.float64).reshape(npts, 6)
            out.update(sv_time=t0 + dt * np.arange(npts), sv_pos=pv[:, :3], sv_vel=pv[:, 3:],
                       sv_day=(year, month, day), sv_frame=frame)
    if 'sampling_rate' not in out or 'sv_pos' not in out:
        raise ValueError('not an ALOS PALSAR level 1.0 leader file: data set summary or platform position record missing')
    return out


def palsar_image_header(data):
    """The SAR data file descriptor of an ALOS PALSAR level 1.0 image file (bytes): records, bits per sample, bytes
    per sample group, samples per line, prefix bytes, SAR data bytes per record, and the record length."""
    rec = data[:720]
    out = dict(records=int(_num(rec, 181, 186)), bits_per_sample=int(_num(rec, 217, 220)), bytes_per_group=int(_num(rec, 225, 228)),
               lines=int(_num(rec, 237, 244)), prefix=int(_num(rec, 277, 280)), data_bytes=int(_num(rec, 281, 288)))
    # the pixels per line (bytes 249 to 256) are blank in JAXA's level 1.0 files (the look-alikes fill them): the
    # data bytes per record over the bytes per sample group
    samples = _num(rec, 249, 256)
    out['samples'] = int(samples) if samples else out['data_bytes'] // out['bytes_per_group']
    out['record_length'] = out['prefix'] + out['data_bytes']
    return out


def palsar_lines(mm, header, lo, hi):
    """Signal data records lo to hi of an ALOS PALSAR level 1.0 image file (a memory map or bytes): the per-line
    fields (line number, year, day of year, milliseconds of day, PRF in Hz, chirp length in s, slant range to the
    first sample in m, window position in s, microseconds since the 1PPS pulse) and the samples as complex64 [hi - lo, samples] (8-bit I and Q, the
    DC bias not removed)."""
    L = header['record_length']
    n = header['samples']
    recs = np.frombuffer(mm, np.uint8, count=(hi - lo) * L, offset=720 + lo * L).reshape(hi - lo, L)
    b4 = lambda a: recs[:, a - 1:a + 3].copy().view('>u4').ravel().astype(np.int64)
    fields = dict(line=b4(13), pixels=b4(25), year=b4(37), doy=b4(41), msec=b4(45), prf=b4(57) * 1e-3, chirp_length=b4(69) * 1e-9,
                  slant_range=b4(117).astype(np.float64), window=b4(121) * 1e-9,
                  # PALSAR auxiliary data (bytes 289 to 388): item 2, the microseconds since the 1PPS pulse, 20 bits of the
                  # second word (Appendix A-2)
                  pps_us=(b4(297) >> 8) & 0xFFFFF)
    raw_iq = recs[:, header['prefix']:header['prefix'] + 2 * n].reshape(hi - lo, n, 2).astype(np.float32)
    z = np.empty((hi - lo, n), np.complex64)
    z.real = raw_iq[:, :, 0]
    z.imag = raw_iq[:, :, 1]
    return fields, z
