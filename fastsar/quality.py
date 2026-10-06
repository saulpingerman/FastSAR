"""Point-target quality of a complex SAR image: 3 dB resolution, peak and integrated sidelobe ratios and the peak
position, measured on a band-limited upsampling of the impulse response.

    m = fastsar.quality.point_target(img, (i, j), d_az=0.31, d_rg=1.2)

Axis 0 of the image is azimuth and axis 1 range. The band of a SAR image is generally off center (Doppler centroid,
range carrier), and in a squinted zero-Doppler image the range band moves with azimuth frequency. A patch around the
peak is therefore transformed in azimuth, each azimuth-frequency row is upsampled in range about its own band (the
circular run of bins with the least energy is taken as the gap), and the azimuth band is upsampled the same way. This
is exact for an image whose band, per azimuth frequency, is narrower than the sampling rate in both directions.

Resolution, PSLR and the one-dimensional ISLR are taken on cuts through the peak along the image axes; the main lobe
of a cut runs between the first minima on either side of the peak. The two-dimensional ISLR takes as main lobe the
rectangle spanned by the two cuts' main lobes, and the whole patch as the window.
"""
import numpy as np


def _gap_shift(e):
    """Circular shift that moves the least-energy run of bins of e to index n/2."""
    n = len(e)
    k = max(1, n // 16)
    sm = np.convolve(np.concatenate([e[-k:], e, e[:k]]), np.ones(2 * k + 1), 'valid')
    return n // 2 - int(np.argmin(sm))


def _row_shifts(G, s_az):
    """Per-row range shifts, unwrapped along azimuth frequency starting from the azimuth gap. A band is known only
    modulo the sampling rate; rows that pick different aliases would be modulated differently between samples."""
    m, n = G.shape
    order = np.argsort((np.arange(m) + s_az - m // 2) % m)
    c = np.array([-_gap_shift(np.abs(g) ** 2) for g in G], np.float64)[order]
    c = np.unwrap(c * 2 * np.pi / n) * n / (2 * np.pi)
    out = np.empty(m, np.int64)
    out[order] = -np.round(c).astype(np.int64)
    return out


def _up_axis1(G, up, shifts):
    """Rows of G are spectra (along axis 1); upsample each about its band and return fine time samples."""
    m, n = G.shape
    k = np.arange(n * up)
    out = np.empty((m, n * up), np.complex128)
    for i in range(m):
        g = np.roll(G[i], shifts[i])
        g = np.concatenate([g[:n // 2], np.zeros(n * (up - 1)), g[n // 2:]])
        out[i] = np.fft.ifft(g) * up * np.exp(-2j * np.pi * shifts[i] * k / (n * up))
    return out


def upsample(patch, up=16):
    """Band-limited upsampling of a complex patch [m, n] (even sizes) by `up` in both directions."""
    A = np.fft.fft(np.asarray(patch, np.complex128), axis=0)
    G = np.fft.fft(A, axis=1)
    s = _gap_shift(np.sum(np.abs(G) ** 2, 1))
    A = _up_axis1(G, up, _row_shifts(G, s))                                # [azimuth frequency, fine range]
    return _up_axis1(A.T, up, [s] * A.shape[1]).T


def _cut(P, c, d):
    """3 dB width, PSLR (dB), ISLR (dB) and main-lobe bounds of a power cut P with its peak at index c."""
    lo = c
    while lo > 0 and P[lo - 1] < P[lo]:
        lo -= 1
    hi = c
    while hi < len(P) - 1 and P[hi + 1] < P[hi]:
        hi += 1
    a = c + np.argmax(P[c:] < 0.5 * P[c])
    b = c - np.argmax(P[c::-1] < 0.5 * P[c])
    yr = a - 1 + (P[a - 1] - 0.5 * P[c]) / (P[a - 1] - P[a])
    yl = b + 1 - (P[b + 1] - 0.5 * P[c]) / (P[b + 1] - P[b])
    side = np.concatenate([P[:lo], P[hi + 1:]])
    return ((yr - yl) * d, 10 * np.log10(side.max() / P[c]), 10 * np.log10(side.sum() / P[lo:hi + 1].sum()), lo, hi)


def point_target(img, ij, d_az=1.0, d_rg=1.0, half=32, up=16):
    """Metrics of the point target whose peak is near pixel ij = (i, j). d_az, d_rg: pixel spacings (m).
    -> dict(i, j: fractional peak position in pixels; res_az, res_rg (m); pslr_az, pslr_rg, islr_az, islr_rg,
    islr_2d (dB); peak: amplitude at the peak). The upsampled phase is not returned: the band of a
    sampled image is known only up to a multiple of the sampling rate, which leaves the phase between samples open."""
    i, j = (int(v) for v in ij)
    patch = img[i - half:i + half, j - half:j + half]
    if patch.shape != (2 * half, 2 * half):
        raise ValueError('target too close to the image edge for this patch size')
    U = upsample(patch, up)
    P = np.abs(U) ** 2
    a, b = np.unravel_index(np.argmax(P), P.shape)

    def vertex(y0, y1, y2):
        den = y0 - 2 * y1 + y2
        return 0.5 * (y0 - y2) / den if den != 0 else 0.0
    da = vertex(*P[a - 1:a + 2, b]) if 0 < a < P.shape[0] - 1 else 0.0
    db = vertex(*P[a, b - 1:b + 2]) if 0 < b < P.shape[1] - 1 else 0.0
    res_az, pslr_az, islr_az, alo, ahi = _cut(P[:, b], a, d_az / up)
    res_rg, pslr_rg, islr_rg, rlo, rhi = _cut(P[a, :], b, d_rg / up)
    main = P[alo:ahi + 1, rlo:rhi + 1].sum()
    return dict(i=i - half + (a + da) / up, j=j - half + (b + db) / up, res_az=res_az, res_rg=res_rg,
                pslr_az=pslr_az, pslr_rg=pslr_rg, islr_az=islr_az, islr_rg=islr_rg,
                islr_2d=10 * np.log10((P.sum() - main) / main), peak=float(np.abs(U[a, b])))
