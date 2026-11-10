"""Spatially variant apodization and the keystone trim (fastsar.weighting), on synthetic images with a rectangular
spectrum at twice Nyquist: sidelobes removed with the unweighted main lobe kept, two close targets still resolved,
zeros and edges, the options; and the trim's rectangle of spatial frequencies."""
import numpy as np
import pytest

from fastsar import weighting

n = 256


def scene(pts, os=2):
    """Point targets (row, col, amplitude) of an unweighted aperture filling 1/os of the band on both axes."""
    k = np.fft.fftfreq(n)
    out = np.zeros((n, n), complex)
    for i, j, a in pts:
        fi = (np.abs(k) < 0.5 / os) * np.exp(-2j * np.pi * k * i)
        fj = (np.abs(k) < 0.5 / os) * np.exp(-2j * np.pi * k * j)
        out += a * np.outer(np.fft.ifft(fi), np.fft.ifft(fj))
    return (out * os * os).astype(np.complex64)


def width(c):
    """3 dB width (samples) of the peak of a cut, from a 16 times upsampled copy."""
    C, m = np.fft.fft(c), len(c)
    Z = np.zeros(16 * m, complex)
    Z[:m // 2], Z[-m // 2:] = C[:m // 2], C[-m // 2:]
    u = np.abs(np.fft.ifft(Z))
    return (u > u.max() / np.sqrt(2)).sum() / 16


@pytest.mark.parametrize('joint', [True, False])
@pytest.mark.parametrize('os', [1, 2, 3])
def test_sidelobes_go_and_the_main_lobe_stays(joint, os):
    x = scene([(100.37, 140.61, 1.0 + 0.5j)], os)
    y = weighting.sva(x, os, joint=joint)
    assert y.dtype == np.complex64 and y.shape == x.shape
    far = np.abs(y).copy()
    far[100 - 3 * os:101 + 3 * os + 1, 140 - 3 * os:141 + 3 * os + 1] = 0
    assert 20 * np.log10(far.max() / np.abs(y).max()) < -40                 # the unweighted aperture's: -13 dB
    assert abs(np.abs(y).max() / np.abs(x).max() - 1) < 1e-3
    if os > 1:
        assert abs(width(y[:, 141]) - width(x[:, 141])) < 0.2 and abs(width(y[100]) - width(x[100])) < 0.2
    if joint:                                                               # the main lobe's phase is the target's
        assert abs(np.angle(y[100, 141] * np.conj(x[100, 141]))) < 1e-3


def test_two_close_targets_stay_resolved_and_a_weak_one_shows():
    x = scene([(100.0, 100.0, 1.0), (100.0, 104.0, 1.0), (100.0, 131.0, 0.1)])    # 2 resolution cells apart; -20 dB
    y = np.abs(weighting.sva(x))[100]
    assert y[102] < 0.3 * min(y[100], y[104])
    assert np.argmax(y[110:150]) in (20, 21, 22) and y[131] > 0.05                   # among sidelobes of 0.02 to 0.05 before
    assert np.abs(x)[100, 110:128].max() > 0.02 and np.delete(y[110:150], [19, 20, 21, 22, 23]).max() < 0.02


def test_zeros_edges_and_options():
    x = scene([(100.3, 140.6, 1.0), (4.0, 3.0, 1.0)])
    z = x.copy()
    z[:, :60], z[200:] = 0, 0
    y = weighting.sva(z)
    assert not y[:, :60].any() and not y[200:].any() and np.isfinite(y).all()
    ref = weighting.sva(x)
    w = x.copy()
    assert weighting.sva(w, inplace=True) is w and np.array_equal(w, ref) and not np.array_equal(x, ref)
    one = weighting.sva(x, axes=(1,))
    assert np.array_equal(weighting.sva(one, axes=(0,)), ref) and not np.array_equal(one, ref)
    assert np.array_equal(weighting.sva(x.astype(np.complex128)), ref)
    assert np.array_equal(weighting.sva(x, axes=()), x)
    for bad in (0, 1.5):
        with pytest.raises(ValueError, match='oversample'):
            weighting.sva(x, bad)
    with pytest.raises(ValueError, match='complex'):
        weighting.sva(np.abs(x))


def test_keystone_keeps_a_rectangle():
    P, K, fmin, df = 400, 300, 9.4e9, 1e6
    y = np.linspace(-300.0, 340.0, P)                                 # an aperture that is not centered
    ant = np.stack([np.full(P, -8e3), y, np.full(P, 5e3)], 1)
    scp, e1 = np.zeros(3), np.array([0.0, -1.0, 0.0])
    S = np.ones((P, K), np.complex64)
    lost, band = weighting.keystone(S, ant, fmin, df, scp, e1, symmetric=False)
    u = (scp - ant) / np.linalg.norm(scp - ant, axis=1, keepdims=True)
    ka = (fmin + df * np.arange(K))[None] * (u @ e1)[:, None]         # azimuth spatial frequency, times c / 2
    lo, hi = fmin * (u @ e1).min(), fmin * (u @ e1).max()
    inside = (ka >= lo - 1e-3 * abs(lo)) & (ka <= hi + 1e-3 * hi)
    assert np.array_equal(S != 0, inside) or (np.abs((S != 0).astype(int) - inside).sum() <= P)   # to within a sample a pulse
    assert S[:, 0].all() and 0.005 < lost < K * df / fmin and abs(lost - (S == 0).mean()) < 1e-12
    assert np.allclose(band, (2 * lo / 299792458.0, 2 * hi / 299792458.0))
    S = np.ones((P, K), np.complex64)                                  # the same width on both sides of zero
    lost2, band2 = weighting.keystone(S, ant, fmin, df, scp, e1)
    assert abs(band2[0] + band2[1]) < 1e-12 and abs(band2[1] - min(band[1], -band[0])) < 1e-12 and lost2 > lost
    assert np.abs((S != 0).astype(int) - (np.abs(ka) <= fmin * 299792458.0 / (2 * fmin) * band2[1] * (1 + 1e-3))).sum() <= P
