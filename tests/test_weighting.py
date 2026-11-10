"""Spatially variant apodization and the keystone trim (fastsar.weighting), on synthetic images with a rectangular
spectrum at twice Nyquist: sidelobes removed with the unweighted main lobe kept, two close targets still resolved,
zeros and edges, the options, the same image on a GPU (CuPy) and as a JAX array; and the trim's rectangle of spatial
frequencies."""
import numpy as np
import pytest

from fastsar import _xp, weighting

n = 256


@pytest.fixture(params=[pytest.param('cupy', marks=pytest.mark.cuda), 'jax'])
def device(request):
    """(kind, put): put(a) is the NumPy array a as a CuPy array (needs a GPU), or as a JAX array on JAX's CPU
    device, which runs the code a TPU runs."""
    if request.param == 'cupy':
        import cupy
        return 'cupy', cupy.asarray
    jnp = pytest.importorskip('jax.numpy')
    request.getfixturevalue('jax_cpu')
    return 'jax', jnp.asarray


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


def test_devices(device):
    """A CuPy or JAX image gives the NumPy result where the image is: CuPy to the bit, JAX to the rounding of the
    products in a weight (a few units in the last place of the brightest pixel, over three passes)."""
    kind, put = device
    x = scene([(100.3, 140.6, 1.0 + 0.5j), (4.0, 3.0, 1.0)])
    x[:, :60], x[200:] = 0, 0                                               # unprocessed: stays 0
    cases = (dict(), dict(joint=False), dict(axes=(0,)), dict(oversample=3, axes=(1, 0, 1)), dict(oversample=300),
             dict(axes=()))
    for kw in cases:
        ref = weighting.sva(x, **kw)
        y = weighting.sva(put(x), **kw)
        assert _xp.kind(y) == kind and y.dtype == np.complex64 and y.shape == x.shape, kw
        y = _xp.host(y)
        assert np.abs(y - ref).max() <= 1e-5 * np.abs(ref).max() and not y[x == 0].any(), kw
        assert kind != 'cupy' or np.array_equal(y, ref), kw
    ref = weighting.sva(x)
    assert np.abs(_xp.host(weighting.sva(put(x).T, axes=(0, 1))) - ref.T).max() <= 1e-6      # not contiguous
    w = put(x)
    y = weighting.sva(w, inplace=True)
    if kind == 'cupy':                                                      # written into the image
        assert y is w and np.array_equal(_xp.host(w), ref)
        assert np.array_equal(_xp.host(weighting.sva(put(x.astype(np.complex128)))), ref)
        w = put(x)
        assert weighting.sva(w, axes=(0,), inplace=True) is w
        assert np.array_equal(_xp.host(w), weighting.sva(x, axes=(0,)))
    else:                                                                   # a JAX array cannot be: a new one
        assert y is not w and np.array_equal(_xp.host(w), x) and np.abs(_xp.host(y) - ref).max() <= 1e-6
    with pytest.raises(ValueError, match='complex'):
        weighting.sva(put(np.abs(x)))


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


def _same_sva(got, ref, x):
    """A blocked result against the whole-image one: float32 rounding apart (the compiled programs of two array shapes
    may round differently, and a null of the apodization at 1e-11 of the peak then lands on zero or beside it), with
    the zeros of the input exactly zero in both."""
    got = np.asarray(got)
    return (not got[x == 0].any() and not ref[x == 0].any()
            and np.abs(got - ref).max() <= 2e-6 * np.abs(ref).max())


def test_sva_jax_in_row_blocks(monkeypatch):
    """A JAX image larger than the device's block budget is apodized in blocks of rows with the rows each pass
    along axis 0 reaches: the values of the whole-image program to float32 rounding, its zeros exactly, for every order
    of the axes, and through one compiled program."""
    jnp = pytest.importorskip('jax.numpy')
    from fastsar import weighting
    rng = np.random.default_rng(21)
    x = (rng.normal(size=(301, 257)) + 1j * rng.normal(size=(301, 257))).astype(np.complex64)
    x[:40, :50] = 0
    x[100] = 0
    for axes in ((1, 0), (0, 1), (0,), (1,), (0, 0, 1)):
        for joint, N in ((True, 2), (False, 3), (True, 400)):
            monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB', raising=False)
            ref = np.asarray(weighting.sva(jnp.asarray(x), N, axes=axes, joint=joint))
            assert not ref[:40, :50].any() and (N > 300 or not np.array_equal(ref, x))
            for mb in ('1.6', '0.6'):                               # 15 rows a slab with their halo, and 7
                monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', mb)
                got = weighting.sva(jnp.asarray(x), N, axes=axes, joint=joint)
                assert _same_sva(got, ref, x), (axes, joint, N, mb)
    size = weighting._DEVICE['core']._cache_size()                  # 21 blocks of rows, one program
    weighting.sva(jnp.asarray(x[::-1].copy()), 2)
    assert size > 0 and weighting._DEVICE['core']._cache_size() == size


@pytest.mark.cuda
def test_small_gpu_gives_the_same_image(monkeypatch):
    """With little device memory to count on (FASTSAR_CUDA_BLOCK_MB) apodization runs in slabs of a few rows, on a
    CuPy image and on a host image passed through the device (process_cphd's, with its ramps), to the bit."""
    cp = pytest.importorskip('cupy')
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((203, 150)) + 1j * rng.standard_normal((203, 150))).astype(np.complex64)
    x[:40, :60] = 0
    ramp = np.exp(-2j * np.pi * 0.013 * np.arange(203)).astype(np.complex64)[:, None]
    rd = (cp.asarray(ramp), cp.asarray(ramp.conj()))
    cases = (dict(), dict(joint=False), dict(axes=(0,)), dict(oversample=3, axes=(1, 0, 1, 0)), dict(axes=(1,)))
    refs = [weighting.sva(x, **kw) for kw in cases]
    ramped = _xp.host(weighting.sva(cp.asarray(x), inplace=True, _cuda=(rd,)))
    assert np.abs(ramped - weighting.sva(x * ramp) * ramp.conj()).max() <= 1e-5 * np.abs(x).max()
    monkeypatch.setenv('FASTSAR_CUDA_BLOCK_MB', '0.05')             # slabs of one to seven rows
    for kw, ref in zip(cases, refs):
        w = cp.asarray(x)
        y = weighting.sva(w, **kw)
        assert y is not w and np.array_equal(_xp.host(y), ref) and np.array_equal(_xp.host(w), x), kw
        assert weighting.sva(w, inplace=True, **kw) is w and np.array_equal(_xp.host(w), ref), kw
        h = x.copy()
        assert weighting.sva(h, inplace=True, _cuda=(), **kw) is h and np.array_equal(h, ref), kw
    for put in (cp.asarray, np.array):
        y = weighting.sva(put(x), inplace=True, _cuda=(rd,))
        assert np.array_equal(_xp.host(y), ramped)
    y = weighting.sva(x.copy(), inplace=True, _cuda=((rd[0], None),))
    assert np.array_equal(y, _xp.host(weighting.sva(cp.asarray(x) * rd[0])))


def test_sva_jax_image_on_the_host(monkeypatch):
    """sva through JAX's device in slabs, for an image on the host or on the device and a result on either
    (process_cphd's placement): the values of the whole-image program exactly, and with the ramps that center the
    band its values to float32 rounding (one program multiplies and apodizes)."""
    jnp = pytest.importorskip('jax.numpy')
    rng = np.random.default_rng(22)
    x = (rng.normal(size=(301, 257)) + 1j * rng.normal(size=(301, 257))).astype(np.complex64)
    x[:40, :50] = 0
    r = np.exp(1j * rng.normal(size=(301, 1))).astype(np.complex64)
    for axes in ((1, 0), (0, 1)):
        monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB', raising=False)
        ref = np.asarray(weighting.sva(jnp.asarray(x), 2, axes=axes))
        ramped = np.asarray(weighting.sva(jnp.asarray(x) * jnp.asarray(r), 2, axes=axes) * jnp.asarray(r.conj()))
        for mb in (None, '1.6', '0.3'):                             # one slab; 15 rows a slab with their halo; 4
            if mb:
                monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', mb)
            for src in (np.array, jnp.asarray):
                for where in ('host', 'device'):
                    got = weighting.sva(src(x), 2, axes=axes, _jax=(None, None, where))
                    assert _xp.kind(got) == ('numpy' if where == 'host' else 'jax')
                    assert _same_sva(_xp.host(got), ref, x)
                    got = _xp.host(weighting.sva(src(x), 2, axes=axes, _jax=(r, r.conj(), where)))
                    assert not got[x == 0].any() and np.abs(got - ramped).max() < 1e-5
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '0.001')
    with pytest.raises(MemoryError, match='apodization on the JAX device'):
        weighting.sva(x.copy(), 2, _jax=(None, None, 'host'))
