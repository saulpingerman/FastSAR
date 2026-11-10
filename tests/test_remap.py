"""Remaps: the eight remap functions against the NGA MATLAB SAR Toolbox's own (Visualization/remap/remap_funcs, run in
Octave 9.2 on the pixels stored in tests/data/remap_reference.npz), MATLAB's rounding, the toolbox's spline
percentile, the valid mask that keeps a zero-filled border out of the statistics, the order statistics from
histograms, and the same pixels on a GPU (CuPy) and as JAX arrays."""
import os

import numpy as np
import pytest

from fastsar import _xp, remap

REF = os.path.join(os.path.dirname(__file__), 'data', 'remap_reference.npz')
NAMES = ('density', 'brighter', 'darker', 'highcontrast', 'pedf', 'nrl', 'log')


@pytest.fixture(scope='module')
def ref():
    return dict(np.load(REF))


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


def speckle(shape=(700, 900), seed=5):
    """A speckle image (complex64) with bright points and an unprocessed corner of zeros."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    i, j = rng.integers(0, shape[0], 300), rng.integers(0, shape[1], 300)
    x[i, j] *= rng.uniform(5.0, 300.0, 300).astype(np.float32)
    x[:shape[0] // 2, :shape[1] // 2] = 0
    return x


@pytest.mark.parametrize('name', NAMES)
def test_matches_toolbox(ref, name):
    y = remap.REMAPS[name](ref['x'])
    assert y.dtype == np.uint8 and y.shape == ref['x'].shape
    assert np.array_equal(y, ref[name]), np.count_nonzero(y != ref[name])
    assert np.array_equal(remap.remap(ref['x'], name + 'remap'), y)


def test_nrl_parameters_and_linear(ref):
    assert np.array_equal(remap.nrl(ref['x'], a=2, c=200), ref['nrl_a2_c200'])
    assert np.allclose(remap.linear(ref['x']), ref['linear'], rtol=1e-12)
    real = np.abs(ref['x'])
    assert np.array_equal(remap.linear(real), real)
    with pytest.raises(ValueError):
        remap.remap(ref['x'], 'gamma')


def test_matlab_rounding():
    x = np.array([-3.0, 0.49, 0.5, 1.5, 2.5, 254.49, 254.5, 300.0, np.nan], np.float32)
    assert remap._uint8(x).tolist() == [0, 0, 1, 2, 3, 254, 255, 255, 0]


def test_percentile_is_the_global_spline():
    from scipy.interpolate import CubicSpline
    v = np.random.default_rng(1).rayleigh(size=5000).astype(np.float32)
    n = v.size
    full = CubicSpline((np.arange(n) + 0.5) / n, np.sort(v).astype(np.float64), bc_type='not-a-knot')
    for p in (0.001, 1, 50, 99, 99.999):
        assert abs(remap.percentile(v, p) - full(p / 100)) < 2e-6 * abs(full(p / 100)) + 1e-7, p
    assert remap.percentile(np.array([3.0]), 99) == 3.0
    assert np.isclose(remap.percentile(np.array([1.0, 2.0]), 50), 1.5)


def test_valid_mask(ref):
    x = ref['x']
    big = np.zeros((x.shape[0] + 20, x.shape[1] + 30), x.dtype)
    big[5:5 + x.shape[0], 7:7 + x.shape[1]] = x
    valid = np.zeros(big.shape, bool)
    valid[5:5 + x.shape[0], 7:7 + x.shape[1]] = True
    for name in NAMES:
        y = remap.REMAPS[name](big, valid=valid)
        assert np.array_equal(y[valid].reshape(x.shape), ref[name]), name     # the border leaves the statistics alone
        assert not y[~valid].any(), name
    assert np.any(remap.density(big)[valid].reshape(x.shape) != ref['density'])   # without the mask it does not


def test_row_blocks_and_shapes(monkeypatch):
    """The host takes the image in row blocks on threads: the result is that of one block, whatever the shape."""
    x = speckle()                                                                 # 630,000 pixels: three blocks
    valid = x != 0
    got = {(name, m is None): remap.REMAPS[name](x, valid=m) for name in NAMES + ('linear',) for m in (None, valid)}
    monkeypatch.setattr(remap, '_each', lambda x, work: work(slice(0, x.shape[0])))
    for (name, whole), y in got.items():
        one = remap.REMAPS[name](x, valid=None if whole else valid)
        assert one.dtype == y.dtype and np.array_equal(one, y), (name, whole)
    monkeypatch.undo()
    for name in NAMES:
        y = got[name, False]
        assert np.array_equal(remap.REMAPS[name](x.ravel(), valid=valid.ravel()).reshape(x.shape), y), name
        assert np.array_equal(remap.REMAPS[name](np.abs(x), valid=valid), y), name       # a detected image
        assert not remap.REMAPS[name](x, valid=valid)[~valid].any(), name
        with np.errstate(invalid='ignore'):                                       # one pixel has no span for log
            assert remap.REMAPS[name](x[3, 500]).shape == (), name
    with pytest.raises(ValueError):
        remap.nrl(x, valid=np.zeros(x.shape, bool))


def test_order_statistics_from_histograms():
    """remap._ranks: the order statistics a sort gives, from two passes of histograms over the values' bit patterns,
    when the ranks asked for lie in one first-pass bin, in two far apart, or in more bins than a pass resolves."""
    rng = np.random.default_rng(2)
    n = 300_000
    for v in (rng.rayleigh(size=n), np.exp(12 * rng.standard_normal(n)), np.where(rng.random(n) < 0.9899999, 1.0, 5.0),
              rng.standard_normal(n), np.full(n, 2.5), np.r_[rng.rayleigh(size=n - 2), np.inf, 0.0]):
        v = v.astype(np.float32)
        s = np.sort(v)
        m, take = remap._ranks(lambda lut: remap._hist(v, None, lut, amp=False))
        ranges = ((0, 1), (n - 1, n), (n - 64, n), (n // 2 - 32, n // 2 + 32), (int(0.99 * n) - 32, int(0.99 * n) + 32))
        assert m == n
        for (lo, hi), got in zip(ranges, take(*ranges)):
            assert got.dtype == np.float32 and np.array_equal(got, s[lo:hi]), (lo, hi)
    mask = rng.random(n) < 0.3                                                    # over the valid values only
    m, take = remap._ranks(lambda lut: remap._hist(v, mask, lut))
    assert m == mask.sum() and np.array_equal(take((5, 70))[0], np.sort(v[mask])[5:70])


def test_devices(ref, device):
    """CuPy and JAX arrays: the NumPy counts but for values on a rounding boundary (never more than one count, on at
    most one pixel in 1,000), as arrays of the same kind; the same percentile; zeros outside the mask."""
    kind, put = device

    def close(got, want, what):
        assert _xp.kind(got) == kind and got.dtype == np.uint8 and got.shape == want.shape, what
        d = np.abs(_xp.host(got).astype(int) - want)
        assert d.max() <= 1 and np.count_nonzero(d) <= 1e-3 * d.size, (what, d.max(), np.count_nonzero(d))
    for x in (ref['x'], speckle()):
        valid = (x != 0) if not x.all() else np.abs(x) > 0.3
        for name in NAMES:
            close(remap.REMAPS[name](put(x)), remap.REMAPS[name](x), name)
            y = remap.remap(put(x), name, valid=put(valid))
            close(y, remap.REMAPS[name](x, valid=valid), name)
            assert not _xp.host(y)[~valid].any(), name
        close(remap.nrl(put(x), a=2, c=200, valid=valid), remap.nrl(x, a=2, c=200, valid=valid), 'nrl parameters')
        close(remap.density(put(x), mean=0.7), remap.density(x, mean=0.7), 'density of a known mean')
        close(remap.log(put(x), span_db=40.0), remap.log(x, span_db=40.0), 'log span')
        lin = remap.linear(put(x), valid=put(valid))
        assert _xp.kind(lin) == kind and lin.dtype == (np.float64 if kind == 'cupy' else np.float32)
        assert np.allclose(_xp.host(lin), remap.linear(x, valid=valid), rtol=1e-6)
        d = remap.amplitude_to_density(put(x), valid=valid)
        assert _xp.kind(d) == kind and d.dtype == np.float32
        assert np.allclose(_xp.host(d), remap.amplitude_to_density(x, valid=valid), atol=1e-3)
        v = np.abs(x)[valid]
        for p in (1, 50, 99, 99.99):                                              # the same order statistics
            assert remap.percentile(put(v), p) == remap.percentile(v, p), p
        assert remap.percentile(put(x.real.ravel()), 30) == remap.percentile(x.real.ravel(), 30)
    with pytest.raises(ValueError):
        remap.remap(put(x), 'gamma')
