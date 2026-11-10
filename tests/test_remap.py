"""Remaps: the eight remap functions against the NGA MATLAB SAR Toolbox's own (Visualization/remap/remap_funcs, run in
Octave 9.2 on the pixels stored in tests/data/remap_reference.npz), MATLAB's rounding, the toolbox's spline
percentile, and the valid mask that keeps a zero-filled border out of the statistics."""
import os

import numpy as np
import pytest

from fastsar import remap

REF = os.path.join(os.path.dirname(__file__), 'data', 'remap_reference.npz')
NAMES = ('density', 'brighter', 'darker', 'highcontrast', 'pedf', 'nrl', 'log')


@pytest.fixture(scope='module')
def ref():
    return dict(np.load(REF))


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
