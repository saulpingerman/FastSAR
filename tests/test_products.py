"""Products: targets above the image plane appear where project predicts (layover), a target on the plane projects
onto itself, geocode samples the image there, multilook keeps the mean power, and the Pauli decomposition sorts
canonical scatterers into its channels. (The SICD round trip is in tests/test_chain.py.)"""
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import sim, products, quality

GRID = dict(nx=256, ny=256, spx=0.4, spy=0.4, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))


@pytest.fixture(scope='module')
def scene():
    col = sim.make_collect(res=0.5, scene=100.0, r0=6e3, graze_deg=35.0)
    tg = np.array([[0.0, 0.0, 0.0], [12.0, -15.0, 10.0], [-15.0, 8.0, 20.0], [10.0, 10.0, 30.0]])
    S = sim.simulate_brute(col, tg, np.ones(len(tg))).astype(np.complex64)
    img = fastsar.form_image(S, col.ant, col.fmin, col.df, **GRID, backend='jax')
    ij = products.project(tg, col.ant, **GRID)
    worst, peaks = 0.0, []
    for k, (x, p) in enumerate(zip(tg, ij)):
        # the actual peak near the predicted pixel
        i0, j0 = (int(round(v)) for v in p)
        w = np.abs(img[i0 - 6:i0 + 7, j0 - 6:j0 + 7])
        a, b = np.unravel_index(np.argmax(w), w.shape)
        m = quality.point_target(img, (i0 - 6 + a, j0 - 6 + b), GRID["spx"], GRID["spy"], half=16)
        d = np.hypot((m['i'] - p[0]) * GRID['spx'], (m['j'] - p[1]) * GRID['spy'])
        peaks.append(w.max())
        plane = np.array([(p[0] - GRID['nx'] / 2) * GRID['spx'], (p[1] - GRID['ny'] / 2) * GRID['spy']])
        lay = np.hypot(plane[0] - x[1], plane[1] - x[0])
        print(f'target at height {x[2]:4.0f} m: layover {lay:6.2f} m, peak found {d * 100:.2f} cm from the projection')
        worst = max(worst, d)
    return SimpleNamespace(col=col, tg=tg, img=img, ij=ij, worst=worst, peaks=np.array(peaks))


def test_layover(scene):
    assert scene.worst < 0.05, scene.worst
    assert np.hypot(*(scene.ij[0] - [GRID['nx'] / 2, GRID['ny'] / 2])) < 1e-9


def test_geocode(scene):
    """The amplitude at each target's 3-D position, bilinear between pixels, is near its peak pixel's."""
    amp = products.geocode(np.abs(scene.img), GRID, scene.col.ant, scene.tg)
    print('geocoded amplitude / brightest pixel of each target:', np.round(amp / scene.peaks, 3))
    assert (amp / scene.peaks > 0.7).all()


def test_multilook(scene):
    """multilook keeps the mean power."""
    img = scene.img
    ml = products.multilook(img, 4, 2)
    assert ml.shape == (64, 128) and abs(ml.mean() / (np.abs(img) ** 2).mean() - 1) < 1e-6


def test_pauli():
    """A trihedral (HH = VV), a dihedral (HH = -VV) and a dipole at 45 degrees (HV only) land in the surface,
    double-bounce and volume channels; total power is kept."""
    hh, hv, vv = np.array([[1.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, -1.0, 0.0]])
    p = products.pauli(hh, hv, vv)[0]
    print('pauli (double bounce, volume, surface):', np.round(p, 3).tolist())
    assert np.allclose(p, [[0, 0, 2], [2, 0, 0], [0, 2, 0]])
