"""Repeat-pass products: coherent change detection on a simulated clutter pair, and interferometric height from two
passes with a vertical baseline formed onto one ground grid (no coregistration or flat-earth step: scatterers on the
grid surface give zero phase), then onto a grid that follows the terrain (zero phase everywhere). Needs finufft
(sim.simulate)."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, binary_erosion

import fastsar
from fastsar import sim, products

pytestmark = pytest.mark.needs('finufft')

C = 299792458.0
scene, res, n, sp = 40.0, 0.5, 96, 0.4
GRID = dict(nx=n, ny=n, spx=sp, spy=sp, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))


@pytest.fixture(scope='module')
def setup():
    """The grid coordinates and the random scenes of both checks, drawn in the original order from one generator."""
    rng = np.random.default_rng(6)
    xs = (np.arange(n) - n / 2) * sp
    X, Y = np.meshgrid(xs, xs, indexing='ij')           # X along e1 (= y of the simulator), Y along e2 (= x)
    pair = sim.clutter_pair(scene, res, rng, per_cell=3, gamma_t=0.98, n_bright=0)
    xy = (rng.random((int(3 * (scene / res) ** 2), 2)) - 0.5) * scene
    hz = np.where((np.abs(xy[:, 1] - 4) < 7) & (np.abs(xy[:, 0] + 3) < 6), 6.0, 0.0)
    pos = np.concatenate([xy, hz[:, None]], 1)
    amp = (rng.standard_normal(len(pos)) + 1j * rng.standard_normal(len(pos))) / np.sqrt(2)
    inner = (np.abs(X) < 0.4 * scene) & (np.abs(Y) < 0.4 * scene)
    return SimpleNamespace(X=X, Y=Y, pair=pair, pos=pos, amp=amp, inner=inner)


def test_change_detection(setup):
    """Same geometry, coherence 0.98 outside the change mask, independent clutter inside."""
    X, Y, inner = setup.X, setup.Y, setup.inner
    col = sim.make_collect(res=res, scene=scene, r0=5e3)
    pos, a1, a2 = setup.pair
    I1, I2 = (fastsar.form_image(sim.simulate(col, pos, a).astype(np.complex64), col.ant, col.fmin, col.df, **GRID, backend='cpu')
              for a in (a1, a2))
    g = products.coherence(I1, I2, 7, 7)
    ch = sim.change_mask(Y, X, scene)
    far = ~binary_dilation(ch, iterations=6) & inner
    core = ch & ~binary_dilation(~ch, iterations=3)
    print(f'coherence: unchanged {np.median(g[far]):.3f}, changed {np.median(g[core]):.3f}')
    assert np.median(g[far]) > 0.85 and np.median(g[core]) < 0.4


@pytest.fixture(scope='module')
def ifg(setup):
    """A 6 m block on flat ground, passes 3.5 m apart vertically."""
    X, Y = setup.X, setup.Y
    h = np.where((np.abs(X - 4) < 7) & (np.abs(Y + 3) < 6), 6.0, 0.0)
    cols = [sim.make_collect(res=res, scene=scene, r0=5e3, offset=(0.0, 0.0, dz)) for dz in (0.0, 3.5)]
    S = [sim.simulate(c, setup.pos, setup.amp).astype(np.complex64) for c in cols]
    # the block's footprint moved by its layover
    lay = products.project(np.array([0.0, 0.0, 6.0]), cols[0].ant, **GRID) - n / 2
    blk = (np.abs(X - lay[0] * sp - 4) < 5) & (np.abs(Y - lay[1] * sp + 3) < 4)
    flat = (h == 0) & setup.inner & ~binary_dilation(h > 0, iterations=8) & ~binary_dilation(blk, iterations=8)
    return SimpleNamespace(h=h, cols=cols, S=S, lay=lay, blk=blk, flat=flat)


def test_interferometric_height(ifg):
    cols, S, blk, flat, lay = ifg.cols, ifg.S, ifg.blk, ifg.flat, ifg.lay
    J1, J2 = (fastsar.form_image(s, c.ant, c.fmin, c.df, **GRID, backend='cpu') for s, c in zip(S, cols))
    ph = np.angle(products._box(J1 * np.conj(J2), 5, 5))
    # phase per meter of height: a point s raised by 1 m appears at its layover position p on the grid (project, pass 1)
    # with phase 4 pi fc / c ((R1(p) - R1(s)) - (R2(p) - R2(s))) in J1 conj(J2)
    fc = cols[0].fref
    s1 = np.array([0.0, 0.0, 1.0])
    ij = products.project(s1, cols[0].ant, **GRID)
    p1 = (ij[0] - n / 2) * sp * np.array(GRID['e1']) + (ij[1] - n / 2) * sp * np.array(GRID['e2'])
    dR = lambda c, x: np.linalg.norm(x - c.ant, axis=1) - np.linalg.norm(c.ant, axis=1)
    w = np.hanning(cols[0].Np)
    kz = 4 * np.pi * fc / C * np.sum(w * ((dR(cols[0], p1) - dR(cols[0], s1)) - (dR(cols[1], p1) - dR(cols[1], s1)))) / w.sum()
    est = ph / kz
    print(f'layover of the block {np.hypot(*lay) * sp:.1f} m, height of ambiguity {2 * np.pi / abs(kz):.1f} m; estimated height: block {np.median(est[blk]):.2f} m (true 6), '
          f'flat ground {np.median(est[flat]):.2f} m; coherence on flat ground {np.median(products.coherence(J1, J2)[flat]):.3f}')
    assert abs(np.median(est[blk]) - 6.0) < 0.5 and abs(np.median(est[flat])) < 0.3


def test_formed_on_terrain(ifg):
    """Onto the terrain itself (exact backprojection on the grid lifted by h): the phase vanishes everywhere."""
    h, cols, S, flat = ifg.h, ifg.cols, ifg.S, ifg.flat
    pts = fastsar.plane_points(**GRID, height=h)
    K1, K2 = (fastsar.backproject(s, c.ant, c.fmin, c.df, pts, backend='cpu') for s, c in zip(S, cols))
    ph2 = np.angle(products._box(K1 * np.conj(K2), 5, 5))
    top = binary_erosion(h > 0, iterations=3)
    print(f'formed on the terrain: median |phase| block top {np.median(np.abs(ph2[top])):.3f} rad, flat {np.median(np.abs(ph2[flat])):.3f} rad')
    assert np.median(np.abs(ph2[top])) < 0.15 and np.median(np.abs(ph2[flat])) < 0.15
