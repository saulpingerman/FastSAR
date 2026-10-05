"""Synthetic spotlight SAR collects in float64: geometry, scenes, phase history.

The phase history follows the CPHD convention of motion compensation to the
scene centre:  S[p, k] = sum_n sigma_n exp(-j 4 pi f_k dR_pn / c)  with
dR_pn = |x_n - a_p| - |a_p|.  It is synthesised with a type-1 NUFFT per pulse
(finufft), which is exact to the requested tolerance and makes million-scatterer
clutter scenes cheap.
"""
from dataclasses import dataclass

import numpy as np

C = 299792458.0


@dataclass
class Collect:
    fmin: float          # Hz, frequency of sample k=0
    df: float            # Hz, frequency step
    K: int               # frequency samples per pulse
    ant: np.ndarray      # [Np, 3] antenna phase centre, scene-centred metres
    res: float           # nominal slant-plane resolution, metres

    @property
    def Np(self):
        return self.ant.shape[0]

    @property
    def fref(self):
        """Frequency of sample K//2, used as the carrier after range compression."""
        return self.fmin + (self.K // 2) * self.df

    @property
    def freqs(self):
        return self.fmin + self.df * np.arange(self.K)


def _even(n):
    n = int(np.ceil(n))
    return n + (n % 2)


def make_collect(res=0.3, scene=150.0, r0=10e3, graze_deg=30.0, fc=9.6e9,
                 margin=1.25, offset=(0.0, 0.0, 0.0)):
    """Broadside linear-track spotlight collect looking along +x.

    `scene` is the side of the square ground patch that must stay unaliased.
    `offset` shifts the whole track (repeat-pass baseline).
    """
    lam = C / fc
    bw = C / (2.0 * res)
    extent = margin * np.sqrt(2.0) * scene
    df = C / (2.0 * extent)
    K = _even(bw / df)
    psi = np.deg2rad(graze_deg)
    L = r0 * lam / (2.0 * res)                 # aperture length
    dy = lam * r0 / (2.0 * extent)             # pulse spacing along track
    Np = _even(L / dy)
    y = (np.arange(Np) - (Np - 1) / 2.0) * dy
    ant = np.stack([np.full(Np, -r0 * np.cos(psi)), y,
                    np.full(Np, r0 * np.sin(psi))], axis=1)
    ant = ant + np.asarray(offset, dtype=np.float64)[None, :]
    return Collect(fmin=fc - (K // 2) * df, df=df, K=K, ant=ant, res=res)


def delta_range(ant, pos):
    """dR[p, n] in float64 for antenna rows `ant` and scatterer rows `pos`."""
    d = pos[None, :, :] - ant[:, None, :]
    return np.sqrt(np.einsum('pnk,pnk->pn', d, d)) - np.linalg.norm(ant, axis=1)[:, None]


def simulate(col, pos, amp, eps=1e-12, nthreads=0):
    """Phase history [Np, K] complex128 via one type-1 NUFFT per pulse."""
    import finufft
    out = np.empty((col.Np, col.K), np.complex128)
    kw = dict(eps=eps, isign=-1)
    if nthreads:
        kw['nthreads'] = nthreads
    r0 = np.linalg.norm(col.ant, axis=1)
    for p in range(col.Np):
        d = pos - col.ant[p]
        dr = np.sqrt(np.einsum('nk,nk->n', d, d)) - r0[p]
        x = (4.0 * np.pi * col.df / C) * dr
        c = amp * np.exp(-1j * (4.0 * np.pi * col.fref / C) * dr)
        out[p] = finufft.nufft1d1(x, c, col.K, **kw)
    return out


def simulate_brute(col, pos, amp):
    """Direct O(Np K N) sum, for validating `simulate` on small cases."""
    dr = delta_range(col.ant, pos)
    ph = -4.0 * np.pi / C * col.freqs[None, :, None] * dr[:, None, :]
    return (amp[None, None, :] * np.exp(1j * ph)).sum(-1)


def add_noise(S, sigma, rng):
    n = rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape)
    return S + (sigma / np.sqrt(2.0)) * n


# ---------------------------------------------------------------- scenes

def change_mask(x, y, scene, part='all'):
    """True where the ground was disturbed between passes.

    Two rectangles plus a 1 m wide diagonal track, all expressed as fractions
    of the scene so the layout scales.
    """
    s = scene
    r1 = (np.abs(x - 0.22 * s) < 0.06 * s) & (np.abs(y + 0.20 * s) < 0.04 * s)
    r2 = (np.abs(x + 0.25 * s) < 0.03 * s) & (np.abs(y - 0.15 * s) < 0.09 * s)
    d = (x + y) / np.sqrt(2.0)
    along = (x - y) / np.sqrt(2.0)
    track = (np.abs(d + 0.05 * s) < 0.5) & (np.abs(along) < 0.3 * s)
    return track if part == 'track' else r1 | r2 | track


def clutter_pair(scene, res, rng, per_cell=4, gamma_t=0.98, n_bright=3,
                 bright_db=30.0):
    """Scatterers for a repeat-pass pair: positions, pass-1 and pass-2 amplitudes.

    Unchanged clutter keeps coherence `gamma_t`; clutter inside the change mask
    is redrawn. A few bright stable reflectors are added to load the dynamic
    range the way corner reflectors and buildings do in real scenes.
    """
    n = int(per_cell * (scene / res) ** 2)
    xy = (rng.random((n, 2)) - 0.5) * scene
    pos = np.concatenate([xy, np.zeros((n, 1))], axis=1)

    def cn(m):
        return (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2.0)

    a1 = cn(n)
    a2 = gamma_t * a1 + np.sqrt(1.0 - gamma_t ** 2) * cn(n)
    ch = change_mask(xy[:, 0], xy[:, 1], scene)
    a2[ch] = cn(int(ch.sum()))

    bxy = (rng.random((n_bright, 2)) - 0.5) * 0.7 * scene
    bpos = np.concatenate([bxy, np.zeros((n_bright, 1))], axis=1)
    # amplitude such that the focused peak sits bright_db above mean clutter
    bamp = np.full(n_bright, np.sqrt(per_cell) * 10 ** (bright_db / 20.0), np.complex128)
    pos = np.concatenate([pos, bpos])
    return pos, np.concatenate([a1, bamp]), np.concatenate([a2, bamp])


def point_scene(scene):
    """Isolated unit point targets at off-grid positions (centre, mid, corner)."""
    s = scene
    xy = np.array([[0.013, -0.021], [0.21 * s + 0.07, 0.19 * s - 0.04],
                   [-0.36 * s + 0.03, -0.37 * s + 0.11], [0.35 * s, -0.1 * s + 0.05],
                   [-0.12 * s + 0.09, 0.33 * s]])
    pos = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
    return pos, np.ones(len(xy), np.complex128)


def taylor_2d(Np, K, nbar=4, sll=35.0):
    from scipy.signal.windows import taylor
    return taylor(Np, nbar=nbar, sll=sll, norm=False)[:, None] * \
        taylor(K, nbar=nbar, sll=sll, norm=False)[None, :]


def ground_grid(n, spacing):
    """Flattened ground-plane pixel coordinates for an n x n image, x = axis 0."""
    ax = (np.arange(n) - n / 2.0) * spacing
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return X.ravel(), Y.ravel(), np.zeros(n * n)


# ------------------------------------------------- harder change-detection scene

STRATA_DB = (0.0, -10.0, -20.0, -30.0)
CLASSES = (0.9, 0.7, 0.5)          # pass-to-pass coherence of the partially changed patches


def hdr_layout(scene):
    """Geometry of the high-dynamic-range scene: four reflectivity strata as
    bands in y, a grid of 4 m patches with reduced coherence in every stratum,
    and a 1 m wide strip that crosses all strata."""
    s = scene
    edges = np.linspace(-0.44 * s, 0.44 * s, len(STRATA_DB) + 1)
    patches = []                                    # (x0, x1, y0, y1, class index)
    for b in range(len(STRATA_DB)):
        yc = 0.5 * (edges[b] + edges[b + 1])
        for i in range(9):
            xc = (-0.36 + 0.09 * i) * s
            patches.append((xc - 2.0, xc + 2.0, yc - 2.0, yc + 2.0, i % len(CLASSES)))
    return edges, patches


def hdr_stratum(y, scene):
    edges, _ = hdr_layout(scene)
    return np.clip(np.searchsorted(edges, y) - 1, 0, len(STRATA_DB) - 1)


def hdr_change_class(x, y, scene):
    """-1 unchanged, 0..len(CLASSES)-1 partial-change class, len(CLASSES) the strip."""
    _, patches = hdr_layout(scene)
    cls = np.full(np.shape(x), -1)
    for x0, x1, y0, y1, c in patches:
        cls = np.where((x >= x0) & (x < x1) & (y >= y0) & (y < y1), c, cls)
    strip = (np.abs(x - 0.40 * scene) < 0.5) & (np.abs(y) < 0.42 * scene)
    return np.where(strip, len(CLASSES), cls)


def clutter_pair_hdr(scene, res, rng, per_cell=4, gamma_t=0.98, bright_db=50.0):
    """Repeat-pass scatterers with reflectivity strata 0 to -30 dB, patches of
    partial decorrelation, a fully changed 1 m strip, and two reflectors
    `bright_db` above the brightest stratum's clutter power per resolution cell."""
    n = int(per_cell * (scene / res) ** 2)
    xy = (rng.random((n, 2)) - 0.5) * scene
    pos = np.concatenate([xy, np.zeros((n, 1))], axis=1)

    def cn(m):
        return (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2.0)

    amp = 10.0 ** (np.asarray(STRATA_DB)[hdr_stratum(xy[:, 1], scene)] / 20.0)
    cls = hdr_change_class(xy[:, 0], xy[:, 1], scene)
    gam = np.full(n, gamma_t)
    for c, g in enumerate(CLASSES):
        gam[cls == c] = g
    gam[cls == len(CLASSES)] = 0.0
    e1 = cn(n)
    e2 = gam * e1 + np.sqrt(1.0 - gam ** 2) * cn(n)
    bxy = np.array([[-0.41 * scene, -0.30 * scene], [-0.41 * scene, 0.31 * scene]])
    bpos = np.concatenate([bxy, np.zeros((2, 1))], axis=1)
    bamp = np.full(2, np.sqrt(per_cell) * 10 ** (bright_db / 20.0), np.complex128)
    return (np.concatenate([pos, bpos]), np.concatenate([amp * e1, bamp]), np.concatenate([amp * e2, bamp]),
            float(n))                               # n = clutter power per sample of a uniformly bright scene
