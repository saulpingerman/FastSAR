"""Factorized backprojection, second version.

The algorithm is that of `ffbp.py` (recursive division of the image into tiles;
at each level the data are re-referenced to a child tile and low-pass filtered
and decimated in frequency and in pulse index; the smallest tiles are formed by
a separable matrix product). This version differs in four ways.

  * The image is a rectangle of any size. It is padded to a whole number of
    tiles whose counts factor into the per-level splits, and cropped at the end.
  * The decimation filters can be applied as dense matrix products (suited to
    the matrix units of a TPU), as strided convolutions, or as a sum over filter
    taps (both suited to devices where a gather or a convolution is cheap).
  * The rotation phases of the first two levels are computed in float64 on the
    host. The offsets between a first-level tile and its children are hundreds
    of metres, and their range differences cannot be held to a small fraction
    of a wavelength in float32 at orbital range.
  * The device program takes one first-level tile at a time, so that only the
    intermediate data of that tile exist on the device.
"""
import math
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .ffbp import POLICIES, decimator, _positions, _chunk, needs_x64, C

FACTORS = (8, 7, 6, 5, 4, 3, 2)


def choose_splits(n, T, nlev):
    """Smallest padded tile count >= ceil(n / T) that is a product of `nlev` factors from FACTORS (largest first)."""
    need = -(-n // T)
    best = None

    def rec(prefix, prod):
        nonlocal best
        if len(prefix) == nlev:
            if prod >= need and (best is None or prod < best[0] or (prod == best[0] and tuple(prefix) > best[1])):
                best = (prod, tuple(prefix))
            return
        for f in FACTORS:
            if prefix and f > prefix[-1]:
                continue
            rec(prefix + [f], prod * f)

    rec([], 1)
    if best is None:
        # too few levels for this image with factors up to 8: allow larger factors (the levels' filters and tile
        # geometry take any factor; only the plans of record avoid them)
        for f in range(16, 8, -1):
            if f ** nlev >= need:
                rec_big = [f] * nlev
                while len(rec_big) and _prod(rec_big[:-1]) * (rec_big[-1] - 1) >= need and rec_big[-1] > 2:
                    rec_big[-1] -= 1
                best = (_prod(rec_big), tuple(rec_big))
        if best is None:
            raise ValueError(f'{nlev} levels cannot split {n} pixels into tiles of {T}: use more levels or a larger T')
    return best[1]


def _prod(v):
    out = 1
    for x in v:
        out *= x
    return out


class LazyDecimator:
    """The decimation matrix of ffbp.decimator, built only when an array is asked for (np.asarray, the JAX paths):
    for a long aperture it is P x P/D dense (an identity of 67 GB at 91,426 pulses when D = 1), while the C++ and
    CUDA paths need only its kernel (decimator_fir)."""

    def __init__(self, n_in, D, pass_frac, atten):
        self.args = (int(n_in), int(D), float(pass_frac), float(atten))
        n_in, D = self.args[:2]
        if D == 1:
            self.m, n_out = 0, n_in
        else:
            dw = (2.0 - 2.0 * pass_frac) * np.pi / D
            half = math.ceil((atten - 8.0) / (2.285 * dw)) / 2.0 + 1.0
            self.m = int(math.ceil(half / D))
            n_out = -(-n_in // D) + 2 * self.m
        self.shape = (n_in, n_out)

    def __array__(self, dtype=None, copy=None):
        F = decimator(*self.args)[0]
        return F if dtype is None else F.astype(dtype)

    def astype(self, dtype):
        return np.asarray(self).astype(dtype)

    @property
    def T(self):
        return np.asarray(self).T


def decimator_fir(n_in, D, pass_frac, atten=70.0):
    """The kernel form of ffbp.decimator (as fir(F, D, m) returns it) without the dense matrix: F is shift-invariant,
    each column one Kaiser-windowed sinc divided by the row sums of its inputs."""
    dw = (2.0 - 2.0 * pass_frac) * np.pi / D
    half = math.ceil((atten - 8.0) / (2.285 * dw)) / 2.0 + 1.0
    m = int(math.ceil(half / D))
    n_out = -(-n_in // D) + 2 * m
    beta = 0.1102 * (atten - 8.7)

    def K(d):
        # the operations of ffbp.decimator in its order, so that the kernel equals its column bit for bit
        return np.where(np.abs(d) <= half, np.sinc(d / D) * (np.i0(beta * np.sqrt(np.clip(1.0 - (d / half) ** 2, 0.0, 1.0))) / np.i0(beta)), 0.0)
    j0 = n_out // 2
    cj = D * (j0 - m) + (D - 1) / 2.0
    i = np.arange(max(0, int(math.floor(cj - half))), min(n_in, int(math.ceil(cj + half)) + 1))
    v = K(i - cj)
    nzm = v != 0
    i, v = i[nzm], v[nzm]
    # row sums: the inputs' weights over every output (all within reach, since each input has full support)
    jj = np.arange(n_out)
    cen = D * (jj - m) + (D - 1) / 2.0
    near = (np.abs(cen[None, :] - i[:, None]) <= half)
    rs = np.where(near, K(i[:, None] - cen[None, :]), 0.0).sum(1)
    kern = v / rs
    lo, L = int(i[0]) - D * (j0 - m), int(i[-1] - i[0] + 1)
    pl = D * m - lo
    assert pl >= 0 and L == len(kern)
    pr = max(D * (n_out - 1) + L - pl - n_in, 0)
    return dict(kern=kern, pl=int(pl), pr=int(pr), n_out=int(n_out), L=int(L)), m



def banded_fir(fr, D, n_in, G=128):
    """The decimation of decimator_fir's kernel (fr) as a blocked banded product: inputs in blocks of D G samples,
    outputs in blocks of G; output block b is the sum over t of input block b + t times W[t] [D G, G]. Equal to the
    dense matrix (edges included), at a cost of a few blocks per output block instead of all n_in inputs."""
    kern, pl, n_out, L = fr['kern'], fr['pl'], fr['n_out'], fr['L']
    G = max(1, min(G, n_out))
    block = D * G
    w = -(-(block - D + L) // block)
    nbo = -(-n_out // G)
    nb = nbo + w - 1
    pr = nb * block - pl - n_in
    while pr < 0:
        nb, pr = nb + 1, pr + block
    W = np.zeros((w, block, G))
    g = np.arange(G)[:, None]
    t, q = np.divmod(D * g + np.arange(L)[None, :], block)
    W[t, q, np.broadcast_to(g, t.shape)] = np.broadcast_to(kern, t.shape)
    return dict(W=W, pl=int(pl), pr=int(pr), nb=int(nb), nbo=int(nbo), block=int(block), G=int(G), w=int(w), n_out=int(n_out))


# T Q Pl, products of tile side, frequency rows and lane-padded pulses, that one call of the TPU final-stage kernel
# keeps in VMEM: 2^21 (T 32, Q 128, Pl 512) fits the v6e's scoped VMEM, 2.5 x 2^21 does not
FINAL_VMEM_TQP = 1 << 21


def dense_pulse_filter(lv):
    """Whether a level's pulse decimation is applied as its dense matrix (short apertures; one matrix product)
    rather than banded_fir (the dense matrix grows with the square of the pulse count: 2.8 GB at 74,203 pulses)."""
    return lv['P'] * lv['Po'] <= (1 << 24)

def fir(F, D, m):
    """The decimation matrix F [n_in, n_out] as one kernel: out[j] = sum_r kern[r] * xpad[D j + r],
    with xpad the input padded by pl zeros on the left and pr on the right. Exact, edges included."""
    n_in, n_out = F.shape
    j0 = n_out // 2
    nz = np.nonzero(F[:, j0])[0]
    lo, L = int(nz[0]) - D * (j0 - m), int(nz[-1] - nz[0] + 1)
    kern = F[nz[0]:nz[0] + L, j0].copy()
    pl = D * m - lo
    assert pl >= 0
    pr = max(D * (n_out - 1) + L - pl - n_in, 0)
    for j in sorted({0, 1, m, j0, n_out - m - 1, n_out - 2, n_out - 1}):
        col = np.zeros(n_in)
        for r in range(L):
            i = D * j + r - pl
            if 0 <= i < n_in:
                col[i] = kern[r]
        assert np.abs(col - F[:, j]).max() < 1e-12, (j, np.abs(col - F[:, j]).max())
    return dict(kern=kern, pl=int(pl), pr=int(pr), n_out=int(n_out), L=int(L))


def make_plan(col, nx, ny, spx, spy, T=32, nlev=3, pmax=0.4, atten=70.0, splits=None, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0)):
    """Tile grid, decimation factors and filters for an nx x ny image with pixel spacings spx, spy in the
    plane spanned by the orthonormal vectors e1, e2 (the ground plane by default), centred on the origin of
    the collection's coordinates. Depends on the imaging mode only."""
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    sxs = choose_splits(nx, T, nlev) if splits is None else tuple(s[0] for s in splits)
    sys_ = choose_splits(ny, T, nlev) if splits is None else tuple(s[1] for s in splits)
    Nx, Ny = T * int(np.prod(sxs)), T * int(np.prod(sys_))
    ox, oy = (Nx - nx) // 2, (Ny - ny) // 2
    ant, f0, df, K, P = col.ant.astype(np.float64), float(col.fmin), float(col.df), col.K, col.Np
    ref = np.zeros((1, 3))
    i0 = np.zeros((1, 2))
    cx, cy = Nx, Ny
    out = dict(levels=[], T=T, nx=nx, ny=ny, Nx=Nx, Ny=Ny, ox=ox, oy=oy, spx=spx, spy=spy, split=list(zip(sxs, sys_)), e1=e1, e2=e2)
    for sx, sy in zip(sxs, sys_):
        cx, cy = cx // sx, cy // sy
        gi, gj = np.meshgrid(np.arange(sx) * cx, np.arange(sy) * cy, indexing='ij')
        ci0 = i0[:, None, :] + np.stack([gi.ravel(), gj.ravel()], axis=1)[None, :, :]          # [B, C, 2]
        cen = (((ci0[..., 0] + (cx - 1) / 2.0 - ox - nx / 2.0) * spx)[..., None] * e1 +
               ((ci0[..., 1] + (cy - 1) / 2.0 - oy - ny / 2.0) * spy)[..., None] * e2)          # [B, C, 3]
        # half extents of a child tile in slant range and in range change per pulse step, over its corners
        u = ant / np.linalg.norm(ant, axis=1)[:, None]
        u1, u2 = u @ e1, u @ e2
        hx, hy = 0.5 * cx * spx, 0.5 * cy * spy
        rk = hx * np.abs(u1).max() + hy * np.abs(u2).max()
        dop = hx * np.abs(np.diff(u1)).max() + hy * np.abs(np.diff(u2)).max()
        Dk = max(1, int(pmax * (C / (2.0 * df) / 2.0) / rk))
        Dp = max(1, int(pmax / (2.0 * (2.0 * (f0 + K * df) / C) * dop)))
        # passband edges rounded up to 1/64 of the output Nyquist rate (a little more band kept), so that patches of
        # a mosaic with nearly equal geometry get identical filters and share one compiled JAX program
        pass_k = math.ceil(64 * rk / (C / (2.0 * df * Dk) / 2.0)) / 64
        pass_p = math.ceil(64 * 2.0 * (2.0 * (f0 + K * df) / C) * dop * Dp) / 64
        Fk, Fp = LazyDecimator(K, Dk, min(pass_k, 0.95), atten), LazyDecimator(P, Dp, min(pass_p, 0.95), atten)
        mk, mp = Fk.m, Fp.m
        # an axis shorter than its decimation kernel (late levels of small or wide-angle collections) is not decimated
        if Dk > 1 and Fk.shape[0] < 2 * mk * Dk + Dk:
            Dk, pass_k = 1, rk / (C / (2.0 * df) / 2.0)
            Fk, mk = LazyDecimator(K, 1, 0.95, atten), 0
        if Dp > 1 and Fp.shape[0] < 2 * mp * Dp + Dp:
            Dp, pass_p = 1, 2.0 * (2.0 * (f0 + K * df) / C) * dop
            Fp, mp = LazyDecimator(P, 1, 0.95, atten), 0
        pidx = Dp * (np.arange(Fp.shape[1]) - mp) + (Dp - 1) / 2.0
        out['levels'].append(dict(sx=sx, sy=sy, C=sx * sy, Dk=Dk, Dp=Dp, Fk=Fk, Fp=Fp, K=K, P=P, Ko=Fk.shape[1], Po=Fp.shape[1],
                                  fir_k=decimator_fir(K, Dk, Fk.args[2], atten)[0] if Dk > 1 else None,
                                  fir_p=decimator_fir(P, Dp, Fp.args[2], atten)[0] if Dp > 1 else None,
                                  f0=f0, df=df, ref=ref, d=cen[0] - ref[0], cen=cen, pidx=pidx,
                                  pass_k=float(pass_k), pass_p=float(pass_p)))
        ant = _positions(ant, pidx)
        f0, df, K, P = f0 + ((Dk - 1) / 2.0 - mk * Dk) * df, df * Dk, Fk.shape[1], Fp.shape[1]
        ref, i0 = cen.reshape(-1, 3), ci0.reshape(-1, 2)
    assert (cx, cy) == (T, T)
    out['final'] = dict(cen=ref, f0=f0, df=df, K=K, P=P, fc=f0 + (K - 1) / 2.0 * df)
    return out


def plan_signature(plan):
    """Everything of a plan that the JAX program (make_ffbp) takes as a compile-time constant, as a hashable tuple:
    two plans with equal signatures can share one compiled program, with their own runtime arrays."""
    def r(x):
        return float(np.float64(x))
    lv = []
    for l in plan['levels']:
        fk = l['fir_k'] and (l['fir_k']['L'], l['fir_k']['pl'], l['fir_k']['pr'], l['fir_k']['n_out'])
        fp = l['fir_p'] and (l['fir_p']['L'], l['fir_p']['pl'], l['fir_p']['pr'], l['fir_p']['n_out'])
        lv.append((l['sx'], l['sy'], l['C'], l['Dk'], l['Dp'], l['K'], l['P'], l['Ko'], l['Po'], r(l['f0']), r(l['df']),
                   fk, fp, l['Fk'].args, l['Fp'].args, r(l['pass_k']), r(l['pass_p'])))
    f = plan['final']
    return (tuple(lv), (f['P'], f['K'], r(f['f0']), r(f['df']), r(f['fc'])), plan['T'], plan['nx'], plan['ny'],
            plan['Nx'], plan['Ny'], plan['ox'], plan['oy'], r(plan['spx']), r(plan['spy']),
            tuple(map(r, plan['e1'])), tuple(map(r, plan['e2'])))


def final_phase_error(ant, fmax, nx, ny, spx, spy, e1, e2, T):
    """Predicted image error (dB, relative to exact backprojection) from the final stage's plane-wave model on
    T x T tiles: the peak phase it leaves out (exact range offset of a pixel from its tile center minus the linear
    term and the aperture-mean curvature, over pulses, tile corners and a 3 x 3 sample of tiles across the image)
    mapped by 20 log10(phase) - 14 dB, which matches measured errors to about 2 dB from 1 km to orbital range."""
    ant, e1, e2 = np.asarray(ant, np.float64), np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    a = ant[::max(1, len(ant) // 2048)]
    hx, hy = 0.5 * T * spx, 0.5 * T * spy
    worst = 1e-30
    for fx in (-0.5, 0.0, 0.5):
        for fy in (-0.5, 0.0, 0.5):
            c = fx * max(0.0, nx * spx - 2 * hx) * e1 + fy * max(0.0, ny * spy - 2 * hy) * e2
            w = a - c
            r = np.linalg.norm(w, axis=1)
            u = w / r[:, None]
            for sx in np.linspace(-1, 1, 5):
                for sy in np.linspace(-1, 1, 5):
                    d = sx * hx * e1 + sy * hy * e2
                    ud = u @ d
                    res = np.linalg.norm(w - d, axis=1) - r + ud - (d @ d - ud * ud).mean() / (2 * r.mean())
                    worst = max(worst, float(np.abs(res - res.mean()).max()))
    return 20.0 * np.log10(4 * np.pi * fmax / C * worst) - 14.0


def choose_T(ant, fmax, nx, ny, spx, spy, e1, e2, target_db=-40.0, choices=(32, 16)):
    """Largest final tile size whose predicted final-stage error meets target_db (the smallest choice otherwise),
    with the predicted error."""
    for T in choices:
        err = final_phase_error(ant, fmax, nx, ny, spx, spy, e1, e2, T)
        if err <= target_db:
            return T, err
    return choices[-1], err


HOST_LEVELS = 2          # levels whose rotation phases are computed in float64 on the host


def collection_arrays(plan, ant, ref=None):
    """Per-collection host work in float64: the antenna path on each level's pulse grid, and for the first
    HOST_LEVELS levels the rotation phase of every child at band centre (wrapped to a cycle) and its slope
    in cycles per frequency sample. ref [P]: the range each pulse's samples are referenced to (one way), when it
    is not the antenna's distance to the origin (a bistatic collection's half path, a vendor's reference point);
    the first level's rotation then starts from it, at no extra cost."""
    ant = np.asarray(ant, np.float64)
    out = dict(levels=[])
    for i, lv in enumerate(plan['levels']):
        r0 = np.linalg.norm(ant, axis=1)
        e = dict(u=ant / r0[:, None], r0=r0)
        if i < HOST_LEVELS:
            kc = (lv['K'] - 1) / 2.0
            B, Cn = lv['ref'].shape[0], lv['C']
            c0 = np.empty((B, Cn, lv['P']), np.float32)
            sl = np.empty((B, Cn, lv['P']), np.float32)
            for b in range(B):
                base = np.linalg.norm(lv['ref'][b][None, :] - ant, axis=1)                       # [P]
                if i == 0 and ref is not None:
                    base = np.asarray(ref, np.float64)
                ddr = np.linalg.norm(lv['cen'][b][:, None, :] - ant[None, :, :], axis=2) - base[None, :]
                cyc = (2.0 * (lv['f0'] + kc * lv['df']) / C) * ddr
                c0[b] = cyc - np.round(cyc)
                sl[b] = (2.0 * lv['df'] / C) * ddr
            e['c0'], e['slope'] = c0, sl
        out['levels'].append(e)
        ant = _positions(ant, lv['pidx'])
    r0 = np.linalg.norm(ant, axis=1)
    out['final'] = dict(u=ant / r0[:, None], r0=r0)
    return out


def static_arrays(policy, plan, filt='dense'):
    """Device arrays that depend on the imaging mode only: the decimation filters and the tile geometry."""
    p = POLICIES[policy]
    mm = jnp.dtype(p['mm'])
    h = np.float64 if p['ew'] == 'float64' else np.float32

    def dev(x):
        return jnp.asarray(np.asarray(x).astype(h))

    lv = []
    for i, l in enumerate(plan['levels']):
        e = {}
        if filt in ('pallas', 'pallas2'):
            if l['Dp'] > 1 and dense_pulse_filter(l):
                e['Fp'] = dev(l['Fp']).astype(mm)
            elif l['Dp'] > 1:
                e['Wp'] = dev(banded_fir(l['fir_p'], l['Dp'], l['P'])['W']).astype(mm)
        elif filt == 'dense':
            if l['Dk'] > 1:
                e['Fk'] = dev(l['Fk']).astype(mm)
            if l['Dp'] > 1:
                e['Fp'] = dev(l['Fp']).astype(mm)
        else:
            if l['Dk'] > 1:
                e['kk'] = dev(l['fir_k']['kern']).astype(mm)
            if l['Dp'] > 1:
                e['kp'] = dev(l['fir_p']['kern']).astype(mm)
        if i >= HOST_LEVELS:
            e.update(ref=dev(l['ref']), d=dev(l['d']))
        lv.append(e)
    return dict(levels=lv, final=dict(cen=dev(plan['final']['cen'])))


def device_arrays(policy, plan, coll, static):
    """Add the per-collection arrays (from `collection_arrays`) to the static ones."""
    h = np.float64 if POLICIES[policy]['ew'] == 'float64' else np.float32

    def dev(x):
        return jnp.asarray(np.asarray(x).astype(h))

    lv = []
    for i, (e, c) in enumerate(zip(static['levels'], coll['levels'])):
        e = dict(e)
        if i < HOST_LEVELS:
            e.update(c0=dev(c['c0']), slope=dev(c['slope']))
        else:
            e.update(u=dev(c['u']), r0=dev(c['r0']))
        lv.append(e)
    return dict(levels=lv, final=dict(static['final'], u=dev(coll['final']['u']), r0=dev(coll['final']['r0'])))


def prepare(policy, S):
    """Phase history [P, K] -> real and imaginary planes in the element-wise type, scaled to unit peak, and the scale."""
    p = POLICIES[policy]
    h = np.float64 if p['ew'] == 'float64' else np.float32
    scale = float(np.abs(S).max()) or 1.0          # an all-zero history: zero planes, not 0/0
    return (jnp.asarray((S.real / scale).astype(h)).astype(p['ew']), jnp.asarray((S.imag / scale).astype(h)).astype(p['ew']), scale)


def _pulse_block(P, pb_max):
    """Pulse block for the fused level kernel: the largest of pb_max, 128, 64, 32 that pads the pulse count by
    less than 10 percent, else the one that pads least."""
    cands = [pb for pb in (pb_max, 256, 128, 64, 32) if pb <= pb_max]
    for pb in cands:
        if (-(-P // pb) * pb - P) / P < 0.1:
            return pb
    return min(cands, key=lambda pb: -(-P // pb) * pb)


def make_ffbp(policy, plan, filt='dense', budget=1 << 26, trig='split', pallas_pb=128, pallas_chunk=512, pallas_nc=8, pallas_ng=8, pallas_final=2, pallas_gen=3):
    """Build form(hre, him, arrays) -> (re, im), each [nx, ny] float32 (float64 for the fp64 policy).

    filt: 'dense' (matrix product with the decimation matrix), 'conv' (strided convolution with its kernel),
    or 'taps' (sum over kernel taps of strided slices).
    trig: 'direct' evaluates one sine and cosine per sample of a phase ramp; 'split' builds the ramp as the
    product of a coarse and a fine table, which needs about 2 sqrt(n) of them per ramp of n samples.
    """
    p = POLICIES[policy]
    ew, mm, prec = jnp.dtype(p['ew']), jnp.dtype(p['mm']), p['prec']
    f = jnp.float64 if p['ew'] == 'float64' else jnp.float32
    T, levels = plan['T'], plan['levels']
    L = len(levels)
    two_pi = 2.0 * math.pi
    npass = 1 if p['prec'] is None else 3            # the fused kernel: one bfloat16 pass or the three-pass split
    if filt in ('pallas', 'pallas2') and p['prec'] == lax.Precision.HIGHEST:
        raise ValueError('the fused kernels implement one- and three-pass products only; the six-pass (highest) setting was not built, '
                         'since three passes reproduce it to within 0.3 dB (use filt=dense for six passes)')
    bands = {}
    on_tpu = jax.devices()[0].platform == 'tpu' or bool(os.environ.get('FFBP_FORCE_TPU_KERNELS'))   # the latter for interpret-mode tests
    if filt == 'pallas':
        if on_tpu:
            from .pallas_ffbp import band_blocks
        else:
            from .pallas_ffbp_gpu import band_blocks_gpu as band_blocks
        for l in levels:
            if l['Dk'] > 1:
                bands[(l['P'], l['K'], l['Dk'])] = band_blocks(np.asarray(l['Fk']), l['Dk'])
    if filt == 'pallas2':
        if on_tpu:
            from .pallas_ffbp import band_blocks2
        else:
            from .pallas_ffbp_gpu import band_blocks_gpu as band_blocks2
        for l in levels:
            if l['Dk'] > 1:
                bands[(l['P'], l['K'], l['Dk'])] = band_blocks2(np.asarray(l['Fk']), l['Dk'])

    def mmul(a, b):
        return jnp.matmul(a.astype(mm), b, precision=prec, preferred_element_type=f)

    def fir_last(x, kern, fr, D):
        """x [..., n_in] filtered and decimated along the last axis."""
        n_out, Lk = fr['n_out'], fr['L']
        if filt == 'conv':
            lead = x.shape[:-1]
            xi = x.reshape((-1, 1, 1, x.shape[-1])).astype(mm)
            y = lax.conv_general_dilated(xi, kern.reshape(1, 1, 1, Lk), (1, D), ((0, 0), (fr['pl'], fr['pr'])),
                                         precision=prec, preferred_element_type=f)
            return y[:, 0, 0, :n_out].reshape(lead + (n_out,))
        xp = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(fr['pl'], fr['pr'])]).astype(f)
        kf = kern.astype(f)
        y = kf[0] * xp[..., 0:D * (n_out - 1) + 1:D]
        for r in range(1, Lk):
            y = y + kf[r] * xp[..., r:r + D * (n_out - 1) + 1:D]
        return y

    def dec_k(x, la, lv):
        """x [N, P, K] -> [N, P, Ko]."""
        if lv['Dk'] == 1:
            return x.astype(f)
        if filt == 'dense':
            return mmul(x, la['Fk'])
        return fir_last(x, la['kk'], lv['fir_k'], lv['Dk'])

    def dec_p(y, la, lv):
        """y [N, P, Ko] -> [N, Po, Ko]."""
        if lv['Dp'] == 1:
            return y.astype(f)
        if 'Wp' in la:                                       # long apertures: banded
            b = banded_fir(lv['fir_p'], lv['Dp'], lv['P'])
            N, Ko = y.shape[0], y.shape[2]
            xb = jnp.pad(y.astype(mm), ((0, 0), (b['pl'], b['pr']), (0, 0))).reshape(N, b['nb'], b['block'], Ko)
            z = None
            for t in range(b['w']):
                zt = jnp.einsum('nbqk,qg->nbgk', xb[:, t:t + b['nbo']], la['Wp'][t], precision=prec, preferred_element_type=f)
                z = zt if z is None else z + zt
            return z.reshape(N, b['nbo'] * b['G'], Ko)[:, :b['n_out']]
        if filt in ('dense', 'pallas', 'pallas2'):
            return jnp.swapaxes(mmul(jnp.swapaxes(y, 1, 2), la['Fp']), 1, 2)
        if filt == 'conv':
            fr = lv['fir_p']
            yi = y[:, None].astype(mm)                                                        # [N, 1, P, Ko]
            z = lax.conv_general_dilated(yi, la['kp'].reshape(1, 1, fr['L'], 1), (lv['Dp'], 1), ((fr['pl'], fr['pr']), (0, 0)),
                                         precision=prec, preferred_element_type=f)
            return z[:, 0, :fr['n_out'], :]
        return jnp.swapaxes(fir_last(jnp.swapaxes(y, 1, 2), la['kp'], lv['fir_p'], lv['Dp']), 1, 2)

    def ramp(c0, sl, n, centre):
        """cos and sin of 2 pi (c0 + (k - centre) sl) for k = 0 .. n - 1; c0 and sl of one shape, result [..., n]."""
        if trig == 'direct':
            kk = jnp.arange(n, dtype=jnp.int32).astype(f) - float(centre)
            cyc = c0[..., None] + kk * sl[..., None]
            ang = (cyc - jnp.round(cyc)) * two_pi
            return jnp.cos(ang).astype(ew), jnp.sin(ang).astype(ew)
        Bq = int(math.ceil(math.sqrt(n)))
        nh = -(-n // Bq)
        kh = jnp.arange(nh, dtype=jnp.int32).astype(f) * float(Bq) - float(centre)
        ch = c0[..., None] + kh * sl[..., None]
        ah = (ch - jnp.round(ch)) * two_pi
        cl = jnp.arange(Bq, dtype=jnp.int32).astype(f) * sl[..., None]
        al = (cl - jnp.round(cl)) * two_pi
        chh, shh, cll, sll = jnp.cos(ah), jnp.sin(ah), jnp.cos(al), jnp.sin(al)
        lead = c0.shape
        cs = (chh[..., :, None] * cll[..., None, :] - shh[..., :, None] * sll[..., None, :]).reshape(lead + (nh * Bq,))[..., :n]
        sn = (chh[..., :, None] * sll[..., None, :] + shh[..., :, None] * cll[..., None, :]).reshape(lead + (nh * Bq,))[..., :n]
        return cs.astype(ew), sn.astype(ew)

    def rotate(pre, pim, c0, sl, K):
        """Data [P', K] times exp(j 2 pi (c0 + (k - kc) slope)) for each of the children: -> [2 cb, P', K]."""
        cs, sn = ramp(c0, sl, K, (K - 1) / 2.0)
        return jnp.concatenate([pre[None] * cs - pim[None] * sn, pre[None] * sn + pim[None] * cs], axis=0)

    def device_phases(la, refb, lv):
        """Band-centre phase and slope of every child of the parent at refb, in the working type."""
        k0c = float(2.0 * (lv['f0'] + (lv['K'] - 1) / 2.0 * lv['df']) / C)
        k1 = float(2.0 * lv['df'] / C)
        u, r0, d = la['u'], la['r0'], la['d']
        uc = u[:, 0] * refb[0] + u[:, 1] * refb[1] + u[:, 2] * refb[2]
        wn = r0 * jnp.sqrt(1 + ((refb * refb).sum() - 2 * r0 * uc) / (r0 * r0))
        ud = d[:, 0:1] * u[None, :, 0] + d[:, 1:2] * u[None, :, 1] + d[:, 2:3] * u[None, :, 2]
        wd = r0[None, :] * ud - (d * refb[None, :]).sum(1)[:, None]
        num = (d * d).sum(1)[:, None] - 2 * wd
        ddr = num / (jnp.sqrt(wn[None, :] * wn[None, :] + num) + wn[None, :])
        c0 = ddr * k0c
        return c0 - jnp.round(c0), ddr * k1

    def children(pre, pim, c0, sl, la, lv):
        """One parent [P, K] -> all its children [C, Po, Ko] (real and imaginary)."""
        Cn, P, K, Po, Ko = lv['C'], lv['P'], lv['K'], lv['Po'], lv['Ko']
        if filt == 'pallas2' and lv['Dk'] > 1:
            if on_tpu:
                from .pallas_ffbp import fused_rotate_dec_k2, pad_columns
                pb = _pulse_block(P, pallas_pb)
            else:
                from .pallas_ffbp_gpu import fused_rotate_dec_k2_gpu, pad_columns_gpu as pad_columns
                gpu_prec = 'highest' if p['prec'] == 'highest' else None
                pb = pallas_pb

                def fused_rotate_dec_k2(a, b, c, d, band, kc, pb, passes):
                    return fused_rotate_dec_k2_gpu(a, b, c, d, band, kc, mm_dtype=mm, precision=gpu_prec, pb=pb)
            band = bands[(P, K, lv['Dk'])]
            nc = min(pallas_nc, Cn)
            if fuse_p_for(lv, band, nc):
                pb = 128 * -(-pb // 128)
            Pp = -(-P // pb) * pb
            Cp = -(-Cn // nc) * nc                                  # children padded to whole groups (zero ramps, discarded)
            if pre.shape == (Pp, band['Kpad']) and pre.dtype == jnp.float32:   # padded once by form.pad
                pre_p, pim_p = pre[None], pim[None]
            else:
                pre_p = pad_columns(jnp.pad(pre.astype(jnp.float32), ((0, Pp - P), (0, 0))), band)[None]
                pim_p = pad_columns(jnp.pad(pim.astype(jnp.float32), ((0, Pp - P), (0, 0))), band)[None]
            c0g = jnp.pad(c0.astype(jnp.float32), ((0, Cp - Cn), (0, Pp - P))).reshape(Cp // nc, 1, nc, Pp)
            slg = jnp.pad(sl.astype(jnp.float32), ((0, Cp - Cn), (0, Pp - P))).reshape(Cp // nc, 1, nc, Pp)

            def one_group(cs):
                if on_tpu:
                    a_, b_ = level_kernel(pre_p, pim_p, cs[0], cs[1], band, K, P, Ko, Po, pb, la, lv)
                    return a_[0], b_[0]
                yr, yi = fused_rotate_dec_k2(pre_p, pim_p, cs[0], cs[1], band, (K - 1) / 2.0, pb=pb, passes=npass)
                y = jnp.concatenate([yr[0, :, :P, :Ko], yi[0, :, :P, :Ko]], 0).astype(f)       # [2 nc, P, Ko]
                z = dec_p(y, la, lv)
                return z[:nc].astype(ew), z[nc:].astype(ew)

            if Cp == nc:
                ore, oim = one_group((c0g[0], slg[0]))
            else:
                ore, oim = lax.map(one_group, (c0g, slg))
                ore, oim = ore.reshape(Cp, Po, Ko), oim.reshape(Cp, Po, Ko)
            return ore[:Cn].reshape(Cn, Po, Ko), oim[:Cn].reshape(Cn, Po, Ko)
        if filt == 'pallas' and lv['Dk'] > 1:
            if on_tpu:
                from .pallas_ffbp import fused_rotate_dec_k, pad_columns
            else:
                from .pallas_ffbp_gpu import fused_rotate_dec_k_gpu, pad_columns_gpu as pad_columns
                gpu_prec = 'highest' if p['prec'] == 'highest' else None

                def fused_rotate_dec_k(a, b, c, d, band, kc, pb, chunk, passes):
                    return fused_rotate_dec_k_gpu(a, b, c, d, band, kc, mm_dtype=mm, precision=gpu_prec, pb=pb)
            band = bands[(P, K, lv['Dk'])]
            Pp = -(-P // pallas_pb) * pallas_pb
            pre_p = pad_columns(jnp.pad(pre.astype(jnp.float32), ((0, Pp - P), (0, 0))), band)
            pim_p = pad_columns(jnp.pad(pim.astype(jnp.float32), ((0, Pp - P), (0, 0))), band)

            def one_child(cs):
                c0c, slc = cs
                c0c = jnp.pad(c0c.astype(jnp.float32), (0, Pp - P))
                slc = jnp.pad(slc.astype(jnp.float32), (0, Pp - P))
                yr, yi = fused_rotate_dec_k(pre_p, pim_p, c0c, slc, band, (K - 1) / 2.0, pb=pallas_pb, chunk=pallas_chunk, passes=npass)
                y = jnp.stack([yr[:P, :Ko], yi[:P, :Ko]], 0).astype(f)
                z = dec_p(y, la, lv)
                return z[0].astype(ew), z[1].astype(ew)

            ore, oim = lax.map(one_child, (c0, sl))
            return ore.reshape(Cn, Po, Ko), oim.reshape(Cn, Po, Ko)
        cb = _chunk(Cn, P * K, budget)
        nb = -(-(cb * P * K) // budget)                    # blocks of pulses, when one child alone exceeds the budget

        def per_chunk(b):
            c0c, slc = b
            if nb > 1:
                edges = [round(i * P / nb) for i in range(nb + 1)]
                y = jnp.concatenate([dec_k(rotate(pre[a:b_], pim[a:b_], c0c[:, a:b_], slc[:, a:b_], K), la, lv)
                                     for a, b_ in zip(edges[:-1], edges[1:])], axis=1)
            else:
                y = dec_k(rotate(pre, pim, c0c, slc, K), la, lv)
            z = dec_p(y, la, lv)
            return z[:cb].astype(ew), z[cb:].astype(ew)

        c0 = c0.reshape(Cn // cb, cb, P)
        sl = sl.reshape(Cn // cb, cb, P)
        if Cn == cb:
            ore, oim = per_chunk((c0[0], sl[0]))
        else:
            ore, oim = lax.map(per_chunk, (c0, sl))
        return ore.reshape(Cn, Po, Ko), oim.reshape(Cn, Po, Ko)

    def fuse_p_for(lv, band, nc):
        """Whether the level kernel fuses the pulse decimation for nc children (the decimated block must fit in vector
        memory); the fused form needs 128-pulse blocks because the filter operand's lane dimension is the pulse block."""
        Po_pad = 8 * -(-lv['Po'] // 8)
        return (on_tpu and pallas_gen >= 3 and lv['Dp'] > 1 and Po_pad * band['kob'] * nc * 16 <= 40 << 20
                and dense_pulse_filter(lv))

    def level_kernel(pre_p, pim_p, c0g, slg, band, K, P, Ko, Po, pb, la, lv):
        """pre_p, pim_p [Np, Pp, Kpad]; c0g, slg [Np, nc, Pp] -> (re, im) [Np, 2 nc .. ] pulse-decimated children
        [Np, nc, Po, Ko] each, through the level kernel of the selected generation (fusing the pulse decimation when
        the decimated block fits in vector memory) and otherwise the dense pulse product."""
        from .pallas_ffbp import fused_rotate_dec_k2, fused_rotate_dec_k3
        Np, nc, Pp = c0g.shape
        fuse_p = fuse_p_for(lv, band, nc) and pb % 128 == 0
        if on_tpu and pallas_gen >= 3:
            FpT = None
            if fuse_p:
                FpT = jnp.pad(jnp.asarray(np.asarray(lv['Fp']).T, jnp.float32), ((0, 0), (0, Pp - P)))
            yr, yi = fused_rotate_dec_k3(pre_p, pim_p, c0g, slg, band, (K - 1) / 2.0, pb=pb, passes=npass, FpT=FpT)
        else:
            yr, yi = fused_rotate_dec_k2(pre_p, pim_p, c0g, slg, band, (K - 1) / 2.0, pb=pb, passes=npass)
        if fuse_p:
            return yr[:, :, :Po, :Ko].astype(ew), yi[:, :, :Po, :Ko].astype(ew)
        y = jnp.concatenate([yr[:, :, :P, :Ko], yi[:, :, :P, :Ko]], 1).reshape(Np * 2 * nc, P, Ko).astype(f)
        z = dec_p(y, la, lv).reshape(Np, 2 * nc, Po, Ko)
        return z[:, :nc].astype(ew), z[:, nc:].astype(ew)

    def children_batch(pre, pim, c0, sl, la, lv):
        """Several parents at once through the fused level kernel: pre, pim [Np, P, K]; c0, sl [Np, C, P] ->
        [Np C, Po, Ko] (real, imaginary). All C children of a parent come from one load when C <= 64."""
        Np, P, K = pre.shape
        Cn, Po, Ko = lv['C'], lv['Po'], lv['Ko']
        if on_tpu:
            from .pallas_ffbp import fused_rotate_dec_k2, pad_columns
            pb = _pulse_block(P, pallas_pb)
        else:
            from .pallas_ffbp_gpu import fused_rotate_dec_k2_gpu, pad_columns_gpu as pad_columns
            gpu_prec = 'highest' if p['prec'] == 'highest' else None
            pb = pallas_pb

            def fused_rotate_dec_k2(a, b, c, d, band, kc, pb, passes):
                return fused_rotate_dec_k2_gpu(a, b, c, d, band, kc, mm_dtype=mm, precision=gpu_prec, pb=pb)
        band = bands[(P, K, lv['Dk'])]
        nc = Cn if Cn <= 64 else pallas_nc
        if fuse_p_for(lv, band, nc):
            pb = 128 * -(-pb // 128)
        Pp = -(-P // pb) * pb
        Cp = -(-Cn // nc) * nc
        pre_p = pad_columns(jnp.pad(pre.astype(jnp.float32), ((0, 0), (0, Pp - P), (0, 0))), band)
        pim_p = pad_columns(jnp.pad(pim.astype(jnp.float32), ((0, 0), (0, Pp - P), (0, 0))), band)
        c0g = jnp.pad(c0.astype(jnp.float32), ((0, 0), (0, Cp - Cn), (0, Pp - P))).reshape(Np, Cp // nc, nc, Pp).transpose(1, 0, 2, 3)
        slg = jnp.pad(sl.astype(jnp.float32), ((0, 0), (0, Cp - Cn), (0, Pp - P))).reshape(Np, Cp // nc, nc, Pp).transpose(1, 0, 2, 3)

        def one_group(cs):
            if on_tpu:
                return level_kernel(pre_p, pim_p, cs[0], cs[1], band, K, P, Ko, Po, pb, la, lv)
            yr, yi = fused_rotate_dec_k2(pre_p, pim_p, cs[0], cs[1], band, (K - 1) / 2.0, pb=pb, passes=npass)
            y = jnp.concatenate([yr[:, :, :P, :Ko], yi[:, :, :P, :Ko]], 1).reshape(Np * 2 * nc, P, Ko).astype(f)
            z = dec_p(y, la, lv).reshape(Np, 2 * nc, Po, Ko)
            return z[:, :nc].astype(ew), z[:, nc:].astype(ew)

        if Cp == nc:
            ore, oim = one_group((c0g[0], slg[0]))                                   # [Np, nc, Po, Ko]
        else:
            ore, oim = lax.map(one_group, (c0g, slg))                                # [G, Np, nc, Po, Ko]
            ore, oim = ore.transpose(1, 0, 2, 3, 4).reshape(Np, Cp, Po, Ko), oim.transpose(1, 0, 2, 3, 4).reshape(Np, Cp, Po, Ko)
        return ore[:, :Cn].reshape(Np * Cn, Po, Ko), oim[:, :Cn].reshape(Np * Cn, Po, Ko)

    fin = plan['final']
    Pf, Qf = fin['P'], fin['K']
    a0, a1, fc2 = float(2.0 * fin['f0'] / C), float(2.0 * fin['df'] / C), float(2.0 * fin['fc'] / C)
    dlx_host = (np.arange(T) - (T - 1) / 2.0) * plan['spx']
    dly_host = (np.arange(T) - (T - 1) / 2.0) * plan['spy']
    e1v = tuple(float(v) for v in plan.get('e1', (1.0, 0.0, 0.0)))
    e2v = tuple(float(v) for v in plan.get('e2', (0.0, 1.0, 0.0)))
    env = tuple(float(v) for v in np.cross(e1v, e2v))

    fused_fin = (filt == 'pallas2' and int(pallas_final)) or 0            # 0: XLA final; 1: direct-trig kernel; 2: recurrence kernel; 3: recurrence, four tiles per step

    def final(hre, him, fa):
        B = hre.shape[0]
        tb = B if fused_fin else _chunk(B, T * Pf * Qf, budget)
        dlx, dly = jnp.asarray(dlx_host, f), jnp.asarray(dly_host, f)

        def rot(g):
            """cos and sin of -2 pi (a0 + a1 q) g over the frequency samples q: g [tb, T, P] -> [tb, T, P, Q]."""
            c = g * (-a0)
            return ramp(c - jnp.round(c), g * (-a1), Qf, 0.0)

        def per_chunk(a):
            tre, tim, cen = a
            u, r0 = fa['u'], fa['r0']
            wx = r0[None, :] * u[None, :, 0] - cen[:, 0:1]
            wy = r0[None, :] * u[None, :, 1] - cen[:, 1:2]
            wz = r0[None, :] * u[None, :, 2] - cen[:, 2:3]
            wn = jnp.sqrt(wx * wx + wy * wy + wz * wz)
            # components of the unit vector to the antenna along the image plane's axes and its normal
            ux = (wx * e1v[0] + wy * e1v[1] + wz * e1v[2]) / wn
            uy = (wx * e2v[0] + wy * e2v[1] + wz * e2v[2]) / wn
            mx, my, mz = ux.mean(1), uy.mean(1), ((wx * env[0] + wy * env[1] + wz * env[2]) / wn).mean(1)
            mn = jnp.sqrt(mx * mx + my * my + mz * mz)
            ucx, ucy, rc = mx / mn, my / mn, wn.mean(1)
            if fused_fin in (2, 3) and on_tpu:
                from .pallas_ffbp import fused_final2, fused_final3
                Pl = 128 * -(-Pf // 128)
                Qp8 = 8 * -(-Qf // 8)
                tiles = 4 if fused_fin == 3 else 1
                Bp = tiles * -(-tb // tiles)
                padt = ((0, Bp - tb), (0, Qp8 - Qf), (0, Pl - Pf))
                padg = ((0, Bp - tb), (0, 0), (0, Pl - Pf))
                dTr = jnp.pad(jnp.swapaxes(tre, 1, 2).astype(jnp.float32), padt)
                dTi = jnp.pad(jnp.swapaxes(tim, 1, 2).astype(jnp.float32), padt)
                gxT = jnp.pad(ux[:, None, :] * dlx[None, :, None], padg).astype(jnp.float32)
                gyT = jnp.pad(uy[:, None, :] * dly[None, :, None], padg).astype(jnp.float32)
                # the kernel holds T x Q x Pl ramp products in VMEM; past FINAL_VMEM_TQP the frequency rows go in
                # chunks, each starting its ramp at its first row, and the chunks' products are summed
                Qc = max(8, (FINAL_VMEM_TQP // (T * Pl * tiles)) // 8 * 8)
                cre = cim = 0.0
                for q0 in range(0, Qp8, Qc):
                    q1 = min(Qp8, q0 + Qc)
                    if fused_fin == 3:
                        r_, i_ = fused_final3(dTr[:, q0:q1], dTi[:, q0:q1], gxT, gyT, a0, a1, min(Qf, q1) - q0, passes=npass, tiles=tiles, q0=q0)
                    else:
                        r_, i_ = fused_final2(dTr[:, q0:q1], dTi[:, q0:q1], gxT, gyT, a0, a1, min(Qf, q1) - q0, passes=npass, q0=q0)
                    cre, cim = cre + r_, cim + i_
                cre, cim = cre[:tb].astype(f), cim[:tb].astype(f)
            elif fused_fin:
                if on_tpu:
                    from .pallas_ffbp import fused_final
                else:
                    from .pallas_ffbp_gpu import fused_final_gpu
                    fused_final = lambda *a, passes: fused_final_gpu(*a, mm_dtype=mm, precision='highest' if p['prec'] == 'highest' else None)
                Qpad = 128 * -(-Qf // 128)
                Pfp = 8 * -(-Pf // 8)
                padq = ((0, 0), (0, Pfp - Pf), (0, Qpad - Qf))
                gxT = jnp.pad(ux[:, None, :] * dlx[None, :, None], ((0, 0), (0, 0), (0, Pfp - Pf))).astype(jnp.float32)
                gyT = jnp.pad(uy[:, None, :] * dly[None, :, None], ((0, 0), (0, 0), (0, Pfp - Pf))).astype(jnp.float32)
                qv = jnp.pad(a0 + a1 * jnp.arange(Qf, dtype=jnp.float32), (0, Qpad - Qf)).reshape(1, Qpad)
                cre, cim = fused_final(jnp.pad(tre.astype(jnp.float32), padq), jnp.pad(tim.astype(jnp.float32), padq), gxT, gyT, qv, passes=npass)
                cre, cim = cre.astype(f), cim.astype(f)
            else:
                cs, sn = rot(ux[:, None, :] * dlx[None, :, None])
                A = jnp.concatenate([tre[:, None] * cs - tim[:, None] * sn, tre[:, None] * sn + tim[:, None] * cs], axis=1)
                cs, sn = rot(uy[:, None, :] * dly[None, :, None])
                Bm = jnp.concatenate([cs, sn], axis=1)
                M = mmul(A.reshape(tb, 2 * T, Pf * Qf), jnp.swapaxes(Bm.reshape(tb, 2 * T, Pf * Qf), 1, 2).astype(mm))
                cre = M[:, :T, :T] - M[:, T:, T:]
                cim = M[:, :T, T:] + M[:, T:, :T]
            dx, dy = dlx[None, :, None], dly[None, None, :]
            los = ucx[:, None, None] * dx + ucy[:, None, None] * dy
            qc = (dx * dx + dy * dy - los * los) * (fc2 / (2.0 * rc[:, None, None]))
            ang = (qc - jnp.round(qc)) * two_pi
            c, s = jnp.cos(ang), jnp.sin(ang)
            return cre * c - cim * s, cre * s + cim * c

        xs = tuple(v.reshape((B // tb, tb) + v.shape[1:]) for v in (hre, him, fa['cen']))
        ire, iim = lax.map(per_chunk, xs)
        return ire.reshape(B, T, T), iim.reshape(B, T, T)

    sx0, sy0 = levels[0]['sx'], levels[0]['sy']
    G = sx0 * sy0
    mx, my = plan['Nx'] // sx0, plan['Ny'] // sy0
    shape_g = [s for lv in levels[1:] for s in (lv['sx'], lv['sy'])] + [T, T]
    perm_g = [2 * i for i in range(L - 1)] + [2 * (L - 1)] + [2 * i + 1 for i in range(L - 1)] + [2 * (L - 1) + 1]

    def final_w(a, b, fa, g):
        """The final stage of first-level tile g, with ImageFormer's aperture weight when fa holds one (w [3, ntiles,
        Pf]: the weight per final subaperture and tile and its gradient along e1 and e2): three final stages, combined
        with the pixel offsets."""
        cen = fa['cen'].reshape(G, -1, 3)[g]
        if 'w' not in fa:
            return final(a, b, dict(fa, cen=cen))
        w = fa['w'].reshape(3, G, -1, Pf)[:, g][..., None].astype(a.dtype)
        re, im = final(a * w[0], b * w[0], dict(fa, cen=cen))
        dx, dy = jnp.asarray(dlx_host, re.dtype)[None, :, None], jnp.asarray(dly_host, re.dtype)[None, None, :]
        for k, dl in ((1, dx), (2, dy)):
            r_, i_ = final(a * w[k], b * w[k], dict(fa, cen=cen))
            re, im = re + dl * r_, im + dl * i_
        return re, im

    @jax.jit
    def one_tile(hre, him, arrs, g):
        """One first-level tile carried through every level: -> its mx x my block of the image."""
        la, lv = arrs['levels'][0], levels[0]
        a, b = children(hre, him, la['c0'][0, g][None], la['slope'][0, g][None], la, dict(lv, C=1))
        for i in range(1, L):
            la, lv = arrs['levels'][i], levels[i]
            if i < HOST_LEVELS:
                a, b = children(a[0], b[0], la['c0'][g], la['slope'][g], la, lv)
            else:
                nb_par = a.shape[0]
                refs = la['ref'].reshape(G, nb_par, 3)[g]
                if filt == 'pallas2' and lv['Dk'] > 1:
                    c0, sl = jax.vmap(lambda r, la=la, lv=lv: device_phases(la, r, lv))(refs)
                    a, b = children_batch(a, b, c0, sl, la, lv)
                else:
                    def per_parent(x, la=la, lv=lv):
                        c0, sl = device_phases(la, x[2], lv)
                        return children(x[0], x[1], c0, sl, la, lv)

                    a, b = lax.map(per_parent, (a, b, refs))
                    a, b = a.reshape((-1,) + a.shape[2:]), b.reshape((-1,) + b.shape[2:])
        fa = arrs['final']
        re, im = final_w(a, b, fa, g)
        return re.reshape(shape_g).transpose(perm_g).reshape(mx, my), im.reshape(shape_g).transpose(perm_g).reshape(mx, my)

    ox, oy, nx, ny = plan['ox'], plan['oy'], plan['nx'], plan['ny']

    @jax.jit
    @jax.jit
    def assemble(re, im):
        """The image from the tiles' outputs in tile order (x major), given as arrays of one tile [mx, my] or of a
        group [ng, mx, my]: one program, not one eager slice per tile, which costs milliseconds of host time each."""
        def one(blocks):
            a = jnp.concatenate([b.reshape((-1,) + b.shape[-2:]) for b in blocks], 0)        # [G, mx, my]
            mx, my = a.shape[1:]
            full = a.reshape(sx0, sy0, mx, my).transpose(0, 2, 1, 3).reshape(sx0 * mx, sy0 * my)
            return full[ox:ox + nx, oy:oy + ny]
        return one(re), one(im)

    ng = min(pallas_ng, G) if filt == 'pallas2' else 1
    while G % ng:
        ng -= 1

    @jax.jit
    def one_group(hre, him, arrs, gs):
        """ng first-level tiles at once: the phase history is read once per group of their first-level children."""
        la, lv = arrs['levels'][0], levels[0]
        a0, b0 = children(hre, him, la['c0'][0, gs], la['slope'][0, gs], la, dict(lv, C=ng))

        def rest(x):
            a, b, g = x
            a, b = a[None], b[None]
            for i in range(1, L):
                la, lv = arrs['levels'][i], levels[i]
                if i < HOST_LEVELS:
                    a, b = children(a[0], b[0], la['c0'][g], la['slope'][g], la, lv)
                else:
                    nb_par = a.shape[0]
                    refs = la['ref'].reshape(G, nb_par, 3)[g]
                    if filt == 'pallas2' and lv['Dk'] > 1:
                        c0, sl = jax.vmap(lambda r, la=la, lv=lv: device_phases(la, r, lv))(refs)
                        a, b = children_batch(a, b, c0, sl, la, lv)
                    else:
                        def per_parent(x, la=la, lv=lv):
                            c0, sl = device_phases(la, x[2], lv)
                            return children(x[0], x[1], c0, sl, la, lv)

                        a, b = lax.map(per_parent, (a, b, refs))
                        a, b = a.reshape((-1,) + a.shape[2:]), b.reshape((-1,) + b.shape[2:])
            fa = arrs['final']
            re, im = final_w(a, b, fa, g)
            return re.reshape(shape_g).transpose(perm_g).reshape(mx, my), im.reshape(shape_g).transpose(perm_g).reshape(mx, my)

        return lax.map(rest, (a0, b0, gs))

    def form(hre, him, arrs):
        if ng > 1:
            parts = [one_group(hre, him, arrs, np.arange(g0, g0 + ng, dtype=np.int32)) for g0 in range(0, G, ng)]
        else:
            parts = [one_tile(hre, him, arrs, np.int32(g)) for g in range(G)]
        del hre, him
        return assemble(tuple(q[0] for q in parts), tuple(q[1] for q in parts))

    lv0 = levels[0]
    if filt == 'pallas2' and lv0['Dk'] > 1 and on_tpu:
        band0 = bands[(lv0['P'], lv0['K'], lv0['Dk'])]
        pb0 = _pulse_block(lv0['P'], pallas_pb)
        if fuse_p_for(lv0, band0, min(pallas_nc, ng if ng > 1 else 1)):
            pb0 = 128 * -(-pb0 // 128)
        Pp0 = -(-lv0['P'] // pb0) * pb0

        @jax.jit
        def pad1(x):
            from .pallas_ffbp import pad_columns
            return pad_columns(jnp.pad(x.astype(jnp.float32), ((0, Pp0 - lv0['P']), (0, 0))), band0)

        def pad(hre, him):
            """The phase history planes padded once for the first-level kernel, which then reads them in place
            instead of padding a copy for every group of children (two planes of temporaries on the device). The
            caller drops its unpadded planes."""
            return pad1(hre), pad1(him)
    else:
        def pad(hre, him):
            return hre, him
    form.pad = pad
    # the stages of one_tile, exposed for per-stage profiling (profile_level0.py)
    form.stages = dict(children=children, final=final, device_phases=device_phases, levels=levels, G=G, host_levels=HOST_LEVELS, one_group=one_group, one_tile=one_tile, ng=ng, bands=bands)
    return form
