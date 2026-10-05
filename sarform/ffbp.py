"""Factorized backprojection built from matrix products, with no lookups.

Backprojection spends its time fetching a range-profile sample at a computed
position for every pixel and pulse. Accelerators whose strength is the matrix
multiplier handle that badly, so this module reaches the same image a
different way:

  1. Split the image into s x s child tiles. For each child, rotate the phase
     history so it is motion-compensated to the child's centre (exact ranges).
  2. The child's signal is now band-limited in both frequency and pulse index,
     so low-pass filter and decimate both axes. The filter is a fixed dense
     matrix: a matrix product.
  3. Recurse. When tiles are a few tens of pixels the wavefront is flat across
     a tile, the kernel separates in x and y, and the tile image is a product
     of two small matrices. A per-pixel quadratic phase restores the curvature.

The host does a little float64 work per collection (the antenna track on each
level's pulse grid and the first level's phases). Per-tile geometry at the
deeper levels and everything proportional to the data runs on the device.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .lowfp import quantize

C = 299792458.0

POLICIES = {
    # element-wise type, matmul operand type, matmul precision request
    'fp64':      dict(ew='float64', mm='float64', prec='highest'),
    'fp32':      dict(ew='float32', mm='float32', prec='highest'),
    'fp32_fast': dict(ew='float32', mm='float32', prec=None),     # device default; bf16 passes on TPU
    'fp32_high': dict(ew='float32', mm='float32', prec='high'),   # the intermediate setting; three bf16 passes on TPU
    'bf16_mm':   dict(ew='float32', mm='bfloat16', prec=None),
    'bf16':      dict(ew='bfloat16', mm='bfloat16', prec=None),
    # first level in true float32, where the whole scene is still mixed in every sample; bf16 below it
    'bf16_mm_l1f32': dict(ew='float32', mm='bfloat16', prec=None, first='float32'),
    'f16_mm_l1f32':  dict(ew='float32', mm='float16', prec=None, first='float32'),
    'f16_mm':    dict(ew='float32', mm='float16', prec=None),
    'f16':       dict(ew='float16', mm='float16', prec=None),
    # emulated tensor-core formats: operands rounded, products accumulated in float32
    'f8_mm':     dict(ew='float32', mm='float32', prec='highest', emul='f8'),
    'f8e5_mm':   dict(ew='float32', mm='float32', prec='highest', emul='f8e5'),
    'f4_mm':     dict(ew='float32', mm='float32', prec='highest', emul='f4'),
}
EMULATED = ('f8_mm', 'f8e5_mm', 'f4_mm')


def needs_x64(policy):
    return POLICIES[policy]['ew'] == 'float64'


def decimator(n_in, D, pass_frac, atten=70.0):
    """Kaiser-windowed sinc decimation matrix [n_in, n_out] and the margin m.

    Output j sits at input index D (j - m) + (D - 1) / 2. The output grid runs
    m samples past each end of the input so every input sample has its full
    interpolation support; without that the aperture and band edges are
    reconstructed from a one-sided filter. `pass_frac` is the passband edge as
    a fraction of the output Nyquist rate; the stopband starts at 2 - pass_frac
    so aliases land outside the passband. Rows sum to one.
    """
    if D == 1:
        return np.eye(n_in), 0
    dw = (2.0 - 2.0 * pass_frac) * np.pi / D
    half = math.ceil((atten - 8.0) / (2.285 * dw)) / 2.0 + 1.0
    m = int(math.ceil(half / D))
    n_out = -(-n_in // D) + 2 * m
    centres = D * (np.arange(n_out) - m) + (D - 1) / 2.0
    d = np.arange(n_in)[:, None] - centres[None, :]
    beta = 0.1102 * (atten - 8.7)
    w = np.where(np.abs(d) <= half, np.i0(beta * np.sqrt(np.clip(1.0 - (d / half) ** 2, 0.0, 1.0))) / np.i0(beta), 0.0)
    F = np.sinc(d / D) * w
    return F / F.sum(1, keepdims=True), m


def banded(F, D, m, block):
    """The decimation matrix F [n_in, n_out] as a blocked banded product.

    Each output sample depends on a few tens of inputs, so F is almost all
    zeros and a dense product wastes most of its work. Because every input has
    its full support, F is shift-invariant: column j is one kernel placed at
    input D (j - m) + lo. Inputs are grouped in blocks of `block` samples and
    outputs in blocks of G = block / D; output block b is the sum over t of
    input block b + t times W[t] ([block, G]). Returns W [w, block, G] and the
    padding and block counts needed to apply it. Exactly equal to F.
    """
    n_in, n_out = F.shape
    assert block % D == 0
    G = block // D
    j0 = n_out // 2
    nz = np.nonzero(F[:, j0])[0]
    lo, L = int(nz[0]) - D * (j0 - m), int(nz[-1] - nz[0] + 1)
    kern = F[nz[0]:nz[0] + L, j0]
    pl = D * m - lo                                   # left padding so output j starts at padded index D j
    assert pl >= 0
    w = -(-(block - D + L) // block)
    nbo = -(-n_out // G)
    nb = nbo + w - 1
    pr = nb * block - pl - n_in
    assert pr >= 0
    W = np.zeros((w, block, G))
    for g in range(G):
        for r in range(L):
            t, q = divmod(D * g + r, block)
            W[t, q, g] = kern[r]
    # the banded form must reproduce the dense matrix, edges included
    for j in sorted({0, 1, m, j0, n_out - m - 1, n_out - 2, n_out - 1}):
        col = np.zeros(n_in)
        for r in range(L):
            i = D * j + r - pl
            if 0 <= i < n_in:
                col[i] = kern[r]
        assert np.abs(col - F[:, j]).max() < 1e-12, (j, np.abs(col - F[:, j]).max())
    return dict(W=W, pl=pl, pr=pr, nb=nb, nbo=nbo, block=block, G=G, w=w, n_out=n_out)


def _positions(ant, idx):
    """Antenna position at fractional pulse indices, linear inside and beyond the ends."""
    i0 = np.clip(np.floor(idx).astype(int), 0, len(ant) - 2)
    return ant[i0] + (idx - i0)[:, None] * (ant[i0 + 1] - ant[i0])


def make_plan(col, n, spacing, levels, T, atten=70.0, block=0):
    """Everything that depends only on the imaging mode: tile grid, decimation
    filters and array shapes for an n x n ground-plane image. Reusable across
    collections with the same sampling; `collection_arrays` adds the flight path.

    levels: [(s, D), ...] children per axis and decimation factor at each level;
    n must equal prod(s) * T.
    """
    assert n == int(np.prod([s for s, _ in levels])) * T
    ant, f0, df, K, P = col.ant.astype(np.float64), col.fmin, col.df, col.K, col.Np
    ref = np.zeros((1, 3))                 # current reference point of each tile's data
    i0 = np.zeros((1, 2))                  # first pixel index of each tile
    size = n
    out = dict(levels=[], T=T, n=n, spacing=spacing, split=[s for s, _ in levels], block=block)
    for s, D in levels:
        child = size // s
        gi, gj = np.meshgrid(np.arange(s) * child, np.arange(s) * child, indexing='ij')
        ci0 = i0[:, None, :] + np.stack([gi.ravel(), gj.ravel()], axis=1)[None, :, :]          # [B, C, 2]
        cen = np.concatenate([(ci0 + (child - 1) / 2.0 - n / 2.0) * spacing,
                              np.zeros(ci0.shape[:2] + (1,))], axis=2)                        # [B, C, 3]
        # passband edges from the tile's half diagonal
        hd = child * spacing / np.sqrt(2.0)
        pass_k = hd / (C / (2.0 * df * D) / 2.0)
        u = ant / np.linalg.norm(ant, axis=1)[:, None]
        du = np.linalg.norm(np.diff(u, axis=0), axis=1).max()
        pass_p = 2.0 * (2.0 * (f0 + K * df) / C) * du * hd * D
        Fk, mk = decimator(K, D, min(pass_k, 0.95), atten)
        Fp, mp = decimator(P, D, min(pass_p, 0.95), atten)
        pidx = D * (np.arange(Fp.shape[1]) - mp) + (D - 1) / 2.0
        band = None
        if block > 0 and D > 1:
            bl = max(block, D)
            band = dict(k=banded(Fk, D, mk, bl), p=banded(Fp, D, mp, bl))
        out['levels'].append(dict(band=band, D=D, Fk=Fk, Fp=Fp, K=K, P=P, Ko=Fk.shape[1], Po=Fp.shape[1], C=s * s, f0=float(f0), df=float(df),
                                  ref=ref, d=cen[0] - ref[0], cen0=cen[0], pidx=pidx,
                                  pass_k=float(pass_k), pass_p=float(pass_p)))
        ant = _positions(ant, pidx)
        f0, df, K, P = f0 + ((D - 1) / 2.0 - mk * D) * df, df * D, Fk.shape[1], Fp.shape[1]
        ref, i0, size = cen.reshape(-1, 3), ci0.reshape(-1, 2), child
    assert size == T
    out['final'] = dict(cen=ref, f0=float(f0), df=float(df), K=K, P=P, fc=float(f0 + (K - 1) / 2.0 * df))
    return out


def collection_arrays(plan, ant):
    """Per-collection host work, float64: the antenna track resampled to each
    level's pulse grid (a few thousand points) and the first level's rotation
    phases. The first level is done here because its range differences span
    the whole scene, where float32 would cost phase accuracy; deeper levels
    have short offsets and are computed on the device.
    """
    ant = np.asarray(ant, np.float64)
    out = dict(levels=[])
    for i, lv in enumerate(plan['levels']):
        r0 = np.linalg.norm(ant, axis=1)
        e = dict(u=ant / r0[:, None], r0=r0)
        if i == 0:
            ddr = np.linalg.norm(lv['cen0'][:, None, :] - ant[None, :, :], axis=2) - r0[None, :]   # [C, P]
            cyc0 = (2.0 * lv['f0'] / C) * ddr
            e['cyc0'] = (cyc0 - np.round(cyc0))[None]
            e['slope'] = ((2.0 * lv['df'] / C) * ddr)[None]
        out['levels'].append(e)
        ant = _positions(ant, lv['pidx'])
    r0 = np.linalg.norm(ant, axis=1)
    out['final'] = dict(u=ant / r0[:, None], r0=r0)
    return out


def _chunk(total, per_item, budget):
    """Largest power-of-two divisor of `total` whose block stays under budget."""
    cb = 1
    while total % (cb * 2) == 0 and cb * 2 * per_item <= budget:
        cb *= 2
    return cb


def make_ffbp(policy, plan, budget=1 << 26, jit=True, left=False, groups=0):
    """Build ffbp(H_re, H_im, arrays) -> (img_re, img_im), n x n, float32/64.

    H_*: [1, Np, K] in the element-wise type; `arrays` from `device_arrays`.
    """
    p = POLICIES[policy]
    ew, mm = jnp.dtype(p['ew']), jnp.dtype(p['mm'])
    prec = p['prec']
    f = jnp.float64 if p['ew'] == 'float64' else jnp.float32
    T, n = plan['T'], plan['n']
    two_pi = 2.0 * math.pi

    emul = p.get('emul')

    first_mm = p.get('first')

    def mmul(a, b, full=False):
        if full:        # true float32 product regardless of the policy's operand type
            return jnp.matmul(a.astype(f), b.astype(f), precision='highest', preferred_element_type=f)
        if emul:
            a, b = quantize(a.astype(f), emul), quantize(b.astype(f), emul)
        return jnp.matmul(a.astype(mm), b, precision=prec, preferred_element_type=f)

    def decimate(x, Fm, b, hi, D):
        """x [..., n_in] times the decimation matrix along the last axis: dense, banded, or (D = 1) nothing."""
        if plan['block'] and D == 1:
            return x.astype(f)
        if plan['block'] < 0:
            # timing bound only: plain subsampling to the output length, no filter. The image is wrong;
            # the run time is what any implementation of the filter could at best approach
            n_out = Fm.shape[1]
            xs = x[..., ::D].astype(f)
            return jnp.pad(xs, [(0, 0)] * (x.ndim - 1) + [(0, n_out - xs.shape[-1])])
        if b is None:
            return mmul(x, Fm, hi)
        xp = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(b['pl'], b['pr'])])
        xb = xp.reshape(x.shape[:-1] + (b['nb'], b['block']))
        y = mmul(xb[..., 0:b['nbo'], :], Fm[0], hi)
        for t in range(1, b['w']):
            y = y + mmul(xb[..., t:t + b['nbo'], :], Fm[t], hi)
        return y.reshape(x.shape[:-1] + (b['nbo'] * b['G'],))[..., :b['n_out']]

    def level(hre, him, la, lv, first, protect):
        K, P, Ko, Po, Cn = lv['K'], lv['P'], lv['Ko'], lv['Po'], lv['C']
        bk, bp = (lv['band']['k'], lv['band']['p']) if lv['band'] else (None, None)
        k0, k1 = 2.0 * lv['f0'] / C, 2.0 * lv['df'] / C
        cb = _chunk(Cn, P * K, budget)
        kk = jnp.arange(K, dtype=jnp.int32).astype(f)

        def phases(refb):
            """Rotation phases for the children of one parent, computed here from
            the parent's reference point: |w - d| - |w| with w the vector from the
            parent to the antenna and d the child offset, in its cancellation-free form."""
            u, r0, d = la['u'], la['r0'], la['d']
            uc = u[:, 0] * refb[0] + u[:, 1] * refb[1] + u[:, 2] * refb[2]                   # [P]
            wn = r0 * jnp.sqrt(1 + ((refb * refb).sum() - 2 * r0 * uc) / (r0 * r0))            # [P]
            ud = d[:, 0:1] * u[None, :, 0] + d[:, 1:2] * u[None, :, 1] + d[:, 2:3] * u[None, :, 2]
            wd = r0[None, :] * ud - (d * refb[None, :]).sum(1)[:, None]                       # [C, P]
            num = (d * d).sum(1)[:, None] - 2 * wd
            ddr = num / (jnp.sqrt(wn[None, :] * wn[None, :] + num) + wn[None, :])
            c0 = ddr * k0
            return c0 - jnp.round(c0), ddr * k1

        def per_parent(a):
            if first:
                pre, pim, c0, sl = a
            else:
                pre, pim, refb = a
                c0, sl = phases(refb)
            c0 = c0.reshape(Cn // cb, cb, P)
            sl = sl.reshape(Cn // cb, cb, P)

            def per_chunk(b):
                c0c, slc = b
                hi = protect and first_mm is not None

                def rotate_k(pre_, pim_, c0_, sl_):
                    cyc = c0_[:, :, None] + kk[None, None, :] * sl_[:, :, None]
                    ang = (cyc - jnp.round(cyc)) * two_pi
                    cs, sn = jnp.cos(ang).astype(ew), jnp.sin(ang).astype(ew)
                    x = jnp.concatenate([pre_[None] * cs - pim_[None] * sn, pre_[None] * sn + pim_[None] * cs], axis=0)
                    return decimate(x, la['Fk'], bk, hi, lv['D'])

                # bounded-memory mode: pulses are independent up to this point, so a level whose input
                # exceeds the budget is rotated and filtered along frequency a block of pulses at a time
                nb = -(-(P * K) // budget) if groups else 1
                if nb > 1:
                    edges = [round(i * P / nb) for i in range(nb + 1)]
                    y = jnp.concatenate([rotate_k(pre[i0:i1], pim[i0:i1], c0c[:, i0:i1], slc[:, i0:i1])
                                         for i0, i1 in zip(edges[:-1], edges[1:])], axis=1)
                else:
                    y = rotate_k(pre, pim, c0c, slc)                                               # [2cb, P, Ko]
                if left and bp is None and lv['D'] > 1 and not emul:
                    # pulse-axis filter applied from the left, which avoids transposing the data twice
                    ft = jnp.swapaxes(la['Fp'], 0, 1)
                    if hi:
                        z = jnp.matmul(ft.astype(f), y.astype(f), precision='highest', preferred_element_type=f)
                    else:
                        z = jnp.matmul(ft, y.astype(mm), precision=prec, preferred_element_type=f)
                else:
                    z = jnp.swapaxes(decimate(jnp.swapaxes(y, 1, 2), la['Fp'], bp, hi, lv['D']), 1, 2)  # [2cb, Po, Ko]
                return z[:cb].astype(ew), z[cb:].astype(ew)

            if groups and Cn == cb:                # one chunk: no loop, so the parent is not held as loop state
                ore, oim = per_chunk((c0[0], sl[0]))
            else:
                ore, oim = lax.map(per_chunk, (c0, sl))
            return ore.reshape(Cn, Po, Ko), oim.reshape(Cn, Po, Ko)

        xs = (hre, him, la['cyc0'], la['slope']) if first else (hre, him, la['ref'])
        if groups and first:                       # the phase history is the only parent
            ore, oim = per_parent(tuple(x[0] for x in xs))
        else:
            ore, oim = lax.map(per_parent, xs)
        return ore.reshape(-1, Po, Ko), oim.reshape(-1, Po, Ko)

    fin = plan['final']
    Pf, Qf = fin['P'], fin['K']
    # plain Python floats: a NumPy float64 scalar would promote float32 work to float64 when x64 is on
    a0, a1, fc2 = float(2.0 * fin['f0'] / C), float(2.0 * fin['df'] / C), float(2.0 * fin['fc'] / C)
    dl_host = (np.arange(T) - (T - 1) / 2.0) * plan['spacing']

    def final(hre, him, fa):
        B = hre.shape[0]
        tb = _chunk(B, T * Pf * Qf, budget)
        dl = jnp.asarray(dl_host, f)
        q = jnp.arange(Qf, dtype=jnp.int32).astype(f)

        def rot(g):
            """cos, sin of -2 pi (2 f_q / c) g for g [tb, T, P, 1]."""
            c = g * a0
            c = c - jnp.round(c)
            c = c + (g * a1) * q[None, None, None, :]
            ang = (c - jnp.round(c)) * (-two_pi)
            return jnp.cos(ang).astype(ew), jnp.sin(ang).astype(ew)

        def per_chunk(a):
            tre, tim, cen = a
            # unit vector from each tile centre to each pulse, and the tile's mean range
            u, r0 = fa['u'], fa['r0']
            wx = r0[None, :] * u[None, :, 0] - cen[:, 0:1]
            wy = r0[None, :] * u[None, :, 1] - cen[:, 1:2]
            wz = r0[None, :] * u[None, :, 2] - cen[:, 2:3]
            wn = jnp.sqrt(wx * wx + wy * wy + wz * wz)
            ux, uy, uz = wx / wn, wy / wn, wz / wn                                             # [tb, P]
            mx, my, mz = ux.mean(1), uy.mean(1), uz.mean(1)
            mn = jnp.sqrt(mx * mx + my * my + mz * mz)
            ucx, ucy, rc = mx / mn, my / mn, wn.mean(1)
            cs, sn = rot(ux[:, None, :, None] * dl[None, :, None, None])
            A = jnp.concatenate([tre[:, None] * cs - tim[:, None] * sn, tre[:, None] * sn + tim[:, None] * cs], axis=1)
            cs, sn = rot(uy[:, None, :, None] * dl[None, :, None, None])
            Bm = jnp.concatenate([cs, sn], axis=1)
            M = mmul(A.reshape(tb, 2 * T, Pf * Qf), jnp.swapaxes(Bm.reshape(tb, 2 * T, Pf * Qf), 1, 2).astype(mm))
            cre = M[:, :T, :T] - M[:, T:, T:]
            cim = M[:, :T, T:] + M[:, T:, :T]
            dx, dy = dl[None, :, None], dl[None, None, :]
            los = ucx[:, None, None] * dx + ucy[:, None, None] * dy
            qc = (dx * dx + dy * dy - los * los) * (fc2 / (2.0 * rc[:, None, None]))
            ang = (qc - jnp.round(qc)) * two_pi
            c, s = jnp.cos(ang), jnp.sin(ang)
            return cre * c - cim * s, cre * s + cim * c

        xs = tuple(v.reshape((B // tb, tb) + v.shape[1:]) for v in (hre, him, fa['cen']))
        ire, iim = lax.map(per_chunk, xs)
        return ire.reshape(B, T, T), iim.reshape(B, T, T)

    split = plan['split']
    L = len(split)
    # tiles are ordered (x1, y1, x2, y2, ...); bring to (x1, x2, ..., Tx, y1, y2, ..., Ty)
    perm = [2 * i for i in range(L)] + [2 * L] + [2 * i + 1 for i in range(L)] + [2 * L + 1]
    shape = [s for s in split for _ in (0, 1)] + [T, T]

    # the level that the mixed policies keep in float32 is the first one that filters
    first_dec = next((i for i, lv in enumerate(plan['levels']) if lv['D'] > 1), 0)

    def ffbp(hre, him, arrs):
        for i, (la, lv) in enumerate(zip(arrs['levels'], plan['levels'])):
            hre, him = level(hre, him, la, lv, i == 0, i == first_dec)
        ire, iim = final(hre, him, arrs['final'])
        return (ire.reshape(shape).transpose(perm).reshape(n, n),
                iim.reshape(shape).transpose(perm).reshape(n, n))

    if not groups:
        return jax.jit(ffbp) if jit else ffbp

    # Bounded-memory form: the first-level tiles are taken in `groups` groups and each group is carried
    # through every level in its own device call, so that only one group's intermediate data exist at a
    # time and the phase history is an argument of each call rather than the state of a compiled loop.
    lv0 = plan['levels'][0]
    C1, P0 = lv0['C'], lv0['P']
    G = min(groups, C1)
    assert C1 % G == 0
    Cg = C1 // G
    lvg = dict(lv0, C=Cg)

    @jax.jit
    def one_group(hre, him, arrs, g):
        la0 = arrs['levels'][0]
        cyc0 = la0['cyc0'][0].reshape(G, Cg, P0)[g]
        slope = la0['slope'][0].reshape(G, Cg, P0)[g]
        a, b = level(hre, him, dict(la0, cyc0=cyc0[None], slope=slope[None]), lvg, True, first_dec == 0)
        for i in range(1, L):
            la = arrs['levels'][i]
            a, b = level(a, b, dict(la, ref=la['ref'].reshape(G, -1, 3)[g]), plan['levels'][i], False, i == first_dec)
        fa = arrs['final']
        re, im = final(a, b, dict(fa, cen=fa['cen'].reshape(G, -1, 3)[g]))
        if Cg == 1:                                # one first-level tile: arrange its pixels here, as an m x m block
            return re.reshape(shape_g).transpose(perm_g).reshape(m, m), im.reshape(shape_g).transpose(perm_g).reshape(m, m)
        return re, im

    s0 = split[0]
    m = n // s0
    shape_g = [s for s in split[1:] for _ in (0, 1)] + [T, T]
    perm_g = [2 * i for i in range(L - 1)] + [2 * (L - 1)] + [2 * i + 1 for i in range(L - 1)] + [2 * (L - 1) + 1]

    @jax.jit
    def assemble(planes):                          # one plane per call, which halves the memory of this step
        if Cg == 1:                                # blocks are ordered (x1, y1); the image is their s0 x s0 grid
            return jnp.concatenate([jnp.concatenate(planes[x * s0:(x + 1) * s0], axis=1) for x in range(s0)], axis=0)
        return jnp.concatenate(planes).reshape(shape).transpose(perm).reshape(n, n)

    def ffbp_grouped(hre, him, arrs):
        parts = [one_group(hre, him, arrs, np.int32(g)) for g in range(G)]
        del hre, him
        ire = assemble([q[0] for q in parts])
        iim = assemble([q[1] for q in parts])
        return ire, iim

    return ffbp_grouped



def make_ffbp_batched(policy, plan, batch, budget=1 << 26):
    """Form `batch` images in one call: ffbp(H_re [B,1,Np,K], H_im, arrays) -> ([B,n,n], [B,n,n]).
    Each image has its own flight path (the per-collection leaves of `arrays` carry a
    leading batch axis); the filters and the tile grid are shared."""
    raw = make_ffbp(policy, plan, max(budget // batch, 1 << 20), jit=False)
    axes = dict(levels=[dict(Fk=None, Fp=None, cyc0=0, slope=0)] +
                       [dict(Fk=None, Fp=None, ref=None, d=None, u=0, r0=0) for _ in plan['levels'][1:]],
                final=dict(cen=None, u=0, r0=0))
    return jax.jit(jax.vmap(raw, in_axes=(0, 0, axes)))


def stack_collections(policy, plan, colls, static):
    """Device arrays for a batch: `static` (from device_arrays) supplies the shared leaves,
    `colls` (one collection_arrays result per image) the per-image ones."""
    h = np.float64 if POLICIES[policy]['ew'] == 'float64' else np.float32

    def dev(key, lvl=None):
        if lvl is None:
            return jnp.asarray(np.stack([c['final'][key] for c in colls]).astype(h))
        return jnp.asarray(np.stack([c['levels'][lvl][key] for c in colls]).astype(h))

    lv = []
    for i, e in enumerate(static['levels']):
        e = dict(e)
        if i == 0:
            e.update(cyc0=dev('cyc0', 0), slope=dev('slope', 0))
        else:
            e.update(u=dev('u', i), r0=dev('r0', i))
        lv.append(e)
    return dict(levels=lv, final=dict(cen=static['final']['cen'], u=dev('u'), r0=dev('r0')))


def device_arrays(policy, plan, coll):
    """Device inputs: mode constants from `plan`, flight path from `coll`."""
    p = POLICIES[policy]
    mm = jnp.dtype(p['mm'])
    h = np.float64 if p['ew'] == 'float64' else np.float32

    def dev(x):
        return jnp.asarray(np.asarray(x).astype(h))

    lv = []
    first_dec = next((i for i, l in enumerate(plan['levels']) if l['D'] > 1), 0)
    for i, (l, c) in enumerate(zip(plan['levels'], coll['levels'])):
        keep = i == first_dec and p.get('first') is not None
        Fk, Fp = (l['band']['k']['W'], l['band']['p']['W']) if l['band'] else (l['Fk'], l['Fp'])
        e = dict(Fk=dev(Fk) if keep else dev(Fk).astype(mm), Fp=dev(Fp) if keep else dev(Fp).astype(mm))
        if i == 0:
            e.update(cyc0=dev(c['cyc0']), slope=dev(c['slope']))
        else:
            e.update(ref=dev(l['ref']), d=dev(l['d']), u=dev(c['u']), r0=dev(c['r0']))
        lv.append(e)
    fin = dict(cen=dev(plan['final']['cen']), u=dev(coll['final']['u']), r0=dev(coll['final']['r0']))
    return dict(levels=lv, final=fin)


def prepare(policy, S, window=None):
    """Phase history -> [1, Np, K] real and imaginary planes, scaled to unit peak."""
    p = POLICIES[policy]
    h = np.float64 if p['ew'] == 'float64' else np.float32
    if window is not None:
        S = S * window
    S = S / np.abs(S).max()
    return (jnp.asarray(S.real.astype(h)[None]).astype(p['ew']),
            jnp.asarray(S.imag.astype(h)[None]).astype(p['ew']))


def default_levels(n, K, T=64, nlev=2):
    """Split for power-of-two n and K that keeps each tile's data sampled at
    about twice its pixel count. Tiles per axis are factored into `nlev`
    powers of two, larger factors first."""
    bits = int(math.log2(n // T))
    per = [bits // nlev + (1 if i < bits % nlev else 0) for i in range(nlev)]
    levels, size, k = [], n, K
    for b in per:
        s = 1 << b
        size //= s
        d = max(1, k // (2 * size))
        levels.append((s, d))
        k //= d
    return levels
