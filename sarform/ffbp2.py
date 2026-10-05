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
    return best[1]


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
        pass_k = rk / (C / (2.0 * df * Dk) / 2.0)
        pass_p = 2.0 * (2.0 * (f0 + K * df) / C) * dop * Dp
        Fk, mk = decimator(K, Dk, min(pass_k, 0.95), atten)
        Fp, mp = decimator(P, Dp, min(pass_p, 0.95), atten)
        pidx = Dp * (np.arange(Fp.shape[1]) - mp) + (Dp - 1) / 2.0
        out['levels'].append(dict(sx=sx, sy=sy, C=sx * sy, Dk=Dk, Dp=Dp, Fk=Fk, Fp=Fp, K=K, P=P, Ko=Fk.shape[1], Po=Fp.shape[1],
                                  fir_k=fir(Fk, Dk, mk) if Dk > 1 else None, fir_p=fir(Fp, Dp, mp) if Dp > 1 else None,
                                  f0=f0, df=df, ref=ref, d=cen[0] - ref[0], cen=cen, pidx=pidx,
                                  pass_k=float(pass_k), pass_p=float(pass_p)))
        ant = _positions(ant, pidx)
        f0, df, K, P = f0 + ((Dk - 1) / 2.0 - mk * Dk) * df, df * Dk, Fk.shape[1], Fp.shape[1]
        ref, i0 = cen.reshape(-1, 3), ci0.reshape(-1, 2)
    assert (cx, cy) == (T, T)
    out['final'] = dict(cen=ref, f0=f0, df=df, K=K, P=P, fc=f0 + (K - 1) / 2.0 * df)
    return out


HOST_LEVELS = 2          # levels whose rotation phases are computed in float64 on the host


def collection_arrays(plan, ant):
    """Per-collection host work in float64: the antenna path on each level's pulse grid, and for the first
    HOST_LEVELS levels the rotation phase of every child at band centre (wrapped to a cycle) and its slope
    in cycles per frequency sample."""
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
            if l['Dp'] > 1:
                e['Fp'] = dev(l['Fp']).astype(mm)
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
    scale = float(np.abs(S).max())
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
        return on_tpu and pallas_gen >= 3 and lv['Dp'] > 1 and Po_pad * band['kob'] * nc * 16 <= 40 << 20

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
                if fused_fin == 3:
                    cre, cim = fused_final3(dTr, dTi, gxT, gyT, a0, a1, Qf, passes=npass, tiles=tiles)
                else:
                    cre, cim = fused_final2(dTr, dTi, gxT, gyT, a0, a1, Qf, passes=npass)
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
        re, im = final(a, b, dict(fa, cen=fa['cen'].reshape(G, -1, 3)[g]))
        return re.reshape(shape_g).transpose(perm_g).reshape(mx, my), im.reshape(shape_g).transpose(perm_g).reshape(mx, my)

    ox, oy, nx, ny = plan['ox'], plan['oy'], plan['nx'], plan['ny']

    @jax.jit
    def assemble(blocks):
        full = jnp.concatenate([jnp.concatenate(blocks[x * sy0:(x + 1) * sy0], axis=1) for x in range(sx0)], axis=0)
        return full[ox:ox + nx, oy:oy + ny]

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
            re, im = final(a, b, dict(fa, cen=fa['cen'].reshape(G, -1, 3)[g]))
            return re.reshape(shape_g).transpose(perm_g).reshape(mx, my), im.reshape(shape_g).transpose(perm_g).reshape(mx, my)

        return lax.map(rest, (a0, b0, gs))

    def form(hre, him, arrs):
        if ng > 1:
            parts = []
            for g0 in range(0, G, ng):
                re, im = one_group(hre, him, arrs, jnp.arange(g0, g0 + ng, dtype=jnp.int32))
                parts += [(re[i], im[i]) for i in range(ng)]
        else:
            parts = [one_tile(hre, him, arrs, np.int32(g)) for g in range(G)]
        del hre, him
        return assemble([q[0] for q in parts]), assemble([q[1] for q in parts])

    # the stages of one_tile, exposed for per-stage profiling (profile_level0.py)
    form.stages = dict(children=children, final=final, device_phases=device_phases, levels=levels, G=G, host_levels=HOST_LEVELS, one_group=one_group, ng=ng, bands=bands)
    return form
