"""Fused phase rotation and frequency-axis decimation for the factorized algorithm, as a Pallas TPU kernel.

The XLA path forms, for every child tile, the rotated phase history S * exp(j 2 pi (c0 + (k - kc) slope)) in memory and
then multiplies it by the decimation matrix. The rotation alone moves 24 bytes per sample through HBM and runs at about
a third of the v6e's memory bandwidth. This kernel keeps a block of pulses in VMEM, applies the ramp as the product of a
coarse table (one angle per pulse and column chunk) and a fine table (one per pulse and lane within the chunk), and
multiplies the rotated block by the banded decimation matrix one output block at a time on the matrix units. The phase
history is read from HBM once per child and the decimated result written once.

    y = fused_rotate_dec_k(S_re, S_im, c0, slope, band, geom)   ->  (y_re, y_im), each [P, Ko]

band: dict from `band_blocks` (the decimation matrix cut into [nb, win, kob] blocks on the padded column index).
geom: dict(K, kc, pl_pad) giving the unpadded sample count, the ramp centre and the left padding in columns.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TWO_PI = 2.0 * math.pi
LANE = 128


def band_blocks(Fk, D, kob=256, lane=LANE, chunk_align=512):
    """Cut the decimation matrix Fk [K, Ko] (output j depends on inputs D j + r - pl, r < L) into blocks.

    Returns dict(blocks [nb, win, kob] float32, Kpad, pl_pad, Ko_pad, kob, win, stride) such that for a row vector z of
    length K padded with pl_pad zeros on the left and to Kpad on the right, z_pad[b*stride : b*stride + win] @ blocks[b]
    equals the outputs b*kob .. (b+1)*kob - 1 of z @ Fk (zero beyond Ko).
    """
    K, Ko = Fk.shape
    nz = np.nonzero(Fk[:, Ko // 2])[0]
    L = int(nz[-1] - nz[0] + 1)
    # padded column index: kp = k + pl_pad; output j uses kp in [D j + off, D j + off + L) with off = first nz of column 0 + pl_pad
    first0 = int(np.nonzero(Fk[:, 0])[0][0]) if np.any(Fk[:, 0]) else 0
    # choose pl_pad so that column 0's window starts at a non-negative padded index and is a multiple of the lane width
    pl_pad = lane * int(math.ceil(max(0, L - first0) / lane))
    stride = D * kob
    nb = -(-Ko // kob)
    Ko_pad = nb * kob
    win = int(lane * math.ceil((stride + L + pl_pad) / lane))
    Kpad = int(chunk_align * math.ceil((pl_pad + K + win) / chunk_align))
    Fpad = np.zeros((Kpad, Ko_pad), np.float64)
    Fpad[pl_pad:pl_pad + K, :Ko] = Fk
    blocks = np.zeros((nb, win, kob), np.float32)
    for b in range(nb):
        r0 = b * stride
        blocks[b] = Fpad[r0:r0 + win, b * kob:(b + 1) * kob]
        # every nonzero of these columns must lie inside the window
        cols = Fpad[:, b * kob:(b + 1) * kob]
        rows = np.nonzero(np.abs(cols).sum(1))[0]
        assert rows.size == 0 or (rows.min() >= r0 and rows.max() < r0 + win), (b, rows.min(), rows.max(), r0, win)
    return dict(blocks=jnp.asarray(blocks), Kpad=Kpad, pl_pad=pl_pad, Ko_pad=Ko_pad, kob=kob, win=win, stride=stride, nb=nb, Ko=Ko, K=K)


def pad_columns(x, band):
    """[..., K] -> [..., Kpad] with pl_pad zeros on the left."""
    K = x.shape[-1]
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(band['pl_pad'], band['Kpad'] - band['pl_pad'] - K)])


def _split_bf16(x):
    hi = x.astype(jnp.bfloat16)
    lo = (x - hi.astype(jnp.float32)).astype(jnp.bfloat16)
    return hi, lo


def fused_rotate_dec_k(S_re, S_im, c0, slope, band, kc, pb=128, chunk=512, passes=1, interpret=False, vmem_limit=100 << 20):
    """S_re, S_im [P, Kpad] float32 (padded with pad_columns); c0, slope [P] float32 cycles and cycles per sample;
    band from band_blocks; kc the ramp centre in unpadded sample index. Returns (y_re, y_im) [P, Ko_pad] float32."""
    P, Kpad = S_re.shape
    assert Kpad == band['Kpad'] and Kpad % chunk == 0 and P % pb == 0, (P, Kpad, band['Kpad'], chunk, pb)
    nb, win, kob, stride, pl_pad = band['nb'], band['win'], band['kob'], band['stride'], band['pl_pad']
    Ko_pad = band['Ko_pad']
    nchunk = Kpad // chunk
    blocks = band['blocks']
    fb = blocks.astype(jnp.bfloat16)
    if passes == 3:
        fb_hi, fb_lo = _split_bf16(blocks)

    def kernel(sre_ref, sim_ref, c0_ref, sl_ref, lane_ref, *rest):
        if passes == 3:
            fhi_ref, flo_ref, ore_ref, oim_ref, zre, zim = rest
        else:
            f_ref, ore_ref, oim_ref, zre, zim = rest
        c0b = c0_ref[...]                                               # [pb, 1]
        slb = sl_ref[...]                                               # [pb, 1]
        # fine table: angle of lane j within a chunk, 2 pi frac(j * slope)
        lanes = lane_ref[...]                                           # [1, chunk] fine index within a chunk
        fcyc = slb * lanes                                              # [pb, chunk]
        fang = (fcyc - jnp.floor(fcyc + 0.5)) * TWO_PI
        fcos, fsin = jnp.cos(fang), jnp.sin(fang)

        def rot_chunk(c, carry):
            k0 = c * chunk - pl_pad - kc                                # unpadded sample offset of the chunk's first lane, minus the centre
            ccyc = c0b + slb * k0.astype(jnp.float32)                   # [pb, 1]
            cang = (ccyc - jnp.floor(ccyc + 0.5)) * TWO_PI
            ccos, csin = jnp.cos(cang), jnp.sin(cang)
            cs = ccos * fcos - csin * fsin                              # cos(A + B)
            sn = csin * fcos + ccos * fsin                              # sin(A + B)
            start = pl.multiple_of(c * chunk, chunk)
            sr = sre_ref[:, pl.ds(start, chunk)]
            si = sim_ref[:, pl.ds(start, chunk)]
            zre[:, pl.ds(start, chunk)] = sr * cs - si * sn
            zim[:, pl.ds(start, chunk)] = sr * sn + si * cs
            return carry

        lax.fori_loop(0, nchunk, rot_chunk, 0)

        def dec_block(b, carry):
            r0 = pl.multiple_of(b * stride, LANE)
            zr = zre[:, pl.ds(r0, win)]
            zi = zim[:, pl.ds(r0, win)]
            o0 = pl.multiple_of(b * kob, kob)
            if passes == 3:
                F_hi, F_lo = fhi_ref[b], flo_ref[b]
                zr_hi, zr_lo = _split_bf16(zr)
                zi_hi, zi_lo = _split_bf16(zi)
                yr = (jnp.dot(zr_hi, F_hi, preferred_element_type=jnp.float32) + jnp.dot(zr_hi, F_lo, preferred_element_type=jnp.float32)
                      + jnp.dot(zr_lo, F_hi, preferred_element_type=jnp.float32))
                yi = (jnp.dot(zi_hi, F_hi, preferred_element_type=jnp.float32) + jnp.dot(zi_hi, F_lo, preferred_element_type=jnp.float32)
                      + jnp.dot(zi_lo, F_hi, preferred_element_type=jnp.float32))
            else:
                F = f_ref[b]
                yr = jnp.dot(zr.astype(jnp.bfloat16), F, preferred_element_type=jnp.float32)
                yi = jnp.dot(zi.astype(jnp.bfloat16), F, preferred_element_type=jnp.float32)
            ore_ref[:, pl.ds(o0, kob)] = yr
            oim_ref[:, pl.ds(o0, kob)] = yi
            return carry

        lax.fori_loop(0, nb, dec_block, 0)

    grid = (P // pb,)
    in_specs = [pl.BlockSpec((pb, Kpad), lambda i: (i, 0)), pl.BlockSpec((pb, Kpad), lambda i: (i, 0)),
                pl.BlockSpec((pb, 1), lambda i: (i, 0)), pl.BlockSpec((pb, 1), lambda i: (i, 0)),
                pl.BlockSpec((1, chunk), lambda i: (0, 0))]
    args = [S_re, S_im, c0.reshape(P, 1), slope.reshape(P, 1), jnp.arange(chunk, dtype=jnp.float32).reshape(1, chunk)]
    if passes == 3:
        in_specs += [pl.BlockSpec((nb, win, kob), lambda i: (0, 0, 0))] * 2
        args += [fb_hi, fb_lo]
    else:
        in_specs += [pl.BlockSpec((nb, win, kob), lambda i: (0, 0, 0))]
        args += [fb]
    out_specs = [pl.BlockSpec((pb, Ko_pad), lambda i: (i, 0))] * 2
    call = pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((P, Ko_pad), jnp.float32)] * 2,
                          scratch_shapes=[pltpu.VMEM((pb, Kpad), jnp.float32)] * 2,
                          compiler_params=None if interpret else pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
                          interpret=interpret)
    return call(*args)


# ----------------------------------------------------------------------------------------------------------------------
# Second version: windowed blocks with a halo, several children per load, no full-row scratch. Serves every level.
#
# The phase history [N, P, Kpad] is cut into column blocks of `stride` = Dk * kob samples. Output block b of a child
# depends on input columns [b stride, b stride + win) with win <= 2 stride, so a grid step over (parent n, output block
# b, pulse block i) receives column blocks b and b + 1, rotates the first `win` columns of their concatenation for each
# of the nc children in VMEM, and multiplies by the banded filter block on the matrix units. The parent is read about
# twice per group of nc children instead of three times per child, and the filter block stays resident across the
# pulse blocks because b is the outer grid axis.
# ----------------------------------------------------------------------------------------------------------------------

def band_blocks2(Fk, D, kob=None, lane=LANE, chunk=128, kob_max=512):
    """Decimation matrix Fk [K, Ko] cut into [nb, win, kob] blocks on the padded column index, with win a multiple of
    `chunk` and at most 2 * stride. When one block of outputs covers the whole row (Ko <= kob_max) there is a single
    block whose window is clipped to the data, so no column is rotated twice or needlessly.
    Returns dict(blocks, Kpad, pl_pad, Ko_pad, kob, win, stride, nb, Ko, K, chunk)."""
    K, Ko = Fk.shape
    nz = np.nonzero(Fk[:, Ko // 2])[0]
    L = int(nz[-1] - nz[0] + 1)
    first0 = int(np.nonzero(Fk[:, 0])[0][0]) if np.any(Fk[:, 0]) else 0
    pl_pad = lane * int(math.ceil(max(0, L - first0) / lane))
    if kob is None:
        kob = lane * int(math.ceil(Ko / lane))
        if kob > kob_max:
            kob = 256
    nb = -(-Ko // kob)
    Ko_pad = nb * kob
    if nb == 1:
        win = int(chunk * math.ceil((pl_pad + K) / chunk))
        stride = win
        Kpad = win
    else:
        stride = D * kob
        assert stride % lane == 0, (D, kob)
        if stride % chunk:
            chunk = lane                                    # the chunks must not straddle the two column blocks
        win = int(chunk * math.ceil((stride + L + pl_pad) / chunk))
        assert win <= 2 * stride, (win, stride, L, pl_pad)
        Kpad = (nb + 1) * stride                            # one spare block so that the halo of the last block exists
    assert Kpad >= pl_pad + K, (Kpad, pl_pad, K)
    Fpad = np.zeros((Kpad, Ko_pad), np.float64)
    Fpad[pl_pad:pl_pad + K, :Ko] = Fk
    blocks = np.zeros((nb, win, kob), np.float32)
    for b in range(nb):
        r0 = b * stride
        blocks[b] = Fpad[r0:r0 + win, b * kob:(b + 1) * kob]
        cols = Fpad[:, b * kob:(b + 1) * kob]
        rows = np.nonzero(np.abs(cols).sum(1))[0]
        assert rows.size == 0 or (rows.min() >= r0 and rows.max() < r0 + win), (b, rows.min(), rows.max(), r0, win)
    return dict(blocks=jnp.asarray(blocks), Kpad=Kpad, pl_pad=pl_pad, Ko_pad=Ko_pad, kob=kob, win=win, stride=stride, nb=nb,
                Ko=Ko, K=K, chunk=chunk)


def fused_rotate_dec_k2(S_re, S_im, c0, slope, band, kc, pb=256, passes=1, interpret=False, vmem_limit=100 << 20):
    """S_re, S_im [N, P, Kpad] float32 (pad_columns with the band); c0, slope [N, nc, P] float32 (cycles, cycles per
    sample); kc the ramp centre in unpadded sample index. Returns (y_re, y_im) [N, nc, P, Ko_pad] float32."""
    N, P, Kpad = S_re.shape
    nc = c0.shape[1]
    nb, win, kob, stride, pl_pad, chunk = band['nb'], band['win'], band['kob'], band['stride'], band['pl_pad'], band['chunk']
    Ko_pad = band['Ko_pad']
    assert Kpad == band['Kpad'] and P % pb == 0 and c0.shape == (N, nc, P), (S_re.shape, c0.shape, band['Kpad'], pb)
    nch = win // chunk
    blocks = band['blocks']
    if passes == 3:
        fb_hi, fb_lo = _split_bf16(blocks)
    else:
        fb = blocks.astype(jnp.bfloat16)

    def kernel(sa_ref, sb_ref, ta_ref, tb_ref, c0_ref, sl_ref, fcos_ref, fsin_ref, chunk_ref, *rest):
        if passes == 3:
            fhi_ref, flo_ref, ore_ref, oim_ref, zre, zim = rest
        else:
            f_ref, ore_ref, oim_ref, zre, zim = rest
        b = pl.program_id(1)
        k0 = (b * stride - pl_pad).astype(jnp.float32) - kc              # unpadded index of the window's first column

        def one_child(c, carry):
            c0b = c0_ref[c]                                             # [pb, 1]
            slb = sl_ref[c]
            fcos, fsin = fcos_ref[c], fsin_ref[c]                       # fine table [pb, chunk]: 2 pi frac(l slope)
            ccyc = c0b + slb * (k0 + chunk_ref[...])                    # coarse table [pb, nch], one lane per chunk
            cang = (ccyc - jnp.floor(ccyc + 0.5)) * TWO_PI
            ccos_all, csin_all = jnp.cos(cang), jnp.sin(cang)
            for m in range(nch):                                        # window chunk m: columns m chunk .. (m + 1) chunk
                ccos, csin = ccos_all[:, m:m + 1], csin_all[:, m:m + 1]
                cs = ccos * fcos - csin * fsin
                sn = csin * fcos + ccos * fsin
                lo = m * chunk
                if lo + chunk <= stride:
                    sr, si = sa_ref[:, lo:lo + chunk], ta_ref[:, lo:lo + chunk]
                else:
                    sr, si = sb_ref[:, lo - stride:lo - stride + chunk], tb_ref[:, lo - stride:lo - stride + chunk]
                zre[:, lo:lo + chunk] = sr * cs - si * sn
                zim[:, lo:lo + chunk] = sr * sn + si * cs
            zr, zi = zre[...], zim[...]
            if passes == 3:
                F_hi, F_lo = fhi_ref[...], flo_ref[...]
                zr_hi, zr_lo = _split_bf16(zr)
                zi_hi, zi_lo = _split_bf16(zi)
                yr = (jnp.dot(zr_hi, F_hi, preferred_element_type=jnp.float32) + jnp.dot(zr_hi, F_lo, preferred_element_type=jnp.float32)
                      + jnp.dot(zr_lo, F_hi, preferred_element_type=jnp.float32))
                yi = (jnp.dot(zi_hi, F_hi, preferred_element_type=jnp.float32) + jnp.dot(zi_hi, F_lo, preferred_element_type=jnp.float32)
                      + jnp.dot(zi_lo, F_hi, preferred_element_type=jnp.float32))
            else:
                F = f_ref[...]
                yr = jnp.dot(zr.astype(jnp.bfloat16), F, preferred_element_type=jnp.float32)
                yi = jnp.dot(zi.astype(jnp.bfloat16), F, preferred_element_type=jnp.float32)
            ore_ref[c] = yr
            oim_ref[c] = yi
            return carry

        lax.fori_loop(0, nc, one_child, 0)

    assert stride % chunk == 0, (stride, chunk)
    grid = (N, nb, P // pb)
    blk = lambda n, b, i: (n, i, b)
    halo = (lambda n, b, i: (n, i, b + 1)) if nb > 1 else blk           # a single block needs no halo
    # fine tables once per child and pulse, [N, nc, P, chunk]. (Building them by doubling, log2(chunk) sines per pulse
    # and seven lane-wise concatenations, doubled the image time on both TPUs: each intermediate table of fewer than
    # 128 lanes occupies full lane tiles, so the concatenations move more data than the sines they save.)
    fcyc = slope[..., None] * jnp.arange(chunk, dtype=jnp.float32)
    fang = (fcyc - jnp.floor(fcyc + 0.5)) * TWO_PI
    fcos, fsin = jnp.cos(fang), jnp.sin(fang)
    in_specs = [pl.BlockSpec((None, pb, stride), blk), pl.BlockSpec((None, pb, stride), halo),
                pl.BlockSpec((None, pb, stride), blk), pl.BlockSpec((None, pb, stride), halo),
                pl.BlockSpec((None, nc, pb, 1), lambda n, b, i: (n, 0, i, 0)), pl.BlockSpec((None, nc, pb, 1), lambda n, b, i: (n, 0, i, 0)),
                pl.BlockSpec((None, nc, pb, chunk), lambda n, b, i: (n, 0, i, 0)), pl.BlockSpec((None, nc, pb, chunk), lambda n, b, i: (n, 0, i, 0)),
                pl.BlockSpec((1, nch), lambda n, b, i: (0, 0))]
    c0r = c0.reshape(N, nc, P, 1)
    slr = slope.reshape(N, nc, P, 1)
    args = [S_re, S_re, S_im, S_im, c0r, slr, fcos, fsin, (jnp.arange(nch, dtype=jnp.float32) * chunk).reshape(1, nch)]
    if passes == 3:
        in_specs += [pl.BlockSpec((None, win, kob), lambda n, b, i: (b, 0, 0))] * 2
        args += [fb_hi, fb_lo]
    else:
        in_specs += [pl.BlockSpec((None, win, kob), lambda n, b, i: (b, 0, 0))]
        args += [fb]
    out_specs = [pl.BlockSpec((None, nc, pb, kob), lambda n, b, i: (n, 0, i, b))] * 2
    call = pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((N, nc, P, Ko_pad), jnp.float32)] * 2,
                          scratch_shapes=[pltpu.VMEM((pb, win), jnp.float32)] * 2,
                          compiler_params=None if interpret else pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
                          interpret=interpret)
    return call(*args)


# ----------------------------------------------------------------------------------------------------------------------
# Final stage: the 2T x 2T product of the phase-ramped tile data against the second ramp, with both ramps built in VMEM.
# The XLA path materializes A = data * exp(j theta_x) and B = exp(j theta_y), each [2T, Pf Qf] float32 per tile, in HBM
# before one matrix product; here the ramps are built eight pulses at a time in VMEM and accumulated on the matrix units.
# ----------------------------------------------------------------------------------------------------------------------

def fused_final(tre, tim, gxT, gyT, qv, passes=1, pg=8, interpret=False, vmem_limit=64 << 20):
    """tre, tim [B, Pf, Qpad] float32 (zero padded beyond Qf); gxT, gyT [B, T, Pf] float32 (dl[t] * u[p] along x and y);
    qv [1, Qpad] float32 (a0 + a1 q). Returns (cre, cim) [B, T, T] float32: sum over p, q of
    data[p, q] exp(-j 2 pi qv[q] (gx[t, p] + gy[t', p]))."""
    B, Pf, Qpad = tre.shape
    T = gxT.shape[1]
    assert Pf % pg == 0 and Qpad % LANE == 0 and gxT.shape == (B, T, Pf), (tre.shape, gxT.shape)
    ng = Pf // pg

    def cast(x):
        return x.astype(jnp.bfloat16)

    def nt(a, b):
        """a [T, n] . b [T, n]^T -> [T, T] in float32, one or three bfloat16 passes."""
        if passes == 3:
            a_hi, a_lo = _split_bf16(a)
            b_hi, b_lo = _split_bf16(b)
            return (lax.dot_general(a_hi, b_hi, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                    + lax.dot_general(a_hi, b_lo, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                    + lax.dot_general(a_lo, b_hi, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32))
        return lax.dot_general(cast(a), cast(b), (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    def kernel(tre_ref, tim_ref, gx_ref, gy_ref, qv_ref, ore_ref, oim_ref):
        q = qv_ref[...]                                                 # [1, Qpad]
        rr = ri = ir = ii = jnp.zeros((T, T), jnp.float32)
        for g in range(ng):
            Ar, Ai, Br, Bi = [], [], [], []
            for p in range(g * pg, (g + 1) * pg):
                gx = gx_ref[:, p:p + 1]                                 # [T, 1]
                gy = gy_ref[:, p:p + 1]
                ax = -(gx * q)                                          # cycles, [T, Qpad]
                ax = (ax - jnp.floor(ax + 0.5)) * TWO_PI
                ay = -(gy * q)
                ay = (ay - jnp.floor(ay + 0.5)) * TWO_PI
                cx, sx = jnp.cos(ax), jnp.sin(ax)
                cy, sy = jnp.cos(ay), jnp.sin(ay)
                dr = tre_ref[p:p + 1, :]                                # [1, Qpad]
                di = tim_ref[p:p + 1, :]
                Ar.append(dr * cx - di * sx)
                Ai.append(dr * sx + di * cx)
                Br.append(cy)
                Bi.append(sy)
            Ar, Ai, Br, Bi = (jnp.concatenate(v, axis=1) for v in (Ar, Ai, Br, Bi))     # [T, pg Qpad]
            rr = rr + nt(Ar, Br)
            ii = ii + nt(Ai, Bi)
            ri = ri + nt(Ar, Bi)
            ir = ir + nt(Ai, Br)
        ore_ref[...] = rr - ii
        oim_ref[...] = ri + ir

    grid = (B,)
    in_specs = [pl.BlockSpec((None, Pf, Qpad), lambda b: (b, 0, 0)), pl.BlockSpec((None, Pf, Qpad), lambda b: (b, 0, 0)),
                pl.BlockSpec((None, T, Pf), lambda b: (b, 0, 0)), pl.BlockSpec((None, T, Pf), lambda b: (b, 0, 0)),
                pl.BlockSpec((1, Qpad), lambda b: (0, 0))]
    out_specs = [pl.BlockSpec((None, T, T), lambda b: (b, 0, 0))] * 2
    call = pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((B, T, T), jnp.float32)] * 2,
                          compiler_params=None if interpret else pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
                          interpret=interpret)
    return call(tre, tim, gxT, gyT, qv)


def fused_final2(dT_re, dT_im, gx, gy, a0, a1, Qf, passes=1, interpret=False, vmem_limit=64 << 20):
    """Final stage with the ramps generated by a complex recurrence instead of per-sample sines: for each tile the
    ramp over frequency index q at fixed (t, p) is a geometric sequence, exp(-j 2 pi g (a0 + a1 q)) = X W^q, so the
    kernel evaluates four lane-dense tables of sines and cosines per tile (X and W for each ramp) and advances them by
    one complex multiplication per q. The data are held transposed, q on sublanes and p on lanes.

    dT_re, dT_im [B, Qpad, Pl] float32 (q rows, p columns, zero padded); gx, gy [B, T, Pl] float32 (dl[t] u[p]);
    a0, a1 floats (cycles per metre at the first frequency sample, per sample step); Qf the number of valid rows.
    Returns (cre, cim) [B, T, T] float32."""
    B, Qpad, Pl = dT_re.shape
    T = gx.shape[1]
    assert Pl % LANE == 0 and gx.shape == (B, T, Pl) and Qf <= Qpad, (dT_re.shape, gx.shape, Qf)
    n = Qf * Pl

    def nt(a, b):
        if passes == 3:
            a_hi, a_lo = _split_bf16(a)
            b_hi, b_lo = _split_bf16(b)
            return (lax.dot_general(a_hi, b_hi, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                    + lax.dot_general(a_hi, b_lo, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                    + lax.dot_general(a_lo, b_hi, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32))
        return lax.dot_general(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    def table(g, k):
        c = -(g * k)
        ang = (c - jnp.floor(c + 0.5)) * TWO_PI
        return jnp.cos(ang), jnp.sin(ang)

    def kernel(dr_ref, di_ref, gx_ref, gy_ref, ore_ref, oim_ref, ar_s, ai_s, br_s, bi_s):
        gxv, gyv = gx_ref[...], gy_ref[...]                             # [T, Pl]
        xr0, xi0 = table(gxv, a0)                                       # base ramp at q = 0
        wr, wi = table(gxv, a1)                                         # one step in q
        yr0, yi0 = table(gyv, a0)
        vr, vi = table(gyv, a1)

        def step(q, carry):
            xr, xi, yr, yi = carry
            dr = dr_ref[pl.ds(q, 1), :]                                 # [1, Pl]
            di = di_ref[pl.ds(q, 1), :]
            off = pl.multiple_of(q * Pl, LANE)
            ar_s[:, pl.ds(off, Pl)] = dr * xr - di * xi
            ai_s[:, pl.ds(off, Pl)] = dr * xi + di * xr
            br_s[:, pl.ds(off, Pl)] = yr
            bi_s[:, pl.ds(off, Pl)] = yi
            return (xr * wr - xi * wi, xr * wi + xi * wr, yr * vr - yi * vi, yr * vi + yi * vr)

        lax.fori_loop(0, Qf, step, (xr0, xi0, yr0, yi0))
        Ar, Ai, Br, Bi = ar_s[:, :n], ai_s[:, :n], br_s[:, :n], bi_s[:, :n]
        ore_ref[...] = nt(Ar, Br) - nt(Ai, Bi)
        oim_ref[...] = nt(Ar, Bi) + nt(Ai, Br)

    in_specs = [pl.BlockSpec((None, Qpad, Pl), lambda b: (b, 0, 0)), pl.BlockSpec((None, Qpad, Pl), lambda b: (b, 0, 0)),
                pl.BlockSpec((None, T, Pl), lambda b: (b, 0, 0)), pl.BlockSpec((None, T, Pl), lambda b: (b, 0, 0))]
    out_specs = [pl.BlockSpec((None, T, T), lambda b: (b, 0, 0))] * 2
    call = pl.pallas_call(kernel, grid=(B,), in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((B, T, T), jnp.float32)] * 2,
                          scratch_shapes=[pltpu.VMEM((T, Qpad * Pl), jnp.float32)] * 4,
                          compiler_params=None if interpret else pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
                          interpret=interpret)
    return call(dT_re, dT_im, gx, gy)


# ----------------------------------------------------------------------------------------------------------------------
# Third version of the level kernel: no transcendental work inside the kernel (the coarse tables are precomputed
# lane-dense by XLA, one row per child and pulse with one lane per chunk), and the pulse-axis decimation fused as a
# matrix-unit accumulation across the pulse blocks when the decimated output of a child fits in vector memory.
# ----------------------------------------------------------------------------------------------------------------------

def fused_rotate_dec_k3(S_re, S_im, c0, slope, band, kc, pb=256, passes=1, FpT=None, interpret=False, vmem_limit=100 << 20):
    """S_re, S_im [N, P, Kpad] float32; c0, slope [N, nc, P] float32; band from band_blocks2; kc the ramp centre.
    FpT: None, or the transposed pulse decimation matrix [Po, P] (float32); when given the kernel returns the pulse-
    decimated children (y_re, y_im) [N, nc, Po_pad, Ko_pad], otherwise [N, nc, P, Ko_pad] as fused_rotate_dec_k2."""
    N, P, Kpad = S_re.shape
    nc = c0.shape[1]
    nb, win, kob, stride, pl_pad, chunk = band['nb'], band['win'], band['kob'], band['stride'], band['pl_pad'], band['chunk']
    Ko_pad = band['Ko_pad']
    assert Kpad == band['Kpad'] and P % pb == 0 and c0.shape == (N, nc, P) and stride % chunk == 0, (S_re.shape, c0.shape, pb)
    nch = win // chunk
    nchl = LANE * -(-nch // LANE)                                            # coarse table lanes, one per chunk, padded
    blocks = band['blocks']
    # coarse tables: angle of the first lane of chunk m of block b for child c and pulse p, lane-dense per (c, p)
    kk0 = jnp.arange(nb, dtype=jnp.float32)[:, None] * stride - pl_pad - kc                 # [nb, 1]
    moff = jnp.pad(jnp.arange(nch, dtype=jnp.float32) * chunk, (0, nchl - nch))            # [nchl]
    ccyc = c0[:, None, :, :, None] + slope[:, None, :, :, None] * (kk0[None, :, None, None, :] + moff[None, None, None, None, :])   # [N, nb, nc, P, nchl]
    cang = (ccyc - jnp.floor(ccyc + 0.5)) * TWO_PI
    # the sines run over all nchl lanes: XLA holds the table 128 lanes wide either way, and computing over the used
    # lanes only and padding afterwards added a relayout copy that cost 6 percent of the image time
    ccos, csin = jnp.cos(cang), jnp.sin(cang)
    fcyc = slope[..., None] * jnp.arange(chunk, dtype=jnp.float32)                          # fine tables [N, nc, P, chunk]
    fang = (fcyc - jnp.floor(fcyc + 0.5)) * TWO_PI
    fcos, fsin = jnp.cos(fang), jnp.sin(fang)
    if passes == 3:
        fb_hi, fb_lo = _split_bf16(blocks)
    else:
        fb = blocks.astype(jnp.bfloat16)
    fuse_p = FpT is not None
    if fuse_p:
        Po = FpT.shape[0]
        Po_pad = 8 * -(-Po // 8)
        FpT_p = jnp.pad(FpT.astype(jnp.float32), ((0, Po_pad - Po), (0, 0)))
        if passes == 3:
            fp_hi, fp_lo = _split_bf16(FpT_p)
        else:
            fp = FpT_p.astype(jnp.bfloat16)

    def mm(a, b):
        """a [m, k] . b [k, n] on the matrix units in one or three bfloat16 passes; either operand is a float32 array
        (split or cast here) or an operand already in bfloat16 (one array, or a (hi, lo) pair for three passes)."""
        if passes == 3:
            a_hi, a_lo = a if isinstance(a, tuple) else _split_bf16(a)
            b_hi, b_lo = b if isinstance(b, tuple) else _split_bf16(b)
            return (jnp.dot(a_hi, b_hi, preferred_element_type=jnp.float32) + jnp.dot(a_hi, b_lo, preferred_element_type=jnp.float32)
                    + jnp.dot(a_lo, b_hi, preferred_element_type=jnp.float32))
        a = a if a.dtype == jnp.bfloat16 else a.astype(jnp.bfloat16)
        b = b if b.dtype == jnp.bfloat16 else b.astype(jnp.bfloat16)
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    def kernel(sa_ref, sb_ref, ta_ref, tb_ref, cc_ref, cs_ref, fc_ref, fs_ref, *rest):
        rest = list(rest)
        if passes == 3:
            F = (rest.pop(0)[...], rest.pop(0)[...])
        else:
            F = rest.pop(0)[...]
        if fuse_p:
            if passes == 3:
                Fp = (rest.pop(0)[...], rest.pop(0)[...])
            else:
                Fp = rest.pop(0)[...]
        ore_ref, oim_ref, zre, zim = rest
        i = pl.program_id(2)
        if fuse_p:
            @pl.when(i == 0)
            def _():
                ore_ref[...] = jnp.zeros_like(ore_ref)
                oim_ref[...] = jnp.zeros_like(oim_ref)

        def one_child(c, carry):
            fcos, fsin = fc_ref[c], fs_ref[c]                                   # [pb, chunk]
            ccos_all, csin_all = cc_ref[c], cs_ref[c]                           # [pb, nchl]
            for m in range(nch):
                ccos, csin = ccos_all[:, m:m + 1], csin_all[:, m:m + 1]
                cs = ccos * fcos - csin * fsin
                sn = csin * fcos + ccos * fsin
                lo = m * chunk
                if lo + chunk <= stride:
                    sr, si = sa_ref[:, lo:lo + chunk], ta_ref[:, lo:lo + chunk]
                else:
                    sr, si = sb_ref[:, lo - stride:lo - stride + chunk], tb_ref[:, lo - stride:lo - stride + chunk]
                zre[:, lo:lo + chunk] = sr * cs - si * sn
                zim[:, lo:lo + chunk] = sr * sn + si * cs
            yr = mm(zre[...], F)                                                # [pb, kob]
            yi = mm(zim[...], F)
            if fuse_p:
                ore_ref[c] += mm(Fp, yr)                                        # [Po_pad, kob], accumulated over the pulse blocks
                oim_ref[c] += mm(Fp, yi)
            else:
                ore_ref[c] = yr
                oim_ref[c] = yi
            return carry

        lax.fori_loop(0, nc, one_child, 0)

    grid = (N, nb, P // pb)
    blk = lambda n, b, i: (n, i, b)
    halo = (lambda n, b, i: (n, i, b + 1)) if nb > 1 else blk
    in_specs = [pl.BlockSpec((None, pb, stride), blk), pl.BlockSpec((None, pb, stride), halo),
                pl.BlockSpec((None, pb, stride), blk), pl.BlockSpec((None, pb, stride), halo),
                pl.BlockSpec((None, None, nc, pb, nchl), lambda n, b, i: (n, b, 0, i, 0)), pl.BlockSpec((None, None, nc, pb, nchl), lambda n, b, i: (n, b, 0, i, 0)),
                pl.BlockSpec((None, nc, pb, chunk), lambda n, b, i: (n, 0, i, 0)), pl.BlockSpec((None, nc, pb, chunk), lambda n, b, i: (n, 0, i, 0))]
    args = [S_re, S_re, S_im, S_im, ccos, csin, fcos, fsin]
    if passes == 3:
        in_specs += [pl.BlockSpec((None, win, kob), lambda n, b, i: (b, 0, 0))] * 2
        args += [fb_hi, fb_lo]
    else:
        in_specs += [pl.BlockSpec((None, win, kob), lambda n, b, i: (b, 0, 0))]
        args += [fb]
    if fuse_p:
        if passes == 3:
            in_specs += [pl.BlockSpec((Po_pad, pb), lambda n, b, i: (0, i))] * 2
            args += [fp_hi, fp_lo]
        else:
            in_specs += [pl.BlockSpec((Po_pad, pb), lambda n, b, i: (0, i))]
            args += [fp]
        out_specs = [pl.BlockSpec((None, nc, Po_pad, kob), lambda n, b, i: (n, 0, 0, b))] * 2
        out_shape = [jax.ShapeDtypeStruct((N, nc, Po_pad, Ko_pad), jnp.float32)] * 2
    else:
        out_specs = [pl.BlockSpec((None, nc, pb, kob), lambda n, b, i: (n, 0, i, b))] * 2
        out_shape = [jax.ShapeDtypeStruct((N, nc, P, Ko_pad), jnp.float32)] * 2
    call = pl.pallas_call(kernel, grid=grid, in_specs=in_specs, out_specs=out_specs, out_shape=out_shape,
                          scratch_shapes=[pltpu.VMEM((pb, win), jnp.float32)] * 2,
                          compiler_params=None if interpret else pltpu.CompilerParams(vmem_limit_bytes=vmem_limit, dimension_semantics=('arbitrary', 'arbitrary', 'arbitrary')),
                          interpret=interpret)
    return call(*args)


def fused_final3(dT_re, dT_im, gx, gy, a0, a1, Qf, passes=1, tiles=4, interpret=False, vmem_limit=96 << 20):
    """fused_final2 with `tiles` tiles per grid step: their [2T, n] ramp matrices are stacked into one [2T tiles, n]
    operand on each side, so the matrix units see a 256 by 256 product instead of a 64 by 64 one (the off-diagonal
    blocks, products between different tiles, are computed and discarded). B must be a multiple of `tiles`."""
    B, Qpad, Pl = dT_re.shape
    T = gx.shape[1]
    assert Pl % LANE == 0 and gx.shape == (B, T, Pl) and Qf <= Qpad and B % tiles == 0, (dT_re.shape, gx.shape, Qf, B, tiles)
    n = Qf * Pl
    R = 2 * T * tiles

    def nt(a, b):
        if passes == 3:
            a_hi, a_lo = _split_bf16(a)
            b_hi, b_lo = _split_bf16(b)
            return (lax.dot_general(a_hi, b_hi, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                    + lax.dot_general(a_hi, b_lo, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                    + lax.dot_general(a_lo, b_hi, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32))
        return lax.dot_general(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    def table(g, k):
        c = -(g * k)
        ang = (c - jnp.floor(c + 0.5)) * TWO_PI
        return jnp.cos(ang), jnp.sin(ang)

    def kernel(dr_ref, di_ref, gx_ref, gy_ref, ore_ref, oim_ref, a_s, b_s):
        # rows of a_s / b_s: tile t occupies [2T t, 2T t + T) for the real part and [2T t + T, 2T (t + 1)) for the imaginary
        tabs = []
        for t in range(tiles):
            gxv, gyv = gx_ref[t], gy_ref[t]                                 # [T, Pl]
            xr0, xi0 = table(gxv, a0)
            wr, wi = table(gxv, a1)
            yr0, yi0 = table(gyv, a0)
            vr, vi = table(gyv, a1)
            tabs.append((xr0, xi0, wr, wi, yr0, yi0, vr, vi))

        def step(q, carry):
            new = []
            off = pl.multiple_of(q * Pl, LANE)
            for t in range(tiles):
                xr, xi, yr, yi = carry[t]
                wr, wi, vr, vi = tabs[t][2], tabs[t][3], tabs[t][6], tabs[t][7]
                dr = dr_ref[t, pl.ds(q, 1), :]                               # [1, Pl]
                di = di_ref[t, pl.ds(q, 1), :]
                r0 = 2 * T * t
                a_s[r0:r0 + T, pl.ds(off, Pl)] = dr * xr - di * xi
                a_s[r0 + T:r0 + 2 * T, pl.ds(off, Pl)] = dr * xi + di * xr
                b_s[r0:r0 + T, pl.ds(off, Pl)] = yr
                b_s[r0 + T:r0 + 2 * T, pl.ds(off, Pl)] = yi
                new.append((xr * wr - xi * wi, xr * wi + xi * wr, yr * vr - yi * vi, yr * vi + yi * vr))
            return tuple(new)

        lax.fori_loop(0, Qf, step, tuple((tb[0], tb[1], tb[4], tb[5]) for tb in tabs))
        M = nt(a_s[:, :n], b_s[:, :n])                                       # [R, R]
        for t in range(tiles):
            r0 = 2 * T * t
            ore_ref[t] = M[r0:r0 + T, r0:r0 + T] - M[r0 + T:r0 + 2 * T, r0 + T:r0 + 2 * T]
            oim_ref[t] = M[r0:r0 + T, r0 + T:r0 + 2 * T] + M[r0 + T:r0 + 2 * T, r0:r0 + T]

    in_specs = [pl.BlockSpec((tiles, Qpad, Pl), lambda b: (b, 0, 0)), pl.BlockSpec((tiles, Qpad, Pl), lambda b: (b, 0, 0)),
                pl.BlockSpec((tiles, T, Pl), lambda b: (b, 0, 0)), pl.BlockSpec((tiles, T, Pl), lambda b: (b, 0, 0))]
    out_specs = [pl.BlockSpec((tiles, T, T), lambda b: (b, 0, 0))] * 2
    call = pl.pallas_call(kernel, grid=(B // tiles,), in_specs=in_specs, out_specs=out_specs,
                          out_shape=[jax.ShapeDtypeStruct((B, T, T), jnp.float32)] * 2,
                          scratch_shapes=[pltpu.VMEM((R, Qpad * Pl), jnp.float32)] * 2,
                          compiler_params=None if interpret else pltpu.CompilerParams(vmem_limit_bytes=vmem_limit),
                          interpret=interpret)
    return call(dT_re, dT_im, gx, gy)
