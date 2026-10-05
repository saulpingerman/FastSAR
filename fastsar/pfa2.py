"""Polar format for measured collections, second round.

Three steps, all expressed as FFTs, element-wise products and one matrix product so that they run on
every device:

  1. Pulse resampling. The pulses are resampled so that the tangent of the look angle in the ground plane
     is uniform in pulse index (an orbit is not quite a straight line). The resampling is a windowed-sinc
     interpolation of each frequency sample's pulse sequence at fractional pulse indices, applied as one
     [P', P] matrix product (dense) or as a sum over taps of gathered rows.
  2. Range resampling by chirp-z transform: each pulse's frequency samples are interpolated onto the
     uniform k_x grid of the rectangular wavenumber raster.
  3. Azimuth resampling by chirp-z transform: each k_x row is interpolated onto the uniform k_y grid.
  Then zero padding to the FFT sizes and a 2-D inverse FFT give the image on the native grid.

The planar-wavefront approximation of polar format is left in: the image is compared with exact
backprojection as it comes out, so the comparison includes the algorithm's own error.
"""
import math

import numpy as np
import jax
import jax.numpy as jnp

C = 299792458.0


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def _fft_size(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = None
    a = 0
    while 2 ** a <= 4 * n:
        b = 0
        while 2 ** a * 3 ** b <= 4 * n:
            c = 0
            while 2 ** a * 3 ** b * 5 ** c <= 4 * n:
                v = 2 ** a * 3 ** b * 5 ** c
                if v >= n and (best is None or v < best):
                    best = v
                c += 1
            b += 1
        a += 1
    return best


def geometry(col, nx, ny, spx, spy, taps=16, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), guard=300.0, kaiser=8.0):
    """Host-side float64 setup. The image axes are x along track (cross range) and y ground range, as in the
    prepared collections; polar format's range axis is y and its azimuth axis is x."""
    u = col.ant / np.linalg.norm(col.ant, axis=1)[:, None]
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    ur = -(u @ e2)                                  # range component toward the radar in the image plane (positive)
    t = (u @ e1) / ur                               # tangent of the look angle in the image plane
    P, K = col.Np, col.K
    p = np.arange(P)
    m = taps // 2
    rev = t[-1] < t[0]                              # pulses are taken in the order of increasing t
    if rev:
        t, ur = t[::-1], ur[::-1]
    assert np.all(np.diff(t) > 0), 'look angle not monotonic in pulse'
    # pulse grid uniform in t, inside the original aperture by m pulses
    t0, t1 = t[m], t[P - 1 - m]
    Pp = P
    tt = np.linspace(t0, t1, Pp)
    pidx = np.interp(tt, t, p)                      # fractional index of each resampled pulse in the ordered sequence
    dt = (t1 - t0) / (Pp - 1)
    ur_r = np.interp(pidx, p, ur)
    if rev:
        pidx = (P - 1) - pidx                       # back to the stored pulse order
    # wavenumber raster: spacing fixed by the native pixel spacing and the FFT sizes. The raster's period
    # exceeds the scene by a guard band on each side, and the range and azimuth profiles are tapered to zero
    # within the guard, so that echoes from beyond the image do not wrap into it.
    nfy, nfx = _fft_size(ny + int(2 * guard / spy)), _fft_size(nx + int(2 * guard / spx))
    dkr = 2.0 * math.pi / (nfy * spy)
    dka = 2.0 * math.pi / (nfx * spx)
    # The grazing angle changes along an orbital aperture, so the ground-plane radial wavenumber band of a
    # pulse, 4 pi f / c times its ground component, drifts from pulse to pulse. The raster covers the union of
    # the bands; samples a pulse does not have are zero, which keeps the full spectral support of
    # backprojection (a trapezoid) instead of the common rectangle, which here would be half the bandwidth.
    fa, fb = col.fmin, col.fmin + (K - 1) * col.df
    kr_min = 4.0 * math.pi * fa / C * ur_r.min()
    kr_max = 4.0 * math.pi * fb / C * ur_r.max()
    nkr = int((kr_max - kr_min) / dkr) + 1
    kr = kr_min + dkr * np.arange(nkr)
    ta, tb = t0 + m * dt, t1 - m * dt
    ka_lo, ka_hi = kr_max * ta, kr_max * tb
    nka = int((ka_hi - ka_lo) / dka) + 1
    scale = C / (4.0 * math.pi * ur_r * col.df)
    extent_r = C / (2.0 * col.df)                            # period of a range profile (m)
    extent_a = 2.0 * math.pi / (kr * dt)                     # period of the azimuth profile of each kr row (m)
    edge = (ny * spy / 2.0, nx * spx / 2.0)
    assert extent_r > 2 * (edge[0] + guard) and extent_a.min() > 2 * (edge[1] + guard), 'scene plus guard exceeds the unambiguous extent'
    return dict(pidx=pidx, taps=taps, kaiser=kaiser, Pp=Pp, K=K, extent_r=extent_r, extent_a=extent_a, edge=edge, guard=guard,
                alpha=kr_min * scale - col.fmin / col.df,        # [Pp] frequency index at kr_min
                beta=dkr * scale,                                # [Pp] frequency index step per kr sample
                gam=dka / (dt * kr),                             # [nkr] pulse index step per ka sample
                shift=-t0 / dt + ka_lo / (kr * dt),              # [nkr] pulse index at ka_lo
                nkr=nkr, nka=nka, nfy=nfy, nfx=nfx, dkr=dkr, dka=dka, kr_min=kr_min, ka_lo=ka_lo,
                oversample_r=float((kr_max - kr_min) / dkr / K), oversample_a=float(nka / Pp))


def distortion(col, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), n=21, deg=3):
    """Where polar format puts a scatterer. Its planar-wavefront model writes the range history of a scatterer at p
    as -u_p . q + c; the least-squares q and c over the aperture give the apparent position (q_x, q_y + c), since a
    constant range offset is a range shift. The mapping p -> p_app is fitted by 2-D polynomials of degree deg on an
    n x n grid over the image. Returns dict(cx, cy, powers, rms_px) with p_app = p + sum coef * x^a y^b."""
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    ant = col.ant.astype(np.float64)
    r0 = np.linalg.norm(ant, axis=1)
    u = ant / r0[:, None]
    U = np.stack([u @ e1, u @ e2], 1)
    A = np.concatenate([-U, np.ones((len(ant), 1))], 1)
    pinv = np.linalg.pinv(A)
    xs = np.linspace(-nx / 2.0, nx / 2.0, n) * spx
    ys = np.linspace(-ny / 2.0, ny / 2.0, n) * spy
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = X.ravel()[:, None] * e1 + Y.ravel()[:, None] * e2                                   # [n*n, 3]
    dR = np.linalg.norm(pts[:, None, :] - ant[None, :, :], axis=2) - r0[None, :]               # [n*n, P]
    sol = dR @ pinv.T                                                                           # [n*n, 3]
    xa, ya = sol[:, 0], sol[:, 1] + sol[:, 2]
    powers = [(a, b) for a in range(deg + 1) for b in range(deg + 1 - a)]
    sx, sy = nx / 2.0 * spx, ny / 2.0 * spy                                                     # normalised coordinates
    B = np.stack([(X.ravel() / sx) ** a * (Y.ravel() / sy) ** b for a, b in powers], 1)
    cx = np.linalg.lstsq(B, xa - X.ravel(), rcond=None)[0]
    cy = np.linalg.lstsq(B, ya - Y.ravel(), rcond=None)[0]
    rms = (float(np.sqrt(np.mean((B @ cx - (xa - X.ravel())) ** 2)) / spx), float(np.sqrt(np.mean((B @ cy - (ya - Y.ravel())) ** 2)) / spy))
    return dict(cx=cx, cy=cy, powers=powers, sx=sx, sy=sy, rms_px=rms,
                max_shift_px=(float(np.abs(xa - X.ravel()).max() / spx), float(np.abs(ya - Y.ravel()).max() / spy)))


def _kernel(d, taps, beta):
    """Kaiser-windowed sinc of `taps` taps at fractional offsets d."""
    half = taps / 2.0
    x = np.clip(d / half, -1.0, 1.0)
    w = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - x * x))) / np.i0(beta)
    return np.where(np.abs(d) < half, np.sinc(d) * w, 0.0)


def pulse_taps(geo, P):
    """Pulse resampling as a gather: indices [Pp, taps] and weights [Pp, taps]."""
    taps, pidx = geo['taps'], geo['pidx']
    i0 = np.floor(pidx).astype(int) - taps // 2 + 1
    idx = i0[:, None] + np.arange(taps)[None, :]
    w = _kernel(pidx[:, None] - idx, taps, geo['kaiser'])
    w = np.where((idx >= 0) & (idx < P), w, 0.0)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return np.clip(idx, 0, P - 1), w


def pulse_matrix(geo, P):
    """The same resampling as a dense [Pp, P] matrix."""
    idx, w = pulse_taps(geo, P)
    W = np.zeros((len(idx), P))
    np.add.at(W, (np.repeat(np.arange(len(idx)), idx.shape[1]), idx.ravel()), w.ravel())
    return W


def make_pfa(geo, nx, ny, spx, spy, resample='dense', mm_dtype=None, precision=None, orient=(1, -1), offset=(0.0, 0.0), rows=1024,
             dist=None, oversample=2, crop=None, taps=6):
    """pfa(S_re, S_im, Wp or (idx, w), alpha, eps_r, shift, eps_a) -> complex64 image [nx, ny] on the native grid.
    mm_dtype / precision apply to the pulse-resampling product (bfloat16 operands, or the device default
    precision for float32 operands); the chirp-z transforms and FFTs run in complex64.
    orient = (sy, sx): sign of the exponent along each image axis, exp(j sy kr y) and exp(j sx ka x); with the
    phase history referenced to the scene centre as prepared here, the two signs are opposite. Pixel i of an
    axis of n pixels sits at (i - n / 2) times the spacing, as in the backprojection grids.
    dist: a distortion model from `distortion`; the image is then formed on a grid oversampled by `oversample`,
    without its carriers, and resampled at the apparent position of every pixel by a 4-tap cubic kernel along
    each axis, with the carriers restored at the apparent position. crop = (i0, j0, h, w) limits the corrected
    output to a region of the native grid. taps: taps per axis of the Lanczos resampling kernel."""
    f = jnp.float32
    two_pi = 2.0 * math.pi
    nkr, nka, nfy, nfx = geo['nkr'], geo['nka'], geo['nfy'], geo['nfx']
    sy, sx = orient
    oy, ox = nfy // 2 - ny // 2, nfx // 2 - nx // 2
    dy, dx = (ny // 2 - ny / 2.0) * spy + offset[1], (nx // 2 - nx / 2.0) * spx + offset[0]   # half-pixel offsets of odd grids
    yy = (np.arange(nfy) - nfy // 2) * spy + dy
    xx = (np.arange(nfx) - nfx // 2) * spx + dx
    ramp_y = np.exp(1j * sy * geo['kr_min'] * yy).astype(np.complex64)
    ramp_x = np.exp(1j * sx * geo['ka_lo'] * xx).astype(np.complex64)
    shift_y = np.exp(1j * sy * geo['dkr'] * np.arange(nkr) * dy).astype(np.complex64)
    shift_x = np.exp(1j * sx * geo['dka'] * np.arange(nka) * dx).astype(np.complex64)
    t_y = jnp.fft.ifft if sy > 0 else jnp.fft.fft
    t_x = jnp.fft.ifft if sx > 0 else jnp.fft.fft

    def cx(re, im):
        return jax.lax.complex(re, im)

    def cis(cyc):
        ang = (cyc - jnp.round(cyc)) * two_pi
        return cx(jnp.cos(ang), jnp.sin(ang))

    def chirp(idx, eps, n, sign):
        i2 = idx * idx
        base = (i2 % (2 * n)).astype(f) / (2 * n)
        extra = eps[:, None] * (i2.astype(f) / (2 * n))
        return cis(sign * (base[None, :] + extra))

    def taper(n, extent, edge):
        """Window over a profile of n samples whose period is `extent` metres (per row, [R]): one inside
        `edge` metres of the centre, a raised cosine to zero over the next `guard` metres."""
        d = jnp.abs(jnp.arange(n, dtype=jnp.int32) - n // 2).astype(f)[None, :] * (extent[:, None] / n)
        gd = geo['guard']
        return jnp.where(d <= edge, 1.0, jnp.where(d >= edge + gd, 0.0, 0.5 + 0.5 * jnp.cos((d - edge) * (math.pi / gd)))).astype(f)

    def czt(x, shift, eps, n_out, j0, extent, edge):
        """The chirp-z transforms in blocks of rows, which bounds the working memory."""
        R = x.shape[0]
        if R <= rows:
            return czt_block(x, shift, eps, n_out, j0, extent, edge)
        return jnp.concatenate([czt_block(x[r:r + rows], shift[r:r + rows], eps[r:r + rows], n_out, j0, extent[r:r + rows], edge) for r in range(0, R, rows)], axis=0)

    def czt_block(x, shift, eps, n_out, j0, extent, edge):
        """Rows of x [R, n] are spectra; their trigonometric interpolation at index shift + (j0 + j)(1 + eps).
        Outputs whose index falls outside the row (where the periodic interpolation would wrap) are zero."""
        n = x.shape[1]
        L = _next_pow2(n + n_out)
        r = jnp.arange(n, dtype=jnp.int32) - n // 2
        g = jnp.fft.fftshift(jnp.fft.ifft(x, axis=1), axes=1) * taper(n, extent, edge)
        pre = g * cis(-(shift[:, None] * r.astype(f)[None, :]) / n) * chirp(r, eps, n, -1.0)
        xin = jnp.concatenate([pre[:, n // 2:], jnp.zeros((x.shape[0], L - n), pre.dtype), pre[:, :n // 2]], axis=1)
        hi = j0 + n_out - 1 + n // 2
        i = jnp.arange(L, dtype=jnp.int32)
        y = jnp.fft.ifft(jnp.fft.fft(xin, axis=1) * jnp.fft.fft(chirp(jnp.where(i <= hi, i, i - L), eps, n, 1.0), axis=1), axis=1)
        j = jnp.arange(n_out, dtype=jnp.int32) + j0
        y = y[:, :n_out] if j0 == 0 else jnp.concatenate([y[:, L + j0:], y[:, :n_out + j0]], axis=1)
        pos = shift[:, None] + j.astype(f)[None, :] * (1.0 + eps[:, None])
        return y * chirp(j, eps, n, -1.0) * ((pos > -0.5) & (pos < n - 0.5))

    def resample_pulses(S_re, S_im, W):
        if resample == 'dense':
            Wd = W if mm_dtype is None else W.astype(mm_dtype)
            a = S_re if mm_dtype is None else S_re.astype(mm_dtype)
            b = S_im if mm_dtype is None else S_im.astype(mm_dtype)
            return (jnp.matmul(Wd, a, precision=precision, preferred_element_type=f),
                    jnp.matmul(Wd, b, precision=precision, preferred_element_type=f))
        idx, w = W
        re = sum(w[:, r:r + 1] * jnp.take(S_re, idx[:, r], axis=0) for r in range(idx.shape[1]))
        im = sum(w[:, r:r + 1] * jnp.take(S_im, idx[:, r], axis=0) for r in range(idx.shape[1]))
        return re, im

    ry, rx = jnp.asarray(ramp_y), jnp.asarray(ramp_x)
    shy, shx = jnp.asarray(shift_y), jnp.asarray(shift_x)

    ext_r = jnp.full((geo['Pp'],), geo['extent_r'], f)
    ext_a = jnp.asarray(geo['extent_a'].astype(np.float32))
    edge_y, edge_x = geo['edge']

    # the stages are separate compiled calls so that the temporaries of one are freed before the next (the
    # whole chain in one call needs 17 GB of temporaries for the Panama scene, more than a v5e holds)
    # Each block of rows of a chirp-z transform is its own compiled call: inside one call the compiler
    # schedules the blocks' transforms together and their temporaries add up (16.5 GB for Panama).
    czt_a = jax.jit(lambda x, sh, ep, ex: czt_block(x, sh, ep, nka, 0, ex, edge_x))

    @jax.jit
    def stage1_block(S_re, S_im, Wb, sh, ep, ex):
        """Resampling and range chirp-z of one block of pulses; the block is the unit of working memory."""
        return czt_block(cx(*resample_pulses(S_re, S_im, Wb)), sh, ep, nkr, 0, ex, edge_y)

    def czt_calls(fn, x, shift, eps, extent):
        R = x.shape[0]
        return jnp.concatenate([fn(x[r:r + rows], shift[r:r + rows], eps[r:r + rows], extent[r:r + rows]) for r in range(0, R, rows)], axis=0)

    def stage1(S_re, S_im, W, alpha, eps_r):
        Pp = alpha.shape[0]
        parts = []
        for r in range(0, Pp, rows):
            Wb = W[r:r + rows] if resample == 'dense' else (W[0][r:r + rows], W[1][r:r + rows])
            parts.append(stage1_block(S_re, S_im, Wb, alpha[r:r + rows], eps_r[r:r + rows], ext_r[r:r + rows]))
        return jnp.concatenate(parts, axis=0).T                                               # [nkr, Pp]

    def stage2(aT, shift, eps_a):
        return czt_calls(czt_a, aT, shift, eps_a, ext_a)                                      # [nkr, nka]

    def spectrum(S_re, S_im, W, alpha, eps_r, shift, eps_a):
        return stage2(stage1(S_re, S_im, W, alpha, eps_r), shift, eps_a)

    @jax.jit
    def image(b):
        b = b * shy[:, None] * shx[None, :]
        G = jnp.pad(b, ((0, nfy - nkr), (0, nfx - nka)))
        img = jnp.fft.fftshift(t_x(t_y(G, axis=0), axis=1))                                   # [nfy (range), nfx (azimuth)]
        img = img * ry[:, None] * rx[None, :]
        return img[oy:oy + ny, ox:ox + nx].T                                                 # -> [nx, ny]

    def pfa(S_re, S_im, W, alpha, eps_r, shift, eps_a):
        return image(spectrum(S_re, S_im, W, alpha, eps_r, shift, eps_a))

    if dist is None:
        return pfa

    # ---- corrected form: oversampled baseband image, then resampling at the apparent positions. The range
    # transform is done once; the azimuth transform and the resampling proceed in blocks of output range
    # columns, so that the oversampled image never exists whole (it would be 4.8 GB for the Panama scene).
    ov = oversample
    Ny2, Nx2 = ov * nfy, ov * nfx
    i0, j0, h, w = crop if crop else (0, 0, nx, ny)
    half = taps // 2
    offs = list(range(-half + 1, half + 1))                    # tap offsets from floor(t)
    kr_c = geo['kr_min'] + (nkr // 2) * geo['dkr']           # the spectrum is centred in the oversampled raster
    ka_c = geo['ka_lo'] + (nka // 2) * geo['dka']
    xs = (np.arange(i0, i0 + h) - nx / 2.0) * spx + offset[0]                                  # true positions of the output pixels
    bw = 512 if w > 512 else w                                  # output range columns per block
    span = ov * bw + 2 * (int(math.ceil(dist['max_shift_px'][1])) * ov + taps + 8)              # base rows per block
    span = min(span, Ny2)
    norm = float(ov) ** ((sy > 0) + (sx > 0))                   # the inverse transforms divide by the larger sizes

    def block_tables(jb):
        """Host-side float64 tables of one block: base row start, integer indices, fractions and carriers."""
        ys = (np.arange(jb, min(jb + bw, j0 + w)) - ny / 2.0) * spy + offset[1]
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        Bp = np.stack([(X / dist['sx']) ** a * (Y / dist['sy']) ** b for a, b in dist['powers']], -1)
        Xa = X + Bp @ dist['cx']
        Ya = Y + Bp @ dist['cy']
        fx_idx = Xa / (spx / ov) + Nx2 / 2.0
        fy_idx = Ya / (spy / ov) + Ny2 / 2.0
        ix, iy = np.floor(fx_idx), np.floor(fy_idx)
        rb0 = int(np.clip(np.floor(iy.min()) - half - 2, 0, Ny2 - span))
        car = np.exp(1j * (sy * kr_c * Ya + sx * ka_c * Xa)).astype(np.complex64)
        pad = bw - len(ys)
        out = [ix.astype(np.int32), (iy - rb0).astype(np.int32), (fx_idx - ix).astype(np.float32), (fy_idx - iy).astype(np.float32), car]
        if pad:                                                 # the last block is padded to the block shape
            out = [np.pad(v, ((0, 0), (0, pad))) for v in out]
        return rb0, out, len(ys)

    def weights(t):
        """Lanczos kernel of `taps` taps (sinc times a sinc window) at the offsets, normalised to unit sum."""
        ws = []
        for o in offs:
            d = t - o
            ws.append(jnp.sinc(d) * jnp.sinc(d / half))
        tot = sum(ws)
        return [w_ / tot for w_ in ws]

    spectrum_jit = spectrum

    @jax.jit
    def range_stage(b):
        G = jnp.roll(jnp.pad(b, ((0, Ny2 - nkr), (0, 0))), -(nkr // 2), axis=0)
        return jnp.fft.fftshift(t_y(G, axis=0), axes=0)                                       # [Ny2, nka], range done

    @jax.jit
    def azimuth_warp(F1, rb0, ix, iyl, tx, ty, carr):
        rows = jax.lax.dynamic_slice_in_dim(F1, rb0, span, axis=0)                              # [span, nka]
        G = jnp.roll(jnp.pad(rows, ((0, 0), (0, Nx2 - nka))), -(nka // 2), axis=1)
        base = jnp.fft.fftshift(t_x(G, axis=1), axes=1).reshape(-1)                            # [span * Nx2] baseband block
        wx = weights(tx)
        wy = weights(ty)
        out = jnp.zeros(ix.shape, base.dtype)
        for a, oa in enumerate(offs):
            yy = jnp.clip(iyl + oa, 0, span - 1)
            for b_, ob in enumerate(offs):
                xx = jnp.clip(ix + ob, 0, Nx2 - 1)
                out = out + (wy[a] * wx[b_]) * jnp.take(base, yy * Nx2 + xx)
        return out * carr * norm

    tables = [block_tables(jb) for jb in range(j0, j0 + w, bw)]

    def pfa_corrected(S_re, S_im, W, alpha, eps_r, shift, eps_a):
        b = spectrum_jit(S_re, S_im, W, alpha, eps_r, shift, eps_a)                             # [nkr, nka]
        F1 = range_stage(b)
        del b
        parts = []
        for rb0, arrs, n in tables:
            parts.append(azimuth_warp(F1, rb0, *arrs)[:, :n])
        return jnp.concatenate(parts, axis=1)                                                 # [h, w] = [azimuth, range]

    return pfa_corrected

    return pfa


def arrays(geo, P, resample='dense'):
    f = np.float32
    W = pulse_matrix(geo, P).astype(f) if resample == 'dense' else tuple(np.asarray(v) for v in pulse_taps(geo, P))
    return (jnp.asarray(W) if resample == 'dense' else (jnp.asarray(W[0].astype(np.int32)), jnp.asarray(W[1].astype(f))),
            jnp.asarray(geo['alpha'].astype(f)), jnp.asarray((geo['beta'] - 1.0).astype(f)),
            jnp.asarray(geo['shift'].astype(f)), jnp.asarray((geo['gam'] - 1.0).astype(f)))
