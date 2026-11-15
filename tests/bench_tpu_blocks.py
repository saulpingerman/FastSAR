"""Device times of the building blocks a polar format on a JAX device can be made of, as JSON.

    python tests/bench_tpu_blocks.py --out tpu_blocks.json
    python tests/bench_tpu_blocks.py --only fft,transfer --sizes 2048,4096 --repeat 5 --out part.json

Written for a TPU VM: it needs jax (jax[tpu]) and numpy, nothing else, for every group but the last. Each block is
compiled, run once, then timed `--repeat` times with its result brought to completion (a value of every array of
it comes to the host before the clock stops); the JSON holds
the least and the median time, the compile time, and the block's error against a float64 NumPy evaluation of a few
of its outputs (a TPU's default matrix products and transforms are not all float32-exact, so a fast block that is
wrong shows here). The file is rewritten after every block, so a run that is stopped keeps what it measured.

Groups (--only):

  fft       2-D and batched 1-D transforms of complex64 arrays, the same from and to two float32 planes, the real
            transform of one plane, and a row transform written as matrix products with the DFT matrix's two planes.
  resample  Rows of a complex array (two float32 planes) read at pos0 + j step with a 16-tap kernel, as polar
            format's range and azimuth stages need: 'gather' (an indexed load per tap), 'window_select' (a window
            row per block of outputs from a table of windows, the taps moved to its front by selects, the weights a
            polynomial in the phase: fastsar/resample_jax.py's blocks path), 'window_dense' (the same window read,
            then a matrix product of each block with weights formed on the device from two sines per output),
            'czt' (a chirp-z transform of each row: three transforms of twice the length), and 'matmul' (one dense
            matrix for all rows, which fits a stage whose positions are the same for every row).
  warp      The final resampling of a baseband image oversampled by 2 at smoothly displaced positions, as two
            separable 1-D passes with a 6-tap kernel (rows, then columns), by the same three reads.
  transfer  Host to device of a cut history (complex64, and as two float32 planes; with a sum over it on the device,
            and that sum alone beside it) and device to host of a picture (uint8, float32, complex64 and its two
            planes), from a fresh device array every time.
  detect    The amplitude of a complex image from two planes and its 8-bit remap by a power law.
  cpu       This package's C++ polar format (fastsar.quick._pfa) on the machine's own cores for a history and a
            picture of the same sizes, corrected and uncorrected: what a device has to beat. Needs this package and
            a C++ compiler; skipped with a note where they are missing.

Sizes (--sizes): N for an N by N picture, or AxB; the default is 1024, 2048, 4096, 8192 and 12288x9216.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

STRIDE = 8          # samples between the starts of the windows in the table of windows
PHASES = 2048       # fractional positions the weights are rounded to, as the host kernels' tables


# ------------------------------------------------------------------------------------------------- host helpers

def kaiser_sinc(d, taps, beta=8.0):
    half = taps / 2.0
    x = np.clip(d / half, -1.0, 1.0)
    w = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - x * x))) / np.i0(beta)
    return np.where(np.abs(d) < half, np.sinc(d) * w, 0.0)


def lanczos(d, taps):
    half = taps / 2.0
    return np.where(np.abs(d) < half, np.sinc(d) * np.sinc(d / half), 0.0)


def kernel_table(taps):
    """[PHASES + 1, taps]: the taps' weights at the fractional position q / PHASES, tap t at the offset
    q / PHASES - (t - taps / 2 + 1), normalised to unit sum (the Kaiser-windowed sinc for 16 taps, Lanczos for 6)."""
    d = np.arange(PHASES + 1)[:, None] / PHASES - (np.arange(taps)[None, :] - taps // 2 + 1)
    w = kaiser_sinc(d, taps) if taps >= 12 else lanczos(d, taps)
    return w / w.sum(1, keepdims=True)


def poly_coef(tab):
    """A kernel table as Horner coefficients [degree + 1, taps] (float32, highest power first) in t = 2 f - 1."""
    from numpy.polynomial import polynomial as npoly
    x = np.linspace(-1.0, 1.0, tab.shape[0])
    for deg in (3, 5, 7, 9, 11, 13, 15):
        c = npoly.polyfit(x, tab, deg)
        if np.abs(npoly.polyval(x, c).T - tab).max() < 3e-7:
            break
    return np.ascontiguousarray(c[::-1], np.float32)


def reference_rows(x, pos, taps, rows):
    """float64: rows `rows` of x [R, n] (complex) at the positions pos [len(rows), m], with the table's weights."""
    tab = kernel_table(taps)
    out = np.zeros((len(rows), pos.shape[1]), np.complex128)
    for a, r in enumerate(rows):
        fl = np.floor(pos[a])
        q = np.rint((pos[a] - fl) * PHASES).astype(int)
        i0 = fl.astype(int) - (taps // 2 - 1)
        for t in range(taps):
            out[a] += tab[q, t] * x[r, np.clip(i0 + t, 0, x.shape[1] - 1)]
    return out


def rel_err(a, b):
    b = np.asarray(b)
    return float(np.sqrt(np.sum(np.abs(np.asarray(a) - b) ** 2) / max(np.sum(np.abs(b) ** 2), 1e-300)))


def block_positions(pos0, step, nblk, B):
    """Positions pos0[r] + j step[r] (float64 [R]) by blocks of B outputs: the integer part at each block's first
    sample (int32 [R, nblk]), its remainder and the step (float32), so that the device forms a block's positions
    in float32 without losing the fraction of a position tens of thousands of samples along a row."""
    p0 = pos0[:, None] + (np.arange(nblk) * B)[None, :] * step[:, None]
    base = np.floor(p0)
    return base.astype(np.int32), (p0 - base).astype(np.float32), np.broadcast_to(step[:, None], p0.shape).astype(np.float32)


# ------------------------------------------------------------------------------------------------- device blocks

def make_blocks(jax, jnp):
    from jax import lax
    f32, i32 = jnp.float32, jnp.int32

    def positions(base, lo, s, B):
        f = lo[..., None] + jnp.arange(B, dtype=f32) * s[..., None]
        fl = jnp.floor(f)
        return base[..., None] + fl.astype(i32), f - fl                      # the sample below, the fraction

    def horner(frac, coef):
        t = jnp.rint(frac * PHASES) * (2.0 / PHASES) - 1.0
        w = jnp.broadcast_to(coef[0], t.shape + (coef.shape[1],))
        for c in coef[1:]:
            w = w * t[..., None] + c
        return w

    def gather(xr, xi, base, lo, s, coef, *, B):
        """An indexed load per tap."""
        taps = coef.shape[1]
        R, n = xr.shape
        ib, fr = positions(base, lo, s, B)
        w = horner(fr, coef).reshape(R, -1, taps)
        idx = jnp.clip((ib - (taps // 2 - 1)).reshape(R, -1, 1) + jnp.arange(taps, dtype=i32), 0, n - 1)
        take = lambda p: (jnp.take_along_axis(p[:, None, :], idx, axis=2, mode='promise_in_bounds') * w).sum(-1)
        return take(xr), take(xi)

    def windows(u, w0, Rw):
        """u [R, N] (N a multiple of STRIDE), w0 [R, nblk] (multiples of STRIDE) -> u[r, w0 : w0 + Rw] [R, nblk, Rw],
        each read as one row of the table of the windows that start at every STRIDE-th sample."""
        R, N = u.shape
        c = u.reshape(R, N // STRIDE, STRIDE)
        nrow = (N - Rw) // STRIDE + 1
        tab = jnp.concatenate([c[:, i:nrow + i] for i in range(Rw // STRIDE)], -1).reshape(R * nrow, Rw)
        return tab.at[jnp.arange(R, dtype=i32)[:, None] * nrow + w0 // STRIDE].get(mode='promise_in_bounds')

    def front(g, rel, W, taps):
        """g [..., 1, Rw], rel [..., B, 1] (0 <= rel < W) -> g[..., rel : rel + taps], one select per bit of rel."""
        bit = 1 << max(W - 1, 1).bit_length() - 1
        while bit:
            nw = bit + taps - 1
            lo, hi = g[..., :nw], g[..., bit:bit + nw]
            lo, hi = (jnp.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, nw - v.shape[-1])]) if v.shape[-1] < nw else v
                      for v in (lo, hi))
            g = jnp.where(rel & bit != 0, hi, lo)
            bit >>= 1
        return g

    def window_of(base, lo, s, taps, B, n, Rw):
        """The window of each block (its start, a multiple of STRIDE at or below the block's first tap) and the
        offsets of its samples' first taps in it."""
        ib, fr = positions(base, lo, s, B)
        i0 = ib - (taps // 2 - 1)
        w0 = jnp.clip((i0[..., 0] // STRIDE) * STRIDE, 0, n - Rw)
        return w0, i0 - w0[..., None], fr

    def window_select(xr, xi, base, lo, s, coef, *, B, Rw):
        """A window row per block, the taps moved to its front by selects, polynomial weights."""
        taps = coef.shape[1]
        R, n = xr.shape
        w0, rel, fr = window_of(base, lo, s, taps, B, n, Rw)
        W = Rw - taps + 1
        rel = jnp.clip(rel, 0, W - 1)[..., None]
        w = horner(fr, coef)
        f = lambda p: (front(windows(p, w0, Rw)[:, :, None, :], rel, W, taps) * w).sum(-1).reshape(R, -1)
        return f(xr), f(xi)

    def window_dense(xr, xi, base, lo, s, *, B, Rw, taps):
        """A window row per block, then a matrix product of the block's B outputs with weights [B, Rw] formed on
        the device: the kernel sinc(d) sinc(d / h), h = taps / 2, from two sines and a cosine per output (the sine of
        pi (u - w) is that of pi u with the sign of w's parity, and the one at a h-th of the angle follows from the
        angle sum with constants of w)."""
        R, n = xr.shape
        h = taps // 2
        w0, rel, fr = window_of(base, lo, s, taps, B, n, Rw)
        u = (rel + (h - 1)).astype(f32) + fr                                 # the position in the window [R, nblk, B]
        wv = jnp.arange(Rw, dtype=f32)
        d = u[..., None] - wv                                                # [R, nblk, B, Rw]
        s1 = jnp.sin(math.pi * fr) * jnp.where((rel + (h - 1)) % 2 == 0, 1.0, -1.0)      # sin(pi u)
        par = jnp.where(jnp.arange(Rw) % 2 == 0, 1.0, -1.0).astype(f32)
        a = (math.pi / h) * u
        sh = jnp.sin(a)[..., None] * jnp.cos((math.pi / h) * wv) - jnp.cos(a)[..., None] * jnp.sin((math.pi / h) * wv)
        k = (s1[..., None] * par) * sh * (h / (math.pi * math.pi)) / jnp.where(jnp.abs(d) < 1e-4, 1.0, d * d)
        k = jnp.where(jnp.abs(d) < 1e-4, 1.0, jnp.where(jnp.abs(d) < h, k, 0.0))
        k = k / k.sum(-1, keepdims=True)
        f = lambda p: jnp.einsum('rbjw,rbw->rbj', k, windows(p, w0, Rw), precision=lax.Precision.HIGHEST).reshape(R, -1)
        return f(xr), f(xi)

    def cis(cyc):
        ang = (cyc - jnp.round(cyc)) * (2.0 * math.pi)
        return lax.complex(jnp.cos(ang), jnp.sin(ang))

    def chirp(idx, eps, n, sign):
        i2 = idx * idx
        return cis(sign * ((i2 % (2 * n)).astype(f32)[None, :] / (2 * n) + eps[:, None] * (i2.astype(f32) / (2 * n))))

    def czt(xr, xi, shift, eps, *, n_out, L):
        """Rows of x at shift + j (1 + eps) by the chirp-z transform of their spectra (band-limited interpolation;
        fastsar/pfa2.py's, without its taper): a transform of n and three of L >= n + n_out."""
        n = xr.shape[1]
        r = jnp.arange(n, dtype=i32) - n // 2
        X = jnp.fft.fftshift(jnp.fft.fft(lax.complex(xr, xi), axis=1), axes=1) / n
        pre = X * cis((shift[:, None] * r.astype(f32)[None, :]) / n) * chirp(r, eps, n, 1.0)
        xin = jnp.concatenate([pre[:, n // 2:], jnp.zeros((xr.shape[0], L - n), pre.dtype), pre[:, :n // 2]], axis=1)
        hi = n_out - 1 + n // 2
        i = jnp.arange(L, dtype=i32)
        y = jnp.fft.ifft(jnp.fft.fft(xin, axis=1) * jnp.fft.fft(chirp(jnp.where(i <= hi, i, i - L), eps, n, -1.0), axis=1), axis=1)
        y = y[:, :n_out] * chirp(jnp.arange(n_out, dtype=i32), eps, n, 1.0)
        return jnp.real(y), jnp.imag(y)

    def matmul(xr, xi, Wm, *, precision):
        """One dense matrix [m, n] for all rows: x [R, n] @ Wm.T, plane by plane."""
        p = lax.Precision.HIGHEST if precision == 'highest' else None
        return jnp.matmul(xr, Wm.T, precision=p), jnp.matmul(xi, Wm.T, precision=p)

    def dft_rows(xr, xi, Wr, Wi):
        """The transform along rows as products with the DFT matrix's planes (four real products)."""
        p = lax.Precision.HIGHEST
        mm = lambda a, b: jnp.matmul(a, b, precision=p)
        return mm(xr, Wr) - mm(xi, Wi), mm(xr, Wi) + mm(xi, Wr)

    return dict(gather=gather, window_select=window_select, window_dense=window_dense, czt=czt, matmul=matmul,
                dft_rows=dft_rows)


# ------------------------------------------------------------------------------------------------------ timing

def finish(res):
    """Bring a result to completion beyond doubt: its last value, of every array of it, comes to the host."""
    import jax
    for leaf in jax.tree_util.tree_leaves(jax.block_until_ready(res)):
        np.asarray(leaf.reshape(-1)[-1:])
    return res


class Bench:
    def __init__(self, out, repeat, meta, skip=()):
        self.out, self.repeat, self.doc, self.skip = out, repeat, dict(meta, results=[]), tuple(skip)

    def save(self):
        if self.out:
            tmp = self.out + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump(self.doc, fh, indent=1)
            os.replace(tmp, self.out)

    def record(self, group, block, size, **kw):
        row = dict(group=group, block=block, size=list(size), **kw)
        self.doc['results'].append(row)
        self.save()
        t = row.get('seconds_min')
        print(f"{group:9s} {block:26s} {'x'.join(str(v) for v in size):>12s} "
              + (f"{t:9.4f} s" if t is not None else f"  {row.get('skipped') or row.get('failed')}")
              + (f"  (median {row['seconds_median']:.4f}, compile {row['compile_seconds']:.1f} s)" if t is not None and 'compile_seconds' in row else '')
              + (f"  error {row['error']:.1e}" if row.get('error') is not None else ''), flush=True)

    def run(self, group, block, size, fn, args, check=None, **params):
        """fn(*args) on the device: compile and first run, then `repeat` timed runs; check(result) -> error."""
        if any(k in block for k in self.skip):
            return
        try:
            t = time.perf_counter()
            res = finish(fn(*args))
            first = time.perf_counter() - t
            ts = []
            for _ in range(self.repeat):
                t = time.perf_counter()
                res = finish(fn(*args))
                ts.append(time.perf_counter() - t)
            err = None if check is None else check(res)
            self.record(group, block, size, seconds_min=min(ts), seconds_median=float(np.median(ts)),
                        compile_seconds=first - min(ts), error=err, **params)
        except Exception as e:                                               # out of memory, an unsupported op
            self.record(group, block, size, failed=f'{type(e).__name__}: {str(e)[:300]}', **params)


def chunked(step, rows, args_of):
    """step over chunks of `rows` rows (one compiled shape; the last chunk is moved back to end at the last row),
    the pieces joined: what a stage does when its working arrays are several times its input."""
    import jax.numpy as jnp

    def run(R):
        starts = list(range(0, max(R - rows, 0) + 1, rows))
        if starts[-1] + rows < R:
            starts.append(R - rows)
        parts, done = [], 0
        for a in starts:
            o = step(*args_of(a, a + rows))
            parts.append(tuple(p[done - a:] for p in o))
            done = a + rows
        return tuple(jnp.concatenate([p[k] for p in parts], 0) for k in range(len(parts[0])))
    return run


# ------------------------------------------------------------------------------------------------------ groups

def group_fft(b, sizes):
    import jax
    import jax.numpy as jnp
    from jax import lax
    blocks = make_blocks(jax, jnp)
    for A, Bn in sizes:
        key = jax.random.PRNGKey(0)
        xr = jax.random.normal(key, (A, Bn), jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(1), (A, Bn), jnp.float32)
        x = lax.complex(xr, xi)
        xs = np.asarray(x[:4, :]).astype(np.complex128)
        ref_rows = np.fft.fft(xs, axis=1)

        def chk_rows(res):
            res = res if not isinstance(res, tuple) else lax.complex(res[0], res[1])
            return rel_err(np.asarray(res[:4]), ref_rows)
        b.run('fft', 'fft2_complex64', (A, Bn), jax.jit(jnp.fft.fft2), (x,))
        b.run('fft', 'fft_rows_complex64', (A, Bn), jax.jit(lambda v: jnp.fft.fft(v, axis=1)), (x,), chk_rows)
        b.run('fft', 'fft_cols_complex64', (A, Bn), jax.jit(lambda v: jnp.fft.fft(v, axis=0)), (x,))

        def planes2(re, im):
            y = jnp.fft.fft2(lax.complex(re, im))
            return jnp.real(y), jnp.imag(y)

        def planes_rows(re, im):
            y = jnp.fft.fft(lax.complex(re, im), axis=1)
            return jnp.real(y), jnp.imag(y)
        b.run('fft', 'fft2_two_planes', (A, Bn), jax.jit(planes2), (xr, xi))
        b.run('fft', 'fft_rows_two_planes', (A, Bn), jax.jit(planes_rows), (xr, xi), chk_rows)
        b.run('fft', 'rfft2_one_plane', (A, Bn), jax.jit(jnp.fft.rfft2), (xr,))
        b.run('fft', 'rfft_rows_one_plane', (A, Bn), jax.jit(lambda v: jnp.fft.rfft(v, axis=1)), (xr,))
        if Bn <= 8192:
            k = np.arange(Bn)
            ang = -2.0 * np.pi * ((k[:, None] * k[None, :]) % Bn) / Bn
            Wr, Wi = jnp.asarray(np.cos(ang), jnp.float32), jnp.asarray(np.sin(ang), jnp.float32)
            b.run('fft', 'dft_rows_matmul', (A, Bn), jax.jit(blocks['dft_rows']), (xr, xi, Wr, Wi), chk_rows)
            del Wr, Wi
        del x, xr, xi


def resample_case(A, Bn, taps, step_lo, step_hi, B, seed=0):
    """A banded resampling of A rows: m = Bn outputs per row at pos0 + j step, step from step_lo to step_hi along the
    rows, out of n source samples; the positions by blocks of B."""
    rng = np.random.default_rng(seed)
    m = -(-Bn // B) * B
    step = np.linspace(step_lo, step_hi, A)
    pos0 = rng.uniform(taps, taps + 4.0, A)
    n = -(-int(pos0.max() + m * max(step_lo, step_hi) + taps + 2 * STRIDE) // STRIDE) * STRIDE
    Rw = -(-int(B * max(step_lo, step_hi) + taps + STRIDE + 1) // STRIDE) * STRIDE
    return dict(m=m, n=n, Rw=Rw, B=B, taps=taps, pos0=pos0, step=step, nblk=m // B,
                blocks=block_positions(pos0, step, m // B, B))


def group_resample(b, sizes, taps=16, B=32):
    import jax
    import jax.numpy as jnp
    blocks = make_blocks(jax, jnp)
    coef = jnp.asarray(poly_coef(kernel_table(taps)))
    for A, Bn in sizes:
        c = resample_case(A, Bn, taps, 1.42, 1.5, B)
        n, m, Rw = c['n'], c['m'], c['Rw']
        xr = jax.random.normal(jax.random.PRNGKey(2), (A, n), jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(3), (A, n), jnp.float32)
        base, lo, s = (jnp.asarray(v) for v in c['blocks'])
        rows = [0, A // 2, A - 1]
        xs = (np.asarray(xr)[rows] + 1j * np.asarray(xi)[rows]).astype(np.complex128)
        pos = c['pos0'][rows, None] + np.arange(m)[None, :] * c['step'][rows, None]
        ref = reference_rows(xs, pos, taps, [0, 1, 2])

        def chk(res):
            return rel_err(np.asarray(res[0])[rows] + 1j * np.asarray(res[1])[rows], ref)
        par = dict(taps=taps, outputs_per_block=B, window=Rw, source=n, outputs=m)
        # a table of windows is Rw / STRIDE times its rows: the rows go through in chunks of one shape
        per = int(max(1, min(A, (1 << 27) // (n * (Rw // STRIDE)))))
        per_g = int(max(1, min(A, (1 << 27) // (m * taps))))                  # the indices: taps per output
        g = jax.jit(lambda *a: blocks['gather'](*a, B=B))
        b.run('resample', 'gather', (A, Bn),
              chunked(g, per_g, lambda a, e: (xr[a:e], xi[a:e], base[a:e], lo[a:e], s[a:e], coef)), (A,), chk,
              rows_per_chunk=per_g, **par)
        ws = jax.jit(lambda *a: blocks['window_select'](*a, B=B, Rw=Rw))
        b.run('resample', 'window_select', (A, Bn),
              chunked(ws, per, lambda a, e: (xr[a:e], xi[a:e], base[a:e], lo[a:e], s[a:e], coef)), (A,), chk,
              rows_per_chunk=per, **par)
        per_d = int(max(1, min(A, (1 << 27) // max(n * (Rw // STRIDE), m * Rw))))
        wd = jax.jit(lambda *a: blocks['window_dense'](*a, B=B, Rw=Rw, taps=taps))
        # its kernel is the sinc-windowed sinc, not the table's Kaiser: no error against the table is given
        b.run('resample', 'window_dense', (A, Bn),
              chunked(wd, per_d, lambda a, e: (xr[a:e], xi[a:e], base[a:e], lo[a:e], s[a:e])), (A,), None,
              rows_per_chunk=per_d, kernel='sinc times sinc', **par)
        L = 1 << int(math.ceil(math.log2(n + m)))
        per_c = int(max(1, min(A, (1 << 27) // (4 * L))))
        shift, eps = jnp.asarray(c['pos0'], jnp.float32), jnp.asarray(c['step'] - 1.0, jnp.float32)
        cz = jax.jit(lambda *a: blocks['czt'](*a, n_out=m, L=L))
        b.run('resample', 'czt', (A, Bn), chunked(cz, per_c, lambda a, e: (xr[a:e], xi[a:e], shift[a:e], eps[a:e])), (A,),
              None, rows_per_chunk=per_c, transform=L, kernel='band-limited (periodic)', **par)
        # one matrix for every row (the positions of row A / 2)
        tab = kernel_table(taps)
        p = c['pos0'][A // 2] + np.arange(m) * c['step'][A // 2]
        fl = np.floor(p).astype(int)
        Wm = np.zeros((m, n), np.float32)
        q = np.rint((p - fl) * PHASES).astype(int)
        for t in range(taps):
            Wm[np.arange(m), np.clip(fl - (taps // 2 - 1) + t, 0, n - 1)] += tab[q, t]
        Wd = jnp.asarray(Wm)
        refm = xs @ Wm.astype(np.float64).T

        def chkm(res):
            return rel_err(np.asarray(res[0])[rows] + 1j * np.asarray(res[1])[rows], refm)
        for prec in ('highest', 'default'):
            mmf = jax.jit(lambda a, c_, w, prec=prec: blocks['matmul'](a, c_, w, precision=prec))
            b.run('resample', f'matmul_{prec}', (A, Bn), mmf, (xr, xi, Wd), chkm, **par)
        del xr, xi, Wd


def group_warp(b, sizes, taps=6, B=32):
    """An image oversampled by 2, [2A, 2Bn], read at the picture's A by Bn pixels displaced by a smooth field of up
    to 20 pixels: rows first (to [2A, Bn]), then columns (to [A, Bn])."""
    import jax
    import jax.numpy as jnp
    blocks = make_blocks(jax, jnp)
    coef = jnp.asarray(poly_coef(kernel_table(taps)))
    for A, Bn in sizes:
        m1, m2 = -(-Bn // B) * B, -(-A // B) * B
        n1 = -(-(2 * m1 + 64 + 2 * STRIDE) // STRIDE) * STRIDE
        n2 = -(-(2 * m2 + 64 + 2 * STRIDE) // STRIDE) * STRIDE
        Rw = -(-int(B * 2.02 + taps + STRIDE + 1) // STRIDE) * STRIDE
        xr = jax.random.normal(jax.random.PRNGKey(4), (n2, n1), jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(5), (n2, n1), jnp.float32)
        # pass 1: row r of the oversampled image at 16 + 2 j (1 + 0.005 v) + 20 v^2, v its place across the rows
        v = np.linspace(-1.0, 1.0, n2)
        p1 = block_positions(16.0 + 20.0 * v * v, 2.0 * (1.0 + 0.005 * v), m1 // B, B)
        u = np.linspace(-1.0, 1.0, m1)
        p2 = block_positions(16.0 + 20.0 * u * u, 2.0 * (1.0 + 0.005 * u), m2 // B, B)
        P1, P2 = tuple(jnp.asarray(a) for a in p1), tuple(jnp.asarray(a) for a in p2)
        par = dict(taps=taps, outputs_per_block=B, window=Rw, source=[n2, n1], outputs=[m2, m1])

        def two_passes(row_pass, c1, c2):
            """Rows in chunks of c1, the result turned, its rows in chunks of c2, turned back."""
            def run(xr, xi):
                a = chunked(row_pass, c1, lambda s, e: (xr[s:e], xi[s:e], P1[0][s:e], P1[1][s:e], P1[2][s:e]))(n2)
                ar, ai = a[0].T, a[1].T
                o = chunked(row_pass, c2, lambda s, e: (ar[s:e], ai[s:e], P2[0][s:e], P2[1][s:e], P2[2][s:e]))(m1)
                return o[0].T, o[1].T
            return run
        chunk = lambda rows, per_row: int(max(1, min(rows, (1 << 27) // per_row)))
        g = jax.jit(lambda a, c_, p, l, s: blocks['gather'](a, c_, p, l, s, coef, B=B))
        c = [chunk(n2, m1 * taps), chunk(m1, m2 * taps)]
        b.run('warp', 'two_passes_gather', (A, Bn), two_passes(g, *c), (xr, xi), None, rows_per_chunk=c, **par)
        ws = jax.jit(lambda a, c_, p, l, s: blocks['window_select'](a, c_, p, l, s, coef, B=B, Rw=Rw))
        c = [chunk(n2, n1 * (Rw // STRIDE)), chunk(m1, n2 * (Rw // STRIDE))]
        b.run('warp', 'two_passes_window_select', (A, Bn), two_passes(ws, *c), (xr, xi), None, rows_per_chunk=c, **par)
        wd = jax.jit(lambda a, c_, p, l, s: blocks['window_dense'](a, c_, p, l, s, B=B, Rw=Rw, taps=taps))
        c = [chunk(n2, max(n1 * (Rw // STRIDE), m1 * Rw)), chunk(m1, max(n2 * (Rw // STRIDE), m2 * Rw))]
        b.run('warp', 'two_passes_window_dense', (A, Bn), two_passes(wd, *c), (xr, xi), None, rows_per_chunk=c,
              kernel='sinc times sinc', **par)
        del xr, xi


def group_transfer(b, sizes):
    """Host to device: the array put on the device and a sum over it brought back, so that every byte has arrived
    before the clock stops (the sum alone, of an array already there, is timed beside it). Device to host: a fresh
    device array for every repeat (JAX keeps the host copy of an array it has converted once), brought to
    completion first, then converted."""
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    for A, Bn in sizes:
        h = (rng.standard_normal((A, Bn), np.float32) + 1j * rng.standard_normal((A, Bn), np.float32)).astype(np.complex64)
        hr, hi = np.ascontiguousarray(h.real), np.ascontiguousarray(h.imag)
        total = jax.jit(lambda *p: sum(jnp.sum(jnp.abs(v)) for v in p))
        bump = jax.jit(lambda v, k: v + k)

        def timed(name, fn, nbytes, prepare=None, **kw):
            ts = []
            for k in range(b.repeat + 1):
                arg = prepare(k) if prepare else None
                t = time.perf_counter()
                fn(arg)
                ts.append(time.perf_counter() - t)
            ts = ts[1:]
            b.record('transfer', name, (A, Bn), seconds_min=min(ts), seconds_median=float(np.median(ts)), megabytes=nbytes / 1e6,
                     megabytes_per_second=nbytes / 1e6 / min(ts), **kw)
        try:
            d = finish(jax.device_put(h))
            dr, di = finish((jax.device_put(hr), jax.device_put(hi)))
            timed('sum_on_device_complex64', lambda _: float(total(d)), h.nbytes)
            timed('sum_on_device_two_planes', lambda _: float(total(dr, di)), h.nbytes)
            fresh = lambda a: (lambda k: a + np.float32(1e-3 * k))        # new bytes on the host for every repeat
            timed('to_device_and_sum_complex64', lambda a: float(total(jax.device_put(a))), h.nbytes, prepare=fresh(h))
            timed('to_device_and_sum_two_planes', lambda a: float(total(jax.device_put(a[0]), jax.device_put(a[1]))), h.nbytes,
                  prepare=lambda k: (hr + np.float32(1e-3 * k), hi + np.float32(1e-3 * k)))
            d8 = finish(jnp.zeros((A, Bn), jnp.uint8))
            timed('to_host_uint8', lambda a: np.asarray(a), A * Bn, prepare=lambda k: finish(bump(d8, np.uint8(k))))
            timed('to_host_float32', lambda a: np.asarray(a), 4 * A * Bn, prepare=lambda k: finish(bump(dr, np.float32(k))))
            timed('to_host_two_planes', lambda a: (np.asarray(a[0]), np.asarray(a[1])), 8 * A * Bn,
                  prepare=lambda k: (finish(bump(dr, np.float32(k))), finish(bump(di, np.float32(k)))))
            timed('to_host_complex64', lambda a: np.asarray(a), 8 * A * Bn, prepare=lambda k: finish(bump(d, np.float32(k))))
            del d, d8, dr, di
        except Exception as e:
            b.record('transfer', 'transfer', (A, Bn), failed=f'{type(e).__name__}: {str(e)[:300]}')


def group_detect(b, sizes):
    """The amplitude of a complex image and its 8-bit remap; a factor that changes with every call makes each
    result new."""
    import jax
    import jax.numpy as jnp
    for A, Bn in sizes:
        xr = jax.random.normal(jax.random.PRNGKey(6), (A, Bn), jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(7), (A, Bn), jnp.float32)
        calls = [0]

        def fresh(fn):
            def run(a, c):
                calls[0] += 1
                return fn(a, c, np.float32(1.0 + 1e-3 * calls[0]))
            return run
        amp = jax.jit(lambda a, c, k: jnp.sqrt(a * a + c * c) * k)
        b.run('detect', 'amplitude', (A, Bn), fresh(amp), (xr, xi))

        def remap(a, c, k):
            v = jnp.sqrt(a * a + c * c) * k
            return jnp.clip(255.0 * (v / (3.0 * jnp.mean(v))) ** 0.7, 0, 255).astype(jnp.uint8)
        b.run('detect', 'amplitude_and_8bit', (A, Bn), fresh(jax.jit(remap)), (xr, xi))


def group_cpu(b, sizes):
    """This package's C++ polar format on the host's cores: a broadside spotlight whose picture is A by Bn pixels of
    0.96 of its resolution, a history of noise (the time does not depend on the scene)."""
    try:
        from fastsar import quick, sim
        quick._lib()
    except Exception as e:
        b.record('cpu', 'polar_format', (0, 0), skipped=f'not available here ({type(e).__name__}: {str(e)[:200]})')
        return
    rng = np.random.default_rng(0)
    for A, Bn in sizes:
        res = 0.5
        col = sim.make_collect(res=res, scene=0.96 * res * max(A, Bn) / (1.25 * math.sqrt(2.0)) * 1.2, r0=600e3, graze_deg=45.0)
        P, K = col.Np, col.K
        a = col.ant[P // 2]
        e2 = -a / np.linalg.norm(a)
        n = np.cross(e2, col.ant[P // 2 + 1] - col.ant[P // 2 - 1])
        n *= np.sign(n[2]) / np.linalg.norm(n)
        e1 = np.cross(e2, n)
        S = np.empty((P, K), np.complex64)
        for r in range(0, P, 256):
            blk = rng.standard_normal((min(256, P - r), K, 2), np.float32)
            S[r:r + 256] = blk[..., 0] + 1j * blk[..., 1]
        sp = 0.96 * res
        for correct in (True, False):
            ts, tm = [], {}
            try:
                for _ in range(b.repeat + 1):
                    tm = {}
                    t = time.perf_counter()
                    quick._pfa(S, col.ant, col.fmin, col.df, A - A % 2, Bn - Bn % 2, sp, sp, e1, e2, correct=correct,
                               window=True, complex_image=False, timing=tm)
                    ts.append(time.perf_counter() - t)
                ts = ts[1:]
                b.record('cpu', 'polar_format_corrected' if correct else 'polar_format_uncorrected', (A, Bn),
                         seconds_min=min(ts), seconds_median=float(np.median(ts)), history=[P, K],
                         threads=quick._workers(), stages={k: round(v, 4) for k, v in tm.items()})
            except Exception as e:
                b.record('cpu', 'polar_format', (A, Bn), failed=f'{type(e).__name__}: {str(e)[:300]}')
        del S


GROUPS = dict(fft=group_fft, resample=group_resample, warp=group_warp, transfer=group_transfer, detect=group_detect,
              cpu=group_cpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='tpu_blocks.json', help='the JSON file (rewritten after every block)')
    ap.add_argument('--sizes', default='1024,2048,4096,8192,12288x9216')
    ap.add_argument('--only', default=','.join(GROUPS), help='groups, comma separated: ' + ', '.join(GROUPS))
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--skip', default='', help='blocks whose names hold one of these (comma separated) are left out')
    a = ap.parse_args()
    sizes = []
    for s in a.sizes.split(','):
        p = [int(v) for v in s.lower().split('x')]
        sizes.append((p[0], p[-1]))
    groups = [g.strip() for g in a.only.split(',') if g.strip()]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        ap.error(f'unknown group {bad[0]!r}: one of {", ".join(GROUPS)}')
    meta = dict(script='bench_tpu_blocks', python=sys.version.split()[0], machine=platform.machine(),
                cpus=os.cpu_count(), repeat=a.repeat, sizes=[list(s) for s in sizes])
    if any(g != 'cpu' for g in groups):
        import jax
        devs = jax.devices()
        meta.update(jax=jax.__version__, backend=jax.default_backend(), devices=[str(d) for d in devs],
                    platform=devs[0].platform)
        print(f"jax {jax.__version__} on {devs[0].platform} ({len(devs)} device(s)); {a.repeat} timed runs per block", flush=True)
    b = Bench(a.out, a.repeat, meta, [k for k in a.skip.split(',') if k])
    for g in groups:
        GROUPS[g](b, sizes)
    b.save()
    print(f'wrote {a.out}: {len(b.doc["results"])} results')


if __name__ == '__main__':
    main()
