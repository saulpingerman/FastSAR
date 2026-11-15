"""The quick look's polar format on a JAX device (a TPU, or whatever JAX runs on): the stages of fastsar.quick in
float32, every complex array as two real planes outside the transforms, and no indexed load of a sample.

What a TPU does fast and slowly decides the form of every stage (tests/bench_tpu_blocks.py, on a v6e at 113
million values): a 2-D transform takes 0.04 s; a resampling pass with an indexed load per tap 119 s; the same pass
as one row of a table of windows per block of 32 outputs, then a matrix product of the block with its weights,
0.19 s; a complex array crosses to the host 68 times slower than its two planes; a block of 4.6 GB ran 50 times
slower than blocks of 2 GB. So:

  Reads. Every resampling is _dense: a block's taps lie in one window of the row, read as a row of the table of
  windows, and the block's outputs are the product of that window with weights formed on the device
  (Precision.HIGHEST: the default leaves 2e-3). The only indexed loads are of whole rows of 32 to 100 samples.

  A sheared raster (a picture on the ground under a twisted slant plane). Each pulse is read on its own rows of the
  raster, from a row that is a multiple of 32, and the azimuth stage takes its rows across the pulses as blocks of
  32 samples (one indexed load a block). The rows fold into the transform's period by a reshape and a sum. The
  baseband image is then smooth along sheared lines and not along its columns; its rows are shifted by the shear
  before their transform (a phase ramp over the azimuth wavenumbers, exact), which makes it smooth along both
  axes, and the read at the apparent positions is two passes of _dense: along the columns of the shifted image to
  the picture's rows, then along those rows to the picture's columns. A plain raster is the same with no shear.

  Missing pulses. The history goes up in blocks of pulses, each pulse at its place in the evenly timed sequence;
  the host builds the blocks that hold a gap (a copy of those blocks alone) and the others are the history's own
  bytes.

  Memory. The history is never on the device whole. Of the three large arrays (the range stage's output, the
  raster, its range transform) those that fit stay on the device, in a few pieces that are joined as their blocks
  come (never there twice); the others live on the host and their blocks cross (_plan). The read at the apparent
  positions works in strips of the picture's rows, and the picture comes down strip by strip. Every block is sized
  from _xp.jax_block() (at most 2 GB).

  Programs. A stage is a chain of three functions: the cut of a block out of the array before it (specific to
  that array's size), the block's layout for the stage (padding, signs, the fold), and the stage's work on a block
  whose sizes are all taken from a short ladder (_ladder: rows per block, outputs per row, windows; a TPU's
  transform lengths are few). FASTSAR_JAX_PROGRAMS=shared compiles the three separately, so collections of
  different sizes share the programs that take long to compile; 'fused' compiles a chain as one program. COMPILED
  lists what this process compiled and how long each first call took.

Positions. A resampling position reaches tens of thousands of samples and must hold to a small part of a sample,
more than float32 carries, and a TPU has no fast float64. Along a row the range positions are linear and the
azimuth positions smooth: the host forms them in float64 at the first, middle and last outputs of blocks of BLOCK
outputs and sends the integer part at the block's first output (int32) and the remainder, the step and the
curvature (float32). The apparent positions of the final read are the pixel's own index (an integer) plus the
displacement polynomial, which the device evaluates in float32. Phases that reach many cycles (the reference to
the grid center, the shear's ramp) are formed in 32-bit integer arithmetic, a cycle being 2^32.
"""
import math
import os
import time

import numpy as np

BLOCK = 32                    # outputs whose positions are one parabola, and whose taps lie in one window
STRIDE = 8                    # samples between the starts of the windows in the table of windows
GROUP = 8                     # pieces a device array is kept in
_PROGRAMS = {}
COMPILED = []                 # (name, key, seconds of the first call) of each program this process compiled
_KAISER = {}
SHEAR_TAPS, SHEAR_KAISER = 10, 7.2   # the kernel of the read at the apparent positions on a sheared raster


def _mode():
    """'shared' or 'fused': FASTSAR_JAX_PROGRAMS (see the module docstring)."""
    v = os.environ.get('FASTSAR_JAX_PROGRAMS', '').strip().lower() or 'shared'
    if v not in ('shared', 'fused'):
        raise ValueError(f"FASTSAR_JAX_PROGRAMS must be 'shared' or 'fused', got {v!r}")
    return v


def _program(name, key, make, detail=None):
    """The compiled program `name` for its static sizes `key`, kept between calls (the newest 256). Its first call
    is brought to completion and timed (COMPILED); with detail (a dict) every call is, and the seconds are added
    to detail[name]."""
    k = (name,) + tuple(key)
    p = _PROGRAMS.get(k)
    if p is None:
        import jax
        if len(_PROGRAMS) >= 256:
            _PROGRAMS.pop(next(iter(_PROGRAMS)))
        p = _PROGRAMS[k] = [jax.jit(make()), False]
    if p[1] and detail is None:
        return p[0]

    def call(*args):
        import jax
        t = time.perf_counter()
        out = jax.block_until_ready(p[0](*args))
        dt = time.perf_counter() - t
        if not p[1]:
            p[1] = True
            COMPILED.append((name, k[1:], dt))
        elif detail is not None:
            detail[name] = detail.get(name, 0.0) + dt
        return out
    return call


def _ladder(n, fine=16):
    """The smallest size >= n of those programs are compiled for: k 2^e with fine <= k <= 2 fine (at most a
    sixteenth more than n, a thirty-second on average), n itself up to 2 fine."""
    n = int(max(n, 1))
    e = max(0, n.bit_length() - fine.bit_length())
    return -(-n >> e) << e


def _ladder_down(n, fine=16):
    """The largest size <= n of _ladder's."""
    n = int(max(n, 1))
    e = max(0, n.bit_length() - fine.bit_length())
    return (n >> e) << e


def _up(n, m):
    return -(-int(n) // m) * m


def _starts(n, per):
    """Blocks of `per` rows over n rows, the last moved back to end at the last row: (per, starts)."""
    per = int(max(1, min(per, n)))
    starts = list(range(0, n - per + 1, per))
    if starts[-1] + per < n:
        starts.append(n - per)
    return per, starts


def _blocks_of(total, per, multiple=1):
    """The blocks of a stage over `total` rows when `per` fit in its memory: (rows of a block, a size of the
    ladder; rows of it that are the array's; starts). The fewest blocks that hold the rows, of the ladder's size
    next above an even share (the last is moved back to end at the last row, and repeats few rows); one block
    where the rows fit, padded to the ladder's next size. multiple: the block's rows are a multiple of it (total
    is)."""
    per = int(max(multiple, (_ladder_down(max(per, 1)) // multiple) * multiple))
    if total <= per:
        return _up(_ladder(total), multiple), int(total), [0]
    per = min(per, _up(_ladder(-(-total // -(-total // per))), multiple))
    return per, per, _starts(total, per)[1]


def _knots(pos, pb=BLOCK):
    """Positions at the first outputs, middles and ends of runs of pb outputs, [R, 2 n + 1] (float64) -> each run's
    parabola through its three: the integer part at its first output (int32 [R, n]), the remainder there, the step
    and the curvature per output (float32): output i of the run lies at base + lo + i s + i^2 c."""
    p0, pm, pe = pos[:, 0:-1:2], pos[:, 1::2], pos[:, 2::2]
    c = 2.0 * (pe - 2.0 * pm + p0) / pb ** 2
    base = np.floor(p0)
    return (base.astype(np.int32), (p0 - base).astype(np.float32), ((pe - p0) / pb - c * pb).astype(np.float32),
            c.astype(np.float32))


def _positions(jnp, base, lo, s, c, pb=BLOCK):
    """The positions of runs of pb outputs as the sample below (int32) and the fraction above it, [R, n * pb]."""
    i = jnp.arange(pb, dtype=jnp.float32)
    f = lo[..., None] + i * s[..., None] + (i * i) * c[..., None]
    fl = jnp.floor(f)
    R = base.shape[0]
    return (base[..., None] + fl.astype(jnp.int32)).reshape(R, -1), (f - fl).reshape(R, -1)


def _cycles(v):
    """Phases in cycles (float64) -> their fractions as 32-bit integers, a cycle being 2^32 (uint32): sums and
    products of them with integers wrap by whole cycles."""
    v = np.asarray(v, np.float64)
    return (np.rint((v - np.floor(v)) * 4294967296.0).astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)


def _angle(jnp, c):
    """A phase as a 32-bit integer (_cycles) -> radians (float32, to 4e-7 rad)."""
    return (c >> 8).astype(jnp.float32) * (2.0 * math.pi / 16777216.0)


def _kaiser_poly(beta=None):
    """A Kaiser window (beta; the host kernels' quick.KAISER by default) as a polynomial in the square of the offset
    over the half width, float32 coefficients of degree 7, highest power first: to 1.2e-6 of the window."""
    if beta is None:
        from . import quick as Q
        beta = Q.KAISER
    if beta not in _KAISER:
        x2 = np.linspace(0.0, 1.0, 4001)
        w = np.i0(beta * np.sqrt(1.0 - x2)) / np.i0(beta)
        _KAISER[beta] = np.polynomial.chebyshev.Chebyshev.fit(x2, w, 7).convert(kind=np.polynomial.Polynomial).coef[::-1].astype(np.float32)
    return _KAISER[beta]


def _window(step, taps):
    """Samples of the window that holds the taps of BLOCK outputs `step` samples apart, a multiple of 16."""
    return _up(math.ceil(BLOCK * float(step)) + taps + STRIDE + 6, 16)


def _windows(jnp, u, w0, Rw):
    """u [R, N] (N a multiple of STRIDE), w0 [R, nblk] (multiples of STRIDE) -> u[r, w0 : w0 + Rw] [R, nblk, Rw], each
    read as one row of the table of the windows that start at every STRIDE-th sample (resample_jax's)."""
    R, N = u.shape
    c = u.reshape(R, N // STRIDE, STRIDE)
    nrow = (N - Rw) // STRIDE + 1
    tab = jnp.concatenate([c[:, i:nrow + i] for i in range(Rw // STRIDE)], -1).reshape(R * nrow, Rw)
    return tab.at[jnp.arange(R, dtype=jnp.int32)[:, None] * nrow + w0 // STRIDE].get(mode='promise_in_bounds')


def _dense(jnp, lax, xr, xi, ib, fr, Rw, taps, wpoly):
    """Rows of x [R, n] (two planes) at the positions ib + fr [R, m] (m a multiple of BLOCK), by blocks of BLOCK
    outputs, without an indexed load of a tap. A block's taps lie in one window of Rw samples, read as a row of
    the table of windows; the block's outputs are the product of that window with weights [BLOCK, Rw] formed on
    the device: the sinc from one sine per output (the sine of pi (u - w) is that of pi u with the sign of w's
    parity), times the Kaiser window as a polynomial (wpoly, 16 taps) or the sinc window (None: its sine follows
    from the angle sum with constants of w). A tap outside the row counts as zero and the rest are normalised
    again; zero where the position lies outside (-0.5, n - 0.5): quicklook_cpu.cpp's ql_resample_rows, with the
    weights at the position itself and not at the nearest of 2048 phases."""
    R, n = xr.shape
    nblk, h = ib.shape[1] // BLOCK, taps // 2
    N = max(-(-n // STRIDE) * STRIDE, Rw)
    ibb, frb = ib.reshape(R, nblk, BLOCK), fr.reshape(R, nblk, BLOCK)
    w0 = jnp.clip(((jnp.min(ibb, axis=-1) - (h - 1)) // STRIDE) * STRIDE, 0, N - Rw)
    rel = ibb - w0[..., None]
    wv = jnp.arange(Rw, dtype=jnp.float32)
    u = rel.astype(jnp.float32) + frb
    d = u[..., None] - wv                                                          # [R, nblk, BLOCK, Rw]
    s1 = jnp.sin(math.pi * frb) * jnp.where(rel % 2 == 0, 1.0, -1.0)
    par = jnp.where(jnp.arange(Rw) % 2 == 0, 1.0, -1.0).astype(jnp.float32)
    near = jnp.abs(d) < 1e-4
    k = (s1[..., None] * par) / (math.pi * jnp.where(near, 1.0, d))
    if wpoly is None:
        a = (math.pi / h) * u
        sh = jnp.sin(a)[..., None] * jnp.cos((math.pi / h) * wv) - jnp.cos(a)[..., None] * jnp.sin((math.pi / h) * wv)
        k = k * sh * (h / math.pi) / jnp.where(near, 1.0, d)
    else:
        x2 = (d * (1.0 / h)) ** 2
        win = jnp.zeros_like(x2) + wpoly[0]
        for c in wpoly[1:]:
            win = win * x2 + c
        k = k * win
    k = jnp.where(near, 1.0, k)
    k = jnp.where((jnp.abs(d) < h) & ((w0[..., None, None] + jnp.arange(Rw, dtype=jnp.int32)) < n), k, 0.0)
    k = k / jnp.maximum(jnp.abs(k.sum(-1, keepdims=True)), 1e-12) * jnp.sign(k.sum(-1, keepdims=True) + 1e-30)
    inside = ((ib >= 0) | ((ib == -1) & (fr > 0.5))) & ((ib < n - 1) | ((ib == n - 1) & (fr < 0.5)))
    pad = lambda p: jnp.pad(p, ((0, 0), (0, N - n)))
    get = lambda p: jnp.where(inside, jnp.einsum('rbjw,rbw->rbj', k, _windows(jnp, pad(p), w0, Rw),
                                                 precision=lax.Precision.HIGHEST).reshape(R, -1), 0.0)
    return get(xr), get(xi)


# ----------------------------------------------------------------------------------------------- a formation's run

class _Run:
    """What a formation's steps share: the programs' mode, the timing of every call (timing['detail'] a dict), the
    host kernels, and the block that is on its way down while the device works on the next."""

    def __init__(self, timing):
        self.detail = timing.get('detail') if isinstance(timing.get('detail'), dict) else None
        self.mode = _mode()
        self.pending = None
        self._lib = None

    @property
    def lib(self):
        if self._lib is None:
            from . import quick as Q
            self._lib = Q._lib()
        return self._lib

    def program(self, name, key, make):
        return _program(name, key, make, self.detail)

    def chain(self, parts):
        """parts: (name, key, fn, args) in order; each fn takes the outputs of the one before it, then its own
        args, and returns a tuple. 'shared': each its own program; 'fused': one program of them all."""
        if self.mode == 'shared' or len(parts) == 1:
            out = ()
            for name, key, fn, args in parts:
                out = self.program(name, key, lambda fn=fn: fn)(*out, *args)
            return out
        fns, counts = [p[2] for p in parts], [len(p[3]) for p in parts]

        def make():
            def run(*flat):
                out, i = (), 0
                for fn, n in zip(fns, counts):
                    out = fn(*out, *flat[i:i + n])
                    i += n
                return out
            return run
        key = tuple(v for p in parts for v in (p[0],) + tuple(p[1]))
        return self.program(' + '.join(p[0] for p in parts), key, make)(*(a for p in parts for a in p[3]))

    def fetch(self, dev, dst):
        """dst[...] = the device block dev (its leading rows and columns; a complex block as [.., 2] float32),
        once the next block has been started: the copy overlaps the device's work."""
        prev, self.pending = self.pending, (dev, dst)
        if prev is not None:
            self._get(*prev)

    def flush(self):
        prev, self.pending = self.pending, None
        if prev is not None:
            self._get(*prev)

    def _get(self, dev, dst):
        t = time.perf_counter()
        a = np.asarray(dev)
        if a.ndim == 3:
            a = a.view(np.complex64)[..., 0]
        dst[...] = a[:dst.shape[0], :dst.shape[1]]
        if self.detail is not None:
            self.detail['blocks to the host'] = self.detail.get('blocks to the host', 0.0) + time.perf_counter() - t


def _unpack(x):
    return x[..., 0], x[..., 1]


class _Device:
    """A large complex array [rows, width] on the device as two float32 planes, kept in at most GROUP pieces of
    rows: its blocks (`per` rows each, the last moved back to end at the last row) are joined as they come, a
    piece at a time, so the array is never on the device twice."""
    host = False

    def __init__(self, run, total, per, starts, width):
        self.run, self.total, self.per, self.width = run, total, per, width
        self.group = max(1, -(-len(starts) // GROUP))
        self.pieces, self.waiting, self.done = [], [], 0

    def put(self, r0, re, im):
        self.waiting.append((r0, re, im))
        if len(self.waiting) == self.group:
            self._join()

    def _join(self):
        trims = []
        for r0, _, _ in self.waiting:
            trims.append(self.done - r0)             # rows of a block moved back that the one before it holds
            self.done = r0 + self.per
        if len(self.waiting) == 1 and trims[0] == 0:
            self.pieces.append(self.waiting[0][1:])
        else:
            n = len(trims)

            def make(trims=tuple(trims)):
                import jax.numpy as jnp

                def run(*a):
                    return (jnp.concatenate([x[t:] for x, t in zip(a[:n], trims)], 0),
                            jnp.concatenate([x[t:] for x, t in zip(a[n:], trims)], 0))
                return run
            join = self.run.program('join', (self.per, self.width, tuple(trims)), make)
            self.pieces.append(join(*(w[1] for w in self.waiting), *(w[2] for w in self.waiting)))
        self.waiting = []

    def close(self):
        if self.waiting:
            self._join()
        self.rows = sum(int(p[0].shape[0]) for p in self.pieces)
        return self

    def cut(self, n, c0, shifts=None):
        """The chain's first part for the block of columns c0 .. c0 + n, turned: [n, rows]. shifts: int32 [rows],
        row r holds the columns from shifts[r] * BLOCK on (a sheared raster's pulses, each on its own rows):
        the block's columns are then read as blocks of BLOCK samples, zero where a row does not reach."""
        import jax.numpy as jnp
        from jax import lax
        npc, W = len(self.pieces), self.width
        sizes = tuple(int(p[0].shape[0]) for p in self.pieces)

        def fn(*a):
            c = a[2 * npc]
            if shifts is None:
                f = lambda p: lax.dynamic_slice_in_dim(p, c, n, 1)
                return jnp.concatenate([f(p) for p in a[:npc]], 0).T, jnp.concatenate([f(p) for p in a[npc:2 * npc]], 0).T
            sb, nwb, out = a[2 * npc + 1], W // BLOCK, []
            for planes in (a[:npc], a[npc:2 * npc]):
                parts, off = [], 0
                for p, R in zip(planes, sizes):
                    idx = c // BLOCK + jnp.arange(n // BLOCK, dtype=jnp.int32)[None, :] - sb[off:off + R, None]
                    flat = jnp.arange(R, dtype=jnp.int32)[:, None] * nwb + jnp.clip(idx, 0, nwb - 1)
                    v = p.reshape(R * nwb, BLOCK).at[flat].get(mode='promise_in_bounds')
                    parts.append(jnp.where(((idx >= 0) & (idx < nwb))[..., None], v, 0.0).reshape(R, n))
                    off += R
                out.append(jnp.concatenate(parts, 0).T)
            return tuple(out)
        args = tuple(p[0] for p in self.pieces) + tuple(p[1] for p in self.pieces) + (np.int32(c0),)
        if shifts is not None:
            args += (jnp.asarray(shifts),)
        return ('cut', (sizes, W, n, shifts is not None), fn, args)


class _Host:
    """The same array on the host (complex64), for a device without room for it: its blocks come down as they are
    made and go up, turned, as the next stage takes them."""
    host = True

    def __init__(self, run, total, per, starts, width):
        self.run, self.total, self.per, self.width = run, total, per, width
        self.rows = max(total, per)
        self.a = np.empty((self.rows, width), np.complex64)

    def put(self, r0, re, im):
        import jax.numpy as jnp
        pack = self.run.program('pack', (self.per, self.width), lambda: lambda a, b: jnp.stack([a, b], -1))
        self.run.fetch(pack(re, im), self.a[r0:r0 + self.per])

    def close(self):
        self.run.flush()
        return self

    def cut(self, n, c0, shifts=None):
        import ctypes
        import jax.numpy as jnp
        lib, a = self.run.lib, self.a
        t = time.perf_counter()
        hb = np.empty((n, self.rows), np.complex64)
        ptr = lambda v: v.ctypes.data_as(ctypes.c_void_p)
        if shifts is None:
            lib.ql_transpose(ctypes.c_void_p(a.ctypes.data + c0 * a.strides[1]), self.width, self.rows, n, ptr(hb), self.rows,
                             None, 0)
        else:
            jb = np.ascontiguousarray(np.asarray(shifts, np.int64) * BLOCK, np.int32)
            lib.ql_unshear(ptr(a), self.width, self.rows, ptr(jb), None, self.rows, c0, n, ptr(hb), self.rows)
        x = jnp.asarray(hb.view(np.float32).reshape(n, self.rows, 2))
        if self.run.detail is not None:
            d = self.run.detail
            d['blocks to the device'] = d.get('blocks to the device', 0.0) + time.perf_counter() - t
        return ('unpack', (n, self.rows), _unpack, (x,))


def _plan(sizes, budget, block):
    """Where the three large arrays live for a device with `budget` bytes to give (None: no limit known) and
    `block` bytes for a stage's own arrays: dict(A, B, F: True on the device; kept: the names of those on the
    host). A stage holds the array it reads, the one it writes with one more piece of it while that is joined,
    and three blocks; the largest go to the host first until what is left fits."""
    names = dict(A="the range stage's output", B='the raster', F="the raster's range transform")
    on = dict.fromkeys(sizes, True)
    if budget is not None:
        def peak():
            s = {k: sizes[k] if on[k] else 0 for k in sizes}
            g = 1.0 + 1.0 / GROUP
            return max(g * s['A'], s['A'] + g * s['B'], s['B'] + g * s['F'], s['F']) + 3 * block
        for k in sorted(sizes, key=lambda k: -sizes[k]):
            if peak() <= budget:
                break
            on[k] = False
    return dict(on, kept=[names[k] for k in sizes if not on[k]])


def plan(g, K=None):
    """The sizes of a formation on the JAX device and where its arrays live, for a geometry of quick._geometry:
    dict(bytes: of A, B and F; A, B, F: on the device; kept; budget, block). What pfa takes."""
    from . import _xp
    lay = _layout(g, _xp.jax_block())
    return dict(_plan(lay['bytes'], _xp.jax_budget(), _xp.jax_block()), bytes=lay['bytes'], budget=_xp.jax_budget(),
                block=_xp.jax_block())


def _layout(g, blk, correct=True):
    """The blocks of the stages for a geometry and `blk` bytes a block: dict of each stage's (rows of a block,
    rows of it that are the array's, starts), the padded widths, the windows, and the bytes of the three arrays."""
    from . import quick as Q
    P, Kf, Nr, nkr, nka, nwin = g['P'], g['Kf'], g['Nr'], g['nkr'], g['nka'], g['nwin']
    Pf, Na, Pp, ov = g['Pf'], g['Na'], g['Pp'], (g['ov'] if correct else 1)
    sheared = g['jlo'] is not None
    # 1. range: each pulse on the rows of the raster it reaches, from a multiple of BLOCK
    nblk = _ladder(-(-(nwin + (BLOCK - 1 if sheared else 0)) // BLOCK))
    W1 = nblk * BLOCK
    Rw1 = _window(float(np.abs(g['beta']).max()), Q.TAPS)
    b1 = _blocks_of(Pp, blk // (40 * max(Kf, Nr, g['K']) + 16 * W1 * Rw1))
    # 2. azimuth: the raster's rows, a multiple of BLOCK of them
    rows2 = _up(nkr, BLOCK)
    nblk2 = _ladder(-(-nka // BLOCK))
    W2 = nblk2 * BLOCK
    j = np.arange(2 * nblk2 + 1, dtype=np.float64) * (BLOCK / 2)
    kr = g['kr_min'] + g['dkr'] * np.array([0.0, nkr - 1.0])
    pos = _az_positions(g, kr, j)
    Rw2 = _window(float(np.abs(np.diff(pos[:, ::2], axis=1)).max()) / BLOCK, Q.TAPS)
    # the outputs whose positions are one parabola: BLOCK where the pulses' spacing is smooth over them (any picture
    # of more than a few hundred pixels), fewer where a parabola through three of them misses the others by more
    # than 2e-4 of a sample (a small picture of unevenly spaced pulses)
    pb2 = BLOCK
    while pb2 > 4:
        q = _az_positions(g, kr, np.arange(4 * W2 // pb2 + 1, dtype=np.float64) * (pb2 / 4))
        a, m, e = q[:, 0:-1:4], q[:, 2::4], q[:, 4::4]               # each run's first output, middle and end
        cc = 2.0 * (e - 2.0 * m + a) / pb2 ** 2
        ss = (e - a) / pb2 - cc * pb2
        ok = (a > -0.5) & (e < Na - 0.5)
        miss = lambda i, v: np.abs(a + ss * i + cc * i * i - v)[ok].max(initial=0.0)
        if max(miss(pb2 / 4, q[:, 1::4]), miss(3 * pb2 / 4, q[:, 3::4])) < 2e-4:
            break
        pb2 //= 2
    b2 = _blocks_of(rows2, blk // (40 * max(Pf, Na) + 16 * W2 * Rw2), BLOCK)
    # 3. the range transform: the raster's columns
    Ny = ov * g['nfy']
    fold = -(-rows2 // Ny)
    rows3 = _up(nka, BLOCK)
    b3 = _blocks_of(rows3, blk // (40 * max(fold * Ny, rows2)))
    size = lambda b, total, width: 8 * max(total, b[0]) * width
    return dict(nblk=nblk, W1=W1, Rw1=Rw1, b1=b1, rows2=rows2, nblk2=nblk2, W2=W2, Rw2=Rw2, pb2=pb2, b2=b2, Ny=Ny, fold=fold,
                rows3=rows3, b3=b3, sheared=sheared,
                bytes=dict(A=size(b1, Pp, W1), B=size(b2, rows2, W2), F=size(b3, rows3, Ny)))


def _az_positions(g, kr, j):
    """The pulse position (in the Na pulses of the gated row) of the azimuth wavenumbers ka_lo + j dka at the range
    wavenumbers kr: [len(kr), len(j)] (float64), from the pulses' own tangents (quick._geometry's table; beyond
    it the line of its last two points: such positions lie outside the row and read zero)."""
    qtab = g['qtab']
    u = ((g['ka_lo'] + j[None, :] * g['dka']) / kr[:, None] - g['tg0']) * g['tg_inv']
    kq = np.clip(np.floor(u), 0, len(qtab) - 2).astype(np.int64)
    return qtab[kq] + (u - kq) * (qtab[kq + 1] - qtab[kq])


def _poly(jnp, coef, powers, x, y):
    """The displacement polynomial (m) at normalised coordinates: the sum of coef[k] x^a y^b over the powers."""
    px, py = [jnp.ones_like(x)], [jnp.ones_like(y)]
    for _ in range(max(a for a, _ in powers)):
        px.append(px[-1] * x)
    for _ in range(max(b for _, b in powers)):
        py.append(py[-1] * y)
    return sum(coef[k] * px[a] * py[b] for k, (a, b) in enumerate(powers))


def _split(jnp, index, frac):
    fl = jnp.floor(frac)
    return index + fl.astype(jnp.int32), frac - fl


def pfa(S, g, window, correct, complex_image, timing, phase=None, info=None):
    """fastsar.quick's polar format on JAX's device -> the image on the host, range-major [ny, nx] (complex64, or
    its amplitude, float32). S [P, K] complex64 on the host (or a one-element list that hands it over); g:
    quick._geometry's; phase: (ph0, ph1) [P] in cycles, a reference phase exp(2 pi j (ph0 + k ph1)) of sample k
    that the device applies with the weights. info['stream']: what was kept on the host, where the device had no
    room. timing: the stages' seconds; with timing['detail'] a dict, every program's call is brought to completion
    and its seconds are added there by name."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from . import _xp
    from . import quick as Q
    f32, i32, c64 = np.float32, np.int32, np.complex64
    run = _Run(timing)
    S = np.ascontiguousarray(S.pop() if isinstance(S, list) else S, c64)
    P, K, Kf, Nr, k0 = g['P'], g['K'], g['Kf'], g['Nr'], g['k0']
    nkr, nka, nfy, nfx, nx, ny = g['nkr'], g['nka'], g['nfy'], g['nfx'], g['nx'], g['ny']
    nwin, jlo, Pf, Na, p0, Pp, place = g['nwin'], g['jlo'], g['Pf'], g['Na'], g['p0'], g['Pp'], g['place']
    blk, budget = _xp.jax_block(), _xp.jax_budget()
    L = _layout(g, blk, correct)
    where = _plan(L['bytes'], budget, blk)
    if where['kept'] and info is not None:
        from .memory import _gb
        one = len(where['kept']) == 1
        info['stream'] = (f"the JAX device has {_gb(budget)} to give: {', '.join(where['kept'])} {'is' if one else 'are'} "
                          f"kept on the host and {'crosses' if one else 'cross'} in blocks")
    store = lambda name, total, b, width: (_Device if where[name] else _Host)(run, total, b[0], b[2], width)
    up = lambda a, t=f32: jnp.asarray(np.ascontiguousarray(a, t))
    HI = lax.Precision.HIGHEST
    wpoly = jnp.asarray(_kaiser_poly())
    sheared = L['sheared']

    def rows_of(v, sl, n, dtype):
        """v[sl] as `dtype`, zeros to n rows, on the device."""
        out = np.zeros((n,) + v.shape[1:], dtype)
        out[:sl.stop - sl.start] = v[sl]
        return jnp.asarray(out)

    t0 = time.perf_counter()
    # ------------------------------------------------------------------------------------------------ 1. range
    # A block of pulses, each at its place: weighted (the reference phase with the weights), padded, gated in range
    # and read at the raster's range wavenumbers, each pulse on its own rows of the raster from row BLOCK * shift
    src = None
    if place is not None:
        src = np.full(Pp, -1, np.int64)
        src[place] = np.arange(P)

    def placed(v, dtype=np.float64):
        v = np.asarray(v, dtype)
        if place is None:
            return v
        out = np.zeros((Pp,) + v.shape[1:], dtype)
        out[place] = v
        return out
    alpha, beta = placed(g['alpha']), placed(g['beta'])
    wp = placed(g['wp'] * (Q._taylor(P) if window else 1.0))
    jl = placed(jlo, np.int64) if sheared else np.zeros(Pp, np.int64)
    shift1 = (jl // BLOCK).astype(i32)
    off = jl - shift1.astype(np.int64) * BLOCK
    shift1 = np.concatenate([shift1, np.zeros(max(0, L['b1'][0] - Pp), i32)])     # (a single block is padded)
    there = np.ones(Pp, bool) if place is None else src >= 0
    clo, chi = np.where(there, off, 0).astype(i32), np.where(there, off + nwin, 0).astype(i32)
    ref = phase is not None
    A0, B0 = (placed(_cycles(v), np.uint32) for v in (phase if ref else (np.zeros(P), np.zeros(P))))
    wk = up(g['wk'] * (Q._taylor(K) if window else 1.0))
    tap_r = up(Q._taper(Nr, g['Er'], g['gy'], g['bin_r']))
    nblk, W1, Rw1 = L['nblk'], L['W1'], L['Rw1']
    nb, n1, starts = L['b1']
    keep = min(Nr, Kf)
    pos_, neg_ = keep - keep // 2, keep // 2

    def weights(x, wk, wpb, a0, b0):
        sr, si = x[..., 0], x[..., 1]
        if ref:
            c = _angle(jnp, jnp.arange(K, dtype=jnp.uint32)[None, :] * b0[:, None] + a0[:, None])
            cs, sn = jnp.cos(c), jnp.sin(c)
            sr, si = sr * cs - si * sn, sr * sn + si * cs
        w = wk[None, :] * wpb[:, None]
        pad = ((0, nb - n1), (k0, Kf - K - k0))
        return jnp.pad(sr * w, pad), jnp.pad(si * w, pad)

    def range_(zr, zi, tap_r, wpoly, base, lo, s, c2, clo, chi):
        prof = jnp.fft.ifft(lax.complex(zr, zi), axis=1)
        gate = jnp.concatenate([prof[:, :pos_], jnp.zeros((nb, Nr - keep), prof.dtype), prof[:, Kf - neg_:]], 1) * tap_r
        G = jnp.fft.fft(gate, axis=1)
        ib, fr = _positions(jnp, base, lo, s, c2)
        o = _dense(jnp, lax, jnp.real(G), jnp.imag(G), ib, fr, Rw1, Q.TAPS, wpoly)
        col = jnp.arange(W1, dtype=jnp.int32)[None, :]
        m = (col >= clo[:, None]) & (col < chi[:, None])
        return jnp.where(m, o[0], 0.0), jnp.where(m, o[1], 0.0)
    S3 = S.view(np.float32).reshape(P, K, 2)
    j = np.arange(2 * nblk + 1, dtype=np.float64) * (BLOCK / 2)
    A = store('A', Pp, L['b1'], W1)
    for r in starts:
        sl = slice(r, r + n1)
        if src is None:
            x = S3[sl]
        elif src[r] >= 0 and src[r + n1 - 1] - src[r] == n1 - 1:       # no gap in the block: the history's own bytes
            x = S3[src[r]:src[r] + n1]
        else:                                                          # the host places the pulses of this block
            x = np.zeros((n1, K, 2), f32)
            q = np.nonzero(src[sl] >= 0)[0]
            x[q] = S3[src[sl][q]]
        kn = _knots(alpha[sl, None] + (j[None, :] - off[sl, None]) * beta[sl, None])
        o = run.chain([('weights', (n1, nb, K, Kf, k0, ref), weights,
                        (jnp.asarray(x), wk, rows_of(wp, sl, n1, f32), jnp.asarray(A0[sl]), jnp.asarray(B0[sl]))),
                       ('range', (nb, Kf, Nr, nblk, Rw1), range_,
                        (tap_r, wpoly) + tuple(rows_of(v, slice(0, n1), nb, v.dtype) for v in kn)
                        + (rows_of(clo, sl, nb, i32), rows_of(chi, sl, nb, i32)))])
        A.put(r, *o)
    A.close()
    del S, S3, x
    jax.block_until_ready(o)
    timing['range'] = timing.get('range', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # ---------------------------------------------------------------------------------------------- 2. azimuth
    # A block of the raster's range wavenumbers, the pulses of each along a row: gated in azimuth (in meters, for
    # each wavenumber's own bins) and read at the raster's azimuth wavenumbers at the pulses' own spacing
    ov = g['ov'] if correct else 1
    Ny = L['Ny']
    rows2, nblk2, W2, Rw2 = L['rows2'], L['nblk2'], L['W2'], L['Rw2']
    nJ, n2, starts = L['b2']
    kk = np.arange(Na)
    nbin = np.where(kk < Na - Na // 2, kk, kk - Na)
    keep2 = min(Na, Pf)
    pa, na_ = keep2 - keep2 // 2, keep2 // 2
    dist_bins, live = up(np.abs(nbin)), up((nbin < pa) & (nbin >= -na_))
    edge, guard, scale = f32(g['Ex']), f32(g['gx']), f32(g['gain'] * Ny * Na / Pf)
    pb2 = L['pb2']
    ja = np.arange(2 * (W2 // pb2) + 1, dtype=np.float64) * (pb2 / 2)

    def pulses(zr, zi):
        pad = ((0, nJ - n2), (p0, Pf - Pp - p0))
        return jnp.pad(zr[:, :Pp], pad), jnp.pad(zi[:, :Pp], pad)

    def azimuth(zr, zi, dist_bins, live, edge, guard, scale, binm, wpoly, base, lo, s, c2, clo, chi):
        prof = jnp.fft.fft(lax.complex(zr, zi), axis=1)
        cz = jnp.concatenate([prof[:, :pa], jnp.zeros((nJ, Na - keep2), prof.dtype), prof[:, Pf - na_:]], 1)
        d = dist_bins[None, :] * binm[:, None]
        w = jnp.where(d <= edge, 1.0, jnp.where(d >= edge + guard, 0.0, 0.5 + 0.5 * jnp.cos((d - edge) * (math.pi / guard))))
        A2 = jnp.fft.ifft(cz * (w * live[None, :] * scale), axis=1)
        ib, fr = _positions(jnp, base, lo, s, c2, pb2)
        o = _dense(jnp, lax, jnp.real(A2), jnp.imag(A2), ib, fr, Rw2, Q.TAPS, wpoly)
        col = jnp.arange(W2, dtype=jnp.int32)[None, :]
        m = (col >= clo[:, None]) & (col < chi[:, None])
        return jnp.where(m, o[0], 0.0), jnp.where(m, o[1], 0.0)
    B = store('B', rows2, L['b2'], W2)
    ntab = len(g['qtab'])
    for J0 in starts:
        kr = g['kr_min'] + g['dkr'] * (J0 + np.arange(nJ, dtype=np.float64))
        kn = _knots(_az_positions(g, kr, ja), pb2)
        binm = 2 * math.pi / (kr * abs(g['dts']) * Pf)
        # the raster's columns whose tangent lies in the table of the pulses' positions (the aperture and 16 pulses
        # beyond each end): the others are zero, as the host kernel leaves them
        u0, sl = (g['ka_lo'] / kr - g['tg0']) * g['tg_inv'], g['dka'] / kr * g['tg_inv']
        ends = np.sort(np.stack([-u0 / sl, (ntab - 1 - u0) / sl]), axis=0)
        c_lo = np.clip(np.ceil(ends[0] - 1e-9), 0, nka).astype(i32)
        c_hi = np.clip(np.ceil(ends[1] - 1e-9), 0, nka).astype(i32)
        o = run.chain([A.cut(n2, J0, shift1 if sheared else None),
                       ('pulses', (n2, nJ, A.rows, Pp, Pf, p0), pulses, ()),
                       ('azimuth', (nJ, Pf, Na, nblk2, Rw2, pb2), azimuth,
                        (dist_bins, live, edge, guard, scale, up(binm), wpoly) + tuple(jnp.asarray(v) for v in kn) + (jnp.asarray(c_lo), jnp.asarray(c_hi)))])
        B.put(J0, *o)
    B.close()
    del A
    jax.block_until_ready(o)
    timing['azimuth'] = timing.get('azimuth', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    # ------------------------------------------------------------------------------ 3. the raster's range transform
    # A block of azimuth wavenumbers, the range wavenumbers of each along a row: the rows of a sheared raster that
    # meet in one bin of the transform's period summed (a reshape and a sum); corrected, the signs that make the
    # padded transforms count from the image's center, and the band's center at index 0
    fold = L['fold']
    hy, hx = (g['hy'], nka // 2) if correct else (0, 0)
    rows3 = L['rows3']
    na, n3, starts = L['b3']
    sy = np.zeros(rows2, f32)
    sy[:nkr] = (-1.0) ** (np.arange(nkr) - hy) if correct else 1.0
    sxh = np.ones(rows3 + na, f32)
    if correct:
        sxh[:nka] = (-1.0) ** (np.arange(nka) - hx)
    sy = jnp.asarray(sy)

    def bins(zr, zi, sy, sxb, hy):
        w = sy[None, :] * sxb[:n3, None]
        pad = ((0, na - n3), (0, fold * Ny - rows2))
        f = lambda z: jnp.roll(jnp.pad(z[:, :rows2] * w, pad).reshape(na, fold, Ny).sum(1), -hy, axis=1)
        return f(zr), f(zi)

    def inverse(zr, zi):
        f = jnp.fft.ifft(lax.complex(zr, zi), axis=1)
        return jnp.real(f), jnp.imag(f)
    F = store('F', rows3, L['b3'], Ny)
    for a0 in starts:
        o = run.chain([B.cut(n3, a0), ('bins', (n3, na, B.rows, rows2, Ny, fold), bins, (sy, jnp.asarray(sxh[a0:a0 + na]), i32(hy))),
                       ('inverse transform', (na, Ny), inverse, ())])
        F.put(a0, *o)
    F.close()
    del B
    jax.block_until_ready(o)
    timing['range transform'] = timing.get('range transform', 0.0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    out = np.empty((ny, nx), c64 if complex_image else np.float32)
    Nx = ov * nfx
    if not correct:
        # 4. the azimuth transform of the picture's range bins, cut to its columns, with the carriers. The bins
        # wrap at the picture's center, so the blocks stop there
        bw, _, _ = _blocks_of(ny // 2, blk // (40 * Nx))
        y, x = (np.arange(ny) - ny / 2.0) * g['spy'], (np.arange(nx) - nx / 2.0) * g['spx']
        cyh = _cycles(g['kr_min'] * y / (2 * math.pi))
        cxd = jnp.asarray(_cycles(-g['ka_lo'] * x / (2 * math.pi)))
        for s_, e_ in ((0, ny // 2), (ny // 2, ny)):
            nv = min(bw, e_ - s_)

            def columns(zr, zi):
                pad = ((0, bw - nv), (0, Nx - nka))
                return jnp.pad(zr[:, :nka], pad), jnp.pad(zi[:, :nka], pad)

            def picture(zr, zi, cx, cy):
                t = jnp.roll(jnp.fft.fft(lax.complex(zr, zi), axis=1), nx // 2, axis=1)[:, :nx]
                if not complex_image:
                    return (jnp.abs(t),)
                c = _angle(jnp, cx[None, :] + cy[:, None])
                t = t * lax.complex(jnp.cos(c), jnp.sin(c))
                return (jnp.stack([jnp.real(t), jnp.imag(t)], -1),)
            for jb in _starts(e_ - s_, nv)[1]:
                jb += s_
                o = run.chain([F.cut(nv, (nfy - ny // 2 + jb) % nfy), ('columns', (nv, bw, F.rows, nka, Nx), columns, ()),
                               ('picture', (bw, Nx, nx, bool(complex_image)), picture,
                                (cxd, rows_of(cyh, slice(jb, jb + nv), bw, np.uint32)))])
                run.fetch(o[0], out[jb:jb + nv])
        run.flush()
        timing['image'] = timing.get('image', 0.0) + time.perf_counter() - t0
        return out
    # 4. the read at the apparent positions, in strips of the picture's rows (range pixels). A strip's rows of the
    # padded image are transformed along azimuth with each row shifted by the raster's shear (and brought to start
    # at the first column the strip reads), which leaves an image H that is smooth along its columns; H's columns
    # are read at the strip's rows (G), and G's rows at the picture's columns
    dist = g['dist']
    powers = tuple((int(a), int(b)) for a, b in dist['powers'])
    cx, cy = up(dist['cx']), up(dist['cy'])
    sx_, sy_ = float(dist['sx']), float(dist['sy'])
    ux, uy = ov / g['spx'], ov / g['spy']
    sig = float(g['shear'] * g['spy'] / g['spx'])                 # columns of the padded image per row, along the shear
    shx, shy = g['shift'][0] * ux, g['shift'][1] * uy             # the displacement's bounds, in padded samples
    half_x, half_y = -0.5 * ((ov * nx) % 2), -0.5 * ((ov * ny) % 2)
    ix0, iy0 = Nx // 2 - (ov * nx) // 2, Ny // 2 - (ov * ny) // 2  # the padded sample of pixel 0, less the half
    # the displacement's slopes over the picture, for the windows: BLOCK outputs span BLOCK ov (1 + slope) samples
    t = np.linspace(-1.0, 1.0, 33)
    Xg, Yg = np.meshgrid(t * nx / 2 * g['spx'] / sx_, t * ny / 2 * g['spy'] / sy_, indexing='ij')
    dxg = sum(c * Xg ** a * Yg ** b for c, (a, b) in zip(dist['cx'], powers))
    dyg = sum(c * Xg ** a * Yg ** b for c, (a, b) in zip(dist['cy'], powers))
    per_x, per_y = nx * g['spx'] / 32, ny * g['spy'] / 32
    dy_dx = float(np.abs(np.diff(dyg, axis=0)).max() / per_x)
    slope_y = float(np.abs(np.diff(dyg, axis=1)).max() / per_y) + abs(sig) * dy_dx * g['spx'] / g['spy']
    slope_x = float(np.abs(np.diff(dxg, axis=0)).max() / per_x) + abs(sig) * dy_dx * g['spx'] / g['spy']
    # the kernel: the host's on a plain raster, where the two passes then take the host kernel's own taps; on a
    # sheared raster the taps lie on the shifted image's samples and cannot be the host's, and a wider kernel
    # costs a product of blocks with their weights nothing
    taps, wfin = (SHEAR_TAPS, jnp.asarray(_kaiser_poly(SHEAR_KAISER))) if sig != 0.0 else (Q.LANCZOS, None)
    RwY, RwX = _window(ov * (1.0 + 1.5 * slope_y), taps), _window(ov * (1.0 + 1.5 * slope_x), taps)
    msy = _up(math.ceil(shy) + taps + 2, 16)
    nxp = _ladder(-(-nx // BLOCK)) * BLOCK
    # a strip: nj rows of the picture, nR rows of the padded image
    nj, nvj, jstarts = _blocks_of(ny, max(BLOCK, (blk // (32 * Nx) - 2 * msy) // ov), 1)
    nj = _up(nj, BLOCK)
    nR = min(ov * nj + 2 * msy, Ny)
    # H's columns: the columns of the padded image a strip reads, less the shear's drift from the image's center
    # row (the same shifted image whatever the strips, so a picture does not depend on their height)
    hw = ov * nx / 2.0 + shx + abs(sig) * (ov * nj / 2.0 + shy + 1) + taps + 4
    Nm = _ladder(-(-int(math.ceil(2 * hw) + 2) // BLOCK)) * BLOCK
    Rg = Ny // 2
    inv = f32(1.0 / (ux * sx_))
    xn = up((np.arange(nxp) - nx / 2.0) * g['spx'] / sx_)
    ac = jnp.asarray((np.arange(nka) - hx).astype(np.int64).astype(np.uint32))
    kr_c, ka_c = g['kr_min'] + hy * g['dkr'], g['ka_lo'] + hx * g['dka']
    cxx = up(np.modf(-ka_c * (np.arange(nxp) - nx / 2.0) * g['spx'] / (2 * math.pi))[0])
    krc, kac = f32(kr_c / (2 * math.pi)), f32(ka_c / (2 * math.pi))
    nm, _, mstarts = _blocks_of(Nm, blk // (24 * nR + 16 * nj * RwY), BLOCK)
    njc, nvc, _ = _blocks_of(nvj, blk // (24 * Nm + 16 * nxp * RwX))
    tiles = -(-Nm // Nx)

    def rows(zr, zi, ac, bq, hx):
        c = _angle(jnp, ac[None, :] * bq[:, None])
        cs, sn = jnp.cos(c), jnp.sin(c)
        zr, zi = zr[:, :nka], zi[:, :nka]
        pad = ((0, 0), (0, Nx - nka))
        return (jnp.roll(jnp.pad(zr * cs - zi * sn, pad), -hx, axis=1), jnp.roll(jnp.pad(zr * sn + zi * cs, pad), -hx, axis=1))

    def transform(zr, zi):
        h = jnp.fft.fft(lax.complex(zr, zi), axis=1)
        h = jnp.concatenate([h] * tiles, 1)[:, :Nm] if tiles > 1 or Nm < Nx else h
        return jnp.real(h), jnp.imag(h)

    def to_rows(zr, zi, am, bj, yn, iyr, cx, cy, inv, sig, ux, uy, half):
        X0, Y = am[:, None] + bj[None, :], yn[None, :]
        X = X0
        for _ in range(2):                           # the picture's column that lies on this column of H at this row
            X = X0 + (sig * uy * _poly(jnp, cy, powers, X, Y) - ux * _poly(jnp, cx, powers, X, Y)) * inv
        ib, fr = _split(jnp, iyr[None, :], _poly(jnp, cy, powers, X, Y) * uy + half)
        return _dense(jnp, lax, zr, zi, ib, fr, RwY, taps, wfin)

    def to_columns(zr, zi, xn, yn, ixm, qi, qf, cx, cy, sig, ux, uy, half, cxx, cyy, krc, kac):
        X, Y = xn[None, :], yn[:, None]
        dx, dy = _poly(jnp, cx, powers, X, Y), _poly(jnp, cy, powers, X, Y)
        ib, fr = _split(jnp, ixm[None, :] + qi[:, None], qf[:, None] + dx * ux - sig * (dy * uy) + half)
        ar, ai = _dense(jnp, lax, zr, zi, ib, fr, RwX, taps, wfin)
        if not complex_image:
            return (jnp.sqrt(ar * ar + ai * ai),)
        c = cxx[None, :] + cyy[:, None] + (krc * dy - kac * dx)
        c = (c - jnp.floor(c)) * (2.0 * math.pi)
        cs, sn = jnp.cos(c), jnp.sin(c)
        return (jnp.stack([ar * cs - ai * sn, ar * sn + ai * cs], -1),)

    def strip_rows(zr, zi):
        pad = ((0, njc - nvc), (0, 0))
        return jnp.pad(zr[:, :Nm], pad), jnp.pad(zi[:, :Nm], pad)
    scal = lambda v: f32(v)
    for j0 in jstarts:
        jj = j0 + np.arange(nj)
        iya = ov * jj + iy0                                                       # the padded row of each picture row
        R0 = int(min(max(0, iya[0] - msy), Ny - nR))
        m_lo = int(math.floor(Nx / 2.0 - sig * (iya[nj // 2] - Rg) - hw))         # the first column of H, of the padded image's
        bq = jnp.asarray(_cycles(-(m_lo + sig * (R0 + np.arange(nR) - Rg)) / Nx))
        yn = ((jj - ny / 2.0) * g['spy'] / sy_).astype(f32)
        am = ((np.arange(Nm) + m_lo - Nx / 2.0) / (ux * sx_)).astype(f32)         # H's columns, normalised (at the center row)
        bj = up(sig * (iya - Rg) / (ux * sx_))
        ixm = jnp.asarray((ov * np.arange(nxp) + ix0 - m_lo).astype(i32))
        q = -sig * (iya - Rg)
        qi, qf = np.floor(q).astype(i32), (q - np.floor(q)).astype(f32)
        cyy = np.modf(kr_c * (jj - ny / 2.0) * g['spy'] / (2 * math.pi))[0].astype(f32)
        H = _Device(run, nR, nR, [0], Nm)
        H.put(0, *run.chain([F.cut(nR, R0), ('rows', (nR, F.rows, nka, Nx), rows, (ac, bq, i32(hx))),
                             ('transform', (nR, Nx, Nm), transform, ())]))
        H.close()
        G = _Device(run, Nm, nm, mstarts, nj)
        for m0 in mstarts:
            G.put(m0, *run.chain([H.cut(nm, m0), ('to rows', (nm, nR, nj, RwY, taps, powers), to_rows,
                                                  (jnp.asarray(am[m0:m0 + nm]), bj, jnp.asarray(yn), jnp.asarray((iya - R0).astype(i32)), cx, cy,
                                                   inv, scal(sig), scal(ux), scal(uy), scal(half_y)))]))
        G.close()
        del H
        for c0 in _starts(nvj, nvc)[1]:
            sl = slice(c0, c0 + nvc)
            o = run.chain([G.cut(nvc, c0), ('strip rows', (nvc, njc, G.rows), strip_rows, ()),
                           ('to columns', (njc, Nm, nxp, RwX, taps, powers, bool(complex_image)), to_columns,
                            (xn, rows_of(yn, sl, njc, f32), ixm, rows_of(qi, sl, njc, i32), rows_of(qf, sl, njc, f32), cx, cy,
                             scal(sig), scal(ux), scal(uy), scal(half_x), cxx, rows_of(cyy, sl, njc, f32), krc, kac))])
            n = min(nvc, ny - (j0 + c0))
            run.fetch(o[0], out[j0 + c0:j0 + c0 + n])
        del G
    run.flush()
    timing['correction'] = timing.get('correction', 0.0) + time.perf_counter() - t0
    return out
