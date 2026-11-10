"""resample and the per-pixel work of planes on an NVIDIA GPU (CuPy): the image stays on the device.

The interpolation is resample_cpu.cpp's as a CUDA kernel, one thread per output sample: the position from the knot
map in double (positions reach tens of thousands of samples and the table has 2048 phases, more than float32
holds), then the taps of the nearest table row. Lines are oversampled by cuFFT in blocks sized to the free device
memory (_xp.cuda_block), with one plan per block shape. A pass along axis 0 reads and writes the image by strides,
so only the block being oversampled is ever transposed. The source, the intermediate image of a two-pass warp and
the result may each be on the host instead, as a device with less memory than the images needs: their lines then
cross in the same blocks, and the values are the same.
"""
import numpy as np
import cupy as cp

from . import _xp, resample

_SRC = r'''
// Bilinear knot map: li, lw [L] the knot interval and weight of each line, oj, ow [m] of each output sample,
// v [na, nb] the positions at the knots (resample.KnotMap, the arithmetic of resample_cpu.cpp).
static __device__ inline double knot(const double* v, int nb, int i0, double wi, int j0, double w)
{
  const double* a = v + (long long)i0 * nb + j0;
  const double r0 = a[0] * (1.0 - wi) + a[nb] * wi, r1 = a[1] * (1.0 - wi) + a[nb + 1] * wi;
  return r0 * (1.0 - w) + r1 * w;
}

// Sample i of line l of the source is at u[ch * (ubase + l * usl + i * usi)] (ch floats: 1 real, 2 complex),
// 0 <= i < nu, zero outside; output j of line l goes to out[ch * (obase + l * osl + j * osj)]. A position x
// (samples of the line before oversampling) lies at x * scale + offset in u; outside [0, nsrc - 1] the output
// is zero. pos [L, m] when haspos, else the knot map.
extern "C" __global__
void resample_lines(const float* u, long long ubase, long long usl, long long usi, long long nu, int ch,
                    const double* pos, int haspos, const int* li, const double* lw, const int* oj, const double* ow,
                    const double* v, int nb, double scale, double offset, double nsrc, long long L, long long m,
                    const float* tab, int nphase, int taps, float* out, long long obase, long long osl, long long osj)
{
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L * m) return;
  const long long l = idx / m, j = idx - l * m;
  double x = haspos ? pos[idx] : knot(v, nb, li[l], lw[l], oj[j], ow[j]);
  if (x < 0.0 && x > -1e-9) x = 0.0;                           // within rounding of an end: on it (resample_cpu.cpp)
  else if (x > nsrc - 1.0 && x < nsrc - 1.0 + 1e-9) x = nsrc - 1.0;
  float* o = out + ch * (obase + l * osl + j * osj);
  float ar = 0.f, ai = 0.f;
  if (x >= 0.0 && x <= nsrc - 1.0) {
    const double p = x * scale + offset, fl = floor(p);
    const long long i0 = (long long)fl - taps / 2 + 1;
    const float* w = tab + llrint((p - fl) * nphase) * taps;
    const float* s = u + ch * (ubase + l * usl);
    for (int t = 0; t < taps; ++t) {
      const long long i = i0 + t;
      if (i < 0 || i >= nu) continue;
      ar += w[t] * s[ch * i * usi];
      if (ch == 2) ai += w[t] * s[ch * i * usi + 1];
    }
  }
  o[0] = ar;
  if (ch == 2) o[1] = ai;
}

// img [L, m] (ch floats per sample) set to zero where the source pixel (x, y) of the knot maps vx, vy is outside
// [xlo, xhi] x [ylo, yhi], or where valid [nx, ny] is zero at the nearest pixel (planes.to_slant's mask).
extern "C" __global__
void zero_outside(float* img, long long L, long long m, int ch, const int* li, const double* lw, const int* oj,
                  const double* ow, const double* vx, const double* vy, int nb, double xlo, double xhi, double ylo,
                  double yhi, const unsigned char* valid, int hasvalid, long long nx, long long ny)
{
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L * m) return;
  const long long l = idx / m, j = idx - l * m;
  const double x = knot(vx, nb, li[l], lw[l], oj[j], ow[j]), y = knot(vy, nb, li[l], lw[l], oj[j], ow[j]);
  bool ok = x >= xlo && x <= xhi && y >= ylo && y <= yhi;
  if (ok && hasvalid) {
    const long long ix = min(max((long long)rint(x), 0LL), nx - 1), iy = min(max((long long)rint(y), 0LL), ny - 1);
    ok = valid[ix * ny + iy] != 0;
  }
  if (!ok) for (int c = 0; c < ch; ++c) img[ch * idx + c] = 0.f;
}
'''

_KERN = {}


def _kernel(name):
    key = (name, cp.cuda.Device().id)
    if key not in _KERN:
        if 'module' not in _KERN:
            _KERN['module'] = cp.RawModule(code=_SRC, options=('--std=c++14',))
        _KERN[key] = _KERN['module'].get_function(name)
    return _KERN[key]


def _launch(kern, n, args):
    kern((int(-(-n // 256)),), (256,), args)


def host_empty(shape, dt):
    """An empty host array that blocks cross to and from quickly: page-locked memory of CuPy's pinned pool, as the
    image a former returns (FASTSAR_PINNED=0: an ordinary array)."""
    from . import memory
    n = int(np.prod(shape))
    if n == 0 or memory._env_choice('FASTSAR_PINNED', '1', {'0', '1'}) == '0':
        return np.empty(shape, dt)
    return np.frombuffer(memory._pinned(n * np.dtype(dt).itemsize), dt, n).reshape(shape)


def upload(a):
    """A C-contiguous host array as a new device array, copied from where it is (cupy.asarray copies a large array
    into a page-locked block of its own size first) and before this returns (ndarray.set leaves a copy from
    page-locked memory in flight, and the caller goes on to write the host block)."""
    d = cp.empty(a.shape, a.dtype)
    if a.size:
        d.data.copy_from_host(a.ctypes.data, a.nbytes)
    return d


def _cells(km, L):
    """A knot map's per-line and per-sample intervals and weights, and its values, on the device."""
    i0, wi = km.cells(slice(0, L))
    return (cp.asarray(i0.astype(np.int32)), cp.asarray(wi), cp.asarray(km.j0.astype(np.int32)), cp.asarray(km.wj),
            cp.asarray(km.v))


def _oversample(z, pad, n2, plans):
    """Lines z [L, n2] (complex64, or float32 by the real transforms) oversampled `pad` times, without the factor
    1 / n2 (the table carries it). plans: dict of the block shape's cuFFT plans, filled here."""
    import cupyx.scipy.fft as cfft
    L, h, N, real = z.shape[0], n2 // 2, pad * n2, z.dtype.kind != 'c'
    if 'f' not in plans:
        plans['f'] = cfft.get_fft_plan(z, axes=1, value_type='R2C' if real else 'C2C')
    F = cfft.rfft(z, axis=1, overwrite_x=True, plan=plans['f']) if real else \
        cfft.fft(z, axis=1, overwrite_x=True, plan=plans['f'])
    del z
    G = cp.zeros((L, N // 2 + 1 if real else N), np.complex64)
    G[:, :h] = F[:, :h]
    if not real:
        G[:, N - h:] = F[:, n2 - h:]
    if n2 % 2:
        G[:, h] = F[:, h]
    elif real:
        G[:, h] = 0.5 * F[:, h]
    else:                                        # the Nyquist bin, split between the two halves
        G[:, N - h] *= 0.5
        G[:, h] = G[:, N - h]
    del F
    if 'b' not in plans:
        plans['b'] = cfft.get_fft_plan(G, shape=(N,), axes=1, value_type='C2R') if real else \
            cfft.get_fft_plan(G, axes=1, value_type='C2C')
    if real:
        return cfft.irfft(G, n=N, axis=1, norm='forward', overwrite_x=True, plan=plans['b'])
    return cfft.ifft(G, axis=1, norm='forward', overwrite_x=True, plan=plans['b'])


def _pass(src, pos, m, axis, out, kernel, taps, pad, after=None):
    """One pass along `axis` of src: axis 1, out[l, j] = src[l, pos[l, j]] (src [L, n], out [L, m]); axis 0,
    out[j, l] = src[pos[l, j], l] (src [n, L], out [m, L]). pos: a KnotMap, a function of a slice of lines or an
    array [L, m] (host or device). out: C-contiguous float32 or complex64, as src is real or complex. src and out
    may each be a NumPy array, whose blocks of lines then cross to the device and back. after(block, a, b): called
    on the device with lines a ... b - 1 of the result of a pass along axis 1, before they reach a host out."""
    real = src.dtype.kind != 'c'
    ch, dt = (1, np.float32) if real else (2, np.complex64)
    hs, ho = not isinstance(src, cp.ndarray), not isinstance(out, cp.ndarray)
    src = np.ascontiguousarray(src, dt) if hs else cp.ascontiguousarray(src, dt)
    L, n = (src.shape[0], src.shape[1]) if axis == 1 else (src.shape[1], src.shape[0])
    tab, pad, guard, n2 = resample._plan(n, kernel, taps, pad)
    nphase, taps = tab.shape[0] - 1, tab.shape[1]
    tabd = cp.asarray(tab if pad == 1 else tab * np.float32(1.0 / n2))
    kern = _kernel('resample_lines')
    knots = isinstance(pos, resample.KnotMap)
    dummy = cp.zeros(1, np.float64)
    if knots:
        li, lw, oj, ow, v = _cells(pos, L)
        nbk = int(v.shape[1])
    else:
        li = lw = oj = ow = v = dummy
        nbk = 0
    # device memory per line of a block (bytes): the positions, the line with its spectra and cuFFT's work area,
    # and the lines of a host array in flight
    line = 8 * m * (not knots and not isinstance(pos, cp.ndarray))
    if pad > 1:
        line += 8 * n2 * (2 + 3 * pad)
    line += 4 * ch * (n * hs + m * ho)
    nb = L if line == 0 else _xp.cuda_rows(L, line, 'resampling on the GPU')
    if pad > 1:
        nb = int(max(1, min(nb, (1 << 26) // (pad * n2))))                  # 512 MB of oversampled lines at a time
    plans = {}
    # lines along axis 0 of a host array cross through a page-locked block [n or m, lines]
    sin = host_empty((n * nb,), dt) if hs and axis == 0 else None
    sout = host_empty((m * nb,), dt) if ho and axis == 0 else None
    for a in range(0, L, nb):
        b = min(L, a + nb)
        if b - a != nb:
            plans = {}
        if knots:
            q, lia, lwa = dummy, li[a:b], lw[a:b]
        else:
            q, lia, lwa = cp.asarray(pos(slice(a, b)) if callable(pos) else pos[a:b]), li, lw
            q = cp.ascontiguousarray(q, np.float64)
        blk = None
        if hs:                                   # lines a ... b - 1 of a host image: [b - a, n], or [n, b - a] (axis 0)
            if axis == 1:
                blk = upload(src[a:b])
            else:
                st = sin[:n * (b - a)].reshape(n, b - a)
                np.copyto(st, src[:, a:b])
                blk = upload(st)
        if pad > 1:
            z = cp.zeros((b - a, n2), np.float32 if real else np.complex64)
            z[:, guard:guard + n] = (blk if axis == 1 else blk.T) if hs else (src[a:b] if axis == 1 else src[:, a:b].T)
            blk = None
            u = _oversample(z, pad, n2, plans)
            del z
            ua = (u, 0, pad * n2, 1, pad * n2)
        elif hs:
            u = blk
            ua = (u, 0, n, 1, n) if axis == 1 else (u, 0, 1, b - a, n)
        else:                                    # the kernel reads the image itself, zero outside each line
            u = src
            ua = (u, a * n, n, 1, n) if axis == 1 else (u, a, 1, L, n)
        if ho:
            ob = cp.empty((b - a, m) if axis == 1 else (m, b - a), dt)
            oa = (ob, 0, m, 1) if axis == 1 else (ob, 0, 1, b - a)
        else:
            ob, oa = None, ((out, a * m, m, 1) if axis == 1 else (out, a, 1, L))
        _launch(kern, (b - a) * m, (ua[0], *map(np.int64, ua[1:]), np.int32(ch), q, np.int32(not knots), lia, lwa,
                                    oj, ow, v, np.int32(nbk), np.float64(pad), np.float64(guard * pad if pad > 1 else 0),
                                    np.float64(n), np.int64(b - a), np.int64(m), tabd, np.int32(nphase),
                                    np.int32(taps), oa[0], *map(np.int64, oa[1:])))
        del u, q, blk
        if ho:
            if after is not None:
                after(ob, a, b)
            if axis == 1:
                ob.get(out=out[a:b])
            else:
                st = sout[:m * (b - a)].reshape(m, b - a)
                ob.get(out=st)
                out[:, a:b] = st
            del ob
    if not ho and after is not None:
        after(out, 0, L)
    return out


def _empty(shape, dt, where, what):
    """A new array on the host ('host') or on the device; MemoryError when the device cannot hold it."""
    if where == 'host':
        return host_empty(shape, dt)
    try:
        return cp.empty(shape, dt)
    except cp.cuda.memory.OutOfMemoryError:
        from .memory import cuda_free
        need = int(np.prod(shape)) * np.dtype(dt).itemsize
        raise MemoryError(f'{what} takes {need / 1e9:.2f} GB of device memory and {cuda_free() / 1e9:.2f} GB are '
                          'free: keep the image on the host (a NumPy array), or use fastsar.process_cphd, which '
                          'streams it through the device') from None


def _check_out(out, shape, dt):
    if out is None:
        return cp.empty(shape, dt)
    if not isinstance(out, cp.ndarray) or out.shape != shape or out.dtype != dt or not out.flags.c_contiguous:
        raise ValueError(f'out must be a C-contiguous CuPy array {shape} of {np.dtype(dt).name}')
    return out


def lines(src, pos, m, *, kernel='sinc', taps=8, pad=None, out=None):
    """resample.lines for a CuPy array."""
    dt = np.float32 if src.dtype.kind != 'c' else np.complex64
    return _pass(src, pos, m, 1, _check_out(out, (src.shape[0], m), dt), kernel, taps, pad)


def warp(src, pos0, pos1, shape, *, kernel='sinc', taps=8, pad=None, transposed=False, mask=None, place=None):
    """resample.warp for a CuPy array. transposed: the warp of src.T, returned transposed ([shape[1], shape[0]]),
    with no transposed copy of the image. mask: (posx, (xlo, xhi), (ylo, yhi), valid): the output is zero where
    the source pixel (posx, pos1), knot maps on the same knots, is outside these bounds, or where valid [nx, ny]
    (or None) is False at the nearest pixel.
    place: where the intermediate image of the two passes and the result are made, ('device' or 'host') for
    each; a host array crosses the device in blocks of lines, and src may be one. Default: the result on the
    device, the intermediate there when both fit in three quarters of _xp.cuda_budget(), else on the host.
    src: or a list that holds it alone, emptied here: an image nothing else refers to is then freed after the
    first pass, and the device has its memory for the result."""
    m0, m1 = shape
    if isinstance(src, list):
        src = src.pop()
    dt = np.float32 if src.dtype.kind != 'c' else np.complex64
    item = np.dtype(dt).itemsize
    smid, sout = ((src.shape[0], m0), (m1, m0)) if transposed else ((m0, src.shape[1]), (m0, m1))
    auto = place is None
    if auto:
        fits = item * (smid[0] * smid[1] + sout[0] * sout[1]) <= 0.75 * _xp.cuda_budget()
        place = ('device' if fits else 'host', 'device')
    mid = _empty(smid, dt, place[0], 'the intermediate image of the resampling')
    mid = _pass(src, pos0, m0, 1 if transposed else 0, mid, kernel, taps, pad)
    del src
    out = _empty(sout, dt, place[1], 'the resampled image')
    after = None
    if mask is not None and not transposed:
        posx, (xlo, xhi), (ylo, yhi), valid = mask
        if not (np.array_equal(posx.lk, pos1.lk) and np.array_equal(posx.ok, pos1.ok)):
            raise ValueError('the maps of a mask must share their knots')
        li, lw, oj, ow, vx = _cells(posx, m0)
        vy = cp.asarray(pos1.v)
        if valid is None:
            vd, nx, ny = cp.zeros(1, np.uint8), 0, 0
        else:
            vd = cp.ascontiguousarray(cp.asarray(valid), np.uint8)
            nx, ny = vd.shape

        def after(blk, a, b):
            _launch(_kernel('zero_outside'), (b - a) * m1,
                    (blk, np.int64(b - a), np.int64(m1), np.int32(1 if dt == np.float32 else 2), li[a:b], lw[a:b], oj,
                     ow, vx, vy, np.int32(vx.shape[1]), np.float64(xlo), np.float64(xhi), np.float64(ylo),
                     np.float64(yhi), vd, np.int32(valid is not None), np.int64(nx), np.int64(ny)))
    out = _pass(mid, pos1, m1, 0 if transposed else 1, out, kernel, taps, pad, after)
    del mid
    if auto and place[0] == 'host':             # its page-locked blocks back to the host (process_cphd frees its own)
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return out


_DERAMP = []


def deramp(img, w0, a, b, r0, k):
    """planes.deramp's multiplication, in place: img [nx, ny] *= exp(j k (|w0 + i a + j b| - r0)) in double, w0 [3]
    the first pixel less the antenna, a and b [3] the steps along the two axes (m). A NumPy image crosses the
    device in blocks of rows (the same values: a block's kernel counts its rows from the image's first)."""
    if not _DERAMP:
        _DERAMP.append(cp.ElementwiseKernel(
            'int64 ny, int64 row0, float64 wx, float64 wy, float64 wz, float64 ax, float64 ay, float64 az, float64 bx, '
            'float64 by, float64 bz, float64 r0, float64 k', 'T v',
            '''const double r = (double)(i / ny + row0), c = (double)(i % ny);
            const double x = wx + r * ax + c * bx, y = wy + r * ay + c * by, z = wz + r * az + c * bz;
            double sn, cs;
            sincos(k * (sqrt(x * x + y * y + z * z) - r0), &sn, &cs);
            const double re = v.real(), im = v.imag();
            v = T(re * cs - im * sn, re * sn + im * cs);''', 'fastsar_deramp'))
    args = [float(x) for x in (*w0, *a, *b, r0, k)]
    if isinstance(img, cp.ndarray):
        _DERAMP[0](np.int64(img.shape[1]), np.int64(0), *args, img)
        return img
    nx, ny = img.shape
    nb = _xp.cuda_rows(nx, 8 * ny, 'baseband on the GPU')
    for i in range(0, nx, nb):
        blk = upload(np.ascontiguousarray(img[i:i + nb]))
        _DERAMP[0](np.int64(ny), np.int64(i), *args, blk)
        if img.flags.c_contiguous:
            blk.get(out=img[i:i + nb])
        else:
            img[i:i + nb] = blk.get()
    return img
