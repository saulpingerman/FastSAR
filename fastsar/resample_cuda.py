"""resample and the per-pixel work of planes on an NVIDIA GPU (CuPy): the image stays on the device.

The interpolation is resample_cpu.cpp's as a CUDA kernel, one thread per output sample: the position from the knot
map in double (positions reach tens of thousands of samples and the table has 2048 phases, more than float32
holds), then the taps of the nearest table row. Lines are oversampled by cuFFT in blocks sized to the free device
memory, with one plan per block shape. A pass along axis 0 reads and writes the image by strides, so only the
block being oversampled is ever transposed.
"""
import numpy as np
import cupy as cp

from . import resample

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


def _free():
    """Bytes a new allocation can take: free on the device, and free in CuPy's pool."""
    return int(cp.cuda.Device().mem_info[0]) + int(cp.get_default_memory_pool().free_bytes())


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


def _pass(src, pos, m, axis, out, kernel, taps, pad):
    """One pass along `axis` of src: axis 1, out[l, j] = src[l, pos[l, j]] (src [L, n], out [L, m]); axis 0,
    out[j, l] = src[pos[l, j], l] (src [n, L], out [m, L]). pos: a KnotMap, a function of a slice of lines or an
    array [L, m] (host or device). out: C-contiguous float32 or complex64, as src is real or complex."""
    real = src.dtype.kind != 'c'
    ch, dt = (1, np.float32) if real else (2, np.complex64)
    src = cp.ascontiguousarray(src, dt)
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
    line = 8 * m * (not knots and not isinstance(pos, cp.ndarray))          # device memory per line of a block (bytes)
    if pad > 1:
        line += 8 * n2 * (2 + 3 * pad)                                      # the line, its spectra and cuFFT's work area
    nb = L if line == 0 else int(max(1, min(L, (_free() // 3) // line)))
    if pad > 1:
        nb = int(max(1, min(nb, (1 << 26) // (pad * n2))))                  # 512 MB of oversampled lines at a time
    plans = {}
    for a in range(0, L, nb):
        b = min(L, a + nb)
        if b - a != nb:
            plans = {}
        if knots:
            q, lia, lwa = dummy, li[a:b], lw[a:b]
        else:
            q, lia, lwa = cp.asarray(pos(slice(a, b)) if callable(pos) else pos[a:b]), li, lw
            q = cp.ascontiguousarray(q, np.float64)
        if pad > 1:
            z = cp.zeros((b - a, n2), np.float32 if real else np.complex64)
            z[:, guard:guard + n] = src[a:b] if axis == 1 else src[:, a:b].T
            u = _oversample(z, pad, n2, plans)
            del z
            ua = (u, 0, pad * n2, 1, pad * n2)
        else:                                    # the kernel reads the image itself, zero outside each line
            u = src
            ua = (u, a * n, n, 1, n) if axis == 1 else (u, a, 1, L, n)
        oa = (out, a * m, m, 1) if axis == 1 else (out, a, 1, L)
        _launch(kern, (b - a) * m, (ua[0], *map(np.int64, ua[1:]), np.int32(ch), q, np.int32(not knots), lia, lwa,
                                    oj, ow, v, np.int32(nbk), np.float64(pad), np.float64(guard * pad if pad > 1 else 0),
                                    np.float64(n), np.int64(b - a), np.int64(m), tabd, np.int32(nphase),
                                    np.int32(taps), oa[0], *map(np.int64, oa[1:])))
        del u, q
    return out


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


def warp(src, pos0, pos1, shape, *, kernel='sinc', taps=8, pad=None, transposed=False, mask=None):
    """resample.warp for a CuPy array. transposed: the warp of src.T, returned transposed ([shape[1], shape[0]]),
    with no transposed copy of the image. mask: (posx, (xlo, xhi), (ylo, yhi), valid): the output is zero where
    the source pixel (posx, pos1), knot maps on the same knots, is outside these bounds, or where valid [nx, ny]
    (or None) is False at the nearest pixel."""
    m0, m1 = shape
    dt = np.float32 if src.dtype.kind != 'c' else np.complex64
    if transposed:
        mid = _pass(src, pos0, m0, 1, cp.empty((src.shape[0], m0), dt), kernel, taps, pad)
        return _pass(mid, pos1, m1, 0, cp.empty((m1, m0), dt), kernel, taps, pad)
    mid = _pass(src, pos0, m0, 0, cp.empty((m0, src.shape[1]), dt), kernel, taps, pad)
    out = _pass(mid, pos1, m1, 1, cp.empty((m0, m1), dt), kernel, taps, pad)
    del mid
    if mask is not None:
        posx, (xlo, xhi), (ylo, yhi), valid = mask
        if not (np.array_equal(posx.lk, pos1.lk) and np.array_equal(posx.ok, pos1.ok)):
            raise ValueError('the maps of a mask must share their knots')
        li, lw, oj, ow, vx = _cells(posx, m0)
        if valid is None:
            vd, nx, ny = cp.zeros(1, np.uint8), 0, 0
        else:
            vd = cp.ascontiguousarray(cp.asarray(valid), np.uint8)
            nx, ny = vd.shape
        _launch(_kernel('zero_outside'), m0 * m1,
                (out, np.int64(m0), np.int64(m1), np.int32(1 if dt == np.float32 else 2), li, lw, oj, ow, vx,
                 cp.asarray(pos1.v), np.int32(vx.shape[1]), np.float64(xlo), np.float64(xhi), np.float64(ylo),
                 np.float64(yhi), vd, np.int32(valid is not None), np.int64(nx), np.int64(ny)))
    return out


_DERAMP = []


def deramp(img, w0, a, b, r0, k):
    """planes.deramp's multiplication, in place: img [nx, ny] *= exp(j k (|w0 + i a + j b| - r0)) in double, w0 [3]
    the first pixel less the antenna, a and b [3] the steps along the two axes (m)."""
    if not _DERAMP:
        _DERAMP.append(cp.ElementwiseKernel(
            'int64 ny, float64 wx, float64 wy, float64 wz, float64 ax, float64 ay, float64 az, float64 bx, float64 by, '
            'float64 bz, float64 r0, float64 k', 'T v',
            '''const double r = (double)(i / ny), c = (double)(i % ny);
            const double x = wx + r * ax + c * bx, y = wy + r * ay + c * by, z = wz + r * az + c * bz;
            double sn, cs;
            sincos(k * (sqrt(x * x + y * y + z * z) - r0), &sn, &cs);
            const double re = v.real(), im = v.imag();
            v = T(re * cs - im * sn, re * sn + im * cs);''', 'fastsar_deramp'))
    _DERAMP[0](np.int64(img.shape[1]), *(float(x) for x in (*w0, *a, *b, r0, k)), img)
    return img
