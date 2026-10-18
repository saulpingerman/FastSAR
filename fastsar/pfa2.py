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


_GATHER_SRC = r"""
#include <cuda_fp16.h>
#ifndef TAPS
#define TAPS 6
#endif
#ifndef NPOW
#define NPOW 10
#endif
#ifndef DEG
#define DEG 3
#endif
#define HALF_T (TAPS / 2)
__device__ __forceinline__ float sincf_(float d) {
  if (fabsf(d) < 1e-6f) return 1.f;
  const float a = 3.14159265358979f * d;
  return sinf(a) / a;
}
// one thread per output pixel (i along e1 / azimuth, fastest; c along e2 / range): the apparent position from the
// distortion polynomial in float64, the Lanczos weights in float32, TAPS x TAPS reads of the baseband block
extern "C" __global__ void pfa_gather(const float2* base, int span, int Nx2, int Ny2, int rb0,
                                      int h, int n, int i0, int jb, int nx, int ny, double spx, double spy, double ox, double oy,
                                      double ov, double dsx, double dsy, const double* cx, const double* cy, const int* pw,
                                      double kr_c, double ka_c, int sy_, int sx_, float norm, const float* scale,
                                      float2* out, long ostride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
  if (i >= h || c >= n) return;
  const double x = (i0 + i - nx / 2.0) * spx + ox, y = (jb + c - ny / 2.0) * spy + oy;
  const double xn = x / dsx, yn = y / dsy;
  double Xa = x, Ya = y, px[DEG + 1], py[DEG + 1];
  px[0] = 1.0; py[0] = 1.0;
  #pragma unroll
  for (int k = 1; k <= DEG; ++k) { px[k] = px[k-1] * xn; py[k] = py[k-1] * yn; }
  #pragma unroll
  for (int k = 0; k < NPOW; ++k) {
    const double t = px[pw[2*k]] * py[pw[2*k+1]];
    Xa += cx[k] * t; Ya += cy[k] * t;
  }
  const double fx = Xa / (spx / ov) + Nx2 / 2.0, fy = Ya / (spy / ov) + Ny2 / 2.0;
  const double fix = floor(fx), fiy = floor(fy);
  const int ix = (int)fix, iyl = (int)fiy - rb0;
  const float tx = (float)(fx - fix), ty = (float)(fy - fiy);
  float wx[TAPS], wy[TAPS];
  float sx = 0.f, sy = 0.f;
  #pragma unroll
  for (int a = 0; a < TAPS; ++a) {
    const float o = (float)(a - HALF_T + 1);
    const float dx = tx - o, dy = ty - o;
    wx[a] = sincf_(dx) * sincf_(dx / HALF_T); wy[a] = sincf_(dy) * sincf_(dy / HALF_T);
    sx += wx[a]; sy += wy[a];
  }
  float2 acc = make_float2(0.f, 0.f);
  #pragma unroll
  for (int a = 0; a < TAPS; ++a) {
    const int yy = min(max(iyl + a - HALF_T + 1, 0), span - 1);
    const float2* row = base + (long)yy * Nx2;
    const float wya = wy[a] / sy;
    #pragma unroll
    for (int b = 0; b < TAPS; ++b) {
      const int xx = min(max(ix + b - HALF_T + 1, 0), Nx2 - 1);
      const float2 v = row[xx];
      const float wgt = wya * (wx[b] / sx);
      acc.x = fmaf(wgt, v.x, acc.x); acc.y = fmaf(wgt, v.y, acc.y);
    }
  }
  double sn, cs;
  sincos(sy_ * kr_c * Ya + sx_ * ka_c * Xa, &sn, &cs);
  const float cr = (float)cs, ci = (float)sn, g = norm * scale[0];
  out[(long)i * ostride + c] = make_float2(g * (acc.x * cr - acc.y * ci), g * (acc.x * ci + acc.y * cr));
}
"""
_GATHER = {}

_RESAMPLE_SRC = r"""
#ifndef TAPS
#define TAPS 16
#endif
#define HALF_T (TAPS / 2)
#ifndef TABN
#define TABN 1024
#endif
// Kaiser-windowed sinc of TAPS taps (the kernel of pfa2._kernel) from a table of its values at TABN points per
// unit offset over [0, HALF_T], linearly interpolated (symmetric in d)
__device__ __forceinline__ float kaiser_sinc(float d, const float* tab) {
  const float ad = fabsf(d);
  if (ad >= HALF_T) return 0.f;
  const float u = ad * TABN;
  const int k = (int)u;
  return fmaf(tab[k + 1] - tab[k], u - k, tab[k]);
}
// rows: out[r, j] = row r of (re, im) [R, n] interpolated at index pos0[r] + j * step[r], j < n_out; outputs
// whose index lies outside (-0.5, n - 0.5) are zero (as the chirp-z transform zeroes them); weights normalised
extern "C" __global__ void resample_rows(const float2* x, int R, int n, const float* pos0, const float* step,
                                         int n_out, const float* tab, float2* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (j >= n_out || r >= R) return;
  const float pos = pos0[r] + j * step[r];
  float2 acc = make_float2(0.f, 0.f);
  if (pos > -0.5f && pos < n - 0.5f) {
    const int i0 = (int)floorf(pos) - HALF_T + 1;
    const float2* rr = x + (long)r * n;
    float w[TAPS], tot = 0.f;
    #pragma unroll
    for (int t = 0; t < TAPS; ++t) {
      const int i = i0 + t;
      w[t] = (i >= 0 && i < n) ? kaiser_sinc(pos - i, tab) : 0.f;
      tot += w[t];
    }
    const float inv = 1.f / fmaxf(tot, 1e-12f);
    #pragma unroll
    for (int t = 0; t < TAPS; ++t) {
      const int i = min(max(i0 + t, 0), n - 1);
      const float2 v = rr[i];
      acc.x = fmaf(w[t] * inv, v.x, acc.x); acc.y = fmaf(w[t] * inv, v.y, acc.y);
    }
  }
  out[(long)r * n_out + j] = acc;
}
// columns: out[m, j] = column m of A [R, M] (complex) interpolated along the rows at index pos0[m] + j * step[m];
// threads along m read consecutive addresses
extern "C" __global__ void resample_cols(const float2* A, int R, int M, const float* pos0, const float* step,
                                         int n_out, const float* tab, float2* out) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (m >= M || j >= n_out) return;
  const float pos = pos0[m] + j * step[m];
  float2 acc = make_float2(0.f, 0.f);
  if (pos > -0.5f && pos < R - 0.5f) {
    const int i0 = (int)floorf(pos) - HALF_T + 1;
    float w[TAPS], tot = 0.f;
    #pragma unroll
    for (int t = 0; t < TAPS; ++t) {
      const int i = i0 + t;
      w[t] = (i >= 0 && i < R) ? kaiser_sinc(pos - i, tab) : 0.f;
      tot += w[t];
    }
    const float inv = 1.f / fmaxf(tot, 1e-12f);
    #pragma unroll
    for (int t = 0; t < TAPS; ++t) {
      const int i = min(max(i0 + t, 0), R - 1);
      const float2 v = A[(long)i * M + m];
      acc.x = fmaf(w[t] * inv, v.x, acc.x); acc.y = fmaf(w[t] * inv, v.y, acc.y);
    }
  }
  out[(long)m * n_out + j] = acc;
}
"""
_RESAMPLE = {}


TABN = 1024          # table points per unit offset of the resampling kernel


def kernel_table(taps, beta):
    """The Kaiser-windowed sinc (_kernel) at TABN points per unit offset over [0, taps / 2], plus one guard point."""
    d = np.arange(TABN * (taps // 2) + 2) / TABN
    return _kernel(d, taps, beta).astype(np.float32)


def _resample_kernels(taps):
    """The CUDA direct-resampling kernels (rows and columns), compiled once per tap count."""
    import cupy as cp
    if taps not in _RESAMPLE:
        m = cp.RawModule(code=_RESAMPLE_SRC, options=(f'-DTAPS={taps}', f'-DTABN={TABN}'))
        _RESAMPLE[taps] = (m.get_function('resample_rows'), m.get_function('resample_cols'))
    return _RESAMPLE[taps]


def _direct_resampling(default=True):
    """Whether the two chirp-z transforms are replaced by direct Kaiser-sinc resampling of the windowed profiles
    (the CUDA or C++ kernels): the default on a GPU and the CPU (the Panama image 1.9 to 1.45 s on an L4, 10.2 to
    8.9 s on the c4d-highmem-16); FASTSAR_PFA_CZT=1 keeps the transforms, which the TPU always uses."""
    import os
    v = os.environ.get('FASTSAR_PFA_CZT')
    return default if v not in ('0', '1') else v == '0'


def _gather_kernel(taps, npow, deg):
    """The CUDA gather kernel of the final resampling (compiled once per taps, polynomial length and degree)."""
    import cupy as cp
    key = (taps, npow, deg)
    if key not in _GATHER:
        _GATHER[key] = cp.RawModule(code=_GATHER_SRC, options=(f'-DTAPS={taps}', f'-DNPOW={npow}', f'-DDEG={deg}')).get_function('pfa_gather')
    return _GATHER[key]


_CPU_LIB = None


def _cpu_gather_lib():
    """The C++ gather of the final resampling (pfa_cpu.cpp), compiled on first use as the other CPU kernels."""
    global _CPU_LIB
    if _CPU_LIB is None:
        import ctypes, os
        from ._build import shared_object, default_flags
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pfa_cpu.cpp')) as fh:
            src = fh.read()
        L = ctypes.CDLL(shared_object('pfa_cpu', src, default_flags() + ['-std=c++17']))
        f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
        i, d, fl = ctypes.c_int, ctypes.c_double, ctypes.c_float
        L.pfa_gather.argtypes = [f32, i, i, i, i, i, i, i, i, i, i, d, d, d, d, d, d, d, f64, f64, i32, i, i, d, d, i, i, fl, fl, i,
                                 ctypes.c_void_p, ctypes.c_long]
        L.pfa_resample_rows.argtypes = [f32, i, i, f32, f32, i, i, f32, i, f32]
        L.pfa_resample_cols.argtypes = [f32, i, i, f32, f32, i, i, f32, i, f32]
        _CPU_LIB = L
    return _CPU_LIB


def _gather_kernels_on():
    """FASTSAR_PFA_GATHER=0 keeps the JAX gather of the final resampling in place of the C++ or CUDA kernel."""
    import os
    return os.environ.get('FASTSAR_PFA_GATHER', '1') != '0'


def _cpu_kernels_available():
    """True when JAX runs on the CPU (the C++ kernels then apply)."""
    import jax
    return jax.default_backend() == 'cpu'


def _cuda_kernels_available():
    """True when JAX runs on a GPU and CuPy can run kernels on it."""
    try:
        import cupy  # noqa: F401
        import jax
        return jax.default_backend() == 'gpu'
    except Exception:
        return False


def _cpu_gather_available():
    return _gather_kernels_on() and _cpu_kernels_available()


def _cuda_gather_available():
    return _gather_kernels_on() and _cuda_kernels_available()


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


def geometry(col, nx, ny, spx, spy, taps=16, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), guard=300.0, kaiser=8.0, support='union'):
    """Host-side float64 setup. support: 'union' keeps every pulse's band (the trapezoidal support of backprojection, zeros
    where a pulse has no sample); 'inscribed' keeps the rectangle common to all pulses, a clean window at the cost of
    the bandwidth outside it (about half the range bandwidth in the ground plane of a squinted orbital spotlight). The image axes are x along track (cross range) and y ground range, as in the
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
    if not np.all(np.diff(t) > 0):
        raise ValueError("polar format needs the look angle in the image plane (from e2 toward e1) to change "
                         "monotonically from pulse to pulse, and it does not: e1 must lie along the track (azimuth) "
                         "and e2 across it (range), and the pulses must be in time order without repeated positions; "
                         "with another grid use algorithm='ffbp' or 'bp'")
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
    if support == 'inscribed':
        kr_min = 4.0 * math.pi * fa / C * ur_r.max()
        kr_max = 4.0 * math.pi * fb / C * ur_r.min()
        if kr_max <= kr_min:
            raise ValueError('polar format: no range band is common to all pulses (support=\'inscribed\'); use support=\'union\'')
    elif support == 'union':
        kr_min = 4.0 * math.pi * fa / C * ur_r.min()
        kr_max = 4.0 * math.pi * fb / C * ur_r.max()
    else:
        raise ValueError(f"support must be 'union' or 'inscribed', not {support!r}")
    nkr = int((kr_max - kr_min) / dkr) + 1
    if nkr > nfy:                                             # a band wider than the grid resolves: its center is kept
        import warnings
        warnings.warn(f'polar format: the range band ({nkr} samples) exceeds the {nfy}-sample raster of the grid; the central '
                      f'{nfy} are kept, so the range resolution is limited by the pixel spacing', UserWarning, stacklevel=3)
        kr_min += ((nkr - nfy) // 2) * dkr
        nkr = nfy
        kr_max = kr_min + (nkr - 1) * dkr
    kr = kr_min + dkr * np.arange(nkr)
    ta, tb = t0 + m * dt, t1 - m * dt
    kref = kr_min if support == 'inscribed' else kr_max      # the azimuth band common to the rows, or the widest row's
    ka_lo, ka_hi = kref * ta, kref * tb
    nka = int((ka_hi - ka_lo) / dka) + 1
    if nka > nfx:                                             # the aperture's azimuth band wider than the grid resolves
        import warnings
        warnings.warn(f'polar format: the azimuth band of the aperture ({nka} samples) exceeds the {nfx}-sample raster of '
                      f'the grid; the central {nfx} are kept, so the azimuth resolution is limited by the pixel spacing '
                      '(a finer spacing or a shorter aperture keeps the whole band)', UserWarning, stacklevel=3)
        ka_lo += ((nka - nfx) // 2) * dka
        nka = nfx
    scale = C / (4.0 * math.pi * ur_r * col.df)
    extent_r = C / (2.0 * col.df)                            # period of a range profile (m)
    extent_a = 2.0 * math.pi / (kr * dt)                     # period of the azimuth profile of each kr row (m)
    edge = (ny * spy / 2.0, nx * spx / 2.0)
    if not (extent_r > 2 * (edge[0] + guard) and extent_a.min() > 2 * (edge[1] + guard)):
        raise ValueError(f'polar format: the scene ({nx * spx:.1f} by {ny * spy:.1f} m) plus pfa_guard={guard:g} m on each '
                         f'side exceeds the unambiguous extent ({extent_a.min():.1f} m in azimuth, {extent_r:.1f} m in '
                         'range); use a smaller pfa_guard or grid')
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
    phase history referenced to the scene center as prepared here, the two signs are opposite. Pixel i of an
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
        """Window over a profile of n samples whose period is `extent` meters (per row, [R]): one inside
        `edge` meters of the center, a raised cosine to zero over the next `guard` meters."""
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

    if _cpu_kernels_available() and _direct_resampling():
        lib = _cpu_gather_lib()
        taps_p = int(geo['taps'])
        tab = kernel_table(taps_p, geo['kaiser'])

        @jax.jit
        def pulses_windowed(S_re, S_im, W, ext):
            x = cx(*resample_pulses(S_re, S_im, W))                                             # [Pp, K]
            g = jnp.fft.fftshift(jnp.fft.ifft(x, axis=1), axes=1) * taper(x.shape[1], ext, edge_y)
            return jnp.fft.fft(jnp.fft.ifftshift(g, axes=1), axis=1)

        @jax.jit
        def columns_windowed(A, ext):
            g = jnp.fft.fftshift(jnp.fft.ifft(A, axis=0), axes=0) * taper(A.shape[0], ext, edge_x).T
            return jnp.fft.fft(jnp.fft.ifftshift(g, axes=0), axis=0)                             # [Pp, nkr]

        def spectrum(S_re, S_im, W, alpha, eps_r, shift, eps_a):
            x = np.ascontiguousarray(np.asarray(pulses_windowed(S_re, S_im, W, ext_r)), np.complex64)
            Pp, K = x.shape
            A = np.empty((Pp, nkr), np.complex64)
            lib.pfa_resample_rows(x.view(np.float32).reshape(Pp, 2 * K), Pp, K, np.ascontiguousarray(np.asarray(alpha), np.float32),
                                  np.ascontiguousarray(1.0 + np.asarray(eps_r), np.float32), nkr, taps_p, tab, TABN, A.view(np.float32).reshape(Pp, 2 * nkr))
            del x
            Aw = np.ascontiguousarray(np.asarray(columns_windowed(A, ext_a)), np.complex64)
            del A
            B = np.empty((nkr, nka), np.complex64)
            lib.pfa_resample_cols(Aw.view(np.float32).reshape(Pp, 2 * nkr), Pp, nkr, np.ascontiguousarray(np.asarray(shift), np.float32),
                                  np.ascontiguousarray(1.0 + np.asarray(eps_a), np.float32), nka, taps_p, tab, TABN, B.view(np.float32).reshape(nkr, 2 * nka))
            return jnp.asarray(B)                                                               # [nkr, nka]

    if _cuda_kernels_available() and _direct_resampling():
        # the two chirp-z transforms as direct resampling by the pulse-resampling kernel (Kaiser-windowed sinc of
        # geo['taps'] taps): a row kernel over the frequency samples of each resampled pulse and a column kernel
        # over the pulses of each wavenumber row, CUDA kernels on CuPy arrays
        import cupy as cp
        k_rows, k_cols = _resample_kernels(geo['taps'])
        d_tab = cp.asarray(kernel_table(geo['taps'], geo['kaiser']))

        # the window of the chirp-z transforms (taper: one over the scene, a raised cosine to zero over the guard,
        # zero beyond) is applied in the profile domain by an unpadded transform pair before each resampling, so
        # that returns from beyond the guard do not wrap into the smaller period of the resampled grid
        @jax.jit
        def pulses_windowed(S_re, S_im, W, ext):
            x = cx(*resample_pulses(S_re, S_im, W))                                             # [rows, K]
            g = jnp.fft.fftshift(jnp.fft.ifft(x, axis=1), axes=1) * taper(x.shape[1], ext, edge_y)
            return jnp.fft.fft(jnp.fft.ifftshift(g, axes=1), axis=1)

        @jax.jit
        def columns_windowed(A, ext):
            g = jnp.fft.fftshift(jnp.fft.ifft(A, axis=0), axes=0) * taper(A.shape[0], ext, edge_x).T
            return jnp.fft.fft(jnp.fft.ifftshift(g, axes=0), axis=0)                             # [Pp, cols]

        def spectrum(S_re, S_im, W, alpha, eps_r, shift, eps_a):
            # blocks of pulses, then blocks of wavenumber columns, so that the temporaries of one block are freed
            # before the next (the whole array at once needed 40 GB on a 213,000-pulse spotlight)
            Pp = alpha.shape[0]
            st_r = cp.from_dlpack(1.0 + eps_r); a0 = cp.from_dlpack(alpha)
            A = cp.empty((Pp, nkr), cp.complex64)
            for r in range(0, Pp, rows):
                Wb = W[r:r + rows] if resample == 'dense' else (W[0][r:r + rows], W[1][r:r + rows])
                x = pulses_windowed(S_re, S_im, Wb, ext_r[r:r + rows])
                x.block_until_ready()
                cxp = cp.from_dlpack(x)
                nb, K = cxp.shape
                k_rows((-(-nkr // 128), nb), (128,), (cxp, np.int32(nb), np.int32(K), a0[r:r + rows], st_r[r:r + rows], np.int32(nkr), d_tab, A[r:r + rows]))
                cp.cuda.get_current_stream().synchronize()
                del cxp, x
            st_a = cp.from_dlpack(1.0 + eps_a); s0 = cp.from_dlpack(shift)
            B = cp.empty((nkr, nka), cp.complex64)
            cols = 1024
            for c in range(0, nkr, cols):
                Ab = cp.ascontiguousarray(A[:, c:c + cols])
                Aw = columns_windowed(jax.dlpack.from_dlpack(Ab), ext_a[c:c + cols])
                Aw.block_until_ready()
                del Ab
                cA = cp.from_dlpack(Aw)
                nc = cA.shape[1]
                k_cols((-(-nc // 128), nka), (128,), (cA, np.int32(Pp), np.int32(nc), s0[c:c + cols], st_a[c:c + cols], np.int32(nka), d_tab, B[c:c + cols]))
                cp.cuda.get_current_stream().synchronize()
                del cA, Aw
            del A
            return jax.dlpack.from_dlpack(B)                                                    # [nkr, nka]

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
    kr_c = geo['kr_min'] + (nkr // 2) * geo['dkr']           # the spectrum is centered in the oversampled raster
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

    cuda = _cuda_gather_available()
    cpu = (not cuda) and _cpu_gather_available()

    def block_start(jb):
        """The base row start of a block from the apparent positions on a coarse grid (with margin), as
        block_tables computes it from every pixel."""
        ys = (np.arange(jb, min(jb + bw, j0 + w)) - ny / 2.0) * spy + offset[1]
        X, Y = np.meshgrid(xs[::max(1, len(xs) // 64)], ys[::max(1, len(ys) // 64)], indexing='ij')
        Bp = np.stack([(X / dist['sx']) ** a * (Y / dist['sy']) ** b for a, b in dist['powers']], -1)
        fy_idx = (Y + Bp @ dist['cy']) / (spy / ov) + Ny2 / 2.0
        return int(np.clip(np.floor(fy_idx.min()) - half - 6, 0, Ny2 - span)), len(ys)

    if cuda or cpu:
        blocks = [(block_start(jb), jb) for jb in range(j0, j0 + w, bw)]

        @jax.jit
        def azimuth_base(F1, rb0):
            rows_ = jax.lax.dynamic_slice_in_dim(F1, rb0, span, axis=0)                          # [span, nka]
            G = jnp.roll(jnp.pad(rows_, ((0, 0), (0, Nx2 - nka))), -(nka // 2), axis=1)
            return jnp.fft.fftshift(t_x(G, axis=1), axes=1)                                     # [span, Nx2]

    if cpu:
        lib = _cpu_gather_lib()
        c_cx, c_cy = np.ascontiguousarray(dist['cx'], np.float64), np.ascontiguousarray(dist['cy'], np.float64)
        c_pw = np.ascontiguousarray(np.asarray(dist['powers'], np.int32).ravel())
        deg = max(a + b for a, b in dist['powers'])

        def pfa_corrected(S_re, S_im, W, alpha, eps_r, shift, eps_a, scale=None):
            b = spectrum_jit(S_re, S_im, W, alpha, eps_r, shift, eps_a)                         # [nkr, nka]
            F1 = range_stage(b)
            del b
            sc = 1.0 if scale is None else float(scale)
            out = np.empty((h, w), np.complex64)
            for (rb0, n), jb in blocks:
                base = np.ascontiguousarray(np.asarray(azimuth_base(F1, rb0)), np.complex64)
                lib.pfa_gather(base.view(np.float32).reshape(span, 2 * Nx2), span, Nx2, Ny2, rb0, h, n, i0, jb, nx, ny,
                               float(spx), float(spy), float(offset[0]), float(offset[1]), float(ov), float(dist['sx']), float(dist['sy']),
                               c_cx, c_cy, c_pw, len(c_cx), deg, float(kr_c), float(ka_c), int(sy), int(sx), float(norm), sc, int(taps),
                               out.ctypes.data + (jb - j0) * 8, w)
            return out                                                                          # numpy [h, w], scaled

        return pfa_corrected

    if cuda:
        import cupy as cp
        kern = _gather_kernel(taps, len(dist['powers']), max(a + b for a, b in dist['powers']))
        d_cx, d_cy = cp.asarray(np.asarray(dist['cx'], np.float64)), cp.asarray(np.asarray(dist['cy'], np.float64))
        d_pw = cp.asarray(np.asarray(dist['powers'], np.int32).ravel())
        f64 = np.float64

        def pfa_corrected(S_re, S_im, W, alpha, eps_r, shift, eps_a, scale=None):
            b = spectrum_jit(S_re, S_im, W, alpha, eps_r, shift, eps_a)                         # [nkr, nka]
            F1 = range_stage(b)
            del b
            sc = cp.ones((1,), cp.float32) if scale is None else cp.from_dlpack(jnp.reshape(scale, (1,)).astype(jnp.float32))
            out = cp.empty((h, w), cp.complex64)
            for (rb0, n), jb in blocks:
                base = azimuth_base(F1, rb0)
                base.block_until_ready()
                bd = cp.from_dlpack(base)
                kern((-(-h // 128), n), (128,),
                     (bd, np.int32(span), np.int32(Nx2), np.int32(Ny2), np.int32(rb0), np.int32(h), np.int32(n), np.int32(i0), np.int32(jb),
                      np.int32(nx), np.int32(ny), f64(spx), f64(spy), f64(offset[0]), f64(offset[1]), f64(ov), f64(dist['sx']), f64(dist['sy']),
                      d_cx, d_cy, d_pw, f64(kr_c), f64(ka_c), np.int32(sy), np.int32(sx), np.float32(norm), sc,
                      out[:, jb - j0:], np.int64(w)))
                cp.cuda.get_current_stream().synchronize()           # the kernel reads JAX's block until it is done
                del bd, base
            return out                                                                          # cupy [h, w], scaled

        return pfa_corrected

    tables = [block_tables(jb) for jb in range(j0, j0 + w, bw)]

    def pfa_corrected(S_re, S_im, W, alpha, eps_r, shift, eps_a, scale=None):
        b = spectrum_jit(S_re, S_im, W, alpha, eps_r, shift, eps_a)                             # [nkr, nka]
        F1 = range_stage(b)
        del b
        parts = []
        for rb0, arrs, n in tables:
            parts.append(azimuth_warp(F1, rb0, *arrs)[:, :n])
        out = jnp.concatenate(parts, axis=1)                                                  # [h, w] = [azimuth, range]
        return out if scale is None else (out * scale).astype(jnp.complex64)

    return pfa_corrected


def arrays(geo, P, resample='dense'):
    """The device arrays of the pulse resampling for the geometry geo and P pulses: the dense [P, P] matrix
    (resample='dense'), or the (indices, weights) of its taps (float32)."""
    f = np.float32
    W = pulse_matrix(geo, P).astype(f) if resample == 'dense' else tuple(np.asarray(v) for v in pulse_taps(geo, P))
    return (jnp.asarray(W) if resample == 'dense' else (jnp.asarray(W[0].astype(np.int32)), jnp.asarray(W[1].astype(f))),
            jnp.asarray(geo['alpha'].astype(f)), jnp.asarray((geo['beta'] - 1.0).astype(f)),
            jnp.asarray(geo['shift'].astype(f)), jnp.asarray((geo['gam'] - 1.0).astype(f)))
