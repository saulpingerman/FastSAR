"""Exact backprojection onto a regular grid, set up once for a collection geometry: the fast path for exact
backprojection, as ImageFormer is for factorized backprojection.

    former = fastsar.ExactFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2)      # backend='auto'
    img = former(S)

The grid is that of form_image (x_i = (i - nx/2) spx along e1, y_j = (j - ny/2) spy along e2, about the scene
reference point, or about `center`). The window, the tiles with their centers, and the range bins each pulse can reach depend only on the geometry and
are computed once (from the grid's nearest point to the antenna, the
antenna projected onto the plane and clamped into the grid with a tile of margin, to its farthest corner). A call
range-compresses each block of pulses (one zero-padded inverse FFT, upsample times the sample count) and keeps only
those bins, then backprojects them. A grid that spans more range than c / (2 df) raises ValueError.

The pixel-to-antenna distance is the tile center's, in float64 once per tile and pulse, plus a third-order
expansion in the offset d of the pixel from the center: with w the center-to-antenna vector, r = |w|, u = w / r,
du = d.u and D2 = |d|^2 - du^2, |w + d| - |w| = du + D2 / (2 r) - du D2 / (2 r^2). The next term is at most
h^4 / (8 r^3) for a tile of half-diagonal h. The tile is chosen per former: the largest of 32 x 32, 16 x 16, 8 x 8
and 4 x 4 pixels (CUDA: 32 x 32, 16 x 16, 8 x 16, 8 x 8) whose bound h^4 / (2 r^3), r the smallest pixel-to-antenna
distance, gives a phase 4 pi f_max / c times that below PHASE_LIMIT (3e-4 rad); when none does, a UserWarning gives
the predicted error. Orbital geometries keep 32 x 32 tiles; X band with 1 m pixels takes 8 x 8 at 1 to 2 km and
32 x 32 from 10 km. The range profiles are read with cubic Lagrange interpolation (four samples) or linear
interpolation (two). On three 512 by 512 pixel regions of the 2023 Umbra Panama collection, against a float64
backprojection with profiles oversampled 64 times, cubic at upsample=8 (the default) measures -77.5 to -81.7 dB,
cubic at upsample=4 -59.8 to -69.6 dB and linear at upsample=8 -53.7 to -56.4 dB. On simulated 128 x 128 pixel
scenes from 0.5 to 20 km and at 600 km, cubic at upsample=4 measures -67 to -68 dB against a float64
backprojection at 64 times oversampling.

On an Nvidia L4 the CUDA kernel runs at about 68 billion pixel-pulse pairs per second with cubic interpolation,
bound by the L1 cache's throughput for its data-dependent reads (86% of it, Nsight Compute), and forms the 12,207 by
8,808 pixel, 15,186-pulse Panama image in 24 s; factorized backprojection (ImageFormer) is faster above about 1024 to
2048 pixels on a side and exact backprojection below.

Backends: 'cuda' (CuPy kernel; pinned, overlapped upload), 'cpu' (C++ with OpenMP, vectorized over each tile's
pixels), 'jax' or 'tpu' (fastsar.backproject's JAX path on the grid's points, without the tiling above).
For arbitrary points, bistatic geometry or a DEM surface use fastsar.backproject.
"""
import ctypes
import os

import numpy as np
from . import memory as _mem

C = 299792458.0

# ---------------------------------------------------------------------------------------------------- CUDA

_CUDA_SRC = r'''
#ifndef PX
#define PX 8
#endif
#ifndef TB
#define TB 128
#endif
#ifndef TX
#define TX 32
#endif
#define TY ((TB * PX) / TX)
extern "C" __global__ void bp_tiles(const float2* rc, int P, int W, const int* lo, const double* ant, const double* ref,
                                    const double* cen, int nty, int nx, int ny, float sx, float sy,
                                    float e1x, float e1y, float e1z, float e2x, float e2y, float e2z,
                                    double inv_dr, double kcyc, float2* out) {
  // one thread block per tile of TX x TY pixels, PX pixels per thread; the tile center's terms for TB pulses at a
  // time are computed in float64 by the block's threads and shared
  __shared__ float s_ux[TB], s_uy[TB], s_uz[TB], s_ir[TB], s_i2[TB], s_tf[TB], s_ph[TB];
  __shared__ int s_ti[TB];
  const int b = blockIdx.x, j = threadIdx.x;
  const int bx = b / nty, by = b % nty;
  const double cx = cen[3*b], cy = cen[3*b+1], cz = cen[3*b+2];
  float dx[PX], dy[PX], dz[PX], dd[PX], are[PX], aim[PX];
  int ix[PX], iy[PX];
  #pragma unroll
  for (int m = 0; m < PX; ++m) {
    const int q = m * TB + j, li = q / TY, lj = q % TY;
    ix[m] = bx * TX + li; iy[m] = by * TY + lj;
    const float a = (li - 0.5f * (TX - 1)) * sx, c = (lj - 0.5f * (TY - 1)) * sy;
    dx[m] = a * e1x + c * e2x; dy[m] = a * e1y + c * e2y; dz[m] = a * e1z + c * e2z;
    dd[m] = dx[m]*dx[m] + dy[m]*dy[m] + dz[m]*dz[m];
    are[m] = 0.f; aim[m] = 0.f;
  }
  const float fdr = (float)inv_dr, fk = (float)kcyc;
  for (int p0 = 0; p0 < P; p0 += TB) {
    const int p = p0 + j;
    if (p < P) {
      const double* A = ant + 3*p;
      const double wx = cx - A[0], wy = cy - A[1], wz = cz - A[2];
      const double r = sqrt(wx*wx + wy*wy + wz*wz), ir = 1.0 / r, dR = r - ref[p];
      const double t = dR * inv_dr - lo[p], tf = floor(t), ph = kcyc * dR;
      s_ux[j] = (float)(wx*ir); s_uy[j] = (float)(wy*ir); s_uz[j] = (float)(wz*ir);
      s_ir[j] = (float)(0.5*ir); s_i2[j] = (float)(0.5*ir*ir);
      s_ti[j] = (int)tf; s_tf[j] = (float)(t - tf); s_ph[j] = (float)(ph - rint(ph));
    }
    __syncthreads();
    const int np = min(TB, P - p0);
    for (int k = 0; k < np; ++k) {
      const float ux = s_ux[k], uy = s_uy[k], uz = s_uz[k], hir = s_ir[k], hi2 = s_i2[k], tfk = s_tf[k], phk = s_ph[k];
      const float2* row = rc + (long)(p0 + k) * W + s_ti[k];
      #pragma unroll
      for (int m = 0; m < PX; ++m) {
        const float du = dx[m]*ux + dy[m]*uy + dz[m]*uz;
        // |w + d| - |w| to third order in d: du + D2 / (2 r) - du D2 / (2 r^2), D2 = |d|^2 - du^2
        const float del = du + (dd[m] - du*du) * (hir - du * hi2);
        const float t = tfk + del * fdr, fl = floorf(t), w = t - fl;
        const int i = (int)fl;
#ifdef CUBIC
        const float2 a0 = row[i - 1], a1 = row[i], a2 = row[i + 1], a3 = row[i + 2];
        const float wm1 = w - 1.f, wm2 = w - 2.f, wp1 = w + 1.f;
        const float c0 = -w * wm1 * wm2 * (1.f / 6.f), c1 = wp1 * wm1 * wm2 * 0.5f;
        const float c2 = -wp1 * w * wm2 * 0.5f, c3 = wp1 * w * wm1 * (1.f / 6.f);
        const float vre = c0*a0.x + c1*a1.x + c2*a2.x + c3*a3.x, vim = c0*a0.y + c1*a1.y + c2*a2.y + c3*a3.y;
#else
        const float2 r0 = row[i], r1 = row[i + 1];
        const float vre = fmaf(w, r1.x - r0.x, r0.x), vim = fmaf(w, r1.y - r0.y, r0.y);
#endif
        float ph = fmaf(fk, del, phk);
        ph -= rintf(ph);
        float s, c;
        __sincosf(6.283185307179586f * ph, &s, &c);
        are[m] = fmaf(vre, c, fmaf(-vim, s, are[m]));
        aim[m] = fmaf(vre, s, fmaf(vim, c, aim[m]));
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int m = 0; m < PX; ++m)
    if (ix[m] < nx && iy[m] < ny) {
      const long o = (long)ix[m] * ny + iy[m];
      out[o] = make_float2(out[o].x + are[m], out[o].y + aim[m]);
    }
}

// the windowed history of one block of pulses, centered and zero padded to nfft (the bins between stay zero)
extern "C" __global__ void pad_window(const float2* s, const float* wp, const float* wk, int n, int K, int nfft, float2* pad) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)n * K) return;
  const int p = idx / K, k = idx % K, h = K / 2;
  const float g = wp[p] * wk[k];
  const float2 v = s[idx];
  pad[(long)p * nfft + ((k >= h) ? (k - h) : (nfft - h + k))] = make_float2(v.x * g, v.y * g);
}

// the reachable bins of each profile, circularly (the fftshift folded in), scaled by nfft
extern "C" __global__ void crop(const float2* prof, const long long* blo, int n, int nfft, int W, float scale, float2* win) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)n * W) return;
  const int p = idx / W, m = idx % W;
  long long src = blo[p] + m - nfft / 2;
  src = ((src % nfft) + nfft) % nfft;
  const float2 v = prof[(long)p * nfft + src];
  win[idx] = make_float2(v.x * scale, v.y * scale);
}
'''

# ---------------------------------------------------------------------------------------------------- CPU (C++)

_CPU_SRC = r'''
#include <cmath>
#include <cstdlib>
#include <omp.h>
// one OpenMP task per tile of TX x TY pixels; per pulse the tile center's terms in float64, then the tile's pixels
// in float32 in one vectorizable loop (the sines from the vector math library)
extern "C" int bp_tiles(const float* wre, const float* wim, int P, int W, const int* lo, const double* ant,
                         const double* ref, const double* cen, int ntx, int nty, int TX, int TY, int nx, int ny,
                         float sx, float sy, const float* e1, const float* e2, double inv_dr, double kcyc, int cubic,
                         float* ore, float* oim) {
  const int T = TX * TY;
  int failed = 0;
  #pragma omp parallel
  {
    float* dx = (float*)aligned_alloc(64, sizeof(float) * T * 8);        // T even: a multiple of 64 bytes
    if (!dx) {
      #pragma omp atomic write
      failed = 1;
    }
    float *dy = dx + T, *dz = dy + T, *dd = dz + T, *are = dd + T, *aim = are + T, *tt = aim + T, *pp = tt + T;
    #pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < ntx * nty; ++b) {
      if (!dx) continue;
      const int bx = b / nty, by = b % nty;
      for (int q = 0; q < T; ++q) {
        const int li = q / TY, lj = q % TY;
        const float a = (li - 0.5f * (TX - 1)) * sx, c = (lj - 0.5f * (TY - 1)) * sy;
        dx[q] = a * e1[0] + c * e2[0]; dy[q] = a * e1[1] + c * e2[1]; dz[q] = a * e1[2] + c * e2[2];
        dd[q] = dx[q]*dx[q] + dy[q]*dy[q] + dz[q]*dz[q];
        are[q] = 0.f; aim[q] = 0.f;
      }
      const double cx = cen[3*b], cy = cen[3*b+1], cz = cen[3*b+2];
      for (int p = 0; p < P; ++p) {
        const double* A = ant + 3*p;
        const double wx = cx - A[0], wy = cy - A[1], wz = cz - A[2];
        const double r = std::sqrt(wx*wx + wy*wy + wz*wz), ir = 1.0 / r, dR = r - ref[p];
        const double t0 = dR * inv_dr - lo[p], tf0 = std::floor(t0), ph0 = kcyc * dR;
        const float ux = (float)(wx*ir), uy = (float)(wy*ir), uz = (float)(wz*ir), hir = (float)(0.5*ir);
        const float hi2 = (float)(0.5*ir*ir);
        const float tfk = (float)(t0 - tf0), phk = (float)(ph0 - std::nearbyint(ph0));
        const float fdr = (float)inv_dr, fk = (float)kcyc;
        const long base = (long)p * W + (long)tf0;
        const float* rr = wre + base;
        const float* ri = wim + base;
        #pragma omp simd
        for (int q = 0; q < T; ++q) {
          const float du = dx[q]*ux + dy[q]*uy + dz[q]*uz;
          // |w + d| - |w| to third order in d: du + D2 / (2 r) - du D2 / (2 r^2), D2 = |d|^2 - du^2
          const float del = du + (dd[q] - du*du) * (hir - du * hi2);
          tt[q] = tfk + del * fdr;
          float ph = phk + fk * del;
          pp[q] = 6.283185307179586f * (ph - std::nearbyint(ph));
        }
        if (cubic) {
          #pragma omp simd
          for (int q = 0; q < T; ++q) {
            const float fl = std::floor(tt[q]), w = tt[q] - fl;
            const int i = (int)fl;
            const float wm1 = w - 1.f, wm2 = w - 2.f, wp1 = w + 1.f;
            const float c0 = -w * wm1 * wm2 * (1.f / 6.f), c1 = wp1 * wm1 * wm2 * 0.5f;
            const float c2 = -wp1 * w * wm2 * 0.5f, c3 = wp1 * w * wm1 * (1.f / 6.f);
            const float vre = c0*rr[i-1] + c1*rr[i] + c2*rr[i+1] + c3*rr[i+2];
            const float vim = c0*ri[i-1] + c1*ri[i] + c2*ri[i+1] + c3*ri[i+2];
            const float s = std::sin(pp[q]), c = std::cos(pp[q]);
            are[q] += vre * c - vim * s;
            aim[q] += vre * s + vim * c;
          }
        } else {
          #pragma omp simd
          for (int q = 0; q < T; ++q) {
            const float fl = std::floor(tt[q]), w = tt[q] - fl;
            const int i = (int)fl;
            const float vre = rr[i] + w * (rr[i+1] - rr[i]), vim = ri[i] + w * (ri[i+1] - ri[i]);
            const float s = std::sin(pp[q]), c = std::cos(pp[q]);
            are[q] += vre * c - vim * s;
            aim[q] += vre * s + vim * c;
          }
        }
      }
      for (int q = 0; q < T; ++q) {
        const int ix = bx * TX + q / TY, iy = by * TY + q % TY;
        if (ix < nx && iy < ny) { ore[(long)ix * ny + iy] += are[q]; oim[(long)ix * ny + iy] += aim[q]; }
      }
    }
    free(dx);
  }
  return failed;
}
'''
_cpu_lib = None


def _cpu():
    global _cpu_lib
    if _cpu_lib is None:
        from ._build import shared_object
        # -ffast-math is always added: without it the sines and cosines do not vectorize (libmvec)
        flags = os.environ.get('FFBP_CPU_FLAGS', '-O3 -march=native').split() + ['-ffast-math']
        L = ctypes.CDLL(shared_object('exact_cpu', _CPU_SRC, flags))
        f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
        ci, cd, cf = ctypes.c_int, ctypes.c_double, ctypes.c_float
        L.bp_tiles.argtypes = [f32, f32, ci, ci, i32, f64, f64, f64, ci, ci, ci, ci, ci, ci, cf, cf, f32, f32, cd, cd, ci, f32, f32]
        L.bp_tiles.restype = ci
        _cpu_lib = L
    return _cpu_lib


def _threads():
    """CPU threads for the range FFTs: the CPUs this process may run on, at most OMP_NUM_THREADS."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    omp = os.environ.get('OMP_NUM_THREADS', '').strip()
    if omp.isdigit() and int(omp) > 0:
        n = min(n, int(omp))
    return max(n, 1)


# ---------------------------------------------------------------------------------------------------- tiles

# candidate tiles (TX, TY), largest first; on CUDA each has a compile-time thread block of TB threads with PX pixels
# per thread (TY = TB * PX / TX)
TILES = {'cpu': ((32, 32), (16, 16), (8, 8), (4, 4)), 'cuda': ((32, 32), (16, 16), (8, 16), (8, 8))}
CUDA_BLOCKS = {(32, 32): (128, 8), (16, 16): (128, 2), (8, 16): (128, 1), (8, 8): (64, 1)}     # (TB, PX)
PHASE_LIMIT = 3e-4          # rad: the largest predicted phase error of the range expansion a tile may have


def _integer(v, name, least):
    try:
        ok = not isinstance(v, (bool, np.bool_)) and np.isscalar(v) and np.isfinite(v) and int(v) == v and v >= least
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f'{name} must be an integer of at least {least}, got {v!r}')
    return int(v)


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _distance_to_rectangle(ant, x0, e1, e2, a0, a1, b0, b1):
    """Per pulse the distance from ant [P, 3] to the rectangle x0 + a e1 + b e2, a0 <= a <= a1, b0 <= b <= b1 (e1, e2
    orthonormal): the antenna projected onto the plane, its plane coordinates clamped into the rectangle. The distance
    is convex on the plane, so this is its exact minimum."""
    v = ant - x0
    a = np.clip(v @ e1, a0, a1)
    b = np.clip(v @ e2, b0, b1)
    return np.linalg.norm(v - a[:, None] * e1 - b[:, None] * e2, axis=1)


# ---------------------------------------------------------------------------------------------------- former

class ExactFormer:
    """Exact backprojection onto the grid of form_image, set up once for a collection geometry. former(S) ->
    complex64 image [nx, ny]. ant [P, 3] antenna phase centers (monostatic), fmin, df, K the frequency samples,
    ref [P] the range each pulse is referenced to (default |ant|, the distance to the origin), center a point the
    grid is centered on (default the origin). interp 'cubic' (default) or 'linear'; upsample the range
    oversampling (default 8); chunk the pulses range compressed at a time.

    The tile size is chosen per former from the predicted error of the range expansion (module docstring) and kept
    as former.tile (TX, TY), with the prediction as former.predicted_error_db. One former must not be called from
    two threads at once (the cuda backend reuses its pinned upload buffers)."""

    def __init__(self, ant, fmin, df, K, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), *, backend='auto',
                 window=True, interp='cubic', upsample=None, ref=None, center=None, chunk=1024):
        from .api import _backend, _check_positions, _check_grid, _window
        if interp not in ('cubic', 'linear'):
            raise ValueError(f"interp must be 'cubic' or 'linear', got {interp!r}")
        self.ant = _check_positions(ant)
        self.P = len(self.ant)
        self.K = _integer(K, 'K (frequency samples per pulse)', 2)
        if not (np.isfinite(_number(fmin)) and np.isfinite(_number(df)) and _number(df) > 0):
            raise ValueError(f'fmin must be a finite frequency and df a positive frequency spacing in Hz, got {fmin} and {df}')
        fmin, df = float(fmin), float(df)
        nx, ny, e1, e2 = _check_grid(nx, ny, spx, spy, e1, e2)
        self.nx, self.ny, self.spx, self.spy = int(nx), int(ny), float(spx), float(spy)
        self.e1, self.e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
        if center is None:
            self.center = np.zeros(3)
        else:
            try:
                self.center = np.asarray(center, np.float64)
            except (TypeError, ValueError):
                raise ValueError(f'center must be a point (a finite 3-vector) in meters, got {center!r}') from None
            if self.center.shape != (3,) or not np.isfinite(self.center).all():
                raise ValueError(f'center must be a point (a finite 3-vector) in meters, got {center!r}')
        if ref is None:
            self.ref = np.linalg.norm(self.ant, axis=1)
        else:
            try:
                self.ref = np.asarray(ref, np.float64)
            except (TypeError, ValueError):
                raise ValueError('ref must be a range per pulse in meters') from None
            if self.ref.shape != (self.P,):
                raise ValueError(f'ref must hold one range per pulse, shape ({self.P},), got {self.ref.shape}')
            if not np.isfinite(self.ref).all():
                raise ValueError(f'ref has non-finite values (NaN or inf), the first in pulse {int(np.argmax(~np.isfinite(self.ref)))}')
        self.cubic = interp == 'cubic'
        self.upsample = 8 if upsample is None else _integer(upsample, 'upsample', 1)
        self.chunk = _integer(chunk, 'chunk', 1)
        b = _backend(backend)
        self.backend = 'jax' if b == 'tpu' else b
        self.fmin, self.df, self.window = fmin, df, window
        self.nfft = 1 << int(np.ceil(np.log2(self.upsample * self.K)))
        self.inv_dr = 2.0 * df * self.nfft / C
        self.kcyc = 2.0 * (fmin + (self.K // 2) * df) / C
        self.wp, self.wk = _window(self.P, self.K) if window else (np.ones(self.P), np.ones(self.K))
        self.tile = self.predicted_error_db = None
        if self.backend == 'jax':
            return
        self._choose_tile()
        self.ntx, self.nty = -(-nx // self.tx), -(-ny // self.ty)
        x0 = self.center + (-nx / 2.0) * spx * self.e1 + (-ny / 2.0) * spy * self.e2          # pixel (0, 0)
        bi = (np.arange(self.ntx) * self.tx + 0.5 * (self.tx - 1))[:, None, None]
        bj = (np.arange(self.nty) * self.ty + 0.5 * (self.ty - 1))[None, :, None]
        self.cen = np.ascontiguousarray((x0 + bi * spx * self.e1 + bj * spy * self.e2).reshape(-1, 3))
        # the bins each pulse can reach: the grid with a tile of margin (the last tiles hang past it), nearest by
        # projection and clamping, farthest at a corner; then the expansion's error and the interpolation's taps
        mx, my = self.tx * spx, self.ty * spy
        a0, a1, b0, b1 = -mx, (nx - 1) * spx + mx, -my, (ny - 1) * spy + my
        near = _distance_to_rectangle(self.ant, x0, self.e1, self.e2, a0, a1, b0, b1)
        corners = np.array([x0 + a * self.e1 + b * self.e2 for a in (a0, a1) for b in (b0, b1)])
        far = np.linalg.norm(corners[None] - self.ant[:, None], axis=2).max(1)
        slack = 4 + int(np.ceil(self._expansion_m * self.inv_dr))
        blo = np.floor((near - self.ref) * self.inv_dr).astype(np.int64) + self.nfft // 2 - slack
        bhi = np.ceil((far - self.ref) * self.inv_dr).astype(np.int64) + self.nfft // 2 + slack
        self.W = int((bhi - blo).max()) + 4
        # the window may wrap past nfft for the pixels of the last tiles that hang past the grid (discarded); the
        # grid itself must fit in one period of the profiles, c / (2 df)
        near0 = _distance_to_rectangle(self.ant, x0, self.e1, self.e2, 0.0, (nx - 1) * spx, 0.0, (ny - 1) * spy)
        far0 = np.linalg.norm(np.array([x0 + a * self.e1 + b * self.e2 for a in (0.0, (nx - 1) * spx)
                                         for b in (0.0, (ny - 1) * spy)])[None] - self.ant[:, None], axis=2).max(1)
        need = int((np.ceil(far0 * self.inv_dr) - np.floor(near0 * self.inv_dr)).max()) + 2 * slack + 4
        if need > self.nfft:
            raise ValueError(f'the grid spans up to {float((far0 - near0).max()):.1f} m of range from one pulse, more than '
                             f'this collection resolves without ambiguity (c / (2 df) = {C / (2 * df):.1f} m less the '
                             f'interpolation margin: the grid needs {need} of the {self.nfft} range bins at '
                             f'upsample={self.upsample}); form a smaller grid (or several) or use samples spaced more '
                             'finely in frequency')
        self.blo = blo
        self.lo = (blo - self.nfft // 2).astype(np.int32)
        if self.backend == 'cuda':
            self._setup_cuda()

    def _choose_tile(self):
        """The largest tile whose predicted phase error stays below PHASE_LIMIT. The pixel offset d from the tile
        center enters through a series in |d| / r, kept to third order; the next term is bounded by h^4 / (2 r^3)
        (h the tile's half-diagonal, r the smallest center-to-antenna distance), a phase of 4 pi f_max / c times that."""
        import warnings
        nx, ny, spx, spy = self.nx, self.ny, self.spx, self.spy
        x0 = self.center + (-nx / 2.0) * spx * self.e1 + (-ny / 2.0) * spy * self.e2
        rmin = float(_distance_to_rectangle(self.ant, x0, self.e1, self.e2, 0.0, (nx - 1) * spx, 0.0, (ny - 1) * spy).min())
        kmax = 4 * np.pi * (self.fmin + (self.K - 1) * self.df) / C
        best = None
        for tx, ty in TILES[self.backend]:
            h = 0.5 * np.hypot((tx - 1) * spx, (ty - 1) * spy)
            r = rmin - h                 # a tile center lies within h of a pixel of the grid
            if r < 4 * h:               # the series is not used this close to the antenna
                continue
            err = h ** 4 / (2 * r ** 3)
            best = (tx, ty, err, kmax * err)
            if kmax * err <= PHASE_LIMIT:
                break
        if best is None:
            raise ValueError(f'the grid comes within {rmin:.1f} m of the antenna, too close for the tiled range expansion '
                             f'of ExactFormer; use fastsar.backproject on fastsar.plane_points of the grid')
        self.tx, self.ty, self._expansion_m, phase = best
        self.tile = (self.tx, self.ty)
        self.predicted_error_db = float(20 * np.log10(max(phase, 1e-30)))
        if phase > PHASE_LIMIT:
            warnings.warn(f'ExactFormer: at {rmin:.0f} m from the antenna, even {self.tx} by {self.ty} pixel tiles leave a '
                          f'predicted phase error of {phase:.1e} rad ({self.predicted_error_db:.1f} dB) from the range '
                          'expansion; fastsar.backproject on fastsar.plane_points of the grid is exact at any range',
                          UserWarning, stacklevel=3)

    # ---------------------------------------------------------------- CUDA
    def _setup_cuda(self):
        import cupy as cp
        self._tb, px = CUDA_BLOCKS[self.tile]
        opts = ('-use_fast_math', f'-DPX={px}', f'-DTB={self._tb}', f'-DTX={self.tx}') + (('-DCUBIC',) if self.cubic else ())
        m = cp.RawModule(code=_CUDA_SRC, options=opts)
        self._k_bp, self._k_pad, self._k_crop = m.get_function('bp_tiles'), m.get_function('pad_window'), m.get_function('crop')
        self._d = dict(cen=cp.asarray(self.cen), blo=cp.asarray(self.blo), lo=cp.asarray(self.lo), ant=cp.asarray(self.ant),
                       ref=cp.asarray(self.ref), wp=cp.asarray(self.wp, cp.float32), wk=cp.asarray(self.wk, cp.float32))
        ch = min(self.chunk, self.P)
        self._pin = [cp.cuda.alloc_pinned_memory(ch * self.K * 8) for _ in range(2)]
        self._hbuf = [np.frombuffer(b, np.complex64, ch * self.K).reshape(ch, self.K) for b in self._pin]
        self._stream = cp.cuda.Stream(non_blocking=True)

    def _form_cuda(self, S, check=False):
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()      # blocks cached by earlier calls or other formers
        P, K, nfft, W, d = self.P, self.K, self.nfft, self.W, self._d
        ch = min(self.chunk, P)
        on_device = not isinstance(S, np.ndarray)          # a CuPy history is read in place
        out = cp.zeros((self.nx, self.ny), cp.complex64)
        dev = [] if on_device else [cp.empty((ch, K), cp.complex64) for _ in range(2)]
        pad = cp.zeros((ch, nfft), cp.complex64)
        win = cp.empty((ch, W), cp.complex64)
        chunks = [(p0, min(P, p0 + ch)) for p0 in range(0, P, ch)]
        used = [None, None]
        bad = cp.zeros((), cp.bool_)            # a non-finite sample seen (read once, at the end)

        def stage(i):        # host copy into a pinned buffer, then an asynchronous upload on the copy stream
            if on_device:
                return None
            p0, p1 = chunks[i]
            b = i % 2
            if used[b] is not None:
                used[b].synchronize()        # this buffer's previous upload has been consumed
            np.copyto(self._hbuf[b][:p1 - p0], S[p0:p1])
            dev[b][:p1 - p0].set(self._hbuf[b][:p1 - p0], stream=self._stream)
            e = cp.cuda.Event()
            e.record(self._stream)
            return e

        f32 = np.float32
        up = stage(0)
        for i, (p0, p1) in enumerate(chunks):
            n, b = p1 - p0, i % 2
            if up is not None:
                cp.cuda.get_current_stream().wait_event(up)
            src = S[p0:p1] if on_device else dev[b]
            self._k_pad((-(-n * K // 256),), (256,), (src, d['wp'][p0:p1], d['wk'], np.int32(n), np.int32(K), np.int32(nfft), pad))
            if check:
                bad |= ~cp.isfinite(src[:n].sum())
            used[b] = cp.cuda.Event()
            used[b].record()
            if i + 1 < len(chunks):
                up = stage(i + 1)            # overlaps the FFT and backprojection below
            prof = cp.fft.ifft(pad[:n], axis=1)
            self._k_crop((-(-n * W // 256),), (256,), (prof, d['blo'][p0:p1], np.int32(n), np.int32(nfft), np.int32(W), f32(nfft), win))
            del prof
            self._k_bp((self.ntx * self.nty,), (self._tb,),
                       (win, np.int32(n), np.int32(W), d['lo'][p0:p1], d['ant'][p0:p1], d['ref'][p0:p1], d['cen'],
                        np.int32(self.nty), np.int32(self.nx), np.int32(self.ny), f32(self.spx), f32(self.spy),
                        f32(self.e1[0]), f32(self.e1[1]), f32(self.e1[2]), f32(self.e2[0]), f32(self.e2[1]), f32(self.e2[2]),
                        np.float64(self.inv_dr), np.float64(self.kcyc), out))
        if check and bool(bad):
            from .api import _check_history
            _check_history(S if isinstance(S, np.ndarray) else cp.asnumpy(S))       # raises, naming the first pulse
        return cp.asnumpy(out)

    # ---------------------------------------------------------------- CPU
    def _form_cpu(self, S):
        import scipy.fft
        L = _cpu()
        P, K, nfft, W, ch = self.P, self.K, self.nfft, self.W, self.chunk
        h = K // 2
        ore = np.zeros(self.nx * self.ny, np.float32)
        oim = np.zeros(self.nx * self.ny, np.float32)
        workers = _threads()
        e1, e2 = self.e1.astype(np.float32), self.e2.astype(np.float32)
        cols = np.arange(W)
        for p0 in range(0, P, ch):
            p1 = min(P, p0 + ch)
            n = p1 - p0
            s = S[p0:p1] * (self.wp[p0:p1, None] * self.wk[None, :]).astype(np.float32)
            pad = np.zeros((n, nfft), np.complex64)
            pad[:, :K - h] = s[:, h:]
            pad[:, nfft - h:] = s[:, :h]
            prof = scipy.fft.ifft(pad, axis=1, workers=workers, overwrite_x=True)
            idx = (self.blo[p0:p1, None] + cols[None, :] - nfft // 2) % nfft
            win = np.take_along_axis(prof, idx, axis=1) * nfft
            if L.bp_tiles(np.ascontiguousarray(win.real, np.float32), np.ascontiguousarray(win.imag, np.float32), n, W,
                          np.ascontiguousarray(self.lo[p0:p1]), np.ascontiguousarray(self.ant[p0:p1]),
                          np.ascontiguousarray(self.ref[p0:p1]), self.cen, self.ntx, self.nty, self.tx, self.ty, self.nx,
                          self.ny, self.spx, self.spy, e1, e2, self.inv_dr, self.kcyc, int(self.cubic), ore, oim):
                raise MemoryError('ExactFormer cpu: a thread could not allocate its tile buffers')
        return (ore + 1j * oim).astype(np.complex64).reshape(self.nx, self.ny)

    # ---------------------------------------------------------------- memory
    def memory(self):
        """What a call needs on this former's device and what is free: dict(backend, needed, available, full_speed,
        parts) in bytes, as ImageFormer.memory(). cuda: device memory for the pulse chunks (two uploaded histories,
        the padded block, its inverse FFT and cuFFT's workspace, the cropped profiles), the image and the geometry;
        cpu: host memory for the same per-chunk arrays and the image; jax: fastsar.backproject's arrays, estimated."""
        from . import memory as _mem
        ch = min(self.chunk, self.P)
        npix = self.nx * self.ny
        if self.backend == 'jax':
            from .bp import BLOCK
            nfft = 1 << int(np.ceil(np.log2(self.upsample * (4 if self.cubic else 1) * self.K)))
            npts = -(-npix // BLOCK) * BLOCK
            pc = min(256, self.P)
            parts = dict(points=npts * 3 * 4, image=npts * 8 * 2, profiles=pc * nfft * 8 * 3,
                         centers=pc * (npts // BLOCK) * 4 * 8, step=npts * 4 * 8)
        else:
            geo = self.cen.nbytes + self.P * (3 * 8 + 8 + 8 + 4 + 8)
            if self.backend == 'cuda':
                parts = dict(histories=2 * ch * self.K * 8, padded=ch * self.nfft * 8,
                             fft=2 * ch * self.nfft * 8, window=ch * self.W * 8, image=npix * 8, geometry=geo)
            else:
                parts = dict(history=ch * self.K * 8 * 2, padded=ch * self.nfft * 8, fft=ch * self.nfft * 8,
                             window=ch * self.W * (8 + 8 + 8 + 8), image=npix * 4 * 2 + npix * 8 * 2, geometry=geo)
        need = float(sum(parts.values()))
        avail = _mem.device_available(self.backend)
        return dict(backend=self.backend, needed=need, available=avail, full_speed=need <= avail, parts=parts)

    # ---------------------------------------------------------------- call
    def __call__(self, S):
        """S [P, K] complex phase history (numpy, or cupy on the cuda backend) -> complex64 image [nx, ny]."""
        from .api import _check_history
        if not (self.backend == 'cuda' and type(S).__module__.startswith('cupy')):
            S = np.asarray(S)
        if self.backend == 'cuda' and isinstance(S, np.ndarray):     # the scan for NaN and inf runs on the GPU
            return self._form(_check_history(S, self.P, finite=False), check=True)
        return self._form(_check_history(S, self.P))

    def _form(self, S, check=False):
        """S already checked (form_image checks it once); check: scan for non-finite samples on the GPU."""
        if S.shape[1] != self.K:
            raise ValueError(f'phase history must have {self.K} samples per pulse, got {S.shape[1]}')
        S = S.astype(np.complex64, copy=False)
        if self.backend == 'cuda':
            if not isinstance(S, np.ndarray):
                import cupy as cp
                S = cp.ascontiguousarray(S)
            import cupy as cp
            while True:
                try:
                    return self._form_cuda(S, check)
                except cp.cuda.memory.OutOfMemoryError:
                    if self.chunk <= 64:
                        raise
                    self.chunk //= 2        # memory held elsewhere (another former, JAX): fewer pulses per chunk
                    _mem._warn(f'cuda: out of GPU memory; ExactFormer continues with {self.chunk} pulses per chunk '
                               f'({_mem._gb(_mem.cuda_free())} free)')
        if self.backend == 'cpu':
            return self._form_cpu(S)
        from .bp import backproject, plane_points
        pts = plane_points(self.nx, self.ny, self.spx, self.spy, self.e1, self.e2) + self.center
        # linear interpolation only: 4 times the oversampling stands in for cubic interpolation's accuracy
        return backproject(S, self.ant, self.fmin, self.df, pts, ref=self.ref, backend='jax',
                           upsample=self.upsample * (4 if self.cubic else 1), window=self.window)
