"""Exact backprojection onto a regular grid, set up once for a collection geometry: the fast path for exact
backprojection, as ImageFormer is for factorized backprojection.

    former = fastsar.ExactFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2)      # backend='auto'
    img = former(S)

The grid is that of form_image (x_i = (i - nx/2) spx along e1, y_j = (j - ny/2) spy along e2, about the scene
reference point, or about `center`). Everything that depends only on the geometry is computed once: the window, the
tiles and their centers, and the range bins each pulse can reach. A call range-compresses each block of pulses (one
zero-padded inverse FFT, upsample times the sample count) and keeps only those bins, then backprojects them.

The pixel-to-antenna distance is the tile center's, in float64 once per tile and pulse, plus a second-order
expansion in the offset d of the pixel from the center, |w + d| - |w| = d.u + (|d|^2 - (d.u)^2) / (2 |w|), whose
error |d|^3 / (2 |w|^2) is nanometres for tiles of tens of metres at airborne or orbital range. The range profiles
are read with cubic Lagrange interpolation (four samples) or linear interpolation (two). On the 2023 Umbra Panama
collection, against the study's float64 reference (16 times oversampled, linear), cubic at upsample=4 measures
-70 dB, at the reference's own accuracy, and linear at upsample=8 -57 dB.

Backends: 'cuda' (CuPy kernel; pinned, overlapped upload), 'cpu' (C++ with OpenMP, vectorized over each tile's
pixels), 'jax' or 'tpu' (fastsar.backproject's JAX path on the grid's points, without the tiling above).
For arbitrary points, bistatic geometry or a DEM surface use fastsar.backproject.
"""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

C = 299792458.0

# ---------------------------------------------------------------------------------------------------- CUDA

_CUDA_SRC = r'''
#ifndef PX
#define PX 4
#endif
#ifndef TB
#define TB 128
#endif
#ifndef TX
#define TX 16
#endif
#define TY ((TB * PX) / TX)
extern "C" __global__ void bp_tiles(const float2* rc, int P, int W, const int* lo, const double* ant, const double* ref,
                                    const double* cen, int nty, int nx, int ny, float sx, float sy,
                                    float e1x, float e1y, float e1z, float e2x, float e2y, float e2z,
                                    double inv_dr, double kcyc, float2* out) {
  // one thread block per tile of TX x TY pixels, PX pixels per thread; the tile center's terms for TB pulses at a
  // time are computed in float64 by the block's threads and shared
  __shared__ float s_ux[TB], s_uy[TB], s_uz[TB], s_ir[TB], s_tf[TB], s_ph[TB];
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
      s_ux[j] = (float)(wx*ir); s_uy[j] = (float)(wy*ir); s_uz[j] = (float)(wz*ir); s_ir[j] = (float)ir;
      s_ti[j] = (int)tf; s_tf[j] = (float)(t - tf); s_ph[j] = (float)(ph - rint(ph));
    }
    __syncthreads();
    const int np = min(TB, P - p0);
    for (int k = 0; k < np; ++k) {
      const float ux = s_ux[k], uy = s_uy[k], uz = s_uz[k], hir = 0.5f * s_ir[k], tfk = s_tf[k], phk = s_ph[k];
      const float2* row = rc + (long)(p0 + k) * W + s_ti[k];
      #pragma unroll
      for (int m = 0; m < PX; ++m) {
        const float du = dx[m]*ux + dy[m]*uy + dz[m]*uz;
        const float del = du + (dd[m] - du*du) * hir;
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
extern "C" void bp_tiles(const float* wre, const float* wim, int P, int W, const int* lo, const double* ant,
                         const double* ref, const double* cen, int ntx, int nty, int TX, int TY, int nx, int ny,
                         float sx, float sy, const float* e1, const float* e2, double inv_dr, double kcyc, int cubic,
                         float* ore, float* oim) {
  const int T = TX * TY;
  #pragma omp parallel
  {
    float* dx = (float*)aligned_alloc(64, sizeof(float) * T * 8);
    float *dy = dx + T, *dz = dy + T, *dd = dz + T, *are = dd + T, *aim = are + T, *tt = aim + T, *pp = tt + T;
    #pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < ntx * nty; ++b) {
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
        const float tfk = (float)(t0 - tf0), phk = (float)(ph0 - std::nearbyint(ph0));
        const float fdr = (float)inv_dr, fk = (float)kcyc;
        const long base = (long)p * W + (long)tf0;
        const float* rr = wre + base;
        const float* ri = wim + base;
        #pragma omp simd
        for (int q = 0; q < T; ++q) {
          const float du = dx[q]*ux + dy[q]*uy + dz[q]*uz;
          const float del = du + (dd[q] - du*du) * hir;
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
}
'''
_cpu_lib = None


def _cpu():
    global _cpu_lib
    if _cpu_lib is None:
        tag = hashlib.sha1(_CPU_SRC.encode()).hexdigest()[:12]
        d = os.path.join(os.path.expanduser('~'), '.cache', 'fastsar')
        os.makedirs(d, exist_ok=True)
        so, src = os.path.join(d, f'libexact_cpu_{tag}.so'), os.path.join(d, f'exact_cpu_{tag}.cpp')
        if not os.path.exists(so):
            with open(src, 'w') as fh:
                fh.write(_CPU_SRC)
            flags = os.environ.get('FFBP_CPU_FLAGS', '-O3 -march=native -ffast-math')
            subprocess.run([os.environ.get('CXX', 'g++')] + flags.split() + ['-fopenmp', '-shared', '-fPIC', src, '-o', so + '.tmp'],
                           check=True)
            os.replace(so + '.tmp', so)
        L = ctypes.CDLL(so)
        f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        i32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
        ci, cd, cf = ctypes.c_int, ctypes.c_double, ctypes.c_float
        L.bp_tiles.argtypes = [f32, f32, ci, ci, i32, f64, f64, f64, ci, ci, ci, ci, ci, ci, cf, cf, f32, f32, cd, cd, ci, f32, f32]
        _cpu_lib = L
    return _cpu_lib


# ---------------------------------------------------------------------------------------------------- former

class ExactFormer:
    """Exact backprojection onto the grid of form_image, set up once for a collection geometry. former(S) ->
    complex64 image [nx, ny]. ant [P, 3] antenna phase centers (monostatic), fmin, df, K the frequency samples,
    ref [P] the range each pulse is referenced to (default |ant|, the distance to the origin), center a point the
    grid is centered on (default the origin). interp 'cubic' (default) or 'linear'; upsample the range
    oversampling (default 4 for cubic, 8 for linear)."""

    def __init__(self, ant, fmin, df, K, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), backend='auto',
                 window=True, interp='cubic', upsample=None, ref=None, center=None, chunk=1024):
        from .api import _backend, _check_positions, _check_grid, _window
        if interp not in ('cubic', 'linear'):
            raise ValueError("interp must be 'cubic' or 'linear'")
        self.ant = _check_positions(ant)
        self.P, self.K = len(self.ant), int(K)
        nx, ny, e1, e2 = _check_grid(nx, ny, spx, spy, e1, e2)
        self.nx, self.ny, self.spx, self.spy = int(nx), int(ny), float(spx), float(spy)
        self.e1, self.e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
        self.center = np.zeros(3) if center is None else np.asarray(center, np.float64)
        self.backend = 'jax' if backend == 'tpu' else _backend(backend)
        self.cubic = interp == 'cubic'
        self.upsample = int(upsample or (4 if self.cubic else 8))
        self.ref = np.linalg.norm(self.ant, axis=1) if ref is None else np.asarray(ref, np.float64)
        if self.ref.shape != (self.P,):
            raise ValueError(f'ref must hold one range per pulse, shape ({self.P},), got {self.ref.shape}')
        self.fmin, self.df, self.window, self.chunk = float(fmin), float(df), window, int(chunk)
        self.nfft = 1 << int(np.ceil(np.log2(self.upsample * K)))
        self.inv_dr = 2.0 * df * self.nfft / C
        self.kcyc = 2.0 * (fmin + (K // 2) * df) / C
        self.wp, self.wk = _window(self.P, K) if window else (np.ones(self.P), np.ones(K))
        if self.backend == 'jax':
            return
        # tiles: CUDA 16 x 32 pixels (128 threads, 4 pixels each); CPU 32 x 32
        self.tx, self.ty = (16, 32) if self.backend == 'cuda' else (32, 32)
        self.ntx, self.nty = -(-nx // self.tx), -(-ny // self.ty)
        x0 = self.center + (-nx / 2.0) * spx * self.e1 + (-ny / 2.0) * spy * self.e2          # pixel (0, 0)
        bi = (np.arange(self.ntx) * self.tx + 0.5 * (self.tx - 1))[:, None, None]
        bj = (np.arange(self.nty) * self.ty + 0.5 * (self.ty - 1))[None, :, None]
        self.cen = np.ascontiguousarray((x0 + bi * spx * self.e1 + bj * spy * self.e2).reshape(-1, 3))
        # the bins each pulse can reach: the grid's corners with a tile of margin, and the interpolation's 2 samples
        far = x0 + (nx - 1) * spx * self.e1 + (ny - 1) * spy * self.e2
        m1, m2 = self.tx * spx * self.e1, self.ty * spy * self.e2
        corners = np.array([x0 - m1 - m2, x0 + (nx - 1) * spx * self.e1 + m1 - m2, x0 + (ny - 1) * spy * self.e2 - m1 + m2,
                            far + m1 + m2])
        rr = np.linalg.norm(corners[None] - self.ant[:, None], axis=2) - self.ref[:, None]
        blo = np.floor(rr.min(1) * self.inv_dr).astype(np.int64) + self.nfft // 2 - 4
        bhi = np.ceil(rr.max(1) * self.inv_dr).astype(np.int64) + self.nfft // 2 + 4
        self.W = min(int((bhi - blo).max()) + 4, self.nfft)
        self.blo = blo
        self.lo = (blo - self.nfft // 2).astype(np.int32)
        if self.backend == 'cuda':
            self._setup_cuda()

    # ---------------------------------------------------------------- CUDA
    def _setup_cuda(self):
        import cupy as cp
        m = cp.RawModule(code=_CUDA_SRC, options=('-use_fast_math', '-DPX=4', '-DTB=128', '-DTX=16') + (('-DCUBIC',) if self.cubic else ()))
        self._k_bp, self._k_pad, self._k_crop = m.get_function('bp_tiles'), m.get_function('pad_window'), m.get_function('crop')
        self._d = dict(cen=cp.asarray(self.cen), blo=cp.asarray(self.blo), lo=cp.asarray(self.lo), ant=cp.asarray(self.ant),
                       ref=cp.asarray(self.ref), wp=cp.asarray(self.wp, cp.float32), wk=cp.asarray(self.wk, cp.float32))
        nbytes = self.chunk * self.K * 8
        self._pin = [cp.cuda.alloc_pinned_memory(nbytes) for _ in range(2)]
        self._hbuf = [np.frombuffer(b, np.complex64, self.chunk * self.K).reshape(self.chunk, self.K) for b in self._pin]
        self._stream = cp.cuda.Stream(non_blocking=True)

    def _form_cuda(self, S):
        import cupy as cp
        P, K, nfft, W, ch, d = self.P, self.K, self.nfft, self.W, self.chunk, self._d
        out = cp.zeros((self.nx, self.ny), cp.complex64)
        dev = [cp.empty((ch, K), cp.complex64) for _ in range(2)]
        pad = cp.zeros((ch, nfft), cp.complex64)
        win = cp.empty((ch, W), cp.complex64)
        chunks = [(p0, min(P, p0 + ch)) for p0 in range(0, P, ch)]
        used = [None, None]

        def stage(i):        # host copy into a pinned buffer, then an asynchronous upload on the copy stream
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
            cp.cuda.get_current_stream().wait_event(up)
            self._k_pad((-(-n * K // 256),), (256,), (dev[b], d['wp'][p0:p1], d['wk'], np.int32(n), np.int32(K), np.int32(nfft), pad))
            used[b] = cp.cuda.Event()
            used[b].record()
            if i + 1 < len(chunks):
                up = stage(i + 1)            # overlaps the FFT and backprojection below
            prof = cp.fft.ifft(pad[:n], axis=1)
            self._k_crop((-(-n * W // 256),), (256,), (prof, d['blo'][p0:p1], np.int32(n), np.int32(nfft), np.int32(W), f32(nfft), win))
            del prof
            self._k_bp((self.ntx * self.nty,), (128,),
                       (win, np.int32(n), np.int32(W), d['lo'][p0:p1], d['ant'][p0:p1], d['ref'][p0:p1], d['cen'],
                        np.int32(self.nty), np.int32(self.nx), np.int32(self.ny), f32(self.spx), f32(self.spy),
                        f32(self.e1[0]), f32(self.e1[1]), f32(self.e1[2]), f32(self.e2[0]), f32(self.e2[1]), f32(self.e2[2]),
                        np.float64(self.inv_dr), np.float64(self.kcyc), out))
        return cp.asnumpy(out)

    # ---------------------------------------------------------------- CPU
    def _form_cpu(self, S):
        import scipy.fft
        L = _cpu()
        P, K, nfft, W, ch = self.P, self.K, self.nfft, self.W, self.chunk
        h = K // 2
        ore = np.zeros(self.nx * self.ny, np.float32)
        oim = np.zeros(self.nx * self.ny, np.float32)
        workers = os.cpu_count() or 1
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
            L.bp_tiles(np.ascontiguousarray(win.real, np.float32), np.ascontiguousarray(win.imag, np.float32), n, W,
                       np.ascontiguousarray(self.lo[p0:p1]), np.ascontiguousarray(self.ant[p0:p1]),
                       np.ascontiguousarray(self.ref[p0:p1]), self.cen, self.ntx, self.nty, self.tx, self.ty, self.nx, self.ny,
                       self.spx, self.spy, e1, e2, self.inv_dr, self.kcyc, int(self.cubic), ore, oim)
        return (ore + 1j * oim).astype(np.complex64).reshape(self.nx, self.ny)

    def __call__(self, S):
        from .api import _check_history
        S = _check_history(np.asarray(S), self.P)
        if S.shape[1] != self.K:
            raise ValueError(f'phase history must have {self.K} samples per pulse, got {S.shape[1]}')
        S = S.astype(np.complex64, copy=False)
        if self.backend == 'cuda':
            return self._form_cuda(S)
        if self.backend == 'cpu':
            return self._form_cpu(S)
        from .bp import backproject, plane_points
        pts = plane_points(self.nx, self.ny, self.spx, self.spy, self.e1, self.e2) + self.center
        # linear interpolation only: 4 times the oversampling stands in for cubic interpolation's accuracy
        return backproject(S, self.ant, self.fmin, self.df, pts, ref=self.ref, backend='jax',
                           upsample=self.upsample * (4 if self.cubic else 1), window=self.window)
