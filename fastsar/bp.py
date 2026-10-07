"""Exact (direct) backprojection onto arbitrary points: any output grid (ground plane, slant plane, a DEM surface,
a map projection), monostatic or bistatic, on every backend. Its cost is pulses x points, so for large planar
images factorized backprojection (form_image) is the fast path; this is for grids FFBP does not cover, for
coregistered stacks formed onto one set of points, and as a reference.

    img = fastsar.backproject(S, ant, fmin, df, points)                 # monostatic, points [..., 3]
    img = fastsar.backproject(S, tx, fmin, df, points, rcv=rcv)        # bistatic

S [P, K] is the frequency-domain phase history motion compensated to the scene reference point (the origin of the
coordinates of ant, tx, rcv and points), so that a scatterer at x contributes exp(-j 4 pi f dR / c) with
dR = |x - a| - |a| (monostatic) or dR = (|x - tx| + |x - rcv| - |tx| - |rcv|) / 2 (bistatic), the CPHD convention.

Each pulse is range compressed by a zero-padded inverse FFT (upsample times the sample count, rounded up to a
power of two) and read with linear interpolation, circularly: a frequency-domain phase history is periodic in range
with period c / (2 df), so a scene may lie anywhere within that period about the reference range. Against a 128x reference on a simulated scene the image error
is -39 dB at upsample=2, -51 at 4, -63 at 8 (the default), -75 at 16 and -88 at 32: 12 dB per doubling.
Ranges are computed in float64 at the center of each block of nearby points and in float32 only for offsets
within the block, so float32 kernels keep their accuracy at orbital range.
"""
import ctypes, hashlib, os, subprocess

import numpy as np

C = 299792458.0
BLOCK = 256


def _window(P, K):
    from .api import _window as w
    return w(P, K)


def _order(points):
    """Permutation grouping points into spatially compact blocks of BLOCK (Morton order on a grid fine enough that
    a block spans few cells), block centers, and offsets of each point from its block's center."""
    p = points.reshape(-1, 3).astype(np.float64)
    N = len(p)
    lo, hi = p.min(0), p.max(0)
    q = ((p - lo) / np.maximum(hi - lo, 1e-9) * 1023).astype(np.int64)

    def spread(v):
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        return (v | (v << 2)) & 0x09249249
    order = np.argsort(spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2), kind='stable')
    nb = -(-N // BLOCK)
    pad = np.concatenate([order, np.full(nb * BLOCK - N, order[-1])])
    pb = p[pad].reshape(nb, BLOCK, 3)
    cen = 0.5 * (pb.min(1) + pb.max(1))
    return order, cen, (pb - cen[:, None, :]).astype(np.float32)


def range_compress(S, nfft, xp=np):
    """S [p, K] -> [p, nfft] complex, zero delay at index nfft // 2, carrier fmin + (K // 2) df removed; the sum of
    the samples, not their mean (the unnormalized inverse transform), so that images have the gain of form_image."""
    P, K = S.shape
    h = K // 2
    pad = xp.zeros((P, nfft), dtype=S.dtype)
    pad[:, :K - h] = S[:, h:]
    pad[:, nfft - h:] = S[:, :h]
    return xp.fft.fftshift(xp.fft.ifft(pad, axis=1), axes=1) * nfft


def _centers(tx, rcv, ref, cen, nfft, inv_dr, kcyc):
    """float64 center terms for pulses tx [p, 3] and blocks cen [nb, 3]: integer and fractional range bin of the
    block center, its phase in cycles (wrapped), and the center-to-antenna vectors and their lengths in float32."""
    wt = cen[None, :, :] - tx[:, None, :]
    rt = np.linalg.norm(wt, axis=2)
    if rcv is None:
        dR = rt - ref[:, None]
        wr = rr = None
    else:
        wr = cen[None, :, :] - rcv[:, None, :]
        rr = np.linalg.norm(wr, axis=2)
        dR = 0.5 * (rt + rr) - ref[:, None]
    t = dR * inv_dr + nfft // 2
    ti = np.floor(t)
    ph = kcyc * dR
    out = dict(ti=ti.astype(np.int32), tf=(t - ti).astype(np.float32), ph=(ph - np.round(ph)).astype(np.float32),
               wt=wt.astype(np.float32), rt=rt.astype(np.float32))
    if rcv is not None:
        out.update(wr=wr.astype(np.float32), rr=rr.astype(np.float32))
    return out


def _delta(d, w, r):
    """|w + d| - |w| for block offsets d [nb, B, 3] and center-to-antenna vectors w [p, nb, 3], r [p, nb], in the
    cancellation-free form (2 d.w + |d|^2) / (|w + d| + |w|)."""
    dd = (d * d).sum(-1)[None]
    dw = (d[None] * w[:, :, None, :]).sum(-1)
    wd = np.sqrt(np.maximum(r[:, :, None] ** 2 + 2 * dw + dd, 0))
    return (2 * dw + dd) / (wd + r[:, :, None])


# ---------------------------------------------------------------------------------------------------- CPU (C++)

_CPU_SRC = r'''
#include <cmath>
#include <complex>
#include <omp.h>
extern "C" void bp_points(const float* rre, const float* rim, int P, int nfft, const double* tx, const double* rcv,
                          const double* ref, const double* pts, int N, double inv_dr, double kcyc, double* ore, double* oim) {
  const double two_pi = 6.283185307179586;
  #pragma omp parallel for schedule(dynamic, 64)
  for (int n = 0; n < N; ++n) {
    const double x = pts[3*n], y = pts[3*n+1], z = pts[3*n+2];
    double are = 0.0, aim = 0.0;
    for (int p = 0; p < P; ++p) {
      const double* a = tx + 3*p;
      double ax = x - a[0], ay = y - a[1], az = z - a[2];
      double dR = std::sqrt(ax*ax + ay*ay + az*az);
      if (rcv) {
        const double* b = rcv + 3*p;
        double bx = x - b[0], by = y - b[1], bz = z - b[2];
        dR = 0.5 * (dR + std::sqrt(bx*bx + by*by + bz*bz));
      }
      dR -= ref[p];
      double t = dR * inv_dr + (nfft / 2);
      double tf = std::floor(t);
      long i = (long)tf % nfft;
      if (i < 0) i += nfft;
      const long i1 = (i + 1 == nfft) ? 0 : i + 1;
      double w = t - tf;
      const float* rr = rre + (long)p * nfft;
      const float* ri = rim + (long)p * nfft;
      double vre = rr[i] + w * (rr[i1] - rr[i]), vim = ri[i] + w * (ri[i1] - ri[i]);
      double ph = kcyc * dR;
      ph = two_pi * (ph - std::nearbyint(ph));
      double c = std::cos(ph), s = std::sin(ph);
      are += vre * c - vim * s;
      aim += vre * s + vim * c;
    }
    ore[n] += are;
    oim[n] += aim;
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
        so, src = os.path.join(d, f'libbp_cpu_{tag}.so'), os.path.join(d, f'bp_cpu_{tag}.cpp')
        if not os.path.exists(so):
            with open(src, 'w') as fh:
                fh.write(_CPU_SRC)
            flags = os.environ.get('FFBP_CPU_FLAGS', '-O3 -march=native')
            subprocess.run([os.environ.get('CXX', 'g++')] + flags.split() + ['-fopenmp', '-shared', '-fPIC', src, '-o', so + '.tmp'],
                           check=True)
            os.replace(so + '.tmp', so)
        L = ctypes.CDLL(so)
        f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
        f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
        L.bp_points.argtypes = [f32, f32, ctypes.c_int, ctypes.c_int, f64, ctypes.c_void_p, f64, f64, ctypes.c_int,
                                ctypes.c_double, ctypes.c_double, f64, f64]
        _cpu_lib = L
    return _cpu_lib


def _run_cpu(S, tx, rcv, ref, pts, nfft, inv_dr, kcyc, chunk):
    L = _cpu()
    N = len(pts)
    ore, oim = np.zeros(N), np.zeros(N)
    pts = np.ascontiguousarray(pts, np.float64)
    for p0 in range(0, len(S), chunk):
        sl = slice(p0, min(len(S), p0 + chunk))
        rc = range_compress(S[sl].astype(np.complex64), nfft).astype(np.complex64)
        t = np.ascontiguousarray(tx[sl])
        r = None if rcv is None else np.ascontiguousarray(rcv[sl])
        L.bp_points(np.ascontiguousarray(rc.real), np.ascontiguousarray(rc.imag), len(t), nfft, t,
                    None if r is None else r.ctypes.data, np.ascontiguousarray(ref[sl]), pts, N, inv_dr, kcyc, ore, oim)
    return (ore + 1j * oim).astype(np.complex64)


# ---------------------------------------------------------------------------------------------------- CUDA

_CUDA_SRC = r'''
extern "C" __global__ void bp_blocks(const float2* rc, int P, int nfft, const double* tx, const double* rcv,
                                     const double* ref, const double* cen, const float* d, double inv_dr, double kcyc, float2* out) {
  // one thread block per block of 256 points (one thread per point); the block's center terms for 256 pulses at a
  // time are computed in float64 by the block's threads and shared, offsets are handled in float32
  __shared__ float s_wt[256][3], s_rt[256], s_wr[256][3], s_rr[256], s_tf[256], s_ph[256];
  __shared__ int s_ti[256];
  const int b = blockIdx.x, j = threadIdx.x;
  const double cx = cen[3*b], cy = cen[3*b+1], cz = cen[3*b+2];
  const float dx = d[(b*256 + j)*3], dy = d[(b*256 + j)*3 + 1], dz = d[(b*256 + j)*3 + 2];
  const float dd = dx*dx + dy*dy + dz*dz;
  float are = 0.f, aim = 0.f;
  for (int p0 = 0; p0 < P; p0 += 256) {
    const int p = p0 + j;
    if (p < P) {
      const double* a = tx + 3*p;
      double wx = cx - a[0], wy = cy - a[1], wz = cz - a[2];
      double rt = sqrt(wx*wx + wy*wy + wz*wz), dR = rt;
      s_wt[j][0] = (float)wx; s_wt[j][1] = (float)wy; s_wt[j][2] = (float)wz; s_rt[j] = (float)rt;
      if (rcv) {
        const double* q = rcv + 3*p;
        double vx = cx - q[0], vy = cy - q[1], vz = cz - q[2];
        double rr = sqrt(vx*vx + vy*vy + vz*vz);
        dR = 0.5 * (dR + rr);
        s_wr[j][0] = (float)vx; s_wr[j][1] = (float)vy; s_wr[j][2] = (float)vz; s_rr[j] = (float)rr;
      }
      dR -= ref[p];
      double t = dR * inv_dr + (nfft / 2), tf = floor(t), ph = kcyc * dR;
      s_ti[j] = (int)tf; s_tf[j] = (float)(t - tf); s_ph[j] = (float)(ph - rint(ph));
    }
    __syncthreads();
    const int np = min(256, P - p0);
    for (int k = 0; k < np; ++k) {
      float dw = dx*s_wt[k][0] + dy*s_wt[k][1] + dz*s_wt[k][2];
      float del = (2.f*dw + dd) / (sqrtf(fmaxf(s_rt[k]*s_rt[k] + 2.f*dw + dd, 0.f)) + s_rt[k]);
      if (rcv) {
        float dv = dx*s_wr[k][0] + dy*s_wr[k][1] + dz*s_wr[k][2];
        del = 0.5f * (del + (2.f*dv + dd) / (sqrtf(fmaxf(s_rr[k]*s_rr[k] + 2.f*dv + dd, 0.f)) + s_rr[k]));
      }
      float t = s_tf[k] + del * (float)inv_dr, tf = floorf(t);
      int i = (s_ti[k] + (int)tf) % nfft;
      if (i < 0) i += nfft;
      const int i1 = (i + 1 == nfft) ? 0 : i + 1;
      float w = t - tf;
      float2 r0 = rc[(long)(p0 + k) * nfft + i], r1 = rc[(long)(p0 + k) * nfft + i1];
      float vre = r0.x + w * (r1.x - r0.x), vim = r0.y + w * (r1.y - r0.y);
      float ph = s_ph[k] + (float)kcyc * del;
      float s, c;
      sincospif(2.f * (ph - rintf(ph)), &s, &c);
      are += vre * c - vim * s;
      aim += vre * s + vim * c;
    }
    __syncthreads();
  }
  float2 o = out[b*256 + j];
  out[b*256 + j] = make_float2(o.x + are, o.y + aim);
}
'''


def _run_cuda(S, tx, rcv, ref, cen, d, nfft, inv_dr, kcyc, chunk):
    import cupy as cp
    k = cp.RawKernel(_CUDA_SRC, 'bp_blocks', options=('-use_fast_math',))
    nb = len(cen)
    out = cp.zeros(nb * BLOCK, cp.complex64)
    cen_d, d_d = cp.asarray(cen, cp.float64), cp.asarray(d.reshape(-1), cp.float32)
    for p0 in range(0, len(S), chunk):
        sl = slice(p0, min(len(S), p0 + chunk))
        rc = range_compress(cp.asarray(S[sl], cp.complex64), nfft, xp=cp).astype(cp.complex64)
        t = cp.asarray(tx[sl], cp.float64)
        r = None if rcv is None else cp.asarray(rcv[sl], cp.float64)
        k((nb,), (BLOCK,), (rc, np.int32(len(t)), np.int32(nfft), t, r if r is not None else cp.uint64(0),
                             cp.asarray(ref[sl], cp.float64), cen_d, d_d,
                             np.float64(inv_dr), np.float64(kcyc), out))
    return cp.asnumpy(out)


# ---------------------------------------------------------------------------------------------------- JAX

def _run_jax(S, tx, rcv, ref, cen, d, nfft, inv_dr, kcyc, chunk):
    import jax, jax.numpy as jnp
    nb = len(cen)
    bis = rcv is not None

    @jax.jit
    def step(acc, rc, ti, tf, ph, wt, rt, wr, rr, dd_):
        def one(acc, xs):
            rcp, ti, tf, ph, wt, rt, wr, rr = xs
            dw = (dd_ * wt[:, None, :]).sum(-1)
            q = (dd_ * dd_).sum(-1)
            de = (2 * dw + q) / (jnp.sqrt(jnp.maximum(rt[:, None] ** 2 + 2 * dw + q, 0)) + rt[:, None])
            if bis:
                dv = (dd_ * wr[:, None, :]).sum(-1)
                de = 0.5 * (de + (2 * dv + q) / (jnp.sqrt(jnp.maximum(rr[:, None] ** 2 + 2 * dv + q, 0)) + rr[:, None]))
            t = tf[:, None] + de * inv_dr
            fl = jnp.floor(t)
            i = ti[:, None] + fl.astype(jnp.int32)
            w = t - fl
            ic = jnp.mod(i, nfft)
            v = rcp[ic] * (1 - w) + rcp[jnp.mod(ic + 1, nfft)] * w
            p = ph[:, None] + kcyc * de
            p = p - jnp.round(p)
            return acc + v * jnp.exp(2j * jnp.pi * p), None
        acc, _ = jax.lax.scan(one, acc, (rc, ti, tf, ph, wt, rt, wr, rr))
        return acc

    acc = jnp.zeros((nb, BLOCK), jnp.complex64)
    dd_ = jnp.asarray(d)
    for p0 in range(0, len(S), chunk):
        sl = slice(p0, min(len(S), p0 + chunk))
        rc = _rc_jax(jnp.asarray(S[sl], jnp.complex64), nfft).astype(jnp.complex64)
        ct = _centers(tx[sl], None if rcv is None else rcv[sl], ref[sl], cen, nfft, inv_dr, kcyc)
        z = np.zeros((sl.stop - sl.start, nb), np.float32)
        acc = step(acc, rc, ct['ti'], ct['tf'], ct['ph'], ct['wt'], ct['rt'], ct.get('wr', np.zeros_like(ct['wt'])),
                   ct.get('rr', z), dd_)
    return np.asarray(acc).reshape(-1)


def _rc_jax(S, nfft):
    import jax.numpy as jnp
    P, K = S.shape
    h = K // 2
    pad = jnp.zeros((P, nfft), S.dtype).at[:, :K - h].set(S[:, h:]).at[:, nfft - h:].set(S[:, :h])
    return jnp.fft.fftshift(jnp.fft.ifft(pad, axis=1), axes=1) * nfft


# ---------------------------------------------------------------------------------------------------- entry point

def backproject(S, ant, fmin, df, points, rcv=None, ref=None, backend='auto', upsample=8, window=True, chunk=256):
    """Complex image at points [..., 3] (shape [...], complex64). ant [P, 3] is the antenna phase center
    (monostatic) or the transmitter position when rcv [P, 3] is given. ref [P] is the range (one way, or the mean
    of the two legs) to which each pulse was motion compensated; by default the distance to the origin, the fixed
    scene reference point. A moving reference point (sliding spotlight, stripmap CPHD) passes its per-pulse range.
    backend: 'cpu' (float64 throughout), 'cuda', 'jax' (float32 with float64 block centers), or 'auto'."""
    from .api import _backend, _check_history, _check_positions
    backend = 'jax' if backend == 'tpu' else _backend(backend)
    S = _check_history(np.asarray(S))
    P, K = S.shape
    tx = _check_positions(ant, 'antenna (transmitter) positions', P)
    rcv = None if rcv is None else _check_positions(rcv, 'receiver positions', P)
    points = np.asarray(points, np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or points.size == 0:
        raise ValueError(f'points must be [..., 3] with at least one point, got shape {points.shape}')
    if ref is None:
        ref = np.linalg.norm(tx, axis=1) if rcv is None else 0.5 * (np.linalg.norm(tx, axis=1) + np.linalg.norm(rcv, axis=1))
    ref = np.asarray(ref, np.float64)
    if ref.shape != (P,):
        raise ValueError(f'ref must hold one range per pulse, shape ({P},), got {ref.shape}')
    shape = points.shape[:-1]
    if window:
        wp, wk = _window(P, K)
        S = S * wp[:, None] * wk[None, :]
    S = S.astype(np.complex64)
    nfft = 1 << int(np.ceil(np.log2(upsample * K)))
    inv_dr = 2.0 * df * nfft / C
    kcyc = 2.0 * (fmin + (K // 2) * df) / C
    if backend == 'tpu':
        backend = 'jax'
    if backend == 'cpu':
        return _run_cpu(S, tx, rcv, ref, points.reshape(-1, 3), nfft, inv_dr, kcyc, chunk).reshape(shape)
    order, cen, d = _order(points)
    if backend == 'cuda':
        flat = _run_cuda(S, tx, rcv, ref, cen, d, nfft, inv_dr, kcyc, chunk)
    elif backend in ('jax', 'tpu'):
        flat = _run_jax(S, tx, rcv, ref, cen, d, nfft, inv_dr, kcyc, chunk)
    else:
        raise ValueError(f'unknown backend {backend!r}')
    out = np.empty(len(order), np.complex64)
    out[order] = flat[:len(order)]
    return out.reshape(shape)


def plane_points(nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), height=None):
    """Points [nx, ny, 3] of the grid form_image uses (x_i = (i - nx/2) spx along e1, y_j = (j - ny/2) spy along e2),
    optionally displaced along the plane normal by height [nx, ny] (a DEM sampled on the grid)."""
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    X, Y = np.meshgrid((np.arange(nx) - nx / 2.0) * spx, (np.arange(ny) - ny / 2.0) * spy, indexing='ij')
    p = X[..., None] * e1 + Y[..., None] * e2
    if height is not None:
        n = np.cross(e1, e2)
        n = n / np.linalg.norm(n) * np.sign(n[2] if abs(n[2]) > 1e-12 else 1.0)
        p = p + np.asarray(height, np.float64)[..., None] * n
    return p
