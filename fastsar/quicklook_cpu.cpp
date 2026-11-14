// The quick look's polar format on the CPU (quick.py): the range resampling, the azimuth resampling for pulses that
// are not evenly spaced in the tangent of the look angle, the final resampling at the apparent positions, and the
// passes that prepare the arrays for the transforms.
//
// Each kernel reads its interpolation weights from a polyphase table: row q of [Q + 1, taps] holds the kernel's
// taps, normalised to unit sum, for the fractional position q / Q (quick.py's _polyphase), and a position takes the
// nearest row. With Q = 2048 the position is rounded by at most 1 / 4096 of a sample, a phase of 5e-4 rad at the
// edge of the band the kernels are asked to read (-66 dB); the weights then cost nothing per tap, and the sums over
// the taps are plain products that the compiler vectorizes. A tap outside the array counts as zero and the
// remaining weights are normalised again, as pfa_cpu.cpp's kernels do.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "par.hpp"

static const int T = 16;       // taps of the range and azimuth resampling (quick.py's TAPS)

// f(i) for i < n in chunks on all cores, or, serial, on the calling thread: quick.py runs the range and the azimuth
// stage on its own pool, a block of rows per thread, and calls the kernels there with serial set
template <class F> static void rows_of(long n, int chunk, int serial, F&& f) {
  if (serial) { for (long i = 0; i < n; ++i) f(i); }
  else par::for_dynamic(n, chunk, f);
}

// out[r, j] = row r of x [R, n] (complex) at the position pos0[r] + j step[r], j < n_out; zero where the position
// lies outside (-0.5, n - 0.5).
extern "C" void ql_resample_rows(const float* x, long R, long n, const double* pos0, const double* step, long n_out,
                                 const float* ptab, int Q, float* out, int serial) {
  const int half = T / 2;
  rows_of(R, 4, serial, [&](long r) {
    const float* xr = x + 2 * r * n;
    float* o = out + 2 * r * n_out;
    const double p0 = pos0[r], st = step[r];
    for (long j = 0; j < n_out; ++j) {
      const double pos = p0 + j * st;
      float re = 0.f, im = 0.f;
      if (pos > -0.5 && pos < n - 0.5) {
        const double fl = std::floor(pos);
        const long i0 = (long)fl - half + 1;
        const float* w = ptab + (long)((pos - fl) * Q + 0.5) * T;
        if (i0 >= 0 && i0 + T <= n) {
          const float* v = xr + 2 * i0;
          #pragma omp simd reduction(+:re, im)
          for (int t = 0; t < T; ++t) { re += w[t] * v[2 * t]; im += w[t] * v[2 * t + 1]; }
        } else {
          float tot = 0.f;
          for (int t = 0; t < T; ++t) {
            const long i = i0 + t;
            if (i >= 0 && i < n) { re += w[t] * xr[2 * i]; im += w[t] * xr[2 * i + 1]; tot += w[t]; }
          }
          const float inv = 1.f / std::max(tot, 1e-12f);
          re *= inv; im *= inv;
        }
      }
      o[2 * j] = re; o[2 * j + 1] = im;
    }
  });
}

// out[j, m] = row m of A [M, R] (complex: the pulses of one range wavenumber, end to end) at the position q(t) for
// t = (ka0 + j dka) inv_kr[m], j < n_out: q by linear interpolation in qtab [nt], the position at t = t0 + i / inv_dt
// (the pulses' own, uneven, spacing in t), formed in double. Zero where t lies outside the table or the position
// outside (-0.5, R - 0.5). The output is azimuth-wavenumber major, so that the range transform that follows runs
// along its rows: out[j, m] lies at out + j ostride + m (complex values), the M columns of an array ostride wide
// that this call's rows of A make. The rows of A are taken 64 at a time, 64 consecutive values of an output row
// written at once.
extern "C" void ql_resample_az(const float* A, long M, long R, const double* inv_kr, double ka0, double dka,
                               const double* qtab, long nt, double t0, double inv_dt, long n_out,
                               const float* ptab, int Q, float* out, long ostride, int serial) {
  const int half = T / 2, MB = 64;
  rows_of((M + MB - 1) / MB, 1, serial, [&](long mb) {
    const long m0 = mb * MB;
    const int nm = (int)(std::min(M, m0 + MB) - m0);
    for (long j = 0; j < n_out; ++j) {
      const double ka = ka0 + j * dka;
      float* o = out + 2 * (j * ostride + m0);
      for (int c = 0; c < nm; ++c) {
        float re = 0.f, im = 0.f;
        const double u = (ka * inv_kr[m0 + c] - t0) * inv_dt;
        if (u >= 0.0 && u < nt - 1) {
          const long k = (long)u;
          const double pos = qtab[k] + (u - k) * (qtab[k + 1] - qtab[k]);
          if (pos > -0.5 && pos < R - 0.5) {
            const double fl = std::floor(pos);
            const long i0 = (long)fl - half + 1;
            const float* w = ptab + (long)((pos - fl) * Q + 0.5) * T;
            const float* row = A + 2 * (m0 + c) * R;
            if (i0 >= 0 && i0 + T <= R) {
              const float* v = row + 2 * i0;
              #pragma omp simd reduction(+:re, im)
              for (int t = 0; t < T; ++t) { re += w[t] * v[2 * t]; im += w[t] * v[2 * t + 1]; }
            } else {
              float tot = 0.f;
              for (int t = 0; t < T; ++t) {
                const long i = i0 + t;
                if (i >= 0 && i < R) { re += w[t] * row[2 * i]; im += w[t] * row[2 * i + 1]; tot += w[t]; }
              }
              const float inv = 1.f / std::max(tot, 1e-12f);
              re *= inv; im *= inv;
            }
          }
        }
        o[2 * c] = re; o[2 * c + 1] = im;
      }
    }
  });
}

// The azimuth gate of R rows (the transforms along the pulses of R range wavenumbers, prof [R, Pf]): out[r, k], k < Na,
// is the transform's bin n (k for k < Na - Na / 2, else k - Na: bin 0 first, the negative bins at the end), cut out
// of the Pf there are or set among zeros in a longer row, times fac[k] (complex: the scale and a phase common to the
// rows) and times the gate at the bin's distance |n| bin[r] in meters: 1 within edge, a raised cosine to 0 over the
// next guard, 0 beyond. The bins of a row are 2 pi / (kr |dt| Pf) wide, wider at a lower wavenumber, so the gate is
// set in meters row by row: one gate in bins for all rows would pass the rows of the low end of a wide band
// further out, and what it passed there would wrap into the picture's far edge. On the calling thread.
extern "C" void ql_azimuth_gate(const float* prof, long R, long Pf, long Na, const float* fac, const double* bin,
                                double edge, double guard, float* out) {
  const long keep = std::min(Na, Pf), pos = keep - keep / 2, neg = keep / 2;
  for (long r = 0; r < R; ++r) {
    const float* p = prof + 2 * r * Pf;
    float* o = out + 2 * r * Na;
    const double b = bin[r];
    for (long k = 0; k < Na; ++k) {
      const long n = k < Na - Na / 2 ? k : k - Na;
      float w = 0.f;
      if (n < pos && n >= -neg) {
        const double d = std::fabs((double)n) * b;
        w = d <= edge ? 1.f : (d >= edge + guard ? 0.f : (float)(0.5 + 0.5 * std::cos((d - edge) * (3.14159265358979323846 / guard))));
      }
      if (w == 0.f) { o[2 * k] = 0.f; o[2 * k + 1] = 0.f; continue; }
      const float* v = p + 2 * (n >= 0 ? n : Pf + n);
      const float fr = fac[2 * k] * w, fi = fac[2 * k + 1] * w;
      o[2 * k] = v[0] * fr - v[1] * fi; o[2 * k + 1] = v[0] * fi + v[1] * fr;
    }
  }
}

// dst[c, r] = src[r, c] s[r] for r < R, c < C (complex): a transpose in tiles that fit the first-level cache, on all
// cores. src's rows are sstride and dst's rows dstride complex values apart; s: a real factor per source row, or
// null. The transforms of the quick look all run along rows, the way that is three times faster than along
// columns, and the arrays are turned between them.
extern "C" void ql_transpose(const float* src, long sstride, long R, long C, float* dst, long dstride, const float* s,
                             int serial) {
  const long TB = 32;
  rows_of((C + TB - 1) / TB, 1, serial, [&](long cb) {
    const long c0 = cb * TB, c1 = std::min(C, c0 + TB);
    for (long r0 = 0; r0 < R; r0 += TB) {
      const long r1 = std::min(R, r0 + TB);
      for (long r = r0; r < r1; ++r) {
        const float g = s ? s[r] : 1.f;
        const float* p = src + 2 * (r * sstride + c0);
        for (long c = c0; c < c1; ++c, p += 2) {
          float* q = dst + 2 * (c * dstride + r);
          q[0] = p[0] * g; q[1] = p[1] * g;
        }
      }
    }
  });
}

// dst[c, r] = src[r, J0 + c - jlo[r]] where that column lies in [0, W), else 0, for c < C and r < R (complex): the
// rows J0 .. of the raster's range wavenumbers out of a history whose pulses each hold their own W rows, from row
// jlo[r] on (a picture on the ground under a twisted slant plane: the band of a pulse drifts along the aperture,
// and each pulse is resampled on the rows it reaches). src's rows are W and dst's rows dstride complex values
// apart. On the calling thread.
extern "C" void ql_unshear(const float* src, long W, long R, const int* jlo, long J0, long C, float* dst, long dstride) {
  for (long c = 0; c < C; ++c) std::fill(dst + 2 * c * dstride, dst + 2 * (c * dstride + R), 0.f);
  for (long r = 0; r < R; ++r) {
    const long c0 = std::max(0L, (long)jlo[r] - J0), c1 = std::min(C, (long)jlo[r] + W - J0);
    const float* p = src + 2 * (r * W + (J0 + c0 - jlo[r]));
    for (long c = c0; c < c1; ++c, p += 2) {
      float* q = dst + 2 * (c * dstride + r);
      q[0] = p[0]; q[1] = p[1];
    }
  }
}

// The image at the apparent positions, range-major: for the range pixel jb + c (c < n) and the azimuth pixel i0 + i
// (i < h) of an nx by ny grid, the apparent position (Xa, Ya) of the pixel's own (x, y) from the distortion
// polynomial in float64 (pfa2.distortion: Xa = x + sum cx[k] xn^a yn^b over the powers (a, b), xn = x / dsx,
// yn = y / dsy; gathered per range pixel into a polynomial in xn), and the baseband image, oversampled by ov, read
// there with a kernel of `taps` taps per axis (at most 8) from its polyphase table ptab [Q + 1, taps]; ptab16
// [Q + 1, 16] holds each row's weights twice over, once for the real and once for the imaginary part of a sample,
// and zeros to 16, so that a row of taps is one product of 16 values. base
// [span, width] holds the rows rb0 .. and the columns cb0 .. of the oversampled image, whose index (fy, fx) lies at
// ((fy - Ny2 / 2) spy / ov, (fx - Nx2 / 2) spx / ov). carrier: the result is multiplied by exp(j (kr_c Ya - ka_c Xa)),
// the phase backprojection gives the pixel. detect: the amplitude is written (float [n, h]) in place of the complex
// value (float pairs [n, h]); out's rows are ostride values apart. shear: where the band's center moves with the
// azimuth wavenumber (a picture on the ground under a twisted slant plane), the baseband image is smooth along
// lines that cross `shear` columns per row, not along its columns; each row of taps is then read at the column
// where such a line through the position meets it, fx + shear (row - fy), with its own weights.
extern "C" void ql_gather(const float* base, long span, long width, long Nx2, long Ny2, long rb0, long cb0,
                          int h, int n, int i0, int jb, int nx, int ny, double spx, double spy, double ov,
                          double dsx, double dsy, const double* cx, const double* cy, const int* pw, int npow, int deg,
                          double kr_c, double ka_c, int carrier, float gain, int taps, const float* ptab,
                          const float* ptab16, int Q, int detect, float* out, long ostride, double shear) {
  const int half = taps / 2;
  const double ux = ov / spx, uy = ov / spy;
  par::for_dynamic(n, 2, [&](long c) {
    double ax[8] = {0}, ay[8] = {0}, py[8];
    const double y = (jb + c - ny / 2.0) * spy, yn = y / dsy;
    py[0] = 1.0;
    for (int k = 1; k <= deg; ++k) py[k] = py[k - 1] * yn;
    for (int k = 0; k < npow; ++k) { ax[pw[2 * k]] += cx[k] * py[pw[2 * k + 1]]; ay[pw[2 * k]] += cy[k] * py[pw[2 * k + 1]]; }
    float* orow = out + c * ostride * (detect ? 1 : 2);
    for (int i = 0; i < h; ++i) {
      const double x = (i0 + i - nx / 2.0) * spx, xn = x / dsx;
      double dx = ax[deg], dy = ay[deg];
      for (int k = deg - 1; k >= 0; --k) { dx = dx * xn + ax[k]; dy = dy * xn + ay[k]; }
      const double Xa = x + dx, Ya = y + dy;
      const double fx = Xa * ux + Nx2 / 2.0, fy = Ya * uy + Ny2 / 2.0;
      const double fix = std::floor(fx), fiy = std::floor(fy);
      const long ix = (long)fix - cb0 - half + 1, iy = (long)fiy - rb0 - half + 1;     // the first tap, in base
      const long qx = (long)((fx - fix) * Q + 0.5);
      const float* wx = ptab + qx * taps;
      const float* wy = ptab + (long)((fy - fiy) * Q + 0.5) * taps;
      float ar = 0.f, ai = 0.f;
      if (shear != 0.0) {
        for (int a = 0; a < taps; ++a) {
          const long yy = std::min(std::max(iy + a, 0L), span - 1);
          const double fxa = fx + shear * ((double)(iy + a + rb0) - fy), fl = std::floor(fxa);
          const long ixa = (long)fl - cb0 - half + 1, qa = (long)((fxa - fl) * Q + 0.5);
          float re = 0.f, im = 0.f;
          if (ixa >= 0 && ixa + 8 <= width) {
            const float* wx16 = ptab16 + qa * 16;
            const float* row = base + 2 * (yy * width + ixa);
            float acc[16];
            #pragma omp simd
            for (int u = 0; u < 16; ++u) acc[u] = wx16[u] * row[u];
            for (int u = 0; u < 16; u += 2) { re += acc[u]; im += acc[u + 1]; }
          } else {
            const float* wxa = ptab + qa * taps;
            for (int b = 0; b < taps; ++b) {
              const long xx = std::min(std::max(ixa + b, 0L), width - 1);
              const float* v = base + 2 * (yy * width + xx);
              re += wxa[b] * v[0]; im += wxa[b] * v[1];
            }
          }
          ar += wy[a] * re; ai += wy[a] * im;
        }
      } else if (ix >= 0 && ix + 8 <= width && iy >= 0 && iy + taps <= span) {   // every tap inside the block
        const float* wx16 = ptab16 + qx * 16;
        float acc[16] = {0.f};
        for (int a = 0; a < taps; ++a) {
          const float* row = base + 2 * ((iy + a) * width + ix);
          const float wa = wy[a];
          #pragma omp simd
          for (int u = 0; u < 16; ++u) acc[u] += wa * (wx16[u] * row[u]);
        }
        for (int u = 0; u < 16; u += 2) { ar += acc[u]; ai += acc[u + 1]; }
      } else {
        for (int a = 0; a < taps; ++a) {
          const long yy = std::min(std::max(iy + a, 0L), span - 1);
          float re = 0.f, im = 0.f;
          for (int b = 0; b < taps; ++b) {
            const long xx = std::min(std::max(ix + b, 0L), width - 1);
            const float* v = base + 2 * (yy * width + xx);
            re += wx[b] * v[0]; im += wx[b] * v[1];
          }
          ar += wy[a] * re; ai += wy[a] * im;
        }
      }
      ar *= gain; ai *= gain;
      if (detect) {
        orow[i] = std::sqrt(ar * ar + ai * ai);
      } else if (carrier) {
        const double ph = kr_c * Ya - ka_c * Xa;
        const float cr = (float)std::cos(ph), ci = (float)std::sin(ph);
        orow[2 * i] = ar * cr - ai * ci; orow[2 * i + 1] = ar * ci + ai * cr;
      } else {
        orow[2 * i] = ar; orow[2 * i + 1] = ai;
      }
    }
  });
}

// out[r, k0 + k] = S[r, k] wk[k] wp[r] for k < K and 0 in the rest of the row of Kf: the weighted, zero-padded rows
// that the range transforms take, in one pass. On the calling thread: quick.py calls it from its own pool, a block of
// rows per thread.
extern "C" void ql_weight_pad(const float* S, long R, long K, const float* wk, const float* wp, long k0, long Kf,
                              float* out) {
  for (long r = 0; r < R; ++r) {
    const float* s = S + 2 * r * K;
    float* o = out + 2 * r * Kf;
    const float g = wp[r];
    std::fill(o, o + 2 * k0, 0.f);
    #pragma omp simd
    for (long k = 0; k < K; ++k) {
      const float w = wk[k] * g;
      o[2 * (k0 + k)] = s[2 * k] * w; o[2 * (k0 + k) + 1] = s[2 * k + 1] * w;
    }
    std::fill(o + 2 * (k0 + K), o + 2 * Kf, 0.f);
  }
}
