// The quick look's polar format on the CPU (quick.py): the range resampling, the azimuth resampling for pulses that
// are not evenly spaced in the tangent of the look angle, and the final resampling at the apparent positions.
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

// out[r, j] = row r of x [R, n] (complex) at the position pos0[r] + j step[r], j < n_out; zero where the position
// lies outside (-0.5, n - 0.5).
extern "C" void ql_resample_rows(const float* x, long R, long n, const double* pos0, const double* step, long n_out,
                                 const float* ptab, int Q, float* out) {
  const int half = T / 2;
  par::for_dynamic(R, 4, [&](long r) {
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

// out[m, j] = column m of A [R, M] (complex) at the row position q(t) for t = (ka0 + j dka) inv_kr[m], j < n_out:
// q by linear interpolation in qtab [nt], the row position at t = t0 + i / inv_dt (the pulses' own, uneven, spacing in
// t), formed in double. Zero where t lies outside the table or the position outside (-0.5, R - 0.5). The columns are
// taken 64 at a time: where their taps start at one row and lie inside the array, each tap is a product of 64
// consecutive samples of a row with the columns' weights, added across the columns at once.
extern "C" void ql_resample_cols(const float* A, long R, long M, const double* inv_kr, double ka0, double dka,
                                 const double* qtab, long nt, double t0, double inv_dt, long n_out,
                                 const float* ptab, int Q, float* out) {
  const int half = T / 2, MB = 64;
  par::for_dynamic((M + MB - 1) / MB, 1, [&](long mb) {
    const long m0 = mb * MB;
    const int nm = (int)(std::min(M, m0 + MB) - m0);
    float wb[T][MB], acc[2 * MB];
    double pos[MB];
    long first[MB];
    bool in[MB];
    for (long j = 0; j < n_out; ++j) {
      const double ka = ka0 + j * dka;
      bool together = true;
      for (int c = 0; c < nm; ++c) {
        const double u = (ka * inv_kr[m0 + c] - t0) * inv_dt;
        double p = -1.0;
        if (u >= 0.0 && u < nt - 1) {
          const long k = (long)u;
          p = qtab[k] + (u - k) * (qtab[k + 1] - qtab[k]);
        }
        in[c] = p > -0.5 && p < R - 0.5;
        pos[c] = p;
        first[c] = in[c] ? (long)std::floor(p) - half + 1 : 0;
        together = together && in[c] && first[c] == first[0];
      }
      if (together && first[0] >= 0 && first[0] + T <= R) {
        for (int c = 0; c < nm; ++c) {
          const float* w = ptab + (long)((pos[c] - std::floor(pos[c])) * Q + 0.5) * T;
          for (int t = 0; t < T; ++t) wb[t][c] = w[t];
        }
        std::fill(acc, acc + 2 * nm, 0.f);
        for (int t = 0; t < T; ++t) {
          const float* v = A + 2 * ((first[0] + t) * M + m0);
          const float* w = wb[t];
          #pragma omp simd
          for (int c = 0; c < nm; ++c) { acc[2 * c] += w[c] * v[2 * c]; acc[2 * c + 1] += w[c] * v[2 * c + 1]; }
        }
        for (int c = 0; c < nm; ++c) {
          float* o = out + 2 * ((m0 + c) * n_out + j);
          o[0] = acc[2 * c]; o[1] = acc[2 * c + 1];
        }
        continue;
      }
      for (int c = 0; c < nm; ++c) {                 // an edge of the aperture, or columns whose taps start apart
        float re = 0.f, im = 0.f;
        if (in[c]) {
          const float* w = ptab + (long)((pos[c] - std::floor(pos[c])) * Q + 0.5) * T;
          float tot = 0.f;
          for (int t = 0; t < T; ++t) {
            const long i = first[c] + t;
            if (i >= 0 && i < R) {
              const float* v = A + 2 * (i * M + m0 + c);
              re += w[t] * v[0]; im += w[t] * v[1]; tot += w[t];
            }
          }
          const float inv = 1.f / std::max(tot, 1e-12f);
          re *= inv; im *= inv;
        }
        float* o = out + 2 * ((m0 + c) * n_out + j);
        o[0] = re; o[1] = im;
      }
    }
  });
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
// value (float pairs [n, h]); out's rows are ostride values apart.
extern "C" void ql_gather(const float* base, long span, long width, long Nx2, long Ny2, long rb0, long cb0,
                          int h, int n, int i0, int jb, int nx, int ny, double spx, double spy, double ov,
                          double dsx, double dsy, const double* cx, const double* cy, const int* pw, int npow, int deg,
                          double kr_c, double ka_c, int carrier, float gain, int taps, const float* ptab,
                          const float* ptab16, int Q, int detect, float* out, long ostride) {
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
      if (ix >= 0 && ix + 8 <= width && iy >= 0 && iy + taps <= span) {          // every tap inside the block
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
