// The quick look's polar format on the CPU (quick.py): the azimuth resampling for pulses that are not evenly
// spaced in the tangent of the look angle, and the final resampling at the apparent positions with its kernel read
// from a table. The range resampling is pfa_cpu.cpp's.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "par.hpp"

// a kernel of `taps` taps from a table of its values at tabn points per unit offset over [0, taps / 2], linearly
// interpolated (symmetric in d): the Kaiser-windowed sinc of pfa2.kernel_table, or quick.py's Lanczos table
static inline float tabled(float d, int taps, const float* tab, int tabn) {
  const float ad = std::fabs(d);
  if (ad >= taps * 0.5f) return 0.f;
  const float u = ad * tabn;
  const int k = (int)u;
  return tab[k] + (tab[k + 1] - tab[k]) * (u - k);
}

// out[m, j] = column m of A [R, M] (complex) at the row position q(t) for t = (ka0 + j dka) inv_kr[m], j < n_out:
// q by linear interpolation in qtab [nt], the row position at t = t0 + i / inv_dt (the pulses' own, uneven, spacing in
// t). The position is formed in double; the kernel's weights are normalised; an output whose t lies outside the table
// or whose position lies outside (-0.5, R - 0.5) is zero. Blocks of columns, so that the rows read stay in cache.
extern "C" void ql_resample_cols(const float* A, int R, int M, const double* inv_kr, double ka0, double dka,
                                 const double* qtab, int nt, double t0, double inv_dt, int n_out,
                                 int taps, const float* tab, int tabn, float* out) {
  const int half = taps / 2, MB = 64;
  par::for_dynamic((M + MB - 1) / MB, 1, [&](long mb) {
    const int m0 = (int)mb * MB, m1 = std::min(M, m0 + MB);
    float w[32];
    for (int j = 0; j < n_out; ++j) {
      const double ka = ka0 + j * dka;
      for (int m = m0; m < m1; ++m) {
        float ar = 0.f, ai = 0.f;
        const double u = (ka * inv_kr[m] - t0) * inv_dt;
        if (u >= 0.0 && u < nt - 1) {
          const int k = (int)u;
          const double pos = qtab[k] + (u - k) * (qtab[k + 1] - qtab[k]);
          if (pos > -0.5 && pos < R - 0.5) {
            const double fl = std::floor(pos);
            const int i0 = (int)fl - half + 1;
            const float fr = (float)(pos - fl);
            float tot = 0.f;
            for (int t = 0; t < taps; ++t) {
              const int i = i0 + t;
              w[t] = (i >= 0 && i < R) ? tabled(fr - (float)(t - half + 1), taps, tab, tabn) : 0.f;
              tot += w[t];
            }
            const float inv = 1.f / std::max(tot, 1e-12f);
            for (int t = 0; t < taps; ++t) {
              const int i = std::min(std::max(i0 + t, 0), R - 1);
              const float* v = A + 2 * ((long)i * M + m);
              ar += w[t] * inv * v[0]; ai += w[t] * inv * v[1];
            }
          }
        }
        float* o = out + 2 * ((long)m * n_out + j);
        o[0] = ar; o[1] = ai;
      }
    }
  });
}

// The image at the apparent positions, range-major: for the range pixel jb + c (c < n) and the azimuth pixel i0 + i
// (i < h) of an nx by ny grid, the apparent position (Xa, Ya) of the pixel's own (x, y) from the distortion
// polynomial in float64 (as pfa_cpu.cpp's pfa_gather), and the baseband image, oversampled by ov, read there with a
// kernel of `taps` taps per axis from the table. base [span, width] holds the rows rb0 .. and the columns cb0 .. of
// the oversampled image, whose index (fy, fx) lies at ((fy - Ny2 / 2) spy / ov, (fx - Nx2 / 2) spx / ov). carrier:
// the result is multiplied by exp(j (kr_c Ya - ka_c Xa)), the phase backprojection gives the pixel. detect: the
// amplitude is written (float [n, h]) in place of the complex value (float pairs [n, h]); out's rows are ostride
// values apart.
extern "C" void ql_gather(const float* base, long span, long width, long Nx2, long Ny2, long rb0, long cb0,
                          int h, int n, int i0, int jb, int nx, int ny, double spx, double spy, double ov,
                          double dsx, double dsy, const double* cx, const double* cy, const int* pw, int npow, int deg,
                          double kr_c, double ka_c, int carrier, float gain, int taps, const float* tab, int tabn,
                          int detect, float* out, long ostride) {
  const int half = taps / 2;
  par::for_dynamic(n, 2, [&](long c) {
    float wx[32], wy[16];
    double px[8], py[8];
    const double y = (jb + c - ny / 2.0) * spy, yn = y / dsy;
    py[0] = 1.0;
    for (int k = 1; k <= deg; ++k) py[k] = py[k - 1] * yn;
    float* orow = out + c * ostride * (detect ? 1 : 2);
    for (int i = 0; i < h; ++i) {
      const double x = (i0 + i - nx / 2.0) * spx, xn = x / dsx;
      px[0] = 1.0;
      for (int k = 1; k <= deg; ++k) px[k] = px[k - 1] * xn;
      double Xa = x, Ya = y;
      for (int k = 0; k < npow; ++k) {
        const double t = px[pw[2 * k]] * py[pw[2 * k + 1]];
        Xa += cx[k] * t; Ya += cy[k] * t;
      }
      const double fx = Xa / (spx / ov) + Nx2 / 2.0, fy = Ya / (spy / ov) + Ny2 / 2.0;
      const double fix = std::floor(fx), fiy = std::floor(fy);
      const long ix = (long)fix - cb0 - half + 1, iy = (long)fiy - rb0 - half + 1;     // the first tap, in base
      const float tx = (float)(fx - fix), ty = (float)(fy - fiy);
      float sx = 0.f, sy = 0.f;
      for (int a = 0; a < taps; ++a) {
        const float o = (float)(a - half + 1);
        const float u = tabled(tx - o, taps, tab, tabn), v = tabled(ty - o, taps, tab, tabn);
        wx[2 * a] = wx[2 * a + 1] = u; wy[a] = v;
        sx += u; sy += v;
      }
      float ar = 0.f, ai = 0.f;
      if (ix >= 0 && ix + taps <= width && iy >= 0 && iy + taps <= span) {       // every tap inside the block
        for (int a = 0; a < taps; ++a) {
          const float* row = base + 2 * ((iy + a) * width + ix);
          float re = 0.f, im = 0.f;
          #pragma omp simd reduction(+:re, im)
          for (int b = 0; b < taps; ++b) { re += wx[2 * b] * row[2 * b]; im += wx[2 * b + 1] * row[2 * b + 1]; }
          ar += wy[a] * re; ai += wy[a] * im;
        }
      } else {
        for (int a = 0; a < taps; ++a) {
          const long yy = std::min(std::max(iy + a, 0L), span - 1);
          float re = 0.f, im = 0.f;
          for (int b = 0; b < taps; ++b) {
            const long xx = std::min(std::max(ix + b, 0L), width - 1);
            const float* v = base + 2 * (yy * width + xx);
            re += wx[2 * b] * v[0]; im += wx[2 * b + 1] * v[1];
          }
          ar += wy[a] * re; ai += wy[a] * im;
        }
      }
      const float g = gain / (sx * sy);
      ar *= g; ai *= g;
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

// out[r, k0 + k] = S[r, k] wk[k] for k < K and 0 in the rest of the row of Kf: the weighted, zero-padded rows that the
// range transforms take (one pass, in place of a product and two fills)
extern "C" void ql_weight_pad(const float* S, long R, long K, const float* wk, long k0, long Kf, float* out) {
  par::for_dynamic(R, 4, [&](long r) {
    const float* s = S + 2 * r * K;
    float* o = out + 2 * r * Kf;
    std::fill(o, o + 2 * k0, 0.f);
    #pragma omp simd
    for (long k = 0; k < K; ++k) { o[2 * (k0 + k)] = s[2 * k] * wk[k]; o[2 * (k0 + k) + 1] = s[2 * k + 1] * wk[k]; }
    std::fill(o + 2 * (k0 + K), o + 2 * Kf, 0.f);
  });
}
