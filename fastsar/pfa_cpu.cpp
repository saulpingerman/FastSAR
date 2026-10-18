// Polar format on the CPU: the final resampling (the planar-wavefront displacement removed) as one pass over the
// output pixels, the apparent position of each pixel from the distortion polynomial in float64, a Lanczos kernel of
// `taps` taps per axis in float32, and the carrier restored at the apparent position. The same arithmetic as the
// CUDA kernel of pfa2.py (pfa_gather) and the JAX gather it replaces.
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "par.hpp"

static inline float sincf_(float d) {
  if (std::fabs(d) < 1e-6f) return 1.f;
  const float a = 3.14159265358979f * d;
  return std::sin(a) / a;
}

extern "C" void pfa_gather(const float* base, int span, int Nx2, int Ny2, int rb0, int h, int n, int i0, int jb, int nx, int ny,
                           double spx, double spy, double ox, double oy, double ov, double dsx, double dsy,
                           const double* cx, const double* cy, const int* pw, int npow, int deg, double kr_c, double ka_c,
                           int sy_, int sx_, float norm, float scale, int taps, float* out, long ostride) {
  const int half = taps / 2;
  const float g = norm * scale;
  // parallel over the output's range columns c; along i (azimuth) the apparent x advances by about ov samples per
  // pixel, so the taps of consecutive pixels read consecutive parts of the same base rows
  par::for_dynamic(n, 2, [&](long c) {
    float wx[16], wy[16];
    double px[8], py[8];
    const double y = (jb + c - ny / 2.0) * spy + oy, yn = y / dsy;
    py[0] = 1.0;
    for (int k = 1; k <= deg; ++k) py[k] = py[k - 1] * yn;
    for (int i = 0; i < h; ++i) {
      const double x = (i0 + i - nx / 2.0) * spx + ox, xn = x / dsx;
      px[0] = 1.0;
      for (int k = 1; k <= deg; ++k) px[k] = px[k - 1] * xn;
      double Xa = x, Ya = y;
      for (int k = 0; k < npow; ++k) {
        const double t = px[pw[2 * k]] * py[pw[2 * k + 1]];
        Xa += cx[k] * t; Ya += cy[k] * t;
      }
      const double fx = Xa / (spx / ov) + Nx2 / 2.0, fy = Ya / (spy / ov) + Ny2 / 2.0;
      const double fix = std::floor(fx), fiy = std::floor(fy);
      const int ix = (int)fix, iyl = (int)fiy - rb0;
      const float tx = (float)(fx - fix), ty = (float)(fy - fiy);
      float sx = 0.f, sy = 0.f;
      for (int a = 0; a < taps; ++a) {
        const float o = (float)(a - half + 1), dx = tx - o, dy = ty - o;
        wx[a] = sincf_(dx) * sincf_(dx / half); wy[a] = sincf_(dy) * sincf_(dy / half);
        sx += wx[a]; sy += wy[a];
      }
      float ar = 0.f, ai = 0.f;
      for (int a = 0; a < taps; ++a) {
        const int yy = std::min(std::max(iyl + a - half + 1, 0), span - 1);
        const float* row = base + 2 * ((long)yy * Nx2);
        const float wya = wy[a] / sy;
        for (int b = 0; b < taps; ++b) {
          const int xx = std::min(std::max(ix + b - half + 1, 0), Nx2 - 1);
          const float wgt = wya * (wx[b] / sx);
          ar += wgt * row[2 * xx]; ai += wgt * row[2 * xx + 1];
        }
      }
      const double ph = sy_ * kr_c * Ya + sx_ * ka_c * Xa;
      const float cr = (float)std::cos(ph), ci = (float)std::sin(ph);
      float* o = out + 2 * ((long)i * ostride + c);
      o[0] = g * (ar * cr - ai * ci); o[1] = g * (ar * ci + ai * cr);
    }
  });
}

// Direct resampling in place of the chirp-z transforms (the CUDA kernels resample_rows / resample_cols of pfa2.py):
// Kaiser-windowed sinc of `taps` taps, weights normalised, outputs whose index lies outside (-0.5, n - 0.5) zero.
static inline float kaiser_sinc(float d, int taps, float beta, float inv_i0) {
  const float half = taps * 0.5f;
  const float x = std::min(std::max(d / half, -1.f), 1.f);
  const float a = 0.5f * beta * std::sqrt(std::max(0.f, 1.f - x * x));
  float term = 1.f, sum = 1.f;
  for (int k = 1; k <= 12; ++k) { term *= (a / k) * (a / k); sum += term; }
  const float sc = sincf_(d);
  return std::fabs(d) < half ? sc * sum * inv_i0 : 0.f;
}

// out[r, j] = row r of x [R, n] (complex) at index pos0[r] + j * step[r], j < n_out
extern "C" void pfa_resample_rows(const float* x, int R, int n, const float* pos0, const float* step, int n_out,
                                  int taps, float beta, float inv_i0, float* out) {
  const int half = taps / 2;
  par::for_dynamic(R, 4, [&](long r) {
    const float* xr = x + 2 * ((long)r * n);
    float* orow = out + 2 * ((long)r * n_out);
    float w[32];
    for (int j = 0; j < n_out; ++j) {
      const float pos = pos0[r] + j * step[r];
      float ar = 0.f, ai = 0.f;
      if (pos > -0.5f && pos < n - 0.5f) {
        const int i0 = (int)std::floor(pos) - half + 1;
        float tot = 0.f;
        for (int t = 0; t < taps; ++t) { const int i = i0 + t; w[t] = (i >= 0 && i < n) ? kaiser_sinc(pos - i, taps, beta, inv_i0) : 0.f; tot += w[t]; }
        const float inv = 1.f / std::max(tot, 1e-12f);
        for (int t = 0; t < taps; ++t) { const int i = std::min(std::max(i0 + t, 0), n - 1); ar += w[t] * inv * xr[2 * i]; ai += w[t] * inv * xr[2 * i + 1]; }
      }
      orow[2 * j] = ar; orow[2 * j + 1] = ai;
    }
  });
}

// out[m, j] = column m of A [R, M] (complex) at row index pos0[m] + j * step[m], j < n_out; blocks of columns so
// that the rows read stay in cache
extern "C" void pfa_resample_cols(const float* A, int R, int M, const float* pos0, const float* step, int n_out,
                                  int taps, float beta, float inv_i0, float* out) {
  const int half = taps / 2, MB = 64;
  par::for_dynamic((M + MB - 1) / MB, 1, [&](long mb) {
    const int m0 = (int)mb * MB, m1 = std::min(M, m0 + MB);
    float w[32];
    for (int j = 0; j < n_out; ++j)
      for (int m = m0; m < m1; ++m) {
        const float pos = pos0[m] + j * step[m];
        float ar = 0.f, ai = 0.f;
        if (pos > -0.5f && pos < R - 0.5f) {
          const int i0 = (int)std::floor(pos) - half + 1;
          float tot = 0.f;
          for (int t = 0; t < taps; ++t) { const int i = i0 + t; w[t] = (i >= 0 && i < R) ? kaiser_sinc(pos - i, taps, beta, inv_i0) : 0.f; tot += w[t]; }
          const float inv = 1.f / std::max(tot, 1e-12f);
          for (int t = 0; t < taps; ++t) { const int i = std::min(std::max(i0 + t, 0), R - 1); const float* v = A + 2 * ((long)i * M + m); ar += w[t] * inv * v[0]; ai += w[t] * inv * v[1]; }
        }
        float* o = out + 2 * ((long)m * n_out + j);
        o[0] = ar; o[1] = ai;
      }
  });
}
