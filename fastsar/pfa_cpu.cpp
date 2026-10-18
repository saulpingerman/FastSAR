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
  par::for_dynamic(h, 4, [&](long i) {
    float wx[16], wy[16];
    double px[8], py[8];
    const double x = (i0 + i - nx / 2.0) * spx + ox;
    const double xn = x / dsx;
    px[0] = 1.0;
    for (int k = 1; k <= deg; ++k) px[k] = px[k - 1] * xn;
    float* orow = out + 2 * ((long)i * ostride);
    for (int c = 0; c < n; ++c) {
      const double y = (jb + c - ny / 2.0) * spy + oy, yn = y / dsy;
      py[0] = 1.0;
      for (int k = 1; k <= deg; ++k) py[k] = py[k - 1] * yn;
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
      orow[2 * c] = g * (ar * cr - ai * ci); orow[2 * c + 1] = g * (ar * ci + ai * cr);
    }
  });
}
