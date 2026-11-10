// Table-driven interpolation of lines (resample.py): each output sample is `taps` consecutive input samples weighted
// by the row of a kernel table that is nearest to the fractional part of its position. The position of output j on
// line l is read from pos [L, m] when it is given, else it is bilinear in a coarse map of knots: lk [na] line
// indices, ok [nb] output indices, v [na, nb] source positions, the line's index line0 + l.
#include <cmath>
#include <algorithm>
#include <vector>
#include "par.hpp"

// u [L, nu] and out [L, m] hold ch floats per sample (1: real, 2: complex). A source position x (samples of the
// line before oversampling) lies at x * scale + offset in u; outside [0, nsrc - 1] the output is zero.
// A position within EDGE of an end of the line is on it: a map that reaches the end exactly lands some 1e-16 to
// either side of it in float64, and on which side depends on whether the compiler fuses the multiply-adds.
static const double EDGE = 1e-9;

extern "C" void resample_lines(const float* u, long L, long nu, int ch, const double* pos, long line0,
                               const double* lk, int na, const double* ok, int nb, const double* v,
                               double scale, double offset, double nsrc, long m,
                               const float* tab, int nphase, int taps, float* out)
{
  const int half = taps / 2;
  par::for_dynamic(L, 8, [&](long l) {
    static thread_local std::vector<double> row;
    const float* ul = u + (long)ch * l * nu;
    float* ol = out + (long)ch * l * m;
    if (!pos) {
      const double li = (double)(line0 + l);
      int i0 = (int)(std::upper_bound(lk, lk + na, li) - lk) - 1;
      i0 = std::min(std::max(i0, 0), na - 2);
      const double wi = (li - lk[i0]) / (lk[i0 + 1] - lk[i0]);
      row.resize(nb);
      for (int b = 0; b < nb; ++b) row[b] = v[(long)i0 * nb + b] * (1.0 - wi) + v[(long)(i0 + 1) * nb + b] * wi;
    }
    int seg = 0;
    for (long j = 0; j < m; ++j) {
      double x;
      if (pos) x = pos[l * m + j];
      else {
        while (seg < nb - 2 && (double)j >= ok[seg + 1]) ++seg;
        const double w = ((double)j - ok[seg]) / (ok[seg + 1] - ok[seg]);
        x = row[seg] * (1.0 - w) + row[seg + 1] * w;
      }
      if (x < 0.0 && x > -EDGE) x = 0.0;
      else if (x > nsrc - 1.0 && x < nsrc - 1.0 + EDGE) x = nsrc - 1.0;
      if (!(x >= 0.0 && x <= nsrc - 1.0)) {
        for (int c = 0; c < ch; ++c) ol[ch * j + c] = 0.f;
        continue;
      }
      const double p = x * scale + offset, fl = std::floor(p);
      const long i0 = (long)fl - half + 1;
      const float* w = tab + (long)std::lrint((p - fl) * nphase) * taps;
      const float* s = ul + ch * i0;
      if (ch == 2) {
        float ar = 0.f, ai = 0.f;
        for (int t = 0; t < taps; ++t) { ar += w[t] * s[2 * t]; ai += w[t] * s[2 * t + 1]; }
        ol[2 * j] = ar; ol[2 * j + 1] = ai;
      } else {
        float a = 0.f;
        for (int t = 0; t < taps; ++t) a += w[t] * s[t];
        ol[j] = a;
      }
    }
  });
}
