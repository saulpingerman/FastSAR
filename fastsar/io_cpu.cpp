// Row passes over a phase history for the readers (io.py), each on all cores.
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "par.hpp"

// S [P, K] complex64 (as float pairs), in place: S[p, k] *= exp(j (a[p] + (first + k) b[p])). The phasor is formed
// exactly in double at the start of every block of 16 samples and stepped inside it by the block's 16 rotations in
// float, so the phase error stays at float32's rounding for any K. first: a multiple of 16, the place of the row's
// first sample in a longer row whose ramp this is a part of (a reader's sample selection): the blocks, and with them
// every value, are then those of the pass over the whole row.
static void ramp_rows(float* S, long P, long K, const double* a, const double* b, long first)
{
  par::for_dynamic(P, 16, [&](long p) {
    float* s = S + 2 * p * K;
    const double a0 = a[p], b0 = b[p];
    if (a0 == 0.0 && b0 == 0.0) return;
    float wr[16], wi[16];
    for (int j = 0; j < 16; ++j) { wr[j] = (float)std::cos(j * b0); wi[j] = (float)std::sin(j * b0); }
    for (long k0 = 0; k0 < K; k0 += 16) {
      const double ph = std::remainder(a0 + (double)(first + k0) * b0, 6.283185307179586476925286766559);
      const float zr = (float)std::cos(ph), zi = (float)std::sin(ph);
      const int n = (int)(K - k0 < 16 ? K - k0 : 16);
      float* q = s + 2 * k0;
      #pragma omp simd
      for (int j = 0; j < n; ++j) {
        const float er = zr * wr[j] - zi * wi[j], ei = zr * wi[j] + zi * wr[j];
        const float xr = q[2 * j], xi = q[2 * j + 1];
        q[2 * j] = xr * er - xi * ei;
        q[2 * j + 1] = xr * ei + xi * er;
      }
    }
  });
}

extern "C" void phase_ramp(float* S, long P, long K, const double* a, const double* b)
{
  ramp_rows(S, P, K, a, b, 0);
}

extern "C" void phase_ramp_from(float* S, long P, long K, const double* a, const double* b, long first)
{
  ramp_rows(S, P, K, a, b, first);
}

// The signal array of a CPHD file into complex64: src [P, K, 2] big-endian samples of kind 1 (int8), 2 (int16) or
// 4 (float32); amp [P] the per-vector scale or null; conj: the samples conjugated; a sample with a part that is not
// finite becomes 0; a, b (or null): phase_ramp's, applied in the same pass. empty[p]: every sample of the row is 0.
extern "C" void cphd_load(const void* src, int kind, long P, long K, const float* amp, int conj,
                          const double* a, const double* b, float* out, unsigned char* empty)
{
  const float sg = conj ? -1.f : 1.f;
  par::for_dynamic(P, 8, [&](long p) {
    float* o = out + 2 * p * K;
    const float g = amp ? amp[p] : 1.f;
    const long n = 2 * K;
    if (kind == 4) {
      const uint32_t* s = (const uint32_t*)src + 2 * p * K;
      #pragma omp simd
      for (long i = 0; i < n; ++i) { uint32_t v = __builtin_bswap32(s[i]); float f; std::memcpy(&f, &v, 4); o[i] = f; }
    } else if (kind == 2) {
      const uint16_t* s = (const uint16_t*)src + 2 * p * K;
      #pragma omp simd
      for (long i = 0; i < n; ++i) o[i] = (float)(int16_t)__builtin_bswap16(s[i]);
    } else {
      const int8_t* s = (const int8_t*)src + 2 * p * K;
      #pragma omp simd
      for (long i = 0; i < n; ++i) o[i] = (float)s[i];
    }
    float any = 0.f;
    #pragma omp simd reduction(+:any)
    for (long k = 0; k < K; ++k) {
      float re = o[2 * k] * g, im = o[2 * k + 1] * g * sg;
      const bool ok = std::isfinite(re) && std::isfinite(im);     // (-ffast-math is not among the flags)
      re = ok ? re : 0.f;
      im = ok ? im : 0.f;
      o[2 * k] = re; o[2 * k + 1] = im;
      any += (re != 0.f || im != 0.f) ? 1.f : 0.f;
    }
    empty[p] = any == 0.f;
  });
  if (a && b) phase_ramp(out, P, K, a, b);
}

// A part of the signal array, read from the file itself: rows p0 .. p0 + P - 1 of an array of Kf samples per row
// that starts at byte off of the open file fd, and of each row the samples k0 .. k0 + K - 1, into out [P, K] as
// cphd_load converts them (amp [P]: the scale of the rows read, or null). The bytes are asked of the file with pread
// on the worker threads, a block of rows at a time: one call per block when whole rows are read, one per row
// otherwise, so that the file is read in proportion to the selection. -> the bytes read, or the negative errno of
// a read that failed (-EIO when the file ends before the selection does).
extern "C" long long cphd_read(int fd, long long off, int kind, long Kf, long p0, long P, long k0, long K,
                               const float* amp, int conj, float* out, unsigned char* empty)
{
  const long long isz = 2LL * kind, rowb = K * isz;
  const long rows = (long)std::min<long long>(256, std::max<long long>(1, (2LL << 20) / rowb));
  std::atomic<long long> total(0);
  std::atomic<int> err(0);
  par::for_dynamic((P + rows - 1) / rows, 1, [&](long blk) {
    static thread_local std::vector<char> buf;
    const long r0 = blk * rows, nr = std::min(P, r0 + rows) - r0;
    if (err.load()) return;
    if ((long long)buf.size() < nr * rowb) buf.resize(nr * rowb);
    const long calls = K == Kf ? 1 : nr;                      // whole rows lie end to end in the file
    const long long each = K == Kf ? nr * rowb : rowb;
    for (long c = 0; c < calls; ++c) {
      long long at = off + ((long long)(p0 + r0 + c) * Kf + k0) * isz, done = 0;
      while (done < each) {
        const ssize_t n = pread(fd, buf.data() + c * each + done, (size_t)(each - done), (off_t)(at + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { err.store(n < 0 ? errno : EIO); return; }
        done += n;
      }
    }
    total += nr * rowb;
    cphd_load(buf.data(), kind, nr, K, amp ? amp + r0 : nullptr, conj, nullptr, nullptr, out + 2 * r0 * K, empty + r0);
  });
  return err.load() ? -(long long)err.load() : total.load();
}
