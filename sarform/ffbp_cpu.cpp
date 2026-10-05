// Factorized backprojection on the CPU: the three stages of the CUDA pipeline (sarbench/ffbp_cuda.py) as C++ with
// OpenMP and 16-wide float vectors (GCC vector extensions; AVX-512 on the instance of record, two AVX2 halves
// elsewhere).
//
//   rot_fir_k   rotate each parent row by the phase ramps of its children and decimate along frequency with the
//               Kaiser filter. The vector lanes are children: a row sample is a scalar broadcast, the rotation is one
//               complex multiply per sample per 16 children with per-lane phase steps, and the filter accumulates
//               outputs vectorized across children, so no strided read occurs. (Build 1 used a polyphase form
//               vectorized along the output columns; its stride-D gathers cost twelve instructions per useful FMA.)
//   fir_p       decimate along pulses: each input row is read once per chunk of outputs and added to every output
//               row it contributes to
//   final_tiles the 2T x 2T product per tile, tables built by two geometric recurrences (over the pulse index and
//               over the pixel index) and the product register-blocked four output rows at a time; then the
//               per-pixel quadratic phase correction
//
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC ffbp_cpu.cpp -o libffbp_cpu.so (done by ffbp_cpu.py).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <omp.h>

namespace {

typedef float v16 __attribute__((vector_size(64)));
constexpr int VL = 16;
constexpr double TWO_PI = 6.283185307179586476925286766559;
constexpr int RESEED = 2048;       // samples between exact re-evaluations of a rotation recurrence

inline void sincos_cyc(double cyc, float& s, float& c)
{
    cyc -= std::nearbyint(cyc);
    double sd, cd;
    sincos(TWO_PI * cyc, &sd, &cd);
    s = (float)sd; c = (float)cd;
}

inline v16 bcast(float x) { return v16{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x}; }

struct Scratch {
    std::vector<v16> zr, zi;        // rotated samples of a window, one vector (16 children) per sample
    std::vector<v16> yr, yi;        // outputs [Ko] per plane
    std::vector<float> tmp;
};

}  // namespace

extern "C" {

// sre, sim [Np, P, K]; c0, sl [Np, Cn, P]; taps [L]; yre, yim [Np * Cn, P, Ko].
// out_c[j] = sum_r taps[r] x_c[D j + r - pl],  x_c[k] = x[k] exp(2 pi i (c0_c + sl_c (k - kc))),  x = 0 outside [0, K).
void rot_fir_k(const float* sre, const float* sim, int Np, int P, int K, const float* c0, const float* sl, int Cn,
               const float* taps, int L, int pl, int D, int Ko, double kc, float* yre, float* yim)
{
    const int kw = D * (Ko - 1) + L;                 // window of samples the outputs read: k = D j + r - pl, j < Ko, r < L
    const int kmin = -pl, kmax = kmin + kw;          // [kmin, kmax)
    const int ncv = (Cn + VL - 1) / VL;              // child vectors
    #pragma omp parallel
    {
        static thread_local Scratch sc;
        if ((int)sc.zr.size() < kw) { sc.zr.resize(kw); sc.zi.resize(kw); }
        if ((int)sc.yr.size() < Ko) { sc.yr.resize(Ko); sc.yi.resize(Ko); }
        #pragma omp for schedule(dynamic, 2) collapse(2)
        for (int n = 0; n < Np; ++n) {
            for (int p = 0; p < P; ++p) {
                const float* xr = sre + ((size_t)n * P + p) * K;
                const float* xi = sim + ((size_t)n * P + p) * K;
                for (int cv = 0; cv < ncv; ++cv) {
                    const int cbase = cv * VL, cn = std::min(VL, Cn - cbase);
                    // per-lane phase step (one sample) and the lane constants
                    v16 sr, scs, wr, wi;
                    double cc[VL], ss[VL];
                    for (int l = 0; l < VL; ++l) {
                        const int c = cbase + std::min(l, cn - 1);
                        cc[l] = c0[((size_t)n * Cn + c) * P + p]; ss[l] = sl[((size_t)n * Cn + c) * P + p];
                        float s_, c_;
                        sincos_cyc(ss[l], s_, c_);
                        sr[l] = s_; scs[l] = c_;
                    }
                    // rotate the window: z[k] = x[k] w_c(k), w advanced by the recurrence and reseeded every RESEED samples
                    v16* zr = sc.zr.data(); v16* zi = sc.zi.data();
                    for (int k0 = kmin; k0 < kmax; k0 += RESEED) {
                        for (int l = 0; l < VL; ++l) {
                            float s_, c_;
                            sincos_cyc(cc[l] + ss[l] * ((double)k0 - kc), s_, c_);
                            wi[l] = s_; wr[l] = c_;
                        }
                        const int k1 = std::min(kmax, k0 + RESEED);
                        for (int k = k0; k < k1; ++k) {
                            const bool ok = (k >= 0) && (k < K);
                            const float vr = ok ? xr[k] : 0.f, vi = ok ? xi[k] : 0.f;
                            const v16 br = bcast(vr), bi = bcast(vi);
                            zr[k - kmin] = br * wr - bi * wi;
                            zi[k - kmin] = br * wi + bi * wr;
                            const v16 nr = wr * scs - wi * sr;
                            wi = wr * sr + wi * scs;
                            wr = nr;
                        }
                    }
                    // the filter: outputs j in blocks of 8, accumulators in registers, each rotated sample used by
                    // every output of the block whose tap window contains it
                    v16* yr = sc.yr.data(); v16* yi = sc.yi.data();
                    // window index of output j, tap r: D j + r - pl - kmin = D j + r
                    int j0 = 0;
                    for (; j0 + 8 <= Ko; j0 += 8) {
                        const v16 z0 = bcast(0.f);
                        v16 r0 = z0, r1 = z0, r2 = z0, r3 = z0, r4 = z0, r5 = z0, r6 = z0, r7 = z0;
                        v16 q0 = z0, q1 = z0, q2 = z0, q3 = z0, q4 = z0, q5 = z0, q6 = z0, q7 = z0;
                        const v16* zrr = zr + D * j0;
                        const v16* zii = zi + D * j0;
                        for (int r = 0; r < L; ++r) {
                            const v16 tv = bcast(taps[r]);
                            r0 += tv * zrr[r];         q0 += tv * zii[r];
                            r1 += tv * zrr[r + D];     q1 += tv * zii[r + D];
                            r2 += tv * zrr[r + 2 * D]; q2 += tv * zii[r + 2 * D];
                            r3 += tv * zrr[r + 3 * D]; q3 += tv * zii[r + 3 * D];
                            r4 += tv * zrr[r + 4 * D]; q4 += tv * zii[r + 4 * D];
                            r5 += tv * zrr[r + 5 * D]; q5 += tv * zii[r + 5 * D];
                            r6 += tv * zrr[r + 6 * D]; q6 += tv * zii[r + 6 * D];
                            r7 += tv * zrr[r + 7 * D]; q7 += tv * zii[r + 7 * D];
                        }
                        yr[j0] = r0; yr[j0 + 1] = r1; yr[j0 + 2] = r2; yr[j0 + 3] = r3; yr[j0 + 4] = r4; yr[j0 + 5] = r5; yr[j0 + 6] = r6; yr[j0 + 7] = r7;
                        yi[j0] = q0; yi[j0 + 1] = q1; yi[j0 + 2] = q2; yi[j0 + 3] = q3; yi[j0 + 4] = q4; yi[j0 + 5] = q5; yi[j0 + 6] = q6; yi[j0 + 7] = q7;
                    }
                    for (int j = j0; j < Ko; ++j) {
                        v16 a_r = bcast(0.f), a_i = bcast(0.f);
                        for (int r = 0; r < L; ++r) {
                            const v16 tv = bcast(taps[r]);
                            a_r += tv * zr[D * j + r];
                            a_i += tv * zi[D * j + r];
                        }
                        yr[j] = a_r; yi[j] = a_i;
                    }
                    // transpose to the child-major planes
                    for (int l = 0; l < cn; ++l) {
                        const int c = cbase + l;
                        float* orow = yre + (((size_t)n * Cn + c) * P + p) * Ko;
                        float* oimr = yim + (((size_t)n * Cn + c) * P + p) * Ko;
                        for (int j = 0; j < Ko; ++j) { orow[j] = yr[j][l]; oimr[j] = yi[j][l]; }
                    }
                }
            }
        }
    }
}

// yre, yim [B, P, Ko]; taps [L]; zre, zim [B, Po, Ko].  z[i] = sum_r taps[r] y[D i + r - pl] (rows), zero outside.
void fir_p(const float* yre, const float* yim, int B, int P, int Ko, const float* taps, int L, int pl, int D, int Po,
           float* zre, float* zim)
{
    const int IC = 16;                                        // output rows per chunk (accumulators stay in L2)
    const int nch = (Po + IC - 1) / IC;
    #pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int ch = 0; ch < nch; ++ch) {
            const int i0 = ch * IC, i1 = std::min(Po, i0 + IC);
            float* zr0 = zre + ((size_t)b * Po + i0) * Ko;
            float* zi0 = zim + ((size_t)b * Po + i0) * Ko;
            std::memset(zr0, 0, sizeof(float) * (size_t)(i1 - i0) * Ko);
            std::memset(zi0, 0, sizeof(float) * (size_t)(i1 - i0) * Ko);
            const int pa = std::max(0, D * i0 - pl), pb = std::min(P, D * (i1 - 1) - pl + L);
            for (int p = pa; p < pb; ++p) {
                const float* a = yre + ((size_t)b * P + p) * Ko;
                const float* c = yim + ((size_t)b * P + p) * Ko;
                // outputs i with 0 <= p - D i + pl < L
                int ilo = (p + pl - L) / D + 1; if (ilo < i0) ilo = i0;
                int ihi = (p + pl) / D; if (ihi > i1 - 1) ihi = i1 - 1;
                for (int i = ilo; i <= ihi; ++i) {
                    const int r = p - D * i + pl;
                    if (r < 0 || r >= L) continue;
                    const float t = taps[r];
                    float* zr = zr0 + (size_t)(i - i0) * Ko;
                    float* zi = zi0 + (size_t)(i - i0) * Ko;
                    #pragma omp simd
                    for (int j = 0; j < Ko; ++j) {
                        zr[j] += t * a[j];
                        zi[j] += t * c[j];
                    }
                }
            }
        }
    }
}

// dre, dim [B, Pf, Qf]; ux, uy [B, Pf]; dlx, dly [T] (T = 16 or 32: pixel offsets, linear in the index); ucx, ucy, rc [B].
// out[i][j] = sum_{p,q} d[p,q] exp(-2 pi i (ux_p dlx_i + uy_p dly_j)(a0 + a1 q)), then times exp(2 pi i qc).
// The tables are built and consumed a few pulses at a time so that they stay in the second-level cache; the
// accumulators of every output block are carried across the chunks in a small array.
void final_tiles(const float* dre, const float* dim, int B, int Pf, int Qf, int T, const double* ux, const double* uy,
                 const double* dlx, const double* dly, double a0, double a1, const double* ucx, const double* ucy,
                 const double* rc, double fc2, float* ore, float* oim)
{
    if (T != VL && T != 2 * VL) { std::abort(); }             // the tables are one or two vectors wide
    const int NH = T / VL;
    const int PC = std::max(1, 24576 / (Qf * T * 4 * 2));     // pulses per chunk: A and B chunks of about 24 KB each per plane
    const int NCmax = PC * Qf;
    const double spx = dlx[1] - dlx[0], spy = dly[1] - dly[0];
    #pragma omp parallel
    {
        static thread_local std::vector<v16> Ar, Ai, Br, Bi, accr, acci;
        if (Ar.size() < (size_t)NCmax * NH) { Ar.resize((size_t)NCmax * NH); Ai.resize((size_t)NCmax * NH); Br.resize((size_t)NCmax * NH); Bi.resize((size_t)NCmax * NH); }
        if (accr.size() < (size_t)T * NH) { accr.resize((size_t)T * NH); acci.resize((size_t)T * NH); }
        #pragma omp for schedule(dynamic, 4)
        for (int b = 0; b < B; ++b) {
            const float* dr = dre + (size_t)b * Pf * Qf;
            const float* di = dim + (size_t)b * Pf * Qf;
            for (size_t k = 0; k < (size_t)T * NH; ++k) { accr[k] = bcast(0.f); acci[k] = bcast(0.f); }
            for (int pc0 = 0; pc0 < Pf; pc0 += PC) {
                const int pc1 = std::min(Pf, pc0 + PC);
                const int NC = (pc1 - pc0) * Qf;
                // tables for the chunk, as [n][lane]: lane = pixel index. For each p the start w_i(q=0) and the per-lane
                // step over q are both geometric in i (dl_i is linear in i): exact at lane 0, then one ratio per lane.
                for (int tab = 0; tab < 2; ++tab) {
                    const double* u = tab == 0 ? ux : uy;
                    const double* dl = tab == 0 ? dlx : dly;
                    const double sp = tab == 0 ? spx : spy;
                    v16* Tr = tab == 0 ? Ar.data() : Br.data();
                    v16* Ti = tab == 0 ? Ai.data() : Bi.data();
                    for (int p = pc0; p < pc1; ++p) {
                        const double g = u[(size_t)b * Pf + p];
                        v16 wr0, wi0, wr1, wi1, cr0, ci0, cr1, ci1;        // phase (w) and step (c = cos, ci = sin) per lane, two halves
                        {
                            float r0s, r0c, r1s, r1c;
                            sincos_cyc(-(g * sp * a0), r0s, r0c);
                            sincos_cyc(-(g * sp * a1), r1s, r1c);
                            for (int h = 0; h < NH; ++h) {
                                float s0, c0_, s1, c1;
                                sincos_cyc(-(g * dl[h * VL] * a0), s0, c0_);
                                sincos_cyc(-(g * dl[h * VL] * a1), s1, c1);
                                v16 wr, wi, cr, ci;
                                for (int l = 0; l < VL; ++l) {
                                    wr[l] = c0_; wi[l] = s0; cr[l] = c1; ci[l] = s1;
                                    const float n0 = c0_ * r0c - s0 * r0s; s0 = c0_ * r0s + s0 * r0c; c0_ = n0;
                                    const float n1 = c1 * r1c - s1 * r1s; s1 = c1 * r1s + s1 * r1c; c1 = n1;
                                }
                                if (h == 0) { wr0 = wr; wi0 = wi; cr0 = cr; ci0 = ci; } else { wr1 = wr; wi1 = wi; cr1 = cr; ci1 = ci; }
                            }
                        }
                        const float* drp = dr + (size_t)p * Qf;
                        const float* dip = di + (size_t)p * Qf;
                        const size_t nb = (size_t)(p - pc0) * Qf;
                        for (int q = 0; q < Qf; ++q) {
                            const size_t n = nb + q;
                            if (tab == 0) {
                                const v16 d_r = bcast(drp[q]), d_i = bcast(dip[q]);
                                Tr[n * NH] = d_r * wr0 - d_i * wi0;
                                Ti[n * NH] = d_r * wi0 + d_i * wr0;
                                if (NH == 2) { Tr[n * NH + 1] = d_r * wr1 - d_i * wi1; Ti[n * NH + 1] = d_r * wi1 + d_i * wr1; }
                            } else {
                                Tr[n * NH] = wr0; Ti[n * NH] = wi0;
                                if (NH == 2) { Tr[n * NH + 1] = wr1; Ti[n * NH + 1] = wi1; }
                            }
                            v16 nr = wr0 * cr0 - wi0 * ci0; wi0 = wr0 * ci0 + wi0 * cr0; wr0 = nr;
                            if (NH == 2) { nr = wr1 * cr1 - wi1 * ci1; wi1 = wr1 * ci1 + wi1 * cr1; wr1 = nr; }
                        }
                    }
                }
                // the product over this chunk: four rows i at a time, j as NH vectors, accumulators carried in accr/acci
                const float* Arf = reinterpret_cast<const float*>(Ar.data());
                const float* Aif = reinterpret_cast<const float*>(Ai.data());
                for (int i0 = 0; i0 < T; i0 += 4) {
                    v16* ar_ = accr.data() + (size_t)i0 * NH;
                    v16* ai_ = acci.data() + (size_t)i0 * NH;
                    if (NH == 2) {
                        v16 p00 = ar_[0], p01 = ar_[1], p10 = ar_[2], p11 = ar_[3], p20 = ar_[4], p21 = ar_[5], p30 = ar_[6], p31 = ar_[7];
                        v16 s00 = ai_[0], s01 = ai_[1], s10 = ai_[2], s11 = ai_[3], s20 = ai_[4], s21 = ai_[5], s30 = ai_[6], s31 = ai_[7];
                        for (int n = 0; n < NC; ++n) {
                            const v16 br0 = Br[(size_t)n * 2], br1 = Br[(size_t)n * 2 + 1];
                            const v16 bi0 = Bi[(size_t)n * 2], bi1 = Bi[(size_t)n * 2 + 1];
                            const float* arow = Arf + (size_t)n * T + i0;
                            const float* airow = Aif + (size_t)n * T + i0;
                            v16 xr, xi;
                            xr = bcast(arow[0]); xi = bcast(airow[0]);
                            p00 += xr * br0; p00 -= xi * bi0; s00 += xr * bi0; s00 += xi * br0; p01 += xr * br1; p01 -= xi * bi1; s01 += xr * bi1; s01 += xi * br1;
                            xr = bcast(arow[1]); xi = bcast(airow[1]);
                            p10 += xr * br0; p10 -= xi * bi0; s10 += xr * bi0; s10 += xi * br0; p11 += xr * br1; p11 -= xi * bi1; s11 += xr * bi1; s11 += xi * br1;
                            xr = bcast(arow[2]); xi = bcast(airow[2]);
                            p20 += xr * br0; p20 -= xi * bi0; s20 += xr * bi0; s20 += xi * br0; p21 += xr * br1; p21 -= xi * bi1; s21 += xr * bi1; s21 += xi * br1;
                            xr = bcast(arow[3]); xi = bcast(airow[3]);
                            p30 += xr * br0; p30 -= xi * bi0; s30 += xr * bi0; s30 += xi * br0; p31 += xr * br1; p31 -= xi * bi1; s31 += xr * bi1; s31 += xi * br1;
                        }
                        ar_[0] = p00; ar_[1] = p01; ar_[2] = p10; ar_[3] = p11; ar_[4] = p20; ar_[5] = p21; ar_[6] = p30; ar_[7] = p31;
                        ai_[0] = s00; ai_[1] = s01; ai_[2] = s10; ai_[3] = s11; ai_[4] = s20; ai_[5] = s21; ai_[6] = s30; ai_[7] = s31;
                    } else {
                        v16 p00 = ar_[0], p10 = ar_[1], p20 = ar_[2], p30 = ar_[3];
                        v16 s00 = ai_[0], s10 = ai_[1], s20 = ai_[2], s30 = ai_[3];
                        for (int n = 0; n < NC; ++n) {
                            const v16 br0 = Br[n], bi0 = Bi[n];
                            const float* arow = Arf + (size_t)n * T + i0;
                            const float* airow = Aif + (size_t)n * T + i0;
                            v16 xr, xi;
                            xr = bcast(arow[0]); xi = bcast(airow[0]); p00 += xr * br0; p00 -= xi * bi0; s00 += xr * bi0; s00 += xi * br0;
                            xr = bcast(arow[1]); xi = bcast(airow[1]); p10 += xr * br0; p10 -= xi * bi0; s10 += xr * bi0; s10 += xi * br0;
                            xr = bcast(arow[2]); xi = bcast(airow[2]); p20 += xr * br0; p20 -= xi * bi0; s20 += xr * bi0; s20 += xi * br0;
                            xr = bcast(arow[3]); xi = bcast(airow[3]); p30 += xr * br0; p30 -= xi * bi0; s30 += xr * bi0; s30 += xi * br0;
                        }
                        ar_[0] = p00; ar_[1] = p10; ar_[2] = p20; ar_[3] = p30;
                        ai_[0] = s00; ai_[1] = s10; ai_[2] = s20; ai_[3] = s30;
                    }
                }
            }
            // quadratic phase correction and store
            float* outr = ore + (size_t)b * T * T;
            float* outi = oim + (size_t)b * T * T;
            const float* accrf = reinterpret_cast<const float*>(accr.data());
            const float* accif = reinterpret_cast<const float*>(acci.data());
            const double cx = ucx[b], cy = ucy[b], f = fc2 / (2.0 * rc[b]);
            for (int i = 0; i < T; ++i) {
                const double dx = dlx[i];
                for (int j = 0; j < T; ++j) {
                    const double dy = dly[j];
                    const double los = cx * dx + cy * dy;
                    const double qc = (dx * dx + dy * dy - los * los) * f;
                    float s, c;
                    sincos_cyc(qc, s, c);
                    const float vr = accrf[(size_t)i * T + j], vi = accif[(size_t)i * T + j];
                    outr[i * T + j] = vr * c - vi * s;
                    outi[i * T + j] = vr * s + vi * c;
                }
            }
        }
    }
}

int ffbp_cpu_threads(void) { return omp_get_max_threads(); }

}  // extern "C"
