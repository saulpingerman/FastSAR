// Parallel loops for the CPU kernels: OpenMP where the kernels are compiled with it (Linux), std::thread elsewhere
// (macOS, where the OpenMP runtime of a wheel such as finufft's and a second one loaded by this library deadlock).
// The simd pragmas of the kernels stay OpenMP's: without the runtime they are enabled by -fopenmp-simd.
//
//   par::max_threads()                   threads a loop uses (OMP_NUM_THREADS, else the hardware's)
//   par::for_dynamic(n, chunk, f)        f(i) for i in [0, n), chunk consecutive indices per grab, dynamic
//                                        scheduling; f is called on the worker threads (thread_local scratch works)
#pragma once
#include <algorithm>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
namespace par {
inline int max_threads() { return omp_get_max_threads(); }
template <class F> void for_dynamic(long n, int chunk, F&& f)
{
    const long nch = (n + chunk - 1) / chunk;
    #pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nch; ++c) {
        const long i1 = std::min(n, (c + 1) * (long)chunk);
        for (long i = c * chunk; i < i1; ++i) f(i);
    }
}
}  // namespace par
#else
#include <atomic>
#include <thread>
#include <vector>
namespace par {
inline int max_threads()
{
    if (const char* s = std::getenv("OMP_NUM_THREADS")) {
        const int n = std::atoi(s);
        if (n > 0) return n;
    }
    const unsigned h = std::thread::hardware_concurrency();
    return h ? (int)h : 1;
}
template <class F> void for_dynamic(long n, int chunk, F&& f)
{
    const long nch = (n + chunk - 1) / chunk;
    const int nt = (int)std::min<long>(max_threads(), nch);
    std::atomic<long> next(0);
    auto work = [&]() {
        for (;;) {
            const long c = next.fetch_add(1);
            if (c >= nch) break;
            const long i1 = std::min(n, (c + 1) * (long)chunk);
            for (long i = c * chunk; i < i1; ++i) f(i);
        }
    };
    if (nt <= 1) { work(); return; }
    std::vector<std::thread> ts;
    ts.reserve(nt - 1);
    for (int t = 1; t < nt; ++t) ts.emplace_back(work);
    work();
    for (auto& t : ts) t.join();
}
}  // namespace par
#endif
