// Parallel loops for the CPU kernels: OpenMP where the kernels are compiled with it (Linux), std::thread elsewhere
// (macOS, where the OpenMP runtime of a wheel such as finufft's and a second one loaded by this library deadlock).
// The simd pragmas of the kernels stay OpenMP's: without the runtime they are enabled by -fopenmp-simd.
// With FASTSAR_PAR_POOL defined and no OpenMP (the prebuilt kernels of the Linux wheels, which must load on a
// machine without an OpenMP runtime) the std::thread loops run on threads that are kept between calls, as an
// OpenMP runtime keeps its team: the kernels' thread_local scratch survives, and no thread is created per call.
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
#if defined(FASTSAR_PAR_POOL) && defined(__linux__)
#include <sched.h>
#endif
namespace par {
inline int max_threads()
{
    if (const char* s = std::getenv("OMP_NUM_THREADS")) {
        const int n = std::atoi(s);
        if (n > 0) return n;
    }
#if defined(FASTSAR_PAR_POOL) && defined(__linux__)
    cpu_set_t set;                                // the CPUs this process may run on (taskset, a cpuset), as OpenMP
    if (sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0) return CPU_COUNT(&set);       // counts
#endif
    const unsigned h = std::thread::hardware_concurrency();
    return h ? (int)h : 1;
}
#ifdef FASTSAR_PAR_POOL
}  // namespace par
#include <condition_variable>
#include <mutex>
#include <pthread.h>
namespace par {
// The kept threads of this shared object. One loop at a time runs on them; a loop that starts while they are busy
// (a second Python thread in a kernel, a loop inside a loop) creates its own threads as the plain path does. A
// thread spins for a fraction of a millisecond after a loop, so that loops in quick succession start without a
// system call, then sleeps on a condition variable: an idle pool costs nothing. After a fork the child has none
// of the threads: it starts a new pool (the old one is abandoned, never destroyed, as at exit).
#if defined(__x86_64__)
inline void cpu_relax() { __builtin_ia32_pause(); }
#elif defined(__aarch64__)
inline void cpu_relax() { __asm__ __volatile__("yield"); }
#else
inline void cpu_relax() {}
#endif
struct Pool {
    enum { SPIN = 4000 };
    std::mutex m, busy;
    std::condition_variable wake, done;
    std::vector<std::thread> ts;
    std::atomic<unsigned long> state{0};          // loop number << 16 | kept threads the loop wants: read as one
    std::atomic<int> pending{0}, sleepers{0};
    void (*job)(void*) = nullptr;                 // written before state, stable until pending is 0
    void* arg = nullptr;

    void worker(unsigned long id)
    {
        unsigned long seen = 0;
        for (;;) {
            unsigned long st = state.load();
            for (int s = 0; st == seen && s < SPIN; ++s) { cpu_relax(); st = state.load(); }
            if (st == seen) {
                std::unique_lock<std::mutex> lk(m);
                ++sleepers;
                wake.wait(lk, [&] { return state.load() != seen; });
                --sleepers;
                st = state.load();
            }
            seen = st;
            if (id >= (st & 0xffff)) continue;    // a loop of fewer threads: not this one's
            job(arg);
            if (--pending == 0) {
                { std::lock_guard<std::mutex> lk(m); }
                done.notify_all();
            }
        }
    }
    // job(arg) on nt kept threads; returns when all have returned. The caller only waits: its own stack holds the
    // loop's captured variables, which every thread reads, and running the loop body on it too puts that body's
    // scratch writes on the same cache lines (measured: a third of the speed on 8 threads)
    void run(int nt, void (*j)(void*), void* a)
    {
        nt = std::min(nt, 0xffff);
        while ((int)ts.size() < nt) {
            const unsigned long id = ts.size();
            ts.emplace_back([this, id] { worker(id); });
        }
        job = j; arg = a;
        pending.store(nt);
        state.store(((state.load() >> 16) + 1) << 16 | (unsigned long)nt);
        if (sleepers.load() > 0) {
            { std::lock_guard<std::mutex> lk(m); }
            wake.notify_all();
        }
        for (int s = 0; pending.load() != 0 && s < SPIN; ++s) cpu_relax();
        if (pending.load() != 0) {
            std::unique_lock<std::mutex> lk(m);
            done.wait(lk, [&] { return pending.load() == 0; });
        }
    }
};
inline Pool*& pool_slot() { static Pool* p = nullptr; return p; }
inline Pool* pool()
{
    static std::once_flag once;
    std::call_once(once, [] { pthread_atfork(nullptr, nullptr, [] { pool_slot() = new Pool; }); pool_slot() = new Pool; });
    return pool_slot();
}
}  // namespace par
namespace par {
#endif
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
#ifdef FASTSAR_PAR_POOL
    {
        Pool* p = pool();
        std::unique_lock<std::mutex> one(p->busy, std::try_to_lock);
        if (one.owns_lock()) {
            p->run(nt, [](void* w) { (*static_cast<decltype(work)*>(w))(); }, &work);
            return;
        }
    }
#endif
    std::vector<std::thread> ts;
    ts.reserve(nt - 1);
    for (int t = 1; t < nt; ++t) ts.emplace_back(work);
    work();
    for (auto& t : ts) t.join();
}
}  // namespace par
#endif
