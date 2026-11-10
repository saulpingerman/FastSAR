// Check of par.hpp's kept threads (FASTSAR_PAR_POOL, the loops of the prebuilt kernels): every index of a loop runs
// exactly once, for loops in quick succession, loops inside loops, loops started from several threads at once, and
// in a child process after a fork. tests/test_kernels.py builds and runs it; for the data-race check build it with
//   g++ -O1 -g -fsanitize=thread -std=c++17 -pthread -DFASTSAR_PAR_POOL -Ifastsar tests/par_pool.cpp -o par_pool_tsan
// and run `setarch -R ./par_pool_tsan nofork` (ThreadSanitizer does not follow a fork of a threaded process).
#include "par.hpp"
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

static long run_once(long n, int chunk)
{
    std::vector<std::atomic<int>> hit(n);
    for (auto& h : hit) h = 0;
    par::for_dynamic(n, chunk, [&](long i) {
        hit[i]++;
        if (i % 7 == 0) par::for_dynamic(5, 1, [&](long) {});
    });
    long bad = 0;
    for (auto& h : hit) bad += h != 1;
    return bad;
}

int main(int argc, char** argv)
{
    long bad = 0;
    for (int it = 0; it < 1500; ++it) bad += run_once(1 + it % 97, 1 + it % 5);
    std::printf("threads %d\nserial %ld\n", par::max_threads(), bad);
    std::atomic<long> cb(0);
    std::vector<std::thread> callers;
    for (int c = 0; c < 6; ++c)
        callers.emplace_back([&] { for (int it = 0; it < 60; ++it) cb += run_once(1 + it % 50, 2); });
    for (auto& t : callers) t.join();
    std::printf("concurrent %ld\n", cb.load());
    bad += cb.load();
    if (argc > 1 && !std::strcmp(argv[1], "nofork")) return bad != 0;
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        long b = 0;
        for (int it = 0; it < 200; ++it) b += run_once(64, 2);
        _exit(b != 0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    for (int it = 0; it < 200; ++it) bad += run_once(64, 2);
    std::printf("child %d\nparent %ld\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1, bad);
    return bad != 0 || status != 0;
}
