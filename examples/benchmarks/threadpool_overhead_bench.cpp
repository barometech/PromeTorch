// Measure raw ThreadPool::parallel_for overhead.
// Calls parallel_for repeatedly with near-empty bodies and reports
// microseconds per call. If overhead × calls-per-token is significant,
// we know pool-sync is a real bottleneck.
//
// Run: ./bench_threadpool_overhead <iters> <threads_hint>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>

#include "c10/util/ThreadPool.h"

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 10000;

    auto& pool = c10::get_thread_pool();
    std::printf("ThreadPool threads = %d\n", pool.num_threads());

    // Warmup
    for (int i = 0; i < 100; ++i) {
        pool.parallel_for(0, 1024, [](int64_t, int64_t){}, 1);
    }

    // Micro-empty: min_grain=1, N=pool.num_threads() → splits evenly
    std::atomic<int64_t> dummy{0};
    auto t0 = std::chrono::high_resolution_clock::now();
    int64_t N = pool.num_threads() * 2048;  // ensures all threads get work
    for (int i = 0; i < iters; ++i) {
        pool.parallel_for(0, N, [&](int64_t, int64_t){
            dummy.fetch_add(1, std::memory_order_relaxed);
        }, 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_total = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("parallel_for(noop body, %d-thread fanout): %d calls in %.0f us = %.2f us/call\n",
                pool.num_threads(), iters, us_total, us_total / iters);

    // Also measure parallel_for with a small per-chunk body doing a sum
    auto t2 = std::chrono::high_resolution_clock::now();
    std::atomic<int64_t> sum{0};
    for (int i = 0; i < iters; ++i) {
        pool.parallel_for(0, N, [&](int64_t s, int64_t e){
            int64_t local = 0;
            for (int64_t k = s; k < e; ++k) local += k;
            sum.fetch_add(local, std::memory_order_relaxed);
        }, 1);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double us_total2 = std::chrono::duration<double, std::micro>(t3 - t2).count();
    std::printf("parallel_for(small-sum body): %d calls in %.0f us = %.2f us/call\n",
                iters, us_total2, us_total2 / iters);

    // Per-token estimate: qwen3:4b forward has ~180 parallel_for calls/token.
    // (5 GEMV/layer × 36 layers = 180)
    double per_token_overhead_ms = (us_total / iters) * 180.0 / 1000.0;
    std::printf("Estimated pure pool-sync overhead per token (180 calls): %.2f ms\n",
                per_token_overhead_ms);
    std::printf("At 3.8 tok/s = 263 ms/token, pool overhead = %.1f%%\n",
                per_token_overhead_ms / 263.0 * 100.0);

    return 0;
}
