#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// ============================================================================
// ThreadPool.h — Persistent thread pool replacing OpenMP fork/join
// ============================================================================
// On Elbrus E2K, OpenMP #pragma omp parallel for creates/destroys threads
// on EACH parallel region (~10ms per fork/join). For MNIST with 937 batches
// x ~10 regions = 9,370 fork/joins = 93 seconds of pure overhead.
//
// This pool creates threads ONCE, they wait on a condition variable for work.
// Compatible with LCC (pthreads) and MSVC (std::thread).
//
// Usage in hot_loops.cpp:
//   c10::parallel_for_1d(n, [&](int64_t start, int64_t end) {
//       for (int64_t i = start; i < end; ++i) out[i] = ...;
//   });
// ============================================================================

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#if defined(__linux__) && !defined(_WIN32)
#  define _GNU_SOURCE
#  include <sched.h>
#  include <pthread.h>
#  if __has_include(<numa.h>)
#    include <numa.h>
#    define C10_HAS_LIBNUMA 1
#  else
#    define C10_HAS_LIBNUMA 0
#  endif
#else
#  define C10_HAS_LIBNUMA 0
#endif

namespace c10 {

// Per-thread NUMA node id. Written once on worker start (or on demand for
// the main thread). Read by NUMA-aware code paths (e.g. quant GEMV with
// per-node weight replicas) to pick the correct local copy.
inline thread_local int t_numa_node = -1;

// Query or lazily-initialize current thread's NUMA node. Returns 0 if the
// system has no NUMA API. Safe to call from any thread, any time.
inline int current_numa_node() {
    if (t_numa_node >= 0) return t_numa_node;
#if defined(__linux__) && !defined(_WIN32)
    int cpu = sched_getcpu();
    if (cpu < 0) { t_numa_node = 0; return 0; }
#  if C10_HAS_LIBNUMA
    if (numa_available() >= 0) {
        int n = numa_node_of_cpu(cpu);
        t_numa_node = (n < 0) ? 0 : n;
        return t_numa_node;
    }
#  endif
    // Fallback without libnuma: assume homogeneous layout where cores are
    // packed per node. Env PT_CORES_PER_NODE overrides default 8 (Elbrus
    // E8C2). Good-enough heuristic for single-socket NUMA boxes.
    int cores_per = 8;
    const char* env = std::getenv("PT_CORES_PER_NODE");
    if (env && env[0]) { int v = std::atoi(env); if (v > 0) cores_per = v; }
    t_numa_node = cpu / cores_per;
    return t_numa_node;
#else
    t_numa_node = 0;
    return 0;
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0) {
        if (num_threads <= 0) {
            // Allow env overrides for multi-process TP on NUMA systems.
            // PT_NUM_THREADS is ours; fall back to OMP_NUM_THREADS for
            // consistency with the rest of the stack, then hardware_concurrency.
            const char* env = std::getenv("PT_NUM_THREADS");
            if (!env || !env[0]) env = std::getenv("OMP_NUM_THREADS");
            if (env && env[0]) {
                int n = std::atoi(env);
                if (n > 0) num_threads = n;
            }
            if (num_threads <= 0) {
                num_threads = static_cast<int>(std::thread::hardware_concurrency());
            }
            if (num_threads <= 0) num_threads = 4;
        }
        num_threads_ = num_threads;
        // If PT_PIN_THREADS=1 we spread worker i across cores round-robin,
        // one core per worker. This is what OMP_PLACES=cores OMP_PROC_BIND=close
        // does for OpenMP pools, but our std::thread pool doesn't inherit that.
        pin_enabled_ = false;
        {
            const char* env = std::getenv("PT_PIN_THREADS");
            if (env && env[0] == '1') pin_enabled_ = true;
        }
        for (int i = 0; i < num_threads; i++) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // ========================================================================
    // parallel_for: splits [begin, end) across workers, blocks until done
    // ========================================================================
    // fn(chunk_start, chunk_end) is called once per worker with its range.
    // min_grain: minimum elements per chunk (avoids overhead for small ops).
    // If total < min_grain * num_threads, runs serially.
    // ========================================================================

    void parallel_for(int64_t begin, int64_t end,
                      const std::function<void(int64_t, int64_t)>& fn,
                      int64_t min_grain = 1024) {
        int64_t total = end - begin;
        if (total <= 0) return;

        // Serial for small work or single-thread pool
        if (total < min_grain * num_threads_ || num_threads_ <= 1) {
            fn(begin, end);
            return;
        }

        int64_t chunk_size = (total + num_threads_ - 1) / num_threads_;
        // Round chunk_size up to a multiple of 16 so thread boundaries never
        // cut through a cacheline (16 × fp32 = 64 B). Otherwise adjacent
        // threads ping-pong the shared cacheline at the boundary, and for
        // decode-sized outputs (N=2560 / T=24 ≈ 107 chunks, 23 mis-aligned
        // boundaries × ~217 parallel_fors / token ≈ ~5k invalidations per
        // token). Only matters above the min_grain threshold where we
        // actually parallelize, so it doesn't regress tiny ops.
        // (agent_4_threadpool_audit.md Q2 / P2)
        chunk_size = ((chunk_size + 15) / 16) * 16;
        int actual_chunks = 0;

        for (int i = 0; i < num_threads_; i++) {
            int64_t start = begin + i * chunk_size;
            if (start >= end) break;
            actual_chunks++;
        }

        if (actual_chunks <= 1) {
            fn(begin, end);
            return;
        }

        std::atomic<int> tasks_done{0};

        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int i = 0; i < actual_chunks; i++) {
                int64_t start = begin + i * chunk_size;
                int64_t chunk_end = (std::min)(start + chunk_size, end);
                tasks_.push([&fn, start, chunk_end, &tasks_done, this] {
                    fn(start, chunk_end);
                    tasks_done.fetch_add(1, std::memory_order_release);
                    // FIX 3.3: wake main thread via condition_variable
                    { std::lock_guard<std::mutex> lk(done_mutex_); }
                    done_cv_.notify_one();
                });
            }
        }
        cv_.notify_all();

        // FIX 3.3: sleep instead of burn CPU (was yield() spinlock)
        {
            std::unique_lock<std::mutex> lk(done_mutex_);
            done_cv_.wait(lk, [&] { return tasks_done.load(std::memory_order_acquire) >= actual_chunks; });
        }
    }

    // Legacy overload: parallel_for(n, fn) == parallel_for(0, n, fn)
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
        parallel_for(0, n, fn);
    }

    int num_threads() const { return num_threads_; }

private:
    void worker_loop(int worker_id) {
#if defined(__linux__) && !defined(_WIN32)
        // Optional affinity pin: worker i → core i (round-robin over available).
        // With NUMA-aware pinning, workers pin to specific NUMA nodes so the
        // per-thread t_numa_node cache stays stable, and the NUMA replica
        // fetch in GEMV returns the same local pointer every iteration.
        if (pin_enabled_) {
            int ncpu = static_cast<int>(std::thread::hardware_concurrency());
            if (ncpu <= 0) ncpu = 1;

            // Stripe workers across NUMA nodes instead of packing them onto
            // the first N cores. With 24 workers and 32 cores (4 nodes × 8
            // cores), naive `worker_id % ncpu` gives nodes 0-2 eight workers
            // each and leaves node 3 empty — its DDR controller is idle while
            // nodes 0-2 contend for cross-chip traffic to read replicas.
            // Striped layout (agent_3_numa_audit.md rank 1):
            //   worker 0 → node 0 core 0
            //   worker 1 → node 1 core 0
            //   worker 2 → node 2 core 0
            //   worker 3 → node 3 core 0
            //   worker 4 → node 0 core 1
            //   ...
            int nodes = 1;
            int cores_per_node = ncpu;
#  if C10_HAS_LIBNUMA
            if (numa_available() >= 0) {
                int n = numa_num_configured_nodes();
                if (n > 0) {
                    nodes = n;
                    cores_per_node = (ncpu + n - 1) / n;
                }
            }
#  endif
            if (nodes <= 1) {
                const char* env = std::getenv("PT_CORES_PER_NODE");
                if (env && env[0]) {
                    int v = std::atoi(env);
                    if (v > 0) {
                        cores_per_node = v;
                        nodes = (ncpu + v - 1) / v;
                    }
                }
            }

            int node = worker_id % nodes;
            int core_on_node = (worker_id / nodes) % cores_per_node;
            int cpu = node * cores_per_node + core_on_node;
            if (cpu >= ncpu) cpu = worker_id % ncpu;  // fallback safety

            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(cpu, &cs);
            pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
        }
#endif
        // Prime thread-local NUMA node cache so first parallel_for doesn't
        // pay sched_getcpu+numa_node_of_cpu latency on the critical path.
        (void)current_numa_node();
        (void)worker_id;

        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::mutex done_mutex_;              // FIX 3.3: for parallel_for completion
    std::condition_variable done_cv_;    // FIX 3.3: wake main thread
    bool stop_ = false;
    int num_threads_;
    bool pin_enabled_ = false;  // PT_PIN_THREADS=1 → 1 worker per core
};

// Global thread pool singleton — created once, lives forever
inline ThreadPool& get_thread_pool() {
    static ThreadPool pool;
    return pool;
}

// Convenience: parallel_for(n, fn) using global pool
inline void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    get_thread_pool().parallel_for(n, fn);
}

// ============================================================================
// parallel_for_1d: threshold-gated parallel loop
// ============================================================================
// Only parallelizes when n > threshold. For small ops, runs inline.
// Use this in hot_loops.cpp to replace #pragma omp parallel for.
//
// Example:
//   c10::parallel_for_1d(n, [&](int64_t s, int64_t e) {
//       for (int64_t i = s; i < e; ++i) out[i] = in[i] * 2.0f;
//   }, 65536);
// ============================================================================

inline void parallel_for_1d(int64_t n,
                            const std::function<void(int64_t, int64_t)>& fn,
                            int64_t threshold = 65536) {
    if (n <= threshold) {
        fn(0, n);
    } else {
        get_thread_pool().parallel_for(0, n, fn);
    }
}

} // namespace c10
