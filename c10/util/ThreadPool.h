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

namespace c10 {

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
        for (int i = 0; i < num_threads; i++) {
            workers_.emplace_back(&ThreadPool::worker_loop, this);
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
    void worker_loop() {
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
