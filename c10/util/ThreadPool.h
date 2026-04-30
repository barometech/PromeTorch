#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// ============================================================================
// ThreadPool.h — Persistent broadcast-dispatch pool (Round 4 rev)
// ============================================================================
// Замена mutex+queue+CV паттерна (10ms fork/join overhead на каждый
// parallel_for) на:
//   - один shared task descriptor (master-written, workers-read)
//   - atomic gen counter с futex wake/wait
//   - per-worker ack slots (cacheline-padded, false-sharing-free)
//   - master также участвует в drain (extra worker для small jobs)
//
// Ключевые инварианты (предотвращение Round 3 deadlock):
//   I1. Master НЕ публикует новый descriptor пока все воркеры не ack'нули
//       предыдущий gen — `slot.ack == prev_gen` для всех слотов.
//   I2. Воркер всегда перечитывает `state_.n_chunks` ПОСЛЕ observed gen
//       change, не кеширует между gen.
//   I3. `next_chunk` reset master'ом ТОЛЬКО после всех ack — workers
//       не могут случайно claim chunks от prev gen.
//   I4. `tp_in_parallel` thread_local флаг блокирует nested parallel_for
//       (превращает его в serial), предотвращая deadlock self-recursion.
//   I5. Watchdog timer — если PT_TP_TIMEOUT_MS установлен, master timeout
//       на done_count → assert (для отладки, не production).
//
// Drop-in replacement для предыдущего ThreadPool.h: API public идентичный.
// ============================================================================

#include <thread>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "Futex.h"

#if defined(__linux__) && !defined(_WIN32)
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
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

// Per-thread NUMA node id. Cached; populated lazily on first call.
inline thread_local int t_numa_node = -1;

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

// I4 reentrancy guard. thread_local — каждый воркер свой, master свой.
inline thread_local bool t_in_parallel = false;

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0) {
        if (num_threads <= 0) {
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
        num_workers_ = num_threads;
        slots_ = std::unique_ptr<Slot[]>(new Slot[num_workers_]);
        for (int i = 0; i < num_workers_; ++i) {
            slots_[i].ack.store(0, std::memory_order_relaxed);
        }

        // Watchdog timeout (debug). 0 = disabled.
        const char* wd = std::getenv("PT_TP_TIMEOUT_MS");
        watchdog_ms_ = (wd && wd[0]) ? static_cast<uint32_t>(std::atoi(wd)) : 0;

        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }

    ~ThreadPool() {
        // Сигнал shutdown: stop=true, бамп gen, wake all workers
        stop_.store(1, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        futex_wake_all(&gen_);
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int num_threads() const { return num_workers_; }

    // Главный API. Семантика идентична предыдущему ThreadPool::parallel_for.
    void parallel_for(int64_t begin, int64_t end,
                      const std::function<void(int64_t, int64_t)>& fn,
                      int64_t min_grain = 1024) {
        const int64_t total = end - begin;
        if (total <= 0) return;

        // I4: nested parallel_for внутри текущего pool'а — serial.
        if (t_in_parallel) { fn(begin, end); return; }

        // Маленькая работа — serial без bouncing на воркеров.
        if (total < min_grain * num_workers_ || num_workers_ <= 1) {
            fn(begin, end);
            return;
        }

        // Round chunk_size до 16 (cache line × 4) для предотвращения
        // false sharing на boundaries.
        int64_t chunk_size = (total + num_workers_ - 1) / num_workers_;
        chunk_size = ((chunk_size + 15) / 16) * 16;

        // Реальное число chunks — сколько workers получит работу.
        int n_chunks = 0;
        for (int i = 0; i < num_workers_; ++i) {
            int64_t s = begin + (int64_t)i * chunk_size;
            if (s >= end) break;
            n_chunks++;
        }
        if (n_chunks <= 1) { fn(begin, end); return; }

        // I1: ждём пока все воркеры ack'нули предыдущий gen.
        const uint32_t prev_gen = gen_.load(std::memory_order_relaxed);
        for (int i = 0; i < num_workers_; ++i) {
            // Простой spin — обычно занят 0 циклов в steady state
            for (int spin = 0; spin < 256; ++spin) {
                if (slots_[i].ack.load(std::memory_order_acquire) == prev_gen) goto next_slot;
            }
            // Slot отстаёт — ждём
            while (slots_[i].ack.load(std::memory_order_acquire) != prev_gen) {
                std::this_thread::yield();
            }
        next_slot:;
        }

        // I3: сейчас ВСЕ воркеры sleeping на gen_, безопасно публиковать
        // descriptor.
        cur_fn_     = &fn;
        cur_begin_  = begin;
        cur_end_    = end;
        cur_chunk_  = chunk_size;
        cur_n_chunks_.store(n_chunks, std::memory_order_relaxed);
        next_chunk_.store(0, std::memory_order_relaxed);
        done_count_.store(0, std::memory_order_relaxed);

        // Бампим gen и будим всех — release-store пары с acquire-load
        // в worker_loop'е.
        const uint32_t new_gen = prev_gen + 1;
        gen_.store(new_gen, std::memory_order_release);
        futex_wake_all(&gen_);

        // Master тоже работает (не ждёт впустую).
        t_in_parallel = true;
        for (;;) {
            const uint32_t k = next_chunk_.fetch_add(1, std::memory_order_acq_rel);
            if (static_cast<int>(k) >= n_chunks) break;
            const int64_t s = begin + (int64_t)k * chunk_size;
            const int64_t e = std::min(s + chunk_size, end);
            fn(s, e);
            const uint32_t d = done_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (d == static_cast<uint32_t>(n_chunks)) {
                futex_wake_all(&done_count_);
            }
        }
        t_in_parallel = false;

        // Ждём пока все chunks dispatched. Spin → futex_wait.
        for (int spin = 0; spin < 1024; ++spin) {
            if (done_count_.load(std::memory_order_acquire)
                >= static_cast<uint32_t>(n_chunks)) goto done_wait;
        }
        for (;;) {
            const uint32_t cur = done_count_.load(std::memory_order_acquire);
            if (cur >= static_cast<uint32_t>(n_chunks)) break;
            if (watchdog_ms_) {
                bool ok = futex_wait_timed(&done_count_, cur, watchdog_ms_);
                if (!ok && done_count_.load() < (uint32_t)n_chunks) {
                    // Watchdog fire — debug assert
                    std::fprintf(stderr,
                        "[ThreadPool] watchdog fire: gen=%u, done=%u/%d\n",
                        new_gen, done_count_.load(), n_chunks);
                    // Не abort — просто продолжаем ждать
                }
            } else {
                futex_wait(&done_count_, cur);
            }
        }
    done_wait:;
        // Note: воркеры теперь ack'нут new_gen в своих slot'ах перед тем
        // как уснуть на следующем gen_. Master НЕ ждёт ack здесь —
        // следующий parallel_for сделает это в I1 wait выше.
    }

    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
        parallel_for(0, n, fn);
    }

private:
    struct alignas(64) Slot {
        std::atomic<uint32_t> ack{0};
        uint32_t pad[15];  // pad до 64 байт
    };

    void worker_loop(int worker_id) {
#if defined(__linux__) && !defined(_WIN32)
        // Pin доступен через PT_PIN_THREADS=1, но для TP-режима
        // (numactl --cpunodebind) НЕ устанавливается — иначе воркеры
        // ranks 1-3 пытаются pin'нуться вне их cpuset → kernel клампит
        // их на одно ядро (см. scripts/run_tp_elbrus.sh guard).
        const char* pin = std::getenv("PT_PIN_THREADS");
        if (pin && pin[0] == '1') {
            int ncpu = static_cast<int>(std::thread::hardware_concurrency());
            if (ncpu <= 0) ncpu = 1;
            int cpu = worker_id % ncpu;
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
            pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
        }
#endif
        (void)current_numa_node();  // прогрев t_numa_node
        (void)worker_id;

        uint32_t observed_gen = 0;

        for (;;) {
            // Ждём изменения gen
            for (;;) {
                if (stop_.load(std::memory_order_acquire)) return;
                const uint32_t g = gen_.load(std::memory_order_acquire);
                if (g != observed_gen) { observed_gen = g; break; }
                // Sleep на futex до wake
                futex_wait(&gen_, observed_gen);
            }

            if (stop_.load(std::memory_order_acquire)) return;

            // I2: перечитываем descriptor КАЖДЫЙ wake — не кешируем.
            const int n_chunks = cur_n_chunks_.load(std::memory_order_acquire);

            // Drain chunks. master также drain'ит — race на next_chunk_
            // отрабатывается atomic fetch_add.
            t_in_parallel = true;
            for (;;) {
                const uint32_t k = next_chunk_.fetch_add(1, std::memory_order_acq_rel);
                if (static_cast<int>(k) >= n_chunks) break;
                const int64_t s = cur_begin_ + (int64_t)k * cur_chunk_;
                const int64_t e = std::min(s + cur_chunk_, cur_end_);
                (*cur_fn_)(s, e);
                const uint32_t d = done_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
                if (d == static_cast<uint32_t>(n_chunks)) {
                    futex_wake_all(&done_count_);
                }
            }
            t_in_parallel = false;

            // Ack текущий gen → master разрешит следующий dispatch
            slots_[worker_id].ack.store(observed_gen, std::memory_order_release);
        }
    }

    int num_workers_ = 0;
    uint32_t watchdog_ms_ = 0;

    // Workers
    std::vector<std::thread>  workers_;
    std::unique_ptr<Slot[]>   slots_;

    // Synchronization
    std::atomic<uint32_t> gen_{0};
    std::atomic<uint32_t> next_chunk_{0};
    std::atomic<uint32_t> done_count_{0};
    std::atomic<int>      cur_n_chunks_{0};
    std::atomic<uint32_t> stop_{0};

    // Descriptor (master-written under wait_workers_idle protection)
    const std::function<void(int64_t,int64_t)>* cur_fn_ = nullptr;
    int64_t cur_begin_ = 0;
    int64_t cur_end_   = 0;
    int64_t cur_chunk_ = 0;
};

inline ThreadPool& get_thread_pool() {
    static ThreadPool pool;
    return pool;
}

inline void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    get_thread_pool().parallel_for(n, fn);
}

inline void parallel_for_1d(int64_t n,
                            const std::function<void(int64_t, int64_t)>& fn,
                            int64_t threshold = 65536) {
    if (n <= threshold) {
        fn(0, n);
    } else {
        get_thread_pool().parallel_for(0, n, fn);
    }
}

}  // namespace c10
