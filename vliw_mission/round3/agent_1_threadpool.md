# Agent 1 — ThreadPool fork/join overhead (Round 3)

## TL;DR

Replace `c10::ThreadPool::parallel_for`'s mutex+queue+cv pattern with a
**persistent broadcast dispatch** (single shared task descriptor + per-worker
futex slot). Target: ~100 µs → ~5 µs per call.
**Practical TP-4 qwen3:4b Q4_K_M ceiling: 211 → ~192 ms/tok ⇒ 4.8 → ~5.2 tok/s
(+8 %).** Honest. Sync was already measured at 6 % of total wallclock; this
attacks the remaining intra-rank fork/join, not inter-rank AllReduce.

## 1. Diagnosis of current implementation

`c10/util/ThreadPool.h:140-198` per `parallel_for` call does:

1. `std::lock_guard<std::mutex> lock(mutex_)` — acquire submit-mutex.
2. Push N task lambdas (heap-alloc via `std::function` type erasure) onto
   `std::queue<std::function<void()>>`.
3. `cv_.notify_all()` — Linux: `futex_wake(INT_MAX)` on the cv's internal
   futex.
4. Each woken worker: `cv_.wait(lock, …)` re-acquires `mutex_`, pops one task,
   releases mutex, runs lambda, on completion `done_cv_.notify_one()` plus
   `done_mutex_` lock/unlock.
5. Master: `done_cv_.wait(lk, …)` blocks on **second** futex.

Per call on E8C2: 1 submit-mutex roundtrip + up to N worker mutex roundtrips
(serialised on the same lock!) + 2 condvar futex syscalls + N
`std::function` heap allocs + N `done_cv_.notify_one()` futex wakes. Measured
~100 µs/call by Round 2 Agent #2.

For TP-4 with 7 workers per rank and ≈200 calls/token (5 per layer × 36
layers + tail): **~20 ms/token, ~10 % of 211 ms decode.**

## 2. Design: broadcast-dispatch pool (drop-in)

### Data structures

```c++
struct alignas(64) WorkerSlot {
    std::atomic<uint32_t> gen;     // per-worker observed generation
    char pad[64 - sizeof(uint32_t)];
};

struct alignas(64) PoolState {
    // Single task descriptor — broadcast, not queued
    int64_t        begin, end, chunk;
    void         (*trampoline)(void*, int64_t, int64_t);
    void*          user_ctx;          // points to caller-stack lambda
    uint32_t       n_chunks;
    std::atomic<uint32_t> gen;        // bumped once per parallel_for
    std::atomic<uint32_t> next_chunk; // dynamic chunk fetch (work-steal)
    std::atomic<uint32_t> done_count; // master waits on this
    char pad[64];
};
```

`PoolState` is shared, single-writer (master) for the descriptor,
multi-reader for workers. `WorkerSlot[N_workers]` is cacheline-padded
to kill false sharing. **No mutex, no queue, no `std::function`.**

### Submit path (master)

```c++
template <class F>
void parallel_for(int64_t begin, int64_t end, F&& fn, int64_t min_grain) {
    /* … existing serial fast-path unchanged … */
    state_.begin       = begin;
    state_.end         = end;
    state_.chunk       = chunk_size;            // already 16-aligned
    state_.user_ctx    = (void*)&fn;            // lambda lives on stack
    state_.trampoline  = &Trampoline<F>::call;  // template stamp
    state_.n_chunks    = actual_chunks;
    state_.next_chunk.store(0, std::memory_order_relaxed);
    state_.done_count.store(0, std::memory_order_relaxed);
    uint32_t g = state_.gen.fetch_add(1, std::memory_order_release) + 1;
    futex_wake_all(&state_.gen);                // ONE syscall, all workers
    // Master also helps (chunk 0)
    run_chunks_inline();
    // Wait for siblings
    spin_then_futex_wait(&state_.done_count, n_chunks, /*spin*/256);
}
```

`Trampoline<F>` is a `static void call(void* ctx, int64_t s, int64_t e)`
that casts and invokes — eliminates `std::function` heap allocs.

### Worker loop

```c++
void worker(int id) {
    WorkerSlot& slot = slots_[id];
    while (!stop_) {
        uint32_t cur_pool = state_.gen.load(std::memory_order_acquire);
        if (cur_pool == slot.gen.load(std::memory_order_relaxed)) {
            // Spin a few iters for low-latency wake, then futex
            for (int i = 0; i < 256; ++i) {
                if (state_.gen.load(std::memory_order_acquire) != cur_pool) goto fast;
                __builtin_ia32_pause();         // or `nop` on E2K
            }
            futex_wait(&state_.gen, cur_pool);
            continue;
        }
fast:
        slot.gen.store(state_.gen.load(std::memory_order_acquire),
                       std::memory_order_relaxed);
        // Dynamic chunk grab — handles imbalance for free
        for (;;) {
            uint32_t k = state_.next_chunk.fetch_add(1, std::memory_order_acq_rel);
            if (k >= state_.n_chunks) break;
            int64_t s = state_.begin + (int64_t)k * state_.chunk;
            int64_t e = std::min(s + state_.chunk, state_.end);
            state_.trampoline(state_.user_ctx, s, e);
            uint32_t d = state_.done_count.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (d == state_.n_chunks) futex_wake_one(&state_.done_count);
        }
    }
}
```

`futex_wait`/`futex_wake_all` reuse the helpers at
`torch/distributed/ddp.cpp:550-572` (no `FUTEX_PRIVATE_FLAG`, since we are
intra-process here we **can** add it — separate static helpers that pass
`FUTEX_WAIT|FUTEX_PRIVATE_FLAG`).

### Cost accounting

| Op                         | Cost         |
| -------------------------- | ------------ |
| Master writes descriptor   | ~30 ns       |
| One `futex_wake_all`       | ~1.5 µs      |
| Spin window catches workers in cache | <1 µs |
| Master ran chunk 0 inline  | (productive) |
| `futex_wait` on done_count | bypassed if last worker just stored ⇒ ~1 µs |

Estimate **3–6 µs** vs current ~100 µs ⇒ **~95 µs × 200 = 19 ms/tok saved**.

## 3. Diff sketch (file:line)

- `c10/util/ThreadPool.h`
  - Remove fields: `tasks_`, `mutex_`, `cv_`, `done_mutex_`, `done_cv_`
    (lines 255–259).
  - Add fields: `PoolState state_`, `std::vector<WorkerSlot> slots_`,
    `std::atomic<bool> stop_`.
  - Rewrite `parallel_for` (lines 140-198) per §2. Keep signature
    `void(int64_t,int64_t,const std::function<void(int64_t,int64_t)>&,int64_t)`
    for ABI; internally branch on a templated overload to skip
    `std::function` when caller uses a lambda directly. Easiest
    incremental: keep `std::function` but accept it by value into a
    descriptor stored in `state_.user_ctx`; trampoline does a single
    indirect call. Saves the queue + heap, even if not the indirection.
  - Rewrite `worker_loop` (lines 208-252) per §2.
  - `~ThreadPool`: set `stop_=true`; `state_.gen.fetch_add(1)`;
    `futex_wake_all(&state_.gen)`; join.

- Helpers (move out of `ddp.cpp` into `c10/util/Futex.h`): `futex_wait`,
  `futex_wake_all`, `futex_wake_one`. Add `FUTEX_PRIVATE_FLAG` variant.

- **No changes** to `cpu_quant_gemv.h` (17 callsites) or `gguf_model.h`
  (7 callsites). API stays `parallel_for(0, N, lambda, grain)`. Drop-in.

## 4. Risks

1. **Lost wakeup.** If master bumps `gen` and `futex_wake_all` fires
   *before* a worker reaches `futex_wait(&gen, cur)`: handled by the
   `cur != current_gen` check inside `futex_wait`'s kernel-side compare —
   returns `EAGAIN` immediately. Standard pattern; identical to ddp.cpp.
2. **Master-as-helper deadlock.** If master runs chunk 0 inline and
   chunk 0 calls `parallel_for` recursively (nested) → reentry breaks
   single-shared-descriptor. **Mitigation:** detect via
   `thread_local in_parallel = true` and run nested call serially. Our
   kernels do not nest.
3. **ABI compat.** `std::function` is kept in the API; only internals
   change. Existing 24 callsites compile unchanged.
4. **Grain rounding.** Existing 16-element alignment of `chunk_size`
   (line 161) preserved verbatim.
5. **Spin pollution on E2K.** 256-iter pause spin = ~0.3 µs at 1.5 GHz.
   Cheap; tunable via env `PT_TP_SPIN`.

## 5. Expected speedup (honest)

- 200 calls × ΔLatency.
  - Optimistic (5 µs): 19.0 ms saved → 211 → 192 ms/tok → **5.21 tok/s (+8.5 %)**.
  - Pessimistic (15 µs): 17.0 ms saved → 211 → 194 ms/tok → **5.15 tok/s (+7.3 %)**.
- This *cannot* lift us to 10 tok/s alone — agrees with Round 2's
  conclusion that sync = 6 % wall, GEMV body = bandwidth-bound. Combine
  with weight repack, fused kernels, or smaller model for compounding.

## 6. Test plan

1. **Microbench** `tools/bench_threadpool.cpp`: 10 000 × `parallel_for(0,
   8192, vadd, 256)` on 8 workers; report median µs/call. Pass: <8 µs
   on E8C2.
2. **Race stress**: 1 000 000 × `parallel_for(0, n, recurse_safe, g)`
   randomised n,g; checksum result. Run under `tsan` on x86 fallback.
3. **MNIST regression**: `train_mnist_mlp` 1 epoch on x86 — no accuracy
   delta vs current.
4. **End-to-end**: 3× clean restart, qwen3:4b Q4_K_M TP-4, 50 tokens.
   Report median tok/s and `PT_PROFILE_LAYER` breakdown. Pass: ≥5.10
   tok/s and `gate_up + ffn_down + attn_phase + output_proj` total
   drops by ≥17 ms.
5. **Shutdown ordering**: spawn/join 1000× in `main` to confirm no
   hung worker after `~ThreadPool`.

## 7. Files referenced

- `C:\Users\paper\Desktop\promethorch\c10\util\ThreadPool.h:140` — current
  fork/join implementation to replace.
- `C:\Users\paper\Desktop\promethorch\torch\distributed\ddp.cpp:550-572` —
  futex helpers to lift into `c10/util/Futex.h`.
- `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h` — 17
  callsites, unchanged.
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h:2551,2762,2870,
  3393,3442,3483,3517` — 7 callsites, unchanged.
