# Agent 4 — Threadpool & Parallelism Audit

Scope: single-process CPU decode path on Elbrus 8C2 (32 cores, usual sweet spot
`OMP_NUM_THREADS=24`). Focus files:

- `c10/util/ThreadPool.h` (custom pool, not OpenMP)
- `torch/io/cpu_quant_gemv.h` (all quant GEMV kernels)
- `torch/io/gguf_model.h` (per-token forward loop, CPU decode + TP decode)
- `torch/distributed/ddp.cpp` (SHM AllReduce used by TP, not by single-proc)

Headline finding: the inference hot path does **NOT** use `#pragma omp`. It was
explicitly migrated to a home-grown persistent `std::thread` pool
(`c10::get_thread_pool().parallel_for(...)`) specifically to eliminate the
OpenMP fork/join cost on LCC/E2K. The residual `#pragma omp` sites
(`gguf_model.h:4532, 4549, 4689`) are only on **batch/prefill** paths
(`matmul_scalar`, CPU-fallback `self_attention`) which single-token decode
never enters when the GGUF quant kernels are wired (the normal case). So
questions framed around OMP `schedule(...)` / `reduction(...)` mostly do not
apply to the 4.7 tok/s number — but the pool itself and its usage pattern
have a separate set of problems documented below.

---

## Q1 — OMP schedule / chunk size

**Direct answer:** N/A for the decode hot path. No `#pragma omp` is hit during
single-token inference.

What actually runs for an output row-split GEMV (e.g. `q4k_gemv_avx2`,
`cpu_quant_gemv.h:337`, `485`, `545`, `1504`, …):

```cpp
c10::get_thread_pool().parallel_for(0, N, [&](int64_t start, int64_t end){
    // process rows [start, end)
}, /*min_grain=*/1);
```

`ThreadPool::parallel_for` (`c10/util/ThreadPool.h:140-189`) hard-codes the
**static block** schedule:

```cpp
int64_t chunk_size = (total + num_threads_ - 1) / num_threads_;  // line 152
for (int i = 0; i < num_threads_; i++) {                         // line 155
    int64_t start = begin + i * chunk_size;
    ...
    tasks_.push([&fn, start, chunk_end, ...]);                   // line 173
}
```

So for `N=2560`, `num_threads=24` → chunk = 107 rows/thread, one contiguous
block per worker, no dynamic stealing. Positive: no per-iteration scheduling
overhead, which is what you want for balanced work. Negative: zero tolerance
for core-to-core frequency skew / NUMA asymmetry — the slowest thread sets the
barrier wall time (see Q3/Q4).

The `min_grain` parameter is misleading: it is a **threshold** (line 147) not
a chunk size. If `N < min_grain * num_threads` the call runs serially on the
caller's thread. For GEMV all quant kernels pass `min_grain=1`
(`cpu_quant_gemv.h:132, 472, 521, 1619`, …), i.e. always parallelize. The
softmax/SiLU parallel_for in `gguf_model.h:2603,2723` use `min_grain=256` and
`64` respectively — reasonable but see Q4.

**Blocker:** need `perf stat -e context-switches,task-clock` on Elbrus to
confirm whether the static block is actually perfectly balanced. If one NUMA
node is slower (likely — weights replicated per node, `ReplicatedWeight::get`
returns a different pointer per thread) the other 3 nodes idle-wait.

---

## Q2 — False sharing on `y[N_out]`

**Confirmed vulnerable.** Every quant GEMV writes one `float` per output row
into the caller-provided `y[]` buffer with `grain=1`, static block. Example
(`cpu_quant_gemv.h:459-460`):

```cpp
y[n]     = sum0;
y[n + 1] = sum1;
```

Static block means thread 0 writes `y[0..106]`, thread 1 writes
`y[107..213]`, etc. Within a thread's block there is no sharing, but **across
the block boundary 64 B = 16 floats straddle two workers**: thread 0's last
cacheline (`y[96..111]`) overlaps with thread 1's first cacheline
(`y[107..122]`). Each kernel writes its tail floats (≥1 cacheline of
boundary) on every tile, so each boundary line ping-pongs between cores.

Count of boundaries: 24 threads → 23 shared cachelines per GEMV. 36 layers ×
(Q+K+V+O+gate+up+down = 7 GEMVs/layer) × 23 = ~5800 boundary-line invalidations
per token. On Elbrus, cacheline is 64 B, coherence traffic over the internal
ring — a few hundred ns each, so O(1-2 ms/token). Not the dominant cost at
4.7 tok/s (= 212 ms/token), but measurable.

**No padding.** `y` is a raw `float*` coming from the scratch allocator in
`ScratchPad` (`gguf_model.h:~337-430`, `buf_attn`, `h_buf`, `logits_buf`,
`gate_buf`, `up_buf`). None of those scratch pointers are 64-B aligned with
per-thread padding slots.

The 2-row-at-a-time kernel makes this **worse, not better**: it writes
`y[n]` and `y[n+1]` adjacent, and across threads boundary is still not
cacheline aligned (107 is odd → boundary falls mid-cacheline). Should round
chunk size up to a multiple of 16 (64 B / 4).

Recommendation: see Q7.

---

## Q3 — Critical sections / reductions

The hot path has **zero** `reduction(+:...)`, zero `#pragma omp critical`,
zero `#pragma omp atomic`. Accumulation into `y[n]` is always thread-private
(each thread owns a disjoint row range), so the GEMVs do not need any
reduction — good.

However there **is** a single implicit cross-thread synchronization per
`parallel_for` call: `ThreadPool::parallel_for` (line 166-188) uses an
`std::atomic<int> tasks_done` + `done_cv_.notify_one()` after every chunk.
The main thread then `wait()`s on `done_cv_` until all chunks report in.
Protocol cost per `parallel_for`:

- 24 × `fetch_add` on `tasks_done` (contended atomic — one cacheline)
- 24 × `lock_guard(done_mutex_)` + `notify_one()` (serialized on `done_mutex_`)
- Main thread: lock `done_mutex_`, `wait()`, re-check predicate on each
  spurious wake

So one `parallel_for` call ≈ 24 serialized mutex handoffs on the main-thread
side. On Elbrus `pthread_mutex_lock` is ~1 µs. 24 × 1 µs × ~20 parallel_for
calls per token ≈ 0.5 ms/token of pure sync. At 4.7 tok/s this is <0.3%,
manageable — but it grows linearly with thread count, so going from 24 to
32 cores *reduces* throughput once per-chunk work is small.

Softmax inside attention (`gguf_model.h:2478-2489`) runs serial **per head**
inside the parallel_for over heads — the max/sum/normalize inner loop is
scalar, not parallel, and `scores[]` is thread-local (stack `local_scores[4096]`,
line 2444). So no cross-thread reduction needed. Good.

RMSNorm (`cpu_quant_gemv.h:1651-1691`, `cpu_rmsnorm_inplace` at 1851) runs
**serially on the calling thread**, SIMD only. It is *not* parallelized. For
`hidden=3584` that is ~450 sum_sq_add FMAs + 450 stores = negligible (~1 µs),
but for `vocab=152k` output rmsnorm it's the full H which is still cheap. OK.

The only real reduction on the hot path is `hsum_avx` at the end of each row
— scalar-lane horizontal sum inside one AVX register, no thread interaction.
Fine.

**Conclusion:** Q3 is not the bottleneck. Reductions are already done
correctly (thread-private accumulators, outer-only scope).

---

## Q4 — `#pragma omp parallel` regions per token (persistent threads?)

The pool **is** persistent (threads created once in
`ThreadPool::ThreadPool()`, live until process exit, wait on `cv_`) — so the
real cost is not fork/join, it's the **per-call dispatch/barrier**.

Count of `parallel_for` calls per token (counted from `gguf_model.h` CPU
decode loop, 36 layers):

Per layer:
1. `cpu_fused_rmsnorm_qkv_gemv` → 1 parallel_for (batched QKV, line 1504)
2. Attention heads → 1 parallel_for (line 2437)
3. `cpu_quant_gemv` output projection → 1 parallel_for (line 2526 → 485)
4. `cpu_fused_rmsnorm_gate_up_gemv` → 1 parallel_for (line 2566 → 1504, fused gate+up)
5. SiLU*up → 1 parallel_for (line 2603)
6. `cpu_quant_gemv` down projection → 1 parallel_for (line 2645 → 485)

= **6 parallel_for calls per layer × 36 layers = 216 dispatches per token**.
Plus 1 for output-proj GEMV (line 2696). ≈ 217 per token.

Each dispatch costs:
- 1 lock_guard on `mutex_` to push 24 tasks into `tasks_` queue (line 169-181)
- 1 `cv_.notify_all()` wake-up
- Wait on `done_cv_` for 24 completions (≥24 lock/unlock on `done_mutex_`)

At ~1-2 µs/dispatch on a well-tuned system → 0.2-0.5 ms/token on overhead
alone. On Elbrus with LCC pthreads and slower futex → likely 2-5 µs, giving
**0.4-1.0 ms/token sync overhead**, real but not the dominant cost.

**But the *real* issue is the lost parallelism between dispatches.** After
each `parallel_for`, workers go back to `cv_.wait()` on `mutex_`. Main thread
does:
1. Serial RMSNorm (line 1651-1691) — ~1 µs
2. RoPE on Q/K (`apply_rope_inplace`, not shown) — serial per-head SIMD
3. QK-norm (line 2392-2395) — serial
4. Residual add (line 2542-2554) — serial AVX2
5. Post-norms (line 2520-2523, 2652-2655) — serial

Each of these is ~1-5 µs, never parallelized. Total serial fraction per
token: ~36 layers × ~5 serial chunks × ~2 µs ≈ 360 µs. Again small.

**The dominant idle time is per-layer inside the GEMV itself** — thread
imbalance (Q1), false sharing (Q2), and NUMA-fetch stalls (Q5). The
dispatcher is not the problem; it's the work dispatched.

**Ideal fix:** a single `omp parallel` (or single persistent worker
activation) wrapping the entire 36-layer forward, with workers coordinating
via lock-free barriers for each stage. This removes 217 mutex+cv round-trips
and lets workers keep their L1/L2 state warm (x vector, RMSNorm gamma,
attention scores). **Blocker:** non-trivial rewrite, 200+ lines of code; need
perf numbers first to prove the sync cost is >5% of token time. The 0.5-1 ms
estimate suggests it's ~3-5% of the 212 ms budget — worth doing eventually,
not the critical fix today.

---

## Q5 — NumaReplica vs interleave: who waits?

Code reads `c10::current_numa_node()` inside each `parallel_for` chunk
(`cpu_quant_gemv.h:70, 339`, …):

```cpp
const uint8_t* raw = (numa && numa->num_replicas > 1)
    ? static_cast<const uint8_t*>(numa->get(c10::current_numa_node()))
    : static_cast<const uint8_t*>(weight_data);
```

`current_numa_node` uses a `thread_local int t_numa_node` cache
(`ThreadPool.h:55`), primed on worker start (line 217). So the lookup is
free, correct after the first call.

**But pinning is disabled by default.** `PT_PIN_THREADS` must be set to `1`
to call `pthread_setaffinity_np` (line 107-110, 205-213). If not set, the
kernel is free to migrate workers across NUMA nodes. When a migration
happens:
- `t_numa_node` (thread_local) stays at the old value
- `numa->get(old_node)` returns a pointer to the *previous* node's replica
- Every load from that pointer is a **remote DRAM** fetch

On Elbrus 8C2 the remote-node memory latency penalty is ~2× local. Q4_K row
= 144 B × 2560 rows = ~360 KB per GEMV (K=3584, 14 blocks/row). At 24 threads
splitting this, each thread reads ~15 KB/GEMV of weights — fits in L2. But
**migration happens on OS scheduling ticks** (every 4-10 ms on Linux), so a
mid-GEMV migration stalls one thread for ~100 µs fetching fresh lines from
remote DRAM.

**The VLIW angle:** On E2K the compiler schedules instructions into wide
issue slots at compile time. A long-latency miss (remote DRAM = several 100s
of cycles) stalls the entire bundle — other issue slots go idle *on that
core*. It doesn't directly stall other cores (they have their own
instruction streams), **but** the `ThreadPool::parallel_for` barrier
(waiting on `done_cv_` on main) means the main thread cannot proceed until
the slowest worker completes. So one remote-miss-stalled worker holds up the
whole GEMV — that's the "serialization" effect.

**Confirmed via:** pool default is `pin_enabled_ = false`. The training docs
say launch with `numactl --cpunodebind=...`, which pins the *entire process*
to one node — but single-process inference does not do this.

**Blocker:** need `OMP_DISPLAY_ENV=VERBOSE` and
`numastat -p <pid>` on Elbrus during inference to confirm migration rate and
remote-DRAM miss percentage. Predicted: PT_PIN_THREADS=1 gives >10% speedup.

---

## Q6 — TP (fork) vs threadpool

TP path uses SHM AllReduce (`ddp.cpp:513-575`). **This is a spin-wait**
(`ddp.cpp:535-537`):

```cpp
while (hdr->arrived[gen % 16].load(std::memory_order_acquire)
       != (uint32_t)s.world_size) {
    __sync_synchronize();        // full memory barrier every iteration
}
```

Same pattern on lines 555-558 (rank 0 waits for workers to depart) and
564-566 (workers wait for rank 0 to publish). Every wait is a hot spin with
a full `mfence`/`sfence` equivalent on every loop turn. **No `pthread_yield`,
no `sched_yield`, no exponential backoff, no `__builtin_ia32_pause` (x86) /
E2K equivalent.**

Consequence: if rank 0 is slow by even 100 µs, the other 3 ranks burn 100%
CPU spinning — which on Elbrus means they occupy 3 × 8 cores = 24 cores
executing useless memory-barrier loops. Since the OS scheduler can't see
they're waiting (they're not blocked), it cannot give their cores to the
rank 0 worker threads that actually have work. This is the single worst
behaviour for multi-process TP on a shared-socket box.

Each TP AllReduce pair (attention-out + FFN-down) runs twice per layer × 36
layers = 72 AllReduces/token. With 4 TP ranks on 32 cores, spin-wait can
easily eat 10-30% of total CPU budget depending on imbalance. For the
**single-process 4.7 tok/s** number this is not triggered (PT_DDP_SHM only
enables with world_size > 1 / TP mode). So Q6 is not the immediate culprit
for 4.7 tok/s, but becomes critical the moment TP is turned on to chase
higher throughput.

Single-process pool (our case) vs TP:
- Pool: 1 heap, 24 std::thread workers, shared weights (no replication
  overhead); single NUMA-node binding recommended to keep `t_numa_node`
  stable. Upside: simpler, less sync. Downside: bounded to one node's DRAM
  bandwidth unless weights are replicated.
- TP: 4 processes, each has own heap + own weight replica (~3 GB × 4 =
  ~12 GB). Gets all 4 nodes' bandwidth. Downside: SHM spin-wait burns idle
  cores (above). File-based AllReduce + signals would be better for this
  workload size than spin-wait.

---

## Q7 — Concrete proposals (file:line)

All suggestions are diagnostic or non-invasive hints. No code changes in
this audit per instructions.

### P1 — Enable thread pinning by default for inference
- **Where:** `c10/util/ThreadPool.h:107-110` — currently `pin_enabled_`
  defaults to false.
- **Why:** keeps `t_numa_node` stable (Q5), locks each worker to one core,
  allows L1/L2 to stay warm across parallel_for calls (Q4).
- **How to test:** `PT_PIN_THREADS=1 ./promeserve_cpu ...` — existing env
  already works, just not documented / not default. Measure delta.

### P2 — Round GEMV row-chunks to 16-element boundaries (cacheline)
- **Where:** `c10/util/ThreadPool.h:152`
  `int64_t chunk_size = (total + num_threads_ - 1) / num_threads_;`
- **Why:** eliminates false-sharing on `y[]` (Q2). 2560/24 = 107 (bad), but
  ceiling-to-16 gives 112 → boundaries at 0, 112, 224, … all on cacheline
  edges, zero ping-pong.
- **Variant:** only round up for fn calls where the caller asks (new
  overload), so small-grain ops (softmax over heads with head_count=32) keep
  fine-grained distribution.

### P3 — Align scratch buffers to 64 B and pad
- **Where:** `torch/io/gguf_model.h` `ScratchPad` allocator (~337-430,
  `buf_attn`, `h_buf`, `logits_buf`, `gate_buf`, `up_buf`, `silu_buf`,
  `scores_buf`, `x_buf[2]`).
- **Why:** guarantees `y[0]` starts on a cacheline; combined with P2
  eliminates all boundary ping-pong. Currently `at::empty({1,dim})` returns
  whatever the allocator produces.
- **How:** force `posix_memalign(..., 64, ...)` in `ScratchPad::alloc()`.

### P4 — Replace SHM AllReduce spin-wait with futex
- **Where:** `torch/distributed/ddp.cpp:535-537, 555-558, 564-566`.
- **Why:** spin-wait burns idle cores (Q6). On 4-rank TP, one slow rank
  = 24 cores burning `__sync_synchronize()` for its duration.
- **How:** `futex(FUTEX_WAIT, &arrived, expected)` on Linux, fall back to
  `pthread_cond_wait` on the shared condition variable in `/dev/shm`. For
  very small payloads (<4 KB) spin with `__builtin_ia32_pause` for a bounded
  number of iterations (~100) before falling into the futex.
- Not a single-proc fix but **critical before enabling multi-proc TP**.

### P5 — Collapse per-layer parallel_for calls
- **Where:** `gguf_model.h:2437, 2603, 2696` and the GEMVs called inside.
- **Why:** 217 dispatches/token → ~1 ms sync (Q4). One persistent worker
  activation with step barriers would save 3-5% token time.
- **Complexity:** ~200 LoC rewrite. Lower priority; do P1-P3 first.

### P6 — Verify SiLU `min_grain`
- **Where:** `gguf_model.h:2627` — `parallel_for(... min_grain=256)` for
  inter=9728. That's 9728/24 ≈ 405 elements/thread > 256 threshold, good.
  But check that `inter` is never <6144 (256×24), else softmax-like regions
  go serial.

### P7 — Check scratch `scores[4096]` false-sharing
- **Where:** `gguf_model.h:2444-2445` —
  `float local_scores[4096]; float* scores = (total_seq <= 4096) ? local_scores : sp.scores_buf;`
  The stack buffer is thread-private → no sharing. Good. But the *fallback*
  `sp.scores_buf` is a **shared pointer** across all threads inside the
  parallel_for → **data race** if `total_seq > 4096`. Blocker: verify this
  is unreachable (max context limit) or fix with per-thread index.

---

## Unblocked / unknown

- Need `perf stat -a -e cpu-migrations,context-switches,LLC-load-misses,
  LLC-store-misses` on Elbrus during a decode run to confirm Q5
  (remote-DRAM rate).
- Need `OMP_DISPLAY_ENV=TRUE` captured to confirm the Elbrus LCC OpenMP
  runtime is not being touched at all (it isn't — we use `std::thread`
  directly — but this is the mission's specific question).
- Need microbenchmark: run one `cpu_quant_gemv` GEMV 10k times with
  `PT_PIN_THREADS=0` vs `PT_PIN_THREADS=1` to quantify P1 impact. Hypothesis:
  +5-15%.
- Need to count actual `parallel_for` dispatches per token via a counter
  in `ThreadPool::parallel_for` (line 166). Hypothesis: 217 ± 5.

## Summary

| Question | Hot path? | Finding | Severity for 4.7 tok/s |
|----------|-----------|---------|------------------------|
| Q1 schedule/chunk | Yes | static block, 107 rows/thread, N/A OMP | Low |
| Q2 false sharing | Yes | chunks not cacheline-aligned, `y[]` shared boundaries | **Medium** |
| Q3 reductions | Yes | none — already thread-private accumulators | Low |
| Q4 parallel regions/token | Yes | ~217 `parallel_for` dispatches, persistent pool | **Medium** |
| Q5 NUMA / migration | Yes | `PT_PIN_THREADS=0` default → migration stalls | **High** |
| Q6 TP SHM spin-wait | TP only | hot spin burns idle cores | High (TP mode) |
| Q7 proposals | — | P1+P3+P2 unblock estimated 10-20% of single-proc cost | — |

**Highest-ROI single change: set `PT_PIN_THREADS=1` by default (P1).**
Followed by 64 B alignment of scratch buffers (P3) and cacheline-aligned
chunk rounding (P2). The 217-dispatch-per-token issue (Q4/P5) is real but
second order until P1-P3 are done.

File paths referenced:
- `C:\Users\paper\Desktop\promethorch\c10\util\ThreadPool.h`
- `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h`
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h`
- `C:\Users\paper\Desktop\promethorch\torch\distributed\ddp.cpp`
