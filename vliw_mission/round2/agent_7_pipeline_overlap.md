# Agent 7 — Layer Pipeline / Weight-Load Overlap

**Date:** 2026-04-22
**Scope:** `forward_decode_cpu` on qwen3:4b (36 layers). Can we overlap
weight-stream reads for layer L+1 with compute of layer L?
**Sources:**
- `vliw_mission/round2/_inputs/mcst_main_guide.txt` (Elbrus programming guide,
  APB/prefetch/software-pipeline sections)
- `vliw_mission/round2/_inputs/cache_optimization.txt` (shared-L3 NUCA)
- `vliw_mission/round2/_inputs/memory_controller.txt` (Elbrus-16СВ DDR4 MC)
- `vliw_mission/round2/_inputs/lcc_auto_parallel.txt` (APB, rotating-SWP)
- `torch/io/gguf_model.h`   — `forward_decode_cpu` (line 2308)
- `torch/io/gguf_loader.h`  — `MmapHandle::open` (line 215-265), `lock_region` (287)
- `torch/io/cpu_quant_gemv.h` — Q4_K kernels
- `torch/io/numa_weight_replica.h` — `ReplicatedWeight`
- `c10/util/ThreadPool.h` — `parallel_for` / busy-sleep semantics

---

## TL;DR

The premise of the question ("overlap weight read for L+1 with compute of L
should give 2× speedup") is wrong on Elbrus-8C2 for this workload, for
**three** independently fatal reasons:

1. **We are already DDR-bandwidth-bound at 30% of the per-chip ceiling
   (MISSION.md:10-12).** Adding a dedicated prefetch stream does not
   increase the aggregate bytes-per-second the DDR4 channel can push — it
   just reallocates the *same* bandwidth between two consumers. The ceiling
   stays at ~50 tok/s single-chip, ~200 tok/s four-chip (MISSION.md:10-11).
   Pipelining only helps when compute and memory are two *independent*
   resources with slack on one; here compute is the slack side (4.6 ms) and
   memory is the bottleneck (~5 ms), which means every byte prefetched
   *competes* with the GEMV of the current layer for the same DDR channel.

2. **The APB hardware cannot bridge layer boundaries.** APB is a per-loop,
   per-procedure resource that is emptied at function call/return and at
   loop exit (`mcst_main_guide.txt:3937-3938`: *"при отсутствии вызовов
   функций в цикле — механизм не допускает сохранения и восстановления
   при вызове"*). We already saturate it on each inner j-loop of the GEMV
   (agent_2 found `#pragma loop count + ivdep` on every j-loop). There is
   no APB state that survives from the end of layer L's `ffn_down` to the
   start of layer L+1's `attn_norm`.

3. **The existing cross-layer prefetch at `gguf_model.h:2356-2399` (64 KB ×
   7 weights = 448 KB, `_mm_prefetch _MM_HINT_T1`) already touches every
   TLB entry we need and hides the mmap page-fault latency. The *bulk* of
   the weights (70 MB / layer for qwen3:4b Q4_K) cannot be prefetched
   productively: 70 MB ≫ 16 MB shared L3 (one Elbrus-8C2 cluster), so any
   byte we pull in early is evicted before GEMV reaches it.**

So this entire report would be a nope-report, except there are a few things
in the question that are *real* sub-optimizations and that we should apply.
Items 2, 4, 5, 6 are rejected with evidence. Items 1 (partial), 3, 7 are
worth a surgical patch. See the "Concrete changes" section at the end.

---

## Per-question analysis

### 1. Decoupled parallelism: overlap L+1 weight read with L compute?

**Claim:** 70 MB/layer × (14 GB/s DDR) = 5 ms read. Compute is 4.6 ms.
Looks perfect for pipelining.

**Why the claim is wrong:**

The 14 GB/s figure is the channel's *peak* burst bandwidth. Actual sustained
bandwidth on Elbrus-8C2 during Q4_K GEMV is measured indirectly through
MISSION.md:10-12: "Per-chip DDR effective: ~20 GB/s × 0.4 KB/token-weights
= ~50 tok/s absolute single-chip". At 5.3 tok/s currently, we are already
consuming 5.3 × 2.4 GB = **12.7 GB/s** of that 20 GB/s chip-local ceiling.
That means DDR has ~7 GB/s of *headroom*, not 14. And that 7 GB/s has to
cover the prefetch-of-L+1 at the *same time* as the demand-fetch-of-L.

Concrete check:
- Layer demand-fetch (L, currently in GEMV): ~70 MB in ~4.6 ms ≈ 15 GB/s
  on one channel.
- Dedicated prefetch of L+1: would need the same ~15 GB/s.
- Channel ceiling: ~20 GB/s.

So the pipeline would stall *both* streams at ~10 GB/s each. Effective
per-layer time becomes the max of (fetch/half-BW, compute) ≈ max(9.2 ms,
4.6 ms) = 9.2 ms. That is WORSE than the current 5 ms serial fetch, and
matches the "interleave regresses 1-proc" result already noted in
MISSION.md:25.

**The only regime where `Decoupled parallelism` wins** is when compute is
>> memory latency, which on qwen3:4b decode it is not. A 14B model or
float32 weights would change this — but for Q4_K 4B we're in the
memory-bound regime. This is exactly what FP16-weights-won't-speedup-decode
covers (MEMORY.md feedback).

**Verdict: REJECT.** No code change. Keep the current 64 KB warming
prefetch and the current serial schedule.

### 2. Prefetching-by-layer with APB engaging on long-range prefetch?

**Claim:** Add `fapb`-like prefetch at the end of layer L that pulls L+1's
weight stream into APB.

**Why APB cannot do this:**

Quote `mcst_main_guide.txt:3922-3932`:
> "Суть его состоит в следующем: доступ к массивам описывается особым
> образом в виде кода асинхронной программы. Она состоит только из
> операций fapb. Операции fapb запускаются по циклу, пополняя буферы
> упреждающих данных для разных массивов. При этом основной поток
> исполнения забирает данные из этого буфера операциями mova (вместо
> запуска операций чтения)."

APB is inseparable from the `mova` consumption pattern. The consumers
(`mova` ops in the main instruction stream) are what *credit* the APB
buffer; an `fapb` issued at the end of layer L has no corresponding `mova`
stream to back it, and the buffer is *invalidated* on function return
anyway (`mcst_main_guide.txt:3937-3938`, and the bullet at 3941: "при
наличии аппаратного счётчика цикла %lsr" — APB needs the hardware loop
counter to be armed).

What would work *instead* is `__builtin_prefetch` (software prefetch via
`ld->empty`, `mcst_main_guide.txt:3842-3853`), and we already do that at
`gguf_model.h:2378-2389` with `_MM_HINT_T1` (which on Elbrus maps via LCC
intrinsic translation to `ld.s … → %empty, mas=0x20` — "не заводить в
L1$"; the L2-hint equivalent). The question is whether we prefetch
*enough* or *far enough ahead*.

Current behavior (`gguf_model.h:2378`):
```
for (int off = 0; off < 65536; off += 4096) {
    _mm_prefetch(cp + off, _MM_HINT_T1);
}
```
That's 16 prefetch ops (1 per 4-KB page) × 7 weights = 112 ops per layer.
Each op warms **one TLB entry** per 4-KB page. This is literally what the
comment at `gguf_model.h:2357-2359` says: the goal is to pre-populate TLB,
not to prefetch the weight data.

Actual weight data for the 70 MB layer would need 70 MB ÷ 64 B/line ≈ 1.1M
prefetch ops — impossible to issue in-band. The real long-range
"prefetch" on Elbrus is the *DDR controller's own streaming predictor*
described in `memory_controller.txt:209-345` (the scheduler's "filter of
priority to open page" keeps a page open if another request is queued for
it). That predictor works on *sequential physical addresses*, and Q4_K
weights *are* stored contiguously per tensor (mmap'd from the GGUF file).
So the MC already does the L2-miss → open-DDR-page → burst-read sequence
for us on the critical inner GEMV loop. No further long-range prefetch
helps.

**Verdict: REJECT.** Don't add layer-prefetch. The existing TLB warming is
correct and sufficient; the *data* prefetch is the DDR controller's job and
is already optimal because Q4_K layout is sequential.

### 3. Attention softmax (8.7 ms) — overlap with FFN weight fetch?

**This is the one idea that is worth thinking about.** Attention reads the
KV cache (not weights) and computes softmax/exp — so in principle its
memory stream and the FFN `gate/up` weight stream hit *different* physical
addresses and the MC can interleave them.

BUT — and this kills the idea for the current structure — attention runs
under `parallel_for` (`gguf_model.h:2550`), meaning every worker thread is
already reading KV cache from every NUMA node. The workers fan out to all
cores. There is no idle core to dedicate to prefetching FFN weights.

Two sub-options:
  (a) Shrink `parallel_for` to `n_heads - 1` workers and use worker N to
      prefetch. For qwen3:4b n_heads=32 on 8 cores/chip: 32/7 ≈ 5 chunks
      vs 32/8 ≈ 4 chunks → +25% attention latency. Even if the prefetch
      saves 1 ms on FFN load, you lose 1.9 ms on attention. Net negative.
  (b) Issue `_mm_prefetch` for FFN weights from *inside* the attention
      kernel (agent_4 / fused_attn_prefetch). This does not require a
      dedicated thread because `_mm_prefetch` is fire-and-forget. But:
      the inner attention loop is dot-product over `total_seq × head_dim`
      (float) which is already 100% BW-bound on L2. Adding software
      prefetches to a DDR address is just increasing DDR pressure while
      attention is competing for the same channel.

The only version of (b) that works is: **prefetch FFN `gate/up` weights
during softmax (the scalar exp-loop), not during the V-sum AVX2 loop.**
Softmax is compute-bound on Elbrus (scalar `std::exp` per token — see
comment `gguf_model.h:2712-2715` which explicitly notes "std::exp on E2K
libm is scalar software, ~200ns/call"). During those scalar exp calls, the
6 ALUs are half-idle and can issue prefetches cheaply.

For `total_seq=1024, head_dim=128` at softmax call-rate: ~1024 exp calls
per head × 32 heads ÷ 8 workers = 4096 exp calls per worker × 200 ns ≈
0.82 ms idle per worker. That can absorb ~16 KB × 8 workers = **128 KB of
L2 warming** while other loads aren't using the channel. Real, but
negligible vs 70 MB layer size.

Expected speedup: **0.2-0.5 ms per layer × 36 = 7-18 ms/token ≈ +0.6 tok/s
at the 5.3 tok/s baseline.** Worth a try if cheap.

**Verdict: WEAK ACCEPT (tiny gain).** Add software prefetches to the
next-FFN weight footprint inside the softmax loop body at
`gguf_model.h:2593-2602`.

### 4. Double-buffered weight mmap + madvise(WILLNEED)?

**Current state:** `gguf_loader.h:261` issues `madvise(data_, size_,
MADV_SEQUENTIAL)` once at mmap-open time. `MADV_SEQUENTIAL` means "read-
ahead more aggressively and drop pages behind". For a 2.4 GB file on a
125 GB box, after first touch every page is resident forever, so
`MADV_SEQUENTIAL` is actually suboptimal for the steady state — it tells
the kernel to drop trailing pages, which invites faults on the *next
token's* use of layer 0 weights.

**`MADV_WILLNEED`** (as the question proposes) explicitly hints "schedule
read-ahead for these pages". Issuing it per-layer is expensive — it
syscalls into the kernel for each call — and the Linux read-ahead scheduler
already detects the sequential access pattern from the GEMV (which linearly
walks 70 MB per layer). So explicit `WILLNEED` would duplicate work the
kernel already does for free.

The **right** fix in this area is different from the question: issue
`MADV_RANDOM` (or nothing) *after* initial scan completes, NOT
`MADV_SEQUENTIAL`. `MADV_RANDOM` turns off the eviction behavior and lets
pages stay resident. The loader comment at line 260 even acknowledges this:
*"Hint: sequential access for initial scan, random for inference"* — but
the code never switches the hint to random.

For truly huge models (>RAM), `MADV_WILLNEED` does help — but that's a
Phase-2 capability (AirLLM layer-streaming, per `project_ollama_killer.md`),
not qwen3:4b.

**Verdict: PARTIAL ACCEPT (wrong hint chosen).** Change the single
`madvise(data_, size_, MADV_SEQUENTIAL)` at `gguf_loader.h:261` to
`MADV_RANDOM` — OR issue the switch-to-RANDOM after the first full token
decode completes. Do NOT introduce per-layer `MADV_WILLNEED`.

### 5. Multi-core pipeline: N cores dedicated to `readahead()`?

**Why this fails:**

- For an mmap'd file that is fully resident (qwen3:4b in 125 GB), every
  page is already cached. `readahead(2)` is a no-op.
- For a file not fully resident, `readahead` *itself* consumes DDR
  bandwidth (reading from disk pumps through the DDR → NVMe DMA, which
  shares the same memory controller on Elbrus-8C2).
- Dedicating cores means losing them from `parallel_for`. On 8 cores with
  32-head attention, dropping to 6 cores raises attention latency by 33%.
  Attention is `sec_timers_.attn_ms` = ~8 ms/layer × 36 = 288 ms/token at
  5.3 tok/s — dominant after GEMV. +33% on that = +95 ms/token, total
  budget goes 189 → 284 ms → 3.5 tok/s. Huge regression.

Work-stealing wouldn't change this: the workers doing GEMV are *already*
DDR-bound, so when attention is in flight they sit on the DDR bus not on
ALU queue. There is no ALU slack to steal.

**Verdict: REJECT.** No change. Cores stay on compute.

### 6. Layer-level async compute: overlap Q/K/V GEMV of layer L with RMSNorm of layer L?

**Dependency graph (current serial order):**
```
x[cur] ──> RMSNorm ──> x_norm ──> Q_GEMV ──┐
                                  K_GEMV ──┼──> attention ──> ...
                                  V_GEMV ──┘
```
The question proposes overlapping Q/K/V GEMV with... what? The RMSNorm of
the *same* layer? That's impossible — RMSNorm produces `x_norm`, which is
the input to Q/K/V. A hard read-after-write dependency.

What the question probably means is: **overlap Q/K/V GEMV of layer L with
RMSNorm of layer L+1.** But layer L+1's RMSNorm needs `x[cur]` after the
L-layer residual add (`gguf_model.h:2770-2785`, which needs `h_buf`, which
needs `ffn_down`). So RMSNorm of L+1 also depends transitively on Q/K/V of
L completing. Serial.

The *only* layer-level parallelism that works for a single decode token is:

  (a) **Intra-layer op-level** — `gate` and `up` GEMV are independent, and
      we already fuse them (`cpu_fused_rmsnorm_gate_up_gemv`,
      `cpu_quant_gemv.h:2255-2261`).
  (b) **Intra-layer head-level** — attention heads are independent, and
      we already parallelize them (`gguf_model.h:2550`).

For layer-level overlap to happen, you need either (i) batch size > 1
(pipeline different tokens through different layers, each at a different
timestep) — which is speculative decode / continuous batching, and is
already in scope via `forward_decode_cpu_batched` and the spec-decode
scaffold (MISSION.md:19-21); or (ii) tensor-parallel where a chip owns
1/4 of each matrix and works on L+1's chunk while waiting for AllReduce
from L (current TP-4 already does this structurally).

**Verdict: REJECT.** Inter-layer data dep is transitive and cannot be
broken for a single-token decode.

### 7. VLIW software pipelining between layers — manual prefetch at boundaries?

**This is the only form of the question that the MCST guide explicitly
supports.** Quote `mcst_main_guide.txt:3830-3842`:

> "Для решения проблемы с блокировками конвейера, вызванными операциями
> чтения, в архитектуре «Эльбрус» предусмотрено несколько методов:
> - Ацикличные участки кода: совмещение блокировок, ограничение на
>   простановку маловероятных чтений в спекулятивный режим.
> - Цикловые участки кода: совмещение блокировок в конвейеризированных
>   циклах; выявление регулярных чтений, предподкачка с помощью
>   prefetch (ld→empty); использование аппаратно-программного механизма
>   для подкачки линейных данных."

The pivot is this: "**Ацикличные участки**" — the acyclic region *between*
two loops is where LCC cannot apply SWP/APB because there is no loop to
pipeline across. The code at `gguf_model.h:2400-2410` (between the
cross-layer prefetch block and `if (can_fuse)`) is exactly such a region.
It computes the `can_fuse` boolean from 9 pointer/int comparisons — 9 ALU
ops that fit in one ~3-tick window but sit on the critical path for the
start of the GEMV. This is a *compute* bubble, not a memory bubble, so
software prefetch here is useful: a `__builtin_prefetch` on the first cache
line of `w_q` / `w_k` / `w_v` immediately before calling
`cpu_fused_rmsnorm_qkv_gemv` primes the L1 cache for the RMSNorm's first
`x` read (which is already resident) and for the start of the GEMV's first
row (which is not).

Concretely, the missing prefetches are:
- `layer.attn_norm.data_ptr<float>()` — small (H×4 = 15 KB), fits in L1,
  currently prefetched *for the NEXT layer* but not explicitly warmed
  before the CURRENT layer's RMSNorm (it *happens* to be warm because the
  previous layer's cross-layer-prefetch loaded it with T0 hint — this is
  correct and already there at `gguf_model.h:2384-2390`).
- `sp.x_buf[cur]` first cacheline — always warm, produced in L0d1
- The `w_q/w_k/w_v` first 64 B of each row — ~192 B total — not prefetched
  right before the GEMV (the GEMV's own internal prefetch at
  `cpu_quant_gemv.h:387-392` is one block ahead but the first block is a
  cold miss).

Gain is tiny: one L2 miss (~20 ticks at 1.5 GHz ≈ 13 ns) × 3 matrices ×
36 layers × 2 fused sections = ~3 µs per token. Not worth it.

**What IS worth doing at the layer boundary** is *moving the existing
cross-layer prefetch (at 2361-2399) to BEFORE the FFN of the previous
layer, not after it.* Current sequence:

```
layer L: attn_output_residadd → [PREFETCH L+1] → FFN_gate_up → FFN_silu → FFN_down → residadd
```

The prefetch fires at line 2361 which is at the *top* of layer L's
iteration (i.e. after layer L-1's FFN_down residadd). So the prefetch of
L's next iteration's weights lands at the START of L. Put differently:
*when layer L begins, the prefetch is for layer L+1, but layer L's weights
were prefetched at the start of L-1.* That is correct in distance (1 layer
ahead) but the prefetch happens while **layer L is about to read its own
QKV matrices** — so the first 64 KB of each of L+1's weights (1 MB total)
competes on the DDR channel with L's QKV GEMV. Since DDR is bottleneck,
this *slows down L*.

The fix: issue the prefetch during the quiet windows within the current
layer:
  - **after** QKV GEMV + before attention (attention is compute-bound on
    softmax scalar exp — BW slack),
  - **during** softmax scalar loop (same reason, see item 3),
  - **during** SiLU (elementwise, L1-resident, BW slack).

Move the 2361-2399 block out of the top-of-layer and scatter its 7 weight
prefetches across those three quiet windows:
  - attn_q / attn_k / attn_v: right before softmax (so by the time we
    finish V-weighted-sum the TLB is warm for the next layer; weights
    themselves won't fit in cache but the TLB entries will)
  - attn_output / ffn_gate / ffn_up: during softmax's scalar exp loop
  - ffn_down: during SiLU+up mul

**Verdict: ACCEPT.** Concrete code move below. This is where the real
win (if any) lives.

---

## Concrete file:line changes

### Change A — `gguf_loader.h:261`

**File:** `torch/io/gguf_loader.h`, line 261

**Before:**
```
madvise(data_, size_, MADV_SEQUENTIAL);
```

**After:**
```
// For decode, we re-read the whole model every token. MADV_SEQUENTIAL
// tells the kernel to drop pages behind the read cursor — wrong for
// decode where layer 0 will be read again next token. MADV_RANDOM
// keeps all pages resident.
// (For initial GGUF header parsing, sequential would be right, but the
// parse is a single pass of <1 MB — not the 2.4 GB weight region.)
madvise(data_, size_, MADV_RANDOM);
```

**Expected gain:** eliminate page-eviction-then-refault on layer 0 at the
start of each new token. On qwen3:4b this was measured at ~12 ms/token
during tail-end decode when the VM subsystem was under pressure (no
current measurement for Elbrus specifically — estimate ±5 ms/token).

**Risk:** if the model >RAM, `MADV_RANDOM` removes the helpful prefetch.
Current models fit. Gate behind `if (size_ < total_ram / 2)`.

---

### Change B — Move & scatter cross-layer prefetch

**File:** `torch/io/gguf_model.h`

**Before:** the prefetch block at lines 2361-2399 fires at top-of-layer.

**After:** delete the top-of-layer prefetch block, introduce three small
helper prefetches at three points in the layer body.

Point B1 — right before attention parallel_for at line 2549 (after QKV
GEMV, before softmax), prefetch **next layer's attn_q/k/v first 64 KB
each** with `_MM_HINT_T1`.

Point B2 — inside the softmax scalar loop at lines 2593-2602, before
the `std::exp`, issue a single `_mm_prefetch` line per exp iteration
targeting the next layer's `attn_output/ffn_gate/ffn_up`. Rate-limit:
only every 8th exp call (softmax does up to 4096 exp calls and we want to
warm 3 × 16 = 48 TLB entries; 48/4096 = 1.2% of iterations).

Point B3 — inside the SiLU parallel_for at lines 2716-2740, issue
`_mm_prefetch` for the next layer's `ffn_down` (16 pages).

Point B4 — also warm next-layer's `attn_norm` and `ffn_norm` with
`_MM_HINT_T0` in the final residual-add loop at lines 2772-2784 (H ≤
8192 floats ≈ 32 KB each, fits in L1, needed in the NEXT iteration's
RMSNorm).

**Expected gain:** the 1 MB of cold-miss traffic for TLB warming no
longer collides with the current layer's GEMV demand-fetch. Saves ~0.5 ms
per layer × 36 = ~18 ms/token. At 188 ms/token baseline: +0.5 tok/s
(5.3 → 5.8). Uncertain ±0.3; could be null if MC already schedules well.

**Risk:** low. If gain is null, revert the scatter (pure code move).

---

### Change C — Drop `parallel_for` sync overhead on SiLU short path

**File:** `torch/io/gguf_model.h`, line 2716

Not the topic of this agent but observed during read: `inter = 9728`,
`min_grain = 256` → 38 chunks, 8 workers, 5 chunks each. The 38 chunks
require 38 condition-variable notifications in ThreadPool
(`ThreadPool.h:184-197`, `tasks_done.fetch_add + done_cv_.notify_one`).
Each CV wakeup is ~1-5 µs on Linux. 38 × 3 µs = 114 µs per SiLU × 36
layers = 4 ms/token pure sync overhead.

If SiLU were fused into the gate+up GEMV epilogue (write SiLU(gate) * up
directly to `gate_buf` as the last AVX store of `cpu_fused_rmsnorm_gate_
up_gemv`), this disappears. But that's a kernel change, out of scope for
this agent.

**Verdict: note for kernel agent.**

---

### Change D — Rework `_MM_HINT_T1` → `_MM_HINT_NTA` on full-64KB scan

**File:** `torch/io/gguf_model.h`, lines 2378-2380

Current prefetch loop touches 16 × 4-KB pages × 7 weights × `_MM_HINT_T1`
(L2) per layer. But the weights will not all fit in L2 (8C2 L2 is small,
L3 is ~16 MB shared). When the next layer runs, L1 is flushed by the
RMSNorm and the GEMV actually benefits from NTA (non-temporal, streaming)
prefetch because the weights are read once per token then evicted anyway.

`_MM_HINT_NTA` on Elbrus maps (via LCC intrinsic translation) to
`ld.s ..., mas=0x40` (*"не заводить в L1$ и L2$"*, `mcst_main_guide.txt:
3865`). This is the right hint for one-shot-per-token weight reads.

However: this is only a TLB warmer, not a data warmer. The TLB entry is
loaded the same way regardless of `mas`. So the `NTA` change is cosmetic
for TLB warming but protects against cache pollution of L2 by the
prefetch (it steals L2 lines that the current layer's GEMV needs).

**Verdict: ACCEPT.** Change the `_MM_HINT_T1` to `_MM_HINT_NTA` at
`gguf_model.h:2379`. Save L2 pollution. Gain small, <0.5 ms/token.

---

## Summary of recommended changes

| # | File | Line | Change | Expected gain | Confidence |
|---|------|------|--------|---------------|------------|
| A | `torch/io/gguf_loader.h` | 261 | `MADV_SEQUENTIAL` → `MADV_RANDOM` | +0-5 ms/tok | medium |
| B | `torch/io/gguf_model.h` | 2361-2399 | Move prefetch out of top-of-layer, scatter across attn/softmax/SiLU/residual-add | +10-18 ms/tok | low-medium |
| D | `torch/io/gguf_model.h` | 2379 | `_MM_HINT_T1` → `_MM_HINT_NTA` | +0-0.5 ms/tok | low |

Combined ceiling: **~3-10% speedup (5.3 → 5.5-5.8 tok/s)**. Nowhere near
the 20 tok/s Round 2 target.

## Why this agent's category is fundamentally capped

MISSION.md:10 states the DDR ceiling: "Per-chip DDR effective: ~20 GB/s ×
0.4 KB/token-weights = ~50 tok/s absolute single-chip". We're at 5.3 =
~10% of that. The gap between 5.3 and 50 is **not** a pipelining problem —
it is a *weight bytes per token* problem. The only levers that close it
are:

1. **Skip weight reads entirely for some tokens** — speculative decode
   (spec_decode_step_cpu, already scaffolded). At 30% acceptance rate,
   3 tokens/draft ≈ 12 tok/s. This is the biggest lever.
2. **Reduce bytes per weight** — Q4 → Q3 or smaller. Out of scope.
3. **Batch tokens** — prefill-style batching doesn't apply to interactive
   decode, but `forward_decode_cpu_batched` for spec-decode verification
   already compresses per-token cost.
4. **Cross-chip bandwidth aggregation** — TP-4 with *proper* AllReduce (we
   only get 6.1 tok/s today, should be ~15 tok/s if AllReduce were free).
   Orthogonal to pipelining.

Layer-compute / weight-load overlap is *not* one of these levers. The
fundamental block is: **compute is 2-4× cheaper than memory on Elbrus-8C2
for Q4_K single-token decode, and overlapping them just reallocates a
fixed memory pie between two consumers.**
