# Agent 6 — Async Layer Pipelining (Round 3)

**Date:** 2026-04-27
**Baseline:** qwen3:4b Q4_K_M TP-4 = 4.8 tok/s on Elbrus 8C2.
**Predecessor:** `vliw_mission/round2/agent_7_pipeline_overlap.md` (verdict: ~3-10% headroom).
**Verdict:** all three proposed angles (A) software prefetch thread, (B) `MADV_WILLNEED`, (C) huge-page faulting are **either already done or measurably 0%**. Async AllReduce overlap is also a dead lever because AllReduce is **6% of token time, not 95%** (JOURNAL 2026-04-24 PM).

---

## What is already in production (don't re-do)

1. **Per-block prefetch with multi-level distance.** `cpu_quant_gemv.h:386-399` issues `__builtin_prefetch(row + (bi+1)*144, 0, 3)` for L1 plus `bi+16` at L2 — Round 2 agent_3 finding (15-cycle vs 200-cycle DRAM gap). Already commit `51ddb13`.
2. **MADV_RANDOM + MADV_HUGEPAGE.** `gguf_loader.h:262-319`. Hugepages fall back when 1191 needed and only 256 available; pages still resident.
3. **Cross-layer TLB warm prefetch.** `gguf_model.h:2380-2396` — 64 KB × 7 weights × `_MM_HINT_T1`. Warms TLB, not data (data won't fit in 16 MB shared L3).
4. **Split AllGather collective.** `ddp.h:88-89` `all_gather_post` / `all_gather_wait` exists. **Used only in `PT_TP_GATHER=1` path**, which on Elbrus measured **4.5 tok/s vs 4.8 baseline** (commit `a4de6c4` bench, JOURNAL 2026-04-24). The overlap-prefetch windows at `gguf_model.h:4642-4704, 4803-4864` are **dead code in production** — they live behind `use_gather` which is off.

---

## Why each Round 3 angle fails

### (A) Dedicated prefetch thread (1 of 8 OMP)

**Math:** sacrifice 1/8 = 12.5% of compute. Profiler shows compute (gate_up + ffn_down + attn_phase + output_proj + attn_output) = 183 ms / 211 ms = **87% of token**. Lose 12.5% × 183 ms = **+22.9 ms/token** on compute. Best-case prefetch save = full elimination of cold-fetch overhead, which the agent_7 spec correctly puts at ~5-8 ms/token. **Net: -15 ms regression. Reject.**

The exception is when prefetch issues are *cheap and cooperative* — i.e. workers issuing `__builtin_prefetch` in their own loops without losing throughput. That's already done in `cpu_quant_gemv.h:386-399` and gives the ~20% (3.9→4.7) win documented in JOURNAL 2026-04-21 commit `5de3954`. There is no further headroom for cooperative-prefetch because the kernel is already issuing 5 prefetches per inner block (3 at bi+1 L1 × 2 rows + 1 at bi+16 L2 × 2 rows).

### (B) `madvise(MADV_WILLNEED)` per-layer

**Why it doesn't help:** weights are already resident (4B model = 2.4 GB on a 125 GB box, fully paged in after token 0). `WILLNEED` issues a syscall (~5 µs on E2K LCC syscall path), then schedules a kernel work-queue page-walk that touches PTEs, then returns. Cost: 36 layers × 7 weights × 5 µs = **1.3 ms/token** of pure syscall overhead. Benefit: zero — pages are already in DDR with TLB entries pre-warmed by the `_mm_prefetch` block. **Net: -1.3 ms regression.**

For the >RAM regime (AirLLM streaming, `project_ollama_killer.md`) `WILLNEED` would help, but qwen3:4b fits.

### (C) MADV_HUGEPAGE for fault-overlap

Already done at startup (`gguf_loader.h:319`). On w205p only **256 of 1191 needed** hugepages are reserved. Remaining 935 / 4 = 234 layers' worth of weights live in 4-KB pages. To enable more we'd need root + sysctl, **not a software change**. JOURNAL 2026-04-21 confirms PT_HUGETLB=1 produced **0% gain** (even regressed 4.8 → 4.6 tok/s). Reject.

### Async AllReduce overlap

Production path uses `all_reduce_inplace` (legacy K-slice AllReduce-sum), not gather. There is **no `all_reduce_post` / `all_reduce_wait` split-API**. Adding one is mechanically possible (~80 LoC mirror of `all_gather_shm_post`/`_wait`), but the savings are bounded:
- Profiler: `allreduce` = 11.7 ms/token total (5.4 ms `_ao` + 6.3 ms `_fdown`).
- Maximum achievable overlap: 50% (compute next layer's RMSNorm + QKV-prep while AR runs, but next op depends on AR output for `h_buf` residual add).
- Realistic gain after barrier-skew: 30% × 11.7 = **3.5 ms/token saved → 4.8 → 4.88 tok/s**.

The 11.3 ms `_fdown` swelled to 17.6 ms when fused-kernels were added (JOURNAL 2026-04-24, "Per-section profiler"). That **+5 ms is straggler skew, not collective work** — the reducer itself is microseconds. Hiding 5 ms of skew via overlap requires the overlapped compute to *also exit early* on the slow rank, which it can't (it's the same rank).

### Intra-layer overlap

Inside one layer, ops are serially dependent: `RMSNorm → QKV → attention → output_proj → AR → residual → RMSNorm → gate/up → SiLU·up → ffn_down → AR`. The only existing parallelism is:
- (a) **gate ⊥ up** GEMVs — already fused into `cpu_fused_rmsnorm_gate_up_gemv` (commit `78e69d0`).
- (b) **Q ⊥ K ⊥ V** GEMVs — already fused into `cpu_fused_rmsnorm_qkv_gemv` (commit `78e69d0`).
- (c) **per-head attention** — already `parallel_for` over `n_heads_local` at `gguf_model.h:4571`.

No further intra-layer slack exists for a single-token decode.

---

## The honest answer

**Modest uplift, not worth the complexity.** The plateau cause is bandwidth-bound replicated weights (575 MB/chip/token at 18% of peak DDR utilization, JOURNAL 2026-04-24 "Bandwidth-bound plateau"). Pipelining trades one form of memory traffic for another against the same DDR channel — it does not raise the ceiling.

**If we had to ship one change**, the smallest-risk option is to add `all_reduce_post` / `all_reduce_wait` mirroring `all_gather_post`/`_wait`, then move existing `__builtin_prefetch` blocks (currently only firing in dead `use_gather` branch) into the legacy AllReduce-path windows. Estimated yield: **+0.05-0.1 tok/s (1-2%)**. That is below the 4.7-4.8 noise floor measured in JOURNAL 2026-04-24 ("Plateau analysis"). Not recommended.

The real levers per JOURNAL 2026-04-24 are:
1. **Speculative decode** (3 tokens/draft × 30% accept = 12 tok/s) — Agent 2's domain.
2. **Weight sharding without replicate** (4× per-chip BW reduction) — major architectural rewrite.
3. **Smaller model** (0.6B already at 22.8 tok/s).

Pipelining sits below the noise floor of all three.

---

## Files reviewed (no edits, per task instructions)

- `torch/io/gguf_model.h:4408-4865` — `forward_decode_cpu_tp` layer loop
- `torch/io/cpu_quant_gemv.h:386-399, 564-575, 1834-1836, 2066-2071` — bi+1/bi+16 prefetch already in place
- `torch/distributed/ddp.{h,cpp}:88-89, 759-893` — split AllGather lives, no AllReduce split
- `torch/io/gguf_loader.h:262-319` — MADV_RANDOM + MADV_HUGEPAGE already applied
- `JOURNAL.md:2553-2879` — Round 2/3 prefetch outcome (no movement past 4.8)
