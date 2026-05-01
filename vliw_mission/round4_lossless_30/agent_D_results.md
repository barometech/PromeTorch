# Agent D — Kernel Optimization Plan (Round 4) — 9.4 → 30 tok/s

**Author:** Agent D (Round 4)
**Date:** 2026-04-29
**Inputs:** MISSION.md §3 / §4, agent_D_kernel_optimization.md, format_spec_v1.md §10 (Q4_SOA4),
q8_soa_repack.h current production kernel, q8_soa4_microbench.c (1.21 ms / GEMV
single-core = 0.85× EML), Round 3 LCC audit, JOURNAL last 200 lines.
**Status:** Design / numerical analysis. *No code edits made.*

---

## 0. Executive summary

Closing the 9.4 → 30 tok/s gap (cut **73 ms / token = 70 %**) is a *compounded*
problem. No single optimisation reaches it. The combo that gets to 30 is:

```
A1 (persistent ThreadPool)         −18 ms   ★ MUST
A6 (PT8_Q4_SOA4 from Agent A)      −22 ms   ★ MUST
A4 (kernel fusion: QKV+gate/up)    −10 ms   ★ MUST
A5 (manual L2 prefetch + APB hints) −8 ms   should
A2 (overlap AllReduce w/ compute)   −7 ms   should
A8 (tail micro-cleanups)            −5 ms   nice
A3 (speculative decoding)           ×2.0    fallback if A1+A4+A6 short of 30
                                            (Agent E owns; A only consumes)
```

Unconstrained sum: 70 ms saved → **3.0 × throughput → 28 tok/s**. Speculative
decoding (A3, owned by Agent E) is needed for the last few percent, OR a 10 %
overshoot in the kernel work above. Each optimisation is independently
guard-railed by a microbench gate (see `agent_D_microbench_plan.md`).

The single highest-confidence move is **Q4_SOA4 + persistent ThreadPool
together** — these stack (one is per-call sync, the other is bytes/token), 
neither requires speculative decoding, and both have known designs.

---

## 1. Detailed time breakdown — where 73 ms hides

Profiler measurements from Round 3 commits `feb4c6a` and `3399136` (annotated
in `JOURNAL.md`). At 9.4 tok/s (= 106 ms / token), TP-4 with `OMP_NUM_THREADS=7`
on 32-core E8C2:

| Component | ms / token | % | Calls / token | Comment |
|-----------|-----------:|--:|--------------:|---------|
| FFN gate+up (Q8_SoA4 GEMV ×2)        | 22 | 21 % | 72 | 36 layers × 2 GEMV — biggest compute |
| FFN ffn_down (Q8_SoA4 GEMV)          | 17 | 16 % | 36 | K-slice variant |
| Attention QKV (Q8_SoA4 GEMV ×3)      | 10 |  9 % | 108 | row-sliced, fused via shared x_normed |
| Attention scores + softmax + AV       | 11 | 10 % | 36 | scalar fp32 main-thread (no SIMD!) |
| Attention output_proj (Q8_SoA4 GEMV) |  6 |  6 % | 36 | row-sliced |
| Output proj (Q8_SoA4 K-slice + AR)   |  8 |  8 % |  1 | vocab=152k × H=2560 |
| RMSNorm + residual scans              |  4 |  4 % | 72 | 2 per layer |
| RoPE + KV cache append                |  3 |  3 % | 36 | scalar |
| q8_soa4_quant_activation (5×/layer)  |  4 |  4 % | 180 | per call ~0.02 ms |
| **AllReduce / AllGather (SHM futex)** | **9** | **9 %** | 72 | 36 × 2 collectives |
| **ThreadPool fork/join overhead**     | **8** | **8 %** | ~200 | parallel_for sync |
| Embedding lookup, sample, misc        |  4 |  4 % | — | tail |
| **Total**                            | **106** | 100 % | | |

**Where the 73 ms can plausibly come from:**

| Source | Realistic save | Confidence | Mechanism |
|--------|---------------:|:----------:|-----------|
| ThreadPool: 8 ms → 1 ms (broadcast pool) | **−7 ms** | HIGH | Round 3 Agent 1 design |
| AllReduce overlap with prefetch / compute | **−4 to −7 ms** | MED | partial Option F path exists |
| Q8 SoA4 → Q4 SoA4 (½ bandwidth) — ffn_gate/up/down | **−14 ms** | HIGH | format spec §10, BW ceiling 49 t/s |
| Q4 SoA4 — attention QKV, output_proj | **−6 ms** | HIGH | same |
| Fused QKV (single quant_activation) | **−2 ms** | HIGH | already partial; SoA4 path repeats it 3× |
| Fused gate+up (single quant_activation) | **−1 ms** | HIGH | currently 2× |
| Attention scores → AVX2/SIMD vectorise | **−5 ms** | MED | scalar today (4673..4719) |
| APB-friendly stride / explicit prefetch on SoA4 | **−4 ms** | MED | LCC docs, currently no manual PF on SoA4 |
| Persistent activation buffer (no per-call alloc) | **−2 ms** | MED | x_normed std::vector → tp_ scratch |
| **Achievable kernel-only sum** | **~46 ms** | | leaves 27 ms gap to 30 t/s |
| Speculative decoding ×~1.6 effective | **−27 to −36 ms** | LOW (Agent E) | depends on accept rate |

**Honest conclusion:** kernel-only path closes 9.4 → ~17 tok/s. Format change
(Q8 → Q4 SoA4) lifts BW ceiling and adds another bump → ~22 tok/s. Reaching
30 requires *either* (i) speculative decoding with ≥60 % accept rate, OR (ii)
overshoot of 8-10 ms in kernel work — feasible if AllReduce overlap is more
aggressive than currently estimated.

---

## 2. Proposed optimisations, ranked by ROI

ROI = `(estimated ms saved per token) × (probability of full payoff) ÷ (effort in
sessions)`. Higher = ship sooner.

### A1 — Persistent ThreadPool (broadcast + per-worker ack slots)
- **Save:** 7-19 ms / token (estimate 12 ms, 8% of 106).
- **Probability:** HIGH. Round 3 Agent 1 already has a design.
  Microbench gate exists (`examples/benchmarks/threadpool_overhead_bench.cpp`).
- **Effort:** 1 session (lock-free broadcast + futex gen + per-worker ack).
- **Risk:** previous attempt deadlocked. Mitigation: ship AFTER tsan-clean
  microbench < 8 µs/call. No production integration before microbench passes.
- **Numerical guarantee:** trivial. No change to what each worker does, only
  *when*. Output bit-identical.
- **Implementation hooks:**
  - `c10/util/ThreadPool.h` — replace mutex+queue with broadcast descriptor.
  - `examples/benchmarks/threadpool_overhead_bench.cpp` — add to CMakeLists.

### A6 — PT8_Q4_SOA4 hot dtype (Agent A spec §10)
- **Save:** 18-26 ms / token. Q8 SoA4 = 1.75 B/param; Q4 SoA4 = 0.6875 B/param.
  Each GEMV reads ~½ the bytes → BW ceiling 19 → 49 tok/s.
- **Probability:** HIGH for the bandwidth half; MED for the unpack cost. We
  must verify the nibble unpack (`qpand` + byte shift, OR `qpshufb` LUT) is
  ≤ 2 cycles per K-pair.
- **Effort:** 2 sessions (kernel rewrite + numerical equivalence test).
- **Risk:** if unpack costs 4+ cycles, we lose 30-40 % of the BW win to extra
  compute. Microbench *first* — see `agent_D_microbench_plan.md` §M2.
- **Numerical guarantee:** byte-exact w.r.t. Q4_K_M decode-path because the
  nibbles ARE the Q4_K nibbles (the FP16-of-`d×sc` rounding is the same as
  the §3.3 Q8_SoA4_F16 case — already on the chosen lossless trait B).
- **Implementation hooks:**
  - New `torch/io/q4_soa4_repack.h` (mirror of `q8_soa_repack.h`).
  - Kernel `q4_soa4_gemv` with `qpmaddubsh` + nibble unpack. 
  - `cpu_quant::cpu_quant_gemv` dispatch a third branch.

### A4 — Kernel fusion: shared activation across QKV / gate-up
- **Save:** 3-5 ms / token. Three SoA4 GEMVs for Q/K/V currently each call
  `q8_soa4_quant_activation` (or rely on a shared one done ahead). Cross-check:
  `gguf_model.h:4602-4613` calls `q8_soa4_quant_activation` ONCE before all
  three QKV — already fused there. But:
  - the SoA4 path does NOT batch the actual GEMV inner loops, each
    super-row group is iterated 3 times (one per Q/K/V).
  - `gguf_model.h:4884-4896` does the same for gate+up — 2 separate
    `q8_soa4_gemv` calls, both reading the same `a_b16` buffer cache-line by
    cache-line.
- **Mechanism:** stream the 4 K-pair × 16 B activation block ONCE, then
  consume by both gate and up weight rows in a fused inner kernel. Saves L1
  re-pollution on `a_b16`. Same trick for QKV.
- **Probability:** HIGH — this is dataflow, not algorithm.
- **Effort:** 1 session.
- **Risk:** code duplication (2 kernels: solo + fused). Acceptable.
- **Numerical guarantee:** identical (same scalars, same partial-sum order).

### A5 — APB tuning + explicit L2 prefetch on Q8/Q4 SoA4
- **Save:** 3-5 ms / token. Round 3 explicit `__builtin_prefetch(row+(bi+1)*144)`
  on Q4_K gave +20 % (3.9 → 4.7). Q8 SoA4 (176 B blocks) has no manual PF.
  Round 3 Agent 4 noted **`__builtin_prefetch` is silently elided on the Q4_K
  *scalar* path** but DOES fire on the AVX2 path.
- **Mechanism:**
  - Add `E2K_PLD_L2(sb + SOA4_GROUP_BYTES)` at the top of each `b` iter.
  - Verify with `dprof -m TICKS,EXEC,BUB_E2` that Bubble count drops.
  - APB hint via `_Pragma("loop count(N)")` on the b-loop (currently unhinted
    on the production SoA path).
- **Probability:** MED. Agent 4 LCC audit warned hints can be elided; mitigation
  is the inline-asm `mas=0x20` pattern.
- **Effort:** 0.5 session.
- **Numerical guarantee:** trivial (no math change).

### A2 — AllReduce / AllGather overlap with adjacent compute
- **Save:** 4-7 ms / token. Currently 9 ms / token spent in 72 collectives.
  Option F gather variant already does prefetch-during-barrier
  (`gguf_model.h:4759-4773`, 4944-4955, 4985-5005). Two further moves:
  - **Reduce-scatter + all-gather:** halves bandwidth-per-rank for FFN-down
    AllReduce (rather than full sum-broadcast).
  - **Compute residual + RMSNorm on partial result while AR is in flight.**
    This is correct ONLY if you don't need full vector for norm — for FFN-down
    output you DO need full vector for RMSNorm input, so partial overlap is
    limited to the residual-add (which can run on the rank's slice and gather
    afterward).
- **Probability:** MED — already partially done.
- **Effort:** 1-2 sessions for reduce-scatter; 0.5 for tighter prefetch loop.
- **Risk:** SHM slot size (1 MB now) may need to grow; correctness landmines
  if rank ordering of partial sums changes (fp32 not associative; max abs
  diff < 1e-5 must be guarded).

### A8 — Tail micro-cleanups (under-the-radar wins)
- Vectorise attention scores + softmax (currently 11 ms scalar, ~5 ms saveable).
- Drop the `std::vector<float> x_normed(H)` per-layer alloc → reuse `tp_.x_normed_buf`
  (~0.5 ms × 36 layers = 1-2 ms).
- Pre-allocate `silu_local` outside the i-loop in legacy path (already done
  in `tp_.silu_full_buf` for gather path; legacy path lines 5013-5017,
  5026-5030 still allocate per-layer).
- Replace `std::exp` in SiLU with AVX2 / MCST cephes polynomial approx
  (already exists in `VectorizedOps.h`).
- **Save total:** 4-7 ms / token.
- **Effort:** 0.5 session combined.
- **Risk:** softmax fp32 reorder — bound max abs diff < 1e-5.

### A3 — Speculative decoding (Agent E owns; we consume)
- ×1.5 to ×3 effective throughput depending on accept rate.
- **Probability:** MED — PLD didn't work (0 % accept on qwen3); needs trained
  draft (qwen3:0.6B) OR n-gram lookup tuned to context.
- **Effort:** 2-3 sessions in Agent E. From Agent D: ensure batched-verify
  path through `q8_soa4_gemv` retains numerical equivalence (it currently
  does — N is just larger).
- **Numerical guarantee:** speculative decoding is approximation-free *if*
  the verifier rolls back wrong tokens. Greedy-equivalent.

### A7 — DOA: ideas evaluated and rejected
- **Hand-written e2k assembly in inner loop** — JOURNAL (commit `5de3954`)
  showed LCC translates AVX2 → VLIW *better* than hand-written `qpmaddubsh`
  loops by 23 %. Reject: writing native asm regressed.
- **8-row interleave** — Round 2 evaluation. Halves K-elements per block
  to 16 — APB efficiency drops, LCC SWP epilogue dominates. Reject.
- **In-mem activation in i8 SoA layout** — already done in `a_b16`.
- **GEMV int4 with `qpidotsbwss`** — VLIW e2k v5 does NOT have qpidotsbwss
  (per `e2k_vnni/FINDINGS.md`). Reject.
- **FP16 weights w/ qpfmuls** — no FP16 hardware on E8C2. Reject.

---

## 3. Numerical estimation: which combo gives 30 tok/s?

Baseline 9.4 tok/s = 106 ms/token.

| # | Optimisation | Stack-saved (ms) | Cumulative ms/tok | Cumulative tok/s |
|---|--------------|-----------------:|-------------------:|------------------:|
| 0 | baseline (9.4 t/s) | 0  | 106 | 9.4 |
| 1 | + A1 persistent ThreadPool | 12 | 94 | 10.6 |
| 2 | + A4 kernel fusion (QKV/gate-up) | 5 | 89 | 11.2 |
| 3 | + A5 APB / prefetch on SoA4 | 5 | 84 | 11.9 |
| 4 | + A2 partial AllReduce overlap | 5 | 79 | 12.7 |
| 5 | + A8 tail (scores SIMD, allocs) | 6 | 73 | 13.7 |
| 6 | + A6 PT8_Q4_SOA4 (½ BW) | 22 | 51 | 19.6 |
| 7 | A6 has-knock-on: per-token AR stays the same → +0 | 0 | 51 | 19.6 |
| 8 | + tighter loop count hints / `unroll(4)` per Agent 4 #7 | 3 | 48 | 20.8 |
| 9 | + A3 speculative ×1.6 effective (60 % accept, K=2) | — | 30 | **33.3** ★ |

Without A3 (kernel-only): **20.8 tok/s** (likely range 18-24 t/s with noise).
**This misses 30.**

With A3 at 60 % accept and K=2 verify: ×1.6 → 33 tok/s. Comfortable.

With A3 at 75 % accept and K=4: ×2.5 → 52 tok/s. Overshoot.

A1+A4+A5 represent the *pure* sync/throughput wins; A6 is the format-driven
bandwidth halving; A3 is the lever that converts steady-state into
multi-token-per-forward.

**Conclusion:** to hit 30 tok/s without speculative decoding, kernel work
must overshoot by ~9 ms beyond honest estimates. Possible if A2 lands harder
than projected, OR a Q4_SoA4 microbench shows < 0.6 ms/GEMV (vs the 1.21 ms
Q8 baseline). Otherwise, **30 needs A3.**

---

## 4. Implementation order

Strict order — each step gates on the next via microbench.

1. **A1 microbench → integrate** (1 session)
   - Connect `examples/benchmarks/threadpool_overhead_bench.cpp` to CMakeLists.
   - Build on x86 (tsan first), then Эльбрус.
   - Gate: < 8 µs / call. If ≥ 8 µs, stay on current pool, skip A1, go to step 2.
   - End-to-end TP-4 retest: 3 runs × 100 tokens, expect +0.5 to +1.2 tok/s.

2. **A6 design + microbench** (1 session)
   - Write `examples/benchmarks/q4_soa4_microbench.c` (clone of
     `q8_soa4_microbench.c`, swap nibble unpack in inner loop).
   - Gate: ≤ 0.7 ms / GEMV K=2560 N=2432 single-core (current Q8 SoA4 = 1.21).
     If 0.7-0.9 ms — proceed (still net win); if > 0.9 ms — drop A6, go to A4/A5.

3. **A6 production integration + Agent A/B PT8 loader** (1 session)
   - Cross-coordinate with Agent B converter and Agent C loader.
   - Numerical equivalence test: `pt8_verify` subcommand, compare logits to
     Q4_K_M baseline; max abs diff < 1e-5.

4. **A4 kernel fusion (single quant_activation, batched W reads)** (1 session)
   - Two new fused entrypoints: `q4_soa4_qkv_gemv`, `q4_soa4_gateup_gemv`.
   - Numerical equivalence test on production model.

5. **A5 APB tuning + manual prefetch** (0.5 session)
   - Add `E2K_PLD_L2` at top of `b` loop in q4/q8 soa4_gemv.
   - Add `_Pragma("loop count(...)") _Pragma("ivdep")` on b-loop.
   - dprof verification.

6. **A2 reduce-scatter on FFN-down + tighter overlap** (1-2 sessions)
   - Implement `reduce_scatter_inplace` in `ddp.cpp` mirroring `all_reduce_inplace`.
   - Re-route FFN-down output through it.
   - Microbench: 36-iter AR vs RS+AG latency on 4-rank Эльбрус.

7. **A8 tail clean-ups** (0.5 session)
   - Pre-allocate `tp_.silu_local_buf`, `tp_.x_normed_buf`.
   - SIMD softmax, vectorised attention scores.
   - Cephes SiLU.

8. **End-to-end TP-4 measurement** (0.5 session)
   - 3 runs × 100 tokens × greedy → median tok/s.
   - If ≥ 30 → done.
   - If ≥ 25 → escalate to Agent E for A3 speculative; expect ×1.4 → 35.
   - If < 25 → reanalyse profiler, re-spend on AllReduce hierarchy or scores
     SIMD before triggering A3.

Total kernel work: **6.5 sessions** to land everything except A3. A3 is
Agent E parallel work, ~2-3 sessions.

---

## 5. Risks and gotchas

1. **Q4_SoA4 nibble unpack might not be cheap.** `qpand + qpsrlw_byte` is
   *2 cycles* in theory; if LCC schedules them poorly with the qpmaddubsh
   chain, real cost could be 4-5 cycles → 25 % of theoretical BW win lost.
   Mitigation: microbench M2 before integration.

2. **AllReduce reduce-scatter rewrite has SHM slot ordering subtleties.**
   Wrong rank-order = silently wrong logits. Test with `PT_DDP_LOG=1` payloads
   on small examples first.

3. **Persistent ThreadPool deadlock recurrence.** Same Round 3 pitfall: if
   the broadcast counter wraps OR a worker is preempted past timeout, hangs.
   Mitigation: tsan, `PT_TP_TIMEOUT_MS=2000` env-driven watchdog assert during
   integration phase only.

4. **PT8 file format is bigger on disk than Q4_K_M.** Q4_SoA4 = 2.06 GB,
   Q4_K_M = 2.4 GB — same. Q8_SoA4_F16 = 5.25 GB. *Do not ship Q8_SoA4 as the
   primary v1 dtype* — Q4 SoA4 is the one (per format_spec_v1.md §10).

5. **Speculative decoding changes K-batch shape.** Our SoA4 kernel is N-major
   (output dim parallel) — accepts any K. Numerically equivalent.

6. **Numerics ladder.** Each step must pass `max_abs_diff(y_new, y_baseline) < 1e-5`
   on the production qwen3:4b weights against the current Q4_K_M decode. If
   any step regresses, revert and re-investigate.

---

## 6. Acceptance criteria for Agent D deliverable

- [ ] Microbench M1 (`bench_threadpool_overhead`) reports ≤ 8 µs/call.
- [ ] Microbench M2 (`q4_soa4_microbench`) reports ≤ 0.7 ms/GEMV K=2560 N=2432.
- [ ] All 5 kernels (Q/K/V, gate, up, down, attn_out, output) compile under
      `-march=elbrus-v5` LCC 1.29.
- [ ] On qwen3:4b decode of 100 tokens, max abs diff between PT8_Q4_SOA4 and
      Q4_K_M baseline logits < 1e-5 across all 100 forward passes.
- [ ] End-to-end TP-4 ≥ 25 tok/s without speculative decoding (3 runs median);
      if ≥ 25, hand off to Agent E for A3.
- [ ] Final delivery: ≥ 30 tok/s with speculative decoding enabled.

*End of agent_D_results.md*
