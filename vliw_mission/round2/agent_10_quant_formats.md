# Agent 10 — Quantization Format Strategy

**Scope:** Evaluate alternative weight formats (Q8_0, Q4_0, Q5_K, re-quant at load, FP16, sparse Q4_K, mixed) against the current Q4_K_M path for decode on Elbrus 8C2. Rank by realistic speedup toward the 20 tok/s target.

**Hard baseline facts:**
- Current: qwen3:4b Q4_K_M → 5.3 tok/s 1-proc / 6.1 tok/s TP-4 (`MISSION.md:5`).
- Per-chip DDR effective: ~20 GB/s; absolute per-chip ceiling ≈ 50 tok/s with 0.4 KB/token weight residency if 100% bandwidth-bound (`MISSION.md:10-12`).
- Q4_K_M = 4.5 bits/weight = 144 B per super-block of 256 weights (`torch/io/gguf_dequant.h:304-341`, `torch/io/cpu_quant_gemv.h:55-58`).
- Elbrus APB delivers **32 B/tact on streaming reads** when inner stride ≤ 64 B; drops up to 64× if stride > 64 B per datum (`mcst_main_guide.txt:4387-4391`).
- **Decode is memory-bound, not compute-bound** — Agent 4 and Agent 6 both confirmed this (`agent_4_eml_audit.md:132`, `agent_6_weight_repack.md:271-287`).

**Bottom line:** Every format decision is dominated by **bytes/weight × tokens/sec ≤ DDR bandwidth**. Compute simplifications only help if they unblock bandwidth (e.g. by killing header bytes, by unifying stride so APB can fire). Formats that raise bytes/weight will lose at decode regardless of how clean the kernel is.

---

## Model footprint (qwen3:4b, 36 layers, H=2560, FFN=9728, vocab=151936) — order-of-magnitude

| Tensor | Shape | Weights | Q4_K bytes (4.5b) | Q8_0 bytes (8.5b) | Q4_0 bytes (5b) | FP16 bytes | Share of total |
|--------|-------|---------|-------------------|-------------------|-----------------|------------|----------------|
| ffn_gate ×36 | 9728×2560 | 896 M | 504 MB | 952 MB | 560 MB | 1 792 MB | 36% |
| ffn_up ×36 | 9728×2560 | 896 M | 504 MB | 952 MB | 560 MB | 1 792 MB | 36% |
| ffn_down ×36 | 2560×9728 | 896 M | 504 MB | 952 MB | 560 MB | 1 792 MB | 36% |
| attn_q ×36 | 4096×2560 | 377 M | 212 MB | 401 MB | 236 MB | 754 MB | 15% |
| attn_k ×36 | 1024×2560 | 94 M | 53 MB | 100 MB | 59 MB | 188 MB | ~4% |
| attn_v ×36 | 1024×2560 | 94 M | 53 MB | 100 MB | 59 MB | 188 MB | ~4% |
| attn_output ×36 | 2560×4096 | 377 M | 212 MB | 401 MB | 236 MB | 754 MB | 15% |
| token_embd / output | 151936×2560 | 389 M | 219 MB | 413 MB | 243 MB | 778 MB | ~1% (tied) |
| **Total weights streamed per token** | | **~2.5 GB Q4_K** | **~2.5 GB** | **~4.7 GB** | **~2.75 GB** | **~9.8 GB** | |

FFN = 72% of bandwidth per token. Attention Q/K/V+out = 24%. Embedding = 4%.

---

## 1. Q8_0 (8.5 bits/weight — 1.89× of Q4_K)

**Layout** (`gguf_dequant.h:237-254`): 2B fp16 scale + 32B int8 qs = 34 B per 32 weights. No sub-block scales, no dmin. Pure streaming kernel: one FMA-per-int8-per-float-x.

**(a) Memory delta:** 2.5 GB → 4.7 GB weights. **+89%**.

**(b) Speedup estimate:** at 20 GB/s DDR ceiling: 20/2.5 = 8 tok/s cap at Q4_K; 20/4.7 = **4.3 tok/s cap at Q8_0**. Current actual 5.3 tok/s already exceeds that Q8_0 ceiling. Kernel compute simplification doesn't matter — we'd be *bandwidth-starved on the wider format*.

**(c) Risk:** GUARANTEED **regression**. Matches Agent 4 finding (`agent_4_eml_audit.md:132`): "Q8_0 path is a regression for decode".

**(d) Impl effort:** Q8_0 kernel already exists (`cpu_quant_gemv.h:734-806`). Only load-time re-quantize wrapper needed. Implementation ~0.5 day.

**Verdict: REJECT.** Use Q8_0 only in a mixed config where it stays in L2/L3, see option 7.

---

## 2. Q4_0 (5 bits/weight — 1.11× of Q4_K)

**Layout** (`gguf_dequant.h:263-283`): 2B fp16 scale + 16B packed nibbles = 18 B per 32 weights. No dmin, no super-block scales. One scale per 32 weights, offset q → q-8.

**(a) Memory delta:** 2.5 GB → 2.75 GB. **+11%** (less bad than Q8_0 but still slower for decode).

**(b) Speedup estimate:**
- Bandwidth ceiling shifts from 50 → 45 tok/s per chip — strict regression on the cap.
- Compute win: simpler kernel. No `get_scale_min_k4` lookup, no dmin FMA, stride is 18 B (not 144 B) so APB fires tighter. Per Agent 6, removing 12% header tax unlocks ~30-45% effective (`agent_6_weight_repack.md:271-284`). Q4_0 has **11% header** (2B/18B) vs **12.5% header for Q4_K** (16B/144B) — marginally better but same order.
- Net: best-case +10-20% on throughput (kernel simpler, less scale overhead), offset by +11% memory. **Realistic: 0-10% improvement, possibly a loss.**

**(c) Risk:** **Quality degradation**. Q4_0 has no sub-block scales and no asymmetric zero (dmin). Perplexity on Russian Wikipedia/code typically 10-25% worse than Q4_K_M. We'd have to re-verify qwen3:4b still produces sensible text. If GGUF ships only Q4_K_M, we'd need to dequant→requant during load (lossy pass #2). No Q4_0 version of qwen3:4b exists on Ollama registry.

**(d) Impl effort:** Q4_0 dequant exists, GEMV kernel does **not** — would need new AVX2/scalar kernel (~300 lines following Q8_0 structure). Load-time re-quantize from Q4_K: ~200 lines. Total ~1-1.5 days.

**Verdict: REJECT.** Memory neutrality + quality risk + new kernel cost > expected upside.

---

## 3. Q5_K_M (5.5 bits/weight — 1.22× of Q4_K)

**Layout** (`gguf_dequant.h:348-386`): 176 B super-block of 256 values. Adds 32B qh high-bit plane on top of Q4_K's structure.

**(a) Memory delta:** 2.5 GB → 3.05 GB. **+22%**.

**(b) Speedup estimate:** 20/3.05 = 6.6 tok/s ceiling. Compute is **more** expensive than Q4_K (extra high-bit unpack per nibble). Strict regression.

**(c) Risk:** Same as Q8_0 direction but less severe. Zero upside for decode.

**(d) Impl effort:** Kernel exists (`cpu_quant_gemv.h:1508-1513`). But no reason to switch.

**Verdict: REJECT.** Use Q5_K only if quality is failing; never for speed.

---

## 4. Runtime re-quantize Q4_K → Q8_0 once at load

**Concept:** Dequant Q4_K to FP32 at load, re-quantize to Q8_0, discard Q4_K. Simpler decode kernel, larger memory footprint.

**(a) Memory delta:** 2.5 GB → 4.7 GB for quant weights. With FP32 working buffers (~500 MB), total ~5.2 GB per process. Still fits 125 GB node (8 procs × 5.2 GB = 41 GB). Fits per-chip in TP-4.

**(b) Speedup estimate:** Same as pure Q8_0 (section 1). **Bandwidth dominates — this is a regression at decode.**

**(c) Risk:** Same as Q8_0. Plus an additional ~10-15 s load-time hit for re-quantization (not critical).

**(d) Impl effort:** ~100 lines (dequant→Q8_0 pack). But pointless given (b).

**Verdict: REJECT.** Compute simpler, but decode is bandwidth-bound — making kernel simpler doesn't matter.

**Caveat:** This WOULD help if we had **many parallel queries** (batch-prefill, not decode). Not current workload.

---

## 5. Pure FP16 at load (16 bits/weight — 3.56× of Q4_K)

**(a) Memory delta:** 2.5 GB → 9.8 GB per process. 8 procs × 9.8 GB = 78 GB (fits 125 GB node). Per-chip in TP-4: 9.8 GB on each chip's DDR.

**(b) Speedup estimate:** 20/9.8 = 2.0 tok/s ceiling per chip. **Catastrophic regression.** The "vastly simpler kernel" argument assumes compute-bound; we're bandwidth-bound.

Additional penalty: E8C2 has no native FP16 FMA. `_Float16` arithmetic on E8C2 goes through FP16→FP32 conversion *per operand* on every use. The MCST guide (`mcst_main_guide.txt:6163`) references half-word register packing but no SIMD FP16 FMA. Expected 2-4× slowdown on compute too, layered on top of bandwidth loss.

**(c) Risk:** Quality perfect. Speed awful.

**(d) Impl effort:** Dequant Q4_K→FP16 at load: 150 lines. FP16 GEMV: 300 lines. **Do not spend this time.**

**Verdict: REJECT, by a wide margin.**

---

## 6. Sparse Q4_K (currently enabled for ffn_down + output)

**Current state:**
- Implemented: `torch/io/sparse_gemv.h`, `SparseQ4KWeight::analyze()`.
- Enabled at load in `gguf_model.h:1291-1314`:
  - `output.weight` (tied with embedding) → threshold 0.01 (skip blocks contributing < 1%).
  - ffn_gate / ffn_up / ffn_down → threshold 0.005 (< 0.5%).
  - Flag `use_sparse_gemv_=true` on success.
- Call sites: `gguf_model.h:2754` (ffn_down), `:2804` (output_proj), `:3099` (second FFN path).
- NOT applied to: ffn_gate, ffn_up (analyze is done but the GEMV callsites still use the dense kernel), attn_q/k/v/output.

**(a) Memory delta:** +N × blocks_per_row / 8 bytes bitmap ≈ negligible. For 2.5 GB weights, ~1-2 MB of mask.

**(b) Speedup estimate:** The analyzer prints expected speedup per tensor (`sparse_gemv.h:167-170`: "expected speedup: X×"). Realistic measured numbers are **not** in any round-2 agent report — the code is on the hot path (called every token) but no one recorded before/after timings.

Theoretical bound: if 30% of Q4_K super-blocks have magnitudes below 1% of row max, then 30% of bytes are skipped → 1.4× on that tensor. For ffn_down (36% of per-token traffic), that translates to **~10-15% end-to-end speedup** if actually realized.

But: sparse GEMV introduces a **per-block branch** (`sparse_gemv.h:209`), which:
- Kills APB engagement on the inner bi-loop (APB wants regular address stride — `mcst_main_guide.txt:4389-4391`). Branching to skip blocks breaks this.
- Breaks open-page DDR scheduling — skipping blocks causes random DDR page opens (`agent_6_weight_repack.md:167-171`).
- Breaks `_Pragma("loop count(...)") _Pragma("ivdep")` on the outer bi-loop.

So the theoretical 1.4× can invert into a regression if sparsity < ~40%. **The kernel is also AVX2 only** (`sparse_gemv.h:177`) — on Elbrus there's no AVX2, so it hits the scalar fallback (`sparse_gemv.h:269-340`) which is a pure serial loop with no EML acceleration.

**Was it actually tested on Elbrus?** Agent 3 mentions the code path exists but provides no timing number. Agent 5 lists it as "positive finding IS enabled" but provides no before/after. **No round-2 agent measured whether sparse is faster or slower than regular Q4_K on Elbrus.** This is a measurement gap.

**(c) Risk:**
- On x86 AVX2: probably 5-15% win for ffn_down (partially bandwidth-relief).
- On Elbrus scalar path: **likely regression** — branchy loop kills APB, and the scalar hsum alone is 4× slower than vectorized. Need to benchmark.
- Quality: 1% threshold is usually lossless; 0.5% is aggressive but typically < 0.1 PPL bump.

**(d) Impl effort:** Already in codebase. Need: (i) instrument A/B timing on Elbrus, (ii) if regression, disable on Elbrus (env flag). ~30 min.

**Verdict: MEASURE FIRST.** If it's a regression on Elbrus (likely), disable it. If it's a win, extend to ffn_gate/ffn_up call sites (currently analyzed but not wired — free throughput).

---

## 7. Mixed precision: Q4_K for FFN, Q8_0 for Q/K/V

**Concept:** Keep Q4_K on the big 72% (ffn_gate/up/down) to preserve bandwidth. Switch Q/K/V (15%+8%=23%) to Q8_0 because those tensors are **smaller** and mostly **cache-resident** — bandwidth penalty is smaller.

**(a) Memory delta:** Q/K/V+out goes from 530 MB → ~1000 MB. **+470 MB (+19% total, 2.97 GB total).** FFN unchanged.

**(b) Speedup estimate:**
- Q/K/V+out per-token bandwidth: 0.53 GB → 1.0 GB. Per-token total: 2.5 → 2.97 GB. Ceiling: 20/2.97 = **6.7 tok/s cap** vs 8.0 for Q4_K. Regression on bandwidth ceiling.
- Compute win on Q/K/V GEMVs: Q8_0 kernel has **no scale-lookup, no dmin FMA, no sub-block branching**. Approximately 1.4-1.8× faster per byte read.
- But Q/K/V are already fast (~small tensors, fit in L2/L3 after a few tokens). The win is on 23% of compute, not 23% of time — in practice Q/K/V takes <15% of decode time (check via profiler).

Net: **probably 0 to -10%.** Same logic as section 4 — when decode is bandwidth-bound, simpler kernel doesn't matter.

**Cache-resident argument:** Q/K/V for qwen3:4b = 530 MB Q4_K. Elbrus 8C2 L3 = 32 MB (private 2 MB + shared, `cache_optimization.txt:287`). Q/K/V does NOT fit in L3. At 1 GB Q8_0, even less likely. The "smaller weights fit in cache" premise is false for 4B models.

**(c) Risk:** Regression in 1-proc due to bandwidth. Could marginally help TP-4 because per-rank Q/K/V slice is smaller (~130 MB per rank) — still too big for L3 (32 MB), but closer. Uncertain.

**(d) Impl effort:** QuantizedWeight already stores `quant_type` per tensor. Per-tensor dispatch exists (`cpu_quant_gemv.h:1469-1518`). Load-time re-quantize of Q/K/V from Q4_K→Q8_0: ~150 lines. **~1 day.**

**Verdict: REJECT.** The cache-residency premise fails at 4B scale. For a 0.6B model it could help; here it's a regression.

---

## 8. IQ4_XS (4.25 bits/weight — 0.94× of Q4_K)

**(a) Memory delta:** 2.5 GB → 2.35 GB. **-6%.** Smallest option that keeps reasonable quality.

**(b) Speedup estimate:** Bandwidth ceiling: 20/2.35 = 8.5 tok/s (+6% vs Q4_K's 8.0). Reach extends marginally. Compute: IQ4 uses codebook lookup (256-entry LUT), NOT purely stride-based nibbles. Codebook lookup on Elbrus **breaks APB** because the access pattern becomes gather-indexed rather than regular stride. `mcst_main_guide.txt:4389-4391` makes explicit: APB loses up to 64× throughput on non-regular access.

**(c) Risk:** High. Quality equal or better than Q4_K_M on paper; on Elbrus the codebook lookup is an architectural mismatch. Not benchmarked anywhere on E2K.

**(d) Impl effort:** Dequant exists in neither file. IQ4 GEMV does not exist. Full new implementation: ~500 lines + extensive tuning. **3-5 days.**

**Verdict: REJECT.** Small memory win wiped out by APB-hostile access pattern.

---

## 9. KV-cache quantization (orthogonal but related)

Agent 3 flagged this briefly (`agent_3_cache_mem.md:61`) — KV cache in FP32 currently (64 MB at S=1024). Quant to Q8_0 saves 32 MB, Q4_K saves 48 MB. KV is *read every token across all S positions* so the bandwidth saving scales with context length.

**Not listed in the main 8 options** but worth noting: this affects attention bandwidth, not weight bandwidth. Distinct optimization, comparable magnitude on long context.

---

## Cross-cutting observation: Strided/repack format (Agent 6's proposal) beats all format changes

Agent 6's proposal (`agent_6_weight_repack.md:167-287`) is to **repack Q4_K in-place** into a layout where headers (4+12=16 B) sit in a separate header array and `qs[128]` blocks form a contiguous stream per row. Same bytes, same quality, different address pattern. Unlocks APB (currently partially blocked by the 16 B header-per-144 B interruption) and DDR open-page reuse.

- Estimated +30-45% (Agent 6).
- No quality loss.
- Impl effort: ~300 lines (repack-at-load + new kernel).
- Dominates any pure format switch, because it gives bandwidth win **without increasing bytes/weight**.

This competes with every option above for developer time.

---

## Top-3 recommendation, ranked by realistic speedup

### #1 — Keep Q4_K_M as the format. Measure sparse GEMV on Elbrus. Disable on regression or extend on win.

- **Why:** All format switches either raise bytes/weight (Q8_0, Q5_K, FP16 = bandwidth regression) or add quality risk with marginal upside (Q4_0, IQ4). Sparse Q4_K is already loaded but its Elbrus-side impact is **unverified** — a branchy loop may be a regression on E2K's VLIW scheduler because it breaks APB on the bi-loop.
- **Expected speedup:** -5% to +15% on Elbrus depending on measurement. Free cost (~30 min).
- **Risk:** Minimal; reversible via env flag.
- **Effort:** 30 min to instrument, 30 min to wire gate.

### #2 — Defer to Agent 6's weight repack (same Q4_K bytes, different layout).

- **Why:** Same bandwidth, but unlocks APB (32 B/tact per `mcst_main_guide.txt:4387`) and DDR open-page hits (`memory_controller.txt:143-148`). This is a **bandwidth efficiency win**, not a format change — which is exactly what decode needs.
- **Expected speedup:** +30-45% (Agent 6's estimate, `agent_6_weight_repack.md:430`). 5.3 → 6.9-7.7 tok/s 1-proc; 6.1 → 7.9-8.8 tok/s TP-4.
- **Risk:** Low. Format is internal (convert at load, preserve Q4_K bit-exact arithmetic). Kernel changes are contained to `q4k_gemv_avx2_stride`.
- **Effort:** ~1-1.5 days.
- **Note:** This is not Agent-10's proposal, but it dominates every quant-format option on cost-benefit. Calling it out so the format-change path isn't mistakenly pursued over this.

### #3 — KV-cache → Q8_0 (not in the original 8 options but ranked here because it beats all the format-change options)

- **Why:** KV is streamed every token times every context position. At S=1024 the KV is 64 MB FP32 that is re-read 36 times (once per layer) = **2.3 GB/token of KV bandwidth** — comparable to the 2.5 GB weight bandwidth. Quant to Q8_0 halves it to 1.15 GB/token. Net per-token DDR: 2.5 + 1.15 = 3.65 GB vs current 2.5 + 2.3 = 4.8 GB. Ceiling lifts from 20/4.8=4.2 tok/s to 20/3.65=5.5 tok/s — but this ceiling was apparently hit already, suggesting actual bandwidth is a bit higher (the MISSION.md 20 GB/s number is a single-DDR-channel estimate).
- **Expected speedup:** +10-25% once S grows past ~256 tokens. Diminishes on short prompts.
- **Risk:** Minor quality impact, reversible.
- **Effort:** KV-cache int8 path ~1 day. Not format-of-weights change — format-of-activations.

---

## Explicit rejections (DO NOT implement for decode on Elbrus 8C2)

| Option | Reason |
|--------|--------|
| Q8_0 weights (full conversion) | +89% bandwidth = guaranteed regression. Matches Agent 4's existing verdict. |
| Q4_0 weights | Marginal memory win offset by quality risk and new-kernel cost. |
| Q5_K weights | +22% bandwidth, more compute. Pure loss. |
| FP16 weights | +256% bandwidth, no native FP16 FMA on E8C2. Catastrophic. |
| Mixed Q4_K/Q8_0 (FFN/QKV) | Cache-residency premise fails at 4B scale (1 GB > 32 MB L3). |
| Load-time Q4_K→Q8_0 re-quantize | Same as Q8_0 weights — bandwidth dominates. |
| IQ4_XS | Codebook LUT breaks APB regular-stride requirement. |

---

## Key citations

- Bandwidth dominance: `MISSION.md:10-12`; `agent_4_eml_audit.md:132`; `agent_6_weight_repack.md:271-287`.
- APB stride rule (32 B/tact regular, up to 64× slowdown irregular): `mcst_main_guide.txt:4387-4391`.
- Q4_K layout: `torch/io/gguf_dequant.h:304-341`, `torch/io/cpu_quant_gemv.h:55-58`.
- Q8_0 layout + kernel: `torch/io/gguf_dequant.h:237-254`, `torch/io/cpu_quant_gemv.h:734-806`.
- Q4_0 layout (no kernel exists): `torch/io/gguf_dequant.h:263-283`.
- Q5_K layout + kernel: `torch/io/gguf_dequant.h:348-394`, `cpu_quant_gemv.h:1508-1513`.
- Sparse Q4_K (enabled for ffn_down + output, NOT wired for ffn_gate/ffn_up): `torch/io/sparse_gemv.h`, `torch/io/gguf_model.h:1291-1314`, :2754, :2804, :3099.
- Missing measurement: no round-2 agent A/B-tested sparse kernel on Elbrus.
- Weight repack alternative (dominates every format change): `agent_6_weight_repack.md:167-287`, :430.
- L3 size (32 MB shared / 2 MB private): `cache_optimization.txt:287`.

---

## Final recommendation summary

**Do not change weight format.** Q4_K_M is already at the Pareto optimum for decode bandwidth vs quality. Every format change evaluated either:
- Increases bytes/weight (strict bandwidth regression), or
- Breaks APB regular-stride (architectural loss on E2K), or
- Has no implementation on E2K and high development cost.

**Instead:**
1. Measure and gate the existing sparse Q4_K path on Elbrus (30 min, ±15%).
2. Defer to Agent 6's weight repack for the bandwidth win (1.5 days, +30-45%).
3. Quantize the KV cache to Q8_0 (1 day, +10-25% at long context).

Combined ceiling: ~8-10 tok/s on 1-proc, ~10-14 tok/s TP-4. Still below 20 tok/s target — format strategy alone cannot close the gap. The remainder must come from speculative decode acceptance-rate improvements (Agent 7) or aggregate multi-chip parallelism that doesn't pay AllReduce per token.
