# Agent 9 — Multi-chip NUMA Aggregate Bandwidth Analysis

**Date:** 2026-04-22
**Scope:** Why does TP-4 on E8C2 (4 chips × 8 cores × 4 NUMA) give only 1.15× vs 1-proc
(6.1 vs 5.3 tok/s) instead of the 4× implied by aggregate DDR (80 GB/s)? Can we recover
the missing 2.9× without new hardware? Concrete option tree from cheapest → biggest.

**Sources read:**
- `vliw_mission/round2/MISSION.md` (baseline numbers, ceilings)
- `vliw_mission/round2/_inputs/hpc_architecture.txt` (MCST HPC tooling paper)
- `vliw_mission/round2/_inputs/memory_controller.txt` (Elbrus-16SV memory controller — DDR4-3200 at 25.6 GB/s/channel peak)
- `vliw_mission/round2/_inputs/cache_optimization.txt` (distributed L3 NUCA, home-bank pinning, victim-migration)
- `vliw_mission/round2/agent_2_loop_vec.md`, `agent_3_cache_mem.md`, `agent_4_eml_audit.md`
- `torch/distributed/ddp.cpp:500-600` (SHM AllReduce — bounded-spin, two-stage, 256 KB slot)
- `torch/io/gguf_model.h:4220-4580` (`forward_decode_cpu_tp` and AllReduce call sites)
- `torch/io/gguf_model.h:551-700` (TP column/row partition scheme)
- `torch/io/gguf_model.h:3964-4180` (`init_tensor_parallel`)
- `torch/distributed/pipeline_schedule.h` (existing 1F1B pipeline infra for training — NOT wired for inference)

---

## 1. Where do the missing 85% of parallelism go in TP-4?

Measured: `1-proc = 5.3 tok/s` → `TP-4 = 6.1 tok/s`. Speedup = 1.15×. Missing 2.85× relative
to the naive 4× target. Decomposition of one token @ 6.1 tok/s ≈ 164 ms/token:

| Bucket | TP-4 ms/token | Source |
|---|---|---|
| AllReduce (72 calls × ~375 µs) | **27 ms** | MISSION ("27ms/token = 13.6%"), ddp.cpp:517 SHM path |
| GEMV over local weight slice (Q/K/V/gate/up row-sliced + ffn_down/attn_out col-sliced) | ~90 ms | 5.3 tok/s ⇒ 189 ms/tok for full weights per chip; TP-4 gives each chip ~1/2 of bytes (see §1.3 below) |
| Embedding + RMSNorm + RoPE + softmax + attention math | ~10 ms | compute-bound tails (small and replicated) |
| Output proj (151,936-row vocab, AllReduce-summed) | ~15 ms | agent_3 estimate; full vocab replicated weight with zero-padded input on each rank, then AllReduce |
| Idle time waiting inside AllReduce spin + yield (not in `allreduce_ms` bucket) | **~15-25 ms** | see §1.2 |
| Inter-chip L3/home-bank miss tax on shared input/scratch buffers | **~5-10 ms** | see §1.4 |
| Sum | ~164 ms ✓ | |

### 1.1 AllReduce is ONLY 16% of wall-clock — not the dominant cost

The MISSION framing "AllReduce 13.6%, where do the other 72% go?" is correct: simply
eliminating AllReduce would take us from 6.1 → ~7.1 tok/s, nowhere near 20. So the
search for 20 tok/s is NOT an AllReduce problem. It is a **bytes-per-token problem
weighted by which-chip-owns-which-bytes**.

### 1.2 Hidden cost: AllReduce synchronizes ALSO the slowest chip's last GEMV

The bounded-spin in `ddp.cpp:543-555` means every rank that finishes its local GEMV
early BURNS CORES spinning until the straggler arrives. With 72 AllReduces × a few
hundred µs of straggler variance (DDR bank conflicts, L3 NUCA victim-migration lookups,
OS jitter on NUMA) you accumulate **15-25 ms of "phantom" idle time per token** that
does not appear in `allreduce_ms` but does appear as reduced effective compute
throughput. This is the Amdahl tax of micro-barrier granularity.

Evidence: `cache_optimization.txt` p.2 lines 85-113 (NUCA home-bank pinning) — a cache
line's `home`-bank is address-determined, so two ranks running identical GEMV kernels
over different weight slices WILL still touch different remote L3 banks depending on
OS first-touch. Timing skew per AllReduce is real.

### 1.3 TP-4 does NOT reduce per-chip bytes by 4×, only by ~2×

This is the load-bearing observation. From `gguf_model.h:499-502`:
- Q/K/V/gate/up: **row-sliced** across ranks (each rank owns 1/N rows × full K).
  Per-chip bytes on these = 1/N of total. ✓
- attn_output / ffn_down / output_weight: **REPLICATED** on every rank
  (see comment line 500 "REPLICATED (full weight on every rank)").
  Per-chip bytes on these = 1.0× total, NOT 1/N. ✗

Weight breakdown for qwen3:4b Q4_K (~2.4 GB, agent_3 numbers):
| Group | Fraction of total | TP-4 per-chip reduction |
|---|---|---|
| Q/K/V (row-sliced) | ~22% | 4× |
| gate/up (row-sliced) | ~30% | 4× |
| attn_output (replicated) | ~11% | 1× |
| ffn_down (replicated) | ~19% | 1× |
| output_weight 151936×H (replicated) | ~12% | 1× |
| embedding + norms + other | ~6% | 1× |

Per-chip bytes-per-token under TP-4 ≈ 0.52 × 1/4 + 0.42 × 1 + 0.06 × 1 = **0.61× of total**.

Per-chip bandwidth saving = 1 / 0.61 = **1.64×**. Combined with ~8-15% SHM AllReduce
+ sync overhead you get the measured 1.15×. **The math checks out.** The design
literally can't deliver 4× because 42% of the weight bytes are REPLICATED, not sliced.

**This is the fix target in §5.** Replicating replicated weights further doesn't help
(they're already on every chip). Slicing them does — but ffn_down row-slice forces an
AllReduce, which is exactly what was avoided by replicating it in the first place.

### 1.4 Cross-chip L3/NUCA tax on the ~70 MB of "replicated" activation scratch

`forward_decode_cpu_tp` uses `tp_.x_buf`, `h_buf`, `attn_full_buf`, `silu_full_buf`,
`logits_buf`. These are first-touch-allocated on whichever NUMA node the TP init ran
on. In TP-4, ranks 1-3 read/write them cross-chip every single layer.

`cache_optimization.txt` p.3-4 describes victim-migration and cooperative caching,
but the **Elbrus-8C2 has none of these** (they are future work in Elbrus-16SV). E8C2
uses inclusive S-NUCA: home-bank is purely address-determined. So a scratch buffer
allocated on rank-0's chip generates 100% remote-home-bank traffic for ranks 1-3.

Even at a small 1.2 MB per token of scratch × 3 ranks × 36 layers ≈ 130 MB of
cross-chip L3 transactions per token. At ~5 GB/s sustained cross-chip L3 (optimistic
E8C2 mesh) = 26 ms. We conservatively estimated 5-10 ms above because most of this
is hidden behind the GEMV streaming.

---

## 2. Speculative decode K=4 and data-parallel verification

Spec-decode is the **highest-leverage** cross-chip option because it turns the
bandwidth ceiling sideways: each of 4 chips processes a DIFFERENT token of the
K-token window in parallel, using its own full weight set (already replicated under
`PT_NUMA_REPLICATE=1` per `gguf_model.h:1548`).

### 2.1 Why this works when TP-4 doesn't

The problem with TP is that layer-sync (AllReduce at attn_out and ffn_down) forces
all 4 chips to traverse the same layer at the same time, serializing the decode.
Spec-verify has **zero inter-layer AllReduce**: each chip runs the full model on its
own token independently. Only a single AllReduce at the very end to gather logits for
the reject-sampling comparison — and even that is avoidable (just do argmax locally
and compare tokens).

### 2.2 Math for K=4 batch-spec-verify

Decoding effort for verifying K tokens in parallel:
- 4 chips × 1 token each × 1 full forward = 4 tokens verified per single-forward time.
- At 5.3 tok/s single-chip single-token baseline → **4 verified tokens per 189 ms
  = 21.2 verified-tokens/s**.
- Acceptance rate `α` converts verified-tokens → generated-tokens: `effective_tps =
  21.2 × α`. Need `α > 0.94` to clear 20 tok/s.

**BUT MISSION already flagged 0% acceptance rate on qwen3:0.6b draft for qwen3:4b
free-form text.** This kills the 20-tok/s path via draft-model spec-decode unless
we change draft strategy.

### 2.3 Alternative draft strategies to raise α

From the existing scaffolding (`speculative_decode.h`, `speculative_verify.h`):

| Strategy | Expected α | Code changes | Comment |
|---|---|---|---|
| Larger draft (qwen3:1.7b) | 0.3-0.6 | swap path | Still too slow to outrun main at 4× speedup factor |
| **Self-speculation** (Medusa-style K-th token head) | 0.4-0.7 | 1-2 weeks | Reuses main model's hidden state — zero draft cost |
| **Prompt-lookup decoding** (copy from context) | 0.2-0.5 on code/repetitive text | ~300 lines | Free, but text-dependent |
| **Low-rank head already in code** (`LowRankOutputProj`) extended to K-token speculative argmax | 0.3 free tokens | 100 lines | gguf_model.h:1270 already rank-256 head; extend to propose K=2 |

**Cheapest path: prompt-lookup** — it's a string-match on the KV history, no second
model, no draft training. For user-provided generation tasks (chat, code) α=0.3-0.5
is realistic. With α=0.4 → effective tps = 21.2 × 0.4 + 5.3 × 0.6 = 8.5 + 3.2 = 11.7
tok/s. That's ~2× over baseline but still below 20.

For 20+ tok/s via DP-verify we need α ≥ 0.85 AND K ≥ 4 AND 4-way parallelism. This
combination realistically requires Medusa heads (multi-week effort).

### 2.4 Concrete recommendation for this option

Implement DP-verify wrapper around existing `forward_decode_cpu` (NOT
`forward_decode_cpu_tp` — we are explicitly turning TP OFF and running 4 independent
full models, one per chip). Each chip is pinned via `numactl --cpunodebind=N
--membind=N` and loads a full replica of weights (already supported — see
`replicate_weights_for_numa` at `gguf_model.h:1548`). Then:
- Rank 0 produces K=4 draft tokens via prompt-lookup (cheap).
- Ranks 0-3 each run one forward on a different position of the K-token window.
- Single broadcast of accept mask → all ranks commit prefix → next round.

**Estimated gain: 2-4× IF α > 0.6 achievable via prompt-lookup on the target workload.**
Implementation cost: **~500 lines C++**, no kernel changes, reuses existing replicated
weights. **Highest cost-benefit ratio of any option here.**

---

## 3. Layer/pipeline parallelism: 36 layers / 4 chips = 9 layers/chip

The pipeline parallelism option treats decode as a conveyor: chip 0 owns layers 0-8,
chip 1 owns 9-17, chip 2 owns 18-26, chip 3 owns 27-35. The single token walks down
the pipeline.

### 3.1 Bytes-per-token math

Per token, each chip reads **1/4 of the weight bytes** (its 9 layers only). That IS
a true 4× per-chip bandwidth reduction — better than TP-4's 1.64×. Good.

### 3.2 Latency vs. throughput

For **single-token decode** (the actual user experience in a chat session), a token
still traverses 4 chips serially. The hidden state handoff between chips is ~H×4 bytes
= 2 KB for qwen3:4b (H=2048, fp32). At ~2 GB/s cross-chip via SHM = 1 µs each handoff
× 3 handoffs = 3 µs. Negligible.

Per-token latency:
- 4 chips × (local-bandwidth 9 layers) = same total bandwidth as 1-chip doing 36 layers
  BUT each chip runs at its full local 20 GB/s (it only owns 1/4 of the bytes → its
  layers' GEMVs complete in 1/4 the time).
- Token latency ≈ 189 ms × (1/4) = **47 ms/token = 21 tok/s** ideal.

**This looks like it hits the target.** Let me honestly check the assumption.

### 3.3 Why 1-proc TODAY gets 5.3 tok/s not ~8 tok/s on a single chip

This is the key skeptical check. MISSION line 12: "Per-chip DDR effective: ~20 GB/s ×
0.4 KB/token-weights = ~50 tok/s absolute single-chip". Current 1-proc achieves 5.3
tok/s — i.e. ~10% of the single-chip ceiling, not the "30% of per-chip" claim.

Why the gap between 5.3 tok/s and the theoretical 50 tok/s/chip:
- Weight-reads alone for qwen3:4b ≈ 2.4 GB/token. At 20 GB/s that's 120 ms = 8.3 tok/s.
  **Already way below 50.**
- So MISSION's "0.4 KB/token-weights" is the WRONG number for qwen3:4b — that's
  per-row, not total. Total is ~2.4 GB/token for the full model. Single-chip ceiling
  is ~8 tok/s, not 50. The 5.3 tok/s measured is **63% of realistic single-chip
  ceiling**, which matches the agent_3 DDR audit (77% of DDR peak sustained).
- **Critical implication:** even with perfect 4-chip pipeline, 4 chips × 8 tok/s
  per chip × 1/4 bytes per chip = **32 tok/s latency-equivalent** ceiling.

With pipeline handoff (3 µs) + imperfect load balance (layer sizes vary slightly) +
straggler sync, realistic pipeline ceiling is **18-26 tok/s**. This is the most
promising non-spec-decode path to 20.

### 3.4 Layer-balance concern

Qwen3:4b layers are near-uniform in weight bytes. Partition 36/4 = 9 layers each is
straightforward. The first chip additionally owns token_embedding (120 MB fp32) and
the last chip owns output_weight (618 MB fp32 at 151k vocab × 2048). To balance:
- Chip 0: embedding + layers 0-7 (8 layers + embed)
- Chip 1: layers 8-16 (9 layers)
- Chip 2: layers 17-26 (10 layers — embed+output are both F32 replicated, not Q4_K, so
  they don't fully count as equivalent bandwidth)
- Chip 3: layers 27-35 + output_weight + final_norm (9 layers + output)

Within 5% balance — acceptable.

### 3.5 Implementation cost

This is the **biggest code change** of any option:
- A new `forward_decode_cpu_pp(token_id)` that (a) if rank==0: reads embedding, runs
  its 9 layers, SHM-send hidden state to rank 1, SHM-recv logits from rank 3; (b) if
  rank 1-2: SHM-recv hidden, run 9 layers, SHM-send hidden; (c) if rank 3: run 9
  layers, apply output_weight, SHM-broadcast logits.
- ~800-1200 lines of C++, reusing `forward_decode_cpu` layer-body code.
- SHM handoff wire format (just ~2 KB, reuse existing SHM region with a new lane).
- Pipeline warm-up: only 1-3 tokens penalty (negligible for generation length > 10).

**Estimated effort: 1-2 weeks.** Estimated gain: 3-4× (5.3 → 16-21 tok/s). Risk: NUCA
cross-chip hidden-state handoff latency if naive. Mitigation: use `/dev/shm` with
madvise(MADV_HUGEPAGE).

---

## 4. Row+col TP mix: reduce per-layer AR count from 2 → 1

This is a **partial optimization** of TP-4, not a replacement. The current TP split
(`gguf_model.h:551-561`):
- Q/K/V: column-parallel (row-sliced weights) — no AR after QKV, just slice-local.
- attn_output: "row-parallel" but implemented as replicated+zero-padded+AR (see
  comment 500-502 "REPLICATED... AllReduce-SUM").
- gate/up: column-parallel — no AR.
- ffn_down: replicated+zero-padded+AR.

Total AllReduces per layer: 2 (attn_out, ffn_down). × 36 layers = **72 AR/token** ✓
(matches MISSION's "27ms/token = 72 calls").

### 4.1 Can we eliminate either AR?

**Yes, if we do true row-parallel on ffn_down (split K-dimension, not N-dimension)**:
- Each rank holds `ffn_down[:, rank*inter/N : (rank+1)*inter/N]` — 1/N bytes, much
  better than replicated 1.0×.
- Each rank computes its partial output from its K-slice of silu(gate)*up.
- Result is a partial N-vector that must be **AllReduce-summed** to get the full
  hidden output. So... still one AR.

The ONLY way to drop an AR is: **if attn_output is column-parallel** (row-sliced
weights), its OUTPUT is already a slice of the hidden dim. If the FOLLOWING op
(ffn RMSNorm + gate/up) can work on a slice, we skip the AR. But RMSNorm requires
the FULL hidden vector (it reduces across all H). So AR is unavoidable before
RMSNorm. And there are two RMSNorms per layer (attn, ffn) → two unavoidable ARs.

**Verdict for row+col mix: no structural reduction in AR count.** The 2-AR-per-layer
floor is a consequence of 2 RMSNorms per layer, not of the split strategy. The only
possible gain is switching ffn_down from replicated to K-sliced (saves 19% × total
weight bytes = ~460 MB per token of DDR traffic per chip = ~25 ms saved per token at
20 GB/s).

**Estimated gain from making ffn_down K-sliced instead of replicated: +15-25%** (5.3 →
6.5-7.5 tok/s in 1-proc-equivalent, or 6.1 → 7.5-9.0 in TP-4). **Implementation cost:
~150 lines** (modify init_tensor_parallel for ffn_down, modify forward_decode_cpu_tp
ffn_down call site, reuse existing AllReduce path). **Single biggest cheap win.**

### 4.2 Similarly for attn_output and output_weight

| Weight | Current | Switch to | Bytes saved/token | AR cost | Net |
|---|---|---|---|---|---|
| attn_output (11%) | replicated + AR | K-sliced + AR | ~11% × 2.4 GB = 264 MB/tok/chip | same | +13% tps |
| ffn_down (19%) | replicated + AR | K-sliced + AR | ~19% × 2.4 GB = 460 MB/tok/chip | same | +23% tps |
| output_weight (12%) | replicated (no AR in current code? check line 4567-4575) | K-sliced + AR | ~12% × 2.4 GB = 290 MB/tok/chip | +1 AR/tok (not per layer) | +14% tps |

Sum: **~50% improvement on TP-4 from switching 3 replicated weights to K-sliced.**
6.1 tok/s → **~9 tok/s**. Still short of 20, but this is the cheapest structural win.

---

## 5. Full replication + row-split output strategy

MISSION framing #5: "Replicate EVERYTHING on all 4 nodes via NumaReplica. Each rank
computes 1/4 of OUTPUT rows (row-split). Zero AllReduce for non-aggregating ops."

### 5.1 What already exists

`gguf_model.h:1548 replicate_weights_for_numa()` under `PT_NUMA_REPLICATE=1` already
replicates ffn_gate/up/down/attn_q/k/v/attn_output across N NUMA nodes. Cost:
10 GB × N = 40 GB for qwen3:4b on 4 nodes — fits in 125 GB/chip.

So the "replicate everything" precondition is already met. What's missing is the
**compute split strategy** that makes replication pay off.

### 5.2 The correct compute split: full row-parallel on every GEMV

With all weights replicated everywhere, the cheapest compute split is: **every GEMV
is row-parallel** (N-dim split). Each rank computes `y[my_rows]` from a full slice of
rows of W × full x. Because each rank's x is full (replicated from previous step),
no AR on INPUT. Output is partial (each rank has its own row range).

**Where does an AR appear?** Only when the NEXT op needs the full y (e.g., RMSNorm
across all H). Then we AR(y). Same 2 ARs per layer.

**What's NEW vs current TP-4?** Every op — including the currently replicated
attn_output/ffn_down — now reads 1/N of its rows instead of 1.0×. Per-chip
bytes-per-token collapses to truly 1/4 of total weight bytes.

### 5.3 Revised bandwidth math

Per-chip bytes/token = (1/4) × total = 600 MB for qwen3:4b.
Per-chip time = 600 MB / 20 GB/s = **30 ms/token = 33 tok/s** compute-bound.
AllReduce overhead = 27 ms (unchanged).
Wall time = 30 + 27 = 57 ms = **17.5 tok/s**.

With the SHM-write/compute-pipeline trick from §6 (reduce AR latency by ~40%), AR
drops to ~16 ms → wall time = 30 + 16 = 46 ms = **21.7 tok/s**. ✓ **Hits target.**

### 5.4 Implementation cost

This is a **rewrite of the TP init path**, not a patch. But most of the forward-pass
code stays (the row-parallel GEMV kernel already exists for Q/K/V and gate/up — we
just extend it to attn_output, ffn_down, output_weight). Estimated ~400 lines
modified + 200 new.

**This is the MEDIUM-cost option with the MOST certain delivery of 20 tok/s.** Risk:
NUMA replica memory pressure during load (40 GB peaks) — already handled by
`numa_weight_replica.h` per agent comments line 1534-1584.

### 5.5 Why THIS is better than the pipeline-parallel option of §3

- **Pipeline (§3) ceiling ≈ 32 tok/s, realistic 18-26.**
- **Full-replication row-parallel (§5) ceiling ≈ 33 tok/s, realistic 17-22.**
- Pipeline has higher single-token latency during first token (serial chip traversal).
- Full-replication has higher memory pressure (4× weight bytes).
- Full-replication reuses existing code paths (Q/K/V row-parallel already written).

**Choose §5 if 40 GB RAM fits (yes). Otherwise fall back to §3.** For qwen3:14b at
9.6 GB quant (MEMORY.md note) → 38 GB replicated → still fits in 125 GB. For 32B
models (25 GB quant → 100 GB replicated) hits the limit — must use pipeline.

---

## 6. SHM AllReduce bottleneck — compute/compute overlap?

### 6.1 Where does 375 µs/call come from?

For a typical per-layer AR payload (H=2048 fp32 = 8 KB):
- Step 1 — memcpy to own slot: 8 KB at 10 GB/s bandwidth = 0.8 µs. ✓
- Step 2 — atomic inc + store fence: ~10 ns. ✓
- Step 3a — rank 0 spin wait (best case all arrived already): ~100 ns to ~20 µs
  depending on straggler variance. **This is the randomness tax.**
- Step 3b — sum reduction: 8 KB × 3 ranks × add = 24 KB reads / 4 add-units-core /
  1.5 GHz ≈ 2 µs. ✓
- Step 4 — memcpy back: 0.8 µs × N. ✓
- Step 5 — barrier departure: 10 ns + spin for stragglers.

Best case minimal AR: ~5 µs. Worst case with straggler variance: 100 µs. **Median
375 µs reported in MISSION suggests straggler-bound, not compute-bound.**

### 6.2 Pipelining SHM write with next-op compute

Yes, this is achievable but delicate. Idea:
- After finishing op K (e.g., ffn_down GEMV), don't IMMEDIATELY call `all_reduce_inplace`.
- Instead: memcpy partial result → SHM slot + signal arrived (step 1-2), THEN start
  next op K+1 IF it doesn't need the reduced value.

**Problem**: in the current loop, op K+1 is usually RMSNorm, which NEEDS the reduced
value. So this only works if we can find a K+1 that's AR-independent. Looking at the
loop at `gguf_model.h:4267-4580`:

| Sequence | AR dependency |
|---|---|
| QKV GEMV | no AR (row-split output) |
| QK norm + RoPE | needs my-rank-slice only ✓ — **can run concurrent with any AR** |
| Attention score + softmax + V | needs my-rank-slice only ✓ — **concurrent with any AR** |
| attn_output GEMV (replicated/partial-sum) | - |
| **AR(attn_out)** | produces full hidden |
| FFN RMSNorm | needs full hidden — serial after AR |
| gate + up GEMV | after RMSNorm |
| SiLU(gate)*up | row-split ✓ |
| ffn_down GEMV (replicated/partial-sum) | - |
| **AR(ffn_down)** | produces full hidden |
| residual + next layer's attn RMSNorm | serial after AR |

**Overlap opportunity**: the local attention math (RoPE, scores, softmax, V-mul) is
all row-split and can overlap with the AR of the PREVIOUS layer's ffn_down AllReduce!
This is exactly Megatron-LM's "async AllReduce" trick.

To implement: split `all_reduce_inplace` into `all_reduce_begin`/`all_reduce_end`.
Begin = steps 1-2 (deposit + signal arrived). End = steps 3-5 (wait for result + copy
back). Between them, each rank runs its local attn math (doesn't need reduced hidden).

**Estimated overhead reduction: 30-50% of AR time hidden behind attn math.** With
27 ms AR → 13-18 ms effective. Saves ~10 ms/token. **On TP-4 = +6% tps (6.1 → 6.5).
On full-replication (§5) = the difference between 17.5 and 21.7 tok/s ⇒ CRITICAL.**

### 6.3 Implementation cost

~150 lines in ddp.cpp + ~80 lines of timing refactor in forward_decode_cpu_tp.
**Medium-cheap. Do this.**

### 6.4 Is the 375 µs SHM compute or sync?

Given the math in 6.1 showing 5 µs baseline compute vs 100 µs straggler variance,
**375 µs median is 95% sync overhead**. This is good news: it means better barrier
primitives (e.g., per-layer pre-allocated counters to avoid mod-16 generation cycling,
or futex-based fast-path with bounded spin < 500 ns) can knock it down to ~100 µs.
**Estimated: 27 ms → 7 ms AR savings (factor 4×).** Another ~10 ms/token saved,
completely additive to 6.2.

---

## 7. Option ranking — cheapest to biggest restructuring

Target: aggregate effective bandwidth 50-80 GB/s. Current ~12 GB/s (1-proc) or
~20 GB/s (TP-4 effective after replicated-weight tax).

| # | Option | Est. gain (from 6.1 tok/s) | LoC | Risk | Target hit? |
|---|---|---|---|---|---|
| **A** | `#pragma loop count + ivdep + unroll(2)` on `bi`-loop (agent_2 rec) | +5-10% → 6.5 tok/s | 3 | none | no |
| **B** | Async AllReduce split (overlap with attn math) | +6% TP, enables §5 | 230 | low | no alone |
| **C** | Tighter SHM barrier (drop 375 → 100 µs median) | +6% TP, enables §5 | 100 | low | no alone |
| **D** | Convert replicated weights (attn_out, ffn_down, output) to K-split | +50% → 9 tok/s | 150 | low | no |
| **E** | Prompt-lookup spec-decode (K=4, DP verify) | +60-120% IF α≥0.4 | 500 | text-dep | ~maybe |
| **F** | **Full replication + row-parallel every GEMV + §B + §C** | **+190-260% → 17-22 tok/s** | 600 | NUMA RAM | **YES** |
| G | Pipeline parallel (9 layers/chip) | +200-330% → 18-26 tok/s | 1200 | pipeline warmup | YES |
| H | Medusa-style self-speculation + DP-verify | +300-500% IF α≥0.8 | ~3000 + retrain | highest | YES but expensive |

### 7.1 Recommended roadmap (highest expected value order)

1. **Do D immediately** (150 LoC, +50%, zero risk). Converts 3 replicated weights to
   K-split. This is the **biggest-bang cheapest** structural change and is a
   prerequisite for F anyway.

2. **Then B + C in parallel** (async AR + tighter barrier, ~330 LoC combined, enables
   AR to cost 7 ms instead of 27 ms = ~12% absolute wall time).

3. **Then F** (full row-parallel on every GEMV with replicated weights — 600 LoC).
   Combined D+B+C+F gets to 17-22 tok/s.

4. **In parallel with F, spike E on a realistic prompt set.** If prompt-lookup gets
   α ≥ 0.5 for the target deployment (chat/code), combine DP-verify with F for a
   multiplicative 1.5-2× on top of F's 17-22 = 25-40 tok/s.

5. **Only if 1-4 together fall short**, invest in G (pipeline parallel, 1200 LoC)
   or H (Medusa, weeks).

### 7.2 What NOT to do

- **Don't try to eliminate all AR.** 2 AR/layer is a RMSNorm consequence, not fixable
  without fundamental architecture change (replacing LayerNorm with a chunked norm
  that works on partial vectors — a research problem, not engineering).
- **Don't invest more in TP-4 with replicated attn_out/ffn_down.** The current 6.1
  tok/s is ALREADY near-optimal for that particular split scheme; more tuning won't
  break the 1.64× per-chip-bandwidth wall.
- **Don't use existing `pipeline_schedule.h`** — it's written for training (1F1B),
  not inference. PP for inference needs a new forward-only path.
- **Don't chase INT8 KV cache** for this specific target — it's a 2-week project
  with +30% gain, strictly worse cost-benefit than D/F.

---

## 8. Required aggregate bandwidth to hit 20 tok/s — exact number

Decode is memory-bound. At 20 tok/s we need 2.4 GB / 50 ms = **48 GB/s aggregate
effective bandwidth**. Per-chip = 12 GB/s — well below the 20 GB/s single-chip
ceiling, so the question "can 4 chips deliver 48" is really **"can we arrange the
work so 4 chips together deliver 12 GB/s each of DISTINCT useful work"** — i.e., no
chip wastes cycles on cross-chip traffic or idle-spin.

The three ways to get there:
- **§5 (full replication row-parallel):** each chip touches 1/4 of weights → each
  chip needs 5 GB/s per-chip. Massive headroom (chip ceiling 20). ✓
- **§3 (pipeline):** each chip owns 1/4 of layers → same 5 GB/s per-chip. ✓
- **§2 spec-decode DP:** each chip runs a full forward on its own token → 2.4 GB per
  chip per token × 4 tokens in parallel = chip sees 20 GB/s × α useful. ✓ if α > 0.6.

All three are architecturally viable. Option F (§5) is the best cost-benefit.

---

## Summary

**Root cause of 1.15× TP-4 speedup:** 42% of weight bytes (attn_out, ffn_down,
output_weight) are REPLICATED across ranks, so per-chip bandwidth only drops to
0.61× of total (not 0.25×). Sync+barrier tax consumes another ~10% of wall time.

**Path to 20 tok/s requires doing ALL of:**
1. K-split the 3 replicated weights (+50%, 150 LoC) — Option D.
2. Async AllReduce overlap with attn math (+6% combined with F, 230 LoC) — Option B.
3. Tighten SHM barrier from 375 µs → 100 µs (+6% combined with F, 100 LoC) — Option C.
4. Full-replication row-parallel compute with all weights sliced at output-row level
   (+30-50%, 600 LoC) — Option F.

Combined projected: 5.3 → **17-22 tok/s**, hitting the target with ~1100 LoC of
structural-but-localized change. No kernel rewrites, no retraining, no new hardware.

**If the project must ship fast and can tolerate workload-dependence, add E (prompt-
lookup spec-decode, 500 LoC) as a multiplicative 1.3-2× layer on top.**

---

## Key citations

1. `MISSION.md` L13-16 — per-chip 20 GB/s, aggregate 80 GB/s, current 15% of aggregate
2. `gguf_model.h:499-502` — explicit comment confirming attn_output/ffn_down/output_weight
   are REPLICATED, not sliced
3. `gguf_model.h:1548 replicate_weights_for_numa()` — infrastructure for §5 already in place
4. `torch/distributed/ddp.cpp:517-597` — bounded-spin SHM AllReduce, 8 KB payload,
   straggler-bound not compute-bound
5. `gguf_model.h:4267-4580 forward_decode_cpu_tp` — 2 AR/layer (attn_output + ffn_down);
   output_proj has 1 extra AR/token
6. `cache_optimization.txt` p.2-3 — NUCA home-bank pinning, S-NUCA on E8C2 (no migration);
   replicated scratch → cross-chip L3 tax
7. `memory_controller.txt` p.3-5 — DDR4-3200 at 25.6 GB/s/channel peak at 800 MHz,
   effective ~20 GB/s; validates single-chip ceiling
8. `hpc_architecture.txt` p.7-8 — MCST programming guide for HPC shows multi-chip
   common pattern is MPI over OFED/UCX RDMA, NOT SHM — but we stay with SHM because
   single-node is an upper bound on latency
9. `agent_3_cache_mem.md` (read for context) — DDR bandwidth sustained 77% of peak at
   1-proc; scratch buffer first-touch is correct for 1-proc, cross-chip cost for TP
10. `agent_2_loop_vec.md` — loop vectorization alone cannot close the 2.9× gap; must
    be bandwidth-level restructuring
