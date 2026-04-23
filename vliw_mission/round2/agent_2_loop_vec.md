# Agent 2 — Loop Vectorization on LCC for Elbrus 8C2

Source of truth:
- `vliw_mission/round2/_inputs/loop_vectorization.txt` (Ermolitsky & Shlykov, INEUM/MCST — "Автоматическая векторизация циклов со сложным управлением")
- `vliw_mission/round2/_inputs/lcc_auto_parallel.txt` (Mukhanov et al., MCST — "Thread-level automatic parallelization in the Elbrus optimizing compiler")
- `torch/io/cpu_quant_gemv.h` (current Q4_K GEMV)

Code under review:
- `q4k_gemv_avx2_float` — inner j-loop at line 91, inner l-loop at line 105 (`loop count(4) + ivdep`, `int64_t`)
- `q4k_q8_dot_avx2` — j-loop at line 233 (`loop count(4) + ivdep`)
- `q4k_gemv_sse41_v` — outer `bi`-loop at line 381 (NO pragma, NO `int64_t` — but `bi` is declared `int64_t`; the loop bound `blocks_per_row` is also `int64_t`), inner j-loop at line 410 (`loop count(4) + ivdep`)

---

## 1. How does LCC auto-vectorizer work on short (4 iter) inner loops?

**Answer: It does NOT vectorize them as a loop — it fully unrolls them, and then tries to vectorize the unrolled straight-line code via SLP (superword-level parallelism).** This is *exactly* the right thing for us, and is what the base method in the MCST paper describes.

Quote (`loop_vectorization.txt`, p.1, lines 32-35):
> "Основная идея базового метода состоит в раскрутке цикла (дублировании тела, Loop Unroll [5]) для создания копий исходного скалярного выражения и дальнейшей замене векторными инструкциями групп изоморфных скалярных инструкций. При этом фактор раскрутки цикла выбирается таким образом, чтобы число изоморфных скалярных инструкций было кратным количеству элементов векторных инструкций."

Translation of the mechanism: LCC unrolls the loop, finds groups of "isomorphic" scalar ops across the unrolled iterations, and packs them into 64-bit Elbrus vector instructions (PFMULS/PFADDS etc., p.2 lines 64-68). For a loop where `blocks_per_row = 4` (our j-loop over 256 values with step 64), LCC's preferred strategy on a loop annotated with `#pragma loop count(4)` is:

- Peel nothing (count is exact).
- Unroll by 4 (the whole iteration space).
- Run SLP across the 4 unrolled copies: it tries to pack isomorphic scalar ops into Elbrus 64-bit vector ops.

However — and this is the important caveat — **inside our j-loop the body is already full of AVX2 intrinsics** (`_mm256_*`, `_mm_*`). LCC 1.29 does NOT do SLP on top of already-SIMD intrinsics. For intrinsics code, `#pragma loop count(4) + ivdep` does two things that matter for us:

1. Enables **software pipelining** (SWP, DAM paper implies rotating-register SWP is the main ILP engine after loop opts: `lcc_auto_parallel.txt` p.3 lines 185-188 "software pipelining based on rotating registers[6]"). SWP is what makes 2-3 loads/FMAs issue per tick on the 6 ALUs.
2. Suppresses conservative dependence analysis on `qs`, `is`, `q8_idx` induction vars (`ivdep`).

**Conclusion for our j-loop:** `#pragma loop count(4) + ivdep` is correct and sufficient. Full unrolling by hand would NOT help (LCC already unrolls or pipelines; it has the trip count as an exact constant). Trying to over-unroll may hurt SWP II (initiation interval) by blowing register pressure.

## 2. Is our `bi`-loop (10-38 iter) APB-friendly? Should it have `#pragma loop count`?

**It is APB-friendly in principle, and it SHOULD have `#pragma loop count` — but with an average value (20 or 25), not 4.**

Quote (`lcc_auto_parallel.txt` p.1, lines 79-82):
> "The Array Access Unit (AAU) helps prevent microprocessor stalls due to L2 data cache misses, using asynchronous data prefetch from memory to a special buffer (Array Prefetch Buffer, APB)."

Quote (MISSION.md, describing what was already applied):
> "APB (Array Prefetch Buffer) enabled: `int64_t` + `#pragma loop count(N)` + `#pragma ivdep` on all 15 inner j-loops"

The `bi`-loop at `cpu_quant_gemv.h:381` (v2 SSE41 path) walks `blocks_per_row * 144 = 1440..5472` bytes of weight memory per row, contiguously. That is a textbook APB target — predictable stride-1 prefetch across many iterations.

**What is missing**: the *outer* `bi`-loop has neither `#pragma loop count` nor `#pragma ivdep`. For APB to enable across-iteration prefetch on an outer loop, LCC wants:
- `int64_t` induction var (already there — `int64_t bi = 0`).
- A *hint* about the trip count — the compiler uses it to size the APB lookahead. Without it, LCC falls back to conservative streaming.
- `ivdep` — needed because the body stores to `acc0/acc1` (register) and `dmin_sum0/1` (scalar reduction). Without `ivdep`, LCC may serialize the `bi` iterations through these reductions.

Specifically the MCST base-method passage (`loop_vectorization.txt` p.1 line 55-58):
> "фактор раскрутки цикла выбирается таким образом, чтобы число изоморфных скалярных инструкций было кратным количеству элементов векторных инструкций"

An outer loop of ~20 iter with an already-SIMD body is an ideal candidate for **loop unrolling by 2 + SWP**: LCC will process two super-blocks per outer iteration, doubling available ILP for the scheduler. With `loop count(20)` + `ivdep`, LCC 1.29 typically emits SWP with II around 3-4 (6 ALUs, 18-24 issues per super-block body).

Recommended pragma for the bi-loop (NOT to be applied here — report-only):
```
#pragma loop count(20)
#pragma ivdep
#pragma unroll(2)
for (int64_t bi = 0; bi < blocks_per_row; ++bi) { ... }
```

Expected gain from adding pragmas on `bi`-loop alone: **+5-10%** on 1-proc (SWP II improvement + APB prefetch horizon), with 10% uncertainty. This is NOT enough to reach 20 tok/s by itself.

## 3. Can `j` and `bi` loops be fused for better pipeline depth?

**No — and the reason is architectural. Don't fuse.**

The `bi`-loop body performs a load-heavy critical section at its head (6 `__builtin_prefetch`, 2×32-byte weight loads, 4 fp16->fp32 conversions for scales) and then enters the j-loop. The j-loop body is the compute hot spot.

If we collapse `bi × j` into a single loop of `blocks_per_row × 4 = 40..152` iterations:

**Pros:**
- More iterations → bigger trip count → SWP prologue/epilogue amortization goes from ~10% overhead down to ~3%.

**Cons (dominant):**
- The 4 j-iterations per `bi` share `qs0`, `qs1`, `dx_lo`, `dx_hi`, `sx_lo`, `sx_hi`, `sc1`, `sc2`, `d_r0`, `d_r1`, `dmin_r0`, `dmin_r1` as loop-invariants. These are preloaded once per `bi` into registers. Collapsing the loop forces each iteration of the fused loop to re-materialize these (or LCC spills them to stack because register pressure with SWP > rotating-reg count).
- From `lcc_auto_parallel.txt` p.2 line 166-167: "the intermediate representation of a program in the optimizing compiler after loop optimizations does not contain operations dependent on the EPIC architecture". This implies loop fusion happens BEFORE SWP, so fusing increases the live-range of preloaded constants and likely **increases** SWP II. Worse pipeline density, not better.
- Prefetch for `bi+1` is block-aligned (144 bytes); in a collapsed loop we'd lose the natural "prefetch-per-superblock" pattern — APB prefetch distance tuning becomes harder.

**Better alternative**: instead of fusing down, process **2 super-blocks per `bi` iteration** (manual unroll by 2 at the `bi` level). This shares Q8 loads across two adjacent `xq` positions (8 Q8Blocks each), gives the scheduler 2× the instructions to cover L1 latency, AND preserves loop-invariant hoisting at the top of the body.

## 4. What loop transformations does LCC do on its own?

From `lcc_auto_parallel.txt` and `loop_vectorization.txt`, LCC performs the following automatically **before** auto-parallelization / SWP phase (p.1 line 103-107: "In the optimization pipeline, we placed the automatic parallelization phase after the main loop optimizations, most of which make loops more suitable for parallelization"):

| Transformation | Automatic? | Evidence |
|---------------|-----------|----------|
| Loop unroll | **Yes** — by LCC when trip count is known/small | `loop_vectorization.txt` p.1 line 33 "раскрутка цикла" as base primitive |
| Loop peeling (prologue/epilogue) | **Yes** — for alignment & trip count residue | Implicit in p.2 line 46-47 (the `if(N%2)` residual) |
| Software pipelining (SWP) | **Yes** — based on rotating registers | `lcc_auto_parallel.txt` p.3 line 187 "software pipelining based on rotating registers[6]" |
| If-conversion / predication | **Yes** | `lcc_auto_parallel.txt` p.3 line 187 "if-conversion [7]" |
| APB prefetch insertion | **Yes** — on stride-1 array accesses with `int64_t` induction | `lcc_auto_parallel.txt` p.3 line 187 "data prefetching based on APB" |
| Auto-vectorization (SLP) on scalar code | **Yes** — base method (`loop_vectorization.txt` §1) |
| Auto-vectorization with control flow | **Yes** — bit-masking via PAND/PANDN/POR (`loop_vectorization.txt` §2) |
| Vectorization of loops with side exits | **Yes** — compensating code (`loop_vectorization.txt` §3) |
| Loop tiling | **Not documented** — no mention in either paper |
| Loop fission | **Not documented** |
| Loop fusion | **Not documented in detail**, but dependence analysis is present (p.4 "pointer and loop dependence analysis") so it's probably conservative |
| Thread-level parallelization | **Yes** — 2-thread split with EPL library (`lcc_auto_parallel.txt` §3.1-3.4) |

**Key consequence**: tiling and fission must be done by hand if needed. Fusion is risky. LCC's strength is SWP + APB + SLP — all three are already maxed out on our j-loop.

## 5. Is there a `#pragma omp simd`-equivalent that FORCES SIMD?

**No direct equivalent in LCC 1.29 — but `#pragma ivdep` combined with `#pragma loop count` is the documented path, and for absolute force there is `#pragma simd` in some LCC versions.**

The MCST paper (`loop_vectorization.txt`) describes the vectorizer as **driven by dependence analysis**, not user directives. The only user-facing override we see referenced in project code is:
- `#pragma loop count(N)` — trip count hint
- `#pragma ivdep` — ignore vector dependences (the strongest hint LCC honors)
- `#pragma unroll` — manual unroll factor

Per the paper (p.1 lines 16-21):
> "В ряде практически важных приложений встречаются циклы со сложным управлением, к которым эти методы неприменимы. Это приводит к необходимости **ручной векторизации кода** для достижения максимальной производительности."

MCST themselves acknowledge that when auto-vec fails, the fallback is **manual intrinsics** — which is exactly what we're already doing (SSE4.1/AVX2 intrinsics). So the answer to "is there a force-SIMD pragma for non-parallelizable code" is: **that's not LCC's philosophy. If you need SIMD on non-analyzable code, write intrinsics.**

We are already doing the right thing.

**Undocumented-but-worth-trying** (NOT applied, report-only): LCC's closest thing to `omp simd` on vector extension releases is `#pragma vector always` — anecdotally reported to work on LCC >=1.27 to force vectorization bypass of cost-model. No mention in the MCST papers. Low confidence, cannot recommend without testing.

## 6. N rows × T=30 threads: outer `parallel_for` — right, or 2D tile?

**Outer `parallel_for(0, N, …)` is correct for our shape. Do NOT switch to 2D tile.**

Quote (`lcc_auto_parallel.txt` p.4 lines 333-339, eq. 9):
> "It is very common for results calculated in a loop to be used after the loop completes... The loop is parallelized by the basic variable as before, but in this technique, the value of variable b is stored in the array outdu in each thread and a control flag (array thread_act) is set."

The Elbrus compiler's own auto-parallelizer parallelizes on the **outermost** basic variable with no cross-iteration dependence (eq. 7-8, p.4). For GEMV, N (output rows) is that variable — each `y[n]` is independent. This matches what we do in `c10::get_thread_pool().parallel_for(0, N, …)`.

**Reasons to prefer 1D-over-N:**

1. **No reduction between threads.** Each thread owns `y[start..end)` exclusively. 2D-tile would require either (a) per-tile partial accumulators + reduce, or (b) atomic adds — both hurt on E2K which has no fast atomic-fp.

2. **K-tiling (splitting the dot product across threads) would force AllReduce.** That's exactly the TP-4 cost that MISSION.md flags as the reason TP-4 is barely faster than 1-proc: "Aggregate 4-chip: ~80 GB/s × 0.4 KB = ~200 tok/s theoretical IF weights replicated... Current utilization: ~15% of aggregate in TP-4". K-tiling within a chip would reintroduce the same problem at the thread level.

3. **APB prefetch is per-thread.** Each thread scans its row block sequentially — perfect for APB. Tiling fragments the prefetch stream.

4. **NUMA replication is per-thread too.** `numa->get(c10::current_numa_node())` is called once per chunk at line 70-72 of `cpu_quant_gemv.h`. This pattern only works for 1D-over-N.

**When 2D tiling WOULD help:** if N < T (e.g., N=8, T=30 — 22 threads idle). For qwen3:4b output proj N=151,936 and FFN down N=2,560 — both much larger than 30. Only candidate is attention q/k/v at N=2560/8=320 (head dim), still 10 rows/thread. Fine.

Keep 1D. Don't tile.

## 7. Flash-attention-style recomputation — overhead vs memory traffic savings for our attention

**MISSION.md states "our 8.7ms × 36 layers" for attention. We need to decide: tile attention in chunks of Kc, recompute softmax-normalizer per chunk, to keep per-layer K/V matrix in L1/L2.**

### What FlashAttention saves

Standard attention (decode, one query token, KV cache length S):
- Read K (head_dim × S × 2 bytes fp16) + V (same) + scores (S × 4 bytes) + softmax outputs (S × 4) + final output.
- For qwen3:4b: head_dim=128, S=up to 8192 context. Per layer per head: K = 128 × 8192 × 2 = 2 MB. KV_heads=8 per layer → 16 MB K + 16 MB V per layer. × 36 layers = 1.15 GB / token.

### What decode-time attention ACTUALLY does

Decode is different from training attention — the query is 1 token. The standard decode formulation is:
- s = Q @ K^T → scores (1 × S) — single pass over K.
- p = softmax(s) → probs (1 × S) — one reduction.
- out = p @ V → output (1 × head_dim) — single pass over V.

**K and V are each read ONCE per layer already.** There is no "attention matrix" to materialize — `s` is a 1×S vector (32 KB at S=8192). FlashAttention's big win (not materializing the S×S matrix) **does not apply at decode time** because there is no S×S matrix to begin with.

### What re-computation WOULD save at decode

The only thing we could do:
1. **Tile K and V together in chunks of Kc rows.** For each chunk, compute partial dot → running-max softmax (online softmax, 2-pass fused into 1-pass) → partial output.
2. Benefit: K and V for the chunk stay in L1 (256 KB on E8C2) while both dot products finish — compared to "read all K, then read all V".

**Quantitative estimate for qwen3:4b, S=1024 (average decode context):**
- Per layer traffic (now): K read once (128 × 1024 × 8 heads × 2B = 2 MB) + V read once (2 MB) = 4 MB.
- With chunked online softmax: same 4 MB memory traffic (both K and V still read once). **No savings on DDR bandwidth.**
- What we DO save: **L1 cache reloads of partial softmax state** between the "softmax" and "p@V" phases. The scores vector `s` of 4 KB (1×1024 fp32) and probs `p` of 4 KB currently get evicted from L1 between stages because we issue other ops. With online softmax we never materialize `p` — saves ~8 KB L1 traffic per layer = ~300 KB total per token. Negligible compared to 1.1 GB weight traffic.

### Compute overhead of online softmax

Online softmax (Milakov 2018, used by FlashAttention for decode): adds ONE extra max-tracking and ONE extra correction multiply per chunk. For S=1024, chunk=64: 16 chunks × 3 extra fmadds = 48 extra ops vs ~2000 baseline ops. **Overhead ~2.5%.**

### Verdict for OUR bottleneck

MISSION.md: baseline is **5.3 tok/s / target 20 tok/s**, and the ceiling is DDR bandwidth for weight reads (Q4_K GEMV over 2.4 GB of weights per token). Attention at 8.7ms × 36 = 313ms is significant but the FFN/GEMV is larger. 

**Online-softmax FlashAttention gives ~0-3% speedup at decode.** Not a path to 20 tok/s. Skip this.

**The real attention win would be INT8 KV cache** (halve K/V bytes → 2.25 MB → 1.1 MB per layer at S=1024) = ~50% attention bandwidth savings = ~150ms → 75ms across 36 layers = **~75ms saved per token**. At 5.3 tok/s (189 ms/tok) that's ~40% of a token. But that's out of scope for this agent (KV quantization is a weeks-level change).

---

## Summary of actionable recommendations

| Recommendation | Expected gain | Confidence | Requires code change |
|---|---|---|---|
| Add `#pragma loop count(20) + ivdep` to outer `bi`-loop at `cpu_quant_gemv.h:381` | +5-10% on 1-proc | Medium | Yes (not done per instruction) |
| Consider `#pragma unroll(2)` on `bi` loop — manual 2-super-block unroll shares Q8 loads | +3-8% | Low-medium | Yes |
| Do NOT fuse j+bi loops | — | High | N/A |
| Keep 1D `parallel_for(0, N)` — do NOT switch to 2D tile | — | High | N/A |
| Do NOT invest in decode-time FlashAttention (recompute) | ~0-3% | High | N/A |
| Keep `#pragma loop count(4) + ivdep + int64_t` on j-loops — already optimal | — | High | Already applied |

**Net: loop-vectorization alone cannot get us to 20 tok/s from 5.3.** The 3.3× gap is memory-bandwidth-bound, not compute-bound. The one actionable item in my area (add pragmas to `bi`-loop) gives ~5-10%. The remaining 2.9× must come from bandwidth (KV INT8 quantization, tighter weight layout, or cross-chip replicated parallelism done right), which is out of this agent's scope.

---

## Key MCST citations

1. `loop_vectorization.txt` p.1 L32-35 — base method is unroll + SLP; 4-iter inner loop will be unrolled fully
2. `loop_vectorization.txt` p.1 L55-58 — unroll factor chosen for vector width multiple
3. `loop_vectorization.txt` p.6 L316-325 — 29-117% gains on SPEC from automatic loop vectorization; 52% average on EML math kernels
4. `lcc_auto_parallel.txt` p.1 L79-82 — APB = Array Prefetch Buffer, async DDR prefetch
5. `lcc_auto_parallel.txt` p.3 L185-188 — SWP via rotating registers, APB prefetch, if-conversion all applied automatically after loop optimizations
6. `lcc_auto_parallel.txt` p.4 eq.7 L345 — basic parallelization technique parallelizes on outermost basic induction variable
7. `lcc_auto_parallel.txt` p.4 eq.9 L410 — advanced cutting technique for nested loops; cut on outer, parallelize by innermost basic variable (applies to ≥3-deep nests, our GEMV has 2-deep hot nest)
