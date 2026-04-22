# Agent 1 — E2K Architecture Audit of Q4_K GEMV kernel

Ref kernel: `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h`
Main Q4_K path: `q4k_gemv_avx2()` lines 311–473 (hot loop 350–457).
Docs consulted: `docs/elbrus/E8C2_ARCHITECTURE.md`, `LCC_OPTIMIZATION.md`, `EML_BLAS_GUIDE.md`,
`EML_THREADING_API.md`, `elbrus_prog_guide_v1.2.pdf`, `elbrus_loop_vectorization.pdf`,
`elbrus_memory_controller.pdf`, probe `examples/benchmarks/q4k_e2k_kernel_probe.cpp`.

---

## Q1 — LCC pragmas / hints

### Used in `cpu_quant_gemv.h`
| Where | Hint | Notes |
|---|---|---|
| lines 360–365 | `__builtin_prefetch(..., 0, 3)` | 6 prefetches per `bi` iteration, locality=3 (L1). Empirically +12% (comment line 357). |
| lines 645–648 | `__builtin_prefetch` in q6k path | 4 prefetches, same locality=3 |
| line 350 | outer loop uses `int64_t bi` | ОК for APB (per `LCC_OPTIMIZATION.md` §4.1) |
| line 384 | inner loop uses **`int j`** | BAD. Per `LCC_OPTIMIZATION.md` §4.1 APB/SWP only enable on `int64_t` counters. |
| line 337 | `parallel_for(..., grain=1)` | row-level parallelism, fine. |

### Missing LCC-specific pragmas (from `LCC_OPTIMIZATION.md` §3 + §4)

1. **`#pragma loop count(...)`** — absent everywhere. The outer block loop runs `K/256` iterations
   (for qwen3:4b K ∈ {2560, 9728} → 10 or 38 blocks). Doc says this hint *enables APB prefetch*
   (LCC_OPTIMIZATION.md lines 75, 126–132). Our `__builtin_prefetch` substitutes but does NOT
   enable the hardware APB.
2. **`#pragma ivdep`** — absent. Tells compiler read/write pointers in the loop body do not alias.
   Relevant on the `qs0 += 32; qs1 += 32; q8_idx += 2;` epilogue of the inner loop (line 454–456) —
   but more importantly on the accumulation to `sum0`/`sum1` (line 426, 451) which the compiler
   must otherwise treat as potentially aliasing `y[]`.
3. **`#pragma unroll(N)`** — absent. The inner `for (int j = 0; j < 256; j += 64)` loop is 4 iterations
   with the body ~80 lines of intrinsic calls. LCC can trivially unroll×4 to expose more ILP but
   only if told (see §4.5 "pролог/эпилог SWP дорогой — минимум 20–30 итераций"; 4 iter is below
   threshold, so SWP unlikely fires without explicit hint).
4. **`__restrict` / `__restrict__` on pointer parameters** — absent on `q4k_gemv_avx2` signature
   (line 311: `const void* weight_data`, `const float* x`, `float* y`). Doc cites **33–80% speedup**
   on E2K from restrict (LCC_OPTIMIZATION.md line 101). Without it LCC must assume `y[n]` may
   overlap with `x[]` or `weight_data`.
5. **`-frestrict-params`** compile flag — not verified here (out of scope: cannot run build), but
   mentioned as the global equivalent.

---

## Q2 — Independent arithmetic chains vs 6 ALC channels

E8C2 core has **6 ALC channels** (АЛК0–АЛК5), of which all 6 have FP/packed capability
(E8C2_ARCHITECTURE.md lines 26–31, 40: *"50 операций за такт, 48 FP32"*).
Peak = 48 FP32 FMA/cycle/core ⇒ needs ≥6 independent FMA dataflow chains to saturate.

### What the hot loop presents to LCC (lines 384–457)

Inside one `j` iteration of the inner loop the compiler sees **two disjoint chains** fed by
shared x_q8 data:

- Row 0 chain: `raw0 → q4_lo0 → p0_lo16 → p0_lo32 → t0 → is0_lo → sum0 +=` and
  `                               p0_hi16 → p0_hi32 → t1 → is0_hi → sum0 +=`
- Row 1 chain: symmetric, feeds `sum1`.

So there are roughly **2 independent halves per row × 2 rows = 4 parallel chains** of integer
maddubs/madd/shuffle, plus **2 scalar FP accumulators** (`sum0`, `sum1`) that depend on every
horizontal-sum result.

### The scalar accumulator serializes everything at the end

Lines 426–427, 451–452:
```cpp
sum0 += d_r0 * sc_a0 * dx_lo * (float)is0_lo - dmin_r0 * m_a0 * sx_lo;
sum0 += d_r0 * sc_b0 * dx_hi * (float)is0_hi - dmin_r0 * m_b0 * sx_hi;
...
sum1 += d_r1 * sc_a1 * dx_lo * (float)is1_lo - dmin_r1 * m_a1 * sx_lo;
sum1 += d_r1 * sc_b1 * dx_hi * (float)is1_hi - dmin_r1 * m_b1 * sx_hi;
```

Four back-to-back `+=` on each of `sum0`, `sum1` = **2 dependency chains of length 4 FP-adds**
per j-iteration. FP-add latency = 4 cycles (E8C2_ARCHITECTURE.md §8) → 16 cycles on the critical
path per j-iter for each row, plus the earlier integer reductions.

Per LCC_OPTIMIZATION.md §4.3 the canonical fix is multiple accumulators, e.g.
`s0, s1, s2, s3` combined at the end. We currently use 2 (one per row). **Compiler sees at most
2 FP accumulator chains** → at most 2 of 6 FP-capable channels packed on the accumulation
step.

### Horizontal sum is a 4-step serial reduction (×2 per row)

Lines 414–418 / 420–424 / 439–443 / 445–449 each contain three sequential `_mm_add_epi32` +
`_mm_shuffle_epi32` pairs. Each one depends on the previous. This is 3-deep per lane, and
LCC will translate them into 128-bit packed `PADDW`-style ops that fit 1 channel — no ILP gained.
If we kept the lane sums in an SIMD accumulator and reduced **once at the end of the block-loop**,
we amortize the tail across 4 j-iters.

**Net answer Q2:** compiler is given 4 structural chains per j-iter but only 2 cross-iteration
accumulators (`sum0`, `sum1`), which is the real cap. Expected utilization: **2/6 ≈ 33%** of ALC
FP channels on the accumulation path. Integer nibble-extract path uses 2/6 channels similarly.

---

## Q3 — Vector width: what does LCC do with AVX2 intrinsics?

### What E2K v5 actually has
From `elbrus_prog_guide_v1.2.pdf`:
- General SIMD on v1–v4 was **64-bit** (PFMULs, PFADDs, PMADDUBSH): 2×fp32 or 8×int8 per op.
- v5+ adds **QP (Quad-Packed) / 128-bit register class** (explicit in the doc:
  *"NB! Начиная с v5+, максимальная ширина 128 бит"* — prog_guide_v1.2, section on wsz).
  QR = 128-bit register, 4×fp32 or 16×int8 per op.
- 128-bit FPU in E8C2 is confirmed in `E8C2_ARCHITECTURE.md` §3: *"ширина FPU увеличена с 64 до 128 бит"*.
- The probe `q4k_e2k_kernel_probe.cpp` uses `__v2di` (128-bit) → `__v8hi` (8×int16), which is a
  **single 128-bit QP op** (qpmaddubsh) — one QP op matches AVX2's **half-register** `_mm256_maddubs_epi16`.

### What LCC does with our `__m256i` code

AVX2 ops are **256-bit**. There is no 256-bit native unit on E8C2 — max is 128-bit QR.
So LCC translates one `_mm256_maddubs_epi16` into **2× 128-bit qpmaddubsh**
(this is exactly what the probe tests: one AVX2-path loop vs a manually-issued pair of QP ops).

The probe result cited in MISSION_BRIEF: hand-written 128-bit QP intrinsics were **23% slower**
than AVX2→LCC. Reason is NOT width — both paths emit the same 128-bit ops — but that
LCC's AVX2 translator does better **instruction scheduling** on the surrounding scalar /
horizontal-sum glue than the handwritten version does.

**We are NOT losing width to 64-bit.** The 8-element AVX2 lanes are packed correctly as 2×128-bit
QP on v5. The probe at `examples/benchmarks/q4k_e2k_kernel_probe.cpp` confirms this:
line 225 `__v8hi p0_lo16_a = qpmaddubsh((__v2di)lo0_a, q8_lo_a)` is exactly half of one
`_mm256_maddubs_epi16` call.

**Caveat:** LCC's ability to fuse two 128-bit QP ops into a single wide-instruction bundle
(VLIW slot-pair) depends on ALC channel assignment. Per E8C2_ARCHITECTURE.md channels АЛК0/3
(FP-capable, integer packed, memory **read**) and АЛК1/4 (FP-capable) can issue two QP ops
per cycle if both are independent. Our loop body has enough independence between row0 and
row1 halves (4 chains) that the compiler should pack 4 QP per cycle. **This is probably why
AVX2→LCC wins:** the 2-row structure exposes 4 independent QP chains per cycle, which is
exactly what 6-wide VLIW wants.

---

## Q4 — EML_MT vs our custom GEMV

### What EML provides (EML_BLAS_GUIDE.md + EML_THREADING_API.md)
- `cblas_sgemm` — **FP32 dense**, 1840 GFLOPS with NUMA tiling (EML_BLAS_GUIDE.md line 116).
- `cblas_sgemv` — **FP32 dense** (line 62–63). Single-precision, NOT quantized.
- EML vector module: `eml_Vector_Add_32F`, `eml_Vector_Mul_32F`, `eml_Vector_Sum_32F` — FP32.
- EML has modules listed in §1: Core, Vector, Algebra, Signal, Image, Video, Graphics, Volume,
  Tensor. **None are described as supporting Q4_K / int4 / int8 quantized GEMV.**

### Not found in documentation
- No mention of quantized INT4/INT8 GEMV / GEMM in EML.
- No mention of asymmetric `uint8 × int8` dot product path (the `maddubs` equivalent) exposed
  as EML API.
- Not found in documentation whether `eml_Vector_*` vectorized over int8 arrays would give
  usable performance for our Q4_K × Q8 inner kernel.

### Compute vs memory for GEMV at token-scale for qwen3:4b Q4_K_M

Per MISSION_BRIEF: model per-token = 2.5 GB, peak achieved 5.5 tok/s → 13.75 GB/s.
Per E8C2 arch (§5): one chip = 68.3 GB/s DDR4-2400 (4 channels), 4 chips = **273 GB/s**
theoretical — but each MCP_TP process is bound to one NUMA node, so its effective ceiling
is **~68 GB/s** aggregate across its tiles if allocation is node-local.

With weights node-local replicated (the `ReplicatedWeight` hint at line 314 shows the code
supports this), 4 processes × 68 GB/s ≈ 273 GB/s. We are at **13.75 / 273 = 5% of sysBW**,
or **13.75 / 68 = 20% of single-node BW** for 1-proc.

So we are NOT yet memory-bandwidth-bound in absolute terms — we are still **compute-limited
in the decoder path** (int4→int8 expand, maddubs issue, horizontal reduction). Lifting
compute frees more nominal BW for us to use.

**Bottom line Q4:** EML offers no quantized GEMV that would replace our path. Our custom
kernel is the right place to optimize. EML_MT is useful **only if** we dequantize weights
to FP32 up-front (trade 2.5 GB of Q4_K for ~10 GB of FP32 → 4× BW, so likely worse for
decode). Stay in Q4_K path, optimize the LCC bundle quality.

---

## Q5 — Concrete proposals (file:line)

All in `torch/io/cpu_quant_gemv.h`. Ordered by expected impact.

### P1. Add `__restrict` on GEMV signature + loop-count pragma on outer block loop
**Where:** line 311 (signature) and before line 350 (outer bi-loop).
```cpp
inline void q4k_gemv_avx2(const void* __restrict__ weight_data,
                          const float* __restrict__ x,
                          float* __restrict__ y, int64_t K, int64_t N,
                          int64_t row_stride_bytes,
                          const torch::io::ReplicatedWeight* numa = nullptr) {
    ...
    #pragma loop count(38)       // K=9728 ⇒ 38 blocks; covers typical LLM range
    for (int64_t bi = 0; bi < blocks_per_row; ++bi) {
```
Also applies to the `for (int64_t bi = ...)` at line 467 (odd-row tail) and
`for (int64_t i = 0; i < nb; ++i)` at line 148 inside `quantize_x_q8`.

**Expected:** APB prefetch kicks in (per LCC_OPTIMIZATION.md §4.4 "~2x ускорение от APB
предвыборки" on vector+FP loops; our loop is already heavy on memory, realistic expectation
is **+10–25%** because we already do manual prefetch for cache lines but the APB is a separate
hardware path that streams values into registers).

**Risk:** If `__restrict` is violated (unlikely — y, x, and weight_data are from different
allocations), result is undefined. `loop count` is a hint only, wrong value just degrades.

### P2. Promote inner `j`-loop counter from `int` to `int64_t` and unroll ×4
**Where:** line 384 (`for (int j = 0; j < 256; j += 64)`).
```cpp
#pragma unroll(4)
for (int64_t j = 0; j < 256; j += 64) {
```
Per LCC_OPTIMIZATION.md §4.1 `int` counter disables APB. The loop has 4 iterations so full
unroll exposes the whole block body to the scheduler as one linear region — the 6-channel
VLIW allocator can now rearrange across j-boundaries.

**Expected:** **+5–15%** from better cross-iteration scheduling. Combined with P1 probably
fully masks the quantize_x_q8 tail.

**Risk:** Code size expansion (4× inner body ≈ 300 lines unrolled). L1I is 128 KB (plenty),
but SWP decisions may change. If regression, drop unroll to 2 or remove.

### P3. Replace scalar `sum0`/`sum1` with 4-lane SIMD accumulators, reduce once at end
**Where:** lines 426–427 / 451–452 inside the inner j-loop; currently writes go to
scalar floats, accumulation serialized at FP-add latency 4.
```cpp
// replace: float sum0 = 0, sum1 = 0;
__m128 vsum0 = _mm_setzero_ps(), vsum1 = _mm_setzero_ps();
__m128 vsum0b = _mm_setzero_ps(), vsum1b = _mm_setzero_ps();
// per j-iter: pack d_r*sc*dx*is and -dmin*m*sx into 4-wide vectors, _mm_add_ps into
// separate accumulators (vsum0 takes the + part, vsum0b the - part, or 2 parallel chains
// per row to double the independent chains the compiler sees).
// after outer bi-loop: horizontal reduce once.
```
Per E8C2_ARCHITECTURE.md §3 we have 6 FP channels; currently compiler sees 2 scalar FP
accumulator chains ≈ 2/6 utilization. Going to 4 chains (2 per row × + and − paths) = 4/6.

**Expected:** **+15–30%** on the FP accumulation path (which is currently the latency tail
of each j-iter, ~16 cycles serialized per row). This is the biggest structural issue the
audit surfaced.

**Risk:** Numerical reordering changes last-bit FP results. Our end-user is LLM inference —
bit-exact is not a requirement, but add an fp32 reference test before enabling.

### P4. Keep integer horizontal-sum as 128-bit partial-sum across bi-iterations
**Where:** lines 414–424 (row 0 hsum) and 439–449 (row 1 hsum). Currently we do a 3-step
scalar-like reduction *per j-iter × per row* = 6 horizontal reductions inside the inner body.
```cpp
// Accumulate at __m128i level across j-iters, fold to scalar ONCE per bi-loop
// (or even once per row):
__m128i vacc_is0_lo = _mm_setzero_si128();
for (int64_t j = 0; j < 256; j += 64) {
    ...
    __m128i t_lo = _mm_add_epi32(_mm256_castsi256_si128(p0_lo32),
                                  _mm256_extracti128_si256(p0_lo32, 1));
    vacc_is0_lo = _mm_add_epi32(vacc_is0_lo, t_lo);  // amortize shuffle+fold
}
// Fold once here, multiply by scales once here.
```
**Important subtlety:** each j-iter uses a *different* (sc_a, sc_b) scale pair (lines 388–389),
so we can't merely sum 4 integer results and multiply by one scalar scale. Instead we can
promote the int32 accumulators to fp32 in-vector (`_mm_cvtepi32_ps`) and multiply by a
4-wide scale vector built from the 4 (sc × d × dx) products — then add into the SIMD
accumulator from P3.

**Expected:** eliminates 3 `_mm_shuffle_epi32` + 3 `_mm_add_epi32` = 6 ops × 4 j-iters ×
2 halves × 2 rows = ~96 shuffle-class ops per block, replaced by 4 vector muls + adds.
Estimate **+5–10%**.

**Risk:** Non-trivial refactor; combine with P3 as one change.

### P5. `quantize_x_q8` scalar store path — replace with `_mm_packus_epi32` + `_mm_packs_epi16`
**Where:** lines 183–187.
```cpp
alignas(32) int32_t tmp[8];
_mm256_store_si256(reinterpret_cast<__m256i*>(tmp), vi);
for (int k = 0; k < 8; ++k) {
    x_q8[i].qs[j + k] = static_cast<int8_t>(tmp[k]);
}
```
This goes through memory (round-trip via `tmp[8]`) for every 8 values. Proper path is
`_mm256_packs_epi32` → `_mm256_packs_epi16` → store — all SIMD. On E8C2 via LCC this
translates to native QP pack ops (PACK* in the ISA, prog_guide_v1.2).

**Expected:** `quantize_x_q8` currently runs once per GEMV call (amortized), so this only
helps when N (output rows) is small. For qwen3:4b `attn_q_proj` (N=2560) the amortization
is 2560, so this is <1%. For embedding/head (N=vocab_size=152064) it's negligible. But if
we move to batch>1, the amortization drops and this matters. Low priority; **+1–3%** expected.

**Risk:** None structural; local change.

---

## Summary

| # | Proposal | file:line | Expected |
|---|---|---|---|
| P1 | `__restrict` + `#pragma loop count` | 311, 350, 467, 148 | +10–25% |
| P2 | `int64_t j` + `#pragma unroll(4)` | 384 | +5–15% |
| P3 | SIMD accumulators instead of scalar sum0/sum1 | 426–427, 451–452 | +15–30% |
| P4 | Amortize horizontal reduction across j-iters | 414–449 | +5–10% |
| P5 | Vectorize int8 pack in `quantize_x_q8` | 183–187 | +1–3% (batch>1: bigger) |

**Combined optimistic:** 1.25 × 1.10 × 1.20 × 1.07 ≈ **1.76×**
→ 4.7 tok/s × 1.76 ≈ **8.3 tok/s** single-process, or 5.5 × 1.76 ≈ **9.7 tok/s** TP4.

Still short of 10–15 tok/s target; remaining gap must come from:
- actual memory profile (P6-scope, MISSION_BRIEF §Гипотеза — different agent)
- cross-layer prefetching / weight-to-weight software pipelining (MISSION_BRIEF §Гипотеза)

**Non-goals found:** EML quantized GEMV does not exist — stay custom; ripping out AVX2 in
favour of hand-written E2K intrinsics does not help (already confirmed −23% in probe); SIMD
width is already 128-bit via LCC translation — no more lane-width to gain.

---

## Blockers

- Cannot run the kernel on Elbrus from this sandbox — all numeric speedup estimates are
  static analysis against documented latencies + the probe benchmark's relative results.
  Needs one `bench_q4k_gemv` run per proposal to confirm (quick: ~5 s per run at K=2560,N=9728
  from the probe's parameters).
- Not found in documentation: whether LCC 1.29 recognises `#pragma unroll(N)` on AVX2 intrinsics
  (it definitely recognises it on scalar code per `LCC_OPTIMIZATION.md` §3). If the compiler
  refuses, fallback is manual 4-way copy-paste of the j-iter body.
- Not found in documentation: exact rules for APB activation. We know `#pragma loop count` +
  `int64_t` counter are necessary; whether our existing manual `__builtin_prefetch` calls
  interfere with or complement APB has not been established.
