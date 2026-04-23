# Agent 4 — EML Audit for Q4_K Decode Acceleration

**Date:** 2026-04-22
**Scope:** Can EML (Elbrus Math Library) accelerate Q4_K GEMV decode path on Elbrus-8C2?
**Sources read:**
- `vliw_mission/round2/_inputs/eml_acceleration.txt` (paper by Ishin/Loginov/Vasiliev, MCST)
- `docs/elbrus/EML_BLAS_GUIDE.md`
- `docs/elbrus/EML_THREADING_API.md`

---

## 1. EML API Surface — What exists, what does NOT exist

### 1.1 Module inventory (from paper, page 2 table + BLAS_GUIDE section 1)

| Module | Function count | Relevance to Q4_K decode |
|--------|----------------|--------------------------|
| Core | 5 | alloc/free/version — irrelevant |
| Vector | 374 | arithmetic, math funcs, statistics — FP32/INT, **no quant** |
| Signal | 153 | convolution, FFT/Hartley — irrelevant |
| Image | 98 | image filters, DFT — irrelevant |
| **Algebra** | **306** | **BLAS 1/2/3 + LAPACK** — this is our `cblas_sgemm` |
| Video | 111 | interpolation, DCT, **"квантизация" (video quantization)** — **NOT neural quant** |
| Graphics | 246 | drawing primitives — irrelevant |
| Volume | 25 | ray casting — irrelevant |
| Tensor | (not listed in paper; BLAS_GUIDE line 25) | existence confirmed, contents undocumented |
| **TOTAL** | **1318** | |

### 1.2 Data types supported (paper page 2, lines 58-67)

Direct quote (line 59-60):
> "Библиотека eml работает с 8, 16, 32, 64 - разрядными целыми (знаковыми и беззнаковыми), плавающими типами одинарной и двойной точности, а так же с комплексными типами."

Type naming: `eml_8u` (uint8), `eml_16s` (int16), `eml_32s` (int32), `eml_32f` (fp32), `eml_32fc` (complex fp32), etc.

**Critically: there is NO 4-bit type.** Q4_K is 4-bit packed. EML does not expose any `eml_4u` / `eml_q4` primitive. Everything 4-bit must be unpacked to 8-bit or 32-bit before touching EML.

### 1.3 Is there `eml_*gemv*` with quantized input?

**NO.** Evidence:
1. Paper lists only BLAS 1/2/3 + LAPACK in Algebra module — standard real/complex types only (page 2, line 36-37).
2. BLAS_GUIDE section 2 shows only `cblas_sgemm`, `cblas_sgemv`, `cblas_sdot`, `cblas_saxpy` — all FP32.
3. There is no mention of `gemv_q4`, `gemv_int8`, or quantized GEMV anywhere in the two docs or the paper.
4. BLAS is by definition typed: `s` (single), `d` (double), `c`/`z` (complex). No `q` / `i8` extension exists in classical BLAS and MCST has not added one.

**Verdict:** No quantized GEMV primitive in EML. We cannot swap `cpu_quant_gemv.h` for an EML call.

---

## 2. EML Multi-core API — What works, what SIGILLs

### 2.1 Three thread-control surfaces

From `EML_THREADING_API.md` + `EML_BLAS_GUIDE.md`:

| Mechanism | Source | Behavior |
|-----------|--------|----------|
| `omp_set_num_threads(N)` | OpenMP runtime | **IGNORED by eml_mt internals** (BLAS_GUIDE line 110) |
| `OMP_NUM_THREADS` env | read at EML init | sets eml_mt worker count once (BLAS_GUIDE line 111) |
| `eml_SetNumThreads(N)` | `eml/eml_core.h` | **EML's own thread control**, independent of OMP (THREADING_API line 17-21) |

Direct quote (THREADING_API lines 13-14):
> "EML имеет **СВОЙ** API управления потоками, НЕЗАВИСИМЫЙ от `omp_set_num_threads()`."

### 2.2 What exactly causes SIGILL

From BLAS_GUIDE lines 121-127:
> "cblas_sgemm из eml_mt НЕЛЬЗЯ вызывать из pthread/std::thread! Вызов из любого потока кроме main → SIGILL (Illegal Instruction). Работает ТОЛЬКО из main thread и из OMP parallel regions."

This matches our MEMORY.md entry (`feedback_eml_pthread_sigill.md`, 2026-04-02):
> "cblas_sgemm from ANY pthread = SIGILL on E2K. Main thread only. Removed all pthread NUMA wrappers."

### 2.3 What works (confirmed patterns)

1. **Single-threaded EML from main:** `-leml` library, no threading, 66 GFLOPS/core.
2. **eml_mt from main:** `-leml_mt`, internal OMP pool, reads OMP_NUM_THREADS at init.
3. **`-leml` (ST) inside `#pragma omp parallel` regions:** safe — each OMP worker calls ST EML on its own tile.
4. **`eml_SetNumThreads(K)` per-tile before each call:** THREADING_API lines 36-39 recommend this for NUMA tiling — set to `cores_per_node` (8 on E8C2) or `1` for fully manual tile control.

### 2.4 What does NOT work

1. `pthread_create` / `std::thread` → `cblas_sgemm` body → SIGILL.
2. Any C++ thread pool (our `c10/util/ThreadPool.h`) calling EML BLAS — same SIGILL class.
3. Cross-NUMA calls from eml_mt without pinning: 152-265 GFLOPS vs. theoretical 2304 (BLAS_GUIDE line 112).

### 2.5 Best measured throughput (BLAS_GUIDE section 5 + MEMORY NUMA scaling entry)

- Default eml_mt 32-core: 324 GFLOPS
- 4× OMP outer × ST EML inner + NUMA pinning: **1840 GFLOPS (92% of peak)**
- But this is **FP32 GEMM**, compute-bound. Decode is memory-bound — irrelevant absolute number.

---

## 3. Is there `eml_quant_decode` / `eml_dequantize_q4`?

**NO.** Evidence:

1. Paper, page 2, lists all 8 modules + function counts. None mention quantization of weights.
2. The word "квантизация" appears ONCE in the paper (line 39, Video module) — it refers to **DCT coefficient quantization** (JPEG/MPEG video compression), not neural-net weight quantization.
3. BLAS_GUIDE line 22: "Video — Интерполяция, DCT" — same confirmation.
4. Vector module has FP↔INT type conversions (paper line 28: "преобразование типов") but these are typed-width conversions (int32→float32), not block-quantized (Q4_K has per-block scales + mins, not a simple type cast).

**Could EML's Vector SIMD ops beat our AVX2-style dequant?** AVX2 doesn't exist on Elbrus. Our current dequant on E2K is either scalar C or hand-written `#pragma ivdep` inner loops. EML's Vector module is hand-tuned VLIW (paper section 3 shows 0.38 cycles/element on `MaxIndex_32F`). **But** EML has no Q4→FP32 primitive to call. Building one from `eml_Vector_Mul_32F` + `eml_Vector_Add_32F` would require per-element scale-broadcast and per-nibble unpack, and the unpack is the expensive part, not the multiply-add. Net gain near zero.

**Verdict:** No hardware-accelerated Q4 dequant path in EML. Writing our own in pure C with `#pragma loop_count` + `#pragma ivdep` + aligned loads is the only route.

---

## 4. RMSNorm / Softmax / LayerNorm in EML?

**NO.** Evidence:

1. Paper module list (page 2) — Vector module contains "арифметические, логические, преобразование типов, математические функции, статистика" (arithmetic, logical, type conversion, math functions, statistics). LayerNorm/RMSNorm/Softmax are composite operations, not exposed as single primitives.
2. BLAS_GUIDE section 2 lists only `Vector_Add_32F`, `Vector_Mul_32F`, `Vector_Sum_32F` as EML-specific. No norm/softmax.
3. Elementwise `exp`, `log`, `tanh` are *probably* in Vector (374 functions, paper claims statistics and math functions) but this is not documented in the docs we have. Even if present, they only give element-wise acceleration — the reduction + reciprocal-sqrt + broadcast still has to be written manually.

**Actionable:** If Vector has `eml_Vector_Exp_32F` / `eml_Vector_Sqrt_32F` / `eml_Vector_Sum_32F` (confirmed — line 84-85 of BLAS_GUIDE shows `eml_Vector_Sum_32F`), we *could* compose softmax in 4 passes. But for H=1024 hidden the call overhead likely dominates. Not worth pursuing unless profiling shows softmax/rmsnorm in top-5 hotspots (for Q4_K decode at 5 tok/s, GEMV is 80%+ of time).

---

## 5. EML INT8 GEMM — Does it exist? Can we pre-dequant Q4_K→Q8_0 and use it?

**NO INT8 GEMM in EML.** Evidence:

1. BLAS is `s/d/c/z` only — single, double, complex-single, complex-double. No `i8gemm`.
2. Paper Algebra section (page 2, line 36-37): "стандартные пакеты работы с матрицами и векторами BLAS 1/2/3, LAPACK" — classical BLAS, which has no integer GEMM.
3. No mention of INT8 matmul, VNNI-equivalent, or quant-aware GEMM anywhere in the docs.
4. E2K has SIMD128 with packed-integer ops (`paddsh`, `pfmax`, etc., shown in paper pages 3-6), but these are exposed via EML's **Vector** module for element-wise, not as a GEMM kernel.

**Could we roll our own INT8 GEMM using EML Vector primitives?** In theory: inner product = `eml_Vector_Mul_16S` + `eml_Vector_Sum_32S` accumulation (with widening). But:
- 8-bit packed-multiply-accumulate-to-32-bit is the critical E2K intrinsic. EML doesn't expose it as a BLAS-grade tuned GEMM kernel — just as a per-vector op. Composing it into a cache-tiled GEMM from Vector calls would hit the same tile-dispatch overhead as our existing code.
- Even with working INT8 GEMM, decode is **memory-bound**, not compute-bound. MISSION.md lines 13-16 say per-chip DDR is ~20 GB/s → ~50 tok/s hard cap. Q4_K already reads 4 bits/weight. Switching to Q8_0 **doubles** memory traffic (8 bits/weight) → halves tok/s. **Q8_0 path is a regression for decode**, regardless of GEMM kernel speed.

**Verdict:** Pre-dequant Q4_K→Q8_0 is architecturally wrong for decode. Even if EML had INT8 GEMM, using it would be slower than current Q4_K GEMV because of 2× bandwidth.

---

## 6. 1840 GFLOPS vs. memory-bound — any FFI vectorized path for quant GEMV?

BLAS_GUIDE section 7 gives GEMM compute numbers (66 GFLOPS/core ST, 1840 GFLOPS aggregate FP32). These are **compute-bound** FP32×FP32×FP32 workloads and irrelevant to decode.

For decode (N=1 GEMV, batch=1):
- Total weights accessed per token ≈ total param bytes ≈ 2.5 GB for qwen3:4b Q4_K (MEMORY / GGUF inference line).
- Per-chip DDR on E2K: ~20 GB/s effective (MISSION.md line 13).
- Hard ceiling: 20 / 2.5 = **8 tok/s per chip**. MISSION already says current 5.3 tok/s = ~30% utilization.

The EML paper and docs expose **no FFI surface specifically for Q4/Q8/INT-GEMV**. The only vectorized hooks are:
- Vector module (element-wise, no block-structured quant)
- Algebra BLAS (FP32/FP64 only)
- Signal/Image/Video (wrong domain)

There is no bypass for the 8 tok/s per-chip DDR wall through EML. The wall is physical, not software.

---

## 7. Are there EML intrinsics specific to inference / neural-net quantization?

**CONFIRMED NEGATIVE.** EML is a general-purpose math/multimedia library targeting DSP, image/video codecs, and classical BLAS/LAPACK. It predates the neural-net quantization era:

- Paper copyright context: authors describe Elbrus-2C+ optimizations, pre-2015 era.
- BLAS_GUIDE lists modules last updated for E8C2 v5; no "nnet" or "quant" module appears.
- E2K's VLIW packed-int SIMD (`padds*`, `psrah`, `insfd`, paper pages 3-6) is visible but only as the mechanism behind Vector-module ops, not as a Q4-aware layer.
- EML copyright line (THREADING_API line 60): "(c) 2006-2024 AO MCST" — long-running library without a quant-inference pivot.

Compare Intel MKL: also had no quant GEMM until oneDNN shipped VNNI-specific `cblas_gemm_s8s8s32` etc. around 2019. MCST has not shipped an equivalent for E2K.

---

## 8. Honest Verdict

**EML delivers NOTHING for Q4_K decode acceleration.** Concretely:

| Hope | Reality |
|------|---------|
| Q4/Q8 quantized GEMV primitive | Does not exist. BLAS only has FP32/FP64/complex. |
| Q4→FP32 hardware-accelerated dequant | Does not exist. No `eml_dequantize_*`. |
| INT8 GEMM we could use after one-shot Q4→Q8 conversion | Does not exist; also architecturally wrong (2× bandwidth → slower decode). |
| RMSNorm / Softmax / LayerNorm primitives | Not exposed. At best compose from Vector Sum/Mul/Exp — call overhead dominates at H=1024. |
| Multi-threaded speedup for decode via eml_mt | Irrelevant. Decode is memory-bound, not compute-bound. Aggregate GFLOPS numbers (1840) don't apply. |
| Neural-net specific intrinsics | Zero. EML is pre-neural-quant, general-purpose DSP/BLAS. |

**Only EML usage already in place that remains correct:**
- `cblas_sgemm` (eml_mt) for **FP32 training** (PIR 342M Local SGD) — keep as-is. Paper confirms 82× speedup vs. naive C GEMM (BLAS_GUIDE line 186). This is compute-bound and EML is optimal here.

**For Q4_K inference on E2K, EML is a dead end.** The real levers are the ones MISSION.md already lists:
1. Manual VLIW tuning of our `cpu_quant_gemv.h` with `#pragma loop_count` / `#pragma ivdep` / aligned packed-int ops.
2. Reducing bytes-per-token (KV cache compression, speculative decode where acceptance > 0%).
3. Tensor-parallel across 4 NUMA nodes with weights replicated to drop aggregate bandwidth per chip — but SHM AllReduce already tried, bounded by synchronization overhead.
4. Per-node weight pinning + non-temporal streaming loads to saturate 20 GB/s DDR — raw memory tuning, not an EML question.

**Do not waste cycles adding EML calls to the Q4_K path.** It will not help. The only EML-adjacent investigation worth zero-extra-work validation: confirm on the actual Elbrus box whether `/usr/include/eml/eml_algebra.h` or `/usr/include/eml/eml_tensor.h` (Tensor module, BLAS_GUIDE line 25, contents undocumented) expose anything surprising. Expected outcome: no. But a single `grep -i "quant\|int8\|q4\|q8\|sgemm_s8"` on `/usr/include/eml/*.h` on the server would definitively close the question in 10 seconds.

---

## Summary table for caller

| Question | Answer |
|----------|--------|
| EML has quantized GEMV? | **No** |
| EML INT8 GEMM? | **No** |
| EML Q4 dequant primitive? | **No** |
| EML RMSNorm/Softmax/LayerNorm? | **No** (composable from Vector Sum/Mul/Exp, overhead dominates) |
| EML safe from pthread? | **No** — main thread or OMP parallel regions only. `eml_SetNumThreads()` + OMP tiling is the correct pattern. |
| Will EML accelerate Q4_K decode? | **No — architecturally irrelevant.** Keep EML for FP32 training GEMM only. |
| Worth one final server-side grep to confirm? | Yes — 10 seconds on `/usr/include/eml/*.h`, expected outcome still "no". |
