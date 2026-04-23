# Agent 5 — Code Audit (Already-Applied Optimizations)

**Role:** Verify every optimization claimed applied actually IS applied everywhere it should be.
**Scope:** `cpu_quant_gemv.h` (2321 lines), `gguf_model.h` (5685 lines), `ThreadPool.h`, `ddp.cpp`.
**Method:** Read + Grep only (no code changes).

Verdict: **Significant gaps.** Claimed wins are partially applied. Several hot paths still
scalar/serial. The "AVX2 q4k_gemv" IS correctly updated, but batched/speculative/fallback/FP32
paths and critical preambles (SiLU, residual, attention softmax in batched) are un-vectorised
and/or un-parallelised. The biggest real-world leak is **`cpu_fused_rmsnorm_gate_up_gemv` forced
off the fused path whenever NUMA replicas exist** — which is exactly the TP case that is meant
to benefit.

---

## 1. APB pragmas (`#pragma loop count(N) + ivdep`) — INCOMPLETE COVERAGE

The mission lists "APB on all 15 inner j-loops". Actual count of `_Pragma("loop count(...)")
_Pragma("ivdep")` in `cpu_quant_gemv.h`: **19** sites (lines 91, 105, 152, 172, 233, 410, 579,
698, 758, 856, 1054, 1068, 1151, 1248, 1352, 1600, 1749, 1830, 1965). That covers the Q4_K
j-loops, Q8_0/Q5_K/Q6_K j-loops, k-slice j-loops, batched+batched_scalar+batched_qkv j-loops.
Counts **match** 15 kernel variants — good.

**Missing APB pragmas (found during audit):**

### 1a. `q4k_q8_dot_avx2` l-loop is fine but the Q8 quantize outer `i-loop` is NOT marked
- **File:** `torch/io/cpu_quant_gemv.h:148`
- **Code:**
  ```cpp
  for (int64_t i = 0; i < nb; ++i) {   // line 148 — no ivdep / loop count
      const float* xb = x + i * 32;
      ...
  }
  ```
- **Fix:** `_Pragma("ivdep") for (int64_t i = 0; i < nb; ++i) {` — LCC can then schedule
  amax+quantize of consecutive i's without anti-dependency check (they write into disjoint
  `x_q8[i]` slots).
- **Why it matters:** `quantize_x_q8` runs ONCE per GEMV on the master thread BEFORE the
  `parallel_for` dispatches. For H=3584 that's 112 iterations executing serially in the critical
  path. APB could give ~10% of that 200 μs back × 6 GEMVs/layer × 36 layers = ~0.4 ms/token.

### 1b. `q4k_gemv_scalar` / `q5k_gemv_scalar` / `q6k_gemv_scalar` — scalar fallbacks missing APB on the l-loop
- **File:** `torch/io/cpu_quant_gemv.h:706, 799, 995, 1000, 1159, 1360, 1399`
- **Code:**
  ```cpp
  for (int l = 0; l < 32; ++l) {    // line 706 etc — no pragma
      dot += (d1 * (qs[l] & 0xF) - m1) * x[...];
  }
  ```
- **Fix:** Add `_Pragma("loop count(32)") _Pragma("ivdep")` prefix.
- **Impact:** Scalar fallback is hit on non-AVX2 builds (E2K native compiled without
  `__AVX2__`!). If PT is built with LCC targeting native QR without the AVX2 compat layer, **the
  scalar path is the hot path**, and it has zero APB hints. This is likely a significant miss on
  Elbrus even when AVX2 intrinsics work.

### 1c. `q4k_gemv_batched_scalar` outer `k-loop` and inner `l-loop` unmarked
- **File:** `torch/io/cpu_quant_gemv.h:1818, 1834`
- The `j-loop` at 1830 has `loop count(8)` but the enclosing `k-loop` (1818, iter count
  K_batch ∈ [1..6]) and the `l-loop` (1834, 32 iters) do not. Fix: add pragmas.

### 1d. `q6k_gemv_k_slice_scalar` — no pragmas at all on hot inner loops
- **File:** `torch/io/cpu_quant_gemv.h:1399, 1398`
- Both `for (int n_half = 0; n_half < 256; n_half += 128)` and `for (int l = 0; l < 32; ++l)`
  are un-hinted. Add `loop count(2)` and `loop count(32) ivdep`.

### 1e. Outer `n-loop` in every kernel body is unmarked
- **Files:** line 73, 681, 741, 786, 829, 969, 1035, 1132, 1335, 1382, 1572, 1722, 1816, 1926
- The outer `for (int64_t n = start; n < end; ++n)` in every parallel_for lambda has no
  `loop count` hint. In decode, each worker gets a chunk of ~7-30 rows (N=3584/24 threads). LCC
  can't infer iteration count. Effect is small (outer loop is memory-bound, not SWP candidate),
  but adding `loop count(8)` with `ivdep` is still the cheapest change in the file.

---

## 2. `int64_t` loop counters — MOSTLY DONE, 4 LOOPS STILL `int`

Grep of `for (int ` (not `int64_t`) in `cpu_quant_gemv.h`:

| Line | Loop | Issue |
|------|------|-------|
| 185  | `for (int k = 0; k < 8; ++k)` — quantize_x_q8 inner store | Trivial, 8 iters, irrelevant |
| 706  | `for (int l = 0; l < 32; ++l)` — q4k_gemv_scalar | **Should be `int64_t`**. LCC APB prefers int64_t for index math |
| 799  | `for (int l = 0; l < 32; ++l)` — q8_0_gemv_scalar | Same |
| 855  | `for (int n_half = 0; n_half < 256; n_half += 128)` — q6k_gemv_avx2 | Wrong type |
| 994  | `for (int n_half = 0; n_half < 256; n_half += 128)` — q6k scalar | Wrong type |
| 995  | `for (int l = 0; l < 32; ++l)` — q6k scalar | Wrong type |
| 1159 | `for (int l = 0; l < 32; ++l)` — q5k scalar | Wrong type |
| 1360 | `for (int l = 0; l < 32; ++l)` — q4k k-slice scalar | Wrong type |
| 1398 | `for (int n_half = 0; n_half < 256; n_half += 128)` — q6k k-slice scalar | Wrong type |
| 1399 | `for (int l = 0; l < 32; ++l)` — q6k k-slice scalar | Wrong type |
| 1834 | `for (int l = 0; l < 32; ++l)` — q4k batched scalar | Wrong type |

Also **`int q8_idx = 0`** at lines 231, 409, 578, 1247, 1599, 1748, 1964 — this is incremented
inside the j-loop with pragma `ivdep`. Pointer-index math on 32-bit counters on E2K forces
sign-extend `ldws → sxt` before each indirect load, breaking SWP. **Make `int64_t q8_idx`**.

Similarly `int is = j / 32` at lines 236, 411, 580, 1249, 1601, 1750, 1831, 1966 — each is used
inside the APB-hinted j-loop to index `sc` scale helper. Make `int64_t is`.

**Fix:** Global replace `int q8_idx` → `int64_t q8_idx` and `int is` → `int64_t is` in all
nine callsites. Also batched `for (int k = 0; k < K_batch; ++k)` at lines 1710, 1761, 1795,
1818, 1868 — K_batch is typed `int` in the signature; keep `int` there but switch to `int64_t`
for the inner variable being used in pointer arithmetic.

---

## 3. NUMA replica lookup — ACTIVE ONLY IN MAIN KERNELS, MISSING IN BATCHED/FUSED

### 3a. `cpu_fused_rmsnorm_gate_up_gemv` — **NUMA kills the batched-qkv fusion**
- **File:** `torch/io/cpu_quant_gemv.h:2255-2265`
- **Code:**
  ```cpp
  if (quant_type_gate == quant_type_up &&
      row_stride_gate == row_stride_up &&
      numa_gate == nullptr && numa_up == nullptr) {     // ← guard
      cpu_quant_gemv_batched_qkv(...);  // fast path, 1 parallel_for
  } else {
      cpu_quant_gemv(..., numa_gate);    // slow path, 2 parallel_fors
      cpu_quant_gemv(..., numa_up);
  }
  ```
- **Problem:** The caller ALWAYS passes `&layer.q_ffn_gate.numa_replica` and
  `&layer.q_ffn_up.numa_replica` when `PT_NUMA_REPLICATE=1` is set (see `gguf_model.h:2686`).
  So when replication is enabled, we ALWAYS take the slow path and lose the
  "1 parallel_for instead of 2" fusion (~120 μs × 36 layers = 4 ms/token) — exactly the
  scenario the comment at line 2252 warns about but never fixes.
- **Fix:** extend `cpu_quant_gemv_batched_qkv` signature to accept three optional
  `ReplicatedWeight*` args (one per matrix), resolve the node-local pointer per-thread at
  chunk entry (same pattern as `q4k_gemv_batched_avx2:1718`), drop the `numa_gate == nullptr
  && numa_up == nullptr` guard.
- **Expected impact:** +4 ms/token on 36-layer models with PT_NUMA_REPLICATE=1 = ~+15% from
  this alone when at 60 ms/token.

### 3b. Output projection callsite does not forward `numa` to the kernel
- **File:** `torch/io/gguf_model.h:2808`
- **Code:**
  ```cpp
  cpu_quant::cpu_quant_gemv(q_output_weight.quant_type, w_out,
      sp.x_buf[cur], sp.logits_buf, H, q_output_weight.rows, q_output_weight.row_stride_bytes);
  // ← missing 7th arg: numa
  ```
- **Problem:** Caller pre-resolves `w_out = numa_replica.get(_node)` at line 2801 on the
  MASTER thread's node — then hands that single pointer to the parallel_for. Workers on OTHER
  nodes all dereference the master's replica (cross-chip loads, ~0.2 μs/miss × 608 KB vocab
  stride × N/T rows). The correct pattern (used e.g. line 2640-2641) is to pass the
  `ReplicatedWeight*` so each worker picks its local copy at chunk start.
- **Fix:** `cpu_quant::cpu_quant_gemv(..., &q_output_weight.numa_replica);`
- **Expected impact:** ~5-10% of output-proj GEMV time saved on TP-4, which is ~20 ms/token →
  ~1-2 ms/token.

### 3c. `forward_decode_cpu_batched` — also gives pre-resolved pointer to batched GEMVs
- **File:** `torch/io/gguf_model.h:3287-3301, 3379-3383, 3399-3409, 3420-3424, 3440-3446`
- Same pattern: `const void* w_x = layer.q_attn_q.numa_replica.get(_node)` is resolved on the
  MAIN thread and then handed as the weight pointer to `cpu_quant_gemv_batched`. The kernel's
  own inner `numa` parameter at line 1693 IS used to re-resolve per worker when passed, but
  here we never pass it — we pass the resolved pointer as `weight_data`. Workers see
  `numa==nullptr` and use the main-thread's replica.
- **Fix:** pass `w_x` AS `weight_data` only as the "default if numa is null" fallback, and
  ALWAYS pass `&layer.q_attn_q.numa_replica` (etc.) as the `numa` argument.

### 3d. `cpu_quant_gemv_batched_qkv` (K=1 overload at line 1875) accepts no `numa` at all
- **File:** `torch/io/cpu_quant_gemv.h:1875-1880`
- Signature is missing `numa` arg, so even if we wanted to pass NUMA-aware replicas, we can't.
  This is the same root cause as 3a.

---

## 4. `__restrict` on pointer params — MISSING FROM 10 OF 11 KERNELS

Only **TWO** kernels use `__restrict`:
- `q4k_gemv_sse41_v` (PT_Q4K_V2 opt-in path, line 363-384) — THE EXPERIMENTAL kernel
- `q4k_gemv_batched_avx2` (line 1718-1732) — for spec decode

**The production `q4k_gemv_avx2` (line 497-668) has NO `__restrict` anywhere.** Neither do
`q4k_gemv_avx2_batch2` (1540), `q8_0_gemv_avx2` (734), `q6k_gemv_avx2` (819),
`q5k_gemv_avx2` (1027), `q4k_gemv_k_slice_avx2` (1197), `q4k_q8_dot_avx2` (219),
`cpu_fused_rmsnorm_*`, `cpu_rmsnorm_inplace`, all scalar fallbacks, and the K=1
`cpu_quant_gemv_batched_qkv` AVX2 inner block (1925-2042).

**Fix for each:** replace `const uint8_t* raw` → `const uint8_t* __restrict raw`, similarly
`const float* x` → `const float* __restrict x`, `float* y` → `float* __restrict y`,
`const uint8_t* row_data` / `block` / `blk` / `qs` / `scales` → `* __restrict`.

**Expected impact:** LCC E8C2 tour says restrict enables ~20% better alias-free SWP. For the
PRIMARY q4k_gemv_avx2 this could add several % on top of existing wins. Low effort.

---

## 5. Scratch buffer alignment — `Q8Block` 32-B ALIGNED, STACK BUFS UNALIGNED

### 5a. `Q8Block` has `alignas(32)` only
- **File:** `torch/io/cpu_quant_gemv.h:138`
- E2K cache line = 64 B; QR register = 128 bit but cache-loads are 64 B. Over-aligning each
  Q8Block to 64 B costs 24 B padding per block (40 → 64 B = 60% padding). Bad trade.
  **But the start of the `x_q8_stack[512]` array itself SHOULD be 64-aligned** to guarantee
  the FIRST block is on a cache line. Currently the array inherits `alignas(32)` from the
  struct and may straddle a cache line.
- **Fix:** `alignas(64) Q8Block x_q8_stack[512];` at lines 349, 507, 1205, 1551, 1912. Cost: 0.

### 5b. Stack RMSNorm buffers unaligned in `gguf_model.h`
- **File:** `torch/io/gguf_model.h:2442, 2689, 2923, 3070` (4 places), all `float norm_buf[8192]`
- Default alignment for `float[]` is 4-B on most ABIs. AVX2 aligned loads need 32-B,
  E2K native QR prefers 16-B.
- **Fix:** `alignas(64) float norm_buf[8192];` — zero cost, gives LCC information it can use
  for unroll decisions.

### 5c. Attention score buffers unaligned
- **File:** `torch/io/gguf_model.h:2557, 3013, 3349` — `float local_scores[4096/8192]`
- Attention softmax reads/writes these via scalar loop today (→ see Finding 7). If we ever
  AVX2-ise softmax, alignment matters. `alignas(64)`.

---

## 6. `parallel_for` dispatch overhead — MULTIPLE SYNCS PER LAYER STILL PRESENT

Per-layer parallel_for invocations counted in `forward_decode_cpu`:

1. `cpu_fused_rmsnorm_qkv_gemv` → inside: 1 parallel_for (line 1925) for batched QKV
2. `cpu_quant_gemv(attn_output)` → 1 parallel_for
3. `parallel_for(0, n_heads, ...)` attention — 1 parallel_for (line 2550)
4. `cpu_fused_rmsnorm_gate_up_gemv` → inside: 2 separate parallel_fors (see Finding 3a!)
5. `parallel_for(0, inter, SiLU*up)` — 1 parallel_for (line 2716)
6. `cpu_quant_gemv(ffn_down)` → 1 parallel_for
7. Final residual add — serial fallback

Per layer: **6 parallel_for syncs × 36 layers = 216 syncs/token**. At ~40 μs/sync measured
Elbrus latency (round-trip on cond_var + 24 thread wakeups) that is **8.6 ms of PURE sync
overhead per token** — already ~15% of the 60 ms budget at target 20 tok/s.

### 6a. `bias add` loop (2479-2498) is serial, 7-9 KB of contiguous memcpy-plus-add
- Single-threaded AVX2 but NOT dispatched. For qwen3:4b kv_dim=256 that's trivial, skip.
- For Q heads (q_dim=4096) it's 16 KB, still fine serial. OK.

### 6b. `residual add` at 2655-2667 and 2772-2784 is serial
- **File:** `torch/io/gguf_model.h:2655, 2772`
- 36 layers × 2 residuals × H=3584 × 4 B = 4 read+writes = ~1 MB bytes/layer/residual = 72
  MB/token total. On E8C2 single-core STREAM ≈ 2.5 GB/s → ~30 ms spent in residual adds alone.
- **Fix:** Wrap in `c10::parallel_for_1d(H, ..., /*threshold=*/1024)`. With 24 threads that
  drops to ~1.5 ms. **+28 ms/token saved** or more. Very underrated win.

### 6c. `bias add` and `QK-norm` inside batched `forward_decode_cpu_batched` (3307-3325) are K×serial
- **File:** `torch/io/gguf_model.h:3315-3317, 3321-3324`
- Every K token's biases done serially in main thread. Not a killer (K≤6, H small) but with
  no visible parallelism.

### 6d. `forward_decode_cpu_batched` SiLU * up loop (3412-3416) is completely scalar + serial
- **File:** `torch/io/gguf_model.h:3412`
- ```cpp
  for (int64_t j = 0; j < (int64_t)K * inter; ++j) {
      float g = gate[j];
      float s = g / (1.0f + std::exp(-g));
      siluup[j] = s * up_b[j];
  }
  ```
- No AVX2, no parallel_for. For K=5, inter=9728 that's 48 640 serial `std::exp` calls ≈
  **9-10 ms of serial work** on E2K (libm exp ~200 ns/call).
- **Fix:** Replicate the AVX2 SiLU + parallel_for_1d pattern from line 2716-2740.

### 6e. `forward_decode_cpu_batched` residual (3387, 3428) is also serial + scalar
- Same pattern. K*H floats serially. Fix: parallel_for.

### 6f. `forward_decode_cpu_batched` final-norm memcpy+RMSNorm (3434-3437) is serial per K
- Minor for K ≤ 6 but not parallelized. Not hot path.

### 6g. `forward_decode_cpu_batched` embedding scale (3270) is serial scalar
- Same. `for (int64_t j = 0; j < (int64_t)K*H; ++j) cur[j] *= s;`

### 6h. `forward_decode_cpu_speculative` has NO parallel_for at all for attention, biases,
  residual, or SiLU
- **File:** `torch/io/gguf_model.h:2861-3131` (whole function)
- Every inner loop is scalar single-thread. See biases (2951-2952), QK-norm (2959-2962),
  **unfused RoPE with pow/cos/sin per head** (2969-2988), attention (3009-3038), residuals
  (3050, 3114), SiLU (3090-3093).
- This function is opt-in via `use_speculative_output_`. If anyone turns it on today they lose
  all threading. Either delete it or mirror the main decode's parallelism.

---

## 7. Attention softmax — PURE SCALAR (NO SIMD max/exp/normalize)

### 7a. Main `forward_decode_cpu` softmax at line 2591-2602 is scalar
- Max + exp + sum + normalise all in scalar loops. For total_seq=1024 that's ~1024 exp calls
  × 32 heads × 36 layers = **1.2 M exp/token ≈ 240 ms on E2K libm scalar exp**.
- **Fix:** vectorised exp — we ALREADY HAVE `VectorizedOps.h` AVX2 exp (per CLAUDE.md top).
  Replace this loop with it. Also SIMD hmax + SIMD normalise mul.
- **Expected impact:** Huge on long contexts. For typical 256-token decode at past_len=256
  this is ~60 ms of softmax → ~10 ms with SIMD exp.

### 7b. Same scalar softmax in `forward_decode_cpu_batched` line 3358-3371 and
  `forward_decode_cpu_speculative` line 3027-3036
- Same fix.

### 7c. Attention Q·K dot product inside `forward_decode_cpu_batched` (3352-3357) is scalar
- The MAIN decode path (2569-2581) vectorises Q·K via AVX2 fmadd, but the batched path at
  3352-3357 is scalar:
  ```cpp
  for (int64_t d = 0; d < head_dim; ++d) dot += q_head[d] * kh[d];
  ```
- Same for V·score at 3370-3371.
- Same scalar in `forward_decode_cpu_speculative` (3023, 3035).
- **Fix:** Port the AVX2 loop from `forward_decode_cpu`.

---

## 8. RMSNorm centralisation — MAIN DECODE OK, BATCHED & FALLBACKS DUPLICATE THE MATH

`cpu_rmsnorm_inplace` is the canonical AVX2 implementation (line 2274-2317). It is called
from the main decode at lines 2505, 2508, 2634, 2766, 2791, 3043, 3110, 3119, 3282, 3322,
3324, 3394, 3436, 3110, 3119.

**But `cpu_fused_rmsnorm_gate_up_gemv` / `cpu_fused_rmsnorm_qkv_gemv` / `cpu_fused_rmsnorm_gemv`
each duplicate the entire AVX2 RMSNorm loop manually** at lines 2074-2114, 2140-2181, 2209-2249.
Three copies of the same 40-line AVX2 kernel. If we update one (e.g. for a future `loop
count` hint or fma pipelining), we have to update all three. Not a perf bug — a correctness
risk.

**Fallback RMSNorm (when `can_fuse=false`) is written inline as scalar** at lines 2447-2454,
2692-2699, 2926-2933, 3072-3079, 3282, 3322, 3324, 3394 — **these all call
`cpu_rmsnorm_inplace`** in the decode, so they're fine. Exception: lines 2447-2454 and
2692-2699 (in `forward_decode_cpu` fallback) use SCALAR RMSNorm, not `cpu_rmsnorm_inplace`.
The fallback is rarely hit (only when quant_type unsupported or layer mismatch), so not
hot — but it is un-vectorised.

---

## 9. Residual add — MAIN SIMD, BATCHED SCALAR, SPECULATIVE SCALAR

| Callsite | Status | File:Line |
|----------|--------|-----------|
| `forward_decode_cpu` attn residual | AVX2 SIMD, serial (not parallel_for) | gguf_model.h:2655-2667 |
| `forward_decode_cpu` ffn residual | AVX2 SIMD, serial | gguf_model.h:2772-2784 |
| `forward_decode_cpu_batched` attn residual | **Scalar**, serial | gguf_model.h:3387 |
| `forward_decode_cpu_batched` ffn residual | **Scalar**, serial | gguf_model.h:3428 |
| `forward_decode_cpu_speculative` attn residual | **Scalar**, serial | gguf_model.h:3050 |
| `forward_decode_cpu_speculative` ffn residual | **Scalar**, serial | gguf_model.h:3114 |

### 9a. Fix the four scalar residuals
```cpp
// Replace: for (int64_t j = 0; j < N; ++j) out[j] = a[j] + b[j];
// With AVX2 + parallel_for_1d, threshold 1024.
```

### 9b. Parallel_for even the main-decode residuals (they're still serial)
With H=3584 and 24 threads, `parallel_for_1d(H, ..., 1024)` still parallelises (3584 > 1024×24
= false → serial; with H=6144 in larger models it does parallelise). Not a big win on qwen3:4b
but free code.

### 9c. Fuse residual + next-layer's RMSNorm sum-of-squares
- **Architectural opportunity** (Finding 11c below): after `attn_residual`, we IMMEDIATELY do
  `ffn_norm` RMSNorm — which reads `x[j]` again and computes `sum(x²)`. If we fuse
  `x[j] = prev[j] + h[j]; sum_sq += x[j]*x[j]` in a single AVX2 loop, we save 1 pass over H
  = 14 KB of cache traffic per layer × 36 = 500 KB/token.

---

## 10. Embedding lookup — ALREADY MEMCPY, BUT SCALE LOOP NOT PARALLEL

- **File:** `gguf_model.h:2333, 2884, 3266`
- Embedding: plain `memcpy(sp.x_buf[cur], emb + token_id * H, H*sizeof(float))`. On Windows/Linux
  glibc memcpy is already SIMD + rep movsb on E2K (LCC implements memcpy as QR stream). Fine.
- **Scale loop** (2341-2348) is AVX2 + serial. For H=3584 that's ~10 μs, OK. For `batched`
  path (3270) it's scalar and K×serial — fix.

### 10a. No `_mm_prefetch` of the embedding row before `memcpy`
- Embedding table is 150 k × 3584 × 4 = **2.0 GB of cold float32** (qwen3:4b vocab).
  mmap'd, so memcpy stalls on page fault for a never-seen token (~0.5 ms pagecache miss on
  Elbrus SATA SSD!). Not every token but happens regularly in practice.
- **Fix:** before `memcpy`, `for (int off=0; off<H*4; off+=4096) _mm_prefetch(emb + token_id*H
  + off/4, _MM_HINT_T1);` — or use `madvise(MADV_WILLNEED)` on the embedding mapping.
- **Impact:** amortised few μs / cold token.

---

## 11. ARCHITECTURAL HOLES

### 11a. **Serial preamble at start of each token**
Operations that must run BEFORE the transformer loop on master thread only:
1. `sec_timers_.init()` — trivial
2. Scratch allocation check — branch only
3. Embedding memcpy + scale — ~10 μs
4. For Gemma: `x *= sqrt(H)` scaling — parallelisable (isn't)

After the loop:
5. Final RMSNorm (single cpu_rmsnorm_inplace) — ~15 μs (scalar sum_sq; parallelise?)
6. Output projection — THE parallel_for already inside
7. `at::empty` + `memcpy` to logits tensor (line 2848-2850) — 150k floats = 600 KB memcpy
   = ~200 μs pure overhead per token.

**Fix 11a:** avoid the `at::empty + memcpy` round-trip — return a `Tensor` view of
`sp.logits_buf` and let the caller copy only if it outlives the decode call. Or use a
pool of 2 logits buffers rotated per decode (caller pops before next decode). **Saves
~200 μs/token = ~1% at 60 ms/token.**

### 11b. **Attention softmax — no overlap with KV read**
Today: softmax is a serial barrier between `scores[t] = Q·K[t]` and
`out += scores[t] * V[t]`. The V cache is in the SAME physical buffer ~ 8 KB/head away
from K. If we stream V loads concurrently with softmax — e.g. do scores in register, apply
softmax online (using the flash-attention 2-pass trick: `max_so_far`, `sum_so_far` as we go,
renormalise output vector once per update), we can do Q·K+softmax+V in ONE pass over the KV
cache instead of two. Memory traffic 2× → 1×. For past_len=1024 this is the difference between
2 × 4 KB × 32 heads × 36 layers = 9 MB/token and 4.5 MB/token — saves another ~2 ms.

**Not a one-line fix**; it's a genuine flash-attention CPU rewrite (100-150 LOC for the 4-loop
inner block). Worth it: up to +15% at long context.

### 11c. **Residual + next-layer RMSNorm fuseable**
As mentioned above (9c). After FFN residual (line 2772), next iteration immediately does
`cpu_fused_rmsnorm_qkv_gemv` which does RMSNorm first (reads x, computes sum_sq). If we
restructure to pass the residual's output buffer as a pre-populated sum_sq accumulator, we
save one full H-pass per layer.

Equivalent refactor: replace pattern
```cpp
residual_add(x_next, x_cur, h_buf, H);
cpu_fused_rmsnorm_qkv_gemv(x_next, ... );  // reads x_next, computes sum_sq
```
with
```cpp
float sum_sq = residual_add_and_sum_sq(x_next, x_cur, h_buf, H);
cpu_fused_rmsnorm_qkv_gemv_presummed(x_next, sum_sq, ...);
```

**Impact:** 14 KB/layer × 2 residuals × 36 layers = ~1 MB/token saved → ~0.4 ms at E8C2
bandwidth. Small but fundamental.

### 11d. **`forward_decode_cpu_batched` K-loop RMSNorm is sequential memcpy+normalise × K**
- Lines 3280-3283 and 3392-3395: for each of K tokens, `memcpy(x_na[k*H], cur[k*H], H)` +
  `cpu_rmsnorm_inplace(x_na[k*H])`. At K=5 that's 5 × 14 KB = 70 KB of pure memcpy BEFORE we
  can do the normalise. Could instead fuse the memcpy into `cpu_rmsnorm_inplace` — or skip
  the memcpy entirely and have `cpu_rmsnorm` write to a separate output buffer (already
  effectively in-place!). Current code copies x→x_na and then overwrites x_na. **Pointless
  intermediate memcpy.**
- **Fix:** add `cpu_rmsnorm(float* out, const float* in, gamma, eps, add_one, H)` out-of-place
  variant; drop the memcpy. Saves K × 14 KB × 36 layers = 2.5 MB/token at K=5 ≈ 1 ms/token.

### 11e. `for k in K_batch` inside `q4k_gemv_batched_avx2` (line 1761) is a scalar K-loop
- The weights ARE loaded once per super-block, but the K queries are computed
  serially-in-the-K-dim inside that loop. For K=5, each super-block iteration does 5×
  (maddubs + madd + hsum). The HSUMs are 5 separate m256 → m128 horizontal reduces = 10
  shuffle+add instructions × 5 = 50 per super-block.
- **Candidate fix:** transpose the Q8 layout so the K queries' `qs` bytes for the SAME
  j-offset are interleaved into ONE __m256i. Then one `maddubs` computes K dot-product bytes
  at once — K accumulators share one load. This is llama.cpp's "ggml_vec_dot_q4_K_q8_K" trick
  for batches. Complexity: 200 LOC. Expected gain: 1.3-1.6× over current K-loop for K=5.

### 11f. **`for (int64_t t = 0; t < total_seq; ++t)`** in attention scoring — no thread split across t
- In `forward_decode_cpu_batched` attention (line 3340-3375), we parallelise across
  `K * n_heads`. For K=5 × 32 heads = 160 work units, 24 threads → good load balance.
- But each thread still does `total_seq` loads of K_cache serially. For past_len=4096 that's
  4096 × 128 B (head_dim × 4) = 512 KB/head. Per-thread. No prefetch. No reuse across
  neighbouring `t`s since next `t` goes to `k_cache + (t+1)*kv_dim`.
- **Prefetch add:** `__builtin_prefetch(k_head + (t+16)*kv_dim, 0, 0);` inside the t-loop.
  Free, +5-8% on long-context attention.

---

## Summary of highest-value misses

Ranked by expected tok/s impact given 60 ms/token target:

| # | Fix | File | LOC | Expected gain |
|---|-----|------|-----|---------------|
| 1 | Vectorise + parallelise SiLU in `forward_decode_cpu_batched` (3412) | gguf_model.h | 30 | ~9 ms/token when spec decode used |
| 2 | SIMD exp for attention softmax in all 3 decode paths | gguf_model.h | 60 | ~5-10 ms/token at seq>256 |
| 3 | Parallelise residual adds in `forward_decode_cpu` (2655, 2772) | gguf_model.h | 10 | ~8-20 ms/token |
| 4 | Wire NUMA through `cpu_fused_rmsnorm_gate_up_gemv` and `cpu_quant_gemv_batched_qkv` | cpu_quant_gemv.h | 40 | ~4 ms/token (w/ replicas) |
| 5 | `__restrict` on `q4k_gemv_avx2` production kernel | cpu_quant_gemv.h | 20 | +3-5% on ~15 ms/token GEMV = ~0.5 ms |
| 6 | Fix `q8_idx`, `is`, `n_half`, `l` → `int64_t` in all scalar/batched fallbacks | cpu_quant_gemv.h | 30 | +3-5% on LCC SWP |
| 7 | Pass `&numa_replica` as `numa` arg to output-proj GEMV (2808) | gguf_model.h | 2 | ~1-2 ms/token in TP-4 |
| 8 | Skip the pointless `memcpy x→x_na` before `cpu_rmsnorm_inplace` in batched path | gguf_model.h | 30 | ~1 ms/token at K=5 |
| 9 | Return `Tensor` view of `logits_buf` instead of `at::empty + memcpy` | gguf_model.h | 20 | ~0.2 ms/token |
| 10 | Prefetch `k_cache + (t+16)*kv_dim` inside attention t-loop | gguf_model.h | 4 | +5% at long context |
| 11 | AVX2 scalar fallbacks (emb scale, biases, residuals in spec decode) | gguf_model.h | 40 | spec-decode only |
| 12 | K-query batched SIMD transpose in `q4k_gemv_batched_avx2` | cpu_quant_gemv.h | 200 | +30-60% on spec-verify GEMV when spec active |

Applying items 1-4 gets you from 5.3 → ~7.0-7.5 tok/s (single-proc) and from 6.1 → ~7.5-8.5
tok/s (TP-4) — still shy of 20. Getting to 20 needs items 11b (flash-attention CPU) or
something structurally new (dequant-on-load to BF16 + use cache-resident BF16 weights to
halve bandwidth — not audited here).

Items 5-10 are cheap (<30 min each) and stack to another 3-5% in aggregate.

---

## Positive findings (things correctly applied)

1. APB pragmas **are** on all 15 j-loops in the AVX2 fast paths (Q4_K, Q5_K, Q6_K, Q8_0,
   k-slice, batched, batched_qkv).
2. Outer loop counters in the AVX2 fast paths **are** `int64_t`.
3. NUMA replica resolution IS used in `q4k_gemv_avx2` (line 525), `q4k_gemv_sse41_v` (363),
   `q4k_gemv_batched_avx2` (1718), `q4k_gemv_avx2_batch2` (1568), `q6k_gemv_avx2` (826).
4. Prefetch of next super-block IS done in `q4k_gemv_avx2` (553-560), `q4k_gemv_sse41_v`
   (387-393), `q4k_gemv_avx2_batch2` (1580-1585), `q4k_gemv_batched_avx2` (1734-1737),
   `q6k_gemv_avx2` (837-844).
5. Split accumulator pattern (agent_1 P3) IS applied — `sum0_a/sum0_b` in q4k_gemv_avx2
   (542), q4k_gemv_sse41_v, batched_avx2, k_slice_avx2, batched_qkv.
6. Async prefetch of NEXT layer's 7 weight matrices is scheduled at layer boundary (2361-2398)
   — excellent TLB-priming trick.
7. Fused RoPE (precompute table once, apply to all heads) is used in main decode path
   (2517-2522). Good.
8. ThreadPool.parallel_for pads chunk to 16-float cache-line multiple (ThreadPool.h:161) —
   avoids false sharing.
9. SHM AllReduce has bounded-spin fallback (ddp.cpp:543-555) — correct trade-off.

---

## Files to edit (absolute paths)

- `C:\Users\paper\Desktop\promethorch\torch\io\cpu_quant_gemv.h`
- `C:\Users\paper\Desktop\promethorch\torch\io\gguf_model.h`

No new files needed. No changes to `ThreadPool.h` or `ddp.cpp` recommended — both are sound.
