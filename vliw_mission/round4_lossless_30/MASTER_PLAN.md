# MASTER PLAN — Round 4 Lossless 30 tok/s on qwen3:4b

**Дата:** 2026-04-30
**Бейслайн:** TP-4 + Q8 SoA4 = 9.4 tok/s (commit `3399136`)
**Цель:** 30 tok/s lossless (max abs diff < 1e-5 vs Q4_K_M)
**Источники:** Agent A `format_spec_v1.md`, Agent B converter pipeline, Agent C `pt8_reader.h`, Agent D `agent_D_results.md`, Agent E `agent_E_results.md`, Gemini DR Max 8 Q&A.

---

## Конвергенция всех источников — единый путь

| Find | Agent A | Agent D | Agent E | Gemini DR |
|---|---|---|---|---|
| **PT8_Q4_SOA4 обязателен** (0.6875 B/p) | ✓ §10 | ✓ A6 (−22 ms) | — | ✓ Q1 (Mandatory) |
| **Persistent ThreadPool** | — | ✓ A1 (−12 ms) | — | ✓ Q4 (Modulo Sched fail) |
| **Strict NUMA mbind/MPOL_BIND** | hint только | A4/A5 | §5.2 5-th proc | ✓ Q7 (must use libnuma) |
| **Kernel fusion + extern "C" + restrict** | — | A4 (−5 ms) | — | ✓ Q4 (LCC SWP) |
| **APB / manual prefetch** | — | A5 (−5 ms) | — | ✓ Q4 |
| **Self-Spec (LayerSkip/CLaSp)** vs draft model | — | A3 fallback | qwen3:0.6B → 0% | ✓ Q3 (Self-Spec PREFERRED) |
| **128*sum_q вынести из горячего цикла** | — | — | — | ✓ Q2 (loop-invariant) |
| **Huge Pages 2 MiB на data section** | hint | — | — | ✓ Q5 |
| **Lossless = Q4_K_M decode equiv** | ✓ Trakt B | ✓ guard 1e-5 | ✓ greedy bit-exact | implicit |

**Ключевое расхождение:** Agent E считает что 30 tok/s НЕ достижимо без EAGLE finetune (требует A100 + days training). Gemini Q3 говорит **Self-Speculative (LayerSkip)** даёт 1.8-2.1× БЕЗ обучения. Это лучший путь — Self-Spec реализуется в коде, без обучения, дает overshoot для 30.

---

## Расчёт ROI к 30 tok/s

baseline = 9.4 tok/s (106 ms/token)

| # | Step | save (ms) | cum (ms) | cum (tok/s) | confidence |
|---|------|----:|----:|----:|---|
| 0 | baseline | 0 | 106 | 9.4 | DONE |
| 1 | A1 Persistent ThreadPool | 12 | 94 | 10.6 | HIGH |
| 2 | NUMA mbind strict | 8 | 86 | 11.6 | HIGH (Gemini Q7) |
| 3 | A6 PT8_Q4_SOA4 kernel | 22 | 64 | 15.6 | HIGH (BW ceiling 49) |
| 4 | A4 Kernel fusion (QKV+gate/up) | 5 | 59 | 16.9 | HIGH |
| 5 | A5 APB + manual prefetch | 5 | 54 | 18.5 | MED |
| 6 | A2 AllReduce reduce-scatter | 6 | 48 | 20.8 | MED |
| 7 | A8 tail SIMD (softmax + alloc reuse) | 5 | 43 | 23.3 | MED |
| 8 | LCC SWP cleanup (extern "C" + restrict) | 4 | 39 | 25.6 | MED-HIGH |
| 9 | Self-Speculative LayerSkip ×1.4 | — | 28 | **35.7** ★ | MED |

Без шага 9 (kernel-only): **~25 tok/s** реалистично.
С шагом 9 (LayerSkip Self-Spec): **30+ tok/s** ✅

---

## STEPS — Implementation Order

Каждый шаг: код → sync на Эльбрус → build → 3-run measure → commit с цифрой → следующий.
**Lossless guard:** max abs diff logits < 1e-5 vs Q4_K_M baseline на 50 токенов после каждого шага.

### Step 0 — Baseline confirmation (zero code, just measure)
- 3 runs `PT_Q8_SOA=1 ./scripts/run_tp_elbrus.sh --greedy "Hello"`
- Цель: подтвердить 9.4 ± 0.3 tok/s. Если ниже — дебагнуть config drift.
- Acceptance: median ≥ 9.0 tok/s.
- Commit: skip (no code change).

### Step 1 — Persistent ThreadPool (A1) ⏱ 1 session
**Files:**
- `c10/util/Futex.h` (new, 80 LoC) — Linux + Windows wrappers
- `c10/util/ThreadPool.h` (rewrite ~280 LoC) — broadcast descriptor + per-worker ack slots + futex gen
- `examples/benchmarks/threadpool_overhead_bench.cpp` (already exists, add to CMakeLists)

**Microbench gate:** `bench_threadpool_overhead 100000` ≤ 8 µs/call (current ~100 µs).

**Acceptance:**
- Microbench < 8 µs ✓
- TP-4 SoA test 3-run median ≥ 10.5 tok/s
- max abs diff = 0 (no math change)
- No deadlock в 1000-iter test cycle

**Risk mitigation:** известный deadlock в прошлой попытке. tsan-clean на x86 build ПЕРЕД интеграцией. PT_TP_TIMEOUT_MS=2000 watchdog.

**Commit:** `perf(c10): persistent ThreadPool broadcast pool — TP-4 9.4 → X.X tok/s`

---

### Step 2 — Strict NUMA mbind (Gemini Q7) ⏱ 0.5 session
**Files:**
- `torch/io/numa_weight_replica.h` patch — `numa_replicate` использует `mbind(MPOL_BIND, &nodemask)` вместо `memcpy + first-touch`
- `torch/io/gguf_model.h` — replicated weights path
- new: `c10/util/numa_alloc.h` обёртка для `mbind`/`set_mempolicy`

**Acceptance:**
- `numastat -p $PID` показывает per-rank weights ровно на 1 node, 0 на других
- TP-4 3-run median ≥ 11.5 tok/s
- max abs diff = 0

**Commit:** `perf(numa): strict mbind MPOL_BIND для replicated weights — TP-4 X.X → Y.Y tok/s`

---

### Step 3 — PT8_Q8_SOA4 converter + loader (Agents A/B/C unblock) ⏱ 1 session
Это infrastructure. Само по себе перфа не даст, но открывает Step 4.

**Files:**
- `tools/gguf2pt8/encoders/encode_q4k_to_q8soa4.cpp` (new, ~150 LoC) — реализует `EncoderRegistry::q4_k` для Q8_SOA4_F16 dtype (FP16 headers)
- `tools/gguf2pt8/CMakeLists.txt` — wire build target
- `torch/io/pt8_reader.h` — reader from Agent C (already there, just verify/adapt)
- `torch/io/gguf_model.h` — auto-detect .pt8 magic, branch loader

**Verification:**
- `gguf2pt8 qwen3-4b-Q4_K_M.gguf -o qwen3-4b.pt8 --verify` — диф логитов < 1e-5
- Cold start `PT_FORMAT=pt8 ./scripts/run_tp_elbrus.sh` — экономит 7s repack

**Acceptance:**
- Conversion < 30s, file size ~5.25 GB
- Verify diff < 1e-5
- TP-4 на pt8 = TP-4 на gguf (no perf delta — Q8_SOA4_F16 same kernel)

**Commit:** `feat(io): PT8 binary format converter + zero-copy mmap loader — equivalent perf, eliminates 7s cold start repack`

---

### Step 4 — PT8_Q4_SOA4 kernel (A6 + Gemini Q1) ⏱ 2 sessions
**Files:**
- new microbench: `examples/benchmarks/q4_soa4_microbench.c` (clone of q8_soa4_microbench)
- `torch/io/q4_soa4_repack.h` (new, ~400 LoC) — Q4_SoA4 layout (88 B / 128 weights), `repack_q4k_to_q4soa4`, `q4_soa4_gemv` kernel
- `tools/gguf2pt8/encoders/encode_q4k_to_q4soa4.cpp` (new, ~150 LoC)
- `torch/io/cpu_quant_gemv.h` — dispatch третья ветка
- `torch/io/gguf_model.h` — SoA branches в forward_decode_cpu_tp используют q4_soa4_gemv когда tl.q4_soa.valid

**Microbench gate (BEFORE production integration):**
- `q4_soa4_microbench` ≤ 0.7 ms/GEMV K=2560 N=2432 single-core (current Q8 = 1.21 ms)
- Если 0.7-0.9 ms → still net win, proceed
- Если > 0.9 ms → drop, оставить Q8 SoA4

**Acceptance:**
- TP-4 3-run median ≥ 17 tok/s
- max abs diff < 1e-5 (FP16 of d×sc rounding ≈ 2^-11 relative — within budget)

**Commit:** `perf(io): PT8_Q4_SOA4 kernel — 0.6875 B/param, BW ceiling 49 tok/s — TP-4 X.X → Y.Y tok/s`

---

### Step 5 — Kernel fusion (A4) ⏱ 1 session
**Files:**
- `torch/io/q4_soa4_repack.h` — добавить `q4_soa4_qkv_gemv` (3-way fused) и `q4_soa4_gateup_gemv` (2-way fused) — shared a_b16 streaming
- `torch/io/gguf_model.h:4602-4613` — заменить 3 отдельных q4_soa4_gemv на 1 fused QKV
- `torch/io/gguf_model.h:4884-4896` — заменить 2 отдельных на 1 fused gate+up

**Acceptance:**
- TP-4 3-run median ≥ 19 tok/s
- max abs diff < 1e-5

**Commit:** `perf(io): fused QKV + gate/up Q4_SoA4 GEMV — single quant_activation, shared a_b16 — TP-4 X.X → Y.Y tok/s`

---

### Step 6 — LCC SWP cleanup (Gemini Q4) ⏱ 0.5 session
**Files:**
- `torch/io/q4_soa4_repack.h` — горячий цикл `q4_soa4_gemv_inner` вынести в `extern "C"` функцию с `__restrict__` на pointers + `_Pragma("loop count(8) ivdep") _Pragma("pipeline enable")` + remove C++ lambda capture
- Verify через `dprof -m TICKS,EXEC,BUB_E2` что Bubble count drops

**Acceptance:**
- Microbench q4_soa4 ≤ 0.55 ms (vs 0.7 ms previous)
- TP-4 3-run median ≥ 20 tok/s
- max abs diff = 0 (no math change)

**Commit:** `perf(elbrus): SWP-friendly extern "C" inner loop + __restrict__ — TP-4 X.X → Y.Y tok/s`

---

### Step 7 — APB + manual prefetch (A5) ⏱ 0.5 session
**Files:**
- `torch/io/q4_soa4_repack.h` — `__builtin_prefetch(b+1, 0, 2)` на L2 + `__builtin_prefetch(b+16, 0, 1)` на L3
- Add `_Pragma("loop count(N)")` на b-loop

**Acceptance:**
- TP-4 3-run median ≥ 22 tok/s
- max abs diff = 0

**Commit:** `perf(elbrus): manual L2/L3 prefetch on Q4_SoA4 — TP-4 X.X → Y.Y tok/s`

---

### Step 8 — AllReduce reduce-scatter (A2) ⏱ 1 session
**Files:**
- `torch/distributed/ddp.cpp` — реализовать `reduce_scatter_inplace` mirror к `all_reduce_inplace`
- `torch/io/gguf_model.h` — заменить FFN-down AllReduce на RS+AG

**Acceptance:**
- TP-4 3-run median ≥ 24 tok/s
- max abs diff < 1e-5 (fp32 reorder)

**Commit:** `perf(distributed): reduce-scatter for FFN-down AllReduce — TP-4 X.X → Y.Y tok/s`

---

### Step 9 — Tail SIMD + 128*sum_q hoist (A8 + Gemini Q2) ⏱ 0.5 session
**Files:**
- `torch/io/cpu_quant_gemv.h` — pre-compute `128 * sum_q` в repack stage, не в горячем цикле (Gemini Q2 finding)
- `aten/src/ATen/native/cpu/MathOps.h` — AVX2 softmax уже есть, привязать в attention scores
- `torch/io/gguf_model.h` — pre-allocate `tp_.silu_local_buf`, `tp_.x_normed_buf` (no per-layer alloc)
- Cephes SiLU из `VectorizedOps.h`

**Acceptance:**
- TP-4 3-run median ≥ 26 tok/s
- max abs diff < 1e-5

**Commit:** `perf(elbrus): hoist 128*sum_q + SIMD softmax/SiLU + buffer reuse — TP-4 X.X → Y.Y tok/s`

---

### Step 10 — Self-Speculative LayerSkip (Gemini Q3) ⏱ 2 sessions
Gemini рекомендует LayerSkip / CLaSp над traditional 2-model spec.
- Draft = первые 8-10 слоев qwen3:4b (early exit)
- Verify = full 36 layers батчем по K=4

**Files:**
- `torch/io/self_spec_decode.h` (new, ~300 LoC) — early-exit draft + batched verify
- `torch/io/gguf_model.h` — branch на PT_SELF_SPEC=1
- `tests/io/test_self_spec_quality.cpp` — bit-exact greedy regression

**Acceptance:**
- Bit-exact greedy ✓
- TP-4 3-run median ≥ 30 tok/s ✅
- accept rate ≥ 0.5 на 5 prompt типах (code/prose/math/ru/structured)

**Commit:** `perf(io): Self-Speculative LayerSkip decoding — TP-4 X.X → Y.Y tok/s`

---

## Risk register

| Risk | Mitigation |
|---|---|
| Persistent ThreadPool deadlock (Round 3 incident) | tsan microbench first; PT_TP_TIMEOUT_MS=2000 watchdog |
| Q4_SoA4 nibble unpack > 2 cycles | Microbench gate (Step 4) — abort if > 0.9 ms |
| FP16 d×sc overflow на edge tensors | Encoder runtime check, fall back to FP32 hdr per-tensor |
| AllReduce reduce-scatter race | PT_DDP_LOG=1 payloads on small examples first |
| LayerSkip quality drop > 1e-5 | Bit-exact greedy test mandatory; revert if fail |
| LCC SWP не активируется | dprof verify per step 6 |

---

## Implementation timeline

10 steps × 0.5-2 sessions each = **~9 sessions total**.
Single overnight session can cover Steps 1-3 (infrastructure), then iterate
2-3 steps per session.

**Юзер уходит спать. Я начинаю STEP 1 немедленно.**
