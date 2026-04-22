# Agent 5 — Alternative Strategies (Q4_K_M qwen3:4b на E8C2)

**Baseline:** 5.5 tok/s (TP 4-proc × 7c, split output_proj + prefetch + gate/up fuse).
**Исходные данные:** BENCH_ELBRUS.md показывает serial floor **107 ms/token → ceiling 9.3 tok/s** (Amdahl fit 1-proc). 1-proc best = 3.8 tok/s. TP 4-proc = 5.5 tok/s. 72 AllReduce × ~10ms = 720 ms/token overhead на TP — это уже 56% бюджета на 5.5 tok/s. Декод = **bandwidth-bound + AllReduce-bound**, per-thread GEMV compute уже 94% peak на single-core (EML 67.9 / 72 GFLOPS).

**Цель:** 10-15 tok/s = 2-3× speedup.

---

## A. Pre-dequant Q4_K → Q8_0 at load time

**Физика:**
- qwen3:4b веса: Q4_K = **2.5 GB**; Q8_0 = **4.5 GB** (1.8×); FP16 = **8 GB** (3.2×).
- E8C2: 4 × DDR4 каналы по ~20 GB/s = **~80 GB/s агрегат**, реально measured decode demand ~8 GB/s (BENCH строка 86: "5% от 100-160 GB/s"), **не saturated**. Значит bandwidth НЕ кэпирует при переходе на Q8.
- Compute: Q4_K GEMV сейчас уже на Q8-inner-loop через `quantize_x_q8` + `maddubs_epi16` (`cpu_quant_gemv.h:219-295`). Horizontal-sum overhead per 64 элементов = ~10-15 cycles; dequant nibble extraction = **уже оптимизирован**. Переход на Q8-weights уберёт ~25-30% cycles (nibble mask+shift+get_scale_min_k4) — не 80%.
- Q8_0 layout: 34B/32val → 8.5 bits/weight (вместо 4.5). `q8_0_gemv_avx2` уже существует (`cpu_quant_gemv.h:539-581`).

**Риск:** 4.5 GB весов помещаются в 125 GB RAM, но при `numactl --interleave=all` на 4 × 31 GB DDR/node — тоже ок. L2 Эльбрус 8C2 = 512 KB/core × 8 cores/chip = 4 MB. Ни Q4 ни Q8 не помещаются в L2, **обоих cache-miss-bound одинаково**. Speedup в основном от ~25% меньше compute cycles → ~1.25× local.

**FP16 sub-вариант:** сложнее — E8C2 нативного FP16 FMA НЕТ (только FP32/FP64 в 6-канальном VLIW). Conversion FP16→FP32 добавит cycles обратно. **Dropped.**

**Реалистичный прирост:** single-proc 3.8 → 4.5-4.7 tok/s. TP 5.5 → 6.5 tok/s.

---

## B. Partial model FP16 hot layers (первые 2 слоя)

**Физика:**
- qwen3:4b = 36 layers. Первые 2 слоя = 5.5% весов (~140 MB FP16).
- Profile BENCH table lines 48-56: все 36 слоёв ~симметричны в compute, **нет "bottleneck layer"**. Я не вижу эмпирических данных что первые 2 слоя съедают 40% — это fit-assumption из ТЗ, но `PT_PROFILE_LAYER` на сегодня усредняет по всем.
- Даже если 10% прироста на 2/36 слоёв → глобально 0.55% speedup.

**Оценка без измерений: плохо мотивировано.** Дополнительно FP16 без HW-FMA даст регрессию (см. A).

**Реалистичный прирост:** 0-2%. **Skip.**

---

## C. Speculative decoding с batched verify (4 draft tokens)

**Физика:**
- На batch=1 decode model-weights читаются **1 раз на token** = 2.5 GB / token. При batch=4 (verify) = 2.5 GB / 4 tokens.
- GEMV становится GEMM (M=4, N=2560, K=2560): `EML cblas_sgemm` на этой форме = ~63 GFLOPS single-core (BENCH 182), 4× лучше чем M=1 (12.9 GFLOPS).
- **Критично:** нужен draft model (e.g., TinyLlama или qwen3:0.6b). Acceptance rate для qwen3-family обычно 60-75% при T=0. Expected accept = 0.7 × 4 = 2.8 tokens/batch, т.е. throughput = 2.8 × 4 = **11.2 tok/s** target.
- AllReduce остаётся 72×, но теперь на batch=4 — латентность AllReduce почти константна для 4×H=10KB vs 1×H=2.5KB, грубо без изменений → **AllReduce overhead **не** amortize'тся** по batch, продолжает давить TP. Значит spec даёт огромный выигрыш на 1-proc, умеренный на TP.

**Риск:**
- Draft model необходим (gguf Q4_K_M). qwen3:0.6b подходит.
- KV-cache rewind после rejected tokens — нетривиально в текущем gguf_model.h (прямой write в `tp_.k_cache_local[i]` at `past_len`; откат = просто `past_len -= delta`, уже поддерживается логически).
- Prefill for draft.

**Реалистичный прирост:** 1-proc 3.8 → 7-9 tok/s, TP 5.5 → 8-10 tok/s. **Сильнейший вариант для decode.**

---

## D. MoE / Mixture quantization

qwen3:4b dense. **Skip.**

---

## E. Offload в EML_MT dense GEMM при batch>1

Повторяется с C — batch=1 GEMV не использует EML_MT (sgemm single-core = 67.9 GFLOPS, но мы уже на 94%). При batched decode (C) — EML_MT peak 1840 GFLOPS на 32 cores, для M=4 × K=2560 × N=2560 (~50 MFLOP) latency = 50e6 / 1.8e12 = 28 μs compute + dispatch. **Упирается в EML_MT spinup/NUMA-sync overhead** (~500 μs/call measured), который на 180 calls/token = **90 ms/token ×1 = не помогает** — overhead сам по себе жрёт бюджет.

**Альтернатива:** использовать EML_MT **per-rank** (внутри процесса) для GEMV превращая его в GEMM через ручной **tiled col-stride** (N-rows → 32 NUMA-local row-tiles × single-core EML). Т.е. свой `pthread-per-tile` wrapper — уже помечено в BENCH "work already marked in roadmap".

**Реалистичный прирост:** 1-proc 3.8 → 5.5-7 tok/s (замена текущего `c10::parallel_for` на pthread-per-tile с EML per-thread). TP 5.5 → ? (AllReduce overhead остаётся, net gain ограничен).

---

## F. VLIW-aware loop restructuring (multi-row interleave, 4-6 accumulators/thread)

**Физика E8C2 VLIW:**
- 6 независимых каналов × 128-bit SIMD FMA.
- Текущий Q4_K GEMV в `cpu_quant_gemv.h:345-461` делает **2 rows simultaneously** (уже!) — 2 accumulators `sum0`, `sum1`.
- Теоретический peak LCC-AVX2-трансляции ограничен тем, сколько **independent dependency chains** компилятор видит. 2 chains на 6-channel VLIW = ~33% utilization. Расширение до 4-6 rows should unlock ~1.7-2× per-thread GFLOPS.

**Проверено в BENCH:** single-thread gate_up = 1402 ms → 2.6 GFLOPs/thread vs peak 8 GFLOPS = **33% utilization — точно совпадает с 2-channel VLIW usage!**

**Реализация:**
- `cpu_quant_gemv.h:345` поменять `for (; n + 1 < end; n += 2)` на `n += 4`.
- 4 независимых `sum0..sum3` + 4 `row0..row3` + x_q8 load shared (уже в 2-row путь), **x_q8 cache line lost между rows 2 и 3** — возможно, нужен второй prefetch-pass.
- Компиляция через LCC-1.29 с `-fopenmp-simd -O3 -funroll-loops=4`.

**Риск:** регистровое давление. 2-row уже использует ~20 XMM-equivalents на E8C2 mapping. 4-row может spill'ить в стек → проигрыш. Нужен замер через `run_decode.sh` с 2-row vs 4-row branch.

**Реалистичный прирост:** 1-proc 3.8 → 5.5-7 tok/s (+45-85%). TP 5.5 → 7-9 tok/s. **Самый дешёвый по времени.**

---

## G. Weight reshuffle: Q4_K → aligned int8/bf16 blocks (64-byte align, at load time)

**Физика:**
- Current Q4_K block = 144 B. 144 не делится на 64-byte cache line → каждый blocк **страдает 2.25 cache-line loads**. Если reshuffle в 128-B int8 blocks (32 val × int8 + 2B scale padded to 16B), получается 2 cache lines ровно, **~12% меньше cache traffic**.
- Реально: Q4_K прочитан через `__builtin_prefetch` trio (lines 359-366 в `cpu_quant_gemv.h`) — уже уменьшает stall. Доп benefit от align = marginal.
- **bf16 на E8C2:** нет нативного FMA — всё равно конверсия в FP32. Только memory-footprint trade, но bf16 = 2× Q4 = сначала хуже.

**Реалистичный прирост:** 0-5%. **Низкий приоритет.**

---

## Финальная таблица

| Стратегия | Speedup (realistic) | Риск 1-5 | Реализация | Приоритет |
|-----------|---------------------|----------|-----------|-----------|
| **A**. Q4→Q8 pre-dequant | ×1.2 | 2 | 1 день (adapter в load path) | **3** |
| **B**. FP16 hot layers | ×1.00-1.02 | 4 (мотивация слабая) | 3 дня | **skip** |
| **C**. Speculative + batched verify | **×2.0-2.6** | 4 (draft model + KV rewind) | 3-5 дней | **1** |
| **D**. MoE/Mixture | — | — | — | **skip** (dense) |
| **E**. EML_MT per-rank via pthread-per-tile | ×1.2-1.5 (1-proc) | 3 (thread spawning overhead) | 2 дня | **4** |
| **F**. VLIW 4-6 accumulators per thread | **×1.5-1.8** | 2 (register spill check) | **2-4 часа** | **2** |
| **G**. 64B-align reshuffle | ×1.0-1.05 | 2 | 1 день | **skip** |

**TP-specific note:** Strategies A, F, E улучшают per-thread compute, но TP baseline 5.5 tok/s уже **AllReduce-bound** (720 ms overhead / 1200 ms total = 60%). Чтобы выжать 2× на TP нужно **ещё и** сократить AllReduce: либо fewer calls per token (gate/up/down fuse AllReduce в один — **1 AllReduce вместо 2 на FFN**), либо shared-memory SIMD broadcast вместо copy-to-shm-then-read.

---

## Топ-3 к внедрению прямо сейчас

### 1. VLIW 4-row interleave (стратегия F)  — **2-4 часа**
Цель: per-thread 2.6 → 4.2 GFLOPS, 1-proc 3.8 → ~6 tok/s, TP 5.5 → ~7.5 tok/s.

Checklist:
- [ ] `torch/io/cpu_quant_gemv.h:345` — переписать loop `for (; n + 4 <= end; n += 4)` с 4 независимыми accumulators `sum0..sum3`, 4 row ptrs, 4 `__builtin_prefetch` trio's.
- [ ] Add tail handler: `for (; n + 2 <= end; n += 2)` (existing) + single-row tail.
- [ ] Собрать `build_elbrus_cpu/` с `-O3 -funroll-loops`, проверить LCC не жалуется на spill через `-fverbose-asm` grep "regs spilled".
- [ ] Benchmark `run_1proc_elbrus.sh --greedy "test"` — сравнить с baseline 3.8 tok/s на identical seed.
- [ ] Если регрессия — branch N≥1024 → 4-row, N<1024 → 2-row (dispatch via runtime check).

### 2. Speculative decoding с qwen3:0.6b draft (стратегия C) — **3-5 дней**
Цель: 5.5 → 8-10 tok/s.

Checklist:
- [ ] Загрузить `qwen3:0.6b Q4_K_M` как второй `GGUFModel* draft_model_` рядом с main.
- [ ] `torch/io/gguf_model.h:3603` добавить branch: `forward_decode_cpu_tp_speculative(token_id, draft_model_, k=4)`.
- [ ] Draft loop: 4 decode calls на draft, получить `[d1,d2,d3,d4]`.
- [ ] Verify batch: **model-weights прочитать 1 раз**, compute M=4 logits через новый `cpu_quant_gemm_m4(...)` в `cpu_quant_gemv.h` (добавить рядом с `q4k_gemv_avx2`). Layout: 4 × x_q8 вместо одного, те же 2-row weight loops.
- [ ] KV rewind на rejected tokens: `past_len -= rejected_count` (уже работает — cache просто overwritten on next token).
- [ ] Acceptance measurement — log accept rate в `run_logs/spec_*.log`.

### 3. Q4→Q8 pre-dequant at load (стратегия A) — **1 день**
Цель: 3.8 → 4.5 tok/s, TP 5.5 → 6.5 tok/s. Не велик, но простой + стабилизирует baseline для 1 и 2.

Checklist:
- [ ] `torch/io/gguf_dequant.h` — добавить `q4k_to_q8_0_reblock(const void* q4k_data, void* q8_0_out, int64_t n_elements)` (128 bytes qs per Q4 block → 32 Q8_0 blocks).
- [ ] `torch/io/gguf_model.h:886` (upload_quant) — если `PT_PREDEQ_Q8=1` env set, reblock во время load, перезаписать `QuantizedWeight::quant_type = GGML_TYPE_Q8_0`, `row_stride_bytes` пересчитать.
- [ ] `cpu_quant_gemv.h:539` (`q8_0_gemv_avx2`) — убедиться что 2-row interleave применим; добавить если нет.
- [ ] Benchmark `run_1proc_elbrus.sh` + `run_tp_elbrus.sh` с `PT_PREDEQ_Q8=1`.

---

## Зона незнания (требует live Elbrus benchmark)

1. **Реальная per-layer profile на TP.** BENCH строки 48-56 — только 1-proc. Не знаем, какие слои/секции доминируют в TP-режиме, где именно `allreduce_ao_ms` + `allreduce_ffn_ms` реально нагружены. Нужен `PT_PROFILE_TP_LAYER=1` по всем 36 слоям, 40 tokens avg.
2. **LCC register-spill on 4-row interleave.** Без запуска через LCC 1.29 `-O3 -fverbose-asm` и grep "spilled" — не узнаем, spill'ит ли оно в стек. На 6-канальном VLIW с 8 general-purpose 64-bit regs per channel (~48 regs) теоретически поместится, но LCC emits dependent-chain scheduling по-своему.
3. **AllReduce latency breakdown.** BENCH говорит "~10ms/call". Не известно — это memcpy-to-shm bound, cache-coherence traffic bound, или spinning on condvar. Без `perf stat -e cache-misses,page-faults` на live кластере — guess-работа.
4. **Acceptance rate qwen3:0.6b → qwen3:4b.** Это предсказание 0.7 — но может быть 0.4 (низкая корреляция distributions) или 0.85. Speedup от spec decoding = `k × accept_rate / (1 + k × rejection_cost_frac)`. Без real measurement number — gambling.
5. **DDR bandwidth ceiling под max load.** BENCH measured only 8 GB/s demand, claims "100-160 GB/s aggregate peak". Но не mерено sustained read from 4 NUMA nodes одновременно под real decode. Может быть 30-40 GB/s effective из-за inter-chip snoop traffic.
6. **Prefetch distance on E8C2.** `__builtin_prefetch` на E2K emit'ит MOVA-based hint. Оптимальное **расстояние** (next-1 vs next-2 vs next-4 blocks) — не измерено. Текущий код prefetch'ит next-1 на каждой итерации; может быть прибыль от prefetch next-4.

---

## Сводный вердикт

**Прямой путь к 10 tok/s:** F (×1.5-1.8) + C (×2.0-2.6) = композитно ×2.5-3.5, перекрывает цель. F — быстрый win за 2-4 часа, C — больший win за неделю. A добавляется третьим стабилизирующим.

**Не трогать:** B (слабо мотивирован), D (не применим), G (marginal) — эти съедят время без прироста.
