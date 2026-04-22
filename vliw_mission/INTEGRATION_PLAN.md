# VLIW MISSION — финальный план интеграции (6/6 агентов)

**Дата:** 2026-04-22
**Статус источников:** 6 Opus-отчётов готовы, Gemini 3.1 Pro + Deep Research заблокированы (ключ revoked).

---

## Что мы на самом деле узнали

### Карта bottleneck'ов (перевернулась по сравнению с гипотезой)

| Слой | Исходное предположение | Реальность по агентам |
|------|------------------------|----------------------|
| Memory BW | Bottleneck (13.75 GB/s из ~20) | **Не bottleneck.** 5.3% от 273 GB/s aggregate, 21% от 68 GB/s single-chip. Потолок 25-104 tok/s. |
| SIMD compute | Bottleneck (2/6 VLIW channels) | **Полу-bottleneck.** 33% util. Структурный fix (SIMD accumulators) = +15-30%. |
| Serial code | Минор | **Главный bottleneck на 1-proc.** Amdahl floor 107 ms/token = 9.3 tok/s. RMSNorm/RoPE/bias serial. |
| AllReduce (TP) | Главный | Транспорт 1.3 MB/token ≈ 7 MB/s — незначим. Но **spin-wait 100% CPU на idle ranks** реален. |
| NUMA placement | Работает | **`PT_NUMA_REPLICATE=1` НИКОГДА не был включён** в script'е. 30% бы cross-chip (LM head 219 MB/tok + QKV 264 MB/tok). |
| Tool choice | AVX2 через LCC | **AVX2 плохой ABI для VLIW.** ilyakurdyukov: SSE2/SSSE3/SSE4.1 LCC транслирует правильно. |

### Reference point из публичного мира
- E2Kports/llama.cpp: **6.7 tok/s на Elbrus-16C** (Alpaca-7B Q4_0, 8 threads, ~5.2 GB). Заморожен 2023.
- Наш текущий: **5.5 tok/s на E8C2** (qwen3:4b Q4_K_M, 2.5 GB). Более сложный формат.
- **Implication:** если сделаем E2K-нативный вместо AVX2-через-LCC — 10-15 tok/s достижимо.

---

## Интеграционный план — по фазам

### Phase 0 — Zero-code wins (5 минут, сейчас)

**Файл: `scripts/run_1proc_elbrus.sh`**

```diff
-OMP_NUM_THREADS=24 \
-OMP_PLACES=cores OMP_PROC_BIND=close \
-numactl --interleave=all \
+OMP_NUM_THREADS=24 \
+PT_NUMA_REPLICATE=1 \
+PT_PIN_THREADS=1 \
+numactl --interleave=all \
```

Удалить `OMP_PLACES`/`OMP_PROC_BIND` — наш ThreadPool их игнорирует (agent 4 Q1, agent 3 Q5 "cargo").

**Ожидание:** 4.7 → 5.8-6.5 tok/s (agent 3 rank 1).

### Phase 1 — ThreadPool pinning + false sharing (1 час)

Файл: `c10/util/ThreadPool.h`.

1. **Line 205** — распределение workers. Сейчас `worker_id % ncpu` → при 24 workers × 4 NUMA node кладёт 24/3=8 на первые 3 узла, node 3 пустой. Фикс:
   ```cpp
   int node = worker_id % 4;
   int core_on_node = (worker_id / 4) % 8;
   int cpu = node * 8 + core_on_node;
   ```

2. **Line 152** — chunk_size rounded to multiple of 16 (одна cacheline = 16 fp32 = 64 B):
   ```cpp
   int chunk = (N + T - 1) / T;
   chunk = ((chunk + 15) / 16) * 16;  // align chunk boundary to cacheline
   ```

3. Scratch buffers через `posix_memalign(..., 64, ...)` — найти все `malloc` в hot path (gguf_model.h `buf_*`), заменить.

**Ожидание:** +5-10% → 6.0-6.8 tok/s.

### Phase 2 — NumaReplica для lm_head + QKV (2 часа)

Файл: `torch/io/gguf_model.h`.

1. Line 1479 (load path) — добавить `rep(q_output_weight)` под `if (use_numa_replicate_)`.
2. Line 1479+ — reprep q/k/v для каждого слоя (если не реализовано).
3. Line 2319 (`cpu_fused_rmsnorm_qkv_gemv`) — добавить `numa_node` param, внутри выбирать реплику.

**Ожидание:** +10-15% → 6.6-7.8 tok/s.

### Phase 3 — SIMD accumulators в Q4_K GEMV (4 часа) — agent 1 P3

Файл: `torch/io/cpu_quant_gemv.h`, lines 345-470.

Сейчас:
```cpp
float sum0 = 0.f, sum1 = 0.f;
for (int j = 0; j < Q; j++) {
    sum0 += scales[j] * dot0;
    sum1 += scales[j] * dot1;
}
```

VLIW может на 1 cycle issue'ить **packed** FMA, но не scalar FMA (они на разных каналах). Замена:
```cpp
__m256 acc0 = _mm256_setzero_ps();
__m256 acc1 = _mm256_setzero_ps();
for (int j = 0; j < Q; j += 8) {
    acc0 = _mm256_fmadd_ps(scales_vec, dot0_vec, acc0);
    acc1 = _mm256_fmadd_ps(scales_vec, dot1_vec, acc1);
}
float sum0 = horizontal_sum(acc0);
float sum1 = horizontal_sum(acc1);
```

**Плюс:** `__restrict` на pointer params, `#pragma loop count (1024)`, `int64_t j` вместо `int`.

**Ожидание:** +15-30% (agent 1 P3 + P1 + P2) → 7.6-10.1 tok/s.

### Phase 4 — Drop TP meta-fix (1 час experiment) — agent 3 meta

1-proc с полной репликацией (ffn_gate/up/down, attn_q/k/v/output, lm_head) — ~9.6 GB на узел = **влезает в 30 GB/node**. 32 pinned threads. Zero AllReduce. Все 32 ядра компутят.

**Ожидание (agent 3):** 7-9 tok/s. Но это **вместо** Phase 2 TP, не дополнительно. Тестировать параллельно.

### Phase 5 — Spin-wait → futex в SHM AllReduce (2 часа) — agent 4 P3

Файл: `torch/distributed/ddp.cpp`, lines 535-566.

```cpp
while (atomic_load(&ready_counter) != N) {
    __sync_synchronize();  // burns 100% CPU on idle
}
```

Заменить на bounded spin (1000 iters) + futex wait. Критично ТОЛЬКО если оставляем TP.

### Phase 6 — SSE-ABI rewrite GEMV (1 неделя) — agent 6 key insight

**Это главная нереализованная ставка.** LCC 1.29 транслирует SSE4.1 → native QP идеально, AVX2 → QP с overhead. Переписать `q4k_gemv_avx2` → `q4k_gemv_sse41`:

- `_mm_maddubs_epi16` вместо `_mm256_maddubs_epi16` (две половинки вручную)
- `__builtin_e2k_qppermb` для 4-bit unpack (agent 6)

Reference: `ilyakurdyukov/e2k-ports` patches.

**Ожидание:** удвоение LCC efficiency → потенциально 2× compute (если compute был bottleneck, но agent 2 говорит bandwidth 5%, так что это менее вероятно). **Риск высокий**, reward тоже.

### Phase 7 — Speculative decode batched verify (3-5 дней) — agent 5 C

Draft: qwen3:0.6b (Q4) на 1 NUMA node. Verify: 4 draft tokens за 1 forward pass на остальных 3 nodes. Acceptance rate от public runs ≈ 60-80% на same-family draft.

**Ожидание:** если phase 0-3 дают 8-10 tok/s, spec × 1.8-2.5 → **14-22 tok/s**.

---

## Конкретный порядок исполнения

| # | Фаза | Время | Ожидание |
|---|------|-------|----------|
| 1 | Phase 0 (env) | 5 мин | 5.8-6.5 |
| 2 | Phase 1 (pinning) | 1 ч | 6.0-6.8 |
| 3 | Phase 2 (replica) | 2 ч | 6.6-7.8 |
| 4 | Phase 3 (SIMD acc) | 4 ч | 7.6-10.1 |
| 5 | Phase 5 (futex) | 2 ч | +stability |
| 6 | Phase 4 (drop TP) | 1 ч expt | 7-9 alt |
| 7 | Phase 6 (SSE) | 1 нед | 10-15+ |
| 8 | Phase 7 (spec) | 3-5 д | 14-22 |

Cumulative **realistic** к концу недели: **12-15 tok/s.** Оптимистично: **18-22.**

## Чего мы НЕ знаем без live Elbrus

1. `numastat -p <pid>` во время decode — видно ли реально numa_miss?
2. `perf stat -e cpu-migrations` — мигрируют ли threads с unpinned config?
3. LCC register spill когда 4 SIMD accumulators parallel — компилятор spill'ит в stack?
4. Acceptance rate qwen3:0.6b→4b на русском корпусе — неизвестен.
5. Real DDR bandwidth ceiling на E8C2 — datasheet 25 GB/s/chip, но практика?
6. Разница SSE4.1 vs AVX2 на LCC 1.29 — цифра только заявлена ilyakurdyukov'ым, не замерена у нас.

Все 6 пунктов — проверяются за 1 день на Elbrus.
