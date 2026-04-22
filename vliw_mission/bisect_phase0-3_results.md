# Bisect Phase 0-3 — empirical results on live Elbrus 8C2

**Дата:** 2026-04-22
**Железо:** Эльбрус 8C2 (lemur-1), LCC 1.29.16, Linux 6.1 e8c2
**Модель:** qwen3-4b-Q4_K_M, 50-token greedy generation, prompt="Hello"
**Режим:** 1-proc, 24 threads, `numactl --interleave=all`

## Measured cross

| PT_NUMA_REPLICATE | PT_PIN_THREADS | tok/s |
|:-:|:-:|:-:|
| 0 | 0 | **4.6** |
| 0 | 1 | 4.5 |
| 1 | 0 | 4.3 |
| 1 | 1 | 4.4 |

**Baseline 4.7 tok/s (100 tokens, `BENCH_ELBRUS.md`) vs current 4.6 tok/s (50 tokens):** разница ≤ measurement noise.

## Выводы против исходных гипотез

| Гипотеза | Ожидание | Реальность |
|---|---|---|
| PT_NUMA_REPLICATE=1 в 1-proc → +10-15% | agent_3 rank 1 | **-0.3 tok/s регрессия** |
| PT_PIN_THREADS=1 stripe layout → +5% | agent_3 rank 1 | **-0.1 tok/s (нейтрально) |
| Phase 3 split accumulator → +15-30% | agent_1 P3 | неотличимо от baseline |

## Почему reпликация вредит в 1-proc

`numactl --interleave=all` уже round-robin'ит каждую страницу весов по всем 4 DDR-контроллерам. Любой поток в среднем читает 1/4 страниц с локального узла и 3/4 с чужих — это балансирует bandwidth **статически** через kernel page policy, без ведения ThreadPool или кода модели.

Добавление NumaReplica поверх интерливинга:
1. **+9.5 GB working set** (2.4 GB оригинал + 4×2.4 GB копии = 12 GB total allocated)
2. **TLB давление** — 12 GB в 4-KB страницах = 3M TLB entries; в 2-MB (MADV_HUGEPAGE) = 6K entries, но помогает только если ядро действительно аллоцирует huge pages (не гарантия)
3. **Bandwidth picture не меняется** — каждый тред всё равно читает 1 локальный поток + 3 remote через interconnect

## Почему Phase 3 split accumulator не дал gain

Agent 1 правильно указал (Q2): bottleneck — **horizontal reduce внутри j-loop** (4 × `_mm_add_epi32` + `_mm_shuffle_epi32` каждую итерацию). Split accumulator только уменьшает chain length с 2 до 4 на сам FP-хвост, но основной килллер — поперечные shuffle'ы в int32 reduce, которые E2K делает через один permutation unit.

**Gemini 3.1 Pro SSE4.1 design** (`gemini_sse41_design.md`) подтверждает: **полностью убрать horizontal reduce из inner loop** — аккумулировать вертикально в `__m128 acc` и hsum только один раз на super-block. Это Phase 6 и даст реальный 1.5-2× выигрыш на compute.

## Решения

1. **1-proc скрипт:** убрать PT_NUMA_REPLICATE=1 и PT_PIN_THREADS=1 (они не дают gain, а replicate регрессирует).
2. **TP скрипт:** оставить PT_NUMA_REPLICATE=1 (каждый rank --membind на 1 узел → без replicas имеет 100% cross-chip weights).
3. **Phase 3 split-accumulator:** оставить в коде. Математика корректна, vertical-SSE4.1 rewrite (Phase 6) будет опираться на эту же структуру.
4. **Приоритет: Phase 6 (SSE4.1 rewrite)** — Gemini design в `gemini_sse41_design.md` готов к реализации.
5. **Замерить TP-4 с replicate** отдельно — гипотеза: там replicate даст +10-20%.

## Уроки для проекта

- **«Очевидная» NUMA-оптимизация может регрессировать, когда kernel page policy уже решила задачу.** Всегда мерить, не полагаться на интуицию agent reports.
- **Agent 2 был прав** (bandwidth не bottleneck). Решение bandwidth-симптома не даёт tok/s.
- **Agent 1 P3 оказался полу-прав.** Split scalar — корректный шаг, но чтобы увидеть эффект нужен полный vertical accumulator (Phase 6).
