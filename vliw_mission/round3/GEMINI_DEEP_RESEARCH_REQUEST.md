# Gemini 3.1 Pro Deep Research — promethorch Round 3 (2026-04-27)

> **Model**: `gemini-3.1-pro-preview`
> **Mode**: Deep Research Max
> **Task**: cross-architecture LLM inference performance audit + breakthrough recommendation

---

## Контекст

Мы пишем PromeTorch — самостоятельный PyTorch-аналог на C++/CUDA. Целевая
платформа: **МЦСТ Эльбрус 8СВ** (8C2, ISA v5), 4 NUMA-узла × 8 ядер @ 1.5 GHz,
4×32 GB DDR.

LCC 1.29 (gcc 11.3 совместим). 8C2 имеет VLIW 4-канал × 2 FMA, **Array
Prefetch Buffer (APB)** + **Software Pipeline (SWP)** в железе. Из VNNI-style
инструкций есть только `qpmaddubsh` (16×u8×s8 → 8×i16 pairwise add).
`qpidotsbwss` (полный INT8 dot+accumulate) только на v7 (12C/16C).

## Текущее состояние

Production: **qwen3:4b Q4_K_M, TP-4 (4 процесса × 8 OMP threads, NUMA-bound)
= 4.8 tok/s** (211 ms/token).

Профайлер раскладка ms/token:
```
gate_up:       65.4   RMSNorm + gate + up GEMV (fused)
ffn_down:      48.9   SiLU + ffn_down GEMV
attn_phase:    29.9   RMSNorm + QKV + attention math
output_proj:   23.7   final RMSNorm + vocab GEMV (151936×2560)
attn_output:   15.0
allreduce:     11.7   2× per layer SHM-based
TOTAL:        211 ms
```

Bandwidth: per-NUMA peak **12.2 GB/s** (memcpy benchmark). Effective **2.9 GB/s
per chip = 23% utilization**. Compute: ~12 GOPS/core (EML cblas_sgemv),
peak per Trushkin = 72 GFLOPS = **17% utilization**.

## Что НЕ помогло

| Попытка | Результат |
|---|---|
| Option F (gather + futex AllReduce) | 0% — sync 6%, не bottleneck |
| fp32-prelude scales (fp16 conv. out of hot loop) | 0% — LCC уже пайплайнит |
| qpmaddubsh VNNI Q8_0 kernel | 12× в синтетике, 0% в проде |
| Q8_0 conversion (bandwidth ×1.83) | 4.8 → 1.5 tok/s |
| `__builtin_prefetch` в скалярных Q4_K | 0% — `fapb` уже linear |
| Multi-row unroll в qpmaddubsh kernel | 0% — reduce-chain serializes |

## Что почти точно поможет (Round 3 agents подтвердили)

1. **Option D** (Round 2 agent_9 предложил, не реализовали): K-slice ВСЕХ replicated weights (attn_output / ffn_down / output_weight) + удаление чтения mmap-родителя → +50% **(5.3 → 9 tok/s)**, ~150 LoC.
2. **ThreadPool persistent workers + futex barrier** взамен mutex+CV per-call: 200×100µs → 200×5µs = -19 ms/token = **+8% → 5.2 tok/s**.
3. **PLD speculative decode** (prompt-lookup K=4-6 over full prompt+history): α=0.65-0.8 acceptance → **6-10 tok/s** при 750 LoC.
4. **HugeTLB 2MB pages** (если sysadmin даст nr_hugepages=1500): TLB-miss rate 5-15% → 0.1%, ожидание ?

Combined потенциал: 9 (Option D) × 1.08 (ThreadPool) × ~2 (PLD) ≈ **19 tok/s**.

## Bandwidth/compute математика

- Модель 2.4 GB / 4 chips × 12 GB/s = 50 ms theoretical bandwidth floor = **20 tok/s ceiling**
- Compute peak 72 GFLOPS × 8 cores × 4 chips = 2.3 TFLOPS aggregate. Per-token compute = 5.6 GFLOP × 2 = 11.2 GFLOP → 5 ms compute. **Compute не bottleneck**
- Реальный gap 4.8 vs 20 = 4.2× (memory pipeline stalls, недоиспользование bandwidth)

## Ваши вопросы к Gemini Deep Research

### Q1: VLIW + bandwidth-bound LLM inference — state of the art

Что современные исследовательские группы (МЦСТ, llama.cpp авторы, Bytedance, Microsoft, Apple Neural Engine team) делают для CPU LLM inference на bandwidth-bound архитектурах в 2026? Не ROP — конкретные **kernel design patterns** что **доказанно** дают boost при 23% bandwidth + 17% compute утилизации одновременно.

Особенно интересует:
- Кэш-aware weight repack (какой layout, какой block size, на каких процах работает)
- Speculative decoding варианты что РЕАЛЬНО работают на CPU (не только Medusa с GPU training)
- 1.58-bit ternary + bitwise GEMV (BitNet) — есть ли E2K-портированные ядра
- Page table / TLB optimization для большой modelи на CPU

### Q2: МЦСТ 8СВ specific

Изучи всё публичное про оптимизацию compute/memory pipeline на 8СВ или E2K v5:
- llama.cpp PR с E2K патчами (если есть) — какие kernels они используют?
- МЦСТ публикации 2025-2026 про AI inference performance
- Habr.com статьи про оптимизацию AI на Эльбрусе
- Что МЦСТ говорил про APB на конференциях / LCC pragmas

### Q3: 4× speedup — реалистичный путь

Дай ТОЧНЫЙ план как от 4.8 tok/s дойти до 19+ tok/s на этом железе.
Прорейтуй техники по ROI (uplift / эффорт). Учти ограничения: software-only,
нет доступа к firmware, LCC компилятор, без переобучения модели.

### Q4: Что мы упускаем фундаментально

Какой угол атаки мы НЕ пробовали и который мог бы дать 2× прорыв?
Не общие фразы — конкретные алгоритмы / layouts / tricks с pointer'ами на papers / repos.

---

**Output format**: structured markdown report ~3000 слов. Цитируй papers с arxiv URL,
github repos с конкретными PR/file ссылками. В конце — приоритезированный список
из 5-7 действий.
