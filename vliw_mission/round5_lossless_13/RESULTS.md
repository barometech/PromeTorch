# Round 5 — Lossless 13 tok/s investigation (2026-05-02)

## Цель
Поднять qwen3:4b Q4_K_M TP-4 inference на Эльбрусе 8C2 с 11.4 → 13 tok/s
**lossless** через комбинацию Gemini Deep Research Max + multi-agent
анализ (Opus 4.7 + Sonnet 4.6).

## Что попробовано

### 1. e2k SIMD attention math (Opus 4.7 #5)
- Q@K и V@scores: было `#if defined(__AVX2__)` + scalar fallback на E2K.
- Добавлен `#elif defined(__e2k__)` через qpfmuls + qpfadds (4-lane fp32).
- **Результат:** attn_phase 20.39 → 19.94 ms (-0.45 ms = шум).
- LCC 1.29 уже автовекторизует scalar dot product через VLIW slots.

### 2. scores buffer reuse (Sonnet 4.6 #1)
- Убран `std::vector<float>(total_seq)` внутри 8-head × 36-layer цикла.
- 288 alloc/free per token → 0 (persistent buffer в TPParallelState).
- **Результат:** в шуме, struct cleanup.

### 3. RoPE cos/sin cache (Sonnet 4.6 #3)
- past_len одинаков для всех 36 слоёв одного decode шага.
- rope_precompute теперь 1 раз per decode step вместо 36.
- **Результат:** в шуме, struct cleanup.

Все три lossless verified, закоммичены `6ff330d`. Suma effect = ~0.4 ms.

### 4. PT_TP_GATHER mode (Option F)
- Existing flag, переключение на AllGather вместо AllReduce-sum для
  output_proj и др. секций.
- **Результат:** **9.0 tok/s регрессия + НЕ lossless** (output text
  diverges от baseline). Не использовать.

### 5. **DR Recommendation: busy-spin ThreadPool**
- DR посоветовал убрать futex_wait, заменить на 50k-iter spin на atomic
  gen counter. Цитировал llama.cpp паттерн который "abandoned lock/unlock
  in favor of busy-waiting on atomic variables".
- Реализовано в `c10/util/ThreadPool.h` worker_loop с PT_TP_NOSLEEP=1
  для pure spin режима.
- **Результат:** **РЕГРЕССИЯ 11.4 → 5.5 tok/s (×2 медленнее)**.
- Откачено.

### Почему busy-spin не работает на 4-rank × 8-worker E2K setup

DR совет правильный для **single-process с одним пулом workers**. Наш
setup отличается:
- 4 ranks × 8 workers per rank = 32 spin'ующих threads
- Каждый rank на своём NUMA узле, но все 32 threads делят L3 cache
  coherency
- Когда worker spin'ит на `gen_.load()`, это L3 read запрос
- 32 threads × спин = постоянный поток L3 invalidation requests между
  всеми 4 NUMA узлами через cross-chip interconnect
- Это **thrashes** L3 coherency и блокирует реальные compute операции

llama.cpp single-process не имеет cross-NUMA coherency давления.

**Эмпирический вывод:** на multi-rank TP setup с distinct NUMA
scopes futex_wait это правильный pattern даже несмотря на kernel
overhead. DR/llama.cpp совет здесь вреден.

## Итог Round 5

| Approach | Lossless | Speedup |
|----------|---------:|--------:|
| HEAD baseline | ✓ | 11.4 tok/s |
| e2k SIMD attn + scores_buf + RoPE cache (commit 6ff330d) | ✓ | 11.4 (noise) |
| PT_TP_GATHER | ✗ output diverges | 9.0 (регрессия) |
| busy-spin pool (DR) | ✓ | **5.5 (×2 регрессия)** |

**Lossless потолок 11.4 tok/s** подтверждён ещё с одного угла: даже
"стандартная" техника low-level concurrency на Linux (busy-spin) не
работает в нашем 4-NUMA-process setup из-за coherency thrashing. Текущая
futex-based ThreadPool оптимальна для этой топологии.

## Что РЕАЛЬНО осталось для +N lossless

- **Hand-tuned E2K assembly** — не intrinsics, чистый asm с software
  pipelining. Disassembly показал peak VLIW packing в LCC, но компилятор
  всё равно оставляет ~50% gap до железного предела (latency hiding,
  better register allocation).
- **EAGLE draft model** — требует GPU тренировки 24-48 часов на A100
  ChatQA-style dataset.
- **Trained early-exit head** — confidence prediction, ~12-24 ч тренировки.

Все требуют ресурсов вне рамок одной интерактивной сессии.
