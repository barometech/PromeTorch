# Round 4 Mission — Lossless 30 tok/s на qwen3:4b на Эльбрусе E8C2

**Дата:** 2026-04-30
**Стартовая позиция:** TP-4 + Q8 SoA4 = **9.4 tok/s** (commit `3399136`)
**Цель:** **30 tok/s на qwen3:4b** *без потерь качества вообще* + конвертер
GGUF→PromeTorch формат с CLI/GUI.

---

## 1. Бизнес-цель

Юзер требует:
1. **Новый формат весов** для PromeTorch, в который конвертится qwen3:4b
   GGUF Q4_K_M (или любой GGUF) **lossless** (см. §2 — определение).
2. **30 tok/s** на этой модели на Эльбрусе E8C2 (4× E8C2, 32 ядра, 1.5 GHz,
   100 GB/s aggregate DDR).
3. **CLI-утилита** + опционально **GUI** (тёмная, минималистичная, простая
   по UX) для конверсии. Пример: `prometorch-convert qwen3-4b-Q4_K_M.gguf --out qwen3-4b.pt8`
4. **Без потерь качества** — *критическое требование*.

---

## 2. Что значит "без потерь качества" — точное определение

Без однозначного определения цель невыполнима. Допустимые трактовки:

**(A) Bit-exact относительно FP32 reference:**
   GGUF файл Q4_K_M уже имеет потери vs FP32 (~1-2% perplexity). Если
   "lossless" значит bit-exact с **исходным GGUF файлом** — то любой
   формат который dequantize'ит Q4_K → fp32 → re-quant'ит обратно
   ОК тогда когда мы можем при нужде dequantize в тот же fp32. Это
   тривиально достижимо ($O(0)$ на качество).

**(B) Numerically equivalent относительно нашего текущего decode-пути:**
   Текущий путь: Q4_K → cpu_quant_gemv (или Q8 SoA4 после repack). Любой
   новый формат должен давать тот же логит-вектор bit-exact (или внутри
   fp32 round-off). Это означает — мы переупаковываем Q4_K под другой
   layout, но математически inference identical.

**(C) Perplexity drop ≤ ε (например ε=0.05%) на benchmark задаче:**
   GGUF Q4_K_M → новый формат может привнести ε ошибки если она ниже
   шума stochasticity. Не "вообще без потерь", но практически ничего.

**Decision (предлагаемое в этом ТЗ):** **трактовка (B)** — новый формат
*математически эквивалентен* текущему Q4_K decode-пути bit-for-bit
(modulo fp32 floating-point reorderings, которые уже неизбежны).

> Если юзер требует (A) bit-exact с FP32 reference — это означает либо
> хранить веса в FP32 (16 GB модель, потолок 6 tok/s по bandwidth), либо
> хранить в Q8_0 (4.3 GB, потолок ~22 tok/s). Любая компрессия ниже Q8_0
> вносит quantization error vs FP32. Но Q4_K_M **уже** ввёл эту ошибку —
> мы её просто наследуем.

---

## 3. Honest physical limit analysis — почему 30 tok/s на грани

Эльбрус 8C2 4-NUMA:
- DDR bandwidth: 4 × 25 GB/s = **100 GB/s aggregate** (измерено на нашем серверe)
- Per-token forward pass читает все веса 1 раз (decode mode, KV cache hot)
- Кол-во весов qwen3:4b: ~4.0e9 параметров
- Размер при разных форматах (только веса GEMV без norms/embd/kv):

| Формат | Бит/параметр | 4B размер | Forward read | Theoretical max @100GB/s |
|---|---|---|---|---|
| FP32 | 32 | 16.0 GB | ~13 GB | **7.7 tok/s** |
| FP16 | 16 | 8.0 GB | ~6.5 GB | 15.4 tok/s |
| Q8_0 | ~9 | 4.3 GB | ~3.5 GB | 28.6 tok/s |
| **Q4_K_M (now)** | **~5** | **2.4 GB** | **~2.0 GB** | **50 tok/s** |
| Q4_0 | ~4.5 | 2.2 GB | ~1.8 GB | 55.5 tok/s |
| Q3_K_M | ~4 | 1.9 GB | ~1.6 GB | 62.5 tok/s |
| Q2_K | ~3 | 1.5 GB | ~1.2 GB | 83 tok/s |

Реальная утилизация bandwidth у нас сейчас при 9.4 tok/s **= ~19%** от
теоретического потолка 50 tok/s (Q4_K_M). Чтобы достичь 30 tok/s нужно
утилизировать **60% bandwidth**. Это значимо больше нашего текущего 19%
но не невозможно — типичные tuned LLM kernels достигают 60-80%.

**Вывод:** 30 tok/s **физически достижимы** на Q4_K_M размере при
аккуратной оптимизации. Мы НЕ упираемся в bandwidth ceiling — мы упираемся
в overhead (sync, dequant compute, AllReduce, kernel launch).

---

## 4. Гипотезы пути к 30 tok/s (без потери качества vs Q4_K_M)

### Гипотеза H1: уменьшить overhead на каждом GEMV
Текущий overhead per GEMV (TP-4):
- AllReduce SHM ~5-10 μs/op × ~80 ops = ~0.5 ms/token
- Q8 SoA4 quant_activation per call: ~0.1 ms × 5 = 0.5 ms/token
- ThreadPool fork/join: 100 μs × 200 calls = 20 ms/token (старый pool)

С персистентным ThreadPool (Round 3 Agent 1, refactor): 5 μs × 200 = 1 ms/token.
**Saved ~19 ms/token = +90% throughput.**

### Гипотеза H2: kernel fusion поверх SoA4
Сейчас SoA4 GEMV отдельно для Q/K/V. Fusion:
- Q+K+V → 1 fused kernel с общим quant_activation для x
- gate+up → 1 fused (уже частично fused)
- Эффект: ~30% меньше memory traffic для x_quantized

### Гипотеза H3: full-replicate + row-parallel + AllGather
TP-4 row-parallel сейчас делает K-slice + AllReduce. Альтернатива
(Option F тестировалась 2026-04-25, не дала прироста на Q4_K, но Q8 SoA4
+ all_gather + row-replicate может быть лучше).

### Гипотеза H4: APB / hardware prefetch на Q8 SoA4
LCC `-fprefetch -faligned -frestrict-all -fswp-maxopers=800` есть, но APB
(Array Prefetch Buffer) может быть не задействован для SoA блоков 176 байт.
Manual `__builtin_prefetch(b+1, 0, 2)` уже добавлен в `cpu_quant_gemv.h`,
для SoA — пока нет.

### Гипотеза H5: speculative decoding с tuned draft
PLD (Prompt-Lookup Decoding) реализован, gated `PT_PLD=1`, регрессировал
на Эльбрусе из-за batched-decode K-serialness. Если accept rate 60% +
batched verify работает корректно, эффективный throughput = 9.4 × 2.5 =
**23.5 tok/s**. С 75% accept = 37 tok/s.

Проблема: **draft model требует обучения** для high accept rate. NgramDraft
дал 0% accept rate на qwen3 (tested Phase 7.5). Нужно либо:
- Trained tiny qwen3 draft model (qwen3:0.6B)
- Tree-based speculative decoding (Medusa, EAGLE)

### Гипотеза H6: новый layout — column-major SoA + persistent activation buffer
Q8 SoA4 — 4-row interleaved. Column-major (K-major) layout даст better
cache reuse при row sequence access. Стоит проверить.

### Гипотеза H7: attention KV-cache compression
KV cache при длинном context'е растёт. Q4 KV cache compression — потеря
качества (есть исследования что perplexity drop минимален). Но это уже
**lossy**, не подходит.

---

## 5. План deliverables

| # | Deliverable | Срок | Lead |
|---|---|---|---|
| 1 | Анализ feasibility 30 tok/s + честный roadmap | 1 сессия | Agent A (этот документ) |
| 2 | Спецификация формата `.pt8` (или другое имя) | 1 сессия | Agent B |
| 3 | Конвертер `gguf2pt8`: GGUF → новый формат, lossless | 1-2 сессии | Agent B |
| 4 | Loader в gguf_model.h: читает новый формат напрямую | 1 сессия | Agent C |
| 5 | Optimized inference path под формат (kernel fusion + APB) | 2-3 сессии | Agent D |
| 6 | Speculative decode с tiny draft model (если нужно для 30) | 2-3 сессии | Agent E |
| 7 | CLI tool `prometorch-convert` + man page | 1 сессия | Agent B |
| 8 | GUI tool (тёмная тема, Tauri/Electron/Qt) | 2-3 сессии | После CLI |
| 9 | E2E benchmark + acceptance test 30 tok/s | 1 сессия | Финал |

---

## 6. Критические файлы для изучения

```
torch/io/gguf_reader.h              # формат GGUF, парсинг metadata
torch/io/gguf_model.h               # loader, decode forward path (5300 строк)
torch/io/gguf_dequant.h             # Q4_K/Q6_K/Q8_0 dequantization formulas
torch/io/cpu_quant_gemv.h           # все quant GEMV kernels (3000+ строк)
torch/io/q8_soa_repack.h            # текущий best-perf путь (Q8 SoA4)
torch/distributed/ddp.cpp           # SHM AllReduce + futex
c10/util/ThreadPool.h               # fork/join (старый mutex+CV)
scripts/run_tp_elbrus.sh            # production run config
JOURNAL.md (last 200 lines)         # все measurements 4.6-9.4 tok/s
BENCH_ELBRUS.md                     # results comparison
vliw_mission/round3/MISSION.md      # предыдущий roadmap к 9.4 tok/s
vliw_mission/round3/agent_1_threadpool.md   # ThreadPool refactor design
vliw_mission/round3/agent_5_soa_repack.md   # как мы получили 9.4
vliw_mission/e2k_vnni/q8_soa4_microbench.c  # standalone microbench
docs/elbrus/                        # MCST документы по APB/SWP/LCC
README.md                           # overview + benchmarks таблица
```

---

## 7. Бриф для 5 агентов Opus 4.7

См. отдельные файлы:
- [agent_A_feasibility_format_design.md](agent_A_feasibility_format_design.md)
- [agent_B_converter_pipeline.md](agent_B_converter_pipeline.md)
- [agent_C_loader_inference.md](agent_C_loader_inference.md)
- [agent_D_kernel_optimization.md](agent_D_kernel_optimization.md)
- [agent_E_speculative_decode.md](agent_E_speculative_decode.md)

Каждому агенту в промпте передаётся: (1) этот MISSION.md, (2) их
конкретный agent_X_*.md, (3) right-sized список файлов для чтения,
(4) явный output spec — что должен выдать.

---

## 8. ТЗ для Gemini 3.1 Pro Deep Research

См. [GEMINI_DEEP_RESEARCH_REQUEST.md](GEMINI_DEEP_RESEARCH_REQUEST.md).

Темы: lossless quantization techniques, runtime decompression formats,
Russian VLIW e2k v5 SIMD optimization tricks, speculative decoding
SOTA для small draft models, CPU kernel libraries для INT8 dot
без full VNNI на VLIW архитектурах.

---

## 9. Critical risks и smell tests

1. **30 tok/s может быть недостижимо без speculative/draft** — physical
   ceiling на Q4_K_M = 50 tok/s, нам нужен 60% bandwidth utilization.
   Текущая утилизация ~19%. Закрыть gap в 3.2× за счёт *чистой*
   оптимизации kernels — амбициозно но не безумно. Если не получится
   — speculative decoding с draft даст +×2.5 поверх 9.4 → **23.5 tok/s**,
   что **не 30**.

2. **"Без потерь качества" неоднозначно** — см. §2. Мы фиксируем
   трактовку (B): новый формат **математически эквивалентен** Q4_K_M
   decode-path. Это означает мы наследуем уже существующую Q4_K
   ошибку vs FP32 (~1-2% perplexity), но не вносим **дополнительной**.

3. **GUI добавляет сложности** — пилить cross-platform GUI (Tauri/Qt)
   на Эльбрусе нетривиально, может не собраться LCC. Альтернатива —
   web-based UI (single binary HTTP server + статика), запускается
   на любой машине без зависимостей.

4. **Конвертер должен ПОЛНО ваять — без потерь и без re-train** —
   это нормально достижимо для (B).

---

## 10. Acceptance criteria финального этапа

```
PT_FORMAT=pt8 ./scripts/run_tp_elbrus.sh --greedy "prompt"

>>> 100 tokens in <3.34s = >= 30.0 tok/s ✓
>>> Logits identical to Q4_K_M baseline within fp32 round-off (max abs diff < 1e-5) ✓
>>> Convert pipeline: gguf2pt8 qwen3-4b-Q4_K_M.gguf -> qwen3-4b.pt8 in <30s ✓
>>> CLI works on x86 + Эльбрус ✓
>>> GUI launches, dark theme, single-button conversion ✓ (опционально)
```

---

## 11. Что делаем СЕЙЧАС (perceived order)

1. Развернуть 5 агентов параллельно — каждый идёт по своему направлению
2. Параллельно запустить Gemini DR с подробным брифом
3. Собрать результаты, синтезировать в **финальный план реализации**
4. Реализовать в порядке: feasibility → format spec → converter →
   loader → optimized inference → CLI → GUI → spec decode (если нужно)
