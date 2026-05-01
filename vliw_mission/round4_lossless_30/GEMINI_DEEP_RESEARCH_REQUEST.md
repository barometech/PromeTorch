# Gemini 3.1 Pro Deep Research — Round 4 Mission Request

## Контекст
PromeTorch — single-dev PyTorch-аналог на C++/CUDA, нативная сборка
под российский Эльбрус 8C2 (E2K v5 VLIW, 32 ядра, 1.5 GHz, 4 NUMA, 100
GB/s aggregate DDR). Текущий результат: **9.4 tok/s** на qwen3-4B Q4_K_M
через TP-4 + custom Q8 SoA4 INT8 quantization layout под `qpmaddubsh`
(VNNI-style v5 intrinsic).

GitHub: https://github.com/barometech/PromeTorch (52 stars).

## Цель Round 4
Достичь **30 токенов/сек на qwen3-4B без потерь качества vs Q4_K_M
baseline** на том же Эльбрусе через:
1. Новый формат весов (custom container, lossless repack из GGUF)
2. Tuned inference kernels (kernel fusion, APB, SWP, ThreadPool refactor)
3. Speculative decoding с подходящим draft model

## Ваша задача (8 вопросов исследования)

### Q1. Lossless quantization layouts — state of the art (CPU)
Какие existing quantization форматы дают **lossless** конверсию из
GGUF Q4_K_M (5.5 bit/param) при сохранении / улучшении inference
скорости на CPU?
- llama.cpp's `q4_0_4_4` / `q4_0_4_8` (ARM NEON-friendly INT4 layouts)
- AVX-VNNI / Intel AMX INT8 layouts для x86
- Что с **VLIW** (Эльбрус)? Кто ещё кроме нас оптимизирует под e2k v5?
  Есть ли проекты на GitHub / Habr / препринты?

### Q2. CPU quantization sub-formats для INT8 dot без полного VNNI
Эльбрус 8C2 имеет **частичный VNNI**: `qpmaddubsh` (16×u8×s8 → 8×i16,
без horizontal reduce). Аналогичные ситуации:
- ARM Cortex-A53 / -A57: `vmlal_u8` без `udot`
- Old Intel pre-VNNI: `vpmaddubsw` без `vpdpbusd`
- Какие layouts / quant tricks применяют такие архитектуры чтобы
  получить close-to-VNNI throughput? Есть ли известные patents,
  реализации (TVM, OpenVINO) для CPU без INT8 dot?

### Q3. Speculative decoding с tuned draft на той же модели family
qwen3 имеет варианты: 0.6B, 1.7B, 4B, 14B. Кто использует qwen3:0.6B
как draft для qwen3:4B? Какой achievable accept rate? Любые
benchmark / paper / blog?
- Vanilla speculative decoding (Leviathan 2022)
- Tree-based: SpecInfer, Medusa, EAGLE
- Self-speculative: skip-layer, layer-pruning drafts

### Q4. Bandwidth utilization tricks для VLIW e2k v5
Наша утилизация DDR = ~33% (9.4 tok/s × 3.5 GB / 100 GB/s aggregate).
Tuned LLM kernels на x86/ARM достигают 60-80%. Что мешает на VLIW?
- LCC compiler специфика (APB, SWP)
- Cache miss patterns
- Известные tricks для VLIW DSP / Itanium / TI C6x для bandwidth
  bound workloads?
- Есть ли open MCST / NTC Module benchmarks LLM на Эльбрус?

### Q5. Custom binary container formats для in-memory ML моделей
Что хорошее существует кроме GGUF / safetensors / GGML?
- FlatBuffers / Cap'n Proto?
- Custom mmap-friendly designs (Apple's ANE format / Mojo's Tensor)?
- Как лучше всего организовать metadata + tensor data layout для
  zero-copy mmap + bytewise verify + cache line alignment?

### Q6. CLI/GUI conversion tools для AI моделей — UX patterns
Утилиты типа `gguf-my-repo`, `transformers-cli`, `ollama push`. Что
лучшие UX-практики:
- Single binary vs multi-step
- Progress reporting (stderr ANSI vs JSON streaming)
- Validation (re-load + sanity inference) opt-in vs default
- GUI: тёмная тема, drag-drop, rust+web (Tauri) vs Qt vs Electron — для
  scientific/dev пользователей какой стек реально *используется*?

### Q7. Memory bandwidth optimization специфика 4-NUMA single-socket
Наш Эльбрус: 4× E8C2 чипа на одной материнке, 4 NUMA нода. NUMA replicate
весов выгоден ($\sim$8% gain). Какие ещё optimizations существуют для
single-host multi-NUMA TP без RDMA / inter-host interconnect?
- DPDK-style busy-poll AllReduce
- huge pages
- thp transparent
- numactl tricks

### Q8. Russian VLIW community / partnership
Кто ещё в РФ занимается ML на Эльбрусе помимо МЦСТ и НТЦ Модуль?
Есть ли open-source проекты, гранты, исследовательские группы (Yandex
Research, Сбер AI, Skoltech)? Чтобы потенциально привлечь.

## Формат отчёта

Структура в JSON или markdown table:
```
{
  "Q1": {
    "summary": "...",
    "key_findings": ["...", "..."],
    "actionable_for_PromeTorch": "...",
    "sources": [{"title": "...", "url": "..."}, ...]
  },
  ...
}
```

Каждый вопрос — отдельный block, с рекомендациями применимыми к
PromeTorch (не общие выводы, а специфика).

## Глубина

**Deep Research Max**: ищи не только в обычных search results, но и:
- arXiv preprints за 2024-2026
- Habr / GitHub / Stack Overflow / forums
- Конференции: NeurIPS, ICML, MLSys, OSDI, USENIX ATC, SC, ISC HPC
- GitHub issues / discussions / wikis на крупных проектах
  (llama.cpp, ollama, vllm, sglang, TGI)
- Russian-language sources (Хабр, vc.ru, Habr blog'и компаний)
- Patents (Google Patents, EPO) для quantization / VLIW tricks
- MCST / Module технические PDF (если найдёшь)

## Бюджет на Deep Research
~30 минут wall clock, неограниченный pages explored. Финальный отчёт
~2000-3000 слов с конкретными references.
