# Бенчмарк PromeTorch + llama.cpp — Windows CPU

**Хост:** Windows 10 Pro 19045, AMD EPYC 7F52 16-Core (32 потока, 3.5 GHz, AVX2)
**Дата:** 2026-05-03

## llama.cpp baseline (release b6048, ggml-cpu-haswell.dll)

| Модель | Размер | tok/s (tg128 / tg64) | Потоки |
|---|---|---|---|
| qwen3 4B Q4_K_M | 2.44 GiB | **15.25 ± 0.90** | 32 |
| qwen3 8B Q4_K_M (deepseek-r1:8b distill) | 4.86 GiB | **8.66 ± 0.00** | 32 |
| qwen3 14B Q4_K_M | 8.63 GiB | **4.43 ± 0.00** (tg64) | 32 |
| gemma3 4B | — | failed_to_load (Ollama blob — не GGUF полный?) | 32 |

Команда: `llama-bench.exe -m <model.gguf> -t 32 -ngl 0 -p 0 -n 128 -r 1`

## PromeTorch CPU build на Windows MSVC — портирование

PromeTorch исторически билдился на Windows только в CUDA-режиме. CPU-only сборка
для GGUF inference (`test_gguf_inference`) на MSVC потребовала следующих фиксов:

1. **TudaBLAS.h** — `__attribute__((unused))` (GCC) → `[[maybe_unused]]` (C++17)
2. **CMakeLists.txt** — для x86_64 MSVC явно добавить `/arch:AVX2`,
   `__AVX2__ __FMA__` defines (без них `TUDA_AVX2` path выключен) и `NOMINMAX`
   (чтобы `<windows.h>` не ломал `std::min/max`).
3. **c10/util/Futex.h** — `WakeByAddressAll/Single`: убран `volatile` cast
   (MSVC не принимает `volatile void*` → `PVOID`).
4. **torch/io/gguf_model.h** — `__builtin_prefetch` (GCC) обёрнут в макрос для
   MSVC: `_mm_prefetch((const char*)addr, _MM_HINT_T0)`.
5. **torch/io/cpu_quant_gemv.h** — `0.0f / 0.0f` (constant divide-by-zero error
   на MSVC) → `std::numeric_limits<float>::quiet_NaN()`.
6. **torch/io/q8_soa_repack.h** — `posix_memalign` отсутствует в MSVC
   → `_aligned_malloc/_aligned_free`.

После этих фиксов CPU-only сборка `test_gguf_inference.exe` собирается на
Windows MSVC 2019 с `/arch:AVX2`.

## PromeTorch tok/s (single-process, OMP=32, PT_Q8_SOA=1)

| Модель | tok/s | llama.cpp tok/s | Ratio | Tokens |
|---|---|---|---|---|
| qwen3-4B Q4_K_M | **13.1** | 15.25 | 0.86× | 64 |
| qwen3-8B Q4_K_M (deepseek-r1:8b distill) | **7.7** | 8.66 | 0.89× | 64 |
| qwen3-14B Q4_K_M | **4.4** | 4.43 | **1.00× (паритет)** | 32 |

Команда: `OMP_NUM_THREADS=32 PT_Q8_SOA=1 ./test_gguf_inference.exe <model.gguf> --max-tokens 64 --greedy "Hello, world. "`

## Выводы

* PromeTorch CPU-only сборка работает на Windows MSVC после 6 портативных
  фиксов (см. выше). Не требует Linux/MinGW — чистый MSVC + AVX2.
* Скорость на AMD EPYC 7F52 32-thread близка к llama.cpp baseline:
  паритет на 14B (memory-bound), 86-89% на меньших моделях
  (где compute-bound start to matter и наш Q8_SoA4 микрокернел
  настроен под Эльбрус, не под x86).
* Tensor Parallelism (TP-4) на Windows не запускался — это требует
  POSIX shared memory + futex, на Windows работает но используется
  только на Эльбрусе из-за NUMA-aware дизайна.

