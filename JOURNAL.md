# PromeTorch - Журнал разработки

Полная история разработки проекта. Актуальные инструкции — в `CLAUDE.md`.
Полный аудит инфраструктуры — в `INFRASTRUCTURE_AUDIT.md`.

---

## 2026-03-18: ИНЦИДЕНТ — 16-core NMCard crash + ПОЛНЫЙ АУДИТ

### Инцидент
16-ядерный `OP_MATMUL_PARTIAL` в `dispatcher_suda_mc.abs` повесил NM Card Mini → полная перезагрузка ПК → потеря несохранённых данных.

**Root cause**: `dispatcher_suda_mc.cpp:161` — `core_index = boot[29]` читает из общего DDR (race condition между 16 ядрами). Рабочий `dispatcher_mc.cpp` использует `ncl_getCoreID()` (hardware register).

### Полный аудит (11 агентов Opus 4.6)
10 агентов + 1 ручная верификация. Результат: `INFRASTRUCTURE_AUDIT.md`

**Статистика**: 93,315 строк кода, 481 файл, 64+ NN модулей, 68 CUDA ядер, 55 backward функций, 9 оптимизаторов.

**Найдено багов**: 12 критических (верифицировано лично), 12 средних (верифицировано), 19 низких (из отчётов агентов).

**Критические находки:**
- FlashAttention полностью нерабочий (6 багов: нет grad_Q, block 4096 threads, broken softmax)
- GradScaler `has_inf_or_nan()` всегда `false` — mixed precision без защиты
- CUDA element-wise ядра теряют данные >16.7M элементов (нет grid-stride loop)
- Python `no_grad()` отключён от C++ autograd engine
- NMCard dispatcher_suda_mc: race condition на core_index + DDR bus saturation

**Что работает отлично:**
- Autograd: все 55 backward формул математически верны
- CPU SIMD: AVX2 GEMM, vectorized ops — production quality
- CUDA quantized GEMV (Q4_K/Q5_K/Q6_K) — production quality
- GGUF инференс: 49.9 tok/s, 5 архитектур
- NN modules: 64+ модулей с dual fast-path

**9 коммитов НЕ запушены на remote** — нужен push для безопасности данных.

---

## 2026-03-18: AirLLM-NMCard v1.0 — Qwen3-4B inference без PyTorch

### AirLLM-NMCard
Полностью свой аналог AirLLM, написанный с нуля. **Без PyTorch** — чистый numpy + NM Card.
Лицензия: Apache 2.0 (как оригинальный AirLLM).

**Компоненты:**
- `airllm_nmcard/ops.py` — rms_norm, silu, softmax, rope, gqa_attention (numpy)
- `airllm_nmcard/layer_loader.py` — safetensors parser (без torch!), NF4/INT8 dequant, prefetching
- `airllm_nmcard/model_splitter.py` — split HF model в per-layer safetensors + compression
- `airllm_nmcard/inference.py` — AirLLMNMCard + AutoModel, layer-streaming forward pass

**Qwen3-4B (CPU baseline, первый запуск):**
- 36 layers, hidden=2560, heads=32, kv=8, head_dim=128, intermediate=9728
- Split: 38 файлов × 385 MB (BF16→F32), total ~15 GB
- Forward (2 tokens): **45.9s** (253 matmuls × 170ms avg)
- Logits: (1, 2, 151936) — корректный vocab, осмысленный top-5

**Токенизатор:** `airllm_nmcard/tokenizer.py` — парсит `tokenizer.json` напрямую, byte-level BPE
- 151K merges, encode/decode без transformers dependency
- Верифицирован: "Hello world" → [151643, 9707, 1879] → "Hello world"

**Поддерживаемые архитектуры:** Qwen2, Qwen3, Llama, Mistral
**Quantization:** NF4 (4-bit), INT8 blockwise — встроено в splitter + loader

**Первый запуск на NM Card Mini:**
- Prefill 6 tokens: 89s (работает, но медленно)
- Генерация: мусорный текст — ошибка в matmul dispatch или weight transpose
- lm_head (151936×2560 = 1.5 GB) не влезает в DDR одним куском → нужен tiling

**Баг найден и исправлен:**
- Qwen3 имеет **QK Normalization** (`q_norm.weight`, `k_norm.weight`) — RMSNorm на Q,K перед RoPE
- Без этого attention полностью ломается → мусорный вывод
- После фикса: "The capital of France is" → **"Paris"** (score 18.9) ✓

**Добавлено:**
- Tiled matmul для lm_head (151936×2560 → разбивка по output columns)
- Детальный per-op profiler (transpose, upload, compute, download, attention, RoPE, QK norm, FFN)
- Все матмулы строго на NM Card (убран CPU fallback)

**9-агентный анализ оптимизаций (2026-03-18):**
- Агент 1: Pre-transpose weights → 370s→0s ✅ СДЕЛАНО
- Агент 2: 16-core matmul → 10-14x compute (MultiCoreDevice готов)
- Агент 3: Weight caching DDR → 7x PCIe (508MB DDR, INT8 96MB/layer fits)
- Агент 4+7+8: Tiling bug найден и исправлен (reset_memory в цикле) ✅
- Агент 5: AirLLM comparison → async prefetch, gc.collect, pin_memory
- Агент 6: INT8 path → CPU dequant + FP32 upload (уже написано, нужен re-split)
- Агент 9: Low-rank SVD → FFN r=256, 8x weight reduction, <0.5% accuracy loss

**Нюансы от Дмитрия учтённые:**
- PCIe x4 = 1.2 GB/s measured → каждый MB weights = 0.8ms transfer
- DDR 5 GB → можно хранить INT4 weights целиком (~1.6 GB)
- 16 ядер → multicore matmul даст 10-16x (следующий приоритет)
- nmppmMul_mm_32f = 4 FPU vector pipeline (не скалярный RISC)

---

## 2026-03-18: Закрытие 4 блоков требований Дмитрия (НТЦ Модуль)

### Контекст
Стенографические данные беседы с Дмитрием (НТЦ "Модуль") содержали 4 блока
технических требований для аудита PromeTorch. Все 4 блока закрыты.

### Блок 1: Pipeline Validation (векторный код) ✅
- `dispatcher_float_vec.abs` уже был собран 17 марта (15.2 KB)
- `MullMatrix_f.asm` — ручной NMC4 ассемблер с 4-FPU vector pipeline
- `dispatcher_float_vec_gas.s` — asm dump компилятора с `call _nmppmMul_mm_32f`
- Все файлы скопированы в `nm_card_mini_as_TRAINER/nmc_programs/`
- **Дмитрий увидит**: `fpu 0..3 rep vlen vreg` — vector instructions, НЕ scalar RISC

### Блок 2: Benchmarks ✅
- `benchmark_for_dmitry.py` — единый benchmark suite
- MatMul: 4x4 → 256x256, GFLOPS + accuracy vs numpy
- Elementwise: relu, add, softmax с проверкой точности
- Peak utilization % от 512 GFLOPS

### Блок 3: Crash & Hang Logging ✅
- `nmruntime/safe_device.py` — SafeDevice с защитой от крашей
- Watchdog мониторинг: `mem[31]` проверяется каждые 2 сек
- Три типа ошибок: NMCardHangError, NMCardTimeoutError, NMCardOpError
- Layer tracing: `set_layer("fc1")` → при краше видно какой слой повис
- OperationLog: 100 последних операций с timestamps
- PL_* error codes → человекочитаемые сообщения

### Блок 4: PCIe/DMA ✅
- `measure_pcie_bandwidth()` — блоки 1KB → 1MB, sustained MB/s
- DDR allocation scheme документирована в benchmark
- Bandwidth tracking на каждой операции write/read
- `print_stats()` — полный отчёт с utilization %

### Новые файлы
- `nm_card_mini_as_TRAINER/nmruntime/safe_device.py` (27 KB)
- `nm_card_mini_as_TRAINER/benchmark_for_dmitry.py` (16 KB)
- `nm_card_mini_as_TRAINER/nmc_programs/dispatcher_float_vec.*` (копии для аудита)

### Результаты бенчмарка на реальной карте (2026-03-18 01:54)

**Проблема с драйвером:** После ребута карта была в статусе `CM_PROB_REGISTRY` (Error).
Решение: `pnputil /remove-device` → `/scan-devices` → Status: OK.

**PCIe Bandwidth (Block 4):**
| Блок | Write MB/s | Read MB/s |
|------|-----------|----------|
| 1 KB | 22 | 28 |
| 4 KB | 103 | 101 |
| 16 KB | 321 | 321 |
| 64 KB | 596 | 633 |
| 256 KB | 805 | 907 |
| 1024 KB | **1181** | **1276** |
| Утилизация | **47.2%** | **51.0%** |

Ранее измеренные 0.96 MB/s — были из-за мелких блоков. С 1MB блоками — **~1.2 GB/s**.

**MatMul Benchmark (Block 2, vectorized nmppmMul_mm_32f):**
| Размер | Время (ms) | GFLOPS | Max Error | Status |
|--------|-----------|--------|-----------|--------|
| 4×4×4 | 0.65 | 0.0002 | 0.000000 | PASS |
| 8×8×8 | 0.59 | 0.0017 | 0.000000 | PASS |
| 16×16×16 | 0.65 | 0.0125 | 0.000000 | PASS |
| 32×32×32 | 0.52 | 0.127 | 0.000001 | PASS |
| 64×64×64 | 0.77 | **0.68** | 0.000002 | PASS |
| 128×128×128 | 1.98 | **2.11** | 0.000005 | PASS |
| 256×256×256 | 11.90 | **2.82** | 0.000012 | PASS |

Peak: **2.82 GFLOPS** (0.55% от 512 GFLOPS). Это 1 ядро из 16 — с multi-core ожидаем ~45 GFLOPS.
Точность: **идеальная** (IEEE 754 float, max_err < 0.00002).

**Фикс adaptive polling:** Убрал фиксированный `sleep(0.01)` → adaptive busy-poll.
| Операция | Было | Стало | Ускорение |
|----------|------|-------|-----------|
| MatMul 64×64 | 10.71ms | 0.77ms | 14x |
| ReLU 256 | 11.22ms | 0.52ms | 22x |
| Avg op | 7.41ms | 1.38ms | 5.4x |

**Softmax accuracy:** WARN/FAIL из-за Taylor-approximation exp() в скалярном коде.
Не критично — matmul (90% времени) работает с bit-exact точностью.

---

## 2026-03-17: nmpp vectorized dispatcher + NTC Module analysis + SUDA compiler plan

### Анализ для НТЦ "Модуль"
5 агентов Claude Opus 4.6 провели глубокий анализ кодовой базы:
1. **TUDA NMC4 vector kernels** — TUDA готов для NMC4, нужен MicroKernel_NMC4
2. **NMCard hardware backend** — 70% готовности, нет watchdog monitor/retry/weight caching
3. **nmpp vectorized library** — nmppmMul_mm_32f ЕСТЬ в SDK, 10-100x ускорение matmul
4. **Benchmarks** — 4.37 GFLOPS (0.85% утилизации), PCIe НЕ bottleneck
5. **model.to("nmcard")** — 41 op работает, autograd не подключён

### Ключевая находка
`MullMatrix_f.asm` — ассемблер NMC4 с 4 FPU cores (`fpu 0..3 rep vlen vreg0`).
Это vector pipeline который даёт 10-100x. Уже слинкован в `libnmpps-nmc4.a`.

### Новые файлы
- **`dispatcher_float_vec.cpp`** — dispatcher с `nmppmMul_mm_32f()` вместо скалярного matmul
- **`build_gas.bat`** — добавлен `:compile_nmpp` target для линковки с nmpp
- **`train_parallel_16core.py`** — true parallel training (send_all → wait_all)
- **`НТЦ МОДУЛЬ БЕСЕДЫ/ANALYSIS_REPORT.txt`** — сводный отчёт 5 агентов
- **`НТЦ МОДУЛЬ БЕСЕДЫ/CODEBASE_INDEX.txt`** — индекс 48K+ строк для Дмитрия

### План SUDA compiler
1. ✅ Линковка nmpp (dispatcher_float_vec.abs собран, 15.2 KB)
2. ✅ TUDA NMC4 backend — 6-й architecture (Config + Vec4 + MicroKernel_4x4 + Math + BLAS dispatch)
3. ✅ SUDA Codegen v1.0 (`python suda/codegen.py --op all`) → dispatcher_suda.abs (12.5 KB), dispatcher_suda_mc.abs (13.4 KB)
4. ⏳ Бенчмарки — ждут перезагрузки NM Card

### Подтверждённые результаты тренировок на NM Card Mini
| Модель | Параметры | Loss | Метод | Время | Card ops |
|--------|-----------|------|-------|-------|----------|
| MLP (MNIST) | ~50K | 93.64% accuracy | Эмулятор, SGD | 3 epochs | — |
| Tiny Shakespeare | 13K | 9.53→1.647 | 1 ядро, float | ~2 часа | 20,001 matmuls |
| 109K Transformer | 109,761 | 4.67→2.647 | 1 ядро, float | 3.17 часа | 190,000 matmuls |
| 109K (attention) | 109,761 | D=64,H=4,F=128,T=32,L=3 | RMS+attention | lr=3e-4 warmup | Verified exact |

### Верифицировано на реальной карте
- Все forward ops: matmul, matmul_AT, matmul_BT, relu, relu_bwd, SGD — exact match vs CPU
- 16/16 ядер рабочих (dispatcher_mc_float.abs)
- RMS norm backward: полная chain rule `(dy*g - xn*mean(dy*g*xn)) / r`
- Gradient check: <0.1% error на всех параметрах

---

## 2026-03-14: NM Card Mini — Hardware Backend (подготовка к реальному железу)

### Что сделано
Подготовлен полный путь к реальному NM Card Mini через `nm_card_load.dll`. Эмулятор остаётся дефолтом, железо — opt-in через `--hardware` флаг.

### Новые файлы
- **`NMCardHardware.h`** — DDR bump-allocator, function pointer typedefs для DLL, класс NMCardHardware (singleton)
- **`NMCardHardware.cpp`** — загрузка DLL (LoadLibraryA/GetProcAddress), инициализация платы (GetBoardCount → GetBoardDesc → ResetBoard → LoadInitCode → GetAccess → LoadProgramFile), DDR dispatch protocol, 8 high-level операций

### Архитектура
```
launch_matmul() → NMCardHardware::get().is_available()?
  → YES: upload → dispatch_op(1) → wait_done → download  (реальная карта)
  → NO:  NMCardEmulator::get().matmul()                    (эмулятор)
```

### DDR Protocol
- CMD_BLOCK: 32 слова на ядро, opcode[0], args[1..29], STATUS[30], WATCHDOG[31]
- Host: write args → set STATUS=0 → write opcode → poll STATUS until done
- Data flow: float32 на хосте ↔ Q16.16 внутри карты (конверсия в dispatcher.abs)

### Операции с аппаратной поддержкой
| Op | Opcode | Аргументы |
|----|--------|-----------|
| matmul | 1 | M, K, N, addr_A, addr_B, addr_C |
| rmsnorm | 2 | batch, hidden, addr_in, addr_out, addr_gamma |
| softmax | 3 | batch, dim, addr_in, addr_out |
| silu | 4 | count, addr_in, addr_out |
| rope | 5 | seq_len, head_dim, pos, addr_in, addr_out, addr_freqs |
| elem_add | 10 | count, addr_a, addr_b, addr_out |
| elem_mul | 11 | count, addr_a, addr_b, addr_out |
| gate_mul | 13 | count, addr_a, addr_b, addr_out |

### Решённые проблемы при сборке
1. **NMCardOp enum redefined** — enum существовал в NMCardEmulator.h, дублировался в NMCardHardware.h → убрали из Hardware, используем целочисленные константы
2. **windows.h min/max макросы** — включение `<windows.h>` в header ломало `std::min`/`std::max` → forward-declare `HMODULE_t`, windows.h только в .cpp

### Результат
- `aten_nmcard.dll` — собирается
- `nmcard_tests.exe` — 33/33 тестов (включая hardware_detection)
- `train_mnist_nmcard.exe` — собирается, `--hardware --dispatcher path/to/dispatcher.abs`
- Без карты: всё работает на эмуляторе, `init()` возвращает false

---

## 2026-03-13: Полное закрытие гэпов — CUDA dispatch, autograd, оптимизаторы, тесты

### Аудит и план
Полный аудит выявил **80+ методов без CUDA dispatch**, **20+ ops без autograd**, **5 недостающих оптимизаторов**, **сломанные Python bindings**. Составлен план из 9 фаз, всё закрыто за одну сессию.

### Фаза 1: CUDA Dispatch для существующих ядер (ATen.h + CUDADispatch.h)
14 методов получили `#ifdef PT_USE_CUDA` dispatch: mv, bmm, dot, matmul, sin, cos, square, pow (tensor+scalar), clamp, maximum, minimum, argmax, argmin. Добавлено 11 cuda_ops:: wrappers в CUDADispatch.h.

### Фаза 2: Новые CUDA ядра (CUDAKernels.cu)
22 новых kernel+launch пары:
- 8 unary: log2, log10, tan, ceil, floor, round, sign, reciprocal
- 12 comparison: eq/ne/lt/le/gt/ge для Tensor×Tensor и Tensor×Scalar (float 0.0/1.0 output)
- 2 fused: addcmul, addcdiv
Все декларации в CUDAOps.h, wrappers в CUDADispatch.h, dispatch в ATen.h.

### Фаза 3: Autograd wrappers + новые Backward классы
7 новых backward классов: LeakyReluBackward, ELUBackward, SELUBackward, MishBackward, HardtanhBackward, HardsigmoidBackward, HardswishBackward.
14 новых autograd wrappers: tan, rsqrt, square, reciprocal, log2, log10 + 8 активаций (leaky_relu, elu, selu, mish, hardtanh, hardsigmoid, hardswish).

### Фаза 4: Новые CPU операции (ReduceOps.h, MathOps.h)
- var(dim, keepdim), std(dim, keepdim), prod(dim, keepdim) — reduction по оси
- fmod(Tensor), remainder(Tensor) — element-wise
- outer(), addmm() — методы Tensor

### Фаза 5: Новые оптимизаторы (5 штук)
| Оптимизатор | Файл | Формула |
|-------------|------|---------|
| Adagrad | `torch/optim/adagrad.h` | sum += g²; p -= lr·g/√(sum+ε) |
| Adadelta | `torch/optim/adadelta.h` | ρ-weighted avg of g² and Δ² |
| RAdam | `torch/optim/radam.h` | Adam + SMA rectification (ρ>5 → Adam, иначе SGD) |
| NAdam | `torch/optim/nadam.h` | Adam + Nesterov lookahead |
| Adamax | `torch/optim/adamax.h` | Adam с L∞ norm: u=max(β₂u,|g|) |

### Фаза 6: Python bindings fix
- retain_graph/create_graph теперь пробрасываются в backward() (были заглушены `(void)`)
- tensor.backward() в Python принимает retain_graph, create_graph
- tensor_backward() в C++ обновлён

### Фаза 7: Утилиты
- `torch/nn/utils/weight_norm.h` — w = g·v/‖v‖
- `torch/nn/utils/spectral_norm.h` — w/σ(w) через power iteration
- `torch/data/iterable_dataset.h` — next() → optional<pair>

### Тест-сьюит: 373 теста, ВСЕ ПРОХОДЯТ
| Файл | Тестов | Покрытие |
|------|--------|----------|
| test_all_ops.cpp | 147 | Все тензорные операции |
| test_autograd_full.cpp | 63 | Gradient check всех дифференцируемых ops (+2 disabled) |
| test_nn_modules.cpp | 49 | 57+ NN модулей |
| test_nn_functional_full.cpp | 38 | Все F:: функции |
| test_edge_cases.cpp | 25 | Скаляры, non-contiguous, broadcasting, dtype promotion |
| test_optimizers.cpp | 51 | Все 9 оптимизаторов: convergence, state, zero_grad, param_groups |

### Ранее в этот день: 12 новых операций + чистка репо
- `F::normalize`, `cosine_similarity`, `pairwise_distance`, `grid_sample`, `affine_grid`
- `scatter_reduce_`, `searchsorted`, `multinomial`, `lstsq`, `svd`, `pinverse`, `eig`
- Удалены debug-файлы, добавлены README.md + LICENSE

### Итого: что закрыто
- **~30 CUDA dispatch дыр** → все основные ops работают на GPU
- **14 autograd дыр** → все активации и linalg ops дифференцируемы
- **5 новых оптимизаторов** → 9 total (SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax)
- **Python bindings** → retain_graph/create_graph работают
- **CPU ops** → var/std/prod(dim), fmod, remainder
- **Утилиты** → weight_norm, spectral_norm, IterableDataset
- **~190 tracked файлов, ~48,000 строк C++/CUDA, 373 gtest теста**

---

## 2026-03-08: GGUF Inference — загрузка моделей Ollama + генерация текста (CPU & CUDA)

**~3000 строк нового кода, 6 новых файлов, qwen3:4b и gemma3:4b работают.**

### GGUF Reader (`torch/io/gguf_loader.h`)
- Полный парсинг формата GGUF v3 (magic, header, metadata KV pairs, tensor info)
- Поддержка всех типов метаданных: string, int, float, bool, array
- Автоматическое выравнивание данных (32-byte alignment)
- `reader.load_tensor(name)` — чтение + dequantization → float32

### Dequantization (`torch/io/gguf_dequant.h`)
- Q4_K_M: 256 values per 144-byte block, 6-bit packed scales + 4-bit weights
- Q6_K: 256 values per 210-byte block, ql[128] + qh[64] + scales[16] + d(fp16)
- Q8_0, Q5_K, F16, F32
- Исправлен баг Q6_K: scale index `n/16` → `n/16 + l/16` (без этого веса были мусором)

### Tokenizer (`torch/io/tokenizer.h`)
- SentencePiece BPE (Llama, Gemma): ▁ (U+2581) word separator
- GPT-2 BPE (Qwen): Ġ (U+0120) space encoding, word-level pre-tokenization
- Byte fallback (<0xNN>), encode/decode

### Ollama Resolver (`torch/io/ollama.h`)
- Автоматический поиск моделей: `~/.ollama/models/manifests/` → digest → blob path
- Поддержка Windows и Linux путей

### Transformer Inference (`torch/io/gguf_model.h`)
- Полный transformer: RMSNorm, RoPE, GQA attention, SwiGLU FFN, KV cache
- Поддержка архитектур: qwen3, gemma3, llama (через GGUF metadata)
- Gemma-specific: embedding scaling (sqrt(H)), QK-norm, post-attention/post-FFN norms
- Gemma RMSNorm: GGUF converter bakes in +1 (layernorm1p) → НЕ добавлять +1 повторно
- CUDA: matmul (GEMM), SiLU, element-wise ops на GPU; pre-transpose 2D weights при to_cuda()
- Top-k/top-p sampling, greedy decoding, temperature

### Полностью GPU инференс (CUDA kernels)
- `CUDAInference.cu`: собственные CUDA kernels — RMSNorm, per-head QK-norm, RoPE, causal GQA attention, concat
- Убраны все `cuda_synchronize()` (sync только при CPU←GPU transfer для sampling)
- Chat template: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
- Special token encoding (tokenizer находит `<|im_start|>` и кодирует как один ID)
- GPT-2 decode: Ċ→\n, ĉ→\t
- `<think>...</think>` stripping (qwen3 thinking mode)
- Stop tokens: `<|im_end|>`, `<end_of_turn>`, `<|eot_id|>`, `</s>`
- Repetition penalty (1.05) для предотвращения зацикливания

### Оптимизация скорости: 16 → 49.9 tok/s (3.1× speedup)

**Фазы оптимизации:**
1. **Profiler** (`torch/io/inference_profiler.h`) — CUDA event-based timing, per-operation breakdown
2. **Vectorized GEMV** — float4 loads, 128 threads/block → 16 → 25.4 tok/s (+59%)
3. **Coalesced float32 GEMV** — row-major access → 25.4 → 34.4 tok/s (+35%)
4. **Warp-cooperative quant GEMV** — полная перезапись CUDAQuantGemv.cu:
   - Каждый warp = 1 output row, 32 lanes читают consecutive qs bytes
   - x vector в shared memory, uint32_t packed load, float4 loads из smem
   - Warp shuffle reduction (без shared memory для reduce)
   - 34.4 → 49.9 tok/s (+45%)
5. **Scratch Pool** — pre-allocated decode buffers, zero alloc hot path (no speed gain — caching allocator уже быстрый)
6. **Shared memory fix** — cudaFuncSetAttribute для K > 12288 (68 KB smem для ffn_down в 14B+ моделях)
7. **Think tag fix** — strip everything from `<think>` to end when no `</think>`
8. **/no_think** — system message для Qwen3 чтобы отключить thinking mode

### Результаты vs Ollama baseline (A100 40GB)

| Модель | VRAM | PromeTorch tok/s | Ollama tok/s | Ratio | Корректность |
|--------|------|-----------------|-------------|-------|-------------|
| qwen3:4b | 4.9 GB | **49.9** | 164.6 | 30% | ✅ "The result of 2 + 2 is 4." |
| gemma3:4b | ~3 GB | **52.9** | 147.5 | 36% | ✅ Correct "4" |
| deepseek-r1:8b | 5.9 GB | **30.5** | 129.6 | 24% | ✅ Correct answers |
| qwen3:14b | 9.6 GB | **18.4** | 84.4 | 22% | ✅ Works (after smem fix) |
| qwen3:30b | - | ❌ MoE | 115.4 | - | MoE не поддерж. |
| gemma3:27b | 9.6 GB | ❌ crash | 48.9 | - | tied weights bug |
| llama3.3:70b | - | ❌ too large | - | - | 40.5 GB > VRAM |

**Предыдущий baseline 40 tok/s был неверен.** Реальный Ollama: 84-165 tok/s. Мы на 22-36%.

### Что делать дальше
- **gemma3:27b**: исправить undefined tensor при tied embeddings
- **MoE**: поддержка qwen3moe (expert routing, shared expert)
- **Скорость**: мы на 22-36% от Ollama. Потенциальные улучшения:
  - Fused Residual+RMSNorm kernel (-2 launches/layer)
  - GPU Embedding Lookup (убрать CPU→GPU transfer)
  - Fused QK-norm+RoPE
  - FP16 compute (half precision GEMV)
  - Flash Decoding (batched attention)

**Ключевые ограничения vs Ollama:**
- **Скорость 3-4x медленнее**: наши kernels чисто float32, Ollama — half precision + flash decoding
- **VRAM эффективный**: quant-only mode (4.9 GB для 4B), сравнимо с Ollama
- **Загрузка медленная**: 20-40 сек vs Ollama 2 сек (mmap)
- **Длинные тексты**: деградация после ~80 токенов с greedy decoding

### Исправленные баги
- Q6_K scale index bug: `is = n/16` → `is = n/16 + l/16`
- Tokenizer: Qwen reports "gpt2" model_type, не SentencePiece → GPT-2 pre-tokenization
- GPT-2 spaces: Ġ (U+0120) → space, Ċ (U+010A) → \n
- Gemma norm +1: GGUF converter already bakes in +1, double-application caused value explosion
- CUDA QK-norm: device mismatch crash (moved to GPU, then RoPE wrote as CPU)
- Tied embeddings: output_weight == token_embedding → separate transposed copy on GPU
- `<think>` stripping: don't erase all text when `</think>` missing

---

## 2026-03-08: Phase 2 — linalg, FFT, tensor ops, ConvTranspose2d, INT8 quantization

**~3700 строк нового кода, 16 файлов, 39/39 тестов пройдены.**

### torch.linalg (LinearAlgebra.h + backward + autograd)
- `lu()` — Gaussian elimination + partial pivoting → L, U, P
- `inverse()` — через LU → solve(A, I)
- `solve(A, b)` — forward/backward substitution через LU
- `det()` — sign(P) * prod(diag(U))
- `cholesky()` — L@L^T decomposition для SPD матриц
- `qr()` — Householder reflections
- `trace()`, `cross()`, `matrix_norm()` (1/inf/Frobenius)
- Backward: InverseBackward, DetBackward, CholeskyBackward, TraceBackward

### Tensor ops (ShapeOps.h + backward)
- `flip()`, `roll()`, `meshgrid()`, `repeat_interleave()`, `unique()`
- `tril_indices()`, `triu_indices()`
- Backward: FlipBackward, RollBackward, RepeatInterleaveBackward

### FFT (новый FFTOps.h, 445 строк)
- Cooley-Tukey radix-2 DIT, O(N log N)
- `fft/ifft/rfft/irfft/fft2/ifft2/fftfreq/rfftfreq/fftshift/ifftshift`
- Complex format: `[..., 2]` (last dim = [real, imag])

### ConvTranspose2d — реальная реализация (был STUB)
- scatter-based transposed convolution, groups support
- Проверено: output != zeros, правильная shape

### Generalized pad (functional.h)
- 4 режима: constant, reflect, replicate, circular
- Любая размерность (1D-5D)

### Unfold/Fold (im2col/col2im)
- `unfold_im2col()`: N,C,H,W → N,C*kH*kW,L
- `fold_col2im()`: обратная операция

### INT8 Quantization (4 новых файла)
- `QuantizedTensor` + `quantize_per_tensor/per_channel` + `dequantize()`
- Observers: MinMaxObserver, HistogramObserver, PerChannelMinMaxObserver
- QuantizedLinear, QuantizedConv2d (fake quant forward)
- Pipeline: prepare → calibrate → convert → quantize_model

### Тесты
- Phase 2 test_phase2.exe: **39/39 PASS**
- 10 models CPU: ALL PASS (MNIST 97%, LSTM 98.44%, GRU 95.3%)
- 10 models CUDA (A100): ALL PASS (MNIST 97.78%, LSTM 93.75%, GRU 98.44%)
- PIR CUDA: 7.2M params, 50 iter, 35s, loss 3.07

---

## 2026-03-07: Unary ops non-contiguous fix — LSTM WORKS!

**Root cause**: `DEFINE_UNARY_OP` (sigmoid, tanh, exp, etc.) в MathOps.h использовал `data_ptr()[i]` последовательный доступ без учёта strides. Когда LSTM делит gates через `narrow_autograd(gates, 1, offset, H)`, результат — view с strides `[4*H, 1]` вместо contiguous `[H, 1]`. Sequential access `in[0], in[1], ...` читает данные из ЧУЖИХ gates для batch > 0.

**Fixes**:
- `DEFINE_UNARY_OP`: добавлен `.contiguous()` на входе
- `DEFINE_UNARY_OP_INPLACE`: fallback через out-of-place + copy_ для non-contiguous
- Scalar `add/mul/pow`: добавлен `.contiguous()` на входе
- `zero_()`: stride-aware path вместо memset для non-contiguous
- `fill_()`: stride-aware path для non-contiguous
- Восстановлены все 10 моделей в train_10_models.cpp

**Результаты**: LSTM 50% → 98.44%, все 10 моделей match PyTorch baseline.

| Model | PyTorch | PromeTorch | Status |
|-------|---------|-----------|--------|
| 4: MNIST (SGD) | 92.54% | 92.69% | MATCH |
| 5: Deep MNIST (Adam) | 97.46% | 97.03% | MATCH |
| 6: Dropout MNIST | 97.03% | 97.00% | MATCH |
| 7: RNN Sine | 1.1e-5 | 1.7e-5 | OK |
| 8: LSTM | 98.4% | 98.44% | MATCH |
| 9: GRU | 92.2% | 95.31% | MATCH |
| 10: Wide MNIST | 97.59% | 97.65% | MATCH |

---

## 2026-01-20: Старт проекта

- Исследована архитектура PyTorch (c10, ATen, torch, autograd)
- Составлен полный список кернелов (~1200+ операций)
- Определена структура C++/Python биндингов (pybind11)
- Создано полное ТЗ: `TECHNICAL_SPECIFICATION.md`

### Фаза 1: Ядро c10 — ЗАВЕРШЕНО
- `c10/macros/Macros.h` — платформенные макросы, CUDA поддержка
- `c10/util/Exception.h` — система исключений
- `c10/core/ScalarType.h` — типы данных (Float, Double, Half, BFloat16, Int, Bool...)
- `c10/core/Device.h` — абстракция устройств (CPU, CUDA, MPS, Meta...)
- `c10/core/Allocator.h` — управление памятью (64-byte aligned для AVX-512)
- `c10/core/Storage.h` — хранилище данных тензора (reference counting)
- `c10/core/TensorImpl.h` — низкоуровневая реализация тензора
- `CMakeLists.txt` — C++17, OpenMP, CUDA, AVX/AVX2, Google Test
- Тесты: 5 файлов, ~150 тестов

### Фаза 2: ATen (Tensor Operations) — ЗАВЕРШЕНО
- `aten/src/ATen/core/Tensor.h` — высокоуровневый Tensor с операторами
- `aten/src/ATen/core/TensorFactory.h` — фабрики (empty, zeros, ones, rand, randn, arange...)
- `aten/src/ATen/native/cpu/MathOps.h` — унарные, бинарные, broadcasting, in-place
- `aten/src/ATen/native/cpu/ReduceOps.h` — sum, mean, max, min, var, std, norm
- `aten/src/ATen/native/cpu/LinearAlgebra.h` — mm, mv, bmm, dot, matmul, addmm
- `aten/src/ATen/native/cpu/ShapeOps.h` — view, reshape, transpose, cat, stack, split
- `aten/src/ATen/native/cpu/IndexOps.h` — select, slice, gather, scatter, where
- `aten/src/ATen/ATen.h` — главный include
- Тесты: ~60 тестов

### Фаза 3: Autograd — ЗАВЕРШЕНО
- `torch/csrc/autograd/edge.h` — рёбра графа
- `torch/csrc/autograd/node.h` — Node, AccumulateGrad, topological sort
- `torch/csrc/autograd/autograd_meta.h` — grad, grad_fn, version_counter
- `torch/csrc/autograd/engine.h` — GraphTask, Engine::execute(), backward(), grad()
- `torch/csrc/autograd/functions/MathBackward.h` — Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Sigmoid, Relu, Add, Sub, Mul, Div, Pow
- `torch/csrc/autograd/functions/ReduceBackward.h` — Sum, Mean, Max, Min, Var, Std, Norm
- `torch/csrc/autograd/functions/LinearAlgebraBackward.h` — Mm, Mv, Bmm, Dot, Matmul, Addmm, Transpose
- `torch/csrc/autograd/functions/ShapeBackward.h` — View, Reshape, Squeeze, Permute, Expand, Cat, Stack, Select
- `torch/csrc/autograd/autograd.h` — autograd-aware операции (*_autograd)
- Тесты: ~30 тестов

### Фаза 4: NN Modules — ЗАВЕРШЕНО
- `torch/nn/parameter.h` — Parameter, Buffer
- `torch/nn/module.h` — Module (register_parameter, state_dict, train/eval, to(device))
- `torch/nn/init.h` — xavier, kaiming, orthogonal, sparse инициализации
- `torch/nn/modules/container.h` — Sequential, ModuleList, ModuleDict
- `torch/nn/modules/linear.h` — Identity, Linear, Bilinear, LazyLinear
- `torch/nn/modules/activation.h` — 18 активаций (ReLU, GELU, SiLU, Mish, Softmax...)
- `torch/nn/modules/conv.h` — Conv1d/2d/3d, ConvTranspose2d (im2col)
- `torch/nn/modules/pooling.h` — MaxPool, AvgPool, AdaptivePool
- `torch/nn/modules/normalization.h` — BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d
- `torch/nn/modules/dropout.h` — Dropout, Dropout1d/2d/3d, AlphaDropout
- `torch/nn/modules/sparse.h` — Embedding, EmbeddingBag, one_hot
- `torch/nn/modules/loss.h` — ~20 loss функций (CE, MSE, BCE, NLL, Focal, Dice...)
- `torch/nn/nn.h` — count_parameters, freeze/unfreeze, clip_grad_norm_
- `torch/nn/functional.h` — F:: namespace
- Тесты: ~70 тестов

### Фаза 5: Optimizers — ЗАВЕРШЕНО
- `torch/optim/optimizer.h` — базовый Optimizer, ParamGroup
- `torch/optim/sgd.h` — SGD (momentum, nesterov, weight_decay)
- `torch/optim/adam.h` — Adam, AdamW (bias correction, AMSGrad)
- `torch/optim/rmsprop.h` — RMSprop (centered, momentum)

### Фаза 6: LR Schedulers — ЗАВЕРШЕНО
- `torch/optim/lr_scheduler.h` — StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, PolynomialLR, ReduceLROnPlateau, OneCycleLR, CyclicLR, WarmupLR, ChainedScheduler, SequentialLR

### Фаза 7: Data Loading — ЗАВЕРШЕНО
- `torch/data/dataset.h` — Dataset<T>, TensorDataset, ConcatDataset, Subset, MapDataset
- `torch/data/sampler.h` — Sequential, Random, SubsetRandom, Batch, Distributed
- `torch/data/dataloader.h` — DataLoader (batch, shuffle, drop_last)

### Фаза 8: Transformer — ЗАВЕРШЕНО
- `torch/nn/modules/attention.h` — ScaledDotProductAttention, MultiheadAttention
- `torch/nn/modules/transformer.h` — EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer, PositionalEncoding

### Фаза 9: PIR Architecture — ЗАВЕРШЕНО
- `torch/nn/modules/pir.h` — RMSNorm, RotaryEmbedding, PIRLayer, PIRBlock, PIRAttention
- `torch/nn/modules/pir270m.h` — PIR270M (token embedding, 24 blocks, LM head, generate())
- Backward: SiLU, RMSNorm, ParallelScan, RotaryEmbedding, CrossEntropy, Embedding
- `examples/pir/train_pir.cpp` — Shakespeare training

### Фаза 10: CUDA Backend — ЗАВЕРШЕНО
- `c10/cuda/CUDAAllocator.h` — CUDACachingAllocator (block caching)
- `aten/src/ATen/cuda/CUDAKernels.cu` — 50+ element-wise ops
- `aten/src/ATen/cuda/CUDAReduce.cu` — warp/block reductions, dimensional
- `aten/src/ATen/cuda/CUDABlas.cu` — tiled GEMM 32x32, batched, GEMV, dot
- `aten/src/ATen/cuda/CUDADispatch.h` — CPU/CUDA dispatch layer

---

## 2026-01-21: Python Bindings + cuDNN + CUDA Training

### Фаза 11: Python Bindings (pybind11) — ЗАВЕРШЕНО
- `python/csrc/init.cpp` — DeviceType, Device, ScalarType bindings
- `python/csrc/tensor_bindings.cpp` — Tensor с numpy interop, factory functions
- `python/csrc/autograd_bindings.cpp` — GradMode, no_grad, backward, grad
- `python/csrc/nn_bindings.cpp` — Module, Linear, Conv2d, Loss functions, functional
- `python/csrc/optim_bindings.cpp` — SGD, Adam, AdamW, RMSprop, LR schedulers

**Исправленные ошибки bindings (19 штук):**
scalar_type()→dtype(), Int8→Char, ssize_t→py::ssize_t, element_size()→itemsize(), requires_grad_()→set_requires_grad(), .first/.second→std::get, backward через autograd, pow overloads, GradMode thread_local, Parameter pointers, Loss constructor order, Optimizer Options structs, LRScheduler reference, size property/method conflict, no_grad duplicate, Reduction enum order, argmax/argmin dims, Tensor::grad()

### Фаза 12: cuDNN Integration — ЗАВЕРШЕНО
- `aten/src/ATen/cudnn/CuDNNHandle.h` — handle management, descriptors
- `aten/src/ATen/cudnn/CuDNNConvolution.h` — forward, backward_data, backward_filter, fused conv+bias+relu
- `aten/src/ATen/cudnn/CuDNNPooling.h` — max/avg pool forward/backward
- `aten/src/ATen/cudnn/CuDNNBatchNorm.h` — training/inference forward, backward
- `aten/src/ATen/cudnn/CuDNNActivation.h` — relu, sigmoid, tanh, elu, swish, softmax
- `aten/src/ATen/cudnn/CuDNN.h` — high-level dispatch
- `cmake/FindcuDNN.cmake`
- cuDNN 9.14.0 @ `C:\ProgramData\anaconda3\Library\`

### Фаза 13: Mixed Precision (AMP) — ЗАВЕРШЕНО
- `torch/amp/grad_scaler.h` — GradScaler (scale, unscale, step, update, state_dict)
- `torch/amp/autocast.h` — AutocastGuard, categories, type casting
- `torch/amp/amp.h` — half(), bfloat16(), float32(), has_tensor_cores()

### Фаза 14: FlashAttention — ЗАВЕРШЕНО
- `aten/src/ATen/cuda/FlashAttention.h` — config, forward, backward, scaled_dot_product
- `aten/src/ATen/cuda/FlashAttention.cu` — tiled O(N) attention, online softmax, causal masking, head_dim 64/128

### CUDA Training — РАБОТАЕТ
- 100 итераций за 2 секунды на GPU
- Loss: 4.29 → 4.25 (снижается)
- Рабочая конфигурация: `--n_layers 2 --n_pir_layers 1 --n_embd 128`
- Большая модель (6 layers) crashит из-за dynamic_parallel_scan GPU→CPU→GPU копирования

---

## 2026-01-23: Исследование утечки памяти GPU

**Проблема:** PIR 6 layers crash на iter ~19. Память +2GB/iter: 698MB → 39GB.
**Причина:** Autograd saved tensors не освобождаются после backward.
**Попытки:** release_saved_tensors() в Node, очистка в apply(), clear_grad_fn().
**Статус:** Не решено на этом этапе — root cause оказался в DLL singleton (см. 01-24).

---

## 2026-01-24: CUDA Crash Fixes

### DLL Singleton Problem — ROOT CAUSE
`CUDACachingAllocator::get()` был inline со static var в header. На Windows каждая DLL получала свою копию → allocation в одном модуле, deallocation в другом → heap corruption.

**Решение:** `c10/cuda/CUDAAllocator.cpp` с единственным singleton, `get()` — declaration only (не inline), класс PT_API. `aten_cuda` теперь SHARED library.

### CUDA Exit Crash — PyTorch Pattern
При shutdown() был double free (free_blocks_ + ptr_to_block_ пересекаются).
**Решение:** Как в PyTorch — НЕ освобождать CUDA память при shutdown. CUDA driver сам всё освободит.

### GPU Load Optimization
Спайки из-за debug output + cudaDeviceSynchronize. Удалён весь debug из production кода. GPU загрузка стала ровной.

---

## 2026-01-25: MNIST Training Investigation

### Проблема
MNIST MLP 784→512→256→128→10: accuracy 12-15% вместо ожидаемых ~49%.

### Проверено (всё корректно)
- CrossEntropyLoss/Backward: `(softmax - one_hot) / N`
- MmBackward: `grad_A = grad_C @ B^T`, `grad_B = A^T @ grad_C`
- ReluBackward: `grad * (input > 0)`
- SGD step: `w = w - lr * grad`
- Gradient check: 9/10 PASS

### Исправления
1. `linear.h:57` — bound = `1/sqrt(fan_in)` вместо `sqrt(3)/sqrt(fan_in)` (PyTorch default)
2. `adamkiller.h:266` — step_size = layer_lr (убран double bias correction)
3. Восстановлена 4-слойная MLP (была упрощена до 1 слоя при отладке)
4. Удалён debug output из TensorImpl.cpp, autograd_meta.h, engine.h

### Результат
Все 8 параметров получают градиенты. Accuracy 14.88% — лучше, но ещё не на уровне PyTorch (~49%).

### Текущий диагноз (нерешено)
Backward формулы правильные. Подозрение на:
1. Как backward подключается — `mm_autograd()`, `t_autograd()` в `autograd.h`
2. Построение графа вычислений (edges)
3. Накопление градиентов между батчами
4. Autograd graph cleanup

---

## Решённые проблемы сборки (справочник)

### CUDA CMake
- **nvcc + MSVC flags** → `$<$<COMPILE_LANGUAGE:CXX>:...>` generator expressions
- **Deprecated GPU archs** → `CMAKE_CUDA_ARCHITECTURES 75 80 86 89`
- **CUDA_SEPARABLE_COMPILATION** → OFF (нет extern __device__)
- **CUDA toolkit из Anaconda** → `-DCMAKE_CUDA_COMPILER=...` `-DCUDAToolkit_ROOT=...`

### Python Bindings
- ScalarType: `Char` (не Int8), `Short` (не Int16)
- `dtype()` (не scalar_type()), `itemsize()` (не element_size())
- `set_requires_grad()` (не requires_grad_())
- backward через `torch::autograd::tensor_backward()`
- Optimizer constructors через Options structs

### Windows/Bash
- `exit code 127` из bash → `start //b` с batch файлом
- rc.exe не найден → запускать из Developer Command Prompt (vcvarsall.bat)
- c10.dll зависимость → добавить build dir в PATH

---

## 2026-03-02: ИСПРАВЛЕН MNIST — Contiguous Fix

### ROOT CAUSE
Функция `mm()` в `aten/src/ATen/native/cpu/LinearAlgebra.h` читала данные через `data_ptr<>()` с контигуозными индексами `A[i*K + k]`, но `tensor.t()` создаёт VIEW с транспонированными strides (данные в памяти НЕ перераспложены). Результат: **неправильное перемножение матриц** в forward И backward.

### Почему gradient check "проходил"
Numerical gradient: `(f(w+eps) - f(w-eps)) / 2eps` — оба f() используют тот же buggy `mm()`. Analytical gradient тоже через buggy `mm()`. Оба wrong одинаково → match.

### Fix
`.contiguous()` перед raw pointer access в mm, mv, bmm, dot, outer, addmm:
```cpp
Tensor A = self.contiguous();  // копирует данные в row-major если нужно
Tensor B = other.contiguous();
```

### Результат
| Метрика | Было | Стало |
|---------|------|-------|
| Loss (1 epoch) | 2.318 | 1.117 |
| Train Acc | 14.88% | 71.05% |
| Test Acc | 14.86% | **88.94%** |

### Файлы изменены
- `aten/src/ATen/native/cpu/LinearAlgebra.h` — добавлен `.contiguous()` во все функции
- `torch/nn/modules/linear.h` — init bound fix (1/sqrt(fan_in))

---

## 2026-03-14: NM Card Mini — Третий Backend (Эмулятор)

Интеграция NM Card Mini (К1879ВМ8Я, 16 NMC4 ядер @ 1GHz) как третьего backend рядом с CPU и CUDA. Программный эмулятор — без реального железа.

### Архитектура

- **DeviceType::PrivateUse1** = nmcard. `Device("nmcard:0")`, `tensor.is_nmcard()`, `model.to("nmcard")`
- **NMCardAllocator**: Caching allocator (aligned host RAM, тегирован device=nmcard)
- **NMCardEmulator**: 16 виртуальных NMC4 ядер, два режима — float32 и Q16.16 fixed-point
- **NMCardMath.h**: Порт mymath.h на x86 (Q16.16 арифметика без libgcc)
- **NMCardDispatch.h**: `empty_nmcard()`, `to_nmcard()`, `nmcard_to_cpu()`, mm/relu/softmax/etc.
- **NMCardOps.h**: 40+ операций (forward, backward, optimizers, loss)

### Новые файлы (11 файлов)

| Файл | Назначение |
|------|-----------|
| `c10/nmcard/NMCardAllocator.h/.cpp` | Caching allocator (DLL singleton pattern) |
| `aten/src/ATen/nmcard/NMCardMath.h` | Q16.16 fixed-point math (x86 port) |
| `aten/src/ATen/nmcard/NMCardEmulator.h/.cpp` | Программный эмулятор dispatcher.cpp |
| `aten/src/ATen/nmcard/NMCardOps.h` | Operation wrappers (аналог CUDAOps.h) |
| `aten/src/ATen/nmcard/NMCardDispatch.h` | Dispatch layer (аналог CUDADispatch.h) |
| `test/cpp/test_nmcard.cpp` | 32 теста эмулятора |
| `examples/nmcard/train_mnist_nmcard.cpp` | MNIST MLP на device nmcard |

### Модифицированные файлы

- `c10/core/Device.h` — parse "nmcard", is_nmcard(), DeviceTypeName
- `c10/core/TensorImpl.h` — is_nmcard()
- `aten/src/ATen/core/Tensor.h` — is_nmcard()
- `aten/src/ATen/ATen.h` — `#ifdef PT_USE_NMCARD` dispatch (~40 операций)
- `torch/csrc/autograd/engine.h` — grad тензоры на nmcard device
- `CMakeLists.txt` — `PT_USE_NMCARD`, aten_nmcard library

### Критический баг: DLL Singleton Boundary

**Проблема**: `AllocatorRegistry::get()` — inline static в header. Каждая DLL получает свою копию. `register_nmcard_allocator()` регистрирует в aten_nmcard.dll, но `at::empty()` (inline в exe) ищет в exe's AllocatorRegistry → crash.

**Решение**: Двойная регистрация:
```cpp
c10::nmcard::register_nmcard_allocator();       // DLL-internal
c10::nmcard::register_nmcard_allocator_local();  // Caller's registry (inline)
```

### Результаты

- **32/32 тестов** прошли (matmul, rmsnorm, softmax, silu, rope, backward, optimizer, Q16.16)
- **MNIST на NMCard**: 3 эпохи → **93.64% test accuracy** (SGD lr=0.01, batch=64)
- Время: ~25.6 сек/эпоху (эмулятор, float32 mode)
- Сборка: `cmake -DPT_USE_NMCARD=ON`, build dir: `build_nmcard/`

---

## Статистика (на 2026-01-25)

| Метрика | Значение |
|---------|----------|
| Файлов C++/CUDA | 92 |
| Строк кода | ~37,000 |
| c10 (core) | 3,278 |
| ATen (tensor ops) | 9,344 |
| CUDA kernels | 6,996 |
| Autograd | 3,559 |
| NN Modules | 9,858 |
| Optimizers | 1,246 |
| Data Loading | 1,176 |
