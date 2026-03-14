# PROMEPEDIA — Полная энциклопедия PromeTorch

> Версия: 0.1.0 | Дата: 2026-03-15 | ~53,000 строк C++/CUDA

---

## Содержание

1. [Обзор проекта](#1-обзор-проекта)
2. [Философия: адаптация под железо](#2-философия-адаптация-под-железо)
3. [Архитектура фреймворка](#3-архитектура-фреймворка)
4. [TUDA — кросс-платформенный CPU dispatch](#4-tuda--кросс-платформенный-cpu-dispatch)
5. [Российское железо](#5-российское-железо)
6. [Backends](#6-backends)
7. [Тензорная библиотека (c10 + ATen)](#7-тензорная-библиотека-c10--aten)
8. [Автоматическое дифференцирование](#8-автоматическое-дифференцирование)
9. [Нейросетевые модули](#9-нейросетевые-модули)
10. [Оптимизаторы и планировщики](#10-оптимизаторы-и-планировщики)
11. [Распределённое обучение](#11-распределённое-обучение)
12. [GGUF Inference Engine](#12-gguf-inference-engine)
13. [CUDA Backend](#13-cuda-backend)
14. [NM Card Mini Backend](#14-nm-card-mini-backend)
15. [LinQ H1M Backend](#15-linq-h1m-backend)
16. [Python Bindings](#16-python-bindings)
17. [Сборка и установка](#17-сборка-и-установка)
18. [Docker-контейнеры](#18-docker-контейнеры)
19. [Тестирование](#19-тестирование)
20. [Хронология разработки](#20-хронология-разработки)
21. [Решённые инженерные проблемы](#21-решённые-инженерные-проблемы)
22. [Будущие планы](#22-будущие-планы)

---

## 1. Обзор проекта

**PromeTorch** — полнофункциональный фреймворк глубокого обучения, написанный с нуля на C++17 и CUDA. Реализует API, совместимый с PyTorch, но с ключевым отличием: нативная поддержка российских процессоров и операционных систем.

### Статистика

| Метрика | Значение |
|---------|----------|
| Строк кода (C++/CUDA) | ~53,000+ |
| Файлов исходного кода | 120+ |
| Тензорных операций | 110+ |
| Backward-функций (autograd) | 90+ |
| NN-модулей | 57+ |
| Оптимизаторов | 9 |
| LR Schedulers | 13 |
| Backends | 5 (CPU, CUDA, NMCard, LinQ, Distributed) |
| Toolchains | 6 |
| Тестов | 440+ |

---

## 2. Философия: адаптация под железо

Существующие ML-фреймворки (PyTorch, TensorFlow, ONNX Runtime) требуют чтобы **железо адаптировалось под фреймворк**: производитель пишет backend-плагин, делает форк, или пользователь перекомпилирует под конкретную платформу.

PromeTorch работает наоборот: **фреймворк адаптируется под железо**.

| Аспект | PyTorch / ONNX RT | PromeTorch |
|--------|------------------|------------|
| Байкал-М/С | Нет поддержки | NEON micro-kernels, tuned под L1d |
| Эльбрус | «Скомпилится LCC-совместимым GCC» | E2K VLIW micro-kernel, 4 FMA units |
| NM Card Mini | Никто не знает что это | Полный backend, Q16.16 |
| Пользовательский код | Нужны #ifdef'ы | `model.to("nmcard")` — и всё |

Один исходник — один `cmake -DCMAKE_TOOLCHAIN_FILE=...` — собирается и работает на любой из поддерживаемых платформ.

---

## 3. Архитектура фреймворка

### Слои абстракции

```
┌─────────────────────────────────────────────┐
│             Python (promethorch)             │  pip install
├─────────────────────────────────────────────┤
│          torch:: (NN, Optim, Data)          │  Высокоуровневый API
├─────────────────────────────────────────────┤
│         at:: (ATen tensor operations)       │  Тензорные операции
├─────────┬───────────┬──────────┬────────────┤
│   CPU   │   CUDA    │  NMCard  │    LinQ    │  Backend dispatch
│  (TUDA) │ (cuDNN)   │ (Q16.16) │  (INT8)   │
├─────────┴───────────┴──────────┴────────────┤
│              c10:: (Core)                   │  Allocator, Device, Storage
└─────────────────────────────────────────────┘
```

### Структура каталогов

```
c10/                           Ядро фреймворка
  core/                        Allocator, Device, Storage, TensorImpl, ScalarType, Scalar
  nmcard/                      NMCardAllocator (PrivateUse1)
  linq/                        LinQAllocator (PrivateUse2)

aten/src/ATen/                 Тензорные операции
  core/                        Tensor.h, TensorFactory.h, TensorOptions.h
  native/cpu/                  CPU реализации
    MathOps.h                  20 unary + 12 binary ops (через VecF)
    ReduceOps.h                sum, mean, var, std, norm, argmax/argmin (через VecF)
    LinearAlgebra.h            mm, mv, bmm, dot, outer, addmm (через TudaBLAS)
    ShapeOps.h                 view, reshape, transpose, permute, cat, stack, split
    IndexOps.h                 index, index_put, scatter, gather, boolean_index
    VectorizedOps.h            Cephes polynomials (exp, log, sin, cos, tanh, sigmoid)
    PromeBLAS.h                Thin wrapper → TudaBLAS
    tuda/                      TUDA: кросс-платформенный dispatch
      TudaConfig.h             Compile-time arch detection, GemmTuning
      TudaVec.h                VecF: Vec8 (AVX2), Vec4 (NEON/E2K), Vec1 (Scalar)
      TudaBLAS.h               Goto BLAS: sgemm, sgemm_nt, sgemv, sdot, saxpy
      TudaMath.h               Vectorized exp/log/sin/cos/tanh/sigmoid/relu/silu/gelu
      kernels/avx2/            (inline в TudaBLAS.h) 6×16 micro-kernel
      kernels/neon/            MicroKernel_4x8.h (A57), MicroKernel_8x12.h (A75)
      kernels/e2k/             MicroKernel_4x4.h (Elbrus VLIW)
      kernels/scalar/          MicroKernel_Scalar.h (fallback)
  cuda/                        CUDA kernels
    CUDAKernels.cu             Element-wise, unary, comparison, fused ops
    CUDABlas.cu                cuBLAS-free GEMM
    CUDAReduce.cu              Parallel reductions
    FlashAttention.cu          O(N) memory attention
    CUDAQuantGemv.cu           Warp-cooperative quantized GEMV
    CUDADispatch.h             Auto-dispatch CPU↔CUDA
  cudnn/                       cuDNN wrappers
    CuDNNConvolution.h         Conv2d forward/backward
    CuDNNPooling.h             MaxPool, AvgPool
    CuDNNBatchNorm.h           BatchNorm forward/backward
    CuDNNActivation.h          ReLU, Sigmoid, Tanh
  nmcard/                      NM Card Mini backend
    NMCardEmulator.h/.cpp      16-ядерный эмулятор NMC4 (float32 + Q16.16)
    NMCardOps.h                Launch wrappers
    NMCardDispatch.h           Tensor-level dispatch (40+ ops)
    NMCardMath.h               Q16.16 fixed-point arithmetic
    NMCardHardware.h/.cpp      DLL loading, DDR dispatch protocol
  linq/                        LinQ H1M backend
    LinQEmulator.h             INT8 GEMM + FP32 эмулятор (50+ ops)
    LinQOps.h                  Launch wrappers
    LinQDispatch.h             Tensor-level dispatch (63 блока)

torch/                         Высокоуровневый API
  csrc/autograd/               Автоматическое дифференцирование
    engine.h                   Backward pass engine
    node.h, edge.h             Вычислительный граф
    autograd_meta.h            Метаданные для тензоров с grad
    autograd.h                 Autograd wrappers (mm_autograd, relu_autograd, ...)
    functions/                 90+ backward-классов
      MathBackward.h           ReluBackward, SigmoidBackward, TanhBackward, ...
      LinearAlgebraBackward.h  MmBackward, MvBackward, BmmBackward, ...
      ReduceBackward.h         SumBackward, MeanBackward, ...
      ShapeBackward.h          ViewBackward, TransposeBackward, ...
  autograd/function.h          Custom autograd functions (CRTP)
  nn/                          Нейросетевые модули
    parameter.h                Parameter (Tensor + requires_grad)
    module.h                   Module base class + hooks
    init.h                     Weight initialization (Xavier, Kaiming, ...)
    nn.h                       Convenience include
    functional.h               Functional API
    modules/                   Конкретные модули
      linear.h                 Linear, Bilinear, LazyLinear
      activation.h             ReLU, GELU, SiLU, Mish, ELU, SELU, ...
      conv.h                   Conv1d, Conv2d, Conv3d, ConvTranspose2d
      pooling.h                MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
      normalization.h          BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm
      dropout.h                Dropout, Dropout2d
      loss.h                   CrossEntropy, MSE, L1, BCE, NLL, KLDiv, ...
      transformer.h            MultiheadAttention, TransformerEncoder/Decoder
      pir.h                    PIR architecture (RoPE, RMSNorm, SwiGLU)
      rnn.h                    RNN, LSTM, GRU (multi-layer)
      sparse.h                 Embedding, EmbeddingBag
      container.h              Sequential, ModuleList, ModuleDict
    utils/
      weight_norm.h            Weight normalization
      spectral_norm.h          Spectral normalization
  optim/                       Оптимизаторы
    optimizer.h                Base Optimizer class
    sgd.h                      SGD (with momentum, weight decay, Nesterov)
    adam.h                     Adam, AdamW
    rmsprop.h                  RMSprop
    adagrad.h                  Adagrad
    adadelta.h                 Adadelta
    radam.h                    RAdam (rectified Adam)
    nadam.h                    NAdam (Adam + Nesterov)
    adamax.h                   Adamax (L∞ norm Adam)
    lr_scheduler.h             13 LR schedulers
    optim.h                    Convenience include
  distributed/                 Распределённое обучение
    distributed.h              AllReduce, Broadcast, Scatter, DataParallel
  data/                        Загрузка данных
    dataset.h                  Dataset base class
    sampler.h                  RandomSampler, SequentialSampler
    dataloader.h               DataLoader (batching, shuffling)
    iterable_dataset.h         IterableDataset
  io/                          Ввод/вывод
    gguf_model.h               GGUF inference engine
  amp/                         Mixed precision
    grad_scaler.h              GradScaler
    autocast.h                 Autocast context
  serialization.h              Save/Load (PTOR binary format)
  utils/checkpoint.h           Gradient checkpointing

cmake/toolchains/              CMake toolchains
  aarch64-baikal-m.cmake       Байкал-М (Cortex-A57, NEON)
  aarch64-baikal-s.cmake       Байкал-С (Cortex-A75, NEON + dotprod)
  e2k-elbrus.cmake             Эльбрус 8C/16C (LCC compiler)
  x86_64-astra.cmake           Astra Linux SE (x86, AVX2)
  x86_64-alt.cmake             ALT Linux SP (x86, AVX2)
  x86_64-redos.cmake           РЕД ОС (x86, AVX2)

scripts/                       Build-скрипты
  build-elbrus.sh              Нативная сборка на Elbrus OS
  build-astra.sh               Сборка на Astra Linux
  build-alt.sh                 Сборка на ALT Linux
  build-redos.sh               Сборка на РЕД ОС
  build-baikal-cross.sh        Кросс-компиляция для Baikal-M/S

docker/                        Docker-контейнеры
  Dockerfile.astra             Astra Linux SE
  Dockerfile.alt               ALT Linux p10
  Dockerfile.redos             РЕД ОС (CentOS-based)
  Dockerfile.elbrus            Elbrus (cross-compile)
  Dockerfile.baikal            Baikal (ARM cross-compile)
  docker-compose.yml           Orchestration
  build-all.sh                 Build all platforms

promethorch/                   Python package
  __init__.py                  Package init, device types
  nn/__init__.py               Neural network modules
  optim/__init__.py            Optimizers
```

---

## 4. TUDA — кросс-платформенный CPU dispatch

TUDA (PromeTorch Unified Device Architecture) — система compile-time выбора оптимальных SIMD-инструкций.

### Compile-time detection

```
__AVX2__ && __FMA__      → AVX2 (Intel/AMD)
__aarch64__ + dotprod    → NEON_A75 (Байкал-С)
__aarch64__              → NEON_A57 (Байкал-М)
__e2k__ / __elbrus__     → E2K (Эльбрус)
(ничего)                 → SCALAR (любая платформа)
```

### VecF — единый SIMD-тип

| Платформа | VecF | Width | Register |
|-----------|------|-------|----------|
| x86 AVX2 | Vec8 | 8 | __m256 |
| ARM NEON | Vec4 | 4 | float32x4_t |
| E2K | Vec4 | 4 | float[4] (LCC auto-vectorizes) |
| Scalar | Vec1 | 1 | float |

Все операции VecF: `load`, `store`, `broadcast`, `+`, `-`, `*`, `/`, `fmadd`, `hsum`, `abs`, `neg`, `max`, `min`, `sqrt`, `rsqrt`, `reciprocal`, `ceil`, `floor`, `round`.

### GEMM Micro-kernels

| Arch | MR×NR | Accumulators | Cache fit | Peak FLOP/cycle |
|------|-------|-------------|-----------|-----------------|
| AVX2 6×16 | 12 YMM | 12 FMA | L2 72KB | 96 |
| NEON_A75 8×12 | 24 V | 24 FMLA | L1d 64KB | 48 |
| NEON_A57 4×8 | 8 V | 8 FMLA | L1d 32KB | 16 |
| E2K 4×4 | 16 scalar | 16 FMA | L1d 64KB | 16 |
| Scalar 4×4 | 16 scalar | — | — | 4 |

### Goto BLAS алгоритм

```
for jc = 0..N step NC:
  for pc = 0..K step KC:
    pack_b(B[pc:pc+KC, jc:jc+NC])     → L3/L2
    for ic = 0..M step MC:
      pack_a(A[ic:ic+MC, pc:pc+KC])   → L2/L1
      macro_kernel(MC, NC, KC)         → micro-kernel calls
```

---

## 5. Российское железо

### Эльбрус (МЦСТ)

- **Архитектура**: E2K VLIW (Very Long Instruction Word)
- **Процессоры**: Эльбрус-8С (8 ядер, 1.3 ГГц), Эльбрус-16С (16 ядер), Эльбрус-8СВ
- **Компилятор**: LCC (MCST C/C++ Compiler) — GCC-совместимый frontend
- **Особенность**: VLIW позволяет выдавать до 24 операций за такт. LCC делает software pipelining автоматически — поэтому E2K micro-kernel написан на чистом C (не intrinsics)
- **TUDA path**: `E2K_V5` (Эльбрус-8С) / `E2K_V6` (Эльбрус-8СВ, 16С)
- **ОС**: Elbrus OS (PDK LE), Astra Linux SE for Elbrus

### Байкал-М (Байкал Электроникс)

- **SoC**: BE-M1000
- **CPU**: 8× ARM Cortex-A57 @ 1.5 ГГц
- **GPU**: Mali-T628 (не используется для ML)
- **L1d**: 32KB, **L2**: 2MB
- **TUDA path**: `NEON_A57` — MR=4, NR=8, MC=48, KC=128
- **Toolchain**: `aarch64-linux-gnu-gcc`, `-march=armv8-a+simd -mtune=cortex-a57`
- **ОС**: Astra Linux SE, ALT Linux

### Байкал-С (Байкал Электроникс)

- **SoC**: BE-S1000
- **CPU**: 48× ARM Cortex-A75 @ 2.0 ГГц
- **L1d**: 64KB, **L2**: 256KB per core, **L3**: 32MB
- **TUDA path**: `NEON_A75` — MR=8, NR=12, MC=64, KC=256
- **Toolchain**: `aarch64-linux-gnu-gcc`, `-march=armv8.2-a+fp16+dotprod -mtune=cortex-a75`
- **Особенность**: dotprod extension для INT8 MAC операций

### NM Card Mini (НТЦ Модуль)

- **Процессор**: K1879VM8YA
- **Ядра**: 16× NMC4 tensor DSP @ 1 ГГц
- **Арифметика**: Q16.16 fixed-point (16 бит целая + 16 бит дробная)
- **Интерфейс**: PCIe x4
- **Backend**: PrivateUse1, `NMCardEmulator` + `NMCardHardware` (DLL loading)
- **Протокол**: DDR bump-allocator, CMD_BLOCK (32 слова на ядро)
- **Результат**: MNIST 93.64% accuracy

### LinQ H1M

- **Тип**: NPU (Neural Processing Unit)
- **Вычисления**: INT8 GEMM с INT32 аккумуляцией (96 TOPS peak)
- **Обучение**: FP32 GEMM
- **Backend**: PrivateUse2, `LinQEmulator` (50+ операций)
- **Квантизация**: FP32 → INT8 per-tensor symmetric

---

## 6. Backends

### Dispatch иерархия

Каждая тензорная операция проверяет device тензора и вызывает соответствующий backend:

```cpp
inline Tensor Tensor::exp() const {
#ifdef PT_USE_NMCARD
    if (is_nmcard()) { return nmc_ops::exp(*this); }
#endif
#ifdef PT_USE_LINQ
    if (device().is_linq()) { return linq_dispatch::exp(*this); }
#endif
#ifdef PT_USE_CUDA
    if (is_cuda()) { return cuda_ops::exp(*this); }
#endif
    return native::exp(*this);   // CPU (через TUDA VecF)
}
```

### Device types

| DeviceType | Index | Имя | Allocator |
|---|---|---|---|
| CPU | 0 | `cpu` | System malloc |
| CUDA | 1 | `cuda` | CUDACachingAllocator |
| PrivateUse1 | 20 | `nmcard` | NMCardAllocator |
| PrivateUse2 | 21 | `linq` | LinQAllocator |

---

## 7. Тензорная библиотека (c10 + ATen)

### c10 — Core library

- **Allocator**: Virtual base class. `allocate()` → `DataPtr`. `raw_deallocate()` для кэширования
- **Device**: `DeviceType` enum + `DeviceIndex`. Parse from string: `"cuda:0"`, `"nmcard"`, `"linq"`
- **Storage**: Reference-counted memory block. `StorageImpl` owns `DataPtr`
- **TensorImpl**: Метаданные тензора: sizes, strides, storage_offset, dtype, device
- **ScalarType**: Float, Double, Half, Int, Long, Bool, BFloat16, Complex, Int8

### ATen — операции

**Unary (20)**: neg, abs, sqrt, rsqrt, square, exp, log, log2, log10, sin, cos, tan, tanh, sigmoid, relu, ceil, floor, round, sign, reciprocal

**Binary (12)**: add, sub, mul, div, pow, fmod, remainder, maximum, minimum, addcmul, addcdiv, where

**Reduction (10)**: sum, sum(dim), mean, mean(dim), var, std, prod, max, min, argmax, argmin, norm

**Linear algebra (8)**: mm, mv, bmm, dot, outer, addmm, matmul, inverse

**Shape (15)**: view, reshape, transpose, permute, contiguous, expand, repeat, cat, stack, split, chunk, squeeze, unsqueeze, narrow, slice

**Index (6)**: index, index_put, scatter, scatter_add, gather, boolean_index

---

## 8. Автоматическое дифференцирование

### Архитектура

```
Forward pass:  Tensor → grad_fn (Node) → Edge → next Node
Backward pass: Engine::execute() → topological sort → process each Node
```

### Backward classes (90+)

`MmBackward`, `MvBackward`, `BmmBackward`, `DotBackward`, `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`, `PowBackward`, `ReluBackward`, `SigmoidBackward`, `TanhBackward`, `SoftmaxBackward`, `LogSoftmaxBackward`, `SumBackward`, `MeanBackward`, `ViewBackward`, `TransposeBackward`, `PermuteBackward`, `CatBackward`, `SliceBackward`, `NarrowBackward`, `ExpandBackward`, `RepeatBackward`, `SqueezeBackward`, `UnsqueezeBackward`, `CrossEntropyBackward`, `MSELossBackward`, `L1LossBackward`, `BCELossBackward`, `NLLLossBackward`, `ExpBackward`, `LogBackward`, `SqrtBackward`, `AbsBackward`, `NegBackward`, `SinBackward`, `CosBackward`, `LeakyReluBackward`, `ELUBackward`, `SELUBackward`, `MishBackward`, `HardtanhBackward`, `HardsigmoidBackward`, `HardswishBackward`, ...

### Custom Autograd Functions

```cpp
class MyFunc : public torch::autograd::Function<MyFunc> {
    static Tensor forward(FunctionCtx& ctx, const Tensor& input) {
        ctx.save_for_backward({input});
        return input * input;
    }
    static std::vector<Tensor> backward(FunctionCtx& ctx, const Tensor& grad) {
        auto saved = ctx.get_saved_tensors();
        return {2.0f * saved[0] * grad};
    }
};
```

---

## 9. Нейросетевые модули

### Полный список (57+)

**Linear**: Linear, Bilinear, LazyLinear
**Convolution**: Conv1d, Conv2d, Conv3d, ConvTranspose2d
**Activation**: ReLU, ReLU6, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU/Swish, Mish, Sigmoid, Tanh, Hardtanh, Hardsigmoid, Hardswish, Softmax, LogSoftmax, Softplus, Softsign, Softmin, Threshold
**Normalization**: BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm2d, RMSNorm
**Pooling**: MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool1d, AdaptiveAvgPool2d, GlobalAvgPool
**Recurrent**: RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU
**Transformer**: MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder, PositionalEncoding
**Dropout**: Dropout, Dropout2d, AlphaDropout
**Embedding**: Embedding, EmbeddingBag
**Container**: Sequential, ModuleList, ModuleDict
**Loss**: MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, KLDivLoss, HuberLoss, SmoothL1Loss, CTCLoss, FocalLoss, DiceLoss, TripletMarginLoss, CosineEmbeddingLoss, MultiMarginLoss

---

## 10. Оптимизаторы и планировщики

### Оптимизаторы (9)

| Оптимизатор | Формула | Файл |
|---|---|---|
| SGD | p -= lr·(g + wd·p + μ·buf) | `sgd.h` |
| Adam | m = β₁m+(1-β₁)g; v = β₂v+(1-β₂)g²; p -= lr·m̂/(√v̂+ε) | `adam.h` |
| AdamW | Adam + decoupled weight decay | `adam.h` |
| RMSprop | v = αv+(1-α)g²; p -= lr·g/(√v+ε) | `rmsprop.h` |
| Adagrad | sum += g²; p -= lr·g/√(sum+ε) | `adagrad.h` |
| Adadelta | ρ-weighted avg of g² and Δ² | `adadelta.h` |
| RAdam | Adam + SMA rectification | `radam.h` |
| NAdam | Adam + Nesterov lookahead | `nadam.h` |
| Adamax | Adam с L∞ norm | `adamax.h` |

### LR Schedulers (13)

StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CyclicLR, OneCycleLR, LinearLR, ConstantLR, SequentialLR, ChainedScheduler, WarmupLR

---

## 11. Распределённое обучение

### API

```cpp
#include "torch/distributed/distributed.h"
using namespace torch::distributed;

dist::init(world_size);              // Инициализация
dist::all_reduce(tensor, rank, op); // AllReduce
dist::broadcast(tensor, rank, src); // Broadcast
dist::scatter(tensor, rank);        // Scatter
dist::finalize();                    // Завершение
```

### Операции

| Операция | Описание |
|---|---|
| AllReduce SUM | Все ранки суммируют свои тензоры |
| AllReduce AVG | Среднее по всем ранкам |
| AllReduce MAX/MIN | Максимум/минимум по ранкам |
| Broadcast | Один ранк отправляет всем |
| Scatter | Разбивает batch по ранкам |
| DataParallel | Автоматический data parallelism |

### Backend

Shared-memory (multi-thread) — работает с любыми device'ами (CPU, CUDA, NMCard, LinQ). Барьерная синхронизация через `std::condition_variable`.

---

## 12. GGUF Inference Engine

Загрузка и запуск моделей в формате GGUF (llama.cpp / Ollama).

### Поддерживаемые модели
- Qwen3 (4B, 14B)
- Gemma3 (4B)
- DeepSeek-R1 (8B)
- Llama 3.x

### Форматы квантизации
Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32

### Оптимизации GPU
- Warp-cooperative GEMV с coalesced access
- Shared memory для x-вектора
- float4 packed loads для квантизованных весов
- `cudaFuncSetAttribute` для dynamic shared memory

### Производительность
- 49.9 tok/s (qwen3:4b, A100)
- 30.5 tok/s (deepseek-r1:8b, A100)
- VRAM: 4.9 GB (qwen3:4b), 9.6 GB (qwen3:14b)

---

## 13. CUDA Backend

### Собственные kernels
- Element-wise: add, sub, mul, div, neg, abs, sqrt, exp, log, sin, cos, tanh, sigmoid, relu
- Comparison: eq, ne, lt, le, gt, ge (Tensor×Tensor и Tensor×Scalar)
- Fused: addcmul, addcdiv
- Reduction: sum, sum_dim, max, min, argmax, argmin
- GEMM/GEMV/BMM
- Softmax, LogSoftmax

### cuDNN Integration
- Convolution: forward + backward (data + weight)
- Pooling: MaxPool, AvgPool
- BatchNorm: forward + backward (running mean/var)
- Activation: ReLU, Sigmoid, Tanh

### FlashAttention
- O(N) memory (не O(N²))
- Causal masking
- Multi-head support

### Mixed Precision (AMP)
- GradScaler: dynamic loss scaling, inf/nan detection
- Autocast: auto FP16 для matmul, FP32 для reductions

---

## 14. NM Card Mini Backend

### Эмулятор
16 виртуальных NMC4 ядер, два режима:
- **float32**: полная точность для отладки
- **Q16.16**: fixed-point как на реальном железе

### Hardware path
```
launch_op() → NMCardHardware::get().is_available()?
  → YES: upload → dispatch_op → wait_done → download  (карта)
  → NO:  NMCardEmulator::get().op()                    (эмулятор)
```

### DDR Protocol
- CMD_BLOCK: 32 слова на ядро
- Host → Device: write args → set STATUS=0 → write opcode → poll STATUS
- Data: float32 ↔ Q16.16 конверсия в dispatcher.abs

### Поддерживаемые операции (40+)
matmul, rmsnorm, softmax, silu, rope, add, mul, neg, abs, sqrt, exp, log, relu, sigmoid, tanh, clamp, comparison ops, reductions, fill, copy, addcmul, addcdiv, ...

---

## 15. LinQ H1M Backend

### Эмулятор
32-ядерный NPU с INT8 GEMM + FP32 training.

### Операции (50+)

**Unary**: neg, abs, sqrt, rsqrt, square, reciprocal, exp, log, log2, log10, sin, cos, tan, tanh, sigmoid, relu, silu, gelu, leaky_relu, ceil, floor, round, sign

**Binary**: add, sub, mul, div, maximum, minimum

**Scalar**: add_scalar, mul_scalar, pow_scalar

**Comparison**: eq, ne, lt, le, gt, ge (Tensor + Scalar variants)

**Reduction**: sum, mean, max, min, argmax, argmin

**Matrix**: mm, mv, dot, softmax

**Normalization**: layernorm, rmsnorm

**Memory**: fill, zero, copy

**Fused**: addcmul, addcdiv

**Quantization**: fp32→int8, int32→fp32, int8 GEMM

---

## 16. Python Bindings

### Установка

```bash
pip install -e .
```

### Использование

```python
import promethorch as pt

# Создание тензора
x = pt.tensor([1.0, 2.0, 3.0])

# Нейросеть
model = pt.nn.Sequential(
    pt.nn.Linear(784, 256),
    pt.nn.ReLU(),
    pt.nn.Linear(256, 10)
)

# Оптимизатор
optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
```

### Env переменные

```bash
PT_USE_CUDA=1 pip install .    # С CUDA
PT_USE_LINQ=1 pip install .    # С LinQ
```

---

## 17. Сборка и установка

### CMake опции

| Опция | Default | Описание |
|---|---|---|
| `PT_USE_TUDA` | ON | TUDA кросс-платформенный dispatch |
| `PT_USE_CUDA` | OFF | CUDA backend |
| `PT_USE_CUDNN` | OFF | cuDNN ускорение |
| `PT_USE_NMCARD` | OFF | NM Card Mini |
| `PT_USE_LINQ` | OFF | LinQ H1M NPU |
| `PT_USE_AVX2` | ON | AVX2 (x86) |
| `PT_USE_OPENMP` | OFF | OpenMP |
| `PT_BUILD_TESTS` | ON | Тесты |
| `PT_BUILD_PYTHON` | OFF | Python bindings |

### Кросс-компиляция

```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-baikal-m.cmake
cmake --build . -j$(nproc)
scp -r build/ user@baikal:/opt/promethorch/
```

---

## 18. Docker-контейнеры

### Структура

```
docker/
  Dockerfile.astra     Debian bullseye → симуляция Astra Linux SE
  Dockerfile.alt       ALT Linux p10
  Dockerfile.redos     CentOS 8 → симуляция РЕД ОС
  Dockerfile.elbrus    Debian → E2K cross-compile simulation
  Dockerfile.baikal    Debian + aarch64 cross-compiler
  docker-compose.yml   Все 5 платформ
  build-all.sh         Последовательная сборка всех
```

### Использование

```bash
# Все платформы
docker compose up --build

# Конкретная платформа
docker build -t promethorch-astra -f docker/Dockerfile.astra ..
docker run promethorch-astra ./build_astra/tuda_tests
```

---

## 19. Тестирование

### Тест-сьюит

| Файл | Тестов | Покрытие |
|---|---|---|
| test_all_ops.cpp | 147 | Все тензорные операции |
| test_autograd_full.cpp | 63 | Gradient check |
| test_nn_modules.cpp | 49 | 57+ NN модулей |
| test_optimizers.cpp | 50+ | 9 оптимизаторов |
| test_tuda.cpp | 38 | VecF, Math, SGEMM, SGEMV |
| test_linq.cpp | 34 | LinQ backend полностью |
| test_nmcard.cpp | 32 | NMCard backend |
| test_distributed.cpp | 7 | AllReduce, Broadcast, Scatter |
| test_edge_cases.cpp | 20+ | Крайние случаи |
| test_nn_functional.cpp | 15+ | Functional API |
| **Итого** | **440+** | **Все проходят** |

---

## 20. Хронология разработки

| Дата | Событие |
|------|---------|
| 2026-01 | Начало проекта. c10 core, ATen, basic autograd |
| 2026-01-25 | MNIST MLP работает (88.94% → 97.65%) |
| 2026-02 | NN модули (57+), оптимизаторы, LR schedulers |
| 2026-02 | CUDA backend, cuDNN интеграция |
| 2026-03-01 | FlashAttention, Mixed Precision |
| 2026-03-02 | Баг mm() non-contiguous найден и исправлен |
| 2026-03-07 | 7 критических фич: Custom Autograd, Hooks, Serialization, RNN/LSTM/GRU, Channels-last |
| 2026-03-07 | Все 10 моделей работают (MNIST, LSTM, GRU match PyTorch baseline) |
| 2026-03-08 | GGUF inference engine (49.9 tok/s GPU) |
| 2026-03-13 | CPU SIMD оптимизация (46x → 1.75x vs PyTorch) |
| 2026-03-13 | Закрытие всех гэпов: 80+ CUDA dispatch, 20+ autograd, 5 оптимизаторов |
| 2026-03-14 | NM Card Mini backend (эмулятор + hardware DLL) |
| 2026-03-15 | TUDA (Baikal/Elbrus/AVX2), LinQ H1M, Distributed, Docker, Python packaging |

---

## 21. Решённые инженерные проблемы

| Проблема | Симптом | Решение |
|----------|---------|---------|
| DLL Singleton | inline static → разные instances в DLL | `.cpp` файл с единственным singleton |
| CUDA Shutdown | double free при cudaFree | Не освобождать CUDA память (как PyTorch) |
| mm() non-contiguous | 15% accuracy вместо 97% | `.contiguous()` перед raw pointer access |
| copy_() strided | Данные мусор после copy_ view | Multi-dim index → physical offset |
| nvcc + MSVC flags | Compilation errors | `$<$<COMPILE_LANGUAGE:CXX>:...>` |
| NMCard DLL Allocator | Разные instances per DLL | Двойная регистрация |
| LinQ raw_deleter | Abstract class instantiation | Override в LinQAllocator |
| torch:: namespace | linq_dispatch not found | `at::linq_dispatch::` полная квалификация |

---

## 22. Будущие планы

| Фаза | Компонент | Описание |
|------|-----------|----------|
| 16 | MPI/NCCL Backend | Межузловое распределённое обучение |
| 17 | TorchScript/JIT | Компиляция графа вычислений |
| 18 | ONNX Export | Экспорт моделей |
| 19 | INT8 Quantization | Post-training quantization |
| 20 | Sparse Tensors | COO/CSR форматы |
| 21 | Profiling | Timeline profiler |
| 22 | RISC-V Backend | Syntacore / RISC-V Chinese chips |
| 23 | Эльбрус GPU | Mali-подобный GPU на Эльбрус-2С3 |
