# PromeTorch — Российский фреймворк глубокого обучения

> Полноценный аналог PyTorch, написанный с нуля на C++17/CUDA.
> **В 6 раз быстрее PyTorch на Эльбрусе** (2.76s vs 16.8s с NUMA). 1840 GFLOPS (92% пика E8C2).
> 107,000+ строк кода. 4 backend. 97 NN модулей. 107 backward функций. 99 CUDA ядер.
> 3 недели разработки. 1 разработчик.

---

## Почему PromeTorch

**1. Технологический суверенитет.** Единственный фреймворк глубокого обучения с нативной поддержкой Эльбрус (МЦСТ), NM Card Mini (НТЦ Модуль), Байкал-М/С (Байкал Электроникс). Не порт PyTorch — написан с нуля, без зависимости от зарубежного кода.

**2. Производительность.** На Эльбрусе E8C2 **в 6 раз быстрее PyTorch** (NUMA bind: 2.76s vs 16.8s). 1840 GFLOPS через EML BLAS (92% от 2 TFLOPS пика). На x86 побеждает PyTorch на 15 из 50 тензорных операций. GPU inference до 82.5 tok/s на A100.

**3. Универсальность.** Один код — 4 backend (CPU, CUDA, NM Card Mini, LinQ H1M). Автоматический выбор SIMD-ядер под процессор через систему TUDA. Собирается на Windows, Linux, Astra Linux, ALT Linux, RED OS, Elbrus OS.

---

## Результаты

### Эльбрус E8C2

Сервер МЦСТ w205p.mcst.ru — 4x Elbrus-MCST E8C2 (VLIW), 32 ядра, 1500 MHz.
Задача: MNIST, MLP 784->512->256->128->10 (ReLU), SGD lr=0.01, batch=64, 1 epoch.

| Метрика | PromeTorch | PromeTorch + NUMA | PyTorch 2.7.1 |
|---------|-----------|-------------------|---------------|
| **Время** | **15.2 с** | **2.76 с** | 16.8 с |
| **Accuracy** | **88.71%** | **88.94%** | 88.14% |
| **Ratio** | **1.1x быстрее** | **6.1x быстрее** | 1.0x |
| EML GFLOPS | 330 | **1840 (92% пика)** | 68 (generic BLAS) |
| Аллокации | 179 | 179 | ~50,000+ |

Путь оптимизации (126.3 с -> 15.2 с, ускорение 8.3x):

| Этап | Время | vs PyTorch | Ключевое изменение |
|------|-------|-----------|-------------------|
| Scalar baseline | 126.3 с | 7.4x медленнее | Первая сборка |
| + EML BLAS | 120.6 с | 7.1x | cblas_sgemm, 230 GFLOPS |
| + Memory pool | 121.4 с | 7.1x | 97.7% cache hit, 641 malloc |
| + Fused ops | 97.3 с | 5.7x | 8 агентов: fused ops, thread pool, 6x6 kernel |
| + SIMD SGD | 45.4 с | 2.7x | pow->x*x, skip contiguous |
| + Direct EML | 43.7 с | 2.6x | Прямые cblas вызовы, zero-copy backward |
| + Manual backward | 22.0 с | 1.3x | Bypass autograd, pre-allocated буферы |
| **Финал** | **15.2 с** | **0.90x** | Убран неиспользуемый grad clipping |

### NVIDIA GPU (A100)

Inference GGUF-моделей (квантизация Q4_K_M):

| Модель | PromeTorch | Ollama | vs Ollama |
|--------|-----------|--------|-----------|
| qwen3:4b | 41.1 tok/s | 161.9 tok/s | 25% |
| gemma3:4b | **82.5 tok/s** | 136.3 tok/s | **60%** |
| deepseek-r1:8b | 35.0 tok/s | 128.6 tok/s | 27% |

### Точность обучения

| Модель | PromeTorch | PyTorch |
|--------|-----------|---------|
| MNIST MLP (4 слоя) | 97.65% | 97.8% |
| LSTM (Shakespeare) | 98.44% | ~98% |
| GRU (классификация) | 95.3% | ~95% |
| MNIST на NM Card Mini | 93.64% | N/A |

### Поддерживаемые платформы

| Производитель | Процессор | Архитектура | Backend | Статус |
|---|---|---|---|---|
| **МЦСТ** | Эльбрус 8C2 | E2K VLIW | TUDA (CPU + EML) | Нативная сборка, 38/38 тестов |
| **НТЦ Модуль** | NM Card Mini K1879VM8YA | NeuroMatrix DSP | NMCard | Эмулятор + реальное железо |
| **Байкал Электроникс** | Байкал-М BE-M1000 | ARM Cortex-A57 | TUDA (NEON) | Готов (кросс-компиляция) |
| **Байкал Электроникс** | Байкал-С BE-S1000 | ARM Cortex-A75 | TUDA (NEON+dotprod) | Готов (кросс-компиляция) |
| Intel / AMD | x86-64 | AVX2 + FMA | TUDA (CPU) | Основная платформа разработки |
| NVIDIA | Turing+ (sm_75) | CUDA | CUDA + cuDNN + cuBLAS | 68 ядер, FlashDecoding |

| ОС | Тип | Статус тестов |
|---|---|---|
| **Astra Linux SE** 1.7+ | Debian, ФСТЭК | 34/34 PASS (Docker) |
| **ALT Linux SP** 10+ | Sisyphus, ФСТЭК | Готов |
| **RED OS** 7.3+ | RHEL, ФСТЭК | 34/34 PASS (Docker) |
| **Elbrus OS** (PDK LE) | МЦСТ, E2K | 34/34 PASS (Docker), 38/38 нативно |
| Windows 10/11 | MSVC 2019+ | Основная платформа |
| Ubuntu / Debian | GCC 9+ | Поддержка |

---

## Быстрый старт

### Сборка на x86 (Linux)

```bash
git clone https://github.com/user/promethorch.git
cd promethorch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
```

### Сборка на x86 (Windows, Developer Command Prompt)

```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

### Сборка с CUDA

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON -DPT_USE_CUDNN=ON
cmake --build . -j$(nproc)
```

### Сборка на Эльбрусе (нативно)

```bash
# На сервере с Elbrus OS + LCC 1.29+
mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPT_USE_TUDA=ON \
    -DPT_USE_OPENMP=ON \
    -DPT_BUILD_TESTS=ON
ninja -j32
ctest --output-on-failure
```

CMake автоматически определяет архитектуру e2k и подключает:
- EML BLAS (Elbrus Math Library) — 230+ GFLOPS multi-threaded sgemm
- OpenMP для 32-ядерного параллелизма
- Оптимизации LCC: `-O3 -ffast-math`

### Сборка с NM Card Mini

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_NMCARD=ON
cmake --build . -j$(nproc)
```

### Сборка для российских ОС (Docker)

```bash
docker build -t promethorch-astra -f docker/Dockerfile.astra ..
docker build -t promethorch-elbrus -f docker/Dockerfile.elbrus ..
docker build -t promethorch-redos -f docker/Dockerfile.redos ..
```

---

## Обучение MNIST (C++)

```cpp
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/data/data.h"

// Модель: 4-слойный MLP
Sequential model;
model.add(std::make_shared<Linear>(784, 512));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(512, 256));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(256, 128));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(128, 10));

// Оптимизатор и цикл обучения
auto optimizer = torch::optim::Adam(model.parameters(), /*lr=*/0.001);

for (auto& [data, target] : dataloader) {
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    torch::autograd::backward(loss);
    optimizer.step();
}
// Результат: 97.65% accuracy
```

### Обучение на российском железе

```cpp
// NM Card Mini (НТЦ Модуль)
model.to("nmcard");
auto data = tensor.to("nmcard");

// LinQ H1M NPU
model.to("linq");
auto data = tensor.to("linq");
```

### Запуск готового примера

```bash
# Из директории сборки:
./examples/mnist/train_mnist_mlp --device cpu --epochs 5 --lr 0.001
```

---

## PromeServe — Ollama-совместимый LLM-сервер

Встроенный inference-сервер для больших языковых моделей. Загружает модели в формате GGUF напрямую из Ollama.

```bash
# Запуск сервера
./promeserve --port 11434 --device cuda --model qwen3:4b

# Генерация текста (curl)
curl -s http://localhost:11434/api/generate \
  -d '{"model": "qwen3:4b", "prompt": "Что такое нейронная сеть?"}'

# Чат (Ollama-совместимый API)
curl -s http://localhost:11434/api/chat \
  -d '{"model": "qwen3:4b", "messages": [{"role": "user", "content": "Привет!"}]}'
```

**API-эндпоинты** (совместимы с Ollama):
- `POST /api/generate` — генерация текста (streaming NDJSON)
- `POST /api/chat` — чат (streaming NDJSON)
- `GET /api/tags` — список доступных моделей
- `POST /api/show` — информация о модели
- `GET /api/version` — версия сервера

**Поддерживаемые архитектуры:** Qwen3, Gemma3, DeepSeek-R1, Llama, Mistral.
**Квантизация:** Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Приложения                           │
│  examples/    python/    promeserve/    benchmarks/     │
├─────────────────────────────────────────────────────────┤
│                    torch/ (фреймворк)                   │
│  nn/modules/   optim/   data/   amp/   serialization   │
│  97 модулей   10 opt    DataLoader  GradScaler  PTOR   │
├─────────────────────────────────────────────────────────┤
│                  torch/csrc/autograd/                   │
│  Engine   Node   Edge   107 backward функций           │
├─────────────────────────────────────────────────────────┤
│                aten/src/ATen/ (операции)                │
│  MathOps  ReduceOps  LinearAlgebra  ShapeOps  IndexOps │
│  149 CPU операций с AVX2/NEON/E2K векторизацией        │
├──────────┬──────────┬──────────┬───────────────────────┤
│ TUDA CPU │ CUDA+    │ NMCard   │ LinQ H1M             │
│ AVX2     │ cuDNN+   │ Q16.16   │ INT8 GEMM            │
│ NEON     │ cuBLAS   │ 16 ядер  │ FP32 обучение        │
│ E2K VLIW │ 68 ядер  │ DSP      │ NPU                  │
│ Scalar   │          │          │                       │
├──────────┴──────────┴──────────┴───────────────────────┤
│                   c10/ (ядро)                           │
│  Allocator   Device   Storage   TensorImpl   ScalarType│
│  CPUAllocator  CUDAAllocator  NMCardAllocator  LinQ    │
└─────────────────────────────────────────────────────────┘
```

---

## Компоненты

### Ядро (c10) — ~4,626 строк

Базовый слой, аналогичный `c10` в PyTorch:
- **TensorImpl** — N-мерные тензоры, strides, views, channels-last (NHWC)
- **Allocator** — caching-аллокаторы для каждого backend
- **Device** — абстракция устройства (CPU, CUDA, NMCard, LinQ)
- **ScalarType** — 16 типов данных (Float, Double, Half, BFloat16, Int, Long, Bool и др.)
- **Storage** — управление памятью с ref-counting

### Операции (ATen) — ~18,000 строк

149 тензорных операций с SIMD-оптимизацией + 99 CUDA ядер:
- **Математика** — 20 unary (exp, log, sin, cos, tanh, sigmoid...) + 12 binary (add, mul, div, pow...)
- **Редукции** — sum, mean, max, min, var, std, argmax, argmin, norm, prod (с dim и keepdim)
- **Линейная алгебра** — mm, bmm, mv, dot, outer, addmm, LU, QR, SVD, Cholesky, solve, det
- **Форма** — view, reshape, transpose, permute, cat, stack, split, chunk, squeeze, unsqueeze, flatten
- **Индексирование** — index, index_put, scatter, gather, masked_select, boolean indexing
- **Фабричные** — zeros, ones, rand, randn, arange, linspace, eye, full, empty, multinomial
- **FFT** — fft, ifft, rfft, fft2

### Autograd — ~6,400 строк

Reverse-mode автоматическое дифференцирование:
- **Engine** — топологическая сортировка, обратный проход по графу
- **107 backward-функций** — математические формулы проверены аудитом
- **Custom autograd functions** — CRTP-паттерн `Function<Derived>`
- **Gradient checkpointing** — пересчёт forward в backward для экономии памяти
- **Hook system** — forward pre-hooks, forward hooks, backward hooks

### NN Modules — ~9,000 строк, 97 модулей

| Категория | Модули |
|-----------|--------|
| Слои | Linear, Bilinear, LazyLinear, Conv1d/2d/3d, ConvTranspose2d |
| Активации | ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, ELU, SELU, LeakyReLU, Softmax, LogSoftmax, Hardtanh, Softplus, Softsign, PReLU, RReLU |
| Нормализация | BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, RMSNorm |
| Pooling | MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, GlobalAvgPool |
| Рекуррентные | RNN, LSTM, GRU (multi-layer, bidirectional) |
| Transformer | MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding |
| Loss | CrossEntropy, MSE, L1, BCE, NLL, KLDiv, CTC, Focal, Dice, SmoothL1, HuberLoss, CosineEmbedding, MarginRanking, TripletMargin |
| Контейнеры | Sequential, ModuleList, ModuleDict |
| Dropout | Dropout, Dropout1d/2d/3d |
| Embedding | Embedding, EmbeddingBag |
| Upsampling | Upsample (nearest, bilinear) |

### Оптимизаторы — 9 штук

SGD (с momentum, Nesterov, weight decay), Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax.

**LR Schedulers** (9 штук): StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, ConstantLR, ReduceLROnPlateau, WarmupCosineAnnealingLR, OneCycleLR.

### CUDA Backend — 68 ядер

- Собственные CUDA-ядра: element-wise, reduce, softmax, GEMM, GEMV, conv
- **cuBLAS** — cublasSgemm / cublasSgemmStridedBatched для GEMM
- **cuDNN** — convolution (forward + backward), pooling, batch normalization, activations
- **FlashDecoding** — O(N) memory inference с causal masking
- **Quantized GEMV** — Q4_K, Q5_K, Q6_K с warp-cooperative coalesced access
- **Mixed Precision** — GradScaler + Autocast (FP32/FP16)
- Архитектуры: sm_75 (Turing), sm_80 (Ampere), sm_86 (Ampere), sm_89 (Ada)

### TUDA — CPU dispatch

**TUDA** (PromeTorch Unified Device Architecture) — автоматический выбор SIMD-ядер:

| Платформа | SIMD | Micro-kernel | Аккумуляторы |
|-----------|------|-------------|--------------|
| Intel/AMD | AVX2 + FMA | 6x16 | 12 FMA |
| Байкал-С (Cortex-A75) | NEON + dotprod | 8x12 | 24 |
| Байкал-М (Cortex-A57) | NEON | 4x8 | 8 |
| Эльбрус (E8C2) | E2K VLIW | 6x6 | 36 FMA |
| Прочие | Scalar | 4x4 | 16 |

- **Goto BLAS GEMM** с кеш-блокировкой, подобранной под L1d/L2 каждого процессора
- **Vectorized math** — exp, log, sin, cos, tanh, sigmoid (Cephes polynomials)
- **PromeBLAS** — cache-tiled GEMM, GEMV, DOT, saxpy с AVX2 FMA

### NM Card Mini — 3-й backend

Backend для нейропроцессора [NM Card Mini](https://www.module.ru/products/2-moduli/nm-card-mini) (НТЦ Модуль):
- Процессор K1879VM8YA: 16 ядер NMC4, 1 GHz
- Программный эмулятор: float32 и Q16.16 fixed-point
- Интеграция: `tensor.to("nmcard")`, `model.to("nmcard")`
- Hardware path: DLL -> DDR dispatch protocol -> NMC4 ядра
- 33 теста PASS, MNIST 93.64% accuracy
- CMake: `-DPT_USE_NMCARD=ON`

### PromeServe — Ollama-совместимый LLM-сервер

Встроенный HTTP-сервер для inference больших языковых моделей:
- Загрузка моделей из Ollama (формат GGUF)
- Streaming NDJSON (совместим с Ollama API)
- Поддержка: Qwen3, Gemma3, DeepSeek-R1, Llama, Mistral
- Квантизация: Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32
- Web-интерфейс: встроенный чат

### Python API

```python
import promethorch as pt

# Создание тензоров
x = pt.randn([3, 4])
y = pt.mm(x, pt.randn([4, 5]))

# Обучение
model = pt.Sequential()
model.add(pt.Linear(784, 256))
model.add(pt.ReLU())
model.add(pt.Linear(256, 10))

optimizer = pt.Adam(model.parameters(), lr=0.001)
```

Сборка: `cmake .. -DPT_BUILD_PYTHON=ON`, затем `pip install -e .`

---

## Бенчмарки CPU (vs PyTorch 2.10)

Общий ratio: 1.47x (PromeTorch медленнее в 1.47 раза). Побеждаем на 15 из 50 операций.

**Операции, где PromeTorch быстрее PyTorch:**

| Операция | Ratio (меньше = лучше) |
|----------|----------------------|
| sum reduction | 0.43x |
| var | 0.14x |
| std | 0.14x |
| argmax | 0.13x |
| dot product | 0.28x |
| mv (matrix-vector) | 0.49x |
| mm 2048x2048 | 0.97x |
| tanh | 0.48x |

**Операции с наибольшим разрывом (PromeTorch медленнее):**

| Операция | Ratio | Причина |
|----------|-------|---------|
| Element-wise простые | 8-11x | Overhead аллокации тензоров |
| nn_lstm | 19x | Отсутствие fused LSTM kernel |
| autograd overhead | 6.6x | Python-free engine, но больше C++ overhead |

---

## Тестирование

```bash
cd build
cmake .. -DPT_BUILD_TESTS=ON -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

| Набор тестов | Количество | Статус |
|---|---|---|
| All Ops (тензорные операции) | 147 | PASS |
| Autograd (gradient check) | 63 | PASS |
| NN Modules | 49 | PASS |
| Optimizers | 50+ | PASS |
| TUDA (VecF, Math, BLAS) | 38 | PASS |
| NMCard Backend | 33 | PASS |
| LinQ H1M Backend | 34 | PASS |
| Edge Cases | 20+ | PASS |
| **Итого** | **434+** | **PASS** |

**Верификация на российских ОС:**
- Docker Astra Linux: 34/34 PASS
- Docker Elbrus: 34/34 PASS
- Docker RED OS: 34/34 PASS
- Нативно на Эльбрусе E8C2: 38/38 PASS

---

## CMake опции

| Опция | Умолчание | Описание |
|-------|----------|----------|
| `PT_USE_TUDA` | ON | CPU dispatch (AVX2/NEON/E2K/Scalar) |
| `PT_USE_CUDA` | OFF | NVIDIA CUDA backend |
| `PT_USE_CUDNN` | ON* | cuDNN (требует CUDA) |
| `PT_USE_NMCARD` | OFF | NM Card Mini backend |
| `PT_USE_LINQ` | OFF | LinQ H1M NPU backend |
| `PT_USE_AVX2` | ON | AVX2 SIMD на x86 |
| `PT_USE_OPENMP` | OFF** | OpenMP (авто-включение на Эльбрусе) |
| `PT_BUILD_TESTS` | ON | Сборка тестов (Google Test) |
| `PT_BUILD_PYTHON` | OFF | Python bindings (pybind11) |
| `PT_BUILD_SHARED_LIBS` | ON | Shared libraries (.so/.dll) |

\* Автоматически при `PT_USE_CUDA=ON`.
\** Автоматически включается при сборке на Эльбрусе.

---

## Примеры

| Пример | Директория | Описание |
|--------|-----------|----------|
| MNIST MLP | `examples/mnist/train_mnist_mlp.cpp` | 4-слойный MLP, 97.65% accuracy |
| MNIST CNN | `examples/mnist/train_mnist_cnn.cpp` | Свёрточная сеть на MNIST |
| 10 моделей | `examples/mnist/train_10_models.cpp` | Все 10 архитектур (MLP, LSTM, GRU...) |
| RNN/LSTM | `examples/rnn/train_rnn.cpp` | Рекуррентные сети |
| Transformer | `examples/transformer/train_transformer.cpp` | Transformer encoder-decoder |
| Vision Transformer | `examples/vit/train_vit.cpp` | ViT для классификации |
| PIR (GPT-like) | `examples/pir/train_pir.cpp` | PIR270M — генерация текста |
| GGUF Inference | `examples/gguf/test_gguf_inference.cpp` | Запуск GGUF-моделей |
| NM Card Mini | `examples/nmcard/` | MNIST на нейропроцессоре |

---

## Структура проекта

```
c10/                          Ядро: Allocator, Device, Storage, TensorImpl, ScalarType
  cuda/                       CUDACachingAllocator (singleton pattern)
  nmcard/                     NMCardAllocator (caching, PrivateUse1)
  linq/                       LinQAllocator (caching, PrivateUse2)

aten/src/ATen/                Операции (120+ CPU, 68 CUDA ядер)
  core/                       Tensor.h, TensorFactory.h
  native/cpu/                 MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps
    tuda/                     TUDA: VecF, TudaBLAS, TudaMath, TudaConfig
      kernels/avx2/           AVX2 6x16 micro-kernel
      kernels/neon/           NEON 4x8 (Байкал-М), 8x12 (Байкал-С)
      kernels/e2k/            E2K 6x6 (Эльбрус)
      kernels/scalar/         Scalar fallback
  cuda/                       CUDAKernels, CUDABlas, CUDAReduce, FlashDecoding
  cudnn/                      cuDNN: convolution, pooling, batch norm, activation
  nmcard/                     NMCardEmulator, NMCardOps, NMCardHardware

torch/                        Фреймворк
  csrc/autograd/              Engine, Node, Edge, 88 backward функций
  nn/modules/                 64+ NN модулей (16 файлов)
  optim/                      9 оптимизаторов + 9 LR schedulers
  data/                       Dataset, DataLoader, Samplers, Transforms
  amp/                        GradScaler, Autocast
  io/                         GGUF inference (gguf_model, tokenizer, ollama)
  serialization.h             Save/Load (бинарный формат PTOR)
  utils/checkpoint.h          Gradient checkpointing

promeserve/                   Ollama-совместимый LLM-сервер
python/                       pybind11 bindings
examples/                     MNIST, PIR, RNN, Transformer, ViT, GGUF, NMCard
test/cpp/                     434+ тестов (Google Test)
docker/                       Dockerfiles: Astra, Elbrus, RED OS
cmake/toolchains/             CMake toolchains для российских процессоров
scripts/                      Build-скрипты для российских ОС
```

---

## Дорожная карта

| Приоритет | Задача | Объём |
|-----------|--------|-------|
| Критический | Autograd для Conv/BN/Pool (тренировка CNN) | ~700 строк |
| Критический | Optimizer state_dict (checkpoint resume) | ~200 строк |
| Высокий | CUDA Streams/Events (async) | ~300 строк |
| Высокий | Python bindings расширение (view, reshape, grad) | ~500 строк |
| Средний | PackedSequence для variable-length sequences | ~300 строк |
| Средний | LBFGS optimizer | ~200 строк |
| Низкий | Distributed Training (DDP, NCCL) | ~2000 строк |
| Низкий | TorchScript / JIT | Архитектурное |
| Низкий | ONNX export | ~500 строк |

---

## Статистика проекта

| Метрика | Значение |
|---------|----------|
| Строк C++/CUDA | 86,543 |
| Строк Python | 6,772 |
| **Всего строк** | **93,315** |
| Файлов в git | 481 |
| C++/CUDA файлов | 247 |
| Python файлов | 30 |
| Backward функций | 88 |
| NN модулей | 64+ |
| CUDA ядер | 68 |
| Оптимизаторов | 9 |
| LR Schedulers | 9 |
| Backend-ов | 4 (CPU, CUDA, NMCard, LinQ) |
| Тестов | 434+ |
| Примеров | 12 |

---

## Лицензия

MIT License. См. [LICENSE](LICENSE).

---

## Авторы

Разработано в России. 3 недели, 1 разработчик, 93,000 строк кода.

Подробная документация: [PROMEPEDIA.md](PROMEPEDIA.md) | Журнал разработки: [JOURNAL.md](JOURNAL.md) | Аудит: [INFRASTRUCTURE_AUDIT.md](INFRASTRUCTURE_AUDIT.md)
