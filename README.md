# PromeTorch

**Первый в мире фреймворк глубокого обучения с нативной поддержкой российского железа.**

Полноценный аналог PyTorch, написанный с нуля на C++17/CUDA. Адаптируется под железо — не железо под него.

**~53,000+ строк C++/CUDA** | **110+ тензорных операций** | **90+ autograd backward** | **57+ NN-модулей** | **5 backends** | **6 российских платформ**

---

## Поддерживаемое железо

| Производитель | Процессор | Архитектура | Backend | Статус |
|---|---|---|---|---|
| **МЦСТ** | Эльбрус 8C/16C/8SV | E2K VLIW | TUDA (CPU) | Готов |
| **Байкал Электроникс** | Байкал-М BE-M1000 | ARM Cortex-A57 | TUDA (CPU) | Готов |
| **Байкал Электроникс** | Байкал-С BE-S1000 | ARM Cortex-A75 | TUDA (CPU) | Готов |
| **partner** | NM Card Mini K1879VM8YA | NeuroMatrix DSP | NMCard | Готов |
| **LinQ** | H1M NPU | INT8/FP32 | LinQ | Готов |
| Intel/AMD | x86-64 | AVX2+FMA | TUDA (CPU) | Готов |
| NVIDIA | Any (sm_50+) | CUDA | CUDA+cuDNN | Готов |

## Поддерживаемые ОС

| ОС | Тип | Toolchain | Сборка |
|---|---|---|---|
| **Astra Linux SE** 1.7+ | Debian-based, ФСТЭК | `x86_64-astra.cmake` | `scripts/build-astra.sh` |
| **ALT Linux SP** 10+ | Sisyphus, ФСТЭК | `x86_64-alt.cmake` | `scripts/build-alt.sh` |
| **РЕД ОС** 7.3+ | RHEL-based, ФСТЭК | `x86_64-redos.cmake` | `scripts/build-redos.sh` |
| **Elbrus OS** (PDK LE) | МЦСТ, E2K | `e2k-elbrus.cmake` | `scripts/build-elbrus.sh` |
| Windows 10/11 | MSVC 2019+ | — | Developer Command Prompt |
| Ubuntu/Debian | GCC 9+ | — | `cmake && make` |

---

## Быстрый старт

### Установка (Python)

```bash
pip install -e .
# С CUDA:
PT_USE_CUDA=1 pip install -e .
# С LinQ эмулятором:
PT_USE_LINQ=1 pip install -e .
```

### Установка (C++)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
```

### Сборка под российское железо

```bash
# Эльбрус (нативно на Elbrus OS)
./scripts/build-elbrus.sh

# Байкал-М (кросс-компиляция с x86)
./scripts/build-baikal-cross.sh baikal-m

# Байкал-С
./scripts/build-baikal-cross.sh baikal-s

# Astra Linux SE
./scripts/build-astra.sh

# ALT Linux
./scripts/build-alt.sh

# РЕД ОС
./scripts/build-redos.sh
```

### Docker

```bash
# Собрать для всех российских ОС
cd docker && ./build-all.sh

# Или по отдельности
docker build -t promethorch-astra -f docker/Dockerfile.astra ..
docker build -t promethorch-elbrus -f docker/Dockerfile.elbrus ..
```

---

## Возможности

### Тензорная библиотека
- N-мерные тензоры с broadcasting, strides, views, memory formats
- 110+ операций: математика, редукции, линейная алгебра (LU, QR, SVD, Cholesky), shape, advanced indexing
- Factory functions: `zeros`, `ones`, `rand`, `randn`, `arange`, `linspace`, `eye`, `multinomial`
- Типы данных: Float, Double, Half, Int, Long, Bool, BFloat16
- Channels-last memory format (NHWC)

### Автоматическое дифференцирование
- Reverse-mode autograd с динамическим вычислительным графом
- 90+ backward-функций для всех дифференцируемых операций
- Custom autograd functions (CRTP pattern)
- Gradient checkpointing для memory-efficient training
- Hook system (forward pre-hooks, forward hooks, backward hooks)

### Нейросетевые модули (57+)
- **Слои**: Linear, Bilinear, LazyLinear, Conv1d/2d/3d, ConvTranspose2d
- **Активации**: ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, ELU, SELU, Softmax, 10+ ещё
- **Нормализация**: BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, RMSNorm
- **Pooling**: MaxPool, AvgPool, AdaptiveAvgPool, GlobalAvgPool
- **Рекуррентные**: RNN, LSTM, GRU (multi-layer, bidirectional)
- **Transformer**: MultiheadAttention, TransformerEncoder/Decoder, PositionalEncoding
- **Loss**: CrossEntropy, MSE, L1, BCE, NLL, KLDiv, CTC, Focal, Dice, 10+ ещё
- **Контейнеры**: Sequential, ModuleList, ModuleDict
- **Embedding**: Embedding, EmbeddingBag

### Оптимизаторы и планировщики
- **Оптимизаторы**: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax
- **LR Schedulers**: StepLR, CosineAnnealing, OneCycleLR, WarmupLR, 9+ ещё

### TUDA — кросс-платформенный CPU dispatch
**TUDA** (PromeTorch Unified Device Architecture) — система автоматического выбора оптимальных SIMD-ядер под целевой процессор.

- **VecF** — единый SIMD-тип: Vec8 (AVX2), Vec4 (NEON/E2K), Vec1 (Scalar)
- **Goto BLAS GEMM** с architecture-specific micro-kernels:
  - AVX2 6×16 (Intel/AMD) — 12 FMA accumulators
  - NEON 8×12 (Байкал-С, Cortex-A75) — 24 accumulators, 64KB L1d
  - NEON 4×8 (Байкал-М, Cortex-A57) — 8 accumulators, 32KB L1d
  - E2K 4×4 (Эльбрус) — 16 scalar FMA, LCC software pipelining
  - Scalar fallback — любая платформа
- **Vectorized math**: exp, log, sin, cos, tanh, sigmoid, relu, silu, gelu (Cephes polynomials для AVX2/NEON)
- **Cache tuning** под каждый процессор: MC, KC, NC подобраны под L1d/L2 размеры

### CUDA Backend
- Собственные CUDA kernels: GEMM, GEMV, reductions, element-wise, softmax
- cuDNN интеграция: convolution, pooling, batch normalization, activations
- FlashAttention (O(N) memory, causal masking)
- Mixed precision (AMP): GradScaler + Autocast
- Quantized GEMV (Q4_K, Q5_K, Q6_K) с warp-cooperative coalesced access

### NM Card Mini Backend (partner)
- Третий backend (после CPU и CUDA) для [NM Card Mini](https://www.module.ru/products/2-moduli/nm-card-mini)
- Процессор K1879VM8YA: 16 ядер NMC4 tensor @ 1 GHz
- Программный эмулятор: float32 и Q16.16 fixed-point (как реальное железо)
- Интеграция: `tensor.to("nmcard")`, `model.to("nmcard")`
- Hardware path: DLL loading → DDR dispatch protocol → NMC4 ядра
- 32 теста, MNIST 93.64% accuracy на `--device nmcard`
- `-DPT_USE_NMCARD=ON`

### LinQ H1M Backend
- Четвёртый backend для NPU-ускорителя LinQ H1M
- INT8 GEMM с INT32 аккумуляцией (96 TOPS peak на железе)
- FP32 GEMM для обучения
- 50+ операций: unary, binary, comparison, reduction, normalization, quantization
- Интеграция: `tensor.to("linq")`, `model.to("linq")`
- 34 теста, все проходят
- `-DPT_USE_LINQ=ON`

### Распределённое обучение
- AllReduce: SUM, AVG, MAX, MIN
- Broadcast, Scatter
- DataParallel — автоматическое разделение batch между виртуальными устройствами
- Shared-memory backend (multi-thread) — работает с любыми device'ами
- 7 тестов, все проходят

### GGUF Inference Engine
- Загрузка и запуск моделей Ollama/llama.cpp напрямую (формат GGUF)
- Архитектуры: Qwen3, Gemma3, DeepSeek-R1, Llama
- Chat template + special tokens
- GPU: ~55 tok/s на A100 (qwen3:4b)
- Форматы квантизации: Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32

### Дополнительно
- Сериализация (save/load тензоров и state_dicts)
- Python bindings через pybind11
- FFT операции (fft, ifft, rfft, fft2)
- Einsum с оптимизированными путями
- Weight norm, Spectral norm

---

## Примеры

### MNIST Training (C++)

```cpp
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"

Sequential model;
model.add(std::make_shared<Linear>(784, 256));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(256, 10));

auto optimizer = torch::optim::Adam(model.parameters(), /*lr=*/0.001);
for (auto& [data, target] : dataloader) {
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    torch::autograd::backward(loss);
    optimizer.step();
}
```

### Обучение на российском железе

```cpp
// На NM Card Mini
model.to("nmcard");
auto data = tensor.to("nmcard");

// На LinQ H1M NPU
model.to("linq");
auto data = tensor.to("linq");
```

### Распределённое обучение

```cpp
#include "torch/distributed/distributed.h"
using namespace torch::distributed;

dist::init(4); // 4 virtual devices
std::vector<std::thread> workers;
for (int rank = 0; rank < 4; ++rank) {
    workers.emplace_back([&, rank]() {
        auto shard = dist::scatter(batch, rank);
        auto loss = train_step(model, shard);
        for (auto& p : model.parameters())
            dist::all_reduce(p.grad(), rank, ReduceOp::AVG);
        optimizer.step();
    });
}
for (auto& w : workers) w.join();
```

### GGUF Inference

```cpp
#include "torch/io/gguf_model.h"

GGUFModel model;
model.load_from_ollama("qwen3:4b", /*use_cuda=*/true);
std::string response = model.chat("What is 2+2?", /*max_tokens=*/64);
```

---

## Бенчмарки

### Точность обучения

| Модель | PromeTorch | PyTorch |
|--------|-----------|---------|
| MNIST MLP (4-layer) | 97.65% | 97.8% |
| LSTM Shakespeare | 98.44% | ~98% |
| GRU Classification | 95.3% | ~95% |
| MNIST on NMCard | 93.64% | N/A |

### Inference

| Модель | PromeTorch (A100) | Ollama (A100) |
|--------|------------------|---------------|
| qwen3:4b | 55 tok/s | 165 tok/s |
| deepseek-r1:8b | 30 tok/s | ~84 tok/s |
| qwen3:14b | 18 tok/s | N/A |

### CPU SIMD (vs PyTorch)

| Операция | Ratio (ниже — лучше) |
|----------|------|
| sum reduction | 0.43x |
| var/std | 0.14x |
| argmax | 0.13x |
| dot product | 0.28x |
| mm 2048x2048 | 0.97x |
| tanh | 0.48x |

---

## Тесты

```bash
cd build
cmake .. -DPT_BUILD_TESTS=ON -DPT_USE_TUDA=ON -DPT_USE_LINQ=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

| Набор тестов | Тестов | Статус |
|---|---|---|
| All Ops (tensor operations) | 147 | PASS |
| Autograd (gradient check) | 63 | PASS |
| NN Modules | 49 | PASS |
| Optimizers | 50+ | PASS |
| TUDA (VecF, Math, BLAS) | 38 | PASS |
| LinQ H1M Backend | 34 | PASS |
| NMCard Backend | 32 | PASS |
| Distributed (AllReduce) | 7 | PASS |
| Edge Cases | 20+ | PASS |
| **Итого** | **440+** | **PASS** |

---

## Архитектура

```
c10/                           Ядро: Allocator, Device, Storage, TensorImpl, ScalarType
  nmcard/                      NMCardAllocator (caching, PrivateUse1)
  linq/                        LinQAllocator (caching, PrivateUse2)

aten/src/ATen/
  core/                        Tensor, TensorFactory, TensorOptions
  native/cpu/                  MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps
    tuda/                      TUDA: VecF, TudaBLAS, TudaMath, TudaConfig
      kernels/avx2/            AVX2 6×16 micro-kernel
      kernels/neon/            NEON 4×8 (Бaйкал-М), 8×12 (Байкал-С)
      kernels/e2k/             E2K 4×4 (Эльбрус)
      kernels/scalar/          Scalar fallback
  cuda/                        CUDAKernels, CUDABlas, CUDAReduce, FlashAttention
  cudnn/                       cuDNN wrappers
  nmcard/                      NMCardEmulator, NMCardOps, NMCardDispatch, NMCardHardware
  linq/                        LinQEmulator, LinQOps, LinQDispatch

torch/
  csrc/autograd/               Engine, Node, Edge, 90+ backward functions
  nn/modules/                  57+ NN module implementations
  optim/                       9 оптимизаторов + 13 LR schedulers
  distributed/                 AllReduce, Broadcast, Scatter, DataParallel
  io/                          GGUF model loading and inference
  amp/                         Mixed precision (GradScaler, Autocast)
  data/                        Dataset, DataLoader, Samplers
  serialization.h              Save/Load (binary PTOR format)
  utils/checkpoint.h           Gradient checkpointing

cmake/toolchains/              CMake toolchains для российских процессоров
scripts/                       Build-скрипты для российских ОС
docker/                        Dockerfiles для Astra, ALT, РЕД ОС, Elbrus, Baikal
python/                        pybind11 bindings
promethorch/                   Python package (pip install)
test/cpp/                      Test suite (16+ test files, 440+ tests)
examples/                      MNIST, PIR, RNN, Transformer, ViT, GGUF, NMCard
```

---

## CMake опции

| Опция | По умолчанию | Описание |
|-------|-------------|----------|
| `PT_USE_TUDA` | ON | Кросс-платформенный CPU dispatch (AVX2/NEON/E2K) |
| `PT_USE_CUDA` | OFF | NVIDIA CUDA backend |
| `PT_USE_CUDNN` | OFF | cuDNN ускорение |
| `PT_USE_NMCARD` | OFF | NM Card Mini backend |
| `PT_USE_LINQ` | OFF | LinQ H1M NPU backend |
| `PT_USE_AVX2` | ON | AVX2 SIMD (x86) |
| `PT_USE_OPENMP` | OFF | OpenMP параллелизм |
| `PT_BUILD_TESTS` | ON | Сборка тестов |
| `PT_BUILD_PYTHON` | OFF | Python bindings |
| `PT_BUILD_SHARED_LIBS` | OFF | Shared libraries (.so/.dll) |

---

## Лицензия

MIT License. См. [LICENSE](LICENSE).

---

**PromeTorch** — полная документация: [PROMEPEDIA.md](PROMEPEDIA.md)
