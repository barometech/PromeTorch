# PromeTorch — Российский training framework

> PyTorch-совместимый обучающий фреймворк на C++17/CUDA с фокусом на float32.
> Нативная поддержка **Эльбрус E8C2 (VLIW)** и **NM Card Mini (НТЦ Модуль, Q16.16)**.
> Настоящий autograd (110 backward функций), 835 gtest, CPU SIMD + CUDA через cuBLAS/cuDNN.
> 91,000+ строк C++. 1 разработчик. ~1 месяц активной разработки.

> ⚠️ **Не полноценная замена PyTorch.** Это solo-проект: dtype coverage ограничен float32
> (FP16/BF16 не обучает), ChannelsLast — только метаданные, FlashAttention имеет баги (см.
> `INFRASTRUCTURE_AUDIT.md`), некоторые ops (Conv3d, CTCLoss) — stubs. См. раздел
> **Known Limitations** ниже.

---

## Почему PromeTorch

**1. Российские ускорители.** Единственный известный нам training framework с нативной
сборкой под **Эльбрус E8C2** (E2K VLIW, LCC 1.29, EML_MT BLAS) и **NM Card Mini**
(Q16.16 fixed-point эмулятор, MNIST 93.64%). Готовая кросс-компиляция под Байкал-М/С.

**2. MNIST MLP быстрее PyTorch на Эльбрусе.** На MNIST MLP-4 (784→512→256→128→10, SGD,
batch=64, 1 epoch) — 2.76 s vs PyTorch 2.7.1 16.8 s (**6.1× на этой узкой задаче**).
1840 GFLOPS через node-local EML_MT (92% пика E8C2 2 TFLOPS). На других задачах
(общий случай / реальные transformers) преимущества PyTorch сохраняются.

**3. Universal build.** Один код — CPU (AVX2+FMA/NEON/E2K), CUDA (Turing+), NM Card
(эмулятор + реальная карта в режиме inference). Собирается на Windows MSVC, Linux GCC,
Astra/ALT/RED/Elbrus OS. Autograd engine работает одинаково на всех backend'ах.

---

## Результаты

### Эльбрус E8C2

Сервер МЦСТ Эльбрус — 4x Elbrus-MCST E8C2 (VLIW), 32 ядра, 1500 MHz.
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

### NVIDIA GPU — GGUF inference

Inference GGUF-моделей (квантизация Q4_K_M) через custom INT4 warp-cooperative GEMV:

| Модель | PromeTorch | Hardware | Ollama | vs Ollama |
|--------|-----------|----------|--------|-----------|
| qwen3:4b | **49.9 tok/s** | **NVIDIA A100** (CUDA Graph) | 165 tok/s | 30% |
| qwen3:4b | **11.3 tok/s** | RTX consumer-class GPU | — | — |
| deepseek-r1:8b | 30.5 tok/s | A100 | 133 tok/s | 23% |

> Числа на A100 получены на арендованном сервере, воспроизводится через `benchmark_gguf.py`.
> Large gap vs Ollama — Ollama использует cuBLAS+cuSPARSELt+kv-page attention, у нас только
> Q4_K GEMV + baseline RoPE. Для consumer GPU gap увеличивается.
> Tokenizer и KV-cache корректны, output верифицирован на простых задачах
> («2+2 → 4»); на длинных генерациях местами наблюдаются повторы — вероятно, баг в RoPE
> rescaling при длинных context, под расследованием.

### Точность обучения (10 training tasks)

Файл `examples/mnist/train_10_models.cpp` — 10 обучающих таcков в одном бинаре:

| Задача | Данные | Accuracy | Примечание |
|--------|--------|----------|------------|
| Linear / Logistic regression | synthetic | match PyTorch | warmup |
| XOR (2-layer MLP) | synthetic | 100% | warmup |
| MNIST MLP (784→128→10) | MNIST real | 92.68% | CUDA |
| MNIST Deep (784→512→256→128→10, Adam) | MNIST real | **97.5%** | CUDA |
| MNIST + Dropout | MNIST real | 97.15% | CUDA |
| MNIST wide + serialize/load | MNIST real | 97.78% | CUDA |
| RNNCell sine regression | synthetic | MSE 1.67e-5 | CUDA |
| LSTM classifier | **sum-sign of random walk** (synthetic) | 93–95% | NOT Shakespeare |
| GRU trend detector | synthetic | **98.44%** | CUDA |

На CPU с Adam loss/accuracy совпадают с PyTorch 2.7.1 baseline в пределах ±0.5%.

**Каких задач нет в этой таблице:** самостоятельного обучения Transformer на больших
датасетах, ViT, ResNet, VAE. Заготовки `examples/transformer/` и `examples/vit/` в репо
есть, но никогда не собирались end-to-end.

### PIR 250M на Эльбрусе (Local SGD data-parallel)

Тренировка собственной linear-RNN LM архитектуры (selective scan, родственник Mamba/HGRN/
RWKV) на русскоязычном корпусе Wikipedia (2 ГБ, BPE-токенизация SentencePiece 100k).
Полностью на PromeTorch — ни строчки PyTorch в обучении.

| Метрика | Значение |
|---------|----------|
| Архитектура | 342M params, 768d, 16 layers × 4 PIR sublayers, SwiGLU FFN (H=1792), seq=2048 |
| Parallelism | 4 процесса × 8 ядер **Local SGD** (file-based weight averaging, не gradient AllReduce — ≠ DDP) |
| Throughput | 4 × ~140 tok/s = **~560 tok/s aggregate** |
| Related arch | PIR — diagonal selective scan `h[t] = σ(gate)*base_decay*h[t-1] + σ(gate)*value`. Близко к Mamba (A diagonal SSM), HGRN/RWKV (multi-scale decay) |
| Checkpoint | Формат: raw float32 concatenation по param order. Python inference: `pir_infer.py` |

> **Статус:** данные, представленные ранее ("loss 1.04, coherent Russian"), получены на версии
> с частично сломанным backward (embedding/scan/gate/value градиенты отсутствовали). Цифры
> воспроизведения на исправленном backward — в процессе. Public-ready числа и sample
> generation обновятся после следующей full-run тренировки.

### NM Quad (профиль на удалённой плате НТЦ Модуль)

| Метрика | Значение | Caveat |
|---------|----------|--------|
| SIMD speedup (nmpp MM 32f) | **100×** | vs собственный скалярный C++ на 1 ядре (не vs cuBLAS / PyTorch) |
| Max stable cores | **16 из 64** | 64-ядерный режим приводит к DDR contention / hang |
| Throughput (16 ядер) | 705 tok/s | на игрушечном GPT D=128, L=2, V=65 (tiny_shakespeare); loss 4.45 (модель не сходится, это throughput microbenchmark, не training run) |
| PIR 250M на NM Quad | **TODO/planned (Phase 23)** | пока не реализовано |

### Поддерживаемые платформы

| Производитель | Процессор | Архитектура | Backend | Статус |
|---|---|---|---|---|
| **МЦСТ** | Эльбрус 8C2 | E2K VLIW | TUDA (CPU + EML_MT) | Нативная сборка, 38/38 тестов, PIR training |
| **НТЦ Модуль** | NM Card Mini K1879VM8YA | NeuroMatrix DSP | NMCard | Q16.16 эмулятор (34 tests, MNIST 93.64%) + 1-core inference на реальной карте |
| **НТЦ Модуль** | NM Quad (4×NM6408) | 64 NMC4 + 20 ARM | NMQuad | 100× SIMD vs own scalar, max 16 cores stable, tiny-GPT microbenchmark only |
| **Байкал Электроникс** | Байкал-М BE-M1000 | ARM Cortex-A57 | TUDA (NEON) | Только кросс-компиляция, не протестировано на железе |
| **Байкал Электроникс** | Байкал-С BE-S1000 | ARM Cortex-A75 | TUDA (NEON+dotprod) | Только кросс-компиляция, не протестировано на железе |
| Intel / AMD | x86-64 | AVX2 + FMA | TUDA (CPU) | Основная платформа разработки |
| NVIDIA | Turing+ (sm_75) | CUDA | cuBLAS wrapper + custom Q4_K GEMV | matmul через cuBLAS; 133 .cu файлов, но большинство — decorative/unused |

| ОС | Тип | Статус тестов |
|---|---|---|
| **Astra Linux SE** 1.7+ | Debian, ФСТЭК | 34/34 PASS (Docker) |
| **ALT Linux SP** 10+ | Sisyphus, ФСТЭК | Готов |
| **RED OS** 7.3+ | RHEL, ФСТЭК | 34/34 PASS (Docker) |
| **Elbrus OS** (PDK LE) | МЦСТ, E2K | 34/34 PASS (Docker), 38/38 нативно |
| Windows 10/11 | MSVC 2019+ | Основная платформа |
| Ubuntu / Debian | GCC 9+ | Поддержка |

---

## Known Limitations (будет исправлено — PR welcome)

Это solo-проект за месяц. Вот что честно **не готово** к production:

### Core
- **dtype coverage**: реально работает только `float32`. `Half`/`BFloat16`/`Complex`/`Bool`
  объявлены в `ScalarType.h`, но `PT_DISPATCH_ALL_TYPES` их пропускает — operations with
  FP16/BF16 input бросают "Unsupported dtype". AMP API есть, но настоящих FP16 CUDA kernel'ов
  нет.
- **ChannelsLast**: метаданные есть (`MemoryFormat::ChannelsLast`), но fast paths для
  Conv2d/BN/Pool не реализованы — layout игнорируется при compute.
- **Double backward (`create_graph=True`)**: флаг принимается и игнорируется. `.grad`
  доступен после первого backward, но вторичный — не поддерживается.

### CUDA
- **FlashAttention** (`aten/src/ATen/cuda/FlashAttention.cu`): 6 подтверждённых багов
  (`INFRASTRUCTURE_AUDIT.md:73-101`), включая `dim3(64,64)=4096 threads > max 1024` — кернел
  физически не запускается. В `torch/` нет callsites — модуль de facto не используется.
- **cuDNN**: wrappers есть (`aten/src/ATen/cudnn/`), но sources НЕ в CMakeLists для aten_cuda.
  0 callsites в torch/ — этот код dead. Conv через custom naive CUDA kernel, не cuDNN.
- **Custom GEMM**: `CUDABlas.cu` декларирует собственный tile-kernel, но используется
  `cublasSgemm`. Custom kernel — dead code.

### Training
- **PIR backward** (`examples/pir/fused_trainer.h`): embedding/scan/gate/value gradients
  были stub'ами до апреля 2026 — текущая версия чинит все 4 пути (см. JOURNAL.md). Старые
  claims про "loss 1.04 coherent Russian" относятся к broken-backward версии и не
  воспроизводимы; валидные числа будут опубликованы после завершения новой тренировки.
- **Distributed**: `grad_sync.h` — это **Local SGD** (file-based weight averaging, period=N
  steps), а не DDP (не gradient AllReduce). Схождение гарантируется только для
  small-local-step регима.

### Stubs / NotImplemented
- `Conv3d::forward` — возвращает `zeros()` с комментарием "simplified for brevity".
- `CTCLoss` — `throw`.
- `CrossEntropyLoss(reduction="none")` — `throw`.
- Некоторые cuDNN CPU fallbacks — `PT_ERROR("not implemented")`.

### Examples never built end-to-end
- `examples/transformer/train_transformer.cpp` (код есть, binary не собирался в 34
  build-директориях).
- `examples/vit/train_vit.cpp`.
- `examples/shakespeare/train.cpp`.

Полный аудит: **`INFRASTRUCTURE_AUDIT.md`** (43 bug-а на момент аудита, из них
31 исправлено в текущей версии).

---

## Быстрый старт

### Сборка на x86 (Linux)

```bash
git clone https://github.com/barometech/PromeTorch.git
cd promethorch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
```

### Сборка на x86 (Windows)

> **ВАЖНО:** Сборка на Windows работает ТОЛЬКО из Developer Command Prompt (не из Git Bash, PowerShell или WSL). `rc.exe` не найдётся без `vcvarsall.bat`.

```batch
REM Открыть Developer Command Prompt for VS 2019 (или запустить вручную):
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\path\to\promethorch
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

## Справочник API (API Reference)

PromeTorch предоставляет API, максимально приближенный к PyTorch, как в C++, так и в Python.

### 1. Тензоры и операции (ATen)
Базовый класс `at::Tensor` (в Python `torch.Tensor`) поддерживает семантику ссылок и copy-on-write.

* **Методы тензора:** `.clone()`, `.detach()`, `.contiguous()`, `.to(device/dtype)`, `.item()`, `.copy_()`, `.view()`, `.reshape()`, `.flatten()`, `.squeeze()`, `.unsqueeze()`, `.transpose()`, `.permute()`, `.expand()`, `.repeat()`, `.split()`, `.chunk()`.
* **Фабрики:** `tensor`, `empty`, `zeros`, `ones`, `full`, `rand`, `randn`, `randint`, `arange`, `linspace`, `eye`, `zeros_like`, `ones_like`, `from_numpy`.
* **Математика:** `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqrt`, `rsqrt`, `square`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `clamp`, `nan_to_num`.
* **Редукции:** `sum`, `mean`, `max`, `min`, `norm`, `topk`, `sort`, `argmax`, `argmin`.
* **Линейная алгебра:** `mm`, `bmm`, `matmul`, `dot`, `einsum`.
* **Проверки:** `isinf`, `isnan`, `isfinite`.

### 2. Нейросетевые модули (`torch::nn` / `torch.nn`)
Все модели наследуются от `torch::nn::Module`.
* **Базовые методы Module:** `.forward()`, `.train()`, `.eval()`, `.to()`, `.zero_grad()`, `.parameters()`, `.named_parameters()`, `.state_dict()`, `.load_state_dict()`.
* **Хуки:** `ForwardPreHook`, `ForwardHook`, `ForwardHookWithReturn`.
* **Слои:** `Linear`, `Bilinear`, `LazyLinear`, `Identity`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose2d`.
* **Активации:** `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `SELU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `Softplus`, `Softsign`, `Hardtanh`, `Hardsigmoid`, `Hardswish`, `SiLU`, `Mish`, `Threshold`.
* **Нормализация:** `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm2d`.
* **RNN & Transformer:** `RNN`, `LSTM`, `GRU`, `TransformerEncoder`, `TransformerDecoder`, `MultiheadAttention`, `PositionalEncoding`.
* **PIR Architecture:** `PIRBlock`, `PIRLayer`, `PIR270M` — Parallel Infinite Retention (без attention, O(T) memory).
* **Функции потерь:** `L1Loss`, `MSELoss`, `SmoothL1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `NLLLoss`, `KLDivLoss`, `CosineEmbeddingLoss`, `TripletMarginLoss`, `CTCLoss`.

### 3. Оптимизаторы и планировщики (`torch::optim`)
* **Оптимизаторы:** `SGD`, `Adam`, `AdamW`, `RMSprop`. Fused версии: `fused_adam_multi`, `fused_sgd_multi` (все параметры за один вызов).
* **LR Schedulers:** `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LinearLR`, `ConstantLR`, `ReduceLROnPlateau`, `WarmupCosineAnnealingLR`, `OneCycleLR`.

### 4. Autograd и Checkpointing
* **Gradient computation:** `torch::autograd::backward()`, `NoGradGuard`, `EnableGradGuard`.
* **Custom Functions:** `torch::autograd::Function<Derived>` + `FunctionCtx` + `save_for_backward()`.
* **Gradient Checkpointing:** `torch::utils::checkpoint(fn, inputs)`.
* **116 backward functions** для всех операций.

### 5. Загрузка данных (`torch::data`)
* **Датасеты:** `Dataset`, `TensorDataset`, `MapDataset`, `ConcatDataset`, `SubsetDataset`, `random_split`.
* **DataLoader:** `batch_size`, `shuffle`, `drop_last`. Samplers: `SequentialSampler`, `RandomSampler`, `BatchSampler`.

### 6. Сериализация (формат PTOR)
* `torch::save()`, `torch::load()`, `torch::save_state_dict()`, `torch::load_state_dict()`.

### 7. Mixed Precision (AMP)
* **GradScaler:** `.scale(loss)`, `.unscale(optimizer)`, `.step(optimizer)`, `.update()`.
* **Autocast:** контекстный менеджер для FP16/BF16.

### 8. Backends
* **CPU:** AVX2 vectorized ops, cache-tiled GEMM, OpenMP parallelism.
* **CUDA:** Custom kernels (GEMM, reduce, element-wise), cuDNN (conv, pool, batchnorm), FlashDecoding (decode attention), Quantized inference (Q4_K_M, Q5_K, Q6_K, Q8_1). *(FlashAttention forward+backward временно отключён — требует доработки.)*
* **Elbrus E2K:** EML BLAS, NUMA-aware 4-chip training, VLIW-optimized fused ops.
* **NMCard Mini:** Q16.16 fixed-point, 64-core NMC4 emulator + hardware driver.

### 9. PromeServe (Inference Server)
Ollama-совместимый LLM сервер. API: `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`.
Web UI с streaming chat, markdown rendering, syntax highlighting.

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
│  Engine   Node   Edge   116 backward функций           │
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
- **TensorImpl** — N-мерные тензоры, strides, views, channels-last (NHWC), `trusted_` flag для zero-overhead dispatch
- **CPUAllocator** — 3-уровневый caching: 16MB arena (lock-free) + thread-local cache (64 слота, zero-mutex) + глобальный bucket cache (256 слотов × 32 bucket). **97.7% cache hit rate, 179 аллокаций за epoch** (было 37,000)
- **Device** — 22 типа устройств (CPU, CUDA, NMCard, LinQ и др.)
- **ScalarType** — 16 типов данных (Float, Double, Half, BFloat16, QInt8 и др.) + type promotion
- **Storage** — ref-counted память с CUDA/NMCard/LinQ backend
- **ThreadPool** — persistent worker threads для Эльбруса (замена OpenMP fork/join)
- **SmallVector<5>** — SSO-оптимизация: тензоры до 5D не аллоцируют heap

### Операции (ATen) — ~18,000 строк

149 тензорных операций с SIMD-оптимизацией + 132 CUDA ядра:
- **Математика** — 20 unary (exp, log, sin, cos, tanh, sigmoid...) + 12 binary (add, mul, div, pow...)
- **Редукции** — sum, mean, max, min, var, std, argmax, argmin, norm, prod (с dim и keepdim)
- **Линейная алгебра** — mm, bmm, mv, dot, outer, addmm, LU, QR, SVD, Cholesky, solve, det
- **Форма** — view, reshape, transpose, permute, cat, stack, split, chunk, squeeze, unsqueeze, flatten
- **Индексирование** — index, index_put, scatter, gather, masked_select, boolean indexing
- **Фабричные** — zeros, ones, rand, randn, arange, linspace, eye, full, empty, multinomial
- **FFT** — fft, ifft, rfft, fft2

### Autograd — ~6,400 строк

Reverse-mode автоматическое дифференцирование:
- **Engine** — топологическая сортировка, cached GraphTask (reuse между backward)
- **116 backward-функций** — Math(46) + LinAlg(13) + Shape(21) + Reduce(16) + Index(2) + Fused(4) + Conv(4) + AccumulateGrad
- **Fused backward** — FusedLinearRelu (4 nodes → 1), FusedMLP (12 nodes → 1), PrecomputedGrad (zero-compute backward)
- **NodePool** — thread-local object pool для backward nodes (zero malloc в hot path)
- **SmallEdgeList<4>** — inline edges без heap allocation (99% ops)
- **Conv2d/BatchNorm/Pool backward** — im2col-based, полная CNN тренировка
- **Custom autograd functions** — CRTP `Function<Derived>` + gradient checkpointing

### NN Modules — ~9,000 строк, 97 модулей

| Категория | Модули |
|-----------|--------|
| Слои | Linear (fused_relu), LowRankLinear (SVD-сжатие), Bilinear, LazyLinear, Conv1d/2d/3d, ConvTranspose2d |
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

### Оптимизаторы — 10 штук

SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax, AdamKiller (экспериментальный).
**Fused multi-param**: `fused_adam_multi` / `fused_sgd_multi` — один вызов на все параметры, VecF SIMD.

**LR Schedulers** (9 штук): StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, ConstantLR, ReduceLROnPlateau, WarmupCosineAnnealingLR, OneCycleLR.

### CUDA Backend — 99 ядер

- **65 element-wise** ядер (unary, binary, comparison, fused) с grid-stride loops
- **18 reduction** ядер (sum, max, min, var, cross_entropy, nll_loss)
- **9 BLAS** ядер (GEMM 4 варианта, batched, GEMV, dot, outer, transpose)
- **cuBLAS** — cublasSgemm для всех GEMM (замена custom 32x32 kernel)
- **cuDNN** — conv (fwd+bwd), pooling (fwd+bwd), batchnorm, activations (~12 ops)
- **FlashAttention** — forward + backward с online softmax, O(N) memory *(временно отключён, требует доработки)*
- **FlashDecoding** — parallel KV cache chunks, fused QKnorm+RoPE+KVwrite
- **Quantized GEMV** — Q4_K (persistent + fused gate+up), Q5_K, Q6_K, dp4a Q4_K×Q8_1
- **Mixed Precision** — GradScaler + Autocast (FP32/FP16/BF16)

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
| GGUF Inference | `examples/gguf/test_gguf_inference.cpp` | Запуск GGUF-моделей |
| NM Card Mini | `examples/nmcard/` | MNIST на нейропроцессоре |

---

## Структура проекта

```
c10/                          Ядро: Allocator, Device, Storage, TensorImpl, ScalarType
  cuda/                       CUDACachingAllocator (singleton pattern)
  nmcard/                     NMCardAllocator (caching, PrivateUse1)
  linq/                       LinQAllocator (caching, PrivateUse2)

aten/src/ATen/                Операции (120+ CPU, 132 CUDA ядра)
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
  csrc/autograd/              Engine, Node, Edge, 116 backward функций
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
| **Всего строк** | **128,000+** |
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
