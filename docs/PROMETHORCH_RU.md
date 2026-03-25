# PromeTorch -- Документация

**Версия:** 1.0
**Дата:** 2026-03-25
**Кодовая база:** ~93,000+ строк C++17/CUDA/Python, 481 файл (277 исходных)

---

## Содержание

1. [Что такое PromeTorch](#1-что-такое-promethorch)
2. [Структура проекта](#2-структура-проекта)
3. [Ядро c10/](#3-ядро-c10)
4. [Тензорные операции aten/](#4-тензорные-операции-aten)
5. [Autograd -- автоматическое дифференцирование](#5-autograd----автоматическое-дифференцирование)
6. [NN модули -- нейросетевые слои](#6-nn-модули----нейросетевые-слои)
7. [Оптимизаторы и LR Schedulers](#7-оптимизаторы-и-lr-schedulers)
8. [Data Loading -- загрузка данных](#8-data-loading----загрузка-данных)
9. [CUDA backend](#9-cuda-backend)
10. [SIMD-оптимизация (AVX2)](#10-simd-оптимизация-avx2)
11. [Serialization -- сериализация](#11-serialization----сериализация)
12. [Channels-last Memory Format](#12-channels-last-memory-format)
13. [GGUF Inference -- инференс LLM](#13-gguf-inference----инференс-llm)
14. [Python bindings](#14-python-bindings)
15. [Сборка](#15-сборка)
16. [Результаты и бенчмарки](#16-результаты-и-бенчмарки)
17. [Дополнительные backend-ы](#17-дополнительные-backend-ы)
18. [Известные ограничения](#18-известные-ограничения)
19. [Планы развития](#19-планы-развития)

---

## 1. Что такое PromeTorch

**PromeTorch** (Prometheus + Torch) -- полнофункциональный фреймворк глубокого обучения, написанный с нуля на C++17 с поддержкой CUDA и Python. Проект не использует код PyTorch, TensorFlow или иных фреймворков -- все компоненты реализованы самостоятельно:

- Тензорная библиотека с автоматическим управлением памятью
- Система автоматического дифференцирования (autograd) с динамическим вычислительным графом
- 80+ нейросетевых модулей (Linear, Conv, Transformer, LSTM, GRU и др.)
- 12 оптимизаторов и 9 планировщиков скорости обучения
- Собственные CUDA-ядра (GEMM, reduce, element-wise, FlashAttention)
- Интеграция cuDNN для свёрточных операций
- Mixed Precision Training (AMP)
- Три аппаратных backend-а: CPU, CUDA, NM Card Mini (эмулятор + реальное железо)
- Python-обёртки через pybind11
- GGUF-инференс для квантованных LLM

Архитектура PromeTorch повторяет архитектуру PyTorch: низкоуровневое ядро `c10`, библиотека тензорных операций `aten`, высокоуровневый API `torch` с autograd, nn-модулями, оптимизаторами и утилитами.

### Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Язык ядра | C++17 |
| GPU-вычисления | CUDA 12.9, cuDNN 9.14, cuBLAS |
| CPU-оптимизации | AVX2/FMA (собственный PromeBLAS) |
| Python-биндинги | pybind11 |
| Система сборки | CMake + NMake Makefiles |
| Компилятор | MSVC 2019 (Windows), LCC 1.29 (Эльбрус) |

---

## 2. Структура проекта

```
promethorch/
├── c10/                          # Ядро фреймворка
│   ├── core/                     # TensorImpl, Storage, Allocator, Device, ScalarType
│   ├── macros/                   # Платформенные макросы (PT_HOST_DEVICE и др.)
│   ├── util/                     # Exception, Logging, SmallVector
│   ├── cuda/                     # CUDACachingAllocator, CUDAStream
│   └── nmcard/                   # NMCardAllocator (caching, PrivateUse1)
│
├── aten/src/ATen/                # Библиотека тензорных операций
│   ├── core/                     # Tensor.h, TensorFactory.h
│   ├── native/cpu/              # CPU-ядра (MathOps, ReduceOps, LinearAlgebra и др.)
│   ├── cuda/                     # CUDA-ядра (.cu файлы)
│   ├── cudnn/                    # cuDNN-обёртки (свёртки, пулинг, нормализация)
│   └── nmcard/                   # NM Card эмулятор, Q16.16 арифметика
│
├── torch/                        # Высокоуровневый API
│   ├── csrc/autograd/            # Движок autograd (engine, node, edge, backward functions)
│   ├── autograd/                 # Custom Autograd Functions
│   ├── nn/                       # Module базовый класс, 80+ слоёв
│   │   └── modules/             # linear, conv, activation, normalization, loss, rnn, transformer, pir
│   ├── optim/                    # Оптимизаторы (SGD, Adam, AdamW и др.)
│   ├── data/                     # Dataset, DataLoader, Sampler
│   ├── amp/                      # GradScaler, Autocast (Mixed Precision)
│   ├── utils/                    # checkpoint.h (gradient checkpointing)
│   └── serialization.h          # Сериализация (формат PTOR)
│
├── python/                       # pybind11-обёртки, пакет promethorch
│
├── examples/                     # Примеры обучения
│   ├── mnist/                    # MNIST MLP
│   ├── pir/                      # PIR (трансформер, генерация текста)
│   └── nmcard/                   # MNIST на NM Card эмуляторе
│
├── PIR/                          # PIR Python API (PROMEPIR.py)
└── docs/                         # Документация
```

---

## 3. Ядро c10/

Ядро `c10` предоставляет базовые абстракции, от которых зависят все остальные компоненты фреймворка.

### 3.1 Device -- абстракция устройства

Файл: `c10/core/Device.h`

Перечисление `DeviceType` определяет все поддерживаемые типы устройств:

| DeviceType | ID | Описание |
|------------|-----|---------|
| `CPU` | 0 | Центральный процессор |
| `CUDA` | 1 | NVIDIA GPU |
| `PrivateUse1` | 20 | NM Card Mini (пользовательский backend) |
| `PrivateUse2` | 21 | LinQ (зарезервировано) |
| `PrivateUse3` | 22 | NM Quad (4x NM6408) |
| `HIP` | 6 | AMD ROCm (зарезервировано) |
| `XLA` | 9 | Google TPU (зарезервировано) |
| `MPS` | 13 | Apple Metal (зарезервировано) |
| `Meta` | 14 | Мета-тензоры без данных |

Класс `Device` инкапсулирует тип устройства и индекс (для мульти-GPU конфигураций):

```cpp
c10::Device cpu_device(c10::DeviceType::CPU);
c10::Device gpu_device(c10::DeviceType::CUDA, 0);
```

### 3.2 ScalarType -- типы данных

Файл: `c10/core/ScalarType.h`

Поддерживаемые скалярные типы:

| Тип | Размер | Описание |
|-----|--------|---------|
| `Float` (float32) | 4 байта | Основной тип для обучения |
| `Double` (float64) | 8 байт | Повышенная точность |
| `Half` (float16) | 2 байта | Смешанная точность (AMP) |
| `BFloat16` | 2 байта | Brain float |
| `Int` (int32) | 4 байта | Целочисленные индексы |
| `Long` (int64) | 8 байт | Длинные индексы |
| `Short` (int16) | 2 байта | Короткие целые |
| `Byte` (uint8) | 1 байт | Беззнаковые байты |
| `Char` (int8) | 1 байт | Знаковые байты |
| `Bool` | 1 байт | Логический тип |
| `ComplexFloat` | 8 байт | Комплексные числа |
| `ComplexDouble` | 16 байт | Комплексные числа двойной точности |

Тип `Half` реализован программно с побитовыми преобразованиями IEEE 754 для платформ без аппаратной поддержки FP16.

### 3.3 Allocator -- управление памятью

Файл: `c10/core/Allocator.h`

Абстрактный интерфейс `Allocator` определяет API выделения/освобождения памяти:

```cpp
class Allocator {
public:
    virtual void* allocate(size_t n) = 0;
    virtual void deallocate(void* ptr) = 0;
};
```

Реализации:
- **CPUAllocator** -- стандартный `malloc`/`free` с выравниванием
- **CUDACachingAllocator** -- кэширующий аллокатор для CUDA (аналог PyTorch: не вызывает `cudaFree` при деаллокации, переиспользует блоки)
- **NMCardAllocator** -- кэширующий аллокатор для NM Card Mini

Реестр аллокаторов (`AllocatorRegistry`) обеспечивает поиск аллокатора по `DeviceType` в рантайме.

**Решённая проблема**: DLL Singleton -- `inline static` в header-файле создавал отдельные экземпляры синглтона в каждой DLL. Решение: единственный экземпляр в `.cpp` файле.

### 3.4 Storage -- хранение данных

Файл: `c10/core/Storage.h`

`Storage` владеет непрерывным блоком памяти и отвечает за его время жизни через подсчёт ссылок. Несколько тензоров могут разделять один `Storage` (views).

```cpp
class Storage {
    void* data_ptr_;        // Указатель на данные
    size_t numel_;          // Количество элементов
    size_t nbytes_;         // Размер в байтах
    Allocator* allocator_;  // Аллокатор для освобождения
    Device device_;         // Устройство, на котором выделена память
    int refcount_;          // Счётчик ссылок
};
```

### 3.5 TensorImpl -- реализация тензора

Файл: `c10/core/TensorImpl.h`

`TensorImpl` -- низкоуровневое представление тензора, содержащее:

- **Storage** -- блок памяти с данными
- **sizes** -- размеры тензора (shape)
- **strides** -- шаги по каждому измерению (для поддержки views и transposes)
- **storage_offset** -- смещение в Storage (для narrow/select)
- **dtype** -- тип данных (ScalarType)
- **device** -- устройство
- **MemoryFormat** -- формат памяти (Contiguous, ChannelsLast)

Поддержка non-contiguous тензоров (views, transposes, narrow) через strides-арифметику. Вычисление физического индекса элемента:

```
physical_offset = storage_offset + sum(logical_index[i] * stride[i])
```

### 3.6 MemoryFormat

```cpp
enum class MemoryFormat {
    Contiguous,    // NCHW (по умолчанию)
    ChannelsLast,  // NHWC (оптимально для cuDNN)
    Preserve       // Сохранить текущий формат
};
```

---

## 4. Тензорные операции aten/

### 4.1 Класс Tensor

Файл: `aten/src/ATen/core/Tensor.h`

Класс `Tensor` -- основной пользовательский интерфейс для работы с тензорами. Это обёртка над `TensorImpl` с подсчётом ссылок и поддержкой autograd.

Основные методы создания:

```cpp
auto x = torch::zeros({3, 4});              // Нулевой тензор 3x4
auto y = torch::ones({2, 3});               // Единичный тензор
auto z = torch::randn({5, 5});              // Нормальное распределение
auto w = torch::rand({3, 3});               // Равномерное распределение [0, 1)
auto e = torch::eye(4);                     // Единичная матрица
auto a = torch::arange(0, 10, 2);           // [0, 2, 4, 6, 8]
auto l = torch::linspace(0, 1, 100);        // 100 точек от 0 до 1
auto f = torch::full({2, 3}, 3.14f);        // Заполнение константой
auto t = torch::from_blob(data, {3, 4});    // Из существующего буфера
```

Файл: `aten/src/ATen/core/TensorFactory.h`

### 4.2 MathOps -- математические операции

Файл: `aten/src/ATen/native/cpu/MathOps.h`

#### Унарные операции (20+):

| Операция | Формула | Backward |
|----------|---------|----------|
| `neg` | -x | -grad |
| `abs` | \|x\| | grad * sign(x) |
| `sqrt` | sqrt(x) | grad / (2 * sqrt(x)) |
| `rsqrt` | 1/sqrt(x) | -grad / (2 * x^(3/2)) |
| `square` | x^2 | 2 * x * grad |
| `exp` | e^x | e^x * grad |
| `log` | ln(x) | grad / x |
| `log2` | log2(x) | grad / (x * ln(2)) |
| `log10` | log10(x) | grad / (x * ln(10)) |
| `sin` | sin(x) | cos(x) * grad |
| `cos` | cos(x) | -sin(x) * grad |
| `tan` | tan(x) | (1 + tan^2(x)) * grad |
| `tanh` | tanh(x) | (1 - tanh^2(x)) * grad |
| `sigmoid` | 1/(1+e^(-x)) | sigmoid(x) * (1 - sigmoid(x)) * grad |
| `relu` | max(0, x) | grad * (x > 0) |
| `silu` | x * sigmoid(x) | ... |
| `reciprocal` | 1/x | -grad / x^2 |
| `ceil` | ceil(x) | 0 |
| `floor` | floor(x) | 0 |
| `round` | round(x) | 0 |

#### Бинарные операции (12):

| Операция | Формула |
|----------|---------|
| `add` | a + b (с alpha-множителем) |
| `sub` | a - b |
| `mul` | a * b (поэлементное) |
| `div` | a / b |
| `pow` | a^b |
| `fmod` | остаток от деления |
| `min` (поэлементный) | min(a, b) |
| `max` (поэлементный) | max(a, b) |
| `clamp` | clamp(x, min, max) |
| `where` | condition ? a : b |
| `masked_fill` | заполнение по маске |
| `lerp` | a + weight * (b - a) |

Все бинарные операции поддерживают **broadcasting** -- автоматическое расширение размеров.

#### Специальные функции:

- `softmax(x, dim)` -- стабильная реализация (вычитание максимума)
- `log_softmax(x, dim)` -- объединённая реализация для стабильности
- `cross_entropy_loss` -- log_softmax + NLL (численно устойчивый)

### 4.3 ReduceOps -- операции свёртки

Файл: `aten/src/ATen/native/cpu/ReduceOps.h`

| Операция | Описание | Поддержка dim |
|----------|---------|---------------|
| `sum` | Сумма элементов | Полная + keepdim |
| `mean` | Среднее | Полная + keepdim |
| `prod` | Произведение | Полная + keepdim |
| `max` / `min` | Максимум / минимум | Полная + argmax/argmin |
| `var` | Дисперсия | С коррекцией Бесселя |
| `std` | Стандартное отклонение | С коррекцией Бесселя |
| `norm` | Норма (Lp) | L1, L2, Frobenius |
| `cumsum` | Кумулятивная сумма | По указанному dim |
| `cumprod` | Кумулятивное произведение | По указанному dim |
| `sort` | Сортировка | По указанному dim |
| `topk` | Top-K элементов | По указанному dim |
| `argmax` / `argmin` | Индексы экстремумов | Глобальный и по dim |
| `any` / `all` | Логические | Глобальный и по dim |

Все операции свёртки имеют backward-функции для autograd.

### 4.4 LinearAlgebra -- линейная алгебра

Файл: `aten/src/ATen/native/cpu/LinearAlgebra.h`

| Операция | Описание | Backward |
|----------|---------|----------|
| `mm(A, B)` | Матричное умножение (MxK @ KxN -> MxN) | dA = dOut @ B^T, dB = A^T @ dOut |
| `mv(A, v)` | Матрица на вектор | Реализован |
| `bmm(A, B)` | Батчевое матричное умножение | Реализован |
| `dot(a, b)` | Скалярное произведение векторов | Реализован |
| `outer(a, b)` | Внешнее произведение | Реализован |
| `addmm(bias, A, B)` | bias + A @ B | Реализован |
| `matmul(A, B)` | Универсальное умножение (1D-4D) | Реализован |
| `einsum` | Нотация Эйнштейна | Реализован |
| `inverse(A)` | Обратная матрица | Реализован |
| `det(A)` | Определитель | Реализован |
| `trace(A)` | След матрицы | Реализован |
| `cholesky(A)` | Разложение Холецкого | Реализован |
| `qr(A)` | QR-разложение | Через Gram-Schmidt |
| `svd(A)` | Сингулярное разложение | Через степенные итерации |

**Критически важно**: все функции линейной алгебры вызывают `.contiguous()` на входных тензорах перед обращением к raw-указателям. Без этого транспонированные views (с нестандартными strides) дают некорректные результаты.

### 4.5 ShapeOps -- операции над формой

Файл: `aten/src/ATen/native/cpu/ShapeOps.h`

| Операция | Описание |
|----------|---------|
| `reshape` / `view` | Изменение формы (view -- только для contiguous) |
| `flatten` | Вытягивание в 1D |
| `squeeze` / `unsqueeze` | Удаление / добавление размерности |
| `permute` | Перестановка осей |
| `transpose` / `t` | Транспонирование |
| `expand` | Виртуальное расширение (без копирования) |
| `repeat` | Повторение данных |
| `cat` | Конкатенация по оси |
| `stack` | Стекирование (новая ось) |
| `split` / `chunk` | Разбиение по оси |
| `select` | Выбор среза по индексу |
| `narrow` | Сужение диапазона по оси |
| `slice` | Общий срез (start:stop:step) |
| `flip` | Отражение по оси |
| `roll` | Циклический сдвиг |
| `contiguous` | Создание contiguous-копии |

Все операции создают views (без копирования данных), если это возможно. Backward-функции реализованы для каждой операции.

### 4.6 IndexOps -- индексирование

Файл: `aten/src/ATen/native/cpu/IndexOps.h`

| Операция | Описание |
|----------|---------|
| `index_with_tensor` | Индексирование тензором (gather-like) |
| `index_put_` | Запись по тензорному индексу (scatter-like) |
| `scatter_add_` | Scatter с суммированием |
| `boolean_index` | Индексирование булевой маской |
| `gather` | Сбор элементов по индексам |
| `scatter_` | Раскидывание элементов по индексам |

### 4.7 FFTOps -- быстрое преобразование Фурье

Файл: `aten/src/ATen/native/cpu/FFTOps.h`

- `fft` / `ifft` -- прямое и обратное одномерное БПФ
- `rfft` / `irfft` -- БПФ для действительных сигналов

---

## 5. Autograd -- автоматическое дифференцирование

PromeTorch реализует **обратное автоматическое дифференцирование** (reverse-mode AD) с динамическим вычислительным графом, аналогично PyTorch.

### 5.1 Архитектура

```
Tensor.backward()
    |
    v
Engine::execute()          # Обход графа в обратном порядке
    |
    v
Node (backward function)  # Вычисление grad_input из grad_output
    |
    v
Edge (input_nr, node_ptr) # Связь: какому входу какого узла передать градиент
```

Основные файлы:
- `torch/csrc/autograd/engine.h` -- движок backward pass
- `torch/csrc/autograd/node.h` -- базовый класс вычислительного узла
- `torch/csrc/autograd/edge.h` -- ребро графа (связь между узлами)
- `torch/csrc/autograd/autograd_meta.h` -- метаданные autograd в тензоре
- `torch/csrc/autograd/autograd.h` -- autograd-обёртки для операций

### 5.2 Backward functions (100+)

Каждая дифференцируемая операция имеет соответствующий struct, наследующий `Node`:

**Математические** (файл: `MathBackward.h`):
NegBackward, AbsBackward, SqrtBackward, RsqrtBackward, SquareBackward, ExpBackward, LogBackward, Log2Backward, Log10Backward, SinBackward, CosBackward, TanBackward, TanhBackward, SigmoidBackward, ReluBackward, LeakyReluBackward, ELUBackward, SELUBackward, MishBackward, HardtanhBackward, HardsigmoidBackward, HardswishBackward, SiLUBackward, ReciprocalBackward, AddBackward, SubBackward, MulBackward, DivBackward, PowBackward, PowScalarBackward, AddScalarBackward, MulScalarBackward, DivScalarBackward, CloneBackward, LogSoftmaxBackward, CrossEntropyBackward, RMSNormBackward, RotaryEmbeddingBackward, EmbeddingBackward, ClampBackward, TriuBackward, TrilBackward, DiagBackward, ParallelScanBackward, MulTensorBackward, PrecomputedGradBackward

**Линейная алгебра** (файл: `LinearAlgebraBackward.h`):
MmBackward, MvBackward, BmmBackward, DotBackward, MatmulBackward, OuterBackward, AddmmBackward, TransposeBackward, EinsumBackward, InverseBackward, DetBackward, CholeskyBackward, TraceBackward

**Свёртки** (файл: `ReduceBackward.h`):
SumBackward, SumDimBackward, MeanBackward, MeanDimBackward, MaxBackward, MaxDimBackward, MinBackward, MinDimBackward, ProdBackward, VarBackward, StdBackward, NormBackward, CumsumBackward, CumprodBackward, SortBackward, TopkBackward

**Форма** (файл: `ShapeBackward.h`):
ViewBackward, ReshapeBackward, FlattenBackward, SqueezeBackward, SqueezeDimBackward, UnsqueezeBackward, PermuteBackward, ExpandBackward, RepeatBackward, CatBackward, StackBackward, SplitBackward, SelectBackward, NarrowBackward, SliceBackward, TBackward, ContiguousBackward, DetachBackward, FlipBackward, RollBackward, RepeatInterleaveBackward

**Свёрточные слои** (файл: `ConvBackward.h`):
Conv2dBackward, BatchNorm2dBackward, MaxPool2dBackward, AvgPool2dBackward

**Индексирование** (файл: `IndexBackward.h`):
IndexWithTensorBackward, BooleanIndexBackward

**Фьюзы** (файл: `FusedBackward.h`):
FusedLinearBackward, FusedLinearReluBackward, FusedMLPBackward, LowRankLinearBackward

### 5.3 Custom Autograd Functions

Файл: `torch/autograd/function.h`

Пользовательские autograd-функции через CRTP-паттерн:

```cpp
class MyFunction : public torch::autograd::Function<MyFunction> {
public:
    static at::Tensor forward(FunctionCtx& ctx, at::Tensor input) {
        ctx.save_for_backward({input});
        return input * input;
    }

    static std::vector<at::Tensor> backward(
            FunctionCtx& ctx, std::vector<at::Tensor> grad_outputs) {
        auto saved = ctx.get_saved_tensors();
        return {2.0f * saved[0] * grad_outputs[0]};
    }
};

// Использование:
auto result = MyFunction::apply(input);
```

### 5.4 Gradient Checkpointing

Файл: `torch/utils/checkpoint.h`

Gradient checkpointing позволяет обменять вычисления на память: forward pass выполняется без сохранения промежуточных тензоров (в режиме `NoGrad`), а при backward промежуточные значения пересчитываются заново.

### 5.5 Hooks System

Файл: `torch/nn/module.h`

Система хуков позволяет перехватывать forward и backward pass:

```cpp
auto handle = module.register_forward_hook(
    [](Module& m, const Tensor& input, Tensor& output) {
        std::cout << "Output shape: " << output.sizes() << std::endl;
    }
);
```

Поддерживаются: `ForwardPreHook`, `ForwardHook`, `BackwardHook`.

---

## 6. NN модули -- нейросетевые слои

Все модули наследуют от базового класса `Module`, который предоставляет:
- Регистрацию параметров и субмодулей
- `parameters()` -- доступ ко всем обучаемым параметрам
- `train()` / `eval()` -- переключение режима
- `to(device)` -- перемещение на устройство
- `state_dict()` / `load_state_dict()` -- сериализация весов
- Система хуков (forward pre/post, backward)

### 6.1 Линейные слои

Файл: `torch/nn/modules/linear.h`

| Модуль | Описание |
|--------|---------|
| `Identity` | Тождественное преобразование |
| `Linear` | y = x @ W^T + b (Kaiming uniform init) |
| `Bilinear` | y = x1^T A x2 + b |
| `LazyLinear` | Linear с отложенной инициализацией (по первому входу) |
| `Flatten` | Вытягивание в 1D (start_dim, end_dim) |
| `Unflatten` | Обратное преобразование |
| `LowRankLinear` | Низкоранговая аппроксимация W = U @ V |

### 6.2 Функции активации

Файл: `torch/nn/modules/activation.h`

| Модуль | Формула |
|--------|---------|
| `ReLU` | max(0, x) |
| `ReLU6` | min(max(0, x), 6) |
| `LeakyReLU` | max(negative_slope * x, x) |
| `PReLU` | max(a * x, x), a -- обучаемый |
| `ELU` | x if x > 0, alpha * (exp(x) - 1) otherwise |
| `SELU` | scale * ELU(x, alpha) |
| `GELU` | x * Phi(x) |
| `Sigmoid` | 1 / (1 + exp(-x)) |
| `Tanh` | tanh(x) |
| `Softmax` | exp(xi) / sum(exp(xj)) |
| `LogSoftmax` | log(softmax(x)) |
| `Softplus` | log(1 + exp(beta * x)) / beta |
| `Softsign` | x / (1 + |x|) |
| `Hardtanh` | clamp(x, min_val, max_val) |
| `Hardsigmoid` | clamp((x + 3) / 6, 0, 1) |
| `Hardswish` | x * hardsigmoid(x) |
| `SiLU` (Swish) | x * sigmoid(x) |
| `Mish` | x * tanh(softplus(x)) |
| `Threshold` | x if x > threshold, value otherwise |

### 6.3 Свёрточные слои

Файл: `torch/nn/modules/conv.h`

| Модуль | Описание |
|--------|---------|
| `Conv1d` | 1D свёртка (для сигналов, текста) |
| `Conv2d` | 2D свёртка (для изображений), im2col + SGEMM |
| `Conv3d` | 3D свёртка (stub -- возвращает нули) |
| `ConvTranspose2d` | Транспонированная 2D свёртка (upsampling) |

Conv2d поддерживает: stride, padding, dilation, groups, bias.

На GPU используется cuDNN для максимальной производительности. На CPU -- оптимизированный im2col + cache-tiled GEMM.

### 6.4 Пулинг

Файл: `torch/nn/modules/pooling.h`

| Модуль | Описание |
|--------|---------|
| `MaxPool1d` | 1D max pooling |
| `MaxPool2d` | 2D max pooling (с индексами для backward) |
| `AvgPool1d` | 1D average pooling |
| `AvgPool2d` | 2D average pooling |
| `AdaptiveAvgPool1d` | Адаптивный (заданный output_size) |
| `AdaptiveAvgPool2d` | Адаптивный 2D |
| `AdaptiveMaxPool2d` | Адаптивный max 2D |

### 6.5 Нормализация

Файл: `torch/nn/modules/normalization.h`

| Модуль | Описание | Формула |
|--------|---------|---------|
| `BatchNorm1d` | Батчевая нормализация 1D | (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta |
| `BatchNorm2d` | Батчевая нормализация 2D | То же для 4D тензоров (NCHW) |
| `LayerNorm` | Нормализация по слою | Нормализация по последним dim |
| `GroupNorm` | Групповая нормализация | Нормализация по группам каналов |
| `InstanceNorm2d` | Инстанс-нормализация | Нормализация по H, W |

Файл: `torch/nn/modules/pir.h`

| Модуль | Описание |
|--------|---------|
| `RMSNorm` | Root Mean Square нормализация (как в LLaMA) |

### 6.6 Функции потерь

Файл: `torch/nn/modules/loss.h`

| Модуль | Описание | Типичное применение |
|--------|---------|-------------------|
| `L1Loss` | MAE = mean(\|y - y_hat\|) | Регрессия |
| `MSELoss` | MSE = mean((y - y_hat)^2) | Регрессия |
| `SmoothL1Loss` | Huber-подобная L1 | Регрессия (робастная) |
| `HuberLoss` | Huber loss | Регрессия (робастная) |
| `BCELoss` | Binary Cross Entropy | Бинарная классификация |
| `BCEWithLogitsLoss` | BCE + Sigmoid (устойчивая) | Бинарная классификация |
| `NLLLoss` | Negative Log Likelihood | Классификация (после LogSoftmax) |
| `CrossEntropyLoss` | LogSoftmax + NLL (fused) | Многоклассовая классификация |
| `KLDivLoss` | KL-дивергенция | Дистилляция, VAE |
| `CosineEmbeddingLoss` | Косинусное расстояние | Metric learning |
| `MarginRankingLoss` | Ранжирование с отступом | Learning to rank |
| `TripletMarginLoss` | Триплетные потери | Metric learning |
| `MultiMarginLoss` | Multi-class hinge | SVM-подобная |
| `PoissonNLLLoss` | Пуассоновский NLL | Счётные данные |
| `GaussianNLLLoss` | Гауссовский NLL | Предсказание неопределённости |
| `CTCLoss` | Connectionist Temporal | Распознавание речи |
| `FocalLoss` | Focal Loss (Lin et al.) | Несбалансированные классы |
| `DiceLoss` | Dice coefficient loss | Сегментация |

### 6.7 Dropout

Файл: `torch/nn/modules/dropout.h`

| Модуль | Описание |
|--------|---------|
| `Dropout` | Стандартный dropout (обнуление с вероятностью p) |
| `Dropout1d` | Dropout для 1D данных |
| `Dropout2d` | Spatial dropout (обнуление каналов целиком) |
| `Dropout3d` | 3D spatial dropout |
| `AlphaDropout` | Dropout для SELU (сохраняет self-normalizing свойство) |
| `FeatureAlphaDropout` | Feature-уровневый AlphaDropout |

### 6.8 Рекуррентные слои (RNN/LSTM/GRU)

Файл: `torch/nn/modules/rnn.h`

| Модуль | Описание |
|--------|---------|
| `RNNCellImpl` | Одна ячейка RNN: h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh) |
| `LSTMCellImpl` | Одна ячейка LSTM: 4 гейта (input, forget, cell, output) |
| `GRUCellImpl` | Одна ячейка GRU: 3 гейта (reset, update, new) |
| `RNN` | Многослойный RNN с поддержкой bidirectional |
| `LSTM` | Многослойный LSTM с поддержкой bidirectional |
| `GRU` | Многослойный GRU с поддержкой bidirectional |

Поддержка: `num_layers`, `bidirectional`, `dropout` (между слоями), `batch_first`.

### 6.9 Transformer

Файл: `torch/nn/modules/transformer.h`

| Модуль | Описание |
|--------|---------|
| `PositionalEncoding` | Синусоидальное позиционное кодирование |
| `MultiheadAttention` | Многоголовое внимание (Q, K, V projection + scaled dot-product) |
| `TransformerEncoderLayer` | Один слой энкодера (MHA + FFN + LayerNorm + Dropout) |
| `TransformerEncoder` | Стек энкодерных слоёв |
| `TransformerDecoderLayer` | Один слой декодера (Self-Attn + Cross-Attn + FFN) |
| `TransformerDecoder` | Стек декодерных слоёв |
| `Transformer` | Полная модель Transformer (encoder + decoder) |

Файл: `torch/nn/modules/attention.h` -- отдельная реализация `MultiheadAttention`.

### 6.10 PIR Architecture

Файл: `torch/nn/modules/pir.h`, `torch/nn/modules/pir270m.h`

Собственная архитектура, вдохновлённая LLaMA/Mistral:

| Модуль | Описание |
|--------|---------|
| `RMSNorm` | Root Mean Square нормализация |
| `RotaryEmbedding` (RoPE) | Ротационное позиционное кодирование |
| `SwiGLUFeedForward` | SwiGLU активация в FFN |
| `PIRLayer` | Один слой PIR (RMSNorm + MHA с RoPE + SwiGLU FFN) |
| `PIRBlock` | Блок PIR-слоёв |
| `PIRTransformerBlock` | Полный трансформерный блок |
| `PIR270M` | Модель на 270M параметров |

### 6.11 Embedding

Файл: `torch/nn/modules/sparse.h`

| Модуль | Описание |
|--------|---------|
| `Embedding` | Таблица эмбеддингов (lookup по индексу), padding_idx |
| `EmbeddingBag` | Embedding + свёртка (sum/mean/max) -- для разреженных данных |

### 6.12 Контейнеры

Файл: `torch/nn/modules/container.h`

| Модуль | Описание |
|--------|---------|
| `Sequential` | Последовательная цепочка модулей |
| `ModuleList` | Список модулей (индексация) |
| `ModuleDict` | Словарь модулей (по имени) |
| `ParameterList` | Список параметров |
| `ParameterDict` | Словарь параметров |

### 6.13 Прочие модули

| Модуль | Файл | Описание |
|--------|------|---------|
| `Upsample` | upsampling.h | Билинейная / ближайшая интерполяция |
| `QuantizedLinear` | quantized.h | Квантованный линейный слой |
| `QuantizedConv2d` | quantized.h | Квантованная свёртка |

---

## 7. Оптимизаторы и LR Schedulers

### 7.1 Оптимизаторы

Файлы: `torch/optim/`

Базовый класс `Optimizer` управляет группами параметров с раздельными гиперпараметрами:

```cpp
auto optimizer = torch::optim::Adam(model.parameters(), /*lr=*/0.001);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    optimizer.zero_grad();
    auto loss = criterion(model.forward(input), target);
    loss.backward();
    optimizer.step();
}
```

| Оптимизатор | Файл | Описание |
|-------------|------|---------|
| `SGD` | sgd.h | Стохастический GD с momentum и Nesterov |
| `Adam` | adam.h | Adaptive Moment Estimation (бета-коррекция) |
| `AdamW` | adam.h | Adam с decoupled weight decay |
| `RMSprop` | rmsprop.h | Root Mean Square Propagation |
| `RAdam` | radam.h | Rectified Adam (адаптивный warmup) |
| `NAdam` | nadam.h | Nesterov + Adam |
| `Adamax` | adamax.h | Adam с L-infinity нормой |
| `Adagrad` | adagrad.h | Adaptive gradient (накопление квадратов) |
| `Adadelta` | adadelta.h | Adadelta (без learning rate) |
| `AdamKiller` | adamkiller.h | Экспериментальный оптимизатор |

Все оптимизаторы поддерживают: `param_groups` с раздельными гиперпараметрами, `state_dict()` / `load_state_dict()` для сериализации, `zero_grad()`, `step()`.

### 7.2 LR Schedulers

Файл: `torch/optim/lr_scheduler.h`

| Планировщик | Описание | Формула |
|-------------|---------|---------|
| `StepLR` | Шаговое уменьшение | lr *= gamma каждые step_size эпох |
| `MultiStepLR` | Уменьшение на milestones | lr *= gamma на заданных эпохах |
| `ExponentialLR` | Экспоненциальное уменьшение | lr *= gamma каждую эпоху |
| `CosineAnnealingLR` | Косинусный отжиг | lr = eta_min + 0.5*(lr0-eta_min)*(1+cos(pi*T/Tmax)) |
| `LinearLR` | Линейная интерполяция | Линейно от start_factor до end_factor |
| `ConstantLR` | Постоянный множитель | lr *= factor на первых total_iters |
| `ReduceLROnPlateau` | Уменьшение при стагнации | lr *= factor если метрика не улучшается patience эпох |
| `WarmupCosineAnnealingLR` | Warmup + косинусный отжиг | Линейный warmup + CosineAnnealing |
| `OneCycleLR` | Суперсходимость (Smith) | Косинусный цикл max_lr -> min_lr |

Пример использования:

```cpp
auto optimizer = SGD(model.parameters(), /*lr=*/0.1);
auto scheduler = StepLR(optimizer, /*step_size=*/30, /*gamma=*/0.1);

for (int epoch = 0; epoch < 100; ++epoch) {
    train_one_epoch();
    scheduler.step();
}
```

---

## 8. Data Loading -- загрузка данных

Файлы: `torch/data/`

### 8.1 Dataset

Абстрактный класс `Dataset` определяет интерфейс источника данных:

```cpp
class Dataset {
public:
    virtual std::pair<Tensor, Tensor> get(size_t index) = 0;
    virtual size_t size() = 0;
};
```

Встроенные реализации: `MNISTDataset` (чтение IDX-формата).

### 8.2 Sampler

Стратегии выборки из датасета:

| Sampler | Описание |
|---------|---------|
| `SequentialSampler` | Последовательный порядок |
| `RandomSampler` | Случайный порядок (shuffle) |
| `SubsetRandomSampler` | Случайная подвыборка |

### 8.3 DataLoader

Загрузчик данных с батчированием:

```cpp
auto dataset = MNISTDataset("./data", /*train=*/true);
auto loader = DataLoader(dataset, /*batch_size=*/64, /*shuffle=*/true);

for (auto& [data, target] : loader) {
    // data: [64, 784], target: [64]
}
```

---

## 9. CUDA backend

### 9.1 Собственные CUDA-ядра

Файлы: `aten/src/ATen/cuda/`

| Файл | Содержание |
|------|-----------|
| `CUDAKernels.cu` | Поэлементные операции (add, mul, relu, sigmoid, tanh и др.) |
| `CUDAReduce.cu` | Редукции (sum, max, min, mean, var) |
| `CUDABlas.cu` | GEMM (матричное умножение), GEMV |
| `CUDAConv.cu` | Свёрточные ядра |
| `FlashAttention.cu` | FlashAttention (O(N) по памяти, causal masking) |
| `FlashDecoding.cu` | Flash Decoding для инференса |
| `CUDAQuantGemv.cu` | Квантованный GEMV (INT4/INT8 для LLM инференса) |
| `CUDAInference.cu` | Ядра для инференса LLM |

### 9.2 CUDACachingAllocator

Файл: `c10/cuda/CUDACachingAllocator.h`

Кэширующий аллокатор GPU-памяти:
- Блоки после `cudaFree` не освобождаются, а возвращаются в пул
- При новом запросе -- поиск подходящего блока в пуле
- Это паттерн PyTorch: `cudaFree` при shutdown вызывает проблемы (double free)

### 9.3 cuDNN Integration

Файлы: `aten/src/ATen/cudnn/`

Обёртки над cuDNN для операций с максимальной производительностью:

| Операция | Описание |
|----------|---------|
| `CuDNNConvolution` | Forward и backward свёрток |
| `CuDNNPooling` | Max/Avg pooling |
| `CuDNNBatchNorm` | Batch normalization |
| `CuDNNActivation` | ReLU, Sigmoid, Tanh |

### 9.4 FlashAttention

Файл: `aten/src/ATen/cuda/FlashAttention.cu`

Реализация FlashAttention (Dao et al.):
- O(N) потребление памяти вместо O(N^2)
- Тайловые вычисления в shared memory
- Causal masking для авторегрессионных моделей

**Внимание**: по результатам аудита обнаружены 6 критических багов. Рекомендуется использовать стандартный attention до исправления.

### 9.5 Mixed Precision (AMP)

Файлы: `torch/amp/`

| Компонент | Файл | Описание |
|-----------|------|---------|
| `GradScaler` | grad_scaler.h | Масштабирование градиентов для предотвращения underflow в FP16 |
| `Autocast` | autocast.h | Автоматическое приведение типов (FP32 -> FP16 для GEMM) |

Пример:

```cpp
auto scaler = GradScaler();

for (auto& batch : loader) {
    optimizer.zero_grad();
    {
        auto guard = autocast_guard();
        auto output = model.forward(input);  // FP16
        auto loss = criterion(output, target);
    }
    scaler.scale(loss).backward();
    scaler.step(optimizer);
    scaler.update();
}
```

---

## 10. SIMD-оптимизация (AVX2)

### 10.1 PromeBLAS -- собственная BLAS

Файл: `aten/src/ATen/native/cpu/PromeBLAS.h`

Собственная реализация BLAS-подобных операций с AVX2/FMA:

| Функция | Описание | Реализация |
|---------|---------|-----------|
| `sgemm_avx2` | GEMM (матричное умножение) | Cache-tiled 6x16 AVX2 FMA, L1/L2/L3 blocking |
| `sgemv_avx2` | GEMV (матрица на вектор) | AVX2 с 4-way unroll |
| `sdot_avx2` | Скалярное произведение | AVX2 с 8-way unroll |
| `saxpy_avx2` | y += alpha * x | AVX2 |

Стратегия тайлинга GEMM: блоки MC x KC x NC с размерами, оптимизированными под L1/L2/L3 кэши. Микроядро 6x16 (6 строк A, 16 столбцов B) использует 12 YMM-регистров для аккумулятора.

### 10.2 VectorizedOps -- векторизованные математические функции

Файл: `aten/src/ATen/native/cpu/VectorizedOps.h`

AVX2-реализации трансцендентных функций через полиномиальные аппроксимации (Cephes):

| Функция | Метод аппроксимации |
|---------|-------------------|
| `exp` | Разложение 2^n * poly(r), полином степени 6 |
| `log` | Разложение на мантиссу и экспоненту + полином |
| `sin` / `cos` | Cephes полиномы (Payne-Hanek range reduction) |
| `tanh` | Через exp: (e^2x - 1) / (e^2x + 1) |
| `sigmoid` | Через exp: 1 / (1 + exp(-x)) |

Все функции обрабатывают 8 float за такт (256-битные YMM-регистры).

### 10.3 Результаты оптимизации

Средневзвешенное отношение к PyTorch (50 бенчмарков): **1.75x** (PromeTorch медленнее).

Операции, где PromeTorch **быстрее** PyTorch:

| Операция | Отношение (меньше = лучше) |
|----------|--------------------------|
| sum (reduction) | 0.43x |
| var | 0.14x |
| std | 0.14x |
| argmax | 0.13x |
| dot product | 0.28x |
| mv (matrix-vector) | 0.49x |
| mm 2048x2048 | 0.97x |
| tanh | 0.48x |

Операции, где PyTorch быстрее (из-за оверхеда аллокации тензоров):
- Простые поэлементные операции: 8-11x
- LSTM: 19x
- Autograd overhead: 6.6x

---

## 11. Serialization -- сериализация

Файл: `torch/serialization.h`

### Формат PTOR (PromeTorch Object Record)

Бинарный формат для сохранения и загрузки тензоров и state_dict:

```
[Magic: "PTOR" (4 bytes)]
[Version: uint32]
[Num tensors: uint32]
For each tensor:
  [Name length: uint32]
  [Name: string]
  [Num dims: uint32]
  [Sizes: int64 x num_dims]
  [ScalarType: uint8]
  [Data: nbytes]
```

API:

```cpp
// Сохранение отдельного тензора
torch::save(tensor, "model.pt");
auto loaded = torch::load("model.pt");

// Сохранение state_dict модели
torch::save(model.state_dict(), "model_weights.pt");

// Загрузка state_dict
auto state = torch::load_state_dict("model_weights.pt");
model.load_state_dict(state);
```

---

## 12. Channels-last Memory Format

Файлы: `c10/core/TensorImpl.h`, `aten/src/ATen/core/Tensor.h`, `aten/src/ATen/native/cpu/ShapeOps.h`

### Что такое Channels-last

Стандартный формат памяти для 4D-тензоров (N, C, H, W) -- **Contiguous** (NCHW): элементы хранятся по каналам, затем по строкам.

**Channels-last** (NHWC): элементы хранятся по пикселям, затем по каналам. Этот формат оптимален для cuDNN и многих CPU-операций (лучшая локальность кэша при доступе ко всем каналам одного пикселя).

```cpp
auto x = torch::randn({1, 3, 224, 224});
auto x_nhwc = x.contiguous(c10::MemoryFormat::ChannelsLast);

// Strides:
// NCHW: [3*224*224, 224*224, 224, 1]
// NHWC: [3*224*224, 1, 224*3, 3]
```

Реализация: `MemoryFormat` enum в TensorImpl, `contiguous(MemoryFormat)` в Tensor и ShapeOps, `is_channels_last_contiguous()` в TensorImpl.

---

## 13. GGUF Inference -- инференс LLM

PromeTorch включает собственный runtime для инференса квантованных LLM в формате GGUF (совместимом с llama.cpp/Ollama).

### Возможности

- Загрузка моделей GGUF (Q4_0, Q4_1, Q8_0, Q5_K, Q6_K и другие квантования)
- Квантованный GEMV на GPU (`CUDAQuantGemv.cu`) с warp-cooperative доступом
- Speculative decoding
- KV-cache
- Поддержка моделей: Qwen3, Gemma3, DeepSeek-R1, LLaMA и др.

### Производительность

| Модель | Токенов/сек (PromeTorch) | Tokенов/сек (Ollama) | VRAM |
|--------|--------------------------|---------------------|------|
| qwen3:4b | 49.9 | 164.6 | 4.9 GB |
| gemma3:4b | 52.9 | -- | -- |
| deepseek-r1:8b | 30.5 | -- | -- |
| qwen3:14b | 18.4 | -- | 9.6 GB |

### Оптимизации CUDA GEMV

- Warp-cooperative coalesced memory access
- Shared memory для x-вектора
- `float4` packed loads
- `uint32_t` packed quantization states
- `cudaFuncSetAttribute` для smem > 48KB (ffn_down в 14B+ моделях)

---

## 14. Python bindings

Файлы: `python/`

Python-интерфейс через pybind11. Пакет `promethorch` (или `_C` как модуль расширения).

### Поддерживаемый API

```python
import promethorch as pt

# Создание тензоров
x = pt.randn(3, 4)
y = pt.zeros(2, 3)
z = pt.mm(x, y.t())

# Нейросетевые модули
linear = pt.Linear(784, 128)
output = linear(x)

# Autograd
x = pt.randn(3, 3, requires_grad=True)
y = pt.mm(x, x)
y.backward()
print(x.grad)
```

### Собранная DLL

Результат сборки: `_C.so` (Linux/Эльбрус) или `_C.pyd` (Windows). Подтверждено: `randn`, `zeros`, `mm`, `Linear` работают на Эльбрус (LCC 1.29).

### Известные ограничения

- `no_grad()` в Python не подключён к C++ engine (BUG-C9 из аудита)

---

## 15. Сборка

### 15.1 Требования

| Компонент | Версия |
|-----------|--------|
| ОС | Windows 10 (основная), Linux (Эльбрус) |
| Компилятор | MSVC 2019 (Windows), LCC 1.29 (Эльбрус) |
| CMake | 3.16+ |
| CUDA | 12.9.86 (опционально) |
| cuDNN | 9.14.0 (опционально) |
| Python | 3.12 (Anaconda) |
| pybind11 | Для Python-биндингов |

### 15.2 Сборка CPU (Windows)

Из Developer Command Prompt for VS 2019:

```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\ProgramData\anaconda3\Library\bin;%PATH%

mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

### 15.3 Сборка с CUDA

```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set CUDA_PATH=C:\ProgramData\anaconda3\Library

mkdir build_cuda && cd build_cuda
cmake .. -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPT_USE_CUDA=ON ^
    -DPT_USE_CUDNN=ON ^
    -DPT_BUILD_TESTS=OFF ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%"
nmake
```

### 15.4 Сборка с NM Card

```batch
cmake .. -G "NMake Makefiles" -DPT_USE_NMCARD=ON
nmake
```

### 15.5 Запуск примеров

```bash
cd /c/Users/paper/Desktop/promethorch
PATH="./build_final3:$PATH" ./build_final3/examples/mnist/train_mnist_mlp.exe --device cpu --epochs 5 --lr 0.001
```

### 15.6 Существующие рабочие сборки

| Директория | Описание |
|-----------|---------|
| `build_final3/` | CPU, MNIST MLP (новейшая) |
| `build_examples/` | CPU, MNIST MLP |
| `build_cudnn/` | CUDA + cuDNN, PIR |
| `build_nmcard/` | NM Card эмулятор, MNIST |

### 15.7 Важные замечания

- Сборка из bash (Cygwin/Git Bash) **не работает**: `rc.exe` (Windows Resource Compiler) не находится. Используйте Developer Command Prompt или batch-файлы.
- CMake из Anaconda **не использовать** -- нужен CMake из Visual Studio.
- nvcc + MSVC: флаги компилятора C++ должны оборачиваться в `$<$<COMPILE_LANGUAGE:CXX>:...>` в CMake, чтобы nvcc не получал несовместимые флаги.

---

## 16. Результаты и бенчмарки

### 16.1 Обучение моделей

Все 10 тестовых моделей обучены и показывают результаты, соответствующие baseline PyTorch:

| Модель | Accuracy | Примечание |
|--------|----------|-----------|
| MNIST MLP | 97.65% | 3-layer MLP, SGD |
| LSTM | 98.44% | Sequence classification |
| GRU | 95.3% | Sequence classification |
| MNIST NMCard | 93.64% | На эмуляторе NM Card, 3 epochs, SGD lr=0.01 |

### 16.2 CPU бенчмарки vs PyTorch

Средневзвешенное: PromeTorch в **1.75x** медленнее PyTorch (50 бенчмарков).

PromeTorch **быстрее** PyTorch на 15 из 50 бенчмарков, включая:
- Редукции (sum, var, std, argmax) -- в 2-7x быстрее
- Скалярное произведение (dot) -- в 3.5x быстрее
- Матрица-вектор (mv) -- в 2x быстрее
- Матричное умножение 2048x2048 -- на уровне PyTorch (0.97x)
- tanh -- в 2x быстрее

### 16.3 GGUF инференс

На GPU: 49.9 tok/s для qwen3:4b (Ollama: 164.6 tok/s). Gap ~3.3x, основные причины: отсутствие cuBLAS для decode, overhead аллокаций.

---

## 17. Дополнительные backend-ы

### 17.1 NM Card Mini

Файлы: `c10/nmcard/`, `aten/src/ATen/nmcard/`

Третий backend после CPU и CUDA. Реализован как `PrivateUse1` в системе устройств.

**NM Card Mini** -- нейропроцессорная плата на базе чипа NM6408 (partner, Россия). Особенности:
- Арифметика фиксированной точки Q16.16
- Программный эмулятор для разработки без реального железа
- 32/32 юнит-тестов проходят
- MNIST 93.64% на эмуляторе

Файлы:
- `NMCardEmulator` -- программный эмулятор логики процессора
- `NMCardOps` -- реализация тензорных операций
- `NMCardDispatch` -- диспетчер операций
- `NMCardMath` -- Q16.16 арифметика

### 17.2 NM Quad

Зарезервирован как `PrivateUse3`. NM Quad -- 4 чипа NM6408, 20GB DDR, 2 TFLOPS FP32, 64 ядра, PCIe x16, 80W.

Файлы: `c10/nmquad/`, `aten/src/ATen/nmquad/`, `docs/nmquad/`

### 17.3 Эльбрус

Python-биндинги (_C.so) собраны и работают на Эльбрус E8C2 (32 ядра, LCC 1.29). Подтверждены операции: `randn`, `zeros`, `mm`, `Linear`.

NUMA scaling: 4x node-local EML = 1840 GFLOPS (92% пика). NUMA-aware = 5.7x ускорение.

---

## 18. Известные ограничения

| Проблема | Статус | Описание |
|----------|--------|---------|
| FlashAttention | 6 багов | Полностью нерабочий, не использовать |
| Conv3d | Stub | Forward возвращает нули |
| Python no_grad() | Баг | Не подключён к C++ engine |
| Поэлементные операции | Медленнее PyTorch | 8-11x из-за оверхеда аллокации тензоров |
| LSTM/GRU | Медленнее PyTorch | 19x из-за отсутствия CUDA-фьюзинга ячеек |
| Distributed Training | Не реализовано | DDP/NCCL отсутствуют |
| JIT/TorchScript | Не реализовано | Нет компиляции вычислительного графа |

---

## 19. Планы развития

| Фаза | Компонент | Описание |
|------|-----------|---------|
| 16 | Distributed Training | DDP, NCCL, multi-GPU |
| 17 | TorchScript/JIT | Компиляция и оптимизация графа |
| 18 | Дополнительные операции | einsum (расширение), scatter_reduce |
| 19 | Quantization | INT8 квантование для инференса |
| 20 | ONNX export | Экспорт моделей в формат ONNX |
| 21 | Profiling tools | Инструменты профилирования |

---

## Приложение A: Решённые архитектурные проблемы

Документация критических проблем, решённых в процессе разработки:

### A.1 DLL Singleton (heap corruption)

**Проблема**: `inline static` переменная в header-файле создаёт отдельный экземпляр в каждой DLL. CUDACachingAllocator имел два экземпляра: один в основном .exe, другой в _C.pyd. Тензор, выделенный одним экземпляром, освобождался другим -> heap corruption.

**Решение**: singleton вынесен в `.cpp` файл, компилируемый в одну единицу трансляции.

### A.2 mm() + non-contiguous tensors

**Проблема**: `mm()` читала данные через `data_ptr<float>()` с контигуозными индексами `[i*K+k]`, но `tensor.t()` создаёт view с транспонированными strides. Данные в памяти НЕ в row-major порядке.

**Симптомы**: gradient check "проходил" (numerical и analytical оба были неверны одинаково), single step "работал", но multi-batch training не сходился (accuracy ~15%).

**Решение**: `.contiguous()` перед вычислениями во всех функциях линейной алгебры (mm, mv, bmm, dot, outer, addmm).

### A.3 copy_() и strided tensors

**Проблема**: `copy_()` использовал последовательный доступ `dst[i] = src[i]`, что некорректно для non-contiguous views (например, после `narrow()`).

**Решение**: strided copy path с multi-dimensional index -> physical offset mapping.

### A.4 Unary ops и non-contiguous tensors

**Проблема**: макрос `DEFINE_UNARY_OP` (sigmoid, tanh и др.) не вызывал `.contiguous()` на входе. Narrow views имеют strides `[4*H, 1]` вместо `[H, 1]`.

**Решение**: `.contiguous()` на входе всех унарных операций.

### A.5 CUDA shutdown (double free)

**Проблема**: `cudaFree()` при завершении программы вызывал double free.

**Решение**: не вызывать `cudaFree()` при завершении (паттерн PyTorch -- OS освобождает память процесса).

### A.6 NMCard DLL Allocator

**Проблема**: `AllocatorRegistry` с `inline static` -- каждая DLL получала свою копию реестра.

**Решение**: двойная регистрация: `register_nmcard_allocator()` (в DLL) + `register_nmcard_allocator_local()` (inline в caller).

---

*PromeTorch (c) 2026. Все компоненты разработаны с нуля без использования кода сторонних фреймворков.*
