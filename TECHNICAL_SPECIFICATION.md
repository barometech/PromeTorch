# PromeTorch - Техническое Задание
## Фреймворк глубокого обучения с нуля

**Версия документа:** 1.0
**Дата:** 2026-01-20

---

## 1. ОБЗОР ПРОЕКТА

### 1.1 Цель
Создание полнофункционального фреймворка глубокого обучения (аналог PyTorch) с нуля, включающего:
- Тензорную библиотеку с поддержкой CPU и GPU
- Систему автоматического дифференцирования (autograd)
- Модульную систему нейронных сетей
- Оптимизаторы
- Загрузчики данных
- Полный набор математических операций (кернелов)

### 1.2 Технологический стек
- **Ядро (Core):** C++17/20
- **GPU вычисления:** CUDA, cuBLAS, cuDNN
- **CPU оптимизации:** OpenBLAS, MKL, AVX/AVX2/AVX-512
- **Python биндинги:** pybind11
- **Сборка:** CMake
- **Тестирование:** Google Test (C++), pytest (Python)

### 1.3 Название проекта
**PromeTorch** (Prometheus + Torch)

---

## 2. АРХИТЕКТУРА СИСТЕМЫ

### 2.1 Структура директорий
```
promethorch/
├── c10/                    # Core library (базовые абстракции)
│   ├── core/
│   │   ├── TensorImpl.h    # Низкоуровневое представление тензора
│   │   ├── Storage.h       # Управление памятью
│   │   ├── Allocator.h     # Аллокаторы памяти
│   │   ├── Device.h        # Абстракция устройства (CPU/CUDA)
│   │   ├── DeviceType.h    # Типы устройств
│   │   ├── ScalarType.h    # Типы данных (float32, float64, int32...)
│   │   ├── DispatchKey.h   # Ключи диспетчеризации
│   │   └── Stream.h        # CUDA streams
│   ├── util/
│   │   ├── Exception.h     # Обработка исключений
│   │   ├── Logging.h       # Логирование
│   │   └── SmallVector.h   # Оптимизированные контейнеры
│   └── macros/
│       └── Macros.h        # Платформенные макросы
│
├── aten/                   # A Tensor Library (операции над тензорами)
│   ├── src/
│   │   └── ATen/
│   │       ├── core/
│   │       │   ├── Tensor.h          # Основной класс Tensor
│   │       │   ├── TensorAccessor.h  # Доступ к данным
│   │       │   └── Generator.h       # Генератор случайных чисел
│   │       ├── native/
│   │       │   ├── cpu/              # CPU кернелы
│   │       │   ├── cuda/             # CUDA кернелы
│   │       │   ├── Math.cpp          # Математические операции
│   │       │   ├── Linear.cpp        # Линейная алгебра
│   │       │   ├── Convolution.cpp   # Свёртки
│   │       │   ├── Pooling.cpp       # Пулинг
│   │       │   ├── Activation.cpp    # Функции активации
│   │       │   ├── Loss.cpp          # Функции потерь
│   │       │   ├── Normalization.cpp # Нормализация
│   │       │   └── ...
│   │       ├── ops/
│   │       │   └── declarations.yaml # Описание всех операций
│   │       └── Dispatch.h            # Диспетчер операций
│   └── CMakeLists.txt
│
├── torch/                  # Высокоуровневый API
│   ├── csrc/               # C++ frontend
│   │   ├── autograd/
│   │   │   ├── Engine.h              # Движок autograd
│   │   │   ├── Function.h            # Базовый класс функций
│   │   │   ├── Variable.h            # Переменная с градиентом
│   │   │   ├── GradMode.h            # Режим градиента
│   │   │   └── generated/            # Сгенерированные backward функции
│   │   ├── api/
│   │   │   ├── nn/                   # Модули нейросетей
│   │   │   │   ├── Module.h
│   │   │   │   ├── Linear.h
│   │   │   │   ├── Conv.h
│   │   │   │   ├── RNN.h
│   │   │   │   ├── Transformer.h
│   │   │   │   └── ...
│   │   │   ├── optim/                # Оптимизаторы
│   │   │   │   ├── Optimizer.h
│   │   │   │   ├── SGD.h
│   │   │   │   ├── Adam.h
│   │   │   │   └── ...
│   │   │   └── data/                 # Загрузка данных
│   │   │       ├── DataLoader.h
│   │   │       ├── Dataset.h
│   │   │       └── Sampler.h
│   │   └── cuda/
│   │       ├── CUDAContext.h
│   │       ├── CUDAAllocator.h       # Caching allocator
│   │       └── CUDAStream.h
│   └── python/             # Python биндинги
│       ├── __init__.py
│       ├── _C/             # pybind11 модуль
│       ├── tensor.py
│       ├── nn/
│       ├── optim/
│       └── autograd/
│
├── tools/                  # Инструменты кодогенерации
│   ├── codegen/
│   │   ├── gen.py
│   │   └── templates/
│   └── autograd/
│       └── derivatives.yaml
│
├── test/                   # Тесты
│   ├── cpp/
│   └── python/
│
├── third_party/            # Внешние зависимости
│   ├── pybind11/
│   ├── googletest/
│   └── ...
│
├── CMakeLists.txt
├── setup.py
└── requirements.txt
```

---

## 3. МОДУЛЬ C10 (CORE)

### 3.1 TensorImpl - Низкоуровневое представление тензора

```cpp
// c10/core/TensorImpl.h
class TensorImpl {
public:
    // Хранилище данных
    Storage storage_;

    // Метаданные
    int64_t storage_offset_;           // Смещение в storage
    SmallVector<int64_t, 5> sizes_;    // Размеры
    SmallVector<int64_t, 5> strides_;  // Шаги
    int64_t numel_;                    // Количество элементов

    // Тип данных и устройство
    ScalarType dtype_;
    Device device_;

    // Флаги
    bool requires_grad_;
    bool is_contiguous_;

    // Autograd
    AutogradMeta* autograd_meta_;
};
```

### 3.2 Storage - Управление памятью

```cpp
// c10/core/Storage.h
class Storage {
public:
    DataPtr data_ptr_;           // Указатель на данные
    int64_t nbytes_;             // Размер в байтах
    Allocator* allocator_;       // Аллокатор
    bool resizable_;
};
```

### 3.3 ScalarType - Поддерживаемые типы данных

| Тип | Размер | Описание |
|-----|--------|----------|
| `Float16` | 2 bytes | Half precision |
| `BFloat16` | 2 bytes | Brain floating point |
| `Float32` | 4 bytes | Single precision |
| `Float64` | 8 bytes | Double precision |
| `Int8` | 1 byte | Signed 8-bit |
| `UInt8` | 1 byte | Unsigned 8-bit |
| `Int16` | 2 bytes | Signed 16-bit |
| `Int32` | 4 bytes | Signed 32-bit |
| `Int64` | 8 bytes | Signed 64-bit |
| `Bool` | 1 byte | Boolean |
| `ComplexFloat` | 8 bytes | Complex float32 |
| `ComplexDouble` | 16 bytes | Complex float64 |

### 3.4 Device - Абстракция устройства

```cpp
// c10/core/Device.h
enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
    // Расширяемо для будущих бэкендов
};

class Device {
    DeviceType type_;
    int8_t index_;  // Индекс GPU (для multi-GPU)
};
```

### 3.5 Allocator - Аллокаторы памяти

```cpp
// c10/core/Allocator.h
class Allocator {
public:
    virtual DataPtr allocate(size_t n) = 0;
    virtual void deallocate(DataPtr ptr) = 0;
};

// Реализации:
class CPUAllocator : public Allocator;
class CUDACachingAllocator : public Allocator;  // С кэшированием
```

---

## 4. МОДУЛЬ ATEN (ОПЕРАЦИИ)

### 4.1 Tensor - Основной класс

```cpp
// aten/src/ATen/core/Tensor.h
class Tensor {
public:
    // Конструкторы
    Tensor();
    Tensor(TensorImpl* impl);

    // Доступ к данным
    template<typename T>
    T* data_ptr() const;

    // Метаданные
    IntArrayRef sizes() const;
    IntArrayRef strides() const;
    int64_t numel() const;
    int64_t dim() const;
    ScalarType dtype() const;
    Device device() const;

    // Флаги
    bool requires_grad() const;
    void set_requires_grad(bool);
    bool is_contiguous() const;

    // Операции (см. раздел 4.2)
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    // ... 1200+ операций

private:
    intrusive_ptr<TensorImpl> impl_;
};
```

### 4.2 ПОЛНЫЙ СПИСОК КЕРНЕЛОВ (ОПЕРАЦИЙ)

#### 4.2.1 Создание тензоров (Factory Functions)
```
zeros, ones, empty, full, eye
arange, linspace, logspace
rand, randn, randint, randperm
tensor (from data), from_numpy
zeros_like, ones_like, empty_like, full_like, rand_like, randn_like
scalar_tensor
complex, polar
```

#### 4.2.2 Индексация и срезы (Indexing & Slicing)
```
index, index_put, index_put_
index_select, masked_select, masked_fill, masked_scatter
gather, scatter, scatter_add, scatter_reduce
take, take_along_dim
nonzero, where
narrow, narrow_copy
select, slice
diagonal, diag, diag_embed, diagflat
tril, triu, tril_indices, triu_indices
```

#### 4.2.3 Изменение формы (Shape Operations)
```
view, reshape, reshape_as
squeeze, unsqueeze
flatten, unflatten
transpose, permute, movedim, moveaxis
expand, expand_as, broadcast_to
repeat, repeat_interleave, tile
chunk, split, tensor_split, hsplit, vsplit, dsplit
cat, concat, concatenate, stack, hstack, vstack, dstack
column_stack, row_stack
unbind
contiguous
clone
detach
```

#### 4.2.4 Унарные математические операции (Unary Math)
```
# Базовые
abs, neg, sign, sgn
ceil, floor, round, trunc, frac
sqrt, rsqrt, square
exp, exp2, expm1
log, log2, log10, log1p
reciprocal

# Тригонометрические
sin, cos, tan
asin, acos, atan
sinh, cosh, tanh
asinh, acosh, atanh
sinc

# Специальные функции
erf, erfc, erfinv
lgamma, digamma, polygamma
i0, i0e, i1, i1e
sigmoid, logit
softplus, mish, silu, gelu

# Битовые операции
bitwise_not
```

#### 4.2.5 Бинарные математические операции (Binary Math)
```
add, sub, mul, div, true_divide, floor_divide
pow, fmod, remainder
atan2, hypot
maximum, minimum, fmax, fmin
gcd, lcm
xlogy, xlog1py
nextafter
copysign
ldexp, frexp

# Битовые операции
bitwise_and, bitwise_or, bitwise_xor
bitwise_left_shift, bitwise_right_shift

# Сравнения
eq, ne, lt, le, gt, ge
isclose, allclose
```

#### 4.2.6 Операции редукции (Reduction Operations)
```
sum, prod, mean, std, var
min, max, amin, amax
argmin, argmax
all, any
count_nonzero
nansum, nanmean
median, nanmedian, quantile, nanquantile
mode
logsumexp
norm, vector_norm, matrix_norm
dist
cumsum, cumprod, cummax, cummin
logcumsumexp
diff
gradient
```

#### 4.2.7 Линейная алгебра (Linear Algebra)
```
# Базовые операции
dot, vdot
mv, mm, bmm, matmul
addmv, addmm, addbmm
baddbmm
outer, inner, tensordot
einsum

# Разложения
cholesky, cholesky_inverse, cholesky_solve
lu, lu_factor, lu_solve, lu_unpack
qr
svd, svdvals
eig, eigh, eigvals, eigvalsh
schur

# Решение систем
solve, lstsq, triangular_solve

# Матричные операции
det, logdet, slogdet
matrix_rank, matrix_power, matrix_exp
inv, pinv
trace
cross

# Нормы
norm (векторная, матричная, Frobenius)
cond

# Специальные
kron (Kronecker product)
vander (Vandermonde matrix)
```

#### 4.2.8 Свёртки (Convolution)
```
# 1D свёртки
conv1d
conv_transpose1d

# 2D свёртки
conv2d
conv_transpose2d

# 3D свёртки
conv3d
conv_transpose3d

# Вспомогательные
unfold, fold
im2col, col2im
```

#### 4.2.9 Пулинг (Pooling)
```
# Max pooling
max_pool1d, max_pool2d, max_pool3d
max_unpool1d, max_unpool2d, max_unpool3d
adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d

# Average pooling
avg_pool1d, avg_pool2d, avg_pool3d
adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d

# Lp pooling
lp_pool1d, lp_pool2d

# Fractional pooling
fractional_max_pool2d, fractional_max_pool3d
```

#### 4.2.10 Функции активации (Activation Functions)
```
relu, relu6, leaky_relu, prelu, rrelu
elu, selu, celu, gelu
sigmoid, hardsigmoid
tanh, hardtanh
softplus, softsign, softshrink, hardshrink
tanhshrink
threshold
mish, silu (swish)
logsigmoid
softmax, log_softmax
gumbel_softmax
```

#### 4.2.11 Функции потерь (Loss Functions)
```
# Классификация
cross_entropy, nll_loss
binary_cross_entropy, binary_cross_entropy_with_logits
hinge_embedding_loss
multi_margin_loss, multilabel_margin_loss
soft_margin_loss
multilabel_soft_margin_loss

# Регрессия
mse_loss, l1_loss, smooth_l1_loss, huber_loss

# Ранжирование
margin_ranking_loss
triplet_margin_loss, triplet_margin_with_distance_loss

# Специальные
cosine_embedding_loss
ctc_loss
poisson_nll_loss
gaussian_nll_loss
kl_div
```

#### 4.2.12 Нормализация (Normalization)
```
batch_norm
layer_norm
instance_norm
group_norm
local_response_norm
normalize (L2 normalization)
```

#### 4.2.13 Dropout
```
dropout, dropout2d, dropout3d
alpha_dropout
feature_alpha_dropout
```

#### 4.2.14 Sparse операции
```
to_sparse, to_sparse_csr, to_sparse_csc
sparse_coo_tensor, sparse_csr_tensor
coalesce
sparse_mask
```

#### 4.2.15 RNN операции
```
rnn_tanh, rnn_relu
lstm, gru
rnn_tanh_cell, rnn_relu_cell
lstm_cell, gru_cell
```

#### 4.2.16 Attention операции
```
scaled_dot_product_attention
multi_head_attention_forward
```

#### 4.2.17 Grid операции (для spatial transformers)
```
affine_grid
grid_sample
```

#### 4.2.18 Операции с комплексными числами
```
real, imag
angle, abs (complex)
conj, conj_physical
view_as_real, view_as_complex
polar
```

#### 4.2.19 Сортировка и поиск
```
sort, argsort
topk
kthvalue
msort (stable sort)
searchsorted
bucketize
unique, unique_consecutive
```

#### 4.2.20 FFT операции
```
fft, ifft
fft2, ifft2
fftn, ifftn
rfft, irfft
rfft2, irfft2
rfftn, irfftn
hfft, ihfft
fftshift, ifftshift
fftfreq, rfftfreq
```

#### 4.2.21 Операции с окнами (Window Functions)
```
bartlett_window
blackman_window
hamming_window
hann_window
kaiser_window
```

#### 4.2.22 Специальные функции (torch.special)
```
# Bessel functions
bessel_j0, bessel_j1, bessel_y0, bessel_y1

# Gamma functions
gammaln, gammainc, gammaincc
digamma, polygamma, multigammaln

# Error functions
erf, erfc, erfinv, erfcx

# Exponential integrals
expit, logit
exp2, expm1
log1p, log_softmax

# Другие
sinc
entr (entropy)
xlog1py, xlogy
zeta (Riemann zeta)
i0, i0e, i1, i1e
ndtr, ndtri
spherical_bessel_j0
```

---

## 5. МОДУЛЬ AUTOGRAD

### 5.1 Архитектура

```
Forward Pass:
  input → op1 → op2 → ... → output
            ↓     ↓
    (записываем backward nodes в граф)

Backward Pass:
  grad_output → backward_op_n → ... → backward_op_1 → grad_input
```

### 5.2 Основные компоненты

#### 5.2.1 Variable (Tensor с градиентом)
```cpp
// torch/csrc/autograd/Variable.h
struct AutogradMeta {
    Tensor grad_;                    // Накопленный градиент
    shared_ptr<Node> grad_fn_;       // Функция для backward
    uint32_t output_nr_;             // Номер выхода в grad_fn_
    bool requires_grad_;
    bool retains_grad_;
    bool is_leaf_;
};
```

#### 5.2.2 Node (Function)
```cpp
// torch/csrc/autograd/Function.h
class Node {
public:
    virtual variable_list apply(variable_list&& inputs) = 0;

    edge_list next_edges_;           // Связи с предыдущими узлами
    uint64_t sequence_nr_;           // Для топологической сортировки
};

// Примеры реализаций:
class AddBackward : public Node;
class MulBackward : public Node;
class MatmulBackward : public Node;
// ... для каждой операции
```

#### 5.2.3 Edge
```cpp
struct Edge {
    shared_ptr<Node> function_;      // Узел
    uint32_t input_nr_;              // Номер входа
};
```

#### 5.2.4 Engine
```cpp
// torch/csrc/autograd/Engine.h
class Engine {
public:
    void execute(
        const edge_list& roots,       // Стартовые узлы
        const variable_list& inputs,  // grad_outputs
        bool keep_graph,
        bool create_graph
    );

private:
    // Топологическая сортировка
    // Многопоточное выполнение
    // Обработка зависимостей
};
```

### 5.3 Производные (derivatives.yaml)
```yaml
# tools/autograd/derivatives.yaml

- name: add(Tensor self, Tensor other, *, Scalar alpha=1)
  self: grad
  other: maybe_multiply(grad, alpha)

- name: mul(Tensor self, Tensor other)
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())

- name: matmul(Tensor self, Tensor other)
  self: matmul_backward(grad, self, other, 0)
  other: matmul_backward(grad, self, other, 1)

- name: relu(Tensor self)
  self: threshold_backward(grad, self, 0)

- name: sigmoid(Tensor self)
  self: sigmoid_backward(grad, result)

- name: tanh(Tensor self)
  self: tanh_backward(grad, result)

# ... для всех дифференцируемых операций
```

---

## 6. МОДУЛЬ NN (NEURAL NETWORKS)

### 6.1 Module - Базовый класс

```cpp
// torch/csrc/api/nn/Module.h
class Module {
public:
    virtual ~Module() = default;

    // Обязательный для переопределения
    virtual Tensor forward(Tensor input) = 0;

    // Управление параметрами
    void register_parameter(string name, Tensor param);
    void register_buffer(string name, Tensor buffer);
    void register_module(string name, shared_ptr<Module> module);

    // Доступ к параметрам
    vector<Tensor> parameters(bool recurse = true);
    vector<pair<string, Tensor>> named_parameters(bool recurse = true);

    // Режимы работы
    void train(bool mode = true);
    void eval();
    bool is_training() const;

    // Перенос на устройство
    void to(Device device);
    void to(ScalarType dtype);

    // Хуки
    void register_forward_hook(function<Tensor(Module*, Tensor, Tensor)> hook);
    void register_backward_hook(function<Tensor(Module*, Tensor, Tensor)> hook);
    void register_forward_pre_hook(function<Tensor(Module*, Tensor)> hook);

    // Сохранение/загрузка
    void save(const string& path);
    void load(const string& path);

protected:
    OrderedDict<string, Tensor> parameters_;
    OrderedDict<string, Tensor> buffers_;
    OrderedDict<string, shared_ptr<Module>> modules_;
    bool training_ = true;
};
```

### 6.2 Слои (Layers)

#### 6.2.1 Линейные слои
```cpp
class Linear : public Module {
    int64_t in_features, out_features;
    Tensor weight, bias;
};

class Bilinear : public Module;
class LazyLinear : public Module;
class Identity : public Module;
```

#### 6.2.2 Свёрточные слои
```cpp
class Conv1d : public Module {
    int64_t in_channels, out_channels;
    array<int64_t, 1> kernel_size;
    array<int64_t, 1> stride, padding, dilation;
    int64_t groups;
    Tensor weight, bias;
};

class Conv2d : public Module;
class Conv3d : public Module;
class ConvTranspose1d : public Module;
class ConvTranspose2d : public Module;
class ConvTranspose3d : public Module;
class LazyConv1d, LazyConv2d, LazyConv3d : public Module;
```

#### 6.2.3 Пулинг слои
```cpp
class MaxPool1d, MaxPool2d, MaxPool3d : public Module;
class AvgPool1d, AvgPool2d, AvgPool3d : public Module;
class AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d : public Module;
class AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d : public Module;
class MaxUnpool1d, MaxUnpool2d, MaxUnpool3d : public Module;
class FractionalMaxPool2d, FractionalMaxPool3d : public Module;
class LPPool1d, LPPool2d : public Module;
```

#### 6.2.4 Слои нормализации
```cpp
class BatchNorm1d, BatchNorm2d, BatchNorm3d : public Module {
    int64_t num_features;
    double eps, momentum;
    bool affine, track_running_stats;
    Tensor weight, bias;
    Tensor running_mean, running_var;
};

class LayerNorm : public Module;
class GroupNorm : public Module;
class InstanceNorm1d, InstanceNorm2d, InstanceNorm3d : public Module;
class LocalResponseNorm : public Module;
```

#### 6.2.5 Dropout слои
```cpp
class Dropout : public Module;
class Dropout2d : public Module;
class Dropout3d : public Module;
class AlphaDropout : public Module;
class FeatureAlphaDropout : public Module;
```

#### 6.2.6 RNN слои
```cpp
class RNN : public Module;
class LSTM : public Module;
class GRU : public Module;
class RNNCell : public Module;
class LSTMCell : public Module;
class GRUCell : public Module;
```

#### 6.2.7 Transformer слои
```cpp
class Transformer : public Module;
class TransformerEncoder : public Module;
class TransformerDecoder : public Module;
class TransformerEncoderLayer : public Module;
class TransformerDecoderLayer : public Module;
class MultiheadAttention : public Module;
```

#### 6.2.8 Embedding слои
```cpp
class Embedding : public Module;
class EmbeddingBag : public Module;
```

#### 6.2.9 Функции активации (как модули)
```cpp
class ReLU : public Module;
class ReLU6 : public Module;
class LeakyReLU : public Module;
class PReLU : public Module;
class ELU : public Module;
class SELU : public Module;
class GELU : public Module;
class Sigmoid : public Module;
class Tanh : public Module;
class Softmax : public Module;
class LogSoftmax : public Module;
class Softplus : public Module;
class Mish : public Module;
class SiLU : public Module;
class Hardtanh : public Module;
class Hardswish : public Module;
class Hardsigmoid : public Module;
```

#### 6.2.10 Контейнеры
```cpp
class Sequential : public Module;
class ModuleList : public Module;
class ModuleDict : public Module;
class ParameterList : public Module;
class ParameterDict : public Module;
```

#### 6.2.11 Утилитарные слои
```cpp
class Flatten : public Module;
class Unflatten : public Module;
class Upsample : public Module;
class UpsamplingNearest2d : public Module;
class UpsamplingBilinear2d : public Module;
class PixelShuffle : public Module;
class PixelUnshuffle : public Module;
class ChannelShuffle : public Module;
class Fold : public Module;
class Unfold : public Module;
```

---

## 7. МОДУЛЬ OPTIM (ОПТИМИЗАТОРЫ)

### 7.1 Базовый класс

```cpp
// torch/csrc/api/optim/Optimizer.h
class Optimizer {
public:
    Optimizer(vector<Tensor> params, OptimizerOptions options);

    virtual void step() = 0;           // Шаг оптимизации
    void zero_grad();                  // Обнуление градиентов

    // Состояние
    void add_param_group(ParamGroup group);
    vector<ParamGroup>& param_groups();

    // Сохранение/загрузка
    void state_dict();
    void load_state_dict(StateDict dict);

protected:
    vector<ParamGroup> param_groups_;
    unordered_map<Tensor, OptimizerState> state_;
};
```

### 7.2 Реализации оптимизаторов

```cpp
// Базовые
class SGD : public Optimizer;           // Stochastic Gradient Descent
class Adam : public Optimizer;          // Adaptive Moment Estimation
class AdamW : public Optimizer;         // Adam with decoupled Weight decay
class Adagrad : public Optimizer;       // Adaptive Gradient
class RMSprop : public Optimizer;       // Root Mean Square Propagation
class Adadelta : public Optimizer;      // Adaptive Delta

// Продвинутые
class Adamax : public Optimizer;
class NAdam : public Optimizer;         // Nesterov Adam
class RAdam : public Optimizer;         // Rectified Adam
class SparseAdam : public Optimizer;
class ASGD : public Optimizer;          // Averaged SGD
class Rprop : public Optimizer;         // Resilient Propagation
class LBFGS : public Optimizer;         // Limited-memory BFGS
```

### 7.3 Learning Rate Schedulers

```cpp
class LRScheduler {
public:
    virtual void step(int epoch = -1) = 0;
    double get_last_lr();
};

// Реализации
class StepLR : public LRScheduler;
class MultiStepLR : public LRScheduler;
class ExponentialLR : public LRScheduler;
class CosineAnnealingLR : public LRScheduler;
class CosineAnnealingWarmRestarts : public LRScheduler;
class CyclicLR : public LRScheduler;
class OneCycleLR : public LRScheduler;
class ReduceLROnPlateau : public LRScheduler;
class LinearLR : public LRScheduler;
class PolynomialLR : public LRScheduler;
class ConstantLR : public LRScheduler;
class SequentialLR : public LRScheduler;
class ChainedScheduler : public LRScheduler;
class LambdaLR : public LRScheduler;
class MultiplicativeLR : public LRScheduler;
```

---

## 8. МОДУЛЬ DATA (ЗАГРУЗКА ДАННЫХ)

### 8.1 Dataset

```cpp
// torch/csrc/api/data/Dataset.h
template<typename Data, typename Target = Data>
class Dataset {
public:
    virtual Example<Data, Target> get(size_t index) = 0;
    virtual size_t size() const = 0;
};

// Трансформации
template<typename D, typename T>
class MapDataset : public Dataset<D, T>;

template<typename... Datasets>
class ChainDataset;

template<typename Dataset>
class SubsetDataset;
```

### 8.2 DataLoader

```cpp
// torch/csrc/api/data/DataLoader.h
template<typename Dataset>
class DataLoader {
public:
    DataLoader(
        Dataset dataset,
        DataLoaderOptions options = {}
    );

    Iterator begin();
    Iterator end();

    size_t size() const;
};

struct DataLoaderOptions {
    int64_t batch_size = 1;
    bool shuffle = false;
    int64_t num_workers = 0;        // Многопоточность
    bool pin_memory = false;        // Pinned memory для GPU
    bool drop_last = false;
    optional<Sampler> sampler;
    optional<Collate> collate_fn;
};
```

### 8.3 Sampler

```cpp
class Sampler {
public:
    virtual Iterator begin() = 0;
    virtual Iterator end() = 0;
};

class SequentialSampler : public Sampler;
class RandomSampler : public Sampler;
class SubsetRandomSampler : public Sampler;
class WeightedRandomSampler : public Sampler;
class BatchSampler : public Sampler;
class DistributedSampler : public Sampler;
```

---

## 9. CUDA BACKEND

### 9.1 CUDA Context

```cpp
// torch/csrc/cuda/CUDAContext.h
class CUDAContext {
public:
    static CUDAContext& getInstance();

    int device_count() const;
    void set_device(int device);
    int current_device() const;

    cudaStream_t current_stream();
    cublasHandle_t cublas_handle();
    cudnnHandle_t cudnn_handle();
};
```

### 9.2 Caching Allocator

```cpp
// torch/csrc/cuda/CUDAAllocator.h
class CUDACachingAllocator : public Allocator {
public:
    DataPtr allocate(size_t size) override;
    void deallocate(DataPtr ptr) override;

    // Управление кэшем
    void empty_cache();
    void memory_stats();

private:
    // Кэш блоков по размерам
    BlockPool small_blocks_;   // < 1MB
    BlockPool large_blocks_;   // >= 1MB

    // Per-stream кэши
    unordered_map<cudaStream_t, BlockPool> stream_pools_;
};
```

### 9.3 CUDA Kernels (примеры)

```cuda
// aten/src/ATen/native/cuda/Activation.cu

__global__ void relu_kernel(float* output, const float* input, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

// aten/src/ATen/native/cuda/Reduce.cu
template<typename T>
__global__ void sum_kernel(T* output, const T* input, int64_t size) {
    // Shared memory reduction
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}
```

---

## 10. PYTHON BINDINGS

### 10.1 Структура

```python
# torch/python/__init__.py
from ._C import *  # C++ биндинги через pybind11

# Основные классы
class Tensor:
    def __init__(self, data=None, dtype=None, device=None, requires_grad=False):
        pass

    # Операции как методы
    def add(self, other, alpha=1): ...
    def mul(self, other): ...
    # ... все 1200+ операций

    # Перегрузка операторов
    def __add__(self, other): return self.add(other)
    def __mul__(self, other): return self.mul(other)
    def __matmul__(self, other): return self.matmul(other)
    # ...

# Функциональный API
import torch

def zeros(*size, dtype=None, device=None): ...
def ones(*size, dtype=None, device=None): ...
# ... все функции
```

### 10.2 pybind11 биндинги

```cpp
// torch/python/_C/init.cpp
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_C, m) {
    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def("add", &Tensor::add)
        .def("__add__", &Tensor::add)
        // ...
        ;

    // Module classes
    py::class_<Module, shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("parameters", &Module::parameters)
        // ...
        ;

    // Functions
    m.def("zeros", &zeros);
    m.def("ones", &ones);
    // ...
}
```

---

## 11. ДИСПЕТЧЕРИЗАЦИЯ ОПЕРАЦИЙ

### 11.1 Dispatcher

```cpp
// aten/src/ATen/Dispatch.h
class Dispatcher {
public:
    template<typename FuncType>
    void registerKernel(
        const OperatorName& op,
        DispatchKey key,
        FuncType* kernel
    );

    template<typename Return, typename... Args>
    Return call(const OperatorName& op, Args... args);

private:
    // Таблица операций
    // op_name -> dispatch_key -> kernel
    unordered_map<string, DispatchTable> ops_;
};

// Ключи диспетчеризации
enum class DispatchKey {
    CPU,
    CUDA,
    Autograd,
    AutogradCPU,
    AutogradCUDA,
    // ...
};
```

### 11.2 Пример регистрации

```cpp
// aten/src/ATen/native/cpu/Activation.cpp
REGISTER_DISPATCH(relu_stub, &relu_kernel_cpu);

// aten/src/ATen/native/cuda/Activation.cu
REGISTER_DISPATCH(relu_stub, &relu_kernel_cuda);

// Диспетчер автоматически выберет нужный кернел
// в зависимости от устройства тензора
```

---

## 12. ПЛАН РАЗРАБОТКИ

### Фаза 1: Ядро (c10)
1. [ ] TensorImpl
2. [ ] Storage
3. [ ] Allocator (CPU)
4. [ ] ScalarType
5. [ ] Device

### Фаза 2: Базовые операции (aten)
1. [ ] Tensor класс
2. [ ] Factory functions (zeros, ones, ...)
3. [ ] Базовые математические операции (+, -, *, /)
4. [ ] Индексация и срезы
5. [ ] Shape операции

### Фаза 3: Autograd
1. [ ] Variable / AutogradMeta
2. [ ] Node базовый класс
3. [ ] Engine
4. [ ] Backward для базовых операций

### Фаза 4: Нейросетевые слои
1. [ ] Module базовый класс
2. [ ] Linear
3. [ ] Conv2d
4. [ ] BatchNorm
5. [ ] Dropout
6. [ ] Activation functions

### Фаза 5: Оптимизаторы
1. [ ] Optimizer базовый класс
2. [ ] SGD
3. [ ] Adam
4. [ ] LR Schedulers

### Фаза 6: Data Loading
1. [ ] Dataset
2. [ ] DataLoader
3. [ ] Samplers

### Фаза 7: CUDA Backend
1. [ ] CUDAAllocator
2. [ ] CUDA kernels для основных операций
3. [ ] cuBLAS интеграция
4. [ ] cuDNN интеграция

### Фаза 8: Python Bindings
1. [ ] pybind11 setup
2. [ ] Tensor биндинги
3. [ ] Module биндинги
4. [ ] Functional API

### Фаза 9: Продвинутые функции
1. [ ] Все математические операции
2. [ ] Все нейросетевые слои
3. [ ] JIT компиляция (опционально)
4. [ ] Distributed training (опционально)

---

## 13. КРИТЕРИИ ГОТОВНОСТИ

Фреймворк считается завершённым когда:

1. **Tensor операции:** Все 1200+ операций реализованы и протестированы
2. **Autograd:** Градиенты корректно вычисляются для всех дифференцируемых операций
3. **nn.Module:** Можно обучить ResNet, Transformer на реальных данных
4. **CUDA:** GPU ускорение работает для всех операций
5. **Performance:** Производительность сравнима с PyTorch (±20%)
6. **Тесты:** 100% покрытие unit тестами
7. **Документация:** API полностью задокументирован

---

## 14. ИСТОЧНИКИ

- [PyTorch Internals (ezyang)](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [PyTorch native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
- [PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)
- [CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)
- [torch.linalg](https://pytorch.org/blog/torch-linalg-autograd/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
