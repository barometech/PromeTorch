Приветствую. Как ревьюер, я проанализировал архитектуру **PromeTorch** (93K LOC, C++17/CUDA/Python). Фреймворк имеет отличную базу (особенно радует поддержка экзотики вроде Эльбрус E2K и NMCard), но для выхода на production-уровень ему критически не хватает структурированной документации и поддержки современных архитектур x86_64 (Intel AMX, AMD Zen4/5) и AMD GPU (ROCm).

Ниже представлен исчерпывающий план по устранению этих пробелов.

---

# ЧАСТЬ 1: ДОКУМЕНТАЦИЯ (DOCUMENTATION GAPS)

Для фреймворка такого масштаба необходима строгая иерархия документации. Вот полный список `.md` файлов, которые необходимо создать в директории `docs/`.

### 1. `BUILD.md` (Руководство по сборке)
*   **Prerequisites:** Требования к компиляторам (GCC >= 7.5, MSVC >= 2019, LCC >= 1.25, NVCC >= 11.0), CMake >= 3.18, Python >= 3.8.
*   **Windows (MSVC):** Специфика сборки (флаги `/O2 /fp:fast`, настройка путей к CUDA Toolkit, решение проблем с `windows.h` макросами `min/max`).
*   **Linux (GCC/Clang):** Стандартный флоу `mkdir build && cd build && cmake ..`.
*   **Elbrus (LCC):** Специфичные флаги для E2K. Использование `lcc` вместо `gcc`, флаги векторизации (`-O4 -ffast-math`), отключение x86-специфичных интринсиков, если они не изолированы.
*   **CUDA Backend:** Как включить/выключить (`-DUSE_CUDA=ON`), настройка `CMAKE_CUDA_ARCHITECTURES` (например, `75;80;86`).
*   **NMCard Mini:** Инструкции по кросс-компиляции с использованием проприетарного SDK NeuroMatrix.

### 2. `QUICKSTART.md` (Быстрый старт)
*   **Тензоры и операции:** Создание тензоров, перенос на устройства (`tensor.to("cuda")`), базовые математические операции.
*   **Autograd:** Как работает `requires_grad=True`, вызов `.backward()`.
*   **Обучение MNIST (MLP):** Полный листинг: загрузка данных -> определение `class Net(prometorch.nn.Module)` -> `prometorch.optim.SGD` -> цикл обучения (forward, loss, backward, step).
*   **Обучение Transformer:** Пример создания Multi-Head Attention, использование `nn.Embedding`, маскирование, цикл обучения на dummy-данных (показ возможностей работы с 3D-тензорами).

### 3. `API_REFERENCE.md` (Структура API)
*   **C++ Core API:**
    *   `prometorch::Tensor` (хранилище, страйды, типы данных).
    *   `prometorch::autograd::Node` (граф вычислений).
    *   `prometorch::Device` и `prometorch::Allocator`.
*   **Python API (Pybind11):**
    *   `prometorch.Tensor`
    *   `prometorch.nn` (Слои: Linear, Conv2d, ReLU, LayerNorm).
    *   `prometorch.optim` (SGD, Adam, AdamW).

### 4. `CUSTOM_OPS.md` (Добавление пользовательских операций)
*   **Шаг 1: C++ Forward & Backward:** Как написать функцию на C++ и структуру для Autograd (наследование от `autograd::Function`).
*   **Шаг 2: CUDA Kernel (опционально):** Как интегрировать `.cu` файл и зарегистрировать диспетчеризацию.
*   **Шаг 3: Pybind11 Binding:** Как прокинуть функцию в Python (макросы или прямые вызовы `m.def`).
*   **Шаг 4: Python Wrapper:** Интеграция в `prometorch.nn.functional`.

### 5. `PERFORMANCE_TUNING.md` (Оптимизация производительности)
*   **Memory Management:** Использование Pinned Memory (`tensor.pin_memory()`), избегание лишних аллокаций в цикле.
*   **Dataloading:** Настройка `num_workers` (если реализован мультипроцессинг).
*   **Hardware-specific:**
    *   CUDA: Использование `cuDNN` бенчмаркинга, асинхронные стримы.
    *   Elbrus: Выравнивание данных в памяти для широких команд (VLIW).
*   **Profiling:** Как использовать встроенный профайлер (если есть) или Nsight Systems / `perf`.

---

# ЧАСТЬ 2: ОПТИМИЗАЦИЯ ПОД AMD И INTEL

Для конкуренции с PyTorch, PromeTorch должен выжимать максимум из x86_64.

## 1. Intel Support

### 1.1 MKL Detection в CMake
Intel MKL (Math Kernel Library) критичен для CPU-матричных умножений.
```cmake
# Современный способ (Intel OneAPI MKL):
find_package(MKL CONFIG)
if(MKL_FOUND)
    target_link_libraries(aten_cpu PUBLIC MKL::MKL)
    target_compile_definitions(aten_cpu PUBLIC PT_USE_MKL)
    message(STATUS "Intel MKL found via OneAPI config")
else()
    # Fallback: ручной поиск (Legacy MKL)
    option(USE_MKL "Use Intel MKL" ON)
    if(USE_MKL)
        find_path(MKL_INCLUDE_DIR mkl.h HINTS $ENV{MKLROOT}/include)
        find_library(MKL_CORE_LIB mkl_core HINTS $ENV{MKLROOT}/lib/intel64)
        find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64 HINTS $ENV{MKLROOT}/lib/intel64)
        find_library(MKL_SEQUENTIAL_LIB mkl_sequential HINTS $ENV{MKLROOT}/lib/intel64)
        if(MKL_INCLUDE_DIR AND MKL_CORE_LIB)
            target_include_directories(aten_cpu PUBLIC ${MKL_INCLUDE_DIR})
            target_link_libraries(aten_cpu PUBLIC
                ${MKL_INTEL_LP64_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB} pthread m dl)
            target_compile_definitions(aten_cpu PUBLIC PT_USE_MKL)
        endif()
    endif()
endif()
```

### 1.2 AVX-512 Kernels
**Какие операции требуют AVX-512:**
1.  *Element-wise:* Add, Mul, ReLU, Sigmoid, Tanh, GELU (аппроксимация).
2.  *Reductions:* Sum, Mean, Max, Softmax (особенно векторизация экспоненты).
3.  *MatMul:* Если MKL недоступен (fallback).

**Структура шаблонов (C++17):**
```cpp
#include <immintrin.h>

namespace prometorch { namespace cpu { namespace kernels {

template <typename T>
void add_avx512(const T* a, const T* b, T* out, size_t size);

template <>
void add_avx512<float>(const float* a, const float* b, float* out, size_t size) {
    size_t i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(&out[i], vc);
    }
    // Tail processing with AVX-512 masked ops (no scalar fallback needed)
    if (i < size) {
        __mmask16 mask = (__mmask16)((1 << (size - i)) - 1);
        __m512 va = _mm512_maskz_loadu_ps(mask, &a[i]);
        __m512 vb = _mm512_maskz_loadu_ps(mask, &b[i]);
        _mm512_mask_storeu_ps(&out[i], mask, _mm512_add_ps(va, vb));
    }
}

}}} // namespace
```

### 1.3 Intel AMX (BF16 MatMul)
AMX (Advanced Matrix Extensions) в Sapphire Rapids дает огромный буст для BF16/INT8.
**Интеграция:**
1.  Необходимо запросить у ОС разрешение на использование AMX (XFD - eXtended Feature Disable).
2.  Настроить тайлы (`_tile_config`).
3.  Использовать `_tile_loadd` и `_tile_dpbf16ps`.

```cpp
#include <immintrin.h>

// Псевдокод для блока 32x32 (используя тайлы 16x16)
void matmul_amx_bf16(const __bf16* A, const __bf16* B, float* C, int K) {
    // 1. Конфигурация тайлов (требует структуры конфигурации)
    // _tile_loadconfig(&config);
    
    // 2. Загрузка аккумулятора (C)
    _tile_loadd(0, C, 16 * sizeof(float)); 
    
    // 3. Цикл по K
    for (int k = 0; k < K; k += 32) { // 32 bf16 elements = 64 bytes
        _tile_loadd(1, A + k, ...); // Загрузка тайла A
        _tile_loadd(2, B + k, ...); // Загрузка тайла B
        _tile_dpbf16ps(0, 1, 2);    // C += A * B (dot product)
    }
    
    // 4. Сохранение результата
    _tile_stored(0, C, 16 * sizeof(float));
    // _tile_release();
}
```

### 1.4 Runtime Dispatch (Динамический выбор ISA)
Нельзя компилировать весь код с `-mavx512f`, иначе он упадет на старых CPU. Нужен CPUID детектор и диспетчер.

```cpp
// cpu_features.h
class CPUFeatures {
public:
    static CPUFeatures& get() { static CPUFeatures instance; return instance; }
    bool has_avx2() const { return has_avx2_; }
    bool has_avx512f() const { return has_avx512f_; }
    bool has_amx_bf16() const { return has_amx_bf16_; }
private:
    CPUFeatures(); // Реализация через инструкцию cpuid (__cpuid)
    bool has_avx2_, has_avx512f_, has_amx_bf16_;
};

// dispatch.cpp
using AddFunc = void(*)(const float*, const float*, float*, size_t);

void add_scalar(const float* a, const float* b, float* out, size_t size) { /*...*/ }
void add_avx2(const float* a, const float* b, float* out, size_t size) { /*...*/ }
// add_avx512 определен в файле, скомпилированном с -mavx512f

AddFunc get_add_kernel() {
    if (CPUFeatures::get().has_avx512f()) return add_avx512;
    if (CPUFeatures::get().has_avx2()) return add_avx2;
    return add_scalar;
}

// Вызов в тензорной операции
void Tensor::add(const Tensor& other) {
    static AddFunc kernel = get_add_kernel();
    kernel(this->data(), other.data(), this->data(), this->size());
}
```

---

## 2. AMD CPU Support

### 2.1 AOCL/BLIS Detection в CMake
AMD Optimizing CPU Libraries (AOCL) содержит BLIS (аналог BLAS).
```cmake
option(USE_BLIS "Use AMD BLIS" OFF)
if(USE_BLIS)
    find_path(BLIS_INCLUDE_DIR blis/blis.h HINTS /opt/AMD/aocl/blis/include)
    find_library(BLIS_LIB blis HINTS /opt/AMD/aocl/blis/lib)
    if(BLIS_INCLUDE_DIR AND BLIS_LIB)
        add_library(prometorch::blis INTERFACE IMPORTED)
        target_include_directories(prometorch::blis INTERFACE ${BLIS_INCLUDE_DIR})
        target_link_libraries(prometorch::blis INTERFACE ${BLIS_LIB} pthread m)
        add_definitions(-DUSE_BLIS)
    endif()
endif()
```

### 2.2 Zen4/Zen5 Specific Tuning
*   **Zen 4:** Поддерживает AVX-512, но физически векторные регистры 256-битные (double-pumping). AVX-512 код от Intel будет работать отлично и без троттлинга частоты (в отличие от старых Intel).
*   **Zen 5:** Имеет полноценный 512-битный FPU.
*   **Тюнинг:** Избегать частых переходов между 256-битными и 512-битными инструкциями (vzeroupper). Инструкции перестановок (`vpermd`, `vpermps`) на Zen4 могут иметь бóльшую латентность, чем на Intel — при написании кастомных ядер для Softmax/Transpose стоит предпочесть `vshufps` где возможно.

### 2.3 Использование BLIS как SGEMM Backend
```cpp
#ifdef USE_BLIS
#include <blis/blis.h>

void sgemm_blis(bool transA, bool transB, int M, int N, int K, 
                float alpha, const float* A, const float* B, float beta, float* C) {
    trans_t ta = transA ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
    trans_t tb = transB ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE;
    
    // BLIS использует типизированные вызовы
    bli_sgemm(ta, tb, M, N, K, 
              &alpha, const_cast<float*>(A), transA ? 1 : K, transA ? M : 1,
              const_cast<float*>(B), transB ? 1 : N, transB ? K : 1,
              &beta, C, N, 1); // Предполагается Row-major (C-style)
}
#endif
```

---

## 3. AMD GPU (ROCm) Support

Для портирования CUDA на AMD ROCm используется HIP (Heterogeneous-Compute Interface for Portability). HIP-код может компилироваться как под NVIDIA (через `nvcc`), так и под AMD (через `hipcc`).

### 3.1 Какие файлы портировать (Список)
Все файлы, содержащие CUDA-специфичный код, должны быть абстрагированы или скопированы.
*   `allocator_cuda.cpp` -> `allocator_hip.cpp`
*   `tensor_cuda.cu` -> `tensor_hip.hip` (или `.cpp` если компилировать через hipcc)
*   `math_cuda.cu` -> `math_hip.hip`
*   `cudnn_wrapper.cpp` -> `miopen_wrapper.cpp` (MIOpen — аналог cuDNN от AMD)
*   `cublas_wrapper.cpp` -> `hipblas_wrapper.cpp`

*Архитектурный совет:* Лучше переименовать `.cu` файлы в `.hip` и использовать их как единый source of truth для обоих GPU-бэкендов, так как HIP синтаксически совместим с CUDA через макросы.

### 3.2 HIP Porting: Механические изменения

> **КРИТИЧНО: WARP_SIZE!** AMD CDNA/RDNA использует wavefront=64 (не 32 как NVIDIA). 
> Все `__shfl_down_sync(0xffffffff, ...)` и `constexpr WARP_SIZE=32` ДОЛЖНЫ быть заменены на макросы.
> Это уже сделано в PromeTorch (`WARP_MASK` + `#ifdef __HIP_PLATFORM_AMD__` в FlashAttention.cu, CUDAReduce.cu, CUDAQuantGemv.cu).
> Также: для AMX (Intel) в Linux нужен `syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` перед использованием — иначе SIGILL.

Можно использовать утилиту `hipify-perl` или `hipify-clang` от AMD, которая сделает 90% работы:
*   `cudaMalloc` -> `hipMalloc`
*   `cudaMemcpy` -> `hipMemcpy`
*   `cudaStream_t` -> `hipStream_t`
*   `__global__`, `__device__`, `__shared__` -> остаются **без изменений**!
*   `threadIdx.x` -> `hipThreadIdx_x` (или оставить `threadIdx.x`, HIP макросы это поддерживают).
*   `cudaError_t` -> `hipError_t`

**Пример ядра:**
```cpp
// Было (CUDA):
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
// Вызов: add_kernel<<<blocks, threads, 0, stream>>>(...);

// Стало (HIP):
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
// Вызов: hipLaunchKernelGGL(add_kernel, dim3(blocks), dim3(threads), 0, stream, ...);
```

### 3.3 CMake Setup для HIP
```cmake
option(USE_ROCM "Build with AMD ROCm support" OFF)

if(USE_ROCM)
    # Ищем HIP
    find_package(HIP REQUIRED)
    
    # Ищем hipBLAS и MIOpen
    find_package(hipblas REQUIRED)
    find_package(miopen REQUIRED)

    set(HIP_SRC 
        src/backend/hip/tensor_hip.hip
        src/backend/hip/math_hip.hip
    )

    # Устанавливаем архитектуры (например, gfx90a для MI200, gfx1030 для RX6000)
    set_property(TARGET prometorch PROPERTY HIP_ARCHITECTURES gfx90a gfx1030)

    # Добавляем библиотеку
    hip_add_library(prometorch_hip ${HIP_SRC})
    target_link_libraries(prometorch_hip PUBLIC hip::host roc::hipblas MIOpen)
    
    add_definitions(-DUSE_ROCM)
endif()
```

### 3.4 hipBLAS vs rocBLAS для SGEMM
*   **rocBLAS:** Низкоуровневая библиотека, работает *только* на AMD. Максимальная производительность, но API отличается от cuBLAS.
*   **hipBLAS:** Маршрутизирующая библиотека (Wrapper). Если компилируем под AMD — вызывает `rocBLAS`. Если под NVIDIA — вызывает `cuBLAS`.
*   **Вердикт для PromeTorch:** Использовать **hipBLAS**. Это позволит иметь один файл `hipblas_wrapper.cpp`, который будет обслуживать и NVIDIA, и AMD, сокращая дублирование кода.

```cpp
// Пример использования hipBLAS (работает и на CUDA, и на ROCm)
#include <hipblas/hipblas.h>

void sgemm_gpu(hipblasHandle_t handle, int M, int N, int K, 
               const float* A, const float* B, float* C) {
    float alpha = 1.0f;
    float beta = 0.0f;
    // hipblasSgemm сигнатура идентична cublasSgemm
    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 
                 M, N, K, &alpha, A, M, B, K, &beta, C, M);
}
```

Внедрение этого плана сделает PromeTorch кросс-платформенным монстром, способным утилизировать как тензорные ядра NVIDIA, так и матричные расширения Intel AMX и ускорители AMD Instinct.