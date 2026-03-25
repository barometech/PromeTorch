# NM Card Mini Backend -- PromeTorch

## 1. Обзор

NM Card Mini Backend -- третий вычислительный backend фреймворка PromeTorch, наряду с CPU и CUDA. Реализует программный эмулятор процессора NM6408 (NeuroMatrix) и интерфейс к реальному оборудованию NM Card Mini (4 кластера x 4 ядра NMC4 = 16 ядер).

Backend обеспечивает:

- **Эмуляцию NMC4** на хост-CPU с двумя режимами: Q16.16 fixed-point (точность NMC4) и native float32 (для отладки)
- **Аппаратное ускорение** через DLL `nm_card_load.dll` и DDR-протокол обмена командами
- **Полную интеграцию** с тензорной системой PromeTorch: device type `PrivateUse1`, автоматический dispatch, autograd-совместимость
- **Обучение моделей**: forward + backward операции, оптимизаторы SGD/Adam

Результат: **MNIST 93.64%** (3 эпохи, SGD lr=0.01), 32/32 тестов пройдены.

---

## 2. Архитектура

### 2.1. Q16.16 Fixed-Point арифметика

Процессор NMC4 не имеет аппаратного умножения с плавающей точкой. Все вычисления выполняются в формате Q16.16:

- **16 бит** -- целая часть (знаковая)
- **16 бит** -- дробная часть
- **Диапазон**: [-32768.0, +32767.99998]
- **Точность**: ~1.5e-5

Конвертация:
```
float -> Q16.16:  fixed32 = (int32_t)(float_val * 65536)
Q16.16 -> float:  float_val = (float)fixed32 / 65536
```

Библиотека `NMCardMath.h` реализует: `add_fixed`, `sub_fixed`, `mul_fixed`, `div_fixed`, `sqrt_fixed`, `exp_fixed_lut`, `sin_fixed`, `cos_fixed`, `silu_fixed`, `gelu_fixed` и обратные функции для backward pass.

### 2.2. Device Type: PrivateUse1

NM Card зарегистрирована как устройство `c10::DeviceType::PrivateUse1` (device index 0). Это стандартный механизм расширения PyTorch-совместимых фреймворков для кастомного оборудования.

```cpp
// Проверка устройства тензора
tensor.is_nmcard()  // true если на NM Card

// Перенос данных
at::to_nmcard(cpu_tensor)    // CPU -> NMCard
at::nmcard_to_cpu(nmc_tensor) // NMCard -> CPU
```

### 2.3. Двухуровневый dispatch

Каждая операция проходит через `NMCardOps.h`, который проверяет доступность аппаратуры:

```
Tensor-level API (NMCardDispatch.h)
  |
  v
Launch wrapper (NMCardOps.h)
  |
  +-- NMCardHardware доступен? --> DDR протокол -> NMC4 ядра
  |
  +-- Нет --> NMCardEmulator (CPU, Q16.16 или float32)
```

Операции с аппаратным ускорением: `matmul`, `rmsnorm`, `softmax`, `silu`, `rope`, `elem_add`, `elem_mul`, `gate_mul`.

CPU-fallback: `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `gelu`, `layernorm`, comparisons, reductions, backward ops, optimizers.

---

## 3. Компоненты

### 3.1. NMCardMath.h -- Q16.16 математика

**Путь**: `aten/src/ATen/nmcard/NMCardMath.h`

Порт `mymath.h` (NMC4 firmware) на x86. Содержит:

| Функция | Описание |
|---------|----------|
| `float_to_fixed(float)` | Конвертация float -> Q16.16 |
| `fixed_to_float(fixed32)` | Конвертация Q16.16 -> float |
| `mul_fixed(a, b)` | Умножение через int64_t промежуточное значение |
| `div_fixed(a, b)` | Деление с сохранением дробной части |
| `sqrt_fixed(x)` | Квадратный корень (итерационный) |
| `exp_fixed_lut(x)` | Экспонента через LUT (look-up table) |
| `sin_fixed(x)`, `cos_fixed(x)` | Тригонометрия (полиномиальная аппроксимация) |
| `silu_fixed(x)` | SiLU = x * sigmoid(x) |
| `gelu_fixed(x)` | GELU (tanh-аппроксимация) |
| `silu_backward(x, grad)` | Обратный проход SiLU |
| `gelu_backward(x, grad)` | Обратный проход GELU |

Константы: `FIXED_ONE = 65536`, `FIXED_PI = 205887`, `FIXED_TWO_PI = 411775`.

### 3.2. NMCardEmulator.h -- эмуляция NMC4 DSP

**Путь**: `aten/src/ATen/nmcard/NMCardEmulator.h`

Программный эмулятор 16 виртуальных ядер NMC4. Singleton (`NMCardEmulator::get()`).

**Конфигурация:**
```cpp
auto& emu = NMCardEmulator::get();
emu.set_fixed_point(true);   // Q16.16 режим (точность NMC4)
emu.set_fixed_point(false);  // float32 режим (для отладки)
emu.set_num_cores(16);       // 1..16 виртуальных ядер
```

**Forward операции** (22 штуки):

| Операция | Метод | Q16.16 | float32 |
|----------|-------|--------|---------|
| MatMul | `matmul(A, B, C, M, K, N)` | + | + |
| RMSNorm | `rmsnorm(input, output, gamma, batch, hidden)` | + | + |
| Softmax | `softmax(input, output, batch, dim)` | + | + |
| SiLU | `silu(input, output, count)` | + | + |
| GELU | `gelu(input, output, count)` | + | + |
| RoPE | `rope(input, output, freqs, seq_len, head_dim, pos_offset)` | + | + |
| LayerNorm | `layernorm(input, output, gamma, beta, batch, hidden)` | - | + |
| Element-wise add/sub/mul | `elem_add`, `elem_sub`, `elem_mul` | + | + |
| Gate Mul | `gate_mul(a, b, out, count)` -- `a * silu(b)` (Llama FFN) | + | + |
| Scalar Mul | `mul_scalar(input, scalar, output, count)` | + | + |
| Neg, ReLU | `neg`, `relu` | - | + |

**Backward операции** (7 штук):
- `matmul_backward` -- dA = dC @ B^T, dB = A^T @ dC
- `silu_backward_op`, `gelu_backward_op`, `relu_backward_op`
- `softmax_backward_op`, `rmsnorm_backward_op`
- `cross_entropy` (loss), `cross_entropy_backward_op`

**Оптимизаторы** (2 штуки):
- `sgd_step` -- SGD: w -= lr * grad (Q16.16 и float32)
- `adam_step` -- Adam: m/v moments + bias-corrected update (Q16.16 и float32)

### 3.3. NMCardDispatch.h -- диспатч тензорных операций

**Путь**: `aten/src/ATen/nmcard/NMCardDispatch.h`

Tensor-level API, аналог `CUDADispatch.h`. Предоставляет:

**Управление устройством:**
- `empty_nmcard(sizes, dtype, device)` -- создание пустого тензора на NMCard
- `to_nmcard(src)` -- копирование CPU-тензора на NMCard (memcpy + сохранение autograd metadata)
- `nmcard_to_cpu(src)` -- копирование NMCard-тензора на CPU
- `ensure_hardware_init()` -- автоматическая инициализация аппаратуры при первом использовании

**Namespace `nmc_ops`** -- 60+ высокоуровневых операций:

*Unary (22):* `neg`, `abs`, `sqrt`, `rsqrt`, `square`, `exp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `tanh`, `sigmoid`, `relu`, `silu`, `gelu`, `ceil`, `floor`, `round`, `sign`, `reciprocal`

*Binary (10):* `add`, `add_scalar`, `add_broadcast`, `sub`, `mul`, `mul_scalar`, `mul_broadcast`, `div`, `pow`, `pow_scalar`

*Comparison (6 + 6 scalar):* `eq`, `ne`, `lt`, `le`, `gt`, `ge` + `*_scalar` варианты

*Reduce (7):* `sum`, `sum_dim`, `mean`, `max`, `min`, `argmax`, `argmin`

*Linear algebra (4):* `mm`, `mv`, `bmm`, `dot`

*Fused (4):* `addcmul`, `addcdiv`, `leaky_relu`, `clamp`

*Утилиты:* `fill_`, `copy_`, `ensure_contiguous_nmcard`

Все операции автоматически вызывают `ensure_contiguous_nmcard()` перед работой с raw-указателями.

### 3.4. NMCardOps.h -- обертки операций

**Путь**: `aten/src/ATen/nmcard/NMCardOps.h`

Тонкий dispatch-слой между `NMCardDispatch.h` (тензоры) и `NMCardEmulator.h`/`NMCardHardware.h` (вычисления). Каждая `launch_*` функция проверяет `NMCardHardware::get().is_available()`:

```cpp
inline void launch_silu(const float* input, float* output, int64_t n) {
    if (NMCardHardware::get().is_available()) {
        NMCardHardware::get().silu(input, output, n);  // аппаратное ускорение
    } else {
        NMCardEmulator::get().silu(input, output, n);   // CPU эмуляция
    }
}
```

Операции без аппаратного opcode (abs, sqrt, exp, log, trig, comparisons, reductions) всегда выполняются через CPU-fallback напрямую.

### 3.5. NMCardAllocator.h -- кэширующий аллокатор

**Путь**: `c10/nmcard/NMCardAllocator.h`

Кэширующий аллокатор памяти, паттерн CUDAAllocator:

- **Выравнивание**: 64 байта (cache line)
- **Кэширование**: освобожденные блоки сохраняются в `free_blocks_` для повторного использования
- **Best-fit**: при поиске свободного блока выбирается наименьший подходящий
- **Device tag**: `DeviceType::PrivateUse1, index=0`
- **Thread-safe**: защита через `std::mutex`
- **Платформа**: `_aligned_malloc`/`_aligned_free` (Windows), `posix_memalign`/`free` (Linux)

Singleton реализован в `.cpp` файле (не inline static) для корректной работы с DLL.

**API:**
```cpp
auto& alloc = NMCardAllocator::get();
DataPtr ptr = alloc.allocate(nbytes);      // аллокация
alloc.raw_deallocate(ptr.get());           // возврат в кэш
alloc.empty_cache();                       // освобождение кэша
alloc.get_allocated_memory();              // статистика
alloc.get_cached_memory();
```

### 3.6. NMCardHardware.h -- аппаратный уровень

**Путь**: `aten/src/ATen/nmcard/NMCardHardware.h`

Интерфейс к реальной карте NM Card Mini через DLL `nm_card_load.dll`.

**Инициализация:**
```cpp
auto& hw = NMCardHardware::get();
if (hw.init("path/to/dispatcher.abs")) {
    // Карта доступна, операции будут ускорены
}
```

**DDR Memory Layout (от DDR_BASE = 0x00340000):**

| Область | Адрес | Размер | Назначение |
|---------|-------|--------|------------|
| Command Blocks | DDR_BASE | 512 слов | 16 блоков x 32 слова (команды для ядер) |
| Data Area | DDR_BASE + 512 | ~64KB | Входные/выходные данные |
| Weight Area | DDR_BASE + 65536 | до DDR_END | Веса моделей (Q16.16) |

**DDRAllocator** -- bump-аллокатор DDR-адресного пространства:
- Выделение с выравниванием 16 слов (64 байта)
- Маппинг `host_ptr -> ddr_addr` для передачи адресов ядрам
- Метод `reset()` для сброса между операциями/эпохами

**API DLL (nm_card_load.dll):**
- `PL_GetBoardCount` -- обнаружение плат
- `PL_GetBoardDesc` / `PL_CloseBoardDesc` -- дескриптор платы
- `PL_ResetBoard` -- сброс
- `PL_LoadInitCode` -- загрузка init-кода
- `PL_GetAccess` / `PL_CloseAccess` -- получение доступа к ядру (per-cluster)
- `PL_LoadProgramFile` -- загрузка .abs файла на ядро
- `PL_WriteMemBlock` / `PL_ReadMemBlock` -- запись/чтение DDR

### 3.7. NMCardMultiCore.h -- мультиядерная модель

**Путь**: `aten/src/ATen/nmcard/NMCardMultiCore.h`

Параллельный dispatch на 16 ядер (4 кластера x 4 ядра NMC4).

**Алгоритм parallel_matmul:**
1. Загрузка матриц A и B в DDR (общие для всех ядер)
2. Разделение столбцов результата C между ядрами: `cols_per_core = N / num_cores`
3. Отправка `OP_MATMUL_PARTIAL` каждому ядру с аргументами `[M, K, N, addr_A, addr_B, addr_C, col_start, col_end]`
4. Ожидание завершения всех ядер (`wait_done`)
5. Скачивание результата C из DDR

**Особенность**: каждое ядро имеет собственный `PL_Access` handle (per-cluster DDR mapping). Нельзя читать/писать DDR чужого ядра через свой handle.

### 3.8. NMCardLoRA.h -- LoRA для NM Card

**Путь**: `aten/src/ATen/nmcard/NMCardLoRA.h`

Low-Rank Adaptation для эффективного fine-tuning на NM Card. Реализует декомпозицию весовой матрицы W + BA, где B и A -- low-rank матрицы.

---

## 4. Dispatcher протокол

### 4.1. DDR Command Block

Каждое ядро NMC4 имеет 32-словный (128 байт) командный блок в DDR:

```
Адрес блока = DDR_BASE + core_index * 32

[0]  = opcode         (код операции, 0 = idle)
[1]  = arg0            (зависит от операции)
[2]  = arg1
...
[10] = arg9
...
[30] = STATUS          (0 = busy, 1 = done, 2 = error)
[31] = WATCHDOG        (инкрементируется в главном цикле)
```

### 4.2. Протокол обмена

**Хост (PromeTorch):**
1. Загрузить данные в DDR через `PL_WriteMemBlock`
2. Записать opcode + аргументы в command block ядра
3. Поллить STATUS (слово [30]) пока не станет 1 (done) или 2 (error)
4. Прочитать результат из DDR через `PL_ReadMemBlock`

**NMC4 ядро (dispatcher.abs):**
1. Бесконечный цикл: поллить слово [0] (opcode)
2. Когда opcode != 0: прочитать аргументы [1..N], выполнить операцию
3. Записать STATUS = 1 (done)
4. Инкрементировать WATCHDOG [31]
5. Сбросить opcode [0] = 0
6. Вернуться к шагу 1

### 4.3. Таблица opcodes

| Opcode | Имя | Аргументы | Описание |
|--------|-----|-----------|----------|
| 0 | NOP | -- | Idle (ядро ждет) |
| 1 | MATMUL | M, K, N, addr_A, addr_B, addr_C | C = A @ B |
| 2 | RMSNORM | batch, hidden, addr_in, addr_out, addr_gamma | RMS нормализация |
| 3 | SOFTMAX | batch, dim, addr_in, addr_out | Softmax по последнему измерению |
| 4 | SILU | count, addr_in, addr_out | x * sigmoid(x) |
| 5 | ROPE | seq_len, head_dim, pos, addr_in, addr_out, addr_freqs | Rotary Position Embedding |
| 6 | ATTENTION | -- | Scaled dot-product attention |
| 10 | ELEM_ADD | count, addr_a, addr_b, addr_out | Поэлементное сложение |
| 11 | ELEM_MUL | count, addr_a, addr_b, addr_out | Поэлементное умножение |
| 12 | ELEM_SUB | count, addr_a, addr_b, addr_out | Поэлементное вычитание |
| 13 | GATE_MUL | count, addr_a, addr_b, addr_out | a * silu(b) (Llama FFN) |
| 14 | MUL_SCALAR | count, scalar, addr_in, addr_out | Умножение на скаляр |
| 15 | GELU | count, addr_in, addr_out | GELU активация |
| 16 | LAYERNORM | batch, hidden, addr_in, addr_out, addr_gamma, addr_beta | Layer Normalization |
| 20 | MATMUL_DDR | -- | MatMul с предзагруженными Q16.16 весами |
| 21 | RMSNORM_DDR | -- | RMSNorm с предзагруженными Q16.16 gamma |
| 22 | MATMUL_PARTIAL | M, K, N, addr_A, addr_B, addr_C, col_start, col_end | Частичный MatMul (мультиядерный) |
| 30 | MATMUL_BACKWARD | -- | Backward MatMul |
| 31 | SILU_BACKWARD | -- | Backward SiLU |
| 32 | GELU_BACKWARD | -- | Backward GELU |
| 33 | SOFTMAX_BACKWARD | -- | Backward Softmax |
| 34 | RMSNORM_BACKWARD | -- | Backward RMSNorm |
| 35 | ROPE_BACKWARD | -- | Backward RoPE |
| 40 | CROSS_ENTROPY | -- | Cross-entropy loss |
| 41 | CROSS_ENTROPY_BACKWARD | -- | Backward cross-entropy |
| 50 | SGD_STEP | -- | SGD optimizer step |
| 51 | ADAM_STEP | -- | Adam optimizer step |
| 255 | EXIT | -- | Graceful shutdown |

---

## 5. NMC программы

Каталог: `aten/src/ATen/nmcard/nmc_programs/`

### 5.1. dispatcher.cpp / dispatcher.abs

Основная firmware для одного ядра NMC4. Единая программа-диспетчер:
- Загружается один раз (`PL_LoadProgramFile`)
- Обрабатывает все операции через опкоды в DDR
- Использует Q16.16 fixed-point арифметику (`mymath.h`)
- **Нет библиотечных вызовов** -- все вычисления кастомные

```c
// Главный цикл (упрощенно)
while (1) {
    unsigned int op = mem[0];    // поллинг opcode
    if (op == OP_EXIT) break;
    if (op != OP_NOP) {
        switch (op) {
            case OP_MATMUL:  op_matmul();  break;
            case OP_SOFTMAX: op_softmax(); break;
            // ...
        }
        mem[STATUS_ADDR] = 1;    // done
        mem[WATCHDOG_ADDR]++;
        mem[0] = 0;              // reset opcode
    }
}
```

### 5.2. dispatcher_mc.cpp / dispatcher_mc.abs

Мультиядерный вариант диспетчера. Каждое ядро определяет свой индекс через `ncl_getCoreID()` / `ncl_getClusterID()` и читает свой командный блок по адресу `DDR_BASE + core_index * 32`.

### 5.3. Специализированные программы (.abs)

43 скомпилированных .abs файла для индивидуальных операций и тестов:

- **Операции**: `matmul.abs`, `rmsnorm.abs`, `softmax.abs`, `silu.abs`, `rope.abs`, `elementwise.abs`, `layernorm.abs`, `attention.abs`, `cross_entropy.abs`
- **Backward**: `matmul_backward.abs`, `silu_backward.abs`, `gelu_backward.abs`, `softmax_backward.abs`, `rmsnorm_backward.abs`, `rope_backward.abs`, `attention_backward.abs`
- **Оптимизаторы**: `sgd_update.abs`, `adam_update.abs`
- **Тесты (QEMU)**: `qemu_float_test.abs`, `qemu_backward_test.abs`, `qemu_training_test.abs`, `qemu_inference_test.abs` и др.
- **Диагностика**: `echo_test.abs`, `hello_card.abs`, `simple_test.abs`, `instant_test.abs`

### 5.4. Вспомогательные файлы

- `dispatcher_float_vec_gas.s` / `.o` -- ассемблерные подпрограммы NMC4
- `nmblas.h`, `nmdef.h` -- заголовки для NMC BLAS
- `mulmv.mlb`, `macros.mlb` -- макробиблиотеки
- `nmplm/` -- библиотека nmplm (NMC math primitives)
- `MullMatrix_f.o`, `mtrMul_mm_32sXs_nm.o`, `nmppmMul_mm_32s32s_nm.o` -- объектные файлы BLAS
- `libnm6408load_nmc.a` -- статическая библиотека загрузчика

---

## 6. Результаты

### 6.1. Тестирование

**32/32 тестов пройдены**, включая:

- Q16.16 арифметика (add, sub, mul, div, sqrt, exp, sin, cos)
- MatMul (forward + backward)
- RMSNorm, Softmax, SiLU, GELU, RoPE (forward + backward)
- Cross-entropy loss + backward
- SGD / Adam optimizer steps
- Tensor transfer (CPU <-> NMCard)
- Аллокатор (allocate, deallocate, cache reuse)
- Device detection (`is_nmcard()`, `is_cpu()`)

### 6.2. MNIST

| Параметр | Значение |
|----------|----------|
| Модель | MLP (784-256-128-10) |
| Оптимизатор | SGD, lr=0.01 |
| Эпохи | 3 |
| Точность | **93.64%** |
| Режим | Эмулятор, float32 |

Сборка и запуск:
```bash
cd /c/Users/paper/Desktop/promethorch
cmake -DPT_USE_NMCARD=ON ...
# из Developer Command Prompt
cd build_nmcard
nmake train_mnist_nmcard
./train_mnist_nmcard.exe
```

---

## 7. Известные проблемы

### 7.1. DLL Singleton (двойная регистрация аллокатора)

**Проблема**: `AllocatorRegistry` использует `inline static` -- каждая DLL получает свою копию реестра. Аллокатор, зарегистрированный в одной DLL, невидим из другой.

**Решение**: двойная регистрация:

```cpp
// В DLL (NMCardAllocator.cpp) -- экспортируемая функция
ATEN_NMCARD_API void register_nmcard_allocator() {
    AllocatorRegistry::get().registerAllocator(
        DeviceType::PrivateUse1, &NMCardAllocator::get());
}

// В main() приложения -- inline регистрация в реестре вызывающего модуля
c10::nmcard::register_nmcard_allocator_local();
```

Singleton `NMCardAllocator::get()` реализован в `.cpp` файле, а не как `inline static` в заголовке, чтобы гарантировать единственный экземпляр.

### 7.2. 16-Core Crash Incident (2026-03-18)

**Инцидент**: запуск 16-ядерного matmul на реальной карте привел к crash и перезагрузке ПК с потерей данных.

**Причина**: race condition в DDR доступе + насыщение DDR пропускной способности при 16 одновременных ядрах.

**Протокол безопасности** (введен после инцидента):
1. Сначала тестировать на эмуляторе
2. Затем 1 ядро на реальной карте
3. Затем 2 ядра
4. Затем 4 ядра
5. Только потом 16 ядер
6. **Всегда спрашивать разрешение** перед запуском на реальной карте

**Файл `dispatcher_suda_mc.abs` -- НЕ ЗАПУСКАТЬ** на реальной карте (DDR race condition).

### 7.3. Прочие ограничения

- **LayerNorm**: нет Q16.16 реализации (только float32)
- **Neg, ReLU**: нет Q16.16 ветки в эмуляторе (только float32)
- **Backward операции**: softmax_backward, rmsnorm_backward, relu_backward -- только float32 (без Q16.16)
- **GELU**: нет аппаратного opcode в dispatcher (только CPU fallback)
- **Размер DDR**: ~500MB (DDR_END = 0x1FF00000), ограничивает размер моделей

---

## 8. Сборка

### 8.1. CMake

```bash
cmake .. -G "NMake Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPT_USE_NMCARD=ON \
  -DPT_BUILD_TESTS=OFF
```

Флаг `-DPT_USE_NMCARD=ON` активирует:
- Компиляцию `c10/nmcard/` и `aten/src/ATen/nmcard/`
- Определение макроса `PT_USE_NMCARD`
- Линковку `NMCardAllocator.cpp`, `NMCardEmulator.cpp`, `NMCardHardware.cpp`

### 8.2. Готовая сборка

```
build_nmcard/train_mnist_nmcard.exe   # MNIST на эмуляторе NM Card
```

### 8.3. Требования

- **Эмулятор**: Windows 10 / Linux, MSVC 2019 / GCC / LCC (Elbrus), CMake 3.15+
- **Реальная карта**: Windows 10, NM Card Mini установлена, `nm_card_load.dll` в PATH, драйверы partner

---

## 9. Структура файлов

```
c10/nmcard/
  NMCardAllocator.h           # Кэширующий аллокатор (PrivateUse1)
  NMCardAllocator.cpp          # Singleton + deleter implementations

aten/src/ATen/nmcard/
  NMCardMath.h                 # Q16.16 fixed-point библиотека
  NMCardEmulator.h             # Программный эмулятор NMC4 (16 ядер)
  NMCardEmulator.cpp           # Singleton
  NMCardOps.h                  # Launch wrappers (hardware vs emulator)
  NMCardDispatch.h             # Tensor-level API (empty/to/from + 60+ ops)
  NMCardHardware.h             # Реальная карта: DLL + DDR протокол
  NMCardHardware.cpp           # Singleton + DLL loading
  NMCardMultiCore.h            # 16-ядерный параллельный dispatch
  NMCardLoRA.h                 # LoRA для fine-tuning

  nmc_programs/
    dispatcher.cpp / .abs      # Основная firmware (single-core)
    dispatcher_mc.cpp / .abs   # Мультиядерная firmware
    mymath.h                   # Q16.16 математика для NMC4
    *.cpp / *.abs              # 43 программы (ops + tests)
    *.s / *.o / *.a            # Ассемблер + объектные файлы NMC
    nmplm/                     # NMC math primitives library

examples/nmcard/
  train_mnist_nmcard.cpp       # MNIST обучение на NM Card (эмулятор)
```
