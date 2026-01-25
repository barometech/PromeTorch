# PromeTorch - Resume Point

**Дата:** 2026-01-24
**Ветка:** `feature/memory-leak-fix`
**Коммиты:**
- `50640d2` - Initial commit: PromeTorch v0.1.0 (117 files, 85K lines)
- `6b64252` - Add Docker support and update gitignore

## Что это

PromeTorch - полностью независимый фреймворк глубокого обучения на C++/CUDA.
Замена PyTorch без зависимостей от него.

## Текущий статус

### Завершено (14 фаз)
- c10 core (Allocator, Device, Storage, TensorImpl)
- ATen tensor operations (MathOps, ReduceOps, LinearAlgebra, ShapeOps)
- Autograd (backward engine, 50+ backward functions)
- NN Modules (50+ слоёв: Linear, Conv, BatchNorm, Transformer, PIR...)
- Optimizers (SGD, Adam, AdamW, RMSprop)
- LR Schedulers (10+ видов)
- Data Loading (Dataset, DataLoader, Sampler)
- CUDA Backend (собственные kernels)
- cuDNN Integration
- FlashAttention
- Mixed Precision (AMP)
- Python Bindings (pybind11)

### В работе: Утечка GPU памяти — ROOT CAUSE НАЙДЕН!

**Проблема:** При обучении PIR модели на CUDA crash на первой итерации (heap corruption).

**Root Cause:** DLL Singleton Problem!
`CUDACachingAllocator::get()` была inline функция со static переменной.
На Windows каждая DLL получала СВОЙ instance allocator'а.
Allocation в одном модуле + deallocation в другом = heap corruption.

**РЕШЕНИЕ (2026-01-24):**
1. Создан `c10/cuda/CUDAAllocator.cpp` с единственным singleton
2. `get()` теперь exported function, не inline
3. `aten_cuda` теперь SHARED library (не STATIC)

**СТАТУС:** ⚠️ Требует пересборки и тестирования

## Рабочие директории сборки

| Директория | Статус | Описание |
|------------|--------|----------|
| `build_cuda_examples` | ✅ Рабочий | Последняя сборка с test_mem_leak.exe |
| `build_cudnn` | ❌ Сломан | CMakeCache удалён, требует полной пересборки |
| `build_cuda_check` | ✅ Рабочий | Резервная CUDA сборка |

## Ключевые файлы

| Файл | Описание |
|------|----------|
| `examples/test_mem_leak.cpp` | Тест утечки памяти |
| `examples/pir/train_pir.cpp` | Обучение PIR модели |
| `torch/csrc/autograd/autograd.h` | Autograd система |
| `torch/csrc/autograd/node.h` | Узлы графа (g_nodes_created/destroyed счётчики) |
| `aten/src/ATen/cuda/CUDADispatch.h` | CUDA операции |

## Команды сборки

```batch
REM Открыть Developer Command Prompt for VS 2019

cd C:\Users\paper\Desktop\promethorch\build_cuda_examples
cmake ..
nmake test_mem_leak

REM Запуск теста
set PATH=C:\Users\paper\Desktop\promethorch\build_cuda_examples;C:\ProgramData\anaconda3\Library\bin;%PATH%
examples\pir\test_mem_leak.exe --device cpu --iterations 20
```

## Следующие шаги

1. **[DONE] Найти root cause CUDA crash** — DLL Singleton Problem
2. **[TODO] Пересобрать проект** с новым CUDAAllocator.cpp
3. **[TODO] Протестировать** test_mem_leak.exe --device cuda

## Команды для тестирования фикса

```batch
REM 1. Очистить старую сборку
cd C:\Users\paper\Desktop\promethorch
rmdir /s /q build_cuda_examples

REM 2. Создать новую сборку
mkdir build_cuda_examples
cd build_cuda_examples

REM 3. Открыть Developer Command Prompt for VS 2019 и выполнить:
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM 4. Конфигурация с CUDA
cmake .. -DPT_USE_CUDA=ON -DPT_BUILD_SHARED_LIBS=ON

REM 5. Сборка
nmake test_mem_leak

REM 6. Тест
set PATH=%CD%;C:\ProgramData\anaconda3\Library\bin;%PATH%
examples\pir\test_mem_leak.exe --device cuda --iterations 20
```

## Важные находки

- **CPU autograd работает корректно** - нет утечек узлов
- **12 nodes на итерацию** (MLP: 3 Linear = 6 Mm + 6 Add = 12)
- **11ms на итерацию** на CPU для batch=32, hidden=256

---

Полная документация: `CLAUDE.md`
