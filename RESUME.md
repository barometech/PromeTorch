# PromeTorch - Resume Point

**Дата:** 2026-01-24
**Ветка:** `feature/memory-leak-fix`

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

### В работе: Утечка GPU памяти

**Проблема:** При обучении PIR модели на CUDA память растёт ~2GB/итерация.

**Диагностика:**
- Создан `examples/test_mem_leak.cpp` - минимальный MLP для тестирования
- **CPU:** Работает без утечек (created=destroyed, alive=0)
- **CUDA:** Crash на первой итерации (heap corruption -1073740791)

**Вывод:** Проблема специфична для CUDA-кода, не для autograd.

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

1. **Отладить CUDA crash** - найти где происходит heap corruption
2. **Проверить CUDA kernels** - возможно проблема в gemm или reduce
3. **Добавить CUDA sync** - возможно race condition

## Важные находки

- **CPU autograd работает корректно** - нет утечек узлов
- **12 nodes на итерацию** (MLP: 3 Linear = 6 Mm + 6 Add = 12)
- **11ms на итерацию** на CPU для batch=32, hidden=256

---

Полная документация: `CLAUDE.md`
