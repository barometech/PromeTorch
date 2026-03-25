# PromeTorch - Resume Point

**Дата:** 2026-03-02
**Ветка:** `fix/adam-optimizer`
**Статус:** ✅ MNIST РАБОТАЕТ — Test Accuracy 88.94%

---

## ИСПРАВЛЕНО (2026-03-02): mm() contiguous fix

### Root cause
`mm()` в `LinearAlgebra.h` читала `data_ptr<>()` с контигуозными индексами, но `tensor.t()` создаёт VIEW с транспонированными strides. Fix: `.contiguous()` перед вычислениями.

### Результат
- **Test Accuracy: 88.94%** (1 epoch, SGD lr=0.01, batch 64)
- Loss: 2.30 → 1.12
- Train Acc: 71.05%

### Ранее исправлено:
1. **Linear initialization**: `bound = 1/sqrt(fan_in)` (не Kaiming)
2. **AdamKiller bias correction**: `step_size = layer_lr`

---

## КРИТИЧНО: Сборка

### Правильный CMake (ИСПОЛЬЗОВАТЬ ТОЛЬКО ЭТОТ!):
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
```

### НЕ использовать CMake из Anaconda:
```
C:\ProgramData\anaconda3\Lib\site-packages\cmake\data\bin\cmake.exe  # НЕПРАВИЛЬНО!
```

### Проблема с bash:
При запуске batch файлов из bash, `rc.exe` не находится. Решение:
- Использовать Developer Command Prompt for VS 2019
- Или использовать существующие сборки

### Рабочие сборки (CPU):
- `build_final3/examples/mnist/train_mnist_mlp.exe` - от 25 янв 11:49
- `build_examples/examples/mnist/train_mnist_mlp.exe` - от 25 янв 10:04

---

## ТЕСТЫ ТОЛЬКО НА CPU!

GPU занят, все тесты выполнять на CPU:
```bash
./train_mnist_mlp.exe --device cpu --epochs 1 --lr 0.01 --batch_size 64
```

---

## Текущий статус

### ✅ Завершено (14 фаз)
- c10 core (Allocator, Device, Storage, TensorImpl)
- ATen tensor operations (MathOps, ReduceOps, LinearAlgebra, ShapeOps)
- Autograd (backward engine, 50+ backward functions) **РАБОТАЕТ!**
- NN Modules (50+ слоёв: Linear, Conv, BatchNorm, Transformer, PIR...)
- Optimizers (SGD, Adam, AdamW, RMSprop)
- LR Schedulers (10+ видов)
- Data Loading (Dataset, DataLoader, Sampler)
- CUDA Backend (собственные kernels)
- cuDNN Integration
- FlashAttention
- Mixed Precision (AMP)
- Python Bindings (pybind11)

### ✅ MNIST MLP Training
- 4-слойная MLP: 784 → 512 → 256 → 128 → 10
- Backward pass работает через все слои
- Градиенты корректно распространяются

---

## Рабочая сборка

### Директория: `build_examples`

### Сборка (из bash):
```bash
cd /path/to/promethorch
start //b rebuild_with_sdk.bat
```

### Или из Developer Command Prompt:
```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\ProgramData\anaconda3\Library\bin;%PATH%
cd C:\Users\paper\Desktop\promethorch\build_examples
nmake train_mnist_mlp
```

### Запуск:
```bash
cd /path/to/promethorch/build_examples/examples/mnist
PATH="/path/to/promethorch/build_examples:$PATH" ./train_mnist_mlp.exe --data /path/to/promethorch/data/mnist --epochs 5 --lr 0.001
```

---

## Ключевые файлы (изменены 2026-01-25)

| Файл | Изменение |
|------|-----------|
| `c10/core/TensorImpl.cpp` | Удалён debug output |
| `torch/csrc/autograd/autograd_meta.h` | Factory registration, удалён debug |
| `torch/csrc/autograd/engine.h` | Удалён debug из backward() |
| `examples/mnist/train_mnist_mlp.cpp` | Восстановлена 4-слойная MLP, lr=0.001 |

---

## Следующий проект: АДАМ-KILLER

Цель: создать оптимизатор превосходящий Adam в 4-10 раз.

Идеи:
1. **Адаптивный momentum** - не фиксированный β1, а зависящий от кривизны
2. **Curvature-aware scaling** - использовать приближённую Гессиану
3. **Per-layer learning rates** - автоматическая настройка lr для каждого слоя
4. **Gradient prediction** - предсказывать следующий градиент для ускорения
5. **Warm restarts + lookahead** - комбинация техник

---

## Команды для продолжения

```bash
# 1. Проверить текущий статус
cd /path/to/promethorch
git status

# 2. Запустить обучение MNIST (проверить что всё работает)
cd build_examples/examples/mnist
PATH="/path/to/promethorch/build_examples:$PATH" ./train_mnist_mlp.exe --data /path/to/promethorch/data/mnist --epochs 5

# 3. Для работы над новым оптимизатором:
# - Изучить torch/optim/adam.h
# - Создать torch/optim/superadam.h
# - Добавить в examples/mnist/train_mnist_mlp.cpp
```

---

Полная документация: `CLAUDE.md`
Гайд по сборке: `BUILD_GUIDE.md`
**КРИТИЧНО: `AVOIDRECURSION.md` - ЧИТАТЬ ПЕРЕД ЛЮБЫМ ДЕЙСТВИЕМ!**
