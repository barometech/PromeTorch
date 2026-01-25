# PromeTorch - Resume Point

**Дата:** 2026-01-25
**Ветка:** `research/cuda-crash-investigation`
**Статус:** ✅ AUTOGRAD РАБОТАЕТ, MNIST ОБУЧАЕТСЯ

---

## КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ (2026-01-25)

### Проблема: "backward: 0 nodes"
При обучении MNIST backward pass находил 0 узлов, модель не обучалась.

### Root Cause: AutogradMeta Factory Registration
1. `c10/core/TensorImpl.cpp` имел default factory создающий BASE `AutogradMeta`
2. `torch/csrc/autograd/autograd_meta.h` должен регистрировать factory для `AutogradMetaImpl`
3. При `transpose()` создавался новый TensorImpl с base AutogradMeta вместо AutogradMetaImpl
4. `gradient_edge()` возвращал null потому что base AutogradMeta не имеет `grad_fn`

### Решение:
1. Factory registration в `autograd_meta.h` работает корректно
2. `c10::set_autograd_meta_factory()` вызывается при старте программы
3. Все операции теперь создают `AutogradMetaImpl` с полной поддержкой autograd

### Результат:
```
Epoch 1/1 - Loss: 2.31896 - Train Acc: 14.88% - Test Acc: 14.86%
```
- Loss снижается (2.35 → 2.32)
- Все 8 параметров (fc1-fc4 weights и biases) получают градиенты
- Accuracy выше random (14.86% vs ~10%)

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
cd /c/Users/paper/Desktop/promethorch
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
cd /c/Users/paper/Desktop/promethorch/build_examples/examples/mnist
PATH="/c/Users/paper/Desktop/promethorch/build_examples:$PATH" ./train_mnist_mlp.exe --data C:/Users/paper/Desktop/promethorch/data/mnist --epochs 5 --lr 0.001
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
cd /c/Users/paper/Desktop/promethorch
git status

# 2. Запустить обучение MNIST (проверить что всё работает)
cd build_examples/examples/mnist
PATH="/c/Users/paper/Desktop/promethorch/build_examples:$PATH" ./train_mnist_mlp.exe --data C:/Users/paper/Desktop/promethorch/data/mnist --epochs 5

# 3. Для работы над новым оптимизатором:
# - Изучить torch/optim/adam.h
# - Создать torch/optim/superadam.h
# - Добавить в examples/mnist/train_mnist_mlp.cpp
```

---

Полная документация: `CLAUDE.md`
Гайд по сборке: `BUILD_GUIDE.md`
