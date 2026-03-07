# AVOIDRECURSION - СТОП ФАЙЛ

**ПРОЧИТАЙ ЭТО ПЕРЕД ЛЮБЫМ ДЕЙСТВИЕМ!**

---

## СТАТИСТИКА БЕСПОЛЕЗНЫХ ПОВТОРЕНИЙ

| Что | Количество |
|-----|------------|
| Build директорий | **34 штуки** |
| Log файлов (.txt) | **195 штук** |
| Batch файлов (.bat) | **90+ штук** |
| Попыток rb_mnist*.txt | **9 за один день** |
| Попыток mlp*.txt | **6 за один день** |
| Попыток rebuild_final*.txt | **5 штук** |

---

## РАБОЧИЕ СБОРКИ (НЕ ПЕРЕСОБИРАТЬ!)

### CPU (для тестов autograd):
```
build_final3/examples/mnist/train_mnist_mlp.exe  # 25 янв 11:49 - НОВЕЙШАЯ
build_examples/examples/mnist/train_mnist_mlp.exe  # 25 янв 10:04
```

### Запуск:
```bash
cd /c/Users/paper/Desktop/promethorch
PATH="./build_final3:$PATH" ./build_final3/examples/mnist/train_mnist_mlp.exe --device cpu --epochs 1
```

---

## ЦИКЛ КОТОРЫЙ НЕЛЬЗЯ ПОВТОРЯТЬ

```
❌ "Проблема в backward"
   ↓
❌ "Проверю MmBackward / CrossEntropy"
   ↓
❌ "Пересоберу с правильным CMake"
   ↓
❌ rc.exe не найден из bash
   ↓
❌ Найду существующую сборку
   ↓
❌ Тест → 12% accuracy
   ↓
❌ "Проблема в backward" ← НАЧАЛО ЗАНОВО
```

---

## ЧТО НЕ ДЕЛАТЬ

1. **НЕ ПЕРЕСОБИРАТЬ** - сборки уже есть
2. **НЕ запускать CMake из bash** - rc.exe не найдётся
3. **НЕ создавать новые build директории** - их уже 34
4. **НЕ создавать новые batch файлы** - их уже 90+
5. **НЕ писать "следующие шаги" без их выполнения**
6. **НЕ проверять "какой CMake использовать"** - см. ниже

---

## CMAKE (ФИНАЛЬНЫЙ ОТВЕТ)

### Правильный:
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
```

### Неправильный (НЕ ИСПОЛЬЗОВАТЬ):
```
C:\ProgramData\anaconda3\Lib\site-packages\cmake\data\bin\cmake.exe
```

### Но сборка из bash НЕ РАБОТАЕТ из-за rc.exe!
Решение: использовать существующие сборки.

---

## РЕАЛЬНАЯ ПРОБЛЕМА

**MNIST accuracy: 12-15% вместо 49%**

Сборка работает. Forward работает. Backward вызывается.
**Градиенты вычисляются НЕПРАВИЛЬНО.**

---

## ЧТО НУЖНО СДЕЛАТЬ (КОНКРЕТНО)

### Шаг 1: Прочитать backward функции
```
torch/csrc/autograd/functions/MathBackward.h
torch/csrc/autograd/functions/LinearAlgebraBackward.h
```

### Шаг 2: Найти баг в одной из:
- `CrossEntropyBackward` - формула `(softmax - one_hot) / batch_size`
- `MmBackward` - формула `grad_A = grad_C @ B^T`
- `ReluBackward` - маска `grad * (input > 0)`
- `TransposeBackward` - перестановка градиентов

### Шаг 3: Сравнить с PyTorch численно
Запустить `test_mlp_gradient.py` и сравнить числа.

---

## КЛЮЧЕВЫЕ ФАЙЛЫ

| Файл | Что содержит |
|------|--------------|
| `torch/csrc/autograd/functions/MathBackward.h` | CrossEntropyBackward, ReluBackward |
| `torch/csrc/autograd/functions/LinearAlgebraBackward.h` | MmBackward, MatmulBackward |
| `torch/nn/modules/linear.h` | Linear forward (исправлен) |
| `torch/optim/adamkiller.h` | AdamKiller (исправлен) |
| `test_mlp_gradient.py` | PyTorch референс для сравнения |
| `test_full_mlp.py` | Полный тест PyTorch MLP |

---

## GPU

**GPU ЗАНЯТ! ВСЕ ТЕСТЫ НА CPU!**

```bash
--device cpu
```

---

## ЕСЛИ ХОЧЕТСЯ ПЕРЕСОБРАТЬ

**СТОП! Сборки уже есть!**

Если ОЧЕНЬ нужно - только из Developer Command Prompt:
1. Start Menu → "x64 Native Tools Command Prompt for VS 2019"
2. Там выполнять команды

---

## КОНТРОЛЬНЫЕ ВОПРОСЫ ПЕРЕД ДЕЙСТВИЕМ

1. Я уже делал это раньше? → Проверь логи
2. Есть ли рабочая сборка? → ДА, build_final3
3. Нужно ли пересобирать? → НЕТ
4. Я читал backward код? → Если нет - ЧИТАЙ
5. Я сравнил числа с PyTorch? → Если нет - СРАВНИ

---

## ИСТОРИЯ ИЗМЕНЕНИЙ КОДА

### Исправлено:
1. `linear.h:57` - bound = 1/sqrt(fan_in) вместо sqrt(3)/sqrt(fan_in)
2. `adamkiller.h:266` - step_size = layer_lr вместо layer_lr/bc1

### ПРОВЕРЕНО (2026-01-25, код прочитан):
1. **MmBackward** - формула ПРАВИЛЬНАЯ: `grad_A = grad_C @ B^T`, `grad_B = A^T @ grad_C`
2. **TransposeBackward** - формула ПРАВИЛЬНАЯ: `grad.transpose(dim0, dim1)`
3. **ReluBackward** - формула ПРАВИЛЬНАЯ: `grad * (self > 0)`
4. **CrossEntropyBackward** - формула ВЫГЛЯДИТ правильной: `(softmax - one_hot) / N`

### ГДЕ ИСКАТЬ БАГ:
Backward функции выглядят правильно. Проблема может быть в:
1. **Как подключаются backward** - проверить `autograd.h` функции `mm_autograd`, `t_autograd`
2. **Как создаётся граф** - проверить что edges правильно связываются
3. **SGD optimizer** - проверить что step() правильно применяет градиенты
4. **Порядок операций в Linear.forward()** - проверить что `x @ W^T + b` правильно

---

---

## ПОСЛЕДНЕЕ ДЕЙСТВИЕ (2026-01-25)

Прочитаны файлы:
- `LinearAlgebraBackward.h` - MmBackward, TransposeBackward выглядят ОК
- `MathBackward.h` - ReluBackward, CrossEntropyBackward выглядят ОК

**Следующий шаг:** Проверить `autograd.h` - функции `mm_autograd()`, `t_autograd()`
(как backward подключается к forward операциям)

---

## ФИНАЛЬНОЕ НАПОМИНАНИЕ

**Проблема НЕ в сборке. Проблема В КОДЕ autograd.**

1. Backward формулы выглядят правильно
2. Нужно проверить КАК они подключаются (mm_autograd и т.д.)
3. Или сравнить числа напрямую с PyTorch

Читай код. Сравнивай числа. Не пересобирай.
