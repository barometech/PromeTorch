# ADAM-KILLER: Оптимизатор нового поколения

**Цель:** Создать оптимизатор, превосходящий Adam в 4-10 раз по скорости сходимости.

---

## Почему Adam можно превзойти?

### Ограничения Adam:
1. **Фиксированные β1=0.9, β2=0.999** — не адаптируются к ландшафту потерь
2. **Одинаковый lr для всех слоёв** — глубокие и мелкие слои требуют разных lr
3. **Не использует информацию о кривизне** — только первый момент (среднее) и второй (дисперсия)
4. **Momentum может замедлять** — в седловых точках momentum идёт не туда
5. **Bias correction медленная** — первые 100-1000 шагов неоптимальны

---

## Идеи для ADAM-KILLER

### 1. Адаптивный Momentum (Curvature-Aware β1)

```python
# Вместо фиксированного β1=0.9:
curvature = |grad_t - grad_{t-1}| / lr  # приближение второй производной
β1_adaptive = clip(0.5 + 0.4 * (1 - curvature/max_curv), 0.5, 0.99)
# Высокая кривизна → меньший momentum (быстрее реагируем)
# Низкая кривизна → больший momentum (ускоряемся)
```

### 2. Per-Layer Learning Rates

```python
# Автоматическая настройка lr для каждого слоя
layer_grad_norm = ||grad_layer||
layer_weight_norm = ||weight_layer||
layer_lr = base_lr * (layer_weight_norm / layer_grad_norm)
# Нормализация: градиенты масштабируются относительно весов
```

### 3. Gradient Prediction (Lookahead на стероидах)

```python
# Предсказываем следующий градиент для упреждающего обновления
grad_predicted = 2 * grad_t - grad_{t-1}  # линейная экстраполяция
# Или используем EMA разности:
delta_grad = β * delta_grad + (1-β) * (grad_t - grad_{t-1})
grad_predicted = grad_t + delta_grad
```

### 4. Hessian-Free Second Order

```python
# Приближение диагонали Гессиана через конечные разности
H_diag ≈ (grad(θ + ε*v) - grad(θ - ε*v)) / (2*ε)
# где v = sign(grad) — направление возмущения
# Используем для адаптивного масштабирования
```

### 5. Warm Restarts с Cosine Annealing

```python
# Периодические "перезапуски" для выхода из локальных минимумов
if step % restart_period == 0:
    reset_momentum_buffers()
    lr = lr_max  # начинаем с высокого lr
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * step / restart_period))
```

---

## Архитектура ADAM-KILLER

### Класс `AdamKiller`:

```cpp
struct AdamKillerOptions {
    float lr = 0.001f;           // базовый learning rate
    float beta1_min = 0.5f;      // минимальный momentum
    float beta1_max = 0.99f;     // максимальный momentum
    float beta2 = 0.999f;        // EMA для v (как в Adam)
    float eps = 1e-8f;
    float weight_decay = 0.01f;  // decoupled как в AdamW
    int restart_period = 1000;   // период warm restart
    bool use_gradient_prediction = true;
    bool use_per_layer_lr = true;
    bool use_adaptive_momentum = true;
};

class AdamKiller : public Optimizer {
    struct ParamState {
        Tensor m;           // первый момент
        Tensor v;           // второй момент
        Tensor prev_grad;   // предыдущий градиент
        Tensor grad_diff;   // EMA разности градиентов
        float layer_lr_scale;  // масштаб lr для этого слоя
        int64_t step;
    };

    void step() {
        for (auto& param : params_) {
            auto& state = states_[param];
            Tensor grad = param->grad();

            // 1. Gradient prediction
            if (options_.use_gradient_prediction && state.prev_grad.defined()) {
                Tensor delta = grad - state.prev_grad;
                state.grad_diff = beta_diff * state.grad_diff + (1 - beta_diff) * delta;
                grad = grad + state.grad_diff;  // lookahead
            }

            // 2. Adaptive momentum (curvature-aware β1)
            float beta1 = options_.beta1_max;
            if (options_.use_adaptive_momentum && state.prev_grad.defined()) {
                float curvature = (grad - state.prev_grad).abs().mean().item<float>() / options_.lr;
                beta1 = clip(options_.beta1_min + 0.4f * (1.0f - curvature),
                            options_.beta1_min, options_.beta1_max);
            }

            // 3. Standard Adam updates
            state.m = beta1 * state.m + (1 - beta1) * grad;
            state.v = options_.beta2 * state.v + (1 - options_.beta2) * grad.square();

            // 4. Bias correction
            float bc1 = 1.0f - pow(beta1, state.step + 1);
            float bc2 = 1.0f - pow(options_.beta2, state.step + 1);
            Tensor m_hat = state.m / bc1;
            Tensor v_hat = state.v / bc2;

            // 5. Per-layer learning rate
            float lr = options_.lr;
            if (options_.use_per_layer_lr) {
                float grad_norm = grad.norm().item<float>();
                float weight_norm = param->data().norm().item<float>();
                if (grad_norm > 1e-8f) {
                    lr = options_.lr * (weight_norm / grad_norm);
                    lr = clip(lr, options_.lr * 0.1f, options_.lr * 10.0f);
                }
            }

            // 6. Weight decay (decoupled)
            if (options_.weight_decay > 0) {
                param->data().sub_(param->data(), lr * options_.weight_decay);
            }

            // 7. Update
            param->data().sub_(m_hat / (v_hat.sqrt() + options_.eps), lr);

            // 8. Warm restart
            if (options_.restart_period > 0 && state.step % options_.restart_period == 0) {
                state.m.zero_();
                // v не сбрасываем - важно для stability
            }

            state.prev_grad = grad.clone();
            state.step++;
        }
    }
};
```

---

## Бенчмарки

### Тесты для сравнения:

1. **MNIST MLP** (784→512→256→128→10)
   - Adam baseline: ~97% за 5 epochs
   - Цель AdamKiller: ~97% за 1-2 epochs

2. **CIFAR-10 ResNet-18**
   - Adam baseline: ~90% за 100 epochs
   - Цель AdamKiller: ~90% за 25-50 epochs

3. **Rosenbrock function** (оптимизация)
   - Adam: ~10000 итераций до сходимости
   - Цель AdamKiller: ~2000 итераций

4. **Saddle point escape**
   - Adam: застревает или медленно выходит
   - Цель AdamKiller: быстрый выход благодаря adaptive momentum

---

## План реализации

### Шаг 1: Базовая структура
- [ ] Создать `torch/optim/adamkiller.h`
- [ ] Реализовать базовый Adam как отправную точку
- [ ] Добавить state tracking (prev_grad, grad_diff)

### Шаг 2: Adaptive Momentum
- [ ] Реализовать curvature estimation
- [ ] Реализовать adaptive β1
- [ ] Тесты на простых функциях

### Шаг 3: Per-Layer LR
- [ ] Реализовать layer_lr_scale
- [ ] Тесты на deep networks

### Шаг 4: Gradient Prediction
- [ ] Реализовать lookahead с EMA
- [ ] Тесты на oscillating gradients

### Шаг 5: Warm Restarts
- [ ] Реализовать periodic reset
- [ ] Тесты на local minima escape

### Шаг 6: Бенчмарки
- [ ] MNIST comparison
- [ ] CIFAR-10 comparison
- [ ] Rosenbrock comparison

---

## Файлы для создания

```
torch/optim/adamkiller.h        # Основная реализация
examples/benchmark_optimizers.cpp   # Сравнительные тесты
docs/ADAM_KILLER_RESULTS.md     # Результаты экспериментов
```

---

## Литература

1. **Adam** - Kingma & Ba, 2014
2. **AdamW** - Loshchilov & Hutter, 2017
3. **Lookahead** - Zhang et al., 2019
4. **RAdam** - Liu et al., 2019
5. **LAMB** - You et al., 2019
6. **Shampoo** - Gupta et al., 2018 (second-order)
7. **K-FAC** - Martens & Grosse, 2015 (curvature)

---

**Начало работы:**
```bash
cd /path/to/promethorch
# Создать adamkiller.h и начать с базовой реализации
```
