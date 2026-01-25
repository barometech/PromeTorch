# CUDA Crash Investigation

**Ветка:** `research/cuda-crash-investigation`
**Статус:** В ПРОЦЕССЕ
**Приоритет:** КРИТИЧЕСКИЙ

## Симптомы

- **Exit code:** -1073740791 (0xC0000409 = STATUS_STACK_BUFFER_OVERRUN)
- **Когда:** Первая итерация обучения на CUDA
- **Где:** После `Model moved to CUDA`, до первого `iter 1: loss=...`
- **CPU:** Работает идеально (20 итераций, 0 утечек)

## Гипотезы

1. **to_cuda() операция** - проблема при переносе тензора на GPU
2. **CUDA forward pass** - проблема в GEMM/Add kernels
3. **CUDA backward pass** - проблема в backward kernels
4. **Memory allocation** - CUDACachingAllocator corruption
5. **cuBLAS/cuDNN** - неправильная инициализация handles

## План исследования

### Фаза 1: Изоляция проблемы
- [ ] Тест 1: Только to_cuda() без forward
- [ ] Тест 2: Только forward без backward
- [ ] Тест 3: Forward + backward без optimizer
- [ ] Тест 4: Полный цикл

### Фаза 2: Проверка компонентов
- [ ] CUDACachingAllocator - проверить allocate/free
- [ ] at::to_cuda() - проверить копирование данных
- [ ] CUDA GEMM - проверить матричное умножение
- [ ] CUDA backward - проверить backward kernels

### Фаза 3: Исправление
- [ ] Найти корневую причину
- [ ] Исправить
- [ ] Протестировать
- [ ] Мерж в main

## Логи

### 2026-01-24: Начало исследования
- Создана ветка research/cuda-crash-investigation
- CPU работает: 20 итераций, created=240, destroyed=240
- CUDA crash на первой итерации
