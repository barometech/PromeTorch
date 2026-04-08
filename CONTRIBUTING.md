# Contributing to PromeTorch

Мы рады вашему вкладу в развитие суверенного фреймворка глубокого обучения!

## Как начать

1. Сделайте форк репозитория
2. Создайте ветку: `git checkout -b feature/my-feature`
3. Соберите проект (см. README.md)
4. Запустите тесты: `cd build && ctest --output-on-failure`
5. Сделайте коммит и отправьте Pull Request

## Правила кода

- **C++17** стандарт
- Избегайте аллокаций в hot-path (используйте Memory Pool / Arena)
- Для новых операций: добавьте SIMD-векторизацию через TUDA
- Для CUDA: используйте `WARP_MASK` вместо hardcoded `0xffffffff` (AMD совместимость)
- Для backward: обязательно `reduce_grad()` при broadcasting
- Тесты обязательны для новых операций

## Структура проекта

```
c10/          — Ядро (Allocator, Device, Storage, TensorImpl)
aten/         — Тензорные операции (CPU, CUDA, NMCard)
torch/        — Autograd, NN modules, Optimizers
examples/     — Примеры обучения
test/cpp/     — C++ тесты (18 файлов)
python/       — Python bindings (pybind11)
promeserve/   — LLM inference сервер
```

## Сборка и тесты

```bash
mkdir build && cd build
cmake .. -DPT_BUILD_TESTS=ON -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

## Поддерживаемые платформы

- x86_64 Linux (GCC 7.5+)
- x86_64 Windows (MSVC 2019+)
- Elbrus E2K (LCC 1.25+)
- NVIDIA CUDA (11.0+)
- NM Card Mini (NMCSDK)
