# Модульное разделение PromeTorch — реалистичный план (HEAD 59822c7)

**Дата:** 2026-06-03
**Контекст:** внешний консультант (DeepSeek) предложил split на `core/ apps/ deploy/ python/`
с `add_library(prometorch_core SHARED src/*.cpp)`. Этот документ оценивает РЕАЛЬНУЮ
применимость к нашей кодовой базе на основе фактического CMake-графа и раскладки файлов.

---

## 1. Карта текущих компонентов (фактическая, из CMakeLists.txt)

| Target | Тип | .cpp файлов | Назначение | Линкуется на |
|--------|-----|-------------|------------|--------------|
| `c10` | **SHARED** (или STATIC) | 4 (`Allocator/Device/TensorImpl/Exception`) | Ядро: Allocator, Device, Storage, TensorImpl, ScalarType | — |
| `aten_cpu` | **STATIC** | 1 (`hot_loops.cpp`) | Скомпилированные горячие циклы CPU (LTO на E2K) + BLAS-линковка | `c10` |
| `aten` | **INTERFACE** | 0 | Зонтик: headers + aten_cpu + опц. бэкенды | `c10 aten_cpu` (+ cuda/nmcard/...) |
| `aten_cuda` | **SHARED** | 9 `.cu` + `CUDAAllocator.cpp` | CUDA/ROCm kernels | `c10` |
| `torch_autograd` | **STATIC** | 2 (`grad_mode.cpp`, `ddp.cpp`) | Autograd engine (header) + DLL-safe GradMode singleton + DDP | `aten` |
| `torch_nn` | **INTERFACE** | 0 | NN-модули | `torch_autograd` |
| `torch_optim` | **INTERFACE** | 0 | Оптимизаторы (20 шт) | `torch_nn` |
| `torch_data` | **INTERFACE** | 0 | DataLoader/Sampler | `torch_optim` |
| `torch_extras` | **INTERFACE** | 0 | ONNX/JIT/MLIR/vision/quant/distributed/serve | `torch_data` |
| `aten_nmcard` / `aten_linq` / `aten_mps` | SHARED (опц.) | 2–3 каждый | Экзотические бэкенды | `c10` |
| `_C` | pybind MODULE | 6 (`python/csrc/*.cpp`) | Python bindings | `c10 torch_data` (+aten_cuda) |
| `promeserve` | EXE | 1 (`main.cpp`) | HTTP inference server, остальное header-only | `c10 aten torch_autograd` |
| examples (mnist/pir/gguf/...) | EXE | по 1 .cpp | Демо/тренировка | `aten torch_*` |

**Ключевой факт:** весь линейный «торч-стек» (`torch_autograd → nn → optim → data → extras`)
несёт ровно **2 .cpp файла**. 127 заголовков `torch/` — header-only inline по дизайну.
Реальный компилируемый код ядра: 4 (c10) + 1 (aten_cpu) + 2 (autograd) = **7 .cpp**.

---

## 2. Реалистичный target-layout (БЕЗ переписывания header-only в .cpp)

Раскладка по каталогам, повторяющая реальность, а не догму DeepSeek:

```
core/      = c10 + aten_cpu + aten(INTERFACE) + torch_autograd + torch_nn +
             torch_optim + torch_data + torch_extras
             → собирается в libprometorch_core (SHARED от c10 уже есть;
               aten_cpu/torch_autograd — STATIC, линкуются внутрь)
backends/  = aten_cuda, aten_nmcard, aten_linq, aten_mps (опциональные плагины)
apps/      = promeserve (server), examples/* (train/infer демо)
python/    = python/csrc/_C (УЖЕ отдельно)
deploy/    = Dockerfile, Dockerfile.cuda, docker/, scripts/, pyproject.toml
```

Это **переразмещение каталогов + переименование CMake-агрегата**, а НЕ рефактор кода.
Сами библиотеки уже существуют и уже разделены по зависимостям — DeepSeek фактически
предлагает то, что у нас в значительной мере уже сделано, только под другими именами.

---

## 3. ABI / PT_API export macro — блокер?

Уточнение к аудиту «8 из 200+ классов». Фактически:
- `PT_API` **определён корректно** в `c10/macros/Macros.h` (dllexport/dllimport/visibility).
- В `c10/` он применён к **~25 классам** (TensorImpl, Storage, StorageImpl, Device,
  DeviceGuard, Allocator, DataPtr, CPUAllocator, AllocatorRegistry, Generator,
  вся иерархия Error/Warning). **Ядро c10 покрыто хорошо.**
- Пробел не в c10, а в **`torch/`**: autograd `Node`/`Edge`/`Engine`, все nn-модули,
  оптимизаторы — **0 вхождений PT_API**.

**Почему это сегодня НЕ блокер:** `torch/` header-only. Inline-классы из заголовков
инстанцируются в каждой TU потребителя — им export-macro не нужен. Граница DLL пересекается
только для c10 (она покрыта) и для специально помеченных launcher'ов в aten_cuda
(`aten_cuda_exports.def` + ATEN_CUDA_API).

**Блокер появится РОВНО ТОГДА**, когда мы захотим вынести логику `torch/` в
`libprometorch_core.so` как настоящий .cpp-код (то, что предлагает DeepSeek). Тогда на MSVC
каждый класс на границе DLL потребует `PT_API`. Оценка работы: пометить ~40–60 публичных
классов в `torch/csrc/autograd` + `torch/nn` + `torch/optim` — **2–4 дня**, плюс отладка
шаблонных символов (Tensor — шаблонно-тяжёлый, dllexport шаблонов на MSVC = боль). Это и есть
главная причина НЕ делать .cpp-вынос ради него самого.

---

## 4. Что из плана DeepSeek НЕ применимо

1. **`add_library(prometorch_core SHARED src/*.cpp)`** — у нас нет `src/*.cpp` с логикой ядра.
   Логика в 127 header-only `.h`. Эта директива собрала бы пустую библиотеку из 7 файлов.
   Предложение исходит из ложной посылки «.cpp-heavy core».
2. **«Вынести core в одну .so»** — для `torch/` потребует масштабного навешивания PT_API
   (см. §3) + ODR-конфликтов шаблонов на MSVC. Высокий риск, цель — чисто косметическая.
3. **Жёсткое деление `core/ apps/ deploy/`** как первичная цель — мы уже имеем граф
   зависимостей (c10←aten←autograd←nn←optim←data). Перемещение файлов ломает 30+ build-скриптов
   (`build_*.bat`) и пути в examples, не добавляя функциональности.

Что ПРИМЕНИМО: вынос `apps/` (promeserve + examples) и `deploy/` (Docker/scripts) —
они уже почти изолированы (promeserve линкует только `c10 aten torch_autograd`,
python/csrc уже отдельный).

---

## 5. Минимальный полезный шаг (80% пользы за 20% работы)

НЕ трогая код и build-скрипты, ввести **CMake-агрегат без перемещения файлов**:

1. `add_library(prometorch_core INTERFACE)` → линкует `torch_extras` (вершина цепочки),
   даёт потребителям одно имя `PromeTorch::core` вместо знания о 8 таргетах. **~30 минут.**
2. `add_library(prometorch_backends INTERFACE)` → агрегирует включённые
   aten_cuda/nmcard/linq/mps по флагам. **~30 минут.**
3. Документировать слои в `docs/ARCHITECTURE.md`: core / backends / apps / python / deploy
   как **логические** слои поверх существующих таргетов. **~1 час.**
4. Чистка корня репо: 30+ `build_*.bat`, 60+ `vit_*.log`, `*.txt`-логи, orphan `test_*.py`
   (status git показывает их untracked) → в `deploy/legacy_scripts/` и `.gitignore`.
   Это даёт бОльшую часть субъективной «модульности» (чистый корень). **~2 часа.**

Итого минимальный шаг: **полдня**, нулевой риск для inference-пути (святые 11.4 tok/s
не затрагиваются — ни один .cpp/.h не меняется).

---

## 6. Оценка трудозатрат

| Объём | Что входит | Время | Риск |
|-------|-----------|-------|------|
| **Минимальный** (рекоменд.) | INTERFACE-агрегаты `core`/`backends` + ARCHITECTURE.md + чистка корня | ~0.5 дня | нулевой |
| **Средний** | + физическое перемещение promeserve→`apps/`, Docker→`deploy/`, правка путей в их CMake | +1 день | низкий (apps уже изолированы) |
| **Полный (DeepSeek)** | + вынос `torch/` логики в `libprometorch_core.so` .cpp + PT_API на 40–60 классов + MSVC шаблонные символы | +1–2 недели | **высокий** (ODR/шаблоны/ABI на 3 платформах: MSVC, LCC E2K, gcc) |

**Рекомендация:** делать «минимальный», при необходимости «средний». «Полный» вариант
DeepSeek НЕ оправдан: header-only `torch/` — осознанный дизайн (быстрая инлайн-оптимизация,
кросс-компиляция под E2K/LCC без забот об ABI). .so-вынос даёт только косметику ценой
двухнедельного ABI-марафона на трёх компиляторах.

---

## Вывод

DeepSeek-план построен на ложной посылке «.cpp-heavy core». Реальность: ядро = 7 .cpp,
логика — header-only inline в 127 `.h`. Граф зависимостей УЖЕ модульный
(c10→aten→autograd→nn→optim→data→extras), бэкенды УЖЕ плагины по флагам, python УЖЕ отдельно,
promeserve УЖЕ изолирован. Полезное действие — не рефактор, а тонкий слой агрегирующих
INTERFACE-таргетов + наведение порядка в корне репозитория (полдня, нулевой риск).
ABI/PT_API — не блокер сегодня и станет им только если форсировать ненужный .so-вынос.
