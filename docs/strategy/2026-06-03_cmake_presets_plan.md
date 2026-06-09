# Унификация системы сборки PromeTorch через CMakePresets.json + toolchains

**Дата:** 2026-06-03 · **HEAD:** 59822c7 · **Автор:** build-audit agent

Реальный план под наш код (НЕ generic). Учитывает поправки на ошибки внешнего
консультанта (DeepSeek), который видел только README.

---

## 0. Что у нас есть сейчас (факты из репо)

**Build-скрипты (`scripts/*.sh`):**

| Скрипт | Toolchain (уже есть!) | Уникальное содержимое |
|--------|------------------------|------------------------|
| `build-elbrus.sh` | `cmake/toolchains/e2k-elbrus.cmake` | **Толстый.** Compiler auto-detect (lcc/lcc++/l++/gcc-elbrus), `detect_e2k_march()` (lcc --version + lscpu + /proc/cpuinfo → v3/v4/v5/v6), `check_header`/`check_lib` через реальный compile-test, `--no-eml` флаг, EML-зависимости, single `cmake --build` со всеми targets, sanity-run `tuda_tests_standalone`. |
| `build-alt.sh` | `cmake/toolchains/x86_64-alt.cmake` | Тонкий. gcc/g++, проверка наличия 4 бинарей, configure+build. |
| `build-astra.sh` | `cmake/toolchains/x86_64-astra.cmake` | Идентичен alt, отличается только TOOLCHAIN/BUILD_DIR/echo. |
| `build-redos.sh` | `cmake/toolchains/x86_64-redos.cmake` | Идентичен alt. |
| `build-baikal-cross.sh` | `aarch64-baikal-{m,s}.cmake` | Cross. Выбор варианта baikal-m/baikal-s по `$1`, проверка `aarch64-linux-gnu-gcc`. |

Три x86 скрипта (alt/astra/redos) **байт-в-байт совпадают** кроме имён — чистое
дублирование. Их toolchain-файлы тоже идентичны (одинаковые `-march=x86-64-v3 -mavx2 -mfma`).

**Windows:** ~100 `.bat` в корне (`build_cuda.bat`, `build_cpu.bat`, `build_python_cmake.bat`...).
Это ad-hoc generator `-G "NMake Makefiles"` + `vcvarsall.bat x64` + Anaconda CUDA paths.
Не трогаем массово (см. §5), но даём один канонический Windows-preset.

**pyproject.toml:** scikit-build-core уже задаёт свои `cmake.define` (PT_BUILD_PYTHON=ON,
тесты/CUDA/NMCard OFF) и `build.targets=["_C"]`. Это **параллельный** способ конфигурации,
который НЕ должен конфликтовать с presets.

**Ключевая логика в root `CMakeLists.txt` (которую DeepSeek не видел):**
- BLAS — трёхуровневый runtime auto-detect (§3): EML → OpenBLAS → TUDA. С `try_run`
  ELF-probe (EML на E16C v6 помечен elbrus-2c3, bug #10075 → loader блокирует).
- `-mtune` НЕ передаётся вообще; опционально через `PT_E2K_MTUNE` (это фикс T2, см. §4).
- E2K intrinsics — `pt_detect_e2k_intrinsics()` через `try_compile` под текущий `-march`.
- На E2K авто-включается OpenMP, `-O3 -ffast -faligned -fprefetch -frestrict-all` и пр.
- Toolchain `e2k-elbrus.cmake` читает `$ENV{PT_E2K_MARCH}` и кладёт в `-march`.

---

## 1. Таблица: build-*.sh → preset + toolchain

CMakePresets группирует **configurePreset** (toolchain + cacheVars) и
**buildPreset** (targets + jobs). Toolchain-файлы переиспользуем как есть.

| Текущий скрипт | configurePreset name | toolchainFile | Новый/существующий toolchain |
|----------------|----------------------|---------------|------------------------------|
| build-elbrus.sh | `elbrus` | `cmake/toolchains/e2k-elbrus.cmake` | существует |
| build-alt.sh | `alt` | `cmake/toolchains/x86_64-alt.cmake` | существует |
| build-astra.sh | `astra` | `cmake/toolchains/x86_64-astra.cmake` | существует |
| build-redos.sh | `redos` | `cmake/toolchains/x86_64-redos.cmake` | существует |
| build-baikal-cross.sh m | `baikal-m` | `cmake/toolchains/aarch64-baikal-m.cmake` | существует |
| build-baikal-cross.sh s | `baikal-s` | `cmake/toolchains/aarch64-baikal-s.cmake` | существует |
| build_cuda.bat (Win) | `windows-cuda` | — (нет toolchain, generator+CUDA define) | новый preset, без toolchain |
| build_cpu.bat (Win) | `windows-cpu` | — | новый preset |
| (нет) | `elbrus-no-eml` | e2k-elbrus.cmake | inherits `elbrus` + `PT_USE_EML_BLAS=OFF` |
| (нет, x86 dev) | `linux-x86-dev` | — | hidden base для alt/astra/redos |

x86 alt/astra/redos сводятся к **одному** `linux-x86` base preset (toolchains всё
равно идентичны); отдельные имена оставляем как тонкие inherit-обёртки только ради
имени build-каталога и понятности — реальной разницы во флагах нет.

---

## 2. Структура `cmake/toolchains/*.cmake` + `CMakePresets.json`

### 2.1 Toolchains — НЕ переписывать

Все 7 toolchain-файлов остаются как есть. Единственное предлагаемое улучшение:
вынести три идентичных x86-файла (alt/astra/redos) в один `x86_64-linux-gnu.cmake`
и сделать остальные `include()`-обёртками — НО это опционально и низкоприоритетно
(дублирование тривиальное, риск что-то сломать > выгода). По умолчанию **оставляем 7**.

### 2.2 CMakePresets.json (в корне репо). Реальные НАШИ флаги:

```jsonc
{
  "version": 3,                       // CMake 3.21+. У нас cmake_minimum 3.18 —
  "cmakeMinimumRequired": {           // подними до 3.21 ТОЛЬКО для presets-фичи,
    "major": 3, "minor": 21, "patch": 0  // либо version:2 (3.20) если 3.21 нет на Эльбрусе.
  },
  "configurePresets": [
    {
      "name": "base", "hidden": true,
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "PT_USE_TUDA": "ON",
        "PT_USE_LINQ": "OFF",
        "PT_BUILD_TESTS": "ON",
        "PT_BUILD_SHARED_LIBS": "ON"
      }
    },
    {
      "name": "elbrus", "inherits": "base",
      "toolchainFile": "${sourceDir}/cmake/toolchains/e2k-elbrus.cmake",
      "cacheVariables": {
        "PT_USE_CUDA": "OFF", "PT_USE_NMCARD": "OFF",
        "PT_BUILD_PYTHON": "OFF"
        // PT_USE_EML_BLAS НЕ задаём здесь — пусть runtime auto-detect (§3)
      }
    },
    {
      "name": "elbrus-no-eml", "inherits": "elbrus",
      "cacheVariables": { "PT_USE_EML_BLAS": "OFF" }   // = старый --no-eml
    },
    {
      "name": "linux-x86", "hidden": true, "inherits": "base"
      // toolchain выставляется в alt/astra/redos
    },
    { "name": "alt",   "inherits": "linux-x86",
      "toolchainFile": "${sourceDir}/cmake/toolchains/x86_64-alt.cmake" },
    { "name": "astra", "inherits": "linux-x86",
      "toolchainFile": "${sourceDir}/cmake/toolchains/x86_64-astra.cmake" },
    { "name": "redos", "inherits": "linux-x86",
      "toolchainFile": "${sourceDir}/cmake/toolchains/x86_64-redos.cmake" },
    { "name": "baikal-m", "inherits": "base",
      "toolchainFile": "${sourceDir}/cmake/toolchains/aarch64-baikal-m.cmake" },
    { "name": "baikal-s", "inherits": "base",
      "toolchainFile": "${sourceDir}/cmake/toolchains/aarch64-baikal-s.cmake" },
    {
      "name": "windows-cuda",
      "generator": "NMake Makefiles",
      "binaryDir": "${sourceDir}/build_cuda",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "PT_USE_CUDA": "ON", "PT_USE_CUDNN": "ON", "PT_BUILD_TESTS": "OFF",
        // Anaconda CUDA — пути из CLAUDE.md (НЕ хардкодим в toolchain, тут ок)
        "CMAKE_CUDA_COMPILER": "C:/ProgramData/anaconda3/Library/bin/nvcc.exe",
        "CUDAToolkit_ROOT": "C:/ProgramData/anaconda3/Library"
      }
      // ВАЖНО: требует запуска из vcvarsall.bat x64 окружения (rc.exe).
    },
    {
      "name": "windows-cpu",
      "generator": "NMake Makefiles",
      "binaryDir": "${sourceDir}/build_cpu",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release", "PT_USE_CUDA": "OFF" }
    }
  ],
  "buildPresets": [
    { "name": "elbrus", "configurePreset": "elbrus", "jobs": 0,
      "targets": ["aten_cpu","torch_autograd","tuda_tests_standalone",
                  "tuda_tests","test_gguf_inference"] },
    { "name": "elbrus-no-eml", "configurePreset": "elbrus-no-eml",
      "targets": ["aten_cpu","torch_autograd","tuda_tests_standalone"] },
    { "name": "alt",   "configurePreset": "alt",   "jobs": 0 },
    { "name": "astra", "configurePreset": "astra", "jobs": 0 },
    { "name": "redos", "configurePreset": "redos", "jobs": 0 },
    { "name": "baikal-m", "configurePreset": "baikal-m", "jobs": 0 },
    { "name": "baikal-s", "configurePreset": "baikal-s", "jobs": 0 },
    { "name": "windows-cuda", "configurePreset": "windows-cuda" }
  ]
}
```

`jobs: 0` = «все ядра» (эквивалент `--parallel $(nproc)` из скриптов).
`targets` в elbrus-buildPreset воспроизводит `DEFAULT_TARGETS` из build-elbrus.sh
(минимальный набор без gguf/promeserve/examples overhead).

---

## 3. Как preset выбирает BLAS-провайдера

**Решение: НЕ передавать провайдера через preset cache var. Оставить runtime
auto-detect в CMakeLists.txt.** Причины (это и есть то, что DeepSeek не знал):

1. Выбор EML/OpenBLAS/TUDA зависит от **рантайма целевой машины**, а не от того,
   какой preset выбран. На 8C2 EML работает, на E16C v6 тот же preset должен сам
   упасть на OpenBLAS через `try_run` ELF-probe. Preset не знает на какой машине
   запущен (особенно при cross-compile).
2. CMakeLists уже делает это правильно: EML `find_library`+`try_run` → при fail
   `PT_USE_EML_BLAS=OFF` → OpenBLAS `find_library` → иначе TUDA. Дублировать это
   решение в preset = развести два источника правды.

**Что preset МОЖЕТ задавать (опт-ауты, не выбор):**
- `elbrus-no-eml` preset → `PT_USE_EML_BLAS=OFF` (явный отказ, = `--no-eml`).
  Это override верхнего уровня каскада, а не «выбор провайдера».
- Опционально завести `elbrus-openblas` (EML=OFF, OPENBLAS=ON) для E16C если
  не хотим тратить время на заведомо-провальный EML-probe.

**Cross-compile нюанс (есть в коде):** при `CMAKE_CROSSCOMPILING` `try_run` падает,
поэтому CMakeLists пропускает EML-probe и идёт на OpenBLAS; override —
`-DPT_EML_RUNTIME_OK=ON`. Это можно выставить в preset для конкретной cross-цели,
но по умолчанию НЕ выставляем.

Вывод: **BLAS остаётся runtime auto-detect; preset трогает только верхние опт-ауты.**

---

## 4. Что НЕЛЬЗЯ переносить из DeepSeek

| Предложение DeepSeek | Почему НЕЛЬЗЯ |
|----------------------|---------------|
| `-mtune=elbrus-8c` (хардкод) | **Это ровно баг T2, который мы убили.** `-mtune` принимает только имена моделей, разные у каждого LCC-релиза; хардкод ломал сборку на E8C/E16C. У нас `-mtune` не передаётся вообще, опционально через `PT_E2K_MTUNE`. В preset НЕ добавлять `-mtune`. |
| `-march=thomas` для NM Card | **Галлюцинация.** NM Quad/NM Card — это DSP NMC4 с отдельным SDK (`libnm_quad_load`, `nmpp`, opcodes), НЕ gcc/lcc-таргет. Нет такого `-march`. NMCard включается через `PT_USE_NMCARD`/`PT_USE_NMQUAD`, а не toolchain. |
| `crossbuild-essential-e2k` (apt-пакет) | **Не существует.** На Эльбрусе компилятор = LCC (`lcc`/`l++`), ставится `lcc-c++`/`gcc-elbrus` (на Альт) или идёт в составе MCST PDK. Никакого debian `crossbuild-essential-e2k` нет. Cross-build x86→e2k у нас вообще не практикуется (собираем нативно на w205p/e16c). |
| gcc как компилятор Эльбруса | На Эльбрусе основной компилятор LCC. `gcc-elbrus` — только запасной вариант. Toolchain `e2k-elbrus.cmake` уже делает auto-detect lcc→l++→gcc-elbrus→gcc; не упрощать до `gcc`. |
| Один универсальный toolchain | Нельзя — у каждой ISA/арки свои `-march` (elbrus-vN vs armv8.2 vs x86-64-v3) и свои опт-ауты AVX/CUDA. |

---

## 5. Migration path

**Принцип: presets — единый источник флагов; build-*.sh остаются тонкими обёртками.**
Не удалять скрипты — на Эльбрусе есть логика, которой нет в CMake (см. ниже).

1. **Добавить `CMakePresets.json`** (§2) — аддитивно, ничего не ломает.
2. **`build-alt.sh`/`build-astra.sh`/`build-redos.sh`** → тело сводится к:
   ```bash
   cmake --preset alt && cmake --build --preset alt
   ```
   плюс existing pre-flight проверка бинарей. Дублирование флагов исчезает.
3. **`build-baikal-cross.sh`** → `cmake --preset baikal-$VARIANT && cmake --build --preset baikal-$VARIANT`.
   Сохранить проверку наличия `aarch64-linux-gnu-gcc`.
4. **`build-elbrus.sh` — НЕ растворять в preset.** Скрипт делает то, что preset
   принципиально не умеет:
   - `detect_e2k_march()` (lcc --version / lscpu / cpuinfo) — **bash, рантайм-детект
     железа.** Результат экспортится в `PT_E2K_MARCH`, который читает toolchain.
     Это нельзя выразить в статичном preset.
   - compiler auto-detect с **полным путём** (CMake требует full path к lcc++).
   - `check_header`/`check_lib` pre-flight (compile-test EML/omp/numa).
   - `--no-eml` CLI.

   Решение: build-elbrus.sh сохраняет detection-преамбулу, но финальный configure
   меняет на:
   ```bash
   PT_E2K_MARCH=$(detect_e2k_march) \
   cmake --preset ${USE_EML:+elbrus}${USE_EML:-elbrus-no-eml} \
         -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
         ${PT_E2K_MTUNE:+-DPT_E2K_MTUNE=$PT_E2K_MTUNE}
   cmake --build --preset elbrus
   ```
   Т.е. preset забирает все статичные cacheVars, а скрипт остаётся ответственным
   за рантайм-детект железа и компилятора.
5. **Windows .bat** — НЕ мигрировать массово (~100 шт, многие — одноразовые
   run-скрипты). Завести `windows-cuda`/`windows-cpu` presets и переписать
   ТОЛЬКО канонические `build_cuda.bat`/`build_cpu.bat` на `cmake --preset` внутри
   `vcvarsall.bat`-окружения. Остальные .bat — отдельная зачистка (вне scope).
6. **pyproject.toml** — оставить `[tool.scikit-build.cmake.define]` как есть.
   scikit-build-core НЕ использует наш CMakePresets автоматически. Опционально
   позже: завести `python-wheel` preset и сослаться через
   `cmake.args = ["--preset","python-wheel"]`, но это не обязательно и может
   конфликтовать с `build-dir`/`build.targets` — НЕ в первой итерации.

**Версия CMake:** presets v3 требует CMake ≥3.21. На w205p — CMake 3.28 (из MEMORY),
на Эльбрусе из CLAUDE — 3.28; Windows BuildTools — 3.x свежий. Безопасно. Поднять
`cmake_minimum_required` в root до 3.21 (сейчас 3.18) ИЛИ оставить 3.18 в проекте
и полагаться на `cmakeMinimumRequired` внутри presets (CMake применит его только
когда вызван через `--preset`). Рекомендую второй вариант — не трогать min для
не-preset потребителей.

---

## 6. Оценка трудозатрат

| Шаг | Effort |
|-----|--------|
| Написать `CMakePresets.json` (§2) | 1–2 ч |
| Переписать 3 x86 + 2 baikal скрипта в обёртки | 1 ч |
| Перепаять `build-elbrus.sh` configure-вызов (сохранив detect) | 1–2 ч |
| `windows-cuda`/`windows-cpu` presets + 2 канонических .bat | 1 ч |
| Тест-прогон: alt/astra локально, Эльбрус на w205p, E16C на e16c.ru, Windows CUDA | 2–4 ч (зависит от доступности железа) |
| **Итого** | **~1 рабочий день + прогоны на заёмном железе** |

Низкий риск: изменения аддитивные, toolchains не трогаем, BLAS-каскад не трогаем.
Главная проверка — что `cmake --preset elbrus` на w205p даёт те же `-march`/BLAS,
что и текущий `build-elbrus.sh` (мерить tok/s: TP-4 qwen3-4b должен остаться 10.9,
порог отката >0.2 — святое правило скорости).
