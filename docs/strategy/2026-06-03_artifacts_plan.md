# План артефактов и упаковки PromeTorch (2026-06-03)

HEAD = 59822c7. Оценка предложения DeepSeek (CPack deb/rpm/tar.gz/NSIS + PyPI + Docker Hub)
против того, что **уже есть в репозитории**.

---

## 1. Что УЖЕ есть и работает ли

| Артефакт | Файл | Состояние |
|----------|------|-----------|
| PyPI metadata | `pyproject.toml` | Готов. scikit-build-core + pybind11, target=`_C` only, packages=`python/prometorch`. CPU-only baseline. |
| cibuildwheel | `pyproject.toml [tool.cibuildwheel]` | Настроен: cp39–cp313, x86_64+arm64 (`auto64`), manylinux_2_28, skip musl/i686/win32/pp. test-command импортит пакет. |
| CI wheels | `.github/workflows/wheels.yml` | Полный pipeline: sdist + matrix (3 OS × 5 Py) + Trusted Publishing (OIDC) на PyPI и TestPyPI. **НЕ проверено что Trusted Publisher зарегистрирован на pypi.org** — это ручной шаг (см. PACKAGING.md). До регистрации publish-job упадёт. |
| GitHub Release | `.github/workflows/release.yml` | На тег `v*` — только `git archive` tar.gz исходников. PromeServe-бинарь НЕ прикладывается. |
| Elbrus build | `scripts/build-elbrus.sh` + `cmake/toolchains/e2k-elbrus.cmake` | Рабочий build-on-target: auto-detect компилятора (LCC/gcc-elbrus), **auto-detect ISA** (`detect_e2k_march` → v3/v4/v5/v6), EML/OpenBLAS/TUDA fallback, sanity-тест. |
| Docker (RU OS) | `docker/Dockerfile.{alt,astra,redos,elbrus,baikal,wheel}` + `build-all.sh` | Это **build-проверки на debian-эмуляции**, НЕ настоящие RU-base-образы и НЕ публикуемые рантайм-образы. `elbrus-sim.yml` гоняет их в CI. |
| Toolchains | `cmake/toolchains/{e2k-elbrus,aarch64-baikal-m/s,x86_64-alt/astra/redos}.cmake` | Есть для всех целей. |
| CMake install/export | `CMakeLists.txt:1646–1697` | `install(EXPORT PromeTorchTargets)` + Config.cmake — т.е. `find_package(PromeTorch)` работает после `make install`. |
| PromeServe | `promeserve/` (main.cpp, *.h, web/index.html, README) | Standalone бинарь, Ollama-совместимый API, web UI. Собирается, но **НЕ имеет install()-правила**, web/ никуда не копируется, в release не входит. |

**Чего НЕТ вообще:** CPack (0 упоминаний), .deb/.rpm/NSIS/dmg, Docker Hub push, single source of truth для версии.

---

## 2. Таблица: платформа → формат → доставка → инфра

| Платформа | Формат | Способ доставки | Инфра? |
|-----------|--------|-----------------|--------|
| x86_64 / arm64 Linux | manylinux wheel | `pip install prometorch` | Есть (cibuildwheel+CI), нужна регистрация Trusted Publisher |
| Windows x64 | wheel | `pip install` | Есть |
| macOS x86_64 | wheel | `pip install` | Есть (arm64 test skip — собирается, не тестируется) |
| Любая → from source | sdist tar.gz | `pip install` (соберёт локально) | Есть |
| Эльбрус e2k (v3/v4/v5/v6) | git clone + build | `./scripts/build-elbrus.sh` | Есть, **лучший путь** (см. §3) |
| Baikal-M/S (aarch64) | wheel ИЛИ from source | sdist / toolchain | Частично (toolchain есть, в cibuildwheel arm64-linux идёт через QEMU — медленно) |
| Alt / Astra / RedOS (x86) | wheel (manylinux совместим) ИЛИ source | `pip install` | Есть |
| NM Quad / NM Card | отдельный SDK | git clone + `-DPT_USE_NMQUAD=ON` | Не pip, проприетарные deps — **правильно что отдельно** |
| PromeServe (любая) | бинарь + web/ | GitHub Release asset | **НЕТ** (см. §4) |

---

## 3. Проблема 4 ISA для Эльбруса

Реально: один бинарь не покроет v3/v4/v5/v6 — `qpmaddubsh` (v5+) физически отсутствует
на v4, а сборка под v6 на v4 не запустится. Варианты:

| Вариант | Оценка |
|---------|--------|
| **A. Build-on-target** (`build-elbrus.sh`) | **РЕКОМЕНДУЕТСЯ.** Уже работает. `detect_e2k_march` сам выбирает ISA хоста, EML/OpenBLAS/TUDA выбирается рантайм-детектом. Юзеру нужны только lcc + eml-devel (одна `apt-get`). Ноль артефактов на нашей стороне. |
| B. 4 предсобранных tar.gz (e2k-v3/v4/v5/v6) | Дорого: нет публичного e2k-CI-раннера → каждый собирать вручную на железе. BLAS-провайдер всё равно нужно линковать статически или авто-детектить. Оправдано только если у клиента нет lcc. |
| C. Fat-binary | LCC не делает мультиверсионный e2k fat-ELF из коробки; function-multiversioning по ISA на e2k не поддержано. **Отпадает.** |

Вывод: для Эльбруса PyPI-wheel невозможен (нет lcc на воркерах PyPI — это и так зафиксировано
в PACKAGING.md). Канон — **build-on-target**, плюс опционально 1 tar.gz под v5 (8C2, самое
распространённое железо) как «скачал-распаковал» для клиентов без компилятора.

---

## 4. PromeServe как отдельный release asset

Сейчас PromeServe собирается, но не упаковывается. Минимальный артефакт:
`promeserve(.exe)` + `web/index.html` + `README.md` в одном `.tar.gz`/`.zip`.

Нужно:
1. Добавить `install(TARGETS promeserve RUNTIME DESTINATION bin)` и
   `install(DIRECTORY web/ DESTINATION share/promeserve/web)` в `promeserve/CMakeLists.txt`.
2. В `release.yml` добавить job: собрать promeserve под win-x64 и linux-x64, упаковать
   с `web/`, приложить к GitHub Release как asset.
3. (Опц.) e2k-вариант — собирать на железе, прикладывать вручную.

Это самый ценный артефакт для аудитории «скачал и запустил LLM» — один бинарь, web UI,
Ollama-совместимый API, без Python.

---

## 5. Версионирование — single source of truth

**Текущий drift (3 источника, 3 значения):**

| Источник | Значение |
|----------|----------|
| `pyproject.toml` / `python/prometorch/__init__.py` | `0.1.0a1` |
| `CMakeLists.txt` PROJECT + `c10/macros/Macros.h` | `0.1.0` |
| `python/csrc/init.cpp:254` (`_C.__version__`) | **`0.2.0`** |

Т.е. `import prometorch; prometorch.__version__` даёт `0.1.0a1`, а `prometorch._C.__version__`
даёт `0.2.0`. PromeServe отдаёт ещё своё `0.1.0`.

**Фикс:**
1. SSOT = `pyproject.toml`. Удалить хардкод `__version__` в `__init__.py` →
   читать `importlib.metadata.version("prometorch")`.
2. Удалить хардкод `"0.2.0"` в `init.cpp` (или прокидывать `PT_VERSION_STRING` из CMake-define).
3. CMake `project(... VERSION)` и `Macros.h` генерировать из одного места (configure_file
   `Macros.h.in` ← `${PROJECT_VERSION}`), а PROJECT_VERSION задавать = pyproject (или наоборот
   читать pyproject в CMake). Минимум — синхронизировать вручную и пометить комментарием SSOT.

---

## 6. Что из DeepSeek переоценено

| Предложение | Вердикт |
|-------------|---------|
| NSIS-инсталлятор (Windows) | **Не нужно.** Аудитория — разработчики/НИИ; `pip install` + zip с promeserve.exe закрывает 100%. Installer-GUI = поддержка ради поддержки. |
| .dmg (macOS) | **Не нужно.** macOS у целевой аудитории (RU CPU/ускорители) почти нет; wheel достаточно. |
| CPack deb/rpm как основной канал | **Переоценено для C++-библиотеки.** Имеет смысл точечно: (а) PromeServe как `.deb`/`.rpm` для Astra(deb)/Alt(rpm) — удобно админам; (б) dev-пакет с хедерами для `find_package`. Но не приоритет №1. |
| Docker Hub публикация рантайм-образов | **Условно.** Текущие Dockerfile — это CI-build-проверки на debian, не RU-base рантайм. Для RU-OS нужны реальные base-образы (Astra/Alt registry), debian-образ на e2k не запустится. Польза только для x86-демо. |
| Один универсальный бинарь под Эльбрус | **Невозможно** (4 ISA, нет fat-ELF) — см. §3. |

DeepSeek прав в основном: PyPI + wheel + sdist (уже есть), tar.gz исходников (есть).
Переоценил desktop-инсталляторы и недооценил build-on-target для e2k.

---

## 7. Минимальный путь «скачал-запустил» по аудиториям

| Аудитория | Путь |
|-----------|------|
| Python-разработчик (x86/arm/win) | `pip install prometorch` (после регистрации Trusted Publisher) |
| Хочет LLM-сервер, без Python | Скачать `promeserve` + `web/` из GitHub Release, `./promeserve --device cpu` |
| Эльбрус | `git clone && ./scripts/build-elbrus.sh` (нужны lcc + eml-devel) |
| Astra/Alt/RedOS x86 | `pip install prometorch` (manylinux совместим) |
| NM Quad/Card | `git clone && cmake -DPT_USE_NMQUAD=ON` (отдельный SDK) |

---

## 8. Effort estimate

| Задача | Объём |
|--------|-------|
| Фикс version drift (§5, 3 файла + опц. configure_file) | S |
| Регистрация Trusted Publisher на PyPI/TestPyPI + smoke на TestPyPI | S (ручной, вне кода) |
| PromeServe install-rules + release-asset job (§4) | M |
| 1 tar.gz e2k-v5 как «скачал-распаковал» (опц.) | M (ручная сборка на железе) |
| PromeServe .deb/.rpm через CPack (опц., точечно) | M |
| Реальные RU-base Docker-образы (опц.) | L (нужен доступ к Astra/Alt registry) |

**Приоритет:** §5 (version) → Trusted Publisher → §4 (PromeServe release). Остальное — опционально.
NSIS/dmg — не делать.
