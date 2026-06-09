# Сборка PromeTorch — единый вход

Два способа: **CMake presets** (простые платформы) и **build-скрипты**
(Эльбрус, где нужен runtime-детект ISA/железа до конфигурации).

## Быстрый старт через presets (CMake ≥ 3.21)

```bash
cmake --preset linux-x86_64      # configure
cmake --build --preset linux-x86_64
```

Доступные presets (`cmake --list-presets`):

| Preset | Платформа | Toolchain |
|--------|-----------|-----------|
| `linux-x86_64` | Linux x86_64 (gcc, AVX2/FMA) | native |
| `cuda` | Linux x86_64 + CUDA | native + nvcc |
| `windows-cpu` | Windows x64 (MSVC) | VS 2022 |
| `windows-cuda` | Windows x64 + CUDA/cuDNN | VS 2022 |
| `alt-x86_64` | ALT Linux | `cmake/toolchains/x86_64-alt.cmake` |
| `astra-x86_64` | Astra Linux | `x86_64-astra.cmake` |
| `redos-x86_64` | RED OS | `x86_64-redos.cmake` |
| `baikal-m` | Baikal-M aarch64 (cross) | `aarch64-baikal-m.cmake` |
| `baikal-s` | Baikal-S aarch64 (cross) | `aarch64-baikal-s.cmake` |
| `elbrus` | Elbrus E2K (lcc) | `e2k-elbrus.cmake` |
| `elbrus-no-eml` | Elbrus без EML (OpenBLAS/TUDA) | `e2k-elbrus.cmake` |

> Presets используют формат v3 (нужен CMake ≥ 3.21). Базовый
> `cmake_minimum_required` проекта = 3.18 — это для прямой сборки без
> presets. На Эльбрусах CMake 3.28, всё работает.

## Эльбрус — через скрипт (рекомендуется)

`cmake --preset elbrus` работает, но **`scripts/build-elbrus.sh`
предпочтительнее** на реальном железе: он делает то, что статичный preset
не может —

1. **Auto-detect ISA** (`detect_e2k_march`): `lcc --version` → elbrus-v3/v4/v5/v6,
   экспортит `PT_E2K_MARCH` до configure. Без этого 8C2 собирается под v4
   вместо v5 → qpmaddubsh выключается → TP-4 падает 10.9 → 5 tok/s (баг,
   который мы ловили 2026-05-20).
2. **Pre-flight compile-test** компилятора.
3. **`--no-eml`** для E16C (libeml_mt помечен elbrus-2c3, bug #10075) →
   OpenBLAS fallback (= `--preset elbrus-no-eml`).
4. **loginctl enable-linger** + sanity-run.

```bash
./scripts/build-elbrus.sh                    # auto всё
PT_TARGETS="test_gguf_inference promeserve" ./scripts/build-elbrus.sh
./scripts/build-elbrus.sh --no-eml           # E16C: форсить OpenBLAS
```

Аналоги для других RU-ОС: `build-alt.sh`, `build-astra.sh`, `build-redos.sh`
(идентичны кроме имени; эквивалент presets `{alt,astra,redos}-x86_64`).
Кросс на Baikal: `build-baikal-cross.sh`.

## Выбор BLAS-провайдера (Эльбрус)

НЕ задаётся preset'ом — определяется в рантайме configure
(`CMakeLists.txt`, трёхуровневый каскад):

1. **EML** (`libeml_mt`) — лучший на 8C2 (1840 GFLOPS, 4 NUMA). try_run
   ELF-probe; если несовместим (E16C elbrus-2c3) → шаг 2.
2. **OpenBLAS** (`libopenblas-*.e2kv6`) — нативный fallback на E16C.
3. **TUDA 6×6 микроядро** — последний fallback, ~30× медленнее, но
   работает без внешних libs.

Override: `-DPT_USE_EML_BLAS=OFF` (или `--no-eml` в скрипте),
`-DPT_USE_OPENBLAS=OFF`, `-DPT_EML_RUNTIME_OK=ON` (форсить EML если
знаешь что работает, в обход cross-compile guard).

## Python (pip)

```bash
pip install prometorch          # CPU wheel с PyPI (когда опубликовано)
pip install -e .                # из исходников (scikit-build-core)
```

См. `docs/PACKAGING.md`.

## Полная инфо по ISA × intrinsics × BLAS

`docs/elbrus_isa/PERFORMANCE_BY_ISA.md` — таблица v3/v4/v5/v6, какие
intrinsics доступны, runtime-требования, реальные tok/s.
