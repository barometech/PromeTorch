# Аудит #13 — Build matrix реальности

**HEAD:** `85c0fb5` · **Дата:** 2026-06-02 · **Метод:** статический parsing `CMakeLists.txt` (root 1738 строк), `examples/*/CMakeLists.txt` (14 шт.), `promeserve/`, `tools/`, `benchmarks/`, `scripts/build-*.sh`, `cmake/toolchains/*`. Без запуска сборки.

## 1. Easy axes / правила определения OS×ISA

| Колонка | Что значит |
|---|---|
| **Win/MSVC** | `WIN32 + MSVC + CMAKE_SYSTEM_PROCESSOR=AMD64` |
| **Lin/GCC x86** | `CMAKE_SYSTEM_NAME=Linux + x86_64` (ALT/Astra/RED OS — один профиль) |
| **E2K v3** | toolchain `e2k-elbrus.cmake`, `PT_E2K_MARCH=elbrus-v3` (E4C/E2C) |
| **E2K v4** | E8C, `elbrus-v4` (script default) |
| **E2K v5** | E8C2/8СВ, `elbrus-v5` (auto-detected на 8C2) |
| **E2K v6** | E16C, `elbrus-v6` |
| **Lin ARM** | `aarch64-baikal-{m,s}.cmake` (Cortex-A57/A75) |

CMake-only кросс-ось guards в репо: `if(MSVC)`, `if(WIN32)`, `if(UNIX AND NOT APPLE)`, `if(NOT MSVC)`, `if(NOT WIN32)`, `if(APPLE)`, `if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k|elbrus")`, `if(CMAKE_SYSTEM_NAME STREQUAL "Linux")`, `if(PT_USE_CUDA)`, `if(PT_USE_ROCM)`, `if(PT_USE_NMCARD)`, `if(PT_USE_NMQUAD)`, `if(PT_USE_LINQ)`, `if(PT_USE_MPS)`, `if(PT_USE_TUDA)`, `if(PT_BUILD_TESTS)`, `if(PT_BUILD_GENERATED_TESTS)`, `if(PT_BUILD_DEEPSPEED_TESTS)`, `if(EXISTS ...)`. ROCm взаимоисключающ с CUDA (FATAL_ERROR строка 21-26).

## 2. Targets × платформа

Легенда: ✓ = собирается при default-флагах скрипта/среды, • = optional (нужен PT_USE_XXX=ON или CUDA SDK), ⊘ = guard не пускает, ☒ = код есть но не зарегистрирован.

### Core libs (root CMakeLists.txt, безусловно при наличии C++ компилятора)

| Target | Файл (строка) | Win/MSVC | Lin x86 | E2Kv3 | E2Kv4 | E2Kv5 | E2Kv6 | ARM | Примечание |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `c10` (SHARED/STATIC) | root:542-545 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Switch SHARED↔STATIC через `PT_BUILD_SHARED_LIBS` |
| `aten_cpu` (STATIC) | root:703 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | На E2K линкуется к EML/OpenBLAS/TUDA в зависимости от probe |
| `aten` (INTERFACE) | root:817 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Umbrella |
| `aten_cuda` (SHARED/STATIC) | root:599/605 | • | • | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | `PT_USE_CUDA=ON` + nvcc; на E2K toolchain принудительно `OFF` |
| `aten_cuda` (HIP path) | root:673/676 | ⊘ | • | ⊘ | ⊘ | ⊘ | ⊘ | • | `PT_USE_ROCM=ON`; same target name (overload) |
| `aten_nmcard` | root:850/853 | • | • | • | • | • | • | • | `PT_USE_NMCARD=ON` |
| `aten_linq` | root:888/891 | • | • | • | • | • | • | • | `PT_USE_LINQ=ON` |
| `aten_mps` | root:939/942 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | `PT_USE_MPS=ON` + APPLE (нет таких сборок в репо) |
| `torch_autograd` (STATIC) | root:1001 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Подключает POSIX socket DDP вне Windows |
| `torch_nn` (INTERFACE) | root:1043 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | header-only |
| `torch_optim` (INTERFACE) | root:1078 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | header-only |
| `torch_data` (INTERFACE) | root:1100 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | header-only |
| `torch_extras` (INTERFACE) | root:1161 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ONNX/JIT/Mobile/MLIR/Vision/Audio/Text headers |
| `_C` (pybind11 module) | root:1549 | • | • | • | • | • | • | • | `PT_BUILD_PYTHON=ON`; на E2K скрипт принудительно OFF |

### Tests (gated `PT_BUILD_TESTS=ON`, default ON)

| Target | Файл | Win | Lin x86 | E2K | ARM | Гард |
|---|---|:-:|:-:|:-:|:-:|---|
| `c10_tests` | root:1191 | ✓ | ✓ | ✓ | ✓ | gtest discover |
| `aten_tests` (test_tensor.cpp) | root:1215 | ✓ | ✓ | ✓ | ✓ | |
| `autograd_tests` | root:1230 | ✓ | ✓ | ✓ | ✓ | |
| `nn_tests` | root:1244 | ✓ | ✓ | ✓ | ✓ | standalone, не gtest |
| `optim_tests` | root:1260 | ✓ | ✓ | ✓ | ✓ | |
| `optimizer_tests` | root:1274 | ✓ | ✓ | ✓ | ✓ | |
| `data_tests` | root:1288 | ✓ | ✓ | ✓ | ✓ | |
| `all_ops_tests` | root:1304 | ✓ | ✓ | ✓ | ✓ | |
| `test_ops_generated` | root:1323 | • | • | • | • | `PT_BUILD_GENERATED_TESTS=OFF` |
| `autograd_full_tests` | root:1340 | ✓ | ✓ | ✓ | ✓ | |
| `nn_modules_tests` | root:1354 | ✓ | ✓ | ✓ | ✓ | |
| `attention_tests` | root:1372 | ✓ | ✓ | ✓ | ✓ | |
| `nn_functional_tests` | root:1388 | ✓ | ✓ | ✓ | ✓ | |
| `edge_case_tests` | root:1408 | ✓ | ✓ | ✓ | ✓ | |
| `ddp_tests` | root:1423 | ✓-сборка, ⊘-add_test | ✓ | ✓ | ✓ | `add_test` только not WIN32 |
| `fsdp_tests` | root:1432 | ✓-build, ⊘-test | ✓ | ✓ | ✓ | |
| `gguf_ddp_tests` | root:1444 | ✓ | ✓ | ✓ | ✓ | не зарегистрирован в `add_test` |
| `deepspeed_tests` | root:1464 | • | • | • | • | `PT_BUILD_DEEPSPEED_TESTS=OFF` |
| `nmcard_tests` | root:1475 | • | • | • | • | `PT_USE_NMCARD=ON` |
| `linq_tests` | root:1487 | • | • | • | • | `PT_USE_LINQ=ON` |
| `tuda_tests` | root:1499 | ✓ | ✓ | ✓ | ✓ | `PT_USE_TUDA=ON` (default) |
| `tuda_tests_standalone` | root:1515 | ✓ | ✓ | ✓ | ✓ | sanity-check для всех ISA |

### Examples (foreach loop root:1610-1614 + nmcard root:1615-1617)

| Target | Файл | Win | Lin x86 | E2K | ARM | Гард |
|---|---|:-:|:-:|:-:|:-:|---|
| `train_mnist_mlp` | examples/mnist:5 | ✓ | ✓ | ✓ | ✓ | универсал |
| `train_mnist_cnn` | examples/mnist:22 | ✓ | ✓ | ✓ | ✓ | |
| `train_mnist_cnn_autograd` | examples/mnist:39 | ✓ | ✓ | ✓ | ✓ | |
| `train_mnist` | examples/mnist:56 | ✓ | ✓ | ✓ | ✓ | legacy |
| `train_10_models` | examples/mnist:73 | ✓ | ✓ | ✓ | ✓ | |
| `test_phase2` | examples/mnist:90 | ✓ | ✓ | ✓ | ✓ | |
| `test_cuda_debug` | examples/mnist:107 | • | • | ⊘ | ⊘ | требует `PT_USE_CUDA` для линка, имя misleading |
| `train_vae` | examples/vae:4 | ✓ | ✓ | ✓ | ✓ | |
| `test_mobile` | examples/mobile:3 | ✓ | ✓ | ✓ | ✓ | |
| `test_jit_compile` | examples/jit:3 | ✓ | ✓ (-ldl) | ✓ (-ldl) | ✓ | |
| `test_mlir_export` | examples/mlir:3 | ✓ | ✓ | ✓ | ✓ | |
| `train_rnn` | examples/rnn:5 | ✓ | ✓ | ✓ | ✓ | |
| `train_rnn_full` | examples/rnn:19 | ✓ | ✓ | ✓ | ✓ | |
| `train_transformer` | examples/transformer:3 | ✓ | ✓ | ✓ | ✓ | |
| `test_transformer_cuda_forward` | tests/test_transformer_cuda_forward.cpp | ✓ | ✓ | ✓ | ✓ | требует `PT_USE_CUDA` для runtime, иначе noop |
| `train_vit` | examples/vit:3 | ✓ | ✓ | ✓ | ✓ | |
| `train_resnet` (CIFAR) | examples/cifar:4 | ✓ | ✓ | ✓ | ✓ | без `PT_USE_CUDNN` flag из-за compile issue |
| `train_pir_elbrus` | examples/pir:5 | ✓ | ✓ (+rt+pthread) | ✓ (+rt+pthread) | ✓ | имя misleading — собирается ВЕЗДЕ, не только E2K |
| `train_mlp_char` | examples/pir:31 | ✓ | ✓ | ✓ | ✓ | |
| `test_gguf_inference` | examples/gguf:4 | ✓ | ✓ | ✓ | ✓ | бывает требует /STACK:64MB |
| `q4k_batched_vs_serial` | examples/gguf:24 | ✓ | ✓ | ✓ | ✓ | |
| `train_gan` | examples/gan:4 | ✓ | ✓ | ✓ | ✓ | |
| `shakespeare_train` | examples/shakespeare:3 | ✓ | ✓ | ✓ | ✓ | |
| `train_mnist_nmcard` | examples/nmcard:3 | • | • | • | • | `PT_USE_NMCARD=ON` |
| `promeserve` | promeserve:4 | ✓ | ✓ | ✓ | ✓ | требует ws2_32 на Windows |
| `bench_prometorch` | benchmarks:2 | ✓ | ✓ | ✓ | ✓ | |
| `bench_optimized` | benchmarks:19 | ✓ | ✓ | ✓ | ✓ | |
| `bench_threadpool_overhead` | benchmarks:36 | ✓ | ✓ | ✓ | ✓ | |
| `gguf2pt8` | tools/gguf2pt8:23 | ✓ | ✓ (-static-libstdc++) | ✓ | ✓ | alias `prometorch-convert` (Unix only) |

**Итого: 60 target'ов (14 libs + 22 tests + 30 executables через add_subdirectory).**

## 3. DEFAULT_TARGETS в scripts/build-*.sh

| Скрипт | DEFAULT_TARGETS | Покрывает Qwen3-4B inference? | MNIST? | PIR? | promeserve? |
|---|---|:-:|:-:|:-:|:-:|
| `build-elbrus.sh` | `aten_cpu torch_autograd tuda_tests_standalone tuda_tests test_gguf_inference` | да (через test_gguf_inference) | нет | нет | нет |
| `build-alt.sh` | `cmake --build . -j` (всё) | да | да | да | да |
| `build-astra.sh` | `cmake --build . -j` (всё) | да | да | да | да |
| `build-redos.sh` | `cmake --build . -j` (всё) | да | да | да | да |
| `build-baikal-cross.sh` | `cmake --build . -j` (всё) | в теории, не верифицировано | да | да | да |

**Gap:** `build-elbrus.sh` НЕ собирает `train_mnist_mlp`, `train_pir_elbrus`, `promeserve`. Юзер должен переопределить через `PT_TARGETS="train_pir_elbrus promeserve ..."`. Это нарочно (overhead `gguf_model.h` ≈ 6 000 строк), но не задокументировано в README — провоцирует "configure on E16C падает" если юзер локально модифицирует target list.

## 4. `--no-eml` / `--no-openblas` флаги

**`--no-eml`** (`build-elbrus.sh` строки 31-33): ✅ РЕАЛИЗОВАН. Прокидывает `-DPT_USE_EML_BLAS=OFF`. Логика в root:227-269 — пробует EML→fallback OpenBLAS→fallback TUDA 6×6 микроядро. На E16C v6 EML probe должен fail (libeml помечена elbrus-2c3, runtime ELF mismatch), CMake автоматически делает switch на OpenBLAS если `_PT_OPENBLAS_LIB` найден.

**`--no-openblas`** ❌ **НЕ РЕАЛИЗОВАН в build-elbrus.sh.** В CMake `option(PT_USE_OPENBLAS ...)` существует (root:222) — можно передать `-DPT_USE_OPENBLAS=OFF` вручную. Скрипт принимает только `--no-eml|--without-eml` (grep подтвердил 0 matches на `no-openblas`). Чтобы реально упасть на TUDA-only нужно `cmake ... -DPT_USE_EML_BLAS=OFF -DPT_USE_OPENBLAS=OFF`, либо удалить системный libopenblas (юзер не догадается).

Все три пути (EML / OpenBLAS / TUDA) реально достижимы в коде (root:727 EML→PUBLIC eml_mt, root:750 OpenBLAS→PUBLIC ${_PT_OPENBLAS_LIB} + define `PT_USE_OPENBLAS_BLAS`, root:763 fallback на встроенный TUDA gemm), но на E16C **single команда `build-elbrus.sh` не позволяет** заставить третий путь — для этого нужен ручной cmake.

## 5. Orphan / Dead / Gated

### Orphan (источник есть, target в CMake отсутствует — построить нельзя)

**`tests/` top-level (9 файлов):** `test_audio.cpp`, `test_autocast_wiring.cpp`, `test_ddp_no_sync.cpp`, `test_new_ops_2026_04_18.cpp`, `test_param_groups.cpp`, `test_pipeline_parallel.cpp`, `test_to_autograd.cpp`, `tests/io/test_pt8_loader.cpp` — НЕ в `CMakeLists.txt`. Подтверждено grep'ом по root. Единственный из `tests/` подключённый — `tests/test_transformer_cuda_forward.cpp` (через `examples/transformer/CMakeLists.txt:17`).

**`test/cpp/` orphans:** `test_distributed.cpp`, `test_mps.cpp`, `test_ops_expansion.cpp` — лежат в test/cpp/ рядом с подключёнными, но ни одного `add_executable` на них.

**`tests/promeserve/` pytest battery (Python):** `conftest.py`, `test_api_contract.py`, `test_tool_call.py` — НЕТ ни pytest target, ни CTest регистрации, ни упоминания в скриптах CI. Запускается только вручную через скрипт `scripts/test_promeserve_tools.sh`.

### Dead code references в CMake

- `set(PT_BUILD_GENERATED_TESTS OFF)` — `test_ops_generated.cpp` + `ops_spec.cpp` всегда skip (root:1322).
- `set(PT_BUILD_DEEPSPEED_TESTS OFF)` — `deepspeed_tests` всегда skip (root:1463).
- `examples/transformer/CMakeLists.txt:17` создаёт `test_transformer_cuda_forward` **в каждом** CUDA-разрешённом билде, но code path внутри `#ifdef PT_USE_CUDA` — без CUDA это no-op binary.
- `train_pir_elbrus` имя misleading — guard `if(EXISTS train_pir_elbrus.cpp)` срабатывает на ВСЕХ платформах. Windows MSVC может его собрать, хоть POSIX `rt`/`pthread` подключаются только `if(NOT MSVC)`.

### Gated и без runtime test

- `gguf_ddp_tests` (root:1444) — собирается, но НЕ `add_test` — требует GGUF путь в env. CI его не запускает.
- `ddp_tests`, `fsdp_tests`: собираются на Windows, но `add_test` только под `NOT WIN32` — на Windows билд бесполезен.
- `aten_mps` — option существует, **в репо нет ни одной сборки** (нет macOS toolchain, не упоминается в build-*.sh).

### Дубли target name

`aten_cuda` определяется ДВАЖДЫ (root:599 для PT_USE_CUDA, root:673 для PT_USE_ROCM). FATAL_ERROR на одновременном включении (root:21) предотвращает collision.

## Краткое summary

Статически разобраны 60+ targets через единственный root CMakeLists.txt (1738 строк) + 14 examples/ + promeserve/ + tools/ + benchmarks/. **Core libs** (c10, aten_cpu, aten, torch_*) собираются на всех 7 платформах (Win MSVC, Lin GCC x86, E2K v3/v4/v5/v6, ARM aarch64) — нет ISA-специфичного кода в их `add_library` блоках. **aten_cuda/ROCm** взаимоисключающи (FATAL_ERROR строка 21), на E2K принудительно OFF в toolchain. **MPS** объявлен но не интегрирован в ни один build-*.sh (нет Apple toolchain в репо). **Tests:** 22 target'а зарегистрированы; `test_ops_generated` и `deepspeed_tests` намеренно gated (defaults OFF); `nmcard_tests`/`linq_tests` — за `PT_USE_*` опциями. **30 example/utility executables** через subdirs; почти все используют один шаблон с `if(EXISTS ...)` guard + опциональный CUDA link + MSVC `/STACK:10MB`. **Orphan tests:** 9 файлов в `tests/` top-level + 3 в `test/cpp/` (test_distributed, test_mps, test_ops_expansion) НЕ зарегистрированы в CMake — подтверждает аудит 2026-05-20. Pytest battery `tests/promeserve/` (3 файла) тоже без CTest target. **scripts/build-elbrus.sh DEFAULT_TARGETS** = только `aten_cpu torch_autograd tuda_tests_standalone tuda_tests test_gguf_inference` — НЕ покрывает MNIST/PIR/promeserve (юзер должен задавать `PT_TARGETS=`). Другие build-*.sh (alt/astra/redos/baikal-cross) делают `cmake --build .` без target — собирают всё что есть, медленно но честно. **`--no-eml` работает** (parsing подтверждён, прокидывается `-DPT_USE_EML_BLAS=OFF`); **`--no-openblas` НЕ реализован в скрипте** — только через ручной `-DPT_USE_OPENBLAS=OFF` в cmake. Три BLAS-пути (EML / OpenBLAS / TUDA 6×6) реально достижимы в CMakeLists, но E16C-юзер не может одним флагом скрипта попасть в TUDA-fallback.

**Файлы:** `C:\Users\USER\Desktop\promethorch\CMakeLists.txt`, `C:\Users\USER\Desktop\promethorch\scripts\build-elbrus.sh`, `C:\Users\USER\Desktop\promethorch\scripts\build-alt.sh`, `C:\Users\USER\Desktop\promethorch\scripts\build-astra.sh`, `C:\Users\USER\Desktop\promethorch\scripts\build-redos.sh`, `C:\Users\USER\Desktop\promethorch\scripts\build-baikal-cross.sh`, `C:\Users\USER\Desktop\promethorch\cmake\toolchains\e2k-elbrus.cmake`, `C:\Users\USER\Desktop\promethorch\examples\*\CMakeLists.txt`, `C:\Users\USER\Desktop\promethorch\promeserve\CMakeLists.txt`, `C:\Users\USER\Desktop\promethorch\tools\gguf2pt8\CMakeLists.txt`, `C:\Users\USER\Desktop\promethorch\benchmarks\CMakeLists.txt`.
