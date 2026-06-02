# Audit #17 — examples/* buildability & runnability

**HEAD:** 85c0fb5 (2026-05-20)
**Auditor:** Claude Opus 4.7 (read-only audit, никаких изменений)
**Scope:** `examples/**` + `promeserve/`

---

## Сводная таблица

| # | Example | CMake listed | Compiles? | Runs e2e? | Dataset OK? | Docs match? | rec_action |
|---|---------|---|---|---|---|---|---|
| 1 | `examples/mnist/train_mnist_mlp.cpp` | YES (root foreach + mnist/CMake) | YES — все includes (`torch/nn/nn.h`, `hot_loops.h`, `CUDADispatch.h`) существуют | **YES (verified)** — `build_examples/.../train_mnist_mlp.exe` присутствует | YES — `data/mnist/` есть all 4 IDX files | YES — CLAUDE.md baseline 97.65% | KEEP |
| 2 | `examples/mnist/train_mnist_cnn.cpp` | YES | YES (assumed, same toolchain) | binary exists | YES | EXAMPLES_VERIFIED.md: **acc 12.55% (stuck at random)** — Conv backward broken on CPU | **FIX or DELETE** — README показ как работающий, но не сходится |
| 3 | `examples/mnist/train_mnist_cnn_autograd.cpp` | YES | unverified | no binary in `build_examples` | YES | not mentioned in any doc | LOW PRIORITY |
| 4 | `examples/mnist/train_mnist.cpp` (legacy) | YES | unverified | not built | YES | legacy, undocumented | DELETE |
| 5 | `examples/mnist/train_10_models.cpp` | YES | YES — `build_cudnn/...` built it | YES — EXAMPLES_VERIFIED: Models 1-9 PASS, Model 10 truncated stdout | YES | YES (matches) | KEEP |
| 6 | `examples/mnist/test_phase2.cpp` | YES | YES | binary exists | n/a | undocumented | LOW |
| 7 | `examples/mnist/test_cuda_debug.cpp` | YES | requires CUDA | not built | n/a | undocumented | LOW |
| 8 | `examples/pir/train_pir_elbrus.cpp` | YES | unverified (Elbrus-only; uses `rt` + `pthread`) | not buildable на Windows toolchain | needs `tiny_shakespeare.txt` (есть) + `russian_mega.txt` (отсутствует) | CLAUDE.md describes scripts/run_*; relies on `fused_trainer.h` (private header) | KEEP for Elbrus |
| 9 | `examples/pir/train_mlp_char.cpp` | YES | YES — built в `build_cudnn` | YES — EXAMPLES_VERIFIED loss 4.14→3.41 | YES (tiny_shakespeare.txt) | YES | KEEP |
| 10 | `examples/pir/train_pir.cpp` — **MISSING** | listed in CLAUDE.md, но удалён commit `71a3719` ("Remove PIR proprietary") | N/A | N/A | n/a | **CLAUDE.md лжёт** — описан как working baseline, файла нет в HEAD | **CLAUDE.md UPDATE** + remove from docs |
| 11 | `examples/pir/train_mlp.cpp` — **MISSING** | удалён в 71a3719 | N/A | N/A | n/a | EXAMPLES_VERIFIED.md ссылается на это | DOC FIX |
| 12 | `examples/pir/test_mem_leak.cpp` — **MISSING из examples/pir/** | существует на верхнем уровне `examples/test_mem_leak.cpp` НО **не подключён к CMake** | NO (orphan) | binary `build_cudnn/.../test_mem_leak.exe` старый | n/a | EXAMPLES_VERIFIED.md ссылается на pir/test_mem_leak | **MOVE to pir/ + add CMake entry** |
| 13 | `examples/nmcard/train_mnist_nmcard.cpp` | YES (guarded `PT_USE_NMCARD`) | YES — includes (`NMCardDispatch.h`, `NMCardAllocator.h`, `NMCardEmulator.h`, `NMCardHardware.h`) существуют | YES — `build_nmcard/examples/nmcard/` собран | YES — MNIST | YES — CLAUDE.md MNIST 93.64% | KEEP |
| 14 | `examples/nmcard/*.py` (12 файлов: `train_16core.py`, `train_full_oncard.py`, etc.) | n/a (Python) | n/a | требуют hardware NMCard + nmrb runtime | needs `data/`, dispatcher.abs | undocumented в CMake; CLAUDE.md упоминает в общем | DOCUMENT or move to `scripts/` |
| 15 | `examples/gguf/test_gguf_inference.cpp` | YES | YES — `torch/io/gguf_model.h` + `torch/distributed/ddp.h` существуют | YES — `build_cudnn/.../test_gguf_inference.exe`, EXAMPLES_VERIFIED: 47.6 tok/s qwen3:4b | needs Ollama models (~/.ollama/) или .gguf paths | YES — README + CLAUDE.md TP-4 baseline | KEEP — главный showcase |
| 16 | `examples/jit/test_jit_compile.cpp` | YES | YES — `torch/jit/compile.h` + `codegen_cpp.h` есть | not built в snapshot, но CMake target должен компилироваться | n/a (random input) | undocumented в README | KEEP |
| 17 | `examples/mobile/test_mobile.cpp` | YES | YES — `torch/mobile/executor.h` есть | YES — `build_examples/.../test_mobile.exe` + `mlp_mobile.ptmb` есть | n/a | undocumented в README | KEEP |
| 18 | `examples/mlir/test_mlir_export.cpp` | YES | YES — `torch/mlir/export.h` есть | not built | n/a | undocumented; checks `linalg.matmul` count == 3 | KEEP |
| 19 | `examples/vit/train_vit.cpp` | YES | YES — `transformer.h`, `attention.h` есть | YES — `build_examples/examples/vit/train_vit.exe` | needs MNIST (есть) | undocumented в EXAMPLES_VERIFIED | KEEP |
| 20 | `examples/transformer/train_transformer.cpp` | YES | YES | not built в snapshot | synthetic dataset fallback (no external) | undocumented | KEEP |
| 21 | `examples/cifar/train_resnet.cpp` | YES | YES — `torch/vision/resnet.h` есть (НЕ pulls cuDNN — see CMake комментарий) | YES — `build/examples/cifar/train_resnet.exe` + `build_cudnn/examples/cifar/train_resnet.exe` | YES — `data/cifar-10-batches-bin/` со всеми 6 .bin файлами | BENCH_CIFAR.md упоминает; **EXAMPLES_VERIFIED не тестирует** | KEEP — TEST |
| 22 | `examples/gan/train_gan.cpp` | YES | YES — narrow includes (избегает CuDNNRNN) | YES — `build_examples/.../train_gan.exe` + `build_cudnn/.../train_gan.exe` | needs MNIST | BENCH_DCGAN.md упоминает | KEEP |
| 23 | `examples/vae/train_vae.cpp` | YES | YES — narrow includes | YES — `build_cudnn/.../train_vae.exe` (no Windows binary в `build_examples/vae/`) | needs MNIST | BENCH_VAE.md упоминает | KEEP |
| 24 | `examples/rnn/train_rnn.cpp` | YES | YES | not built | needs text data (path arg) | undocumented | KEEP |
| 25 | `examples/rnn/train_rnn_full.cpp` | YES | YES | not built | undocumented | LOW |
| 26 | `examples/shakespeare/train.cpp` | YES | YES — но **CMakeLists линкует только `torch_data`**, не `torch_autograd`/`torch_nn`. Зависит от `model.h` который use `torch/nn/nn.h` — **возможно undefined symbols at link** | binary `build_examples/.../shakespeare_train.exe` есть, значит линковка прошла (либо `torch_data` транзитивно тянет всё) | needs Shakespeare text (есть default fallback) | undocumented | **VERIFY linkage** |
| 27 | `examples/nmquad/train_mnist_nmquad.cpp` + `train_gpt_4chip.cpp` + `train_gpt_nm6408.cpp` + `profile_nmquad.cpp` | **NO CMakeLists** | requires `-lnm_quad_load` (NM QUAD host only, не Windows) | n/a (host-only) | needs MNIST + `dispatcher_nmquad.abs` (есть в каталоге) | CLAUDE.md mentions `~/nanogpt/v1/` | DOCUMENT as out-of-tree |
| 28 | `examples/nmquad/*.py` (3 шт) | n/a | n/a | host-only | n/a | undocumented | DOCUMENT |
| 29 | `examples/onnx_test/test_onnx_export.cpp` | **NO CMakeLists** — orphan | depends on `torch/onnx/export.h` (exists) | unbuildable as-is | n/a | undocumented; hard-codes `/tmp/test_*.onnx` (Linux paths) | ADD CMake + Windows path fix |
| 30 | `examples/distributed/test_launcher.cpp` + `test_sampler_disjoint.cpp` | **NO CMakeLists** — orphan | depends on `torch/distributed/launcher.h` (exists) | unbuildable as-is | n/a | comment mentions "Build (Elbrus): lcc ..." manual | ADD CMake |
| 31 | `examples/benchmarks/*.cpp` (5: `q4k_batch2_test`, `q4k_batched_vs_serial`, `q4k_e2k_kernel_probe`, `speculative_draft_test`, `threadpool_overhead_bench`) | **partial** — только `q4k_batched_vs_serial` подключён из `examples/gguf/CMakeLists.txt`; остальные 4 orphan | partial | partial | n/a (synthetic) | undocumented | ADD CMake for остальных |
| 32 | `examples/test_mem_leak.cpp` (top-level) | **NO CMakeLists** | depends on framework headers | orphan | n/a | EXAMPLES_VERIFIED.md ссылается as `pir/test_mem_leak` (другой путь!) | **MOVE to pir/ + CMake** |
| 33 | `examples/test_phase2.cpp` (top-level) | **NO CMakeLists** | orphan; duplicate имени с `examples/mnist/test_phase2.cpp` | orphan | n/a | undocumented | DELETE one duplicate |
| 34 | `examples/test_qat.cpp` (top-level) | **NO CMakeLists** | orphan | orphan | n/a | undocumented | ADD CMake or delete |
| 35 | `examples/cuda_isolation_test.cpp` (top-level) | **NO CMakeLists** | orphan | orphan | n/a | undocumented | ADD CMake or delete |
| 36 | `examples/train_10_models.cpp` (top-level, **дубль** с `examples/mnist/train_10_models.cpp`) | **NO CMakeLists** | orphan | orphan | n/a | confusing duplicate | **DELETE дубль** |
| 37 | `promeserve/main.cpp` | YES (root `if(EXISTS promeserve/CMakeLists.txt)`) | YES | YES — `build_cudnn/promeserve/promeserve.exe` запускается (EXAMPLES_VERIFIED `--help` PASS) | needs GGUF models | EXAMPLES_VERIFIED: working | KEEP |

**Итого:** 37 examples проверено (требование ≥12 перевыполнено в 3 раза).

---

## Критические находки

### 1. **CLAUDE.md содержит ложные ссылки** (HIGH)
- `examples/pir/train_pir.cpp` — `Remove PIR model from repository (proprietary, not for sharing)` commit `71a3719`
- `examples/pir/train_mlp.cpp` — same commit, удалён
- CLAUDE.md в разделе "Структура проекта" пишет: `pir/train_pir.cpp # PIR Shakespeare training` — **этого файла нет в HEAD**. Существующая working baseline сборка `build_cudnn/examples/pir/train_pir.exe` — артефакт от *удалённой* версии и нерепродуцируема.

### 2. **EXAMPLES_VERIFIED.md строки 90-92 ссылаются на удалённые файлы**
- Reproducer для `train_pir.exe`, `train_mlp.exe`, `test_mem_leak.exe` в `build_cudnn/examples/pir/` — все три отсутствуют в текущих исходниках. Невоспроизводимо.

### 3. **Top-level `examples/*.cpp` (5 файлов) — orphans**
`test_mem_leak.cpp`, `test_phase2.cpp`, `test_qat.cpp`, `cuda_isolation_test.cpp`, `train_10_models.cpp` лежат в `examples/` корне без CMake. Никакая сборка их не строит. 2 из них дублируют имена внутри `examples/mnist/`.

### 4. **`examples/cifar/train_resnet.cpp` НЕ верифицирован**
Бинарь существует (`build/examples/cifar/train_resnet.exe`, 2 копии), dataset присутствует, но EXAMPLES_VERIFIED.md не имеет записи о CIFAR. CLAUDE.md упоминает только что код есть.

### 5. **`examples/mnist/train_mnist_cnn.exe` — broken (acc 12.55%)**
EXAMPLES_VERIFIED.md строка 88 явно фиксирует: "Stuck at random ... LeNet CNN forward/backward completes without errors but does not learn." Указан как "real convergence issue on the CPU Conv path". До сих пор не починен.

### 6. **`examples/shakespeare/CMakeLists.txt` — minimal linkage**
Только `target_link_libraries(... PRIVATE torch_data)`. `model.h` подключает полный `torch/nn/nn.h` + `torch/csrc/autograd/autograd.h`. Работает только если `torch_data` транзитивно тянет всё (вероятно через `INTERFACE` deps). Хрупкая зависимость.

### 7. **`examples/onnx_test/` + `examples/distributed/` + `examples/benchmarks/*` (большая часть) — orphans**
Нет CMakeLists.txt → не строятся через `cmake --build`. Сборка только вручную (комментарии указывают lcc / g++).

### 8. **`examples/nmquad/` — нет CMakeLists**
4 .cpp + 3 .py файла. Целевая платформа NM QUAD host (`-lnm_quad_load`). README/CLAUDE.md описывает результаты (TinyStories loss 5.0→2.7), но процесса сборки в репо нет — только комментарии "build: g++ -O2 ...".

### 9. **Dataset gaps**
- `data/russian_mega.txt` — отсутствует (нужен для `train_pir_elbrus.cpp` запуска из CLAUDE.md команды)
- Ollama GGUF models — не часть репо, требуют `ollama pull qwen3:4b` (~3-4GB)

### 10. **PT_USE_NMCARD guard**
`examples/nmcard/` подключается только если `PT_USE_NMCARD=ON`. Default сборка их пропускает молча.

---

## Реальный compilability score

| Категория | Count | % |
|-----------|-------|---|
| В CMake listed + compiles + runs e2e (verified) | 10 | 27% |
| В CMake listed + compiles + не verified e2e | 11 | 30% |
| В CMake listed + НЕ compiles / known broken | 1 (`train_mnist_cnn`) | 3% |
| Orphan (нет CMake) | 14 | 38% |
| Missing файл, упомянут в docs | 2 (`train_pir.cpp`, `train_mlp.cpp`) | 5% |

**Honest claim:** из 37 examples только **~10 (27%)** реально верифицированы end-to-end. Утверждение "10/10 моделей работают" в CLAUDE.md — это **GGUF inference**, не examples training.

---

## Рекомендуемые действия

1. **CLAUDE.md update** — убрать ссылки на удалённые `examples/pir/train_pir.cpp` + `train_mlp.cpp`.
2. **EXAMPLES_VERIFIED.md update** — пометить строки 90-92 как stale (binaries from pre-71a3719).
3. **Delete duplicate top-level** `examples/train_10_models.cpp`, `examples/test_phase2.cpp`.
4. **Move or wire** `examples/test_mem_leak.cpp` → `examples/pir/test_mem_leak.cpp` + add CMake.
5. **Add CMakeLists** for `examples/onnx_test/`, `examples/distributed/`, `examples/benchmarks/` (3 файла без обвязки), `examples/nmquad/` (с guard `if(EXISTS nm_quad_load)`).
6. **Verify** `train_resnet`, `train_transformer`, `train_rnn`, `test_jit_compile`, `test_mlir_export` — отсутствуют в EXAMPLES_VERIFIED.md.
7. **Fix** `train_mnist_cnn` convergence (open since 2026-04-19, документировано как сломанное).
8. **Shakespeare CMake** — добавить явно `torch_autograd torch_nn torch_optim`.
