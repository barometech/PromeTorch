# Аудит #19 — Env-driven behavior + magic constants

**Дата:** 2026-06-02
**HEAD:** 85c0fb5
**Скоп:** весь C++ (`getenv(...)`, `#ifdef PT_...`, числовые литералы в hot
path / dispatch / IO).

Цель — каталог всех «скрытых рычагов», которыми код перенастраивается
извне без прохода через CLI / config-файл. Пользователь должен видеть, что
поведение/перформанс/корректность могут переключаться env-переменной, и
понимать, какие из них опасны.

Сводный счёт по C++ (без скриптов, без `.md`):

| Категория | Кол-во | Расположение основных хитов |
|---|---|---|
| `getenv("PT_*")` / `getenv("PROMETORCH_*")` / `getenv("PROMESERVE_*")` | **56 уникальных callsites** | `torch/io/gguf_model.h` (28), `c10/util/ThreadPool.h` (4), `aten/.../hot_loops.cpp` (3), `aten/.../tuda/TudaBLAS.h` (4), `torch/io/deepseek2_forward.h` (6), `torch/io/q8_soa_repack.h` (2), и др. |
| `getenv("OMP_*")` / `getenv("MASTER_*")` / `getenv("RANK")` / `getenv("WORLD_SIZE")` / `getenv("HOME")` / `getenv("USERPROFILE")` / `getenv("TEMP")` / `getenv("TMP")` / `getenv("TMPDIR")` / `getenv("OLLAMA_MODELS")` / `getenv("NMCARD_DLL_PATH")` | **13** | thread pool, distributed launcher, JIT codegen, ollama loader, NM Card driver |
| CMake `option(PT_USE_*)` | 14 | `CMakeLists.txt` |
| CMake auto-detect `PT_HAVE_*` (E2K intrinsics) | 16 | `cmake/E2KIntrinsicChecks.cmake` |
| `PT_HAVE_NUMA_H` / `PT_HAS_LIBNUMA` | 2 | `torch/io/numa_weight_replica.h`, `cmake/...` |
| `#ifdef PT_DEBUG_*` | 2 (`PT_DEBUG_HTTP`, `PT_DEBUG_DECODE`) | `promeserve/`, `torch/io/gguf_model.h` |
| `PT_TRACE_*` | 0 | — нет, заявлено, но не существует |

---

## 1. Env vars (полный каталог)

Колонка «Док» — что указано в репозиторных доках; **(нет)** = только в коде
комментарием. Колонка «Эффект» — измеренное/задокументированное влияние
на корректность или tok/s.

### 1.1 Перформанс-критичные (PT_Q8_SOA, PT_NO_NUMA_POOL и Ко)

| Env | file:line | Default | Values | Док | Эффект |
|---|---|---|---|---|---|
| `PT_Q8_SOA` | `torch/io/gguf_model.h:5461` | OFF (legacy GEMV) | `=1` enable | `docs/BUILD_ELBRUS.md:283` (**СВЯТОЕ**), CLAUDE memory `feedback_pt_q8_soa_required.md` | **+48% tok/s** (7.7 → 11.4 на qwen3-4B TP-4). Юзер ловил отсутствие 20+ раз. |
| `PT_NO_NUMA_POOL` | `hot_loops.cpp:72, 416` | OFF (pthread NUMA pool) | `=1` bypass | `docs/BUILD_ELBRUS.md:266,285` | **bisect 2026-04-30: 11→1.4 tok/s если выключить на 8C2 4-NUMA, +значит-ли-то «correctness» если on/off — нет, только перф.** ВКЛ полезно с `numactl --cpunodebind` + EML_MT (245 GFLOPS/node). |
| `PT_DDP_SHM` | `torch/distributed/ddp.cpp:428` | OFF (TCP fallback) | `=1` | `docs/BUILD_ELBRUS.md:269,286` | ~10× быстрее AllReduce (2560 floats) vs TCP loopback. |
| `PT_DDP_FUTEX` | `torch/distributed/ddp.cpp:429` | OFF (busy spin) | `=1` | **(нет)** | Futex-based wait в SHM AllReduce. |
| `PT_TP_GATHER` | `gguf_model.h:5208, 5696` | OFF (K-slice all-reduce) | `=1` row-slice gather | README:189 (gemma3 only) | Альт. TP схема; ТРЕБУЕТ `inter%nprocs==0` иначе runtime_error. Включена только для gemma3 TP-4. |
| `PT_PER_BLOCK_SCALE` | `q8_soa_repack.h:275,344` | OFF (per-tensor scale) | `=1` per-block | README:165 | **Влияет на КОРРЕКТНОСТЬ** русского вывода (cyrillic outlier activations). Без флага = калич, с флагом = читаемо. Скорость не меряна — пишут *11.4 tok/s lossless baseline* про per-tensor. |
| `PT_LM_HEAD_FP` | `gguf_model.h:6501` | OFF (Q8 SoA весь) | `=1` only-lm-head в FP | README:205, код-комментарий 6494-6500 | КОРРЕКТНОСТЬ финального argmax при cyrillic. Без = 11.4 tok/s broken Russian; с = чинит. |
| `PT_NO_FFN_SOA` | `gguf_model.h:5813, 6257, 6400` | OFF | `=1` отключить Q8 SoA в FFN | README:205, код-комментарий 5811 | **Падение до ~7 tok/s** (с 11.4), зато стабильный русский (Massive Activations outlier fix). |
| `PT_Q4K_V2` | `torch/io/cpu_quant_gemv.h:1647` | OFF (AVX2 horiz) | `=1` Phase 6 vertical-accum | **(нет)** | Альт. AVX2 q4k kernel (gemini_sse41_design.md ссылка в коде). Перф effect unknown. |
| `PT_HUGETLB` | `torch/io/gguf_loader.h:267` | OFF (file-backed mmap) | `=1` | **(нет в user-facing docs)** | Hugetlb (2 MB) anon mmap + pread весь файл. Может ускорить, но требует `vm.nr_hugepages`. Молча fallback. |
| `PT_NUMA_REPLICATE` | `torch/io/numa_weight_replica.h:46` | `0`/unset = OFF | `=1` enable | `docs/BUILD_ELBRUS.md:243,268,287` | **HARD WARNING в коде:** ставить только в SP-режиме. В TP даёт ×4 RAM и регрессию. |
| `PT_CORES_PER_NODE` | `c10/util/ThreadPool.h:81` | 8 (E8C2) | int | **(нет)** | Fallback path когда `libnuma` недоступна; влияет на маппинг cpu→numa node. |
| `PT_PIN_THREADS` | `c10/util/ThreadPool.h:257` | OFF | `=1` | `docs/BUILD_ELBRUS.md:290` (**HARD GUARD**) — **запрещено в TP-режиме.** | В TP `numactl --cpunodebind` уже сделал affinity; повторный pin клампит rank 1-3 на ядро 0. **МОЖЕТ СЛОМАТЬ TP скорость до 1.4 tok/s.** |
| `PT_NUM_THREADS` | `c10/util/ThreadPool.h:98` | от `OMP_NUM_THREADS` → `hardware_concurrency()` | int | **(нет)** | Размер пула воркеров. Принимает любое >0. |
| `PT_TP_TIMEOUT_MS` | `c10/util/ThreadPool.h:116` | 0 (disabled) | uint32 ms | **(нет)** | Watchdog для futex_wait в ThreadPool. |
| `PT_NO_EML` | `tuda/TudaBLAS.h:32, 59` | OFF | `=1` | **(нет)** | Полностью отключает EML cblas_sgemm callsites. Скрытый kill-switch. |
| `PT_NO_BLAS` | `tuda/TudaBLAS.h:58, 79` | OFF | `=1` | **(нет)** | Аналог `PT_NO_EML` для OpenBLAS-path. |

### 1.2 Architecture / model selection

| Env | file:line | Default | Values | Док | Эффект |
|---|---|---|---|---|---|
| `PT_FORMAT_AUTO` | `gguf_model.h:2042` | auto-detect ON | `=0` force GGUF | inline comment 2039-2041 | Wrong-detect крайне маловероятен. Debug only. |
| `PT_SYSTEM_PROMPT` | `gguf_model.h:4186` | `"You are a helpful assistant."` | string (`""` отключает) | inline 4180-4185 | Без system prompt qwen3 уходит в галлюцинации (см. комментарий BUG-12). |
| `PT_NO_THINK` | `gguf_model.h:4195` | OFF | `=1` (qwen3 only) | inline | Inject `/no_think` в system. |
| `PT_LAYER_SKIP` | `gguf_model.h:5764` | "" (skip none) | csv `"12,14,..."` | README:247 | **LOSSY.** Сэкономить ~2.5 ms/layer/token на qwen3-4B 36L. Корректность падает. |
| `PT_NO_LONGROPE` | `gguf_model.h:287` | OFF | `=1` | inline 284-285 (bisect helper) | Игнорирует rope_factors. Phi-3 bisect tool. |
| `PT_FORCE_ALL_GLOBAL` | `gguf_model.h:276` | OFF | `=1` | inline 272-274 (bisect) | Gemma3 SWA bisect tool: все слои → global. Если результат тот же — баг не в SWA. |
| `PT_QK_AFTER_ROPE` | `gguf_model.h:5880` | OFF | `=1` | inline 5877-5878 (bisect) | Переставить qk_norm после RoPE. |

### 1.3 Spec decode

| Env | file:line | Default | Values | Док | Эффект |
|---|---|---|---|---|---|
| `PT_SPEC_K` | `torch/io/speculative_verify.h:34` | 1 (off) | int 1..6 | `vliw_mission/round3/agent_2_spec_decode.md` | Speculative decode draft count. |
| `PT_SPEC_DRAFT_PATH` | `gguf_model.h:1026` | empty | path string | **(нет в user docs)** | Путь к draft модели для spec decode. |
| `PT_PLD` | `gguf_model.h:4638` | OFF | `=1` | inline 4631-4636 | Saxena 2023 prompt-lookup draft. **На Эльбрусе РЕГРЕССИЯ 4.3 → 2.7 tok/s** (комментарий). |

### 1.4 Debug / instrumentation (нулевая нагрузка off, никакой документации в публичных доках)

| Env | file:line | Эффект |
|---|---|---|
| `PT_PROFILE_LAYER` | `gguf_model.h:3191, 5645` | Per-section ms/token dump. |
| `PT_DUMP_HIDDEN` | `gguf_model.h:3270, 5729` | mean/std/min/max каждого слоя для bisect vs llama.cpp. |
| `PT_DUMP_TOKENS` | `gguf_model.h:4752` | Token IDs после tokenizer.encode. |
| `PT_DEBUG_LOGITS` | `gguf_model.h:4908` | Top-5 raw logits до rep penalty. |
| `PT_DEBUG_TOKENS` | `examples/gguf/test_gguf_inference.cpp:245` | Debug print в тесте. |
| `PT_NO_QUANT_GEMV` | `gguf_model.h:919` | bisect — FP fallback для всех Q-GEMV. |

### 1.5 DeepSeek2 / GigaChat3 MLA debug (6 штук, ни одной в публичных доках, только memory `project_gigachat3_mla_bugs.md`)

| Env | file:line |
|---|---|
| `PT_DS2_DEBUG` | `gguf_model.h:7769` |
| `PT_DS2_NO_ATTN` | `gguf_model.h:7770` |
| `PT_DS2_NO_MOE` | `gguf_model.h:7771` |
| `PT_DS2_NO_SHEXP` | `gguf_model.h:7772`, `deepseek2_forward.h:422` |
| `PT_DS2_LOGITS` | `gguf_model.h:7935` |
| `PT_DS2_DEEP` | `deepseek2_forward.h:186, 320` |
| `PT_DS2_DEEP_LAYER` | `deepseek2_forward.h:187, 321` (int — index) |
| `PT_DS2_CHECK_KB` | `deepseek2_forward.h:219` |
| `PT_DS2_NO_MSCALE` | `hot_loops.cpp:1920` |

### 1.6 Non-PT-prefix (ещё больше скрытых)

| Env | file:line | Назначение |
|---|---|---|
| `OMP_NUM_THREADS` | `ThreadPool.h:99` | Fallback для `PT_NUM_THREADS`. |
| `OMP_PLACES`, `OMP_PROC_BIND` | `examples/pir/train_pir_elbrus.cpp:1097-1098` | Just logging. |
| `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` | `examples/distributed/test_launcher.cpp:39-42` | torchrun-style. |
| `HOME`, `USERPROFILE` | `torch/io/ollama.h:39-40`, `promeserve/model_manager.h:225-227` | Поиск `~/.ollama`. |
| `OLLAMA_MODELS` | `torch/io/ollama.h:32` | Override default `~/.ollama/models`. |
| `NMCARD_DLL_PATH` | `aten/src/ATen/nmcard/NMCardHardware.cpp:75` | Путь к NM Card vendor DLL. |
| `PROMETORCH_JIT` | `torch/jit/compile.h:131` | `=0/off/OFF` глобально выключает JIT codegen. |
| `PROMETORCH_JIT_THRESH` | `torch/jit/compile.h:138` | Порог n для codegen. |
| `PROMETORCH_CACHE_DIR` | `torch/jit/codegen_cpp.h:69` | Куда складывать `.so` JIT артефакты. |
| `PROMETORCH_CC` | `torch/jit/codegen_cpp.h:219` | Override компилятор для JIT. |
| `TEMP`, `TMP`, `TMPDIR` | `torch/jit/codegen_cpp.h:72-77`, `promeserve/tool_call.h:507` | Каталог временных файлов. |
| `PROMESERVE_TOOL_ROOT` | `promeserve/tool_call.h:505` | Корень для tool sandbox FS. |
| `PROMESERVE_MAX_TOOL_ITER` | `promeserve/tool_call.h:1162` | Максимум tool-call итераций. |
| `PROMESERVE_MCP_CONFIG` | `promeserve/mcp_client.h:276` | Путь к MCP конфигу. |
| `GGUF_DDP_MODEL` | `test/cpp/test_gguf_ddp.cpp:151` | Только в тесте. |

**Итого PT_-env vars в коде: 37 уникальных.** В `docs/BUILD_ELBRUS.md` упомянуто **6** (`PT_Q8_SOA`, `PT_NO_NUMA_POOL`, `PT_DDP_SHM`, `PT_NUMA_REPLICATE`, `PT_PIN_THREADS`, `PT_E2K_MARCH/MTUNE` для build). **Около 31 PT_-env остаются НЕЗАДОКУМЕНТИРОВАННЫМИ.**

---

## 2. Magic constants

### 2.1 GGUF quant format

| Константа | file:line | Назначение | Док |
|---|---|---|---|
| `256` (`QK_K`) | `torch/io/gguf_dequant.h:67` (named const) + literal в ~50 callsites `CUDAQuantGemv.cu`, `cpu_quant_gemv.h`, `q4k_e2k_kernel_probe.cpp:43` | Super-block size Q4_K/Q5_K/Q6_K | `JOURNAL.md:2386`, `docs/research/F2_REPORT.md:118`, **есть `constexpr int QK_K`** (HOTSPOT: half of literals не используют именованную константу) |
| `144` | hot path: `CUDAQuantGemv.cu` ×11 callsites, `cpu_quant_gemv.h` ×10, `examples/benchmarks/q4k_*.cpp` ×8 | Bytes per Q4_K super-block (`2+2+12+256/2`) | Вычислено в `gguf_dequant.h:158`. **Hard-coded литерал, нет именованной константы.** Изменить layout = найти ~40 magic 144. |
| `32` | везде в quant kernels | Q4_0/Q5_0/Q8_0 block size (`QK4_0=QK8_0=32`) и Q4_K sub-block | `gguf_dequant.h:61-67` (named), но в hot path literal |
| `12` (Q4_K), `14` (Q6_K), `8` (Q8_0) | `cpu_quant_gemv.h:1641,1660,1667` | GGUF type IDs | Только enum в `GGMLType`; используются как магические числа в switch'е. |
| `22` (Q5_0 per-row bytes) | `deepseek2_forward.h:206-207` | `(no_rope / 32) * 22` | inline comment **есть** |
| `45056` | `deepseek2_forward.h:204` | Per-head Q5_0 bytes (512 × 88) для GigaChat3 K-up | inline comment |
| `0x100u \| PT8_TYPE_Q8_0_SOA4` | `gguf_model.h:5471` | PT8 native marker | inline |

### 2.2 Threading / caching

| Константа | file:line | Назначение | Док |
|---|---|---|---|
| `64` (cache line) | `c10/util/ThreadPool.h:246` (`struct alignas(64) Slot`), `cpu_quant_gemv.h:733,565,...` (+64 prefetch offsets), `TudaConfig.h:88` (`ALIGN=64`) | x86/E2K cache line size | Не задокументирована как `kCacheLineSize`, инлайн в коде. |
| `16` (chunk align) | `ThreadPool.h:158` (`((chunk_size + 15) / 16) * 16`) | False-sharing guard на boundaries | inline comment 155-156 |
| `256` spin | `ThreadPool.h:173` | Spin before yield в I1 wait | — |
| `1024` spin | `ThreadPool.h:215` | Spin перед futex_wait в done | — |
| `4` (prefetch distance) | `cpu_quant_gemv.h:730` | Prefetch lead 4 super-blocks ≈ 576 B ≈ 50 ns DDR latency | inline comment 728-729 (**отличная док**) |
| `16` (far-prefetch) | `cpu_quant_gemv.h:398` | Дальний prefetch к L2 | — |
| `NUMA_POOL_MAX=32`, `NUMA_GEMM_THRESHOLD=512` | `hot_loops.cpp:112-113` | Cap'ы для NUMA pool | inline |
| `kMaxNumaNodes=8` | `numa_weight_replica.h:41` | Compile-time cap | inline comment **есть** |

### 2.3 GEMM tuning (well-documented)

`aten/src/ATen/native/cpu/tuda/TudaConfig.h` — `kAVX2 {6,16,72,256,4096,64}`, `kE2K {6,6,96,256,2048,16}` и пр. Каждое поле prosа-комментировано (L1/L2 fit). **Образец как надо.**

### 2.4 Inference / IO

| Константа | file:line | Назначение | Док |
|---|---|---|---|
| `11434` | `promeserve/main.cpp:7,46,63,70`, `promeserve/promeserve.h:11,35`, `README.md:915,918,922`, `EXAMPLES_VERIFIED.md:95`, `JOURNAL.md:3171`, `scripts/start_promeserve_1proc.sh:23` | Ollama port default | **Задокументирована везде**, тесты используют 18434 (см. `tests/promeserve/conftest.py:22` "11434 + offset 7000"). |
| `2048` | `gguf_model.h:4743` | `prompt + 2048` margin для KV cache | inline comment **есть** |
| `4096` | `gguf_model.h:3241, 5714` | Fallback `max_seq` для KV cache | — |
| `120000` (timeout) | `INFRASTRUCTURE_AUDIT.md` упоминается в fsdp.h:144 как **bug** | wait_file timeout | флаг audit'а #17 |

### 2.5 Performance claims (документ, не код)

| Число | Где | Реалистичность |
|---|---|---|
| `1840 GFLOPS` | `BENCH_ELBRUS.md:188`, `docs/BUILD_ELBRUS.md:174`, `docs/PROMETHORCH_RU.md:1215`, `MEMORY.md` reference | Замерено `benchmarks/sgemm_peak_probe.c`. **80%** от `2304 = 6ch × 128b × 1.5 GHz × 32c` (92% — только от округлённых «2 TFLOPS» МЦСТ; прежняя запись «92% от 2304» была арифметической ошибкой). **Воспроизводимо.** |
| `2304 GFLOPS` (peak) | `benchmarks/sgemm_peak_probe.c:12,76`, `docs/BUILD_ELBRUS.md:32` | Теоретический peak. |
| `11.4 tok/s` (qwen3-4B TP-4 baseline) | `CLAUDE.md`, `MEMORY.md`, `feedback_speed_first.md` | Sacred number, любое падение >0.2 → откат. |
| `10.9 tok/s` | `CLAUDE.md` (после Russian fix) | Post-`0ba114a`. |
| `49.9 tok/s` (qwen3-4B A100 GPU) | `MEMORY.md` GGUF секция | A100, vs Ollama 164.6. |
| `245 GFLOPS/node`, `463 GFLOPS/chip`, `330 GFLOPS` (cross-NUMA degrade) | `hot_loops.cpp:412-414`, `:96` | Inline в комментариях кода. |

---

## 3. Build flags (`#ifdef PT_...`)

### 3.1 CMake `option(...)` (14 user-toggles)

`CMakeLists.txt:15-39, 221-222, 406, 28-29`:

`PT_USE_CUDA`, `PT_USE_ROCM`, `PT_USE_OPENMP`, `PT_USE_AVX`, `PT_USE_AVX2`,
`PT_USE_NMCARD`, `PT_USE_NMQUAD`, `PT_USE_TUDA`, `PT_USE_LINQ`, `PT_USE_MPS`,
`PT_USE_EML_BLAS`, `PT_USE_OPENBLAS`, `PT_USE_CUDNN`,
`PT_BUILD_TESTS`, `PT_BUILD_PYTHON`, `PT_BUILD_SHARED_LIBS`,
`PT_BUILD_VALIDATE`, `PT_BUILD_GENERATED_TESTS`, `PT_BUILD_DEEPSPEED_TESTS`.

Все живые (audit `docs/audit/2026-05-20_build_ci.md:64`). `PT_USE_AVX`
(без 2) — semi-dead. `PT_USE_NUMA` устанавливается из `find_path(numa.h)`, не option().

### 3.2 Auto-detected (`PT_HAVE_*`)

`cmake/E2KIntrinsicChecks.cmake` — 16 переменных через `try_compile`:

`PT_HAVE_QPMADDUBSH`, `PT_HAVE_QPMADDH`, `PT_HAVE_QPADDW`, `PT_HAVE_QPSUBW`,
`PT_HAVE_QPMULLW`, `PT_HAVE_QPISTOFS`, `PT_HAVE_QPFMULS`, `PT_HAVE_QPFADDS`,
`PT_HAVE_QPFSUBS`, `PT_HAVE_QPFNMAS`, `PT_HAVE_QPFMAS`, `PT_HAVE_PMADDUBSH`,
`PT_HAVE_PMADDH`, `PT_HAVE_PADDW`, `PT_HAVE_PFMULS`, `PT_HAVE_PFADDS`.

**Кто выставляет:** только CMake, нельзя override env-переменной. Чтобы
отключить искусственно — нужно вручную убрать `target_compile_definitions`
из `pt_apply_e2k_intrinsic_defines()`.

`PT_HAVE_NUMA_H` — флаг наличия `<numa.h>` (find_path в CMakeLists).
`PT_HAS_LIBNUMA` — derived от `PT_HAVE_NUMA_H && __linux__ && !_WIN32`.

### 3.3 `#ifdef PT_DEBUG_*`

Существуют только два:

* `PT_DEBUG_HTTP` — `promeserve/api_handlers.h` (9 sites), `http_server.h` (4),
  `model_manager.h` (2). Включает verbose request/response логи. Документ:
  `README.md:943`. **Не выставляется CMake'ом — надо `-DPT_DEBUG_HTTP` руками.**
* `PT_DEBUG_DECODE` — `torch/io/gguf_model.h:4958, 5009`. Документ:
  `JOURNAL.md:1314`. Включает `std::cout` flush перед/после CUDA capture.

**`PT_TRACE_*` / `PT_DEBUG_AUTOGRAD` — НЕ СУЩЕСТВУЮТ в кодовой базе.** Заявленные в задании имена — выдумка/опечатка.

### 3.4 Эффект «включить все DEBUG»

* `PT_DEBUG_HTTP` — ~20 fprintf на запрос. Throughput промеcerve просядет (~10-20% при /api/generate), не сломает корректность.
* `PT_DEBUG_DECODE` — `std::cout` flush до/после CUDA-graph capture. CUDA Graphs **не отключаются**, но overhead на flush. Безопасно.

### 3.5 Эффект `PT_HAVE_QPMADDUBSH=0` artificially

Если убрать guard вручную:

* `cpu_quant_gemv.h` E2K-path / Q4_K kernel пойдёт по 2-lane `PMADDUBSH`
  fallback (если `PT_HAVE_PMADDUBSH=1`) — ожидаемо ×2 slowdown.
* Если убрать и `PMADDUBSH` — scalar fallback. **JOURNAL.md:26: «qpmaddubsh
  отключён → scalar fallback → 5.0 tok/s вместо 10.9»** (×2.2 регрессия).

---

## 4. Undocumented env vars (нужно добавить)

Минимум кандидатов на `docs/BUILD_ELBRUS.md` / `README.md`:

**Critical (correctness):**
* `PT_PER_BLOCK_SCALE=1` — для корректного русского (упомянут только в README:165, но без раздела)
* `PT_LM_HEAD_FP=1` — для корректного финального токена при cyrillic
* `PT_NO_FFN_SOA=1` — fallback когда Massive Activations ломают
* `PT_SYSTEM_PROMPT` — undocumented, юзер должен знать, что есть инжект

**Perf:**
* `PT_DDP_FUTEX=1` — пара к `PT_DDP_SHM` без неё перформанс хуже
* `PT_NUM_THREADS` / `PT_CORES_PER_NODE` / `PT_TP_TIMEOUT_MS` — тюнинг ThreadPool
* `PT_HUGETLB=1` — потенциальное ускорение
* `PT_Q4K_V2=1` — альт. kernel
* `PT_SPEC_K`, `PT_SPEC_DRAFT_PATH`, `PT_PLD` — speculative decode

**Debug (для troubleshooting раздела):**
* `PT_PROFILE_LAYER=1`, `PT_DUMP_HIDDEN=1`, `PT_DUMP_TOKENS=1`, `PT_DEBUG_LOGITS=1`, `PT_NO_QUANT_GEMV=1`

**Bisect helpers (важно вынести в "Debugging GGUF" раздел):**
* `PT_FORCE_ALL_GLOBAL`, `PT_NO_LONGROPE`, `PT_QK_AFTER_ROPE`, `PT_FORMAT_AUTO`, `PT_LAYER_SKIP`

**DeepSeek2/GigaChat3 family** (9 env vars `PT_DS2_*`): отдельный раздел в гайде MLA.

**Kill switches без док:** `PT_NO_EML`, `PT_NO_BLAS`.

---

## 5. Dangerous env vars (могут сломать correctness/perf)

| Env | Опасность | Симптом | Источник |
|---|---|---|---|
| `PT_NO_NUMA_POOL` (выкл./некорректное сочетание) | TP-4 на 8C2 без `numactl --cpunodebind` → 11→1.4 tok/s | Драматическое падение throughput | bisect 2026-04-30, упомянут в задании |
| `PT_PIN_THREADS=1` в TP-режиме | Worker_id маппится на cpu#0..N-1, конфликт с `numactl` → все воркеры на 1 ядре | TP падает в 10-20× | `docs/BUILD_ELBRUS.md:290` (HARD GUARD), `c10/util/ThreadPool.h:253-256` |
| `PT_NUMA_REPLICATE=1` в SP/single-process | ×4 RAM + cross-NUMA reads | OOM или slowdown | `docs/BUILD_ELBRUS.md:243` |
| `PT_LAYER_SKIP="..."` | **LOSSY** — поломанная корректность ответа | Бред на выходе | inline comment, README:247 |
| `PT_NO_QUANT_GEMV=1` | FP fallback для всех Q-GEMV | ×5-10 slowdown | bisect-only |
| `PT_NO_FFN_SOA=1` | 11.4 → ~7 tok/s | Намеренный trade-off на качество | inline 6395-6398 |
| `PT_LM_HEAD_FP=1` (если default менялся) | Доп. GEMV per token (минорно) | Минус ~0.1-0.2 tok/s | inline 6494-6500 |
| `PT_PLD=1` на Эльбрусе | **РЕГРЕССИЯ 4.3 → 2.7 tok/s** | Speculative decode хуже baseline | inline 4631-4636 |
| `PT_FORMAT_AUTO=0` + `.pt8` файл | Уйдёт в GGUF reader → exception | Load failure | inline 2039-2041 |
| `PT_Q8_SOA=1` *забыли поставить* | 7.7 vs 11.4 (-32%) | Юзер уже 20 раз ловил | `feedback_pt_q8_soa_required.md` |
| `PT_SYSTEM_PROMPT=""` для qwen3 | Галлюцинации | Бред на выходе | inline 4180-4182 |
| `PT_NO_EML=1` | EML cblas_sgemm callsites disabled → TUDA 6×6 fallback | Сильное падение GEMM perf | необъяснено в docs |

---

## 6. Hotspot-выводы

1. **`144` как magic literal встречается ~30 раз** в hot path (`CUDAQuantGemv.cu`, `cpu_quant_gemv.h`, `q4k_*` benchmarks) — отсутствует именованная `constexpr int Q4K_BLOCK_BYTES = 144` (есть только в `gguf_dequant.h:158` как returned value). Если завтра сменится layout (Q4_K_v2) — найти все 30 не тривиально.
2. **31 из 37 PT_-env vars не задокументированы** для пользователя. Документация есть только в inline-комментариях исходников (ОЧЕНЬ хорошие comments — но это devops не пойдёт читать).
3. **DS2 семейство (9 env vars)** полностью скрыто — только memory `project_gigachat3_mla_bugs.md`. Поскольку GigaChat3 ещё в активном дебаге — нужен либо `docs/elbrus_report/GIGACHAT3_DEBUG_FLAGS.md`, либо `--help` экспоит их runtime.
4. **`PT_TRACE_*` и `PT_DEBUG_AUTOGRAD` — выдумка.** Их нет в коде. Audit-задание содержит неточность.
5. **Корректность зависит от env-переменных** (`PT_PER_BLOCK_SCALE`, `PT_LM_HEAD_FP`, `PT_NO_FFN_SOA`, `PT_SYSTEM_PROMPT`). Это серьёзный концептуальный долг — корректность вывода LLM не должна управляться opt-in env-переменными без явного flag-комбо в CLI / config.
6. **`PT_HAVE_*` (16 штук)** auto-detected через `try_compile` — нельзя override env'ом. Это правильно, но артикулировать в docs стоит ("если intrinsic есть в детекте — он используется; нет способа выключить без edit CMake").
