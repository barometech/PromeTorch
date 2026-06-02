# Аудит #11 — claims drift в README.md / CLAUDE.md / JOURNAL.md / TECHNICAL_SPECIFICATION.md / docs/elbrus_isa/PERFORMANCE_BY_ISA.md

Дата: 2026-06-02
HEAD: `85c0fb5` (2026-05-20)
Контекст: повторный проход после `docs/audit/2026-05-20_docs_vs_code.md` и 15+ закоммиченных правок (`597d0a9` v4 SIMD, `8261c85` OpenBLAS, `45f9acf` intrinsic auto-detect, `e92aaa3`/`1e247f8` README cleanup и т.д.).

Метод: каждое утверждение перепроверено `Grep`/`wc -l`/чтением исходников. Verdict:
- **TRUE** — claim соответствует коду на HEAD.
- **PARTIAL** — частично верно, но cifra / scope расходятся.
- **STALE** — было верно раньше, после моих фиксов больше не верно.
- **FALSE** — прямо противоречит коду.

---

## Таблица находок (30)

| # | Claim | Location | Check / file:line | Verdict | Рек. фикс |
|---|---|---|---|---|---|
| 1 | "**16 optimizers** (SGD/Adam/AdamW/RMSprop + Lion/Sophia/LAMB/Adafactor/NAdam/RAdam/Adagrad/Adadelta/Adamax/AdamKiller/ASGD/LBFGS)" | README.md L26-27, L93, L668, L1098-1101, L1314, L1362 | `Grep "class \w+ : public Optimizer" torch/optim/*.h` → 16 классов точно | **TRUE** | — |
| 2 | "16 LR schedulers" — Step/MultiStep/Exp/CosineAnnealing/Linear/Const/ReduceLROnPlateau/WarmupCosine/OneCycle/CosineAnnealingWarmRestarts/Cyclic/Polynomial/Lambda/Multiplicative/Sequential/Chained | README.md L94, L672, L1104-1106, L1314, L1363 | `Grep ^class torch/optim/lr_scheduler.h` → 16 классов (включая ReduceLROnPlateau, OneCycleLR, SequentialLR, ChainedScheduler как stand-alone не-LRScheduler-наследники) | **TRUE** | — |
| 3 | "API Reference §3 Оптимизаторы: SGD, Adam, AdamW, RMSprop" | README.md L983 | Эта секция перечисляет 4, остальной README говорит 16. Дубликат-секция не обновлена в этой сессии | **STALE** | Привести L983 к полному списку из L93, или удалить дубликат-секцию |
| 4 | "API Reference §3 LR Schedulers: StepLR..OneCycleLR" (9 имён) | README.md L984 | Та же секция: 9, должно 16 | **STALE** | Привести L984 к L94 / L1104-1106 |
| 5 | "112 backward functions" / "119 backward functions" | README.md L16, L25, L92, L93, L705, L990, L1075, L1312, L1359 | `Grep "^(struct|class)\s+\w+Backward\s*:\s*public\s+Node" torch/` → **121** в 10 файлах | **PARTIAL** | Поставить везде **121** либо "120+". Сейчас разнобой 112/119/121. |
| 6 | "**149 CUDA ядра**" (footer/structure) и "**~150 CUDA kernels** `launch_*`" | README.md L1031, L1108, L1361 | `Grep '__global__ void' aten/src/ATen/cuda/*.cu` → **149** kernels (`CUDABlas`=9, `CUDAConv`=5, `CUDAInference`=9, `CUDAKernels`=65, `CUDAQuantGemv`=17, `CUDAReduce`=18, `FlashAttention`=2, `FlashDecoding`=9, `FP16Kernels`=15). `aten_cuda_exports.def` содержит **133** `?launch_*` exports. README L1108 говорит "149 ядер: 65+18+9=92" — арифметика не сходится. | **PARTIAL** | Заменить L1108 "65+18+9" → "65 element-wise + 18 reduce + 9 BLAS + 5 conv + 17 quant + 15 FP16 + 9 flash decoding + 9 inference + 2 flash attention = 149"; в footer L1361 указать "149 `__global__` kernels (133 exported via aten_cuda_exports.def)" |
| 7 | "**~132,000 строк C++/CUDA** (114K ядро framework + 17.8K examples)" / "114,253 (torch/aten/c10/python/csrc)" | README.md L34, L1355-1357 | `find torch aten c10 python -name "*.{cpp,h,cu}" \| xargs wc -l` → **122,978** core (torch=65,216 + aten=48,610 + c10=5,675 + python=3,477). examples = **18,622**. Total ≈ 141,600 C++/CUDA + 4,637 Python. | **STALE** | L1355 = 122,978; L1356 = 18,622; total = ~141K; Python = 4,637 |
| 8 | "FlashAttention forward + backward с online softmax, O(N) memory *(временно отключён, требует доработки)*" | README.md L1115, footer | `aten/src/ATen/native/Attention.h:237` вызывает `at::cuda::can_use_flash_attention` + `at::cuda::scaled_dot_product_attention`. Wired. | **STALE/FALSE** | Удалить "временно отключён". Уточнить ограничения (head_dim ∈ {64,128}, CUDA-only) |
| 9 | "FlashAttention wiring — Headers есть, 6 known bugs, 0 callsites в sdpa_forward_cuda" в roadmap | README.md L1342 | Тот же Attention.h:239 + FlashAttention.cu:565-585 `scaled_dot_product_attention` уже принимает Q/K/V и сам вызывает forward/backward. | **STALE** | Удалить пункт из roadmap |
| 10 | "create_graph=True (double backward) — Flag есть в backward signature, но engine игнорирует" в roadmap | README.md L1345 | `torch/csrc/autograd/engine.h:282` — реальный `if (task.create_graph) { fn->apply(...) } else { NoGradGuard ... }`. README L617 правильно говорит "wired". L1345 устарел. | **STALE** | Удалить L1345 из roadmap |
| 11 | "PT_DISPATCH_FLOATING_TYPES_HALF / PT_DISPATCH_COMPLEX_TYPES — теперь ops поддерживают Half/BF16/FP8" | README.md L622-625 | `PT_DISPATCH_COMPLEX_TYPES` — **9 callsites** в `aten/src/ATen/native/cpu/MathOps.h` (8) и `LinearAlgebra.h` (1) — то есть Complex64/128 РЕАЛЬНО поддерживаются (предыдущий аудит #11 ошибался по этой части). `PT_DISPATCH_FLOATING_TYPES_HALF` — **0 callsites**: Half/BF16/FP8 в CPU ops не диспатчатся. | **PARTIAL** | Уточнить: "Complex64/128 диспатчатся через PT_DISPATCH_COMPLEX_TYPES (9 callsites). FP16/BF16/FP8 объявлены в ScalarType, но в CPU ops НЕ диспатчатся — макрос PT_DISPATCH_FLOATING_TYPES_HALF не используется." |
| 12 | "LLM serving — weights loader extension-point stub, нужно прошить с safetensors_reader.py" | README.md L687, L732 | `torch/serve/llm.h:572-601` `load_weights_()` — реальный, parsит single safetensors + sharded `model.safetensors.index.json`. Только pytorch_model.bin (pickle) не реализован. | **STALE** | Удалить "extension-point stub". Уточнить: "Safetensors single + sharded работают. pytorch_model.bin pickle — не реализован." |
| 13 | "NMCard Backend 33 PASS" / "(34 tests, MNIST 93.64%)" / "33 теста PASS, MNIST 93.64%" | README.md L520, L1143, L1220 | `grep -c "TEST(" test/cpp/test_nmcard.cpp` → **34**. README L1220 говорит "33". Accuracy: `BENCH_NMCARD.md:25` явно показывает **88.94%** (canonical), 93.64% было "from a different config" и не воспроизводимо. README L22=88.94% но L119/L520/L1143=93.64% — внутренние противоречия. | **FALSE** | L520/L1143 → "34 tests, MNIST 88.94%". L119 → 88.94%. Заменить везде 93.64% на 88.94% |
| 14 | "LinQ H1M Backend 34 tests" | README.md L1221 | `grep -c "TEST(" test/cpp/test_linq.cpp` → **35** | **PARTIAL** | L1221 → 35 |
| 15 | "TUDA (VecF, Math, BLAS) 38" tests | README.md L1219, L1229 (E8C2: 38/38 нативно) | `test_tuda.cpp`=28 + `test_tuda_standalone.cpp`=14 = **42**. README консервативен. | **PARTIAL** | L1219 → 42 |
| 16 | "**Тестов 720+** (gtest TEST/TEST_F/TEST_P across test/cpp + tests/)" | README.md L11, L36, L1223, L1365 | `Grep TEST\\( test/cpp/` → **907** test cases (905 в .cpp + custom tests/). Реально вдвое больше заявленного. | **PARTIAL** | "720+" → "900+" (или точное 907) |
| 17 | "12 примеров" (footer "Примеров: 12") | README.md L1366 | `find examples -maxdepth 1 -type d` → **19** sub-dirs (+ top-level cpp файлы). Из них 14 имеют CMakeLists.txt. | **PARTIAL** | "12" → "14 директорий с CMakeLists / 19 sub-dirs" |
| 18 | "ТЕКУЩИЙ СТАТУС: 10/10 GGUF моделей работают" | CLAUDE.md L1 | Список L4-12: 9 моделей с ✅ + qwen3-0.6B ⚠ (не работает). 9/10. | **PARTIAL** | "9/10 — qwen3-0.6B заявлен как capacity issue не code-fixable" |
| 19 | "Conv3d — stub" в Phase 4 table; "Conv3d forward — stub (возвращает нули)" в Known Issues | CLAUDE.md L82, L217 | `torch/nn/modules/conv.h:670-712` — реальный 7-loop direct conv (OpenMP-parallel). README L617 правильно говорит "Conv3d::forward real". | **STALE** | Обновить L82 → "Conv3d real (forward only, CPU only, нет Conv3dBackward)". Удалить L217. |
| 20 | "CTCLoss — throw" в Phase 4 table | CLAUDE.md L82 | `torch/nn/modules/loss.h:1599-1709` — реальный Graves DP forward+backward. README L618 правильно. | **STALE** | Обновить L82 → "CTCLoss real (требует 4-arg API)" |
| 21 | "cuDNN Integration BROKEN (headers есть, 0 callsites в torch/, не в CMakeLists aten_cuda)" | CLAUDE.md L90 | `torch/nn/modules/conv.h:432` вызывает `at::cudnn::cudnn_convolution_forward`. Callsites в rnn.h/pooling.h/normalization.h тоже. README L661-662 правильно говорит "cuDNN wiring Conv2d/BN/MaxPool/RNN/LSTM/GRU dispatch при PT_USE_CUDNN". | **STALE/FALSE** | Удалить "BROKEN, 0 callsites" из L90. Поставить: "cuDNN PARTIAL (CUDA runtime не проверен в этой среде, headers wired в conv/pool/norm/rnn)" |
| 22 | "FlashAttention BROKEN (6 подтверждённых багов, dim3(64,64) не запускается, нет callsites)" | CLAUDE.md L92, L216 | Тот же factual claim что #8/#9 — `scaled_dot_product_attention` использует FlashAttention CUDA path (Attention.h:237). | **STALE** | Привести L92/L216 к "FlashAttention wired в sdpa, head_dim 64/128, CPU fallback throws" |
| 23 | "Python bindings: no_grad() не подключён к C++ engine (BUG-C9)" | CLAUDE.md L89, L219 | README L574 говорит "BUG-C9 closed" в 2026-04-19 sprint. JOURNAL подтверждает. | **STALE** | Удалить пункт из L89/L219 (или пометить "FIXED 2026-04-19") |
| 24 | "Phase 3: Autograd DONE (real BFS, multi-output, no_grad; BUT create_graph ignored → no double-bwd)" | CLAUDE.md L81 | Engine wires create_graph (line 282). README L617 правильно. | **STALE** | Удалить "create_graph ignored" — оставить только описание DONE части |
| 25 | "Phase 16: NM Quad PARTIAL — 100× vs own scalar; max 16 cores stable; 705 tok/s на toy GPT" | CLAUDE.md L94 | MEMORY показывает дальнейший прогресс (Qwen3-4B 16-core bit-exact, real token 139.5s, BPE inference). README L426-497 описывает Qwen-4B foundation complete (12/12 ops bit-exact). | **STALE** | Обновить L94 — добавить "Qwen3-4B foundation 12/12 ops bit-exact (2026-05-12), один real token 139.5s на 16 cores" |
| 26 | "(34 tests, MNIST 93.64% на эмуляторе)" — Phase 15 NM Card | CLAUDE.md L93 | Тот же conflict 88.94 vs 93.64 — см. #13 | **FALSE** | "33 tests, MNIST 88.94%" (`grep -c TEST test/cpp/test_nmcard.cpp` = 34, но BENCH=88.94%) |
| 27 | PERFORMANCE_BY_ISA.md — "v3 ~0.5 tok/s" / "v4 ~4-6 tok/s" / "v5 10.9 tok/s" / "v6 ~12-14 tok/s" | docs/elbrus_isa/PERFORMANCE_BY_ISA.md L17-20 | v5 (10.9) — `BENCH_ELBRUS.md` подтверждает. v4 (4-6) и v6 (12-14) явно помечены "эстимат, не валидирован" (L187, L20). v3 (0.5) — теоретический. | **PARTIAL** (честно помечено) | — (документ адекватно помечает "не валидировано"). Можно добавить ссылку на отдельный TODO list. |
| 28 | PERFORMANCE_BY_ISA.md — "v4 SIMD path в q8_soa4_gemv через pmaddubsh/pmaddh/paddw" | docs/elbrus_isa/PERFORMANCE_BY_ISA.md L11, L200 | `torch/io/q8_soa_repack.h:546-547` — реальный `__builtin_e2k_pmaddubsh(W_lo, A_lo)` + `(W_hi, A_hi)` за guard `PT_E2K_VNNI_HALF`. v5+ `qpmaddubsh` на L479, L657-658, L762. | **TRUE** | — |
| 29 | "OpenBLAS поддержка" (OpenBLAS добавлен 2026-05-20 коммитами `8261c85`/`a5eb408`/`c305eeb`) | JOURNAL.md L17-18 | `CMakeLists.txt:211-271` — реальный OpenBLAS auto-detect с fallback EML→OpenBLAS→TUDA. **НЕ упомянут в README.md, CLAUDE.md, PERFORMANCE_BY_ISA.md.** | **PARTIAL** (код есть, docs драфт) | Добавить раздел OpenBLAS в README "Поддерживаемые BLAS"; в PERFORMANCE_BY_ISA.md → "EML BLAS" column добавить "или OpenBLAS на E16C/v6"; в CLAUDE.md упомянуть. |
| 30 | "PyPI ready, version 0.2.0" в JOURNAL | JOURNAL.md L68 | `pyproject.toml:version = "0.1.0a1"`; `setup.py:version="0.1.0"`. README не упоминает PyPI вообще. | **STALE/PARTIAL** | JOURNAL устарел — реальная version в pyproject.toml = 0.1.0a1. Sync. |

---

## Сводка

- **TRUE**: 3 (1, 2, 28)
- **PARTIAL**: 12 (5, 6, 11, 14, 15, 16, 17, 18, 27, 29)
- **STALE** (после моих фиксов 2026-05-20): 9 (3, 4, 7, 12, 19, 20, 22, 23, 24, 25)
- **STALE/FALSE**: 3 (8, 9, 10, 21)
- **FALSE**: 2 (13, 26)

**Главный pattern:** значительная часть docs/CLAUDE.md ссылается на статус коду до 2026-03-18 audit (Conv3d stub, FlashAttention broken, cuDNN 0 callsites, create_graph ignored, BUG-C9 open). Code-real status — всё это закрыто 2026-04-18/19 сессией.

**Регрессии после 2026-05-20 audit:**
- README L983/L984 (API Reference) НЕ обновлены — там по-прежнему 4 opt / 9 sched, хотя L93/L94/L668/L1098 — 16/16.
- L1108 (CUDA): арифметика "65+18+9=92" не сходится с "149 ядер" в той же строке.
- L520 "(34 tests, MNIST 93.64%)" — 93.64% не воспроизводимо (BENCH_NMCARD.md ясно говорит 88.94%).
- Line counts (L1355-1357) — устарели после v4/OpenBLAS/intrinsic-detect коммитов (114K → 122,978 core).
- OpenBLAS fallback path (CMakeLists.txt:211-271) добавлен в коде, но НЕТ в README/CLAUDE/PERFORMANCE_BY_ISA.
- CLAUDE.md в целом — снапшот апрельского аудита, не обновлён ни 2026-05-03 (multi-arch), ни 2026-05-20 (v4 SIMD, intrinsic auto-detect, OpenBLAS).

---

## Конкретные строки README.md под правку

| Файл:Строка | Текущий текст | Что заменить |
|---|---|---|
| README.md:11 | `tests-720%2B` badge | `tests-900%2B` (907 actual) |
| README.md:15 | "Real autograd (112 backward ops)" | "Real autograd (121 backward ops)" |
| README.md:25 | "Real autograd (112 backward ops + ...)" | "Real autograd (121 backward ops + ...)" |
| README.md:34 | "~132,000 строк C++/CUDA (114K ядро framework + 17.8K examples)" | "~141,600 строк C++/CUDA (~123K ядро + 18.6K examples)" |
| README.md:36 | "**~4,700 Python** = ~137K LOC. ... 720+ тестов." | "~4,640 Python = ~146K LOC ... 900+ тестов." |
| README.md:92 | "Backward functions \| ~1500 \| 119 + hooks + anomaly \| ~8%" | "121 + hooks + anomaly" |
| README.md:119 | "(Q16.16 fixed-point эмулятор, MNIST 93.64%)" | "MNIST 88.94%" |
| README.md:520 | "Q16.16 эмулятор (34 tests, MNIST 93.64%)" | "Q16.16 эмулятор (34 tests, MNIST 88.94%)" |
| README.md:705 | "Core autograd (112 backward + hooks + anomaly + create_graph)" | "121 backward" |
| README.md:732 | "**LLM serving engine** — forward loop + KV cache + sampling работают, load_weights_() is extension-point stub" | "load_weights_() работает для safetensors single + sharded; pytorch_model.bin pickle — не реализован" |
| README.md:767 | "**FlashAttention wiring** — headers есть, 0 callsites. Нужно подключить к sdpa_forward_cuda." | удалить (уже wired в Attention.h:237) |
| README.md:983 | "Оптимизаторы: SGD, Adam, AdamW, RMSprop." | расширить до 16 (как L93) |
| README.md:984 | "LR Schedulers: StepLR, MultiStepLR ... OneCycleLR." (9) | расширить до 16 (как L94) |
| README.md:990 | "**112 backward functions** для всех операций" | "**121 backward functions**" |
| README.md:1031 | "149 CPU операций с AVX2/NEON/E2K векторизацией" | (это OK если речь о CPU, но проверить число CPU ops отдельно — рядом в L1108 уже CUDA) |
| README.md:1108 | "### CUDA Backend — 149 ядер" + L1110-1112 "65 + 18 + 9" (sum=92, не 149) | разнести: 65 element-wise + 18 reduce + 9 BLAS + 5 conv + 17 quant + 15 FP16 + 9 flash decoding + 9 inference + 2 flash attention = 149 |
| README.md:1115 | "FlashAttention — forward + backward ... *(временно отключён)*" | "FlashAttention — wired в `scaled_dot_product_attention`, CUDA-only, head_dim ∈ {64, 128}" |
| README.md:1143 | "33 теста PASS, MNIST 93.64% accuracy" | "34 теста PASS, MNIST 88.94% accuracy" |
| README.md:1218 | "Optimizers \| 50+ \| PASS" | (нужна проверка отдельно через `test_optim.cpp`=22 + `test_optimizers.cpp`=51 → 73) |
| README.md:1219 | "TUDA (VecF, Math, BLAS) \| 38 \| PASS" | "42" (28+14) |
| README.md:1220 | "NMCard Backend \| 33 \| PASS" | "34" |
| README.md:1221 | "LinQ H1M Backend \| 34 \| PASS" | "35" |
| README.md:1223 | "**Итого** \| **720+** \| **PASS**" | "**900+** (907 актуально)" |
| README.md:1312 | "Engine, Node, Edge, 112 backward функций" | "121 backward" |
| README.md:1342 | "Средний \| FlashAttention wiring \| Headers есть, 6 known bugs, 0 callsites в sdpa_forward_cuda" | удалить строку (wired) |
| README.md:1345 | "Низкий \| create_graph=True ... но engine игнорирует" | удалить (wired в engine.h:282) |
| README.md:1355 | "Строк C++/CUDA (core framework) \| 114,253" | "122,978" |
| README.md:1356 | "Строк C++/CUDA (examples) \| 17,819" | "18,622" |
| README.md:1357 | "Строк Python \| 4,756" | "4,637" |
| README.md:1359 | "Backward функций \| 119" | "121" |
| README.md:1361 | "CUDA ядер \| ~150 launch_*" | "149 `__global__` kernels (133 exported)" |
| README.md:1365 | "Тестов \| 720+" | "900+ (907)" |
| README.md:1366 | "Примеров \| 12" | "14 (директорий с CMakeLists)" |

## Конкретные строки CLAUDE.md под правку

| Строка | Текущий | Замена |
|---|---|---|
| L1 | "10/10 GGUF моделей работают" | "9/10 (qwen3-0.6B capacity issue)" — либо удалить qwen3-0.6B из счёта |
| L81 | "Phase 3 ... BUT create_graph ignored → no double-bwd" | "Phase 3 ... create_graph wired (2026-04-19)" |
| L82 | "Phase 4 ... (Conv3d — stub, CTCLoss — throw)" | "Phase 4 ... (Conv3d real forward CPU only, CTCLoss real Graves DP)" |
| L89 | "Phase 11 ... no_grad не подключён к C++ engine — BUG-C9" | "Phase 11 PARTIAL (BUG-C9 closed 2026-04-19)" |
| L90 | "Phase 12 cuDNN BROKEN (headers есть, 0 callsites)" | "Phase 12 cuDNN PARTIAL (wired в Conv/Pool/Norm/RNN; runtime untested на этой машине)" |
| L92 | "Phase 14 FlashAttention BROKEN (6 багов, dim3(64,64), нет callsites)" | "Phase 14 FlashAttention WIRED (sdpa CUDA path), head_dim 64/128, CPU fallback throws" |
| L93 | "Phase 15 ... (34 tests, MNIST 93.64% ...)" | "(34 tests, MNIST 88.94%)" |
| L94 | "Phase 16 NM Quad PARTIAL ... 705 tok/s на toy GPT" | добавить: "Qwen3-4B foundation 12/12 ops bit-exact (2026-05-12), 1 real token 139.5s на 16 cores" |
| L216 | "FlashAttention полностью нерабочий — не использовать" | удалить (или "FlashAttention wired в sdpa") |
| L217 | "Conv3d forward — stub (возвращает нули)" | удалить |
| L219 | "Python bindings: no_grad() не подключён к C++ engine (BUG-C9)" | удалить |

## Не правится автоматически — нужен решением юзера

- TECHNICAL_SPECIFICATION.md (1327 строк) — изначальный план 2026-01-20 с пустыми чекбоксами. Решение: либо удалить, либо пометить "ARCHIVED — see README/CLAUDE for actuals".
- README.md L687 — "**Weights loader extension point (stub)**" в Что нового секции — STALE, но эта секция исторический log, можно оставить как было на 2026-04-19 контекст с пометкой "[обновлено: load_weights_ работает для safetensors с 2026-04-XX]".
- OpenBLAS — целиком новая фича без документации в README/CLAUDE/PERFORMANCE_BY_ISA — это отдельная задача, не drift.

---

## Суммари (≤300 слов)

Проверено 30 утверждений в `README.md` (1422 строки), `CLAUDE.md` (279), `JOURNAL.md` (последние entries), `TECHNICAL_SPECIFICATION.md` (архив), `docs/elbrus_isa/PERFORMANCE_BY_ISA.md` (новый). Verdict: **3 TRUE, 12 PARTIAL, 9 STALE, 3 STALE/FALSE, 2 FALSE**.

**Главная находка — CLAUDE.md полностью устарел.** Phase-table (L77-96) — снапшот апрельского аудита. Cuda Integration "BROKEN, 0 callsites" — `conv.h:432` вызывает `at::cudnn::cudnn_convolution_forward`. FlashAttention "BROKEN, нет callsites" — wired через `Attention.h:237`. Conv3d "stub" — реальный 7-loop CPU. CTCLoss "throw" — реальный Graves DP. BUG-C9 (no_grad) — закрыт 2026-04-19. create_graph "ignored" — wired в `engine.h:282`. CLAUDE.md нужен полный update под текущее состояние.

**README — частично закрытые drift'ы остались:**
1. L983/L984 API Reference — по-прежнему 4 opt / 9 sched (в L93/L94/L668/L1098 правильно 16/16).
2. L1108 "CUDA Backend — 149 ядер" + breakdown "65+18+9=92" — не сходится.
3. L1355-1357 line counts: 114,253 core → реально **122,978**; 17,819 examples → **18,622**; 4,756 Python → **4,637**.
4. L520/L1143 NMCard "93.64%" — `BENCH_NMCARD.md` показывает 88.94% canonical, 93.64% не воспроизводимо. README сам себе противоречит (L22=88.94% vs L119=93.64%).
5. L1219-1221 — test counts занижены (TUDA 38→42, NMCard 33→34, LinQ 34→35, total 720→907).
6. Backward count 112/119/121 — в одном README три разных числа.

**Полностью отсутствует в docs:** OpenBLAS fallback (commits `8261c85`/`a5eb408`/`c305eeb`) — добавлен в CMakeLists.txt:211-271 для E16C/v6 где EML несовместим. NIGHTLY drift между JOURNAL и README/CLAUDE/PERFORMANCE_BY_ISA.

**v4 SIMD path** (`pmaddubsh` в `q8_soa_repack.h:546-547`) задокументирован в PERFORMANCE_BY_ISA.md — единственный документ, который точно отражает текущий код.
