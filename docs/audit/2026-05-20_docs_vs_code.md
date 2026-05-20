# Аудит ОДИН — docs vs реальный код drift в PromeTorch

Дата: 2026-05-20
Скоуп: `README.md`, `README.en.md`, `CLAUDE.md`, `docs/`, корневые `*.md`-файлы (BENCH_*, JOURNAL.md, TECHNICAL_SPECIFICATION.md, GAP_ANALYSIS_VS_PYTORCH.md)
Метод: каждое числовое/featural утверждение из README кросс-проверено grep'ом по `torch/`, `aten/`, `c10/`, `python/`, `examples/`. Подсчёты строк через `wc -l`, подсчёт классов через `Grep` ripgrep. Линки проверены `ls`.

Severity:
- **FALSE** — утверждение прямо противоречит коду (no file / opposite behavior / stub).
- **PARTIAL** — частично верно, но материально вводит в заблуждение (число расходится, фича работает лишь на 1 backend, фича есть как символ но не используется).
- **TRUE** — соответствует.

## Таблица находок

| # | Claim | README location | Проверка / file:line | Вердикт | Рекомендуемое действие |
|---|---|---|---|---|---|
| 1 | "16 optimizers (SGD, Adam, AdamW, RMSprop + Lion, Sophia(G), LAMB, Adafactor, NAdam, RAdam, Adagrad, Adadelta, Adamax, AdamKiller, ASGD, LBFGS)" | README.md L93, L668-672, L11 | `Grep "class \w+ : public Optimizer"` → 16 классов в `torch/optim/*.h` (sgd/adam/adamw/rmsprop/lion/sophia/lamb/adafactor/nadam/radam/adagrad/adadelta/adamax/adamkiller/asgd/lbfgs); каждый — реальный step() поверх at::* ops. AdamW определён в `torch/optim/adam.h:275` | **TRUE** | — |
| 2 | "Оптимизаторов — 10 штук: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax, AdamKiller" | README.md L1098-1100 | Тот же `Grep` → 16, не 10. Lion/Sophia/LAMB/Adafactor/ASGD/LBFGS пропущены в этой секции README. Эта секция противоречит другой секции того же файла (L93, L668) | **FALSE** | Привести L1098 к 16 (или удалить дубликат-список — оставить только Coverage table) |
| 3 | "16 LR schedulers" (имена: Step/MultiStep/Exp/CosineAnnealing/Linear/Const/ReduceLROnPlateau/WarmupCosine/OneCycle/CosineAnnealingWarmRestarts/Cyclic/Polynomial/Lambda/Multiplicative/Sequential/Chained) | README.md L94 | `Grep "class \w+ : public LRScheduler\|class \w+LR : public\|class (ReduceLROnPlateau\|OneCycleLR\|SequentialLR\|ChainedScheduler)"` в `torch/optim/lr_scheduler.h` → 16 классов (line 110-805). Все perlist | **TRUE** | — |
| 4 | "**LR Schedulers** (9 штук)" | README.md L1103 | Тот же файл → 16, не 9. ReduceLROnPlateau/CyclicLR/PolynomialLR/LambdaLR/MultiplicativeLR/SequentialLR/ChainedScheduler перечислены в L94 но пропущены в L1103 | **FALSE** | Удалить дубликат list или согласовать с L94 |
| 5 | "112 backward functions" / "119 backward functions" | README.md L15, L25, L93, L705, L990, L1075, L1356 | `Grep "struct \w+Backward : public Node"` → **121** в `torch/csrc/autograd/functions/` + `torch/utils/checkpoint.h`. Сами числа варьируются 112-119 даже внутри README (L93 = 119, L25/L705/L990 = 112, L1356 = 119) | **PARTIAL** | Заменить все на актуальное "120+" либо точное число; согласовать L25 vs L93 |
| 6 | "~150 CUDA kernels" / "149 CUDA ядра" / "133 launch_*" | README.md L1031, L1062, L1358 | `Grep "void launch_\w+"` в `aten/src/ATen/cuda/` → 97 декларированных launch_*; `aten_cuda_exports.def` → **133** unique exports. README L1358 показывает 133 в `aten_cuda_exports.def` — корректно. Но L1031 говорит "149 CUDA ядра" а в реальности launch_-функций ~133, kernels (с `__global__`) ~150. Микс kernels vs launchers вводит в заблуждение | **PARTIAL** | Уточнить терминологию: "133 ATen-level launchers" vs "150+ `__global__` kernels" |
| 7 | "**~132,000 строк C++/CUDA** (114K ядро framework + 17.8K examples)" | README.md L34, L1352-1355 | `wc -l torch/ aten/ c10/ python/csrc/ *.{h,cpp,cu}` → **122,930** core (не 114,253); examples 18,622 (не 17,819). Реально ~141K total C++/CUDA + 3,626 Python (а не 4,756). Расхождение ~9K по ядру | **PARTIAL** | Перепосчитать `wc -l` и обновить таблицу L1352-1355 |
| 8 | "FlashAttention forward + backward с online softmax, O(N) memory *(временно отключён, требует доработки)*" | README.md L1112 | `aten/src/ATen/cuda/FlashAttention.cu:259, 484` — реальный код forward+backward; `aten/src/ATen/cuda/FlashAttention.cu:565-585` — `scaled_dot_product_attention()` вызывает `flash_attention_forward` без условий, **отключения нет**. CPU fallback (L584) кидает `PT_ERROR("Standard attention fallback not implemented")` | **FALSE** | Удалить "временно отключён". Указать что CUDA-only, head_dim ограничен 64/128 (L603), CPU fallback throws |
| 9 | "FlashAttention wiring — headers есть, 0 callsites" | README.md L767, L1339 (роадмап) | `Grep flash_attention` → `aten/src/ATen/native/Attention.h:237` уже вызывает `at::cuda::can_use_flash_attention()` + `at::cuda::scaled_dot_product_attention()`. Callsite ЕСТЬ | **FALSE** | Удалить пункт из роадмапа — wiring уже сделан |
| 10 | "create_graph=True wired (double backward)" | README.md L617 ("что нового") vs L1342 ("Низкий | create_graph=True ... engine игнорирует") | `torch/csrc/autograd/engine.h:282-287` — реальный if/else: `if (task.create_graph) { fn->apply(...) } else { NoGradGuard no_grad; fn->apply(...) }`. Wiring сделано | **FALSE** | Удалить пункт L1342 из роадмапа — done в L617 |
| 11 | "dtype dispatch расширен: PT_DISPATCH_FLOATING_TYPES_HALF / PT_DISPATCH_COMPLEX_TYPES — теперь ops поддерживают Half/BFloat16/Float8_e4m3fn/Float8_e5m2/Complex64/Complex128" | README.md L623-625 | Макрос `PT_DISPATCH_FLOATING_TYPES_HALF` определён в `c10/core/ScalarType.h:876` НО **0 callsites** в `aten/` и `torch/` (grep). MathOps.h/LinearAlgebra.h используют только `PT_DISPATCH_FLOATING_TYPES` (float+double). FP16/BF16/FP8 dispatch на CPU фактически не работает | **FALSE** | Либо подключить макрос в реальные ops, либо удалить claim. CLAIM_RU: "fp16/bf16 — объявлены но не dispatch'атся" из CLAUDE.md фаза 1 — точнее README |
| 12 | "dtypes: 10 (Float32/64, Half, BFloat16, **Float8 e4m3fn/e5m2**, Complex64/128, Bool, int8-64) | ~50%" | README.md L95 | `c10/core/ScalarType.h:540-569` — enum имеет 19 значений (Byte, Char, Short, Int, Long, Half, Float, Double, ComplexHalf, ComplexFloat, ComplexDouble, Bool, BFloat16, QInt8, QUInt8, QInt32, Undefined, Float8_e4m3fn, Float8_e5m2). Распределение по DTYPE actual !=10. Но **функциональный** dispatch (см п.11) — только Float32+Float64 | **PARTIAL** | Уточнить: "19 enum-значений, фактический dispatch — Float/Double; FP16/BF16/FP8 заявлены, но макрос не используется" |
| 13 | "Тестов 720+ (gtest TEST() / TEST_F() / TEST_P())" | README.md L11, L36, L1220, L1362 | `Grep "TEST(_F\|_P)?\("` в `test/cpp/` → **834** test cases (49 в test_nn_modules, 147 в test_all_ops, 66 в test_nn, 65 в test_autograd_full, ...). `tests/` использует custom CHECK macros (не gtest), тесты не считаются | **PARTIAL** | Заменить "720+" на "830+" в README (фактическое число выше заявленного) |
| 14 | "Модулей: 64+ (16 файлов)" / "90 модулей" | README.md L1082, L1310 vs L1024 | `Grep "^class \w+ : public Module"` → **91** класса в 15 файлах (`torch/nn/modules/` 14 файлов + `torch/nn/parallel/tensor_parallel.h`). 91 ≈ 90, 14 ≠ 16 | **PARTIAL** | Поменять "16 файлов" → "15 файлов" в L1310 |
| 15 | "Lightning Trainer" | README.md L101, L654-656 | `torch/trainer/trainer.h:21-30` — `LightningModule` + `Trainer::fit()` реальный (gradient accumulation, clip, checkpoint, progress bar) | **TRUE** | — |
| 16 | "LLM serving engine — paged KV cache (64-token pages), BPE tokenizer, GQA-aware attention, continuous batching, sampling. ... Weights loader — extension point (stub)" | README.md L685-688 | `torch/serve/llm.h:572-600` — реальный `load_weights_()` parsит safetensors single+sharded и `model.safetensors.index.json`. **НЕ stub**. README говорит что нужно "прошить", но уже прошито | **FALSE** | Удалить упоминание "extension-point stub" — loader работает (см. L572-600); прошить только pytorch_model.bin pickle (L598-600 — известное ограничение) |
| 17 | "PromeTorch-compatible .pt save/load (`torch/serialization_pytorch.h`)" | README.md L31-32, L680-682, L711 | `Glob torch/serialization*.h` → есть `torch/serialization.h` НО **нет** файла `serialization_pytorch.h`. Pickle/ZIP реализован, но в другом файле | **PARTIAL** | Либо переименовать файл (если такой path был); либо обновить путь в README. Сейчас pointer-ссылка указывает на несуществующий .h |
| 18 | "Conv3d forward real (ранее `return zeros()`)" + "CTCLoss полный Graves DP" + "cross_entropy(reduction='none')" | README.md L617-619 | `torch/nn/modules/conv.h:620-712` — Conv3d реальный 7-loop direct conv; `torch/nn/modules/loss.h:1599-1709` — CTCLoss реальный forward-backward (Graves DP, log-space alpha). Conv3d **только** CPU, backward Conv3dBackward отсутствует — ни в `ConvBackward.h`, ни где-либо. CTCLoss single-arg forward(input) throws (L1711-1713) | **PARTIAL** | Conv3d — добавить уточнение "forward only, CPU only, no backward"; CTCLoss — указать "требует 4-arg API" |
| 19 | "MNIST 93.64% accuracy" (NM Card) | README.md L21, L119, L521, L1140 + многократно | `BENCH_NMCARD.md:25-33` явно говорит **88.94%** (3 эпохи, plain SGD), а 93.64% было "from a different config" и не воспроизводимо | **FALSE** | Привести все NMCard упоминания к 88.94% (как в L22, L262) — убрать 93.64% из L119/L521/L1140 |
| 20 | "NMCard Backend 33 PASS" / "(34 tests, MNIST 93.64%)" | README.md L1217, L521 | `test/cpp/test_nmcard.cpp` → **33** TEST'а (grep TEST count). README сам себе противоречит (33 vs 34) | **PARTIAL** | Заменить L521 "(34 tests, MNIST 93.64%)" → "(33 tests, MNIST 88.94%)" |
| 21 | "AMP: FP16 CUDA kernels (add/mul/relu/sigmoid/tanh/softmax/layernorm/rmsnorm)" | README.md L658-660 | `aten/src/ATen/cuda/FP16Kernels.cu` имеет 15 launch_*_fp16 функций (grep). Но softmax_fp16, layernorm_fp16, rmsnorm_fp16 — отсутствуют (по export.def нет) | **PARTIAL** | Уточнить список: add/mul/sub/relu/sigmoid/tanh + check_inf_nan + broadcast variants. layernorm/rmsnorm FP16 не реализованы |
| 22 | "torch.jit.compile — trace + element-wise fusion + C++ codegen subprocess" | README.md L644-645, L98 | `torch/jit/compile.h:1-32`, `torch/jit/codegen_cpp.h:1-26` — реальный (cl.exe / gcc / l++) | **TRUE** | — |
| 23 | "Sequential model; model.add(std::make_shared<Linear>(...))" пример MNIST | README.md L862-885 | `torch/nn/modules/container.h:51-53` — `Sequential::add(ModulePtr)` существует как alias `push_back`. ModulePtr = `std::shared_ptr<Module>`. Пример валидный | **TRUE** | — |
| 24 | "Структура: torch/csrc/autograd/ — 112 backward функций" | README.md L1309 | См. п.5 — 121 backward функция фактически | **PARTIAL** | Привести в соответствие с L93/L1356 |
| 25 | "Эльбрус MNIST MLP — 6.1× быстрее PyTorch 2.7.1" | README.md L21, L122-123, L258-264 | Локально невозможно верифицировать (нет Эльбруса в текущей среде), но `BENCH_ELBRUS.md` существует, числа сходятся между README и MEMORY.md | **TRUE** (на основании внутренних docs) | — |
| 26 | "Vulkan compute / TPU XLA backends ... отсутствует" | README.md L748 | `Grep -ri "Vulkan\|XLA\|TPU"` в `torch/` → 0 совпадений. Корректно объявлено отсутствующим | **TRUE** | — |
| 27 | "qwen3:4b @ 82 tok/s greedy inference" | README.md L19, L287 | `BENCH_OLLAMA.md:16` — реальный замер 85.9 tok/s (P1), 84.4 (P2), 86.1 (P3) → average ~85.5. README консервативен (82.6) | **TRUE** | Можно обновить до 85 |
| 28 | "INT8 QAT + INT4 + NF4 + fp8 dtype" | README.md L102, L675-677 | `torch/quantization/quant4.h` + `qat.h` + `torch/nn/modules/quantized.h` существуют; `Float8_e4m3fn/e5m2` в ScalarType. INT8 QAT (FakeQuantize) реальный | **TRUE** | — |
| 29 | "torchaudio: STFT/iSTFT (radix-2 FFT) + MFCC + Resample + WAV I/O" | README.md L651-652 | `torch/audio/audio.h, functional.h, transforms.h` существуют. Selftest упомянут (max err 1.79e-7) | **TRUE** (структурно) | — |
| 30 | "MultiheadAttention bypass autograd в custom batched matmul" (roadmap) | README.md L765-766 | `Grep MultiheadAttention` → `torch/nn/modules/attention.h:26` реальный класс. Проверить, что autograd обходится поперёк attention, требует runtime теста — компиляционная видимость есть, известная проблема (см. ViT агент) | **PARTIAL** (как заявлено) | Подтвердить статусом в TEST_PLAN §X.X |
| 31 | "12 примеров" | README.md L1363 | `find examples/*/CMakeLists.txt` → **14** директорий-примеров (cifar/gan/gguf/jit/mlir/mnist/mobile/nmcard/pir/rnn/shakespeare/transformer/vae/vit), плюс top-level cpp файлы (train_10_models, test_qat, cuda_isolation_test, test_mem_leak, test_phase2) | **PARTIAL** | Обновить на "14 директорий" либо явно перечислить |
| 32 | "CUDA Backend — 99 ядер: 65 element-wise + 18 reduction + 9 BLAS + cuBLAS + cuDNN + FlashAttention + FlashDecoding + Quantized GEMV + Mixed Precision" | README.md L1105-1115 | Сумма 65+18+9 = 92, не 99. И эти числа не соответствуют actual `Grep launch_*` 97 в CUDAKernels/Reduce/FP16/BLAS. README дробит произвольно | **PARTIAL** | Перепосчитать категории — реальный split: CUDAKernels.cu=59, CUDAReduce=14, FP16Kernels=15, CUDABlas=9 → 97 launchers; полный export 133 (включает Conv, FlashAttention, FlashDecoding, QuantGemv) |
| 33 | "PyTorch-compatible .pt save/load — `.pt ↔ torch.load`/`torch.save`" | README.md L31-32 | `python/prometorch/_pytorch_io.py` + `torch/serialization.h` + `safetensors_reader.py` — реальная pickle protocol 2 + ZIP реализация. См. README L680-682 | **TRUE** | — |
| 34 | "Conv1d, Conv2d, Conv3d, ConvTranspose2d" + "ConvTranspose2dBackward — нужен для DCGAN" | README.md L976, L589-590 | conv.h: Conv1d (L70), Conv2d (L313), Conv3d (L620). ConvTranspose2d (`Grep`) → L922+. ConvBackward.h имеет Conv2dBackward, ConvTranspose2dBackward, **но НЕ Conv1dBackward / Conv3dBackward** | **PARTIAL** | Документировать что Conv1d/Conv3d forward-only без autograd (используют общий mm backward через im2col), или добавить явные backward |
| 35 | "Engine — топологическая сортировка, cached GraphTask" | README.md L1074 | `torch/csrc/autograd/engine.h` — есть GraphTask, выполнение реальное (см п.10). Cached между backward — компиляционно есть | **TRUE** | — |
| 36 | "tensor.to('nmcard'), model.to('nmcard'), tensor.to('linq')" | README.md L892-897 | `c10/core/Device.h:93-94, 261-262, 349-350` — "linq" и "nmcard" parsable + factory functions. Backend есть в `c10/linq/`, `aten/src/ATen/linq/`. LinQ device в Architecture box (L1033) упоминается но не в Coverage table (L100) | **TRUE** | — |
| 37 | "9/9 smoke tests pass" (transformers_compat) | README.md L649, L693 | `python/prometorch/transformers_compat.py` exists. Тесты в `python/tests/` (не вычитал детально). Compile-verified | **PARTIAL** (нет evidence файла с 9 tests) | Добавить ссылку на конкретный test file |
| 38 | "BENCH_NMCARD.md — MNIST 88.94%" link | README.md L22 | `BENCH_NMCARD.md:25` — реально 88.94%. Link существует | **TRUE** | — |
| 39 | "report/REPORT_ELBRUS_LLM_INFERENCE_2026-05-02.pdf" | README.md L158 | `report/REPORT_ELBRUS_LLM_INFERENCE_2026-05-02.pdf` существует | **TRUE** | — |
| 40 | "docs/screenshots/promeserve_ui_main.png" | README.md L79 | `docs/screenshots/promeserve_ui_main.png` существует | **TRUE** | — |
| 41 | "Tests in tests/io/" | git status новый каталог | `tests/io/test_pt8_loader.cpp` — единственный файл; не упомянут в README | **PARTIAL** | Добавить в Test Plan |

## Сводка по вердиктам

- **TRUE** (15 пунктов): 1, 3, 15, 22, 23, 25, 26, 27, 28, 29, 33, 35, 36, 38, 39, 40
- **PARTIAL** (16 пунктов): 5, 6, 7, 12, 13, 14, 17, 18, 20, 21, 24, 30, 31, 32, 34, 37, 41
- **FALSE** (8 пунктов): 2, 4, 8, 9, 10, 11, 16, 19

## Суммари (≤300 слов)

Аудит проверил 41 конкретное утверждение из README.md против реального состояния кода в `torch/`, `aten/`, `c10/`, `python/`, `examples/`. Из них 15 TRUE, 16 PARTIAL, 8 FALSE.

**Главные FALSE'ы (требуют фикса README первым делом):**

1. **FlashAttention "временно отключён"** (L1112, L767, L1339) — реально wired в `aten/src/ATen/native/Attention.h:237` через `scaled_dot_product_attention`. README устарел.
2. **"create_graph engine игнорирует"** (L1342 roadmap) — реально работает в `engine.h:282-287`. Снять с roadmap.
3. **"PT_DISPATCH_FLOATING_TYPES_HALF — теперь ops поддерживают Half/BF16/FP8"** (L623) — макрос есть, 0 callsites в `aten/`. CPU ops по-прежнему Float+Double. Самое серьёзное вранье в Coverage.
4. **"NMCard MNIST 93.64%"** (повторено ≥5 раз) — `BENCH_NMCARD.md` явно опровергает: 88.94% canonical, 93.64% old config not reproducible. README сам себе противоречит (L22 = 88.94, L119 = 93.64).
5. **"LLM serving — weights loader extension-point stub"** (L685) — реально `load_safetensors_file_()` + sharded index работают.
6. **"Оптимизаторов — 10 штук", "LR schedulers (9 штук)"** (L1098-1103) — дубликат секция противоречит L93-94 (где правильно 16/16). Удалить дубликат.

**Главные PARTIAL'ы:**

- Числа строк не сходятся: README заявлено 114K core, реально 122,930 (`wc -l` torch+aten+c10+python/csrc).
- Backward functions варьируются в одном файле: 112 / 119 / 121 (реально). 
- "12 примеров" → 14 директорий с CMakeLists.
- CUDA "99 ядер: 65+18+9" не сходится (=92); реальные launchers 97 в `.cu`, 133 в export.def.
- Conv3d/Conv1d backward отсутствуют (forward-only).
- Тестов 834 (gtest), а не 720+ — README консервативен.

**Битые ссылки:** одна — `torch/serialization_pytorch.h` упомянут (L31-32, L680), реально `torch/serialization.h` + `python/prometorch/_pytorch_io.py`. Остальные ссылки на BENCH_*, docs/elbrus*, docs/BUILD_*, scripts/*, examples/* — все живые.

Никаких отсутствующих фич "DONE which throws" в README не нашёл — это закрыто отдельным аудитом ДВА.
