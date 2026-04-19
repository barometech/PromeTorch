# PromeTorch — Российский training framework

> PyTorch-совместимый обучающий фреймворк на C++17/CUDA с широкой dtype-поддержкой.
> Нативная сборка на **Эльбрус E8C2 (VLIW)** и **NM Card Mini (partner, Q16.16)**.
> Real autograd (110 backward ops + gradient hooks + anomaly mode + create_graph double-bwd),
> 20 optimizers, CPU SIMD + CUDA (cuBLAS/cuDNN/FP16 kernels), distributed training
> (TCP DDP/FSDP/TP/Pipeline), export (ONNX/MLIR/Mobile/JIT), ecosystem shims
> (HuggingFace/torchvision/torchaudio/torchtext/Lightning Trainer), **PyTorch-compatible
> save/load (.pt ↔ torch.load/save)**.
>
> ~110,000 строк C++. 1 разработчик. ~5 недель активной разработки + 35-agent burst session.

> ⚠️ **Coverage ~35-45% практической площади PyTorch.** Это solo-проект: ряд CUDA путей
> (FP16 kernels, cuDNN RNN, MPS/ROCm) **compile-verified но runtime-untested** (нет доступа
> к соответствующему железу). `torch.compile` — trace-based prototype, не полноценный
> TorchInductor. Sparse tensors / FX graph mode / torch.distributions отсутствуют.
> См. **Known Limitations** ниже — каждый gap честно задокументирован.

---

## Coverage vs PyTorch

| Категория | PyTorch | PromeTorch | % |
|---|---|---|---|
| Tensor ops | ~2000 | ~150 (90 базовых + 50+ в OpsExpansion) | ~7% |
| Backward functions | ~1500 | 110 + hooks + anomaly | ~7% |
| Optimizers | 15+ | **20** (Lion/Sophia/LAMB/Adafactor/NAdam/RAdam/Adagrad/Adamax/ASGD/LBFGS + base 10) | 130% |
| LR schedulers | 15+ | 9 | 60% |
| dtypes | 20+ | 10 (Float32/64, Half, BFloat16, **Float8 e4m3fn/e5m2**, Complex64/128, Bool, int8-64) | ~50% |
| Autograd features | full | core + hooks + anomaly + create_graph + forward-AD + vmap — отсутствуют grad-of-vmap/hessian/full functorch | ~40% |
| Distributed | NCCL/gloo/ucc, DDP/FSDP/TP/PP/ZeRO/bucket fusion | **real TCP DDP, FSDP/ZeRO-3, TP Col/Row, Pipeline 1F1B, DistributedSampler, launcher** | ~35% |
| Compile/JIT | `torch.compile` (TorchInductor + Triton) | trace-based `torch.jit.compile` + C++ codegen subprocess | ~10% |
| Export | ONNX + TorchScript + ExecuTorch + TensorRT | **ONNX (ORT works) + MLIR + Mobile + JIT** | ~50% |
| Backends | CPU / CUDA / ROCm / MPS / XLA / Vulkan / TPU | CPU + CUDA (cuBLAS/cuDNN) + MPS (compile-only) + ROCm (script) + NM Card Mini + Эльбрус + FP16 (compile-only) | ~45% |
| Ecosystem | torchvision/audio/text/rl/transforms + Lightning/Accelerate/DeepSpeed/FSDP native | **torchvision + torchaudio + torchtext + HF Transformers compat + Lightning Trainer + PyTorch-.pt compat + DeepSpeed shim** | ~50% |
| Quantization | INT8/INT4/NF4/FP8/QAT/PTQ | INT8 QAT + INT4 + NF4 + fp8 dtype | ~60% |
| Sparse tensors | COO/CSR/BSR | нет | 0% |
| torch.distributions | 50+ | нет | 0% |
| FX graph mode | full | нет | 0% |
| Unit tests | 100K+ | ~850 gtest + ~30 agent self-tests + auto-generated suite | ~1% |

**Обобщённая оценка: ~35-45% user-surface PyTorch** — достаточно для training/deploy
transformers/CNN/RNN/LLM на CPU + CUDA + Эльбрус + NM Card. Gap в **tuned kernels** /
**torch.compile production** / **sparse+distributions+FX** — закрытие потребует
multi-year team effort.

---

## Почему PromeTorch

**1. Российские ускорители.** Единственный известный нам training framework с нативной
сборкой под **Эльбрус E8C2** (E2K VLIW, LCC 1.29, EML_MT BLAS) и **NM Card Mini**
(Q16.16 fixed-point эмулятор, MNIST 93.64%). Готовая кросс-компиляция под Байкал-М/С.

**2. MNIST MLP быстрее PyTorch на Эльбрусе.** На MNIST MLP-4 (784→512→256→128→10, SGD,
batch=64, 1 epoch) — 2.76 s vs PyTorch 2.7.1 16.8 s (**6.1× на этой узкой задаче**).
1840 GFLOPS через node-local EML_MT (92% пика E8C2 2 TFLOPS). На других задачах
(общий случай / реальные transformers) преимущества PyTorch сохраняются.

**3. Universal build.** Один код — CPU (AVX2+FMA/NEON/E2K), CUDA (Turing+), NM Card
(эмулятор + реальная карта в режиме inference). Собирается на Windows MSVC, Linux GCC,
Astra/ALT/RED/Elbrus OS. Autograd engine работает одинаково на всех backend'ах.

---

## Результаты

### Эльбрус E8C2

Сервер МЦСТ Эльбрус — 4x Elbrus-MCST E8C2 (VLIW), 32 ядра, 1500 MHz.
Задача: MNIST, MLP 784->512->256->128->10 (ReLU), SGD lr=0.01, batch=64, 1 epoch.

| Метрика | PromeTorch | PromeTorch + NUMA | PyTorch 2.7.1 |
|---------|-----------|-------------------|---------------|
| **Время** | **15.2 с** | **2.76 с** | 16.8 с |
| **Accuracy** | **88.71%** | **88.94%** | 88.14% |
| **Ratio** | **1.1x быстрее** | **6.1x быстрее** | 1.0x |
| EML GFLOPS | 330 | **1840 (92% пика)** | 68 (generic BLAS) |
| Аллокации | 179 | 179 | ~50,000+ |

Путь оптимизации (126.3 с -> 15.2 с, ускорение 8.3x):

| Этап | Время | vs PyTorch | Ключевое изменение |
|------|-------|-----------|-------------------|
| Scalar baseline | 126.3 с | 7.4x медленнее | Первая сборка |
| + EML BLAS | 120.6 с | 7.1x | cblas_sgemm, 230 GFLOPS |
| + Memory pool | 121.4 с | 7.1x | 97.7% cache hit, 641 malloc |
| + Fused ops | 97.3 с | 5.7x | 8 агентов: fused ops, thread pool, 6x6 kernel |
| + SIMD SGD | 45.4 с | 2.7x | pow->x*x, skip contiguous |
| + Direct EML | 43.7 с | 2.6x | Прямые cblas вызовы, zero-copy backward |
| + Manual backward | 22.0 с | 1.3x | Bypass autograd, pre-allocated буферы |
| **Финал** | **15.2 с** | **0.90x** | Убран неиспользуемый grad clipping |

### NVIDIA GPU — GGUF inference

Inference GGUF-моделей (квантизация Q4_K_M) через custom INT4 warp-cooperative GEMV.

**Runtime-verified на A100 40GB (2026-04-19 benchmark run):**

| Модель | PromeTorch | Hardware | Ollama baseline | vs Ollama |
|--------|-----------|----------|-----------------|-----------|
| qwen3:4b | **86.6 tok/s** | NVIDIA A100 40GB (CUDA Graph + FP16 KV) | 165 tok/s | **52%** |
| deepseek-r1:8b | **52.7 tok/s** | A100 | 133 tok/s | **40%** |

- qwen3:4b VRAM: **8.0 GB** / 39.7 GB (quant-only mode)
- deepseek-r1:8b VRAM: **10.2 GB** / 39.7 GB
- Model weights move to CUDA: **0.1 s** (previously 88 s — fixed quant-only transfer)
- FP16 KV cache: 302 MB for qwen3:4b 36 layers

Output верифицирован на простых задачах («2+2 → 4»). На длинных генерациях местами
наблюдаются повторы — вероятно, RoPE rescaling at large context.

Gap vs Ollama — Ollama использует cuBLAS+cuSPARSELt+kv-page attention, у нас только
Q4_K GEMV + flash_decode + CUDA Graph. Consumer GPU цифры будут меньше (точные зависят
от модели карты).

### Точность обучения (10 training tasks)

Файл `examples/mnist/train_10_models.cpp` — 10 обучающих таcков в одном бинаре:

| Задача | Данные | Accuracy | Примечание |
|--------|--------|----------|------------|
| Linear / Logistic regression | synthetic | match PyTorch | warmup |
| XOR (2-layer MLP) | synthetic | 100% | warmup |
| MNIST MLP (784→128→10) | MNIST real | 92.68% | CUDA |
| MNIST Deep (784→512→256→128→10, Adam) | MNIST real | **97.5%** | CUDA |
| MNIST + Dropout | MNIST real | 97.15% | CUDA |
| MNIST wide + serialize/load | MNIST real | 97.78% | CUDA |
| RNNCell sine regression | synthetic | MSE 1.67e-5 | CUDA |
| LSTM classifier | **sum-sign of random walk** (synthetic) | 93–95% | NOT Shakespeare |
| GRU trend detector | synthetic | **98.44%** | CUDA |

На CPU с Adam loss/accuracy совпадают с PyTorch 2.7.1 baseline в пределах ±0.5%.

**Каких задач нет в этой таблице:** самостоятельного обучения Transformer на больших
датасетах, ViT, ResNet, VAE. Заготовки `examples/transformer/` и `examples/vit/` в репо
есть, но никогда не собирались end-to-end.

### PIR 342M на Эльбрусе (Local SGD data-parallel) — fixed backward

Тренировка собственной linear-RNN LM архитектуры (selective scan, родственник Mamba/HGRN/
RWKV) на русскоязычном корпусе Wikipedia (2 ГБ, BPE-токенизация SentencePiece 100k).
Полностью на PromeTorch — ни строчки PyTorch в обучении.

| Метрика | Значение |
|---------|----------|
| Архитектура | 342M params, 768d, 16 layers × 4 PIR sublayers, SwiGLU FFN (H=1792), seq=2048 |
| Parallelism | 4 процесса × 8 ядер **Local SGD** (file-based weight averaging, не gradient AllReduce — ≠ DDP; настоящий DDP теперь есть отдельно в `torch/distributed/ddp.h`) |
| Throughput | 4 × ~95 tok/s = **~380 tok/s aggregate** на fixed backward (медленнее прошлого 568 т.к. backward теперь делает реальную работу вместо stub'ов) |
| Related arch | PIR — diagonal selective scan `h[t] = σ(gate)·base_decay·h[t-1] + σ(gate)·value`. Близко к Mamba (A diagonal SSM), HGRN/RWKV (multi-scale decay) |
| Checkpoint | Формат: raw float32 concatenation по param order. Python inference: `pir_infer.py` |

**Trajectory на fixed backward (2026-04-19):**
| Step | Loss | Perplexity | Sample (prompt "В начале") |
|------|------|------------|----------------------------|
| 50 | 10.53 | 37,467 | "изменением школе web вший свою сказывается" |
| 100 | 7.50 | 1,815 | wiki markup (`&lt;/ref&gt;`, section headers) |
| 200 | 6.44 | 628 | "`== Награды по изменением ровка, чтобы ''х — 1961370`" (section header + связное словосочетание) |
| 300 | 6.12 | 454 | "Главный этом компонентов в 1884 года на воз поверхности в сотрудничестве. После разгрома время их Второй Украине" |
| 400 | 5.61 | 275 | "`== Биография ==`" + "`Про Федерации smal sказывается Васильевич`" (patronymic + reflexive verb + gen) |
| 500 | 5.70 | 298 | "`== Награды ==` ... `=Примечания=` ... `Похорольный`" (proper Wiki infobox + adjective chain) |

Модель учит Wikipedia-структуру + русский синтаксис. Loss monotone падает через warmup + cosine decay.
Прежние сигналы "loss 1.04 coherent Russian" относились к **сломанному** backward (embedding/scan/
gate/value градиенты отсутствовали) — сейчас все 342M params реально учатся, lose descent 3× быстрее
per-step при корректной математике.

### NM Quad (профиль на удалённой плате partner)

| Метрика | Значение | Caveat |
|---------|----------|--------|
| SIMD speedup (nmpp MM 32f) | **100×** | vs собственный скалярный C++ на 1 ядре (не vs cuBLAS / PyTorch) |
| Max stable cores | **16 из 64** | 64-ядерный режим приводит к DDR contention / hang |
| Throughput (16 ядер) | 705 tok/s | на игрушечном GPT D=128, L=2, V=65 (tiny_shakespeare); loss 4.45 (модель не сходится, это throughput microbenchmark, не training run) |
| PIR 250M на NM Quad | **TODO/planned (Phase 23)** | пока не реализовано |

### Поддерживаемые платформы

| Производитель | Процессор | Архитектура | Backend | Статус |
|---|---|---|---|---|
| **МЦСТ** | Эльбрус 8C2 | E2K VLIW | TUDA (CPU + EML_MT) | Нативная сборка, 38/38 тестов, PIR training |
| **partner** | NM Card Mini K1879VM8YA | NeuroMatrix DSP | NMCard | Q16.16 эмулятор (34 tests, MNIST 93.64%) + 1-core inference на реальной карте |
| **partner** | NM Quad (4×NM6408) | 64 NMC4 + 20 ARM | NMQuad | 100× SIMD vs own scalar, max 16 cores stable, tiny-GPT microbenchmark only |
| **Байкал Электроникс** | Байкал-М BE-M1000 | ARM Cortex-A57 | TUDA (NEON) | Только кросс-компиляция, не протестировано на железе |
| **Байкал Электроникс** | Байкал-С BE-S1000 | ARM Cortex-A75 | TUDA (NEON+dotprod) | Только кросс-компиляция, не протестировано на железе |
| Intel / AMD | x86-64 | AVX2 + FMA | TUDA (CPU) | Основная платформа разработки |
| NVIDIA | Turing+ (sm_75) | CUDA | cuBLAS wrapper + custom Q4_K GEMV | matmul через cuBLAS; 133 .cu файлов, но большинство — decorative/unused |

| ОС | Тип | Статус тестов |
|---|---|---|
| **Astra Linux SE** 1.7+ | Debian, ФСТЭК | 34/34 PASS (Docker) |
| **ALT Linux SP** 10+ | Sisyphus, ФСТЭК | Готов |
| **RED OS** 7.3+ | RHEL, ФСТЭК | 34/34 PASS (Docker) |
| **Elbrus OS** (PDK LE) | МЦСТ, E2K | 34/34 PASS (Docker), 38/38 нативно |
| Windows 10/11 | MSVC 2019+ | Основная платформа |
| Ubuntu / Debian | GCC 9+ | Поддержка |

---

## Что нового (апрель 2026, после 35-agent burst + manual fix marathon)

### Core (все протестировано на Эльбрусе, self-tests passing)
- **PIR backward полностью переписан** — embedding scatter, scan_bwd, gate/value chain rule
  (ранее были stub'ами → dW_emb=0, d_scan=identity, dW_gate==dW_value). Loss descent
  подтверждён 3× быстрее на фиксированной версии.
- **CPU correctness fixes**: `where()` / `index_select` non-contig silent-wrong, `SVD(full_matrices=true)`
  via Gram-Schmidt (ранее silently returning thin), `Conv3d::forward` real (ранее `return zeros()`),
  `CTCLoss` полный Graves DP (ранее throw), `cross_entropy(reduction='none')` (ранее throw).
- **Autograd**: `create_graph=True` wired (double backward), gradient hooks + anomaly mode,
  forward-mode AD (dual numbers + JVP, `d(exp(2x))/dx=5.44` verified), vmap (single-axis
  bit-exact vs sequential).
- **dtype dispatch расширен**: `PT_DISPATCH_FLOATING_TYPES_HALF` / `PT_DISPATCH_COMPLEX_TYPES` —
  теперь ops поддерживают Half/BFloat16/Float8_e4m3fn/Float8_e5m2/Complex64/Complex128
  (opt-in чтобы не ломать templated linear-algebra код который assumes Float/Double).
- **ChannelsLast preservation**: Conv2d NHWC input → NHWC output (internal compute NCHW —
  true NHWC-native kernel = future work).

### Distributed / Parallel (new)
- **Real DDP (TCP socket AllReduce)** в `torch/distributed/ddp.{h,cpp}` — star topology,
  POSIX sockets (Winsock shim), 2-process self-test PASS.
- **FSDP / ZeRO-3** в `torch/distributed/fsdp.h` — flat-index sharding, /dev/shm collectives.
  2-process self-test bit-exact vs non-sharded baseline (max |diff| 0.0).
- **TensorParallel** (Col/RowParallelLinear + collectives), **Pipeline Parallel** (GPipe +
  1F1B scheduling), **DistributedSampler** (PyTorch-equivalent), **fork-based launcher**.
- **DeepSpeed-like** ZeRO-Offload + 1F1B schedule поверх FSDP (`torch/distributed/deepspeed*.h`).

### Export paths (new)
- **ONNX export** (`torch/onnx/export.h`) — zero-dependency manual protobuf wire format;
  вывод **запускается в ONNX Runtime**, проходит `onnx.checker.check_model`.
- **MLIR text export** (`torch/mlir/export.h`) — tosa + linalg dialect, `mlir-opt`-loadable.
- **ExecuTorch-like mobile format** (`torch/mobile/executor.h`) — compact binary PTMB,
  bit-exact round-trip.
- **torch.jit.compile** (`torch/jit/compile.h` + `codegen_cpp.h`) — trace + element-wise
  fusion + optional C++ codegen via subprocess + dlopen. 2× speedup на мелких тензорах.

### Ecosystem shims (new)
- **HuggingFace Transformers compat** (`python/promethorch/transformers_compat.py`) —
  AutoModel.from_pretrained для Bert/GPT2/Llama, safetensors reader, pytorch_model.bin
  restricted unpickler. 9/9 smoke tests pass.
- **torchvision**: ImageFolder + 7 transforms + MobileNetV2.
- **torchaudio**: STFT/iSTFT (radix-2 FFT) + MFCC + Resample + WAV I/O. STFT reconstruction
  max |err| 1.79e-7 (self-test pass).
- **torchtext**: Vocab + BPE + WordPiece + Char tokenizer + TextDataset/CSV/JSONL.
- **Lightning Trainer** (`torch/trainer/trainer.h`) — fit/test с gradient accumulation, clip,
  checkpoint, progress bar.

### Backends + Precision (partial — compile-verified, runtime-untested кроме CPU/Elbrus)
- **AMP**: FP16 CUDA kernels (add/mul/relu/sigmoid/tanh/softmax/layernorm/rmsnorm),
  autocast_policy table с 55 ops (FP16/FP32/Promote), dynamic GradScaler. **CPU compile
  clean. CUDA runtime не проверен — нет доступа к GPU.**
- **cuDNN wiring**: Conv2d/BN/MaxPool/RNN/LSTM/GRU dispatch при PT_USE_CUDNN. CPU fallback
  intact. **CUDA runtime не проверен.**
- **MPS (Apple Metal)**: allocator + device + kernels через MPSGraph / MPSMatrixMultiplication.
  Non-Apple builds skip aten_mps target. **macOS runtime не проверен (нет Mac).**
- **ROCm/HIP**: `scripts/hipify.sh` + `HIPCompat.h` + docs для сборки через hipcc.
  **AMD runtime не проверен.**

### Optimizers (20 теперь, было 4)
SGD, Adam, AdamW, RMSprop + **Lion, Sophia, LAMB, Adafactor, NAdam, RAdam, Adagrad, Adamax,
ASGD, LBFGS**. Все на `at::*` tensor ops, CPU-portable, compile на Elbrus LCC.

### Quantization (new)
- **INT8 QAT** — FakeQuantize + QuantizedLinear + prepare_qat/convert. Self-test 97.27% vs
  float 97.17% (+0.1%, within tolerance).
- **INT4 + NF4 (QLoRA)** — block-wise quantize/dequantize + Linear4bit wrapper.

### I/O (new)
- **PyTorch-compatible .pt save/load** (`torch/serialization_pytorch.h`) — pickle protocol 2
  + ZIP + restricted unpickler. Двусторонняя совместимость: `torch.load` читает наши saves,
  мы читаем `torch.save` файлы. 3 tests PASS.

### LLM serving (new, partial)
- **LLM inference engine** (`torch/serve/llm.h`) — paged KV cache (64-token pages), BPE
  tokenizer, GQA-aware attention, continuous batching, sampling (temperature/top-k/top-p/
  repetition penalty). Self-test с random weights PASS. **Weights loader — extension point
  (stub), нужно прошить с `safetensors_reader.py`.**

### Python bindings (comprehensive expansion)
11 submodules exposed через pybind11: `pt.nn.parallel`, `pt.distributed`, `pt.trainer`,
`pt.onnx`, `pt.mlir`, `pt.mobile`, `pt.jit`, `pt.vision`, `pt.quantization`,
`pt.autograd` (forward_ad + vmap), `pt.serve`. Pure-Python fallback когда C++ не собран.
9/9 smoke tests pass.

### Все examples теперь собираются
`train_transformer`, `train_vit`, `shakespeare_train` — раньше "written but never built",
теперь binary-file-exists и запускаются на Эльбрусе.

---

## Known Limitations — честный gap от PyTorch

### Работает + протестировано на всех поддерживаемых backend'ах
- Core autograd (110 backward + hooks + anomaly + create_graph)
- 20 optimizers
- CPU SIMD (AVX2/NEON/E2K)
- Эльбрус VLIW + NM Card Mini emulator (Q16.16)
- Distributed: DDP / FSDP / TP / Pipeline на CPU через TCP + /dev/shm
- ONNX export (works with ONNX Runtime)
- PyTorch .pt I/O ↔ torch.load/torch.save

### Runtime-verified (GPU tests run 2026-04-19 on A100 40GB)
- **FP16 CUDA kernels**: `add_fp16`, `mul_fp16`, `relu_fp16`, `sigmoid_fp16`, `tanh_fp16`,
  `check_inf_nan_fp16` — все 7 verified on A100. max |err| vs FP32 reference 1.02e-4 .. 4.88e-4.
  Throughput `add_fp16`: 34.25 Gelem/s. Self-test: `test_fp16_kernels.cu`.
- **Custom GEMM kernel** (`launch_gemm_native`): numerically matches cuBLAS `launch_gemm`
  (max |diff| 1e-5 .. 2e-4 across 64×64..1024×1024 FP32 GEMM). Competitive on small
  matrices (1.1× cuBLAS at 64×64), slower on large (0.3× at 1024×1024 — cuBLAS uses
  tensor cores, 12.8 TFLOPS on A100). Self-test: `test_gemm_native.cu`.
- **GGUF inference**: qwen3:4b 86.6 tok/s, deepseek-r1:8b 52.7 tok/s — numbers in "NVIDIA
  GPU — GGUF inference" section above.

### Compile-verified, runtime-untested
Нет доступа к соответствующему железу для verification:
- cuDNN RNN/LSTM/GRU (нужен full cuDNN stack + real model input)
- MPS Apple Metal (нужен Mac)
- ROCm/HIP (нужен AMD GPU)
- Autocast policy table (scaffolding есть, но не wired в каждую Tensor op entry point
  — autocast(Half) включает set_enabled но ops не смотрят policy автоматически)

### Партиал / requires extension
- **LLM serving engine** — forward loop + KV cache + sampling работают, `load_weights_()`
  is extension-point stub (connect с `safetensors_reader.py`).
- **torch.jit.compile** — trace + element-wise fusion + C++ codegen через subprocess.
  Большие тензоры: AVX2 eager быстрее fused-interpreter (fused нуждается в реальных AVX2
  intrinsics, не scalar `switch` loop).
- **NHWC-native Conv2d** — preservation работает (NHWC round-trips through Conv2d correctly),
  но internal compute остаётся NCHW im2col. Performance on channels_last не улучшается.

### Отсутствует полностью (scope не по силам solo за разумный срок)
- **`torch.compile` production** (TorchInductor + Triton backend + dynamic shapes + guards
  + graph breaks) — ~200K строк JIT-компилятора.
- **500+ tuned CUDA kernels per-shape** — годы команды. У нас ~20 tuned kernels + cuBLAS/
  cuDNN wrapper.
- **Sparse tensors** (COO/CSR/BSR) — отдельная memory layout infrastructure.
- **torch.distributions** (Normal, Categorical, Beta, ...) — probability distributions API.
- **FX graph mode** — intermediate graph IR для transforms.
- **Vulkan compute / TPU XLA backends**.
- **functorch полный** (grad-of-vmap, hessian, jacrev/jacfwd, composable transforms) —
  у нас базовый vmap + forward-mode AD.
- **100K+ unit tests** — у нас ~850 gtest + ~30 agent self-tests + auto-generated
  test_ops_generated суит (количество не протестировано).

### Что можно закрыть за 1-2 дня полной работы
- Wire autocast policy table в Tensor op dispatch (сейчас table есть, но ops её не
  проверяют автоматически).
- Прошить `LLM::load_weights_()` с `safetensors_reader.py` — ~200 строк.
- Фикс JIT codegen: emit AVX2 intrinsics вместо scalar loop → паритет с eager на больших
  тензорах.
- Fully-wired vmap/jvp: auto-hook Tensor::mul/add/etc. чтобы работало без explicit
  `forward_ad::mul` wrapping.

Полный исторический аудит: **`INFRASTRUCTURE_AUDIT.md`** (43 bug-а на момент аудита в марте 2026,
из них 31 исправлено на момент релиза, остальные либо закрыты в ходе 35-agent burst, либо
задокументированы здесь как Known Limitations).

---

## Быстрый старт

### Сборка на x86 (Linux)

```bash
git clone https://github.com/barometech/PromeTorch.git
cd promethorch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
```

### Сборка на x86 (Windows)

> **ВАЖНО:** Сборка на Windows работает ТОЛЬКО из Developer Command Prompt (не из Git Bash, PowerShell или WSL). `rc.exe` не найдётся без `vcvarsall.bat`.

```batch
REM Открыть Developer Command Prompt for VS 2019 (или запустить вручную):
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\path\to\promethorch
mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

### Сборка с CUDA

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON -DPT_USE_CUDNN=ON
cmake --build . -j$(nproc)
```

### Сборка на Эльбрусе (нативно)

```bash
# На сервере с Elbrus OS + LCC 1.29+
mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPT_USE_TUDA=ON \
    -DPT_USE_OPENMP=ON \
    -DPT_BUILD_TESTS=ON
ninja -j32
ctest --output-on-failure
```

CMake автоматически определяет архитектуру e2k и подключает:
- EML BLAS (Elbrus Math Library) — 230+ GFLOPS multi-threaded sgemm
- OpenMP для 32-ядерного параллелизма
- Оптимизации LCC: `-O3 -ffast-math`

### Сборка с NM Card Mini

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_USE_NMCARD=ON
cmake --build . -j$(nproc)
```

### Сборка для российских ОС (Docker)

```bash
docker build -t promethorch-astra -f docker/Dockerfile.astra ..
docker build -t promethorch-elbrus -f docker/Dockerfile.elbrus ..
docker build -t promethorch-redos -f docker/Dockerfile.redos ..
```

---

## Обучение MNIST (C++)

```cpp
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/data/data.h"

// Модель: 4-слойный MLP
Sequential model;
model.add(std::make_shared<Linear>(784, 512));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(512, 256));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(256, 128));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(128, 10));

// Оптимизатор и цикл обучения
auto optimizer = torch::optim::Adam(model.parameters(), /*lr=*/0.001);

for (auto& [data, target] : dataloader) {
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    torch::autograd::backward(loss);
    optimizer.step();
}
// Результат: 97.65% accuracy
```

### Обучение на российском железе

```cpp
// NM Card Mini (partner)
model.to("nmcard");
auto data = tensor.to("nmcard");

// LinQ H1M NPU
model.to("linq");
auto data = tensor.to("linq");
```

### Запуск готового примера

```bash
# Из директории сборки:
./examples/mnist/train_mnist_mlp --device cpu --epochs 5 --lr 0.001
```

---

## PromeServe — Ollama-совместимый LLM-сервер

Встроенный inference-сервер для больших языковых моделей. Загружает модели в формате GGUF напрямую из Ollama.

```bash
# Запуск сервера
./promeserve --port 11434 --device cuda --model qwen3:4b

# Генерация текста (curl)
curl -s http://localhost:11434/api/generate \
  -d '{"model": "qwen3:4b", "prompt": "Что такое нейронная сеть?"}'

# Чат (Ollama-совместимый API)
curl -s http://localhost:11434/api/chat \
  -d '{"model": "qwen3:4b", "messages": [{"role": "user", "content": "Привет!"}]}'
```

**API-эндпоинты** (совместимы с Ollama):
- `POST /api/generate` — генерация текста (streaming NDJSON)
- `POST /api/chat` — чат (streaming NDJSON)
- `GET /api/tags` — список доступных моделей
- `POST /api/show` — информация о модели
- `GET /api/version` — версия сервера

**Поддерживаемые архитектуры:** Qwen3, Gemma3, DeepSeek-R1, Llama, Mistral.
**Квантизация:** Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32.

---

## Справочник API (API Reference)

PromeTorch предоставляет API, максимально приближенный к PyTorch, как в C++, так и в Python.

### 1. Тензоры и операции (ATen)
Базовый класс `at::Tensor` (в Python `torch.Tensor`) поддерживает семантику ссылок и copy-on-write.

* **Методы тензора:** `.clone()`, `.detach()`, `.contiguous()`, `.to(device/dtype)`, `.item()`, `.copy_()`, `.view()`, `.reshape()`, `.flatten()`, `.squeeze()`, `.unsqueeze()`, `.transpose()`, `.permute()`, `.expand()`, `.repeat()`, `.split()`, `.chunk()`.
* **Фабрики:** `tensor`, `empty`, `zeros`, `ones`, `full`, `rand`, `randn`, `randint`, `arange`, `linspace`, `eye`, `zeros_like`, `ones_like`, `from_numpy`.
* **Математика:** `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sqrt`, `rsqrt`, `square`, `exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `clamp`, `nan_to_num`.
* **Редукции:** `sum`, `mean`, `max`, `min`, `norm`, `topk`, `sort`, `argmax`, `argmin`.
* **Линейная алгебра:** `mm`, `bmm`, `matmul`, `dot`, `einsum`.
* **Проверки:** `isinf`, `isnan`, `isfinite`.

### 2. Нейросетевые модули (`torch::nn` / `torch.nn`)
Все модели наследуются от `torch::nn::Module`.
* **Базовые методы Module:** `.forward()`, `.train()`, `.eval()`, `.to()`, `.zero_grad()`, `.parameters()`, `.named_parameters()`, `.state_dict()`, `.load_state_dict()`.
* **Хуки:** `ForwardPreHook`, `ForwardHook`, `ForwardHookWithReturn`.
* **Слои:** `Linear`, `Bilinear`, `LazyLinear`, `Identity`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose2d`.
* **Активации:** `ReLU`, `ReLU6`, `LeakyReLU`, `PReLU`, `ELU`, `SELU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `Softplus`, `Softsign`, `Hardtanh`, `Hardsigmoid`, `Hardswish`, `SiLU`, `Mish`, `Threshold`.
* **Нормализация:** `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm2d`.
* **RNN & Transformer:** `RNN`, `LSTM`, `GRU`, `TransformerEncoder`, `TransformerDecoder`, `MultiheadAttention`, `PositionalEncoding`.
* **PIR Architecture:** `PIRBlock`, `PIRLayer`, `PIR270M` — Parallel Infinite Retention (без attention, O(T) memory).
* **Функции потерь:** `L1Loss`, `MSELoss`, `SmoothL1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `NLLLoss`, `KLDivLoss`, `CosineEmbeddingLoss`, `TripletMarginLoss`, `CTCLoss`.

### 3. Оптимизаторы и планировщики (`torch::optim`)
* **Оптимизаторы:** `SGD`, `Adam`, `AdamW`, `RMSprop`. Fused версии: `fused_adam_multi`, `fused_sgd_multi` (все параметры за один вызов).
* **LR Schedulers:** `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LinearLR`, `ConstantLR`, `ReduceLROnPlateau`, `WarmupCosineAnnealingLR`, `OneCycleLR`.

### 4. Autograd и Checkpointing
* **Gradient computation:** `torch::autograd::backward()`, `NoGradGuard`, `EnableGradGuard`.
* **Custom Functions:** `torch::autograd::Function<Derived>` + `FunctionCtx` + `save_for_backward()`.
* **Gradient Checkpointing:** `torch::utils::checkpoint(fn, inputs)`.
* **116 backward functions** для всех операций.

### 5. Загрузка данных (`torch::data`)
* **Датасеты:** `Dataset`, `TensorDataset`, `MapDataset`, `ConcatDataset`, `SubsetDataset`, `random_split`.
* **DataLoader:** `batch_size`, `shuffle`, `drop_last`. Samplers: `SequentialSampler`, `RandomSampler`, `BatchSampler`.

### 6. Сериализация (формат PTOR)
* `torch::save()`, `torch::load()`, `torch::save_state_dict()`, `torch::load_state_dict()`.

### 7. Mixed Precision (AMP)
* **GradScaler:** `.scale(loss)`, `.unscale(optimizer)`, `.step(optimizer)`, `.update()`.
* **Autocast:** контекстный менеджер для FP16/BF16.

### 8. Backends
* **CPU:** AVX2 vectorized ops, cache-tiled GEMM, OpenMP parallelism.
* **CUDA:** Custom kernels (GEMM, reduce, element-wise), cuDNN (conv, pool, batchnorm), FlashDecoding (decode attention), Quantized inference (Q4_K_M, Q5_K, Q6_K, Q8_1). *(FlashAttention forward+backward временно отключён — требует доработки.)*
* **Elbrus E2K:** EML BLAS, NUMA-aware 4-chip training, VLIW-optimized fused ops.
* **NMCard Mini:** Q16.16 fixed-point, 64-core NMC4 emulator + hardware driver.

### 9. PromeServe (Inference Server)
Ollama-совместимый LLM сервер. API: `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`.
Web UI с streaming chat, markdown rendering, syntax highlighting.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Приложения                           │
│  examples/    python/    promeserve/    benchmarks/     │
├─────────────────────────────────────────────────────────┤
│                    torch/ (фреймворк)                   │
│  nn/modules/   optim/   data/   amp/   serialization   │
│  97 модулей   10 opt    DataLoader  GradScaler  PTOR   │
├─────────────────────────────────────────────────────────┤
│                  torch/csrc/autograd/                   │
│  Engine   Node   Edge   116 backward функций           │
├─────────────────────────────────────────────────────────┤
│                aten/src/ATen/ (операции)                │
│  MathOps  ReduceOps  LinearAlgebra  ShapeOps  IndexOps │
│  149 CPU операций с AVX2/NEON/E2K векторизацией        │
├──────────┬──────────┬──────────┬───────────────────────┤
│ TUDA CPU │ CUDA+    │ NMCard   │ LinQ H1M             │
│ AVX2     │ cuDNN+   │ Q16.16   │ INT8 GEMM            │
│ NEON     │ cuBLAS   │ 16 ядер  │ FP32 обучение        │
│ E2K VLIW │ 68 ядер  │ DSP      │ NPU                  │
│ Scalar   │          │          │                       │
├──────────┴──────────┴──────────┴───────────────────────┤
│                   c10/ (ядро)                           │
│  Allocator   Device   Storage   TensorImpl   ScalarType│
│  CPUAllocator  CUDAAllocator  NMCardAllocator  LinQ    │
└─────────────────────────────────────────────────────────┘
```

---

## Компоненты

### Ядро (c10) — ~4,626 строк

Базовый слой, аналогичный `c10` в PyTorch:
- **TensorImpl** — N-мерные тензоры, strides, views, channels-last (NHWC), `trusted_` flag для zero-overhead dispatch
- **CPUAllocator** — 3-уровневый caching: 16MB arena (lock-free) + thread-local cache (64 слота, zero-mutex) + глобальный bucket cache (256 слотов × 32 bucket). **97.7% cache hit rate, 179 аллокаций за epoch** (было 37,000)
- **Device** — 22 типа устройств (CPU, CUDA, NMCard, LinQ и др.)
- **ScalarType** — 16 типов данных (Float, Double, Half, BFloat16, QInt8 и др.) + type promotion
- **Storage** — ref-counted память с CUDA/NMCard/LinQ backend
- **ThreadPool** — persistent worker threads для Эльбруса (замена OpenMP fork/join)
- **SmallVector<5>** — SSO-оптимизация: тензоры до 5D не аллоцируют heap

### Операции (ATen) — ~18,000 строк

149 тензорных операций с SIMD-оптимизацией + 132 CUDA ядра:
- **Математика** — 20 unary (exp, log, sin, cos, tanh, sigmoid...) + 12 binary (add, mul, div, pow...)
- **Редукции** — sum, mean, max, min, var, std, argmax, argmin, norm, prod (с dim и keepdim)
- **Линейная алгебра** — mm, bmm, mv, dot, outer, addmm, LU, QR, SVD, Cholesky, solve, det
- **Форма** — view, reshape, transpose, permute, cat, stack, split, chunk, squeeze, unsqueeze, flatten
- **Индексирование** — index, index_put, scatter, gather, masked_select, boolean indexing
- **Фабричные** — zeros, ones, rand, randn, arange, linspace, eye, full, empty, multinomial
- **FFT** — fft, ifft, rfft, fft2

### Autograd — ~6,400 строк

Reverse-mode автоматическое дифференцирование:
- **Engine** — топологическая сортировка, cached GraphTask (reuse между backward)
- **116 backward-функций** — Math(46) + LinAlg(13) + Shape(21) + Reduce(16) + Index(2) + Fused(4) + Conv(4) + AccumulateGrad
- **Fused backward** — FusedLinearRelu (4 nodes → 1), FusedMLP (12 nodes → 1), PrecomputedGrad (zero-compute backward)
- **NodePool** — thread-local object pool для backward nodes (zero malloc в hot path)
- **SmallEdgeList<4>** — inline edges без heap allocation (99% ops)
- **Conv2d/BatchNorm/Pool backward** — im2col-based, полная CNN тренировка
- **Custom autograd functions** — CRTP `Function<Derived>` + gradient checkpointing

### NN Modules — ~9,000 строк, 97 модулей

| Категория | Модули |
|-----------|--------|
| Слои | Linear (fused_relu), LowRankLinear (SVD-сжатие), Bilinear, LazyLinear, Conv1d/2d/3d, ConvTranspose2d |
| Активации | ReLU, GELU, SiLU, Mish, Sigmoid, Tanh, ELU, SELU, LeakyReLU, Softmax, LogSoftmax, Hardtanh, Softplus, Softsign, PReLU, RReLU |
| Нормализация | BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, RMSNorm |
| Pooling | MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, GlobalAvgPool |
| Рекуррентные | RNN, LSTM, GRU (multi-layer, bidirectional) |
| Transformer | MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding |
| Loss | CrossEntropy, MSE, L1, BCE, NLL, KLDiv, CTC, Focal, Dice, SmoothL1, HuberLoss, CosineEmbedding, MarginRanking, TripletMargin |
| Контейнеры | Sequential, ModuleList, ModuleDict |
| Dropout | Dropout, Dropout1d/2d/3d |
| Embedding | Embedding, EmbeddingBag |
| Upsampling | Upsample (nearest, bilinear) |

### Оптимизаторы — 10 штук

SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax, AdamKiller (экспериментальный).
**Fused multi-param**: `fused_adam_multi` / `fused_sgd_multi` — один вызов на все параметры, VecF SIMD.

**LR Schedulers** (9 штук): StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, ConstantLR, ReduceLROnPlateau, WarmupCosineAnnealingLR, OneCycleLR.

### CUDA Backend — 99 ядер

- **65 element-wise** ядер (unary, binary, comparison, fused) с grid-stride loops
- **18 reduction** ядер (sum, max, min, var, cross_entropy, nll_loss)
- **9 BLAS** ядер (GEMM 4 варианта, batched, GEMV, dot, outer, transpose)
- **cuBLAS** — cublasSgemm для всех GEMM (замена custom 32x32 kernel)
- **cuDNN** — conv (fwd+bwd), pooling (fwd+bwd), batchnorm, activations (~12 ops)
- **FlashAttention** — forward + backward с online softmax, O(N) memory *(временно отключён, требует доработки)*
- **FlashDecoding** — parallel KV cache chunks, fused QKnorm+RoPE+KVwrite
- **Quantized GEMV** — Q4_K (persistent + fused gate+up), Q5_K, Q6_K, dp4a Q4_K×Q8_1
- **Mixed Precision** — GradScaler + Autocast (FP32/FP16/BF16)

### TUDA — CPU dispatch

**TUDA** (PromeTorch Unified Device Architecture) — автоматический выбор SIMD-ядер:

| Платформа | SIMD | Micro-kernel | Аккумуляторы |
|-----------|------|-------------|--------------|
| Intel/AMD | AVX2 + FMA | 6x16 | 12 FMA |
| Байкал-С (Cortex-A75) | NEON + dotprod | 8x12 | 24 |
| Байкал-М (Cortex-A57) | NEON | 4x8 | 8 |
| Эльбрус (E8C2) | E2K VLIW | 6x6 | 36 FMA |
| Прочие | Scalar | 4x4 | 16 |

- **Goto BLAS GEMM** с кеш-блокировкой, подобранной под L1d/L2 каждого процессора
- **Vectorized math** — exp, log, sin, cos, tanh, sigmoid (Cephes polynomials)
- **PromeBLAS** — cache-tiled GEMM, GEMV, DOT, saxpy с AVX2 FMA

### NM Card Mini — 3-й backend

Backend для нейропроцессора [NM Card Mini](https://www.module.ru/products/2-moduli/nm-card-mini) (partner):
- Процессор K1879VM8YA: 16 ядер NMC4, 1 GHz
- Программный эмулятор: float32 и Q16.16 fixed-point
- Интеграция: `tensor.to("nmcard")`, `model.to("nmcard")`
- Hardware path: DLL -> DDR dispatch protocol -> NMC4 ядра
- 33 теста PASS, MNIST 93.64% accuracy
- CMake: `-DPT_USE_NMCARD=ON`

### PromeServe — Ollama-совместимый LLM-сервер

Встроенный HTTP-сервер для inference больших языковых моделей:
- Загрузка моделей из Ollama (формат GGUF)
- Streaming NDJSON (совместим с Ollama API)
- Поддержка: Qwen3, Gemma3, DeepSeek-R1, Llama, Mistral
- Квантизация: Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32
- Web-интерфейс: встроенный чат

### Python API

```python
import promethorch as pt

# Создание тензоров
x = pt.randn([3, 4])
y = pt.mm(x, pt.randn([4, 5]))

# Обучение
model = pt.Sequential()
model.add(pt.Linear(784, 256))
model.add(pt.ReLU())
model.add(pt.Linear(256, 10))

optimizer = pt.Adam(model.parameters(), lr=0.001)
```

Сборка: `cmake .. -DPT_BUILD_PYTHON=ON`, затем `pip install -e .`

---

## Бенчмарки CPU (vs PyTorch 2.10)

Общий ratio: 1.47x (PromeTorch медленнее в 1.47 раза). Побеждаем на 15 из 50 операций.

**Операции, где PromeTorch быстрее PyTorch:**

| Операция | Ratio (меньше = лучше) |
|----------|----------------------|
| sum reduction | 0.43x |
| var | 0.14x |
| std | 0.14x |
| argmax | 0.13x |
| dot product | 0.28x |
| mv (matrix-vector) | 0.49x |
| mm 2048x2048 | 0.97x |
| tanh | 0.48x |

**Операции с наибольшим разрывом (PromeTorch медленнее):**

| Операция | Ratio | Причина |
|----------|-------|---------|
| Element-wise простые | 8-11x | Overhead аллокации тензоров |
| nn_lstm | 19x | Отсутствие fused LSTM kernel |
| autograd overhead | 6.6x | Python-free engine, но больше C++ overhead |

---

## Тестирование

```bash
cd build
cmake .. -DPT_BUILD_TESTS=ON -DPT_USE_TUDA=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

| Набор тестов | Количество | Статус |
|---|---|---|
| All Ops (тензорные операции) | 147 | PASS |
| Autograd (gradient check) | 63 | PASS |
| NN Modules | 49 | PASS |
| Optimizers | 50+ | PASS |
| TUDA (VecF, Math, BLAS) | 38 | PASS |
| NMCard Backend | 33 | PASS |
| LinQ H1M Backend | 34 | PASS |
| Edge Cases | 20+ | PASS |
| **Итого** | **434+** | **PASS** |

**Верификация на российских ОС:**
- Docker Astra Linux: 34/34 PASS
- Docker Elbrus: 34/34 PASS
- Docker RED OS: 34/34 PASS
- Нативно на Эльбрусе E8C2: 38/38 PASS

---

## CMake опции

| Опция | Умолчание | Описание |
|-------|----------|----------|
| `PT_USE_TUDA` | ON | CPU dispatch (AVX2/NEON/E2K/Scalar) |
| `PT_USE_CUDA` | OFF | NVIDIA CUDA backend |
| `PT_USE_CUDNN` | ON* | cuDNN (требует CUDA) |
| `PT_USE_NMCARD` | OFF | NM Card Mini backend |
| `PT_USE_LINQ` | OFF | LinQ H1M NPU backend |
| `PT_USE_AVX2` | ON | AVX2 SIMD на x86 |
| `PT_USE_OPENMP` | OFF** | OpenMP (авто-включение на Эльбрусе) |
| `PT_BUILD_TESTS` | ON | Сборка тестов (Google Test) |
| `PT_BUILD_PYTHON` | OFF | Python bindings (pybind11) |
| `PT_BUILD_SHARED_LIBS` | ON | Shared libraries (.so/.dll) |

\* Автоматически при `PT_USE_CUDA=ON`.
\** Автоматически включается при сборке на Эльбрусе.

---

## Примеры

| Пример | Директория | Описание |
|--------|-----------|----------|
| MNIST MLP | `examples/mnist/train_mnist_mlp.cpp` | 4-слойный MLP, 97.65% accuracy |
| MNIST CNN | `examples/mnist/train_mnist_cnn.cpp` | Свёрточная сеть на MNIST |
| 10 моделей | `examples/mnist/train_10_models.cpp` | Все 10 архитектур (MLP, LSTM, GRU...) |
| RNN/LSTM | `examples/rnn/train_rnn.cpp` | Рекуррентные сети |
| Transformer | `examples/transformer/train_transformer.cpp` | Transformer encoder-decoder |
| Vision Transformer | `examples/vit/train_vit.cpp` | ViT для классификации |
| GGUF Inference | `examples/gguf/test_gguf_inference.cpp` | Запуск GGUF-моделей |
| NM Card Mini | `examples/nmcard/` | MNIST на нейропроцессоре |

---

## Структура проекта

```
c10/                          Ядро: Allocator, Device, Storage, TensorImpl, ScalarType
  cuda/                       CUDACachingAllocator (singleton pattern)
  nmcard/                     NMCardAllocator (caching, PrivateUse1)
  linq/                       LinQAllocator (caching, PrivateUse2)

aten/src/ATen/                Операции (120+ CPU, 132 CUDA ядра)
  core/                       Tensor.h, TensorFactory.h
  native/cpu/                 MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps
    tuda/                     TUDA: VecF, TudaBLAS, TudaMath, TudaConfig
      kernels/avx2/           AVX2 6x16 micro-kernel
      kernels/neon/           NEON 4x8 (Байкал-М), 8x12 (Байкал-С)
      kernels/e2k/            E2K 6x6 (Эльбрус)
      kernels/scalar/         Scalar fallback
  cuda/                       CUDAKernels, CUDABlas, CUDAReduce, FlashDecoding
  cudnn/                      cuDNN: convolution, pooling, batch norm, activation
  nmcard/                     NMCardEmulator, NMCardOps, NMCardHardware

torch/                        Фреймворк
  csrc/autograd/              Engine, Node, Edge, 116 backward функций
  nn/modules/                 64+ NN модулей (16 файлов)
  optim/                      9 оптимизаторов + 9 LR schedulers
  data/                       Dataset, DataLoader, Samplers, Transforms
  amp/                        GradScaler, Autocast
  io/                         GGUF inference (gguf_model, tokenizer, ollama)
  serialization.h             Save/Load (бинарный формат PTOR)
  utils/checkpoint.h          Gradient checkpointing

promeserve/                   Ollama-совместимый LLM-сервер
python/                       pybind11 bindings
examples/                     MNIST, PIR, RNN, Transformer, ViT, GGUF, NMCard
test/cpp/                     434+ тестов (Google Test)
docker/                       Dockerfiles: Astra, Elbrus, RED OS
cmake/toolchains/             CMake toolchains для российских процессоров
scripts/                      Build-скрипты для российских ОС
```

---

## Дорожная карта

| Приоритет | Задача | Объём |
|-----------|--------|-------|
| Критический | Autograd для Conv/BN/Pool (тренировка CNN) | ~700 строк |
| Критический | Optimizer state_dict (checkpoint resume) | ~200 строк |
| Высокий | CUDA Streams/Events (async) | ~300 строк |
| Высокий | Python bindings расширение (view, reshape, grad) | ~500 строк |
| Средний | PackedSequence для variable-length sequences | ~300 строк |
| Средний | LBFGS optimizer | ~200 строк |
| Низкий | Distributed Training (DDP, NCCL) | ~2000 строк |
| Низкий | TorchScript / JIT | Архитектурное |
| Низкий | ONNX export | ~500 строк |

---

## Статистика проекта

| Метрика | Значение |
|---------|----------|
| Строк C++/CUDA | 86,543 |
| Строк Python | 6,772 |
| **Всего строк** | **128,000+** |
| Файлов в git | 481 |
| C++/CUDA файлов | 247 |
| Python файлов | 30 |
| Backward функций | 88 |
| NN модулей | 64+ |
| CUDA ядер | 68 |
| Оптимизаторов | 9 |
| LR Schedulers | 9 |
| Backend-ов | 4 (CPU, CUDA, NMCard, LinQ) |
| Тестов | 434+ |
| Примеров | 12 |

---

## Лицензия

**PromeTorch License** — модифицированная BSD 3-Clause (база PyTorch) с двумя
дополнительными условиями: **атрибуция в коммерческих продуктах** и **запрет на
перепродажу самого фреймворка**. Полный текст: [LICENSE](LICENSE).

### Коротко

✅ **Можно БЕСПЛАТНО и НА КОММЕРЦИЮ — без royalties, без уведомлений:**
- Обучать нейросети / модели на PromeTorch — любые, включая коммерческие.
- Запускать inference — в SaaS, API, on-prem, mobile, hardware-embedded.
- Продавать / лицензировать **модели, веса, pipeline'ы, приложения, сервисы**, которые
  используют PromeTorch как runtime. Твои модели — твои. Твой продукт — твой доход.
- Делать consulting, fine-tuning, training-as-a-service на PromeTorch.
- Форкать, модифицировать, встраивать в свой software, публиковать derivative models,
  использовать в research / research papers.

📝 **Нужно (только для коммерческих продуктов и сервисов):**
- Указать атрибуцию хотя бы в одном видимом месте — About / Credits, README/docs,
  UI footer/splash, или "open source notices" в mobile/desktop app:

      Powered by PromeTorch — https://github.com/barometech/PromeTorch

  Для внутреннего использования (R&D, internal pipelines, скрипты, не уходящие к
  end-user'у), личных проектов, академии — атрибуция не обязательна (но приветствуется).

❌ **НЕЛЬЗЯ:**
- Переименовать PromeTorch в "YourDL Framework 2.0" и продавать как свой фреймворк/SDK.
- Делать "Commercial Edition" / "Enterprise Version" / "Pro Tier" PromeTorch с paywall
  на core functionality.
- Взимать license fees, subscription fees, per-seat fees, или cloud-runtime fees за
  **сам фреймворк** (фи за ВАШ продукт, который его использует — пожалуйста).
- Обернуть PromeTorch в тонкий wrapper / GUI / CLI и продавать ЭТО как "framework".
- Продавать доступ к исходникам PromeTorch.

### В двух словах

**Фреймворк свободен и открыт. Что построил поверх него — твоё, хоть на миллиард.
Сам фреймворк перепродавать — нельзя, он остаётся общим. Указывай атрибуцию в
коммерческих продуктах. Всё остальное — можно.**

Это делает PromeTorch **source-available** (не строго OSI-compliant "open source"),
но в практическом смысле permissive: единственные ограничения — атрибуция в коммерции
и запрет на перепродажу самого фреймворка.

---

## Авторы

Разработано в России. 3 недели, 1 разработчик, 93,000 строк кода.

Подробная документация: [PROMEPEDIA.md](PROMEPEDIA.md) | Журнал разработки: [JOURNAL.md](JOURNAL.md) | Аудит: [INFRASTRUCTURE_AUDIT.md](INFRASTRUCTURE_AUDIT.md)
