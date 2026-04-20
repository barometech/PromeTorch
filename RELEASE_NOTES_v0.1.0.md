# PromeTorch v0.1.0 — First public release

**Дата:** 2026-04-20

Single-dev PyTorch-совместимый обучающий фреймворк с нуля на C++17/CUDA.
Нативная сборка на NVIDIA A100, Эльбрус E8C2 VLIW, NM Card Mini (НТЦ
Модуль). ~137 000 строк кода, 1 разработчик, ~5 недель активной разработки
+ 2 agent-burst'а.

📊 **Headline numbers:** [RESULTS.md](RESULTS.md)

---

## Что работает (runtime-verified на реальном железе)

### Core
- **Autograd engine** — 119 backward functions, BFS execution,
  gradient hooks, anomaly mode, `create_graph=True` (double backward),
  forward-mode AD (dual numbers), vmap (single-axis bit-exact).
- **16 optimizers** — SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta,
  Adamax, AdamKiller, ASGD, Lion, SophiaG, LAMB, Adafactor, NAdam,
  RAdam, LBFGS. Все на `at::*` tensor ops, CPU-portable.
- **16 LR schedulers** — Step, MultiStep, Exp, CosineAnnealing, Linear,
  Const, ReduceLROnPlateau, WarmupCosine, OneCycle,
  CosineAnnealingWarmRestarts, Cyclic, Polynomial, Lambda, Multiplicative,
  Sequential, Chained.
- **ParamGroup** — per-group lr / momentum / betas / eps / amsgrad с
  NaN-sentinel наследованием от defaults (discriminative LR для fine-tuning).
- **EMA, clip_grad_norm_, clip_grad_value_, gradient checkpointing.**

### NN modules (64+)
- Linear / Bilinear / LazyLinear / Identity
- Conv1d/2d/3d/ConvTranspose2d (Conv3d — real OpenMP-parallel direct
  convolution, не stub)
- 20 activations (ReLU family, GELU, SiLU, Mish, Hardswish, ...)
- BatchNorm1d/2d, LayerNorm, RMSNorm, GroupNorm, InstanceNorm2d
- RNN / LSTM / GRU (pure-C++ forward primary; cuDNN 8 optional)
- Transformer Encoder/Decoder/MultiheadAttention/PositionalEncoding
- 12 loss functions, включая полный CTCLoss (Graves DP)
- PIR architecture (diagonal selective scan)

### Distributed
- Real TCP DDP (star topology, POSIX sockets + Winsock shim)
- FSDP / ZeRO-3 (flat sharding via /dev/shm)
- TensorParallel (Col/Row), Pipeline Parallel (1F1B)
- DistributedSampler, DDP `.no_sync()` context manager

### Backends
- **CPU x86_64** — PromeBLAS 6×16 AVX2+FMA, vectorized exp/log/sin/cos/
  tanh/sigmoid, im2col conv. Production.
- **CUDA (NVIDIA)** — cuBLAS, custom Q4_K GEMV (HBM-saturation-optimal
  для N=1 decode), CUDA Graph capture, FP16 kernels (7 ops A100-verified).
- **Эльбрус 8C2** — E2K VLIW + EML_MT BLAS, 1840 GFLOPS (92% пика),
  MNIST MLP 6.1× быстрее PyTorch 2.7.1 на той же задаче.
- **NM Card Mini** — Q16.16 fixed-point эмулятор, 32/32 backend tests,
  MNIST 88.94% на 16 virtual cores.

### Export / interop
- ONNX (zero-dep protobuf, runs в ONNX Runtime, passes `onnx.checker`)
- MLIR (tosa + linalg)
- Mobile (PTMB, bit-exact round-trip)
- `torch.jit.compile` — trace + C++ codegen через subprocess
- **PyTorch `.pt` I/O** — pickle protocol 2 + ZIP. Bidirectional:
  мы читаем `torch.save`, PyTorch читает наши saves.

### Ecosystem shims
- HuggingFace Transformers compat (AutoModel, safetensors, pytorch_model.bin)
- torchvision (ImageFolder + 7 transforms + MobileNetV2 + ResNet-20)
- torchaudio (STFT/iSTFT + MFCC + WAV)
- torchtext (Vocab + BPE + WordPiece)
- Lightning Trainer

### LLM serving — PromeServe
- Ollama-compatible HTTP API
- GGUF Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16 / F32
- Flash-decode + CUDA Graph + paged KV (64-token pages)
- Qwen3, Gemma3, DeepSeek-R1, Llama, Mistral
- Production guardrails: thread pool + bounded queue (503 Retry-After),
  per-request timeout, CORS

---

## Benchmarks в этом релизе

### A100 GGUF inference (5-run median, greedy)
| Модель | PromeTorch | Ollama | Ratio |
|--------|-----------:|-------:|------:|
| qwen3:4b | **82.6 tok/s** | 164.7 | 50% |
| gemma3:4b | **81.4 tok/s** | 145.4 | 56% |
| deepseek-r1:8b | **51.1 tok/s** | 127.8 | 40% |

- 10-min stress: 46.5 ± 0.19 tok/s stable at T=0.7
- Concurrent training (PIR 13.78 GB) does not disturb inference

### A100 training
- **PIR 33.5M params** loss 4.0 → 1.20 over 2000 iters (89 min)

### CPU training (vs PyTorch 2.6.0 на identical arch/data)
- MNIST MLP 784-128-10: **92.69%** vs 92.61% (+0.08pp)
- MNIST + Dropout: **97.25%** vs 97.15% (+0.10pp)
- LSTM seq cls: **98.44%** vs 96.88% (+1.56pp)
- Wide MNIST + save/load round-trip: **97.65%** vs 97.36% (+0.29pp)
- VAE 50 epochs test ELBO: **101.8 nats** vs 102.15 (0.35 nats tighter)

### Эльбрус 8C2 (исторический, 2026-04-16)
- MNIST MLP: **2.76 s vs PyTorch 2.7.1 16.8 s = 6.1× faster**
- NUMA-aware EML_MT: **1840 GFLOPS (92% пика)**

### NM Card Mini эмулятор
- MNIST MLP: **88.94%** test accuracy в 3 epochs (53 s)

### DCGAN MNIST
- 30 epochs CPU, stable equilibrium, **readable digits** at epoch 30
  (0, 2, 3, 7, 8, 9 visible, no mode collapse)

Подробности: [RESULTS.md](RESULTS.md), 7 BENCH_*.md файлов в корне
репозитория.

---

## Известные ограничения

**Compile-verified только (нет железа для runtime verify):**
- MPS (Apple Metal) — нужен Mac
- ROCm/HIP — нужен AMD GPU

**Partial:**
- **Autocast** — policy table + `to_autograd` + ToBackward foundation
  готовы, wired в Linear и Conv2d forwards. MHA wiring + FP16 mm через
  cuBLAS dispatch — отдельная задача.
- **Sampling path overhead на A100** — greedy 82 tok/s, sampling T=0.7
  46 tok/s (1.84× slowdown). Per-token CPU-GPU sync на random draws +
  unfused top-k/top-p extractor. Fix: pinned-memory async draws.
- **TransformerEncoderLayer на CUDA** — forward не крэшится (CPU-bounce
  fallback, commit `81b08ba`), но training медленный. CPU — production
  (85.8% val acc verified).
- **ResNet-20 CIFAR-10 на CUDA** — step 0 completes, step 1 backward
  silently dies. Documented blocker (autograd engine / CUDA dispatch
  deeper issue). CPU training ~16 ч/epoch непрактично.
- **MultiheadAttention** bypass'ит autograd в custom batched matmul.
  ViT обходит через mean-pool workaround.
- **FlashAttention** — headers есть, 0 callsites, 6 known bugs.

**Отсутствует:**
- `torch.compile` production (TorchInductor + Triton)
- Sparse tensors (COO/CSR/BSR)
- FX graph mode
- torch.distributions
- functorch composable transforms (только базовый vmap + forward AD)
- Vulkan / TPU XLA backends

Полный честный аудит: [TEST_PLAN.md](TEST_PLAN.md),
[INFRASTRUCTURE_AUDIT.md](INFRASTRUCTURE_AUDIT.md).

---

## Как собрать

```bash
git clone https://github.com/barometech/PromeTorch.git
cd PromeTorch
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DPT_USE_TUDA=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure  # 720+ tests
```

CUDA: добавьте `-DPT_USE_CUDA=ON`. Windows details в
[docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md) (MSVC + NVIDIA Toolkit
12.4 workaround).

---

## Лицензия

**PromeTorch License** — модифицированная BSD-3 + 2 добавки:
1. Атрибуция `Powered by PromeTorch — github.com/barometech/PromeTorch`
   в коммерческих продуктах (не нужна для research / internal / academic).
2. Запрет на перепродажу самого фреймворка как продукта (rebrand /
   Commercial Edition / paywall-core).

Всё остальное свободно. Models, weights, pipelines, apps, SaaS — твои.
Полный текст: [LICENSE](LICENSE).

---

## Что дальше

Roadmap в [TEST_PLAN.md §7](TEST_PLAN.md). Приоритеты:
1. **Continuous batching** для Ollama-parity (Tensor Cores любят GEMM)
2. **Sampling path fix** (82 → 46 regression устранить)
3. **ResNet CIFAR A100 step-1 crash** — deeper autograd/CUDA dispatch debug
4. **MHA autograd** — rewire через `*_autograd` ops, снять CLS-mean-pool workaround
5. **FlashAttention rewrite** (6 known bugs, 0 callsites)

---

## Благодарности

Разработано в России. Single developer. C++17 + CUDA + pybind11.
