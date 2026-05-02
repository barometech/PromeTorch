# PromeTorch — Russian training framework

🌐 **Languages:** [Русский](README.md) · **English (this file)**

[![Release](https://img.shields.io/github/v/release/barometech/PromeTorch?color=blue&label=release)](https://github.com/barometech/PromeTorch/releases)
[![License](https://img.shields.io/badge/license-PromeTorch%20%28BSD--3%20%2B%20attribution%20%2B%20no--resale%29-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B)](CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2F12.8-76B900.svg?logo=nvidia)](docs/BUILD_WINDOWS.md)
[![Эльбрус](https://img.shields.io/badge/Эльбрус-E8C2%20VLIW-red.svg)](docs/elbrus/)
[![NM Card](https://img.shields.io/badge/NTC%20Module-NM%20Card%20Mini-blueviolet.svg)](BENCH_NMCARD.md)
[![Tests](https://img.shields.io/badge/tests-720%2B-green.svg)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Astra%20%7C%20ALT%20%7C%20RED%20%7C%20Elbrus-informational.svg)](docker/)

> **Single-dev PyTorch rewrite.** Native Эльбрус E8C2 VLIW + NM Card Mini +
> NVIDIA A100. Real autograd (119 backward ops), 16 optimizers, ONNX export,
> PyTorch-compatible `.pt` I/O. ~35-45% PyTorch practical surface. ~137K LOC.
>
> 📊 **[RESULTS.md](RESULTS.md) — single-page canonical benchmarks.**
> On A100: qwen3:4b @ **82 tok/s** greedy inference (50% of Ollama).
> On CPU: MNIST matches PyTorch ±0.5 pp, VAE **0.35 nats tighter**.
> On Эльбрус 8C2: **6.1× faster** than PyTorch 2.7.1 on MNIST MLP.
> On NM Card Mini emulator: MNIST 88.94%.

> PyTorch-compatible training framework in C++17/CUDA with broad dtype support.
> Real autograd (119 backward ops + gradient hooks + anomaly mode + create_graph
> double-bwd), **16 optimizers** (SGD/Adam/AdamW/RMSprop + Lion/Sophia/LAMB/
> Adafactor/NAdam/RAdam/Adagrad/Adadelta/Adamax/AdamKiller/ASGD/LBFGS),
> **16 LR schedulers**, CPU SIMD + CUDA (cuBLAS/cuDNN 8/FP16 kernels),
> distributed training (TCP DDP/FSDP/TP/Pipeline + `no_sync()`),
> export (ONNX/MLIR/Mobile/JIT), ecosystem shims (HuggingFace/torchvision/
> torchaudio/torchtext/Lightning Trainer), **PyTorch-compatible save/load
> (`.pt` ↔ `torch.load`/`torch.save`)**.
>
> **~132,000 lines of C++/CUDA** (114K framework core + 17.8K examples) +
> **~4,700 Python** = ~137K LOC. 1 developer. ~5 weeks of active
> development + two agent-bursts (35 + 15). 720+ tests.

> ⚠️ **Coverage ~35-45% of PyTorch's practical surface.** This is a solo project:
> some paths are runtime-verified on actual hardware (CPU x86 / Эльбрус / A100
> GGUF inference); others are **compile-verified only** (MPS/ROCm — no Mac/AMD GPU,
> cuDNN-accelerated RNN legacy API guarded for cuDNN 9). `torch.compile` is a
> trace-based prototype, not full TorchInductor. Sparse tensors / FX graph mode /
> torch.distributions are absent. See **Known Limitations** below — every gap is
> honestly documented.

---

## Coverage vs PyTorch

| Category | PyTorch | PromeTorch | % |
|---|---|---|---|
| Tensor ops | ~2000 | ~150 | ~7% |
| Backward functions | ~1500 | 119 + hooks + anomaly | ~8% |
| Optimizers | 15+ | **16** | ~100% |
| LR schedulers | 15+ | 16 | ~100% |
| dtypes | 20+ | 10 (Float32/64, Half, BFloat16, **Float8 e4m3fn/e5m2**, Complex64/128, Bool, int8-64) | ~50% |
| Autograd features | full | core + hooks + anomaly + create_graph + forward-AD + vmap | ~40% |
| Distributed | NCCL/gloo/ucc | real TCP DDP, FSDP/ZeRO-3, TP Col/Row, Pipeline 1F1B | ~35% |
| Compile/JIT | TorchInductor + Triton | trace-based `torch.jit.compile` + C++ codegen | ~10% |
| Export | ONNX + TorchScript + ExecuTorch + TensorRT | ONNX (ORT works) + MLIR + Mobile + JIT | ~50% |
| Backends | CPU / CUDA / ROCm / MPS / XLA / Vulkan / TPU | CPU + CUDA + MPS (compile-only) + NM Card + Эльбрус | ~45% |
| Quantization | INT8/INT4/NF4/FP8/QAT/PTQ | INT8 QAT + INT4 + NF4 + fp8 dtype | ~60% |
| Sparse tensors | COO/CSR/BSR | none | 0% |
| torch.distributions | 50+ | none | 0% |
| FX graph mode | full | none | 0% |
| Unit tests | 100K+ | 720+ gtest + agent self-tests | ~1% |

**Aggregate estimate: ~35-45% of PyTorch user-surface** — sufficient for
training/deploy of transformers/CNN/RNN/LLM on CPU + CUDA + Эльбрус + NM Card.
Gap in **tuned kernels** / **torch.compile production** / **sparse + distributions
+ FX** would require multi-year team effort to close.

---

## Why PromeTorch

**1. Russian accelerators.** The only training framework we know of with native
build for **Эльбрус E8C2** (E2K VLIW, LCC 1.29, EML_MT BLAS) and **NM Card Mini**
(Q16.16 fixed-point emulator, MNIST 93.64%). Cross-compilation to Baikal-M/S
ready out of the box.

**2. MNIST MLP faster than PyTorch on Эльбрус.** On MNIST MLP-4
(784→512→256→128→10, SGD, batch=64, 1 epoch) — 2.76 s vs PyTorch 2.7.1 16.8 s
(**6.1× on this narrow task**). 1840 GFLOPS via node-local EML_MT (92% of
E8C2 2 TFLOPS peak). On other tasks (general / real transformers) PyTorch's
advantage holds.

**3. Universal build.** One codebase — CPU (AVX2+FMA/NEON/E2K), CUDA (Turing+),
NM Card (emulator + real card in inference mode). Builds on Windows MSVC, Linux
GCC, Astra/ALT/RED/Elbrus OS. Autograd engine works identically on all backends.

---

## Results

### Эльбрус E8C2

МЦСТ Эльбрус server — 4× Elbrus-MCST E8C2 (VLIW), 32 cores, 1500 MHz.

**LLM inference — qwen3:4b Q4_K_M (2026-05-02):**

| Config | Cores | tok/s | vs A100 PromeTorch (82.6) | Lossless |
|--------|------:|------:|--------------------------:|:--------:|
| PromeTorch TP-4 + LayerSkip 12 alt (decode-only, opt-in lossy) | 32/32 | **15.5** | ×5.3 | ✗ |
| PromeTorch TP-4 + LayerSkip 6 alt (decode-only, opt-in lossy) | 32/32 | 13.2 | ×6.3 | ✗ |
| **PromeTorch TP-4 + Q8 SoA4 + AVX2/e2k attn + fused SiLU+Q8 + triple QKV** | **32/32** | **11.4** ★ | **×7.2** | ✓ |
| PromeTorch TP-4 + fused gate+up Q8 SoA4 GEMV | 32/32 | 10.8 | ×7.6 | ✓ |
| PromeTorch TP-4 + Q8 SoA4 + persistent ThreadPool + 8t/rank | 32/32 | 10.6 | ×7.8 | ✓ |
| PromeTorch TP-4 + Q8 SoA4 + persistent ThreadPool (7t/rank, NUMA replicate) | 28/32 | 9.9 | ×8.3 | ✓ |
| PromeTorch TP-4 + Q8 SoA4 (qpmaddubsh) — `PT_Q8_SOA=1` | 28/32 | 9.4 | ×8.8 | ✓ |
| PromeTorch 1-proc, 24t + interleave + Q4_K/Q6_K block prefetch | 24/32 | 5.2 | ×16 | ✓ |
| llama.cpp pure-C 32t (no SIMD, no EML) | 32/32 | 3.3 | ×25 | ✓ |

**Lossless ceiling 11.4 tok/s** confirmed from three independent angles:
disassembly of `q8_soa4_gemv` (peak 6 ops/cycle on 6-wide VLIW), batched
GEMM K=4 microbench (0.42× regression — compute-bound proven), busy-spin
ThreadPool probe (×2 regression — NUMA coherency thrashing). Full technical
report: [report/REPORT_ELBRUS_LLM_INFERENCE_2026-05-02.pdf](report/REPORT_ELBRUS_LLM_INFERENCE_2026-05-02.pdf).

**Run (lossless 11.4):**
```bash
PT_Q8_SOA=1 ./scripts/run_tp_elbrus.sh --greedy "Hello"
```

**Run (lossy 15.5, output text degrades):**
```bash
PT_Q8_SOA=1 PT_LAYER_SKIP="12,14,16,18,20,22,24,26,28,30,32,34" \
    ./scripts/run_tp_elbrus.sh --greedy "Hello"
```

**IMPORTANT:** do not set `PT_PIN_THREADS=1` in TP mode — ThreadPool tries to
pin workers of ranks 1-3 to CPUs outside their cpuset (set by `numactl
--cpunodebind`). Kernel clamps them to a single CPU and tok/s drops to 1.4.

**MNIST MLP training (1 epoch, lr=0.01, batch=64):**

| Metric | PromeTorch | PromeTorch + NUMA | PyTorch 2.7.1 |
|---|---|---|---|
| **Time** | **15.2 s** | **2.76 s** | 16.8 s |
| **Accuracy** | **88.71%** | **88.94%** | 88.14% |
| **Ratio** | **1.1× faster** | **6.1× faster** | 1.0× |
| EML GFLOPS | 330 | **1840 (92% peak)** | 68 (generic BLAS) |
| Allocations | 179 | 179 | ~50,000+ |

Optimization path (126.3 s → 15.2 s = 8.3× speedup):

| Step | Time | vs PyTorch | Key change |
|---|---|---|---|
| Scalar baseline | 126.3 s | 7.4× slower | First build |
| + EML BLAS | 120.6 s | 7.1× | cblas_sgemm, 230 GFLOPS |
| + Memory pool | 121.4 s | 7.1× | 97.7% cache hit, 641 mallocs |
| + Fused ops | 97.3 s | 5.7× | 8 agents: fused ops, thread pool, 6×6 kernel |
| + SIMD SGD | 45.4 s | 2.7× | pow→x*x, skip contiguous |
| + Direct EML | 43.7 s | 2.6× | Direct cblas calls, zero-copy backward |
| + Manual backward | 22.0 s | 1.3× | Bypass autograd, pre-allocated buffers |
| **Final** | **15.2 s** | **0.90×** | Removed unused grad clipping |

### NVIDIA GPU — GGUF inference

GGUF model inference (Q4_K_M quantization) via custom INT4 warp-cooperative GEMV.

**Runtime-verified on A100 40GB (2026-04-20):**

| Model | PromeTorch (greedy) | PromeTorch (sampling T=0.7) | Ollama | vs Ollama |
|---|---|---|---|---|
| **qwen3:4b** Q4_K_M | **82.6 tok/s** | 46.5 tok/s | 164.7 tok/s | **50%** (greedy) |
| **gemma3:4b** | **81.4 tok/s** | — | 145.4 tok/s | **56%** |
| **deepseek-r1:8b** Q4_K_M | **51.1 tok/s** | — | 127.8 tok/s | **40%** |

**Live numbers**, all 5-run median, 200 tokens. Full matrix in
[BENCH_A100_HEAVY.md](BENCH_A100_HEAVY.md) + [BENCH_OLLAMA.md](BENCH_OLLAMA.md).

- **10-min stress test** (qwen3:4b, 9,500 tokens): 46.52 ± 0.19 tok/s stable,
  peak 25.4 GB VRAM / 135 W, no thermal throttle, no crashes.
- **Concurrent training** on the same A100 (PIR 13.78 GB) is unnoticeable in
  inference (std 0.4% of mean).
- qwen3:4b VRAM: **8.0 GB** / 40 GB (quant-only mode).
- Model weights move to CUDA: **0.1 s** (previously 88 s — fixed quant-only
  transfer).
- FP16 KV cache: 306 MB for qwen3:4b 36 layers, full decode graph captured.

### NM Card Mini — emulator

NTC Module NM Card Mini — fixed-point Q16.16 ML accelerator (programmable
via emulator).

| Task | Score | Notes |
|---|---|---|
| MNIST 3 epochs | **93.64%** | Within ±0.3 pp of FP32 PyTorch baseline |
| GEMM 1024×1024 | 50× over scalar | 32/32 tests pass |
| Conv2d 3×3×128 | 27× | im2col + Q16.16 mac chain |

Full benchmarks: [BENCH_NMCARD.md](BENCH_NMCARD.md).

---

## Build

### Linux (Ubuntu / Debian / RHEL):
```bash
git clone https://github.com/barometech/PromeTorch
cd PromeTorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DPT_USE_CUDA=ON   # or OFF for CPU-only
cmake --build build -j
```

### Windows (MSVC 2019+ Build Tools, Anaconda CUDA):
```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set CUDA_PATH=C:\ProgramData\anaconda3\Library
cmake -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
    -DPT_USE_CUDA=ON -DPT_USE_CUDNN=ON -DPT_BUILD_TESTS=OFF ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
cmake --build build -j
```

### Docker (Astra Linux / ALT Linux / RED OS / Elbrus OS):
```bash
docker build -f docker/Dockerfile.astra -t prometorch:astra .
docker build -f docker/Dockerfile.alt   -t prometorch:alt .
docker build -f docker/Dockerfile.red   -t prometorch:red .
```

---

## License

PromeTorch BSD-3 + attribution + no-resale. See [LICENSE](LICENSE).

**Permitted (free):** train and sell models, sell apps using PromeTorch, fork,
internal corporate use, education.

**Required:** attribution in commercial products that ship PromeTorch.

**Forbidden:** reselling the framework itself as a product (rebranding,
SDK wrapper, paywall around the framework). The framework stays common.

This makes PromeTorch **source-available** (not strictly OSI-compliant
"open source") but practically permissive: the only restrictions are
commercial attribution and no framework-resale.

---

## Authors

Developed in Russia. ~5 weeks of active development, 1 developer,
~137,000 lines of code (132K C++/CUDA + 4.7K Python).

Detailed docs: [PROMEPEDIA.md](PROMEPEDIA.md) · Dev journal:
[JOURNAL.md](JOURNAL.md) · Audit: [INFRASTRUCTURE_AUDIT.md](INFRASTRUCTURE_AUDIT.md)
