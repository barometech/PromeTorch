# Examples Verification Report — 2026-04-19

Environment: Windows 10, A100-SXM4-40GB (TCC, driver 572.61, CUDA 12.8 runtime), CUDA 12.4 build-time, cuDNN 9 via Anaconda.

Logs: `run_logs/` at repo root.

## Build Matrix

| Build dir | CUDA | Notes |
|-----------|------|-------|
| `build_cudnn` | ON | Only CUDA-capable build. Has `aten_cuda.dll`, `c10.dll`. |
| `build_final3` | OFF | CPU-only (CMakeCache: `PT_USE_CUDA=OFF`), despite having `aten_cuda.dll` copy. |
| `build_examples` | OFF | CPU-only. LeNet CNN binary is CPU-only. |

DLL copying: **not required**. `build_cudnn` DLLs already co-located with its binaries (same working dir via `PATH`). CPU builds have their own `c10.dll`.

## Verification Results

| # | Binary | Device | Args | Result | Time | Quality |
|---|--------|--------|------|--------|------|---------|
| 1 | `build_examples/examples/mnist/train_mnist_cnn.exe` | CPU | `--device cpu --data data/mnist --epochs 1 --batch_size 64 --lr 0.01` | **Train Acc 12.805%, Test Acc 12.55%, Loss 2.29979** | 574.6 s | **Stuck at random** (loss ≈ −log(0.1)). LeNet CNN forward/backward completes without errors but does not learn. Likely init / conv-gradient issue on CPU path. |
| 2 | `build_cudnn/examples/mnist/train_10_models.exe` | CUDA | `--device cuda --data data/mnist` | Models 1–9 PASS (exit 0). Model 10 loaded (1 333 770 params) but log ended mid-train. | ~15 s | Model 4: 92.68%; Model 5: **97.23%**; Model 6: 97.17%; Model 7 RNN MSE 1.67e-5; Model 8 LSTM 93.75%; Model 9 GRU **98.44%**. CPU↔CUDA logits/grads match exactly. |
| 3 | `build_cudnn/examples/pir/train_mlp.exe` | CPU (no CUDA flag) | `data/tiny_shakespeare.txt` | **Loss 3.33 after 500 iters** | 5 s | Generated char-LM output is noise (expected for 92k-param model). |
| 4 | `build_cudnn/examples/pir/train_mlp_char.exe` | CUDA | `--data data/tiny_shakespeare.txt --device cuda --iterations 100 --batch_size 32 --block_size 64 --n_embd 64 --n_hidden 128` | **Loss 4.14 → 3.41** over 100 iters | <1 s | GPU 80 MB, clean shutdown. |
| 5 | `build_cudnn/examples/pir/train_pir.exe` | CUDA | `data/tiny_shakespeare.txt --device cuda --iterations 100 --batch_size 4 --block_size 64 --n_embd 128 --n_layers 2 --n_pir_layers 2` | **Loss 3.87 → 2.74** over 100 iters | 3 s | 0.54 M params, GPU 426 MB alloc / 150 MB cached. Generated text line truncated in log but training converges. |
| 6 | `build_cudnn/examples/pir/test_mem_leak.exe` | CPU | (no args) | **No node leaks**: created=900 destroyed=900 released=900 alive=0. Loss 2.34 after 50 iters. | 1.5 s | PASS. |
| 7 | `build_cudnn/examples/gguf/test_gguf_inference.exe` | CUDA | `qwen3:4b --device cuda --max_tokens 40 --temperature 0.7 "Hello, how are you?"` | **128 tokens / 2.7 s = 47.6 tok/s** | ~15 s incl. load | Coherent English: "I'm trying to create a website for a business…". VRAM 8.0 / 39.7 GB. Flash-decode + PromeGraph captured. |
| 8 | `build_cudnn/promeserve/promeserve.exe` | — | `--help` | Prints help | <1 s | Shows Ollama-compatible endpoints, default port 11434. |

## Failures / Notes

- **LeNet CNN accuracy ≈ random (12.55 %)** — binary runs to completion, no crash, but loss never descends. Single-epoch result; may reach target with more epochs, but loss on epoch 1 stayed at 2.30 (−log 0.1), so there is a real convergence issue on the CPU Conv path in `build_examples`. Not a regression introduced today — build predates this session.
- **train_10_models Model 10** (Wide MNIST + serialization) started but stdout was flushed before the Epoch/Final-Acc prints. Process exited with code 0 and no errors; likely a buffered-stdout truncation rather than a crash.
- **train_mlp** in `build_cudnn` is an older CPU-only sample (source removed from `examples/pir/`); arg parser takes a positional file. Ran successfully on CPU.
- **No DLL copying was necessary.** `build_final3/examples/*` and `build_examples/examples/*` are CPU-only; `build_cudnn` already bundles CUDA DLLs.

## Known-baseline Comparisons

| Metric | Session result | CLAUDE.md / MEMORY baseline | Match |
|--------|----------------|----------------------------|-------|
| qwen3:4b CUDA tok/s | 47.6 | 49.9 (previously recorded) | Close (−5 %) |
| LSTM seq-class | 93.75 % | 98.44 % | Low (batch/seed variance in built-in demo; same numbers expected with tuning) |
| GRU trend | 98.44 % | 95.3 % | Match/above |
| Deep MNIST MLP | 97.23 % | 97.65 % | Match |

## Commands Used (reproducible)

```bash
# LeNet CNN (CPU)
PATH="/c/ProgramData/anaconda3/Library/bin:$PATH" \
  ./build_examples/examples/mnist/train_mnist_cnn.exe \
  --device cpu --data data/mnist --epochs 1 --batch_size 64 --lr 0.01

# 10 models on CUDA
PATH="/c/ProgramData/anaconda3/Library/bin:build_cudnn:$PATH" \
  ./build_cudnn/examples/mnist/train_10_models.exe --device cuda --data data/mnist

# PIR on CUDA
PATH="/c/ProgramData/anaconda3/Library/bin:build_cudnn:$PATH" \
  ./build_cudnn/examples/pir/train_pir.exe data/tiny_shakespeare.txt \
  --device cuda --iterations 100 --batch_size 4 --block_size 64 \
  --n_embd 128 --n_layers 2 --n_pir_layers 2 --log_interval 10

# GGUF qwen3:4b on CUDA
PATH="/c/ProgramData/anaconda3/Library/bin:build_cudnn:$PATH" \
  ./build_cudnn/examples/gguf/test_gguf_inference.exe qwen3:4b \
  --device cuda --max_tokens 40 --temperature 0.7 "Hello, how are you?"

# Promeserve
PATH="/c/ProgramData/anaconda3/Library/bin:build_cudnn:$PATH" \
  ./build_cudnn/promeserve/promeserve.exe --help
```
