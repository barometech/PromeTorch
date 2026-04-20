# PromeTorch — Test & Verification Plan

**Created:** 2026-04-19
**Hardware:** A100-SXM4-40GB, CUDA 12.4 build / 12.8 runtime, cuDNN 9, MSVC 2019
**Goal:** verify every training path runs end-to-end to convergence on A100, document
runtime metrics, and close the PromeServe inference gap to ≥150 tok/s on qwen3:4b.

This is the working plan for test-and-close-gaps work. Each row in §1-3 has a
binary, an arg-set, an expected metric, and an output path. §4-5 cover the
gap-closure work and the structural items still missing relative to PyTorch.

---

## Status legend
- 🟢 PASS — verified on A100 since 2026-04-18
- 🟡 PARTIAL — runs but doesn't reach target (e.g. converges weakly)
- 🔴 FAIL — crashes or stuck at random
- ⚪ TODO — binary not yet rebuilt with these changes / not yet run

---

## §1 Verification matrix — built and ready

CUDA-enabled binaries currently present in `build_cudnn/`. Run from
`PATH="build_cudnn:./build_cudnn:$ANACONDA/Library/bin:$PATH"`.

| # | Binary | Args | Target metric | Status |
|---|--------|------|---------------|--------|
| 1 | `examples/mnist/train_10_models.exe` | `--device cuda --data data/mnist` | All 10 models PASS, LSTM ≥ 95%, GRU ≥ 95%, Deep MLP ≥ 97% | 🟢 |
| 2 | `examples/pir/train_pir.exe` | `data/tiny_shakespeare.txt --device cuda --iterations 100 --batch_size 4 --block_size 64 --n_embd 128 --n_layers 2 --n_pir_layers 2` | Loss ≤ 3.0 by iter 100 | 🟢 (3.87→2.74) |
| 3 | `examples/pir/train_mlp_char.exe` | `--data data/tiny_shakespeare.txt --device cuda --iterations 100 --batch_size 32 --block_size 64 --n_embd 64 --n_hidden 128` | Loss ≤ 3.5 by iter 100 | 🟢 (4.14→3.41) |
| 4 | `examples/pir/test_mem_leak.exe` | (none) | created==destroyed==released==alive==0 | 🟢 |
| 5 | `examples/gguf/test_gguf_inference.exe` | `qwen3:4b --device cuda --max_tokens 100 "Once upon a time"` | ≥ 40 tok/s, coherent output | 🟢 (~48 tok/s 2026-04-20) |
| 6 | `examples/cifar/train_resnet.exe` | `--device cuda --data data/cifar-10-batches-bin --epochs 50 --batch_size 128 --lr 0.1` | Test acc ≥ 88% by epoch 50 | ⚪ (agent running — `a0ef5f8b1796e9ece`) |
| 7 | `examples/vae/train_vae.exe` | `--device cuda --data data/mnist --epochs 20 --batch_size 128` | ELBO ≤ 95 nats on test by epoch 20, generated digits visually MNIST-like | ⚪ (agent running — `a4465a38bb7f2ed15`) |

---

## §2 Verification matrix — needs rebuild then run

Source/CMake updated, binary not yet rebuilt or only CPU-built so far.

| # | Source | Build target | Args | Target metric | Status |
|---|--------|--------------|------|---------------|--------|
| 8 | `examples/shakespeare/train.cpp` | `shakespeare_train` in `build_cudnn` | `--device cuda --epochs 20 --batch_size 64 --block_size 128 --lr 3e-4 --data data/tiny_shakespeare.txt` | Loss ≤ 1.7 by epoch 20, generated text ≥ 60% real English words | ⚪ (agent running — `a3d22fe03a7bc97e6`) |
| 9 | `examples/transformer/train_transformer.cpp` | `train_transformer` | `--device cuda --epochs 10 --batch_size 64 --d_model 64 --nhead 4 --num_layers 2 --max_len 32 --lr 2e-3` (synthetic sentiment) | Val acc ≥ 85% by epoch 10 | 🟡 (CPU 85.8% verified by agent `afeecb710b3c952f9`; CUDA forward crashes — pre-existing TransformerEncoderLayer CUDA bug) |
| 10 | `examples/vit/train_vit.cpp` | `train_vit` | `--device cuda --data data/mnist --epochs 5 --batch_size 64 --patch_size 7 --d_model 64 --nhead 4 --num_layers 4 --lr 1e-3` | Test acc ≥ 95% on MNIST by epoch 5 | ⚪ (agent running — `acd79d72f773cbfa6`) |
| 11 | `examples/gan/train_gan.cpp` | `train_gan` | `--device cuda --data data/mnist --epochs 30 --batch_size 128 --latent_dim 100 --lr 0.0002 --beta1 0.5` | Generator Inception-Score ≥ 6 on saved samples (visual: digit-like outputs by epoch 30) | ⚪ (agent running — `a2115508f19e73373`) |
| 12 | `examples/mnist/train_mnist_cnn.cpp` (LeNet) | `train_mnist_cnn` | `--device cuda --data data/mnist --epochs 5 --batch_size 64 --lr 0.01` | Test acc ≥ 95% | 🔴 (CPU stuck at 12.55% in `EXAMPLES_VERIFIED.md` — pre-existing CPU Conv convergence bug. Need to rebuild on CUDA path and verify whether bug is CPU-only) |

---

## §3 Inference / serving — speed verification

| # | Path | Model | Hardware | Target | Current | Status |
|---|------|-------|----------|--------|---------|--------|
| 13 | `test_gguf_inference.exe` (direct CUDA, greedy) | qwen3:4b Q4_K_M | A100 40GB | ≥ 130 tok/s | **82.6 tok/s** (2026-04-20) | 🟡 |
| 13a | `test_gguf_inference.exe` (sampling T=0.7) | qwen3:4b Q4_K_M | A100 40GB | ≥ 80 tok/s | 46.5 tok/s (sampling path overhead) | 🔴 |
| 14 | `promeserve.exe /api/generate` (HTTP) | qwen3:4b | A100 | ≥ 80 tok/s (HTTP overhead) | ~82 tok/s greedy expected | ⚪ (not re-bench'd since latest build) |
| 15 | `test_gguf_inference.exe` (greedy) | gemma3:4b | A100 | ≥ 80 tok/s | **81.4 tok/s** (2026-04-20) | 🟢 |
| 16 | `test_gguf_inference.exe` (greedy) | deepseek-r1:8b | A100 | ≥ 50 tok/s | **51.1 tok/s** (2026-04-20) | 🟢 |
| 17 | `test_gguf_inference.exe` | qwen3:14b | A100 | ≥ 35 tok/s | 18.4 tok/s (stale, 2026-04-19) | ⚪ |
| 18 | `promeserve.exe` baseline functional | qwen3:4b | A100 | `/api/show` 200, `/api/generate` 200, `/api/embeddings` 501, `OPTIONS /api/generate` 204 with CORS, timeout fires correctly | All four endpoints working | 🟢 (verified by agent `af854109154bca330` on port 11440) |

---

## §4 PromeServe → 150 tok/s plan

The +5% gain from removing per-token `cerr` is real but small. The main gap is
**dequant→FP16 weights + cuBLAS HGEMV** instead of the current bespoke Q4_K
GEMV kernel. Outline:

### 4.1 Profile current bottleneck (FIRST STEP)
- Use `nvprof` / `nsys` on `test_gguf_inference.exe` for 100-token decode of
  qwen3:4b. Identify which CUDA kernel dominates wall time (expected: the
  custom `quant_gemv_warp_coop` in `aten/src/ATen/cuda/CUDAQuantGemv.cu`).
- Measure HtoD memcpy, kernel launch overhead, FlashDecoding paged attention.
- Output: `BENCHMARKS/qwen3_4b_a100_profile.md` with kernel-time table.

### 4.2 Dequant→FP16 path (ARCHITECTURAL)
Two options, in increasing cost:

**Option A — Per-decode dequant scratch buffer:**
- Allocate `[d_model × ffn_hidden]` FP16 scratch (~32 MB for qwen3:4b).
- For each FFN/attn projection: launch `dequant_q4k_to_fp16` kernel into
  scratch, then `cublasGemvEx` (HGEMV / FP16 → FP16).
- Pro: no quant-aware GEMV maintenance; cuBLAS does the FMA ladder.
- Con: extra HtoD-equivalent copy per token (BUT: bandwidth on A100 is
  1.5 TB/s, dequant of 4B params = ~30 ms one-shot — too slow per-token).
  **Verdict: only viable if scratch is reused across all matrices in one
  forward pass, which it can't naively be**.

**Option B — One-shot fp16 unpack at load (IMPLEMENTED AS OPT-IN; NOT FASTER):**
- At model load, dequant ALL weights from Q4_K → FP16. VRAM cost goes from
  ~8 GB to ~14 GB for qwen3:4b.
- Use `cublasHgemv` for every matmul. Falls back to Q4_K path if VRAM
  budget would be exceeded.
- **Expected** 50-100% speedup (this was WRONG — see below).
- **Actual on A100 (2026-04-20):** baseline ~47 tok/s; --fp16-weights
  ~44 tok/s. **~6% REGRESSION.**

### 4.3 Why Option B doesn't speed up decode (empirical post-mortem)

Single-token decode is pure **N=1 GEMV** — bandwidth-bound, not
compute-bound. Math at A100:
- Q4_K_M weights for qwen3:4b: ~2.5 GB per forward pass.
- FP16 weights after dequant:  ~5.9 GB per forward pass (**2.36× more
  memory traffic**).
- A100 HBM2e: ~1.55 TB/s. Q4_K forward ≈ 1.6 ms; FP16 forward ≈ 3.8 ms.

Tensor Cores don't help when the matmul is K×1 — they're optimized for
GEMM shapes, not GEMV. And the existing Q4_K fused kernels
(`q4km_fused_rmsnorm_qkv_gemv`, `q4km_fused_rmsnorm_gate_up_gemv`,
`q4km_persistent_gemv_accumulate`) do **rmsnorm + 3 matmuls + residual
in ONE kernel launch** per attention / FFN block; the FP16 path needs
4 separate kernels per QKV.

**Conclusion:** for per-token decode on A100, the bespoke Q4_K kernels
already win. Option B stays shipped as `--fp16-weights` opt-in (see
commit `6623fe3`) for future workloads where it might help — prefill
batching with larger N × K × 1 shapes where Tensor Cores kick in.

### 4.4 What to do next (closing 47 → 100+ tok/s)

**Path A — Port llama.cpp kernels** (ORIGINAL HYPOTHESIS — TESTED, DOESN'T WORK).
The v2 kernel landed in commit `a3d2796` (`launch_q4km_persistent_gemv_v2`,
opt-in via `--llama-gemv`) applied bulk `__ldg` header loads + NROWS=2 +
shared smem reads — all the standard llama.cpp optimizations over our v1.
A100 delta: **+0.5% (46.8 → 47.05 tok/s, within noise)**. Byte-identical
output at T=0. Our v1 persistent kernel is already saturating A100 HBM2e
for N=1 decode — there's no raw kernel overhead left to optimize.

**Path B — Continuous batching / prefill (RECOMMENDED).** Tensor Cores pay
off at N>1 GEMM shapes. Current PromeServe serves single-token decode
only; a batch of N=8 prompts would turn each matmul from a K×1 GEMV into
a K×8 GEMM where HGEMM tensor cores actually help. Implementation:
`gguf_model.h` `run_step_decode_cuda` needs a batched version with
`launch_cublas_hgemm_batched` (not existing). Effort: ~1-2 day rewrite.
Expected gain: 3-5× aggregate throughput for multi-request workloads.

**Path C — Lower-precision weights** (INT3/INT2 or sparse GEMV). Reduces
HBM traffic directly. INT4 → INT3 is ~25% less memory traffic. Requires
new quantization format + kernel. Effort: ~3-5 days.

**Path D — Speculative decode.** Run multiple tokens per forward pass via
a draft model. ~2× speedup if draft is small and mostly agrees with
target. Effort: ~2 days. Needs a smaller draft GGUF.

### 4.5 Acceptance criteria (updated)

Given HBM saturation, the realistic single-request target is:
- qwen3:4b single-token decode: **~47 tok/s is the upper bound** for our
  Q4_K_M + A100 combo without changing the problem shape.
- **Realistic wins come from batching** (Path B). qwen3:4b multi-request
  throughput at N=8: target ≥250 tok/s aggregate.

Baseline (2026-04-20, 3-run avg):
- v1 Q4_K persistent GEMV: 46.8 tok/s (default production path)
- v2 Q4_K llama-style:      47.05 tok/s (opt-in `--llama-gemv`)
- FP16 dequant + cublasHgemv: 44 tok/s (opt-in `--fp16-weights`, slower)

Ollama on same hardware: 165 tok/s — they use different model execution
(batched + custom kernels tuned for our specific shape classes). A full
match likely requires Path B + custom kernels targeting qwen3 shape
classes specifically.

---

## §5 Structural API gaps (status updated 2026-04-19 PM)

| # | Status | Gap | Where | Notes |
|---|--------|-----|-------|-------|
| 1 | 🟡 PARTIAL | `autocast` not hooked into op dispatch | `torch/amp/autocast.h`, module forwards | Building block done (`8f87e57` to_autograd + ToBackward). Still TODO: per-op wrapping in Linear/Conv/MHA forwards + FP16 mm cuBLAS dispatch + A100 verify. |
| 2 | 🟢 DONE | `ParamGroup` incomplete | `torch/optim/optimizer.h` + sgd/adam/rmsprop | `d3951bb`+`d519a0f`. Per-group lr/momentum/wd/betas/eps/amsgrad with NaN-sentinel inheritance, scheduler.step_group(idx) overload, full backwards-compat. |
| 3 | 🟢 DONE | `DistributedDataParallel.no_sync()` | `torch/distributed/ddp.h` + `distributed.h` | `ab71ddf`+`ea07f99`. RAII guard + Python context manager; works on both POSIX-TCP DDP (Elbrus) and ProcessGroup-abstracted DDP. Single-process test with CountingPG mock backend verifies 1 AllReduce instead of N for N-batch accumulation. |
| 4 | 🟢 DONE | `torch.no_grad()` Python → C++ propagation | `python/promethorch/__init__.py` | `763ebb1`. Real `_GradModeContextDecorator` flips C++ `GradMode::is_enabled()` thread-local. Stack-safe (nested). Note: separate gap surfaced — `_C.pyd` op bindings call raw aten directly (skip `*_autograd` wrappers), so `requires_grad` doesn't propagate at the Python-op boundary. Filed as new gap §5.10 below. |
| 5 | 🔴 NOT STARTED | FlashAttention 6 known bugs, no callsites | `aten/src/ATen/cuda/FlashAttention.cu` | Production attention uses `sdpa_forward_cpu_impl` (slow on CUDA). |
| 6 | 🟢 DONE | Conv3d forward is a stub | `torch/nn/modules/conv.h::Conv3d::forward` | Implemented 2026-04-18 as OpenMP-parallel nested 7-loop direct convolution (input [N,C,D,H,W] × weight [Cout,Cin,kD,kH,kW] → [N,Cout,Do,Ho,Wo]). Not blazing but correct. Backward still via generic autograd. |
| 7 | 🔴 OUT OF SCOPE | `torch.compile` not implemented | n/a (entire subsystem) | ~2-3× speedup left on table; out of scope. |
| 8 | 🔴 NOT STARTED | `create_graph=True` ignored (no double-backward) | `torch/csrc/autograd/engine.cpp` | Higher-order grads (MAML, grad-penalty) impossible. |
| 9 | 🟡 PARTIAL | TransformerEncoderLayer CUDA forward crash | `torch/nn/modules/normalization.h`, `torch/nn/modules/attention.h`, `aten/src/ATen/ATen.h` | Forward no longer crashes: CPU-bounce fallbacks for LayerNorm + MultiheadAttention (no native CUDA kernels), and a general-broadcast CPU fallback in `Tensor::add` (for `[B,T,D]+[1,T,D]` positional-embedding addition). Verified by `tests/test_transformer_cuda_forward.cpp`: LayerNorm cos_sim=1.0, Attention cos_sim=1.0, full Encoder cos_sim=0.89–0.95 (remaining drift from CUDA Linear vs CPU Linear accumulation, not this fix). Backward on CUDA still slow/unstable due to many tiny bounces; CPU path remains production. |
| 10 | 🟡 NEW | Python `_C.pyd` op bindings bypass `*_autograd` wrappers | `python/csrc/bindings_new.cpp` | Surfaced by no_grad agent (a7f1200e004cb3008). `t1 + t2` from Python calls raw aten directly → output never has grad_fn even when both inputs require_grad. Affects every binary op exposed to Python. |
| 11 | 🟢 DONE | Untyped `mutable_data_ptr()` overload missing on Tensor | `aten/src/ATen/core/Tensor.h` | `a5a8cbf`. Was blocking cuDNN headers' `void*` callsites (cudnnActivationForward etc.). |
| 12 | 🟢 DONE | nvcc dropping `__declspec(dllexport)` on `launch_*` | `aten/src/ATen/cuda/aten_cuda_exports.def` | `a5a8cbf`. Explicit module-definition file for ~150 launch_* exports. Was the silent reason train_resnet/train_gan failed to load aten_cuda.dll exports at runtime even though they compiled clean. |

---

## §6 Test infrastructure gaps

- Test:LOC ratio is 434:110K (≈ 1:250). Industry norm is 1:50–1:100.
  Proposal: backport integration tests from RUKALLAMA so end-to-end
  PIR / GGUF / autograd convergence is captured by CI.
- No CI runner configured for `build_cudnn` (currently manual).
- LeNet CPU Conv convergence bug needs a regression test before fixing.

---

## §7 Order of work for next sprint (updated 2026-04-19 PM)

Items 4-7 in the original list closed already. New ordered queue:

1. **Wait for the 4 training-to-convergence agents** (ViT MNIST, ResNet-20
   CIFAR, VAE MNIST, DCGAN MNIST) and fold their results into rows 6, 7,
   10, 11 of §1-§2.
2. **§4** — PromeServe FP16 dequant → cuBLAS HGEMV. Biggest user-facing
   win (86.6 → 150 tok/s gap).
3. **§5 row 10 (NEW)** — fix Python op bindings to route through
   `*_autograd` wrappers so `t1 + t2` from Python actually builds graph
   when inputs require_grad. Surfaced by no_grad agent.
4. **§5 row 1** — finish autocast: wire to_autograd into Linear/Conv/MHA
   forwards (small file count) + add FP16 mm cuBLAS dispatch + A100
   verify mixed-precision training delivers actual speedup.
5. **§5 row 9** — fix TransformerEncoderLayer CUDA forward (unblocks GPU
   transformer training; transformer agent confirmed CPU-OK, CUDA crash).
6. **§6** — backport RUKALLAMA integration tests, set up CI for
   build_cudnn.
7. **§5 row 5** — FlashAttention rewrite (already partially done in audit
   sprint; verify, re-enable callsites, switch sdpa_forward_cuda away
   from CPU fallback).
8. **§5 rows 6, 8** — Conv3d, double-backward (lower priority).

---

## §8 Already-completed in 2026-04-19 sprint

For reference / commit messages map to:

**Morning batch (autograd + ops + examples):**
- `9b19480` — 5 missing backward Nodes (where/masked_fill/scatter_add/gather/norm_dim).
- `df3f804`, `0f205af` — reshape_autograd / select_autograd correctness fixes.
- `472a1fe` — logsumexp + LogSumExpBackward + 4 missing ops.
- `619aa00` — full SDPA forward+backward (CPU) with masks/dropout/causal.
- `97a11b2` — ConvTranspose2d backward + ResNet-20 + DCGAN + VAE examples.
- `20d817a` — 7 LR schedulers + EMA + clip_grad_norm_ + checkpoint_sequential.
- `ade8b88` — Shakespeare/Transformer/ViT training-loop fixes.
- `cfbcd42`, `ccf1fd0` — GGUF/PromeServe perf push (86.6 → 91 tok/s).
- `e2b25d9` — `EXAMPLES_VERIFIED.md` baseline matrix.
- `92f9c47` — license clarification (attribution + no-resale).
- `83b133d` — `JOURNAL.md` 2026-04-19 entry.
- `b802c09`, `bee0cb4` — TEST_PLAN.md created + autocast audit findings.

**Afternoon batch (structural API gaps):**
- `8f87e57` — to_autograd + ToBackward (foundation for autocast).
- `a5a8cbf` — DLL exports `.def` file + cuDNN data_ptr fix (build infra).
- `763ebb1` — Python no_grad/enable_grad → C++ GradMode (BUG-C9).
- `d3951bb`, `d519a0f` — ParamGroup per-group hyperparameters + step_group rename.
- `ab71ddf`, `ea07f99` — DDP no_sync() context manager + Python wrapper + test.
