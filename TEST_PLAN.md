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
| 5 | `examples/gguf/test_gguf_inference.exe` | `qwen3:4b --device cuda --max_tokens 100 "Once upon a time"` | ≥ 80 tok/s, coherent output | 🟢 (86.6 tok/s) |
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
| 13 | `test_gguf_inference.exe` (direct CUDA) | qwen3:4b Q4_K_M | A100 40GB | ≥ 150 tok/s | 86.6 tok/s | 🟡 |
| 14 | `promeserve.exe /api/generate` (HTTP) | qwen3:4b | A100 | ≥ 140 tok/s (HTTP overhead) | 89.8 tok/s | 🟡 |
| 15 | `test_gguf_inference.exe` | gemma3:4b | A100 | ≥ 90 tok/s | 52.9 tok/s | 🟡 |
| 16 | `test_gguf_inference.exe` | deepseek-r1:8b | A100 | ≥ 60 tok/s | 30.5 tok/s | 🟡 |
| 17 | `test_gguf_inference.exe` | qwen3:14b | A100 | ≥ 35 tok/s | 18.4 tok/s | 🟡 |
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

**Option B — One-shot fp16 unpack at load (RECOMMENDED):**
- At model load, dequant ALL weights from Q4_K → FP16. VRAM cost goes from
  ~5 GB to ~8 GB for qwen3:4b — fits A100 40GB easily; consumer 24 GB cards
  would lose qwen3:14b but those are not the target.
- Use `cublasHgemv` (or `cublasSgemmEx` with FP16 inputs / FP32 accum) for
  every matmul. Throw away the custom kernel entirely.
- Pro: 50-100% expected speedup (matches Ollama path); simpler code.
- Con: 8 GB instead of 5 GB VRAM; need to expose `--keep-quant` for
  consumer-card users.

**Decision (this plan):** Option B, gated behind `--fp16-weights` flag
(default ON when free VRAM > 1.5 × Q4_K size; auto-fallback otherwise).

### 4.3 Implementation steps
1. Add `dequant_q4k_to_fp16` CUDA kernel (one-shot at load).
2. Switch `cuda_quant_gemv` callsites (in `gguf_model.h`) to use the FP16
   tensor + `cublasHgemv` when `--fp16-weights`.
3. Add VRAM-pressure auto-detect: query `cudaMemGetInfo` after weight load
   stage, if FP16 unpack would exceed 90% utilization, fall back.
4. Re-enable CUDA Graph capture along the pure-cuBLAS path (the custom
   GEMV kernel doesn't capture cleanly; cuBLAS does).
5. Verify on A100 with 100-token decode.

### 4.4 Acceptance criteria
- qwen3:4b: ≥ 150 tok/s (target), ≥ 130 tok/s (acceptable).
- gemma3:4b: ≥ 100 tok/s.
- Coherence: same logits ± 0.5 cosine vs Q4_K kernel (no quality regression).
- Memory: VRAM ≤ 10 GB for qwen3:4b unpacked.

---

## §5 Structural API gaps remaining (identified by Sonnet audit agents)

These don't have a single binary to run — they are coverage gaps. Listed
in priority order.

| Priority | Gap | Where | Impact |
|----------|-----|-------|--------|
| 1 | `autocast` context not hooked into op dispatch | `torch/amp/autocast.h` + per-op cast wrapping in `aten/src/ATen/native/cpu/MathOps.h` | AMP appears to work but does no autocast in practice. Mixed-precision training silently runs in FP32. |
| 2 | `ParamGroup` incomplete (per-group lr/momentum/weight_decay) | `torch/optim/optimizer.h` | Layer-wise lr scheduling not possible — common for fine-tuning (e.g. discriminative LRs for BERT). |
| 3 | `DistributedDataParallel.no_sync()` missing | `torch/distributed/ddp.h` | Gradient accumulation across micro-batches in DDP forces a sync at every step — wastes AllReduce bandwidth. |
| 4 | `torch.no_grad()` Python binding doesn't propagate to C++ engine | `python/promethorch/__init__.py` (pybind11), `torch/csrc/autograd/grad_mode.h` | Inference Python loops still build autograd graph (wastes memory + time). BUG-C9 in audit. |
| 5 | FlashAttention has 6 known bugs, no callsites | `aten/src/ATen/cuda/FlashAttention.cu` | Documented as broken; all production attention falls back to `sdpa_forward_cpu_impl` (slow on CUDA). |
| 6 | Conv3d forward is a stub returning zeros | `torch/nn/modules/conv.h::Conv3dImpl::forward` | Video / 3D-medical models impossible. Lower priority (not in our 10-model set). |
| 7 | `torch.compile` not implemented | n/a (entire missing subsystem) | No graph capture / fusion. ~2-3× speedup left on table for training. Out of scope for this plan. |
| 8 | `create_graph=True` ignored (no double-backward) | `torch/csrc/autograd/engine.cpp` | Higher-order grads (meta-learning, MAML, gradient-penalty losses) impossible. |
| 9 | TransformerEncoderLayer CUDA forward crash | (suspected) `torch/nn/modules/transformer.h` + missing CUDA kernel for `LayerNorm` over the right axes | Blocks GPU training of `train_transformer.exe`; CPU works. |

---

## §6 Test infrastructure gaps

- Test:LOC ratio is 434:110K (≈ 1:250). Industry norm is 1:50–1:100.
  Proposal: backport integration tests from RUKALLAMA so end-to-end
  PIR / GGUF / autograd convergence is captured by CI.
- No CI runner configured for `build_cudnn` (currently manual).
- LeNet CPU Conv convergence bug needs a regression test before fixing.

---

## §7 Order of work for next sprint

1. **Wait for the 4 background agents** (ViT MNIST, ResNet-20 CIFAR, VAE
   MNIST, DCGAN MNIST) and fold their results into rows 6, 7, 10, 11 above.
2. **§4** — PromeServe FP16 dequant → cuBLAS HGEMV. Single biggest user-
   facing win.
3. **§5 row 1** — wire autocast into dispatch (small file count, big real
   impact for mixed-precision training claims).
4. **§5 row 4** — propagate `torch.no_grad()` from Python to engine.
5. **§5 rows 2-3** — ParamGroup + DDP `no_sync`.
6. **§5 row 9** — fix TransformerEncoderLayer CUDA forward (unblocks GPU
   transformer training).
7. **§6** — backport RUKALLAMA integration tests, set up CI.
8. **§5 row 5** — FlashAttention rewrite (already partially done — verify
   and re-enable callsites).
9. **§5 rows 6-8** — Conv3d, double-backward (lower priority, fill in as
   needed).

---

## §8 Already-completed in 2026-04-19 sprint

For reference / commit messages map to:
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
