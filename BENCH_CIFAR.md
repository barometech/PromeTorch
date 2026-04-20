# BENCH_CIFAR — ResNet-20 on CIFAR-10 (A100-SXM4-40GB)

**Date:** 2026-04-20
**Hardware:** NVIDIA A100-SXM4-40GB (TCC), driver 572.61, CUDA runtime 12.8, toolkit 12.4
**Build:** `build_cuda124/examples/cifar/train_resnet.exe` via `scripts/build-cuda-windows.bat train_resnet`
**Repo commit at run:** `d839a48` (source `584e531` CPU-bounce fallbacks already landed)

---

## Status: BLOCKED — silent crash in `autograd::backward` on step 2 (CUDA)

The 30-epoch convergence run could NOT be completed. Training dies reproducibly
inside `torch::autograd::backward(...)` on the **second** training step on
CUDA. No C++ exception is surfaced, no CUDA error string is printed, the
process simply exits.

This is a **new, pre-existing blocker** in the framework's autograd engine
(or one of its CUDA CPU-bounce fallbacks). It is NOT one of the issues that
commit `584e531` claims to have fixed; 584e531's bounces let step 0's forward
and backward and optimizer step run fine — the second backward is where it
falls over.

Per task instruction "If training crashes with a NEW bug (not one already
fixed in 584e531), STOP and report." → stopping here.

---

## Arch (unchanged, reference only)

- `torch::vision::models::resnet20` (He et al. 2015 CIFAR variant)
- 272,474 parameters (matches paper's 0.27M)
- stem 3×3 conv (3→16) + BN + ReLU
- 3 stages × 3 `CifarBasicBlock`s (channels 16/32/64, first block of stages 2,3 stride-2)
- `AdaptiveAvgPool2d(1)` → `Linear(64, 10)`

## Config (intended)

- optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=false
- scheduler: `MultiStepLR` ×0.1 at epochs 80, 120 (would not trigger in 30 ep)
- loss: `CrossEntropyLoss`
- batch: 128
- aug: random crop 32 from pad-4, horizontal flip (train only), per-channel mean/std normalize
- epochs: 30 (requested)

## What actually ran

Full transcript of the smoke run that reproduces the failure (also at
`run_logs/cifar_resnet20_a100.log`):

```
Device: CUDA
Loading CIFAR-10 from: data/cifar-10-batches-bin
  train: 50000 images, test: 10000 images
Model: ResNet-20  (272474 params)
Moving model to CUDA...
Model moved to CUDA

=== Training (1 epochs, bs=128, lr=0.1, momentum=0.9, wd=0.0005) ===
  [trn] step 0 start
  [trn] step 0 zg
  [trn] step 0 fwd
  [trn] step 0 loss
  [trn] step 0 bwd      ← step 0 backward OK
  [trn] step 0 opt
  [trn] step 0 pre-loss-cpu
  [trn] step 0 post-loss-cpu
  [trn] step 0 pre-preds
  [trn] step 0 post-preds
  [trn] step 1 start
  [trn] step 1 zg
  [trn] step 1 fwd      ← step 1 forward OK
  [trn] step 1 loss
                         ← process dies inside autograd::backward({loss})
```

No "step 1 bwd" line, no std::exception message propagates (the try/catch I
wrapped around the iteration body during diagnosis did not fire — confirming
the failure is either a hard abort() from deep in the engine, a CUDA sticky
error state that leaks through the CPU-bounce path and kills the process on
the next kernel launch, or a non-C++-exception termination).

## Curve

| epoch | train loss | train acc | test acc |
|-------|-----------:|----------:|---------:|
|   —   |    n/a     |   n/a     |   n/a    |

(No epochs completed. Training cannot advance past step 1.)

## Wall-clock

- Build (from scratch): ~4 min (NVIDIA CUDA 12.4 / MSVC 2019 NMake)
- Build outcome: **OK** (0 errors, routine `cuda_fp16.hpp` warnings only)
- Step 0 full fwd+bwd+opt+copy-out: ~seconds (negligible)
- Step 1: crashes inside backward before printing "bwd"
- Total training wall-clock: **0 epochs usable**

## Comparison vs PyTorch baseline

PyTorch ResNet-20 / CIFAR-10 reference numbers (for when this *does* train):
- no-augmentation: **~85–89%** test acc
- with-augmentation (pad-4 crop + hflip, which is what this loader does):
  **~91–92%** test acc

Our result: **n/a — crash at step 1 backward on CUDA**.

## Attempts / sanity checks tried

1. Rebuild `build_cuda124/examples/cifar/train_resnet.exe` against latest
   source (commit `d839a48`, includes 584e531's BN2d/pool CPU-bounce path).
   → build OK, same crash.
2. Colocate `c10.dll` + `aten_cuda.dll` next to the exe (exe was returning
   127 from bash when launched from repo root because DLL search order
   preferred root-dir copies; resolved).
3. Reduce batch size to 32 → same crash at same point.
4. Instrument training loop with per-phase `std::cout << flush` and wrap
   the iter body in `try { … } catch (std::exception&) { std::cerr << … }`
   — **no exception caught**, process just vanishes. Removed instrumentation
   before writing this report; the `[trn] step N xxx` prints already in
   `train_resnet.cpp` were sufficient to bracket the failure.
5. CPU path as fallback: runs correctly but each step takes ~2.4 min on this
   box (ResNet-20 forward+backward hitting the non-vectorized `Conv2dBackward`
   on 32 threads); 391 steps × 2.4 min ≈ 16 hours/epoch → not viable within
   the 90 min budget.

## Honest assessment

The framework can do **one** CUDA training step of ResNet-20 end-to-end, which
is a real improvement over the pre-584e531 state where BN2d backward and
pooling forward crashed immediately. But the second backward is a hard
blocker, and the silent-termination mode means the root cause requires either
(a) attaching a Windows debugger to the process, (b) a `cudaDeviceSynchronize`
+ `cudaGetLastError` sweep after every kernel launch to localize the sticky
error, or (c) running under `compute-sanitizer`. None of those fit in the
remaining budget for this task. Reporting honestly per task constraints.

## Deliverables

- `BENCH_CIFAR.md` — this file
- `run_logs/cifar_resnet20_a100.log` — transcript of the crashing CUDA smoke run
