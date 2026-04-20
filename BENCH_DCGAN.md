# BENCH_DCGAN — PromeTorch DCGAN on MNIST (30 epochs, CPU)

DCGAN training on MNIST, trained end-to-end with PromeTorch's own autograd
engine, Adam optimizer, and BatchNorm2d. Non-saturating generator loss,
BCE-with-logits discriminator loss. Samples are a 4×4 grid of images produced
from the same fixed noise vector at every milestone epoch, so the visual
progression is directly comparable across epochs.

- Source: `examples/gan/train_gan.cpp` (unmodified for this run)
- Binary: `build_examples/examples/gan/train_gan.exe` (CPU build)
- Host: Windows 10, MSVC 2019, single-threaded PromeTorch CPU path
- Full log: `run_logs/gan_train_full.log`
- Sample grids: `docs/screenshots/gan_samples/epoch_{05,10,15,20,25,30}.ppm` (and `.png`)

## Architecture

```
Generator (765 761 params)
  z[B,100]                          (i.i.d. N(0,1))
   ├─ Linear(100 → 128·7·7)         → [B,128·7·7]
   ├─ ReLU
   ├─ reshape                       → [B,128,7,7]
   ├─ ConvTranspose2d(128→64, k4 s2 p1)  → [B,64,14,14]
   ├─ BatchNorm2d(64)
   ├─ ReLU
   ├─ ConvTranspose2d(64→1,  k4 s2 p1)   → [B,1,28,28]
   └─ Tanh                          ∈ [-1, 1]

Discriminator (138 817 params)
  x[B,1,28,28]                      (real MNIST normalized to [-1, 1])
   ├─ Conv2d(1→64,   k4 s2 p1)       → [B,64,14,14]
   ├─ LeakyReLU(0.2)
   ├─ Conv2d(64→128, k4 s2 p1)       → [B,128,7,7]
   ├─ BatchNorm2d(128)
   ├─ LeakyReLU(0.2)
   ├─ reshape                        → [B,128·7·7]
   └─ Linear(128·7·7 → 1)            logits (BCEWithLogits)
```

Generator objective: **non-saturating** `G: minimize BCE(D(G(z)), 1)` (as
opposed to the original `min log(1 - D(G(z)))` formulation), which is the
DCGAN-paper recommendation and avoids early-epoch vanishing gradients.

## Config

| Knob | Value |
|---|---|
| Device | **CPU** (CUDA build hung on this DCGAN — see "Device notes" below) |
| Dataset | MNIST train, 60 000 images, normalized to `[-1, 1]` |
| Batch size | 64 |
| Epochs | 30 |
| Learning rate | 2 × 10⁻⁴ (both G and D) |
| Optimizer | Adam, β₁ = 0.5, β₂ = 0.999 (DCGAN-paper values) |
| Sample interval | every 5 epochs (4×4 grid, fixed noise) |
| Seed (shuffle + noise) | 1234 |

Command:

```
train_gan.exe --device cpu --data data/mnist --epochs 30 \
              --batch_size 64 --lr 0.0002 --beta1 0.5 \
              --sample_every 5 --out run_logs/gan_samples
```

## Loss curves (D / G)

Means over each full epoch (`log_interval` was per-50-batch; epoch-level
values shown here as the headline signal).

| Epoch | D_loss | G_loss | Wall-clock (s) |
|------:|-------:|-------:|---------------:|
|  1 | 0.334 | 2.44 | 168.4 |
|  2 | 0.035 | 4.22 | 169.4 |
|  3 | 0.024 | 4.74 | 172.8 |
|  5 | 0.034 | 4.97 | 173.5 |
| 10 | 0.315 | 3.21 | 170.6 |
| 15 | 0.554 | 1.95 | 168.1 |
| 20 | 0.447 | 1.96 | 167.1 |
| 25 | 0.401 | 2.22 | 166.7 |
| 30 | 0.264 | 2.47 | 166.6 |

**Shape of the curve.** Classic DCGAN dynamics:

- **Epochs 1–5:** D dominates hard (D_loss collapses toward 0, G_loss climbs
  past 4) — the discriminator is winning, the generator is producing obvious
  noise, so D's job is trivial. This matches the epoch-5 sample grid (pure
  static, no shape).
- **Epochs 6–10:** The non-saturating G objective kicks in, G starts
  producing credible structure, D's task gets genuinely harder, D_loss rises
  back into the 0.3–0.5 band, G_loss falls from ~5 down through ~3.
- **Epochs 11–30:** Game enters a stable adversarial equilibrium. D_loss
  oscillates in **0.3–0.55** (nowhere near 0 — so no mode-collapse signal),
  G_loss in **1.9–2.5** (nowhere near exploding — so G is not being
  out-gunned). This is the healthy plateau GAN training is supposed to
  settle into.
- **No divergence, no mode collapse.** Neither D_loss → 0 nor
  G_loss → ∞ at any point after epoch 5.

**Wall-clock.** Mean 169.6 s/epoch, **total 5 089.5 s (84.8 min)** across all
30 epochs, single-threaded CPU PromeTorch path.

## Samples (fixed noise, 4×4 grid, same z for every row)

All samples generated from **the same 16 noise vectors** across all 6
checkpoints — so each cell in the grid shows the same G(zᵢ) as training
progresses. This is the standard way to visualize DCGAN convergence.

| Epoch 5 | Epoch 10 |
|---|---|
| ![epoch 5](docs/screenshots/gan_samples/epoch_05.png) | ![epoch 10](docs/screenshots/gan_samples/epoch_10.png) |

| Epoch 20 | Epoch 30 |
|---|---|
| ![epoch 20](docs/screenshots/gan_samples/epoch_20.png) | ![epoch 30](docs/screenshots/gan_samples/epoch_30.png) |

(PPM originals: `docs/screenshots/gan_samples/epoch_{05,10,15,20,25,30}.ppm`. Grids
at epoch 15 and 25 also exist and show the expected intermediate stages.)

### Per-epoch visual assessment

- **Epoch 5** — Pure high-frequency noise. No stroke structure, no
  recognizable background/foreground separation. Histogram is ~50 % dark,
  18 % bright, 31 % mid-gray (std = 93 on 0–255) — the generator is still
  emitting a near-uniform mid-gray texture that Tanh has only partially
  bimodalized. Matches the D_loss ≈ 0.03 / G_loss ≈ 5 regime (D wins
  trivially).
- **Epoch 10** — Blobs resolve into **digit-shaped clumps on black
  background.** The 16 samples all look like "things on a dark canvas". No
  digit is yet *readable*, but MNIST's statistics have locked in:
  histogram shifts to 77 % dark / 8 % bright / 15 % mid — i.e., the
  background/foreground split is correct and the bimodal distribution
  expected of MNIST is present.
- **Epoch 20** — **Recognizable strokes.** About half of the 16 samples have
  clear digit-like topology (closed loops, vertical strokes, curves). A few
  are still ambiguous blobs. Background is crisp black, strokes are bright
  white — the Tanh output has fully bimodalized (dark 79 %, bright 11 %,
  mid only 10 %).
- **Epoch 30** — **Readable digits.** Visually identifiable in the grid are
  0, 2, 3, 7, 8, 9 (and plausible 4, 5, 6). Strokes have appropriate width,
  closed loops are actually closed, and the digits show **genuine variety
  across the 16 fixed-noise cells** (different digits emerge from different
  zᵢ — not 16 copies of the same digit). That is the crucial
  *no-mode-collapse* sanity check, and it passes by eye.

Image statistics tracking MNIST's own (~80 % dark background, ~12 % bright
strokes) from epoch 15 onward, stable through 30, is also consistent with
the Frechet-distance literature: DCGAN on MNIST typically converges to
"recognizable digits with occasional artifacts" at this capacity and 30
epochs is firmly in that regime.

## Device notes

- **CUDA build attempted first** (`build_cuda124/examples/gan/train_gan.exe`).
  The binary launches, loads MNIST, prints parameter counts, then **hangs at
  the first forward pass with 0 % GPU utilization** — a known issue on this
  particular ConvTranspose2d + BatchNorm2d path on CUDA for this snapshot
  (the BatchNorm2d CUDA bounce was added in `584e531` but DCGAN's combination
  still deadlocks pre-launch; likely a transposed-convolution CUDA kernel
  dispatch issue, outside the scope of "do not modify model"). CUDA was
  therefore not usable for this benchmark and the task's explicit CPU
  fallback was taken. 4 separate smoke attempts (1-batch, 3-batch, 5-batch,
  60-batch) all hung with identical symptom: prints up to "D params: 138817"
  then silence, process eventually reaped by timeout, 0 % GPU util
  throughout.
- **CPU path worked end-to-end with zero intervention** — no NaNs, no
  divergence, no mode collapse, 30 / 30 epochs completed. Single-threaded;
  the PromeBLAS multi-threaded AVX2 path is not routed for
  ConvTranspose2d's im2col-on-output path in this binary.

## Summary

| Metric | Value |
|---|---|
| Epochs completed | **30 / 30** |
| Final D_loss | 0.264 |
| Final G_loss | 2.47 |
| D_loss equilibrium band (ep 15–30) | 0.26 – 0.55 |
| G_loss equilibrium band (ep 15–30) | 1.86 – 2.58 |
| Mode collapse? | **No** — 16 fixed-noise samples show distinct digits |
| Divergence? | **No** — neither loss escapes bounds |
| Wall-clock per epoch (mean) | 169.6 s |
| Wall-clock total | **5 089.5 s (84.8 min)** |
| Device | CPU (CUDA hung) |
| Sample gallery | `docs/screenshots/gan_samples/epoch_{05,10,15,20,25,30}.{ppm,png}` |

DCGAN trains stably on PromeTorch's autograd + Adam + BatchNorm2d stack and
produces recognizable MNIST digits by epoch 30 with the canonical DCGAN
hyperparameters (lr 2e-4, β₁ 0.5, batch 64).
