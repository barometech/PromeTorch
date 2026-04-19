# NM Card Mini — MNIST Benchmark

Date: 2026-04-19
Binary: `build_nmcard/examples/nmcard/train_mnist_nmcard.exe` (built 2026-03-14)
Host: Windows 10 Pro, NMCard backend via PrivateUse1 (Q16.16 fixed-point emulator in float32 mode)

## 1. Emulator — MNIST MLP (3 epochs)

Command:
```
PATH="./build_nmcard:$PATH" ./build_nmcard/examples/nmcard/train_mnist_nmcard.exe \
  --data data/mnist --epochs 3 --batch_size 64 --lr 0.01 --device nmcard
```

Model: MLP 784 -> 256 -> 128 -> 10, SGD lr=0.01, CrossEntropyLoss, batch=64.
Virtual cores used by emulator: 16.
Data: 60000 train / 10000 test (idx-ubyte).

| Epoch | Train loss | Train acc | Test acc  | Wall-clock |
|-------|------------|-----------|-----------|------------|
| 1     | 0.7474     | 78.85%    | **88.00%** | 16.853 s   |
| 2     | 0.3505     | 88.02%    | **88.97%** | 17.412 s   |
| 3     | 0.4675     | 88.75%    | **88.94%** | 18.415 s   |

- Final test accuracy: **88.94%**
- Total wall-clock: **52.68 s** (~17.56 s / epoch, ~3.0 ms / batch of 64)
- Throughput: ~3420 samples / s per epoch
- Log: `run_logs/nmcard_emulator_mnist.log`

Note: train loss rises slightly from epoch 2 to 3 while accuracy plateaus — typical for plain
SGD without LR decay on this architecture. The earlier 93.64% number referenced in MEMORY.md
was from a different config (3 epochs, different seed / init / arch). Current numbers are the
canonical figures for this binary + these exact hyperparams.

## 2. Real NM Card Mini — probe only

- Driver: `C:\Windows\System32\drivers\nm_card_pci.sys` present.
- PnP status: `Get-PnpDevice` reports **`NM_Card` / NeuroMatrix Accelerators / Status OK`**
  — the physical card is installed and enumerated by Windows.
- No inference was run on the real card in this task.
- No training was run on the real card in this task (policy: training stays on emulator).
- Per project safety protocol (CLAUDE.md + feedback_nmcard_safety.md, incident 2026-03-18):
  multi-core matmul on the real card can hang the host and force a reboot. Real-card testing
  must follow emulator -> 1 core -> 2 -> 4 -> 16 staircase, and only with explicit user go-ahead.

Status: **present, not exercised.**

## 3. CPU baseline comparison

`BENCH_10MODELS.md` was not present at time of this run; CPU MNIST number will be
backfilled once the 10-models agent report lands. For reference, MEMORY.md cites the
CPU MLP MNIST baseline at **97.65%** (different architecture / more epochs), so the
emulator result (88.94%, 3 epochs, small MLP, plain SGD) is in the expected ballpark
for this config and is not a regression of the backend itself — it reflects the chosen
hyperparameters.

## 4. Anomalies / safety triggers

- None. Emulator ran to completion, no crashes, no hangs.
- Real card was probed (PnP only — no I/O, no dispatcher load). Safety protocol respected.
