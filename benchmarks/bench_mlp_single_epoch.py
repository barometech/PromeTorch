#!/usr/bin/env python3
"""
Канонический MLP MNIST benchmark для сравнения PromeTorch vs PyTorch на одном
железе (Эльбрус 8C2 или A100). Идентичный с PromeTorch'овским
`examples/mnist/train_mnist_mlp.cpp` arch + optimizer + batch.

Arch:    784 → 512 → 256 → 128 → 10 (ReLU, no dropout)
Opt:     Adam lr=1e-3
Batch:   64
Epochs:  1
Seed:    42 (torch + numpy + python)

Вывод: final train loss, test accuracy, wall-clock.
"""
import argparse
import gzip
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_mnist_ubyte(data_dir: Path):
    def _read_images(path: Path) -> np.ndarray:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            return arr.reshape(n, rows * cols).astype(np.float32) / 255.0

    def _read_labels(path: Path) -> np.ndarray:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            assert magic == 2049
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    # Try common naming schemes
    candidates = {
        "train_x": ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"],
        "train_y": ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"],
        "test_x": ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"],
        "test_y": ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"],
    }
    paths = {}
    for k, names in candidates.items():
        for n in names:
            p = data_dir / n
            if p.exists():
                paths[k] = p
                break
        else:
            raise FileNotFoundError(f"Missing MNIST file: tried {names} in {data_dir}")
    return (
        _read_images(paths["train_x"]),
        _read_labels(paths["train_y"]),
        _read_images(paths["test_x"]),
        _read_labels(paths["test_y"]),
    )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--data", default="data/mnist", type=Path)
    ap.add_argument("--epochs", default=1, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--num_workers", default=0, type=int,
                    help="DataLoader workers. Keep 0 for Эльбрус (forking under ALT/Elbrus Linux is slow)")
    ap.add_argument("--threads", default=0, type=int,
                    help="torch.set_num_threads() — set if known (32 on E8C2). 0 = default")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    print(f"=== PyTorch MNIST MLP benchmark ===")
    print(f"Arch:       784→512→256→128→10 (ReLU)")
    print(f"Optimizer:  Adam lr={args.lr}")
    print(f"Batch:      {args.batch_size}")
    print(f"Epochs:     {args.epochs}")
    print(f"Device:     {device}")
    print(f"Threads:    {torch.get_num_threads()}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Seed:       {args.seed}")
    sys.stdout.flush()

    train_x, train_y, test_x, test_y = load_mnist_ubyte(args.data)
    print(f"Loaded MNIST: train={len(train_x)}, test={len(test_x)}")

    train_t = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_t = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = DataLoader(train_t, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_t, batch_size=1024, shuffle=False,
                             num_workers=args.num_workers)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params:     {n_params:,}")

    model.train()
    t_train_start = time.perf_counter()
    total_loss = 0.0
    total_batches = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_batches += 1
        total_loss += epoch_loss
        total_batches += epoch_batches
        print(f"Epoch {epoch+1}: avg_loss={epoch_loss/epoch_batches:.4f}")
        sys.stdout.flush()

    train_sec = time.perf_counter() - t_train_start

    model.eval()
    correct = 0
    total = 0
    t_eval_start = time.perf_counter()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    eval_sec = time.perf_counter() - t_eval_start
    acc = 100.0 * correct / total

    print("\n=== RESULT ===")
    print(f"Train wall-clock:   {train_sec:.2f} s  ({train_sec/args.epochs:.2f} s/epoch)")
    print(f"Eval wall-clock:    {eval_sec:.2f} s  (test set {total} samples)")
    print(f"Final train loss:   {total_loss/max(total_batches,1):.4f}")
    print(f"Test accuracy:      {acc:.2f}%")
    print(f"Throughput:         {len(train_x)*args.epochs / train_sec:.0f} samples/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
