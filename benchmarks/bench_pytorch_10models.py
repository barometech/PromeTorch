"""
PyTorch 2.x CPU reproduction of PromeTorch's train_10_models.cpp for apples-to-apples
convergence comparison. Mirrors:
  - Model architecture (layer dims, activations)
  - Optimizer + LR (SGD 0.01 / SGD 0.1 / Adam 1e-3 / AdamW 1e-3 wd=0.01)
  - Epochs / iterations
  - Data distribution (same synthetic generators where applicable)
  - MNIST source files (data/mnist/*.ubyte)

Differences that are intentional (baseline is "PyTorch with its own defaults"):
  - PyTorch's Linear init is U(-k, k) with k=1/sqrt(fan_in) — matches PromeTorch's fix
  - Random seeds are fixed to 42 for NumPy, torch, python's random

Usage: python bench_pytorch_10models.py --data data/mnist > run_logs/bench_10models_cpu_pytorch.log
"""

import argparse
import math
import os
import random
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Seeds (match PromeTorch's g_rng(42))
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# MNIST loader (ubyte format, identical to PromeTorch reader)
# ---------------------------------------------------------------------------
def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(num * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)
    return arr


def load_mnist_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        buf = f.read(num)
    return np.frombuffer(buf, dtype=np.uint8)


def normalize_mnist(images_u8):
    # PromeTorch: (pixel/255 - 0.1307) / 0.3081
    x = images_u8.astype(np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    return torch.from_numpy(x)


def evaluate_mnist(model, images_t, labels_np, bs=256):
    model.eval()
    correct = 0
    n = images_t.shape[0]
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = images_t[i:i + bs]
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            correct += int((preds == labels_np[i:i + bs]).sum())
    return 100.0 * correct / n


def print_header(num, name, desc):
    print()
    print("=" * 70)
    print(f"  MODEL {num}: {name}")
    print(f"  {desc}")
    print("=" * 70, flush=True)


def print_result(metric, value):
    print(f"  RESULT: {metric} = {value}", flush=True)


# ---------------------------------------------------------------------------
# Model 1: Linear regression, synthetic y = 3x1 + 2x2 - 1 + noise
#   SGD lr=0.01, 100 epochs, N=500 per epoch
# ---------------------------------------------------------------------------
def train_model_1():
    print_header(1, "Linear Regression", "y = 3x1 + 2x2 - 1 + noise (MSE)")
    t0 = time.time()
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    N = 500
    final = 0.0
    for epoch in range(1, 101):
        X = torch.randn(N, 2)
        Y = (3.0 * X[:, 0:1] + 2.0 * X[:, 1:2] - 1.0 + 0.1 * torch.randn(N, 1))
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        opt.step()
        final = loss.item()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch} | MSE = {final:.6g}")
    w = model.weight.detach().flatten().tolist()
    b = model.bias.detach().item()
    print(f"  Learned: w1={w[0]:.4g}, w2={w[1]:.4g}, b={b:.4g} (true: 3, 2, -1)")
    print_result("Final MSE", final)
    return {"metric": final, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 2: Logistic regression, x1^2+x2^2<1 class
#   SGD lr=0.1, 50 epochs, N=200 per epoch, sigmoid + BCE-manual
# ---------------------------------------------------------------------------
def train_model_2():
    print_header(2, "Logistic Regression", "Binary x1^2+x2^2<1 (BCE)")
    t0 = time.time()
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    N = 200
    final_acc = 0.0
    for epoch in range(1, 51):
        X = 1.5 * torch.randn(N, 2)
        Y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).float().unsqueeze(1)
        opt.zero_grad()
        logits = model(X)
        probs = torch.sigmoid(logits)
        eps = 1e-7
        bce = -(Y * torch.log(probs + eps) + (1 - Y) * torch.log(1 - probs + eps)).mean()
        bce.backward()
        opt.step()
        with torch.no_grad():
            preds = (probs > 0.5).float()
            final_acc = 100.0 * (preds == Y).float().mean().item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} | Loss = {bce.item():.6g} | Acc = {final_acc:.2f}%")
    print_result("Accuracy", final_acc)
    return {"metric": final_acc, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 3: XOR MLP 2->16->1, SGD lr=0.05, 2000 epochs
# ---------------------------------------------------------------------------
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train_model_3():
    print_header(3, "MLP on XOR", "2->16->1 SGD lr=0.05, 2000 epochs")
    t0 = time.time()
    model = XORNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = torch.tensor([[0.], [1.], [1.], [0.]])
    final = 0.0
    for epoch in range(1, 2001):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        opt.step()
        final = loss.item()
        if epoch % 500 == 0:
            print(f"  Epoch {epoch} | MSE = {final:.6g}")
    with torch.no_grad():
        p = model(X).flatten().tolist()
        print(f"  Predictions: [0,0]={p[0]:.4g} [0,1]={p[1]:.4g} [1,0]={p[2]:.4g} [1,1]={p[3]:.4g}")
    print_result("Final MSE", final)
    return {"metric": final, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 4: Simple MNIST 784->128->10, SGD lr=0.01, 2 epochs, bs=64
# ---------------------------------------------------------------------------
class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def mnist_train_loop(model, opt, epochs, bs, train_x, train_y, test_x, test_y_np, log_label):
    criterion = nn.CrossEntropyLoss()
    N = train_x.shape[0]
    labels_t = torch.from_numpy(train_y.astype(np.int64))
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        batches = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            inp = train_x[idx]
            tgt = labels_t[idx]
            opt.zero_grad()
            logits = model(inp)
            loss = criterion(logits, tgt)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        acc = evaluate_mnist(model, test_x, test_y_np)
        print(f"  Epoch {epoch} | Loss = {epoch_loss / batches:.6g} | Test Acc = {acc:.2f}%", flush=True)
    return evaluate_mnist(model, test_x, test_y_np)


def train_model_4(train_x, train_y, test_x, test_y):
    print_header(4, "Simple MNIST MLP", "784->128->10 SGD lr=0.01, 2 epochs")
    t0 = time.time()
    model = SimpleMNIST()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    acc = mnist_train_loop(model, opt, 2, 64, train_x, train_y, test_x, test_y, "m4")
    print_result("Test Accuracy", acc)
    return {"metric": acc, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 5: Deep MNIST 784->512->256->128->10, Adam lr=1e-3, 2 epochs, bs=64
# ---------------------------------------------------------------------------
class DeepMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc4(h)


def train_model_5(train_x, train_y, test_x, test_y):
    print_header(5, "Deep MNIST MLP", "784->512->256->128->10 Adam lr=0.001, 2 epochs")
    t0 = time.time()
    model = DeepMNIST()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    acc = mnist_train_loop(model, opt, 2, 64, train_x, train_y, test_x, test_y, "m5")
    print_result("Test Accuracy", acc)
    return {"metric": acc, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 6: MNIST with Dropout 784->256->128->10, Adam lr=1e-3, dropout 0.2, 2 epochs
# ---------------------------------------------------------------------------
class DropoutMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop1(h)
        h = F.relu(self.fc2(h))
        h = self.drop2(h)
        return self.fc3(h)


def train_model_6(train_x, train_y, test_x, test_y):
    print_header(6, "MNIST with Dropout", "784->256->128->10 Adam + Dropout 0.2, 2 epochs")
    t0 = time.time()
    model = DropoutMNIST()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    acc = mnist_train_loop(model, opt, 2, 64, train_x, train_y, test_x, test_y, "m6")
    print_result("Test Accuracy", acc)
    return {"metric": acc, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 7: RNNCell sine regression, 100 epochs, batch=32, seq_len=10
# ---------------------------------------------------------------------------
class SineRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.RNNCell(1, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, T, 1)
        B, T, _ = x.shape
        h = torch.zeros(B, 32)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


def train_model_7():
    print_header(7, "RNN Sine Wave", "RNNCell 1->32 + Linear, Adam lr=0.005, 100 epochs")
    t0 = time.time()
    model = SineRNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    seq_len = 10
    batch = 32
    final = 0.0
    for epoch in range(1, 101):
        phase = (torch.rand(batch) * 6.28).unsqueeze(1)  # (B,1)
        t_idx = torch.arange(seq_len, dtype=torch.float32) * 0.5
        inputs = torch.sin(phase + t_idx).unsqueeze(-1)  # (B,T,1)
        targets = torch.sin(phase.squeeze(1) + seq_len * 0.5).unsqueeze(1)  # (B,1)
        opt.zero_grad()
        pred = model(inputs)
        loss = ((pred - targets) ** 2).mean()
        loss.backward()
        opt.step()
        final = loss.item()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch} | MSE = {final:.6g}")
    print_result("Final MSE", final)
    return {"metric": final, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 8: LSTMCell sequence classification (sum sign), 80 epochs, batch=64, seq_len=8
# ---------------------------------------------------------------------------
class SeqClassLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(1, 32)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 32)
        c = torch.zeros(B, 32)
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
        return self.fc(h)


def train_model_8():
    print_header(8, "LSTM Sequence Classifier", "LSTMCell 1->32 + Linear, Adam lr=0.005, 80 epochs")
    t0 = time.time()
    model = SeqClassLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    seq_len = 8
    batch = 64
    final_acc = 0.0
    for epoch in range(1, 81):
        inp = torch.randn(batch, seq_len, 1)
        sums = inp.sum(dim=(1, 2))
        tgt = (sums > 0).long()
        opt.zero_grad()
        logits = model(inp)
        loss = criterion(logits, tgt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            final_acc = 100.0 * (preds == tgt).float().mean().item()
        if epoch % 20 == 0 or epoch <= 3:
            print(f"  Epoch {epoch} | Loss = {loss.item():.6g} | Acc = {final_acc:.2f}%")
    print_result("Final Accuracy", final_acc)
    return {"metric": final_acc, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 9: GRUCell trend classification, 80 epochs, batch=64, seq_len=10
# ---------------------------------------------------------------------------
class TrendGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(1, 24)
        self.fc = nn.Linear(24, 2)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, 24)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


def train_model_9():
    print_header(9, "GRU Trend Detector", "GRUCell 1->24 + Linear, Adam lr=0.005, 80 epochs")
    t0 = time.time()
    model = TrendGRU()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    seq_len = 10
    batch = 64
    final_acc = 0.0
    for epoch in range(1, 81):
        slope = torch.empty(batch).uniform_(-1.0, 1.0)
        base = 0.3 * torch.randn(batch)
        t_idx = torch.arange(seq_len, dtype=torch.float32)
        inp = base.unsqueeze(1) + slope.unsqueeze(1) * t_idx / seq_len + 0.03 * torch.randn(batch, seq_len)
        inp = inp.unsqueeze(-1)
        tgt = (slope > 0).long()
        opt.zero_grad()
        logits = model(inp)
        loss = criterion(logits, tgt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            final_acc = 100.0 * (preds == tgt).float().mean().item()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch} | Loss = {loss.item():.6g} | Acc = {final_acc:.2f}%")
    print_result("Final Accuracy", final_acc)
    return {"metric": final_acc, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Model 10: Wide MNIST 784->1024->512->10, AdamW lr=0.001 wd=0.01, 2 epochs + save/load
# ---------------------------------------------------------------------------
class WideMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def train_model_10(train_x, train_y, test_x, test_y):
    print_header(10, "Wide MNIST + Serialization", "784->1024->512->10 AdamW lr=0.001 wd=0.01, 2 epochs")
    t0 = time.time()
    model = WideMNIST()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    acc = mnist_train_loop(model, opt, 2, 64, train_x, train_y, test_x, test_y, "m10")
    acc_before = evaluate_mnist(model, test_x, test_y)
    torch.save(model.state_dict(), "model10_pytorch.pt")
    m2 = WideMNIST()
    m2.load_state_dict(torch.load("model10_pytorch.pt", weights_only=True))
    m2.eval()
    acc_after = evaluate_mnist(m2, test_x, test_y)
    print(f"  Acc before save: {acc_before:.2f}% | Acc after load: {acc_after:.2f}%")
    if abs(acc_before - acc_after) < 0.01:
        print("  Serialization: PASS (identical accuracy)")
    else:
        print("  Serialization: MISMATCH!")
    print_result("Test Accuracy", acc_after)
    return {"metric": acc_after, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  PyTorch {torch.__version__} — 10 Models Training (CPU)")
    print(f"  threads: {torch.get_num_threads()}")
    print("=" * 60, flush=True)

    total_start = time.time()
    results = {}

    results[1] = train_model_1()
    results[2] = train_model_2()
    results[3] = train_model_3()

    print("\n  Loading MNIST data...")
    train_u8 = load_mnist_images(os.path.join(args.data, "train-images-idx3-ubyte"))
    train_lbl = load_mnist_labels(os.path.join(args.data, "train-labels-idx1-ubyte"))
    test_u8 = load_mnist_images(os.path.join(args.data, "t10k-images-idx3-ubyte"))
    test_lbl = load_mnist_labels(os.path.join(args.data, "t10k-labels-idx1-ubyte"))
    train_x = normalize_mnist(train_u8)
    test_x = normalize_mnist(test_u8)
    print(f"  MNIST loaded: {train_x.shape[0]} train, {test_x.shape[0]} test", flush=True)

    results[4] = train_model_4(train_x, train_lbl, test_x, test_lbl)
    results[5] = train_model_5(train_x, train_lbl, test_x, test_lbl)
    results[6] = train_model_6(train_x, train_lbl, test_x, test_lbl)

    results[7] = train_model_7()
    results[8] = train_model_8()
    results[9] = train_model_9()

    results[10] = train_model_10(train_x, train_lbl, test_x, test_lbl)

    total_s = time.time() - total_start
    print()
    print("=" * 70)
    print("  ALL 10 MODELS COMPLETE (PyTorch CPU)")
    print(f"  Total time: {total_s:.2f} seconds")
    print("=" * 70)

    # Summary table
    print("\n  --- Per-model summary ---")
    for i in range(1, 11):
        r = results[i]
        print(f"  Model {i}: metric={r['metric']:.6g}  time={r['time_s']:.2f}s")


if __name__ == "__main__":
    main()
