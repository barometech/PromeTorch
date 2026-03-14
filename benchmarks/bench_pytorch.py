"""
PromeTorch vs PyTorch Benchmark — PyTorch side
Single-threaded CPU, outputs JSON to results_pytorch.json
"""
import torch
import time
import json
import os
import sys

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42)

WARMUP = 5

def bench(fn, warmup=WARMUP, iters=100):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed / iters

results = {}

# ============================================================================
# 1. Tensor Creation
# ============================================================================
print("=== Tensor Creation ===")
for sz in [64, 256, 1024, 2048]:
    results[f"randn_{sz}"] = bench(lambda s=sz: torch.randn(s, s), iters=500)
    results[f"zeros_{sz}"] = bench(lambda s=sz: torch.zeros(s, s), iters=500)
    results[f"ones_{sz}"]  = bench(lambda s=sz: torch.ones(s, s), iters=500)
    print(f"  {sz}x{sz}: randn={results[f'randn_{sz}']:.4f} zeros={results[f'zeros_{sz}']:.4f} ones={results[f'ones_{sz}']:.4f} ms")

# ============================================================================
# 2. Element-wise Operations (1024x1024)
# ============================================================================
print("=== Element-wise Ops ===")
a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)
a_pos = a.abs() + 1e-6  # for log/sqrt

elem_ops = {
    "add_1024": lambda: a + b,
    "mul_1024": lambda: a * b,
    "sub_1024": lambda: a - b,
    "div_1024": lambda: a / b,
    "exp_1024": lambda: torch.exp(a),
    "log_1024": lambda: torch.log(a_pos),
    "sin_1024": lambda: torch.sin(a),
    "cos_1024": lambda: torch.cos(a),
    "tanh_1024": lambda: torch.tanh(a),
    "sigmoid_1024": lambda: torch.sigmoid(a),
    "relu_1024": lambda: torch.relu(a),
    "sqrt_1024": lambda: torch.sqrt(a_pos),
    "abs_1024": lambda: torch.abs(a),
    "neg_1024": lambda: -a,
}
for name, fn in elem_ops.items():
    results[name] = bench(fn, iters=200)
    print(f"  {name}: {results[name]:.4f} ms")

# ============================================================================
# 3. Reductions (1024x1024)
# ============================================================================
print("=== Reductions ===")
red_ops = {
    "sum_1024": lambda: a.sum(),
    "mean_1024": lambda: a.mean(),
    "max_1024": lambda: a.max(),
    "min_1024": lambda: a.min(),
    "var_1024": lambda: a.var(),
    "std_1024": lambda: a.std(),
    "argmax_1024": lambda: a.argmax(),
    "sum_dim0_1024": lambda: a.sum(dim=0),
    "mean_dim1_1024": lambda: a.mean(dim=1),
}
for name, fn in red_ops.items():
    results[name] = bench(fn, iters=200)
    print(f"  {name}: {results[name]:.4f} ms")

# ============================================================================
# 4. Linear Algebra
# ============================================================================
print("=== Linear Algebra ===")
for sz in [256, 512, 1024, 2048]:
    ma = torch.randn(sz, sz)
    mb = torch.randn(sz, sz)
    it = 3 if sz >= 2048 else (5 if sz >= 1024 else (20 if sz >= 512 else 100))
    results[f"mm_{sz}"] = bench(lambda: torch.mm(ma, mb), iters=it)
    gflops = (2.0 * sz * sz * sz) / (results[f"mm_{sz}"] * 1e-3) / 1e9
    print(f"  mm_{sz}: {results[f'mm_{sz}']:.4f} ms ({gflops:.1f} GFLOPS)")

# mv
m = torch.randn(512, 256)
v = torch.randn(256)
results["mv_512x256"] = bench(lambda: torch.mv(m, v), iters=500)
print(f"  mv_512x256: {results['mv_512x256']:.4f} ms")

# bmm
ba = torch.randn(8, 256, 256)
bb = torch.randn(8, 256, 256)
results["bmm_8x256"] = bench(lambda: torch.bmm(ba, bb), iters=10)
print(f"  bmm_8x256: {results['bmm_8x256']:.4f} ms")

# dot
d1 = torch.randn(10000)
d2 = torch.randn(10000)
results["dot_10k"] = bench(lambda: torch.dot(d1, d2), iters=1000)
print(f"  dot_10k: {results['dot_10k']:.4f} ms")

# ============================================================================
# 5. Autograd: Linear forward+backward
# ============================================================================
print("=== Autograd ===")
linear_ag = torch.nn.Linear(512, 256)
x_ag = torch.randn(64, 512, requires_grad=True)

def fn_autograd():
    if x_ag.grad is not None:
        x_ag.grad = None
    linear_ag.zero_grad()
    y = linear_ag(x_ag)
    loss = y.sum()
    loss.backward()

results["autograd_linear_fwd_bwd"] = bench(fn_autograd, iters=100)
print(f"  autograd_linear_fwd_bwd: {results['autograd_linear_fwd_bwd']:.4f} ms")

# ============================================================================
# 6. NN Modules Forward (no grad)
# ============================================================================
print("=== NN Modules ===")

# Linear
lin = torch.nn.Linear(512, 256)
x_lin = torch.randn(64, 512)
with torch.no_grad():
    results["nn_linear_fwd"] = bench(lambda: lin(x_lin), iters=200)
print(f"  nn_linear_fwd: {results['nn_linear_fwd']:.4f} ms")

# Conv2d
conv = torch.nn.Conv2d(3, 16, 3, padding=1)
x_conv = torch.randn(16, 3, 32, 32)
with torch.no_grad():
    results["nn_conv2d_fwd"] = bench(lambda: conv(x_conv), iters=100)
print(f"  nn_conv2d_fwd: {results['nn_conv2d_fwd']:.4f} ms")

# BatchNorm1d
bn = torch.nn.BatchNorm1d(256)
bn.train()
x_bn = torch.randn(64, 256)
results["nn_batchnorm1d_fwd"] = bench(lambda: bn(x_bn), iters=200)
print(f"  nn_batchnorm1d_fwd: {results['nn_batchnorm1d_fwd']:.4f} ms")

# LSTM
lstm = torch.nn.LSTM(128, 64, num_layers=1, batch_first=True)
x_lstm = torch.randn(32, 10, 128)
with torch.no_grad():
    results["nn_lstm_fwd"] = bench(lambda: lstm(x_lstm), iters=100)
print(f"  nn_lstm_fwd: {results['nn_lstm_fwd']:.4f} ms")

# ============================================================================
# 7. Optimizer Step (Adam, SGD on 3-layer MLP)
# ============================================================================
print("=== Optimizers ===")

model = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
)
x_opt = torch.randn(64, 256)
tgt_opt = torch.randint(0, 10, (64,))
criterion = torch.nn.CrossEntropyLoss()

# Adam
opt_adam = torch.optim.Adam(model.parameters(), lr=0.001)
def fn_adam():
    opt_adam.zero_grad()
    out = model(x_opt)
    loss = criterion(out, tgt_opt)
    loss.backward()
    opt_adam.step()

results["optim_adam_step"] = bench(fn_adam, iters=100)
print(f"  optim_adam_step: {results['optim_adam_step']:.4f} ms")

# SGD
opt_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
def fn_sgd():
    opt_sgd.zero_grad()
    out = model(x_opt)
    loss = criterion(out, tgt_opt)
    loss.backward()
    opt_sgd.step()

results["optim_sgd_step"] = bench(fn_sgd, iters=100)
print(f"  optim_sgd_step: {results['optim_sgd_step']:.4f} ms")

# ============================================================================
# 8. Full Training Loop: MLP 784->512->256->10, 100 batches
# ============================================================================
print("=== Training Loop ===")

model2 = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)
opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
criterion2 = torch.nn.CrossEntropyLoss()

start = time.perf_counter()
for _ in range(100):
    x = torch.randn(64, 784)
    target = torch.randint(0, 10, (64,))
    opt2.zero_grad()
    out = model2(x)
    loss = criterion2(out, target)
    loss.backward()
    opt2.step()
total_ms = (time.perf_counter() - start) * 1000.0
results["train_100batch_total_ms"] = total_ms
results["train_per_batch_ms"] = total_ms / 100.0
print(f"  100 batches: {total_ms:.1f} ms total, {total_ms/100:.2f} ms/batch")

# ============================================================================
# Save
# ============================================================================
outpath = os.path.join(os.path.dirname(__file__), "results_pytorch.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")
print(f"PyTorch version: {torch.__version__}")
