"""
Analyze PromeTorch optimization benchmark results.
Reads results_optimized.json and shows speedup of fast paths vs regular paths.
"""
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))

def load(name):
    path = os.path.join(DIR, name)
    with open(path) as f:
        return json.load(f)

data = load("results_optimized.json")

print("=" * 80)
print(f"{'PromeTorch Optimization Analysis':^80}")
print("=" * 80)

# 1. FastOps dispatch comparison
comparisons = [
    ("add_1024",     "fast_add_1024",     "regular_add_1024"),
    ("mul_1024",     "fast_mul_1024",     "regular_mul_1024"),
    ("exp_1024",     "fast_exp_1024",     "regular_exp_1024"),
    ("tanh_1024",    "fast_tanh_1024",    "regular_tanh_1024"),
    ("sigmoid_1024", "fast_sigmoid_1024", "regular_sigmoid_1024"),
]

print(f"\n{'FastOps (trusted dispatch) vs Regular Dispatch':^80}")
print("-" * 80)
print(f"  {'Operation':<25} {'Fast ms':>10} {'Regular ms':>12} {'Speedup':>10}")
print(f"  {'-'*60}")

for name, fast_key, reg_key in comparisons:
    if fast_key in data and reg_key in data:
        f = data[fast_key]
        r = data[reg_key]
        speedup = r / f if f > 0 else 0
        print(f"  {name:<25} {f:>10.4f} {r:>12.4f} {speedup:>9.2f}x")

# 2. Fused linear
print(f"\n{'Fused Linear Performance':^80}")
print("-" * 80)
fused_keys = sorted([k for k in data if "linear" in k.lower() or "mm_add" in k.lower()])
for k in fused_keys:
    print(f"  {k:<45} {data[k]:>10.4f} ms")

# 3. NodePool
print(f"\n{'NodePool vs std::make_shared':^80}")
print("-" * 80)
pool_key = "nodepool_mm_acquire_release"
stl_key = "stl_make_shared_mm"
if pool_key in data and stl_key in data:
    p = data[pool_key]
    s = data[stl_key]
    speedup = s / p if p > 0 else 0
    print(f"  NodePool<MmBackward>:  {p:.4f} ms")
    print(f"  std::make_shared<Mm>:  {s:.4f} ms")
    print(f"  Speedup:               {speedup:.2f}x")

# 4. SGEMM
print(f"\n{'hot::sgemm GFLOPS':^80}")
print("-" * 80)
print(f"  {'Size':<15} {'Time ms':>10} {'GFLOPS':>10}")
print(f"  {'-'*40}")
for k in sorted(data.keys()):
    if k.startswith("sgemm_"):
        sz_str = k.split("_")[-1]
        try:
            sz = int(sz_str)
            t = data[k]
            gflops = (2.0 * sz * sz * sz) / (t * 1e-3) / 1e9
            suffix = " (nt)" if "nt" in k else ""
            print(f"  {sz}{suffix:<10} {t:>10.4f} {gflops:>10.1f}")
        except ValueError:
            pass

# 5. Tensor allocation overhead
print(f"\n{'Tensor Allocation Overhead':^80}")
print("-" * 80)
alloc_keys = [k for k in data if "empty" in k or "malloc" in k]
for k in sorted(alloc_keys):
    print(f"  {k:<35} {data[k]:>10.4f} ms")
if "empty_1024x1024" in data and "raw_malloc_1024x1024" in data:
    overhead = data["empty_1024x1024"] / data["raw_malloc_1024x1024"]
    print(f"  Tensor overhead vs malloc (1024):  {overhead:.1f}x")
if "empty_64x256" in data and "raw_malloc_64x256" in data:
    overhead = data["empty_64x256"] / data["raw_malloc_64x256"]
    print(f"  Tensor overhead vs malloc (64):    {overhead:.1f}x")

# 6. Training
print(f"\n{'End-to-End Training':^80}")
print("-" * 80)
train_keys = sorted([k for k in data if "train" in k or "adam_full" in k or "inference" in k or "adam_step" in k])
for k in train_keys:
    print(f"  {k:<45} {data[k]:>10.4f} ms")

# Also compare with PyTorch if available
try:
    pt = load("results_pytorch.json")
    pm = load("results_promethorch.json")
    print(f"\n{'vs PyTorch (from main benchmark)':^80}")
    print("-" * 80)
    key_targets = [
        "nn_linear_fwd", "autograd_linear_fwd_bwd",
        "train_per_batch_ms", "optim_adam_step",
    ]
    for k in key_targets:
        if k in pt and k in pm:
            ratio = pm[k] / pt[k] if pt[k] > 0 else float('inf')
            print(f"  {k:<35} PT:{pt[k]:>8.4f}  PM:{pm[k]:>8.4f}  {ratio:.2f}x")
except FileNotFoundError:
    pass

print(f"\n{'=' * 80}")
