"""
Compare PromeTorch vs PyTorch benchmark results
Reads results_pytorch.json and results_promethorch.json
"""
import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

def load(name):
    path = os.path.join(DIR, name)
    with open(path) as f:
        return json.load(f)

pt = load("results_pytorch.json")
pm = load("results_promethorch.json")

categories = [
    ("Tensor Creation", lambda k: k.startswith(("randn_", "zeros_", "ones_"))),
    ("Element-wise Ops", lambda k: k.endswith("_1024") and not any(k.startswith(p) for p in
        ["sum_", "mean_", "max_", "min_", "var_", "std_", "argmax_", "sum_dim", "mean_dim",
         "mm_", "mv_", "bmm_", "dot_", "randn_", "zeros_", "ones_"])),
    ("Reductions", lambda k: any(k.startswith(p) for p in
        ["sum_", "mean_", "max_", "min_", "var_", "std_", "argmax_"]) and "1024" in k),
    ("Linear Algebra", lambda k: k.startswith(("mm_", "mv_", "bmm_", "dot_"))),
    ("Autograd", lambda k: k.startswith("autograd_")),
    ("NN Modules", lambda k: k.startswith("nn_")),
    ("Optimizers", lambda k: k.startswith("optim_")),
    ("Training Loop", lambda k: k.startswith("train_")),
]

all_keys = set(pt.keys()) | set(pm.keys())
common_keys = set(pt.keys()) & set(pm.keys())

print("=" * 82)
print(f"{'PromeTorch vs PyTorch — CPU Benchmark Comparison':^82}")
print("=" * 82)

total_pt = 0.0
total_pm = 0.0
count = 0

for cat_name, filt in categories:
    keys = sorted([k for k in common_keys if filt(k)])
    if not keys:
        continue

    print(f"\n{'-'*82}")
    print(f"  {cat_name}")
    print(f"{'-'*82}")
    print(f"  {'Operation':<35} {'PyTorch ms':>12} {'PromeTorch ms':>14} {'Ratio':>8}")
    print(f"  {'-'*73}")

    for key in keys:
        pt_val = pt[key]
        pm_val = pm[key]
        if pt_val > 0:
            ratio = pm_val / pt_val
        else:
            ratio = float('inf')

        marker = ""
        if ratio > 10:
            marker = " <<<"
        elif ratio > 3:
            marker = " <<"
        elif ratio > 1.5:
            marker = " <"
        elif ratio < 0.67:
            marker = " >>>"
        elif ratio < 0.9:
            marker = " >>"

        print(f"  {key:<35} {pt_val:>12.4f} {pm_val:>14.4f} {ratio:>7.2f}x{marker}")

        # accumulate for summary (skip training totals, only per-batch)
        if "total" not in key:
            total_pt += pt_val
            total_pm += pm_val
            count += 1

print(f"\n{'=' * 82}")
if count > 0 and total_pt > 0:
    avg_ratio = total_pm / total_pt
    print(f"  Overall weighted ratio: {avg_ratio:.2f}x  ({count} benchmarks)")
print(f"  Ratio > 1.0 = PyTorch faster | < 1.0 = PromeTorch faster")
print(f"  <<< >10x slower | << >3x slower | < >1.5x slower")
print(f"  >>> >1.5x faster | >> >1.1x faster")
print(f"{'=' * 82}")

# only-in lists
only_pt = sorted(set(pt.keys()) - set(pm.keys()))
only_pm = sorted(set(pm.keys()) - set(pt.keys()))
if only_pt:
    print(f"\n  Only in PyTorch results: {', '.join(only_pt)}")
if only_pm:
    print(f"\n  Only in PromeTorch results: {', '.join(only_pm)}")
