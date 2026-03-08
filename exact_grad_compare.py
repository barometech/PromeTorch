"""
Exact gradient comparison: PyTorch vs PromeTorch
Uses EXACT same random seed and data
"""
import torch
import torch.nn as nn
import numpy as np

# Set seed for reproducibility
torch.manual_seed(12345)
np.random.seed(12345)

print("=" * 60)
print("EXACT Gradient Comparison Test")
print("=" * 60)

# Create deterministic weights (to match PromeTorch initialization)
in_features = 784
out_features = 10
batch_size = 1  # Same as PromeTorch gradient check

# PyTorch Linear initialization: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
bound = 1.0 / np.sqrt(in_features)
weight = torch.empty(out_features, in_features).uniform_(-bound, bound)
bias = torch.empty(out_features).uniform_(-bound, bound)

print(f"\nWeight shape: {weight.shape}")
print(f"Bias shape: {bias.shape}")
print(f"Bound: {bound:.6f}")

# Create test input (normalized like MNIST)
input_data = torch.randn(batch_size, in_features) * 0.5  # Similar range to MNIST
target = torch.tensor([3])  # Class 3

print(f"\nInput shape: {input_data.shape}")
print(f"Input mean: {input_data.mean():.4f}, std: {input_data.std():.4f}")
print(f"Target: {target.item()}")

# Create model
model = nn.Linear(in_features, out_features)
with torch.no_grad():
    model.weight.copy_(weight)
    model.bias.copy_(bias)

# Forward
logits = model(input_data)
print(f"\nLogits: {logits.detach().numpy().flatten()}")

# Compute loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
print(f"Loss: {loss.item():.6f}")

# Backward
loss.backward()

# Print gradients
print(f"\n=== GRADIENTS ===")
print(f"weight.grad shape: {model.weight.grad.shape}")
print(f"weight.grad norm: {model.weight.grad.norm().item():.6f}")

# Print first few values
print(f"\nFirst 5 weight gradients (row 0):")
for i in range(5):
    print(f"  w[{i}] = {model.weight.grad[0, i].item():.8f}")

print(f"\nSample weight gradients from different rows:")
indices = [0, 1, 2, 784, 785, 1568, 2352, 3136, 3920, 7839]
flat_grad = model.weight.grad.flatten()
for i in indices:
    row = i // 784
    col = i % 784
    print(f"  w[{i}] (row {row}, col {col}) = {flat_grad[i].item():.8f}")

# Verify formula manually
print(f"\n=== MANUAL VERIFICATION ===")
softmax = torch.softmax(logits, dim=1)
print(f"Softmax: {softmax.detach().numpy().flatten()}")

# Manual gradient: (softmax - one_hot) / N
one_hot = torch.zeros_like(softmax)
one_hot[0, target[0]] = 1.0
manual_grad_logits = (softmax - one_hot) / batch_size
print(f"Manual grad_logits: {manual_grad_logits.detach().numpy().flatten()}")

# grad_weight = grad_logits.T @ input = [10, 1] @ [1, 784] = [10, 784]
manual_grad_weight = manual_grad_logits.t() @ input_data
print(f"\nManual grad_weight norm: {manual_grad_weight.norm().item():.6f}")
print(f"PyTorch grad_weight norm: {model.weight.grad.norm().item():.6f}")
print(f"Difference: {(manual_grad_weight - model.weight.grad).abs().max().item():.2e}")

# Check specific indices
print(f"\nCompare at specific indices:")
for i in indices:
    row = i // 784
    col = i % 784
    manual = manual_grad_weight[row, col].item()
    pytorch = flat_grad[i].item()
    diff = abs(manual - pytorch)
    print(f"  w[{i}]: manual={manual:.8f}, pytorch={pytorch:.8f}, diff={diff:.2e}")
