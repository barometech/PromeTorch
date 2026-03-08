"""
Minimal test to compare gradient values between PromeTorch and PyTorch.

This test matches EXACTLY the SimpleLinear test in train_mnist_mlp.cpp:
- 784 -> 10 linear layer
- Batch size 1
- CrossEntropyLoss
- Print numerical vs analytical gradient for specific weight indices
"""
import torch
import torch.nn as nn
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("PyTorch Gradient Reference Test")
print("=" * 60)

# Create simple linear layer
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

model = SimpleLinear()

# Test batch - batch size 1 to match C++ test
batch_size = 1
inputs = torch.rand(batch_size, 784)
targets = torch.randint(0, 10, (batch_size,))

print(f"\nInput shape: {inputs.shape}")
print(f"Target: {targets.tolist()}")

# Get weight reference
weight = model.fc.weight  # Shape: [10, 784]
print(f"Weight shape: {weight.shape}")

# Forward
logits = model(inputs)
print(f"Logits shape: {logits.shape}")
print(f"Logits: {logits.detach().numpy()}")

# Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"\nLoss: {loss.item()}")

# Backward
loss.backward()

# Check gradient shape
grad = model.fc.weight.grad
print(f"\nGradient shape: {grad.shape}")

# Check specific indices (matching C++ test)
indices_to_check = [0, 1, 2, 784, 785, 1568, 2352, 3136, 3920, 7839]
print(f"\nAnalytical gradients at specific indices:")
for idx in indices_to_check:
    row = idx // 784
    col = idx % 784
    print(f"  grad[{row}, {col}] (flat idx {idx}): {grad[row, col].item():.8f}")

# Now compute numerical gradient for verification
print("\n" + "=" * 60)
print("Numerical Gradient Check")
print("=" * 60)

eps = 1e-4
w_ptr = model.fc.weight.data

for idx in indices_to_check[:5]:  # Just check first 5
    row = idx // 784
    col = idx % 784
    orig = w_ptr[row, col].item()

    # f(w + eps)
    w_ptr[row, col] = orig + eps
    logits_plus = model(inputs)
    loss_plus = criterion(logits_plus, targets).item()

    # f(w - eps)
    w_ptr[row, col] = orig - eps
    logits_minus = model(inputs)
    loss_minus = criterion(logits_minus, targets).item()

    # Restore
    w_ptr[row, col] = orig

    numerical = (loss_plus - loss_minus) / (2 * eps)
    analytical = grad[row, col].item()
    rel_error = abs(numerical - analytical) / (abs(numerical) + abs(analytical) + 1e-8)

    status = "OK" if rel_error < 0.01 else "MISMATCH!"
    print(f"  w[{row},{col}]: numerical={numerical:.8f} analytical={analytical:.8f} rel_err={rel_error:.2e} {status}")

# Additional test: verify gradient formula
# For CrossEntropyLoss, grad_logits = softmax - one_hot
# Then grad_W = grad_logits.T @ inputs
print("\n" + "=" * 60)
print("Gradient Formula Verification")
print("=" * 60)

# Recompute with fresh model
model2 = SimpleLinear()
model2.fc.weight.data.copy_(model.fc.weight.data)
model2.fc.bias.data.copy_(model.fc.bias.data)
model2.zero_grad()

logits2 = model2(inputs)
loss2 = criterion(logits2, targets)
loss2.backward()

# Manual gradient computation
softmax = torch.softmax(logits2.detach(), dim=1)  # [1, 10]
one_hot = torch.zeros_like(softmax)
one_hot[0, targets[0]] = 1.0
grad_logits = (softmax - one_hot) / batch_size  # [1, 10]

print(f"Softmax: {softmax.numpy()}")
print(f"One-hot: {one_hot.numpy()}")
print(f"Grad logits (softmax - one_hot) / N: {grad_logits.numpy()}")

# For y = x @ W.T, grad_W = grad_y.T @ x
# grad_W[i,j] = sum_k grad_y[k,i] * x[k,j]
manual_grad_w = grad_logits.T @ inputs  # [10, 1] @ [1, 784] = [10, 784]
print(f"\nManual grad_W shape: {manual_grad_w.shape}")
print(f"PyTorch grad_W shape: {model2.fc.weight.grad.shape}")

# Compare
diff = (model2.fc.weight.grad - manual_grad_w).abs().max()
print(f"Max difference between PyTorch and manual: {diff.item():.2e}")

if diff < 1e-5:
    print("*** Formula matches! ***")
else:
    print("*** Formula MISMATCH! ***")

# Print first few values of both
print(f"\nFirst 5 values:")
print(f"  PyTorch: {model2.fc.weight.grad.flatten()[:5].numpy()}")
print(f"  Manual:  {manual_grad_w.flatten()[:5].numpy()}")

# Test with batch_size > 1
print("\n" + "=" * 60)
print("Test with batch_size=4")
print("=" * 60)

model3 = SimpleLinear()
inputs3 = torch.rand(4, 784)
targets3 = torch.tensor([3, 7, 1, 9])

logits3 = model3(inputs3)
loss3 = criterion(logits3, targets3)
loss3.backward()

# Manual
softmax3 = torch.softmax(logits3.detach(), dim=1)
one_hot3 = torch.zeros_like(softmax3)
for i in range(4):
    one_hot3[i, targets3[i]] = 1.0
grad_logits3 = (softmax3 - one_hot3) / 4  # Mean reduction

manual_grad_w3 = grad_logits3.T @ inputs3

diff3 = (model3.fc.weight.grad - manual_grad_w3).abs().max()
print(f"Max difference with batch_size=4: {diff3.item():.2e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
