"""
Minimal test to compare CrossEntropyLoss between PyTorch and PromeTorch
This will help identify if the issue is in the loss function or gradient flow
"""
import torch
import torch.nn as nn
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("Cross-Entropy Loss Comparison Test")
print("=" * 60)

# Create simple test case with known values
batch_size = 4
num_classes = 10

# Simple logits (before softmax)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
], requires_grad=True)

targets = torch.tensor([9, 0, 5, 9])  # Class labels

print(f"\nLogits shape: {logits.shape}")
print(f"Targets: {targets.tolist()}")

# Compute softmax manually
softmax = torch.softmax(logits, dim=1)
print(f"\nSoftmax[0] (should peak at idx 9): {softmax[0].detach().numpy()}")
print(f"Softmax sum per row: {softmax.sum(dim=1).detach().numpy()}")

# Compute cross entropy
criterion = nn.CrossEntropyLoss(reduction='mean')
loss = criterion(logits, targets)
print(f"\nCrossEntropyLoss (mean): {loss.item()}")

# Compute gradient
loss.backward()
print(f"\nGradient shape: {logits.grad.shape}")
print(f"Gradient row 0 (target=9):")
print(f"  {logits.grad[0].numpy()}")
print(f"  Expected: softmax - one_hot = {(softmax[0] - torch.zeros(10).scatter_(0, torch.tensor(9), 1.0)).detach().numpy()}")
print(f"  Divided by N=4: {((softmax[0] - torch.zeros(10).scatter_(0, torch.tensor(9), 1.0)) / 4).detach().numpy()}")

# Print the full gradient
print(f"\nFull gradient matrix:")
for i in range(batch_size):
    print(f"  Row {i} (target={targets[i].item()}): {logits.grad[i].numpy()}")

# Now test a simple linear layer gradient
print("\n" + "=" * 60)
print("Linear Layer Gradient Test")
print("=" * 60)

# Create simple linear layer
W = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
], requires_grad=True)  # [2, 3] - 2 output, 3 input

x = torch.tensor([[1.0, 2.0, 3.0]])  # [1, 3] - 1 sample, 3 features

# y = x @ W^T -> [1, 2]
y = torch.mm(x, W.t())
print(f"\nInput x: {x.numpy()}")
print(f"Weight W:\n{W.detach().numpy()}")
print(f"Output y = x @ W^T: {y.detach().numpy()}")

# Loss = sum(y) for simplicity
loss = y.sum()
loss.backward()

print(f"\nGradient of W (dL/dW):\n{W.grad.numpy()}")
print(f"Expected: outer(grad_y, x) = [1, 1]^T @ [1, 2, 3] =")
grad_y = torch.ones(2)
expected = torch.outer(grad_y, x.squeeze())
print(f"{expected.numpy()}")

# Now test the transpose gradient
print("\n" + "=" * 60)
print("Transpose Gradient Flow Test")
print("=" * 60)

W2 = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
], requires_grad=True)

x2 = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# y = x @ W^T using explicit transpose
W_t = W2.t()  # [3, 2]
y2 = torch.mm(x2, W_t)  # [1, 2]

loss2 = y2.sum()
loss2.backward()

print(f"W2.grad (via explicit transpose):\n{W2.grad.numpy()}")
print(f"x2.grad:\n{x2.grad.numpy()}")

print("\n" + "=" * 60)
print("This output can be compared with PromeTorch C++ test")
print("=" * 60)
