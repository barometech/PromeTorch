"""
Test 2-layer MLP gradient flow in PyTorch.
This will help us understand what gradients should look like.
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("2-Layer MLP Gradient Test")
print("=" * 60)

# Create a simple 2-layer MLP: 784 -> 128 -> 10 with ReLU
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()

# Print initial weight stats
print(f"\nInitial weights:")
print(f"  fc1.weight: shape={model.fc1.weight.shape}, norm={model.fc1.weight.norm().item():.4f}")
print(f"  fc1.bias: shape={model.fc1.bias.shape}, norm={model.fc1.bias.norm().item():.4f}")
print(f"  fc2.weight: shape={model.fc2.weight.shape}, norm={model.fc2.weight.norm().item():.4f}")
print(f"  fc2.bias: shape={model.fc2.bias.shape}, norm={model.fc2.bias.norm().item():.4f}")

# Create simple input
batch_size = 4
inputs = torch.randn(batch_size, 784) * 0.5  # Moderate magnitude
targets = torch.tensor([3, 7, 1, 9])  # Class labels

print(f"\nInput stats: mean={inputs.mean():.4f}, std={inputs.std():.4f}")

# Forward pass
logits = model(inputs)
print(f"\nLogits shape: {logits.shape}")
print(f"Logits (first sample): {logits[0].detach().numpy()}")

# Compute loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"\nLoss: {loss.item():.4f}")

# Backward
loss.backward()

# Print gradients
print(f"\nGradients after backward:")
print(f"  fc1.weight.grad: norm={model.fc1.weight.grad.norm().item():.6f}")
print(f"  fc1.bias.grad: norm={model.fc1.bias.grad.norm().item():.6f}")
print(f"  fc2.weight.grad: norm={model.fc2.weight.grad.norm().item():.6f}")
print(f"  fc2.bias.grad: norm={model.fc2.bias.grad.norm().item():.6f}")

# Print first few gradient values
print(f"\nFirst 5 values of fc2.weight.grad:")
print(f"  {model.fc2.weight.grad[0, :5].numpy()}")

print(f"\nFirst 5 values of fc1.weight.grad:")
print(f"  {model.fc1.weight.grad[0, :5].numpy()}")

# Verify gradient flow through ReLU
# The intermediate output after ReLU should have some zeros
with torch.no_grad():
    h1 = model.fc1(inputs)
    h1_relu = model.relu(h1)
    num_zeros = (h1_relu == 0).sum().item()
    total_elements = h1_relu.numel()
    print(f"\nReLU sparsity: {num_zeros}/{total_elements} = {100*num_zeros/total_elements:.1f}% zeros")

# Now test a single SGD step
print("\n" + "=" * 60)
print("Single SGD Step Test")
print("=" * 60)

# Compute loss before step
model2 = SimpleMLP()
model2.load_state_dict(model.state_dict())  # Copy weights
for p in model2.parameters():
    if p.grad is not None:
        p.grad.zero_()

logits2 = model2(inputs)
loss_before = criterion(logits2, targets).item()
print(f"Loss before step: {loss_before:.4f}")

# Compute gradients
loss2 = criterion(logits2, targets)
loss2.backward()

# Manual SGD step with lr=0.01
lr = 0.01
with torch.no_grad():
    for p in model2.parameters():
        p.data -= lr * p.grad

# Compute loss after step
logits3 = model2(inputs)
loss_after = criterion(logits3, targets).item()
print(f"Loss after step (lr={lr}): {loss_after:.4f}")

if loss_after < loss_before:
    print(f"*** PASS: Loss decreased by {loss_before - loss_after:.4f} ***")
else:
    print(f"*** FAIL: Loss INCREASED by {loss_after - loss_before:.4f} ***")

# Now test with smaller lr (like PromeTorch default)
print("\n" + "=" * 60)
print("SGD Step with lr=0.001")
print("=" * 60)

model3 = SimpleMLP()
model3.load_state_dict(model.state_dict())
for p in model3.parameters():
    if p.grad is not None:
        p.grad.zero_()

logits3 = model3(inputs)
loss_before_3 = criterion(logits3, targets).item()
print(f"Loss before step: {loss_before_3:.4f}")

loss3 = criterion(logits3, targets)
loss3.backward()

lr = 0.001
with torch.no_grad():
    for p in model3.parameters():
        p.data -= lr * p.grad

logits4 = model3(inputs)
loss_after_3 = criterion(logits4, targets).item()
print(f"Loss after step (lr={lr}): {loss_after_3:.4f}")
print(f"Loss change: {loss_before_3 - loss_after_3:.6f}")

# Test gradient formula
print("\n" + "=" * 60)
print("Gradient Formula Verification")
print("=" * 60)

# For the last layer (fc2): y = fc2(relu(fc1(x)))
# grad_fc2_weight = grad_logits.T @ h1_relu
# where grad_logits = (softmax - one_hot) / batch_size

with torch.no_grad():
    h1_relu_data = model.relu(model.fc1(inputs))  # [4, 128]
    softmax = torch.softmax(logits.detach(), dim=1)  # [4, 10]
    one_hot = torch.zeros_like(softmax)
    for i in range(batch_size):
        one_hot[i, targets[i]] = 1.0
    grad_logits = (softmax - one_hot) / batch_size  # [4, 10]

    # grad_fc2_weight = grad_logits.T @ h1_relu
    manual_grad_fc2 = grad_logits.T @ h1_relu_data  # [10, 4] @ [4, 128] = [10, 128]

    print(f"Manual fc2.weight.grad norm: {manual_grad_fc2.norm().item():.6f}")
    print(f"PyTorch fc2.weight.grad norm: {model.fc2.weight.grad.norm().item():.6f}")
    diff = (manual_grad_fc2 - model.fc2.weight.grad).abs().max().item()
    print(f"Max difference: {diff:.2e}")

# For fc1: grad_fc1_weight = (grad_h1_relu * relu_mask).T @ x
# where grad_h1_relu = grad_logits @ fc2.weight
# and relu_mask = (h1 > 0)

with torch.no_grad():
    h1_pre_relu = model.fc1(inputs)  # [4, 128]
    relu_mask = (h1_pre_relu > 0).float()  # [4, 128]

    # grad_h1_relu = grad_logits @ fc2.weight = [4, 10] @ [10, 128] = [4, 128]
    grad_h1_relu = grad_logits @ model.fc2.weight.data  # [4, 128]

    # grad_h1 = grad_h1_relu * relu_mask
    grad_h1 = grad_h1_relu * relu_mask  # [4, 128]

    # grad_fc1_weight = grad_h1.T @ x = [128, 4] @ [4, 784] = [128, 784]
    manual_grad_fc1 = grad_h1.T @ inputs  # [128, 784]

    print(f"\nManual fc1.weight.grad norm: {manual_grad_fc1.norm().item():.6f}")
    print(f"PyTorch fc1.weight.grad norm: {model.fc1.weight.grad.norm().item():.6f}")
    diff = (manual_grad_fc1 - model.fc1.weight.grad).abs().max().item()
    print(f"Max difference: {diff:.2e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
