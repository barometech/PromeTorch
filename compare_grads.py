"""
Compare gradients between PyTorch and PromeTorch
Using EXACT same architecture: 784 -> 128 -> 10
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("EXACT PyTorch 2-layer MLP Gradient Test")
print("Architecture: 784 -> 128 -> 10")
print("=" * 60)

# Create model
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

# Get weight norms after initialization
print(f"\nWeight norms after init (PyTorch default init):")
print(f"  fc1.weight: {model.fc1.weight.norm().item():.4f}, shape {list(model.fc1.weight.shape)}")
print(f"  fc1.bias: {model.fc1.bias.norm().item():.4f}")
print(f"  fc2.weight: {model.fc2.weight.norm().item():.4f}, shape {list(model.fc2.weight.shape)}")
print(f"  fc2.bias: {model.fc2.bias.norm().item():.4f}")

# Create batch
batch_size = 64
inputs = torch.randn(batch_size, 784)
targets = torch.randint(0, 10, (batch_size,))

print(f"\nInput: shape {list(inputs.shape)}, mean={inputs.mean():.4f}, std={inputs.std():.4f}")

# Forward
logits = model(inputs)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

# Print gradient norms
print(f"\nGradient norms after backward:")
print(f"  fc1.weight.grad: {model.fc1.weight.grad.norm().item():.6f}")
print(f"  fc1.bias.grad: {model.fc1.bias.grad.norm().item():.6f}")
print(f"  fc2.weight.grad: {model.fc2.weight.grad.norm().item():.6f}")
print(f"  fc2.bias.grad: {model.fc2.bias.grad.norm().item():.6f}")

# Total gradient norm
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm().item() ** 2
total_norm = total_norm ** 0.5
print(f"\nTotal gradient norm: {total_norm:.6f}")

# SGD step test
print(f"\n" + "=" * 60)
print("SGD Step Test (lr=0.001)")
print("=" * 60)

model2 = SimpleMLP()
model2.load_state_dict(model.state_dict())
for p in model2.parameters():
    if p.grad is not None:
        p.grad.zero_()

logits2 = model2(inputs)
loss_before = criterion(logits2, targets).item()
loss2 = criterion(logits2, targets)
loss2.backward()

# Manual SGD
with torch.no_grad():
    for p in model2.parameters():
        p.data -= 0.001 * p.grad

logits3 = model2(inputs)
loss_after = criterion(logits3, targets).item()
print(f"Loss before: {loss_before:.4f}")
print(f"Loss after: {loss_after:.4f}")
print(f"Decrease: {loss_before - loss_after:.6f}")

# Check weight change
print(f"\nWeight change after 1 step:")
old_w = model.fc1.weight.data.clone()
new_w = model2.fc1.weight.data
diff = (new_w - old_w).norm().item()
print(f"  fc1.weight norm change: {diff:.6f}")
print(f"  Expected: lr * grad_norm = 0.001 * {model.fc1.weight.grad.norm().item():.4f} = {0.001 * model.fc1.weight.grad.norm().item():.6f}")

print("\n" + "=" * 60)
print("For PromeTorch comparison, values to match:")
print("=" * 60)
print(f"fc1.weight.grad norm should be: {model.fc1.weight.grad.norm().item():.6f}")
print(f"fc2.weight.grad norm should be: {model.fc2.weight.grad.norm().item():.6f}")
print(f"Total grad norm should be: {total_norm:.6f}")
print(f"Loss decrease should be: {loss_before - loss_after:.6f}")
