"""
Compare EXACT gradient values between PyTorch and PromeTorch
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(12345)
np.random.seed(12345)

print("=" * 60)
print("EXACT Gradient Values Comparison")
print("=" * 60)

# Create deterministic input
input_data = torch.randn(4, 784)  # Small batch for easy comparison
targets = torch.tensor([3, 7, 1, 9])

print(f"\nInput (first 5 elements): {input_data[0, :5].tolist()}")
print(f"Targets: {targets.tolist()}")

# Simple 2-layer MLP for easier debugging
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

# Print initial weights (for reproducibility check)
print(f"\nfc1.weight[0, :5]: {model.fc1.weight[0, :5].tolist()}")
print(f"fc1.bias[:5]: {model.fc1.bias[:5].tolist()}")
print(f"fc2.weight[0, :5]: {model.fc2.weight[0, :5].tolist()}")

# Forward
logits = model(input_data)
print(f"\nLogits shape: {logits.shape}")
print(f"Logits[0]: {logits[0].detach().tolist()}")

# Softmax probabilities
probs = torch.softmax(logits, dim=1)
print(f"Softmax probs[0]: {probs[0].detach().tolist()}")

# Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"\nLoss: {loss.item():.6f}")

# Backward
loss.backward()

# Print gradient details
print(f"\n" + "=" * 60)
print("GRADIENTS")
print("=" * 60)

print(f"\nfc1.weight.grad shape: {model.fc1.weight.grad.shape}")
print(f"fc1.weight.grad norm: {model.fc1.weight.grad.norm().item():.6f}")
print(f"fc1.weight.grad[0, :5]: {model.fc1.weight.grad[0, :5].tolist()}")
print(f"fc1.weight.grad[0, :5] (formatted): [{', '.join([f'{x:.8f}' for x in model.fc1.weight.grad[0, :5].tolist()])}]")

print(f"\nfc1.bias.grad shape: {model.fc1.bias.grad.shape}")
print(f"fc1.bias.grad[:5]: {model.fc1.bias.grad[:5].tolist()}")

print(f"\nfc2.weight.grad shape: {model.fc2.weight.grad.shape}")
print(f"fc2.weight.grad norm: {model.fc2.weight.grad.norm().item():.6f}")
print(f"fc2.weight.grad[0, :5]: {model.fc2.weight.grad[0, :5].tolist()}")

print(f"\nfc2.bias.grad shape: {model.fc2.bias.grad.shape}")
print(f"fc2.bias.grad: {model.fc2.bias.grad.tolist()}")

# Manual gradient computation for verification
print(f"\n" + "=" * 60)
print("MANUAL GRADIENT COMPUTATION (for verification)")
print("=" * 60)

with torch.no_grad():
    # h1 = fc1(x)
    h1 = input_data @ model.fc1.weight.t() + model.fc1.bias
    print(f"h1[0, :5]: {h1[0, :5].tolist()}")

    # h1_relu = relu(h1)
    h1_relu = torch.relu(h1)
    print(f"h1_relu[0, :5]: {h1_relu[0, :5].tolist()}")

    # logits = fc2(h1_relu)
    logits2 = h1_relu @ model.fc2.weight.t() + model.fc2.bias
    print(f"logits2[0]: {logits2[0].tolist()}")

    # grad_logits = (softmax - one_hot) / batch_size
    probs2 = torch.softmax(logits2, dim=1)
    one_hot = torch.zeros_like(probs2)
    for i in range(4):
        one_hot[i, targets[i]] = 1.0
    grad_logits = (probs2 - one_hot) / 4  # batch_size = 4
    print(f"grad_logits[0]: {grad_logits[0].tolist()}")

    # grad_fc2_weight = grad_logits.T @ h1_relu
    # shape: [10, 4] @ [4, 128] = [10, 128]
    manual_grad_fc2_w = grad_logits.t() @ h1_relu
    print(f"manual fc2.weight.grad[0, :5]: {manual_grad_fc2_w[0, :5].tolist()}")
    print(f"pytorch fc2.weight.grad[0, :5]: {model.fc2.weight.grad[0, :5].tolist()}")
    diff = (manual_grad_fc2_w - model.fc2.weight.grad).abs().max().item()
    print(f"Max diff: {diff:.2e}")

    # grad_fc2_bias = grad_logits.sum(dim=0)
    manual_grad_fc2_b = grad_logits.sum(dim=0)
    print(f"\nmanual fc2.bias.grad: {manual_grad_fc2_b.tolist()}")
    print(f"pytorch fc2.bias.grad: {model.fc2.bias.grad.tolist()}")

    # grad_h1_relu = grad_logits @ fc2.weight
    # shape: [4, 10] @ [10, 128] = [4, 128]
    grad_h1_relu = grad_logits @ model.fc2.weight
    print(f"\ngrad_h1_relu[0, :5]: {grad_h1_relu[0, :5].tolist()}")

    # grad_h1 = grad_h1_relu * (h1 > 0)
    relu_mask = (h1 > 0).float()
    grad_h1 = grad_h1_relu * relu_mask
    print(f"relu_mask[0, :5]: {relu_mask[0, :5].tolist()}")
    print(f"grad_h1[0, :5]: {grad_h1[0, :5].tolist()}")

    # grad_fc1_weight = grad_h1.T @ input_data
    # shape: [128, 4] @ [4, 784] = [128, 784]
    manual_grad_fc1_w = grad_h1.t() @ input_data
    print(f"\nmanual fc1.weight.grad[0, :5]: {manual_grad_fc1_w[0, :5].tolist()}")
    print(f"pytorch fc1.weight.grad[0, :5]: {model.fc1.weight.grad[0, :5].tolist()}")
    diff = (manual_grad_fc1_w - model.fc1.weight.grad).abs().max().item()
    print(f"Max diff: {diff:.2e}")

print(f"\n" + "=" * 60)
print("VALUES FOR PROMETHORCH COMPARISON")
print("=" * 60)
print(f"\nTo match in PromeTorch with same seed and input:")
print(f"  - fc2.weight.grad[0, 0] should be: {model.fc2.weight.grad[0, 0].item():.8f}")
print(f"  - fc1.weight.grad[0, 0] should be: {model.fc1.weight.grad[0, 0].item():.8f}")
print(f"  - Loss should be: {loss.item():.6f}")
