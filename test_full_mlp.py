"""
Test full 4-layer MNIST MLP gradient flow in PyTorch.
Matches exactly the architecture in train_mnist_mlp.cpp.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

torch.manual_seed(42)

print("=" * 60)
print("Full MNIST MLP Training Test (PyTorch)")
print("=" * 60)

# Model matching MNISTMLP from C++
class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = MNISTMLP()

# Print weight norms (should match PromeTorch)
print("\nInitial weight norms:")
print(f"  fc1.weight: {model.fc1.weight.norm().item():.4f}")
print(f"  fc2.weight: {model.fc2.weight.norm().item():.4f}")
print(f"  fc3.weight: {model.fc3.weight.norm().item():.4f}")
print(f"  fc4.weight: {model.fc4.weight.norm().item():.4f}")

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data/mnist', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test with SGD (no momentum) - should match PromeTorch
print("\n" + "=" * 60)
print("Training with SGD (no momentum), lr=0.001")
print("=" * 60)

model = MNISTMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)

# Train for first 10 batches with debug output
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= 10:
        break

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Print gradient norms (matching PromeTorch debug output)
    print(f"\n[DBG] batch={batch_idx}")
    print(f"  fc1.w=(g:{model.fc1.weight.grad.norm().item():.5f},w:{model.fc1.weight.norm().item():.5f})")
    print(f"  fc1.b=(g:{model.fc1.bias.grad.norm().item():.5f},w:{model.fc1.bias.norm().item():.5f})")
    print(f"  fc2.w=(g:{model.fc2.weight.grad.norm().item():.5f},w:{model.fc2.weight.norm().item():.5f})")
    print(f"  fc2.b=(g:{model.fc2.bias.grad.norm().item():.5f},w:{model.fc2.bias.norm().item():.5f})")
    print(f"  fc3.w=(g:{model.fc3.weight.grad.norm().item():.5f},w:{model.fc3.weight.norm().item():.5f})")
    print(f"  fc3.b=(g:{model.fc3.bias.grad.norm().item():.5f},w:{model.fc3.bias.norm().item():.5f})")
    print(f"  fc4.w=(g:{model.fc4.weight.grad.norm().item():.5f},w:{model.fc4.weight.norm().item():.5f})")
    print(f"  fc4.b=(g:{model.fc4.bias.grad.norm().item():.5f},w:{model.fc4.bias.norm().item():.5f})")

    # Total grad norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"[CLIP] total_grad_norm={total_norm:.5f}")

    # Accuracy
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    acc = 100 * correct / len(target)
    print(f"[LOSS] batch={batch_idx} loss={loss.item():.5f} acc={acc:.2f}%")

    optimizer.step()

# Continue training for full epoch
print("\n" + "=" * 60)
print("Training full epoch...")
print("=" * 60)

model = MNISTMLP()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
model.train()

train_loss = 0
correct = 0
total = 0

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    pred = output.argmax(dim=1)
    correct += (pred == target).sum().item()
    total += target.size(0)

    if batch_idx % 100 == 0:
        print(f"  Batch {batch_idx}/{len(train_loader)}: loss={train_loss/(batch_idx+1):.4f}, acc={100*correct/total:.2f}%")

print(f"\nEpoch 1: Loss={train_loss/len(train_loader):.4f}, Train Acc={100*correct/total:.2f}%")

# Test
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += (pred == target).sum().item()
        test_total += target.size(0)

print(f"Test Accuracy: {100*test_correct/test_total:.2f}%")

print("\n" + "=" * 60)
print("Expected: SGD no momentum should achieve ~49% test accuracy")
print("If PromeTorch also gets ~49%, the autograd is working correctly")
print("=" * 60)
