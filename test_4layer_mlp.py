"""
Test 4-layer MLP (same as PromeTorch) in PyTorch.
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("4-Layer MLP Gradient Test (like PromeTorch)")
print("=" * 60)

# Create 4-layer MLP: 784 -> 512 -> 256 -> 128 -> 10
class MLP4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

model = MLP4Layer()

# Print initial weight stats
print(f"\nInitial weights:")
print(f"  fc1.weight: norm={model.fc1.weight.norm().item():.4f}")
print(f"  fc2.weight: norm={model.fc2.weight.norm().item():.4f}")
print(f"  fc3.weight: norm={model.fc3.weight.norm().item():.4f}")
print(f"  fc4.weight: norm={model.fc4.weight.norm().item():.4f}")

# Create test batch
batch_size = 64
inputs = torch.randn(batch_size, 784) * 0.5
targets = torch.randint(0, 10, (batch_size,))

# Forward
logits = model(inputs)
print(f"\nLogits shape: {logits.shape}")

# Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

# Print gradients
print(f"\nGradients:")
print(f"  fc1.weight.grad: norm={model.fc1.weight.grad.norm().item():.6f}")
print(f"  fc1.bias.grad: norm={model.fc1.bias.grad.norm().item():.6f}")
print(f"  fc2.weight.grad: norm={model.fc2.weight.grad.norm().item():.6f}")
print(f"  fc2.bias.grad: norm={model.fc2.bias.grad.norm().item():.6f}")
print(f"  fc3.weight.grad: norm={model.fc3.weight.grad.norm().item():.6f}")
print(f"  fc3.bias.grad: norm={model.fc3.bias.grad.norm().item():.6f}")
print(f"  fc4.weight.grad: norm={model.fc4.weight.grad.norm().item():.6f}")
print(f"  fc4.bias.grad: norm={model.fc4.bias.grad.norm().item():.6f}")

# Training test with lr=0.001
print("\n" + "=" * 60)
print("Training Test: 3 epochs, lr=0.001")
print("=" * 60)

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Reset model
model = MLP4Layer()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    train_acc = 100. * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    
    # Test
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 784)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
    test_acc = 100. * test_correct / len(test_loader.dataset)
    
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
