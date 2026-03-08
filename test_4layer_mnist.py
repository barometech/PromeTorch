"""
Test 4-layer MLP on MNIST with EXACT same parameters as PromeTorch:
- Architecture: 784 -> 512 -> 256 -> 128 -> 10 with ReLU
- LR: 0.001
- Batch size: 64
- SGD no momentum
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

torch.manual_seed(42)

print("=" * 60)
print("4-Layer MLP MNIST Training (PyTorch)")
print("Architecture: 784 -> 512 -> 256 -> 128 -> 10")
print("=" * 60)

# Model - EXACT same as PromeTorch MNISTMLP
class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

model = MNISTMLP()

# Print weight norms (for comparison with PromeTorch)
print(f"\nWeight norms after initialization:")
print(f"  fc1.weight: {model.fc1.weight.norm().item():.4f}, shape {list(model.fc1.weight.shape)}")
print(f"  fc2.weight: {model.fc2.weight.norm().item():.4f}, shape {list(model.fc2.weight.shape)}")
print(f"  fc3.weight: {model.fc3.weight.norm().item():.4f}, shape {list(model.fc3.weight.shape)}")
print(f"  fc4.weight: {model.fc4.weight.norm().item():.4f}, shape {list(model.fc4.weight.shape)}")

# Data - MNIST normalized same as PromeTorch
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data/mnist', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Optimizer - SGD with lr=0.001, no momentum (EXACT same as PromeTorch)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
criterion = nn.CrossEntropyLoss()

print(f"\nOptimizer: SGD(lr=0.001, momentum=0)")
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print(f"Batches per epoch: {len(train_loader)}")

# Training
def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Print first few batches for comparison
        if batch_idx < 5:
            print(f"\n[Batch {batch_idx}]")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  fc1.weight.grad norm: {model.fc1.weight.grad.norm().item():.6f}")
            print(f"  fc4.weight.grad norm: {model.fc4.weight.grad.norm().item():.6f}")

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    elapsed = time.time() - start_time
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Train Acc={accuracy:.2f}%, Time={elapsed:.1f}s")
    return avg_loss, accuracy

def test(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

print("\n" + "=" * 60)
print("Training for 1 epoch...")
print("=" * 60)

train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, 1)
test_acc = test(model, test_loader, criterion)

print(f"\nTest Accuracy: {test_acc:.2f}%")

print("\n" + "=" * 60)
print("Summary for comparison with PromeTorch:")
print("=" * 60)
print(f"After 1 epoch:")
print(f"  Train Loss: {train_loss:.4f}")
print(f"  Train Acc: {train_acc:.2f}%")
print(f"  Test Acc: {test_acc:.2f}%")
print(f"\nPromeTorch shows:")
print(f"  Train Loss: 2.30 (flat)")
print(f"  Train Acc: ~11.8% (random)")
print(f"  Test Acc: ~11.6% (random)")
