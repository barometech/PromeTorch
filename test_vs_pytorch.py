"""
Compare PromeTorch with PyTorch on MNIST
This will help identify where the problem is
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("PyTorch MNIST MLP Benchmark")
print("=" * 60)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data/mnist', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model - same as PromeTorch
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
criterion = nn.CrossEntropyLoss()

# Test different optimizers
optimizers = {
    'SGD (no momentum)': optim.SGD(model.parameters(), lr=0.001, momentum=0.0),
    'SGD (momentum=0.9)': optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
}

for opt_name, optimizer in optimizers.items():
    print(f"\n{'='*60}")
    print(f"Testing: {opt_name}")
    print("=" * 60)

    # Reset model
    model = MNISTMLP()
    if 'SGD' in opt_name and 'no momentum' in opt_name:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    elif 'SGD' in opt_name:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 1 epoch
    model.train()
    start = time.time()
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

    elapsed = time.time() - start
    print(f"\nEpoch 1: Loss={train_loss/len(train_loader):.4f}, Train Acc={100*correct/total:.2f}%, Time={elapsed:.1f}s")

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
print("Expected: SGD no momentum ~85%, SGD momentum ~95%, Adam ~97%")
print("If PromeTorch gets ~15%, the problem is in the framework")
print("=" * 60)
