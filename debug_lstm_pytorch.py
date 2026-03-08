"""
Compare LSTM gradient magnitudes with PromeTorch.
Same architecture: LSTMCell(1,32) + Linear(32,2), Adam lr=0.005
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
g_rng = np.random.RandomState(42)

class SeqClassLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(1, 32)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        batch = x.size(0)
        seq_len = x.size(1)
        h = torch.zeros(batch, 32)
        c = torch.zeros(batch, 32)
        for t in range(seq_len):
            xt = x[:, t, :]  # [batch, 1]
            h, c = self.cell(xt, (h, c))
        return self.fc(h)

model = SeqClassLSTM()
opt = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

seq_len = 8
batch = 64

for epoch in range(1, 4):
    inputs = g_rng.randn(batch, seq_len, 1).astype(np.float32)
    sums = inputs.sum(axis=1).squeeze()
    targets = (sums > 0).astype(np.int64)

    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)

    opt.zero_grad()
    logits = model(x)
    loss = criterion(logits, t)
    loss.backward()

    print(f"Epoch {epoch}: loss={loss.item():.6f}")
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            print(f"  {name} shape={list(p.shape)} grad_norm={gn:.6g}")
        else:
            print(f"  {name} NO GRAD")

    opt.step()

# Train to completion
for epoch in range(4, 81):
    inputs = g_rng.randn(batch, seq_len, 1).astype(np.float32)
    sums = inputs.sum(axis=1).squeeze()
    targets = (sums > 0).astype(np.int64)
    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)
    opt.zero_grad()
    logits = model(x)
    loss = criterion(logits, t)
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == t).float().mean().item() * 100
        print(f"Epoch {epoch}: loss={loss.item():.6f} acc={acc:.1f}%")

preds = logits.argmax(dim=1)
acc = (preds == t).float().mean().item() * 100
print(f"\nFinal: loss={loss.item():.6f} acc={acc:.1f}%")
