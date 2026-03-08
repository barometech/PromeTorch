"""
Manual LSTM implementation matching PromeTorch's code exactly,
compared against PyTorch's LSTMCell.
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

# Create matching architecture
cell_pt = nn.LSTMCell(1, 32)
fc_pt = nn.Linear(32, 2)

# Our LSTM uses two Linear layers: ih(1->128) and hh(32->128)
ih = nn.Linear(1, 128)
hh = nn.Linear(32, 128)
fc = nn.Linear(32, 2)

# Copy PyTorch's LSTMCell weights into our two-Linear structure
with torch.no_grad():
    ih.weight.copy_(cell_pt.weight_ih)
    ih.bias.copy_(cell_pt.bias_ih)
    hh.weight.copy_(cell_pt.weight_hh)
    hh.bias.copy_(cell_pt.bias_hh)
    fc.weight.copy_(fc_pt.weight)
    fc.bias.copy_(fc_pt.bias)

# Test data: batch=4, seq_len=3, input_size=1
np.random.seed(42)
batch = 4
seq_len = 3
inputs_np = np.random.randn(batch, seq_len, 1).astype(np.float32)
sums = inputs_np.sum(axis=1).squeeze()
targets_np = (sums > 0).astype(np.int64)

x_pt = torch.from_numpy(inputs_np.copy())
x_manual = torch.from_numpy(inputs_np.copy())
target = torch.from_numpy(targets_np)

# === PyTorch LSTMCell ===
h_pt = torch.zeros(batch, 32)
c_pt = torch.zeros(batch, 32)
for t in range(seq_len):
    xt = x_pt[:, t, :]  # [batch, 1]
    h_pt, c_pt = cell_pt(xt, (h_pt, c_pt))
logits_pt = fc_pt(h_pt)
loss_pt = nn.CrossEntropyLoss()(logits_pt, target)
loss_pt.backward()

# === Manual (matching PromeTorch code) ===
h_man = torch.zeros(batch, 32)
c_man = torch.zeros(batch, 32)
for t in range(seq_len):
    xt = x_manual[:, t, :]  # [batch, 1] - this is a view!
    # gates = ih(xt) + hh(h_man)
    gates = ih(xt) + hh(h_man)  # [batch, 128]

    H = 32
    # Split gates - using narrow (like PromeTorch)
    i_gate = torch.sigmoid(gates.narrow(1, 0, H))
    f_gate = torch.sigmoid(gates.narrow(1, H, H))
    g_gate = torch.tanh(gates.narrow(1, 2*H, H))
    o_gate = torch.sigmoid(gates.narrow(1, 3*H, H))

    # LSTM equations
    c_man = f_gate * c_man + i_gate * g_gate
    h_man = o_gate * torch.tanh(c_man)

logits_man = fc(h_man)
loss_man = nn.CrossEntropyLoss()(logits_man, target)
loss_man.backward()

print("=== Comparison ===")
print(f"PyTorch loss:  {loss_pt.item():.8f}")
print(f"Manual loss:   {loss_man.item():.8f}")
print(f"Match: {abs(loss_pt.item() - loss_man.item()) < 1e-6}")

print(f"\nLogits PyTorch:  {logits_pt.data[0].tolist()}")
print(f"Logits Manual:   {logits_man.data[0].tolist()}")

print(f"\nh_final PyTorch first 5: {h_pt.data[0, :5].tolist()}")
print(f"h_final Manual  first 5: {h_man.data[0, :5].tolist()}")

print(f"\n=== Gradient comparison ===")
print(f"W_ih grad_norm: PT={cell_pt.weight_ih.grad.norm():.6f} Man={ih.weight.grad.norm():.6f}")
print(f"b_ih grad_norm: PT={cell_pt.bias_ih.grad.norm():.6f} Man={ih.bias.grad.norm():.6f}")
print(f"W_hh grad_norm: PT={cell_pt.weight_hh.grad.norm():.6f} Man={hh.weight.grad.norm():.6f}")
print(f"b_hh grad_norm: PT={cell_pt.bias_hh.grad.norm():.6f} Man={hh.bias.grad.norm():.6f}")
print(f"fc_W grad_norm: PT={fc_pt.weight.grad.norm():.6f} Man={fc.weight.grad.norm():.6f}")
print(f"fc_b grad_norm: PT={fc_pt.bias.grad.norm():.6f} Man={fc.bias.grad.norm():.6f}")

# Check individual gradient values
diff_wih = (cell_pt.weight_ih.grad - ih.weight.grad).abs().max().item()
diff_bih = (cell_pt.bias_ih.grad - ih.bias.grad).abs().max().item()
diff_whh = (cell_pt.weight_hh.grad - hh.weight.grad).abs().max().item()
diff_bhh = (cell_pt.bias_hh.grad - hh.bias.grad).abs().max().item()
print(f"\nMax abs diff W_ih: {diff_wih:.8f}")
print(f"Max abs diff b_ih: {diff_bih:.8f}")
print(f"Max abs diff W_hh: {diff_whh:.8f}")
print(f"Max abs diff b_hh: {diff_bhh:.8f}")

# Now train the manual model for 80 epochs
print("\n=== Training manual LSTM for 80 epochs ===")
optimizer = torch.optim.Adam([
    {'params': ih.parameters()},
    {'params': hh.parameters()},
    {'params': fc.parameters()},
], lr=0.005)

np.random.seed(42)
g_rng = np.random.RandomState(42)

for epoch in range(1, 81):
    inputs = g_rng.randn(64, 8, 1).astype(np.float32)
    sums = inputs.sum(axis=1).squeeze()
    targets = (sums > 0).astype(np.int64)

    x = torch.from_numpy(inputs)
    t = torch.from_numpy(targets)

    optimizer.zero_grad()
    h = torch.zeros(64, 32)
    c = torch.zeros(64, 32)
    for ts in range(8):
        xt = x[:, ts, :]
        gates = ih(xt) + hh(h)
        H = 32
        i_gate = torch.sigmoid(gates.narrow(1, 0, H))
        f_gate = torch.sigmoid(gates.narrow(1, H, H))
        g_gate = torch.tanh(gates.narrow(1, 2*H, H))
        o_gate = torch.sigmoid(gates.narrow(1, 3*H, H))
        c = f_gate * c + i_gate * g_gate
        h = o_gate * torch.tanh(c)

    logits = fc(h)
    loss = nn.CrossEntropyLoss()(logits, t)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        preds = logits.argmax(dim=1)
        acc = (preds == t).float().mean().item() * 100
        print(f"  Epoch {epoch} | Loss = {loss.item():.6f} | Acc = {acc:.1f}%")

preds = logits.argmax(dim=1)
acc = (preds == t).float().mean().item() * 100
print(f"\nFinal: loss={loss.item():.6f} acc={acc:.1f}%")
