"""
Single-step LSTM comparison: exact numerical values.
LSTMCell(1,32), one timestep, one sample.
Print intermediate values AND gradients.
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

# Create LSTMCell with known weights
cell = nn.LSTMCell(1, 32)
fc = nn.Linear(32, 2)

# Print weight shapes and first few values
print("=== Weights ===")
print(f"W_ih: {cell.weight_ih.shape}, first 5: {cell.weight_ih.data[:5, 0].tolist()}")
print(f"b_ih: {cell.bias_ih.shape}, first 5: {cell.bias_ih.data[:5].tolist()}")
print(f"W_hh: {cell.weight_hh.shape}, first 5: {cell.weight_hh.data[:5, 0].tolist()}")
print(f"b_hh: {cell.bias_hh.shape}, first 5: {cell.bias_hh.data[:5].tolist()}")
print(f"fc.W: {fc.weight.shape}, first 5: {fc.weight.data[0, :5].tolist()}")
print(f"fc.b: {fc.bias.shape}: {fc.bias.data.tolist()}")

# Simple input
x = torch.tensor([[0.5]])  # [1, 1]
h = torch.zeros(1, 32)
c = torch.zeros(1, 32)
target = torch.tensor([1])

# Forward through LSTM
h_new, c_new = cell(x, (h, c))
print(f"\n=== After 1 LSTM step ===")
print(f"h_new first 5: {h_new.data[0, :5].tolist()}")
print(f"c_new first 5: {c_new.data[0, :5].tolist()}")

# Forward through fc
logits = fc(h_new)
print(f"logits: {logits.data[0].tolist()}")

# Loss
loss = nn.CrossEntropyLoss()(logits, target)
print(f"loss: {loss.item():.8f}")

# Backward
loss.backward()

# Print all gradient norms
print(f"\n=== Gradient norms ===")
print(f"W_ih grad_norm: {cell.weight_ih.grad.norm().item():.8f}")
print(f"b_ih grad_norm: {cell.bias_ih.grad.norm().item():.8f}")
print(f"W_hh grad_norm: {cell.weight_hh.grad.norm().item():.8f}")
print(f"b_hh grad_norm: {cell.bias_hh.grad.norm().item():.8f}")
print(f"fc.W grad_norm: {fc.weight.grad.norm().item():.8f}")
print(f"fc.b grad_norm: {fc.bias.grad.norm().item():.8f}")

# Print actual gradient values for biases (small enough to see)
print(f"\n=== Actual b_ih gradient (first 16) ===")
print([f"{v:.6f}" for v in cell.bias_ih.grad[:16].tolist()])

# Also compute what the gates look like
with torch.no_grad():
    gates = x @ cell.weight_ih.T + cell.bias_ih + h @ cell.weight_hh.T + cell.bias_hh
    print(f"\n=== Raw gates (first 8 per gate) ===")
    H = 32
    print(f"i_raw: {gates[0, 0:8].tolist()}")
    print(f"f_raw: {gates[0, H:H+8].tolist()}")
    print(f"g_raw: {gates[0, 2*H:2*H+8].tolist()}")
    print(f"o_raw: {gates[0, 3*H:3*H+8].tolist()}")
    i = torch.sigmoid(gates[0, 0:H])
    f = torch.sigmoid(gates[0, H:2*H])
    g = torch.tanh(gates[0, 2*H:3*H])
    o = torch.sigmoid(gates[0, 3*H:4*H])
    print(f"\ni: {i[:8].tolist()}")
    print(f"f: {f[:8].tolist()}")
    print(f"g: {g[:8].tolist()}")
    print(f"o: {o[:8].tolist()}")
