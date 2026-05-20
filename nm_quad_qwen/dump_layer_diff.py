#!/usr/bin/env python3
"""Forward через L0 моего Python для prompt "Once" (id=12522) и сравнить
с llama.cpp reference hidden state."""
import sys
import numpy as np
sys.path.insert(0, "/home/paperclipdnb/qwen/v1")
from qwen_embed_lookup import parse_gguf_tensor_table, lookup_embedding
from qwen_layer_full import qwen_full_layer, K_DIM, rmsnorm

path = sys.argv[1]
tid = 12522  # 'Once'
emb = np.array(lookup_embedding(path, tid), dtype=np.float32)
print(f"[emb] tid={tid} L2={np.linalg.norm(emb):.4f}")
print(f"  first 8: {emb[:8].tolist()}")

x = emb.copy()
for L in range(6):
    x_new = qwen_full_layer(path, L, x, 0)
    print(f"[L{L}] L2(out)={np.linalg.norm(x_new):.4f}")
    print(f"  first 4: {x_new[:4].tolist()}")
    x = x_new

# Final output_norm + lm_head simulation
tt, base = parse_gguf_tensor_table(path)
by_name = {n: (base+off, dims, t) for n, dims, t, off in tt}
on_off = by_name["output_norm.weight"][0]
import struct
with open(path, "rb") as f:
    f.seek(on_off)
    on = np.frombuffer(f.read(K_DIM*4), dtype=np.float32).copy()
x_norm = rmsnorm(x, on)
print(f"\n[after L0-L5 + output_norm]")
print(f"L2={np.linalg.norm(x_norm):.4f}")
print(f"first 8: {x_norm[:8].tolist()}")
