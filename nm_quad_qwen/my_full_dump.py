#!/usr/bin/env python3
"""My Python 36-layer forward для single token "Hello", + output_norm.
Сравнить с llama.cpp embedding output."""
import sys
import numpy as np
sys.path.insert(0, "/home/paperclipdnb/qwen/v1")
from qwen_embed_lookup import parse_gguf_tensor_table, lookup_embedding
from qwen_layer_full import qwen_full_layer, K_DIM, rmsnorm, load_fp32

path = sys.argv[1]
tid = 9707  # 'Hello'
emb = np.array(lookup_embedding(path, tid), dtype=np.float32)
print(f"[emb] tid={tid} L2={np.linalg.norm(emb):.4f}")

x = emb.copy()
for L in range(36):
    x = qwen_full_layer(path, L, x, 0)

print(f"[after 36 layers] L2={np.linalg.norm(x):.4f}")

# output_norm
tt, base = parse_gguf_tensor_table(path)
by_name = {n: (base+off, dims, t) for n, dims, t, off in tt}
f = open(path, "rb")
on = load_fp32(f, by_name["output_norm.weight"][0], K_DIM)
f.close()
x_norm = rmsnorm(x, on)
print(f"[after output_norm] L2={np.linalg.norm(x_norm):.4f}")
print(f"first 8: {x_norm[:8].tolist()}")

print("\n[llama.cpp reference for Hello]:")
print("L2=113.32 first 8=[-0.012, 2.08, 4.02, -2.74, -0.69, -0.47, 1.90, 1.18]")
