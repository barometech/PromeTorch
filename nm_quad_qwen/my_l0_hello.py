#!/usr/bin/env python3
"""My L0 forward для "Hello" only."""
import sys, numpy as np
sys.path.insert(0, "/home/paperclipdnb/qwen/v1")
from qwen_embed_lookup import lookup_embedding
from qwen_layer_full import qwen_full_layer

path = sys.argv[1]
emb = np.array(lookup_embedding(path, 9707), dtype=np.float32)
print(f"[emb] L2={np.linalg.norm(emb):.4f}")
print(f"  first 8: {emb[:8].tolist()}")

x = qwen_full_layer(path, 0, emb.copy(), 0)
print(f"\n[after my L0]")
print(f"L2={np.linalg.norm(x):.4f}")
print(f"first 8: {x[:8].tolist()}")

print(f"\n[HF reference for Hello after layer 0]:")
print(f"L2=19.80, first 8=[7.00, -2.61, 1.46, 0.71, 1.66, 0.42, -0.96, -0.02]")
