#!/usr/bin/env python3
"""Compare scalar vs vectorized Q6_K dequant on Wd (K=9728)."""
import sys, numpy as np
sys.path.insert(0, "/home/paperclipdnb/qwen/v1")
from qwen_embed_lookup import parse_gguf_tensor_table, dequant_q6k_block
from qwen_dequant_np import dequant_q6k_rows_np

path = sys.argv[1]
tt, base = parse_gguf_tensor_table(path)
for n, dims, t, off in tt:
    if n == "blk.0.ffn_down.weight":
        K = dims[0]; M = dims[1]
        bs = 210; bpr = K // 256
        row_bytes = bpr * bs
        print(f"ffn_down: K={K} M={M} bpr={bpr} row_bytes={row_bytes}")
        # row 0
        with open(path, "rb") as f:
            f.seek(base + off)
            raw = f.read(row_bytes)
        scalar = []
        for b in range(bpr):
            scalar.extend(dequant_q6k_block(raw[b*bs:(b+1)*bs]))
        vec = dequant_q6k_rows_np(raw, 1, K)[0]
        diff = np.abs(np.array(scalar) - vec).max()
        print(f"row 0: scalar L2={np.linalg.norm(scalar):.4f} vec L2={np.linalg.norm(vec):.4f} max_diff={diff:.6f}")
        # check diff position
        idx = np.argmax(np.abs(np.array(scalar) - vec))
        print(f"  max diff at idx={idx}, scalar={scalar[idx]} vec={vec[idx]}")
        # row 100
        with open(path, "rb") as f:
            f.seek(base + off + 100 * row_bytes)
            raw = f.read(row_bytes)
        scalar = []
        for b in range(bpr):
            scalar.extend(dequant_q6k_block(raw[b*bs:(b+1)*bs]))
        vec = dequant_q6k_rows_np(raw, 1, K)[0]
        diff = np.abs(np.array(scalar) - vec).max()
        print(f"row 100: scalar L2={np.linalg.norm(scalar):.4f} vec L2={np.linalg.norm(vec):.4f} max_diff={diff:.6f}")
        break
