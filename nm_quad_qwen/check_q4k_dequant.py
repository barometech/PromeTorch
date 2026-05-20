#!/usr/bin/env python3
"""Compare scalar vs vectorized Q4_K dequant on real Wq row."""
import sys
import numpy as np
sys.path.insert(0, "/home/elbrus-user/qwen/v1")
from qwen_embed_lookup import parse_gguf_tensor_table
from qwen_q4k_dequant import dequant_q4k_block as scalar_block
from qwen_dequant_np import dequant_q4k_rows_np

path = sys.argv[1]
tt, base = parse_gguf_tensor_table(path)
for n, dims, t, off in tt:
    if n == "blk.0.attn_q.weight":
        K = dims[0]; M = dims[1]
        bs = 144; bpr = K // 256
        row_bytes = bpr * bs
        # row 0
        with open(path, "rb") as f:
            f.seek(base + off)
            raw = f.read(row_bytes)
        scalar = []
        for b in range(bpr):
            scalar.extend(scalar_block(raw[b*bs:(b+1)*bs]))
        vec = dequant_q4k_rows_np(raw, 1, K)[0]
        diff = np.abs(np.array(scalar) - vec).max()
        print(f"attn_q row 0: K={K}, scalar L2={np.linalg.norm(scalar):.4f}, "
              f"vec L2={np.linalg.norm(vec):.4f}, max_diff={diff:.6f}")
        print(f"  scalar[:8]={scalar[:8]}")
        print(f"  vec[:8]={vec[:8].tolist()}")
        # Check rows 10, 100, 1000 too
        for ri in [10, 100, 1000]:
            with open(path, "rb") as f:
                f.seek(base + off + ri * row_bytes)
                raw = f.read(row_bytes)
            scalar = []
            for b in range(bpr): scalar.extend(scalar_block(raw[b*bs:(b+1)*bs]))
            vec = dequant_q4k_rows_np(raw, 1, K)[0]
            diff = np.abs(np.array(scalar) - vec).max()
            print(f"attn_q row {ri}: scalar L2={np.linalg.norm(scalar):.4f}, "
                  f"vec L2={np.linalg.norm(vec):.4f}, max_diff={diff:.6f}")
        break
