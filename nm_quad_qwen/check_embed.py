#!/usr/bin/env python3
"""Sanity check embedding lookup для qwen3-4b.

Token 'Once' = 12522. Lookup raw embedding row, compute L2.
Expected: для qwen3 raw embedding L2 ~ 1-3 (normalized scale).
"""
import sys
import numpy as np
from qwen_embed_lookup import parse_gguf_tensor_table, lookup_embedding
from qwen_dequant_np import dequant_q6k_rows_np
from qwen_layer_full import K_DIM, rmsnorm


def main():
    path = sys.argv[1]
    tt, base = parse_gguf_tensor_table(path)
    target = None
    for n, dims, t, off in tt:
        if n == "token_embd.weight":
            target = (base + off, dims, t)
            break
    off_t, dims_t, ttype_t = target
    print(f"[token_embd] dims={dims_t} ttype={ttype_t}")

    bs = 210
    bpr = K_DIM // 256
    row_bytes = bpr * bs

    f = open(path, "rb")
    for tid, name in [(12522, "Once"), (1052, "Ġthere"), (9707, "Hello"), (264, "Ġa")]:
        # vectorized lookup
        v = lookup_embedding(path, tid)
        v_arr = np.array(v, dtype=np.float32)
        # raw via dequant_q6k_rows_np (1 row)
        f.seek(off_t + tid * row_bytes)
        raw = f.read(row_bytes)
        vec = dequant_q6k_rows_np(raw, 1, K_DIM)[0]
        diff = np.abs(v_arr - vec).max()
        print(f"  id={tid:6d} ({name}) lookup L2={np.linalg.norm(v_arr):.4f} "
              f"vec L2={np.linalg.norm(vec):.4f} diff_max={diff:.6f} "
              f"vec[:4]={vec[:4].tolist()}")
    f.close()


if __name__ == "__main__":
    main()
