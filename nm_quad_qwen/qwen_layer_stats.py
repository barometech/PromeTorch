#!/usr/bin/env python3
"""
qwen_layer_stats.py — print stats для всех weights одного Qwen3-4B layer.

Validates что все 11 tensors блока (attn_norm, attn_q/k/v/output,
attn_q_norm, attn_k_norm, ffn_norm, ffn_gate, ffn_up, ffn_down) могут
быть loaded и dequantized correctly. Sample row из каждого tensor.
"""

import sys
import math
import struct
from qwen_embed_lookup import parse_gguf_tensor_table, fp16_to_fp32
from qwen_embed_lookup import dequant_q6k_block
from qwen_q4k_dequant import dequant_q4k_block


def dequant_row(f, off: int, row_idx: int, K: int, ttype: int) -> list:
    if ttype == 0:  # F32
        f.seek(off + row_idx * K * 4)
        return list(struct.unpack(f"<{K}f", f.read(K * 4)))
    if ttype == 12:  # Q4_K
        block_size = 144
        bpr = K // 256
        row_bytes = bpr * block_size
        f.seek(off + row_idx * row_bytes)
        row = f.read(row_bytes)
        out = []
        for b in range(bpr):
            out.extend(dequant_q4k_block(row[b * block_size:(b + 1) * block_size]))
        return out
    if ttype == 14:  # Q6_K
        block_size = 210
        bpr = K // 256
        row_bytes = bpr * block_size
        f.seek(off + row_idx * row_bytes)
        row = f.read(row_bytes)
        out = []
        for b in range(bpr):
            out.extend(dequant_q6k_block(row[b * block_size:(b + 1) * block_size]))
        return out
    raise RuntimeError(f"unsupported type {ttype}")


def stats(vals: list) -> dict:
    n = len(vals)
    s = sum(vals)
    s2 = sum(v * v for v in vals)
    mn = min(vals)
    mx = max(vals)
    mean = s / n
    return {"n": n, "mean": mean, "std": math.sqrt(s2 / n - mean * mean) if (s2 / n - mean * mean) > 0 else 0,
            "min": mn, "max": mx, "L2": math.sqrt(s2)}


def main():
    if len(sys.argv) < 2:
        print("Usage: qwen_layer_stats.py <gguf_path> [layer_idx]")
        sys.exit(1)
    path = sys.argv[1]
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    tt, base = parse_gguf_tensor_table(path)
    targets = [
        f"blk.{L}.attn_norm.weight",
        f"blk.{L}.attn_q.weight",
        f"blk.{L}.attn_k.weight",
        f"blk.{L}.attn_v.weight",
        f"blk.{L}.attn_q_norm.weight",
        f"blk.{L}.attn_k_norm.weight",
        f"blk.{L}.attn_output.weight",
        f"blk.{L}.ffn_norm.weight",
        f"blk.{L}.ffn_gate.weight",
        f"blk.{L}.ffn_up.weight",
        f"blk.{L}.ffn_down.weight",
    ]
    by_name = {n: (n, base + off, dims, t) for (n, dims, t, off) in tt}

    print(f"=== Layer {L} weight statistics (sample row 0) ===")
    print(f"{'tensor':<35} {'dims':<18} {'type':<6} {'mean':>10} {'std':>10} {'L2':>10}")
    print(f"{'-'*35} {'-'*18} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    f = open(path, "rb")
    for name in targets:
        if name not in by_name:
            print(f"{name:<35} NOT FOUND")
            continue
        _, off, dims, ttype = by_name[name]
        if len(dims) == 1:
            K = dims[0]
            row = dequant_row(f, off, 0, K, ttype)
        else:
            K = dims[0]
            row = dequant_row(f, off, 0, K, ttype)
        s = stats(row)
        type_name = {0: "F32", 12: "Q4_K", 14: "Q6_K"}.get(ttype, str(ttype))
        dim_str = "x".join(str(d) for d in dims)
        print(f"{name:<35} {dim_str:<18} {type_name:<6} {s['mean']:>10.4f} {s['std']:>10.4f} {s['L2']:>10.4f}")
    f.close()


if __name__ == "__main__":
    main()
