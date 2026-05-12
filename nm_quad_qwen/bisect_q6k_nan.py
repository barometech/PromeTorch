#!/usr/bin/env python3
"""Bisect: какой именно Q6_K тензор какого layer даёт NaN после vectorized dequant.

Для каждого L в [0..35] и tensor_name in (attn_v, ffn_down):
  1) загрузить полностью
  2) vectorized dequant
  3) np.any(~np.isfinite(W)) → если True, выяснить какие строки/блоки
"""
import sys
import numpy as np
from qwen_embed_lookup import parse_gguf_tensor_table
from qwen_dequant_np import dequant_q6k_rows_np


def check_tensor(path, tt, base, name):
    target = None
    for (n, dims, t, off) in tt:
        if n == name:
            target = (base + off, dims, t)
            break
    if target is None:
        return None
    off, dims, ttype = target
    if ttype != 14:
        return ("not_q6k", ttype)
    K, M = dims
    bs = 210
    bpr = K // 256
    row_bytes = bpr * bs
    f = open(path, "rb")
    f.seek(off)
    raw = f.read(M * row_bytes)
    f.close()
    W = dequant_q6k_rows_np(raw, M, K)
    bad = ~np.isfinite(W)
    if not bad.any():
        return ("ok", float(np.max(np.abs(W))))
    # find first bad row
    rows_bad = bad.any(axis=1)
    first_row = int(np.argmax(rows_bad))
    cols_in_row = bad[first_row]
    first_col = int(np.argmax(cols_in_row))
    block_idx = first_col // 256
    in_block = first_col % 256
    # Inspect raw bytes for that block
    block_off = first_row * row_bytes + block_idx * bs
    block_bytes = raw[block_off:block_off + bs]
    d_bits = block_bytes[208] | (block_bytes[209] << 8)
    scales = np.frombuffer(block_bytes[192:208], dtype=np.int8).tolist()
    nan_count = int(bad.sum())
    return ("nan", first_row, first_col, block_idx, in_block,
            d_bits, scales, nan_count)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/<user>/gguf/qwen3-4b-q4km.gguf"
    tt, base = parse_gguf_tensor_table(path)
    print(f"[gguf] {path}")
    for L in range(36):
        for tname in (f"blk.{L}.attn_v.weight", f"blk.{L}.ffn_down.weight"):
            r = check_tensor(path, tt, base, tname)
            if r is None:
                continue
            tag = r[0]
            if tag == "ok":
                print(f"  L{L:2d} {tname.split('.')[-2]:10s}: OK  max={r[1]:.4f}", flush=True)
            elif tag == "not_q6k":
                print(f"  L{L:2d} {tname.split('.')[-2]:10s}: not_q6k ttype={r[1]}", flush=True)
            else:
                _, row, col, block, in_blk, d_bits, scales, count = r
                print(f"  L{L:2d} {tname.split('.')[-2]:10s}: NaN! row={row} col={col} blk={block} in_blk={in_blk} "
                      f"d_bits=0x{d_bits:04x} scales={scales} nan_count={count}", flush=True)


if __name__ == "__main__":
    main()
