#!/usr/bin/env python3
"""
qwen_q4k_dequant.py — Q4_K dequant в Python для full-layer reference.

Parity с C++ host_q4k_test.cpp / nmc_q4k_test.c (bit-exact ggml layout).
Используется для extracting Qwen3-4B attn_q/k, ffn_gate/up как fp32 для
Python reference forward.
"""

import sys
import struct
import math
from qwen_embed_lookup import parse_gguf_tensor_table, fp16_to_fp32


def get_scale_min_k4(j: int, q: bytes):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def dequant_q4k_block(blk: bytes) -> list:
    """Decode 256 fp32 values из 144-byte Q4_K block."""
    d = fp16_to_fp32((blk[1] << 8) | blk[0])
    dmin = fp16_to_fp32((blk[3] << 8) | blk[2])
    scales = blk[4:16]
    qs = blk[16:144]
    out = [0.0] * 256
    is_ = 0
    for j in range(0, 256, 64):
        sc, m = get_scale_min_k4(is_, scales)
        d1 = d * sc
        m1 = dmin * m
        sc2, m2v = get_scale_min_k4(is_ + 1, scales)
        d2 = d * sc2
        m2 = dmin * m2v
        for l in range(32):
            qb = qs[is_ // 2 * 32 + l]
            out[j + l] = d1 * (qb & 0xF) - m1
            out[j + l + 32] = d2 * (qb >> 4) - m2
        is_ += 2
    return out


def dequant_row_q4k(f, file_offset: int, row_idx: int, K: int) -> list:
    """Извлекает row из Q4_K matrix (K = row length)."""
    block_size = 144
    blocks_per_row = K // 256
    row_bytes = blocks_per_row * block_size
    f.seek(file_offset + row_idx * row_bytes)
    row = f.read(row_bytes)
    out = []
    for b in range(blocks_per_row):
        out.extend(dequant_q4k_block(row[b * block_size:(b + 1) * block_size]))
    return out


def main():
    if len(sys.argv) < 4:
        print("Usage: qwen_q4k_dequant.py <gguf> <tensor_name> <row_idx>")
        sys.exit(1)
    path = sys.argv[1]
    name = sys.argv[2]
    row_idx = int(sys.argv[3])

    tt, base = parse_gguf_tensor_table(path)
    target = None
    for (n, dims, t, off) in tt:
        if n == name:
            target = (n, base + off, dims, t)
            break
    if not target:
        print(f"tensor {name!r} not found")
        sys.exit(2)
    _, off, dims, ttype = target
    print(f"[{name}] dims={dims} type={ttype} off={off}")
    assert ttype == 12, f"expected Q4_K, got {ttype}"
    K = dims[0]
    M = dims[1]
    print(f"[{name}] K={K} M={M}, extracting row {row_idx} of {M}")

    with open(path, "rb") as f:
        row = dequant_row_q4k(f, off, row_idx, K)
    s2 = sum(x * x for x in row)
    print(f"  L2 = {math.sqrt(s2):.4f}")
    print(f"  first 8 = {row[:8]}")
    print(f"  last 8  = {row[-8:]}")


if __name__ == "__main__":
    main()
