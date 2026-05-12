#!/usr/bin/env python3
"""
qwen_dequant_np.py — numpy-vectorized Q4_K + Q6_K dequant.

Замена scalar Python loops для real-dim inference. Ожидаемый speedup ~50×.
"""

import numpy as np
import struct


def fp16_to_fp32_np(h: np.ndarray) -> np.ndarray:
    """Vectorized fp16 → fp32. h is uint16 array. Handles subnormals correctly."""
    # Use numpy's built-in fp16 view to fp32 — handles all cases including subnormals
    return h.astype(np.uint16).view(np.float16).astype(np.float32)


def dequant_q4k_rows_np(raw: bytes, n_rows: int, K: int) -> np.ndarray:
    """Decode n_rows × K floats from packed Q4_K weight buffer.
    raw = n_rows * (K/256) * 144 bytes contiguous."""
    bs = 144
    bpr = K // 256
    row_bytes = bpr * bs
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, bpr, bs)
    # Per-block: d (2 bytes fp16), dmin (2 bytes fp16), scales[12], qs[128]
    d_bits = arr[:, :, 0].astype(np.uint16) | (arr[:, :, 1].astype(np.uint16) << 8)
    dmin_bits = arr[:, :, 2].astype(np.uint16) | (arr[:, :, 3].astype(np.uint16) << 8)
    d = fp16_to_fp32_np(d_bits)            # (n_rows, bpr)
    dmin = fp16_to_fp32_np(dmin_bits)
    scales = arr[:, :, 4:16]              # (n_rows, bpr, 12)
    qs = arr[:, :, 16:144]                # (n_rows, bpr, 128)

    # Decode 8 sub-block scales/mins per block (j=0..7)
    sc_low4 = (scales[:, :, 0:4] & 63).astype(np.int32)
    m_low4  = (scales[:, :, 4:8] & 63).astype(np.int32)
    # high 4: combine from scales[8..11] + bits из low8
    sc_high4 = ((scales[:, :, 8:12] & 0xF).astype(np.int32) |
                ((scales[:, :, 0:4] >> 6).astype(np.int32) << 4))
    m_high4  = ((scales[:, :, 8:12] >> 4).astype(np.int32) |
                ((scales[:, :, 4:8] >> 6).astype(np.int32) << 4))
    sc = np.concatenate([sc_low4, sc_high4], axis=-1)    # (n_rows, bpr, 8)
    m_ = np.concatenate([m_low4, m_high4], axis=-1)

    # Each block: 256 outputs. Sub-blocks (j*32 to (j+1)*32) for j=0..7.
    # qs[128] = 256 nibbles. For j-th sub-block (32 values):
    #   indices i = j*32 + l, l=0..31
    #   nibble_byte = qs[j*16 + (l - (j&1)*0)] hmm pairing differs
    # Pattern from C reference:
    #   For each block, j increments by 2 (j=0,2,4,6 means sub-block pairs 0-1, 2-3, 4-5, 6-7)
    #   At each pair: 32 bytes of qs, lower nibble → sub_j, upper nibble → sub_j+1
    out = np.zeros((n_rows, bpr, 256), dtype=np.float32)
    for j2 in range(0, 8, 2):
        # Pair (j2, j2+1) uses qs[j2/2*32 + l] for l=0..31
        qs_off = (j2 // 2) * 32
        qs_block = qs[:, :, qs_off:qs_off+32].astype(np.int32)   # (n_rows, bpr, 32)
        d1 = d * sc[:, :, j2].astype(np.float32)       # (n_rows, bpr)
        m1 = dmin * m_[:, :, j2].astype(np.float32)
        d2 = d * sc[:, :, j2+1].astype(np.float32)
        m2 = dmin * m_[:, :, j2+1].astype(np.float32)
        # Lower nibbles (sub-block j2): 32 values
        lo = qs_block & 0xF
        hi = qs_block >> 4
        # Output positions
        # In C ref: out[j + l + 0] = d1 * lo - m1; out[j + l + 32] = d2 * hi - m2
        # where j = (j2/2) * 64 (since we iterate j=0,64,128,192)
        base = (j2 // 2) * 64
        out[:, :, base:base+32]    = d1[:, :, None] * lo.astype(np.float32) - m1[:, :, None]
        out[:, :, base+32:base+64] = d2[:, :, None] * hi.astype(np.float32) - m2[:, :, None]
    return out.reshape(n_rows, K)


def dequant_q6k_rows_np(raw: bytes, n_rows: int, K: int) -> np.ndarray:
    """Vectorized Q6_K dequant."""
    bs = 210
    bpr = K // 256
    row_bytes = bpr * bs
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, bpr, bs)
    ql = arr[:, :, 0:128].astype(np.int32)    # (n_rows, bpr, 128)
    qh = arr[:, :, 128:192].astype(np.int32)  # (n_rows, bpr, 64)
    scales = arr[:, :, 192:208].view(np.int8).astype(np.int32)   # signed
    d_bits = arr[:, :, 208].astype(np.uint16) | (arr[:, :, 209].astype(np.uint16) << 8)
    d = fp16_to_fp32_np(d_bits)               # (n_rows, bpr)

    out = np.zeros((n_rows, bpr, 256), dtype=np.float32)
    for i in range(256):
        is_ = i // 16
        ql_idx = (i % 64) + 64 * (i // 128)
        q_lo = (ql[:, :, ql_idx] >> (4 * ((i // 32) & 1))) & 0xF
        qh_idx = (i % 32) + 32 * (i // 128)
        q_hi = (qh[:, :, qh_idx] >> (2 * ((i // 16) & 3))) & 0x3
        sc = scales[:, :, is_]
        out[:, :, i] = (d * sc.astype(np.float32) *
                        ((q_lo | (q_hi << 4)) - 32).astype(np.float32))
    return out.reshape(n_rows, K)


# Module test
if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) < 2:
        print("Usage: qwen_dequant_np.py <gguf_path>")
        sys.exit(1)
    path = sys.argv[1]
    from qwen_embed_lookup import parse_gguf_tensor_table

    tt, base = parse_gguf_tensor_table(path)
    for name in ("blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_gate.weight"):
        for (n, dims, t, off) in tt:
            if n == name:
                K, M = dims if len(dims) == 2 else (dims[0], 1)
                if t == 12:
                    bs = 144
                elif t == 14:
                    bs = 210
                else:
                    continue
                bpr = K // 256
                rb = bpr * bs
                # Time scalar vs vectorized
                with open(path, "rb") as f:
                    f.seek(base + off)
                    chunk = f.read(min(M, 128) * rb)
                t0 = time.time()
                if t == 12:
                    res = dequant_q4k_rows_np(chunk, min(M, 128), K)
                else:
                    res = dequant_q6k_rows_np(chunk, min(M, 128), K)
                dt = time.time() - t0
                print(f"{name}: {min(M,128)} rows × {K} K → {res.shape} ({dt*1000:.0f}ms, type={t})")
                print(f"  first 4 = {res[0, :4].tolist()}")
