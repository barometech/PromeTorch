#!/usr/bin/env python3
"""
qwen_python_layer.py — pure-Python (numpy) reference для полного Qwen3-4B layer.

Match exactly nmc_qwen_full_layer.c (step7): 2-head attention + FFN block с
subset M_FFN=256. Производит x_final[2560] для заданного x[2560] и pos.

Используется для validation NMC4 kernel output. Может также служить как
slow reference inference для 36-layer chain.
"""

import sys
import math
import struct
import numpy as np
from qwen_embed_lookup import parse_gguf_tensor_table, fp16_to_fp32, dequant_q6k_block
from qwen_q4k_dequant import dequant_q4k_block

K_DIM = 2560
HEAD_DIM = 128
N_HEADS_SUB = 2
M_FFN = 256
EPS = 1e-6
ROPE_BASE = 1e6


def load_q4k_rows(f, off: int, n_rows: int, K: int) -> np.ndarray:
    out = np.zeros((n_rows, K), dtype=np.float32)
    bs = 144
    bpr = K // 256
    row_bytes = bpr * bs
    for r in range(n_rows):
        f.seek(off + r * row_bytes)
        row = f.read(row_bytes)
        for b in range(bpr):
            vals = dequant_q4k_block(row[b * bs:(b + 1) * bs])
            out[r, b * 256:(b + 1) * 256] = vals
    return out


def load_q6k_rows(f, off: int, n_rows: int, K: int) -> np.ndarray:
    out = np.zeros((n_rows, K), dtype=np.float32)
    bs = 210
    bpr = K // 256
    row_bytes = bpr * bs
    for r in range(n_rows):
        f.seek(off + r * row_bytes)
        row = f.read(row_bytes)
        for b in range(bpr):
            vals = dequant_q6k_block(row[b * bs:(b + 1) * bs])
            out[r, b * 256:(b + 1) * 256] = vals
    return out


def load_fp32(f, off: int, n: int) -> np.ndarray:
    f.seek(off)
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def qwen_layer_forward(gguf_path: str, layer_idx: int, x: np.ndarray, pos: int,
                       n_heads: int = N_HEADS_SUB, m_ffn: int = M_FFN) -> np.ndarray:
    """Полный Qwen3-4B layer forward на CPU.

    Subset: первые n_heads*HEAD_DIM строк attn_q/k/v, первые n_heads*HEAD_DIM
    столбцов attn_output, первые m_ffn строк ffn_gate/up, первые m_ffn столбцов
    ffn_down. Matches step7 NMC kernel layout."""
    tt, base = parse_gguf_tensor_table(gguf_path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    L = layer_idx

    f = open(gguf_path, "rb")
    M_HEADS = n_heads * HEAD_DIM
    # gammas
    off, _, _ = by_name[f"blk.{L}.attn_norm.weight"]
    attn_norm = load_fp32(f, off, K_DIM)
    off, _, _ = by_name[f"blk.{L}.attn_q_norm.weight"]
    q_norm = load_fp32(f, off, HEAD_DIM)
    off, _, _ = by_name[f"blk.{L}.attn_k_norm.weight"]
    k_norm = load_fp32(f, off, HEAD_DIM)
    off, _, _ = by_name[f"blk.{L}.ffn_norm.weight"]
    ffn_norm = load_fp32(f, off, K_DIM)
    # weights (subset)
    off, _, _ = by_name[f"blk.{L}.attn_q.weight"]
    Wq = load_q4k_rows(f, off, M_HEADS, K_DIM)
    off, _, _ = by_name[f"blk.{L}.attn_k.weight"]
    Wk = load_q4k_rows(f, off, M_HEADS, K_DIM)
    off, _, _ = by_name[f"blk.{L}.attn_v.weight"]
    Wv = load_q6k_rows(f, off, M_HEADS, K_DIM)
    # attn_output [4096, 2560] in GGUF dims = [K=4096 in, M=2560 out].
    # Need subset K=M_HEADS, full M=K_DIM. Subset = first M_HEADS cols of each row.
    off, _, _ = by_name[f"blk.{L}.attn_output.weight"]
    # Wo_full_K is 4096 cols per row, we want first M_HEADS=256 cols.
    # ggml stores K-wise: each row has 4096 weights. K_DIM=2560 rows.
    Wo = np.zeros((K_DIM, M_HEADS), dtype=np.float32)
    bs = 144
    real_bpr = 4096 // 256   # 16
    real_row_bytes = real_bpr * bs
    sub_bpr = M_HEADS // 256  # 1
    for r in range(K_DIM):
        f.seek(off + r * real_row_bytes)
        for b in range(sub_bpr):
            blk = f.read(bs)
            Wo[r, b * 256:(b + 1) * 256] = dequant_q4k_block(blk)
    # FFN: gate, up [2560, 9728] → subset first m_ffn rows (K=2560, M=m_ffn)
    off, _, _ = by_name[f"blk.{L}.ffn_gate.weight"]
    Wgate = load_q4k_rows(f, off, m_ffn, K_DIM)
    off, _, _ = by_name[f"blk.{L}.ffn_up.weight"]
    Wup   = load_q4k_rows(f, off, m_ffn, K_DIM)
    # ffn_down [9728, 2560] → subset K=m_ffn (first m_ffn cols of each row), M=2560 full
    off, _, _ = by_name[f"blk.{L}.ffn_down.weight"]
    bs6 = 210
    real_bpr6 = 9728 // 256   # 38
    real_row_bytes6 = real_bpr6 * bs6
    sub_bpr6 = m_ffn // 256
    Wd = np.zeros((K_DIM, m_ffn), dtype=np.float32)
    for r in range(K_DIM):
        f.seek(off + r * real_row_bytes6)
        for b in range(sub_bpr6):
            blk = f.read(bs6)
            Wd[r, b * 256:(b + 1) * 256] = dequant_q6k_block(blk)
    f.close()

    # === ATTENTION ===
    inv_rms = 1.0 / np.sqrt(np.mean(x * x) + EPS)
    y = x * inv_rms * attn_norm

    q = Wq @ y; k = Wk @ y; v = Wv @ y   # M_HEADS=256 each
    # per-head norm
    q = q.reshape(n_heads, HEAD_DIM)
    k = k.reshape(n_heads, HEAD_DIM)
    for h in range(n_heads):
        qi = 1.0 / np.sqrt(np.mean(q[h] * q[h]) + EPS)
        ki = 1.0 / np.sqrt(np.mean(k[h] * k[h]) + EPS)
        q[h] = q[h] * qi * q_norm
        k[h] = k[h] * ki * k_norm
    # RoPE
    for h in range(n_heads):
        for i in range(0, HEAD_DIM, 2):
            theta = 1.0 / (ROPE_BASE ** (i / HEAD_DIM))
            angle = pos * theta
            c, s = np.cos(angle), np.sin(angle)
            q0, q1 = q[h, i], q[h, i + 1]
            k0, k1 = k[h, i], k[h, i + 1]
            q[h, i] = q0 * c - q1 * s; q[h, i + 1] = q0 * s + q1 * c
            k[h, i] = k0 * c - k1 * s; k[h, i + 1] = k0 * s + k1 * c
    # cache_len=1 → softmax trivial, attn = V
    attn_concat = v   # since prob=1.0 per head, attn_head[h] = V[h]
    attn_out = Wo @ attn_concat
    x_post = x + attn_out

    # === FFN SwiGLU ===
    inv_rms2 = 1.0 / np.sqrt(np.mean(x_post * x_post) + EPS)
    y2 = x_post * inv_rms2 * ffn_norm
    g = Wgate @ y2
    u = Wup @ y2
    silu = g / (1.0 + np.exp(-g))
    mul = silu * u
    ffn_out = Wd @ mul
    x_final = x_post + ffn_out

    return x_final.astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print("Usage: qwen_python_layer.py <gguf> <emb.bin> [layer_idx] [pos]")
        sys.exit(1)
    path = sys.argv[1]
    in_path = sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    pos = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    with open(in_path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        x = np.frombuffer(f.read(K_DIM * 4), dtype=np.float32).copy()
    print(f"[load] x[{K_DIM}] L2={np.linalg.norm(x):.4f}, layer={L}, pos={pos}")

    print(f"[forward] computing Python layer reference (slow, dequanting all subset weights)...")
    x_final = qwen_layer_forward(path, L, x, pos)
    print(f"[done] x_final[0..3] = {x_final[:4].tolist()}")
    print(f"[done] x_final[2557..] = {x_final[-3:].tolist()}")
    print(f"[done] x_final L2 = {np.linalg.norm(x_final):.4f}")


if __name__ == "__main__":
    main()
