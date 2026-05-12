#!/usr/bin/env python3
"""
qwen_layer_full.py — REAL-DIMS Qwen3-4B layer forward в numpy.

Полные dims: N_HEADS=32, N_KV_HEADS=8 (GQA), M_FFN=9728. Использует
vectorized dequant qwen_dequant_np для acceptable CPU speed.
"""

import sys
import math
import time
import struct
import numpy as np
from qwen_embed_lookup import parse_gguf_tensor_table
from qwen_dequant_np import dequant_q4k_rows_np, dequant_q6k_rows_np

K_DIM = 2560
HEAD_DIM = 128
N_HEADS = 32
N_KV_HEADS = 8
M_FFN = 9728
EPS = 1e-6
ROPE_BASE = 1e6


def load_q4k(f, off: int, n_rows: int, K: int) -> np.ndarray:
    bs = 144
    bpr = K // 256
    f.seek(off)
    raw = f.read(n_rows * bpr * bs)
    return dequant_q4k_rows_np(raw, n_rows, K)


def load_q6k(f, off: int, n_rows: int, K: int) -> np.ndarray:
    bs = 210
    bpr = K // 256
    f.seek(off)
    raw = f.read(n_rows * bpr * bs)
    return dequant_q6k_rows_np(raw, n_rows, K)


def load_quant(f, off: int, n_rows: int, K: int, ttype: int) -> np.ndarray:
    """Generic dequant. ttype=12 Q4_K, ttype=14 Q6_K (Qwen3-4B Q4_K_M mix)."""
    if ttype == 12:
        return load_q4k(f, off, n_rows, K)
    if ttype == 14:
        return load_q6k(f, off, n_rows, K)
    raise ValueError(f"unsupported ttype={ttype}")


def load_fp32(f, off: int, n: int) -> np.ndarray:
    f.seek(off)
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()


def rmsnorm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return x * (1.0 / np.sqrt(np.mean(x * x) + EPS)) * gamma


def rope(v: np.ndarray, pos: int) -> np.ndarray:
    """RoPE per-head [..., HEAD_DIM]. Rotates pairs (2i, 2i+1)."""
    v = v.copy()
    for i in range(0, HEAD_DIM, 2):
        theta = 1.0 / (ROPE_BASE ** (i / HEAD_DIM))
        angle = pos * theta
        c, s = np.cos(angle), np.sin(angle)
        v0 = v[..., i].copy()
        v1 = v[..., i + 1].copy()
        v[..., i]     = v0 * c - v1 * s
        v[..., i + 1] = v0 * s + v1 * c
    return v


def qwen_full_layer(gguf_path: str, layer_idx: int, x: np.ndarray, pos: int):
    """Полный real-dim Qwen3-4B layer forward."""
    tt, base = parse_gguf_tensor_table(gguf_path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    L = layer_idx
    f = open(gguf_path, "rb")

    def wq(name, rows, cols):
        off_t, dims_t, ttype_t = by_name[name]
        return load_quant(f, off_t, rows, cols, ttype_t)

    t0 = time.time()
    attn_norm = load_fp32(f, by_name[f"blk.{L}.attn_norm.weight"][0], K_DIM)
    q_norm    = load_fp32(f, by_name[f"blk.{L}.attn_q_norm.weight"][0], HEAD_DIM)
    k_norm    = load_fp32(f, by_name[f"blk.{L}.attn_k_norm.weight"][0], HEAD_DIM)
    ffn_norm  = load_fp32(f, by_name[f"blk.{L}.ffn_norm.weight"][0], K_DIM)
    print(f"  [load] gammas {time.time()-t0:.2f}s", flush=True)

    t1 = time.time()
    Wq = wq(f"blk.{L}.attn_q.weight", N_HEADS * HEAD_DIM, K_DIM)
    print(f"  [load] Wq {time.time()-t1:.2f}s shape={Wq.shape}", flush=True)
    t1 = time.time()
    Wk = wq(f"blk.{L}.attn_k.weight", N_KV_HEADS * HEAD_DIM, K_DIM)
    print(f"  [load] Wk {time.time()-t1:.2f}s shape={Wk.shape}", flush=True)
    t1 = time.time()
    Wv = wq(f"blk.{L}.attn_v.weight", N_KV_HEADS * HEAD_DIM, K_DIM)
    print(f"  [load] Wv {time.time()-t1:.2f}s shape={Wv.shape}", flush=True)
    t1 = time.time()
    Wo = wq(f"blk.{L}.attn_output.weight", K_DIM, N_HEADS * HEAD_DIM)
    print(f"  [load] Wo {time.time()-t1:.2f}s shape={Wo.shape}", flush=True)
    t1 = time.time()
    Wgate = wq(f"blk.{L}.ffn_gate.weight", M_FFN, K_DIM)
    print(f"  [load] Wgate {time.time()-t1:.2f}s shape={Wgate.shape}", flush=True)
    t1 = time.time()
    Wup = wq(f"blk.{L}.ffn_up.weight", M_FFN, K_DIM)
    print(f"  [load] Wup {time.time()-t1:.2f}s shape={Wup.shape}", flush=True)
    t1 = time.time()
    Wd = wq(f"blk.{L}.ffn_down.weight", K_DIM, M_FFN)
    print(f"  [load] Wd {time.time()-t1:.2f}s shape={Wd.shape}", flush=True)
    f.close()

    # === ATTENTION ===
    y = rmsnorm(x, attn_norm)
    q_full = Wq @ y    # [N_HEADS*HEAD_DIM]
    k_full = Wk @ y    # [N_KV_HEADS*HEAD_DIM]
    v_full = Wv @ y    # [N_KV_HEADS*HEAD_DIM]

    q = q_full.reshape(N_HEADS, HEAD_DIM)
    k = k_full.reshape(N_KV_HEADS, HEAD_DIM)
    v = v_full.reshape(N_KV_HEADS, HEAD_DIM)

    # per-head Q/K norm
    for h in range(N_HEADS):
        q[h] = rmsnorm(q[h], q_norm)
    for h in range(N_KV_HEADS):
        k[h] = rmsnorm(k[h], k_norm)

    # RoPE
    for h in range(N_HEADS):     q[h] = rope(q[h], pos)
    for h in range(N_KV_HEADS):  k[h] = rope(k[h], pos)

    # GQA: каждый Q head attends к (h // (N_HEADS // N_KV_HEADS))-th KV head
    # cache_len=1: softmax=1.0, attn_out[h] = V[kv_head]
    attn_out = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(N_HEADS):
        kv_h = h // (N_HEADS // N_KV_HEADS)
        attn_out[h] = v[kv_h]
    attn_concat = attn_out.reshape(N_HEADS * HEAD_DIM)

    # attn_output projection
    attn_proj = Wo @ attn_concat
    x_post = x + attn_proj

    # === FFN ===
    y2 = rmsnorm(x_post, ffn_norm)
    g = Wgate @ y2
    u = Wup @ y2
    silu = g / (1.0 + np.exp(-g))
    mul = silu * u
    ffn_out = Wd @ mul
    x_final = x_post + ffn_out

    return x_final.astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print("Usage: qwen_layer_full.py <gguf> <emb.bin> [layer] [pos]")
        sys.exit(1)
    path = sys.argv[1]
    in_path = sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    pos = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    with open(in_path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        x = np.frombuffer(f.read(K_DIM * 4), dtype=np.float32).copy()
    print(f"[in] x[{K_DIM}] L2={np.linalg.norm(x):.4f}, layer={L}, pos={pos}")

    t0 = time.time()
    x_final = qwen_full_layer(path, L, x, pos)
    dt = time.time() - t0
    print(f"[done] wall={dt:.2f}s")
    print(f"[done] x_final[0..3]={x_final[:4].tolist()}")
    print(f"[done] L2={np.linalg.norm(x_final):.4f}, finite={np.all(np.isfinite(x_final))}")


if __name__ == "__main__":
    main()
