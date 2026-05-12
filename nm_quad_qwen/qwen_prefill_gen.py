#!/usr/bin/env python3
"""
qwen_prefill_gen.py — Qwen3-4B inference с proper prefill + KV cache.

Real generation flow:
  1. Tokenize prompt → token IDs [t0, t1, ... tN-1]
  2. Prefill: for each token, forward через 36 layers с per-layer KV cache
  3. After last token, x_final → lm_head → next token

Каждый layer's K_cache и V_cache хранятся между tokens. С каждым новым
token, attention combines current Q с накопленными K_cache (positions 0..t).

Per-layer cache: K[seq_len, n_kv_heads, head_dim], V same.

CPU wall: ~100 sec/token (limited by dequant). Prefill 4 tokens = ~7 min.
"""

import sys
import math
import time
import numpy as np
from qwen_tokenizer import parse_gguf_tokenizer, QwenBPE
from qwen_embed_lookup import lookup_embedding, parse_gguf_tensor_table
from qwen_dequant_np import dequant_q4k_rows_np, dequant_q6k_rows_np
from qwen_layer_full import (load_q4k, load_q6k, load_fp32, rmsnorm, rope,
                              K_DIM, HEAD_DIM, N_HEADS, N_KV_HEADS, M_FFN,
                              EPS, ROPE_BASE)


def load_layer_weights(path: str, L: int):
    """Загружает ВСЕ weights одного layer (~30 sec на CPU)."""
    tt, base = parse_gguf_tensor_table(path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    f = open(path, "rb")
    w = {}
    w["attn_norm"] = load_fp32(f, by_name[f"blk.{L}.attn_norm.weight"][0], K_DIM)
    w["q_norm"]    = load_fp32(f, by_name[f"blk.{L}.attn_q_norm.weight"][0], HEAD_DIM)
    w["k_norm"]    = load_fp32(f, by_name[f"blk.{L}.attn_k_norm.weight"][0], HEAD_DIM)
    w["ffn_norm"]  = load_fp32(f, by_name[f"blk.{L}.ffn_norm.weight"][0], K_DIM)
    w["Wq"] = load_q4k(f, by_name[f"blk.{L}.attn_q.weight"][0], N_HEADS * HEAD_DIM, K_DIM)
    w["Wk"] = load_q4k(f, by_name[f"blk.{L}.attn_k.weight"][0], N_KV_HEADS * HEAD_DIM, K_DIM)
    w["Wv"] = load_q6k(f, by_name[f"blk.{L}.attn_v.weight"][0], N_KV_HEADS * HEAD_DIM, K_DIM)
    w["Wo"] = load_q4k(f, by_name[f"blk.{L}.attn_output.weight"][0], K_DIM, N_HEADS * HEAD_DIM)
    w["Wgate"] = load_q4k(f, by_name[f"blk.{L}.ffn_gate.weight"][0], M_FFN, K_DIM)
    w["Wup"]   = load_q4k(f, by_name[f"blk.{L}.ffn_up.weight"][0], M_FFN, K_DIM)
    w["Wd"]    = load_q6k(f, by_name[f"blk.{L}.ffn_down.weight"][0], K_DIM, M_FFN)
    f.close()
    return w


def layer_forward_with_cache(w, x, pos, K_cache, V_cache):
    """Single token forward с KV cache. Appends current k,v to cache, attention
    over cache[0..pos]. K_cache, V_cache shape [seq_max, N_KV_HEADS, HEAD_DIM]."""
    y = rmsnorm(x, w["attn_norm"])
    q = (w["Wq"] @ y).reshape(N_HEADS, HEAD_DIM)
    k = (w["Wk"] @ y).reshape(N_KV_HEADS, HEAD_DIM)
    v = (w["Wv"] @ y).reshape(N_KV_HEADS, HEAD_DIM)

    for h in range(N_HEADS):
        q[h] = rmsnorm(q[h], w["q_norm"])
    for h in range(N_KV_HEADS):
        k[h] = rmsnorm(k[h], w["k_norm"])

    for h in range(N_HEADS):     q[h] = rope(q[h], pos)
    for h in range(N_KV_HEADS):  k[h] = rope(k[h], pos)

    # Append к cache
    K_cache[pos] = k    # [N_KV_HEADS, HEAD_DIM]
    V_cache[pos] = v

    # Attention: каждый Q head attends to cache[0..pos] of its KV head
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
    seq_len = pos + 1
    for h in range(N_HEADS):
        kv_h = h // (N_HEADS // N_KV_HEADS)
        # scores[t] = q[h] · K_cache[t, kv_h] * scale
        Ks = K_cache[:seq_len, kv_h]   # [seq_len, HEAD_DIM]
        scores = (Ks @ q[h]) * scale
        # softmax
        m = scores.max()
        e = np.exp(scores - m)
        p = e / e.sum()
        # attn = sum_t p[t] * V_cache[t, kv_h]
        Vs = V_cache[:seq_len, kv_h]
        attn[h] = p @ Vs
    attn_concat = attn.reshape(N_HEADS * HEAD_DIM)
    attn_out = w["Wo"] @ attn_concat
    x_post = x + attn_out

    y2 = rmsnorm(x_post, w["ffn_norm"])
    g = w["Wgate"] @ y2
    u = w["Wup"] @ y2
    silu = g / (1.0 + np.exp(-g))
    mul = silu * u
    ffn_out = w["Wd"] @ mul
    return (x_post + ffn_out).astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print("Usage: qwen_prefill_gen.py <gguf> [prompt] [n_layers] [topk]")
        sys.exit(1)
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 36
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"[prompt] {prompt!r}")
    tokens, merges, bos, eos = parse_gguf_tokenizer(path)
    bpe = QwenBPE(tokens, merges, bos, eos)
    ids = bpe.encode(prompt)
    print(f"[tokens] {ids}")
    seq_len_max = len(ids)

    # Загружаем embeddings для каждого token
    embs = [np.array(lookup_embedding(path, tid), dtype=np.float32) for tid in ids]

    # Pre-load ALL layer weights once
    print(f"[load] Loading all {n_layers} layer weights (memory ~{n_layers*30}MB)...", flush=True)
    t0 = time.time()
    layers = []
    for L in range(n_layers):
        w = load_layer_weights(path, L)
        layers.append(w)
        print(f"  [layer {L:2d}/{n_layers}] loaded  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[load] all layers loaded in {time.time()-t0:.0f}s")

    # KV cache per layer
    K_caches = [np.zeros((seq_len_max, N_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(n_layers)]
    V_caches = [np.zeros((seq_len_max, N_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(n_layers)]

    # Prefill каждый token
    print(f"[prefill] forwarding {len(ids)} tokens through {n_layers} layers...")
    x_last = None
    t0 = time.time()
    for pos, tid in enumerate(ids):
        x = embs[pos]
        for L in range(n_layers):
            x = layer_forward_with_cache(layers[L], x, pos, K_caches[L], V_caches[L])
        x_last = x
        print(f"  [token {pos}: {tokens[tid]!r}] L2={np.linalg.norm(x):.3f}  finite={np.all(np.isfinite(x))}", flush=True)

    print(f"[prefill] total={time.time()-t0:.0f}s")

    # lm_head projection
    print(f"[lm_head] projecting через 151936 vocab Q6_K...")
    tt, base = parse_gguf_tensor_table(path)
    for (n, dims, t, off) in tt:
        if n == "token_embd.weight":
            tem_off = base + off
            break
    K, M = 2560, 151936
    bs = 210; bpr = K // 256; row_bytes = bpr * bs
    top = []
    f = open(path, "rb")
    t0 = time.time()
    for start in range(0, M, 4096):
        end = min(start + 4096, M)
        n = end - start
        f.seek(tem_off + start * row_bytes)
        raw = f.read(n * row_bytes)
        W = dequant_q6k_rows_np(raw, n, K)
        logits = W @ x_last
        for tid_off, lg in enumerate(logits):
            tid = start + tid_off
            if len(top) < topk:
                top.append((float(lg), tid)); top.sort()
            elif lg > top[0][0]:
                top[0] = (float(lg), tid); top.sort()
    f.close()
    print(f"[lm_head] {time.time()-t0:.1f}s")

    print(f"[top-{topk}]:")
    for lg, tid in top[::-1]:
        print(f"  id={tid:6d} logit={lg:>8.3f} token={tokens[tid]!r}")
    print(f"\n[GENERATED] {prompt!r} + {tokens[top[-1][1]]!r}")


if __name__ == "__main__":
    main()
