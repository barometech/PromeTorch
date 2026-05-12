#!/usr/bin/env python3
"""Same as qwen_prefill_gen but skip q_norm/k_norm — sanity test
whether qk_norm application is the source of garbage top-1."""
import sys
import math
import time
import numpy as np
from qwen_tokenizer import parse_gguf_tokenizer, QwenBPE
from qwen_embed_lookup import lookup_embedding, parse_gguf_tensor_table
from qwen_dequant_np import dequant_q6k_rows_np
from qwen_layer_full import (load_quant, load_fp32, rmsnorm, rope,
                              K_DIM, HEAD_DIM, N_HEADS, N_KV_HEADS, M_FFN,
                              EPS, ROPE_BASE)


def load_layer_weights(path, L):
    tt, base = parse_gguf_tensor_table(path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    f = open(path, "rb")
    def wq(name, rows, cols):
        off_t, _, ttype_t = by_name[name]
        return load_quant(f, off_t, rows, cols, ttype_t)
    w = {}
    w["attn_norm"] = load_fp32(f, by_name[f"blk.{L}.attn_norm.weight"][0], K_DIM)
    w["ffn_norm"]  = load_fp32(f, by_name[f"blk.{L}.ffn_norm.weight"][0], K_DIM)
    w["Wq"]    = wq(f"blk.{L}.attn_q.weight",      N_HEADS * HEAD_DIM, K_DIM)
    w["Wk"]    = wq(f"blk.{L}.attn_k.weight",      N_KV_HEADS * HEAD_DIM, K_DIM)
    w["Wv"]    = wq(f"blk.{L}.attn_v.weight",      N_KV_HEADS * HEAD_DIM, K_DIM)
    w["Wo"]    = wq(f"blk.{L}.attn_output.weight", K_DIM, N_HEADS * HEAD_DIM)
    w["Wgate"] = wq(f"blk.{L}.ffn_gate.weight",    M_FFN, K_DIM)
    w["Wup"]   = wq(f"blk.{L}.ffn_up.weight",      M_FFN, K_DIM)
    w["Wd"]    = wq(f"blk.{L}.ffn_down.weight",    K_DIM, M_FFN)
    f.close()
    return w


def fwd(w, x, pos, K_cache, V_cache):
    y = rmsnorm(x, w["attn_norm"])
    q = (w["Wq"] @ y).reshape(N_HEADS, HEAD_DIM)
    k = (w["Wk"] @ y).reshape(N_KV_HEADS, HEAD_DIM)
    v = (w["Wv"] @ y).reshape(N_KV_HEADS, HEAD_DIM)
    # SKIP qk_norm
    for h in range(N_HEADS):     q[h] = rope(q[h], pos)
    for h in range(N_KV_HEADS):  k[h] = rope(k[h], pos)
    K_cache[pos] = k
    V_cache[pos] = v
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
    seq_len = pos + 1
    for h in range(N_HEADS):
        kv_h = h // (N_HEADS // N_KV_HEADS)
        Ks = K_cache[:seq_len, kv_h]
        scores = (Ks @ q[h]) * scale
        m = scores.max(); e = np.exp(scores - m); p = e / e.sum()
        Vs = V_cache[:seq_len, kv_h]
        attn[h] = p @ Vs
    attn_concat = attn.reshape(N_HEADS * HEAD_DIM)
    attn_out = w["Wo"] @ attn_concat
    x_post = x + attn_out
    y2 = rmsnorm(x_post, w["ffn_norm"])
    g = w["Wgate"] @ y2; u = w["Wup"] @ y2
    silu = g / (1.0 + np.exp(-g)); mul = silu * u
    ffn_out = w["Wd"] @ mul
    return (x_post + ffn_out).astype(np.float32)


def main():
    path = sys.argv[1]; prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 36
    tokens, merges, bos, eos = parse_gguf_tokenizer(path)
    bpe = QwenBPE(tokens, merges, bos, eos)
    ids = bpe.encode(prompt)
    print(f"[tokens] {ids}", flush=True)
    seq_len_max = len(ids)
    embs = [np.array(lookup_embedding(path, tid), dtype=np.float32) for tid in ids]
    layers = []
    print(f"[load] loading {n_layers} layers...", flush=True)
    t0 = time.time()
    for L in range(n_layers):
        layers.append(load_layer_weights(path, L))
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    K_caches = [np.zeros((seq_len_max, N_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(n_layers)]
    V_caches = [np.zeros((seq_len_max, N_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(n_layers)]
    t0 = time.time()
    x_last = None
    for pos, tid in enumerate(ids):
        x = embs[pos]
        for L in range(n_layers):
            x = fwd(layers[L], x, pos, K_caches[L], V_caches[L])
        x_last = x
        print(f"  pos={pos} {tokens[tid]!r} L2={np.linalg.norm(x):.2f}", flush=True)
    print(f"[prefill] {time.time()-t0:.0f}s", flush=True)
    # output_norm + lm_head
    tt, base = parse_gguf_tensor_table(path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    f = open(path, "rb")
    on = load_fp32(f, by_name["output_norm.weight"][0], K_DIM)
    f.close()
    x_last = rmsnorm(x_last, on)
    print(f"[output_norm] L2={np.linalg.norm(x_last):.2f}", flush=True)
    tem_off = by_name["token_embd.weight"][0]
    K, M = 2560, 151936; bs = 210; bpr = K // 256; row_bytes = bpr * bs
    top = []
    f = open(path, "rb")
    for start in range(0, M, 4096):
        end = min(start + 4096, M); n = end - start
        f.seek(tem_off + start * row_bytes); raw = f.read(n * row_bytes)
        W = dequant_q6k_rows_np(raw, n, K); logits = W @ x_last
        for tid_off, lg in enumerate(logits):
            tid = start + tid_off
            if len(top) < 5: top.append((float(lg), tid)); top.sort()
            elif lg > top[0][0]: top[0] = (float(lg), tid); top.sort()
    f.close()
    print(f"[top-5 NO_QKNORM]:")
    for lg, tid in top[::-1]:
        print(f"  id={tid:6d} logit={lg:>7.3f} token={tokens[tid]!r}")

if __name__ == "__main__":
    main()
