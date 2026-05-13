#!/usr/bin/env python3
"""Multi-token Qwen3-4B generation on NM Quad host x86 (CPU).
После Q6_K dequant fix теперь генерирует осмысленный текст."""
import sys, math, time, numpy as np
sys.path.insert(0, "/home/<user>/qwen/v1")
from qwen_tokenizer import parse_gguf_tokenizer, QwenBPE
from qwen_embed_lookup import lookup_embedding, parse_gguf_tensor_table
from qwen_dequant_np import dequant_q6k_rows_np
from qwen_layer_full import (load_quant, load_fp32, rmsnorm, rope,
                              K_DIM, HEAD_DIM, N_HEADS, N_KV_HEADS, M_FFN, EPS, ROPE_BASE)


def load_layer(path, L):
    tt, base = parse_gguf_tensor_table(path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    f = open(path, "rb")
    def wq(name, rows, cols):
        off_t, _, ttype_t = by_name[name]
        return load_quant(f, off_t, rows, cols, ttype_t)
    w = {
        "attn_norm": load_fp32(f, by_name[f"blk.{L}.attn_norm.weight"][0], K_DIM),
        "q_norm":    load_fp32(f, by_name[f"blk.{L}.attn_q_norm.weight"][0], HEAD_DIM),
        "k_norm":    load_fp32(f, by_name[f"blk.{L}.attn_k_norm.weight"][0], HEAD_DIM),
        "ffn_norm":  load_fp32(f, by_name[f"blk.{L}.ffn_norm.weight"][0], K_DIM),
        "Wq":    wq(f"blk.{L}.attn_q.weight",      N_HEADS * HEAD_DIM, K_DIM),
        "Wk":    wq(f"blk.{L}.attn_k.weight",      N_KV_HEADS * HEAD_DIM, K_DIM),
        "Wv":    wq(f"blk.{L}.attn_v.weight",      N_KV_HEADS * HEAD_DIM, K_DIM),
        "Wo":    wq(f"blk.{L}.attn_output.weight", K_DIM, N_HEADS * HEAD_DIM),
        "Wgate": wq(f"blk.{L}.ffn_gate.weight",    M_FFN, K_DIM),
        "Wup":   wq(f"blk.{L}.ffn_up.weight",      M_FFN, K_DIM),
        "Wd":    wq(f"blk.{L}.ffn_down.weight",    K_DIM, M_FFN),
    }
    f.close()
    return w


def layer_fwd(w, x, pos, Kc, Vc):
    y = rmsnorm(x, w["attn_norm"])
    q = (w["Wq"] @ y).reshape(N_HEADS, HEAD_DIM).copy()
    k = (w["Wk"] @ y).reshape(N_KV_HEADS, HEAD_DIM).copy()
    v = (w["Wv"] @ y).reshape(N_KV_HEADS, HEAD_DIM).copy()
    for h in range(N_HEADS):     q[h] = rmsnorm(q[h], w["q_norm"])
    for h in range(N_KV_HEADS):  k[h] = rmsnorm(k[h], w["k_norm"])
    for h in range(N_HEADS):     q[h] = rope(q[h], pos)
    for h in range(N_KV_HEADS):  k[h] = rope(k[h], pos)
    Kc[pos] = k; Vc[pos] = v
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
    seq = pos + 1
    for h in range(N_HEADS):
        kv_h = h // (N_HEADS // N_KV_HEADS)
        Ks = Kc[:seq, kv_h]; Vs = Vc[:seq, kv_h]
        s = (Ks @ q[h]) * scale
        m = s.max(); e = np.exp(s - m); p = e / e.sum()
        attn[h] = p @ Vs
    x_post = x + w["Wo"] @ attn.reshape(N_HEADS * HEAD_DIM)
    y2 = rmsnorm(x_post, w["ffn_norm"])
    g = w["Wgate"] @ y2; u = w["Wup"] @ y2
    silu = g / (1.0 + np.exp(-g))
    return (x_post + w["Wd"] @ (silu * u)).astype(np.float32)


def project_logits(path, x):
    """Greedy top-1 logit projection через 151936-vocab Q6_K lm_head."""
    tt, base = parse_gguf_tensor_table(path)
    by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
    on_off = by_name["output_norm.weight"][0]
    tem_off = by_name["token_embd.weight"][0]
    f = open(path, "rb")
    on = load_fp32(f, on_off, K_DIM)
    f.close()
    x_n = rmsnorm(x, on)
    K, M = 2560, 151936; bs = 210; bpr = K // 256; row_bytes = bpr * bs
    best_lg = -1e30; best_id = 0
    f = open(path, "rb")
    for start in range(0, M, 4096):
        end = min(start + 4096, M); n = end - start
        f.seek(tem_off + start * row_bytes)
        raw = f.read(n * row_bytes)
        W = dequant_q6k_rows_np(raw, n, K)
        lg = W @ x_n
        i = int(np.argmax(lg))
        if lg[i] > best_lg:
            best_lg = float(lg[i]); best_id = start + i
    f.close()
    return best_id, best_lg


def main():
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    n_layers = 36

    tokens, merges, bos, eos = parse_gguf_tokenizer(path)
    bpe = QwenBPE(tokens, merges, bos, eos)
    ids = list(bpe.encode(prompt))
    print(f"[prompt] {prompt!r} → tokens={ids}", flush=True)

    print(f"[load] loading {n_layers} layers...", flush=True)
    t0 = time.time()
    layers = [load_layer(path, L) for L in range(n_layers)]
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    seq_max = len(ids) + n_gen + 8
    Kc = [np.zeros((seq_max, N_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(n_layers)]
    Vc = [np.zeros((seq_max, N_KV_HEADS, HEAD_DIM), dtype=np.float32) for _ in range(n_layers)]

    # Prefill
    print(f"[prefill] {len(ids)} tokens...", flush=True)
    t0 = time.time()
    x_last = None
    for pos, tid in enumerate(ids):
        x = np.array(lookup_embedding(path, tid), dtype=np.float32)
        for L in range(n_layers):
            x = layer_fwd(layers[L], x, pos, Kc[L], Vc[L])
        x_last = x
    print(f"[prefill] {time.time()-t0:.0f}s", flush=True)

    # Generate n_gen tokens
    generated = []
    for g_i in range(n_gen):
        t0 = time.time()
        tid, lg = project_logits(path, x_last)
        generated.append(tid)
        ids.append(tid)
        pos = len(ids) - 1
        # forward new token
        x = np.array(lookup_embedding(path, tid), dtype=np.float32)
        for L in range(n_layers):
            x = layer_fwd(layers[L], x, pos, Kc[L], Vc[L])
        x_last = x
        tok_text = tokens[tid] if tid < len(tokens) else f"<{tid}>"
        print(f"[gen {g_i+1}] id={tid} logit={lg:.2f} token={tok_text!r}  ({time.time()-t0:.1f}s)", flush=True)

    # Decode
    full_text_ids = [int(i) for i in ids]
    print(f"\n[FULL TEXT] {bpe.decode(full_text_ids)}")


if __name__ == "__main__":
    main()
