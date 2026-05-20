#!/usr/bin/env python3
"""Dump my Python L0 intermediate for "Hello"."""
import sys, math, numpy as np
sys.path.insert(0, "/home/elbrus-user/qwen/v1")
from qwen_embed_lookup import parse_gguf_tensor_table, lookup_embedding
from qwen_layer_full import (load_quant, load_fp32, rmsnorm, rope,
                              K_DIM, HEAD_DIM, N_HEADS, N_KV_HEADS, M_FFN, EPS, ROPE_BASE)

path = sys.argv[1]
L = 0; pos = 0
tt, base = parse_gguf_tensor_table(path)
by_name = {n: (base + off, dims, t) for (n, dims, t, off) in tt}
f = open(path, "rb")
def wq(name, rows, cols):
    off_t, _, ttype_t = by_name[name]
    return load_quant(f, off_t, rows, cols, ttype_t)
attn_norm = load_fp32(f, by_name[f"blk.{L}.attn_norm.weight"][0], K_DIM)
q_norm    = load_fp32(f, by_name[f"blk.{L}.attn_q_norm.weight"][0], HEAD_DIM)
k_norm    = load_fp32(f, by_name[f"blk.{L}.attn_k_norm.weight"][0], HEAD_DIM)
ffn_norm  = load_fp32(f, by_name[f"blk.{L}.ffn_norm.weight"][0], K_DIM)
Wq    = wq(f"blk.{L}.attn_q.weight",      N_HEADS * HEAD_DIM, K_DIM)
Wk    = wq(f"blk.{L}.attn_k.weight",      N_KV_HEADS * HEAD_DIM, K_DIM)
Wv    = wq(f"blk.{L}.attn_v.weight",      N_KV_HEADS * HEAD_DIM, K_DIM)
Wo    = wq(f"blk.{L}.attn_output.weight", K_DIM, N_HEADS * HEAD_DIM)
Wgate = wq(f"blk.{L}.ffn_gate.weight",    M_FFN, K_DIM)
Wup   = wq(f"blk.{L}.ffn_up.weight",      M_FFN, K_DIM)
Wd    = wq(f"blk.{L}.ffn_down.weight",    K_DIM, M_FFN)
f.close()

x = np.array(lookup_embedding(path, 9707), dtype=np.float32)
print(f"emb L2={np.linalg.norm(x):.4f}")
y = rmsnorm(x, attn_norm)
print(f"input_layernorm L2={np.linalg.norm(y):.4f}")
q_full = Wq @ y; k_full = Wk @ y; v_full = Wv @ y
print(f"q_proj L2={np.linalg.norm(q_full):.4f}")
print(f"k_proj L2={np.linalg.norm(k_full):.4f}")
print(f"v_proj L2={np.linalg.norm(v_full):.4f}")

q = q_full.reshape(N_HEADS, HEAD_DIM).copy()
k = k_full.reshape(N_KV_HEADS, HEAD_DIM).copy()
v = v_full.reshape(N_KV_HEADS, HEAD_DIM).copy()
for h in range(N_HEADS):     q[h] = rmsnorm(q[h], q_norm)
for h in range(N_KV_HEADS):  k[h] = rmsnorm(k[h], k_norm)
print(f"q_norm out L2={np.linalg.norm(q):.4f}")
print(f"k_norm out L2={np.linalg.norm(k):.4f}")

for h in range(N_HEADS):     q[h] = rope(q[h], pos)
for h in range(N_KV_HEADS):  k[h] = rope(k[h], pos)
attn_out = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
for h in range(N_HEADS):
    kv_h = h // (N_HEADS // N_KV_HEADS)
    attn_out[h] = v[kv_h]
attn_concat = attn_out.reshape(N_HEADS * HEAD_DIM)
attn_proj = Wo @ attn_concat
print(f"o_proj L2={np.linalg.norm(attn_proj):.4f}")

x_post = x + attn_proj
print(f"x_post (residual1) L2={np.linalg.norm(x_post):.4f}")

y2 = rmsnorm(x_post, ffn_norm)
print(f"post_attention_layernorm L2={np.linalg.norm(y2):.4f}")
g = Wgate @ y2
u = Wup @ y2
print(f"gate_proj L2={np.linalg.norm(g):.4f}")
print(f"up_proj L2={np.linalg.norm(u):.4f}")

silu = g / (1.0 + np.exp(-g))
mul = silu * u
ffn_out = Wd @ mul
print(f"mlp.down_proj L2={np.linalg.norm(ffn_out):.4f}")

x_final = x_post + ffn_out
print(f"\n[L0 final out] L2={np.linalg.norm(x_final):.4f} first8={x_final[:8].tolist()}")
