#!/usr/bin/env python3
"""
qwen_generate.py — END-TO-END inference: prompt → next token prediction.

Полный inference flow используя Python reference path:
  1. Tokenize prompt → token IDs
  2. Lookup embedding для последнего token (Q6_K)
  3. Layer chain forward × N layers (subset M_FFN=256)
  4. lm_head projection → logits
  5. argmax → predicted next token ID
  6. Decode → output text

Это первая END-TO-END inference demonstration foundation Qwen 4B на NM Quad.
"""

import sys
import math
import time
import struct
import numpy as np
from qwen_tokenizer import parse_gguf_tokenizer, QwenBPE
from qwen_embed_lookup import lookup_embedding, parse_gguf_tensor_table, dequant_q6k_block
from qwen_python_layer import qwen_layer_forward, K_DIM


def project_logits_top(gguf_path: str, x_final: np.ndarray, topk: int = 5):
    """Quick top-k projection через token_embd (tied lm_head)."""
    tt, base = parse_gguf_tensor_table(gguf_path)
    target = None
    for (n, dims, t, off) in tt:
        if n == "token_embd.weight":
            target = (base + off, dims, t)
            break
    off, dims, ttype = target
    K, M = dims
    assert ttype == 14
    bs = 210
    bpr = K // 256
    row_bytes = bpr * bs
    top = []
    f = open(gguf_path, "rb")
    print(f"[project] projecting through {M} vocab rows (~few min)...", flush=True)
    t0 = time.time()
    for tid in range(M):
        if tid % 30000 == 0:
            print(f"  [progress] {tid}/{M}  ({time.time()-t0:.1f}s)", flush=True)
        f.seek(off + tid * row_bytes)
        row_bytes_data = f.read(row_bytes)
        row = np.zeros(K, dtype=np.float32)
        for b in range(bpr):
            row[b * 256:(b + 1) * 256] = dequant_q6k_block(row_bytes_data[b * bs:(b + 1) * bs])
        logit = float(np.dot(x_final, row))
        if len(top) < topk:
            top.append((logit, tid))
            top.sort()
        elif logit > top[0][0]:
            top[0] = (logit, tid)
            top.sort()
    f.close()
    return top[::-1]


def main():
    if len(sys.argv) < 2:
        print("Usage: qwen_generate.py <gguf> [prompt] [n_layers] [topk]")
        sys.exit(1)
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"[step1] tokenizing prompt {prompt!r}")
    tokens, merges, bos, eos = parse_gguf_tokenizer(path)
    bpe = QwenBPE(tokens, merges, bos, eos)
    ids = bpe.encode(prompt)
    print(f"[step1] tokens = {ids} ({[tokens[i] for i in ids]})")

    last_id = ids[-1]
    print(f"[step2] embedding for last token {last_id} {tokens[last_id]!r}")
    emb = lookup_embedding(path, last_id)
    x = np.array(emb, dtype=np.float32)
    print(f"[step2] embedding L2={np.linalg.norm(x):.4f}")

    print(f"[step3] layer chain forward × {n_layers} layers")
    pos = len(ids) - 1
    t0 = time.time()
    for L in range(n_layers):
        x = qwen_layer_forward(path, L, x, pos)
        print(f"  [layer {L}] L2={np.linalg.norm(x):.4f}")
        if np.isnan(x).any():
            print(f"  [warning] NaN detected at layer {L}, truncating")
            break
    print(f"[step3] chain wall={time.time()-t0:.1f}s")

    print(f"[step4] lm_head projection → logits → top-{topk}")
    top = project_logits_top(path, x, topk)
    print(f"[step4] top {topk} predicted next tokens:")
    for logit, tid in top:
        print(f"  id={tid:6d}  logit={logit:>8.4f}  token={tokens[tid]!r}")

    pred_id = top[0][1]
    print(f"\n[GENERATED] {prompt!r} + {tokens[pred_id]!r}")


if __name__ == "__main__":
    main()
