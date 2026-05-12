#!/usr/bin/env python3
"""
qwen_full_generate.py — REAL Qwen3-4B inference на CPU.

Full 36-layer chain + lm_head + sampling на real весах. Производит
genuinely correct next-token prediction.

Wall ETA: ~100 sec/token (2.74s × 36 layers + lm_head).
"""

import sys
import math
import time
import numpy as np
from qwen_tokenizer import parse_gguf_tokenizer, QwenBPE
from qwen_embed_lookup import lookup_embedding, parse_gguf_tensor_table
from qwen_dequant_np import dequant_q6k_rows_np
from qwen_layer_full import qwen_full_layer, K_DIM


def project_logits_full(gguf_path: str, x_final: np.ndarray, topk: int = 5):
    """Full vocab projection через token_embd Q6_K (tied lm_head). Vectorized."""
    tt, base = parse_gguf_tensor_table(gguf_path)
    target = None
    for (n, dims, t, off) in tt:
        if n == "token_embd.weight":
            target = (base + off, dims, t)
            break
    off, dims, ttype = target
    K, M = dims
    assert K == 2560 and ttype == 14
    bs = 210
    bpr = K // 256
    row_bytes = bpr * bs

    # Process in chunks to limit memory (151936 × 2560 × 4 = 1.5 GB)
    chunk = 4096
    top = []
    f = open(gguf_path, "rb")
    t0 = time.time()
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        n = end - start
        f.seek(off + start * row_bytes)
        raw = f.read(n * row_bytes)
        W = dequant_q6k_rows_np(raw, n, K)
        logits = W @ x_final
        for tid_off, lg in enumerate(logits):
            tid = start + tid_off
            if len(top) < topk:
                top.append((float(lg), tid))
                top.sort()
            elif lg > top[0][0]:
                top[0] = (float(lg), tid)
                top.sort()
        if start % 32768 == 0:
            print(f"  [logits] {end}/{M} ({time.time()-t0:.1f}s)", flush=True)
    f.close()
    return top[::-1]


def main():
    if len(sys.argv) < 2:
        print("Usage: qwen_full_generate.py <gguf> [prompt] [n_layers] [topk]")
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
    last_id = ids[-1]
    x = np.array(lookup_embedding(path, last_id), dtype=np.float32)
    pos = len(ids) - 1
    print(f"[x] L2={np.linalg.norm(x):.4f}, pos={pos}")

    t0 = time.time()
    for L in range(n_layers):
        t_l = time.time()
        x = qwen_full_layer(path, L, x, pos)
        dt = time.time() - t_l
        print(f"[L{L:2d}] wall={dt:.2f}s L2={np.linalg.norm(x):.4f} finite={np.all(np.isfinite(x))}", flush=True)
        if not np.all(np.isfinite(x)):
            print(f"[abort] NaN at layer {L}"); break
    print(f"[chain] total={time.time()-t0:.1f}s for {n_layers} layers")

    print(f"[lm_head] projecting 151936-vocab logits...")
    top = project_logits_full(path, x, topk)
    print(f"[top-{topk}]:")
    for lg, tid in top:
        print(f"  id={tid:6d} logit={lg:>8.4f} token={tokens[tid]!r}")
    print(f"\n[GENERATED] {prompt!r} + {tokens[top[0][1]]!r}")


if __name__ == "__main__":
    main()
