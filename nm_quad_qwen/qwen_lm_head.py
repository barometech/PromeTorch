#!/usr/bin/env python3
"""
qwen_lm_head.py — final inference step: x_final[2560] → logits[151936] → next token.

В Qwen3-4B output projection обычно tied к token_embd (same Q6_K matrix).
GEMV: logits[i] = sum_k token_embd[i, k] * x_final[k]
Затем argmax/sampling.

Validate using test embedding — feed embedding back through и checking
top tokens correspond к self-similar (token closest to its own embedding).
"""

import sys
import math
import struct
from qwen_embed_lookup import parse_gguf_tensor_table, dequant_q6k_block


def find_output_or_embed(gguf_path: str):
    """Возвращает (name, offset, dims) для lm_head или token_embd."""
    tt, base = parse_gguf_tensor_table(gguf_path)
    out = None
    embd = None
    for (name, dims, ttype, off) in tt:
        if name == "output.weight":
            out = (name, base + off, dims, ttype)
        elif name == "token_embd.weight":
            embd = (name, base + off, dims, ttype)
    return out, embd


def dequant_row_q6k(f, file_offset: int, row_idx: int, K: int) -> list:
    """Извлекает row row_idx из Q6_K matrix (K = row length)."""
    block_size = 210
    blocks_per_row = K // 256
    row_bytes = blocks_per_row * block_size
    f.seek(file_offset + row_idx * row_bytes)
    row = f.read(row_bytes)
    out = []
    for b in range(blocks_per_row):
        out.extend(dequant_q6k_block(row[b * block_size:(b + 1) * block_size]))
    return out


def project_logits_topk(gguf_path: str, x_final: list, topk: int = 10):
    """Project x_final[2560] → logits[151936], return top K (token_id, logit)."""
    out_t, embd_t = find_output_or_embed(gguf_path)
    target = out_t if out_t else embd_t
    if not target:
        raise RuntimeError("neither output.weight nor token_embd.weight found")
    name, off, dims, ttype = target
    print(f"[lm_head] using {name!r} {dims} type={ttype}")
    K, M = dims     # K=2560 input, M=151936 vocab
    assert K == 2560
    assert ttype == 14, f"expected Q6_K, got {ttype}"

    f = open(gguf_path, "rb")
    top = []  # (logit, token_id) list
    for tid in range(M):
        if tid % 20000 == 0:
            print(f"  [progress] {tid}/{M}")
        row = dequant_row_q6k(f, off, tid, K)
        logit = sum(x_final[i] * row[i] for i in range(K))
        if len(top) < topk:
            top.append((logit, tid))
            top.sort()
        elif logit > top[0][0]:
            top[0] = (logit, tid)
            top.sort()
    f.close()
    return top[::-1]   # high → low


def main():
    if len(sys.argv) < 3:
        print("Usage: qwen_lm_head.py <gguf_path> <input.bin> [topk]")
        print("  input.bin: int32 count + count*2560 fp32 (от qwen_input_pipeline.py)")
        sys.exit(1)
    path = sys.argv[1]
    in_path = sys.argv[2]
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    with open(in_path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        x = list(struct.unpack(f"<{2560}f", f.read(2560 * 4)))
    print(f"[load] reading first of {count} embeddings as x[2560], L2={math.sqrt(sum(v*v for v in x)):.4f}")

    top = project_logits_topk(path, x, topk)
    print(f"[top-{topk}] tokens by logit (raw embedding × token_embd):")
    for logit, tid in top:
        print(f"  {tid:6d}  logit={logit:.4f}")


if __name__ == "__main__":
    main()
