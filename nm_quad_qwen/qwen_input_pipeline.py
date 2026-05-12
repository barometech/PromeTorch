#!/usr/bin/env python3
"""
qwen_input_pipeline.py — END-TO-END preprocessing для Qwen3-4B inference:
  prompt → tokenize → embed → [token_count][2560] fp32 array

Это вход для NMC4 layer pipeline (каждый embedding идёт через 36 layers).
Сохраняет input embeddings в binary file для дальнейшего использования
host_qwen_full_layer.cpp.
"""

import sys
import struct
import math
from qwen_tokenizer import parse_gguf_tokenizer, QwenBPE
from qwen_embed_lookup import lookup_embedding


def encode_prompt_to_embeddings(gguf_path: str, prompt: str):
    """Returns (token_ids, embeddings_2d_list)."""
    tokens, merges, bos, eos = parse_gguf_tokenizer(gguf_path)
    bpe = QwenBPE(tokens, merges, bos, eos)
    ids = bpe.encode(prompt)
    embs = [lookup_embedding(gguf_path, tid) for tid in ids]
    return ids, embs, tokens


def main():
    if len(sys.argv) < 2:
        print("Usage: qwen_input_pipeline.py <gguf_path> [prompt] [out.bin]")
        sys.exit(1)
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/qwen_input_embeddings.bin"

    print(f"[prompt] {prompt!r}")
    ids, embs, tokens = encode_prompt_to_embeddings(path, prompt)
    print(f"[tokens] {len(ids)} tokens: {ids}")
    for i, tid in enumerate(ids):
        print(f"  [{i}] {tid:6d} {tokens[tid]!r}")

    print(f"[embed] each token → 2560 fp32 embedding")
    for i, e in enumerate(embs):
        s2 = sum(x * x for x in e)
        print(f"  token[{i}] L2={math.sqrt(s2):.4f}  first 4=[{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}, {e[3]:.4f}]")

    # Save flat binary: token_count int32, then token_count × 2560 fp32
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for e in embs:
            for v in e:
                f.write(struct.pack("<f", v))
    total = 4 + len(ids) * 2560 * 4
    print(f"[save] {out_path} ({total} bytes)")
    print(f"[ready] embeddings готовы для NMC4 layer pipeline")
    print(f"        host_qwen_full_layer.cpp может загружать /tmp/qwen_input_embeddings.bin")
    print(f"        и подавать каждый embedding в layer kernel for 36 layers")


if __name__ == "__main__":
    main()
