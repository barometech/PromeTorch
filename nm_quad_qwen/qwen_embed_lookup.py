#!/usr/bin/env python3
"""
qwen_embed_lookup.py — extract token embedding (Q6_K row) для одного token ID
из Qwen3-4B GGUF. Это первая операция inference: tokens → embeddings.

token_embd.weight [2560, 151936] Q6_K — каждый из 151936 vocab tokens имеет
2560-fp32 embedding. На NMC4 в layer pipeline embedding идёт как x[2560]
input.
"""

import struct
import sys
import math
from typing import List


def fp16_to_fp32(h: int) -> float:
    sign = (h & 0x8000) >> 15
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        return ((-1.0) ** sign) * (mant / 1024.0) * (2.0 ** -14)
    if exp == 0x1F:
        return float("nan") if mant else (float("-inf") if sign else float("inf"))
    return ((-1.0) ** sign) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def dequant_q6k_block(blk: bytes) -> List[float]:
    """Decode 256 fp32 values из 210-byte Q6_K block."""
    ql = blk[0:128]
    qh = blk[128:192]
    sc = blk[192:208]
    d_bits = ql[0]  # not used here
    d_bits = (blk[208]) | (blk[209] << 8)
    d = fp16_to_fp32(d_bits)
    out = [0.0] * 256
    for i in range(256):
        is_ = i // 16
        ql_idx = (i % 64) + 64 * (i // 128)
        ql_shift = 4 * ((i // 32) & 1)
        q_lo = (ql[ql_idx] >> ql_shift) & 0xF
        qh_idx = (i % 32) + 32 * (i // 128)
        qh_shift = 2 * ((i // 16) & 3)
        q_hi = (qh[qh_idx] >> qh_shift) & 0x3
        scv = sc[is_]
        if scv >= 128: scv -= 256
        out[i] = d * scv * ((q_lo | (q_hi << 4)) - 32)
    return out


def parse_gguf_tensor_table(path: str):
    """Возвращает (kv, tensor_table, data_offset)."""
    f = open(path, "rb")
    def rd(fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    def rstr(): n = rd("<Q"); return f.read(n).decode(errors="replace")
    def skip(t):
        if t in (0, 1, 7): f.seek(1, 1)
        elif t in (2, 3): f.seek(2, 1)
        elif t in (4, 5, 6): f.seek(4, 1)
        elif t in (10, 11, 12): f.seek(8, 1)
        elif t == 8: rstr()
        elif t == 9:
            at = rd("<I"); an = rd("<Q")
            for _ in range(an): skip(at)
    assert f.read(4) == b"GGUF"
    ver = rd("<I"); nt = rd("<Q"); nk = rd("<Q")
    for _ in range(nk):
        k = rstr(); t = rd("<I"); skip(t)
    tt = []
    for _ in range(nt):
        n = rstr(); nd = rd("<I")
        dims = [rd("<Q") for _ in range(nd)]
        ttype = rd("<I"); off = rd("<Q")
        tt.append((n, dims, ttype, off))
    pos = f.tell()
    aln = 32
    base = (pos + aln - 1) & ~(aln - 1)
    f.close()
    return tt, base


def lookup_embedding(path: str, token_id: int) -> List[float]:
    """Извлекает embedding [2560] для token_id из token_embd.weight (Q6_K)."""
    tt, base = parse_gguf_tensor_table(path)
    for (name, dims, ttype, off) in tt:
        if name == "token_embd.weight":
            assert ttype == 14, f"expected Q6_K type=14, got {ttype}"
            K, M = dims  # [2560, 151936]
            assert K == 2560 and M == 151936
            # Q6_K rows: каждый row = K bytes weights = 2560/256 = 10 blocks
            block_size = 210
            blocks_per_row = K // 256  # 10
            row_bytes = blocks_per_row * block_size
            with open(path, "rb") as f:
                f.seek(base + off + token_id * row_bytes)
                row = f.read(row_bytes)
            out = []
            for b in range(blocks_per_row):
                out.extend(dequant_q6k_block(row[b * block_size:(b + 1) * block_size]))
            return out
    raise RuntimeError("token_embd.weight not found")


def main():
    if len(sys.argv) < 3:
        print("Usage: qwen_embed_lookup.py <gguf_path> <token_id>")
        sys.exit(1)
    path = sys.argv[1]
    token_id = int(sys.argv[2])

    print(f"[load] tensor table from {path}")
    print(f"[lookup] token_id = {token_id}")
    emb = lookup_embedding(path, token_id)
    print(f"[lookup] embedding[{len(emb)}] for token {token_id}:")
    print(f"  first 8 = {emb[:8]}")
    print(f"  last 8  = {emb[-8:]}")
    s = sum(emb)
    s2 = sum(e * e for e in emb)
    print(f"  mean = {s / len(emb):.6f}")
    print(f"  L2   = {math.sqrt(s2):.6f}")
    print(f"  rms  = {math.sqrt(s2 / len(emb)):.6f}")


if __name__ == "__main__":
    main()
