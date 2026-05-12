#!/usr/bin/env python3
"""
qwen_tokenizer.py — Qwen3 BPE tokenizer прямо из GGUF.

Extracts tokenizer.ggml.tokens + .merges + .bos_token_id + .eos_token_id
из Qwen3-4B Q4_K_M GGUF и реализует BPE encode для prompt → token IDs.

Используется для генерации input token sequence для NMC4 inference pipeline.
"""

import struct
import sys
from typing import List, Tuple


def parse_gguf_tokenizer(path: str):
    """Парсит GGUF metadata, возвращает (tokens, merges, bos_id, eos_id)."""
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

    tokens = []
    merges = []
    bos_id = None
    eos_id = None
    for _ in range(nk):
        k = rstr(); t = rd("<I")
        if k == "tokenizer.ggml.tokens" and t == 9:
            at = rd("<I"); an = rd("<Q")
            assert at == 8, f"expected string array, got type {at}"
            tokens = [rstr() for _ in range(an)]
        elif k == "tokenizer.ggml.merges" and t == 9:
            at = rd("<I"); an = rd("<Q")
            assert at == 8
            merges = [rstr() for _ in range(an)]
        elif k == "tokenizer.ggml.bos_token_id":
            assert t == 4
            bos_id = rd("<I")
        elif k == "tokenizer.ggml.eos_token_id":
            assert t == 4
            eos_id = rd("<I")
        else:
            skip(t)
    f.close()
    return tokens, merges, bos_id, eos_id


def bytes_to_unicode():
    """GPT-2 style byte-to-unicode mapping."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class QwenBPE:
    def __init__(self, tokens, merges, bos_id, eos_id):
        self.tokens = tokens
        self.id_of = {t: i for i, t in enumerate(tokens)}
        self.merges = {}
        for i, m in enumerate(merges):
            a, b = m.split(" ", 1)
            self.merges[(a, b)] = i
        self.bos = bos_id
        self.eos = eos_id
        self.byte2u = bytes_to_unicode()

    def _bpe(self, word_chars: List[str]) -> List[str]:
        word = list(word_chars)
        while len(word) > 1:
            best = None
            best_rank = float("inf")
            for i in range(len(word) - 1):
                rank = self.merges.get((word[i], word[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best = i
            if best is None:
                break
            word = word[:best] + [word[best] + word[best + 1]] + word[best + 2:]
        return word

    def encode(self, text: str) -> List[int]:
        """Простая BPE без pre-tokenizer regex. Для production нужен полный
        Qwen pre-tokenizer."""
        # Byte-level encode + GPT-2 unicode map
        bs = text.encode("utf-8")
        chars = "".join(self.byte2u[b] for b in bs)
        # Single-word BPE (без word-splitting — упрощение)
        word = list(chars)
        merged = self._bpe(word)
        ids = []
        for piece in merged:
            tid = self.id_of.get(piece)
            if tid is None:
                # fallback: encode each char as token if exists
                for ch in piece:
                    cid = self.id_of.get(ch)
                    if cid is not None:
                        ids.append(cid)
            else:
                ids.append(tid)
        return ids

    def decode(self, ids: List[int]) -> str:
        """ID array → text."""
        u2b = {v: k for k, v in self.byte2u.items()}
        s = "".join(self.tokens[i] for i in ids if 0 <= i < len(self.tokens))
        bs = bytearray()
        for ch in s:
            if ch in u2b:
                bs.append(u2b[ch])
            else:
                bs.extend(ch.encode("utf-8"))
        return bs.decode(errors="replace")


def main():
    if len(sys.argv) < 2:
        print("Usage: qwen_tokenizer.py <gguf_path> [prompt]")
        sys.exit(1)
    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"

    print(f"[load] parsing GGUF tokenizer from {path}")
    tokens, merges, bos, eos = parse_gguf_tokenizer(path)
    print(f"[load] {len(tokens)} tokens, {len(merges)} merges, bos={bos} eos={eos}")

    bpe = QwenBPE(tokens, merges, bos, eos)
    ids = bpe.encode(prompt)
    print(f"[encode] {prompt!r} -> {len(ids)} tokens: {ids}")
    print("[encode] piece-by-piece:")
    for i in ids:
        print(f"    {i}: {tokens[i]!r}")

    # Roundtrip
    roundtrip = bpe.decode(ids)
    print(f"[decode] {roundtrip!r}")


if __name__ == "__main__":
    main()
