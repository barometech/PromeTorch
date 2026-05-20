#!/usr/bin/env python3
"""Dump GGUF metadata: rope, eps, head_count, etc."""
import sys
from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
keys_of_interest = (
    "eps", "rope", "freq", "head", "embed", "context",
    "block_count", "general.architecture", "qwen3", "rms",
)
for k in r.fields:
    f = r.fields[k]
    if not any(s in k for s in keys_of_interest):
        continue
    if not f.data:
        print(f"{k}: <no data>")
        continue
    try:
        v = f.parts[f.data[0]]
        s = str(v)
        if hasattr(v, "__len__") and len(v) == 1:
            s = str(v[0])
        if len(s) > 80:
            s = s[:80] + "..."
        print(f"{k}: {s}")
    except Exception as e:
        print(f"{k}: <err: {e}>")
