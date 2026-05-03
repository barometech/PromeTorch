#!/usr/bin/env python3
"""Stripped-down GGUF metadata reader. No external deps. Reads key/value
metadata only (skips tensor info).
"""
import struct, sys

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")

def read_val(f, t):
    if t == 0:  return struct.unpack("<B", f.read(1))[0]
    if t == 1:  return struct.unpack("<b", f.read(1))[0]
    if t == 2:  return struct.unpack("<H", f.read(2))[0]
    if t == 3:  return struct.unpack("<h", f.read(2))[0]
    if t == 4:  return struct.unpack("<I", f.read(4))[0]
    if t == 5:  return struct.unpack("<i", f.read(4))[0]
    if t == 6:  return struct.unpack("<f", f.read(4))[0]
    if t == 7:  return struct.unpack("<B", f.read(1))[0]
    if t == 8:  return read_str(f)
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    if t == 9:
        sub = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        out = []
        for _ in range(n):
            out.append(read_val(f, sub))
        return out
    raise ValueError("unknown type %d" % t)

def dump(path, filter_keys=None):
    print("=== %s" % path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print("  NOT GGUF: magic=%r" % magic); return
        ver = struct.unpack("<I", f.read(4))[0]
        n_t = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print("  version=%d tensors=%d kv=%d" % (ver, n_t, n_kv))
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            v = read_val(f, t)
            if filter_keys is None or any(s in k for s in filter_keys):
                if isinstance(v, list) and len(v) > 8:
                    v = v[:8] + ["..."]
                if isinstance(v, str) and len(v) > 200:
                    v = v[:200] + "..."
                print("  %s = %r" % (k, v))

if __name__ == "__main__":
    keys = ["architecture", "head_count", "key_length", "value_length",
            "rope.", "embedding_length", "block_count", "feed_forward",
            "context_length", "rms_eps", "epsilon", "vocab_size",
            "tokenizer.ggml.model", "general.name", "tie_embed",
            "expert_count", "sliding_window", "attention.causal",
            "use_parallel"]
    for path in sys.argv[1:]:
        dump(path, keys)
        print()
