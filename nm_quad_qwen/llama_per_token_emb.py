#!/usr/bin/env python3
"""Get per-token last hidden state through llama-cpp-python с pooling=NONE."""
import sys
import numpy as np
from llama_cpp import Llama

path = sys.argv[1]
llm = Llama(model_path=path, n_ctx=512, n_threads=4,
            embedding=True, pooling_type=0,  # NONE → per-token
            verbose=False)

for p in ["Hello", "Once upon a time"]:
    print(f"\n=== {p!r} ===")
    res = llm.create_embedding(p)
    embs = res["data"][0]["embedding"]
    arr = np.array(embs, dtype=np.float32)
    if arr.ndim == 1:
        # single emb (pooled or single token)
        print(f"  shape={arr.shape} L2={np.linalg.norm(arr):.4f}")
        print(f"  first 8={arr[:8].tolist()}")
    else:
        # per token
        print(f"  shape={arr.shape}")
        for i in range(min(len(arr), 4)):
            v = arr[i]
            print(f"  token[{i}]: L2={np.linalg.norm(v):.4f} first 8={v[:8].tolist()}")
