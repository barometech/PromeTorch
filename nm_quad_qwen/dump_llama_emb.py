#!/usr/bin/env python3
"""Get embedding via llama-cpp-python — это hidden state с pooling.
Используем как cross-check."""
import sys
import numpy as np
from llama_cpp import Llama

path = sys.argv[1]
prompts = ["Once upon a time", "Hello"]

llm = Llama(model_path=path, n_ctx=512, n_threads=4, logits_all=True,
            verbose=False, embedding=True, pooling_type=2)  # MEAN pool

for p in prompts:
    emb = llm.embed(p)
    arr = np.array(emb)
    print(f"prompt={p!r} emb_shape={arr.shape} L2={np.linalg.norm(arr):.4f}")
    print(f"  first 8: {arr[:8].tolist()}")
