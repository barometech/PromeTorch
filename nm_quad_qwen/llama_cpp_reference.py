#!/usr/bin/env python3
"""Run llama-cpp-python on Qwen3-4B and print top-5 next token for prompts.
Compare with our Python inference."""
import sys
from llama_cpp import Llama

path = sys.argv[1] if len(sys.argv) > 1 else "/home/elbrus-user/gguf/qwen3-4b-q4km.gguf"
prompts = ["Once upon a time", "The capital of France is", "Hello"]

llm = Llama(model_path=path, n_ctx=512, n_threads=4, logits_all=True,
            verbose=False, embedding=False)

for p in prompts:
    print(f"\n=== prompt: {p!r} ===")
    tokens = llm.tokenize(p.encode("utf-8"), add_bos=False)
    print(f"tokens={tokens}")
    llm.reset()
    llm.eval(tokens)
    # logits для последнего токена
    logits = llm.eval_logits[-1]
    # top-5
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])[:5]
    for tid, lg in indexed:
        tok_bytes = llm.detokenize([tid])
        try:
            tok = tok_bytes.decode("utf-8", errors="replace")
        except:
            tok = repr(tok_bytes)
        print(f"  id={tid:6d} logit={lg:>8.3f} token={tok!r}")
