#!/usr/bin/env python3
"""Run Qwen3-4B through HuggingFace transformers (loaded from GGUF) и
дамп hidden state после L0 для prompt = single token 'Hello'."""
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

gguf_path = sys.argv[1]
print(f"[load] loading {gguf_path}", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", gguf_file=gguf_path,
    torch_dtype=torch.float32, device_map="cpu",
    output_hidden_states=True
)
print(f"[load] done", flush=True)

prompt = "Hello"
tokens = tokenizer(prompt, return_tensors="pt").input_ids
print(f"[tokens] {tokens.tolist()}")

with torch.no_grad():
    out = model(tokens, output_hidden_states=True)
hs = out.hidden_states  # list of (1, seq, hidden)
print(f"\n[hidden_states] {len(hs)} layers")
for i in [0, 1, 5, 10, 35]:
    if i < len(hs):
        v = hs[i][0, -1].numpy()
        print(f"  L{i}: L2={np.linalg.norm(v):.4f}  first 8={v[:8].tolist()}")

logits = out.logits[0, -1].numpy()
top = np.argsort(-logits)[:5]
print(f"\n[top-5 logits]:")
for tid in top:
    print(f"  id={tid:6d} logit={logits[tid]:>8.4f} token={tokenizer.decode([tid])!r}")
