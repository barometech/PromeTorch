#!/usr/bin/env python3
"""Загрузка через GGUFTokenizer/Model — пытаемся работать без HF Hub."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import numpy as np
import torch
from transformers import Qwen3ForCausalLM, AutoConfig, AutoTokenizer

gguf_path = sys.argv[1]
# Создаём минимальный config для qwen3-4b
config_dict = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
    "use_cache": True,
    "attention_bias": False,
    "use_qk_norm": True,
    "torch_dtype": "float32",
}
import tempfile
tmpdir = tempfile.mkdtemp()
with open(f"{tmpdir}/config.json", "w") as f:
    json.dump(config_dict, f)

print("[load] loading from gguf...", flush=True)
config = AutoConfig.from_pretrained(tmpdir)
print(f"[config] {config.model_type}, n_layer={config.num_hidden_layers}", flush=True)
model = Qwen3ForCausalLM.from_pretrained(
    tmpdir, gguf_file=gguf_path,
    torch_dtype=torch.float32,
    device_map="cpu",
)
print("[load] done", flush=True)

# Manual token: 'Hello' = 9707
tokens = torch.tensor([[9707]])
with torch.no_grad():
    out = model(tokens, output_hidden_states=True)

hs = out.hidden_states
print(f"\nhidden_states list len={len(hs)}")
for i in [0, 1, 5, 10, 35, 36]:
    if i < len(hs):
        v = hs[i][0, -1].numpy()
        print(f"  L{i}: L2={np.linalg.norm(v):.4f}  first 8={v[:8].tolist()}")

logits = out.logits[0, -1].numpy()
top = np.argsort(-logits)[:5]
print(f"\n[top-5 logits]:")
for tid in top:
    print(f"  id={tid:6d} logit={logits[tid]:>8.4f}")
