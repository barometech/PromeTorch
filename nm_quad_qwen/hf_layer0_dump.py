#!/usr/bin/env python3
"""Dump HF Qwen3 intermediate values within layer 0 for "Hello"."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys, json, tempfile, numpy as np, torch
from transformers import Qwen3ForCausalLM, AutoConfig

gguf_path = sys.argv[1]
tmpdir = tempfile.mkdtemp()
with open(f"{tmpdir}/config.json", "w") as f:
    json.dump({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2560, "intermediate_size": 9728,
        "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "max_position_embeddings": 32768,
        "rope_theta": 1000000.0, "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True, "hidden_act": "silu",
        "attention_bias": False, "use_qk_norm": True,
        "torch_dtype": "float32",
    }, f)

print("[load]", flush=True)
model = Qwen3ForCausalLM.from_pretrained(
    tmpdir, gguf_file=gguf_path, torch_dtype=torch.float32, device_map="cpu")

# Hook into layer 0 sub-modules to capture intermediate
layer0 = model.model.layers[0]
print(f"[layer 0 modules]: {[n for n,_ in layer0.named_children()]}")

captured = {}
def make_hook(name):
    def hk(mod, inputs, output):
        if isinstance(output, tuple):
            captured[name] = output[0].detach().cpu().numpy()
        else:
            captured[name] = output.detach().cpu().numpy()
    return hk

# hooks
hooks = []
for n, m in layer0.named_modules():
    if n in ("input_layernorm", "post_attention_layernorm", "self_attn",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
            "mlp", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        hooks.append(m.register_forward_hook(make_hook(n)))

tokens = torch.tensor([[9707]])  # 'Hello'
with torch.no_grad():
    out = model(tokens, output_hidden_states=True)

for n, arr in sorted(captured.items()):
    flat = arr.reshape(-1) if arr.ndim > 1 else arr
    print(f"{n}: shape={arr.shape} L2={np.linalg.norm(flat):.4f} first8={flat[:8].tolist()}")

l0 = out.hidden_states[1][0,-1].numpy()
print(f"\n[L0 final out] L2={np.linalg.norm(l0):.4f} first8={l0[:8].tolist()}")
