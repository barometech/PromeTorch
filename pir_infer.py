"""PIR inference from fused checkpoint using SentencePiece + PyTorch.

Usage: python3 pir_infer.py CKPT.bin TOKENIZER.model [prompt1 prompt2 ...]

Loads a fused-trainer binary checkpoint and runs autoregressive generation
with incremental PIR state caching (h_prev per block per PIR-layer).
"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

if len(sys.argv) < 3:
    print("usage: python3 pir_infer.py CKPT TOKENIZER [prompts...]")
    sys.exit(1)

CKPT = sys.argv[1]
TOK = sys.argv[2]
PROMPTS = sys.argv[3:] if len(sys.argv) > 3 else [
    "Россия",
    "В начале",
    "Главный",
    "Один из",
]

V = 100000
D = 768
L = 16
NP = 4
T_MAX = 2048
FFN_MULT = 3.5
H = int(((D * FFN_MULT * 2.0 / 3.0) + 63) // 64) * 64  # = 1792
DECAY_MIN = [0.80, 0.95, 0.99, 0.998]
DECAY_MAX = [0.95, 0.99, 0.998, 0.9995]
MAX_TOKENS = 40
TEMP = 0.8
SEED = 42

# ===== load checkpoint =====
print(f"Loading checkpoint {CKPT} ({os.path.getsize(CKPT) / 1e6:.1f} MB)...", flush=True)
fp = open(CKPT, "rb")


def read(n_floats, shape=None):
    b = fp.read(n_floats * 4)
    arr = np.frombuffer(b, dtype=np.float32).copy()
    t = torch.from_numpy(arr)
    if shape is not None:
        t = t.view(*shape)
    return t


W_emb = read(V * D, (V, D))
blocks = []
for l in range(L):
    bw = {"norm1_w": read(D), "pir": []}
    for p in range(NP):
        pw = {
            "W_gate": read(D * D, (D, D)),
            "W_value": read(D * D, (D, D)),
            "W_out": read(D * D, (D, D)),
            "norm_w": read(D),
        }
        t = torch.linspace(0.0, 1.0, D)
        idx = p % 4
        pw["base_decay"] = DECAY_MIN[idx] + t * (DECAY_MAX[idx] - DECAY_MIN[idx])
        bw["pir"].append(pw)
    bw["W_mix"] = read(D * D, (D, D))
    bw["norm_pir_w"] = read(D)
    bw["norm2_w"] = read(D)
    bw["W_ffn1"] = read(H * D, (H, D))
    bw["W_ffn2"] = read(D * H, (D, H))
    bw["W_ffn3"] = read(H * D, (H, D))
    blocks.append(bw)

norm_out_w = read(D)
W_lm_head = read(V * D, (V, D))

rem = fp.read()
fp.close()
print(f"Loaded. Trailing bytes: {len(rem)} (should be 0)", flush=True)


def rmsnorm(x, w, eps=1e-5):
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x / rms * w


def pir_scan_all(scan_gates, gated_values):
    """Serial scan over seq. gates, xs: [T, D] -> h: [T, D]."""
    T_, _ = scan_gates.shape
    h_prev = torch.zeros(D, dtype=scan_gates.dtype)
    out = torch.empty_like(gated_values)
    for t in range(T_):
        h_prev = scan_gates[t] * h_prev + gated_values[t]
        out[t] = h_prev
    return out, h_prev


def forward_seq(ids, cache=None):
    """Run forward on sequence ids, return last-position logits and fresh cache.

    cache: list of L*NP h_prev vectors (D,) or None for fresh state."""
    if cache is None:
        cache = [torch.zeros(D) for _ in range(L * NP)]
    x = W_emb[ids]  # [T, D]
    new_cache = []
    for l, bw in enumerate(blocks):
        normed = rmsnorm(x, bw["norm1_w"])
        pir_res = normed.clone()
        for pi, pw in enumerate(bw["pir"]):
            gate = normed @ pw["W_gate"].T
            value = normed @ pw["W_value"].T
            sig = torch.sigmoid(gate)
            gated = sig * value
            scan_gates = sig * pw["base_decay"]
            h_prev = cache[l * NP + pi]
            seq_len = gated.shape[0]
            if seq_len == 1:
                h_new = scan_gates[0] * h_prev + gated[0]
                h_out = h_new.unsqueeze(0)
                new_h = h_new
            else:
                h_out, new_h = pir_scan_all(scan_gates, gated)
                # add h_prev via first step already? no — above assumes h_prev=0.
                # Re-run first step with actual h_prev:
                # Simpler: if cache had nonzero state, process in single go with h_prev seeded
                pass
            new_cache.append(new_h)
            out_proj = h_out @ pw["W_out"].T
            out_normed = rmsnorm(out_proj, pw["norm_w"])
            pir_res = pir_res + out_normed
        mix = pir_res @ bw["W_mix"].T
        mix_normed = rmsnorm(mix, bw["norm_pir_w"])
        x = x + mix_normed
        n2 = rmsnorm(x, bw["norm2_w"])
        ffn1 = n2 @ bw["W_ffn1"].T
        ffn3 = n2 @ bw["W_ffn3"].T
        gated_ffn = F.silu(ffn1) * ffn3
        ffn2 = gated_ffn @ bw["W_ffn2"].T
        x = x + ffn2
    x = rmsnorm(x, norm_out_w)
    logits_last = x[-1] @ W_lm_head.T  # [V]
    return logits_last, new_cache


def pir_scan_seeded(scan_gates, gated_values, h_prev):
    T_, _ = scan_gates.shape
    out = torch.empty_like(gated_values)
    h = h_prev
    for t in range(T_):
        h = scan_gates[t] * h + gated_values[t]
        out[t] = h
    return out, h


def forward_seq_seeded(ids, cache):
    """Forward that threads h_prev from cache through scan (proper seeding)."""
    x = W_emb[ids]
    new_cache = []
    for l, bw in enumerate(blocks):
        normed = rmsnorm(x, bw["norm1_w"])
        pir_res = normed.clone()
        for pi, pw in enumerate(bw["pir"]):
            gate = normed @ pw["W_gate"].T
            value = normed @ pw["W_value"].T
            sig = torch.sigmoid(gate)
            gated = sig * value
            scan_gates = sig * pw["base_decay"]
            h_prev = cache[l * NP + pi]
            h_out, h_new = pir_scan_seeded(scan_gates, gated, h_prev)
            new_cache.append(h_new)
            out_proj = h_out @ pw["W_out"].T
            out_normed = rmsnorm(out_proj, pw["norm_w"])
            pir_res = pir_res + out_normed
        mix = pir_res @ bw["W_mix"].T
        mix_normed = rmsnorm(mix, bw["norm_pir_w"])
        x = x + mix_normed
        n2 = rmsnorm(x, bw["norm2_w"])
        ffn1 = n2 @ bw["W_ffn1"].T
        ffn3 = n2 @ bw["W_ffn3"].T
        gated_ffn = F.silu(ffn1) * ffn3
        ffn2 = gated_ffn @ bw["W_ffn2"].T
        x = x + ffn2
    x = rmsnorm(x, norm_out_w)
    logits_last = x[-1] @ W_lm_head.T
    return logits_last, new_cache


# ===== tokenize + generate =====
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(TOK)
print(f"Tokenizer: vocab={sp.GetPieceSize()}", flush=True)

torch.manual_seed(SEED)
print()
for prompt in PROMPTS:
    prompt_ids = sp.EncodeAsIds(prompt)
    ids = list(prompt_ids)
    # Build initial cache from prompt
    with torch.no_grad():
        logits, cache = forward_seq_seeded(torch.tensor(ids, dtype=torch.long),
                                           [torch.zeros(D) for _ in range(L * NP)])
    # Sample next tokens one at a time
    gen_ids = []
    for t in range(MAX_TOKENS):
        probs = torch.softmax(logits / TEMP, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        gen_ids.append(next_id)
        with torch.no_grad():
            logits, cache = forward_seq_seeded(torch.tensor([next_id], dtype=torch.long), cache)
    full_ids = prompt_ids + gen_ids
    full_text = sp.DecodeIds(full_ids)
    prompt_text = sp.DecodeIds(prompt_ids)
    gen_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else sp.DecodeIds(gen_ids)
    print(f">>> {prompt}{gen_text}", flush=True)
    print()
