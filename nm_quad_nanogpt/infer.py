#!/usr/bin/env python3
"""
Standalone inference for the NM Quad nanoGPT model.
Reads trained_weights.bin saved by host_train_and_gen and runs the same
1-layer transformer forward pass in numpy, generating text.

Usage:
    python3 infer.py [prompt] [n_gen]
"""
import sys
import math
import struct
import numpy as np

VOCAB = 128
T     = 32
D     = 32
FF    = 64

# Auto-detect 1-layer (17600) or 2-layer (25920) checkpoint
PER_LAYER = D*3*D + D*D + D*FF + FF*D + 4*D   # 8320
SHARED    = VOCAB*D + T*D + D*VOCAB + 2*D      # 9280

def load_weights(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    floats = np.frombuffer(raw, dtype=np.float32)
    n = len(floats)
    # determine L
    extra = n - SHARED
    assert extra % PER_LAYER == 0, f"len {n} - shared {SHARED} = {extra} not multiple of {PER_LAYER}"
    L = extra // PER_LAYER
    off = 0
    def take(k):
        nonlocal off
        chunk = floats[off:off + k]
        off += k
        return chunk
    Wtok = take(VOCAB * D).reshape(VOCAB, D).copy()
    Wpos = take(T * D).reshape(T, D).copy()
    layers = []
    for _ in range(L):
        Wqkv = take(D * 3 * D).reshape(D, 3 * D).copy()
        Wout = take(D * D).reshape(D, D).copy()
        Wfc1 = take(D * FF).reshape(D, FF).copy()
        Wfc2 = take(FF * D).reshape(FF, D).copy()
        g1 = take(D).copy(); b1 = take(D).copy()
        g2 = take(D).copy(); b2 = take(D).copy()
        layers.append(dict(Wqkv=Wqkv, Wout=Wout, Wfc1=Wfc1, Wfc2=Wfc2,
                           g1=g1, b1=b1, g2=g2, b2=b2))
    Wunemb = take(D * VOCAB).reshape(D, VOCAB).copy()
    gF = take(D).copy(); bF = take(D).copy()
    assert off == n, (off, n)
    return dict(Wtok=Wtok, Wpos=Wpos, layers=layers, Wunemb=Wunemb, gF=gF, bF=bF, L=L)


def layernorm(x, g, b, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x - mu) * inv * g + b


def layer_fwd_full(x, layer):
    """Full forward over all positions; needed because next layer's attention is causal over all."""
    n = x.shape[0]
    ln1 = layernorm(x, layer["g1"], layer["b1"])
    qkv = ln1 @ layer["Wqkv"]
    Q, K, V = qkv[:, :D], qkv[:, D:2*D], qkv[:, 2*D:]
    scale = 1.0 / math.sqrt(D)
    attn = np.zeros_like(x)
    for t in range(n):
        s = Q[t] @ K[:t+1].T * scale
        s = s - s.max()
        p = np.exp(s); p /= p.sum()
        attn[t] = p @ V[:t+1]
    proj = attn @ layer["Wout"]
    res1 = x + proj
    ln2 = layernorm(res1, layer["g2"], layer["b2"])
    fc1_pre = ln2 @ layer["Wfc1"]
    fc1 = np.maximum(fc1_pre, 0.0)
    ffn = fc1 @ layer["Wfc2"]
    return res1 + ffn

def forward(w, tokens):
    """tokens: list[int] of length n <= T. Returns logits over VOCAB for position n-1."""
    n = len(tokens)
    assert n <= T, n
    x = w["Wtok"][tokens] + w["Wpos"][:n]              # (n, D)
    for layer in w["layers"]:
        x = layer_fwd_full(x, layer)
    ln_final = layernorm(x[-1], w["gF"], w["bF"])
    logits = ln_final @ w["Wunemb"]
    return logits


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time "
    n_gen  = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    temp   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    w = load_weights("trained_weights.bin")
    print(f"# model: L={w['L']} layers, D={D}, T={T}, VOCAB={VOCAB}, FF={FF}")
    print(f"# weights: Wtok mean={w['Wtok'].mean():.4f} std={w['Wtok'].std():.4f}")
    print(f"# weights: Wunemb mean={w['Wunemb'].mean():.4f} std={w['Wunemb'].std():.4f}")
    print(f"# prompt:  {prompt!r}  n_gen={n_gen}  temp={temp}")
    print()
    sys.stdout.write(prompt); sys.stdout.flush()

    ctx = [ord(c) & 0x7F for c in prompt]
    rng = np.random.default_rng(42)
    for _ in range(n_gen):
        window = ctx[-T:] if len(ctx) >= T else ctx
        logits = forward(w, window)
        if temp <= 0.0:
            nxt = int(logits.argmax())
        else:
            p = np.exp((logits - logits.max()) / temp)
            p /= p.sum()
            nxt = int(rng.choice(VOCAB, p=p))
        ctx.append(nxt)
        sys.stdout.write(chr(nxt) if 32 <= nxt < 127 else f"<{nxt}>")
        sys.stdout.flush()
    print()


if __name__ == "__main__":
    main()
