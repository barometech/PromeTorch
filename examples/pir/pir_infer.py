#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIR fused-checkpoint inference + BPE decode (numpy).
Полностью повторяет forward из examples/pir/fused_trainer.h::generate_text.
Развязан от C++ сборки: читает сырой float32-дамп all_params + декодит через sentencepiece.

Формат чекпоинта (последовательный float32-дамп, порядок reg() в allocate()):
  W_emb (V,D)
  per layer L:  norm1_w(D); per p in NP:[W_gate(D,D),W_value(D,D),W_out(D,D),norm_w(D)];
                W_mix(D,D); norm_pir_w(D); norm2_w(D); W_ffn1(H,D); W_ffn2(D,H); W_ffn3(H,D)
  norm_out_w (D); W_lm_head (V,D)
base_decay НЕ сохраняется — реконструируется из формулы init.
"""
import sys, argparse, numpy as np

def build_cfg(a):
    D = a.n_embd
    H = ((int(D * a.ffn_mult * 2.0 / 3.0) + 63) // 64) * 64
    return dict(V=a.vocab_size, D=D, L=a.n_layers, NP=a.n_pir_layers, T=a.block_size,
                H=H,
                decay_min=[0.80, 0.95, 0.99, 0.998],
                decay_max=[0.95, 0.99, 0.998, 0.9995])

def load_ckpt(path, cfg):
    V, D, L, NP, H = cfg['V'], cfg['D'], cfg['L'], cfg['NP'], cfg['H']
    raw = np.fromfile(path, dtype=np.float32)
    off = [0]
    def take(*shape):
        n = int(np.prod(shape))
        arr = raw[off[0]:off[0]+n].reshape(shape)
        off[0] += n
        return arr
    W = {}
    W['W_emb'] = take(V, D)
    W['layers'] = []
    for l in range(L):
        ly = {}
        ly['norm1_w'] = take(D)
        ly['pir'] = []
        for p in range(NP):
            pw = {}
            pw['W_gate']  = take(D, D)
            pw['W_value'] = take(D, D)
            pw['W_out']   = take(D, D)
            pw['norm_w']  = take(D)
            ly['pir'].append(pw)
        ly['W_mix']      = take(D, D)
        ly['norm_pir_w'] = take(D)
        ly['norm2_w']    = take(D)
        ly['W_ffn1']     = take(H, D)
        ly['W_ffn2']     = take(D, H)
        ly['W_ffn3']     = take(H, D)
        W['layers'].append(ly)
    W['norm_out_w'] = take(D)
    W['W_lm_head']  = take(V, D)
    used, total = off[0], raw.size
    if used != total:
        sys.stderr.write(f"[warn] ckpt size mismatch: use={used} file={total} "
                         f"(diff={total-used}); проверь config!\n")
    # base_decay[p] (D,)
    bd = []
    for p in range(NP):
        idx = p % 4
        t = (np.arange(D, dtype=np.float32) / (D - 1)) if D > 1 else np.zeros(D, np.float32)
        bd.append(cfg['decay_min'][idx] + t * (cfg['decay_max'][idx] - cfg['decay_min'][idx]))
    W['base_decay'] = bd
    return W

def rmsnorm(x, w, eps=1e-6):
    # x:(S,D) -> out:(S,D)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * inv * w

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def silu(x):    return x * sigmoid(x)

def scan(gate, x):
    # gate,x:(S,D) -> out[t]=gate[t]*out[t-1]+x[t]
    S, D = x.shape
    out = np.empty_like(x)
    h = np.zeros(D, dtype=x.dtype)
    for t in range(S):
        h = gate[t] * h + x[t]
        out[t] = h
    return out

def forward(W, cfg, ids, all_pos=False):
    # ids: list[int] len S<=T. returns logits (V,) at last pos, or (S,V) if all_pos.
    x = W['W_emb'][np.asarray(ids, dtype=np.int64)].astype(np.float32)  # (S,D)
    for l, ly in enumerate(W['layers']):
        n1 = rmsnorm(x, ly['norm1_w'])
        pir = n1.copy()
        for p, pw in enumerate(ly['pir']):
            gate = pir @ pw['W_gate'].T
            val  = pir @ pw['W_value'].T
            sig  = sigmoid(gate)
            gated = val * sig
            scan_gate = sig * W['base_decay'][p]      # broadcast (D,)
            scanned = scan(scan_gate, gated)
            out_proj = scanned @ pw['W_out'].T
            normed = rmsnorm(out_proj, pw['norm_w'])
            pir = pir + normed
        mix = pir @ ly['W_mix'].T
        normed_mix = rmsnorm(mix, ly['norm_pir_w'])
        x = x + normed_mix
        n2 = rmsnorm(x, ly['norm2_w'])
        ffn1 = n2 @ ly['W_ffn1'].T
        ffn3 = n2 @ ly['W_ffn3'].T
        ffn_gated = silu(ffn1) * ffn3
        ffn2 = ffn_gated @ ly['W_ffn2'].T
        x = x + ffn2
    fin = rmsnorm(x, W['norm_out_w'])
    if all_pos:
        return fin @ W['W_lm_head'].T           # (S,V)
    return fin[-1] @ W['W_lm_head'].T           # (V,)

def sample(logits, temp, top_k, rng):
    if temp <= 0:
        return int(np.argmax(logits))
    lg = logits.astype(np.float64) / temp
    if top_k and top_k < lg.size:
        kth = np.partition(lg, -top_k)[-top_k]
        lg = np.where(lg < kth, -1e30, lg)
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    return int(rng.choice(p.size, p=p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--spm', required=True, help='sentencepiece .model')
    ap.add_argument('--vocab_size', type=int, default=16000)
    ap.add_argument('--n_embd', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--n_pir_layers', type=int, default=4)
    ap.add_argument('--block_size', type=int, default=256)
    ap.add_argument('--ffn_mult', type=float, default=3.5)
    ap.add_argument('--prompt', default='Россия ')
    ap.add_argument('--max_tokens', type=int, default=60)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--top_k', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val_tokens', default='', help='.tokens file → печатает CE loss (валидация forward)')
    ap.add_argument('--val_n', type=int, default=256)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    cfg = build_cfg(args)
    W = load_ckpt(args.ckpt, cfg)

    # --- validation: CE loss на срезе .tokens (проверка что forward совпал с C++) ---
    if args.val_tokens:
        toks = np.fromfile(args.val_tokens, dtype=np.uint32)
        S = min(args.val_n, cfg['T'])
        seg = toks[:S+1].astype(np.int64)
        logits = forward(W, cfg, list(seg[:S]), all_pos=True)   # (S,V)
        m = logits.max(axis=1, keepdims=True)
        logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
        tgt = seg[1:S+1]
        ce = -logp[np.arange(S), tgt].mean()
        print(f"[val] CE loss on {S} tokens = {ce:.4f}  (ppl={np.exp(ce):.1f}; "
              f"random init≈{np.log(cfg['V']):.2f})")

    # --- generation ---
    rng = np.random.default_rng(args.seed)
    ids = sp.encode(args.prompt, out_type=int)
    for _ in range(args.max_tokens):
        ctx = ids[-cfg['T']:]
        logits = forward(W, cfg, ctx)
        nxt = sample(logits, args.temp, args.top_k, rng)
        ids.append(nxt)
    text = sp.decode(ids)
    print("=== PROMPT ===");   print(args.prompt)
    print("=== GENERATED ==="); print(text)

if __name__ == '__main__':
    main()
