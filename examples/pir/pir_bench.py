# -*- coding: utf-8 -*-
"""
pir_bench.py — бенчмарк базовой PIR-LM на русских задачах для маленьких моделей.

Метрики честные для BASE-LM (без инструкт-тюнинга):
  1. Held-out perplexity — на текстах, которых НЕ было в претрейне (пассажи RSG).
  2. Zero-shot accuracy через log-likelihood — модель не отвечает, а «выбирает»
     вариант с большей вероятностью продолжения. Работает на любой базовой LM.
     Задачи Russian SuperGLUE:
       PARus   — выбор причины/следствия (chance 50%)
       TERRa   — текстовое следование (chance 50%)
       DaNetQA — да/нет вопрос по пассажу (chance ~50%, но классы несбаланс.)

Запуск:
  python3 pir_bench.py --ckpt ck.bin --spm ru_bpe_16k.model \
      --n_embd 256 --n_layers 4 --block_size 512 --bench-dir bench
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# переиспользуем forward/загрузку из pir_infer.py (лежит рядом)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pir_infer import build_cfg, load_ckpt, forward

SP = None


def enc(s):
    return SP.encode(s, out_type=int)


def logprobs(W, cfg, ids):
    """(S,V) → per-position log-softmax; logits[i] предсказывает ids[i+1]."""
    ids = ids[-cfg['T']:]
    lg = forward(W, cfg, ids, all_pos=True).astype(np.float64)  # (S,V)
    lg -= lg.max(axis=1, keepdims=True)
    lg -= np.log(np.exp(lg).sum(axis=1, keepdims=True))
    return lg, ids


def ll_cont(W, cfg, prefix_ids, cont_ids, norm=True):
    """Средний (или суммарный) log p(cont | prefix)."""
    ids = prefix_ids + cont_ids
    lg, ids = logprobs(W, cfg, ids)
    S, n = len(ids), len(cont_ids)
    n = min(n, S - 1)
    if n <= 0:
        return -1e9
    tot = 0.0
    for k in range(S - n, S):
        tot += lg[k - 1, ids[k]]
    return tot / n if norm else tot


def ppl_on_text(W, cfg, text, max_tokens=8000):
    """Перплексия на длинном тексте окнами по block_size."""
    ids = enc(text)[:max_tokens]
    T = cfg['T']
    nll, cnt = 0.0, 0
    for s in range(0, len(ids) - 1, T):
        chunk = ids[s:s + T]
        if len(chunk) < 2:
            break
        lg, chunk = logprobs(W, cfg, chunk)
        for k in range(1, len(chunk)):
            nll += -lg[k - 1, chunk[k]]
            cnt += 1
    return np.exp(nll / cnt), cnt


def find_val(bench_dir, task):
    g = glob.glob(os.path.join(bench_dir, task, '**', 'val.jsonl'), recursive=True)
    return g[0] if g else None


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def norm_label(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).lower()
    if s in ('entailment', 'true', 'да', '1'):
        return 0 if s == 'entailment' else 1
    if s in ('not_entailment', 'false', 'нет', '0'):
        return 1 if s == 'not_entailment' else 0
    return None


def bench_parus(W, cfg, rows):
    ok = 0
    for r in rows:
        conn = ' потому что ' if r['question'] == 'cause' else ' поэтому '
        pre = enc(r['premise'].rstrip('.') + conn)
        s1 = ll_cont(W, cfg, pre, enc(r['choice1']))
        s2 = ll_cont(W, cfg, pre, enc(r['choice2']))
        pred = 0 if s1 >= s2 else 1
        if pred == int(r['label']):
            ok += 1
    return ok / len(rows)


def bench_yesno(W, cfg, rows, passage_key, q_key, label_key):
    ok = 0
    da, net = enc(' Да'), enc(' Нет')
    for r in rows:
        ctx = (r.get(passage_key, '') + '\n' + r.get(q_key, '')).strip()
        pre = enc(ctx + ' Ответ:')
        s_da = ll_cont(W, cfg, pre, da)
        s_net = ll_cont(W, cfg, pre, net)
        pred = 0 if s_da >= s_net else 1     # 0=Да(true/entail), 1=Нет
        lab = norm_label(r.get(label_key))
        if lab is not None and pred == lab:
            ok += 1
    return ok / len(rows)


def bench_terra(W, cfg, rows):
    ok = 0
    da, net = enc(' Да'), enc(' Нет')
    for r in rows:
        pre = enc(r['premise'] + ' Вопрос: ' + r['hypothesis'] + ' Верно? Ответ:')
        pred = 0 if ll_cont(W, cfg, pre, da) >= ll_cont(W, cfg, pre, net) else 1
        lab = norm_label(r.get('label'))
        if lab is not None and pred == lab:
            ok += 1
    return ok / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--spm', required=True)
    ap.add_argument('--vocab_size', type=int, default=16000)
    ap.add_argument('--n_embd', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--n_pir_layers', type=int, default=4)
    ap.add_argument('--block_size', type=int, default=512)
    ap.add_argument('--ffn_mult', type=float, default=3.5)
    ap.add_argument('--bench-dir', default='bench', dest='bench_dir')
    args = ap.parse_args()

    global SP
    import sentencepiece as spm
    SP = spm.SentencePieceProcessor(model_file=args.spm)
    cfg = build_cfg(args)
    W = load_ckpt(args.ckpt, cfg)
    print(f"модель: D={cfg['D']} L={cfg['L']} NP={cfg['NP']} V={cfg['V']}  ckpt={os.path.basename(args.ckpt)}\n")

    # ---- held-out perplexity: пассажи DaNetQA (свежий русский, не в претрейне) ----
    dq = find_val(args.bench_dir, 'DaNetQA')
    if dq:
        rows = load_jsonl(dq)
        pk = 'passage' if 'passage' in rows[0] else ('paragraph' if 'paragraph' in rows[0] else 'text')
        text = '\n'.join(r.get(pk, '') for r in rows[:400])
        ppl, n = ppl_on_text(W, cfg, text)
        print(f"[held-out ppl] DaNetQA-пассажи ({n} токенов): ppl = {ppl:.1f}")

    print("\n=== Zero-shot (log-likelihood, база без инструкт-тюна) ===")
    # PARus
    p = find_val(args.bench_dir, 'PARus')
    if p:
        rows = [r for r in load_jsonl(p) if 'label' in r]
        print(f"PARus   (n={len(rows)}, chance 50%): acc = {bench_parus(W, cfg, rows)*100:.1f}%")
    # TERRa
    t = find_val(args.bench_dir, 'TERRa')
    if t:
        rows = [r for r in load_jsonl(t) if 'label' in r]
        print(f"TERRa   (n={len(rows)}, chance 50%): acc = {bench_terra(W, cfg, rows)*100:.1f}%")
    # DaNetQA
    if dq:
        rows = [r for r in load_jsonl(dq) if 'label' in r]
        pk = 'passage' if 'passage' in rows[0] else ('paragraph' if 'paragraph' in rows[0] else 'text')
        acc = bench_yesno(W, cfg, rows, pk, 'question', 'label')
        base = max(np.mean([norm_label(r['label']) for r in rows]),
                   1 - np.mean([norm_label(r['label']) for r in rows]))
        print(f"DaNetQA (n={len(rows)}, majority {base*100:.0f}%): acc = {acc*100:.1f}%")


if __name__ == '__main__':
    main()
