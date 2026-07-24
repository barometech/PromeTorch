# -*- coding: utf-8 -*-
"""
build_sft_corpus.py — сборка SFT-корпуса для PIR 13.5M (агентские действия +
инструкции + факты) из 6 HF-источников в plain-text шаблоны.

Шаблоны (в BPE-16k нет спец-токенов — разделители текстовые):
  Инструкция:  ### Задача:\n{u}\n### Ответ:\n{a}\n\n
  Факты (QA):  ### Факты:\n{ctx}\n### Вопрос:\n{q}\n### Ответ:\n{a}\n\n
  Действие:    ### Инструменты:\n{tools}\n### Задача:\n{u}\n### Действие:\n{call}\n### Результат:\n{res}\n### Ответ:\n{a}\n\n

Фильтры: только русский (где есть язык-колонки), качество (za: overall_score>=8,
saiga: opus_score>=8), кап длины сэмпла MAX_CHARS (~450 BPE-токенов — целиком
влезает в окно block=512 лоадера случайных окон), дедуп по md5 префикса.
"""
import glob
import hashlib
import json
import random
import sys

import pandas as pd

MAX_CHARS = 2400  # ~окно block 512; лоадер всё равно режет случайными окнами
random.seed(42)

samples = []          # list[str]
seen = set()
stats = {}


def add(src, text):
    text = text.strip()
    if not text or len(text) > MAX_CHARS:
        stats.setdefault(src, [0, 0])[1] += 1
        return
    h = hashlib.md5(text.encode('utf-8')).digest()  # полный текст: префикс ложно резал sberquad (общие контексты)
    if h in seen:
        stats.setdefault(src, [0, 0])[1] += 1
        return
    seen.add(h)
    samples.append(text + "\n\n")
    stats.setdefault(src, [0, 0])[0] += 1


def pairs_from_conv(conv):
    """[{role, content}...] → [(user, assistant)...] последовательные пары."""
    out, u = [], None
    for m in conv:
        r = (m.get('role') or '').lower()
        c = (m.get('content') or '').strip()
        if r in ('user', 'human'):
            u = c
        elif r in ('assistant', 'bot', 'gpt') and u:
            out.append((u, c))
            u = None
    return out


def is_ru(s):
    # доля кириллицы среди букв > 0.5
    letters = [c for c in s[:400] if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if 'Ѐ' <= c <= 'ӿ')
    return cyr / len(letters) > 0.5


# ---------- 1. sberquad: факты ----------
# Контексты часто 2000-4000 символов (кап резал 80%). Вырезаем окно ±600
# символов вокруг ответа (answer_start известен) по границам предложений —
# сэмпл влезает, фактность сохраняется.
try:
    df = pd.read_parquet('sberquad_train.parquet')
    for _, r in df.iterrows():
        ans = r['answers']['text']
        if not len(ans):
            continue
        a = ans[0]
        ctx = r['context']
        st = int(r['answers']['answer_start'][0]) if len(r['answers']['answer_start']) else 0
        lo = max(0, st - 600)
        hi = min(len(ctx), st + len(a) + 600)
        # расширяем до границ предложений
        dot = ctx.rfind('. ', 0, lo)
        lo = dot + 2 if dot > 0 else lo
        dot = ctx.find('. ', hi)
        hi = dot + 1 if dot > 0 else hi
        add('sberquad', f"### Факты:\n{ctx[lo:hi].strip()}\n### Вопрос:\n{r['question']}\n### Ответ:\n{a}")
except Exception as e:
    print('sberquad SKIP:', e)

# ---------- 2. russian_instructions: dialogue = чередующийся список ----------
try:
    df = pd.read_parquet('ru_instr.parquet')
    for _, r in df.iterrows():
        d = list(r['dialogue'])
        for i in range(0, len(d) - 1, 2):
            add('ru_instr', f"### Задача:\n{d[i]}\n### Ответ:\n{d[i+1]}")
except Exception as e:
    print('ru_instr SKIP:', e)

# ---------- 3. saiga_scored ----------
try:
    df = pd.read_parquet('saiga.parquet')
    df = df[(df['opus_score'] >= 8) & (df['language'].str.lower().str.startswith('ru'))]
    for _, r in df.iterrows():
        for u, a in pairs_from_conv(r['messages']):
            add('saiga', f"### Задача:\n{u}\n### Ответ:\n{a}")
except Exception as e:
    print('saiga SKIP:', e)

# ---------- 4. GrandMaster-PRO-MAX ----------
try:
    for f in ('gm_train0.parquet', 'gm_train1.parquet'):
        df = pd.read_parquet(f)
        df = df[df['answer_lang'] == 'ru'] if 'answer_lang' in df else df
        for _, r in df.iterrows():
            for u, a in pairs_from_conv(r['conversation']):
                add('grandmaster', f"### Задача:\n{u}\n### Ответ:\n{a}")
except Exception as e:
    print('grandmaster SKIP:', e)

# ---------- 5. glaive-function-calling-ru: агентские действия ----------
# messages/functions — JSON-строки. Роли: system (описание функций), user,
# assistant, function_call (json), function_response (json).
try:
    df = pd.read_parquet('glaive_ru.parquet')
    for _, r in df.iterrows():
        try:
            msgs = json.loads(r['messages'])
        except Exception:
            continue
        # инструменты: имена+описания из functions либо из system
        tool_txt = ''
        try:
            tl = json.loads(r['functions']) if r['functions'] else []
            if isinstance(tl, dict):
                tl = [tl]
            tool_txt = '\n'.join(
                f"{t.get('name','')} — {(t.get('description','') or '')[:90]}"
                for t in tl[:4] if isinstance(t, dict))
        except Exception:
            pass
        if not tool_txt:
            for m in msgs:
                if m.get('role') == 'system':
                    try:
                        j = m['content'].index('{')
                        sd = json.loads(m['content'][j:m['content'].rindex('}')+1])
                        tool_txt = f"{sd.get('name','')} — {(sd.get('description','') or '')[:90]}"
                    except Exception:
                        pass
                    break
        u = call = res = None
        for m in msgs:
            role = (m.get('role') or '').lower()
            c = (m.get('content') or '').strip()
            if role == 'user':
                u = c
            elif role == 'function_call':
                call = c
            elif role == 'function_response':
                res = c
            elif role == 'assistant':
                if u and call:
                    t = (f"### Инструменты:\n{tool_txt}\n### Задача:\n{u}\n"
                         f"### Действие:\n{call}"
                         + (f"\n### Результат:\n{res}" if res else '')
                         + f"\n### Ответ:\n{c}")
                    add('glaive_act', t)
                elif u:
                    add('glaive_chat', f"### Задача:\n{u}\n### Ответ:\n{c}")
                u = call = res = None
except Exception as e:
    print('glaive SKIP:', e)

# ---------- 6. ZeroAgency ru-big (бэкбон) ----------
za_files = sorted(glob.glob('za_train*.parquet'))
print('ZeroAgency файлов:', len(za_files))
for f in za_files:
    try:
        df = pd.read_parquet(f, columns=['conversation', 'overall_score', 'refusal'])
        # датасет уже отфильтрован автором gpt-4.1-скорингом; ≥7 расширяет пул
        # (бюджет 675M по формуле юзера), refusal-ответы выкидываем всегда
        df = df[(df['overall_score'] >= 7) & (df['refusal'] == 0)]
        for _, r in df.iterrows():
            for u, a in pairs_from_conv(r['conversation']):
                if is_ru(u):
                    add('zeroagency', f"### Задача:\n{u}\n### Ответ:\n{a}")
    except Exception as e:
        print(f, 'SKIP:', e)

# ---------- итог ----------
random.shuffle(samples)
with open('sft_corpus.txt', 'w', encoding='utf-8') as f:
    f.writelines(samples)

total_chars = sum(len(s) for s in samples)
print('\n=== СТАТИСТИКА ===')
for k, (ok, drop) in sorted(stats.items()):
    print(f"{k:12s}: взято {ok:8d}, отброшено {drop:8d}")
print(f"сэмплов: {len(samples)}, символов: {total_chars/1e6:.1f}M, "
      f"~токенов (х0.22): {total_chars*0.22/1e6:.0f}M")
