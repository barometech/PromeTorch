#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_bpe.py — подготовка BPE-данных для тренировки PIR.

Делает две вещи:
  1. Тренирует sentencepiece BPE-токенизатор на текстовом корпусе.
  2. Токенизирует корпус в бинарный файл uint32 (.tokens) — формат,
     который train_pir_elbrus авто-детектит по расширению.

Пример (русская Википедия, словарь 16k — проверенная конфигурация):
    python3 prepare_bpe.py --input data/russian_all.txt \
        --out-prefix data/ru_bpe_16k --vocab-size 16000

Выход:
    data/ru_bpe_16k.model    — sentencepiece модель (нужна pir_infer.py)
    data/ru_bpe_16k.vocab    — словарь (для глаз)
    data/ru_bpe_16k.tokens   — uint32 токены корпуса (вход тренировки)

Зависимости: pip install sentencepiece  (на Эльбрусе есть в системном pip)
"""
import argparse
import os
import struct
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='текстовый корпус (UTF-8)')
    ap.add_argument('--out-prefix', required=True,
                    help='префикс выходных файлов (.model/.vocab/.tokens)')
    ap.add_argument('--vocab-size', type=int, default=16000,
                    help='размер BPE-словаря (default 16000; передавать его же '
                         'в train_pir_elbrus --vocab_size!)')
    ap.add_argument('--character-coverage', type=float, default=0.9995,
                    help='0.9995 для кириллицы (default), 1.0 для чистого ASCII')
    ap.add_argument('--max-sentence-length', type=int, default=65536)
    ap.add_argument('--skip-train', action='store_true',
                    help='не тренировать модель (уже есть) — только токенизация')
    args = ap.parse_args()

    import sentencepiece as spm

    model_path = args.out_prefix + '.model'

    # --- 1. Тренировка BPE ---
    if not args.skip_train:
        print(f'[1/2] Тренирую BPE {args.vocab_size} на {args.input} ...')
        spm.SentencePieceTrainer.train(
            input=args.input,
            model_prefix=args.out_prefix,
            vocab_size=args.vocab_size,
            model_type='bpe',
            character_coverage=args.character_coverage,
            max_sentence_length=args.max_sentence_length,
            # спец-токены не нужны: PIR тренируется как чистая LM на потоке
            pad_id=-1, unk_id=0, bos_id=-1, eos_id=-1,
        )
        print(f'      → {model_path}')
    elif not os.path.exists(model_path):
        sys.exit(f'--skip-train, но {model_path} не найден')

    # --- 2. Токенизация корпуса → uint32 .tokens ---
    print(f'[2/2] Токенизирую {args.input} → {args.out_prefix}.tokens ...')
    sp = spm.SentencePieceProcessor(model_file=model_path)
    n_total = 0
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(args.out_prefix + '.tokens', 'wb') as fout:
        buf = []
        for line in fin:
            buf.append(line)
            # батчим по ~4 МБ текста, чтобы не звать encode на каждую строку
            if sum(len(s) for s in buf) > 4 * 1024 * 1024:
                ids = sp.encode(''.join(buf), out_type=int)
                fout.write(struct.pack('<{}I'.format(len(ids)), *ids))
                n_total += len(ids)
                buf = []
                print(f'      {n_total/1e6:.2f}M токенов ...', end='\r')
        if buf:
            ids = sp.encode(''.join(buf), out_type=int)
            fout.write(struct.pack('<{}I'.format(len(ids)), *ids))
            n_total += len(ids)

    print(f'\nГотово: {n_total/1e6:.2f}M токенов, словарь {args.vocab_size}.')
    print(f'Запуск тренировки: --data {args.out_prefix}.tokens '
          f'--vocab_size {args.vocab_size}')


if __name__ == '__main__':
    main()
