#!/usr/bin/env python3
"""
multihost_avg.py — равно-весовое усреднение PIR fp32 checkpoints.

Используется в Hierarchical Local SGD между Эльбрус-хостами:
  1. Каждый host пишет /tmp/pt_mh/local_$HOSTNAME.bin (своими intra-DDP weights)
  2. SCP-coordinator собирает их на master
  3. Этот скрипт усредняет → /tmp/pt_mh/avg.bin
  4. SCP-coordinator broadcast обратно

Формат файла — raw fp32 dump (как train_pir_elbrus --save_dir).
ВСЕ файлы должны быть ОДИНАКОВОГО размера (один и тот же model config).

Usage:
  python3 multihost_avg.py OUT.bin IN1.bin IN2.bin [IN3.bin ...]

Опционально:
  --weighted W1,W2,W3   вес каждого хоста (учёт compute capacity).
                        Default — равно. Сумма не обязательно 1, нормализуется.
                        Пример: 0.3,1.0,2.4 (4C/16C/8C2 пропорция).

Без зависимостей кроме stdlib + numpy (есть на всех машинах Эльбрус Linux).
"""
import argparse
import os
import sys
import time

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy не установлен. apt-get install python3-numpy", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", help="Куда писать усреднённые weights")
    ap.add_argument("inputs", nargs="+", help="Входные checkpoint'ы (>= 2)")
    ap.add_argument("--weighted", type=str, default=None,
                    help="Comma-separated веса для каждого input (нормализуется)")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="Точность накопителя (default f32)")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        print(f"ERROR: нужно ≥2 inputs, дано {len(args.inputs)}", file=sys.stderr)
        sys.exit(1)

    # Проверка размеров.
    sizes = [os.path.getsize(p) for p in args.inputs]
    if len(set(sizes)) > 1:
        print("ERROR: размеры файлов отличаются:", file=sys.stderr)
        for p, s in zip(args.inputs, sizes):
            print(f"  {s:>12} bytes  {p}", file=sys.stderr)
        sys.exit(1)

    n_bytes = sizes[0]
    n_floats = n_bytes // 4  # raw fp32
    if n_bytes % 4 != 0:
        print(f"ERROR: размер {n_bytes} не делится на 4 (не fp32 dump?)",
              file=sys.stderr)
        sys.exit(1)

    # Веса.
    if args.weighted:
        weights = [float(w) for w in args.weighted.split(",")]
        if len(weights) != len(args.inputs):
            print(f"ERROR: --weighted имеет {len(weights)} значений, "
                  f"нужно {len(args.inputs)}", file=sys.stderr)
            sys.exit(1)
    else:
        weights = [1.0] * len(args.inputs)

    total_w = sum(weights)
    if total_w <= 0:
        print("ERROR: сумма весов = 0", file=sys.stderr)
        sys.exit(1)
    norm_weights = [w / total_w for w in weights]

    print(f"[multihost_avg] {len(args.inputs)} хостов, {n_floats * 4 / 1e6:.1f} MB каждый")
    print(f"[multihost_avg] нормализованные веса: " +
          ", ".join(f"{p}={w:.3f}" for p, w in zip(args.inputs, norm_weights)))

    # Стриминг по чанкам — иначе 8GB модель не поместится в RAM
    # на маленьких машинах. Чанк 64 MB → 16M fp32.
    CHUNK_FLOATS = 64 * 1024 * 1024 // 4
    t0 = time.time()
    bytes_written = 0

    with open(args.output, "wb") as fout:
        files = [open(p, "rb") for p in args.inputs]
        try:
            remaining = n_floats
            while remaining > 0:
                chunk_n = min(CHUNK_FLOATS, remaining)
                # Читаем по chunk из всех файлов, складываем взвешенно
                acc = np.zeros(chunk_n, dtype=np.float32)
                for f, w in zip(files, norm_weights):
                    buf = np.fromfile(f, dtype=np.float32, count=chunk_n)
                    if buf.size != chunk_n:
                        print(f"ERROR: чтение {f.name} вернуло {buf.size} вместо "
                              f"{chunk_n}", file=sys.stderr)
                        sys.exit(1)
                    acc += buf * w
                acc.tofile(fout)
                remaining -= chunk_n
                bytes_written += chunk_n * 4
        finally:
            for f in files:
                f.close()

    dt = time.time() - t0
    print(f"[multihost_avg] OK — {bytes_written / 1e6:.1f} MB записано в {args.output} "
          f"за {dt:.1f}s ({bytes_written / 1e6 / dt:.0f} MB/s)")


if __name__ == "__main__":
    main()
