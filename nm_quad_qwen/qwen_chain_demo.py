#!/usr/bin/env python3
"""
qwen_chain_demo.py — multi-layer chain demo для Qwen3-4B inference на CPU.

Запускает qwen_python_layer.qwen_layer_forward последовательно для
layers 0..N-1, передавая output одного слоя на вход следующего. Это
демонстрация архитектуры 36-layer inference (Qwen3-4B имеет 36 layers
total). Subset M_FFN=256 для разумного времени compute.
"""

import sys
import struct
import time
import numpy as np
from qwen_python_layer import qwen_layer_forward, K_DIM


def main():
    if len(sys.argv) < 3:
        print("Usage: qwen_chain_demo.py <gguf> <emb.bin> [n_layers] [pos]")
        sys.exit(1)
    path = sys.argv[1]
    in_path = sys.argv[2]
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    pos = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    with open(in_path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        x = np.frombuffer(f.read(K_DIM * 4), dtype=np.float32).copy()
    print(f"[chain] n_layers={n_layers} pos={pos}")
    print(f"[chain] x_input L2={np.linalg.norm(x):.4f}")

    t0 = time.time()
    for L in range(n_layers):
        t_layer = time.time()
        x = qwen_layer_forward(path, L, x, pos)
        dt = time.time() - t_layer
        print(f"[layer {L:2d}] wall={dt:6.2f}s  x_final L2={np.linalg.norm(x):.4f}  first 4=[{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}, {x[3]:.4f}]")

    total = time.time() - t0
    print(f"[chain] total wall={total:.2f}s for {n_layers} layers")
    print(f"[chain] x_final[0..3] = {x[:4].tolist()}")
    print(f"[chain] final L2 = {np.linalg.norm(x):.4f}")

    # Extrapolation for 36 layers
    if n_layers > 0:
        per = total / n_layers
        print(f"[extrapol] {per:.2f}s/layer × 36 = {per*36:.1f}s = {per*36/60:.1f} min for full 36-layer chain")


if __name__ == "__main__":
    main()
