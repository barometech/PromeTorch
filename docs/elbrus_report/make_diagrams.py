"""
Генерация диаграмм для arXiv-style статьи PromeTorch на Эльбрусе.
Все надписи на русском. Чёрно-белый стиль.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "C:/Users/paper/Desktop/promethorch/docs/elbrus_report/figs"
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ========== ДИАГРАММА 1: Квадрант сопоставления (где PromeTorch) ==========
fig, ax = plt.subplots(figsize=(8, 6))

# Точки: (x=поддержка современных LLM 0..10, y=скорость на ELBRUS-8C2 в tok/s)
# Размещаем по диагонали чтобы лейблы не перекрывались
# y = реальная скорость в tok/s (или 0 если не LLM)
projects = [
    # name, x, y, size, color, label_dx, label_dy
    ("llama.cpp-e2k\n(LLaMA-1 7B Q4_0,\n8SV 8 потоков)",
                                     3.0, 6.2,  280, "#888888",  10,   8),
    ("Smart Engines\n(только CV: UNet, OCR)",
                                     0.5, 2.5,  240, "#888888",  10, -28),
    ("NCNN + Vulkan демо\n(на GPU Radeon, не CPU)",
                                     2.0, 1.0,  240, "#888888",  10,   8),
    ("MCST EML\n(BLAS, не LLM)",
                                     0.5, 0.5,  220, "#888888",  10, -22),
    ("PromeTorch\n(qwen3-1.7B TP-4)",
                                     9.0, 17.1, 380, "#000000", -110,  -12),
    ("PromeTorch\n(qwen3-4B TP-4)",
                                     9.0, 10.9, 320, "#000000", -110,  -12),
    ("PromeTorch\n(mistral-7B TP-4)",
                                     9.0, 8.5,  300, "#000000",  10,    8),
]
for name, x, y, s, c, dx, dy in projects:
    ax.scatter(x, y, s=s, c=c, zorder=3, edgecolors='black', linewidths=1.5)
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=9, fontweight='bold' if 'PromeTorch' in name else 'normal',
                color='black' if 'PromeTorch' in name else '#444')

ax.set_xlim(-0.5, 11)
ax.set_ylim(-1, 20)
ax.set_xlabel("Поддержка современных LLM (qwen3, gemma3, phi3, deepseek)", fontsize=11)
ax.set_ylabel("Скорость на Эльбрус-8С2, токенов/сек\n(лучший результат, 7B-класс или меньше)", fontsize=11)
ax.set_title("Сопоставление работ по машинному обучению на Эльбрус (e2k)\nположение PromeTorch относительно публичных проектов", fontsize=12)
ax.grid(True, alpha=0.3, zorder=0)
ax.axhline(y=10, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(x=5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

# Quadrant labels
ax.text(2.5, 18, "↖ старые LLM,\nвысокая скорость", fontsize=9, color='gray', ha='center')
ax.text(8, 18, "↗ современные LLM,\nвысокая скорость", fontsize=9, color='black', ha='center', fontweight='bold')
ax.text(2.5, 1, "↙ старые LLM,\nнизкая скорость", fontsize=9, color='gray', ha='center')
ax.text(8, 1, "↘ современные LLM,\nнизкая скорость", fontsize=9, color='gray', ha='center')

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_quadrant.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_quadrant.png")


# ========== ДИАГРАММА 2: Speedup bar chart vs llama.cpp 32t ==========
fig, ax = plt.subplots(figsize=(9, 5.5))

models = ["qwen3-1.7B", "qwen3-4B", "gemma3-4B", "mistral-7B", "phi3.5-mini",
          "qwen2.5-7B*", "llama3-8B*", "qwen3-14B*"]
prometorch = [17.1, 10.9, 6.7, 8.5, 6.4, 2.9, 2.7, 1.5]   # * = SP only
llama_cpp = [2.71, 1.82, 1.30, 1.74, 2.08, 1.71, 1.65, 1.02]

x = np.arange(len(models))
width = 0.38
b1 = ax.bar(x - width/2, prometorch, width, label="PromeTorch (TP-4 / SP)", color='#000000', edgecolor='black')
b2 = ax.bar(x + width/2, llama_cpp,  width, label="llama.cpp 32t (numactl interleave)", color='#cccccc', edgecolor='black')

# Speedup labels on top of PromeTorch bars
for i, (pt, lc) in enumerate(zip(prometorch, llama_cpp)):
    sp = pt / lc
    ax.text(i - width/2, pt + 0.3, f"×{sp:.1f}", ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel("Модель (звёздочка = только single-process; TP-4 OOM)", fontsize=11)
ax.set_ylabel("Скорость декодирования, токенов/сек", fontsize=11)
ax.set_title("Скорость PromeTorch против llama.cpp v3 32-thread baseline\nЭльбрус-8С2, Q4_K_M, greedy decode", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 20)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_speedup.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_speedup.png")


# ========== ДИАГРАММА 3: Архитектура PromeTorch (поток forward pass) ==========
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

def box(x, y, w, h, text, fontsize=9, fc='white', ec='black'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          fc=fc, ec=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='black'))

ax.set_title("Архитектура PromeTorch — forward-pass на токен (TP-4)", fontsize=13, pad=12)

# Top — input
box(4, 6.0, 2, 0.5, "token_id (int32)", 10, fc='#eeeeee')

# Embedding
box(3.5, 5.2, 3, 0.5, "embedding lookup → x[H]", 10)
arrow(5, 5.95, 5, 5.7)

# Layer block frame
ax.add_patch(mp.Rectangle((0.5, 1.2), 9, 3.7, fill=False, ec='black', linewidth=1.5, linestyle='--'))
ax.text(0.6, 4.75, "× num_layers (28..40)", fontsize=10, fontweight='bold')

# Inside layer:
box(0.7, 4.2, 2.0, 0.45, "RMSNorm(x)", 9)
arrow(5, 5.15, 1.7, 4.65)

box(2.9, 4.2, 2.0, 0.45, "QKV proj (Q4_K)", 9, fc='#f0e0e0')
arrow(2.7, 4.42, 2.9, 4.42)

box(5.1, 4.2, 1.8, 0.45, "RoPE\n(NORM/NeoX)", 9)
arrow(4.9, 4.42, 5.1, 4.42)

box(7.1, 4.2, 2.3, 0.45, "attention (GQA + SWA)", 9, fc='#e0e0f0')
arrow(6.9, 4.42, 7.1, 4.42)

box(0.7, 3.4, 2.0, 0.45, "output proj", 9, fc='#f0e0e0')
arrow(8.2, 4.18, 1.7, 3.85)

box(2.9, 3.4, 2.0, 0.45, "(post_attn_norm\nGemma3)", 8)
arrow(2.7, 3.62, 2.9, 3.62)

box(5.1, 3.4, 1.8, 0.45, "+ residual", 9)
arrow(4.9, 3.62, 5.1, 3.62)

box(7.1, 3.4, 2.3, 0.45, "RMSNorm(x)", 9)
arrow(6, 3.4, 7.1, 3.62)

box(0.7, 2.6, 2.0, 0.45, "gate × up (Q4_K)", 9, fc='#f0e0e0')
arrow(8.2, 3.38, 1.7, 3.05)

box(2.9, 2.6, 2.0, 0.45, "SiLU / GeGLU", 9)
arrow(2.7, 2.82, 2.9, 2.82)

box(5.1, 2.6, 1.8, 0.45, "ffn_down (Q4_K)", 9, fc='#f0e0e0')
arrow(4.9, 2.82, 5.1, 2.82)

box(7.1, 2.6, 2.3, 0.45, "(post_ffw_norm)", 9)
arrow(6.9, 2.82, 7.1, 2.82)

box(2.5, 1.7, 2, 0.45, "+ residual", 9)
arrow(8.2, 2.58, 3.5, 1.95)

box(5.5, 1.7, 2, 0.45, "TP-4 SHM AllReduce", 9, fc='#e0f0e0')
arrow(4.5, 1.92, 5.5, 1.92)

# Output
box(3.5, 0.4, 3, 0.45, "lm_head + argmax", 10)
arrow(5, 1.65, 5, 0.85)

# Memory annotations on right
ax.text(9.65, 4.4, "Q4_K\n2.5 GB\nmmap", fontsize=8, ha='left', va='center',
        bbox=dict(boxstyle="round,pad=0.2", fc='#fff8dc', ec='gray'))

# Legend
ax.text(0.5, 0.1, "Жёлто-розовый = квантизованные веса (Q4_K)   |   Зелёный = inter-rank communication",
        fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_arch.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_arch.png")

print("ALL DIAGRAMS DONE")
