# Эльбрус 8C2 — Verified output samples 2026-05-03

После commit `d9dce9e` (phi3 mmap fix) + `0ba114a` (gemma3 TP-4 wire),
rebuild17/18. Все измерения с `PT_PER_BLOCK_SCALE=1 PT_NO_NUMA_POOL=1
PT_Q8_SOA=1 PT_LM_HEAD_FP=1 PT_NO_FFN_SOA=1`. Greedy decode.

## Single-process (OMP_NUM_THREADS=32)

### phi3.5-mini Q4_K_M
**RU prompt:** «Что такое космос? Расскажи коротко.»

```
Космос - это всеобъемлющая область пространства, включающая все
газопылевые объекты и их окружения. Он охватывает все извест...
```
40 tokens / 11.5s = **3.5 tok/s**

**EN prompt:** «Hello! Tell me briefly about Moscow.»
```
Moscow, the capital city of Russia, is not only the political heart
of the country but also a vibrant cultural and economic center.
With its rich history that dates back to 1...
```
40 tokens / 11.4s = **3.5 tok/s**

### qwen3-1.7B Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.»
```
<think>
Хорошо, пользователь спрашивает, что такое космос и хочет ответить
коротко. Начну с того, что космос —
```
40 tokens / 4.9s = **8.1 tok/s**

### qwen3-4B Q4_K_M (SP)
**RU:** аналогично qwen3-1.7B (CoT thinking блок). **5.5 tok/s**.

### qwen2.5-7B Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.»
```
Космос - это пространство за пределами атмосферы Земли, включающее
все звезды, планеты и другие небес...
```
40 tokens / 13.8s = **2.9 tok/s**

### qwen3-8B Q4_K_M (SP)
**RU:**
```
<think>
Хорошо, пользователь спрашивает: "Что такое космос? Расскажи коротко."
Нужно дать краткий ответ
```
40 tokens / 15.3s = **2.6 tok/s**. CoT thinking начало.

### llama3-8B Q4_K_M (SP)
**RU:**
```
Космос - это все, что находится за пределами Земли. Это включает в себя
звезды, планеты, галактики и все остальное.
```
36 tokens / 13.1s = **2.7 tok/s**.

### deepseek-coder-7B Q4_K_M (SP)
After commit `81a79bd` (rope.scale_linear legacy key fallback).
**Code completion:** «def fibonacci(n):»
```
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        fib_sequence = [0, 1] # start with the first two numbers in the sequence
```
50 tokens / 16.4s = **3.0 tok/s**.

**Chat instruct:** «Write a Python function to compute Fibonacci numbers.»
```
Sure, here is a simple Python function to compute Fibonacci numbers:
```python
def fibonacci(n):
    if n <= 0:  # base
```
40 tokens / 14.0s = **2.9 tok/s**.

### mistral-7B Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.» (без чат-шаблона):
```
Космос - это обширное пространство, заполненное звездами, планетами
и галактиками. Он начинается от Земли и простирается до границы извест...
```
**SP: 2.9 tok/s. TP-4: 8.5 tok/s.**

### gemma3-4B Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.»
```
Космос – это всё, что существует: пространство, время, материя и
энергия. Это огромнейшая пустота, в которой находятся звезды, планеты,
галактики и всё остальное.
Вот основные моменты:
*
```
50 tokens / 10.7s = **4.7 tok/s**. Markdown structure preserved!

## Tensor Parallel — 4 ranks, OMP_NUM_THREADS=8 each, NUMA-bound

### phi3.5-mini Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.»
```
Космос - это всеобъемлющий, неограниченный пространственный контекст,
включающий все галактики, звезды и другие астрономические объекты.
Он характеризу...
```
50 tokens / 7.8s = **6.4 tok/s** (1.8× speedup vs SP)

### qwen3-4B Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.»
```
<think>
Хорошо, пользователь спрашивает, что такое космос, и хочет короткий
ответ. Начну с того, что космос — это пространство за пределами Земли.
Нужно упомянуть звезды, планеты, галактики и другие объекты. Важно
подчеркнуть, что это огромное пространство с разными явлениями
```
100 tokens / 9.2s = **10.9 tok/s** (почти baseline 11.4 + correct RU!)

### mistral-7B Q4_K_M
**RU:** «Что такое космос? Расскажи коротко.»
```
Космос - это обширное пространство, заполненное звездами, планетами
и галактиками. Он начинается от Земли и простирается до границы извест...
```
50 tokens / 5.9s = **8.5 tok/s**

### qwen3-1.7B Q4_K_M
TP-4: **17.1 tok/s** (verified prior session).

### gemma3-4B Q4_K_M (TP-4)
After commit `0ba114a` (post_attention_norm + post_ffw_norm wired
correctly on full h_buf после output_proj). Требует `PT_TP_GATHER=1`.
**RU:** «Что такое космос? Расскажи коротко.»
```
Космос – это всё, что существует за пределами нашей Земли. Это огромная,
бесконечная область, включающая в себя:
*   **Вселенная:**  Все известные объекты, включая звезды, планеты, галактики
```
50 tokens / 7.5s = **6.7 tok/s** (1.4× speedup vs SP, structured markdown
сохранён, ×5.2 vs llama.cpp 32t baseline 1.30 tok/s).

## Speedups vs llama.cpp 32-thread (numactl --interleave=all)

| Модель | PromeTorch TP-4 | llama.cpp 32t | Speedup |
|---|---:|---:|---:|
| qwen3-1.7B | 17.1 | 2.71 | **×6.3** |
| qwen3-4B | 10.9 | 1.82 | **×6.0** |
| mistral-7B | 8.5 | 1.74 | **×4.9** |
| phi3.5-mini | 6.4 | (3.5 SP) | n/a (no llama.cpp baseline) |

## Не работает / capacity issues

* **qwen3-0.6B** — `!!!!!` output. Не RoPE, не tokenizer. Capacity
  issue (модель слишком маленькая для chat instruct).
* **qwen3-14B / llama3-8B / qwen2.5-7B** — OOM в TP-4 при PT_Q8_SOA=1
  (1958 MB SoA storage × 4 ranks = 8 GB). SP path работает.
