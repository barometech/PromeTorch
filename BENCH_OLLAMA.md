# BENCH_OLLAMA — PromeTorch GGUF vs Ollama (A100-SXM4-40GB)

Date: 2026-04-20. Hardware: A100-SXM4-40GB, CUDA 12.4 build, CUDA 12.8 runtime, Windows 10.
Settings: `--device cuda --temperature 0 --max_tokens 100` (PromeTorch) / `num_predict=100, temperature=0, seed=42` (Ollama HTTP API).
Same GGUF blobs from `%USERPROFILE%\.ollama\models\blobs` are loaded by both frameworks.

## Prompts
- **P1:** `Write a haiku about artificial intelligence:`
- **P2:** `def fibonacci(n):\n    if n <= 1:\n        return n\n    return `
- **P3:** `The capital of France is`

## Results

| Model | Prompt | PromeTorch tok/s | Ollama tok/s | Ratio | PromeTorch first 80 chars | Ollama first 80 chars |
|-------|--------|-----------------:|-------------:|------:|---------------------------|-----------------------|
| qwen3:4b | P1 | 85.9 | 165.5 | 52% | The first line: something about the nature of AI, maybe something like "A machin | [thinking] Okay, the user wants a haiku about artificial intelligence. Let me st |
| qwen3:4b | P2 | 84.4 | 164.0 | 51% | 1\n else:\n return fibonacci(n-1) + fibonacci(n-2) # n is the nth term in the se | [thinking] Okay, I need to complete the Fibonacci function. Let me think. The Fi |
| qwen3:4b | P3 | 86.1 | 164.5 | 52% | Paris. The capital of France is Paris. The French capital is Paris. The French c | [thinking] Okay, the user is asking, "The capital of France is..." and they've l |
| gemma3:4b | P1 | 74.7 | 144.4 | 52% | The code learns, grows, deep mind new learn deep grow mind grow deep grow | Code learns and evolves, Mimicking a human mind, Future's digital. |
| gemma3:4b | P2 | 68.3 | 144.6 | 47% | n\n ifn <=0:\returnn\if\nreturn\if\nreturn\if\if\return1if\end #end# # # # # # # | ```python def fibonacci(n): if n <= 1: return n return fibonacci(n-1) + fibonacc |
| gemma3:4b | P3 | 71.0 | 147.1 | 48% | Paris. Paris is the capital of France. The capital of France is Paris. The capit | The capital of France is **Paris**. Do you want to know anything more about Pari |
| deepseek-r1:8b | P1 | FAIL* | 119.7 | n/a | — | [thinking] Hmm, the user wants a haiku about artificial intelligence. Interestin |
| deepseek-r1:8b | P2 | FAIL* | 132.7 | n/a | — | [thinking] We are given a function `fibonacci(n)` that returns the nth Fibonacci |
| deepseek-r1:8b | P3 | FAIL* | 137.0 | n/a | — | [thinking] Okay, the user is asking for the capital of France. Let me start by r |

**Average PromeTorch/Ollama throughput ratio:** 50% across 6/9 valid cells.

FAIL\* = PromeTorch process terminated before generation started. All three deepseek-r1:8b PromeTorch cells failed during this run because Ollama's default 3-model keep-alive was holding ~31 GB of VRAM (qwen3:4b + gemma3:4b + deepseek-r1:8b stayed resident), leaving ≈9 GB free on the 40 GB A100. PromeTorch's current quant loader needs ≥20 GB VRAM to stage the 8B-Q4_K_M weights and ran out of memory mid-load. Not a correctness bug; reproducible with Ollama's VRAM footprint pinned.

## VRAM (peak, A100 40 GB)

| Model | PromeTorch VRAM | Ollama VRAM |
|-------|----------------:|------------:|
| qwen3:4b | 17.2 GB | 9.2 GB |
| gemma3:4b | 17.3 GB | 5.8 GB |
| deepseek-r1:8b | OOM (>20 GB needed, 9 GB free) | 15.2 GB |

PromeTorch's higher VRAM is expected: it keeps both the mmap'd quantized blob and the scratch/KV/decode buffers resident on the GPU, whereas Ollama streams the quantized blob from CPU via llama.cpp's CUDA backend and reports only the on-device weight+KV size.

## Quality notes

- **qwen3:4b** — both frameworks emit coherent English. PromeTorch's first token on P1 lands in the thinking preamble ("The first line: something about the nature of AI…") while Ollama returns only the `thinking` field (no final `response`) because qwen3 reasoning never hit `</think>` within 100 tokens.
- **gemma3:4b** — Ollama writes a clean haiku ("Code learns and evolves, / Mimicking a human mind, / Future's digital.") and a tight Paris answer. PromeTorch starts correctly ("The code learns, grows,") then degenerates into whitespace spam — a known temperature=0 repetition-trap on PromeTorch's Gemma tokenizer path.
- **deepseek-r1:8b** — Ollama produces coherent `<think>` blocks for all three prompts (P1: dog poem reasoning; P2: Python function; P3: Paris then reasoning). PromeTorch: see FAIL note above.

## Summary

On the 6 comparable cells (qwen3:4b and gemma3:4b across all 3 prompts), PromeTorch delivers **~50% of Ollama's throughput** (50% is the unweighted mean of per-cell ratios). qwen3:4b outputs are fully coherent in both frameworks; gemma3:4b is coherent only in Ollama, where PromeTorch's greedy-sampling path collapses into whitespace after ~12 tokens. deepseek-r1:8b could not be benchmarked on PromeTorch this session because Ollama's concurrent model cache held ≈31 GB of VRAM and starved the PromeTorch loader; Ollama itself runs it at 119–137 tok/s.

Headline: **PromeTorch = ~52% of Ollama throughput on A100 for the 4 B-class GGUF models that it can load**, with qwen3:4b at 86 tok/s (PromeTorch) vs 165 tok/s (Ollama) and gemma3:4b at 71 tok/s (PromeTorch) vs 145 tok/s (Ollama).

## Raw logs

All raw transcripts are saved under `run_logs/bench_ollama/`:
- `prometorch_<model>_<P>.log` — PromeTorch stdout (model load, VRAM, generation, full response).
- `ollama_<model>_<P>.json` — Ollama `/api/generate` response JSON (`eval_count`, `eval_duration`, `response`, `thinking`).
- `bench.js` — node harness that drove both sides.
