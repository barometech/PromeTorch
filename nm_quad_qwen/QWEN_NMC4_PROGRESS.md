# Qwen3-4B NMC4 port — Progress 2026-05-14

## Цель
Qwen3-4B inference на NMC4 ядрах NM Quad **без CPU compute** — token generation end-to-end на DSP.

## Сегодня: 13 commits в port (4c21b6a → 658cc4b)

### ✅ Atomic ops bit-exact на real Qwen3-4B weights (9/12)

| # | Op | Detail | Commit |
|---|----|--------|--------|
| 1 | Q4_K GEMV 4-core M=32 | 9.88 MMACs/sec, max_diff 4e-7, DMA race fix | `9186e76` |
| 2 | Q4_K GEMV tile M=4096 | 32 invocations, full Wq, max_diff 0.17 | `152623a` |
| 3 | Q6_K GEMV M=128 | bit-exact 0 diff, 253ms на 1 core, shift fix | `9717017` |
| 4 | RMSNorm K=2560 | 3.3ms, max_diff 3e-8 | `1d331ed` |
| 5 | NEOX RoPE HEAD_DIM=128 | 3.3ms, max_diff 4e-7 | `0be2443` |
| 6 | Attention single-head | Q·K + softmax + V·scores, 55ms, max_diff 1.5e-8 | `22427cf` |
| 7 | SiLU+Softmax | SiLU K=8192, softmax K=128, 14.8ms | `22427cf` |
| 8 | Step1 composed | RMSNorm + Q-proj together, max_diff 3e-8 | `aaaa51f` |
| 8.5 | FFN substep | RMSNorm+Wgate+Wup+SiLU*u, max_diff 7e-8 | `749e806` |

### ⚠ Composed chains: runs end-to-end но не bit-exact

| Chain | Pipeline | Wall | max_diff | Commit |
|-------|----------|------|----------|--------|
| attn_full | RMSNorm+QKV+norm+RoPE+attn+Wo+residual | 182ms | 1.01 @ row 1536 | `5c15548` |
| full_layer | attn + FFN | 1140ms | 1.04 @ row 1536 | `d4967f9` |

**Bug location:** worst row 1536 same в обоих → bug specifically в **Wo @ attn_concat block** или в residual `x + attn_out`. Все sub-ops bit-exact в isolation, но composition divergence.

Подозрение: `q4k_block_dot` в attn_full kernel использует single-accumulator order vs standalone Q4_K kernel который 8 parallel accumulators. Возможен float drift, но magnitude 1.0+ suggests systematic difference (не drift).

### ❌ Pending (Steps 10-12)

10. **36-layer loop**: host orchestrator с per-layer weight streaming EMI.
11. **lm_head 151936-vocab Q6_K**: tile-based GEMV, 4-chip parallel.
12. **KV cache + token sampling**: prefill multi-pos + argmax + chain.

## Эталон working на NM Quad host x86 (не NMC4)

Python pipeline в `nm_quad_qwen/qwen_multi_token.py` генерирует `"Once upon a time, there was a young man who was"` — commit `4c21b6a`. 26s/token (CPU).

## Cron состояние

5 staggered crons каждые 15 мин (session-only, 7-day auto-expire):
- :00,:15,:30,:45 — финальная цель reminder
- :03,:18,:33,:48 — next step prompt
- :06,:21,:36,:51 — делай / не отчёт
- :09,:24,:39,:54 — commit или смерть
- :12,:27,:42,:57 — quick prompt
