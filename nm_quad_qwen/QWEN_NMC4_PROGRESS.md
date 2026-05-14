# Qwen3-4B NMC4 port — Progress 2026-05-14

## Цель
Qwen3-4B inference на NMC4 ядрах NM Quad **без CPU compute** — token generation end-to-end на DSP.

## Сегодня: 21+ commits в port (4c21b6a → 090fd14)

### ✅ Atomic ops + composed chains BIT-EXACT (10/12)

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

### ✅ Composed chains bit-exact (FIXED 2026-05-14 после fresh bisect)

| Chain | Pipeline | Wall | max_diff | Commit |
|-------|----------|------|----------|--------|
| attn_full | RMSNorm+QKV+norm+RoPE+attn+Wo+residual | 182ms | **5.96e-08 BIT-EXACT** | `ee5cbc1` |
| full_layer | attn + FFN (18 ops end-to-end) | 5.5s | **1.19e-07 BIT-EXACT** | `090fd14` |

**Root causes найдены и зафиксированы:**

1. Q4_K block_dot — 8-accumulator reduction tree вместо single-acc
2. Q6_K block_dot — same 8-acc fix
3. Host `q6k_dot_h` для Wd — multi-block (FFN_DOWN_BLOCKS) loop, не single-block
4. Host `q4k_dot_h` / `q6k_dot_h` — переписаны на 8-acc reduction для bit-exact host vs NMC
5. NMC Q4_K/Q6_K GEMV M>128 splits на M=128 chunks (Wo, Wgate, Wup, Wd, Wv)
6. **PL_WriteMemBlock byte-address overlap**: Wup (1.47M PL_Words) overwrites Wv в EMI. Fix: re-upload Wv после всех других weights — последняя запись побеждает.

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
