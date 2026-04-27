# Round 3 — 10 параллельных Opus 4.7 агентов после bandwidth plateau (2026-04-27)

## Состояние

Production: qwen3:4b Q4_K_M TP-4 = **4.8 tok/s** на Эльбрусе 8СВ.
Bandwidth peak per chip = 12.2 GB/s (memcpy). Effective = 2.9 GB/s = **23%**.
Compute peak per core = 72 GFLOPS (Trushkin). Effective = ~12 GOPS = **17%**.

Профайлер раскладка ms/token (TP-4):
  gate_up:    65.4   (RMSNorm + gate + up GEMV, fused)
  ffn_down:   48.9   (incl SiLU * up)
  attn_phase: 29.9   (RMSNorm + QKV + attention)
  output_proj:23.7
  attn_output:15.0
  allreduce: 11.7
  TOTAL:     211 ms/token

## Что НЕ помогло (исключить из фокуса)

- Option F (gather + futex): 0% — sync был всего 6%, не bottleneck
- fp32-prelude scales (fp16 conversion out of hot loop): 0% — LCC и так пайплайнит
- VNNI qpmaddubsh micro-bench: 12× в синтетике, 0% в проде (LCC's fapb уже линейный prefetch)
- Q8_0 conversion: bandwidth × 1.83 → хуже (1.5 vs 4.8 tok/s)
- __builtin_prefetch hints в scalar Q4_K: 0% (LCC fapb redundant)

## Что МОГЛО бы помочь (подтверждённые пути)

- parallel_for fork/join overhead: ~200 calls/token × 100 μs = **20 ms/token saving** (Agent #2, Round 2 finding)
- Batch decode / speculative: amortize weight read across N токенов
- Smaller model (0.6B = 22.8 tok/s already)

## Миссия Round 3

10 опус-агентов работают параллельно по разным направлениям. Каждый получает:
- путь к репо: C:\Users\paper\Desktop\promethorch
- журнал: JOURNAL.md (последние записи Round 2 + plateau + Q8_0)
- predecessor agents: vliw_mission/round2/agent_*.md
- профайлер цифры выше
- ОДИН конкретный угол атаки

Цель — **поднять 4B Q4_K_M TP-4 с 4.8 до 10+ tok/s** через software-only changes.
