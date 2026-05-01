# Round 4 — статус ночи 2026-04-30

## ✅ Что сделано

### Step 0 — Baseline confirmation
- 3-run TP-4 SoA = 9.5 tok/s median ✓

### Step 1 — Persistent ThreadPool (commit a338ae6, push 514191c)
- `c10/util/Futex.h` + `c10/util/ThreadPool.h` rewrite
- Microbench: 100 → 17 µs/call (×6 в production-config)
- TP-4: 9.5 → **9.9 tok/s** (+4%)
- Lossless: identical text output ✓
- README/BENCH обновлены, push на GitHub

### Profile data собран (PT_PROFILE_LAYER=1)
```
attn_phase:         25.72 ms
attn_output:         4.54 ms
allreduce(ao):      14.85 ms  ← главный sync overhead
gate_up:            19.46 ms
ffn_down:           28.04 ms  ← главный compute bucket
allreduce(fdown):    1.39 ms
output_proj:        19.49 ms
```

## ❌ Step 7 — manual prefetch (попробовал, откатил)
- Добавил `__builtin_prefetch` в q8_soa_repack — регрессия −0.1 tok/s
- LCC уже делает adequate prefetch, manual конфликтует
- Reverted

## ⏸ Не сделано (riskily — отложено)

- **Step 4 — PT8_Q4_SOA4 kernel** — большая работа (~400 LoC + microbench),
  без юзера для отладки рискованно. Открывает BW ceiling 49 tok/s.
- **Step 8 — Reduce-scatter для attn_output AR** — saving 11 ms/token
  (~+10%). Сложная синхронизация cross-rank без тестов.
- **Steps 5/6/9** (kernel fusion, SWP cleanup, tail SIMD) — отложены.

## Honest читалка к 30 tok/s

Текущий 9.9 tok/s.
Realistic next steps без spec decode:
- Step 8 (RS attn_output): +1 tok/s → ~11
- Step 4 (Q4_SOA4): +5-7 tok/s → ~16-18
- Steps 5+6+9 (fusion+SWP+tail): +3-5 tok/s → ~21-23

Без Self-Speculative LayerSkip (Gemini Q3 рекомендация) **30 tok/s
потолок** ~22-24. Self-Speculative требует bit-exact greedy implementation
+ batched verify (2-3 sessions работы).

## Файлы для будущих сессий

- `vliw_mission/round4_lossless_30/MASTER_PLAN.md` — пронумерованный план
- `vliw_mission/round4_lossless_30/format_spec_v1.md` — Agent A PT8 spec
- `vliw_mission/round4_lossless_30/agent_D_results.md` — kernel optimization
- `vliw_mission/round4_lossless_30/agent_E_results.md` — speculative decode
- `vliw_mission/round4_lossless_30/gemini_response.md` — DR Max Q&A
- `tools/gguf2pt8/` — Agent B converter skeleton (готов к доработке)
- `torch/io/pt8_reader.h` — Agent C loader

## Финальное измерение

```
PT_Q8_SOA=1 ./scripts/run_tp_elbrus.sh --greedy "Hello"
→ 100 tokens in 10.2s = 9.9 tok/s
→ output bit-identical to Q4_K_M baseline (lossless ✓)
```

GitHub `barometech/PromeTorch` HEAD = `514191c`. 53 stars.
