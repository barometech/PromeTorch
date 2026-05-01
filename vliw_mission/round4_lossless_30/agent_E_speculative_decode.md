# Agent E — Speculative decoding с tiny draft model для прыжка к 30 tok/s

## Роль
ML-инженер. Делает speculative decoding с обученной/выбранной draft
model. Это **путь к 30 tok/s** если чистая kernel-оптимизация (Agent D)
не дотянет до цели.

## Зависимости
**Независим** от Agent A/B/C на ранних этапах. На финале — интегрируется
с loader/inference path.

## Что прочитать
1. `vliw_mission/round4_lossless_30/MISSION.md`
2. `torch/io/speculative_draft.h` — `NgramDraft`, `predict_pld`
3. `torch/io/gguf_model.h:3650-3935` — `forward_decode_cpu_batched`,
   spec_decode_step_cpu, существующая verify-логика
4. `vliw_mission/gemini_speculative_decode.md` — старый research dump
5. `vliw_mission/round2/agent_*` — что искали раньше про spec decode
6. JOURNAL Phase 7 / Phase 7.2 / Phase 7.5 — измерения accept rate
   - **Финдинг:** 0% accept rate с n-gram draft на cross-scale qwen3
   - **Финдинг:** PLD регрессирует на Эльбрусе из-за batched K-serial

## Что должен сделать

### Часть 1: Анализ feasibility "30 tok/s через spec decode"
Базовая math:
```
tps_spec = tps_base × (1 + accept_rate × K) / (1 + α)
```
где `α = (verify cost - single forward cost) / single forward cost` —
overhead batched verify. На Q8 SoA4 batched verify ≈ серийный (потому
что N большой, K малый), α ~ 0.1-0.3.

При tps_base = 9.4 (ну или 13 если Agent D дотянул):
| accept | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|
| 30% | 11.5 | 12.2 | 12.4 | 12.5 |
| 50% | 13.6 | 15.3 | 16.3 | 16.9 |
| 70% | 15.8 | 18.7 | 21.0 | 22.7 |
| 90% | 17.9 | 22.4 | 26.8 | 30.5 |

**Вывод:** 30 tok/s достижимо при accept rate **≥ 90%** + K≥4. Это
*очень* высокая планка. Достижимо только с tuned draft.

### Часть 2: Draft model selection
Кандидаты:
1. **qwen3:0.6B Q4_K_M** — same family, размер 8x меньше. Загрузка
   дополнительно ~430 MB. На Эльбрусе ~30 tok/s draft inference. Накладные
   расходы verify должны быть ≤ outright forward time.
2. **DistilQwen-tiny** — distilled версии qwen, специально для drafting.
   Проверить наличие в HF.
3. **EAGLE / Medusa heads** — добавляются на основную модель, генерируют
   drafts параллельно. Требуют finetuning.

### Часть 3: Implementation
1. Загрузить draft model отдельным `GGUFModel` instance (или sub-model).
2. Per token decode loop:
   ```
   for step:
     drafts[K] = draft_model.generate(K)        # cheap
     verify_logits[K] = main_model.forward_batched(drafts)  # expensive
     accepted = match_prefix(drafts, argmax(verify_logits))
     yield accepted tokens
     if last accepted < K: yield argmax(verify_logits[last])
   ```
3. Tree-spec decoding: K параллельных drafts по разным веткам, accept
   максимальный prefix.

### Часть 4: Tuning
- Find optimal K for each accept rate
- Adapt K dynamically (если последний accept_rate низкий, K↓)
- Fast path для repetitive contexts (return to NgramDraft с K=5)

### Часть 5: Quality check
Critical: speculative decoding bit-exact как greedy если correctly
implemented. Acceptance criteria:
- При temperature=0 (greedy), output identical к non-spec версии. **Bit-exact.**
- При temperature>0, distribution равна base distribution. (rejection
  sampling корректно реализован)

## Output

1. `vliw_mission/round4_lossless_30/agent_E_results.md` — accept rates,
   timings, end-to-end tps
2. `torch/io/spec_decode_v2.h` — production speculative decoder
3. Tests: `tests/io/test_spec_decode_quality.cpp` (bit-exact greedy
   regression)
4. `scripts/run_spec_elbrus.sh` — env-controlled запуск с draft model

## Constraints
- **Quality regression = НЕТ.** Если tps↑ но accuracy↓ — revert.
- Draft model должен **загружаться вместе** с main, не отдельная сессия.
- Max accept rate за бесплатно = ~50% (n-gram). Для 90% нужен trained.
- **Если за сессию не выходит на 30 tok/s** — задокументируй honest
  potential: при каких accept/K возможно, какой draft нужен.
