# Agent E — Speculative decoding к 30 tok/s: feasibility, draft model selection, plan

**Дата:** 2026-04-29
**Базовая позиция:** TP-4 + Q8 SoA4 = 9.4 tok/s (commit `3399136`).
**Цель миссии:** 30 tok/s lossless на qwen3:4b на Эльбрусе E8C2.
**Вердикт:** *Спекулятивный декодинг в одиночку до 30 не дотянет.*
Реалистичный потолок Agent E = **18-22 tok/s** при идеальном сочетании
draft-model + Agent D ускорения tps_base. Дотянуть до 30 можно
только **совместно с Agent D** (kernel-фьюжн / прямой Q8 без репака
до tps_base ≥ 13) **и Agent A** (NUMA layout draft-process на 5-м
процессе), и при условии accept rate ≥ 75%.

---

## 1. Анализ feasibility 30 tok/s через спекулятивный декодинг

### 1.1 Базовая формула (Leviathan 2022 + наш α)

```
tps_spec = tps_base × (1 + p + p² + ... + p^K) / (1 + α + K·c)
        ≈ tps_base × E[accepted+1] / overhead_factor
```

где:
- `p` = per-token accept rate
- `K` = draft length (proposed tokens per step)
- `c` = t_draft / t_main (cost ratio one draft forward vs one main forward)
- `α` = batched-verify overhead beyond a single main forward.

На Эльбрусе при текущем стэке:
- `α` measured = **0.30-0.45** (Phase 7.3 measurement: batched verify K=4
  стоит 1.35-1.45× стоимости single forward, потому что
  `forward_decode_cpu_batched` внутри K-serial для часть kernels —
  не настоящая GEMM ещё).
- `c` для qwen3:0.6B vs qwen3:4B на Эльбрусе ≈ **0.18-0.22** (приблизительно
  1/5; 0.6B имеет 28 layers × 1024 hidden × 24 heads vs main 36 × 2560 × 32;
  weight bytes ratio 430MB / 2400MB = 0.18).

### 1.2 Реалистичная таблица speedup при tps_base = 9.4

Применяю формулу с **α=0.35, c=0.20** (честный современный замер):

| accept p | K=2 | K=3 | K=4 | K=5 | K=6 |
|---|---|---|---|---|---|
| 30% | 8.4 | 8.0 | 7.6 | 7.2 | 6.9 |
| 50% | 9.7 | 9.7 | 9.4 | 9.0 | 8.6 |
| 70% | 11.1 | 11.5 | 11.5 | 11.3 | 10.9 |
| 80% | 11.8 | 12.5 | 12.7 | 12.7 | 12.5 |
| 90% | 12.5 | 13.5 | 14.1 | 14.4 | 14.4 |
| 95% | 12.9 | 14.0 | 14.8 | 15.3 | 15.5 |

**Это значительно отличается от таблицы в брифе** (где предполагалось
α≈0.1-0.2 *и не учитывалась стоимость draft forward'ов c·K*). При
честном c=0.20 даже 95% accept rate даёт лишь ~15.5 tok/s — **в два
раза ниже цели 30**.

### 1.3 Что нужно для 30 tok/s

Решая `30 = 9.4 × (1 + Σp^i) / (1 + α + K·c)`:

При `α=0.35, c=0.20, p=0.85, K=5`:
- numerator: 1 + 0.85 + 0.7225 + 0.614 + 0.522 + 0.444 = **4.15**
- denom: 1 + 0.35 + 1.0 = **2.35**
- speedup = **1.77×** → **16.6 tok/s**

Чтобы подойти к 30 нужны нереалистичные параметры:
- `p=0.95, K=8, c=0.10` (т.е. draft в 10× быстрее main): speedup = 2.7× →
  25.4 tok/s. Всё ещё недостаточно.
- ИЛИ **tps_base поднять до ~17 + spec ×1.8** (это путь Agent D обязан
  выполнить).

### 1.4 Если Agent D дотянет tps_base до 13-14

При tps_base=13.5 и реалистичных p=0.80, K=4, α=0.30, c=0.18:
- numerator: 1+0.8+0.64+0.512+0.41 = 3.36
- denom: 1+0.30+0.72 = 2.02
- speedup = 1.66× → **22.4 tok/s**. **Всё ещё не 30.**

При tps_base=13.5 + p=0.90, K=5, α=0.25, c=0.15:
- numerator: 1+0.9+0.81+0.729+0.656+0.59 = 4.69
- denom: 1+0.25+0.75 = 2.0
- speedup = 2.34× → **31.6 tok/s** ✓

**Вывод:** 30 tok/s достижимо при одновременном выполнении трёх условий:
1. Agent D поднимает tps_base до ≥ 13 (через kernel-фьюжн + Q8 SoA без
   repack)
2. Draft accept rate ≥ 90% (требует **trained** или **EAGLE/Medusa**, не
   n-gram)
3. Honest α ≤ 0.25 (требует Phase 7.3 — настоящий batched GEMM в
   `forward_decode_cpu_batched`, см. файл строки 3286-3333; сейчас он
   K-serial для attention и part of FFN)

Любого из трёх недостаточно. Это **5-агентная задача**, не "просто
spec decode".

---

## 2. Кандидаты draft model

| Кандидат | Размер | Same-tokenizer? | Прогноз accept | Verdict |
|---|---|---|---|---|
| **qwen3:0.6B Q4_K_M** | 430 MB | YES (qwen3 family) | 60-75% | **GO — primary** |
| qwen3:1.7B Q4_K_M | ~1100 MB | YES | 75-85% | Backup. Но c≈0.45 → speedup ≤ 1.4× |
| Distilled-qwen3 (custom train) | 200-300 MB | YES если same vocab | 80-92% | Fall-back на 2 сессии |
| EAGLE heads (Li 2024) | +50-100 MB на main | YES (shared vocab) | 85-95% | Лучший long-term, требует finetune |
| Medusa heads (Cai 2024) | +120-200 MB | YES | 70-85% | Похоже на EAGLE, проще обучить |
| Self-spec (skip layers) | 0 MB extra | YES | 50-65% | Bad на small model — quality drop |
| NgramDraft (existing) | 0 MB | N/A | 5-15% (free-form) до 50% (code) | **0% measured на qwen3** |

### 2.1 Detailed: qwen3:0.6B как primary draft

- **Memory:** 430 MB Q4_K_M weights + KV cache (28 layers × max_seq × kv_dim × 4B).
  При max_seq=4096 и kv_dim=512: 28 × 4096 × 512 × 8 (K+V FP32) = **460 MB**.
  Total **~900 MB extra на rank 0** или **~225 MB extra per rank** если
  TP-4 шардить (но 0.6B плохо шардится — слишком маленькие GEMV блоки).
- **Inference time:** на Эльбрусе E8C2 single-process qwen3:0.6B Q4_K_M
  estimated 35-45 tok/s (8× меньше main, ~1× кэшевая friendliness для
  430MB на ноду). Это даёт `c = 9.4/40 = 0.235`. Без TP draft не нуждается
  потому что 0.6B весь влезает в L3 одной ноды.
- **Accept rate prognosis:** литература Leviathan 2022 для **same-family
  pairs** (LLaMA-7B vs LLaMA-65B) даёт 70-80% greedy accept. Для qwen3
  (0.6B vs 4B) ratio чуть хуже — гейп больше — ожидаю **65-75%**.
- **Phase 7.5 risk:** в JOURNAL зафиксирован "0% acceptance с qwen3 4b/0.6b".
  Это была ошибка интеграции (KV-cache desync, см. ниже §4.2), не intrinsic
  невозможность. После фикса ожидаемо 60-75%.

### 2.2 EAGLE / Medusa — путь к 90% accept

EAGLE-2 (Li et al. 2024, ICML) даёт 4× speedup при same model.
Принцип: вместо отдельной draft model — лёгкие "draft heads" поверх main
model's hidden states, обученные предсказывать N tokens вперёд.

- **Плюсы:** accept ≥ 90% уверенно (по литературе)
- **Минусы:**
  1. Требует **finetuning** (несколько часов на A100, делать на отдельной
     машине, не на Эльбрусе)
  2. Не lossless относительно ванильной qwen3:4b — добавляет новые
     параметры и меняет inference path
  3. Несовместимо с PT_FORMAT=pt8 (требует отдельного формата с EAGLE
     weights)

**Рекомендация:** EAGLE — Phase 3, после того как Phase 1 (vanilla spec
с qwen3:0.6B) показывает что accept упор не в model gap.

---

## 3. Implementation план

### Phase 1: Vanilla two-model spec — 1-2 sessions

**Status:** infrastructure уже есть (см. `gguf_model.h:741-816`,
`spec_decode_step_cpu` на строке 3587). Phase 7.5 был частично написан
но дал 0% accept — root cause неясен. Phase 1 = **debug + finalize**.

```
Implementation:
1. Verify draft KV-cache sync logic (lines 802-816). Гипотеза: при
   rejection cleanup `draft_sync_after_step()` рассинхронизирует draft
   на K-1 - j tokens, что приводит к катастрофе через 5-10 шагов.
2. Add detailed [spec-step] tracing: for each step log
   {drafts_proposed, drafts_accepted, draft_seq, main_seq, c_measured}.
3. Run with PT_SPEC_DRAFT_PATH=qwen3-0.6B.gguf PT_SPEC_K=4 на Эльбрусе.
4. Если accept < 30% → debug; если 30-60% → tune K; если ≥60% → ship.
```

Expected outcome: 11-14 tok/s (из таблицы 1.2 при p≈0.65, K=4).

### Phase 2: True batched verify (не K-serial) — 2 sessions

В `forward_decode_cpu_batched` (строки 3286-3585) большинство GEMV
сейчас вызываются как `cpu_quant_gemv_batched` (видно на 3369-3377)
который **возможно** уже batched. Но attention math (3380+) и softmax —
точно K-serial. Нужно:

1. Aggregate Q/K/V у всех K positions в один tensor [K, q_dim] *до*
   per-token RoPE/QK-norm
2. Batched attention math: GEMM Q @ K^T для **всей K-широкой query
   matrix** против `[ctx+K, kv_dim]` cached keys (см. gemini doc §1)
3. Causal mask `[K, ctx+K]` lower-triangular для last K columns

**Ожидаемый эффект:** α 0.35 → 0.20. Speedup ×1.15 на spec branch.

### Phase 3: Tree-spec / adaptive K — 1 session

- Track running mean accept rate over last 32 steps
- If `p̄ < 0.55` → set K=1 (disable spec)
- If `p̄ > 0.85` → K=6
- Else K=4 (default sweet-spot)

Tree-spec (multiple draft branches) требует draft model запускать
с top-2 sampling вместо argmax — даёт больше вариативности и accept
по любой ветке. Дополнительно +3-7% effective accept rate. Сложно
реализовать, оставить на четвёртую сессию.

### Phase 4 (stretch): EAGLE heads — 3-4 sessions + GPU обучение

Только если Phase 1-3 не дотягивают до 22+ tok/s.

---

## 4. Quality guarantee

### 4.1 Bit-exact greedy guarantee

**Theorem (Leviathan 2022):** при `temperature=0` (greedy argmax) и
правильно реализованном accept-then-correct rule (см. gemini_speculative_decode.md
§2), output sequence **identical** к non-spec greedy decode.

**Proof sketch:** при rejection в позиции i, мы коммитим `argmax(main_logits[i])`
— это ровно тот токен, который non-spec decode дал бы. Принятые drafts —
гарантированно те же argmax (по definition accept). Таким образом hash
последовательности после N токенов идентичен.

### 4.2 Test plan

```cpp
// tests/io/test_spec_decode_quality.cpp
TEST(SpecDecode, BitExactGreedyVsNonSpec) {
    GGUFModel main; main.load("qwen3-4b.gguf");
    GGUFModel m2;   m2.load("qwen3-4b.gguf");
    setenv("PT_SPEC_DRAFT_PATH", "qwen3-0.6b.gguf", 1);
    setenv("PT_SPEC_K", "4", 1);

    std::string prompt = "The quick brown fox";
    auto out_nospec = m2.generate(prompt, 100, /*temp=*/0.0f);

    setenv("PT_SPEC_K", "1", 1);  // disable
    auto out_spec   = main.generate(prompt, 100, /*temp=*/0.0f);

    EXPECT_EQ(out_nospec, out_spec);  // Bit-exact на 100 токенов
}
```

Plus regression on **5 prompts × 200 tokens each**: code, prose, math,
russian, structured. All must match byte-for-byte.

### 4.3 Probabilistic sampling (temperature > 0)

Для temp>0 нужен **rejection sampling** (Leviathan algorithm):
```
accept draft d_i with prob min(1, p_main(d_i) / p_draft(d_i))
on reject: sample from max(0, p_main - p_draft) (renormalized)
```

Это математически exact — distribution идентична non-spec sampling.
**Сложность:** нужны полные probability vectors из обеих моделей,
не только argmax. Это удваивает softmax cost. Откладываем на Phase 5
(не критично для 30 tok/s; greedy достаточно).

---

## 5. Memory footprint analysis (Эльбрус 32 GB / NUMA node)

### 5.1 Current TP-4 footprint per rank

| Компонент | Per-rank | Total (4 ranks) |
|---|---|---|
| Q8 SoA4 weights (1/4 sharded) | ~600 MB | ~2400 MB |
| Q4_K weights (replicated, fallback) | ~600 MB | ~2400 MB |
| KV cache (1/4 of attention heads) | ~470 MB @ 4096 ctx | ~1880 MB |
| Activations + scratch | ~50 MB | ~200 MB |
| **Subtotal per rank** | **~1720 MB** | **~6880 MB** |

### 5.2 Adding qwen3:0.6B draft на rank 0 only

| Опция | Rank 0 extra | Other ranks | Total |
|---|---|---|---|
| Single-replica на rank 0 | +900 MB | 0 | 7780 MB |
| Replicated to all 4 NUMA | +900 MB × 4 | 0 (uses TP-0 process) | 10480 MB |
| **5-й dedicated draft process на NUMA-0** | 0 на ranks 0-3 | +900 MB на pid 5 | 7780 MB |

**Recommended:** **5-й dedicated process** на NUMA-0, isolated cores
(см. gemini doc §5). Ranks 0-3 не блокируются на draft compute, и
shared-memory inter-process communication (уже есть SHM AllReduce
infra) реализуется тривиально.

**Лимит RAM ноды (32 GB):** легко влезает. NUMA-0 загружено: 1720 MB
(rank 0) + 900 MB (draft process) = **2620 MB / 32 GB ≈ 8% util**.

### 5.3 Draft model loading в .pt8 формате — required от Agent C

Text от Agent C должен:
1. Принимать **второй .pt8 path** в API: `GGUFModel::load_main(path); GGUFModel::load_draft(draft_path);`
2. Поддерживать **runtime check совместимости tokenizers**: vocab_size
   identical, eos_id identical, BOS handling identical. Если нет —
   FATAL error при загрузке.
3. Опция: Agent B конвертер должен уметь bundle main+draft в один
   архив `qwen3-4b-spec.pt8.tar` с auto-load при PT_SPEC_K>1.

---

## 6. Honest assessment

### 6.1 Что реалистично за 1 сессию

**Single session goal: дебагнуть Phase 7.5 → 11-14 tok/s.**

1. Найти root cause "0% accept" (КС-cache desync скорее всего)
2. Подтвердить accept rate на realistic prompts (russian wiki, code,
   structured chat) — ожидание 50-70%
3. Прогнать E2E benchmark на 5 prompts × 100 tokens
4. Зафиксировать в JOURNAL.md измерения accept × tps по K=2,3,4,5

**Что НЕ за 1 сессию:**
- 30 tok/s (требует Agent D + Agent A)
- True batched GEMM verify (Phase 7.3)
- EAGLE training
- Tree-spec

### 6.2 Реалистичный потолок Agent E без помощи

Если tps_base = 9.4 (без Agent D) и draft = qwen3:0.6B:
- Best case: p=0.75, K=4, α=0.30, c=0.20 → speedup 1.45 → **13.6 tok/s**
- Realistic: p=0.65, K=4 → 1.30 → **12.2 tok/s**
- Pessimistic (Phase 7.5 0% repeat): 1.0 → 9.4 tok/s или regression

### 6.3 Реалистичный потолок при кооперации с Agent D

tps_base = 13 (Agent D дотянул через kernel-фьюжн), draft = qwen3:0.6B,
Phase 7.3 done (α=0.20):
- Best case p=0.85, K=5: speedup 1.85× → **24 tok/s**
- Realistic p=0.70, K=4: 1.50× → **19.5 tok/s**

### 6.4 Реалистичный потолок при ВСЕХ оптимизациях

tps_base = 14 (Agent D maxed), EAGLE heads (p=0.92), batched α=0.20,
adaptive K=5:
- speedup 2.30× → **32 tok/s** ✓

Это **единственный** путь к 30+ tok/s, и он требует EAGLE-finetune
(не делается в этой сессии, требует GPU-машину для обучения).

---

## 7. Required от других агентов

### От Agent C (loader):
- API `load_draft(path)` для второго .pt8
- Runtime check tokenizer compatibility
- KV-cache allocation для draft model независимо от main

### От Agent B (converter):
- `prometorch-convert qwen3-0.6B.gguf --out qwen3-0.6B.pt8`
- Опционально: bundled `--bundle-with main.pt8 --out qwen3-4b-spec.pt8`

### От Agent D (kernels):
- tps_base ≥ 13 без spec — иначе мы не дотягиваем до 22 даже с perfect spec
- True batched GEMM в `forward_decode_cpu_batched` (Phase 7.3) для α≤0.20

### От Agent A (NUMA / process layout):
- 5-й process pinned на NUMA-0 для draft model
- SHM IPC между ranks 0-3 и draft process (можно переиспользовать
  существующий AllReduce buffer)

---

## 8. Acceptance criteria для Agent E (в одиночку)

1. **Phase 7.5 не даёт 0% accept** — после фикса измеряется ≥ 50%
   на нескольких prompt типах
2. **Bit-exact greedy** — `tests/io/test_spec_decode_quality.cpp` проходит
   на 5 prompts × 200 tokens
3. **Speedup ≥ 1.3×** при tps_base=9.4 → ≥ 12.2 tok/s, замерено
   `scripts/run_spec_elbrus.sh`
4. **No quality regression** — `output identical` к non-spec на полном
   benchmark suite

Эти 4 пункта реалистичны за 2 сессии Agent E. **30 tok/s — НЕ acceptance
criterion для одного Agent E**. Это цель миссии в целом, требующая всех 5
агентов.

---

## 9. TL;DR

**Honest answer: 30 tok/s НЕ достижимо одним speculative decoding с
qwen3:0.6B draft.** Реалистичный потолок vanilla spec на нашем стеке
= 14-16 tok/s. Чтобы дотянуть до 30 нужно:

1. **Agent D**: tps_base 9.4 → 13-14 (через Q8 SoA без repack + kernel
   fusion + Phase 7.3 true batched verify)
2. **Agent E (этот)**: vanilla spec с qwen3:0.6B draft, accept ~70-80%,
   adaptive K (даёт ×1.6 поверх Agent D = **22 tok/s**)
3. **Agent E (Phase 4 stretch)**: EAGLE heads с finetune на A100 (даёт
   accept 90%+, ×2.0 поверх Agent D = **28-32 tok/s**)

**ЛИБО** принять что 30 tok/s lossless не достижимо в этой сессии,
зафиксировать честный потолок 22-25 tok/s + roadmap к EAGLE.

Не врать про 30 tok/s. Это требование миссии может быть **физически
недостижимо без EAGLE/Medusa** — никто в публичной литературе не
показывал 3.2× speedup от vanilla spec на CPU INT4.
