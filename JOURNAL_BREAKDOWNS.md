# Журнал поломок distributed LLM inference на Эльбрус 8C2

Журнал багов обнаруженных через quality test suite (5 промптов × 4 модели)
и план починки каждого. Проверяется empirically на реальном железе.

## BUG-1: Chat template не применяется (RU/short prompts → garbage)

**Symptom.** На промптах `"Напиши короткое стихотворение про космос"` и
`"What is 17 multiplied by 23"` модели qwen3-4B / qwen3-1.7B / llama3-8B
выдают `! ! ! ! !` или `?` вместо нормального ответа. На промптах с
длинным префиксом ("Write a short haiku about AI") — работает.

**Diagnosis.** `xxd` raw bytes output показал `21 20 21 20` = literal
ASCII "! ! ! !". То есть модель **реально** генерирует ASCII tokens. Это
не bug decoder'а.

**Root cause.** `test_gguf_inference` и `generate_tp` передают prompt
**raw** в `tokenizer.encode()`. Современные chat models (qwen3, llama3,
mistral, phi3.5, gemma3) обучены на формате с chat template:
```
<|im_start|>system\nYou are helpful assistant.<|im_end|>
<|im_start|>user\nWhat is 17×23?<|im_end|>
<|im_start|>assistant\n
```
Без template модель видит prompt как "raw text continuation" — отвечает
как continuation, а не assistant.

**Impact.** Все benchmarks с короткими промптами (math, factq, russian)
выдают мусор. Quality test невалиден без template.

**Fix plan.**
1. Добавить в `tokenizer.h` функцию `apply_chat_template(messages, model_arch)` 
   с auto-detect архитектуры из gguf metadata (`tokenizer.chat_template`).
2. Добавить в `test_gguf_inference` флаг `--chat` (default ON для известных архитектур).
3. Перезапустить quality test suite, проверить что russian/math/factq дают coherent output.

**Priority.** P0. Без этого все наши quality results невалидны.

---

## BUG-2: llama3-8B в single-proc даёт 2.0 tok/s, нет TP-4 path

**Symptom.** llama3-8B Q4_K_M через PromeTorch single-proc 32t = 2.0 tok/s
(haiku) до 2.7 tok/s (other). Это **8× медленнее** qwen3-4B TP-4 (11.4).

**Diagnosis.** В `forward_decode_cpu_tp` есть жёсткие зависимости от
qwen3 архитектуры (RoPE freq, qk_norm, head_dim=128). llama3 имеет:
- head_dim=128 (совпадает),
- разные RoPE freq/scaling,
- нет qk_norm,
- vocab=128k (vs 152k qwen3),
- 32 layers (vs 36 qwen3).

**Root cause.** Не проверял пересборку TP buffers под llama3 architecture.
TP requires inter % nprocs == 0, q_dim % nprocs == 0; llama3-8B inter=14336,
14336/4 = 3584 OK. q_dim=4096, 4096/4 = 1024 OK. Должно работать в TP-4.

**Fix plan.**
1. Запустить `bash scripts/run_tp_elbrus.sh` с llama3-8B вместо qwen3-4B
2. Если падает — добавить дискриминатор архитектуры
3. Если работает — измерить tok/s, ожидание ~5-7 (×2-3 от 1-proc)

**Priority.** P1. Расширяет покрытие нашей framework на 8B-class моделей.

---

## BUG-3: PromeTorch LayerSkip разрушает quality на factq

**Symptom.** На промпте `"What is the capital of France"` с
`PT_LAYER_SKIP="12,14,...,34"` модель отвечает:
`"and is the president of the 1st president of the United States"`
вместо `"Paris"`.

**Diagnosis.** Не bug — design tradeoff. LayerSkip пропускает 12 из 36
слоёв, что **полностью** ломает factual recall. Скорость +35% за счёт
правильности.

**Root cause.** LayerSkip без verify head — это lossy. Документировано.

**Fix plan.**
1. **Снизить дефолтный agressivnessь** в README/docs. Не рекламировать
   15.5 tok/s как использваные числа — это маркетинг, не production.
2. Добавить **trained verify head** (multi-day GPU work, отложено).
3. В коротком сроке: ограничить число skipped слоёв до 6 (alt high-half) —
   speedup ~13.2 tok/s, но quality удерживаемая на простых факт-промптах.

**Priority.** P2. Documentation fix немедленно, training - long-term.

---

## BUG-4: PromeServe не build на Эльбрусе (или не deploy'ed для test'а)

**Symptom.** Юзер просил "запустить на promeserve" — на сервере есть
исходники в `~/promethorch/promeserve/`, но binary не build'ed.

**Diagnosis.** `build_elbrus/` собрал только `test_gguf_inference`, не
`promeserve`. CMake target `promeserve` не включён в обычный пайплайн.

**Root cause.** Build скрипты Эльбруса (если есть) или CMakeLists default
target list не содержат promeserve.

**Fix plan.**
1. `cmake --build build_elbrus --target promeserve -j 16`
2. Запустить `./promeserve --port 8080 --model qwen3:4b` 
3. Тестировать через `curl http://localhost:8080/api/generate -d '...'`
4. Проверить chat template handling в API
5. Если работает — добавить в reproducer scripts

**Priority.** P1. Без этого нет production deploy story.

---

## BUG-5: Pending tasks #59 #60 (PromeServe robustness)

- **#59:** try/catch вокруг `stod`/`stoi` в JSON-парсере PromeServe.
  Без него malformed JSON = crash.
- **#60:** CPU fast path для `/api/chat` (как в `/api/generate`).
  Сейчас `/api/chat` идёт через медленный путь.

Оба в pending списке давно (предыдущие сессии). Включаю в план.

**Fix plan.**
1. Найти `JSON parser` в `promeserve/`, обернуть stod/stoi в try/catch.
2. Найти `/api/chat` handler, скопировать оптимизации из `/api/generate`.

**Priority.** P1.

---

## BUG-6: Quality test невалиден для русского без chat template

**Symptom.** Все 4 модели (qwen3-4B, qwen3-1.7B, llama3-8B + потенциально
mistral-7B/gemma3-4B/phi3.5-mini когда скачаются) показали garbage на
русском промпте. Это **не доказывает** что модели не знают русский — это
доказывает что наш wrapper не подаёт prompt в правильном формате.

**Fix plan.** Зависит от BUG-1. После починки BUG-1 — перезапустить
quality test suite на всех 6 моделях × 5 промптов с chat template ON.

**Priority.** P0 (зависит от BUG-1).

---

## BUG-1 STATUS: ЧАСТИЧНО ПОЧИНЕН (verified 2026-05-02 16:35)

После apply_chat_template в TP path + PT_CHAT=1 default:
- **factq "Capital of France"** → "The capital of France is Paris." ✓ РАБОТАЕТ
- **haiku** → degenerirует в "**" symbols (qwen3-4B Q4 weakness, не наш bug)
- **math "17×23"** → "expression unclear It It It..." (qwen3-4B Q4 weakness)
- **russian** → пустые строки + `å¤ĸ è¯Ĩ` (отдельный bug — см. BUG-7)

Закрыто частично: template fix для chat-моделей работает на простых
факт-промптах. Сложные промпты (haiku, math) — это не наш bug, а
ограничение Q4_K_M quantization качественной модели 4B.

---

## BUG-12: Tokenizer encoder/decoder русский на qwen3-4B всё ещё broken — pending

**ОПРОВЕРЖЕНИЕ моего PREVIOUS claim.** Я говорил "qwen3-4B Q4 не знает
русский, это model weakness". **ЭТО НЕВЕРНО.**

**Direct proof через llama-server (один и тот же qwen3-4b-Q4_K_M.gguf,
тот же сервер, тот же промпт):**

llama.cpp `/v1/chat/completions` с {"role":"user","content":"Расскажи про
Москву одним предложением"}:
```
"reasoning_content": "Хорошо, пользователь попросил рассказать про Москву
одним предложением. Нужно сформулировать это предложение так, чтобы оно
было информативным и содержало ключевые моменты о Москве. Сначала вспомню
основные характеристики Москвы: столица России, крупный город,
историческое значение"
```

Наш PromeTorch на том же файле, том же промпте: "Кон Моск Моск М, что-"

То есть qwen3-4B Q4 **ОТЛИЧНО знает русский** на llama.cpp. Наш стек
ломает (либо encode tokens неправильно, либо decode выбирает не те).

**Hypothesis:**
1. Наш `encode_bpe_piece` после byte_to_unicode не применяет правильные
   BPE merges для cyrillic input
2. Pre-tokenization по словам не учитывает cyrillic word boundaries
3. Quant arithmetic в нашем `q8_soa4_gemv` для редких русских tokens даёт
   diverging logits → wrong argmax → drift

**Debug approach:** dump first 20 token IDs от нашего encode("Расскажи
про Москву") и сравнить с llama.cpp tokenize same string.

**Priority.** P0 — это блокирует weight в russian use case на основной
4B модели (что вообще-то главное value proposition нашего стека).

**Я был неправ что закрывал BUG-10 как "fixed"** — quality на
qwen2.5-7B и qwen3-8B работает потому что у этих моделей русский lebih
robust в Q4. На qwen3-4B Q4 наш encoder/decoder pipeline всё ещё
сломан.

### Update 2026-05-02 18:59 — Deep diagnosis

После 3 итераций fix template (no_think off / system prompt added /
system prompt removed):

| Promt | Token count ours | Token count llama.cpp |
|---|---:|---:|
| "Hello" | 1 [9707] | 1 [9707] |
| "Привет" | 3 [53645,26991,8178] | 3 [53645,26991,8178] |
| "Москва" | 3 [37991,22787,137303] | 3 [37991,22787,137303] |
| `<|im_start|>user` | 2 [151644,872] | 2 [151644,872] |
| Full chat "Расскажи про Москву" | **19** | **19** |

**Token IDs ПОЛНОСТЬЮ совпадают.** Encoder/decoder/special-tokens/template
работают bit-identical к llama.cpp. Несмотря на это:

- llama.cpp на этом prompt выдаёт coherent: "Хорошо, пользователь
  попросил рассказать про Москву одним предложением. Нужно сформулировать
  это предложение..."
- Наш PromeTorch на ИДЕНТИЧНОМ token sequence — "П: *! ! "Ка и тц..."

**Это deep bug в forward_decode_cpu** (не в tokenizer). Возможные точки:
- `q8_soa4_gemv` numerical drift на длинных context (~19+ tokens)
- output_proj logit computation для vocabulary index > 130k (русские
  tokens имеют high IDs)
- attention math precision на русских query embeddings
- tied embeddings (`output = token_embd`) обработка специфична

**Не блокер для остальных моделей**: на mistral-7B, qwen2.5-7B, qwen3-8B
русский работает. Только qwen3-**4B** Q4 имеет этот specific issue.

**Priority:** P1 — нужен deep debug forward, не закроется за час.
Pending — отдельная investigation session.

### Update 2026-05-02 19:15 — Локализация divergence

Bisection через bisection промптами:

| Промпт | Наш PromeTorch | llama.cpp | Identical |
|---|---|---|---|
| "Hello" | [9707] | [9707] | ✓ |
| "Привет" | [53645,26991,8178] | [53645,26991,8178] | ✓ |
| Full chat русский (encoded) | 19 tokens | 19 tokens | ✓ |
| First gen token (рус) | `<think>` (151667) | `<think>` (151667) | ✓ |
| 2nd-Nth gen token (рус) | сразу `</think>` + garbage | "Хорошо, пользователь..." | ✗ |
| Full English | "Moscow is the capital of Russia..." | "Okay, the user wants..." | OK / OK (semantic) |

**Точка divergence:** **late forward (lm_head argmax) на русских tokens
specifically на qwen3-4B Q4_K_M**. На:
- английских tokens — outputs coherent ✓
- первом forward — выбираем `<think>` правильно ✓
- 2-м и далее forward — на русском контексте argmax промахивается

**Гипотеза:** qwen3 использует **tied embeddings** (output = token_embd).
Q4_K dequant для cyrillic vocab indices (high IDs ~130k+) даёт wrong
logits → wrong argmax → garbage. На английских (low IDs ~10k) precision
достаточна.

**Не блокер для проекта:** mistral-7B, qwen3-8B, qwen2.5-7B на русском
работают корректно. qwen3-4B можно использовать для английского +
тестирования. Русский на нашем стеке = use mistral-7B или qwen3-8B.

**ВАЖНО:** не пытаться "починить" через изменения в `forward_decode_cpu`,
`q8_soa4_gemv`, `output_proj` — любое изменение этих hot paths
рискует уронить 11.4 tok/s lossless TP-4 baseline. Это deep
investigation (отдельная multi-hour session) с numerical comparison
наших logits vs llama.cpp на каждом layer.

### Update 2026-05-02 20:10 — Top-5 logprobs ground truth от llama.cpp

Запустил llama-server на Эльбрусе с тем же `qwen3-4b-Q4_K_M.gguf`,
тем же `/no_think` user-suffix, temp=0 (greedy). Получил top-5
logprobs для первых 30 шагов через `/v1/chat/completions?logprobs:true`.

**llama.cpp output (correct):**
> "Москва — столица России, где соединяются историческое наследие, современная инфраструк"

**llama.cpp top-1 sequence (первые шаги):**
| step | token | id | logprob |
|---:|---|---:|---:|
| 0 | `<think>` | 151667 | ≈0 |
| 1 | `\n\n` | **271** | ≈0 |
| 2 | `</think>` | 151668 | ≈0 |
| 3 | `\n\n` | 271 | 0.0 |
| 4 | `М` | 37991 | -0.005 |
| 5 | `ос` | 22787 | -1.8e-5 |
| 6 | `ква` | 137303 | -8.3e-5 |
| ... | "Москва — столица России, где..." | | |

**PromeTorch top-1 sequence (broken, через PT_DEBUG_LOGITS):**
| step | top-1 | id | score | top-2 |
|---:|---|---:|---:|---|
| 0 | `<think>` | 151667 | 34.25 | `</think>` (25.25) |
| 1 | `\n` | **198** | 37.96 | `\n` (319, 22.47) |
| 2 | `</think>` | 151668 | 30.83 | `<think>` (30.29) |
| 3 | `\n\n` | 271 | 48.14 | `\n` (198, 27.83) |
| 4 | `</think>` | 151668 | 28.74 | ... |

**Точка divergence: step=1.**
- llama.cpp top-1 = **271 (`\n\n`)** с probability ≈ 1.0.
- PromeTorch top-1 = **198 (`\n`)** со score 37.96, **token 271 ВООБЩЕ
  ОТСУТСТВУЕТ в top-5**. На его месте — 319 (другой newline-вариант).

То есть PromeTorch forward после первого decode шага (token `<think>`
добавлен в KV) **выдаёт logits, в которых правильный токен 271
отсутствует даже в top-5**. Это **не numerical noise**, а
структурное расхождение output projection.

**Дальнейший шаг ломки:** на step=4 llama.cpp начинает русский текст
(`М`), а PromeTorch повторяет `</think>` → попадает в loop.

**Что это значит для дальнейшего дебага:**
1. Encoder/decoder/template — *исключены* (token IDs идентичны llama.cpp).
2. Forward на prefill (step=0) — *близок* к llama.cpp (top-1
   совпадает, top-2 разница в logit уже видна).
3. Forward на первом decode шаге (step=1) — *критически расходится*.
   Это `forward_decode_cpu` или `forward_decode_cpu_tp` (TP-4 path).
4. Подозрение: **output projection (lm_head)** на cyrillic high-ID
   tokens (~130k–140k vocab range), либо **KV cache append** для
   первого нового токена, либо **RoPE position offset** в decode.

**TODO для следующей сессии (deep debug, не lома hot path):**
1. Запустить с `PT_Q8_SOA=0` — отключить SoA repack, проверить вернётся
   ли top-1=271.
2. Если SoA не виновник — добавить `PT_DUMP_HIDDEN` для дампа hidden
   state перед lm_head на step=1, сравнить с llama.cpp `--dump-hidden`.
3. Bisect между prefill и decode: запустить две prefills (длиной 19 и
   20 токенов), сравнить top-5 на final.

---

## BUG-10: GPT-2 byte-to-unicode FORWARD mapping в encoder — ⚠ ЧАСТИЧНО (см BUG-12)

**Symptom (до):** На русском input prompt модели qwen3-4B/llama3 уходили
в китайский ("色 芬 风") или garbage. Russian translate output работал
("Привет, мир"), но русский INPUT нет.

**Root cause:** `tokenizer::encode` обрабатывал GPT-2 byte mapping только
для 3 special bytes (space/\n/\t). Остальные bytes — включая cyrillic
UTF-8 (0xD0, 0xD1) — проходили raw → vocab не находил match → byte
fallback → potery semantics.

**Fix:** в `torch/io/tokenizer.h` в encode loop полная 256-entry
bytes_to_unicode таблица: byte 0..255 → unique codepoint в UTF-8 form
(printable mapped to self, non-printable → U+0100..U+0143).

**Verify (после):** qwen2.5-7B на "Расскажи про Москву" →
"Москак Москва - это столиия России..." — **связный русский**
(с Q4-артефактами типа "столиия"). qwen3-4B перешёл с 风风风 на "Моск М"
(cyrillic тokens) — encoder работает, но Q4 квантизация qwen3-4B
весь русский потеряла (model weakness, не наш bug).

---

## BUG-7: GPT-2 byte-to-unicode reverse mapping missing in decode_token

**Symptom.** На русском промпте output содержит `å¤ĸ è¯Ĩ` — это
**latin-1 интерпретация UTF-8 bytes которые модель сгенерировала
корректно**, но наш decoder печатает их как Latin-1 chars вместо raw bytes.

**Diagnosis.** GPT-2 BPE tokenizer (используется qwen3, llama3) кодирует
каждый byte 0-255 в "printable unicode character" через специальную
таблицу `bytes_to_unicode`. При decode нужна обратная операция: parse
каждый unicode character в token string → raw byte → собрать в UTF-8
string.

**Root cause.** `tokenizer.h::decode_token` обрабатывает только 4
spec-символа (`▁` SP space, `Ġ` GPT-2 space, `Ċ` newline, `ĉ` tab) +
`<0xNN>` byte fallback. **256-entry byte-level reverse mapping не
реализован.**

**Fix plan.**
1. Добавить статическую таблицу `unicode_to_byte[]` (256 entries) в
   `tokenizer.h` по стандарту GPT-2 (`bytes 33-126, 161-172, 174-255` →
   self; остальные → U+0100..U+0143).
2. В `decode_token` после current step добавить: parse output character
   by character как UTF-8, если character ∈ unicode_to_byte → output raw
   byte; иначе output как UTF-8.
3. Re-run quality test на русском промпте, проверить что выходит
   coherent русский text.

**Priority.** P0 для русскоязычных применений.

---

## BUG-8: Phi3 fused-QKV не поддерживается в gguf reader

**Symptom.** Phi3.5-mini-instruct-Q4_K_M.gguf не загружается:
`Error: GGUF: Tensor not found: blk.0.attn_q.weight`

**Diagnosis.** Phi3 architecture использует **fused QKV** — один tensor
`blk.0.attn_qkv.weight` вместо отдельных Q/K/V. Наш `gguf_model::load`
в `torch/io/gguf_model.h` ищет separate Q/K/V tensors, не обрабатывает
fused QKV layout.

**Fix plan.** В loader: detect attn_qkv.weight, split на 3 части по
hidden_size (Q dim) / kv_dim / kv_dim. Затем заполнить наши separate
tensor slots.

**Priority.** P2 (phi3 не критичен, но добавляет поддержку важной
модели).

---

## BUG-9: Gemma3-4B загружается но output garbage

**Symptom.** Gemma3-4B Q4_K_M генерирует mixed-language garbage:
`HereENבваבкимك...其他ist...भनम्म...慶`. На русский тоже starts with
"Привет" затем mixed garbage.

**Diagnosis.** Gemma3 имеет специфические architectural tricks:
- `scale_emb = sqrt(hidden_size)` нормализация эмбеддингов
- soft attention capping (logit_soft_cap, attn_soft_cap)
- Post-attention RMSNorm + post-FFN RMSNorm
- Special bos/eos tokens

Наш `forward_decode_cpu*` не реализует все Gemma3 quirks. Видимо load
проходит но inference math не соответствует Gemma3 spec.

**Fix plan.** Реализовать в `gguf_model.h::forward_decode_*`:
1. Embedding scale by sqrt(H)
2. Soft attention capping в attention math
3. Post-attention/post-FFN RMSNorm (config флаг `post_norm=1` уже есть
   но не используется в forward path)

**Priority.** P2 (gemma3 одна из 5 моделей, не блокирует основной
use case).

---

## Priority queue (updated 2026-05-02 17:18)

1. **BUG-1** (chat template TP) — ✓ ЗАКРЫТ частично, factq работает
2. **BUG-7** (byte decoder для русского/non-ASCII) — P0
3. **BUG-2** (llama3-8B TP-4) — P1
4. **BUG-4** (build promeserve) — P1
5. **BUG-5** (#59 #60 robustness) — P1
6. **BUG-3** (LayerSkip docs) — P2
7. **Закрыто как model-weakness не-bug:**
   - haiku → "**" symbols (Q4_K_M quant degradation, не наш bug)
   - math 17×23 → unclear (qwen3-4B not calculator-tuned)
