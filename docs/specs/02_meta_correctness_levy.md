# 02 — Матрица архитектурной корректности PromeServe (per-model)

Автор: Aaron Levy (llama.cpp/ggml core, Meta GenAI), ревью корректности инференса PromeTorch.
Файл движка: `torch/io/gguf_model.h` (8040 строк), GPU-attention: `aten/src/ATen/cuda/FlashDecoding.cu`,
токенайзер: `torch/io/tokenizer.h`.

Цель документа — зафиксировать, какие per-model архитектурные фичи обязан учитывать движок, чтобы
давать **bit-identical** результат с референсом (llama.cpp / HF), где они реализованы в коде, где
пропущены, и эталонное поведение. Особый разбор — почему **gemma3 сыпет зацикленный мусор**.

---

## 0. Карта code-path'ов (критично — их ЧЕТЫРЕ, и они расходятся)

| Path | Функция | Когда | gelu? | SWA-mask? | softcap? | query_pre_attn? |
|------|---------|-------|-------|-----------|----------|-----------------|
| A. CPU prefill | `forward()` → `transformer_layer()` → `self_attention()` + `swiglu_ffn()` | CPU, весь промпт (`generate` стр. 4826) | ❌ хардкод SiLU | ❌ только causal | ❌ | ❌ `1/√head_dim` |
| B. CPU decode | `forward_decode_cpu()` (стр. 3226+) | CPU, каждый сгенерированный токен | ✅ `use_gelu` (3731) | ✅ `attn_start` (3501) | ❌ | ❌ |
| C. GPU prefill+decode | `forward_decode()` (стр. 2575+) | CUDA, всё | ❌ хардкод `launch_silu_mul` (2989) | ❌ | ❌ | ❌ |
| D. GPU attn-помощник | `self_attention()` → `launch_causal_attention` (7460) | CUDA prefill через `forward()` | ❌ SiLU (7590) | ❌ (window не передаётся) | ❌ | ❌ |

**Ключевой вывод:** реализация фич НЕ симметрична между prefill и decode и между CPU и GPU.
Только path B (CPU decode) частично корректен для gemma3. Промпт всегда кодируется неверно
(path A на CPU, path C/D на GPU) → испорченный KV-cache → петля с первого же токена.

---

## 1. Матрица фич × модели

Легенда: ✅ есть и верно · ⚠ есть, но частично/не во всех path · ❌ нет · — не требуется.

| Фича | qwen3 | gemma3 | gemma2 | llama3 | phi3.5 | mistral | deepseek2 | Где в коде | Статус |
|------|:----:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|-----------|--------|
| RoPE NeoX vs NORM | NeoX | NeoX | NeoX | NORM | NeoX | NORM | NeoX | `rope_neox` 207, `apply_rope_inplace` 7556 | ✅ |
| RoPE linear scale | — | 8.0(global) | — | 1.1x* | — | — | yarn | `layer_rope_scale` 301 | ✅ |
| RoPE per-layer (local≠global freq) | — | **✅ нужно** | — | — | — | — | — | `layer_rope_freq_base` 298 | ✅ (B/D), но A/C тоже зовут — ок |
| LongRoPE/YaRN factors | — | — | — | ✅ | ✅ | — | yarn | `rope_factors_for`, `rope_attn_factor` | ✅ (B), ⚠ A/C не применяют factors |
| GQA (head_count_kv) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | MLA | `heads_per_group` | ✅ |
| QK-norm (per-head RMS) | ✅ | ✅ | — | — | — | — | — | `has_qk_norm`, `apply_qk_norm_inplace` 7432 | ✅ |
| FFN активация GELU vs SiLU | SiLU | **GELU** | **GELU** | SiLU | SiLU | SiLU | SiLU | `ffn_gelu` 88; path B 3731 | ⚠ **только B; A/C/D хардкод SiLU** |
| scale_embeddings ×√d | — | ✅ | ✅ | — | — | — | — | `scale_embeddings` 2520/2684/3257 | ✅ во всех path |
| post-attn / post-ffn norm | — | ✅ | ✅ | — | — | — | — | `has_post_norm`; 7373/7391 + decode | ✅ |
| Sliding-Window Attention (local layers) | — | **✅ 1024** | **✅ 4096** | — | — | — | — | `swa_window`,`layer_is_global`; path B 3501 | ⚠ **только B; A/C/D нет маски** |
| attn-logit softcap | — | — | **✅ 50** | — | — | — | — | **нигде** | ❌ **отсутствует поле и код** |
| final-logit softcap | — | — | **✅ 30** | — | — | — | — | **нигде** | ❌ **отсутствует** |
| query_pre_attn_scalar (≠head_dim) | — | =head_dim(ок) | **✅ 144** | — | — | — | — | **нигде**, scale=`1/√head_dim` 7454/3491 | ❌ ломает gemma2-27B |
| tied embeddings (output=tok_emb) | — | ✅ | ✅ | — | ⚠ | — | — | `tie_word_embeddings` 84/186 | ✅ |
| MLA (latent KV) | — | — | — | — | — | — | ✅ | `is_mla`, deepseek2 path 7825 | ⚠ отдельная сессия, garbage |
| MoE (top-k experts) | — | — | — | — | — | — | ✅ | `is_moe`, `moe_ffn_forward_decode` | ⚠ deepseek2 only |
| Chat template | ChatML | gemma-turn | gemma-turn | llama3 hdr | phi3 | [INST] | — | `apply_chat_template` 4181 | ✅ (gemma — без system turn, верно) |
| QKV bias | ✅(qwen) | — | — | — | — | — | — | `attn_q_bias` 7423 | ✅ |

\* llama-3.1 long-context linear factor; базовый llama3-8B = 1.0.

---

## 2. ПОЧЕМУ ЛОМАЕТСЯ GEMMA3 — root-cause разбор

Симптом «and the same is true for the same…» — классическая дегенерация при испорченном
контексте: модель не «видит» промпт правильно, скатывается в самый вероятный n-грамм-цикл.
Это вызвано НЕ одной, а стопкой расхождений с референсом. По убыванию тяжести:

### 2.1. (CRITICAL) FFN-активация: SiLU вместо GELU на prefill — все слои промпта
Gemma-семейство использует **GeGLU** (`down(GELU_tanh(gate) * up)`), не SwiGLU.
- Поле `config.ffn_gelu=true` для gemma выставляется (стр. 192) — **но читается только в path B**
  (`forward_decode_cpu`, 3731). `swiglu_ffn()` (path A, стр. 7626/7633) и GPU `forward_decode()`
  (path C, `launch_silu_mul` 2989) **хардкодят SiLU** и игнорируют флаг.
- Промпт gemma3 на CPU проходит через `forward()` → `swiglu_ffn()` → **каждый FFN всех 62 слоёв
  (27B) считается с неправильной нелинейностью**. SiLU(x)=x·σ(x) ≠ GELU(x)=x·Φ(x): на отрицательных
  и больших значениях расхождение десятки процентов → активации «плывут» → итоговый hidden-state и
  KV-cache промпта мусорные ещё до первого decode-шага.
- На GPU то же самое на КАЖДОМ токене (prefill и decode). Это самостоятельно объясняет петлю.

### 2.2. (CRITICAL) Sliding-Window Attention не маскируется на prefill
Gemma3: pattern 5:1 — 5 local-слоёв (window=1024) на 1 global. Эталон: local-слой видит только
последние 1024 позиции; всё дальше — `-inf` ДО softmax.
- Маска есть **только в path B** (`attn_start`, стр. 3501). В `self_attention()` (path A/D) и GPU
  `forward_decode()` (path C) окно НЕ применяется — `launch_causal_attention` (7460) получает
  `nullptr` вместо window и считает полную причинную attention.
- Для коротких промптов (<1024) эффекта нет, но для длинных (>1024 ток.) local-слои на prefill
  «подмешивают» далёкий контекст, которого в обучении не было → сдвиг распределения. У 27B с
  длинным system+user это усугубляет 2.1.

### 2.3. (HIGH) query_pre_attn_scalar не парсится → неверный масштаб attention
Эталон gemma2/3: `attn_scale = 1/√(query_pre_attn_scalar)`, НЕ `1/√head_dim`.
- В коде scale жёстко `1.0f/√head_dim` (стр. 7454 path A, 3491 path B).
- Для **gemma3** query_pre_attn_scalar == head_dim (256) → совпадает случайно, ок.
- Для **gemma2-27B** query_pre_attn_scalar=144, head_dim=128 → scale завышен в √(128/144)=0.94 →
  логиты attention раздуты → пересглаженный/переострый softmax. Если в зоопарке есть gemma2-27B —
  он тоже поедет. Поле в `TransformerConfig` отсутствует целиком.

### 2.4. (MEDIUM, только gemma2) logit soft-capping отсутствует
Gemma2 (НЕ gemma3) клампит:
- attn-logits: `softcap·tanh(score/softcap)`, softcap=50 — внутри attention до softmax;
- final-logits: `30·tanh(logits/30)` — перед сэмплингом.
Gemma3 от обоих **отказалась** (заменено на QK-norm), поэтому для gemma3 это НЕ root-cause.
Но: (а) в движке нет ни полей `attn_logit_softcap`/`final_logit_softcap`, ни кода → **gemma2 любой
размерности будет неточен**; (б) важно не «случайно» включить softcap для gemma3.

### 2.5. (LOW) BOS-токен и chat-template
gemma-шаблон (стр. 4218) верный: `<start_of_turn>user…<end_of_turn>\n<start_of_turn>model\n`,
system вшит в user (gemma не имеет system-turn — корректно). Проверить, что токенайзер добавляет
ведущий `<bos>` ровно один раз (HF gemma добавляет BOS; двойной BOS тоже даёт сдвиг). См. `tokenizer.h`.

### Вывод по gemma3
Достаточно **2.1 (SiLU→GELU на prefill) + 2.2 (нет SWA-маски)**, чтобы получить именно
наблюдаемую петлю: промпт кодируется неправильной сетью, KV-cache мусорный, decode (даже корректный
path B) стартует из испорченного состояния и валится в n-грамм-цикл. query_pre_attn для gemma3
совпадает случайно и петлю не вызывает, softcap для gemma3 не нужен.

---

## 3. Что реально нужно (приоритезированный fix-list, без правок здесь)

1. **Унифицировать FFN-активацию через `config.ffn_gelu`** в `swiglu_ffn()` (path A) и в GPU
   `forward_decode()`/path C (нужен `launch_gelu_mul` или ветка). Сейчас GeGLU только в path B.
2. **Прокинуть SWA-окно во все attention-path'ы**: в `self_attention()` (CPU и `launch_causal_attention`)
   считать `attn_start`/window от `layer_is_global(layer_idx)`; в FlashDecoding.cu добавить `window` param.
3. **Сделать prefill и decode одним кодом** (или хотя бы共享 FFN/attention примитивы), иначе любая
   новая фича снова разойдётся между путями. Корень всех 4 багов — дубликаты.
4. Добавить в `TransformerConfig`: `query_pre_attn_scalar`, `attn_logit_softcap`, `final_logit_softcap`
   (парсить `<arch>.attention.*`); применить scale=`1/√query_pre_attn` и softcap в attention/перед сэмплом
   **только когда поле>0** (gemma2 — да, gemma3 — нули → no-op).
5. GPU prefill для gemma3 сейчас идёт через path A/D с теми же дырами — закрыть вместе с (1)(2).

---

## 4. ЧЕК-ЛИСТ КОРРЕКТНОСТИ (gate перед «модель X готова»)

Для каждой модели прогнать промпт >1100 токенов и сверить top-1 logit/первые 30 токенов с
llama.cpp (greedy, temp=0). Галочка = bit-близко (logit diff < ~0.5).

- [ ] RoPE-тип (NeoX/NORM) — верный (qwen/gemma/phi=NeoX, llama/mistral=NORM)
- [ ] RoPE per-layer freq для gemma3 (local 10000 / global 1e6, scale 8.0 global)
- [ ] LongRoPE/YaRN factors применяются на **prefill тоже** (phi3.5, deepseek2), не только decode
- [ ] FFN: GELU для gemma*, SiLU для остальных — **в prefill И decode, CPU И GPU**
- [ ] scale_embeddings ×√hidden — gemma (все path) ✅
- [ ] post-attn/post-ffn norm — gemma3 (prefill тоже)
- [ ] QK-norm — qwen3, gemma3 (prefill тоже)
- [ ] SWA-маска (window=1024 gemma3 / 4096 gemma2) на local-слоях — **в prefill И decode, CPU И GPU**
- [ ] attn-scale = 1/√query_pre_attn_scalar (gemma2-27B=144), иначе 1/√head_dim
- [ ] attn-logit softcap 50 + final-logit softcap 30 — **только gemma2**, у gemma3 выключено
- [ ] tied embeddings — gemma/phi
- [ ] chat-template + одиночный BOS, корректные stop/EOS токены (`<end_of_turn>` для gemma)
- [ ] QKV bias — qwen
- [ ] длинный (>SWA window) прогон НЕ деградирует после N токенов (регресс-тест на петлю)
