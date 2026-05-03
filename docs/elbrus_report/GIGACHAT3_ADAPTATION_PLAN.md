# GigaChat 3.1 10B-A1.8B — план адаптации к PromeTorch

**Файл:** `gguf_models/GigaChat3.1-10B-A1.8B-q4_K_M.gguf` (6.1 GB, скачан 2026-05-03 22:23)
**Архитектура:** `deepseek2` (Multi-head Latent Attention + Mixture-of-Experts + YaRN)
**Источник:** [ai-sage/GigaChat3.1-10B-A1.8B-GGUF](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF)

## Текущее состояние

PromeTorch распознаёт `general.architecture=deepseek2` но падает на загрузке весов:
```
Error: GGUF: Tensor not found: blk.0.attn_k.weight
```
Причина: deepseek2 MLA не имеет отдельных K/V тензоров.

## Архитектурные параметры (из GGUF metadata)

| Параметр | Значение |
|---|---|
| `block_count` | 26 |
| `embedding_length` | 1536 |
| `vocab_size` | 128256 |
| `feed_forward_length` (dense) | 8960 |
| `expert_feed_forward_length` (MoE) | 1280 |
| `attention.head_count` | 32 |
| `attention.head_count_kv` | 1 (MQA-стиль) |
| `attention.kv_lora_rank` | 512 |
| `attention.key_length` | 576 (= 512 + 64 rope) |
| `attention.key_length_mla` | 192 (per-head no-rope) |
| `attention.value_length` | 512 |
| `attention.value_length_mla` | 192 |
| `expert_count` | 64 |
| `expert_used_count` | 4 (top-k routing) |
| `expert_shared_count` | 1 |
| `expert_weights_norm` | True |
| `leading_dense_block_count` | 1 (первый блок dense, остальные MoE) |
| `rope.scaling.type` | yarn |
| `rope.scaling.factor` | 64.0 |
| `rope.scaling.original_context_length` | 4096 |
| `rope.scaling.yarn_beta_fast` | 32.0 |
| `rope.scaling.yarn_beta_slow` | 1.0 |
| `rope.scaling.yarn_log_multiplier` | 0.1 |
| `rope.dimension_count` | 64 (только rope-часть!) |
| `rope.freq_base` | 100000 |
| `tokenizer.ggml.pre` | `gigachat` |

## Tensor layout (per layer)

### Dense layer (только blk.0):
- `attn_norm.weight` [1536]
- `attn_q.weight` [1536, 6144]  — Q proj, выход = 32×192
- `attn_kv_a_mqa.weight` [1536, 576] — KV down-proj (split: 512 latent + 64 rope)
- `attn_kv_a_norm.weight` [512] — RMSNorm для latent KV
- `attn_k_b.weight` [128, 512, 32] (3D!) — K up-proj per-head
- `attn_v_b.weight` [512, 192, 32] (3D!) — V up-proj per-head
- `attn_output.weight` [6144, 1536]
- `ffn_norm.weight` [1536]
- `ffn_gate.weight` [1536, 8960]
- `ffn_up.weight` [1536, 8960]
- `ffn_down.weight` [8960, 1536]

### MoE layers (blk.1..blk.25):
- attention идентична dense
- `ffn_norm.weight` [1536]
- `ffn_gate_inp.weight` [1536, 64] — router (64 experts)
- `ffn_gate_exps.weight` [1536, 1280, 64] (3D!) — gate per expert
- `ffn_up_exps.weight` [1536, 1280, 64] (3D!) — up per expert
- `ffn_down_exps.weight` [1280, 1536, 64] (3D!) — down per expert
- `ffn_gate_shexp.weight` [1536, 1280] — shared expert gate (всегда применяется)
- `ffn_up_shexp.weight` [1536, 1280]
- `ffn_down_shexp.weight` [1280, 1536]

Итого 414 тензоров, 6.1 GB Q4_K_M.

## План работ (для следующей сессии)

### Phase 1: ModelConfig + loader (estimate ~300 LoC)
1. В `ModelConfig::parse()` добавить чтение всех `deepseek2.*` ключей.
2. Завести поля: `is_mla`, `kv_lora_rank`, `key_length_mla`, `value_length_mla`, `is_moe`, `expert_count`, `expert_used_count`, `expert_shared_count`, `leading_dense_block_count`, `yarn_*` параметры.
3. В `load_quantized_mmap` для arch=deepseek2 ожидать MLA tensor set (per-layer condition: `i < leading_dense_block_count` → dense ffn, иначе → MoE ffn).
4. 3D тензоры (`attn_k_b/v_b`, `ffn_*_exps`) — добавить `QuantizedWeight3D` или хранить как 2D с extra stride.

### Phase 2: MLA forward (~400 LoC)
1. Q проекция: `q = x @ attn_q.T` → [seq, n_heads * head_dim_mla]
2. KV down: `kv_compressed = x @ attn_kv_a_mqa.T` → [seq, kv_lora_rank + rope_dim] = [seq, 576]
3. Split: `kv_latent = kv_compressed[:, :512]`, `k_rope = kv_compressed[:, 512:576]`
4. RMSNorm на `kv_latent` через `attn_kv_a_norm`
5. K up per-head: для каждого head h: `k_no_rope[h] = kv_latent @ attn_k_b[:, :, h]` → [seq, 128]
6. V up per-head: `v[h] = kv_latent @ attn_v_b[:, :, h]` → [seq, 192]
7. RoPE на k_rope (одна копия общая для всех heads, MQA стиль) и на последние 64 dim of Q
8. Concat: `k_full[h] = [k_no_rope[h]; k_rope]` → [seq, 192]
9. Стандартный attention: softmax(Q · K_full / sqrt(d)) · V → [seq, n_heads * 192]
10. Output proj: `out = attn_out @ attn_output.T` → [seq, hidden]

### Phase 3: MoE forward (~600 LoC)
1. RMSNorm(x) → x_norm
2. Router: `scores = x_norm @ ffn_gate_inp.T` → [seq, n_experts=64]
3. Apply gating function (sigmoid + group-limited top-k):
   - Top-k=4 selection
   - Normalize selected weights (если `expert_weights_norm=True`)
4. Per-token: для каждого выбранного эксперта e:
   - `gate_e = ffn_gate_exps[:, :, e] @ x_norm` → [1, 1280]
   - `up_e = ffn_up_exps[:, :, e] @ x_norm` → [1, 1280]
   - `expert_out_e = SiLU(gate_e) * up_e @ ffn_down_exps[:, :, e]` → [1, 1536]
   - aggregate: `result += router_score_e * expert_out_e`
5. Shared expert (всегда применяется):
   - `shared_out = SiLU(x_norm @ ffn_gate_shexp.T) * (x_norm @ ffn_up_shexp.T) @ ffn_down_shexp.T`
6. Final: `result + shared_out + x` (residual)

### Phase 4: YaRN RoPE (~80 LoC)
1. Реализовать `rope_apply_yarn` с factor / beta_fast / beta_slow / log_multiplier
2. Mscale corrected attention scale: `attn_factor = 1 + log_multiplier * log(factor)`

### Phase 5: Tokenizer (~100 LoC)
1. Pre-tokenizer "gigachat" — изучить какие regex'ы / split rules в llama.cpp `llama-vocab.cpp:tokenizer_pre = "gigachat"`. Скорее всего совпадает с llama3 / qwen2 общим стилем.

### Phase 6: Q4_K kernel для 3D weights (~200 LoC)
1. `cpu_quant_gemv_3d_indexed` — выбор слайса по index (для experts).
2. Для MLA `attn_k_b` per-head — можно реплицировать в плоский [n_heads*128, 512] layout при загрузке (×32 памяти, OK).

**Общий объём: ~1700 LoC + тесты.** Тестирование займёт 2-3 дня само по себе (модель compile-time правильная не значит inference-correct — bisect bugs как с phi3 mmap split).

## Альтернативный путь: llama.cpp-e2k

Если время критично — можно попробовать собрать апстрим llama.cpp через LCC на e2k и использовать его deepseek2 поддержку. Поддержка GigaChat3 уже merged в llama.cpp согласно
discussions на HuggingFace ([ссылка](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B/discussions/1)).
Но это путь без TP-4 и без интеграции в PromeTorch экосистему.

## Решение

Адаптация GigaChat3 в PromeTorch — **отдельная сессия минимум на 3-5 дней работы**.
Сейчас зафиксирована вся требуемая информация для следующей сессии.

Файл скачан, архитектура изучена, plan готов — можно стартовать Phase 1
немедленно когда time будет.
