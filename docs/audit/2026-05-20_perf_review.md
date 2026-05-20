# Аудит #9 — Performance hot paths review

**Дата:** 2026-05-20
**Целевая платформа:** Эльбрус 8C2 (4 NUMA × 8 cores), TP-4 qwen3-4b
**Текущий baseline:** 10.9 tok/s
**Цель Round 4:** 30 tok/s lossless

Только статический анализ. Никаких изменений кода.

---

## Сводная таблица предложений

| #  | Optimization | Location (file:line) | Expected | Risk | Complexity |
|----|--------------|----------------------|---------:|------|------------|
| 1  | Параллелизация CPU attention по головам (`parallel_for` over `hl`) | `torch/io/gguf_model.h:5969` (TP path) и `:3444` (non-TP) | +12-18% | Низкий (head-independent, scores_buf — per-thread alloc) | M |
| 2  | FP16 KV cache на CPU (read-path bandwidth -50%) | `torch/io/gguf_model.h:5985,6049` (loads `k_cache + t*kv_dim_local + …`) | +10-15% (attention bound на длинном контексте) | Средний (нужен dequant в hot loop или FP16 dot intrinsic) | H |
| 3  | Persistent `tp_.scores_buf` per worker thread (сейчас shared → false-sharing если параллелить #1) | `torch/io/gguf_model.h:5977-5980` | +2-3% (только после #1) | Низкий | S |
| 4  | NgramDraft: hash-map ngram→idx вместо O(N) reverse-scan + поддержка multi-token PLD continue | `torch/io/speculative_draft.h:63-97, 157-194` | +5-10% (если accept rate подскочит >40%) | Низкий | M |
| 5  | Fuse `q8_soa4_silu_quant_activation_fused` 2-passes → single (slu+quant в одном проходе по K) | `torch/io/q8_soa_repack.h:340-388` | +3-5% (FFN активация) | Низкий (lossless по формуле, но нужна доп. абсолютная редукция онлайн) | M |
| 6  | TLB-friendly KV layout: per-head 2D `[head][t][d]` вместо interleaved `[t][kv_dim]` | `torch/io/gguf_model.h:827-828, 5954-5955, 5985, 6049` | +4-7% (особенно при past_len>2048) | Высокий (затрагивает все RoPE/attn callsites) | H |
| 7  | Q8_SoA4 dual: убрать redundant `qpsubw`/`qpistofs` хранение fp_acc через 2 матрицы (можно интерливить компоненты scalar broadcast через 1 общий load `sa_v`) | `torch/io/q8_soa_repack.h:587-628` | +1-2% | Низкий | S |
| 8  | LayerSkip → Adaptive (confidence-based fallback на полный layer) | `torch/io/gguf_model.h:5786-5797` (сейчас static skip-list) | +8-15% **lossless** (запустить >5 layers под порогом и full при энтропии>τ) | Средний (требует metric — logit norm/entropy) | H |
| 9  | Q4_K_M direct path: pre-scan и держать готовую smasked-bitmap pure Q4_K vs Q6_K dispatch в `sparse_gemv` style без runtime ttype-check | `torch/io/gguf_model.h:7863+` (per-layer ttype dispatch) и `cpu_quant_gemv.h:1635` | +2-4% | Низкий | M |
| 10 | RoPE cos/sin cache — broadcast в bf16/fp16 (8KB vs 16KB) | `torch/io/gguf_model.h:5917-5929` | +1% (cache footprint) | Низкий | S |
| 11 | K-slice GEMV: stack Q8Block `[512]` × 40B = 20KB — спил L1 при K_local>16384; pre-allocate в TP scratch | `cpu_quant_gemv.h:1269-1276` (вариант), `:517-524` | +2-3% (для длинных FFN K) | Низкий | S |
| 12 | Mas­ter `wait_workers_idle` spin → adaptive: при N=4 chunks почти всегда мастер заканчивает первым; current 256-spin × N workers = ~1µs waste per parallel_for × 36 layers × 4 GEMVs = ~150µs/token | `c10/util/ThreadPool.h:171-181` | +1-2% | Средний (нужен бенчмарк grain) | M |
| 13 | Q8 dual `gate+up`: hoist `sa_v` broadcast вне dual-block fork (сейчас вычисляется в каждом из 2 блоков отдельно из одного и того же `sum_a_per_block[b]`) | `torch/io/q8_soa_repack.h:594-628` (dual) и `:693-698` (triple) | +0.5% | Нулевой | S |
| 14 | Q4_K scalar fallback: `dpair_buf[80*2]` — стек, но `gguf::get_scale_min_k4` пересчитывается дважды per pair (по `is` и `is+1`) — extract один раз для всех 8 sub-blocks | `cpu_quant_gemv.h:748-762` (scalar) и `:1312-1319` (k-slice) | +2-3% (scalar fallback) | Низкий | S |
| 15 | `cpu_fused_rmsnorm_gate_up_gemv`: alloca-stack `x_norm[8192]` × 3 fused функции — большой stack footprint; используется одна и та же арифметика — выделить общий helper | `cpu_quant_gemv.h:2298-2299, 2364-2365, 2433-2434` | 0% (refactor) | Низкий | S |
| 16 | Speculative batched verify: `kMaxSpecBatch=6` — но `predict_pld` отдаёт до K=N+: убедиться что N=K (sync constants) | `cpu_quant_gemv.h:1888`, `speculative_draft.h:182` | +2-5% (accept rate) | Низкий | S |
| 17 | Per-block sum_a hot-path: умножение на scalar `dmin*m*sa` можно SIMD-broadcast 1 раз для всех 4 rows в SoA4 (сейчас `qpfmuls(dmins_v, sa_v)` правильно делается, но `sa_v` пересоздаётся каждый block из scalar via memcpy → должен быть FP register-resident) | `q8_soa_repack.h:470-477` | +0.5% | Нулевой | S |

---

## Ключевые находки по разделам

### A. Attention hot path (gguf_model.h ~5965-6090)

`for (hl = 0; hl < n_heads_l; ++hl)` НЕ обёрнут в `parallel_for`. На rank-local 8 heads (qwen3-4b: 32 heads / 4 ranks = 8). Каждый head — 2× K-loop (Q@K, V@scores) по `total_seq=past_len+1`. При past_len=1024 — это ~256 KB чтений KV cache последовательно одним потоком. 8 cores rank-local простаивают (только master тред делает attention). **Это #1 одиночное ускорение.**

Однако: внутри уже идёт RoPE через `at::native::hot::rope_apply_fused_neox` — он сам parallel_for'ит. Так что между этим и GEMV-driven parallel_for нет conflict, но head-loop сериализован.

Подтверждение: `n_heads * 2 * head_dim * total_seq * 4B = 8 * 2 * 128 * 1024 * 4 = 8.4 MB` чтения KV cache на 1 master core ≈ 0.7ms/layer × 36 = 25ms/token = ~3.7 tok/s упускается.

### B. ThreadPool dispatch (Item 77)

`c10/util/ThreadPool.h:140` — есть persistent broadcast pool (gen + futex). Реализован, не недостаток. Но: при `min_grain=1` (как в `q8_soa4_gemv`) — каждый parallel_for делает full broadcast wake. Для N=4 ranks × 36 layers × ~6 parallel_for (RMS+QKV+attn+O+FFN) = 864 dispatches/token. Master spin=256 × N workers = ~1-2 µs/dispatch = ~1ms wasted/token.

### C. KV cache layout

`[t][kv_dim_local]` row-major. При attention `V@scores` чтения по dimension `[t]` с stride `kv_dim_local*4`. Для qwen3-4b: kv_dim_local = 2 heads × 128 = 256 floats = **1024 bytes stride**. На длинном контексте (1024 token) — 1024 TLB misses per V-head. Per-head 3D layout `[kv_head][t][d]` ликвидирует это (stride = 128 floats внутри head, contig).

### D. Allocations

`tp_.scores_buf` (item 8 Round 5) — закрывает 288 alloc/token. `rope_cos_cache` cached. `silu_scratch_buf`, `x_normed_buf` — pre-allocated. Аллокаций per-token осталось мало — основной hot path аллокации убраны.

### E. Speculative

`NgramDraft.predict_pld` — O(H × n × buffer_size) reverse scan, где n=1..3. На history 2048 = ~6000 cmp ops/draft. Замена на hash-map ngram→last_pos: O(1) lookup, O(1) update.

### F. Q8 SoA4 / Q4_K kernels

Inner K-loop = 8 итераций `kg=0..7` — уже compact с `_Pragma("loop count(8)") _Pragma("ivdep")`. LCC SWP может развернуть × 2. Prefetch lead = 4 blocks (576B) — корректно для DDR latency. Микробенчмарк дал 0.85× EML cblas_sgemv → headroom ~15%.

### G. Memory bandwidth

Q4_K weights: 2.5 GB/token (заявлено). На 4-NUMA каждый rank читает 625 MB локально. 8C2 BW per node ≈ 12 GB/s → 52ms theoretical. Текущее 92ms/token (10.9 tok/s) → utilization 57%, не 23%. 23% (Item 76) — возможно про aggregate cross-node BW. Скорее всего sustainable target — bandwidth-bound, 30 tok/s требует Q3/Q2 либо FP16 KV cache (#2) + persistent #1+#8.

### H. LayerSkip lossless?

Currently — static skip-list (`PT_LAYER_SKIP="20,21,22"` — lossy hard skip). **Возможна lossless вариация:** считать `||h_buf_after - h_buf_before||` на калибровочном run, выбрать N слоев с минимальным contribution, при inference применять только если `||residual_predicted|| < τ` (online check). Это требует дополнительного per-layer norm op, но ~10 слоев пропусков × 2.5ms = 25 ms/token savings vs ~5 µs τ-check.

---

## Suммари (≤300 слов)

В hot path обнаружены **3 крупных недо­использования**: (1) CPU attention `for hl=0..n_heads_l` в TP-режиме идёт сериально на master-треде, 7 worker-ов простаивают — ~25 ms/token потерь; (2) KV cache живёт в FP32 на CPU (FP16 только для CUDA пути) — двукратная избыточная bandwidth на attention reads; (3) interleaved layout `[t][kv_dim]` даёт 1024-byte stride V-чтений → TLB-thrash на длинном контексте.

**Средние выигрыши:** (4) `NgramDraft.predict/predict_pld` O(buffer_size) reverse-scan заменить hash-map; (5) `q8_soa4_silu_quant_activation_fused` имеет 2 прохода по K — можно слить в один; (8) LayerSkip — сделать confidence-based fallback вместо static (lossless при правильном τ).

**Микро-уровень:** (7,13,17) дубликаты broadcast `sa_v` в Q8 SoA4 dual/triple; (14) `gguf::get_scale_min_k4` пересчёт в scalar fallback; (10) RoPE cache можно в FP16.

**ThreadPool** уже persistent broadcast (Item 77 закрыт). `tp_.scores_buf` resize-only persistent (Item 8 закрыт). Prefetch chains корректны. Q4_K AVX2 kernel uses 4 split-accumulators (E8C2 ALC). SoA4 inner K-loop полностью развёрнут (8 iter ivdep).

**Bandwidth budget:** weights 2.5 GB/token + KV 8.4 MB/token ÷ 4-NUMA × 12 GB/s = 52 ms theoretical, текущие 92 ms = 57% utilization. Цель 30 tok/s = 33 ms/token требует **либо ужать веса** (Q3_K), **либо FP16 KV** + closing attention-parallelism gap. Все 17 предложений совокупно дают +40-60% оценочно — close to цели, при условии что bandwidth-нижняя оценка корректна.

**Ничего не имплементировано** — только аудит.
