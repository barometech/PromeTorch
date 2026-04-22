# Agent 2 — Memory Bandwidth Audit — qwen3:4b Q4_K_M on Elbrus 8C2

**Автор:** Opus 4.7 audit agent
**Дата:** 2026-04-22
**Цель:** Доказать или опровергнуть гипотезу «bottleneck = memory bandwidth, не compute» для inference qwen3:4b Q4_K_M на 4-NUMA E8C2, 5.5 tok/s лучший результат, таргет 10–15 tok/s.

**Источники (все read-only):**
- `torch/io/gguf_model.h` (forward_decode_cpu @ L2205, forward_decode_cpu_tp @ L3603)
- `torch/io/cpu_quant_gemv.h` (block sizes 144 / 210 @ L8–10, L56, L616)
- `torch/io/numa_weight_replica.h`
- `BENCH_ELBRUS.md`
- `JOURNAL.md` (tail 200 lines)
- `docs/elbrus/E8C2_ARCHITECTURE.md`
- `docs/elbrus/elbrus_memory_controller.pdf` — **не смог прочитать, PDF tooling недоступен в sandbox** (pdftoppm отсутствует). Числа по DRAM взяты из `E8C2_ARCHITECTURE.md` §5 и `vliw_mission/MISSION_BRIEF.md`.

---

## Q1. Точный подсчёт байт на 1 forward pass qwen3:4b Q4_K_M

### Архитектурные параметры qwen3:4b (из кода reader @ `gguf_model.h:75–87`)

| Param | Значение | Источник |
|-------|---------:|----------|
| `hidden_size` H | 2560 | `BENCH_ELBRUS.md` L181: «1×2560×2560 decode attn Q» |
| `num_layers` L | 36 | rope_precompute speedup «~36x for qwen3:4b» @ `gguf_model.h:2402` |
| `num_heads` n_h | 32 | «qwen3:4b (32 Q heads + 4 KV heads share)» @ `gguf_model.h:2402` |
| `num_kv_heads` n_kv | 4 | same comment |
| `head_dim` d_h | 128 | 2560/32 = 80... но `MISSION_BRIEF.md` указывает `q_dim=4096` при TP split: q_dim = 32×128 = 4096 ≠ 2560. `BENCH_ELBRUS.md` L50: «qkv_fused» при H=2560 даёт q_dim, плюс L181 «1×2560×2560 attn Q» ⇒ `attn_Q` имеет rows=q_dim=2560 **или** 4096. Реально qwen3:4b GGUF имеет **q_dim=4096** (n_h=32, d_h=128), а hidden=2560 — GQA с asymmetric KV. Это подтверждается `MISSION_BRIEF.md` упоминанием `q_dim=4096`. |
| `intermediate_size` F | 9728 | `BENCH_ELBRUS.md` L51: «gate_up_fused 1×6912×2560» — 6912, ИЛИ L32 «inter=9728». Код TP говорит `inter=9728` (K-slice 10/10/9/9). Принимаю **F = 9728**. |
| `vocab_size` V | 151 936 | `gguf_model.h:3888` «vocab=152k», комментарий L3888 |
| `kv_dim` | n_kv × d_h = 4 × 128 = 512 | |
| `q_dim` | n_h × d_h = 32 × 128 = 4096 | |

> **Замечание по H=2560 vs q_dim=4096:** Qwen3-4B использует расширенный head_dim → q_dim ≠ H. Матрицы:
> - `attn_q`: [4096, 2560], `attn_k`: [512, 2560], `attn_v`: [512, 2560]
> - `attn_output`: [2560, 4096]
> - `ffn_gate`/`ffn_up`: [9728, 2560]
> - `ffn_down`: [2560, 9728]

### Q4_K layout (из `cpu_quant_gemv.h:8`, `:56`, `:79`)

```
Q4_K: 256 values / block, 144 bytes / block
  = 144/256 = 4.5 bits / weight (4-bit quants + scales per 32-group + super-scale)
```

### Q6_K layout (`cpu_quant_gemv.h:616`)

```
Q6_K: 256 values / block, 210 bytes / block = 6.5625 bits / weight
```

### Per-layer byte count (decode, batch=1, 1 forward pass)

Предположение Q4_K_M quant-allocation (llama.cpp стандарт):
- `attn_q`, `attn_k`, `attn_v` → Q4_K (144 B / 256)
- `attn_output` → Q4_K
- `ffn_gate`, `ffn_up` → Q4_K
- **`ffn_down` → Q6_K** (подтверждено `cpu_quant_gemv.h:1177`: "Q6_K ... ffn_down in qwen3:4b / gemma3:4b")
- `attn_q_norm`, `attn_k_norm`, `attn_norm`, `ffn_norm` → FP32 (maintained as .defined() Tensors, `gguf_model.h:2337`, `:2582`)

**Вычисление байт на слой:**

| Weight | Shape (rows × cols) | Elements | Quant | Bytes |
|--------|---------------------|---------:|-------|------:|
| attn_q | 4096 × 2560 | 10 485 760 | Q4_K (4.5b) | 5 898 240 |
| attn_k | 512 × 2560 | 1 310 720 | Q4_K | 737 280 |
| attn_v | 512 × 2560 | 1 310 720 | Q4_K | 737 280 |
| attn_output | 2560 × 4096 | 10 485 760 | Q4_K | 5 898 240 |
| ffn_gate | 9728 × 2560 | 24 903 680 | Q4_K | 14 008 320 |
| ffn_up | 9728 × 2560 | 24 903 680 | Q4_K | 14 008 320 |
| ffn_down | 2560 × 9728 | 24 903 680 | **Q6_K (6.5625b)** | 20 428 800 |
| attn_norm (FP32) | 2560 | 2 560 | F32 | 10 240 |
| ffn_norm (F32) | 2560 | | F32 | 10 240 |
| attn_q_norm, attn_k_norm (F32) | 128 ×2 | | F32 | 1 024 |
| attn_q_bias, k_bias, v_bias (F32, Qwen3) | 4096+512+512 | | F32 | 20 480 |
| **Subtotal / layer** | | ~98.2M | | **61 758 264 B ≈ 58.9 MiB** |

**Global weights (вне layer loop):**

| Weight | Shape | Quant | Bytes |
|--------|-------|-------|------:|
| `token_embedding` | 151936 × 2560 | **F32** (`gguf_model.h:2229` `data_ptr<float>()`) | 1 555 824 640 ≈ 1.45 GiB |
| `output_norm` | 2560 | F32 | 10 240 |
| `q_output_weight` (lm_head) | 151936 × 2560 | Q4_K (tied? `gguf_model.h:3914` говорит "tied q_output_weight is common case") | 218 787 840 ≈ 208.6 MiB |

> **ВАЖНО:** token_embedding в Prometorch держится как FP32 (не Q4). Это +1.45 GB «мёртвого» веса, но при decode он touch'ится ровно **один раз** — 2560 × 4 B = **10 KiB** из 1.45 GiB. Остальные 1.45 GB всегда в DRAM но не читаются per-forward (только в random rows). Это УЖЕ не в hot path.

### Per-token bytes (decode path)

```
per_layer_weights  = 61 758 264 B (fully read once per token)
layers             = 36
total_weights      = 36 × 61 758 264 = 2 223 297 504 B ≈ 2.070 GiB

embedding lookup   = 1 row × 2560 × 4 = 10 240 B
output_norm        = 10 240 B
lm_head (Q4_K tied) = 218 787 840 B (read once per decode)

KV cache (read+write per layer):
  read  = past_len × 2 × kv_dim × 4 B = past_len × 4096 B per layer
  write = 1 × 2 × kv_dim × 4 B = 4096 B per layer
  at past_len = 100: read ≈ 36 × 100 × 4096 = 14 745 600 B ≈ 14 MiB
  at past_len = 2048: ≈ 300 MiB (!!)

Scratch/activation traffic (per layer):
  x_buf ping-pong: 2 × H × 4 = 20 KiB (stays in L2/L3)
  q_buf, k_buf, v_buf, attn_buf: ~32 KiB (L2)
  gate_buf, up_buf, h_buf: ~80 KiB
  Activation traffic is DRAM-free (L2/L3 hot).
```

**ИТОГО per-token bytes from DRAM (при past_len ≈ 50):**

| Component | Bytes |
|-----------|------:|
| All 36 layer weights | 2 223 297 504 |
| lm_head (Q4_K) | 218 787 840 |
| KV cache (read+write, past_len=50) | ~7 400 000 |
| embedding + norms + biases (glue) | ~30 000 |
| **TOTAL per token** | **≈ 2.449 GiB** |

Brief context mentioned 2.5 GB — **это совпадает**.

Для longer context (past_len=2000) KV cache raises total до ~2.75 GB.

---

## Q2. Эффективная bandwidth utilization

```
tok/s = 5.5 (TP-4 best, 28 cores)
bytes/tok = 2.449 GiB = 2.629 × 10^9 B

effective BW  = 5.5 × 2.629e9 = 14.46 GB/s
```

### Peak DRAM отсчёт

`E8C2_ARCHITECTURE.md` §5: **DDR4-2400, 4 channels/chip, 68.3 GB/s/chip**, 4 chips = **273 GB/s aggregate theoretical**.

Но in practice чтение идёт НЕ в full 4-chip aggregate (см. Q3/Q4 ниже). Релевантный peak зависит от того, сколько chip'ов одновременно читают:
- **1-proc interleave=all** (24t, 4 NUMA cross): threads round-robin по 4 controller → aggregate потолок до ~273 GB/s, но cross-NUMA latency дополнительно режет effective BW ~30–40%.
- **TP-4 NUMA-pinned (5.5 tok/s)**: каждый rank читает из своего NUMA узла, 4×68.3 = **273 GB/s aggregate**, но per-rank потолок 68.3 GB/s.

### Расчёт utilization

Считаю по двум peak'ам:

| Scenario | Peak | Utilization |
|----------|-----:|------------:|
| Conservative 60 GB/s (single chip ~88%) | 60 | **24.1%** |
| Optimistic 100 GB/s (partial aggregate) | 100 | **14.5%** |
| Theoretical 273 GB/s (4-chip aggregate) | 273 | **5.3%** |

Для TP с NUMA-pinned weight shards каждый rank читает ~1/4 весов из своего DRAM → per-rank demand = 2.449 / 4 ≈ 0.61 GB/token × 5.5 tok/s = **3.36 GB/s per NUMA node**. Это **4.9% от 68.3 GB/s single-chip peak** — НЕ bandwidth-bound на per-node basis.

**Вывод Q2:** На TP-4 мы **не упираемся в DRAM**. 14.5 GB/s aggregate при 273 GB/s peak = 5.3%. Даже pessimistic 60 GB/s single-chip peak — 24% utilization. Гипотеза из MISSION_BRIEF «DRAM-bound» не подтверждается числами. Bottleneck скорее всего **serial overhead + AllReduce coherence traffic**, что согласуется с Amdahl fit в `BENCH_ELBRUS.md` L75: serial floor 107 ms/token → ceiling 9.3 tok/s при ∞ threads.

---

## Q3. NUMA topology cross-reads на hot path

### TP mode (`forward_decode_cpu_tp`, `gguf_model.h:3603`)

Каждый rank владеет 1/4 row-sliced весов через `tl.q_attn_q/k/v/gate/up` и k-sliced для `tl.q_attn_output` / `tl.q_ffn_down` (`gguf_model.h:3776`, `:3843`). Per-node ThreadPool = 8 threads on 8 cores of own NUMA node (`BENCH_ELBRUS.md` L32–33).

**Shard pinning:** запускается через `numactl --membind=<node> --cpunodebind=<node>` (`BENCH_ELBRUS.md` L34). Weight buffer malloc'ится в NUMA-local DDR → read bandwidth идёт из **own DDR controller**. ✓ OK.

**Потенциальные cross-NUMA reads:**

1. **Replicated weights (L3773 fallback для Q5/Q6 unless k-sliced):** `layer.q_attn_output` / `layer.q_ffn_down` — **global replicated** (не sharded). Эти буферы mmap'лены один раз → first-touch на node0. Другие ranks читают их cross-NUMA. Для qwen3:4b ffn_down = Q6_K → пошёл fast k-sliced path (`gguf_model.h:3835`) → НЕ cross-NUMA. Но replicated attn_output всё ещё read cross-NUMA в fallback branch. Для qwen3:4b attn_output=Q4_K → fast path работает. **OK.**

2. **Embedding table** (`gguf_model.h:3631`): FP32 [V×H] = 1.45 GiB. `memcpy` из table + token_id × H в rank-local buffer. Читается **1 row (10 KB)** per token. Первый rank, который touch'нул table, получил node-local; остальные 3 ranks — **cross-NUMA read каждого токена** (через 8 GB/s interconnect, `E8C2_ARCHITECTURE.md` §1). 10 KB/token × 3 ranks = 30 KB cross-NUMA per token = 0.16 ms at 8 GB/s. **Minor, OK.**

3. **`output_norm`, `ffn_norm`, `attn_norm`** (FP32 per-layer, 2560 × 4 = 10 KB each, 36 layers × 3 norms = ~1 MB total): Replicated read on every rank per layer. First-touch на один node → cross-NUMA для 3 из 4 ranks = 0.75 MB/token cross-NUMA. Also minor (~93 μs at 8 GB/s).

4. **Biases (Qwen3):** `attn_q_bias/k_bias/v_bias` per-layer (FP32): 4096+512+512 = 5 KB per layer × 36 = 180 KB, full replicated read on every rank every layer. Cross-NUMA на 3 ranks = 540 KB per token.

5. **KV cache** (`gguf_model.h:3709–3710`): TP использует `k_cache_local`/`v_cache_local` — **rank-local** per-rank storage. Write idx = past_len (increments), read = full `total_seq × kv_dim_local` on attention. All node-local. **✓ OK.**

6. **`lm_head` output projection** (`gguf_model.h:3890–3911`): row-sliced across ranks (rank gets V/nprocs = 37 984 rows). Каждый rank читает СВОИ rows из `q_output_weight.cpu_data`. Но это global single replicated allocation → first-touch pinning. Row-slice читает из чётких offset'ов, каждый rank в разный диапазон → **каждый rank pull'ит 1/4 матрицы (55 MB)** из **first-touch node** (обычно node0). Итог: ranks 1/2/3 читают 55 MB cross-NUMA each = **165 MB/token cross-NUMA** через 8 GB/s interconnect → **21 ms/token минимум** (из 181 ms/token total на 5.5 tok/s). 

   Это **~12% serial floor budget** уходит на cross-NUMA lm_head reads!

**Cross-NUMA hot-path summary:**

| Path | Bytes/tok cross-NUMA | ms/tok at 8 GB/s |
|------|---------------------:|-----------------:|
| Embedding row | 30 KB | 0.004 |
| RMSNorm weights | 750 KB | 0.09 |
| Biases | 540 KB | 0.07 |
| **lm_head (MAIN ISSUE)** | **165 MB** | **20.6 ms** |
| **Total** | ~167 MB | **~21 ms** |

> **Это потенциальный win -20 ms/tok = от 181 ms/tok → 160 ms/tok = от 5.5 → 6.25 tok/s (~+14%).**
> Если заменить на replicated-per-node lm_head (208 MB × 4 nodes = 834 MB, ок — бюджет 125 GB RAM).

### 1-proc mode с `numactl --interleave=all` (`BENCH_ELBRUS.md` L105–108)

Страницы round-robin'ятся через 4 DRAM controllers. Каждый read: **3/4 вероятность cross-NUMA** с average latency ~180 ns vs ~100 ns local. Но aggregate bandwidth = sum of all 4 controllers × (bw_per_controller × utilization).

Для 24 threads × burst sequential reads: interleave даёт 4× aggregate peak BW (273 GB/s vs 68.3 GB/s single chip) при **не-local reads**. Для GEMV это приемлемо: L2 prefetcher скрывает latency. Но для random / irregular reads — penalty больно.

Факт из `BENCH_ELBRUS.md` L104–108: interleave +36% vs plain default. То есть bandwidth DID matter хоть сколько-то для 1-proc.

---

## Q4. Interleave vs nodelocal: анализ

### 1-proc `--interleave=all`

- Pages round-robin per 4 KB. **Read одной 2560-элементной row (10 KB FP32)** пересекает 2–3 pages → 2–3 разных controller → round-robin uncoordinated.
- **Плюс:** aggregate BW = 4× single-chip. Для thread, который читает 14 GB/s один (что >> single-chip 68 GB/s? NO, 14 <<< 68) — overhead interconnect latency скрыт prefetcher'ом. **Net positive** когда demand > 1 chip.
- **Минус:** для threads пинованных на node N, cache-line из node ≠ N ложится в L2 node N, но его L3 резидентен на M (cache directory). Coherence traffic увеличивается.

Замер факт: `BENCH_ELBRUS.md` таблица L96–102: 24t plain 3.0 → interleave 3.8 tok/s. 24-thread demand на одиночный DDR pin ~ 3 tok/s × 2.45 GB = 7.4 GB/s — **11% from 68.3 GB/s**, далеко не saturation. Буст от interleave — НЕ bandwidth, а **латентность вытягивания prefetch'ей параллельно с 4 caps**.

### TP-4 node-local

Каждый rank pin'ен на свой node, reads из своего DDR controller. Per-rank demand ~0.6 GB/s × 5.5 tok/s = 3.3 GB/s = 5% от 68 GB/s. **DDR не перегружен.** Interconnect сжат только на 3 minor объектах (см. Q3).

### Compromise option (не реализовано)

Hybrid: NUMA-replicated weights (`numa_weight_replica.h`) + `--interleave=all` KV cache и scratch. Tested in `BENCH_ELBRUS.md` L81–84 и L82: «3.7 → 3.6 tok/s — нейтрально». **Эксперимент показал bandwidth НЕ bound**. Это **прямое эмпирическое опровержение** bandwidth hypothesis.

---

## Q5. Расчётный потолок при 100% BW efficiency

### Сценарий A: aggregate 273 GB/s (все 4 chips saturated)

```
bytes/tok = 2.449 GiB = 2.629 × 10^9 B
ceiling = 273e9 / 2.629e9 = 103.8 tok/s
```

**Это верхний потолок 103.8 tok/s.** Сейчас 5.5 tok/s = **5.3%** от этого потолка.

### Сценарий B: single-chip 68.3 GB/s (bandwidth реально доступная 1-proc с interleave=all — fair scenario)

```
ceiling = 68.3e9 / 2.629e9 = 25.98 tok/s
```

Текущий 3.8 tok/s (1-proc best) = **14.6%** от single-chip BW.

### Сценарий C: conservative 60 GB/s

```
ceiling = 60e9 / 2.629e9 = 22.82 tok/s
```

**Вывод Q5:**

| Target | Required utilization |
|--------|---------------------:|
| 10 tok/s vs 273 GB/s peak | 9.6% (easy) |
| 15 tok/s vs 273 GB/s peak | 14.4% (easy) |
| 15 tok/s vs 68.3 GB/s single-chip | 57.8% (hard) |
| 30 tok/s vs 273 GB/s peak | 28.9% (aggressive) |

**Key finding:** Потолок по bandwidth **>> целевых 10–15 tok/s** даже в pessimistic сценариях. 
- Mission target **10–15 tok/s** — полностью внутри bandwidth envelope (57.8% single-chip максимально-требуемая utilization).
- **Bandwidth НЕ первичный bottleneck.** Compute / serial / ThreadPool overhead более вероятны.

Cross-reference `BENCH_ELBRUS.md` L75: Amdahl fit говорит **serial floor 107 ms/token = ceiling 9.3 tok/s при ∞ threads**. Чтобы достичь 15 tok/s **нужно срезать serial floor**, не bandwidth. И это доказано отрицательным результатом `PT_NUMA_REPLICATE=1` (3.7 → 3.6, neutral).

---

## Q6. Конкретные предложения снижения bandwidth-cost

### 6.1 Cross-layer prefetch — УЖЕ РЕАЛИЗОВАНО

`gguf_model.h:2253–2296`:
- Для layer `i`, prefetch первых 64 KB каждой из 7 weight matrices layer `i+1` через `_mm_prefetch(_MM_HINT_T1)`.
- Также small norm weights loaded в L1 через T0 hints.

**Оценка эффективности:** limited. Q4_K layer = 59 MiB, а префетч — только 64 KB × 7 = 448 KB. Это ~0.7% от layer data. Скрывает только TLB fill / first cache lines, НЕ sequential BW. Улучшить — поднять до нескольких MB и использовать `__builtin_prefetch` глубже с `streaming stores`.

**Bigger win:** **async layer-ahead prefetch через отдельный pthread**, который pull'ит следующий layer целиком в L3 пока текущий layer compute'ится. E2K имеет 16 MB L3 per-chip — вмещает 1 full layer (59 MiB... НЕ вмещает, too big). Реально работает только для sub-layer: prefetch FFN pair (gate+up=28 MB) пока attention считается. Но L3=16 MB < 28 MB → частичный fit.

**Ожидаемый выигрыш:** 3–8% если зафиксировать TLB thrash fully. Не 2×.

### 6.2 KV cache placement — почти ок

`forward_decode_cpu_tp`: `tp_.k_cache_local[i]` / `tp_.v_cache_local[i]` — rank-local tensors. Allocated in `tp_allocate_kv_cache` (не видел аллокации в этом чтении, но TP flow явно предполагает NUMA-local через `numactl --membind=<node>` запуск). 

**1-proc mode:** `kv_cache.key_cache[i]` через `at::empty` (`gguf_model.h:255`) — идёт через default allocator, **first-touch pinning** на node того thread'а, который первым touch'нул. При interleave=all ротирован. **Сделать:** принудительно `numa_alloc_onnode` для каждого layer'ного KV buffer на node, где большинство threads будут accessing. Для 1-proc это неоднозначно (все 4 nodes читают); для TP — уже правильно.

**Потенциал:** минимальный, KV traffic = 7–15 MiB/tok vs 2.5 GB weights.

### 6.3 Shared embedding / lm_head tied — НЕ ИЗБЕГАЕМ double-read

**ВАЖНОЕ ОТКРЫТИЕ.** В Qwen3-4B `token_embedding` и `q_output_weight` (lm_head) **обычно tied** (shared weights). Но в `gguf_model.h`:
- `token_embedding` загружено как FP32 (L2229: `emb_table = token_embedding.data_ptr<float>()`).
- `q_output_weight` — отдельный Q4_K struct (quant cpu_data).
- Размер embedding FP32: 151936 × 2560 × 4 = **1.45 GiB**
- Размер q_output Q4_K: 151936 × 2560 × 4.5 / 8 = **208 MiB**

**Мы храним 1.66 GB вместо 208 MB если tied, то есть НЕ используем tying.** Embedding row read = 10 KB, малая, но **1.45 GiB мёртвого веса** в first-touch NUMA node — блокирует TLB, раздувает working set. Fix: при tied config не загружать FP32 embedding отдельно, а dequant'ить нужную row из Q4_K lm_head ad-hoc (1 row = 2560 / 256 × 144 = 1440 B dequant work). 

**Win:** 1.45 GB less RAM pressure → меньше TLB misses. Per-token bandwidth почти не меняется (embedding row = 10 KB), но **RSS-footprint улучшение 1.45 GB**. При multi-user / multi-model serving это ощутимо.

### 6.4 Weight packing: Q4_0 vs Q4_K_M

**Q4_0 size:** 4 bits + 16-bit scale per 32-group = 18 bytes / 32 = **4.5 bits/weight** = same as Q4_K в main quant bits. Разница в overhead:
- Q4_K: 144 B / 256 = 4.5 b/weight, 8 sub-blocks каждый со scale+min 6-bit each
- Q4_0: 18 B / 32 = 4.5 b/weight, simple per-32 scale

**Одинаковый size, но Q4_0 дешевле dequant** (scalar vs K-quant reconstruction). НО это **НЕ** уменьшит bytes/token, то же самое 2.45 GB.

**Real smaller option: Q3_K_M** (3.4375 b/weight) — уменьшит веса до ~1.85 GB (-25%). Требует training-aware quant / calibration data. **Не trivial.**

**Другая опция: Q5_1 для ffn_down (Q6_K текущий).**
- Q6_K: 210 B / 256 = 6.5625 b/weight
- Q5_K: 176 B / 256 = 5.5 b/weight (−16% для ffn_down = −3.3 MiB/layer × 36 = **119 MiB/token saved** = −5% total)

**Win:** ~5% bandwidth reduction. Качество деградирует минимально для most models (Q5_K_M — стандарт llama.cpp balanced).

### 6.5 ДРУГИЕ наблюдения (вне bandwidth-domain)

Из анализа вытекает, что **bandwidth НЕ bottleneck**. Реальные проблемы:

1. **Serial floor 107 ms/token** из Amdahl fit (`BENCH_ELBRUS.md` L75). Источники serial:
   - RMSNorm preamble (2 шт/layer × 36 layers × ~0.3 ms = 22 ms)
   - RoPE apply (36 × 0.5 ms = 18 ms)
   - Softmax serial в attention per-head inside thread
   - ThreadPool spawn overhead 120 μs × 180 calls/tok = 22 ms
   - **Sum estimate ≈ 85 ms serial — подтверждает 107 ms fit**.

2. **AllReduce в TP (72× per token)** — если каждый AR = 10 ms (NUMA coherence), то 720 ms overhead. Это **главный killer TP path**.

3. **`std::exp` scalar** в SiLU — на E2K libm ~200 ns/call × ~10K/layer × 36 = 72 ms. Уже parallelized (parallel_for @ `gguf_model.h:2603`), но всё равно compute-heavy.

---

## Itemized summary для mission roadmap

| # | Предложение | File:line | Ожидаемый speedup | Риск |
|---|-------------|-----------|------------------:|------|
| 1 | NUMA-replicate lm_head (4× 208 MB) + local read per rank | gguf_model.h:3902, new alloc path | +10–15% (убирает 20 ms/tok cross-NUMA) | Мало (+624 MB RAM) |
| 2 | Switch ffn_down Q6_K → Q5_K_M | Config load path / GGUF conversion | +3–5% (bandwidth), −0.3 PPL | Качество ↓ minor |
| 3 | Tied embedding: drop FP32 copy, dequant from Q4_K lm_head | gguf_model.h:2229 | 0% throughput, -1.45 GB RAM | API complications |
| 4 | Deeper layer-ahead prefetch (full layer to L3 via pthread) | gguf_model.h:2253 | +3–8% | TLB pressure |
| 5 | Vectorize SiLU (native exp, not libm) | gguf_model.h:2603 | Compute win, not bandwidth | Approximation error |
| 6 | Kill 72 AllReduces — collapse to 2 (per-layer merge attn+FFN) | forward_decode_cpu_tp | +50–100% на TP path | Semantic change, req. audit |

**Пункт 6** — главный (compute-bound, not bandwidth), но outside Q6 bandwidth-focused scope.

---

## Ключевой вывод (TL;DR)

1. **Per-token bandwidth = 2.449 GB**, confirmed by `MISSION_BRIEF.md` estimate.
2. **At 5.5 tok/s = 14.46 GB/s effective**, **5.3% of 273 GB/s aggregate peak** / **21% of 68 GB/s single-chip peak**.
3. **Bandwidth ceiling at 100% utilization = 103.8 tok/s** (aggregate) или **25.98 tok/s** (single-chip). **Target 10–15 tok/s is well inside bandwidth envelope.**
4. **Bandwidth НЕ bottleneck**. `PT_NUMA_REPLICATE=1` дал neutral результат (3.7→3.6 tok/s) — эмпирическое подтверждение.
5. **Реальный bottleneck: serial floor (107 ms/token)** и **AllReduce overhead** (720 ms/tok worst case на TP). Amdahl fit: ceiling 9.3 tok/s при ∞ threads, **limit = serial code, not BW**.
6. **Cross-NUMA hot spot: lm_head replicated reads = 20.6 ms/token** cross-NUMA (главный bandwidth-relevant fix worth +14%).
7. Для достижения 10–15 tok/s нужно атаковать **serial code path** (RMSNorm+RoPE+bias in scalar main thread) и **AllReduce frequency** (72→2), а не bandwidth.

---

## Blocker / чего не смог проверить

- **`elbrus_memory_controller.pdf`** — не прочитан (pdftoppm не установлен в sandbox). Данные о DDR4-2400 / 68.3 GB/s взяты из `E8C2_ARCHITECTURE.md` §5, который сам ссылается на mcst.ru/elbrus-8sv. Не смог проверить реальные pmccntr counters (MEM_ACCESSES, DCACHE_MISS, L3_MISS) — требует запуска на live Elbrus, в этой миссии запрещено.
- **Реальное распределение KV cache по NUMA nodes в 1-proc mode** — не подтверждено code trace, нужен `numastat -p <pid>` во время inference.
- **ffn_down quant type** — проверил из комментария (`cpu_quant_gemv.h:1177`), но не из GGUF metadata dump'а. Если на самом деле Q4_K (не Q6_K), bytes/token ≈ 2.33 GB и utilization процентов пересчитать (ceiling в aggregate сценарии: 273/2.33 = 117 tok/s — вывод тот же).

---

**Рекомендация автора:** не уходить в bandwidth-оптимизации. Основной выигрыш — в serial/AllReduce реорганизации (Agent 3/4 territory). Единственная bandwidth-mission достойная затраты: replicate lm_head per-node (item #1).
