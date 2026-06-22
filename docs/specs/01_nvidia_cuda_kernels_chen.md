# Спецификация CUDA-ядер PromeServe: путь к bandwidth-пику A100

**Автор:** Dr. Marcus Chen (Senior CUDA Kernel Architect)
**Зона ответственности:** квантованный GEMV и attention для decode (N=1) и prefill.
**Цель:** упереться в memory-bandwidth roofline A100 (1.55 ТБ/с HBM2e) и обогнать Ollama/llama.cpp.
**Дата:** 2026-06-21

---

## 0. TL;DR замеров и где мы стоим относительно roofline

Decode LLM (N=1, greedy) — это **чистый memory-bound GEMV-марафон**. Производительность определяется
тем, сколько байт весов мы прочитали и насколько близко к пику HBM. Арифметика (dequant + MAC) **обязана
быть бесплатной** — спрятана за загрузкой памяти. Если она не спрятана — мы compute-bound на ALU, и это баг.

Текущее (A100, decode, greedy):

| Модель      | Мы, tok/s | Ollama, tok/s | Отставание |
|-------------|-----------|---------------|------------|
| qwen3:4b    | 88        | 188           | 2.14×      |
| qwen3:14b   | 33        | 88            | 2.67×      |
| gemma3:27b  | 16.5 (+кривой вывод) | — | — |

**Вывод сразу:** отставание растёт с размером модели (2.14× → 2.67×). Это сигнатура того, что мы **не
насыщаем HBM на больших матрицах** — то есть основное ядро GEMV эффективно по bandwidth НЕ дотягивает,
и чем больше байт надо прокачать, тем заметнее.

---

## 1. Roofline-расчёт: сколько вообще «стоит» токен

### 1.1 Байты на токен (qwen3:4b, Q4_K_M)

Параметры qwen3:4b: hidden H≈2560, intermediate I≈9728, n_layers L=36, vocab≈151k, head_dim 128,
n_heads 32, n_kv_heads 8 (GQA). Q4_K_M = 4.5 бит/вес эффективно (144 байта на 256 весов = 4.5 bit).

Веса на слой (в байтах, при 4.5 bit/вес = 0.5625 байт/вес):
- QKV proj: H×(q_dim+2·kv_dim) = 2560×(4096+1024+1024)=2560×6144 ≈ 15.7M весов → 8.85 МБ
- attn_output: q_dim×H = 4096×2560 ≈ 10.5M → 5.9 МБ
- gate+up: 2×H×I = 2×2560×9728 ≈ 49.8M → 28.0 МБ
- down: I×H = 9728×2560 ≈ 24.9M → 14.0 МБ

Итого на слой ≈ **56.8 МБ**. На 36 слоёв ≈ **2.05 ГБ**. Плюс lm_head (vocab×H = 151k×2560 ≈ 387M весов;
если Q4_K ≈ 218 МБ, если FP16 ≈ 774 МБ) и embedding lookup (один ряд, копейки). Плюс KV-cache read в
attention (контекст-зависимо, при 1–2k контексте ~10–20 МБ/токен на всю сеть).

**Итог: ~2.3–2.5 ГБ чтения весов на токен** для qwen3:4b — цифра из ТЗ подтверждается.

### 1.2 Идеальное время на A100

A100 SXM HBM2e: пик ~1555 ГБ/с. Реально достижимо хорошим streaming-ядром: **~1350–1450 ГБ/с (87–93%)**.

- При 2.4 ГБ/токен и 1.4 ТБ/с: **t = 2.4/1400 ≈ 1.71 мс/токен → 585 tok/s** (теоретический потолок ядра).
- Ollama (llama.cpp) даёт 188 tok/s = 5.32 мс/токен = эффективная полоса **2.4 ГБ / 5.32 мс ≈ 451 ГБ/с**.
  llama.cpp тоже не на пике (launch overhead, KV, attention, sampling), но держит ~32% от HBM.
- Мы: 88 tok/s = 11.36 мс/токен = эффективная полоса **2.4 ГБ / 11.36 мс ≈ 211 ГБ/с ≈ 14% HBM**.

**Это диагноз №1: мы крутим ~14% полосы HBM против ~32% у llama.cpp.** Между нами и физикой ядра (585) —
фактор 6.6×; между нами и llama.cpp — 2.1×. Нам не нужно «обгонять физику», нам нужно перестать терять
полосу на ровном месте. Цель реалистичная: **300–400 tok/s** на qwen3:4b (≥ llama.cpp ×1.7–2).

### 1.3 Roofline для отдельного ядра gate+up (самое толстое)

gate+up читает 28 МБ/слой. При 1.4 ТБ/с — 20 мкс. При нашей эффективной 211 ГБ/с — 133 мкс. На 36 слоёв
разница 36×(133−20) = **4.1 мс** только на gate+up. Это и есть «потерянные такты».

---

## 2. Анализ текущих ядер: где утекает полоса

Дефолтный GPU decode-путь (`forward_decode`, gguf_model.h:2575+) при `use_quant_gemv_=true` и
`use_llama_gemv_=false` (это дефолт, см. строки 911/925) использует **FP32-shared-memory dequant-GEMV**:
`q4km_fused_rmsnorm_qkv_gemv_kernel`, `q4km_fused_gate_up_kernel`, `q4km_persistent_gemv[_accumulate]`,
`q6k_gemv_kernel`, `q5k_gemv_kernel`. dp4a-вариант `q4km_persistent_gemv_v2_kernel` существует, но
**висит за флагом `use_llama_gemv_`, который по умолчанию `false`** (строка 3127). То есть лучший по дизайну
kernel в проекте по умолчанию **не вызывается на главном пути** (QKV/gate-up/down идут через FP32-fused).

### 2.1 ПРОБЛЕМА A: главный путь — FP32 dequant, не dp4a (CUDAQuantGemv.cu:570-661, 1463-1740)

Возьмём ядро тела цикла `q4km_fused_gate_up_kernel` (строки 644-651) и его клонов в fused_rmsnorm_*:

```cpp
sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;   // 8 скалярных FMA/итерацию
sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
... ещё 6 строк ...
```

Что здесь не так с точки зрения roofline:

1. **Один warp = одна строка, dequant в FP32 на каждой итерации.** На каждый блок из 256 весов warp
   делает: 2× `get_scale_min_k4_device` (внутри — ветвления `if (j<4)`, см. строки 49-59), 4 `fp16→fp32`,
   и 8 FP32-FMA на лейн. FP32-FMA на A100 = 19.5 TFLOPS. Для bandwidth-bound это не должно мешать — НО при
   warp-per-row каждый лейн обрабатывает только 8 весов на блок, регистровое давление высокое, а **MAC
   считается в FP32 по одному**. llama.cpp здесь использует `__dp4a` (INT8×INT8→INT32, 4 MAC/инструкция,
   на A100 это путь через DP4A с эффективной пропускной в разы выше) — ровно то, что делает `..._v2_kernel`,
   который выключен.

2. **x грузится в shared memory как FP32 (`x_shared`, строка 579, 588-590).** Для gate+up при H=2560 это
   10.2 КБ smem только под x. Это ограничивает occupancy: при `grid = sm_count*4` и 256 threads/block
   с 10–12 КБ динамического smem на блок мы получаем ~4 блока/SM, но реальный bottleneck — **не смем, а то,
   что warp-per-row не даёт достаточно warp'ов в полёте, чтобы спрятать латентность загрузки 144-байтных
   блоков весов**. На A100 для насыщения HBM нужно держать ~достаточно независимых memory-транзакций в полёте
   (Little's law: BW × latency ≈ in-flight bytes; при 1.4 ТБ/с и ~500 нс латентности это ~700 КБ в полёте).
   8 warps × несколько блоков на SM этого едва хватает.

3. **`get_scale_min_k4_device` с ветвлениями вызывается дважды на блок на каждый warp** (строки 631-632 и
   во всех fused-клонах). Это дивергентный код в горячем цикле. llama.cpp распаковывает все 8 scale/min
   **один раз на блок в регистры** (это и есть «bulk scale load» в `..._v2`, строки 446-475). На главном
   пути этого нет.

4. **Дублирование тела цикла 6 раз.** Тело GEMV скопировано в: `q4km_gemv_kernel`, `fused_gate_up`,
   `fused_qkv`, `fused_rmsnorm_gemv`, `fused_rmsnorm_qkv`, `fused_rmsnorm_gate_up`,
   `persistent_gemv_accumulate`. Все — FP32 dequant вариант. Любой фикс надо делать в 6 местах → риск
   рассинхрона. Архитектурно: тело должно быть **одним `__device__ __forceinline__` примитивом**.

### 2.2 ПРОБЛЕМА B: `q6k_gemv_kernel` крайне неэффективен (строки 881-958)

Q6_K используется в Q4_K_M миксе (attn_v/ffn_down на каждом ~3-м слое — см. MEMORY) и для Q6_K-моделей.

```cpp
if (lane < 32) {            // строка 929 — ВСЕГДА true, лейнов ровно 32. Мёртвая ветка.
    ...
    float x1 = x_shared[k_base + l +  0];   // строки 939-942: 4 СКАЛЯРНЫХ load из smem,
    float x2 = x_shared[k_base + l + 32];   // НЕ float4. Bank conflicts + 4× инструкций.
    ...
    sum += d * scales[is + 0] * q1 * x1;    // FP32 MAC по одному
}
```

Проблемы:
- `qh` читается побайтово (`qh_h[l]`), `ql` — тоже скалярно. **Нет vectorized load** (нет uint4/float4 для x).
- scale читается из `int8_t* scales` индексами `scales[is+0..is+6]` — некоалесцированно.
- Нет dp4a-пути для Q6_K вообще. llama.cpp имеет `vec_dot_q6_K_q8_1` на dp4a.
- Внешний цикл `for half` + внутренний разбор 2-битных qh → высокая ALU-нагрузка, при том что
  всё ещё memory-bound. На больших down-проекциях (I×H) это ядро — вероятный пик потери на 14B/27B.

### 2.3 ПРОБЛЕМА C: attention (FlashDecoding.cu) — скалярные дот-произведения, thread-0 редукции

`flash_decode_partial_graph_kernel` (строки 756-839) и его FP16-вариант:

1. **QK^T полностью скалярный:** строка 802 `for (int d=0; d<head_dim; d++) dot += q_shared[d]*k_vec[d];`.
   head_dim=128, один thread считает весь дот по одному float. Нет float4, нет warp-кооперации по d.
   `k_vec` читается из global построчно — на FP32-кэше это 128×4=512 байт некоалесцированно по потоку.
2. **Редукции max/sum делает ТОЛЬКО thread 0** (строки 808-812, 818-822): `if (tid==0) { for t ... }`.
   Это сериализует softmax по chunk_len на одном потоке. При длинном контексте — десятки последовательных
   итераций на одном лейне, остальные простаивают.
3. **Weighted-V sum (строки 825-832):** каждый thread по d делает скалярный цикл по chunk_len, читая
   `V_cache[...]` по одному float. Снова нет векторизации, нет coalescing по соседним t.
4. **FP16 KV-cache в decode отключён** (комментарий 2817-2820: «baked offsets break CUDA Graph»). То есть
   attention в decode читает KV в **FP32 — вдвое больше байт**, чем нужно. При длинном контексте KV-чтение
   становится заметной долей, и мы её удваиваем.
5. **Два kernel-launch на attention** (partial + reduce) на каждый слой; reduce запускается даже при
   `num_splits==1` (строки 230-243, no-op ветка пустая — kernel всё равно стартует).

llama.cpp flash-decode: warp/quarter-warp кооперативный дот по head_dim, FP16 KV cache по умолчанию,
один проход с online-softmax. Наш вариант алгоритмически правильный, но реализован «учебно».

### 2.4 ПРОБЛЕМА D: CUDA Graph есть, но launch-структура раздута

Хорошо: граф захватывается со 2-го токена (строки 2630-2640, 2656), replay = 2 `cudaMemcpyAsync` +
`cudaGraphLaunch` + sync. Это правильно и убирает CPU launch overhead. НО:

1. Внутри графа на слой запускается: fused_qkv (1) + 3 bias-add (опц.) + qknorm_rope_kv (1) +
   flash_decode partial+reduce (2) + output (1) + fused_gate_up или norm_gate_up (1) + silu_mul (1) +
   down/accumulate (1) + residual (часто отдельный add). Это **~9-12 ядер/слой × 36 = 350-430 ядер в графе**.
   Графа прячет launch latency, но НЕ прячет то, что каждое ядро — отдельный проход с собственным
   `__syncthreads` барьером и (часто) перезагрузкой x в smem. Bias-add (3 launch, строки 2773-2783) —
   три отдельных ядра по одному вектору; должно быть слито в GEMV-эпилог.
2. **`silu_mul` отдельным ядром** + down отдельным — между ними round-trip gate/up через global. На больших
   I это лишний proход 2×I×4 байт чтение/запись на слой.
3. Финальный `cudaStreamSynchronize` на каждый токен (строка 2637) — корректно для чтения логитов CPU, но
   значит мы **не конвейеризуем sampling со следующим decode**. llama.cpp на GPU держит sampling на девайсе.

### 2.5 ПРОБЛЕМА E: lm_head FP16 через cuBLAS — но это GEMV, тензор-ядра простаивают

Строки 3066-3074: lm_head идёт через `cublasHgemv`/`cublasGemmEx` с FP16 (Tensor Cores). Для N=1 это GEMV
с m=N_vocab, n=1, k=H. **Tensor Cores на GEMV (n=1) бесполезны** — это bandwidth-bound, MMA не помогает
(подтверждается нашей же памятью: «FP16-weights ≠ speedup для decode»). Хуже: FP16 lm_head = vocab×H×2 =
**774 МБ** чтения против Q4_K **218 МБ**. Мы читаем **в 3.5× больше байт на lm_head ради Tensor Cores,
которые на GEMV не работают.** На qwen3 (vocab 151k) lm_head — это ~0.55 мс лишних на токен при 1.4 ТБ/с,
а при FP16 — почти 2 мс. Это прямая регрессия на каждом токене.

### 2.6 Сводная таблица «где такты»

| Место | Файл:строки | Проблема | Цена |
|-------|-------------|----------|------|
| QKV/gate/up/down | CUDAQuantGemv.cu:570-661,1463-1984 | FP32 dequant, не dp4a; warp-per-row; scale-unpack в цикле | главный лосс полосы |
| dp4a-ядро выключено | gguf_model.h:3127 | `use_llama_gemv_=false` дефолт | лучший kernel не используется |
| Q6_K GEMV | CUDAQuantGemv.cu:881-958 | скалярные smem-load, нет float4/dp4a, мёртвая ветка | пик на down/14B/27B |
| Attention QK/V | FlashDecoding.cu:799-832 | скалярный дот, thread-0 softmax | растёт с контекстом |
| KV FP16 off в decode | gguf_model.h:2817-2820 | KV читается FP32 (2× байт) | растёт с контекстом |
| lm_head FP16 | gguf_model.h:3066-3074 | 3.5× байт ради бесполезных TC | ~1.5 мс/токен |
| bias-add ×3, silu отдельно | gguf_model.h:2773-2783, FFN | лишние проходы global | мелочь × 36 слоёв |

---

## 3. Целевой дизайн ядер

Принцип: **одно семейство GEMV-примитивов, dp4a-первое, vectorized loads, scale-unpack один раз на блок,
эпилоги слиты, x квантуется в Q8_1 один раз на слой.** Ниже — целевые контракты.

### 3.1 Базовый примитив: `q4k_q8_dot_block` (device inline)

Единый `__device__ __forceinline__` для тела Q4_K×Q8_1 блока (256 весов), используемый ВСЕМИ launch-обёртками
(QKV, gate, up, down, accumulate, lm_head). Контракт:

- Вход: указатель на 144-байтный Q4_K блок, указатель на соответствующие Q8_1 sub-блоки x в smem, lane.
- **Загрузки весов через `uint4`/`int4`** (16 байт за транзакцию): header (d|dmin|12 scale байт) — один
  `__ldg(uint4*)` как в `..._v2` (строка 448); qs — `__ldg(uint32_t*)` или, лучше, по 2 блока за раз `uint2`.
- **scale/min всех 8 sub-блоков распаковываются ОДИН раз на блок в регистры** (убрать дважды-вызов
  `get_scale_min_k4_device` в горячем цикле — заменить на табличную/битовую распаковку без ветвлений).
- **MAC через `__dp4a`** (INT8): `dot = __dp4a(v_lo, u_lo, dot)` (как строки 503-504), масштабирование во
  float в самом конце на блок. Bias-коррекция `dmin·m·sum(q8)` через `__dp4a(0x01010101, u, 0)`.
- Накопление в FP32 register-аккумуляторе.

Это превращает 8 FP32-FMA/лейн/блок в 2-4 dp4a + горстку scale-float — арифметика гарантированно прячется
за загрузкой 144 байт. Численность бит-идентична `..._v2` (T=0), уже проверена.

### 3.2 GEMV launcher: 2 строки/warp (ILP) + персистентный grid-stride

Взять структуру `q4km_persistent_gemv_v2_kernel` (NROWS=2, bulk scale, dp4a — строки 375-536) как **главный
и единственный** Q4_K GEMV. Доработки:

- **NROWS=4** (а не 2): 4 выходных строки на warp, общий Q8_1-x из smem читается раз и переиспользуется 4×.
  Это поднимает arithmetic-per-byte и, главное, увеличивает число независимых weight-load в полёте на warp
  → лучше прячет латентность HBM (Little's law). Регистров хватает (4 FP32 аккумулятора).
- **x квантуется в Q8_1 один раз на слой**, а не в каждом GEMV-ядре. Сейчас каждое persistent-ядро
  переквантует x в smem заново (строки 188-209). При fused_qkv + gate+up + down это 3-4 переквантования
  одного и того же (после нормы) вектора. Вынести `quantize_q8_1` в отдельный мелкий launch (или в эпилог
  rmsnorm) → x_q8 в global, GEMV читает Q8_1 из L2 (он крошечный, кэшируется).
- **grid = sm_count × (2..4) блока**, 8-16 warps/block. Подобрать так, чтобы occupancy не резалась smem
  (после выноса Q8_1 в global smem под GEMV почти не нужен → выше occupancy → больше warp'ов → выше BW).
- Сделать дефолтным: убрать `use_llama_gemv_` развилку, этот путь — основной (3.1+3.2 покрывают и численность).

Ожидание: gate+up с 133 мкс/слой → к 25-35 мкс/слой (× ~4 на этой части).

### 3.3 Q6_K / Q5_K: переписать на dp4a + vectorized

- **Q6_K**: ввести `vec_dot_q6_k_q8_1` по образцу llama.cpp: ql/qh грузить через `uint4`/`uint2`,
  собрать 6-битные значения в int8, `__dp4a` против Q8_1-x. Убрать мёртвую `if (lane<32)` (строка 929),
  убрать 4 скалярных `x_shared[...]` (строки 939-942) → читать x как Q8_1 из smem/L2.
- **Q5_K**: аналогично, 5-й бит из qh складывать в int8 перед dp4a; заменить `float4 x_lo/x_hi`
  (строки 1068-1069) на Q8_1-путь.
- Цель: Q6_K/Q5_K по эффективной полосе сравнялись с Q4_K (сейчас они кратно медленнее).

### 3.4 Attention: warp-кооперативный flash-decode + FP16 KV в decode

- **QK^T:** один warp на (head, kv-tile); дот по head_dim=128 кооперативно 32 лейнами через `float4`
  (или half2 при FP16 KV), warp-shuffle-reduce. Убрать скалярный `for d` (FlashDecoding.cu:802).
- **softmax:** max и sum через warp-shuffle, НЕ thread-0 (убрать строки 808-822). online-softmax (running
  max/denom) чтобы слить шаги 2-4 в один проход по KV-тайлу.
- **V-sum:** кооперативно по d, `float4`/`half2` загрузки V, соседние t коалесцированно.
- **Включить FP16 KV cache в decode.** Проблема «baked offsets break CUDA Graph» (комментарий 2817) решается
  тем же приёмом, что уже работает для FP32: читать `past_len` из `d_past_len_` внутри ядра (graph-вариант
  `flash_decode_fp16_partial_graph_kernel` уже существует, строки 868-931 — он graph-совместим!). Значит KV
  cache write в FP16 нужно тоже сделать через device-`past_len` (по аналогии с
  `fused_qknorm_rope_kvwrite_graph_kernel`). Это **−50% байт на KV-чтение** в attention.
- При `num_splits==1` (типичный decode при контексте <256) **не запускать reduce-kernel** — нормализовать
  в самом partial-ядре. Убирает 36 лишних launch из графа.

### 3.5 lm_head: вернуть Q4_K GEMV (или dp4a), убрать FP16

Заменить `cublasHgemv` на lm_head (строки 3066-3074) на dp4a Q4_K GEMV из 3.1. Читаем 218 МБ вместо 774 МБ.
Tensor Cores на GEMV(n=1) не дают ничего; FP16 здесь — чистая регрессия. Экономия ~1.3-1.5 мс/токен.
(Tensor Cores оставить ТОЛЬКО для prefill — см. 3.7.)

### 3.6 Эпилоги: слить bias, silu, residual в GEMV

- **bias-add** (строки 2773-2783): передавать bias-указатели в QKV-GEMV, добавлять в эпилоге перед записью
  `y[n]`. Минус 3 launch/слой.
- **silu·mul**: эпилог fused_gate_up должен писать сразу `silu(gate)·up` (gate и up считаются в одном ядре —
  уже есть структура fused_gate_up, добавить в конце на лейн-0 `silu(g)*u`). Убирает отдельный silu-kernel и
  round-trip gate/up через global.
- **residual**: уже частично слит (`persistent_gemv_accumulate`, `y[n]+=`). Распространить на down-проекцию.

### 3.7 Prefill: вот ЗДЕСЬ нужны MMA/dp4a-GEMM, а не GEMV

Prefill (seq_len>1) — это compute-bound GEMM (M=seq, N=rows, K=hidden). Сейчас prefill идёт через `matmul_q`
(строки 7059+) → `launch_q4km_persistent_gemv` per-token (GEMV в цикле). Это **катастрофически медленно для
длинных промптов** — мы делаем M отдельных GEMV вместо одного GEMM. Целевой дизайн:
- Dequant Q4_K → FP16 тайлами в smem (ядро `dequant_q4k_to_fp16` уже есть, строки 1115-1190) и гнать через
  **cuBLAS GemmEx FP16 (Tensor Cores)** для prefill, ИЛИ
- Прямой **mma.m16n8k16 INT8** (dp4a/IMMA) Q4_K×Q8 GEMM — тут MMA реально окупается (M велик).
Это вне горячего decode-пути, но определяет TTFT (time-to-first-token) на больших промптах.

---

## 4. План внедрения по приоритету (ожидаемый прирост)

| # | Действие | Файлы | Эффект |
|---|----------|-------|--------|
| 1 | Сделать dp4a `..._v2` (NROWS=4, Q8_1-x в global, single quantize/layer) дефолтным GEMV для QKV/gate/up/down/lm_head; вынести тело в общий device-inline | CUDAQuantGemv.cu, gguf_model.h:3127,3066 | **×1.6-2.2** (главный) |
| 2 | lm_head: FP16 cuBLAS → Q4_K dp4a GEMV | gguf_model.h:3066-3074 | −1.3-1.5 мс/токен (~×1.1-1.15) |
| 3 | Attention: warp-кооп дот+softmax + FP16 KV в decode + skip reduce при splits=1 | FlashDecoding.cu, gguf_model.h:2817 | ×1.1-1.3 (растёт с контекстом) |
| 4 | Q6_K/Q5_K → dp4a + vectorized | CUDAQuantGemv.cu:881-1106 | ×1.2-1.4 на Q4_K_M-микс/14B/27B |
| 5 | Слить bias/silu/residual эпилоги; sampling на GPU (убрать per-token sync) | gguf_model.h | ×1.05-1.1 |
| 6 | Prefill GEMM (MMA/cuBLAS FP16) вместо GEMV-в-цикле | gguf_model.h:7059+ | TTFT ×3-10 на длинных промптах |

Совокупно по decode: реалистичный таргет **qwen3:4b 88 → 280-360 tok/s** (обгон Ollama 188 в 1.5-1.9×),
qwen3:14b 33 → ~110-130. Это всё ещё ниже физического потолка ядра (585), что нормально: остаётся attention,
KV, embedding, sampling.

---

## 5. Метрика приёмки

Каждое изменение GEMV-пути ОБЯЗАНО мериться через Nsight Compute по одному ядру:
- **`dram__throughput.avg.pct_of_peak_sustained_elapsed`** — целимся ≥ 80% на gate/up/down GEMV.
- **`sm__warps_active.avg.pct_of_peak_sustained`** — occupancy (хотим ≥ 50%, чтобы прятать латентность).
- Эффективная полоса = (байты весов слоя)/(время ядра); цель ≥ 1.1 ТБ/с на толстых GEMV.
- End-to-end tok/s + бит-точность top-1 vs текущий (T=0). Падение tok/s >0.2 на горячем пути = откат
  (правило проекта «скорость 11.4 — святая», тут аналогично для GPU).

Без Nsight-числа полосы изменение принимать нельзя — «стало быстрее на глаз» не считается.

---

## Приложение: почему llama.cpp быстрее (по памяти ggml-cuda)

1. **dp4a/quantized-q8_1 mul-mat-vec по умолчанию** для всех K-квантов (mmvq). Мы — FP32 dequant на главном пути.
2. **Один scale-unpack на блок в регистры**, vectorized weight loads. У нас scale-unpack дважды на блок в цикле.
3. **FP16 KV cache + warp-кооперативный flash-decode** из коробки. У нас FP16 KV в decode выключен, дот скалярный.
4. **GPU-side sampling**, нет per-token хост-синка в горячем цикле. У нас `cudaStreamSynchronize` на токен.
5. **lm_head тем же квантом**, не раздувается в FP16. У нас lm_head FP16 = 3.5× байт.
Мы НЕ отстаём архитектурно — у нас есть и graph, и flash-decode, и dp4a-ядро. Мы просто **не включили
лучший путь по умолчанию** и оставили attention/lm_head/Q6_K в «учебной» FP32-форме.
