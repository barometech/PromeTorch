# Спецификация: квант- и precision-стратегия PromeServe

**Автор:** Dr. Lin Wei (NVIDIA, quantization / Tensor Cores)
**Дата:** 2026-06-21
**Зона:** квант-форматы GGUF (Q4_K/Q5_K/Q6_K/Q8_0/F16) и утилизация Tensor Cores
**Железо-референс:** NVIDIA A100 80GB SXM (BW 2039 GB/s, FP16/BF16 TC 312 TFLOPS, FP32 19.5 TFLOPS, INT8 TC 624 TOPS, dp4a INT8 ALU ~78 TOPS)

---

## Executive summary (где недоутилизировано железо)

1. **Prefill НЕ использует Tensor Cores вообще.** `matmul_q()` на `M>1` (gguf_model.h:7088-7096) запускает **M отдельных bandwidth-bound GEMV-кернелов в цикле** — по одному на токен промпта. Это катастрофа: prefill compute-bound, а мы гоним его как M независимых memory-bound проходов по весам. Веса перечитываются из HBM M раз. Для prompt в 512 токенов это ×512 лишних чтений всех весов. **Главная причина gap 88 vs 188 tok/s на коротких ответах — медленный prefill (TTFT).**
2. **Decode (M=1) сделан правильно по формату:** on-the-fly Q4_K dp4a-GEMV (`q4km_persistent_gemv_v2`) — оптимальный bandwidth-путь. FP16-веса на decode НЕ выгодны (доказательство ниже §3): Q4_K читает 2.25 GB/проход, FP16 — 5.9 GB, ×2.6 больше HBM. `use_fp16_weights_` для decode оставить выключенным.
3. **F16-веса (attn_v) на decode идут через `launch_fp16_gemv` — наивный скалярный `__half2float` GEMV** (CUDAQuantGemv.cu:1277). Без half2-векторизации, без `__hfma2`. На decode это допустимо (bandwidth-bound), но лучше читать как `half2` (×2 ld-ширина, меньше инструкций).
4. **dp4a-упаковка Q4_K не оптимальна:** quantize-x в Q8_1 идёт **внутри каждого GEMV-кернела заново** (CUDAQuantGemv.cu:189-209). На decode x квантуется 7×/layer (qkv+o+gate+up+down). Нужен один `quantize_q8_1` на x перед всеми проекциями слоя.
5. **lm_head через cuBLAS HGEMV — это GEMV (n=1), не GEMM.** Tensor Cores на `n=1` простаивают (TC хотят M,N,K ≥ 16). На decode это всё равно bandwidth-bound и ок, но на prefill lm_head проецирует только последнюю позицию (gguf_model.h:2553) — упускается возможность настоящего HGEMM, если бы кому-то понадобились все логиты.
6. **Приоритет изменений:** (A) prefill → dequant-to-FP16 + cuBLAS **HGEMM** (M×N×K, TC) — ×5-10 TTFT; (B) единый Q8_1-quant x на слой; (C) half2-векторизация fp16_gemv; (D) опционально INT8 IMMA-prefill для ещё ×2.

---

## 1. Карта путей сегодня (что куда диспетчится)

### Decode (M=1) — graph-путь `forward_decode` → `gemv_scratch` (gguf_model.h:3120)
| Вес | Формат на диске | Путь сейчас | Оценка |
|-----|-----------------|-------------|--------|
| attn_q / attn_k | Q4_K | `q4km_persistent_gemv(_v2)` dp4a | ✅ оптимально |
| attn_v | F16 (часто) | `launch_fp16_gemv` наивный | ⚠ векторизовать half2 |
| attn_output | Q4_K | `q4km_persistent_gemv` | ✅ |
| ffn_gate / up | Q4_K | fused gate+up GEMV | ✅ |
| ffn_down | Q4_K / Q6_K | `q4km`/`q6k_gemv` | ✅ Q4_K; Q6_K скалярный (см §5) |
| lm_head | Q4_K (→FP16) / tied | `cublas_hgemv` (n=1) | ✅ bandwidth-ок |

### Prefill (M>1) — `forward` → `transformer_layer` → `matmul_q` (gguf_model.h:7088)
**Все** проекции: цикл `for m in [0,M): launch_gemv(x+m*K, y+m*N)`.
→ **0% Tensor Core utilization. M× redundant HBM-чтение весов.** ❌❌❌

Опциональный `use_fp16_weights_` decode-путь (gguf_model.h:2705+) гонит cuBLAS HGEMV на FP16-дубликаты весов — но это всё ещё n=1 GEMV, не GEMM, и стоит +5-6 GB VRAM. Для decode бесполезен (см §3); для prefill — **не подключён** (prefill идёт мимо, через `matmul_q`).

---

## 2. Decode (M=1): почему on-the-fly Q4_K оптимален — bandwidth-арифметика

Decode N=1 — чистый bandwidth-bound. Время ≈ (байты весов, прочитанные из HBM) / BW. FLOPs ничтожны (2·K на выход), TC бесполезны.

**qwen3:4b, ~28 слоёв, hidden H=2560, inter≈9728, GQA.** Веса на слой (in/out features) дают ~ **2.4 GB Q4_K total** (4.5 bit/вес).

Байты на один decode-проход (читаем КАЖДЫЙ вес ровно раз):

| Формат | bit/вес | Веса qwen3:4b | HBM/проход | t = bytes/2039GB/s | Потолок tok/s |
|--------|---------|---------------|------------|--------------------|---------------|
| **Q4_K** | 4.5 | ~4.0 B params | **~2.25 GB** | 1.10 ms | **~900** |
| Q8_0 | 8.5 | — | ~4.25 GB | 2.08 ms | ~480 |
| **F16** | 16 | — | **~8.0 GB** | 3.92 ms | ~255 |

**Вывод цифрами:** FP16-веса на decode читают ×3.6 больше HBM, чем Q4_K (8.0 vs 2.25 GB). Потолок падает с ~900 до ~255 tok/s. **FP16 на decode — категорически невыгоден.** Текущий выбор (on-the-fly Q4_K dp4a) — правильный. `use_fp16_weights_` для decode оставить `false`.

> Замечание о точности dp4a: квантование x в Q8_1 на лету — стандарт llama.cpp, расхождение vs FP16-аккумуляции < 0.1% на logit, подтверждено в проекте (bit-exact с llama.cpp). dp4a даёт INT8-dot за 1 инструкцию = меньше ALU-давления, что важно, т.к. на decode мы хотим максимум warps для latency-hiding HBM.

**Почему мы не на 900, а на 88 tok/s:** decode-потолок 900 — теоретический BW-предел. Реальный decode упирается в (а) launch-overhead множества мелких кернелов, (б) attention/softmax/RoPE/RMSNorm накладные, (в) недостаточную occupancy для насыщения 2 TB/s. Но **decode — не главная потеря.** Главная — prefill (§4).

---

## 3. FP16-веса vs Q4_K — окончательный вердикт по слоям

| Слой | Decode (M=1) | Prefill (M>1) |
|------|--------------|---------------|
| attn_q/k/o, ffn_* | **Q4_K on-the-fly dp4a** (НЕ fp16) | **dequant→FP16 + HGEMM (TC)** |
| attn_v (F16 на диске) | F16 half2-GEMV | F16 напрямую в HGEMM (TC) |
| ffn_down (Q6_K) | Q6_K dp4a (переписать, §5) | dequant→FP16 + HGEMM |
| lm_head | FP16 HGEMV (n=1, ок) | при необходимости — HGEMM по всем позициям |

Принцип: **decode = минимизируй байты (Q4_K)**, **prefill = максимизируй FLOPs/байт (FP16 в TC)**. Это два разных режима, и их нельзя обслуживать одним форматом.

---

## 4. Prefill (M>1): главный фикс — dequant→FP16 + cuBLAS HGEMM

### Почему сейчас провал
`matmul_q` на M>1 (gguf_model.h:7088) — это roofline-катастрофа:
- Compute prefill-GEMM: 2·M·N·K FLOPs. Для M=512, типичной проекции K=N=2560: 2·512·2560·2560 ≈ **6.7 GFLOP** на проекцию.
- На FP32-CUDA-cores (19.5 TFLOPS) это 0.34 ms; на **FP16 Tensor Cores (312 TFLOPS) — 21 µs (×16)**.
- Текущий цикл из M GEMV вообще не делает GEMM — он M раз перечитывает веса (compute intensity = M·N·K / (M·K + N·K) ≈ 1 при mem-bound GEMV). Получаем **bandwidth-bound prefill с ×M чтением весов** вместо compute-bound TC-GEMM.

### Целевой путь prefill (по слою)
1. **Один раз при load:** dequant Q4_K/Q5_K/Q6_K → FP16 `[N,K]` через `launch_dequant_q4k_to_fp16` (уже есть, CUDAQuantGemv.cu:1115) — но включать **только если хватает VRAM** (qwen3:4b: +~5-6 GB; на A100 80GB не проблема). F16-веса (attn_v) уже готовы.
2. **Prefill-проекция:** `cublasGemmEx(CUBLAS_OP_T, OP_N, N, M, K, FP16, FP16, COMPUTE_16F/32F, TENSOR_OP)` — это настоящий **HGEMM M×N×K на Tensor Cores**.
   - x промпта [M,K] → FP16 один раз на слой.
   - `CUBLAS_COMPUTE_32F` для аккумуляции (точность), вход FP16 → HMMA активируется автоматически на A100 при выровненных lda/ldb (кратность 8).
3. **Где включить в коде:** в `matmul_q`, ветка `else // M>1` (gguf_model.h:7088). Если `qw.fp16_data` готов (или dequant on-demand в scratch) → один `cublasGemmEx` вместо цикла. lm_head на prefill (gguf_model.h:2559) — туда же.

### Ожидаемый эффект
- Prefill-проекции: bandwidth-bound ×M-чтение → compute-bound HGEMM. **TTFT ×5-10** на промптах 128-2048 токенов.
- VRAM-бюджет: dequant-to-FP16 всех весов = +~5.6 GB (qwen3:4b). На A100 80GB ок. На consumer-GPU — dequant **в scratch потайлово** (tile [N,K_tile]→FP16, HGEMM, reuse буфера), чтобы не держать все FP16-веса.

### Альтернатива без полного FP16-дубля: dequant-в-scratch на проекцию
Для VRAM-ограниченных карт: на каждую prefill-проекцию dequant Q4_K→FP16 в общий scratch [N,K] (один раз, амортизируется по M токенам), затем HGEMM. При M≥16 цена dequant (1× чтение Q4_K) ничтожна против выигрыша TC. **Это и есть правильный on-the-fly-dequant→HMMA для prefill.**

---

## 5. Точечные дефекты форматных кернелов

### 5.1 Q6_K GEMV — скалярный, 1 lane = 1 позиция (CUDAQuantGemv.cu:881-958)
`q6k_gemv_kernel`: только `if (lane < 32)`, ветвление, нет dp4a, нет coalesced 4-byte. ffn_down часто Q6_K → это горячий путь. **Переписать под dp4a** аналогично Q4_K: ql/qh распаковать в int8 [-32..31], x→Q8_1, `__dp4a`. Ожидаемо ×2-3 на этих слоях decode.

### 5.2 Q8_1-квант x дублируется внутри каждого GEMV (CUDAQuantGemv.cu:189-209, 396-416)
На decode x одинаков для qkv (после attn_norm) и для gate/up (после ffn_norm). Сейчас каждый из 7 кернелов/layer квантует x в Q8_1 заново в свою smem. **Вынести `quantize_q8_1(x)` один раз** в начало attn-блока и ffn-блока, передавать готовый Q8_1-буфер (как делает `q4km_q8_gemv_kernel`, CUDAQuantGemv.cu:736, принимающий `block_q8_1* x_q8`). Экономия: 6 лишних quant-проходов/layer.

### 5.3 fp16_gemv наивен (CUDAQuantGemv.cu:1277-1304)
`__half2float(row[k])` поэлементно. Читать `half2`/`__ldg(const half2*)`, копить `__hfma2` → ×2 ld-эффективность. attn_v на decode.

### 5.4 Упаковка Q4_K под dp4a
Текущая распаковка `qs & 0x0F0F0F0F` и `(qs>>4)&0x0F0F0F0F` (v2-кернел, CUDAQuantGemv.cu:501-502) корректна и уже даёт 4 int4 в один `__dp4a`. Это правильно. Для дальнейшего — переупаковка весов при load в «dp4a-friendly» порядок (low/high nibble уже разнесены по словам) убрала бы маскирование в рантайме, но выигрыш мал (decode bandwidth-bound, ALU не узкое место). **Низкий приоритет.**

---

## 6. INT8 IMMA prefill (опционально, ещё ×2 поверх FP16)

A100 INT8 TC = 624 TOPS (×2 vs FP16 312 TFLOPS). Если веса Q4_K и активации квантовать в INT8 и гнать prefill через `cublasGemmEx(CUDA_R_8I, COMPUTE_32I, IMMA)`:
- Веса: dequant Q4_K→INT8 (scale на сублок) — почти бесплатно, Q4_K уже целочисленный.
- Активации: per-token INT8 (как Q8_1).
- GEMM IMMA даёт ×2 throughput prefill vs HGEMM при сохранении точности (W4A8 — стандарт TensorRT-LLM).

**Порядок внедрения:** сначала FP16-HGEMM (§4, низкий риск, cuBLAS делает всё), потом — при желании выжать TTFT — IMMA W8A8/W4A8. IMMA требует аккуратной раскладки и масштабов, поэтому фаза 2.

---

## 7. Сводная таблица «что менять»

| # | Изменение | Файл / место | Эффект | Приоритет |
|---|-----------|--------------|--------|-----------|
| 1 | Prefill M>1: цикл GEMV → cuBLAS **HGEMM** на FP16 (dequant в scratch или fp16_data) | gguf_model.h:7088-7096; lm_head 2559 | **TTFT ×5-10**, TC включены | 🔴 крит |
| 2 | Единый `quantize_q8_1(x)` на attn- и ffn-блок | gemv_scratch / forward_decode | −6 quant-проходов/layer | 🟠 высокий |
| 3 | Q6_K GEMV → dp4a (как Q4_K) | CUDAQuantGemv.cu:881 | ×2-3 на ffn_down Q6_K decode | 🟠 высокий |
| 4 | fp16_gemv → half2 + `__hfma2` | CUDAQuantGemv.cu:1277 | ×2 ld attn_v decode | 🟡 средний |
| 5 | Decode: подтвердить `use_fp16_weights_=false` (BW-проигрыш ×3.6) | gguf_model.h:924 | не регрессить decode | 🟡 guard |
| 6 | INT8 IMMA W4A8 prefill | новый путь | TTFT ещё ×2 | 🟢 фаза 2 |

---

## 8. Числа A100 (референс для решений)

| Метрика | A100 80GB SXM |
|---------|---------------|
| HBM2e bandwidth | 2039 GB/s |
| FP16/BF16 Tensor Core | 312 TFLOPS |
| TF32 Tensor Core | 156 TFLOPS |
| FP32 CUDA core | 19.5 TFLOPS |
| INT8 Tensor Core (IMMA) | 624 TOPS |
| INT8 dp4a (ALU) | ~78 TOPS |
| L2 cache | 40 MB |

**Roofline ridge point (FP16 TC):** 312e12 / 2039e9 ≈ **153 FLOP/byte**. GEMV (intensity ~2) — глубоко memory-bound → decode правильно остаётся на Q4_K dp4a. Prefill-GEMM с M≥16 (intensity ~M) пересекает ridge → обязан идти на Tensor Cores. **Сегодня prefill сидит слева от ridge из-за ×M-GEMV — это и есть незакрытая дыра.**

---

*Код в этой спецификации не правился — только анализ и план. Реализация п.1 (HGEMM prefill) — следующий шаг.*
