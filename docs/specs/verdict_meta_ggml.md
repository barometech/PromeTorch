# Вердикт ggml-мейнтейнера: что у llama.cpp даёт полосу, которой нет у PromeTorch

Контекст: qwen3:4b decode 90 у нас vs 188 Ollama; 270 ГБ/с = 17% HBM vs Ollama 32%.
Читал: `CUDAQuantGemv.cu` (v2 dp4a, fused gate+up, fused rmsnorm+gate+up, quantize_q8_1,
q6k_gemv), decode-цикл `gguf_model.h` `forward_decode()`, спеки 01/03/05.

## Диагноз: главная потеря — НЕ микрооптимизация ядра, а структура пайплайна

1. **Production-путь вообще не dp4a — он FP32-dequant.** Реально вызывается
   `q4km_fused_rmsnorm_gate_up_kernel` (CUDAQuantGemv.cu:1846): x в smem лежит **FP32**,
   веса распаковываются 4 бита→float и идут 8 скалярных FMA/блок (стр. 1941-1948).
   dp4a-ядро `q4km_persistent_gemv_v2_kernel` (375-536) существует, но `use_llama_gemv_=false`
   по умолчанию (gguf_model.h:3127) — **лучший кернел выключен.** Это объясняет, почему твой
   замер «dp4a v2 не быстрее fused (+1%)» — ты сравнивал v2 с FP32-fused, но fused делает
   ту же работу в FP32; узкое место не в этом, а в lifecycle (п.2) и в раскладке (п.3).

2. **quantize-x НЕ один раз — мы его не делаем вовсе на горячем пути.** В llama.cpp
   `quantize_q8_1` гонится по normed-x ОДИН раз на слой, а MMVQ только читает int8-y. У нас
   `quantize_q8_1_kernel` (692) написан, но fused-путь его не зовёт — он держит x как FP32 и
   «переквантует» смыслово на каждом обращении к весу. Где это бьёт: gate и up читают ОДИН и
   тот же normed-x, и даже в dp4a-варианте каждое ядро квантует x в smem заново (v2 стр. 396-417).
   Лечится отдельным `quantize_q8_1` в буфер один раз/слой → gate/up/down читают готовый Q8_1.
   Это **lifecycle-фикс, не kernel-фикс** — ровно как ты сказал.

3. **Раскладка mat-vec: warp-per-row NROWS=2 vs block-per-(tile) у ggml.** У нас 1 warp =
   1 строка (v2 — 2 строки). x в smem переиспользуется внутри блока, но bytes-in-flight по
   весам = строго N·rowbytes без тайлинга по нескольким строкам на варп; ggml MMVQ держит
   несколько строк на блок и больше Q8_1-y переиспользования на загруженный кусок весов →
   плотнее HBM-burst. Это даёт часть из 17%→32%, но это **микрооптимизация раскладки**, не
   архитектура (порядок ×1.2-1.4, не ×2).

4. **Prefill (M>1) = M отдельных GEMV — TTFT-катастрофа, ПОДТВЕРЖДАЮ.** spec05 §4: ветка
   M>1 в `matmul_q` (gguf_model.h:7088) гонит M раз GEMV → веса перечитываются ×M, intensity≈1,
   bandwidth-bound слева от roofline-ridge (153 FLOP/byte). ggml на M>1 идёт MMQ (dequant→Q8 +
   mma/Tensor Cores). Это **чисто архитектурная дыра**, самый крупный единичный выигрыш.

5. **Q6_K скалярный — ПОДТВЕРЖДАЮ.** `q6k_gemv_kernel` (881-958): ql/qh побайтно, x скаляром
   из smem, нет float4/dp4a, мёртвая `if(lane<32)`. В Q4_K_M миксе attn_v/ffn_down каждый ~3-й
   слой — Q6_K, плюс down 14B/27B. ggml имеет `vec_dot_q6_K_q8_1` на dp4a.

## Что переносить, и сколько это даёт (честно)

- **(A) Отдельный quantize-x-once/слой + перевести fused-путь на dp4a** (включить v2-семейство,
  читать готовый Q8_1): ×1.3-1.6 decode. Это закрывает «FP32-dequant вместо dp4a» + двойную
  квантацию gate/up. ГЛАВНЫЙ decode-фикс.
- **(B) MMQ для prefill** (dequant→FP16/Q8 + cublasGemmEx HMMA, spec05 §4): TTFT ×5-10.
  Архитектура, не ядро — самый большой эффект на сквозную скорость на длинных промптах.
- **(C) Q6_K на dp4a** (`vec_dot_q6_K_q8_1`, vectorized ql/qh): +полоса на down/Q4_K_M/14B+.

Микрооптимизация (раскладка block-per-tile, occupancy) даёт остаток ×1.1-1.3 ПОСЛЕ A.
Твои провалы (расфьюз gate+up −10%, occupancy grid 0) логичны: ты крутил kernel-микро, а
потолок держат lifecycle (двойная квантация, FP32 вместо int8) и пайплайн (M-GEMV prefill).
Делай A → B → C, не наоборот. Код не правил.
