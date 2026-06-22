# Вердикт: структура decode-step, свежий взгляд (Dana Reyes, Groq)

**Объект:** `torch/io/gguf_model.h` `forward_decode` (2575+) / `generate` (4794+), `aten/.../FlashDecoding.cu`.
**Замер-якорь:** qwen3:4b A100 — 90 tok/s = 11.4 мс/ток = 8.9 мс GPU (`cudaStreamSynchronize`) + 2.5 мс CPU. `cudaGraphLaunch`=55 мкс — граф здоров.

---

## Главный тезис: GEMV трогать НЕ надо

Декод запечён в один CUDA Graph (~363 ядра, 1 launch). dp4a/occupancy/argmax-on-device дали +1/0/+4% — это шум. Значит 8.9 мс GPU — это **полезная память кернелов**, а не launch-overhead и не плохой GEMV. Подтверждаю код: bandwidth-bound N=1, Q4_K-веса уже минимальный трафик (`feedback_fp16_wont_speedup_decode`). **Дальше копать GEMV = ходить кругами.** Дефицит — в (а) лишнем KV-bandwidth в графе и (б) sampling-пути ВНЕ графа.

## 1. Пузыри/megakernel — НЕ приоритет
Граф убивает launch-overhead уже сейчас. Kernel-to-kernel gap в графе на A100 ≈ 0.3–1 мкс, не 2–5; 360×~0.5 ≈ 0.2 мс — потолок выигрыша от megakernel. Per-layer megakernel (10→1) — это месяцы риска ради ~0.2 мс и поломки `PT_NO_GRAPH`-bypass. **Отклоняю.** Уже сделанные фьюзы (norm+QKV, norm+gate+up, rope+kv-write) — правильный уровень гранулярности; дальше fuse только copy+accumulate (дёшево).

## 2. Sampling on-device — КРУПНЫЙ выигрыш в serving (ТОП-1 по ROI)
Код 4905-4995: greedy-GPU делает argmax на `nullptr`-стриме (не `decode_stream_`) + блокирующий `cudaMemcpy` D2H = 2-й full-pipeline stall/ток. А serving (rep_penalty=1.05 всегда) уходит в `get_row` → **608 КБ D2H всего вектора** + CPU-цикл по `generated`. Это 2.5 мс CPU целиком. Device-side argmax+rep-penalty (кольцевой `d_generated_`, 4-байтный D2H next_token, async, совмещённый с единственным sync) убирает почти весь CPU-хвост. **Выигрыш ~1.5–2.0 мс/ток в реальном serving.**

## 3. FP16-KV в графе — растёт с контекстом
Сейчас граф читает FP32 KV (`launch_flash_decode_graph`); FP16-версия отключена («baked offsets»), хотя `d_past_len_` уже доказал device-pointer-приём. qwen3:4b: 8 kv-h×128 = 36864 элем/слой/ток. На 2k контексте чтение K+V ≈ 580 МБ FP32 → 290 МБ FP16. На A100 ~1.5 ТБ/с это **~0.2 мс @2k, ~0.4 мс @4k, ~0.8 мс @8k.** На коротком — мелочь, на serving-контексте — ощутимо и бесплатно по numerics при проверке на 110+ ток.

## 4. Что Groq сделала бы ПЕРВЫМ (min риск / max выигрыш)
Device-sampler в графе (п.2) + один sync на токон + 4-байтный async D2H. Это срезает 2.5 мс CPU и 2-й stall, не трогая ни одного GEMV, ляжет в существующий device-pointer-паттерн. Затем FP16-KV в графе (п.3), затем copy+accumulate→fused residual-GEMV (−72 ядра, ~0.3 мс).

| Действие | мс/ток | риск |
|---|---|---|
| Device argmax+rep-penalty в графе, 4-байт async D2H, 1 sync | 1.5–2.0 | низкий |
| FP16-KV внутри графа (device-ptr длина) | 0.2–0.8 (растёт с ctx) | средний |
| copy+accumulate → fused residual-GEMV | 0.3–0.5 | низкий |
| megakernel per-layer | ~0.2 | высокий — **НЕ делать** |
| возня с GEMV-ядрами | ~0 | — **НЕ делать** |

**Итог:** структура decode-step, а НЕ GEMV. Реалистично 11.4 → ~8 мс/ток (~125 tok/s) от sampler+sync+residual без переписывания ядер; +FP16-KV даёт ещё запас на длинном контексте. Совпадает с вердиктом Nair (03); расхождение лишь в акцентах — я ставлю device-sampler выше FP16-KV, т.к. он бьёт по 2.5 мс CPU, которые видны на коротком контексте уже сейчас.
