# pre-SINQ GGUF в PromeTorch

## Что это

**SINQ** (Sinkhorn-Normalized Quantization) — метод квантизации от Huawei
Computing Systems Lab (arXiv:2509.22944, ICML 2026, **Apache-2.0**).
Режет память LLM на 60-70% при сохранении качества, calibration-free.

Две ключевые идеи:
1. **Dual-Axis Scaling** — отдельные масштабы по строкам И столбцам матрицы
   весов (а не один общий). Outlier'ы (крупные веса) больше не ломают шкалу.
2. **Sinkhorn-Knopp нормализация** — итеративно перебалансирует variance
   строк и столбцов до минимального «дисбаланса» (отношение max/min std).
   Сбалансированная матрица квантуется равномернее → меньше ошибка.

## Почему это релевантно нам

PromeTorch — quantization-heavy inference (Q4_K/Q6_K/Q8 GGUF на CPU/Эльбрусе).
**pre-SINQ** (`presinq_gguf.py` в репо SINQ) применяет Sinkhorn-нормализацию
к весам HF-модели, затем стандартным `llama.cpp convert_hf_to_gguf` +
`llama-quantize` производит **обычный GGUF**. Выход — 100% стандартный
GGUF-файл: **наш загрузчик читает его без единой строки изменений.**

Профит: на той же модели, том же quant-типе (Q4_K), той же скорости
(тот же compute-путь) — **лучше качество** (меньше ошибка квантования).
Это особенно ценно для русского/математики, где мы ловили деградацию
квантованного qwen3-4b.

## Как это работает в нашем стеке

```
HF weights → [SINQ Sinkhorn-нормализация] → llama.cpp quantize → GGUF (Q4_K)
                                                                     │
                                          PromeTorch GGUF loader ────┘  (без изменений)
```

Никакого кода SINQ в PromeTorch НЕ вшито — мы только потребляем результат
(стандартный GGUF). Атрибуция — в `NOTICE`.

## Готовые модели (huawei-csl, Apache-2.0)

| Модель | Quant | Совпадает с нашим baseline |
|--------|-------|----------------------------|
| `huawei-csl/Qwen3-4B-PreSINQ-GGUF` | Q4_K_S / Q3_K_S | base = Qwen3-4B (наш TP-4 10.9 tok/s) |
| `huawei-csl/Qwen3-0.6B-PreSINQ-GGUF` | Q4_K_S / Q3_K_S | — |
| `huawei-csl/Qwen3-1.7B-PreSINQ-GGUF` | Q4_K_S / Q3_K_S | — |
| `huawei-csl/Qwen3-8B-PreSINQ-GGUF` | Q4_K_S / Q3_K_S | — |

> ⚠ pre-SINQ репо даёт **Q4_K_S** (не Q4_K_M). S vs M: M использует Q6_K
> для части attn_v/ffn_down. Для чистого изолирования SINQ-эффекта нужен
> и обычный Q4_K_S того же qwen3-4b. M-vs-S A/B показывает направление.

## A/B проверка на Эльбрусе

```bash
# 1. Скачать pre-SINQ на 8C2:
cd ~/gguf_models
curl -sL -o qwen3-4b-presinq-Q4_K_S.gguf \
  https://huggingface.co/huawei-csl/Qwen3-4B-PreSINQ-GGUF/resolve/main/Qwen3-4B-presinq-Q4_K_S.gguf

# 2. A/B (baseline Q4_K_M vs pre-SINQ Q4_K_S), TP-4 на 8C2:
cd ~/prometorch
bash scripts/run_presinq_ab.sh "Объясни что такое тензор."
```

Проверяем 3 вещи:
1. **Загружается** ли pre-SINQ GGUF (должен — стандартный формат).
2. **Тот же tok/s** (должен — тот же Q4_K compute).
3. **Лучше качество** — связность русского/математики (ручная оценка A/B).

Результаты A/B — в `docs/SINQ_INTEGRATION.md` § Результаты (заполняется
после прогона на 8C2 lemur-1).

## Результаты (8C2 lemur-1)

_(заполняется по факту прогона `run_presinq_ab.sh`)_

| Метрика | baseline Q4_K_M | pre-SINQ Q4_K_S |
|---------|-----------------|------------------|
| Загрузка | ✅ | _TBD_ |
| TP-4 tok/s | 10.9 | _TBD_ |
| Русский/связность | (известная деградация) | _TBD_ |

## Дальше (опционально, research)

Портировать сам Sinkhorn-алгоритм (`sinq/sinkhorn.py`, ~70 строк) в наш
offline-квантизатор `.pt8` (Q8 SoA4) — тогда мы сможем сами производить
SINQ-conditioned веса для любой модели, не только готовые от Huawei.
Это требует layer-folding (инверсный масштаб поглощается соседними слоями)
— нетривиально, отдельный этап.
