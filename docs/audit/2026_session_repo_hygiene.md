# Аудит чистоты и корректности репозитория — сессия 2026

Статический анализ (Read/Grep/Glob), без сборки/запуска. Все находки перепроверены
чтением исходников; ложные срабатывания агентов отсеяны и помечены ниже.

---

## HIGH — чинить в первую очередь

### H1. `gemv_scratch` молча оставляет `y` нетронутым для необработанных quant-типов
`torch/io/gguf_model.h:3124-3143`
Тот же класс, что баг «F16 attn_v → V=0» из журнала. Логика:
```cpp
if (use_quant_gemv_ && qw.valid) {
    if (q4k) ... else if (q6k) ... else if (q5k) ... else if (f16) ...
    // НЕТ else — если qw.valid, но тип Q8_0/Q5_0/иной, НИ ОДНА ветка не сработает
} else if (float_w.defined()) { ... }  // сюда НЕ попадём, т.к. вошли в первый if
```
Если вес валиден, но тип не {Q4_K,Q6_K,Q5_K,F16}, выход `y` не заполняется →
нули/мусор, БЕЗ ошибки. Латентно: срабатывает только если модель протащит
Q8_0/Q5_0 через этот GPU-путь.
**Фикс:** добавить `else { fallback на float_w или CPU dequant + abort/лог }`,
либо `assert(dispatched)`. То же проверить в upload_quant (GPU/CPU/mmap загрузчиках):
там ветки заканчиваются `else return;` и оставляют `qw.valid=false` тихо — как
минимум нужен лог/ошибка вместо молчаливого пропуска проекции.

### H2. Гонка выбора модели: `ensure_model_loaded` вне `generate_mutex_`
`promeserve/api_handlers.h:766-778` (generate) и `873-883` (chat)
`ensure_model_loaded(model_name)` вызывается ДО захвата `generate_mutex_`.
Два конкурентных запроса на РАЗНЫЕ модели: оба проходят ensure, затем один
свопает модель под другим. UAF здесь НЕТ (get_loaded_model возвращает копию
`shared_ptr`, объект жив до конца генерации — старый UAF действительно закрыт,
см. model_manager.h:188-196). Но это логическая гонка: генерация может пойти
на только что выгруженной/подменённой модели.
**Фикс:** внести `ensure_model_loaded()` ПОД `generate_mutex_` (взять lock до ensure),
либо привязать выбранный `shared_ptr` к запросу атомарно (load+get под одним lock).

---

## MED

### M1. `_build_cpu_gguf_win.bat` — опечатка пути ломает сборку у всех
`scripts/_build_cpu_gguf_win.bat:5` → `cd /d C:\Users\USER\Desktop\prometorch`
Каталог называется `promethorch` (с `h`), плюс это хардкод домашнего пути `paper`.
Скрипт трекается git → у любого, кто им воспользуется, `cd` упадёт и сборка не
стартует.
**Фикс:** `cd /d %~dp0..` (относительно расположения bat) вместо абсолютного пути.

### M2. Attention scale не учитывает YaRN `attn_factor` (нужно проверить на реальной модели)
`torch/io/gguf_model.h:~2822` — `scale = 1/sqrt(head_dim)` без домножения на
`rope_attn_factor`. Для Phi-3/Gemma3 с YaRN mscale это даёт неверные attention
scores. Требует проверки: возможно фактор применяется в другом месте RoPE-пути.
**Фикс:** если не применяется нигде — `scale *= config.rope_attn_factor`.

### M3. LongRoPE factors на GPU decode-пути (потенциальный prefill/decode divergence)
`torch/io/gguf_model.h`: prefill (CPU rope_apply_fused) использует
`config.rope_factors_for(pos)`, GPU decode fused-kernel — проверить, передаются ли
туда rope_factors. Если нет — Phi-3 LongRoPE decode разойдётся с prefill. Классика
багов проекта (prefill/decode дублируют логику). Требует адресной проверки сигнатуры
`launch_fused_qknorm_rope_kvwrite`.

### M4. `http_server.h` — `queued_` и `queue_.size()` могут расходиться
Счётчик `queued_` декрементируется в worker, а accept-loop читает `queue_.size()`;
без общей синхронизации метрика backpressure может врать (не корректность генерации,
а качество отбоя при перегрузке).

---

## LOW / гигиена

### L1. 249 scratch-`.bat` в корне репозитория
Все `*.bat`/`*.ps1` в корне игнорируются `.gitignore` (`*.bat` + `!scripts/*.bat`),
поэтому В GIT НЕ ПОПАДАЮТ — репо чистое. Но 249 файлов физически засоряют корень и
путают (`build_*`, `rebuild_*`, `run_*`, `test_*`, `do_build2`, `temp_*`).
**Рекомендация:** удалить/перенести в `scripts/dev_archive/` локально. На сборку у
людей не влияют (не трекаются).

### L2. Незакоммиченные integration-тесты
`python/tests/test_{bindings_new,no_grad,pytorch_io,transformers_compat}.py` — untracked.
.gitignore имеет исключение `!python/tests/test_*.py` (должны быть tracked), но эти
4 файла ещё не добавлены. Либо закоммитить, либо удалить, чтобы не потерялись.
Корневые `test_*.py` (test_4layer_mlp и т.п.) — корректно игнорируются `/test_*.py`.

### L3. `TODO: FP32→FP16 lm_head` не реализован — выделенный буфер освобождается впустую
`torch/io/gguf_model.h:1349` — оптимизация cuBLAS HGEMV для lm_head не работает
на FP32-весах (буфер аллоцируется и сразу освобождается). Только потеря скорости.

### L4. ~30 `getenv("PT_*")` тогглов разбросаны по gguf_model.h
Большинство закэшированы в `static const`, но часть (напр. `PT_DUMP_TOKENS`)
дёргает getenv в decode-пути. Диагностический мусор + мелкая потеря скорости.

### L5. `.gitignore` — состояние ХОРОШЕЕ
Веса/чекпоинты (`*.gguf,*.bin,*.ckpt,*.safetensors,*.pth,pir_ckpt/,checkpoints/`),
build-дирректории, логи — все игнорируются. Проверено: НИ ОДИН большой бинарь и НИ
один build-артефакт не трекается git. `docs/audit/*.md` — trackable (годится для
этого отчёта).

---

## Ложные срабатывания агентов (НЕ баги — задокументировано, чтобы не ловить повторно)

- **CUDABlas.cu:465** — «перепутаны trans_a/trans_b». НЕ баг: cuBLAS column-major,
  своп op_a/op_b вместе со свопом операндов A/B в вызове (B первым, A вторым) —
  корректная идиома для получения row-major `C = A@B`. Документировано в комментарии
  строки 468.
- **api_handlers.h:897 «BUG-12 backslash comments»** — артефакт вывода grep; в файле
  корректные `//`-комментарии.
- **UAF в get_loaded_model** — уже ИСПРАВЛЕН: возвращает копию `shared_ptr`
  (model_manager.h:188-196), объект переживает своп в другом потоке.

---

## Приоритет действий
1. **H1** — добавить fallback/assert в `gemv_scratch` и загрузчики quant (латентный
   молчаливый-нуль баг, семейство уже кусало проект дважды).
2. **H2** — внести `ensure_model_loaded` под `generate_mutex_`.
3. **M1** — починить опечатку пути в `_build_cpu_gguf_win.bat` (ломает чужую сборку).
4. **M2/M3** — проверить YaRN scale и LongRoPE factors на decode (Phi-3/Gemma3).
5. Гигиена: снести scratch-`.bat` из корня, закоммитить/удалить 4 python-теста.
