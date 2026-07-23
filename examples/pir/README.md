# PIR — тренировка своей LM на Эльбрусе (BPE + Local SGD)

Полный цикл: корпус → BPE-токенизатор → распределённая тренировка на
4 NUMA-узлах E8C2 → генерация текста. Всё, что здесь описано, реально
запускалось и работает (2026-07: русская LM 13.5M параметров, BPE 16k,
~4450 tok/s суммарно на E8C2 32 ядра).

Архитектура блока и математика: [docs/PIR_ARCHITECTURE.md](../../docs/PIR_ARCHITECTURE.md).
Сборка на Эльбрусе: [docs/BUILD_ELBRUS.md](../../docs/BUILD_ELBRUS.md).

## 0. Сборка

```bash
mkdir -p build_elbrus && cd build_elbrus
cmake .. -DCMAKE_BUILD_TYPE=Release
make train_pir_elbrus -j8
```

**Обязательная проверка после сборки:**

```bash
ldd examples/pir/train_pir_elbrus | grep eml
#  → libeml_mt.so = OK (многопоточный GEMM, 120-240 GFLOPS)
#  → пусто        = сборка тихо упала в сериальный TudaBLAS (~5 GFLOPS!)
```

Если пусто — CMake-проба EML могла закэшировать давний фейл. Фикс:

```bash
cmake -U '_PT_EML*' -DPT_EML_RUNTIME_OK=ON . && make train_pir_elbrus
```

Симптомы сериального фолбэка: время шага не зависит от `OMP_NUM_THREADS`,
`mpstat -P ALL` показывает 1 занятое ядро на процесс.

## 1. Данные: BPE-токенизатор

Нужен сырой текстовый корпус UTF-8 (мы использовали дамп русской
Википедии ~2 ГБ). Словарь 16k — проверенный баланс для русского.

```bash
python3 prepare_bpe.py --input data/corpus.txt \
    --out-prefix data/ru_bpe_16k --vocab-size 16000
# → data/ru_bpe_16k.model  (нужен для декода в pir_infer.py)
# → data/ru_bpe_16k.tokens (uint32, вход тренировки)
```

`train_pir_elbrus` авто-детектит формат по расширению: `.tokens` =
uint32 BPE, любой другой файл = байтовый char-level. **`--vocab_size`
передавать явно** — из файла он не читается.

## 2. Smoke-тест (1 процесс, 2 минуты)

```bash
PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 \
./build_elbrus/examples/pir/train_pir_elbrus \
    --fused --full --rank -1 \
    --vocab_size 16000 --data data/ru_bpe_16k.tokens \
    --n_embd 256 --n_layers 4 --block_size 256 --batch_size 2 \
    --max_steps 3 --log_interval 1
```

Ожидаемо: `loss ≈ ln(vocab)` на первом шаге (9.68 для 16k) и снижение.
Если segfault или loss = ln(256)=5.55 — забыл `--vocab_size`.

## 3. Распределённая тренировка (Local SGD, 4 NUMA-узла)

```bash
loginctl enable-linger $USER          # ОБЯЗАТЕЛЬНО (см. ниже)
DATA=data/ru_bpe_16k.tokens VOCAB=16000 ./examples/pir/train_local_sgd.sh
```

Что делает скрипт: 1 процесс на NUMA-узел (`numactl --cpunodebind=N
--preferred=N`, 8 OMP-потоков), каждый тренируется на своих случайных
батчах, веса усредняются через `/dev/shm` каждые `GRAD_ACCUM` шагов.
Это **Local SGD** (weight averaging), не DDP: градиенты не гоняются
через межчиповый интерконнект вообще.

Почему так: кросс-NUMA доступ на E8C2 роняет GEMM с 463 до 330 GFLOPS;
4 node-local процесса дают линейное масштабирование (проверено: 1840
GFLOPS = 92% пика на больших GEMM).

### Устойчивость

- `loginctl enable-linger $USER` — без этого systemd убивает процессы
  при SSH-disconnect. **После reboot сервера выполнять заново.**
- Reboot переживается чекпоинтами: `RESUME=1 ./train_local_sgd.sh`
  подхватит последний `pir_fused_step_N.bin` и продолжит с него
  (`--start_step` смещает LR-график; первые 200 шагов после резюма LR
  разгоняется линейно — Adam-моменты в чекпоинт не пишутся, без
  warmup был бы спайк loss).

## 4. Генерация / оценка

Встроенная генерация тренера — char-level; для BPE-моделей декодер —
`pir_infer.py` (numpy + sentencepiece, C++ не нужен):

```bash
python3 pir_infer.py \
    --ckpt checkpoints_pir/pir_fused_step_5000.bin \
    --spm data/ru_bpe_16k.model --vocab_size 16000 \
    --n_embd 256 --n_layers 4 --block_size 256 \
    --prompt "Москва является " --max_tokens 60 --temp 0.7 --top_k 40
```

Сверка корректности C++ vs numpy forward: `--val_tokens
data/ru_bpe_16k.tokens` печатает cross-entropy на срезе — должна
совпадать с тренировочным loss (±0.1).

## Проверенная конфигурация (2026-07)

| Параметр | Значение |
|----------|----------|
| Модель | D=256, L=4, NP=4, block 256 → 13.57M параметров |
| Данные | русская Википедия, BPE 16k, 16.7M токенов |
| Запуск | 4 процесса × 8 потоков, batch 2/процесс, grad_accum 10 |
| Скорость | ~0.46 с/шаг, ~1100 tok/s/процесс, **~4450 tok/s суммарно** |
| Loss | 9.68 (init) → 5.89 @ 5000 шагов |

## Troubleshooting

| Симптом | Причина / фикс |
|---------|----------------|
| Медленно, 1 ядро на процесс | EML не слинкован → секция 0 |
| Segfault при vocab > 2048 | старая версия TudaBLAS (heap overflow в PackBuffers, исправлено `78b0613`) — обнови репо |
| loss = 5.55 вместо 9.68 | не передан `--vocab_size` |
| Процессы умерли после disconnect | `loginctl enable-linger` не выполнен |
| Спайк loss после resume | старая версия без resume-warmup (`0d7b89b`) — обнови |
| Мусор в генерации BPE-модели | использована встроенная генерация (char-level) вместо `pir_infer.py` |
