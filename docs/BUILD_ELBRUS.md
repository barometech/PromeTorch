# Сборка и запуск PromeTorch на Эльбрусе E8C2

Полный практический гайд: от `git clone` на голом сервере до `tok/s`-цифр,
матчащих `BENCH_ELBRUS.md`. Все команды реальные, скрипты существуют в репо,
env-переменные проверены на 4-NUMA E8C2.

## Содержание

- [1. Целевое железо и ОС](#1-целевое-железо-и-ос)
- [2. Подключение к серверу](#2-подключение-к-серверу)
- [3. Сборка](#3-сборка)
- [4. Запуск GGUF inference](#4-запуск-gguf-inference)
- [5. Бенчмарки и сравнение с llama.cpp](#5-бенчмарки-и-сравнение-с-llama.cpp)
- [6. Поддерживаемые модели](#6-поддерживаемые-модели)
- [7. Подводные камни / Troubleshooting](#7-подводные-камни--troubleshooting)
- [8. Известные ограничения](#8-известные-ограничения)
- [9. PIR 250M training](#9-pir-250m-training)
- [10. Ссылки](#10-ссылки)

---

## 1. Целевое железо и ОС

| Параметр | Значение |
|----------|----------|
| Процессор | Эльбрус-8СВ (E8C2), VLIW v5 |
| Чипов на сервере | 4 |
| Ядер на чип | 8 |
| Ядер всего | **32** |
| Частота | 1500 МГц |
| FP32 пик на чип | 576 GFLOPS (6 ALU × 4 FP32 × 2 FMA × 1.5 GHz) |
| FP32 пик на сервер | **2304 GFLOPS** |
| RAM | 125 ГБ DDR4 ECC, 4 NUMA-нода |
| Bandwidth | 4 × 68.3 ГБ/с (одна нода) |
| L1d / L2 / L3 | 64 КБ / 512 КБ / 16 МБ |

ОС и тулчейн:

- **ОС:** Эльбрус Linux 6.1.0 (ALT) или Astra Linux SE for Elbrus
- **Компилятор:** LCC 1.29+ (GCC 11.3 frontend-совместимый)
- **CMake:** 3.15+ (на тестовом сервере 3.28)
- **Python:** 3.11
- **Зависимости:** EML (Elbrus Math Library, BLAS/LAPACK от МЦСТ), libnuma, OpenMP

EML и libnuma — обязательны. CMake определяет их автоматически и активирует
`PT_USE_EML_BLAS` + `PT_USE_NUMA`. Без EML PromeTorch свалится на TUDA 6×6
микроядро (рабочее, но в ~30× медленнее на GEMM).

### Установка зависимостей

**Альт Линукс под Эльбрус** (проверено 2026-05-04):

```bash
# Под root или через sudo
apt-get update
apt-get install -y \
    eml-devel \
    libomp11-devel \
    libnuma-devel \
    cmake \
    git
```

Если `eml-devel` не находится — проверь подключённые репозитории Альт.
EML устанавливается в `/opt/mcst/eml/` (заголовки и `libeml.so`). LCC
(`lcc`/`l++`) обычно идёт уже предустановленным с системой.

**Эльбрус Linux от МЦСТ** (стандартная сборка):

```bash
# Обычно EML и libnuma уже в составе системы.
# Дополнительно:
apt-get install -y libomp-devel cmake git
```

**Astra Linux SE for Elbrus**:

```bash
# Аналогично Альт — eml-devel + libomp-devel + libnuma-devel
apt-get install -y eml-devel libomp-devel libnuma-devel cmake git
```

Если CMake при конфигурации пишет `EML BLAS not found` или
`OpenMP_CXX: NOT FOUND` — пакеты выше не установлены или установлены
не туда. Проверь `find /opt/mcst -name "cblas.h"` (для EML) и
`find / -name "omp.h" 2>/dev/null` (для OpenMP).

---

## 2. Подключение к серверу

```bash
plink.exe -P <port> -i <ssh-key>.ppk \
  -hostkey "<hostkey>" \
  <user>@<elbrus-host>
```

Из Linux/Mac:

```bash
ssh -p <port> -i <ssh-key> <user>@<elbrus-host>
```

### tmux / screen для долгих задач

Сборка ~15-25 мин, бенчмарки до 30 мин на модель. SSH-сессия не выживет.

```bash
tmux new -s build       # сборка
tmux new -s bench       # бенчмарки
tmux attach -t build    # вернуться
```

### loginctl enable-linger — ОБЯЗАТЕЛЬНО

`systemd-logind` по умолчанию убивает все процессы пользователя при
SSH disconnect. Это касается и tmux — без linger'а сессия умрёт.

```bash
loginctl enable-linger $USER
```

Делать **после каждого reboot сервера**. Скрипты `run_tp_elbrus.sh` и
`run_1proc_elbrus.sh` пытаются вызвать его сами, но без sudo это no-op
если не было выполнено вручную хотя бы раз.

---

## 3. Сборка

### Один-командный путь

```bash
cd ~/promethorch
bash scripts/build-elbrus.sh
```

Что делает скрипт:

1. Проверяет наличие `lcc` и `cmake` в PATH.
2. Создаёт `build_elbrus/`.
3. Конфигурит CMake с toolchain'ом `cmake/toolchains/e2k-elbrus.cmake`:
   - `CMAKE_TOOLCHAIN_FILE=cmake/toolchains/e2k-elbrus.cmake`
   - `CMAKE_BUILD_TYPE=Release`
   - `PT_USE_TUDA=ON`
   - `PT_BUILD_TESTS=ON`
   - `PT_BUILD_SHARED_LIBS=ON`
4. Запускает сборку через `cmake --build . -j$(nproc)`.

### Что включается автоматически в CMake (для E2K)

CMake при `CMAKE_SYSTEM_PROCESSOR=e2k` (выставляется toolchain'ом) добавляет:

- `-DTUDA_E2K` — активация TUDA E2K-микроядра 6×6 (36 FMA-аккумуляторов)
- `-DPT_USE_EML_BLAS` — диспатч `cblas_sgemm` в EML
- `-DPT_USE_NUMA` — NUMA-aware tiled GEMM (5.7× ускорение от 324 до 1840 GFLOPS)
- `-O3 -ffast -faligned -fprefetch -fcache-opt -mtune=elbrus-8c2 -frestrict-all -fswp-maxopers=800`
- `-fopenmp` (32-ядерный параллелизм критичен)

Флаги `-ffast` и `-faligned` обязательны для активации APB hardware prefetch на
ISA V1-V5 — даёт ×1.5 на streaming Q4_K reads.

### Output

```
build_elbrus/
  examples/gguf/test_gguf_inference     # основной inference binary
  examples/mnist/train_mnist_mlp        # MNIST training (15.2с/epoch на 32c)
  examples/pir/train_pir_elbrus         # PIR 342M training
  test/cpp/                              # 505/507 тестов проходят
```

Время сборки на 32 ядрах: ~15-25 мин (Release с -O3 + LCC software pipelining).

### Ручная сборка (без скрипта)

```bash
mkdir build_elbrus && cd build_elbrus
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/e2k-elbrus.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPT_USE_TUDA=ON \
    -DPT_USE_CUDA=OFF \
    -DPT_USE_AVX=OFF -DPT_USE_AVX2=OFF \
    -DPT_BUILD_TESTS=ON
cmake --build . -j$(nproc)
```

### Только один target (быстро)

```bash
cmake --build build_elbrus --target test_gguf_inference -j 16
```

---

## 4. Запуск GGUF inference

Бинарник: `./build_elbrus/examples/gguf/test_gguf_inference`. Принимает GGUF Q4_K_M
файлы. Два режима запуска через готовые скрипты.

### 4.1. Single-process (для больших моделей где TP-4 OOM)

Использовать для qwen3-8B/14B, qwen2.5-7B, llama3-8B, deepseek-coder-7B —
TP-4 OOM-ит на ~32 ГБ per-rank.

```bash
bash scripts/run_1proc_elbrus.sh --greedy "Write a haiku about AI"
```

Что внутри:

```bash
OMP_NUM_THREADS=30 \
numactl --interleave=all \
    ./build_elbrus/examples/gguf/test_gguf_inference \
    "$MODEL" --max-tokens 100 --greedy "$PROMPT"
```

Ключевые env'ы:

- `OMP_NUM_THREADS=30` — 24-30 оптимально, 32 регрессирует на 8% (см. BENCH_ELBRUS.md)
- `numactl --interleave=all` — round-robin раскладка страниц по 4 DDR-контроллерам
- **НЕ ставить `PT_NUMA_REPLICATE=1`** в 1-proc — увеличивает рабочее множество в 4×,
  при этом BW в 1-proc уже сбалансирован interleave'ом
- **НЕ ставить `OMP_PLACES`/`OMP_PROC_BIND`** — горячий GEMV использует
  `c10::ThreadPool`, а не OpenMP, эти переменные не доходят до worker'ов

Указать другую модель:

```bash
PT_MODEL=$HOME/gguf_models/qwen3-8b-Q4_K_M.gguf \
    bash scripts/run_1proc_elbrus.sh --greedy "..."
```

### 4.2. Tensor Parallel TP-4 (4 чипа × 8 ядер)

Для моделей до 7B включительно. **Это основной production-режим.**

```bash
PT_Q8_SOA=1 bash scripts/run_tp_elbrus.sh --greedy "Write a haiku about AI"
```

Что внутри (для каждого rank=0..3):

```bash
PT_NO_NUMA_POOL=1 \
OMP_NUM_THREADS=8 \
PT_NUMA_REPLICATE=0 \
PT_DDP_SHM=1 \
PT_Q8_SOA=1 \
numactl --cpunodebind=$rank --membind=$rank \
    ./build_elbrus/examples/gguf/test_gguf_inference "$MODEL" \
    --nprocs 4 --rank $rank \
    --master-addr 127.0.0.1 --master-port 29500 \
    --max-tokens 100 --greedy --chat "$PROMPT" \
    > run_logs/tp4_rank${rank}.log 2>&1 &
```

Обязательные env'ы — **разбираем каждый**:

| Env | Зачем |
|-----|-------|
| `PT_Q8_SOA=1` | **СВЯТОЕ** — Q8 SoA4 4-row interleaved INT8 layout под `qpmaddubsh`. Без флага = 7.7 tok/s, с флагом = 11.4. Сэкономит часы дебага |
| `OMP_NUM_THREADS=8` | По одному NUMA-ноду на rank, sweet-spot после persistent ThreadPool (commit `a338ae6`) |
| `PT_NO_NUMA_POOL=1` | Отключает NUMA-aware ThreadPool у пользовательского кода — каждый rank уже numactl-привязан, замаскирует |
| `PT_DDP_SHM=1` | SHM AllReduce через `/dev/shm` вместо TCP loopback (~10× быстрее на 2560 floats) |
| `PT_NUMA_REPLICATE=0` | В TP уже membind'нуто к одному node, реплика бессмысленна |
| `numactl --cpunodebind=$rank --membind=$rank` | Жёсткая локализация compute и memory на свой NUMA-ноду |

**HARD GUARD: НЕ ставить `PT_PIN_THREADS=1`.** ThreadPool маппит worker_id на
абсолютные CPU ID 0..31. Ранки 1-3 numactl-привязаны к CPU 8-15/16-23/24-31, и
pin на 0..7 либо отбрасывается ядром, либо клампит всех воркеров на одно
allowed-CPU. Падение с 9.4 до **1.4 tok/s** (×7 регрессия). Скрипт явно проверяет
этот env и падает с ошибкой, если он = 1.

### Указать другую модель TP-4

```bash
PT_Q8_SOA=1 PT_MODEL=$HOME/gguf_models/mistral-7b-Q4_K_M.gguf \
    bash scripts/run_tp_elbrus.sh --greedy "Write code in Python"
```

Логи: `run_logs/tp4_rank{0,1,2,3}.log`. Output печатает только rank 0.

---

## 5. Бенчмарки и сравнение с llama.cpp

### Бенчмарк PromeTorch TP-4

```bash
PT_Q8_SOA=1 bash scripts/run_tp_elbrus.sh --greedy "Write code"
# смотреть tok/s в run_logs/tp4_rank0.log
```

### Fair compare с llama.cpp (32 потока, тот же NUMA interleave)

```bash
bash scripts/elbrus_llama_bench.sh
# результаты в /tmp/llama_bench.log
```

Скрипт прогоняет `~/llama.cpp/build/bin/llama-bench` через
`numactl --interleave=all -t 32 -p 32 -n 64 -r 2` для всех моделей.

### Реальные числа (из BENCH_ELBRUS.md, 2026-05-03)

**qwen3:4b Q4_K_M, decode tok/s:**

| Конфигурация | tok/s | Speedup |
|--------------|------:|---------|
| llama.cpp 32t pthread (pure-C) | 3.3 | baseline |
| PromeTorch 1-proc 32t plain | 2.8 | ×0.85 |
| PromeTorch 1-proc 24t + interleave=all | 3.8 | ×1.15 |
| PromeTorch 1-proc 30t + interleave=all + Q4_K prefetch | 4.7 | ×1.42 |
| PromeTorch TP-4 SHM (без Q8 SoA) | 7.7 | ×2.3 |
| **PromeTorch TP-4 + Q8 SoA4 (PT_Q8_SOA=1)** | **9.4** | **×2.85** |
| **PromeTorch TP-4 + Q8 SoA4 + persistent ThreadPool 8t** | **10.6** | **×3.2** |
| **PromeTorch TP-4 + fused QKV + AVX2 attn + Q8 SiLU+quant** ★ | **11.4** | **×3.5 lossless ceiling** |
| PromeTorch TP-4 + LayerSkip 12 alt (lossy) | 15.5 | ×4.7 |

### Speedups vs llama.cpp 32t (по моделям)

| Модель | PromeTorch | llama.cpp 32t | Ratio |
|--------|-----------:|--------------:|------:|
| qwen3-1.7B | 17.1 (TP-4) | 2.7 | **×6.3** |
| qwen3-4B | 10.9 (TP-4) | 1.8 | **×6.0** |
| gemma3-4B | 6.7 (TP-4) | 1.3 | **×5.2** |
| mistral-7B | 8.5 (TP-4) | 1.7 | **×4.9** |
| phi3.5-mini | 6.4 (TP-4) | 2.1 | **×3.1** |

11.4 tok/s lossless ceiling на qwen3-4B закреплён disassembly-анализом
(см. BENCH_ELBRUS.md): inner loop `q8_soa4_gemv` упаковывает 6 ops/cycle на
6-wide VLIW — peak instruction density, дальше не ускоряется без hand-asm.

---

## 6. Поддерживаемые модели

10/10 GGUF моделей работают на Эльбрусе (по состоянию на 2026-05-03):

| Модель | Mode | tok/s | Примечание |
|--------|------|------:|------------|
| qwen3-1.7B | TP-4 | 17.1 | sweet-spot, ×6.3 vs llama.cpp |
| qwen3-4B | TP-4 | 10.9 | baseline 11.4 — production target |
| qwen3-8B | SP | 2.6 | TP-4 OOM |
| qwen3-14B | SP | 1.5 | большая модель, только SP |
| qwen2.5-7B | SP | 2.9 | TP-4 OOM |
| mistral-7B | TP-4 | 8.5 | Llama-architecture compat |
| gemma3-4B | TP-4 | 6.7 | post-norm wire fix `0ba114a` |
| phi3.5-mini | TP-4 | 6.4 | mmap split fix `d9dce9e` |
| llama3-8B | SP | 2.7 | TP-4 OOM |
| deepseek-coder-7B | SP | 3.0 | rope.scale_linear legacy ключ `81a79bd` |
| qwen3-0.6B | — | — | capacity issue, не code-fixable |

**TP-4 OOM:** ~32 ГБ peak per rank, на 4 nodes = 128 ГБ. На моделях >7B parameter
+ KV-cache + activations это превышает 32 ГБ DDR на ноду. Для таких — SP режим.

**SP моделей пока без TP:** портирование на k-slice требует пересчёта raw-block
boundaries для конкретной FFN/attention shape. Roadmap.

---

## 7. Подводные камни / Troubleshooting

### Сборка падает с `omp.h: No such file` или `cblas.h not found`

На свежей Альт-инсталляции под Эльбрус (проверено 2026-05-04) **по
умолчанию НЕТ** пакетов `libomp11-devel` и `eml-devel`. CMake при этом
проходит, но `aten_cpu` валится при компиляции с десятком ошибок. Фикс:

```bash
apt-get install -y eml-devel libomp11-devel libnuma-devel
rm -rf build_elbrus && ./scripts/build-elbrus.sh
```

После установки `eml-devel` CMake должен залогировать `EML found`,
после `libomp11-devel` — `Found OpenMP_CXX`. Если всё ещё нет — проверь
include-пути в выводе CMake.

### Тренировка/inference умирает при SSH disconnect

```bash
loginctl enable-linger $USER
```

Без linger'а systemd-logind убьёт все процессы. Касается inference, тренировки,
tmux-сессий. См. JOURNAL.md 2026-04-16.

### EML SIGILL при cblas_sgemm из pthread

Симптом: SIGILL (illegal instruction) при первом вызове `cblas_sgemm` из любого
pthread, даже на 64×64 single-threaded матрице.

Root cause (2026-04-01): `cblas_sgemm` на E2K работает **только из main thread**.
Любой pthread/std::thread → SIGILL.

Фикс уже в коде: убраны все pthread NUMA-обёртки, EML обрабатывает NUMA через
свой внутренний OpenMP. См. CMakeLists.txt:181-184 комментарий.

### `PT_PIN_THREADS=1` в TP режиме

**НЕ ставить.** Падение с 9.4 до 1.4 tok/s. Скрипт `run_tp_elbrus.sh` падает с
ошибкой при таком env. Подробнее — секция 4.2 выше.

### `numactl` ругается "Conflicting policies"

Не использовать `--membind` и `--preferred` одновременно. На Эльбрусе numactl
строже чем на x86 — exit с ошибкой и rank умирает тихо. В TP режиме —
**только `--membind`**. См. `run_tp_elbrus.sh:50-52` комментарий.

### qwen3-4B TP-4 показывает 7.7 tok/s вместо 11.4

Забыл `PT_Q8_SOA=1`. Без него используется legacy GEMV path. **Этот баг ловят
20+ раз подряд — всегда проверять env первым делом.**

### LCC quirks

| Проблема | Workaround |
|----------|-----------|
| Structured bindings в лямбдах | Использовать поля структуры |
| `throw` внутри `#pragma omp parallel` | Вынести `throw` за пределы omp |
| Variadic macro `##__VA_ARGS__` | Явный `if/throw` |
| LTO + ar wrapper | `-fno-lto` по умолчанию |
| C++20 features | Не поддерживается, использовать C++17 |
| `-fwhole` | Не использовать — генерит EIR который `/usr/bin/ld` не принимает |

### phi3.5-mini не загружается / 0 токенов

Был баг mmap split на split-tensor блоках Q4_K_M. Фикс — commit `d9dce9e`. Если
сборка свежая, баг уже исправлен.

### deepseek-coder-7B даёт garbage output

GGUF использует legacy ключ `rope.scale_linear` вместо `rope.scaling.factor`.
Фикс — commit `81a79bd`. Проверить наличие в build_elbrus.

### gemma3-4B TP-4 даёт мусор

Post-norm wire не был подключён в TP режиме. Фикс — commit `0ba114a`.

### `/dev/shm/prometorch_ddp_*` остался от прошлого прогона

Если предыдущий TP-4 упал не штатно — SHM сегмент не был очищен. Симптом:
второй запуск висит на `shm_open`. Чистить:

```bash
rm -f /dev/shm/prometorch_ddp_*
```

---

## 8. Известные ограничения

- **TP-4 OOM на моделях >7B** — ~32 ГБ peak per rank; для qwen3-8B/14B,
  qwen2.5-7B, llama3-8B, deepseek-7B используем SP режим.
- **MoE архитектуры не поддерживаются** — ни qwen2-moe, ни mixtral.
- **Tied weights крашат для gemma3:27b** (известно для x86 build тоже).
- **`item()` в Python биндингах** возвращает 0 на float32 тензорах (E2K-specific
  dtype dispatch баг). Workaround: `tolist()[0]`.
- **`from_numpy()` на int64** несовместим. Использовать `np.int32` + `.long()`.
- **`pt.tensor([list])`** не принимает Python lists — обернуть в `np.array(...)`.
- **promeserve target** (Ollama-killer inference server) ещё не собирался на
  сервере — TODO.
- **`-fwhole` / `-fwhole-shared`** отключены — LCC EIR не линкуется системным
  `/usr/bin/ld`. Кросс-TU оптимизация частичная.
- **Lossless ceiling 11.4 tok/s** на qwen3-4B TP-4 закреплён disassembly. Дальше
  только hand-asm или EAGLE/спекулятивный декод.

---

## 9. PIR 250M training

Опционально — тренировка PIR 342M архитектуры (родственник Mamba/HGRN/RWKV) в
4-процесс Local SGD (file-based weight averaging, **не** DDP gradient AllReduce).

```bash
loginctl enable-linger $USER
for node in 0 1 2 3; do
  PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
  numactl --cpunodebind=$node --preferred=$node \
  ./build_elbrus/examples/pir/train_pir_elbrus \
    --fused --full --batch_size 4 --rank $node --nprocs 4 \
    --max_steps 1000 --log_interval 50 --gen_interval 200 --gen_tokens 200 \
    --save_interval 200 --save_dir checkpoints \
    --grad_accum 10 --lr 0.0006 \
    --data data/russian_mega.txt &
done
```

Здесь `--preferred` допустим (Local SGD — не TP, нет numactl strict-conflict).

Реальные результаты (loss / tok/s на тренировке):

| Step | Loss | Perplexity | tok/s | Сэмпл генерации |
|------|------|-----------:|------:|-----------------|
| 200  | 1.41 | 4.1 | 158 | "соположение", "Кроины" |
| 400  | 1.23 | 3.4 | 147 | "Первой", "специального", "военных" |
| 600  | 1.15 | 3.1 | 147 | "Российская", "города", "музей" |
| 800  | 1.04 | 2.8 | 147 | "В России", "полагается", "15 марта 2008 года" |

Backward fix 2026-04-18 (commit полная сессия): `embedding`/`parallel_scan`/
`gate-value` backward были stub'ом, теперь полноценный.

---

## 10. Ссылки

- [BENCH_ELBRUS.md](../BENCH_ELBRUS.md) — полная матрица бенчмарков, peak compute
  disassembly анализ, scaling sweeps
- [docs/elbrus/README_ELBRUS_RU.md](elbrus/README_ELBRUS_RU.md) — детальный
  архитектурный гайд по TUDA E2K, EML, NUMA-aware GEMM
- [docs/elbrus_report/ELBRUS_REPORT_v2.md](elbrus_report/ELBRUS_REPORT_v2.md) —
  отчёт о производительности для МЦСТ
- [docs/elbrus_report/PROMETORCH_ELBRUS_ARTICLE.pdf](elbrus_report/PROMETORCH_ELBRUS_ARTICLE.pdf) —
  публикационная версия
- [JOURNAL.md](../JOURNAL.md) — полная история фиксов (2026-04-12..2026-05-03)
- [scripts/build-elbrus.sh](../scripts/build-elbrus.sh) — build entry-point
- [scripts/run_tp_elbrus.sh](../scripts/run_tp_elbrus.sh) — TP-4 runner
- [scripts/run_1proc_elbrus.sh](../scripts/run_1proc_elbrus.sh) — SP runner
- [scripts/elbrus_llama_bench.sh](../scripts/elbrus_llama_bench.sh) — fair-compare
  с llama.cpp 32t
- [cmake/toolchains/e2k-elbrus.cmake](../cmake/toolchains/e2k-elbrus.cmake) —
  LCC toolchain
