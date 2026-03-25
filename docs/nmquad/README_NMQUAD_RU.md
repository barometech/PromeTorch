# NM QUAD Backend -- PromeTorch

## 1. Обзор

NM QUAD backend -- четвертый backend PromeTorch (после CPU, CUDA и NM Card Mini). Реализует обучение нейронных сетей (GPT-трансформеры) на отечественной 4-чиповой плате NM QUAD производства НТЦ "Модуль" на базе процессоров NM6408.

Ключевые особенности:
- **Row-parallel архитектура** -- каждое ядро выполняет полный fused forward/backward для своей порции batch
- **SIMD matmul через nmpp** -- 100x ускорение по сравнению со скалярной реализацией
- **До 64 ядер** (4 чипа x 16 ядер) работают параллельно без inter-core координации
- **Gradient accumulation** -- безопасный параллельный backward без race condition
- **Модели до 250M параметров** помещаются в DDR (2 ГБ из 5 ГБ на чип)

### Файловая структура

```
c10/nmquad/
  NMQuadAllocator.h           # Host-side caching allocator (PrivateUse3)

aten/src/ATen/nmquad/
  NMQuadDispatch.h             # Host-side dispatch API
  NMQuadHardware.h             # Hardware abstraction (PL_* API wrappers)
  NMQuadOps.h                  # Tensor operations для NM QUAD
  nmc_programs/
    dispatcher_nmquad.cpp      # v1: простой matmul dispatcher
    dispatcher_nmquad_v2.cpp   # v2: coordinator+workers (DEPRECATED -- зависает)
    dispatcher_nmquad_v3.cpp   # v3: row-parallel fused (АКТУАЛЬНЫЙ)
    dispatcher_nmquad_v3_scalar.cpp  # v3 без nmpp SIMD (отладочный)
    nm6408brd.lds              # Linker script для NMC ядер

examples/nmquad/
    train_gpt_4chip.cpp        # 4-chip GPT training (основной пример)
    train_gpt_nm6408.cpp       # 1-chip GPT training
    train_mnist_nmquad.cpp     # MNIST на NM QUAD
    profile_nmquad.cpp         # Profiling утилита

docs/nmquad/
    nm6408load_nmc.h           # API загрузки программ на NMC ядра
    nm_quad_load.h             # API инициализации платы
    inter_nmc_user.h           # Inter-core communication API
```

---

## 2. Архитектура железа

### Плата NM QUAD

| Параметр | Значение |
|----------|----------|
| Процессоры | 4 x NM6408 |
| Ядра на чип | 16 (4 кластера x 4 NMC ядра) |
| Всего ядер | **64** |
| DDR на чип | **5 ГБ** (20 ГБ суммарно) |
| Пиковая производительность | **2 TFLOPS FP32** |
| Интерфейс | PCIe x16 Gen 3 |
| TDP | ~80 Вт |

### Процессор NM6408

Каждый NM6408 содержит 16 NMC (NeuroMatrix Core) ядер, организованных в 4 кластера по 4 ядра. Каждое ядро имеет:
- Скалярный процессор (RISC)
- Векторный сопроцессор (VPU) с SIMD-инструкциями для матричных операций
- Доступ к общей DDR памяти чипа через DMA
- Локальную внутреннюю память (PRAM/WRAM)

---

## 3. Топология

### Board = Chip

Ключевое открытие: `PL_GetBoardCount()` возвращает **4** -- по одному board на каждый NM6408. Каждый board = отдельный чип со своей 5 ГБ DDR.

```
NM QUAD Плата
├── Board 0 (NM6408 #0, DDR 5 ГБ)
│   ├── Cluster 0: Core 0, 1, 2, 3
│   ├── Cluster 1: Core 4, 5, 6, 7
│   ├── Cluster 2: Core 8, 9, 10, 11
│   └── Cluster 3: Core 12, 13, 14, 15
├── Board 1 (NM6408 #1, DDR 5 ГБ)
│   ├── Cluster 0: Core 0, 1, 2, 3
│   ├── ...
├── Board 2 (NM6408 #2, DDR 5 ГБ)
│   └── ...
└── Board 3 (NM6408 #3, DDR 5 ГБ)
    └── ...
```

### Адресация ядер

Каждое ядро идентифицируется парой `PL_CoreNo{nm_id, cluster_id}`:
- `nm_id` = 0..3 (номер ядра внутри кластера)
- `cluster_id` = 0..3 (номер кластера внутри чипа)
- Линейный индекс ядра: `core_id = (cluster_id << 2) + nm_id`

```c
PL_CoreNo cn = {nm_id, cluster_id};
PL_Access* access;
PL_GetAccess(board, &cn, &access);
PL_LoadProgramFile(access, "dispatcher_nmquad_v3.abs");
```

### DDR организация (на каждом чипе)

```
DDR_BASE = 0x00340000

[0x00340000 .. 0x00340000 + 16*32)   CMD блоки (32 слова на ядро)
[DATA0 = DDR_BASE + 512 ..)          Область данных:
  ├── wte[V*D]                       Embedding (shared)
  ├── wpe[T*D]                       Position embedding (shared)
  ├── layers[L*layer_size]           Packed layer weights (shared)
  ├── lm_head[D*V]                   Language model head (shared)
  ├── Per-core 0:
  │   ├── tokens[B_mine*T]
  │   ├── logits[B_mine*T*V]
  │   ├── h_cache[(L+1)*BT*D + L*BT*D + L*BT*FF]
  │   ├── fwd_scratch[...]
  │   ├── dlogits[BT*V]
  │   ├── bwd_scratch[...]
  │   └── grad_buf[D*V + L*lsz + V*D]  (при --parallel-backward)
  ├── Per-core 1: ...
  └── ...
```

---

## 4. Dispatcher v3 -- Row-Parallel Fused Forward/Backward

### Эволюция архитектуры

| Версия | Архитектура | Статус |
|--------|-------------|--------|
| v1 | Простой matmul dispatch | Работает, медленный |
| v2 | Coordinator (Core 0) + Workers (1-15) | **DEPRECATED** -- зависает из-за DDR polling |
| v3 | **Row-parallel: все ядра равноправны** | **АКТУАЛЬНЫЙ** |

### Принцип работы v3

Каждое из 16 ядер на чипе запускает **полный fused transformer** (forward + backward) для своей порции batch. Нет inter-core координации, нет coordinator/worker разделения.

```
Host dispatch:
  Core 0:  rows 0..B_mine-1        → полный forward+backward
  Core 1:  rows B_mine..2*B_mine-1 → полный forward+backward
  ...
  Core 15: rows 15*B_mine..B-1     → полный forward+backward

Каждое ядро:
  - Читает ОБЩИЕ веса (read-only)
  - Пишет в СВОЮ часть DDR (non-overlapping)
  - Полностью независимо: нет барьеров, нет синхронизации
```

### Opcodes

| Opcode | Константа | Назначение |
|--------|-----------|------------|
| 0 | `OP_NOP` | Нет операции |
| 1 | `OP_MATMUL` | Одиночное умножение матриц |
| 32 | `OP_FUSED_FORWARD_ROWPAR` | Полный forward для B_mine строк batch |
| 33 | `OP_FUSED_BACKWARD_ROWPAR` | Полный backward + SGD update с lr/N_cores |
| 34 | `OP_FUSED_BACKWARD_GRADONLY` | Backward без weight update (градиенты в буфер) |
| 255 | `OP_EXIT` | Завершение работы ядра |

### CMD блок (32 слова на ядро)

Forward (`OP_FUSED_FORWARD_ROWPAR`):
```
[0]  opcode         = 32
[1]  B_mine         -- количество строк batch для этого ядра
[2]  T              -- длина последовательности
[3]  D              -- размерность модели
[4]  H              -- количество голов внимания
[5]  FF             -- размерность FFN hidden
[6]  V              -- размер словаря
[7]  L              -- количество слоёв
[8]  tokens_addr    -- адрес токенов этого ядра [B_mine*T]
[9]  wte_addr       -- адрес embedding весов (shared)
[10] wpe_addr       -- адрес position embeddings (shared)
[11] layers_addr    -- адрес packed layer weights (shared)
[12] lm_head_addr   -- адрес lm_head [D*V] (shared)
[13] logits_addr    -- адрес выхода logits этого ядра
[14] h_out_addr     -- адрес кеша h для backward
[15] scratch_addr   -- адрес scratch-памяти этого ядра
[30] status         -- 0=busy, 1=done, 2=error
```

### Forward pipeline (на каждом ядре)

1. **Embedding**: `h[b,t,d] = wte[token] + wpe[t]`
2. Для каждого слоя `l = 0..L-1`:
   - **RMSNorm**: `hn = rmsnorm(h) * gamma`
   - **QKV проекции**: `Q,K,V = hn @ Wq, hn @ Wk, hn @ Wv` (SIMD matmul)
   - **Reshape**: Q,K,V в multi-head формат `[BH, T, HD]`
   - **Attention**: `scores = Q @ K^T * (1/sqrt(HD))` → causal softmax → `attn = scores @ V`
   - **Output projection**: `proj = attn @ Wo` (SIMD matmul)
   - **Residual**: `h += proj`
   - **FFN**: `ff1 = relu(h @ W1)`, `ff2 = ff1 @ W2`, `h += ff2` (SIMD matmul)
3. **LM Head**: `logits = h @ lm_head` (SIMD matmul)

### Backward pipeline

Полный обратный проход через все слои с вычислением градиентов для всех весов. Поддерживает два режима:
- **OP 33 (ROWPAR)**: вычисляет градиенты и сразу обновляет веса (`lr_scaled = lr / N_cores`)
- **OP 34 (GRADONLY)**: только записывает градиенты в per-core буфер, host суммирует и применяет SGD

---

## 5. nmpp SIMD matmul

### Ускорение

| Режим | Время forward (small model) | tok/s (1 ядро) | Ускорение |
|-------|----------------------------|----------------|-----------|
| Скалярный C++ | 6675 мс | 4.8 | 1x |
| **nmpp SIMD** | **66.5 мс** | **481** | **100x** |

### Функция `nmppmMul_mm_32f`

Библиотека nmpp предоставляет оптимизированное матричное умножение, использующее SIMD-инструкции VPU процессора NM6408.

```c
extern "C" {
    void nmppmMul_mm_32f(
        float* A, int nHeight1, int nStride1,   // A[M x K], stride = K
        float* B, int nWidth1,  int nStride2,   // B[K x N], stride = N
        float* C, int nWidth2,  int nStrideDst,  // C[M x N], stride = N
        int bPlusDst                              // 0 = overwrite, 1 = accumulate
    );
}
```

Обёртка в dispatcher:
```c
static void nmppmMul_mm_32f_wrap(float* A, int M, int K, float* B, int N, float* C) {
    nmppmMul_mm_32f(A, M, K, B, K, N, C, N, N, 0);
}
```

### Сборка asm-файла

Исходный код SIMD matmul находится в `MullMatrix_f.asm` -- написан на ассемблере NM6408 и использует VPU для параллельного вычисления скалярных произведений строк и столбцов.

```bash
# Сборка объектного файла из asm
nmc-as -o MullMatrix_f.o MullMatrix_f.asm

# Или через nmcc (если есть C-обёртка)
nmc-gcc -c -O2 -o MullMatrix_f.o MullMatrix_f.asm
```

---

## 6. Gradient Accumulation (OP 34)

### Проблема

При `OP_FUSED_BACKWARD_ROWPAR` (opcode 33) каждое ядро напрямую обновляет ОБЩИЕ веса в DDR. Когда несколько ядер на одном чипе делают это одновременно, возникает **race condition** (read-modify-write на одни и те же адреса).

### Решение: GRADONLY режим

Opcode 34 (`OP_FUSED_BACKWARD_GRADONLY`) реализует безопасный параллелизм:

1. Каждое ядро вычисляет градиенты и записывает их в **свой собственный буфер** (`grad_buf`) в DDR
2. Ядра **не трогают** общие веса -- они read-only
3. Host **суммирует** градиенты со всех ядер
4. Host **применяет SGD** к весам

```
Per-core gradient buffer layout:
  grad_buf[0 .. D*V-1]           -- grad_lm_head
  grad_buf[D*V .. D*V+L*lsz-1]  -- grad_layers (packed)
  grad_buf[D*V+L*lsz .. end]    -- grad_wte
```

Это обеспечивает **нулевой race condition** при полном параллелизме backward.

---

## 7. DDR ограничения

### Проблема DMA конфликтов

Процессор NM6408 имеет VPU, который использует DMA для чтения данных из DDR. Когда **8 и более ядер** одновременно читают **одни и те же адреса** (общие веса), возникают DDR bank conflicts, приводящие к зависанию.

### Ограничение: max 8 concurrent SIMD DMA

- **16 ядер forward** = работает (10M polling delay помогает)
- **8+ ядер backward** на одном чипе = **зависание** (write contention)
- **4 ядра/чип backward** = стабильно

### Решение: 10M polling delay

Idle ядра (ожидающие команду) в цикле опроса DDR блокируют VPU DMA активных ядер. Добавление задержки 10 миллионов тактов в polling loop устраняет проблему для forward:

```c
// В dispatcher, цикл ожидания команды:
while (mem[core_index * CBS] == OP_NOP) {
    // Без этой задержки: idle cores забивают DDR bus
    for (volatile int d = 0; d < 10000000; d++);
}
```

### Решение: Per-core weight copy

Перед fused forward/backward каждое ядро копирует нужные веса в **свою приватную область DDR**. Это устраняет DDR bank conflicts, так как каждое ядро читает из разных адресов:

```
Стоимость:  ~1 МБ дополнительной DDR на ядро (для small model)
Выигрыш:   Все 16 ядер работают SIMD matmul одновременно без DDR stall
```

### Wave Dispatch

Для backward, где per-core weight copy не помогает (write contention), используется wave dispatch -- backward запускается волнами по `--wave-size` ядер:

```
--wave-size 4 (default):
  Wave 1: Cores 0-3 backward   (4 ядра x 4 чипа = 16 параллельно)
  Wave 2: Cores 4-7 backward
  Wave 3: Cores 8-11 backward
  Wave 4: Cores 12-15 backward
```

---

## 8. Сборка

### Dispatcher (NMC-программа)

Dispatcher компилируется кросс-компилятором `nmc-g++` и линкуется с nmpp и загрузчиком:

```bash
nmc-g++ -std=gnu++11 -O2 \
    -o dispatcher_nmquad_v3.abs \
    dispatcher_nmquad_v3.cpp \
    -Wl,--whole-archive \
    -l nm6408load_nmc \
    -lnmpp-nm6408 \
    -Wl,--no-whole-archive \
    -T nm6408brd.lds
```

Зависимости:
- `nm6408load_nmc` -- загрузчик программ на NMC ядра
- `nmpp-nm6408` -- библиотека SIMD-операций (включая `nmppmMul_mm_32f`)
- `nm6408brd.lds` -- linker script, определяющий memory map (PRAM, WRAM, DDR)

### Host-программа

Host-программа компилируется обычным `g++` и линкуется с драйвером платы:

```bash
g++ -O2 -o train_gpt_4chip \
    train_gpt_4chip.cpp \
    -ldl \
    -lnm_quad_load
```

### Asm-файлы

NM ассемблер используется для низкоуровневых SIMD-ядер:

```bash
# Сборка asm-объекта
nmc-as -o MullMatrix_f.o MullMatrix_f.asm

# Сборка со всеми объектами
nmc-g++ -std=gnu++11 -O2 \
    -o dispatcher.abs \
    dispatcher.cpp \
    MullMatrix_f.o \
    nmppmMul_mm_32sXs_nm.o \
    -lnm6408load_nmc \
    -T nm6408brd.lds
```

---

## 9. Запуск

### CLI-аргументы

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--data` | `tiny_shakespeare.txt` | Путь к обучающему тексту |
| `--dispatcher` | `dispatcher_nmquad_v3.abs` | Путь к скомпилированному dispatcher |
| `--model` | `small` | Размер модели: `small`, `large`, `250m` |
| `--epochs` | `10` | Количество эпох |
| `--steps` | `200` | Шагов на эпоху |
| `--batch` | `64` | Общий batch size (делится между ядрами) |
| `--lr` | `0.001` | Learning rate |
| `--boards` | `4` | Максимум используемых чипов (1-4) |
| `--clusters` | `4` | Максимум кластеров на чип (1-4) |
| `--cores` | `4` | Максимум ядер на кластер (1-4) |
| `--wave-size` | `4` | Ядер в волне backward (на чип) |
| `--parallel-backward` | `false` | Использовать GRADONLY opcode 34 |

### Конфигурации моделей

| Модель | D | H | FF | L | T | Параметры | DDR/чип |
|--------|---|---|----|---|---|-----------|---------|
| `small` | 128 | 4 | 256 | 2 | 32 | ~200K | <100 МБ |
| `large` | 768 | 12 | 1536 | 12 | 64 | ~85M | ~1.5 ГБ |
| `250m` | 768 | 12 | 3072 | 36 | 64 | ~255M | ~2.0 ГБ (39.8%) |

### Примеры запуска

```bash
# 1 чип, 4 ядра -- минимальная конфигурация
./train_gpt_4chip --data tiny_shakespeare.txt --model small \
    --boards 1 --clusters 1 --cores 4

# 4 чипа, 16 ядер -- стабильная конфигурация
./train_gpt_4chip --data tiny_shakespeare.txt --model small \
    --boards 4 --clusters 1 --cores 4 --wave-size 4

# 4 чипа с параллельным backward (gradient accumulation)
./train_gpt_4chip --data tiny_shakespeare.txt --model large \
    --boards 4 --clusters 4 --cores 4 --parallel-backward --wave-size 4

# 250M модель
./train_gpt_4chip --data corpus.txt --model 250m \
    --boards 4 --clusters 4 --cores 4 --parallel-backward --lr 0.0003
```

---

## 10. Результаты

### Производительность (small model)

| Конфигурация | Ядер | tok/s | Loss | Статус |
|-------------|------|-------|------|--------|
| 1 board, 1 cluster, scalar | 4 | 7 | 4.17 -> drops | Стабильно |
| 1 board, 1 cluster, SIMD | 4 | **220** | 5.27 | Стабильно (gradonly) |
| 1 board, 1 cluster, SIMD seq | 4 | **147** | 4.17 -> drops | Стабильно (rowpar) |
| 2 boards, 1 cluster, SIMD | 8 | **407** | 4.72 | Стабильно (wave) |
| **4 boards, 1 cluster, SIMD** | **16** | **705** | 4.45 | **Стабильно (wave)** |

### Ускорение SIMD matmul

| Операция | Скалярный | nmpp SIMD | Ускорение |
|----------|----------|-----------|-----------|
| Forward (1 ядро) | 6675 мс | 66.5 мс | **100x** |
| tok/s (1 ядро) | 4.8 | 481 | **100x** |

### Масштабирование

| Ядер | tok/s | Линейная экстраполяция | Эффективность |
|------|-------|----------------------|---------------|
| 4 | 220 | 220 (baseline) | 100% |
| 8 | 407 | 440 | 92% |
| 16 | 705 | 880 | 80% |

---

## 11. Известные ограничения

### DDR DMA контроллер

- **Max 8 concurrent SIMD DMA** на одном чипе для backward
- 16 ядер forward работают (с 10M polling delay), но backward 8+ ядер на одном чипе зависает
- **Root cause**: idle cores в DDR polling loop блокируют VPU DMA активных ядер
- **Workaround**: wave dispatch по 4 ядра на чип

### Multi-board backward

- Максимум **4 ядра на board** для стабильного multi-board backward
- Итого: 4 boards x 4 cores = 16 ядер при wave dispatch
- Увеличение `--wave-size` > 4 может привести к зависанию

### Backward race condition

- `OP_FUSED_BACKWARD_ROWPAR` (33): все ядра обновляют общие веса -- race condition при >1 ядра
- Решение: `OP_FUSED_BACKWARD_GRADONLY` (34) + host SGD
- Sequential backward (1 ядро за раз) как fallback для точности

### Координация между чипами

- У каждого чипа своя DDR с копией весов
- **Cross-chip weight sync**: host усредняет веса между чипами каждые N шагов
- Это Data-Parallel подход (не Model-Parallel)

### Модель 250M

- DDR: ~2.0 ГБ на чип (39.8% от 5 ГБ) -- помещается
- `HD = 768/12 = 64` -- совпадает с массивами `Q_h[64*64]` в dispatcher
- Тренировка до loss 1.5 -- в процессе

---

## 12. Roadmap

### Ближайшие задачи

- **250M model training**: запуск полноценного обучения с loss convergence до 1.5
- **Custom tiled matmul**: замена nmppmMul_mm_32f на кастомный tiled GEMM, оптимизированный для DDR bandwidth -- обход лимита 8 concurrent DMA
- **Firmware-level DDR fix**: работа с НТЦ "Модуль" по устранению DDR contention в firmware

### Среднесрочные цели

- **64 ядра полный параллелизм** (4 чипа x 16 ядер) -- forward + backward
- **Model-Parallel** для моделей >5 ГБ (слои на разных чипах)
- **FP16/INT8 квантизация** на VPU -- удвоение throughput
- **Gradient compression** для inter-chip sync -- снижение PCIe трафика

### Долгосрочные цели

- **Distributed training** через несколько плат NM QUAD (NCCL-аналог по PCIe)
- **Inference runtime** для развёрнутых моделей с layer streaming

---

## Приложение A: Host API (PL_* функции)

```c
// Инициализация
int PL_GetBoardCount(unsigned int* count);       // Возвращает количество чипов (4)
int PL_GetBoardDesc(unsigned int idx, PL_Board**);  // Дескриптор чипа
int PL_ResetBoard(PL_Board*);                     // Сброс чипа
int PL_LoadInitCode(PL_Board*);                   // Загрузка init-кода

// Доступ к ядрам
int PL_GetAccess(PL_Board*, PL_CoreNo*, PL_Access**);  // Подключение к ядру
int PL_LoadProgramFile(PL_Access*, const char*);        // Загрузка .abs на ядро

// DDR I/O
int PL_WriteMemBlock(PL_Access*, const PL_Word*, PL_Addr, unsigned int);  // Host -> DDR
int PL_ReadMemBlock(PL_Access*, PL_Word*, PL_Addr, unsigned int);         // DDR -> Host

// Настройки
int PL_SetTimeout(unsigned int ms);
int PL_CloseAccess(PL_Access*);
int PL_CloseBoardDesc(PL_Board*);
```

## Приложение B: Упаковка весов слоя

Все веса одного transformer-слоя упаковываются последовательно:

```
layer_size = 4*D*D + 2*D*FF + D

Смещения:
  Wq:     0                 -- [D x D] Query projection
  Wk:     D*D               -- [D x D] Key projection
  Wv:     2*D*D             -- [D x D] Value projection
  Wo:     3*D*D             -- [D x D] Output projection
  W1:     4*D*D             -- [D x FF] FFN up-projection
  W2:     4*D*D + D*FF      -- [FF x D] FFN down-projection
  gamma:  4*D*D + 2*D*FF    -- [D] RMSNorm scale
```

## Приложение C: Важные баги и их решения

| # | Баг | Решение |
|---|-----|---------|
| 1 | bwd_scratch overflow: dW выделялось D*V, нужно max(D*V,D*D,D*FF,FF*D) | DDR corruption, расширение буфера |
| 2 | lr_scaled = lr/n_cores -- неправильно для independent data per core | Убрано деление |
| 3 | dlogits /= B_per_core, должно быть /= B_total | Исправлена нормализация loss |
| 4 | d_O из modified dx -- attention backward bug | Сохранение чистого d_O |
| 5 | Scalar backward: Wv = same grad as Wk | Исправлен порядок QKV backprop |
| 6 | gradonly scratch mismatch -- тот же overflow | Синхронизация размеров с rowpar |
| 7 | v2 coordinator/workers зависает | Полная переработка в v3 row-parallel |
| 8 | DDR bank conflicts при 16 cores SIMD | Per-core weight copy + wave dispatch |
