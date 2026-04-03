# ДОКУМЕНТАЦИЯ ЭЛЬБРУСА: LCC — Оптимизация компилятора для E2K

> **ЭТО ДОКУМЕНТАЦИЯ ЭЛЬБРУСА!** Флаги LCC для максимальной производительности на E8C2.
> Источник: dev.mcst.ru/book/chapter6.html, altlinux.org/Эльбрус/оптимизация

---

## 1. Уровни оптимизации

| Флаг | Описание |
|------|----------|
| `-O0` | Debug, сохраняет соответствие исходнику |
| `-O1` | Локальные оптимизации внутри линейных участков |
| `-O2` | Внутрипроцедурные оптимизации, циклы, полное использование HW |
| `-O3` | **Стандарт для продакшена.** Межпроцедурные оптимизации |
| `-O4` | Агрессивные/экспериментальные. Может помочь или ухудшить. Тестировать! |

---

## 2. Ключевые флаги LCC

### Производительность

| Флаг | Описание |
|------|----------|
| `-ffast` | Ослабленная FP семантика (как -ffast-math + больше) |
| `-ffast-math` | Переупорядочивание FP, FMA fusion |
| `-fwhole` | Целопрограммная межпроцедурная оптимизация (несовместима с -fPIC) |
| `-fwhole-shared` | Как -fwhole но совместима с -fPIC/-fPIE |
| `-march=elbrus-v5` | ISA для E8C2 |
| `-mtune=elbrus-8c2` | Тюнинг планировщика под конкретный процессор |
| `-fstrict-aliasing` | Оптимизация алиасинга |
| `-fprefetch` | Включить предвыборку данных |
| `-fcache-opt` | Блочная оптимизация кэша в конвейеризованных циклах |

### Инлайн

| Флаг | Описание |
|------|----------|
| `-finline-level=N` | Коэффициент раскрытия инлайна [0.1-20.0] |
| `-finline-scale=N` | Лимит ресурсов инлайна [0.1-5.0] |
| `-fno-inline` | Отключить весь инлайн |

### Спекулятивное исполнение

| Флаг | Описание |
|------|----------|
| `-fno-spec-fp` | Отключить спекулятивный FP (избегает SIGILL при FP исключениях) |
| `-fno-dam` | Отключить DAM table спекулятивное исполнение |

### Software pipelining (SWP)

| Флаг | Описание |
|------|----------|
| `-fswp-maxopers=N` | Макс. операций для SWP (default 300) |
| `-fforce-swp` | Форсировать попытки SWP |
| `-fno-apb` | Отключить APB механизм |

### PGO (Profile-Guided Optimization)

```bash
# Шаг 1: Сгенерировать профиль
lcc -O3 -fprofile-generate=./profile program.c -o program_prof
./program_prof  # запустить с типичной нагрузкой

# Шаг 2: Использовать профиль
lcc -O3 -fprofile-use=./profile/program.gcda program.c -o program_fast
```

---

## 3. Pragma-директивы

```c
#pragma loop count(N)     // Подсказка кол-ва итераций (включает APB)
#pragma ivdep             // Разрешить разрыв зависимостей чтение/запись
#pragma unroll(N)         // Управление фактором развёртки
```

---

## 4. Правила кода для VLIW

### 4.1. Используй int64_t для счётчиков циклов

```c
// ПЛОХО — APB не работает
for (int i = 0; i < N; i++) { ... }

// ХОРОШО — APB включается
for (int64_t i = 0; i < N; i++) { ... }
```

### 4.2. Добавляй restrict на указатели

```c
// ПЛОХО — компилятор считает что A и C могут алиаситься
void add(float* A, float* B, float* C, int N);

// ХОРОШО — 33-80% ускорение от разрешения алиасов
void add(float* __restrict A, float* __restrict B, float* __restrict C, int N);
```

Или глобально: `-frestrict-all` или `-frestrict-params`

### 4.3. Максимизируй ILP (Instruction-Level Parallelism)

VLIW может выполнить 6 FP операций параллельно. Нужны НЕЗАВИСИМЫЕ операции:

```c
// ПЛОХО — последовательная зависимость
float sum = 0;
for (int64_t i = 0; i < N; i++)
    sum += A[i];

// ХОРОШО — 4 независимых аккумулятора
float s0=0, s1=0, s2=0, s3=0;
for (int64_t i = 0; i < N; i += 4) {
    s0 += A[i]; s1 += A[i+1]; s2 += A[i+2]; s3 += A[i+3];
}
float sum = s0 + s1 + s2 + s3;
```

### 4.4. Используй #pragma loop count для APB

```c
#pragma loop count(1000)  // компилятор знает что итераций ~1000
for (int64_t i = 0; i < N; i++) {
    C[i] = A[i] * B[i];
}
// Результат: ~2x ускорение от APB предвыборки
```

### 4.5. Избегай коротких циклов

Пролог/эпилог SWP дорогой. Минимум ~20-30 итераций для окупаемости.

### 4.6. -fwhole для межмодульной оптимизации

```bash
lcc -O3 -fwhole file1.c file2.c file3.c -o program
```

---

## 5. Рекомендуемая компиляция для МАКСИМУМА

```bash
# Для PromeTorch на Elbrus-8SV:
lcc -O3 -ffast -ffast-math \
    -mtune=elbrus-8c2 \
    -fprefetch -fcache-opt \
    -frestrict-params \
    -fopenmp \
    -leml_mt -lnuma -lpthread \
    program.c -o program

# Ещё быстрее (с PGO):
lcc -O3 -ffast -fprofile-generate=./prof program.c -o program_prof
./program_prof  # тренировочный прогон
lcc -O3 -ffast -fprofile-use=./prof/program.gcda program.c -o program_fast
```

---

## Источники

- https://dev.mcst.ru/book/chapter6.html (Оптимизация)
- https://www.altlinux.org/Эльбрус/оптимизация
- https://habr.com/ru/articles/647165/ (FP оптимизация на Эльбрусе)
