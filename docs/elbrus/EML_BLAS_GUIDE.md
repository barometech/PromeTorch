# ДОКУМЕНТАЦИЯ ЭЛЬБРУСА: EML (Elbrus Math Library) — Руководство

> **ЭТО ДОКУМЕНТАЦИЯ ЭЛЬБРУСА!** EML — библиотека МЦСТ, оптимизированная под VLIW.
> Используется в PromeTorch для GEMM (cblas_sgemm) на Elbrus-8SV.

---

## 1. Что такое EML

EML — проприетарная библиотека МЦСТ, ручная оптимизация под E2K VLIW.
Использует SIMD128, APB prefetch, software pipelining.

### Модули

| Модуль | Функции | Библиотека |
|--------|---------|------------|
| Core | Аллокация, версия | libeml |
| **Vector** | Арифметика, мат.функции, статистика | libeml_vector |
| **Algebra** | **BLAS 1/2/3, LAPACK** | libeml_algebra |
| Signal | Свёртка, фильтрация, FFT | libeml_signal |
| Image | Фильтры, трансформации, DFT | libeml_image |
| Video | Интерполяция, DCT | libeml_video |
| Graphics | Примитивы рисования | libeml_graphics |
| Volume | Ray casting, воксели | libeml_volume |
| Tensor | Тензорные операции | libeml_tensor |

### Варианты линковки

| Библиотека | Описание |
|------------|----------|
| `-leml` | **Однопоточная** — для ручного OMP тайлинга |
| `-leml_mt` | **Многопоточная** — внутренний OMP |
| `-leml_algebra` | Только BLAS/LAPACK (ST) |
| `-leml_algebra_mt` | Только BLAS/LAPACK (MT) |
| `-leml_vector` | Только векторные операции (ST) |
| `-leml_vector_mt` | Только векторные операции (MT) |

---

## 2. API: BLAS GEMM

```c
#include <eml/cblas.h>

// C = alpha * A @ B + beta * C
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            alpha, A, lda, B, ldb,
            beta, C, ldc);

// C = alpha * A @ B^T + beta * C
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K,
            alpha, A, lda, B, ldb,
            beta, C, ldc);
```

### Другие BLAS функции

```c
// GEMV: y = alpha * A @ x + beta * y
cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N,
            alpha, A, lda, x, 1, beta, y, 1);

// DOT: result = x . y
float result = cblas_sdot(N, x, 1, y, 1);

// AXPY: y = alpha * x + y
cblas_saxpy(N, alpha, x, 1, y, 1);
```

### Векторные функции (EML-specific)

```c
#include <eml/eml_vector.h>

// Поэлементное сложение: C = A + B
eml_Vector_Add_32F(A, B, C, N);

// Поэлементное умножение: C = A * B
eml_Vector_Mul_32F(A, B, C, N);

// Сумма массива
float sum;
eml_Vector_Sum_32F(data, N, &sum);
```

---

## 3. Компиляция

```bash
# Однопоточная EML
cc -O3 -leml -o program program.c

# Многопоточная EML + OpenMP
cc -O3 -fopenmp -leml_mt -o program program.c

# Только BLAS
cc -O3 -fopenmp -leml_algebra_mt -o program program.c
```

---

## 4. Модель потоков

### eml_mt (многопоточная)

- Использует OMP внутри
- **НЕ уважает omp_set_num_threads()** — всегда использует все ядра
- Читает OMP_NUM_THREADS из окружения при инициализации
- На 4-socket NUMA: cross-NUMA penalty → только 152-265 GFLOPS (вместо 2304)

### eml (однопоточная)

- Один поток, без OMP внутри
- 66.4 GFLOPS на одном ядре E8C2
- Можно параллелить снаружи через OMP тайлинг:
  - 16 OMP tiles × ST EML = **360 GFLOPS** (лучший замеренный результат)

### КРИТИЧЕСКИЙ БАГ: pthread + eml_mt = SIGILL

**cblas_sgemm из eml_mt НЕЛЬЗЯ вызывать из pthread/std::thread!**
Вызов из любого потока кроме main → SIGILL (Illegal Instruction).
Работает ТОЛЬКО из main thread и из OMP parallel regions.

---

## 5. NUMA оптимизация

### Переменные окружения

```bash
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

### Аллокация на NUMA ноде

```c
#include <numa.h>

// Аллокировать на конкретной ноде
float* A = (float*)numa_alloc_onnode(size_bytes, node_id);

// Привязать поток к ноде
numa_run_on_node(node_id);

// Освободить
numa_free(A, size_bytes);
```

### Стратегия для максимума GFLOPS

1. Разбить матрицу A по строкам на 4 тайла (по числу NUMA нод)
2. Скопировать каждый тайл на свою NUMA ноду
3. Запустить OMP parallel num_threads(4)
4. Каждый поток: numa_run_on_node + cblas_sgemm на своём тайле
5. B — read-only, расшарен (в L3 кэше)

**Результат: 1840 GFLOPS (80% пика)** — но только через pthread (SIGILL!).
Через OMP: пока 360 GFLOPS (16%). Нужна доработка.

---

## 6. Документация на сервере

```bash
# Проверить наличие документации EML
ls /opt/mcst/doc/eml/
ls /usr/share/doc/eml-doc-*/

# Прочитать
cat /opt/mcst/doc/eml/index.html | head -200
```

---

## 7. Производительность EML (замеры)

| Операция | Naive | EML | Speedup |
|----------|-------|-----|---------|
| Matrix multiply (N=1024) | 1207.73s | 14.72s | **82x** |
| Vector sum (8U, N=1M) | slow | 16-19x SIMD | ~18x |
| GEMM 768×768×768 (32 cores) | - | 154 GFLOPS | - |
| GEMM 2048×2048 (1 core ST) | - | 66 GFLOPS | - |

---

## Источники

- https://dev.mcst.ru/book/chapter7.html (Библиотеки)
- https://www.altlinux.org/Эльбрус/eml
- https://habr.com/ru/articles/978730/
- /opt/mcst/doc/eml/ (на сервере)
- Наши бенчмарки: test_eml_diag.c, bench_eml_st.c, bench_numa_omp.c
