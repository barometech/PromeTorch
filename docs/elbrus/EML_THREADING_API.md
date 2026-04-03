# ДОКУМЕНТАЦИЯ ЭЛЬБРУСА: EML Threading API — КРИТИЧЕСКОЕ ОТКРЫТИЕ

> **ЭТО ДОКУМЕНТАЦИЯ ЭЛЬБРУСА!** Из `/usr/include/eml/eml_core.h` на сервере.
> Найдено 2026-04-03. Ключ к решению проблемы утилизации GFLOPS.

---

## КРИТИЧЕСКОЕ: eml_SetNumThreads / eml_GetNumThreads

EML имеет **СВОЙ** API управления потоками, НЕЗАВИСИМЫЙ от `omp_set_num_threads()`.

### eml_SetNumThreads

```c
#include <eml/eml_core.h>

// Устанавливает число OMP потоков ВНУТРИ EML
// НЕ зависит от omp_set_num_threads!
eml_Status eml_SetNumThreads(eml_32s NumThreads);
// Возвращает: EML_OK или EML_INVALIDPARAMETER (если <= 0)
```

### eml_GetNumThreads

```c
// Возвращает текущее число потоков EML
eml_Status eml_GetNumThreads(eml_32s* pNumThreads);
```

---

## Что это значит

1. `omp_set_num_threads(1)` в нашем коде **НЕ ВЛИЯЕТ на EML**.
2. EML всегда использует столько потоков, сколько установлено через `eml_SetNumThreads()` или `OMP_NUM_THREADS` при инициализации.
3. Для NUMA тайлинга нужно:
   - Вызвать `eml_SetNumThreads(8)` в каждом OMP tile (8 ядер на NUMA ноду)
   - Или `eml_SetNumThreads(1)` для полностью ручного тайлинга 32 потоками

---

## Другие полезные функции из eml_core.h

```c
// Аллокация выровненной памяти (16 байт на E8C2 v5+)
void* eml_Malloc(eml_addr length);
void  eml_Free(void* pointer);

// Версия библиотеки
char* eml_GetVersion(void);

// Оптимальное выравнивание
#define eml_GetPreferredBufferAlignment 16  // для E8C2 (v5)
```

---

## Источник

Файл: `/usr/include/eml/eml_core.h`
Copyright: (c) 2006-2024 AO "MCST"
