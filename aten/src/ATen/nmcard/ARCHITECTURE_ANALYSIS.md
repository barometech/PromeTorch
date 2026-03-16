# Архитектурный Анализ: Официальный API vs Наша Реализация

## Дата анализа: 2026-01-22

---

## 1. ОФИЦИАЛЬНЫЙ СПОСОБ (RC Module documentation)

### На карте (NMC4):
```cpp
#include "nm6408load_nmc.h"

// Данные в ЛОКАЛЬНОЙ памяти (NMMB, 512KB)
WORD32 input[SIZE];
WORD32 output[SIZE];

int main() {
    while (1) {
        // БЛОКИРУЮЩИЙ вызов - ждёт пока хост ответит
        // Использует IPC регистры 0x18018000
        int cmd = ncl_hostSyncArray(VALUE, input, SIZE, NULL, NULL);

        if (cmd == EXIT_CMD) break;

        // Вычисления...

        // Отправить результат
        ncl_hostSyncArray(RESULT, output, SIZE, NULL, NULL);
    }
    return 0;
}
```

### На хосте (Windows):
```cpp
#include "nm_card_load.h"

// ОБЯЗАТЕЛЬНО перед загрузкой программы:
PL_ResetBoard(board);
PL_LoadInitCode(board);
PL_LoadProgramFile(access, "program.abs");

// Барьерная синхронизация
int ret_value;
PL_Addr addr;
PL_Word len;

PL_SyncArray(access, CMD, 0, 0, &ret_value, &addr, &len);

// Чтение/запись по адресу от карты
PL_ReadMemBlock(access, buffer, addr, len);
PL_WriteMemBlock(access, data, addr, len);

// Завершение
PL_Sync(access, EXIT_CMD, NULL);
```

---

## 2. НАША РЕАЛИЗАЦИЯ (DDR Polling)

### На карте:
```cpp
// Прямой доступ к DDR памяти
volatile unsigned int* mem = (volatile unsigned int*)0x00340000;

int main() {
    while (1) {
        // POLLING - не блокируется, крутится в цикле
        if (mem[0] == OP_MATMUL) {
            // Читаем параметры из DDR
            unsigned int M = mem[1];
            unsigned int K = mem[2];
            // ...

            // Вычисления...

            // Пишем статус
            mem[STATUS_ADDR] = 1;  // done
            mem[0] = 0;            // ready
        }
    }
    return 0;
}
```

### На хосте (Python):
```python
nm.PL_ResetBoard(board)
nm.PL_LoadInitCode(board)
nm.PL_LoadProgramFile(access, "dispatcher.abs")

# Пишем команду и данные напрямую в DDR
cmd = [OP_MATMUL, M, K, N, addr_A, addr_B, addr_C] + [0]*25
nm.PL_WriteMemBlock(access, cmd, DDR_BASE, 32)

# Polling статуса
while True:
    status = nm.PL_ReadMemBlock(access, status_buf, DDR_BASE + STATUS_ADDR, 1)
    if status == 1:  # done
        break
    time.sleep(0.01)

# Читаем результат
nm.PL_ReadMemBlock(access, result, addr_C, M*N)
```

---

## 3. КЛЮЧЕВЫЕ ОТЛИЧИЯ

| Аспект | Официальный | Наша реализация |
|--------|-------------|-----------------|
| **IPC механизм** | Аппаратные IPC регистры `0x18018000` | Общая DDR память `0x00340000` |
| **Синхронизация kernel** | Blocking `ncl_hostSyncArray()` | Busy-wait `while(mem[0]==0)` |
| **Синхронизация host** | `PL_SyncArray()` с барьером | Polling через `PL_ReadMemBlock()` |
| **Расположение данных** | Локальная NMMB (512KB) | DDR (5GB) |
| **Атомарность** | Гарантирована hardware | Не гарантирована |
| **Энергоэффективность** | Высокая (blocking) | Низкая (busy-wait) |
| **Cache coherency** | Автоматически через IPC | Через DMA |

---

## 4. ДИЗАССЕМБЛИРОВАННЫЙ IPC

Из `libnm6408load_nmc.a`:

```asm
_ncl_hostSyncArray:
    ; ...
    ar0 = ar5 + 0x18018000 addr    ; БАЗОВЫЙ АДРЕС IPC
    [ar5+0x1801800a]=gr4           ; Длина
    [ar5+0x1801800b]=ar1           ; Адрес массива
    [ar5+0x18018009]=gr3           ; Sync value
    [ar1] = 0x1                    ; Trigger (запуск)

wait_loop:
    gr6 = [ar5+0x1801800c]         ; Читаем статус
    if gr6 == 0 goto wait_loop     ; Ждём ответа от хоста

    ; Получаем возвращаемые значения
    gr7 = [ar0+0xd]                ; Return value
    gr0 = [ar0+0xe]                ; Return address
    gr0 = [ar0+0xf]                ; Return length
```

**IPC регистры (обнаружено при дизассемблировании):**
- `0x18018008` - Trigger (запуск)
- `0x18018009` - Sync value
- `0x1801800a` - Длина массива
- `0x1801800b` - Адрес массива
- `0x1801800c` - Status (ожидание)
- `0x1801800d` - Return value
- `0x1801800e` - Return address
- `0x1801800f` - Return length

---

## 5. ПОЧЕМУ НАША РЕАЛИЗАЦИЯ РАБОТАЕТ

1. **DMA через PCIe** - `PL_WriteMemBlock`/`PL_ReadMemBlock` делают DMA операции которые автоматически обеспечивают cache coherency

2. **Нет конкурентного доступа** - В каждый момент либо хост пишет, либо kernel читает

3. **QEMU эмулятор** - Не имеет проблем с timing, виртуальная память

4. **Простые операции** - Для простых последовательных операций race conditions редки

---

## 6. КОГДА МОЖЕТ НЕ РАБОТАТЬ

1. **Race conditions** при быстрой последовательности операций
2. **Ctrl+C** прерывает хост, kernel остаётся в бесконечном цикле
3. **Timeout** - kernel не имеет механизма выхода по timeout
4. **Многопоточность** - Если хост делает concurrent доступ

---

## 7. ОШИБКА В build_gas.bat (ИСПРАВЛЕНО)

**Было:**
```batch
set LIBPATH=%IDE%\nmc4\lib    ; НЕ СУЩЕСТВУЕТ!
```

**Стало:**
```batch
set LIBPATH=C:\Program Files\Module\NM_Card\libload\lib
set INCPATH=C:\Program Files\Module\NM_Card\libload\include
```

---

## 8. РЕКОМЕНДАЦИИ

### Для прототипа (текущий статус):
✅ Наша реализация **достаточна** для:
- Тестирования и разработки
- QEMU эмуляции
- Простых последовательных операций

### Для production:
⚠️ Рекомендуется переписать на официальный sync API:
- Большая надёжность
- Лучшая энергоэффективность
- Соответствие документации

### Минимальные улучшения текущей реализации:
1. Добавить watchdog timeout в kernel
2. Добавить команду EXIT для graceful shutdown
3. Добавить retry логику в хост

---

## 9. ФАЙЛЫ ДЛЯ СРАВНЕНИЯ

| Файл | Описание |
|------|----------|
| `matmul_custom.cpp` | Наша реализация (DDR polling) |
| `matmul_official.cpp` | **НОВЫЙ** - Официальный sync API |
| `test_official_sync.py` | **НОВЫЙ** - Тест официального API |
| `dispatcher.cpp` | Наш unified dispatcher |

---

## 10. СТРУКТУРА ПАМЯТИ

### NMMB (локальная, 512KB):
```
0x00000800 - 0x0007FFFF
├── .text (код программы)
├── .data (глобальные переменные)
├── .bss  (неинициализированные)
└── .stack (стек)
```

### EMI_CPU (DDR, 5GB):
```
0x00340000 - 0x1FFC0000
├── [0..31]  - Command block (наш протокол)
├── [32...]  - Data arrays
```

### IPC Registers (hardware):
```
0x18018000 - 0x1801800F
├── Trigger, value, address, length
└── Return value, address, length
```
