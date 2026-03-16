# NM Card Mini - Инструкция по перезапуску карты

## ПРОБЛЕМА

Карта может зависнуть или перестать отвечать. Симптомы:
- `PL_GetBoardCount()` возвращает 0
- `nm_card_run` выдаёт "Failed reset board!"
- `PL_GetBoardDesc()` зависает

## РЕШЕНИЕ (100% РАБОТАЕТ)

### ⚠️ ТРЕБУЮТСЯ ПРАВА АДМИНИСТРАТОРА! ⚠️

**Все способы ниже работают ТОЛЬКО от имени администратора!**
Без прав админа - ничего не сработает.

### Способ 1: PowerShell скрипт (РЕКОМЕНДУЕТСЯ)

Запустить PowerShell **от имени администратора** (ПКМ → Run as Administrator) и выполнить:

```powershell
# Найти и перезапустить NM Card
$device = Get-PnpDevice | Where-Object { $_.InstanceId -like '*17CD*' }
Write-Host "Found: $($device.InstanceId)"
Write-Host "Status: $($device.Status)"

# Отключить
Disable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false
Start-Sleep -Seconds 2

# Включить
Enable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false
Start-Sleep -Seconds 3

# Проверить
$device = Get-PnpDevice | Where-Object { $_.InstanceId -like '*17CD*' }
Write-Host "New status: $($device.Status)"
```

### Способ 2: Готовый скрипт

Файл `restart_device.ps1` уже создан в проекте. Запуск:

```batch
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy Bypass -File restart_device.ps1
```

### Способ 3: Device Manager (вручную)

1. Win+X → Device Manager
2. Найти "NeuroMatrix Accelerators" → "NM_Card"
3. ПКМ → Disable device
4. Подождать 2 секунды
5. ПКМ → Enable device
6. Подождать 3 секунды

## ПРОВЕРКА ЧТО КАРТА РАБОТАЕТ

```python
import ctypes, os

# Добавить пути
for path in [r'C:\Program Files\Module\NM_Card\libload\bin']:
    os.environ['PATH'] = path + ';' + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'): os.add_dll_directory(path)

# Загрузить DLL
nm = ctypes.CDLL(r'C:\Program Files\Module\NM_Card\libload\bin\nm_card_load.dll')

# Проверить
nm.PL_GetBoardCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]
nm.PL_GetBoardCount.restype = ctypes.c_int
count = ctypes.c_uint()
r = nm.PL_GetBoardCount(ctypes.byref(count))

if r == 0 and count.value > 0:
    print(f"OK! Найдено карт: {count.value}")
else:
    print("ОШИБКА: карта не найдена")
```

## ВАЖНЫЕ ДАННЫЕ КАРТЫ

| Параметр | Значение |
|----------|----------|
| Instance ID | `PCI\VEN_17CD&DEV_0002&SUBSYS_000017CD&REV_00\4&131C662&0&000B` |
| Vendor ID | 17CD (RC Module) |
| Device ID | 0002 |
| Class | NeuroMatrix Accelerators |
| Friendly Name | NM_Card |
| Serial | 122 |
| Firmware | 2.1 |

## КОГДА НУЖЕН REBOOT

PowerShell disable/enable НЕ поможет если:
- pnputil говорит "требуется перезагрузка"
- Устройство полностью удалено через `pnputil /remove-device`
- Драйвер в состоянии "pending removal"
- **PL_ResetBoard возвращает ERROR=1** (даже если PL_GetBoardDesc работает!)

### ВАЖНО: PL_ResetBoard = ERROR

Если карта "видна" (PL_GetBoardCount=1, Serial читается), но `PL_ResetBoard()` возвращает ERROR - это значит **программа на карте зависла** (например, бесконечный цикл `while(1)`).

**Disable/Enable НЕ сбросит firmware!** Нужен полный **REBOOT**.

Симптомы:
```
PL_GetBoardCount: 0 (OK), count=1  ✓
PL_GetBoardDesc: 0 (OK)            ✓
PL_GetSerialNumber: 122            ✓
PL_GetFirmwareVersion: 2.1         ✓
PL_ResetBoard: 1 (ERROR)           ✗ <-- ЗАВИСЛА!
```

В этих случаях нужен полный **REBOOT** компьютера.

## ПРЕДОТВРАЩЕНИЕ ЗАВИСАНИЙ

1. **Всегда используй таймаут:**
   ```
   nm_card_run -c0 -n0 -t5000 program.abs
   ```

2. **Закрывай дескриптор:**
   ```python
   nm.PL_CloseBoardDesc(board)
   ```

3. **Не прерывай Ctrl+C** во время выполнения

4. **Один процесс** - не запускай два nm_card_run одновременно

## АВТОМАТИЗАЦИЯ

Скрипт `unlock_card.py` для быстрого исправления:

```python
import subprocess, time

INSTANCE_ID = r"PCI\VEN_17CD&DEV_0002&SUBSYS_000017CD&REV_00\4&131C662&0&000B"

print("Disabling NM Card...")
subprocess.run(['powershell', '-Command',
    f'Disable-PnpDevice -InstanceId "{INSTANCE_ID}" -Confirm:$false'],
    timeout=30)
time.sleep(2)

print("Enabling NM Card...")
subprocess.run(['powershell', '-Command',
    f'Enable-PnpDevice -InstanceId "{INSTANCE_ID}" -Confirm:$false'],
    timeout=30)
time.sleep(3)

print("Done! Check with: python check_board_v3.py")
```

---

## РЕШЕНИЕ БЕЗ REBOOT - OP_EXIT! (ГЛАВНОЕ ОТКРЫТИЕ)

### Проблема
Если программа зависла в `while(1)`, карта не отвечает на `PL_ResetBoard`.

### Решение
**PL_GetAccess() работает БЕЗ reset!** Это значит можно записать команду в память!

Если dispatcher написан с командой `OP_EXIT`:
```cpp
#define OP_EXIT 255

while (mem[CMD] != OP_EXIT) {
    // ... обработка команд ...
}
return 0;  // Программа завершается!
```

То можно **записать OP_EXIT в память** и программа сама завершится:
```python
# Python - аварийный выход БЕЗ reboot!
dev.get_access()  # Работает без reset!
dev.write_mem(CMD_ADDR, OP_EXIT)  # Программа завершится!
```

### Файлы

| Файл | Описание |
|------|----------|
| `nmc_programs/dispatcher_safe.cpp` | Dispatcher с OP_EXIT |
| `nmruntime/device_safe.py` | Python API с emergency_exit() |

### Использование

```python
from nmruntime.device_safe import DeviceSafe

# Нормальная работа
with DeviceSafe() as dev:
    dev.load_dispatcher()
    dev.ping(42)
    dev.shutdown()  # Корректное завершение

# Аварийный выход (если зависла)
dev = DeviceSafe()
dev.emergency_exit()  # Записывает OP_EXIT без reset!
dev.close()
```

### ВАЖНО

**Это работает ТОЛЬКО если:**
1. Dispatcher имеет проверку `if (cmd == OP_EXIT) break;`
2. Программа НЕ зависла в бесконечном цикле БЕЗ проверки команды

**НЕ поможет если:**
- Программа имеет `while(1)` без проверки `mem[CMD]`
- Программа зависла в вычислениях (не читает память)

---

*Создано 2026-01-22. Обновлено с решением OP_EXIT!*
