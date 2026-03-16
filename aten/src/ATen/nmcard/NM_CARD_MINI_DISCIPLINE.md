# NM Card Mini - Полное руководство
## АО НТЦ "Модуль" | Нейроускоритель на базе архитектуры NeuroMatrix

---

# 1. АППАРАТНЫЕ ХАРАКТЕРИСТИКИ

## 1.1 Процессор К1879ВМ8Я (NM6408)

| Параметр | Значение |
|----------|----------|
| Техпроцесс | 28 нм |
| Архитектура | NeuroMatrix + ARM |
| Тензорные ядра NMC4 | 16 шт @ 1000 МГц |
| RISC ядра ARM Cortex-A5 | 5 шт @ 800 МГц |
| Производительность FP32 | 512 GFLOPs |
| Производительность FP64 | 128 GFLOPs |

## 1.2 Память и интерфейсы

| Параметр | Значение |
|----------|----------|
| Оперативная память | 5 ГБ DDR3L |
| Пропускная способность памяти | до 32 ГБ/с |
| Интерфейс подключения | PCIe 2.0 x4 |
| Ethernet | 100 Мбит/с |
| Коммуникационные порты | 4 шт, суммарно до 16 ГБ/с |

## 1.3 Форм-фактор и питание

| Параметр | Значение |
|----------|----------|
| Форм-фактор | PCIe x4, занимает 1 слот |
| Максимальная мощность | 25 Вт |
| Типовая мощность | 10.5 Вт (MLPerf inference r0.5) |

## 1.4 Архитектура кластеров

Процессор К1879ВМ8Я организован как кластерная система:
- **4 вычислительных кластера**
- Каждый кластер содержит:
  - 1 RISC-процессор ARM Cortex-A5 (800 МГц)
  - 4 ядра NMC4 NeuroMatrix (1 ГГц)
- **Режимы работы:**
  - Параллельная обработка всеми кластерами
  - Независимые задачи на каждом кластере

---

# 2. ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ

## 2.1 Поддерживаемые ОС

### Windows
- Windows 7 (64-bit)
- Windows 10 (64-bit)
- Windows 11 (64-bit) - предположительно

### Linux
- Debian (64-bit)
- Ubuntu (64-bit)
- Astra Linux (сертифицирован)
- Эльбрус ОС 8С, 16С

## 2.2 NMDL (NeuroMatrix Deep Learning)

**Назначение:** Запуск предобученных глубоких нейронных сетей

### Состав SDK:

```
Библиотеки:
├── nmdl.dll / nmdl.so        - Inference движок
├── nmdl.lib                   - Статическая линковка (MSVC++)
├── nmdl.h                     - C/C++ API заголовок
│
├── nmdl_compiler.dll/.so      - Компилятор моделей
├── nmdl_compiler.lib          - Статическая линковка
├── nmdl_compiler.h            - API компилятора
│
├── nmdl_image_converter.dll   - Конвертер изображений
└── nmdl_image_converter.h     - API конвертера
```

### Поддерживаемые форматы моделей:
- **ONNX** (.onnx, .pb)
- **DarkNet** (.cfg + .weights)

### Предобученные сети (демо):
- AlexNet
- ResNet-18, ResNet-50
- SqueezeNet
- Inception v3
- YOLOv2, YOLOv3, YOLOv5s
- U-Net

## 2.3 NMDL Plus (Расширенная версия)

### Python API пакеты:

```python
# Основные модули
import compiler          # Компиляция моделей ONNX/DarkNet
import image_converter   # Подготовка изображений
import nmdlp             # Базовые команды и inference
import session           # Высокоуровневый API (как TensorFlow)

# Вспомогательные
import constants         # Константы
import datatypes         # Типы данных
```

### Новые возможности:
- Полная поддержка Python
- MobileNet v2
- YOLO v7, YOLO v7 Tiny, YOLO v8
- Тайлинг больших изображений
- Мультиязычная поддержка

### Новые операторы:
- Softmax
- DepthwiseConvolution
- Exp, Ceil, Floor
- Neg, Sqrt, Reciprocal

## 2.4 NMC SDK (Низкоуровневая разработка)

**Назначение:** Прямое программирование ядер NeuroMatrix

### Состав тулчейна:

| Компонент | Описание |
|-----------|----------|
| Ассемблер | GAS и legacy нотации |
| Компилятор | GNU C/C++ для NMC |
| Отладчик | NMC-GDB |
| Симулятор | QEMU |
| IDE | Eclipse |
| Дизассемблер | Анализ бинарного кода |
| Редактор связей | Линковка модулей |
| Стандартная библиотека | libc для NMC |

### Поддерживаемые языки:
- C / C++
- Ассемблер NeuroMatrix

---

# 3. БЫСТРЫЙ СТАРТ

## 3.1 Установка драйверов (Windows)

```powershell
# 1. Скачать драйвер с официального сайта или из комплекта поставки
# 2. Установить .exe файл драйвера
# 3. Перезагрузить систему

# Проверка установки в PowerShell:
Get-PnpDevice | Where-Object { $_.FriendlyName -like "*NM*" -or $_.FriendlyName -like "*Module*" }
```

## 3.2 Установка драйверов (Linux)

```bash
# Debian/Ubuntu
sudo dpkg -i nm-card-driver.deb

# Проверка
lspci | grep -i "module\|nm\|1879"
dmesg | grep -i nm
```

## 3.3 Установка NMDL

```bash
# Windows: запустить установщик .exe
# Linux (Debian/Ubuntu):
sudo dpkg -i nmdl-*.deb

# Проверка
nmdl --version
```

## 3.4 Первый запуск (Python)

```python
import nmdlp
import session

# Инициализация сессии
sess = session.Session()

# Загрузка модели
model = sess.load_model("resnet18.onnx")

# Подготовка изображения
img = image_converter.load("test.jpg")
img = image_converter.preprocess(img, (224, 224))

# Inference
result = sess.run(model, img)
print(f"Predicted class: {result.argmax()}")
```

## 3.5 Первый запуск (C/C++)

```cpp
#include <nmdl.h>
#include <nmdl_compiler.h>
#include <nmdl_image_converter.h>

int main() {
    // Инициализация
    nmdl_context_t ctx;
    nmdl_init(&ctx);

    // Загрузка модели
    nmdl_model_t model;
    nmdl_load_model(&ctx, "resnet18.nmbin", &model);

    // Подготовка входных данных
    float* input = prepare_input("image.jpg");

    // Inference
    float* output;
    nmdl_run(&ctx, &model, input, &output);

    // Очистка
    nmdl_destroy(&ctx);
    return 0;
}
```

---

# 4. АРХИТЕКТУРА NEUROMATRIX

## 4.1 Ядро NMC4

NMC4 (NeuroMatrix Core 4) - векторно-матричное процессорное ядро:

- **VLIW архитектура** - параллельное выполнение инструкций
- **Тензорные операции** - нативная поддержка матричных вычислений
- **Поддержка FP32/FP64** - высокоточные вычисления
- **Частота 1 ГГц** - высокая производительность

## 4.2 Распределение задач

```
┌─────────────────────────────────────────────────────┐
│                    К1879ВМ8Я                        │
├─────────────┬─────────────┬─────────────┬───────────┤
│  Кластер 0  │  Кластер 1  │  Кластер 2  │ Кластер 3 │
├─────────────┼─────────────┼─────────────┼───────────┤
│ ARM Cortex  │ ARM Cortex  │ ARM Cortex  │ARM Cortex │
│    A5       │    A5       │    A5       │   A5      │
├─────────────┼─────────────┼─────────────┼───────────┤
│ NMC4 x 4    │ NMC4 x 4    │ NMC4 x 4    │ NMC4 x 4  │
└─────────────┴─────────────┴─────────────┴───────────┘
```

## 4.3 Модель программирования

1. **Хост (PC)** - подготовка данных, загрузка модели
2. **ARM ядра** - управление, диспетчеризация задач
3. **NMC4 ядра** - тяжелые вычисления (свертки, матрицы)

---

# 5. ПОДДЕРЖИВАЕМЫЕ НЕЙРОСЕТИ

## 5.1 Классификация изображений

| Сеть | Входной размер | Top-1 Accuracy |
|------|----------------|----------------|
| AlexNet | 227x227 | ~57% |
| SqueezeNet | 227x227 | ~58% |
| ResNet-18 | 224x224 | ~70% |
| ResNet-50 | 224x224 | ~76% |
| Inception v3 | 299x299 | ~78% |
| MobileNet v2 | 224x224 | ~72% |

## 5.2 Детекция объектов

| Сеть | Входной размер | mAP |
|------|----------------|-----|
| YOLOv2 | 416x416 | ~76% |
| YOLOv3 | 416x416 | ~80% |
| YOLOv5s | 640x640 | ~37% (COCO) |
| YOLOv7 | 640x640 | ~51% (COCO) |
| YOLOv7-tiny | 640x640 | ~38% (COCO) |
| YOLOv8 | 640x640 | ~53% (COCO) |

## 5.3 Сегментация

| Сеть | Применение |
|------|------------|
| U-Net | Медицинская визуализация |

---

# 6. ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ

## 6.1 Использование памяти

```python
# Минимизация копирования данных
input_buffer = nmdlp.allocate_device_memory(size)
nmdlp.copy_to_device(input_buffer, host_data)

# Использование пайплайна
pipeline = session.Pipeline()
pipeline.add_stage(preprocess)
pipeline.add_stage(inference)
pipeline.add_stage(postprocess)
pipeline.run_async()
```

## 6.2 Батчинг

```python
# Обработка нескольких изображений
batch_size = 4
inputs = [load_image(f"img_{i}.jpg") for i in range(batch_size)]
results = sess.run_batch(model, inputs)
```

## 6.3 Квантизация

- FP32 → INT8 для ускорения inference
- Поддержка mixed precision

---

# 7. ДИАГНОСТИКА И ОТЛАДКА

## 7.1 Проверка оборудования

### Windows PowerShell:
```powershell
# Список PCIe устройств
Get-PnpDevice -Class "System" | Where-Object Status -eq "OK"

# Диспетчер устройств
devmgmt.msc
```

### Linux:
```bash
# PCIe устройства
lspci -v | grep -A 10 "Module\|1879\|NM"

# Системные сообщения
dmesg | grep -i "nm\|neuromatrix\|module"

# Загруженные модули
lsmod | grep nm
```

## 7.2 Тестирование производительности

```python
import time

# Бенчмарк inference
start = time.time()
for _ in range(100):
    result = sess.run(model, input_data)
elapsed = time.time() - start

fps = 100 / elapsed
print(f"Performance: {fps:.2f} FPS")
print(f"Latency: {elapsed/100*1000:.2f} ms")
```

---

# 8. КОНТАКТЫ И ПОДДЕРЖКА

## Официальные ресурсы

| Ресурс | Контакт |
|--------|---------|
| Сайт | https://www.module.ru |
| Продукт | https://www.module.ru/directions/iskusstvennyj-intellekt/nm-card-mini |
| Техподдержка | nm-support@module.ru |
| Продажи | rusales@module.ru |
| Телефон | +7 (495) 531-30-80 |
| Адрес | г. Москва, 4-я ул 8 Марта, д.3 |

## Получение ПО

1. **NMDL/NMDL Plus** - запрос на nm-support@module.ru
2. **NMC SDK** - скачать с официального сайта
3. **Драйверы** - в комплекте поставки или по запросу

## Удаленный доступ для тестирования

НТЦ "Модуль" предоставляет удаленный доступ к оборудованию для тестирования.
Заявка: rusales@module.ru или nm-support@module.ru

---

# 9. СОВМЕСТИМОСТЬ

## Реестры и сертификация

- **Реестр российской промышленной продукции** (ПП РФ 719)
- **Единый реестр радиоэлектронной продукции** (ПП РФ 878)
- Подходит для государственных закупок по программе импортозамещения

## Совместимые продукты НТЦ "Модуль"

| Продукт | Процессоры | Назначение |
|---------|------------|------------|
| NM Card | 1x К1879ВМ8Я | Рабочие станции |
| NM Card Mini | 1x К1879ВМ8Я | Компактные ПК |
| NM Mezzo Mini | 1x К1879ВМ8Я | Встраиваемые системы |
| NM Quad | 4x К1879ВМ8Я | Серверы AI |

---

*Документ создан на основе официальных источников НТЦ "Модуль"*
*Версия: 1.0 | Дата: 2025*
