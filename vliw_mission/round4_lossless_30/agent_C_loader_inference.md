# Agent C — .pt8 loader + integration в forward path

## Роль
Интеграционный инженер. Подключает новый формат к существующему
`gguf_model.h::GGUFModel` так, чтобы `init_tensor_parallel()` мог
загружать `.pt8` файл вместо `.gguf` БЕЗ изменения forward decode logic.

## Зависимости
**ЖДЁТ** Agent A (format spec) и Agent B (writer + reader stub).

## Что прочитать
1. `vliw_mission/round4_lossless_30/MISSION.md`
2. `torch/io/gguf_model.h:730-960` — `load_quantized_to_cpu` / mmap path
3. `torch/io/gguf_model.h:4180-4410` — TPSlicedWeight + Q8_SOA repack
   (вот в эту точку нужно вставить новый формат)
4. `torch/io/gguf_model.h:4570-5100` — forward_decode_cpu_tp,
   текущие SoA branches (как использовать загруженные веса)
5. `torch/io/q8_soa_repack.h` — образец Q8SoA4 struct, lifecycle

## Что должен сделать

### Часть 1: Spec -> impl reader
Прочитать формат spec от Agent A. Реализовать:
```cpp
// torch/io/pt8_reader.h
struct PT8Header { ... };
struct PT8TensorRecord { ... };
class PT8Reader {
public:
    bool open(const std::string& path);
    const PT8Header& header() const;
    std::vector<PT8TensorRecord> tensors() const;
    // mmap-based zero-copy access:
    const void* tensor_data(const std::string& name) const;
    size_t tensor_size(const std::string& name) const;
};
```

### Часть 2: Loader integration
1. Detect .pt8 vs .gguf по magic bytes на open
2. Если .pt8:
   - Skip `init_tensor_parallel`'s repack step (формат уже в SoA-friendly
     виде с Agent A spec)
   - Прямая mmap → `Q8SoA4` struct без копирования
3. Если .gguf:
   - Старый путь (репак на лету) — fallback для совместимости
4. Добавить env var `PT_FORMAT_AUTO=1` (default ON) — auto-detect

### Часть 3: Memory savings от direct .pt8 load
Поскольку формат **уже в SoA-friendly виде**, мы:
- Не делаем repack на load (-7s загрузка)
- Не держим original Q4_K cpu_data + Q8 SoA4 mem параллельно
  (-1.7 GB / rank, освобождаем)
- mmap'им файл MAP_PRIVATE | MAP_POPULATE → прямой доступ

### Часть 4: Sanity check
- Загрузить .pt8 → бенчмарк → должно совпадать с .gguf+repack путь
- Проверить numerics: max diff между .pt8 и .gguf+repack логитами < 1e-5

## Output

Файлы:
1. `torch/io/pt8_reader.h` (~200 строк)
2. Patch in `torch/io/gguf_model.h` (~150 строк изменений в init_tensor_parallel + load functions)
3. Tests: `tests/io/test_pt8_loader.cpp`
4. Документация: `torch/io/PT8_FORMAT.md` (со ссылкой на Agent A spec)

## Constraints
- forward_decode_cpu_tp **не должна меняться** — только loader.
  SoA branches уже работают с Q8SoA4 struct, это интерфейс.
- mmap path должен быть NUMA-aware: на TP-4 каждый ранк делает свой mmap
  с `MADV_RANDOM` или специфичный per-NUMA-node placement.
- backward compat: старые `.gguf` файлы должны продолжать работать.
