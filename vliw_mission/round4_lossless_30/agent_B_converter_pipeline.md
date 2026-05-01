# Agent B — Converter pipeline GGUF → .pt8 + CLI utility

## Роль
Инженер pipeline. Реализует `gguf2pt8` CLI — bit-lossless конвертер
из GGUF в новый формат PromeTorch.

## Зависимость
**ЖДЁТ** `agent_A_feasibility_format_design.md` → format_spec_v1.md.
Без полной спеки формата работать не может, но может **параллельно**
проектировать общую архитектуру converter'а исходя из текущего GGUF
parsing-кода.

## Что прочитать
1. `vliw_mission/round4_lossless_30/MISSION.md`
2. `torch/io/gguf_reader.h` (~500 строк) — как парсится GGUF
3. `torch/io/gguf_model.h:1100-1300` — `load_quantized_to_cpu`,
   `load_quantized_to_gpu`, mmap path
4. `torch/io/q8_soa_repack.h:90-220` — `repack_q4k_to_q8soa4` как образец
   что лежит в основе нашего "lossless repack"
5. `examples/gguf/test_gguf_inference.cpp` — entry point existing
6. `tools/` — есть ли уже CLI tools? Обследовать.

## Что должен сделать

### Часть 1: Архитектура CLI
1. **Single binary** `prometorch-convert` (или `gguf2pt8`)
2. **Командный интерфейс:**
   ```
   prometorch-convert <input.gguf> [-o output.pt8]
                      [--format pt8|pt8_full|...]
                      [--validate]   (read back и сравни logits на test prompt)
                      [--threads N]
                      [--progress]   (progress bar)
                      [-v|--verbose]
   ```
3. **Output codes:** 0 success, 1 user error, 2 input format error,
   3 disk write error, 4 validation failed.

### Часть 2: Pipeline implementation
```
GGUF reader (mmap) ─┐
                    ├─→ tensor iterator ─→ per-tensor repack ─→ .pt8 writer
metadata extractor ─┘                          │
                                               └─→ progress callback
```
- Многопоточная конверсия (один tensor на thread)
- Mmap input GGUF (zero-copy read)
- Streaming write output (не держим всё в RAM)
- Progress bar через `\r` или ANSI escape codes
- Logging via stderr / stdout split

### Часть 3: Integration tests
1. Small sanity test: создать tiny GGUF (1 tensor 256×256 Q4_K), конвертнуть,
   прочитать обратно через `pt8_reader.h`, сверить bytes и logits.
2. Real test: `qwen3-4b-Q4_K_M.gguf` → `qwen3-4b.pt8`, диф между логитами
   на одном prompt < 1e-5 max abs.
3. Performance: время конверсии < 30s для 4B params.

### Часть 4: Build integration
- `tools/CMakeLists.txt` добавить новый target
- Cross-compile для x86_64 (dev) + e2k (Эльбрус) — обе платформы
- Static linking где возможно

### Часть 5 (опционально): GUI
Если есть время и Agent A одобрил, простой GUI:
- **Tauri** (Rust + web frontend) — cross-platform, но Rust toolchain на Эльбрусе нужен
- **Qt 5/6 (QML)** — Эльбрус OS обычно идёт с Qt
- **Web-based** (HTTP server + статика) — самое простое, любая ОС
   - PromeServe-style: один binary, embedded HTML/JS, drag-and-drop GGUF файла
- Тёмная тема, минимализм: drag-and-drop, кнопка Convert, прогресс, готово

## Output формат

Файлы:
1. `tools/gguf2pt8/main.cpp` — CLI entry point
2. `tools/gguf2pt8/converter.h` — core convert logic
3. `tools/gguf2pt8/CMakeLists.txt`
4. `torch/io/pt8_reader.h` — read .pt8 в model loader
5. `tools/gguf2pt8/README.md` — usage, examples
6. (опционально) `tools/gguf2pt8/gui/` — GUI source

Каждый файл должен компилироваться `cmake --build build_elbrus --target gguf2pt8`.

## Constraints
- НЕ полагайся на `agent_A` спеку до того как она готова — параллельно
  делай инфраструктуру (CLI parsing, mmap, threads), затем
  подключай specific encoding logic
- Pure stdlib + already-existing headers PromeTorch. Никаких новых
  зависимостей.
- Тестируй конверсию на реальном `qwen3-4b-Q4_K_M.gguf` перед коммитом.
