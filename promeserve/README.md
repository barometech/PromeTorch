# PromeServe — Ollama-совместимый LLM сервер на PromeTorch

## Обзор

PromeServe — inference сервер для LLM моделей, совместимый с Ollama API. Подключается к Open WebUI, LangChain, curl без модификаций. Построен на PromeTorch C++ (без Python runtime).

## Запуск

```bash
# GPU
./promeserve --port 11434 --device cuda --model qwen3:4b

# CPU
./promeserve --port 11434 --device cpu
```

Web UI: откройте `http://localhost:11434` в браузере.

## API (Ollama-совместимый)

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Health check |
| GET | `/api/version` | Версия сервера |
| GET | `/api/tags` | Список доступных моделей |
| POST | `/api/show` | Информация о модели |
| POST | `/api/generate` | Генерация текста (стриминг) |
| POST | `/api/chat` | Чат (стриминг) |

### Пример: генерация

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:4b",
  "prompt": "Что такое нейросеть?",
  "stream": true
}'
```

### Пример: чат

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:4b",
  "messages": [{"role": "user", "content": "Привет!"}],
  "stream": true
}'
```

## Поддерживаемые форматы моделей

- **GGUF** — нативный формат Ollama (квантизованные модели)
- **SafeTensors** — HuggingFace формат
- **PyTorch** — .bin/.pt файлы
- **ONNX** — .onnx файлы

Модели обнаруживаются автоматически из `~/.ollama/manifests/`.

## Web UI

Встроенный чат-интерфейс (`web/index.html`):
- Тёмная тема
- Стриминг ответов
- Markdown рендеринг с подсветкой кода
- История чатов (localStorage)
- Настройки: temperature, top_p, top_k, max tokens, system prompt

## Архитектура

```
PromeServe
├── main.cpp          — CLI entry point
├── promeserve.h      — главный класс, координирует всё
├── http_server.h     — HTTP/1.1 сервер (raw sockets, без зависимостей)
├── api_handlers.h    — обработчики REST endpoints
├── model_manager.h   — сканирование и загрузка моделей
├── model_loader.h    — универсальный загрузчик (GGUF/SafeTensors/PyTorch/ONNX)
└── web/index.html    — браузерный GUI
```

Нет внешних зависимостей кроме PromeTorch core (c10, aten) и опционально CUDA.

## Сборка

```bash
cd build_gguf_cuda
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON
nmake promeserve
```

## Текущие ограничения

- GPU inference: 34.8 tok/s (vs Ollama 161.9 tok/s) — нужна cuBLAS FP16 интеграция
- Повторные GPU запросы могут зависать (VRAM allocation bug)
- CPU inference: 2-3x медленнее llama.cpp (нет MKL Q4_K GEMV)
