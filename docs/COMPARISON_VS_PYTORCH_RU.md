# PromeTorch vs PyTorch — честное сравнение

## Что такое PromeTorch

PromeTorch — фреймворк глубокого обучения, написанный с нуля на C++17/CUDA/Python. ~91,000 строк кода. Повторяет архитектуру PyTorch (c10 → ATen → torch), но с поддержкой российского железа: Эльбрус-8СВ, NM Card Mini, NM QUAD.

Не обёртка над PyTorch. Не форк. Полностью свой код: tensor runtime, autograd engine, 50+ backward functions, 80+ NN модулей, 10 оптимизаторов, CUDA kernels, cuDNN, FlashAttention.

---

## Где PromeTorch НЕ уступает PyTorch

| Возможность | PromeTorch | PyTorch |
|-------------|-----------|---------|
| Autograd engine | ✅ C++ DAG, topological sort | ✅ C++/Python hybrid |
| Linear, Conv1d/2d/3d | ✅ | ✅ |
| BatchNorm, LayerNorm, RMSNorm | ✅ | ✅ |
| LSTM, GRU, RNN | ✅ 98.44% LSTM | ✅ |
| Transformer, MultiheadAttention | ✅ | ✅ |
| Adam, SGD, RMSprop, AdamW | ✅ | ✅ |
| 9 LR schedulers | ✅ | ✅ |
| DataLoader, Sampler | ✅ | ✅ |
| Serialization (save/load) | ✅ binary PTOR | ✅ pickle |
| Gradient checkpointing | ✅ | ✅ |
| Custom autograd functions | ✅ CRTP | ✅ |
| Hooks (forward, backward) | ✅ | ✅ |
| Channels-last memory format | ✅ | ✅ |
| CUDA kernels | ✅ собственные | ✅ |
| cuDNN integration | ✅ | ✅ |
| Mixed precision (AMP) | ✅ GradScaler+Autocast | ✅ |
| MNIST accuracy | ✅ 97.65% | ✅ ~98% |
| CPU SIMD (AVX2) | ✅ PromeBLAS | ✅ MKL/OpenBLAS |
| Эльбрус-8СВ | ✅ **10% быстрее PyTorch** | ⚠️ через OpenBLAS |
| NM Card Mini | ✅ **единственный** | ❌ нет |
| NM QUAD (64 ядра) | ✅ **единственный** | ❌ нет |

---

## Где PromeTorch УСТУПАЕТ PyTorch

### 1. Экосистема и сообщество
| | PromeTorch | PyTorch |
|--|-----------|---------|
| Разработчиков | 1 человек | 1000+ (Meta + community) |
| Пакетов/расширений | 0 | 10,000+ (torchvision, torchaudio, HuggingFace, etc.) |
| Документация | 4 файла RU | docs.pytorch.org (тысячи страниц) |
| Tutorials | 0 | сотни |
| Stack Overflow ответов | 0 | 100,000+ |
| pip install | ❌ | `pip install torch` |

### 2. Производительность на стандартном железе
| Операция | PromeTorch vs PyTorch | Причина |
|---------|----------------------|---------|
| Element-wise ops (add, mul) | **8-11x медленнее** | Tensor allocation overhead |
| nn_lstm | **19x медленнее** | PyTorch: cuDNN LSTM, мы: скалярный |
| Autograd overhead | **6.6x медленнее** | PyTorch: оптимизированный C++ graph |
| Small tensor ops | **5-10x медленнее** | PyTorch: memory pool + operator fusion |
| mm 2048×2048 | **0.97x** (паритет) | Оба используют BLAS |
| Reductions (sum, var) | **0.14-0.43x** (быстрее!) | Наш AVX2 быстрее |

**Общий CPU benchmark: 1.47x медленнее** (PyTorch быстрее на 50 тестах взвешенно).
Побеждаем на 15/50 тестов (reductions, dot, mv, tanh).

### 3. Функциональность
| Фича | PromeTorch | PyTorch |
|------|-----------|---------|
| Dynamic shapes | ⚠️ базовый | ✅ полный |
| torch.compile / JIT | ❌ нет | ✅ TorchScript, Dynamo, Inductor |
| Distributed (DDP, FSDP) | ❌ заглушка | ✅ NCCL, Gloo, MPI |
| ONNX export | ❌ нет | ✅ |
| Quantization (PTQ, QAT) | ⚠️ базовый INT8 | ✅ полный |
| torch.fx / graph transforms | ❌ нет | ✅ |
| Profiler | ⚠️ базовый | ✅ TensorBoard, Chrome trace |
| Mobile (iOS, Android) | ❌ нет | ✅ PyTorch Mobile |
| torch.distributed.rpc | ❌ нет | ✅ |
| Autograd higher-order grads | ❌ нет | ✅ |
| Complex tensor support | ❌ нет | ✅ |
| Sparse tensors | ⚠️ базовый | ✅ полный (COO, CSR, CSC) |
| Named tensors | ❌ нет | ✅ |
| Nested tensors | ❌ нет | ✅ |
| torch.vmap | ❌ нет | ✅ |
| Custom operators registry | ❌ нет | ✅ torch.library |

### 4. Python API
| | PromeTorch | PyTorch |
|--|-----------|---------|
| Pythonic API | ⚠️ pybind11, базовый | ✅ полный, idiomatic |
| `__getitem__`, `__setitem__` | ⚠️ ограниченный | ✅ полный slicing |
| `torch.no_grad()` context | ❌ не подключён к engine | ✅ |
| `model.eval()` / `model.train()` | ⚠️ | ✅ |
| `nn.Module` subclassing в Python | ❌ только C++ | ✅ |
| DataParallel / DistributedDataParallel | ❌ | ✅ |
| TensorBoard logging | ❌ | ✅ |

### 5. Качество и надёжность
| | PromeTorch | PyTorch |
|--|-----------|---------|
| CI/CD | Docker (5 платформ) | GitHub Actions, CircleCI |
| Тестов | ~500 | ~100,000 |
| Покрытие backward | 50+ функций | 1000+ |
| FlashAttention | ⚠️ нерабочий (6 багов) | ✅ |
| Edge cases | Minimal | Extensive |
| Error messages | Базовые | Подробные с suggestions |
| Memory leak detection | Базовый | ASAN/MSAN/TSAN интеграция |

---

## Где PromeTorch УНИКАЛЕН

### 1. Российское железо — единственный фреймворк
- **Эльбрус-8СВ**: нативная поддержка, EML BLAS, NUMA-aware GEMM, 10% быстрее PyTorch
- **NM Card Mini**: fused forward/backward transformer на DSP, Q16.16 эмулятор
- **NM QUAD**: 64-ядерный row-parallel training, nmpp SIMD 100x, gradient accumulation

Ни PyTorch, ни TensorFlow, ни ONNX Runtime не поддерживают это железо.

### 2. GGUF inference
- Загрузка и inference моделей в формате GGUF (Llama, Qwen, Gemma, DeepSeek)
- Quantized GEMV с warp-cooperative access
- 49.9 tok/s GPU qwen3:4b

### 3. Header-only C++ (почти)
- Весь фреймворк в .h файлах — подключил и используй
- Минимальные зависимости (только CUDA SDK опционально)
- Компилируется на Windows, Linux, ALT Linux, Astra Linux, RED OS

---

## Количественное сравнение

| Метрика | PromeTorch | PyTorch 2.x |
|---------|-----------|-------------|
| Строк кода | 91,000 | ~3,000,000 |
| Файлов | 194 | ~15,000 |
| Backends | 5 (CPU, CUDA, NMCard, NMQuad, Эльбрус) | 10+ (CPU, CUDA, XLA, MPS, Vulkan...) |
| NN модулей | 80+ | 200+ |
| Backward функций | 50+ | 1000+ |
| Оптимизаторов | 10 | 13 |
| Поддерживаемых dtype | 4 (f32, f64, i32, i64) | 15+ (f16, bf16, f32, f64, complex, ...) |
| GPU поколений | Compute 5.0+ | Compute 3.5+ |

---

## Вывод

**PromeTorch — не замена PyTorch.** Это специализированный фреймворк для:
1. Обучения на российском железе (Эльбрус, NM Card, NM QUAD)
2. Embedded/edge deployment без Python runtime
3. Понимания внутренней архитектуры DL фреймворков (образовательная ценность)

**Для стандартных задач ML** (классификация, NLP, CV на NVIDIA GPU) → используйте PyTorch.
**Для российского железа** → PromeTorch единственный вариант с autograd + NN + optimizer.
