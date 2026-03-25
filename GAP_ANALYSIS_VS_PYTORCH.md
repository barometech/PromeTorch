# PromeTorch vs PyTorch — Gap Analysis

**Дата:** 2026-03-18 | **Метод:** 3 агента Opus 4.6, полный аудит каждого файла

---

## СВОДКА

| Категория | PromeTorch | PyTorch | Покрытие |
|-----------|-----------|---------|----------|
| Строк кода | 93K | ~3M | 3% (но архитектура полная) |
| CPU ops | ~120 | ~2000 | 6% |
| CUDA kernels | 68 | ~5000 | 1.4% |
| Backward functions | 88 | ~1500 | 6% |
| NN modules | 64+ | ~200 | 32% |
| Optimizers | 9 | 13 | 69% |
| LR schedulers | 9 | 15+ | 60% |
| ScalarTypes | 16 | 20+ | 80% |
| Backends | 3 (CPU, CUDA, NMCard) | 5+ | 60% |

---

## КРИТИЧЕСКИЕ ПРОБЕЛЫ (блокируют реальное использование)

### 1. ~~Autograd НЕ подключён к Conv/BN/Pool/активациям~~ FIXED (2026-03-25)
**Статус:** Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d + 14 activation modules wired to autograd.
CNN training on CPU now works. New backward nodes: GeluBackward, SoftplusBackward.

### 2. cuBLAS не используется для mm/bmm
**Влияние:** GEMM на CUDA в 5-10x медленнее cuBLAS.
**Что есть:** cuBLAS handle создан, используется только для FP16 HGEMV (inference).
**Что нужно:** Заменить custom 32x32 GEMM на `cublasSgemm` в `launch_gemm`.
**Объём:** ~20 строк.

### 3. Higher-order gradients (create_graph) не работают
**Влияние:** Нет Hessian, MAML, physics-informed NN.
**Что нужно:** Backward ops должны создавать autograd graph.
**Объём:** Значительный — переписка всех backward через autograd-tracked ops.

### 4. No Stream/Event (async CUDA)
**Влияние:** Нет async H2D/D2H, нет multi-stream overlap.
**Объём:** ~300 строк.

### 5. No from_blob()
**Влияние:** Zero-copy tensor из внешней памяти невозможен.
**Объём:** ~50 строк.

### 6. No DispatchKeySet (runtime dispatch)
**Влияние:** Backend выбирается compile-time (#ifdef), нет dynamic switching.
**Объём:** Архитектурное изменение, 1000+ строк.

---

## ВАЖНЫЕ ПРОБЕЛЫ

### NN Modules без backward
| Модуль | Forward | cuDNN Backward | Autograd Node | Статус |
|--------|---------|---------------|---------------|--------|
| Conv1d | CPU im2col+mm | Нет | Нет | BROKEN (no backward) |
| Conv2d | CPU im2col+mm + cuDNN | Есть | **DONE** | **WORKING** |
| Conv3d | STUB (zeros) | Нет | Нет | DEAD |
| BatchNorm1d | CPU raw | Нет | Нет | BROKEN (no backward) |
| BatchNorm2d | CPU raw | cuDNN есть | **DONE** | **WORKING** |
| LayerNorm | CPU raw | Нет | Нет | BROKEN |
| GroupNorm | CPU raw | Нет | Нет | BROKEN |
| MaxPool2d | CPU raw + cuDNN | cuDNN есть | **DONE** | **WORKING** |
| AvgPool2d | CPU raw + cuDNN | cuDNN есть | **DONE** | **WORKING** |
| Dropout | Mask mul_autograd | N/A | **DONE** | **WORKING** |
| ReLU | relu_autograd | N/A | **DONE** | **WORKING** |
| ReLU6 | hardtanh_autograd | N/A | **DONE** | **WORKING** |
| LeakyReLU | leaky_relu_autograd | Нет | **DONE** | **WORKING** |
| ELU | elu_autograd | Нет | **DONE** | **WORKING** |
| SELU | selu_autograd | Нет | **DONE** | **WORKING** |
| GELU | gelu_autograd | Нет | **DONE** | **WORKING** |
| SiLU | silu_autograd | Нет | **DONE** | **WORKING** |
| Mish | mish_autograd | Нет | **DONE** | **WORKING** |
| Sigmoid | sigmoid_autograd | Нет | **DONE** | **WORKING** |
| Tanh | tanh_autograd | Нет | **DONE** | **WORKING** |
| Hardtanh | hardtanh_autograd | Нет | **DONE** | **WORKING** |
| Hardsigmoid | hardsigmoid_autograd | Нет | **DONE** | **WORKING** |
| Hardswish | hardswish_autograd | Нет | **DONE** | **WORKING** |
| Softplus | softplus_autograd | Нет | **DONE** | **WORKING** |
| PReLU (1 param) | leaky_relu_autograd | Нет | **DONE** | **WORKING** |
| Softmax (module) | Tensor ops chain | Нет | Нет | NO AUTOGRAD (use CrossEntropyLoss) |

### Оптимизаторы
- **LBFGS** — отсутствует (IMPORTANT для scientific computing)
- **state_dict() / load_state_dict()** — отсутствует (CRITICAL для checkpoint resume)
- **foreach/fused mode** — отсутствует (10-20% speedup)
- Per-group betas/eps для Adam — не поддержано

### Data Loading
- **num_workers > 0** — заглушка (CRITICAL — single-threaded bottleneck)
- **PackedSequence** — отсутствует (CRITICAL для variable-length sequences)
- **pin_memory** — заглушка
- **pad_sequence / pack_sequence** — отсутствует

### CUDA
- **cuBLAS для mm/bmm** — не используется (CRITICAL perf)
- **CUDA sort, topk, cumsum** — отсутствуют
- **FP16/BF16 element-wise ops** — нет (float32 only)
- **Tensor Core (WMMA)** — нет
- **cuFFT** — нет (CPU FFT only)

### Python Bindings
- **view, reshape, permute, transpose** — не привязаны
- **requires_grad, grad** — не привязаны
- **item(), shape, to(device)** — не привязаны
- **Tensor.__repr__** — нет
- **LSTM/GRU/Transformer** — не привязаны
- **save/load** — не привязаны
- **DataLoader/Dataset** — не привязаны

### Инфраструктура
- **ONNX export** — нет
- **Distributed (DDP/FSDP)** — нет
- **TorchScript/JIT** — нет
- **torch.compile** — нет
- **Profiler** — нет (кроме GGUF inference profiler)

---

## NICE-TO-HAVE (не блокируют)

- Sparse tensors (COO/CSR)
- Complex tensor ops
- Named tensors
- Nested tensors
- FP8 types
- Additional activations (GLU, CELU, RReLU)
- Additional losses (MultiLabelMarginLoss, etc.)
- Additional factory functions (bartlett_window, etc.)
- Additional math ops (asin, acos, atan2, erf, lgamma)
- BatchNorm3d, InstanceNorm1d/3d
- ConvTranspose1d/3d
- 3D pooling variants

---

## ПЛАН ЗАКРЫТИЯ КРИТИЧЕСКИХ ПРОБЕЛОВ

### Phase 16: Autograd для Conv/BN/Pool (~1 неделя)
1. ConvBackward node (CPU im2col + cuDNN)
2. BatchNormBackward node (CPU + cuDNN)
3. PoolBackward nodes (MaxPool, AvgPool, Adaptive)
4. LayerNormBackward node
5. Wire все modules через autograd
6. Тест: CNN MNIST (Conv2d → Pool → Linear)

### Phase 17: cuBLAS + Performance (~2 дня)
1. Заменить custom GEMM на cublasSgemm
2. Benchmark mm/bmm vs PyTorch

### Phase 18: Optimizer state_dict + PackedSequence (~3 дня)
1. state_dict() / load_state_dict() для всех оптимизаторов
2. PackedSequence + pack_padded_sequence/pad_packed_sequence
3. LSTM dropout between layers

### Phase 19: Python bindings expansion (~3 дня)
1. Tensor methods (view, reshape, permute, to, requires_grad, item, shape)
2. NN modules (LSTM, Transformer, Conv2d в Python)
3. DataLoader в Python
4. save/load

---

*Полный аудит: INFRASTRUCTURE_AUDIT.md*
*Ссылки: CLAUDE.md, JOURNAL.md*
