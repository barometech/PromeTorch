# PromeTorch - Журнал разработки

Полная история разработки проекта. Актуальные инструкции — в `CLAUDE.md`.

---

## 2026-03-08: GGUF Inference — загрузка моделей Ollama + генерация текста (CPU & CUDA)

**~3000 строк нового кода, 6 новых файлов, qwen3:4b и gemma3:4b работают.**

### GGUF Reader (`torch/io/gguf_loader.h`)
- Полный парсинг формата GGUF v3 (magic, header, metadata KV pairs, tensor info)
- Поддержка всех типов метаданных: string, int, float, bool, array
- Автоматическое выравнивание данных (32-byte alignment)
- `reader.load_tensor(name)` — чтение + dequantization → float32

### Dequantization (`torch/io/gguf_dequant.h`)
- Q4_K_M: 256 values per 144-byte block, 6-bit packed scales + 4-bit weights
- Q6_K: 256 values per 210-byte block, ql[128] + qh[64] + scales[16] + d(fp16)
- Q8_0, Q5_K, F16, F32
- Исправлен баг Q6_K: scale index `n/16` → `n/16 + l/16` (без этого веса были мусором)

### Tokenizer (`torch/io/tokenizer.h`)
- SentencePiece BPE (Llama, Gemma): ▁ (U+2581) word separator
- GPT-2 BPE (Qwen): Ġ (U+0120) space encoding, word-level pre-tokenization
- Byte fallback (<0xNN>), encode/decode

### Ollama Resolver (`torch/io/ollama.h`)
- Автоматический поиск моделей: `~/.ollama/models/manifests/` → digest → blob path
- Поддержка Windows и Linux путей

### Transformer Inference (`torch/io/gguf_model.h`)
- Полный transformer: RMSNorm, RoPE, GQA attention, SwiGLU FFN, KV cache
- Поддержка архитектур: qwen3, gemma3, llama (через GGUF metadata)
- Gemma-specific: embedding scaling (sqrt(H)), QK-norm, post-attention/post-FFN norms
- Gemma RMSNorm: GGUF converter bakes in +1 (layernorm1p) → НЕ добавлять +1 повторно
- CUDA: matmul (GEMM), SiLU, element-wise ops на GPU; pre-transpose 2D weights при to_cuda()
- Top-k/top-p sampling, greedy decoding, temperature

### Полностью GPU инференс (CUDA kernels)
- `CUDAInference.cu`: собственные CUDA kernels — RMSNorm, per-head QK-norm, RoPE, causal GQA attention, concat
- Убраны все `cuda_synchronize()` (sync только при CPU←GPU transfer для sampling)
- Chat template: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
- Special token encoding (tokenizer находит `<|im_start|>` и кодирует как один ID)
- GPT-2 decode: Ċ→\n, ĉ→\t
- `<think>...</think>` stripping (qwen3 thinking mode)
- Stop tokens: `<|im_end|>`, `<end_of_turn>`, `<|eot_id|>`, `</s>`
- Repetition penalty (1.05) для предотвращения зацикливания

### Оптимизация скорости: 16 → 49.9 tok/s (3.1× speedup)

**Фазы оптимизации:**
1. **Profiler** (`torch/io/inference_profiler.h`) — CUDA event-based timing, per-operation breakdown
2. **Vectorized GEMV** — float4 loads, 128 threads/block → 16 → 25.4 tok/s (+59%)
3. **Coalesced float32 GEMV** — row-major access → 25.4 → 34.4 tok/s (+35%)
4. **Warp-cooperative quant GEMV** — полная перезапись CUDAQuantGemv.cu:
   - Каждый warp = 1 output row, 32 lanes читают consecutive qs bytes
   - x vector в shared memory, uint32_t packed load, float4 loads из smem
   - Warp shuffle reduction (без shared memory для reduce)
   - 34.4 → 49.9 tok/s (+45%)
5. **Scratch Pool** — pre-allocated decode buffers, zero alloc hot path (no speed gain — caching allocator уже быстрый)
6. **Shared memory fix** — cudaFuncSetAttribute для K > 12288 (68 KB smem для ffn_down в 14B+ моделях)
7. **Think tag fix** — strip everything from `<think>` to end when no `</think>`
8. **/no_think** — system message для Qwen3 чтобы отключить thinking mode

### Результаты vs Ollama baseline (A100 40GB)

| Модель | VRAM | PromeTorch tok/s | Ollama tok/s | Ratio | Корректность |
|--------|------|-----------------|-------------|-------|-------------|
| qwen3:4b | 4.9 GB | **49.9** | 164.6 | 30% | ✅ "The result of 2 + 2 is 4." |
| gemma3:4b | ~3 GB | **52.9** | 147.5 | 36% | ✅ Correct "4" |
| deepseek-r1:8b | 5.9 GB | **30.5** | 129.6 | 24% | ✅ Correct answers |
| qwen3:14b | 9.6 GB | **18.4** | 84.4 | 22% | ✅ Works (after smem fix) |
| qwen3:30b | - | ❌ MoE | 115.4 | - | MoE не поддерж. |
| gemma3:27b | 9.6 GB | ❌ crash | 48.9 | - | tied weights bug |
| llama3.3:70b | - | ❌ too large | - | - | 40.5 GB > VRAM |

**Предыдущий baseline 40 tok/s был неверен.** Реальный Ollama: 84-165 tok/s. Мы на 22-36%.

### Что делать дальше
- **gemma3:27b**: исправить undefined tensor при tied embeddings
- **MoE**: поддержка qwen3moe (expert routing, shared expert)
- **Скорость**: мы на 22-36% от Ollama. Потенциальные улучшения:
  - Fused Residual+RMSNorm kernel (-2 launches/layer)
  - GPU Embedding Lookup (убрать CPU→GPU transfer)
  - Fused QK-norm+RoPE
  - FP16 compute (half precision GEMV)
  - Flash Decoding (batched attention)

**Ключевые ограничения vs Ollama:**
- **Скорость 3-4x медленнее**: наши kernels чисто float32, Ollama — half precision + flash decoding
- **VRAM эффективный**: quant-only mode (4.9 GB для 4B), сравнимо с Ollama
- **Загрузка медленная**: 20-40 сек vs Ollama 2 сек (mmap)
- **Длинные тексты**: деградация после ~80 токенов с greedy decoding

### Исправленные баги
- Q6_K scale index bug: `is = n/16` → `is = n/16 + l/16`
- Tokenizer: Qwen reports "gpt2" model_type, не SentencePiece → GPT-2 pre-tokenization
- GPT-2 spaces: Ġ (U+0120) → space, Ċ (U+010A) → \n
- Gemma norm +1: GGUF converter already bakes in +1, double-application caused value explosion
- CUDA QK-norm: device mismatch crash (moved to GPU, then RoPE wrote as CPU)
- Tied embeddings: output_weight == token_embedding → separate transposed copy on GPU
- `<think>` stripping: don't erase all text when `</think>` missing

---

## 2026-03-08: Phase 2 — linalg, FFT, tensor ops, ConvTranspose2d, INT8 quantization

**~3700 строк нового кода, 16 файлов, 39/39 тестов пройдены.**

### torch.linalg (LinearAlgebra.h + backward + autograd)
- `lu()` — Gaussian elimination + partial pivoting → L, U, P
- `inverse()` — через LU → solve(A, I)
- `solve(A, b)` — forward/backward substitution через LU
- `det()` — sign(P) * prod(diag(U))
- `cholesky()` — L@L^T decomposition для SPD матриц
- `qr()` — Householder reflections
- `trace()`, `cross()`, `matrix_norm()` (1/inf/Frobenius)
- Backward: InverseBackward, DetBackward, CholeskyBackward, TraceBackward

### Tensor ops (ShapeOps.h + backward)
- `flip()`, `roll()`, `meshgrid()`, `repeat_interleave()`, `unique()`
- `tril_indices()`, `triu_indices()`
- Backward: FlipBackward, RollBackward, RepeatInterleaveBackward

### FFT (новый FFTOps.h, 445 строк)
- Cooley-Tukey radix-2 DIT, O(N log N)
- `fft/ifft/rfft/irfft/fft2/ifft2/fftfreq/rfftfreq/fftshift/ifftshift`
- Complex format: `[..., 2]` (last dim = [real, imag])

### ConvTranspose2d — реальная реализация (был STUB)
- scatter-based transposed convolution, groups support
- Проверено: output != zeros, правильная shape

### Generalized pad (functional.h)
- 4 режима: constant, reflect, replicate, circular
- Любая размерность (1D-5D)

### Unfold/Fold (im2col/col2im)
- `unfold_im2col()`: N,C,H,W → N,C*kH*kW,L
- `fold_col2im()`: обратная операция

### INT8 Quantization (4 новых файла)
- `QuantizedTensor` + `quantize_per_tensor/per_channel` + `dequantize()`
- Observers: MinMaxObserver, HistogramObserver, PerChannelMinMaxObserver
- QuantizedLinear, QuantizedConv2d (fake quant forward)
- Pipeline: prepare → calibrate → convert → quantize_model

### Тесты
- Phase 2 test_phase2.exe: **39/39 PASS**
- 10 models CPU: ALL PASS (MNIST 97%, LSTM 98.44%, GRU 95.3%)
- 10 models CUDA (A100): ALL PASS (MNIST 97.78%, LSTM 93.75%, GRU 98.44%)
- PIR CUDA: 7.2M params, 50 iter, 35s, loss 3.07

---

## 2026-03-07: Unary ops non-contiguous fix — LSTM WORKS!

**Root cause**: `DEFINE_UNARY_OP` (sigmoid, tanh, exp, etc.) в MathOps.h использовал `data_ptr()[i]` последовательный доступ без учёта strides. Когда LSTM делит gates через `narrow_autograd(gates, 1, offset, H)`, результат — view с strides `[4*H, 1]` вместо contiguous `[H, 1]`. Sequential access `in[0], in[1], ...` читает данные из ЧУЖИХ gates для batch > 0.

**Fixes**:
- `DEFINE_UNARY_OP`: добавлен `.contiguous()` на входе
- `DEFINE_UNARY_OP_INPLACE`: fallback через out-of-place + copy_ для non-contiguous
- Scalar `add/mul/pow`: добавлен `.contiguous()` на входе
- `zero_()`: stride-aware path вместо memset для non-contiguous
- `fill_()`: stride-aware path для non-contiguous
- Восстановлены все 10 моделей в train_10_models.cpp

**Результаты**: LSTM 50% → 98.44%, все 10 моделей match PyTorch baseline.

| Model | PyTorch | PromeTorch | Status |
|-------|---------|-----------|--------|
| 4: MNIST (SGD) | 92.54% | 92.69% | MATCH |
| 5: Deep MNIST (Adam) | 97.46% | 97.03% | MATCH |
| 6: Dropout MNIST | 97.03% | 97.00% | MATCH |
| 7: RNN Sine | 1.1e-5 | 1.7e-5 | OK |
| 8: LSTM | 98.4% | 98.44% | MATCH |
| 9: GRU | 92.2% | 95.31% | MATCH |
| 10: Wide MNIST | 97.59% | 97.65% | MATCH |

---

## 2026-01-20: Старт проекта

- Исследована архитектура PyTorch (c10, ATen, torch, autograd)
- Составлен полный список кернелов (~1200+ операций)
- Определена структура C++/Python биндингов (pybind11)
- Создано полное ТЗ: `TECHNICAL_SPECIFICATION.md`

### Фаза 1: Ядро c10 — ЗАВЕРШЕНО
- `c10/macros/Macros.h` — платформенные макросы, CUDA поддержка
- `c10/util/Exception.h` — система исключений
- `c10/core/ScalarType.h` — типы данных (Float, Double, Half, BFloat16, Int, Bool...)
- `c10/core/Device.h` — абстракция устройств (CPU, CUDA, MPS, Meta...)
- `c10/core/Allocator.h` — управление памятью (64-byte aligned для AVX-512)
- `c10/core/Storage.h` — хранилище данных тензора (reference counting)
- `c10/core/TensorImpl.h` — низкоуровневая реализация тензора
- `CMakeLists.txt` — C++17, OpenMP, CUDA, AVX/AVX2, Google Test
- Тесты: 5 файлов, ~150 тестов

### Фаза 2: ATen (Tensor Operations) — ЗАВЕРШЕНО
- `aten/src/ATen/core/Tensor.h` — высокоуровневый Tensor с операторами
- `aten/src/ATen/core/TensorFactory.h` — фабрики (empty, zeros, ones, rand, randn, arange...)
- `aten/src/ATen/native/cpu/MathOps.h` — унарные, бинарные, broadcasting, in-place
- `aten/src/ATen/native/cpu/ReduceOps.h` — sum, mean, max, min, var, std, norm
- `aten/src/ATen/native/cpu/LinearAlgebra.h` — mm, mv, bmm, dot, matmul, addmm
- `aten/src/ATen/native/cpu/ShapeOps.h` — view, reshape, transpose, cat, stack, split
- `aten/src/ATen/native/cpu/IndexOps.h` — select, slice, gather, scatter, where
- `aten/src/ATen/ATen.h` — главный include
- Тесты: ~60 тестов

### Фаза 3: Autograd — ЗАВЕРШЕНО
- `torch/csrc/autograd/edge.h` — рёбра графа
- `torch/csrc/autograd/node.h` — Node, AccumulateGrad, topological sort
- `torch/csrc/autograd/autograd_meta.h` — grad, grad_fn, version_counter
- `torch/csrc/autograd/engine.h` — GraphTask, Engine::execute(), backward(), grad()
- `torch/csrc/autograd/functions/MathBackward.h` — Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Sigmoid, Relu, Add, Sub, Mul, Div, Pow
- `torch/csrc/autograd/functions/ReduceBackward.h` — Sum, Mean, Max, Min, Var, Std, Norm
- `torch/csrc/autograd/functions/LinearAlgebraBackward.h` — Mm, Mv, Bmm, Dot, Matmul, Addmm, Transpose
- `torch/csrc/autograd/functions/ShapeBackward.h` — View, Reshape, Squeeze, Permute, Expand, Cat, Stack, Select
- `torch/csrc/autograd/autograd.h` — autograd-aware операции (*_autograd)
- Тесты: ~30 тестов

### Фаза 4: NN Modules — ЗАВЕРШЕНО
- `torch/nn/parameter.h` — Parameter, Buffer
- `torch/nn/module.h` — Module (register_parameter, state_dict, train/eval, to(device))
- `torch/nn/init.h` — xavier, kaiming, orthogonal, sparse инициализации
- `torch/nn/modules/container.h` — Sequential, ModuleList, ModuleDict
- `torch/nn/modules/linear.h` — Identity, Linear, Bilinear, LazyLinear
- `torch/nn/modules/activation.h` — 18 активаций (ReLU, GELU, SiLU, Mish, Softmax...)
- `torch/nn/modules/conv.h` — Conv1d/2d/3d, ConvTranspose2d (im2col)
- `torch/nn/modules/pooling.h` — MaxPool, AvgPool, AdaptivePool
- `torch/nn/modules/normalization.h` — BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d
- `torch/nn/modules/dropout.h` — Dropout, Dropout1d/2d/3d, AlphaDropout
- `torch/nn/modules/sparse.h` — Embedding, EmbeddingBag, one_hot
- `torch/nn/modules/loss.h` — ~20 loss функций (CE, MSE, BCE, NLL, Focal, Dice...)
- `torch/nn/nn.h` — count_parameters, freeze/unfreeze, clip_grad_norm_
- `torch/nn/functional.h` — F:: namespace
- Тесты: ~70 тестов

### Фаза 5: Optimizers — ЗАВЕРШЕНО
- `torch/optim/optimizer.h` — базовый Optimizer, ParamGroup
- `torch/optim/sgd.h` — SGD (momentum, nesterov, weight_decay)
- `torch/optim/adam.h` — Adam, AdamW (bias correction, AMSGrad)
- `torch/optim/rmsprop.h` — RMSprop (centered, momentum)

### Фаза 6: LR Schedulers — ЗАВЕРШЕНО
- `torch/optim/lr_scheduler.h` — StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, PolynomialLR, ReduceLROnPlateau, OneCycleLR, CyclicLR, WarmupLR, ChainedScheduler, SequentialLR

### Фаза 7: Data Loading — ЗАВЕРШЕНО
- `torch/data/dataset.h` — Dataset<T>, TensorDataset, ConcatDataset, Subset, MapDataset
- `torch/data/sampler.h` — Sequential, Random, SubsetRandom, Batch, Distributed
- `torch/data/dataloader.h` — DataLoader (batch, shuffle, drop_last)

### Фаза 8: Transformer — ЗАВЕРШЕНО
- `torch/nn/modules/attention.h` — ScaledDotProductAttention, MultiheadAttention
- `torch/nn/modules/transformer.h` — EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer, PositionalEncoding

### Фаза 9: PIR Architecture — ЗАВЕРШЕНО
- `torch/nn/modules/pir.h` — RMSNorm, RotaryEmbedding, PIRLayer, PIRBlock, PIRAttention
- `torch/nn/modules/pir270m.h` — PIR270M (token embedding, 24 blocks, LM head, generate())
- Backward: SiLU, RMSNorm, ParallelScan, RotaryEmbedding, CrossEntropy, Embedding
- `examples/pir/train_pir.cpp` — Shakespeare training

### Фаза 10: CUDA Backend — ЗАВЕРШЕНО
- `c10/cuda/CUDAAllocator.h` — CUDACachingAllocator (block caching)
- `aten/src/ATen/cuda/CUDAKernels.cu` — 50+ element-wise ops
- `aten/src/ATen/cuda/CUDAReduce.cu` — warp/block reductions, dimensional
- `aten/src/ATen/cuda/CUDABlas.cu` — tiled GEMM 32x32, batched, GEMV, dot
- `aten/src/ATen/cuda/CUDADispatch.h` — CPU/CUDA dispatch layer

---

## 2026-01-21: Python Bindings + cuDNN + CUDA Training

### Фаза 11: Python Bindings (pybind11) — ЗАВЕРШЕНО
- `python/csrc/init.cpp` — DeviceType, Device, ScalarType bindings
- `python/csrc/tensor_bindings.cpp` — Tensor с numpy interop, factory functions
- `python/csrc/autograd_bindings.cpp` — GradMode, no_grad, backward, grad
- `python/csrc/nn_bindings.cpp` — Module, Linear, Conv2d, Loss functions, functional
- `python/csrc/optim_bindings.cpp` — SGD, Adam, AdamW, RMSprop, LR schedulers

**Исправленные ошибки bindings (19 штук):**
scalar_type()→dtype(), Int8→Char, ssize_t→py::ssize_t, element_size()→itemsize(), requires_grad_()→set_requires_grad(), .first/.second→std::get, backward через autograd, pow overloads, GradMode thread_local, Parameter pointers, Loss constructor order, Optimizer Options structs, LRScheduler reference, size property/method conflict, no_grad duplicate, Reduction enum order, argmax/argmin dims, Tensor::grad()

### Фаза 12: cuDNN Integration — ЗАВЕРШЕНО
- `aten/src/ATen/cudnn/CuDNNHandle.h` — handle management, descriptors
- `aten/src/ATen/cudnn/CuDNNConvolution.h` — forward, backward_data, backward_filter, fused conv+bias+relu
- `aten/src/ATen/cudnn/CuDNNPooling.h` — max/avg pool forward/backward
- `aten/src/ATen/cudnn/CuDNNBatchNorm.h` — training/inference forward, backward
- `aten/src/ATen/cudnn/CuDNNActivation.h` — relu, sigmoid, tanh, elu, swish, softmax
- `aten/src/ATen/cudnn/CuDNN.h` — high-level dispatch
- `cmake/FindcuDNN.cmake`
- cuDNN 9.14.0 @ `C:\ProgramData\anaconda3\Library\`

### Фаза 13: Mixed Precision (AMP) — ЗАВЕРШЕНО
- `torch/amp/grad_scaler.h` — GradScaler (scale, unscale, step, update, state_dict)
- `torch/amp/autocast.h` — AutocastGuard, categories, type casting
- `torch/amp/amp.h` — half(), bfloat16(), float32(), has_tensor_cores()

### Фаза 14: FlashAttention — ЗАВЕРШЕНО
- `aten/src/ATen/cuda/FlashAttention.h` — config, forward, backward, scaled_dot_product
- `aten/src/ATen/cuda/FlashAttention.cu` — tiled O(N) attention, online softmax, causal masking, head_dim 64/128

### CUDA Training — РАБОТАЕТ
- 100 итераций за 2 секунды на GPU
- Loss: 4.29 → 4.25 (снижается)
- Рабочая конфигурация: `--n_layers 2 --n_pir_layers 1 --n_embd 128`
- Большая модель (6 layers) crashит из-за dynamic_parallel_scan GPU→CPU→GPU копирования

---

## 2026-01-23: Исследование утечки памяти GPU

**Проблема:** PIR 6 layers crash на iter ~19. Память +2GB/iter: 698MB → 39GB.
**Причина:** Autograd saved tensors не освобождаются после backward.
**Попытки:** release_saved_tensors() в Node, очистка в apply(), clear_grad_fn().
**Статус:** Не решено на этом этапе — root cause оказался в DLL singleton (см. 01-24).

---

## 2026-01-24: CUDA Crash Fixes

### DLL Singleton Problem — ROOT CAUSE
`CUDACachingAllocator::get()` был inline со static var в header. На Windows каждая DLL получала свою копию → allocation в одном модуле, deallocation в другом → heap corruption.

**Решение:** `c10/cuda/CUDAAllocator.cpp` с единственным singleton, `get()` — declaration only (не inline), класс PT_API. `aten_cuda` теперь SHARED library.

### CUDA Exit Crash — PyTorch Pattern
При shutdown() был double free (free_blocks_ + ptr_to_block_ пересекаются).
**Решение:** Как в PyTorch — НЕ освобождать CUDA память при shutdown. CUDA driver сам всё освободит.

### GPU Load Optimization
Спайки из-за debug output + cudaDeviceSynchronize. Удалён весь debug из production кода. GPU загрузка стала ровной.

---

## 2026-01-25: MNIST Training Investigation

### Проблема
MNIST MLP 784→512→256→128→10: accuracy 12-15% вместо ожидаемых ~49%.

### Проверено (всё корректно)
- CrossEntropyLoss/Backward: `(softmax - one_hot) / N`
- MmBackward: `grad_A = grad_C @ B^T`, `grad_B = A^T @ grad_C`
- ReluBackward: `grad * (input > 0)`
- SGD step: `w = w - lr * grad`
- Gradient check: 9/10 PASS

### Исправления
1. `linear.h:57` — bound = `1/sqrt(fan_in)` вместо `sqrt(3)/sqrt(fan_in)` (PyTorch default)
2. `adamkiller.h:266` — step_size = layer_lr (убран double bias correction)
3. Восстановлена 4-слойная MLP (была упрощена до 1 слоя при отладке)
4. Удалён debug output из TensorImpl.cpp, autograd_meta.h, engine.h

### Результат
Все 8 параметров получают градиенты. Accuracy 14.88% — лучше, но ещё не на уровне PyTorch (~49%).

### Текущий диагноз (нерешено)
Backward формулы правильные. Подозрение на:
1. Как backward подключается — `mm_autograd()`, `t_autograd()` в `autograd.h`
2. Построение графа вычислений (edges)
3. Накопление градиентов между батчами
4. Autograd graph cleanup

---

## Решённые проблемы сборки (справочник)

### CUDA CMake
- **nvcc + MSVC flags** → `$<$<COMPILE_LANGUAGE:CXX>:...>` generator expressions
- **Deprecated GPU archs** → `CMAKE_CUDA_ARCHITECTURES 75 80 86 89`
- **CUDA_SEPARABLE_COMPILATION** → OFF (нет extern __device__)
- **CUDA toolkit из Anaconda** → `-DCMAKE_CUDA_COMPILER=...` `-DCUDAToolkit_ROOT=...`

### Python Bindings
- ScalarType: `Char` (не Int8), `Short` (не Int16)
- `dtype()` (не scalar_type()), `itemsize()` (не element_size())
- `set_requires_grad()` (не requires_grad_())
- backward через `torch::autograd::tensor_backward()`
- Optimizer constructors через Options structs

### Windows/Bash
- `exit code 127` из bash → `start //b` с batch файлом
- rc.exe не найден → запускать из Developer Command Prompt (vcvarsall.bat)
- c10.dll зависимость → добавить build dir в PATH

---

## 2026-03-02: ИСПРАВЛЕН MNIST — Contiguous Fix

### ROOT CAUSE
Функция `mm()` в `aten/src/ATen/native/cpu/LinearAlgebra.h` читала данные через `data_ptr<>()` с контигуозными индексами `A[i*K + k]`, но `tensor.t()` создаёт VIEW с транспонированными strides (данные в памяти НЕ перераспложены). Результат: **неправильное перемножение матриц** в forward И backward.

### Почему gradient check "проходил"
Numerical gradient: `(f(w+eps) - f(w-eps)) / 2eps` — оба f() используют тот же buggy `mm()`. Analytical gradient тоже через buggy `mm()`. Оба wrong одинаково → match.

### Fix
`.contiguous()` перед raw pointer access в mm, mv, bmm, dot, outer, addmm:
```cpp
Tensor A = self.contiguous();  // копирует данные в row-major если нужно
Tensor B = other.contiguous();
```

### Результат
| Метрика | Было | Стало |
|---------|------|-------|
| Loss (1 epoch) | 2.318 | 1.117 |
| Train Acc | 14.88% | 71.05% |
| Test Acc | 14.86% | **88.94%** |

### Файлы изменены
- `aten/src/ATen/native/cpu/LinearAlgebra.h` — добавлен `.contiguous()` во все функции
- `torch/nn/modules/linear.h` — init bound fix (1/sqrt(fan_in))

---

## Статистика (на 2026-01-25)

| Метрика | Значение |
|---------|----------|
| Файлов C++/CUDA | 92 |
| Строк кода | ~37,000 |
| c10 (core) | 3,278 |
| ATen (tensor ops) | 9,344 |
| CUDA kernels | 6,996 |
| Autograd | 3,559 |
| NN Modules | 9,858 |
| Optimizers | 1,246 |
| Data Loading | 1,176 |
