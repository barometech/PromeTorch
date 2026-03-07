## GPU СВОБОДЕН — можно тестировать на CPU и CUDA

---

## Правила работы

1. Создать свой фреймворк аналог PyTorch с нуля (C++, Python, CUDA)
2. Запрещены упрощения. Нельзя откладывать на потом
3. При сложностях — искать в интернете, если свои итерации не помогают
4. Перед кодом — считать математически, проверять гипотезы в скриптах, потом наполнять
5. Пополнять журнал (`JOURNAL.md`)
6. **ПРОЧИТАЙ `AVOIDRECURSION.md` перед любым действием** — там описаны циклы которые НЕЛЬЗЯ повторять

---

## Статус проекта

**14 основных фаз + 7 критических фич ЗАВЕРШЕНЫ.** ~40,000+ строк C++/CUDA, 97+ файлов.

| Фаза | Компонент | Статус |
|------|-----------|--------|
| 1 | c10 core (Allocator, Device, Storage, TensorImpl) | DONE |
| 2 | ATen (MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps) | DONE |
| 3 | Autograd (engine, 50+ backward functions) | DONE |
| 4 | NN Modules (50+ слоёв: Linear, Conv, BN, Transformer, PIR) | DONE |
| 5 | Optimizers (SGD, Adam, AdamW, RMSprop) | DONE |
| 6 | LR Schedulers (13 видов) | DONE |
| 7 | Data Loading (Dataset, DataLoader, Sampler) | DONE |
| 8 | Transformer (Encoder, Decoder, MultiheadAttention) | DONE |
| 9 | PIR Architecture (RMSNorm, RoPE, PIR270M) | DONE |
| 10 | CUDA Backend (собственные kernels: GEMM, reduce, element-wise) | DONE |
| 11 | Python Bindings (pybind11) | DONE |
| 12 | cuDNN Integration (conv, pool, batchnorm, activations) | DONE |
| 13 | Mixed Precision AMP (GradScaler, Autocast) | DONE |
| 14 | FlashAttention (O(N) memory, causal masking) | DONE |

### Критические фичи (2026-03-07)

| Фича | Файл(ы) | Статус |
|------|---------|--------|
| Custom Autograd Functions | `torch/autograd/function.h` | DONE |
| Hooks System | `torch/nn/module.h` | DONE |
| Serialization (save/load) | `torch/serialization.h` | DONE |
| Advanced Indexing | `IndexOps.h`, `IndexBackward.h` | DONE |
| Gradient Checkpointing | `torch/utils/checkpoint.h` | DONE |
| RNN/LSTM/GRU | `torch/nn/modules/rnn.h` | DONE |
| Channels-last Memory Format | `TensorImpl.h`, `Tensor.h`, `ShapeOps.h` | DONE |

### Все 10 моделей — РЕШЕНО (2026-03-07)
- MNIST 97.65%, LSTM 98.44%, GRU 95.3% — все match PyTorch baseline
- Root causes: mm() non-contiguous, copy_() strided, unary ops (sigmoid/tanh) non-contiguous

---

## Сборка

### Окружение
- Windows 10, MSVC 2019, Anaconda Python 3.12
- CUDA 12.9.86 @ `C:\ProgramData\anaconda3\Library`
- cuDNN 9.14.0 @ `C:\ProgramData\anaconda3\Library`
- CMake: `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`

### Существующие рабочие сборки (НЕ пересобирать без причины!)
```
build_final3/examples/mnist/train_mnist_mlp.exe    # CPU, MNIST (новейшая)
build_examples/examples/mnist/train_mnist_mlp.exe   # CPU, MNIST
build_cudnn/examples/pir/train_pir.exe              # CUDA+cuDNN, PIR
```

### Запуск MNIST
```bash
cd /c/Users/paper/Desktop/promethorch
PATH="./build_final3:$PATH" ./build_final3/examples/mnist/train_mnist_mlp.exe --device cpu --epochs 1 --lr 0.001
```

### Сборка CPU (из Developer Command Prompt!)
```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set PATH=C:\ProgramData\anaconda3\Library\bin;%PATH%
cd /d C:\Users\paper\Desktop\promethorch\build_examples
nmake train_mnist_mlp
```

### Сборка CUDA
```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set CUDA_PATH=C:\ProgramData\anaconda3\Library
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_CUDA=ON -DPT_USE_CUDNN=ON -DPT_BUILD_TESTS=OFF -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%"
nmake
```

### Сборка из bash НЕ РАБОТАЕТ (rc.exe не находится). Использовать batch файлы или Developer Command Prompt.

---

## Структура проекта

```
c10/                          # Ядро: Allocator, Device, Storage, TensorImpl, ScalarType
aten/src/ATen/
  core/                       # Tensor.h, TensorFactory.h
  native/cpu/                 # MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps
  cuda/                       # CUDAKernels.cu, CUDAReduce.cu, CUDABlas.cu, FlashAttention.cu
  cudnn/                      # CuDNNConvolution, Pooling, BatchNorm, Activation
torch/
  csrc/autograd/              # edge, node, engine, autograd_meta, backward functions
  autograd/                   # function.h (custom autograd functions)
  nn/modules/                 # linear, activation, conv, pooling, normalization, loss, transformer, pir, rnn
  optim/                      # sgd, adam, rmsprop, adamkiller, lr_scheduler
  data/                       # dataset, sampler, dataloader
  amp/                        # grad_scaler, autocast
  utils/                      # checkpoint.h (gradient checkpointing)
  serialization.h             # save/load tensors and state_dicts
python/                       # pybind11 bindings, promethorch package
examples/
  mnist/train_mnist_mlp.cpp   # MNIST MLP training
  pir/train_pir.cpp           # PIR Shakespeare training
```

---

## Ключевые файлы для отладки autograd

| Файл | Что содержит |
|------|--------------|
| `torch/csrc/autograd/autograd.h` | `mm_autograd()`, `t_autograd()` — подключение backward к forward |
| `torch/csrc/autograd/engine.h` | `Engine::execute()` — backward pass |
| `torch/csrc/autograd/functions/LinearAlgebraBackward.h` | MmBackward, TransposeBackward |
| `torch/csrc/autograd/functions/MathBackward.h` | ReluBackward, CrossEntropyBackward |
| `torch/nn/modules/linear.h` | Linear forward: `x @ W^T + b` |
| `torch/nn/modules/loss.h` | CrossEntropyLoss forward + backward |
| `examples/mnist/train_mnist_mlp.cpp` | Training loop |

---

## Известные решённые проблемы

1. **DLL Singleton** — CUDACachingAllocator inline static → разные instances в разных DLL → heap corruption. Решение: `.cpp` файл с единственным singleton
2. **CUDA Shutdown** — double free при cudaFree. Решение: не освобождать CUDA память (как PyTorch)
3. **Linear init** — было `sqrt(3)/sqrt(fan_in)`, надо `1/sqrt(fan_in)` (PyTorch default)
4. **nvcc + MSVC flags** — использовать `$<$<COMPILE_LANGUAGE:CXX>:...>` в CMake
5. **Python bindings** — 19 исправлений (см. `JOURNAL.md`)

---

## Будущие фазы (опционально)

- Фаза 15: Distributed Training (DDP, NCCL)
- Фаза 16: TorchScript/JIT
- Фаза 17: Дополнительные операции (einsum, scatter_reduce)
- Фаза 18: Quantization (INT8)
- Фаза 19: ONNX export
- Фаза 20: Profiling tools

---

Полная история: `JOURNAL.md` | ТЗ: `TECHNICAL_SPECIFICATION.md` | Anti-loop: `AVOIDRECURSION.md`
