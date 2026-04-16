## ТЕКУЩИЙ СТАТУС: PIR 250M ТРЕНИРУЕТСЯ НА ЭЛЬБРУСЕ

**4-процесс DDP, 568 tok/s, loss 1.41 (step 200). Checkpoint сохранён.**
**Генерация: русскоязычный текст с правильной морфологией ("В России", "полагается", "специального страны задача").**

**НЕ делать полумеры, workaround'ы, "пока на 1 потоке". ТОЛЬКО полное решение.**
**НЕ использовать агентов для критических операций (Gemini API, SSH). Только Bash напрямую.**
**Агенты — ТОЛЬКО для аудитов/поиска. Модель агентов — ТОЛЬКО Opus, НИКОГДА Sonnet.**

---

## ЭЛЬБРУС: loginctl enable-linger ОБЯЗАТЕЛЬНО

**Root cause падений тренировки (2026-04-16):** `Linger=no` в systemd-logind убивал процессы при SSH disconnect.
**Фикс:** `loginctl enable-linger user` — выполнять после каждого reboot сервера!

---

## ПОЛНЫЙ АУДИТ ИНФРАСТРУКТУРЫ: `INFRASTRUCTURE_AUDIT.md` (43 бага, 93K строк)

---

## КРИТИЧЕСКОЕ ПРАВИЛО: НЕ ХОДИТЬ КРУГАМИ

**Инцидент 2026-04-08:** 6 раз подряд написал "нужны CUDA Graphs" и 6 раз ушёл оптимизировать GEMV kernel вместо этого. 6 бесполезных коммитов, 0 прогресса.

**ПРАВИЛО:** Если задача X блокирует прогресс — ДЕЛАТЬ X НЕМЕДЛЕННО. Не подменять простыми задачами A, B, C которые "тоже полезны". Если внутри X есть блокер Y — решать Y, не уходить в сторону. **НИКОГДА не коммитить "не помогло, нужно Y" больше 1 раза.** Написал "нужно Y" — ДЕЛАЙ Y.

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

**17 основных фаз + 7 критических фич ЗАВЕРШЕНЫ.** ~93,000+ строк C++/CUDA/Python, 631 файл (352 source).

| Фаза | Компонент | Статус |
|------|-----------|--------|
| 1 | c10 core (Allocator, Device, Storage, TensorImpl) | DONE |
| 2 | ATen (MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps) | DONE |
| 3 | Autograd (engine, 50+ backward functions) | DONE |
| 4 | NN Modules (50+ слоёв: Linear, Conv, BN, Transformer, PIR) | DONE |
| 5 | Optimizers (SGD, Adam, AdamW, RMSprop) | DONE |
| 6 | LR Schedulers (9 видов) | DONE |
| 7 | Data Loading (Dataset, DataLoader, Sampler) | DONE |
| 8 | Transformer (Encoder, Decoder, MultiheadAttention) | DONE |
| 9 | PIR Architecture (RMSNorm, RoPE, PIR270M) | DONE |
| 10 | CUDA Backend (собственные kernels: GEMM, reduce, element-wise) | DONE |
| 11 | Python Bindings (pybind11) | DONE |
| 12 | cuDNN Integration (conv, pool, batchnorm, activations) | DONE |
| 13 | Mixed Precision AMP (GradScaler, Autocast) | DONE |
| 14 | FlashAttention (O(N) memory, causal masking) | DONE |
| 15 | NM Card Mini Backend (эмулятор, Q16.16, 32 теста, MNIST 93.64%) | DONE |
| 16 | NM Quad Backend (4-чип, 64 ядра, SIMD 100x, 705 tok/s) | DONE |
| 17 | PIR 250M DDP Training on Elbrus (4×142=568 tok/s, loss 1.04) | DONE |

---

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
build_nmcard/train_mnist_nmcard.exe                 # NMCard эмулятор, MNIST
```

### Запуск MNIST
```bash
cd /path/to/promethorch
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
  nmcard/                     # NMCardAllocator (caching, PrivateUse1)
aten/src/ATen/
  core/                       # Tensor.h, TensorFactory.h
  native/cpu/                 # MathOps, ReduceOps, LinearAlgebra, ShapeOps, IndexOps
  cuda/                       # CUDAKernels.cu, CUDAReduce.cu, CUDABlas.cu, FlashAttention.cu
  cudnn/                      # CuDNNConvolution, Pooling, BatchNorm, Activation
  nmcard/                     # NMCardEmulator, NMCardOps, NMCardDispatch, NMCardMath (Q16.16)
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
  nmcard/train_mnist_nmcard.cpp # MNIST на NMCard эмуляторе
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
6. **NMCard DLL Allocator** — AllocatorRegistry inline static → разные instances per DLL. Решение: двойная регистрация (`register_nmcard_allocator()` + `register_nmcard_allocator_local()`)

---

## Известные проблемы (из аудита 2026-03-18)

- FlashAttention полностью нерабочий (6 критических багов) — не использовать
- Conv3d forward — stub (возвращает нули)
- dispatcher_suda_mc.abs — НЕ запускать на реальной карте (race condition + DDR saturation)
- Python bindings: no_grad() не подключён к C++ engine (BUG-C9)

---

## Будущие фазы (опционально)

- Фаза 18: Дополнительные операции (einsum, scatter_reduce)
- Фаза 19: Quantization (INT8)
- Фаза 20: ONNX export
- Фаза 21: Profiling tools
- Фаза 22: BPE tokenizer (RUKANIZER) для PIR тренировки
- Фаза 23: PIR 250M тренировка на NM Quad (послойная стратегия)

---

---

## Эльбрус PIR DDP

### Запуск тренировки (4-процесс DDP на 32 ядрах E8C2)
```bash
loginctl enable-linger user  # ОБЯЗАТЕЛЬНО после reboot!
for node in 0 1 2 3; do
  PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
  numactl --cpunodebind=$node --preferred=$node \
  ./build_mt/examples/pir/train_pir_elbrus \
    --fused --full --batch_size 4 --rank $node --nprocs 4 \
    --max_steps 1000 --log_interval 50 --gen_interval 200 --gen_tokens 200 \
    --save_interval 200 --save_dir checkpoints \
    --grad_accum 10 --lr 0.0006 \
    --load checkpoints/pir_fused_step_200.bin \
    --data data/russian_mega.txt &
done
```

### Ключевые фиксы (2026-04-12..16)
1. **OmpNestedGuard bypass** при PT_NO_NUMA_POOL=1 — EML_MT нуждается в полном OMP
2. **ffn2 dimension fix**: (BT,D,H) → (BT,H,D) — buffer overflow + wrong gradients
3. **generate_text() PIR residual**: out_proj перезаписывал buf_pir вместо +=
4. **grad_sync timeout**: 120s вместо бесконечного ожидания
5. **loginctl enable-linger**: systemd убивал процессы при SSH disconnect

### Результаты
| Step | Loss | Perplexity | tok/s | Генерация |
|------|------|------------|-------|-----------|
| 200 | 1.41 | 4.1 | 158 | "соположение", "Кроины" |
| 400 | 1.23 | 3.4 | 147 | "Первой", "специального", "военных" |
| 600 | 1.15 | 3.1 | 147 | "Российская", "города", "музей", "количеству" |
| 800 | 1.04 | 2.8 | 147 | "В России", "полагается", "15 марта 2008 года" |

---

Полная история: `JOURNAL.md` | ТЗ: `TECHNICAL_SPECIFICATION.md` | Anti-loop: `AVOIDRECURSION.md` | **Аудит инфраструктуры: `INFRASTRUCTURE_AUDIT.md`**

---

**ФИНАЛЬНОЕ НАПОМИНАНИЕ: НИКОГДА НЕ ЗАПУСКАТЬ ТРЕНИРОВКУ БЕЗ `loginctl enable-linger`. НИКОГДА.**

---

**ФИНАЛЬНОЕ НАПОМИНАНИЕ: PIR 250M = МИНИМУМ 900 TOK/S. НЕ ЗАПУСКАТЬ МУСОР. НЕ ЖДАТЬ. ЧИНИТЬ.**
