# Аудит ВОСЕМЬ — Python API (`prometorch` PyPI пакет)

**Дата:** 2026-05-20
**Источник:** `python/prometorch/` (verified live import + dir() через Python 3.12 / kancuda env, `_C.cp312-win_amd64.pyd` available)
**Метод:** загружен реальный wheel-equivalent (`Release/_C.cp312-win_amd64.pyd` + os.add_dll_directory), `import prometorch as pt; dir(pt.<sub>)` для каждого namespace.

`pt._C_AVAILABLE = True` — все C++ символы доступны. Defensive fallbacks из `__init__.py` (sys.modules stub etc.) активируются только при failed `_C` load.

## Top-level inventory (live `dir(prometorch)`)

84 публичных символа. Полный список: Tensor, DeviceType, TensorOptions, device, dtype, amp, arange, autograd, backward, bmm, cat, chunk, clamp, compile, cos, cuda, cuda_device_count, cuda_is_available, cumsum, data, distributed, dot, einsum, empty, enable_grad, exp, eye, from_numpy, full, grad, is_grad_enabled, isfinite, isinf, isnan, jit, linspace, load, load_pytorch, load_state_dict, log, manual_seed, matmul, max, mean, min, mlir, mm, mobile, multinomial, nan_to_num, nn, no_grad, norm, ones, ones_like, onnx, optim, quantization, rand, randint, randn, relu, rsqrt, save, save_pytorch, save_state_dict, serve, set_grad_enabled, sigmoid, sin, softmax, sort, split, sqrt, stack, sum, tanh, tensor, topk, trainer, vision, where, zeros, zeros_like.

---

## Таблица: PyTorch namespace vs prometorch

### 1. `torch` (top-level functions)

| Категория PyTorch | prometorch имеет | gap |
|---|---|---|
| Tensor factories | tensor, empty, zeros, ones, full, rand, randn, randint, arange, linspace, eye, from_numpy, zeros_like, ones_like | **missing:** `as_tensor`, `range`, `logspace`, `empty_like`, `full_like`, `rand_like`, `randn_like`, `meshgrid` |
| Math/element-wise | exp, log, sin, cos, tanh, sigmoid, relu, softmax, sqrt, rsqrt, clamp, norm, multinomial, einsum, nan_to_num, isinf, isnan, isfinite | **missing:** `add/sub/mul/div/pow` (только как Tensor methods, не top-level), `abs/neg/sign/reciprocal/square`, `tan/asin/acos/atan/atan2/sinh/cosh`, `log2/log10/log1p/expm1`, `floor/ceil/round/trunc`, `clip` (alias) |
| Reduce | sum, mean, max, min, sort, topk, cumsum | **missing:** `argmax/argmin/argsort/median/var/std/prod/any/all/cumprod/logsumexp/allclose/isclose` (на top-level; argmax есть как Tensor method) |
| Comparison | where | **missing:** `eq/ne/lt/le/gt/ge/logical_*/bitwise_*/masked_fill/masked_select/nonzero` |
| Shape | cat, stack, split, chunk | **missing:** `tensor_split/dstack/hstack/vstack/column_stack/row_stack/block_diag/diag/diagflat/diagonal/tril/triu/flatten/unflatten/t/swapaxes/swapdims/movedim/moveaxis/permute/transpose/reshape/view/squeeze/unsqueeze/expand` (всё на Tensor methods, не top-level) |
| Indexing | (нет на top-level) | **missing:** `index_select/gather/scatter/scatter_add/take/put/flip/roll/tile/repeat_interleave/as_strided/narrow/select/unbind` |
| Linear algebra | mm, bmm, matmul, dot | **missing:** `vdot/outer/inner`, namespace `linalg` (нет даже как stub) |
| dtypes | dtype, DeviceType | **missing:** `int8/int16/int32/int64/float16/float32/float64/bfloat16/bool/uint8/complex64/complex128` (нет attribute-level dtype constants) |
| Random | manual_seed | **missing:** `seed`, `get_default_dtype/set_default_dtype`, `get_num_threads/set_num_threads` |
| FFT | — | **missing:** namespace `fft` целиком |
| Autograd | backward, grad, no_grad, enable_grad, is_grad_enabled, set_grad_enabled | OK (no_grad/enable_grad — pure-Python, делегируют в C++ через `set_grad_enabled`) |
| Save/load | save, load, save_state_dict, load_state_dict, save_pytorch, load_pytorch | OK |
| Compile | compile (no-op fallback если _C lacks) | partial — реальная JIT через `pt.jit.compile` |

Top-level gap: ~140 PyTorch функций отсутствуют (см. список в audit-bash output). Из них **критичные для научного кода:** `argmax`, `flatten`, `permute`, `transpose`, `reshape` — присутствуют как Tensor methods, но не как top-level functions, поэтому `pt.argmax(t)` упадёт, а `t.argmax()` работает. PyTorch экспортирует оба вызова — это **compatibility regression** для кода, использующего functional форму.

### 2. `torch.nn`

47 публичных имён есть. **Отсутствует 65 модулей** из стандартного PyTorch nn inventory:

| Категория | prometorch имеет | gap |
|---|---|---|
| Containers | Module, Sequential, ModuleList, ModuleDict, Parameter, Identity | **missing:** `ParameterList`, `ParameterDict` |
| Linear | Linear | **missing:** `Bilinear`, `LazyLinear`, `Flatten`, `Unflatten` |
| Conv | Conv2d | **missing:** `Conv1d`, `Conv3d`, `ConvTranspose1d/2d/3d` |
| Pooling | MaxPool2d, AvgPool2d | **missing:** `MaxPool1d/3d`, `AvgPool1d/3d`, `AdaptiveMaxPool1d/2d`, `AdaptiveAvgPool1d/2d` |
| Normalization | BatchNorm1d, BatchNorm2d, LayerNorm | **missing:** `BatchNorm3d`, `InstanceNorm1d/2d`, `GroupNorm`, `LocalResponseNorm`, **`RMSNorm`** (критично для LLM — qwen/llama/mistral) |
| Dropout | Dropout | **missing:** `Dropout1d/2d/3d`, `AlphaDropout` |
| Sparse | Embedding | **missing:** `EmbeddingBag` |
| Recurrent | RNN, LSTM, GRU | **missing:** `RNNCell`, `LSTMCell`, `GRUCell` |
| Transformer | — | **missing:** `Transformer`, `TransformerEncoder/Decoder`, `TransformerEncoderLayer/DecoderLayer`, **`MultiheadAttention`** (CLAUDE.md заявляет "PARTIAL — код есть, end-to-end никогда не билдился"; в Python API не экспортирован вообще) |
| Pixel/Upsample | — | **missing:** `PixelShuffle/Unshuffle`, `Upsample`, `UpsamplingNearest2d`, `UpsamplingBilinear2d` |
| Activations | ReLU, ReLU6, LeakyReLU, PReLU, ELU, SELU, Sigmoid, Tanh, GELU, SiLU, Mish, Softmax, LogSoftmax, Softplus, Softsign, Hardtanh, Hardsigmoid, Hardswish | **missing:** `Threshold`, `CELU`, `GLU`, `Softmax2d`, `Softshrink`, `Tanhshrink`, `Hardshrink` |
| Loss | MSELoss, CrossEntropyLoss, NLLLoss, BCELoss, L1Loss | **missing:** `BCEWithLogitsLoss`, `KLDivLoss`, `MarginRankingLoss`, `HingeEmbeddingLoss`, `MultiLabelMarginLoss`, `MultiMarginLoss`, `SmoothL1Loss`, `HuberLoss`, `SoftMarginLoss`, `CTCLoss` (CLAUDE.md: "CTCLoss — throw"), `PoissonNLLLoss`, `GaussianNLLLoss`, `TripletMarginLoss`, `CosineEmbeddingLoss` |
| Submodules | functional, init, utils, parallel | OK (parallel — TP/Pipeline через `_C.parallel` либо Python fallback) |

### 3. `torch.nn.functional`

17 функций: relu, leaky_relu, elu, selu, gelu, silu, sigmoid, tanh, softmax, log_softmax, dropout, linear, cross_entropy, nll_loss, binary_cross_entropy, mse_loss, l1_loss.
**Gap vs PyTorch:** отсутствует `conv1d/2d/3d`, `max_pool*`, `avg_pool*`, `adaptive_*_pool*`, `batch_norm`, `layer_norm`, `rms_norm`, `group_norm`, `embedding`, `embedding_bag`, `pad`, `interpolate`, `grid_sample`, `affine_grid`, `one_hot`, `scaled_dot_product_attention` (критично для трансформеров), `multi_head_attention_forward`, `relu6`, `mish`, `hardswish`, `hardsigmoid`, `softplus`, `softsign`, `prelu`, `tanhshrink`, `glu`, `bce_with_logits_loss`, `kl_div`, `huber_loss`, `smooth_l1_loss`, `ctc_loss`, etc. — порядка 80+ функций.

### 4. `torch.nn.init`

7 функций: kaiming_uniform_, normal_, ones_, orthogonal_, uniform_, xavier_uniform_, zeros_.
**Gap:** `kaiming_normal_`, `xavier_normal_`, `trunc_normal_`, `constant_`, `eye_`, `dirac_`, `sparse_`, `calculate_gain`.

### 5. `torch.nn.utils`

Только `clip_grad_norm_`. **Gap:** `clip_grad_value_`, `parameters_to_vector`, `vector_to_parameters`, `remove_weight_norm`, `weight_norm`, `spectral_norm`, `prune.*`, `rnn.pack_padded_sequence/pad_packed_sequence/pack_sequence`, `parametrize.*`.

### 6. `torch.optim`

6 имён: Optimizer, SGD, Adam, AdamW, RMSprop, lr_scheduler.
**Gap (9 optimizers):** `Adamax`, `ASGD`, `LBFGS`, `NAdam`, `RAdam`, `Rprop`, `Adadelta`, `Adagrad`, `SparseAdam`. (CLAUDE.md заявляет 16 optimizers — Python exposes only 4 + base.)

### 7. `torch.optim.lr_scheduler`

8 schedulers: CosineAnnealingLR, ExponentialLR, LRScheduler, LinearLR, MultiStepLR, OneCycleLR, ReduceLROnPlateau, StepLR.
**Gap (8 schedulers):** `CosineAnnealingWarmRestarts`, `PolynomialLR`, `LambdaLR`, `MultiplicativeLR`, `CyclicLR`, `ConstantLR`, `ChainedScheduler`, `SequentialLR`. (CLAUDE.md заявляет 9 видов — Python exposes 8, "missing" большая часть PyTorch стандарта.)

### 8. `torch.utils.data` (mapped to `prometorch.data`)

4 имени: TensorDataset, DataLoader, DataLoaderOptions, Batch.
**Gap (КРИТИЧНО):** `Dataset` (base class!), `IterableDataset`, `ConcatDataset`, `Subset`, `Sampler`, `SequentialSampler`, `RandomSampler`, `BatchSampler`, `WeightedRandomSampler`, `DistributedSampler`, `default_collate`, `random_split`. Без `Dataset` base пользователи не могут писать custom dataset класс — это блокер для любого реального ML-кода.

### 9. `torch.cuda`

2 имени: `is_available`, `device_count`. **Gap:** `current_device`, `set_device`, `get_device_name`, `get_device_properties`, `synchronize`, `empty_cache`, `memory_allocated`, `memory_reserved`, `Stream`, `Event`, `manual_seed`, `manual_seed_all`. CPU-only wheel должен иметь stubs для всех — сейчас просто `cuda.is_available()→False` и больше ничего.

### 10. `torch.autograd`

10 имён (после `prometorch.autograd` import override): DualLevel, backward, grad, jvp, make_dual, unpack_dual, vmap + 3 typing imports утечкой в namespace (Callable, Tuple, annotations — это **API hygiene bug**, надо убрать из публичного API).
**Gap:** `Function` (custom autograd functions — в C++ есть, Python не экспонирует), `gradcheck`, `gradgradcheck`, `Variable` (legacy alias), `set_detect_anomaly`, `detect_anomaly`, `graph.allow_mutation_on_saved_tensors`, `profiler.*`, `functional.jacobian`, `functional.hessian`, `functional.vjp`, `functional.jvp` (есть у нас как top-level `jvp`).

### 11. `torch.amp`

2 имени: `autocast`, `GradScaler`. **CRITICAL:** оба — **no-op stubs из `__init__.py`** (`_AutocastContext` ничего не делает, `_GradScaler.scale(loss)` возвращает loss без масштабирования). Это **silent failure** — пользователь думает что mixed precision работает, а на самом деле его нет. CLAUDE.md: "AMP PARTIAL (API есть, FP16 CUDA kernels нет)". **Должны бросать NotImplementedError**, а не молча работать как identity. **Gap:** `autocast_mode`, `custom_fwd`, `custom_bwd`.

### 12. `torch.distributed`

18 имён: ReduceOp, BackendType, ProcessGroup, DistributedDataParallel, FullyShardedDataParallel, FSDPConfig, DistArgs, init_process_group, all_reduce, broadcast, barrier, launch, get_rank, get_world_size, is_initialized + 3 typing leaks (Callable, Optional, annotations).
**Gap:** `all_gather`, `gather`, `scatter`, `reduce`, `reduce_scatter`, `send/recv`, `isend/irecv`, `new_group`, `destroy_process_group`, `algorithms.*`, `rpc.*`, `optim.ZeroRedundancyOptimizer`. И no `nccl` backend в CPU wheel.

### 13. `torch.jit`

7 имён: ScriptModule, compile, script, trace + typing leaks.
**Реализация:** `compile()/trace()/script()` все — **identity fallback** если `_C.jit_compile` отсутствует. ScriptModule — просто wrapper над callable. Нет save/load для traced graphs, нет `freeze`, `optimize_for_inference`. **Молча работает как no-op** — это compatibility risk.

### 14. `torch.fx`

**Отсутствует полностью.** `prometorch.fx` не существует.

### 15. `torch.compile()` (top-level)

`pt.compile(model, **kwargs)` — try `_C.compile`, fallback returns `model` unchanged. **Silent no-op** — compatible vibe но ноль optimization при отсутствии C++ binding.

### 16. `torch.onnx`

2 имени: `export`, `self_test`. Реальный export через `_C.onnx_export`. **Gap:** `register_custom_op_symbolic`, `dynamic_axes`, `enable_log`, `OperatorExportTypes`.

### 17. `torchvision` (mapped `prometorch.vision`)

4 имени: ImageFolder, mobilenet_v2, transforms + annotations leak. transforms содержит: Compose, Lambda, ToTensor, Transform + 2 typing leaks.
**Gap (массивный):** `datasets.MNIST/CIFAR10/CIFAR100/ImageNet/...`, `models.resnet*/vgg*/efficientnet*/vit_*`, `transforms.Normalize/Resize/CenterCrop/RandomCrop/RandomHorizontalFlip/Pad/ColorJitter/RandomAffine/RandomRotation/RandomGrayscale/...`, `ops.nms/roi_align/box_iou`, `io.read_image/write_image`, `utils.make_grid/save_image`. **3 transforms из ~30+ стандартных.**

### 18. `torch.quantization`

5 имён: QuantizedLinear, fake_quantize, prepare_qat, convert + annotations leak. Pure-Python fallback — `prepare_qat`/`convert` возвращают модель unchanged (**silent no-op**), `QuantizedLinear` throws.
**Gap:** `QConfig`, `default_qconfig`, `MinMaxObserver`, `quantize_dynamic`, `quantize_jit`, etc.

### 19. Дополнительные namespaces (не в стандартном PyTorch / proprietary)

- `prometorch.mlir` — MLIR export (proprietary feature). Real C++ binding, иначе throws.
- `prometorch.mobile` — ExecuTorch-like, real binding или throws.
- `prometorch.trainer` — Lightning-style. Pure-Python `Trainer.fit()`, работает.
- `prometorch.serve` — `LLMEngine`, pure-Python, требует пользовательский forward_fn+tokenizer.

### 20. `torch.Tensor` methods (live `dir(pt.Tensor)`)

89 методов: abs, abs_, add_, argmax, argmin, backward, bmm, bool, clamp, clamp_, clone, contiguous, copy_, cos, cpu, cuda, cumsum, data_ptr, detach, device, dim, div_, dot, double, dtype, exp, exp_, expand, fill_, flatten, float, gather, grad, half, int, is_contiguous, is_leaf, item, log, log_, long, masked_fill, masked_fill_, matmul, max, mean, min, mm, mul_, mv, ndim, neg, neg_, norm, numel, numpy, permute, pow, relu, repeat, requires_grad, requires_grad_, reshape, retain_grad, rsqrt, scatter, scatter_, shape, sigmoid, sin, size, sort, sqrt, sqrt_, squeeze, std, sub_, sum, t, tanh, to, tolist, topk, transpose, type_as, unsqueeze, var, view, zero_.
**Gap (Tensor methods):** PyTorch Tensor имеет ~600+ методов; здесь 89. Отсутствуют `__matmul__` (verify), `floor/ceil/round/trunc`, `tan/asin/acos/atan`, `log2/log10/log1p/expm1`, `argsort`, `cumprod`, `index_*`, `scatter_add_`, `roll/flip/tile`, `nonzero`, `unbind`, `narrow`, `select`, `chunk`, `split`, `bitwise_*`, `eq/ne/lt/le/gt/ge`, `unique`, `unique_consecutive`, `bincount`, `cross`, `inverse`, `pinverse`, `svd`, `qr`, `cholesky`, `triangular_solve`, etc.

---

## Sumcheck / hygiene issues

1. **Typing import leaks в публичный API:** `Callable`, `Optional`, `Tuple`, `List`, `Iterable`, `annotations`, `os`, `time` видны в `dir()` для `autograd/distributed/jit/vision/quantization/onnx/mlir/serve/trainer`. Должны быть `_Callable` или удалены через `__all__`.
2. **AMP молчаливо работает как no-op** (`_AutocastContext` ничего не делает) — silent failure, опасно для production.
3. **`pt.compile()` молчаливо identity** при отсутствии `_C.compile` — должен warn хотя бы раз.
4. **`pt.jit.compile/trace/script`** — identity fallback без warning.
5. **`pt.quantization.prepare_qat/convert`** — identity fallback без warning.
6. **`pt.cuda`** — это lazy-built dict-like объект из `__init__.py` line 184 (`cuda = type('cuda',()...)()`), а не настоящий submodule. `import prometorch.cuda` упадёт.
7. **`prometorch.autograd` потерял типы:** после `setattr(autograd, "backward", _cpp_backward)` namespace mixed с typing imports.
8. **Top-level дубли отсутствуют:** `pt.argmax(t)` работает через delegation? — нет, **отсутствует**. Только `t.argmax()`. Aналогично `pt.flatten`, `pt.permute`, `pt.transpose`, `pt.reshape`. Любой PyTorch-код используя functional форму этих ops сломается.
9. **`pt.dtype` есть как объект, но dtype constants `pt.float32`/`pt.int64` etc. отсутствуют** — невозможно делать `tensor.to(pt.float32)`, нужно знать как construct dtype.

---

## Bottom line

Заявленные ~35-45% PyTorch surface — оптимистично. Реально (по числу публичных имён):
- **Top-level:** 84 vs PyTorch ~600+ → **~14%** (с учётом, что 8 из них — typing leaks)
- **nn:** 47 vs PyTorch ~120+ → **~39%**
- **nn.functional:** 17 vs PyTorch ~100+ → **~17%**
- **optim:** 4 optimizers vs 13 → **~31%**
- **lr_scheduler:** 8 vs 16 → **50%**
- **data:** 4 vs 12 base classes → **~33%** (но без `Dataset` — блокер)
- **autograd:** 7 vs ~20 → **~35%**
- **cuda:** 2 vs ~25 → **~8%**
- **vision:** ImageFolder + 5 transforms vs torchvision ~300+ → **<5%**
- **fx:** 0% (отсутствует)

**Weighted overall ~25-30%**, не 35-45%. Критичные блокеры для PyTorch-compatibility migration: отсутствие `Dataset` base, `MultiheadAttention`, `Transformer*`, `RMSNorm`, `Conv1d/3d`, `Conv*Transpose*`, `scaled_dot_product_attention`, dtype constants (`pt.float32`), top-level functional forms (`pt.argmax`, `pt.flatten`, `pt.reshape`, `pt.permute`). Silent no-op'ы в AMP/jit/compile/quantization — опасны.
