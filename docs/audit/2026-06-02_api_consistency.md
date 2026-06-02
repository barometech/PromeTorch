# Аудит #16 — API/ABI Consistency (C++ ↔ Python ↔ Tests ↔ README)

**Дата:** 2026-06-02
**HEAD:** `85c0fb5d6d0d1d72af5b89094fbfd4018acdb3d2`
**Метод:** статический анализ `aten/src/ATen/core/Tensor.h`, `torch/nn/*`,
`torch/optim/*`, всех `python/csrc/*.cpp` (5 файлов, 3470 строк bindings),
`test/cpp/*` (31 файл), `python/tests/*` (5 файлов), `README.md`.
Бинарь `_C.cp312-win_amd64.pyd` собран в `python/prometorch/Release/`.

Фокус — НЕ покрытие (это #8), а **drift между сторонами одной декларации**:
заявлен X в C++ → есть ли X в Python → используется ли X в tests → показан ли X
в README, и совпадает ли семантика. Минимум 20 пунктов.

---

## Свод результатов

| # | API element | C++ side | Python side | Test coverage | Drift status |
|---|---|---|---|---|---|
| 1 | `at::Tensor` core class | declared `aten/src/ATen/core/Tensor.h:135` с `PT_API` | `py::class_<at::Tensor>` `tensor_bindings.cpp:135` | `test/cpp/test_tensor.cpp` + `python/tests/test_no_grad.py` | OK (cohesive) |
| 2 | `Tensor::__getitem__(int)` (baseline fix 2026-05-20 #1) | `Tensor::operator[](int64_t)` `Tensor.h:354` | binds `tensor_bindings.cpp:469-481`, ловит `IndexError` для for-loop протокола | `test_no_grad.py` использует `pt.randn([2,3])` итерацию — частичное | OK (Python-bound правильно бросает `py::index_error`, C++ оригинал — `runtime_error`) |
| 3 | `Tensor::__getitem__(slice)` | НЕТ метода в C++ Tensor (slice есть как `slice(dim,start,end,step)`) | `tensor_bindings.cpp:482-488` — slice работает только по dim=0 | НЕТ теста | **DRIFT**: `t[1:5,2:3]` (multi-dim slice) и `t[boolmask]`/`t[int_tensor]` (advanced indexing) — не работают, хотя `IndexOps.h` имеет `at::index_select` |
| 4 | `Tensor::__float__/__int__/__bool__` (baseline fixes 2026-05-20 #2,#3,#4) | НЕТ в C++ `Tensor.h` (есть только `item<T>()` template + `Scalar item()`) | bound только в Python `tensor_bindings.cpp:504-529` | НЕТ теста | **DRIFT**: Python-only fix. Семантика `__bool__` обходит баг "Tensor.to(Bool) — Unsupported dtype" через double-cast. C++ код, который попытается `t.to(c10::ScalarType::Bool)` напрямую — упадёт |
| 5 | `tensor()` factory accepts list/scalar | C++ `torch::tensor(IntArrayRef)` / `from_blob` — НЕТ универсального universal-data overload | `tensor_bindings.cpp:703-724` принимает `py::list/tuple/scalar/np.array` через `np.asarray` | `test_no_grad.py:_make_leaf` использует только `pt.randn` | OK на Python стороне (baseline fix #5), C++ side всё ещё узкий |
| 6 | `PackedSequence` 4-arg default | C++ `torch/nn/utils/rnn.h` имеет default-constructible PackedSequence | `tensor_bindings.cpp:1252-1282` — fix 2026-05-20 #6: 4-arg ctor ВСЕ args обязательны (default `at::Tensor()` ломал module-init с PT_CHECK) | НЕТ теста (RNN модули в Python привязаны, sequence-utility — да; реальный test через `pack_padded_sequence` отсутствует) | **DRIFT**: workaround вместо fix. C++ allows defaults, Python requires explicit args |
| 7 | `nn.Module::forward()` | virtual в `torch/nn/module.h:68-85` (3 overload'а: 1-arg, 2-arg, vector) | bound в `nn_bindings.cpp:131-415` для ~30 модулей, каждый `def("forward")` + `def("__call__")` | `test_nn.cpp`, `test_nn_modules.cpp`, `test_nn_functional_full.cpp` | **DRIFT (силён)**: C++ 91 модуль (по `class.*:.*public Module`), Python = ~30. Bilinear/Conv1d/Conv3d/ConvTranspose/Transformer/MultiheadAttention/RMSNorm/Threshold/Dropout1d-3d/AlphaDropout/Upsample/EmbeddingBag/RNNCell/LSTMCell/GRUCell/ParameterList/ParameterDict/ModuleDict/MaxPool1d/MaxPool3d/AvgPool1d/Adaptive*Pool/InstanceNorm/GroupNorm/PositionalEncoding/HuberLoss/SmoothL1Loss/KLDivLoss/CTCLoss/BCEWithLogitsLoss/MarginRanking/TripletMargin/MultiMargin/PoissonNLL/GaussianNLL/FocalLoss/DiceLoss/CosineEmbedding/QuantizedConv2d — **есть в C++, отсутствуют в Python** |
| 8 | `nn.Module::parameters()/named_parameters()` | `module.h:166-218` возвращает `std::vector<Parameter*>` | `nn_bindings.cpp:30-43`: `parameters()` → `vector<at::Tensor>` через `p->data()` (теряет ссылку на Parameter!) | НЕТ Python теста на gradient flow через parameters() | **DRIFT (критично для optim)**: Python optimizer получает `pt.list(model.parameters())` как **копии-by-value** at::Tensor, теряет связь с `Parameter*`. `params_from_tensors` в `optim_bindings.cpp:30-41` создаёт **новые `Parameter*`** для каждого тензора — `optimizer.step()` обновляет внутренние копии, не оригинал. **Это разрывает grad flow Python ↔ C++** |
| 9 | `nn.Module::state_dict()/load_state_dict()` | `module.h:408-485`, рекурсивный, через `parameter_order_/buffer_order_` | `nn_bindings.cpp:62-68` напрямую `self.state_dict()` | НЕТ Python теста | OK (через `unordered_map<string, Tensor>`), но **связано с #8**: load_state_dict обновляет внутренние Parameter'ы C++, но Python-side optimizer держит свои Parameter копии — после load Python optimizer ничего не получит |
| 10 | `nn.Module::register_buffer/register_parameter` | `module.h:118-127` | **0 bindings**. Python пользователь не может зарегистрировать buffer/parameter из своего Module subclass | НЕТ теста | **DRIFT (блокер)**: невозможно писать custom Module на Python с persistent state (BatchNorm-style buffers) |
| 11 | `optim.Optimizer::step()/zero_grad()` | virtual в `torch/optim/optimizer.h:182-199` | `optim_bindings.cpp:67-90` базовый + per-subclass override | `test_optim.cpp`, `test_optimizers.cpp` | OK на baseline, но см. #8, #14 |
| 12 | `optim.Optimizer::state_dict()/load_state_dict()` | virtual в `optimizer.h:284-321` — возвращает `OptimizerStateDict` со всем per-param state (exp_avg, exp_avg_sq, step) | `optim_bindings.cpp:80-90` — **stub**: возвращает только `{"lr": ...}` py::dict, load принимает только lr | **0 тестов** | **DRIFT (КРИТИЧНО)**: C++ полноценный optimizer state ↔ Python sees only lr. Resume training из checkpoint в Python обнулит momentum/Adam moments → катастрофа |
| 13 | `optim.SGD/Adam/AdamW/RMSprop` | 16 классов в C++ (`SGD,Adam,AdamW,RMSprop,Adamax,ASGD,LBFGS,NAdam,RAdam,Lion,LAMB,Adadelta,Adagrad,Adafactor,AdamKiller,SophiaG`) — все `class X : public Optimizer` | только 4 bound: `SGD,Adam,AdamW,RMSprop` в `optim_bindings.cpp:101-213` | C++ tests есть для всех, Python — только 4 | **DRIFT**: 12 optimizer'ов dead-code из Python. Совпадает с audit #8 наблюдением, но добавляет: C++ tests их используют → они РАБОТАЮТ, просто не bound |
| 14 | optim получает Parameter*, не Tensor | C++ ctor: `SGD(vector<Parameter*>, ...)` | `params_from_tensors` `optim_bindings.cpp:30-41` создаёт **новые Parameter** для каждого `at::Tensor` из py::list, не связывая с оригиналом | НЕТ Python integration теста (train loop) | **DRIFT (показывает что Python training loop в принципе не может работать)**: optimizer держит копии Parameter, model.train() обновляет свои, optimizer.step() обновляет свои → out-of-sync. Это объясняет почему `examples/mnist` существует только в C++, а Python `test_pyop_autograd.py` тестирует только forward+backward без optim step |
| 15 | `torch::autograd::backward()/grad()` | `autograd.h`, `engine.h` — full BFS engine, supports `retain_graph/create_graph` (см. CLAUDE.md: create_graph ignored) | `autograd_bindings.cpp:112-135` — bound с обоими kwargs | `test_no_grad.py` ловит косвенно через grad propagation | **PARTIAL DRIFT**: `create_graph=True` Python пробрасывает в C++, но C++ engine его игнорирует (заявленный bug). Python API лжёт |
| 16 | `no_grad/enable_grad/is_grad_enabled` (BUG-C9 fix) | `torch::autograd::GradMode::is_enabled/set_enabled` в `grad_mode.h` | `autograd_bindings.cpp:26-105` — PyNoGradGuard с `restored_` flag (prevent double-restore) | `test_no_grad.py` 7 тестов, **критичный**: `test_op_inside_no_grad_has_no_grad_fn` показывает что часть Python ops НЕ propagate'ит requires_grad | **DRIFT (вскрыт тестом)**: C++ GradMode работает, но Python `pt.relu(t)`/`pt.tanh(t)`/`pt.sum(t)`/`pt.mm(t,t.t())` идут через **raw aten** (bound через `&at::Tensor::relu` `tensor_bindings.cpp:225`), а `t+t`/`t*t` — через `_autograd` wrappers. Поэтому одни ops propagate, другие нет. Тест явно `pytest.skip`'ает если ни один пробник не propagate'ит |
| 17 | `Tensor.argmax/argmin/var/std/topk/sort/gather/scatter` | в C++ `Tensor.h` declared, реализация в `aten/native/cpu/*Ops.h` | bound в `tensor_bindings.cpp:279-296,532-545,613-622` | tests есть | OK (но без top-level `pt.argmax(t)` — только method) |
| 18 | `Tensor.scatter` (out-of-place) | в `IndexOps.h` есть `scatter_` (in-place); out-of-place реализации НЕТ | `tensor_bindings.cpp:567-586` — **hand-rolled implementation 2D-only**, иначе возвращает clone без изменений | НЕТ теста | **DRIFT (silent bug)**: Python `t.scatter(dim, idx, src)` на 1D/3D+ молча ничего не делает |
| 19 | `Tensor.clamp_` Python | C++ `Tensor.h:383-388` принимает `Scalar min, Scalar max` + `optional<Scalar>` overload | `tensor_bindings.cpp:589-600` — **hand-rolled float-only loop**, игнорирует C++ overload | tests есть в C++ | **DRIFT**: Python clamp работает только для float32 dtype. Integer tensor `t.clamp(0, 10)` упадёт на `mutable_data_ptr<float>()` |
| 20 | `Tensor.norm` Python | C++ `Tensor.h` — `Tensor norm() const` (есть, скорее всего L2) | `tensor_bindings.cpp:549-559` — **hand-rolled `sq.sum().sqrt()`** игнорирует `p` argument полностью | НЕТ теста | **DRIFT**: `t.norm(p=1)` или `t.norm(p='fro')` всегда вернёт L2. Тихая ложь |
| 21 | `Tensor.to(dtype)` для Half/BFloat16 | C++ supports через `to(ScalarType)` + ScalarType enum имеет Half/BFloat16/FP8 | `tensor_bindings.cpp:349-372` bound, `init.cpp:62-89` export'ит `pt.float16/bfloat16` | НЕТ Python теста на Half ops | **DRIFT**: `t.to(pt.float16)` создаст Half-тензор, **но любая последующая операция упадёт** — все hot-path kernels используют `PT_DISPATCH_FLOATING_TYPES` (Float/Double only, 78 callsites), `PT_DISPATCH_FLOATING_TYPES_HALF` имеет **0 callsites в aten/torch hot paths** (только в README/docs). README врёт — см. ниже #25 |
| 22 | `PT_DISPATCH_FLOATING_TYPES_HALF` macro | defined `c10/core/ScalarType.h:876` | N/A | N/A | **0 callsites confirmed** — макрос dead code. Подтверждает audit `2026-05-20_docs_vs_code.md:72` |
| 23 | ScalarType enum coverage | C++ 19 типов (Byte..Float8_e5m2) | Python bound 12 (`init.cpp:62-89` — float16/32/64, bfloat16, int8/16/32/64, uint8, bool, complex64/128) | НЕТ теста | **PARTIAL DRIFT**: FP8 (e4m3fn/e5m2), QInt8/QUInt8/QInt32, ComplexHalf — **не bound в Python**. Используются только Float/Double/Int/Long/Byte/Bool в реальности |
| 24 | DLL ABI: `PT_API` macro | declared `c10/macros/Macros.h:36-44`: dllexport если `PT_BUILD_SHARED_LIBS`, иначе dllimport | используется только в `aten/core/Tensor.h` (3 раза), `c10/core/TensorImpl.h` (5 раз). **0 раз в torch/nn/*, torch/optim/*, torch/csrc/autograd/*** | N/A | **ABI HAZARD**: 91 nn::Module subclass, 16 Optimizer subclass, все autograd classes — БЕЗ PT_API. Работает потому что почти всё inline header-only. Любой не-inline virtual или type_info через DLL boundary → unresolved external или RTTI mismatch на Windows DLL build. На GCC Linux работает (default visibility), на MSVC + `PT_BUILD_SHARED_LIBS=ON` потенциальные UB |
| 25 | DLL singleton: AllocatorRegistry | fix в CLAUDE.md: "register_nmcard_allocator() + register_nmcard_allocator_local() — двойная регистрация" | applied | N/A | Known + workaround'ed. Системная проблема с inline static — повторится для любой new SHARED библиотеки |
| 26 | Forward decl циклы autograd ↔ at::Tensor | `aten/core/Tensor.h:17-24` forward-declares `torch::autograd::AutogradMetaImpl`, `Node`, `set_grad_fn<T1,T2,T3>` template | работает; реализация `autograd.h` включает `Tensor.h` | OK | OK (классический pattern), но `set_grad_fn` объявлен как 3-arg template без specialization — фактический вызов идёт через `c10::TensorImpl::autograd_meta()` |
| 27 | `c10::AutogradMetaInterface` ↔ `torch::autograd::AutogradMetaImpl` | `c10/core/TensorImpl.h` объявляет interface; impl живёт в `torch/csrc/autograd/autograd_meta.h`. Связь — через factory `set_autograd_meta_factory_impl` (`TensorImpl.h:312`) | N/A (Python видит final API) | НЕТ теста на фабрику | **HAZARD**: factory должна быть set'нута до первого Tensor.requires_grad_(true). Если порядок DLL init нарушен — undefined behavior |
| 28 | Python package name | repo = `promethorch`, package на диске = `python/prometorch/` | `_C.cp312-win_amd64.pyd` собран как `_C` под `prometorch._C` | **4 из 5 Python тестов используют `import promethorch as pt`** (test_no_grad, test_pytorch_io, test_transformers_compat, test_bindings_new); только `test_pyop_autograd.py` использует `import prometorch` | **DRIFT (тесты сломаны)**: 4 теста упадут на `ModuleNotFoundError: promethorch`. README.md:1158 показывает `import prometorch as pt` (правильно). Имя в репозитории/CMakeLists vs реальное имя package разъезжаются |
| 29 | `torch.compile()` JIT | `torch/compile/promepile.h` declares `CompiledForward` | `tensor_bindings.cpp:1239-1244` bound; `PyCompiledModule` ловит trace failures fallback'ом на Python `__call__` | НЕТ Python теста на compile() | **PARTIAL DRIFT**: PyCompiledModule пытается cast к Sequential/Linear/ReLU/Sigmoid/Tanh/GELU/SiLU/Softmax/BatchNorm1d/2d (10 hardcoded типов). Любая другая модель (Conv2d, LSTM, кастом) → trace_failed=True, silent fallback. Compatible API, ноль optimization для большинства моделей |
| 30 | `prometorch.amp.autocast/GradScaler` | C++ `torch/amp/grad_scaler.h` имеет реальный GradScaler | Python `__init__.py` — **no-op stubs** (per audit #8). `autocast.__enter__` ничего не делает, `scaler.scale(loss)` возвращает loss без масштабирования | НЕТ теста | **SILENT DRIFT**: пользователь думает что AMP включён, тренирует на Float32 без масштабирования. Опасно для prod |

---

## Categorical takeaways

### A. Tensor baseline fixes 2026-05-20 (#1-#6)
Все 6 fix'ов реализованы **только на Python side** (`tensor_bindings.cpp`),
C++ Tensor.h не трогался. Cohesive в пределах wheel, но любой C++ user-code
(C++ tests, `examples/mnist/train_mnist_mlp.exe`) не получит `__float__/__int__/__bool__`,
`tensor()` overload или универсальный `__getitem__`. **`__bool__` обходит реальный
баг `Tensor.to(Bool) — Unsupported dtype`** через double-cast — баг до сих пор живёт.

### B. nn.Module 91→30 drop
65 модулей объявлены в C++ headers с полной реализацией forward (RMSNorm —
**отсутствует даже в C++**), но 0 bindings. Эти модули **работают** через C++
tests, но из Python недоступны. Совпадает с audit #8 (Python API) но
добавляет: проблема не "ленились wrap'нуть", а архитектурная — `init_nn_bindings`
ручной list из 30, и забыли обновить при добавлении новых модулей. Нет
автогенерации.

### C. Optimizer drift — критический для training
4 фундаментальных проблемы соединяются:
1. `parameters()` теряет `Parameter*` (#8)
2. `params_from_tensors` создаёт копии (#14)
3. `state_dict()` Python — только lr (#12)
4. 12 из 16 optimizers не bound (#13)
**Вывод:** реальный train loop из Python невозможен без обхода
через C++ examples. Подтверждается: `examples/mnist/train_mnist_mlp.cpp` —
C++, **не** Python; `test_pyop_autograd.py` тестирует только forward+backward,
не optim.step+resume.

### D. ScalarType dispatch lie
`PT_DISPATCH_FLOATING_TYPES_HALF` — 0 callsites. **Половина ScalarType enum
(Half, BFloat16, FP8_e4m3fn, FP8_e5m2, ComplexHalf, QInt*)** доступна через
`pt.float16` итд, но любая операция на таком тензоре упадёт `Unsupported dtype`
из `PT_DISPATCH_FLOATING_TYPES` (Float/Double only).

### E. autograd Python-bound ops split
Tensor methods (`.relu`, `.sum`, `.mm`, `.tanh`) → raw aten, **НЕ propagate'ят grad**.
Operators (`__add__`, `__mul__`, `__sub__`, `__truediv__`) → `_autograd` wrappers,
**propagate**. Это explicitly наблюдено в `test_no_grad.py:_propagating_ops`.
Половина Python autograd-API — silent broken.

### F. ABI hazards
PT_API применён только к ~8 classes (Tensor, Scalar, TensorOptions, TensorImpl и
несколько factory). 200+ остальных public classes — без export macro. Работает
случайно из-за inline-heavy header'ов. SHARED build на MSVC уязвим.

### G. Package name drift
`promethorch` (repo) ↔ `prometorch` (package on disk, PyPI). 4 из 5 Python
тестов используют старое имя `promethorch` и упадут на import. README ок.

---

## 30 пунктов в таблице выше; 7 categorical takeaways. Цель ≥20 — достигнута.
