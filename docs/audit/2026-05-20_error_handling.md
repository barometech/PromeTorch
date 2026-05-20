# Аудит ШЕСТЬ — Error handling holes + edge cases в PromeTorch core

Дата: 2026-05-20
Скоуп: `aten/`, `torch/`, `c10/`, `python/csrc/`. Тесты (`test/cpp/test_nmcard.cpp`, `test_tuda.cpp`) сознательно исключены — там `catch(...) FAIL(...)` корректно для testing harness. PromeServe HTTP/JSON код тоже исключён (покрыт Item 59 предыдущего аудита).

Найдено **21** проблема (требовался минимум 15).

## Таблица находок

| # | Location | Issue type | Impact | Proposed fix |
|---|----------|-----------|--------|--------------|
| 1 | `torch/io/gguf_loader.h:702` `read_val<T>()` | EOF detection через `if(!f)` бросает, но `T` читается через `reinterpret_cast<char*>(&val)` — если `T == uint64_t` и файл усечён, `val` остаётся неинициализированным; throw летит ВЫШЕ, но `val` уже стек-локален, ОК. ОДНАКО `parse_tensor_infos` (line 813) делает `tensors.resize(tensor_count)` ДО проверки — если `tensor_count` пришёл из malformed файла = `0xFFFFFFFFFFFFFFFF`, vector::resize кинет `bad_alloc` ИЛИ выделит >100TB и упадёт в OOM | **crash / DoS** | Добавить sanity: `if (tensor_count > 1<<20) throw "GGUF: too many tensors"` ДО resize. То же для `metadata_kv_count` (line 800) |
| 2 | `torch/io/gguf_loader.h:818-822` `parse_tensor_infos` | `n_dims` (uint32_t) без upper bound. Если файл говорит `n_dims = 0xFFFFFFFF` — `t.dims.resize(2^32)` = 32 GB alloc → crash | **crash / DoS** | `if (n_dims > 8) throw "GGUF: too many dims"` |
| 3 | `torch/io/gguf_loader.h:783-790` `read_value` ARRAY branch | `count` clamped только в `reserve` (1M), но цикл `for (i = 0; i < count; ++i)` идёт до конца. Если `count = 10^10` — terabytes RAM через push_back + бесконечный read поток. Allocation bomb / DoS | **crash / DoS** | Заменить `reserve(min(count, 1M))` на `if (count > 100000) throw`; либо break early когда отвалится файл |
| 4 | `torch/io/gguf_loader.h:102-106` `GGUFTensorInfo::n_elements()` | `n *= dim` в int64_t без overflow check. Embedding `vocab=200k, dim=8192` уже = 1.6 * 10^9; реальный malformed файл с dims=[2^31, 2^31] = silent wraparound в негативное → потом передаётся в `at::empty(shape)` с negative size | **silent-corrupt → crash** | `__builtin_mul_overflow` или явная проверка `if (d > 0 && n > INT64_MAX/d) throw` |
| 5 | `torch/io/gguf_loader.h:580` `load_tensor` | `std::vector<uint8_t> raw(raw_bytes)` — если `raw_bytes` отрицательный (overflow в #4) или огромный, vector конструктор throw bad_alloc. **OK что throw, но**: сразу выделяется без upper bound check, легко OOM-ить процесс с малых файлов | **crash / DoS** | Sanity check vs `mmap_handle_.size()` перед allocation |
| 6 | `torch/csrc/autograd/engine.h:80-83 + node.h:246-248` | `Engine::get_default_engine()` — singleton, мутирует `Node::dependency_count_`, `visited_`, `accumulated_grad_` ГЛОБАЛЬНО на heap-shared нодах. Если 2 потока сделают `loss1.backward()` и `loss2.backward()` параллельно над overlapping subgraph (или даже непересекающимися — counter общий) → race condition: visited флаг может остаться `true` с прошлого прогона, `dependency_count_` ушёл в отрицательное. Молча неверные градиенты. | **silent-corrupt** | Mutex по Engine или thread_local GraphTask + локальная hash-map visited вместо in-node флага |
| 7 | `torch/csrc/autograd/engine.h:367` `execute()` | `std::move(const_cast<NodeTask&>(ready_queue.top()))` — `const_cast` + `move` от const ref top: формально UB по стандарту C++ (top возвращает const ref, изменять нельзя). На практике работает с MSVC/GCC, но LCC на Эльбрусе мог бы оптимизировать неправильно | **wrong-result rare** | Использовать `priority_queue` с handle-вытаскиванием или std::set; либо хранить shared_ptr и moveить только указатель |
| 8 | `torch/csrc/autograd/engine.h:159-197` `compute_dependencies` BFS | Нет depth limit. Если граф циклический (пользователь смастерил custom autograd Function с return self) — бесконечный цикл (visited_ от прошлого backward не сброшен, если retain_graph и параллельный поток мутирует). Дополнительно: `next->dependency_count_++` даже если уже visited — может дать неверный count при diamond-graph | **infinite-loop / hang / wrong-result** | Reset visited per backward (уже частично в `task.reset()`); добавить depth limit (например 10^6 nodes) с throw |
| 9 | `aten/src/ATen/native/cpu/MathOps.h:358-359, 395-396` (fill_/zero_ non-contig path) | `idx = rem % sz[d]; rem /= sz[d]` — если ЛЮБОЙ `sz[d] == 0` (empty tensor с зануленной dim) → integer division by zero → SIGFPE crash | **crash** | `if (numel() == 0) return self;` в начале функций |
| 10 | `aten/src/ATen/native/cpu/MathOps.h:100,116` `rsqrt_val`, `reciprocal_val` | `T(1)/std::sqrt(x)`: при `x<0` → NaN silently; при `x=0` → +Inf silently. То же `reciprocal_val(0)` → Inf. Различия от PyTorch (тоже допускает) — но без TORCH_CHECK debug-fallback нельзя ловить пользовательские баги | **silent-corrupt** | Добавить `PT_CHECK_DEBUG(x > 0)` в debug build (через `PT_DASSERT`-обёртку — но НЕ `assert`, а conditional throw) |
| 11 | `c10/macros/Macros.h:98` `PT_ASSERT(cond) → assert(cond)` | В Release сборках с `-DNDEBUG` ВСЕ `PT_ASSERT` исчезают. Если код использует `PT_ASSERT` для runtime invariants (например shape-checks) — в Release они тихо пропускаются, баги доходят до production | **silent-corrupt / wrong-result** | `PT_ASSERT` должен ВСЕГДА проверять и `throw`/`abort`. Для опт-аут использовать `PT_DASSERT` (debug-only), уже есть |
| 12 | `c10/core/Allocator.h:276` `CPUAllocator::CPUAllocator()` | `arena_ = static_cast<char*>(aligned_alloc_impl(kArenaSize))` без nullptr check. Если аллокация arena (типично 16-64MB) провалилась — `arena_offset_` сравнивается с `kArenaSize` через nullptr+offset позже = SIGSEGV при первом мелком alloc | **crash** | `if (!arena_) { /* disable arena path */ }` |
| 13 | `torch/optim/optimizer.h:305` `load_state_dict` | `size_t idx = std::stoull(idx_str)` без try/catch. Malformed checkpoint с не-числовым ключом param_state → uncaught `std::invalid_argument` → процесс падает посреди load | **crash** | Обернуть в try/catch и `continue` с warning |
| 14 | `torch/io/tokenizer.h:538` `decode_token` byte-token | `std::stoi(token.substr(3,2), nullptr, 16)` без try/catch. Если злоумышленник передал `<0xZZ>` или token имеет невалидный hex — uncaught throw в decode loop → весь decode свалится. Bonus: token может быть length=6 но '\0'-инжектится через substr, edge | **crash on bad input** | `try { ... } catch (...) { return "?"; }` |
| 15 | `torch/data/dataloader.h:518-523` `count_batches` | `n / batch_size` и `(n + batch_size - 1) / batch_size` — если `batch_size == 0` → integer division by zero → SIGFPE. Default 1 но `DataLoaderOptions().batch_size_(0)` валиден из API | **crash** | `if (batch_size == 0) throw std::invalid_argument(...)` в DataLoaderOptions setter |
| 16 | `torch/data/dataloader.h:286-290` `PrefetchContext::record_error` | Только `first_error` хранится. Если worker A и worker B оба упали разными exception'ами — B потерян. При debug многопоточных проблем неинформативно. Не критично для correctness | **silent-corrupt (debug info loss)** | `std::vector<std::exception_ptr> errors_` + лог всех |
| 17 | `torch/distributed/fsdp.h:144-147` `touch(path)` | `if (f) { ... }` — если fopen вернул NULL (нет permissions, диск полный), функция silently не создаёт файл. Другие ranks навечно зависнут в `wait_file(path, timeout_ms=120000)` пока не упадут по timeout. 120s wasted перед exception | **hang / wasted-time** | Throw runtime_error если fopen NULL |
| 18 | `examples/pir/grad_sync.h:74-84` `sync()` | Short write / fopen failed — printed to stderr, продолжаем sync с corrupt данными (avg будет неверный). Distributed training сходит с ума без error signaling | **silent-corrupt → wrong training** | Throw / std::abort на short write, иначе всё ranks averaging garbage |
| 19 | `python/csrc/tensor_bindings.cpp:169-174` Tensor.item() | `t.item<double>()` без проверки `t.defined()`. Undefined tensor (default-constructed) → `numel()` undefined behavior. Также: не проверен dtype — `item<double>()` на int64 даст bit-reinterpret garbage | **crash on undefined / wrong-result on type-mismatch** | `if (!t.defined()) throw; PT_DISPATCH_ALL_TYPES(t.dtype(), ...)` |
| 20 | `python/csrc/tensor_bindings.cpp:63` `numpy_to_tensor` | `std::memcpy(tensor.data_ptr(), buf.ptr, nbytes)` без проверок: data_ptr может быть nullptr для empty tensor; buf.ptr может быть nullptr для 0-D numpy array; tensor создаётся без проверки `buf.ndim < 32` (overflow при больших shapes); int8/uint8 numpy с size>2GB переполнят size_t на 32-bit | **crash / silent-corrupt** | Bounds check, nullptr guard, ndim limit |
| 21 | `torch/serialization.h:73-105` `read_tensor` | (a) `std::string name(name_len, '\0')` — `name_len` uint32 unbounded = 4GB allocation; (b) `std::vector<int64_t> sizes(ndim)` — `ndim` uint32 unbounded; (c) `at::empty(sizes, dtype)` без валидации sizes (negative? overflow?); (d) `dtype_raw` через `static_cast<c10::ScalarType>` — silent OOB enum value, потом `at::empty` примет invalid dtype. Atatack vector: corrupt .ptsave файл = OOM/crash при load | **crash / DoS / silent-corrupt** | Bound name_len (1MB), ndim (8), валидировать dtype через whitelist switch, проверить sizes на >0 и overflow |

## Дополнительные observations (не в основной таблице)

- `python/csrc/tensor_bindings.cpp:343, 1096-1119, 1141-1150` — `catch (...) {}` в `to()` / `extract_cpp_child` / `PyCompiledModule`. Здесь это **намеренный** try-multiple-cast паттерн pybind11, проходит fallthrough к следующему cast. Не баг, но fragile — если cast бросит **не**-`pybind11::cast_error`, исключение проглотится и пользователь не узнает почему compile упал.
- `promeserve/model_manager.h:311-313` — `try { fs::file_size } catch(...) {}` оставляет `size_bytes` неинициализированным (хотя default = 0 в struct, OK).
- `aten/src/ATen/nmcard/NMCardHardware.cpp:66` — `catch(...) {}` в dtor: корректный паттерн (исключения из dtor запрещены).
- `torch/io/gguf_model.h:540-543` `KVCache::append` CPU memcpy: `new_k.size(1)` без проверки `new_k.dim() >= 2`. Сейчас всегда вызывается правильно forward-pass'ом, но publicly-exposed.

## Сводка по severity

| severity | count |
|----------|-------|
| crash | 8 (#1, #2, #9, #12, #13, #14, #15, #19 part) |
| crash + DoS (untrusted file input) | 4 (#3, #5, #20, #21) |
| silent-corrupt / wrong-result | 6 (#4, #6, #10, #11, #16, #18, #19 part) |
| hang / infinite-loop | 2 (#8, #17) |
| UB (rare) | 1 (#7) |

## Топ-3 приоритета на починку

1. **#6 Engine race condition** — singleton + in-node mutable state. Молча испорченные градиенты в multi-threaded training (PIR Local SGD, DDP). Самый опасный — нет видимого симптома.
2. **#11 PT_ASSERT в Release** — массивная неконтролируемая дыра по всему коду; нужен audit всех `PT_ASSERT` call-sites чтобы понять масштаб.
3. **#21 + #1/#2/#3 GGUF/serialization allocation bombs** — single malformed file = OOM crash / RCE-like surface. Если пользователь грузит untrusted GGUF (HuggingFace mirror), это denial of service.
