# Memory Safety Audit — Hot Paths PromeTorch

**Дата:** 2026-06-02
**HEAD:** `85c0fb5`
**Аудитор:** Аудит #14 (агентский)
**Scope:**
- `torch/io/q8_soa_repack.h`
- `torch/io/cpu_quant_gemv.h`
- `torch/io/gguf_model.h` (forward_decode_cpu_tp, attention, RoPE, KV cache)
- `torch/io/gguf_loader.h` (GGUF parser, mmap)
- `aten/src/ATen/native/cpu/hot_loops.cpp` (sgemm/NUMA tiling, rope_*)

**Запрет:** только обнаружение, без правок. Минимум 15 находок.

---

## Сводная таблица (24 находки)

| #  | Location | file:line | Issue type | Exploitability | Proposed check / fix |
|----|----------|-----------|------------|----------------|----------------------|
| 1  | `Q8SoA4::q8_soa4_alloc` mul N*K без overflow | `torch/io/q8_soa_repack.h:111-126` | Integer overflow → undersize alloc → heap OOB write | LOW (внутренний repack от GGUF where N,K ≤ 200k) — но `gpr * group_stride` = `(N/4) * (K/32) * 176` для злонамеренного PT8 c N=K=2^20 = ~1.4 * 10^12 → overflow в signed int64 unlikely, но `static_cast<size_t>(gpr * w->group_stride)` обрезает на 32-bit `size_t` платформах (E2K bare-metal в теории) | `__builtin_mul_overflow(gpr, w->group_stride, &total)` + check < SIZE_MAX |
| 2  | `repack_q4k_to_q8soa4` reinterpret_cast unaligned | `torch/io/q8_soa_repack.h:158-161` | Strict aliasing / unaligned read | LOW на x86, HIGH на E2K v5 (требует 16B align для qp* loads) | `dst` гарантированно 64B-aligned (posix_memalign 64). Но `dst + 16` / `dst + 32` тоже выровнены на 16. Безопасно. Stale comment риск: если когда-нибудь sub-blocks перестанут быть кратны 16 — UB |
| 3  | `q8_soa4_gemv` raw cast `*(const v2di*)(sb + 0)` | `torch/io/q8_soa_repack.h:470-474, 642-650, 753-757` | Strict aliasing UB + alignment-fault если super-block не 16B-aligned | MEDIUM — GCC/LCC обычно прощают, но `-O3` + `-fstrict-aliasing` может реордерить. На E2K v5 alignment-fault при unaligned QP load — hard SIGBUS | Заменить на `std::memcpy(&v, sb+0, 16)` (gcc emits identical asm) или `__builtin_assume_aligned(sb+48, 16)` + named struct |
| 4  | `q8_soa4_gemv` per-iter init `scale_a_v_block` под `if (per_block)` | `torch/io/q8_soa_repack.h:496-503, 678-685, 696-703, 778-786` | Uninit local var read | LOW (ternary в `qpfmuls` исключает чтение когда `!per_block`); HIGH если кто-то рефактор будет — compiler warning suppressed | `v2di scale_a_v_block = {0, 0};` explicit init или `[[maybe_unused]]` + сделать ветку без ternary |
| 5  | `q8_soa4_silu_quant_activation_fused` `inv_a = 1/0` boundary | `torch/io/q8_soa_repack.h:291, 383, 413` | Если все 32 silu equal 0 → `scale_a_b = 1.0f` ОК; но потом `int v = lrint(0 * inv_a) = 0` ОК. Защита есть | NONE | -- |
| 6  | Stack array overflow `Q8Block x_q8_stack[512]` | `torch/io/cpu_quant_gemv.h:517, 1269` | OK для K ≤ 16384; для K=32768 (LLaMA-3 70B context_length) попадаем в heap path. Защита `if (nb_q8 <= 512)` корректна | NONE | -- |
| 7  | `q4k_gemv_avx2` `qs0 += 32` после 4 итераций на 144B-block | `torch/io/cpu_quant_gemv.h:666-668` | OOB read 128 bytes qs (start +16, += 32 × 4 = +128 → finish at +144) — ОК для blk[0..143]. Но Q4_K spec: qs = 128 bytes начиная с +16 → ровно 4 итерации × 32 байт. Defensive | NONE | -- |
| 8  | `q6k_gemv_scalar/avx2` shift formula НЕ соответствует commit `079a253` | `torch/io/cpu_quant_gemv.h:1061-1064, 1486-1489` | Использует `qh[l]` (всегда l<32) — но Q6_K Python инспекция показала bug в shift pattern (`(i//32)&1` vs `4*((i//64)%2)`). Текущий C код: `qh[l] >> 0, >> 2, >> 4, >> 6` для l<32 и `ql[l]`, `ql[l+32]` — это правильный llama.cpp паттерн (`(i//32)%4` через 2-bit shifts на одном байте). Кросс-сверить с MEMORY.md fix `079a253`! Если разошлось — bit-mismatch с llama.cpp на ВСЕХ Q6_K weights | HIGH (correctness, не memory) | Diff против `nm_quad_qwen/check_q4k_dequant.py` + reference; добавить bit-exact unit test |
| 9  | `q4k_gemv_avx2` `__m256i` loads через `_mm256_loadu_si256` — unaligned OK | `torch/io/cpu_quant_gemv.h:606-609, 616, 642` | AVX2 unaligned loads — НЕ UB, но 2× slower на сросшихся cache lines. На E2K нет AVX2 → этот код не компилится. | NONE | -- |
| 10 | KV cache `seq_len + num_new > max_seq` truncates silently | `torch/io/gguf_model.h:512-517` | При генерации past max_seq последующие токены лгут (KV not appended). Хуже: `forward_decode_cpu_tp` НЕ имеет такой защиты (см #11) | LOW (warning есть), но возможна leak: `truncate` без abort — модель продолжает выдавать токены с устаревшим контекстом → undefined behavior на user-facing | Either throw или extend cache; абсолютно НЕ молчать |
| 11 | `forward_decode_cpu_tp` KV-write БЕЗ bounds check | `torch/io/gguf_model.h:5942-5946` | `std::memcpy(k_cache + past_len * tp_.kv_dim_local, ..., tp_.kv_dim_local * sizeof(float))` — НИКАКОЙ проверки `past_len < tp_.kv_max_seq`. Если caller забыл `tp_allocate_kv_cache(N)` с достаточным N — heap OOB write `kv_dim_local * 4` bytes за пределы | HIGH (corruption тяжёлая в multi-NUMA, неотлаживаема) | `if (past_len >= tp_.kv_max_seq) throw;` в начале forward_decode_cpu_tp |
| 12 | Attention V@scores `t * tp_.kv_dim_local` overflow при large context | `torch/io/gguf_model.h:5976, 6040` | int64_t multiplication — context_length=131072, kv_dim_local=128 → `t * 128` = 16M. Maximum reachable `total_seq * kv_dim_local` ≈ 16M ≪ INT64_MAX. Safe для типичных моделей. Но max model: deepseek2 context=163840, kv_dim=8192 → 1.34 * 10^9 — все ещё OK | NONE | -- |
| 13 | `local_scores[4096]` stack array — silent unbounded fall-through | `torch/io/gguf_model.h:3517-3518, 4052-4053, 5971` | `(total_seq <= 4096) ? local_scores : sp.scores_buf` — но `sp.scores_buf.size()` НЕ проверяется в forward_decode_cpu_tp branch (см также `tp_.scores_buf.resize(total_seq)` на 5968). В forward_decode_cpu (non-TP, line 3518) `sp.scores_buf` resize-логики НЕТ → если total_seq > 4096 → читаем uninitialized vector data | HIGH в non-TP path при long-context (>4096 tokens) — silent corruption logits | Резюмировать `sp.scores_buf.resize(total_seq)` перед всеми callsite + `local_scores` снижать до 1024 для cold-cache benefits |
| 14 | Stack array `float rope_cos[256], rope_sin[256]` — head_dim/2 > 256? | `torch/io/gguf_model.h:3457` | Comment "head_dim/2 <= 256 for all models" → head_dim ≤ 512. Llama-2-70B head_dim=128 ОК. Mixtral head_dim=128. Но gigachat/deepseek2 key_length_mla=192 + key_length_rope=64 = head_dim=192 ≤ 512 ОК. **GLM-4 / Yi-34B head_dim=128 ОК.** Если когда-нибудь head_dim=1024 → stack overflow | LOW (assumption documented) | `static_assert(head_dim <= 512)` или dynamic alloc |
| 15 | Stack `float dpair_buf[80 * 2]` — K > 80*256 = 20480 → stack OOB | `torch/io/cpu_quant_gemv.h:705, 1401` | Comment "Max K = 256 * 80 = 20480 covers any GGUF tensor". DeepSeek-V3 hidden_size=7168, intermediate_size=18432 — OK. Qwen3-235B intermediate=12288 OK. Но `blocks_per_row = K / 256` — для k_slice partial blocks возможно > 80 если кто-то даст эксцентричный override | LOW (audit пройден на текущих моделях) | Use heap fallback если `blocks_per_row > 80` |
| 16 | `cpu_fused_rmsnorm_gemv` `float stack_buf[MAX_STACK_HIDDEN=8192]` | `torch/io/cpu_quant_gemv.h:2299, 2365, 2434` | Стек = 32KB. Defaults stack ~1MB на pthread, ~8MB на main, но c10::ThreadPool создаёт pthreads с default stack — может быть 64KB на embedded/E2K | LOW | Heap fallback при hidden > 8192 ЕСТЬ. ОК |
| 17 | `parallel_for` boundary `y[g*4 + 0..3]` при N % 4 — невозможно | `torch/io/q8_soa_repack.h:111` allocator rejects | `q8_soa4_alloc` возвращает false при `N % 4 != 0`. Callers (`forward_decode_cpu_tp`) **молча**? Проверить. Если callsite не проверяет valid=false → garbage GEMV | MEDIUM | `assert(w->valid)` в начале `q8_soa4_gemv` |
| 18 | `Q8SoA4` move-assign: `_aligned_free` guard корректен | `torch/io/q8_soa_repack.h:89-106` | `#ifdef _MSC_VER` — OK для Windows; на MinGW (`__MINGW64__` без `_MSC_VER`) попадёт в `std::free` который crash'нет on `_aligned_malloc`-allocated memory. PromeTorch собирается NMake+MSVC → ОК. Если кто-то перейдёт на mingw → silent crash | LOW (build matrix MSVC only) | Use `__MINGW32__ \|\| _MSC_VER` или just `posix_memalign` cross-platform |
| 19 | `MmapHandle::at_offset` overflow `static_cast<const char*>(data_) + offset` | `torch/io/gguf_loader.h:374-377` | Check `offset >= size_` есть, НО НЕТ check на `offset + len <= size_` для caller (e.g. `get_tensor_data_ptr` → caller читает `data_bytes()` без bounds check vs mmap size). Malformed GGUF может указать offset = size_ - 1, data_bytes = INT64_MAX → tensor читает за пределы mmap → segfault или information disclosure | HIGH (untrusted GGUF) | Добавить `at_offset(offset, len)` overload который проверяет `offset + len <= size_` |
| 20 | `GGUFTensorInfo::n_elements()` mul без overflow | `torch/io/gguf_loader.h:102-106` (повтор аудита 2026-05-20 #4) | Известный баг. Malformed dims=[2^31, 2^31] → silent wraparound → `at::empty(shape)` с negative size → подается в `dequantize` где `n_elements` используется в loop bound → OOB | HIGH (attacker-controlled GGUF файл) | `__builtin_mul_overflow` per-dim |
| 21 | `read_string` len > 1MB rejected, но `arr.reserve(min(count, 1M))` под integer overflow | `torch/io/gguf_loader.h:786` | `count` = `read_val<uint64_t>(f)` контрольно. `min(count, 1M)` для reserve ОК; **но цикл `for (i=0; i < count; ++i) val.arr.push_back(...)` НЕ ограничен 1M** — count = UINT64_MAX → бесконечный цикл, RAM exhausted, OOM | HIGH (untrusted GGUF) | `if (count > 1'000'000) throw;` перед циклом |
| 22 | `numa_get_B` `g_B_cache[node]` data race | `aten/src/ATen/native/cpu/hot_loops.cpp:201-210` | Глобальный non-atomic write `g_B_cache[node] = numa_alloc_onnode(...)` без mutex. Если 2 GEMM вызова concurrent → double-alloc + leak + free прошлого → use-after-free на одного из caller | HIGH в multi-threaded training (PIR Local SGD), MEDIUM в single-thread inference | Mutex per-node или TLS pool. Также `g_B_cache_size` race |
| 23 | `numa_tiled_sgemm` static `g_numa_tiles[NUMA_POOL_MAX]` shared between sequential calls — OK; concurrent calls (multi-thread caller) — RACE | `aten/src/ATen/native/cpu/hot_loops.cpp:115-127, 231-260` | Если 2 thread вызовут `sgemm` одновременно → оба пишут в `g_numa_tiles[i]`, оба ждут общий barrier → corruption | HIGH в multi-threaded training | Mutex обёртка `g_numa_pool_mutex` вокруг 231-260 |
| 24 | E2K `*(const v1di*)(W + kg*16 + 0)` 8B unaligned read | `torch/io/q8_soa_repack.h:541-544` | W = `sb + 48`, sb = 16B-aligned. `W + kg*16 + 0` всегда 16B-aligned (т.к. 48 ≡ 0 mod 16). `W + kg*16 + 8` — 8B-aligned. v1di = 8B → требует 8B alignment → ОК. Но `A = a_b16 + b*128` — a_b16 alignment? `tp_.soa_act_b16` это `std::vector<uint8_t>` — alignment = 1B! Только удача что allocator выравнивает на 16B. На LCC может быть SIGBUS | HIGH potential — silent corruption или crash | Aligned allocator для `soa_act_b16` (`std::vector<uint8_t, AlignedAllocator<16>>`) |

---

## Дополнительные замечания

### Engine race (#6 аудита 2026-05-20)
**Статус: всё ещё ОТКРЫТ.** В `torch/csrc/autograd/engine.h` дефолтный singleton + `Node::dependency_count_`, `visited_` mutable per-node. Inference hot-path не использует autograd (forward_decode_cpu_tp вызывает только cpu_quant ops), но если PIR training run параллельно с inference в одном процессе — silent grad corruption. Не блокирует hot-path inference, но может блокировать смешанные сценарии.

### MCPClient lifetime race (#20 аудита 2026-05-20)
**Статус: подтверждён.** Комментарий «race потенциально остаётся» из аудита убран из текущего `promeserve/mcp_client.h` (комментарии в районе 994 говорят о thread-safety гарантиях, но `reconnect()` всё ещё делает `clients_.erase(it)` под одним mutex'ом без `shared_ptr` refcount). Не блокирует inference hot-path; релевантно только при concurrent reconnect + dispatch.

### Strict aliasing — общее замечание
По всем hot kernels (q8_soa4_gemv, q8_soa4_gemv_dual, q8_soa4_gemv_triple, q4k_gemv_avx2): raw cast `*(const v2di*)(ptr)` используется ШИРОКО. LCC и GCC обычно толерантны при `-fno-strict-aliasing`, но дефолтный `-O3` без флага — UB. **Рекомендация:** добавить `-fno-strict-aliasing` в CMake для всех hot_loops/io/ TU, или заменить на `std::memcpy` (компилятор оптимизирует в тот же load).

### Top-3 фикс-приоритет
1. **#11** — TP KV-cache OOB write без bounds check (HIGH, легко эксплуатируется в long-context)
2. **#21** — GGUF array `count` cycle без upper-bound (HIGH, untrusted file)
3. **#22 / #23** — NUMA tile globals race (HIGH в multi-threaded training)

### Список файлов с найденными issues
- `C:\Users\USER\Desktop\promethorch\torch\io\q8_soa_repack.h` (#1, #2, #3, #4, #17, #18, #24)
- `C:\Users\USER\Desktop\promethorch\torch\io\cpu_quant_gemv.h` (#6, #7, #8, #9, #15, #16)
- `C:\Users\USER\Desktop\promethorch\torch\io\gguf_model.h` (#10, #11, #12, #13, #14)
- `C:\Users\USER\Desktop\promethorch\torch\io\gguf_loader.h` (#19, #20, #21)
- `C:\Users\USER\Desktop\promethorch\aten\src\ATen\native\cpu\hot_loops.cpp` (#22, #23)
- `C:\Users\USER\Desktop\promethorch\promeserve\mcp_client.h` (lifetime race ref)
- `C:\Users\USER\Desktop\promethorch\torch\csrc\autograd\engine.h` (Engine race ref)
