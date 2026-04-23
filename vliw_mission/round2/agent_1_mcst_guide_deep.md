# Agent 1 — MCST Guide Deep Audit

Source: `vliw_mission/round2/_inputs/mcst_main_guide.txt` (6766 lines, 178 pages, русский).
Target workload: qwen3:4b Q4_K_M decode, Elbrus 8C2, 1-proc 5.3 tok/s / TP-4 6.1 tok/s.
Pipeline is weight-streaming GEMV-dominated → **memory-bandwidth bound**.

Currently applied (from MISSION.md): `-O3 -ffast -faligned -fprefetch -fcache-opt -mtune=elbrus-8c2 -frestrict-all -fswp-maxopers=800` + `#pragma loop count(N) + #pragma ivdep` + `int64_t` indices in inner loops.

Everything below is extracted from full read of the guide (chapters 1–10 + index).

---

## 1. PRAGMA DIRECTIVES — full inventory

The guide documents ONLY TWO pragmas by name, and points to `/opt/mcst/doc/pragma.html` for the rest.

**Quote, line 1753–1755 (ch. 3.1.3):**
> «3.1.3 Прагмы. Информация о поддержанных прагмах в документации компилятора: /opt/mcst/doc/pragma.html»

**Quote, lines 3089–3107 (ch. 6.3.5 "Управление конвейеризацией"):**
> «#pragma loop count(N) — подсказка компилятору о среднем количестве итераций цикла. … Эту же подсказку можно использовать для того, чтобы отключить конвейеризацию с негативным эффектом для циклов с малым количеством итераций.»
> «#pragma ivdep — разрыв зависимостей между операциями чтения и записи, расположенными на разных итерациях цикла; позволяет уменьшить длину последней стадии и увеличить эффективность конвейеризации.»

**OpenMP pragmas (ch. 3.1.4.2, p.36, lines 1766–1776):**
OpenMP 3.1 is supported, `#pragma omp for` / `#pragma omp parallel`. Restrictions:
> «Nested параллелизм не поддержан … Не поддержан clause collapse … Для C/C++ после директивы #pragma omp всегда должен следовать statement … #pragma omp for не поддержана для итераторов C++ … clause'ы if и num_threads своими параметрами могут иметь только константы и переменные целого типа, выражения не допускаются.»

**Findings about pragma #unroll / flatten / noinline / simd etc.:**
The guide does NOT mention `#pragma unroll`, `#pragma GCC ivdep`, `#pragma omp simd`, `#pragma flatten`, `#pragma noinline`, `#pragma forceinline` anywhere. It points to the external doc. The only compile-time unroll control documented is `-ffast-math` triggering it automatically (ch. 7.4.2, line 4272) and manual source-level unroll (ch. 7.4.2, lines 4281–4308).

**Applicability to qwen3 Q4_K_M GEMV:**
- `#pragma loop count(N)` ALREADY APPLIED — no further upside.
- `#pragma ivdep` ALREADY APPLIED — no further upside.
- `__attribute__((always_inline))` on the Q4_K unpack helpers is mentioned at line 2575: «__attribute__((always_inline))». **NOT listed as applied in MISSION.md.** Worth auditing `cpu_quant_gemv.h` for missing always_inline on hot inner helpers (e.g. `q4_k_dequantize_block`, scale unpackers). Estimated gain: 2–8% if helpers are currently out-of-line.

**Code suggestion (NOT a hardware change, NOT confirmed pragma):**
```c
// torch/io/cpu_quant_gemv.h — add to all Q4_K helpers
static inline __attribute__((always_inline)) void unpack_q4k_scales(...) { ... }
```

**Risk:** The guide warns `__attribute__((always_inline))` combined with deep recursion will blow up code size and hurt I-cache (line 2554: «снижение hit rate для кэша команд»). Apply only to small leaf helpers.

---

## 2. __builtin_* intrinsics for memory hints

### 2.1 `__builtin_prefetch` — fully documented

**Quote, lines 3842–3866 (ch. 6.6):**
> «Встроенная функция __builtin_prefetch() позволяет заблаговременно подкачать данные в кэш-память. Подкачивается целая порция кэш-строк — 64 байта.»
> «Синтаксис: __builtin_prefetch(addr, rw, locality)»
> «rw (0..1): 0 (умолчание) — подкачка для чтения; 1 — подкачка записи»
> «locality (0..3) — уровень темпоральной локальности (ожидание переиспользования). 3 (умолчание) — сохранение во всех кэшах. 0 — можно выкидывать из кэша сразу после использования.»

**MAS bits for prefetch level control, line 3861–3866:**
> «Уровень кэш-памяти регулируется параметром mas:
> 0x0 (умолчание) — заводить везде
> 0x20 — не заводить в L1$
> 0x40 — не заводить в L1$ и L2$
> 0x60 — не заводить в кэшах (для подкачки не используется)»

**Multi-hop (n-linear) prefetch template, lines 3887–3920:**
For `c[b[a[i]]]` pattern the guide recommends 3 prefetches at offsets `i+3*dist`, `i+2*dist`, `i+dist`.

### 2.2 `__builtin_expect` — for branch hints without profile

**Quote, line 3465 (ch. 6.4.5):** «без профиля помогают подсказки __builtin_expect().»

### 2.3 NO non-temporal store / streaming store intrinsic documented

The guide does NOT document `__builtin_nontemporal_store`, `_mm_stream`, `st_nt`, or equivalent. The only way to control write-cache behaviour in the guide is via `locality=0` on a prefetch (read side). For writes, the only hinting is `rw=1` (line 3848–3850): «1 — подкачка записи, означает простановку признака эксклюзивности подкачанной кэш-строке (зарезервировано для версий системы команд >5).» **Reserved for ISA v>5 — Elbrus-8C2 is v4, so not applicable.**

### 2.4 Applicability to qwen3 GEMV

**Q4_K_M block layout:** each K=256 superblock is 144 B (6 bytes scales/mins + 64 B q4s + etc.). Two blocks = 288 B (~4.5 cache lines). For one GEMV pass over a [4096, 4096] weight matrix = **576 MB of weight data**, far exceeds 16 MB L3. → weights are streamed from DDR every token.

**Current status in `torch/io/cpu_quant_gemv.h`:** MISSION.md does NOT list explicit `__builtin_prefetch` calls. Only `-fprefetch` compiler auto-prefetch. Per ch. 8.3.3 line 4513–4514:
> «механизм apb не применяется из-за нерегулярного изменения адреса; рекомендуется использовать в качестве цикловых счетчиков, определяющих адрес чтения, переменные типа long (не unsigned), не производить инкрементов счетчиков под условиями»

**AND line 4536–4537:**
> «блокировки от операций чтения из-за кэш-промахов (BUB_E0): рекомендуется попробовать опции -fcache-opt, -flist-prefetch, включающие режим предварительной подкачки данных в кэш»

**Concrete gap:** `-flist-prefetch` is NOT in the current Elbrus flag set. Current set has `-fcache-opt -fprefetch` but NOT `-flist-prefetch`. See section 3 below.

**Code suggestion for `torch/io/cpu_quant_gemv.h` inner j-loop:**
```c
// dist tuned empirically; for Q4_K superblocks of 144 B and ~10–20 tact iteration
// length, dist = ~200 blocks ahead = 200*144 B = ~28 KB ahead (fits in L2)
#define PREF_DIST_BLOCKS 200
for (int j = 0; j < n_blocks; j++) {
    if (j + PREF_DIST_BLOCKS < n_blocks) {
        __builtin_prefetch(&w_q4k[j + PREF_DIST_BLOCKS].qs, 0, 2); // L2
        __builtin_prefetch(&w_q4k[j + PREF_DIST_BLOCKS].scales, 0, 3); // L1 (hot)
    }
    // ... existing GEMV body using w_q4k[j]
}
```

**Estimated speedup uncertainty:** 5–15% if `-fprefetch` auto-prefetch currently fails on Q4_K's irregular superblock stride (144 B not aligned to 64 B line boundary at block start). Lower bound 0% if compiler already prefetches.

---

## 3. Compiler options NOT currently applied

Current LCC flags (from MISSION.md line 26):
`-O3 -ffast -faligned -fprefetch -fcache-opt -mtune=elbrus-8c2 -frestrict-all -fswp-maxopers=800`

**Options documented in guide but NOT applied:**

### 3.1 `-O4` — aggressive optimizations (ch. 1.3.1, lines 353–356)
> «Включает все предыдущие и дополнительные агрессивные оптимизации. Стоит пробовать экспериментально. Может приводить как к повышению производительности, так в некоторых случаях и к деградации производительности.»

**Applicability:** HIGH — experimentally test. Risk: regressions on specific kernels.
**Change:** `-O3` → `-O4`.
**Expected:** 0–15% (bimodal — will either help or hurt). Guide explicitly labels experimental.

### 3.2 `-fwhole` — whole-program (ch. 1.3.2 lines 383–398, ch. 6.2 line 2562, ch. 8.3.2 line 4504)
> «Режим «вся программа» позволяет выполнять межпроцедурные оптимизации. В этом режиме анализируемый контекст не ограничивается единственной функцией.»
> Ch. 8.3.2: «инлайн-подстановка: собирать в режиме -fwhole»

**Applicability to inference:** HIGH. `forward_decode_cpu` calls multiple translation units (`attention`, `ffn`, `rmsnorm`, `gemv`). If these are in separate `.o`, cross-module inlining is blocked without `-fwhole`. The guide at line 2559–2562:
> «Первое — вызываемая процедура должна находиться в том же самом модуле, где и вызвавшая ее процедура. Для решения этой проблемы предусмотрена возможность сборки проекта в режиме межмодульной оптимизации. Для этого на вход компилятору следует подать опцию -fwhole.»

**Change in `CMakeLists.txt:130-170`:**
```cmake
if(ELBRUS)
    add_compile_options(-fwhole)   # or -fPIE -fwhole-shared if PIE needed
    add_link_options(-fwhole)      # must be on BOTH compile and link
endif()
```
**Caveats (line 388):** `-fwhole` incompatible with `.so` libraries and PIC. If the Elbrus build produces shared libs, use `-fPIC -fwhole-shared` instead. MISSION.md doesn't say, but the training build likely is a monolithic executable.

**Expected gain:** 5–20% for inference. Biggest win if hot path spans multiple TUs. This is the single most impactful missing flag.

### 3.3 `-fipo-invup` — hoist loads out of procedure calls (ch. 8.3.2 line 4508)
> «включение режима выноса чтений из процедур -fipo-invup»

**Applicability:** Only useful if hot path has calls inside a loop where reads are invariant across calls. For our GEMV inner loop there are no calls (already flattened). Likely modest — 0–2%.

### 3.4 `-flist-prefetch` — list-traversal prefetch for pointer-chase (ch. 8.3.3 line 4537)
> «блокировки от операций чтения из-за кэш-промахов (BUB_E0): рекомендуется попробовать опции -fcache-opt, -flist-prefetch, включающие режим предварительной подкачки данных в кэш»

**Applicability:** MEDIUM. Q4_K block traversal is contiguous, not pointer-chase, so `-flist-prefetch` may not help the weights. But KV-cache access (if stored as per-token ptr arrays) could benefit. Worth adding — it's free if unused.

**Change:** add `-flist-prefetch` to flags.
**Expected:** 0–5%.

### 3.5 `-fprofile-generate` / `-fprofile-use` — two-phase PGO (ch. 1.3.2 lines 367–381, ch. 6.4.5 line 3464, ch. 8.3.2 line 4505)
> «Профиль от двухфазной компиляции помогает весьма сильно» (line 3464)
> Ch. 8.3.6 (line 4557) for switch-heavy code: «эффективно при наличии адекватного профиля».

**Applicability:** HIGH. Inference is a deterministic hot-loop workload — PGO should deliver reliably. The guide emphasises this is the #1 recommendation for code slicing / merging control flow.

**Two-phase build process:**
```bash
# Phase 1 — instrumented build
lcc $FLAGS -fprofile-generate=/tmp/gguf.prof -O3 ... -o run_inference_prof
./run_inference_prof  # run actual qwen3:4b prompt to collect profile

# Phase 2 — optimized build with profile
lcc $FLAGS -fprofile-use=/tmp/gguf.prof -O3 ... -o run_inference
```

**Expected gain:** 5–25% on tok/s. Sometimes dramatic for large procedures.
**Risk:** Profile mismatch (quote line 377–379): «Если в дальнейшем реальное исполнение программы существенно отличается от того варианта, на котором получали профиль, то производительность реального исполнения может значительно ухудшиться.»
**Mitigation:** Profile on a representative qwen3 prompt.

### 3.6 `-fno-loop-rtmd` — disable runtime memory disambiguation (ch. 6.5.4.2 line 3762)
Only relevant if profile shows many failing RTMD checks. Not applicable to our GEMV (no aliasing risk). Skip.

### 3.7 `-fno-dam` — disable data-speculation DAM table (ch. 6.5.4.2 line 3797)
Same reasoning. Skip unless profiling shows DAM compensation code is hot.

### 3.8 `-fforce-swp` — force software pipelining (ch. 6.3.5 line 3105)
> «отключает оценки качества конвейеризации в пользу безусловного применения. Использовать можно для экспериментов.»

**Applicability:** LOW — `-fswp-maxopers=800` is already generous. Adding `-fforce-swp` would only help if the pipeliner is voluntarily giving up on our GEMV. Unlikely with 800 ops budget. Try only as experiment.

### 3.9 `-finline-level=N` / `-finline-scale=N` (ch. 6.2 lines 2580–2581)
> «-finline-level=1.0 — коэффициент k [0.1–20.0]»
> «-finline-scale=1.0 — коэффициент увеличения основных ресурсных ограничений [0.1–5.0]»

**Applicability:** MEDIUM. After adding `-fwhole`, consider bumping `-finline-level=2.0` or `-finline-scale=2.0` to force aggressive cross-module inlining of the hot GEMV path.
**Expected:** +1–3% on top of `-fwhole`.

### 3.10 `-fmax-iter-for-ovlpeel=0` (ch. 2.2.5 line 1084, ch. 6.3)
Used in examples for cleaner assembly. NOT a production tuning flag. Skip.

### 3.11 `-fassociative-math` (ch. 6.3.4 line 3069)
Already included in `-ffast`. No change.

---

## 4. Memory-bound workload recipes (from the guide)

### 4.1 The guide flatly acknowledges bandwidth ceiling

**Quote, ch. 8.3.3, line 4515:** «блокировки из-за превышения пропускной способности устройства памяти.»

**Quote, ch. 8.3.3, line 4538–4542:**
> «блокировки по темпу работы памяти (BUB_E2): рекомендуется проверить темп обработки данных — сколько тактов работает цикл, сколько в нем операций чтения и записи, каков размер этих операций, какова локальность данных, какие данные могут быть найдены в кэше. Если темп существенно ниже ожидаемого, возможно, проблема в неравномерности использования ресурсов кэша второго уровня.»

**Tooling:** `dprof -m BUB_E2` to quantify memory-stall tacts. If BUB_E2 is > 30% of TICKS, we're memory-bound (which we are). Guide offers no magic to raise DDR bandwidth — only reduce bytes transferred.

### 4.2 Recipes for bandwidth reduction that the guide recommends

**Ch. 8.1, lines 4385–4396 (structures of arrays, NOT array of structures):**
> «При регулярном считывании/записи элементов массива могут достигаться теоретически максимальные показатели темпа доступа к памяти — 32 байта в такт при попадании в L2$, 32 байта в 3 такта при гарантированном отсутствии в L2$ … при регулярном обращении с большим шагом (>64b) APB все еще применим, но темп существенно падает (до 64 раз при побайтовой обработке)»
> «весьма полезен (в ущерб наглядности) переход от массивов структур к набору массивов, хранящих отдельные поля»

**Applicability to Q4_K_M:**
The GGUF Q4_K block layout is an **array-of-structs**: `{float d, float dmin, uint8 scales[12], uint8 qs[128]} per K=256`. For GEMV we touch ALL fields each block, so SoA would NOT help raw bandwidth. But if the q4s alone were separated, for large batches the scales could be cached separately. **For B=1 decode (our case), this is neutral.** Skip.

### 4.3 The "32 B/tact in L2, 32 B / 3 tact from memory" ceiling

**Line 4387:** peak per-core memory throughput from DDR is **32 B / 3 tact**. At 1.5 GHz that's `32 / 3 * 1.5e9 = 16 GB/s per core`. With 4 cores running in-lockstep-ish, theoretical cap is 64 GB/s — but MCU-16CB (not 8C2!) has 4 controllers. Elbrus-8C2 has **4 memory controllers at DDR3-1600** (ch. 4.2 tab 4.2, line 1855): «Количество контроллеров памяти: 4, Организация оперативной памяти DDR3-1600 ECC» — aggregate peak ~51.2 GB/s.

**Current measurement:** 5.3 tok/s × 2.5 GB/tok (Q4_K weight pass per token) ≈ 13.25 GB/s. **That's 26% of aggregate DDR peak** — consistent with 1 of 4 memory channels active.

**Implication:** To reach 20 tok/s we need to either:
1. Engage all 4 DDR channels with non-redundant streams → requires NUMA weight replication per channel-domain OR cross-channel block striping. **HARDWARE-TOPOLOGY task, not pragma-level.**
2. Reduce bytes-per-token (compute speculative decode acceptance rate > 0, batched decode, KV-cache tricks).

**The guide does NOT offer a software recipe to increase DDR effective bandwidth.** That's a physical ceiling.

---

## 5. APB mechanism — buffer size, parallel streams, layout

### 5.1 APB buffer sizing (hardware ceiling)

**Quote, ch. 10.11.6, lines 6554–6573 (FAPB operation):**
> «asz — спецификатор размера области назначения в APB; размер области определяется как area_size = (64 байта)*(2**asz). Диапазон корректных значений asz ограничивается размером APB и для данной реализации включает числа от 0 до 5»

**So: per-area max size = 64 B × 2^5 = 2 KB per APB area.**

> «abs — адрес базы области назначения в APB (в терминах 64 байтов). База области должна быть выровнена до размера области; для последовательности операций асинхронной программы области APB должны назначаться также последовательно, по возрастающим адресам; области, назначенные для разных операций, не должны перекрываться»

**Hardware limit:** The APB register file is partitioned into non-overlapping areas, each ≤ 2 KB. The guide does NOT state total APB size in bytes, only area count limits. Ch. 4.4 line 1945: **"4 устройства для команд асинхронного чтения данных по регулярным адресам в цикле (APB)"** and line 1999: **"В одной широкой команде можно исполнить до 4 операций чтения из буфера APB."**

**Answer to question 5a (increase APB buffer size):** NO software knob. APB size is fixed in hardware. You can only increase `asz` per-area up to 5 (= 2 KB) and allocate multiple areas.

**Answer to question 5b (parallel APB streams):** YES — up to **4 parallel APB streams** (the 4 MOVA slots in one wide instruction). This is already fully consumed by the compiler when multiple arrays are read in a hot loop (see ch. 7.4.2 line 4352 where matmul uses 14 MOVA slots via unroll — spilling beyond 4 via multiple wide instructions).

**Applicability:** For GEMV we read 2 arrays per block (scales, qs) + 1 activation vector. `-fprefetch -ffast -faligned` with aligned stride = 1 block (144 B) ought to trigger APB for all 3. The guide (line 3941) says APB requires:
> «при наличии аппаратного счетчика цикла %lsr»
> «при отсутствии зависимостей с записями между итерациями»
> «при достаточно большом числе итераций»
> «при инкременте < 32b»

**Potential gap:** If block size 144 B is NOT a power of 2 and not a simple multiple of 32 B (it's 144 = 2.25 * 64), the alignment check may reject APB. This is the 144 vs 128 (2×64) conflict.

**Change for `cpu_quant_gemv.h`:** Pad Q4_K blocks to 160 B or 192 B (multiples of 64) during weight loading, so inner loop stride is cache-line-aligned. Memory cost: 160/144 = +11% weight bytes. Speed cost: +11% more bytes transferred. **Net: likely WORSE.** Skip.

**Alternative:** Split the inner loop into two phases — one over `scales` (6 B × n_blocks = contig 6-byte stride, APB-friendly only after transpose), one over `qs` (128 B / block = power-of-2 friendly). Two passes. Extra reads of activation → cache-hit. Worth prototyping but the guide doesn't suggest this idiom.

### 5.2 APB layout requirements

**Quote lines 6565–6567:**
> «база области должна быть выровнена до размера области; для последовательности операций асинхронной программы области APB должны назначаться также последовательно, по возрастающим адресам; области, назначенные для разных операций, не должны перекрываться»

This is about APB register allocation, NOT source data layout. Compiler-managed.

**Quote line 3955–3956:**
> «-ffast, -faligned — включение режима оптимизации в предположении выровненности обращений к памяти; необходимы для возможности применения apb в версиях системы команд V1-V5»

Both applied. OK.

### 5.3 Weight layout recommendations

**Ch. 8.1, line 4397–4402:**
> «Для повышения локальности нужно следить за тем, чтобы внутренняя размерность массивов (первая) индексировалась индуктивной переменной самого внутреннего цикла:
> `for i: for j: for k: A(k,j,i) // хорошо`»

**Applicability to Q4_K weight layout:** Q4_K is stored row-major `[out_features, in_features]`. GEMV inner loop indexes along `in_features` (K dim) for each output row. In C this means `W[i][k]` with `k` inner — matches recommendation. ALREADY OPTIMAL for row-major.

But Ollama's layout (which we're reading) stores `[out, in_blocks]` with blocks-of-256-along-K. Inner loop strides along block dim (good), within block reads 128 bytes sequentially (good). ALREADY OPTIMAL.

---

## 6. APB layout for weights

Covered above (§5.3). No additional findings.

**One overlooked detail (lines 4392–4396):**
> «одномерные массивы структур. При регулярной обработке применим APB, однако, следует следить за тем, чтобы набор одновременно читаемых/записываемых полей в горячих участках был как можно более компактным; весьма полезен (в ущерб наглядности) переход от массивов структур к набору массивов, хранящих отдельные поля»

**The `block_q4_K` struct contains `d`, `dmin`, `scales[12]`, `qs[128]` (total 144 B).** In our GEMV inner loop we use ALL these fields once per block. **The fields are NOT used at different times** — they're all touched on the same iteration. So SoA transform wouldn't reduce bandwidth for B=1 decode. It WOULD help if we batched K=4 tokens and fetched scales once for 4 activations, but current spec decode has 0% acceptance anyway.

**Recommendation:** Defer SoA until batched decode is real.

---

## 7. Software pipelining — "wide pipeline" modes and -fswp-* params

**Ch. 6.3.5 (lines 3088–3107) documents exactly two SWP options:**

> «-fswp-maxopers=N — максимальное количество операций для цикла, чтобы его можно было рассматривать для конвейера. Конвейеризация полностью отключается при значении swp-maxopers = 0. Для значений порядка 1000 нужно бояться сильного роста времени компиляции проекта. Значение по умолчанию — 300.»

Current setting: `-fswp-maxopers=800` — already 2.67× default. Compile time already paid.

> «-fforce-swp — отключает оценки качества конвейеризации в пользу безусловного применения. Использовать можно для экспериментов.»

NOT applied. Worth experimenting but risky — forces pipelining even when cost-model says don't. Could regress.

**"Wide pipeline mode":** The guide does NOT document any `-fswp-wide` or similar. The concept is implicit in the unrolling ratio (ch. 7.4.2 line 4351 uses unroll 8×6). **The only way to get "wider" pipelining is source-level unrolling.**

**Recommendation (cpu_quant_gemv.h):**
Manually unroll the j-loop by a factor of 4 or 8 (matching the 4-way APB slot capacity). MISSION.md says batched K=4 kernel exists (1.54× on standalone). Push further — K=8 — and rely on `-fswp-maxopers=800` to pipeline. The guide example ch. 7.4.2 line 4351 achieves 12 flops/cycle with unroll 8×6 ≈ 4× what plain SWP gets.

**Estimated gain:** 5–20% if current kernel is unrolled < 4×. Diminishing returns past 8 due to register pressure (256 regs total, 224 on stack).

### 7.1 Recurrence balancing

**Ch. 6.3.4 (lines 3015–3069)** — unroll + reduction splitting for associative recurrences. Needs `-ffast-math` or `-fassociative-math`. Already enabled via `-ffast`.

**Applicability to GEMV:** The dot-product accumulator is the one recurrence: `acc += w * x`. With `-ffast` the compiler splits `acc` into 4 or 8 partial accumulators, balancing the 4-tact fadd recurrence. Should already be active. Verify with `ldis -P` on the compiled binary.

---

## 8. Other chapters — profiling, memory model, assembler, data segments

### 8.1 Chapter 5 (profiling) — already covered in §4

Key takeaway: **dprof with events (BUB_E2, BUB_E0, L1 miss, APB miss, L2 miss, IB_NO_COMMAND, TICKS, EXEC)** is the correct diagnostic tool. MISSION.md mentions iteration hit count but no event-breakdown. Before further tuning, run:
```bash
dprof -m TICKS,EXEC,BUB_E0,BUB_E2,IB_NO_COMMAND,L2_QUERY,L2_HIT ./run_inference
```
If `BUB_E2 / TICKS > 30%` → memory-bandwidth bound (expected). If `BUB_E0 / TICKS > 20%` → L1 miss bound; more aggressive prefetch helps. If `IB_NO_COMMAND > 10%` → code I-cache thrashing; **-fwhole + PGO** helps.

### 8.2 Chapter 9 (memory model) — layout alignment

**Ch. 9.1.4.1 lines 4688–4698 (alignment for globals):**
> «Переменные размера большего, чем 8 байт, должны быть выровнены на 16 байт.»

Q4_K blocks at 144 B → must be 16-byte aligned. mmap'd GGUF typically IS page-aligned. Weights pointer should be 64-byte aligned for best APB. **Verify `gguf_model.h` that weights are 64-byte aligned on Elbrus.** If not, alignment peel at loop start costs cycles.

**Change suggestion:** explicit `__attribute__((aligned(64)))` on the mmap buffer or use `aligned_alloc(64, ...)` when loading. Should already be free from mmap but worth verifying.

### 8.3 Chapter 2.2.5 (assembly for loops, lines 940–1111)

**Line 1107–1108 shows the target:** «Тело цикла компактно упаковано в одну широкую команду, и исполняется с темпом 1 итерация за 1 такт.»

**1 iteration / 1 tact is the VLIW ceiling.** For a 4-FMA-per-tact unroll (since Elbrus-8C has 4 ALUs doing fmul), that's 4 FMA/tact = 8 flops/tact * 1.5 GHz = 12 Gflops/core FP32. Aggregate 8 cores = 96 GFlops FP32. Q4_K is integer → different budget.

### 8.4 Chapter 6.1 (code scheduling, operation latencies)

**Line 2443–2466 — latency table:**
- Int, Bitwise: 1 tact
- Int_combined (shl_add): 2 tacts
- Fp (fadd, fmul, muls): **4 tacts**
- Fdivs: 11 tacts; Fdivd: 14 tacts
- Ld L1: **3 tacts**
- Ld L2: 11 tacts
- Ld L3: **40 tacts**
- Ld mem: **~100 tacts**
- fp→int: +2 tacts penalty
- int→fp: +1 tact penalty

**Implication for Q4_K GEMV:**
- Dequantize (int8 shift/mask + fp32 multiply by scale) = int→fp = +1 tact penalty per multiply
- The critical path is multiply-accumulate with 4-tact fmul recurrence
- Must unroll by ≥ 4 to keep fp pipeline full

---

## 9. Summary — priority-ranked changes

| # | Change | Location | Expected | Risk |
|---|--------|----------|----------|------|
| 1 | Add `-fwhole` + `-fPIE -fwhole-shared` (as appropriate) to Elbrus flags | `CMakeLists.txt:130-170` | **+5–20%** | LOW — compile-time increase |
| 2 | Two-phase PGO: `-fprofile-generate` → run → `-fprofile-use` | build script | **+5–25%** | MED — profile-representativeness |
| 3 | `__builtin_prefetch(w + dist, 0, 2)` in GEMV inner loop | `torch/io/cpu_quant_gemv.h` | **+5–15%** | LOW |
| 4 | Try `-O4` experimentally | `CMakeLists.txt` | +0–15% or regression | MED |
| 5 | Add `__attribute__((always_inline))` on Q4_K unpack helpers | `cpu_quant_gemv.h` | +2–8% | LOW |
| 6 | Add `-flist-prefetch` (for KV cache if applicable) | CMake flags | +0–5% | LOW |
| 7 | Run `dprof -m TICKS,EXEC,BUB_E2,BUB_E0,L2_HIT` first — diagnose bottleneck | — | measurement | — |
| 8 | Manually unroll GEMV j-loop to 8 (matches 4-APB-slot hardware, pairs with `-fswp-maxopers=800`) | `cpu_quant_gemv.h` | +5–20% | MED — reg pressure |
| 9 | `-finline-level=2.0` on top of `-fwhole` | CMake | +1–3% | LOW |
| 10 | Verify 64-B alignment on mmap'd GGUF weights | `gguf_model.h` | 0–2% | LOW |

**Explicitly NOT achievable without hardware changes:**
- Increasing APB buffer size (fixed at 4 streams × 2 KB each, hardware ceiling)
- Breaking the 32 B / 3 tact DDR3-1600 per-controller ceiling
- Using ISA v>5 `prefetch-with-exclusive-cache-line` hint for writes (Elbrus-8C2 is v4)

**Not listed as needed for GEMV:**
- `#pragma omp simd` — not documented in this guide (OpenMP 3.1 doesn't have simd clause)
- `#pragma unroll` — not documented; use source unrolling + `-fswp-maxopers`
- Non-temporal stores — not documented; weights read-only so irrelevant
- `-fno-apb`, `-fno-dam`, `-fno-loop-rtmd` — all "disable" flags; irrelevant unless profile shows these mechanisms are hurting
- `-fforce-swp` — diagnostic-only

---

## 10. Cross-refs to other inputs

Several input docs confirm findings:
- `loop_vectorization.txt` — `-faligned` + aligned addresses mandatory for vectorization (line 106 main guide echo)
- `eml_acceleration.txt` — confirms EML already exploits packed-integer ops for small sets; cannot beat EML for dense primitives but Q4_K is not in EML
- `hpc_architecture.txt` line 220: recommends for OpenCV: `-O4 -ffast -ffast-math -faligned -D_LITTLE_ENDIAN -fPIC -fno-ipo-region -fno-vir` — NEW flags `-fno-ipo-region` and `-fno-vir` NOT documented in main guide; they may reduce IPO overhead in large codebases. Worth experimental try.

---

## 11. Explicit quote-bank (page-numbered)

| Finding | Line | Quote |
|---|---|---|
| `-fwhole` exists | 383 | «опция компиляции в режиме «вся программа»» |
| `-fwhole` for inlining | 2562 | «подать опцию -fwhole» |
| PGO boosts ILP & merge | 3464 | «профиль от двухфазной компиляции помогает весьма сильно» |
| `__builtin_prefetch` API | 3845 | «__builtin_prefetch(addr, rw, locality)» |
| APB 4 streams | 1999 | «В одной широкой команде можно исполнить до 4 операций чтения из буфера APB» |
| APB area size ≤ 2 KB | 6560 | «area_size = (64 байта)*(2**asz)» with asz 0..5 |
| Ld mem = 100+ tacts | 2457 | «Ld mem ~100» |
| DDR peak per-core | 4387 | «32 байта в такт при попадании в L2$, 32 байта в 3 такта при гарантированном отсутствии» |
| Memory bandwidth ceiling acknowledged | 4515 | «блокировки из-за превышения пропускной способности устройства памяти» |
| BUB_E2 diagnostic | 2244 | «BUB_E2 — блокировки конвейера на стадии E2» |
| `-flist-prefetch` | 4537 | «-fcache-opt, -flist-prefetch, включающие режим предварительной подкачки» |
| `-finline-level` | 2580 | «-finline-level=1.0 – коэффициент k [0.1-20.0]» |
| `-fipo-invup` | 4508 | «режима выноса чтений из процедур -fipo-invup» |
| `#pragma loop count / ivdep` — documented 2 only | 3090, 3096 | (quoted above) |
| OpenMP 3.1, no collapse, no C++-iter for | 1765 | «Не поддержан clause collapse» |

---

**End of Agent 1 report.**
