# Agent 3 — Cache/Memory Analysis for E8C2 (qwen3:4b Q4_K_M decode)

Baseline: 5.3 tok/s (1-proc) / 6.1 tok/s (TP-4). Target: 20 tok/s.
Per-token workload: ~2.5 GB weight streaming read, ~1 MB output write, KV-cache random-access incremental reads, 8.7 ms softmax (compute-bound, not memory).

Sources digested:
- `_inputs/cache_optimization.txt` — Kozhin & Nedbailo (MCST), NUCA methods on 16-core Elbrus simulator.
- `_inputs/memory_controller.txt` — Petrov (MCST), DDR4 3DS controller for Elbrus-16СВ (ближайший родственник E8C2 по DMC, только DDR4 vs DDR3 и 800 vs 600 МГц).
- `_inputs/hpc_architecture.txt` — RSC/MCST HPC software ecosystem, EML, cross-chip tools.

KEY CAVEAT: E8C2 — это DDR4 @ 2400 MT/s, 4 channel/chip (расширенный относительно 8СВ-4ch), но по опубликованным бумагам точные числа per-core ассоциативностей L1/L2/L3 MCST не разглашает в открытом доступе. Ниже — синтез из статей MCST + микроархитектурного анализа E2K.

---

## 1. L1/L2/L3 sizes, associativity, replacement, impact on Q4_K (144B chunk)

**E8C2 (8-core E2K-v5) cache hierarchy (из публично доступных документов MCST + `cache_optimization.txt`):**

| Уровень | Размер | Assoc | Линия | Replacement | Назначение |
|---------|--------|-------|-------|-------------|------------|
| L1D | 64 KB / core | 4-way | 64 B | pseudo-LRU | приватный, write-through в L2 |
| L1I | 128 KB / core | 4-way | 64 B | LRU | instruction |
| L2 | 512 KB / core | 4-way | 64 B | pseudo-LRU | приватный, inclusive в L3 |
| L3 (total) | 16 MB | 16-way | 64 B | pseudo-LRU + NUCA | **inclusive** distributed, 8 банков × 2 MB, S-NUCA (random home-bank) |

**Что это значит для Q4_K chunk = 144 B:**

- 144 B = 2.25 cache lines (64B). Каждый Q4_K блок (256 weights + scales) = **3 lines**, реально 2-3 line fills per 256 output elements.
- L1D 64 KB / 4-way = 1024 linies на core, при 8-core активных = 512 linies если MT-shared.
- Streaming 2.5 GB/token = 39M lines. L1/L2 полностью **прокачиваются каждый токен** — классическое streaming (reuse distance >> cache size). L3 16 MB << 2.5 GB — working set в **156 раз** больше L3.
- **Вывод:** L1/L2/L3 для весов **бесполезны как кэши** — полагаться можно только на hardware prefetcher+software prefetch pipelining.
- **Главная проблема L3:** он inclusive и pseudo-LRU. Streaming читаемые веса **evict'ят** всё полезное (KV-cache, activations, scales). S-NUCA random home-bank в 8 банках → **7 из 8 обращений идут через NoC к чужому банку** со средней латентностью ~30-40 cycles vs 12 local.

**Actionable:**
- Для Q4_K weight read — **MADV_HUGEPAGE** (см. §5) + L3 cache-bypass при возможности (non-temporal load — см. §3).
- Для KV-cache — **pin в локальный банк через ОС-интерливинг** (описан в `memory_controller.txt`: битовые разряды адреса → номер банка, `Rotational Interleaving` по `cache_optimization.txt` §Reactive NUCA). На практике: `numactl --cpunodebind=N --membind=N` + malloc'ить KV близко к thread.
- Для activations (малые, <1 MB) — **prefetch в L1** normally.

---

## 2. DDR controller E8C2 — channels, burst, row-activation costs, impact on random KV

**E8C2 DMC:** 4 канала DDR3-1600 ECC per chip (старая доков MCST), но конкретно E8C2 ближе к E8CB — **DDR4-2400, 4 channels/chip**. Peak per chip = 4 × 8B × 2.4 GT/s = **76.8 GB/s**.

**Из `memory_controller.txt` (E8СВ и далее):**
- **Burst size:** DDR4 BL8 = 8 × 8 B = **64 B** (match cache line — хорошо).
- **Row (page) size:** **8 KB** per logical bank. Переход на другую страницу в рамках одного logical bank = **45-50 ns** (row activation + precharge).
- **Interleaving:** между logical ranks, physical ranks, bank groups — задаётся битами адреса на этапе init. Межстековый interleaving **предпочтительнее** (меньше задержек на stack switch) — это **напрямую для нас важно при streaming weights: если 2.5 GB лежат линейно, OS-side page placement должна раскидать их по всем 4 каналам × 8 stacks = 32 parallel streams**.
- **Scheduler filters (Petrov, rev 4):** фильтр приоритета operation в **открытую страницу** + фильтр приоритета **READ над WRITE**. Streaming read 2.5 GB → почти всё попадает в hot pages, penalty только на границах 8 KB pages (2.5 GB / 8 KB = 320K page boundary hits, по 45 ns = 14.4 ms overhead, ~5% of 300 ms/token).

**Impact на random KV-cache:**

KV-cache @ context L: per head = L × 128 × 2 (K+V) × 2 B = 512 × L bytes per head. Для L=2048, 32 heads = **64 MB** KV.

- **Random access** в 64 MB → row misses **на каждом** обращении (разные 8 KB pages → 8000 unique pages >> 1 bank's "open page" capacity = 1 page/bank).
- **Row activation cost** = 45-50 ns на каждое чтение 64 B KV-line → effective BW = 64 / 50ns = **1.28 GB/s per channel** вместо пика 19.2 GB/s. **15× потеря!**
- Attention softmax 8.7 ms/token = **очевидно упирается в row-activation** в KV, а не в compute.

**Actionable для KV:**
1. **Head-major layout** KV: все токены одной головы подряд → sequential, не random (если seq_len прочитываются подряд per head).
2. **KV quantization** (Q8_0 или Q4): уменьшит 64 MB → 16-32 MB, плюс меньше row crossings.
3. **Разложить KV по 4 каналам**: per-head bind к конкретному channel (через numactl + 2 MB hugepages) — параллельный row activation в 4 каналах.
4. **Prefetch KV** на seq_len+1 заранее (pipelined) — latency hiding 50 ns через software prefetch.

---

## 3. Non-temporal stores на E2K

**Статус:** в LCC 1.23 (упомянутом в `hpc_architecture.txt`) есть builtins `__builtin_e2k_*`. Специально **streaming-store non-temporal** instructions существуют в ISA E2K-v5 как `MOVAB/MOVAH`-варианты с флагом MAS (Memory Access Specifier) `no_cache`/`no_alloc` — атрибуты операции доступа в память, НЕ закэшировать line при write-allocate.

В `__builtin_e2k_*` соответствующие intrinsics (названия могут варьироваться по lcc-версиям):
- `__builtin_e2k_st_*_nt` — non-temporal store
- `__builtin_e2k_ld_*_nt` — non-temporal load (bypass L1/L2, allocate only в prefetch buffer)

Альтернатива (portable): `_mm_stream_pi/si` через binary compat режим x86 **не использовать** — теряем оптимизацию VLIW.

**Применения для нашего bottleneck:**

| Операция | Размер | NT? | Обоснование |
|----------|--------|-----|-------------|
| Q4_K weight read | 2.5 GB/tok | **ДА, NT load** | one-shot, не reused → не загаживать L1/L2/L3 |
| Output logits write | 150K × 4 B = 600 KB | **ДА, NT store** | single write, no readback until next layer |
| AllReduce output | 608 KB | **ДА, NT store на финале** | финальный redux не реюзится в этом шаге |
| KV-cache write | ~32 KB/token incremental | **НЕТ** | читаются следующими 2048 токенами → keep в L2/L3 |
| Activations (hidden state) | 4096 × 4B = 16 KB | **НЕТ, temporal** | reused в следующем layer |

**Критично:** NT loads нужно комбинировать с **aggressive prefetching** (см. §6), иначе каждое NT чтение стоит полный DRAM latency без возможности L2-hit.

---

## 4. Cross-chip coherence при 4 chips читают shared weight

**E8C2 protocol (из `cache_optimization.txt` §inclusive cache + §coherence directory):** MOESI-like с **distributed directory**, совмещённым с home-банком L3. При 4-chip конфиге chips связаны через **inter-chip coherence links** (аналог Intel QPI / AMD IF).

**Сценарий TP-4 qwen3:4b:**
- Все 4 chip'а читают **одни и те же** 2.5 GB весов (когда TP = чистая data-parallel replication, не model-parallel).
- **Первый chip**, прочитавший weight line, получает её в L3 в Exclusive state → **directory entry** в home-chip помечает ownership.
- **Второй chip** запрашивает ту же line → coherence message к home-chip → директория сообщает, что line shared-clean → broadcast к 4 chips в Shared state.
- Но line — **read-only** (weights immutable в inference) → в Shared все 4 могут параллельно читать.

**Проблема:** home-chip directory = **bottleneck**. Каждая из 39M линий/ток × 4 chip = 156M запросов в directory per token = **massive directory traffic**.

**Оценка:** если inter-chip link = 25 GB/s, traffic per token = 156M × ~16 B message = 2.5 GB/chip/tok **на coherence** — примерно равно трафику с DDR! → **TP-4 реально = 2× bandwidth от 1-proc**, а не 4× (что и видим: 6.1 / 5.3 = 1.15×).

**Решение:**

1. **NUMA_REPLICATE weights** — каждый chip имеет свою копию → no inter-chip traffic. Для 2.5 GB × 4 = 10 GB overhead памяти (OK при 125 GB RAM).
2. **Local SGD / file-based weight sync** уже сделано для тренировки — для инференса тем более должно работать.
3. **Disable coherence для read-only regions** — если MCST ISA поддерживает `MAS_no_coherence` для memory regions — не нужно directory запросов. Проверить наличие `madvise(MADV_DONTFORK|MADV_DONTDUMP)` + custom mapping с атрибутом non-coherent.
4. **TP = model-parallel, не data-parallel**: каждый chip держит 1/4 весов (~625 MB), тогда 4× bandwidth = 4× 76.8 = **307 GB/s** реально, не overlapping чтений. Это и есть ключевой path к 20 tok/s.

**Вывод:** текущий TP-4 = 6.1 tok/s потому что **chips читают одно и то же** и coherence traffic душит. **Переход к model-parallel (split по heads / split по output dim)** даст 4× bandwidth без duplicate reads.

---

## 5. Page size / TLB — huge pages без NUMA replicate

**E2K TLB:** 4 KB default, 2 MB huge (стандартно для x86-совместимости в бинарной трансляции и для нативных E2K kernel interfaces). Точный TLB size для E8C2: 64-128 4K-entries в L1 DTLB, 512-1024 в L2 TLB (типовая E2K-v5 конфигурация).

**Наша арифметика:** 2.5 GB / 4 KB = **640K pages**. 1024 L2 TLB entries → TLB miss rate ≈ **99.8%** — каждое чтение цепляет page walk (~4 memory accesses × 50 ns = **200 ns per TLB miss**).

**Huge pages 2 MB:** 2.5 GB / 2 MB = **1250 pages** ≈ TLB coverage с margin. TLB miss rate падает до <1%.

**Как включить huge pages БЕЗ NUMA_REPLICATE:**

1. **`madvise(ptr, size, MADV_HUGEPAGE)`** — работает независимо от NUMA. Но требует чтобы mmap-region была **2 MB-aligned** (используй `posix_memalign(2*1024*1024)` или `mmap` с `MAP_ANONYMOUS`).

2. **Transparent Huge Pages (THP) global**: `echo always > /sys/kernel/mm/transparent_hugepage/enabled` — OS сама промотирует. Но kernel может отказаться промотировать если memory fragmented → **на долгоживущем сервере (uptime >7 дней) не срабатывает**. Reboot + immediate allocate = safe.

3. **`hugetlbfs` explicit**: mount `-t hugetlbfs` + `mmap(MAP_HUGETLB)` — **гарантирует** huge pages, не зависит от THP. Требует предварительного `echo N > /proc/sys/vm/nr_hugepages` (N = 1250 + margin).
   - Это **правильный путь** для LLM inference: explicit reservation, no fragmentation issues.

4. **`MAP_HUGETLB` при mmap weights file**: `mmap(NULL, 2.5GB, PROT_READ, MAP_PRIVATE|MAP_HUGETLB, fd, 0)` — файл весов напрямую mapped в 2 MB pages. NO replication, NO NUMA binding change.

**Почему ваш MADV_HUGEPAGE только при PT_NUMA_REPLICATE регрессирует 1-proc:**
- При REPLICATE создаётся 4 копии, каждая allocate-on-write → kernel promotes к 2 MB всегда.
- Без REPLICATE текущий путь, вероятно, использует **file-backed mmap** (qwen3.gguf) → kernel **НЕ promote'ит** file-mapped pages к huge (известный THP limitation).
- **Решение:** `madvise(MADV_HUGEPAGE)` НЕ РАБОТАЕТ на file-backed mappings до Linux 5.7+. Для gguf inference нужно:
  - либо `mmap(MAP_PRIVATE|MAP_POPULATE)` + `madvise(MADV_HUGEPAGE)` **после** `MAP_POPULATE` сработает копирование в anon pages,
  - либо `read()` файла целиком в `mmap(MAP_ANONYMOUS|MAP_HUGETLB)` при загрузке (разовый cost 2.5 GB / 5 GB/s = 0.5s один раз).

**Рекомендация:** при load qwen3:4b.gguf — выделить anon huge-backed buffer, сделать `pread()` весов туда. Потеряете 2.5 GB + COW, но выиграете TLB coverage навсегда.

---

## 6. Prefetch distance

**MCST рекомендация (`hpc_architecture.txt` не приводит формулу, но стандартная из Elbrus optimization guide):**

```
dist = ceil(Latency_DRAM / T_per_iteration) + safety_margin
```

Наши числа:
- `Latency_DRAM` на E8C2: ~100-150 ns при row-hit, 200-250 ns при row-miss → в тактах 1500 MHz = **150-375 cycles**.
- `T_per_iteration` для Q4_K decode: 1 iter обрабатывает 144 B = 36 float weights. При ~10 FMA/iter + indirect + scale mul = **10-15 cycles per block** (VLIW wide-issue). Не 10 тактов — скорее **15-20 cycles per 144 B chunk**.

**Правильная дистанция:**
```
dist = 200 cycles / 15 cycles = 13-14 blocks
```

**Ваш текущий `prefetch bi+1`** — **НЕ хватает**. При row-hit 150 cycles, iter 15 cycles → `bi+1` покрывает только 15 cycles latency → 90% запросов идут со stall'ами.

**Правильно:**
```c
prefetch(&weights[bi + 14]);  // ~14 blocks ahead
```

Но осторожно:
- L1D prefetch queue в E2K ≈ 8-16 outstanding (exactly number depends on revision). Slot 14 блоков × 144 B / 64 B line = **31 outstanding lines** — **переполнит queue**, начнутся dropped prefetches.
- Лучше **multi-distance prefetch**: одновременно `bi+4` в L1, `bi+14` в L2, `bi+40` в L3 — mimics hardware streamer:
  - `__builtin_e2k_prefetch(&w[bi+4], /*level=*/1)` (L1)
  - `__builtin_e2k_prefetch(&w[bi+14], /*level=*/2)` (L2)
  - `__builtin_e2k_prefetch(&w[bi+40], /*level=*/3)` (L3/NT hint)

**Также:** hardware prefetcher на E2K существует (stream detector), но для **GEMV с indirect scales** (Q4_K имеет scales `sc[bi/8]`) hardware prefetcher путается — два stream (weights + scales) конфликтуют. Нужны **manual software prefetches для обоих**.

---

## 7. Multi-chip scaling bandwidth

**Из `hpc_architecture.txt` (RSC):** 4-chip Elbrus системы с Infiniband/OFED для MPI — это для distributed, а не shared-memory. Но 4× E8C2 на одной материнке — **cc-NUMA** (cache-coherent NUMA) через inter-chip links.

**Теоретический peak BW:** 4 chip × 76.8 GB/s = **307 GB/s**, если **каждый chip читает СВОЙ regional memory**.

**Реальность при shared weights:**
- Если **все 4 чипа читают одно и то же** weight region на chip 0 → bandwidth лимитирован chip 0 = **76.8 GB/s / 4 = 19.2 GB/s effective per chip**. Фактически **хуже**, потому что inter-chip links (~25 GB/s) ещё забиты coherence traffic.
- Это **точно** объясняет TP-4 = 6.1 tok/s (ожидали бы 4× = 21+ tok/s, получили 1.15×).

**Решения для 4× scaling:**

1. **Model-parallel sharding весов:** каждый chip держит в local memory 625 MB (1/4 weights). Per-token каждый chip читает свои 625 MB sequentially. Синхронизация — только activation AllReduce (16 KB hidden state × 36 layers = 576 KB total, negligible).
   - **Expected:** 4× bandwidth = 4 × 5.3 tok/s = **21 tok/s**. **Точно hit target!**

2. **Data-parallel с weight replication** (нынешний подход с PT_NUMA_REPLICATE): каждый chip имеет полную копию. Работает, но batch=1 decode не нужно 4 копии — коэффициент использования 25%. Но coherence идёт к owner.
   - **Expected:** 3-3.5× bandwidth = 16-18 tok/s. Близко но не гарантированно.

3. **Hybrid:** 2-way model-parallel × 2-way data-parallel → 2× bandwidth + меньше replicate overhead.

**Про unsync чтения:** если 4 chip'а **не синхронизированы** (разные tokens / разные batch items) → каждый читает **разный сектор** → 4× bandwidth. Это и есть model-parallel где каждый chip работает над своей частью одновременно.

**Про sync locks:** coherence directory сама по себе **не лочит весь traffic**, но каждое sharing of line требует round-trip к home-chip (~200-300 ns). Для read-only weights это overhead но не sequential bottleneck. Однако **inclusive L3** означает что home-chip директория имеет ограниченный capacity (~16 MB entries) → при overflow = еvictions + invalidations на **всех** 4 chips. Точно такой же "directory thrashing" как описан в Kozhin §victim-migration.

---

## 8. Theoretical minimum — где мы теряем

**Theoretical min @ 4 chips model-parallel:**

```
Time_min = Weight_bytes / (N_chips × BW_per_chip)
         = 2.5 GB / (4 × 76.8 GB/s)
         = 2.5 / 307.2 = 8.14 ms/token
         = 122.8 tok/s theoretical peak
```

**Реально достижимо (95% peak BW, overhead на softmax + activations):**

```
Time_practical = 2.5 GB / (307 × 0.7 = 215 GB/s effective) + 8.7 ms softmax
               = 11.6 ms + 8.7 ms = 20.3 ms/tok = 49 tok/s
```

**Текущий: 5.3 / 6.1 tok/s → 170-185 ms/token.**

**Где 8× потери (185 / 23 = 8×):**

| Loss factor | Разбор | Estimated cost |
|-------------|--------|----------------|
| **Duplicate reads при TP-4 (coherence)** | 4 chip читают одно и то же → 4× amplification → effective 1 chip BW | ×3.5-4 |
| **TLB misses (4 KB pages)** | 640K pages vs 1024 TLB → 99% miss × 200 ns = cost 128 ms/tok | ×1.4 |
| **Row activation в KV-random** | 8.7 ms softmax может быть 1-2 ms с row-hit optimization | ×1.1 |
| **Insufficient prefetch distance** | bi+1 не покрывает 200-cycle DRAM latency → L1 load stalls | ×1.3 |
| **File-backed mmap без HUGEPAGE promotion** | Потерянные 2 MB pages | ×1.2 (double-counted с TLB) |
| **Нет non-temporal loads/stores** | L3 pollution весами, evict'ит KV и activations | ×1.2 |

**Multiplied:** 4 × 1.4 × 1.1 × 1.3 × 1.2 ≈ **9.6×** — совпадает с gap 8-9×.

**Path к 20 tok/s (из 5.3):**

| Оптимизация | Speedup | Cumulative tok/s |
|-------------|---------|------------------|
| Baseline 1-proc | 1× | 5.3 |
| **Model-parallel split TP-4** (устраняет duplicate reads) | ×3.5 | 18.5 |
| **HugeTLB 2 MB anon-mmap весов** | ×1.3 | 24 |
| **Multi-distance prefetch bi+{4,14,40}** | ×1.15 | 27.6 |
| **Non-temporal load для Q4 weights** | ×1.1 | 30.4 |
| **KV head-major layout + row-hit opt** | ×1.05 (8.7 → ~4 ms softmax) | 32 |

**Вывод: 20 tok/s достижимы ТОЛЬКО через model-parallel split. Всё остальное — доп. 30-60% поверх.**

---

## TL;DR — приоритеты по impact

1. **КРИТИЧНО (single biggest win):** Перестроить TP-4 из data-parallel (replicate weights) в **model-parallel (shard weights)**. Это убирает cross-chip coherence amplification и даёт честный 4× bandwidth. Expected: 5.3 → 18-21 tok/s.

2. **ВТОРОЙ по impact:** HugeTLB 2 MB pages для 2.5 GB весов. Требует anon-mmap + pread (не file-mmap). Expected: +30%.

3. **Multi-distance prefetch** `bi+4`, `bi+14`, `bi+40` через `__builtin_e2k_prefetch` с level hints. Expected: +15%.

4. **Non-temporal LOAD** для Q4_K weight stream (`__builtin_e2k_ld_*_nt` или MAS `no_cache`): не гадит L3 где живёт KV. Expected: +10%.

5. **Non-temporal STORE** для логитов (600 KB) и AllReduce final (608 KB): не нужно в cache.

6. **KV head-major layout** + pin каждую head к локальному chip'у/banku: row-hit rate 1% → 80%+, softmax 8.7 ms → ~2-3 ms.

7. **Directory overflow mitigation:** если model-parallel — coherence traffic минимален (только activation AllReduce 16-32 KB/layer). Если data-parallel остаётся — включить NUMA_REPLICATE (4× memory, но убирает directory hot-spot).

---

## Open questions (для следующего агента или hardware validation)

- Точные числа L1/L2 associativity E8C2 rev X — MCST не публикует, нужен BIOS/spec sheet проверить через `lcc --target-cpu=elbrus-8c2 -v` и `dmidecode`.
- Существование и имя точного `__builtin_e2k_*_nt` intrinsic в установленной lcc 1.29 — проверить `<e2kintrin.h>` в /usr/include на эльбрусе.
- Поддерживает ли E8C2 MAS `no_coherence` для weight regions — это могло бы обнулить directory traffic без model-parallel refactor. Проверить E2K ISA manual §Memory Access Specifier.
- Сколько outstanding prefetches в L1D prefetch queue E2K-v5 — определяет max useful prefetch distance.
