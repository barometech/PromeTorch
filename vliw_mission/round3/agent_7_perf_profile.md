# Agent 7 — HW-counter profiling план для Эльбрус 8C2

Target: qwen3:4b Q4_K_M TP-4 = 4.8 tok/s. BW 23%, compute 17% — gap в pipeline-stall событиях, отсутствующих в обеих метриках.

## Главная находка: `perf` бесполезен — нужен `dprof`

`perf list` на E8C2 даёт ~5 events (`cycles`, `instructions`, `stalled-cycles-frontend`, `L1-dcache-stores`, `LLC-stores`). **Нет L1-load-misses, dTLB, branch-misses.** Raw `r000…r07f` работают, имена не задокументированы.

**Инструмент: `/opt/mcst/bin/dprof`.** `dprof --mlist` = **178 E2K-events** на 4 каналах (DDM0/1, DIM0/1). Снимает **до 4 events за прогон**: `dprof -m 0:DTLB_QUERY -m 1:L1_LD_HIT -m 2:TICKS -m 3:EXEC ./prog`.

System: 8c×4 sockets, L1d 64KB, L2 512KB, L3 16MB/socket. **HugePages_Total=256 (512MB) при 2.5GB weights → покрытие ~20%, серьёзное TLB-давление.**

## Top events для weight-streaming GEMV

| dprof event | channel | гипотеза |
|-------------|----|----------|
| `TICKS`, `EXEC` | 2/3 | IPC = EXEC/TICKS, ceiling = 6 |
| `IB_NO_COMMAND` | 2/3 | front-end starvation (I-cache, branch) |
| `BUB_E2_MEM` | 2/3 | **главный bandwidth-stall** (replay E2 из-за нет операндов из памяти) |
| `BUB_E0`, `TICKS_E0` | 2/3 | mispredict + scheduling conflicts |
| `DTLB_QUERY`, `TLB_MISS_STALL` | 0 | **TLB-thrash test** на 2.5 GB weights |
| `L2_QUERY`, `L2_HIT` | 0/1 | L2 hit-rate (streaming weights должны полностью миссить) |
| `MAU_LDB_LINE`, `MAU_LDBOVFL_STALL` | 0 | глубина outstanding loads, переполнение очереди |
| `MAU_READ_RI_LOCAL` vs `MAU_READ_RI_REMOTE` | 1 | **NUMA cross-traffic** между socket'ами в TP-4 |
| `L2_EXT_QUERY` | 0 | coherence ping-pong (cache-line bouncing) |

## Интерпретация ratio

- `EXEC/TICKS < 0.5` — катастрофа (GEMV ожидаемо 0.3–0.6).
- `BUB_E2_MEM/TICKS > 30%` — BW-bound; `>60%` — окно outstanding loads мало.
- `IB_NO_COMMAND/TICKS > 10%` — front-end starvation; `-fwhole`+PGO (R2 Agent 1).
- `TLB_MISS_STALL/TICKS > 5%` — расширять hugepage пул.
- `MAU_LDBOVFL_STALL/TICKS > 5%` — prefetch чрезмерен.
- `L2_HIT/L2_QUERY > 30%` — подозрительно (re-reads, weights не партиционированы).
- **`MAU_READ_RI_REMOTE/(LOCAL+REMOTE) > 10%` при `numactl --membind`** — критично: ranks читают чужой socket → объясняет 23% BW (4 sockets × 12.2 GB/s = 48.8 peak, получаем 2.9 = 6% aggregate).

## Эксперименты

### 1. Pipeline breakdown (3 прогона по 50 токенов)

```bash
# back-end / front-end split
dprof -m 2:TICKS -m 2:EXEC -m 2:IB_NO_COMMAND -m 2:BUB_E2_MEM ./run_inference ...
# memory pipeline
dprof -m 0:DTLB_QUERY -m 0:TLB_MISS_STALL -m 0:MAU_LDB_LINE -m 0:MAU_LDBOVFL_STALL ...
# cache
dprof -m 0:L1_QUERY -m 1:L1_LD_HIT -m 0:L2_QUERY -m 1:L2_HIT ...
```

Решения: IPC<0.4 + BUB_E2_MEM>50% + L2_HIT<5% → BW ceiling. BUB_E2_MEM<30% + IPC низкий → не память (BUB_E0/BUB_B). TLB_MISS_STALL>5% → Эксп.2. REMOTE>10% → Эксп.3.

### 2. HugeTLB sweep

Сейчас 256 hugepages = 512MB / 2.5GB weights.

```bash
sudo sh -c 'echo 1500 > /proc/sys/vm/nr_hugepages'
PT_HUGETLB=1 dprof -m 0:DTLB_QUERY -m 0:TLB_MISS_STALL -m 2:TICKS -m 3:EXEC ./tp4_run.sh
```

TLB_MISS_STALL падает >5%→<1% + tok/s 4.8→5.5–6.5 = TLB был узким местом. 0% delta (как R2 на 256) = TLB не ограничитель.

### 3. NUMA cross-socket sanity

```bash
for r in 0 1 2 3; do
  numactl --cpunodebind=$r --membind=$r \
    dprof -m 1:MAU_READ_RI_LOCAL -m 1:MAU_READ_RI_REMOTE -m 0:L2_EXT_QUERY -m 2:TICKS \
    ./tp4_worker --rank $r > numa_r$r.log &
done; wait
```

REMOTE/(LOCAL+REMOTE)>10% при `--membind` = утечка (SHM на чужом узле, weights не реплицированы). Фикс: реплицировать weights ×4 (×4 RAM, но aggregate 48.8 GB/s).

## Дальнейшее

Если оба отриц.: `BUB_E2_ALU0_MEM…ALU5_MEM` — какая VLIW-полоса холостая. `BUB_B_FAPB_CNFLCT` — APB-конфликт (R2 §5.2: 144B stride не степень 2). IB_NO_COMMAND>10% → `-fwhole`+PGO. Полный список — `dprof --mlist`.
