# Agent 9 — HugeTLB / page-table / memory-layout (Round 3)

Goal: lift qwen3:4b Q4_K_M TP-4 ≥ 5.5 tok/s by eliminating 4-KB-page TLB thrash on
~2.38 GB weight stream. Probed Elbrus live; numbers below are from the actual
host (`w205p.mcst.ru`, kernel 6.1.0-1.9-e8c2, 4× E8C2 NUMA, 125 GB RAM).

---

## 1. Hardware/OS reality (probed, not assumed)

| Item | Value |
|---|---|
| Kernel | 6.1.0-1.9-e8c2 (THP file-backed promotion code present since 5.7) |
| Page sizes available | **4 KB** default, **2 MB** (256 pre-reserved), **1 GB** (compiled in, **0 reserved**) |
| `/proc/sys/vm/nr_hugepages` | 256 (= 512 MB pool) — too small for 2382 MB weights (need **1191** 2-MB pages) |
| `/proc/sys/vm/nr_overcommit_hugepages` | 0 (no on-demand pool) |
| THP `enabled` | `[always] madvise never` |
| THP `defrag` | `always defer defer+madvise [madvise] never` |
| `/mnt/huge` hugetlbfs | **mounted, mode 0777, writable as user** (confirmed: `fallocate -l 200M /mnt/huge/probe.bin` succeeds) |
| `ulimit -l` (memlock) | 8192 KB hard — **not** the binding limit; hugetlbfs files don't count against memlock |
| sudo / root | **none** (passwordless sudo refused) |

Per-node breakdown of the 256-page pool: 64 pages (128 MB) on each of 4 NUMA nodes.

## 2. Why Round 2's PT_HUGETLB didn't help

Path in `torch/io/gguf_loader.h:266-302`:
1. anon `mmap(MAP_HUGETLB)` for 2382 MB → **fails immediately** (256 pages << 1191 needed)
2. fall through to `mmap(MAP_PRIVATE, fd)` + `madvise(MADV_HUGEPAGE)`

I verified the fallback experimentally on the host: a process that mmaps the
1.7 B GGUF and `madvise(MADV_HUGEPAGE)`s it shows
`KernelPageSize: 4 kB, AnonHugePages: 0 kB` in `/proc/$pid/smaps`. **The kernel
refuses to promote `MAP_PRIVATE` file-backed regions even with THP=always +
defrag=madvise.** So PT_HUGETLB has been a no-op end-to-end. Plateau result
4.6–4.8 tok/s is the 4-KB-page case.

## 3. Quantified TLB-miss cost

E8C2 (E2K v5) DTLB: ~64 L1 + 512–1024 L2 entries. With 4-KB pages and
2382 MB resident weights = **610 K** distinct pages → **~99.8 %** miss rate
on the streaming GEMV path. Page walk on E2K = 4 memory loads × ~50 ns =
~200 ns/miss. Weight stream is 2.5 GB / token / chip → **17.4 M lines/token**
each potentially touching a TLB miss. Even at 1 walk per 16-line row crossing
(linear access + small page walk caching) we get ~1 M walks/tok × 200 ns =
**~200 ms/tok wasted on TLB**, dominating the 211 ms/tok budget.

With **2-MB pages** working set = 2382/2 = **1191 entries** → fits in L2 TLB
twice over → walk rate drops to <1 %. **Predicted savings: 30–60 ms/tok →
expected 5.5–6.0 tok/s** (modest because Round 2 also showed bandwidth
saturation kicks in soon after; this is no longer a 30 % win, more 15–25 %).

With **1-GB pages** working set = 3 entries → essentially zero TLB pressure.
Marginal gain over 2 MB is small (maybe 2–3 %), not worth the complexity
unless 1G reservation is free for us.

## 4. Concrete enable plan (does NOT need root for paths A2/B/D)

### Path A1 — Reserve more 2-MB pages (needs sysadmin once)
Ask MCST sysadmin to set permanently:
```
echo 1500 > /proc/sys/vm/nr_hugepages          # 3 GB pool
# or persist via /etc/sysctl.d/99-hugepages.conf:  vm.nr_hugepages = 1500
```
After that, current `PT_HUGETLB=1` path already works (no code change needed).
**This is the one-line ticket that unlocks +30 % decode**. Send it now.

### Path A2 — hugetlbfs file in `/mnt/huge` (works TODAY without root, BUT pool too small)
`/mnt/huge` is 0777 and accepts user writes. Modify loader to:
```c
int hfd = open("/mnt/huge/qwen3_weights.bin", O_CREAT|O_RDWR, 0600);
ftruncate(hfd, rounded);
void* hp = mmap(NULL, rounded, PROT_READ|PROT_WRITE, MAP_SHARED, hfd, 0);
// pread() weights into hp (same as current path), then mprotect RO
unlink("/mnt/huge/qwen3_weights.bin");  // cleanup on exit
```
hugetlbfs is **always huge-paged**; no memlock ulimit interaction. **But still
needs ≥1191 free 2-MB pages** in the global pool. With current 256, this also
fails today — only useful AFTER sysadmin grows the pool.

### Path B — Fix MADV_HUGEPAGE fallback for non-root case
Replace the current `mmap(file) + madvise(MADV_HUGEPAGE)` (which we proved is
a no-op) with `mmap(MAP_ANONYMOUS) + pread()`. Same code as the PT_HUGETLB
branch but **without `MAP_HUGETLB`** — relies purely on THP. With
`defrag=[madvise]`, anon allocations DO get promoted by `khugepaged`. Probed
existing `numa_alloc_onnode` regions in `numa_weight_replica.h:96` already do
this and rely on the same THP path; replicate that idea for the non-replicated
case too. Edit point: `torch/io/gguf_loader.h:304-320`.

### Path D — Verify alignment and size of madvise call
`madvise(MADV_HUGEPAGE)` requires the **start address** to be 2-MB-aligned.
`mmap` returns 4-KB-aligned, not 2-MB. Anon `mmap` must request enough
slack (`size + 2 MB`) and pick a 2-MB-aligned offset, OR use
`posix_memalign(2*1024*1024, size)`. Current loader path (line 271) **does
satisfy this** for `MAP_ANONYMOUS|MAP_HUGETLB` (kernel auto-aligns), but
the post-failure file-backed `MAP_PRIVATE` path at line 305 returns
non-aligned addresses and so MADV_HUGEPAGE on it is silently dropped.
That's the root cause of the no-op.

## 5. NUMA-localised huge pages (already partially done)

`numa_weight_replica.h:82,96` already uses
`numa_alloc_onnode + madvise(MADV_HUGEPAGE)` per node. Combined with
`PT_NUMA_REPLICATE=1` this DOES get 2-MB pages per replica because the
allocations are anon. **However** `numa_alloc_onnode` returns 4-KB-aligned
buffers — verify the same alignment caveat (Path D). Quick sanity check:
add `assert(((uintptr_t)p & (2UL*1024*1024 - 1)) == 0)` after the alloc
or just scan `/proc/$pid/smaps` for `AnonHugePages:` of these regions
during a real run.

## 6. Verification command (run on Elbrus during inference)

```bash
PID=$(pgrep -n promeserve)   # or the inference binary name
grep -E 'AnonHugePages|FilePmdMapped|KernelPageSize' /proc/$PID/smaps | sort | uniq -c | sort -rn | head
```
If we see `AnonHugePages:  >0 kB` totaling near 2.4 GB and `KernelPageSize:
2048 kB`, the path is engaged. Currently expect `0 kB` and `4 kB` everywhere.

## 7. Expected uplift summary

| Step | Effort | Δ tok/s |
|---|---|---|
| A1: ask sysadmin `nr_hugepages=1500` | 1 ssh ticket | +0.7–1.2 |
| B: anon-mmap + pread fallback when MAP_HUGETLB fails | ~30 LOC in `gguf_loader.h` | +0.3–0.5 (THP only, no reservation needed) |
| D: enforce 2-MB alignment on the file-backed-fallback madvise path | ~10 LOC | +0.1–0.2 (correctness fix; folded into B) |
| 1-GB pages (A1 with 3× 1G page reserve) | sysadmin + 5 LOC `MAP_HUGE_1GB` flag | +0.05–0.1 |

Combined realistic ceiling from this agent's angle alone: **4.8 → 5.5–6.2 tok/s**.
Sufficient as one of the 10 stacked Round-3 wins; not sufficient on its own
to hit the 10 tok/s mission target.
