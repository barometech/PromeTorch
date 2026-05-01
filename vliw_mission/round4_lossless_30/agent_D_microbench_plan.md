# Agent D — Microbench Plan (Round 4)

**Author:** Agent D (Round 4)
**Date:** 2026-04-29
**Goal:** Define every microbench that MUST run, with pass/fail numeric gates,
*before* any production integration of the Round 4 kernel optimisations
(A1, A4, A5, A6 from `agent_D_results.md`).

Rule: **microbench-driven**. No optimisation lands in production until its
microbench gate is met on the actual Эльбрус 8C2.

---

## 0. Test environment baseline

All microbenches run on:
- Host: `paperclipdnb@w205p.mcst.ru:8199` (32-core E8C2, 1.5 GHz, 125 GB RAM)
- LCC: 1.29
- Build: `lcc -O3 -ffast -march=elbrus-v5 -mtune=elbrus-8c2 -faligned -fprefetch
  -frestrict-all -fswp-maxopers=800 -fopenmp -I/usr/include/eml`
- Repeat ≥ 3 runs; report median; coefficient of variation < 5 %.
- Env: `loginctl enable-linger` already set on user; `tmux` for long runs.

---

## M1 — ThreadPool overhead microbench (gates A1)

### What
Re-use existing `examples/benchmarks/threadpool_overhead_bench.cpp`. Currently
NOT in CMakeLists. First step: wire it up.

### Build (Эльбрус)
```bash
cd ~/promethorch
# Add to examples/benchmarks/CMakeLists.txt:
#   add_executable(bench_threadpool_overhead threadpool_overhead_bench.cpp)
#   target_link_libraries(bench_threadpool_overhead PRIVATE c10 pthread)
mkdir -p build_round4 && cd build_round4
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=lcc -DPT_BUILD_BENCHMARKS=ON
make bench_threadpool_overhead -j8
```

### Run (per-config)
```bash
# Old pool baseline (mutex+CV+queue)
PT_NUM_THREADS=7 PT_PIN_THREADS=1 \
    ./examples/benchmarks/bench_threadpool_overhead 10000 7

# After A1 patch (broadcast pool with futex gen)
PT_NUM_THREADS=7 PT_PIN_THREADS=1 PT_TP_BROADCAST=1 \
    ./examples/benchmarks/bench_threadpool_overhead 10000 7
```

### Gate (PASS criteria)
| Metric | Old pool baseline | A1 broadcast pool gate | Hard ceiling |
|--------|-------------------|------------------------|--------------|
| `parallel_for(noop, 7-fanout)` | 90-150 µs/call | **≤ 8 µs/call** | 12 µs/call |
| `parallel_for(small-sum, 7-fanout)` | 100-180 µs/call | **≤ 12 µs/call** | 18 µs/call |
| Per-token estimated overhead | 16-27 ms | **≤ 1.4 ms** | 2.2 ms |

If A1 broadcast pool fails (≥ 12 µs/call) → revert, keep current pool, invest
budget elsewhere.

### Numerical equivalence check
Run inference smoke test: `./test_gguf_inference qwen3-4b-Q4_K_M.gguf "The sky" 8`
under both old and new pool. Logits must match within fp32 round-off (max abs
diff < 1e-5). If mismatch — race condition in broadcast pool. Revert.

---

## M2 — Q4_SoA4 GEMV microbench (gates A6)

### What
Mirror of `vliw_mission/e2k_vnni/q8_soa4_microbench.c`, but inner loop uses
4-bit packed weights (16 weights / 16 bytes per K-pair, see format_spec_v1
§10). Tests: nibble unpack cost, BW utilisation, correctness vs FP32 ref.

### File to create
`vliw_mission/e2k_vnni/q4_soa4_microbench.c` (this microbench is created BY
Agent D, not Agent B). Layout per 4-row × 32-K super-block: 88 bytes:
- 8 B  = `fp16 d_w[4]`
- 8 B  = `fp16 dmin_m[4]`
- 8 B  = `i16 sum_q[4]`
- 64 B = packed nibbles, 4 rows × 32 elem × 4 bits, byte-interleaved per K-pair

### Inner loop sketch (target VLIW issue rate)
```c
v2di MASK_0F = ...;  // 0x0F broadcast
for (int kg = 0; kg < 4; kg++) {  // 4 K-groups × 8 K = 32 K-elements / block
    v2di W_packed = W_v[kg];                                  // 16 B nibbles
    v2di W_lo = qpand(W_packed, MASK_0F);                     // 16 i8 lanes
    v2di W_hi = __builtin_e2k_qpsrlw(W_packed, 4);            // 16 i8 lanes
    W_hi = qpand(W_hi, MASK_0F);
    v2di p16_lo = __builtin_e2k_qpmaddubsh(W_lo, A_v[kg*2]);
    v2di p16_hi = __builtin_e2k_qpmaddubsh(W_hi, A_v[kg*2+1]);
    v2di p32_lo = __builtin_e2k_qpmaddh(p16_lo, ONES16);
    v2di p32_hi = __builtin_e2k_qpmaddh(p16_hi, ONES16);
    acc_i32 = __builtin_e2k_qpaddw(acc_i32,
              __builtin_e2k_qpaddw(p32_lo, p32_hi));
}
```

### Build (Эльбрус)
```bash
cd ~/promethorch/vliw_mission/e2k_vnni
lcc -O3 -ffast -march=elbrus-v5 -mtune=elbrus-8c2 -faligned -fprefetch \
    -frestrict-all -fswp-maxopers=800 -I/usr/include/eml \
    q4_soa4_microbench.c -leml_algebra_mt -lm -o q4_soa4_microbench
```

### Run
```bash
./q4_soa4_microbench   # K=2560 N=2432, 200 iters
```

### Gate (PASS criteria, single-core)
| Metric | Target | Floor (still ship) | Reject |
|--------|-------:|-------------------:|-------:|
| ms / GEMV | **≤ 0.70 ms** | 0.90 ms | > 1.0 ms |
| GOPS | ≥ 17.8 | ≥ 13.8 | < 12.5 |
| BW util @ 25 GB/s/rank | ≥ 60 % | ≥ 50 % | < 40 % |
| max_err vs fp32 ref | < 0.001 | < 0.005 | ≥ 0.01 |

Reference: Q8 SoA4 = 1.21 ms, 10.3 GOPS. Q4 SoA4 should hit ~0.65 ms because
half the bytes per param. If actual ≥ 1.0 ms → unpack overhead ate the win,
abandon A6, stay on Q8 SoA4.

### Numerical equivalence check
Re-dequantise random Q4 SoA4 block to fp32 → compare with Q4_K → fp32 of same
nibbles. Diff must be 0 (bit-exact reorder).

---

## M3 — Fused QKV / gate-up GEMV microbench (gates A4)

### What
Compare 3 separate `q8_soa4_gemv` calls vs 1 fused call sharing the `a_b16`
buffer streaming. Measures cache-line reuse benefit on `a_b16`.

### File to create
`examples/benchmarks/fused_qkv_soa_microbench.cpp`. Allocates 3 weight matrices
(N_q=2048, N_kv=512, N_kv=512, K=2560), shared activation. Runs:
1. 3 separate `q8_soa4_gemv` calls
2. 1 fused `q8_soa4_qkv_gemv` (new fn)

### Build / Run
```bash
make bench_fused_qkv_soa -j8
PT_NUM_THREADS=1 ./examples/benchmarks/bench_fused_qkv_soa 200
```

### Gate
| Metric | Target |
|--------|-------:|
| Fused / 3-separate ratio | ≤ 0.85 (≥ 15 % saving) |
| max_abs_diff Y_q,Y_k,Y_v separate vs fused | 0 (bit-exact) |

If ratio ≥ 0.95 → fusion gives nothing, drop A4.

---

## M4 — Manual prefetch / APB benefit microbench (gates A5)

### What
Compare `q8_soa4_gemv` (or Q4 if A6 landed) with and without explicit
`E2K_PLD_L2(sb + SOA4_GROUP_BYTES)` at the top of the b-loop. Validate APB
hint via `_Pragma("loop count(K/32)") _Pragma("ivdep")`.

### File
Edit existing `q8_soa4_microbench.c` (or `q4_soa4_microbench.c`):
- macro-toggleable `#define ENABLE_PLD 1`
- macro-toggleable `#define ENABLE_LOOP_HINT 1`

Run all 4 combinations.

### Gate
At least one combination delivers **≥ 8 % improvement** over both flags off,
on single-core.

If neither helps → APB / prefetch dead-end, drop A5.

### dprof verification (optional but informative)
```bash
dprof -m TICKS,EXEC,BUB_E2 ./q4_soa4_microbench 2>&1 | tee dprof.log
```
Compare BUB_E2 (memory-stall bubbles) before/after PLD.

---

## M5 — End-to-end TP-4 baseline (gates everything)

### What
After each optimisation lands, re-run full TP-4 inference and confirm
end-to-end tok/s improvement matches microbench prediction.

### Run
```bash
cd ~/promethorch
./scripts/run_tp_elbrus.sh --greedy "The history of Russia begins" --tokens 100
# expect output median tok/s reported by harness
```

3 runs × 100 tokens × greedy decoding. Report median.

### Gate ladder
| Stage | Min acceptable tok/s | Target |
|-------|---------------------:|-------:|
| Baseline (commit 3399136) | 9.0 | 9.4 |
| After A1 | 10.0 | 10.6 |
| After A1+A4 | 10.8 | 11.2 |
| After A1+A4+A5 | 11.5 | 11.9 |
| After A1+A4+A5+A2 | 12.5 | 12.7 |
| After +A8 tail | 13.5 | 13.7 |
| After +A6 (Q4 SoA4 swap) | 18.0 | 19.6 |
| After loop hints / Agent 4 #7 | 20.0 | 20.8 |
| **Final (kernel-only) acceptance** | **20** | **22** |
| With Agent E A3 spec decode | 28 | 30 |

Each stage MUST pass numerical equivalence: `max abs diff vs Q4_K_M baseline
logits < 1e-5` over 100-token greedy decode.

---

## M6 — AllReduce overlap / reduce-scatter microbench (gates A2)

### What
Standalone benchmark of `all_reduce_inplace(H=2560)` vs new
`reduce_scatter_inplace + all_gather_inplace`, both with SHM+futex backend on
4-rank Эльбрус.

### File to create
`examples/benchmarks/allreduce_topology_microbench.cpp`.

### Run
```bash
mpirun -n 4 ./bench_allreduce_topology 10000 2560
# or with our launcher:
for r in 0 1 2 3; do
    PT_DDP_RANK=$r PT_DDP_NPROCS=4 PT_DDP_SHM=1 \
        numactl --cpunodebind=$r --preferred=$r \
        ./bench_allreduce_topology 10000 2560 &
done
wait
```

### Gate
| Metric | Now | After A2 target |
|--------|----:|----------------:|
| 1 AR(H=2560) µs (SHM) | 100-200 | < 60 |
| 1 RS+AG(H=2560) µs | n/a | < 90 |
| 1 RS+AG(inter=6912 / 4) µs | n/a | < 70 |
| Per-token total AR (72 calls) | 9 ms | ≤ 5 ms |

If RS+AG is *worse* than AR — keep AR, scrap the rewrite half of A2 (still
keep the prefetch-overlap half).

---

## M7 — Activation quantization microbench (gates A4 indirect)

### What
`q8_soa4_quant_activation(K=2560)` is called 5×/layer × 36 = 180 times/token.
Each costs ~0.02 ms = 3.6 ms/token. If we drop redundant calls (QKV shares,
gate+up shares, ffn_down separate), we go to 3 calls/layer = 108/token → 2.16 ms.

### File
`examples/benchmarks/quant_activation_microbench.cpp` — measures absolute call
cost and confirms shared-x trick correctness.

### Gate
- Single call ≤ 0.025 ms (no regression).
- Shared-x output bit-identical to per-call quant (since x is the same).

---

## Run order on Эльбрус (full session)

```bash
# 1. ssh in
plink -P 8199 -i elbrusssh.ppk -hostkey "SHA256:..." paperclipdnb@w205p.mcst.ru
loginctl enable-linger $USER  # MUST after every reboot

# 2. tmux session
tmux new -s round4

# 3. Build round4 once
cd ~/promethorch && mkdir -p build_round4 && cd build_round4
cmake .. -DCMAKE_BUILD_TYPE=Release -DPT_BUILD_BENCHMARKS=ON
make -j8 bench_threadpool_overhead bench_fused_qkv_soa \
        bench_allreduce_topology bench_quant_activation \
        test_gguf_inference

# 4. Microbench gauntlet
./examples/benchmarks/bench_threadpool_overhead 10000 7  | tee m1.log
cd ../vliw_mission/e2k_vnni && lcc -O3 -ffast -march=elbrus-v5 ... \
    q4_soa4_microbench.c -o q4_soa4_microbench
./q4_soa4_microbench | tee m2.log
cd -
./examples/benchmarks/bench_fused_qkv_soa 200 | tee m3.log
./examples/benchmarks/bench_quant_activation 1000 | tee m7.log

# 5. AR topology (4-rank parallel)
for r in 0 1 2 3; do
    PT_DDP_RANK=$r PT_DDP_NPROCS=4 PT_DDP_SHM=1 \
        numactl --cpunodebind=$r --preferred=$r \
        ./examples/benchmarks/bench_allreduce_topology 10000 2560 \
        > m6_rank$r.log &
done; wait

# 6. End-to-end TP-4 (production smoke)
./scripts/run_tp_elbrus.sh --greedy "The sky is" --tokens 100 | tee m5.log
```

---

## What "good" looks like

After all microbenches pass and end-to-end TP-4 hits **≥ 22 tok/s** on
3-run-median, escalate to Agent E for speculative decoding integration. If
spec decode adds × 1.4-1.6 effective throughput, **30 tok/s is reached**.

If at any microbench gate we miss target *and* there's no workaround within
the gate's session budget → escalate decision: drop that optimisation, move
budget to next, re-evaluate the 30-target feasibility with Agent A.

---

*End of agent_D_microbench_plan.md*
