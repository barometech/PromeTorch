# nanoGPT TinyStories на NM Quad (NMC4 VLIW)

End-to-end byte-level transformer training pipeline на NM Quad плате (4 × NM6408
chips × 4 NMC4 cores = 16 cores). Forward, backward, AdamW, host inference — всё в
чистом C для `nmc-gcc 4.8.3` toolchain.

## Архитектура моделей

| Версия | Layers | D | T | FF | Params | Файл |
|---|---|---|---|---|---|---|
| v1 (`nanogpt_train.c`) | 1 | 32 | 32 | 64 | 17 K | first end-to-end |
| v2 (`..._v2.c`) | 1 | 32 | 32 | 64 | 17 K | + EMI dataset + per-core slot |
| v3 (`..._v3.c`) | 1 | 32 | 32 | 64 | 17 K | + AdamW + grad clip + cosine LR |
| **v4** (`..._v4.c`) | **2** | 32 | 32 | 64 | **26 K** | 2-layer + best-snapshot checkpointing |

VOCAB всегда 128 (byte-level ASCII).

## Host drivers (libnm_quad_load)

| Файл | Описание |
|---|---|
| `host_one_core.cpp` | Single core 0:0, чистый запуск без races |
| `host_train_all.cpp` | Все 16 cores параллельно (default-mode binary, **WARNING: shared-memory race на static arrays in EMI**) |
| `host_train_real.cpp` | + TinyStories upload в EMI через PL_WriteMemBlock |
| `host_train_and_gen.cpp` | + read best weights + host inference (greedy) |
| `host_4chip_safe.cpp` | 4 чипа × 1 core (cluster physical EMI isolation, безопасно) |
| `host_all16_batched.cpp` | Все 16 cores через 4 sequential batches (1 core per cluster в каждом раунде) |
| `host_16core_fixed.cpp` | Через 4 NMC_INDEX'd binaries (USE_ONLY_EM mode — медленно, code тоже в EMI) |

## Python inference

`infer.py` auto-detects 1- или 2-layer checkpoint по размеру файла. Запуск:

```bash
python3 infer.py "Once upon a time " 300 0.6   # 0.6 = temperature
```

## Сборка на NM Quad сервере

```bash
ssh <user>@<nmquad-host> -p 4079
cd ~/nanogpt/v1
export PATH=/usr/local/rc_module_nmc4_toolchain/bin:$PATH
export NM_QUAD=/usr/local/rc_module/board-nm_quad

# NMC binary
make BOARD=nm_quad BOARD_PATH=$NM_QUAD

# Host driver
g++ -O2 -pthread host_one_core.cpp -DNM_QUAD \
    -I$NM_QUAD/include -L$NM_QUAD/lib \
    -Wl,-rpath=$NM_QUAD/lib -lnm_quad_load -lio_host -ldl \
    -o host_one_core

./host_one_core "Once upon a time " 300
```

## Memory model (важные nuances)

- **NMC4 IM = 512 KB** per core (fast, private)
- **EMI = ~5 GB** per cluster (shared across 4 cores within cluster, slower)
- **Char = 32-bit word** на NMC4: 1MB chars в исходниках → 4MB EMI footprint
- **CRITICAL bug** для multi-core: default linker mode кладёт ВСЕ static arrays
  (`Wtok`, `vWtok`, `mWtok`, etc.) в EMI shared region → 4 cores within cluster
  пишут в одни и те же веса → race. Симптомы: identical loss between cores in
  cluster, generation выдаёт NULL bytes.

## Fix для multi-core

Три варианта (все задокументированы, выбран B):

| Опция | Описание | Trade-off |
|---|---|---|
| A: `USE_ONLY_EM=y NMC_INDEX=0..3` | 4 binaries × disjoint EMI subregions | Код тоже в EMI = ~30× медленнее |
| **B: Sequential batches** ⭐ | 4 батча × 4 cores (1 per cluster, разные физические EMI banks) | 4× wall time но fast код |
| C: Custom linker .im_data | Per-core writable IM section | Complex linker script change |

`host_all16_batched.cpp` — option B implementation.

## Лучший результат

1-layer AdamW (`v3`) on 4MB TinyStories, 2000 steps, ~50 min wall:

- **FINAL loss = 2.37 nats ≈ 3.42 bits/byte**
- Real language modeling, не overfit
- Generation показывает разборчивые English fragments

Sampling temp=0.6:
> Once upon a time **thed wed lathe be the wite agede fe an thed tond ad
> thele a thed she lang aled then Sand m the se ay thithe athe s tished an
> pe an we s was ndand d iasohe tand athand the s wa ly tothend and. as
> anlamey a ban pnd tothesamede w sa anx. the asod soond ton bit an ther
> f a aland foto amand he tatd uy th**

Видны: **the, be, an, then, she, was, and, he, lathe (=lathe), lang (=long?),
ban, foto (=photo?)**, периоды и капитализация после "." → "Sand".

## Открытые задачи (next session)

1. **Qwen-4B GGUF inference на NMC4** — основная цель.
   - Per-cluster EMI ~5GB hosts Q4_K quantized 4B model (~2.5GB)
   - Per-NMC core IM (512KB) — workspace для one-layer-at-a-time compute
   - Требуется: Q4_K dequant on NMC4, attention kernel, KV cache, GGUF loader
   - Объём: ~3000 LoC, ~2 недели работы
2. **Real gradient sync** между cores (vs current independent ensemble training)
3. **BPE tokenizer** вместо byte-level (vocab 1-8K)
4. **Persistent checkpointing** при training overrun (best-loss filter уже добавлен)
