# `.pt8` — PromeTorch native weight container

Round 4 Mission, Agent C (loader/integration). 2026-04-30.

This document describes the **runtime contract** between the on-disk `.pt8`
file and the runtime path implemented in `torch/io/pt8_reader.h` +
`torch/io/gguf_model.h::GGUFModel::load_pt8()`. It is the authoritative
description of the bytes that any compliant writer or reader must produce
or accept.

For the high-level format motivation, dtype rationale, and the future
`PT8_Q4_SOA4` ⭐ hot path, see Agent A's spec:

> `vliw_mission/round4_lossless_30/format_spec_v1.md` — *PromeTorch `.pt8`
> Binary Format — Specification v1*

For the writer pipeline (CLI / GUI / threading / encoder registry), see
Agent B's converter:

> `tools/gguf2pt8/converter.{h,cpp}` and `tools/gguf2pt8/main.cpp`

---

## On-disk layout (v1)

```
+-----------------------------------------+ offset 0
|  FILE HEADER  (256 bytes, fixed)        |
|    u32  magic   = 'PT8\0' (0x00385450)  |
|    u32  version = 1                     |
|    u64  flags                           |
|    u64  tensor_table_offset             |
|    u64  tensor_count                    |
|    u8[224] reserved (must be zero)      |
+-----------------------------------------+
|  TENSOR DATA  (each tensor 64-byte      |
|  aligned, written in submit-order by    |
|  the writer thread)                     |
+-----------------------------------------+ tensor_table_offset
|  TENSOR TABLE (variable length, packed) |
|    repeat tensor_count times:           |
|      u32     name_length                |
|      u8[]    name (UTF-8, no NUL)       |
|      u32     pt8_type                   |
|      u32     n_dims (≤ 8)               |
|      u64[n_dims]  dims                  |
|      u64     data_offset (absolute)     |
|      u64     data_size                  |
|      u64     row_stride (bytes/row, 0   |
|              if N/A)                    |
|      u32     meta_length                |
|      u8[]    meta_blob (per-encoder     |
|              side data)                 |
+-----------------------------------------+ EOF
```

All multi-byte integers are little-endian. Tensor data is 64-byte aligned
to satisfy E2K v5 cache-line boundaries and APB prefetch.

Compared to Agent A's spec §3 (which uses a fixed 96-byte `TensorEntry` and
a separate string pool), the **actually shipping** Round 4 v1 layout is the
variable-length tail-table form above: it allows the writer thread to emit
data into the file in submit-order without knowing offsets up-front, then
patch the table offset into the header at the end. The contract is
upheld through the dtype enum and `pt8_type` field, both of which match
the spec's intent. Future versions may switch to the fixed-entry form
behind a `version=2` bump.

---

## `pt8_type` enum

```cpp
enum Pt8Type : uint32_t {
    PT8_TYPE_F32           = 0,    // raw fp32
    PT8_TYPE_F16           = 1,    // raw fp16
    PT8_TYPE_BF16          = 2,    // raw bfloat16
    // 100..199 — Agent A primary lossless layouts
    PT8_TYPE_Q4K_SOA4      = 100,  // 4-row Q4 SoA4, 0.6875 B/param  (spec §10) ⭐
    PT8_TYPE_Q6K_NATIVE    = 101,
    PT8_TYPE_Q5K_NATIVE    = 102,
    PT8_TYPE_Q8_0_SOA4     = 103,  // 4-row Q8 SoA4, 1.75 B/param   (Round 3 path)
    PT8_TYPE_Q4_0_SOA4     = 104,
};
```

Definitions are duplicated between
`torch/io/pt8_reader.h::torch::io::Pt8Type` and
`tools/gguf2pt8/converter.h::prometorch::convert::Pt8Type`. The numeric
values are the **stable contract** — adding a new value requires bumping
`PT8_VERSION`; renumbering an existing one is a hard compatibility break.

### Q8_SoA4 byte layout

`PT8_TYPE_Q8_0_SOA4` tensors store their data byte-identically to the
in-memory `cpu_quant::Q8SoA4` struct used by `torch/io/q8_soa_repack.h`:

> 176 bytes per super-row block (= 4 N-rows × 32 K-elems):
> - bytes  0..15  : 4 × fp32 d_w
> - bytes 16..31  : 4 × fp32 dmin_m
> - bytes 32..47  : 4 × i32 sum_q
> - bytes 48..175 : 128 bytes byte-interleaved int8 weights, 8 K-groups × 16 B

A full `[N, K]` weight is `gpr × bpr × 176` bytes where `gpr = N/4`,
`bpr = K/32`. Both `N % 4 == 0` and `K % 32 == 0` are required.

When the runtime sees `PT8_TYPE_Q8_0_SOA4`, it skips the on-load Q4_K →
Q8 SoA4 repack — saving the ~7 s repack step measured in Round 3 — and
points `cpu_quant::Q8SoA4::mem` directly into the mmap'd region.

### Q4_SoA4 (reserved, hot dtype for 30 tok/s target)

`PT8_TYPE_Q4K_SOA4 = 100` is reserved for the format described in spec
§10 (88 bytes per 4-row × 32-K block, 0.6875 B/param). This is the
recommended hot dtype for shipping qwen3:4b at 30 tok/s — Agent D's kernel
work is the gating dependency, not the format.

The current PT8Reader recognises the dtype tag and exposes it through
`tensor_data()` / `pt8_is_q4_soa4_layout()`; the consuming kernel
(`q4_soa4_gemv`) is Agent D's deliverable.

---

## Loader integration in `gguf_model.h`

`GGUFModel::load(path)` auto-detects format by reading the magic at offset
0. The detection is gated on `PT_FORMAT_AUTO=1` (default ON; set to `0` to
force the GGUF path even on a `.pt8` for debug A/B testing).

```cpp
if (PT_FORMAT_AUTO != 0 && PT8Reader::is_pt8_file(path)) {
    load_pt8(path);   // new path — zero-copy mmap, no repack
} else {
    /* existing GGUF path */
}
```

`load_pt8()` performs:

1. `PT8Reader::open()` — single mmap, header parse, tail-table walk.
2. Read model config + tokenizer from a sibling `.gguf` file (until the
   `.pt8` metadata KV section ships in v1.1; current writer omits it).
3. FP weights (norms, embd, biases) — copy into owned `at::Tensor`s
   (small total: < 100 MB for qwen3:4b — copy cost negligible).
4. Quantized weights — `QuantizedWeight::cpu_data` points zero-copy into
   the mmap. `QuantizedWeight::quant_type` is set to:
   - `12` (Q4_K) / `14` (Q6_K) / `8` (Q8_0) for passthrough variants
     → existing GGUF dispatch in `forward_decode_cpu_tp` Just Works
   - `0x100 | <Pt8Type>` for SoA4-native variants → `init_tensor_parallel`
     branches on this to skip the Q4_K → Q8 SoA4 repack
5. NUMA replication identical to GGUF path (`PT_NUMA_REPLICATE=1`).

`init_tensor_parallel()` is updated in exactly one location: the Q8 SoA4
repack block now treats `quant_type == 0x100 | PT8_TYPE_Q8_0_SOA4` as a
no-op repack — the sliced bytes are already in the runtime layout, the
function only binds `tl.q8_soa.mem`/N/K/group_stride to the slice's
buffer.

`forward_decode_cpu_tp()` is **unchanged**. It already dispatches via
`tl.q8_soa.valid` / Q4_K / Q6_K branches; PT8 native populates the
`q8_soa` view without touching forward.

---

## Memory savings on PT8 SoA4 cold start

Round 3 GGUF + repack path:

```
mmap GGUF Q4_K (2.4 GB)  +  malloc Q8 SoA4 (3.2 GB)  ≈  5.6 GB resident
                                 ↑ repack ~7 s
```

PT8 SoA4 path:

```
mmap PT8 Q8_SoA4 (3.2 GB)
                     ↑ no repack, no malloc, no per-layer slice copy when
                       slice grain is super-row aligned (head_dim ≥ 4)
```

For the recommended `PT_NUMA_REPLICATE=1` deployment, NUMA replication
still copies — but copies the *already-repacked* bytes, so each rank
spends ~3 s/node instead of 3 s + 7 s.

---

## Backwards compatibility

`.gguf` files continue to work exactly as before — the dispatch only
diverges at `load()` based on the first 8 bytes. No breaking change to
forward code or to existing `init_tensor_parallel` behaviour for GGUF
inputs.

---

## Open questions for v1.1

1. **Metadata KV section** — Agent A's spec §5 asks for one. Agent B's
   writer does not yet emit it; the current loader recovers config from
   a sibling `.gguf`. Adding the section closes the circle so
   `prometorch run model.pt8` works standalone. Only needs ~50 LoC in
   the writer + 30 in the reader.

2. **Q4 SoA4 hot path** — agent_D_kernel_optimization.md is the gate.
   Once `q4_soa4_gemv` lands, switching the converter default from
   `q8soa` to `q4soa` requires zero loader changes (the dtype tag flips,
   the dispatch in `init_tensor_parallel` already keys on it).

3. **CRC footer** — spec §1 reserves 32 bytes of footer for CRC32 of
   header + table + data. Useful for catching truncated downloads. Wire
   in when we ship the CLI (Agent B follow-up).
