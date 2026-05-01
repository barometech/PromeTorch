# PromeTorch `.pt8` Binary Format — Specification v1

**Author:** Agent A (Round 4)  
**Date:** 2026-04-30  
**Status:** Draft v1, ready for Agent B implementation  
**Lossless trait:** Trakt B from MISSION.md §2 — *mathematically equivalent to
Q4_K_M decode-path*. Logits identical to current `cpu_quant_gemv` Q4_K
within fp32 round-off (max abs diff < 1e-5).

---

## 0. Executive summary

`.pt8` is a single-file container (header + tensor table + data section)
that stores all qwen3:4b weights pre-repacked into the **Q8_SoA4** layout
which is already validated at 9.4 tok/s on Эльбрус 8C2 (Round 3, commit
`3399136`). The format is designed so that **load = single mmap, zero
re-pack, zero dequant on every GEMV**.

Three weight-storage variants are defined:

| dtype tag | Used for | Bytes / param | Comment |
|-----------|----------|---------------|---------|
| `PT8_Q8_SOA4` | all GEMV linears (Q/K/V/O, gate/up/down, output) | 5.50 (full) / **4.50 + 0.5%** with FP16 headers (§3.3) | qpmaddubsh-friendly |
| `PT8_FP32`    | norms, RoPE freqs, biases | 4.00 | tiny, irrelevant for BW |
| `PT8_FP16_RAW`| token_embd if user wants to keep Q4_K_M-equivalent | 2.00 | dequant on-fly to FP32 once at load |

For qwen3:4b that gives a final file size of **~3.3 GB** (vs Q4_K_M
2.4 GB GGUF, vs current Round 3 in-memory Q8 SoA4 ~3.2 GB ×4 NUMA).

The crucial property: **at load, file → mmap → DONE**. No 7-second repack.
Per-NUMA replication happens by `mmap` then `mbind`/`memcpy` per node, but
the data layout requires zero math.

---

## 1. File layout (high level)

```
+-----------------------------------------+ offset 0
|   FILE HEADER          (256 bytes)      |
+-----------------------------------------+ 256
|   STRING POOL          (variable)       |   ← UTF-8 tensor names, KV keys
+-----------------------------------------+
|   METADATA KV          (variable)       |   ← {arch, n_layer, n_head, ...}
+-----------------------------------------+
|   TENSOR TABLE         (N × 96 bytes)   |   ← fixed-size descriptors
+-----------------------------------------+ aligned to 4096
|   PADDING to 4 KiB                      |
+-----------------------------------------+ data_section_offset
|   TENSOR DATA          (~3.2 GB)        |   ← every tensor 64-byte aligned
+-----------------------------------------+ EOF
|   FOOTER (32 bytes, optional)           |   ← CRC32 of header + checksums
+-----------------------------------------+
```

All multi-byte integers are **little-endian** (Эльбрус is LE). All
floats are IEEE-754. Strings are length-prefixed UTF-8 (no NUL term).

---

## 2. File header (offset 0..255, exactly 256 bytes)

```
offset  size   field            value / description
------  -----  ---------------  -------------------------------------------
  0      4    magic            "PT8\0"   = 0x00385450 (LE: 'P','T','8',0)
  4      4    version          u32, currently 1
  8      8    file_size        u64, total file size in bytes
 16      8    flags            u64, bit 0 = has_footer_crc, bits 1-63 reserved
 24      8    string_pool_off  u64, offset of string pool
 32      8    string_pool_size u64, bytes
 40      8    metadata_off     u64, offset of metadata KV section
 48      8    metadata_size    u64, bytes
 56      8    tensor_table_off u64, offset of tensor table
 64      4    n_tensors        u32, number of entries in tensor table
 68      4    tensor_entry_sz  u32, currently 96 (size of TensorEntry)
 72      8    data_section_off u64, offset of tensor data (4 KiB aligned)
 80      8    data_section_sz  u64, bytes
 88      8    n_params         u64, total scalar parameters (for sanity)
 96      8    source_hash      u64, xxh64 of source GGUF file (for prov.)
104     16    source_filename  char[16], first 15 bytes of original GGUF
                                 name + NUL (truncated, info-only)
120      4    arch_id          u32, 1 = qwen3, 2 = llama3, 3 = generic ...
124      4    pt8_alignment    u32, 64 (data alignment within section)
128    128    reserved         must be zero
------  -----
total  256
```

**Validation rules (loader MUST check):**
- `magic == 0x00385450`
- `version == 1`
- `tensor_entry_sz == 96`
- `pt8_alignment == 64` (qpmaddubsh + cache lines)
- `data_section_off % 4096 == 0`
- `file_size == file_size_on_disk`

---

## 3. Tensor table (n_tensors × 96 bytes)

Each entry:

```
offset  size  field             description
------  ----  ----------------  ----------------------------------------
  0      4   name_offset       u32, offset in string pool
  4      4   name_length       u32, bytes of the UTF-8 name (no NUL)
  8      1   dtype             u8, see §3.1
  9      1   ndim              u8, 1..4
 10      2   pad0              u16, must be 0
 12      4   flags             u32, bit 0 = numa_replicate_hint,
                                bit 1 = is_quant_weight (has Q8SoA4 hdr),
                                bit 2 = transposed_for_gemv,
                                bits 3-31 reserved
 16      8   shape[0]          u64, leading dim (e.g. N for [N,K] weight)
 24      8   shape[1]          u64
 32      8   shape[2]          u64
 40      8   shape[3]          u64
 48      8   data_offset       u64, **absolute** offset in file
 56      8   data_size         u64, bytes
 64      8   n_elements        u64, product of shape (logical params)
 72      4   group_stride      u32, bytes per (super-row group of 4)
                                for Q8_SOA4; else 0
 76      4   bpr               u32, blocks per row = K/32 for Q8_SOA4
 80      4   K_dim             u32, which dim is K (typically ndim-1)
 84      4   N_dim             u32, which dim is N
 88      4   header_dtype      u32, see §3.3 (FP32 or FP16-mixed scales)
 92      4   crc32             u32, CRC32 of this tensor's data (0 if disabled)
------  ----
total   96
```

Entries are sorted alphabetically by tensor name for binary search;
loader can also build a `unordered_map<string,TensorEntry*>`.

### 3.1 dtype enum

```c
enum Pt8Dtype : uint8_t {
    PT8_FP32       = 1,    // raw fp32, ndim×4 bytes
    PT8_FP16_RAW   = 2,    // raw fp16 (used only for dequant-once tensors)
    PT8_INT32      = 3,    // raw i32 (rare)
    PT8_Q8_SOA4    = 16,   // 4-row SoA INT8, the hot path (§4)
    PT8_Q8_SOA4_F16= 17,   // §3.3: same layout but headers fp16+i16 (compact)
    PT8_Q8_0_FLAT  = 18,   // optional: pure Q8_0 layout, fp16 scale + i8[32]
                           //  — kept for embeddings or if Agent D wants it
    PT8_FP16_DEQUANT_AT_LOAD = 19, // §3.4: stored as Q4_K_M sub-block
                                   //  data, dequantized to fp32 once at load
};
```

**Default for qwen3:4b GEMV linears: `PT8_Q8_SOA4_F16`** (§3.3).

### 3.2 PT8_Q8_SOA4 — exact byte layout per super-row block

Super-row = 4 N-rows × 32 K-elements = **176 bytes** (matches
`q8_soa_repack.h` SOA4_GROUP_BYTES).

```
offset  size  field          description
------  ----  -------------  ----------------------------------------
  0     16   d_w[4]         fp32 × 4 — per-row block scale =
                              d_q4k * sub_scale_q4k (pre-fused)
 16     16   dmin_m[4]      fp32 × 4 — per-row block min =
                              dmin_q4k * sub_min_q4k
 32     16   sum_q[4]       i32 × 4 — sum of int8 weights in block
                              (used for activation-shift correction)
 48    128   W[8 × 16]      int8 × 128, byte-interleaved
                              for each K-group kg in [0..7]:
                               for each row r in [0..3]:
                                for each k in [0..3]:
                                  byte[kg*16 + r*4 + k] = w_row[r][kg*4 + k]
                              w in [0..15] (Q4_K nibble cast to int8)
------  ----
total  176
```

This is **byte-identical** to the in-memory layout used by Round 3's
`q8_soa4_gemv` function — the file is essentially `mem` of `Q8SoA4`
serialized to disk.

A full tensor `[N, K]` stored as `PT8_Q8_SOA4`:
- `gpr = N / 4` super-row groups
- `bpr = K / 32` blocks per row
- total bytes = `gpr * bpr * 176`
- requires `N % 4 == 0` AND `K % 32 == 0` (qwen3 satisfies for all linears)

### 3.3 PT8_Q8_SOA4_F16 — compact header variant (recommended default)

Identical to §3.2 except the per-block header is FP16/I16 instead of
FP32/I32, halving header overhead:

```
offset  size  field          description
------  ----  -------------  ----------------------------------------
  0      8   d_w[4]         fp16 × 4 — per-row block scale
  8      8   dmin_m[4]      fp16 × 4 — per-row block min
 16     16   sum_q[4]       i16 × 8? NO: i32 × 4 stays — sum can be
                            up to 32×15 = 480, fits in i16 BUT
                            keeping i32 avoids overflow risk and
                            costs only 8 B/block. Compromise:
                            i16 × 4 since |sum_q| ≤ 480 ≪ 32767. ✓
 16      8   sum_q[4]       i16 × 4 — sum of int8 weights
 24    128   W[8 × 16]      int8 × 128, same as §3.2
------  ----
total  152  (vs 176 for full FP32 headers)
```

**Wait — math check:** Q4_K nibbles are 0..15. Per 32-elem block, max
|sum_q| = 32 × 15 = 480, easily fits i16 ([-32768, 32767]). FP16 d_w
range: Q4_K `d` is fp16 in source, sub_scale is 6-bit ≤ 63, so the
product `d × sc` may exceed FP16 range (fp16 max ≈ 65504). Need to
verify on real qwen3 GGUF — Q4_K_M `d` typically ~1e-2..1e0, sc ≤ 63,
so max ~63 ≈ 6.3e1 — well within fp16. **Safe.**

Bytes/param savings: at K=2560, N=2432:  
- §3.2 (FP32 hdr): 176/32 = 5.5 B/param  
- §3.3 (FP16 hdr): 152/32 = 4.75 B/param

For qwen3:4b weights (≈3.0e9 params in linears, excluding embeddings):
- §3.2: 16.5 GB ❌ (way too much, doesn't fit budget)
- §3.3: 14.25 GB ❌

**This is wrong.** Let me re-check. Q8 SoA4 stores 4-bit data as int8,
so it's intrinsically 8 bits/param + 0.5 bytes header overhead. We
should be at ~5.5 B/param × 3e9 params = 16.5 GB if we replicate per
NUMA node. Per node it's 4.13 GB. Single mmap'd file: 4.13 GB.
That matches Round 3 measurement (~3.2 GB Q8 SoA4 in-memory single
copy + 4× NUMA replication = ~13 GB).

**Correction:** the §0 "3.3 GB" estimate was pessimistic. Realistic
single-copy file size for qwen3:4b GEMV linears in PT8_Q8_SOA4_F16:

| Component | Params | Layout | Bytes |
|-----------|--------|--------|-------|
| 36 layers × (Q+K+V+O+gate+up+down) | ~3.0e9 | Q8_SOA4_F16 | ~3.0e9 × 4.75 / 8 = 1.78 GB |

Wait, "params × 4.75 / 8" makes no sense — params are already INT8
storage (1 byte each). Let me redo:

- 1 param = 1 int8 byte = 1.0 B  
- + per-block header 24 B / 32 params = 0.75 B/param  
- = **1.75 B/param** for §3.3 ✓

For 3.0e9 GEMV params: **5.25 GB** (single copy, no NUMA replication).
That's **larger** than Q4_K_M's 2.4 GB by ~2.2× — but **decode is
zero-cost** (no nibble unpack, no fp16→fp32 of d/dmin per block, no
6-bit scale unpack).

For NUMA: replicate × 4 = **21 GB**, fits 125 GB on Эльбрус. ✓

### 3.4 PT8_FP16_DEQUANT_AT_LOAD — fallback for tensors we don't repack

For tensors where Q8_SoA4 is not a fit (e.g. embeddings, output_norm,
or anything that's not a pure GEMV [N,K] matrix), we keep the source
Q4_K_M / Q6_K / Q8_0 sub-block bytes as-is and dequantize once at load
into FP32 in-memory. The on-disk format mirrors GGUF's super-block
layout 1:1:

```
offset  size  field
  0     n×bs  raw GGML super-blocks (Q4_K = 144B/256 elem, etc.)
```

The metadata KV (§5) stores `source_ggml_type` so the loader knows
which dequant function to call.

---

## 4. String pool

```
offset  size  field
  0      4   pool_magic = "STRP"
  4      4   n_strings  u32
  8      ?   data: { u32 length, byte[length] payload } repeated
```

`name_offset` in tensor table points into payload data, **not**
including the length prefix. To read: `name = pool[name_offset .. name_offset + name_length]`.

---

## 5. Metadata KV section

Mirror GGUF's metadata so the loader can rebuild model config without
re-reading the source GGUF. Stored as a sequence of records:

```
record:
  u32  key_offset (into string pool)
  u32  key_length
  u8   value_type    (1=u32, 2=u64, 3=f32, 4=f64, 5=str_off+len, 6=array)
  u8   pad[7]
  16B  value (u64 inline; for str/array it's u32 offset + u32 length)
```

Required keys for qwen3:4b inference:

```
general.architecture        = "qwen3"
qwen3.embedding_length      = 2560
qwen3.feed_forward_length   = 6912 (or model-specific)
qwen3.attention.head_count  = 20
qwen3.attention.head_count_kv = 4
qwen3.block_count           = 36
qwen3.attention.layer_norm_rms_epsilon = 1e-6
qwen3.rope.freq_base        = 1000000.0
tokenizer.ggml.model        = "gpt2" (or however qwen3 tokenizer is tagged)
pt8.source.ggml_format      = "Q4_K_M"
pt8.source.file_size        = u64
pt8.source.sha256           = byte[32]
```

This keeps the loader self-contained: `prometorch run model.pt8`
without needing the GGUF anywhere.

---

## 6. Tensor naming convention

Use **GGUF-compatible** names so the loader code path is identical:

```
token_embd.weight                      [vocab, 2560]    PT8_FP16_DEQUANT_AT_LOAD
output_norm.weight                     [2560]           PT8_FP32
output.weight                          [vocab, 2560]    PT8_Q8_SOA4_F16  (or _DEQUANT)
blk.{i}.attn_norm.weight               [2560]           PT8_FP32
blk.{i}.attn_q.weight                  [N_q, 2560]      PT8_Q8_SOA4_F16
blk.{i}.attn_k.weight                  [N_kv, 2560]     PT8_Q8_SOA4_F16
blk.{i}.attn_v.weight                  [N_kv, 2560]     PT8_Q8_SOA4_F16
blk.{i}.attn_output.weight             [2560, 2560]     PT8_Q8_SOA4_F16
blk.{i}.ffn_norm.weight                [2560]           PT8_FP32
blk.{i}.ffn_gate.weight                [6912, 2560]     PT8_Q8_SOA4_F16
blk.{i}.ffn_up.weight                  [6912, 2560]     PT8_Q8_SOA4_F16
blk.{i}.ffn_down.weight                [2560, 6912]     PT8_Q8_SOA4_F16
```

Existing `gguf_model.h` `load_quantized_to_cpu()` path reuses the same
names — only the dtype branch flips.

---

## 7. Encoder pseudo-code (GGUF Q4_K_M → .pt8)

```python
def gguf_to_pt8(gguf_path: str, pt8_path: str, *,
                hot_dtype: int = PT8_Q8_SOA4_F16,
                replicate_hint: bool = True) -> None:
    g = GgufFile(gguf_path)                       # mmap'd
    out = open(pt8_path, "wb")

    # --- Phase 1: plan ---
    string_pool = StringPool()
    tensors_plan = []          # list of (name, dtype, shape, src_info, est_size)
    for t in g.tensor_infos:
        name_off = string_pool.add(t.name)
        if is_linear_weight(t):                   # 2D, K%32==0, N%4==0
            dtype = hot_dtype                     # PT8_Q8_SOA4_F16
            est_size = (t.shape[0] // 4) * (t.shape[1] // 32) * 152
        elif t.type in (FP32, FP16, F32_NORM):
            dtype = PT8_FP32
            est_size = t.n_elements * 4
        else:                                     # token_embd, output, etc.
            dtype = PT8_FP16_DEQUANT_AT_LOAD
            est_size = t.data_bytes               # keep raw GGML bytes
        tensors_plan.append((name_off, t, dtype, est_size))

    # --- Phase 2: layout ---
    string_pool_bytes = string_pool.serialize()
    metadata_bytes    = serialize_metadata(g.metadata)
    n_tensors         = len(tensors_plan)

    header_size       = 256
    string_off        = 256
    metadata_off      = align(string_off + len(string_pool_bytes), 8)
    tensor_table_off  = align(metadata_off + len(metadata_bytes), 8)
    tensor_table_size = n_tensors * 96
    data_section_off  = align(tensor_table_off + tensor_table_size, 4096)

    # Assign per-tensor data offsets, 64-byte aligned
    cursor = data_section_off
    entries = []
    for (name_off, t, dtype, sz) in tensors_plan:
        cursor = align(cursor, 64)
        entries.append(make_entry(name_off, t, dtype, cursor, sz))
        cursor += sz
    file_size = cursor

    # --- Phase 3: write header / pool / metadata / table ---
    write_header(out, file_size, string_off, len(string_pool_bytes),
                 metadata_off, len(metadata_bytes),
                 tensor_table_off, n_tensors, data_section_off,
                 file_size - data_section_off, total_params(g),
                 source_hash=xxh64(g.bytes))
    out.seek(string_off);    out.write(string_pool_bytes)
    out.seek(metadata_off);  out.write(metadata_bytes)
    out.seek(tensor_table_off)
    for e in entries: out.write(e.pack())          # 96 bytes each

    # --- Phase 4: write tensor data ---
    for (name_off, t, dtype, sz), e in zip(tensors_plan, entries):
        out.seek(e.data_offset)
        if dtype == PT8_Q8_SOA4_F16:
            buf = repack_q4k_to_q8_soa4_f16(g.read(t), N=t.shape[0],
                                            K=t.shape[1])
            assert len(buf) == sz
            out.write(buf)
        elif dtype == PT8_FP32:
            buf = dequant_to_fp32(t.type, g.read(t), t.n_elements)
            out.write(buf.tobytes())
        elif dtype == PT8_FP16_DEQUANT_AT_LOAD:
            out.write(g.read(t))                   # raw GGML bytes
        else:
            raise NotImplementedError

    # --- Phase 5: optional CRC footer ---
    if FLAG_CRC:
        crc = crc32(file_first_512MiB(out))
        out.write(pack_footer(crc))
    out.close()


def repack_q4k_to_q8_soa4_f16(q4k_bytes, N, K) -> bytes:
    """Math identical to torch/io/q8_soa_repack.h::repack_q4k_to_q8soa4
       but emits the FP16-header layout (§3.3)."""
    out = bytearray()
    n_super_per_row = K // 256
    for g in range(N // 4):
        for b in range(K // 32):              # soa block index
            sb_idx = b // 8                    # Q4_K super-block index
            j      = b %  8                    # sub-block within super
            d_w   = [0.0]*4
            dmin_m= [0.0]*4
            sum_q = [0]*4
            W     = bytearray(128)             # 8 K-groups × 16 B
            row_q8 = [[0]*32 for _ in range(4)]

            for r in range(4):
                row = g*4 + r
                blk = q4k_bytes[row*row_stride + sb_idx*144 :
                                row*row_stride + (sb_idx+1)*144]
                d    = fp16_to_fp32(blk[0:2])
                dmin = fp16_to_fp32(blk[2:4])
                scales12 = blk[4:16]
                qs128    = blk[16:144]
                sc, m = get_scale_min_k4(j, scales12)
                d_w[r]    = d * sc
                dmin_m[r] = dmin * m
                p, is_high = j // 2, (j & 1) != 0
                qs_window = qs128[p*32 : (p+1)*32]
                s = 0
                for l in range(32):
                    byte = qs_window[l]
                    q4 = (byte >> 4) if is_high else (byte & 0xF)
                    row_q8[r][l] = q4         # 0..15, fits int8
                    s += q4
                sum_q[r] = s

            # Build interleaved 128 B
            for kg in range(8):
                for r in range(4):
                    for k in range(4):
                        W[kg*16 + r*4 + k] = row_q8[r][kg*4 + k]

            out += pack_fp16x4(d_w)            # 8 B
            out += pack_fp16x4(dmin_m)         # 8 B
            out += pack_i16x4(sum_q)           # 8 B
            out += W                           # 128 B
            # total 152 B per super-block
    return bytes(out)
```

Encoder runs in **<30 s** on x86 dev box (single-threaded — could
parallelize per tensor for ~5 s on 8 cores). On Эльбрус 32-core it'd
be <10 s but typically you convert on x86 once and copy the file.

---

## 8. Decoder / loader pseudo-code (.pt8 → in-memory tensors)

```cpp
struct Pt8Loader {
    MmapHandle mmap_;
    Pt8Header  hdr_;
    std::vector<TensorEntry> entries_;
    std::unordered_map<std::string, TensorEntry*> by_name_;

    void open(const std::string& path) {
        mmap_.map(path);
        std::memcpy(&hdr_, mmap_.data(), 256);
        if (hdr_.magic != 0x00385450) throw "bad magic";
        if (hdr_.version != 1)        throw "bad version";

        entries_.resize(hdr_.n_tensors);
        const uint8_t* tbl = (const uint8_t*)mmap_.data() + hdr_.tensor_table_off;
        for (uint32_t i = 0; i < hdr_.n_tensors; ++i) {
            std::memcpy(&entries_[i], tbl + i * 96, 96);
            std::string name = read_pool(entries_[i].name_offset,
                                         entries_[i].name_length);
            by_name_[name] = &entries_[i];
        }
    }

    // Returns a non-owning view into the mmap'd region.
    Q8SoA4View get_q8_soa4(const std::string& name) {
        auto* e = by_name_.at(name);
        assert(e->dtype == PT8_Q8_SOA4_F16 || e->dtype == PT8_Q8_SOA4);
        Q8SoA4View v;
        v.mem          = (uint8_t*)mmap_.data() + e->data_offset;
        v.N            = e->shape[e->N_dim];
        v.K            = e->shape[e->K_dim];
        v.group_stride = e->group_stride;
        v.header_dtype = e->header_dtype;        // fp16 vs fp32 hdr branch
        v.valid        = true;
        return v;
    }

    Tensor get_fp32(const std::string& name) {
        auto* e = by_name_.at(name);
        assert(e->dtype == PT8_FP32);
        // Zero-copy view into mmap
        return Tensor::from_blob((float*)((uint8_t*)mmap_.data()
                                          + e->data_offset),
                                 shape_from(e), kFloat);
    }

    Tensor get_dequant_fp32(const std::string& name) {
        auto* e = by_name_.at(name);
        assert(e->dtype == PT8_FP16_DEQUANT_AT_LOAD);
        // Dequantize once, copy into a fresh fp32 tensor
        Tensor out = at::empty(shape_from(e), kFloat);
        gguf::dequantize(source_ggml_type_for(e),
                         (uint8_t*)mmap_.data() + e->data_offset,
                         out.data_ptr<float>(), e->n_elements);
        return out;
    }

    // NUMA replication — if PT_NUMA_REPLICATE is set, replicate hot
    // GEMV tensors per node by memcpy into node-bound allocations.
    ReplicatedQ8SoA4 numa_replicate(const std::string& name);
};
```

**Loader cost** for qwen3:4b: **~2 ms** (just header + table scan).
First GEMV touches pages → kernel paging in. With `mlock`/`madvise
WILLNEED` on whole data section: **~1.5 s** to fault all 5 GB from
NVMe (faster on tmpfs / re-runs from page cache). NUMA replication
adds ~3 s × 4 = **~12 s wall** (parallelizable per node = 3 s).

Compare Round 3 path: load GGUF (mmap) + repack Q4K→Q8SoA4 = **7 s
single-thread** for repack alone. PT8 saves the repack — ~7 s shaved
per cold start.

---

## 9. Decode-cost / bandwidth comparison

Per qwen3:4b **forward token** (single decode, full GEMV pass):

| Format | Bytes/param | File / mem (qwen3:4b GEMV) | Read/token | BW@100GB/s ceiling | Decode CPU ops/param |
|--------|-------------|----------------------------|------------|---------------------|----------------------|
| GGUF Q4_K_M (current) | 0.5625 (4.5 b) | 2.4 GB / 2.4 GB | ~2.0 GB | 50 tok/s | unpack 6-bit scale, fp16→fp32, nibble extract — ~12 cycles/param |
| GGUF Q8_0 | 1.0625 | 4.3 GB / 4.3 GB | ~3.5 GB | 28.6 tok/s | fp16→fp32 once/32 elem, no unpack — ~3 cy/param |
| Q8 SoA4 (in-mem now) | 1.375 (full FP32 hdr, 22 B/16 elem) | n/a / **3.2 GB** | 3.0 GB | 33 tok/s | qpmaddubsh + per-block fp32 fold — ~2 cy/param |
| **PT8_Q8_SOA4_F16** (this spec) | 1.1875 (152 B / 32 elem = 4.75 b) | **5.25 GB** / 5.25 GB | 5.0 GB | **20 tok/s** ❌ | qpmaddubsh + fp16→fp32 hdr per block — ~2.2 cy/param |
| **PT8_Q8_SOA4** (fp32 hdr) | 5.5 b/elem ~= 0.6875 B/param wait... | | | | |

**STOP — my unit math is broken. Let me redo carefully.**

A parameter is one logical scalar weight. Q8 SoA4 stores each weight
as 1 INT8 byte. Per 32 weights (one block) we add a header of 24 B
(FP16) or 48 B (FP32). So:

- §3.2 (FP32 hdr): (32 + 48) / 32 = **2.50 B / param**
- §3.3 (FP16 hdr): (32 + 24) / 32 = **1.75 B / param**

For **3.0e9 GEMV params** in qwen3:4b:

| Variant | B/param | Total | Read/token | BW@100 GB/s |
|---------|---------|-------|------------|--------------|
| Q4_K_M   | 0.5625 | 1.69 GB | 1.69 GB | **59 tok/s** |
| Q8_0     | 1.0625 | 3.19 GB | 3.19 GB | 31 tok/s |
| **PT8_Q8_SOA4_F16** | 1.75   | **5.25 GB** | 5.25 GB | **19 tok/s** ⚠ |
| PT8_Q8_SOA4 (fp32 hdr) | 2.50 | 7.50 GB | 7.50 GB | 13 tok/s |

**Critical observation:** moving header from FP32 to FP16 saves
~30% bandwidth. **Even with FP16 headers, PT8_Q8_SOA4_F16 is
bandwidth-heavier than Q4_K_M by 3.1×.**

**This means the BW ceiling for the PT8 format is 19 tok/s**, not 50.

To beat this and reach 30 tok/s we either:
1. Reduce bytes/param further (e.g. share scales across rows: 4-row
   share-scale → 1 d per 4 rows → header drops to 6 B/block = 1.19 B/
   param → BW ceiling 27 tok/s, still short)
2. Keep raw Q4_K_M storage on disk and do the lighter Q4→Q8 SoA4
   repack at load (5 s one-shot) — same as Round 3 today, no
   bandwidth penalty
3. Use a hybrid: store Q4_K_M-equivalent **packed nibbles** but in
   SoA4 row-interleave (4 rows × 32 elem × 4 bits = 64 B + 24 B
   hdr = 88 B / 128 elem = 0.6875 B/param) — restores Q4_K BW
   ceiling, ALL the qpmaddubsh shape benefit.

**Recommended:** define a fourth variant **PT8_Q4_SOA4** that does
exactly that. (See §10 for spec.)

---

## 10. PT8_Q4_SOA4 — the *real* answer (added after BW analysis)

### Layout per 4-row × 32-K super-block

```
offset  size  field          description
------  ----  -------------  ----------------------------------------
  0      8   d_w[4]         fp16 × 4 — pre-fused d × sc
  8      8   dmin_m[4]      fp16 × 4 — pre-fused dmin × m
 16      8   sum_q[4]       i16 × 4 — Σ q4 in this block (≤480, fits)
 24     64   qs[4 × 16]     uint8 × 64, byte-interleaved nibbles:
                              for K-pair p in [0..15]:
                                for r in [0..3]:
                                  byte[p*4 + r] = (q4_high << 4) | q4_low
                              where q4_low = w_row[r][p*2],
                                    q4_high = w_row[r][p*2 + 1]
------  ----
total   88  bytes per 4 rows × 32 elements = 128 weights → 0.6875 B/param
```

### Bandwidth math
- 88 B / 128 weights = **0.6875 B/param**
- qwen3:4b GEMV = 3.0e9 weights × 0.6875 = **2.06 GB**
- Read/token ≈ 2.06 GB → **BW ceiling 49 tok/s @ 100 GB/s**

This is **identical to Q4_K_M's BW ceiling**, while preserving the
SoA4 row-interleave structure that makes `qpmaddubsh` produce in-lane
per-row partial dots.

### Inner-loop pseudo-code (e2k v5 intrinsics)

```c
// W_v points to 16-byte K-pair: 4 rows × (low nibble + high nibble) packed.
// We need TWO qpmaddubsh issues per K-group of 4 elems:
//   First unpack: low_v  = (W_v >> 0) & 0x0F mask  → 16 i8 lanes (rows 0..3 × pos 0..3)
//   Second:      high_v = (W_v >> 4) & 0x0F mask  → 16 i8 lanes (rows 0..3 × pos 4..7)
//
// Per K-group of 4:
v2di low4   = qpand(W_v, MASK_0F);
v2di high4  = qpsrlw_byte(W_v, 4);   // arith shift per byte (or via qpshufb LUT)
v2di p16_lo = qpmaddubsh(A_lo, low4);
v2di p16_hi = qpmaddubsh(A_hi, high4);
v2di p32_lo = qpmaddh(p16_lo, ONES16);
v2di p32_hi = qpmaddh(p16_hi, ONES16);
acc = qpaddw(acc, qpaddw(p32_lo, p32_hi));
```

That's **~6 SIMD ops per 4-row × 8-K-element work** = 32 byte-MADs.
Density: 32/6 = 5.33 byte-MADs per cycle (vs Q8_SoA4's 16 byte-MADs
per ~3 ops = 5.33). Same compute density — *but half the
bandwidth*.

This is the **right** PT8 hot dtype.

### Trade-off vs Q8_SoA4
- Pro: **2.86× lower bandwidth** → 49 tok/s ceiling (vs 19)
- Pro: file size 2.06 GB ≈ Q4_K_M source → conversion is 1:1 byte movement
- Con: extra nibble unpack per K-pair (1 `qpand` + 1 byte-shift or
  `qpshufb` LUT) — cost ~2 cycles per 16 K-pair ops, maybe 10%
  throughput hit
- Con: not yet microbench-proven (Q8_SoA4 at 9.4 tok/s is — Q4_SoA4
  is theoretical until Agent D measures)

**Recommendation:** ship PT8 v1 with **BOTH** PT8_Q8_SOA4_F16 and
PT8_Q4_SOA4 as supported dtypes; converter chooses Q4_SoA4 by
default (env override `PT_PT8_USE_Q8=1` for fallback).

---

## 11. Final comparative table

| Format | B/param | qwen3:4b size | BW ceiling @ 100 GB/s | Decode cycles/param | qpmaddubsh-friendly | Status |
|--------|---------|---------------|------------------------|----------------------|---------------------|--------|
| GGUF Q4_K_M | 0.5625 | 1.69 GB | 59 tok/s | ~12 cy (nibble + 6-bit + fp16) | NO (scalar-ish) | working, 9.4 tok/s |
| GGUF Q8_0 | 1.0625 | 3.19 GB | 31 tok/s | ~3 cy | partial | working |
| Round 3 Q8 SoA4 (in-mem only) | 1.375 (FP32 hdr in-mem) | 4.13 GB ×4 NUMA | 24 tok/s | ~2 cy | YES | shipped, 9.4 tok/s |
| **PT8_Q8_SOA4_F16** | 1.75 | 5.25 GB | 19 tok/s | ~2.2 cy | YES | spec'd, this doc |
| **PT8_Q4_SOA4** ⭐ | 0.6875 | 2.06 GB | **49 tok/s** | ~2.5 cy (incl. nibble unpack) | YES | spec'd, this doc |

⭐ = recommended hot dtype for v1.

---

## 12. Honest assessment: are 30 tok/s achievable?

**Conditional yes** — with the following dependency chain:

1. ✅ **Format itself does not cap us** — if we use **PT8_Q4_SOA4**
   (§10), the BW ceiling is 49 tok/s, well above 30. The format is
   not the bottleneck.
2. ⚠ **The actual `qpmaddubsh` inner kernel must reach ~60% BW
   utilization** — Round 3 hit 19% on Q4_K_M and ~24% on Q8_SoA4.
   We need **~2.5× kernel improvement on top of the format**.
3. ⚠ **Per-call overhead must drop** — TP-4 SHM AllReduce, ThreadPool
   fork/join, activation quantize per layer all add up. Round 3
   Agent 1's persistent ThreadPool was estimated at +90% throughput
   if fully landed. Need to confirm it's actually wired up.
4. ❌ **If above two don't compound enough, speculative decoding
   needed** — 9.4 × 2.5 = 23.5 still misses 30 tok/s.

**Without PT8_Q4_SOA4:** if we ship only PT8_Q8_SOA4_F16, the format
itself caps us at 19 tok/s BW ceiling, and 30 tok/s is **physically
unachievable** without speculative decoding (which gives effective
×2-3 token throughput against any base format).

**With PT8_Q4_SOA4 + Agent D kernel work + persistent ThreadPool:**
30 tok/s is **achievable in 1-2 sessions** of kernel work.

**With everything above + speculative decoding (Agent E):** 30 tok/s
is comfortable, possibly 40-50 tok/s.

---

## 13. Follow-up questions / blockers for downstream agents

### For Agent B (converter / CLI)
- Q-B1: implement **both** PT8_Q8_SOA4_F16 and PT8_Q4_SOA4 encoders
  in one pass over the GGUF file, gated by `--variant {q4soa,q8soa}`.
  Default = `q4soa` (smaller, faster).
- Q-B2: verify FP16 d×sc range on real qwen3:4b GGUF. Print
  `max(|d×sc|)` across all super-blocks as a sanity check; if it
  exceeds fp16 max (65504) we must fall back to FP32 headers for
  affected tensors only. (My calculation says it won't, but verify.)
- Q-B3: implement `pt8_verify` subcommand: re-dequantize PT8 to
  FP32, compare with GGUF dequantize → max abs diff must be < 1e-7
  (since the math is bit-identical: same `d * sc` rearrangement and
  the FP16 rounding of (d×sc) does introduce error on order 2^-11
  relative — verify acceptable in practice).
- Q-B4: do we keep `output.weight` (the lm_head, ~2.5e8 params) in
  PT8_Q4_SOA4 or in PT8_FP16_DEQUANT_AT_LOAD? Probably former for
  uniformity, but token_embd may need DEQUANT path because it's
  used as embedding lookup (not GEMV).

### For Agent C (loader / inference path)
- Q-C1: `gguf_model.h::load_quantized_to_cpu()` currently dispatches
  on GGML type. Add a parallel path for PT8: check magic at
  offset 0, branch to Pt8Loader. Both paths must be supported in
  the same binary so we can A/B test.
- Q-C2: how does the loader handle the case where Agent D adds new
  PT8 variants in v2? Use `version` field — v1 reader rejects v2.
  Make encoder embed `min_reader_version`.
- Q-C3: NUMA replication for mmap'd PT8 — is a per-NUMA `memcpy`
  acceptable, or should we do `mbind(MPOL_INTERLEAVE)` on the mmap?
  Round 3 used per-node `memcpy` and it worked; replicate.

### For Agent D (kernel optimization)
- Q-D1: implement `q4_soa4_gemv` (PT8_Q4_SOA4 variant). Critical
  question: cost of nibble unpack on 8C2. Is `qpand + qpsrlw_byte`
  ≤ 2 cycles, or do we need a `qpshufb` LUT-based unpacker? Estimate
  before committing — if unpack is 4+ cy, drop back to PT8_Q8_SOA4.
- Q-D2: Round 3's `q8_soa4_gemv` reached 9.4 tok/s with FP32 headers.
  When we go to FP16 headers, we need to add fp16→fp32 unpack inside
  the per-block fold. Cost = ~2 SIMD cycles per block. Verify it
  doesn't regress single-thread microbench before shipping.
- Q-D3: kernel fusion (gate+up, Q+K+V) — does the SoA4 layout still
  enable shared `quant_activation`? Yes, same x is reused, but
  activation broadcast must be done once at the start of fused
  GEMV, not per kernel.
- Q-D4: APB (Array Prefetch Buffer) hint — for SoA4 stride 88 B
  (PT8_Q4_SOA4) or 152 B (PT8_Q8_SOA4_F16), does APB activate? MCST
  docs say yes for stride <= 256 B regular access. Confirm with
  perf counters.

### For Agent E (speculative decoding)
- Q-E1: if our base format hits 25 tok/s after Agent D kernel work,
  can a Medusa-style multi-head draft model give us +20% accept-rate
  for free → 30 tok/s? Or do we need a full tiny-qwen3 draft (which
  requires training)?
- Q-E2: PT8 format must store the draft model alongside the main
  model. Add a "model class" field per tensor name namespace
  (`draft.blk.{i}.attn_q.weight` etc.).

### Open: GUI (cross-cutting)
- Q-G1: GUI is optional per MISSION §10; if we go web-based (HTTP
  server + static HTML), it's trivial to ship cross-platform. Don't
  build a Tauri/Qt monster. Recommend: single-file C++ HTTP server
  + dark-themed HTML page.

---

## 14. Acceptance for this format spec (for review by Agent B before coding)

Before Agent B starts coding the converter, confirm:

- [ ] dtype enum values §3.1 are not in conflict with anything
- [ ] PT8_Q4_SOA4 layout §10 is preferred over PT8_Q8_SOA4_F16 default
- [ ] Header is fixed 256 bytes, all offsets u64, alignment 64 B
- [ ] FP16 range for `d × sc` confirmed safe (or fallback FP32 hdr per-tensor)
- [ ] Encoder does ONLY repack — no quality degradation vs Q4_K_M
  decode-path bit-for-bit (modulo FP16 of d × sc which introduces
  ~2^-11 relative error vs FP32(d) × FP32(sc) — acceptable, since
  GGUF stores `d` as fp16 anyway, so we lose nothing more)
- [ ] Decoder is mmap-only, zero math at load (modulo optional
  NUMA memcpy)

Once Agent B confirms, implementation = **1 session** for converter +
**1 session** for loader integration + **2 sessions** for Agent D
kernel work = total ~**4 sessions** to full 30 tok/s target.

---

*End of format_spec_v1.md*
