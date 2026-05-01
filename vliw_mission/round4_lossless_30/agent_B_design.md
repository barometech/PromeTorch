# Agent B — Design: GGUF → .pt8 converter pipeline + CLI

**Status:** infra-only (CLI parsing, mmap, threading, progress, validation harness)
ready. Encoding-logic stubs marked `TODO[AGENT_A_SPEC]` await
`format_spec_v1.md` from Agent A.

**Date:** 2026-04-30
**Owner:** Agent B (Round 4)
**Coordinates with:** Agent A (format spec), Agent C (loader/inference reader).

---

## 1. Scope

Produce a single static binary `gguf2pt8` (alias `prometorch-convert`) that
converts a GGUF Q4_K_M (or Q4_0 / Q6_K / Q8_0 / F16 / F32) checkpoint into the
new `.pt8` format defined by Agent A. The converter is **bit-lossless**
relative to the GGUF source (interpretation B from `MISSION.md` §2): every
weight in the output `.pt8` is mathematically equivalent to a re-layout of
the GGUF block; no precision is dropped.

Out of scope:
- Defining encoding (Agent A).
- Loading `.pt8` for inference (Agent C — `torch/io/pt8_reader.h`).
- Optimised kernels over `.pt8` (Agent D).

---

## 2. CLI surface

```
prometorch-convert <input.gguf> [-o output.pt8]
                                [--format pt8|pt8_full|pt8_q8soa4]
                                [--validate]
                                [--threads N]
                                [--progress]
                                [-v|--verbose]
                                [-h|--help]
```

Exit codes (matches MISSION constraint):
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | User error (bad CLI args, missing input, conflicting flags) |
| 2 | Input format error (bad GGUF magic, truncated header, unknown tensor type) |
| 3 | Disk write error (no space, permission, I/O error mid-write) |
| 4 | Validation failed (logits mismatch above tolerance, or magic-readback fail) |
| 5 | Internal error (unexpected exception) |

Default output filename derivation: `model.gguf` → `model.pt8` in the same
directory as input. `-o -` writes to stdout (binary; useful for piping).

`--format` selects the encoding variant. Default `pt8` = whatever Agent A
calls "primary"; `pt8_q8soa4` forces Q8 SoA4 layout (proven path); `pt8_full`
keeps F32/F16/BF16 tensors verbatim plus quant tensors in primary layout.

`--validate` forces a full re-load of the freshly-written `.pt8` and a
logits diff against a forward pass on the source GGUF using a fixed test
prompt ("Hello, world. The capital of France is "). Tolerance: max |Δlogit|
< 1e-5. Validation requires the inference path linked in (gguf_model.h
+ pt8_reader.h from Agent C). On Elbrus, build with `-DPT_BUILD_VALIDATE=ON`;
on x86 dev — default ON.

`--threads N` sets OpenMP thread count. Default = 1 fewer than
`std::thread::hardware_concurrency()` (leave one core for OS).

`--progress` prints one-line `\r`-overwriting status:
`[ 47 / 219 tensors | 21.4 % | 312 MB/s | ETA 12s ]`. Auto-disabled when
stderr is not a TTY (use `isatty(2)` / `_isatty(_fileno(stderr))` on win32).

`-v` prints per-tensor lines on stderr; `-vv` adds dump of header KV count
and offsets. `-h` prints usage to stdout (exit 0).

---

## 3. Pipeline architecture

```
                ┌─────────────────────────────────────────────┐
   GGUF file ──►│  gguf::GGUFReader (mmap, parse header+meta) │
                └────────────┬────────────────────────────────┘
                             │  metadata KV, tensor_info[]
                             ▼
                ┌─────────────────────────────────────────────┐
                │     converter::Pt8Writer  (header reserve)  │
                └────────────┬────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │      Tensor work-queue  │ (one entry per gguf tensor)
                └────────────┬────────────┘
                             │
   ┌──────────────┬──────────┼──────────┬───────────────┐
   │              │          │          │               │
   ▼              ▼          ▼          ▼               ▼
 worker 0     worker 1   worker 2   ...           worker N-1
 (thread)                                          (thread)
   │              │          │          │               │
   │              │          │          │               │
   └──────────────┴────┬─────┴──────────┴───────────────┘
                       │   each worker:
                       │     1. claim next tensor (mutex'd index)
                       │     2. read raw bytes via mmap (zero-copy view)
                       │     3. encode → owned buffer
                       │        (TODO[AGENT_A_SPEC] kernel)
                       │     4. enqueue (offset_request, buffer) to writer
                       ▼
            ┌─────────────────────────────────────┐
            │  WriterQueue  (single producer-side │
            │   serialiser; multi-producer in)    │
            └────────────┬────────────────────────┘
                         │ pwrite() on output fd
                         ▼
                   .pt8 file
```

### Threading model
- `std::thread` pool of size `--threads N`.
- Work distribution: dynamic — workers pop the next tensor index from an
  atomic counter (no per-worker static partition; tensor sizes vary 8 KB to
  500 MB).
- Output offsets allocated by the writer thread under a `std::mutex`. Each
  worker hands the writer (offset_already_decided, encoded_buffer); writer
  does `pwrite()` (POSIX) / `WriteFile + SetFilePointerEx` (Win32). The
  writer is the *only* code that calls into the output file.
- Alternative considered: per-worker pre-allocated offset (compute total
  size upfront, hand each worker its slot). Not chosen: encoding is variable
  (e.g. row padding) so size is only known after encode. We instead reserve
  the header up front with placeholder offsets, fill the data section in
  arrival order, and write the final tensor table at the **end** of the
  file with absolute offsets. The header points to the tail-table.

### Mmap (zero-copy input)
Reuses `torch::io::gguf::MmapHandle` from `torch/io/gguf_loader.h` (already
cross-platform, Win32 + POSIX). On Linux we additionally `madvise(...,
MADV_SEQUENTIAL)` for the input — converter touches each weight exactly once
in tensor-table order, opposite to inference-time `MADV_RANDOM`. Override
the default in `gguf_loader.h` is via a flag we'll add to `MmapHandle::open`,
or — cheaper — we just call `posix_madvise` ourselves after the mmap is up.

### Streaming write
Output is opened with `O_WRONLY | O_CREAT | O_TRUNC` (Linux) / `CREATE_ALWAYS,
GENERIC_WRITE` (Win32). We `ftruncate` / `SetEndOfFile` to a small upper
bound only at the end. Each tensor's encoded buffer is `pwrite`'d directly,
no intermediate buffer beyond the worker's per-tensor output. For Win32,
non-overlapped synchronous `WriteFile` with explicit `SetFilePointerEx` is
fine — we serialise writes on the writer mutex anyway.

### Progress + ETA
A monitor thread polls (`std::condition_variable_any` woken by every worker
on tensor completion) and writes a `\r`-line every 100 ms. Bytes/sec
computed from `(total_input_bytes_consumed) / elapsed_wall`. ETA from
`(remaining_input_bytes) / current_bytes_per_sec`.

---

## 4. Per-tensor encoder dispatch (interface only; logic deferred)

Defined in `tools/gguf2pt8/converter.h::encode_tensor()`. Switches on
GGUF type:

| GGUF type | Action | Implementation owner |
|---|---|---|
| `F32` | copy bytes verbatim, write as `PT8_TYPE_F32` | Agent B (trivial memcpy) |
| `F16` / `BF16` | copy verbatim | Agent B |
| `Q4_K` | repack via Agent A spec → primary `PT8_TYPE_*` | **TODO[AGENT_A_SPEC]** |
| `Q6_K` | repack via Agent A spec | **TODO[AGENT_A_SPEC]** |
| `Q5_K` | repack via Agent A spec | **TODO[AGENT_A_SPEC]** |
| `Q8_0` | repack via Agent A spec | **TODO[AGENT_A_SPEC]** |
| `Q4_0`, `Q5_0`, `Q4_1`, `Q5_1` | repack via Agent A spec | **TODO[AGENT_A_SPEC]** |
| anything else | fail with code 2 |  |

Stub today: for Q* tensors we call into a function pointer
`g_encoder_q4k = nullptr;`. Until Agent A registers a real callback, the
converter emits a clear error: `"Encoder for Q4_K not yet wired — awaiting
Agent A format spec. See vliw_mission/round4_lossless_30/format_spec_v1.md"`
and exits 2 (input format error — closest match).

The interface Agent A must fill:
```cpp
// In tools/gguf2pt8/converter.h
struct EncoderRegistry {
    using EncodeFn = bool(*)(
        const void* src,            // mmap'd raw block
        int64_t rows, int64_t cols,
        int64_t src_row_stride,     // bytes
        std::vector<uint8_t>& dst,  // output buffer, sized by encoder
        uint32_t& out_pt8_type,     // dtype-tag the writer should record
        std::string& err_msg
    );
    EncodeFn q4_k = nullptr;
    EncodeFn q6_k = nullptr;
    EncodeFn q5_k = nullptr;
    EncodeFn q8_0 = nullptr;
    EncodeFn q4_0 = nullptr;
    // ... etc.
};
extern EncoderRegistry g_encoders;
```

Agent A populates it at TU init time (e.g. via a static initialiser inside
a `pt8_encode_q4k.cpp` they author). Until then `g_encoders` is empty, the
binary builds, the tests for F32/F16 pass, and the Q* path emits the
explicit-stub error.

---

## 5. .pt8 file layout (provisional, mirrors Agent A direction)

This is **placeholder** until `format_spec_v1.md` lands. The converter
header reserves these fields and Agent A may rename / re-shape:

```
┌────────────────────── header (256 bytes, padded) ─────────────────────┐
│  0..3      magic        "PT8\0" (0x00385450 LE)                       │
│  4..7      version      u32 = 1                                       │
│  8..15     flags        u64  bit0=tied_embeddings bit1=has_validation │
│ 16..23     tensor_table_offset  u64 (filled at end of write)          │
│ 24..31     tensor_count  u64                                          │
│ 32..63     source_gguf_sha256   32 bytes (provenance, lossless audit) │
│ 64..71     param_count   u64                                          │
│ 72..79     header_kv_offset u64 (metadata: arch, hidden, layers …)    │
│ 80..87     header_kv_size   u64                                       │
│ 88..255    reserved (zero)                                            │
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────── KV metadata block ──────────────────────────────┐
│   GGUF-style KV pairs copied verbatim from input (so loader can       │
│   read tokenizer, rope-base, etc. without re-parsing GGUF)            │
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────── data section ───────────────────────────────────┐
│   per-tensor encoded blobs, 64-byte aligned                           │
│   (alignment chosen for E2K APB cache lines, no other reason)         │
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────── tensor table (at tail) ─────────────────────────┐
│ for each tensor:                                                      │
│   name_len   u32                                                      │
│   name       bytes[name_len]                                          │
│   pt8_type   u32  (Agent A defines codes)                             │
│   n_dims     u32                                                      │
│   dims       u64[n_dims]                                              │
│   data_off   u64  (absolute, from file start)                         │
│   data_size  u64                                                      │
│   row_stride u64                                                      │
│   meta_blob_len  u32                                                  │
│   meta_blob      bytes (per-encoder extras — e.g. precomputed sum_q   │
│                  scale block strides, anything Agent A wants)         │
└───────────────────────────────────────────────────────────────────────┘
```

The tail-table approach lets us stream-write data without knowing final
offsets up front. The header field `tensor_table_offset` is patched in
place when writing finishes (`pwrite(fd, &offset, 8, 16)`).

Agent A is expected to (a) approve / reshape this layout, (b) define
`pt8_type` codes, (c) define `meta_blob` for Q4_K (likely empty:
the SoA4 spec is fully implied by `(rows,cols)`).

---

## 6. Validation (`--validate`)

Implemented in `converter.cpp::validate_roundtrip()`:

1. Open just-written `.pt8` via `pt8_reader.h` (Agent C).
2. Construct two `GGUFModel`s — one from source GGUF, one from `.pt8`.
3. Run a deterministic forward pass on a hard-coded prompt
   ("The capital of France is ") with greedy sampling, **single token**.
4. Compute `max_abs_diff` across the logit vector.
5. Pass: `< 1e-5`. Fail: exit code 4.

If `pt8_reader.h` is not yet available (Agent C still working) `--validate`
prints `[validate] reader not yet integrated — skipping (warn, exit 0)`.

For initial sanity (before Agent C lands) we have a cheaper validation:
re-open the `.pt8`, walk the tensor table, verify magic / table consistency
/ each tensor's `data_size` is reachable in the file. This always runs
under `--validate` regardless of reader availability.

---

## 7. Build integration

New files & targets:

```
tools/
  CMakeLists.txt           # new — adds gguf2pt8 subdir
  gguf2pt8/
    CMakeLists.txt
    main.cpp               # CLI driver
    converter.h            # core pipeline (header-only, includes only)
    converter.cpp          # pipeline impl
    pt8_format.h           # shared layout constants (Agent A may extend)
    encoders/
      encode_passthrough.h # F32/F16/BF16 — Agent B done
      encode_q4k.cpp       # Q4_K — Agent A TODO
      encode_q6k.cpp       # Q6_K — Agent A TODO
      ...
    gui/
      README.md            # optional design (see §9)
```

`tools/CMakeLists.txt`:
```cmake
add_subdirectory(gguf2pt8)
```

Top-level `CMakeLists.txt` patch (one line, opt-in):
```cmake
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
endif()
```

Cross-compile:
- **x86_64-msvc** (Windows dev): standard MSVC 2019 NMake, links `c10` and
  `aten` static archives, plus `std::thread` (built-in).
- **x86_64-gcc** (Linux dev): adds `-pthread`.
- **e2k-lcc** (Elbrus): adds `-pthread`, no AVX, must compile under LCC
  1.29 — code is plain C++17, no intrinsics, no `__AVX__` paths in the
  converter (encoder kernels themselves may use intrinsics, but the
  converter machinery must not).

The CMakeLists detects compiler and gates accordingly:
```cmake
if(CMAKE_SYSTEM_PROCESSOR MATCHES "e2k")
    target_compile_options(gguf2pt8 PRIVATE -fno-stack-protector)
endif()
```

Static linking (`-static-libstdc++ -static-libgcc` on Linux,
`/MT` on MSVC) so the binary is self-contained and copyable to Elbrus
without dependency dance. Skipped on macOS (no static libc).

---

## 8. Acceptance criteria for Agent B's portion

- [ ] `prometorch-convert --help` works on x86 + Elbrus.
- [ ] `prometorch-convert tiny.gguf` (1 F32 tensor) round-trips byte-exact.
- [ ] `--threads 4` shows ≥ 3× speedup on a 4 B parameter model versus
      `--threads 1`.
- [ ] `--progress` paints a single updating line on TTY, plain log when piped.
- [ ] Returns 2 (with clear error) when Agent A encoders not registered.
- [ ] Returns 0 on F32/F16/BF16 inputs without Agent A integration.
- [ ] After Agent A registers Q4_K encoder: full `qwen3-4b-Q4_K_M.gguf`
      conversion in < 30 s on x86 dev box.
- [ ] `--validate` succeeds within 1e-5 max abs diff.

---

## 9. (Optional) Web-based GUI

Single-binary HTTP server design lives at `tools/gguf2pt8/gui/README.md`.
Architecture sketch:
- Embedded HTML/JS/CSS in a `gui_assets.h` (raw string literal).
- Tiny stdlib HTTP server (~200 LOC) listening on `127.0.0.1:0` (random port,
  printed on launch).
- Endpoints:
  - `GET /` → embedded `index.html` (drag-drop area, dark theme, single
    "Convert" button, progress bar, log tail).
  - `POST /convert?in=<path>&format=pt8` → runs the same converter pipeline,
    streams progress as Server-Sent Events.
  - `GET /progress` → SSE stream.
  - `POST /cancel` → atomic cancel flag.
- Drag-and-drop in browser uploads the **path** (Electron/file:// trick is
  not allowed for a pure-HTTP design); for true upload we'd need to spool
  the file. MVP: user pastes the absolute path in a textbox.
- No JS frameworks: vanilla DOM + `EventSource` for SSE.
- Static linking, single executable (`prometorch-convert-gui`) shares the
  `converter.h` library with the CLI.
- Theme: dark (#0e0e10 bg, #f5f5f7 text, #6c5ce7 accent), one font
  (system-ui), no images, < 8 KB total payload.

Low priority — MVP target is CLI only.

---

## 10. Coordination with other agents

- **Agent A:** owns `format_spec_v1.md` + `tools/gguf2pt8/encoders/encode_q*.cpp`
  + extends `pt8_format.h` with type tags. Plug-in interface specified in §4.
- **Agent C:** owns `torch/io/pt8_reader.h`. Validation in §6 calls it; if
  not present, validation gracefully degrades.
- **Agent D:** consumer of `pt8_reader.h`, no direct interface with B.
- **Agent E:** orthogonal (speculative decode), no interface.

If Agent A's spec radically changes the layout (e.g. moves tensor table to
the *front* with a two-pass write), only `pt8_format.h` constants and
`Pt8Writer::finalise()` need updating. All threading / mmap / progress code
is invariant to that.
