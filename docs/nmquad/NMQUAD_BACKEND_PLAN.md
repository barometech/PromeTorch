# NM QUAD Backend for PromeTorch — Comprehensive Plan

**Date:** 2026-03-20
**Author:** Claude Opus 4.6
**Status:** PLAN (no code written yet)

---

## 1. Executive Summary

Create a **4th backend** for PromeTorch targeting the NM QUAD board (4x NM6408 DSP chips).
Unlike the NM Card Mini backend (which started as an emulator), this will be a **real hardware backend from day one**, using `nm_quad_load.dll` for host-device communication and `nmc-g++` for compiling NMC kernels.

### Key Differences from NM Card Mini Backend

| Aspect | NM Card Mini | NM QUAD |
|--------|-------------|---------|
| Chip | 1x NMC4 (16 cores, 4 clusters) | 4x NM6408 (each with multiple cores) |
| Host API DLL | `nm_card_load.dll` | `nm_quad_load.dll` |
| Import lib | `nm_card_load.lib` | `nm_quad_load.lib` |
| NMC compiler | Same `nmc-g++` | Same `nmc-g++` |
| NMC-side lib | `nm6408load_nmc` | `nm6408load_nmc` (same chip family) |
| Run tool | N/A (DLL-based) | `nm_quad_run -c<chip> -n<core> program.abs` |
| Remote exec | N/A | Via nmrb proxy (`nm_quad_load_proxy.dll`) |
| Memory model | DDR3L ~500MB shared | DDR per chip, NMMB 512KB per core |
| Device type | `PrivateUse1` | Need new `PrivateUse3` (or sub-device of PrivateUse1) |
| Emulator | Yes (primary) | No (real hardware only) |
| Float type | Q16.16 fixed OR float32 | Native float32 (NM6408 has FPU) |

---

## 2. Hardware Architecture

### NM6408 Chip
- **VLIW DSP** with NeuroMatrix vector core
- **FPU**: 4 floating-point units per core (confirmed by `nmppmMul_mm_32f` in dispatcher)
- **NMMB**: 512KB internal memory (0x00000800 to 0x0007F800, from linker script)
- **EMI_CPU**: External DDR access (0x00340000 to 0x1FCC0000 = ~509MB per chip)
- **nmc-g++**: GCC-based compiler for NMC architecture
- **nmpp**: NeuroMatrix Performance Primitives (BLAS, vector ops)

### NM QUAD Board
- **4 NM6408 chips** (addressed as chip 0..3)
- Each chip has cores addressed as `(cluster_id, nm_id)` pairs
- Total accessible memory: 4 * ~509MB DDR + 4 * 512KB NMMB
- Host communication via PCIe through `nm_quad_load.dll`
- Remote execution supported via `nmrb` proxy DLL

### Host API (from `nm_quad_load.h` / examples)
```c
// Core types
typedef struct { int nm_id; int cluster_id; } PL_CoreNo;
typedef unsigned int PL_Word;
typedef unsigned int PL_Addr;

// Board management
int PL_GetBoardCount(unsigned int* count);
int PL_GetBoardDesc(unsigned int index, PL_Board** board);
int PL_ResetBoard(PL_Board* board);
int PL_LoadInitCode(PL_Board* board);

// Core access
int PL_GetAccess(PL_Board* board, PL_CoreNo* id, PL_Access** access);
int PL_LoadProgramFile(PL_Access* access, const char* path);
int PL_CloseAccess(PL_Access* access);
int PL_CloseBoardDesc(PL_Board* board);

// Memory transfer
int PL_WriteMemBlock(PL_Access* access, PL_Word* buf, PL_Addr addr, PL_Word count);
int PL_ReadMemBlock(PL_Access* access, PL_Word* buf, PL_Addr addr, PL_Word count);

// Synchronization
int PL_Sync(PL_Access* access, int value, int* retval);
int PL_SyncArray(PL_Access* access, int value, int flags, int timeout,
                 int* retval, PL_Addr* addr, PL_Word* length);
int PL_GetStatus(PL_Access* access, PL_Word* status);
```

### Execution Model
Two modes available:
1. **Standalone**: `nm_quad_run -c<chip> -n<core> program.abs` — runs program, returns exit code
2. **Host-driven**: Host loads program via `PL_LoadProgramFile`, exchanges data via shared DDR, synchronizes via `PL_Sync`/`PL_SyncArray`

**For PromeTorch we use mode 2** (host-driven), same as NM Card Mini backend.

---

## 3. Device Registration Strategy

### Option A: New DeviceType `PrivateUse3` (RECOMMENDED)
Add `PrivateUse3 = 22` to `DeviceType` enum in `c10/core/Device.h`.

**Pros:**
- Clean separation from NM Card Mini
- Independent allocator, dispatch, device checks
- Can run NM Card Mini and NM QUAD simultaneously

**Cons:**
- Requires touching core device infrastructure
- Need to bump `COMPILE_TIME_MAX_DEVICE_TYPES`

### Option B: Sub-device of PrivateUse1
Use `PrivateUse1` with device index: `nmcard=device:0`, `nmquad=device:1..4`

**Pros:** No changes to Device.h

**Cons:** Allocator collision, confusing semantics, can't distinguish backends

### Decision: **Option A** — add `PrivateUse3 = 22`

Changes needed in `c10/core/Device.h`:
```cpp
PrivateUse3 = 22,  // NM QUAD (4x NM6408)
COMPILE_TIME_MAX_DEVICE_TYPES = 23
```
Plus: `DeviceTypeName` switch case, `is_nmquad()` helper, string parser for "nmquad".

---

## 4. File Structure

### New Files to Create

```
c10/nmquad/
    NMQuadAllocator.h          # DDR allocator (host-side caching + DDR tracking)
    NMQuadAllocator.cpp        # Singleton (.cpp for DLL safety)

aten/src/ATen/nmquad/
    NMQuadHardware.h           # nm_quad_load.dll wrapper, multi-chip access
    NMQuadHardware.cpp         # Singleton, DLL loading, init/shutdown
    NMQuadDispatch.h           # Tensor-level ops: empty_nmquad, to_nmquad, etc.
    NMQuadOps.h                # launch_* functions (dispatch to hardware)
    NMQuadMultiChip.h          # 4-chip parallel dispatch (matmul split across chips)

aten/src/ATen/nmquad/nmc_programs/
    dispatcher_nmquad.cpp      # NMC-side dispatcher (runs on each NM6408 core)
    nm6408brd.lds              # Linker script (copy from SDK)
    Makefile                   # Build dispatcher.abs with nmc-g++

test/cpp/
    test_nmquad.cpp            # Unit tests (hardware required)

examples/nmquad/
    train_mnist_nmquad.cpp     # MNIST training on NM QUAD
    CMakeLists.txt
```

### Files to Modify

```
c10/core/Device.h              # Add PrivateUse3, is_nmquad(), "nmquad" parser
c10/core/TensorImpl.h          # Add is_nmquad() convenience method
aten/src/ATen/core/Tensor.h    # Add is_nmquad() method
aten/src/ATen/ATen.h           # Add #include nmquad headers under PT_USE_NMQUAD
CMakeLists.txt                 # Add PT_USE_NMQUAD option, nmquad library target
```

---

## 5. Memory Architecture

### Host-Side (NMQuadAllocator)

Same pattern as NMCardAllocator — **caching allocator** using aligned host RAM, tagged with `DeviceType::PrivateUse3`. Tensors "on NM QUAD" live in host RAM but are marked as device=nmquad.

```
NMQuadAllocator : public c10::Allocator
├── allocate(nbytes) → DataPtr with Device(PrivateUse3, chip_id)
├── raw_deallocate(ptr) → cache for reuse
├── empty_cache() → free cached blocks
└── Singleton in .cpp (DLL safety!)
```

The `device_index` field (0..3) indicates which NM6408 chip the data is destined for.

### Device-Side (DDR Allocator per Chip)

Each NM6408 chip has its own DDR address space (0x00340000..0x1FCC0000). The `NMQuadHardware` class manages 4 independent `DDRAllocator` instances, one per chip.

```
NMQuadHardware
├── chip_[0] → { PL_Access*, DDRAllocator, dispatcher loaded }
├── chip_[1] → { PL_Access*, DDRAllocator, dispatcher loaded }
├── chip_[2] → { PL_Access*, DDRAllocator, dispatcher loaded }
└── chip_[3] → { PL_Access*, DDRAllocator, dispatcher loaded }
```

### Data Flow

```
Host RAM tensor → PL_WriteMemBlock → NM6408 DDR → NMC core executes
                                                     ↓
Host RAM result ← PL_ReadMemBlock  ← NM6408 DDR ← result written
```

### Memory Regions (per chip, from linker script)

| Region | Start | End | Size | Purpose |
|--------|-------|-----|------|---------|
| NMMB | 0x00000800 | 0x0007F800 | 510 KB | Code + stack (dispatcher lives here) |
| EMI_CPU (DDR) | 0x00340000 | 0x1FCC0000 | ~509 MB | Data (tensors, weights, activations) |

### DDR Layout (per chip)

```
0x00340000  +------------------+
            | Cmd block (32 words per core) |  512 bytes
0x00340200  +------------------+
            | Data area (bump allocator) |
            | ... tensors ...              |
0x1FCC0000  +------------------+  (DDR end)
```

---

## 6. NMC-Side Dispatcher

### Architecture

The dispatcher is a C program compiled with `nmc-g++` that runs on each NM6408 core. It polls a command block in DDR for opcodes, executes the operation, then signals completion.

```c
// dispatcher_nmquad.cpp — compiled to dispatcher_nmquad.abs
volatile unsigned int* mem = (volatile unsigned int*)DDR_BASE;

int main() {
    // Calculate own cmd block based on core index
    int core_index = /* from ncl_getCoreID/ncl_getClusterID */;
    volatile unsigned int* cmd = mem + core_index * CMD_BLOCK_SIZE;

    while (1) {
        unsigned int opcode = cmd[0];
        if (opcode == OP_NOP) continue;  // Spin-wait
        if (opcode == OP_EXIT) break;

        // Reset status
        cmd[STATUS_OFFSET] = 0;

        switch (opcode) {
            case OP_MATMUL:    op_matmul(cmd);    break;
            case OP_ELEM_ADD:  op_elem_add(cmd);  break;
            case OP_ELEM_MUL:  op_elem_mul(cmd);  break;
            case OP_RELU:      op_relu(cmd);      break;
            case OP_SILU:      op_silu(cmd);      break;
            case OP_SOFTMAX:   op_softmax(cmd);   break;
            case OP_RMSNORM:   op_rmsnorm(cmd);   break;
            case OP_SGD:       op_sgd(cmd);       break;
            // ... more ops
        }

        cmd[STATUS_OFFSET] = 1;  // Done
        cmd[0] = OP_NOP;         // Ready for next
    }
    return 0;
}
```

### NMC Operations to Implement

**Phase 1 (MVP — inference):**

| Op | Opcode | Description | NMC Implementation |
|----|--------|-------------|-------------------|
| MATMUL | 1 | C = A @ B | `nmppmMul_mm_32f` from nmpp |
| ELEM_ADD | 10 | c = a + b | Manual loop or nmpp `nmppsAdd_32f` |
| ELEM_MUL | 11 | c = a * b | Manual loop or nmpp |
| RELU | 4 | max(0, x) | Manual loop |
| SOFTMAX | 7 | softmax(x) | Manual (max, exp, sum, div) |
| FILL | 20 | fill(val) | Manual loop |
| COPY | 21 | memcpy | Manual loop |

**Phase 2 (training):**

| Op | Opcode | Description |
|----|--------|-------------|
| MATMUL_AT | 2 | C = A^T @ B (for backward) |
| MATMUL_BT | 3 | C = A @ B^T (for backward) |
| RELU_BWD | 5 | ReLU backward |
| SILU | 8 | SiLU activation |
| SILU_BWD | 9 | SiLU backward |
| SGD | 50 | SGD optimizer step |
| ADAM | 51 | Adam optimizer step |
| RMSNORM | 6 | RMSNorm |
| ROPE | 12 | Rotary Position Embedding |

**Phase 3 (multi-chip):**

| Op | Opcode | Description |
|----|--------|-------------|
| MATMUL_PARTIAL | 22 | Partial matmul (column range) |
| ALLREDUCE | 30 | Sum across chips |

### Build System for NMC Programs

```batch
@echo off
:: Build dispatcher_nmquad.abs
set NMC_GCC=nmc-g++
set NM_QUAD=C:\Module\NM_Quad

%NMC_GCC% -o dispatcher_nmquad.abs dispatcher_nmquad.cpp ^
    -I %NM_QUAD%\include ^
    -Wl,--whole-archive -l nm6408load_nmc -Wl,--no-whole-archive ^
    -L %NM_QUAD%\lib ^
    -T nm6408brd.lds
```

Key flags:
- `-Wl,--whole-archive -l nm6408load_nmc` — link NMC support library (for `ncl_hostSync`, `ncl_getCoreID`)
- `-T nm6408brd.lds` — memory layout (NMMB for code, EMI_CPU for data)
- `-I` and `-L` — SDK headers and libs

---

## 7. Host-Side Hardware Layer (NMQuadHardware)

### Initialization Sequence

```
1. LoadLibrary("nm_quad_load.dll")
   - Search: C:\Module\NM_Quad\bin\nm_quad_load.dll
   - Or via nmrb proxy: nm_quad_load_proxy.dll (for remote execution)
2. Resolve all PL_* functions
3. PL_GetBoardCount() → verify board present
4. PL_GetBoardDesc(0, &board)
5. PL_ResetBoard(board)
6. PL_LoadInitCode(board)
7. For each chip (0..3):
     PL_GetAccess(board, {chip, 0}, &access[chip])
     PL_LoadProgramFile(access[chip], "dispatcher_nmquad.abs")
8. Wait 100ms for dispatchers to start polling
9. Verify: read cmd[0] == OP_NOP for each chip
```

### Remote Execution Support

NM QUAD supports remote execution via nmrb (Network Remote Board):
- Replace `nm_quad_load.dll` with `nm_quad_load_proxy.dll` (hardlink swap)
- Configure nmrb-settings.exe to point to remote board server
- Same API, transparent network transport

This is critical because the NM QUAD board may not be physically installed in the development PC.

### Multi-Chip Dispatch

```cpp
class NMQuadHardware {
    // 4 chips, each with its own PL_Access and DDR allocator
    struct ChipState {
        void* access = nullptr;
        DDRAllocator ddr;
        bool active = false;
    };
    ChipState chips_[4];

    // Dispatch to specific chip
    void dispatch_op(int chip, uint32_t opcode, const uint32_t* args, int nargs);
    void wait_done(int chip, float timeout_sec = 10.0f);

    // Upload/download to specific chip's DDR
    uint32_t upload(int chip, const float* data, int64_t count);
    void download(int chip, float* data, uint32_t ddr_addr, int64_t count);
};
```

---

## 8. Multi-Chip Parallelism Strategy

### Strategy 1: Data Parallelism (for training)
- Split batch across 4 chips
- Each chip processes batch/4 samples
- AllReduce gradients across chips (host-mediated)

```
Batch [32 samples]
  ├── Chip 0: samples 0-7
  ├── Chip 1: samples 8-15
  ├── Chip 2: samples 16-23
  └── Chip 3: samples 24-31
```

### Strategy 2: Tensor Parallelism (for large matmuls)
- Split output columns across chips
- Each chip computes partial result
- Host concatenates results

```
C[M,N] = A[M,K] @ B[K,N]
  ├── Chip 0: C[:, 0:N/4]   = A @ B[:, 0:N/4]
  ├── Chip 1: C[:, N/4:N/2] = A @ B[:, N/4:N/2]
  ├── Chip 2: C[:, N/2:3N/4]= A @ B[:, N/2:3N/4]
  └── Chip 3: C[:, 3N/4:N]  = A @ B[:, 3N/4:N]
```

### Strategy 3: Pipeline Parallelism (for inference)
- Different layers on different chips
- Chip 0: layers 0-L/4
- Chip 1: layers L/4-L/2
- etc.

### Implementation Order
1. **Phase 1**: Single-chip ops (one chip at a time)
2. **Phase 2**: Tensor parallelism for matmul
3. **Phase 3**: Data parallelism for training
4. **Phase 4**: Pipeline parallelism for inference

---

## 9. Integration with PromeTorch Tensor System

### Dispatch Layer (NMQuadDispatch.h)

Following the exact pattern of `NMCardDispatch.h`:

```cpp
namespace at {
#ifdef PT_USE_NMQUAD

// Create empty tensor on NM QUAD
Tensor empty_nmquad(IntArrayRef sizes, ScalarType dtype, int chip = 0);

// Transfer host ↔ NM QUAD
Tensor to_nmquad(const Tensor& src, int chip = 0);
Tensor nmquad_to_cpu(const Tensor& src);

// Operations (same interface as nmc_ops namespace)
namespace nmquad_ops {
    Tensor mm(const Tensor& a, const Tensor& b);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& input);
    Tensor softmax(const Tensor& input, int64_t dim);
    // ... all ops from NMCardDispatch.h
}

#endif
}
```

### Autograd Integration

The autograd engine dispatches based on device type. For NM QUAD:
- Forward ops execute on NM6408
- Backward ops: initially CPU fallback (same as NM Card Mini pattern)
- Phase 2: backward ops on NM6408 (matmul_at, matmul_bt, relu_bwd)

### Device Transfer in Training Loop

```cpp
// Example: MNIST on NM QUAD
auto x = at::to_nmquad(batch_x, /*chip=*/0);
auto y = at::to_nmquad(batch_y, /*chip=*/0);

// Forward (on NM6408)
auto h1 = nmquad_ops::relu(nmquad_ops::mm(x, w1));
auto out = nmquad_ops::mm(h1, w2);

// Loss + backward (CPU fallback initially)
auto out_cpu = at::nmquad_to_cpu(out);
auto loss = cross_entropy(out_cpu, labels);
loss.backward();

// Optimizer step (CPU)
optimizer.step();

// Copy updated weights back
w1 = at::to_nmquad(w1_cpu, 0);
w2 = at::to_nmquad(w2_cpu, 0);
```

---

## 10. CMake Integration

```cmake
option(PT_USE_NMQUAD "Enable NM QUAD support (4x NM6408)" OFF)

if(PT_USE_NMQUAD)
    set(NMQUAD_SOURCES
        c10/nmquad/NMQuadAllocator.cpp
        aten/src/ATen/nmquad/NMQuadHardware.cpp
    )

    if(BUILD_SHARED_LIBS)
        add_library(aten_nmquad SHARED ${NMQUAD_SOURCES})
        target_compile_definitions(aten_nmquad PRIVATE ATEN_NMQUAD_EXPORTS PT_BUILD_SHARED_LIBS)
    else()
        add_library(aten_nmquad STATIC ${NMQUAD_SOURCES})
    endif()

    target_include_directories(aten_nmquad PUBLIC
        ${CMAKE_SOURCE_DIR}
        C:/Module/NM_Quad/include
    )
    target_link_libraries(aten_nmquad PUBLIC c10)

    # nm_quad_load.lib for linking
    target_link_directories(aten_nmquad PUBLIC C:/Module/NM_Quad/lib)
    target_link_libraries(aten_nmquad PUBLIC nm_quad_load)

    target_link_libraries(aten INTERFACE aten_nmquad)
    target_compile_definitions(aten INTERFACE PT_USE_NMQUAD)
endif()
```

### Build Command

```batch
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set NM_QUAD=C:\Module\NM_Quad
set PATH=%NM_QUAD%\bin;C:\Module\NMCSDK\bin;%PATH%
cd /d C:\Users\paper\Desktop\promethorch\build_nmquad
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DPT_USE_NMQUAD=ON
nmake
```

### NMC Program Build (separate step)

```batch
cd /d C:\Users\paper\Desktop\promethorch\aten\src\ATen\nmquad\nmc_programs
set PATH=C:\Module\NMCSDK\bin;C:\Module\NM_Quad\bin;%PATH%
nmc-g++ -o dispatcher_nmquad.abs dispatcher_nmquad.cpp ^
    -I C:\Module\NM_Quad\include ^
    -Wl,--whole-archive -l nm6408load_nmc -Wl,--no-whole-archive ^
    -L C:\Module\NM_Quad\lib ^
    -T nm6408brd.lds
```

---

## 11. Safety Protocol

Based on the NM Card Mini incident (2026-03-18 crash that required reboot):

### Escalation Ladder
1. **Test dispatcher on emulator first** (if possible — NM QUAD may not have an emulator mode)
2. **Single chip, single core** — verify basic dispatch works
3. **Single chip, matmul** — verify nmpp integration
4. **Multi-chip** — only after single-chip is stable
5. **Training loop** — only after forward pass is verified

### Safeguards in Code
- **Timeout on all wait_done()** — max 10 seconds, throw exception
- **DDR bounds checking** — assert addr + size < DDR_END
- **Graceful shutdown** — OP_EXIT to all cores before closing handles
- **Board reset on init** — PL_ResetBoard clears stale state
- **Error checking** — every PL_* call checked for PL_OK

### Testing Strategy
1. Simple factorial test: compile simple.cpp, run via nm_quad_run, verify result
2. Dispatcher load test: load dispatcher_nmquad.abs, send OP_NOP, verify response
3. Memory test: write 1024 floats to DDR, read back, compare
4. Matmul test: small 4x4 matmul, verify against CPU
5. MNIST forward pass: compare output vs CPU backend
6. MNIST training: compare loss curve vs CPU backend

---

## 12. Implementation Roadmap

### Phase 1: Foundation (1-2 days)
- [ ] Add `PrivateUse3` to Device.h, `is_nmquad()` helpers
- [ ] Create `c10/nmquad/NMQuadAllocator.{h,cpp}`
- [ ] Create `NMQuadHardware.{h,cpp}` — DLL loading, single-chip init
- [ ] Write minimal `dispatcher_nmquad.cpp` (OP_NOP + OP_EXIT only)
- [ ] Build dispatcher with nmc-g++
- [ ] Test: load dispatcher, verify heartbeat

### Phase 2: Basic Ops (1-2 days)
- [ ] Add matmul to dispatcher (via nmppmMul_mm_32f)
- [ ] Add elem_add, elem_mul, relu, fill, copy
- [ ] Create `NMQuadOps.h` — launch_* functions
- [ ] Create `NMQuadDispatch.h` — tensor-level wrappers
- [ ] Test: small matmuls, element-wise ops

### Phase 3: MNIST Forward Pass (1 day)
- [ ] Add softmax, silu, gelu, sigmoid, tanh to dispatcher
- [ ] Create `train_mnist_nmquad.cpp`
- [ ] Test: MNIST forward pass, compare accuracy vs CPU

### Phase 4: Training (1-2 days)
- [ ] Add backward ops: matmul_at, matmul_bt, relu_bwd, silu_bwd
- [ ] Add SGD/Adam optimizer ops
- [ ] Integrate with autograd engine
- [ ] Test: MNIST training, verify convergence

### Phase 5: Multi-Chip (2-3 days)
- [ ] Initialize all 4 chips
- [ ] Implement parallel matmul (tensor parallelism)
- [ ] Implement data parallelism for training
- [ ] Create `NMQuadMultiChip.h`
- [ ] Benchmark: 1-chip vs 4-chip speedup

### Phase 6: Optimization (ongoing)
- [ ] Use nmpp BLAS functions wherever available
- [ ] Optimize DDR access patterns (avoid DDR saturation)
- [ ] Implement persistent data (avoid re-uploading weights every op)
- [ ] Profile and tune

---

## 13. Risk Analysis

### Known Risks

| Risk | Mitigation |
|------|------------|
| Board crash / reboot | Safety escalation ladder, timeouts |
| DDR saturation (known issue from NM Card) | Batch DDR writes, minimize transfers |
| nmpp functions may differ on NM6408 vs NMC4 | Test each function individually |
| nmc-g++ may not be in PATH | Set NM_QUAD env var, explicit paths |
| Remote execution latency | Local execution for development |
| DLL singleton issue (same as NM Card) | .cpp file singletons, double registration |
| Limited internal memory (512KB) | Dispatcher code must be compact |

### Open Questions

1. **How many cores per NM6408 chip?** The NM Card Mini had 16 cores (4 clusters x 4). Need to verify for NM6408.
2. **Can all 4 chips share DDR?** Or does each have independent DDR? (Likely independent based on linker script.)
3. **Is nmpp (`nmppmMul_mm_32f`) available for NM6408?** It was used in the NM Card dispatcher — likely yes since same chip family.
4. **PL_SyncArray vs DDR polling** — which synchronization method is better for our dispatcher model? (DDR polling matches NM Card pattern and is simpler.)
5. **Maximum DDR transfer rate?** Important for deciding whether to keep weights persistent on-chip.

---

## 14. Dependencies

### Required Software
- `nmc-g++` (NMC GCC compiler) — from NMCSDK
- `nm_quad_load.dll` — at `C:\Module\NM_Quad\bin\`
- `nm_quad_load.lib` — at `C:\Module\NM_Quad\lib\`
- `nm6408load_nmc` library — at `C:\Module\NM_Quad\lib\`
- `nm_quad_run.exe` — at `C:\Module\NM_Quad\bin\` (for standalone testing)
- Headers: `nm_quad_load.h`, `nm6408load_nmc.h` — at `C:\Module\NM_Quad\include\`

### Required Hardware
- NM QUAD board (connected via PCIe or remote via nmrb)

### Environment Variables
```batch
set NM_QUAD=C:\Module\NM_Quad
set PATH=%NM_QUAD%\bin;C:\Module\NMCSDK\bin;C:\Module\nmrb-client\bin;%PATH%
```

---

## 15. File-by-File Specification

### c10/nmquad/NMQuadAllocator.h
- Copy pattern from `c10/nmcard/NMCardAllocator.h`
- Use `DeviceType::PrivateUse3` instead of `PrivateUse1`
- Add `device_index` support (0..3) for chip targeting
- Export macro: `ATEN_NMQUAD_API`
- Registration: `register_nmquad_allocator()` + `register_nmquad_allocator_local()`

### c10/nmquad/NMQuadAllocator.cpp
- Global singleton in .cpp (DLL safety)
- Same pattern as NMCardAllocator.cpp

### aten/src/ATen/nmquad/NMQuadHardware.h
- Similar to NMCardHardware.h but with 4 chips
- DLL: `nm_quad_load.dll` (not `nm_card_load.dll`)
- `ChipState` struct with per-chip access handle and DDR allocator
- Multi-chip dispatch and wait
- Upload/download per chip
- Same PL_* function pointer typedefs

### aten/src/ATen/nmquad/NMQuadHardware.cpp
- DLL loading from `C:\Module\NM_Quad\bin\nm_quad_load.dll`
- Init all 4 chips in sequence
- Load dispatcher on each chip
- Timeout and error handling

### aten/src/ATen/nmquad/NMQuadOps.h
- Same structure as NMCardOps.h
- All launch_* functions dispatch to NMQuadHardware
- No emulator fallback (real hardware only)
- CPU fallback for ops not yet implemented on NMC

### aten/src/ATen/nmquad/NMQuadDispatch.h
- Same structure as NMCardDispatch.h
- `empty_nmquad()`, `to_nmquad()`, `nmquad_to_cpu()`
- All tensor-level ops in `nmquad_ops` namespace

### aten/src/ATen/nmquad/NMQuadMultiChip.h
- Parallel matmul across 4 chips
- Data parallelism helpers
- AllReduce via host-mediated DDR reads

### aten/src/ATen/nmquad/nmc_programs/dispatcher_nmquad.cpp
- NMC-side dispatcher (compiled with nmc-g++)
- DDR polling loop
- Opcode dispatch switch
- Uses nmpp for matmul where available
- Same command block protocol as NM Card dispatcher

---

## 16. Summary

The NM QUAD backend follows the proven NM Card Mini pattern but targets real hardware with 4x the compute. The key architectural decisions are:

1. **New DeviceType** (PrivateUse3) for clean separation
2. **Same dispatch protocol** (DDR command blocks) as NM Card Mini
3. **Per-chip state** (4 independent DDR allocators, 4 PL_Access handles)
4. **Incremental rollout** (single chip first, then multi-chip)
5. **nmpp for BLAS** (nmppmMul_mm_32f proven in NM Card dispatcher)
6. **Safety-first** (timeouts, bounds checks, escalation ladder)

Estimated total effort: **5-10 days** for a working MNIST training pipeline.
