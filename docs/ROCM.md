# PromeTorch on AMD ROCm / HIP

PromeTorch ships a single CUDA source tree under `aten/src/ATen/cuda/`. To
build for AMD GPUs we hipify-translate those sources at configure time and
compile them with `hipcc`. There is no second source tree to maintain.

## Requirements

- Linux (Ubuntu 22.04 / RHEL 9 / SLES 15 — whatever AMD supports)
- ROCm 5.7+ (tested: 6.0, 6.2)
- `hipcc`, `hipify-perl` (shipped with ROCm)
- `rocblas`, optional `MIOpen` (cuDNN analogue)
- CMake 3.21+ (for first-class `HIP` language support)
- `bash` (the hipify driver is a bash script)

Supported GPU ISAs (default, override with `-DCMAKE_HIP_ARCHITECTURES=...`):

| gfx    | Card family                |
|--------|----------------------------|
| gfx906 | MI50, Radeon VII           |
| gfx908 | MI100                      |
| gfx90a | MI210 / MI250              |
| gfx1030| RX 6800 / 6900 (RDNA2)     |
| gfx1100| RX 7900 XTX (RDNA3)        |
| gfx1101| RX 7800 / 7700 (RDNA3)     |

## Build

```bash
mkdir -p build_rocm && cd build_rocm
cmake -DPT_USE_ROCM=ON \
      -DPT_USE_CUDA=OFF \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_BUILD_TYPE=Release \
      ..
cmake --build . -j"$(nproc)"
```

If ROCm lives somewhere other than `/opt/rocm`, pass either
`-DROCM_PATH=/opt/rocm-6.2` or set `ROCM_PATH` in the environment.

Override the ISA list to build only for the card you have:

```bash
cmake -DPT_USE_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a ..
```

## What the configure step does

1. Enables CMake's `HIP` language.
2. Runs `scripts/hipify.sh aten/src/ATen/cuda  build_rocm/hipified/aten/src/ATen/cuda`.
   - Prefers `hipify-perl` if present.
   - Falls back to an in-tree sed map covering ~30 tokens (cudaMalloc ->
     hipMalloc, cublasSgemm -> rocblas_sgemm, `__shfl_*_sync` -> `__shfl_*`,
     includes, etc.).
3. Each translated file gets `#include "aten/src/ATen/hip/HIPCompat.h"`
   inserted so anything the translator missed still resolves.
4. Adds the hipified dir as an include path.
5. `aten_cuda` is built from those translated sources and linked against
   `hip::host` + `roc::rocblas` (+ `MIOpen` if found).

## Q4_K / GGUF GEMV

`CUDAQuantGemv.cu` is the only hand-written kernel in the tree and it
already contains an `__HIP_PLATFORM_AMD__` guard so `__ballot_sync` is
correctly widened to a 64-bit wavefront mask on AMD. No changes required.

## `HIPCompat.h`

`aten/src/ATen/hip/HIPCompat.h` is the safety net. It maps:

- Runtime + memory API (`cudaMalloc`, `cudaStream_t`, ...)
- `cuBLAS` -> `rocBLAS` symbols
- Volta+ `*_sync` warp intrinsics -> mask-less HIP intrinsics
- A portable `PT_WARP_SIZE` (32 on NVIDIA, 64 on AMD)

It is a no-op when `__HIPCC__` / `__HIP_PLATFORM_AMD__` are not defined, so
the CUDA build is untouched.

## Mutual exclusion with CUDA

`PT_USE_CUDA` and `PT_USE_ROCM` cannot both be ON — they share CMake's CUDA
language slot and target different toolchains. CMake errors out early if
you try.

## CPU-only / Elbrus builds

Default is `PT_USE_CUDA=OFF PT_USE_ROCM=OFF`. The ROCm block is completely
skipped; CPU/TUDA/EML paths are unaffected.
