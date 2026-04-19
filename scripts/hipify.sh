#!/bin/bash
# ============================================================================
# PromeTorch: CUDA → HIP source translator
# ============================================================================
# Usage: hipify.sh <input_dir> <output_dir>
#
# Translates .cu / .cuh / .h files from CUDA to HIP syntax so the AMD ROCm
# backend can be built from the single-source CUDA tree. Uses `hipify-perl`
# if available, otherwise falls back to an in-tree sed map covering the
# ~30 most common tokens used by PromeTorch's kernels.
#
# Design notes:
# - We never mutate the original CUDA tree; translated files land in
#   <output_dir> mirroring the input structure.
# - The `CUDA*` filenames are preserved (CUDAQuantGemv.cu stays named that
#   way) so #include paths remain valid. HIPCompat.h maps the tokens.
# - Idempotent: rerunning overwrites outputs. `make` picks up changes via
#   mtime.
# ============================================================================

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 2
fi

IN_DIR="$1"
OUT_DIR="$2"

if [ ! -d "$IN_DIR" ]; then
    echo "hipify: input dir does not exist: $IN_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

HAVE_HIPIFY=0
if command -v hipify-perl >/dev/null 2>&1; then
    HAVE_HIPIFY=1
    echo "hipify: using hipify-perl ($(command -v hipify-perl))"
else
    echo "hipify: hipify-perl not found — falling back to in-tree sed map"
fi

# ----------------------------------------------------------------------------
# sed translation table (runtime + libs + warp intrinsics that hipify covers)
# ----------------------------------------------------------------------------
read -r -d '' SED_EXPR <<'SED_SCRIPT' || true
s/\bcudaError_t\b/hipError_t/g
s/\bcudaSuccess\b/hipSuccess/g
s/\bcudaStream_t\b/hipStream_t/g
s/\bcudaEvent_t\b/hipEvent_t/g
s/\bcudaStreamCreate\b/hipStreamCreate/g
s/\bcudaStreamDestroy\b/hipStreamDestroy/g
s/\bcudaStreamSynchronize\b/hipStreamSynchronize/g
s/\bcudaStreamDefault\b/hipStreamDefault/g
s/\bcudaStreamNonBlocking\b/hipStreamNonBlocking/g
s/\bcudaEventCreate\b/hipEventCreate/g
s/\bcudaEventRecord\b/hipEventRecord/g
s/\bcudaEventSynchronize\b/hipEventSynchronize/g
s/\bcudaEventDestroy\b/hipEventDestroy/g
s/\bcudaMalloc\b/hipMalloc/g
s/\bcudaMallocManaged\b/hipMallocManaged/g
s/\bcudaMallocHost\b/hipHostMalloc/g
s/\bcudaFree\b/hipFree/g
s/\bcudaFreeHost\b/hipHostFree/g
s/\bcudaMemcpy\b/hipMemcpy/g
s/\bcudaMemcpyAsync\b/hipMemcpyAsync/g
s/\bcudaMemset\b/hipMemset/g
s/\bcudaMemsetAsync\b/hipMemsetAsync/g
s/\bcudaMemcpyHostToDevice\b/hipMemcpyHostToDevice/g
s/\bcudaMemcpyDeviceToHost\b/hipMemcpyDeviceToHost/g
s/\bcudaMemcpyDeviceToDevice\b/hipMemcpyDeviceToDevice/g
s/\bcudaMemcpyKind\b/hipMemcpyKind/g
s/\bcudaGetLastError\b/hipGetLastError/g
s/\bcudaGetErrorString\b/hipGetErrorString/g
s/\bcudaDeviceSynchronize\b/hipDeviceSynchronize/g
s/\bcudaSetDevice\b/hipSetDevice/g
s/\bcudaGetDevice\b/hipGetDevice/g
s/\bcudaGetDeviceCount\b/hipGetDeviceCount/g
s/\bcudaDeviceProp\b/hipDeviceProp_t/g
s/\bcudaGetDeviceProperties\b/hipGetDeviceProperties/g
s/\bcudaFuncSetAttribute\b/hipFuncSetAttribute/g
s/\bcudaFuncAttributeMaxDynamicSharedMemorySize\b/hipFuncAttributeMaxDynamicSharedMemorySize/g
s|<cuda_runtime\.h>|<hip/hip_runtime.h>|g
s|<cuda_fp16\.h>|<hip/hip_fp16.h>|g
s|<cuda\.h>|<hip/hip_runtime.h>|g
s|<cublas_v2\.h>|<rocblas/rocblas.h>|g
s|<cublas\.h>|<rocblas/rocblas.h>|g
s|<cudnn\.h>|<miopen/miopen.h>|g
s/\bcublasHandle_t\b/rocblas_handle/g
s/\bcublasCreate\b/rocblas_create_handle/g
s/\bcublasDestroy\b/rocblas_destroy_handle/g
s/\bcublasSetStream\b/rocblas_set_stream/g
s/\bcublasSgemm\b/rocblas_sgemm/g
s/\bcublasSgemv\b/rocblas_sgemv/g
s/\bcublasStatus_t\b/rocblas_status/g
s/\bCUBLAS_STATUS_SUCCESS\b/rocblas_status_success/g
s/\bCUBLAS_OP_N\b/rocblas_operation_none/g
s/\bCUBLAS_OP_T\b/rocblas_operation_transpose/g
s/__shfl_down_sync([[:space:]]*0xffffffff[[:space:]]*,[[:space:]]*/__shfl_down(/g
s/__shfl_up_sync([[:space:]]*0xffffffff[[:space:]]*,[[:space:]]*/__shfl_up(/g
s/__shfl_xor_sync([[:space:]]*0xffffffff[[:space:]]*,[[:space:]]*/__shfl_xor(/g
s/__shfl_sync([[:space:]]*0xffffffff[[:space:]]*,[[:space:]]*/__shfl(/g
s/__ballot_sync([[:space:]]*0xffffffff[[:space:]]*,[[:space:]]*/__ballot(/g
SED_SCRIPT

process_file() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [ "$HAVE_HIPIFY" = "1" ]; then
        hipify-perl "$src" > "$dst"
    else
        # portable sed: write script to tempfile (macOS / GNU differ on -E semantics)
        local tmp
        tmp="$(mktemp)"
        printf '%s\n' "$SED_EXPR" > "$tmp"
        sed -f "$tmp" "$src" > "$dst"
        rm -f "$tmp"
    fi
    # Ensure HIPCompat.h is pulled in so leftover CUDA tokens still resolve.
    if ! grep -q "HIPCompat.h" "$dst"; then
        # Insert include right after the first #include (or at top).
        local header='#include "aten/src/ATen/hip/HIPCompat.h"'
        if grep -qE '^\s*#include' "$dst"; then
            awk -v hdr="$header" '
                BEGIN { ins=0 }
                { print }
                /^\s*#include/ && !ins { print hdr; ins=1 }
            ' "$dst" > "$dst.tmp" && mv "$dst.tmp" "$dst"
        else
            { echo "$header"; cat "$dst"; } > "$dst.tmp" && mv "$dst.tmp" "$dst"
        fi
    fi
}

COUNT=0
# Find .cu, .cuh, and .h files under IN_DIR (preserve relative paths).
while IFS= read -r -d '' f; do
    rel="${f#$IN_DIR/}"
    process_file "$f" "$OUT_DIR/$rel"
    COUNT=$((COUNT + 1))
done < <(find "$IN_DIR" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) -print0)

echo "hipify: translated $COUNT file(s) from $IN_DIR -> $OUT_DIR"
