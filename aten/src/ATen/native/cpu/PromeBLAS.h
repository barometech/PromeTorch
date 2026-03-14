#pragma once
// ============================================================================
// PromeBLAS — Thin wrapper over TUDA BLAS for backward compatibility
// ============================================================================
// All implementations now live in tuda/TudaBLAS.h with architecture dispatch.
// This file re-exports into at::native::blas for existing callers.
// ============================================================================

#include "aten/src/ATen/native/cpu/tuda/TudaBLAS.h"

namespace at {
namespace native {
namespace blas {

// Re-export TUDA BLAS functions into the original namespace
using tuda::blas::sgemm;
using tuda::blas::sgemm_nt;
using tuda::blas::sgemv;
using tuda::blas::sdot;
using tuda::blas::saxpy;
using tuda::blas::aligned_alloc_f32;
using tuda::blas::aligned_free;

} // namespace blas
} // namespace native
} // namespace at
