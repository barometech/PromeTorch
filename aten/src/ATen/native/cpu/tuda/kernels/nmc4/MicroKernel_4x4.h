#pragma once
// ============================================================================
// MicroKernel_4x4.h — NMC4 NeuroMatrix GEMM micro-kernel (SUDA)
// ============================================================================
// C[4×4] += A[4×K] × B[K×4]
//
// On NMC4: large matmuls go through nmpp (nmppmMul_mm_32f) which uses the
// vector pipeline (4 FPU cores, rep vlen, vreg). This microkernel is the
// fallback for small/edge tiles in the Goto BLAS macro-kernel.
//
// 16 independent scalar accumulators → NMC GCC can pipeline across 4 FPU.
// K-loop unrolled 2× for software pipelining on RISC core.
// ============================================================================

#include <cstdint>

namespace at {
namespace native {
namespace tuda {
namespace kernels {

static void microkernel_4x4_nmc4(
    int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    float c00=0,c01=0,c02=0,c03=0;
    float c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0;
    float c30=0,c31=0,c32=0,c33=0;

    int64_t p = 0;
    for (; p + 2 <= K; p += 2) {
        float a0=A[p*4+0],a1=A[p*4+1],a2=A[p*4+2],a3=A[p*4+3];
        float b0=B[p*4+0],b1=B[p*4+1],b2=B[p*4+2],b3=B[p*4+3];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;

        a0=A[(p+1)*4+0];a1=A[(p+1)*4+1];a2=A[(p+1)*4+2];a3=A[(p+1)*4+3];
        b0=B[(p+1)*4+0];b1=B[(p+1)*4+1];b2=B[(p+1)*4+2];b3=B[(p+1)*4+3];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;
    }
    for (; p < K; ++p) {
        float a0=A[p*4+0],a1=A[p*4+1],a2=A[p*4+2],a3=A[p*4+3];
        float b0=B[p*4+0],b1=B[p*4+1],b2=B[p*4+2],b3=B[p*4+3];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;
    }

    if (beta == 0.0f) {
        C[0*ldc+0]=alpha*c00; C[0*ldc+1]=alpha*c01; C[0*ldc+2]=alpha*c02; C[0*ldc+3]=alpha*c03;
        C[1*ldc+0]=alpha*c10; C[1*ldc+1]=alpha*c11; C[1*ldc+2]=alpha*c12; C[1*ldc+3]=alpha*c13;
        C[2*ldc+0]=alpha*c20; C[2*ldc+1]=alpha*c21; C[2*ldc+2]=alpha*c22; C[2*ldc+3]=alpha*c23;
        C[3*ldc+0]=alpha*c30; C[3*ldc+1]=alpha*c31; C[3*ldc+2]=alpha*c32; C[3*ldc+3]=alpha*c33;
    } else {
        C[0*ldc+0]=alpha*c00+beta*C[0*ldc+0]; C[0*ldc+1]=alpha*c01+beta*C[0*ldc+1];
        C[0*ldc+2]=alpha*c02+beta*C[0*ldc+2]; C[0*ldc+3]=alpha*c03+beta*C[0*ldc+3];
        C[1*ldc+0]=alpha*c10+beta*C[1*ldc+0]; C[1*ldc+1]=alpha*c11+beta*C[1*ldc+1];
        C[1*ldc+2]=alpha*c12+beta*C[1*ldc+2]; C[1*ldc+3]=alpha*c13+beta*C[1*ldc+3];
        C[2*ldc+0]=alpha*c20+beta*C[2*ldc+0]; C[2*ldc+1]=alpha*c21+beta*C[2*ldc+1];
        C[2*ldc+2]=alpha*c22+beta*C[2*ldc+2]; C[2*ldc+3]=alpha*c23+beta*C[2*ldc+3];
        C[3*ldc+0]=alpha*c30+beta*C[3*ldc+0]; C[3*ldc+1]=alpha*c31+beta*C[3*ldc+1];
        C[3*ldc+2]=alpha*c32+beta*C[3*ldc+2]; C[3*ldc+3]=alpha*c33+beta*C[3*ldc+3];
    }
}

} // namespace kernels
} // namespace tuda
} // namespace native
} // namespace at
