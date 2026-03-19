#pragma once
// ============================================================================
// MicroKernel_6x6.h — E2K VLIW-optimized GEMM micro-kernel for Elbrus
// ============================================================================
// C[6x6] += A[6xK] x B[Kx6]
// Pure C — LCC compiler handles VLIW scheduling automatically.
//
// 36 independent FMA accumulators per K-step:
//   - 6 FMA units x 6 cycles = perfect VLIW pipeline utilization
//   - 36 accum regs + 6 A loads + 6 B loads = 48 registers (of 256 available)
//   - No loop-carried dependencies between accumulators
//
// K-loop unrolled 4x for software pipelining — gives LCC enough
// instructions to fill VLIW bundles across multiple K iterations.
//
// Data layout: A packed column-major [K x MR], B packed row-major [K x NR]
//   A[p*6 + row], B[p*6 + col]
//
// E2K has no branch prediction — avoid branches in hot loop.
// Ternary expressions compile to conditional moves (merges in VLIW).
// ============================================================================

#include <cstdint>

namespace at {
namespace native {
namespace tuda {
namespace kernels {

static void microkernel_6x6_e2k(
    int64_t K,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int64_t ldc,
    float alpha,
    float beta
) {
    // 36 independent accumulators — LCC keeps all in registers (48/256 used)
    // Row 0
    float c00=0, c01=0, c02=0, c03=0, c04=0, c05=0;
    // Row 1
    float c10=0, c11=0, c12=0, c13=0, c14=0, c15=0;
    // Row 2
    float c20=0, c21=0, c22=0, c23=0, c24=0, c25=0;
    // Row 3
    float c30=0, c31=0, c32=0, c33=0, c34=0, c35=0;
    // Row 4
    float c40=0, c41=0, c42=0, c43=0, c44=0, c45=0;
    // Row 5
    float c50=0, c51=0, c52=0, c53=0, c54=0, c55=0;

    // Main loop: unroll 4x for LCC software pipelining
    // Each iteration has 36 FMA + 12 loads = 48 ops — plenty for VLIW scheduling
    int64_t p = 0;
    for (; p + 4 <= K; p += 4) {
        // Iteration 0
        {
            float a0 = A[(p+0)*6+0], a1 = A[(p+0)*6+1], a2 = A[(p+0)*6+2];
            float a3 = A[(p+0)*6+3], a4 = A[(p+0)*6+4], a5 = A[(p+0)*6+5];
            float b0 = B[(p+0)*6+0], b1 = B[(p+0)*6+1], b2 = B[(p+0)*6+2];
            float b3 = B[(p+0)*6+3], b4 = B[(p+0)*6+4], b5 = B[(p+0)*6+5];
            c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5;
            c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5;
            c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5;
            c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5;
            c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5;
            c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5;
        }
        // Iteration 1
        {
            float a0 = A[(p+1)*6+0], a1 = A[(p+1)*6+1], a2 = A[(p+1)*6+2];
            float a3 = A[(p+1)*6+3], a4 = A[(p+1)*6+4], a5 = A[(p+1)*6+5];
            float b0 = B[(p+1)*6+0], b1 = B[(p+1)*6+1], b2 = B[(p+1)*6+2];
            float b3 = B[(p+1)*6+3], b4 = B[(p+1)*6+4], b5 = B[(p+1)*6+5];
            c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5;
            c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5;
            c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5;
            c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5;
            c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5;
            c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5;
        }
        // Iteration 2
        {
            float a0 = A[(p+2)*6+0], a1 = A[(p+2)*6+1], a2 = A[(p+2)*6+2];
            float a3 = A[(p+2)*6+3], a4 = A[(p+2)*6+4], a5 = A[(p+2)*6+5];
            float b0 = B[(p+2)*6+0], b1 = B[(p+2)*6+1], b2 = B[(p+2)*6+2];
            float b3 = B[(p+2)*6+3], b4 = B[(p+2)*6+4], b5 = B[(p+2)*6+5];
            c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5;
            c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5;
            c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5;
            c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5;
            c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5;
            c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5;
        }
        // Iteration 3
        {
            float a0 = A[(p+3)*6+0], a1 = A[(p+3)*6+1], a2 = A[(p+3)*6+2];
            float a3 = A[(p+3)*6+3], a4 = A[(p+3)*6+4], a5 = A[(p+3)*6+5];
            float b0 = B[(p+3)*6+0], b1 = B[(p+3)*6+1], b2 = B[(p+3)*6+2];
            float b3 = B[(p+3)*6+3], b4 = B[(p+3)*6+4], b5 = B[(p+3)*6+5];
            c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5;
            c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5;
            c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5;
            c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5;
            c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5;
            c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5;
        }
    }

    // Remainder loop (0-3 iterations)
    for (; p < K; ++p) {
        float a0 = A[p*6+0], a1 = A[p*6+1], a2 = A[p*6+2];
        float a3 = A[p*6+3], a4 = A[p*6+4], a5 = A[p*6+5];
        float b0 = B[p*6+0], b1 = B[p*6+1], b2 = B[p*6+2];
        float b3 = B[p*6+3], b4 = B[p*6+4], b5 = B[p*6+5];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5;
        c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5;
        c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5;
    }

    // Store results — branchless: beta==0 is common case, use ternary for cmov
    // E2K has no branch prediction; ternary compiles to merge (conditional move)
    if (beta == 0.0f) {
        // Fast path: pure store, no read-modify-write
        C[0*ldc+0]=alpha*c00; C[0*ldc+1]=alpha*c01; C[0*ldc+2]=alpha*c02;
        C[0*ldc+3]=alpha*c03; C[0*ldc+4]=alpha*c04; C[0*ldc+5]=alpha*c05;
        C[1*ldc+0]=alpha*c10; C[1*ldc+1]=alpha*c11; C[1*ldc+2]=alpha*c12;
        C[1*ldc+3]=alpha*c13; C[1*ldc+4]=alpha*c14; C[1*ldc+5]=alpha*c15;
        C[2*ldc+0]=alpha*c20; C[2*ldc+1]=alpha*c21; C[2*ldc+2]=alpha*c22;
        C[2*ldc+3]=alpha*c23; C[2*ldc+4]=alpha*c24; C[2*ldc+5]=alpha*c25;
        C[3*ldc+0]=alpha*c30; C[3*ldc+1]=alpha*c31; C[3*ldc+2]=alpha*c32;
        C[3*ldc+3]=alpha*c33; C[3*ldc+4]=alpha*c34; C[3*ldc+5]=alpha*c35;
        C[4*ldc+0]=alpha*c40; C[4*ldc+1]=alpha*c41; C[4*ldc+2]=alpha*c42;
        C[4*ldc+3]=alpha*c43; C[4*ldc+4]=alpha*c44; C[4*ldc+5]=alpha*c45;
        C[5*ldc+0]=alpha*c50; C[5*ldc+1]=alpha*c51; C[5*ldc+2]=alpha*c52;
        C[5*ldc+3]=alpha*c53; C[5*ldc+4]=alpha*c54; C[5*ldc+5]=alpha*c55;
    } else {
        // General case: C = alpha * acc + beta * C
        C[0*ldc+0]=alpha*c00+beta*C[0*ldc+0]; C[0*ldc+1]=alpha*c01+beta*C[0*ldc+1];
        C[0*ldc+2]=alpha*c02+beta*C[0*ldc+2]; C[0*ldc+3]=alpha*c03+beta*C[0*ldc+3];
        C[0*ldc+4]=alpha*c04+beta*C[0*ldc+4]; C[0*ldc+5]=alpha*c05+beta*C[0*ldc+5];
        C[1*ldc+0]=alpha*c10+beta*C[1*ldc+0]; C[1*ldc+1]=alpha*c11+beta*C[1*ldc+1];
        C[1*ldc+2]=alpha*c12+beta*C[1*ldc+2]; C[1*ldc+3]=alpha*c13+beta*C[1*ldc+3];
        C[1*ldc+4]=alpha*c14+beta*C[1*ldc+4]; C[1*ldc+5]=alpha*c15+beta*C[1*ldc+5];
        C[2*ldc+0]=alpha*c20+beta*C[2*ldc+0]; C[2*ldc+1]=alpha*c21+beta*C[2*ldc+1];
        C[2*ldc+2]=alpha*c22+beta*C[2*ldc+2]; C[2*ldc+3]=alpha*c23+beta*C[2*ldc+3];
        C[2*ldc+4]=alpha*c24+beta*C[2*ldc+4]; C[2*ldc+5]=alpha*c25+beta*C[2*ldc+5];
        C[3*ldc+0]=alpha*c30+beta*C[3*ldc+0]; C[3*ldc+1]=alpha*c31+beta*C[3*ldc+1];
        C[3*ldc+2]=alpha*c32+beta*C[3*ldc+2]; C[3*ldc+3]=alpha*c33+beta*C[3*ldc+3];
        C[3*ldc+4]=alpha*c34+beta*C[3*ldc+4]; C[3*ldc+5]=alpha*c35+beta*C[3*ldc+5];
        C[4*ldc+0]=alpha*c40+beta*C[4*ldc+0]; C[4*ldc+1]=alpha*c41+beta*C[4*ldc+1];
        C[4*ldc+2]=alpha*c42+beta*C[4*ldc+2]; C[4*ldc+3]=alpha*c43+beta*C[4*ldc+3];
        C[4*ldc+4]=alpha*c44+beta*C[4*ldc+4]; C[4*ldc+5]=alpha*c45+beta*C[4*ldc+5];
        C[5*ldc+0]=alpha*c50+beta*C[5*ldc+0]; C[5*ldc+1]=alpha*c51+beta*C[5*ldc+1];
        C[5*ldc+2]=alpha*c52+beta*C[5*ldc+2]; C[5*ldc+3]=alpha*c53+beta*C[5*ldc+3];
        C[5*ldc+4]=alpha*c54+beta*C[5*ldc+4]; C[5*ldc+5]=alpha*c55+beta*C[5*ldc+5];
    }
}

} // namespace kernels
} // namespace tuda
} // namespace native
} // namespace at
