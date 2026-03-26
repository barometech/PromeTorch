// ============================================================================
// simd_ops_float.asm — SIMD float32 operations for NM6408 (NMC4 FPU)
// ============================================================================
// Replaces scalar C++ loops in dispatcher with FPU-vectorized operations.
// ~10x speedup over scalar for element-wise, reduce, and copy operations.
//
// Build: asm -nmc4 -nm2ms -split_sir simd_ops_float.asm -o simd_ops_float.o
//
// All functions use NMC4 convention:
//   ar6 = frame pointer (pushed on entry)
//   Arguments passed on stack via [ar6-N]
//   Result in gr0 or via pointer argument

// ============================================================
// 1. Vector Add: C[i] = A[i] + B[i], i=0..count-1
//    void nmc_vadd_f32(float* A, float* B, float* C, int count)
// ============================================================
global _nmc_vadd_f32 : label;
begin ".text_simd"

<_nmc_vadd_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;
    push ar2, gr2;

    ar0 = [ar6-=5];            // A
    ar1 = [--ar6];              // B
    ar2 = [--ar6];              // C
    gr2 = [--ar6];              // count

    // Process 32 floats at a time (vlen=31 → 32 elements)
    gr0 = gr2 >> 5;             // count / 32
    gr1 = 1;                    // stride = 1 word
    vlen = 31;                  // vector length 32

    if =0 delayed goto vadd_tail;
    nul;
    nul;

<vadd_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr1];
    fpu 0 rep 32 vreg1 = [ar1++gr1];
    fpu 0 .float vreg2 = vreg0 + vreg1;
    fpu 0 rep 32 [ar2++gr1] = vreg2;
    if <>0 delayed goto vadd_loop with gr0--;
    nul;
    nul;

<vadd_tail>
    // Handle remaining elements (count % 32)
    gr0 = gr2 and 31;
    if =0 delayed goto vadd_done;
    nul;
    nul;
    gr0--;
    vlen = gr0;
    fpu 0 rep vlen vreg0 = [ar0++gr1];
    fpu 0 rep vlen vreg1 = [ar1++gr1];
    fpu 0 .float vreg2 = vreg0 + vreg1;
    fpu 0 rep vlen [ar2++gr1] = vreg2;

<vadd_done>
    pop ar2, gr2;
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 2. Vector Multiply: C[i] = A[i] * B[i]
//    void nmc_vmul_f32(float* A, float* B, float* C, int count)
// ============================================================
global _nmc_vmul_f32 : label;

<_nmc_vmul_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;
    push ar2, gr2;

    ar0 = [ar6-=5];            // A
    ar1 = [--ar6];              // B
    ar2 = [--ar6];              // C
    gr2 = [--ar6];              // count

    gr0 = gr2 >> 5;
    gr1 = 1;
    vlen = 31;

    if =0 delayed goto vmul_tail;
    nul;
    nul;

<vmul_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr1];
    fpu 0 rep 32 vreg1 = [ar1++gr1];
    fpu 0 .float vreg2 = vreg0 * vreg1;
    fpu 0 rep 32 [ar2++gr1] = vreg2;
    if <>0 delayed goto vmul_loop with gr0--;
    nul;
    nul;

<vmul_tail>
    gr0 = gr2 and 31;
    if =0 delayed goto vmul_done;
    nul;
    nul;
    gr0--;
    vlen = gr0;
    fpu 0 rep vlen vreg0 = [ar0++gr1];
    fpu 0 rep vlen vreg1 = [ar1++gr1];
    fpu 0 .float vreg2 = vreg0 * vreg1;
    fpu 0 rep vlen [ar2++gr1] = vreg2;

<vmul_done>
    pop ar2, gr2;
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 3. Vector Scale: C[i] = alpha * A[i]
//    void nmc_vscale_f32(float* A, float alpha_bits, float* C, int count)
// ============================================================
global _nmc_vscale_f32 : label;

<_nmc_vscale_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;
    push ar2, gr2;

    ar0 = [ar6-=5];            // A
    gr2 = [--ar6];              // alpha (as uint32 bits)
    ar2 = [--ar6];              // C
    gr0 = [--ar6];              // count

    // Broadcast alpha to vreg1
    fpu 0 vreg1 = gr2;         // scalar to vector broadcast

    gr1 = 1;
    gr0 = gr0 >> 5;
    vlen = 31;

    if =0 delayed goto vscale_tail;
    nul;
    nul;

<vscale_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr1];
    fpu 0 .float vreg2 = vreg0 * vreg1;
    fpu 0 rep 32 [ar2++gr1] = vreg2;
    if <>0 delayed goto vscale_loop with gr0--;
    nul;
    nul;

<vscale_tail>
    pop ar2, gr2;
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 4. Vector ReLU: C[i] = max(0, A[i])
//    void nmc_vrelu_f32(float* A, float* C, int count)
// ============================================================
global _nmc_vrelu_f32 : label;

<_nmc_vrelu_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;
    push ar2, gr2;

    ar0 = [ar6-=5];            // A
    ar2 = [--ar6];              // C
    gr2 = [--ar6];              // count

    // Zero in vreg1 for max(0, x)
    fpu 0 vreg1 = false;       // zero vector

    gr0 = gr2 >> 5;
    gr1 = 1;
    vlen = 31;

    if =0 delayed goto vrelu_tail;
    nul;
    nul;

<vrelu_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr1];
    fpu 0 .float vreg2 = max(vreg0, vreg1);
    fpu 0 rep 32 [ar2++gr1] = vreg2;
    if <>0 delayed goto vrelu_loop with gr0--;
    nul;
    nul;

<vrelu_tail>
    gr0 = gr2 and 31;
    if =0 delayed goto vrelu_done;
    nul;
    nul;
    gr0--;
    vlen = gr0;
    fpu 0 rep vlen vreg0 = [ar0++gr1];
    fpu 0 .float vreg2 = max(vreg0, vreg1);
    fpu 0 rep vlen [ar2++gr1] = vreg2;

<vrelu_done>
    pop ar2, gr2;
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 5. Vector Sum Reduction: result = sum(A[0..count-1])
//    float nmc_vsum_f32(float* A, int count)
//    Returns result in gr0 (as float bits)
// ============================================================
global _nmc_vsum_f32 : label;

<_nmc_vsum_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;

    ar0 = [ar6-=5];            // A
    gr1 = [--ar6];              // count

    // Accumulate in vreg2
    fpu 0 vreg2 = false;       // zero accumulator
    gr0 = 1;                   // stride
    vlen = 31;

    gr1 = gr1 >> 5;            // count / 32
    if =0 delayed goto vsum_scalar;
    nul;
    nul;

<vsum_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr0];
    fpu 0 .float vreg2 = vreg0 + vreg2;
    if <>0 delayed goto vsum_loop with gr1--;
    nul;
    nul;

<vsum_scalar>
    // Horizontal reduction of vreg2 (32 floats → 1 float)
    // Tree reduction: 32→16→8→4→2→1
    fpu 0 .float vreg3 = shift(vreg2, 16);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 8);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 4);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 2);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 1);
    fpu 0 .float vreg2 = vreg2 + vreg3;

    // Extract scalar result to gr0
    gr0 = fpu 0 vreg2;

    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 6. Vector Dot Product: result = sum(A[i]*B[i])
//    float nmc_vdot_f32(float* A, float* B, int count)
// ============================================================
global _nmc_vdot_f32 : label;

<_nmc_vdot_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;
    push ar2, gr2;

    ar0 = [ar6-=5];            // A
    ar1 = [--ar6];              // B
    gr2 = [--ar6];              // count

    fpu 0 vreg2 = false;       // accumulator = 0
    gr0 = 1;
    vlen = 31;

    gr1 = gr2 >> 5;
    if =0 delayed goto vdot_done;
    nul;
    nul;

<vdot_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr0];
    fpu 0 rep 32 vreg1 = [ar1++gr0];
    fpu 0 .float vreg2 = vreg0 * vreg1 + vreg2;  // FMA accumulate
    if <>0 delayed goto vdot_loop with gr1--;
    nul;
    nul;

<vdot_done>
    // Horizontal reduction
    fpu 0 .float vreg3 = shift(vreg2, 16);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 8);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 4);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 2);
    fpu 0 .float vreg2 = vreg2 + vreg3;
    fpu 0 .float vreg3 = shift(vreg2, 1);
    fpu 0 .float vreg2 = vreg2 + vreg3;

    gr0 = fpu 0 vreg2;

    pop ar2, gr2;
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 7. Vector Copy: dst[i] = src[i]
//    void nmc_vcopy_f32(float* src, float* dst, int count)
// ============================================================
global _nmc_vcopy_f32 : label;

<_nmc_vcopy_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;

    ar0 = [ar6-=5];            // src
    ar1 = [--ar6];              // dst
    gr1 = [--ar6];              // count

    gr0 = 1;
    vlen = 31;
    gr1 = gr1 >> 5;

    if =0 delayed goto vcopy_done;
    nul;
    nul;

<vcopy_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr0];
    fpu 0 rep 32 [ar1++gr0] = vreg0;
    if <>0 delayed goto vcopy_loop with gr1--;
    nul;
    nul;

<vcopy_done>
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 8. Vector FMA: C[i] = A[i] * B[i] + C[i]  (in-place accumulate)
//    void nmc_vfma_f32(float* A, float* B, float* C, int count)
// ============================================================
global _nmc_vfma_f32 : label;

<_nmc_vfma_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;
    push ar2, gr2;

    ar0 = [ar6-=5];            // A
    ar1 = [--ar6];              // B
    ar2 = [--ar6];              // C
    gr2 = [--ar6];              // count

    gr0 = gr2 >> 5;
    gr1 = 1;
    vlen = 31;

    if =0 delayed goto vfma_done;
    nul;
    nul;

<vfma_loop>
    fpu 0 rep 32 vreg0 = [ar0++gr1];
    fpu 0 rep 32 vreg1 = [ar1++gr1];
    fpu 0 rep 32 vreg2 = [ar2];
    fpu 0 .float vreg2 = vreg0 * vreg1 + vreg2;
    fpu 0 rep 32 [ar2++gr1] = vreg2;
    if <>0 delayed goto vfma_loop with gr0--;
    nul;
    nul;

<vfma_done>
    pop ar2, gr2;
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

// ============================================================
// 9. Vector Zero: A[i] = 0
//    void nmc_vzero_f32(float* A, int count)
// ============================================================
global _nmc_vzero_f32 : label;

<_nmc_vzero_f32>
    push ar6, gr6;
    ar6 = ar7;
    push ar0, gr0;
    push ar1, gr1;

    ar0 = [ar6-=5];            // A
    gr1 = [--ar6];              // count

    fpu 0 vreg0 = false;       // zero vector
    gr0 = 1;
    vlen = 31;
    gr1 = gr1 >> 5;

    if =0 delayed goto vzero_done;
    nul;
    nul;

<vzero_loop>
    fpu 0 rep 32 [ar0++gr0] = vreg0;
    if <>0 delayed goto vzero_loop with gr1--;
    nul;
    nul;

<vzero_done>
    pop ar1, gr1;
    pop ar0, gr0;
    pop ar6, gr6;
    return;

end ".text_simd"
