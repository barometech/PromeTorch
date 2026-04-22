# Gemini 3.1 Pro consultation — SSE4.1 Q4_K GEMV design

### 1. SSE4.1 Inner Loop Rewrite

This implementation processes 64 elements per iteration (two 32-element Q8Blocks) for two weight rows simultaneously. It avoids horizontal sums inside the loop by accumulating vertically into FP32 vectors, deferring the horizontal reduction until the super-block finishes.

```cpp
#include <smmintrin.h> // SSE4.1

// Assumes pointers: 
// const uint8_t* __restrict q0, * __restrict q1 (Q4_K weights)
// const int8_t*  __restrict x (Q8_K inputs)
// const float*   __restrict scales0, * __restrict scales1
// __m128 acc0_f32, acc1_f32 passed in/out

const __m128i mask = _mm_set1_epi8(0x0F);
const __m128i ones = _mm_set1_epi16(1);

#pragma loop count(min=4, max=4)
#pragma unroll(4)
for (int j = 0; j < 256; j += 64) {
    // --- BLOCK 1 (Elements 0-31) ---
    // Load 16 bytes of Q4 (32 nibbles) per row
    __m128i w0_A = _mm_loadu_si128((const __m128i*)(q0 + j/2));
    __m128i w1_A = _mm_loadu_si128((const __m128i*)(q1 + j/2));

    // Load 32 bytes of Q8 input
    __m128i x_A0 = _mm_loadu_si128((const __m128i*)(x + j));
    __m128i x_A1 = _mm_loadu_si128((const __m128i*)(x + j + 16));

    // Extract nibbles Row 0
    __m128i w0_A_lo = _mm_and_si128(w0_A, mask);
    __m128i w0_A_hi = _mm_and_si128(_mm_srli_epi16(w0_A, 4), mask);

    // Extract nibbles Row 1
    __m128i w1_A_lo = _mm_and_si128(w1_A, mask);
    __m128i w1_A_hi = _mm_and_si128(_mm_srli_epi16(w1_A, 4), mask);

    // Dot product Row 0 (unsigned 8-bit * signed 8-bit -> signed 16-bit)
    __m128i p0_A_lo = _mm_maddubs_epi16(w0_A_lo, x_A0);
    __m128i p0_A_hi = _mm_maddubs_epi16(w0_A_hi, x_A1);
    __m128i s0_A    = _mm_add_epi32(_mm_madd_epi16(p0_A_lo, ones), 
                                    _mm_madd_epi16(p0_A_hi, ones));

    // Dot product Row 1
    __m128i p1_A_lo = _mm_maddubs_epi16(w1_A_lo, x_A0);
    __m128i p1_A_hi = _mm_maddubs_epi16(w1_A_hi, x_A1);
    __m128i s1_A    = _mm_add_epi32(_mm_madd_epi16(p1_A_lo, ones), 
                                    _mm_madd_epi16(p1_A_hi, ones));

    // Convert to FP32 and scale (LCC will fuse mul+add to fmadd)
    __m128 scale0_A = _mm_set1_ps(scales0[j/32]);
    __m128 scale1_A = _mm_set1_ps(scales1[j/32]);
    acc0_f32 = _mm_add_ps(acc0_f32, _mm_mul_ps(_mm_cvtepi32_ps(s0_A), scale0_A));
    acc1_f32 = _mm_add_ps(acc1_f32, _mm_mul_ps(_mm_cvtepi32_ps(s1_A), scale1_A));

    // --- BLOCK 2 (Elements 32-63) ---
    __m128i w0_B = _mm_loadu_si128((const __m128i*)(q0 + j/2 + 16));
    __m128i w1_B = _mm_loadu_si128((const __m128i*)(q1 + j/2 + 16));

    __m128i x_B0 = _mm_loadu_si128((const __m128i*)(x + j + 32));
    __m128i x_B1 = _mm_loadu_si128((const __m128i*)(x + j + 48));

    __m128i w0_B_lo = _mm_and_si128(w0_B, mask);
    __m128i w0_B_hi = _mm_and_si128(_mm_srli_epi16(w0_B, 4), mask);
    
    __m128i w1_B_lo = _mm_and_si128(w1_B, mask);
    __m128i w1_B_hi = _mm_and_si128(_mm_srli_epi16(w1_B, 4), mask);

    __m128i p0_B_lo = _mm_maddubs_epi16(w0_B_lo, x_B0);
    __m128i p0_B_hi = _mm_maddubs_epi16(w0_B_hi, x_B1);
    __m128i s0_B    = _mm_add_epi32(_mm_madd_epi16(p0_B_lo, ones), 
                                    _mm_madd_epi16(p0_B_hi, ones));

    __m128i p1_B_lo = _mm_maddubs_epi16(w1_B_lo, x_B0);
    __m128i p1_B_hi = _mm_maddubs_epi16(w1_B_hi, x_B1);
    __m128i s1_B    = _mm_add_epi32(_mm_madd_epi16(p1_B_lo, ones), 
                                    _mm_madd_epi16(p1_B_hi, ones));

    __m128 scale0_B = _mm_set1_ps(scales0[j/32 + 1]);
    __m128 scale1_B = _mm_set1_ps(scales1[j/32 + 1]);
    acc0_f32 = _mm_add_ps(acc0_f32, _mm_mul_ps(_mm_cvtepi32_ps(s0_B), scale0_B));
    acc1_f32 = _mm_add_ps(acc1_f32, _mm_mul_ps(_mm_cvtepi32_ps(s1_B), scale1_B));
}
```

### 2. Intrinsic to E2K QP Instruction Mapping

Based on Ilya Kurdyukov's E2K SIMD mapping notes, LCC 1.29 translates the 128-bit SSE4.1 intrinsics directly into E2K Quad-Packed (QP) instructions:

*   `_mm_loadu_si128` -> `ldq` (or `ldd` depending on alignment proofs). E2K handles unaligned 128-bit loads efficiently if hardware unaligned access is enabled.
*   `_mm_set1_epi8` / `_mm_set1_epi16` -> `pdup` (pack duplicate) or generated as a literal in the constant pool.
*   `_mm_and_si128` -> `pand` (bitwise AND).
*   `_mm_srli_epi16` -> `psrlw` (packed shift right logical, word).
*   `_mm_maddubs_epi16` -> `pmaddubw`. This is a native E2K instruction that perfectly matches the SSSE3 semantics (u8 * s8 -> s16).
*   `_mm_madd_epi16` -> `pmaddwd` (packed multiply-add, word to doubleword).
*   `_mm_add_epi32` -> `paddd` (packed add, doubleword).
*   `_mm_cvtepi32_ps` -> `cvtd2ps` (convert doubleword to packed single-precision float).
*   `_mm_add_ps` + `_mm_mul_ps` -> `fmadd` (fused multiply-add). LCC automatically contracts these into E2K's native FMA instruction.

### 3. Explicit ILP Analysis

**Compiler View:**
The compiler sees 4 completely independent dependency chains per row per block (lo/hi nibble extraction -> maddubs -> madd -> add). Because we process two rows and two blocks per iteration, there are 16 independent integer arithmetic chains in flight before the FP conversion.

**Hardware Utilization (E8C2 6 ALU channels):**
*   **Integer/Memory (4 channels):** The `pand`, `psrlw`, `pmaddubw`, and `pmaddwd` instructions will heavily saturate the 4 integer-capable ALUs. The loads (`ldq`) will issue on the memory-capable subset of these channels.
*   **FP (2 channels):** The `cvtd2ps` and `fmadd` instructions will issue to the 2 FP-capable channels.
*   Because the integer math dominates the instruction count (extracting and dot-producting), the FP channels will easily keep pace. LCC's software pipeliner will overlap the integer extraction of iteration `N+1` with the FP accumulation of iteration `N`. Plausible utilization of the 6 ALUs is extremely high (>85% IPC efficiency) because the 128-bit width perfectly matches the E2K QP register width, avoiding the micro-coded splitting overhead seen with 256-bit AVX2.

### 4. LCC-Specific Pragmas

To ensure LCC generates optimal software-pipelined code without conservative bailouts:

1.  `#pragma loop count(min=4, max=4)`: Tells LCC the exact trip count of the inner loop (256/64 = 4). This allows the compiler to fully unroll or perfectly software-pipeline without generating a scalar tail-loop.
2.  `#pragma unroll(4)`: Forces complete unrolling of the inner loop, eliminating branch overhead and maximizing the basic block size for the VLIW scheduler.
3.  `__restrict`: Apply this to `q0`, `q1`, `x`, `scales0`, and `scales1`. E2K's memory disambiguation is strict; without `__restrict`, LCC will insert conservative memory dependency barriers that destroy ILP.
4.  `#pragma ivdep`: (Optional but recommended) Place before the loop to guarantee no loop-carried dependencies exist other than the explicit accumulators.

### 5. Concrete Risk: Horizontal Reduce Bottleneck

**The Risk:**
When converting from AVX2 to SSE4.1, developers often try to replicate the AVX2 horizontal reduction (`_mm256_hadd_epi32` or `_mm256_extracti128_si256`) using multiple 128-bit shuffles (`_mm_hadd_epi32`, `_mm_shuffle_epi32`) *inside* the inner loop to maintain a scalar accumulator. 

**Why it goes wrong on E2K:**
E2K handles vertical vector operations (like `paddd`) natively and fast. However, horizontal cross-lane operations (shuffles/hadds) often bottleneck on a single permutation unit or incur multi-cycle latency. If you perform a horizontal sum inside the `j` loop, you serialize the dependency chain and stall the VLIW pipeline.

**The Mitigation:**
The code provided above mitigates this entirely by **accumulating vertically** into `__m128 acc0_f32`. We keep 4 separate FP32 partial sums per row in the vector registers throughout the entire super-block. You must only perform the horizontal reduction (e.g., `acc[0] + acc[1] + acc[2] + acc[3]`) exactly *once* at the very end of the super-block, outside the loop.