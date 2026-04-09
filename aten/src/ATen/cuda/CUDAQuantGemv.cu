// ============================================================================
// Fused Quantized Dequant-GEMV Kernels for PromeTorch (v3)
// ============================================================================
// Warp-cooperative design: each warp handles one output row.
// 32 threads read consecutive qs bytes → perfect coalescing.
// x vector loaded into shared memory for zero-latency reuse.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "aten/src/ATen/cuda/CUDAOps.h"
#include "aten/src/ATen/cuda/CuBLASHandle.h"

// AMD HIP compatibility
#ifdef __HIP_PLATFORM_AMD__
constexpr unsigned WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;
#else
constexpr unsigned WARP_MASK = 0xFFFFFFFF;
#endif

namespace at {
namespace cuda {

// ============================================================================
// Q8_1 block structure (matching llama.cpp for dp4a GEMV)
// ============================================================================
struct block_q8_1 {
    half2 ds;        // d (scale) and sum, packed as half2
    int8_t qs[32];   // quantized int8 values
};
// sizeof(block_q8_1) = 36 bytes

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float fp16_to_fp32_device(uint16_t h) {
    __half hval;
    memcpy(&hval, &h, sizeof(__half));
    return __half2float(hval);
}

__device__ __forceinline__ void get_scale_min_k4_device(
    int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ============================================================================
// Q4_K_M GEMV — FP16 hfma2 version for A100+ (4x faster than FP32 __dp4a)
// ============================================================================
// Same coalesced access pattern, but uses half2 + __hfma2 instead of float.
// On A100: FP16 ALU = 78 TFLOPS vs FP32 = 19.5 TFLOPS = 4x throughput.
// X stored in shared memory as FP16 (half the bandwidth of FP32 smem).

__global__ void q4km_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int K, int N,
    int64_t row_stride_bytes)
{
    extern __shared__ half x_shared_h[];  // FP16 shared memory

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int n = blockIdx.x * warps_per_block + warp_id;

    // Load x into shared memory as FP16 (convert on the fly)
    for (int k = tid; k < K; k += block_size) {
        x_shared_h[k] = __float2half(x[k]);
    }
    __syncthreads();

    if (n >= N) return;

    const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
    const int num_blocks_per_row = K / 256;

    const int group = lane / 8;           // 0-3
    const int pos = (lane & 7) * 4;       // 0,4,8,...,28

    float sum = 0.0f;

    for (int blk = 0; blk < num_blocks_per_row; ++blk) {
        const uint8_t* bp = row_data + blk * 144;

        // Read d, dmin as native FP16
        half d_h = *reinterpret_cast<const half*>(bp);
        half dmin_h = *reinterpret_cast<const half*>(bp + 2);

        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
        get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

        // FIX: __ldg forces L2 cache read (was memcpy = scalar loads)
        uint32_t qs4 = __ldg(reinterpret_cast<const uint32_t*>(bp + 16 + lane * 4));

        // Compute scale/min as half2 for hfma2
        half dl_h = __hmul(d_h, __short2half_rn(sc_lo));
        half ml_h = __hmul(dmin_h, __short2half_rn(m_lo));
        half dh_h = __hmul(d_h, __short2half_rn(sc_hi));
        half mh_h = __hmul(dmin_h, __short2half_rn(m_hi));

        // x from shared memory as half2 (vectorized)
        const int k_base = blk * 256 + group * 64 + pos;
        const half2* xp_lo = reinterpret_cast<const half2*>(&x_shared_h[k_base]);
        const half2* xp_hi = reinterpret_cast<const half2*>(&x_shared_h[k_base + 32]);

        // Dequant + dot product in FP16 using __hfma2
        // Low nibbles (4 weights): q[0..3] * dl - ml
        half2 w01 = __halves2half2(
            __hsub(__hmul(__short2half_rn(qs4 & 0xF), dl_h), ml_h),
            __hsub(__hmul(__short2half_rn((qs4 >> 8) & 0xF), dl_h), ml_h));
        half2 w23 = __halves2half2(
            __hsub(__hmul(__short2half_rn((qs4 >> 16) & 0xF), dl_h), ml_h),
            __hsub(__hmul(__short2half_rn((qs4 >> 24) & 0xF), dl_h), ml_h));
        // High nibbles (4 weights): q[4..7] * dh - mh
        half2 w45 = __halves2half2(
            __hsub(__hmul(__short2half_rn((qs4 >> 4) & 0xF), dh_h), mh_h),
            __hsub(__hmul(__short2half_rn((qs4 >> 12) & 0xF), dh_h), mh_h));
        half2 w67 = __halves2half2(
            __hsub(__hmul(__short2half_rn((qs4 >> 20) & 0xF), dh_h), mh_h),
            __hsub(__hmul(__short2half_rn((qs4 >> 28) & 0xF), dh_h), mh_h));

        // Dot product via __hfma2 (2 FMA per cycle on FP16 ALU)
        half2 acc = __float2half2_rn(0.0f);
        acc = __hfma2(w01, xp_lo[0], acc);
        acc = __hfma2(w23, xp_lo[1], acc);
        acc = __hfma2(w45, xp_hi[0], acc);
        acc = __hfma2(w67, xp_hi[1], acc);

        sum += __half2float(__low2half(acc)) + __half2float(__high2half(acc));
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(WARP_MASK, sum, offset);
    }
    if (lane == 0) y[n] = sum;
}

// ============================================================================
// Persistent Q4_K GEMV — Load x once, grid-stride over rows
// ============================================================================
// Launches one block per SM. Each block loads x into shared memory ONCE,
// then loops over multiple rows. Eliminates redundant x loading.

// Q4_K_M × Q8_1 GEMV: dp4a INT8 dot product (llama.cpp approach)
// x pre-quantized to Q8_1 in shared memory → 4 MACs per __dp4a instruction
// Less registers → more warps → better memory latency hiding → higher DRAM utilization
__global__ void q4km_persistent_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int K, int N,
    int64_t row_stride_bytes)
{
    // Shared memory: Q8_1 quantized x [K int8] + scales [K/32 float] + sums [K/32 float]
    extern __shared__ char smem_raw[];
    const int num_q8_blocks = K / 32;  // Q8_1: 32 elements per block
    int8_t* x_q8 = reinterpret_cast<int8_t*>(smem_raw);
    float* x_q8_d = reinterpret_cast<float*>(smem_raw + K);        // scale per 32 elements
    float* x_q8_s = reinterpret_cast<float*>(smem_raw + K + num_q8_blocks * 4);  // sum per 32

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // Quantize x to Q8_1 in shared memory (cooperative, all threads)
    for (int blk = tid; blk < num_q8_blocks; blk += block_size) {
        const float* xb = x + blk * 32;
        // Find max abs value in this block of 32
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(xb[j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        float sum = 0.0f;
        int8_t* qb = x_q8 + blk * 32;
        for (int j = 0; j < 32; j++) {
            int8_t q = (int8_t)roundf(xb[j] * id);
            qb[j] = q;
            sum += (float)q;
        }
        x_q8_d[blk] = d;
        x_q8_s[blk] = sum;
    }
    __syncthreads();

    const int group = lane / 8;
    const int pos_in_group = lane & 7;
    const int num_blocks_per_row = K / 256;

    // Grid-stride: 1 warp per row, 8 rows per block
    for (int base_n = blockIdx.x * warps_per_block; base_n < N;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N) continue;

        const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
        float sum = 0.0f;

        // SOFTWARE PIPELINING: prefetch block N+1 while computing block N
        // Process 2 blocks per iteration for better compute/memory overlap
        const uint8_t* bp0 = row_data;
        // Prefetch first block
        uint32_t pf_d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp0));
        uint32_t pf_qs4 = __ldg(reinterpret_cast<const uint32_t*>(bp0 + 16 + lane * 4));

        for (int blk = 0; blk < num_blocks_per_row; blk += 2) {
            // === BLOCK A (blk): use prefetched data ===
            const uint8_t* bpA = row_data + blk * 144;
            uint32_t d_dminA = pf_d_dmin;
            uint32_t qs4A = pf_qs4;

            // Prefetch BLOCK B (blk+1) while computing A
            const uint8_t* bpB = bpA + 144;
            uint32_t d_dminB = 0, qs4B = 0;
            bool has_B = (blk + 1 < num_blocks_per_row);
            if (has_B) {
                d_dminB = __ldg(reinterpret_cast<const uint32_t*>(bpB));
                qs4B = __ldg(reinterpret_cast<const uint32_t*>(bpB + 16 + lane * 4));
            }

            // Compute BLOCK A
            {
                float dA = fp16_to_fp32_device(d_dminA & 0xFFFF);
                float dmA = fp16_to_fp32_device(d_dminA >> 16);
                uint8_t sc_lo, m_lo, sc_hi, m_hi;
                get_scale_min_k4_device(group * 2, bpA + 4, &sc_lo, &m_lo);
                get_scale_min_k4_device(group * 2 + 1, bpA + 4, &sc_hi, &m_hi);

                uint32_t v_lo = qs4A & 0x0F0F0F0F;          // low nibbles of 4 bytes
                uint32_t v_hi = (qs4A >> 4) & 0x0F0F0F0F; // high nibbles of 4 bytes

                int q8A_lo = blk * 8 + group * 2;
                int q8A_hi = q8A_lo + 1;
                int k_off = pos_in_group * 4;
                uint32_t u_lo = *reinterpret_cast<const uint32_t*>(&x_q8[q8A_lo * 32 + k_off]);
                uint32_t u_hi = *reinterpret_cast<const uint32_t*>(&x_q8[q8A_hi * 32 + k_off]);

                int dot_lo = __dp4a((int)v_lo, (int)u_lo, (int)0);
                int dot_hi = __dp4a((int)v_hi, (int)u_hi, (int)0);
                int sum_lo = __dp4a((int)0x01010101, (int)u_lo, (int)0);
                int sum_hi = __dp4a((int)0x01010101, (int)u_hi, (int)0);

                sum += dA * sc_lo * x_q8_d[q8A_lo] * (float)dot_lo - dmA * m_lo * x_q8_d[q8A_lo] * (float)sum_lo;
                sum += dA * sc_hi * x_q8_d[q8A_hi] * (float)dot_hi - dmA * m_hi * x_q8_d[q8A_hi] * (float)sum_hi;
            }

            // Prefetch BLOCK C (blk+2) for next iteration
            if (blk + 2 < num_blocks_per_row) {
                const uint8_t* bpC = row_data + (blk + 2) * 144;
                pf_d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bpC));
                pf_qs4 = __ldg(reinterpret_cast<const uint32_t*>(bpC + 16 + lane * 4));
            }

            // Compute BLOCK B (if exists)
            if (has_B) {
                float dB = fp16_to_fp32_device(d_dminB & 0xFFFF);
                float dmB = fp16_to_fp32_device(d_dminB >> 16);
                uint8_t sc_lo, m_lo, sc_hi, m_hi;
                get_scale_min_k4_device(group * 2, bpB + 4, &sc_lo, &m_lo);
                get_scale_min_k4_device(group * 2 + 1, bpB + 4, &sc_hi, &m_hi);

                uint32_t v_lo = qs4B & 0x0F0F0F0F;          // low nibbles of 4 bytes
                uint32_t v_hi = (qs4B >> 4) & 0x0F0F0F0F; // high nibbles of 4 bytes

                int q8B_lo = (blk+1) * 8 + group * 2;
                int q8B_hi = q8B_lo + 1;
                int k_off = pos_in_group * 4;
                uint32_t u_lo = *reinterpret_cast<const uint32_t*>(&x_q8[q8B_lo * 32 + k_off]);
                uint32_t u_hi = *reinterpret_cast<const uint32_t*>(&x_q8[q8B_hi * 32 + k_off]);

                int dot_lo = __dp4a((int)v_lo, (int)u_lo, (int)0);
                int dot_hi = __dp4a((int)v_hi, (int)u_hi, (int)0);
                int sum_lo = __dp4a((int)0x01010101, (int)u_lo, (int)0);
                int sum_hi = __dp4a((int)0x01010101, (int)u_hi, (int)0);

                sum += dB * sc_lo * x_q8_d[q8B_lo] * (float)dot_lo - dmB * m_lo * x_q8_d[q8B_lo] * (float)sum_lo;
                sum += dB * sc_hi * x_q8_d[q8B_hi] * (float)dot_hi - dmB * m_hi * x_q8_d[q8B_hi] * (float)sum_hi;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y[n] = sum;
    }
}

ATEN_CUDA_API void launch_q4km_gemv(
    const void* weights,
    const float* x,
    float* y,
    int K, int N,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = (N + WARPS - 1) / WARPS;
    int smem_bytes = K * sizeof(half);  // FP16 smem (was float = 2x smaller now)

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weights), x, y,
        K, N, row_stride_bytes);
}

// ============================================================================
// Persistent GEMV launcher — one block per SM, grid-stride over rows
// ============================================================================

ATEN_CUDA_API void launch_q4km_persistent_gemv(
    const void* weights,
    const float* x,
    float* y,
    int K, int N,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    // Query SM count for optimal grid size
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;  // 256 threads per block
    // Launch 2 blocks per SM for latency hiding
    int grid = sm_count * 4;
    int num_q8 = K / 32;
    int smem_bytes = K * 1 + num_q8 * 4 + num_q8 * 4;  // Q8 int8[K] + scales[K/32] + sums[K/32]

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_persistent_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_persistent_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weights), x, y,
        K, N, row_stride_bytes);
}

// ============================================================================
// Fused gate+up GEMV — two GEMVs in a single kernel launch
// ============================================================================
// Eliminates one kernel launch per layer. Both projections share the same
// input vector x, which is loaded into shared memory once.

__global__ void q4km_fused_gate_up_kernel(
    const uint8_t* __restrict__ w_gate,
    const uint8_t* __restrict__ w_up,
    const float* __restrict__ x,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes)
{
    extern __shared__ float x_shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // Load x into shared memory ONCE
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    const int group = lane / 8;
    const int pos = (lane & 7) * 4;
    const int num_blocks_per_row = K / 256;
    const int N_total = N_gate + N_up;

    // Grid-stride loop over both gate and up rows
    for (int base_n = blockIdx.x * warps_per_block; base_n < N_total;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N_total) continue;

        // Determine which weight matrix and output buffer
        const uint8_t* weights;
        float* y_out;
        int row_idx;
        if (n < N_gate) {
            weights = w_gate;
            y_out = y_gate;
            row_idx = n;
        } else {
            weights = w_up;
            y_out = y_up;
            row_idx = n - N_gate;
        }

        const uint8_t* row_data = weights + (int64_t)row_idx * row_stride_bytes;
        float sum = 0.0f;

        for (int blk = 0; blk < num_blocks_per_row; ++blk) {
            const uint8_t* bp = row_data + blk * 144;

            // Vectorized d/dmin read via __ldg (1 transaction instead of 2 memcpy)
            uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp));
            const float d = fp16_to_fp32_device(d_dmin & 0xFFFF);
            const float dm = fp16_to_fp32_device(d_dmin >> 16);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

            uint32_t qs4;
            memcpy(&qs4, bp + 16 + lane * 4, 4);

            float dl = d * sc_lo, ml = dm * m_lo;
            float dh = d * sc_hi, mh = dm * m_hi;

            const int k_base = blk * 256 + group * 64 + pos;
            const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_base]);
            const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_base + 32]);

            sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;
            sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
            sum += (dl * (float)((qs4 >> 16) & 0xF) - ml) * x_lo.z;
            sum += (dl * (float)((qs4 >> 24) & 0xF) - ml) * x_lo.w;
            sum += (dh * (float)((qs4 >>  4) & 0xF) - mh) * x_hi.x;
            sum += (dh * (float)((qs4 >> 12) & 0xF) - mh) * x_hi.y;
            sum += (dh * (float)((qs4 >> 20) & 0xF) - mh) * x_hi.z;
            sum += (dh * (float)((qs4 >> 28) & 0xF) - mh) * x_hi.w;
        }

        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y_out[row_idx] = sum;
    }
}

ATEN_CUDA_API void launch_q4km_fused_gate_up_gemv(
    const void* w_gate, const void* w_up,
    const float* x, float* y_gate, float* y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = sm_count * 4;  // 4x occupancy for A100 bandwidth saturation
    int smem_bytes = K * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_fused_gate_up_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_fused_gate_up_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(w_gate),
        static_cast<const uint8_t*>(w_up),
        x, y_gate, y_up,
        K, N_gate, N_up,
        row_stride_bytes);
}

// ============================================================================
// Q8_1 Quantization Kernel — Quantize float32 x to int8 for dp4a
// ============================================================================
// Each warp (32 threads) quantizes one Q8_1 block of 32 values.
// Stores scale d and quantized int8 values.

__global__ void quantize_q8_1_kernel(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    int K)
{
    const int block_id = blockIdx.x;
    const int lane = threadIdx.x;  // 0..31
    const int idx = block_id * 32 + lane;

    float xi = (idx < K) ? x[idx] : 0.0f;

    // Warp max abs for scale computation
    float amax = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(WARP_MASK, amax, offset));

    // Quantize to int8
    float d = amax / 127.0f;
    int8_t q = (d > 0.0f) ? static_cast<int8_t>(__float2int_rn(xi / d)) : 0;

    y[block_id].qs[lane] = q;
    if (lane == 0) {
        y[block_id].ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

ATEN_CUDA_API void launch_quantize_q8_1(
    const float* x, void* y_q8, int K, cudaStream_t stream)
{
    int num_blocks = (K + 31) / 32;
    quantize_q8_1_kernel<<<num_blocks, 32, 0, stream>>>(
        x, static_cast<block_q8_1*>(y_q8), K);
}

// ============================================================================
// Q4_K × Q8_1 GEMV — dp4a integer dot product, 4 warps per output row
// ============================================================================
// Key innovation from llama.cpp:
// - x is pre-quantized to Q8_1 (int8), enabling __dp4a (4 multiply-adds/cycle)
// - No shared memory needed (Q8_1 data is tiny, cached in L2)
// - 4 warps cooperate on K-dimension reduction for each output row
// - More thread blocks → better SM utilization and memory-level parallelism

__global__ void q4km_q8_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const block_q8_1* __restrict__ x_q8,
    float* __restrict__ y,
    int K, int N,
    int64_t row_stride_bytes)
{
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;  // 0..127
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
    const int num_blocks = K / 256;

    const int group = lane / 8;        // 0-3
    const int pos = (lane & 7) * 4;    // 0,4,...,28

    float sum = 0.0f;

    // 4 warps stride across Q4_K blocks, unrolled ×2 for better latency hiding
    // Each iteration: process blk and blk+4 simultaneously (8 blocks/iter total)
    for (int blk = warp_id; blk < num_blocks; blk += 8) {
        // ---- Block A (blk) ----
        const uint8_t* bpA = row_data + blk * 144;
        uint32_t qs4A;
        memcpy(&qs4A, bpA + 16 + lane * 4, 4);  // issue load early

        // ---- Block B (blk+4) — pre-load while A is in flight ----
        const int blkB = blk + 4;
        const bool validB = (blkB < num_blocks);
        const uint8_t* bpB = validB ? row_data + blkB * 144 : bpA;
        uint32_t qs4B;
        memcpy(&qs4B, bpB + 16 + lane * 4, 4);  // second load overlaps first

        // ---- Process block A ----
        {
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, bpA, 2);
            memcpy(&dmin_bits, bpA + 2, 2);
            float d_q4 = fp16_to_fp32_device(d_bits);
            float dm_q4 = fp16_to_fp32_device(dmin_bits);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bpA + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bpA + 4, &sc_hi, &m_hi);

            int v_lo = (int)(qs4A & 0x0F0F0F0F);
            int v_hi = (int)((qs4A >> 4) & 0x0F0F0F0F);

            const int q8A_lo = blk * 8 + group * 2;
            const int q8A_hi = q8A_lo + 1;
            float d8_lo = __half2float(*reinterpret_cast<const __half*>(&x_q8[q8A_lo].ds));
            float d8_hi = __half2float(*reinterpret_cast<const __half*>(&x_q8[q8A_hi].ds));
            int u_lo, u_hi;
            memcpy(&u_lo, &x_q8[q8A_lo].qs[pos], 4);
            memcpy(&u_hi, &x_q8[q8A_hi].qs[pos], 4);

            int dot_lo = __dp4a(v_lo, u_lo, 0);
            int dot_hi = __dp4a(v_hi, u_hi, 0);
            int sum_lo = __dp4a(0x01010101, u_lo, 0);
            int sum_hi = __dp4a(0x01010101, u_hi, 0);

            sum += d_q4 * (float)sc_lo * d8_lo * (float)dot_lo
                 - dm_q4 * (float)m_lo * d8_lo * (float)sum_lo;
            sum += d_q4 * (float)sc_hi * d8_hi * (float)dot_hi
                 - dm_q4 * (float)m_hi * d8_hi * (float)sum_hi;
        }

        // ---- Process block B (if valid) ----
        if (validB) {
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, bpB, 2);
            memcpy(&dmin_bits, bpB + 2, 2);
            float d_q4 = fp16_to_fp32_device(d_bits);
            float dm_q4 = fp16_to_fp32_device(dmin_bits);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bpB + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bpB + 4, &sc_hi, &m_hi);

            int v_lo = (int)(qs4B & 0x0F0F0F0F);
            int v_hi = (int)((qs4B >> 4) & 0x0F0F0F0F);

            const int q8B_lo = blkB * 8 + group * 2;
            const int q8B_hi = q8B_lo + 1;
            float d8_lo = __half2float(*reinterpret_cast<const __half*>(&x_q8[q8B_lo].ds));
            float d8_hi = __half2float(*reinterpret_cast<const __half*>(&x_q8[q8B_hi].ds));
            int u_lo, u_hi;
            memcpy(&u_lo, &x_q8[q8B_lo].qs[pos], 4);
            memcpy(&u_hi, &x_q8[q8B_hi].qs[pos], 4);

            int dot_lo = __dp4a(v_lo, u_lo, 0);
            int dot_hi = __dp4a(v_hi, u_hi, 0);
            int sum_lo = __dp4a(0x01010101, u_lo, 0);
            int sum_hi = __dp4a(0x01010101, u_hi, 0);

            sum += d_q4 * (float)sc_lo * d8_lo * (float)dot_lo
                 - dm_q4 * (float)m_lo * d8_lo * (float)sum_lo;
            sum += d_q4 * (float)sc_hi * d8_hi * (float)dot_hi
                 - dm_q4 * (float)m_hi * d8_hi * (float)sum_hi;
        }
    }

    // Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(WARP_MASK, sum, offset);

    // Inter-warp reduction via shared memory
    __shared__ float warp_sums[4];
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (tid == 0) {
        y[n] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    }
}

ATEN_CUDA_API void launch_q4km_q8_gemv(
    const void* weights, const void* x_q8, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 128;  // 4 warps × 32 threads
    q4km_q8_gemv_kernel<<<N, BLOCK_SIZE, 0, stream>>>(
        static_cast<const uint8_t*>(weights),
        static_cast<const block_q8_1*>(x_q8),
        y, K, N, row_stride_bytes);
}

// ============================================================================
// Q6_K GEMV — Warp-cooperative with coalesced access
// ============================================================================
// Q6_K block (210 bytes = 256 values):
//   ql[128]: low 4 bits
//   qh[64]:  high 2 bits packed
//   scales[16]: int8 scales
//   d: fp16 at offset 208
//
// 32 lanes read 4 consecutive ql bytes (128 bytes, 1 transaction).
// qh read is less perfect but small (64 bytes).

__global__ void q6k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int K, int N,
    int64_t row_stride_bytes)
{
    extern __shared__ float x_shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int n = blockIdx.x * warps_per_block + warp_id;

    // Load x into shared memory
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    if (n >= N) return;

    const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
    const int num_blocks_per_row = K / 256;

    float sum = 0.0f;

    for (int blk = 0; blk < num_blocks_per_row; ++blk) {
        const uint8_t* block_ptr = row_data + blk * 210;

        const uint8_t* ql = block_ptr;
        const uint8_t* qh = block_ptr + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block_ptr + 192);

        uint16_t d_bits;
        memcpy(&d_bits, block_ptr + 208, 2);
        float d = fp16_to_fp32_device(d_bits);

        int base_k = blk * 256;

        // Process 2 halves of 128 values
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql_h = ql + half * 64;
            const uint8_t* qh_h = qh + half * 32;

            // Each lane handles 1 position (l) out of 32
            if (lane < 32) {
                int l = lane;
                int is = (half * 128) / 16 + l / 16;

                int8_t q1 = (int8_t)(((ql_h[l +  0] & 0xF) | (((qh_h[l] >> 0) & 3) << 4))) - 32;
                int8_t q2 = (int8_t)(((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4))) - 32;
                int8_t q3 = (int8_t)(((ql_h[l +  0] >> 4)  | (((qh_h[l] >> 4) & 3) << 4))) - 32;
                int8_t q4 = (int8_t)(((ql_h[l + 32] >> 4)  | (((qh_h[l] >> 6) & 3) << 4))) - 32;

                int k_base = base_k + half * 128;
                float x1 = x_shared[k_base + l +  0];
                float x2 = x_shared[k_base + l + 32];
                float x3 = x_shared[k_base + l + 64];
                float x4 = x_shared[k_base + l + 96];

                sum += d * scales[is + 0] * q1 * x1;
                sum += d * scales[is + 2] * q2 * x2;
                sum += d * scales[is + 4] * q3 * x3;
                sum += d * scales[is + 6] * q4 * x4;
            }
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(WARP_MASK, sum, offset);
    }
    if (lane == 0) y[n] = sum;
}

ATEN_CUDA_API void launch_q6k_gemv(
    const void* weights,
    const float* x,
    float* y,
    int K, int N,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = (N + WARPS - 1) / WARPS;
    int smem_bytes = K * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q6k_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q6k_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weights), x, y,
        K, N, row_stride_bytes);
}

// ============================================================================
// Q5_K GEMV — Warp-cooperative with coalesced access
// ============================================================================
// Q5_K block (176 bytes = 256 values):
//   d(fp16) + dmin(fp16) + scales[12] + qh[32] + qs[128]
//   5th bit stored in qh: bit (group*2 + is_upper) of qh[pos_in_group]

__global__ void q5k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int K, int N,
    int64_t row_stride_bytes)
{
    extern __shared__ float x_shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int n = blockIdx.x * warps_per_block + warp_id;

    // Load x into shared memory
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    if (n >= N) return;

    const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
    const int num_blocks_per_row = K / 256;

    float sum = 0.0f;

    const int group = lane / 8;           // 0-3
    const int pos = (lane & 7) * 4;       // 0,4,8,...,28

    for (int blk = 0; blk < num_blocks_per_row; ++blk) {
        const uint8_t* block_ptr = row_data + blk * 176;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        const float d = fp16_to_fp32_device(d_bits);
        const float dmin = fp16_to_fp32_device(dmin_bits);

        const uint8_t* scales = block_ptr + 4;
        const uint8_t* qh = block_ptr + 16;
        const uint8_t* qs = block_ptr + 48;

        // Scales for this group
        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4_device(group * 2, scales, &sc_lo, &m_lo);
        get_scale_min_k4_device(group * 2 + 1, scales, &sc_hi, &m_hi);
        const float d_lo = d * sc_lo;
        const float m_lo_f = dmin * m_lo;
        const float d_hi = d * sc_hi;
        const float m_hi_f = dmin * m_hi;

        // Read 4 qs bytes (coalesced)
        const int byte_ofs = lane * 4;
        uint32_t qs4;
        memcpy(&qs4, qs + byte_ofs, 4);

        // qh bit masks for low/high halves
        const uint8_t mask_lo = (uint8_t)(1 << (group * 2));
        const uint8_t mask_hi = (uint8_t)(1 << (group * 2 + 1));

        // Read 4 qh bytes for the 5th bit
        uint8_t qh0 = qh[pos + 0];
        uint8_t qh1 = qh[pos + 1];
        uint8_t qh2 = qh[pos + 2];
        uint8_t qh3 = qh[pos + 3];

        // Low nibble values + 5th bit
        float v0_lo = (float)( qs4        & 0xF) + ((qh0 & mask_lo) ? 16.0f : 0.0f);
        float v1_lo = (float)((qs4 >>  8) & 0xF) + ((qh1 & mask_lo) ? 16.0f : 0.0f);
        float v2_lo = (float)((qs4 >> 16) & 0xF) + ((qh2 & mask_lo) ? 16.0f : 0.0f);
        float v3_lo = (float)((qs4 >> 24) & 0xF) + ((qh3 & mask_lo) ? 16.0f : 0.0f);

        // High nibble values + 5th bit
        float v0_hi = (float)((qs4 >>  4) & 0xF) + ((qh0 & mask_hi) ? 16.0f : 0.0f);
        float v1_hi = (float)((qs4 >> 12) & 0xF) + ((qh1 & mask_hi) ? 16.0f : 0.0f);
        float v2_hi = (float)((qs4 >> 20) & 0xF) + ((qh2 & mask_hi) ? 16.0f : 0.0f);
        float v3_hi = (float)((qs4 >> 28) & 0xF) + ((qh3 & mask_hi) ? 16.0f : 0.0f);

        // x from shared memory
        const int k_lo = blk * 256 + group * 64 + pos;
        const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_lo]);
        const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_lo + 32]);

        sum += (d_lo * v0_lo - m_lo_f) * x_lo.x;
        sum += (d_lo * v1_lo - m_lo_f) * x_lo.y;
        sum += (d_lo * v2_lo - m_lo_f) * x_lo.z;
        sum += (d_lo * v3_lo - m_lo_f) * x_lo.w;

        sum += (d_hi * v0_hi - m_hi_f) * x_hi.x;
        sum += (d_hi * v1_hi - m_hi_f) * x_hi.y;
        sum += (d_hi * v2_hi - m_hi_f) * x_hi.z;
        sum += (d_hi * v3_hi - m_hi_f) * x_hi.w;
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(WARP_MASK, sum, offset);
    }
    if (lane == 0) y[n] = sum;
}

ATEN_CUDA_API void launch_q5k_gemv(
    const void* weights,
    const float* x,
    float* y,
    int K, int N,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = (N + WARPS - 1) / WARPS;
    int smem_bytes = K * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q5k_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q5k_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weights), x, y,
        K, N, row_stride_bytes);
}

// ============================================================================
// Q4_K_M → FP16 Dequantization Kernel
// ============================================================================
// Dequantizes a Q4_K_M weight matrix [N, K] to half-precision [N, K].
// Each Q4_K block encodes 256 values in 144 bytes.
// Launch: grid=(N, blocks_per_row), block=256 threads.

__global__ void dequant_q4k_to_fp16_kernel(
    const uint8_t* __restrict__ weights,
    __half* __restrict__ out,
    int K, int N, int64_t row_stride_bytes)
{
    const int row = blockIdx.x;
    const int blk = blockIdx.y;
    const int tid = threadIdx.x;

    if (row >= N) return;

    const int blocks_per_row = K / 256;
    if (blk >= blocks_per_row) return;

    // Point to Q4_K block for this row
    const uint8_t* block_ptr = weights + row * row_stride_bytes + blk * 144;

    // Parse Q4_K block header
    uint16_t d_raw, dmin_raw;
    memcpy(&d_raw, block_ptr, 2);
    memcpy(&dmin_raw, block_ptr + 2, 2);
    float d = fp16_to_fp32_device(d_raw);
    float dmin = fp16_to_fp32_device(dmin_raw);

    const uint8_t* scales = block_ptr + 4;   // 12 bytes of packed scales
    const uint8_t* qs = block_ptr + 16;      // 128 bytes of quant data

    // Q4_K layout: 4 groups of 64 values.
    // Each group j (0..3) uses 32 bytes of qs at qs[j*32 .. j*32+31].
    // First 32 values = lower nibbles (scale_idx = j*2)
    // Next 32 values = upper nibbles (scale_idx = j*2+1)
    //
    // tid 0..255 maps to output position:
    //   group = tid / 64  (0..3)
    //   within = tid % 64
    //   if within < 32: lower nibble, scale_idx = group*2
    //   else:           upper nibble, scale_idx = group*2+1

    int group = tid / 64;           // 0..3
    int within = tid % 64;          // 0..63
    int is_upper = (within >= 32);
    int pos = within & 31;          // 0..31

    int scale_idx = group * 2 + is_upper;
    uint8_t sc_byte, m_byte;
    get_scale_min_k4_device(scale_idx, scales, &sc_byte, &m_byte);
    float scale = d * sc_byte;
    float min_val = dmin * m_byte;

    // Read quantized nibble
    uint8_t byte_val = qs[group * 32 + pos];
    uint8_t q4 = is_upper ? (byte_val >> 4) : (byte_val & 0xF);

    float val = scale * q4 - min_val;

    // Write to output in COLUMN-MAJOR order [N, K] with lda=N
    // col = blk * 256 + tid (the K-dimension index)
    // row = row (the N-dimension index)
    int col = blk * 256 + tid;
    int out_idx = col * N + row;
    out[out_idx] = __float2half(val);
}

ATEN_CUDA_API void launch_dequant_q4k_to_fp16(
    const void* weights, void* out_fp16,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream)
{
    int blocks_per_row = K / 256;
    dim3 grid(N, blocks_per_row);
    dim3 block(256);
    dequant_q4k_to_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(weights),
        static_cast<__half*>(out_fp16),
        K, N, row_stride_bytes);
}

// ============================================================================
// Float32 ↔ FP16 conversion kernels
// ============================================================================

__global__ void f32_to_f16_kernel(const float* __restrict__ in, __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void f16_to_f32_kernel(const __half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// ============================================================================
// cuBLAS Hgemm GEMV: FP16 weights × FP16 vector → FP16 → FP32 output
// ============================================================================
// Converts x to FP16, calls cublasHgemm, converts y back to FP32.
// Requires pre-allocated FP16 scratch buffers (x_fp16, y_fp16).

ATEN_CUDA_API void launch_cublas_hgemv(
    const void* W_fp16,  // [N, K] row-major FP16 (dequantized weights)
    const float* x,      // [K] FP32 input vector
    float* y,            // [N] FP32 output vector
    int K, int N,
    void* x_fp16_buf,    // scratch buffer: at least K * sizeof(half) bytes
    void* y_fp16_buf,    // scratch buffer: at least N * sizeof(half) bytes
    cudaStream_t stream)
{
    // Convert x: FP32 → FP16
    int threads = 256;
    f32_to_f16_kernel<<<(K + threads - 1) / threads, threads, 0, stream>>>(
        x, static_cast<__half*>(x_fp16_buf), K);

    // cuBLAS GemmEx: FP16 A × FP16 B → FP32 C with FP32 compute
    // Weights stored in column-major [N, K] with lda=N for coalesced reads
    cublasHandle_t handle = CuBLASHandle::get();
    if (stream) cublasSetStream(handle, stream);

    float alpha = 1.0f;
    float beta = 0.0f;

    // W is column-major [N, K] with lda=N
    // y[N] = W[N,K] @ x[K] → CUBLAS_OP_N, m=N, n=1, k=K
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,    // op(A) = A (no transpose, column-major access = coalesced)
        CUBLAS_OP_N,    // op(B) = B
        N,              // m
        1,              // n
        K,              // k
        &alpha,
        W_fp16, CUDA_R_16F, N,                    // A [N,K] col-major, FP16, lda=N
        x_fp16_buf, CUDA_R_16F, K,                // B [K,1], FP16, ldb=K
        &beta,
        y, CUDA_R_32F, N,                          // C [N,1], FP32, ldc=N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
}

// ============================================================================
// Fused QKV GEMV — THREE GEMVs in a single kernel launch
// ============================================================================
// Q, K, V projections share the same input vector x. Loading x into shared
// memory once and computing all three outputs saves 2 kernel launches and
// 2 shared memory loads of x per layer. This targets the 57% qkv_proj
// bottleneck directly.
//
// Strategy: concatenate Q/K/V rows into a single virtual [N_total, K] matrix.
// Each warp processes one output row. blockIdx.x * warps_per_block + warp_id
// determines which row (and thus which output — Q, K, or V).

__global__ void q4km_fused_qkv_gemv_kernel(
    const uint8_t* __restrict__ w_q,
    const uint8_t* __restrict__ w_k,
    const uint8_t* __restrict__ w_v,
    const float* __restrict__ x,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int K, int N_q, int N_k, int N_v,
    int64_t row_stride_bytes)
{
    extern __shared__ float x_shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // Load x into shared memory ONCE (shared across all 3 projections)
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    const int group = lane / 8;
    const int pos = (lane & 7) * 4;
    const int num_blocks_per_row = K / 256;
    const int N_total = N_q + N_k + N_v;

    // Grid-stride loop over all Q, K, V rows
    for (int base_n = blockIdx.x * warps_per_block; base_n < N_total;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N_total) continue;

        // Determine which weight matrix and output buffer
        const uint8_t* weights;
        float* y_out;
        int row_idx;
        if (n < N_q) {
            weights = w_q;
            y_out = out_q;
            row_idx = n;
        } else if (n < N_q + N_k) {
            weights = w_k;
            y_out = out_k;
            row_idx = n - N_q;
        } else {
            weights = w_v;
            y_out = out_v;
            row_idx = n - N_q - N_k;
        }

        const uint8_t* row_data = weights + (int64_t)row_idx * row_stride_bytes;
        float sum = 0.0f;

        for (int blk = 0; blk < num_blocks_per_row; ++blk) {
            const uint8_t* bp = row_data + blk * 144;

            // Vectorized d/dmin read via __ldg (1 transaction instead of 2 memcpy)
            uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp));
            const float d = fp16_to_fp32_device(d_dmin & 0xFFFF);
            const float dm = fp16_to_fp32_device(d_dmin >> 16);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

            uint32_t qs4;
            memcpy(&qs4, bp + 16 + lane * 4, 4);

            float dl = d * sc_lo, ml = dm * m_lo;
            float dh = d * sc_hi, mh = dm * m_hi;

            const int k_base = blk * 256 + group * 64 + pos;
            const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_base]);
            const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_base + 32]);

            sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;
            sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
            sum += (dl * (float)((qs4 >> 16) & 0xF) - ml) * x_lo.z;
            sum += (dl * (float)((qs4 >> 24) & 0xF) - ml) * x_lo.w;
            sum += (dh * (float)((qs4 >>  4) & 0xF) - mh) * x_hi.x;
            sum += (dh * (float)((qs4 >> 12) & 0xF) - mh) * x_hi.y;
            sum += (dh * (float)((qs4 >> 20) & 0xF) - mh) * x_hi.z;
            sum += (dh * (float)((qs4 >> 28) & 0xF) - mh) * x_hi.w;
        }

        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y_out[row_idx] = sum;
    }
}

ATEN_CUDA_API void launch_q4km_fused_qkv_gemv(
    const void* w_q, const void* w_k, const void* w_v,
    const float* x, float* out_q, float* out_k, float* out_v,
    int K, int N_q, int N_k, int N_v,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = sm_count * 4;  // 4x occupancy for A100 bandwidth saturation
    int smem_bytes = K * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_fused_qkv_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_fused_qkv_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(w_q),
        static_cast<const uint8_t*>(w_k),
        static_cast<const uint8_t*>(w_v),
        x, out_q, out_k, out_v,
        K, N_q, N_k, N_v,
        row_stride_bytes);
}

// ============================================================================
// Fused RMSNorm + Q4_K GEMV — Normalize x, then use it for GEMV
// ============================================================================
// Eliminates a separate RMSNorm kernel launch + a global memory round-trip.
// Step 1: Load x into shared memory, compute RMS norm in-place.
// Step 2: Use normalized x_shared for GEMV (same as persistent kernel).
// This saves 1 kernel launch + 2 * hidden_size * sizeof(float) bandwidth/layer.

__global__ void q4km_fused_rmsnorm_gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ weights,
    float* __restrict__ y,
    int K, int N,
    int64_t row_stride_bytes,
    float eps, bool add_one)
{
    extern __shared__ float x_shared[];
    // x_shared[0..K-1] = normalized x
    // x_shared[K..K+blockDim.x-1] = reduction scratch

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    float* reduce_buf = x_shared + K;

    // Step 1: Load x into shared memory
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    // Step 2: Compute sum of squares (parallel reduction)
    float local_sum_sq = 0.0f;
    for (int k = tid; k < K; k += block_size) {
        float val = x_shared[k];
        local_sum_sq += val * val;
    }
    reduce_buf[tid] = local_sum_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(reduce_buf[0] / K + eps);

    // Step 3: Apply RMSNorm in-place in shared memory
    for (int k = tid; k < K; k += block_size) {
        float w = add_one ? (1.0f + norm_weight[k]) : norm_weight[k];
        x_shared[k] = x_shared[k] * rms_inv * w;
    }
    __syncthreads();

    // Step 4: GEMV using normalized x_shared (same as persistent kernel)
    const int group = lane / 8;
    const int pos = (lane & 7) * 4;
    const int num_blocks_per_row = K / 256;

    for (int base_n = blockIdx.x * warps_per_block; base_n < N;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N) continue;

        const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
        float sum = 0.0f;

        for (int blk = 0; blk < num_blocks_per_row; ++blk) {
            const uint8_t* bp = row_data + blk * 144;

            // Vectorized d/dmin read via __ldg (1 transaction instead of 2 memcpy)
            uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp));
            const float d = fp16_to_fp32_device(d_dmin & 0xFFFF);
            const float dm = fp16_to_fp32_device(d_dmin >> 16);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

            uint32_t qs4;
            memcpy(&qs4, bp + 16 + lane * 4, 4);

            float dl = d * sc_lo, ml = dm * m_lo;
            float dh = d * sc_hi, mh = dm * m_hi;

            const int k_base = blk * 256 + group * 64 + pos;
            const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_base]);
            const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_base + 32]);

            sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;
            sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
            sum += (dl * (float)((qs4 >> 16) & 0xF) - ml) * x_lo.z;
            sum += (dl * (float)((qs4 >> 24) & 0xF) - ml) * x_lo.w;
            sum += (dh * (float)((qs4 >>  4) & 0xF) - mh) * x_hi.x;
            sum += (dh * (float)((qs4 >> 12) & 0xF) - mh) * x_hi.y;
            sum += (dh * (float)((qs4 >> 20) & 0xF) - mh) * x_hi.z;
            sum += (dh * (float)((qs4 >> 28) & 0xF) - mh) * x_hi.w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y[n] = sum;
    }
}

ATEN_CUDA_API void launch_q4km_fused_rmsnorm_gemv(
    const float* x, const float* norm_weight,
    const void* weights, float* y,
    int K, int N, int64_t row_stride_bytes,
    float eps, bool add_one,
    cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = sm_count * 4;  // 4x occupancy for A100 bandwidth saturation
    // Shared: K floats for x + BLOCK_SIZE floats for reduction
    int smem_bytes = (K + BLOCK_SIZE) * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_fused_rmsnorm_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_fused_rmsnorm_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        x, norm_weight,
        static_cast<const uint8_t*>(weights), y,
        K, N, row_stride_bytes,
        eps, add_one);
}

// ============================================================================
// Fused RMSNorm + QKV GEMV — Normalize x, then compute Q, K, V in one kernel
// ============================================================================
// Combines fused_rmsnorm_gemv + fused_qkv_gemv: a single kernel that
// normalizes x and produces all 3 attention projections.
// Saves: 1 RMSNorm launch + 2 GEMV launches + 3 shared memory x loads = 4 launches/layer.

__global__ void q4km_fused_rmsnorm_qkv_gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ w_q,
    const uint8_t* __restrict__ w_k,
    const uint8_t* __restrict__ w_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int K, int N_q, int N_k, int N_v,
    int64_t row_stride_bytes,
    float eps, bool add_one)
{
    extern __shared__ float x_shared[];
    float* reduce_buf = x_shared + K;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // Step 1: Load x into shared memory
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    // Step 2: Compute sum of squares
    float local_sum_sq = 0.0f;
    for (int k = tid; k < K; k += block_size) {
        float val = x_shared[k];
        local_sum_sq += val * val;
    }
    reduce_buf[tid] = local_sum_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(reduce_buf[0] / K + eps);

    // Step 3: Apply RMSNorm in-place in shared memory
    for (int k = tid; k < K; k += block_size) {
        float w = add_one ? (1.0f + norm_weight[k]) : norm_weight[k];
        x_shared[k] = x_shared[k] * rms_inv * w;
    }
    __syncthreads();

    // Step 4: Fused QKV GEMV using normalized x_shared
    const int group = lane / 8;
    const int pos = (lane & 7) * 4;
    const int num_blocks_per_row = K / 256;
    const int N_total = N_q + N_k + N_v;

    for (int base_n = blockIdx.x * warps_per_block; base_n < N_total;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N_total) continue;

        const uint8_t* w_ptr;
        float* y_out;
        int row_idx;
        if (n < N_q) {
            w_ptr = w_q; y_out = out_q; row_idx = n;
        } else if (n < N_q + N_k) {
            w_ptr = w_k; y_out = out_k; row_idx = n - N_q;
        } else {
            w_ptr = w_v; y_out = out_v; row_idx = n - N_q - N_k;
        }

        const uint8_t* row_data = w_ptr + (int64_t)row_idx * row_stride_bytes;
        float sum = 0.0f;

        for (int blk = 0; blk < num_blocks_per_row; ++blk) {
            const uint8_t* bp = row_data + blk * 144;

            // Vectorized d/dmin read via __ldg (1 transaction instead of 2 memcpy)
            uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp));
            const float d = fp16_to_fp32_device(d_dmin & 0xFFFF);
            const float dm = fp16_to_fp32_device(d_dmin >> 16);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

            uint32_t qs4;
            memcpy(&qs4, bp + 16 + lane * 4, 4);

            float dl = d * sc_lo, ml = dm * m_lo;
            float dh = d * sc_hi, mh = dm * m_hi;

            const int k_base = blk * 256 + group * 64 + pos;
            const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_base]);
            const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_base + 32]);

            sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;
            sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
            sum += (dl * (float)((qs4 >> 16) & 0xF) - ml) * x_lo.z;
            sum += (dl * (float)((qs4 >> 24) & 0xF) - ml) * x_lo.w;
            sum += (dh * (float)((qs4 >>  4) & 0xF) - mh) * x_hi.x;
            sum += (dh * (float)((qs4 >> 12) & 0xF) - mh) * x_hi.y;
            sum += (dh * (float)((qs4 >> 20) & 0xF) - mh) * x_hi.z;
            sum += (dh * (float)((qs4 >> 28) & 0xF) - mh) * x_hi.w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y_out[row_idx] = sum;
    }
}

ATEN_CUDA_API void launch_q4km_fused_rmsnorm_qkv_gemv(
    const float* x, const float* norm_weight,
    const void* w_q, const void* w_k, const void* w_v,
    float* out_q, float* out_k, float* out_v,
    int K, int N_q, int N_k, int N_v,
    int64_t row_stride_bytes,
    float eps, bool add_one,
    cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = sm_count * 4;  // 4x occupancy for A100 bandwidth saturation
    int smem_bytes = (K + BLOCK_SIZE) * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_fused_rmsnorm_qkv_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_fused_rmsnorm_qkv_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        x, norm_weight,
        static_cast<const uint8_t*>(w_q),
        static_cast<const uint8_t*>(w_k),
        static_cast<const uint8_t*>(w_v),
        out_q, out_k, out_v,
        K, N_q, N_k, N_v,
        row_stride_bytes,
        eps, add_one);
}

// ============================================================================
// GEMV with beta=1 accumulate — y[n] += sum_k dequant(W[n,k]) * x[k]
// ============================================================================
// Used for output_proj: residual[i] += W @ attn_out[i]
// Eliminates the separate residual_add kernel launch.

__global__ void q4km_persistent_gemv_accumulate_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,  // read-modify-write (y += W@x)
    int K, int N,
    int64_t row_stride_bytes)
{
    extern __shared__ float x_shared[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // Load x into shared memory ONCE
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    const int group = lane / 8;
    const int pos = (lane & 7) * 4;
    const int num_blocks_per_row = K / 256;

    for (int base_n = blockIdx.x * warps_per_block; base_n < N;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N) continue;

        const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
        float sum = 0.0f;

        for (int blk = 0; blk < num_blocks_per_row; ++blk) {
            const uint8_t* bp = row_data + blk * 144;

            // Vectorized d/dmin read via __ldg (1 transaction instead of 2 memcpy)
            uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp));
            const float d = fp16_to_fp32_device(d_dmin & 0xFFFF);
            const float dm = fp16_to_fp32_device(d_dmin >> 16);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

            uint32_t qs4;
            memcpy(&qs4, bp + 16 + lane * 4, 4);

            float dl = d * sc_lo, ml = dm * m_lo;
            float dh = d * sc_hi, mh = dm * m_hi;

            const int k_base = blk * 256 + group * 64 + pos;
            const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_base]);
            const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_base + 32]);

            sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;
            sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
            sum += (dl * (float)((qs4 >> 16) & 0xF) - ml) * x_lo.z;
            sum += (dl * (float)((qs4 >> 24) & 0xF) - ml) * x_lo.w;
            sum += (dh * (float)((qs4 >>  4) & 0xF) - mh) * x_hi.x;
            sum += (dh * (float)((qs4 >> 12) & 0xF) - mh) * x_hi.y;
            sum += (dh * (float)((qs4 >> 20) & 0xF) - mh) * x_hi.z;
            sum += (dh * (float)((qs4 >> 28) & 0xF) - mh) * x_hi.w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y[n] += sum;  // ACCUMULATE, not overwrite
    }
}

ATEN_CUDA_API void launch_q4km_persistent_gemv_accumulate(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes,
    cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = sm_count * 4;  // 4x occupancy for A100 bandwidth saturation
    int smem_bytes = K * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_persistent_gemv_accumulate_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_persistent_gemv_accumulate_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weights), x, y,
        K, N, row_stride_bytes);
}

// ============================================================================
// Fused RMSNorm + Gate+Up GEMV — Normalize x, then compute gate and up
// ============================================================================
// Same pattern as fused_rmsnorm_qkv but for FFN path.
// Saves: 1 RMSNorm launch + 1 GEMV launch + 2 shared memory x loads.

__global__ void q4km_fused_rmsnorm_gate_up_kernel(
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ w_gate,
    const uint8_t* __restrict__ w_up,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes,
    float eps, bool add_one)
{
    extern __shared__ float x_shared[];
    float* reduce_buf = x_shared + K;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // Step 1: Load x into shared memory
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    // Step 2: Compute sum of squares
    float local_sum_sq = 0.0f;
    for (int k = tid; k < K; k += block_size) {
        float val = x_shared[k];
        local_sum_sq += val * val;
    }
    reduce_buf[tid] = local_sum_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(reduce_buf[0] / K + eps);

    // Step 3: Apply RMSNorm in-place
    for (int k = tid; k < K; k += block_size) {
        float w = add_one ? (1.0f + norm_weight[k]) : norm_weight[k];
        x_shared[k] = x_shared[k] * rms_inv * w;
    }
    __syncthreads();

    // Step 4: Fused gate+up GEMV
    const int group = lane / 8;
    const int pos = (lane & 7) * 4;
    const int num_blocks_per_row = K / 256;
    const int N_total = N_gate + N_up;

    for (int base_n = blockIdx.x * warps_per_block; base_n < N_total;
         base_n += gridDim.x * warps_per_block)
    {
        int n = base_n + warp_id;
        if (n >= N_total) continue;

        const uint8_t* w_ptr;
        float* y_out;
        int row_idx;
        if (n < N_gate) {
            w_ptr = w_gate; y_out = y_gate; row_idx = n;
        } else {
            w_ptr = w_up; y_out = y_up; row_idx = n - N_gate;
        }

        const uint8_t* row_data = w_ptr + (int64_t)row_idx * row_stride_bytes;
        float sum = 0.0f;

        for (int blk = 0; blk < num_blocks_per_row; ++blk) {
            const uint8_t* bp = row_data + blk * 144;

            // Vectorized d/dmin read via __ldg (1 transaction instead of 2 memcpy)
            uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(bp));
            const float d = fp16_to_fp32_device(d_dmin & 0xFFFF);
            const float dm = fp16_to_fp32_device(d_dmin >> 16);

            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_device(group * 2, bp + 4, &sc_lo, &m_lo);
            get_scale_min_k4_device(group * 2 + 1, bp + 4, &sc_hi, &m_hi);

            uint32_t qs4;
            memcpy(&qs4, bp + 16 + lane * 4, 4);

            float dl = d * sc_lo, ml = dm * m_lo;
            float dh = d * sc_hi, mh = dm * m_hi;

            const int k_base = blk * 256 + group * 64 + pos;
            const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_base]);
            const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_base + 32]);

            sum += (dl * (float)( qs4        & 0xF) - ml) * x_lo.x;
            sum += (dl * (float)((qs4 >>  8) & 0xF) - ml) * x_lo.y;
            sum += (dl * (float)((qs4 >> 16) & 0xF) - ml) * x_lo.z;
            sum += (dl * (float)((qs4 >> 24) & 0xF) - ml) * x_lo.w;
            sum += (dh * (float)((qs4 >>  4) & 0xF) - mh) * x_hi.x;
            sum += (dh * (float)((qs4 >> 12) & 0xF) - mh) * x_hi.y;
            sum += (dh * (float)((qs4 >> 20) & 0xF) - mh) * x_hi.z;
            sum += (dh * (float)((qs4 >> 28) & 0xF) - mh) * x_hi.w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(WARP_MASK, sum, offset);
        }
        if (lane == 0) y_out[row_idx] = sum;
    }
}

ATEN_CUDA_API void launch_q4km_fused_rmsnorm_gate_up_gemv(
    const float* x, const float* norm_weight,
    const void* w_gate, const void* w_up,
    float* y_gate, float* y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes,
    float eps, bool add_one,
    cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    const int WARPS = 8;
    const int BLOCK_SIZE = WARPS * 32;
    int grid = sm_count * 4;  // 4x occupancy for A100 bandwidth saturation
    int smem_bytes = (K + BLOCK_SIZE) * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_fused_rmsnorm_gate_up_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_fused_rmsnorm_gate_up_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        x, norm_weight,
        static_cast<const uint8_t*>(w_gate),
        static_cast<const uint8_t*>(w_up),
        y_gate, y_up,
        K, N_gate, N_up,
        row_stride_bytes,
        eps, add_one);
}

} // namespace cuda
} // namespace at
