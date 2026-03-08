// ============================================================================
// Fused Quantized Dequant-GEMV Kernels for PromeTorch (v3)
// ============================================================================
// Warp-cooperative design: each warp handles one output row.
// 32 threads read consecutive qs bytes → perfect coalescing.
// x vector loaded into shared memory for zero-latency reuse.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "aten/src/ATen/cuda/CUDAOps.h"

namespace at {
namespace cuda {

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
// Q4_K_M GEMV — Warp-cooperative with coalesced qs reads
// ============================================================================
// Each warp processes one output row. 32 lanes read 4 consecutive qs bytes
// each per Q4_K block (128 bytes total = perfectly coalesced single transaction).
// x vector is in shared memory for fast repeated access.

__global__ void q4km_gemv_kernel(
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

    // Phase 1: Cooperatively load x into shared memory
    for (int k = tid; k < K; k += block_size) {
        x_shared[k] = x[k];
    }
    __syncthreads();

    if (n >= N) return;

    // Phase 2: Each warp processes one output row
    const uint8_t* row_data = weights + (int64_t)n * row_stride_bytes;
    const int num_blocks_per_row = K / 256;

    float sum = 0.0f;

    // Lane assignment: lane l reads qs bytes [4*l .. 4*l+3]
    // group = (4*l) / 32 = l / 8
    // pos_in_group = (4*l) % 32 = (l % 8) * 4
    const int group = lane / 8;           // 0-3
    const int pos = (lane & 7) * 4;       // 0,4,8,...,28

    for (int blk = 0; blk < num_blocks_per_row; ++blk) {
        const uint8_t* block_ptr = row_data + blk * 144;

        // Read d, dmin (all lanes read same — hardware broadcast)
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block_ptr, 2);
        memcpy(&dmin_bits, block_ptr + 2, 2);
        const float d = fp16_to_fp32_device(d_bits);
        const float dmin = fp16_to_fp32_device(dmin_bits);

        const uint8_t* scales = block_ptr + 4;

        // Scales for this group's lower and upper halves
        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4_device(group * 2, scales, &sc_lo, &m_lo);
        get_scale_min_k4_device(group * 2 + 1, scales, &sc_hi, &m_hi);
        const float d_lo = d * sc_lo;
        const float m_lo_f = dmin * m_lo;
        const float d_hi = d * sc_hi;
        const float m_hi_f = dmin * m_hi;

        // Read 4 qs bytes as single coalesced 32-bit load
        const uint8_t* qs = block_ptr + 16;
        const int byte_ofs = lane * 4;    // = group*32 + pos
        uint32_t qs4;
        memcpy(&qs4, qs + byte_ofs, 4);

        // Extract nibbles (4 low + 4 high = 8 values)
        const float v0_lo = (float)( qs4        & 0xF);
        const float v1_lo = (float)((qs4 >>  8) & 0xF);
        const float v2_lo = (float)((qs4 >> 16) & 0xF);
        const float v3_lo = (float)((qs4 >> 24) & 0xF);

        const float v0_hi = (float)((qs4 >>  4) & 0xF);
        const float v1_hi = (float)((qs4 >> 12) & 0xF);
        const float v2_hi = (float)((qs4 >> 20) & 0xF);
        const float v3_hi = (float)((qs4 >> 28) & 0xF);

        // x from shared memory (low nibble: group*64 + pos, high: +32)
        const int k_lo = blk * 256 + group * 64 + pos;
        const float4 x_lo = *reinterpret_cast<const float4*>(&x_shared[k_lo]);
        const float4 x_hi = *reinterpret_cast<const float4*>(&x_shared[k_lo + 32]);

        // Accumulate: dequant × x
        sum += (d_lo * v0_lo - m_lo_f) * x_lo.x;
        sum += (d_lo * v1_lo - m_lo_f) * x_lo.y;
        sum += (d_lo * v2_lo - m_lo_f) * x_lo.z;
        sum += (d_lo * v3_lo - m_lo_f) * x_lo.w;

        sum += (d_hi * v0_hi - m_hi_f) * x_hi.x;
        sum += (d_hi * v1_hi - m_hi_f) * x_hi.y;
        sum += (d_hi * v2_hi - m_hi_f) * x_hi.z;
        sum += (d_hi * v3_hi - m_hi_f) * x_hi.w;
    }

    // Warp shuffle reduction (no shared memory needed)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) y[n] = sum;
}

// ============================================================================
// Launch wrapper
// ============================================================================

ATEN_CUDA_API void launch_q4km_gemv(
    const void* weights,
    const float* x,
    float* y,
    int K, int N,
    int64_t row_stride_bytes,
    cudaStream_t stream)
{
    const int WARPS = 4;
    const int BLOCK_SIZE = WARPS * 32;  // 128 threads
    int grid = (N + WARPS - 1) / WARPS;
    int smem_bytes = K * sizeof(float);

    // Request extended shared memory for large K (default 48 KB limit)
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(q4km_gemv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    q4km_gemv_kernel<<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weights), x, y,
        K, N, row_stride_bytes);
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
        sum += __shfl_down_sync(0xffffffff, sum, offset);
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
    const int WARPS = 4;
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
        sum += __shfl_down_sync(0xffffffff, sum, offset);
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
    const int WARPS = 4;
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

} // namespace cuda
} // namespace at
