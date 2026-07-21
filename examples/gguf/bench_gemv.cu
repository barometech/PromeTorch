// ============================================================================
// bench_gemv — микробенч-харнесс для Q4_K decode-GEMV (Phase 0 плана GEMV 2×).
// Гоняет ОДНО ядро изолированно (без пайплайн-confound), мерит CUDA-events и
// эффективную полосу памяти. Профилируется под ncu для dram%/occupancy/лимитера.
//
// Использование:
//   bench_gemv <kernel> <K> <N> [iters]
//     kernel: gate_up | persistent | qkv
//   пример: bench_gemv gate_up 2560 19456 200
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace at { namespace cuda {
void launch_q4km_fused_rmsnorm_gate_up_gemv(
    const float* x, const float* norm_weight,
    const void* w_gate, const void* w_up,
    float* y_gate, float* y_up,
    int K, int N_gate, int N_up,
    int64_t row_stride_bytes, float eps, bool add_one, cudaStream_t stream);
void launch_q4km_persistent_gemv(
    const void* weights, const float* x, float* y,
    int K, int N, int64_t row_stride_bytes, cudaStream_t stream);
void launch_quantize_q8_1(const float* x, void* y_q8, int K, cudaStream_t stream);
void launch_q4km_q8_gemv(
    const void* weights, const void* x_q8, float* y,
    int K, int N, int64_t row_stride_bytes, cudaStream_t stream);
}}

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @ %d: %s\n",#call,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: bench_gemv <gate_up|persistent> <K> <N> [iters]\n"); return 1; }
    std::string kern = argv[1];
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    int iters = (argc > 4) ? atoi(argv[4]) : 200;

    const int64_t row_stride = (int64_t)(K / 256) * 144;   // Q4_K: 144 B / 256 values
    const int64_t w_bytes = (int64_t)N * row_stride;

    // Псевдослучайные веса/x (детерминированно)
    std::vector<uint8_t> hW(w_bytes);
    for (int64_t i = 0; i < w_bytes; ++i) hW[i] = (uint8_t)((i * 131 + 7) & 0xFF);
    std::vector<float> hx(K), hnw(K);
    for (int i = 0; i < K; ++i) { hx[i] = 0.01f * (i % 17) - 0.08f; hnw[i] = 1.0f; }

    uint8_t *dW, *dW2; float *dx, *dnw, *dy, *dy2;
    CK(cudaMalloc(&dW,  w_bytes));
    CK(cudaMalloc(&dW2, w_bytes));
    CK(cudaMalloc(&dx,  K * sizeof(float)));
    CK(cudaMalloc(&dnw, K * sizeof(float)));
    CK(cudaMalloc(&dy,  (int64_t)N * sizeof(float)));
    CK(cudaMalloc(&dy2, (int64_t)N * sizeof(float)));
    CK(cudaMemcpy(dW,  hW.data(),  w_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW2, hW.data(),  w_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx,  hx.data(),  K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dnw, hnw.data(), K * sizeof(float), cudaMemcpyHostToDevice));

    // Q8_1 pre-quantized x buffer для dp4a (36 B / 32 values)
    void* dxq8 = nullptr;
    CK(cudaMalloc(&dxq8, (int64_t)(K / 32) * 36));
    at::cuda::launch_quantize_q8_1(dx, dxq8, K, nullptr);  // Phase 1: quantize-x-once
    CK(cudaDeviceSynchronize());

    auto run = [&](){
        if (kern == "gate_up") {
            at::cuda::launch_q4km_fused_rmsnorm_gate_up_gemv(
                dx, dnw, dW, dW2, dy, dy2, K, N/2, N/2, row_stride, 1e-6f, false, nullptr);
        } else if (kern == "dp4a") {
            // Phase 2: dp4a GEMV на пред-квантованном x (x уже в Q8_1, НЕ реквантим)
            at::cuda::launch_q4km_q8_gemv(dW, dxq8, dy, K, N, row_stride, nullptr);
        } else {
            at::cuda::launch_q4km_persistent_gemv(dW, dx, dy, K, N, row_stride, nullptr);
        }
    };

    // Прогрев
    for (int i = 0; i < 20; ++i) run();
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) run();
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));

    double per_us = (ms * 1000.0) / iters;
    // gate_up читает 2 матрицы весов (gate+up); persistent — 1
    double wpass = (kern == "gate_up") ? (double)w_bytes * 2.0 : (double)w_bytes;
    double gbps = wpass / (per_us * 1e-6) / 1e9;

    printf("kernel=%s K=%d N=%d iters=%d | %.2f us/call | weights %.1f MB/call | %.1f GB/s\n",
           kern.c_str(), K, N, iters, per_us, wpass/1e6, gbps);
    return 0;
}
