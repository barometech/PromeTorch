// Custom GEMM kernel (launch_gemm_native) runtime verification vs cuBLAS reference.
// Both entry points live in aten_cuda.dll.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

namespace at { namespace cuda {
void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, float alpha, float beta,
                 bool trans_a, bool trans_b, cudaStream_t s);
void launch_gemm_native(const float* A, const float* B, float* C,
                        int M, int N, int K, float alpha, float beta,
                        bool trans_a, bool trans_b, cudaStream_t s);
}}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error at %d: %s\n", __LINE__, cudaGetErrorString(e)); \
    return 1; }} while(0)

static void ref_cpu_gemm(const float* A, const float* B, float* C,
                         int M, int N, int K, float alpha, float beta,
                         bool trans_a, bool trans_b) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = trans_a ? A[k * M + i] : A[i * K + k];
                float b = trans_b ? B[j * K + k] : B[k * N + j];
                s += a * b;
            }
            C[i * N + j] = alpha * s + beta * C[i * N + j];
        }
    }
}

int main() {
    printf("== PromeTorch launch_gemm_native vs launch_gemm (cuBLAS) verification ==\n");

    struct Config { int M, N, K; const char* label; };
    std::vector<Config> cases = {
        {64, 64, 64, "64x64x64 small"},
        {256, 256, 256, "256x256x256 medium"},
        {1024, 1024, 1024, "1024x1024x1024 large"},
    };

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (const auto& cfg : cases) {
        int M = cfg.M, N = cfg.N, K = cfg.K;
        printf("\n--- %s (M=%d N=%d K=%d) ---\n", cfg.label, M, N, K);

        std::vector<float> hA(M * K), hB(K * N), hC_ref(M * N, 0.0f);
        for (auto& v : hA) v = dist(rng);
        for (auto& v : hB) v = dist(rng);

        float *dA, *dB, *dC_cublas, *dC_native;
        CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
        CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
        CHECK(cudaMalloc(&dC_cublas, M * N * sizeof(float)));
        CHECK(cudaMalloc(&dC_native, M * N * sizeof(float)));
        CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dC_cublas, 0, M * N * sizeof(float)));
        CHECK(cudaMemset(dC_native, 0, M * N * sizeof(float)));

        // Warmup + time cuBLAS
        for (int w = 0; w < 3; ++w)
            at::cuda::launch_gemm(dA, dB, dC_cublas, M, N, K, 1.0f, 0.0f, false, false, nullptr);
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        int iters = 50;
        cudaEventRecord(s);
        for (int i = 0; i < iters; ++i)
            at::cuda::launch_gemm(dA, dB, dC_cublas, M, N, K, 1.0f, 0.0f, false, false, nullptr);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms_cublas; cudaEventElapsedTime(&ms_cublas, s, e);

        // Warmup + time native
        for (int w = 0; w < 3; ++w)
            at::cuda::launch_gemm_native(dA, dB, dC_native, M, N, K, 1.0f, 0.0f, false, false, nullptr);
        CHECK(cudaDeviceSynchronize());

        cudaEventRecord(s);
        for (int i = 0; i < iters; ++i)
            at::cuda::launch_gemm_native(dA, dB, dC_native, M, N, K, 1.0f, 0.0f, false, false, nullptr);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms_native; cudaEventElapsedTime(&ms_native, s, e);

        // Copy results back, compare
        std::vector<float> hC_cublas(M * N), hC_native(M * N);
        CHECK(cudaMemcpy(hC_cublas.data(), dC_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hC_native.data(), dC_native, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        float max_diff = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            float d = std::fabs(hC_cublas[i] - hC_native[i]);
            if (d > max_diff) max_diff = d;
        }
        float rel = max_diff / std::sqrt((float)K);
        const char* pass = (rel < 1e-3) ? "PASS" : "FAIL";

        double flops = 2.0 * M * N * K;
        double gflops_cublas = (flops * iters) / (ms_cublas * 1e6);
        double gflops_native = (flops * iters) / (ms_native * 1e6);

        printf("  cuBLAS  : %.3f ms/iter, %.1f GFLOPS\n", ms_cublas / iters, gflops_cublas);
        printf("  native  : %.3f ms/iter, %.1f GFLOPS  (%.1fx of cuBLAS)\n",
               ms_native / iters, gflops_native, gflops_native / gflops_cublas);
        printf("  max |diff| = %.4e (rel/sqrt(K) = %.4e) %s\n", max_diff, rel, pass);

        cudaFree(dA); cudaFree(dB); cudaFree(dC_cublas); cudaFree(dC_native);
        cudaEventDestroy(s); cudaEventDestroy(e);
    }

    printf("\nDone.\n");
    return 0;
}
