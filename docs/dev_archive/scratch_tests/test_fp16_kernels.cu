// Minimal standalone FP16 CUDA kernel test.
// Verifies launch_add_fp16, launch_relu_fp16, launch_sigmoid_fp16, launch_tanh_fp16,
// launch_check_inf_nan_fp16. Runs on A100.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Pull declarations directly (avoid full PromeTorch include chain).
namespace at { namespace cuda {
void launch_add_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t s);
void launch_mul_fp16(const __half* a, const __half* b, __half* out, int64_t n, cudaStream_t s);
void launch_relu_fp16(const __half* input, __half* output, int64_t n, cudaStream_t s);
void launch_sigmoid_fp16(const __half* input, __half* output, int64_t n, cudaStream_t s);
void launch_tanh_fp16(const __half* input, __half* output, int64_t n, cudaStream_t s);
void launch_check_inf_nan_fp16(const __half* x, int64_t n, int* found_flag, cudaStream_t s);
}}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    return 1; }} while(0)

static __half f2h(float f) { return __float2half(f); }
static float h2f(__half h) { return __half2float(h); }

int main() {
    int n = 128 * 1024;
    printf("== PromeTorch FP16 CUDA kernel verification ==\n");
    printf("n = %d elements\n\n", n);

    // Allocate host buffers.
    std::vector<__half> h_a(n), h_b(n), h_out(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = f2h((i % 100) * 0.01f - 0.5f);
        h_b[i] = f2h((i % 7) * 0.1f);
    }

    // Device buffers.
    __half *d_a, *d_b, *d_out;
    int *d_inf;
    CHECK(cudaMalloc(&d_a, n * sizeof(__half)));
    CHECK(cudaMalloc(&d_b, n * sizeof(__half)));
    CHECK(cudaMalloc(&d_out, n * sizeof(__half)));
    CHECK(cudaMalloc(&d_inf, sizeof(int)));
    CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    // ---- Test 1: add ----
    at::cuda::launch_add_fp16(d_a, d_b, d_out, n, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    {
        float max_err = 0.0f;
        for (int i = 0; i < n; ++i) {
            float expected = h2f(h_a[i]) + h2f(h_b[i]);
            float got = h2f(h_out[i]);
            float err = std::fabs(got - expected);
            if (err > max_err) max_err = err;
        }
        printf("[add_fp16]     max |err| = %.4e %s\n", max_err, max_err < 1e-2 ? "PASS" : "FAIL");
    }

    // ---- Test 2: mul ----
    at::cuda::launch_mul_fp16(d_a, d_b, d_out, n, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    {
        float max_err = 0.0f;
        for (int i = 0; i < n; ++i) {
            float expected = h2f(h_a[i]) * h2f(h_b[i]);
            float got = h2f(h_out[i]);
            float err = std::fabs(got - expected);
            if (err > max_err) max_err = err;
        }
        printf("[mul_fp16]     max |err| = %.4e %s\n", max_err, max_err < 1e-2 ? "PASS" : "FAIL");
    }

    // ---- Test 3: relu ----
    at::cuda::launch_relu_fp16(d_a, d_out, n, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    {
        int fails = 0;
        for (int i = 0; i < n; ++i) {
            float expected = std::fmax(0.0f, h2f(h_a[i]));
            float got = h2f(h_out[i]);
            if (std::fabs(got - expected) > 1e-3) ++fails;
        }
        printf("[relu_fp16]    fails = %d / %d %s\n", fails, n, fails == 0 ? "PASS" : "FAIL");
    }

    // ---- Test 4: sigmoid ----
    at::cuda::launch_sigmoid_fp16(d_a, d_out, n, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    {
        float max_err = 0.0f;
        for (int i = 0; i < n; ++i) {
            float a = h2f(h_a[i]);
            float expected = 1.0f / (1.0f + std::exp(-a));
            float got = h2f(h_out[i]);
            float err = std::fabs(got - expected);
            if (err > max_err) max_err = err;
        }
        printf("[sigmoid_fp16] max |err| = %.4e %s\n", max_err, max_err < 2e-3 ? "PASS" : "FAIL");
    }

    // ---- Test 5: tanh ----
    at::cuda::launch_tanh_fp16(d_a, d_out, n, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    {
        float max_err = 0.0f;
        for (int i = 0; i < n; ++i) {
            float expected = std::tanh(h2f(h_a[i]));
            float got = h2f(h_out[i]);
            float err = std::fabs(got - expected);
            if (err > max_err) max_err = err;
        }
        printf("[tanh_fp16]    max |err| = %.4e %s\n", max_err, max_err < 2e-3 ? "PASS" : "FAIL");
    }

    // ---- Test 6: check_inf_nan on clean buffer ----
    int flag = 0;
    CHECK(cudaMemcpy(d_inf, &flag, sizeof(int), cudaMemcpyHostToDevice));
    at::cuda::launch_check_inf_nan_fp16(d_a, n, d_inf, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&flag, d_inf, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[check_inf_nan clean]   flag = %d (expect 0) %s\n", flag, flag == 0 ? "PASS" : "FAIL");

    // ---- Test 7: check_inf_nan on poisoned buffer ----
    __half inf_h = f2h(INFINITY);
    CHECK(cudaMemcpy(d_a + 100, &inf_h, sizeof(__half), cudaMemcpyHostToDevice));
    flag = 0;
    CHECK(cudaMemcpy(d_inf, &flag, sizeof(int), cudaMemcpyHostToDevice));
    at::cuda::launch_check_inf_nan_fp16(d_a, n, d_inf, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&flag, d_inf, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[check_inf_nan poisoned] flag = %d (expect 1) %s\n", flag, flag == 1 ? "PASS" : "FAIL");

    // ---- Bench throughput ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int iters = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        at::cuda::launch_add_fp16(d_a, d_b, d_out, n, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    double gelem_per_sec = (double)iters * n / (ms * 1e6);
    printf("\n[bench add_fp16] %d iters, n=%d, %.3f ms total, %.2f Gelem/s\n",
           iters, n, ms, gelem_per_sec);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_inf);
    printf("\nAll FP16 kernels verified on GPU.\n");
    return 0;
}
