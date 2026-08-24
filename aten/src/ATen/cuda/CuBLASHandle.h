#pragma once

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace at {
namespace cuda {

class CuBLASHandle {
public:
    static cublasHandle_t get() {
        static thread_local CuBLASHandle instance;
        return instance.handle_;
    }

    ~CuBLASHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CuBLASHandle(const CuBLASHandle&) = delete;
    CuBLASHandle& operator=(const CuBLASHandle&) = delete;

private:
    CuBLASHandle() : handle_(nullptr) {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS create failed: " + std::to_string(static_cast<int>(status)));
        }
        // Enable TF32 tensor cores for FP32 (cublasSgemm) matmuls. On A100 this
        // uses the tensor-core units for ~8x GEMM throughput vs plain FP32, with
        // negligible accuracy loss (10-bit mantissa) — no data/dtype change needed.
        cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    cublasHandle_t handle_;
};

} // namespace cuda
} // namespace at
