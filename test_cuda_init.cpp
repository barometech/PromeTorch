// Minimal CUDA initialization test
#include <iostream>
#ifdef PT_USE_CUDA
#include <cuda_runtime.h>
#endif

int main() {
    std::cout << "Starting CUDA test..." << std::endl;
#ifdef PT_USE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "CUDA devices: " << count << std::endl;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;

    // Try allocating memory
    void* ptr = nullptr;
    err = cudaMalloc(&ptr, 1024);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "cudaMalloc: OK" << std::endl;
    cudaFree(ptr);
    std::cout << "CUDA test PASSED" << std::endl;
#else
    std::cout << "CUDA not compiled in" << std::endl;
#endif
    return 0;
}
