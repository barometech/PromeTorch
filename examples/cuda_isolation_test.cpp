// ============================================================================
// CUDA Isolation Test - Find exactly where the crash happens
// ============================================================================
// This test isolates each CUDA operation to find the crash point.
// Run with: cuda_isolation_test.exe
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#include "c10/cuda/CUDAAllocator.h"
#endif

#include <iostream>
#include <cstdlib>

using at::Tensor;

// ============================================================================
// Test helpers
// ============================================================================

void sync_cuda() {
#ifdef PT_USE_CUDA
    c10::cuda::cuda_synchronize();
#endif
}

void print_test(const char* name) {
    std::cout << "\n=== TEST: " << name << " ===" << std::endl;
    std::cout.flush();
}

void print_ok() {
    std::cout << "  [OK]" << std::endl;
    std::cout.flush();
}

void print_tensor_info(const Tensor& t, const char* name) {
    std::cout << "  " << name << ": shape=[";
    for (int64_t i = 0; i < t.dim(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << t.size(i);
    }
    std::cout << "], device=" << (t.is_cuda() ? "CUDA" : "CPU");
    std::cout << ", data_ptr=" << t.data_ptr<float>();
    std::cout << std::endl;
    std::cout.flush();
}

// ============================================================================
// Tests
// ============================================================================

bool test_cuda_available() {
    print_test("CUDA Available");
#ifdef PT_USE_CUDA
    std::cout << "  PT_USE_CUDA is defined" << std::endl;
    print_ok();
    return true;
#else
    std::cout << "  PT_USE_CUDA is NOT defined - CUDA not available!" << std::endl;
    return false;
#endif
}

bool test_cuda_allocator() {
    print_test("CUDA Allocator");
#ifdef PT_USE_CUDA
    std::cout << "  Getting allocator..." << std::endl;
    std::cout.flush();

    auto& alloc = c10::cuda::CUDACachingAllocator::get();
    std::cout << "  Got allocator instance" << std::endl;
    std::cout.flush();

    // Try to allocate 1MB
    size_t size = 1024 * 1024;
    std::cout << "  Allocating 1MB..." << std::endl;
    std::cout.flush();

    void* ptr = alloc.allocate(size);
    std::cout << "  Allocated at " << ptr << std::endl;
    std::cout.flush();

    std::cout << "  Deallocating..." << std::endl;
    std::cout.flush();
    alloc.deallocate(ptr);
    std::cout << "  Deallocated" << std::endl;
    std::cout.flush();

    sync_cuda();
    print_ok();
    return true;
#else
    return false;
#endif
}

bool test_cpu_tensor_create() {
    print_test("CPU Tensor Create");

    std::cout << "  Creating tensor..." << std::endl;
    std::cout.flush();

    Tensor t = at::randn({4, 4});
    print_tensor_info(t, "t");

    // Check values
    float* data = t.data_ptr<float>();
    std::cout << "  First 4 values: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    print_ok();
    return true;
}

bool test_to_cuda_small() {
    print_test("to_cuda (small tensor 4x4)");
#ifdef PT_USE_CUDA
    std::cout << "  Creating CPU tensor..." << std::endl;
    std::cout.flush();

    Tensor cpu_t = at::randn({4, 4});
    print_tensor_info(cpu_t, "cpu_t");

    std::cout << "  Calling at::to_cuda()..." << std::endl;
    std::cout.flush();

    Tensor cuda_t = at::to_cuda(cpu_t);

    std::cout << "  Synchronizing..." << std::endl;
    std::cout.flush();
    sync_cuda();

    print_tensor_info(cuda_t, "cuda_t");
    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_to_cuda_medium() {
    print_test("to_cuda (medium tensor 256x256)");
#ifdef PT_USE_CUDA
    std::cout << "  Creating CPU tensor 256x256..." << std::endl;
    std::cout.flush();

    Tensor cpu_t = at::randn({256, 256});
    print_tensor_info(cpu_t, "cpu_t");

    std::cout << "  Calling at::to_cuda()..." << std::endl;
    std::cout.flush();

    Tensor cuda_t = at::to_cuda(cpu_t);
    sync_cuda();

    print_tensor_info(cuda_t, "cuda_t");
    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_to_cuda_large() {
    print_test("to_cuda (large tensor 784x256 - like MLP weights)");
#ifdef PT_USE_CUDA
    std::cout << "  Creating CPU tensor 784x256..." << std::endl;
    std::cout.flush();

    Tensor cpu_t = at::randn({784, 256});
    print_tensor_info(cpu_t, "cpu_t");

    std::cout << "  Calling at::to_cuda()..." << std::endl;
    std::cout.flush();

    Tensor cuda_t = at::to_cuda(cpu_t);
    sync_cuda();

    print_tensor_info(cuda_t, "cuda_t");
    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_cuda_matmul() {
    print_test("CUDA matmul");
#ifdef PT_USE_CUDA
    // Create tensors on CPU first
    std::cout << "  Creating CPU tensors..." << std::endl;
    std::cout.flush();

    Tensor a_cpu = at::randn({32, 784});  // batch x input
    Tensor b_cpu = at::randn({784, 256}); // input x hidden

    std::cout << "  Transferring to CUDA..." << std::endl;
    std::cout.flush();

    Tensor a = at::to_cuda(a_cpu);
    Tensor b = at::to_cuda(b_cpu);
    sync_cuda();
    print_tensor_info(a, "a");
    print_tensor_info(b, "b");

    std::cout << "  Computing matmul via at::matmul..." << std::endl;
    std::cout.flush();

    Tensor c = at::matmul(a, b);
    sync_cuda();

    print_tensor_info(c, "c");

    // Verify shape
    if (c.size(0) != 32 || c.size(1) != 256) {
        std::cout << "  ERROR: Wrong output shape!" << std::endl;
        return false;
    }

    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_cuda_add_tensors() {
    print_test("CUDA Add Tensors");
#ifdef PT_USE_CUDA
    std::cout << "  Creating tensors..." << std::endl;
    std::cout.flush();

    Tensor a_cpu = at::randn({32, 256});
    Tensor b_cpu = at::randn({32, 256});

    Tensor a = at::to_cuda(a_cpu);
    Tensor b = at::to_cuda(b_cpu);
    sync_cuda();

    std::cout << "  Computing a + b..." << std::endl;
    std::cout.flush();

    // Use operator+ instead of at::add
    Tensor c = a + b;
    sync_cuda();

    print_tensor_info(c, "c");
    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_cuda_relu() {
    print_test("CUDA ReLU");
#ifdef PT_USE_CUDA
    std::cout << "  Creating tensor..." << std::endl;
    std::cout.flush();

    Tensor a_cpu = at::randn({32, 256});
    Tensor a = at::to_cuda(a_cpu);
    sync_cuda();

    std::cout << "  Computing relu..." << std::endl;
    std::cout.flush();

    Tensor b = at::relu(a);
    sync_cuda();

    print_tensor_info(b, "b");
    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_cuda_to_cpu() {
    print_test("CUDA to_cpu (round trip)");
#ifdef PT_USE_CUDA
    std::cout << "  Creating CPU tensor..." << std::endl;
    std::cout.flush();

    Tensor cpu1 = at::randn({4, 4});
    float original_val = cpu1.data_ptr<float>()[0];
    std::cout << "  Original value[0]: " << original_val << std::endl;

    std::cout << "  Transferring to CUDA..." << std::endl;
    std::cout.flush();

    Tensor cuda = at::to_cuda(cpu1);
    sync_cuda();

    std::cout << "  Transferring back to CPU..." << std::endl;
    std::cout.flush();

    Tensor cpu2 = at::to_cpu(cuda);
    sync_cuda();

    float final_val = cpu2.data_ptr<float>()[0];
    std::cout << "  After round-trip value[0]: " << final_val << std::endl;

    if (std::abs(original_val - final_val) > 1e-6) {
        std::cout << "  ERROR: Value mismatch!" << std::endl;
        return false;
    }

    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_forward_pass() {
    print_test("Forward Pass (Linear layer simulation)");
#ifdef PT_USE_CUDA
    // Simulate: y = relu(x @ W + b)
    int batch = 32;
    int in_features = 784;
    int out_features = 256;

    std::cout << "  Creating tensors..." << std::endl;
    std::cout.flush();

    Tensor x_cpu = at::randn({batch, in_features});
    Tensor W_cpu = at::randn({in_features, out_features});
    Tensor b_cpu = at::randn({1, out_features});  // Make b broadcastable

    std::cout << "  Transferring to CUDA..." << std::endl;
    std::cout.flush();

    Tensor x = at::to_cuda(x_cpu);
    Tensor W = at::to_cuda(W_cpu);
    Tensor b = at::to_cuda(b_cpu);
    sync_cuda();

    std::cout << "  Computing x @ W..." << std::endl;
    std::cout.flush();

    Tensor y1 = at::matmul(x, W);
    sync_cuda();
    print_tensor_info(y1, "y1 (after matmul)");

    std::cout << "  Computing + b..." << std::endl;
    std::cout.flush();

    Tensor y2 = y1 + b;
    sync_cuda();
    print_tensor_info(y2, "y2 (after add)");

    std::cout << "  Computing relu..." << std::endl;
    std::cout.flush();

    Tensor y3 = at::relu(y2);
    sync_cuda();
    print_tensor_info(y3, "y3 (after relu)");

    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

bool test_memory_stats() {
    print_test("Memory Statistics");
#ifdef PT_USE_CUDA
    auto& alloc = c10::cuda::CUDACachingAllocator::get();
    size_t allocated = alloc.get_allocated_memory();
    size_t cached = alloc.get_cached_memory();

    std::cout << "  Allocated: " << (allocated / 1048576.0) << " MB" << std::endl;
    std::cout << "  Cached: " << (cached / 1048576.0) << " MB" << std::endl;

    print_ok();
    return true;
#else
    std::cout << "  SKIPPED (no CUDA)" << std::endl;
    return false;
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "CUDA ISOLATION TEST" << std::endl;
    std::cout << "Finding where the crash happens..." << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout.flush();

    int passed = 0;
    int failed = 0;

    // Run tests in order of complexity
    if (test_cuda_available()) passed++; else failed++;
    if (test_cuda_allocator()) passed++; else failed++;
    if (test_cpu_tensor_create()) passed++; else failed++;
    if (test_to_cuda_small()) passed++; else failed++;
    if (test_to_cuda_medium()) passed++; else failed++;
    if (test_to_cuda_large()) passed++; else failed++;
    if (test_cuda_matmul()) passed++; else failed++;
    if (test_cuda_add_tensors()) passed++; else failed++;
    if (test_cuda_relu()) passed++; else failed++;
    if (test_cuda_to_cpu()) passed++; else failed++;
    if (test_forward_pass()) passed++; else failed++;
    if (test_memory_stats()) passed++; else failed++;

    std::cout << "\n============================================" << std::endl;
    std::cout << "RESULTS: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout.flush();

    return (failed > 0) ? 1 : 0;
}
