// Minimal CUDA debug tests - isolate the problem
#include <iostream>
#include "aten/src/ATen/ATen.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"

using namespace at;
using namespace torch::nn;

void test_basic_cuda() {
    std::cout << "=== TEST 1: Basic CUDA tensor ===" << std::endl;
    Tensor cpu_t = at::randn({2, 3});
    std::cout << "CPU tensor created" << std::endl;
#ifdef PT_USE_CUDA
    Tensor gpu_t = at::to_cuda(cpu_t);
    std::cout << "GPU tensor created, is_cuda=" << gpu_t.is_cuda() << std::endl;
    Tensor back = at::to_cpu(gpu_t);
    std::cout << "Back to CPU OK" << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

void test_cuda_add() {
    std::cout << "=== TEST 2: CUDA add ===" << std::endl;
#ifdef PT_USE_CUDA
    Tensor a = at::to_cuda(at::randn({2, 3}));
    Tensor b = at::to_cuda(at::randn({2, 3}));
    std::cout << "Tensors on GPU" << std::endl;
    Tensor c = a.add(b);
    std::cout << "Add done, is_cuda=" << c.is_cuda() << std::endl;
    Tensor c_cpu = at::to_cpu(c);
    std::cout << "Result[0]=" << c_cpu.data_ptr<float>()[0] << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

void test_cuda_matmul() {
    std::cout << "=== TEST 3: CUDA matmul ===" << std::endl;
#ifdef PT_USE_CUDA
    Tensor a = at::to_cuda(at::randn({2, 3}));
    Tensor b = at::to_cuda(at::randn({3, 4}));
    std::cout << "Tensors on GPU" << std::endl;
    Tensor c = a.mm(b);
    std::cout << "Matmul done, shape=" << c.size(0) << "x" << c.size(1) << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

void test_linear_cpu() {
    std::cout << "=== TEST 4: Linear CPU ===" << std::endl;
    auto linear = std::make_shared<Linear>(10, 5);
    Tensor x = at::randn({2, 10});
    Tensor y = linear->forward(x);
    std::cout << "Output shape: " << y.size(0) << "x" << y.size(1) << std::endl;
    std::cout << "PASS\n" << std::endl;
}

void test_linear_cuda() {
    std::cout << "=== TEST 5: Linear CUDA ===" << std::endl;
#ifdef PT_USE_CUDA
    auto linear = std::make_shared<Linear>(10, 5);
    linear->to(c10::Device(c10::DeviceType::CUDA, 0));
    std::cout << "Linear moved to CUDA" << std::endl;
    
    Tensor x = at::to_cuda(at::randn({2, 10}));
    std::cout << "Input on CUDA" << std::endl;
    
    Tensor y = linear->forward(x);
    std::cout << "Forward done, is_cuda=" << y.is_cuda() << std::endl;
    std::cout << "Output shape: " << y.size(0) << "x" << y.size(1) << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

void test_conv2d_cpu() {
    std::cout << "=== TEST 6: Conv2d CPU ===" << std::endl;
    auto conv = std::make_shared<Conv2d>(1, 8, 3, 1, 1);  // in=1, out=8, k=3, s=1, p=1
    Tensor x = at::randn({1, 1, 8, 8});  // Small image
    std::cout << "Input shape: 1x1x8x8" << std::endl;
    Tensor y = conv->forward(x);
    std::cout << "Output shape: " << y.size(0) << "x" << y.size(1) << "x" << y.size(2) << "x" << y.size(3) << std::endl;
    std::cout << "PASS\n" << std::endl;
}

void test_conv2d_cuda() {
    std::cout << "=== TEST 7: Conv2d CUDA ===" << std::endl;
#ifdef PT_USE_CUDA
    auto conv = std::make_shared<Conv2d>(1, 8, 3, 1, 1);
    conv->to(c10::Device(c10::DeviceType::CUDA, 0));
    std::cout << "Conv2d moved to CUDA" << std::endl;
    
    Tensor x = at::to_cuda(at::randn({1, 1, 8, 8}));
    std::cout << "Input on CUDA" << std::endl;
    
    Tensor y = conv->forward(x);
    std::cout << "Forward done, is_cuda=" << y.is_cuda() << std::endl;
    std::cout << "Output shape: " << y.size(0) << "x" << y.size(1) << "x" << y.size(2) << "x" << y.size(3) << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

void test_maxpool_cpu() {
    std::cout << "=== TEST 8: MaxPool2d CPU ===" << std::endl;
    auto pool = std::make_shared<MaxPool2d>(2, 2);
    Tensor x = at::randn({1, 1, 8, 8});
    Tensor y = pool->forward(x);
    std::cout << "Output shape: " << y.size(0) << "x" << y.size(1) << "x" << y.size(2) << "x" << y.size(3) << std::endl;
    std::cout << "PASS\n" << std::endl;
}

void test_maxpool_cuda() {
    std::cout << "=== TEST 9: MaxPool2d CUDA ===" << std::endl;
#ifdef PT_USE_CUDA
    auto pool = std::make_shared<MaxPool2d>(2, 2);
    // MaxPool has no parameters, just operates on input
    
    Tensor x = at::to_cuda(at::randn({1, 1, 8, 8}));
    std::cout << "Input on CUDA" << std::endl;
    
    Tensor y = pool->forward(x);
    std::cout << "Forward done, is_cuda=" << y.is_cuda() << std::endl;
    std::cout << "Output shape: " << y.size(0) << "x" << y.size(1) << "x" << y.size(2) << "x" << y.size(3) << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

void test_relu_cuda() {
    std::cout << "=== TEST 10: ReLU CUDA ===" << std::endl;
#ifdef PT_USE_CUDA
    auto relu = std::make_shared<ReLU>();
    Tensor x = at::to_cuda(at::randn({2, 3}));
    Tensor y = relu->forward(x);
    std::cout << "Forward done, is_cuda=" << y.is_cuda() << std::endl;
#endif
    std::cout << "PASS\n" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "CUDA Debug Tests - Isolating Problems" << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    try {
        test_basic_cuda();
        test_cuda_add();
        test_cuda_matmul();
        test_linear_cpu();
        test_linear_cuda();
        test_conv2d_cpu();
        test_conv2d_cuda();
        test_maxpool_cpu();
        test_maxpool_cuda();
        test_relu_cuda();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n!!! EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
