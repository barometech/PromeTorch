// ============================================================================
// Memory Leak Test - Minimal MLP Training Loop
// ============================================================================
// Purpose: Isolate and diagnose GPU memory leaks in autograd
// Usage: test_mem_leak.exe --device cuda --iterations 50
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/ATen.h"

#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#include "c10/cuda/CUDAAllocator.h"
#endif

#include <iostream>
#include <chrono>
#include <vector>

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using at::Tensor;

// Global device
static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

// ============================================================================
// Simple MLP - No convolutions, minimal complexity
// ============================================================================

class SimpleMLP : public Module {
public:
    SimpleMLP(int64_t input_dim, int64_t hidden_dim, int64_t output_dim)
        : Module("SimpleMLP")
    {
        fc1 = std::make_shared<Linear>(input_dim, hidden_dim);
        fc2 = std::make_shared<Linear>(hidden_dim, hidden_dim);
        fc3 = std::make_shared<Linear>(hidden_dim, output_dim);
        relu = std::make_shared<ReLU>();

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    Tensor forward(const Tensor& x) override {
        Tensor h = fc1->forward(x);
        h = relu->forward(h);
        h = fc2->forward(h);
        h = relu->forward(h);
        h = fc3->forward(h);
        return h;
    }

private:
    std::shared_ptr<Linear> fc1, fc2, fc3;
    std::shared_ptr<ReLU> relu;
};

// ============================================================================
// Memory Statistics
// ============================================================================

void print_memory_stats(int64_t iter) {
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        auto& alloc = c10::cuda::CUDACachingAllocator::get();
        size_t allocated = alloc.get_allocated_memory();
        size_t cached = alloc.get_cached_memory();

        std::cout << "[MEM iter " << iter << "] "
                  << "allocated=" << (allocated / 1048576.0) << " MB, "
                  << "cached=" << (cached / 1048576.0) << " MB"
                  << std::endl;
    }
#endif
}

void print_node_stats() {
    std::cout << "[NODES] "
              << "created=" << torch::autograd::g_nodes_created.load() << ", "
              << "destroyed=" << torch::autograd::g_nodes_destroyed.load() << ", "
              << "released=" << torch::autograd::g_nodes_released.load() << ", "
              << "alive=" << (torch::autograd::g_nodes_created.load() -
                             torch::autograd::g_nodes_destroyed.load())
              << std::endl;
}

// ============================================================================
// Transfer functions
// ============================================================================

inline Tensor to_device(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.is_cuda() && !t.is_cuda()) {
        return at::to_cuda(t);
    }
#endif
    return t;
}

inline Tensor tensor_to_cpu(const Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) {
        return at::to_cpu(t);
    }
#endif
    return t;
}

// ============================================================================
// Clear entire autograd graph from parameters
// ============================================================================

void clear_param_grad_graph(Module& model) {
    for (auto* param : model.parameters()) {
        // Clear gradient's grad_fn (breaks chain to old graph)
        if (param->grad().defined()) {
            Tensor grad = param->grad();  // Store as lvalue
            torch::autograd::clear_grad_fn(grad);
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Defaults
    std::string device_str = "cpu";
    int64_t batch_size = 32;
    int64_t input_dim = 784;   // Like MNIST flattened
    int64_t hidden_dim = 256;
    int64_t output_dim = 10;
    int64_t iterations = 50;
    bool verbose = false;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--hidden" && i + 1 < argc) {
            hidden_dim = std::stoll(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoll(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
    }

    // Set device
    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA device" << std::endl;
#else
        std::cerr << "CUDA not available, using CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU device" << std::endl;
    }

    std::cout << "Config: batch=" << batch_size
              << ", input=" << input_dim
              << ", hidden=" << hidden_dim
              << ", output=" << output_dim
              << ", iterations=" << iterations
              << std::endl;

    // Register autograd meta factory
    c10::set_autograd_meta_factory(&torch::autograd::create_autograd_meta_impl);

    // Create model
    auto model = std::make_shared<SimpleMLP>(input_dim, hidden_dim, output_dim);

    // Count parameters
    int64_t total_params = 0;
    for (auto* p : model->parameters()) {
        total_params += p->data().numel();
    }
    std::cout << "Model parameters: " << total_params << std::endl;

    // Move to device
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Optimizer
    AdamOptions opts(0.001f);
    Adam optimizer(model->parameters(), opts);

    // Loss function
    CrossEntropyLoss criterion;

    // Reset node counters
    torch::autograd::g_nodes_created = 0;
    torch::autograd::g_nodes_destroyed = 0;
    torch::autograd::g_nodes_released = 0;

    std::cout << "\n=== Starting Training Loop ===" << std::endl;
    print_memory_stats(0);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int64_t iter = 1; iter <= iterations; ++iter) {
        // Create random batch
        Tensor inputs = at::randn({batch_size, input_dim});
        Tensor targets = at::empty({batch_size});

        // Random labels 0-9
        float* tgt_ptr = targets.mutable_data_ptr<float>();
        for (int64_t i = 0; i < batch_size; ++i) {
            tgt_ptr[i] = static_cast<float>(rand() % output_dim);
        }

        inputs = to_device(inputs);
        targets = to_device(targets);

        // Forward
        optimizer.zero_grad();
        Tensor logits = model->forward(inputs);
        Tensor loss = criterion.forward(logits, targets);

        // Get loss value before backward
        float loss_val = tensor_to_cpu(loss).data_ptr<float>()[0];

        // Backward
        torch::autograd::backward({loss});

        // === CRITICAL: Clear autograd graph ===
        // 1. Clear grad_fn from loss and logits
        torch::autograd::clear_grad_fn(loss);
        torch::autograd::clear_grad_fn(logits);

        // 2. Clear grad_fn from parameter gradients
        clear_param_grad_graph(*model);

        // 3. Release tensor references
        loss = Tensor();
        logits = Tensor();
        inputs = Tensor();
        targets = Tensor();

        // Optimizer step
        optimizer.step();

        // === CUDA memory cleanup ===
#ifdef PT_USE_CUDA
        if (g_device.is_cuda()) {
            c10::cuda::cuda_synchronize();
            // Only empty cache every 10 iterations to see accumulation
            if (iter % 10 == 0) {
                c10::cuda::CUDACachingAllocator::get().empty_cache();
            }
        }
#endif

        // Print stats
        if (iter <= 10 || iter % 10 == 0 || verbose) {
            std::cout << "iter " << iter << ": loss=" << loss_val << std::endl;
            print_memory_stats(iter);
            print_node_stats();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << elapsed << " ms" << std::endl;
    std::cout << "Time per iteration: " << (elapsed / iterations) << " ms" << std::endl;

    print_memory_stats(iterations);
    print_node_stats();

    // Final memory check
    std::cout << "\n=== Final Memory Check ===" << std::endl;
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        c10::cuda::CUDACachingAllocator::get().empty_cache();
        print_memory_stats(-1);
    }
#endif

    // Check for leaks
    int64_t alive = torch::autograd::g_nodes_created.load() -
                    torch::autograd::g_nodes_destroyed.load();
    if (alive > 0) {
        std::cout << "\n!!! WARNING: " << alive << " nodes still alive (potential leak) !!!" << std::endl;
    } else {
        std::cout << "\nNo node leaks detected." << std::endl;
    }

    return 0;
}
