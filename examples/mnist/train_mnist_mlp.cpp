// ============================================================================
// MNIST Training Example - MLP (GPU-Compatible)
// ============================================================================
// Simple MLP model using only Linear layers (works on CUDA)
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/optim/adamkiller.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "c10/core/Allocator.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#include "c10/cuda/CUDAAllocator.h"
#endif
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

// Global device
static c10::Device g_device = c10::Device(c10::DeviceType::CPU);

inline at::Tensor to_device(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (g_device.type() == c10::DeviceType::CUDA) {
        return at::to_cuda(t);
    }
#endif
    return t;
}

inline at::Tensor move_to_cpu(const at::Tensor& t) {
#ifdef PT_USE_CUDA
    if (t.is_cuda()) {
        return at::to_cpu(t);
    }
#endif
    return t;
}

// ============================================================================
// MNIST Data Loading (IDX format)
// ============================================================================

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    int32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = bswap32(magic);
    num_images = bswap32(num_images);
    rows = bswap32(rows);
    cols = bswap32(cols);

    std::cout << "MNIST images: " << num_images << " x " << rows << "x" << cols << std::endl;

    std::vector<std::vector<uint8_t>> images(num_images);
    for (int i = 0; i < num_images; ++i) {
        images[i].resize(rows * cols);
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    return images;
}

std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << std::endl;
        return {};
    }

    int32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = bswap32(magic);
    num_labels = bswap32(num_labels);

    std::cout << "MNIST labels: " << num_labels << std::endl;

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

// ============================================================================
// Simple MLP Model (GPU-Compatible - no Conv2d)
// ============================================================================

// Simple single-layer model for gradient testing
class SimpleLinear : public Module {
public:
    SimpleLinear() : Module("SimpleLinear") {
        // Just one linear layer: 784 -> 10
        fc = std::make_shared<Linear>(784, 10);
        register_module("fc", fc);
    }

    Tensor forward(const Tensor& x) override {
        return fc->forward(x);  // [B, 10]
    }

private:
    std::shared_ptr<Linear> fc;
};

class MNISTMLP : public Module {
public:
    MNISTMLP() : Module("MNISTMLP") {
        // Full 4-layer MLP: 784 -> 512 -> 256 -> 128 -> 10
        // Use fused_relu=true for first 3 layers (Linear+ReLU in one op)
        // fc4 feeds into CrossEntropy which handles softmax — no fused relu
        register_module("fc1", std::make_shared<Linear>(784, 512, true, true));
        register_module("fc2", std::make_shared<Linear>(512, 256, true, true));
        register_module("fc3", std::make_shared<Linear>(256, 128, true, true));
        register_module("fc4", std::make_shared<Linear>(128, 10));

        // Cache layer names for forward pass (in order)
        layer_names_ = {"fc1", "fc2", "fc3", "fc4"};
        relu_layers_ = {true, true, true, false};
    }

    Tensor forward(const Tensor& x) override {
        // Forward through submodule registry (supports module replacement)
        Tensor h = x;
        for (size_t i = 0; i < layer_names_.size(); ++i) {
            auto layer = get_submodule(layer_names_[i]);
            h = layer->forward(h);
            // LowRankLinear doesn't have fused relu, so apply it separately
            if (relu_layers_[i] && std::dynamic_pointer_cast<LowRankLinear>(layer)) {
                h = torch::autograd::relu_autograd(h);
            }
        }
        return h;
    }

    // =========================================================================
    // Manual forward: raw sgemm, saves intermediates for manual_backward()
    // ZERO autograd overhead — no graph nodes, no shared_ptr, no metadata
    // =========================================================================
    void manual_forward(const Tensor& input,
                        Tensor& h1, Tensor& h2, Tensor& h3, Tensor& logits) {
        // Get weight/bias pointers (these don't change between batches)
        auto* p_W1 = get_submodule("fc1")->get_parameter("weight");
        auto* p_b1 = get_submodule("fc1")->get_parameter("bias");
        auto* p_W2 = get_submodule("fc2")->get_parameter("weight");
        auto* p_b2 = get_submodule("fc2")->get_parameter("bias");
        auto* p_W3 = get_submodule("fc3")->get_parameter("weight");
        auto* p_b3 = get_submodule("fc3")->get_parameter("bias");
        auto* p_W4 = get_submodule("fc4")->get_parameter("weight");
        auto* p_b4 = get_submodule("fc4")->get_parameter("bias");

        const int64_t B = input.size(0);

        // Layer 1: h1 = relu(input @ W1^T + b1)  [B,784] @ [784,512] + [512] -> [B,512]
        at::native::hot::sgemm_nt(B, 784, 512, 1.0f,
            input.data_ptr<float>(), 784,
            p_W1->data().data_ptr<float>(), 784,
            0.0f, h1.mutable_data_ptr<float>(), 512);
        at::native::hot::bias_relu_fused(h1.mutable_data_ptr<float>(),
            p_b1->data().data_ptr<float>(), B, 512);

        // Layer 2: h2 = relu(h1 @ W2^T + b2)  [B,512] @ [512,256] + [256] -> [B,256]
        at::native::hot::sgemm_nt(B, 512, 256, 1.0f,
            h1.data_ptr<float>(), 512,
            p_W2->data().data_ptr<float>(), 512,
            0.0f, h2.mutable_data_ptr<float>(), 256);
        at::native::hot::bias_relu_fused(h2.mutable_data_ptr<float>(),
            p_b2->data().data_ptr<float>(), B, 256);

        // Layer 3: h3 = relu(h2 @ W3^T + b3)  [B,256] @ [256,128] + [128] -> [B,128]
        at::native::hot::sgemm_nt(B, 256, 128, 1.0f,
            h2.data_ptr<float>(), 256,
            p_W3->data().data_ptr<float>(), 256,
            0.0f, h3.mutable_data_ptr<float>(), 128);
        at::native::hot::bias_relu_fused(h3.mutable_data_ptr<float>(),
            p_b3->data().data_ptr<float>(), B, 128);

        // Layer 4: logits = h3 @ W4^T + b4  [B,128] @ [128,10] + [10] -> [B,10]
        at::native::hot::sgemm_nt(B, 128, 10, 1.0f,
            h3.data_ptr<float>(), 128,
            p_W4->data().data_ptr<float>(), 128,
            0.0f, logits.mutable_data_ptr<float>(), 10);
        // bias add (no relu for last layer)
        at::native::hot::add_broadcast_loop(
            logits.data_ptr<float>(), p_b4->data().data_ptr<float>(),
            logits.mutable_data_ptr<float>(), B, 10, 1.0f);
    }

private:
    std::vector<std::string> layer_names_;
    std::vector<bool> relu_layers_;
};

// =============================================================================
// Manual backward: computes ALL gradients for 4-layer MLP in ONE function call
// ZERO autograd overhead: no graph, no nodes, no shared_ptr, no priority queue
// =============================================================================
// Model: fc1(relu) -> fc2(relu) -> fc3(relu) -> fc4 -> cross_entropy
//
// All gradient and scratch tensors are PRE-ALLOCATED and REUSED across batches.
// The only allocation per batch is the int64_t targets conversion (on stack).
static void manual_backward(
    // Inputs
    const Tensor& input,        // [B, 784]
    const Tensor& target,       // [B] float (class indices stored as float)
    // Layer outputs (saved from manual_forward)
    const Tensor& h1,           // [B, 512] after relu
    const Tensor& h2,           // [B, 256] after relu
    const Tensor& h3,           // [B, 128] after relu
    const Tensor& logits,       // [B, 10]
    // Weights (read-only, for grad_input computation)
    const Tensor& W1, const Tensor& W2,
    const Tensor& W3, const Tensor& W4,
    // Output: parameter gradients (pre-allocated, reused!)
    Tensor& grad_W1, Tensor& grad_b1,
    Tensor& grad_W2, Tensor& grad_b2,
    Tensor& grad_W3, Tensor& grad_b3,
    Tensor& grad_W4, Tensor& grad_b4,
    // Scratch buffers for inter-layer gradients (pre-allocated, reused!)
    Tensor& grad_h4,  // [B, 10]  — softmax grad from cross-entropy
    Tensor& grad_h3,  // [B, 128] — after fc4 backward + relu mask
    Tensor& grad_h2,  // [B, 256] — after fc3 backward + relu mask
    Tensor& grad_h1   // [B, 512] — after fc2 backward + relu mask
) {
    const int64_t B = input.size(0);

    // Step 1: Cross-entropy gradient -> grad_h4 [B, 10]
    // softmax(logits) - one_hot(target), divided by batch_size
    // Convert float targets to int64_t
    const float* tgt_f = target.data_ptr<float>();
    std::vector<int64_t> targets_i64(B);
    for (int64_t i = 0; i < B; ++i) {
        targets_i64[i] = static_cast<int64_t>(tgt_f[i]);
    }
    float loss_scratch;  // not used, but cross_entropy_fused writes to it
    at::native::hot::cross_entropy_fused(
        logits.data_ptr<float>(), targets_i64.data(),
        &loss_scratch,
        grad_h4.mutable_data_ptr<float>(), B, 10);

    // Step 2: fc4 backward (no relu — last layer)
    // grad_W4 = grad_h4^T @ h3    [10,B] @ [B,128] = [10,128]
    at::native::hot::sgemm_tn(B, 10, 128, 1.0f,
        grad_h4.data_ptr<float>(), 10,
        h3.data_ptr<float>(), 128,
        0.0f, grad_W4.mutable_data_ptr<float>(), 128);
    // grad_b4 = grad_h4.sum(dim=0)  [10]
    at::native::hot::col_sum(grad_h4.data_ptr<float>(),
        grad_b4.mutable_data_ptr<float>(), B, 10);
    // grad_h3 = grad_h4 @ W4  [B,10] @ [10,128] = [B,128]
    // W4 is stored [10,128], so this is sgemm(B, K=10, N=128)
    at::native::hot::sgemm(B, 10, 128, 1.0f,
        grad_h4.data_ptr<float>(), 10,
        W4.data_ptr<float>(), 128,
        0.0f, grad_h3.mutable_data_ptr<float>(), 128);
    // Apply relu mask for h3: grad_h3[i] = (h3[i] > 0) ? grad_h3[i] : 0
    at::native::hot::relu_mask_mul(grad_h3.data_ptr<float>(),
        h3.data_ptr<float>(), grad_h3.mutable_data_ptr<float>(), B * 128);

    // Step 3: fc3 backward
    // grad_W3 = grad_h3^T @ h2    [128,B] @ [B,256] = [128,256]
    at::native::hot::sgemm_tn(B, 128, 256, 1.0f,
        grad_h3.data_ptr<float>(), 128,
        h2.data_ptr<float>(), 256,
        0.0f, grad_W3.mutable_data_ptr<float>(), 256);
    // grad_b3 = grad_h3.sum(dim=0)  [128]
    at::native::hot::col_sum(grad_h3.data_ptr<float>(),
        grad_b3.mutable_data_ptr<float>(), B, 128);
    // grad_h2 = grad_h3 @ W3  [B,128] @ [128,256] = [B,256]
    at::native::hot::sgemm(B, 128, 256, 1.0f,
        grad_h3.data_ptr<float>(), 128,
        W3.data_ptr<float>(), 256,
        0.0f, grad_h2.mutable_data_ptr<float>(), 256);
    // relu mask for h2
    at::native::hot::relu_mask_mul(grad_h2.data_ptr<float>(),
        h2.data_ptr<float>(), grad_h2.mutable_data_ptr<float>(), B * 256);

    // Step 4: fc2 backward
    // grad_W2 = grad_h2^T @ h1    [256,B] @ [B,512] = [256,512]
    at::native::hot::sgemm_tn(B, 256, 512, 1.0f,
        grad_h2.data_ptr<float>(), 256,
        h1.data_ptr<float>(), 512,
        0.0f, grad_W2.mutable_data_ptr<float>(), 512);
    // grad_b2 = grad_h2.sum(dim=0)  [256]
    at::native::hot::col_sum(grad_h2.data_ptr<float>(),
        grad_b2.mutable_data_ptr<float>(), B, 256);
    // grad_h1 = grad_h2 @ W2  [B,256] @ [256,512] = [B,512]
    at::native::hot::sgemm(B, 256, 512, 1.0f,
        grad_h2.data_ptr<float>(), 256,
        W2.data_ptr<float>(), 512,
        0.0f, grad_h1.mutable_data_ptr<float>(), 512);
    // relu mask for h1
    at::native::hot::relu_mask_mul(grad_h1.data_ptr<float>(),
        h1.data_ptr<float>(), grad_h1.mutable_data_ptr<float>(), B * 512);

    // Step 5: fc1 backward (no grad_input needed — input is data, not a parameter)
    // grad_W1 = grad_h1^T @ input  [512,B] @ [B,784] = [512,784]
    at::native::hot::sgemm_tn(B, 512, 784, 1.0f,
        grad_h1.data_ptr<float>(), 512,
        input.data_ptr<float>(), 784,
        0.0f, grad_W1.mutable_data_ptr<float>(), 784);
    // grad_b1 = grad_h1.sum(dim=0)  [512]
    at::native::hot::col_sum(grad_h1.data_ptr<float>(),
        grad_b1.mutable_data_ptr<float>(), B, 512);
    // Skip grad_input — we don't backprop into input data
}

// ============================================================================
// Training
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    std::string device_str = "cpu";
    int64_t batch_size = 64;
    int64_t epochs = 5;
    float lr = 0.001f;  // Standard learning rate for MLP
    bool do_compress = false;
    double compress_ratio = 0.5;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoll(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--compress") {
            do_compress = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                compress_ratio = std::stod(argv[++i]);
            }
        }
    }

    // Set device
    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
        std::cout << "Using CUDA" << std::endl;
#else
        std::cerr << "CUDA not available, using CPU" << std::endl;
#endif
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    // Load data
    std::cout << "Loading MNIST from: " << data_dir << std::endl;
    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    auto train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    auto test_images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    auto test_labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    if (train_images.empty()) {
        std::cerr << "Failed to load MNIST. Download from http://yann.lecun.com/exdb/mnist/" << std::endl;
        return 1;
    }

    int64_t n_train = train_images.size();
    int64_t n_test = test_images.size();

    // =========== NUMERICAL GRADIENT CHECK ===========
    std::cout << "\n=== Running Numerical Gradient Check ===\n";
    {
        // Create simple one-layer model for testing
        auto test_model = std::make_shared<SimpleLinear>();
        CrossEntropyLoss test_criterion;

        // Create small test batch - use batch=1 to eliminate batch averaging issues
        int64_t test_B = 1;
        Tensor test_inputs = at::empty({test_B, 784});
        Tensor test_targets = at::empty({test_B});
        float* in_ptr = test_inputs.mutable_data_ptr<float>();
        float* tgt_ptr = test_targets.mutable_data_ptr<float>();

        // Fill with random data
        for (int64_t i = 0; i < test_B * 784; ++i) {
            in_ptr[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        for (int64_t i = 0; i < test_B; ++i) {
            tgt_ptr[i] = static_cast<float>(rand() % 10);
        }

        // Get first weight parameter from the model's parameters list
        auto params = test_model->parameters();
        if (params.empty()) {
            std::cout << "ERROR: No parameters found in model!\n";
            return 1;
        }
        auto* weight_param = params[0];  // First parameter is weight
        std::cout << "Weight shape: [" << weight_param->size(0) << ", " << weight_param->size(1) << "]\n";

        Tensor weight = weight_param->data();
        float* w_ptr = weight.mutable_data_ptr<float>();

        // Compute analytical gradient
        std::cout << "Running forward pass...\n";
        Tensor logits = test_model->forward(test_inputs);
        std::cout << "Computing loss...\n";
        Tensor loss = test_criterion.forward(logits, test_targets);
        float loss_val = loss.data_ptr<float>()[0];
        std::cout << "Initial loss: " << loss_val << std::endl;

        std::cout << "Running backward pass...\n";
        torch::autograd::backward({loss});
        std::cout << "Getting gradient...\n";
        Tensor analytical_grad = weight_param->grad();
        if (!analytical_grad.defined()) {
            std::cout << "ERROR: Gradient is not defined after backward!\n";
            return 1;
        }
        std::cout << "Gradient shape: [" << analytical_grad.size(0) << ", " << analytical_grad.size(1) << "]\n";

        // Compute numerical gradient for first few weights
        float eps = 1e-4f;
        std::cout << "Batch size: " << test_B << std::endl;
        std::cout << "Input shape: [" << test_inputs.size(0) << ", " << test_inputs.size(1) << "]\n";
        std::cout << "Checking 10 weights (across different rows):\n";
        bool grad_ok = true;
        // Check indices from different parts of the weight matrix
        std::vector<int> indices_to_check = {0, 1, 2, 784, 785, 1568, 2352, 3136, 3920, 7839};
        for (int idx = 0; idx < (int)indices_to_check.size(); ++idx) {
            int i = indices_to_check[idx];
            float orig = w_ptr[i];

            // f(w + eps)
            w_ptr[i] = orig + eps;
            Tensor logits_plus = test_model->forward(test_inputs);
            Tensor loss_plus = test_criterion.forward(logits_plus, test_targets);
            float loss_p = loss_plus.data_ptr<float>()[0];

            // f(w - eps)
            w_ptr[i] = orig - eps;
            Tensor logits_minus = test_model->forward(test_inputs);
            Tensor loss_minus = test_criterion.forward(logits_minus, test_targets);
            float loss_m = loss_minus.data_ptr<float>()[0];

            // Restore
            w_ptr[i] = orig;

            float numerical = (loss_p - loss_m) / (2 * eps);
            float analytical = analytical_grad.data_ptr<float>()[i];
            float rel_error = std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-8f);

            std::cout << "  w[" << i << "]: numerical=" << numerical
                      << " analytical=" << analytical
                      << " rel_error=" << rel_error;
            if (rel_error > 0.01f) {
                std::cout << " MISMATCH!";
                grad_ok = false;
            }
            std::cout << std::endl;
        }

        if (!grad_ok) {
            std::cout << "\n*** GRADIENT CHECK FAILED! ***\n";
            std::cout << "There is a bug in the gradient computation.\n\n";
        } else {
            std::cout << "\n*** GRADIENT CHECK PASSED! ***\n\n";
        }
    }
    // =========== END GRADIENT CHECK ===========

    // =========== SINGLE STEP TEST ===========
    // Verify that one gradient step reduces loss
    std::cout << "\n=== Single Step Gradient Descent Test ===\n";
    {
        auto test_model = std::make_shared<SimpleLinear>();

        // Create test batch
        int64_t test_B = 32;
        Tensor test_inputs = at::empty({test_B, 784});
        Tensor test_targets = at::empty({test_B});
        float* in_ptr = test_inputs.mutable_data_ptr<float>();
        float* tgt_ptr = test_targets.mutable_data_ptr<float>();

        // Fill with normalized random data
        for (int64_t i = 0; i < test_B * 784; ++i) {
            float pixel = static_cast<float>(rand()) / RAND_MAX;
            in_ptr[i] = (pixel - 0.1307f) / 0.3081f;
        }
        for (int64_t i = 0; i < test_B; ++i) {
            tgt_ptr[i] = static_cast<float>(rand() % 10);
        }

        CrossEntropyLoss test_criterion;

        // Compute initial loss
        Tensor logits1 = test_model->forward(test_inputs);
        Tensor loss1 = test_criterion.forward(logits1, test_targets);
        float loss1_val = loss1.data_ptr<float>()[0];
        std::cout << "Initial loss: " << loss1_val << std::endl;

        // Compute gradients
        torch::autograd::backward({loss1});

        // Manual SGD step with very small lr
        float tiny_lr = 0.0001f;
        for (auto* param : test_model->parameters()) {
            if (param->grad().defined()) {
                Tensor g = param->grad();
                Tensor w = param->data();

                // Check gradient stats
                float g_sum = 0, w_sum = 0;
                const float* gd = g.data_ptr<float>();
                const float* wd = w.data_ptr<float>();
                for (int64_t i = 0; i < g.numel(); ++i) {
                    g_sum += gd[i];
                    w_sum += wd[i];
                }
                std::cout << "Param: numel=" << g.numel()
                          << ", grad_sum=" << g_sum
                          << ", weight_sum=" << w_sum << std::endl;

                // Update: w = w - lr * g
                w.sub_(g, at::Scalar(tiny_lr));
            }
        }

        // Compute loss after update (need fresh forward pass)
        test_model->zero_grad();
        Tensor logits2 = test_model->forward(test_inputs);
        Tensor loss2 = test_criterion.forward(logits2, test_targets);
        float loss2_val = loss2.data_ptr<float>()[0];
        std::cout << "Loss after step: " << loss2_val << std::endl;

        if (loss2_val < loss1_val) {
            std::cout << "*** PASS: Loss decreased by " << (loss1_val - loss2_val) << " ***\n\n";
        } else {
            std::cout << "*** FAIL: Loss INCREASED by " << (loss2_val - loss1_val) << " ***\n";
            std::cout << "This indicates gradient direction is WRONG!\n\n";
        }
    }
    // =========== END SINGLE STEP TEST ===========

    // =========== ADAM vs SGD vs ADAMKILLER TEST ===========
    // Compare Adam, SGD and AdamKiller on same problem
    std::cout << "\n=== Adam vs SGD vs AdamKiller Comparison Test ===\n";
    {
        // Simple quadratic: L = sum(w^2) / 2, grad = w
        // Optimal w = 0

        // Test with Adam
        std::cout << "Testing ADAM:" << std::endl;
        {
            Tensor w_data = at::ones({10});  // Start at 1
            Parameter w(w_data);
            AdamOptions opts(0.001);
            Adam optimizer({&w}, opts);

            for (int step = 1; step <= 10; ++step) {
                // Compute grad = w
                Tensor grad = w.data().clone();
                w.set_grad(grad);

                float w0 = w.data().data_ptr<float>()[0];
                float loss = 0;
                for (int i = 0; i < 10; ++i) {
                    float wi = w.data().data_ptr<float>()[i];
                    loss += wi * wi;
                }
                loss /= 2;

                optimizer.step();

                float w0_new = w.data().data_ptr<float>()[0];
                std::cout << "  Step " << step << ": w[0]=" << w0 << " -> " << w0_new
                          << ", loss=" << loss << std::endl;
            }
        }

        // Test with SGD
        std::cout << "Testing SGD:" << std::endl;
        {
            Tensor w_data = at::ones({10});  // Start at 1
            Parameter w(w_data);
            SGDOptions opts(0.001);
            SGD optimizer({&w}, opts);

            for (int step = 1; step <= 10; ++step) {
                // Compute grad = w
                Tensor grad = w.data().clone();
                w.set_grad(grad);

                float w0 = w.data().data_ptr<float>()[0];
                float loss = 0;
                for (int i = 0; i < 10; ++i) {
                    float wi = w.data().data_ptr<float>()[i];
                    loss += wi * wi;
                }
                loss /= 2;

                optimizer.step();

                float w0_new = w.data().data_ptr<float>()[0];
                std::cout << "  Step " << step << ": w[0]=" << w0 << " -> " << w0_new
                          << ", loss=" << loss << std::endl;
            }
        }

        // Test with AdamKiller (AGGRESSIVE mode for max speed)
        std::cout << "Testing ADAMKILLER (aggressive):" << std::endl;
        {
            Tensor w_data = at::ones({10});  // Start at 1
            Parameter w(w_data);
            AdamKiller optimizer = make_adamkiller_aggressive({&w}, 0.001);

            for (int step = 1; step <= 10; ++step) {
                // Compute grad = w
                Tensor grad = w.data().clone();
                w.set_grad(grad);

                float w0 = w.data().data_ptr<float>()[0];
                float loss = 0;
                for (int i = 0; i < 10; ++i) {
                    float wi = w.data().data_ptr<float>()[i];
                    loss += wi * wi;
                }
                loss /= 2;

                optimizer.step();

                float w0_new = w.data().data_ptr<float>()[0];
                std::cout << "  Step " << step << ": w[0]=" << w0 << " -> " << w0_new
                          << ", loss=" << loss << std::endl;
            }
        }

        // Test with AdamKiller (STABLE mode)
        std::cout << "Testing ADAMKILLER (stable):" << std::endl;
        {
            Tensor w_data = at::ones({10});  // Start at 1
            Parameter w(w_data);
            AdamKiller optimizer = make_adamkiller_stable({&w}, 0.001);

            for (int step = 1; step <= 10; ++step) {
                // Compute grad = w
                Tensor grad = w.data().clone();
                w.set_grad(grad);

                float w0 = w.data().data_ptr<float>()[0];
                float loss = 0;
                for (int i = 0; i < 10; ++i) {
                    float wi = w.data().data_ptr<float>()[i];
                    loss += wi * wi;
                }
                loss /= 2;

                optimizer.step();

                float w0_new = w.data().data_ptr<float>()[0];
                std::cout << "  Step " << step << ": w[0]=" << w0 << " -> " << w0_new
                          << ", loss=" << loss << std::endl;
            }
        }
        std::cout << std::endl;
    }
    // =========== END OPTIMIZER COMPARISON TEST ===========

    // Create model
    auto model = std::make_shared<MNISTMLP>();
    std::cout << "MLP Model created (784->512->256->128->10)" << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Optimizer - Use vanilla SGD for fair comparison with PyTorch
    // AdamKiller optimizer = make_adamkiller_aggressive(model->parameters(), lr);
    // std::cout << "AdamKiller optimizer (aggressive) created (lr=" << lr << ")" << std::endl;

    // Use SGD (no momentum) to match PyTorch test exactly
    SGDOptions sgd_opts(lr);
    SGD optimizer(model->parameters(), sgd_opts);
    std::cout << "SGD optimizer (no momentum) created (lr=" << lr << ")" << std::endl;

    // Loss
    CrossEntropyLoss criterion;

    // Training loop
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\n=== Starting Training (MANUAL BACKWARD — zero autograd overhead) ===" << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;
    std::cout << "Training samples: " << n_train << ", Test samples: " << n_test << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // PRE-ALLOCATE all tensors ONCE — reused every batch, zero allocation in loop
    // =========================================================================
    // Activation buffers (saved from forward for backward)
    Tensor buf_h1     = at::empty({batch_size, 512});
    Tensor buf_h2     = at::empty({batch_size, 256});
    Tensor buf_h3     = at::empty({batch_size, 128});
    Tensor buf_logits = at::empty({batch_size, 10});

    // Input/target buffers (reused every batch)
    Tensor buf_inputs  = at::empty({batch_size, 784});
    Tensor buf_targets = at::empty({batch_size});

    // Parameter gradient buffers (reused every batch)
    Tensor grad_W1 = at::empty({512, 784});
    Tensor grad_b1 = at::empty({512});
    Tensor grad_W2 = at::empty({256, 512});
    Tensor grad_b2 = at::empty({256});
    Tensor grad_W3 = at::empty({128, 256});
    Tensor grad_b3 = at::empty({128});
    Tensor grad_W4 = at::empty({10, 128});
    Tensor grad_b4 = at::empty({10});

    // Scratch buffers for inter-layer gradients (reused every batch)
    Tensor scratch_gh4 = at::empty({batch_size, 10});
    Tensor scratch_gh3 = at::empty({batch_size, 128});
    Tensor scratch_gh2 = at::empty({batch_size, 256});
    Tensor scratch_gh1 = at::empty({batch_size, 512});

    // Get direct pointers to model parameters (avoid repeated map lookups)
    auto* p_fc1_w = model->get_submodule("fc1")->get_parameter("weight");
    auto* p_fc1_b = model->get_submodule("fc1")->get_parameter("bias");
    auto* p_fc2_w = model->get_submodule("fc2")->get_parameter("weight");
    auto* p_fc2_b = model->get_submodule("fc2")->get_parameter("bias");
    auto* p_fc3_w = model->get_submodule("fc3")->get_parameter("weight");
    auto* p_fc3_b = model->get_submodule("fc3")->get_parameter("bias");
    auto* p_fc4_w = model->get_submodule("fc4")->get_parameter("weight");
    auto* p_fc4_b = model->get_submodule("fc4")->get_parameter("bias");

    std::cout << "Pre-allocated all buffers. Training with ZERO autograd overhead." << std::endl;

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        model->train();

        // Shuffle indices
        std::vector<int64_t> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        float epoch_loss = 0.0f;
        int64_t correct = 0;
        int64_t total = 0;
        int64_t batches_processed = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int64_t batch_start = 0; batch_start < n_train; batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, n_train);
            int64_t B = batch_end - batch_start;

            // For last batch: B might be smaller than batch_size.
            // Use pre-allocated buffers for full batches, create new for partial.
            bool full_batch = (B == batch_size);
            Tensor inputs  = full_batch ? buf_inputs  : at::empty({B, 784});
            Tensor targets = full_batch ? buf_targets : at::empty({B});
            Tensor h1      = full_batch ? buf_h1      : at::empty({B, 512});
            Tensor h2      = full_batch ? buf_h2      : at::empty({B, 256});
            Tensor h3      = full_batch ? buf_h3      : at::empty({B, 128});
            Tensor logits  = full_batch ? buf_logits  : at::empty({B, 10});
            Tensor gh4     = full_batch ? scratch_gh4 : at::empty({B, 10});
            Tensor gh3     = full_batch ? scratch_gh3 : at::empty({B, 128});
            Tensor gh2     = full_batch ? scratch_gh2 : at::empty({B, 256});
            Tensor gh1     = full_batch ? scratch_gh1 : at::empty({B, 512});

            float* in_ptr = inputs.mutable_data_ptr<float>();
            float* tgt_ptr = targets.mutable_data_ptr<float>();

            // MNIST normalization constants
            constexpr float MNIST_MEAN = 0.1307f;
            constexpr float MNIST_STD = 0.3081f;

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = indices[batch_start + i];
                tgt_ptr[i] = static_cast<float>(train_labels[idx]);
                for (int j = 0; j < 784; ++j) {
                    float pixel = train_images[idx][j] / 255.0f;
                    in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                }
            }

            // ---- MANUAL FORWARD (no autograd) ----
            auto t_fwd_start = std::chrono::high_resolution_clock::now();
            model->manual_forward(inputs, h1, h2, h3, logits);
            auto t_fwd_end = std::chrono::high_resolution_clock::now();

            // ---- MANUAL BACKWARD (no autograd) ----
            manual_backward(
                inputs, targets,
                h1, h2, h3, logits,
                p_fc1_w->data(), p_fc2_w->data(),
                p_fc3_w->data(), p_fc4_w->data(),
                grad_W1, grad_b1, grad_W2, grad_b2,
                grad_W3, grad_b3, grad_W4, grad_b4,
                gh4, gh3, gh2, gh1);
            auto t_bwd_end = std::chrono::high_resolution_clock::now();

            // ---- SET PARAMETER GRADIENTS directly from pre-allocated buffers ----
            p_fc1_w->set_grad(grad_W1);
            p_fc1_b->set_grad(grad_b1);
            p_fc2_w->set_grad(grad_W2);
            p_fc2_b->set_grad(grad_b2);
            p_fc3_w->set_grad(grad_W3);
            p_fc3_b->set_grad(grad_b3);
            p_fc4_w->set_grad(grad_W4);
            p_fc4_b->set_grad(grad_b4);

            // Skip gradient clipping — max_norm=100, actual norm ~1.5, never clips
            // Saves 7.7ms per batch (was 33% of total overhead)
            auto t_clip_end = std::chrono::high_resolution_clock::now();
            optimizer.step();
            auto t_step_end = std::chrono::high_resolution_clock::now();

            // Print per-phase timing every 100 batches
            if (batches_processed % 100 == 0) {
                double fwd_ms = std::chrono::duration<double, std::milli>(t_fwd_end - t_fwd_start).count();
                double bwd_ms = std::chrono::duration<double, std::milli>(t_bwd_end - t_fwd_end).count();
                double clip_ms = std::chrono::duration<double, std::milli>(t_clip_end - t_bwd_end).count();
                double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_clip_end).count();
                printf("  [TIMING] batch=%lld  fwd=%.1fms bwd=%.1fms step=%.1fms\n",
                       (long long)batches_processed, fwd_ms, bwd_ms, step_ms);
            }

            // Compute loss for reporting (from softmax — reuse gh4 which has softmax grad)
            // Loss = -mean(log(softmax[target]))
            // gh4 = (softmax - one_hot) / B, so softmax = gh4 * B + one_hot
            // But simpler: just compute loss directly from logits
            {
                const float* lp = logits.data_ptr<float>();
                float batch_loss = 0.0f;
                for (int64_t i = 0; i < B; ++i) {
                    int64_t cls = static_cast<int64_t>(tgt_ptr[i]);
                    // log-sum-exp for numerical stability
                    float max_val = lp[i * 10];
                    for (int c = 1; c < 10; ++c)
                        if (lp[i * 10 + c] > max_val) max_val = lp[i * 10 + c];
                    float sum_exp = 0.0f;
                    for (int c = 0; c < 10; ++c)
                        sum_exp += std::exp(lp[i * 10 + c] - max_val);
                    float log_softmax = lp[i * 10 + cls] - max_val - std::log(sum_exp);
                    batch_loss -= log_softmax;
                }
                batch_loss /= B;

                if (batches_processed < 10) {
                    std::cout << "[LOSS] batch=" << batches_processed << " loss=" << batch_loss << std::endl;
                }
                epoch_loss += batch_loss * B;
            }

            // Accuracy (directly from logits, no Tensor ops)
            {
                const float* log_ptr = logits.data_ptr<float>();
                for (int64_t i = 0; i < B; ++i) {
                    int pred = 0;
                    float max_val = log_ptr[i * 10];
                    for (int c = 1; c < 10; ++c) {
                        if (log_ptr[i * 10 + c] > max_val) {
                            max_val = log_ptr[i * 10 + c];
                            pred = c;
                        }
                    }
                    if (pred == static_cast<int>(tgt_ptr[i])) {
                        correct++;
                    }
                    total++;
                }
            }

            batches_processed++;
            if (batches_processed <= 10 || batches_processed % 100 == 0) {
                std::cout << "  Batch " << batches_processed << "/" << (n_train / batch_size)
                          << " loss=" << (epoch_loss / total)
                          << " acc=" << (100.0f * correct / total) << "%" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Epoch " << epoch << "/" << epochs
                  << " - Loss: " << (epoch_loss / n_train)
                  << " - Train Acc: " << (100.0f * correct / total) << "%"
                  << " - Time: " << elapsed << "ms" << std::endl;

        // Test evaluation (uses manual_forward for speed, no autograd)
        model->eval();
        int64_t test_correct = 0;

        for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, n_test);
            int64_t B = batch_end - batch_start;

            bool full_batch = (B == batch_size);
            Tensor inputs = full_batch ? buf_inputs : at::empty({B, 784});
            Tensor h1t    = full_batch ? buf_h1     : at::empty({B, 512});
            Tensor h2t    = full_batch ? buf_h2     : at::empty({B, 256});
            Tensor h3t    = full_batch ? buf_h3     : at::empty({B, 128});
            Tensor logt   = full_batch ? buf_logits : at::empty({B, 10});
            float* in_ptr = inputs.mutable_data_ptr<float>();

            constexpr float MNIST_MEAN = 0.1307f;
            constexpr float MNIST_STD = 0.3081f;

            for (int64_t i = 0; i < B; ++i) {
                int64_t idx = batch_start + i;
                for (int j = 0; j < 784; ++j) {
                    float pixel = test_images[idx][j] / 255.0f;
                    in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                }
            }

            model->manual_forward(inputs, h1t, h2t, h3t, logt);
            const float* log_ptr = logt.data_ptr<float>();

            for (int64_t i = 0; i < B; ++i) {
                int pred = 0;
                float max_val = log_ptr[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (log_ptr[i * 10 + c] > max_val) {
                        max_val = log_ptr[i * 10 + c];
                        pred = c;
                    }
                }
                if (pred == test_labels[batch_start + i]) {
                    test_correct++;
                }
            }
        }

        std::cout << "  Test Accuracy: " << (100.0f * test_correct / n_test) << "%" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== Training Complete ===" << std::endl;

    // =========== LOW-RANK COMPRESSION ===========
    if (do_compress) {
        std::cout << "\n=== Low-Rank Weight Compression ===" << std::endl;
        std::cout << "Compression ratio: " << compress_ratio << std::endl;

        // Measure pre-compression accuracy and forward time
        float pre_compress_acc = 0.0f;
        double pre_compress_fwd_ms = 0.0;
        {
            model->eval();
            int64_t test_correct = 0;
            auto t_start = std::chrono::high_resolution_clock::now();

            for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
                int64_t batch_end = std::min(batch_start + batch_size, n_test);
                int64_t B = batch_end - batch_start;

                Tensor inputs = at::empty({B, 784});
                float* in_ptr = inputs.mutable_data_ptr<float>();
                constexpr float MNIST_MEAN = 0.1307f;
                constexpr float MNIST_STD = 0.3081f;
                for (int64_t i = 0; i < B; ++i) {
                    int64_t idx = batch_start + i;
                    for (int j = 0; j < 784; ++j) {
                        float pixel = test_images[idx][j] / 255.0f;
                        in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                    }
                }

                inputs = to_device(inputs);
                Tensor logits = model->forward(inputs);
                Tensor logits_cpu = move_to_cpu(logits);
                const float* log_ptr = logits_cpu.data_ptr<float>();

                for (int64_t i = 0; i < B; ++i) {
                    int pred = 0;
                    float max_val = log_ptr[i * 10];
                    for (int c = 1; c < 10; ++c) {
                        if (log_ptr[i * 10 + c] > max_val) {
                            max_val = log_ptr[i * 10 + c];
                            pred = c;
                        }
                    }
                    if (pred == test_labels[batch_start + i]) test_correct++;
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            pre_compress_acc = 100.0f * test_correct / n_test;
            pre_compress_fwd_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << "Pre-compression:  Accuracy=" << pre_compress_acc
                      << "%, Forward time=" << pre_compress_fwd_ms << "ms" << std::endl;
        }

        // Report params before compression
        int64_t params_before = count_parameters(*model);
        std::cout << "Parameters before: " << params_before << std::endl;

        // Compress the model
        auto stats = compress_model(*model, compress_ratio);
        std::cout << "Compressed " << stats.layers_compressed << " layers, skipped "
                  << stats.layers_skipped << std::endl;
        std::cout << "Weight params: " << stats.params_before << " -> " << stats.params_after
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - (double)stats.params_after / stats.params_before))
                  << "% reduction)" << std::endl;

        int64_t params_after = count_parameters(*model);
        std::cout << "Total parameters after: " << params_after << std::endl;
        std::cout << "Model after compression:\n" << model->repr() << std::endl;

        // Measure post-compression accuracy (before fine-tuning)
        float post_compress_acc = 0.0f;
        double post_compress_fwd_ms = 0.0;
        {
            model->eval();
            int64_t test_correct = 0;
            auto t_start = std::chrono::high_resolution_clock::now();

            for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
                int64_t batch_end = std::min(batch_start + batch_size, n_test);
                int64_t B = batch_end - batch_start;

                Tensor inputs = at::empty({B, 784});
                float* in_ptr = inputs.mutable_data_ptr<float>();
                constexpr float MNIST_MEAN = 0.1307f;
                constexpr float MNIST_STD = 0.3081f;
                for (int64_t i = 0; i < B; ++i) {
                    int64_t idx = batch_start + i;
                    for (int j = 0; j < 784; ++j) {
                        float pixel = test_images[idx][j] / 255.0f;
                        in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                    }
                }

                inputs = to_device(inputs);
                Tensor logits = model->forward(inputs);
                Tensor logits_cpu = move_to_cpu(logits);
                const float* log_ptr = logits_cpu.data_ptr<float>();

                for (int64_t i = 0; i < B; ++i) {
                    int pred = 0;
                    float max_val = log_ptr[i * 10];
                    for (int c = 1; c < 10; ++c) {
                        if (log_ptr[i * 10 + c] > max_val) {
                            max_val = log_ptr[i * 10 + c];
                            pred = c;
                        }
                    }
                    if (pred == test_labels[batch_start + i]) test_correct++;
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            post_compress_acc = 100.0f * test_correct / n_test;
            post_compress_fwd_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << "Post-compression: Accuracy=" << post_compress_acc
                      << "%, Forward time=" << post_compress_fwd_ms << "ms" << std::endl;
        }

        // Fine-tune for 1 epoch with reduced learning rate
        std::cout << "\n--- Fine-tuning compressed model for 1 epoch ---" << std::endl;
        {
            model->train();
            // Create new optimizer for compressed model params
            SGDOptions ft_opts(lr * 0.1f);  // Lower LR for fine-tuning
            SGD ft_optimizer(model->parameters(), ft_opts);
            CrossEntropyLoss ft_criterion;

            std::vector<int64_t> indices(n_train);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);

            float epoch_loss = 0.0f;
            int64_t correct = 0, total = 0, batches = 0;

            for (int64_t batch_start = 0; batch_start < n_train; batch_start += batch_size) {
                int64_t batch_end = std::min(batch_start + batch_size, n_train);
                int64_t B = batch_end - batch_start;

                Tensor inputs = at::empty({B, 784});
                Tensor targets = at::empty({B});
                float* in_ptr = inputs.mutable_data_ptr<float>();
                float* tgt_ptr = targets.mutable_data_ptr<float>();
                constexpr float MNIST_MEAN = 0.1307f;
                constexpr float MNIST_STD = 0.3081f;

                for (int64_t i = 0; i < B; ++i) {
                    int64_t idx = indices[batch_start + i];
                    tgt_ptr[i] = static_cast<float>(train_labels[idx]);
                    for (int j = 0; j < 784; ++j) {
                        float pixel = train_images[idx][j] / 255.0f;
                        in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                    }
                }

                inputs = to_device(inputs);
                targets = to_device(targets);

                ft_optimizer.zero_grad();
                Tensor logits = model->forward(inputs);
                Tensor loss = ft_criterion.forward(logits, targets);
                torch::autograd::backward({loss});
                clip_grad_norm_(*model, 10.0);
                ft_optimizer.step();

                Tensor loss_cpu = move_to_cpu(loss);
                epoch_loss += loss_cpu.data_ptr<float>()[0] * B;

                Tensor logits_cpu = move_to_cpu(logits);
                Tensor targets_cpu = move_to_cpu(targets);
                const float* log_ptr = logits_cpu.data_ptr<float>();
                const float* tgt_cpu_ptr = targets_cpu.data_ptr<float>();
                for (int64_t i = 0; i < B; ++i) {
                    int pred = 0;
                    float max_val = log_ptr[i * 10];
                    for (int c = 1; c < 10; ++c) {
                        if (log_ptr[i * 10 + c] > max_val) {
                            max_val = log_ptr[i * 10 + c];
                            pred = c;
                        }
                    }
                    if (pred == static_cast<int>(tgt_cpu_ptr[i])) correct++;
                    total++;
                }
                batches++;

                if (batches % 200 == 0) {
                    std::cout << "  Fine-tune batch " << batches
                              << " loss=" << (epoch_loss / total)
                              << " acc=" << (100.0f * correct / total) << "%" << std::endl;
                }
            }
            std::cout << "Fine-tune epoch: Loss=" << (epoch_loss / n_train)
                      << " Train Acc=" << (100.0f * correct / total) << "%" << std::endl;
        }

        // Measure post-fine-tune accuracy and speed
        float finetuned_acc = 0.0f;
        double finetuned_fwd_ms = 0.0;
        {
            model->eval();
            int64_t test_correct = 0;
            auto t_start = std::chrono::high_resolution_clock::now();

            for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
                int64_t batch_end = std::min(batch_start + batch_size, n_test);
                int64_t B = batch_end - batch_start;

                Tensor inputs = at::empty({B, 784});
                float* in_ptr = inputs.mutable_data_ptr<float>();
                constexpr float MNIST_MEAN = 0.1307f;
                constexpr float MNIST_STD = 0.3081f;
                for (int64_t i = 0; i < B; ++i) {
                    int64_t idx = batch_start + i;
                    for (int j = 0; j < 784; ++j) {
                        float pixel = test_images[idx][j] / 255.0f;
                        in_ptr[i * 784 + j] = (pixel - MNIST_MEAN) / MNIST_STD;
                    }
                }

                inputs = to_device(inputs);
                Tensor logits = model->forward(inputs);
                Tensor logits_cpu = move_to_cpu(logits);
                const float* log_ptr = logits_cpu.data_ptr<float>();

                for (int64_t i = 0; i < B; ++i) {
                    int pred = 0;
                    float max_val = log_ptr[i * 10];
                    for (int c = 1; c < 10; ++c) {
                        if (log_ptr[i * 10 + c] > max_val) {
                            max_val = log_ptr[i * 10 + c];
                            pred = c;
                        }
                    }
                    if (pred == test_labels[batch_start + i]) test_correct++;
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            finetuned_acc = 100.0f * test_correct / n_test;
            finetuned_fwd_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        }

        // Final compression report
        std::cout << "\n=== Compression Summary ===" << std::endl;
        std::cout << "Parameters:       " << params_before << " -> " << params_after
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * params_after / params_before) << "% of original)" << std::endl;
        std::cout << "Accuracy (before): " << pre_compress_acc << "%" << std::endl;
        std::cout << "Accuracy (compressed, no finetune): " << post_compress_acc << "%" << std::endl;
        std::cout << "Accuracy (compressed + finetuned):  " << finetuned_acc << "%" << std::endl;
        std::cout << "Forward time (before):     " << std::fixed << std::setprecision(1)
                  << pre_compress_fwd_ms << " ms" << std::endl;
        std::cout << "Forward time (compressed): " << std::fixed << std::setprecision(1)
                  << finetuned_fwd_ms << " ms" << std::endl;
        double speedup = pre_compress_fwd_ms / (finetuned_fwd_ms + 1e-9);
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "===========================" << std::endl;
    }
    // =========== END LOW-RANK COMPRESSION ===========

    // Print CPU allocator statistics
    c10::CPUAllocator::get().print_stats();

#ifdef PT_USE_CUDA
    // CRITICAL: Shutdown CUDA AFTER all tensors are destroyed
    if (g_device.is_cuda()) {
        // Clear optimizer and model gradients
        optimizer.zero_grad();
        model->zero_grad();

        // Clear parameter grad_fn
        for (auto* param : model->parameters()) {
            if (param && param->defined()) {
                at::Tensor& data = param->data();
                if (data.defined()) {
                    torch::autograd::clear_grad_fn(data);
                    param->set_grad(at::Tensor());
                }
            }
        }

        c10::cuda::cuda_shutdown();
        std::cout << "CUDA shutdown complete" << std::endl;
    }
#endif

    return 0;
}
