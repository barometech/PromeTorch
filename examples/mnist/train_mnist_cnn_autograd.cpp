// ============================================================================
// MNIST Training Example - CNN with full autograd backward
// ============================================================================
// LeNet-style CNN that tests Conv2d, MaxPool2d, ReLU backward through autograd.
// Gradients flow: CrossEntropy -> Linear -> ReLU -> Linear -> Flatten ->
//                 MaxPool2d -> ReLU -> Conv2d -> MaxPool2d -> ReLU -> Conv2d
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/functions/ShapeBackward.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "c10/core/Allocator.h"
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

// ============================================================================
// MNIST Data Loading (IDX format) — same as train_mnist_mlp.cpp
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
// Flatten with autograd support
// ============================================================================
// ShapeOps::flatten / view don't propagate grad_fn, so we wire it manually.

at::Tensor flatten_autograd(const at::Tensor& input, int64_t start_dim = 1) {
    // Flatten from start_dim to end
    std::vector<int64_t> orig_sizes = input.sizes().vec();
    at::Tensor result = input.flatten(start_dim, -1);

    if (torch::autograd::GradMode::is_enabled() && input.requires_grad()) {
        // FlattenBackward reshapes grad back to original input shape
        auto grad_fn = std::make_shared<torch::autograd::FlattenBackward>(orig_sizes);
        grad_fn->add_input_metadata(input);
        torch::autograd::set_grad_fn(result, grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

// ============================================================================
// LeNet-style CNN Model
// ============================================================================
// Conv2d(1,16,3,pad=1) -> ReLU -> MaxPool2d(2)   [28->14]
// Conv2d(16,32,3,pad=1) -> ReLU -> MaxPool2d(2)  [14->7]
// Flatten -> Linear(32*7*7, 128) -> ReLU -> Linear(128, 10)

class MNISTConvNet : public Module {
public:
    MNISTConvNet() : Module("MNISTConvNet") {
        conv1 = std::make_shared<Conv2d>(1, 16, 3, /*stride=*/1, /*padding=*/1);
        conv2 = std::make_shared<Conv2d>(16, 32, 3, /*stride=*/1, /*padding=*/1);
        pool = std::make_shared<MaxPool2d>(2);
        fc1 = std::make_shared<Linear>(32 * 7 * 7, 128);
        fc2 = std::make_shared<Linear>(128, 10);

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("pool", pool);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    at::Tensor forward(const at::Tensor& x) override {
        // x: [B, 1, 28, 28]
        at::Tensor h = conv1->forward(x);                   // [B, 16, 28, 28]
        h = torch::autograd::relu_autograd(h);               // ReLU
        h = pool->forward(h);                                // [B, 16, 14, 14]

        h = conv2->forward(h);                               // [B, 32, 14, 14]
        h = torch::autograd::relu_autograd(h);               // ReLU
        h = pool->forward(h);                                // [B, 32, 7, 7]

        h = flatten_autograd(h, 1);                          // [B, 32*7*7]

        h = fc1->forward(h);                                 // [B, 128]
        h = torch::autograd::relu_autograd(h);               // ReLU
        h = fc2->forward(h);                                 // [B, 10]
        return h;
    }

private:
    std::shared_ptr<Conv2d> conv1, conv2;
    std::shared_ptr<MaxPool2d> pool;
    std::shared_ptr<Linear> fc1, fc2;
};

// ============================================================================
// Gradient norm helper
// ============================================================================

float grad_norm(const Parameter* p) {
    if (!p->grad().defined()) return 0.0f;
    const float* g = p->grad().data_ptr<float>();
    int64_t n = p->grad().numel();
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum_sq += g[i] * g[i];
    return std::sqrt(sum_sq);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    int64_t batch_size = 64;
    int64_t epochs = 1;
    float lr = 0.01f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--batch_size" && i + 1 < argc) batch_size = std::stoll(argv[++i]);
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoll(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
    }

    std::cout << "=== MNIST CNN Autograd Test ===" << std::endl;
    std::cout << "Testing: Conv2d, MaxPool2d, ReLU backward through autograd" << std::endl;
    std::cout << "Device: CPU | Batch: " << batch_size
              << " | Epochs: " << epochs << " | LR: " << lr << std::endl;

    // Load data
    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    auto train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    auto test_images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    auto test_labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    if (train_images.empty()) {
        std::cerr << "Failed to load MNIST data from: " << data_dir << std::endl;
        return 1;
    }

    int64_t n_train = train_images.size();
    int64_t n_test = test_images.size();

    // Create model and optimizer
    auto model = std::make_shared<MNISTConvNet>();
    auto params = model->parameters();
    SGDOptions sgd_opts(lr);
    SGD optimizer(params, sgd_opts);
    CrossEntropyLoss criterion;

    std::cout << "\nModel parameters: " << count_parameters(*model) << std::endl;

    // Print model structure
    std::cout << "Architecture:" << std::endl;
    for (auto& [name, mod] : model->named_children()) {
        std::cout << "  " << name << ": " << mod->name() << std::endl;
    }

    // Shuffle indices
    std::vector<int64_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    // =========================================================================
    // Training loop
    // =========================================================================
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        int64_t correct = 0;
        int64_t total = 0;
        int64_t n_batches = n_train / batch_size;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int64_t batch = 0; batch < n_batches; ++batch) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            // Build batch: [B, 1, 28, 28] images and [B] labels
            at::Tensor images = at::empty({batch_size, 1, 28, 28});
            at::Tensor labels = at::empty({batch_size});
            float* img_data = images.mutable_data_ptr<float>();
            float* lbl_data = labels.mutable_data_ptr<float>();

            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t idx = indices[batch * batch_size + i];
                lbl_data[i] = static_cast<float>(train_labels[idx]);
                // Normalize to [0,1] and reshape to [1, 28, 28]
                for (int j = 0; j < 784; ++j) {
                    img_data[i * 784 + j] = train_images[idx][j] / 255.0f;
                }
            }
            images.set_requires_grad(true);

            // Forward
            auto fwd_start = std::chrono::high_resolution_clock::now();
            at::Tensor logits = model->forward(images);
            at::Tensor loss = criterion.forward(logits, labels);
            auto fwd_end = std::chrono::high_resolution_clock::now();

            float loss_val = loss.data_ptr<float>()[0];
            epoch_loss += loss_val;

            // Compute accuracy
            const float* logits_data = logits.data_ptr<float>();
            for (int64_t i = 0; i < batch_size; ++i) {
                int pred = 0;
                float max_val = logits_data[i * 10];
                for (int c = 1; c < 10; ++c) {
                    if (logits_data[i * 10 + c] > max_val) {
                        max_val = logits_data[i * 10 + c];
                        pred = c;
                    }
                }
                if (pred == static_cast<int>(lbl_data[i])) ++correct;
                ++total;
            }

            // Backward
            model->zero_grad();
            auto bwd_start = std::chrono::high_resolution_clock::now();
            torch::autograd::backward({loss});
            auto bwd_end = std::chrono::high_resolution_clock::now();

            // Optimizer step
            auto step_start = std::chrono::high_resolution_clock::now();
            optimizer.step();
            auto step_end = std::chrono::high_resolution_clock::now();

            // Print progress every 100 batches + gradient norms
            if (batch % 100 == 0 || batch == n_batches - 1) {
                double fwd_ms = std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count();
                double bwd_ms = std::chrono::duration<double, std::milli>(bwd_end - bwd_start).count();
                double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Epoch " << epoch + 1 << " [" << batch + 1 << "/" << n_batches << "]"
                          << "  loss=" << loss_val
                          << "  acc=" << (100.0f * correct / total) << "%"
                          << "  fwd=" << fwd_ms << "ms"
                          << "  bwd=" << bwd_ms << "ms"
                          << "  step=" << step_ms << "ms";

                // Print gradient norms for each layer
                std::cout << "\n  Gradient norms:";
                for (auto& [name, mod] : model->named_children()) {
                    auto* w = mod->get_parameter("weight");
                    if (w) {
                        float gn = grad_norm(w);
                        std::cout << "  " << name << ".w=" << gn;
                    }
                    auto* b = mod->get_parameter("bias");
                    if (b) {
                        float gn = grad_norm(b);
                        std::cout << "  " << name << ".b=" << gn;
                    }
                }
                std::cout << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

        float avg_loss = epoch_loss / n_batches;
        float train_acc = 100.0f * correct / total;
        std::cout << "\nEpoch " << epoch + 1 << " summary:"
                  << "  avg_loss=" << avg_loss
                  << "  train_acc=" << train_acc << "%"
                  << "  time=" << epoch_ms / 1000.0 << "s" << std::endl;

        // =====================================================================
        // Evaluate on test set
        // =====================================================================
        {
            torch::autograd::NoGradGuard no_grad;
            int64_t test_correct = 0;
            int64_t test_batches = n_test / batch_size;

            for (int64_t batch = 0; batch < test_batches; ++batch) {
                at::Tensor images = at::empty({batch_size, 1, 28, 28});
                at::Tensor labels_t = at::empty({batch_size});
                float* img_data = images.mutable_data_ptr<float>();
                float* lbl_data = labels_t.mutable_data_ptr<float>();

                for (int64_t i = 0; i < batch_size; ++i) {
                    int64_t idx = batch * batch_size + i;
                    lbl_data[i] = static_cast<float>(test_labels[idx]);
                    for (int j = 0; j < 784; ++j) {
                        img_data[i * 784 + j] = test_images[idx][j] / 255.0f;
                    }
                }

                at::Tensor logits = model->forward(images);
                const float* logits_data = logits.data_ptr<float>();
                for (int64_t i = 0; i < batch_size; ++i) {
                    int pred = 0;
                    float max_val = logits_data[i * 10];
                    for (int c = 1; c < 10; ++c) {
                        if (logits_data[i * 10 + c] > max_val) {
                            max_val = logits_data[i * 10 + c];
                            pred = c;
                        }
                    }
                    if (pred == static_cast<int>(lbl_data[i])) ++test_correct;
                }
            }

            float test_acc = 100.0f * test_correct / (test_batches * batch_size);
            std::cout << "Test accuracy: " << test_acc << "%" << std::endl;

            // Verify gradient flow
            std::cout << "\n=== Gradient Flow Verification ===" << std::endl;
            bool all_grads_ok = true;
            for (auto& [name, mod] : model->named_children()) {
                auto* w = mod->get_parameter("weight");
                if (w) {
                    float gn = grad_norm(w);
                    bool ok = gn > 1e-8f;
                    std::cout << "  " << name << ".weight grad_norm=" << gn
                              << (ok ? " OK" : " ZERO GRADIENT!") << std::endl;
                    if (!ok) all_grads_ok = false;
                }
            }
            if (all_grads_ok) {
                std::cout << "*** ALL GRADIENTS FLOW CORRECTLY ***" << std::endl;
            } else {
                std::cout << "*** GRADIENT FLOW BROKEN! ***" << std::endl;
                return 1;
            }

            // Final verdict
            if (test_acc > 85.0f) {
                std::cout << "\n*** TEST PASSED: CNN accuracy " << test_acc
                          << "% > 85% threshold ***" << std::endl;
            } else {
                std::cout << "\n*** TEST FAILED: CNN accuracy " << test_acc
                          << "% < 85% threshold ***" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
