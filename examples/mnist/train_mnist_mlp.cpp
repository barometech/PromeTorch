// ============================================================================
// MNIST Training Example - MLP (GPU-Compatible)
// ============================================================================
// Simple MLP model using only Linear layers (works on CUDA)
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
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
        fc1 = std::make_shared<Linear>(784, 512);
        fc2 = std::make_shared<Linear>(512, 256);
        fc3 = std::make_shared<Linear>(256, 128);
        fc4 = std::make_shared<Linear>(128, 10);

        relu1 = std::make_shared<ReLU>();
        relu2 = std::make_shared<ReLU>();
        relu3 = std::make_shared<ReLU>();

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc4", fc4);
    }

    Tensor forward(const Tensor& x) override {
        // x: [B, 784] -> fc1 -> relu -> fc2 -> relu -> fc3 -> relu -> fc4 -> [B, 10]
        Tensor h = fc1->forward(x);
        h = relu1->forward(h);
        h = fc2->forward(h);
        h = relu2->forward(h);
        h = fc3->forward(h);
        h = relu3->forward(h);
        h = fc4->forward(h);
        return h;
    }

private:
    std::shared_ptr<Linear> fc1, fc2, fc3, fc4;
    std::shared_ptr<ReLU> relu1, relu2, relu3;
};

// ============================================================================
// Training
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    std::string device_str = "cpu";
    int64_t batch_size = 64;
    int64_t epochs = 5;
    float lr = 0.001f;  // Standard learning rate for MLP

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

    // =========== ADAM vs SGD TEST ===========
    // Compare Adam and SGD on same problem
    std::cout << "\n=== Adam vs SGD Comparison Test ===\n";
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
        std::cout << std::endl;
    }
    // =========== END ADAM vs SGD TEST ===========

    // Create model
    auto model = std::make_shared<MNISTMLP>();
    std::cout << "MLP Model created (784->512->256->128->10)" << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        model->to(g_device);
        std::cout << "Model moved to CUDA" << std::endl;
    }
#endif

    // Optimizer - Simple SGD without momentum for clean baseline
    SGDOptions opts(lr);
    opts.momentum_(0.0);  // NO momentum to avoid accumulation issues
    SGD optimizer(model->parameters(), opts);
    std::cout << "SGD optimizer (NO momentum) created (lr=" << lr << ")" << std::endl;

    // Loss
    CrossEntropyLoss criterion;

    // Training loop
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;
    std::cout << "Training samples: " << n_train << ", Test samples: " << n_test << std::endl;
    std::cout << std::endl;

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

            // Create batch tensors - FLATTENED for MLP
            Tensor inputs = at::empty({B, 784});  // Flattened 28x28
            Tensor targets = at::empty({B});
            float* in_ptr = inputs.mutable_data_ptr<float>();
            float* tgt_ptr = targets.mutable_data_ptr<float>();

            // MNIST normalization constants (from PyTorch official example)
            // mean = 0.1307, std = 0.3081
            // normalized = (pixel/255 - mean) / std
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

            // Forward
            optimizer.zero_grad();
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);

            // Backward
            torch::autograd::backward({loss});

            // DEBUG: Check gradient and weight scale EVERY batch for first 10, then every 100
            if (batches_processed < 10 || batches_processed % 100 == 0) {
                auto params = model->parameters();
                std::cout << "[DBG] batch=" << batches_processed;

                // Check all layer gradients and weights
                const char* names[] = {"fc1.w", "fc1.b", "fc2.w", "fc2.b", "fc3.w", "fc3.b", "fc4.w", "fc4.b"};
                std::cout << "\n";
                for (size_t i = 0; i < params.size() && i < 8; ++i) {
                    Tensor g = params[i]->grad();
                    Tensor w = params[i]->data();

                    if (g.defined()) {
                        Tensor g_cpu = move_to_cpu(g);
                        float g_norm = 0;
                        const float* gd = g_cpu.data_ptr<float>();
                        for (int64_t j = 0; j < g_cpu.numel(); ++j) {
                            g_norm += gd[j] * gd[j];
                        }
                        g_norm = std::sqrt(g_norm);

                        Tensor w_cpu = move_to_cpu(w);
                        float w_norm = 0;
                        const float* wd = w_cpu.data_ptr<float>();
                        for (int64_t j = 0; j < w_cpu.numel(); ++j) {
                            w_norm += wd[j] * wd[j];
                        }
                        w_norm = std::sqrt(w_norm);

                        std::cout << "  " << names[i] << "=(g:" << g_norm << ",w:" << w_norm << ")\n";
                    } else {
                        std::cout << "  " << names[i] << "=(NO GRADIENT!)\n";
                    }
                }
            }

            // Gradient clipping (disabled to see raw behavior)
            double grad_norm = clip_grad_norm_(*model, 100.0);  // Very loose clipping
            if (batches_processed < 10 || batches_processed % 100 == 0) {
                std::cout << "[CLIP] total_grad_norm_before_clip=" << grad_norm << std::endl;

                // Verify clipping worked
                auto params = model->parameters();
                float total_clipped = 0;
                for (auto* p : params) {
                    if (p->grad().defined()) {
                        Tensor g = move_to_cpu(p->grad());
                        const float* gd = g.data_ptr<float>();
                        for (int64_t j = 0; j < g.numel(); ++j) {
                            total_clipped += gd[j] * gd[j];
                        }
                    }
                }
                std::cout << "[AFTER_CLIP] total_grad_norm=" << std::sqrt(total_clipped) << std::endl;
            }

            // Optimizer step
            optimizer.step();

            // Stats
            Tensor loss_cpu = move_to_cpu(loss);
            float batch_loss = loss_cpu.data_ptr<float>()[0];
            if (batches_processed < 10) {
                std::cout << "[LOSS] batch=" << batches_processed << " loss=" << batch_loss << std::endl;

                // Print weight sum after step
                auto params = model->parameters();
                if (!params.empty()) {
                    Tensor w_cpu = move_to_cpu(params[0]->data());
                    float w_sum = 0;
                    const float* wd = w_cpu.data_ptr<float>();
                    for (int64_t j = 0; j < w_cpu.numel(); ++j) {
                        w_sum += wd[j];
                    }
                    std::cout << "[WEIGHT_AFTER] weight_sum=" << w_sum << std::endl;
                }
            }
            epoch_loss += batch_loss * B;

            // Accuracy
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
                if (pred == static_cast<int>(tgt_cpu_ptr[i])) {
                    correct++;
                }
                total++;
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

        // Test evaluation
        model->eval();
        int64_t test_correct = 0;

        for (int64_t batch_start = 0; batch_start < n_test; batch_start += batch_size) {
            int64_t batch_end = std::min(batch_start + batch_size, n_test);
            int64_t B = batch_end - batch_start;

            Tensor inputs = at::empty({B, 784});  // Flattened
            float* in_ptr = inputs.mutable_data_ptr<float>();

            // Same MNIST normalization for test data
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
                if (pred == test_labels[batch_start + i]) {
                    test_correct++;
                }
            }
        }

        std::cout << "  Test Accuracy: " << (100.0f * test_correct / n_test) << "%" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== Training Complete ===" << std::endl;

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
