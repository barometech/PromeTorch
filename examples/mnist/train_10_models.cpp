// ============================================================================
// 10 Models Training — Simple to Complex (CPU + GPU)
// ============================================================================
// Uses ONLY PromeTorch framework. No PyTorch, no external ML libs.
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/serialization.h"
#include "torch/nn/modules/rnn.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#include "c10/cuda/CUDAAllocator.h"
#endif
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>

#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#endif

using namespace torch;
using namespace torch::nn;
using namespace torch::optim;

static std::mt19937 g_rng(42);

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
// MNIST Data Loading
// ============================================================================

std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    int32_t magic, num, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);
    num = bswap32(num); rows = bswap32(rows); cols = bswap32(cols);
    std::vector<std::vector<uint8_t>> images(num);
    for (int i = 0; i < num; ++i) {
        images[i].resize(rows * cols);
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }
    return images;
}

std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    int32_t magic, num;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    num = bswap32(num);
    std::vector<uint8_t> labels(num);
    file.read(reinterpret_cast<char*>(labels.data()), num);
    return labels;
}

// ============================================================================
// Helpers
// ============================================================================

Tensor make_batch(const std::vector<std::vector<uint8_t>>& images,
                  const std::vector<uint8_t>& labels,
                  const std::vector<int64_t>& indices,
                  int64_t start, int64_t end) {
    // Returns (inputs [B, 784], targets [B])
    // Uses static to avoid repeated allocation
    int64_t B = end - start;
    Tensor inputs = at::empty({B, 784});
    Tensor targets = at::empty({B});
    float* in_ptr = inputs.mutable_data_ptr<float>();
    float* tgt_ptr = targets.mutable_data_ptr<float>();
    for (int64_t i = 0; i < B; ++i) {
        int64_t idx = indices[start + i];
        for (int64_t j = 0; j < 784; ++j) {
            in_ptr[i * 784 + j] = (images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
        }
        tgt_ptr[i] = static_cast<float>(labels[idx]);
    }
    return inputs; // targets accessed via second call
}

float evaluate_mnist(Module& model, const std::vector<std::vector<uint8_t>>& images,
                     const std::vector<uint8_t>& labels, int64_t max_samples = 0) {
    int64_t n = images.size();
    if (max_samples > 0 && max_samples < n) n = max_samples;
    int64_t correct = 0;
    int64_t bs = 256;

    torch::autograd::NoGradGuard no_grad;
    for (int64_t i = 0; i < n; i += bs) {
        int64_t B = std::min(bs, n - i);
        Tensor inputs = at::empty({B, 784});
        float* in_ptr = inputs.mutable_data_ptr<float>();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t j = 0; j < 784; ++j) {
                in_ptr[b * 784 + j] = (images[i + b][j] / 255.0f - 0.1307f) / 0.3081f;
            }
        }
        Tensor logits = model.forward(to_device(inputs));
        // argmax returns Long tensor — read as int64_t
        Tensor preds = move_to_cpu(logits).argmax(1);
        const int64_t* pred_data = preds.data_ptr<int64_t>();
        for (int64_t b = 0; b < B; ++b) {
            if (pred_data[b] == labels[i + b]) ++correct;
        }
    }
    return 100.0f * correct / n;
}

void print_header(int num, const std::string& name, const std::string& desc) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  MODEL " << num << ": " << name << std::endl;
    std::cout << "  " << desc << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_result(const std::string& metric, float value) {
    std::cout << "  RESULT: " << metric << " = " << value << std::endl;
}

// ============================================================================
// MODEL 1: Linear Regression (y = Wx + b on synthetic data)
// ============================================================================

void train_model_1() {
    print_header(1, "Linear Regression", "y = 3x₁ + 2x₂ - 1 + noise (MSE loss)");

    auto model = std::make_shared<Linear>(2, 1);
    std::cout << "  [DEBUG] Model created" << std::endl; std::cout.flush();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        std::cout << "  [DEBUG] Moving model to CUDA..." << std::endl; std::cout.flush();
        model->to(g_device);
        std::cout << "  [DEBUG] Model on CUDA" << std::endl; std::cout.flush();
    }
#endif
    std::cout << "  [DEBUG] Creating optimizer..." << std::endl; std::cout.flush();
    SGD optimizer(model->parameters(), SGDOptions(0.01));
    std::cout << "  [DEBUG] Optimizer created" << std::endl; std::cout.flush();

    // Quick CUDA mm test before training
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        try {
            std::cout << "  [DEBUG] Testing basic CUDA mm..." << std::endl; std::cout.flush();
            Tensor test_a = at::ones({2, 3});
            Tensor test_b = at::ones({3, 2});
            test_a = to_device(test_a);
            test_b = to_device(test_b);
            std::cout << "  [DEBUG] Tensors on CUDA, calling mm..." << std::endl; std::cout.flush();
            Tensor test_c = at::cuda_ops::mm(test_a, test_b);
            std::cout << "  [DEBUG] mm done, moving to CPU..." << std::endl; std::cout.flush();
            Tensor test_cpu = move_to_cpu(test_c);
            std::cout << "  [DEBUG] CUDA mm test: result[0,0]=" << test_cpu.data_ptr<float>()[0] << " (expect 3)" << std::endl; std::cout.flush();

            // Test mm with transpose (like Linear forward does: input @ weight.t())
            // weight [4,3], weight_t [3,4], input [2,3] → mm([2,3], [3,4]) = [2,4]
            std::cout << "  [DEBUG] Testing CUDA mm with transpose..." << std::endl; std::cout.flush();
            Tensor test_w = at::ones({4, 3});  // weight-like
            test_w = to_device(test_w);
            Tensor test_wt = at::native::t(test_w);  // [3, 4], non-contiguous
            std::cout << "  [DEBUG] Transposed: shape=[" << test_wt.size(0) << "," << test_wt.size(1)
                      << "] contiguous=" << test_wt.is_contiguous()
                      << " stride=[" << test_wt.stride(0) << "," << test_wt.stride(1) << "]" << std::endl; std::cout.flush();
            std::cout << "  [DEBUG] mm dims: [2,3] x [" << test_wt.size(0) << "," << test_wt.size(1) << "]" << std::endl; std::cout.flush();
            Tensor test_c2 = at::cuda_ops::mm(test_a, test_wt);
            std::cout << "  [DEBUG] mm with transpose done!" << std::endl; std::cout.flush();
            Tensor test_cpu2 = move_to_cpu(test_c2);
            std::cout << "  [DEBUG] Result: [" << test_c2.size(0) << "," << test_c2.size(1) << "] val=" << test_cpu2.data_ptr<float>()[0] << " (expect 3)" << std::endl; std::cout.flush();
        } catch (const std::exception& e) {
            std::cerr << "  [ERROR in CUDA mm test] " << e.what() << std::endl; std::cerr.flush();
        }
    }
#endif

    // Generate synthetic data: y = 3*x1 + 2*x2 - 1
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int64_t N = 500;
    float total_loss = 0;

    for (int epoch = 1; epoch <= 100; ++epoch) {
        if (epoch == 1) { std::cout << "  [DEBUG] Creating tensors..." << std::endl; std::cout.flush(); }
        Tensor X = at::empty({N, 2});
        Tensor Y = at::empty({N, 1});
        float* xp = X.mutable_data_ptr<float>();
        float* yp = Y.mutable_data_ptr<float>();

        for (int64_t i = 0; i < N; ++i) {
            float x1 = dist(g_rng), x2 = dist(g_rng);
            xp[i * 2] = x1;
            xp[i * 2 + 1] = x2;
            yp[i] = 3.0f * x1 + 2.0f * x2 - 1.0f + 0.1f * dist(g_rng);
        }
        if (epoch == 1) { std::cout << "  [DEBUG] to_device X..." << std::endl; std::cout.flush(); }
        X = to_device(X);
        if (epoch == 1) { std::cout << "  [DEBUG] to_device Y..." << std::endl; std::cout.flush(); }
        Y = to_device(Y);

        if (epoch == 1) {
            std::cout << "  [DEBUG] zero_grad..." << std::endl; std::cout.flush();
        }
        model->zero_grad();
        if (epoch == 1) {
            std::cout << "  [DEBUG] zero_grad done, calling forward..." << std::endl; std::cout.flush();
#ifdef PT_USE_CUDA
            if (g_device.is_cuda()) {
                cudaError_t err = cudaGetLastError();
                std::cout << "  [DEBUG] CUDA status before forward: " << cudaGetErrorString(err) << std::endl; std::cout.flush();
                // Verify tensors
                std::cout << "  [DEBUG] X: device=" << (X.is_cuda()?"CUDA":"CPU") << " shape=[" << X.size(0) << "," << X.size(1) << "] contiguous=" << X.is_contiguous() << std::endl; std::cout.flush();
                auto* wp = model->get_parameter("weight");
                Tensor wd = wp->data();
                std::cout << "  [DEBUG] weight: device=" << (wd.is_cuda()?"CUDA":"CPU") << " shape=[" << wd.size(0) << "," << wd.size(1) << "] contiguous=" << wd.is_contiguous() << std::endl; std::cout.flush();
            }
#endif
        }
        Tensor pred = model->forward(X);  // [N, 1]
        if (epoch == 1) { std::cout << "  [DEBUG] forward done! pred shape=[" << pred.size(0) << "," << pred.size(1) << "] cuda=" << pred.is_cuda() << std::endl; std::cout.flush(); }
        Tensor diff = torch::autograd::sub_autograd(pred, Y);
        if (epoch == 1) { std::cout << "  [DEBUG] sub done" << std::endl; std::cout.flush(); }
        Tensor sq = torch::autograd::mul_autograd(diff, diff);
        if (epoch == 1) { std::cout << "  [DEBUG] mul done" << std::endl; std::cout.flush(); }
        Tensor loss = torch::autograd::mean_autograd(sq);
        if (epoch == 1) { std::cout << "  [DEBUG] mean done" << std::endl; std::cout.flush(); }

        total_loss = move_to_cpu(loss).data_ptr<float>()[0];
        if (epoch == 1) { std::cout << "  [DEBUG] loss=" << total_loss << std::endl; std::cout.flush(); }
        torch::autograd::backward({loss});
        if (epoch == 1) { std::cout << "  [DEBUG] backward done" << std::endl; std::cout.flush(); }
        optimizer.step();
        if (epoch == 1) { std::cout << "  [DEBUG] step done" << std::endl; std::cout.flush(); }

        if (epoch % 20 == 0) {
            std::cout << "  Epoch " << epoch << " | MSE = " << total_loss << std::endl;
        }
    }

    // Check learned weights
    auto* w = model->get_parameter("weight");
    auto* b = model->get_parameter("bias");
    Tensor w_cpu = move_to_cpu(w->data());
    Tensor b_cpu = move_to_cpu(b->data());
    std::cout << "  Learned: w1=" << w_cpu.data_ptr<float>()[0]
              << ", w2=" << w_cpu.data_ptr<float>()[1]
              << ", b=" << b_cpu.data_ptr<float>()[0]
              << " (true: 3, 2, -1)" << std::endl;
    print_result("Final MSE", total_loss);
}

// ============================================================================
// MODEL 2: Logistic Regression (binary classification)
// ============================================================================

void train_model_2() {
    print_header(2, "Logistic Regression", "Binary classification: x₁²+x₂² < 1 (BCE loss)");

    auto model = std::make_shared<Linear>(2, 1);
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) model->to(g_device);
#endif
    SGD optimizer(model->parameters(), SGDOptions(0.1));

    std::normal_distribution<float> dist(0.0f, 1.5f);
    int64_t N = 200;
    float final_acc = 0;

    for (int epoch = 1; epoch <= 50; ++epoch) {
        Tensor X = at::empty({N, 2});
        Tensor Y = at::empty({N, 1});
        float* xp = X.mutable_data_ptr<float>();
        float* yp = Y.mutable_data_ptr<float>();

        // Keep CPU copies for accuracy check
        std::vector<float> y_cpu(N);
        for (int64_t i = 0; i < N; ++i) {
            float x1 = dist(g_rng), x2 = dist(g_rng);
            xp[i * 2] = x1;
            xp[i * 2 + 1] = x2;
            yp[i] = (x1 * x1 + x2 * x2 < 1.0f) ? 1.0f : 0.0f;
            y_cpu[i] = yp[i];
        }
        X = to_device(X);
        Y = to_device(Y);

        model->zero_grad();
        Tensor logits = model->forward(X);
        Tensor probs = torch::autograd::sigmoid_autograd(logits);

        // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        Tensor ones = to_device(at::ones(probs.sizes()));
        Tensor eps_t = to_device(at::empty(probs.sizes()));
        eps_t.fill_(at::Scalar(1e-7f));
        Tensor log_p = torch::autograd::log_autograd(
            torch::autograd::add_autograd(probs, eps_t));
        Tensor log_1_p = torch::autograd::log_autograd(
            torch::autograd::add_autograd(
                torch::autograd::sub_autograd(ones, probs), eps_t));
        Tensor bce = torch::autograd::neg_autograd(
            torch::autograd::mean_autograd(
                torch::autograd::add_autograd(
                    torch::autograd::mul_autograd(Y, log_p),
                    torch::autograd::mul_autograd(
                        torch::autograd::sub_autograd(ones, Y), log_1_p))));

        float loss_val = move_to_cpu(bce).data_ptr<float>()[0];
        torch::autograd::backward({bce});
        optimizer.step();

        // Compute accuracy
        Tensor probs_cpu = move_to_cpu(probs);
        const float* pp = probs_cpu.data_ptr<float>();
        int correct = 0;
        for (int64_t i = 0; i < N; ++i) {
            float pred = pp[i] > 0.5f ? 1.0f : 0.0f;
            if (pred == y_cpu[i]) ++correct;
        }
        final_acc = 100.0f * correct / N;

        if (epoch % 10 == 0) {
            std::cout << "  Epoch " << epoch << " | Loss = " << loss_val
                      << " | Acc = " << final_acc << "%" << std::endl;
        }
    }
    print_result("Accuracy", final_acc);
}

// ============================================================================
// MODEL 3: 2-Layer MLP on XOR problem
// ============================================================================

class XORNet : public Module {
public:
    XORNet() : Module("XORNet") {
        fc1 = std::make_shared<Linear>(2, 16);
        fc2 = std::make_shared<Linear>(16, 1);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    Tensor forward(const Tensor& x) override {
        Tensor h = torch::autograd::relu_autograd(fc1->forward(x));
        return fc2->forward(h);
    }
private:
    std::shared_ptr<Linear> fc1, fc2;
};

void train_model_3() {
    print_header(3, "MLP on XOR", "2-layer MLP solving XOR (4 data points)");

    auto model = std::make_shared<XORNet>();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) model->to(g_device);
#endif
    SGD optimizer(model->parameters(), SGDOptions(0.05));

    // XOR data
    Tensor X = at::empty({4, 2});
    Tensor Y = at::empty({4, 1});
    float xor_x[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float xor_y[] = {0, 1, 1, 0};
    std::memcpy(X.mutable_data_ptr<float>(), xor_x, 8 * sizeof(float));
    std::memcpy(Y.mutable_data_ptr<float>(), xor_y, 4 * sizeof(float));
    X = to_device(X);
    Y = to_device(Y);

    float final_loss = 0;
    for (int epoch = 1; epoch <= 2000; ++epoch) {
        model->zero_grad();
        Tensor pred = model->forward(X);
        Tensor diff = torch::autograd::sub_autograd(pred, Y);
        Tensor loss = torch::autograd::mean_autograd(
            torch::autograd::mul_autograd(diff, diff));
        final_loss = move_to_cpu(loss).data_ptr<float>()[0];
        torch::autograd::backward({loss});
        optimizer.step();

        if (epoch % 500 == 0) {
            std::cout << "  Epoch " << epoch << " | MSE = " << final_loss << std::endl;
        }
    }

    // Show predictions
    {
        torch::autograd::NoGradGuard ng;
        Tensor out = move_to_cpu(model->forward(X));
        const float* p = out.data_ptr<float>();
        std::cout << "  Predictions: [0,0]=" << p[0] << " [0,1]=" << p[1]
                  << " [1,0]=" << p[2] << " [1,1]=" << p[3] << std::endl;
    }
    print_result("Final MSE", final_loss);
}

// ============================================================================
// MODEL 4: Simple MNIST (784 → 128 → 10, SGD)
// ============================================================================

class SimpleMNIST : public Module {
public:
    SimpleMNIST() : Module("SimpleMNIST") {
        fc1 = std::make_shared<Linear>(784, 128);
        fc2 = std::make_shared<Linear>(128, 10);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    Tensor forward(const Tensor& x) override {
        Tensor h = torch::autograd::relu_autograd(fc1->forward(x));
        return fc2->forward(h);
    }
private:
    std::shared_ptr<Linear> fc1, fc2;
};

void train_model_4(const std::vector<std::vector<uint8_t>>& train_images,
                   const std::vector<uint8_t>& train_labels,
                   const std::vector<std::vector<uint8_t>>& test_images,
                   const std::vector<uint8_t>& test_labels) {
    print_header(4, "Simple MNIST MLP", "784 → 128 → 10 (SGD lr=0.01, 2 epochs)");

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        // === CPU vs CUDA gradient comparison ===
        std::cout << "  [TEST] CPU vs CUDA gradient comparison..." << std::endl;
        // Small model: 4 -> 3
        auto small = std::make_shared<Linear>(4, 3);
        // Save weights as CPU copy
        Tensor w_cpu_copy = small->named_parameters()[0].second->data().clone();
        Tensor b_cpu_copy = small->named_parameters()[1].second->data().clone();

        // Create fixed input and target
        Tensor x_cpu = at::empty({2, 4});
        float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, 0.3f, 2.0f};
        std::memcpy(x_cpu.mutable_data_ptr<float>(), x_data, 8 * sizeof(float));
        Tensor t_cpu = at::empty({2});
        float t_data[] = {1.0f, 2.0f};
        std::memcpy(t_cpu.mutable_data_ptr<float>(), t_data, 2 * sizeof(float));

        // --- CPU forward+backward ---
        x_cpu.set_requires_grad(true);
        CrossEntropyLoss ce_cpu;
        Tensor logits_cpu = small->forward(x_cpu);
        Tensor loss_cpu = ce_cpu.forward(logits_cpu, t_cpu);
        torch::autograd::backward({loss_cpu});

        // Get CPU gradients
        Tensor grad_w_cpu = small->named_parameters()[0].second->grad();
        Tensor grad_b_cpu = small->named_parameters()[1].second->grad();
        std::cout << "  [TEST] CPU loss=" << loss_cpu.data_ptr<float>()[0] << std::endl;
        std::cout << "  [TEST] CPU logits[0][:3]=";
        const float* lc = logits_cpu.data_ptr<float>();
        for (int k = 0; k < 3; ++k) std::cout << lc[k] << " ";
        std::cout << std::endl;
        std::cout << "  [TEST] CPU grad_w[:3]=";
        const float* gw = grad_w_cpu.data_ptr<float>();
        for (int k = 0; k < 3; ++k) std::cout << gw[k] << " ";
        std::cout << std::endl;
        std::cout << "  [TEST] CPU grad_b[:3]=";
        const float* gb = grad_b_cpu.data_ptr<float>();
        for (int k = 0; k < 3; ++k) std::cout << gb[k] << " ";
        std::cout << std::endl;

        // --- CUDA forward+backward ---
        // Create new model with SAME weights on CUDA
        auto small_cuda = std::make_shared<Linear>(4, 3);
        small_cuda->named_parameters()[0].second->set_data(w_cpu_copy.clone());
        small_cuda->named_parameters()[1].second->set_data(b_cpu_copy.clone());
        small_cuda->to(g_device);

        Tensor x_cuda = at::to_cuda(x_cpu.detach());
        x_cuda.set_requires_grad(true);
        Tensor t_cuda = at::to_cuda(t_cpu);

        CrossEntropyLoss ce_cuda;
        Tensor logits_cuda = small_cuda->forward(x_cuda);
        Tensor loss_cuda = ce_cuda.forward(logits_cuda, t_cuda);
        torch::autograd::backward({loss_cuda});

        Tensor grad_w_cuda = small_cuda->named_parameters()[0].second->grad();
        Tensor grad_b_cuda = small_cuda->named_parameters()[1].second->grad();
        Tensor loss_cuda_cpu = at::to_cpu(loss_cuda);
        std::cout << "  [TEST] CUDA loss=" << loss_cuda_cpu.data_ptr<float>()[0] << std::endl;
        Tensor logits_cuda_cpu = at::to_cpu(logits_cuda);
        std::cout << "  [TEST] CUDA logits[0][:3]=";
        const float* lcc = logits_cuda_cpu.data_ptr<float>();
        for (int k = 0; k < 3; ++k) std::cout << lcc[k] << " ";
        std::cout << std::endl;
        Tensor gw_cuda_cpu = at::to_cpu(grad_w_cuda);
        std::cout << "  [TEST] CUDA grad_w[:3]=";
        const float* gwc = gw_cuda_cpu.data_ptr<float>();
        for (int k = 0; k < 3; ++k) std::cout << gwc[k] << " ";
        std::cout << std::endl;
        Tensor gb_cuda_cpu = at::to_cpu(grad_b_cuda);
        std::cout << "  [TEST] CUDA grad_b[:3]=";
        const float* gbc = gb_cuda_cpu.data_ptr<float>();
        for (int k = 0; k < 3; ++k) std::cout << gbc[k] << " ";
        std::cout << std::endl;

        // Cleanup
        torch::autograd::clear_autograd_graph(loss_cpu);
        torch::autograd::clear_autograd_graph(loss_cuda);
        std::cout << "  [TEST] Comparison done." << std::endl;
    }
#endif

    auto model = std::make_shared<SimpleMNIST>();
    // DIAGNOSTIC: Compare CPU vs CUDA training with SAME weights and data
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        std::cout << "  [DIAG] Same-weight CPU vs CUDA comparison (5 batches)..." << std::endl;

        // Create shared data: first batch
        int64_t B = 64;
        Tensor inp = at::empty({B, 784});
        Tensor tgt = at::empty({B});
        float* iip = inp.mutable_data_ptr<float>();
        float* ttp = tgt.mutable_data_ptr<float>();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t j = 0; j < 784; ++j)
                iip[b * 784 + j] = (train_images[b][j] / 255.0f - 0.1307f) / 0.3081f;
            ttp[b] = static_cast<float>(train_labels[b]);
        }

        // Save initial weights
        auto w1 = model->named_parameters()[0].second->data().clone();
        auto b1 = model->named_parameters()[1].second->data().clone();
        auto w2 = model->named_parameters()[2].second->data().clone();
        auto b2 = model->named_parameters()[3].second->data().clone();

        // === CPU training: 5 batches on same data ===
        {
            SGD opt(model->parameters(), SGDOptions(0.01));
            CrossEntropyLoss crit;
            std::cout << "  [CPU] ";
            for (int step = 0; step < 5; ++step) {
                model->zero_grad();
                Tensor logits_c = model->forward(inp);
                Tensor loss_c = crit.forward(logits_c, tgt);
                torch::autograd::backward({loss_c});
                opt.step();
                std::cout << "loss=" << loss_c.data_ptr<float>()[0] << " ";
                torch::autograd::clear_autograd_graph(loss_c);
            }
            std::cout << std::endl;
        }

        // === CUDA training: 5 batches on same data with SAME initial weights ===
        {
            // Reset model to saved weights
            auto model_cuda = std::make_shared<SimpleMNIST>();
            model_cuda->named_parameters()[0].second->set_data(w1.clone());
            model_cuda->named_parameters()[1].second->set_data(b1.clone());
            model_cuda->named_parameters()[2].second->set_data(w2.clone());
            model_cuda->named_parameters()[3].second->set_data(b2.clone());
            model_cuda->to(g_device);

            Tensor inp_cuda = at::to_cuda(inp);
            Tensor tgt_cuda = at::to_cuda(tgt);

            SGD opt(model_cuda->parameters(), SGDOptions(0.01));
            CrossEntropyLoss crit;
            std::cout << "  [CUDA] ";
            for (int step = 0; step < 5; ++step) {
                model_cuda->zero_grad();
                Tensor logits_c = model_cuda->forward(inp_cuda);
                Tensor loss_c = crit.forward(logits_c, tgt_cuda);
                torch::autograd::backward({loss_c});
                opt.step();
                std::cout << "loss=" << move_to_cpu(loss_c).data_ptr<float>()[0] << " ";
                torch::autograd::clear_autograd_graph(loss_c);
            }
            std::cout << std::endl;
        }

        // Reset model for actual training
        model = std::make_shared<SimpleMNIST>();
        model->to(g_device);
    }
#endif
    SGD optimizer(model->parameters(), SGDOptions(0.01));
    CrossEntropyLoss criterion;

    int64_t N = train_images.size();
    int64_t bs = 64;

    for (int epoch = 1; epoch <= 2; ++epoch) {
        model->train();
        std::vector<int64_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g_rng);

        float epoch_loss = 0;
        int64_t batches = 0;

        for (int64_t i = 0; i < N; i += bs) {
            int64_t B = std::min(bs, N - i);
            Tensor inputs = at::empty({B, 784});
            Tensor targets = at::empty({B});
            float* ip = inputs.mutable_data_ptr<float>();
            float* tp = targets.mutable_data_ptr<float>();
            for (int64_t b = 0; b < B; ++b) {
                int64_t idx = indices[i + b];
                for (int64_t j = 0; j < 784; ++j)
                    ip[b * 784 + j] = (train_images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
                tp[b] = static_cast<float>(train_labels[idx]);
            }

            model->zero_grad();
            Tensor logits = model->forward(to_device(inputs));
            Tensor loss = criterion.forward(logits, to_device(targets));
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += move_to_cpu(loss).data_ptr<float>()[0];
            ++batches;

            torch::autograd::clear_autograd_graph(loss);
        }
        float acc = evaluate_mnist(*model, test_images, test_labels);
        std::cout << "  Epoch " << epoch << " | Loss = " << epoch_loss / batches
                  << " | Test Acc = " << acc << "%" << std::endl;
    }
    float final_acc = evaluate_mnist(*model, test_images, test_labels);
    print_result("Test Accuracy", final_acc);
}

// ============================================================================
// MODEL 5: Deep MNIST (784 → 512 → 256 → 128 → 10, Adam)
// ============================================================================

class DeepMNIST : public Module {
public:
    DeepMNIST() : Module("DeepMNIST") {
        fc1 = std::make_shared<Linear>(784, 512);
        fc2 = std::make_shared<Linear>(512, 256);
        fc3 = std::make_shared<Linear>(256, 128);
        fc4 = std::make_shared<Linear>(128, 10);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc4", fc4);
    }
    Tensor forward(const Tensor& x) override {
        Tensor h = torch::autograd::relu_autograd(fc1->forward(x));
        h = torch::autograd::relu_autograd(fc2->forward(h));
        h = torch::autograd::relu_autograd(fc3->forward(h));
        return fc4->forward(h);
    }
private:
    std::shared_ptr<Linear> fc1, fc2, fc3, fc4;
};

void train_model_5(const std::vector<std::vector<uint8_t>>& train_images,
                   const std::vector<uint8_t>& train_labels,
                   const std::vector<std::vector<uint8_t>>& test_images,
                   const std::vector<uint8_t>& test_labels) {
    print_header(5, "Deep MNIST MLP", "784 → 512 → 256 → 128 → 10 (Adam lr=0.001, 2 epochs)");

    auto model = std::make_shared<DeepMNIST>();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) model->to(g_device);
#endif
    Adam optimizer(model->parameters(), AdamOptions(0.001));
    CrossEntropyLoss criterion;

    int64_t N = train_images.size();
    int64_t bs = 64;

    for (int epoch = 1; epoch <= 2; ++epoch) {
        model->train();
        std::vector<int64_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g_rng);

        float epoch_loss = 0;
        int64_t batches = 0;

        for (int64_t i = 0; i < N; i += bs) {
            int64_t B = std::min(bs, N - i);
            Tensor inputs = at::empty({B, 784});
            Tensor targets = at::empty({B});
            float* ip = inputs.mutable_data_ptr<float>();
            float* tp = targets.mutable_data_ptr<float>();
            for (int64_t b = 0; b < B; ++b) {
                int64_t idx = indices[i + b];
                for (int64_t j = 0; j < 784; ++j)
                    ip[b * 784 + j] = (train_images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
                tp[b] = static_cast<float>(train_labels[idx]);
            }

            model->zero_grad();
            Tensor logits = model->forward(to_device(inputs));
            Tensor loss = criterion.forward(logits, to_device(targets));
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += move_to_cpu(loss).data_ptr<float>()[0];
            ++batches;

            torch::autograd::clear_autograd_graph(loss);
        }
        float acc = evaluate_mnist(*model, test_images, test_labels);
        std::cout << "  Epoch " << epoch << " | Loss = " << epoch_loss / batches
                  << " | Test Acc = " << acc << "%" << std::endl;
    }
    float final_acc = evaluate_mnist(*model, test_images, test_labels);
    print_result("Test Accuracy", final_acc);
}

// ============================================================================
// MODEL 6: MNIST with Dropout (784 → 256 → 128 → 10, Adam + Dropout)
// ============================================================================

class DropoutMNIST : public Module {
public:
    DropoutMNIST() : Module("DropoutMNIST") {
        fc1 = std::make_shared<Linear>(784, 256);
        fc2 = std::make_shared<Linear>(256, 128);
        fc3 = std::make_shared<Linear>(128, 10);
        drop1 = std::make_shared<Dropout>(0.2);
        drop2 = std::make_shared<Dropout>(0.2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("drop1", drop1);
        register_module("drop2", drop2);
    }
    Tensor forward(const Tensor& x) override {
        Tensor h = torch::autograd::relu_autograd(fc1->forward(x));
        h = drop1->forward(h);
        h = torch::autograd::relu_autograd(fc2->forward(h));
        h = drop2->forward(h);
        return fc3->forward(h);
    }
private:
    std::shared_ptr<Linear> fc1, fc2, fc3;
    std::shared_ptr<Dropout> drop1, drop2;
};

void train_model_6(const std::vector<std::vector<uint8_t>>& train_images,
                   const std::vector<uint8_t>& train_labels,
                   const std::vector<std::vector<uint8_t>>& test_images,
                   const std::vector<uint8_t>& test_labels) {
    print_header(6, "MNIST with Dropout", "784 → 256 → 128 → 10 (Adam + Dropout 0.2, 2 epochs)");

    auto model = std::make_shared<DropoutMNIST>();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) model->to(g_device);
#endif
    Adam optimizer(model->parameters(), AdamOptions(0.001));
    CrossEntropyLoss criterion;

    int64_t N = train_images.size();
    int64_t bs = 64;

    for (int epoch = 1; epoch <= 2; ++epoch) {
        model->train();
        std::vector<int64_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g_rng);

        float epoch_loss = 0;
        int64_t batches = 0;

        for (int64_t i = 0; i < N; i += bs) {
            int64_t B = std::min(bs, N - i);
            Tensor inputs = at::empty({B, 784});
            Tensor targets = at::empty({B});
            float* ip = inputs.mutable_data_ptr<float>();
            float* tp = targets.mutable_data_ptr<float>();
            for (int64_t b = 0; b < B; ++b) {
                int64_t idx = indices[i + b];
                for (int64_t j = 0; j < 784; ++j)
                    ip[b * 784 + j] = (train_images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
                tp[b] = static_cast<float>(train_labels[idx]);
            }

            model->zero_grad();
            Tensor logits = model->forward(to_device(inputs));
            Tensor loss = criterion.forward(logits, to_device(targets));
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += move_to_cpu(loss).data_ptr<float>()[0];
            ++batches;

            torch::autograd::clear_autograd_graph(loss);
        }
        model->eval();
        float acc = evaluate_mnist(*model, test_images, test_labels);
        std::cout << "  Epoch " << epoch << " | Loss = " << epoch_loss / batches
                  << " | Test Acc = " << acc << "%" << std::endl;
    }
    model->eval();
    float final_acc = evaluate_mnist(*model, test_images, test_labels);
    print_result("Test Accuracy", final_acc);
}

// ============================================================================
// MODEL 7: RNN on sequence prediction (predict next element in sine wave)
// ============================================================================

class SineRNN : public Module {
public:
    SineRNN() : Module("SineRNN") {
        cell = std::make_shared<RNNCellImpl>(1, 32);
        fc = std::make_shared<Linear>(32, 1);
        register_module("cell", cell);
        register_module("fc", fc);
    }
    Tensor forward(const Tensor& x) override {
        // x: [batch, seq_len, 1]
        int64_t batch = x.size(0);
        int64_t seq_len = x.size(1);
        Tensor h = at::zeros({batch, 32});

        for (int64_t t = 0; t < seq_len; ++t) {
            Tensor xt = torch::autograd::select_autograd(x, 1, t); // [batch, 1]
            h = cell->forward(xt, h);
        }
        return fc->forward(h); // [batch, 1]
    }
private:
    std::shared_ptr<RNNCellImpl> cell;
    std::shared_ptr<Linear> fc;
};

void train_model_7() {
    print_header(7, "RNN Sine Wave", "Predict next value in sin(x) sequence (RNNCell + Linear)");

    auto model = std::make_shared<SineRNN>();
    Adam optimizer(model->parameters(), AdamOptions(0.005));

    int64_t seq_len = 10;
    int64_t batch = 32;
    float final_loss = 0;

    for (int epoch = 1; epoch <= 100; ++epoch) {
        // Generate sine wave sequences
        Tensor inputs = at::empty({batch, seq_len, 1});
        Tensor targets = at::empty({batch, 1});
        float* ip = inputs.mutable_data_ptr<float>();
        float* tp = targets.mutable_data_ptr<float>();

        std::uniform_real_distribution<float> phase_dist(0.0f, 6.28f);
        for (int64_t b = 0; b < batch; ++b) {
            float phase = phase_dist(g_rng);
            for (int64_t t = 0; t < seq_len; ++t) {
                ip[b * seq_len + t] = std::sin(phase + t * 0.5f);
            }
            tp[b] = std::sin(phase + seq_len * 0.5f);
        }

        model->zero_grad();
        Tensor pred = model->forward(inputs);
        Tensor diff = torch::autograd::sub_autograd(pred, targets);
        Tensor loss = torch::autograd::mean_autograd(
            torch::autograd::mul_autograd(diff, diff));
        final_loss = loss.data_ptr<float>()[0];
        torch::autograd::backward({loss});
        optimizer.step();

        torch::autograd::clear_autograd_graph(loss);

        if (epoch % 20 == 0) {
            std::cout << "  Epoch " << epoch << " | MSE = " << final_loss << std::endl;
        }
    }
    print_result("Final MSE", final_loss);
}

// ============================================================================
// MODEL 8: LSTM on sequence classification (sum positive vs negative)
// ============================================================================

class SeqClassLSTM : public Module {
public:
    SeqClassLSTM() : Module("SeqClassLSTM") {
        cell = std::make_shared<LSTMCellImpl>(1, 32);
        fc = std::make_shared<Linear>(32, 2); // binary classification
        register_module("cell", cell);
        register_module("fc", fc);
    }
    Tensor forward(const Tensor& x) override {
        // x: [batch, seq_len, 1]
        int64_t batch = x.size(0);
        int64_t seq_len = x.size(1);
        Tensor h = at::zeros({batch, 32});
        Tensor c = at::zeros({batch, 32});

        for (int64_t t = 0; t < seq_len; ++t) {
            Tensor xt = torch::autograd::select_autograd(x, 1, t);
            auto [h_new, c_new] = cell->forward_lstm(xt, h, c);
            h = h_new;
            c = c_new;
        }
        return fc->forward(h);
    }
private:
    std::shared_ptr<LSTMCellImpl> cell;
    std::shared_ptr<Linear> fc;
};

void train_model_8() {
    print_header(8, "LSTM Sequence Classifier",
                 "Classify sequences by sum sign (LSTM + Linear, 2 classes)");

    auto model = std::make_shared<SeqClassLSTM>();
    Adam optimizer(model->parameters(), AdamOptions(0.005));
    CrossEntropyLoss criterion;

    int64_t seq_len = 8;
    int64_t batch = 64;
    float final_acc = 0;

    for (int epoch = 1; epoch <= 80; ++epoch) {
        Tensor inputs = at::empty({batch, seq_len, 1});
        Tensor targets = at::empty({batch});
        float* ip = inputs.mutable_data_ptr<float>();
        float* tp = targets.mutable_data_ptr<float>();

        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int64_t b = 0; b < batch; ++b) {
            float sum = 0;
            for (int64_t t = 0; t < seq_len; ++t) {
                float v = nd(g_rng);
                ip[b * seq_len + t] = v;
                sum += v;
            }
            tp[b] = sum > 0 ? 1.0f : 0.0f;
        }

        model->zero_grad();
        Tensor logits = model->forward(inputs);
        Tensor loss = criterion.forward(logits, targets);
        torch::autograd::backward({loss});

        optimizer.step();

        // Accuracy
        {
            torch::autograd::NoGradGuard ng;
            Tensor preds = logits.argmax(1);
            const int64_t* pp = preds.data_ptr<int64_t>();
            int correct = 0;
            for (int64_t b = 0; b < batch; ++b) {
                if (pp[b] == static_cast<int64_t>(tp[b])) ++correct;
            }
            final_acc = 100.0f * correct / batch;
        }

        torch::autograd::clear_autograd_graph(loss);

        if (epoch % 20 == 0 || epoch <= 3) {
            std::cout << "  Epoch " << epoch << " | Loss = " << loss.data_ptr<float>()[0]
                      << " | Acc = " << final_acc << "%" << std::endl;
        }
    }
    print_result("Final Accuracy", final_acc);
}

// ============================================================================
// MODEL 9: GRU on pattern recognition (detect trend direction)
// ============================================================================

class TrendGRU : public Module {
public:
    TrendGRU() : Module("TrendGRU") {
        cell = std::make_shared<GRUCellImpl>(1, 24);
        fc = std::make_shared<Linear>(24, 2); // up or down
        register_module("cell", cell);
        register_module("fc", fc);
    }
    Tensor forward(const Tensor& x) override {
        int64_t batch = x.size(0);
        int64_t seq_len = x.size(1);
        Tensor h = at::zeros({batch, 24});

        for (int64_t t = 0; t < seq_len; ++t) {
            Tensor xt = torch::autograd::select_autograd(x, 1, t);
            h = cell->forward(xt, h);
        }
        return fc->forward(h);
    }
private:
    std::shared_ptr<GRUCellImpl> cell;
    std::shared_ptr<Linear> fc;
};

void train_model_9() {
    print_header(9, "GRU Trend Detector",
                 "Detect upward vs downward trend (GRU + Linear, 2 classes)");

    auto model = std::make_shared<TrendGRU>();
    Adam optimizer(model->parameters(), AdamOptions(0.005));
    CrossEntropyLoss criterion;

    int64_t seq_len = 10;
    int64_t batch = 64;
    float final_acc = 0;

    for (int epoch = 1; epoch <= 80; ++epoch) {
        Tensor inputs = at::empty({batch, seq_len, 1});
        Tensor targets = at::empty({batch});
        float* ip = inputs.mutable_data_ptr<float>();
        float* tp = targets.mutable_data_ptr<float>();

        std::normal_distribution<float> nd(0.0f, 0.3f);
        std::uniform_real_distribution<float> slope_dist(-1.0f, 1.0f);
        for (int64_t b = 0; b < batch; ++b) {
            float slope = slope_dist(g_rng);
            float base = nd(g_rng);
            for (int64_t t = 0; t < seq_len; ++t) {
                ip[b * seq_len + t] = base + slope * t / seq_len + nd(g_rng) * 0.1f;
            }
            tp[b] = slope > 0 ? 1.0f : 0.0f;
        }

        model->zero_grad();
        Tensor logits = model->forward(inputs);
        Tensor loss = criterion.forward(logits, targets);
        torch::autograd::backward({loss});
        optimizer.step();

        {
            torch::autograd::NoGradGuard ng;
            Tensor preds = logits.argmax(1);
            const int64_t* pp = preds.data_ptr<int64_t>();
            int correct = 0;
            for (int64_t b = 0; b < batch; ++b) {
                if (pp[b] == static_cast<int64_t>(tp[b])) ++correct;
            }
            final_acc = 100.0f * correct / batch;
        }

        torch::autograd::clear_autograd_graph(loss);

        if (epoch % 20 == 0) {
            std::cout << "  Epoch " << epoch << " | Loss = " << loss.data_ptr<float>()[0]
                      << " | Acc = " << final_acc << "%" << std::endl;
        }
    }
    print_result("Final Accuracy", final_acc);
}

// ============================================================================
// MODEL 10: Wide MNIST with Save/Load test (784 → 1024 → 512 → 10, AdamW)
// ============================================================================

class WideMNIST : public Module {
public:
    WideMNIST() : Module("WideMNIST") {
        fc1 = std::make_shared<Linear>(784, 1024);
        fc2 = std::make_shared<Linear>(1024, 512);
        fc3 = std::make_shared<Linear>(512, 10);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }
    Tensor forward(const Tensor& x) override {
        Tensor h = torch::autograd::relu_autograd(fc1->forward(x));
        h = torch::autograd::relu_autograd(fc2->forward(h));
        return fc3->forward(h);
    }
private:
    std::shared_ptr<Linear> fc1, fc2, fc3;
};

void train_model_10(const std::vector<std::vector<uint8_t>>& train_images,
                    const std::vector<uint8_t>& train_labels,
                    const std::vector<std::vector<uint8_t>>& test_images,
                    const std::vector<uint8_t>& test_labels) {
    print_header(10, "Wide MNIST + Serialization",
                 "784 → 1024 → 512 → 10 (AdamW lr=0.001, 2 epochs, save/load test)");

    auto model = std::make_shared<WideMNIST>();
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) model->to(g_device);
#endif
    AdamWOptions opts(0.001);
    opts.weight_decay_(0.01);
    AdamW optimizer(model->parameters(), opts);
    CrossEntropyLoss criterion;

    int64_t N = train_images.size();
    int64_t bs = 64;

    std::cout << "  Parameters: " << model->num_parameters() << std::endl;

    for (int epoch = 1; epoch <= 2; ++epoch) {
        model->train();
        std::vector<int64_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g_rng);

        float epoch_loss = 0;
        int64_t batches = 0;

        for (int64_t i = 0; i < N; i += bs) {
            int64_t B = std::min(bs, N - i);
            Tensor inputs = at::empty({B, 784});
            Tensor targets = at::empty({B});
            float* ip = inputs.mutable_data_ptr<float>();
            float* tp = targets.mutable_data_ptr<float>();
            for (int64_t b = 0; b < B; ++b) {
                int64_t idx = indices[i + b];
                for (int64_t j = 0; j < 784; ++j)
                    ip[b * 784 + j] = (train_images[idx][j] / 255.0f - 0.1307f) / 0.3081f;
                tp[b] = static_cast<float>(train_labels[idx]);
            }

            model->zero_grad();
            Tensor logits = model->forward(to_device(inputs));
            Tensor loss = criterion.forward(logits, to_device(targets));
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += move_to_cpu(loss).data_ptr<float>()[0];
            ++batches;

            torch::autograd::clear_autograd_graph(loss);
        }
        model->eval();
        float acc = evaluate_mnist(*model, test_images, test_labels);
        std::cout << "  Epoch " << epoch << " | Loss = " << epoch_loss / batches
                  << " | Test Acc = " << acc << "%" << std::endl;
    }

    model->eval();
    float acc_before = evaluate_mnist(*model, test_images, test_labels);

    // Save model (serialize CPU tensors)
    std::cout << "  Saving model..." << std::endl;
    auto state = model->state_dict();
    // Move tensors to CPU for serialization
    std::unordered_map<std::string, Tensor> cpu_state;
    for (auto& [k, v] : state) cpu_state[k] = move_to_cpu(v);
    torch::save_state_dict(cpu_state, "model10.ptor");

    // Load into fresh model
    auto model2 = std::make_shared<WideMNIST>();
    auto loaded_state = torch::load_state_dict("model10.ptor");
    model2->load_state_dict(loaded_state);
#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) model2->to(g_device);
#endif
    model2->eval();

    float acc_after = evaluate_mnist(*model2, test_images, test_labels);
    std::cout << "  Acc before save: " << acc_before << "%"
              << " | Acc after load: " << acc_after << "%" << std::endl;

    if (std::abs(acc_before - acc_after) < 0.01f) {
        std::cout << "  Serialization: PASS (identical accuracy)" << std::endl;
    } else {
        std::cout << "  Serialization: MISMATCH!" << std::endl;
    }

    print_result("Test Accuracy", acc_after);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_dir = ".";
    std::string device_str = "cpu";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
    }

    // Set device
    if (device_str == "cuda" || device_str == "gpu") {
#ifdef PT_USE_CUDA
        g_device = c10::Device(c10::DeviceType::CUDA, 0);
#else
        std::cerr << "CUDA not available in this build, using CPU" << std::endl;
#endif
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "  PromeTorch — 10 Models Training ("
              << (g_device.is_cuda() ? "CUDA" : "CPU") << ")" << std::endl;
    std::cout << "============================================================" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    // Models 1-3: synthetic data, no MNIST needed
    train_model_1();
    train_model_2();
    train_model_3();

    // Load MNIST for models 4-6, 10
    std::cout << "\n  Loading MNIST data..." << std::endl;

    auto train_images = load_mnist_images(data_dir + "/train-images-idx3-ubyte");
    auto train_labels = load_mnist_labels(data_dir + "/train-labels-idx1-ubyte");
    auto test_images = load_mnist_images(data_dir + "/t10k-images-idx3-ubyte");
    auto test_labels = load_mnist_labels(data_dir + "/t10k-labels-idx1-ubyte");

    if (train_images.empty()) {
        std::cerr << "  Failed to load MNIST from " << data_dir << std::endl;
        std::cerr << "  Skipping MNIST models (4, 5, 6, 10)" << std::endl;
    } else {
        std::cout << "  MNIST loaded: " << train_images.size() << " train, "
                  << test_images.size() << " test" << std::endl;
        train_model_4(train_images, train_labels, test_images, test_labels);
        train_model_5(train_images, train_labels, test_images, test_labels);
        train_model_6(train_images, train_labels, test_images, test_labels);
    }

    // Models 7-9: sequence models with synthetic data (always CPU — RNN cells create
    // intermediate tensors that don't auto-move to device yet)
    if (g_device.is_cuda()) {
        std::cout << "\n  [Models 7-9: RNN/LSTM/GRU running on CPU]" << std::endl;
    }
    train_model_7();
    train_model_8();
    train_model_9();

    // Model 10: wide MNIST + serialization
    if (!train_images.empty()) {
        train_model_10(train_images, train_labels, test_images, test_labels);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  ALL 10 MODELS COMPLETE (" << (g_device.is_cuda() ? "CUDA" : "CPU") << ")" << std::endl;
    std::cout << "  Total time: " << total_ms / 1000.0 << " seconds" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

#ifdef PT_USE_CUDA
    if (g_device.is_cuda()) {
        c10::cuda::cuda_shutdown();
    }
#endif

    return 0;
}
