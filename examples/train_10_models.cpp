// ============================================================================
// 10 Models Training — Simple to Complex (CPU only)
// ============================================================================
// Uses ONLY PromeTorch framework. No PyTorch, no external ML libs.
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/serialization.h"
#include "torch/nn/modules/rnn.h"
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
        Tensor logits = model.forward(inputs);
        Tensor preds = logits.argmax(1);
        const float* pred_data = preds.data_ptr<float>();
        for (int64_t b = 0; b < B; ++b) {
            if (static_cast<int>(pred_data[b]) == labels[i + b]) ++correct;
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
    SGD optimizer(model->parameters(), SGDOptions(0.01));

    // Generate synthetic data: y = 3*x1 + 2*x2 - 1
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int64_t N = 500;
    float total_loss = 0;

    for (int epoch = 1; epoch <= 100; ++epoch) {
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

        model->zero_grad();
        Tensor pred = model->forward(X);  // [N, 1]
        Tensor diff = torch::autograd::sub_autograd(pred, Y);
        Tensor loss = torch::autograd::mean_autograd(
            torch::autograd::mul_autograd(diff, diff));

        total_loss = loss.data_ptr<float>()[0];
        torch::autograd::backward({loss});
        optimizer.step();

        if (epoch % 20 == 0) {
            std::cout << "  Epoch " << epoch << " | MSE = " << total_loss << std::endl;
        }
    }

    // Check learned weights
    auto* w = model->get_parameter("weight");
    auto* b = model->get_parameter("bias");
    std::cout << "  Learned: w1=" << w->data().data_ptr<float>()[0]
              << ", w2=" << w->data().data_ptr<float>()[1]
              << ", b=" << b->data().data_ptr<float>()[0]
              << " (true: 3, 2, -1)" << std::endl;
    print_result("Final MSE", total_loss);
}

// ============================================================================
// MODEL 2: Logistic Regression (binary classification)
// ============================================================================

void train_model_2() {
    print_header(2, "Logistic Regression", "Binary classification: x₁²+x₂² < 1 (BCE loss)");

    auto model = std::make_shared<Linear>(2, 1);
    SGD optimizer(model->parameters(), SGDOptions(0.1));

    std::normal_distribution<float> dist(0.0f, 1.5f);
    int64_t N = 200;
    float final_acc = 0;

    for (int epoch = 1; epoch <= 50; ++epoch) {
        Tensor X = at::empty({N, 2});
        Tensor Y = at::empty({N, 1});
        float* xp = X.mutable_data_ptr<float>();
        float* yp = Y.mutable_data_ptr<float>();

        int correct = 0;
        for (int64_t i = 0; i < N; ++i) {
            float x1 = dist(g_rng), x2 = dist(g_rng);
            xp[i * 2] = x1;
            xp[i * 2 + 1] = x2;
            yp[i] = (x1 * x1 + x2 * x2 < 1.0f) ? 1.0f : 0.0f;
        }

        model->zero_grad();
        Tensor logits = model->forward(X);
        Tensor probs = torch::autograd::sigmoid_autograd(logits);

        // BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        Tensor ones = at::ones(probs.sizes());
        Tensor eps_t = at::empty(probs.sizes());
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

        float loss_val = bce.data_ptr<float>()[0];
        torch::autograd::backward({bce});
        optimizer.step();

        // Compute accuracy
        const float* pp = probs.data_ptr<float>();
        for (int64_t i = 0; i < N; ++i) {
            float pred = pp[i] > 0.5f ? 1.0f : 0.0f;
            if (pred == yp[i]) ++correct;
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
    SGD optimizer(model->parameters(), SGDOptions(0.05));

    // XOR data
    Tensor X = at::empty({4, 2});
    Tensor Y = at::empty({4, 1});
    float xor_x[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float xor_y[] = {0, 1, 1, 0};
    std::memcpy(X.mutable_data_ptr<float>(), xor_x, 8 * sizeof(float));
    std::memcpy(Y.mutable_data_ptr<float>(), xor_y, 4 * sizeof(float));

    float final_loss = 0;
    for (int epoch = 1; epoch <= 2000; ++epoch) {
        model->zero_grad();
        Tensor pred = model->forward(X);
        Tensor diff = torch::autograd::sub_autograd(pred, Y);
        Tensor loss = torch::autograd::mean_autograd(
            torch::autograd::mul_autograd(diff, diff));
        final_loss = loss.data_ptr<float>()[0];
        torch::autograd::backward({loss});
        optimizer.step();

        if (epoch % 500 == 0) {
            std::cout << "  Epoch " << epoch << " | MSE = " << final_loss << std::endl;
        }
    }

    // Show predictions
    {
        torch::autograd::NoGradGuard ng;
        Tensor out = model->forward(X);
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

    auto model = std::make_shared<SimpleMNIST>();
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
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += loss.data_ptr<float>()[0];
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
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += loss.data_ptr<float>()[0];
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
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += loss.data_ptr<float>()[0];
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
            Tensor xt = at::native::select(x, 1, t); // [batch, 1]
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
            Tensor xt = at::native::select(x, 1, t);
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
            const float* pp = preds.data_ptr<float>();
            int correct = 0;
            for (int64_t b = 0; b < batch; ++b) {
                if (static_cast<int>(pp[b]) == static_cast<int>(tp[b])) ++correct;
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
            Tensor xt = at::native::select(x, 1, t);
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
            const float* pp = preds.data_ptr<float>();
            int correct = 0;
            for (int64_t b = 0; b < batch; ++b) {
                if (static_cast<int>(pp[b]) == static_cast<int>(tp[b])) ++correct;
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
    AdamWOptions opts(0.001);
    opts.weight_decay_ = 0.01;
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
            Tensor logits = model->forward(inputs);
            Tensor loss = criterion.forward(logits, targets);
            torch::autograd::backward({loss});
            optimizer.step();

            epoch_loss += loss.data_ptr<float>()[0];
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

    // Save model
    std::cout << "  Saving model..." << std::endl;
    auto state = model->state_dict();
    torch::save_state_dict(state, "model10.ptor");

    // Load into fresh model
    auto model2 = std::make_shared<WideMNIST>();
    auto loaded_state = torch::load_state_dict("model10.ptor");
    model2->load_state_dict(loaded_state);
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
    std::cout << "============================================================" << std::endl;
    std::cout << "  PromeTorch — 10 Models Training (CPU)" << std::endl;
    std::cout << "============================================================" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    // Models 1-3: synthetic data, no MNIST needed
    train_model_1();
    train_model_2();
    train_model_3();

    // Load MNIST for models 4-6, 10
    std::cout << "\n  Loading MNIST data..." << std::endl;
    std::string data_dir = ".";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) data_dir = argv[++i];
    }

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

    // Models 7-9: sequence models with synthetic data
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
    std::cout << "  ALL 10 MODELS COMPLETE" << std::endl;
    std::cout << "  Total time: " << total_ms / 1000.0 << " seconds" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}
