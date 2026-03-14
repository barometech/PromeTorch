// ============================================================================
// PromeTorch vs PyTorch Benchmark — PromeTorch side
// CPU single-threaded, outputs JSON to stdout
// ============================================================================
#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <cstdlib>
#include <fstream>

using namespace at;
using namespace torch;
using namespace torch::nn;
using namespace torch::optim;
using namespace torch::autograd;

static const int WARMUP = 5;

static double bench(std::function<void()> fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms / iters;
}

static bool first_entry = true;
static std::ostream* out_stream = &std::cout;

static void emit(const std::string& key, double val) {
    if (!first_entry) *out_stream << ",\n";
    first_entry = false;
    *out_stream << "  \"" << key << "\": " << std::fixed << std::setprecision(4) << val;
}

// ============================================================================
// 1. Tensor Creation
// ============================================================================
static void bench_creation() {
    std::cerr << "=== Tensor Creation ===\n";
    for (int sz : {64, 256, 1024, 2048}) {
        double t_randn = bench([sz]() { auto t = randn({(int64_t)sz, (int64_t)sz}); }, WARMUP, 500);
        double t_zeros = bench([sz]() { auto t = zeros({(int64_t)sz, (int64_t)sz}); }, WARMUP, 500);
        double t_ones  = bench([sz]() { auto t = ones({(int64_t)sz, (int64_t)sz}); }, WARMUP, 500);

        emit("randn_" + std::to_string(sz), t_randn);
        emit("zeros_" + std::to_string(sz), t_zeros);
        emit("ones_" + std::to_string(sz), t_ones);
        std::cerr << "  " << sz << "x" << sz
                  << ": randn=" << std::fixed << std::setprecision(4) << t_randn
                  << " zeros=" << t_zeros << " ones=" << t_ones << " ms\n";
    }
}

// ============================================================================
// 2. Element-wise Operations (1024x1024)
// ============================================================================
static void bench_elementwise() {
    std::cerr << "=== Element-wise Ops ===\n";
    Tensor a = randn({1024, 1024});
    Tensor b = randn({1024, 1024});
    Tensor a_pos = at::native::abs(a) + Scalar(1e-6f);

    struct Op { std::string name; std::function<void()> fn; };
    std::vector<Op> ops = {
        {"add_1024",     [&]() { auto r = a + b; }},
        {"mul_1024",     [&]() { auto r = a * b; }},
        {"sub_1024",     [&]() { auto r = a - b; }},
        {"div_1024",     [&]() { auto r = a / b; }},
        {"exp_1024",     [&]() { auto r = at::native::exp(a); }},
        {"log_1024",     [&]() { auto r = at::native::log(a_pos); }},
        {"sin_1024",     [&]() { auto r = at::native::sin(a); }},
        {"cos_1024",     [&]() { auto r = at::native::cos(a); }},
        {"tanh_1024",    [&]() { auto r = at::native::tanh(a); }},
        {"sigmoid_1024", [&]() { auto r = at::native::sigmoid(a); }},
        {"relu_1024",    [&]() { auto r = at::native::relu(a); }},
        {"sqrt_1024",    [&]() { auto r = at::native::sqrt(a_pos); }},
        {"abs_1024",     [&]() { auto r = at::native::abs(a); }},
        {"neg_1024",     [&]() { auto r = at::native::neg(a); }},
    };

    for (auto& op : ops) {
        double t = bench(op.fn, WARMUP, 200);
        emit(op.name, t);
        std::cerr << "  " << op.name << ": " << std::fixed << std::setprecision(4) << t << " ms\n";
    }
}

// ============================================================================
// 3. Reductions (1024x1024)
// ============================================================================
static void bench_reductions() {
    std::cerr << "=== Reductions ===\n";
    Tensor a = randn({1024, 1024});

    struct Op { std::string name; std::function<void()> fn; };
    std::vector<Op> ops = {
        {"sum_1024",      [&]() { auto r = a.sum(); }},
        {"mean_1024",     [&]() { auto r = a.mean(); }},
        {"max_1024",      [&]() { auto r = a.max(); }},
        {"min_1024",      [&]() { auto r = a.min(); }},
        {"var_1024",      [&]() { auto r = a.var(); }},
        {"std_1024",      [&]() { auto r = a.std(); }},
        {"argmax_1024",   [&]() { auto r = a.argmax(); }},
        {"sum_dim0_1024", [&]() { auto r = a.sum(0); }},
        {"mean_dim1_1024",[&]() { auto r = a.mean(1); }},
    };

    for (auto& op : ops) {
        double t = bench(op.fn, WARMUP, 200);
        emit(op.name, t);
        std::cerr << "  " << op.name << ": " << std::fixed << std::setprecision(4) << t << " ms\n";
    }
}

// ============================================================================
// 4. Linear Algebra
// ============================================================================
static void bench_linalg() {
    std::cerr << "=== Linear Algebra ===\n";
    for (int sz : {256, 512, 1024, 2048}) {
        Tensor a = randn({(int64_t)sz, (int64_t)sz});
        Tensor b = randn({(int64_t)sz, (int64_t)sz});
        int it = (sz >= 2048) ? 3 : (sz >= 1024) ? 5 : (sz >= 512) ? 20 : 100;
        double t = bench([&]() { auto r = a.mm(b); }, WARMUP, it);
        double gflops = (2.0 * sz * sz * sz) / (t * 1e-3) / 1e9;
        emit("mm_" + std::to_string(sz), t);
        std::cerr << "  mm_" << sz << ": " << std::fixed << std::setprecision(4)
                  << t << " ms (" << std::setprecision(1) << gflops << " GFLOPS)\n";
    }

    // mv
    {
        Tensor m = randn({512, 256});
        Tensor v = randn({256});
        double t = bench([&]() { auto r = m.mv(v); }, WARMUP, 500);
        emit("mv_512x256", t);
        std::cerr << "  mv_512x256: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }

    // bmm
    {
        Tensor ba = randn({8, 256, 256});
        Tensor bb = randn({8, 256, 256});
        double t = bench([&]() { auto r = ba.bmm(bb); }, WARMUP, 10);
        emit("bmm_8x256", t);
        std::cerr << "  bmm_8x256: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }

    // dot
    {
        Tensor d1 = randn({10000});
        Tensor d2 = randn({10000});
        double t = bench([&]() { auto r = d1.dot(d2); }, WARMUP, 1000);
        emit("dot_10k", t);
        std::cerr << "  dot_10k: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }
}

// ============================================================================
// 5. Autograd: Linear forward+backward
// ============================================================================
static void bench_autograd_fn() {
    std::cerr << "=== Autograd ===\n";
    auto linear = std::make_shared<Linear>(512, 256);
    Tensor x = randn({64, 512});
    x.set_requires_grad(true);

    auto fn = [&]() {
        linear->zero_grad();
        Tensor y = linear->forward(x);
        // sum_autograd for backward
        Tensor loss = sum_autograd(y);
        tensor_backward(loss);
    };

    double t = bench(fn, WARMUP, 100);
    emit("autograd_linear_fwd_bwd", t);
    std::cerr << "  autograd_linear_fwd_bwd: " << std::fixed << std::setprecision(4) << t << " ms\n";
}

// ============================================================================
// 6. NN Modules Forward
// ============================================================================
static void bench_nn_modules() {
    std::cerr << "=== NN Modules ===\n";
    torch::autograd::NoGradGuard no_grad;

    // Linear forward
    {
        auto lin = std::make_shared<Linear>(512, 256);
        Tensor x = randn({64, 512});
        double t = bench([&]() { auto r = lin->forward(x); }, WARMUP, 200);
        emit("nn_linear_fwd", t);
        std::cerr << "  nn_linear_fwd: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }

    // Conv2d forward: (16, 3, 32, 32) -> Conv2d(3, 16, 3, stride=1, padding=1)
    {
        auto conv = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
        Tensor x = randn({16, 3, 32, 32});
        double t = bench([&]() { auto r = conv->forward(x); }, WARMUP, 100);
        emit("nn_conv2d_fwd", t);
        std::cerr << "  nn_conv2d_fwd: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }

    // BatchNorm1d forward
    {
        auto bn = std::make_shared<BatchNorm1d>(256);
        bn->train();
        Tensor x = randn({64, 256});
        double t = bench([&]() { auto r = bn->forward(x); }, WARMUP, 200);
        emit("nn_batchnorm1d_fwd", t);
        std::cerr << "  nn_batchnorm1d_fwd: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }

    // LSTM forward: (32, 10, 128) with batch_first=true
    {
        auto lstm = std::make_shared<LSTM>(128, 64, 1, true, true);
        Tensor x = randn({32, 10, 128});
        double t = bench([&]() { auto r = lstm->forward(x); }, WARMUP, 100);
        emit("nn_lstm_fwd", t);
        std::cerr << "  nn_lstm_fwd: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }
}

// ============================================================================
// 7. Optimizer Step (Adam, SGD)
// ============================================================================
static void bench_optimizers() {
    std::cerr << "=== Optimizers ===\n";

    // Build 3-layer MLP using Sequential
    auto model = std::make_shared<Sequential>(
        std::initializer_list<ModulePtr>{
            std::make_shared<Linear>(256, 128),
            std::make_shared<ReLU>(),
            std::make_shared<Linear>(128, 64),
            std::make_shared<ReLU>(),
            std::make_shared<Linear>(64, 10)
        }
    );

    Tensor x = randn({64, 256});
    // Target: float tensor with class indices
    Tensor target = at::empty({64});
    float* tgt = target.mutable_data_ptr<float>();
    for (int i = 0; i < 64; ++i) tgt[i] = static_cast<float>(std::rand() % 10);

    CrossEntropyLoss criterion;

    // Adam
    {
        Adam opt(model->parameters(), AdamOptions(0.001));
        auto fn = [&]() {
            opt.zero_grad();
            Tensor out = model->forward(x);
            Tensor loss = criterion.forward(out, target);
            tensor_backward(loss);
            opt.step();
        };
        double t = bench(fn, 2, 10);
        emit("optim_adam_step", t);
        std::cerr << "  optim_adam_step: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }

    // SGD
    {
        SGD opt(model->parameters(), SGDOptions(0.01));
        auto fn = [&]() {
            opt.zero_grad();
            Tensor out = model->forward(x);
            Tensor loss = criterion.forward(out, target);
            tensor_backward(loss);
            opt.step();
        };
        double t = bench(fn, 2, 10);
        emit("optim_sgd_step", t);
        std::cerr << "  optim_sgd_step: " << std::fixed << std::setprecision(4) << t << " ms\n";
    }
}

// ============================================================================
// 8. Training Loop: MLP 784->512->256->10, 100 batches
// ============================================================================
static void bench_training() {
    std::cerr << "=== Training Loop ===\n";

    auto model = std::make_shared<Sequential>(
        std::initializer_list<ModulePtr>{
            std::make_shared<Linear>(784, 512),
            std::make_shared<ReLU>(),
            std::make_shared<Linear>(512, 256),
            std::make_shared<ReLU>(),
            std::make_shared<Linear>(256, 10)
        }
    );

    Adam opt(model->parameters(), AdamOptions(0.001));
    CrossEntropyLoss criterion;

    const int N_BATCHES = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < N_BATCHES; ++b) {
        Tensor x = randn({64, 784});
        Tensor target = at::empty({64});
        float* tgt = target.mutable_data_ptr<float>();
        for (int i = 0; i < 64; ++i) tgt[i] = static_cast<float>(std::rand() % 10);

        opt.zero_grad();
        Tensor out = model->forward(x);
        Tensor loss = criterion.forward(out, target);
        tensor_backward(loss);
        opt.step();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    emit("train_100batch_total_ms", total_ms * (100.0 / N_BATCHES));
    emit("train_per_batch_ms", total_ms / N_BATCHES);
    std::cerr << "  " << N_BATCHES << " batches: " << std::fixed << std::setprecision(1)
              << total_ms << " ms total, " << std::setprecision(2) << total_ms / N_BATCHES << " ms/batch\n";
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::srand(42);

    // Output to file if --output specified, otherwise stdout
    std::ofstream file_out;
    if (argc > 2 && std::string(argv[1]) == "--output") {
        file_out.open(argv[2]);
        out_stream = &file_out;
    }

    *out_stream << "{\n";

    bench_creation();
    bench_elementwise();
    bench_reductions();
    bench_linalg();
    bench_autograd_fn();
    bench_nn_modules();
    bench_optimizers();
    bench_training();

    *out_stream << "\n}\n";

    if (file_out.is_open()) {
        file_out.close();
        std::cerr << "\nResults saved to " << argv[2] << "\n";
    }

    return 0;
}
