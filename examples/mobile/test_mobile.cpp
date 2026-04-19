// Self-test for torch::mobile::export_model + MobileExecutor.
// Builds a 3-layer MLP, exports it, loads it back, and compares outputs.

#include "torch/mobile/executor.h"
#include "torch/nn/modules/container.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <cstdio>
#include <cmath>
#include <memory>

int main() {
    using namespace torch;

    // 3-layer MLP: 16 -> 32 -> 16 -> 4
    nn::Sequential model({
        std::shared_ptr<nn::Module>(new nn::Linear(16, 32)),
        std::shared_ptr<nn::Module>(new nn::ReLU()),
        std::shared_ptr<nn::Module>(new nn::Linear(32, 16)),
        std::shared_ptr<nn::Module>(new nn::Tanh()),
        std::shared_ptr<nn::Module>(new nn::Linear(16, 4)),
        std::shared_ptr<nn::Module>(new nn::Sigmoid()),
    });

    // Fixed random input (deterministic seed for reproducibility).
    std::srand(1234);
    at::Tensor x = at::randn({2, 16});

    at::Tensor y_orig = model.forward(x);

    const std::string path = "mlp_mobile.ptmb";
    if (!mobile::export_model(model, x, path)) {
        std::fprintf(stderr, "[mobile] export_model failed\n");
        return 1;
    }

    mobile::MobileExecutor exec;
    if (!exec.load(path)) {
        std::fprintf(stderr, "[mobile] load failed\n");
        return 2;
    }

    at::Tensor y_loaded = exec.forward(x);

    if (y_loaded.numel() != y_orig.numel()) {
        std::fprintf(stderr, "[mobile] numel mismatch: %lld vs %lld\n",
                     (long long)y_loaded.numel(), (long long)y_orig.numel());
        return 3;
    }

    const float* a = y_orig.contiguous().data_ptr<float>();
    const float* b = y_loaded.contiguous().data_ptr<float>();
    float max_diff = 0.0f;
    for (int64_t i = 0; i < y_orig.numel(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }

    std::printf("[mobile] weights=%zu ops=%zu max_diff=%.3e\n",
                exec.num_weights(), exec.num_ops(), max_diff);

    if (max_diff > 1e-4f) {
        std::fprintf(stderr, "[mobile] FAIL: max_diff above tolerance\n");
        return 4;
    }
    std::printf("[mobile] PASS\n");
    return 0;
}
