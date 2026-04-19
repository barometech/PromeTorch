// ============================================================================
// Standalone verifier for TEST_PLAN §5.9 fix:
// TransformerEncoderLayer CUDA forward must not crash and must produce a
// finite output. Runs 10 forward passes and compares CUDA output to CPU
// output using cosine similarity.
// ============================================================================
#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/nn/modules/attention.h"
#include "torch/nn/modules/transformer.h"
#include "torch/csrc/autograd/autograd.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#endif
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

using namespace torch::nn;
using at::Tensor;

static float cos_sim(const Tensor& a, const Tensor& b) {
    const float* ap = a.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    int64_t n = a.numel();
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; ++i) {
        dot += double(ap[i]) * double(bp[i]);
        na  += double(ap[i]) * double(ap[i]);
        nb  += double(bp[i]) * double(bp[i]);
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

static bool finite_check(const Tensor& t) {
    const float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

// Copy data from one module to another (same architecture) so that they
// produce identical outputs. Does host-to-host memcpy on the Parameter
// Tensor storage, which works only when both modules' params live on CPU.
static void copy_params(Module& dst, Module& src) {
    auto dst_params = dst.parameters();
    auto src_params = src.parameters();
    if (dst_params.size() != src_params.size()) {
        std::cerr << "param count mismatch " << dst_params.size()
                  << " vs " << src_params.size() << "\n";
        std::exit(1);
    }
    for (size_t i = 0; i < dst_params.size(); ++i) {
        Tensor s = src_params[i]->data();
        Tensor d = dst_params[i]->data();
        if (s.numel() != d.numel()) {
            std::cerr << "param " << i << " numel mismatch\n";
            std::exit(1);
        }
        std::memcpy(d.mutable_data_ptr<float>(), s.data_ptr<float>(),
                    s.numel() * sizeof(float));
    }
}

int main() {
#ifndef PT_USE_CUDA
    std::cout << "CUDA not built. Skipping.\n";
    return 0;
#else
    torch::autograd::NoGradGuard no_grad;

    const int64_t B = 4, T = 8, D = 32, H = 4;

    // ------------------------------------------------------------
    // Part 1: LayerNorm alone (CPU vs CUDA-bounce)
    // ------------------------------------------------------------
    {
        std::srand(42);
        auto ln_cpu = std::make_shared<LayerNorm>(std::vector<int64_t>{D}, 1e-5);
        auto ln_cuda = std::make_shared<LayerNorm>(std::vector<int64_t>{D}, 1e-5);
        copy_params(*ln_cuda, *ln_cpu);
        ln_cuda->to(c10::Device(c10::DeviceType::CUDA, 0));
        ln_cpu->eval(); ln_cuda->eval();

        Tensor x = at::empty({B, T, D});
        float* p = x.mutable_data_ptr<float>();
        for (int64_t i = 0; i < x.numel(); ++i)
            p[i] = static_cast<float>((std::rand() % 1000 - 500) * 0.001);

        Tensor y_cpu  = ln_cpu->forward(x);
        Tensor y_cuda = ln_cuda->forward(at::to_cuda(x));
        Tensor y_cuda_cpu = at::to_cpu(y_cuda);
        std::cout << "[LayerNorm] finite=" << finite_check(y_cuda_cpu)
                  << " cos_sim=" << cos_sim(y_cpu, y_cuda_cpu) << std::endl;
    }

    // ------------------------------------------------------------
    // Part 2: MultiheadAttention alone (CPU vs CUDA-bounce)
    // ------------------------------------------------------------
    {
        std::srand(42);
        auto attn_cpu  = std::make_shared<MultiheadAttention>(
            D, H, 0.0, true, false, false, 0, 0, /*batch_first=*/true);
        auto attn_cuda = std::make_shared<MultiheadAttention>(
            D, H, 0.0, true, false, false, 0, 0, /*batch_first=*/true);
        copy_params(*attn_cuda, *attn_cpu);
        attn_cuda->to(c10::Device(c10::DeviceType::CUDA, 0));
        attn_cpu->eval(); attn_cuda->eval();

        Tensor x = at::empty({B, T, D});
        float* p = x.mutable_data_ptr<float>();
        for (int64_t i = 0; i < x.numel(); ++i)
            p[i] = static_cast<float>((std::rand() % 1000 - 500) * 0.001);

        Tensor y_cpu = attn_cpu->forward_attention(x, x, x).first;
        Tensor xc = at::to_cuda(x);
        Tensor y_cuda = attn_cuda->forward_attention(xc, xc, xc).first;
        Tensor y_cuda_cpu = at::to_cpu(y_cuda);
        std::cout << "[Attention] finite=" << finite_check(y_cuda_cpu)
                  << " cos_sim=" << cos_sim(y_cpu, y_cuda_cpu) << std::endl;
    }

    // ------------------------------------------------------------
    // Part 3: Full TransformerEncoderLayer (CPU vs CUDA), 10 forwards
    // ------------------------------------------------------------
    std::srand(42);
    auto layer_cpu = std::make_shared<TransformerEncoderLayer>(
        D, H, D * 4, 0.0, "relu", 1e-5, /*batch_first=*/true, /*norm_first=*/false);
    auto layer_cuda = std::make_shared<TransformerEncoderLayer>(
        D, H, D * 4, 0.0, "relu", 1e-5, /*batch_first=*/true, /*norm_first=*/false);
    copy_params(*layer_cuda, *layer_cpu);
    layer_cuda->to(c10::Device(c10::DeviceType::CUDA, 0));
    layer_cpu->eval();  layer_cuda->eval();

    bool all_ok = true;
    float min_sim = 1.0f, max_sim = 0.0f;
    for (int step = 0; step < 10; ++step) {
        Tensor input = at::empty({B, T, D});
        float* p = input.mutable_data_ptr<float>();
        for (int64_t i = 0; i < input.numel(); ++i) {
            p[i] = static_cast<float>((std::rand() % 1000 - 500) * 0.001);
        }

        Tensor out_cpu = layer_cpu->forward(input);
        Tensor input_cuda = at::to_cuda(input);
        Tensor out_cuda = layer_cuda->forward(input_cuda);
        Tensor out_cuda_cpu = at::to_cpu(out_cuda);

        bool fin = finite_check(out_cuda_cpu);
        float cs = cos_sim(out_cpu, out_cuda_cpu);
        if (cs < min_sim) min_sim = cs;
        if (cs > max_sim) max_sim = cs;
        std::cout << "[Encoder] step " << step
                  << ": cuda_finite=" << fin
                  << " cos_sim=" << cs << std::endl;
        if (!fin) all_ok = false;
    }

    std::cout << "--- cos_sim min=" << min_sim << " max=" << max_sim << " ---\n";
    if (!all_ok) {
        std::cout << "FAIL: non-finite outputs on CUDA path\n";
        return 1;
    }
    std::cout << "PASS: TransformerEncoderLayer CUDA forward does not crash; output finite\n";
    return 0;
#endif
}
