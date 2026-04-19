// ============================================================================
// Self-test for torch::nn::parallel::Pipeline
// ----------------------------------------------------------------------------
// Builds a 4-layer MLP Sequential, splits it across 2 pipeline stages with
// 4 micro-batches, and checks that the pipeline output matches the direct
// Sequential forward (same weights) within float tolerance.
// ============================================================================

#include "torch/nn/modules/container.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/activation.h"
#include "torch/nn/parallel/pipeline.h"
#include "aten/src/ATen/ATen.h"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace torch::nn;
using at::Tensor;

int main(int argc, char** argv) {
    const int64_t in_features  = 32;
    const int64_t hidden       = 64;
    const int64_t out_features = 16;
    const int64_t batch        = 16;
    const int     num_stages   = 2;
    const int     chunks       = 4;

    // --activations-only: skip Linear layers (use only ReLU).
    // Useful on platforms where BLAS cannot be called from non-main threads
    // (e.g. Elbrus E2K + EML — cblas_sgemm SIGILLs on pthreads).
    bool activations_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--activations-only") activations_only = true;
    }

    // Build a 4-layer MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear
    // (we keep an even number of layers for clean partitioning across 2 stages)
    auto seq = std::make_shared<Sequential>();
    if (!activations_only) {
        seq->push_back(std::make_shared<Linear>(in_features, hidden));
        seq->push_back(std::make_shared<ReLU>());
        seq->push_back(std::make_shared<Linear>(hidden, hidden));
        seq->push_back(std::make_shared<ReLU>());
        seq->push_back(std::make_shared<Linear>(hidden, hidden));
        seq->push_back(std::make_shared<ReLU>());
        seq->push_back(std::make_shared<Linear>(hidden, out_features));
        seq->push_back(std::make_shared<ReLU>());
    } else {
        // 4 ReLU layers — exercises threading + queues + cat without sgemm.
        seq->push_back(std::make_shared<ReLU>());
        seq->push_back(std::make_shared<ReLU>());
        seq->push_back(std::make_shared<ReLU>());
        seq->push_back(std::make_shared<ReLU>());
    }

    // Wrap into Pipeline (shares the same Module shared_ptrs => same weights).
    auto pipe = std::make_shared<parallel::Pipeline>(seq, num_stages, chunks);

    // Random input. activations_only mode keeps width = in_features throughout.
    Tensor x = activations_only ? at::randn({batch, in_features})
                                : at::randn({batch, in_features});
    (void)hidden; (void)out_features;

    // Reference: direct sequential forward
    Tensor y_ref = seq->forward(x);

    // Pipeline forward
    Tensor y_pipe = pipe->forward(x);

    // Compare shapes
    if (y_ref.dim() != y_pipe.dim() ||
        y_ref.size(0) != y_pipe.size(0) ||
        y_ref.size(1) != y_pipe.size(1)) {
        std::fprintf(stderr,
            "FAIL: shape mismatch ref [%lld,%lld] vs pipe [%lld,%lld]\n",
            (long long)y_ref.size(0), (long long)y_ref.size(1),
            (long long)y_pipe.size(0), (long long)y_pipe.size(1));
        return 1;
    }

    // Element-wise compare
    Tensor a = y_ref.contiguous();
    Tensor b = y_pipe.contiguous();
    const float* pa = a.data_ptr<float>();
    const float* pb = b.data_ptr<float>();
    int64_t n = a.numel();
    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::fabs(static_cast<double>(pa[i]) -
                             static_cast<double>(pb[i]));
        if (d > max_abs) max_abs = d;
    }

    std::printf("Pipeline self-test: max |ref - pipe| = %.3e (n=%lld)\n",
                max_abs, (long long)n);

    const double tol = 1e-5;
    if (max_abs > tol) {
        std::fprintf(stderr, "FAIL: tolerance exceeded (tol=%.1e)\n", tol);
        return 1;
    }

    // Sanity: pipeline parameter count == sequential parameter count.
    auto pipe_params = pipe->parameters();
    auto seq_params  = seq->parameters(true);
    if (pipe_params.size() != seq_params.size()) {
        std::fprintf(stderr, "FAIL: parameter count mismatch %zu vs %zu\n",
                     pipe_params.size(), seq_params.size());
        return 1;
    }

    std::printf("PASS (%zu params, stages=%d, chunks=%d)\n",
                pipe_params.size(), num_stages, chunks);
    return 0;
}
