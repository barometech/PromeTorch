// ============================================================================
// Self-test for torch::jit::compile (CPU-only).
//
// Builds f(x) = relu(x * 2 + 1), compiles it, runs on a 4x4 input, compares
// against eager execution, and benchmarks replay vs eager latency.
// ============================================================================

#include "torch/nn/nn.h"
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "torch/jit/compile.h"
#include "torch/jit/codegen_cpp.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace torch::jit;
using at::Tensor;

static double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

static Tensor eager_f(const Tensor& x) {
    // f(x) = relu(x * 2 + 1)
    return x.mul(at::Scalar(2.0f)).add(at::Scalar(1.0f)).relu();
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    auto ac = a.contiguous(); auto bc = b.contiguous();
    int64_t n = ac.numel();
    const float* ap = ac.data_ptr<float>();
    const float* bp = bc.data_ptr<float>();
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::fabs(double(ap[i]) - double(bp[i]));
        if (d > m) m = d;
    }
    return m;
}

int main() {
    // Build a 4x4 input.
    Tensor x = at::randn({4, 4});

    // Trace + compile.
    auto compiled = compile(
        [](TracedTensor x) -> TracedTensor {
            auto a = traced_mul(x, 2.0f);   // mul_s
            auto b = traced_add(a, 1.0f);   // add_s
            auto c = traced_relu(b);        // relu
            return c;
        },
        x
    );

    std::printf("[trace] raw ops: %zu  fused ops: %zu\n",
                compiled.raw_trace_len(), compiled.trace_len());

    Tensor y_eager = eager_f(x);
    Tensor y_jit   = compiled(x);

    double diff = max_abs_diff(y_eager, y_jit);
    std::printf("[verify] max |y_eager - y_jit| = %.3e\n", diff);
    if (diff > 1e-5) {
        std::printf("FAIL: outputs differ\n");
        return 1;
    }

    // Latency benchmark.
    const int warmup = 50;
    const int iters  = 5000;
    for (int i = 0; i < warmup; ++i) (void)eager_f(x);
    for (int i = 0; i < warmup; ++i) (void)compiled(x);

    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) (void)eager_f(x);
    double t_eager = (now_ms() - t0) / iters;

    t0 = now_ms();
    for (int i = 0; i < iters; ++i) (void)compiled(x);
    double t_jit = (now_ms() - t0) / iters;

    std::printf("[bench] eager: %.4f ms/call    jit: %.4f ms/call    "
                "speedup: %.2fx\n", t_eager, t_jit, t_eager / t_jit);

    // Try a larger tensor too — fusion savings grow with size.
    Tensor xb = at::randn({1024, 1024});
    auto compiled_big = compile(
        [](TracedTensor x) -> TracedTensor {
            auto a = traced_mul(x, 2.0f);
            auto b = traced_add(a, 1.0f);
            auto c = traced_relu(b);
            return c;
        },
        xb);
    std::printf("[trace 1024x1024] raw=%zu fused=%zu\n",
                compiled_big.raw_trace_len(), compiled_big.trace_len());

    auto big_eager = [&]() {
        return xb.mul(at::Scalar(2.0f)).add(at::Scalar(1.0f)).relu();
    };
    for (int i = 0; i < 10; ++i) { (void)big_eager(); (void)compiled_big(xb); }
    t0 = now_ms();
    const int big_iters = 200;
    for (int i = 0; i < big_iters; ++i) (void)big_eager();
    double tb_eager = (now_ms() - t0) / big_iters;
    t0 = now_ms();
    for (int i = 0; i < big_iters; ++i) (void)compiled_big(xb);
    double tb_jit = (now_ms() - t0) / big_iters;
    std::printf("[bench 1024x1024] eager: %.3f ms  jit: %.3f ms  speedup: "
                "%.2fx\n", tb_eager, tb_jit, tb_eager / tb_jit);

    // --- Codegen-only path verification ----------------------------------
    // Force interpreter for baseline measurement, then force codegen.
    {
        // Verify codegen emits correct results.
        Tensor y_cg = compiled_big(xb);
        Tensor y_eg = xb.mul(at::Scalar(2.0f)).add(at::Scalar(1.0f)).relu();
        double d = max_abs_diff(y_cg, y_eg);
        std::printf("[codegen verify 1024x1024] max|cg-eager| = %.3e\n", d);
        if (d > 1e-5) { std::printf("FAIL: codegen numeric mismatch\n");
                        return 2; }
    }
    // Report whether codegen actually kicked in:
    bool cg_used = false;
    for (auto s : compiled_big.codegen_state) if (s == 1) { cg_used = true;
                                                            break; }
    std::printf("[codegen status] hook=%p  used=%s\n",
                (void*)torch::jit::codegen_hook(), cg_used ? "yes" : "no");

    // Bench interpreter-only path (env PROMETORCH_JIT=0 not possible
    // mid-process, so measure via a fresh compile with threshold above n).
    {
        // Clone by re-compiling the same trace and bumping threshold so
        // codegen is skipped.
        auto interp_only = compile(
            [](TracedTensor x) -> TracedTensor {
                auto a = traced_mul(x, 2.0f);
                auto b = traced_add(a, 1.0f);
                auto c = traced_relu(b);
                return c;
            }, xb);
        // manually mark all records as codegen-failed (state=0)
        interp_only.codegen_state.assign(interp_only.program.size(),
                                         int8_t(0));
        interp_only.codegen_kernels.assign(interp_only.program.size(),
                                           nullptr);
        for (int i = 0; i < 20; ++i) (void)interp_only(xb);
        t0 = now_ms();
        for (int i = 0; i < big_iters; ++i) (void)interp_only(xb);
        double tb_interp = (now_ms() - t0) / big_iters;
        std::printf("[bench 1024x1024 interp-only] %.3f ms  "
                    "codegen speedup vs interp: %.2fx\n",
                    tb_interp, tb_interp / tb_jit);
    }

    // Longer chain — fusion benefit compounds (more eager allocations saved).
    Tensor xc = at::randn({64, 64});
    auto compiled_long = compile(
        [](TracedTensor x) -> TracedTensor {
            auto a = traced_mul(x, 2.0f);
            auto b = traced_add(a, 1.0f);
            auto c = traced_relu(b);
            auto d = traced_mul(c, 0.5f);
            auto e = traced_add(d, -0.25f);
            auto f = traced_sigmoid(e);
            auto g = traced_mul(f, 3.0f);
            return g;
        }, xc);
    std::printf("[trace long chain] raw=%zu fused=%zu\n",
                compiled_long.raw_trace_len(), compiled_long.trace_len());
    auto long_eager = [&]() {
        return xc.mul(at::Scalar(2.0f)).add(at::Scalar(1.0f)).relu()
                 .mul(at::Scalar(0.5f)).add(at::Scalar(-0.25f))
                 .sigmoid().mul(at::Scalar(3.0f));
    };
    for (int i = 0; i < 50; ++i) { (void)long_eager(); (void)compiled_long(xc); }
    t0 = now_ms();
    for (int i = 0; i < 2000; ++i) (void)long_eager();
    double tl_eager = (now_ms() - t0) / 2000;
    t0 = now_ms();
    for (int i = 0; i < 2000; ++i) (void)compiled_long(xc);
    double tl_jit = (now_ms() - t0) / 2000;
    std::printf("[bench long chain 64x64] eager: %.4f ms  jit: %.4f ms  "
                "speedup: %.2fx\n", tl_eager, tl_jit, tl_eager / tl_jit);

    std::printf("PASS\n");
    return 0;
}
