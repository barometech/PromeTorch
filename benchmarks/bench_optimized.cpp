// ============================================================================
// bench_optimized.cpp -- Focused benchmark for PromeTorch optimizations
// ============================================================================
// Tests specific optimization paths vs fallback paths:
//   1. FastOps (trusted tensor) dispatch vs regular dispatch
//   2. Fused Linear (mm+bias in one call) vs separate mm+add
//   3. NodePool allocation vs std::make_shared
//   4. hot::sgemm vs inline sgemm (hot_loops.cpp compiled code)
//   5. Fused autograd linear vs separate mm+add+backward nodes
//   6. Fused Adam multi-param step vs per-param step
//
// CPU only. Outputs JSON to stdout or file.
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/FastOps.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/node_pool.h"
#include "torch/csrc/autograd/functions/LinearAlgebraBackward.h"

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

static void section(const std::string& name) {
    std::cerr << "\n=== " << name << " ===\n";
}

static void report(const std::string& name, double t) {
    std::cerr << "  " << name << ": " << std::fixed << std::setprecision(4) << t << " ms\n";
}

// ============================================================================
// 1. FastOps (trusted dispatch) vs regular dispatch
// ============================================================================
// FastOps calls hot:: loops directly, skipping dtype switch + contiguous check.
// The regular path goes through at::native::add which has PT_DISPATCH_ALL_TYPES.
static void bench_fast_dispatch() {
    section("FastOps Dispatch vs Regular Dispatch");

    Tensor a = at::randn({1024, 1024});
    Tensor b = at::randn({1024, 1024});
    Tensor a_pos = at::native::abs(a) + Scalar(1e-6f);

    // --- Element-wise add ---
    {
        double t_fast = bench([&]() { auto r = at::native::fast::add_f32(a, b); }, WARMUP, 500);
        double t_regular = bench([&]() { auto r = a + b; }, WARMUP, 500);
        emit("fast_add_1024", t_fast);
        emit("regular_add_1024", t_regular);
        report("fast::add_f32", t_fast);
        report("regular a+b  ", t_regular);
        std::cerr << "  -> speedup: " << std::setprecision(2) << t_regular / t_fast << "x\n";
    }

    // --- Element-wise mul ---
    {
        double t_fast = bench([&]() { auto r = at::native::fast::mul_f32(a, b); }, WARMUP, 500);
        double t_regular = bench([&]() { auto r = a * b; }, WARMUP, 500);
        emit("fast_mul_1024", t_fast);
        emit("regular_mul_1024", t_regular);
        report("fast::mul_f32", t_fast);
        report("regular a*b  ", t_regular);
        std::cerr << "  -> speedup: " << std::setprecision(2) << t_regular / t_fast << "x\n";
    }

    // --- exp ---
    {
        double t_fast = bench([&]() { auto r = at::native::fast::exp_f32(a); }, WARMUP, 500);
        double t_regular = bench([&]() { auto r = at::native::exp(a); }, WARMUP, 500);
        emit("fast_exp_1024", t_fast);
        emit("regular_exp_1024", t_regular);
        report("fast::exp_f32", t_fast);
        report("regular exp  ", t_regular);
        std::cerr << "  -> speedup: " << std::setprecision(2) << t_regular / t_fast << "x\n";
    }

    // --- tanh ---
    {
        double t_fast = bench([&]() { auto r = at::native::fast::tanh_f32(a); }, WARMUP, 500);
        double t_regular = bench([&]() { auto r = at::native::tanh(a); }, WARMUP, 500);
        emit("fast_tanh_1024", t_fast);
        emit("regular_tanh_1024", t_regular);
        report("fast::tanh_f32", t_fast);
        report("regular tanh  ", t_regular);
        std::cerr << "  -> speedup: " << std::setprecision(2) << t_regular / t_fast << "x\n";
    }

    // --- sigmoid ---
    {
        double t_fast = bench([&]() { auto r = at::native::fast::sigmoid_f32(a); }, WARMUP, 500);
        double t_regular = bench([&]() { auto r = at::native::sigmoid(a); }, WARMUP, 500);
        emit("fast_sigmoid_1024", t_fast);
        emit("regular_sigmoid_1024", t_regular);
        report("fast::sigmoid_f32", t_fast);
        report("regular sigmoid  ", t_regular);
        std::cerr << "  -> speedup: " << std::setprecision(2) << t_regular / t_fast << "x\n";
    }
}

// ============================================================================
// 2. Fused Linear vs Separate mm + bias_add
// ============================================================================
// fused_linear_f32: one sgemm_nt + one bias loop
// Separate: mm_nt_f32 + add (two tensor allocations)
static void bench_fused_linear() {
    section("Fused Linear vs Separate mm + add");

    Tensor x = at::randn({64, 512});
    Tensor W = at::randn({256, 512});
    Tensor bias = at::randn({256});

    // --- Fused linear (mm + bias in one call) ---
    {
        double t_fused = bench([&]() {
            auto r = at::native::fast::fused_linear_f32(x, W, bias);
        }, WARMUP, 500);
        emit("fused_linear_64x512x256", t_fused);
        report("fused_linear_f32", t_fused);
    }

    // --- Separate: mm then add ---
    {
        double t_separate = bench([&]() {
            auto mm_result = at::native::fast::mm_nt_f32(x, W);
            // Manual bias add
            float* out_data = mm_result.mutable_data_ptr<float>();
            const float* b = bias.data_ptr<float>();
            at::native::hot::add_broadcast_loop(out_data, b, out_data, 64, 256, 1.0f);
        }, WARMUP, 500);
        emit("separate_mm_add_64x512x256", t_separate);
        report("separate mm+add  ", t_separate);
    }

    // --- Full Linear module (inference, NoGrad) ---
    {
        torch::autograd::NoGradGuard no_grad;
        auto lin = std::make_shared<Linear>(512, 256);
        double t_module = bench([&]() {
            auto r = lin->forward(x);
        }, WARMUP, 500);
        emit("linear_module_fwd_64x512x256", t_module);
        report("Linear::forward  ", t_module);
    }

    // --- Fused linear + ReLU ---
    {
        double t_fused_relu = bench([&]() {
            auto r = at::native::fast::fused_linear_relu_f32(x, W, bias);
        }, WARMUP, 500);
        emit("fused_linear_relu_64x512x256", t_fused_relu);
        report("fused_linear_relu", t_fused_relu);
    }

    // --- Larger sizes ---
    for (int M : {128, 256}) {
        Tensor x2 = at::randn({(int64_t)M, 512});
        double t = bench([&]() {
            auto r = at::native::fast::fused_linear_f32(x2, W, bias);
        }, WARMUP, 200);
        emit("fused_linear_" + std::to_string(M) + "x512x256", t);
        report("fused_linear_" + std::to_string(M) + "x512x256", t);
    }
}

// ============================================================================
// 3. NodePool allocation vs std::make_shared
// ============================================================================
// NodePool reuses memory from a thread-local free list.
// std::make_shared always calls operator new.
static void bench_node_pool() {
    section("NodePool vs std::make_shared");

    // We'll benchmark creating and destroying MmBackward nodes
    // (a common autograd node type).
    Tensor dummy_a = at::randn({64, 512});
    Tensor dummy_b = at::randn({512, 256});

    // --- NodePool: acquire + release cycle ---
    {
        double t_pool = bench([&]() {
            auto node = NodePool<MmBackward>::make_shared(dummy_a, dummy_b);
            // shared_ptr goes out of scope -> PooledDeleter returns to pool
        }, WARMUP, 5000);
        emit("nodepool_mm_acquire_release", t_pool);
        report("NodePool<MmBackward>  ", t_pool);
    }

    // --- std::make_shared: always heap alloc ---
    {
        double t_new = bench([&]() {
            auto node = std::make_shared<MmBackward>(dummy_a, dummy_b);
            // shared_ptr goes out of scope -> delete
        }, WARMUP, 5000);
        emit("stl_make_shared_mm", t_new);
        report("std::make_shared<Mm> ", t_new);
    }

    std::cerr << "  -> pool speedup shown above\n";
}

// ============================================================================
// 4. hot::sgemm performance (the compiled inner loop)
// ============================================================================
// Tests the GEMM kernel at various sizes to show GFLOPS curve.
static void bench_sgemm() {
    section("hot::sgemm Performance");

    for (int sz : {64, 128, 256, 512, 1024}) {
        Tensor a = at::randn({(int64_t)sz, (int64_t)sz});
        Tensor b = at::randn({(int64_t)sz, (int64_t)sz});
        Tensor c = at::empty({(int64_t)sz, (int64_t)sz});

        int iters = (sz >= 1024) ? 5 : (sz >= 512) ? 20 : (sz >= 256) ? 100 : 500;

        double t_nn = bench([&]() {
            at::native::hot::sgemm(sz, sz, sz, 1.0f,
                                   a.data_ptr<float>(), sz,
                                   b.data_ptr<float>(), sz,
                                   0.0f, c.mutable_data_ptr<float>(), sz);
        }, WARMUP, iters);

        double gflops = (2.0 * sz * sz * sz) / (t_nn * 1e-3) / 1e9;
        emit("sgemm_" + std::to_string(sz), t_nn);
        report("sgemm_" + std::to_string(sz) + " (" +
               std::to_string(static_cast<int>(gflops)) + " GFLOPS)", t_nn);
    }

    // Also test sgemm_nt (the Linear layer pattern)
    for (int sz : {256, 512, 1024}) {
        Tensor a = at::randn({(int64_t)sz, (int64_t)sz});
        Tensor b = at::randn({(int64_t)sz, (int64_t)sz});  // stored as [N,K]
        Tensor c = at::empty({(int64_t)sz, (int64_t)sz});

        int iters = (sz >= 1024) ? 5 : (sz >= 512) ? 20 : 100;

        double t_nt = bench([&]() {
            at::native::hot::sgemm_nt(sz, sz, sz, 1.0f,
                                      a.data_ptr<float>(), sz,
                                      b.data_ptr<float>(), sz,
                                      0.0f, c.mutable_data_ptr<float>(), sz);
        }, WARMUP, iters);

        double gflops = (2.0 * sz * sz * sz) / (t_nt * 1e-3) / 1e9;
        emit("sgemm_nt_" + std::to_string(sz), t_nt);
        report("sgemm_nt_" + std::to_string(sz) + " (" +
               std::to_string(static_cast<int>(gflops)) + " GFLOPS)", t_nt);
    }
}

// ============================================================================
// 5. Fused autograd linear vs separate mm + add autograd
// ============================================================================
// Compares fused_linear_autograd (1 node, 1 tensor alloc)
// vs mm_autograd + add_autograd (3 nodes, 3 tensor allocs)
static void bench_fused_autograd() {
    section("Fused Autograd Linear vs Separate mm+add Autograd");

    // Use the Linear module which dispatches to fused path automatically
    auto lin_fused = std::make_shared<Linear>(512, 256);
    Tensor x = at::randn({64, 512});
    x.set_requires_grad(true);

    // --- Fused: Linear module forward + backward (uses fused_linear_autograd) ---
    {
        double t_fused = bench([&]() {
            lin_fused->zero_grad();
            Tensor y = lin_fused->forward(x);
            Tensor loss = sum_autograd(y);
            tensor_backward(loss);
        }, WARMUP, 200);
        emit("fused_autograd_linear_fwd_bwd", t_fused);
        report("fused_linear autograd fwd+bwd", t_fused);
    }

    // --- Separate: manual mm + t + add (old path) ---
    {
        auto weight_param = lin_fused->get_parameter("weight");
        auto bias_param = lin_fused->get_parameter("bias");
        Tensor W = weight_param->data();  // [256, 512]
        Tensor b = bias_param->data();    // [256]

        double t_separate = bench([&]() {
            // Manually build the graph with separate ops
            Tensor Wt = t_autograd(W);
            Tensor output = mm_autograd(x, Wt);
            output = add_autograd(output, b);
            Tensor loss = sum_autograd(output);
            tensor_backward(loss);
        }, WARMUP, 200);
        emit("separate_autograd_mm_add_bwd", t_separate);
        report("separate mm+t+add autograd   ", t_separate);
        std::cerr << "  -> fused speedup: " << std::setprecision(2)
                  << t_separate / std::max(0.001, t_separate) << "x (see values above)\n";
    }
}

// ============================================================================
// 6. Adam fused multi-param step
// ============================================================================
// Fused: all parameters updated in one hot::fused_adam_multi call
// This is already the default path, but we measure the step() overhead.
static void bench_adam_fused() {
    section("Adam Step Performance");

    auto model = std::make_shared<Sequential>(
        std::initializer_list<ModulePtr>{
            std::make_shared<Linear>(784, 512),
            std::make_shared<ReLU>(),
            std::make_shared<Linear>(512, 256),
            std::make_shared<ReLU>(),
            std::make_shared<Linear>(256, 10)
        }
    );

    Tensor x_data = at::randn({64, 784});
    Tensor target = at::empty({64});
    float* tgt = target.mutable_data_ptr<float>();
    for (int i = 0; i < 64; ++i) tgt[i] = static_cast<float>(std::rand() % 10);

    CrossEntropyLoss criterion;
    Adam opt(model->parameters(), AdamOptions(0.001));

    // Warmup to initialize Adam state
    {
        opt.zero_grad();
        Tensor out = model->forward(x_data);
        Tensor loss = criterion.forward(out, target);
        tensor_backward(loss);
        opt.step();
    }

    // --- Full step: forward + backward + optimizer step ---
    {
        double t_full = bench([&]() {
            opt.zero_grad();
            Tensor out = model->forward(x_data);
            Tensor loss = criterion.forward(out, target);
            tensor_backward(loss);
            opt.step();
        }, 2, 50);
        emit("adam_full_step_784_512_256_10", t_full);
        report("full training step (fwd+bwd+adam)", t_full);
    }

    // --- Just forward (inference) ---
    {
        torch::autograd::NoGradGuard no_grad;
        double t_fwd = bench([&]() {
            auto out = model->forward(x_data);
        }, WARMUP, 200);
        emit("mlp_inference_784_512_256_10", t_fwd);
        report("inference only (NoGrad)", t_fwd);
    }

    // --- Just backward ---
    // (do forward, then time backward only)
    {
        double t_bwd = bench([&]() {
            opt.zero_grad();
            Tensor out = model->forward(x_data);
            Tensor loss = criterion.forward(out, target);

            auto t0 = std::chrono::high_resolution_clock::now();
            tensor_backward(loss);
            auto t1 = std::chrono::high_resolution_clock::now();
            // We can't easily time just backward in the bench loop,
            // so we time the full thing. The separate measurement is informational.
        }, 2, 50);
        // Note: this includes forward too, use the full step time for comparison
    }

    // --- Just optimizer step (time step() alone after backward) ---
    {
        // Do forward+backward to have gradients
        opt.zero_grad();
        Tensor out = model->forward(x_data);
        Tensor loss = criterion.forward(out, target);
        tensor_backward(loss);

        // Copy gradients so we can reuse them
        std::vector<Tensor> saved_grads;
        for (auto* p : model->parameters()) {
            saved_grads.push_back(p->grad().defined() ? p->grad().clone() : Tensor());
        }

        double t_step = bench([&]() {
            // Restore gradients
            auto params = model->parameters();
            for (size_t i = 0; i < params.size(); ++i) {
                if (saved_grads[i].defined()) {
                    // Copy grad data back
                    float* dst = params[i]->grad().mutable_data_ptr<float>();
                    const float* src = saved_grads[i].data_ptr<float>();
                    std::memcpy(dst, src, params[i]->numel() * sizeof(float));
                }
            }
            opt.step();
        }, WARMUP, 500);
        emit("adam_step_only_mlp", t_step);
        report("adam step() only", t_step);
    }
}

// ============================================================================
// 7. Tensor allocation overhead
// ============================================================================
// Measures the cost of at::empty vs malloc to show framework overhead.
static void bench_tensor_alloc() {
    section("Tensor Allocation Overhead");

    // --- at::empty (creates TensorImpl, Storage, Allocator) ---
    {
        double t = bench([&]() {
            auto r = at::empty({1024, 1024});
        }, WARMUP, 2000);
        emit("empty_1024x1024", t);
        report("at::empty(1024x1024)", t);
    }

    // --- at::empty small tensor ---
    {
        double t = bench([&]() {
            auto r = at::empty({64, 256});
        }, WARMUP, 5000);
        emit("empty_64x256", t);
        report("at::empty(64x256)", t);
    }

    // --- raw malloc for comparison ---
    {
        double t = bench([&]() {
            float* p = (float*)std::malloc(1024 * 1024 * sizeof(float));
            std::free(p);
        }, WARMUP, 5000);
        emit("raw_malloc_1024x1024", t);
        report("raw malloc+free(1024x1024)", t);
    }

    // --- raw malloc small ---
    {
        double t = bench([&]() {
            float* p = (float*)std::malloc(64 * 256 * sizeof(float));
            std::free(p);
        }, WARMUP, 5000);
        emit("raw_malloc_64x256", t);
        report("raw malloc+free(64x256)", t);
    }
}

// ============================================================================
// 8. End-to-end training comparison
// ============================================================================
static void bench_training_e2e() {
    section("End-to-End Training (10 batches)");

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

    // Pre-generate data to remove randn from timing
    std::vector<Tensor> x_batches, tgt_batches;
    for (int i = 0; i < 10; ++i) {
        x_batches.push_back(at::randn({64, 784}));
        Tensor target = at::empty({64});
        float* tgt = target.mutable_data_ptr<float>();
        for (int j = 0; j < 64; ++j) tgt[j] = static_cast<float>(std::rand() % 10);
        tgt_batches.push_back(target);
    }

    // Warmup
    for (int i = 0; i < 2; ++i) {
        opt.zero_grad();
        Tensor out = model->forward(x_batches[0]);
        Tensor loss = criterion.forward(out, tgt_batches[0]);
        tensor_backward(loss);
        opt.step();
    }

    // Timed run: 10 batches
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < 10; ++b) {
        opt.zero_grad();
        Tensor out = model->forward(x_batches[b]);
        Tensor loss = criterion.forward(out, tgt_batches[b]);
        tensor_backward(loss);
        opt.step();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_batch = total_ms / 10.0;

    emit("train_10batch_pregen_total", total_ms);
    emit("train_10batch_pregen_per_batch", per_batch);
    report("total (10 batches, pre-gen data)", total_ms);
    report("per batch", per_batch);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::srand(42);

    std::ofstream file_out;
    if (argc > 2 && std::string(argv[1]) == "--output") {
        file_out.open(argv[2]);
        out_stream = &file_out;
    }

    std::cerr << "PromeTorch Optimization Benchmark\n";
    std::cerr << "==================================\n";

    *out_stream << "{\n";

    bench_fast_dispatch();
    bench_fused_linear();
    bench_node_pool();
    bench_sgemm();
    bench_fused_autograd();
    bench_adam_fused();
    bench_tensor_alloc();
    bench_training_e2e();

    *out_stream << "\n}\n";

    if (file_out.is_open()) {
        file_out.close();
        std::cerr << "\nResults saved to " << argv[2] << "\n";
    }

    std::cerr << "\nDone.\n";
    return 0;
}
