// ============================================================================
// test_param_groups.cpp — unit tests for ParamGroup support in optimizers.
//
// Covers:
//   1. Adam with two ParamGroups (lr=1e-4 and lr=1e-3) — slow group's params
//      move LESS than fast group's params after a single step on the same grad.
//   2. add_param_group() builds groups incrementally, post-construction.
//   3. Per-group hyperparameter overrides (eps, betas) are honored.
//   4. LR scheduler scales BOTH groups by the same factor (preserving ratio).
//   5. step(group_idx) advances only one group's lr, leaves others alone.
//   6. Backward-compat: old single-vector ctor still works and exposes one
//      default group via param_groups().
//
// Build: link against aten_cpu + torch optim.
// ============================================================================

#include "torch/nn/parameter.h"
#include "torch/optim/optim.h"
#include "torch/optim/lr_scheduler.h"
#include "aten/src/ATen/ATen.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>

using at::Tensor;
using torch::nn::Parameter;
using torch::optim::Adam;
using torch::optim::AdamOptions;
using torch::optim::ParamGroup;
using torch::optim::SGD;
using torch::optim::SGDOptions;
using torch::optim::StepLR;

static int failed = 0;
static int passed = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }              \
    else      { ++failed; std::printf("  FAIL: %s\n", msg); }              \
} while (0)

static bool close(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

// Build a fresh Parameter holding a single-element float tensor.
static std::unique_ptr<Parameter> make_param(float init_value) {
    Tensor t = at::ones({1});
    t.mutable_data_ptr<float>()[0] = init_value;
    return std::make_unique<Parameter>(t);
}

// ============================================================================

int main() {
    std::printf("=== param_groups self-test ===\n");

    // ------------------------------------------------------------------------
    // Test 1: Adam with two groups, different lrs => different update magnitudes
    // ------------------------------------------------------------------------
    {
        std::printf("\n[Test 1] Adam with 2 groups (lr=1e-4 vs lr=1e-3):\n");

        auto p_slow = make_param(1.0f);
        auto p_fast = make_param(1.0f);

        std::vector<ParamGroup> groups;
        groups.push_back(ParamGroup({p_slow.get()}, /*lr=*/1e-4, "slow"));
        groups.push_back(ParamGroup({p_fast.get()}, /*lr=*/1e-3, "fast"));

        Adam opt(std::move(groups));

        CHECK(opt.param_groups().size() == 2, "two groups installed");
        CHECK(close(opt.param_groups()[0].lr, 1e-4),  "group[0].lr is 1e-4");
        CHECK(close(opt.param_groups()[1].lr, 1e-3),  "group[1].lr is 1e-3");
        CHECK(opt.param_groups()[0].name == "slow",   "group[0] name preserved");
        CHECK(opt.param_groups()[1].name == "fast",   "group[1] name preserved");

        // Same gradient for both params.
        Tensor g_slow = at::ones({1});
        Tensor g_fast = at::ones({1});
        p_slow->set_grad(g_slow);
        p_fast->set_grad(g_fast);

        opt.step();

        float w_slow = p_slow->data().data_ptr<float>()[0];
        float w_fast = p_fast->data().data_ptr<float>()[0];
        float dslow = std::fabs(w_slow - 1.0f);
        float dfast = std::fabs(w_fast - 1.0f);

        std::printf("  slow w: 1.0 -> %.8f (delta %.8f)\n", w_slow, dslow);
        std::printf("  fast w: 1.0 -> %.8f (delta %.8f)\n", w_fast, dfast);

        // Adam first step with grad=1: |delta| ~= lr (because m_hat=v_hat=1).
        CHECK(dslow < dfast, "slow group moved less than fast group");
        CHECK(dfast > 5.0f * dslow,
              "fast group's step is ~10x larger than slow group's step");
        CHECK(close((double)dslow, 1e-4, 1e-6), "slow group |delta| ~= 1e-4");
        CHECK(close((double)dfast, 1e-3, 1e-5), "fast group |delta| ~= 1e-3");
    }

    // ------------------------------------------------------------------------
    // Test 2: add_param_group() builds groups incrementally
    // ------------------------------------------------------------------------
    {
        std::printf("\n[Test 2] add_param_group() incremental build:\n");

        auto p1 = make_param(0.5f);
        auto p2 = make_param(0.5f);

        // Construct empty Adam, then add groups one-by-one.
        Adam opt(std::vector<ParamGroup>{});
        CHECK(opt.param_groups().empty(), "starts with zero groups");

        opt.add_param_group(ParamGroup({p1.get()}, /*lr=*/0.01, "head"));
        opt.add_param_group(ParamGroup({p2.get()}, /*lr=*/0.001, "backbone"));
        CHECK(opt.param_groups().size() == 2,        "two groups after add");
        CHECK(opt.param_groups()[0].name == "head",  "first added is 'head'");
        CHECK(opt.param_groups()[1].name == "backbone", "second is 'backbone'");

        p1->set_grad(at::ones({1}));
        p2->set_grad(at::ones({1}));
        opt.step();  // should not crash; both groups update with their own lr

        float w1 = p1->data().data_ptr<float>()[0];
        float w2 = p2->data().data_ptr<float>()[0];
        CHECK(std::fabs(w1 - 0.5f) > std::fabs(w2 - 0.5f),
              "group with lr=0.01 moves more than group with lr=0.001");
    }

    // ------------------------------------------------------------------------
    // Test 3: Per-group hyperparameter overrides (eps + betas)
    // ------------------------------------------------------------------------
    {
        std::printf("\n[Test 3] per-group eps + betas overrides:\n");

        auto p_default = make_param(0.0f);
        auto p_custom  = make_param(0.0f);

        ParamGroup g_default({p_default.get()}, 1e-3);  // inherit eps/betas
        ParamGroup g_custom ({p_custom.get()},  1e-3);
        // Force a much larger eps in the custom group → smaller step magnitude.
        g_custom.eps = 1.0;

        std::vector<ParamGroup> groups;
        groups.push_back(g_default);
        groups.push_back(g_custom);

        AdamOptions opts(1e-3);
        opts.eps_(1e-8);
        Adam opt(std::move(groups), opts);

        Tensor grad = at::ones({1});
        p_default->set_grad(grad.clone());
        p_custom ->set_grad(grad.clone());
        opt.step();

        float wd = p_default->data().data_ptr<float>()[0];
        float wc = p_custom ->data().data_ptr<float>()[0];
        std::printf("  default-eps step: %.8f\n", wd);
        std::printf("  custom-eps  step: %.8f\n", wc);
        // With grad=1, default eps is tiny → |Δ| ≈ lr; custom eps=1 makes
        // denom ≈ 2 → |Δ| ≈ lr/2.  Custom must be strictly smaller in magnitude.
        CHECK(std::fabs(wc) < std::fabs(wd),
              "larger per-group eps -> smaller param update");
    }

    // ------------------------------------------------------------------------
    // Test 4: LR scheduler scales both groups by the same factor
    // ------------------------------------------------------------------------
    {
        std::printf("\n[Test 4] StepLR scales both groups proportionally:\n");

        auto p_a = make_param(0.0f);
        auto p_b = make_param(0.0f);
        std::vector<ParamGroup> groups;
        groups.push_back(ParamGroup({p_a.get()}, /*lr=*/0.1));
        groups.push_back(ParamGroup({p_b.get()}, /*lr=*/0.01));
        SGD opt(std::move(groups), SGDOptions(0.1));

        StepLR sched(opt, /*step_size=*/1, /*gamma=*/0.5);

        double lr0_before = opt.param_groups()[0].lr;
        double lr1_before = opt.param_groups()[1].lr;
        sched.step();  // last_epoch_ = 0  -> factor = 1.0 (no change yet)
        sched.step();  // last_epoch_ = 1  -> factor = 0.5
        double lr0_after = opt.param_groups()[0].lr;
        double lr1_after = opt.param_groups()[1].lr;

        std::printf("  group 0: %.6f -> %.6f\n", lr0_before, lr0_after);
        std::printf("  group 1: %.6f -> %.6f\n", lr1_before, lr1_after);

        CHECK(close(lr0_after, lr0_before * 0.5, 1e-9),
              "group 0 lr halved");
        CHECK(close(lr1_after, lr1_before * 0.5, 1e-9),
              "group 1 lr halved");
        CHECK(close(lr0_after / lr1_after, lr0_before / lr1_before, 1e-9),
              "ratio between groups preserved");
    }

    // ------------------------------------------------------------------------
    // Test 5: scheduler.step(group_idx) advances only one group
    // ------------------------------------------------------------------------
    {
        std::printf("\n[Test 5] step(group_idx) only touches that group:\n");

        auto p_a = make_param(0.0f);
        auto p_b = make_param(0.0f);
        std::vector<ParamGroup> groups;
        groups.push_back(ParamGroup({p_a.get()}, /*lr=*/0.1));
        groups.push_back(ParamGroup({p_b.get()}, /*lr=*/0.1));
        SGD opt(std::move(groups), SGDOptions(0.1));

        StepLR sched(opt, /*step_size=*/1, /*gamma=*/0.5);

        // Advance schedule + write back only to group 1.
        sched.step((size_t)1);  // last_epoch -> 0  (factor 1.0)
        sched.step((size_t)1);  // last_epoch -> 1  (factor 0.5)

        double lr0 = opt.param_groups()[0].lr;
        double lr1 = opt.param_groups()[1].lr;
        std::printf("  group 0 lr (untouched): %.6f\n", lr0);
        std::printf("  group 1 lr (scheduled): %.6f\n", lr1);
        CHECK(close(lr0, 0.1, 1e-9),  "group 0 lr unchanged");
        CHECK(close(lr1, 0.05, 1e-9), "group 1 lr halved");
    }

    // ------------------------------------------------------------------------
    // Test 6: backward compat — single-vector ctor still works
    // ------------------------------------------------------------------------
    {
        std::printf("\n[Test 6] backward-compat single-vector ctor:\n");

        auto p = make_param(1.0f);
        Adam opt({p.get()}, /*lr=*/0.01);

        CHECK(opt.param_groups().size() == 1,
              "single-vector ctor produces exactly one default group");
        CHECK(close(opt.param_groups()[0].lr, 0.01),
              "default group's lr matches ctor argument");

        p->set_grad(at::ones({1}));
        opt.step();  // must not crash
        float w = p->data().data_ptr<float>()[0];
        CHECK(std::fabs(w - 1.0f) > 0.0f, "single-group step actually moved param");
    }

    // ========================================================================
    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
