// ============================================================================
// PromeTorch Optimizer Tests - Comprehensive tests for all 9 optimizers
// ============================================================================
// Tests: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax
//
// Each optimizer is tested for:
//   1. Convergence on a quadratic objective f(x) = (x - 3)^2
//   2. State initialization (step count, internal buffers)
//   3. Weight decay effect
//   4. zero_grad correctness

#include "torch/optim/optim.h"
#include "torch/nn/parameter.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace torch;
using namespace torch::optim;
using at::Tensor;

// ============================================================================
// Helper: Manually compute gradient of f(x) = (x - target)^2
// gradient = 2 * (x - target)
// ============================================================================

static void set_quadratic_gradient(nn::Parameter& param, float target) {
    Tensor grad = (param.data() - at::scalar_tensor(at::Scalar(target)))
                      .mul(at::Scalar(2.0f));
    param.set_grad(grad.clone());
}

// ============================================================================
// SGD Tests
// ============================================================================

TEST(OptimizersSGD, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersSGD, StateNotCreatedWithoutMomentum) {
    // SGD without momentum does not create state
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    // Without momentum, no state should be created
    auto* state = optimizer.get_state<SGDParamState>(&param);
    EXPECT_EQ(state, nullptr);
}

TEST(OptimizersSGD, MomentumBufferCreated) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    SGDOptions opts(0.1);
    opts.momentum_(0.9);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<SGDParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(state->momentum_buffer.defined());
}

TEST(OptimizersSGD, WeightDecay) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    SGDOptions opts(0.1);
    opts.weight_decay_(0.01);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // Set zero gradient to isolate weight decay effect
    param.set_grad(at::zeros({1}));
    optimizer.step();

    // x_new = x - lr * (grad + wd * x) = 1.0 - 0.1 * (0 + 0.01 * 1.0) = 0.999
    EXPECT_NEAR(param.data().data_ptr<float>()[0], 0.999f, 1e-6f);
}

TEST(OptimizersSGD, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// Adam Tests
// ============================================================================

TEST(OptimizersAdam, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersAdam, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<AdamParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->exp_avg.defined());
    EXPECT_TRUE(state->exp_avg_sq.defined());
}

TEST(OptimizersAdam, WeightDecay) {
    Tensor x = at::full({1}, 2.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamOptions opts(0.1);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    // Should still converge, but weight decay pulls toward 0
    // The equilibrium shifts slightly from 3.0 toward 0
    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersAdam, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    AdamOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// AdamW Tests
// ============================================================================

TEST(OptimizersAdamW, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamWOptions opts(0.1);
    opts.weight_decay_(0.0);  // Disable weight decay for clean convergence test
    std::vector<nn::Parameter*> params = {&param};
    AdamW optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersAdamW, DecoupledWeightDecay) {
    // Verify decoupled weight decay: with zero grad, only weight decay acts
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamWOptions opts(0.1);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    AdamW optimizer(params, opts);

    param.set_grad(at::zeros({1}));
    optimizer.step();

    // Decoupled: x_new = x * (1 - lr * wd) = 1.0 * (1 - 0.1*0.1) = 0.99
    EXPECT_NEAR(param.data().data_ptr<float>()[0], 0.99f, 1e-5f);
}

TEST(OptimizersAdamW, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamWOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    AdamW optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<AdamParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->exp_avg.defined());
    EXPECT_TRUE(state->exp_avg_sq.defined());
}

TEST(OptimizersAdamW, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    AdamWOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    AdamW optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// RMSprop Tests
// ============================================================================

TEST(OptimizersRMSprop, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RMSpropOptions opts(0.05);
    std::vector<nn::Parameter*> params = {&param};
    RMSprop optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersRMSprop, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RMSpropOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    RMSprop optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<RMSpropParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(state->square_avg.defined());
}

TEST(OptimizersRMSprop, WeightDecay) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RMSpropOptions opts(0.01);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    RMSprop optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    // Should still converge reasonably
    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersRMSprop, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    RMSpropOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    RMSprop optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// Adagrad Tests
// ============================================================================

TEST(OptimizersAdagrad, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdagradOptions opts(0.5);
    std::vector<nn::Parameter*> params = {&param};
    Adagrad optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersAdagrad, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdagradOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    Adagrad optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<AdagradParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->sum.defined());
}

TEST(OptimizersAdagrad, AccumulatorGrows) {
    // Adagrad accumulates squared gradients, so the sum should grow
    Tensor x = at::full({1}, 0.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdagradOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adagrad optimizer(params, opts);

    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, 3.0f);
        optimizer.step();
    }

    auto* state = optimizer.get_state<AdagradParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 10);
    // Accumulated sum should be positive (sum of squared gradients)
    EXPECT_GT(state->sum.data_ptr<float>()[0], 0.0f);
}

TEST(OptimizersAdagrad, WeightDecay) {
    Tensor x = at::full({1}, 2.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdagradOptions opts(0.5);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adagrad optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersAdagrad, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    AdagradOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    Adagrad optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// Adadelta Tests
// ============================================================================

TEST(OptimizersAdadelta, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdadeltaOptions opts(50.0);
    opts.rho_(0.9);
    std::vector<nn::Parameter*> params = {&param};
    Adadelta optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 500; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersAdadelta, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdadeltaOptions opts(1.0);
    std::vector<nn::Parameter*> params = {&param};
    Adadelta optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<AdadeltaParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->square_avg.defined());
    EXPECT_TRUE(state->acc_delta.defined());
}

TEST(OptimizersAdadelta, WeightDecay) {
    Tensor x = at::full({1}, 2.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdadeltaOptions opts(1.0);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adadelta optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersAdadelta, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    AdadeltaOptions opts(1.0);
    std::vector<nn::Parameter*> params = {&param};
    Adadelta optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// RAdam Tests
// ============================================================================

TEST(OptimizersRAdam, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RAdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    RAdam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersRAdam, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RAdamOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    RAdam optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<RAdamParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->exp_avg.defined());
    EXPECT_TRUE(state->exp_avg_sq.defined());
}

TEST(OptimizersRAdam, EarlyStepsUseSGDFallback) {
    // RAdam uses SGD-like update when rho_t <= 5 (first few steps)
    // Just verify it doesn't crash and makes progress in the right direction
    Tensor x = at::full({1}, 0.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RAdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    RAdam optimizer(params, opts);

    float target = 3.0f;
    // First step: rho_t should be < 5, so SGD fallback
    optimizer.zero_grad();
    set_quadratic_gradient(param, target);
    optimizer.step();

    // x should have moved toward 3.0 (from 0.0)
    EXPECT_GT(param.data().data_ptr<float>()[0], 0.0f);
}

TEST(OptimizersRAdam, WeightDecay) {
    Tensor x = at::full({1}, 2.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RAdamOptions opts(0.1);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    RAdam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersRAdam, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    RAdamOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    RAdam optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// NAdam Tests
// ============================================================================

TEST(OptimizersNAdam, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    NAdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    NAdam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersNAdam, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    NAdamOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    NAdam optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<NAdamParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->exp_avg.defined());
    EXPECT_TRUE(state->exp_avg_sq.defined());
    // mu_product should have been updated from initial 1.0
    EXPECT_NE(state->mu_product, 1.0);
}

TEST(OptimizersNAdam, WeightDecay) {
    Tensor x = at::full({1}, 2.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    NAdamOptions opts(0.1);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    NAdam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersNAdam, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    NAdamOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    NAdam optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// Adamax Tests
// ============================================================================

TEST(OptimizersAdamax, ConvergesToTarget) {
    Tensor x = at::zeros({1});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamaxOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adamax optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    EXPECT_NEAR(param.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersAdamax, StateInitialized) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamaxOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    Adamax optimizer(params, opts);

    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    auto* state = optimizer.get_state<AdamaxParamState>(&param);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->step, 1);
    EXPECT_TRUE(state->exp_avg.defined());
    EXPECT_TRUE(state->exp_inf.defined());
    // Infinity norm should be positive after seeing a gradient
    EXPECT_GT(state->exp_inf.data_ptr<float>()[0], 0.0f);
}

TEST(OptimizersAdamax, WeightDecay) {
    Tensor x = at::full({1}, 2.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamaxOptions opts(0.1);
    opts.weight_decay_(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adamax optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float result = param.data().data_ptr<float>()[0];
    EXPECT_GT(result, 1.0f);
    EXPECT_LT(result, 4.0f);
}

TEST(OptimizersAdamax, ZeroGrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    set_quadratic_gradient(param, 3.0f);

    AdamaxOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    Adamax optimizer(params, opts);

    optimizer.zero_grad();

    Tensor g = param.grad();
    if (g.defined()) {
        EXPECT_NEAR(g.data_ptr<float>()[0], 0.0f, 1e-6f);
    }
}

// ============================================================================
// Multi-dimensional convergence test (all optimizers on {4} tensor)
// ============================================================================

TEST(OptimizersMultiDim, SGDConvergesMultiDim) {
    Tensor x = at::zeros({4});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    SGDOptions opts(0.1);
    opts.momentum_(0.9);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float* data = param.data().data_ptr<float>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[j], target, 0.5f);
    }
}

TEST(OptimizersMultiDim, AdamConvergesMultiDim) {
    Tensor x = at::zeros({4});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float* data = param.data().data_ptr<float>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[j], target, 0.5f);
    }
}

TEST(OptimizersMultiDim, AdagradConvergesMultiDim) {
    Tensor x = at::zeros({4});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    AdagradOptions opts(0.5);
    std::vector<nn::Parameter*> params = {&param};
    Adagrad optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float* data = param.data().data_ptr<float>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[j], target, 0.5f);
    }
}

TEST(OptimizersMultiDim, RAdamConvergesMultiDim) {
    Tensor x = at::zeros({4});
    x.set_requires_grad(true);
    nn::Parameter param(x);

    RAdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    RAdam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(param, target);
        optimizer.step();
    }

    float* data = param.data().data_ptr<float>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[j], target, 0.5f);
    }
}

// ============================================================================
// Multi-parameter test (two parameter groups)
// ============================================================================

TEST(OptimizersMultiParam, AdamTwoParameters) {
    Tensor x1 = at::zeros({1});
    Tensor x2 = at::full({1}, 5.0f);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    nn::Parameter p1(x1);
    nn::Parameter p2(x2);

    AdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&p1, &p2};
    Adam optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(p1, target);
        set_quadratic_gradient(p2, target);
        optimizer.step();
    }

    EXPECT_NEAR(p1.data().data_ptr<float>()[0], target, 0.5f);
    EXPECT_NEAR(p2.data().data_ptr<float>()[0], target, 0.5f);
}

TEST(OptimizersMultiParam, AdagradTwoParameters) {
    Tensor x1 = at::zeros({1});
    Tensor x2 = at::full({1}, 5.0f);
    x1.set_requires_grad(true);
    x2.set_requires_grad(true);
    nn::Parameter p1(x1);
    nn::Parameter p2(x2);

    AdagradOptions opts(0.5);
    std::vector<nn::Parameter*> params = {&p1, &p2};
    Adagrad optimizer(params, opts);

    float target = 3.0f;
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        set_quadratic_gradient(p1, target);
        set_quadratic_gradient(p2, target);
        optimizer.step();
    }

    EXPECT_NEAR(p1.data().data_ptr<float>()[0], target, 0.5f);
    EXPECT_NEAR(p2.data().data_ptr<float>()[0], target, 0.5f);
}

// ============================================================================
// Step counter test (verify step increments correctly)
// ============================================================================

TEST(OptimizersStepCount, AllOptimizersIncrementStep) {
    // Test that step count increments for optimizers that track it
    auto run_steps = [](auto& optimizer, nn::Parameter& param, int n) {
        for (int i = 0; i < n; ++i) {
            optimizer.zero_grad();
            set_quadratic_gradient(param, 3.0f);
            optimizer.step();
        }
    };

    // Adam
    {
        Tensor x = at::full({1}, 1.0f);
        x.set_requires_grad(true);
        nn::Parameter param(x);
        Adam opt({&param}, AdamOptions(0.01));
        run_steps(opt, param, 5);
        auto* s = opt.get_state<AdamParamState>(&param);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->step, 5);
    }

    // Adagrad
    {
        Tensor x = at::full({1}, 1.0f);
        x.set_requires_grad(true);
        nn::Parameter param(x);
        Adagrad opt({&param}, AdagradOptions(0.01));
        run_steps(opt, param, 7);
        auto* s = opt.get_state<AdagradParamState>(&param);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->step, 7);
    }

    // Adadelta
    {
        Tensor x = at::full({1}, 1.0f);
        x.set_requires_grad(true);
        nn::Parameter param(x);
        Adadelta opt({&param}, AdadeltaOptions(1.0));
        run_steps(opt, param, 3);
        auto* s = opt.get_state<AdadeltaParamState>(&param);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->step, 3);
    }

    // RAdam
    {
        Tensor x = at::full({1}, 1.0f);
        x.set_requires_grad(true);
        nn::Parameter param(x);
        RAdam opt({&param}, RAdamOptions(0.01));
        run_steps(opt, param, 10);
        auto* s = opt.get_state<RAdamParamState>(&param);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->step, 10);
    }

    // NAdam
    {
        Tensor x = at::full({1}, 1.0f);
        x.set_requires_grad(true);
        nn::Parameter param(x);
        NAdam opt({&param}, NAdamOptions(0.01));
        run_steps(opt, param, 4);
        auto* s = opt.get_state<NAdamParamState>(&param);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->step, 4);
    }

    // Adamax
    {
        Tensor x = at::full({1}, 1.0f);
        x.set_requires_grad(true);
        nn::Parameter param(x);
        Adamax opt({&param}, AdamaxOptions(0.01));
        run_steps(opt, param, 6);
        auto* s = opt.get_state<AdamaxParamState>(&param);
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->step, 6);
    }
}

// ============================================================================
// Factory function tests (from optim.h convenience wrappers)
// ============================================================================

TEST(OptimizersFactory, MakeAdagrad) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    auto optimizer = make_adagrad({&param}, 0.5);
    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    // Should have moved toward target (from 1.0 toward 3.0)
    EXPECT_GT(param.data().data_ptr<float>()[0], 1.0f);
}

TEST(OptimizersFactory, MakeAdadelta) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    auto optimizer = make_adadelta({&param}, 1.0, 0.9);
    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    // Should have moved toward target
    EXPECT_GT(param.data().data_ptr<float>()[0], 1.0f);
}

TEST(OptimizersFactory, MakeRAdam) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    auto optimizer = make_radam({&param}, 0.1);
    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    EXPECT_GT(param.data().data_ptr<float>()[0], 1.0f);
}

TEST(OptimizersFactory, MakeNAdam) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    auto optimizer = make_nadam({&param}, 0.1);
    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    EXPECT_GT(param.data().data_ptr<float>()[0], 1.0f);
}

TEST(OptimizersFactory, MakeAdamax) {
    Tensor x = at::full({1}, 1.0f);
    x.set_requires_grad(true);
    nn::Parameter param(x);

    auto optimizer = make_adamax({&param}, 0.1);
    set_quadratic_gradient(param, 3.0f);
    optimizer.step();

    EXPECT_GT(param.data().data_ptr<float>()[0], 1.0f);
}
