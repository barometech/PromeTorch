// ============================================================================
// PromeTorch Optimizer Tests
// ============================================================================

#include "torch/optim/optim.h"
#include "torch/nn/parameter.h"
#include "torch/csrc/autograd/autograd.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace torch;
using namespace torch::optim;
using namespace torch::autograd;

// ============================================================================
// Helper Functions
// ============================================================================

// Simple gradient descent on f(x) = 0.5 * sum(x^2)
// For quadratic loss f(x) = 0.5*x^2, gradient is x
void set_simple_gradient(nn::Parameter& param) {
    // Gradient is just x for quadratic loss
    at::Tensor grad = param.data().clone();
    param.set_grad(grad);
}

// ============================================================================
// SGD Tests
// ============================================================================

TEST(SGDTest, BasicStep) {
    // Create parameter with initial value 1.0
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    // Set gradient manually (gradient = 1.0 for x=1.0)
    set_simple_gradient(param);

    // Create SGD optimizer with lr=0.1
    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // Take one step: x_new = x - lr * grad = 1.0 - 0.1 * 1.0 = 0.9
    optimizer.step();

    float result = param.data().data_ptr<float>()[0];
    EXPECT_NEAR(result, 0.9f, 1e-6f);
}

TEST(SGDTest, MultipleSteps) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // Multiple steps towards optimal x=0
    for (int i = 0; i < 10; i++) {
        optimizer.zero_grad();
        set_simple_gradient(param);
        optimizer.step();
    }

    // After 10 steps: x = 1.0 * (1 - 0.1)^10 ≈ 0.349
    float result = param.data().data_ptr<float>()[0];
    float expected = std::pow(0.9f, 10);
    EXPECT_NEAR(result, expected, 1e-5f);
}

TEST(SGDTest, Momentum) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(0.1);
    opts.momentum_(0.9);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // With momentum, should converge faster - run more iterations
    for (int i = 0; i < 30; i++) {
        optimizer.zero_grad();
        set_simple_gradient(param);
        optimizer.step();
    }

    // Should converge towards 0 with momentum
    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(std::abs(result), 0.25f);  // Momentum oscillates, so allow larger threshold
}

TEST(SGDTest, WeightDecay) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(0.1);
    opts.weight_decay_(0.01);  // L2 regularization
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // Set zero gradient to isolate weight decay effect
    at::Tensor zero_grad = at::zeros({1});
    param.set_grad(zero_grad);

    optimizer.step();

    // With weight decay: x_new = x - lr * (grad + wd * x) = 1.0 - 0.1 * (0 + 0.01 * 1.0) = 0.999
    float result = param.data().data_ptr<float>()[0];
    EXPECT_NEAR(result, 0.999f, 1e-6f);
}

TEST(SGDTest, ZeroGrad) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    // Set gradient
    set_simple_gradient(param);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // Zero out gradient
    optimizer.zero_grad();

    // Gradient should be zero
    at::Tensor g = param.grad();
    if (g.defined()) {
        float grad_val = g.data_ptr<float>()[0];
        EXPECT_NEAR(grad_val, 0.0f, 1e-6f);
    }
}

// ============================================================================
// Adam Tests
// ============================================================================

TEST(AdamTest, BasicStep) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    set_simple_gradient(param);

    AdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    optimizer.step();

    // Adam update is more complex, just check it moved
    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(result, 1.0f);
    EXPECT_GT(result, 0.0f);
}

TEST(AdamTest, Convergence) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    AdamOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    // Multiple steps
    for (int i = 0; i < 100; i++) {
        optimizer.zero_grad();
        set_simple_gradient(param);
        optimizer.step();
    }

    // Should converge close to 0
    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(std::abs(result), 0.1f);
}

TEST(AdamTest, AMSGrad) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    AdamOptions opts(0.1);
    opts.amsgrad_(true);
    std::vector<nn::Parameter*> params = {&param};
    Adam optimizer(params, opts);

    for (int i = 0; i < 50; i++) {
        optimizer.zero_grad();
        set_simple_gradient(param);
        optimizer.step();
    }

    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(std::abs(result), 0.2f);
}

// ============================================================================
// AdamW Tests
// ============================================================================

TEST(AdamWTest, DecoupledWeightDecay) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    // Set zero gradient to isolate weight decay effect
    at::Tensor zero_grad = at::zeros({1});
    param.set_grad(zero_grad);

    AdamWOptions opts(0.1);
    opts.weight_decay_(0.1);  // Strong weight decay
    std::vector<nn::Parameter*> params = {&param};
    AdamW optimizer(params, opts);

    optimizer.step();

    // With decoupled weight decay: x_new = x * (1 - lr * wd) = 1.0 * (1 - 0.1 * 0.1) = 0.99
    float result = param.data().data_ptr<float>()[0];
    EXPECT_NEAR(result, 0.99f, 1e-5f);
}

// ============================================================================
// RMSprop Tests
// ============================================================================

TEST(RMSpropTest, BasicStep) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    set_simple_gradient(param);

    RMSpropOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    RMSprop optimizer(params, opts);

    optimizer.step();

    // Just check it moved in right direction
    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(result, 1.0f);
}

TEST(RMSpropTest, Convergence) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    RMSpropOptions opts(0.01);
    std::vector<nn::Parameter*> params = {&param};
    RMSprop optimizer(params, opts);

    for (int i = 0; i < 100; i++) {
        optimizer.zero_grad();
        set_simple_gradient(param);
        optimizer.step();
    }

    // Should converge close to 0
    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(std::abs(result), 0.2f);
}

// ============================================================================
// Learning Rate Scheduler Tests
// ============================================================================

TEST(StepLRTest, DecayAtStepSize) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    StepLR scheduler(optimizer, 2, 0.1);  // Decay every 2 epochs by 0.1

    // Initial LR (last_epoch = -1, so factor = 0.1^(-1/2) = 0.1^0 = 1 with integer division)
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);

    scheduler.step();  // epoch 0: 0.1 * 0.1^(0/2) = 0.1 * 1 = 0.1
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);

    scheduler.step();  // epoch 1: 0.1 * 0.1^(1/2) = 0.1 * 1 = 0.1
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);

    scheduler.step();  // epoch 2: 0.1 * 0.1^(2/2) = 0.1 * 0.1 = 0.01
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-6);

    scheduler.step();  // epoch 3: 0.1 * 0.1^(3/2) = 0.1 * 0.1 = 0.01
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-6);

    scheduler.step();  // epoch 4: 0.1 * 0.1^(4/2) = 0.1 * 0.01 = 0.001
    EXPECT_NEAR(scheduler.get_last_lr(), 0.001, 1e-6);
}

TEST(MultiStepLRTest, DecayAtMilestones) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    MultiStepLR scheduler(optimizer, {3, 7}, 0.1);

    // Initial LR
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);

    // 4 steps to reach epoch 3 (from -1 to 0,1,2,3)
    for (int i = 0; i < 4; i++) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-6);  // After epoch 3

    // 4 more steps to reach epoch 7
    for (int i = 0; i < 4; i++) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.001, 1e-6);  // After epoch 7
}

TEST(ExponentialLRTest, ExponentialDecay) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(1.0);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    ExponentialLR scheduler(optimizer, 0.9);

    // Initial LR
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);

    // After step: lr = base_lr * gamma^epoch
    scheduler.step();  // epoch 0: 1.0 * 0.9^0 = 1.0
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);

    scheduler.step();  // epoch 1: 1.0 * 0.9^1 = 0.9
    EXPECT_NEAR(scheduler.get_last_lr(), 0.9, 1e-6);

    scheduler.step();  // epoch 2: 1.0 * 0.9^2 = 0.81
    EXPECT_NEAR(scheduler.get_last_lr(), 0.81, 1e-6);
}

TEST(CosineAnnealingLRTest, CosineSchedule) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(1.0);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    CosineAnnealingLR scheduler(optimizer, 10, 0.0);  // T_max=10, eta_min=0

    // Initial LR
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);

    // At epoch T_max/2 = 5: lr = 0.5 * (1 + cos(pi * 5 / 10)) = 0.5 * (1 + 0) = 0.5
    // Need 6 steps to reach epoch 5 (from -1 to 0,1,2,3,4,5)
    for (int i = 0; i < 6; i++) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.5, 0.05);

    // At epoch T_max = 10: lr = 0
    // Need 5 more steps
    for (int i = 0; i < 5; i++) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0, 0.05);
}

TEST(ReduceLROnPlateauTest, ReduceOnPlateau) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    SGDOptions opts(1.0);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    ReduceLROnPlateau scheduler(optimizer, ReduceLROnPlateau::Mode::Min, 0.1, 2);  // patience=2

    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);

    // Decreasing loss - should not reduce
    scheduler.step(1.0);
    scheduler.step(0.9);
    scheduler.step(0.8);
    EXPECT_NEAR(scheduler.get_last_lr(), 1.0, 1e-6);

    // Plateau - should reduce after patience
    scheduler.step(0.8);
    scheduler.step(0.8);
    scheduler.step(0.8);  // patience exceeded
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-6);  // Reduced by factor 0.1
}

// ============================================================================
// Factory Functions Tests
// ============================================================================

TEST(FactoryTest, MakeSGD) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    std::vector<nn::Parameter*> params = {&param};
    auto optimizer = make_sgd(params, 0.01, 0.9);

    set_simple_gradient(param);
    optimizer.step();

    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(result, 1.0f);
}

TEST(FactoryTest, MakeAdam) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    std::vector<nn::Parameter*> params = {&param};
    auto optimizer = make_adam(params, 0.001);

    set_simple_gradient(param);
    optimizer.step();

    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(result, 1.0f);
}

TEST(FactoryTest, MakeAdamW) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    std::vector<nn::Parameter*> params = {&param};
    auto optimizer = make_adamw(params, 0.001, 0.01);

    set_simple_gradient(param);
    optimizer.step();

    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(result, 1.0f);
}

TEST(FactoryTest, MakeRMSprop) {
    at::Tensor t = at::full({1}, 1.0);
    t.set_requires_grad(true);
    nn::Parameter param(t);

    std::vector<nn::Parameter*> params = {&param};
    auto optimizer = make_rmsprop(params, 0.01);

    set_simple_gradient(param);
    optimizer.step();

    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(result, 1.0f);
}

// ============================================================================
// Multi-Parameter Tests
// ============================================================================

TEST(MultiParamTest, SGDMultipleParameters) {
    at::Tensor t1 = at::full({2}, 1.0);
    at::Tensor t2 = at::full({3}, 2.0);
    t1.set_requires_grad(true);
    t2.set_requires_grad(true);

    nn::Parameter param1(t1);
    nn::Parameter param2(t2);

    // Set gradients
    set_simple_gradient(param1);
    set_simple_gradient(param2);

    SGDOptions opts(0.1);
    std::vector<nn::Parameter*> params = {&param1, &param2};
    SGD optimizer(params, opts);

    optimizer.step();

    // Check both parameters updated
    float* data1 = param1.data().mutable_data_ptr<float>();
    float* data2 = param2.data().mutable_data_ptr<float>();

    EXPECT_NEAR(data1[0], 0.9f, 1e-6f);
    EXPECT_NEAR(data1[1], 0.9f, 1e-6f);
    EXPECT_NEAR(data2[0], 1.8f, 1e-6f);
    EXPECT_NEAR(data2[1], 1.8f, 1e-6f);
    EXPECT_NEAR(data2[2], 1.8f, 1e-6f);
}

// ============================================================================
// Integration Test with Autograd
// ============================================================================

TEST(IntegrationTest, AutogradWithSGD) {
    // Create parameter
    at::Tensor w = at::full({1}, 2.0);
    w.set_requires_grad(true);
    nn::Parameter param(w);

    SGDOptions opts(0.5);
    std::vector<nn::Parameter*> params = {&param};
    SGD optimizer(params, opts);

    // Simple training loop: minimize f(w) = w^2
    // Optimal w = 0
    for (int i = 0; i < 20; i++) {
        optimizer.zero_grad();

        // Compute loss = w^2 using autograd
        at::Tensor loss = mul_autograd(param.data(), param.data());

        // Backward pass
        tensor_backward(loss);

        // Update
        optimizer.step();
    }

    // Should be close to 0
    float result = param.data().data_ptr<float>()[0];
    EXPECT_LT(std::abs(result), 0.01f);
}
