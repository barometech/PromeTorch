// ============================================================================
// PromeTorch Neural Network Module Tests
// ============================================================================
// Comprehensive test suite for torch::nn module functionality
// ============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <sstream>

#include "torch/nn/nn.h"
#include "torch/nn/functional.h"

using namespace torch::nn;
using namespace at;

// ============================================================================
// Test Utilities
// ============================================================================

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) run_test(#name, test_##name)

void run_test(const std::string& name, void (*test_fn)()) {
    try {
        test_fn();
        std::cout << "[PASS] " << name << std::endl;
        tests_passed++;
    } catch (const std::exception& e) {
        std::cout << "[FAIL] " << name << ": " << e.what() << std::endl;
        tests_failed++;
    }
}

void assert_close(float a, float b, float tol = 1e-5f, const std::string& msg = "") {
    if (std::abs(a - b) > tol) {
        std::ostringstream oss;
        oss << "Assertion failed: " << a << " != " << b;
        if (!msg.empty()) oss << " (" << msg << ")";
        throw std::runtime_error(oss.str());
    }
}

void assert_shape(const Tensor& t, const std::vector<int64_t>& expected) {
    auto actual = t.sizes().vec();
    if (actual != expected) {
        std::ostringstream oss;
        oss << "Shape mismatch: got (";
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << actual[i];
        }
        oss << "), expected (";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << expected[i];
        }
        oss << ")";
        throw std::runtime_error(oss.str());
    }
}

// ============================================================================
// Module Base Tests
// ============================================================================

TEST(module_parameter_registration) {
    auto linear = std::make_shared<Linear>(10, 5);

    auto params = linear->parameters();
    assert(params.size() == 2);  // weight and bias

    auto named = linear->named_parameters();
    assert(named.size() == 2);
    assert(named.count("weight") == 1);
    assert(named.count("bias") == 1);
}

TEST(module_train_eval_mode) {
    auto module = std::make_shared<Dropout>(0.5);

    module->train();
    assert(module->is_training());

    module->eval();
    assert(!module->is_training());
}

TEST(module_state_dict) {
    auto linear1 = std::make_shared<Linear>(10, 5);
    auto linear2 = std::make_shared<Linear>(10, 5);

    // Save state
    auto state = linear1->state_dict();
    assert(state.size() == 2);

    // Load state into another module
    linear2->load_state_dict(state);

    // Verify weights match
    auto w1 = linear1->get_parameter("weight")->data();
    auto w2 = linear2->get_parameter("weight")->data();

    const float* d1 = w1.data_ptr<float>();
    const float* d2 = w2.data_ptr<float>();

    for (int64_t i = 0; i < w1.numel(); ++i) {
        assert_close(d1[i], d2[i], 1e-6f);
    }
}

TEST(module_zero_grad) {
    auto linear = std::make_shared<Linear>(10, 5);

    // Manually set some gradients
    auto& weight = linear->get_parameter("weight")->data();
    Tensor grad = at::ones(weight.sizes());
    linear->get_parameter("weight")->set_grad(grad);

    // Zero gradients
    linear->zero_grad();

    // Verify gradients are zeroed
    auto new_grad = linear->get_parameter("weight")->grad();
    if (new_grad.defined()) {
        const float* g = new_grad.data_ptr<float>();
        for (int64_t i = 0; i < new_grad.numel(); ++i) {
            assert_close(g[i], 0.0f);
        }
    }
}

// ============================================================================
// Container Tests
// ============================================================================

TEST(sequential_forward) {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(10, 20));
    seq->add(std::make_shared<ReLU>());
    seq->add(std::make_shared<Linear>(20, 5));

    Tensor input = at::randn({2, 10});
    Tensor output = seq->forward(input);

    assert_shape(output, {2, 5});
}

TEST(sequential_parameters) {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(10, 20));
    seq->add(std::make_shared<Linear>(20, 5));

    auto params = seq->parameters();
    // 2 layers x 2 params (weight + bias) = 4 parameters
    assert(params.size() == 4);
}

TEST(module_list) {
    auto list = std::make_shared<ModuleList>();
    list->append(std::make_shared<Linear>(10, 10));
    list->append(std::make_shared<Linear>(10, 10));
    list->append(std::make_shared<Linear>(10, 10));

    assert(list->size() == 3);

    Tensor x = at::randn({2, 10});
    for (size_t i = 0; i < list->size(); ++i) {
        x = (*list)[i]->forward(x);
    }
    assert_shape(x, {2, 10});
}

TEST(module_dict) {
    auto dict = std::make_shared<ModuleDict>();
    dict->insert("encoder", std::make_shared<Linear>(10, 20));
    dict->insert("decoder", std::make_shared<Linear>(20, 10));

    assert(dict->size() == 2);
    assert(dict->contains("encoder"));
    assert(dict->contains("decoder"));

    Tensor x = at::randn({2, 10});
    x = dict->at("encoder")->forward(x);
    assert_shape(x, {2, 20});
    x = dict->at("decoder")->forward(x);
    assert_shape(x, {2, 10});
}

// ============================================================================
// Linear Layer Tests
// ============================================================================

TEST(linear_forward) {
    auto linear = std::make_shared<Linear>(10, 5, true);

    Tensor input = at::randn({2, 10});
    Tensor output = linear->forward(input);

    assert_shape(output, {2, 5});
}

TEST(linear_no_bias) {
    auto linear = std::make_shared<Linear>(10, 5, false);

    auto params = linear->parameters();
    assert(params.size() == 1);  // Only weight, no bias

    Tensor input = at::randn({2, 10});
    Tensor output = linear->forward(input);
    assert_shape(output, {2, 5});
}

TEST(linear_batch) {
    auto linear = std::make_shared<Linear>(10, 5);

    Tensor input = at::randn({4, 3, 10});  // Batch with extra dim
    Tensor output = linear->forward(input);
    assert_shape(output, {4, 3, 5});
}

TEST(identity_forward) {
    auto identity = std::make_shared<Identity>();

    Tensor input = at::randn({2, 10});
    Tensor output = identity->forward(input);

    assert_shape(output, {2, 10});

    // Values should be identical
    const float* in = input.data_ptr<float>();
    const float* out = output.data_ptr<float>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        assert_close(in[i], out[i]);
    }
}

TEST(bilinear_forward) {
    auto bilinear = std::make_shared<Bilinear>(5, 6, 3);

    Tensor input1 = at::randn({2, 5});
    Tensor input2 = at::randn({2, 6});
    Tensor output = bilinear->forward(input1, input2);

    assert_shape(output, {2, 3});
}

// ============================================================================
// Activation Tests
// ============================================================================

TEST(relu_forward) {
    auto relu = std::make_shared<ReLU>();

    Tensor input = at::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    input = input.reshape({1, 5});
    Tensor output = relu->forward(input);

    const float* out = output.data_ptr<float>();
    assert_close(out[0], 0.0f);
    assert_close(out[1], 0.0f);
    assert_close(out[2], 0.0f);
    assert_close(out[3], 1.0f);
    assert_close(out[4], 2.0f);
}

TEST(leaky_relu_forward) {
    auto lrelu = std::make_shared<LeakyReLU>(0.1);

    Tensor input = at::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    input = input.reshape({1, 5});
    Tensor output = lrelu->forward(input);

    const float* out = output.data_ptr<float>();
    assert_close(out[0], -0.2f);
    assert_close(out[1], -0.1f);
    assert_close(out[2], 0.0f);
    assert_close(out[3], 1.0f);
    assert_close(out[4], 2.0f);
}

TEST(sigmoid_forward) {
    auto sigmoid = std::make_shared<Sigmoid>();

    Tensor input = at::tensor({0.0f});
    input = input.reshape({1, 1});
    Tensor output = sigmoid->forward(input);

    assert_close(output.data_ptr<float>()[0], 0.5f);
}

TEST(tanh_forward) {
    auto tanh_act = std::make_shared<Tanh>();

    Tensor input = at::tensor({0.0f});
    input = input.reshape({1, 1});
    Tensor output = tanh_act->forward(input);

    assert_close(output.data_ptr<float>()[0], 0.0f);
}

TEST(softmax_forward) {
    auto softmax = std::make_shared<Softmax>(1);

    Tensor input = at::tensor({1.0f, 2.0f, 3.0f});
    input = input.reshape({1, 3});
    Tensor output = softmax->forward(input);

    const float* out = output.data_ptr<float>();

    // Sum should be 1
    float sum = out[0] + out[1] + out[2];
    assert_close(sum, 1.0f, 1e-4f);

    // Larger input should have larger probability
    assert(out[2] > out[1]);
    assert(out[1] > out[0]);
}

TEST(gelu_forward) {
    auto gelu = std::make_shared<GELU>();

    Tensor input = at::tensor({0.0f});
    input = input.reshape({1, 1});
    Tensor output = gelu->forward(input);

    // GELU(0) = 0
    assert_close(output.data_ptr<float>()[0], 0.0f, 1e-4f);
}

TEST(silu_forward) {
    auto silu = std::make_shared<SiLU>();

    Tensor input = at::tensor({0.0f});
    input = input.reshape({1, 1});
    Tensor output = silu->forward(input);

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert_close(output.data_ptr<float>()[0], 0.0f);
}

TEST(mish_forward) {
    auto mish = std::make_shared<Mish>();

    Tensor input = at::tensor({0.0f});
    input = input.reshape({1, 1});
    Tensor output = mish->forward(input);

    // Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) ≈ 0
    assert_close(output.data_ptr<float>()[0], 0.0f);
}

// ============================================================================
// Convolution Tests
// ============================================================================

TEST(conv1d_forward) {
    auto conv = std::make_shared<Conv1d>(3, 6, 3, 1, 1);

    Tensor input = at::randn({2, 3, 10});  // batch=2, channels=3, length=10
    Tensor output = conv->forward(input);

    assert_shape(output, {2, 6, 10});  // Same length due to padding=1
}

TEST(conv2d_forward) {
    auto conv = std::make_shared<Conv2d>(3, 6, 3, 1, 1);

    Tensor input = at::randn({2, 3, 8, 8});  // batch=2, channels=3, H=W=8
    Tensor output = conv->forward(input);

    assert_shape(output, {2, 6, 8, 8});  // Same spatial size due to padding=1
}

TEST(conv2d_stride) {
    auto conv = std::make_shared<Conv2d>(1, 1, 3, 2, 0);  // stride=2, no padding

    Tensor input = at::randn({1, 1, 8, 8});
    Tensor output = conv->forward(input);

    // Output size: (8 - 3) / 2 + 1 = 3
    assert_shape(output, {1, 1, 3, 3});
}

TEST(conv3d_forward) {
    auto conv = std::make_shared<Conv3d>(1, 2, 3, 1, 1);

    Tensor input = at::randn({1, 1, 4, 4, 4});  // batch=1, channels=1, D=H=W=4
    Tensor output = conv->forward(input);

    assert_shape(output, {1, 2, 4, 4, 4});
}

TEST(conv_transpose2d_forward) {
    auto conv_t = std::make_shared<ConvTranspose2d>(1, 1, 3, 2, 0);

    Tensor input = at::randn({1, 1, 3, 3});
    Tensor output = conv_t->forward(input);

    // Transposed conv output: (3 - 1) * 2 + 3 = 7
    assert_shape(output, {1, 1, 7, 7});
}

// ============================================================================
// Pooling Tests
// ============================================================================

TEST(maxpool2d_forward) {
    auto pool = std::make_shared<MaxPool2d>(2, 2);

    Tensor input = at::randn({1, 1, 4, 4});
    Tensor output = pool->forward(input);

    assert_shape(output, {1, 1, 2, 2});
}

TEST(avgpool2d_forward) {
    auto pool = std::make_shared<AvgPool2d>(2, 2);

    Tensor input = at::randn({1, 1, 4, 4});
    Tensor output = pool->forward(input);

    assert_shape(output, {1, 1, 2, 2});
}

TEST(adaptive_avgpool2d_forward) {
    auto pool = std::make_shared<AdaptiveAvgPool2d>(std::array<int64_t, 2>{1, 1});

    Tensor input = at::randn({2, 3, 8, 8});
    Tensor output = pool->forward(input);

    assert_shape(output, {2, 3, 1, 1});
}

TEST(global_avgpool2d_forward) {
    auto pool = std::make_shared<GlobalAvgPool2d>();

    Tensor input = at::randn({2, 3, 8, 8});
    Tensor output = pool->forward(input);

    assert_shape(output, {2, 3, 1, 1});
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST(batchnorm1d_forward) {
    auto bn = std::make_shared<BatchNorm1d>(10);

    Tensor input = at::randn({4, 10});
    Tensor output = bn->forward(input);

    assert_shape(output, {4, 10});
}

TEST(batchnorm2d_forward) {
    auto bn = std::make_shared<BatchNorm2d>(3);

    Tensor input = at::randn({2, 3, 8, 8});
    Tensor output = bn->forward(input);

    assert_shape(output, {2, 3, 8, 8});
}

TEST(batchnorm2d_running_stats) {
    auto bn = std::make_shared<BatchNorm2d>(2, 1e-5, 0.1, true, true);

    bn->train();

    // Pass some data to update running stats
    for (int i = 0; i < 10; ++i) {
        Tensor input = at::randn({4, 2, 4, 4});
        bn->forward(input);
    }

    // Check running stats were updated
    auto& running_mean = bn->get_buffer("running_mean")->data();
    auto& running_var = bn->get_buffer("running_var")->data();

    // Running mean should be close to 0 for normal input
    // Running var should be close to 1 for normal input
    assert(running_mean.defined());
    assert(running_var.defined());
}

TEST(layernorm_forward) {
    auto ln = std::make_shared<LayerNorm>(std::vector<int64_t>{10});

    Tensor input = at::randn({2, 3, 10});
    Tensor output = ln->forward(input);

    assert_shape(output, {2, 3, 10});
}

TEST(groupnorm_forward) {
    auto gn = std::make_shared<GroupNorm>(4, 8);  // 4 groups for 8 channels

    Tensor input = at::randn({2, 8, 4, 4});
    Tensor output = gn->forward(input);

    assert_shape(output, {2, 8, 4, 4});
}

TEST(instancenorm2d_forward) {
    auto inst_norm = std::make_shared<InstanceNorm2d>(3);

    Tensor input = at::randn({2, 3, 8, 8});
    Tensor output = inst_norm->forward(input);

    assert_shape(output, {2, 3, 8, 8});
}

// ============================================================================
// Dropout Tests
// ============================================================================

TEST(dropout_train_vs_eval) {
    auto dropout = std::make_shared<Dropout>(0.5);

    Tensor input = at::ones({100, 100});

    // In eval mode, output should equal input
    dropout->eval();
    Tensor output_eval = dropout->forward(input);
    float sum_eval = 0.0f;
    const float* d_eval = output_eval.data_ptr<float>();
    for (int64_t i = 0; i < output_eval.numel(); ++i) {
        sum_eval += d_eval[i];
    }
    assert_close(sum_eval, 10000.0f);  // All ones

    // In train mode, some values should be zeroed
    dropout->train();
    Tensor output_train = dropout->forward(input);
    int zeros = 0;
    int scaled = 0;
    const float* d_train = output_train.data_ptr<float>();
    for (int64_t i = 0; i < output_train.numel(); ++i) {
        if (d_train[i] == 0.0f) zeros++;
        else if (std::abs(d_train[i] - 2.0f) < 0.01f) scaled++;  // Scaled by 1/(1-0.5) = 2
    }
    // Should have roughly 50% zeros (with some variance)
    assert(zeros > 4000 && zeros < 6000);
}

TEST(dropout2d_forward) {
    auto dropout = std::make_shared<Dropout2d>(0.5);
    dropout->train();

    Tensor input = at::ones({2, 10, 4, 4});
    Tensor output = dropout->forward(input);

    assert_shape(output, {2, 10, 4, 4});
}

// ============================================================================
// Embedding Tests
// ============================================================================

TEST(embedding_forward) {
    auto emb = std::make_shared<Embedding>(10, 5);  // vocab=10, dim=5

    Tensor indices = at::tensor({0.0f, 2.0f, 5.0f, 9.0f});
    indices = indices.reshape({2, 2});
    Tensor output = emb->forward(indices);

    assert_shape(output, {2, 2, 5});
}

TEST(embedding_padding_idx) {
    auto emb = std::make_shared<Embedding>(10, 5, 0);  // padding_idx=0

    // Check that padding index embedding is zeros
    auto& weight = emb->get_parameter("weight")->data();
    const float* w = weight.data_ptr<float>();

    for (int64_t i = 0; i < 5; ++i) {
        assert_close(w[i], 0.0f);  // First row should be zeros
    }
}

TEST(embeddingbag_forward) {
    auto emb_bag = std::make_shared<EmbeddingBag>(10, 5, -1.0, 2.0, false, EmbeddingBagMode::Mean);

    Tensor indices = at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    indices = indices.reshape({2, 3});  // 2 bags of 3 indices each
    Tensor output = emb_bag->forward(indices);

    assert_shape(output, {2, 5});
}

// ============================================================================
// Loss Function Tests
// ============================================================================

TEST(mse_loss_forward) {
    auto mse = std::make_shared<MSELoss>(Reduction::Mean);

    Tensor input = at::tensor({1.0f, 2.0f, 3.0f});
    Tensor target = at::tensor({1.0f, 2.0f, 3.0f});

    Tensor loss = mse->forward(input, target);
    assert_close(loss.data_ptr<float>()[0], 0.0f);

    // Different values
    target = at::tensor({0.0f, 0.0f, 0.0f});
    loss = mse->forward(input, target);
    // (1 + 4 + 9) / 3 = 14/3 ≈ 4.667
    assert_close(loss.data_ptr<float>()[0], 14.0f / 3.0f, 1e-4f);
}

TEST(l1_loss_forward) {
    auto l1 = std::make_shared<L1Loss>(Reduction::Mean);

    Tensor input = at::tensor({1.0f, 2.0f, 3.0f});
    Tensor target = at::tensor({0.0f, 0.0f, 0.0f});

    Tensor loss = l1->forward(input, target);
    // (1 + 2 + 3) / 3 = 2
    assert_close(loss.data_ptr<float>()[0], 2.0f);
}

TEST(bce_loss_forward) {
    auto bce = std::make_shared<BCELoss>(Reduction::Mean);

    Tensor input = at::tensor({0.5f, 0.5f});
    Tensor target = at::tensor({1.0f, 0.0f});

    Tensor loss = bce->forward(input, target);
    // -[1*log(0.5) + 0*log(0.5) + 0*log(0.5) + 1*log(0.5)] / 2
    // = -2 * log(0.5) / 2 = -log(0.5) ≈ 0.693
    assert_close(loss.data_ptr<float>()[0], std::log(2.0f), 0.01f);
}

TEST(cross_entropy_loss_forward) {
    auto ce = std::make_shared<CrossEntropyLoss>(Reduction::Mean);

    // Input: logits (N=2, C=3)
    Tensor input = at::tensor({1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
    input = input.reshape({2, 3});
    // Target: class indices
    Tensor target = at::tensor({2.0f, 0.0f});

    Tensor loss = ce->forward(input, target);

    // CrossEntropy includes softmax
    // For uniform prediction [1,2,3], softmax gives [0.09, 0.24, 0.67]
    // -log(0.67) ≈ 0.4 for class 2
    // -log(0.09) ≈ 2.4 for class 0
    // Mean ≈ 1.4
    assert(loss.data_ptr<float>()[0] > 0.0f);
}

TEST(nll_loss_forward) {
    auto nll = std::make_shared<NLLLoss>(Reduction::Mean);

    // Input: log-probabilities (already log-softmax applied)
    Tensor input = at::tensor({-0.5f, -1.0f, -2.0f, -0.5f, -1.0f, -2.0f});
    input = input.reshape({2, 3});
    Tensor target = at::tensor({0.0f, 1.0f});

    Tensor loss = nll->forward(input, target);
    // -(-0.5 + -1.0) / 2 = 0.75
    assert_close(loss.data_ptr<float>()[0], 0.75f, 0.01f);
}

TEST(smooth_l1_loss_forward) {
    auto smooth_l1 = std::make_shared<SmoothL1Loss>(Reduction::Mean, 1.0);

    Tensor input = at::tensor({0.0f, 2.0f});
    Tensor target = at::tensor({0.5f, 0.0f});

    Tensor loss = smooth_l1->forward(input, target);
    // |0 - 0.5| = 0.5 < 1: 0.5 * 0.25 / 1 = 0.125
    // |2 - 0| = 2 >= 1: 2 - 0.5 = 1.5
    // Mean = (0.125 + 1.5) / 2 = 0.8125
    assert_close(loss.data_ptr<float>()[0], 0.8125f, 0.01f);
}

TEST(kl_div_loss_forward) {
    auto kl = std::make_shared<KLDivLoss>(Reduction::Sum, false);

    // Input: log(Q), Target: P
    Tensor input = at::tensor({std::log(0.5f), std::log(0.5f)});
    Tensor target = at::tensor({0.5f, 0.5f});

    Tensor loss = kl->forward(input, target);
    // KL(P||Q) = sum(P * (log(P) - log(Q)))
    // = 0.5 * (log(0.5) - log(0.5)) + 0.5 * (log(0.5) - log(0.5)) = 0
    assert_close(loss.data_ptr<float>()[0], 0.0f, 1e-4f);
}

TEST(triplet_margin_loss_forward) {
    auto triplet = std::make_shared<TripletMarginLoss>(Reduction::Mean, 1.0, 2.0);

    // Anchor and positive close, negative far
    Tensor anchor = at::tensor({0.0f, 0.0f});
    anchor = anchor.reshape({1, 2});
    Tensor positive = at::tensor({0.1f, 0.1f});
    positive = positive.reshape({1, 2});
    Tensor negative = at::tensor({10.0f, 10.0f});
    negative = negative.reshape({1, 2});

    Tensor loss = triplet->forward(anchor, positive, negative);
    // d(a, p) << d(a, n), so loss should be ~0
    assert(loss.data_ptr<float>()[0] < 0.5f);
}

TEST(focal_loss_forward) {
    auto focal = std::make_shared<FocalLoss>(Reduction::Mean, 0.25, 2.0);

    Tensor input = at::tensor({0.9f, 0.1f});
    Tensor target = at::tensor({1.0f, 0.0f});

    Tensor loss = focal->forward(input, target);
    // Focal loss should be low for correct predictions with high confidence
    assert(loss.data_ptr<float>()[0] < 0.1f);
}

TEST(dice_loss_forward) {
    auto dice = std::make_shared<DiceLoss>(Reduction::Mean, 1.0);

    // Perfect prediction
    Tensor input = at::tensor({1.0f, 0.0f, 1.0f, 0.0f});
    input = input.reshape({1, 4});
    Tensor target = at::tensor({1.0f, 0.0f, 1.0f, 0.0f});
    target = target.reshape({1, 4});

    Tensor loss = dice->forward(input, target);
    // Dice = 2*|X∩Y| / (|X| + |Y|) = 2*2 / (2+2) = 1
    // Loss = 1 - Dice ≈ 0
    assert_close(loss.data_ptr<float>()[0], 0.0f, 0.1f);
}

// ============================================================================
// Functional API Tests
// ============================================================================

TEST(functional_relu) {
    Tensor input = at::tensor({-1.0f, 0.0f, 1.0f});
    Tensor output = F::relu(input);

    const float* out = output.data_ptr<float>();
    assert_close(out[0], 0.0f);
    assert_close(out[1], 0.0f);
    assert_close(out[2], 1.0f);
}

TEST(functional_softmax) {
    Tensor input = at::tensor({1.0f, 2.0f, 3.0f});
    input = input.reshape({1, 3});
    Tensor output = F::softmax(input, 1);

    const float* out = output.data_ptr<float>();
    float sum = out[0] + out[1] + out[2];
    assert_close(sum, 1.0f, 1e-4f);
}

TEST(functional_linear) {
    Tensor input = at::ones({2, 3});
    Tensor weight = at::ones({4, 3});
    Tensor bias = at::zeros({4});

    Tensor output = F::linear(input, weight, &bias);

    assert_shape(output, {2, 4});
    // Each output should be 3 (sum of three 1s)
    assert_close(output.data_ptr<float>()[0], 3.0f);
}

TEST(functional_mse_loss) {
    Tensor input = at::tensor({1.0f, 2.0f, 3.0f});
    Tensor target = at::tensor({1.0f, 2.0f, 3.0f});

    Tensor loss = F::mse_loss(input, target, "mean");
    assert_close(loss.data_ptr<float>()[0], 0.0f);
}

TEST(functional_max_pool2d) {
    Tensor input = at::randn({1, 1, 4, 4});
    Tensor output = F::max_pool2d(input, {2, 2}, {2, 2});

    assert_shape(output, {1, 1, 2, 2});
}

TEST(functional_dropout) {
    Tensor input = at::ones({100});

    // Training mode
    Tensor out_train = F::dropout(input, 0.5, true);
    int zeros = 0;
    const float* d = out_train.data_ptr<float>();
    for (int64_t i = 0; i < 100; ++i) {
        if (d[i] == 0.0f) zeros++;
    }
    assert(zeros > 30 && zeros < 70);  // ~50% zeros

    // Eval mode
    Tensor out_eval = F::dropout(input, 0.5, false);
    const float* d_eval = out_eval.data_ptr<float>();
    for (int64_t i = 0; i < 100; ++i) {
        assert_close(d_eval[i], 1.0f);
    }
}

// ============================================================================
// Init Tests
// ============================================================================

TEST(init_zeros) {
    Tensor t = at::empty({3, 3});
    init::zeros_(t);

    const float* data = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        assert_close(data[i], 0.0f);
    }
}

TEST(init_ones) {
    Tensor t = at::empty({3, 3});
    init::ones_(t);

    const float* data = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        assert_close(data[i], 1.0f);
    }
}

TEST(init_constant) {
    Tensor t = at::empty({3, 3});
    init::constant_(t, 5.0);

    const float* data = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        assert_close(data[i], 5.0f);
    }
}

TEST(init_uniform) {
    Tensor t = at::empty({1000});
    init::uniform_(t, 0.0, 1.0);

    const float* data = t.data_ptr<float>();
    float min_val = 1.0f, max_val = 0.0f;
    for (int64_t i = 0; i < t.numel(); ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    assert(min_val >= 0.0f);
    assert(max_val <= 1.0f);
    assert(max_val - min_val > 0.5f);  // Should have good spread
}

TEST(init_normal) {
    Tensor t = at::empty({10000});
    init::normal_(t, 0.0, 1.0);

    const float* data = t.data_ptr<float>();
    double sum = 0.0, sq_sum = 0.0;
    for (int64_t i = 0; i < t.numel(); ++i) {
        sum += data[i];
        sq_sum += data[i] * data[i];
    }

    double mean = sum / t.numel();
    double var = sq_sum / t.numel() - mean * mean;

    // Mean should be close to 0
    assert(std::abs(mean) < 0.1);
    // Variance should be close to 1
    assert(std::abs(var - 1.0) < 0.2);
}

TEST(init_kaiming_uniform) {
    Tensor t = at::empty({64, 32});
    init::kaiming_uniform_(t, 0, init::FanMode::FanIn, "relu");

    // Check that values are within expected range
    const float* data = t.data_ptr<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < t.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }

    // For fan_in=32 and relu: bound = sqrt(3) * sqrt(2/32) ≈ 0.433
    assert(max_abs < 1.0f);
    assert(max_abs > 0.1f);
}

TEST(init_xavier_uniform) {
    Tensor t = at::empty({64, 32});
    init::xavier_uniform_(t, 1.0);

    const float* data = t.data_ptr<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < t.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }

    // Xavier: a = sqrt(6 / (fan_in + fan_out)) ≈ 0.25
    assert(max_abs < 0.5f);
    assert(max_abs > 0.05f);
}

// ============================================================================
// Utility Tests
// ============================================================================

TEST(count_parameters) {
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(10, 20));  // 10*20 + 20 = 220
    model->add(std::make_shared<Linear>(20, 5));   // 20*5 + 5 = 105

    int64_t total = count_parameters(*model);
    assert(total == 325);
}

TEST(freeze_unfreeze) {
    auto linear = std::make_shared<Linear>(10, 5);

    // All params should require grad by default
    for (auto* param : linear->parameters()) {
        assert(param->requires_grad());
    }

    freeze(*linear);
    for (auto* param : linear->parameters()) {
        assert(!param->requires_grad());
    }

    unfreeze(*linear);
    for (auto* param : linear->parameters()) {
        assert(param->requires_grad());
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PromeTorch Neural Network Module Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Module Base Tests
    RUN_TEST(module_parameter_registration);
    RUN_TEST(module_train_eval_mode);
    RUN_TEST(module_state_dict);
    RUN_TEST(module_zero_grad);

    // Container Tests
    RUN_TEST(sequential_forward);
    RUN_TEST(sequential_parameters);
    RUN_TEST(module_list);
    RUN_TEST(module_dict);

    // Linear Layer Tests
    RUN_TEST(linear_forward);
    RUN_TEST(linear_no_bias);
    RUN_TEST(linear_batch);
    RUN_TEST(identity_forward);
    RUN_TEST(bilinear_forward);

    // Activation Tests
    RUN_TEST(relu_forward);
    RUN_TEST(leaky_relu_forward);
    RUN_TEST(sigmoid_forward);
    RUN_TEST(tanh_forward);
    RUN_TEST(softmax_forward);
    RUN_TEST(gelu_forward);
    RUN_TEST(silu_forward);
    RUN_TEST(mish_forward);

    // Convolution Tests
    RUN_TEST(conv1d_forward);
    RUN_TEST(conv2d_forward);
    RUN_TEST(conv2d_stride);
    RUN_TEST(conv3d_forward);
    RUN_TEST(conv_transpose2d_forward);

    // Pooling Tests
    RUN_TEST(maxpool2d_forward);
    RUN_TEST(avgpool2d_forward);
    RUN_TEST(adaptive_avgpool2d_forward);
    RUN_TEST(global_avgpool2d_forward);

    // Normalization Tests
    RUN_TEST(batchnorm1d_forward);
    RUN_TEST(batchnorm2d_forward);
    RUN_TEST(batchnorm2d_running_stats);
    RUN_TEST(layernorm_forward);
    RUN_TEST(groupnorm_forward);
    RUN_TEST(instancenorm2d_forward);

    // Dropout Tests
    RUN_TEST(dropout_train_vs_eval);
    RUN_TEST(dropout2d_forward);

    // Embedding Tests
    RUN_TEST(embedding_forward);
    RUN_TEST(embedding_padding_idx);
    RUN_TEST(embeddingbag_forward);

    // Loss Function Tests
    RUN_TEST(mse_loss_forward);
    RUN_TEST(l1_loss_forward);
    RUN_TEST(bce_loss_forward);
    RUN_TEST(cross_entropy_loss_forward);
    RUN_TEST(nll_loss_forward);
    RUN_TEST(smooth_l1_loss_forward);
    RUN_TEST(kl_div_loss_forward);
    RUN_TEST(triplet_margin_loss_forward);
    RUN_TEST(focal_loss_forward);
    RUN_TEST(dice_loss_forward);

    // Functional API Tests
    RUN_TEST(functional_relu);
    RUN_TEST(functional_softmax);
    RUN_TEST(functional_linear);
    RUN_TEST(functional_mse_loss);
    RUN_TEST(functional_max_pool2d);
    RUN_TEST(functional_dropout);

    // Init Tests
    RUN_TEST(init_zeros);
    RUN_TEST(init_ones);
    RUN_TEST(init_constant);
    RUN_TEST(init_uniform);
    RUN_TEST(init_normal);
    RUN_TEST(init_kaiming_uniform);
    RUN_TEST(init_xavier_uniform);

    // Utility Tests
    RUN_TEST(count_parameters);
    RUN_TEST(freeze_unfreeze);

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
