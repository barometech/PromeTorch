// ============================================================================
// PromeTorch: Comprehensive NN Modules Test Suite
// ============================================================================
// Tests for all nn::Module subclasses: activations, linear, conv, pooling,
// normalization, dropout, loss, containers, embedding, and RNN modules.

#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"

using namespace at;
using namespace torch;
using namespace torch::nn;

// ============================================================================
// Helper: create a known tensor from initializer list
// ============================================================================
static Tensor make_tensor(std::initializer_list<float> vals, std::vector<int64_t> shape) {
    Tensor t = at::empty(shape);
    float* data = t.mutable_data_ptr<float>();
    int i = 0;
    for (float v : vals) {
        data[i++] = v;
    }
    return t;
}

// ============================================================================
// 1. Activation Modules (10 tests)
// ============================================================================

TEST(NNModulesActivation, ReLU) {
    ReLU relu;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = relu.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 1.0f);
}

TEST(NNModulesActivation, LeakyReLU) {
    LeakyReLU lrelu(0.1);
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = lrelu.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    EXPECT_NEAR(d[0], -0.1f, 1e-6);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 1.0f);
}

TEST(NNModulesActivation, GELU) {
    GELU gelu;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = gelu.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // GELU(0) = 0
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    // GELU(1) ~= 0.8413
    EXPECT_NEAR(d[2], 0.8413f, 0.01f);
    // GELU(-1) ~= -0.1587
    EXPECT_NEAR(d[0], -0.1587f, 0.01f);
}

TEST(NNModulesActivation, Sigmoid) {
    Sigmoid sig;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = sig.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // sigmoid(0) = 0.5
    EXPECT_NEAR(d[1], 0.5f, 1e-5);
    // sigmoid values are in (0, 1)
    EXPECT_GT(d[0], 0.0f);
    EXPECT_LT(d[0], 0.5f);
    EXPECT_GT(d[2], 0.5f);
    EXPECT_LT(d[2], 1.0f);
}

TEST(NNModulesActivation, Tanh) {
    Tanh th;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = th.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // tanh(0) = 0
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    // tanh values are in (-1, 1)
    EXPECT_NEAR(d[0], std::tanh(-1.0f), 1e-5);
    EXPECT_NEAR(d[2], std::tanh(1.0f), 1e-5);
}

TEST(NNModulesActivation, Softmax) {
    Softmax sm(-1);
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor output = sm.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // All values positive
    for (int i = 0; i < 3; ++i) {
        EXPECT_GT(d[i], 0.0f);
    }
    // Sum should be ~1.0
    float sum = d[0] + d[1] + d[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5);
    // Values should be monotonically increasing
    EXPECT_LT(d[0], d[1]);
    EXPECT_LT(d[1], d[2]);
}

TEST(NNModulesActivation, ELU) {
    ELU elu(1.0);
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = elu.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // ELU(x) = x for x > 0, alpha * (exp(x) - 1) for x <= 0
    EXPECT_NEAR(d[0], 1.0f * (std::exp(-1.0f) - 1.0f), 1e-5);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 1.0f);
}

TEST(NNModulesActivation, SELU) {
    SELU selu;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = selu.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;
    // SELU(0) = 0
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    // SELU(1) = scale * 1
    EXPECT_NEAR(d[2], scale, 1e-5);
    // SELU(-1) = scale * alpha * (exp(-1) - 1)
    EXPECT_NEAR(d[0], scale * alpha * (std::exp(-1.0f) - 1.0f), 1e-4);
}

TEST(NNModulesActivation, SiLU) {
    SiLU silu;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = silu.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // SiLU(0) = 0 * sigmoid(0) = 0
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    // SiLU(1) = 1 * sigmoid(1) ~= 0.7311
    EXPECT_NEAR(d[2], 1.0f / (1.0f + std::exp(-1.0f)), 1e-4);
    // SiLU(-1) = -1 * sigmoid(-1) ~= -0.2689
    EXPECT_NEAR(d[0], -1.0f / (1.0f + std::exp(1.0f)), 1e-4);
}

TEST(NNModulesActivation, Mish) {
    Mish mish;
    Tensor input = make_tensor({-1.0f, 0.0f, 1.0f}, {1, 3});
    Tensor output = mish.forward(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    // Mish(0) = 0 * tanh(ln(1+1)) = 0 * tanh(ln2) = 0
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    // Mish(x) = x * tanh(softplus(x))
    float sp1 = std::log(1.0f + std::exp(1.0f));
    EXPECT_NEAR(d[2], 1.0f * std::tanh(sp1), 1e-4);
}

// ============================================================================
// 2. Linear Layers (5 tests)
// ============================================================================

TEST(NNModulesLinear, LinearBasic) {
    Linear linear(10, 5);
    auto params = linear.parameters();
    // weight + bias = 2 parameters
    EXPECT_EQ(params.size(), 2u);

    // Forward: [B=4, 10] -> [B=4, 5]
    Tensor input = at::randn({4, 10});
    Tensor output = linear.forward(input);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 4);
    EXPECT_EQ(output.size(1), 5);
}

TEST(NNModulesLinear, LinearNoBias) {
    Linear linear(8, 4, /*bias=*/false);
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 1u); // weight only
}

TEST(NNModulesLinear, Bilinear) {
    Bilinear bilinear(5, 6, 3);
    Tensor x1 = at::randn({2, 5});
    Tensor x2 = at::randn({2, 6});
    Tensor output = bilinear.forward(x1, x2);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 3);
}

TEST(NNModulesLinear, LazyLinear) {
    LazyLinear lazy(7);
    Tensor input = at::randn({3, 12});
    Tensor output = lazy.forward(input);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 3);
    EXPECT_EQ(output.size(1), 7);
    // After first forward, parameters should be initialized
    auto params = lazy.parameters();
    EXPECT_GE(params.size(), 1u);
}

TEST(NNModulesLinear, FlattenAndIdentity) {
    // Flatten
    Flatten flatten;
    Tensor input = at::randn({2, 3, 4});
    Tensor output = flatten.forward(input);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 12); // 3*4

    // Identity
    Identity identity;
    Tensor id_out = identity.forward(input);
    EXPECT_EQ(id_out.sizes(), input.sizes());
    // Same data
    const float* in_d = input.data_ptr<float>();
    const float* out_d = id_out.data_ptr<float>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_d[i], out_d[i]);
    }
}

// ============================================================================
// 3. Convolutional Modules (3 tests)
// ============================================================================

TEST(NNModulesConv, Conv1d) {
    Conv1d conv(3, 8, /*kernel_size=*/3, /*stride=*/1, /*padding=*/1);
    // Input: [N=2, C_in=3, L=16]
    Tensor input = at::randn({2, 3, 16});
    Tensor output = conv.forward(input);
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 8);
    // L_out = (16 + 2*1 - 1*(3-1) - 1)/1 + 1 = 16
    EXPECT_EQ(output.size(2), 16);
}

TEST(NNModulesConv, Conv2d) {
    Conv2d conv(3, 16, /*kernel_size=*/3, /*stride=*/1, /*padding=*/1);
    // Input: [N=2, C_in=3, H=8, W=8]
    Tensor input = at::randn({2, 3, 8, 8});
    Tensor output = conv.forward(input);
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 16);
    // H_out = (8 + 2*1 - 1*(3-1) - 1)/1 + 1 = 8
    EXPECT_EQ(output.size(2), 8);
    EXPECT_EQ(output.size(3), 8);
}

TEST(NNModulesConv, Conv3d) {
    Conv3d conv(2, 4, /*kernel_size=*/3, /*stride=*/1, /*padding=*/1);
    // Input: [N=1, C_in=2, D=4, H=4, W=4]
    Tensor input = at::randn({1, 2, 4, 4, 4});
    Tensor output = conv.forward(input);
    EXPECT_EQ(output.dim(), 5);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 4);
    // D_out = (4 + 2*1 - 1*(3-1) - 1)/1 + 1 = 4
    EXPECT_EQ(output.size(2), 4);
    EXPECT_EQ(output.size(3), 4);
    EXPECT_EQ(output.size(4), 4);
}

// ============================================================================
// 4. Pooling Modules (4 tests)
// ============================================================================

TEST(NNModulesPooling, MaxPool2d) {
    MaxPool2d pool(2); // kernel_size=2, stride=2 (default)
    // Input: [N=1, C=1, H=4, W=4]
    Tensor input = at::randn({1, 1, 4, 4});
    Tensor output = pool.forward(input);
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 1);
    EXPECT_EQ(output.size(2), 2);
    EXPECT_EQ(output.size(3), 2);
}

TEST(NNModulesPooling, AvgPool2d) {
    AvgPool2d pool(2); // kernel_size=2, stride=2
    Tensor input = at::ones({1, 1, 4, 4});
    Tensor output = pool.forward(input);
    EXPECT_EQ(output.size(2), 2);
    EXPECT_EQ(output.size(3), 2);
    // All ones -> average is 1
    const float* d = output.data_ptr<float>();
    for (int64_t i = 0; i < output.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f, 1e-5);
    }
}

TEST(NNModulesPooling, AdaptiveAvgPool2d) {
    AdaptiveAvgPool2d pool(1);
    // Input: [N=2, C=3, H=8, W=8]
    Tensor input = at::randn({2, 3, 8, 8});
    Tensor output = pool.forward(input);
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 3);
    EXPECT_EQ(output.size(2), 1);
    EXPECT_EQ(output.size(3), 1);
}

TEST(NNModulesPooling, GlobalAvgPool2d) {
    GlobalAvgPool2d gap;
    Tensor input = at::ones({2, 4, 6, 6});
    Tensor output = gap.forward(input);
    EXPECT_EQ(output.size(2), 1);
    EXPECT_EQ(output.size(3), 1);
    // Average of all ones is 1
    const float* d = output.data_ptr<float>();
    for (int64_t i = 0; i < output.numel(); ++i) {
        EXPECT_NEAR(d[i], 1.0f, 1e-5);
    }
}

// ============================================================================
// 5. Normalization Modules (5 tests)
// ============================================================================

TEST(NNModulesNorm, BatchNorm1d) {
    BatchNorm1d bn(8);
    bn.train();
    // Input: [N=4, C=8]
    Tensor input = at::randn({4, 8});
    Tensor output = bn.forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());

    // Check that running_mean buffer exists
    auto* rm = bn.get_buffer("running_mean");
    ASSERT_NE(rm, nullptr);
    EXPECT_EQ(rm->data().size(0), 8);
}

TEST(NNModulesNorm, BatchNorm2d) {
    BatchNorm2d bn(16);
    bn.train();
    // Input: [N=2, C=16, H=4, W=4]
    Tensor input = at::randn({2, 16, 4, 4});
    Tensor output = bn.forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());

    // Check running_var buffer
    auto* rv = bn.get_buffer("running_var");
    ASSERT_NE(rv, nullptr);
    EXPECT_EQ(rv->data().size(0), 16);
}

TEST(NNModulesNorm, LayerNorm) {
    LayerNorm ln({10});
    Tensor input = at::randn({4, 10});
    Tensor output = ln.forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());

    // After layer norm, each row should have mean ~0, var ~1
    const float* d = output.data_ptr<float>();
    for (int64_t b = 0; b < 4; ++b) {
        float mean = 0.0f;
        for (int64_t i = 0; i < 10; ++i) {
            mean += d[b * 10 + i];
        }
        mean /= 10.0f;
        EXPECT_NEAR(mean, 0.0f, 0.1f);
    }
}

TEST(NNModulesNorm, GroupNorm) {
    GroupNorm gn(4, 16); // 4 groups, 16 channels
    // Input: [N=2, C=16, H=4, W=4]
    Tensor input = at::randn({2, 16, 4, 4});
    Tensor output = gn.forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
    // Has weight parameter
    auto params = gn.parameters();
    EXPECT_GE(params.size(), 1u);
}

TEST(NNModulesNorm, InstanceNorm2d) {
    InstanceNorm2d in2d(8);
    // Input: [N=2, C=8, H=4, W=4]
    Tensor input = at::randn({2, 8, 4, 4});
    Tensor output = in2d.forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

// ============================================================================
// 6. Dropout (2 tests)
// ============================================================================

TEST(NNModulesDropout, DropoutTraining) {
    Dropout dropout(0.5);
    dropout.train();
    Tensor input = at::ones({100, 100});
    Tensor output = dropout.forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());

    // Count zeros: roughly 50% should be zero
    const float* d = output.data_ptr<float>();
    int64_t n = output.numel();
    int64_t zeros = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (d[i] == 0.0f) ++zeros;
    }
    double zero_ratio = static_cast<double>(zeros) / n;
    // Should be roughly 0.5 (allow wide margin for randomness)
    EXPECT_GT(zero_ratio, 0.3);
    EXPECT_LT(zero_ratio, 0.7);
}

TEST(NNModulesDropout, DropoutEval) {
    Dropout dropout(0.5);
    dropout.eval();
    Tensor input = at::ones({10, 10});
    Tensor output = dropout.forward(input);
    // In eval mode, dropout returns the same tensor
    const float* in_d = input.data_ptr<float>();
    const float* out_d = output.data_ptr<float>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_d[i], out_d[i]);
    }
}

// ============================================================================
// 7. Loss Modules (8 tests)
// ============================================================================

TEST(NNModulesLoss, MSELoss) {
    MSELoss loss;
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor l = loss.forward(input, target);
    // Same input and target -> loss = 0
    EXPECT_NEAR(l.data_ptr<float>()[0], 0.0f, 1e-6);

    // Different values
    Tensor target2 = make_tensor({2.0f, 3.0f, 4.0f}, {3});
    Tensor l2 = loss.forward(input, target2);
    // Each diff is 1, squared = 1, mean = 1
    EXPECT_NEAR(l2.data_ptr<float>()[0], 1.0f, 1e-5);
}

TEST(NNModulesLoss, L1Loss) {
    L1Loss loss;
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({2.0f, 3.0f, 4.0f}, {3});
    Tensor l = loss.forward(input, target);
    // Each diff is 1, mean = 1
    EXPECT_NEAR(l.data_ptr<float>()[0], 1.0f, 1e-5);
}

TEST(NNModulesLoss, CrossEntropyLoss) {
    CrossEntropyLoss loss;
    // Input: logits [N=2, C=3]
    Tensor input = make_tensor({
        2.0f, 1.0f, 0.1f,
        0.5f, 2.0f, 0.3f
    }, {2, 3});
    // Target: class indices [N=2]
    Tensor target = make_tensor({0.0f, 1.0f}, {2});
    Tensor l = loss.forward(input, target);
    // Loss should be positive scalar
    EXPECT_GT(l.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(l.numel(), 1);
}

TEST(NNModulesLoss, BCELoss) {
    BCELoss loss;
    // Input: probabilities in [0, 1]
    Tensor input = make_tensor({0.8f, 0.2f, 0.6f}, {3});
    Tensor target = make_tensor({1.0f, 0.0f, 1.0f}, {3});
    Tensor l = loss.forward(input, target);
    EXPECT_GT(l.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(l.numel(), 1);
}

TEST(NNModulesLoss, BCEWithLogitsLoss) {
    BCEWithLogitsLoss loss;
    Tensor input = make_tensor({2.0f, -1.0f, 0.5f}, {3});
    Tensor target = make_tensor({1.0f, 0.0f, 1.0f}, {3});
    Tensor l = loss.forward(input, target);
    EXPECT_GT(l.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(l.numel(), 1);
}

TEST(NNModulesLoss, NLLLoss) {
    NLLLoss loss;
    // Input: log-probabilities [N=2, C=3]
    Tensor input = make_tensor({
        -0.5f, -1.5f, -2.5f,
        -2.0f, -0.3f, -1.0f
    }, {2, 3});
    Tensor target = make_tensor({0.0f, 1.0f}, {2});
    Tensor l = loss.forward(input, target);
    // NLL = -((-0.5) + (-0.3)) / 2 = 0.4
    EXPECT_NEAR(l.data_ptr<float>()[0], 0.4f, 1e-5);
}

TEST(NNModulesLoss, SmoothL1Loss) {
    SmoothL1Loss loss;
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({1.5f, 2.5f, 3.5f}, {3});
    Tensor l = loss.forward(input, target);
    // Each diff = 0.5, < beta=1.0, so: 0.5 * 0.25 / 1.0 = 0.125 each, mean = 0.125
    EXPECT_NEAR(l.data_ptr<float>()[0], 0.125f, 1e-4);
}

TEST(NNModulesLoss, HuberLoss) {
    HuberLoss loss;
    Tensor input = make_tensor({0.0f, 1.0f, 5.0f}, {3});
    Tensor target = make_tensor({0.0f, 0.0f, 0.0f}, {3});
    Tensor l = loss.forward(input, target);
    // Loss > 0 for non-matching
    EXPECT_GT(l.data_ptr<float>()[0], 0.0f);
    EXPECT_EQ(l.numel(), 1);
}

// ============================================================================
// 8. Container Modules (3 tests)
// ============================================================================

TEST(NNModulesContainer, Sequential) {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(10, 5));
    seq->add(std::make_shared<ReLU>());
    seq->add(std::make_shared<Linear>(5, 2));

    EXPECT_EQ(seq->size(), 3u);

    Tensor input = at::randn({4, 10});
    Tensor output = seq->forward(input);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 4);
    EXPECT_EQ(output.size(1), 2);

    // Should have parameters from both Linear layers
    auto params = seq->parameters();
    // 2 Linear layers * 2 params (weight+bias) = 4
    EXPECT_EQ(params.size(), 4u);
}

TEST(NNModulesContainer, ModuleList) {
    ModuleList ml;
    ml.append(std::make_shared<Linear>(5, 3));
    ml.append(std::make_shared<Linear>(3, 2));
    EXPECT_EQ(ml.size(), 2u);
    EXPECT_FALSE(ml.empty());

    // Access by index
    auto m0 = ml[0];
    ASSERT_NE(m0, nullptr);

    // Parameters from all children
    auto params = ml.parameters();
    EXPECT_EQ(params.size(), 4u); // 2 linears * (weight + bias)
}

TEST(NNModulesContainer, ModuleDict) {
    ModuleDict md;
    md.insert("encoder", std::make_shared<Linear>(10, 5));
    md.insert("decoder", std::make_shared<Linear>(5, 10));
    EXPECT_EQ(md.size(), 2u);
    EXPECT_TRUE(md.contains("encoder"));
    EXPECT_TRUE(md.contains("decoder"));

    auto keys = md.keys();
    EXPECT_EQ(keys.size(), 2u);

    auto enc = md["encoder"];
    ASSERT_NE(enc, nullptr);
}

// ============================================================================
// 9. Embedding Modules (2 tests)
// ============================================================================

TEST(NNModulesEmbedding, EmbeddingLookup) {
    Embedding emb(10, 4); // 10 words, 4-dim embeddings
    // Lookup indices [0, 5, 3]
    Tensor indices = make_tensor({0.0f, 5.0f, 3.0f}, {3});
    Tensor output = emb.forward(indices);
    // Output shape: [3, 4]
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 3);
    EXPECT_EQ(output.size(1), 4);
}

TEST(NNModulesEmbedding, EmbeddingBag2D) {
    EmbeddingBag ebag(10, 4);
    // 2D input: [batch=2, seq_len=3]
    Tensor indices = make_tensor({
        0.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 5.0f
    }, {2, 3});
    Tensor output = ebag.forward(indices);
    // Output: [2, 4]
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 4);
}

// ============================================================================
// 10. RNN Modules (3 tests)
// ============================================================================

TEST(NNModulesRNN, RNNBasic) {
    // RNN(input_size=5, hidden_size=10, num_layers=1)
    RNN rnn(5, 10, /*num_layers=*/1, /*bias=*/true, /*batch_first=*/true);
    // Input: [batch=2, seq_len=3, features=5] (batch_first)
    Tensor input = at::randn({2, 3, 5});
    auto [output, h_n] = rnn.forward_rnn(input);
    // Output: [batch=2, seq_len=3, hidden=10]
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 3);
    EXPECT_EQ(output.size(2), 10);
    // h_n: [num_layers=1, batch=2, hidden=10]
    EXPECT_EQ(h_n.dim(), 3);
    EXPECT_EQ(h_n.size(0), 1);
    EXPECT_EQ(h_n.size(1), 2);
    EXPECT_EQ(h_n.size(2), 10);
}

TEST(NNModulesRNN, LSTMBasic) {
    LSTM lstm(5, 10, /*num_layers=*/1, /*bias=*/true, /*batch_first=*/true);
    Tensor input = at::randn({2, 3, 5});
    auto result = lstm.forward_lstm(input);
    // Output: [batch=2, seq_len=3, hidden=10]
    EXPECT_EQ(result.output.dim(), 3);
    EXPECT_EQ(result.output.size(0), 2);
    EXPECT_EQ(result.output.size(1), 3);
    EXPECT_EQ(result.output.size(2), 10);
    // h_n: [1, 2, 10]
    EXPECT_EQ(result.h_n.size(0), 1);
    EXPECT_EQ(result.h_n.size(1), 2);
    EXPECT_EQ(result.h_n.size(2), 10);
    // c_n: [1, 2, 10]
    EXPECT_EQ(result.c_n.size(0), 1);
    EXPECT_EQ(result.c_n.size(1), 2);
    EXPECT_EQ(result.c_n.size(2), 10);
}

TEST(NNModulesRNN, GRUBasic) {
    GRU gru(5, 10, /*num_layers=*/1, /*bias=*/true, /*batch_first=*/true);
    Tensor input = at::randn({2, 3, 5});
    auto [output, h_n] = gru.forward_gru(input);
    // Output: [batch=2, seq_len=3, hidden=10]
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 3);
    EXPECT_EQ(output.size(2), 10);
    // h_n: [1, 2, 10]
    EXPECT_EQ(h_n.size(0), 1);
    EXPECT_EQ(h_n.size(1), 2);
    EXPECT_EQ(h_n.size(2), 10);
}

// ============================================================================
// Bonus: Module API Tests
// ============================================================================

TEST(NNModulesAPI, TrainEvalMode) {
    Linear linear(10, 5);
    linear.train();
    EXPECT_TRUE(linear.is_training());
    linear.eval();
    EXPECT_FALSE(linear.is_training());
}

TEST(NNModulesAPI, ZeroGrad) {
    Linear linear(4, 2);
    // Set some gradient
    auto params = linear.parameters();
    for (auto* p : params) {
        Tensor grad = at::ones(p->data().sizes());
        p->data().set_grad(grad);
    }
    linear.zero_grad();
    for (auto* p : params) {
        Tensor g = p->data().grad();
        if (g.defined()) {
            const float* d = g.data_ptr<float>();
            for (int64_t i = 0; i < g.numel(); ++i) {
                EXPECT_FLOAT_EQ(d[i], 0.0f);
            }
        }
    }
}

TEST(NNModulesAPI, StateDict) {
    Linear linear(4, 2);
    auto sd = linear.state_dict();
    // Should contain "weight" and "bias"
    EXPECT_TRUE(sd.find("weight") != sd.end());
    EXPECT_TRUE(sd.find("bias") != sd.end());
}

TEST(NNModulesAPI, CountParameters) {
    Linear linear(10, 5);
    int64_t count = count_parameters(linear);
    // weight: 10*5=50, bias: 5, total=55
    EXPECT_EQ(count, 55);
}
