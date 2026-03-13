// ============================================================================
// PromeTorch: Comprehensive nn::functional Test Suite
// ============================================================================
// Tests for all functions in torch::nn::functional namespace: activations,
// softmax, losses, spatial ops, one-hot, normalize, grid_sample, batch/layer
// norm functional, and dropout.

#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"
#include "torch/nn/functional.h"

using namespace at;
using namespace torch;
namespace F = torch::nn::functional;

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
// 1. Activation Functions (10 tests)
// ============================================================================

TEST(NNFunctionalActivation, ReLU) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::relu(input);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 0.0f);
    EXPECT_FLOAT_EQ(d[3], 1.0f);
    EXPECT_FLOAT_EQ(d[4], 2.0f);
}

TEST(NNFunctionalActivation, ReLU6) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 3.0f, 7.0f}, {5});
    Tensor output = F::relu6(input);
    const float* d = output.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 0.0f);
    EXPECT_FLOAT_EQ(d[3], 3.0f);
    EXPECT_FLOAT_EQ(d[4], 6.0f); // clamped at 6
}

TEST(NNFunctionalActivation, LeakyReLU) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::leaky_relu(input, 0.1);
    const float* d = output.data_ptr<float>();
    EXPECT_NEAR(d[0], -0.2f, 1e-6);
    EXPECT_NEAR(d[1], -0.1f, 1e-6);
    EXPECT_FLOAT_EQ(d[2], 0.0f);
    EXPECT_FLOAT_EQ(d[3], 1.0f);
    EXPECT_FLOAT_EQ(d[4], 2.0f);
}

TEST(NNFunctionalActivation, ELU) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::elu(input, 1.0);
    const float* d = output.data_ptr<float>();
    // For x > 0: elu(x) = x
    EXPECT_FLOAT_EQ(d[3], 1.0f);
    EXPECT_FLOAT_EQ(d[4], 2.0f);
    // For x <= 0: elu(x) = alpha * (exp(x) - 1)
    EXPECT_NEAR(d[0], std::exp(-2.0f) - 1.0f, 1e-5);
    EXPECT_NEAR(d[1], std::exp(-1.0f) - 1.0f, 1e-5);
    EXPECT_NEAR(d[2], 0.0f, 1e-5);
}

TEST(NNFunctionalActivation, SELU) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::selu(input);
    const float* d = output.data_ptr<float>();
    constexpr float ALPHA = 1.6732632423543772848170429916717f;
    constexpr float SCALE = 1.0507009873554804934193349852946f;
    // x > 0: scale * x
    EXPECT_NEAR(d[3], SCALE * 1.0f, 1e-5);
    EXPECT_NEAR(d[4], SCALE * 2.0f, 1e-5);
    // x == 0: 0
    EXPECT_NEAR(d[2], 0.0f, 1e-5);
    // x < 0: scale * alpha * (exp(x) - 1)
    EXPECT_NEAR(d[1], SCALE * ALPHA * (std::exp(-1.0f) - 1.0f), 1e-4);
}

TEST(NNFunctionalActivation, GELU) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::gelu(input);
    const float* d = output.data_ptr<float>();
    // GELU(0) = 0
    EXPECT_NEAR(d[2], 0.0f, 1e-5);
    // GELU(x) is approximately x for large positive x
    EXPECT_NEAR(d[4], 2.0f * 0.5f * (1.0f + std::erf(2.0f / std::sqrt(2.0f))), 1e-4);
    // GELU is odd-ish: GELU(-x) approaches 0 for large negative x
    EXPECT_LT(d[0], 0.0f);
}

TEST(NNFunctionalActivation, FunctionalSigmoid) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::sigmoid(input);
    const float* d = output.data_ptr<float>();
    // sigmoid(0) = 0.5
    EXPECT_NEAR(d[2], 0.5f, 1e-5);
    // All values in (0, 1)
    for (int i = 0; i < 5; ++i) {
        EXPECT_GT(d[i], 0.0f);
        EXPECT_LT(d[i], 1.0f);
    }
    // Monotonically increasing
    for (int i = 0; i < 4; ++i) {
        EXPECT_LT(d[i], d[i + 1]);
    }
}

TEST(NNFunctionalActivation, FunctionalTanh) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::tanh(input);
    const float* d = output.data_ptr<float>();
    // tanh(0) = 0
    EXPECT_NEAR(d[2], 0.0f, 1e-5);
    // All values in (-1, 1)
    for (int i = 0; i < 5; ++i) {
        EXPECT_GT(d[i], -1.0f);
        EXPECT_LT(d[i], 1.0f);
    }
    // Antisymmetric: tanh(-x) = -tanh(x)
    EXPECT_NEAR(d[0], -d[4], 1e-5);
    EXPECT_NEAR(d[1], -d[3], 1e-5);
}

TEST(NNFunctionalActivation, SiLU) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::silu(input);
    const float* d = output.data_ptr<float>();
    // SiLU(0) = 0 * sigmoid(0) = 0
    EXPECT_NEAR(d[2], 0.0f, 1e-5);
    // SiLU(x) = x * sigmoid(x)
    for (int i = 0; i < 5; ++i) {
        float x = -2.0f + i;
        float expected = x / (1.0f + std::exp(-x));
        EXPECT_NEAR(d[i], expected, 1e-5);
    }
}

TEST(NNFunctionalActivation, Mish) {
    Tensor input = make_tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
    Tensor output = F::mish(input);
    const float* d = output.data_ptr<float>();
    // Mish(0) = 0 * tanh(ln(2)) = 0
    EXPECT_NEAR(d[2], 0.0f, 1e-5);
    // Mish(x) = x * tanh(softplus(x))
    for (int i = 0; i < 5; ++i) {
        float x = -2.0f + i;
        float sp = std::log(1.0f + std::exp(x));
        float expected = x * std::tanh(sp);
        EXPECT_NEAR(d[i], expected, 1e-4);
    }
}

// ============================================================================
// 2. Softmax and LogSoftmax (2 tests)
// ============================================================================

TEST(NNFunctionalSoftmax, SoftmaxSumOne) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    Tensor output = F::softmax(input, /*dim=*/1);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();

    // All positive
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(d[i], 0.0f);
    }

    // Sum should be 1.0
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += d[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5);

    // Values should be monotonically increasing (since input is)
    for (int i = 0; i < 3; ++i) {
        EXPECT_LT(d[i], d[i + 1]);
    }
}

TEST(NNFunctionalSoftmax, LogSoftmaxNonPositive) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor output = F::log_softmax(input, /*dim=*/1);
    ASSERT_EQ(output.sizes(), input.sizes());
    const float* d = output.data_ptr<float>();

    // All log_softmax values should be <= 0
    for (int i = 0; i < 3; ++i) {
        EXPECT_LE(d[i], 0.0f + 1e-6);
    }

    // exp(log_softmax) should sum to 1
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += std::exp(d[i]);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4);
}

// ============================================================================
// 3. Loss Functions (5 tests)
// ============================================================================

TEST(NNFunctionalLoss, CrossEntropy) {
    // Input: logits [N=2, C=3]
    Tensor input = make_tensor({
        2.0f, 1.0f, 0.1f,
        0.5f, 2.0f, 0.3f
    }, {2, 3});
    Tensor target = make_tensor({0.0f, 1.0f}, {2});

    Tensor loss = F::cross_entropy(input, target);
    EXPECT_EQ(loss.numel(), 1);
    EXPECT_GT(loss.data_ptr<float>()[0], 0.0f);
}

TEST(NNFunctionalLoss, MSELoss) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({1.5f, 2.5f, 3.5f}, {3});

    Tensor loss = F::mse_loss(input, target);
    EXPECT_EQ(loss.numel(), 1);
    // Each diff = 0.5, squared = 0.25, mean = 0.25
    EXPECT_NEAR(loss.data_ptr<float>()[0], 0.25f, 1e-5);
}

TEST(NNFunctionalLoss, L1Loss) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({1.5f, 2.5f, 3.5f}, {3});

    Tensor loss = F::l1_loss(input, target);
    EXPECT_EQ(loss.numel(), 1);
    // Each diff = 0.5, mean = 0.5
    EXPECT_NEAR(loss.data_ptr<float>()[0], 0.5f, 1e-5);
}

TEST(NNFunctionalLoss, BinaryCrossEntropy) {
    // Input must be probabilities in [0, 1]
    Tensor input = make_tensor({0.9f, 0.1f, 0.8f}, {3});
    Tensor target = make_tensor({1.0f, 0.0f, 1.0f}, {3});

    Tensor loss = F::binary_cross_entropy(input, target);
    EXPECT_EQ(loss.numel(), 1);
    EXPECT_GT(loss.data_ptr<float>()[0], 0.0f);

    // Perfect prediction should have near-zero loss
    Tensor perfect_in = make_tensor({0.9999f, 0.0001f}, {2});
    Tensor perfect_tgt = make_tensor({1.0f, 0.0f}, {2});
    Tensor perfect_loss = F::binary_cross_entropy(perfect_in, perfect_tgt);
    EXPECT_LT(perfect_loss.data_ptr<float>()[0], 0.01f);
}

TEST(NNFunctionalLoss, NLLLoss) {
    // Input: log-probabilities [N=3, C=4]
    Tensor input = make_tensor({
        -0.2f, -1.5f, -2.0f, -3.0f,
        -3.0f, -0.1f, -2.0f, -1.0f,
        -2.0f, -2.0f, -0.3f, -2.0f
    }, {3, 4});
    Tensor target = make_tensor({0.0f, 1.0f, 2.0f}, {3});

    Tensor loss = F::nll_loss(input, target);
    EXPECT_EQ(loss.numel(), 1);
    // NLL = -mean(-0.2, -0.1, -0.3) = mean(0.2, 0.1, 0.3) = 0.2
    EXPECT_NEAR(loss.data_ptr<float>()[0], 0.2f, 1e-5);
}

// ============================================================================
// 4. Spatial Functions (4 tests)
// ============================================================================

TEST(NNFunctionalSpatial, PadConstant) {
    // Input: [1, 1, 3, 3]
    Tensor input = at::ones({1, 1, 3, 3});
    // Pad: left=1, right=1, top=1, bottom=1
    Tensor output = F::pad(input, {1, 1, 1, 1}, "constant", 0.0);
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(2), 5); // 3 + 1 + 1
    EXPECT_EQ(output.size(3), 5); // 3 + 1 + 1

    // Corners should be 0 (padding value)
    const float* d = output.data_ptr<float>();
    EXPECT_FLOAT_EQ(d[0], 0.0f); // top-left
    // Center should be 1
    int center = 2 * 5 + 2; // row=2, col=2 in 5x5
    EXPECT_FLOAT_EQ(d[center], 1.0f);
}

TEST(NNFunctionalSpatial, PadReflect) {
    // Input: [1, 1, 4, 4]
    Tensor input = at::randn({1, 1, 4, 4});
    // Pad: left=1, right=1
    Tensor output = F::pad(input, {1, 1}, "reflect");
    EXPECT_EQ(output.size(3), 6); // 4 + 1 + 1
    // Height should remain 4 (only last dim padded)
    EXPECT_EQ(output.size(2), 4);
}

TEST(NNFunctionalSpatial, InterpolateNearest) {
    // Input: [1, 1, 2, 2]
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 2, 2});
    // Upsample 2x
    Tensor output = F::interpolate(input, {4, 4}, {}, "nearest");
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(2), 4);
    EXPECT_EQ(output.size(3), 4);

    // Top-left 2x2 block should all be 1.0
    const float* d = output.data_ptr<float>();
    EXPECT_NEAR(d[0], 1.0f, 1e-5);          // (0,0)
    EXPECT_NEAR(d[1], 1.0f, 1e-5);          // (0,1)
    EXPECT_NEAR(d[4], 1.0f, 1e-5);          // (1,0)
}

TEST(NNFunctionalSpatial, InterpolateBilinear) {
    // Input: [1, 1, 2, 2]
    Tensor input = make_tensor({0.0f, 1.0f, 1.0f, 0.0f}, {1, 1, 2, 2});
    // Upsample to 4x4
    Tensor output = F::interpolate(input, {4, 4}, {}, "bilinear");
    EXPECT_EQ(output.size(2), 4);
    EXPECT_EQ(output.size(3), 4);
    // Output should have smooth interpolation between corners
    const float* d = output.data_ptr<float>();
    // All values should be in [0, 1] range
    for (int64_t i = 0; i < output.numel(); ++i) {
        EXPECT_GE(d[i], -0.01f);
        EXPECT_LE(d[i], 1.01f);
    }
}

// ============================================================================
// 5. One-Hot (1 test)
// ============================================================================

TEST(NNFunctionalOneHot, Basic) {
    Tensor indices = make_tensor({0.0f, 1.0f, 2.0f}, {3});
    Tensor output = F::one_hot(indices, 3);
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 3);
    EXPECT_EQ(output.size(1), 3);

    const float* d = output.data_ptr<float>();
    // Row 0: [1, 0, 0]
    EXPECT_FLOAT_EQ(d[0], 1.0f); EXPECT_FLOAT_EQ(d[1], 0.0f); EXPECT_FLOAT_EQ(d[2], 0.0f);
    // Row 1: [0, 1, 0]
    EXPECT_FLOAT_EQ(d[3], 0.0f); EXPECT_FLOAT_EQ(d[4], 1.0f); EXPECT_FLOAT_EQ(d[5], 0.0f);
    // Row 2: [0, 0, 1]
    EXPECT_FLOAT_EQ(d[6], 0.0f); EXPECT_FLOAT_EQ(d[7], 0.0f); EXPECT_FLOAT_EQ(d[8], 1.0f);
}

// ============================================================================
// 6. Normalize, Cosine Similarity, Pairwise Distance (3 tests)
// ============================================================================

TEST(NNFunctionalNorm, Normalize) {
    // Input: [2, 4] -- normalize along dim=1
    Tensor input = make_tensor({
        3.0f, 0.0f, 4.0f, 0.0f,
        0.0f, 5.0f, 0.0f, 12.0f
    }, {2, 4});
    Tensor output = F::normalize(input, 2.0, 1);
    EXPECT_EQ(output.sizes(), input.sizes());

    const float* d = output.data_ptr<float>();
    // Row 0: norm = sqrt(9+0+16+0) = 5, so [0.6, 0, 0.8, 0]
    EXPECT_NEAR(d[0], 0.6f, 1e-5);
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    EXPECT_NEAR(d[2], 0.8f, 1e-5);
    EXPECT_NEAR(d[3], 0.0f, 1e-5);

    // Row 1: norm = sqrt(0+25+0+144) = 13, so [0, 5/13, 0, 12/13]
    EXPECT_NEAR(d[4], 0.0f, 1e-5);
    EXPECT_NEAR(d[5], 5.0f / 13.0f, 1e-5);
    EXPECT_NEAR(d[6], 0.0f, 1e-5);
    EXPECT_NEAR(d[7], 12.0f / 13.0f, 1e-5);

    // L2 norm of each row should be ~1.0
    float norm0 = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2] + d[3]*d[3]);
    float norm1 = std::sqrt(d[4]*d[4] + d[5]*d[5] + d[6]*d[6] + d[7]*d[7]);
    EXPECT_NEAR(norm0, 1.0f, 1e-5);
    EXPECT_NEAR(norm1, 1.0f, 1e-5);
}

TEST(NNFunctionalNorm, CosineSimilarity) {
    // Parallel vectors -> cosine similarity = 1.0
    Tensor x1 = make_tensor({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor x2 = make_tensor({2.0f, 4.0f, 6.0f}, {1, 3});
    Tensor sim = F::cosine_similarity(x1, x2, /*dim=*/1);
    EXPECT_NEAR(sim.data_ptr<float>()[0], 1.0f, 1e-5);

    // Orthogonal vectors -> cosine similarity = 0.0
    Tensor a = make_tensor({1.0f, 0.0f, 0.0f}, {1, 3});
    Tensor b = make_tensor({0.0f, 1.0f, 0.0f}, {1, 3});
    Tensor sim2 = F::cosine_similarity(a, b, /*dim=*/1);
    EXPECT_NEAR(sim2.data_ptr<float>()[0], 0.0f, 1e-5);

    // Anti-parallel vectors -> cosine similarity = -1.0
    Tensor c = make_tensor({1.0f, 0.0f}, {1, 2});
    Tensor d = make_tensor({-1.0f, 0.0f}, {1, 2});
    Tensor sim3 = F::cosine_similarity(c, d, /*dim=*/1);
    EXPECT_NEAR(sim3.data_ptr<float>()[0], -1.0f, 1e-5);
}

TEST(NNFunctionalNorm, PairwiseDistance) {
    // Same vectors -> distance = 0
    Tensor x1 = make_tensor({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor x2 = make_tensor({1.0f, 2.0f, 3.0f}, {1, 3});
    Tensor dist = F::pairwise_distance(x1, x2);
    EXPECT_NEAR(dist.data_ptr<float>()[0], 0.0f, 1e-4);

    // Known distance: (1,0,0) to (0,0,0) = 1.0
    Tensor a = make_tensor({1.0f, 0.0f, 0.0f}, {1, 3});
    Tensor b = make_tensor({0.0f, 0.0f, 0.0f}, {1, 3});
    Tensor dist2 = F::pairwise_distance(a, b);
    EXPECT_NEAR(dist2.data_ptr<float>()[0], 1.0f, 1e-4);

    // (3, 4, 0) to (0, 0, 0) = 5.0
    Tensor c = make_tensor({3.0f, 4.0f, 0.0f}, {1, 3});
    Tensor d = make_tensor({0.0f, 0.0f, 0.0f}, {1, 3});
    Tensor dist3 = F::pairwise_distance(c, d);
    EXPECT_NEAR(dist3.data_ptr<float>()[0], 5.0f, 1e-4);
}

// ============================================================================
// 7. Grid Sample and Affine Grid (2 tests)
// ============================================================================

TEST(NNFunctionalGrid, AffineGridIdentity) {
    // Identity transformation theta: [[1,0,0],[0,1,0]]
    Tensor theta = make_tensor({
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    }, {1, 2, 3});

    std::vector<int64_t> size = {1, 1, 4, 4}; // N=1, C=1, H=4, W=4
    Tensor grid = F::affine_grid(theta, size, /*align_corners=*/true);

    EXPECT_EQ(grid.dim(), 4);
    EXPECT_EQ(grid.size(0), 1);
    EXPECT_EQ(grid.size(1), 4); // H
    EXPECT_EQ(grid.size(2), 4); // W
    EXPECT_EQ(grid.size(3), 2); // (x, y)

    const float* d = grid.data_ptr<float>();
    // With align_corners=true and identity theta:
    // Corner (0,0) should map to (-1, -1)
    EXPECT_NEAR(d[0], -1.0f, 1e-5); // x at (0,0)
    EXPECT_NEAR(d[1], -1.0f, 1e-5); // y at (0,0)

    // Corner (0, W-1=3) should map to (1, -1)
    int idx_03 = (0 * 4 + 3) * 2;
    EXPECT_NEAR(d[idx_03], 1.0f, 1e-5);     // x at (0,3)
    EXPECT_NEAR(d[idx_03 + 1], -1.0f, 1e-5); // y at (0,3)

    // Corner (H-1=3, 0) should map to (-1, 1)
    int idx_30 = (3 * 4 + 0) * 2;
    EXPECT_NEAR(d[idx_30], -1.0f, 1e-5);    // x at (3,0)
    EXPECT_NEAR(d[idx_30 + 1], 1.0f, 1e-5); // y at (3,0)
}

TEST(NNFunctionalGrid, GridSampleIdentity) {
    // Create a simple 4x4 input with known values
    Tensor input = at::arange(at::Scalar(0), at::Scalar(16)).reshape({1, 1, 4, 4});

    // Identity theta
    Tensor theta = make_tensor({
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    }, {1, 2, 3});

    std::vector<int64_t> size = {1, 1, 4, 4};
    Tensor grid = F::affine_grid(theta, size, /*align_corners=*/true);

    // Grid sample should approximately recover the input
    Tensor output = F::grid_sample(input, grid, "bilinear", "zeros", /*align_corners=*/true);
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 1);
    EXPECT_EQ(output.size(2), 4);
    EXPECT_EQ(output.size(3), 4);

    // Compare values (should be very close to input for identity transform)
    const float* in_d = input.data_ptr<float>();
    const float* out_d = output.data_ptr<float>();
    for (int64_t i = 0; i < 16; ++i) {
        EXPECT_NEAR(out_d[i], in_d[i], 0.5f); // Allow some interpolation error
    }
}

// ============================================================================
// 8. Batch Norm and Layer Norm Functional (2 tests)
// ============================================================================

TEST(NNFunctionalNormalization, BatchNorm) {
    // Input: [N=2, C=3, H=2, W=2]
    Tensor input = at::randn({2, 3, 2, 2});
    Tensor running_mean = at::zeros({3});
    Tensor running_var = at::ones({3});
    Tensor weight = at::ones({3});
    Tensor bias = at::zeros({3});

    Tensor output = F::batch_norm(input, running_mean, running_var, &weight, &bias,
                                   /*training=*/false, /*momentum=*/0.1, /*eps=*/1e-5);
    EXPECT_EQ(output.sizes(), input.sizes());

    // With running_mean=0, running_var=1, weight=1, bias=0:
    // output = (input - 0) / sqrt(1 + eps) * 1 + 0 ~= input
    const float* in_d = input.data_ptr<float>();
    const float* out_d = output.data_ptr<float>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_NEAR(out_d[i], in_d[i], 1e-3);
    }
}

TEST(NNFunctionalNormalization, LayerNorm) {
    // Input: [N=2, features=4]
    Tensor input = make_tensor({
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.0f, 1.0f, 2.0f
    }, {2, 4});

    std::vector<int64_t> normalized_shape = {4};
    Tensor output = F::layer_norm(input, normalized_shape);
    EXPECT_EQ(output.sizes(), input.sizes());

    // Each row should have mean ~0 and var ~1 (without affine)
    const float* d = output.data_ptr<float>();
    for (int64_t b = 0; b < 2; ++b) {
        float mean = 0.0f;
        for (int64_t i = 0; i < 4; ++i) {
            mean += d[b * 4 + i];
        }
        mean /= 4.0f;
        EXPECT_NEAR(mean, 0.0f, 1e-5);

        float var = 0.0f;
        for (int64_t i = 0; i < 4; ++i) {
            float diff = d[b * 4 + i] - mean;
            var += diff * diff;
        }
        var /= 4.0f;
        EXPECT_NEAR(var, 1.0f, 0.1f);
    }
}

// ============================================================================
// 9. Dropout Functional (1 test)
// ============================================================================

TEST(NNFunctionalDropout, EvalModeIdentity) {
    Tensor input = at::randn({10, 10});
    // In eval mode (training=false), dropout should be identity
    Tensor output = F::dropout(input, 0.5, /*training=*/false);

    const float* in_d = input.data_ptr<float>();
    const float* out_d = output.data_ptr<float>();
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_FLOAT_EQ(in_d[i], out_d[i]);
    }
}

// ============================================================================
// Bonus: Additional functional tests
// ============================================================================

TEST(NNFunctionalActivation, Softplus) {
    Tensor input = make_tensor({-2.0f, 0.0f, 2.0f, 20.0f}, {4});
    Tensor output = F::softplus(input);
    const float* d = output.data_ptr<float>();
    // softplus(0) = ln(2) ~= 0.693
    EXPECT_NEAR(d[1], std::log(2.0f), 1e-5);
    // softplus(x) ~= x for large x
    EXPECT_NEAR(d[3], 20.0f, 0.1f);
    // All values should be positive
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(d[i], 0.0f);
    }
}

TEST(NNFunctionalActivation, Softsign) {
    Tensor input = make_tensor({-2.0f, 0.0f, 2.0f}, {3});
    Tensor output = F::softsign(input);
    const float* d = output.data_ptr<float>();
    // softsign(0) = 0
    EXPECT_NEAR(d[1], 0.0f, 1e-5);
    // softsign(x) = x / (1 + |x|)
    EXPECT_NEAR(d[0], -2.0f / 3.0f, 1e-5);
    EXPECT_NEAR(d[2], 2.0f / 3.0f, 1e-5);
}

TEST(NNFunctionalActivation, Hardswish) {
    Tensor input = make_tensor({-4.0f, -3.0f, 0.0f, 3.0f, 4.0f}, {5});
    Tensor output = F::hardswish(input);
    const float* d = output.data_ptr<float>();
    // x <= -3: 0
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    // x == 0: 0 * (0+3)/6 = 0
    EXPECT_FLOAT_EQ(d[2], 0.0f);
    // x >= 3: x
    EXPECT_FLOAT_EQ(d[3], 3.0f);
    EXPECT_FLOAT_EQ(d[4], 4.0f);
}

TEST(NNFunctionalActivation, Hardsigmoid) {
    Tensor input = make_tensor({-4.0f, -3.0f, 0.0f, 3.0f, 4.0f}, {5});
    Tensor output = F::hardsigmoid(input);
    const float* d = output.data_ptr<float>();
    // x <= -3: 0
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    // x == 0: (0+3)/6 = 0.5
    EXPECT_FLOAT_EQ(d[2], 0.5f);
    // x >= 3: 1
    EXPECT_FLOAT_EQ(d[3], 1.0f);
    EXPECT_FLOAT_EQ(d[4], 1.0f);
}

TEST(NNFunctionalLoss, MSELossNoReduction) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({2.0f, 2.0f, 5.0f}, {3});
    Tensor loss = F::mse_loss(input, target, "none");
    EXPECT_EQ(loss.sizes(), input.sizes());
    const float* d = loss.data_ptr<float>();
    EXPECT_NEAR(d[0], 1.0f, 1e-5);  // (1-2)^2 = 1
    EXPECT_NEAR(d[1], 0.0f, 1e-5);  // (2-2)^2 = 0
    EXPECT_NEAR(d[2], 4.0f, 1e-5);  // (3-5)^2 = 4
}

TEST(NNFunctionalLoss, MSELossSumReduction) {
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f}, {3});
    Tensor target = make_tensor({2.0f, 2.0f, 5.0f}, {3});
    Tensor loss = F::mse_loss(input, target, "sum");
    EXPECT_EQ(loss.numel(), 1);
    // 1 + 0 + 4 = 5
    EXPECT_NEAR(loss.data_ptr<float>()[0], 5.0f, 1e-5);
}

TEST(NNFunctionalSpatial, PadCircular) {
    // 1D-like test: [1, 1, 4]
    Tensor input = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 4});
    // Pad left=1, right=1
    Tensor output = F::pad(input, {1, 1}, "circular");
    EXPECT_EQ(output.size(2), 6);
    const float* d = output.data_ptr<float>();
    // Circular: left pad with last element, right pad with first element
    EXPECT_FLOAT_EQ(d[0], 4.0f); // wrapped from end
    EXPECT_FLOAT_EQ(d[1], 1.0f);
    EXPECT_FLOAT_EQ(d[2], 2.0f);
    EXPECT_FLOAT_EQ(d[3], 3.0f);
    EXPECT_FLOAT_EQ(d[4], 4.0f);
    EXPECT_FLOAT_EQ(d[5], 1.0f); // wrapped from beginning
}

TEST(NNFunctionalOneHot, AutoClasses) {
    Tensor indices = make_tensor({0.0f, 3.0f, 1.0f}, {3});
    // Auto-detect num_classes = 4 (max + 1)
    Tensor output = F::one_hot(indices);
    EXPECT_EQ(output.size(0), 3);
    EXPECT_EQ(output.size(1), 4);

    const float* d = output.data_ptr<float>();
    // Row 0: class 0 -> [1,0,0,0]
    EXPECT_FLOAT_EQ(d[0], 1.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    // Row 1: class 3 -> [0,0,0,1]
    EXPECT_FLOAT_EQ(d[4 + 3], 1.0f);
    // Row 2: class 1 -> [0,1,0,0]
    EXPECT_FLOAT_EQ(d[8 + 1], 1.0f);
}
