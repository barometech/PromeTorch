#include <gtest/gtest.h>
#include "torch/csrc/autograd/autograd.h"
#include <cmath>

using namespace at;
using namespace torch;
using namespace torch::autograd;

// ============================================================================
// Helper Functions
// ============================================================================

// Check if two floats are approximately equal
bool approx_equal(float a, float b, float rtol = 1e-4f, float atol = 1e-6f) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

// Check if tensor values are approximately equal
bool tensor_approx_equal(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-6f) {
    if (a.sizes() != b.sizes()) return false;

    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();

    for (int64_t i = 0; i < a.numel(); ++i) {
        if (!approx_equal(a_data[i], b_data[i], rtol, atol)) {
            return false;
        }
    }
    return true;
}

// Get gradient from a tensor
Tensor get_grad(const Tensor& tensor) {
    // Access grad_ from base c10::AutogradMeta (no need to cast to AutogradMetaImpl)
    auto* raw_meta = tensor.autograd_meta();
    if (raw_meta && raw_meta->grad_) {
        return Tensor(raw_meta->grad_);
    }
    return Tensor();
}

// ============================================================================
// Basic Autograd Tests
// ============================================================================

TEST(AutogradTest, RequiresGrad) {
    Tensor x = ones({2, 3});
    EXPECT_FALSE(x.requires_grad());

    x.set_requires_grad(true);
    EXPECT_TRUE(x.requires_grad());
    EXPECT_TRUE(x.is_leaf());
}

TEST(AutogradTest, NegBackward) {
    // y = -x, dy/dx = -1
    Tensor x = randn({3, 4});
    x.set_requires_grad(true);

    Tensor y = neg_autograd(x);
    EXPECT_TRUE(y.requires_grad());

    // Backward with ones
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);
    EXPECT_TRUE(grad_x.defined());

    // Expected gradient is -1
    Tensor expected = full({3, 4}, Scalar(-1.0));
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, AddBackward) {
    // z = x + y, dz/dx = 1, dz/dy = 1
    Tensor x = randn({2, 3});
    Tensor y = randn({2, 3});
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    Tensor z = add_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor grad_x = get_grad(x);
    Tensor grad_y = get_grad(y);

    EXPECT_TRUE(grad_x.defined());
    EXPECT_TRUE(grad_y.defined());

    // Both gradients should be ones
    Tensor expected = ones({2, 3});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
    EXPECT_TRUE(tensor_approx_equal(grad_y, expected));
}

TEST(AutogradTest, MulBackward) {
    // z = x * y, dz/dx = y, dz/dy = x
    Tensor x = tensor({1.0f, 2.0f, 3.0f});
    Tensor y = tensor({4.0f, 5.0f, 6.0f});
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    Tensor z = mul_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor grad_x = get_grad(x);
    Tensor grad_y = get_grad(y);

    // grad_x should be y, grad_y should be x
    EXPECT_TRUE(tensor_approx_equal(grad_x, y));
    EXPECT_TRUE(tensor_approx_equal(grad_y, x));
}

TEST(AutogradTest, DivBackward) {
    // z = x / y, dz/dx = 1/y, dz/dy = -x/y^2
    Tensor x = tensor({1.0f, 2.0f, 3.0f});
    Tensor y = tensor({2.0f, 4.0f, 5.0f});
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    Tensor z = div_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor grad_x = get_grad(x);
    Tensor grad_y = get_grad(y);

    // grad_x = 1/y = [0.5, 0.25, 0.2]
    Tensor expected_grad_x = tensor({0.5f, 0.25f, 0.2f});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected_grad_x));

    // grad_y = -x/y^2 = [-0.25, -0.125, -0.12]
    Tensor expected_grad_y = tensor({-0.25f, -0.125f, -0.12f});
    EXPECT_TRUE(tensor_approx_equal(grad_y, expected_grad_y));
}

TEST(AutogradTest, SqrtBackward) {
    // y = sqrt(x), dy/dx = 1/(2*sqrt(x))
    Tensor x = tensor({1.0f, 4.0f, 9.0f});
    x.set_requires_grad(true);

    Tensor y = sqrt_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = 1/(2*sqrt(x)) = [0.5, 0.25, 0.167]
    Tensor expected = tensor({0.5f, 0.25f, 1.0f/6.0f});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, ExpBackward) {
    // y = exp(x), dy/dx = exp(x)
    Tensor x = tensor({0.0f, 1.0f, 2.0f});
    x.set_requires_grad(true);

    Tensor y = exp_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = exp(x)
    EXPECT_TRUE(tensor_approx_equal(grad_x, y));
}

TEST(AutogradTest, LogBackward) {
    // y = log(x), dy/dx = 1/x
    Tensor x = tensor({1.0f, 2.0f, std::exp(1.0f)});
    x.set_requires_grad(true);

    Tensor y = log_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = 1/x
    Tensor expected = tensor({1.0f, 0.5f, 1.0f/std::exp(1.0f)});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, SinCosBackward) {
    // y = sin(x), dy/dx = cos(x)
    Tensor x = tensor({0.0f, static_cast<float>(M_PI/2), static_cast<float>(M_PI)});
    x.set_requires_grad(true);

    Tensor y = sin_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = cos(x) = [1, 0, -1]
    Tensor expected = tensor({1.0f, 0.0f, -1.0f});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected, 1e-3f, 1e-5f));
}

TEST(AutogradTest, TanhBackward) {
    // y = tanh(x), dy/dx = 1 - tanh^2(x)
    Tensor x = tensor({0.0f, 1.0f, -1.0f});
    x.set_requires_grad(true);

    Tensor y = tanh_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = 1 - tanh^2(x)
    Tensor expected = ones({3}).sub(y.square());
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, SigmoidBackward) {
    // y = sigmoid(x), dy/dx = sigmoid(x) * (1 - sigmoid(x))
    Tensor x = tensor({0.0f, 1.0f, -1.0f});
    x.set_requires_grad(true);

    Tensor y = sigmoid_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = sigmoid(x) * (1 - sigmoid(x))
    Tensor expected = y.mul(ones({3}).sub(y));
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, ReluBackward) {
    // y = relu(x), dy/dx = 1 if x > 0 else 0
    Tensor x = tensor({-1.0f, 0.0f, 1.0f, 2.0f});
    x.set_requires_grad(true);

    Tensor y = relu_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // grad = [0, 0, 1, 1]
    Tensor expected = tensor({0.0f, 0.0f, 1.0f, 1.0f});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, SumBackward) {
    // y = sum(x), dy/dx = 1
    Tensor x = randn({3, 4});
    x.set_requires_grad(true);

    Tensor y = sum_autograd(x);
    tensor_backward(y);

    Tensor grad_x = get_grad(x);

    // grad should be ones
    Tensor expected = ones({3, 4});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, MeanBackward) {
    // y = mean(x), dy/dx = 1/numel
    Tensor x = randn({3, 4});
    x.set_requires_grad(true);

    Tensor y = mean_autograd(x);
    tensor_backward(y);

    Tensor grad_x = get_grad(x);

    // grad should be 1/12 everywhere
    Tensor expected = full({3, 4}, Scalar(1.0f/12.0f));
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

// ============================================================================
// Linear Algebra Backward Tests
// ============================================================================

TEST(AutogradTest, MmBackward) {
    // C = A @ B
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC
    Tensor A = randn({2, 3});
    Tensor B = randn({3, 4});
    A.set_requires_grad(true);
    B.set_requires_grad(true);

    Tensor C = mm_autograd(A, B);
    tensor_backward(sum_autograd(C));

    Tensor grad_A = get_grad(A);
    Tensor grad_B = get_grad(B);

    EXPECT_TRUE(grad_A.defined());
    EXPECT_TRUE(grad_B.defined());

    // Check shapes
    EXPECT_EQ(grad_A.size(0), 2);
    EXPECT_EQ(grad_A.size(1), 3);
    EXPECT_EQ(grad_B.size(0), 3);
    EXPECT_EQ(grad_B.size(1), 4);

    // Manual check: grad_A = ones(2,4) @ B^T
    Tensor expected_grad_A = ones({2, 4}).mm(B.t());
    EXPECT_TRUE(tensor_approx_equal(grad_A, expected_grad_A));

    // grad_B = A^T @ ones(2,4)
    Tensor expected_grad_B = A.t().mm(ones({2, 4}));
    EXPECT_TRUE(tensor_approx_equal(grad_B, expected_grad_B));
}

TEST(AutogradTest, MvBackward) {
    // y = A @ x
    Tensor A = randn({3, 4});
    Tensor x = randn({4});
    A.set_requires_grad(true);
    x.set_requires_grad(true);

    Tensor y = mv_autograd(A, x);
    tensor_backward(sum_autograd(y));

    Tensor grad_A = get_grad(A);
    Tensor grad_x = get_grad(x);

    EXPECT_TRUE(grad_A.defined());
    EXPECT_TRUE(grad_x.defined());

    // grad_A = outer(ones(3), x)
    Tensor expected_grad_A = native::outer(ones({3}), x);
    EXPECT_TRUE(tensor_approx_equal(grad_A, expected_grad_A));

    // grad_x = A^T @ ones(3)
    Tensor expected_grad_x = A.t().mv(ones({3}));
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected_grad_x));
}

TEST(AutogradTest, DotBackward) {
    // y = a · b
    Tensor a = tensor({1.0f, 2.0f, 3.0f});
    Tensor b = tensor({4.0f, 5.0f, 6.0f});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor y = dot_autograd(a, b);
    tensor_backward(y);

    Tensor grad_a = get_grad(a);
    Tensor grad_b = get_grad(b);

    // grad_a = b, grad_b = a
    EXPECT_TRUE(tensor_approx_equal(grad_a, b));
    EXPECT_TRUE(tensor_approx_equal(grad_b, a));
}

// ============================================================================
// Shape Operations Backward Tests
// ============================================================================

TEST(AutogradTest, ViewBackward) {
    Tensor x = randn({2, 3, 4});
    x.set_requires_grad(true);

    Tensor y = view_autograd(x, {6, 4});
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    EXPECT_TRUE(grad_x.defined());
    EXPECT_EQ(grad_x.size(0), 2);
    EXPECT_EQ(grad_x.size(1), 3);
    EXPECT_EQ(grad_x.size(2), 4);

    // Gradient should be ones reshaped back
    Tensor expected = ones({2, 3, 4});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, TransposeBackward) {
    Tensor x = randn({2, 3});
    x.set_requires_grad(true);

    Tensor y = transpose_autograd(x, 0, 1);  // [3, 2]
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    EXPECT_TRUE(grad_x.defined());
    EXPECT_EQ(grad_x.size(0), 2);
    EXPECT_EQ(grad_x.size(1), 3);

    // Gradient flows back through transpose
    Tensor expected = ones({2, 3});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, SqueezeUnsqueezeBackward) {
    Tensor x = randn({1, 3, 1, 4});
    x.set_requires_grad(true);

    Tensor y = squeeze_autograd(x);  // [3, 4]
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    EXPECT_TRUE(grad_x.defined());
    EXPECT_EQ(grad_x.dim(), 4);
    EXPECT_EQ(grad_x.size(0), 1);
    EXPECT_EQ(grad_x.size(1), 3);
    EXPECT_EQ(grad_x.size(2), 1);
    EXPECT_EQ(grad_x.size(3), 4);
}

// ============================================================================
// Chain Rule Tests
// ============================================================================

TEST(AutogradTest, ChainRule) {
    // y = (x + 1)^2
    // dy/dx = 2(x + 1)
    Tensor x = tensor({1.0f, 2.0f, 3.0f});
    x.set_requires_grad(true);

    Tensor x_plus_1 = add_autograd(x, ones({3}));
    Tensor y = mul_autograd(x_plus_1, x_plus_1);  // square

    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // Expected: 2*(x+1) = [4, 6, 8]
    Tensor expected = tensor({4.0f, 6.0f, 8.0f});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, MultiplePathsChainRule) {
    // y = x * x = x^2
    // dy/dx = 2x
    Tensor x = tensor({1.0f, 2.0f, 3.0f});
    x.set_requires_grad(true);

    Tensor y = mul_autograd(x, x);
    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // Expected: 2x = [2, 4, 6]
    Tensor expected = tensor({2.0f, 4.0f, 6.0f});
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected));
}

TEST(AutogradTest, ComplexExpression) {
    // y = exp(x * 2) + sin(x)
    // dy/dx = 2*exp(x*2) + cos(x)
    Tensor x = tensor({0.0f, 0.5f, 1.0f});
    x.set_requires_grad(true);

    Tensor two = full({3}, Scalar(2.0f));
    Tensor x_times_2 = mul_autograd(x, two);
    Tensor exp_part = exp_autograd(x_times_2);
    Tensor sin_part = sin_autograd(x);
    Tensor y = add_autograd(exp_part, sin_part);

    tensor_backward(sum_autograd(y));

    Tensor grad_x = get_grad(x);

    // Expected: 2*exp(2x) + cos(x)
    float* x_data = x.data_ptr<float>();
    Tensor expected = tensor({
        2.0f * std::exp(2.0f * x_data[0]) + std::cos(x_data[0]),
        2.0f * std::exp(2.0f * x_data[1]) + std::cos(x_data[1]),
        2.0f * std::exp(2.0f * x_data[2]) + std::cos(x_data[2])
    });

    EXPECT_TRUE(tensor_approx_equal(grad_x, expected, 1e-3f, 1e-5f));
}

// ============================================================================
// Broadcasting Backward Tests
// ============================================================================

TEST(AutogradTest, BroadcastingAdd) {
    // z = x + y where x is [3, 4] and y is [4]
    Tensor x = randn({3, 4});
    Tensor y = randn({4});
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    Tensor z = add_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor grad_x = get_grad(x);
    Tensor grad_y = get_grad(y);

    EXPECT_TRUE(grad_x.defined());
    EXPECT_TRUE(grad_y.defined());

    // grad_x should be ones [3, 4]
    EXPECT_TRUE(tensor_approx_equal(grad_x, ones({3, 4})));

    // grad_y should be summed over broadcast dim: ones [4] * 3 = [3, 3, 3, 3]
    Tensor expected_grad_y = full({4}, Scalar(3.0f));
    EXPECT_TRUE(tensor_approx_equal(grad_y, expected_grad_y));
}

TEST(AutogradTest, BroadcastingMul) {
    // z = x * y where x is [2, 3] and y is scalar-like [1]
    Tensor x = tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    Tensor y = tensor({2.0f});
    x.set_requires_grad(true);
    y.set_requires_grad(true);

    Tensor z = mul_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor grad_x = get_grad(x);
    Tensor grad_y = get_grad(y);

    // grad_x should be y expanded = [2, 2, 2; 2, 2, 2]
    Tensor expected_grad_x = full({2, 3}, Scalar(2.0f));
    EXPECT_TRUE(tensor_approx_equal(grad_x, expected_grad_x));

    // grad_y should be sum of x = 21
    Tensor expected_grad_y = tensor({21.0f});
    EXPECT_TRUE(tensor_approx_equal(grad_y, expected_grad_y));
}

// ============================================================================
// Gradient Flow Control Tests
// ============================================================================

TEST(AutogradTest, NoGradInput) {
    // If input doesn't require grad, output shouldn't either
    Tensor x = randn({3, 4});
    // x.requires_grad is false by default

    Tensor y = neg_autograd(x);
    EXPECT_FALSE(y.requires_grad());
}

TEST(AutogradTest, MixedGradInputs) {
    // If only one input requires grad, output should require grad
    // and only that input should receive gradient
    Tensor x = randn({3});
    Tensor y = randn({3});
    x.set_requires_grad(true);
    // y doesn't require grad

    Tensor z = add_autograd(x, y);
    EXPECT_TRUE(z.requires_grad());

    tensor_backward(sum_autograd(z));

    Tensor grad_x = get_grad(x);
    EXPECT_TRUE(grad_x.defined());

    // y has no gradient (wasn't tracking)
    Tensor grad_y = get_grad(y);
    EXPECT_FALSE(grad_y.defined());
}

// ============================================================================
// Neural Network Building Block Tests
// ============================================================================

TEST(AutogradTest, LinearLayerForward) {
    // Linear layer: y = x @ W^T + b
    // Simplified: y = W @ x + b
    Tensor W = randn({10, 5});  // [out_features, in_features]
    Tensor b = randn({10});     // [out_features]
    Tensor x = randn({5});      // [in_features]

    W.set_requires_grad(true);
    b.set_requires_grad(true);

    // y = W @ x + b
    Tensor Wx = mv_autograd(W, x);
    Tensor y = add_autograd(Wx, b);

    EXPECT_EQ(y.size(0), 10);
    EXPECT_TRUE(y.requires_grad());

    tensor_backward(sum_autograd(y));

    Tensor grad_W = get_grad(W);
    Tensor grad_b = get_grad(b);

    EXPECT_TRUE(grad_W.defined());
    EXPECT_TRUE(grad_b.defined());

    // grad_W = outer(ones(10), x)
    Tensor expected_grad_W = native::outer(ones({10}), x);
    EXPECT_TRUE(tensor_approx_equal(grad_W, expected_grad_W));

    // grad_b = ones(10)
    EXPECT_TRUE(tensor_approx_equal(grad_b, ones({10})));
}

TEST(AutogradTest, SoftmaxCrossEntropyLike) {
    // Simplified softmax-like computation
    // logits -> exp -> sum -> log
    Tensor logits = tensor({1.0f, 2.0f, 3.0f});
    logits.set_requires_grad(true);

    Tensor exp_logits = exp_autograd(logits);
    Tensor sum_exp = sum_autograd(exp_logits);
    Tensor log_sum = log_autograd(sum_exp);

    tensor_backward(log_sum);

    Tensor grad_logits = get_grad(logits);
    EXPECT_TRUE(grad_logits.defined());

    // Analytical gradient: d/dx[log(sum(exp(x)))] = exp(x) / sum(exp(x)) = softmax(x)
    float sum_val = sum_exp.item().toFloat();
    Tensor expected = exp_logits.div(Scalar(sum_val));
    EXPECT_TRUE(tensor_approx_equal(grad_logits, expected));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
