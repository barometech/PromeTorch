// ============================================================================
// Comprehensive Gradient Check Tests for PromeTorch Autograd
// ============================================================================
// Tests gradient correctness for ALL differentiable operations using both:
// 1. Analytical gradient verification (known derivatives)
// 2. Numerical gradient verification (finite differences)
//
// Coverage:
//   - Unary math ops (15 tests)
//   - Binary ops (10 tests)
//   - Reduction ops (8 tests)
//   - Linear algebra ops (8 tests)
//   - Shape ops (5 tests)
//   - Composed / chain rule ops (5 tests)
// ============================================================================

#include <gtest/gtest.h>
#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include <cmath>
#include <functional>
#include <numeric>

using namespace at;
using namespace torch;
using namespace torch::autograd;

// ============================================================================
// Helper: Numerical gradient via central finite differences
// ============================================================================
// For f: R^n -> R (after .sum()), compute df/dx_i numerically.
// We clone x, perturb element i by +/- eps, recompute f, and use
//   (f(x+eps) - f(x-eps)) / (2*eps)
// The function `func` must NOT require grad on its input (we call it
// on plain tensors without autograd wrappers).

static Tensor numerical_gradient(
    std::function<Tensor(const Tensor&)> func,
    const Tensor& input,
    double eps = 1e-3
) {
    Tensor result = zeros(input.sizes());
    float* res_data = result.mutable_data_ptr<float>();
    int64_t n = input.numel();

    for (int64_t i = 0; i < n; ++i) {
        // +eps
        Tensor x_plus = input.clone();
        x_plus.mutable_data_ptr<float>()[i] += static_cast<float>(eps);
        float f_plus = func(x_plus).sum().item<float>();

        // -eps
        Tensor x_minus = input.clone();
        x_minus.mutable_data_ptr<float>()[i] -= static_cast<float>(eps);
        float f_minus = func(x_minus).sum().item<float>();

        res_data[i] = static_cast<float>((f_plus - f_minus) / (2.0 * eps));
    }
    return result;
}

// Two-input numerical gradient (w.r.t. first input)
static Tensor numerical_gradient_2(
    std::function<Tensor(const Tensor&, const Tensor&)> func,
    const Tensor& input1,
    const Tensor& input2,
    double eps = 1e-3
) {
    Tensor result = zeros(input1.sizes());
    float* res_data = result.mutable_data_ptr<float>();
    int64_t n = input1.numel();

    for (int64_t i = 0; i < n; ++i) {
        Tensor x_plus = input1.clone();
        x_plus.mutable_data_ptr<float>()[i] += static_cast<float>(eps);
        float f_plus = func(x_plus, input2).sum().item<float>();

        Tensor x_minus = input1.clone();
        x_minus.mutable_data_ptr<float>()[i] -= static_cast<float>(eps);
        float f_minus = func(x_minus, input2).sum().item<float>();

        res_data[i] = static_cast<float>((f_plus - f_minus) / (2.0 * eps));
    }
    return result;
}

// ============================================================================
// Helper: Compare analytical vs numerical gradients
// ============================================================================
static void check_gradients_close(
    const Tensor& analytical,
    const Tensor& numerical,
    float atol = 1e-3f,
    float rtol = 1e-2f,
    const std::string& op_name = ""
) {
    ASSERT_EQ(analytical.numel(), numerical.numel())
        << "Gradient size mismatch for " << op_name;

    const float* a = analytical.data_ptr<float>();
    const float* n_data = numerical.data_ptr<float>();

    for (int64_t i = 0; i < analytical.numel(); ++i) {
        float diff = std::abs(a[i] - n_data[i]);
        float tol = atol + rtol * std::abs(n_data[i]);
        EXPECT_LE(diff, tol)
            << op_name << " gradient mismatch at element " << i
            << ": analytical=" << a[i] << " numerical=" << n_data[i]
            << " diff=" << diff << " tol=" << tol;
    }
}

// ============================================================================
// Helper: Get gradient from tensor (handles base AutogradMeta)
// ============================================================================
static Tensor get_grad(const Tensor& tensor) {
    auto* raw_meta = tensor.autograd_meta();
    if (raw_meta && raw_meta->grad_) {
        return Tensor(raw_meta->grad_);
    }
    return Tensor();
}

// ============================================================================
//  1. UNARY MATH OPERATIONS (15 tests)
// ============================================================================

TEST(AutogradFullTest, NegGrad) {
    // d/dx[-x] = -1
    Tensor x = at::tensor({1.0f, -2.0f, 3.0f, -0.5f}).set_requires_grad(true);
    Tensor y = neg_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(g[i], -1.0f, 1e-5f) << "NegGrad element " << i;
    }
}

TEST(AutogradFullTest, ExpGrad) {
    // d/dx[exp(x)] = exp(x)
    Tensor x = at::tensor({0.0f, 1.0f, -1.0f, 0.5f}).set_requires_grad(true);
    Tensor y = exp_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(g[i], std::exp(xd[i]), 1e-5f) << "ExpGrad element " << i;
    }
}

TEST(AutogradFullTest, LogGrad) {
    // d/dx[log(x)] = 1/x  (x > 0)
    Tensor x = at::tensor({1.0f, 2.0f, 0.5f, 3.0f}).set_requires_grad(true);
    Tensor y = log_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(g[i], 1.0f / xd[i], 1e-5f) << "LogGrad element " << i;
    }
}

TEST(AutogradFullTest, SqrtGrad) {
    // d/dx[sqrt(x)] = 1 / (2*sqrt(x))  (x > 0)
    Tensor x = at::tensor({1.0f, 4.0f, 0.25f, 9.0f}).set_requires_grad(true);
    Tensor y = sqrt_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float expected = 1.0f / (2.0f * std::sqrt(xd[i]));
        EXPECT_NEAR(g[i], expected, 1e-5f) << "SqrtGrad element " << i;
    }
}

TEST(AutogradFullTest, SinGrad) {
    // d/dx[sin(x)] = cos(x)
    Tensor x = at::tensor({0.0f, 1.0f, -0.5f, 2.0f}).set_requires_grad(true);
    Tensor y = sin_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(g[i], std::cos(xd[i]), 1e-5f) << "SinGrad element " << i;
    }
}

TEST(AutogradFullTest, CosGrad) {
    // d/dx[cos(x)] = -sin(x)
    Tensor x = at::tensor({0.0f, 1.0f, -0.5f, 2.0f}).set_requires_grad(true);
    Tensor y = cos_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(g[i], -std::sin(xd[i]), 1e-5f) << "CosGrad element " << i;
    }
}

TEST(AutogradFullTest, TanhGrad) {
    // d/dx[tanh(x)] = 1 - tanh(x)^2
    Tensor x = at::tensor({0.0f, 1.0f, -1.0f, 0.5f}).set_requires_grad(true);
    Tensor y = tanh_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float t = std::tanh(xd[i]);
        float expected = 1.0f - t * t;
        EXPECT_NEAR(g[i], expected, 1e-5f) << "TanhGrad element " << i;
    }
}

TEST(AutogradFullTest, SigmoidGrad) {
    // d/dx[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))
    Tensor x = at::tensor({0.0f, 1.0f, -1.0f, 2.0f}).set_requires_grad(true);
    Tensor y = sigmoid_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-xd[i]));
        float expected = s * (1.0f - s);
        EXPECT_NEAR(g[i], expected, 1e-5f) << "SigmoidGrad element " << i;
    }
}

TEST(AutogradFullTest, ReluGrad) {
    // d/dx[relu(x)] = 1 if x > 0, 0 if x <= 0
    Tensor x = at::tensor({-2.0f, -0.5f, 0.0f, 0.5f, 2.0f}).set_requires_grad(true);
    Tensor y = relu_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float expected = xd[i] > 0.0f ? 1.0f : 0.0f;
        EXPECT_NEAR(g[i], expected, 1e-5f) << "ReluGrad element " << i;
    }
}

TEST(AutogradFullTest, AbsGradNumerical) {
    // d/dx[|x|] = sign(x) -- avoid x=0
    Tensor x_val = at::tensor({1.5f, -2.0f, 0.3f, -0.7f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = abs_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor analytical = get_grad(x);
    ASSERT_TRUE(analytical.defined());

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.abs(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "abs");
}

TEST(AutogradFullTest, PowScalarGrad) {
    // d/dx[x^3] = 3*x^2
    Tensor x = at::tensor({1.0f, 2.0f, -1.0f, 0.5f}).set_requires_grad(true);
    Tensor y = pow_autograd(x, Scalar(3.0));
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float expected = 3.0f * xd[i] * xd[i];
        EXPECT_NEAR(g[i], expected, 1e-4f) << "PowScalarGrad element " << i;
    }
}

TEST(AutogradFullTest, ClampGrad) {
    // d/dx[clamp(x, lo, hi)] = 1 if lo <= x <= hi, 0 otherwise
    Tensor x = at::tensor({-2.0f, -0.5f, 0.0f, 0.5f, 2.0f}).set_requires_grad(true);
    Tensor y = clamp_autograd(x, Scalar(-1.0), Scalar(1.0));
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* g = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float expected = (xd[i] >= -1.0f && xd[i] <= 1.0f) ? 1.0f : 0.0f;
        EXPECT_NEAR(g[i], expected, 1e-5f) << "ClampGrad element " << i;
    }
}

TEST(AutogradFullTest, ExpGradNumerical) {
    // Numerical check for exp
    Tensor x_val = at::tensor({0.5f, -0.3f, 1.2f, -1.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = exp_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor analytical = get_grad(x);
    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.exp(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "exp_numerical");
}

TEST(AutogradFullTest, LogGradNumerical) {
    // Numerical check for log
    Tensor x_val = at::tensor({0.5f, 1.0f, 2.0f, 3.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = log_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor analytical = get_grad(x);
    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.log(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "log_numerical");
}

TEST(AutogradFullTest, SqrtGradNumerical) {
    // Numerical check for sqrt
    Tensor x_val = at::tensor({0.5f, 1.0f, 4.0f, 9.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = sqrt_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor analytical = get_grad(x);
    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.sqrt(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "sqrt_numerical");
}

TEST(AutogradFullTest, TanhGradNumerical) {
    // Numerical check for tanh
    Tensor x_val = at::tensor({-1.0f, 0.0f, 0.5f, 1.5f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = tanh_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor analytical = get_grad(x);
    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.tanh(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "tanh_numerical");
}

// ============================================================================
//  2. BINARY OPERATIONS (10 tests)
// ============================================================================

TEST(AutogradFullTest, AddTensorGrad) {
    // d/dx[x + y] = 1, d/dy[x + y] = 1
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor y = at::tensor({4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor z = add_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f, 1e-5f);
        EXPECT_NEAR(gy.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, SubTensorGrad) {
    // d/dx[x - y] = 1, d/dy[x - y] = -1
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor y = at::tensor({4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor z = sub_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f, 1e-5f);
        EXPECT_NEAR(gy.data_ptr<float>()[i], -1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, MulTensorGrad) {
    // d/dx[x * y] = y, d/dy[x * y] = x
    Tensor x = at::tensor({2.0f, 3.0f, 4.0f}).set_requires_grad(true);
    Tensor y = at::tensor({5.0f, 6.0f, 7.0f}).set_requires_grad(true);
    Tensor z = mul_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    const float* xd = x.data_ptr<float>();
    const float* yd = y.data_ptr<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gx.data_ptr<float>()[i], yd[i], 1e-5f);
        EXPECT_NEAR(gy.data_ptr<float>()[i], xd[i], 1e-5f);
    }
}

TEST(AutogradFullTest, DivTensorGrad) {
    // d/dx[x / y] = 1/y, d/dy[x / y] = -x/y^2
    Tensor x = at::tensor({2.0f, 6.0f, 12.0f}).set_requires_grad(true);
    Tensor y = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor z = div_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    const float* xd = x.data_ptr<float>();
    const float* yd = y.data_ptr<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f / yd[i], 1e-4f)
            << "DivGrad dx element " << i;
        EXPECT_NEAR(gy.data_ptr<float>()[i], -xd[i] / (yd[i] * yd[i]), 1e-4f)
            << "DivGrad dy element " << i;
    }
}

TEST(AutogradFullTest, MulTensorGradNumerical) {
    // Numerical gradient check for mul
    Tensor x_val = at::tensor({2.0f, 3.0f, -1.0f});
    Tensor y_val = at::tensor({5.0f, -2.0f, 4.0f});

    // Check grad w.r.t. x
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = y_val.clone().set_requires_grad(true);
    Tensor z = mul_autograd(x, y);
    tensor_backward(sum_autograd(z));
    Tensor analytical_x = get_grad(x);

    Tensor numerical_x = numerical_gradient_2(
        [](const Tensor& a, const Tensor& b) { return a.mul(b); },
        x_val, y_val);

    check_gradients_close(analytical_x, numerical_x, 1e-3f, 1e-2f, "mul_x");
}

TEST(AutogradFullTest, DivTensorGradNumerical) {
    // Numerical gradient check for div
    Tensor x_val = at::tensor({2.0f, 6.0f, 12.0f});
    Tensor y_val = at::tensor({1.0f, 2.0f, 3.0f});

    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = y_val.clone().set_requires_grad(true);
    Tensor z = div_autograd(x, y);
    tensor_backward(sum_autograd(z));
    Tensor analytical_x = get_grad(x);
    Tensor analytical_y = get_grad(y);

    Tensor numerical_x = numerical_gradient_2(
        [](const Tensor& a, const Tensor& b) { return a.div(b); },
        x_val, y_val);

    Tensor numerical_y = numerical_gradient_2(
        [](const Tensor& a, const Tensor& b) { return a.div(b); },
        y_val, x_val);
    // For dy: swap order so we perturb y
    numerical_y = numerical_gradient(
        [&x_val](const Tensor& b) { return x_val.div(b); }, y_val);

    check_gradients_close(analytical_x, numerical_x, 1e-3f, 1e-2f, "div_x");
    check_gradients_close(analytical_y, numerical_y, 1e-3f, 1e-2f, "div_y");
}

TEST(AutogradFullTest, PowTensorGrad) {
    // z = x^y, dz/dx = y * x^(y-1), dz/dy = x^y * log(x)
    Tensor x = at::tensor({2.0f, 3.0f, 4.0f}).set_requires_grad(true);
    Tensor y = at::tensor({3.0f, 2.0f, 0.5f}).set_requires_grad(true);
    Tensor z = pow_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    const float* xd = x.data_ptr<float>();
    const float* yd = y.data_ptr<float>();
    for (int64_t i = 0; i < 3; ++i) {
        float expected_gx = yd[i] * std::pow(xd[i], yd[i] - 1.0f);
        float expected_gy = std::pow(xd[i], yd[i]) * std::log(xd[i]);
        EXPECT_NEAR(gx.data_ptr<float>()[i], expected_gx, 1e-3f)
            << "PowTensor dx element " << i;
        EXPECT_NEAR(gy.data_ptr<float>()[i], expected_gy, 1e-3f)
            << "PowTensor dy element " << i;
    }
}

TEST(AutogradFullTest, AddWithAlphaGrad) {
    // z = x + alpha * y, dz/dx = 1, dz/dy = alpha
    float alpha = 2.5f;
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor y = at::tensor({4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor z = add_autograd(x, y, Scalar(alpha));
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f, 1e-5f);
        EXPECT_NEAR(gy.data_ptr<float>()[i], alpha, 1e-5f);
    }
}

TEST(AutogradFullTest, SubWithAlphaGrad) {
    // z = x - alpha * y, dz/dx = 1, dz/dy = -alpha
    float alpha = 3.0f;
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor y = at::tensor({4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor z = sub_autograd(x, y, Scalar(alpha));
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gx.data_ptr<float>()[i], 1.0f, 1e-5f);
        EXPECT_NEAR(gy.data_ptr<float>()[i], -alpha, 1e-5f);
    }
}

TEST(AutogradFullTest, MulGradMultidim) {
    // Test mul grad with 2D tensors
    Tensor x_val = at::randn({3, 4});
    Tensor y_val = at::randn({3, 4});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = y_val.clone().set_requires_grad(true);
    Tensor z = mul_autograd(x, y);
    tensor_backward(sum_autograd(z));

    Tensor gx = get_grad(x);
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gx.defined() && gy.defined());

    // gx should equal y_val, gy should equal x_val
    const float* gx_d = gx.data_ptr<float>();
    const float* gy_d = gy.data_ptr<float>();
    const float* xd = x_val.data_ptr<float>();
    const float* yd = y_val.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        EXPECT_NEAR(gx_d[i], yd[i], 1e-5f);
        EXPECT_NEAR(gy_d[i], xd[i], 1e-5f);
    }
}

// ============================================================================
//  3. REDUCTION OPERATIONS (8 tests)
// ============================================================================

TEST(AutogradFullTest, SumGrad) {
    // d/dx[sum(x)] = 1 for all elements
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f, 4.0f}).set_requires_grad(true);
    Tensor y = sum_autograd(x);
    tensor_backward(y);

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.numel(), 4);

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, SumDimGrad) {
    // sum along dim=1 of [2, 3] tensor: gradient is ones broadcast back
    Tensor x = at::randn({2, 3}).set_requires_grad(true);
    Tensor y = sum_autograd(x, /*dim=*/1);
    // y has shape [2], need to sum it to get scalar
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.numel(), 6);

    // All gradients should be 1 (sum of outer sum grad = 1 expanded)
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, MeanGrad) {
    // d/dx[mean(x)] = 1/n for all elements
    int64_t n = 6;
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor y = mean_autograd(x);
    tensor_backward(y);

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    float expected = 1.0f / static_cast<float>(n);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], expected, 1e-5f);
    }
}

TEST(AutogradFullTest, MeanDimGrad) {
    // mean along dim=0 of [3, 2] tensor
    Tensor x_val = at::randn({3, 2});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = mean_autograd(x, /*dim=*/0);
    // y has shape [2], sum to scalar
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    // Each element gets gradient = 1/3 (3 elements along dim 0)
    float expected = 1.0f / 3.0f;
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], expected, 1e-5f)
            << "MeanDimGrad element " << i;
    }
}

TEST(AutogradFullTest, SumGrad2D) {
    // Sum of 2D tensor: all grads should be 1
    Tensor x = at::randn({4, 5}).set_requires_grad(true);
    Tensor y = sum_autograd(x);
    tensor_backward(y);

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, MeanGrad2D) {
    // Mean of 3x4 tensor: all grads should be 1/12
    Tensor x = at::randn({3, 4}).set_requires_grad(true);
    Tensor y = mean_autograd(x);
    tensor_backward(y);

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    float expected = 1.0f / 12.0f;
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], expected, 1e-5f);
    }
}

TEST(AutogradFullTest, SumDimKeepdimGrad) {
    // sum(x, dim=1, keepdim=true) on [2, 3]: grad shape should be [2, 3]
    Tensor x = at::randn({2, 3}).set_requires_grad(true);
    Tensor y = sum_autograd(x, /*dim=*/1, /*keepdim=*/true);
    // y shape is [2, 1]
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.sizes()[0], 2);
    ASSERT_EQ(grad.sizes()[1], 3);

    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, MeanDimKeepdimGrad) {
    // mean(x, dim=0, keepdim=true) on [4, 3]
    Tensor x = at::randn({4, 3}).set_requires_grad(true);
    Tensor y = mean_autograd(x, /*dim=*/0, /*keepdim=*/true);
    // y shape is [1, 3]
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    float expected = 1.0f / 4.0f;
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], expected, 1e-5f);
    }
}

// ============================================================================
//  4. LINEAR ALGEBRA OPERATIONS (8 tests)
// ============================================================================

TEST(AutogradFullTest, MmGrad) {
    // C = A @ B, dC/dA = grad @ B^T, dC/dB = A^T @ grad
    Tensor A = at::randn({3, 4}).set_requires_grad(true);
    Tensor B = at::randn({4, 5}).set_requires_grad(true);
    Tensor C = mm_autograd(A, B);
    tensor_backward(sum_autograd(C));

    Tensor gA = get_grad(A);
    Tensor gB = get_grad(B);
    ASSERT_TRUE(gA.defined() && gB.defined());
    ASSERT_EQ(gA.size(0), 3);
    ASSERT_EQ(gA.size(1), 4);
    ASSERT_EQ(gB.size(0), 4);
    ASSERT_EQ(gB.size(1), 5);

    // Numerical check: grad_output is all-ones [3, 5]
    // dC/dA = ones(3,5) @ B^T = sum_cols(B^T) for each row
    // Verify numerically
    Tensor A_val = A.clone().detach();
    Tensor B_val = B.clone().detach();
    Tensor numerical_gA = numerical_gradient_2(
        [](const Tensor& a, const Tensor& b) {
            return at::native::mm(a, b);
        }, A_val, B_val, 1e-3);

    check_gradients_close(gA.contiguous(), numerical_gA, 2e-3f, 2e-2f, "mm_A");
}

TEST(AutogradFullTest, MvGrad) {
    // y = A @ x, dy/dA = outer(grad, x), dy/dx = A^T @ grad
    Tensor A = at::randn({3, 4}).set_requires_grad(true);
    Tensor x = at::randn({4}).set_requires_grad(true);
    Tensor y = mv_autograd(A, x);
    tensor_backward(sum_autograd(y));

    Tensor gA = get_grad(A);
    Tensor gx = get_grad(x);
    ASSERT_TRUE(gA.defined() && gx.defined());
    ASSERT_EQ(gA.size(0), 3);
    ASSERT_EQ(gA.size(1), 4);
    ASSERT_EQ(gx.size(0), 4);

    // Numerical check for dx
    Tensor A_val = A.clone().detach();
    Tensor x_val = x.clone().detach();
    Tensor numerical_gx = numerical_gradient(
        [&A_val](const Tensor& v) {
            return at::native::mv(A_val, v);
        }, x_val, 1e-3);

    check_gradients_close(gx, numerical_gx, 2e-3f, 2e-2f, "mv_x");
}

TEST(AutogradFullTest, DotGrad) {
    // c = a . b, dc/da = b, dc/db = a
    Tensor a = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor b = at::tensor({4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor c = dot_autograd(a, b);
    tensor_backward(c);

    Tensor ga = get_grad(a);
    Tensor gb = get_grad(b);
    ASSERT_TRUE(ga.defined() && gb.defined());

    const float* ad = a.data_ptr<float>();
    const float* bd = b.data_ptr<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(ga.data_ptr<float>()[i], bd[i], 1e-5f);
        EXPECT_NEAR(gb.data_ptr<float>()[i], ad[i], 1e-5f);
    }
}

TEST(AutogradFullTest, MatmulGrad2D) {
    // matmul for 2D is same as mm
    Tensor A = at::randn({3, 4}).set_requires_grad(true);
    Tensor B = at::randn({4, 2}).set_requires_grad(true);
    Tensor C = matmul_autograd(A, B);
    tensor_backward(sum_autograd(C));

    Tensor gA = get_grad(A);
    Tensor gB = get_grad(B);
    ASSERT_TRUE(gA.defined() && gB.defined());

    // Verify shapes
    ASSERT_EQ(gA.size(0), 3);
    ASSERT_EQ(gA.size(1), 4);
    ASSERT_EQ(gB.size(0), 4);
    ASSERT_EQ(gB.size(1), 2);
}

TEST(AutogradFullTest, BmmGrad) {
    // Batched mm: C[b] = A[b] @ B[b]
    Tensor A = at::randn({2, 3, 4}).set_requires_grad(true);
    Tensor B = at::randn({2, 4, 5}).set_requires_grad(true);
    Tensor C = bmm_autograd(A, B);
    tensor_backward(sum_autograd(C));

    Tensor gA = get_grad(A);
    Tensor gB = get_grad(B);
    ASSERT_TRUE(gA.defined() && gB.defined());
    ASSERT_EQ(gA.size(0), 2);
    ASSERT_EQ(gA.size(1), 3);
    ASSERT_EQ(gA.size(2), 4);
    ASSERT_EQ(gB.size(0), 2);
    ASSERT_EQ(gB.size(1), 4);
    ASSERT_EQ(gB.size(2), 5);
}

TEST(AutogradFullTest, MmGradNumerical) {
    // Full numerical gradient check for mm (both inputs)
    Tensor A_val = at::randn({2, 3});
    Tensor B_val = at::randn({3, 2});

    // Analytical: grad w.r.t. A
    Tensor A = A_val.clone().set_requires_grad(true);
    Tensor B = B_val.clone().set_requires_grad(true);
    Tensor C = mm_autograd(A, B);
    tensor_backward(sum_autograd(C));
    Tensor analytical_gA = get_grad(A).contiguous();
    Tensor analytical_gB = get_grad(B).contiguous();

    // Numerical grad w.r.t. A
    Tensor numerical_gA = numerical_gradient_2(
        [](const Tensor& a, const Tensor& b) {
            return at::native::mm(a, b);
        }, A_val, B_val, 1e-3);

    // Numerical grad w.r.t. B
    Tensor numerical_gB = numerical_gradient(
        [&A_val](const Tensor& b) {
            return at::native::mm(A_val, b);
        }, B_val, 1e-3);

    check_gradients_close(analytical_gA, numerical_gA, 2e-3f, 2e-2f, "mm_A_numerical");
    check_gradients_close(analytical_gB, numerical_gB, 2e-3f, 2e-2f, "mm_B_numerical");
}

TEST(AutogradFullTest, TransposeGrad) {
    // t(x) is just a view; d/dx[sum(t(x))] = ones
    Tensor x = at::randn({3, 4}).set_requires_grad(true);
    Tensor y = t_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.size(0), 3);
    ASSERT_EQ(grad.size(1), 4);

    // sum of transposed = sum of original, so grad = ones
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

TEST(AutogradFullTest, EinsumGrad) {
    // einsum("ij,jk->ik", A, B) = A @ B
    Tensor A = at::randn({2, 3}).set_requires_grad(true);
    Tensor B = at::randn({3, 4}).set_requires_grad(true);
    Tensor C = einsum_autograd("ij,jk->ik", {A, B});
    tensor_backward(sum_autograd(C));

    Tensor gA = get_grad(A);
    Tensor gB = get_grad(B);
    ASSERT_TRUE(gA.defined() && gB.defined());

    // grad_A from einsum should be equivalent to mm_autograd grad
    Tensor A2 = A.clone().detach().set_requires_grad(true);
    Tensor B2 = B.clone().detach().set_requires_grad(true);
    Tensor C2 = mm_autograd(A2, B2);
    tensor_backward(sum_autograd(C2));
    Tensor gA2 = get_grad(A2);
    Tensor gB2 = get_grad(B2);

    // They should be approximately equal
    check_gradients_close(gA.contiguous(), gA2.contiguous(), 1e-4f, 1e-3f, "einsum_A_vs_mm");
    check_gradients_close(gB.contiguous(), gB2.contiguous(), 1e-4f, 1e-3f, "einsum_B_vs_mm");
}

// ============================================================================
//  5. SHAPE OPERATIONS (5 tests)
// ============================================================================

TEST(AutogradFullTest, ViewReshapeGrad) {
    // view/reshape should just reshape the gradient back
    Tensor x = at::randn({2, 3}).set_requires_grad(true);
    Tensor y = view_autograd(x, {6});
    // Apply some operation to y so gradient isn't trivially ones
    Tensor z = mul_autograd(y, y);  // y^2
    tensor_backward(sum_autograd(z));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.size(0), 2);
    ASSERT_EQ(grad.size(1), 3);

    // grad should be 2*x reshaped back to [2, 3]
    const float* gd = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(gd[i], 2.0f * xd[i], 1e-4f)
            << "ViewReshapeGrad element " << i;
    }
}

TEST(AutogradFullTest, TransposeDimGrad) {
    // transpose(x, 0, 1) then sum, grad flows through transpose back
    Tensor x = at::randn({3, 4}).set_requires_grad(true);
    Tensor y = transpose_autograd(x, 0, 1);
    // y is [4, 3]. Multiply element-wise with a constant pattern
    Tensor w = at::ones({4, 3}).mul(Scalar(2.0f));
    Tensor z = mul_autograd(y, w);
    tensor_backward(sum_autograd(z));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.size(0), 3);
    ASSERT_EQ(grad.size(1), 4);

    // Since w = 2*ones, grad(x) should be 2*ones (transposed back)
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 2.0f, 1e-4f);
    }
}

TEST(AutogradFullTest, SelectGrad) {
    // select(x, dim=0, index=1): picks row 1 from [3, 4]
    // Gradient goes only to row 1
    Tensor x = at::randn({3, 4}).set_requires_grad(true);
    Tensor y = select_autograd(x, 0, 1);
    // y shape: [4]
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.size(0), 3);
    ASSERT_EQ(grad.size(1), 4);

    const float* gd = grad.data_ptr<float>();
    for (int64_t r = 0; r < 3; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            float expected = (r == 1) ? 1.0f : 0.0f;
            EXPECT_NEAR(gd[r * 4 + c], expected, 1e-5f)
                << "SelectGrad [" << r << "," << c << "]";
        }
    }
}

TEST(AutogradFullTest, NarrowGrad) {
    // narrow(x, dim=1, start=1, length=2) on [2, 4]
    // Gradient goes only to columns 1 and 2
    Tensor x = at::randn({2, 4}).set_requires_grad(true);
    Tensor y = narrow_autograd(x, 1, 1, 2);
    // y shape: [2, 2]
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.size(0), 2);
    ASSERT_EQ(grad.size(1), 4);

    const float* gd = grad.data_ptr<float>();
    for (int64_t r = 0; r < 2; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            float expected = (c >= 1 && c < 3) ? 1.0f : 0.0f;
            EXPECT_NEAR(gd[r * 4 + c], expected, 1e-5f)
                << "NarrowGrad [" << r << "," << c << "]";
        }
    }
}

TEST(AutogradFullTest, SqueezeUnsqueezeGrad) {
    // unsqueeze(x, 0) on [3] -> [1, 3], then squeeze back -> [3]
    // Gradient should pass through unchanged
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor y = unsqueeze_autograd(x, 0);  // [1, 3]
    ASSERT_EQ(y.size(0), 1);
    ASSERT_EQ(y.size(1), 3);

    Tensor z = squeeze_autograd(y, 0);  // back to [3]
    ASSERT_EQ(z.dim(), 1);
    ASSERT_EQ(z.size(0), 3);

    // Apply a non-trivial operation
    Tensor w = mul_autograd(z, z);  // z^2
    tensor_backward(sum_autograd(w));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.size(0), 3);

    // grad = 2*x
    const float* gd = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gd[i], 2.0f * xd[i], 1e-4f)
            << "SqueezeUnsqueezeGrad element " << i;
    }
}

// ============================================================================
//  6. COMPOSED / CHAIN RULE OPERATIONS (5 tests)
// ============================================================================

TEST(AutogradFullTest, ExpOfSinChainRule) {
    // f(x) = exp(sin(x)), df/dx = exp(sin(x)) * cos(x)
    Tensor x_val = at::tensor({0.5f, 1.0f, -0.3f, 2.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = exp_autograd(sin_autograd(x));
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* gd = grad.data_ptr<float>();
    const float* xd = x_val.data_ptr<float>();
    for (int64_t i = 0; i < x.numel(); ++i) {
        float expected = std::exp(std::sin(xd[i])) * std::cos(xd[i]);
        EXPECT_NEAR(gd[i], expected, 1e-4f) << "ExpOfSin element " << i;
    }
}

TEST(AutogradFullTest, SigmoidOfLinear) {
    // f(x) = sigmoid(W @ x + b) summed
    // This tests chain rule through mm + add + sigmoid + sum
    Tensor W = at::randn({3, 4}).set_requires_grad(true);
    Tensor x = at::randn({4, 1}).set_requires_grad(true);
    Tensor b = at::randn({3, 1}).set_requires_grad(true);

    Tensor Wx = mm_autograd(W, x);          // [3, 1]
    Tensor z = add_autograd(Wx, b);          // [3, 1]
    Tensor a = sigmoid_autograd(z);          // [3, 1]
    Tensor loss = sum_autograd(a);           // scalar
    tensor_backward(loss);

    Tensor gW = get_grad(W);
    Tensor gx = get_grad(x);
    Tensor gb = get_grad(b);
    ASSERT_TRUE(gW.defined());
    ASSERT_TRUE(gx.defined());
    ASSERT_TRUE(gb.defined());
    ASSERT_EQ(gW.size(0), 3);
    ASSERT_EQ(gW.size(1), 4);
    ASSERT_EQ(gx.size(0), 4);
    ASSERT_EQ(gx.size(1), 1);

    // Numerical check for x gradient
    Tensor W_val = W.clone().detach();
    Tensor b_val = b.clone().detach();
    Tensor x_val = x.clone().detach();
    Tensor numerical_gx = numerical_gradient(
        [&W_val, &b_val](const Tensor& inp) {
            Tensor Wx = at::native::mm(W_val, inp);
            Tensor z = Wx.add(b_val);
            return z.sigmoid();
        }, x_val, 1e-3);

    check_gradients_close(gx.contiguous(), numerical_gx, 2e-3f, 5e-2f, "sigmoid_linear_x");
}

TEST(AutogradFullTest, ReluOfMmThenSum) {
    // f(W, x) = sum(relu(W @ x))
    Tensor W = at::randn({3, 4}).set_requires_grad(true);
    Tensor x = at::randn({4, 2}).set_requires_grad(true);

    Tensor Wx = mm_autograd(W, x);
    Tensor r = relu_autograd(Wx);
    Tensor loss = sum_autograd(r);
    tensor_backward(loss);

    Tensor gW = get_grad(W);
    Tensor gx = get_grad(x);
    ASSERT_TRUE(gW.defined() && gx.defined());
    ASSERT_EQ(gW.size(0), 3);
    ASSERT_EQ(gW.size(1), 4);

    // Numerical check for W
    Tensor W_val = W.clone().detach();
    Tensor x_val = x.clone().detach();
    Tensor numerical_gW = numerical_gradient_2(
        [](const Tensor& w, const Tensor& inp) {
            return at::native::mm(w, inp).relu();
        }, W_val, x_val, 1e-3);

    check_gradients_close(gW.contiguous(), numerical_gW, 5e-3f, 5e-2f, "relu_mm_W");
}

TEST(AutogradFullTest, MultiLayerChain) {
    // Two-layer: y = tanh(W2 @ relu(W1 @ x))
    Tensor W1 = at::randn({4, 3}).set_requires_grad(true);
    Tensor W2 = at::randn({2, 4}).set_requires_grad(true);
    Tensor x = at::randn({3, 1}).set_requires_grad(true);

    Tensor h = relu_autograd(mm_autograd(W1, x));   // [4, 1]
    Tensor y = tanh_autograd(mm_autograd(W2, h));    // [2, 1]
    Tensor loss = sum_autograd(y);
    tensor_backward(loss);

    Tensor gW1 = get_grad(W1);
    Tensor gW2 = get_grad(W2);
    Tensor gx = get_grad(x);
    ASSERT_TRUE(gW1.defined());
    ASSERT_TRUE(gW2.defined());
    ASSERT_TRUE(gx.defined());

    // Check shapes
    ASSERT_EQ(gW1.size(0), 4);
    ASSERT_EQ(gW1.size(1), 3);
    ASSERT_EQ(gW2.size(0), 2);
    ASSERT_EQ(gW2.size(1), 4);

    // Numerical gradient check for x
    Tensor W1_val = W1.clone().detach();
    Tensor W2_val = W2.clone().detach();
    Tensor x_val = x.clone().detach();
    Tensor numerical_gx = numerical_gradient(
        [&W1_val, &W2_val](const Tensor& inp) {
            Tensor h = at::native::mm(W1_val, inp).relu();
            return at::native::mm(W2_val, h).tanh();
        }, x_val, 1e-3);

    check_gradients_close(gx.contiguous(), numerical_gx, 5e-3f, 5e-2f, "multilayer_x");
}

TEST(AutogradFullTest, LogSumExpPattern) {
    // f(x) = log(sum(exp(x))) -- logsumexp
    // df/dx_i = exp(x_i) / sum(exp(x))  = softmax(x)_i
    Tensor x_val = at::tensor({1.0f, 2.0f, 3.0f, 0.5f});
    Tensor x = x_val.clone().set_requires_grad(true);

    Tensor e = exp_autograd(x);
    Tensor s = sum_autograd(e);
    Tensor y = log_autograd(s);
    tensor_backward(y);

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    // Expected: softmax
    const float* xd = x_val.data_ptr<float>();
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < 4; ++i) {
        sum_exp += std::exp(xd[i]);
    }

    const float* gd = grad.data_ptr<float>();
    for (int64_t i = 0; i < 4; ++i) {
        float expected = std::exp(xd[i]) / sum_exp;
        EXPECT_NEAR(gd[i], expected, 1e-4f) << "LogSumExp element " << i;
    }
}

// ============================================================================
//  ADDITIONAL TESTS: Gradient existence and shape correctness
// ============================================================================

TEST(AutogradFullTest, SigmoidGradNumerical) {
    Tensor x_val = at::tensor({-2.0f, -0.5f, 0.0f, 1.0f, 2.5f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = sigmoid_autograd(x);
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.sigmoid(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "sigmoid_numerical");
}

TEST(AutogradFullTest, ReluGradNumerical) {
    // Avoid testing exactly at 0 for numerical stability
    Tensor x_val = at::tensor({-2.0f, -0.5f, 0.1f, 1.0f, 3.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = relu_autograd(x);
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.relu(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "relu_numerical");
}

TEST(AutogradFullTest, SinGradNumerical) {
    Tensor x_val = at::tensor({0.0f, 0.7f, -1.0f, 2.5f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = sin_autograd(x);
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return at::native::sin(t); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "sin_numerical");
}

TEST(AutogradFullTest, CosGradNumerical) {
    Tensor x_val = at::tensor({0.0f, 0.7f, -1.0f, 2.5f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = cos_autograd(x);
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return at::native::cos(t); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "cos_numerical");
}

TEST(AutogradFullTest, ClampGradNumerical) {
    Tensor x_val = at::tensor({-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = clamp_autograd(x, Scalar(-1.0), Scalar(1.0));
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.clamp(Scalar(-1.0), Scalar(1.0)); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "clamp_numerical");
}

TEST(AutogradFullTest, PowScalarGradNumerical) {
    Tensor x_val = at::tensor({0.5f, 1.0f, 2.0f, 3.0f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = pow_autograd(x, Scalar(2.5));
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.pow(Scalar(2.5)); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "pow_scalar_numerical");
}

TEST(AutogradFullTest, NegGradNumerical) {
    Tensor x_val = at::tensor({-1.0f, 0.0f, 2.0f, -3.5f});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = neg_autograd(x);
    tensor_backward(sum_autograd(y));
    Tensor analytical = get_grad(x);

    Tensor numerical = numerical_gradient(
        [](const Tensor& t) { return t.neg(); }, x_val);

    check_gradients_close(analytical, numerical, 1e-3f, 1e-2f, "neg_numerical");
}

// ============================================================================
//  Test: Gradient accumulation (multiple paths to same input)
// ============================================================================

TEST(AutogradFullTest, GradientAccumulation) {
    // z = x + x = 2*x, dz/dx = 2
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor z = add_autograd(x, x);
    tensor_backward(sum_autograd(z));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 2.0f, 1e-5f)
            << "GradAccumulation element " << i;
    }
}

TEST(AutogradFullTest, GradientAccumulationMul) {
    // z = x * x = x^2, dz/dx = 2*x
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f}).set_requires_grad(true);
    Tensor z = mul_autograd(x, x);
    tensor_backward(sum_autograd(z));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());

    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad.data_ptr<float>()[i], 2.0f * xd[i], 1e-5f)
            << "GradAccumMul element " << i;
    }
}

// ============================================================================
//  Test: Operations on higher-dimensional tensors
// ============================================================================

TEST(AutogradFullTest, ExpGrad3D) {
    Tensor x_val = at::randn({2, 3, 4});
    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = exp_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.numel(), 24);

    // grad = exp(x)
    const float* gd = grad.data_ptr<float>();
    const float* xd = x_val.data_ptr<float>();
    for (int64_t i = 0; i < 24; ++i) {
        EXPECT_NEAR(gd[i], std::exp(xd[i]), 1e-4f)
            << "ExpGrad3D element " << i;
    }
}

TEST(AutogradFullTest, MulThenSigmoid) {
    // f(x, y) = sigmoid(x * y), test chain rule through mul then sigmoid
    Tensor x_val = at::tensor({0.5f, 1.0f, -0.5f});
    Tensor y_val = at::tensor({2.0f, -1.0f, 3.0f});

    Tensor x = x_val.clone().set_requires_grad(true);
    Tensor y = y_val.clone().set_requires_grad(true);
    Tensor prod = mul_autograd(x, y);
    Tensor out = sigmoid_autograd(prod);
    tensor_backward(sum_autograd(out));

    Tensor gx = get_grad(x);
    ASSERT_TRUE(gx.defined());

    // Numerical gradient check
    Tensor numerical_gx = numerical_gradient_2(
        [](const Tensor& a, const Tensor& b) {
            return a.mul(b).sigmoid();
        }, x_val, y_val, 1e-3);

    check_gradients_close(gx, numerical_gx, 2e-3f, 5e-2f, "mul_sigmoid_x");
}

// ============================================================================
//  Test: Gradient does NOT flow through detach / no-grad tensors
// ============================================================================

TEST(AutogradFullTest, NoGradTensorHasNoGrad) {
    Tensor x = at::tensor({1.0f, 2.0f, 3.0f});
    // x does NOT require grad
    Tensor y = at::tensor({4.0f, 5.0f, 6.0f}).set_requires_grad(true);
    Tensor z = add_autograd(x, y);
    tensor_backward(sum_autograd(z));

    // x should have no gradient
    Tensor gx = get_grad(x);
    EXPECT_FALSE(gx.defined());

    // y should have gradient = 1
    Tensor gy = get_grad(y);
    ASSERT_TRUE(gy.defined());
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(gy.data_ptr<float>()[i], 1.0f, 1e-5f);
    }
}

// ============================================================================
//  Test: Large tensor gradient (stress test)
// ============================================================================

TEST(AutogradFullTest, LargeTensorGrad) {
    // 1000-element tensor with exp autograd
    Tensor x = at::randn({100, 10}).set_requires_grad(true);
    Tensor y = exp_autograd(x);
    tensor_backward(sum_autograd(y));

    Tensor grad = get_grad(x);
    ASSERT_TRUE(grad.defined());
    ASSERT_EQ(grad.numel(), 1000);

    // Spot check a few elements
    const float* gd = grad.data_ptr<float>();
    const float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(gd[i], std::exp(xd[i]), 1e-4f)
            << "LargeTensorGrad element " << i;
    }
}
