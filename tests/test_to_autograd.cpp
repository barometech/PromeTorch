// ============================================================================
// test_to_autograd.cpp — unit test for torch::autograd::to_autograd().
//
// Covers:
//   1. FP32 -> Half -> FP32 round-trip with grad — gradient flows back to the
//      source FP32 leaf with the right dtype and the right values.
//   2. dtype no-op (`to_autograd(t, t.dtype())`) returns t unchanged and does
//      NOT install a ToBackward node.
//   3. Integer source dtype is silently passed through with no grad_fn (we do
//      not allow integer casts to participate in autograd).
//
// Build: link against aten_cpu + torch autograd.
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/engine.h"
#include <cstdio>
#include <cmath>

using at::Tensor;
using at::TensorOptions;
using c10::ScalarType;

static int failed = 0;
static int passed = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }              \
    else      { ++failed; std::printf("  FAIL: %s\n", msg); }              \
} while (0)

static bool close(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    std::printf("=== to_autograd self-test ===\n");

    // ------------------------------------------------------------------------
    // 1. FP32 -> Half -> FP32, with grad flowing back through both casts.
    //
    // Forward computation:
    //   x_fp32 (leaf, requires_grad=True, [1,2,3,4])
    //   x_half = to_autograd(x_fp32, Half)     # cast down
    //   y_fp32 = to_autograd(x_half, Float)    # cast back up to allow CPU ops
    //   z      = mul(y_fp32, 2)                # autograd-aware
    //   loss   = sum(z)
    //   loss.backward()
    //
    // Expected gradient w.r.t. x_fp32: all-twos, dtype Float.
    // The two ToBackward nodes should each call .to() on the gradient; since
    // the dtype path is Float -> Half -> Float, the round-trip is value-stable
    // for grad=2.0 (representable exactly in fp16).
    // ------------------------------------------------------------------------
    {
        Tensor x = at::zeros({4}, TensorOptions().dtype(ScalarType::Float));
        x.mutable_data_ptr<float>()[0] = 1.0f;
        x.mutable_data_ptr<float>()[1] = 2.0f;
        x.mutable_data_ptr<float>()[2] = 3.0f;
        x.mutable_data_ptr<float>()[3] = 4.0f;
        x.set_requires_grad(true);

        Tensor x_half = torch::autograd::to_autograd(x, ScalarType::Half);
        CHECK(x_half.dtype() == ScalarType::Half,    "forward cast 1: result dtype is Half");
        CHECK(x_half.requires_grad(),                "forward cast 1: grad tracking propagated");

        Tensor y_fp32 = torch::autograd::to_autograd(x_half, ScalarType::Float);
        CHECK(y_fp32.dtype() == ScalarType::Float,   "forward cast 2: result dtype is Float");
        CHECK(y_fp32.requires_grad(),                "forward cast 2: grad tracking propagated");

        Tensor z    = torch::autograd::mul_autograd(y_fp32, at::full({4}, at::Scalar(2.0f),
                                                                     TensorOptions().dtype(ScalarType::Float)));
        Tensor loss = torch::autograd::sum_autograd(z);

        torch::autograd::tensor_backward(loss);

        Tensor g = x.grad();
        CHECK(g.defined(),                           "grad is defined on source FP32 leaf");
        CHECK(g.dtype() == ScalarType::Float,        "grad dtype matches source (Float)");
        CHECK(g.numel() == 4,                        "grad numel matches source");

        const float* gd = g.data_ptr<float>();
        bool values_ok = close(gd[0], 2.0f) && close(gd[1], 2.0f)
                      && close(gd[2], 2.0f) && close(gd[3], 2.0f);
        std::printf("  grad = [%.4f, %.4f, %.4f, %.4f] (expected all 2.0)\n",
                    gd[0], gd[1], gd[2], gd[3]);
        CHECK(values_ok, "grad values are all ~2.0 after FP32->Half->FP32 round-trip");
    }

    // ------------------------------------------------------------------------
    // 2. No-op cast: to_autograd(t, t.dtype()) returns t unchanged.
    //    We assert the underlying TensorImpl is identical (no copy, no new
    //    grad_fn installed).
    // ------------------------------------------------------------------------
    {
        Tensor x = at::zeros({2}, TensorOptions().dtype(ScalarType::Float));
        x.set_requires_grad(true);

        Tensor y = torch::autograd::to_autograd(x, ScalarType::Float);
        CHECK(y.getIntrusivePtr().get() == x.getIntrusivePtr().get(),
              "no-op cast returns the same tensor (zero copy, zero grad_fn)");
    }

    // ------------------------------------------------------------------------
    // 3. Integer source: should NOT install a ToBackward node. Result has the
    //    target dtype but no grad_fn (autograd does not run through int casts).
    // ------------------------------------------------------------------------
    {
        Tensor xi = at::zeros({3}, TensorOptions().dtype(ScalarType::Long));
        // Note: requires_grad on integer leaves is a user error in PyTorch too,
        // but we still want to verify our guard short-circuits cleanly even if
        // the flag were somehow set. We do NOT call set_requires_grad() here
        // because TensorImpl rejects it for integer types — the guard against
        // wiring the backward node lives in to_autograd itself.
        Tensor yf = torch::autograd::to_autograd(xi, ScalarType::Float);
        CHECK(yf.dtype() == ScalarType::Float,       "int -> float cast produces Float result");
        CHECK(!yf.requires_grad(),                   "int -> float cast does not install grad_fn");
    }

    std::printf("\n=== to_autograd: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
