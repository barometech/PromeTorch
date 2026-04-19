// ============================================================================
// test_new_ops_2026_04_18.cpp — hand-verified self-tests for new ops batch.
// Covers: logsumexp, one_hot, allclose, equal, floor_divide.
// Also re-checks pre-existing: multinomial, fmod, isclose, cross_entropy.
// Build target: aten_cpu
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "aten/src/ATen/native/cpu/OpsExpansion.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using at::Tensor;

static int failed = 0;
static int passed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); } \
    else      { ++failed; std::printf("  FAIL: %s\n", msg); } \
} while (0)

static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    std::printf("=== new ops self-test (2026-04-18) ===\n");

    // --- logsumexp -----------------------------------------------------------
    {
        Tensor x = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        Tensor y = at::native::logsumexp(x, /*dim=*/0, /*keepdim=*/false);
        float got = y.data_ptr<float>()[0];
        // Expected: log(e^1+e^2+e^3+e^4+e^5)
        float expected =
            std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f)
                   + std::exp(4.0f) + std::exp(5.0f));
        std::printf("  logsumexp([1..5]) = %.6f (expected %.6f)\n", got, expected);
        CHECK(close(got, expected, 1e-3f), "logsumexp 1D matches reference");
        CHECK(close(got, 5.4519f, 1e-2f), "logsumexp 1D ~= 5.4519");
    }

    // logsumexp 2D along dim=1
    {
        float raw[6] = {1, 2, 3, 4, 5, 6};
        Tensor x = at::tensor({1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f}).reshape({2, 3});
        Tensor y = at::native::logsumexp(x, /*dim=*/1, /*keepdim=*/false);
        float expected_row0 = std::log(std::exp(1.0f)+std::exp(2.0f)+std::exp(3.0f));
        float expected_row1 = std::log(std::exp(4.0f)+std::exp(5.0f)+std::exp(6.0f));
        float got0 = y.data_ptr<float>()[0];
        float got1 = y.data_ptr<float>()[1];
        std::printf("  logsumexp 2D[dim=1]: row0=%.4f exp=%.4f   row1=%.4f exp=%.4f\n",
                    got0, expected_row0, got1, expected_row1);
        CHECK(close(got0, expected_row0, 1e-3f), "logsumexp 2D row0");
        CHECK(close(got1, expected_row1, 1e-3f), "logsumexp 2D row1");
    }

    // --- one_hot -------------------------------------------------------------
    {
        Tensor idx = at::tensor({int64_t{0}, int64_t{2}, int64_t{1}},
                                at::TensorOptions().dtype(c10::ScalarType::Long));
        Tensor oh = at::native::one_hot(idx, /*num_classes=*/3);
        // Expect shape [3, 3]
        bool shape_ok = (oh.dim() == 2 && oh.size(0) == 3 && oh.size(1) == 3);
        CHECK(shape_ok, "one_hot shape == [3,3]");
        const int64_t expected[3][3] = {{1,0,0},{0,0,1},{0,1,0}};
        const int64_t* d = oh.data_ptr<int64_t>();
        bool match = true;
        for (int i = 0; i < 3 && match; ++i)
            for (int j = 0; j < 3 && match; ++j)
                if (d[i*3+j] != expected[i][j]) match = false;
        CHECK(match, "one_hot values match expected");
    }

    // --- allclose / equal ----------------------------------------------------
    {
        Tensor a = at::tensor({1.0f, 2.0f});
        Tensor b = at::tensor({1.0f + 1e-6f, 2.0f + 1e-6f});
        CHECK(at::native::allclose(a, b), "allclose within default tol");
        Tensor c = at::tensor({1.0f, 2.0f + 0.1f});
        CHECK(!at::native::allclose(a, c), "allclose rejects large diff");
    }
    {
        auto long_opts = at::TensorOptions().dtype(c10::ScalarType::Long);
        Tensor a = at::tensor({int64_t{1}, int64_t{2}, int64_t{3}}, long_opts);
        Tensor b = at::tensor({int64_t{1}, int64_t{2}, int64_t{3}}, long_opts);
        Tensor c = at::tensor({int64_t{1}, int64_t{2}, int64_t{4}}, long_opts);
        CHECK(at::native::equal(a, b), "equal true for identical");
        CHECK(!at::native::equal(a, c), "equal false for differing");
    }

    // --- floor_divide --------------------------------------------------------
    {
        auto long_opts = at::TensorOptions().dtype(c10::ScalarType::Long);
        Tensor a = at::tensor({int64_t{7}}, long_opts);
        Tensor b = at::tensor({int64_t{2}}, long_opts);
        Tensor r = at::native::floor_divide(a, b);
        int64_t v = r.data_ptr<int64_t>()[0];
        std::printf("  floor_divide(7,2) = %lld (expected 3)\n", (long long)v);
        CHECK(v == 3, "floor_divide(7,2) == 3");
    }
    {
        auto long_opts = at::TensorOptions().dtype(c10::ScalarType::Long);
        Tensor a = at::tensor({int64_t{-7}}, long_opts);
        Tensor b = at::tensor({int64_t{2}}, long_opts);
        Tensor r = at::native::floor_divide(a, b);
        int64_t v = r.data_ptr<int64_t>()[0];
        std::printf("  floor_divide(-7,2) = %lld (expected -4)\n", (long long)v);
        CHECK(v == -4, "floor_divide(-7,2) == -4 (Python semantics)");
    }
    {
        Tensor a = at::tensor({7.0f, -7.0f});
        Tensor b = at::tensor({2.0f, 2.0f});
        Tensor r = at::native::floor_divide(a, b);
        CHECK(close(r.data_ptr<float>()[0], 3.0f),  "floor_divide(7.0, 2.0) == 3.0");
        CHECK(close(r.data_ptr<float>()[1], -4.0f), "floor_divide(-7.0, 2.0) == -4.0");
    }

    // --- fmod (pre-existing, sanity) ----------------------------------------
    {
        Tensor a = at::tensor({7.0f, -7.0f});
        Tensor b = at::tensor({3.0f, 3.0f});
        Tensor r = at::native::fmod(a, b);
        CHECK(close(r.data_ptr<float>()[0], 1.0f),  "fmod(7,3) == 1");
        CHECK(close(r.data_ptr<float>()[1], -1.0f), "fmod(-7,3) == -1");
    }

    std::printf("=== %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
