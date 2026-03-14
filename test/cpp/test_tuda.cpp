// ============================================================================
// test_tuda.cpp — TUDA CPU Architecture Dispatch Tests
// ============================================================================
// Tests TUDA vector abstraction (VecF), vectorized math, and BLAS micro-kernels.
// Compares TUDA output against naive reference implementations.

#include "aten/src/ATen/ATen.h"
#include "aten/src/ATen/native/cpu/tuda/TudaConfig.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/tuda/TudaBLAS.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include <chrono>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  [TEST] " << name << "... "; \
    try

#define PASS() \
    std::cout << "PASSED" << std::endl; \
    tests_passed++

#define FAIL(msg) \
    std::cout << "FAILED: " << msg << std::endl; \
    tests_failed++

static bool close(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) < tol;
}

static bool close_rel(float a, float b, float tol = 1e-4f) {
    float denom = std::max(std::abs(a), std::abs(b));
    if (denom < 1e-7f) return std::abs(a - b) < tol;
    return std::abs(a - b) / denom < tol;
}

// ============================================================================
// Test 1: TUDA Config — verify compile-time arch detection
// ============================================================================
void test_config() {
    using namespace at::native::tuda;

    std::cout << "\n=== TUDA Config Tests ===" << std::endl;

    TEST("Architecture detection") {
        const char* arch_name = nullptr;
        switch (kArch) {
            case Arch::AVX2:     arch_name = "AVX2"; break;
            case Arch::NEON_A57: arch_name = "NEON_A57"; break;
            case Arch::NEON_A75: arch_name = "NEON_A75"; break;
            case Arch::E2K_V5:   arch_name = "E2K_V5"; break;
            case Arch::E2K_V6:   arch_name = "E2K_V6"; break;
            case Arch::SCALAR:   arch_name = "SCALAR"; break;
        }
        std::cout << "[" << arch_name << "] ";
        assert(arch_name != nullptr);
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("GEMM tuning constants") {
        assert(kTuning.MR > 0);
        assert(kTuning.NR > 0);
        assert(kTuning.MC > 0);
        assert(kTuning.KC > 0);
        assert(kTuning.NC > 0);
        std::cout << "MR=" << kTuning.MR << " NR=" << kTuning.NR << " ";
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("VecF width") {
        int w = VecF::width;
        assert(w == 1 || w == 4 || w == 8);
        std::cout << "width=" << w << " ";
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 2: VecF operations
// ============================================================================
void test_vecf() {
    using namespace at::native::tuda;

    std::cout << "\n=== VecF Operations Tests ===" << std::endl;

    constexpr int W = VecF::width;

    TEST("broadcast + store") {
        VecF v = VecF::broadcast(3.14f);
        float buf[8] = {};
        v.store(buf);
        for (int i = 0; i < W; ++i)
            assert(close(buf[i], 3.14f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("load + store roundtrip") {
        float src[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        VecF v = VecF::load(src);
        float dst[8] = {};
        v.store(dst);
        for (int i = 0; i < W; ++i)
            assert(close(src[i], dst[i]));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("add / sub / mul / div") {
        float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        VecF va = VecF::load(a);
        VecF vb = VecF::load(b);

        float r[8];
        (va + vb).store(r);
        for (int i = 0; i < W; ++i) assert(close(r[i], a[i] + b[i]));

        (va - vb).store(r);
        for (int i = 0; i < W; ++i) assert(close(r[i], a[i] - b[i]));

        (va * vb).store(r);
        for (int i = 0; i < W; ++i) assert(close(r[i], a[i] * b[i]));

        (va / vb).store(r);
        for (int i = 0; i < W; ++i) assert(close(r[i], a[i] / b[i]));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("fmadd: a*b+c") {
        float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float b[8] = {2, 3, 4, 5, 6, 7, 8, 9};
        float c[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        VecF va = VecF::load(a);
        VecF vb = VecF::load(b);
        VecF vc = VecF::load(c);
        float r[8];
        VecF::fmadd(va, vb, vc).store(r);
        for (int i = 0; i < W; ++i)
            assert(close(r[i], a[i] * b[i] + c[i]));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("hsum") {
        float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        VecF va = VecF::load(a);
        float sum = va.hsum();
        float expected = 0;
        for (int i = 0; i < W; ++i) expected += a[i];
        assert(close(sum, expected));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("max / min") {
        float a[8] = {3, 1, 4, 1, 5, 9, 2, 6};
        float b[8] = {2, 7, 1, 8, 2, 8, 1, 8};
        VecF va = VecF::load(a);
        VecF vb = VecF::load(b);
        float r[8];
        va.max(vb).store(r);
        for (int i = 0; i < W; ++i)
            assert(close(r[i], std::max(a[i], b[i])));
        va.min(vb).store(r);
        for (int i = 0; i < W; ++i)
            assert(close(r[i], std::min(a[i], b[i])));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sqrt") {
        float a[8] = {1, 4, 9, 16, 25, 36, 49, 64};
        VecF va = VecF::load(a);
        float r[8];
        va.sqrt().store(r);
        for (int i = 0; i < W; ++i)
            assert(close(r[i], std::sqrt(a[i])));
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 3: Vectorized math functions
// ============================================================================
void test_math() {
    using namespace at::native::tuda;

    std::cout << "\n=== Vectorized Math Tests ===" << std::endl;

    constexpr int W = VecF::width;
    float input[8] = {-1.5f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    float result[8];

    TEST("exp_vec") {
        exp_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close_rel(result[i], std::exp(input[i]), 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("log_vec (positive inputs)") {
        float pos[8] = {0.1f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f, 10.0f, 100.0f};
        log_vec(VecF::load(pos)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close_rel(result[i], std::log(pos[i]), 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sigmoid_vec") {
        sigmoid_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i) {
            float expected = 1.0f / (1.0f + std::exp(-input[i]));
            assert(close_rel(result[i], expected, 1e-3f));
        }
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("tanh_vec") {
        tanh_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close_rel(result[i], std::tanh(input[i]), 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("relu_vec") {
        relu_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i) {
            float expected = input[i] > 0 ? input[i] : 0;
            assert(close(result[i], expected));
        }
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("neg_vec") {
        neg_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close(result[i], -input[i]));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("abs_vec") {
        abs_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close(result[i], std::abs(input[i])));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("rsqrt_vec") {
        float pos[8] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 0.25f, 0.01f, 100.0f};
        rsqrt_vec(VecF::load(pos)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close_rel(result[i], 1.0f / std::sqrt(pos[i]), 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sin_vec / cos_vec") {
        sin_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close_rel(result[i], std::sin(input[i]), 1e-3f));
        cos_vec(VecF::load(input)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close_rel(result[i], std::cos(input[i]), 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("ceil_vec / floor_vec / round_vec") {
        float vals[8] = {1.3f, 2.7f, -0.5f, -1.8f, 3.0f, 0.1f, -0.1f, 4.5f};
        ceil_vec(VecF::load(vals)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close(result[i], std::ceil(vals[i])));
        floor_vec(VecF::load(vals)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close(result[i], std::floor(vals[i])));
        round_vec(VecF::load(vals)).store(result);
        for (int i = 0; i < W; ++i)
            assert(close(result[i], std::round(vals[i]), 1.0f)); // rounding mode may differ
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 4: SGEMM accuracy
// ============================================================================
void test_sgemm() {
    using namespace at::native::tuda::blas;

    std::cout << "\n=== SGEMM Tests ===" << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto naive_matmul = [](const float* A, const float* B, float* C,
                           int M, int K, int N) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0;
                for (int k = 0; k < K; ++k)
                    sum += A[i * K + k] * B[k * N + j];
                C[i * N + j] = sum;
            }
    };

    auto test_size = [&](int M, int K, int N, const char* label) {
        TEST(label) {
            std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_tuda(M * N);
            for (auto& v : A) v = dist(rng);
            for (auto& v : B) v = dist(rng);

            naive_matmul(A.data(), B.data(), C_ref.data(), M, K, N);
            std::fill(C_tuda.begin(), C_tuda.end(), 0.0f);
            sgemm(M, K, N, 1.0f, A.data(), K, B.data(), N, 0.0f, C_tuda.data(), N);

            float max_err = 0;
            for (int i = 0; i < M * N; ++i)
                max_err = std::max(max_err, std::abs(C_ref[i] - C_tuda[i]));

            if (max_err > 1e-3f) {
                std::string msg = "max_err=" + std::to_string(max_err);
                FAIL(msg.c_str());
            } else {
                std::cout << "max_err=" << max_err << " ";
                PASS();
            }
        } catch (...) { FAIL("exception"); }
    };

    test_size(1, 1, 1, "1x1x1 (trivial)");
    test_size(4, 4, 4, "4x4x4 (small)");
    test_size(16, 16, 16, "16x16x16 (medium)");
    test_size(64, 64, 64, "64x64x64 (L1 fit)");
    test_size(128, 256, 128, "128x256x128 (L2 fit)");
    test_size(7, 13, 11, "7x13x11 (non-aligned)");
    test_size(1, 128, 1, "1x128x1 (dot product)");
    test_size(128, 1, 128, "128x1x128 (outer product)");
}

// ============================================================================
// Test 5: SGEMV accuracy
// ============================================================================
void test_sgemv() {
    using namespace at::native::tuda::blas;

    std::cout << "\n=== SGEMV Tests ===" << std::endl;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto naive_gemv = [](const float* A, const float* x, float* y, int M, int N) {
        for (int i = 0; i < M; ++i) {
            float sum = 0;
            for (int j = 0; j < N; ++j)
                sum += A[i * N + j] * x[j];
            y[i] = sum;
        }
    };

    auto test_size = [&](int M, int N, const char* label) {
        TEST(label) {
            std::vector<float> A(M * N), x(N), y_ref(M), y_tuda(M);
            for (auto& v : A) v = dist(rng);
            for (auto& v : x) v = dist(rng);

            naive_gemv(A.data(), x.data(), y_ref.data(), M, N);
            std::fill(y_tuda.begin(), y_tuda.end(), 0.0f);
            sgemv(M, N, 1.0f, A.data(), N, x.data(), 0.0f, y_tuda.data());

            float max_err = 0;
            for (int i = 0; i < M; ++i)
                max_err = std::max(max_err, std::abs(y_ref[i] - y_tuda[i]));

            if (max_err > 1e-4f) {
                std::string msg = "max_err=" + std::to_string(max_err);
                FAIL(msg.c_str());
            } else {
                PASS();
            }
        } catch (...) { FAIL("exception"); }
    };

    test_size(1, 1, "1x1");
    test_size(16, 16, "16x16");
    test_size(64, 128, "64x128");
    test_size(128, 256, "128x256");
    test_size(7, 13, "7x13 (non-aligned)");
}

// ============================================================================
// Test 6: MathOps through Tensor API (end-to-end)
// ============================================================================
void test_tensor_ops() {
    std::cout << "\n=== Tensor-Level MathOps Tests ===" << std::endl;

    TEST("neg()") {
        auto t = at::rand({32});
        auto r = t.neg();
        for (int i = 0; i < 32; ++i)
            assert(close(r.data_ptr<float>()[i], -t.data_ptr<float>()[i]));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("exp()") {
        auto t = at::rand({32});
        auto r = t.exp();
        for (int i = 0; i < 32; ++i)
            assert(close_rel(r.data_ptr<float>()[i], std::exp(t.data_ptr<float>()[i]), 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sigmoid()") {
        auto t = at::rand({32});
        auto r = t.sigmoid();
        for (int i = 0; i < 32; ++i) {
            float expected = 1.0f / (1.0f + std::exp(-t.data_ptr<float>()[i]));
            assert(close_rel(r.data_ptr<float>()[i], expected, 1e-3f));
        }
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("mm() 16x32 @ 32x8") {
        auto a = at::rand({16, 32});
        auto b = at::rand({32, 8});
        auto c = a.mm(b);
        assert(c.size(0) == 16);
        assert(c.size(1) == 8);
        // Spot-check one element
        float expected = 0;
        for (int k = 0; k < 32; ++k)
            expected += a.data_ptr<float>()[0 * 32 + k] * b.data_ptr<float>()[k * 8 + 0];
        assert(close_rel(c.data_ptr<float>()[0], expected, 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("add() / sub() / mul() / div()") {
        auto a = at::rand({64}) + at::full({64}, 0.1f);  // avoid div by zero
        auto b = at::rand({64}) + at::full({64}, 0.1f);
        auto sum = a.add(b);
        auto diff = a.sub(b);
        auto prod = a.mul(b);
        auto quot = a.div(b);
        for (int i = 0; i < 64; ++i) {
            float ai = a.data_ptr<float>()[i], bi = b.data_ptr<float>()[i];
            assert(close(sum.data_ptr<float>()[i], ai + bi));
            assert(close(diff.data_ptr<float>()[i], ai - bi));
            assert(close(prod.data_ptr<float>()[i], ai * bi));
            assert(close_rel(quot.data_ptr<float>()[i], ai / bi, 1e-4f));
        }
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " TUDA CPU Architecture Dispatch Tests   " << std::endl;
    std::cout << "========================================" << std::endl;

    test_config();
    test_vecf();
    test_math();
    test_sgemm();
    test_sgemv();
    test_tensor_ops();

    std::cout << "\n========================================" << std::endl;
    std::cout << " Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
