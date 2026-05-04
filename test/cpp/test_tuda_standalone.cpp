// ============================================================================
// test_tuda_standalone.cpp — TUDA tests без зависимостей от Tensor/autograd
// ============================================================================
// Только TUDA primitives: VecF, vectorized math, sgemm/sgemv micro-kernels.
// Линкуется ТОЛЬКО с aten_cpu, не требует torch_autograd. Подходит для
// быстрой проверки сборки на новой платформе (Эльбрус под Альт/Astra,
// Baikal, ARM, или scalar fallback).
//
// Сравнивает TUDA output с naive reference. Выход: 0 = pass, 1 = fail.
// ============================================================================

#include "aten/src/ATen/native/cpu/tuda/TudaConfig.h"
#include "aten/src/ATen/native/cpu/tuda/TudaVec.h"
#include "aten/src/ATen/native/cpu/tuda/TudaMath.h"
#include "aten/src/ATen/native/cpu/tuda/TudaBLAS.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include <algorithm>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { std::cout << "  [TEST] " << std::left << std::setw(40) << name << " "; } while(0)

#define PASS() \
    do { std::cout << "PASS" << std::endl; tests_passed++; } while(0)

#define FAIL(msg) \
    do { std::cout << "FAIL: " << msg << std::endl; tests_failed++; } while(0)

static bool close_abs(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) < tol;
}

static bool close_rel(float a, float b, float tol = 1e-4f) {
    float denom = std::max(std::abs(a), std::abs(b));
    if (denom < 1e-7f) return std::abs(a - b) < tol;
    return std::abs(a - b) / denom < tol;
}

// ============================================================================
// Test: TUDA arch detection
// ============================================================================
static void test_arch() {
    using namespace at::native::tuda;
    std::cout << "\n=== TUDA Arch ===" << std::endl;

    TEST("Architecture detected");
    const char* name = nullptr;
    switch (kArch) {
        case Arch::AVX2:     name = "AVX2"; break;
        case Arch::NEON_A57: name = "NEON_A57"; break;
        case Arch::NEON_A75: name = "NEON_A75"; break;
        case Arch::E2K_V5:   name = "E2K_V5"; break;
        case Arch::E2K_V6:   name = "E2K_V6"; break;
        case Arch::SCALAR:   name = "SCALAR"; break;
    }
    if (!name) { FAIL("kArch not recognised"); return; }
    std::cout << "[" << name << "] ";
    PASS();

    TEST("VecF::width sane");
    int w = VecF::width;
    if (w < 1 || w > 16) { FAIL("width out of range"); return; }
    std::cout << "[width=" << w << "] ";
    PASS();
}

// ============================================================================
// Test: VecF basic ops
// ============================================================================
static void test_vecf() {
    using namespace at::native::tuda;
    std::cout << "\n=== VecF ===" << std::endl;
    constexpr int W = VecF::width;

    TEST("broadcast + store");
    {
        float buf[16] = {};
        VecF::broadcast(3.14f).store(buf);
        bool ok = true;
        for (int i = 0; i < W; ++i) ok = ok && close_abs(buf[i], 3.14f);
        if (ok) PASS(); else FAIL("mismatch");
    }

    TEST("add/sub/mul/div");
    {
        float a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float b[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        float r[16];
        VecF va = VecF::load(a), vb = VecF::load(b);
        bool ok = true;
        (va + vb).store(r);
        for (int i = 0; i < W; ++i) ok = ok && close_abs(r[i], a[i] + b[i]);
        (va - vb).store(r);
        for (int i = 0; i < W; ++i) ok = ok && close_abs(r[i], a[i] - b[i]);
        (va * vb).store(r);
        for (int i = 0; i < W; ++i) ok = ok && close_abs(r[i], a[i] * b[i]);
        (va / vb).store(r);
        for (int i = 0; i < W; ++i) ok = ok && close_rel(r[i], a[i] / b[i]);
        if (ok) PASS(); else FAIL("arithmetic mismatch");
    }

    TEST("fmadd a*b+c");
    {
        float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        float b[16] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
        float c[16] = {10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160};
        float r[16];
        VecF::fmadd(VecF::load(a), VecF::load(b), VecF::load(c)).store(r);
        bool ok = true;
        for (int i = 0; i < W; ++i) ok = ok && close_rel(r[i], a[i] * b[i] + c[i]);
        if (ok) PASS(); else FAIL("fmadd mismatch");
    }

    TEST("hsum");
    {
        float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        float s = VecF::load(a).hsum();
        float exp = 0;
        for (int i = 0; i < W; ++i) exp += a[i];
        if (close_rel(s, exp, 1e-3f)) PASS(); else FAIL("hsum mismatch");
    }

    TEST("sqrt");
    {
        float a[16] = {1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256};
        float r[16];
        VecF::load(a).sqrt().store(r);
        bool ok = true;
        for (int i = 0; i < W; ++i) ok = ok && close_rel(r[i], std::sqrt(a[i]), 1e-3f);
        if (ok) PASS(); else FAIL("sqrt mismatch");
    }
}

// ============================================================================
// Test: vectorized math
// ============================================================================
static void test_math() {
    using namespace at::native::tuda;
    std::cout << "\n=== TudaMath ===" << std::endl;
    constexpr int W = VecF::width;

    float in[16] = {-1.5f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f,
                    -2.0f, -1.0f, -0.1f, 0.1f, 1.2f, 1.8f, 2.2f, 3.0f};
    float r[16];

    TEST("exp_vec");
    exp_vec(VecF::load(in)).store(r);
    {
        bool ok = true;
        for (int i = 0; i < W; ++i) ok = ok && close_rel(r[i], std::exp(in[i]), 5e-3f);
        if (ok) PASS(); else FAIL("exp mismatch");
    }

    TEST("sigmoid_vec");
    sigmoid_vec(VecF::load(in)).store(r);
    {
        bool ok = true;
        for (int i = 0; i < W; ++i) {
            float exp_v = 1.0f / (1.0f + std::exp(-in[i]));
            ok = ok && close_rel(r[i], exp_v, 5e-3f);
        }
        if (ok) PASS(); else FAIL("sigmoid mismatch");
    }

    TEST("tanh_vec");
    tanh_vec(VecF::load(in)).store(r);
    {
        bool ok = true;
        for (int i = 0; i < W; ++i) ok = ok && close_rel(r[i], std::tanh(in[i]), 5e-3f);
        if (ok) PASS(); else FAIL("tanh mismatch");
    }

    TEST("relu_vec");
    relu_vec(VecF::load(in)).store(r);
    {
        bool ok = true;
        for (int i = 0; i < W; ++i) ok = ok && close_abs(r[i], in[i] > 0 ? in[i] : 0.0f);
        if (ok) PASS(); else FAIL("relu mismatch");
    }
}

// ============================================================================
// Test: sgemm correctness against naive reference
// ============================================================================
static void test_sgemm() {
    using namespace at::native::tuda::blas;
    std::cout << "\n=== sgemm ===" << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto naive = [](const float* A, const float* B, float* C, int M, int K, int N) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float s = 0;
                for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
                C[i * N + j] = s;
            }
    };

    auto run = [&](int M, int K, int N, const char* label) {
        TEST(label);
        std::vector<float> A(M * K), B(K * N), Cref(M * N), Ctuda(M * N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        naive(A.data(), B.data(), Cref.data(), M, K, N);
        std::fill(Ctuda.begin(), Ctuda.end(), 0.0f);
        sgemm(M, K, N, 1.0f, A.data(), K, B.data(), N, 0.0f, Ctuda.data(), N);
        float err = 0;
        for (int i = 0; i < M * N; ++i) err = std::max(err, std::abs(Cref[i] - Ctuda[i]));
        if (err < 1e-3f) {
            std::cout << "[err=" << err << "] ";
            PASS();
        } else {
            FAIL("max_err=" + std::to_string(err));
        }
    };

    run(1, 1, 1, "1x1x1");
    run(4, 4, 4, "4x4x4");
    run(16, 16, 16, "16x16x16");
    run(64, 64, 64, "64x64x64 L1");
    run(128, 256, 128, "128x256x128 L2");
    run(7, 13, 11, "7x13x11 odd");
    run(1, 128, 1, "1x128x1 dot");
    run(128, 1, 128, "128x1x128 outer");
}

// ============================================================================
// Test: sgemv correctness
// ============================================================================
static void test_sgemv() {
    using namespace at::native::tuda::blas;
    std::cout << "\n=== sgemv ===" << std::endl;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto naive = [](const float* A, const float* x, float* y, int M, int N) {
        for (int i = 0; i < M; ++i) {
            float s = 0;
            for (int j = 0; j < N; ++j) s += A[i * N + j] * x[j];
            y[i] = s;
        }
    };

    auto run = [&](int M, int N, const char* label) {
        TEST(label);
        std::vector<float> A(M * N), x(N), yref(M), ytuda(M);
        for (auto& v : A) v = dist(rng);
        for (auto& v : x) v = dist(rng);
        naive(A.data(), x.data(), yref.data(), M, N);
        std::fill(ytuda.begin(), ytuda.end(), 0.0f);
        sgemv(M, N, 1.0f, A.data(), N, x.data(), 0.0f, ytuda.data());
        float err = 0;
        for (int i = 0; i < M; ++i) err = std::max(err, std::abs(yref[i] - ytuda[i]));
        if (err < 1e-4f) PASS(); else FAIL("max_err=" + std::to_string(err));
    };

    run(1, 1, "1x1");
    run(16, 16, "16x16");
    run(64, 128, "64x128");
    run(128, 256, "128x256");
    run(7, 13, "7x13 odd");
    run(1, 1024, "1x1024 large dot");
}

// ============================================================================
// Main — без atexit, без globals, без std::terminate. Только основное.
// ============================================================================
int main() {
    std::cout << "================================================" << std::endl;
    std::cout << " PromeTorch · TUDA standalone tests             " << std::endl;
    std::cout << " (без зависимости от torch_autograd / at::Tensor) " << std::endl;
    std::cout << "================================================" << std::endl;

    test_arch();
    test_vecf();
    test_math();
    test_sgemm();
    test_sgemv();

    std::cout << "\n================================================" << std::endl;
    std::cout << " " << tests_passed << " passed · " << tests_failed << " failed" << std::endl;
    std::cout << "================================================" << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
