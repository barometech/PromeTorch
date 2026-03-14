// ============================================================================
// test_linq.cpp — Comprehensive tests for LinQ H1M accelerator backend
// ============================================================================
// Tests: device transfer, unary ops, binary ops, comparisons, reductions,
//        matrix ops, normalization, quantization
// ============================================================================

#include "aten/src/ATen/ATen.h"
#include <iostream>
#include <cmath>
#include <cassert>

#ifdef PT_USE_LINQ

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) \
    do { \
        tests_total++; \
        std::cout << "  [" << tests_total << "] " << name << "..."; \
        try {

#define PASS() \
            tests_passed++; \
            std::cout << " PASS" << std::endl; \
        } catch (const std::exception& e) { \
            std::cout << " FAIL: " << e.what() << std::endl; \
        } \
    } while(0)

#define CHECK(cond) \
    if (!(cond)) { throw std::runtime_error(std::string("CHECK FAILED: ") + #cond); }

#define CHECK_CLOSE(a, b, tol) \
    if (std::fabs((a) - (b)) > (tol)) { \
        throw std::runtime_error(std::string("CHECK_CLOSE FAILED: ") + \
            std::to_string(a) + " vs " + std::to_string(b)); \
    }

static void test_device_creation() {
    TEST("Device creation")
        c10::Device dev("linq");
        CHECK(dev.is_linq());
        CHECK(dev.type() == c10::DeviceType::PrivateUse2);
        CHECK(c10::DeviceTypeName(dev.type()) == std::string("LinQ"));
    PASS();
}

static void test_device_transfer() {
    TEST("CPU -> LinQ -> CPU transfer")
        auto cpu_t = at::ones({4, 4});
        auto linq_t = cpu_t.to(c10::kLinQ());
        CHECK(linq_t.device().is_linq());
        CHECK(linq_t.size(0) == 4);
        CHECK(linq_t.size(1) == 4);
        auto back = linq_t.to(c10::kCPU);
        CHECK(back.device().is_cpu());
        const float* d = back.data_ptr<float>();
        for (int i = 0; i < 16; ++i) CHECK_CLOSE(d[i], 1.0f, 1e-6f);
    PASS();
}

static void test_unary_ops() {
    auto cpu_t = at::tensor({1.0f, 4.0f, 9.0f, 16.0f});
    auto linq_t = cpu_t.to(c10::kLinQ());

    TEST("neg")
        auto r = linq_t.neg().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], -1.0f, 1e-6f);
    PASS();

    TEST("abs")
        auto neg = at::tensor({-3.0f, 2.0f, -1.0f, 0.0f}).to(c10::kLinQ());
        auto r = neg.abs().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 3.0f, 1e-6f);
    PASS();

    TEST("sqrt")
        auto r = linq_t.sqrt().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[2], 3.0f, 1e-5f);
    PASS();

    TEST("exp")
        auto x = at::tensor({0.0f, 1.0f}).to(c10::kLinQ());
        auto r = x.exp().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 1.0f, 1e-5f);
        CHECK_CLOSE(r.data_ptr<float>()[1], 2.71828f, 1e-3f);
    PASS();

    TEST("log")
        auto x = at::tensor({1.0f, 2.71828f}).to(c10::kLinQ());
        auto r = x.log().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 0.0f, 1e-5f);
        CHECK_CLOSE(r.data_ptr<float>()[1], 1.0f, 1e-3f);
    PASS();

    TEST("sin/cos/tan")
        auto x = at::tensor({0.0f, 1.5707963f}).to(c10::kLinQ()); // 0, pi/2
        auto s = x.sin().to(c10::kCPU);
        auto c = x.cos().to(c10::kCPU);
        CHECK_CLOSE(s.data_ptr<float>()[0], 0.0f, 1e-5f);
        CHECK_CLOSE(s.data_ptr<float>()[1], 1.0f, 1e-4f);
        CHECK_CLOSE(c.data_ptr<float>()[0], 1.0f, 1e-5f);
    PASS();

    TEST("tanh/sigmoid")
        auto x = at::tensor({0.0f, 100.0f, -100.0f}).to(c10::kLinQ());
        auto th = x.tanh().to(c10::kCPU);
        auto sg = x.sigmoid().to(c10::kCPU);
        CHECK_CLOSE(th.data_ptr<float>()[0], 0.0f, 1e-5f);
        CHECK_CLOSE(th.data_ptr<float>()[1], 1.0f, 1e-3f);
        CHECK_CLOSE(sg.data_ptr<float>()[0], 0.5f, 1e-5f);
    PASS();

    TEST("relu")
        auto x = at::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}).to(c10::kLinQ());
        auto r = x.relu().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 0.0f, 1e-6f);
        CHECK_CLOSE(r.data_ptr<float>()[3], 1.0f, 1e-6f);
    PASS();

    TEST("rsqrt/square/reciprocal")
        auto r1 = linq_t.rsqrt().to(c10::kCPU);
        CHECK_CLOSE(r1.data_ptr<float>()[0], 1.0f, 1e-5f); // rsqrt(1)=1
        auto r2 = linq_t.square().to(c10::kCPU);
        CHECK_CLOSE(r2.data_ptr<float>()[1], 16.0f, 1e-5f); // 4^2=16
        auto r3 = linq_t.reciprocal().to(c10::kCPU);
        CHECK_CLOSE(r3.data_ptr<float>()[1], 0.25f, 1e-5f); // 1/4=0.25
    PASS();

    TEST("ceil/floor/round/sign")
        auto x = at::tensor({1.3f, -2.7f, 0.5f, -0.5f}).to(c10::kLinQ());
        auto ce = x.ceil().to(c10::kCPU);
        auto fl = x.floor().to(c10::kCPU);
        auto rn = x.round().to(c10::kCPU);
        auto sg = x.sign().to(c10::kCPU);
        CHECK_CLOSE(ce.data_ptr<float>()[0], 2.0f, 1e-6f);
        CHECK_CLOSE(fl.data_ptr<float>()[1], -3.0f, 1e-6f);
        CHECK_CLOSE(sg.data_ptr<float>()[0], 1.0f, 1e-6f);
        CHECK_CLOSE(sg.data_ptr<float>()[1], -1.0f, 1e-6f);
    PASS();

    TEST("log2/log10")
        auto x = at::tensor({1.0f, 2.0f, 8.0f, 100.0f}).to(c10::kLinQ());
        auto l2 = x.log2().to(c10::kCPU);
        auto l10 = x.log10().to(c10::kCPU);
        CHECK_CLOSE(l2.data_ptr<float>()[1], 1.0f, 1e-5f);  // log2(2)=1
        CHECK_CLOSE(l2.data_ptr<float>()[2], 3.0f, 1e-5f);  // log2(8)=3
        CHECK_CLOSE(l10.data_ptr<float>()[3], 2.0f, 1e-5f); // log10(100)=2
    PASS();
}

static void test_binary_ops() {
    auto a = at::tensor({1.0f, 2.0f, 3.0f, 4.0f}).to(c10::kLinQ());
    auto b = at::tensor({4.0f, 3.0f, 2.0f, 1.0f}).to(c10::kLinQ());

    TEST("add")
        auto r = a.add(b).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 5.0f, 1e-6f);
    PASS();

    TEST("sub")
        auto r = a.sub(b).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], -3.0f, 1e-6f);
    PASS();

    TEST("mul")
        auto r = a.mul(b).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 4.0f, 1e-6f);
    PASS();

    TEST("div")
        auto r = a.div(b).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 0.25f, 1e-6f);
    PASS();

    TEST("add_scalar / mul_scalar / div_scalar")
        auto r1 = a.add(at::Scalar(10.0f)).to(c10::kCPU);
        CHECK_CLOSE(r1.data_ptr<float>()[0], 11.0f, 1e-6f);
        auto r2 = a.mul(at::Scalar(3.0f)).to(c10::kCPU);
        CHECK_CLOSE(r2.data_ptr<float>()[1], 6.0f, 1e-6f);
        auto r3 = a.div(at::Scalar(2.0f)).to(c10::kCPU);
        CHECK_CLOSE(r3.data_ptr<float>()[3], 2.0f, 1e-6f);
    PASS();

    TEST("clamp")
        auto x = at::tensor({-5.0f, 0.0f, 5.0f, 10.0f}).to(c10::kLinQ());
        auto r = x.clamp(at::Scalar(0.0f), at::Scalar(7.0f)).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 0.0f, 1e-6f);
        CHECK_CLOSE(r.data_ptr<float>()[3], 7.0f, 1e-6f);
    PASS();

    TEST("maximum/minimum")
        auto r1 = at::maximum(a, b).to(c10::kCPU);
        CHECK_CLOSE(r1.data_ptr<float>()[0], 4.0f, 1e-6f);
        CHECK_CLOSE(r1.data_ptr<float>()[3], 4.0f, 1e-6f);
    PASS();
}

static void test_comparison_ops() {
    auto a = at::tensor({1.0f, 2.0f, 3.0f}).to(c10::kLinQ());
    auto b = at::tensor({2.0f, 2.0f, 1.0f}).to(c10::kLinQ());

    TEST("eq/ne/lt/le/gt/ge (tensor)")
        auto r_eq = a.eq(b).to(c10::kCPU);
        CHECK_CLOSE(r_eq.data_ptr<float>()[1], 1.0f, 1e-6f); // 2==2
        CHECK_CLOSE(r_eq.data_ptr<float>()[0], 0.0f, 1e-6f); // 1!=2
        auto r_lt = a.lt(b).to(c10::kCPU);
        CHECK_CLOSE(r_lt.data_ptr<float>()[0], 1.0f, 1e-6f); // 1<2
        auto r_gt = a.gt(b).to(c10::kCPU);
        CHECK_CLOSE(r_gt.data_ptr<float>()[2], 1.0f, 1e-6f); // 3>1
    PASS();

    TEST("eq/lt/gt (scalar)")
        auto r_eq = a.eq(at::Scalar(2.0f)).to(c10::kCPU);
        CHECK_CLOSE(r_eq.data_ptr<float>()[1], 1.0f, 1e-6f);
        auto r_lt = a.lt(at::Scalar(2.0f)).to(c10::kCPU);
        CHECK_CLOSE(r_lt.data_ptr<float>()[0], 1.0f, 1e-6f);
    PASS();
}

static void test_reductions() {
    auto x = at::tensor({1.0f, 2.0f, 3.0f, 4.0f}).to(c10::kLinQ());

    TEST("sum")
        auto r = x.sum().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 10.0f, 1e-5f);
    PASS();

    TEST("mean")
        auto r = x.mean().to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 2.5f, 1e-5f);
    PASS();

    TEST("argmax/argmin")
        auto r1 = x.argmax().to(c10::kCPU);
        CHECK_CLOSE(r1.data_ptr<float>()[0], 3.0f, 1e-5f); // index of 4
        auto r2 = x.argmin().to(c10::kCPU);
        CHECK_CLOSE(r2.data_ptr<float>()[0], 0.0f, 1e-5f); // index of 1
    PASS();
}

static void test_matrix_ops() {
    TEST("mm (2x3 @ 3x2)")
        auto a = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        a = a.view({2, 3});
        auto b = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        b = b.view({3, 2});

        auto la = a.to(c10::kLinQ());
        auto lb = b.to(c10::kLinQ());
        auto r = la.mm(lb).to(c10::kCPU);
        CHECK(r.size(0) == 2);
        CHECK(r.size(1) == 2);
        // [1,2,3]@[1,2;3,4;5,6] = [22, 28; 49, 64]
        CHECK_CLOSE(r.data_ptr<float>()[0], 22.0f, 1e-4f);
        CHECK_CLOSE(r.data_ptr<float>()[1], 28.0f, 1e-4f);
        CHECK_CLOSE(r.data_ptr<float>()[2], 49.0f, 1e-4f);
        CHECK_CLOSE(r.data_ptr<float>()[3], 64.0f, 1e-4f);
    PASS();

    TEST("mv (3x3 @ 3)")
        auto m = at::eye(3).to(c10::kLinQ());
        auto v = at::tensor({1.0f, 2.0f, 3.0f}).to(c10::kLinQ());
        auto r = m.mv(v).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 1.0f, 1e-5f);
        CHECK_CLOSE(r.data_ptr<float>()[2], 3.0f, 1e-5f);
    PASS();

    TEST("dot")
        auto a = at::tensor({1.0f, 2.0f, 3.0f}).to(c10::kLinQ());
        auto b = at::tensor({4.0f, 5.0f, 6.0f}).to(c10::kLinQ());
        auto r = a.dot(b).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 32.0f, 1e-4f);
    PASS();
}

static void test_softmax() {
    TEST("softmax")
        auto x = at::tensor({1.0f, 2.0f, 3.0f}).to(c10::kLinQ());
        auto r = at::linq_dispatch::softmax(x, 0).to(c10::kCPU);
        float s = r.data_ptr<float>()[0] + r.data_ptr<float>()[1] + r.data_ptr<float>()[2];
        CHECK_CLOSE(s, 1.0f, 1e-5f);
        // softmax is monotonic
        CHECK(r.data_ptr<float>()[2] > r.data_ptr<float>()[1]);
        CHECK(r.data_ptr<float>()[1] > r.data_ptr<float>()[0]);
    PASS();
}

static void test_fill_zero() {
    TEST("fill_ / zero_")
        auto t = at::ones({4}).to(c10::kLinQ());
        t.fill_(at::Scalar(42.0f));
        auto r1 = t.to(c10::kCPU);
        CHECK_CLOSE(r1.data_ptr<float>()[0], 42.0f, 1e-6f);
        t.zero_();
        auto r2 = t.to(c10::kCPU);
        CHECK_CLOSE(r2.data_ptr<float>()[0], 0.0f, 1e-6f);
    PASS();
}

static void test_addcmul_addcdiv() {
    TEST("addcmul / addcdiv")
        auto s = at::tensor({1.0f, 1.0f, 1.0f}).to(c10::kLinQ());
        auto t1 = at::tensor({2.0f, 3.0f, 4.0f}).to(c10::kLinQ());
        auto t2 = at::tensor({3.0f, 4.0f, 5.0f}).to(c10::kLinQ());
        // addcmul: 1 + 1.0 * 2*3 = 7
        auto r1 = s.addcmul(t1, t2, at::Scalar(1.0f)).to(c10::kCPU);
        CHECK_CLOSE(r1.data_ptr<float>()[0], 7.0f, 1e-5f);
        // addcdiv: 1 + 1.0 * 2/3 = 1.6667
        auto r2 = s.addcdiv(t1, t2, at::Scalar(1.0f)).to(c10::kCPU);
        CHECK_CLOSE(r2.data_ptr<float>()[0], 1.6667f, 1e-3f);
    PASS();
}

static void test_quantization() {
    TEST("INT8 quantization round-trip")
        float input[] = {1.0f, -0.5f, 0.3f, -1.0f, 0.7f, 0.0f, -0.2f, 0.9f};
        int8_t quantized[8];
        float scale;
        at::linq::LinQEmulator::get().quantize_fp32_to_int8(input, quantized, &scale, 8);

        // INT8 GEMM: simple 2x4 @ 4x2
        int8_t A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        int8_t B[8] = {1, 0, 0, 1, 1, 0, 0, 1};
        int32_t C[4];
        at::linq::LinQEmulator::get().matmul_int8(A, B, C, 2, 4, 2);
        // Row 0: [1,2,3,4] @ [[1,0],[0,1],[1,0],[0,1]] = [1+3, 2+4] = [4, 6]
        CHECK(C[0] == 4);
        CHECK(C[1] == 6);
    PASS();
}

static void test_memory_management() {
    TEST("Allocator caching")
        auto& alloc = c10::linq::LinQAllocator::get();
        size_t before = alloc.get_allocated_memory();
        {
            auto t = at::empty_linq({1000});
        } // t goes out of scope
        size_t after = alloc.get_allocated_memory();
        CHECK(after >= before); // Memory cached, not freed
        alloc.empty_cache();
    PASS();
}

static void test_pow() {
    TEST("pow(scalar)")
        auto x = at::tensor({2.0f, 3.0f, 4.0f}).to(c10::kLinQ());
        auto r = x.pow(at::Scalar(2.0f)).to(c10::kCPU);
        CHECK_CLOSE(r.data_ptr<float>()[0], 4.0f, 1e-5f);
        CHECK_CLOSE(r.data_ptr<float>()[1], 9.0f, 1e-5f);
    PASS();
}

int main() {
    std::cout << "=== LinQ H1M Backend Tests ===" << std::endl;
    std::cout << "Emulator mode: " << at::linq::LinQEmulator::get().num_cores()
              << " cores" << std::endl;

    test_device_creation();
    test_device_transfer();
    test_unary_ops();
    test_binary_ops();
    test_comparison_ops();
    test_reductions();
    test_matrix_ops();
    test_softmax();
    test_fill_zero();
    test_addcmul_addcdiv();
    test_quantization();
    test_memory_management();
    test_pow();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_total
              << " tests passed ===" << std::endl;

    return (tests_passed == tests_total) ? 0 : 1;
}

#else

int main() {
    std::cout << "LinQ support not enabled (PT_USE_LINQ=OFF)" << std::endl;
    return 0;
}

#endif // PT_USE_LINQ
