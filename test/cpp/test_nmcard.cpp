// ============================================================================
// test_nmcard.cpp - NM Card Mini Emulator Tests
// ============================================================================
// Tests the NMCard emulator backend: device creation, memory allocation,
// Q16.16 fixed-point math, forward/backward ops, optimizer steps.
// Compares with CPU results (tolerance ~1e-2 for Q16.16 precision).

#include "aten/src/ATen/ATen.h"
#include "c10/nmcard/NMCardAllocator.h"
#include "aten/src/ATen/nmcard/NMCardEmulator.h"
#include "aten/src/ATen/nmcard/NMCardMath.h"
#include "aten/src/ATen/nmcard/NMCardHardware.h"

#include <iostream>
#include <cmath>
#include <cassert>

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

bool close(float a, float b, float tol = 1e-2f) {
    return std::abs(a - b) < tol;
}

// ============================================================================
// Test 1: Device parsing and creation
// ============================================================================
void test_device() {
    std::cout << "\n=== Device Tests ===" << std::endl;

    TEST("Parse 'nmcard' string") {
        c10::Device dev("nmcard");
        assert(dev.is_nmcard());
        assert(dev.type() == c10::DeviceType::PrivateUse1);
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("Parse 'nmcard:0' string") {
        c10::Device dev("nmcard:0");
        assert(dev.is_nmcard());
        assert(dev.index() == 0);
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("DeviceTypeName returns 'nmcard'") {
        const char* name = c10::DeviceTypeName(c10::DeviceType::PrivateUse1, true);
        assert(std::string(name) == "nmcard");
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("kNMCard constant") {
        auto dev = c10::kNMCard(0);
        assert(dev.is_nmcard());
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 2: Q16.16 Fixed-Point Math
// ============================================================================
void test_q16_math() {
    std::cout << "\n=== Q16.16 Math Tests ===" << std::endl;
    using namespace at::nmcard;

    TEST("float_to_fixed / fixed_to_float roundtrip") {
        float vals[] = {0.0f, 1.0f, -1.0f, 3.14159f, -2.5f, 0.001f};
        for (float v : vals) {
            float result = fixed_to_float(float_to_fixed(v));
            assert(close(v, result, 2e-4f));
        }
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("mul_fixed") {
        fixed32 a = float_to_fixed(2.5f);
        fixed32 b = float_to_fixed(3.0f);
        float result = fixed_to_float(mul_fixed(a, b));
        assert(close(result, 7.5f, 1e-3f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("div_fixed") {
        fixed32 a = float_to_fixed(10.0f);
        fixed32 b = float_to_fixed(3.0f);
        float result = fixed_to_float(div_fixed(a, b));
        assert(close(result, 3.333f, 1e-2f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sqrt_fixed") {
        fixed32 a = float_to_fixed(9.0f);
        float result = fixed_to_float(sqrt_fixed(a));
        assert(close(result, 3.0f, 1e-2f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("exp_fixed_lut") {
        fixed32 x = float_to_fixed(1.0f);
        float result = fixed_to_float(exp_fixed_lut(x));
        assert(close(result, 2.718f, 0.1f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sigmoid_fixed") {
        fixed32 x = float_to_fixed(0.0f);
        float result = fixed_to_float(sigmoid_fixed(x));
        assert(close(result, 0.5f, 0.05f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("silu_fixed") {
        fixed32 x = float_to_fixed(1.0f);
        float result = fixed_to_float(silu_fixed(x));
        float expected = 1.0f / (1.0f + std::exp(-1.0f)); // sigmoid(1) * 1
        assert(close(result, expected, 0.05f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sin_fixed / cos_fixed") {
        fixed32 x = float_to_fixed(1.0f);
        float sin_result = fixed_to_float(sin_fixed(x));
        float cos_result = fixed_to_float(cos_fixed(x));
        assert(close(sin_result, std::sin(1.0f), 0.05f));
        assert(close(cos_result, std::cos(1.0f), 0.05f));
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 3: Allocator
// ============================================================================
void test_allocator() {
    std::cout << "\n=== Allocator Tests ===" << std::endl;

    TEST("NMCard allocator is registered") {
        // Already registered in main(), just verify it's available
        auto& alloc = c10::nmcard::NMCardAllocator::get();
        (void)alloc; // Verify singleton accessible
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("Allocate and write memory") {
        auto& alloc = c10::nmcard::NMCardAllocator::get();
        auto ptr = alloc.allocate(1024);
        assert(ptr.get() != nullptr);
        assert(ptr.device().is_nmcard());
        // Write and read back
        float* data = static_cast<float*>(ptr.get());
        data[0] = 42.0f;
        assert(data[0] == 42.0f);
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 4: Tensor on NMCard
// ============================================================================
void test_tensor_nmcard() {
    std::cout << "\n=== Tensor NMCard Tests ===" << std::endl;

    TEST("Create tensor on nmcard") {
        auto t = at::empty_nmcard({3, 4});
        assert(t.is_nmcard());
        assert(t.size(0) == 3);
        assert(t.size(1) == 4);
        assert(t.numel() == 12);
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("to_nmcard / nmcard_to_cpu") {
        auto cpu_t = at::ones({2, 3});
        auto nmc_t = at::to_nmcard(cpu_t);
        assert(nmc_t.is_nmcard());
        assert(nmc_t.data_ptr<float>()[0] == 1.0f);

        auto back = at::nmcard_to_cpu(nmc_t);
        assert(back.is_cpu());
        assert(back.data_ptr<float>()[0] == 1.0f);
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("tensor.to(device) dispatch") {
        auto t = at::ones({2, 2});
        auto nmc = t.to(c10::Device("nmcard"));
        assert(nmc.is_nmcard());

        auto cpu = nmc.to(c10::Device("cpu"));
        assert(cpu.is_cpu());
        assert(close(cpu.data_ptr<float>()[0], 1.0f));
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 5: Emulator operations
// ============================================================================
void test_emulator_ops() {
    std::cout << "\n=== Emulator Operation Tests ===" << std::endl;
    auto& emu = at::nmcard::NMCardEmulator::get();

    TEST("MatMul 2x3 @ 3x2") {
        float A[] = {1, 2, 3, 4, 5, 6};  // 2x3
        float B[] = {1, 2, 3, 4, 5, 6};  // 3x2
        float C[4] = {0};
        emu.matmul(A, B, C, 2, 3, 2);
        // C[0,0] = 1*1 + 2*3 + 3*5 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 28
        assert(close(C[0], 22.0f, 0.1f));
        assert(close(C[1], 28.0f, 0.1f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("SiLU") {
        float input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float output[5] = {0};
        emu.silu(input, output, 5);
        for (int i = 0; i < 5; i++) {
            float x = input[i];
            float expected = x / (1.0f + std::exp(-x));
            assert(close(output[i], expected, 0.1f));
        }
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("Softmax") {
        float input[] = {1.0f, 2.0f, 3.0f};
        float output[3] = {0};
        emu.softmax(input, output, 1, 3);
        float sum = output[0] + output[1] + output[2];
        assert(close(sum, 1.0f, 0.05f));
        assert(output[2] > output[1]);
        assert(output[1] > output[0]);
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("RMSNorm") {
        float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float output[4] = {0};
        emu.rmsnorm(input, output, gamma, 1, 4);
        // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        float rms = std::sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-5f);
        assert(close(output[0], 1.0f / rms, 0.05f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("SGD step") {
        float weights[] = {1.0f, 2.0f, 3.0f};
        float grads[] = {0.1f, 0.2f, 0.3f};
        emu.sgd_step(weights, grads, 0.01f, 3);
        assert(close(weights[0], 1.0f - 0.01f * 0.1f));
        assert(close(weights[1], 2.0f - 0.01f * 0.2f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("Adam step") {
        float weights[] = {1.0f, 2.0f};
        float grads[] = {0.5f, 0.5f};
        float m[] = {0.0f, 0.0f};
        float v[] = {0.0f, 0.0f};
        emu.adam_step(weights, grads, m, v, 0.001f, 0.9f, 0.999f, 1e-8f, 2);
        // After one step, weights should decrease
        assert(weights[0] < 1.0f);
        assert(weights[1] < 2.0f);
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Test 6: Fixed-point mode
// ============================================================================
void test_fixed_point_mode() {
    std::cout << "\n=== Fixed-Point Mode Tests ===" << std::endl;
    auto& emu = at::nmcard::NMCardEmulator::get();

    TEST("MatMul in Q16.16 mode") {
        emu.set_fixed_point(true);
        float A[] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2
        float B[] = {5.0f, 6.0f, 7.0f, 8.0f}; // 2x2
        float C[4] = {0};
        emu.matmul(A, B, C, 2, 2, 2);
        // C[0,0] = 1*5 + 2*7 = 19
        assert(close(C[0], 19.0f, 0.5f));
        // C[0,1] = 1*6 + 2*8 = 22
        assert(close(C[1], 22.0f, 0.5f));
        emu.set_fixed_point(false);
        PASS();
    } catch (...) { emu.set_fixed_point(false); FAIL("exception"); }

    TEST("SiLU in Q16.16 mode") {
        emu.set_fixed_point(true);
        float input[] = {0.5f, 1.0f, -0.5f};
        float output[3] = {0};
        emu.silu(input, output, 3);
        for (int i = 0; i < 3; i++) {
            float x = input[i];
            float expected = x / (1.0f + std::exp(-x));
            assert(close(output[i], expected, 0.15f)); // Q16.16 tolerance
        }
        emu.set_fixed_point(false);
        PASS();
    } catch (...) { emu.set_fixed_point(false); FAIL("exception"); }
}

// ============================================================================
// Test 7: Tensor-level dispatch on NMCard
// ============================================================================
void test_tensor_dispatch() {
    std::cout << "\n=== Tensor Dispatch Tests ===" << std::endl;

    TEST("neg() on nmcard") {
        auto t = at::to_nmcard(at::ones({3}));
        auto r = t.neg();
        assert(r.is_nmcard());
        auto cpu_r = at::nmcard_to_cpu(r);
        assert(close(cpu_r.data_ptr<float>()[0], -1.0f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("add() on nmcard") {
        auto a = at::to_nmcard(at::ones({4}));
        auto b = at::to_nmcard(at::ones({4}));
        auto r = a.add(b);
        assert(r.is_nmcard());
        auto cpu_r = at::nmcard_to_cpu(r);
        assert(close(cpu_r.data_ptr<float>()[0], 2.0f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("mul() on nmcard") {
        auto cpu_t = at::ones({3});
        cpu_t.mutable_data_ptr<float>()[0] = 2.0f;
        cpu_t.mutable_data_ptr<float>()[1] = 3.0f;
        cpu_t.mutable_data_ptr<float>()[2] = 4.0f;
        auto a = at::to_nmcard(cpu_t);
        auto b = at::to_nmcard(cpu_t);
        auto r = a.mul(b);
        auto cpu_r = at::nmcard_to_cpu(r);
        assert(close(cpu_r.data_ptr<float>()[0], 4.0f));
        assert(close(cpu_r.data_ptr<float>()[1], 9.0f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("mm() on nmcard") {
        auto A_cpu = at::ones({2, 3});
        auto B_cpu = at::ones({3, 2});
        auto A = at::to_nmcard(A_cpu);
        auto B = at::to_nmcard(B_cpu);
        auto C = A.mm(B);
        assert(C.is_nmcard());
        assert(C.size(0) == 2 && C.size(1) == 2);
        auto cpu_C = at::nmcard_to_cpu(C);
        assert(close(cpu_C.data_ptr<float>()[0], 3.0f)); // 1*1 + 1*1 + 1*1
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("sum() on nmcard") {
        auto cpu_t = at::ones({5});
        auto t = at::to_nmcard(cpu_t);
        auto s = t.sum();
        auto cpu_s = at::nmcard_to_cpu(s);
        assert(close(cpu_s.data_ptr<float>()[0], 5.0f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("zero_() and fill_() on nmcard") {
        auto t = at::empty_nmcard({4});
        t.fill_(at::Scalar(7.0f));
        assert(close(t.data_ptr<float>()[0], 7.0f));
        t.zero_();
        assert(close(t.data_ptr<float>()[0], 0.0f));
        PASS();
    } catch (...) { FAIL("exception"); }

    TEST("item() on nmcard") {
        auto cpu_t = at::ones({1});
        cpu_t.mutable_data_ptr<float>()[0] = 42.0f;
        auto t = at::to_nmcard(cpu_t);
        float val = t.item().to<float>();
        assert(close(val, 42.0f));
        PASS();
    } catch (...) { FAIL("exception"); }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "NM Card Mini Emulator Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Register allocator
    c10::nmcard::register_nmcard_allocator();
    c10::nmcard::register_nmcard_allocator_local();

    test_device();
    test_q16_math();
    test_allocator();
    test_tensor_nmcard();
    test_emulator_ops();
    test_fixed_point_mode();
    test_tensor_dispatch();

    // Hardware detection test — always runs, just reports availability
    std::cout << "\n--- Hardware Detection ---" << std::endl;
    TEST("hardware_detection") {
        auto& hw = at::nmcard::NMCardHardware::get();
        // Don't call init() by default — it would need the real card
        // Just verify the singleton works and is_available() returns false
        if (!hw.is_available()) {
            std::cout << "no card (expected in emulator mode)... ";
            PASS();
        } else {
            std::cout << "HARDWARE DETECTED! " << hw.num_cores() << " core(s)... ";
            PASS();
        }
    } catch (const std::exception& e) {
        FAIL(e.what());
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
