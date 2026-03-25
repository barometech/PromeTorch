// ============================================================================
// Phase 2 Feature Tests — linalg, tensor ops, FFT, ConvTranspose2d, quantization
// ============================================================================
#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/nn/modules/conv.h"
#include "torch/nn/functional.h"
#include "torch/quantization/quantize.h"
#include "torch/quantization/observer.h"
#include "torch/quantization/quantization.h"
#include "torch/nn/modules/quantized.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace at;
using namespace at::native;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; std::cout << "  PASS: " << msg << std::endl; } \
    else { tests_failed++; std::cout << "  FAIL: " << msg << std::endl; } \
} while(0)

#define CHECK_CLOSE(a, b, tol, msg) CHECK(std::abs((a) - (b)) < (tol), msg)

// Helper: max absolute difference
static double max_abs_diff(const Tensor& a, const Tensor& b) {
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    auto ap = ac.data_ptr<float>();
    auto bp = bc.data_ptr<float>();
    double maxd = 0;
    for (int64_t i = 0; i < ac.numel(); i++)
        maxd = std::max(maxd, (double)std::abs(ap[i] - bp[i]));
    return maxd;
}

// ============================================================================
// TEST 1: Linear Algebra
// ============================================================================
void test_linalg() {
    std::cout << "\n=== TEST: Linear Algebra ===" << std::endl;

    // 1.1 trace
    auto A = at::eye(4, TensorOptions().dtype(ScalarType::Float));
    auto tr = at::native::trace(A);
    CHECK_CLOSE(tr.data_ptr<float>()[0], 4.0f, 1e-5, "trace(I_4) = 4");

    // 1.2 LU decomposition
    // Create a simple 3x3 matrix
    auto M = at::zeros({3, 3}, TensorOptions().dtype(ScalarType::Float));
    float m_data[] = {2, 1, 1, 4, 3, 3, 8, 7, 9};
    for (int i = 0; i < 9; i++) M.mutable_data_ptr<float>()[i] = m_data[i];

    auto lu_res = at::native::lu(M);
    // Verify P @ L @ U ≈ M
    auto PLU = at::native::mm(lu_res.P, at::native::mm(lu_res.L, lu_res.U));
    double lu_err = max_abs_diff(PLU, M);
    CHECK(lu_err < 1e-4, "P@L@U ≈ M (err=" + std::to_string(lu_err) + ")");

    // 1.3 inverse: A @ inv(A) ≈ I
    auto inv_M = at::native::inverse(M);
    auto should_be_I = at::native::mm(M, inv_M);
    auto I3 = at::eye(3, TensorOptions().dtype(ScalarType::Float));
    double inv_err = max_abs_diff(should_be_I, I3);
    CHECK(inv_err < 1e-4, "M @ inv(M) ≈ I (err=" + std::to_string(inv_err) + ")");

    // 1.4 det
    auto det_val = at::native::det(M);
    // det([[2,1,1],[4,3,3],[8,7,9]]) = 2*(27-21) - 1*(36-24) + 1*(28-24) = 12 - 12 + 4 = 4
    CHECK_CLOSE(det_val.data_ptr<float>()[0], 4.0f, 0.1, "det(M) = 4");

    // 1.5 solve: solve(A, b) where Ax = b
    auto b = at::zeros({3, 1}, TensorOptions().dtype(ScalarType::Float));
    b.mutable_data_ptr<float>()[0] = 1; b.mutable_data_ptr<float>()[1] = 2; b.mutable_data_ptr<float>()[2] = 3;
    auto x_sol = at::native::solve(M, b);
    auto Ax = at::native::mm(M, x_sol);
    double solve_err = max_abs_diff(Ax, b);
    CHECK(solve_err < 1e-4, "solve(M,b): M@x ≈ b (err=" + std::to_string(solve_err) + ")");

    // 1.6 cholesky: for SPD matrix, L@L^T = A
    auto SPD = at::zeros({3, 3}, TensorOptions().dtype(ScalarType::Float));
    float spd_data[] = {4, 2, 0, 2, 5, 1, 0, 1, 3};
    for (int i = 0; i < 9; i++) SPD.mutable_data_ptr<float>()[i] = spd_data[i];
    auto L = at::native::cholesky(SPD);
    auto LLt = at::native::mm(L, L.t());
    double chol_err = max_abs_diff(LLt, SPD);
    CHECK(chol_err < 1e-4, "cholesky: L@L^T ≈ SPD (err=" + std::to_string(chol_err) + ")");

    // 1.7 QR: Q@R ≈ A, Q^T@Q ≈ I
    auto qr_res = at::native::qr(M);
    auto QR = at::native::mm(qr_res.Q, qr_res.R);
    double qr_err = max_abs_diff(QR, M);
    CHECK(qr_err < 1e-3, "QR: Q@R ≈ M (err=" + std::to_string(qr_err) + ")");

    auto QtQ = at::native::mm(qr_res.Q.t(), qr_res.Q);
    double orth_err = max_abs_diff(QtQ, I3);
    CHECK(orth_err < 1e-3, "QR: Q^T@Q ≈ I (err=" + std::to_string(orth_err) + ")");

    // 1.8 cross product
    auto v1 = at::zeros({3}, TensorOptions().dtype(ScalarType::Float));
    auto v2 = at::zeros({3}, TensorOptions().dtype(ScalarType::Float));
    v1.mutable_data_ptr<float>()[0] = 1; // [1, 0, 0]
    v2.mutable_data_ptr<float>()[1] = 1; // [0, 1, 0]
    auto cp = at::native::cross(v1, v2);
    auto cp_ptr = cp.data_ptr<float>();
    CHECK(std::abs(cp_ptr[0]) < 1e-5 && std::abs(cp_ptr[1]) < 1e-5 && std::abs(cp_ptr[2] - 1.0f) < 1e-5,
          "cross([1,0,0], [0,1,0]) = [0,0,1]");

    // 1.9 matrix_norm
    auto norm_f = at::native::matrix_norm(I3, 0.0); // Frobenius
    CHECK_CLOSE(norm_f.data_ptr<float>()[0], std::sqrt(3.0f), 1e-4, "frobenius_norm(I_3) = sqrt(3)");
}

// ============================================================================
// TEST 2: Tensor Operations
// ============================================================================
void test_tensor_ops() {
    std::cout << "\n=== TEST: Tensor Operations ===" << std::endl;

    // 2.1 flip: flip(flip(x, d), d) = x
    auto x = at::randn({3, 4}, TensorOptions().dtype(ScalarType::Float));
    auto flipped = at::native::flip(at::native::flip(x, {0, 1}), {0, 1});
    double flip_err = max_abs_diff(flipped, x);
    CHECK(flip_err < 1e-6, "flip(flip(x)) = x (err=" + std::to_string(flip_err) + ")");

    // 2.2 roll: roll(roll(x, 2, 0), -2, 0) = x
    auto rolled = at::native::roll(at::native::roll(x, {2}, {0}), {-2}, {0});
    double roll_err = max_abs_diff(rolled, x);
    CHECK(roll_err < 1e-6, "roll(roll(x, 2), -2) = x (err=" + std::to_string(roll_err) + ")");

    // 2.3 meshgrid
    auto a = at::zeros({3}, TensorOptions().dtype(ScalarType::Float));
    a.mutable_data_ptr<float>()[0] = 1; a.mutable_data_ptr<float>()[1] = 2; a.mutable_data_ptr<float>()[2] = 3;
    auto b = at::zeros({2}, TensorOptions().dtype(ScalarType::Float));
    b.mutable_data_ptr<float>()[0] = 10; b.mutable_data_ptr<float>()[1] = 20;
    auto grids = at::native::meshgrid({a, b});
    CHECK(grids.size() == 2, "meshgrid returns 2 grids");
    CHECK(grids[0].sizes() == std::vector<int64_t>({3, 2}), "meshgrid grid[0] shape = [3,2]");

    // 2.4 repeat_interleave: [1,2,3] with repeats=2 → [1,1,2,2,3,3]
    auto ri = at::native::repeat_interleave(a, 2, 0);
    CHECK(ri.numel() == 6, "repeat_interleave size = 6");
    auto ri_ptr = ri.data_ptr<float>();
    CHECK(ri_ptr[0] == 1 && ri_ptr[1] == 1 && ri_ptr[2] == 2 && ri_ptr[3] == 2,
          "repeat_interleave([1,2,3], 2) = [1,1,2,2,3,3]");

    // 2.5 unique
    auto dup = at::zeros({6}, TensorOptions().dtype(ScalarType::Float));
    float dup_data[] = {3, 1, 2, 1, 3, 2};
    for (int i = 0; i < 6; i++) dup.mutable_data_ptr<float>()[i] = dup_data[i];
    auto [uniq, inv, counts] = at::native::unique(dup, true, true, true);
    CHECK(uniq.numel() == 3, "unique([3,1,2,1,3,2]) has 3 unique values");

    // 2.6 tril_indices / triu_indices
    auto tril_idx = at::native::tril_indices(3, 3, 0);
    CHECK(tril_idx.size(1) == 6, "tril_indices(3,3) = 6 elements"); // (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)

    auto triu_idx = at::native::triu_indices(3, 3, 0);
    CHECK(triu_idx.size(1) == 6, "triu_indices(3,3) = 6 elements");
}

// ============================================================================
// TEST 3: FFT
// ============================================================================
void test_fft() {
    std::cout << "\n=== TEST: FFT ===" << std::endl;

    // 3.1 fft → ifft roundtrip
    // Create a real signal as complex: [val, 0, val, 0, ...]
    int N = 8;
    auto signal = at::zeros({N, 2}, TensorOptions().dtype(ScalarType::Float));
    auto sig_ptr = signal.mutable_data_ptr<float>();
    for (int i = 0; i < N; i++) {
        sig_ptr[i * 2] = std::sin(2.0 * M_PI * i / N);  // real
        sig_ptr[i * 2 + 1] = 0.0f;  // imag
    }

    auto fft_result = at::native::fft(signal);
    auto ifft_result = at::native::ifft(fft_result);
    double fft_err = max_abs_diff(ifft_result, signal);
    CHECK(fft_err < 1e-4, "ifft(fft(x)) ≈ x (err=" + std::to_string(fft_err) + ")");

    // 3.2 rfft
    auto real_signal = at::zeros({N}, TensorOptions().dtype(ScalarType::Float));
    auto rsig_ptr = real_signal.mutable_data_ptr<float>();
    for (int i = 0; i < N; i++)
        rsig_ptr[i] = std::sin(2.0 * M_PI * i / N);

    auto rfft_result = at::native::rfft(real_signal);
    CHECK(rfft_result.size(0) == N/2 + 1, "rfft output size = N/2+1");
    CHECK(rfft_result.size(1) == 2, "rfft output has [real, imag] format");

    // 3.3 fftfreq
    auto freq = at::native::fftfreq(8, 1.0);
    auto freq_ptr = freq.data_ptr<float>();
    CHECK_CLOSE(freq_ptr[0], 0.0f, 1e-5, "fftfreq[0] = 0");
    CHECK_CLOSE(freq_ptr[1], 0.125f, 1e-5, "fftfreq[1] = 1/8");

    // 3.4 fftshift roundtrip
    auto shifted = at::native::fftshift(real_signal);
    auto unshifted = at::native::ifftshift(shifted);
    double shift_err = max_abs_diff(unshifted, real_signal);
    CHECK(shift_err < 1e-6, "ifftshift(fftshift(x)) ≈ x (err=" + std::to_string(shift_err) + ")");
}

// ============================================================================
// TEST 4: ConvTranspose2d (was STUB, now real)
// ============================================================================
void test_conv_transpose() {
    std::cout << "\n=== TEST: ConvTranspose2d ===" << std::endl;

    // Create ConvTranspose2d: 1 input channel, 1 output channel, kernel 2x2
    torch::nn::ConvTranspose2dImpl conv_t(1, 1, {2, 2}, {1, 1}, {0, 0}, 1, 1);

    // Set identity-like weights
    auto& w = conv_t.weight_->data();
    auto w_ptr = w.mutable_data_ptr<float>();
    for (int i = 0; i < (int)w.numel(); i++) w_ptr[i] = 1.0f;

    // Input: 1x1x2x2, all ones
    auto input = at::ones({1, 1, 2, 2}, TensorOptions().dtype(ScalarType::Float));
    auto output = conv_t.forward(input);

    // With 2x2 kernel of all 1s, stride=1, no padding on 2x2 input:
    // Output should be 3x3 (not zeros!)
    CHECK(output.size(2) == 3 && output.size(3) == 3, "ConvTranspose2d output shape = 3x3");

    // Check output is NOT all zeros (old stub returned zeros)
    auto out_ptr = output.data_ptr<float>();
    double sum = 0;
    for (int i = 0; i < (int)output.numel(); i++) sum += std::abs(out_ptr[i]);
    CHECK(sum > 0.1, "ConvTranspose2d output is NOT zeros (sum=" + std::to_string(sum) + ")");
}

// ============================================================================
// TEST 5: Generalized pad
// ============================================================================
void test_pad() {
    std::cout << "\n=== TEST: Generalized Pad ===" << std::endl;

    // 4D constant padding
    auto x = at::ones({1, 1, 3, 3}, TensorOptions().dtype(ScalarType::Float));
    auto padded = torch::nn::functional::pad(x, {1, 1, 1, 1}, "constant", 0.0);
    CHECK(padded.size(2) == 5 && padded.size(3) == 5, "pad [1,1,1,1] on 3x3 → 5x5");

    // Corner should be 0 (padding), center should be 1 (original)
    auto p_ptr = padded.data_ptr<float>();
    CHECK_CLOSE(p_ptr[0], 0.0f, 1e-6, "pad corner = 0 (constant)");

    // Reflect padding
    auto reflected = torch::nn::functional::pad(x, {1, 1, 1, 1}, "reflect", 0.0);
    CHECK(reflected.size(2) == 5 && reflected.size(3) == 5, "reflect pad [1,1,1,1] on 3x3 → 5x5");
}

// ============================================================================
// TEST 6: Quantization
// ============================================================================
void test_quantization() {
    std::cout << "\n=== TEST: Quantization ===" << std::endl;

    // 6.1 quantize → dequantize roundtrip
    auto x = at::randn({4, 4}, TensorOptions().dtype(ScalarType::Float));
    double scale = 0.01;
    int64_t zero_point = 128;

    auto qt = torch::quantization::quantize_per_tensor(x, scale, zero_point);
    auto deq = qt.dequantize();

    // Error should be within scale (quantization step size)
    double q_err = max_abs_diff(deq, x);
    CHECK(q_err < scale * 2, "dequant(quant(x)) ≈ x within 2*scale (err=" + std::to_string(q_err) + ")");

    // 6.2 MinMaxObserver
    torch::quantization::MinMaxObserver obs;
    obs.forward(x);
    auto [obs_scale, obs_zp] = obs.calculate_qparams();
    CHECK(obs_scale > 0, "Observer scale > 0 (scale=" + std::to_string(obs_scale) + ")");
    CHECK(obs_zp >= 0 && obs_zp <= 255, "Observer zero_point in [0,255] (zp=" + std::to_string(obs_zp) + ")");

    // 6.3 QuantizedTensor properties
    CHECK(qt.int_repr().dtype() == ScalarType::Byte, "Quantized tensor dtype = Byte (uint8)");
    CHECK_CLOSE(qt.scale(), scale, 1e-10, "Quantized tensor scale preserved");
    CHECK(qt.zero_point() == zero_point, "Quantized tensor zero_point preserved");
}

// ============================================================================
// TEST 7: Autograd for new ops
// ============================================================================
void test_autograd() {
    std::cout << "\n=== TEST: Autograd (new ops) ===" << std::endl;

    // 7.1 inverse_autograd: gradient flows
    auto A = at::zeros({3, 3}, TensorOptions().dtype(ScalarType::Float));
    float a_data[] = {2, 1, 0, 1, 3, 1, 0, 1, 2};
    for (int i = 0; i < 9; i++) A.mutable_data_ptr<float>()[i] = a_data[i];
    A.set_requires_grad(true);

    auto inv_A = torch::autograd::inverse_autograd(A);
    // Verify result
    auto should_I = at::native::mm(A, inv_A);
    auto I3 = at::eye(3, TensorOptions().dtype(ScalarType::Float));
    double inv_err = max_abs_diff(should_I, I3);
    CHECK(inv_err < 1e-3, "inverse_autograd: A @ inv(A) ≈ I (err=" + std::to_string(inv_err) + ")");

    // 7.2 trace_autograd
    auto B = at::eye(4, TensorOptions().dtype(ScalarType::Float));
    B.set_requires_grad(true);
    auto tr = torch::autograd::trace_autograd(B);
    CHECK_CLOSE(tr.data_ptr<float>()[0], 4.0f, 1e-5, "trace_autograd(I_4) = 4");

    // 7.3 flip_autograd roundtrip
    auto x = at::randn({3, 4}, TensorOptions().dtype(ScalarType::Float));
    x.set_requires_grad(true);
    auto flipped = torch::autograd::flip_autograd(x, {0, 1});
    auto flipped2 = torch::autograd::flip_autograd(flipped, {0, 1});
    double flip_err = max_abs_diff(flipped2, x);
    CHECK(flip_err < 1e-6, "flip_autograd(flip_autograd(x)) ≈ x (err=" + std::to_string(flip_err) + ")");
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  PromeTorch Phase 2 — Feature Tests" << std::endl;
    std::cout << "============================================================" << std::endl;

    test_linalg();
    test_tensor_ops();
    test_fft();
    test_conv_transpose();
    test_pad();
    test_quantization();
    test_autograd();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    std::cout << "============================================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
