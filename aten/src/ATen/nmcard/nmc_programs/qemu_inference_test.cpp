// qemu_inference_test.cpp - Test TinyLlama inference operations in QEMU
// Tests: MatMul, RMSNorm, SiLU, Softmax, RoPE, Elementwise ops
// These are all operations needed for TinyLlama forward pass

#include "mymath.h"

// Helper to check if values are close
inline int is_close(fixed32 a, fixed32 b, int tolerance) {
    int diff = a - b;
    if (diff < 0) diff = -diff;
    return diff <= tolerance;
}

// ============================================================
// Mini MatMul test: C[2,2] = A[2,3] @ B[3,2]
// ============================================================
int test_matmul() {
    // A = [[1, 2, 3], [4, 5, 6]]
    fixed32 A[6] = {
        FIXED_ONE, INT_TO_FIXED(2), INT_TO_FIXED(3),
        INT_TO_FIXED(4), INT_TO_FIXED(5), INT_TO_FIXED(6)
    };
    // B = [[1, 2], [3, 4], [5, 6]]
    fixed32 B[6] = {
        FIXED_ONE, INT_TO_FIXED(2),
        INT_TO_FIXED(3), INT_TO_FIXED(4),
        INT_TO_FIXED(5), INT_TO_FIXED(6)
    };
    // Expected C = [[22, 28], [49, 64]]
    // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64

    fixed32 C[4];
    int M = 2, K = 3, N = 2;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fixed32 sum = 0;
            for (int k = 0; k < K; k++) {
                int a_idx = mul_u32(i, K) + k;
                int b_idx = mul_u32(k, N) + j;
                sum = add_fixed(sum, mul_fixed(A[a_idx], B[b_idx]));
            }
            C[mul_u32(i, N) + j] = sum;
        }
    }

    if (!is_close(C[0], INT_TO_FIXED(22), 100)) return 1;
    if (!is_close(C[1], INT_TO_FIXED(28), 100)) return 2;
    if (!is_close(C[2], INT_TO_FIXED(49), 100)) return 3;
    if (!is_close(C[3], INT_TO_FIXED(64), 100)) return 4;

    return 0;
}

// ============================================================
// RMSNorm test: y = x * gamma / sqrt(mean(x^2) + eps)
// ============================================================
int test_rmsnorm() {
    // x = [2, 4, 6, 8], gamma = [1, 1, 1, 1]
    fixed32 x[4] = {INT_TO_FIXED(2), INT_TO_FIXED(4), INT_TO_FIXED(6), INT_TO_FIXED(8)};
    fixed32 gamma[4] = {FIXED_ONE, FIXED_ONE, FIXED_ONE, FIXED_ONE};
    fixed32 y[4];
    fixed32 eps = 1;

    // mean(x^2) = (4 + 16 + 36 + 64) / 4 = 120 / 4 = 30
    // rms = sqrt(30 + eps) ~= 5.477
    // y[0] = 2 / 5.477 ~= 0.365
    // y[1] = 4 / 5.477 ~= 0.730
    // y[2] = 6 / 5.477 ~= 1.095
    // y[3] = 8 / 5.477 ~= 1.461

    fixed32 sum_sq = 0;
    for (int i = 0; i < 4; i++) {
        sum_sq = add_fixed(sum_sq, mul_fixed(x[i], x[i]));
    }
    fixed32 mean_sq = div_fixed(sum_sq, INT_TO_FIXED(4));
    fixed32 rms = sqrt_fixed(add_fixed(mean_sq, eps));
    fixed32 inv_rms = div_fixed(FIXED_ONE, rms);

    for (int i = 0; i < 4; i++) {
        y[i] = mul_fixed(mul_fixed(x[i], inv_rms), gamma[i]);
    }

    // 0.365 in Q16.16 ~= 23921
    // 0.730 in Q16.16 ~= 47841
    // 1.095 in Q16.16 ~= 71762
    // 1.461 in Q16.16 ~= 95683
    if (!is_close(y[0], 23921, 2000)) return 10;
    if (!is_close(y[1], 47841, 2000)) return 11;
    if (!is_close(y[2], 71762, 2000)) return 12;
    if (!is_close(y[3], 95683, 2000)) return 13;

    return 0;
}

// ============================================================
// SiLU test: y = x * sigmoid(x)
// ============================================================
int test_silu() {
    // silu(0) = 0
    // silu(1) ~= 0.731
    // silu(2) ~= 1.762
    // silu(-1) ~= -0.269

    fixed32 s0 = silu_fixed(0);
    if (s0 != 0) return 20;

    fixed32 s1 = silu_fixed(FIXED_ONE);
    // 0.731 * 65536 ~= 47908
    if (!is_close(s1, 47908, 3000)) return 21;

    fixed32 s2 = silu_fixed(INT_TO_FIXED(2));
    // 1.762 * 65536 ~= 115476
    if (!is_close(s2, 115476, 5000)) return 22;

    fixed32 sm1 = silu_fixed(-FIXED_ONE);
    // -0.269 * 65536 ~= -17629
    if (!is_close(sm1, -17629, 3000)) return 23;

    return 0;
}

// ============================================================
// Softmax test: softmax([1, 2, 3])
// ============================================================
int test_softmax() {
    // softmax([1, 2, 3]) = [0.090, 0.245, 0.665]
    fixed32 x[3] = {FIXED_ONE, INT_TO_FIXED(2), INT_TO_FIXED(3)};
    fixed32 y[3];

    // Find max
    fixed32 max_val = x[0];
    for (int i = 1; i < 3; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // exp(x - max) and sum
    fixed32 exp_sum = 0;
    for (int i = 0; i < 3; i++) {
        fixed32 exp_val = exp_fixed_lut(sub_fixed(x[i], max_val));
        y[i] = exp_val;
        exp_sum = add_fixed(exp_sum, exp_val);
    }

    // Normalize
    for (int i = 0; i < 3; i++) {
        y[i] = div_fixed(y[i], exp_sum);
    }

    // 0.090 * 65536 ~= 5898
    // 0.245 * 65536 ~= 16056
    // 0.665 * 65536 ~= 43581
    if (!is_close(y[0], 5898, 2000)) return 30;
    if (!is_close(y[1], 16056, 3000)) return 31;
    if (!is_close(y[2], 43581, 3000)) return 32;

    // Sum should be ~1.0
    fixed32 sum = add_fixed(add_fixed(y[0], y[1]), y[2]);
    if (!is_close(sum, FIXED_ONE, 1000)) return 33;

    return 0;
}

// ============================================================
// RoPE test: rotate pairs by angle
// x' = x*cos - y*sin, y' = x*sin + y*cos
// ============================================================
int test_rope() {
    // For angle = 0: cos=1, sin=0 -> output = input
    fixed32 x0 = FIXED_ONE;
    fixed32 x1 = INT_TO_FIXED(2);

    fixed32 cos_0 = cos_fixed(0);  // should be ~1
    fixed32 sin_0 = sin_fixed(0);  // should be ~0

    if (!is_close(cos_0, FIXED_ONE, 1000)) return 40;
    if (!is_close(sin_0, 0, 1000)) return 41;

    fixed32 y0 = sub_fixed(mul_fixed(x0, cos_0), mul_fixed(x1, sin_0));
    fixed32 y1 = add_fixed(mul_fixed(x0, sin_0), mul_fixed(x1, cos_0));

    if (!is_close(y0, FIXED_ONE, 1000)) return 42;
    if (!is_close(y1, INT_TO_FIXED(2), 1000)) return 43;

    // For angle = pi/2: cos=0, sin=1 -> x'=-y, y'=x
    const fixed32 PI_HALF = 102944;  // pi/2 in Q16.16
    fixed32 cos_90 = cos_fixed(PI_HALF);
    fixed32 sin_90 = sin_fixed(PI_HALF);

    // cos(pi/2) ~= 0, sin(pi/2) ~= 1
    if (!is_close(cos_90, 0, 3000)) return 44;
    if (!is_close(sin_90, FIXED_ONE, 3000)) return 45;

    return 0;
}

// ============================================================
// Elementwise ops test
// ============================================================
int test_elementwise() {
    fixed32 a = INT_TO_FIXED(3);
    fixed32 b = INT_TO_FIXED(2);

    // Add
    fixed32 add_result = add_fixed(a, b);
    if (add_result != INT_TO_FIXED(5)) return 50;

    // Sub
    fixed32 sub_result = sub_fixed(a, b);
    if (sub_result != FIXED_ONE) return 51;

    // Mul
    fixed32 mul_result = mul_fixed(a, b);
    if (mul_result != INT_TO_FIXED(6)) return 52;

    // Gate mul: a * silu(b) = 3 * silu(2) ~= 3 * 1.762 ~= 5.286
    fixed32 gate_result = mul_fixed(a, silu_fixed(b));
    // 5.286 * 65536 ~= 346428
    if (!is_close(gate_result, 346428, 10000)) return 53;

    return 0;
}

// ============================================================
// Mini attention test:
// scores = Q @ K^T / sqrt(d_k), attention = softmax(scores) @ V
// ============================================================
int test_attention() {
    // Simplified: Q[1,2], K[1,2], V[1,2] -> output[1,2]
    // Q = [1, 0], K = [1, 0], V = [2, 3]
    // score = Q @ K^T = 1*1 + 0*0 = 1
    // sqrt(d_k) = sqrt(2) ~= 1.414
    // scaled_score = 1 / 1.414 ~= 0.707
    // softmax([0.707]) = [1.0] (single element)
    // output = 1.0 * V = [2, 3]

    fixed32 Q[2] = {FIXED_ONE, 0};
    fixed32 K[2] = {FIXED_ONE, 0};
    fixed32 V[2] = {INT_TO_FIXED(2), INT_TO_FIXED(3)};

    // Q @ K^T (dot product for single head)
    fixed32 score = add_fixed(mul_fixed(Q[0], K[0]), mul_fixed(Q[1], K[1]));
    if (!is_close(score, FIXED_ONE, 100)) return 60;

    // Scale by 1/sqrt(d_k) = 1/sqrt(2)
    fixed32 d_k = INT_TO_FIXED(2);
    fixed32 sqrt_d_k = sqrt_fixed(d_k);
    fixed32 scaled_score = div_fixed(score, sqrt_d_k);
    // 0.707 * 65536 ~= 46341
    if (!is_close(scaled_score, 46341, 2000)) return 61;

    // Softmax of single value is always 1.0
    fixed32 attn_weight = FIXED_ONE;

    // Weighted sum: output = attn_weight * V
    fixed32 out0 = mul_fixed(attn_weight, V[0]);
    fixed32 out1 = mul_fixed(attn_weight, V[1]);

    if (!is_close(out0, INT_TO_FIXED(2), 100)) return 62;
    if (!is_close(out1, INT_TO_FIXED(3), 100)) return 63;

    return 0;
}

// ============================================================
// Integration test: Mini FFN block
// FFN(x) = down(up(x) * silu(gate(x)))
// Simplified: out = x * silu(x) (when gate = up = identity)
// ============================================================
int test_ffn_block() {
    fixed32 x[4] = {FIXED_ONE, INT_TO_FIXED(2), -FIXED_ONE, 0};
    fixed32 out[4];

    // gate_mul: x * silu(x)
    for (int i = 0; i < 4; i++) {
        out[i] = mul_fixed(x[i], silu_fixed(x[i]));
    }

    // out[0] = 1 * silu(1) = 1 * 0.731 ~= 0.731
    // out[1] = 2 * silu(2) = 2 * 1.762 ~= 3.524
    // out[2] = -1 * silu(-1) = -1 * (-0.269) ~= 0.269
    // out[3] = 0 * silu(0) = 0

    if (!is_close(out[0], 47908, 5000)) return 70;  // 0.731 * 65536
    if (!is_close(out[1], 230952, 10000)) return 71; // 3.524 * 65536
    if (!is_close(out[2], 17629, 5000)) return 72;  // 0.269 * 65536
    if (out[3] != 0) return 73;

    return 0;
}

// ============================================================
// Integration: Mini Llama layer forward pass
// x -> rmsnorm -> attention -> residual -> rmsnorm -> ffn -> residual
// ============================================================
int test_llama_layer() {
    // Very simplified single-element test
    fixed32 x = INT_TO_FIXED(2);
    fixed32 gamma = FIXED_ONE;
    fixed32 eps = 1;

    // RMSNorm (single element: rms = |x|, so y = sign(x) * gamma)
    // Actually for single element: y = x * gamma / sqrt(x^2 + eps) = x / |x| * gamma ~= gamma (if x > 0)
    fixed32 x_sq = mul_fixed(x, x);  // 4
    fixed32 rms = sqrt_fixed(add_fixed(x_sq, eps));  // ~2
    fixed32 inv_rms = div_fixed(FIXED_ONE, rms);  // ~0.5
    fixed32 norm_x = mul_fixed(mul_fixed(x, inv_rms), gamma);  // ~1.0

    if (!is_close(norm_x, FIXED_ONE, 1000)) return 80;

    // "Attention" (identity for simplicity)
    fixed32 attn_out = norm_x;

    // Residual add
    fixed32 h1 = add_fixed(x, attn_out);  // 2 + 1 = 3
    if (!is_close(h1, INT_TO_FIXED(3), 100)) return 81;

    // Second RMSNorm
    fixed32 h1_sq = mul_fixed(h1, h1);  // 9
    fixed32 rms2 = sqrt_fixed(add_fixed(h1_sq, eps));  // ~3
    fixed32 inv_rms2 = div_fixed(FIXED_ONE, rms2);  // ~0.333
    fixed32 norm_h1 = mul_fixed(mul_fixed(h1, inv_rms2), gamma);  // ~1.0

    if (!is_close(norm_h1, FIXED_ONE, 2000)) return 82;

    // FFN: silu activation
    fixed32 ffn_out = silu_fixed(norm_h1);  // silu(1) ~= 0.731

    // Final residual
    fixed32 output = add_fixed(h1, ffn_out);  // 3 + 0.731 ~= 3.731
    // 3.731 * 65536 ~= 244527
    if (!is_close(output, 244527, 5000)) return 83;

    return 0;
}

// ============================================================
// Main
// ============================================================
int main() {
    int result;

    // Test 1-4: MatMul
    result = test_matmul();
    if (result != 0) return result;

    // Test 10-13: RMSNorm
    result = test_rmsnorm();
    if (result != 0) return result;

    // Test 20-23: SiLU
    result = test_silu();
    if (result != 0) return result;

    // Test 30-33: Softmax
    result = test_softmax();
    if (result != 0) return result;

    // Test 40-45: RoPE
    result = test_rope();
    if (result != 0) return result;

    // Test 50-53: Elementwise
    result = test_elementwise();
    if (result != 0) return result;

    // Test 60-63: Attention
    result = test_attention();
    if (result != 0) return result;

    // Test 70-73: FFN block
    result = test_ffn_block();
    if (result != 0) return result;

    // Test 80-83: Full Llama layer
    result = test_llama_layer();
    if (result != 0) return result;

    // All tests passed!
    return 0;
}
