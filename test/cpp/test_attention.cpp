// ============================================================================
// test_attention.cpp — CPU correctness tests for scaled_dot_product_attention
// and its autograd wrapper (SdpaBackward).
//
// Coverage:
//   - 2D input [N, D]           vs hand-coded softmax(QK^T/sqrt(d))V
//   - 4D input [B, H, N, D]     vs hand-coded reference
//   - Causal mask lower-triangular pattern
//   - attn_mask bool: [N_q, N_k] and [B, H, N_q, N_k]
//   - Gradient via SdpaBackward vs finite-difference numerical gradient
// ============================================================================

#include <gtest/gtest.h>
#include "torch/csrc/autograd/autograd.h"
#include "aten/src/ATen/native/Attention.h"
#include <cmath>
#include <vector>

using namespace at;
using namespace torch::autograd;

namespace {

// ---- local helpers ---------------------------------------------------------

bool approx(float a, float b, float rtol = 1e-4f, float atol = 1e-5f) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

bool tensor_close(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-5f) {
    if (a.sizes() != b.sizes()) {
        std::cerr << "tensor_close: shape mismatch\n";
        return false;
    }
    Tensor ac = a.is_contiguous() ? a : a.contiguous();
    Tensor bc = b.is_contiguous() ? b : b.contiguous();
    const float* ad = ac.data_ptr<float>();
    const float* bd = bc.data_ptr<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        if (!approx(ad[i], bd[i], rtol, atol)) {
            std::cerr << "tensor_close: diff at i=" << i
                      << " a=" << ad[i] << " b=" << bd[i] << "\n";
            return false;
        }
    }
    return true;
}

// Hand-coded reference: O = softmax(Q @ K^T / sqrt(D)) @ V, shapes [N,D]/[N,D]/[N,D].
Tensor sdpa_ref_2d(const Tensor& Q, const Tensor& K, const Tensor& V,
                   const Tensor& mask = Tensor(), bool is_causal = false)
{
    int64_t N_q = Q.size(0);
    int64_t N_k = K.size(0);
    int64_t D = Q.size(1);
    float s = 1.0f / std::sqrt((float)D);
    const float* Qd = Q.data_ptr<float>();
    const float* Kd = K.data_ptr<float>();
    const float* Vd = V.data_ptr<float>();
    const bool has_mask = mask.defined();
    const bool mask_is_bool = has_mask && mask.dtype() == c10::ScalarType::Bool;

    Tensor out = at::zeros({N_q, D});
    float* Od = out.mutable_data_ptr<float>();
    for (int64_t i = 0; i < N_q; ++i) {
        std::vector<float> scores(N_k);
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < N_k; ++j) {
            float x = 0.0f;
            for (int64_t d = 0; d < D; ++d) x += Qd[i*D+d] * Kd[j*D+d];
            x *= s;
            if (is_causal && j > i) x = -std::numeric_limits<float>::infinity();
            if (has_mask) {
                if (mask_is_bool) {
                    if (!mask.data_ptr<bool>()[i*N_k+j])
                        x = -std::numeric_limits<float>::infinity();
                } else {
                    x += mask.data_ptr<float>()[i*N_k+j];
                }
            }
            scores[j] = x;
            if (x > mx) mx = x;
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < N_k; ++j) {
            float e = (mx == -std::numeric_limits<float>::infinity()) ? 0.0f
                       : std::exp(scores[j] - mx);
            scores[j] = e;
            sum += e;
        }
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (int64_t j = 0; j < N_k; ++j) scores[j] *= inv;
        for (int64_t d = 0; d < D; ++d) {
            float o = 0.0f;
            for (int64_t j = 0; j < N_k; ++j) o += scores[j] * Vd[j*D+d];
            Od[i*D+d] = o;
        }
    }
    return out;
}

// Scalar sum of all elements in tensor.
float tensor_sum(const Tensor& t) {
    Tensor c = t.is_contiguous() ? t : t.contiguous();
    const float* d = c.data_ptr<float>();
    double s = 0.0;
    for (int64_t i = 0; i < t.numel(); ++i) s += d[i];
    return (float)s;
}

} // namespace

// ============================================================================
// 1. Simple 2D input forward
// ============================================================================
TEST(AttentionTest, Forward2D_MatchesReference) {
    Tensor Q = at::randn({4, 8});
    Tensor K = at::randn({4, 8});
    Tensor V = at::randn({4, 8});

    Tensor got = at::scaled_dot_product_attention(Q, K, V);
    Tensor expected = sdpa_ref_2d(Q, K, V);

    EXPECT_TRUE(tensor_close(got, expected, /*rtol*/1e-4f, /*atol*/1e-5f));
}

// ============================================================================
// 2. 4D input [B, N, H, D] forward (PromeTorch + FlashAttention layout)
// ============================================================================
TEST(AttentionTest, Forward4D_MatchesReferencePerHead) {
    const int64_t B = 2, N = 8, H = 4, D = 16;
    Tensor Q = at::randn({B, N, H, D});
    Tensor K = at::randn({B, N, H, D});
    Tensor V = at::randn({B, N, H, D});

    Tensor got = at::scaled_dot_product_attention(Q, K, V);
    ASSERT_EQ(got.dim(), 4);
    EXPECT_EQ(got.size(0), B);
    EXPECT_EQ(got.size(1), N);
    EXPECT_EQ(got.size(2), H);
    EXPECT_EQ(got.size(3), D);

    // Compare head-by-head against 2D reference.
    Tensor got_c = got.contiguous();
    Tensor Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    const float* Qd = Qc.data_ptr<float>();
    const float* Kd = Kc.data_ptr<float>();
    const float* Vd = Vc.data_ptr<float>();
    const float* gotd = got_c.data_ptr<float>();
    auto idx4 = [&](int64_t b, int64_t n, int64_t h, int64_t d) {
        return ((b * N + n) * H + h) * D + d;
    };
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            Tensor q2 = at::zeros({N, D});
            Tensor k2 = at::zeros({N, D});
            Tensor v2 = at::zeros({N, D});
            float* q2d = q2.mutable_data_ptr<float>();
            float* k2d = k2.mutable_data_ptr<float>();
            float* v2d = v2.mutable_data_ptr<float>();
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t d = 0; d < D; ++d) {
                    q2d[n * D + d] = Qd[idx4(b, n, h, d)];
                    k2d[n * D + d] = Kd[idx4(b, n, h, d)];
                    v2d[n * D + d] = Vd[idx4(b, n, h, d)];
                }
            }
            Tensor ref = sdpa_ref_2d(q2, k2, v2);
            const float* refd = ref.data_ptr<float>();
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t d = 0; d < D; ++d) {
                    EXPECT_TRUE(approx(gotd[idx4(b, n, h, d)], refd[n * D + d], 1e-4f, 1e-5f))
                        << "mismatch b=" << b << " h=" << h << " n=" << n << " d=" << d;
                }
            }
        }
    }
}

// ============================================================================
// 3. Causal mask lower-triangular attention pattern
// ============================================================================
// We verify by setting V to the identity-of-rows (V[j] = onehot(j)) so that
// output row i equals the attention distribution row i. Then row 0 must have
// only position 0 nonzero, and row N-1 must have all positions nonzero.
TEST(AttentionTest, CausalMask_LowerTriangular) {
    const int64_t N = 5, D = 5;
    // Use constant Q and K so all raw scores are equal; softmax then reveals
    // the causal mask: row i gets uniform 1/(i+1) over positions 0..i.
    Tensor Q = at::ones({N, D});
    Tensor K = at::ones({N, D});
    Tensor V = at::zeros({N, D});
    float* Vd = V.mutable_data_ptr<float>();
    for (int64_t j = 0; j < N; ++j) Vd[j * D + j] = 1.0f;   // V[j] = e_j

    Tensor out = at::scaled_dot_product_attention(
        Q, K, V, /*mask*/Tensor(), /*dropout*/0.0f, /*is_causal*/true);
    ASSERT_EQ(out.dim(), 2);
    const float* od = out.contiguous().data_ptr<float>();

    for (int64_t i = 0; i < N; ++i) {
        int64_t nonzero = 0;
        for (int64_t j = 0; j < N; ++j) {
            if (std::abs(od[i * D + j]) > 1e-6f) ++nonzero;
            if (j > i) {
                EXPECT_NEAR(od[i * D + j], 0.0f, 1e-6f)
                    << "causal mask failure at (" << i << "," << j << ")";
            }
        }
        EXPECT_EQ(nonzero, i + 1) << "row " << i << " should attend to " << (i+1) << " positions";
    }
}

// ============================================================================
// 4a. attn_mask bool: [N_q, N_k]
// ============================================================================
TEST(AttentionTest, BoolMask_2D) {
    const int64_t N = 4, D = 8;
    Tensor Q = at::randn({N, D});
    Tensor K = at::randn({N, D});
    Tensor V = at::randn({N, D});

    // Bool mask: mask out position (0,1) and (2,3).
    Tensor mask = at::empty({N, N}, TensorOptions().dtype(c10::ScalarType::Bool));
    bool* md = mask.mutable_data_ptr<bool>();
    for (int64_t i = 0; i < N*N; ++i) md[i] = true;   // all allowed
    md[0 * N + 1] = false;
    md[2 * N + 3] = false;

    Tensor got = at::scaled_dot_product_attention(Q, K, V, mask);
    Tensor expected = sdpa_ref_2d(Q, K, V, mask);

    EXPECT_TRUE(tensor_close(got, expected, 1e-4f, 1e-5f));
}

// ============================================================================
// 4b. attn_mask bool: [B, H, N_q, N_k] with 4D Q/K/V = [B, N, H, D]
// ============================================================================
TEST(AttentionTest, BoolMask_4D) {
    const int64_t B = 1, H = 2, N = 4, D = 8;
    Tensor Q = at::randn({B, N, H, D});
    Tensor K = at::randn({B, N, H, D});
    Tensor V = at::randn({B, N, H, D});

    // Mask is still [B, H, N_q, N_k] because that's the attention-scores layout.
    Tensor mask = at::empty({B, H, N, N}, TensorOptions().dtype(c10::ScalarType::Bool));
    bool* md = mask.mutable_data_ptr<bool>();
    for (int64_t i = 0; i < B*H*N*N; ++i) md[i] = true;
    md[((0*H + 0)*N + 1)*N + 2] = false;   // head 0: query 1 cannot see key 2
    md[((0*H + 1)*N + 3)*N + 0] = false;   // head 1: query 3 cannot see key 0

    Tensor got = at::scaled_dot_product_attention(Q, K, V, mask);
    ASSERT_EQ(got.dim(), 4);
    EXPECT_EQ(got.size(0), B);
    EXPECT_EQ(got.size(1), N);
    EXPECT_EQ(got.size(2), H);

    // Compare each head against 2D reference + 2D mask slice.
    Tensor Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    Tensor got_c = got.contiguous();
    auto idx4 = [&](int64_t b, int64_t n, int64_t h, int64_t d) {
        return ((b * N + n) * H + h) * D + d;
    };
    for (int64_t h = 0; h < H; ++h) {
        Tensor q2 = at::zeros({N, D}), k2 = at::zeros({N, D}), v2 = at::zeros({N, D});
        Tensor m2 = at::empty({N, N}, TensorOptions().dtype(c10::ScalarType::Bool));
        float* q2d = q2.mutable_data_ptr<float>();
        float* k2d = k2.mutable_data_ptr<float>();
        float* v2d = v2.mutable_data_ptr<float>();
        bool*  m2d = m2.mutable_data_ptr<bool>();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t d = 0; d < D; ++d) {
                q2d[n*D+d] = Qc.data_ptr<float>()[idx4(0, n, h, d)];
                k2d[n*D+d] = Kc.data_ptr<float>()[idx4(0, n, h, d)];
                v2d[n*D+d] = Vc.data_ptr<float>()[idx4(0, n, h, d)];
            }
        }
        for (int64_t i = 0; i < N; ++i)
            for (int64_t j = 0; j < N; ++j)
                m2d[i*N+j] = md[((0*H+h)*N+i)*N+j];
        Tensor ref = sdpa_ref_2d(q2, k2, v2, m2);
        const float* refd = ref.data_ptr<float>();
        const float* gotd = got_c.data_ptr<float>();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t d = 0; d < D; ++d) {
                EXPECT_TRUE(approx(gotd[idx4(0, n, h, d)], refd[n*D+d], 1e-4f, 1e-5f))
                    << "head=" << h << " n=" << n << " d=" << d;
            }
        }
    }
}

// ============================================================================
// 5. Gradient check: analytic (SdpaBackward) vs numerical (finite diff).
// Uses 2D tensors [N, D] because it already covers the full kernel in a
// sequential case and has small enough parameter count for FD to be stable.
// ============================================================================
TEST(AttentionTest, GradientCheck_2D_FiniteDiff) {
    const int64_t N = 3, D = 4;
    Tensor Q = at::randn({N, D});
    Tensor K = at::randn({N, D});
    Tensor V = at::randn({N, D});
    Q.set_requires_grad(true);
    K.set_requires_grad(true);
    V.set_requires_grad(true);

    // Forward with autograd.
    Tensor out = scaled_dot_product_attention_autograd(Q, K, V);
    // Use a deterministic, non-symmetric "loss" weight so the summed scalar
    // is a non-trivial function of every output element.
    Tensor W = at::randn({N, D});
    // Must use autograd-aware multiplication + sum so the graph flows back.
    Tensor loss = sum_autograd(mul_autograd(out, W));
    tensor_backward(loss);

    auto get_grad = [](const Tensor& t) -> Tensor {
        auto* m = t.autograd_meta();
        return (m && m->grad_) ? Tensor(m->grad_) : Tensor();
    };
    Tensor gQ = get_grad(Q);
    Tensor gK = get_grad(K);
    Tensor gV = get_grad(V);
    ASSERT_TRUE(gQ.defined());
    ASSERT_TRUE(gK.defined());
    ASSERT_TRUE(gV.defined());

    // Numerical gradient via central difference on a loss of the same form.
    const float h = 1e-3f;
    auto numerical_grad = [&](Tensor& X) -> Tensor {
        Tensor gX = at::zeros(X.sizes());
        float* gXd = gX.mutable_data_ptr<float>();
        float* Xd  = X.mutable_data_ptr<float>();
        int64_t n = X.numel();
        for (int64_t i = 0; i < n; ++i) {
            float orig = Xd[i];
            Xd[i] = orig + h;
            Tensor out_p = at::scaled_dot_product_attention(Q, K, V);
            float loss_p = tensor_sum(out_p * W);
            Xd[i] = orig - h;
            Tensor out_m = at::scaled_dot_product_attention(Q, K, V);
            float loss_m = tensor_sum(out_m * W);
            Xd[i] = orig;
            gXd[i] = (loss_p - loss_m) / (2.0f * h);
        }
        return gX;
    };

    Tensor ngQ = numerical_grad(Q);
    Tensor ngK = numerical_grad(K);
    Tensor ngV = numerical_grad(V);

    EXPECT_TRUE(tensor_close(gQ, ngQ, /*rtol*/1e-2f, /*atol*/1e-3f))
        << "dQ mismatch (analytic vs finite-diff)";
    EXPECT_TRUE(tensor_close(gK, ngK, /*rtol*/1e-2f, /*atol*/1e-3f))
        << "dK mismatch (analytic vs finite-diff)";
    EXPECT_TRUE(tensor_close(gV, ngV, /*rtol*/1e-2f, /*atol*/1e-3f))
        << "dV mismatch (analytic vs finite-diff)";
}

// ============================================================================
// 6. is_causal and attn_mask together must raise (PyTorch compat).
// ============================================================================
TEST(AttentionTest, IsCausalAndAttnMask_AreMutuallyExclusive) {
    Tensor Q = at::randn({2, 4});
    Tensor K = at::randn({2, 4});
    Tensor V = at::randn({2, 4});
    Tensor m = at::zeros({2, 2});

    EXPECT_THROW({
        (void)at::scaled_dot_product_attention(Q, K, V, m, 0.0f, /*is_causal*/true);
    }, std::exception);
}

// ============================================================================
// 7. dropout_p=0 in training mode must be deterministic and match no-dropout.
// ============================================================================
TEST(AttentionTest, DropoutZero_Deterministic) {
    Tensor Q = at::randn({3, 8});
    Tensor K = at::randn({3, 8});
    Tensor V = at::randn({3, 8});

    Tensor a = at::scaled_dot_product_attention(Q, K, V, Tensor(), /*dropout*/0.0f);
    Tensor b = at::scaled_dot_product_attention(Q, K, V, Tensor(), /*dropout*/0.0f);
    EXPECT_TRUE(tensor_close(a, b, 0.0f, 0.0f));
}
