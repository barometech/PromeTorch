#pragma once
// ============================================================================
// AttentionBackward — backward pass for scaled_dot_product_attention.
//
// Forward (canonical [B, N, H, D] layout):
//   S[b,h,i,j] = (Q[b,i,h,:] . K[b,j,h,:]) * scale   + mask (+ causal -inf)
//   P[b,h,i,j] = softmax_j( S )
//   P_drop     = P * drop_mask                       (training, p>0)
//   O[b,i,h,:] = sum_j P_drop[b,h,i,j] * V[b,j,h,:]
//
// Backward — standard SDPA gradients:
//   dO:   [B, N_q, H, D]
//   dP_drop[b,h,i,j]   = sum_d dO[b,i,h,d] * V[b,j,h,d]
//   dV[b,j,h,d]        = sum_i P_drop[b,h,i,j] * dO[b,i,h,d]
//   dP[b,h,i,j]        = dP_drop[b,h,i,j] * drop_mask[b,h,i,j]
//   softmax-jvp:
//     dS[b,h,i,j]      = ( dP[b,h,i,j] - sum_k (dP[b,h,i,k] * P[b,h,i,k]) ) * P[b,h,i,j]
//   dQ[b,i,h,d]        = scale * sum_j dS[b,h,i,j] * K[b,j,h,d]
//   dK[b,j,h,d]        = scale * sum_i dS[b,h,i,j] * Q[b,i,h,d]
//
// Where masked positions had P=0, gradient automatically stays 0 (no special
// handling needed for bool / causal masks — they don't participate in grad).
// ============================================================================

#include "torch/csrc/autograd/node.h"
#include "aten/src/ATen/ATen.h"
#include <cmath>

namespace torch {
namespace autograd {

using at::Tensor;

struct SdpaBackward : public Node {
    // Saved tensors (canonical [B, N, H, D] layout for Q/K/V; [B,H,N_q,N_k] for P/M).
    Tensor Q_;                // [B, N_q, H, D]
    Tensor K_;                // [B, N_k, H, D]
    Tensor V_;                // [B, N_k, H, D]
    Tensor probs_;            // [B, H, N_q, N_k]  softmax(S) (pre-dropout)
    Tensor dropout_mask_;     // [B, H, N_q, N_k]  (only used if dropout_p>0; all-1 otherwise)
    float  scale_;
    int    orig_rank_;        // Original rank of Q (2, 3, 4, ...)

    SdpaBackward(const Tensor& Q, const Tensor& K, const Tensor& V,
                 const Tensor& probs, const Tensor& drop_mask,
                 float scale, int orig_rank)
        : Q_(Q), K_(K), V_(V), probs_(probs), dropout_mask_(drop_mask),
          scale_(scale), orig_rank_(orig_rank) {}

    void release_saved_tensors() override {
        Q_ = Tensor(); K_ = Tensor(); V_ = Tensor();
        probs_ = Tensor(); dropout_mask_ = Tensor();
    }

    variable_list apply(variable_list&& grads) override {
        auto& grad_out = grads[0];  // grad w.r.t. output in the caller's layout
        if (!grad_out.defined()) return {Tensor(), Tensor(), Tensor()};

        // ----- Bring grad_out back to canonical [B, N_q, H, D] ------------
        Tensor dO;
        {
            const int64_t nd = orig_rank_;
            if (nd == 2) {
                dO = grad_out.unsqueeze(0).unsqueeze(2);        // [N, D] → [1, N, 1, D]
            } else if (nd == 3) {
                dO = grad_out.unsqueeze(2);                     // [B, N, D] → [B, N, 1, D]
            } else if (nd == 4) {
                dO = grad_out;                                  // already [B, N, H, D]
            } else {
                int64_t D_loc = grad_out.size(nd - 1);
                int64_t H_loc = grad_out.size(nd - 2);
                int64_t N_loc = grad_out.size(nd - 3);
                int64_t Bp = 1;
                for (int64_t i = 0; i < nd - 3; ++i) Bp *= grad_out.size(i);
                dO = grad_out.reshape({Bp, N_loc, H_loc, D_loc});
            }
            dO = dO.contiguous();
        }

        const int64_t B   = Q_.size(0);
        const int64_t N_q = Q_.size(1);
        const int64_t H   = Q_.size(2);
        const int64_t D   = Q_.size(3);
        const int64_t N_k = K_.size(1);

        Tensor dQ = at::zeros({B, N_q, H, D}, at::TensorOptions().dtype(Q_.dtype()));
        Tensor dK = at::zeros({B, N_k, H, D}, at::TensorOptions().dtype(K_.dtype()));
        Tensor dV = at::zeros({B, N_k, H, D}, at::TensorOptions().dtype(V_.dtype()));

        const float* Qd  = Q_.data_ptr<float>();
        const float* Kd  = K_.data_ptr<float>();
        const float* Vd  = V_.data_ptr<float>();
        const float* Pd  = probs_.data_ptr<float>();       // [B, H, N_q, N_k]
        const bool   has_drop = dropout_mask_.defined();
        const float* Md  = has_drop ? dropout_mask_.data_ptr<float>() : nullptr;
        const float* dOd = dO.data_ptr<float>();

        float* dQd = dQ.mutable_data_ptr<float>();
        float* dKd = dK.mutable_data_ptr<float>();
        float* dVd = dV.mutable_data_ptr<float>();

        auto qkv_idx = [&](int64_t b, int64_t n, int64_t h, int64_t d, int64_t N) {
            return ((b * N + n) * H + h) * D + d;
        };
        auto p_idx = [&](int64_t b, int64_t h, int64_t i, int64_t j) {
            return ((b * H + h) * N_q + i) * N_k + j;
        };

        const float s = scale_;

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t i = 0; i < N_q; ++i) {
                    // dP_drop[j] = sum_d dO[b,i,h,d] * V[b,j,h,d]
                    // dV[b,j,h,d] += P_drop[b,h,i,j] * dO[b,i,h,d]
                    // (P_drop = P * drop_mask)
                    std::vector<float> dP(N_k, 0.0f);
                    for (int64_t j = 0; j < N_k; ++j) {
                        float m = has_drop ? Md[p_idx(b, h, i, j)] : 1.0f;
                        float P_drop = Pd[p_idx(b, h, i, j)] * m;
                        float dP_drop = 0.0f;
                        for (int64_t d = 0; d < D; ++d) {
                            float dOd_v = dOd[qkv_idx(b, i, h, d, N_q)];
                            dP_drop += dOd_v * Vd[qkv_idx(b, j, h, d, N_k)];
                            dVd[qkv_idx(b, j, h, d, N_k)] += P_drop * dOd_v;
                        }
                        // Undo dropout scaling when computing dP w.r.t. pre-dropout P.
                        dP[j] = dP_drop * m;
                    }

                    // softmax backward: dS[j] = (dP[j] - sum_k dP[k]*P[k]) * P[j]
                    float sum_dPP = 0.0f;
                    for (int64_t j = 0; j < N_k; ++j) {
                        sum_dPP += dP[j] * Pd[p_idx(b, h, i, j)];
                    }
                    std::vector<float> dS(N_k);
                    for (int64_t j = 0; j < N_k; ++j) {
                        dS[j] = (dP[j] - sum_dPP) * Pd[p_idx(b, h, i, j)];
                    }

                    // dQ[b,i,h,d] += scale * sum_j dS[j] * K[b,j,h,d]
                    // dK[b,j,h,d] += scale * dS[j] * Q[b,i,h,d]
                    for (int64_t j = 0; j < N_k; ++j) {
                        float ds_scaled = dS[j] * s;
                        for (int64_t d = 0; d < D; ++d) {
                            dQd[qkv_idx(b, i, h, d, N_q)] += ds_scaled * Kd[qkv_idx(b, j, h, d, N_k)];
                            dKd[qkv_idx(b, j, h, d, N_k)] += ds_scaled * Qd[qkv_idx(b, i, h, d, N_q)];
                        }
                    }
                }
            }
        }

        // ----- Reshape dQ/dK/dV back to caller's layout -------------------
        // Canonical layout is [B, N, H, D]. We match the forward's convention.
        auto from_canonical = [this](const Tensor& t) -> Tensor {
            const int64_t nd = orig_rank_;
            if (nd == 2) {
                return t.squeeze(2).squeeze(0);  // [1, N, 1, D] → [N, D]
            } else if (nd == 3) {
                return t.squeeze(2);             // [B, N, 1, D] → [B, N, D]
            } else if (nd == 4) {
                return t;                        // already [B, N, H, D]
            } else {
                // For >=5D inputs, the leading dims were flattened to B. We
                // return a 4D [B', N, H, D] gradient; the autograd engine will
                // reshape to the input's sizes via its input-metadata hook.
                return t;
            }
        };

        Tensor dQ_out = from_canonical(dQ);
        Tensor dK_out = from_canonical(dK);
        Tensor dV_out = from_canonical(dV);

        release_saved_tensors();
        return {dQ_out, dK_out, dV_out};
    }

    std::string name() const override { return "SdpaBackward"; }
};

} // namespace autograd
} // namespace torch
