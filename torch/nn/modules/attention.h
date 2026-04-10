#pragma once

#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include <cmath>
#include <limits>

namespace torch {
namespace nn {

// ============================================================================
// MultiheadAttention - Multi-Head Attention mechanism
// ============================================================================
// Allows the model to jointly attend to information from different
// representation subspaces.
//
// MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
// where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
//
// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

class MultiheadAttention : public Module {
public:
    MultiheadAttention(
        int64_t embed_dim,
        int64_t num_heads,
        double dropout = 0.0,
        bool bias = true,
        bool add_bias_kv = false,
        bool add_zero_attn = false,
        int64_t kdim = 0,
        int64_t vdim = 0,
        bool batch_first = false
    )
        : Module("MultiheadAttention")
        , embed_dim_(embed_dim)
        , num_heads_(num_heads)
        , dropout_(dropout)
        , bias_(bias)
        , add_zero_attn_(add_zero_attn)
        , batch_first_(batch_first)
        , kdim_(kdim == 0 ? embed_dim : kdim)
        , vdim_(vdim == 0 ? embed_dim : vdim)
        , head_dim_(embed_dim / num_heads)
    {
        if (embed_dim % num_heads != 0) {
            throw std::runtime_error(
                "embed_dim must be divisible by num_heads"
            );
        }

        // In-projection weights (combined for efficiency)
        // W_q, W_k, W_v each of size [embed_dim, *dim]
        Tensor in_proj_weight = at::empty({3 * embed_dim, embed_dim});
        register_parameter("in_proj_weight", Parameter(in_proj_weight));

        if (bias) {
            Tensor in_proj_bias = at::empty({3 * embed_dim});
            register_parameter("in_proj_bias", Parameter(in_proj_bias));
        }

        // Out-projection
        Tensor out_proj_weight = at::empty({embed_dim, embed_dim});
        register_parameter("out_proj.weight", Parameter(out_proj_weight));

        if (bias) {
            Tensor out_proj_bias = at::empty({embed_dim});
            register_parameter("out_proj.bias", Parameter(out_proj_bias));
        }

        reset_parameters();
    }

    void reset_parameters() override {
        // Xavier uniform initialization
        double fan_in = static_cast<double>(embed_dim_);
        double std = std::sqrt(2.0 / (fan_in + embed_dim_));
        double bound = std::sqrt(3.0) * std;

        auto* in_proj_weight = get_parameter("in_proj_weight");
        if (in_proj_weight && in_proj_weight->defined()) {
            Tensor w = in_proj_weight->data();
            float* data = w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < w.numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
            }
        }

        auto* out_proj_weight = get_parameter("out_proj.weight");
        if (out_proj_weight && out_proj_weight->defined()) {
            Tensor w = out_proj_weight->data();
            float* data = w.mutable_data_ptr<float>();
            for (int64_t i = 0; i < w.numel(); ++i) {
                data[i] = static_cast<float>((2.0 * ::rand() / RAND_MAX - 1.0) * bound);
            }
        }

        if (bias_) {
            auto* in_proj_bias = get_parameter("in_proj_bias");
            if (in_proj_bias && in_proj_bias->defined()) {
                in_proj_bias->data().zero_();
            }
            auto* out_proj_bias = get_parameter("out_proj.bias");
            if (out_proj_bias && out_proj_bias->defined()) {
                out_proj_bias->data().zero_();
            }
        }
    }

    // Forward with query, key, value
    // Input shapes (when batch_first=False):
    //   query: [L, N, E], key: [S, N, E], value: [S, N, E]
    // Input shapes (when batch_first=True):
    //   query: [N, L, E], key: [N, S, E], value: [N, S, E]
    // Returns: [L, N, E] or [N, L, E] depending on batch_first
    std::pair<Tensor, Tensor> forward_attention(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& attn_mask = Tensor(),
        const Tensor& key_padding_mask = Tensor(),
        bool need_weights = true,
        bool average_attn_weights = true
    ) {
        Tensor q = query;
        Tensor k = key;
        Tensor v = value;

        // Convert to [L, N, E] format if batch_first
        if (batch_first_) {
            q = q.transpose(0, 1);
            k = k.transpose(0, 1);
            v = v.transpose(0, 1);
        }

        int64_t tgt_len = q.size(0);
        int64_t batch_size = q.size(1);
        int64_t src_len = k.size(0);

        // Linear projections
        // q, k, v shape after: [L, N, E], [S, N, E], [S, N, E]
        auto* in_proj_weight = get_parameter("in_proj_weight");
        Tensor W = in_proj_weight->data();  // [3*E, E]

        // Reshape for batch matrix multiply
        // [L*N, E] @ [E, 3*E]^T = [L*N, 3*E]
        Tensor q_flat = q.reshape({tgt_len * batch_size, embed_dim_});
        Tensor k_flat = k.reshape({src_len * batch_size, embed_dim_});
        Tensor v_flat = v.reshape({src_len * batch_size, embed_dim_});

        // Split weight into W_q, W_k, W_v
        // W shape: [3*E, E]
        Tensor W_q = W.slice(0, 0, embed_dim_);              // [E, E]
        Tensor W_k = W.slice(0, embed_dim_, 2 * embed_dim_); // [E, E]
        Tensor W_v = W.slice(0, 2 * embed_dim_, 3 * embed_dim_); // [E, E]

        // Project: Q = XW_q^T, K = XW_k^T, V = XW_v^T
        Tensor Q = q_flat.mm(W_q.t());  // [L*N, E]
        Tensor K = k_flat.mm(W_k.t());  // [S*N, E]
        Tensor V = v_flat.mm(W_v.t());  // [S*N, E]

        // Add bias if present
        if (bias_) {
            auto* in_proj_bias = get_parameter("in_proj_bias");
            if (in_proj_bias && in_proj_bias->defined()) {
                Tensor bias = in_proj_bias->data();
                Tensor b_q = bias.slice(0, 0, embed_dim_);
                Tensor b_k = bias.slice(0, embed_dim_, 2 * embed_dim_);
                Tensor b_v = bias.slice(0, 2 * embed_dim_, 3 * embed_dim_);
                Q = Q.add(b_q);
                K = K.add(b_k);
                V = V.add(b_v);
            }
        }

        // Reshape to [N, num_heads, L, head_dim] for batched attention
        // Q: [L*N, E] -> [L, N, num_heads, head_dim] -> [N, num_heads, L, head_dim]
        Q = Q.reshape({tgt_len, batch_size, num_heads_, head_dim_})
             .permute({1, 2, 0, 3});  // [N, num_heads, L, head_dim]
        K = K.reshape({src_len, batch_size, num_heads_, head_dim_})
             .permute({1, 2, 0, 3});  // [N, num_heads, S, head_dim]
        V = V.reshape({src_len, batch_size, num_heads_, head_dim_})
             .permute({1, 2, 0, 3});  // [N, num_heads, S, head_dim]

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // [N, num_heads, L, head_dim] @ [N, num_heads, head_dim, S] = [N, num_heads, L, S]
        Tensor K_t = K.transpose(-2, -1);  // [N, num_heads, head_dim, S]

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        Tensor attn_scores = batch_matmul(Q, K_t);  // [N, num_heads, L, S]
        attn_scores = attn_scores.mul(at::Scalar(scale));

        // Apply attention mask if provided
        if (attn_mask.defined()) {
            // attn_mask shape: [L, S] or [N*num_heads, L, S]
            // Add to scores (masking with -inf for positions to ignore)
            if (attn_mask.dim() == 2) {
                // Broadcast [L, S] to [N, num_heads, L, S]
                apply_2d_mask(attn_scores, attn_mask);
            } else {
                apply_3d_mask(attn_scores, attn_mask);
            }
        }

        // Apply key padding mask if provided
        if (key_padding_mask.defined()) {
            // key_padding_mask shape: [N, S]
            // True positions should be masked out
            apply_key_padding_mask(attn_scores, key_padding_mask);
        }

        // Softmax over last dimension (source sequence)
        Tensor attn_weights = softmax_last_dim(attn_scores);  // [N, num_heads, L, S]

        // Apply dropout during training
        if (is_training() && dropout_ > 0.0) {
            attn_weights = apply_dropout(attn_weights, dropout_);
        }

        // Apply attention to values
        // [N, num_heads, L, S] @ [N, num_heads, S, head_dim] = [N, num_heads, L, head_dim]
        Tensor attn_output = batch_matmul(attn_weights, V);

        // Reshape back: [N, num_heads, L, head_dim] -> [L, N, E]
        attn_output = attn_output.permute({2, 0, 1, 3})  // [L, N, num_heads, head_dim]
                                 .reshape({tgt_len, batch_size, embed_dim_});

        // Output projection
        auto* out_proj_weight = get_parameter("out_proj.weight");
        Tensor out_flat = attn_output.reshape({tgt_len * batch_size, embed_dim_});
        Tensor output = out_flat.mm(out_proj_weight->data().t());

        if (bias_) {
            auto* out_proj_bias = get_parameter("out_proj.bias");
            if (out_proj_bias && out_proj_bias->defined()) {
                output = output.add(out_proj_bias->data());
            }
        }

        output = output.reshape({tgt_len, batch_size, embed_dim_});

        // Convert back to batch_first format if needed
        if (batch_first_) {
            output = output.transpose(0, 1);
        }

        // Average attention weights across heads if needed
        Tensor weights_out;
        if (need_weights) {
            if (average_attn_weights) {
                // [N, num_heads, L, S] -> [N, L, S]
                weights_out = attn_weights.mean(1);
            } else {
                weights_out = attn_weights;
            }
        }

        return {output, weights_out};
    }

    // Single input forward (self-attention)
    Tensor forward(const Tensor& input) override {
        return forward_attention(input, input, input).first;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "embed_dim=" << embed_dim_
           << ", num_heads=" << num_heads_
           << ", dropout=" << dropout_;
        if (batch_first_) ss << ", batch_first=True";
        return ss.str();
    }

    int64_t embed_dim() const { return embed_dim_; }
    int64_t num_heads() const { return num_heads_; }

private:
    // Batched matrix multiplication
    Tensor batch_matmul(const Tensor& a_in, const Tensor& b_in) {
        // a: [N, num_heads, L, d1], b: [N, num_heads, d1, S]
        // result: [N, num_heads, L, S]
        // Must be contiguous — permuted views have non-trivial strides
        Tensor a = a_in.contiguous();
        Tensor b = b_in.contiguous();
        auto a_sizes = a.sizes().vec();
        auto b_sizes = b.sizes().vec();

        int64_t N = a_sizes[0];
        int64_t H = a_sizes[1];
        int64_t L = a_sizes[2];
        int64_t d1 = a_sizes[3];
        int64_t S = b_sizes[3];

        Tensor result = at::zeros({N, H, L, S});
        float* out_data = result.mutable_data_ptr<float>();
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();

        // omp removed for LCC
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t l = 0; l < L; ++l) {
                    for (int64_t s = 0; s < S; ++s) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < d1; ++k) {
                            // a[n, h, l, k] * b[n, h, k, s]
                            int64_t a_idx = ((n * H + h) * L + l) * d1 + k;
                            int64_t b_idx = ((n * H + h) * d1 + k) * S + s;
                            sum += a_data[a_idx] * b_data[b_idx];
                        }
                        int64_t out_idx = ((n * H + h) * L + l) * S + s;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }

        return result;
    }

    // Softmax over last dimension
    Tensor softmax_last_dim(const Tensor& input) {
        auto sizes = input.sizes().vec();
        int64_t last_dim = sizes.back();
        int64_t outer_size = input.numel() / last_dim;

        Tensor result = input.clone();
        float* out_data = result.mutable_data_ptr<float>();
        const float* in_data = input.data_ptr<float>();

        for (int64_t i = 0; i < outer_size; ++i) {
            int64_t offset = i * last_dim;

            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < last_dim; ++j) {
                max_val = std::max(max_val, in_data[offset + j]);
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int64_t j = 0; j < last_dim; ++j) {
                out_data[offset + j] = std::exp(in_data[offset + j] - max_val);
                sum_exp += out_data[offset + j];
            }

            // Normalize
            for (int64_t j = 0; j < last_dim; ++j) {
                out_data[offset + j] /= sum_exp;
            }
        }

        return result;
    }

    // Apply 2D attention mask
    void apply_2d_mask(Tensor& scores, const Tensor& mask) {
        // scores: [N, num_heads, L, S]
        // mask: [L, S]
        auto sizes = scores.sizes().vec();
        int64_t N = sizes[0];
        int64_t H = sizes[1];
        int64_t L = sizes[2];
        int64_t S = sizes[3];

        float* score_data = scores.mutable_data_ptr<float>();
        const float* mask_data = mask.data_ptr<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t l = 0; l < L; ++l) {
                    for (int64_t s = 0; s < S; ++s) {
                        int64_t score_idx = ((n * H + h) * L + l) * S + s;
                        int64_t mask_idx = l * S + s;
                        score_data[score_idx] += mask_data[mask_idx];
                    }
                }
            }
        }
    }

    // Apply 3D attention mask
    void apply_3d_mask(Tensor& scores, const Tensor& mask) {
        // scores: [N, num_heads, L, S]
        // mask: [N*num_heads, L, S]
        float* score_data = scores.mutable_data_ptr<float>();
        const float* mask_data = mask.data_ptr<float>();

        for (int64_t i = 0; i < scores.numel(); ++i) {
            score_data[i] += mask_data[i];
        }
    }

    // Apply key padding mask
    void apply_key_padding_mask(Tensor& scores, const Tensor& mask) {
        // scores: [N, num_heads, L, S]
        // mask: [N, S] - True where key is padding
        auto sizes = scores.sizes().vec();
        int64_t N = sizes[0];
        int64_t H = sizes[1];
        int64_t L = sizes[2];
        int64_t S = sizes[3];

        float* score_data = scores.mutable_data_ptr<float>();
        const float* mask_data = mask.data_ptr<float>();
        constexpr float NEG_INF = -1e9f;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < S; ++s) {
                // If mask[n, s] is True (> 0.5), mask out
                if (mask_data[n * S + s] > 0.5f) {
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t l = 0; l < L; ++l) {
                            int64_t idx = ((n * H + h) * L + l) * S + s;
                            score_data[idx] = NEG_INF;
                        }
                    }
                }
            }
        }
    }

    // Apply dropout
    Tensor apply_dropout(const Tensor& input, double p) {
        Tensor result = input.clone();
        float* data = result.mutable_data_ptr<float>();
        float scale = 1.0f / (1.0f - static_cast<float>(p));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - p);

        for (int64_t i = 0; i < result.numel(); ++i) {
            if (dist(gen)) {
                data[i] *= scale;
            } else {
                data[i] = 0.0f;
            }
        }

        return result;
    }

    int64_t embed_dim_;
    int64_t num_heads_;
    double dropout_;
    bool bias_;
    bool add_zero_attn_;
    bool batch_first_;
    int64_t kdim_;
    int64_t vdim_;
    int64_t head_dim_;
};

// ============================================================================
// Utility: Generate causal mask for autoregressive models
// ============================================================================

inline Tensor generate_square_subsequent_mask(int64_t sz) {
    // Creates an upper-triangular mask of -inf
    // Used for causal (autoregressive) attention
    Tensor mask = at::zeros({sz, sz});
    float* data = mask.mutable_data_ptr<float>();
    constexpr float NEG_INF = -1e9f;

    for (int64_t i = 0; i < sz; ++i) {
        for (int64_t j = i + 1; j < sz; ++j) {
            data[i * sz + j] = NEG_INF;
        }
    }

    return mask;
}

} // namespace nn
} // namespace torch
