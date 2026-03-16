#pragma once
// ============================================================================
// NMCardLoRA.h — Low-Rank Adaptation for NM Card Mini
// ============================================================================
// LoRA: W_effective = W_frozen + A @ B
//   W_frozen: [D, D] — large, stays on DDR, never updated
//   A: [D, R] — small, trainable (R=4-8)
//   B: [R, D] — small, trainable
//
// Benefits for NM Card:
//   - Less data transfer per step (only A,B updated, not full W)
//   - Smaller matmuls for gradient computation
//   - W_frozen preloaded to DDR once, reused forever
//   - Q16.16 safe: smaller matrices = smaller intermediate sums
//
// Usage:
//   LoRALayer lora(D, D, rank=4);
//   lora.preload_to_card(hw);  // upload W_frozen + A + B
//   Tensor y = lora.forward(x, hw);  // x @ (W + A@B)
//   lora.update(dA, dB, lr);  // only update A, B
// ============================================================================

#include <cstdint>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

namespace at {
namespace nmcard {

struct LoRALayer {
    int in_dim;
    int out_dim;
    int rank;

    // Frozen weight (large)
    std::vector<float> W_frozen;  // [in_dim, out_dim]

    // LoRA adapters (small)
    std::vector<float> A;  // [in_dim, rank]
    std::vector<float> B;  // [rank, out_dim]

    // DDR addresses (after preload)
    uint32_t ddr_W = 0;
    uint32_t ddr_A = 0;
    uint32_t ddr_B = 0;

    LoRALayer(int in_d, int out_d, int r = 4)
        : in_dim(in_d), out_dim(out_d), rank(r),
          W_frozen(in_d * out_d),
          A(in_d * r),
          B(r * out_d)
    {
        // Xavier init for W
        std::mt19937 rng(42);
        float scale_w = std::sqrt(2.0f / (in_d + out_d));
        std::normal_distribution<float> dist_w(0.0f, scale_w);
        for (auto& w : W_frozen) w = dist_w(rng);

        // Small init for A, zero for B (LoRA convention)
        float scale_a = 0.01f;
        std::normal_distribution<float> dist_a(0.0f, scale_a);
        for (auto& a : A) a = dist_a(rng);
        std::fill(B.begin(), B.end(), 0.0f);
    }

    // Total effective weight = W_frozen + A @ B
    // But we compute x @ W_frozen + x @ A @ B separately
    // x @ W_frozen = big matmul (preloaded, card does once)
    // x @ A = small matmul [T, in_dim] @ [in_dim, rank] = [T, rank]
    // [T, rank] @ B = [T, rank] @ [rank, out_dim] = [T, out_dim]
    // Much less computation than x @ (W_frozen + A@B)

    // Forward: y = x @ W_frozen + x @ A @ B
    // Separate into: y1 = card_matmul(x, W_frozen)
    //                y2 = card_matmul(card_matmul(x, A), B)
    //                y = y1 + y2

    // Gradient for LoRA:
    //   dy/dA = x^T @ (dl @ B^T)
    //   dy/dB = (x @ A)^T @ dl
    // Both are small matmuls involving rank-sized matrices

    int trainable_params() const { return in_dim * rank + rank * out_dim; }
    int frozen_params() const { return in_dim * out_dim; }
    int total_params() const { return frozen_params() + trainable_params(); }

    float compression_ratio() const {
        return static_cast<float>(trainable_params()) / total_params();
    }
};

// Full model with LoRA layers
struct LoRAModel {
    std::vector<LoRALayer> layers;
    int vocab_size;
    int embed_dim;
    int num_layers;
    int rank;

    std::vector<float> embed;     // [V, D]
    std::vector<float> pos_embed; // [T, D]
    std::vector<float> lm_head;   // [D, V]

    LoRAModel(int V, int D, int num_layers, int ffn_dim, int r = 4, int T = 32)
        : vocab_size(V), embed_dim(D), num_layers(num_layers), rank(r),
          embed(V * D), pos_embed(T * D), lm_head(D * V)
    {
        // Create LoRA layers for each FFN
        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(D, ffn_dim, r);  // W1: D→F
            layers.emplace_back(ffn_dim, D, r);  // W2: F→D
        }

        // Init embed, lm_head
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& e : embed) e = dist(rng);
        for (auto& p : pos_embed) p = dist(rng) * 0.5f;
        float scale = std::sqrt(2.0f / (D + V));
        std::normal_distribution<float> dist_h(0.0f, scale);
        for (auto& h : lm_head) h = dist_h(rng);
    }

    int trainable_params() const {
        int total = 0;
        for (auto& l : layers) total += l.trainable_params();
        total += embed.size() + pos_embed.size() + lm_head.size();
        return total;
    }

    int frozen_params() const {
        int total = 0;
        for (auto& l : layers) total += l.frozen_params();
        return total;
    }
};

} // namespace nmcard
} // namespace at
