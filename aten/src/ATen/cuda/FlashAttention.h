#pragma once

// ============================================================================
// FlashAttention: Memory-Efficient Exact Attention
// ============================================================================
//
// FlashAttention is an IO-aware exact attention algorithm that achieves:
// - O(N) memory instead of O(N²) by not materializing the full attention matrix
// - 2-4x speedup by reducing memory bandwidth
// - Exact computation (not an approximation)
//
// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with
//             IO-Awareness" by Dao et al. (2022)
//
// Algorithm Overview:
// 1. Divide Q, K, V into tiles that fit in SRAM (shared memory)
// 2. For each tile of Q:
//    a. Load Q tile into SRAM
//    b. For each tile of K, V:
//       - Compute S = Q @ K^T (in SRAM)
//       - Apply causal mask if needed
//       - Compute online softmax (track running max and sum)
//       - Update output: O = softmax(S) @ V (accumulate)
// 3. Final normalization with accumulated statistics
//
// ============================================================================

#ifdef PT_USE_CUDA

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <tuple>

namespace at {
namespace cuda {

// ============================================================================
// FlashAttention Configuration
// ============================================================================

struct FlashAttentionConfig {
    // Block sizes (tile dimensions)
    int block_size_q = 64;      // Queries per block
    int block_size_kv = 64;     // Keys/Values per block

    // Whether to use causal masking
    bool is_causal = false;

    // Dropout probability (0 = no dropout)
    float dropout_p = 0.0f;

    // Softmax scale (default: 1/sqrt(head_dim))
    float softmax_scale = -1.0f;  // -1 = auto

    // Return attention weights for debugging (disables memory savings)
    bool return_attention_weights = false;
};

// ============================================================================
// FlashAttention Forward
// ============================================================================
//
// Input shapes:
//   query:  [batch_size, seq_len_q, num_heads, head_dim]
//   key:    [batch_size, seq_len_k, num_heads, head_dim]
//   value:  [batch_size, seq_len_k, num_heads, head_dim]
//
// Output shape:
//   output: [batch_size, seq_len_q, num_heads, head_dim]
//
// Returns:
//   - output tensor
//   - logsumexp (for backward): [batch_size, num_heads, seq_len_q]
//   - (optional) attention weights: [batch_size, num_heads, seq_len_q, seq_len_k]

std::tuple<Tensor, Tensor, Tensor> flash_attention_forward(
    const Tensor& query,     // [B, N_q, H, D]
    const Tensor& key,       // [B, N_k, H, D]
    const Tensor& value,     // [B, N_k, H, D]
    const FlashAttentionConfig& config = FlashAttentionConfig()
);

// ============================================================================
// FlashAttention Backward
// ============================================================================
//
// Returns:
//   - grad_query:  [batch_size, seq_len_q, num_heads, head_dim]
//   - grad_key:    [batch_size, seq_len_k, num_heads, head_dim]
//   - grad_value:  [batch_size, seq_len_k, num_heads, head_dim]

std::tuple<Tensor, Tensor, Tensor> flash_attention_backward(
    const Tensor& grad_output,  // [B, N_q, H, D]
    const Tensor& query,        // [B, N_q, H, D]
    const Tensor& key,          // [B, N_k, H, D]
    const Tensor& value,        // [B, N_k, H, D]
    const Tensor& output,       // [B, N_q, H, D] - forward output
    const Tensor& logsumexp,    // [B, H, N_q] - from forward
    const FlashAttentionConfig& config = FlashAttentionConfig()
);

// ============================================================================
// Scaled Dot-Product Attention (high-level API)
// ============================================================================
//
// Drop-in replacement for standard attention with automatic FlashAttention
// when inputs are on CUDA and meet requirements.

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask = Tensor(),  // Optional mask
    float dropout_p = 0.0f,
    bool is_causal = false,
    float scale = -1.0f  // -1 = 1/sqrt(head_dim)
);

// ============================================================================
// Multi-Head Attention (convenience wrapper)
// ============================================================================
//
// Handles:
// - Linear projections (Q, K, V)
// - Reshaping for multi-head
// - FlashAttention
// - Output projection

Tensor multi_head_flash_attention(
    const Tensor& query,      // [B, N_q, embed_dim]
    const Tensor& key,        // [B, N_k, embed_dim]
    const Tensor& value,      // [B, N_k, embed_dim]
    const Tensor& q_proj_weight,   // [embed_dim, embed_dim]
    const Tensor& k_proj_weight,   // [embed_dim, embed_dim]
    const Tensor& v_proj_weight,   // [embed_dim, embed_dim]
    const Tensor& out_proj_weight, // [embed_dim, embed_dim]
    const Tensor& q_proj_bias = Tensor(),
    const Tensor& k_proj_bias = Tensor(),
    const Tensor& v_proj_bias = Tensor(),
    const Tensor& out_proj_bias = Tensor(),
    int num_heads = 8,
    float dropout_p = 0.0f,
    bool is_causal = false
);

// ============================================================================
// Utility Functions
// ============================================================================

// Check if FlashAttention can be used for given inputs
bool can_use_flash_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value
);

// Get optimal block sizes for given dimensions
std::pair<int, int> get_flash_attention_block_sizes(
    int head_dim,
    int seq_len,
    c10::ScalarType dtype
);

} // namespace cuda
} // namespace at

#endif // PT_USE_CUDA
