"""
ops.py -- Pure numpy implementations of Qwen3-4B operations.

These run on the HOST CPU as fallbacks for ops that are not yet dispatched
to the NM Card.  For matmul we provide a stub that the inference engine
can replace with an NMCard dispatch call.

All shapes follow the convention:
  - hidden = 2560
  - num_heads = 32, num_kv_heads = 8, head_dim = 128
  - intermediate = 9728
  - vocab = 151936
  - num_layers = 36
"""

import numpy as np
from typing import Optional, Tuple

# ============================================================================
# RMSNorm
# ============================================================================

def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)

    x: (..., hidden)
    weight: (hidden,)
    """
    # Compute variance along last axis
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_normed = x * np.reciprocal(np.sqrt(variance + eps))
    return (x_normed * weight).astype(x.dtype)


# ============================================================================
# RoPE (Rotary Position Embedding)
# ============================================================================

def precompute_freqs_cis(head_dim: int, max_seq_len: int,
                          theta: float = 1000000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin tables for RoPE.

    Returns:
        cos_table: (max_seq_len, head_dim//2)  float32
        sin_table: (max_seq_len, head_dim//2)  float32
    """
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # (seq_len, head_dim//2)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
               position_offset: int = 0) -> np.ndarray:
    """Apply rotary embeddings to x.

    x: (batch, num_heads, seq_len, head_dim) or (seq_len, num_heads, head_dim)
    cos, sin: (max_seq_len, head_dim//2) -- sliced to [position_offset : position_offset+seq_len]
    """
    if x.ndim == 4:
        seq_len = x.shape[2]
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        c = cos[position_offset:position_offset + seq_len][np.newaxis, np.newaxis, :, :]
        s = sin[position_offset:position_offset + seq_len][np.newaxis, np.newaxis, :, :]
    elif x.ndim == 3:
        seq_len = x.shape[0]
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        c = cos[position_offset:position_offset + seq_len][:, np.newaxis, :]
        s = sin[position_offset:position_offset + seq_len][:, np.newaxis, :]
    else:
        raise ValueError(f"apply_rope: unsupported ndim={x.ndim}")

    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return np.concatenate([out1, out2], axis=-1).astype(x.dtype)


# ============================================================================
# SiLU (Swish) activation
# ============================================================================

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x)"""
    # Use float32 for numerical stability
    xf = x.astype(np.float32)
    return (xf / (1.0 + np.exp(-xf))).astype(x.dtype)


# ============================================================================
# Softmax
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    xf = x.astype(np.float32)
    x_max = np.max(xf, axis=axis, keepdims=True)
    e_x = np.exp(xf - x_max)
    return (e_x / np.sum(e_x, axis=axis, keepdims=True)).astype(x.dtype)


# ============================================================================
# GQA (Grouped Query Attention)
# ============================================================================

def gqa_attention(
    q: np.ndarray,       # (batch, num_heads, seq_len, head_dim)
    k: np.ndarray,       # (batch, num_kv_heads, seq_len, head_dim)
    v: np.ndarray,       # (batch, num_kv_heads, seq_len, head_dim)
    num_heads: int = 32,
    num_kv_heads: int = 8,
    causal: bool = True,
    scale: Optional[float] = None,
    kv_cache_k: Optional[np.ndarray] = None,  # (batch, num_kv_heads, prev_len, head_dim)
    kv_cache_v: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grouped-Query Attention with optional KV cache.

    Returns:
        output: (batch, num_heads, seq_len, head_dim)
        new_k_cache: (batch, num_kv_heads, total_len, head_dim)
        new_v_cache: (batch, num_kv_heads, total_len, head_dim)
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim).astype(np.float32)

    # Append to KV cache
    if kv_cache_k is not None:
        k = np.concatenate([kv_cache_k, k], axis=2)
        v = np.concatenate([kv_cache_v, v], axis=2)

    # GQA: repeat KV heads to match Q heads
    # num_heads=32, num_kv_heads=8 => repeat_factor=4
    repeat_factor = num_heads // num_kv_heads
    if repeat_factor > 1:
        # k: (B, kv_heads, S, D) -> (B, kv_heads, 1, S, D) -> (B, kv_heads, rep, S, D) -> (B, heads, S, D)
        k_expanded = np.repeat(k, repeat_factor, axis=1)
        v_expanded = np.repeat(v, repeat_factor, axis=1)
    else:
        k_expanded = k
        v_expanded = v

    # Attention scores: (B, heads, q_len, kv_len)
    scores = np.matmul(q, k_expanded.transpose(0, 1, 3, 2)) * scale

    # Causal mask
    if causal:
        q_len = q.shape[2]
        kv_len = k_expanded.shape[2]
        if q_len == 1:
            # Single token decode: no future tokens to mask, skip mask creation entirely
            pass  # scores are already correct
        else:
            # Full causal mask for prefill
            row_idx = np.arange(q_len)[:, np.newaxis]
            col_idx = np.arange(kv_len)[np.newaxis, :]
            offset = kv_len - q_len
            mask = col_idx > (row_idx + offset)
            scores[mask[np.newaxis, np.newaxis, :, :]] = -1e9  # In-place, no copy

    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, v_expanded)

    return output, k, v


# ============================================================================
# Linear (matmul + optional bias) -- host fallback
# ============================================================================

def linear(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None) -> np.ndarray:
    """y = x @ weight^T + bias

    x: (..., in_features)
    weight: (out_features, in_features)
    bias: (out_features,) or None
    """
    # For single-token generation, x is (1, hidden) or (hidden,)
    out = np.matmul(x, weight.T)
    if bias is not None:
        out = out + bias
    return out


# ============================================================================
# Embedding lookup (CPU-only, no card dispatch needed)
# ============================================================================

def embedding(input_ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Simple embedding lookup.

    input_ids: (batch, seq_len) int64
    weight: (vocab_size, hidden) float32
    Returns: (batch, seq_len, hidden)
    """
    return weight[input_ids]


# ============================================================================
# Top-k / Top-p sampling
# ============================================================================

def sample_top_p(logits: np.ndarray, temperature: float = 1.0,
                 top_p: float = 0.9, top_k: int = 50) -> int:
    """Sample next token from logits with temperature, top-k, top-p.

    logits: (vocab_size,) -- raw logits for one position
    Returns: token_id (int)
    """
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float32) / temperature

    # Top-k filtering
    if top_k > 0 and top_k < len(logits):
        indices_to_remove = np.argsort(logits)[:-top_k]
        logits[indices_to_remove] = -np.inf

    # Softmax
    probs = softmax(logits)

    # Top-p (nucleus) filtering
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    # Remove tokens with cumulative prob above top_p
    cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
    allowed_indices = sorted_indices[:cutoff_idx]
    allowed_probs = probs[allowed_indices]

    # Renormalize
    allowed_probs = allowed_probs / allowed_probs.sum()

    # Sample
    chosen = np.random.choice(allowed_indices, p=allowed_probs)
    return int(chosen)
