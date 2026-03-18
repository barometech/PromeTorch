# nmruntime/model.py
# TinyLlama model implementation for NM Card Mini

import json
import struct
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from .quantize import (
    unpack_uint32_to_int8,
    q16_16_to_float32,
    dequantize_weights
)


class TinyLlamaConfig:
    """TinyLlama model configuration."""

    def __init__(self, config_dict: dict):
        self.hidden_size = config_dict.get("hidden_size", 2048)
        self.intermediate_size = config_dict.get("intermediate_size", 5632)
        self.num_hidden_layers = config_dict.get("num_hidden_layers", 22)
        self.num_attention_heads = config_dict.get("num_attention_heads", 32)
        self.num_key_value_heads = config_dict.get("num_key_value_heads", 4)
        self.head_dim = config_dict.get("head_dim", 64)
        self.vocab_size = config_dict.get("vocab_size", 32000)
        self.rms_norm_eps = config_dict.get("rms_norm_eps", 1e-5)
        self.max_position_embeddings = config_dict.get("max_position_embeddings", 2048)


class WeightLoader:
    """Loads quantized weights from binary files."""

    def __init__(self, weights_dir: Path):
        self.weights_dir = Path(weights_dir)
        self.manifest = self._load_manifest()
        self.config = TinyLlamaConfig(self.manifest["config"])
        self._cache = {}

    def _load_manifest(self) -> dict:
        manifest_path = self.weights_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            return json.load(f)

    def load_weight(self, name: str) -> np.ndarray:
        """Load and dequantize a weight tensor."""
        if name in self._cache:
            return self._cache[name]

        layer_info = self.manifest["layers"][name]
        file_path = self.weights_dir / layer_info["file"]
        shape = tuple(layer_info["shape"])

        with open(file_path, "rb") as f:
            if layer_info["type"] == "int8":
                # Read header
                out_features, in_features = struct.unpack("<II", f.read(8))

                # Read packed INT8 data
                packed_words = layer_info["packed_words"]
                packed_data = np.frombuffer(f.read(packed_words * 4), dtype=np.uint32)

                # Read scales
                scales_count = layer_info["scales_count"]
                scales_q16 = np.frombuffer(f.read(scales_count * 4), dtype=np.int32)
                scales = q16_16_to_float32(scales_q16)

                # Unpack and dequantize
                int8_data = unpack_uint32_to_int8(packed_data, out_features * in_features)
                int8_data = int8_data.reshape(out_features, in_features)
                weight = dequantize_weights(int8_data, scales)

            else:  # q16
                # Read header
                ndim = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]

                # Read Q16.16 data
                total_elements = 1
                for d in dims:
                    total_elements *= d
                q16_data = np.frombuffer(f.read(total_elements * 4), dtype=np.int32)
                weight = q16_16_to_float32(q16_data).reshape(dims)

        self._cache[name] = weight
        return weight

    def get_layer_weights(self, layer_idx: int) -> dict:
        """Get all weights for a specific layer."""
        prefix = f"model.layers.{layer_idx}"
        return {
            "q_proj": self.load_weight(f"{prefix}.self_attn.q_proj.weight"),
            "k_proj": self.load_weight(f"{prefix}.self_attn.k_proj.weight"),
            "v_proj": self.load_weight(f"{prefix}.self_attn.v_proj.weight"),
            "o_proj": self.load_weight(f"{prefix}.self_attn.o_proj.weight"),
            "gate_proj": self.load_weight(f"{prefix}.mlp.gate_proj.weight"),
            "up_proj": self.load_weight(f"{prefix}.mlp.up_proj.weight"),
            "down_proj": self.load_weight(f"{prefix}.mlp.down_proj.weight"),
            "input_layernorm": self.load_weight(f"{prefix}.input_layernorm.weight"),
            "post_attention_layernorm": self.load_weight(f"{prefix}.post_attention_layernorm.weight"),
        }


class RoPE:
    """Rotary Position Embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)

        self.cos = np.cos(freqs).astype(np.float32)
        self.sin = np.sin(freqs).astype(np.float32)

    def apply(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """Apply RoPE to input tensor [batch, seq_len, num_heads, head_dim].

        Uses Llama/HuggingFace style: x * cos + rotate_half(x) * sin
        where rotate_half splits tensor in half and swaps with negation.
        """
        seq_len = x.shape[1]
        head_dim = x.shape[-1]

        # Get cos/sin for these positions [seq_len, dim//2]
        cos = self.cos[start_pos:start_pos + seq_len]
        sin = self.sin[start_pos:start_pos + seq_len]

        # Duplicate to full head_dim: [seq_len, dim//2] -> [seq_len, dim]
        cos = np.repeat(cos, 2, axis=-1)
        sin = np.repeat(sin, 2, axis=-1)

        # Broadcast to match x shape [batch, seq_len, num_heads, head_dim]
        cos = cos[np.newaxis, :, np.newaxis, :]
        sin = sin[np.newaxis, :, np.newaxis, :]

        # rotate_half: split in half and swap with negation
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        x_rotated = np.concatenate([-x2, x1], axis=-1)

        # Apply: x * cos + rotate_half(x) * sin
        return x * cos + x_rotated * sin


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMS Normalization."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return x_norm * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation (Swish) with numerical stability."""
    # Clip to prevent overflow in exp
    x_clipped = np.clip(x, -88.0, 88.0)
    return x * (1.0 / (1.0 + np.exp(-x_clipped)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax with numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class KVCache:
    """Key-Value cache for efficient inference with pre-allocated ring buffer."""

    def __init__(self, num_layers: int, max_seq_len: int,
                 num_kv_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre-allocate cache buffers: [batch=1, max_seq_len, num_kv_heads, head_dim]
        self.k_cache = [np.zeros((1, max_seq_len, num_kv_heads, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, max_seq_len, num_kv_heads, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.seq_len = 0

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update cache in-place and return valid slice of K, V."""
        new_seq_len = k.shape[1]
        # Layer 0 is always called first -- snapshot write position for all layers
        if layer_idx == 0:
            self._write_pos = self.seq_len
            self.seq_len += new_seq_len

        pos = self._write_pos
        # Write new tokens into pre-allocated buffer (no concatenation)
        self.k_cache[layer_idx][:, pos:pos + new_seq_len] = k
        self.v_cache[layer_idx][:, pos:pos + new_seq_len] = v

        return self.k_cache[layer_idx][:, :self.seq_len], self.v_cache[layer_idx][:, :self.seq_len]

    def clear(self):
        """Clear the cache (zero out buffers)."""
        for i in range(self.num_layers):
            self.k_cache[i][:] = 0
            self.v_cache[i][:] = 0
        self.seq_len = 0


class TinyLlamaModel:
    """TinyLlama model for inference."""

    def __init__(self, weights_dir: Path, use_cache: bool = True):
        self.loader = WeightLoader(weights_dir)
        self.config = self.loader.config
        self.use_cache = use_cache

        # Load embeddings
        print("Loading embeddings...")
        self.embed_tokens = self.loader.load_weight("model.embed_tokens.weight")
        self.lm_head = self.loader.load_weight("lm_head.weight")
        self.norm = self.loader.load_weight("model.norm.weight")

        # Initialize RoPE
        self.rope = RoPE(
            dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings
        )

        # Initialize KV cache
        if use_cache:
            self.kv_cache = KVCache(
                num_layers=self.config.num_hidden_layers,
                max_seq_len=self.config.max_position_embeddings,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim
            )
        else:
            self.kv_cache = None

        # Pre-load all layer weights
        print("Loading layer weights...")
        self.layers = []
        for i in range(self.config.num_hidden_layers):
            print(f"  Layer {i+1}/{self.config.num_hidden_layers}")
            self.layers.append(self.loader.get_layer_weights(i))

        print("Model loaded!")

    def _attention(self, hidden: np.ndarray, layer_weights: dict,
                   layer_idx: int, start_pos: int) -> np.ndarray:
        """Compute attention for a single layer."""
        batch_size, seq_len, _ = hidden.shape

        # Project to Q, K, V
        q = hidden @ layer_weights["q_proj"].T  # [B, S, hidden] @ [hidden, hidden] -> [B, S, hidden]
        k = hidden @ layer_weights["k_proj"].T  # [B, S, hidden] @ [kv_dim, hidden].T -> [B, S, kv_dim]
        v = hidden @ layer_weights["v_proj"].T

        # Reshape for multi-head attention
        # Q: [B, S, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim)
        # K, V: [B, S, num_kv_heads, head_dim]
        k = k.reshape(batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim)
        v = v.reshape(batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim)

        # Apply RoPE
        q = self.rope.apply(q, start_pos)
        k = self.rope.apply(k, start_pos)

        # Update KV cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(layer_idx, k, v)

        # Expand K, V for GQA (repeat for each query head group)
        num_rep = self.config.num_attention_heads // self.config.num_key_value_heads
        if num_rep > 1:
            k = np.repeat(k, num_rep, axis=2)
            v = np.repeat(v, num_rep, axis=2)

        # Transpose for attention: [B, num_heads, S, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Attention scores: [B, num_heads, S_q, S_kv]
        scale = 1.0 / np.sqrt(self.config.head_dim)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        kv_len = k.shape[2]
        if seq_len > 1:
            # During prefill, apply full causal mask
            mask = np.triu(np.ones((seq_len, kv_len), dtype=np.float32) * -1e9, k=1)
            scores = scores + mask
        # During generation (seq_len=1), no mask needed as we only attend to past

        # Softmax
        attn_weights = softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = np.matmul(attn_weights, v)  # [B, num_heads, S, head_dim]

        # Reshape back: [B, S, hidden]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output = attn_output @ layer_weights["o_proj"].T

        return output

    def _ffn(self, hidden: np.ndarray, layer_weights: dict) -> np.ndarray:
        """Feed-forward network."""
        # Gate and up projections
        gate = hidden @ layer_weights["gate_proj"].T
        up = hidden @ layer_weights["up_proj"].T

        # SiLU activation on gate, multiply with up
        hidden = silu(gate) * up

        # Down projection
        output = hidden @ layer_weights["down_proj"].T

        return output

    def forward(self, input_ids: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            start_pos: Starting position for RoPE (used with KV cache)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden = self.embed_tokens[input_ids]  # [B, S, hidden]

        # Transformer layers
        for layer_idx, layer_weights in enumerate(self.layers):
            # Pre-attention norm
            residual = hidden
            hidden = rms_norm(hidden, layer_weights["input_layernorm"], self.config.rms_norm_eps)

            # Attention
            hidden = self._attention(hidden, layer_weights, layer_idx, start_pos)
            hidden = residual + hidden

            # Pre-FFN norm
            residual = hidden
            hidden = rms_norm(hidden, layer_weights["post_attention_layernorm"], self.config.rms_norm_eps)

            # FFN
            hidden = self._ffn(hidden, layer_weights)
            hidden = residual + hidden

        # Final norm
        hidden = rms_norm(hidden, self.norm, self.config.rms_norm_eps)

        # LM head
        logits = hidden @ self.lm_head.T  # [B, S, vocab]

        return logits

    def generate_next_token(self, logits: np.ndarray, temperature: float = 1.0,
                           top_k: int = 50, top_p: float = 0.9) -> int:
        """Sample next token from logits."""
        # Get last token logits
        logits = logits[0, -1, :]  # [vocab]

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits[indices_to_remove] = -np.inf

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = np.cumsum(softmax(sorted_logits))

            # Use searchsorted to find cutoff index efficiently
            cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
            logits[sorted_indices[cutoff_idx:]] = -np.inf

        # Sample
        probs = softmax(logits)
        next_token = np.random.choice(len(probs), p=probs)

        return int(next_token)

    def reset_cache(self):
        """Reset KV cache for new generation."""
        if self.kv_cache is not None:
            self.kv_cache.clear()
