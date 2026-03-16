# nmruntime/quantize.py
# INT8 Quantization utilities for NM Card Mini
# Converts FP32 weights to INT8 + scales for 4x memory reduction

import numpy as np
from typing import Tuple

def quantize_weights_perchannel(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel symmetric quantization for weights.

    Args:
        weights: FP32 weights [out_features, in_features]

    Returns:
        weights_int8: INT8 quantized weights [out_features, in_features]
        scales: FP32 scales per output channel [out_features]
    """
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    # Compute max abs value per output channel (row)
    max_vals = np.max(np.abs(weights), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)  # Avoid division by zero

    # Scale to fit in INT8 range [-127, 127]
    scales = max_vals / 127.0

    # Quantize
    weights_q = np.round(weights / scales)
    weights_q = np.clip(weights_q, -127, 127).astype(np.int8)

    return weights_q, scales.flatten().astype(np.float32)


def quantize_weights_pertensor(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Per-tensor symmetric quantization (simpler, less accurate).

    Args:
        weights: FP32 weights of any shape

    Returns:
        weights_int8: INT8 quantized weights (same shape)
        scale: Single FP32 scale factor
    """
    max_val = np.max(np.abs(weights))
    if max_val < 1e-8:
        return np.zeros_like(weights, dtype=np.int8), 1.0

    scale = max_val / 127.0
    weights_q = np.round(weights / scale)
    weights_q = np.clip(weights_q, -127, 127).astype(np.int8)

    return weights_q, float(scale)


def quantize_activation(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Dynamic per-tensor quantization for activations.
    Called at runtime for each input.

    Args:
        x: FP32 activation tensor

    Returns:
        x_int8: INT8 quantized activation
        scale: FP32 scale factor
    """
    max_val = np.max(np.abs(x))
    if max_val < 1e-8:
        return np.zeros_like(x, dtype=np.int8), 1.0

    scale = max_val / 127.0
    x_q = np.round(x / scale)
    x_q = np.clip(x_q, -127, 127).astype(np.int8)

    return x_q, float(scale)


def pack_int8_to_uint32(data: np.ndarray) -> np.ndarray:
    """
    Pack INT8 array into UINT32 words (4 bytes per word).
    Used for efficient transfer to NM Card.

    Args:
        data: INT8 array (any shape, will be flattened)

    Returns:
        packed: UINT32 array with 4 INT8 values per word
    """
    flat = data.flatten().view(np.uint8)

    # Pad to multiple of 4
    remainder = len(flat) % 4
    if remainder:
        flat = np.pad(flat, (0, 4 - remainder), mode='constant')

    # Reshape and pack: [b0, b1, b2, b3] -> b0 | (b1<<8) | (b2<<16) | (b3<<24)
    words = flat.reshape(-1, 4)
    packed = (words[:, 0].astype(np.uint32) |
              (words[:, 1].astype(np.uint32) << 8) |
              (words[:, 2].astype(np.uint32) << 16) |
              (words[:, 3].astype(np.uint32) << 24))

    return packed


def unpack_uint32_to_int8(packed: np.ndarray, count: int) -> np.ndarray:
    """
    Unpack UINT32 words back to INT8 array.

    Args:
        packed: UINT32 array
        count: Number of INT8 elements to extract

    Returns:
        data: INT8 array of length count
    """
    # Extract bytes
    b0 = (packed & 0xFF).astype(np.uint8)
    b1 = ((packed >> 8) & 0xFF).astype(np.uint8)
    b2 = ((packed >> 16) & 0xFF).astype(np.uint8)
    b3 = ((packed >> 24) & 0xFF).astype(np.uint8)

    # Interleave
    unpacked = np.column_stack([b0, b1, b2, b3]).flatten()

    return unpacked[:count].view(np.int8)


def float32_to_q16_16(x: np.ndarray) -> np.ndarray:
    """Convert FP32 to Q16.16 fixed-point as INT32."""
    return (x * 65536.0).astype(np.int32)


def q16_16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert Q16.16 fixed-point to FP32."""
    return x.astype(np.float32) / 65536.0


def dequantize_weights(weights_int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize INT8 weights back to FP32 (for verification).

    Args:
        weights_int8: INT8 weights [out_features, in_features]
        scales: FP32 scales [out_features]

    Returns:
        weights_fp32: Dequantized weights
    """
    return weights_int8.astype(np.float32) * scales.reshape(-1, 1)


def compute_quantization_error(original: np.ndarray, weights_int8: np.ndarray,
                                scales: np.ndarray) -> dict:
    """
    Compute quantization error statistics.

    Returns:
        dict with 'max_error', 'mean_error', 'relative_error'
    """
    reconstructed = dequantize_weights(weights_int8, scales)
    error = np.abs(original - reconstructed)

    return {
        'max_error': float(np.max(error)),
        'mean_error': float(np.mean(error)),
        'relative_error': float(np.mean(error) / (np.mean(np.abs(original)) + 1e-8))
    }


# Test
if __name__ == "__main__":
    print("=== INT8 Quantization Test ===\n")

    # Create random weights
    np.random.seed(42)
    W = np.random.randn(64, 128).astype(np.float32) * 0.1

    print(f"Original weights: {W.shape}, dtype={W.dtype}")
    print(f"  Range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"  Size: {W.nbytes / 1024:.1f} KB")

    # Quantize
    W_q, scales = quantize_weights_perchannel(W)

    print(f"\nQuantized weights: {W_q.shape}, dtype={W_q.dtype}")
    print(f"  Range: [{W_q.min()}, {W_q.max()}]")
    print(f"  Size: {W_q.nbytes / 1024:.1f} KB")
    print(f"  Scales: {scales.shape}, range=[{scales.min():.6f}, {scales.max():.6f}]")

    # Compute error
    error = compute_quantization_error(W, W_q, scales)
    print(f"\nQuantization error:")
    print(f"  Max error: {error['max_error']:.6f}")
    print(f"  Mean error: {error['mean_error']:.6f}")
    print(f"  Relative error: {error['relative_error']*100:.2f}%")

    # Pack for transfer
    packed = pack_int8_to_uint32(W_q)
    print(f"\nPacked for transfer: {packed.shape} uint32 words")
    print(f"  Size: {packed.nbytes / 1024:.1f} KB")

    # Unpack and verify
    unpacked = unpack_uint32_to_int8(packed, W_q.size)
    unpacked = unpacked.reshape(W_q.shape)
    assert np.array_equal(W_q, unpacked), "Pack/unpack mismatch!"
    print("  Pack/unpack verified OK")

    # Q16.16 conversion
    scales_q16 = float32_to_q16_16(scales)
    print(f"\nScales in Q16.16: range=[{scales_q16.min()}, {scales_q16.max()}]")

    print("\n=== All tests passed! ===")
