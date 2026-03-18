"""
layer_loader.py -- Safetensors parser WITHOUT torch dependency.

Reads .safetensors files directly using the binary format spec:
  - 8 bytes: header_size (uint64 LE)
  - header_size bytes: JSON header with tensor metadata
  - remaining: raw tensor data

Supports BF16 -> FP32 conversion and optional INT8 quantization.
"""

import json
import struct
import os
import numpy as np
from typing import Dict, Optional, Tuple, List

# Safetensors dtype mapping to numpy
SAFETENSORS_DTYPE_MAP = {
    "F32":  (np.float32, 4),
    "F16":  (np.float16, 2),
    "BF16": (None, 2),       # No native numpy BF16; handled specially
    "F64":  (np.float64, 8),
    "I8":   (np.int8, 1),
    "I16":  (np.int16, 2),
    "I32":  (np.int32, 4),
    "I64":  (np.int64, 8),
    "U8":   (np.uint8, 1),
    "U16":  (np.uint16, 2),
    "U32":  (np.uint32, 4),
    "U64":  (np.uint64, 8),
    "BOOL": (np.bool_, 1),
}


def bf16_to_fp32(raw_bytes: bytes, count: int) -> np.ndarray:
    """Convert BF16 raw bytes to FP32 numpy array.

    BF16 is the upper 16 bits of FP32, so we shift left by 16.
    """
    bf16 = np.frombuffer(raw_bytes, dtype=np.uint16, count=count)
    # Shift to upper 16 bits of 32-bit float
    fp32_bits = bf16.astype(np.uint32) << 16
    return fp32_bits.view(np.float32)


def quantize_int8_per_channel(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weight to INT8 with per-output-channel (row) scaling.

    weight: (out_features, in_features) float32
    Returns:
        q_weight: (out_features, in_features) int8
        scales: (out_features,) float32  -- multiply q_weight * scales to dequantize
    """
    # Per-row absmax
    amax = np.max(np.abs(weight), axis=-1, keepdims=True)
    amax = np.maximum(amax, 1e-10)  # avoid division by zero
    scales = amax.squeeze(-1) / 127.0

    # Quantize
    q_weight = np.clip(np.round(weight / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    return q_weight, scales.astype(np.float32)


def dequantize_int8(q_weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize INT8 weight back to FP32."""
    return q_weight.astype(np.float32) * scales[:, np.newaxis]


class SafetensorsFile:
    """Reads a single .safetensors file lazily (memory-mapped)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header: Dict = {}
        self.data_offset: int = 0
        self._parse_header()

    def _parse_header(self):
        with open(self.filepath, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_json = f.read(header_size).decode("utf-8")
            self.header = json.loads(header_json)
            self.data_offset = 8 + header_size

    def tensor_names(self) -> List[str]:
        """List all tensor names (excluding __metadata__)."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor_info(self, name: str) -> Optional[Dict]:
        """Get metadata for a tensor: dtype, shape, data_offsets."""
        return self.header.get(name)

    def load_tensor(self, name: str, to_float32: bool = True) -> np.ndarray:
        """Load a single tensor from the file.

        Args:
            name: tensor name
            to_float32: if True, convert BF16/FP16 to FP32
        Returns:
            numpy array
        """
        info = self.header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]

        numel = 1
        for s in shape:
            numel *= s

        with open(self.filepath, "rb") as f:
            f.seek(self.data_offset + start)
            raw = f.read(end - start)

        if dtype_str == "BF16":
            arr = bf16_to_fp32(raw, numel)
            arr = arr.reshape(shape)
            return arr  # already float32
        else:
            np_dtype, _ = SAFETENSORS_DTYPE_MAP[dtype_str]
            arr = np.frombuffer(raw, dtype=np_dtype, count=numel).reshape(shape)
            if to_float32 and arr.dtype != np.float32 and np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)
            return arr.copy()  # copy to detach from mmap


class LayerLoader:
    """Loads per-layer weight files created by ModelSplitter.

    Each layer is a .safetensors file containing all weights for that layer.
    Supports optional INT8 quantization on load.
    """

    def __init__(self, split_dir: str, quantize_int8: bool = False):
        """
        Args:
            split_dir: directory with per-layer .safetensors files
            quantize_int8: if True, quantize linear weights to INT8 on load
        """
        self.split_dir = split_dir
        self.quantize_int8 = quantize_int8
        self._layer_files: Dict[str, str] = {}
        self._scan_directory()

    def _scan_directory(self):
        """Find all layer-*.safetensors files."""
        for fname in sorted(os.listdir(self.split_dir)):
            if fname.endswith(".safetensors"):
                # Extract layer identifier from filename
                # e.g., "layer_00.safetensors", "embed.safetensors", "lm_head.safetensors"
                layer_id = fname.replace(".safetensors", "")
                self._layer_files[layer_id] = os.path.join(self.split_dir, fname)

    def layer_names(self) -> List[str]:
        """Return ordered list of layer identifiers."""
        return list(self._layer_files.keys())

    def load_layer(self, layer_id: str) -> Dict[str, np.ndarray]:
        """Load all weights for a single layer.

        Returns dict of {short_name: np.ndarray} where short_name is the
        weight name relative to the layer (e.g., "self_attn.q_proj.weight").

        If quantize_int8 is True, linear weight matrices are quantized and
        returned as (q_weight, scales) tuples under keys "name.qweight" and
        "name.scales".
        """
        filepath = self._layer_files[layer_id]
        sf = SafetensorsFile(filepath)

        state_dict: Dict[str, np.ndarray] = {}

        for tensor_name in sf.tensor_names():
            arr = sf.load_tensor(tensor_name, to_float32=True)

            if self.quantize_int8 and self._is_linear_weight(tensor_name, arr):
                q_weight, scales = quantize_int8_per_channel(arr)
                state_dict[tensor_name + ".qweight"] = q_weight
                state_dict[tensor_name + ".scales"] = scales
            else:
                state_dict[tensor_name] = arr

        return state_dict

    def unload_layer(self, state_dict: Dict[str, np.ndarray]):
        """Explicitly release layer weights to free memory."""
        state_dict.clear()

    def estimate_layer_bytes(self, layer_id: str) -> int:
        """Estimate memory needed for a layer in bytes."""
        filepath = self._layer_files[layer_id]
        sf = SafetensorsFile(filepath)
        total = 0
        for tensor_name in sf.tensor_names():
            info = sf.get_tensor_info(tensor_name)
            numel = 1
            for s in info["shape"]:
                numel *= s
            if self.quantize_int8 and "weight" in tensor_name and len(info["shape"]) == 2:
                # INT8: 1 byte per element + 4 bytes per row for scale
                total += numel + info["shape"][0] * 4
            else:
                total += numel * 4  # float32
        return total

    @staticmethod
    def _is_linear_weight(name: str, arr: np.ndarray) -> bool:
        """Check if this tensor is a linear weight (2D matrix, not norm/bias)."""
        if arr.ndim != 2:
            return False
        if "norm" in name or "layernorm" in name:
            return False
        if name.endswith(".bias"):
            return False
        return "weight" in name


class ModelConfigLoader:
    """Load model config.json without transformers dependency."""

    def __init__(self, model_dir: str):
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    @property
    def hidden_size(self) -> int:
        return self.config.get("hidden_size", 2560)

    @property
    def num_attention_heads(self) -> int:
        return self.config.get("num_attention_heads", 32)

    @property
    def num_key_value_heads(self) -> int:
        return self.config.get("num_key_value_heads", 8)

    @property
    def head_dim(self) -> int:
        return self.config.get("head_dim", self.hidden_size // self.num_attention_heads)

    @property
    def intermediate_size(self) -> int:
        return self.config.get("intermediate_size", 9728)

    @property
    def num_hidden_layers(self) -> int:
        return self.config.get("num_hidden_layers", 36)

    @property
    def vocab_size(self) -> int:
        return self.config.get("vocab_size", 151936)

    @property
    def rms_norm_eps(self) -> float:
        return self.config.get("rms_norm_eps", 1e-6)

    @property
    def rope_theta(self) -> float:
        return self.config.get("rope_theta", 1000000.0)

    @property
    def max_position_embeddings(self) -> int:
        return self.config.get("max_position_embeddings", 32768)

    @property
    def tie_word_embeddings(self) -> bool:
        return self.config.get("tie_word_embeddings", False)

    def __repr__(self):
        return (f"Qwen3Config(hidden={self.hidden_size}, heads={self.num_attention_heads}, "
                f"kv_heads={self.num_key_value_heads}, layers={self.num_hidden_layers}, "
                f"vocab={self.vocab_size}, intermediate={self.intermediate_size})")
