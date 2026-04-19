"""
Standalone reader for the safetensors file format.

Layout (https://github.com/huggingface/safetensors):
    [8 bytes little-endian uint64 N]   header length
    [N bytes JSON header]              metadata + per-tensor descriptors
    [raw tensor bytes]                 tightly packed, row-major (C order)

Header JSON example::

    {
        "__metadata__": {"format": "pt"},
        "weight": {
            "dtype": "F32",
            "shape": [768, 768],
            "data_offsets": [0, 2359296]
        },
        ...
    }

This module returns a dict ``{name: numpy.ndarray}`` so callers can convert to
PromeTorch tensors via ``promethorch.from_numpy``. It also provides a lazy
``SafeTensorsFile`` class that mmaps the file and reads tensors on demand —
useful for large checkpoints where loading every tensor at once would OOM.

No external dependencies beyond ``numpy`` and the standard library.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Dict, Iterator, Optional, Tuple

import numpy as np


# Mapping from safetensors dtype string -> (numpy dtype, item size in bytes).
# bf16/f8 do not have native numpy dtypes; we expose them as raw bytes the
# caller can reinterpret. For most HF checkpoints F32/F16/BF16 cover 99%.
_SAFETENSORS_DTYPES: Dict[str, Tuple[Optional[np.dtype], int]] = {
    "F64": (np.dtype("float64"), 8),
    "F32": (np.dtype("float32"), 4),
    "F16": (np.dtype("float16"), 2),
    "BF16": (None, 2),  # numpy has no native bfloat16
    "I64": (np.dtype("int64"), 8),
    "I32": (np.dtype("int32"), 4),
    "I16": (np.dtype("int16"), 2),
    "I8":  (np.dtype("int8"),  1),
    "U8":  (np.dtype("uint8"), 1),
    "BOOL": (np.dtype("bool"), 1),
}


def _bf16_to_f32(raw: bytes, count: int) -> np.ndarray:
    """Convert raw bfloat16 bytes (little-endian, 2 bytes each) to float32.

    bfloat16 = top 16 bits of float32. Pad with two zero bytes on the low end.
    """
    src = np.frombuffer(raw, dtype=np.uint16, count=count)
    high = src.astype(np.uint32) << 16
    return high.view(np.float32).copy()


class SafeTensorsFile:
    """Memory-mapped safetensors reader. Use as a context manager."""

    def __init__(self, path: str):
        self.path = path
        self._fp = open(path, "rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        # Header
        (header_len,) = struct.unpack("<Q", self._mm[:8])
        self._header_len = header_len
        self._data_start = 8 + header_len
        header_json = self._mm[8:8 + header_len].decode("utf-8")
        self._header = json.loads(header_json)
        self.metadata: Dict[str, str] = self._header.get("__metadata__", {}) or {}

    # context manager
    def __enter__(self) -> "SafeTensorsFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fp.close()
        except Exception:
            pass

    # introspection
    def keys(self) -> Iterator[str]:
        for k in self._header.keys():
            if k != "__metadata__":
                yield k

    def __contains__(self, name: str) -> bool:
        return name in self._header and name != "__metadata__"

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    def info(self, name: str) -> dict:
        return self._header[name]

    # read
    def get_tensor(self, name: str) -> np.ndarray:
        if name not in self._header or name == "__metadata__":
            raise KeyError(name)
        info = self._header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        offs = info["data_offsets"]
        start = self._data_start + offs[0]
        end = self._data_start + offs[1]
        raw = self._mm[start:end]
        if dtype_str not in _SAFETENSORS_DTYPES:
            raise NotImplementedError(f"safetensors dtype {dtype_str} unsupported")
        np_dtype, itemsize = _SAFETENSORS_DTYPES[dtype_str]
        n_elem = 1
        for d in shape:
            n_elem *= d
        if dtype_str == "BF16":
            arr = _bf16_to_f32(bytes(raw), n_elem)
            return arr.reshape(shape) if shape else arr.reshape(())
        # Copy into a writable buffer (mmap slice is read-only)
        arr = np.frombuffer(bytes(raw), dtype=np_dtype, count=n_elem)
        if shape:
            arr = arr.reshape(shape)
        else:
            arr = arr.reshape(())
        return arr.copy()


def load_file(path: str) -> Dict[str, np.ndarray]:
    """Eagerly load every tensor from a single safetensors file."""
    out: Dict[str, np.ndarray] = {}
    with SafeTensorsFile(path) as f:
        for name in f.keys():
            out[name] = f.get_tensor(name)
    return out


def load_index(index_json_path: str) -> Dict[str, np.ndarray]:
    """Load a sharded safetensors checkpoint via its model.safetensors.index.json.

    The index file maps every parameter name to the shard filename that holds
    it. We walk all referenced shards and merge their tensors into one dict.
    """
    with open(index_json_path, "r", encoding="utf-8") as fp:
        index = json.load(fp)
    weight_map = index["weight_map"]
    base_dir = os.path.dirname(os.path.abspath(index_json_path))
    out: Dict[str, np.ndarray] = {}
    # Group keys by shard so each file is only opened once
    shards: Dict[str, list] = {}
    for k, fname in weight_map.items():
        shards.setdefault(fname, []).append(k)
    for fname, names in shards.items():
        path = os.path.join(base_dir, fname)
        with SafeTensorsFile(path) as f:
            for name in names:
                out[name] = f.get_tensor(name)
    return out


__all__ = ["SafeTensorsFile", "load_file", "load_index"]
