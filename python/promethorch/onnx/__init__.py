"""
promethorch.onnx — Export models to ONNX.

Supports a subset of nn modules (Linear, Conv2d, activations, pooling,
BatchNorm, Flatten, Softmax, Sequential). See torch/onnx/export.h.
"""

from __future__ import annotations
from typing import Optional


def export(model, example_input, path: str,
           input_name: str = "input", output_name: str = "output") -> bool:
    """Export `model` to an ONNX file at `path`. Returns True on success."""
    try:
        from promethorch._C import onnx_export as _cpp
        return bool(_cpp.export(model, example_input, path, input_name, output_name))
    except (ImportError, AttributeError):
        raise RuntimeError(
            "promethorch.onnx.export requires _C to be built with "
            "bindings_new.cpp (torch::onnx)."
        )


def self_test(tmp_path: str = "/tmp/test.onnx") -> bool:
    try:
        from promethorch._C import onnx_export as _cpp
        return bool(_cpp.self_test(tmp_path))
    except (ImportError, AttributeError):
        return False


__all__ = ["export", "self_test"]
