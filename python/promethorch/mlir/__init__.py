"""
promethorch.mlir — Export models to MLIR text (linalg/tosa/arith dialects).
"""

from __future__ import annotations
from typing import List


def export(model, input_shape: List[int], path: str) -> bool:
    """Export `model` to an MLIR text file at `path`."""
    try:
        from promethorch._C import mlir_export as _cpp
        return bool(_cpp.export(model, list(input_shape), path))
    except (ImportError, AttributeError):
        raise RuntimeError(
            "promethorch.mlir.export requires _C to be built with "
            "bindings_new.cpp (torch::mlir)."
        )


__all__ = ["export"]
