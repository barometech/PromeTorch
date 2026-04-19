"""
promethorch.vision.transforms — torchvision-compatible image transforms.

Compose([...]) wraps a list of callables. Transforms are callable objects
taking a Tensor (uint8 HWC or float CHW) and returning a Tensor.
"""

from __future__ import annotations
from typing import Callable, List

try:
    from promethorch._C import vision as _cpp_vision
    _tm = _cpp_vision.transforms
    _HAS_CPP = True
except (ImportError, AttributeError):
    _tm = None
    _HAS_CPP = False


if _HAS_CPP:
    Compose  = _tm.Compose
    ToTensor = _tm.ToTensor
    Transform = _tm.Transform
else:
    class Transform:
        def __call__(self, x): return x

    class Compose(Transform):
        def __init__(self, transforms=None):
            self.transforms = list(transforms) if transforms else []

        def push_back(self, t):
            self.transforms.append(t)

        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class ToTensor(Transform):
        def __call__(self, x):
            try:
                import numpy as np
                if hasattr(x, "numpy"):
                    arr = x.numpy()
                else:
                    arr = np.asarray(x)
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
                if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                    arr = arr.transpose(2, 0, 1)   # HWC -> CHW
                import promethorch as _pt
                return _pt.from_numpy(arr) if hasattr(_pt, "from_numpy") else arr
            except Exception:
                return x


class _CallableTransform(Transform):
    def __init__(self, fn, name="lambda"):
        self._fn = fn
        self._name = name

    def __call__(self, x):
        return self._fn(x)

    def __repr__(self):
        return f"Transform({self._name})"


def Lambda(fn: Callable) -> Transform:  # noqa: N802 (match torchvision name)
    return _CallableTransform(fn, getattr(fn, "__name__", "lambda"))


__all__ = ["Compose", "ToTensor", "Transform", "Lambda"]
