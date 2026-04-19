"""
promethorch.mobile — ExecuTorch-like compact on-device inference format.
"""

from __future__ import annotations


def export(model, example_input, path: str) -> bool:
    """Serialize `model` (must be a Sequential) to an on-device binary."""
    try:
        from promethorch._C import mobile_export as _cpp
        return bool(_cpp.export(model, example_input, path))
    except (ImportError, AttributeError):
        raise RuntimeError(
            "promethorch.mobile.export requires _C to be built with "
            "bindings_new.cpp (torch::mobile)."
        )


class MobileExecutor:
    """Loads and runs a compiled mobile model."""
    def __init__(self, path: str = None):
        try:
            from promethorch._C import mobile_export as _cpp
            self._impl = _cpp.MobileExecutor()
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                "MobileExecutor requires _C built with bindings_new.cpp"
            ) from e
        if path is not None:
            self.load(path)

    def load(self, path: str):
        self._impl.load(path)

    def forward(self, x):
        return self._impl.forward(x)

    __call__ = forward


__all__ = ["export", "MobileExecutor"]
