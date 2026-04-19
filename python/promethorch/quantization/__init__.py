"""
promethorch.quantization — Quantization-Aware Training (QAT) + INT8 helpers.
"""

from __future__ import annotations

try:
    from promethorch._C import quantization as _cpp
    _HAS_CPP = True
except (ImportError, AttributeError):
    _cpp = None
    _HAS_CPP = False


if _HAS_CPP:
    QuantizedLinear = _cpp.QuantizedLinear
    fake_quantize   = _cpp.fake_quantize
    prepare_qat     = _cpp.prepare_qat
    convert         = _cpp.convert
else:
    # Pure-Python fallbacks: no-op prepare/convert, identity fake_quantize.

    def prepare_qat(model):
        return model

    def convert(model):
        return model

    def fake_quantize(input, scale, zero_point=0, qmin=-128, qmax=127):
        return input

    class QuantizedLinear:
        def __init__(self, in_features, out_features, bias=True):
            raise RuntimeError(
                "QuantizedLinear requires _C built with bindings_new.cpp"
            )


__all__ = ["QuantizedLinear", "fake_quantize", "prepare_qat", "convert"]
