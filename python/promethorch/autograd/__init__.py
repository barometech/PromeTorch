"""
promethorch.autograd — forward-mode AD (jvp) + vmap.

This submodule extends the base autograd (backward pass lives on tensors).
"""

from __future__ import annotations
from typing import Callable, Tuple

try:
    from promethorch._C import autograd_fwd as _cpp
    _HAS_CPP = True
except (ImportError, AttributeError):
    _cpp = None
    _HAS_CPP = False


if _HAS_CPP:
    DualLevel    = _cpp.DualLevel
    make_dual    = _cpp.make_dual
    unpack_dual  = _cpp.unpack_dual
    jvp          = _cpp.jvp
    vmap         = _cpp.vmap
else:
    class DualLevel:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def make_dual(primal, tangent):
        return primal  # no-op if bindings absent

    def unpack_dual(t):
        return (t, None)

    def jvp(f, primal, tangent):
        """Finite-difference fallback: (f(x+eps*v) - f(x))/eps."""
        eps = 1e-5
        y0 = f(primal)
        try:
            y1 = f(primal + tangent * eps)
            tangent_out = (y1 - y0) * (1.0 / eps)
        except Exception:
            tangent_out = None
        return (y0, tangent_out)

    def vmap(f, input, in_dim: int = 0, out_dim: int = 0):
        """Fallback: slice the input along in_dim, call f, stack results."""
        try:
            import promethorch as _pt
        except ImportError:
            raise RuntimeError("vmap fallback needs promethorch")
        n = input.size(in_dim) if hasattr(input, "size") else len(input)
        # Slice along in_dim
        results = []
        for i in range(n):
            if hasattr(input, "narrow"):
                sl = input.narrow(in_dim, i, 1)
                # Squeeze that dim
                try:
                    sl = sl.squeeze(in_dim)
                except Exception:
                    pass
            else:
                sl = input[i]
            results.append(f(sl))
        try:
            return _pt.stack(results, dim=out_dim)
        except Exception:
            return results


__all__ = ["DualLevel", "make_dual", "unpack_dual", "jvp", "vmap"]
