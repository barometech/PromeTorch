"""
promethorch.jit — Tracing JIT (prototype).

`compile(fn, example_input)` returns a callable. If the C++ binding is
unavailable the identity wrapper is returned (fn(...) == compile(fn)(...)),
so user code can always call `pt.jit.compile(...)`.
"""

from __future__ import annotations
from typing import Callable, Optional


def compile(fn: Callable, example_input=None) -> Callable:
    """Return a compiled version of `fn`. Falls back to `fn` if unsupported."""
    try:
        from promethorch._C import jit_compile as _cpp
        return _cpp.compile(fn, example_input)
    except (ImportError, AttributeError):
        return fn


class ScriptModule:
    """Placeholder ScriptModule that simply wraps a callable."""
    def __init__(self, fn: Callable):
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def script(fn: Callable) -> ScriptModule:
    return ScriptModule(fn)


def trace(fn: Callable, example_input=None) -> Callable:
    return compile(fn, example_input)


__all__ = ["compile", "trace", "script", "ScriptModule"]
