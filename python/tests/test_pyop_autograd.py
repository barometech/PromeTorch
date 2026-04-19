"""
Tests for Python operator bindings (__add__ / __sub__ / __mul__ / __truediv__
and their reflected + scalar variants, plus __neg__) routing through the
*_autograd wrappers in torch/csrc/autograd/autograd.h.

Before the fix, `py::self + py::self`-style bindings delegated to
`at::Tensor::operator+`, which calls raw `at::add()` and bypasses autograd —
so `(a + b).requires_grad` was False even when `a.requires_grad == True`.

After the fix each Python operator is an explicit lambda that calls the
corresponding `*_autograd` wrapper, attaching a backward node via
`compute_requires_grad()`. Inference is unaffected (no-grad tensors
short-circuit in the wrapper).

Runs under pytest OR as a standalone script:
    python -m pytest python/tests/test_pyop_autograd.py -q
    python python/tests/test_pyop_autograd.py
"""

from __future__ import annotations
import sys
import os

# Make sure python/ is on sys.path when invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import promethorch as pt


SHAPE = [3, 3]


def _make_leaf(requires_grad: bool = True):
    x = pt.randn(SHAPE, requires_grad=requires_grad)
    if requires_grad and not x.requires_grad:
        setter = getattr(x, "requires_grad_", None)
        if callable(setter):
            setter(True)
        else:
            x.requires_grad = True
    assert x.requires_grad is requires_grad, (
        f"leaf requires_grad setup failed (expected {requires_grad}, got {x.requires_grad})"
    )
    return x


# ----------------------------------------------------------------------------
# 1. Tensor-tensor ops propagate requires_grad
# ----------------------------------------------------------------------------

def test_tensor_tensor_add_propagates():
    a = _make_leaf(requires_grad=True)
    b = _make_leaf(requires_grad=False)
    assert (a + b).requires_grad is True
    # Reverse operand order: Python dispatches to b.__add__(a) first — since
    # `b` is a Tensor, it returns a result; for this op the result still has
    # a grad path because add_autograd's compute_requires_grad(self, other)
    # checks both.
    assert (b + a).requires_grad is True


def test_tensor_tensor_sub_propagates():
    a = _make_leaf(requires_grad=True)
    b = _make_leaf(requires_grad=False)
    assert (a - b).requires_grad is True
    assert (b - a).requires_grad is True


def test_tensor_tensor_mul_propagates():
    a = _make_leaf(requires_grad=True)
    b = _make_leaf(requires_grad=False)
    assert (a * b).requires_grad is True
    assert (b * a).requires_grad is True


def test_tensor_tensor_div_propagates():
    a = _make_leaf(requires_grad=True)
    b = _make_leaf(requires_grad=False)
    assert (a / b).requires_grad is True
    assert (b / a).requires_grad is True


# ----------------------------------------------------------------------------
# 2. All-detached inputs → no autograd graph (short-circuit path)
# ----------------------------------------------------------------------------

def test_detached_inputs_no_graph():
    a = _make_leaf(requires_grad=False)
    b = _make_leaf(requires_grad=False)
    assert (a + b).requires_grad is False
    assert (a * b).requires_grad is False
    assert (a - b).requires_grad is False
    assert (a / b).requires_grad is False


# ----------------------------------------------------------------------------
# 3. Tensor-scalar ops (t op s)
# ----------------------------------------------------------------------------

def test_tensor_scalar_propagates():
    a = _make_leaf(requires_grad=True)
    assert (a + 2.0).requires_grad is True
    assert (a - 2.0).requires_grad is True
    assert (a * 2.0).requires_grad is True
    assert (a / 2.0).requires_grad is True


def test_tensor_scalar_detached():
    a = _make_leaf(requires_grad=False)
    assert (a + 2.0).requires_grad is False
    assert (a * 2.0).requires_grad is False


# ----------------------------------------------------------------------------
# 4. Scalar-tensor ops (s op t)  — radd / rsub / rmul / rtruediv
# ----------------------------------------------------------------------------

def test_scalar_tensor_propagates():
    a = _make_leaf(requires_grad=True)
    assert (2.0 + a).requires_grad is True    # __radd__
    assert (2.0 - a).requires_grad is True    # __rsub__  — handled by rsub_scalar_autograd
    assert (2.0 * a).requires_grad is True    # __rmul__
    assert (2.0 / a).requires_grad is True    # __rtruediv__ — rdiv_scalar_autograd


# ----------------------------------------------------------------------------
# 5. Unary negation
# ----------------------------------------------------------------------------

def test_neg_propagates():
    a = _make_leaf(requires_grad=True)
    assert (-a).requires_grad is True
    b = _make_leaf(requires_grad=False)
    assert (-b).requires_grad is False


# ----------------------------------------------------------------------------
# 6. Backward flows: call .backward(gradient=ones) on result of an op, verify
#    a.grad is defined and has the right shape.
# ----------------------------------------------------------------------------

def _assert_grad_like(a, expected_shape):
    g = a.grad
    assert g is not None, ".grad is None after backward"
    # Some bindings return an undefined Tensor sentinel — probe defensively.
    try:
        if not g.defined:
            raise AssertionError(".grad is undefined after backward")
    except AttributeError:
        pass
    shape = tuple(g.shape) if hasattr(g, "shape") else tuple(g.size())
    assert tuple(shape) == tuple(expected_shape), (
        f".grad has shape {shape}, expected {tuple(expected_shape)}"
    )


def test_backward_tensor_tensor_add():
    a = _make_leaf(requires_grad=True)
    b = _make_leaf(requires_grad=False)
    y = a + b
    assert y.requires_grad is True
    grad_out = pt.ones_like(y)
    y.backward(gradient=grad_out)
    _assert_grad_like(a, SHAPE)


def test_backward_tensor_scalar_add():
    a = _make_leaf(requires_grad=True)
    y = a + 2.0
    assert y.requires_grad is True
    y.backward(gradient=pt.ones_like(y))
    _assert_grad_like(a, SHAPE)


def test_backward_scalar_tensor_rsub():
    a = _make_leaf(requires_grad=True)
    y = 2.0 - a
    assert y.requires_grad is True
    y.backward(gradient=pt.ones_like(y))
    _assert_grad_like(a, SHAPE)


def test_backward_scalar_tensor_rmul():
    a = _make_leaf(requires_grad=True)
    y = 3.0 * a
    assert y.requires_grad is True
    y.backward(gradient=pt.ones_like(y))
    _assert_grad_like(a, SHAPE)


def test_backward_neg():
    a = _make_leaf(requires_grad=True)
    y = -a
    assert y.requires_grad is True
    y.backward(gradient=pt.ones_like(y))
    _assert_grad_like(a, SHAPE)


# ----------------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------------

def _collect_tests():
    return [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]


def main():
    tests = _collect_tests()
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc!r}")
            failures.append((name, exc))
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
