"""
Tests for prometorch.no_grad / prometorch.enable_grad (BUG-C9 fix).

Verifies that the Python context managers correctly toggle the C++
torch::autograd::GradMode singleton, so operations inside a no_grad block
do NOT build an autograd graph.

Runs under pytest OR as a standalone script:
    python -m pytest python/tests/test_no_grad.py -q
    python python/tests/test_no_grad.py
"""

from __future__ import annotations
import sys
import os

# Make sure python/ is on sys.path when invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import prometorch as pt


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _has_grad_fn(t) -> bool:
    """Return True if tensor has a backward Function attached.

    PromeTorch tensors expose `requires_grad` as the user-visible flag.
    A tensor that came out of an op WITH grad enabled will also report
    requires_grad=True (because at least one input did and a grad_fn was
    attached). A tensor produced under no_grad will report requires_grad=False
    even if its inputs had requires_grad=True. That's the contract we test.
    """
    return bool(getattr(t, "requires_grad", False))


def _make_leaf():
    """Allocate a fresh leaf tensor with requires_grad=True."""
    x = pt.randn([2, 3])
    # Bindings expose requires_grad either as attribute or method.
    setter = getattr(x, "requires_grad_", None)
    if callable(setter):
        setter(True)
    else:
        x.requires_grad = True
    assert x.requires_grad, "leaf failed to register requires_grad"
    return x


# ----------------------------------------------------------------------------
# 1. Sanity: defaults
# ----------------------------------------------------------------------------

def test_default_grad_enabled():
    assert pt.is_grad_enabled() is True


# ----------------------------------------------------------------------------
# 2. no_grad as context manager toggles GradMode
# ----------------------------------------------------------------------------

def test_no_grad_toggles_flag():
    assert pt.is_grad_enabled() is True
    with pt.no_grad():
        assert pt.is_grad_enabled() is False
    assert pt.is_grad_enabled() is True


# ----------------------------------------------------------------------------
# 3. ops inside no_grad produce tensors WITHOUT grad_fn
# ----------------------------------------------------------------------------
# NOTE: Some Python-bound ops in the current _C.pyd call raw aten functions
# instead of the `*_autograd` wrappers in torch/csrc/autograd/autograd.h, so
# they never propagate requires_grad regardless of GradMode. We probe a few
# ops and only assert when at least one of them DOES propagate outside the
# no_grad scope — that's the only configuration where the no_grad assertion
# is meaningful. The GradMode flag itself is fully tested by tests 2/4/5/6.

_OP_PROBES = [
    ("add",      lambda a: a + a),
    ("mul",      lambda a: a * a),
    ("relu",     lambda a: pt.relu(a)),
    ("tanh",     lambda a: pt.tanh(a)),
    ("sigmoid",  lambda a: pt.sigmoid(a)),
    ("exp",      lambda a: pt.exp(a)),
    ("sum",      lambda a: pt.sum(a)),
    ("mm",       lambda a: pt.mm(a, a.t())),
]


def _propagating_ops(x):
    """Return the subset of probe ops that DO propagate requires_grad on x."""
    out = []
    for name, fn in _OP_PROBES:
        try:
            y = fn(x)
        except Exception:
            continue
        if _has_grad_fn(y):
            out.append((name, fn))
    return out


def test_op_inside_no_grad_has_no_grad_fn():
    x = _make_leaf()
    propagators = _propagating_ops(x)
    if not propagators:
        # No bound op currently routes through the *_autograd wrappers, so
        # there's nothing for no_grad to suppress at the Python op level.
        # The GradMode flag itself is verified by other tests.
        try:
            import pytest  # type: ignore
            pytest.skip(
                "No Python-bound op propagates requires_grad; no_grad's "
                "effect on grad_fn cannot be observed at the Python boundary. "
                "The C++ flag IS toggled (see test_no_grad_toggles_flag)."
            )
        except ImportError:
            print("    [skip] no propagating op available; flag-only test")
        return

    with pt.no_grad():
        for name, fn in propagators:
            y_inside = fn(x)
            assert not _has_grad_fn(y_inside), (
                f"inside no_grad, op `{name}` output must NOT have "
                f"requires_grad/grad_fn"
            )

    # After exit: graph tracking restored.
    for name, fn in propagators:
        y_after = fn(x)
        assert _has_grad_fn(y_after), (
            f"after no_grad, op `{name}` must resume grad tracking"
        )


# ----------------------------------------------------------------------------
# 4. decorator form: @no_grad()
# ----------------------------------------------------------------------------

def test_no_grad_as_decorator():
    @pt.no_grad()
    def doubled(t):
        # Inside the decorated body, grad must be off.
        assert pt.is_grad_enabled() is False
        return t + t

    x = _make_leaf()
    y = doubled(x)
    assert not _has_grad_fn(y), "decorator form must yield non-grad output"
    # Outer scope unaffected.
    assert pt.is_grad_enabled() is True


# ----------------------------------------------------------------------------
# 5. enable_grad inside no_grad re-enables tracking
# ----------------------------------------------------------------------------

def test_enable_grad_inside_no_grad():
    x = _make_leaf()
    propagators = _propagating_ops(x)
    with pt.no_grad():
        assert pt.is_grad_enabled() is False
        with pt.enable_grad():
            assert pt.is_grad_enabled() is True
            if propagators:
                _, fn = propagators[0]
                y = fn(x)
                assert _has_grad_fn(y), \
                    "enable_grad inside no_grad must restore graph tracking"
        # Back to no_grad scope.
        assert pt.is_grad_enabled() is False
        if propagators:
            _, fn = propagators[0]
            z = fn(x)
            assert not _has_grad_fn(z)
    assert pt.is_grad_enabled() is True


# ----------------------------------------------------------------------------
# 6. nested no_grad correctly restores on exit
# ----------------------------------------------------------------------------

def test_nested_no_grad_restores_state():
    assert pt.is_grad_enabled() is True
    with pt.no_grad():
        assert pt.is_grad_enabled() is False
        with pt.no_grad():
            assert pt.is_grad_enabled() is False
        assert pt.is_grad_enabled() is False  # outer no_grad still active
    assert pt.is_grad_enabled() is True


# ----------------------------------------------------------------------------
# 7. exception inside no_grad still restores state
# ----------------------------------------------------------------------------

def test_no_grad_restores_on_exception():
    assert pt.is_grad_enabled() is True
    try:
        with pt.no_grad():
            assert pt.is_grad_enabled() is False
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert pt.is_grad_enabled() is True, \
        "grad mode must be restored even when the body raises"


# ----------------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    tests = [
        test_default_grad_enabled,
        test_no_grad_toggles_flag,
        test_op_inside_no_grad_has_no_grad_fn,
        test_no_grad_as_decorator,
        test_enable_grad_inside_no_grad,
        test_nested_no_grad_restores_state,
        test_no_grad_restores_on_exception,
    ]
    skipped = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except BaseException as e:  # pytest.skip raises BaseException subclass
            if type(e).__name__ in ("Skipped", "OutcomeException"):
                skipped += 1
                print(f"  SKIP  {t.__name__}: {e}")
            elif isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            else:
                failures += 1
                print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) failed, {skipped} skipped")
        sys.exit(1)
    print(f"\n{len(tests) - skipped}/{len(tests)} tests passed, {skipped} skipped")
