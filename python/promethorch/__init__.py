"""
PromeTorch: A PyTorch-like Deep Learning Framework
"""

from ._C import (
    # Types
    Tensor,
    device,
    DeviceType,
    dtype,
    TensorOptions,

    # Factory functions
    tensor,
    empty,
    zeros,
    ones,
    full,
    rand,
    randn,
    randint,
    arange,
    linspace,
    eye,

    # Operations
    cat,
    stack,
    split,
    chunk,
    mm,
    bmm,
    matmul,
    dot,
    sum,
    mean,
    max,
    min,
    sqrt,
    exp,
    log,
    sin,
    cos,
    tanh,
    sigmoid,
    relu,
    softmax,
    clamp,
    where,

    # New operations for PIR support
    cumsum,
    rsqrt,
    norm,
    topk,
    sort,
    zeros_like,
    ones_like,
    from_numpy,
    nan_to_num,
    isinf,
    isnan,
    isfinite,
    multinomial,
    einsum,
    # Autograd
    no_grad as _CppNoGrad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
    backward,
    grad,

    # Serialization
    save,
    load,
    save_state_dict,
    load_state_dict,

    # CUDA
    cuda_is_available,
    cuda_device_count,

    # Version
    __version__,
)

# PyTorch-compatible .pt/.pth (ZIP+pickle) save/load — optional.
try:
    from ._pytorch_io import save_pytorch, load_pytorch
except ImportError:
    def save_pytorch(*_a, **_k):
        raise RuntimeError("save_pytorch: _C extension lacks this binding; rebuild.")
    def load_pytorch(*_a, **_k):
        raise RuntimeError("load_pytorch: _C extension lacks this binding; rebuild.")

# Import submodules. These may fail on stale builds where the C extension is
# missing newer symbols (e.g. BatchNorm1d). We swallow the error so that
# core ops and helpers like ``promethorch.transformers_compat`` remain
# usable; ``promethorch.nn`` will simply be unavailable until the C
# extension is rebuilt.
try:
    from . import nn
except ImportError as _e:
    import warnings as _w
    _w.warn(f"promethorch.nn unavailable: {_e}. Rebuild the _C extension.")
try:
    from . import optim
except ImportError as _e:
    import warnings as _w
    _w.warn(f"promethorch.optim unavailable: {_e}. Rebuild the _C extension.")
try:
    from . import data
except ImportError as _e:
    import warnings as _w
    _w.warn(f"promethorch.data unavailable: {_e}. Rebuild the _C extension.")

# ----------------------------------------------------------------------------
# New submodules: lazy imports so a pre-existing _C.pyd without
# bindings_new.cpp still lets `import promethorch` succeed. Each submodule
# provides its own Python-level fallback for the missing C++ symbols.
# ----------------------------------------------------------------------------
def _lazy_import(_name: str):
    import importlib as _il
    try:
        return _il.import_module(f"promethorch.{_name}")
    except Exception as _e:
        import warnings as _w
        _w.warn(f"promethorch.{_name} unavailable: {_e}")
        return None

# Eagerly attach each as an attribute so `pt.nn.parallel` etc. work without
# the caller doing a separate `import`.
for _sub in ("distributed", "trainer", "onnx", "mlir", "mobile",
             "jit", "vision", "quantization", "serve"):
    _mod = _lazy_import(_sub)
    if _mod is not None:
        globals()[_sub] = _mod

# nn.parallel attaches to the nn submodule. On stale _C builds where
# promethorch.nn's own __init__.py fails, create a lightweight placeholder
# module so pt.nn.parallel remains accessible.
try:
    import importlib as _il
    import types as _types
    import sys as _sys
    if "nn" not in globals() or globals().get("nn") is None:
        _nn_stub = _types.ModuleType("promethorch.nn")
        _nn_stub.__path__ = [__import__('os').path.join(
            __import__('os').path.dirname(__file__), "nn")]
        _sys.modules.setdefault("promethorch.nn", _nn_stub)
        globals()["nn"] = _nn_stub
    try:
        _parallel = _il.import_module("promethorch.nn.parallel")
        setattr(globals()["nn"], "parallel", _parallel)
    except Exception as _pe:
        import warnings as _w
        _w.warn(f"promethorch.nn.parallel unavailable: {_pe}")
except Exception as _e:
    import warnings as _w
    _w.warn(f"promethorch.nn.parallel setup failed: {_e}")

# Replace the base `autograd` namespace with the new submodule that offers
# jvp / vmap / forward-mode AD alongside the existing backward helpers.
try:
    import importlib as _il
    autograd = _il.import_module("promethorch.autograd")
    # Re-expose low-level backward/grad for convenience.
    from ._C import backward as _cpp_backward, grad as _cpp_grad  # noqa: F401
    setattr(autograd, "backward", _cpp_backward)
    setattr(autograd, "grad", _cpp_grad)
except Exception as _e:
    import warnings as _w
    _w.warn(f"promethorch.autograd extended API unavailable: {_e}")

# Convenience
cuda = type('cuda', (), {
    'is_available': staticmethod(cuda_is_available),
    'device_count': staticmethod(cuda_device_count),
})()

class no_grad:
    """Context manager and decorator that disables gradient computation."""
    def __enter__(self):
        self._guard = _CppNoGrad()
        self._guard.__enter__()
        return self

    def __exit__(self, *args):
        self._guard.__exit__(*args)

    def __call__(self, func):
        """Support usage as @torch.no_grad() decorator."""
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

def compile(model, **kwargs):  # no-op if _C doesn't support it
    """Compile a model for fast inference using PromePile JIT.

    Traces the model's forward pass on first call and then executes via
    pre-allocated buffers with fused ops. No autograd overhead.

    Supported layers: Linear, ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax,
    BatchNorm1d, Sequential containers.

    Usage::

        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )
        compiled = torch.compile(model)
        output = compiled(input_tensor)  # first call traces, subsequent calls are fast

    Args:
        model: An nn.Module to compile.
        **kwargs: Optional compilation hints (currently unused, reserved for future).

    Returns:
        A CompiledModule wrapper. Calling it with a tensor traces on the first
        invocation and then executes the optimized graph on subsequent calls
        with the same input shape.
    """
    try:
        from ._C import compile as _compile_raw
        return _compile_raw(model, **kwargs)
    except ImportError:
        return model  # no-op fallback


def manual_seed(seed: int):
    """Set the random seed for reproducibility."""
    import random
    random.seed(seed)


# ============================================================================
# AMP (Automatic Mixed Precision) — no-op shims for PIR compatibility
# ============================================================================

class _AutocastContext:
    """No-op autocast context manager"""
    def __init__(self, device_type='cuda', dtype=None, enabled=True):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class _GradScaler:
    """No-op GradScaler for compatibility"""
    def __init__(self, device='cuda', enabled=True):
        self._enabled = enabled
        self._scale = 1.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return {'scale': self._scale}

    def load_state_dict(self, state_dict):
        self._scale = state_dict.get('scale', 1.0)

# Create amp submodule
class _AmpModule:
    autocast = _AutocastContext
    GradScaler = _GradScaler

amp = _AmpModule()


__all__ = [
    # Core
    'Tensor',
    'device',
    'DeviceType',
    'dtype',
    'TensorOptions',

    # Factory
    'tensor',
    'empty',
    'zeros',
    'ones',
    'full',
    'rand',
    'randn',
    'randint',
    'arange',
    'linspace',
    'eye',

    # Ops
    'cat',
    'stack',
    'split',
    'chunk',
    'mm',
    'bmm',
    'matmul',
    'dot',
    'sum',
    'mean',
    'max',
    'min',
    'sqrt',
    'exp',
    'log',
    'sin',
    'cos',
    'tanh',
    'sigmoid',
    'relu',
    'softmax',
    'clamp',
    'where',

    # New ops
    'cumsum',
    'rsqrt',
    'norm',
    'topk',
    'sort',
    'zeros_like',
    'ones_like',
    'from_numpy',
    'nan_to_num',
    'isinf',
    'isnan',
    'isfinite',
    'multinomial',
    'einsum',
    'compile',
    # 'CompiledModule',  # requires rebuilt _C.so

    # Autograd
    'no_grad',
    'enable_grad',
    'is_grad_enabled',
    'set_grad_enabled',
    'backward',
    'grad',

    # Serialization
    'save',
    'load',
    'save_state_dict',
    'load_state_dict',
    'save_pytorch',
    'load_pytorch',

    # CUDA
    'cuda',

    # AMP
    'amp',

    # Submodules
    'nn',
    'optim',
    'data',
    'distributed',
    'trainer',
    'onnx',
    'mlir',
    'mobile',
    'jit',
    'vision',
    'quantization',
    'autograd',
    'serve',

    # Utils
    'manual_seed',
    '__version__',
]
