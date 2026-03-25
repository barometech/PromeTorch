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
    compile as _compile_raw,
    CompiledModule,

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

# Import submodules
from . import nn
from . import optim
from . import data

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

def compile(model, **kwargs):
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
    return _compile_raw(model, **kwargs)


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
    'CompiledModule',

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

    # CUDA
    'cuda',

    # AMP
    'amp',

    # Submodules
    'nn',
    'optim',
    'data',

    # Utils
    'manual_seed',
    '__version__',
]
