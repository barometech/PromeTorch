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

    # Autograd
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
    backward,
    grad,

    # CUDA
    cuda_is_available,
    cuda_device_count,

    # Version
    __version__,
)

# Import submodules
from . import nn
from . import optim

# Convenience
cuda = type('cuda', (), {
    'is_available': staticmethod(cuda_is_available),
    'device_count': staticmethod(cuda_device_count),
})()

def manual_seed(seed: int):
    """Set the random seed for reproducibility."""
    import random
    random.seed(seed)

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

    # Autograd
    'no_grad',
    'enable_grad',
    'is_grad_enabled',
    'set_grad_enabled',
    'backward',
    'grad',

    # CUDA
    'cuda',

    # Submodules
    'nn',
    'optim',

    # Utils
    'manual_seed',
    '__version__',
]
