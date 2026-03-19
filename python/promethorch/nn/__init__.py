"""
PromeTorch Neural Network Module
"""

from .._C.nn import (
    # Base
    Module,

    # Containers
    Sequential,
    ModuleList,

    # Linear
    Linear,

    # Activations
    ReLU,
    ReLU6,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    Mish,
    Softmax,
    LogSoftmax,
    Softplus,
    Softsign,
    Hardtanh,
    Hardsigmoid,
    Hardswish,

    # Normalization
    BatchNorm2d,
    LayerNorm,

    # Dropout
    Dropout,

    # Convolution
    Conv2d,

    # Pooling
    MaxPool2d,
    AvgPool2d,

    # Sparse
    Embedding,

    # Recurrent
    RNN,
    LSTM,
    GRU,

    # Loss
    MSELoss,
    CrossEntropyLoss,
    NLLLoss,
    BCELoss,
    L1Loss,

    # Reduction enum
    Reduction,

    # Submodules
    functional,
    utils,
)

# Alias for functional
import promethorch.nn.functional as F

__all__ = [
    'Module',
    'Sequential',
    'ModuleList',
    'Linear',
    'ReLU',
    'ReLU6',
    'LeakyReLU',
    'PReLU',
    'ELU',
    'SELU',
    'Sigmoid',
    'Tanh',
    'GELU',
    'SiLU',
    'Mish',
    'Softmax',
    'LogSoftmax',
    'Softplus',
    'Softsign',
    'Hardtanh',
    'Hardsigmoid',
    'Hardswish',
    'BatchNorm2d',
    'LayerNorm',
    'Dropout',
    'Conv2d',
    'MaxPool2d',
    'AvgPool2d',
    'Embedding',
    'RNN',
    'LSTM',
    'GRU',
    'MSELoss',
    'CrossEntropyLoss',
    'NLLLoss',
    'BCELoss',
    'L1Loss',
    'Reduction',
    'functional',
    'utils',
    'F',
]
