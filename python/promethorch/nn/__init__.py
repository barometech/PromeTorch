"""
PromeTorch Neural Network Module
"""

from .._C.nn import (
    # Base
    Module,

    # Linear
    Linear,

    # Activations
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    Softmax,

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
)

# Alias for functional
import promethorch.nn.functional as F

__all__ = [
    'Module',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'GELU',
    'SiLU',
    'Softmax',
    'BatchNorm2d',
    'LayerNorm',
    'Dropout',
    'Conv2d',
    'MaxPool2d',
    'AvgPool2d',
    'Embedding',
    'MSELoss',
    'CrossEntropyLoss',
    'NLLLoss',
    'BCELoss',
    'L1Loss',
    'Reduction',
    'functional',
    'F',
]
