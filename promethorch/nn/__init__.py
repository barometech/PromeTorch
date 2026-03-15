"""
Neural network modules.

Usage:
    import promethorch as pt
    model = pt.nn.Sequential(
        pt.nn.Linear(784, 256),
        pt.nn.ReLU(),
        pt.nn.Linear(256, 10)
    )
"""

try:
    from promethorch._C.nn import *
except ImportError:
    try:
        from promethorch._C import (
            Linear, Conv2d, BatchNorm2d,
            ReLU, Sigmoid, Tanh, Softmax,
            CrossEntropyLoss, MSELoss,
        )
    except ImportError:
        pass
