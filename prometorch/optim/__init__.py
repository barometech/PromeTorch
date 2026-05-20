"""
Optimizers.

Usage:
    optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
"""

try:
    from prometorch._C.optim import *
except ImportError:
    try:
        from prometorch._C import SGD, Adam, AdamW, RMSprop
    except ImportError:
        pass
