"""
PromeTorch — Russian hardware-native deep learning framework.

Supported backends: CPU (TUDA), CUDA, NMCard, LinQ
Supported hardware: Intel/AMD (AVX2), Baikal-M/S (NEON), Elbrus (E2K), NM Card Mini, LinQ H1M

Usage:
    import promethorch as pt
    x = pt.tensor([1.0, 2.0, 3.0])
    y = pt.zeros(3, 4)
    z = x.mm(y)
"""

__version__ = "0.1.0"

# Try to import C++ extension
_C_LOADED = False
try:
    from promethorch._C import *
    from promethorch._C import (
        __version__ as _cpp_version,
        cuda_is_available,
        cuda_device_count,
    )
    _C_LOADED = True
except ImportError:
    pass

if not _C_LOADED:
    # Pure Python fallback — minimal API for when C++ extension not built
    import warnings
    warnings.warn(
        "PromeTorch C++ extension not found. "
        "Install with: pip install -e . "
        "For full functionality, build from source with cmake."
    )

    def cuda_is_available():
        return False

    def cuda_device_count():
        return 0


# Device types — always available
class device:
    """Device specification for tensor placement."""
    CPU = "cpu"
    CUDA = "cuda"
    NMCARD = "nmcard"
    LINQ = "linq"

    def __init__(self, type_str, index=0):
        if isinstance(type_str, str) and ":" in type_str:
            parts = type_str.split(":")
            type_str = parts[0]
            index = int(parts[1])
        self.type = type_str
        self.index = index

    def __repr__(self):
        if self.index > 0:
            return f"device('{self.type}:{self.index}')"
        return f"device('{self.type}')"

    def __str__(self):
        if self.index > 0:
            return f"{self.type}:{self.index}"
        return self.type

    def __eq__(self, other):
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other
        return False


# Convenience device constants
cpu = device("cpu")
cuda = device("cuda")
nmcard = device("nmcard")
linq = device("linq")


# Submodules
from promethorch import nn
from promethorch import optim
