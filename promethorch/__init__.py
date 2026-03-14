"""PromeTorch — Russian hardware-native deep learning framework."""

__version__ = "0.1.0"

try:
    from promethorch._C import *
except ImportError:
    import warnings
    warnings.warn(
        "PromeTorch C++ extension not found. "
        "Install with: pip install -e . "
        "or build from source."
    )

# Device types
class device:
    """Device specification."""
    CPU = "cpu"
    CUDA = "cuda"
    NMCARD = "nmcard"
    LINQ = "linq"

    def __init__(self, type_str, index=0):
        self.type = type_str
        self.index = index

    def __repr__(self):
        if self.index > 0:
            return f"device('{self.type}:{self.index}')"
        return f"device('{self.type}')"

# Convenience
cpu = device("cpu")
cuda = device("cuda")
nmcard = device("nmcard")
linq = device("linq")
