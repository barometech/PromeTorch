"""PyTorch-compatible .pt / .pth save & load for PromeTorch state dicts.

Thin wrapper around the C++ `save_pytorch` / `load_pytorch` bindings
(see torch/serialization_pytorch.h). The produced file can be opened
with standard `torch.load(path)` from upstream PyTorch, and this
module can read most state_dict .pt files produced by `torch.save`.

The loader uses a restricted C++ unpickler that resolves only
`torch._utils._rebuild_tensor_v2`, the built-in `torch.*Storage`
classes, and `collections.OrderedDict` — any other GLOBAL aborts.
"""

from __future__ import annotations

from typing import Dict

from ._C import save_pytorch as _save_pytorch
from ._C import load_pytorch as _load_pytorch


def save_pytorch(state_dict: Dict[str, "Tensor"], path: str) -> None:
    """Save a PromeTorch state_dict as a PyTorch-compatible .pt file.

    Parameters
    ----------
    state_dict : dict[str, Tensor]
        Name -> PromeTorch Tensor mapping.
    path : str
        Destination file path (typically ``*.pt`` or ``*.pth``).

    The result is a ZIP archive with ``archive/data.pkl`` (pickle
    protocol 2) and ``archive/data/<key>`` raw little-endian bytes —
    exactly what upstream PyTorch >= 1.6 produces.
    """
    _save_pytorch(state_dict, path)


def load_pytorch(path: str) -> Dict[str, "Tensor"]:
    """Load a PyTorch .pt/.pth file into a name -> Tensor map.

    Works on files produced by standard ``torch.save`` as well as
    ``promethorch.save_pytorch``. Unknown pickle classes are rejected.
    """
    return _load_pytorch(path)


__all__ = ["save_pytorch", "load_pytorch"]
