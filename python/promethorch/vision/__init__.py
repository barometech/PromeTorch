"""
promethorch.vision — torchvision-compatible datasets, transforms, models.
"""

from __future__ import annotations

try:
    from promethorch._C import vision as _cpp
    _HAS_CPP = True
except (ImportError, AttributeError):
    _cpp = None
    _HAS_CPP = False


# ---- transforms submodule ----
from . import transforms  # noqa: E402  (must import after _cpp lookup)


if _HAS_CPP:
    ImageFolder = _cpp.ImageFolder
    mobilenet_v2 = _cpp.mobilenet_v2
else:
    # Pure-Python fallback ImageFolder scans directory entries and loads PPMs.
    import os
    import struct

    class ImageFolder:
        def __init__(self, root: str, transform=None):
            self.root = root
            self.transform = transform
            if not os.path.isdir(root):
                raise RuntimeError(f"ImageFolder: {root} is not a directory")
            self.classes = sorted([
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ])
            if not self.classes:
                raise RuntimeError(f"ImageFolder: no class subdirs in {root}")
            self.samples = []
            for ci, cls in enumerate(self.classes):
                cdir = os.path.join(root, cls)
                for name in sorted(os.listdir(cdir)):
                    if name.lower().endswith((".ppm", ".pgm", ".pnm")):
                        self.samples.append((os.path.join(cdir, name), ci))
            if not self.samples:
                raise RuntimeError(f"ImageFolder: no images under {root}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, target = self.samples[idx]
            img = _load_pnm(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def mobilenet_v2(num_classes: int = 1000, width_mult: float = 1.0):
        raise RuntimeError(
            "mobilenet_v2 requires _C built with bindings_new.cpp (torchvision models)"
        )


def _load_pnm(path: str):
    """Minimal P5/P6 PPM/PGM reader (pure Python fallback)."""
    import promethorch as _pt
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P5", b"P6"):
            raise RuntimeError(f"Unsupported PNM magic: {magic!r}")
        channels = 3 if magic == b"P6" else 1
        header = []
        while len(header) < 3:
            line = f.readline().strip()
            if line.startswith(b"#"):
                continue
            header.extend(line.split())
        w, h, _max = int(header[0]), int(header[1]), int(header[2])
        data = f.read(w * h * channels)
    # Return a simple tuple (data, shape) if promethorch tensors unavailable.
    try:
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, channels)
        return _pt.from_numpy(arr) if hasattr(_pt, "from_numpy") else arr
    except Exception:
        return data


__all__ = ["ImageFolder", "mobilenet_v2", "transforms"]
