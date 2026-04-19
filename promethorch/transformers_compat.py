"""Re-export of the real shim in ``python/promethorch/transformers_compat.py``.

Two ``promethorch`` packages live in the repo (top-level and the canonical
``python/`` one that holds the C extension). When users run scripts from the
repo root, the top-level package wins, so we forward to the real shim by
loading it as a standalone module.
"""

import importlib.util as _ilu
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REAL_DIR = os.path.normpath(os.path.join(_HERE, "..", "python", "promethorch"))
_REAL = os.path.join(_REAL_DIR, "transformers_compat.py")


def _load_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load safetensors_reader so its relative imports inside the shim resolve.
_sr_path = os.path.join(_REAL_DIR, "safetensors_reader.py")
_load_module("promethorch.safetensors_reader", _sr_path)

# Now load the canonical shim
_real = _load_module("promethorch.transformers_compat", _REAL)

for _n in getattr(_real, "__all__", ()):
    globals()[_n] = getattr(_real, _n)
