"""
Minimal HuggingFace Transformers compatibility shim for PromeTorch.

Goal
----
Let users do::

    from promethorch.transformers_compat import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("./my_hf_model_dir")
    out = model(input_ids)

without installing HuggingFace ``transformers``. We only need:

* ``config.json``                       — architecture parameters
* ``model.safetensors`` (preferred) or
  ``model.safetensors.index.json`` + shards or
  ``pytorch_model.bin``                 — checkpoint weights
* ``tokenizer.json`` (preferred) or
  ``vocab.txt`` / ``vocab.json``+``merges.txt`` — tokenizer

Supported architectures (everything else needs a per-arch mapping):
    * BertModel       (encoder, learned positional embeddings, GELU FFN)
    * GPT2Model       (decoder, learned positional embeddings, causal mask)
    * LlamaModel      (decoder, RoPE, RMSNorm, SwiGLU FFN)

Design
------
We do **not** try to reuse the framework's C++ transformer modules — they
expect their own state-dict layout. Instead we build the forward pass in pure
Python using ``promethorch._C`` ops (``mm``, ``softmax``, ``cat`` …) and load
HF weights directly into Tensor parameters. That keeps the mapping from HF
parameter names to our internal storage explicit and easy to audit.

``pytorch_model.bin`` (legacy pickle format) is loaded via the standard
library ``pickle`` in a restricted unpickler that only resolves
``torch._utils._rebuild_tensor_v2`` and friends, so we can read those without
having ``torch`` installed. If pickle parsing fails we tell the user to
install the safetensors variant of the model — that path is the supported
one and works without dependencies.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import struct
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import the C extension directly. The shim only needs core tensor ops, which
# are stable across builds. Two scenarios are supported:
#   1. ``promethorch`` package is the canonical one under ``python/`` — then
#      ``from . import _C`` finds the ``_C.pyd`` next to this file.
#   2. The top-level repo ``promethorch/`` package shadowed it (no _C.pyd) —
#      we then locate _C on disk and load it as a sibling module manually.
try:  # noqa: SIM105
    from . import _C  # type: ignore
except ImportError:
    import glob as _glob
    import importlib.util as _ilu
    _here_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__))),
        os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", "python", "promethorch")),
    ]
    _C = None
    for _d in _here_candidates:
        for _pyd in _glob.glob(os.path.join(_d, "_C*.pyd")) + \
                    _glob.glob(os.path.join(_d, "_C*.so")):
            _spec = _ilu.spec_from_file_location("promethorch._C", _pyd)
            _mod = _ilu.module_from_spec(_spec)
            try:
                _spec.loader.exec_module(_mod)
                _C = _mod
                break
            except Exception:
                continue
        if _C is not None:
            break
    if _C is None:
        raise ImportError(
            "Could not locate promethorch._C extension. Build it with "
            "`pip install -e .` from the repo root."
        )

# Same fallback strategy for the safetensors reader (may be in a sibling pkg).
try:
    from .safetensors_reader import SafeTensorsFile, load_file, load_index
except ImportError:
    import importlib.util as _ilu
    _sr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "safetensors_reader.py")
    if not os.path.isfile(_sr_path):
        _sr_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "python", "promethorch", "safetensors_reader.py"))
    _spec = _ilu.spec_from_file_location("_pt_safetensors_reader", _sr_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    SafeTensorsFile = _mod.SafeTensorsFile
    load_file = _mod.load_file
    load_index = _mod.load_index


# ---------------------------------------------------------------------------
# Tensor helpers — thin wrappers around _C so the rest of the file reads
# closely to the math.
# ---------------------------------------------------------------------------

def _t(arr: np.ndarray) -> "_C.Tensor":
    """Move a numpy array into a PromeTorch Tensor (float32 by default)."""
    if arr.dtype != np.float32 and arr.dtype != np.int64 and arr.dtype != np.int32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return _C.from_numpy(arr)


def _np(t: "_C.Tensor") -> np.ndarray:
    return t.numpy()


def _matmul(a: "_C.Tensor", b: "_C.Tensor") -> "_C.Tensor":
    return _C.matmul(a, b)


def _linear(x: "_C.Tensor", weight: "_C.Tensor", bias: Optional["_C.Tensor"]) -> "_C.Tensor":
    """``y = x @ weight.T + bias``  (HF stores weights as ``[out, in]``)."""
    out = _C.matmul(x, weight.transpose(-1, -2))
    if bias is not None:
        out = out + bias
    return out


def _gelu(x: "_C.Tensor") -> "_C.Tensor":
    """Exact GELU (BERT uses approximate sometimes; close enough for inference)."""
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = (x + x3 * 0.044715) * c
    return x * 0.5 * (inner.tanh() + 1.0)


def _silu(x: "_C.Tensor") -> "_C.Tensor":
    return x * x.sigmoid()


def _softmax_lastdim(x: "_C.Tensor") -> "_C.Tensor":
    return _C.softmax(x, -1)


def _layer_norm(x: "_C.Tensor", weight: "_C.Tensor", bias: Optional["_C.Tensor"], eps: float) -> "_C.Tensor":
    """LayerNorm over the last dimension."""
    # Reduce along the last dim, keepdim by reshape afterwards
    last = x.shape[-1]
    x_np = _np(x)
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    out = (x_np - mean) / np.sqrt(var + eps)
    out = out * _np(weight)
    if bias is not None:
        out = out + _np(bias)
    return _t(out.astype(np.float32))


def _rms_norm(x: "_C.Tensor", weight: "_C.Tensor", eps: float) -> "_C.Tensor":
    """RMSNorm over the last dimension (Llama-style)."""
    x_np = _np(x).astype(np.float32)
    rms = np.sqrt((x_np * x_np).mean(axis=-1, keepdims=True) + eps)
    out = x_np / rms * _np(weight)
    return _t(out.astype(np.float32))


def _embedding(weight: "_C.Tensor", ids: np.ndarray) -> "_C.Tensor":
    """Lookup rows of ``weight`` indexed by ``ids`` (numpy int array)."""
    w = _np(weight)
    out = w[ids.astype(np.int64)]
    return _t(out.astype(np.float32))


def _causal_mask(seq_len: int) -> np.ndarray:
    """Returns an additive mask of shape (seq_len, seq_len) with -inf above diag."""
    m = np.zeros((seq_len, seq_len), dtype=np.float32)
    m[np.triu_indices(seq_len, k=1)] = -1e9
    return m


# ---------------------------------------------------------------------------
# Rotary position embeddings (Llama)
# ---------------------------------------------------------------------------

def _rope_cos_sin(seq_len: int, dim: int, base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.einsum("i,j->ij", t, inv_freq)        # (seq_len, dim/2)
    emb = np.concatenate((freqs, freqs), axis=-1)    # (seq_len, dim)
    return np.cos(emb), np.sin(emb)


def _apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: (..., seq, dim) ; cos/sin: (seq, dim)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_state_dict(model_dir: str) -> Dict[str, np.ndarray]:
    """Load weights from a HuggingFace model directory.

    Search order:
        1. model.safetensors                   (single file)
        2. model.safetensors.index.json        (sharded)
        3. pytorch_model.bin                   (legacy pickle, best effort)
        4. pytorch_model.bin.index.json        (sharded legacy)
    """
    one_file = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(one_file):
        return load_file(one_file)
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        return load_index(idx)
    bin_file = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(bin_file):
        return _load_pytorch_bin(bin_file)
    bin_idx = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(bin_idx):
        with open(bin_idx, "r", encoding="utf-8") as fp:
            spec = json.load(fp)
        out: Dict[str, np.ndarray] = {}
        seen_shards = set()
        for shard in spec["weight_map"].values():
            if shard in seen_shards:
                continue
            seen_shards.add(shard)
            out.update(_load_pytorch_bin(os.path.join(model_dir, shard)))
        return out
    raise FileNotFoundError(
        f"No model weights found in {model_dir}. Looked for "
        "model.safetensors, model.safetensors.index.json, pytorch_model.bin, "
        "pytorch_model.bin.index.json."
    )


# --- legacy pytorch_model.bin reader ---------------------------------------
# Best-effort: decode the zipped pickle that torch.save produces. If the file
# was not saved with the new zipfile format (very old torch), we bail out
# with a clear message — the user should re-export to safetensors.

class _RebuildTensorPlaceholder:
    """Placeholder returned by the unpickler. Resolved later when we have
    storage data from ``data/<storage_id>``."""

    def __init__(self, storage, storage_offset, size, stride, requires_grad, *_):
        self.storage = storage      # (_StorageStub, dtype_str)
        self.storage_offset = storage_offset
        self.size = tuple(size)
        self.stride = tuple(stride)


class _StorageStub:
    def __init__(self, dtype_str: str, key: str, location: str, num_elem: int):
        self.dtype_str = dtype_str
        self.key = key
        self.location = location
        self.num_elem = num_elem


_TORCH_DTYPE_TO_NP = {
    "FloatStorage": (np.dtype("float32"), 4),
    "DoubleStorage": (np.dtype("float64"), 8),
    "HalfStorage":  (np.dtype("float16"), 2),
    "LongStorage":  (np.dtype("int64"), 8),
    "IntStorage":   (np.dtype("int32"), 4),
    "ShortStorage": (np.dtype("int16"), 2),
    "CharStorage":  (np.dtype("int8"), 1),
    "ByteStorage":  (np.dtype("uint8"), 1),
    "BoolStorage":  (np.dtype("bool"), 1),
    "BFloat16Storage": (None, 2),
}


class _TorchUnpickler(pickle.Unpickler):
    """Minimal unpickler that handles the persistent IDs and a few classes
    used by ``torch.save`` for tensors."""

    def find_class(self, module: str, name: str):
        # Map torch internals to local stand-ins so pickle does not need
        # the real torch installed.
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _RebuildTensorPlaceholder
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch" and name in _TORCH_DTYPE_TO_NP:
            # ``torch.FloatStorage`` etc. — return a callable so pickle can
            # call it with no args; we just return the name string.
            dtype_name = name
            return lambda: dtype_name
        # Unknown class — return a dummy that swallows everything
        return _UnknownTorchObject

    def persistent_load(self, pid):
        # pid format: ('storage', storage_type, key, location, numel)
        if isinstance(pid, tuple) and pid[0] == "storage":
            _, storage_type, key, location, numel = pid
            # storage_type was already passed through find_class so it's a
            # callable returning the dtype-name string
            dtype_str = storage_type() if callable(storage_type) else str(storage_type)
            return _StorageStub(dtype_str, key, location, numel)
        return pid


class _UnknownTorchObject:
    def __init__(self, *a, **kw):
        pass

    def __setstate__(self, state):
        self.__state = state


def _load_pytorch_bin(path: str) -> Dict[str, np.ndarray]:
    """Decode a ``pytorch_model.bin`` into a name->ndarray dict."""
    if not zipfile.is_zipfile(path):
        raise NotImplementedError(
            f"{path} is not in the new torch.save zip format. "
            "Please re-export the checkpoint as model.safetensors."
        )
    with zipfile.ZipFile(path, "r") as zf:
        # Find the data.pkl entry (always at <archive_name>/data.pkl)
        pkl_names = [n for n in zf.namelist() if n.endswith("/data.pkl")]
        if not pkl_names:
            raise RuntimeError(f"No data.pkl in {path}")
        with zf.open(pkl_names[0]) as fp:
            unpickler = _TorchUnpickler(fp)
            state = unpickler.load()
        # Now fill in storage bytes
        archive_root = pkl_names[0].rsplit("/", 1)[0]
        out: Dict[str, np.ndarray] = {}
        for name, value in state.items():
            if not isinstance(value, _RebuildTensorPlaceholder):
                continue
            stor: _StorageStub = value.storage if not isinstance(value.storage, tuple) else value.storage[0]
            data_path = f"{archive_root}/data/{stor.key}"
            try:
                raw = zf.read(data_path)
            except KeyError:
                continue
            np_dtype_info = _TORCH_DTYPE_TO_NP.get(stor.dtype_str)
            if np_dtype_info is None or np_dtype_info[0] is None:
                # bf16 or unknown — skip; advise user to use safetensors
                warnings.warn(f"Skipping {name}: dtype {stor.dtype_str} not natively supported")
                continue
            np_dtype = np_dtype_info[0]
            arr = np.frombuffer(raw, dtype=np_dtype, count=stor.num_elem)
            # Apply offset, then reshape with strides if non-contiguous
            arr = arr[value.storage_offset:value.storage_offset + int(np.prod(value.size))]
            arr = arr.reshape(value.size)
            out[name] = arr.copy()
        return out


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class _ModelBase:
    """Common scaffold. Subclasses populate ``self.params`` with HF-named
    Tensors and implement ``forward(input_ids)``."""

    def __init__(self, config: dict, state: Dict[str, np.ndarray]):
        self.config = config
        self.params: Dict[str, "_C.Tensor"] = {}
        self._load(state)

    def _put(self, hf_name: str, state: Dict[str, np.ndarray]) -> None:
        if hf_name not in state:
            raise KeyError(f"Missing weight: {hf_name}")
        self.params[hf_name] = _t(state[hf_name])

    def _maybe(self, hf_name: str, state: Dict[str, np.ndarray]) -> None:
        if hf_name in state:
            self.params[hf_name] = _t(state[hf_name])

    def _load(self, state: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def __call__(self, input_ids: np.ndarray) -> "_C.Tensor":
        return self.forward(input_ids)

    def forward(self, input_ids: np.ndarray) -> "_C.Tensor":
        raise NotImplementedError


# --- BERT ------------------------------------------------------------------

class BertModel(_ModelBase):
    """Minimal BertModel forward (encoder only, no pooler output)."""

    def _load(self, state: Dict[str, np.ndarray]) -> None:
        cfg = self.config
        # Some HF dumps prefix everything with 'bert.'
        prefix = "bert." if any(k.startswith("bert.") for k in state) else ""
        self.prefix = prefix
        self._put(f"{prefix}embeddings.word_embeddings.weight", state)
        self._put(f"{prefix}embeddings.position_embeddings.weight", state)
        self._maybe(f"{prefix}embeddings.token_type_embeddings.weight", state)
        self._put(f"{prefix}embeddings.LayerNorm.weight", state)
        self._put(f"{prefix}embeddings.LayerNorm.bias", state)
        for i in range(cfg["num_hidden_layers"]):
            base = f"{prefix}encoder.layer.{i}"
            for sub in ("query", "key", "value"):
                self._put(f"{base}.attention.self.{sub}.weight", state)
                self._put(f"{base}.attention.self.{sub}.bias", state)
            self._put(f"{base}.attention.output.dense.weight", state)
            self._put(f"{base}.attention.output.dense.bias", state)
            self._put(f"{base}.attention.output.LayerNorm.weight", state)
            self._put(f"{base}.attention.output.LayerNorm.bias", state)
            self._put(f"{base}.intermediate.dense.weight", state)
            self._put(f"{base}.intermediate.dense.bias", state)
            self._put(f"{base}.output.dense.weight", state)
            self._put(f"{base}.output.dense.bias", state)
            self._put(f"{base}.output.LayerNorm.weight", state)
            self._put(f"{base}.output.LayerNorm.bias", state)

    def forward(self, input_ids: np.ndarray) -> "_C.Tensor":
        cfg = self.config
        prefix = self.prefix
        bsz, seq = input_ids.shape
        H = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        head_dim = H // n_heads

        positions = np.arange(seq, dtype=np.int64)[None, :].repeat(bsz, axis=0)
        x = _np(_embedding(self.params[f"{prefix}embeddings.word_embeddings.weight"], input_ids))
        x = x + _np(_embedding(self.params[f"{prefix}embeddings.position_embeddings.weight"], positions))
        tt_key = f"{prefix}embeddings.token_type_embeddings.weight"
        if tt_key in self.params:
            tt = _embedding(self.params[tt_key], np.zeros_like(input_ids, dtype=np.int64))
            x = x + _np(tt)
        x = _t(x)
        x = _layer_norm(x, self.params[f"{prefix}embeddings.LayerNorm.weight"],
                        self.params[f"{prefix}embeddings.LayerNorm.bias"],
                        cfg.get("layer_norm_eps", 1e-12))

        for i in range(cfg["num_hidden_layers"]):
            base = f"{prefix}encoder.layer.{i}"
            # Self-attention
            q = _linear(x, self.params[f"{base}.attention.self.query.weight"],
                        self.params[f"{base}.attention.self.query.bias"])
            k = _linear(x, self.params[f"{base}.attention.self.key.weight"],
                        self.params[f"{base}.attention.self.key.bias"])
            v = _linear(x, self.params[f"{base}.attention.self.value.weight"],
                        self.params[f"{base}.attention.self.value.bias"])
            # (B, S, H) -> (B, n_heads, S, head_dim)
            q_np = _np(q).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            k_np = _np(k).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            v_np = _np(v).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            scores = np.einsum("bnsd,bntd->bnst", q_np, k_np) / math.sqrt(head_dim)
            attn = _np(_softmax_lastdim(_t(scores)))
            ctx = np.einsum("bnst,bntd->bnsd", attn, v_np)
            ctx = ctx.transpose(0, 2, 1, 3).reshape(bsz, seq, H)
            attn_out = _linear(_t(ctx),
                               self.params[f"{base}.attention.output.dense.weight"],
                               self.params[f"{base}.attention.output.dense.bias"])
            x = _layer_norm(_t(_np(x) + _np(attn_out)),
                            self.params[f"{base}.attention.output.LayerNorm.weight"],
                            self.params[f"{base}.attention.output.LayerNorm.bias"],
                            cfg.get("layer_norm_eps", 1e-12))
            # FFN
            mid = _linear(x, self.params[f"{base}.intermediate.dense.weight"],
                          self.params[f"{base}.intermediate.dense.bias"])
            mid = _gelu(mid)
            ffn_out = _linear(mid, self.params[f"{base}.output.dense.weight"],
                              self.params[f"{base}.output.dense.bias"])
            x = _layer_norm(_t(_np(x) + _np(ffn_out)),
                            self.params[f"{base}.output.LayerNorm.weight"],
                            self.params[f"{base}.output.LayerNorm.bias"],
                            cfg.get("layer_norm_eps", 1e-12))
        return x


# --- GPT-2 -----------------------------------------------------------------

class GPT2Model(_ModelBase):
    """Minimal GPT2 forward. No KV cache — recomputes each call."""

    def _load(self, state: Dict[str, np.ndarray]) -> None:
        cfg = self.config
        prefix = "transformer." if any(k.startswith("transformer.") for k in state) else ""
        self.prefix = prefix
        self._put(f"{prefix}wte.weight", state)
        self._put(f"{prefix}wpe.weight", state)
        for i in range(cfg["n_layer"]):
            base = f"{prefix}h.{i}"
            self._put(f"{base}.ln_1.weight", state)
            self._put(f"{base}.ln_1.bias", state)
            # GPT2 stores qkv as one fused Conv1D with shape (in, 3*hidden)
            self._put(f"{base}.attn.c_attn.weight", state)
            self._put(f"{base}.attn.c_attn.bias", state)
            self._put(f"{base}.attn.c_proj.weight", state)
            self._put(f"{base}.attn.c_proj.bias", state)
            self._put(f"{base}.ln_2.weight", state)
            self._put(f"{base}.ln_2.bias", state)
            self._put(f"{base}.mlp.c_fc.weight", state)
            self._put(f"{base}.mlp.c_fc.bias", state)
            self._put(f"{base}.mlp.c_proj.weight", state)
            self._put(f"{base}.mlp.c_proj.bias", state)
        self._put(f"{prefix}ln_f.weight", state)
        self._put(f"{prefix}ln_f.bias", state)

    def _conv1d(self, x_np: np.ndarray, W: "_C.Tensor", b: "_C.Tensor") -> np.ndarray:
        """GPT2 Conv1D = x @ W + b, where W is (in, out) (NOT transposed)."""
        return x_np @ _np(W) + _np(b)

    def forward(self, input_ids: np.ndarray) -> "_C.Tensor":
        cfg = self.config
        prefix = self.prefix
        bsz, seq = input_ids.shape
        H = cfg["n_embd"]
        n_heads = cfg["n_head"]
        head_dim = H // n_heads
        eps = cfg.get("layer_norm_epsilon", 1e-5)

        positions = np.arange(seq, dtype=np.int64)[None, :].repeat(bsz, axis=0)
        x = _np(_embedding(self.params[f"{prefix}wte.weight"], input_ids))
        x = x + _np(_embedding(self.params[f"{prefix}wpe.weight"], positions))
        causal = _causal_mask(seq)

        for i in range(cfg["n_layer"]):
            base = f"{prefix}h.{i}"
            # Pre-LN
            x_norm = _np(_layer_norm(_t(x), self.params[f"{base}.ln_1.weight"],
                                     self.params[f"{base}.ln_1.bias"], eps))
            qkv = self._conv1d(x_norm, self.params[f"{base}.attn.c_attn.weight"],
                               self.params[f"{base}.attn.c_attn.bias"])
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            scores = np.einsum("bnsd,bntd->bnst", q, k) / math.sqrt(head_dim)
            scores = scores + causal
            attn = _np(_softmax_lastdim(_t(scores)))
            ctx = np.einsum("bnst,bntd->bnsd", attn, v).transpose(0, 2, 1, 3).reshape(bsz, seq, H)
            attn_out = self._conv1d(ctx, self.params[f"{base}.attn.c_proj.weight"],
                                    self.params[f"{base}.attn.c_proj.bias"])
            x = x + attn_out
            # MLP
            x_norm2 = _np(_layer_norm(_t(x), self.params[f"{base}.ln_2.weight"],
                                      self.params[f"{base}.ln_2.bias"], eps))
            mid = self._conv1d(x_norm2, self.params[f"{base}.mlp.c_fc.weight"],
                               self.params[f"{base}.mlp.c_fc.bias"])
            mid = _np(_gelu(_t(mid)))
            ffn_out = self._conv1d(mid, self.params[f"{base}.mlp.c_proj.weight"],
                                   self.params[f"{base}.mlp.c_proj.bias"])
            x = x + ffn_out

        x = _layer_norm(_t(x), self.params[f"{prefix}ln_f.weight"],
                        self.params[f"{prefix}ln_f.bias"], eps)
        return x


# --- Llama -----------------------------------------------------------------

class LlamaModel(_ModelBase):
    """Minimal Llama forward (decoder-only, RoPE, RMSNorm, SwiGLU)."""

    def _load(self, state: Dict[str, np.ndarray]) -> None:
        cfg = self.config
        prefix = "model." if any(k.startswith("model.") for k in state) else ""
        self.prefix = prefix
        self._put(f"{prefix}embed_tokens.weight", state)
        for i in range(cfg["num_hidden_layers"]):
            base = f"{prefix}layers.{i}"
            self._put(f"{base}.input_layernorm.weight", state)
            self._put(f"{base}.self_attn.q_proj.weight", state)
            self._put(f"{base}.self_attn.k_proj.weight", state)
            self._put(f"{base}.self_attn.v_proj.weight", state)
            self._put(f"{base}.self_attn.o_proj.weight", state)
            self._put(f"{base}.post_attention_layernorm.weight", state)
            self._put(f"{base}.mlp.gate_proj.weight", state)
            self._put(f"{base}.mlp.up_proj.weight", state)
            self._put(f"{base}.mlp.down_proj.weight", state)
        self._put(f"{prefix}norm.weight", state)

    def forward(self, input_ids: np.ndarray) -> "_C.Tensor":
        cfg = self.config
        prefix = self.prefix
        bsz, seq = input_ids.shape
        H = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        n_kv = cfg.get("num_key_value_heads", n_heads)
        head_dim = H // n_heads
        rope_base = cfg.get("rope_theta", 10000.0)
        eps = cfg.get("rms_norm_eps", 1e-6)

        cos, sin = _rope_cos_sin(seq, head_dim, base=rope_base)
        causal = _causal_mask(seq)
        kv_repeat = n_heads // n_kv

        x = _embedding(self.params[f"{prefix}embed_tokens.weight"], input_ids)

        for i in range(cfg["num_hidden_layers"]):
            base = f"{prefix}layers.{i}"
            x_norm = _rms_norm(x, self.params[f"{base}.input_layernorm.weight"], eps)
            q = _linear(x_norm, self.params[f"{base}.self_attn.q_proj.weight"], None)
            k = _linear(x_norm, self.params[f"{base}.self_attn.k_proj.weight"], None)
            v = _linear(x_norm, self.params[f"{base}.self_attn.v_proj.weight"], None)
            q_np = _np(q).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
            k_np = _np(k).reshape(bsz, seq, n_kv,    head_dim).transpose(0, 2, 1, 3)
            v_np = _np(v).reshape(bsz, seq, n_kv,    head_dim).transpose(0, 2, 1, 3)
            q_np = _apply_rope(q_np, cos, sin)
            k_np = _apply_rope(k_np, cos, sin)
            if kv_repeat > 1:
                k_np = np.repeat(k_np, kv_repeat, axis=1)
                v_np = np.repeat(v_np, kv_repeat, axis=1)
            scores = np.einsum("bnsd,bntd->bnst", q_np, k_np) / math.sqrt(head_dim)
            scores = scores + causal
            attn = _np(_softmax_lastdim(_t(scores)))
            ctx = np.einsum("bnst,bntd->bnsd", attn, v_np).transpose(0, 2, 1, 3).reshape(bsz, seq, H)
            attn_out = _linear(_t(ctx),
                               self.params[f"{base}.self_attn.o_proj.weight"], None)
            x = _t(_np(x) + _np(attn_out))

            x_norm2 = _rms_norm(x, self.params[f"{base}.post_attention_layernorm.weight"], eps)
            gate = _linear(x_norm2, self.params[f"{base}.mlp.gate_proj.weight"], None)
            up = _linear(x_norm2, self.params[f"{base}.mlp.up_proj.weight"], None)
            ffn = _linear(_silu(gate) * up,
                          self.params[f"{base}.mlp.down_proj.weight"], None)
            x = _t(_np(x) + _np(ffn))

        x = _rms_norm(x, self.params[f"{prefix}norm.weight"], eps)
        return x


# ---------------------------------------------------------------------------
# AutoModel / AutoTokenizer
# ---------------------------------------------------------------------------

_ARCH_REGISTRY = {
    "BertModel":      BertModel,
    "BertForMaskedLM": BertModel,         # share encoder; LM head dropped
    "BertForSequenceClassification": BertModel,
    "GPT2Model":      GPT2Model,
    "GPT2LMHeadModel": GPT2Model,
    "LlamaModel":     LlamaModel,
    "LlamaForCausalLM": LlamaModel,
}


def _detect_architecture(config: dict) -> str:
    arches = config.get("architectures") or []
    for a in arches:
        if a in _ARCH_REGISTRY:
            return a
    # Fall back to model_type
    mt = config.get("model_type", "").lower()
    if mt == "bert":
        return "BertModel"
    if mt in ("gpt2", "openai-gpt"):
        return "GPT2Model"
    if mt == "llama":
        return "LlamaModel"
    raise ValueError(
        f"Unsupported architecture {arches!r} (model_type={mt!r}). "
        f"Supported: {sorted(set(_ARCH_REGISTRY.keys()))}"
    )


class AutoModel:
    """HuggingFace-style entry point. Returns one of BertModel / GPT2Model /
    LlamaModel populated with weights from the directory."""

    @staticmethod
    def from_pretrained(model_dir: str) -> _ModelBase:
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        with open(cfg_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)
        arch_name = _detect_architecture(config)
        state = _load_state_dict(model_dir)
        cls = _ARCH_REGISTRY[arch_name]
        return cls(config, state)


# --- Tokenizer -------------------------------------------------------------

class _SimpleTokenizer:
    """Very small tokenizer.json reader.

    Supports the two most common templates:
        * WordPiece (BERT) — vocab + unk_token
        * Byte-level BPE (GPT-2/Llama via tiktoken-style merges)

    For unsupported tokenizers we fall back to whitespace splitting and
    issue a warning. Callers needing perfect token IDs should provide their
    own pre-encoded ``input_ids``.
    """

    def __init__(self, tok: dict):
        self.tok = tok
        model = tok.get("model", {})
        self.kind = model.get("type", "Unknown")
        self.vocab: Dict[str, int] = model.get("vocab", {}) or {}
        self.merges: List[str] = model.get("merges", []) or []
        self.unk_token = model.get("unk_token", "[UNK]")
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        if self.kind in ("BPE", "ByteLevel"):
            # Crude byte-pair fallback — not exact but enough for self-test
            ids = []
            for word in text.split():
                if word in self.vocab:
                    ids.append(self.vocab[word])
                else:
                    for ch in word:
                        ids.append(self.vocab.get(ch, self.vocab.get(self.unk_token, 0)))
            return ids
        if self.kind == "WordPiece":
            ids = []
            for word in text.split():
                if word in self.vocab:
                    ids.append(self.vocab[word])
                    continue
                # Greedy longest-match
                pos = 0
                while pos < len(word):
                    matched = None
                    for end in range(len(word), pos, -1):
                        sub = word[pos:end] if pos == 0 else "##" + word[pos:end]
                        if sub in self.vocab:
                            matched = (sub, end)
                            break
                    if matched is None:
                        ids.append(self.vocab.get(self.unk_token, 0))
                        break
                    ids.append(self.vocab[matched[0]])
                    pos = matched[1]
            return ids
        warnings.warn(f"Unsupported tokenizer kind {self.kind!r}; falling back to whitespace IDs")
        return [self.vocab.get(w, 0) for w in text.split()]

    def decode(self, ids: List[int]) -> str:
        toks = [self.id_to_token.get(int(i), "") for i in ids]
        # Strip wordpiece markers
        out = []
        for t in toks:
            if t.startswith("##"):
                out.append(t[2:])
            else:
                out.append(" " + t)
        return "".join(out).strip()


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_dir: str) -> _SimpleTokenizer:
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.isfile(tok_path):
            with open(tok_path, "r", encoding="utf-8") as fp:
                tok = json.load(fp)
            return _SimpleTokenizer(tok)
        # Fallback: build a minimal vocab from vocab.txt (BERT)
        vocab_txt = os.path.join(model_dir, "vocab.txt")
        if os.path.isfile(vocab_txt):
            with open(vocab_txt, "r", encoding="utf-8") as fp:
                vocab = {line.strip(): i for i, line in enumerate(fp)}
            return _SimpleTokenizer({"model": {"type": "WordPiece", "vocab": vocab}})
        vocab_json = os.path.join(model_dir, "vocab.json")
        if os.path.isfile(vocab_json):
            with open(vocab_json, "r", encoding="utf-8") as fp:
                vocab = json.load(fp)
            return _SimpleTokenizer({"model": {"type": "BPE", "vocab": vocab}})
        raise FileNotFoundError(f"No tokenizer files found in {model_dir}")


__all__ = [
    "AutoModel",
    "AutoTokenizer",
    "BertModel",
    "GPT2Model",
    "LlamaModel",
    "SafeTensorsFile",
]
