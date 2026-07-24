"""
Self-test for prometorch.transformers_compat.

We synthesize a tiny safetensors checkpoint on disk (so the test is fully
self-contained — no HuggingFace download required) for each of the three
supported architectures (BERT, GPT-2, Llama). Then we load it via
``AutoModel.from_pretrained`` and run inference on a short prompt.

Verifies:
    1. ``import prometorch.transformers_compat`` works
    2. ``safetensors_reader`` round-trips arrays
    3. Each architecture produces output with the expected shape
    4. Tokenizer fallback works
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PKG_DIR, "python"))

from prometorch.safetensors_reader import SafeTensorsFile, load_file  # noqa: E402
from prometorch.transformers_compat import AutoModel, AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to write a tiny safetensors file
# ---------------------------------------------------------------------------

_DTYPE_TO_TAG = {np.dtype("float32"): "F32", np.dtype("float16"): "F16",
                 np.dtype("int64"): "I64"}


def write_safetensors(path: str, tensors: dict) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        header[name] = {
            "dtype": _DTYPE_TO_TAG[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        payload.extend(arr.tobytes())
        offset += nbytes
    header_json = json.dumps(header).encode("utf-8")
    # Pad header to 8-byte alignment (HF does this; we don't strictly need to)
    pad = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * pad
    with open(path, "wb") as fp:
        fp.write(struct.pack("<Q", len(header_json)))
        fp.write(header_json)
        fp.write(payload)


# ---------------------------------------------------------------------------
# Fake checkpoint factories
# ---------------------------------------------------------------------------

def _rand(*shape):
    return (np.random.randn(*shape) * 0.02).astype(np.float32)


def make_bert_dir(root: str) -> str:
    H, V, L, NH, IS = 32, 100, 2, 4, 64
    cfg = {
        "model_type": "bert",
        "architectures": ["BertModel"],
        "hidden_size": H,
        "vocab_size": V,
        "num_hidden_layers": L,
        "num_attention_heads": NH,
        "intermediate_size": IS,
        "max_position_embeddings": 64,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
    }
    state = {
        "bert.embeddings.word_embeddings.weight": _rand(V, H),
        "bert.embeddings.position_embeddings.weight": _rand(64, H),
        "bert.embeddings.token_type_embeddings.weight": _rand(2, H),
        "bert.embeddings.LayerNorm.weight": np.ones(H, dtype=np.float32),
        "bert.embeddings.LayerNorm.bias":   np.zeros(H, dtype=np.float32),
    }
    for i in range(L):
        b = f"bert.encoder.layer.{i}"
        for sub in ("query", "key", "value"):
            state[f"{b}.attention.self.{sub}.weight"] = _rand(H, H)
            state[f"{b}.attention.self.{sub}.bias"]   = np.zeros(H, dtype=np.float32)
        state[f"{b}.attention.output.dense.weight"] = _rand(H, H)
        state[f"{b}.attention.output.dense.bias"]   = np.zeros(H, dtype=np.float32)
        state[f"{b}.attention.output.LayerNorm.weight"] = np.ones(H, dtype=np.float32)
        state[f"{b}.attention.output.LayerNorm.bias"]   = np.zeros(H, dtype=np.float32)
        state[f"{b}.intermediate.dense.weight"] = _rand(IS, H)
        state[f"{b}.intermediate.dense.bias"]   = np.zeros(IS, dtype=np.float32)
        state[f"{b}.output.dense.weight"] = _rand(H, IS)
        state[f"{b}.output.dense.bias"]   = np.zeros(H, dtype=np.float32)
        state[f"{b}.output.LayerNorm.weight"] = np.ones(H, dtype=np.float32)
        state[f"{b}.output.LayerNorm.bias"]   = np.zeros(H, dtype=np.float32)
    d = os.path.join(root, "bert_tiny")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fp:
        json.dump(cfg, fp)
    write_safetensors(os.path.join(d, "model.safetensors"), state)
    return d


def make_gpt2_dir(root: str) -> str:
    H, V, L, NH = 32, 100, 2, 4
    cfg = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": H, "vocab_size": V, "n_layer": L, "n_head": NH,
        "n_positions": 64, "layer_norm_epsilon": 1e-5,
    }
    state = {
        "transformer.wte.weight": _rand(V, H),
        "transformer.wpe.weight": _rand(64, H),
        "transformer.ln_f.weight": np.ones(H, dtype=np.float32),
        "transformer.ln_f.bias":   np.zeros(H, dtype=np.float32),
    }
    for i in range(L):
        b = f"transformer.h.{i}"
        state[f"{b}.ln_1.weight"] = np.ones(H, dtype=np.float32)
        state[f"{b}.ln_1.bias"]   = np.zeros(H, dtype=np.float32)
        # GPT2 c_attn weight: (in=H, out=3H)
        state[f"{b}.attn.c_attn.weight"] = _rand(H, 3 * H)
        state[f"{b}.attn.c_attn.bias"]   = np.zeros(3 * H, dtype=np.float32)
        state[f"{b}.attn.c_proj.weight"] = _rand(H, H)
        state[f"{b}.attn.c_proj.bias"]   = np.zeros(H, dtype=np.float32)
        state[f"{b}.ln_2.weight"] = np.ones(H, dtype=np.float32)
        state[f"{b}.ln_2.bias"]   = np.zeros(H, dtype=np.float32)
        state[f"{b}.mlp.c_fc.weight"]   = _rand(H, 4 * H)
        state[f"{b}.mlp.c_fc.bias"]     = np.zeros(4 * H, dtype=np.float32)
        state[f"{b}.mlp.c_proj.weight"] = _rand(4 * H, H)
        state[f"{b}.mlp.c_proj.bias"]   = np.zeros(H, dtype=np.float32)
    d = os.path.join(root, "gpt2_tiny")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fp:
        json.dump(cfg, fp)
    write_safetensors(os.path.join(d, "model.safetensors"), state)
    return d


def make_llama_dir(root: str) -> str:
    H, V, L, NH, IS = 32, 100, 2, 4, 64
    cfg = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": H, "vocab_size": V, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NH,
        "intermediate_size": IS, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    }
    state = {
        "model.embed_tokens.weight": _rand(V, H),
        "model.norm.weight": np.ones(H, dtype=np.float32),
    }
    for i in range(L):
        b = f"model.layers.{i}"
        state[f"{b}.input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        state[f"{b}.self_attn.q_proj.weight"] = _rand(H, H)
        state[f"{b}.self_attn.k_proj.weight"] = _rand(H, H)
        state[f"{b}.self_attn.v_proj.weight"] = _rand(H, H)
        state[f"{b}.self_attn.o_proj.weight"] = _rand(H, H)
        state[f"{b}.post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)
        state[f"{b}.mlp.gate_proj.weight"] = _rand(IS, H)
        state[f"{b}.mlp.up_proj.weight"]   = _rand(IS, H)
        state[f"{b}.mlp.down_proj.weight"] = _rand(H, IS)
    d = os.path.join(root, "llama_tiny")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fp:
        json.dump(cfg, fp)
    write_safetensors(os.path.join(d, "model.safetensors"), state)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_safetensors_reader_roundtrip(tmpdir):
    arrs = {"a": np.arange(12, dtype=np.float32).reshape(3, 4),
            "b": np.ones((2, 5), dtype=np.float32)}
    p = os.path.join(tmpdir, "rt.safetensors")
    write_safetensors(p, arrs)
    loaded = load_file(p)
    assert set(loaded) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], arrs["a"])
    np.testing.assert_array_equal(loaded["b"], arrs["b"])
    print("OK: safetensors round-trip")


def test_bert(tmpdir):
    d = make_bert_dir(tmpdir)
    model = AutoModel.from_pretrained(d)
    input_ids = np.array([[1, 5, 9, 3]], dtype=np.int64)
    out = model(input_ids)
    shape = list(out.shape)
    assert shape == [1, 4, 32], f"unexpected shape {shape}"
    arr = out.numpy()
    assert np.isfinite(arr).all(), "non-finite values in BERT output"
    print(f"OK: BertModel forward, output shape={shape}")


def test_gpt2(tmpdir):
    d = make_gpt2_dir(tmpdir)
    model = AutoModel.from_pretrained(d)
    input_ids = np.array([[1, 5, 9, 3, 7]], dtype=np.int64)
    out = model(input_ids)
    shape = list(out.shape)
    assert shape == [1, 5, 32], f"unexpected shape {shape}"
    arr = out.numpy()
    assert np.isfinite(arr).all(), "non-finite values in GPT2 output"
    print(f"OK: GPT2Model forward, output shape={shape}")


def test_llama(tmpdir):
    d = make_llama_dir(tmpdir)
    model = AutoModel.from_pretrained(d)
    input_ids = np.array([[1, 5, 9, 3]], dtype=np.int64)
    out = model(input_ids)
    shape = list(out.shape)
    assert shape == [1, 4, 32], f"unexpected shape {shape}"
    arr = out.numpy()
    assert np.isfinite(arr).all(), "non-finite values in Llama output"
    print(f"OK: LlamaModel forward, output shape={shape}")


def test_tokenizer_fallback(tmpdir):
    d = os.path.join(tmpdir, "tok_only")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8") as fp:
        for w in ["[UNK]", "hello", "world", "##s"]:
            fp.write(w + "\n")
    tok = AutoTokenizer.from_pretrained(d)
    ids = tok.encode("hello worlds")
    assert isinstance(ids, list) and len(ids) >= 2
    print(f"OK: tokenizer.encode -> {ids}")


def main():
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_safetensors_reader_roundtrip(tmpdir)
        test_bert(tmpdir)
        test_gpt2(tmpdir)
        test_llama(tmpdir)
        test_tokenizer_fallback(tmpdir)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
