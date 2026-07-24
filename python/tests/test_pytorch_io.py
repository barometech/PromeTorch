"""Round-trip tests for prometorch.save_pytorch / load_pytorch.

1. Save a PromeTorch Linear's state_dict → reopen via upstream ``torch.load``
   (if installed) and assert element-wise equality.
2. Produce a tiny reference .pt with upstream ``torch.save`` and load it
   back through PromeTorch, assert equality.
3. PromeTorch → PromeTorch round-trip.

Run:  python -m pytest python/tests/test_pytorch_io.py -s
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

import prometorch as pt


def _arr(t):
    """PromeTorch tensor → numpy."""
    return np.array(t, copy=True)


def test_round_trip_pt_to_pt():
    sd = {
        "w":  pt.randn([4, 3]),
        "b":  pt.zeros([3]),
        "ix": pt.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3)),
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pt")
        pt.save_pytorch(sd, p)
        loaded = pt.load_pytorch(p)
    assert set(loaded.keys()) == set(sd.keys())
    for k in sd:
        np.testing.assert_array_equal(_arr(loaded[k]), _arr(sd[k]))


def test_pt_save_torch_load():
    """PromeTorch saves → upstream torch.load reads."""
    try:
        import torch
    except ImportError:
        pytest.skip("upstream torch not installed")

    sd = {"weight": pt.randn([5, 7]), "bias": pt.zeros([7])}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "y.pt")
        pt.save_pytorch(sd, p)
        got = torch.load(p, weights_only=True)
    for k in sd:
        np.testing.assert_allclose(got[k].detach().cpu().numpy(), _arr(sd[k]))


def test_torch_save_pt_load():
    """Upstream torch.save writes → PromeTorch load_pytorch reads."""
    try:
        import torch
    except ImportError:
        pytest.skip("upstream torch not installed")

    ref = {"w": torch.randn(3, 4), "b": torch.arange(4, dtype=torch.float32)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.pt")
        torch.save(ref, p)
        got = pt.load_pytorch(p)
    for k in ref:
        np.testing.assert_allclose(_arr(got[k]), ref[k].numpy())


if __name__ == "__main__":
    test_round_trip_pt_to_pt()
    print("OK: pt -> pt round trip")
    try:
        test_pt_save_torch_load()
        print("OK: pt save -> torch.load")
        test_torch_save_pt_load()
        print("OK: torch.save -> pt.load_pytorch")
    except Exception as e:
        print("torch interop skipped/failed:", e)
