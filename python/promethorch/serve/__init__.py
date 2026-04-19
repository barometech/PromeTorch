"""
promethorch.serve — Minimal LLM inference engine.

The heavy lifting (prompt caching, KV-cache, batched sampling) lives in
Python and calls down to the _C primitives for tensor ops and state-dict
I/O. This file defines the `LLMEngine` surface.
"""

from __future__ import annotations
import os
from typing import Iterable, List, Optional


class LLMEngine:
    """A very small inference engine wrapper.

    Parameters
    ----------
    model_dir : str
        Directory containing a ``model.ptor`` state dict and optional
        ``tokenizer.json`` file.
    forward_fn : callable, optional
        A forward function of signature ``forward_fn(tokens) -> logits``.
        If provided, `generate` will use it; otherwise `generate` raises.
    """

    def __init__(self,
                 model_dir: str,
                 forward_fn=None,
                 tokenizer=None):
        self.model_dir = model_dir
        self.forward_fn = forward_fn
        self.tokenizer = tokenizer
        self._state_dict = None

        model_path = os.path.join(model_dir, "model.ptor")
        if os.path.isfile(model_path):
            try:
                from promethorch._C import serve as _cpp
                self._state_dict = _cpp._load_state_dict(model_path)
            except (ImportError, AttributeError):
                try:
                    from promethorch import load_state_dict
                    self._state_dict = load_state_dict(model_path)
                except Exception:
                    self._state_dict = None

    def state_dict(self):
        return self._state_dict

    def generate(self,
                 prompts: Iterable[str],
                 max_tokens: int = 32,
                 temperature: float = 1.0,
                 top_k: int = 0) -> List[str]:
        """Autoregressive generation. Requires `forward_fn` + `tokenizer`."""
        if self.forward_fn is None:
            raise RuntimeError(
                "LLMEngine.generate needs a forward_fn (callable) — "
                "pass one at construction time.")
        if self.tokenizer is None:
            raise RuntimeError(
                "LLMEngine.generate needs a tokenizer with .encode/.decode.")
        out = []
        for p in prompts:
            tokens = list(self.tokenizer.encode(p))
            for _ in range(max_tokens):
                logits = self.forward_fn(tokens)
                # Greedy by default
                nxt = int(self._sample(logits, temperature, top_k))
                tokens.append(nxt)
            out.append(self.tokenizer.decode(tokens))
        return out

    @staticmethod
    def _sample(logits, temperature: float, top_k: int):
        try:
            import promethorch as _pt
            if hasattr(logits, "argmax"):
                return int(logits.argmax().item())
        except Exception:
            pass
        # Fallback: numpy argmax.
        try:
            import numpy as np
            arr = logits if isinstance(logits, (list, tuple)) else list(logits)
            return int(np.argmax(np.asarray(arr)))
        except Exception:
            return 0


__all__ = ["LLMEngine"]
