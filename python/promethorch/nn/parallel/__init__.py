"""
promethorch.nn.parallel — Tensor + pipeline parallelism.

C++ bindings come from `_C.parallel`. If the extension was built without
bindings_new.cpp (older .pyd) we expose a minimal Python fallback so that
`import promethorch.nn.parallel` still succeeds.

API:
    TPConfig(rank, world_size, sync_dir=..., timeout_us=...)
    ColumnParallelLinear(in_features, out_features, tp_config, gather_output=True, bias=True)
    RowParallelLinear(in_features, out_features, tp_config, input_is_parallel=True, bias=True)
    Pipeline(model, num_stages, chunks=4)
    tp_barrier(cfg); tp_all_gather(t, dim, cfg); tp_all_reduce_sum(t, cfg)
"""

from __future__ import annotations

try:
    from promethorch._C import parallel as _cpp
    _HAS_CPP = True
except (ImportError, AttributeError):
    _cpp = None
    _HAS_CPP = False


if _HAS_CPP:
    TPConfig              = _cpp.TPConfig
    ColumnParallelLinear  = _cpp.ColumnParallelLinear
    RowParallelLinear     = _cpp.RowParallelLinear
    Pipeline              = _cpp.Pipeline
    tp_barrier            = _cpp.tp_barrier
    tp_all_gather         = _cpp.tp_all_gather
    tp_all_reduce_sum     = _cpp.tp_all_reduce_sum
else:
    # ---- Python fallback: single-rank TP is the identity pass-through ----
    # We deliberately avoid importing promethorch.nn here because that
    # submodule may fail to load on stale _C builds (e.g. missing BatchNorm1d).
    # The fallback TP layers use pure-Python placeholders that record their
    # configuration and let the user swap in real nn.Linear manually.

    class TPConfig:  # pylint: disable=too-few-public-methods
        def __init__(self, rank: int = 0, world_size: int = 1,
                     sync_dir: str = "/dev/shm/pt_tp",
                     timeout_us: int = 60_000_000):
            self.rank = rank
            self.world_size = world_size
            self.sync_dir = sync_dir
            self.timeout_us = timeout_us

    def _no_collective(tensor, *_, **__):
        return tensor

    tp_barrier         = lambda cfg: None            # noqa: E731
    tp_all_gather      = _no_collective
    tp_all_reduce_sum  = _no_collective

    class _TPLinearBase:
        """Minimal parameter holder so fallback TP layers are still inspectable."""
        def __init__(self, in_features, out_features, bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.has_bias = bias
            # Try to use a real nn.Linear when available, else a stub.
            try:
                from promethorch.nn import Linear as _Linear
                self._inner = _Linear(in_features, out_features, bias=bias)
            except Exception:
                self._inner = None

        def forward(self, x):
            if self._inner is not None:
                return self._inner(x)
            raise RuntimeError(
                "Fallback TP layer cannot forward without promethorch.nn.Linear")

        __call__ = forward

    class ColumnParallelLinear(_TPLinearBase):
        """Fallback: behaves like a plain Linear on rank 0, world_size 1."""
        def __init__(self, in_features, out_features, tp_config,
                     gather_output=True, bias=True, init_seed=1234567):
            super().__init__(in_features, out_features, bias=bias)
            self._tp = tp_config
            self._gather = gather_output
            self.shard_size = (out_features // max(1, tp_config.world_size))

    class RowParallelLinear(_TPLinearBase):
        def __init__(self, in_features, out_features, tp_config,
                     input_is_parallel=True, bias=True, init_seed=7654321):
            super().__init__(in_features, out_features, bias=bias)
            self._tp = tp_config
            self._input_parallel = input_is_parallel
            self.shard_size = (in_features // max(1, tp_config.world_size))

    class Pipeline:
        """Fallback: runs the Sequential end-to-end (no threading)."""
        def __init__(self, model, num_stages: int, chunks: int = 4):
            if num_stages <= 0 or chunks <= 0:
                raise ValueError("num_stages and chunks must be > 0")
            self._model = model
            self.num_stages = num_stages
            self.chunks = chunks

        def forward(self, x):
            return self._model(x)

        __call__ = forward


__all__ = [
    "TPConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "Pipeline",
    "tp_barrier",
    "tp_all_gather",
    "tp_all_reduce_sum",
]
