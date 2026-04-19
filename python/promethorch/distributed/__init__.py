"""
promethorch.distributed — multi-rank training primitives.

API:
    init_process_group(backend='shared_memory', rank=0, world_size=1,
                       master_addr='127.0.0.1', master_port=29500)
    all_reduce(tensor, op=ReduceOp.SUM)
    broadcast(tensor, src=0)
    barrier()
    DistributedDataParallel(module, process_group)
    FullyShardedDataParallel(module, config)
    launch(world_size, worker_fn, master_addr='127.0.0.1', master_port=29500)

Backend 'shared_memory' always works (intra-process). 'nccl' requires the
extension to be built with PT_USE_CUDA=ON + PT_USE_NCCL=ON.
"""

from __future__ import annotations
from typing import Callable, Optional

try:
    from promethorch._C import distributed as _cpp
    _HAS_CPP = True
except (ImportError, AttributeError):
    _cpp = None
    _HAS_CPP = False


# ---------------------------------------------------------------------------
# Module-level singleton: the ProcessGroup currently bound to this process.
# ---------------------------------------------------------------------------
_current_pg = None


if _HAS_CPP:
    ReduceOp                    = _cpp.ReduceOp
    BackendType                 = _cpp.BackendType
    ProcessGroup                = _cpp.ProcessGroup
    DistributedDataParallel     = _cpp.DistributedDataParallel
    FullyShardedDataParallel    = _cpp.FullyShardedDataParallel
    FSDPConfig                  = _cpp.FSDPConfig
    DistArgs                    = _cpp.DistArgs

    def init_process_group(backend: str = "shared_memory",
                           rank: int = 0,
                           world_size: int = 1,
                           master_addr: str = "127.0.0.1",
                           master_port: int = 29500):
        """Create a per-rank ProcessGroup. For multi-process training each
        process calls this once with its own `rank`."""
        global _current_pg
        _current_pg = _cpp.init_process_group_single_rank(
            backend, rank, world_size)
        _current_pg.__master_addr = master_addr
        _current_pg.__master_port = master_port
        return _current_pg

    def all_reduce(tensor, op=None):
        if _current_pg is None:
            raise RuntimeError("init_process_group() was not called")
        if op is None:
            op = ReduceOp.SUM
        _current_pg.all_reduce(tensor, op)

    def broadcast(tensor, src: int = 0):
        if _current_pg is None:
            raise RuntimeError("init_process_group() was not called")
        _current_pg.broadcast(tensor, src)

    def barrier():
        if _current_pg is None:
            raise RuntimeError("init_process_group() was not called")
        _current_pg.barrier()

    def launch(world_size: int, worker_fn: Callable[[int, int], Optional[int]],
               master_addr: str = "127.0.0.1", master_port: int = 29500):
        return _cpp.launch(world_size, worker_fn, master_addr, master_port)

else:
    # ---- Pure-Python fallback for a 1-rank "distributed" setup. ----

    class ReduceOp:  # pylint: disable=too-few-public-methods
        SUM, AVG, MAX, MIN = 0, 1, 2, 3

    class BackendType:
        SHARED_MEMORY = 0
        NCCL          = 1

    class ProcessGroup:
        def __init__(self, rank=0, world_size=1, backend="shared_memory"):
            self.rank = rank
            self.world_size = world_size
            self.backend = backend

        def all_reduce(self, tensor, op=None):  # no-op for world_size=1
            return tensor

        def broadcast(self, tensor, src=0):
            return tensor

        def barrier(self):
            pass

    class FSDPConfig:
        class ShardingStrategy:
            FULL_SHARD, SHARD_GRAD_OP, NO_SHARD = 0, 1, 2

        def __init__(self):
            self.rank = 0
            self.world_size = 1
            self.sync_dir = "/dev/shm/pt_fsdp"
            self.timeout_ms = 120000
            self.poll_us = 1000
            self.strategy = FSDPConfig.ShardingStrategy.FULL_SHARD

    class _NoSyncCtx:
        """Pure-Python fallback context manager for DDP.no_sync()."""
        def __init__(self, ddp):
            self._ddp = ddp
            self._prev = True

        def __enter__(self):
            self._prev = self._ddp.require_grad_sync
            self._ddp.require_grad_sync = False
            return self

        def __exit__(self, exc_type, exc_val, tb):
            self._ddp.require_grad_sync = self._prev
            return False

    class DistributedDataParallel:
        def __init__(self, module, process_group, broadcast_parameters=True):
            self.module = module
            self.process_group = process_group
            # Default True preserves legacy behaviour (every sync call runs).
            self.require_grad_sync = True

        def forward(self, x):
            return self.module(x)

        def __call__(self, x):
            return self.forward(x)

        def finish_gradient_synchronization(self):
            # No real AllReduce in fallback (world_size always 1), but honour
            # the no_sync() flag so caller code is identical to the C++ path.
            if not self.require_grad_sync:
                return

        def sync_gradients(self):
            if not self.require_grad_sync:
                return

        def no_sync(self):
            """Context manager that suppresses gradient sync within the
            with-block. See C++ DDPNoSyncGuard / Python DDP.no_sync docs."""
            return _NoSyncCtx(self)

    class FullyShardedDataParallel:
        def __init__(self, module, config):
            self.module = module
            self.config = config
            self.rank = config.rank
            self.world_size = config.world_size

        def forward(self, x):
            return self.module(x)

        def __call__(self, x):
            return self.forward(x)

        def all_gather_params(self): pass
        def reshard_params(self): pass
        def reduce_scatter_grads(self): pass

    class DistArgs:
        def __init__(self):
            self.rank = 0
            self.world_size = 1
            self.master_addr = "127.0.0.1"
            self.master_port = 29500

    def init_process_group(backend="shared_memory", rank=0, world_size=1,
                           master_addr="127.0.0.1", master_port=29500):
        global _current_pg
        _current_pg = ProcessGroup(rank, world_size, backend)
        return _current_pg

    def all_reduce(tensor, op=None):
        if _current_pg is None:
            raise RuntimeError("init_process_group() was not called")
        return _current_pg.all_reduce(tensor, op)

    def broadcast(tensor, src=0):
        if _current_pg is None:
            raise RuntimeError("init_process_group() was not called")
        return _current_pg.broadcast(tensor, src)

    def barrier():
        if _current_pg is not None:
            _current_pg.barrier()

    def launch(world_size, worker_fn, master_addr="127.0.0.1", master_port=29500):
        """Fallback launcher: runs worker_fn serially for each rank in-process.
        Not true multi-process but good enough for tests."""
        max_rc = 0
        for r in range(world_size):
            rc = worker_fn(r, world_size) or 0
            if rc > max_rc:
                max_rc = rc
        return max_rc


def get_rank() -> int:
    return _current_pg.rank if _current_pg is not None else 0


def get_world_size() -> int:
    return _current_pg.world_size if _current_pg is not None else 1


def is_initialized() -> bool:
    return _current_pg is not None


__all__ = [
    "ReduceOp", "BackendType", "ProcessGroup",
    "DistributedDataParallel", "FullyShardedDataParallel", "FSDPConfig",
    "DistArgs",
    "init_process_group", "all_reduce", "broadcast", "barrier", "launch",
    "get_rank", "get_world_size", "is_initialized",
]
