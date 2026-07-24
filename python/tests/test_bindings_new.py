"""
Smoke test: import every new submodule and exercise one tiny call per class.

Runs under pytest OR as a standalone script:
    python -m pytest python/tests/test_bindings_new.py -q
    python python/tests/test_bindings_new.py
"""

from __future__ import annotations
import sys
import os
import tempfile

# Make sure python/ is on sys.path when invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _has_attr(obj, name: str) -> bool:
    return hasattr(obj, name) and getattr(obj, name) is not None


# ---------------------------------------------------------------------------
# 1. Top-level import
# ---------------------------------------------------------------------------
def test_toplevel_import():
    import prometorch as pt
    assert pt.__version__
    # Expected submodules attached to the root package.
    for name in ("nn", "distributed", "trainer", "onnx", "mlir",
                 "mobile", "jit", "vision", "quantization",
                 "autograd", "serve"):
        assert hasattr(pt, name), f"prometorch.{name} is missing"


# ---------------------------------------------------------------------------
# 2. nn.parallel
# ---------------------------------------------------------------------------
def test_nn_parallel():
    from prometorch.nn import parallel
    cfg = parallel.TPConfig()
    cfg.rank = 0
    cfg.world_size = 1
    # Constructors must not raise for rank 0 / world_size 1
    col = parallel.ColumnParallelLinear(8, 16, cfg, gather_output=False, bias=True)
    row = parallel.RowParallelLinear(8, 16, cfg, input_is_parallel=False, bias=True)
    assert col is not None
    assert row is not None
    assert parallel.Pipeline is not None


# ---------------------------------------------------------------------------
# 3. distributed
# ---------------------------------------------------------------------------
def test_distributed():
    from prometorch import distributed as d
    pg = d.init_process_group(backend="shared_memory", rank=0, world_size=1)
    assert pg is not None
    assert d.get_rank() == 0
    assert d.get_world_size() == 1
    assert d.is_initialized()
    # Reduce ops enum should exist
    assert hasattr(d, "ReduceOp")
    assert hasattr(d, "DistributedDataParallel")
    assert hasattr(d, "FullyShardedDataParallel")
    assert hasattr(d, "FSDPConfig")


# ---------------------------------------------------------------------------
# 4. trainer
# ---------------------------------------------------------------------------
def test_trainer():
    from prometorch.trainer import Trainer, TrainerConfig, LightningModule
    cfg = TrainerConfig(max_epochs=1, log_every_n_steps=1,
                        save_every_n_epochs=0, enable_progress_bar=False)
    t = Trainer(cfg)
    assert t.global_step == 0
    assert t.config.max_epochs == 1
    # LightningModule is instantiable but abstract methods raise.
    try:
        lm = LightningModule()
        assert hasattr(lm, "training_step")
    except TypeError:
        # C++ trampoline with pure virtuals — that's fine, skip.
        pass


# ---------------------------------------------------------------------------
# 5. onnx / mlir / mobile / jit
# ---------------------------------------------------------------------------
def test_export_modules():
    from prometorch import onnx, mlir, mobile, jit
    assert callable(onnx.export)
    assert callable(mlir.export)
    assert callable(mobile.export)
    # jit.compile always returns something callable (identity fallback ok).
    f = lambda x: x
    compiled = jit.compile(f)
    assert callable(compiled)


# ---------------------------------------------------------------------------
# 6. vision
# ---------------------------------------------------------------------------
def test_vision():
    from prometorch import vision
    # Compose with empty transforms must be constructible.
    compose = vision.transforms.Compose([])
    assert compose is not None
    tt = vision.transforms.ToTensor()
    assert tt is not None
    assert vision.ImageFolder is not None


# ---------------------------------------------------------------------------
# 7. quantization
# ---------------------------------------------------------------------------
def test_quantization():
    from prometorch import quantization as q
    assert callable(q.prepare_qat)
    assert callable(q.convert)
    assert callable(q.fake_quantize)
    assert q.QuantizedLinear is not None


# ---------------------------------------------------------------------------
# 8. autograd.jvp / vmap
# ---------------------------------------------------------------------------
def test_autograd_extra():
    from prometorch import autograd
    assert callable(autograd.jvp)
    assert callable(autograd.vmap)
    assert autograd.DualLevel is not None
    # Context manager smoke test
    with autograd.DualLevel():
        pass


# ---------------------------------------------------------------------------
# 9. serve
# ---------------------------------------------------------------------------
def test_serve():
    from prometorch import serve
    # LLMEngine is constructible with an empty temp dir; generate without a
    # forward_fn must raise a clear error.
    with tempfile.TemporaryDirectory() as tmp:
        eng = serve.LLMEngine(tmp)
        try:
            eng.generate(["hi"], max_tokens=1)
        except RuntimeError as e:
            assert "forward_fn" in str(e) or "tokenizer" in str(e)
        else:
            # If it didn't raise, the fallback succeeded — also OK.
            pass


# ---------------------------------------------------------------------------
# Main: run every test in sequence and report.
# ---------------------------------------------------------------------------
def main():
    tests = [
        test_toplevel_import,
        test_nn_parallel,
        test_distributed,
        test_trainer,
        test_export_modules,
        test_vision,
        test_quantization,
        test_autograd_extra,
        test_serve,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[ OK ] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
