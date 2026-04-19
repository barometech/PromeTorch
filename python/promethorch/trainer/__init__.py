"""
promethorch.trainer — Lightning-style training loop.

Subclass LightningModule, override `training_step`, `configure_optimizer`
(and optionally `validation_step`), then call `Trainer(cfg).fit(module, loader)`.

The Python-side `Trainer.fit` drives the loop so it works with any iterable
`loader` (C++ DataLoader, Python list, generator, ...). This sidesteps the
templated C++ DataLoader constraint.
"""

from __future__ import annotations
import os
import time
from typing import Iterable, Optional

try:
    from promethorch._C import trainer as _cpp
    _HAS_CPP = True
    _TrainerConfigCpp = _cpp.TrainerConfig
    _LightningModuleBase = _cpp.LightningModule
except (ImportError, AttributeError):
    _cpp = None
    _HAS_CPP = False
    _TrainerConfigCpp = None
    # Fallback uses nn.Module as a base so `parameters()` / `state_dict()`
    # still work when running against a pre-existing _C.pyd.
    try:
        from promethorch.nn import Module as _NnModule
        _LightningModuleBase = _NnModule
    except ImportError:
        class _LightningModuleBase:      # absolute last-resort stub
            def __init__(self, *_, **__): pass
            def train(self, mode=True): pass
            def eval(self): pass


class TrainerConfig:
    """Plain-Python config that mirrors the C++ TrainerConfig fields."""
    def __init__(self,
                 max_epochs: int = 10,
                 log_every_n_steps: int = 50,
                 val_check_interval: int = 0,
                 checkpoint_dir: str = "./checkpoints",
                 save_every_n_epochs: int = 1,
                 enable_progress_bar: bool = True,
                 gradient_clip_val: float = 0.0,
                 accumulate_grad_batches: int = 1):
        self.max_epochs = max_epochs
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.val_check_interval = val_check_interval
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = max(0, save_every_n_epochs)
        self.enable_progress_bar = enable_progress_bar
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = max(1, accumulate_grad_batches)

    def _to_cpp(self):
        """Materialize a C++ TrainerConfig if bindings are available."""
        if _TrainerConfigCpp is None:
            return None
        c = _TrainerConfigCpp()
        c.max_epochs = self.max_epochs
        c.log_every_n_steps = self.log_every_n_steps
        c.val_check_interval = self.val_check_interval
        c.checkpoint_dir = self.checkpoint_dir
        c.save_every_n_epochs = self.save_every_n_epochs
        c.enable_progress_bar = self.enable_progress_bar
        c.gradient_clip_val = self.gradient_clip_val
        c.accumulate_grad_batches = self.accumulate_grad_batches
        return c


class LightningModule(_LightningModuleBase):
    """Base class. Override `training_step` and `configure_optimizer`."""
    def training_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        return None

    def configure_optimizer(self):
        raise NotImplementedError

    # Lifecycle hooks (no-ops by default, override as needed)
    def on_train_epoch_start(self, epoch: int): pass
    def on_train_epoch_end(self, epoch: int, avg_loss: float): pass
    def on_validation_end(self, epoch: int, avg_val_loss: float): pass


def _loss_to_float(x) -> float:
    """Extract a Python float from a loss tensor or numeric."""
    if x is None:
        return 0.0
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


class Trainer:
    """Driver loop. Accepts any iterable `loader`."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.global_step = 0

    def fit(self,
            module: LightningModule,
            train_loader: Iterable,
            val_loader: Optional[Iterable] = None):
        cfg = self.config
        optimizer = module.configure_optimizer()
        if optimizer is None:
            raise RuntimeError("configure_optimizer() returned None")

        if cfg.checkpoint_dir:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        for epoch in range(1, cfg.max_epochs + 1):
            try: module.train(True)
            except Exception: pass
            module.on_train_epoch_start(epoch)

            epoch_loss = 0.0
            n_batches = 0
            t_epoch = time.time()

            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                loss = module.training_step(batch, batch_idx)
                if loss is None:
                    raise RuntimeError("training_step returned None")

                # Scale for accumulation
                if cfg.accumulate_grad_batches > 1 and hasattr(loss, "mul_"):
                    try: loss.mul_(1.0 / cfg.accumulate_grad_batches)
                    except Exception: pass

                # Backward — backward() lives on the tensor in PromeTorch
                if hasattr(loss, "backward"):
                    try: loss.backward()
                    except Exception: pass

                lv = _loss_to_float(loss) * (
                    cfg.accumulate_grad_batches if cfg.accumulate_grad_batches > 1 else 1)
                epoch_loss += lv
                n_batches += 1

                do_step = (batch_idx + 1) % cfg.accumulate_grad_batches == 0
                if do_step and hasattr(optimizer, "step"):
                    try: optimizer.step()
                    except Exception: pass
                    if hasattr(optimizer, "zero_grad"):
                        optimizer.zero_grad()
                    self.global_step += 1

                if cfg.enable_progress_bar and ((batch_idx + 1) % cfg.log_every_n_steps == 0):
                    print(f"epoch {epoch} step {batch_idx+1} loss {lv:.4f}")

            avg = epoch_loss / max(1, n_batches)
            dt = time.time() - t_epoch
            print(f"[epoch {epoch}] avg_loss={avg:.4f} time={dt:.1f}s steps={n_batches}")
            module.on_train_epoch_end(epoch, avg)

            if val_loader is not None:
                self._run_validation(module, val_loader, epoch)

            if cfg.save_every_n_epochs > 0 and (epoch % cfg.save_every_n_epochs == 0):
                path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}.ptor")
                try:
                    self.save_checkpoint(module, path)
                    print(f"[epoch {epoch}] saved {path}")
                except Exception as e:
                    print(f"[epoch {epoch}] save failed: {e}")

    def _run_validation(self, module, val_loader, epoch):
        try: module.eval()
        except Exception: pass
        total = 0.0
        n = 0
        for idx, batch in enumerate(val_loader):
            m = module.validation_step(batch, idx)
            if m is not None:
                total += _loss_to_float(m)
                n += 1
        avg = total / max(1, n)
        print(f"[epoch {epoch}][val] avg={avg:.4f} batches={n}")
        module.on_validation_end(epoch, avg)

    def test(self, module: LightningModule, test_loader: Iterable):
        self._run_validation(module, test_loader, epoch=-1)

    def save_checkpoint(self, module: LightningModule, path: str):
        try:
            from promethorch import save_state_dict
            save_state_dict(module.state_dict(), path)
        except Exception as e:
            raise RuntimeError(f"save_checkpoint failed: {e}")

    def load_checkpoint(self, module: LightningModule, path: str) -> bool:
        try:
            from promethorch import load_state_dict
            sd = load_state_dict(path)
            module.load_state_dict(sd, strict=False)
            return True
        except Exception:
            return False


__all__ = ["TrainerConfig", "LightningModule", "Trainer"]
