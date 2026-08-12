"""Shared, opt-in Lightning training instrumentation.

The callbacks in this module only observe training.  They do not alter the
model, optimizer, dataloader, precision, or distributed strategy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import lightning as L
from lightning.pytorch.callbacks import Callback
import torch


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class ExecutedStepRecorder(Callback):
    """Capture the restored global step so resumed chunks report new work only."""

    def __init__(self) -> None:
        super().__init__()
        self.start_step = 0

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        self.start_step = int(trainer.global_step)


class MilestoneCheckpoint(Callback):
    """Save explicitly preregistered epoch checkpoints without changing selection."""

    def __init__(self, directory: Path, epochs: List[int]) -> None:
        super().__init__()
        self.directory = directory
        self.epochs = {int(value) for value in epochs}
        if any(value < 1 for value in self.epochs):
            raise ValueError("checkpoint milestones must be positive epochs")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch not in self.epochs:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.directory / f"epoch-{completed_epoch:04d}.ckpt")


class BatchTimingRecorder(Callback):
    """Persist device-compute and inter-batch timing for opt-in profiling runs."""

    def __init__(self, output: Path, flush_every: int = 100) -> None:
        super().__init__()
        self.output = output
        self.flush_every = max(1, int(flush_every))
        self.compute_seconds: List[float] = []
        self.data_wait_seconds: List[float] = []
        self.compute_started: Optional[float] = None
        self.previous_batch_ended: Optional[float] = None
        self.train_started_at: Optional[str] = None

    @staticmethod
    def _summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "count": 0.0, "total": 0.0, "mean": 0.0,
                "p50": 0.0, "p95": 0.0, "maximum": 0.0,
            }
        ordered = sorted(values)
        count = len(ordered)
        return {
            "count": float(count),
            "total": float(sum(ordered)),
            "mean": float(sum(ordered) / count),
            "p50": float(ordered[(count - 1) // 2]),
            "p95": float(ordered[min(count - 1, int(count * 0.95))]),
            "maximum": float(ordered[-1]),
        }

    def _save(self, trainer: L.Trainer) -> None:
        if not trainer.is_global_zero:
            return
        compute = self._summary(self.compute_seconds)
        wait = self._summary(self.data_wait_seconds)
        total = compute["total"] + wait["total"]
        batches_per_epoch_value = getattr(trainer, "num_training_batches", None)
        try:
            batches_per_epoch = int(batches_per_epoch_value)
        except (TypeError, ValueError, OverflowError):
            batches_per_epoch = None
        seconds_per_batch = total / len(self.compute_seconds) if self.compute_seconds else None
        _atomic_json(self.output, {
            # Kept for compatibility with the first profiler output schema.
            "optimizer_steps_observed": len(self.compute_seconds),
            "train_batches_observed": len(self.compute_seconds),
            "compute_seconds": compute,
            "inter_batch_wait_seconds": wait,
            "observed_seconds": total,
            "compute_fraction": compute["total"] / total if total else None,
            "inter_batch_wait_fraction": wait["total"] / total if total else None,
            "global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "batches_per_epoch": batches_per_epoch,
            "seconds_per_observed_batch": seconds_per_batch,
            "projected_epoch_seconds": (
                seconds_per_batch * batches_per_epoch
                if seconds_per_batch is not None and batches_per_epoch is not None else None
            ),
            "train_start_time": self.train_started_at,
            "last_update_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        })

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del trainer, pl_module
        self.train_started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        batch: Any, batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        now = time.perf_counter()
        if self.previous_batch_ended is not None:
            self.data_wait_seconds.append(now - self.previous_batch_ended)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.compute_started = time.perf_counter()

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.compute_started is not None:
            self.compute_seconds.append(time.perf_counter() - self.compute_started)
        self.previous_batch_ended = time.perf_counter()
        if (batch_idx + 1) % self.flush_every == 0:
            self._save(trainer)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        self._save(trainer)

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module, exception
        self._save(trainer)
