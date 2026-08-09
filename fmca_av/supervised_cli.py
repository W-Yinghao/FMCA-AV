"""Lightning supervised-reference training for transfer and map controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .backbones import build_backbone
from .baseline_cli import _classes, _data_module
from .config import load_config
from .operators import SCIENTIFIC_CORRECTNESS_VERSION
from .profiling import BatchTimingRecorder, ExecutedStepRecorder


class RandomLabelDataset(Dataset):
    """Assign one reproducible random class to every training example."""

    def __init__(self, base: Dataset, classes: int, seed: int) -> None:
        self.base = base
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(classes, (len(base),), generator=generator)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        sample = self.base[index]
        return sample[0], self.labels[index]


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class SupervisedVision(L.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(); self.save_hyperparameters({"config": config})
        model = config["model"]
        self.backbone = build_backbone(str(model.get("backbone", "resnet18_cifar")), width=int(model.get("backbone_width", 64)))
        self.classifier = nn.Linear(self.backbone.output_dim, _classes(str(config["data"]["dataset"])))

    @property
    def config(self) -> Dict[str, Any]: return self.hparams["config"]

    def forward(self, images: Tensor) -> Tensor: return self.classifier(self.backbone(images))

    def _step(self, batch: Tuple[Tensor, Tensor], split: str) -> Tensor:
        images, labels = batch; logits = self(images); loss = torch.nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(1) == labels).float().mean(); top = min(5, logits.shape[1])
        top5 = (logits.topk(top, 1).indices == labels[:, None]).any(1).float().mean()
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{split}/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{split}/top5_accuracy", top5, on_step=False, on_epoch=True)
        if split == "val": self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor: return self._step(batch, "train")
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None: self._step(batch, "val")
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None: self._step(batch, "test")

    def configure_optimizers(self) -> object:
        values = self.config["optimizer"]
        optimizer = torch.optim.SGD(
            self.parameters(), lr=float(values.get("learning_rate", 0.1)),
            momentum=float(values.get("momentum", 0.9)), weight_decay=float(values.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.config["trainer"]["max_epochs"]))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config); L.seed_everything(int(config["seed"]) + 15000, workers=True)
    run_dir = Path(args.output).resolve() if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; run_dir.mkdir(parents=True, exist_ok=True)
    data = _data_module(config, probe=True); data.setup()
    random_labels = bool(config["experiment"].get("random_labels", False))
    if random_labels:
        dataset_name = str(config["data"]["dataset"])
        data.datasets["train"] = RandomLabelDataset(
            data.datasets["train"], _classes(dataset_name), int(config["seed"]) + 15100,
        )
    model = SupervisedVision(config)
    callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", filename="best-{epoch:03d}-{val_accuracy:.6f}",
        monitor="val_accuracy", mode="max", save_last=True, auto_insert_metric_name=False,
    )
    trainer_config = config["trainer"]
    step_recorder = ExecutedStepRecorder()
    callbacks: list[Callback] = [callback, step_recorder]
    if bool(trainer_config.get("profile_batches", False)):
        callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(trainer_config.get("accelerator", "gpu")), devices=trainer_config.get("devices", 1),
        strategy=str(trainer_config.get("strategy", "auto")), max_epochs=int(trainer_config["max_epochs"]),
        max_steps=int(trainer_config.get("max_steps", -1)), precision=str(trainer_config.get("precision", "32-true")),
        deterministic=bool(trainer_config.get("deterministic", True)), gradient_clip_val=trainer_config.get("gradient_clip_val"),
        limit_train_batches=trainer_config.get("limit_train_batches", 1.0), limit_val_batches=trainer_config.get("limit_val_batches", 1.0),
        limit_test_batches=trainer_config.get("limit_test_batches", trainer_config.get("limit_val_batches", 1.0)),
        callbacks=callbacks, logger=CSVLogger(save_dir=str(run_dir), name="lightning_logs"),
        enable_progress_bar=bool(trainer_config.get("enable_progress_bar", False)), log_every_n_steps=int(trainer_config.get("log_every_n_steps", 20)),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    train_loader = data.train_dataloader()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=data.val_dataloader())
    training_duration = time.perf_counter() - training_started
    metrics = trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path="best")[0]
    if not trainer.is_global_zero:
        return 0
    result = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "method": "supervised_random_labels" if random_labels else "supervised",
        "random_labels": random_labels, "dataset": str(config["data"]["dataset"]),
        "config_path": str(Path(args.config).resolve()),
        "dataset_sizes": {name: len(dataset) for name, dataset in data.datasets.items()},
        "seed": int(config["seed"]),
        "claim_id": str(config["experiment"].get("claim_id", "")),
        "best_checkpoint": callback.best_model_path or callback.last_model_path, "last_checkpoint": callback.last_model_path,
        "best_validation_accuracy": float(callback.best_model_score) if callback.best_model_score is not None else None,
        "test_accuracy": float(metrics["test/accuracy"]), "test_top5_accuracy": float(metrics["test/top5_accuracy"]),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    world_size = int(getattr(trainer, "world_size", 1)); global_step = int(trainer.global_step)
    completed_steps = max(0, global_step - int(step_recorder.start_step))
    batch_size = int(getattr(train_loader, "batch_size", 0) or config.get("data", {}).get("batch_size", 0))
    processed_images = completed_steps * batch_size * world_size
    result.update({
        "training_duration_seconds": training_duration,
        "completed_optimizer_steps": completed_steps,
        "global_optimizer_step": global_step,
        "restored_optimizer_step": int(step_recorder.start_step),
        "world_size": world_size,
        "global_batch_size": batch_size * world_size,
        "processed_images": processed_images,
        "images_per_second": processed_images / training_duration if training_duration > 0 else None,
        "gpu_hours": training_duration * world_size / 3600.0,
        "peak_memory_mb_per_rank": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    })
    atomic_json(run_dir / "supervised_result.json", result)
    harness_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if harness_dir:
        with (Path(harness_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": "supervised_train", **result}, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output", default="")
    return train(parser.parse_args())


if __name__ == "__main__": raise SystemExit(main())
