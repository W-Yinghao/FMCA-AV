"""Independent Lightning entry point for fair SSL baselines."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from .baselines import BaselineSSL
from .config import load_config
from .data.cifar import CIFARDataModule, CIFARProbeDataModule
from .data.imagenet import ImageNetDataModule, ImageNetProbeDataModule
from .data.small_vision import SmallVisionDataModule, SmallVisionProbeDataModule
from .operators import SCIENTIFIC_CORRECTNESS_VERSION
from .probe_module import FineTuneClassifier, LinearProbe
from .profiling import BatchTimingRecorder, ExecutedStepRecorder


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_directory(output: str = "") -> Path:
    if output:
        path = Path(output).resolve()
    elif os.environ.get("FMCA_HARNESS_RUN_DIR"):
        path = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    else:
        path = Path("experiment_runs") / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S_baseline")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_metric(payload: Dict[str, Any]) -> None:
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir and int(os.environ.get("RANK", "0")) == 0:
        record = {"time": datetime.now(timezone.utc).isoformat(), **payload}
        with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _data_module(config: Dict[str, Any], probe: bool = False) -> Any:
    data = config["data"]
    dataset = str(data["dataset"])
    seed = int(config["seed"]) + (3000 if probe else 0)
    if dataset in {"cifar10", "cifar100"}:
        return CIFARProbeDataModule(data, config.get("probe", {}), seed) if probe else CIFARDataModule(data, seed)
    if dataset in {"stl10", "tinyimagenet200"}:
        return SmallVisionProbeDataModule(data, config.get("probe", {}), seed) if probe else SmallVisionDataModule(data, seed)
    return ImageNetProbeDataModule(data, config.get("probe", {}), seed) if probe else ImageNetDataModule(data, seed)


def _classes(dataset: str) -> int:
    return {
        "cifar10": 10,
        "cifar100": 100,
        "stl10": 10,
        "tinyimagenet200": 200,
        "imagenet100": 100,
        "imagenet1k": 1000,
    }[dataset]


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not config["experiment"].get("method"):
        raise ValueError("baseline config requires experiment.method")
    L.seed_everything(int(config["seed"]), workers=True)
    run_dir = _run_directory(args.output)
    data = _data_module(config)
    data.setup()
    model = BaselineSSL(config)
    callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_score:.6f}",
        monitor="val_score",
        mode="max",
        save_last=True,
        save_top_k=int(config["trainer"].get("checkpoint_save_top_k", 1)),
        auto_insert_metric_name=False,
    )
    trainer_config = config["trainer"]
    step_recorder = ExecutedStepRecorder()
    callbacks: list[Callback] = [callback, step_recorder]
    if bool(trainer_config.get("profile_batches", False)):
        callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(trainer_config.get("accelerator", "gpu")),
        devices=trainer_config.get("devices", 1),
        strategy=str(trainer_config.get("strategy", "auto")),
        max_epochs=int(trainer_config["max_epochs"]),
        precision=str(trainer_config.get("precision", "32-true")),
        deterministic=bool(trainer_config.get("deterministic", True)),
        gradient_clip_val=trainer_config.get("gradient_clip_val"),
        logger=CSVLogger(save_dir=str(run_dir), name="lightning_logs"),
        callbacks=callbacks,
        enable_progress_bar=bool(trainer_config.get("enable_progress_bar", False)),
        log_every_n_steps=int(trainer_config.get("log_every_n_steps", 20)),
        limit_train_batches=trainer_config.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_config.get("limit_val_batches", 1.0),
        max_steps=int(trainer_config.get("max_steps", -1)),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
        ckpt_path=args.resume or None,
    )
    training_duration = time.perf_counter() - training_started
    if trainer.is_global_zero:
        result = {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "method": config["experiment"]["method"],
            "config_path": str(Path(args.config).resolve()),
            "dataset_name": str(config["data"]["dataset"]),
            "dataset_sizes": {name: len(dataset) for name, dataset in data.datasets.items()},
            "seed": int(config["seed"]),
            "claim_id": str(config["experiment"].get("claim_id", "")),
            "best_checkpoint": callback.best_model_path or callback.last_model_path,
            "last_checkpoint": callback.last_model_path,
            "best_validation_score": float(callback.best_model_score) if callback.best_model_score is not None else None,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        world_size = int(getattr(trainer, "world_size", 1)); global_step = int(trainer.global_step)
        completed_steps = max(0, global_step - int(step_recorder.start_step))
        batch_size = int(config.get("data", {}).get("batch_size", 0)); views = int(config.get("data", {}).get("num_views", 1))
        encoded_views = completed_steps * batch_size * world_size * views
        result.update({
            "training_duration_seconds": training_duration, "completed_optimizer_steps": completed_steps,
            "global_optimizer_step": global_step, "restored_optimizer_step": int(step_recorder.start_step),
            "world_size": world_size, "global_parent_batch_size": batch_size * world_size,
            "encoded_views": encoded_views,
            "encoded_views_per_second": encoded_views / training_duration if training_duration > 0 else None,
            "gpu_hours": training_duration * world_size / 3600.0,
            "peak_memory_mb_per_rank": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        })
        _write_json(run_dir / "train_result.json", result)
        _append_metric({"stage": "baseline_train", **result})
        print(json.dumps(result, indent=2))
    return 0


def linear_probe(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]) + 3000, workers=True)
    run_dir = _run_directory(args.output)
    source = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu")
    probe_config = config.get("probe", {})
    model = LinearProbe(
        source.backbone,
        source.backbone.output_dim,
        _classes(str(config["data"]["dataset"])),
        probe_config,
    )
    data = _data_module(config, probe=True)
    data.setup()
    callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_accuracy:.6f}",
        monitor="val_accuracy",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    probe_callbacks: list[Callback] = [callback]
    if bool(probe_config.get("profile_batches", False)):
        probe_callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(probe_config.get("accelerator", "gpu")),
        devices=probe_config.get("devices", 1),
        max_epochs=int(probe_config.get("max_epochs", 100)),
        precision=str(probe_config.get("precision", "32-true")),
        deterministic=True,
        logger=CSVLogger(save_dir=str(run_dir), name="lightning_logs"),
        callbacks=probe_callbacks,
        enable_progress_bar=bool(probe_config.get("enable_progress_bar", False)),
        log_every_n_steps=int(probe_config.get("log_every_n_steps", 20)),
        limit_train_batches=probe_config.get("limit_train_batches", 1.0),
        limit_val_batches=probe_config.get("limit_val_batches", 1.0),
        limit_test_batches=probe_config.get("limit_test_batches", 1.0),
    )
    trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
    test_results = trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path="best")
    if trainer.is_global_zero:
        metrics = test_results[0]
        result = {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "method": config["experiment"]["method"],
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "probe_checkpoint": callback.best_model_path,
            "best_validation_accuracy": float(callback.best_model_score),
            "test_accuracy": float(metrics["test/accuracy"]),
            "test_top5_accuracy": float(metrics["test/top5_accuracy"]),
            "protocol": "frozen backbone, single linear layer",
            "label_fraction": float(probe_config.get("label_fraction", 1.0)),
        }
        _write_json(run_dir / "probe_result.json", result)
        _append_metric({"stage": "baseline_linear_probe", **result})
        print(json.dumps(result, indent=2))
    return 0


def fine_tune(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]) + 4000, workers=True)
    run_dir = _run_directory(args.output)
    source = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu")
    probe_config = config.get("probe", {})
    model = FineTuneClassifier(
        source.backbone,
        source.backbone.output_dim,
        _classes(str(config["data"]["dataset"])),
        probe_config,
    )
    data = _data_module(config, probe=True); data.setup()
    callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", filename="best-{epoch:03d}-{val_accuracy:.6f}",
        monitor="val_accuracy", mode="max", save_last=True, auto_insert_metric_name=False,
    )
    finetune_callbacks: list[Callback] = [callback]
    if bool(probe_config.get("profile_batches", False)):
        finetune_callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(probe_config.get("accelerator", "gpu")), devices=probe_config.get("devices", 1),
        max_epochs=int(probe_config.get("max_epochs", 100)), precision=str(probe_config.get("precision", "32-true")),
        deterministic=True, logger=CSVLogger(save_dir=str(run_dir), name="lightning_logs"), callbacks=finetune_callbacks,
        enable_progress_bar=bool(probe_config.get("enable_progress_bar", False)),
        log_every_n_steps=int(probe_config.get("log_every_n_steps", 20)),
        limit_train_batches=probe_config.get("limit_train_batches", 1.0),
        limit_val_batches=probe_config.get("limit_val_batches", 1.0),
        limit_test_batches=probe_config.get("limit_test_batches", 1.0),
        gradient_clip_val=probe_config.get("gradient_clip_val"),
    )
    trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
    test_results = trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path="best")
    if trainer.is_global_zero:
        metrics = test_results[0]
        result = {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "method": config["experiment"]["method"],
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "finetune_checkpoint": callback.best_model_path,
            "best_validation_accuracy": float(callback.best_model_score),
            "test_accuracy": float(metrics["test/accuracy"]),
            "test_top5_accuracy": float(metrics["test/top5_accuracy"]),
            "protocol": "end-to-end fine-tuning of pretrained baseline backbone and linear classifier",
            "label_fraction": float(probe_config.get("label_fraction", 1.0)),
        }
        _write_json(run_dir / "finetune_result.json", result)
        _append_metric({"stage": "baseline_fine_tune", **result})
        print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightning SSL baseline runner")
    children = parser.add_subparsers(required=True)
    train_parser = children.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output", default="")
    train_parser.add_argument("--resume", default="")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--overrides-json")
    train_parser.set_defaults(func=train)
    probe_parser = children.add_parser("linear-probe")
    probe_parser.add_argument("--config", required=True)
    probe_parser.add_argument("--checkpoint", required=True)
    probe_parser.add_argument("--output", default="")
    probe_parser.add_argument("--seed", type=int)
    probe_parser.add_argument("--overrides-json")
    probe_parser.set_defaults(func=linear_probe)
    finetune_parser = children.add_parser("fine-tune")
    finetune_parser.add_argument("--config", required=True)
    finetune_parser.add_argument("--checkpoint", required=True)
    finetune_parser.add_argument("--output", default="")
    finetune_parser.add_argument("--seed", type=int)
    finetune_parser.add_argument("--overrides-json")
    finetune_parser.set_defaults(func=fine_tune)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed is not None:
        os.environ["FMCA_SEED_OVERRIDE"] = str(args.seed)
    if args.overrides_json is not None:
        overrides = json.loads(args.overrides_json)
        if not isinstance(overrides, dict):
            raise ValueError("--overrides-json must decode to a JSON object")
        os.environ["FMCA_CONFIG_OVERRIDES"] = json.dumps(overrides, separators=(",", ":"))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
