"""Unified train, calibrate, and evaluate entry point for FMCA-AV."""

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from .analytic import finite_channel_spectrum
from .config import load_config
from .data.cifar import (
    CIFARDataModule,
    CIFARCorruptionDataset,
    CIFARFiles,
    CIFARProbeDataModule,
    CIFARProbeTransform,
    LabeledCIFARDataset,
)
from .data.finite import FiniteDataModule, sample_finite_conditionals
from .data.factors import FactorDataModule
from .data.gaussian import GaussianDataModule, gaussian_product_eigenvalues, sample_conditionals
from .data.imagenet import (
    ImageNetDataModule,
    ImageNetFiles,
    ImageNetProbeDataModule,
    ImageNetProbeTransform,
    LabeledImageDataset,
)
from .data.small_vision import (
    SmallVisionDataModule,
    SmallVisionProbeDataModule,
    small_vision_base_datasets,
)
from .lightning_module import GaussianFMCAAV
from .knn import weighted_knn_accuracy, weighted_knn_accuracy_chunked
from .finite_module import FiniteFMCAAV
from .operators import (
    SCIENTIFIC_CORRECTNESS_VERSION,
    calibration_to_state,
    evaluate_heldout_spectrum,
    fit_spectral_calibration,
)
from .probe_module import FineTuneClassifier, LinearProbe
from .profiling import BatchTimingRecorder, ExecutedStepRecorder
from .robustness import (
    ClassificationMetricAccumulator,
    IMAGENET_ALEXNET_CLEAN_ERROR,
    IMAGENET_C_ALEXNET_ERRORS,
    WNIDFolderDataset,
)
from .vision_module import VisionFMCAAV
from torch.utils.data import DataLoader


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _run_directory(config: Dict[str, Any], requested: str = "") -> Path:
    if requested:
        directory = Path(requested).resolve()
        exclusive = True
    elif os.environ.get("FMCA_HARNESS_RUN_DIR"):
        directory = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
        exclusive = False
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = str(config["experiment"]["name"])
        directory = (Path(config.get("output_root", "experiment_runs")) / f"{stamp}_{name}").resolve()
        exclusive = int(os.environ.get("WORLD_SIZE", "1")) == 1
    directory.mkdir(parents=True, exist_ok=not exclusive)
    return directory


def _global_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _append_harness_metric(payload: Dict[str, Any]) -> None:
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if not run_dir or not _global_process():
        return
    record = {"time": datetime.now(timezone.utc).isoformat(), **payload}
    with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _data_module(config: Dict[str, Any]) -> Any:
    family = config["experiment"]["family"]
    if family == "gaussian_1d":
        module = GaussianDataModule(config["data"], int(config["seed"]))
    elif family == "finite_channel":
        module = FiniteDataModule(config["data"], int(config["seed"]))
    elif family in {"cifar_ssl", "vision_ssl"}:
        dataset = str(config["data"]["dataset"])
        if dataset in {"cifar10", "cifar100"}:
            module = CIFARDataModule(config["data"], int(config["seed"]))
        elif dataset in {"stl10", "tinyimagenet200"}:
            module = SmallVisionDataModule(config["data"], int(config["seed"]))
        elif dataset in {"imagenet1k", "imagenet100"}:
            module = ImageNetDataModule(config["data"], int(config["seed"]))
        elif dataset in {"dsprites", "shapes3d", "smallnorb", "mpi3d_toy", "mpi3d_realistic", "mpi3d_real"}:
            module = FactorDataModule(config["data"], int(config["seed"]))
        else:
            raise ValueError(f"unsupported vision dataset: {dataset}")
    else:
        raise ValueError(f"unsupported experiment family: {family}")
    module.setup()
    return module


def train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]), workers=True)
    run_dir = _run_directory(config, args.output)
    if _global_process():
        _write_json(run_dir / "config.json", {k: v for k, v in config.items() if not k.startswith("_")})
    data = _data_module(config)
    if config["experiment"]["family"] == "gaussian_1d":
        model = GaussianFMCAAV(config)
    elif config["experiment"]["family"] == "finite_channel":
        model = FiniteFMCAAV(config)
    else:
        model = VisionFMCAAV(config)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_score:.6f}",
        monitor="val/score",
        mode="max",
        save_last=True,
        save_top_k=int(config["trainer"].get("checkpoint_save_top_k", 1)),
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    step_recorder = ExecutedStepRecorder()
    trainer_config = config["trainer"]
    callbacks: List[Callback] = [checkpoint, step_recorder]
    if bool(trainer_config.get("profile_batches", False)):
        callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(trainer_config.get("accelerator", "auto")),
        devices=trainer_config.get("devices", 1),
        max_epochs=int(trainer_config["max_epochs"]),
        precision=str(trainer_config.get("precision", "32-true")),
        deterministic=bool(trainer_config.get("deterministic", True)),
        strategy=str(trainer_config.get("strategy", "auto")),
        gradient_clip_val=trainer_config.get("gradient_clip_val"),
        log_every_n_steps=int(trainer_config.get("log_every_n_steps", 10)),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=bool(trainer_config.get("enable_progress_bar", False)),
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
    result = {
        "run_dir": str(run_dir),
        "config_path": str(Path(args.config).resolve()),
        "dataset_name": str(config.get("data", {}).get("dataset", config["experiment"]["family"])),
        "dataset_sizes": {name: len(dataset) for name, dataset in data.datasets.items()},
        "seed": int(config["seed"]),
        "claim_id": str(config["experiment"].get("claim_id", "")),
        "best_checkpoint": checkpoint.best_model_path,
        "best_validation_score": float(checkpoint.best_model_score) if checkpoint.best_model_score else None,
        "last_checkpoint": checkpoint.last_model_path,
    }
    if trainer.is_global_zero:
        result["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        result["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        world_size = int(getattr(trainer, "world_size", 1)); global_step = int(trainer.global_step)
        completed_steps = max(0, global_step - int(step_recorder.start_step))
        batch_size = int(config.get("data", {}).get("batch_size", 0))
        views = int(config.get("data", {}).get("num_views", 1))
        views += int(bool(config.get("data", {}).get("include_raw_parent", False)))
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
        result["spectral_convention"] = "eigenvalues are squared canonical singular values; constant mode removed by centering; ridge is relative to mean marginal variance"
        result["scientific_correctness_version"] = SCIENTIFIC_CORRECTNESS_VERSION
        _write_json(run_dir / "train_result.json", result)
        _append_harness_metric({"stage": "train", **result})
        print(json.dumps(result, indent=2))
    return 0


def _load_model(config: Dict[str, Any], checkpoint: str, device: torch.device) -> Any:
    if config["experiment"]["family"] == "gaussian_1d":
        model = GaussianFMCAAV.load_from_checkpoint(checkpoint, config=config, map_location=device)
    elif config["experiment"]["family"] == "finite_channel":
        model = FiniteFMCAAV.load_from_checkpoint(checkpoint, config=config, map_location=device)
    else:
        model = VisionFMCAAV.load_from_checkpoint(checkpoint, config=config, map_location=device)
    model.to(device)
    model.eval()
    return model


def _collect_features(
    model: Any,
    loader: Iterable[Any],
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_f: List[torch.Tensor] = []
    all_g: List[torch.Tensor] = []
    data = config["data"]
    with torch.inference_mode():
        for batch in loader:
            if config["experiment"]["family"] == "gaussian_1d":
                x = batch[0].to(device)
                y = sample_conditionals(x, int(data["num_views"]), float(data["noise_variance"]))
                f_features, g_features = model.feature_maps(x, y)
            elif config["experiment"]["family"] == "finite_channel":
                x = batch[0].to(device)
                y = sample_finite_conditionals(x, model.conditional, int(data["num_views"]))
                f_features, g_features = model.feature_maps(x, y)
            else:
                views = batch[0].to(device)
                parent_view = batch[3].to(device) if len(batch) > 3 else None
                f_features, g_features, _ = model.feature_maps(views, parent_view)
            all_f.append(f_features.cpu())
            all_g.append(g_features.cpu())
    return torch.cat(all_f), torch.cat(all_g)


def calibrate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]) + 1000, workers=True)
    device = torch.device(args.device)
    data = _data_module(config)
    model = _load_model(config, args.checkpoint, device)
    f_features, g_features = _collect_features(model, data.calibration_dataloader(), config, device)
    objective = config["objective"]
    calibration = fit_spectral_calibration(
        f_features.double(),
        g_features.double(),
        ridge=float(objective.get("ridge", 1e-3)),
        centered=bool(config["model"].get("centered", True)),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration_to_state(calibration), output)
    metrics = calibration.as_metrics()
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["calibration_file"] = str(output)
    _write_json(output.with_suffix(".json"), metrics)
    _append_harness_metric({"stage": "calibrate", **metrics})
    print(json.dumps(metrics, indent=2))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]) + 2000, workers=True)
    device = torch.device(args.device)
    data = _data_module(config)
    model = _load_model(config, args.checkpoint, device)
    f_features, g_features = _collect_features(model, data.test_dataloader(), config, device)
    state = torch.load(args.calibration, map_location="cpu", weights_only=True)
    heldout = evaluate_heldout_spectrum(f_features.double(), g_features.double(), state)
    empirical = heldout.singular_values
    empirical_eigenvalues = heldout.eigenvalues
    result = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "calibration_scientific_correctness_version": state.get(
            "scientific_correctness_version", "pre_fix_unversioned",
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "calibration": str(Path(args.calibration).resolve()),
        "heldout_spectral_method": "svdvals_of_full_canonical_cross_operator",
        "test_canonical_cross_operator": heldout.cross_operator.tolist(),
        "test_empirical_singular_values": empirical.tolist(),
        "test_empirical_eigenvalues": empirical_eigenvalues.tolist(),
        "test_diagonal_correlations": heldout.diagonal_correlations.tolist(),
    }
    if config["experiment"]["family"] == "gaussian_1d":
        truth = gaussian_product_eigenvalues(
            float(config["data"]["noise_variance"]),
            int(config["data"].get("dimension", 1)),
            empirical.numel(),
        )
        count = min(empirical_eigenvalues.numel(), truth.numel())
        absolute_error = (empirical_eigenvalues[:count] - truth[:count]).abs()
        result.update({
            "ground_truth_eigenvalues": truth.tolist(),
            "eigenvalue_mae": float(absolute_error.mean()),
            "eigenvalue_max_error": float(absolute_error.max()),
        })
    elif config["experiment"]["family"] == "finite_channel":
        truth_spectrum = finite_channel_spectrum(torch.tensor(config["data"]["joint_probability"]))
        truth = truth_spectrum.eigenvalues
        count = min(empirical_eigenvalues.numel(), truth.numel())
        absolute_error = (empirical_eigenvalues[:count] - truth[:count]).abs()
        result.update({
            "ground_truth_eigenvalues": truth.tolist(),
            "eigenvalue_mae": float(absolute_error.mean()),
            "eigenvalue_max_error": float(absolute_error.max()),
        })
    output = Path(args.output).resolve()
    _write_json(output, result)
    _append_harness_metric({"stage": "evaluate", **result})
    print(json.dumps(result, indent=2))
    return 0


def analytic(args: argparse.Namespace) -> int:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    spectrum = finite_channel_spectrum(torch.tensor(config["joint_probability"]))
    result = {"name": config["name"], **spectrum.as_metrics()}
    if args.output:
        _write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


def linear_probe(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config["experiment"]["family"] not in {"cifar_ssl", "vision_ssl"}:
        raise ValueError("linear-probe requires a vision SSL configuration")
    probe_config = config.get("probe", {})
    L.seed_everything(int(config["seed"]) + 3000, workers=True)
    run_dir = _run_directory(config, args.output)
    source = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu")
    dataset = str(config["data"]["dataset"])
    classes = {
        "cifar10": 10,
        "cifar100": 100,
        "stl10": 10,
        "tinyimagenet200": 200,
        "imagenet100": 100,
        "imagenet1k": 1000,
    }[dataset]
    model = LinearProbe(source.backbone, source.backbone.output_dim, classes, probe_config)
    if dataset in {"cifar10", "cifar100"}:
        data = CIFARProbeDataModule(config["data"], probe_config, int(config["seed"]) + 3000)
    elif dataset in {"stl10", "tinyimagenet200"}:
        data = SmallVisionProbeDataModule(config["data"], probe_config, int(config["seed"]) + 3000)
    else:
        data = ImageNetProbeDataModule(config["data"], probe_config, int(config["seed"]) + 3000)
    data.setup()
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_accuracy:.6f}",
        monitor="val_accuracy",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    probe_callbacks: List[Callback] = [checkpoint]
    if bool(probe_config.get("profile_batches", False)):
        probe_callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(probe_config.get("accelerator", "gpu")),
        devices=probe_config.get("devices", 1),
        max_epochs=int(probe_config.get("max_epochs", 100)),
        precision=str(probe_config.get("precision", "32-true")),
        deterministic=True,
        logger=logger,
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
        test_metrics = test_results[0]
        result = {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "probe_checkpoint": checkpoint.best_model_path,
            "best_validation_accuracy": float(checkpoint.best_model_score),
            "test_accuracy": float(test_metrics["test/accuracy"]),
            "test_top5_accuracy": float(test_metrics["test/top5_accuracy"]),
            "protocol": "frozen backbone, single linear layer",
            "label_fraction": float(probe_config.get("label_fraction", 1.0)),
        }
        _write_json(run_dir / "probe_result.json", result)
        _append_harness_metric({"stage": "linear_probe", **result})
        print(json.dumps(result, indent=2))
    return 0


def fine_tune(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config["experiment"]["family"] not in {"cifar_ssl", "vision_ssl"}:
        raise ValueError("fine-tune requires a vision SSL configuration")
    probe_config = config.get("probe", {})
    L.seed_everything(int(config["seed"]) + 4000, workers=True)
    run_dir = _run_directory(config, args.output)
    source = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu")
    dataset = str(config["data"]["dataset"])
    classes = {
        "cifar10": 10, "cifar100": 100, "stl10": 10,
        "tinyimagenet200": 200, "imagenet100": 100, "imagenet1k": 1000,
    }[dataset]
    model = FineTuneClassifier(source.backbone, source.backbone.output_dim, classes, probe_config)
    if dataset in {"cifar10", "cifar100"}:
        data = CIFARProbeDataModule(config["data"], probe_config, int(config["seed"]) + 4000)
    elif dataset in {"stl10", "tinyimagenet200"}:
        data = SmallVisionProbeDataModule(config["data"], probe_config, int(config["seed"]) + 4000)
    else:
        data = ImageNetProbeDataModule(config["data"], probe_config, int(config["seed"]) + 4000)
    data.setup()
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:03d}-{val_accuracy:.6f}",
        monitor="val_accuracy", mode="max", save_last=True, auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    finetune_callbacks: List[Callback] = [checkpoint]
    if bool(probe_config.get("profile_batches", False)):
        finetune_callbacks.append(BatchTimingRecorder(run_dir / "batch_profile.json"))
    trainer = L.Trainer(
        accelerator=str(probe_config.get("accelerator", "gpu")),
        devices=probe_config.get("devices", 1),
        max_epochs=int(probe_config.get("max_epochs", 100)),
        precision=str(probe_config.get("precision", "32-true")),
        deterministic=True, logger=logger, callbacks=finetune_callbacks,
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
        test_metrics = test_results[0]
        result = {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "finetune_checkpoint": checkpoint.best_model_path,
            "best_validation_accuracy": float(checkpoint.best_model_score),
            "test_accuracy": float(test_metrics["test/accuracy"]),
            "test_top5_accuracy": float(test_metrics["test/top5_accuracy"]),
            "protocol": "end-to-end fine-tuning of pretrained backbone and linear classifier",
            "label_fraction": float(probe_config.get("label_fraction", 1.0)),
        }
        _write_json(run_dir / "finetune_result.json", result)
        _append_harness_metric({"stage": "fine_tune", **result})
        print(json.dumps(result, indent=2))
    return 0


def knn_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config["experiment"]["family"] not in {"cifar_ssl", "vision_ssl"}:
        raise ValueError("knn requires a vision SSL configuration")
    device = torch.device(args.device)
    if config["experiment"].get("method"):
        from .baselines import BaselineSSL
        source = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    else:
        source = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    source.to(device).eval()
    data = config["data"]
    augmentation = data.get("augmentation", {})
    mean = augmentation.get("mean", [0.4914, 0.4822, 0.4465])
    std = augmentation.get("std", [0.2470, 0.2435, 0.2616])
    dataset_name = str(data["dataset"])
    if dataset_name in {"cifar10", "cifar100"}:
        clean = CIFARProbeTransform(False, mean, std)
        train_dataset = LabeledCIFARDataset(
            CIFARFiles(str(data["root"]), dataset_name, train=True), clean
        )
        test_dataset = LabeledCIFARDataset(
            CIFARFiles(str(data["root"]), dataset_name, train=False), clean
        )
    else:
        clean_large = ImageNetProbeTransform(False, augmentation)
        if dataset_name in {"stl10", "tinyimagenet200"}:
            train_base, test_base = small_vision_base_datasets(data, probe=True)
        else:
            classes_selected = data.get("class_wnids")
            if classes_selected is None and data.get("class_wnids_file"):
                classes_selected = [
                    line.strip() for line in Path(str(data["class_wnids_file"])).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            train_base = ImageNetFiles(
                str(data["root"]), "train", class_wnids=classes_selected,
                manifest=str(data.get("train_manifest", "")),
            )
            test_base = ImageNetFiles(
                str(data["root"]), "val", str(data["val_labels"]),
                class_wnids=classes_selected,
                manifest=str(data.get("val_manifest", "")),
            )
        train_dataset = LabeledImageDataset(train_base, clean_large)
        test_dataset = LabeledImageDataset(test_base, clean_large)
    workers = int(args.workers)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
    )
    classes = {
        "cifar10": 10,
        "cifar100": 100,
        "stl10": 10,
        "tinyimagenet200": 200,
        "imagenet100": 100,
        "imagenet1k": 1000,
    }[dataset_name]
    if dataset_name in {"cifar10", "cifar100"} and args.bank_limit == 0:
        accuracy = weighted_knn_accuracy(
            source.backbone,
            train_loader,
            test_loader,
            classes,
            device,
            neighbors=args.neighbors,
            temperature=args.temperature,
        )
        bank_samples = len(train_dataset)
    else:
        accuracy, bank_samples = weighted_knn_accuracy_chunked(
            source.backbone,
            train_loader,
            test_loader,
            classes,
            device,
            neighbors=args.neighbors,
            temperature=args.temperature,
            bank_chunk_size=args.bank_chunk_size,
            bank_limit=args.bank_limit,
        )
    result = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "method": config["experiment"].get("method", "fmca_av"),
        "knn_accuracy": accuracy,
        "neighbors": args.neighbors,
        "temperature": args.temperature,
        "bank_samples": bank_samples,
        "bank_limit": args.bank_limit,
        "representation": "frozen backbone",
    }
    output = Path(args.output).resolve() if args.output else _run_directory(config) / "knn_result.json"
    _write_json(output, result)
    _append_harness_metric({"stage": "knn", **result})
    print(json.dumps(result, indent=2))
    return 0


@torch.inference_mode()
def corruption_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config["experiment"]["family"] not in {"cifar_ssl", "vision_ssl"}:
        raise ValueError("corruption-eval requires a vision SSL configuration")
    if config["data"]["dataset"] not in {"cifar10", "cifar100"}:
        raise ValueError("this corruption-eval command currently supports CIFAR-C only")
    device = torch.device(args.device)
    if config["experiment"].get("method"):
        from .baselines import BaselineSSL
        source = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    else:
        source = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    classes = 10 if config["data"]["dataset"] == "cifar10" else 100
    probe = LinearProbe(source.backbone, source.backbone.output_dim, classes, config.get("probe", {}))
    probe_state = torch.load(args.probe_checkpoint, map_location=device, weights_only=True)["state_dict"]
    probe.load_state_dict(probe_state)
    probe.to(device).eval()
    augmentation = config["data"].get("augmentation", {})
    mean = augmentation.get("mean", [0.4914, 0.4822, 0.4465])
    std = augmentation.get("std", [0.2470, 0.2435, 0.2616])
    transform = CIFARProbeTransform(False, mean, std)
    root = Path(args.root).resolve()
    labels = root / "labels.npy"
    corruption_files = sorted(path for path in root.glob("*.npy") if path.name != "labels.npy")
    if not corruption_files:
        raise FileNotFoundError(f"no corruption arrays found under {root}")
    details: Dict[str, Dict[str, float]] = {}
    all_accuracies = []
    clean_dataset = LabeledCIFARDataset(
        CIFARFiles(str(config["data"]["root"]), str(config["data"]["dataset"]), train=False), transform,
    )
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                              persistent_workers=False, pin_memory=True)
    clean_correct = clean_total = 0
    for images, batch_labels in clean_loader:
        predictions = probe(images.to(device)).argmax(dim=1).cpu()
        clean_correct += int((predictions == batch_labels).sum()); clean_total += batch_labels.numel()
    clean_accuracy = clean_correct / clean_total
    for corruption_file in corruption_files:
        severity_results: Dict[str, float] = {}
        for severity in range(1, 6):
            dataset = CIFARCorruptionDataset(
                str(corruption_file), str(labels), severity, transform
            )
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                persistent_workers=False, pin_memory=True,
            )
            correct = 0
            total = 0
            for images, batch_labels in loader:
                predictions = probe(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == batch_labels).sum())
                total += batch_labels.numel()
            accuracy = correct / total
            severity_results[str(severity)] = accuracy
            all_accuracies.append(accuracy)
            del loader, dataset
        details[corruption_file.stem] = severity_results
    result = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "probe_checkpoint": str(Path(args.probe_checkpoint).resolve()),
        "mean_corruption_accuracy": sum(all_accuracies) / len(all_accuracies),
        "mean_corruption_error_percent": 100.0 * (1.0 - sum(all_accuracies) / len(all_accuracies)),
        "relative_mean_corruption_error_percent": 100.0 * (
            (1.0 - sum(all_accuracies) / len(all_accuracies)) - (1.0 - clean_accuracy)
        ),
        "clean_accuracy": clean_accuracy,
        "corruptions": details,
        "note": "CIFAR-C mCE is the unnormalized mean top-1 corruption error; relative mCE subtracts clean top-1 error",
    }
    corruption_output = (Path(args.output).resolve() if args.output else
                         _run_directory(config) / "cifar_corruption.json")
    _write_json(corruption_output, result)
    _append_harness_metric({
        "stage": "corruption_evaluate",
        "mean_corruption_accuracy": result["mean_corruption_accuracy"],
        "checkpoint": result["checkpoint"],
    })
    print(json.dumps(result, indent=2))
    return 0


@torch.inference_mode()
def imagenet_robustness_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset_name = str(config["data"]["dataset"])
    if dataset_name not in {"imagenet1k", "imagenet100"}:
        raise ValueError("imagenet-robustness requires an ImageNet configuration")
    device = torch.device(args.device)
    if config["experiment"].get("method"):
        from .baselines import BaselineSSL
        source = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    else:
        source = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    classes = 1000 if dataset_name == "imagenet1k" else 100
    probe = LinearProbe(source.backbone, source.backbone.output_dim, classes, config.get("probe", {}))
    probe_state = torch.load(args.probe_checkpoint, map_location=device, weights_only=True)["state_dict"]
    probe.load_state_dict(probe_state)
    probe.to(device).eval()
    data_config = config["data"]
    root = Path(str(data_config["root"]))
    train_root = root / "Data" / "CLS-LOC" / "train"
    if not train_root.is_dir():
        train_root = root / "train"
    wnids = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if data_config.get("class_wnids_file"):
        selected = {
            line.strip() for line in Path(str(data_config["class_wnids_file"])).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        wnids = [wnid for wnid in wnids if wnid in selected]
    elif data_config.get("class_wnids"):
        selected = {str(item) for item in data_config["class_wnids"]}
        wnids = [wnid for wnid in wnids if wnid in selected]
    transform = ImageNetProbeTransform(False, data_config.get("augmentation", {}))
    robustness_root = Path(args.root)

    def evaluate_dataset(dataset: object) -> Dict[str, float]:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=False,
            pin_memory=True,
        )
        accumulator = ClassificationMetricAccumulator()
        for images, labels in loader:
            logits = probe(images.to(device))
            accumulator.update(logits, labels.to(device))
        metrics = accumulator.metrics()
        del loader, dataset
        return metrics

    def evaluate_folder(path: Path) -> Dict[str, float]:
        return evaluate_dataset(WNIDFolderDataset(str(path), wnids, transform))

    result: Dict[str, Any] = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "probe_checkpoint": str(Path(args.probe_checkpoint).resolve()),
        "dataset": dataset_name,
    }
    suites = {args.suite} if args.suite != "all" else {"imagenet-c", "imagenet-r", "imagenet-a"}
    if "imagenet-c" in suites:
        clean_base = ImageNetFiles(
            str(data_config["root"]), "val", str(data_config["val_labels"]), class_wnids=wnids,
            manifest=str(data_config.get("val_manifest", "")),
        )
        clean_metrics = evaluate_dataset(LabeledImageDataset(clean_base, transform))
        clean_error = 1.0 - clean_metrics["top1_accuracy"]
        corruption_root = robustness_root / "imagenet-c"
        details: Dict[str, Dict[str, Dict[str, float]]] = {}
        accuracies = []
        normalized_errors = []
        relative_normalized_errors = []
        extra_accuracies = []
        for corruption in sorted(path for path in corruption_root.iterdir() if path.is_dir()):
            severities: Dict[str, Dict[str, float]] = {}
            corruption_errors = []
            for severity in range(1, 6):
                metrics = evaluate_folder(corruption / str(severity))
                severities[str(severity)] = metrics
                accuracies.append(metrics["top1_accuracy"])
                corruption_errors.append(1.0 - metrics["top1_accuracy"])
            details[corruption.name] = severities
            mean_error = sum(corruption_errors) / len(corruption_errors)
            if corruption.name in IMAGENET_C_ALEXNET_ERRORS:
                reference_error = IMAGENET_C_ALEXNET_ERRORS[corruption.name]
                normalized_errors.append(mean_error / reference_error)
                denominator = reference_error - IMAGENET_ALEXNET_CLEAN_ERROR
                relative_normalized_errors.append((mean_error - clean_error) / denominator)
            else:
                extra_accuracies.extend(1.0 - value for value in corruption_errors)
        if not accuracies:
            raise FileNotFoundError(f"no ImageNet-C corruption folders under {corruption_root}")
        result["imagenet_c"] = {
            "mean_corruption_accuracy": sum(accuracies) / len(accuracies),
            "mce": 100.0 * sum(normalized_errors) / len(normalized_errors),
            "relative_mce": 100.0 * sum(relative_normalized_errors) / len(relative_normalized_errors),
            "clean": clean_metrics,
            "canonical_corruptions": sorted(IMAGENET_C_ALEXNET_ERRORS),
            "extra_mean_accuracy": sum(extra_accuracies) / len(extra_accuracies) if extra_accuracies else None,
            "corruptions": details,
            "note": "mCE/relative mCE use the canonical 15 ImageNet-C AlexNet reference errors; gaussian_blur, saturate, spatter, and speckle_noise are extra and excluded from mCE",
        }
    if "imagenet-r" in suites:
        result["imagenet_r"] = evaluate_folder(robustness_root / "imagenet-r")
    if "imagenet-a" in suites:
        result["imagenet_a"] = evaluate_folder(robustness_root / "imagenet-a")
    output = (Path(args.output).resolve() if args.output else
              Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "imagenet_robustness.json")
    _write_json(output, result)
    _append_harness_metric({"stage": "imagenet_robustness", **result})
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FMCA-AV Lightning experiment runner")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output", default="")
    train_parser.add_argument("--resume", default="")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--overrides-json")
    train_parser.set_defaults(func=train)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--config", required=True)
    calibrate_parser.add_argument("--checkpoint", required=True)
    calibrate_parser.add_argument("--output", required=True)
    calibrate_parser.add_argument("--device", default="cpu")
    calibrate_parser.set_defaults(func=calibrate)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--calibration", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.set_defaults(func=evaluate)

    analytic_parser = subparsers.add_parser("analytic")
    analytic_parser.add_argument("--config", required=True)
    analytic_parser.add_argument("--output", default="")
    analytic_parser.set_defaults(func=analytic)

    probe_parser = subparsers.add_parser("linear-probe")
    probe_parser.add_argument("--config", required=True)
    probe_parser.add_argument("--checkpoint", required=True)
    probe_parser.add_argument("--output", default="")
    probe_parser.add_argument("--seed", type=int)
    probe_parser.add_argument("--overrides-json")
    probe_parser.set_defaults(func=linear_probe)

    finetune_parser = subparsers.add_parser("fine-tune")
    finetune_parser.add_argument("--config", required=True)
    finetune_parser.add_argument("--checkpoint", required=True)
    finetune_parser.add_argument("--output", default="")
    finetune_parser.add_argument("--seed", type=int)
    finetune_parser.add_argument("--overrides-json")
    finetune_parser.set_defaults(func=fine_tune)

    knn_parser = subparsers.add_parser("knn")
    knn_parser.add_argument("--config", required=True)
    knn_parser.add_argument("--checkpoint", required=True)
    knn_parser.add_argument("--output", default="")
    knn_parser.add_argument("--device", default="cuda")
    knn_parser.add_argument("--neighbors", type=int, default=20)
    knn_parser.add_argument("--temperature", type=float, default=0.07)
    knn_parser.add_argument("--batch-size", type=int, default=256)
    knn_parser.add_argument("--workers", type=int, default=8)
    knn_parser.add_argument("--bank-chunk-size", type=int, default=8192)
    knn_parser.add_argument("--bank-limit", type=int, default=0)
    knn_parser.add_argument("--seed", type=int)
    knn_parser.add_argument("--overrides-json")
    knn_parser.set_defaults(func=knn_evaluate)

    corruption_parser = subparsers.add_parser("corruption-eval")
    corruption_parser.add_argument("--config", required=True)
    corruption_parser.add_argument("--checkpoint", required=True)
    corruption_parser.add_argument("--probe-checkpoint", required=True)
    corruption_parser.add_argument("--root", required=True)
    corruption_parser.add_argument("--output", default="")
    corruption_parser.add_argument("--device", default="cuda")
    corruption_parser.add_argument("--batch-size", type=int, default=256)
    corruption_parser.add_argument("--workers", type=int, default=8)
    corruption_parser.add_argument("--seed", type=int)
    corruption_parser.add_argument("--overrides-json")
    corruption_parser.set_defaults(func=corruption_evaluate)
    imagenet_robustness_parser = subparsers.add_parser("imagenet-robustness")
    imagenet_robustness_parser.add_argument("--config", required=True)
    imagenet_robustness_parser.add_argument("--checkpoint", required=True)
    imagenet_robustness_parser.add_argument("--probe-checkpoint", required=True)
    imagenet_robustness_parser.add_argument("--root", required=True)
    imagenet_robustness_parser.add_argument("--suite", choices=("imagenet-c", "imagenet-r", "imagenet-a", "all"), default="all")
    imagenet_robustness_parser.add_argument("--output", default="")
    imagenet_robustness_parser.add_argument("--device", default="cuda")
    imagenet_robustness_parser.add_argument("--batch-size", type=int, default=256)
    imagenet_robustness_parser.add_argument("--workers", type=int, default=8)
    imagenet_robustness_parser.add_argument("--seed", type=int)
    imagenet_robustness_parser.add_argument("--overrides-json")
    imagenet_robustness_parser.set_defaults(func=imagenet_robustness_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "seed", None) is not None:
        os.environ["FMCA_SEED_OVERRIDE"] = str(args.seed)
    if getattr(args, "overrides_json", None) is not None:
        overrides = json.loads(args.overrides_json)
        if not isinstance(overrides, dict):
            raise ValueError("--overrides-json must decode to a JSON object")
        os.environ["FMCA_CONFIG_OVERRIDES"] = json.dumps(overrides, separators=(",", ":"))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
