#!/usr/bin/env python3
"""Evaluate factor predictability across preregistered spectral-coordinate selections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from fmca_av.config import load_config
from fmca_av.data.factors import FACTOR_CARDINALITIES, FACTOR_NAMES, factor_dataset
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from fmca_av.data.imagenet import ImageNetProbeTransform
from fmca_av.vision_module import VisionFMCAAV


class TransformedFactors(Dataset):
    def __init__(self, base: Dataset, size: int) -> None:
        self.base = base
        self.transform = ImageNetProbeTransform(
            False,
            {
                "size": size,
                "eval_resize": size,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, factors = self.base[index]
        return self.transform(image), factors


def collect(
    model: VisionFMCAAV,
    loader: Iterable,
    calibration: dict[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    raw_values = []
    canonical_values = []
    labels = []
    mean = calibration["mean_g"].to(device)
    transform = calibration["transform_g"].to(device)
    with torch.inference_mode():
        for images, batch_labels in loader:
            representation = model.backbone(images.to(device))
            raw = model.g_head(representation)
            canonical = (raw - mean) @ transform
            raw_values.append(raw.cpu())
            canonical_values.append(canonical.cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(raw_values), torch.cat(canonical_values), torch.cat(labels)


def ridge_accuracy(train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor, classes: int, ridge: float) -> float:
    if classes == 1:
        return 1.0
    x = torch.cat((train_x.double(), torch.ones(len(train_x), 1, dtype=torch.float64)), dim=1)
    z = torch.cat((test_x.double(), torch.ones(len(test_x), 1, dtype=torch.float64)), dim=1)
    target = torch.nn.functional.one_hot(train_y.long(), classes).double()
    gram = x.transpose(0, 1) @ x
    regularizer = ridge * torch.trace(gram) / max(1, gram.shape[0])
    gram.diagonal().add_(regularizer)
    weights = torch.linalg.solve(gram, x.transpose(0, 1) @ target)
    prediction = torch.argmax(z @ weights, dim=1)
    return float((prediction == test_y).double().mean())


def evaluate_selection(
    name: str,
    train: Tensor,
    test: Tensor,
    train_labels: Tensor,
    test_labels: Tensor,
    cardinalities: tuple[int, ...],
    k: int,
    ridge: float,
    repeat: int,
) -> list[dict[str, object]]:
    values = []
    for factor, classes in enumerate(cardinalities):
        accuracy = ridge_accuracy(train, train_labels[:, factor], test, test_labels[:, factor], classes, ridge)
        values.append({"selection": name, "k": k, "repeat": repeat, "factor_index": factor, "accuracy": accuracy})
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--rotation-repeats", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_name = str(config["data"]["dataset"])
    if dataset_name not in FACTOR_CARDINALITIES:
        raise ValueError("factor spectral probe requires a factor dataset configuration")
    root = str(config["data"]["root"])
    generator = torch.Generator().manual_seed(int(config["seed"]) + 7000)
    if dataset_name == "smallnorb":
        train_base = factor_dataset(root, dataset_name, "train")
        test_base = factor_dataset(root, dataset_name, "test")
        train_indices = torch.randperm(len(train_base), generator=generator)[:args.train_samples].tolist()
        test_indices = torch.randperm(len(test_base), generator=generator)[:args.test_samples].tolist()
    else:
        train_base = factor_dataset(root, dataset_name, "train")
        test_base = train_base
        indices = torch.randperm(len(train_base), generator=generator)[:args.train_samples + args.test_samples]
        train_indices = indices[:args.train_samples].tolist()
        test_indices = indices[args.train_samples:].tolist()
    size = int(config["data"].get("augmentation", {}).get("size", 64))
    train_loader = DataLoader(
        TransformedFactors(Subset(train_base, train_indices), size),
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TransformedFactors(Subset(test_base, test_indices), size),
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )
    device = torch.device(args.device)
    model = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
    train_raw, train_canonical, train_labels = collect(model, train_loader, calibration, device)
    test_raw, test_canonical, test_labels = collect(model, test_loader, calibration, device)
    raw_mean = train_raw.mean(0, keepdim=True)
    covariance = (train_raw - raw_mean).transpose(0, 1) @ (train_raw - raw_mean) / max(1, len(train_raw) - 1)
    _, basis = torch.linalg.eigh(covariance.double())
    train_pca = (train_raw - raw_mean).double() @ basis.flip(1)
    test_pca = (test_raw - raw_mean).double() @ basis.flip(1)
    feature_dim = train_canonical.shape[1]
    ks = [value for value in (1, 2, 4, 8, 16, 32, 64, 128) if value <= feature_dim]
    cardinalities = FACTOR_CARDINALITIES[dataset_name]
    records: list[dict[str, object]] = []
    for k in ks:
        records.extend(evaluate_selection("eigen_top", train_canonical[:, :k], test_canonical[:, :k], train_labels, test_labels, cardinalities, k, args.ridge, 0))
        records.extend(evaluate_selection("eigen_bottom", train_canonical[:, -k:], test_canonical[:, -k:], train_labels, test_labels, cardinalities, k, args.ridge, 0))
        records.extend(evaluate_selection("pca_top", train_pca[:, :k], test_pca[:, :k], train_labels, test_labels, cardinalities, k, args.ridge, 0))
        records.extend(evaluate_selection("unranked_first", train_raw[:, :k], test_raw[:, :k], train_labels, test_labels, cardinalities, k, args.ridge, 0))
        for repeat in range(args.random_repeats):
            permutation = torch.randperm(feature_dim, generator=generator)[:k]
            records.extend(evaluate_selection("random", train_canonical[:, permutation], test_canonical[:, permutation], train_labels, test_labels, cardinalities, k, args.ridge, repeat))
        for repeat in range(args.rotation_repeats):
            matrix = torch.randn(feature_dim, feature_dim, generator=generator, dtype=torch.float64)
            rotation = torch.linalg.qr(matrix).Q[:, :k]
            records.extend(evaluate_selection("random_rotation_first", train_canonical.double() @ rotation, test_canonical.double() @ rotation, train_labels, test_labels, cardinalities, k, args.ridge, repeat))
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "dataset": dataset_name,
        "factor_names": FACTOR_NAMES[dataset_name],
        "factor_cardinalities": cardinalities,
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "calibration": str(Path(args.calibration).resolve()),
        "records": records,
    }
    if args.output:
        output = Path(args.output)
    elif os.environ.get("FMCA_HARNESS_RUN_DIR"):
        output = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "factor_probe.json"
    else:
        raise ValueError("--output is required outside the harness")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir:
        metric = {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": "factor_spectral_probe",
            "dataset": dataset_name,
            "train_samples": len(train_labels),
            "test_samples": len(test_labels),
        }
        with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
