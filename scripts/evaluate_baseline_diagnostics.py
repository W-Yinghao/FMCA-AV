#!/usr/bin/env python3
"""Collapse and covariance diagnostics for a frozen external SSL baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fmca_av.baselines import BaselineSSL
from fmca_av.config import load_config
from fmca_av.data.cifar import CIFARFiles, CIFARProbeTransform, LabeledCIFARDataset
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def spectral_diagnostics(values: torch.Tensor) -> dict[str, object]:
    values = values.double()
    centered = values - values.mean(0)
    covariance = centered.T @ centered / max(1, len(values) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    trace = eigenvalues.sum()
    probabilities = eigenvalues / trace if trace > 0 else torch.zeros_like(eigenvalues)
    positive_probabilities = probabilities[probabilities > 0]
    effective_rank = (
        torch.exp(-(positive_probabilities * positive_probabilities.log()).sum())
        if len(positive_probabilities) else trace.new_zeros(())
    )
    maximum = eigenvalues.max() if len(eigenvalues) else trace.new_zeros(())
    numerical = eigenvalues[eigenvalues > maximum * 1e-6]
    standard_deviation = centered.std(0, unbiased=True)
    scale = standard_deviation[:, None] * standard_deviation[None, :]
    correlation = covariance / scale.clamp_min(torch.finfo(covariance.dtype).tiny)
    off_diagonal = correlation.flatten()[:-1].view(len(correlation) - 1, len(correlation) + 1)[:, 1:]
    return {
        "samples": len(values),
        "dimensions": values.shape[1],
        "covariance_trace": float(trace),
        "covariance_max_eigenvalue": float(maximum),
        "covariance_min_positive_eigenvalue": float(numerical.min()) if len(numerical) else 0.0,
        "covariance_condition_number": float(maximum / numerical.min()) if len(numerical) else None,
        "numerical_rank_relative_1e-6": int(len(numerical)),
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(effective_rank / values.shape[1]),
        "feature_std_mean": float(standard_deviation.mean()),
        "feature_std_min": float(standard_deviation.min()),
        "collapsed_dimension_fraction_std_lt_1e-2": float((standard_deviation < 1e-2).double().mean()),
        "mean_absolute_off_diagonal_correlation": float(off_diagonal.abs().mean()),
    }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["data"]["dataset"] != "cifar10":
        raise ValueError("this scoped diagnostic accepts CIFAR-10 only")
    device = torch.device("cuda")
    model = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    augmentation = config["data"].get("augmentation", {})
    transform = CIFARProbeTransform(
        False,
        augmentation.get("mean", [0.4914, 0.4822, 0.4465]),
        augmentation.get("std", [0.2470, 0.2435, 0.2616]),
    )
    dataset = LabeledCIFARDataset(CIFARFiles(config["data"]["root"], "cifar10", train=False), transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )
    backbone_values = []
    projection_values = []
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        backbone = model.backbone(images)
        projection = model.diagnostic_projection(backbone)
        take = min(len(images), args.max_samples - seen)
        backbone_values.append(backbone[:take].cpu())
        projection_values.append(projection[:take].cpu())
        seen += take
        if seen >= args.max_samples:
            break
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "method": config["experiment"]["method"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": "cifar10-test-clean",
        "backbone": spectral_diagnostics(torch.cat(backbone_values)),
        "projector": spectral_diagnostics(torch.cat(projection_values)),
        "projector_input_semantics": (
            "HAI final-stage projector with neutral [brightness=1, contrast=1, saturation=1, hue=0] embedding"
            if model.method == "hai_simsiam" else "baseline projector applied to frozen backbone representation"
        ),
        "threshold_note": "numerical rank uses lambda > 1e-6 * lambda_max; collapsed dimensions use std < 1e-2",
    }
    output = Path(args.output).resolve() if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "diagnostics.json"
    atomic_json(output, payload)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "baseline_diagnostics", **payload}, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
