#!/usr/bin/env python3
"""Separate FroSSL encoder collapse from mean and BatchNorm-state effects.

This is an evaluation-only audit.  It never updates learned parameters.  BN
recalibration conditions operate on an in-memory model copy and update only BN
running buffers before frozen clean-CIFAR evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader

from fmca_av.baselines import BaselineSSL, frossl_loss_components
from fmca_av.config import load_config
from fmca_av.data.cifar import (
    CIFARFiles,
    CIFARProbeTransform,
    CIFARViewTransform,
    LabeledCIFARDataset,
    MultiViewDataset,
)
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


CONDITIONS = (
    "eval_saved",
    "batch_stats",
    "recal_flat_aug",
    "recal_sequential_aug",
    "recal_clean",
)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def reset_batch_norm(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.reset_running_stats()


@torch.inference_mode()
def recalibrate(
    model: BaselineSSL,
    loader: Iterable[object],
    device: torch.device,
    mode: str,
    maximum_samples: int,
) -> int:
    model.apply(reset_batch_norm)
    model.train()
    seen = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        if images.ndim == 5:
            if mode == "flattened":
                features = model.backbone(images.flatten(0, 1))
                model.diagnostic_projection(features)
            elif mode == "sequential":
                for index in range(images.shape[1]):
                    features = model.backbone(images[:, index])
                    model.diagnostic_projection(features)
            else:
                raise ValueError(f"unsupported augmented recalibration mode: {mode}")
        else:
            features = model.backbone(images)
            model.diagnostic_projection(features)
        seen += int(images.shape[0])
        if seen >= maximum_samples:
            break
    model.eval()
    return seen


def spectrum(values: Tensor) -> dict[str, object]:
    values = values.double()
    mean = values.mean(0)
    centered = values - mean
    denominator = max(1, len(values) - 1)
    raw = values.T @ values / max(1, len(values))
    covariance = centered.T @ centered / denominator

    def summarize(matrix: Tensor) -> dict[str, object]:
        eigenvalues = torch.linalg.eigvalsh(matrix).clamp_min(0)
        trace = eigenvalues.sum()
        probabilities = eigenvalues / trace if float(trace) > 0 else torch.zeros_like(eigenvalues)
        positive = probabilities[probabilities > 0]
        effective = (
            torch.exp(-(positive * positive.log()).sum())
            if len(positive) else trace.new_zeros(())
        )
        maximum = eigenvalues.max() if len(eigenvalues) else trace.new_zeros(())
        numerical = int((eigenvalues > maximum * 1e-6).sum()) if float(maximum) > 0 else 0
        descending = eigenvalues.flip(0)
        cumulative = descending.cumsum(0)
        rank99 = (
            int(torch.searchsorted(cumulative, 0.99 * trace).item()) + 1
            if float(trace) > 0 else 0
        )
        return {
            "trace": float(trace),
            "effective_rank": float(effective),
            "normalized_effective_rank": float(effective / values.shape[1]),
            "top_eigenvalue_share": float(maximum / trace) if float(trace) > 0 else 0.0,
            "rank99": rank99,
            "numerical_rank_relative_1e-6": numerical,
        }

    mean_energy = mean.square().sum()
    total_energy = values.square().sum(1).mean()
    return {
        "samples": int(len(values)),
        "dimensions": int(values.shape[1]),
        "mean_energy_ratio": float(mean_energy / total_energy) if float(total_energy) > 0 else 0.0,
        "raw_second_moment": summarize(raw),
        "centered_covariance": summarize(covariance),
    }


@torch.inference_mode()
def extract(
    model: BaselineSSL,
    loader: DataLoader,
    device: torch.device,
    use_batch_stats: bool,
    maximum: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    model.train(use_batch_stats)
    backbone_values: list[Tensor] = []
    projector_values: list[Tensor] = []
    labels: list[Tensor] = []
    seen = 0
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        features = model.backbone(images)
        projections = model.diagnostic_projection(features)
        take = len(images) if not maximum else min(len(images), maximum - seen)
        backbone_values.append(features[:take].float().cpu())
        projector_values.append(projections[:take].float().cpu())
        labels.append(batch_labels[:take].cpu())
        seen += take
        if maximum and seen >= maximum:
            break
    return torch.cat(backbone_values), torch.cat(projector_values), torch.cat(labels)


@torch.inference_mode()
def weighted_knn_from_features(
    train_values: Tensor,
    train_labels: Tensor,
    test_values: Tensor,
    test_labels: Tensor,
    device: torch.device,
    neighbors: int = 20,
    temperature: float = 0.07,
    query_batch: int = 256,
    bank_chunk: int = 8192,
) -> float:
    bank = functional.normalize(train_values, dim=1).half()
    correct = 0
    for query_start in range(0, len(test_values), query_batch):
        query_stop = min(query_start + query_batch, len(test_values))
        query = functional.normalize(test_values[query_start:query_stop].to(device), dim=1)
        best_scores = torch.empty(len(query), 0, device=device)
        best_labels = torch.empty(len(query), 0, dtype=torch.long, device=device)
        for start in range(0, len(bank), bank_chunk):
            stop = min(start + bank_chunk, len(bank))
            similarities = query @ bank[start:stop].to(device=device, dtype=query.dtype).T
            count = min(neighbors, similarities.shape[1])
            scores, indices = similarities.topk(count, dim=1)
            labels = train_labels[start:stop].to(device)[indices]
            candidate_scores = torch.cat((best_scores, scores), dim=1)
            candidate_labels = torch.cat((best_labels, labels), dim=1)
            keep = min(neighbors, candidate_scores.shape[1])
            best_scores, positions = candidate_scores.topk(keep, dim=1)
            best_labels = candidate_labels.gather(1, positions)
        votes = torch.zeros(len(query), 10, device=device)
        votes.scatter_add_(1, best_labels, torch.exp(best_scores / temperature))
        prediction = votes.argmax(1).cpu()
        correct += int((prediction == test_labels[query_start:query_stop]).sum())
    return correct / len(test_values)


@torch.inference_mode()
def loss_snapshot(model: BaselineSSL, views: Tensor, gamma: float, mode: str) -> dict[str, float]:
    model.train()
    if mode == "sequential":
        projections = torch.stack(
            [model.diagnostic_projection(model.backbone(views[:, index])) for index in range(views.shape[1])],
            dim=1,
        )
    else:
        projections = model.diagnostic_projection(model.backbone(views.flatten(0, 1))).reshape(
            views.shape[0], views.shape[1], -1
        )
    total, invariance, regularization = frossl_loss_components(projections, gamma)
    return {
        "total": float(total),
        "invariance": float(invariance),
        "regularization": float(regularization),
        "invariance_over_abs_regularization": float(
            invariance / regularization.abs().clamp_min(torch.finfo(invariance.dtype).tiny)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recalibration-samples", type=int, default=10000)
    parser.add_argument("--knn-bank-samples", type=int, default=20000)
    args = parser.parse_args()

    config = load_config(args.config)
    if config["experiment"].get("method") != "frossl" or config["data"].get("dataset") != "cifar10":
        raise ValueError("this audit is scoped to CIFAR-10 FroSSL")
    checkpoint_path = Path(args.checkpoint).resolve()
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configured_model = BaselineSSL(config)
    state_dict = raw_checkpoint.get("state_dict", {})
    incompatible = configured_model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint/config mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    device = torch.device("cuda")
    configured_model.to(device)
    data_config = config["data"]
    augmentation = data_config.get("augmentation", {})
    mean = augmentation.get("mean", [0.4914, 0.4822, 0.4465])
    std = augmentation.get("std", [0.2470, 0.2435, 0.2616])
    clean = CIFARProbeTransform(False, mean, std)
    train_base = CIFARFiles(str(data_config["root"]), "cifar10", train=True)
    test_base = CIFARFiles(str(data_config["root"]), "cifar10", train=False)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "pin_memory": True,
    }
    train_clean = DataLoader(LabeledCIFARDataset(train_base, clean), **loader_options)
    test_clean = DataLoader(LabeledCIFARDataset(test_base, clean), **loader_options)
    augmented = MultiViewDataset(
        train_base,
        CIFARViewTransform(augmentation),
        int(data_config["num_views"]),
        deterministic_seed=int(config["seed"]) + 700000,
    )
    augmented_loader = DataLoader(augmented, **loader_options)
    snapshot_views = next(iter(augmented_loader))[0][: min(32, args.batch_size)].to(device)

    results: dict[str, object] = {}
    for condition in args.conditions:
        model = copy.deepcopy(configured_model)
        recalibrated_samples = 0
        use_batch_stats = condition == "batch_stats"
        if condition == "recal_flat_aug":
            recalibrated_samples = recalibrate(
                model, augmented_loader, device, "flattened", args.recalibration_samples
            )
        elif condition == "recal_sequential_aug":
            recalibrated_samples = recalibrate(
                model, augmented_loader, device, "sequential", args.recalibration_samples
            )
        elif condition == "recal_clean":
            recalibrated_samples = recalibrate(
                model, train_clean, device, "clean", args.recalibration_samples
            )

        train_backbone, _, train_labels = extract(
            model, train_clean, device, use_batch_stats, args.knn_bank_samples
        )
        test_backbone, test_projector, test_labels = extract(
            model, test_clean, device, use_batch_stats
        )
        results[condition] = {
            "recalibrated_parent_samples": recalibrated_samples,
            "bn_inference": "current_batch_statistics" if use_batch_stats else "saved_or_recalibrated_running_statistics",
            "backbone": spectrum(test_backbone),
            "projector": spectrum(test_projector),
            "knn_accuracy": weighted_knn_from_features(
                train_backbone, train_labels, test_backbone, test_labels, device
            ),
            "knn_bank_samples": int(len(train_backbone)),
            "loss_snapshot": loss_snapshot(
                model,
                snapshot_views,
                float(config["objective"].get("invariance_weight", 1.0)),
                "sequential" if condition == "recal_sequential_aug" else "flattened",
            ),
        }
        del model
        torch.cuda.empty_cache()

    checkpoint_hparams = raw_checkpoint.get("hyper_parameters", {})
    embedded_config = checkpoint_hparams.get("config", {}) if isinstance(checkpoint_hparams, dict) else {}
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "audit": "FroSSL collapse: mean direction versus BN state versus encoder weights",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": int(raw_checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(raw_checkpoint.get("global_step", -1)),
        "requested_seed": int(config["seed"]),
        "checkpoint_embedded_seed": embedded_config.get("seed"),
        "load_missing_keys": list(incompatible.missing_keys),
        "load_unexpected_keys": list(incompatible.unexpected_keys),
        "views": int(data_config["num_views"]),
        "unique_parent_batch_size": int(data_config["batch_size"]),
        "encoded_views_per_batch": int(data_config["batch_size"]) * int(data_config["num_views"]),
        "projector_dimension": int(config["model"]["projection_dim"]),
        "configured_forward_mode": str(config["model"].get("view_forward_mode", "flattened")),
        "invariance_weight": float(config["objective"].get("invariance_weight", 1.0)),
        "conditions": results,
        "interpretation_note": (
            "batch_stats is diagnostic only; recalibration changes only in-memory BN running buffers. "
            "All reported spectra use both raw second moments and explicitly centered covariance."
        ),
    }
    output = (
        Path(args.output).resolve()
        if args.output
        else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "frossl_collapse_audit.json"
    )
    atomic_json(output, payload)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "frossl_collapse_audit", **payload}, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
