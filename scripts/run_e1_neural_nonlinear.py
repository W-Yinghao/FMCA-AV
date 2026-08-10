#!/usr/bin/env python3
"""Train Lightning FMCA-AV MLPs on continuous nonlinear toy channels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from fmca_av.analytic import finite_channel_spectrum
from fmca_av.lightning_module import GaussianFMCAAV
from fmca_av.operators import (
    SCIENTIFIC_CORRECTNESS_VERSION,
    evaluate_heldout_spectrum,
    fit_spectral_calibration,
)


FAMILIES = ("two_moons", "gmm", "spiral")


def clean_samples(family: str, count: int, generator: torch.Generator) -> Tensor:
    """Sample the clean continuous parent coordinate for a nonlinear family."""

    if family == "two_moons":
        component = torch.randint(2, (count,), generator=generator)
        phase = torch.rand(count, generator=generator) * math.pi
        values = torch.stack((torch.cos(phase), torch.sin(phase)), dim=1)
        mask = component == 1
        values[mask, 0] = 1.0 - values[mask, 0]
        values[mask, 1] = 0.5 - values[mask, 1]
        return values
    if family == "gmm":
        component = torch.randint(8, (count,), generator=generator)
        phase = component.double() * (2.0 * math.pi / 8.0)
        centers = torch.stack((2.0 * torch.cos(phase), 2.0 * torch.sin(phase)), dim=1).float()
        return centers + 0.15 * torch.randn(count, 2, generator=generator)
    if family == "spiral":
        arm = torch.randint(3, (count,), generator=generator)
        phase_unit = torch.rand(count, generator=generator)
        phase = phase_unit * (3.0 * math.pi) + arm * (2.0 * math.pi / 3.0)
        radius = 0.15 + phase_unit
        return torch.stack((radius * torch.cos(phase), radius * torch.sin(phase)), dim=1)
    raise ValueError(f"unknown nonlinear family: {family}")


def parse_conditions(value: str) -> list[tuple[str, int]]:
    result = []
    for item in value.split(","):
        family, raw_seed = item.split(":", 1)
        if family not in FAMILIES:
            raise ValueError(f"unsupported family {family!r}")
        seed_index = int(raw_seed)
        if seed_index not in (1, 2, 3):
            raise ValueError("seed indices must be 1, 2, or 3")
        pair = (family, seed_index)
        if pair in result:
            raise ValueError(f"duplicate condition {item}")
        result.append(pair)
    if not result:
        raise ValueError("at least one condition is required")
    return result


def configuration(args: argparse.Namespace, family: str, seed: int) -> dict[str, object]:
    return {
        "experiment": {"name": f"e1-neural-{family}", "family": "gaussian_1d", "claim_id": "E1/C1"},
        "seed": seed,
        "data": {
            "dimension": 2, "noise_variance": args.noise_std ** 2,
            "num_views": args.views, "batch_size": args.batch_size,
        },
        "model": {
            "feature_dim": args.feature_dim, "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "activation": "gelu", "centered": True,
        },
        "objective": {"name": "trace", "ridge": 1e-3, "logdet_margin": 1e-6},
        "optimizer": {"learning_rate": args.learning_rate, "weight_decay": 1e-6},
    }


def feature_maps(model: GaussianFMCAAV, parents: Tensor, views: int, noise_std: float,
                 batch_size: int, device: torch.device, seed: int) -> tuple[Tensor, Tensor]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    f_values, g_values = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(parents), batch_size):
            x = parents[start:start + batch_size].to(device)
            y = x[:, None, :] + noise_std * torch.randn(
                len(x), views, x.shape[1], device=device, dtype=x.dtype,
            )
            f, g = model.feature_maps(x, y)
            f_values.append(f.cpu())
            g_values.append(g.cpu())
    return torch.cat(f_values), torch.cat(g_values)


def bin_indices(values: Tensor, bins: int) -> Tensor:
    indices = (((values.clamp(-4.0, 4.0) + 4.0) / 8.0) * bins).long().clamp(0, bins - 1)
    return indices[:, 0] * bins + indices[:, 1]


def numerical_oracle(parents: Tensor, views: Tensor, bins: int) -> Tensor:
    parent_index = bin_indices(parents, bins)
    child_index = bin_indices(views.flatten(0, 1), bins)
    repeated_parent = parent_index[:, None].expand(-1, views.shape[1]).reshape(-1)
    states = bins * bins
    counts = torch.bincount(
        repeated_parent * states + child_index, minlength=states * states,
    ).reshape(states, states).double()
    # A negligible positive pseudocount retains the fixed alphabet when a tail
    # bin is empty, without fitting or selecting bins from evaluation results.
    joint = counts + 1e-8
    return finite_channel_spectrum(joint / joint.sum()).eigenvalues


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_condition(args: argparse.Namespace, family: str, seed_index: int, run_dir: Path) -> dict[str, object]:
    seed = args.seed_base + seed_index
    L.seed_everything(seed, workers=True)
    train = clean_samples(family, args.train_samples, torch.Generator().manual_seed(seed + 10))
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True).clamp_min(1e-6)
    train = (train - mean) / std
    validation = (
        clean_samples(family, args.val_samples, torch.Generator().manual_seed(seed + 20)) - mean
    ) / std
    train_loader = DataLoader(
        TensorDataset(train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(seed + 30),
    )
    val_loader = DataLoader(
        TensorDataset(validation), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    config = configuration(args, family, seed)
    model = GaussianFMCAAV(config)
    trainer = L.Trainer(
        accelerator="gpu", devices=1, max_epochs=args.max_epochs, precision="32-true",
        deterministic=True, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, log_every_n_steps=10,
    )
    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    duration = time.perf_counter() - started
    checkpoint = run_dir / f"{family}_seed{seed_index}.ckpt"
    trainer.save_checkpoint(checkpoint)
    device = model.device
    calibration_parent = (
        clean_samples(family, args.calibration_samples, torch.Generator().manual_seed(seed + 40)) - mean
    ) / std
    test_parent = (
        clean_samples(family, args.test_samples, torch.Generator().manual_seed(seed + 50)) - mean
    ) / std
    oracle_parent = (
        clean_samples(family, args.oracle_samples, torch.Generator().manual_seed(seed + 60)) - mean
    ) / std
    calibration_f, calibration_g = feature_maps(
        model, calibration_parent, args.views, args.noise_std, args.eval_batch_size, device, seed + 70,
    )
    test_f, test_g = feature_maps(
        model, test_parent, args.views, args.noise_std, args.eval_batch_size, device, seed + 80,
    )
    torch.manual_seed(seed + 90)
    oracle_views = oracle_parent[:, None, :] + args.noise_std * torch.randn(
        len(oracle_parent), args.views, 2,
    )
    oracle = numerical_oracle(oracle_parent, oracle_views, args.oracle_bins)
    calibration = fit_spectral_calibration(calibration_f, calibration_g, ridge=1e-3, centered=True)
    heldout = evaluate_heldout_spectrum(test_f, test_g, calibration)
    count = min(args.feature_dim, len(oracle), len(heldout.eigenvalues))
    neural = heldout.eigenvalues[:count].double()
    target = oracle[:count].double()
    metrics = trainer.callback_metrics
    return {
        "family": family, "seed_index": seed_index, "seed": seed,
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "parent_definition": "continuous clean coordinate",
        "conditional_definition": "independent additive-Gaussian subviews of the same parent",
        "views": args.views, "noise_std": args.noise_std,
        "train_samples": args.train_samples, "calibration_samples": args.calibration_samples,
        "test_samples": args.test_samples, "oracle_samples": args.oracle_samples,
        "max_epochs": args.max_epochs, "global_optimizer_step": int(trainer.global_step),
        "training_duration_seconds": duration,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_validation_score": float(metrics["val/score"].detach().cpu()),
        "checkpoint": str(checkpoint.resolve()),
        "heldout_singular_values": heldout.singular_values.detach().cpu().tolist(),
        "heldout_eigenvalues": heldout.eigenvalues.detach().cpu().tolist(),
        "oracle_eigenvalues": oracle[:args.feature_dim].tolist(),
        "topk_spectrum_mae": float((neural - target).abs().mean()),
        "topk_spectrum_relative_l1": float((neural - target).abs().sum() / target.abs().sum().clamp_min(1e-12)),
        "heldout_trace": float(heldout.eigenvalues.sum()),
        "oracle_topk_trace": float(target.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True, help="comma-separated family:seed-index pairs")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--calibration-samples", type=int, default=20000)
    parser.add_argument("--test-samples", type=int, default=20000)
    parser.add_argument("--oracle-samples", type=int, default=100000)
    parser.add_argument("--oracle-bins", type=int, default=10)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=20310000)
    args = parser.parse_args()
    conditions = parse_conditions(args.conditions)
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "e1_neural_nonlinear.json"
    payload: dict[str, object] = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "method": "fmca_av_lightning_mlp", "parameters": vars(args), "records": [],
    }
    for family, seed_index in conditions:
        record = run_condition(args, family, seed_index, run_dir)
        records = list(payload["records"]); records.append(record); payload["records"] = records
        atomic_json(output, payload)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "stage": "e1_neural_nonlinear", "conditions": len(conditions),
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        }) + "\n")
    print(json.dumps({"conditions": len(conditions), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
