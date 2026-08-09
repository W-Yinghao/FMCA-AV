#!/usr/bin/env python3
"""Confirmatory E3 estimator controls on analytic Gaussian and finite channels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.operators import (
    FMCAMoments,
    SCIENTIFIC_CORRECTNESS_VERSION,
    estimate_moments,
    inverse_sqrt_covariance,
)


def spectral_values(moments: FMCAMoments, ridge: float, whitening: str) -> torch.Tensor:
    identity = torch.eye(moments.r_f.shape[0], dtype=moments.r_f.dtype)
    left = inverse_sqrt_covariance(moments.r_f, ridge) if whitening in {"left", "dual", "dual_posthoc"} else identity
    right = inverse_sqrt_covariance(moments.r_g, ridge) if whitening in {"right", "dual", "dual_posthoc"} else identity
    singular = torch.linalg.svdvals(left @ moments.p_fg @ right)
    if whitening == "dual_posthoc":
        # Make the post-hoc spectral projection explicit and discard sample-null modes.
        singular = singular[: min(len(singular), moments.count_x - int(moments.centered))]
    return singular.square()


def hermite(values: torch.Tensor, dimension: int) -> torch.Tensor:
    columns = []
    previous = torch.ones_like(values)
    current = values
    for degree in range(1, dimension + 1):
        columns.append(current)
        following = (values * current - math.sqrt(degree) * previous) / math.sqrt(degree + 1)
        previous, current = current, following
    return torch.cat(columns, 1)


def gaussian_moments(samples: int, dimension: int, centered: bool, dtype: torch.dtype, seed: int) -> tuple[FMCAMoments, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, 1, generator=generator, dtype=dtype)
    y = x + torch.randn(samples, 1, generator=generator, dtype=dtype)
    moments = estimate_moments(
        hermite(x, dimension), hermite(y / math.sqrt(2.0), dimension)[:, None, :], centered=centered
    )
    truth = torch.tensor([0.5 ** degree for degree in range(1, dimension + 1)], dtype=dtype)
    return moments, truth


def finite_channel(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    states = 12
    joint = torch.zeros(states, states, dtype=dtype)
    for left in range(states):
        joint[left, left] += 0.55 / states
        joint[left, (left + 1) % states] += 0.25 / states
        joint[left, (left - 1) % states] += 0.10 / states
        joint[left, :] += 0.10 / (states * states)
    marginal_x = joint.sum(1)
    marginal_y = joint.sum(0)
    standardized = joint / torch.sqrt(marginal_x[:, None] * marginal_y[None, :])
    truth = torch.linalg.svdvals(standardized)[1:].square()
    return joint, truth


def finite_moments(samples: int, centered: bool, dtype: torch.dtype, seed: int) -> tuple[FMCAMoments, torch.Tensor]:
    joint, truth = finite_channel(dtype)
    generator = torch.Generator().manual_seed(seed)
    pair = torch.multinomial(joint.flatten(), samples, replacement=True, generator=generator)
    states = joint.shape[0]
    x = torch.nn.functional.one_hot(pair // states, states).to(dtype)
    y = torch.nn.functional.one_hot(pair % states, states).to(dtype)
    return estimate_moments(x, y[:, None, :], centered=centered), truth


def condition_number(matrix: torch.Tensor) -> float:
    values = torch.linalg.eigvalsh(matrix).abs()
    positive = values[values > torch.finfo(values.dtype).eps * max(1, matrix.shape[0])]
    return float(positive.max() / positive.min()) if len(positive) else float("inf")


def evaluate(dataset: str, samples: int, dimension: int, centered: bool, precision: str,
             whitening: str, ridge_rule: str, objective: str, seed: int) -> dict[str, object]:
    dtype = torch.float32 if precision == "fp32" else torch.float64
    moments, truth = (gaussian_moments(samples, dimension, centered, dtype, seed)
                      if dataset == "gaussian" else finite_moments(samples, centered, dtype, seed))
    ridge = 1e-3
    if ridge_rule == "adaptive":
        scale = float((moments.r_f.diagonal().mean() + moments.r_g.diagonal().mean()) / 2)
        ridge = max(1e-5, min(1e-2, scale / math.sqrt(samples)))
    try:
        values = spectral_values(moments, ridge, whitening)
        count = min(len(values), len(truth))
        clipped = values.clamp(0.0, 1.0 - 1e-7)
        score = float(clipped.sum()) if objective == "trace" else float(-torch.log1p(-clipped).sum())
        absolute = (values[:count] - truth[:count]).abs()
        result = {
            "state": "SUCCEEDED", "score": score, "spectrum_mae": float(absolute.mean()),
            "spectrum_relative_error": float((absolute / truth[:count].clamp_min(1e-8)).mean()),
            "effective_rank": int((values > 1e-6).sum()), "largest_eigenvalue": float(values.max()),
            "rf_condition": condition_number(moments.r_f), "rg_condition": condition_number(moments.r_g),
            "minimum_variance": float(min(moments.r_f.diagonal().min(), moments.r_g.diagonal().min())),
            "nonfinite": int(not torch.isfinite(values).all()),
        }
    except (RuntimeError, ValueError) as error:
        result = {"state": "FAILED", "failure_reason": str(error), "nonfinite": 1}
    return {
        "dataset": dataset, "samples": samples, "dimension": dimension, "centered": centered,
        "precision": precision, "whitening": whitening, "ridge_rule": ridge_rule, "ridge": ridge,
        "objective": objective, "seed": seed, **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20308000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    # One-factor coverage plus a deterministic 32-cell interaction matrix.
    reference = ("gaussian", 512, 16, True, "fp64", "dual", "fixed", "trace")
    designs = {reference}
    designs.update((dataset, 512, 11 if dataset == "finite" else 16, True, "fp64", "dual", "fixed", "trace") for dataset in ("gaussian", "finite"))
    designs.update(("gaussian", samples, 16, True, "fp64", "dual", "fixed", "trace") for samples in (64, 128, 256, 512))
    designs.update(("gaussian", 512, dimension, True, "fp64", "dual", "fixed", "trace") for dimension in (4, 8, 16, 32))
    designs.update(("gaussian", 512, 16, centered, "fp64", "dual", "fixed", "trace") for centered in (False, True))
    designs.update(("gaussian", 512, 16, True, precision, "dual", "fixed", "trace") for precision in ("fp32", "fp64"))
    designs.update(("gaussian", 512, 16, True, "fp64", whitening, "fixed", "trace") for whitening in ("none", "left", "right", "dual", "dual_posthoc"))
    designs.update(("gaussian", 512, 16, True, "fp64", "dual", ridge, "trace") for ridge in ("fixed", "adaptive"))
    designs.update(("gaussian", 512, 16, True, "fp64", "dual", "fixed", objective) for objective in ("trace", "logdet"))
    datasets = ("gaussian", "finite"); samples_axis = (128, 512); precision_axis = ("fp32", "fp64")
    whitening_axis = ("left", "right", "dual", "dual_posthoc"); ridge_axis = ("fixed", "adaptive"); objective_axis = ("trace", "logdet")
    for cell in range(32):
        dataset = datasets[cell % 2]
        designs.add((dataset, samples_axis[(cell // 2) % 2], 11 if dataset == "finite" else 16,
                     bool((cell // 4) % 2), precision_axis[(cell // 8) % 2], whitening_axis[cell % 4],
                     ridge_axis[(cell // 16) % 2], objective_axis[(cell // 3) % 2]))
    records = []
    for design_index, design in enumerate(sorted(designs, key=str)):
        for replicate in range(args.replicates):
            records.append(evaluate(*design, args.seed + design_index * 1000 + replicate))
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "parameters": vars(args),
        "design_count": len(designs),
        "records": records,
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "e3_estimator_controls.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "e3_estimator_controls", "records": len(records)}) + "\n")
    print(json.dumps({"designs": len(designs), "records": len(records), "failures": sum(row["state"] != "SUCCEEDED" for row in records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
