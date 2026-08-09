#!/usr/bin/env python3
"""E3 estimator/numerics ablations using exact Gaussian Hermite features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from fmca_av.objectives import logdet_score, trace_score
from fmca_av.operators import (
    FMCAMoments,
    SCIENTIFIC_CORRECTNESS_VERSION,
    estimate_moments,
    inverse_sqrt_covariance,
)


def hermite_features(values: torch.Tensor, dimension: int) -> torch.Tensor:
    features = []
    previous = torch.ones_like(values)
    current = values
    for degree in range(1, dimension + 1):
        features.append(current)
        following = (values * current - math.sqrt(degree) * previous) / math.sqrt(degree + 1)
        previous, current = current, following
    return torch.cat(features, dim=1)


def spectrum(moments: FMCAMoments, ridge: float, whitening: str) -> torch.Tensor:
    identity = torch.eye(moments.r_f.shape[0], dtype=moments.r_f.dtype)
    left = inverse_sqrt_covariance(moments.r_f, ridge) if whitening in {"left", "dual"} else identity
    right = inverse_sqrt_covariance(moments.r_g, ridge) if whitening in {"right", "dual"} else identity
    return torch.linalg.svdvals(left @ moments.p_fg @ right).square()


def condition_number(matrix: torch.Tensor) -> float:
    values = torch.linalg.eigvalsh(matrix).abs()
    positive = values[values > torch.finfo(values.dtype).eps]
    return float(positive.max() / positive.min()) if len(positive) else float("inf")


def make_moments(batch: int, views: int, features: int, centered: bool, seed: int) -> FMCAMoments:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 1, generator=generator, dtype=torch.float64)
    y = x[:, None, :] + torch.randn(batch, views, 1, generator=generator, dtype=torch.float64)
    f = hermite_features(x, features)
    g = hermite_features(y.reshape(-1, 1) / math.sqrt(2.0), features).reshape(batch, views, features)
    return estimate_moments(f, g, centered=centered)


def wrong_rg_moments(moments: FMCAMoments, batch: int, views: int, features: int, centered: bool, seed: int) -> FMCAMoments:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 1, generator=generator, dtype=torch.float64)
    y = x[:, None, :] + torch.randn(batch, views, 1, generator=generator, dtype=torch.float64)
    f = hermite_features(x, features)
    g = hermite_features(y.reshape(-1, 1) / math.sqrt(2.0), features).reshape(batch, views, features)
    if centered:
        f = f - f.mean(0, keepdim=True)
        g = g - g.mean((0, 1), keepdim=True)
    g_mean = g.mean(1)
    return FMCAMoments(
        r_f=f.T @ f / batch,
        r_g=g_mean.T @ g_mean / batch,
        p_fg=f.T @ g_mean / batch,
        count_x=batch,
        count_y=batch * views,
        centered=centered,
    )


def row(batch: int, views: int, features: int, ridge: float, centered: bool, whitening: str, seed: int, rg: str = "correct") -> dict[str, Any]:
    moments = (
        make_moments(batch, views, features, centered, seed)
        if rg == "correct"
        else wrong_rg_moments(make_moments(batch, views, features, centered, seed), batch, views, features, centered, seed)
    )
    eigenvalues = spectrum(moments, ridge, whitening)
    truth = torch.tensor([0.5 ** degree for degree in range(1, features + 1)], dtype=torch.float64)
    count = min(len(eigenvalues), len(truth))
    trace_value = float(trace_score(moments, ridge))
    try:
        logdet_value = float(logdet_score(moments, ridge, 1e-6))
    except (RuntimeError, ValueError):
        logdet_value = float("nan")
    return {
        "batch": batch,
        "views": views,
        "features": features,
        "ridge": ridge,
        "centered": centered,
        "whitening": whitening,
        "rg_estimator": rg,
        "trace_score": trace_value,
        "logdet_score": logdet_value,
        "spectrum_mae": float((eigenvalues[:count] - truth[:count]).abs().mean()),
        "largest_eigenvalue": float(eigenvalues.max()),
        "effective_rank": int((eigenvalues > 1e-6).sum()),
        "rf_condition": condition_number(moments.r_f),
        "rg_condition": condition_number(moments.r_g),
    }


def constant_mode_control(batch: int, views: int, seed: int) -> dict[str, Any]:
    moments = make_moments(batch, views, 8, False, seed)
    # Explicit constant coordinates reproduce the unremoved lambda=1 failure mode.
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(batch, 1, generator=generator, dtype=torch.float64)
    y = x[:, None, :] + torch.randn(batch, views, 1, generator=generator, dtype=torch.float64)
    f = torch.cat((torch.ones(batch, 1, dtype=torch.float64), hermite_features(x, 8)), 1)
    g = torch.cat((torch.ones(batch, views, 1, dtype=torch.float64), hermite_features(y.reshape(-1, 1) / math.sqrt(2.0), 8).reshape(batch, views, 8)), 2)
    uncentered = estimate_moments(f, g, centered=False)
    centered = estimate_moments(f, g, centered=True)
    return {
        "uncentered_largest_eigenvalue": float(spectrum(uncentered, 1e-8, "dual").max()),
        "uncentered_logdet": float(logdet_score(uncentered, 1e-8, 1e-12)),
        "centered_largest_eigenvalue": float(spectrum(centered, 1e-3, "dual").max()),
        "centered_effective_rank": int((spectrum(centered, 1e-3, "dual") > 1e-6).sum()),
        "reference_uncentered_trace": float(trace_score(moments, 1e-3)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20263000)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    # Reference plus one-factor axes from the preregistered E3 matrix.
    designs = {(256, 8, 8, 1e-3, True, "dual", "correct")}
    designs.update((batch, 8, 8, 1e-3, True, "dual", "correct") for batch in (64, 128, 256, 512, 1024))
    designs.update((256, views, 8, 1e-3, True, "dual", "correct") for views in (1, 2, 4, 8, 16))
    designs.update((256, 8, features, 1e-3, True, "dual", "correct") for features in (4, 8, 16, 32, 64, 128, 256))
    designs.update((256, 8, 8, ridge, True, "dual", "correct") for ridge in (1e-2, 1e-3, 1e-4, 1e-5))
    designs.update((256, 8, 8, 1e-3, True, whitening, "correct") for whitening in ("none", "left", "right", "dual"))
    designs.update((256, 8, 8, 1e-3, centered, "dual", "correct") for centered in (False, True))
    designs.update((256, 8, 8, 1e-3, True, "dual", rg) for rg in ("correct", "outer_of_conditional_mean"))
    for index, design in enumerate(sorted(designs, key=str)):
        records.append(row(*design[:-1], args.seed + index, rg=design[-1]))
    control = constant_mode_control(4096, 8, args.seed + 10000)
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "records": records,
        "constant_mode_failure_control": control,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir:
        with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "e3_numerical_ablation", "conditions": len(records)}) + "\n")
    print(json.dumps({"conditions": len(records), "constant_mode_failure_control": control}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
