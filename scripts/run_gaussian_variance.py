#!/usr/bin/env python3
"""Monte Carlo variance study with exact Gaussian Hermite eigenfunctions."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import torch

from fmca_av.objectives import trace_score
from fmca_av.operators import FMCAMoments, estimate_moments


def hermite_features(values: torch.Tensor, dimension: int) -> torch.Tensor:
    features = []
    previous = torch.ones_like(values)
    current = values
    features.append(current)
    for degree in range(1, dimension):
        following = (values * current - math.sqrt(degree) * previous) / math.sqrt(degree + 1)
        features.append(following)
        previous, current = current, following
    return torch.cat(features, dim=1)


def run_condition(
    parents: int,
    views: int,
    repetitions: int,
    feature_dim: int,
    noise_variance: float,
    ridge: float,
    seed: int,
) -> Dict[str, float]:
    correlation = 1.0 / math.sqrt(1.0 + noise_variance)
    singular = torch.tensor(
        [correlation ** degree for degree in range(1, feature_dim + 1)], dtype=torch.float64
    )
    identity = torch.eye(feature_dim, dtype=torch.float64)
    true_cross = torch.diag(singular)
    truth = FMCAMoments(identity, identity, true_cross, parents, parents * views, True)
    true_score = float(trace_score(truth, ridge=ridge))
    errors_f: List[float] = []
    errors_g: List[float] = []
    errors_p: List[float] = []
    scores: List[float] = []
    for repetition in range(repetitions):
        generator = torch.Generator().manual_seed(seed + repetition)
        x = torch.randn(parents, 1, generator=generator, dtype=torch.float64)
        noise = torch.randn(parents, views, 1, generator=generator, dtype=torch.float64)
        y = x.unsqueeze(1) + math.sqrt(noise_variance) * noise
        f = hermite_features(x, feature_dim)
        g = hermite_features(y.reshape(-1, 1) / math.sqrt(1.0 + noise_variance), feature_dim)
        g = g.reshape(parents, views, feature_dim)
        moments = estimate_moments(f, g, centered=True)
        errors_f.append(float((moments.r_f - identity).square().mean()))
        errors_g.append(float((moments.r_g - identity).square().mean()))
        errors_p.append(float((moments.p_fg - true_cross).square().mean()))
        scores.append(float(trace_score(moments, ridge=ridge)))
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    return {
        "parents": parents,
        "views": views,
        "total_views": parents * views,
        "repetitions": repetitions,
        "rf_mse_mean": float(torch.tensor(errors_f).mean()),
        "rg_mse_mean": float(torch.tensor(errors_g).mean()),
        "pfg_mse_mean": float(torch.tensor(errors_p).mean()),
        "trace_true": true_score,
        "trace_mean": float(score_tensor.mean()),
        "trace_bias": float(score_tensor.mean() - true_score),
        "trace_variance": float(score_tensor.var(unbiased=True)),
        "trace_std": float(score_tensor.std(unbiased=True)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--parents", type=int, default=1024)
    parser.add_argument("--total-view-budget", type=int, default=8192)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.output:
        output = Path(args.output).resolve()
    elif os.environ.get("FMCA_HARNESS_RUN_DIR"):
        output = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "gaussian_variance.csv"
    else:
        raise ValueError("--output is required outside the harness")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for design in ("fixed_parent", "fixed_total_views"):
        for views in (1, 2, 4, 8, 16):
            parents = args.parents if design == "fixed_parent" else args.total_view_budget // views
            row: Dict[str, object] = {"design": design}
            row.update(run_condition(
                parents,
                views,
                args.repetitions,
                args.feature_dim,
                args.noise_variance,
                args.ridge,
                args.seed + views * 10000 + (0 if design == "fixed_parent" else 1000000),
            ))
            rows.append(row)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"output": str(output), "conditions": rows}
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

