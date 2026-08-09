#!/usr/bin/env python3
"""Frozen-feature loss/gradient bias-variance for conditional Gaussian sampling."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.objectives import trace_score
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION, estimate_moments


def hermite(values: torch.Tensor, dimension: int) -> torch.Tensor:
    columns = []
    previous = torch.ones_like(values)
    current = values
    for degree in range(1, dimension + 1):
        columns.append(current)
        following = (values * current - math.sqrt(degree) * previous) / math.sqrt(degree + 1)
        previous, current = current, following
    return torch.cat(columns, 1)


def sample_features(parents: int, views: int, dimension: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(parents, 1, generator=generator, dtype=torch.float64)
    y = x[:, None, :] + torch.randn(parents, views, 1, generator=generator, dtype=torch.float64)
    f = hermite(x, dimension)
    g = hermite(y.reshape(-1, 1) / math.sqrt(2.0), dimension).reshape(parents, views, dimension)
    return f, g


def score_gradient(f: torch.Tensor, g: torch.Tensor, parameter: torch.Tensor, ridge: float) -> tuple[float, torch.Tensor]:
    transformed_f = f @ parameter
    transformed_g = g @ parameter
    score = trace_score(estimate_moments(transformed_f, transformed_g, centered=True), ridge)
    gradient, = torch.autograd.grad(score, parameter)
    return float(score.detach()), gradient.detach().flatten()


def condition(parents: int, views: int, repetitions: int, dimension: int, ridge: float, seed: int, reference_score: float, reference_gradient: torch.Tensor) -> dict[str, object]:
    scores = []
    gradients = []
    for repetition in range(repetitions):
        generator = torch.Generator().manual_seed(seed + repetition)
        f, g = sample_features(parents, views, dimension, generator)
        parameter = torch.eye(dimension, dtype=torch.float64, requires_grad=True)
        score, gradient = score_gradient(f, g, parameter, ridge)
        scores.append(score)
        gradients.append(gradient)
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    gradient_tensor = torch.stack(gradients)
    mean_gradient = gradient_tensor.mean(0)
    errors = gradient_tensor - reference_gradient
    cosine = torch.nn.functional.cosine_similarity(gradient_tensor, reference_gradient[None, :], dim=1)
    return {
        "parents": parents,
        "views": views,
        "total_views": parents * views,
        "repetitions": repetitions,
        "score_reference": reference_score,
        "score_mean": float(score_tensor.mean()),
        "score_bias": float(score_tensor.mean() - reference_score),
        "score_variance": float(score_tensor.var(unbiased=True)),
        "gradient_bias_l2": float(torch.linalg.vector_norm(mean_gradient - reference_gradient)),
        "gradient_variance": float(((gradient_tensor - mean_gradient).square().sum(1)).mean()),
        "gradient_mse_to_reference": float(errors.square().sum(1).mean()),
        "gradient_cosine_to_reference_mean": float(cosine.mean()),
        "gradient_cosine_to_reference_std": float(cosine.std(unbiased=True)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--parents", type=int, default=256)
    parser.add_argument("--total-view-budget", type=int, default=2048)
    parser.add_argument("--reference-parents", type=int, default=32768)
    parser.add_argument("--reference-views", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20264000)
    args = parser.parse_args()
    reference_generator = torch.Generator().manual_seed(args.seed)
    reference_f, reference_g = sample_features(args.reference_parents, args.reference_views, args.feature_dim, reference_generator)
    reference_parameter = torch.eye(args.feature_dim, dtype=torch.float64, requires_grad=True)
    reference_score, reference_gradient = score_gradient(reference_f, reference_g, reference_parameter, args.ridge)
    records = []
    for design in ("fixed_parent", "fixed_total_views"):
        for views in (1, 2, 4, 8, 16):
            parents = args.parents if design == "fixed_parent" else args.total_view_budget // views
            record = condition(
                parents, views, args.repetitions, args.feature_dim, args.ridge,
                args.seed + (0 if design == "fixed_parent" else 1_000_000) + views * 10_000,
                reference_score, reference_gradient,
            )
            record["design"] = design
            records.append(record)
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "reference": {"parents": args.reference_parents, "views": args.reference_views, "score": reference_score, "gradient_norm": float(torch.linalg.vector_norm(reference_gradient))},
        "conditions": records,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "e2_gradient_variance", "conditions": len(records)}) + "\n")
    print(json.dumps({"reference": payload["reference"], "conditions": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
