#!/usr/bin/env python3
"""E1 multivariate Gaussian exact-spectrum and sample-complexity suite."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.operators import fit_spectral_calibration


def channel_matrix(case: str, dimension: int, generator: torch.Generator) -> torch.Tensor:
    if case == "isotropic":
        return 0.75 * torch.eye(dimension, dtype=torch.float64)
    if case == "repeated":
        diagonal = torch.tensor(([0.9] * (dimension // 2)) + ([0.45] * (dimension - dimension // 2)), dtype=torch.float64)
    elif case == "anisotropic":
        diagonal = torch.linspace(0.95, 0.15, dimension, dtype=torch.float64)
    elif case == "ill_conditioned":
        diagonal = torch.logspace(math.log10(0.99), math.log10(0.01), dimension, dtype=torch.float64)
    elif case == "low_rank":
        rank = max(2, dimension // 5)
        diagonal = torch.cat((torch.linspace(0.9, 0.45, rank, dtype=torch.float64),
                              torch.zeros(dimension - rank, dtype=torch.float64)))
    else:
        raise ValueError(case)
    left, _ = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator, dtype=torch.float64))
    return left @ torch.diag(diagonal) @ right.T


def covariance(condition: float, dimension: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.logspace(0.0, -math.log10(condition), dimension, dtype=torch.float64)
    square_root = torch.diag(values.sqrt()); inverse_square_root = torch.diag(values.rsqrt())
    return torch.diag(values), square_root, inverse_square_root


def truth(matrix: torch.Tensor, noise: float, covariance_x: torch.Tensor,
          square_root_x: torch.Tensor, inverse_square_root_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    covariance_y = matrix @ covariance_x @ matrix.T + noise * torch.eye(matrix.shape[0], dtype=torch.float64)
    values, vectors = torch.linalg.eigh(covariance_y)
    inverse_sqrt = (vectors * values.rsqrt()) @ vectors.T
    operator = square_root_x @ matrix.T @ inverse_sqrt
    u, singular, _ = torch.linalg.svd(operator)
    return singular.square(), inverse_square_root_x @ u


def projector_error(left: torch.Tensor, right: torch.Tensor, modes: int) -> float:
    left_q = torch.linalg.qr(left[:, :modes]).Q
    right_q = torch.linalg.qr(right[:, :modes]).Q
    left_projector = left_q @ left_q.T; right_projector = right_q @ right_q.T
    return float(torch.linalg.matrix_norm(left_projector - right_projector) / math.sqrt(2.0 * modes))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20270000)
    parser.add_argument("--sample-sizes", default="256,1024,4096")
    parser.add_argument("--case-indices", default="", help="comma-separated zero-based subset of the frozen case list")
    args = parser.parse_args()
    cases = [
        ("isotropic", 1, 0.1, 1.0), ("isotropic", 2, 1.0, 1.0), ("anisotropic", 2, 1.0, 1.0),
        ("repeated", 10, 1.0, 1.0), ("anisotropic", 10, 1.0, 1.0),
        ("low_rank", 20, 1.0, 1.0), ("anisotropic", 50, 1.0, 1.0),
        ("ill_conditioned", 100, 1.0, 1e6), ("anisotropic", 5, 1.0, 1.0),
        ("ill_conditioned", 20, 1.0, 1e3),
    ]
    indexed_cases = list(enumerate(cases))
    if args.case_indices:
        indices = [int(value) for value in args.case_indices.split(",") if value]
        indexed_cases = [indexed_cases[index] for index in indices]
    sample_sizes = tuple(int(value) for value in args.sample_sizes.split(",") if value)
    if not sample_sizes or any(value < 2 for value in sample_sizes):
        raise ValueError("--sample-sizes must contain positive integers >= 2")
    records = []
    for case_index, (case, dimension, noise, covariance_condition) in indexed_cases:
        matrix_generator = torch.Generator().manual_seed(args.seed + case_index)
        matrix = channel_matrix(case, dimension, matrix_generator)
        covariance_x, square_root_x, inverse_square_root_x = covariance(covariance_condition, dimension)
        true_eigenvalues, true_left = truth(matrix, noise, covariance_x, square_root_x, inverse_square_root_x)
        for samples in sample_sizes:
            for views in (1, 4, 16):
                for replicate in range(args.replicates):
                    generator = torch.Generator().manual_seed(args.seed + 100000 + case_index * 10000 + samples + views * 100 + replicate)
                    x = torch.randn(samples, dimension, generator=generator, dtype=torch.float64) @ square_root_x
                    epsilon = torch.randn(samples, views, dimension, generator=generator, dtype=torch.float64)
                    y = torch.einsum("bd,ed->be", x, matrix).unsqueeze(1) + math.sqrt(noise) * epsilon
                    calibration = fit_spectral_calibration(x, y, ridge=1e-6, centered=True)
                    estimate = calibration.eigenvalues
                    count = min(len(estimate), len(true_eigenvalues))
                    modes = min(5, int((true_eigenvalues > 1e-10).sum()))
                    records.append({
                        "case": case, "dimension": dimension, "noise_variance": noise,
                        "covariance_condition": covariance_condition,
                        "signal_rank": int(torch.linalg.matrix_rank(matrix)),
                        "samples": samples, "views": views, "replicate": replicate,
                        "spectrum_mae": float((estimate[:count] - true_eigenvalues[:count]).abs().mean()),
                        "spectrum_relative_l1": float((estimate[:count] - true_eigenvalues[:count]).abs().sum() / true_eigenvalues[:count].sum().clamp_min(1e-12)),
                        "top_projector_error": projector_error(
                            square_root_x @ calibration.transform_f, square_root_x @ true_left, modes,
                        ),
                        "estimated_eigenvalues": estimate.tolist(),
                        "ground_truth_eigenvalues": true_eigenvalues.tolist(),
                    })
    payload = {"parameters": vars(args), "records": records}
    output = (
        Path(args.output) if args.output else
        Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "e1_high_resource_gaussian.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "e1_gaussian_operator", "conditions": len(records)}) + "\n")
    print(json.dumps({"cases": len(indexed_cases), "conditions": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
