#!/usr/bin/env python3
"""Finite-channel sample-complexity recovery with exact held-out truth."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.analytic import finite_channel_spectrum
from scripts.run_exact_channel_suite import asymmetric_cycle, block_channel, symmetric_channel


def normalized(joint: torch.Tensor) -> torch.Tensor:
    px = joint.sum(1); py = joint.sum(0)
    return px.rsqrt()[:, None] * joint * py.rsqrt()[None, :]


def projector_error(left: torch.Tensor, right: torch.Tensor, modes: int) -> float:
    left_q = torch.linalg.qr(left[:, :modes]).Q
    right_q = torch.linalg.qr(right[:, :modes]).Q
    return float(torch.linalg.matrix_norm(left_q @ left_q.T - right_q @ right_q.T) / math.sqrt(2.0 * modes))


def condition(family: str) -> torch.Tensor:
    if family == "near_identity":
        return symmetric_channel(16, 0.01)
    if family == "block":
        return block_channel(8, 0.05)
    if family == "near_independent":
        uniform = torch.full((16, 16), 1.0 / 16, dtype=torch.float64)
        return 0.99 * uniform + 0.01 * torch.eye(16, dtype=torch.float64)
    if family == "asymmetric_cycle":
        return asymmetric_cycle(16, 0.8, 0.05)
    raise ValueError(family)


def one_record(family: str, samples: int, replicate: int, seed: int) -> dict[str, object]:
    transition = condition(family)
    states, outputs = transition.shape
    px = torch.full((states,), 1.0 / states, dtype=torch.float64)
    truth_joint = px[:, None] * transition
    truth = finite_channel_spectrum(truth_joint)
    true_normalized = normalized(truth_joint)
    true_u, true_s, true_vh = torch.linalg.svd(true_normalized)
    generator = torch.Generator().manual_seed(seed)
    x = torch.multinomial(px, samples, replacement=True, generator=generator)
    y = torch.multinomial(transition[x], 1, replacement=True, generator=generator)[:, 0]
    # Jeffreys smoothing prevents undefined rare-symbol marginals without hiding it.
    counts = torch.bincount(x * outputs + y, minlength=states * outputs).reshape(states, outputs).double() + 0.5
    empirical_joint = counts / counts.sum()
    estimate = finite_channel_spectrum(empirical_joint)
    estimate_normalized = normalized(empirical_joint)
    estimate_u, estimate_s, estimate_vh = torch.linalg.svd(estimate_normalized)
    count = min(len(truth.eigenvalues), len(estimate.eigenvalues))
    modes = min(4, count)
    informative_modes = int((truth.eigenvalues[:count] > 1e-10).sum())
    projector_modes = count if family in {"near_identity", "near_independent"} else min(modes, informative_modes)
    reconstructed_normalized = estimate_u[:, : modes + 1] @ torch.diag(estimate_s[: modes + 1]) @ estimate_vh[: modes + 1]
    estimated_ratio = reconstructed_normalized / (
        estimate.p_x.sqrt()[:, None] * estimate.p_y.sqrt()[None, :]
    )
    true_ratio = truth_joint / (truth.p_x[:, None] * truth.p_y[None, :])
    density_weight = truth.p_x[:, None] * truth.p_y[None, :]
    oracle_normalized = true_u[:, : modes + 1] @ torch.diag(true_s[: modes + 1]) @ true_vh[: modes + 1]
    oracle_ratio = oracle_normalized / (truth.p_x.sqrt()[:, None] * truth.p_y.sqrt()[None, :])
    density_error = float(((estimated_ratio - true_ratio).square() * density_weight).sum().sqrt())
    oracle_density_error = float(((oracle_ratio - true_ratio).square() * density_weight).sum().sqrt())
    return {
        "family": family, "samples": samples, "replicate": replicate, "seed": seed,
        "jeffreys_pseudocount_per_cell": 0.5, "reconstruction_modes_with_constant": modes + 1,
        "projector_modes": projector_modes,
        "spectrum_mae": float((estimate.eigenvalues[:count] - truth.eigenvalues[:count]).abs().mean()),
        "spectrum_relative_l1": float(
            (estimate.eigenvalues[:count] - truth.eigenvalues[:count]).abs().sum()
            / truth.eigenvalues[:count].sum().clamp_min(1e-12)
        ),
        "top_projector_error": projector_error(estimate_u[:, 1:], true_u[:, 1:], projector_modes),
        "density_ratio_weighted_rmse": density_error,
        "oracle_truncation_density_ratio_weighted_rmse": oracle_density_error,
        "density_ratio_excess_rmse": density_error - oracle_density_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="")
    parser.add_argument("--replicates", type=int, default=20); parser.add_argument("--seed", type=int, default=20302000)
    args = parser.parse_args(); records = []
    for family_index, family in enumerate(("near_identity", "block", "near_independent", "asymmetric_cycle")):
        for samples in (500, 2000, 10000):
            for replicate in range(args.replicates):
                seed = args.seed + family_index * 100000 + samples * 10 + replicate
                records.append(one_record(family, samples, replicate, seed))
    payload = {
        "parameters": vars(args), "sample_sizes": [500, 2000, 10000],
        "protocol": "empirical joint matrix with Jeffreys smoothing; exact truth is used only for held-out error reporting",
        "records": records,
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "finite_sample_recovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "finite_sample_recovery", "conditions": len(records)}) + "\n")
    print(json.dumps({"conditions": len(records), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
