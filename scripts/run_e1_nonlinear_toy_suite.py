#!/usr/bin/env python3
"""E1 nonlinear-toy empirical operator spectra and sample-complexity suite."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.analytic import finite_channel_spectrum


def generate(family: str, samples: int, noise: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, int]:
    if family == "two_moons":
        component = torch.randint(2, (samples,), generator=generator)
        phase = torch.rand(samples, generator=generator) * math.pi
        x = torch.stack((torch.cos(phase), torch.sin(phase)), 1)
        x[component == 1, 0] = 1.0 - x[component == 1, 0]
        x[component == 1, 1] = 0.5 - x[component == 1, 1]
        parent = component * 16 + torch.clamp((phase / math.pi * 16).long(), max=15)
        states = 32
    elif family == "circles":
        ring = torch.randint(2, (samples,), generator=generator)
        phase = torch.rand(samples, generator=generator) * (2 * math.pi)
        radius = 0.5 + 0.5 * ring
        x = torch.stack((radius * torch.cos(phase), radius * torch.sin(phase)), 1)
        parent = ring * 16 + torch.clamp((phase / (2 * math.pi) * 16).long(), max=15)
        states = 32
    elif family == "spiral":
        arm = torch.randint(3, (samples,), generator=generator)
        phase_unit = torch.rand(samples, generator=generator)
        phase = phase_unit * (3 * math.pi) + arm * (2 * math.pi / 3)
        radius = 0.15 + phase_unit
        x = torch.stack((radius * torch.cos(phase), radius * torch.sin(phase)), 1)
        parent = arm * 16 + torch.clamp((phase_unit * 16).long(), max=15)
        states = 48
    elif family == "gmm":
        component = torch.randint(8, (samples,), generator=generator)
        phase = component.double() * (2 * math.pi / 8)
        centers = torch.stack((2.0 * torch.cos(phase), 2.0 * torch.sin(phase)), 1).float()
        x = centers + 0.15 * torch.randn(samples, 2, generator=generator)
        parent = component
        states = 8
    elif family == "swiss_roll":
        phase_unit = torch.rand(samples, generator=generator)
        height_unit = torch.rand(samples, generator=generator)
        phase = (1.5 + 3.0 * phase_unit) * math.pi
        x = torch.stack((phase * torch.cos(phase) / 12.0, 2.0 * height_unit - 1.0, phase * torch.sin(phase) / 12.0), 1)
        parent = torch.clamp((phase_unit * 16).long(), max=15) * 4 + torch.clamp((height_unit * 4).long(), max=3)
        states = 64
    else:
        raise ValueError(f"unknown family {family}")
    observation = x + noise * torch.randn(x.shape, generator=generator)
    return parent.long(), observation, states


def discretize(values: torch.Tensor, bins: int, mean: torch.Tensor | None = None,
               std: torch.Tensor | None = None) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    # Per-coordinate standardization followed by fixed clipped bins makes spectra
    # comparable across sample sizes without fitting a clustering model.
    fitted_mean = values.mean(0, keepdim=True) if mean is None else mean
    fitted_std = values.std(0, keepdim=True).clamp_min(1e-6) if std is None else std
    normalized = (values - fitted_mean) / fitted_std
    indices = torch.clamp(((normalized + 3.0) / 6.0 * bins).long(), 0, bins - 1)
    multiplier = 1
    flat = torch.zeros(len(values), dtype=torch.long)
    for column in range(indices.shape[1]):
        flat += indices[:, column] * multiplier
        multiplier *= bins
    return flat, multiplier, fitted_mean, fitted_std


def empirical_joint(parent: torch.Tensor, observation: torch.Tensor, parent_states: int, bins: int,
                    pseudocount: float, mean: torch.Tensor | None = None,
                    std: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, output_states, fitted_mean, fitted_std = discretize(observation, bins, mean, std)
    counts = torch.full((parent_states, output_states), pseudocount, dtype=torch.float64)
    counts += torch.bincount(parent * output_states + output, minlength=parent_states * output_states).reshape(parent_states, output_states)
    return counts / counts.sum(), fitted_mean, fitted_std


def normalized(joint: torch.Tensor) -> torch.Tensor:
    px = joint.sum(1); py = joint.sum(0)
    return px.rsqrt()[:, None] * joint * py.rsqrt()[None, :]


def projector_error(left: torch.Tensor, right: torch.Tensor, modes: int) -> float:
    left_q = torch.linalg.qr(left[:, :modes]).Q; right_q = torch.linalg.qr(right[:, :modes]).Q
    return float(torch.linalg.matrix_norm(left_q @ left_q.T - right_q @ right_q.T) / math.sqrt(2.0 * modes))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--reference-samples", type=int, default=200000)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20265000)
    args = parser.parse_args()
    families = ("two_moons", "circles", "spiral", "gmm", "swiss_roll")
    noises = (0.05, 0.2, 0.5, 1.0)
    sample_sizes = (2000, 10000, 50000)
    references: dict[str, list[float]] = {}
    reference_data: dict[str, dict[str, torch.Tensor]] = {}
    for family_index, family in enumerate(families):
        for noise_index, noise in enumerate(noises):
            generator = torch.Generator().manual_seed(args.seed + family_index * 1000 + noise_index)
            parent, observation, states = generate(family, args.reference_samples, noise, generator)
            joint, mean, std = empirical_joint(parent, observation, states, args.bins, 0.05)
            reference_spectrum = finite_channel_spectrum(joint)
            reference_normalized = normalized(joint); u, singular, vh = torch.linalg.svd(reference_normalized)
            key = f"{family}:{noise}"
            references[key] = reference_spectrum.eigenvalues.tolist()
            reference_data[key] = {"joint": joint, "mean": mean, "std": std, "u": u,
                                   "singular": singular, "vh": vh}
    records = []
    for family_index, family in enumerate(families):
        for noise_index, noise in enumerate(noises):
            key = f"{family}:{noise}"; truth = torch.tensor(references[key], dtype=torch.float64)
            reference = reference_data[key]
            for samples in sample_sizes:
                for replicate in range(args.replicates):
                    generator = torch.Generator().manual_seed(args.seed + 100000 + family_index * 10000 + noise_index * 1000 + samples + replicate)
                    parent, observation, states = generate(family, samples, noise, generator)
                    joint, _, _ = empirical_joint(
                        parent, observation, states, args.bins, 0.05,
                        reference["mean"], reference["std"],
                    )
                    estimate_spectrum = finite_channel_spectrum(joint); estimate = estimate_spectrum.eigenvalues
                    count = min(len(estimate), len(truth), 16)
                    estimate_normalized = normalized(joint); estimate_u, estimate_s, estimate_vh = torch.linalg.svd(estimate_normalized)
                    modes = max(1, min(count, int((truth[:count] > 1e-10).sum())))
                    reconstruction_modes = modes + 1
                    estimate_reconstruction = estimate_u[:, :reconstruction_modes] @ torch.diag(estimate_s[:reconstruction_modes]) @ estimate_vh[:reconstruction_modes]
                    estimate_ratio = estimate_reconstruction / (estimate_spectrum.p_x.sqrt()[:, None] * estimate_spectrum.p_y.sqrt()[None, :])
                    truth_joint = reference["joint"]; truth_spectrum = finite_channel_spectrum(truth_joint)
                    truth_ratio = truth_joint / (truth_spectrum.p_x[:, None] * truth_spectrum.p_y[None, :])
                    oracle_reconstruction = reference["u"][:, :reconstruction_modes] @ torch.diag(reference["singular"][:reconstruction_modes]) @ reference["vh"][:reconstruction_modes]
                    oracle_ratio = oracle_reconstruction / (truth_spectrum.p_x.sqrt()[:, None] * truth_spectrum.p_y.sqrt()[None, :])
                    density_weight = truth_spectrum.p_x[:, None] * truth_spectrum.p_y[None, :]
                    density_error = float(((estimate_ratio - truth_ratio).square() * density_weight).sum().sqrt())
                    oracle_density_error = float(((oracle_ratio - truth_ratio).square() * density_weight).sum().sqrt())
                    records.append({
                        "family": family,
                        "noise": noise,
                        "samples": samples,
                        "replicate": replicate,
                        "top16_spectrum_mae": float((estimate[:count] - truth[:count]).abs().mean()),
                        "top16_spectrum_relative_l1": float((estimate[:count] - truth[:count]).abs().sum() / truth[:count].abs().sum().clamp_min(1e-12)),
                        "top_subspace_projector_error": projector_error(estimate_u[:, 1:], reference["u"][:, 1:], modes),
                        "density_ratio_weighted_rmse": density_error,
                        "oracle_truncation_density_ratio_weighted_rmse": oracle_density_error,
                        "density_ratio_excess_rmse": density_error - oracle_density_error,
                        "trace_dependence": float(estimate.sum()),
                        "effective_rank": int((estimate > 1e-5).sum()),
                        "leading_eigenvalues": estimate[:16].tolist(),
                    })
    payload = {"parameters": vars(args), "references": references, "records": records}
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "e1_nonlinear_toy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "e1_nonlinear_toy", "conditions": len(records)}) + "\n")
    print(json.dumps({"families": families, "conditions": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
