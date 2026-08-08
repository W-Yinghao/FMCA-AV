#!/usr/bin/env python3
"""Empirical direct-vs-composed lag diagnostics for OU and double-well dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from fmca_av.markov import lag_spectrum


def simulate_ou(length: int, step: float, generator: torch.Generator) -> torch.Tensor:
    rho = torch.exp(torch.tensor(-step, dtype=torch.float64))
    scale = torch.sqrt(1.0 - rho.square())
    values = torch.empty(length, dtype=torch.float64)
    values[0] = torch.randn((), generator=generator, dtype=torch.float64)
    noise = torch.randn(length - 1, generator=generator, dtype=torch.float64)
    for index in range(1, length):
        values[index] = rho * values[index - 1] + scale * noise[index - 1]
    return values


def simulate_double_well(
    length: int,
    step: float,
    diffusion: float,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.empty(length, dtype=torch.float64)
    values[0] = torch.randn((), generator=generator, dtype=torch.float64)
    noise = torch.randn(length - 1, generator=generator, dtype=torch.float64)
    scale = (2.0 * diffusion * step) ** 0.5
    for index in range(1, length):
        previous = values[index - 1]
        drift = previous - previous.pow(3)
        candidate = previous + step * drift + scale * noise[index - 1]
        # A numerical guard only; the selected step is stable in normal operation.
        values[index] = candidate.clamp(-6.0, 6.0)
    return values


def discretize(values: torch.Tensor, bins: int) -> torch.Tensor:
    probabilities = torch.linspace(0, 1, bins + 1, dtype=torch.float64)[1:-1]
    boundaries = torch.quantile(values, probabilities)
    return torch.bucketize(values, boundaries).long()


def transition_at_lag(states: torch.Tensor, bins: int, lag: int, pseudocount: float) -> torch.Tensor:
    counts = torch.full((bins, bins), pseudocount, dtype=torch.float64)
    flat = states[:-lag] * bins + states[lag:]
    counts += torch.bincount(flat, minlength=bins * bins).reshape(bins, bins)
    return counts / counts.sum(dim=1, keepdim=True)


def weighted_transition_error(direct: torch.Tensor, composed: torch.Tensor) -> float:
    stationary = direct.sum(dim=0)
    stationary = stationary / stationary.sum()
    return float(torch.sqrt((stationary[:, None] * (direct - composed).square()).sum()))


def diagnostic(
    trajectory: torch.Tensor,
    bins: int,
    lags: list[int],
    modes: int,
    pseudocount: float,
) -> dict[str, object]:
    states = discretize(trajectory, bins)
    one_step = transition_at_lag(states, bins, 1, pseudocount)
    records = []
    for lag in lags:
        direct = transition_at_lag(states, bins, lag, pseudocount)
        composed = torch.linalg.matrix_power(one_step, lag)
        direct_spectrum = lag_spectrum(direct, 1).singular_values[:modes]
        composed_spectrum = lag_spectrum(composed, 1).singular_values[:modes]
        records.append(
            {
                "lag": lag,
                "direct_singular_values": direct_spectrum.tolist(),
                "composed_singular_values": composed_spectrum.tolist(),
                "spectrum_mae": float((direct_spectrum - composed_spectrum).abs().mean()),
                "transition_weighted_l2": weighted_transition_error(direct, composed),
                "chapman_kolmogorov_max_abs": float((direct - composed).abs().max()),
            }
        )
    return {"bins": bins, "trajectory_length": len(trajectory), "lags": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=100_000)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--diffusion", type=float, default=0.18)
    parser.add_argument("--pseudocount", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for replicate in range(args.replicates):
        generator = torch.Generator().manual_seed(20262000 + replicate)
        dynamics = {
            "ornstein_uhlenbeck": simulate_ou(args.length, args.step, generator),
            "double_well": simulate_double_well(args.length, args.step, args.diffusion, generator),
        }
        for name, trajectory in dynamics.items():
            records.append(
                {
                    "replicate": replicate,
                    "dynamics": name,
                    **diagnostic(trajectory, args.bins, [2, 4, 8, 16], 8, args.pseudocount),
                }
            )
    summary: dict[str, object] = {}
    for name in ("ornstein_uhlenbeck", "double_well"):
        selected = [record for record in records if record["dynamics"] == name]
        summary[name] = {}
        for lag in [2, 4, 8, 16]:
            per_lag = [next(item for item in record["lags"] if item["lag"] == lag) for record in selected]
            summary[name][str(lag)] = {}
            for metric in ("spectrum_mae", "transition_weighted_l2", "chapman_kolmogorov_max_abs"):
                values = [float(item[metric]) for item in per_lag]
                summary[name][str(lag)][metric] = {
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                    "max": max(values),
                }
    payload = {
        "parameters": vars(args),
        "summary": summary,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
