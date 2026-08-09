#!/usr/bin/env python3
"""E8 exact and continuous Markov boundary-condition sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.markov import directed_cycle, lag_composition_diagnostic, metastable_chain, nonnormal_chain, reversible_chain
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.run_markov_continuous import diagnostic


def simulate(dynamics: str, length: int, step: float, diffusion: float, initial: str, generator: torch.Generator) -> torch.Tensor:
    values = torch.empty(length, dtype=torch.float64)
    if initial == "non_equilibrium":
        values[0] = 4.0
    else:
        values[0] = torch.randn((), generator=generator, dtype=torch.float64)
    noise = torch.randn(length - 1, generator=generator, dtype=torch.float64)
    for index in range(1, length):
        previous = values[index - 1]
        if dynamics == "ornstein_uhlenbeck":
            rho = math.exp(-step)
            values[index] = rho * previous + math.sqrt(1.0 - rho * rho) * noise[index - 1]
            continue
        if dynamics == "double_well":
            drift = previous - previous.pow(3)
        elif dynamics == "multi_well":
            drift = -torch.sin(3.0 * previous) - 0.08 * previous
        else:
            raise ValueError(dynamics)
        values[index] = (previous + step * drift + math.sqrt(2.0 * diffusion * step) * noise[index - 1]).clamp(-8.0, 8.0)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20266000)
    args = parser.parse_args()
    exact_records = []
    for states in (8, 20, 50):
        for replicate in range(20):
            generator = torch.Generator().manual_seed(args.seed + states * 100 + replicate)
            chains = {
                "reversible": reversible_chain(states, generator),
                "metastable_reversible": metastable_chain(states if states % 2 == 0 else states + 1, generator),
                "directed_cycle_normal": directed_cycle(states),
                "directed_nonnormal": nonnormal_chain(states, generator),
            }
            for name, transition in chains.items():
                exact_records.append({"states": states, "replicate": replicate, "chain": name, **lag_composition_diagnostic(transition, [2, 4, 8, 16], modes=8)})
    base = {"length": 100000, "step": 0.02, "bins": 32, "initial": "equilibrium",
            "diffusion": 0.18, "observation_stride": 1}
    conditions = []
    for length in (10000, 50000, 200000): conditions.append({**base, "length": length, "axis": "trajectory_length"})
    for step in (0.01, 0.05): conditions.append({**base, "step": step, "axis": "sampling_step"})
    for bins in (16, 64): conditions.append({**base, "bins": bins, "axis": "discretization"})
    for diffusion in (0.05, 0.5): conditions.append({**base, "diffusion": diffusion, "axis": "diffusion_noise"})
    for stride in (2, 4):
        conditions.append({**base, "length": 50000, "observation_stride": stride,
                           "axis": "observation_interval"})
    conditions.append({**base, "initial": "non_equilibrium", "axis": "initial_distribution"})
    continuous_records = []
    for condition_index, condition in enumerate(conditions):
        for replicate in range(args.replicates):
            for dynamics_index, dynamics in enumerate(("ornstein_uhlenbeck", "double_well", "multi_well")):
                generator = torch.Generator().manual_seed(args.seed + 1_000_000 + condition_index * 1000 + replicate * 10 + dynamics_index)
                stride = int(condition.get("observation_stride", 1))
                trajectory = simulate(
                    dynamics, int(condition["length"]) * stride, condition["step"],
                    condition["diffusion"], condition["initial"], generator,
                )[::stride]
                continuous_records.append({
                    "condition": condition,
                    "replicate": replicate,
                    "dynamics": dynamics,
                    **diagnostic(trajectory, condition["bins"], [2, 4, 8, 16], 8, 0.25),
                })
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "exact_records": exact_records,
        "continuous_records": continuous_records,
        "parameters": vars(args),
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "e8_markov_full.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "stage": "e8_markov_full_sweep",
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
                "exact_conditions": len(exact_records),
                "continuous_conditions": len(continuous_records),
            }) + "\n")
    print(json.dumps({"exact_conditions": len(exact_records), "continuous_conditions": len(continuous_records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
