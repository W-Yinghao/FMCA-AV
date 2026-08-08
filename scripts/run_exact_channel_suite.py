#!/usr/bin/env python3
"""Exact spectra for the preregistered finite discrete channel families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fmca_av.analytic import finite_channel_spectrum


def symmetric_channel(states: int, error: float) -> torch.Tensor:
    transition = torch.full((states, states), error / (states - 1), dtype=torch.float64)
    transition.diagonal().fill_(1.0 - error)
    return transition


def erasure_channel(states: int, probability: float) -> torch.Tensor:
    transition = torch.zeros(states, states + 1, dtype=torch.float64)
    transition[torch.arange(states), torch.arange(states)] = 1.0 - probability
    transition[:, -1] = probability
    return transition


def block_channel(states: int, cross_probability: float) -> torch.Tensor:
    half = states // 2
    transition = torch.empty(states, states, dtype=torch.float64)
    for source in range(states):
        same = slice(0, half) if source < half else slice(half, states)
        other = slice(half, states) if source < half else slice(0, half)
        transition[source, same] = (1.0 - cross_probability) / half
        transition[source, other] = cross_probability / half
    return transition


def asymmetric_cycle(states: int, forward: float, backward: float) -> torch.Tensor:
    transition = torch.zeros(states, states, dtype=torch.float64)
    stay = 1.0 - forward - backward
    for source in range(states):
        transition[source, source] = stay
        transition[source, (source + 1) % states] = forward
        transition[source, (source - 1) % states] = backward
    return transition


def record(family: str, parameter: dict[str, object], transition: torch.Tensor) -> dict[str, object]:
    marginal = torch.full((transition.shape[0],), 1.0 / transition.shape[0], dtype=torch.float64)
    joint = marginal[:, None] * transition
    spectrum = finite_channel_spectrum(joint)
    singular = spectrum.singular_values
    nonconstant = singular[1:] if len(singular) and abs(float(singular[0]) - 1.0) < 1e-8 else singular
    return {
        "family": family,
        "parameter": parameter,
        "input_states": transition.shape[0],
        "output_states": transition.shape[1],
        "singular_values": singular.tolist(),
        "nonconstant_eigenvalues": nonconstant.square().tolist(),
        "trace_dependence": float(nonconstant.square().sum()),
        "rank": int((singular > 1e-12).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for error in (0.001, 0.01, 0.05, 0.1, 0.25, 0.49):
        records.append(record("binary_symmetric", {"error": error}, symmetric_channel(2, error)))
    for states in (4, 8, 16):
        for error in (0.01, 0.1, 0.25, 0.5):
            records.append(record("qary_symmetric", {"states": states, "error": error}, symmetric_channel(states, error)))
    for probability in (0.01, 0.1, 0.25, 0.5, 0.9):
        records.append(record("erasure", {"states": 4, "erasure_probability": probability}, erasure_channel(4, probability)))
    for probability in (0.001, 0.01, 0.05, 0.2, 0.49):
        records.append(record("block", {"states": 8, "cross_probability": probability}, block_channel(8, probability)))
    for error in (1e-6, 1e-4, 1e-2):
        records.append(record("near_identity", {"states": 16, "error": error}, symmetric_channel(16, error)))
    uniform = torch.full((16, 16), 1.0 / 16, dtype=torch.float64)
    identity = torch.eye(16, dtype=torch.float64)
    for strength in (1e-6, 1e-4, 1e-2, 0.1):
        transition = (1.0 - strength) * uniform + strength * identity
        records.append(record("near_independent", {"states": 16, "identity_strength": strength}, transition))
    for forward, backward in ((0.6, 0.1), (0.8, 0.05), (0.95, 0.01)):
        records.append(record("asymmetric_cycle", {"states": 16, "forward": forward, "backward": backward}, asymmetric_cycle(16, forward, backward)))
    payload = {"records": records, "families": sorted({item["family"] for item in records})}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    print(json.dumps({"families": payload["families"], "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
