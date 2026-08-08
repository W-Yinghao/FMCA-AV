#!/usr/bin/env python3
"""Run E8 exact direct-vs-composed Markov diagnostics."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from fmca_av.markov import (
    directed_cycle,
    lag_composition_diagnostic,
    metastable_chain,
    nonnormal_chain,
    reversible_chain,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for replicate in range(args.replicates):
        generator = torch.Generator().manual_seed(20261000 + replicate)
        chains = {
            "reversible": reversible_chain(args.states, generator),
            "metastable_reversible": metastable_chain(args.states, generator),
            "directed_cycle_normal": directed_cycle(args.states),
            "directed_nonnormal": nonnormal_chain(args.states, generator),
        }
        for name, transition in chains.items():
            records.append(
                {
                    "replicate": replicate,
                    "chain": name,
                    **lag_composition_diagnostic(transition, [2, 4, 8], modes=8),
                }
            )
    summary = {}
    for name in sorted({record["chain"] for record in records}):
        selected = [record for record in records if record["chain"] == name]
        summary[name] = {}
        for lag in [2, 4, 8]:
            values = [next(item for item in record["lags"] if item["lag"] == lag)["mae"] for record in selected]
            summary[name][str(lag)] = {
                "mae_mean": statistics.fmean(values),
                "mae_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "mae_max": max(values),
            }
    payload = {"replicates": args.replicates, "states": args.states, "summary": summary, "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
