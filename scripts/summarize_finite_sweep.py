#!/usr/bin/env python3
"""Aggregate completed finite-channel seed runs without inventing missing values."""

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.runs).glob("*_e1-finite-reference-seed-*/artifacts/evaluation.json")):
        status_path = path.parents[1] / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "SUCCEEDED":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "run_id": status["run_id"],
                "eigenvalue_mae": result.get("eigenvalue_mae"),
                "eigenvalue_max_error": result.get("eigenvalue_max_error"),
                "empirical_eigenvalues": result.get("empirical_eigenvalues"),
            }
        )

    maes = [record["eigenvalue_mae"] for record in records if record["eigenvalue_mae"] is not None]
    payload = {
        "completed_runs": len(records),
        "runs_with_eigenvalue_mae": len(maes),
        "eigenvalue_mae_mean": statistics.fmean(maes) if maes else None,
        "eigenvalue_mae_median": statistics.median(maes) if maes else None,
        "eigenvalue_mae_sample_std": statistics.stdev(maes) if len(maes) > 1 else None,
        "eigenvalue_mae_min": min(maes) if maes else None,
        "eigenvalue_mae_max": max(maes) if maes else None,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
