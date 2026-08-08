#!/usr/bin/env python3
"""Aggregate E7 factor-probe curves and preregistered spectral controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def channel_from_name(name: str, dataset: str) -> str:
    prefix = f"e7-{dataset}-"
    if name.startswith(prefix) and name.endswith("-factor-probe"):
        return name[len(prefix):-len("-factor-probe")]
    full_prefix = f"full-factor-{dataset}-"
    if name.startswith(full_prefix):
        value = name[len(full_prefix):]
        return value.rsplit("-seed", 1)[0]
    return "default"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", default="runs"); parser.add_argument("--output-dir", default="results/e7")
    args = parser.parse_args(); runs = Path(args.runs).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    curves = []
    for path in sorted(runs.glob("*/artifacts/factor_probe.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); run_dir = path.parents[1]; status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        dataset = str(payload["dataset"]); channel = channel_from_name(str(status["name"]), dataset); names = payload["factor_names"]
        grouped = defaultdict(list)
        for record in payload["records"]: grouped[(record["selection"], int(record["k"]), int(record["factor_index"]))].append(float(record["accuracy"]))
        for (selection, k, factor), values in grouped.items():
            curves.append({"run_id": status["run_id"], "dataset": dataset, "channel": channel, "selection": selection, "k": k, "factor_index": factor, "factor_name": names[factor], "accuracy_mean": statistics.fmean(values), "accuracy_std": statistics.stdev(values) if len(values) > 1 else 0.0, "repeats": len(values)})
    curve_fields = ["run_id", "dataset", "channel", "selection", "k", "factor_index", "factor_name", "accuracy_mean", "accuracy_std", "repeats"]
    temp = output / "factor_probe_curves.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=curve_fields); writer.writeheader(); writer.writerows(curves)
    temp.replace(output / "factor_probe_curves.csv")
    indexed = {(row["run_id"], row["selection"], row["k"], row["factor_index"]): row for row in curves}; summaries = []
    identities = sorted({(row["run_id"], row["dataset"], row["channel"], row["factor_index"], row["factor_name"]) for row in curves})
    for run_id, dataset, channel, factor, factor_name in identities:
        top = sorted((row for row in curves if row["run_id"] == run_id and row["factor_index"] == factor and row["selection"] == "eigen_top"), key=lambda row: row["k"])
        if not top: continue
        maximum = float(top[-1]["accuracy_mean"]); threshold = 0.95 * maximum
        min_k = next((int(row["k"]) for row in top if float(row["accuracy_mean"]) >= threshold), int(top[-1]["k"]))
        max_k = float(top[-1]["k"]); auc = 0.0
        for left, right in zip(top, top[1:]):
            auc += (float(right["k"]) - float(left["k"])) * (float(left["accuracy_mean"]) + float(right["accuracy_mean"])) / 2
        auc /= max(1.0, max_k - float(top[0]["k"]))
        differences = {}
        for control in ("random", "eigen_bottom", "pca_top", "unranked_first", "random_rotation_first"):
            paired = [float(row["accuracy_mean"]) - float(indexed[(run_id, control, row["k"], factor)]["accuracy_mean"]) for row in top if (run_id, control, row["k"], factor) in indexed]
            differences[f"mean_top_minus_{control}"] = statistics.fmean(paired) if paired else ""
        summaries.append({"run_id": run_id, "dataset": dataset, "channel": channel, "factor_index": factor, "factor_name": factor_name, "top_auc": auc, "full_top_accuracy": maximum, "min_k_95pct_top": min_k, **differences})
    summary_fields = list(summaries[0]) if summaries else ["run_id"]
    temp = output / "factor_probe_summary.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summaries)
    temp.replace(output / "factor_probe_summary.csv")
    snapshot = {"curve_rows": len(curves), "summary_rows": len(summaries), "runs": sorted({row["run_id"] for row in curves})}
    temp = output / "factor_probe_snapshot.json.tmp"; temp.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temp.replace(output / "factor_probe_snapshot.json")
    print(json.dumps(snapshot, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
