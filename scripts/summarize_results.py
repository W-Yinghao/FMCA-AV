#!/usr/bin/env python3
"""Collect scalar experiment outcomes without inventing missing metrics."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="runs/experiment_summary.csv")
    args = parser.parse_args()
    runs = Path(args.runs).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in sorted(path for path in runs.iterdir() if path.is_dir()):
        status = read_json(directory / "status.json")
        request = read_json(directory / "request.json")
        if not status or not request:
            continue
        artifacts = directory / "artifacts"
        train = read_json(artifacts / "train_result.json") or {}
        calibration = read_json(artifacts / "calibration.json") or {}
        evaluation = read_json(artifacts / "evaluation.json") or {}
        probe = read_json(artifacts / "probe_result.json") or {}
        knn = read_json(artifacts / "knn.json") or {}
        corruption = (
            read_json(artifacts / "cifar10c.json")
            or read_json(artifacts / "cifar100c.json")
            or {}
        )
        imagenet_robustness = read_json(artifacts / "imagenet_robustness.json") or {}
        voc = read_json(artifacts / "voc2007_multilabel.json") or {}
        localization = read_json(artifacts / "localization.json") or {}
        factor_probe = read_json(artifacts / "factor_probe.json") or {}
        markov = (
            read_json(artifacts / "markov_exact.json")
            or read_json(artifacts / "markov_continuous.json")
            or {}
        )
        if not any((train, calibration, evaluation, probe, knn, corruption, imagenet_robustness, voc, localization, factor_probe, markov)):
            continue
        rows.append({
            "run_id": status.get("run_id", ""),
            "name": status.get("name", ""),
            "state": status.get("state", ""),
            "slurm_job_id": status.get("slurm_job_id", ""),
            "requested_gpus": status.get("requested_gpus", ""),
            "retry_from": status.get("retry_from", ""),
            "best_validation_score": train.get("best_validation_score", ""),
            "calibration_trace_score": calibration.get("trace_score", ""),
            "eigenvalue_mae": evaluation.get("eigenvalue_mae", ""),
            "eigenvalue_max_error": evaluation.get("eigenvalue_max_error", ""),
            "linear_probe_test_accuracy": probe.get("test_accuracy", ""),
            "linear_probe_test_top5_accuracy": probe.get("test_top5_accuracy", ""),
            "linear_probe_label_fraction": probe.get("label_fraction", ""),
            "knn_accuracy": knn.get("knn_accuracy", ""),
            "mean_corruption_accuracy": corruption.get("mean_corruption_accuracy", ""),
            "imagenet_c_mean_accuracy": (
                imagenet_robustness.get("imagenet_c", {}).get("mean_corruption_accuracy", "")
                if isinstance(imagenet_robustness.get("imagenet_c", {}), dict) else ""
            ),
            "imagenet_r_top1": (
                imagenet_robustness.get("imagenet_r", {}).get("top1_accuracy", "")
                if isinstance(imagenet_robustness.get("imagenet_r", {}), dict) else ""
            ),
            "imagenet_a_top1": (
                imagenet_robustness.get("imagenet_a", {}).get("top1_accuracy", "")
                if isinstance(imagenet_robustness.get("imagenet_a", {}), dict) else ""
            ),
            "voc2007_multilabel_map": voc.get("test_map", ""),
            "localization_dataset": localization.get("dataset", ""),
            "localization_spectral_pointing": (
                localization.get("summary", {}).get("spectral", {}).get("pointing_game", "")
                if isinstance(localization.get("summary", {}), dict) else ""
            ),
            "localization_spectral_box_iou": (
                localization.get("summary", {}).get("spectral", {}).get("box_iou_top20", "")
                if isinstance(localization.get("summary", {}), dict) else ""
            ),
            "factor_probe_dataset": factor_probe.get("dataset", ""),
            "markov_record_count": len(markov.get("records", [])) if isinstance(markov.get("records", []), list) else "",
        })
    fields = list(rows[0]) if rows else ["run_id"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
