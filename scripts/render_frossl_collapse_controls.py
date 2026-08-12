#!/usr/bin/env python3
"""Render the immutable result summary for the FroSSL collapse controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    state = read(Path(args.state_file))
    records = state["records"]
    lines = [
        "# FroSSL CIFAR-10 collapse mechanism audit",
        "",
        "The recovery gate was fixed before the controls: clean kNN >= 60%, centered backbone effective rank >= 20, and centered top-eigenvalue share <= 0.8 at epoch 200.",
        "",
        "## Existing M=8 checkpoint audit",
        "",
        "| Seed | Condition | kNN | Centered effective rank | Centered top share | Mean energy ratio |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index in range(1, 4):
        record = records[f"existing_audit:s{index}"]
        payload = read(Path("runs") / record["run_id"] / "artifacts" / "frossl_collapse_audit.json")
        for condition, result in payload["conditions"].items():
            backbone = result["backbone"]
            centered = backbone["centered_covariance"]
            lines.append(
                f"| {payload['requested_seed']} | {condition} | {100 * result['knn_accuracy']:.2f}% | "
                f"{centered['effective_rank']:.3f} | {centered['top_eigenvalue_share']:.4f} | "
                f"{backbone['mean_energy_ratio']:.4f} |"
            )
    lines.extend([
        "", "## Short M=8 controls", "",
        "| Cell | Epoch | kNN | Centered effective rank | Centered top share | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for cell in ("B", "C", "D"):
        for epoch in (25, 50, 100, 200):
            record = records[f"control_audit:{cell}:e{epoch}"]
            if record["state"] == "SKIPPED":
                lines.append(f"| {cell} | {epoch} | — | — | — | SKIPPED |")
                continue
            payload = read(Path("runs") / record["run_id"] / "artifacts" / "frossl_collapse_audit.json")
            result = payload["conditions"]["eval_saved"]
            centered = result["backbone"]["centered_covariance"]
            passed = (
                result["knn_accuracy"] >= 0.60 and centered["effective_rank"] >= 20
                and centered["top_eigenvalue_share"] <= 0.8
            )
            lines.append(
                f"| {cell} | {epoch} | {100 * result['knn_accuracy']:.2f}% | "
                f"{centered['effective_rank']:.3f} | {centered['top_eigenvalue_share']:.4f} | "
                f"{'PASS' if passed and epoch == 200 else 'not applied' if epoch != 200 else 'FAIL'} |"
            )
    lines.extend([
        "", "## Official-code-style M=2", "",
        "This path uses two crops, no random resized crop, sequential per-view forward, full 50k pretraining images, gamma=1.0, weight decay 1e-4, FP32, and the detached online classifier. It remains a harness port rather than execution inside solo-learn.",
        "",
        "| Seed | Linear probe | kNN | Backbone effective rank |",
        "|---:|---:|---:|---:|",
    ])
    for index in range(1, 4):
        seed = records[f"official_train:s{index}"]["seed"]
        probe = read(Path("runs") / records[f"official_probe:s{index}"]["run_id"] / "artifacts" / "probe_result.json")
        knn = read(Path("runs") / records[f"official_knn:s{index}"]["run_id"] / "artifacts" / "knn_result.json")
        diagnostics = read(Path("runs") / records[f"official_diagnostics:s{index}"]["run_id"] / "artifacts" / "diagnostics.json")
        lines.append(
            f"| {seed} | {100 * probe['test_accuracy']:.2f}% | {100 * knn['knn_accuracy']:.2f}% | "
            f"{diagnostics['backbone']['effective_rank']:.3f} |"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "Existing M=8 results are a matched-protocol stress test, not a missing official CIFAR-10 M=8 cell. BN-state recovery, augmentation recovery, and encoder-weight collapse are reported separately. Objective-level centering and pair normalization were not mixed into these first controls.",
    ])
    report = Path("SERVER_FROSSL_COLLAPSE_AUDIT.md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir:
        artifact = Path(run_dir) / "artifacts" / report.name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
