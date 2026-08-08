#!/usr/bin/env python3
"""Rebuild every standardized result asset from completed harness artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


PYTHON = sys.executable
POLL_SECONDS = 300
REQUIRED_STATES = {
    "formal SSL": Path("results/orchestration/formal_ssl_state.json"),
    "formal ImageNet": Path("results/orchestration/imagenet_formal_state.json"),
    "formal transfer": Path("results/orchestration/formal_transfer_state.json"),
    "formal localization": Path("results/orchestration/formal_localization_state.json"),
    "full factor probes": Path("results/orchestration/full_factor_probes_state.json"),
    "matched compute": Path("results/orchestration/matched_compute_state.json"),
    "formal low label": Path("results/orchestration/formal_low_label_state.json"),
    "formal ImageNet-1K low label": Path("results/orchestration/formal_imagenet_low_label_state.json"),
    "ImageNet-100 E3 recheck": Path("results/orchestration/e3_imagenet100_recheck_state.json"),
}
REQUIRED_RUNS = {
    "E8 Markov observation/noise extension": "20260807-114546_e8-markov-observation-noise-10rep-fixed-output",
    "E7 high-resource calibration axis": "20260807-120707_e7-tsd-calibration-high-resource-10rep",
    "E3 Gaussian/finite estimator controls": "20260807-121735_e3-gaussian-finite-estimator-controls-20rep-v2",
    "ConvNeXt/ViT localization launcher": "20260807-122813_launch-architecture-localization-wave",
    "E1 finite learned reference retry": "20260807-122853_e1-finite-reference-seed-20260915",
}
REQUIRED_CHILD_MANIFESTS = {
    "ConvNeXt/ViT localization children": Path("runs/20260807-122813_launch-architecture-localization-wave/artifacts/submitted.json"),
    "E4 aggregation/head/source children": Path("runs/20260807-062346_launch-e4-aggregation-ablation-wave/artifacts/submitted.json"),
}
REQUIRED_RUN_POINTERS = {
    "cross-scale TSD utility probes": "*_continue-crossscale-tsd-after-recovery/artifacts/utility_watcher.txt",
}


def successful_artifact(path: Path) -> bool:
    status_path = path.parents[1] / "status.json"
    if not status_path.is_file():
        return False
    try:
        return str(json.loads(status_path.read_text(encoding="utf-8")).get("state", "")) == "SUCCEEDED"
    except (OSError, json.JSONDecodeError):
        return False


def newest(pattern: str, preferred: str = "") -> Path:
    candidates = sorted((path for path in Path("runs").glob(pattern) if successful_artifact(path)),
                        key=lambda path: path.stat().st_mtime)
    if preferred:
        preferred_values = [path for path in candidates if preferred in str(path)]
        if preferred_values: candidates = preferred_values
    if not candidates: raise FileNotFoundError(f"required result input not found: {pattern}")
    return candidates[-1]


def run(*arguments: str) -> None:
    subprocess.run([PYTHON, *arguments], check=True)


def wait_required_states() -> None:
    """Prevent a result freeze while any formal state machine is unfinished."""
    while True:
        values = {}
        for label, path in REQUIRED_STATES.items():
            if not path.is_file():
                values[label] = "MISSING"
                continue
            values[label] = str(json.loads(path.read_text(encoding="utf-8")).get("state", "RUNNING"))
        failures = {label: value for label, value in values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures:
            raise RuntimeError("formal prerequisite failed: " + json.dumps(failures, sort_keys=True))
        run_values = {}
        for label, run_id in REQUIRED_RUNS.items():
            status_path = Path("runs") / run_id / "status.json"
            run_values[label] = (
                str(json.loads(status_path.read_text(encoding="utf-8")).get("state", "RUNNING"))
                if status_path.is_file() else "MISSING"
            )
        run_failures = {
            label: value for label, value in run_values.items()
            if value in {"FAILED", "STOPPED", "BLOCKED"}
        }
        if run_failures:
            raise RuntimeError("required extension failed: " + json.dumps(run_failures, sort_keys=True))
        child_values = {}
        for label, manifest in REQUIRED_CHILD_MANIFESTS.items():
            if not manifest.is_file():
                child_values[label] = "MISSING"
                continue
            records = json.loads(manifest.read_text(encoding="utf-8"))
            states = []
            for record in records:
                status_path = Path("runs") / str(record["run_id"]) / "status.json"
                states.append(str(json.loads(status_path.read_text(encoding="utf-8")).get("state", "RUNNING")) if status_path.is_file() else "MISSING")
            if any(value in {"FAILED", "STOPPED", "BLOCKED"} for value in states):
                child_values[label] = "FAILED"
            elif states and all(value == "SUCCEEDED" for value in states):
                child_values[label] = "SUCCEEDED"
            else:
                child_values[label] = "RUNNING"
        child_failures = {label: value for label, value in child_values.items() if value == "FAILED"}
        if child_failures:
            raise RuntimeError("required child manifest failed: " + json.dumps(child_failures, sort_keys=True))
        pointer_values = {}
        for label, pattern in REQUIRED_RUN_POINTERS.items():
            candidates = sorted(
                (path for path in Path("runs").glob(pattern) if successful_artifact(path)),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates:
                pointer_values[label] = "MISSING"; continue
            pointer = candidates[-1]
            if not pointer.read_text(encoding="utf-8").strip():
                pointer_values[label] = "MISSING"; continue
            run_id = pointer.read_text(encoding="utf-8").strip().splitlines()[0]
            status_path = Path("runs") / run_id / "status.json"
            pointer_values[label] = (str(json.loads(status_path.read_text(encoding="utf-8")).get("state", "RUNNING"))
                                     if status_path.is_file() else "MISSING")
        pointer_failures = {label: value for label, value in pointer_values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if pointer_failures:
            raise RuntimeError("required pointer run failed: " + json.dumps(pointer_failures, sort_keys=True))
        if (values and all(value == "SUCCEEDED" for value in values.values())
                and run_values and all(value == "SUCCEEDED" for value in run_values.values())
                and child_values and all(value == "SUCCEEDED" for value in child_values.values())
                and pointer_values and all(value == "SUCCEEDED" for value in pointer_values.values())):
            return
        time.sleep(POLL_SECONDS)
        subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    if not os.environ.get("FMCA_HARNESS_RUN_DIR"):
        raise RuntimeError("final result rendering must run through the Slurm harness")
    wait_required_states()
    gaussian = newest("*/artifacts/e1_gaussian_operator.json")
    nonlinear = newest("*/artifacts/e1_nonlinear_toy.json")
    discrete = newest("*/artifacts/exact_channels.json")
    finite_sample = newest("*/artifacts/finite_sample_recovery.json")
    e2_candidates = [path for path in Path("runs").glob("*/artifacts/cifar_gradient_variance.json")
                     if successful_artifact(path)]
    e2 = max(e2_candidates, key=lambda path: path.stat().st_mtime) if e2_candidates else newest("*/artifacts/e2_gradient_variance.json")
    e8 = newest("*/artifacts/e8_markov_full.json")
    complexity = newest("*/artifacts/complexity.json")
    operator_complexity = newest("*/artifacts/operator_complexity.json")
    flops_profile = newest("*/artifacts/flops_profile.json")
    gaussian_extra = sorted(path for path in Path("runs").glob("*/artifacts/e1_high_resource_gaussian.json")
                            if successful_artifact(path))
    e1_arguments = ["scripts/render_e1_recovery_assets.py", "--gaussian", str(gaussian)]
    for extra in gaussian_extra:
        e1_arguments += ["--gaussian-extra", str(extra)]
    e1_arguments += ["--nonlinear", str(nonlinear), "--discrete", str(discrete), "--finite-sample", str(finite_sample)]
    run(*e1_arguments)
    run("scripts/render_e1_estimator_assets.py")
    run("scripts/render_e2_variance_assets.py", "--input", str(e2))
    for script in (
        "scripts/render_e3_cifar_assets.py", "scripts/render_e4_e5_assets.py",
        "scripts/render_e6_robustness_assets.py", "scripts/render_e6_generalization_assets.py",
        "scripts/render_e7_tsd_assets.py", "scripts/render_e9_localization_assets.py",
    ): run(script)
    run("scripts/summarize_factor_probes.py")
    run("scripts/render_matched_compute_assets.py")
    run("scripts/render_e8_markov_assets.py", "--input", str(e8))
    run("scripts/render_complexity_assets.py", "--input", str(complexity), "--operator", str(operator_complexity), "--flops", str(flops_profile))
    run("scripts/build_confirmatory_statistics.py")
    run("scripts/build_claim_cards.py")
    run("scripts/build_experiment_completion_matrix.py")
    run("scripts/build_result_index.py", "--runs", "runs", "--output-dir", "results/index")
    outputs = sorted(str(path) for path in Path("results").glob("*/*") if path.is_file())
    artifact_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary = artifact_dir / "finalized_assets.json.tmp"
    temporary.write_text(json.dumps({"assets": outputs, "count": len(outputs)}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(artifact_dir / "finalized_assets.json")
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "finalize_result_assets", "assets": len(outputs)}) + "\n")
    print(json.dumps({"assets": len(outputs)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
