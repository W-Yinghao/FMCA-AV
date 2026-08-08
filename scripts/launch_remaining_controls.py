#!/usr/bin/env python3
"""Run remaining E9 sanity and E7 image-chain controls sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from scripts.launch_e9_layer_randomization_wave import main as layer_randomization_main
from scripts.launch_random_label_sanity_wave import main as random_label_main
from scripts.launch_e7_image_data_processing_chain import main as image_chain_main


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"


def completed_high_resource_gaussian() -> bool:
    """Reuse a complete standalone extension instead of recomputing it."""
    for output in sorted(Path("runs").glob("*/artifacts/e1_high_resource_gaussian.json")):
        status_path = output.parents[1] / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parameters = dict(payload.get("parameters", {}))
        if (status.get("state") == "SUCCEEDED"
                and int(parameters.get("replicates", 0)) >= 20
                and str(parameters.get("sample_sizes", "")) == "20000"
                and len(payload.get("records", [])) >= 420):
            return True
    return False


def run_state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def latest_named_run(name: str) -> str:
    jobs = dict(json.loads(Path("harness/state/jobs.json").read_text(encoding="utf-8")).get("jobs", {}))
    matches = [value for value in jobs.values() if str(value.get("name", "")) == name]
    if not matches:
        return ""
    return str(max(matches, key=lambda value: str(value.get("created_at", "")))["run_id"])


def submit_with_capacity(command: list[str]) -> str:
    while True:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def wait_success(run_id: str, label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
        value = run_state(run_id)
        if value == "SUCCEEDED":
            return
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"{label} {run_id} ended in {value}")


def start_high_resource_gaussian() -> str:
    """Submit the CPU-heavy extension to Slurm; the local process remains orchestration-only."""
    if completed_high_resource_gaussian():
        return ""
    prior = latest_named_run("e1-high-resource-gaussian-extension")
    if prior and run_state(prior) in {"QUEUED", "RUNNING"}:
        return prior
    if prior and run_state(prior) in {"FAILED", "STOPPED", "BLOCKED"}:
        return submit_with_capacity([
            "python3", "-m", "harness.cli", "retry", "--run", prior,
        ])
    return submit_with_capacity([
        "python3", "-m", "harness.cli", "submit", "--name", "e1-high-resource-gaussian-extension",
        "--gpus", "0", "--", PYTHON, "-m", "scripts.run_e1_gaussian_operator_suite",
        "--replicates", "20", "--sample-sizes", "20000", "--case-indices", "3,4,5,6,7,8,9", "--seed", "20300000",
    ])


def run_estimator_baselines() -> None:
    for output in sorted(Path("runs").glob("*/artifacts/e1_estimator_baselines.json")):
        status_path = output.parents[1] / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (status.get("state") == "SUCCEEDED"
                and int(dict(payload.get("parameters", {})).get("replicates", 0)) >= 10
                and len(payload.get("records", [])) >= 450):
            return
    prior = latest_named_run("e1-estimator-baselines-full")
    if prior and run_state(prior) in {"QUEUED", "RUNNING"}:
        wait_success(prior, "estimator baselines")
        return
    if prior and run_state(prior) in {"FAILED", "STOPPED", "BLOCKED"}:
        retried = submit_with_capacity([
            "python3", "-m", "harness.cli", "retry", "--run", prior,
        ])
        wait_success(retried, "estimator baselines")
        return
    run_id = submit_with_capacity([
        "python3", "-m", "harness.cli", "submit", "--name", "e1-estimator-baselines-full",
        "--gpus", "0", "--", PYTHON, "-m", "scripts.run_e1_estimator_baselines",
        "--replicates", "10", "--seed", "20301000",
    ])
    wait_success(run_id, "estimator baselines")


def composition_smoke() -> None:
    source_run = "20260807-052202_imagenet1k-lightning-32step-smoke"
    result = json.loads((Path("runs") / source_run / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
    checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
    command = [
        "python3", "-m", "harness.cli", "submit", "--name", "e9-cnn-composition-5sample-smoke",
        "--gpus", "1", "--profile", "imagenet", "--", PYTHON, "-m", "scripts.run_cnn_composition_maps",
        "--config", "configs/ssl/imagenet1k_smoke.json", "--checkpoint", str(checkpoint), "--model-type", "fmca",
        "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/cub", "--calibration-samples", "5", "--evaluation-samples", "5",
    ]
    while True:
        submitted = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if submitted.returncode == 0:
            run_id = submitted.stdout.strip(); break
        if "GPU limit exceeded" not in submitted.stderr: raise RuntimeError(submitted.stderr.strip())
        time.sleep(POLL_SECONDS); subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
    while True:
        time.sleep(POLL_SECONDS); subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
        state = json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"]
        if state == "SUCCEEDED": return
        if state in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"composition smoke {run_id} ended in {state}")


def operator_complexity() -> None:
    command = [
        "python3", "-m", "harness.cli", "submit", "--name", "e10-operator-complexity-k32-512",
        "--gpus", "1", "--", PYTHON, "-m", "scripts.run_operator_complexity_benchmark",
    ]
    while True:
        submitted = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if submitted.returncode == 0:
            run_id = submitted.stdout.strip(); break
        if "GPU limit exceeded" not in submitted.stderr: raise RuntimeError(submitted.stderr.strip())
        time.sleep(POLL_SECONDS); subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
    while True:
        time.sleep(POLL_SECONDS); subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
        state = json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"]
        if state == "SUCCEEDED": return
        if state in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"operator complexity {run_id} ended in {state}")


def flops_profile() -> None:
    command = [
        "python3", "-m", "harness.cli", "submit", "--name", "e10-supported-op-flops-profile",
        "--gpus", "1", "--", PYTHON, "-m", "scripts.run_flops_profile",
    ]
    while True:
        submitted = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if submitted.returncode == 0:
            run_id = submitted.stdout.strip(); break
        if "GPU limit exceeded" not in submitted.stderr: raise RuntimeError(submitted.stderr.strip())
        time.sleep(POLL_SECONDS); subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
    while True:
        time.sleep(POLL_SECONDS); subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
        state = json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"]
        if state == "SUCCEEDED": return
        if state in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"FLOPs profile {run_id} ended in {state}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--non-imagenet-tail",
        action="store_true",
        help="Run only the remaining CIFAR/Gaussian controls; defer ImageNet-based E9 controls.",
    )
    args = parser.parse_args()
    high_resource = start_high_resource_gaussian()
    if args.non_imagenet_tail:
        result = int(image_chain_main())
        if result:
            return result
        if high_resource:
            wait_success(high_resource, "high-resource Gaussian recovery extension")
        run_estimator_baselines()
        return 0
    composition_smoke()
    operator_complexity()
    flops_profile()
    for operation in (layer_randomization_main, random_label_main, image_chain_main):
        result = int(operation())
        if result:
            return result
    if high_resource:
        wait_success(high_resource, "high-resource Gaussian recovery extension")
    run_estimator_baselines()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
