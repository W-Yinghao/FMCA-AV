#!/usr/bin/env python3
"""Chain formal transfer/localization and final result collection via Slurm."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
STATE_PATH = Path("results/orchestration/formal_downstream_state.json")
IMAGE_STATE = Path("results/orchestration/imagenet_formal_state.json")
SSL_STATE = Path("results/orchestration/formal_ssl_state.json")
TRANSFER_STATE = Path("results/orchestration/formal_transfer_state.json")
LOCALIZATION_STATE = Path("results/orchestration/formal_localization_state.json")
FACTOR_STATE = Path("results/orchestration/full_factor_probes_state.json")
MATCHED_COMPUTE_STATE = Path("results/orchestration/matched_compute_state.json")
LOW_LABEL_STATE = Path("results/orchestration/formal_low_label_state.json")
IMAGENET_LOW_LABEL_STATE = Path("results/orchestration/formal_imagenet_low_label_state.json")
RECOVERY_RUN = "20260807-073411_recover-interrupted-e3-tsd-coco-v2"
FACTOR_DEPENDENCIES = (
    "20260807-063008_launch-e7-factor-channel-wave",
    "20260807-070515_launch-e7-factor-stability-wave",
)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(payload: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def load() -> dict[str, object]:
    if STATE_PATH.is_file():
        return read_json(STATE_PATH)
    return {
        "state": "RUNNING",
        "chain_runs": [],
        "transfer_run": "",
        "localization_run": "",
        "factor_run": "",
        "matched_compute_run": "",
        "post_ssl_run": "",
        "remaining_controls_run": "",
        "imagenet_low_label_run": "",
        "finalizer_run": "",
    }


def wait_state(path: Path, label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        if not path.is_file():
            continue
        value = str(read_json(path).get("state", "RUNNING"))
        if value == "SUCCEEDED":
            return
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"{label} state ended in {value}: {path}")


def state_file_value(path: Path) -> str:
    return str(read_json(path).get("state", "RUNNING")) if path.is_file() else "MISSING"


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def run_state(run_id: str) -> str:
    return str(read_json(Path("runs") / run_id / "status.json")["state"])


def operator_paused_watcher(run_id: str, value: str) -> bool:
    if value != "STOPPED":
        return False
    record = read_json(Path("runs") / run_id / "status.json")
    return bool(record.get("local_watcher")) and record.get("failure_reason") == "stopped by operator"


def latest_retry(run_id: str) -> str:
    jobs = dict(read_json(Path("harness/state/jobs.json")).get("jobs", {}))
    current = run_id
    visited = {current}
    while True:
        children = [
            value for value in jobs.values()
            if str(value.get("retry_from", "")) == current and str(value.get("run_id", "")) not in visited
        ]
        if not children:
            return current
        chosen = max(children, key=lambda value: str(value.get("created_at", "")))
        current = str(chosen["run_id"])
        visited.add(current)


def wait_run(run_id: str, label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        run_id = latest_retry(run_id)
        value = run_state(run_id)
        if value == "SUCCEEDED":
            return
        if operator_paused_watcher(run_id, value):
            continue
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"{label} run {run_id} ended in {value}")


def main() -> int:
    state = load()
    for field in ("transfer_run", "localization_run", "factor_run", "matched_compute_run", "post_ssl_run", "remaining_controls_run", "imagenet_low_label_run", "finalizer_run"):
        state.setdefault(field, "")
    chain_runs = list(state.get("chain_runs", []))
    current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain_runs: chain_runs.append(current_chain_run)
    state["chain_runs"] = chain_runs
    state["state"] = "RUNNING"
    save(state)

    while not all(state.get(field) for field in ("transfer_run", "localization_run", "factor_run", "post_ssl_run", "remaining_controls_run", "imagenet_low_label_run")):
        time.sleep(POLL_SECONDS)
        refresh()
        resolved_factor_dependencies = [latest_retry(run_id) for run_id in FACTOR_DEPENDENCIES]
        factor_states = {run_id: run_state(run_id) for run_id in resolved_factor_dependencies}
        factor_failures = {
            key: value for key, value in factor_states.items()
            if value in {"FAILED", "STOPPED", "BLOCKED"}
            and not operator_paused_watcher(key, value)
        }
        if factor_failures:
            raise RuntimeError("factor prerequisites failed: " + json.dumps(factor_failures, sort_keys=True))
        if not state.get("factor_run") and all(value == "SUCCEEDED" for value in factor_states.values()):
            state["factor_run"] = submit([
                "python3", "-m", "harness.cli", "watch", "--name", "full-factor-probes",
                "--", PYTHON, "-m", "scripts.launch_full_factor_probes",
            ])
            save(state)

        image_value = state_file_value(IMAGE_STATE)
        if image_value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"formal ImageNet state ended in {image_value}")
        if image_value == "SUCCEEDED":
            if not state.get("transfer_run"):
                state["transfer_run"] = submit([
                    "python3", "-m", "harness.cli", "watch", "--name", "formal-transfer-state-machine",
                    "--", PYTHON, "-m", "scripts.formal_transfer_state_machine",
                    "--state-file", str(TRANSFER_STATE), "--pretrain-state", str(IMAGE_STATE),
                ])
                save(state)
            if not state.get("localization_run"):
                state["localization_run"] = submit([
                    "python3", "-m", "harness.cli", "watch", "--name", "formal-localization-state-machine",
                    "--", PYTHON, "-m", "scripts.formal_localization_state_machine",
                    "--state-file", str(LOCALIZATION_STATE), "--pretrain-state", str(IMAGE_STATE),
                ])
                save(state)
            if not state.get("imagenet_low_label_run"):
                if IMAGENET_LOW_LABEL_STATE.is_file():
                    existing = list(read_json(IMAGENET_LOW_LABEL_STATE).get("chain_runs", []))
                    if existing:
                        state["imagenet_low_label_run"] = str(existing[-1])
                if not state.get("imagenet_low_label_run"):
                    state["imagenet_low_label_run"] = submit([
                        "python3", "-m", "harness.cli", "watch", "--name", "formal-imagenet1k-low-label-state-machine",
                        "--", PYTHON, "-m", "scripts.formal_imagenet_low_label_state_machine",
                        "--state-file", str(IMAGENET_LOW_LABEL_STATE), "--pretrain-state", str(IMAGE_STATE),
                    ])
                save(state)

        ssl_value = state_file_value(SSL_STATE)
        if ssl_value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"formal SSL state ended in {ssl_value}")
        if ssl_value == "SUCCEEDED" and not state.get("post_ssl_run"):
            state["post_ssl_run"] = submit([
                "python3", "-m", "harness.cli", "watch", "--name", "post-formal-ssl-tracks",
                "--", PYTHON, "-m", "scripts.launch_post_formal_ssl",
            ])
            save(state)

        recovery_value = run_state(latest_retry(RECOVERY_RUN))
        if recovery_value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"TSD recovery ended in {recovery_value}")
        if recovery_value == "SUCCEEDED" and not state.get("remaining_controls_run"):
            state["remaining_controls_run"] = submit([
                "python3", "-m", "harness.cli", "watch", "--name", "remaining-e7-e9-controls",
                "--", PYTHON, "-m", "scripts.launch_remaining_controls",
            ])
            save(state)

    wait_state(TRANSFER_STATE, "formal transfer")
    wait_state(LOCALIZATION_STATE, "formal localization")
    wait_state(SSL_STATE, "formal SSL")
    wait_state(FACTOR_STATE, "full factor probes")
    wait_state(MATCHED_COMPUTE_STATE, "matched compute")
    wait_state(LOW_LABEL_STATE, "formal low label and CIFAR robustness")
    wait_state(IMAGENET_LOW_LABEL_STATE, "formal ImageNet-1K low label")
    wait_run(str(state["remaining_controls_run"]), "remaining E7/E9 controls")

    if not state.get("finalizer_run"):
        state["finalizer_run"] = submit([
            "python3", "-m", "harness.cli", "submit", "--name", "finalize-all-result-assets",
            "--gpus", "0", "--", PYTHON, "-m", "scripts.finalize_result_assets",
        ])
        save(state)
    wait_run(str(state["finalizer_run"]), "result finalizer")
    subprocess.run(["python3", "-m", "harness.cli", "collect"], check=True)
    state["state"] = "SUCCEEDED"
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
