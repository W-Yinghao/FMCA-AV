#!/usr/bin/env python3
"""Run the versioned E10 benchmark chain through the Slurm harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from fmca_av.operators import MOMENT_ACCUMULATION_POLICY, SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh() -> None:
    subprocess.run(
        ["python3", "-m", "harness.cli", "status"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_status(run_id: str) -> dict:
    return json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))


def artifact(run_id: str, name: str) -> Path:
    return Path("runs") / run_id / "artifacts" / name


def valid_artifact(run_id: str, name: str, required: dict[str, object] | None = None) -> bool:
    path = artifact(run_id, name)
    if run_status(run_id).get("state") != "SUCCEEDED" or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("scientific_correctness_version") == SCIENTIFIC_CORRECTNESS_VERSION
        and all(payload.get(key) == value for key, value in (required or {}).items())
    )


def submit_with_capacity(argv: list[str]) -> str:
    while True:
        submitted = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if submitted.returncode == 0:
            return submitted.stdout.strip()
        if "GPU limit exceeded" not in submitted.stderr:
            raise RuntimeError(submitted.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def wait_success(run_id: str) -> None:
    while True:
        state = str(run_status(run_id).get("state"))
        if state == "SUCCEEDED":
            return
        if state in TERMINAL:
            raise RuntimeError(f"E10 child {run_id} ended in {state}")
        time.sleep(POLL_SECONDS)
        refresh()


def load_state(path: Path) -> dict:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        recorded = state.get("scientific_correctness_version")
        if recorded != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(
                f"refusing mismatched E10 state {path}: {recorded!r} != "
                f"{SCIENTIFIC_CORRECTNESS_VERSION!r}"
            )
        return state
    return {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING",
        "runs": {},
    }


def ensure_gpu_stage(
    state: dict,
    state_path: Path,
    key: str,
    name: str,
    module: str,
    artifact_name: str,
    required: dict[str, object] | None = None,
) -> str:
    runs = state.setdefault("runs", {})
    run_id = str(runs.get(key, ""))
    if run_id and valid_artifact(run_id, artifact_name, required):
        return run_id
    if run_id:
        child_state = str(run_status(run_id).get("state"))
        if child_state not in TERMINAL:
            wait_success(run_id)
            if valid_artifact(run_id, artifact_name, required):
                return run_id
        superseded = state.setdefault("superseded_runs", {}).setdefault(key, [])
        if run_id not in superseded:
            superseded.append(run_id)
    run_id = submit_with_capacity([
        "python3", "-m", "harness.cli", "submit", "--name", name,
        "--gpus", "1", "--", PYTHON, "-m", module,
    ])
    runs[key] = run_id
    atomic_json(state_path, state)
    wait_success(run_id)
    if not valid_artifact(run_id, artifact_name, required):
        raise RuntimeError(f"E10 child {run_id} did not produce valid {artifact_name}")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file",
        default=f"results/orchestration/e10_{SCIENTIFIC_CORRECTNESS_VERSION}.json",
    )
    args = parser.parse_args()
    state_path = Path(args.state_file)
    state = load_state(state_path)

    complexity = ensure_gpu_stage(
        state, state_path, "complexity", "postfix-e10-complexity",
        "scripts.run_complexity_benchmark", "complexity.json",
        {"moment_accumulation_policy": MOMENT_ACCUMULATION_POLICY},
    )
    operator = ensure_gpu_stage(
        state, state_path, "operator", "postfix-e10-operator-complexity",
        "scripts.run_operator_complexity_benchmark", "operator_complexity.json",
    )
    flops = ensure_gpu_stage(
        state, state_path, "flops", "postfix-e10-flops-profile",
        "scripts.run_flops_profile", "flops_profile.json",
    )

    render_id = str(state["runs"].get("render", ""))
    if render_id and run_status(render_id).get("state") not in TERMINAL:
        wait_success(render_id)
    if not render_id or run_status(render_id).get("state") != "SUCCEEDED":
        output_dir = f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}/e10"
        render_id = submit_with_capacity([
            "python3", "-m", "harness.cli", "submit", "--name", "postfix-e10-render",
            "--gpus", "0", "--", PYTHON, "-m", "scripts.render_complexity_assets",
            "--input", str(artifact(complexity, "complexity.json")),
            "--operator", str(artifact(operator, "operator_complexity.json")),
            "--flops", str(artifact(flops, "flops_profile.json")),
            "--output-dir", output_dir,
        ])
        state["runs"]["render"] = render_id
        atomic_json(state_path, state)
    wait_success(render_id)
    output_dir = Path(f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}/e10")
    if not (output_dir / "complexity_table.csv").is_file():
        raise RuntimeError(f"E10 renderer {render_id} did not produce complexity_table.csv")
    state["state"] = "SUCCEEDED"
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
