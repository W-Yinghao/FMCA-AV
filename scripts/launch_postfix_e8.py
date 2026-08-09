#!/usr/bin/env python3
"""Run and render the post-fix E8 Markov suite through Slurm."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
VERSION = SCIENTIFIC_CORRECTNESS_VERSION
STATE_PATH = Path(f"results/orchestration/e8_{VERSION}.json")
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def load(path: Path = STATE_PATH) -> dict:
    if path.is_file():
        state = read(path)
        if state.get("scientific_correctness_version") != VERSION:
            raise RuntimeError(f"refusing legacy E8 state: {path}")
        return state
    return {
        "scientific_correctness_version": VERSION,
        "state": "RUNNING",
        "chain_runs": [],
        "sweep_run": "",
        "render_run": "",
    }


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def wait_run(run_id: str, label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        state = run_state(run_id)
        if state == "SUCCEEDED":
            return
        if state in TERMINAL:
            raise RuntimeError(f"{label} {run_id} ended in {state}")


def main() -> int:
    state = load()
    current = os.environ["FMCA_HARNESS_RUN_ID"]
    if current not in state["chain_runs"]:
        state["chain_runs"].append(current)
        save(state)

    sweep_run = str(state.get("sweep_run", ""))
    if not sweep_run:
        sweep_run = submit([
            "python3", "-m", "harness.cli", "submit",
            "--name", "postfix-e8-markov-full-10rep", "--gpus", "0", "--",
            PYTHON, "-m", "scripts.run_e8_markov_full_sweep", "--replicates", "10",
        ])
        state["sweep_run"] = sweep_run
        save(state)
    wait_run(sweep_run, "E8 sweep")

    render_run = str(state.get("render_run", ""))
    if not render_run:
        source = Path("runs") / sweep_run / "artifacts" / "e8_markov_full.json"
        render_run = submit([
            "python3", "-m", "harness.cli", "submit",
            "--name", "postfix-e8-render", "--gpus", "0", "--",
            PYTHON, "-m", "scripts.render_e8_markov_assets", "--input", str(source),
        ])
        state["render_run"] = render_run
        save(state)
    wait_run(render_run, "E8 renderer")
    state["state"] = "SUCCEEDED"
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
