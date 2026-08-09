#!/usr/bin/env python3
"""Persistently connect all post-fix training, evaluation, and render chains."""

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
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}
STATE_PATH = Path(f"results/orchestration/downstream_{VERSION}.json")
SSL_STATE = Path("results/orchestration/formal_ssl_postfix_state.json")
IMAGE_STATE = Path(f"results/orchestration/imagenet_formal_{VERSION}.json")
FACTOR_STATE = Path(f"results/orchestration/factor_suite_{VERSION}.json")
E10_STATE = Path(f"results/orchestration/e10_{VERSION}.json")
E8_STATE = Path(f"results/orchestration/e8_{VERSION}.json")
MATCHED_STATE = Path(f"results/orchestration/matched_compute_{VERSION}.json")
LOW_LABEL_STATE = Path(f"results/orchestration/formal_low_label_{VERSION}.json")
E3_IMAGENET100_STATE = Path(f"results/orchestration/e3_imagenet100_recheck_{VERSION}.json")
TRANSFER_STATE = Path(f"results/orchestration/formal_transfer_{VERSION}.json")
LOCALIZATION_STATE = Path(f"results/orchestration/formal_localization_{VERSION}.json")
IMAGENET_LOW_LABEL_STATE = Path(f"results/orchestration/formal_imagenet_low_label_{VERSION}.json")
EXTERNAL_WATCHERS = {
    "tsd_cifar10": "20260809-043414_postfix-cifar10-tsd-full-severity-sweep",
    "tsd_cifar100": "20260809-043436_postfix-cifar100-tsd-full-severity-after-cifar10-retry",
    "factor": "20260809-030719_postfix-e7-factor-suite",
    "e10": "20260809-050058_postfix-e10-benchmark-chain-fp32moments",
    "imagenet": "20260809-045027_postfix-imagenet-formal-state-machine",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(payload: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def load(path: Path = STATE_PATH) -> dict:
    if path.is_file():
        state = read(path)
        if state.get("scientific_correctness_version") != VERSION:
            raise RuntimeError(f"refusing legacy downstream state: {path}")
        # Operational watcher retries preserve scientific artifacts but receive
        # new human-readable run IDs.  Keep persisted dependencies current.
        state["external_watchers"] = dict(EXTERNAL_WATCHERS)
        return state
    return {
        "scientific_correctness_version": VERSION,
        "state": "RUNNING", "chain_runs": [], "post_ssl_run": "",
        "transfer_run": "", "localization_run": "", "imagenet_low_label_run": "",
        "render_runs": {}, "external_watchers": EXTERNAL_WATCHERS,
    }


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def wait_run(run_id: str, label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        value = run_state(run_id)
        if value == "SUCCEEDED":
            return
        if value in TERMINAL:
            raise RuntimeError(f"{label} {run_id} ended in {value}")


def wait_state(path: Path, label: str) -> dict:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        if not path.is_file():
            continue
        payload = read(path)
        if payload.get("scientific_correctness_version") != VERSION:
            raise RuntimeError(f"refusing legacy {label} state: {path}")
        value = str(payload.get("state", "RUNNING"))
        if value == "SUCCEEDED":
            return payload
        if value in TERMINAL:
            raise RuntimeError(f"{label} state ended in {value}: {path}")


def ensure_watch(state: dict, field: str, name: str, command: list[str]) -> str:
    run_id = str(state.get(field, ""))
    if run_id:
        value = run_state(run_id)
        if value not in TERMINAL or value == "SUCCEEDED":
            return run_id
        raise RuntimeError(f"post-fix watcher {field} ended in {value}: {run_id}")
    run_id = submit([
        "python3", "-m", "harness.cli", "watch", "--name", name, "--", *command,
    ])
    state[field] = run_id
    save(state)
    return run_id


def render_commands() -> dict[str, list[str]]:
    return {
        "e3": [PYTHON, "-m", "scripts.render_e3_cifar_assets"],
        "e4_e5": [PYTHON, "-m", "scripts.render_e4_e5_assets"],
        "matched_compute": [
            PYTHON, "-m", "scripts.render_matched_compute_assets",
            "--state-file", str(MATCHED_STATE),
        ],
        "e6_generalization": [PYTHON, "-m", "scripts.render_e6_generalization_assets"],
        "e6_robustness": [PYTHON, "-m", "scripts.render_e6_robustness_assets"],
        "e7_tsd": [PYTHON, "-m", "scripts.render_e7_tsd_assets"],
        "e7_factors": [PYTHON, "-m", "scripts.summarize_factor_probes"],
        "e9": [PYTHON, "-m", "scripts.render_e9_localization_assets"],
        "completion_audit": [PYTHON, "-m", "scripts.build_experiment_completion_matrix"],
    }


def render_all(state: dict) -> None:
    render_runs = dict(state.get("render_runs", {}))
    for key, command in render_commands().items():
        run_id = str(render_runs.get(key, ""))
        if run_id and run_state(run_id) == "SUCCEEDED":
            continue
        if run_id and run_state(run_id) not in TERMINAL:
            wait_run(run_id, f"renderer {key}")
            continue
        run_id = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"postfix-render-{key}",
            "--gpus", "0", "--", *command,
        ])
        render_runs[key] = run_id
        state["render_runs"] = render_runs
        save(state)
        wait_run(run_id, f"renderer {key}")


def main() -> int:
    state = load()
    chain_runs = list(state.get("chain_runs", []))
    current_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_run not in chain_runs:
        chain_runs.append(current_run)
    state["chain_runs"] = chain_runs
    save(state)

    wait_state(SSL_STATE, "formal SSL")
    post_ssl = ensure_watch(
        state, "post_ssl_run", "postfix-post-formal-ssl",
        [PYTHON, "-m", "scripts.launch_post_formal_ssl"],
    )

    wait_state(IMAGE_STATE, "formal ImageNet")
    transfer = ensure_watch(
        state, "transfer_run", "postfix-formal-transfer",
        [PYTHON, "-m", "scripts.formal_transfer_state_machine",
         "--state-file", str(TRANSFER_STATE), "--pretrain-state", str(IMAGE_STATE)],
    )
    localization = ensure_watch(
        state, "localization_run", "postfix-formal-localization",
        [PYTHON, "-m", "scripts.formal_localization_state_machine",
         "--state-file", str(LOCALIZATION_STATE), "--pretrain-state", str(IMAGE_STATE)],
    )
    imagenet_low_label = ensure_watch(
        state, "imagenet_low_label_run", "postfix-formal-imagenet-low-label",
        [PYTHON, "-m", "scripts.formal_imagenet_low_label_state_machine",
         "--state-file", str(IMAGENET_LOW_LABEL_STATE), "--pretrain-state", str(IMAGE_STATE)],
    )

    wait_run(post_ssl, "post-formal SSL")
    wait_run(transfer, "formal transfer")
    wait_run(localization, "formal localization")
    wait_run(imagenet_low_label, "formal ImageNet low-label")
    wait_state(MATCHED_STATE, "matched compute")
    wait_state(LOW_LABEL_STATE, "formal low-label")
    wait_state(E3_IMAGENET100_STATE, "E3 ImageNet-100")
    wait_state(TRANSFER_STATE, "formal transfer")
    wait_state(LOCALIZATION_STATE, "formal localization")
    wait_state(IMAGENET_LOW_LABEL_STATE, "formal ImageNet low-label")
    wait_state(FACTOR_STATE, "factor suite")
    wait_state(E10_STATE, "E10 suite")
    wait_state(E8_STATE, "E8 Markov suite")
    for key in ("tsd_cifar10", "tsd_cifar100", "factor", "e10", "imagenet"):
        wait_run(str(dict(state["external_watchers"])[key]), key)

    render_all(state)
    state["state"] = "SUCCEEDED"
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
