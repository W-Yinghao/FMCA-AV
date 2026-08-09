#!/usr/bin/env python3
"""Run formal low-label linear-probe and fine-tuning protocols after SSL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts import formal_ssl_state_machine as formal
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
FORMAL_STATE = Path("results/orchestration/formal_ssl_state.json")
STATE_PATH = Path("results/orchestration/formal_low_label_state.json")
METHOD_VIEWS = {
    "fmca_av": 8, "fmca_av_matched_head": 8, "hfmca_style": 8, "regular_fmca": 1,
    "simclr": 8, "vicreg": 8, "moco_v2": 8, "dino": 8,
}
DATASETS = ("cifar10", "cifar100", "imagenet100")
PROTOCOLS = (("linear-probe", 0.01), ("linear-probe", 0.1), ("linear-probe", 1.0),
             ("fine-tune", 0.01), ("fine-tune", 0.1))


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(payload: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True); temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(STATE_PATH)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def wait_formal() -> dict[str, object]:
    while True:
        time.sleep(POLL_SECONDS); refresh()
        if not FORMAL_STATE.is_file(): continue
        payload = read(FORMAL_STATE)
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy formal SSL state: {FORMAL_STATE}")
        value = str(payload.get("state", "RUNNING"))
        if value == "SUCCEEDED": return payload
        if value in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"formal SSL state ended in {value}")


def wait_all(run_ids: list[str]) -> None:
    while True:
        time.sleep(POLL_SECONDS); refresh(); states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {key: value for key, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures: raise RuntimeError("formal low-label job failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()): return


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def checkpoint(run_id: str) -> str:
    payload = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy formal low-label checkpoint from {run_id}")
    value = payload.get("last_checkpoint") or payload.get("best_checkpoint")
    if not value or not Path(str(value)).is_file(): raise RuntimeError(f"formal low-label checkpoint missing for {run_id}")
    return str(value)


def main(argv: list[str] | None = None) -> int:
    global FORMAL_STATE, STATE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-state", default=str(FORMAL_STATE))
    parser.add_argument("--state-file", default=str(STATE_PATH))
    args = parser.parse_args(argv)
    FORMAL_STATE = Path(args.formal_state)
    STATE_PATH = Path(args.state_file)
    state = read(STATE_PATH) if STATE_PATH.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING", "chain_runs": [], "submitted": [],
    }
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy formal low-label state: {STATE_PATH}")
    chain = list(state.get("chain_runs", [])); current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain: chain.append(current_chain_run)
    state["chain_runs"] = chain; state["state"] = "RUNNING"; save(state)
    formal_state = wait_formal(); completed = list(formal_state["completed"])
    probe_runs = {str(dict(record["action"])["key"]): str(record["run_id"]) for record in completed
                  if str(dict(record["action"]).get("kind")) == "probe"}
    sources = []
    for record in completed:
        action = dict(record["action"])
        if str(action.get("kind")) != "train" or str(action.get("dataset")) not in DATASETS: continue
        method = str(action.get("method")); dataset = str(action["dataset"])
        if method not in METHOD_VIEWS or int(action.get("views", 0)) != METHOD_VIEWS[method]: continue
        if int(action.get("target", 0)) != int(formal.DATASETS[dataset]["epochs"]): continue
        if action.get("backbone") or action.get("aggregation"): continue
        sources.append({"action": action, "train_run": str(record["run_id"])})
    submitted = list(state.get("submitted", [])); existing = {str(record["key"]) for record in submitted}
    for source in sources:
        action = dict(source["action"]); dataset = str(action["dataset"]); method = str(action["method"])
        source_checkpoint = checkpoint(str(source["train_run"])); config = str(formal.DATASETS[dataset]["config"])
        profile = ["--profile", "imagenet"] if dataset == "imagenet100" else []
        for protocol, fraction in PROTOCOLS:
            key = f"{action['key']}:{protocol}:{fraction}"
            if key in existing: continue
            override = formal.evaluation_override(action)
            probe = dict(override.get("probe", {})); probe["label_fraction"] = fraction; override["probe"] = probe
            tag = str(fraction).replace(".", "p"); name = f"formal-lowlabel-{dataset}-{method}-seed{action['seed_index']}-{protocol}-{tag}"
            module = "fmca_av.cli" if method in formal.FMCA_METHODS else "fmca_av.baseline_cli"
            run_id = submit([
                "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", *profile, "--",
                PYTHON, "-m", module, protocol, "--config", config, "--checkpoint", source_checkpoint,
                "--seed", str(action["seed"]), "--overrides-json", json.dumps(override, separators=(",", ":")),
            ])
            submitted.append({"key": key, "action": action, "protocol": protocol, "fraction": fraction,
                              "source_run": source["train_run"], "run_id": run_id})
            existing.add(key); state["submitted"] = submitted; save(state)
        if dataset in {"cifar10", "cifar100"}:
            key = f"{action['key']}:corruption-eval"
            if key not in existing:
                probe_run = probe_runs[str(action["key"])]
                probe_result = read(Path("runs") / probe_run / "artifacts" / "probe_result.json")
                if probe_result.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
                    raise RuntimeError(f"refusing legacy probe for corruption evaluation: {probe_run}")
                override = formal.evaluation_override(action)
                corruption_root = f"/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/{dataset}-c"
                run_id = submit([
                    "python3", "-m", "harness.cli", "submit", "--name", f"formal-robustness-{dataset}-{method}-seed{action['seed_index']}",
                    "--gpus", "1", "--", PYTHON, "-m", "fmca_av.cli", "corruption-eval", "--config", config,
                    "--checkpoint", source_checkpoint, "--probe-checkpoint", str(probe_result["probe_checkpoint"]),
                    "--root", corruption_root, "--batch-size", "256", "--workers", "8",
                    "--seed", str(action["seed"]), "--overrides-json", json.dumps(override, separators=(",", ":")),
                ])
                submitted.append({"key": key, "action": action, "protocol": "corruption-eval", "fraction": 1.0,
                                  "source_run": source["train_run"], "probe_run": probe_run, "run_id": run_id})
                existing.add(key); state["submitted"] = submitted; save(state)
    wait_all([str(record["run_id"]) for record in submitted])
    state["state"] = "SUCCEEDED"; save(state); return 0


if __name__ == "__main__": raise SystemExit(main())
