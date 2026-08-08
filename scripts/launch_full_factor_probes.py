#!/usr/bin/env python3
"""Run the preregistered full-strength factor spectral controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CHANNEL_WATCHER = "20260807-063008_launch-e7-factor-channel-wave"
STABILITY_WATCHER = "20260807-070515_launch-e7-factor-stability-wave"
BASE_SOURCE = Path("runs/20260807-061336_continue-factor-wave-fixed/artifacts/submitted_jobs.tsv")
CONFIGS = {
    "dsprites": "configs/ssl/dsprites_smoke.json",
    "shapes3d": "configs/ssl/shapes3d_smoke.json",
    "smallnorb": "configs/ssl/smallnorb_smoke.json",
    "mpi3d_toy": "configs/ssl/mpi3d_toy_smoke.json",
    "mpi3d_realistic": "configs/ssl/mpi3d_realistic_smoke.json",
    "mpi3d_real": "configs/ssl/mpi3d_real_smoke.json",
}
STATE_PATH = Path("results/orchestration/full_factor_probes_state.json")


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def latest_retry(run_id: str) -> str:
    jobs = dict(read(Path("harness/state/jobs.json")).get("jobs", {}))
    current = run_id; visited = {current}
    while True:
        children = [value for value in jobs.values() if str(value.get("retry_from", "")) == current
                    and str(value.get("run_id", "")) not in visited]
        if not children: return current
        chosen = max(children, key=lambda value: str(value.get("created_at", "")))
        current = str(chosen["run_id"]); visited.add(current)


def save(payload: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def wait_success(run_ids: list[str], label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        resolved = [latest_retry(run_id) for run_id in run_ids]
        states = {run_id: run_state(run_id) for run_id in resolved}
        failed = {run_id: value for run_id, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failed:
            raise RuntimeError(f"{label} failed: " + json.dumps(failed, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            return


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def checkpoint(run_id: str) -> str:
    result = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    assert isinstance(result, dict)
    value = result.get("best_checkpoint") or result.get("last_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"checkpoint missing for factor source {run_id}")
    return str(value)


def sources(extra_trainings: list[dict[str, object]], source_files: tuple[Path, ...]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for line in BASE_SOURCE.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 3 or fields[0] != "train":
            continue
        dataset, train_run = fields[1], fields[2]
        key = (dataset, "default", 1)
        seen.add(key)
        records.append({
            "dataset": dataset, "channel": "default", "seed_index": 1,
            "config": CONFIGS[dataset], "train_run": train_run,
        })
    for source_file in source_files:
        payload = read(source_file)
        if not isinstance(payload, list):
            raise RuntimeError(f"invalid factor source file: {source_file}")
        for raw in payload:
            record = dict(raw)
            dataset = str(record["dataset"])
            channel = str(record.get("channel", "default"))
            seed_index = int(record.get("seed_index", 1))
            key = (dataset, channel, seed_index)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "dataset": dataset,
                "channel": channel,
                "seed_index": seed_index,
                "config": str(record["config"]),
                "train_run": str(record["train_run"]),
            })
    for raw in extra_trainings:
        record = dict(raw)
        key = (str(record["dataset"]), "default", int(record["seed_index"]))
        if key not in seen:
            seen.add(key)
            records.append({
                "dataset": key[0], "channel": key[1], "seed_index": key[2],
                "config": str(record["config"]), "train_run": str(record["train_run"]),
            })
    return records


def main() -> int:
    state = read(STATE_PATH) if STATE_PATH.is_file() else {
        "state": "RUNNING", "chain_runs": [], "sources": [], "extra_trainings": [], "submitted": [],
    }
    assert isinstance(state, dict)
    chain_runs = list(state.get("chain_runs", []))
    current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain_runs: chain_runs.append(current_chain_run)
    state["chain_runs"] = chain_runs
    state["state"] = "RUNNING"
    save(state)

    wait_success([CHANNEL_WATCHER, STABILITY_WATCHER], "factor source watchers")
    source_files = (
        Path("runs") / latest_retry(CHANNEL_WATCHER) / "artifacts" / "submitted.json",
        Path("runs") / latest_retry(STABILITY_WATCHER) / "artifacts" / "submitted.json",
    )
    extra_trainings = list(state.get("extra_trainings", []))
    existing_extra = {(str(record["dataset"]), int(record["seed_index"])) for record in extra_trainings}
    for dataset_index, dataset in enumerate(("mpi3d_realistic", "mpi3d_real")):
        for seed_index in (2, 3):
            if (dataset, seed_index) in existing_extra:
                continue
            seed = 20273500 + dataset_index * 10 + seed_index
            name = f"e7-{dataset}-stability-seed{seed_index}-full"
            override = json.dumps({"experiment": {"name": name}}, separators=(",", ":"))
            train_run = submit([
                "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--",
                "env", "FMCA_CONFIG_OVERRIDES=" + override, f"FMCA_SEED_OVERRIDE={seed}",
                "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIGS[dataset],
            ])
            extra_trainings.append({
                "dataset": dataset, "seed_index": seed_index, "seed": seed,
                "config": CONFIGS[dataset], "train_run": train_run,
            })
            state["extra_trainings"] = extra_trainings
            save(state)
    wait_success([str(record["train_run"]) for record in extra_trainings], "extra factor stability training")
    source_records = sources(extra_trainings, source_files)
    wait_success([str(record["train_run"]) for record in source_records], "factor source training")
    state["sources"] = source_records
    save(state)

    submitted = list(state.get("submitted", []))
    completed_keys = {str(record["key"]) for record in submitted}
    for record in source_records:
        key = f"{record['dataset']}:{record['channel']}:seed{record['seed_index']}"
        if key in completed_keys:
            continue
        source_run = str(record["train_run"])
        run_id = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"full-factor-{record['dataset']}-{record['channel']}-seed{record['seed_index']}",
            "--gpus", "1", "--", "env", "FMCA_SEED_OVERRIDE=20273001", PYTHON, "-m", "scripts.run_factor_spectral_probe",
            "--config", str(record["config"]), "--checkpoint", checkpoint(source_run),
            "--calibration", f"runs/{source_run}/artifacts/calibration.pt", "--train-samples", "20000",
            "--test-samples", "5000", "--random-repeats", "100", "--rotation-repeats", "20", "--device", "cuda",
        ])
        submitted.append({"key": key, "source": record, "run_id": run_id})
        state["submitted"] = submitted
        save(state)

    wait_success([str(record["run_id"]) for record in submitted], "full factor probes")
    state["state"] = "SUCCEEDED"
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
