#!/usr/bin/env python3
"""HAI CIFAR-10 controller using only the existing Slurm harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/cifar10_hai_simsiam.json"
SEEDS = (20260821, 20260822, 20260823)
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def definitions() -> list[dict[str, object]]:
    values: list[dict[str, object]] = [
        {"key": "cpu_test", "kind": "prerequisite", "dependencies": []},
        {"key": "smoke", "kind": "prerequisite", "dependencies": []},
        {"key": "flops:hai_simsiam:v8", "kind": "flops", "method": "hai_simsiam", "views": 8,
         "dependencies": ["cpu_test", "smoke"]},
    ]
    for seed_index, seed in enumerate(SEEDS, 1):
        train_key = f"train:hai_simsiam:v8:s{seed_index}"
        values.append({
            "key": train_key, "kind": "train", "method": "hai_simsiam", "views": 8,
            "seed": seed, "seed_index": seed_index, "dependencies": ["cpu_test", "smoke"],
        })
        for kind in ("probe", "knn", "diagnostics"):
            values.append({
                "key": f"{kind}:hai_simsiam:v8:s{seed_index}", "kind": kind,
                "method": "hai_simsiam", "views": 8, "seed": seed,
                "seed_index": seed_index, "train_key": train_key, "dependencies": [train_key],
            })
    final = [str(value["key"]) for value in values if value["kind"] not in {"prerequisite", "train"}]
    values.append({"key": "aggregate", "kind": "aggregate", "dependencies": final})
    return values


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    path = Path("runs") / run_id / "status.json"
    return str(read(path)["state"]) if path.is_file() else "BLOCKED"


def override(seed: int) -> str:
    return json.dumps({
        "experiment": {"name": "cifar10-hai-simsiam-v8", "method": "hai_simsiam"},
        "seed": seed,
    }, separators=(",", ":"))


def checkpoint(record: dict[str, object], records: dict[str, dict[str, object]]) -> str:
    train = records[str(record["train_key"])]
    payload = read(Path("runs") / str(train["run_id"]) / "artifacts" / "train_result.json")
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError("refusing checkpoint from another correctness version")
    value = payload.get("last_checkpoint") or payload.get("best_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"missing HAI checkpoint from {train['run_id']}")
    return str(value)


def command(
    record: dict[str, object], records: dict[str, dict[str, object]], state_file: Path
) -> tuple[str, int, list[str]]:
    kind = str(record["kind"])
    if kind == "aggregate":
        return "hai-c10-aggregate", 0, [
            PYTHON, "-m", "scripts.render_external_multiview_baselines", "--state-file", str(state_file),
            "--output-subdir", "hai",
        ]
    if kind == "flops":
        return "hai-c10-flops", 1, [
            PYTHON, "-m", "scripts.profile_external_baseline_flops", "--config", CONFIG,
            "--views", "8", "--batch", "2",
        ]
    seed = int(record["seed"]); seed_index = int(record["seed_index"]); settings = override(seed)
    if kind == "train":
        return f"hai-c10-seed{seed_index}-pretrain", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", CONFIG,
            "--seed", str(seed), "--overrides-json", settings,
        ]
    source = checkpoint(record, records)
    if kind == "probe":
        return f"hai-c10-seed{seed_index}-linear-probe", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe", "--config", CONFIG,
            "--checkpoint", source, "--seed", str(seed), "--overrides-json", settings,
        ]
    if kind == "knn":
        return f"hai-c10-seed{seed_index}-knn", 1, [
            "env", "FMCA_CONFIG_OVERRIDES=" + settings, f"FMCA_SEED_OVERRIDE={seed}", PYTHON,
            "-m", "fmca_av.cli", "knn", "--config", CONFIG, "--checkpoint", source,
            "--workers", "8", "--batch-size", "256", "--bank-chunk-size", "8192",
        ]
    if kind == "diagnostics":
        return f"hai-c10-seed{seed_index}-diagnostics", 1, [
            "env", "FMCA_CONFIG_OVERRIDES=" + settings, f"FMCA_SEED_OVERRIDE={seed}", PYTHON,
            "-m", "scripts.evaluate_baseline_diagnostics", "--config", CONFIG, "--checkpoint", source,
        ]
    raise ValueError(f"unknown HAI action kind {kind}")


def submit(record: dict[str, object], records: dict[str, dict[str, object]], state_file: Path) -> str:
    name, gpus, payload = command(record, records, state_file)
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", str(gpus)]
    if gpus:
        argv.extend(["--profile", "imagenet_ddp"])
    argv.extend(["--", *payload])
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        message = result.stderr.strip()
        if "GPU limit exceeded" in message or "Slurm job limit reached" in message:
            return ""
        raise RuntimeError(message or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--cpu-test-run", required=True)
    parser.add_argument("--smoke-run", required=True)
    args = parser.parse_args()
    state_file = Path(args.state_file)
    state = read(state_file) if state_file.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "scope": "CIFAR-10 faithful SimSiam + HAI reimplementation",
        "poll_seconds": POLL_SECONDS, "state": "RUNNING", "created_at": now(), "records": {},
    }
    records = dict(state.get("records", {}))
    records.setdefault("cpu_test", {"key": "cpu_test", "kind": "prerequisite", "run_id": args.cpu_test_run, "state": "RUNNING"})
    records.setdefault("smoke", {"key": "smoke", "kind": "prerequisite", "run_id": args.smoke_run, "state": "RUNNING"})
    ordered = definitions()
    for value in ordered:
        records.setdefault(str(value["key"]), {**value, "state": "WAITING"})
    state["records"] = records; state["state"] = "RUNNING"; write(state_file, state)
    while True:
        time.sleep(POLL_SECONDS)
        refresh(); state["last_polled_at"] = now()
        for record in records.values():
            run_id = str(record.get("run_id", ""))
            if run_id and record.get("state") not in TERMINAL:
                record["state"] = run_state(run_id); record["updated_at"] = now()
        failures = [record for record in records.values() if record.get("state") in {"FAILED", "STOPPED", "BLOCKED"}]
        if failures:
            state["state"] = "FAILED"; state["failed_at"] = now(); write(state_file, state)
            raise RuntimeError("HAI action failed: " + json.dumps(failures, sort_keys=True))
        if records["aggregate"].get("state") == "SUCCEEDED":
            state["state"] = "SUCCEEDED"; state["completed_at"] = now(); write(state_file, state)
            return 0
        write(state_file, state)
        for definition in ordered:
            record = records[str(definition["key"])]
            if record.get("state") != "WAITING":
                continue
            if not all(records[str(key)].get("state") == "SUCCEEDED" for key in record.get("dependencies", [])):
                continue
            run_id = submit(record, records, state_file)
            if not run_id:
                break
            record["run_id"] = run_id; record["state"] = "QUEUED"; record["submitted_at"] = now()
            write(state_file, state)


if __name__ == "__main__":
    raise SystemExit(main())
