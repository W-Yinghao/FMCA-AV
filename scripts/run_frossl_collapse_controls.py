#!/usr/bin/env python3
"""300-second Slurm-only controller for the preregistered FroSSL audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
BASE_CONFIG = "configs/ssl/cifar10_frossl.json"
OFFICIAL_CONFIG = "configs/ssl/cifar10_frossl_official_m2.json"
SEEDS = (20260821, 20260822, 20260823)
OFFICIAL_SEEDS = (20260841, 20260842, 20260843)
CONTROL_SEED = 20260831
MILESTONES = (25, 50, 100, 200)
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED", "SKIPPED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def external_checkpoints() -> dict[int, str]:
    state = read(Path("results/orchestration/external_multiview_baselines_20260809_scientific_correctness_v1.json"))
    records = state["records"]
    values: dict[int, str] = {}
    for index, seed in enumerate(SEEDS, 1):
        record = records[f"train:frossl:v8:s{index}"]
        result = read(Path("runs") / str(record["run_id"]) / "artifacts" / "train_result.json")
        checkpoint = str(result["last_checkpoint"])
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        values[seed] = checkpoint
    return values


def control_override(cell: str) -> dict[str, object]:
    if cell not in {"B", "C", "D"}:
        raise ValueError(cell)
    sequential = cell in {"B", "D"}
    use_rrc = cell == "B"
    return {
        "experiment": {"name": f"cifar10-frossl-m8-control-{cell.lower()}"},
        "seed": CONTROL_SEED,
        "data": {
            "num_views": 8,
            "batch_size": 256,
            "augmentation": {"random_resized_crop": use_rrc},
        },
        "model": {"view_forward_mode": "sequential" if sequential else "flattened"},
        "objective": {"invariance_weight": 2.0},
        "optimizer": {"scheduler_t_max": 200},
        "trainer": {
            "max_epochs": 200,
            "checkpoint_milestones": list(MILESTONES),
        },
    }


def definitions(cpu_run: str) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for index, seed in enumerate(SEEDS, 1):
        values.append({
            "key": f"existing_audit:s{index}", "kind": "existing_audit", "seed": seed,
            "seed_index": index, "dependencies": ["cpu_regression"],
        })
    for cell in ("B", "C"):
        train_key = f"control_train:{cell}"
        values.append({"key": train_key, "kind": "control_train", "cell": cell, "dependencies": ["cpu_regression"]})
        for epoch in MILESTONES:
            values.append({
                "key": f"control_audit:{cell}:e{epoch}", "kind": "control_audit", "cell": cell,
                "epoch": epoch, "train_key": train_key, "dependencies": [train_key],
            })
    for index, seed in enumerate(OFFICIAL_SEEDS, 1):
        train_key = f"official_train:s{index}"
        values.append({
            "key": train_key, "kind": "official_train", "seed": seed, "seed_index": index,
            "dependencies": ["cpu_regression"],
        })
        for kind in ("official_probe", "official_knn", "official_diagnostics"):
            values.append({
                "key": f"{kind}:s{index}", "kind": kind, "seed": seed, "seed_index": index,
                "train_key": train_key, "dependencies": [train_key],
            })
    d_dependencies = ["control_audit:B:e200", "control_audit:C:e200"]
    values.append({
        "key": "control_train:D", "kind": "conditional_d_train", "cell": "D",
        "dependencies": d_dependencies,
    })
    for epoch in MILESTONES:
        values.append({
            "key": f"control_audit:D:e{epoch}", "kind": "control_audit", "cell": "D",
            "epoch": epoch, "train_key": "control_train:D", "dependencies": ["control_train:D"],
        })
    final_dependencies = [str(value["key"]) for value in values]
    values.append({"key": "aggregate", "kind": "aggregate", "dependencies": final_dependencies})
    values.insert(0, {
        "key": "cpu_regression", "kind": "external", "run_id": cpu_run,
        "state": "QUEUED", "dependencies": [],
    })
    return values


def refresh() -> None:
    subprocess.run(
        ["python3", "-m", "harness.cli", "status"],
        check=True, stdout=subprocess.DEVNULL,
    )


def run_state(run_id: str) -> str:
    path = Path("runs") / run_id / "status.json"
    return str(read(path)["state"]) if path.is_file() else "BLOCKED"


def train_checkpoint(record: dict[str, object], records: dict[str, dict[str, object]]) -> str:
    train = records[str(record["train_key"])]
    result = read(Path("runs") / str(train["run_id"]) / "artifacts" / "train_result.json")
    value = str(result.get("last_checkpoint") or result.get("best_checkpoint"))
    if not Path(value).is_file():
        raise FileNotFoundError(value)
    return value


def milestone_checkpoint(record: dict[str, object], records: dict[str, dict[str, object]]) -> str:
    train = records[str(record["train_key"])]
    path = (
        Path("runs") / str(train["run_id"]) / "artifacts" / "checkpoints"
        / f"epoch-{int(record['epoch']):04d}.ckpt"
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def audit_result(record: dict[str, object]) -> dict[str, object]:
    return read(Path("runs") / str(record["run_id"]) / "artifacts" / "frossl_collapse_audit.json")


def recovered(record: dict[str, object]) -> bool:
    result = audit_result(record)["conditions"]["eval_saved"]
    centered = result["backbone"]["centered_covariance"]
    return (
        float(result["knn_accuracy"]) >= 0.60
        and float(centered["effective_rank"]) >= 20.0
        and float(centered["top_eigenvalue_share"]) <= 0.80
    )


def command(
    record: dict[str, object], records: dict[str, dict[str, object]], state_file: Path,
    checkpoints: dict[int, str],
) -> tuple[str, int, list[str]]:
    kind = str(record["kind"])
    if kind == "aggregate":
        return "frossl-controls-aggregate", 0, [
            PYTHON, "-m", "scripts.render_frossl_collapse_controls", "--state-file", str(state_file),
        ]
    if kind == "existing_audit":
        seed = int(record["seed"]); index = int(record["seed_index"])
        override = {"seed": seed, "data": {"num_views": 8}, "objective": {"invariance_weight": 2.0}}
        return f"frossl-m8-existing-seed{index}-audit", 1, [
            "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
            PYTHON, "-m", "scripts.audit_frossl_collapse", "--config", BASE_CONFIG,
            "--checkpoint", checkpoints[seed],
        ]
    if kind in {"control_train", "conditional_d_train"}:
        cell = str(record["cell"])
        return f"frossl-m8-control-{cell.lower()}-train", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", BASE_CONFIG,
            "--seed", str(CONTROL_SEED), "--overrides-json",
            json.dumps(control_override(cell), separators=(",", ":")),
        ]
    if kind == "control_audit":
        cell = str(record["cell"]); epoch = int(record["epoch"])
        override = control_override(cell)
        return f"frossl-m8-control-{cell.lower()}-e{epoch}-audit", 1, [
            "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
            PYTHON, "-m", "scripts.audit_frossl_collapse", "--config", BASE_CONFIG,
            "--checkpoint", milestone_checkpoint(record, records), "--conditions", "eval_saved",
        ]
    if kind == "official_train":
        index = int(record["seed_index"]); seed = int(record["seed"])
        return f"frossl-official-m2-seed{index}-train", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", OFFICIAL_CONFIG,
            "--seed", str(seed),
        ]
    source = train_checkpoint(record, records)
    index = int(record["seed_index"]); seed = int(record["seed"])
    if kind == "official_probe":
        return f"frossl-official-m2-seed{index}-probe", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe", "--config", OFFICIAL_CONFIG,
            "--checkpoint", source, "--seed", str(seed),
        ]
    if kind == "official_knn":
        return f"frossl-official-m2-seed{index}-knn", 1, [
            "env", f"FMCA_SEED_OVERRIDE={seed}", PYTHON, "-m", "fmca_av.cli", "knn",
            "--config", OFFICIAL_CONFIG, "--checkpoint", source, "--workers", "8",
            "--batch-size", "256", "--bank-chunk-size", "8192",
        ]
    if kind == "official_diagnostics":
        return f"frossl-official-m2-seed{index}-diagnostics", 1, [
            "env", f"FMCA_SEED_OVERRIDE={seed}", PYTHON, "-m", "scripts.evaluate_baseline_diagnostics",
            "--config", OFFICIAL_CONFIG, "--checkpoint", source,
        ]
    raise ValueError(kind)


def submit(
    record: dict[str, object], records: dict[str, dict[str, object]], state_file: Path,
    checkpoints: dict[int, str],
) -> str:
    name, gpus, payload = command(record, records, state_file, checkpoints)
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", str(gpus)]
    if gpus:
        argv.extend(["--profile", "imagenet_ddp"])
    argv.extend(["--", *payload])
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        if "GPU limit exceeded" in message or "Slurm job limit reached" in message:
            return ""
        raise RuntimeError(message)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--cpu-run", required=True)
    args = parser.parse_args()
    state_file = Path(args.state_file)
    checkpoints = external_checkpoints()
    all_definitions = definitions(args.cpu_run)
    definitions_by_key = {str(item["key"]): item for item in all_definitions}
    state = read(state_file) if state_file.is_file() else {
        "scope": "FroSSL M8 collapse audit and official CIFAR M2 replication",
        "poll_seconds": POLL_SECONDS,
        "gpu_budget": 6,
        "recovery_gate": read(Path("configs/experiments/frossl_collapse_controls_20260812.json"))["preregistered_recovery_gate"],
        "created_at": now(), "state": "RUNNING", "records": {},
    }
    records = dict(state.get("records", {}))
    for key, definition in definitions_by_key.items():
        records.setdefault(key, {**definition, "state": definition.get("state", "WAITING")})
    state["records"] = records
    write(state_file, state)

    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        state["last_polled_at"] = now()
        for record in records.values():
            run_id = str(record.get("run_id", ""))
            if run_id and record.get("state") not in TERMINAL:
                current = run_state(run_id)
                if current != record.get("state"):
                    record["state"] = current
                    record["updated_at"] = now()
        failures = [r for r in records.values() if r.get("state") in {"FAILED", "STOPPED", "BLOCKED"}]
        if failures:
            state["state"] = "FAILED"; state["failed_at"] = now(); write(state_file, state)
            raise RuntimeError("FroSSL control action failed: " + json.dumps(failures, sort_keys=True))
        if records["aggregate"].get("state") == "SUCCEEDED":
            state["state"] = "SUCCEEDED"; state["completed_at"] = now(); write(state_file, state)
            return 0

        d_record = records["control_train:D"]
        if d_record.get("state") == "WAITING":
            left = records["control_audit:B:e200"]
            right = records["control_audit:C:e200"]
            if left.get("state") == right.get("state") == "SUCCEEDED" and (recovered(left) or recovered(right)):
                d_record["state"] = "SKIPPED"
                d_record["reason"] = "B or C passed the preregistered recovery gate"
                for epoch in MILESTONES:
                    child = records[f"control_audit:D:e{epoch}"]
                    child["state"] = "SKIPPED"; child["reason"] = "conditional D training skipped"

        for key in definitions_by_key:
            record = records[key]
            if record.get("state") != "WAITING" or record.get("kind") == "external":
                continue
            dependency_states = [records[str(item)].get("state") for item in record.get("dependencies", [])]
            if not all(value in {"SUCCEEDED", "SKIPPED"} for value in dependency_states):
                continue
            run_id = submit(record, records, state_file, checkpoints)
            if not run_id:
                break
            record["run_id"] = run_id; record["state"] = "QUEUED"; record["submitted_at"] = now()
            write(state_file, state)
        write(state_file, state)


if __name__ == "__main__":
    raise SystemExit(main())
