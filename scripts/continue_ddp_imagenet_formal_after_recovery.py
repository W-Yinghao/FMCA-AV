#!/usr/bin/env python3
"""Run DDP scaling and formal ImageNet-1K FMCA/baseline references after recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
RECOVERY = "20260807-073411_recover-interrupted-e3-tsd-coco-v2"
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TORCHRUN = "/home/infres/yinwang/FMCA-AV/scripts/torchrun"
REFERENCE = "configs/ssl/imagenet1k_reference.json"
FMCA_SEEDS = (20267001, 20267002, 20267003)
BASELINE_METHODS = ("simclr", "vicreg", "moco_v2", "dino")
BASELINE_SEEDS = (20267301, 20267302, 20267303)
EPOCH_TARGETS = tuple(range(25, 101, 25))


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_success(run_id: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        value = state(run_id)
        if value == "SUCCEEDED":
            return
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"{run_id} ended in {value}")


def wait_terminal(run_id: str) -> str:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        value = state(run_id)
        if value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}:
            return value


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
    payload = json.loads((Path("runs") / run_id / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
    value = payload.get("best_checkpoint") or payload.get("last_checkpoint")
    if not value:
        raise RuntimeError(f"checkpoint missing for {run_id}")
    return str(value)


def last_checkpoint(run_id: str) -> str:
    result_path = Path("runs") / run_id / "artifacts" / "train_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        value = payload.get("last_checkpoint") or payload.get("best_checkpoint")
        if value and Path(str(value)).is_file():
            return str(value)
    fallback = Path("runs") / run_id / "artifacts" / "checkpoints" / "last.ckpt"
    return str(fallback) if fallback.is_file() else ""


def save(records: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_fmca_chunks(seed: int, record: dict[str, object], records: list[dict[str, object]], output: Path) -> str:
    resume = ""
    chunks: list[dict[str, object]] = []
    record["chunks"] = chunks
    for target in EPOCH_TARGETS:
        for attempt in range(1, 4):
            override = {
                "trainer": {"max_epochs": target, "checkpoint_save_top_k": 0 if target < 100 else 1},
                "optimizer": {"scheduler_t_max": 100},
            }
            command = [
                TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=2", "-m", "scripts.run_fmca_pipeline",
                "--config", REFERENCE, "--seed", str(seed), "--overrides-json", json.dumps(override, separators=(",", ":")),
            ]
            if resume:
                command += ["--resume", resume]
            if target < EPOCH_TARGETS[-1]:
                command.append("--train-only")
            run_id = submit([
                "python3", "-m", "harness.cli", "submit", "--name", f"imagenet1k-fmca-seed-{seed}-epoch-{target}-try-{attempt}",
                "--gpus", "2", "--profile", "imagenet_ddp", "--", *command,
            ])
            chunk: dict[str, object] = {"target_epoch": target, "attempt": attempt, "run_id": run_id}
            chunks.append(chunk); save(records, output)
            terminal = wait_terminal(run_id); chunk["state"] = terminal
            candidate = last_checkpoint(run_id)
            if candidate:
                resume = candidate
                chunk["last_checkpoint"] = candidate
            save(records, output)
            if terminal == "SUCCEEDED":
                break
            if not candidate or attempt == 3:
                raise RuntimeError(f"FMCA ImageNet chunk {run_id} ended in {terminal} without recoverable completion")
        else:
            raise RuntimeError(f"FMCA ImageNet seed {seed} could not reach epoch {target}")
    record["train_run"] = str(chunks[-1]["run_id"])
    return str(chunks[-1]["run_id"])


def train_baseline_chunks(method: str, seed: int, seed_index: int, base_override: dict[str, object],
                          record: dict[str, object], records: list[dict[str, object]], output: Path) -> str:
    resume = ""
    chunks: list[dict[str, object]] = []
    record["chunks"] = chunks
    for target in EPOCH_TARGETS:
        for attempt in range(1, 4):
            override = {
                **base_override,
                "trainer": {"max_epochs": target, "checkpoint_save_top_k": 0 if target < 100 else 1},
                "optimizer": {"scheduler_t_max": 100},
            }
            command = [
                TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=2", "-m", "fmca_av.baseline_cli", "train",
                "--config", REFERENCE, "--seed", str(seed), "--overrides-json", json.dumps(override, separators=(",", ":")),
            ]
            if resume:
                command += ["--resume", resume]
            run_id = submit([
                "python3", "-m", "harness.cli", "submit", "--name",
                f"imagenet1k-{method}-seed{seed_index}-epoch-{target}-try-{attempt}",
                "--gpus", "2", "--profile", "imagenet_ddp", "--", *command,
            ])
            chunk: dict[str, object] = {"target_epoch": target, "attempt": attempt, "run_id": run_id}
            chunks.append(chunk); save(records, output)
            terminal = wait_terminal(run_id); chunk["state"] = terminal
            candidate = last_checkpoint(run_id)
            if candidate:
                resume = candidate
                chunk["last_checkpoint"] = candidate
            save(records, output)
            if terminal == "SUCCEEDED":
                break
            if not candidate or attempt == 3:
                raise RuntimeError(f"baseline ImageNet chunk {run_id} ended in {terminal} without recoverable completion")
        else:
            raise RuntimeError(f"baseline {method} seed {seed} could not reach epoch {target}")
    record["train_run"] = str(chunks[-1]["run_id"])
    return str(chunks[-1]["run_id"])


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / "submitted.json"
    records: list[dict[str, object]] = []
    wait_success(RECOVERY)

    ddp_configs = {
        1: "configs/ssl/cifar10_ddp1_smoke.json",
        2: "configs/ssl/cifar10_ddp2_smoke.json",
    }
    for gpus, config in ddp_configs.items():
        name = f"e10-cifar10-ddp{gpus}-100step-scaling-recovered"
        if gpus == 1:
            command = [PYTHON, "-m", "scripts.run_fmca_pipeline", "--config", config]
        else:
            command = [TORCHRUN, "--standalone", "--nnodes=1", f"--nproc_per_node={gpus}",
                       "-m", "scripts.run_fmca_pipeline", "--config", config]
        run_id = submit(["python3", "-m", "harness.cli", "submit", "--name", name,
                         "--gpus", str(gpus), "--profile", "v100", "--", *command])
        records.append({"stage": "ddp_scaling", "gpus": gpus, "run_id": run_id})
        save(records, output)
        wait_success(run_id)

    probe_override = json.dumps({"probe": {"devices": 1, "accelerator": "gpu"}}, separators=(",", ":"))
    for seed in FMCA_SEEDS:
        record: dict[str, object] = {"stage": "fmca", "seed": seed}
        records.append(record); save(records, output)
        train_run = train_fmca_chunks(seed, record, records, output)
        probe_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"imagenet1k-fmca-av-linear-probe-seed-{seed}",
            "--gpus", "1", "--profile", "imagenet", "--", "env", "FMCA_CONFIG_OVERRIDES=" + probe_override,
            f"FMCA_SEED_OVERRIDE={seed}", PYTHON, "-m", "fmca_av.cli", "linear-probe", "--config", REFERENCE,
            "--checkpoint", checkpoint(train_run),
        ])
        record["probe_run"] = probe_run; save(records, output); wait_success(probe_run)
        knn_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"imagenet1k-fmca-av-knn-seed-{seed}",
            "--gpus", "1", "--profile", "imagenet", "--", "env", f"FMCA_SEED_OVERRIDE={seed}",
            PYTHON, "-m", "fmca_av.cli", "knn", "--config", REFERENCE, "--checkpoint", checkpoint(train_run),
            "--workers", "12", "--batch-size", "256", "--bank-chunk-size", "8192",
        ])
        record["knn_run"] = knn_run; save(records, output); wait_success(knn_run)

    for method in BASELINE_METHODS:
        for seed_index, seed in enumerate(BASELINE_SEEDS, 1):
            method_seed = seed + 10 * BASELINE_METHODS.index(method)
            override = {
                "experiment": {"name": f"imagenet1k-{method}-reference", "method": method},
                "data": {"num_views": 8},
            }
            record = {"stage": "baseline", "method": method, "seed": method_seed, "views": 8}
            records.append(record); save(records, output)
            train_run = train_baseline_chunks(method, method_seed, seed_index, override, record, records, output)
            evaluation_override = {
                **override,
                "probe": {"devices": 1, "accelerator": "gpu"},
            }
            probe_run = submit([
                "python3", "-m", "harness.cli", "submit", "--name", f"imagenet1k-{method}-linear-probe-seed{seed_index}",
                "--gpus", "1", "--profile", "imagenet", "--", PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe",
                "--config", REFERENCE, "--checkpoint", checkpoint(train_run), "--seed", str(method_seed),
                "--overrides-json", json.dumps(evaluation_override, separators=(",", ":")),
            ])
            record["probe_run"] = probe_run; save(records, output); wait_success(probe_run)
            knn_run = submit([
                "python3", "-m", "harness.cli", "submit", "--name", f"imagenet1k-{method}-knn-seed{seed_index}",
                "--gpus", "1", "--profile", "imagenet", "--", "env",
                "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
                f"FMCA_SEED_OVERRIDE={method_seed}", PYTHON, "-m", "fmca_av.cli", "knn",
                "--config", REFERENCE, "--checkpoint", checkpoint(train_run), "--workers", "12",
                "--batch-size", "256", "--bank-chunk-size", "8192",
            ])
            record["knn_run"] = knn_run; save(records, output); wait_success(knn_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
