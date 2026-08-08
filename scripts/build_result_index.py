#!/usr/bin/env python3
"""Build an auditable run/result index from harness artifacts without hashes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
from typing import Any


TERMINAL_FAILURES = {"FAILED", "STOPPED", "BLOCKED"}


GROUP_CLAIMS = {
    "E0": "C1", "E1": "C1", "E2": "C2", "E3": "C1,C2",
    "E4": "C2,C3", "E5": "C3", "E6": "C3", "E7": "C4,C5",
    "E8": "C6", "E9": "C7", "E10": "C3",
}


def experiment_group(name: str, artifact_files: list[str]) -> str:
    """Infer the plan work package from stable run/artifact naming conventions."""
    value = name.lower()
    for number in range(11):
        if re.search(rf"(?:^|[-_])e{number}(?:[-_]|$)", value):
            return f"E{number}"
    joined = " ".join(artifact_files).lower()
    rules = (
        ("E1", ("gaussian_operator", "nonlinear_toy", "exact_channel", "finite-reference")),
        ("E2", ("gradient_variance", "fixed-parent", "fixed-budget")),
        ("E3", ("numerical", "logdet", "fractional")),
        ("E4", ("aggregation", "operator-baseline", "architecture-smoke")),
        ("E5", ("formal-ssl", "matched-view", "matched-compute", "linear-probe", "knn")),
        ("E6", ("robustness", "lowlabel", "low-label", "transfer", "detection", "segmentation")),
        ("E7", ("factor", "tsd", "data-processing")),
        ("E8", ("markov",)),
        ("E9", ("localization", "faithfulness", "composition-map", "random-label")),
        ("E10", ("complexity", "scaling", "ddp")),
    )
    searchable = value + " " + joined
    for group, needles in rules:
        if any(needle in searchable for needle in needles):
            return group
    return ""


def command_metadata(command: str, name: str = "") -> dict[str, Any]:
    """Extract reproducibility selectors without executing or hashing the command."""
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = command.split()

    def option(flag: str) -> str:
        try:
            return argv[argv.index(flag) + 1]
        except (ValueError, IndexError):
            return ""

    config = option("--config")
    seed = option("--seed")
    dataset = ""
    orchestration_tokens = ("state-machine", "chain-step", "launch-", "continue-", "validate-",
                            "render-", "summarize-", "inspect-", "prepare-", "recover-")
    candidates = (
        "tinyimagenet200", "imagenet100", "imagenet1k", "cifar100", "cifar10",
        "stl10", "smallnorb", "shapes3d", "dsprites", "mpi3d", "coco", "voc", "cub",
    )
    if not any(token in name.lower() for token in orchestration_tokens):
        for searchable in (option("--dataset").lower(), option("--root").lower(), name.lower(), config.lower()):
            dataset = next((candidate for candidate in candidates if candidate in searchable), "")
            if dataset:
                break
    return {"config": config, "seed": seed, "dataset": dataset}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def scalar_metrics(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Flatten scalar summaries while deliberately skipping per-sample arrays."""
    if depth > 5:
        return {}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {prefix: value} if prefix else {}
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(scalar_metrics(item, name, depth + 1))
        return flattened
    if isinstance(value, list):
        # Lists usually contain spectra, samples, or sweep records. Preserve only
        # their size so the source JSON remains the sole raw-data authority.
        return {f"{prefix}.__count": len(value)} if prefix else {}
    return {}


def metric_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                records.append({"line": line_number, "parse_error": True})
                continue
            records.append({"line": line_number, **scalar_metrics(payload)})
    return records


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output-dir", default="results/index")
    args = parser.parse_args()
    runs = Path(args.runs).resolve(); output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    indexed = []
    for directory in sorted(path for path in runs.iterdir() if path.is_dir()):
        status = read_json(directory / "status.json")
        request = read_json(directory / "request.json")
        if not isinstance(status, dict) or not isinstance(request, dict):
            continue
        artifacts = directory / "artifacts"
        json_summaries: dict[str, dict[str, Any]] = {}
        artifact_files = []
        if artifacts.is_dir():
            for path in sorted(item for item in artifacts.rglob("*") if item.is_file()):
                relative = str(path.relative_to(directory))
                artifact_files.append(relative)
                if path.suffix == ".json" and path.stat().st_size <= 64 * 1024 * 1024:
                    payload = read_json(path)
                    if payload is not None:
                        json_summaries[relative] = scalar_metrics(payload)
        command_path = directory / "command.txt"
        command = command_path.read_text(encoding="utf-8").strip() if command_path.is_file() else ""
        metadata = command_metadata(command, str(status.get("name", "")))
        group = experiment_group(str(status.get("name", "")), artifact_files)
        indexed.append({
            "run_id": status.get("run_id", directory.name),
            "name": status.get("name", ""),
            "state": status.get("state", ""),
            "requested_gpus": status.get("requested_gpus", ""),
            "actual_gpu_ids": status.get("actual_gpu_ids", []),
            "slurm_job_id": status.get("slurm_job_id", ""),
            "start_time": status.get("start_time"), "end_time": status.get("end_time"),
            "exit_code": status.get("exit_code"), "failure_reason": status.get("failure_reason"),
            "retry_from": status.get("retry_from"), "profile": status.get("profile", ""),
            "experiment_group": group, "claim_id": GROUP_CLAIMS.get(group, ""),
            "dataset_name": metadata["dataset"], "config": metadata["config"], "seed": metadata["seed"],
            "command": command,
            "artifact_files": artifact_files,
            "artifact_scalar_summaries": json_summaries,
            "metric_records": metric_records(directory / "metrics.jsonl"),
        })

    generated_at = datetime.now(timezone.utc).isoformat()
    retry_children: dict[str, list[dict[str, Any]]] = {}
    for record in indexed:
        origin = str(record.get("retry_from") or "")
        if origin:
            retry_children.setdefault(origin, []).append(record)

    def descendants(run_id: str) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        pending = list(retry_children.get(run_id, []))
        seen: set[str] = set()
        while pending:
            value = pending.pop(0); child_id = str(value["run_id"])
            if child_id in seen:
                continue
            seen.add(child_id); values.append(value); pending.extend(retry_children.get(child_id, []))
        return values

    for record in indexed:
        children = descendants(str(record["run_id"]))
        successful = [value for value in children if value["state"] == "SUCCEEDED"]
        selected = (successful[-1] if successful else children[-1]) if children else None
        record["resolved_by_run"] = selected["run_id"] if selected else ""
        record["resolution_state"] = selected["state"] if selected else ""
        record["resolution_note"] = (
            "recovered by retry" if selected and selected["state"] == "SUCCEEDED" else
            "retry in progress" if selected and selected["state"] in {"QUEUED", "RUNNING"} else
            "retry chain remains terminal" if selected else ""
        )

    atomic_json(output / "experiment_index.json", {"generated_at": generated_at, "runs": indexed})
    failure_rows = [record for record in indexed if record["state"] in TERMINAL_FAILURES]
    atomic_json(output / "failure_atlas.json", {"generated_at": generated_at, "runs": failure_rows})

    fields = [
        "run_id", "name", "experiment_group", "claim_id", "dataset_name", "config", "seed",
        "state", "requested_gpus", "slurm_job_id", "start_time", "end_time", "exit_code",
        "failure_reason", "retry_from", "resolved_by_run", "resolution_state", "resolution_note", "profile",
    ]
    for filename, rows in (("experiment_index.csv", indexed), ("failure_atlas.csv", failure_rows)):
        temporary = output / (filename + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        temporary.replace(output / filename)
    print(json.dumps({"indexed_runs": len(indexed), "failure_runs": len(failure_rows), "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
