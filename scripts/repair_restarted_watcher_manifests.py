#!/usr/bin/env python3
"""Restore the two launcher manifests from their intact harness run records."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

FACTOR_RUNS = (
    ("dsprites", "crop", "20260807-063825_e7-dsprites-crop-32step"),
    ("dsprites", "color", "20260807-063826_e7-dsprites-color-32step"),
    ("dsprites", "rotation", "20260807-063827_e7-dsprites-rotation-32step"),
    ("dsprites", "blur", "20260807-064329_e7-dsprites-blur-32step"),
    ("dsprites", "grayscale", "20260807-102908_e7-dsprites-grayscale-32step"),
    ("dsprites", "noise", "20260807-102908_e7-dsprites-noise-32step"),
    ("shapes3d", "crop", "20260807-130521_e7-shapes3d-crop-32step"),
    ("shapes3d", "color", "20260807-132523_e7-shapes3d-color-32step"),
    ("shapes3d", "rotation", "20260807-133023_e7-shapes3d-rotation-32step"),
    ("shapes3d", "blur", "20260807-133524_e7-shapes3d-blur-32step"),
)

CROSS_RUNS = (
    ("cifar100", "simclr", "20260807-102941_cifar100-simclr-screening"),
    ("cifar100", "barlow_twins", "20260807-134022_cifar100-barlow_twins-screening"),
    ("cifar100", "vicreg", "20260807-134022_cifar100-vicreg-screening"),
    ("cifar100", "spectral_contrastive", "20260807-134523_cifar100-spectral_contrastive-screening"),
    ("cifar100", "fastsiam", "20260807-134523_cifar100-fastsiam-screening"),
    ("cifar100", "byol", "20260807-154531_cifar100-byol-screening"),
    ("cifar100", "moco_v2", "20260807-154531_cifar100-moco_v2-screening"),
    ("cifar100", "dino", "20260807-154531_cifar100-dino-screening"),
    ("stl10", "simclr", "20260807-155032_stl10-simclr-screening"),
    ("stl10", "barlow_twins", "20260807-155532_stl10-barlow_twins-screening"),
    ("stl10", "vicreg", "20260807-160033_stl10-vicreg-screening"),
    ("stl10", "spectral_contrastive", "20260807-161034_stl10-spectral_contrastive-screening"),
    ("stl10", "fastsiam", "20260807-161534_stl10-fastsiam-screening"),
    ("stl10", "byol", "20260807-161535_stl10-byol-screening"),
)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def factor_record(dataset: str, channel: str, run_id: str) -> dict[str, object]:
    request = read(RUNS / run_id / "request.json")
    status = read(RUNS / run_id / "status.json")
    if status["state"] in {"FAILED", "STOPPED", "BLOCKED"}:
        raise RuntimeError(f"cannot restore unsuccessful factor run {run_id}: {status['state']}")
    command = [str(item) for item in request["original_command"]]
    override = json.loads(next(item.split("=", 1)[1] for item in command if item.startswith("FMCA_CONFIG_OVERRIDES=")))
    seed = int(next(item.split("=", 1)[1] for item in command if item.startswith("FMCA_SEED_OVERRIDE=")))
    config = command[command.index("--config") + 1]
    return {"dataset": dataset, "channel": channel, "config": config, "seed": seed,
            "override": override, "train_run": run_id}


def main() -> int:
    factor_path = RUNS / "20260807-063008_launch-e7-factor-channel-wave" / "artifacts" / "submitted.json"
    cross_path = RUNS / "20260807-060445_launch-cross-dataset-baseline-wave-fixed" / "artifacts" / "submitted_jobs.tsv"
    write_atomic(factor_path, [factor_record(*entry) for entry in FACTOR_RUNS])
    cross_path.write_text("".join("\t".join(entry) + "\n" for entry in CROSS_RUNS), encoding="utf-8")
    methods = ("simclr", "barlow_twins", "vicreg", "spectral_contrastive", "fastsiam", "byol", "moco_v2", "dino")
    selected: dict[tuple[int, str, int], dict[str, object]] = {}
    pattern = re.compile(r"^e5-cifar10-(.+)-v(2|8)-seed([123])-screening$")
    for run_dir in sorted(RUNS.iterdir()):
        request_path = run_dir / "request.json"; status_path = run_dir / "status.json"
        if not request_path.is_file() or not status_path.is_file(): continue
        request = read(request_path); match = pattern.fullmatch(str(request.get("name", "")))
        if not match: continue
        status = read(status_path)
        if status.get("state") in {"FAILED", "STOPPED", "BLOCKED"}: continue
        method, views_text, seed_index_text = match.groups(); key = (int(views_text), method, int(seed_index_text))
        if key in selected: continue
        command = [str(item) for item in request["original_command"]]
        seed = int(next(item.split("=", 1)[1] for item in command if item.startswith("FMCA_SEED_OVERRIDE=")))
        selected[key] = {"method": method, "views": key[0], "seed_index": key[2], "seed": seed,
                         "train_run": str(request["run_id"])}
    e5_records = [selected[(views, method, seed_index)] for views in (2, 8) for method in methods
                  for seed_index in range(1, 4) if (views, method, seed_index) in selected]
    if len(e5_records) != 44:
        raise RuntimeError(f"expected 44 intact E5 records, found {len(e5_records)}")
    e5_path = RUNS / "20260807-065037_launch-e5-cifar10-matched-views-screening" / "artifacts" / "submitted.json"
    write_atomic(e5_path, e5_records)
    print(f"factor={len(FACTOR_RUNS)} cross={len(CROSS_RUNS)} e5={len(e5_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
