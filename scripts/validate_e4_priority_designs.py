#!/usr/bin/env python3
"""Validate E4 parameter and encoded-forward matching before submission."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from fmca_av.config import load_config
from fmca_av.vision_module import VisionFMCAAV
from scripts.e4_priority_designs import DESIGNS, MAX_PARAMETER_DELTA, TARGET_PARAMETERS, encoded_forwards, override


def merge(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)  # type: ignore[index]
        else:
            target[key] = copy.deepcopy(value)


def main() -> int:
    config = load_config("configs/ssl/cifar10_reference.json")
    rows = []
    for design in DESIGNS:
        value = copy.deepcopy(config); merge(value, override(design, 1))
        model = VisionFMCAAV(value)
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        delta = parameters - TARGET_PARAMETERS; forwards = encoded_forwards(design)
        if abs(delta) > MAX_PARAMETER_DELTA:
            raise RuntimeError(f"{design} parameter delta {delta} exceeds {MAX_PARAMETER_DELTA}")
        if forwards != 8:
            raise RuntimeError(f"{design} encodes {forwards} views instead of 8")
        rows.append({"design": design, "trainable_parameters": parameters,
                     "target_parameters": TARGET_PARAMETERS, "parameter_delta": delta,
                     "relative_parameter_delta": delta / TARGET_PARAMETERS,
                     "encoded_forwards_per_parent": forwards})
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]); output = run_dir / "artifacts" / "e4_matched_designs.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "e4_design_validation", "designs": len(rows),
                                 "max_absolute_parameter_delta": max(abs(int(row["parameter_delta"])) for row in rows)}) + "\n")
    print(json.dumps(rows, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
