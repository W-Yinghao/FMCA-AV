#!/usr/bin/env python3
"""Copy a post-fix E2 artifact while validating its checkpoint provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


RUN_PATTERN = re.compile(r"/runs/([^/]+)/artifacts/")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); args = parser.parse_args()
    source = Path(args.input).resolve(); payload = json.loads(source.read_text(encoding="utf-8"))
    checkpoint = str(payload.get("checkpoint", "")); match = RUN_PATTERN.search(checkpoint)
    if not match:
        raise RuntimeError("E2 checkpoint is not inside an auditable harness run")
    train_result = Path("runs") / match.group(1) / "artifacts" / "train_result.json"
    training = json.loads(train_result.read_text(encoding="utf-8"))
    if training.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError("E2 checkpoint is not a post-fix scientific-version checkpoint")
    if int(len(payload.get("conditions", []))) != 10:
        raise RuntimeError("E2 artifact does not contain the expected ten conditions")
    payload["scientific_correctness_version"] = SCIENTIFIC_CORRECTNESS_VERSION
    payload["provenance_source_result"] = str(source)
    payload["provenance_source_train_result"] = str(train_result.resolve())
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]); output = run_dir / "artifacts" / "cifar_gradient_variance.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "e2_postfix_provenance", "conditions": 10,
                                 "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION}) + "\n")
    print(str(output)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
