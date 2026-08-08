#!/usr/bin/env python3
"""Complete a previously submitted TSV wave without duplicating its children."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300


def run_state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-column", type=int, default=-1)
    args = parser.parse_args()
    rows = [line.split("\t") for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    run_ids = [row[args.run_column] for row in rows]
    while True:
        time.sleep(POLL_SECONDS)
        subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
        states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {key: value for key, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures:
            raise RuntimeError("existing wave contains failed children: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            break
    output = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "completed_wave.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps({"source_manifest": str(Path(args.manifest).resolve()), "runs": states}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "complete_existing_wave", "runs": len(run_ids)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
