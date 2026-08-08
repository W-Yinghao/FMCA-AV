#!/usr/bin/env python3
"""Resume selected local watchers after ImageNet DDP scaling has handed off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--min-action-index", type=int, default=3)
    parser.add_argument("--run", action="append", required=True)
    return parser.parse_args()


def restart(run_id: str) -> None:
    status_path = Path("runs") / run_id / "status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if payload.get("state") == "RUNNING":
        return
    if payload.get("state") == "SUCCEEDED":
        return
    subprocess.run(
        ["python3", "-m", "harness.cli", "restart-watchers", "--run", run_id],
        check=True,
    )


def main() -> int:
    args = parse_args()
    state_path = Path(args.state_file)
    while True:
        time.sleep(POLL_SECONDS)
        subprocess.run(
            ["python3", "-m", "harness.cli", "status"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        formal_state = str(state.get("state", ""))
        handed_off = (
            int(state.get("action_index", 0)) >= args.min_action_index
            and bool(state.get("current_run"))
        )
        terminal = formal_state in {"SUCCEEDED", "FAILED", "BLOCKED", "STOPPED"}
        if handed_off or terminal:
            for run_id in args.run:
                restart(run_id)
            return 0 if formal_state not in {"FAILED", "BLOCKED", "STOPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
