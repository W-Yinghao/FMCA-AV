#!/usr/bin/env python3
"""Resume cross-scale TSD and chain its corrected utility probes."""

from __future__ import annotations

import os
import subprocess
import time

import launch_e7_crossscale_tsd_sweep as crossscale


POLL_SECONDS = 300
RECOVERY = "20260807-073411_recover-interrupted-e3-tsd-coco-v2"
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit_utility(cross_watcher: str) -> str:
    command = [
        "python3", "-m", "harness.cli", "watch", "--name", "continue-tsd-utility-after-recovery",
        "--", PYTHON, "scripts/continue_tsd_utility_after_recovery.py",
        "--c10-watcher", RECOVERY, "--cross-watcher", cross_watcher,
    ]
    while True:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def main() -> int:
    crossscale.PREREQUISITE = RECOVERY
    result = int(crossscale.main("non-imagenet"))
    if result:
        return result
    run_id = os.environ["FMCA_HARNESS_RUN_ID"]
    utility = submit_utility(run_id)
    artifacts = os.environ["FMCA_HARNESS_RUN_DIR"] + "/artifacts/utility_watcher.txt"
    with open(artifacts, "w", encoding="utf-8") as handle:
        handle.write(utility + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
