"""Classify formal-run failures without spending scientific retries on infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


MAX_SCIENTIFIC_ATTEMPTS = 3
MAX_INFRASTRUCTURE_ATTEMPTS = 10


def status_record(run_id: str) -> Dict[str, Any]:
    path = Path("runs") / run_id / "status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def is_infrastructure_failure(run_id: str, terminal: str) -> bool:
    """Return true only for failures outside the experimental computation."""
    if terminal in {"BLOCKED", "STOPPED"}:
        return True
    record = status_record(run_id)
    scheduler_state = str(record.get("scheduler_state", "")).upper()
    if scheduler_state in {
        "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "TIMEOUT", "DEADLINE", "NOT_IN_SQUEUE",
    }:
        return True
    reason = str(record.get("failure_reason", "")).lower()
    markers = (
        "runner could not start command",
        "sbatch failed",
        "sbatch returned no job id",
        "stale active state repaired",
        "slurm state node_fail",
        "slurm state boot_fail",
    )
    return any(marker in reason for marker in markers)


def retry_record(
    action: Dict[str, object],
    run_id: str,
    terminal: str,
    scientific_attempt: int,
    infrastructure_attempt: int = 0,
) -> Dict[str, object] | None:
    """Build the next retry; return None when the appropriate budget is exhausted."""
    if is_infrastructure_failure(run_id, terminal):
        next_infrastructure = infrastructure_attempt + 1
        if next_infrastructure >= MAX_INFRASTRUCTURE_ATTEMPTS:
            return None
        return {
            "action": action,
            "attempt": scientific_attempt,
            "infrastructure_attempt": next_infrastructure,
            "retry_kind": "infrastructure",
            "retry_from": run_id,
        }
    if scientific_attempt >= MAX_SCIENTIFIC_ATTEMPTS:
        return None
    return {
        "action": action,
        "attempt": scientific_attempt + 1,
        "infrastructure_attempt": infrastructure_attempt,
        "retry_kind": "scientific",
        "retry_from": run_id,
    }
