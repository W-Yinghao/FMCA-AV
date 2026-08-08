#!/usr/bin/env python3
"""DDP-safe train, calibration, and evaluation pipeline."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from fmca_av.cli import calibrate, evaluate, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overrides-json")
    parser.add_argument("--resume", default="")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.seed is not None:
        os.environ["FMCA_SEED_OVERRIDE"] = str(args.seed)
    if args.overrides_json is not None:
        overrides = json.loads(args.overrides_json)
        if not isinstance(overrides, dict):
            raise ValueError("--overrides-json must decode to a JSON object")
        os.environ["FMCA_CONFIG_OVERRIDES"] = json.dumps(overrides, separators=(",", ":"))
    run_dir_value = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if not run_dir_value:
        raise RuntimeError("this pipeline must be launched through the Slurm harness")
    run_dir = Path(run_dir_value)
    artifacts = run_dir / "artifacts"
    result = train(SimpleNamespace(config=args.config, output="", resume=args.resume))
    if result:
        return int(result)
    if args.train_only:
        return 0
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return 0
    train_result_path = artifacts / "train_result.json"
    with train_result_path.open("r", encoding="utf-8") as handle:
        train_result = json.load(handle)
    checkpoint = train_result.get("best_checkpoint") or train_result.get("last_checkpoint")
    if not checkpoint:
        raise RuntimeError("training produced neither a best nor a last checkpoint")
    calibration_path = artifacts / "calibration.pt"
    result = calibrate(
        SimpleNamespace(
            config=args.config,
            checkpoint=checkpoint,
            output=str(calibration_path),
            device="cuda",
        )
    )
    if result:
        return int(result)
    return int(
        evaluate(
            SimpleNamespace(
                config=args.config,
                checkpoint=checkpoint,
                calibration=str(calibration_path),
                output=str(artifacts / "evaluation.json"),
                device="cuda",
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
