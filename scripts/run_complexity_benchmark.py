#!/usr/bin/env python3
"""E10 one-factor GPU throughput and peak-memory benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from fmca_av.config import load_config
from fmca_av.objectives import fmca_score
from fmca_av.operators import MOMENT_ACCUMULATION_POLICY, SCIENTIFIC_CORRECTNESS_VERSION, estimate_moments
from fmca_av.vision_module import VisionFMCAAV


def measure(base: dict[str, object], condition: dict[str, object], warmup: int, iterations: int) -> dict[str, object]:
    config = copy.deepcopy(base)
    config["data"]["num_views"] = int(condition["views"])
    config["data"]["batch_size"] = int(condition["batch"])
    config["model"]["feature_dim"] = int(condition["features"])
    config["model"]["backbone"] = str(condition["backbone"])
    config["trainer"]["precision"] = str(condition["precision"])
    size = int(condition["size"]); batch = int(condition["batch"]); views = int(condition["views"])
    try:
        model = VisionFMCAAV(config).cuda().train(); optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.randn(batch, views, 3, size, size, device="cuda")
        mixed = str(condition["precision"]) != "32-true"
        torch.cuda.reset_peak_memory_stats(); durations = []
        for iteration in range(warmup + iterations):
            optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
                f, g, _ = model.feature_maps(images); moments = estimate_moments(f, g, centered=True)
                loss = -fmca_score(moments, "trace", ridge=1e-3)
            loss.backward(); optimizer.step(); torch.cuda.synchronize()
            if iteration >= warmup: durations.append(time.perf_counter() - start)
        mean = sum(durations) / len(durations)
        return {**condition, "status": "success", "seconds_per_step": mean, "encoded_images_per_second": batch * views / mean, "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2), "parameters": sum(parameter.numel() for parameter in model.parameters())}
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache(); return {**condition, "status": "oom", "failure_reason": str(error)}
    except (RuntimeError, ValueError) as error:
        torch.cuda.empty_cache(); return {**condition, "status": "failed", "failure_reason": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="")
    parser.add_argument("--warmup", type=int, default=3); parser.add_argument("--iterations", type=int, default=10); args = parser.parse_args()
    base = load_config("configs/ssl/cifar10_smoke.json")
    reference = {"views": 2, "batch": 128, "features": 128, "backbone": "resnet18_cifar", "size": 32, "precision": "32-true"}
    conditions = []
    for views in (1, 2, 4, 8, 16): conditions.append({**reference, "axis": "views", "views": views})
    for features in (32, 64, 128, 256, 512): conditions.append({**reference, "axis": "features", "features": features})
    for batch in (64, 128, 256, 512): conditions.append({**reference, "axis": "batch", "batch": batch})
    for precision in ("32-true", "16-mixed"): conditions.append({**reference, "axis": "precision", "precision": precision})
    large = {"views": 2, "batch": 8, "features": 128, "size": 224, "precision": "16-mixed"}
    for backbone in ("resnet18_imagenet", "resnet50_imagenet", "convnext_tiny", "vit_s_16"):
        conditions.append({**large, "axis": "backbone", "backbone": backbone})
    records = [measure(base, condition, args.warmup, args.iterations) for condition in conditions]
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "moment_accumulation_policy": MOMENT_ACCUMULATION_POLICY,
        "device": torch.cuda.get_device_name(),
        "conditions": records,
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "complexity.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix + ".tmp"); temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"stage": "complexity", "conditions": len(records), "device": payload["device"], "moment_accumulation_policy": MOMENT_ACCUMULATION_POLICY}) + "\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
