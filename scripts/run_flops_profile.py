#!/usr/bin/env python3
"""Profile supported-op training FLOPs for the E10 backbone/view/K axes."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

from fmca_av.config import load_config
from fmca_av.objectives import fmca_score
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION, estimate_moments
from fmca_av.vision_module import VisionFMCAAV


def measure(base: dict[str, object], condition: dict[str, object]) -> dict[str, object]:
    config = copy.deepcopy(base); config["model"]["backbone"] = condition["backbone"]
    config["model"]["feature_dim"] = condition["features"]; config["data"]["num_views"] = condition["views"]
    batch = int(condition["batch"]); views = int(condition["views"]); size = int(condition["size"])
    model = VisionFMCAAV(config).cuda().train(); images = torch.randn(batch, views, 3, size, size, device="cuda")
    model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True,
                 record_shapes=False, profile_memory=False) as trace:
        f_features, g_features, _ = model.feature_maps(images)
        moments = estimate_moments(f_features, g_features, centered=True)
        loss = -fmca_score(moments, "trace", ridge=1e-3); loss.backward()
    torch.cuda.synchronize()
    flops = sum(int(event.flops or 0) for event in trace.key_averages())
    return {
        **condition, "supported_operator_training_step_flops": flops,
        "flops_per_encoded_view": flops / (batch * views),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "scope": "one forward+objective+backward step; PyTorch profiler supported operators only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=""); args = parser.parse_args()
    base = load_config("configs/ssl/cifar10_smoke.json")
    conditions = []
    reference = {"axis": "views", "backbone": "resnet18_cifar", "features": 128, "batch": 2, "views": 2, "size": 32}
    for views in (1, 2, 4, 8, 16): conditions.append({**reference, "axis": "views", "views": views})
    for features in (32, 64, 128, 256, 512): conditions.append({**reference, "axis": "features", "features": features})
    for backbone in ("resnet18_imagenet", "resnet50_imagenet", "convnext_tiny", "vit_s_16"):
        conditions.append({"axis": "backbone", "backbone": backbone, "features": 128, "batch": 2, "views": 2, "size": 224})
    records = []
    for condition in conditions:
        try: records.append({"status": "success", **measure(base, condition)})
        except (RuntimeError, ValueError) as error:
            torch.cuda.empty_cache(); records.append({"status": "failed", **condition, "failure_reason": str(error)})
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "device": torch.cuda.get_device_name(),
        "conditions": records,
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "flops_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "flops_profile", "conditions": len(records)}) + "\n")
    failures = [record for record in records if record["status"] != "success"]
    print(json.dumps({"conditions": len(records), "successes": len(records) - len(failures),
                      "failures": len(failures)}, indent=2))
    if failures:
        raise RuntimeError(f"FLOPs profiling failed for {len(failures)} of {len(records)} frozen conditions")
    return 0


if __name__ == "__main__": raise SystemExit(main())
