#!/usr/bin/env python3
"""Supported-operator FLOPs for external CIFAR-10 baseline steps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from fmca_av.baselines import (
    BaselineSSL,
    fastssl_barlow_twins_loss,
    fastssl_vicreg_loss,
    frossl_loss,
)
from fmca_av.config import load_config
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--views", type=int, choices=(2, 8), required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    config["data"]["num_views"] = args.views
    model = BaselineSSL(config).cuda().train()
    images = torch.randn(args.batch, args.views, 3, 32, 32, device="cuda")
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as trace:
        objective = config["objective"]
        if model.method == "hai_simsiam":
            if args.views != 8:
                raise ValueError("HAI FLOP profiling requires its eight hierarchical views")
            parameters = torch.randn(args.batch, 8, 4, device="cuda")
            loss, _ = model.hai_heads.loss(model.backbone, images, parameters)
        else:
            projections = model.projector(model.backbone(images.flatten(0, 1))).reshape(args.batch, args.views, -1)
        if model.method == "fastssl_barlow_twins":
            loss = fastssl_barlow_twins_loss(projections, objective["off_diagonal_weight"])
        elif model.method == "fastssl_vicreg":
            loss = fastssl_vicreg_loss(projections, objective["invariance_weight"], objective["variance_weight"])
        elif model.method == "frossl":
            loss = frossl_loss(projections, objective["invariance_weight"])
        elif model.method != "hai_simsiam":
            raise ValueError("unsupported external baseline FLOP profile")
        loss.backward()
    torch.cuda.synchronize()
    flops = sum(int(event.flops or 0) for event in trace.key_averages())
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "method": config["experiment"]["method"],
        "views": args.views,
        "parents": args.batch,
        "supported_operator_training_step_flops": flops,
        "flops_per_parent": flops / args.batch,
        "flops_per_encoded_view": flops / (args.batch * args.views),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "scope": "one forward+objective+backward step; PyTorch profiler supported operators only",
        "gpu_name": torch.cuda.get_device_name(),
    }
    output = Path(args.output).resolve() if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "flops.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "external_baseline_flops", **payload}, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
