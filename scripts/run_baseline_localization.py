#!/usr/bin/env python3
"""Localization controls for Lightning SSL baseline checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F

from fmca_av.baseline_cli import _data_module
from fmca_av.baselines import BaselineSSL
from fmca_av.config import load_config
from scripts.run_dependence_localization import (
    MEAN, STD, cub_samples, faithfulness, feature_map, image_tensor,
    imagenet_samples, localization_metrics, normalize_map, upsample, voc_samples,
)


def baseline_maps(model: BaselineSSL, inputs: Tensor) -> dict[str, Tensor]:
    spatial = feature_map(model.backbone, inputs)
    _, channels, height, width = spatial.shape
    local = spatial.permute(0, 2, 3, 1).reshape(-1, channels)
    centered = local - local.mean(0, keepdim=True)
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    eigen_cam = (centered @ vectors[0]).abs().reshape(height, width)
    projector_energy = model.projector(local).square().sum(1).sqrt().reshape(height, width)
    activation = local.square().sum(1).sqrt().reshape(height, width)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=inputs.device),
        torch.linspace(-1.0, 1.0, width, device=inputs.device), indexing="ij",
    )
    center = torch.exp(-(x_grid.square() + y_grid.square()) / 0.35)
    grayscale = (inputs * STD.to(inputs.device) + MEAN.to(inputs.device)).mean(1, keepdim=True)
    dx = F.pad((grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edge = F.interpolate(dx + dy, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return {
        "eigen_cam": normalize_map(eigen_cam), "projector_energy": normalize_map(projector_energy),
        "activation_norm": normalize_map(activation), "center_gaussian": normalize_map(center),
        "edge_gradient": normalize_map(edge), "random": torch.rand(height, width, device=inputs.device),
    }


def backbone_faithfulness(model: BaselineSSL, inputs: Tensor, value: Tensor, generator: torch.Generator) -> dict[str, float]:
    # The shared helper only requires an object with a backbone attribute.
    return faithfulness(model, inputs, value, generator)  # type: ignore[arg-type]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("cub", "voc", "imagenet"), required=True)
    parser.add_argument("--root", required=True); parser.add_argument("--labels", default="")
    parser.add_argument("--samples", type=int, default=100); parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--randomize-backbone", action="store_true"); parser.add_argument("--output", default="")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config); torch.manual_seed(int(config["seed"]) + 12000)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(config["seed"]) + 12000)
    device = torch.device("cuda")
    model = BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    if args.randomize_backbone:
        for module in model.backbone.modules():
            if hasattr(module, "reset_parameters"): module.reset_parameters()
    root = Path(args.root)
    iterator = cub_samples(root) if args.dataset == "cub" else voc_samples(root) if args.dataset == "voc" else imagenet_samples(root, Path(args.labels))
    generator = torch.Generator().manual_seed(int(config["seed"]) + 12000)
    records = []
    for sample_index, (identifier, image_path, mask_path, box) in enumerate(iterator):
        if sample_index >= args.samples: break
        with Image.open(image_path) as source:
            image = source.convert("RGB"); original_shape = (image.height, image.width); inputs = image_tensor(image, args.size, device)
        foreground = None
        if mask_path is not None:
            import numpy as np
            with Image.open(mask_path) as source: mask = np.asarray(source)
            foreground = torch.from_numpy(((mask > 0) & (mask < 255)).copy()).to(device)
        with torch.inference_mode(): calculated = baseline_maps(model, inputs)
        per_sample = {"id": identifier, "maps": {}}
        for name, low_resolution in calculated.items():
            full = upsample(low_resolution, original_shape)
            per_sample["maps"][name] = localization_metrics(full, foreground, box)
            if name in {"eigen_cam", "projector_energy"}: per_sample["maps"][name].update(backbone_faithfulness(model, inputs, low_resolution, generator))
        records.append(per_sample)
    summary = {}
    for map_name in sorted({name for record in records for name in record["maps"]}):
        keys = sorted({key for record in records for key in record["maps"][map_name]})
        summary[map_name] = {key: statistics.fmean(record["maps"][map_name][key] for record in records if key in record["maps"][map_name]) for key in keys}
        accuracies = []
        for quantile in range(10, 100, 10):
            key = f"box_iou_q{quantile:02d}"; values = [float(record["maps"][map_name][key] >= 0.5) for record in records if key in record["maps"][map_name]]
            if values: accuracies.append((statistics.fmean(values), quantile / 100.0))
        if accuracies: summary[map_name]["max_box_acc_iou50"], summary[map_name]["max_box_acc_quantile"] = max(accuracies)
    payload = {"method": config["experiment"]["method"], "dataset": args.dataset, "samples": len(records),
               "randomize_backbone": args.randomize_backbone, "runtime_seconds": time.perf_counter() - started,
               "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2), "summary": summary, "records": records}
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "localization.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp"); temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"stage": "baseline_localization", **{key: value for key, value in payload.items() if key != "records"}}, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
