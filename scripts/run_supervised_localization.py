#!/usr/bin/env python3
"""Quantitative CAM/localization controls for supervised Lightning checkpoints."""

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

from fmca_av.config import load_config
from fmca_av.supervised_cli import SupervisedVision
from scripts.run_dependence_localization import (
    MEAN, STD, cub_samples, feature_map, image_tensor, imagenet_samples,
    localization_metrics, normalize_map, upsample, voc_samples,
)


def calculated_maps(model: SupervisedVision, inputs: Tensor) -> dict[str, Tensor]:
    spatial = feature_map(model.backbone, inputs); _, channels, height, width = spatial.shape
    local = spatial.permute(0, 2, 3, 1).reshape(-1, channels)
    centered = local - local.mean(0, keepdim=True); _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    eigen_cam = (centered @ vectors[0]).abs().reshape(height, width)
    predicted = int(model(inputs).argmax(1)); class_cam = F.relu(local @ model.classifier.weight[predicted]).reshape(height, width)
    activation = local.square().sum(1).sqrt().reshape(height, width)
    y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, height, device=inputs.device), torch.linspace(-1, 1, width, device=inputs.device), indexing="ij")
    center = torch.exp(-(x_grid.square() + y_grid.square()) / 0.35)
    grayscale = (inputs * STD.to(inputs.device) + MEAN.to(inputs.device)).mean(1, keepdim=True)
    dx = F.pad((grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edge = F.interpolate(dx + dy, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return {name: normalize_map(value) for name, value in {
        "class_cam": class_cam, "eigen_cam": eigen_cam, "activation_norm": activation,
        "center_gaussian": center, "edge_gradient": edge, "random": torch.rand(height, width, device=inputs.device),
    }.items()}


def faithfulness(model: SupervisedVision, inputs: Tensor, value: Tensor, generator: torch.Generator) -> dict[str, float]:
    resized = upsample(value, inputs.shape[-2:]); pixels = resized.numel(); count = max(1, round(0.2 * pixels)); order = torch.argsort(resized.flatten())
    masks = {"bottom": order[:count], "top": order[-count:], "random": torch.randperm(pixels, generator=generator)[:count].to(inputs.device)}
    side = max(1, round(count ** 0.5)); y0 = (inputs.shape[-2] - side) // 2; x0 = (inputs.shape[-1] - side) // 2
    center = torch.zeros(inputs.shape[-2:], dtype=torch.bool, device=inputs.device); center[y0:y0 + side, x0:x0 + side] = True
    masks["center"] = torch.nonzero(center.flatten(), as_tuple=False).flatten()[:count]
    with torch.inference_mode():
        reference_representation = model.backbone(inputs); reference_logits = model.classifier(reference_representation); predicted = reference_logits.argmax(1)
        result = {}
        for name, indices in masks.items():
            modified = inputs.clone().flatten(2); modified[:, :, indices] = 0.0; modified = modified.reshape_as(inputs)
            changed_representation = model.backbone(modified); changed_logits = model.classifier(changed_representation)
            result[f"{name}_representation_cosine_drop"] = float(1.0 - F.cosine_similarity(reference_representation, changed_representation).mean())
            result[f"{name}_predicted_logit_drop"] = float((reference_logits.gather(1, predicted[:, None]) - changed_logits.gather(1, predicted[:, None])).mean())
        fractions = torch.linspace(0.0, 1.0, 11, device=inputs.device)
        curve_orders = {"top": torch.flip(order, dims=(0,)), "bottom": order,
                        "random": torch.randperm(pixels, generator=generator).to(inputs.device)}
        flattened = inputs.flatten(2); reference_batch = reference_representation.expand(len(fractions), -1)
        for name, ranked in curve_orders.items():
            deletion = flattened.repeat(len(fractions), 1, 1); insertion = torch.zeros_like(deletion)
            for step, fraction in enumerate(fractions):
                selected_count = int(round(float(fraction) * pixels))
                if selected_count:
                    selected = ranked[:selected_count]; deletion[step, :, selected] = 0.0; insertion[step, :, selected] = flattened[0, :, selected]
            deletion_features = model.backbone(deletion.reshape(-1, *inputs.shape[1:]))
            insertion_features = model.backbone(insertion.reshape(-1, *inputs.shape[1:]))
            deletion_similarity = F.cosine_similarity(reference_batch, deletion_features)
            insertion_similarity = F.cosine_similarity(reference_batch, insertion_features)
            result[f"{name}_deletion_cosine_auc"] = float(torch.trapezoid(deletion_similarity, fractions))
            result[f"{name}_insertion_cosine_auc"] = float(torch.trapezoid(insertion_similarity, fractions))
            result[f"{name}_faithfulness_auc_gap"] = result[f"{name}_insertion_cosine_auc"] - result[f"{name}_deletion_cosine_auc"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("cub", "voc", "imagenet"), required=True); parser.add_argument("--root", required=True); parser.add_argument("--labels", default="")
    parser.add_argument("--samples", type=int, default=100); parser.add_argument("--size", type=int, default=224); parser.add_argument("--randomize-backbone", action="store_true"); parser.add_argument("--output", default="")
    args = parser.parse_args(); started = time.perf_counter(); config = load_config(args.config); device = torch.device("cuda")
    model = SupervisedVision.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    if args.randomize_backbone:
        for module in model.backbone.modules():
            if hasattr(module, "reset_parameters"): module.reset_parameters()
    root = Path(args.root); iterator = cub_samples(root) if args.dataset == "cub" else voc_samples(root) if args.dataset == "voc" else imagenet_samples(root, Path(args.labels))
    generator = torch.Generator().manual_seed(int(config["seed"]) + 16000); records = []
    for sample_index, (identifier, image_path, mask_path, box) in enumerate(iterator):
        if sample_index >= args.samples: break
        with Image.open(image_path) as source: image = source.convert("RGB"); original_shape = (image.height, image.width); inputs = image_tensor(image, args.size, device)
        foreground = None
        if mask_path is not None:
            import numpy as np
            with Image.open(mask_path) as source: mask = np.asarray(source)
            foreground = torch.from_numpy(((mask > 0) & (mask < 255)).copy()).to(device)
        with torch.inference_mode(): maps = calculated_maps(model, inputs)
        per_sample = {"id": identifier, "maps": {}}
        for name, low_resolution in maps.items():
            per_sample["maps"][name] = localization_metrics(upsample(low_resolution, original_shape), foreground, box)
            if name in {"class_cam", "eigen_cam"}: per_sample["maps"][name].update(faithfulness(model, inputs, low_resolution, generator))
        records.append(per_sample)
    summary = {}
    for map_name in sorted({name for record in records for name in record["maps"]}):
        keys = sorted({key for record in records for key in record["maps"][map_name]})
        summary[map_name] = {key: statistics.fmean(record["maps"][map_name][key] for record in records if key in record["maps"][map_name]) for key in keys}
    payload = {"method": "supervised_random_labels" if config["experiment"].get("random_labels", False) else "supervised",
               "dataset": args.dataset, "samples": len(records), "randomize_backbone": args.randomize_backbone,
               "runtime_seconds": time.perf_counter() - started,
               "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
               "summary": summary, "records": records}
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "supervised_localization.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix + ".tmp"); temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"stage": "supervised_localization", **{key: value for key, value in payload.items() if key != "records"}}, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
