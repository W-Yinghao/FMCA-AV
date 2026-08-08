#!/usr/bin/env python3
"""Quantitative spectral dependence-map localization and faithfulness evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F

from fmca_av.config import load_config
from fmca_av.vision_module import VisionFMCAAV


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def image_tensor(image: Image.Image, size: int, device: torch.device) -> Tensor:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    value = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0) / 255.0
    return ((value - MEAN) / STD).to(device)


def feature_map(backbone, inputs: Tensor) -> Tensor:
    network = getattr(backbone, "network", backbone)
    if all(hasattr(network, name) for name in ("conv1", "bn1", "relu", "layer1", "layer2", "layer3", "layer4")):
        values = network.relu(network.bn1(network.conv1(inputs)))
        if hasattr(network, "maxpool"):
            values = network.maxpool(values)
        for name in ("layer1", "layer2", "layer3", "layer4"):
            values = getattr(network, name)(values)
        return values
    if all(hasattr(network, name) for name in ("stem", "layer1", "layer2", "layer3", "layer4")):
        values = network.stem(inputs)
        for name in ("layer1", "layer2", "layer3", "layer4"):
            values = getattr(network, name)(values)
        return values
    if hasattr(network, "features"):
        return network.features(inputs)
    if all(hasattr(network, name) for name in ("_process_input", "class_token", "encoder")):
        # ViT is an explicit extension: patch tokens form a spatial lattice, but
        # this is not interpreted as the CNN Markov chain assumed by Case 3.
        patches = network._process_input(inputs)
        batch, count, width = patches.shape
        tokens = torch.cat((network.class_token.expand(batch, -1, -1), patches), dim=1)
        encoded = network.encoder(tokens)[:, 1:]
        side = int(round(count ** 0.5))
        if side * side != count:
            raise ValueError(f"ViT patch count {count} is not a square lattice")
        return encoded.transpose(1, 2).reshape(batch, width, side, side)
    raise ValueError("localization currently supports ResNet and ConvNeXt feature maps")


def randomize_from_stage(backbone, stage: int) -> None:
    """Reset a backbone suffix using architecture-aware stage boundaries."""
    network = getattr(backbone, "network", backbone)
    groups: list[list[torch.nn.Module]] = []
    if all(hasattr(network, name) for name in ("conv1", "layer1", "layer2", "layer3", "layer4")):
        stem = [network.conv1]
        if hasattr(network, "bn1"): stem.append(network.bn1)
        groups = [stem] + [[getattr(network, f"layer{index}")] for index in range(1, 5)]
    elif hasattr(network, "features"):
        modules = list(network.features.children())
        for index in range(5):
            left = round(index * len(modules) / 5); right = round((index + 1) * len(modules) / 5)
            groups.append(modules[left:right])
    elif hasattr(network, "encoder") and hasattr(network.encoder, "layers"):
        modules = list(network.encoder.layers.children())
        groups = [[network.conv_proj]] if hasattr(network, "conv_proj") else [[]]
        for index in range(1, 5):
            left = round((index - 1) * len(modules) / 4); right = round(index * len(modules) / 4)
            groups.append(modules[left:right])
    else:
        groups = [[network]]
    start = min(max(stage, 0), len(groups) - 1)
    visited: set[int] = set()
    for group in groups[start:]:
        for root in group:
            for module in root.modules():
                if id(module) in visited: continue
                visited.add(id(module))
                if hasattr(module, "reset_parameters"): module.reset_parameters()


def normalize_map(value: Tensor) -> Tensor:
    value = value - value.min()
    return value / value.max().clamp_min(1e-12)


def maps(model: VisionFMCAAV, calibration: dict[str, object], inputs: Tensor, modes: int) -> dict[str, Tensor]:
    spatial = feature_map(model.backbone, inputs)
    _, channels, height, width = spatial.shape
    local = spatial.permute(0, 2, 3, 1).reshape(-1, channels)
    g = model.g_head(local)
    mean = calibration["mean_g"].to(inputs.device)
    transform = calibration["transform_g"].to(inputs.device)
    eigenvalues = calibration["eigenvalues"].to(inputs.device)
    canonical = (g - mean) @ transform
    count = min(modes, canonical.shape[1])
    spectral = (canonical[:, :count].square() * eigenvalues[:count]).sum(dim=1).reshape(height, width)
    activation = local.square().sum(dim=1).sqrt().reshape(height, width)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=inputs.device),
        torch.linspace(-1.0, 1.0, width, device=inputs.device), indexing="ij",
    )
    center = torch.exp(-(x_grid.square() + y_grid.square()) / 0.35)
    grayscale = (inputs * STD.to(inputs.device) + MEAN.to(inputs.device)).mean(1, keepdim=True)
    dx = F.pad((grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edge = F.interpolate(dx + dy, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    random = torch.rand(height, width, device=inputs.device)
    return {
        "spectral": normalize_map(spectral),
        "activation_norm": normalize_map(activation),
        "center_gaussian": normalize_map(center),
        "edge_gradient": normalize_map(edge),
        "random": normalize_map(random),
    }


def upsample(value: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(value[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def bbox_from_mask(mask: Tensor) -> Optional[tuple[int, int, int, int]]:
    coordinates = torch.nonzero(mask, as_tuple=False)
    if not len(coordinates):
        return None
    y0, x0 = coordinates.min(dim=0).values.tolist()
    y1, x1 = coordinates.max(dim=0).values.tolist()
    return int(x0), int(y0), int(x1 + 1), int(y1 + 1)


def bbox_iou(left, right) -> float:
    if left is None or right is None:
        return 0.0
    x0 = max(left[0], right[0]); y0 = max(left[1], right[1])
    x1 = min(left[2], right[2]); y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def localization_metrics(value: Tensor, foreground: Optional[Tensor], box) -> dict[str, float]:
    if box is None and foreground is not None:
        box = bbox_from_mask(foreground.bool())
    flat = value.flatten()
    threshold = torch.quantile(flat, 0.8)
    selected = value >= threshold
    prediction_box = bbox_from_mask(selected)
    record = {"box_iou_top20": bbox_iou(prediction_box, box)}
    threshold_ious = []
    for quantile in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        candidate = value >= torch.quantile(flat, quantile)
        candidate_iou = bbox_iou(bbox_from_mask(candidate), box)
        threshold_ious.append((candidate_iou, quantile))
        record[f"box_iou_q{int(round(quantile * 100)):02d}"] = candidate_iou
    maximum_iou, best_quantile = max(threshold_ious)
    record["max_box_iou"] = maximum_iou
    record["max_box_iou_quantile"] = best_quantile
    if foreground is not None:
        foreground = foreground.bool()
        maximum = int(torch.argmax(flat))
        y, x = divmod(maximum, value.shape[1])
        record["pointing_game"] = float(foreground[y, x])
        record["foreground_energy_ratio"] = float(value[foreground].sum() / value.sum().clamp_min(1e-12))
        intersection = (selected & foreground).sum()
        union = (selected | foreground).sum()
        record["top20_mask_iou"] = float(intersection / union.clamp_min(1))
        order = torch.argsort(flat, descending=True)
        labels = foreground.flatten()[order].to(torch.float64)
        cumulative_precision = labels.cumsum(0) / torch.arange(1, len(labels) + 1, device=labels.device, dtype=torch.float64)
        record["pixel_auprc"] = float((cumulative_precision * labels).sum() / labels.sum().clamp_min(1.0))
    return record


def faithfulness(model: VisionFMCAAV, inputs: Tensor, value: Tensor, generator: torch.Generator) -> dict[str, float]:
    resized = upsample(value, inputs.shape[-2:])
    pixels = resized.numel()
    count = max(1, int(round(0.2 * pixels)))
    order = torch.argsort(resized.flatten())
    masks = {
        "bottom": order[:count],
        "top": order[-count:],
        "random": torch.randperm(pixels, generator=generator)[:count].to(inputs.device),
    }
    side = max(1, int(round(count ** 0.5)))
    y0 = (inputs.shape[-2] - side) // 2; x0 = (inputs.shape[-1] - side) // 2
    center = torch.zeros(inputs.shape[-2:], dtype=torch.bool, device=inputs.device)
    center[y0:y0 + side, x0:x0 + side] = True
    masks["center"] = torch.nonzero(center.flatten(), as_tuple=False).flatten()[:count]
    with torch.inference_mode():
        reference = model.backbone(inputs)
        result = {}
        for name, indices in masks.items():
            modified = inputs.clone().flatten(2)
            modified[:, :, indices] = 0.0
            changed = model.backbone(modified.reshape_as(inputs))
            result[f"{name}_representation_cosine_drop"] = float(1.0 - F.cosine_similarity(reference, changed).mean())
        # A full perturbation curve distinguishes a genuinely ranked map from a
        # single-threshold masking effect. Zero is the normalized dataset mean.
        fractions = torch.linspace(0.0, 1.0, 11, device=inputs.device)
        curve_orders = {
            "top": torch.flip(order, dims=(0,)),
            "bottom": order,
            "random": torch.randperm(pixels, generator=generator).to(inputs.device),
        }
        reference_batch = reference.expand(len(fractions), -1)
        flattened = inputs.flatten(2)
        for name, ranked in curve_orders.items():
            deletion = flattened.repeat(len(fractions), 1, 1)
            insertion = torch.zeros_like(deletion)
            for step, fraction in enumerate(fractions):
                selected_count = int(round(float(fraction) * pixels))
                if selected_count:
                    selected = ranked[:selected_count]
                    deletion[step, :, selected] = 0.0
                    insertion[step, :, selected] = flattened[0, :, selected]
            deletion_features = model.backbone(deletion.reshape(-1, *inputs.shape[1:]))
            insertion_features = model.backbone(insertion.reshape(-1, *inputs.shape[1:]))
            deletion_similarity = F.cosine_similarity(reference_batch, deletion_features)
            insertion_similarity = F.cosine_similarity(reference_batch, insertion_features)
            result[f"{name}_deletion_cosine_auc"] = float(torch.trapezoid(deletion_similarity, fractions))
            result[f"{name}_insertion_cosine_auc"] = float(torch.trapezoid(insertion_similarity, fractions))
            result[f"{name}_faithfulness_auc_gap"] = result[f"{name}_insertion_cosine_auc"] - result[f"{name}_deletion_cosine_auc"]
    return result


def cub_samples(root: Path):
    images = {int(line.split()[0]): line.split(maxsplit=1)[1] for line in (root / "images.txt").read_text().splitlines()}
    splits = {int(line.split()[0]): int(line.split()[1]) for line in (root / "train_test_split.txt").read_text().splitlines()}
    boxes = {}
    for line in (root / "bounding_boxes.txt").read_text().splitlines():
        fields = line.split(); index = int(fields[0]); x, y, width, height = map(float, fields[1:])
        boxes[index] = (int(x), int(y), int(x + width), int(y + height))
    for index in sorted(images):
        if splits[index] != 0:
            continue
        relative = images[index]
        image_path = root / "images" / relative
        mask_path = root / "segmentations" / str(Path(relative).with_suffix(".png"))
        yield str(index), image_path, mask_path if mask_path.is_file() else None, boxes[index]


def voc_samples(root: Path):
    for image_id in (root / "ImageSets" / "Segmentation" / "val.txt").read_text().splitlines():
        yield image_id, root / "JPEGImages" / f"{image_id}.jpg", root / "SegmentationClass" / f"{image_id}.png", None


def imagenet_samples(root: Path, labels: Path):
    data_root = root / "Data" / "CLS-LOC" if (root / "Data" / "CLS-LOC").is_dir() else root
    with labels.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fields = row["PredictionString"].split()
            boxes = []
            for offset in range(0, len(fields), 5):
                if offset + 4 < len(fields):
                    boxes.append(tuple(map(int, fields[offset + 1:offset + 5])))
            if boxes:
                yield row["ImageId"], data_root / "val" / f"{row['ImageId']}.JPEG", None, boxes[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dataset", choices=("cub", "voc", "imagenet"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--randomize-backbone", action="store_true")
    parser.add_argument("--randomize-from-stage", type=int, choices=range(5))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    torch.manual_seed(int(config["seed"]) + 11000)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(config["seed"]) + 11000)
    device = torch.device("cuda")
    model = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    if args.randomize_backbone or args.randomize_from_stage is not None:
        randomize_from_stage(model.backbone, 0 if args.randomize_backbone else int(args.randomize_from_stage))
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
    root = Path(args.root)
    if args.dataset == "cub": iterator = cub_samples(root)
    elif args.dataset == "voc": iterator = voc_samples(root)
    else: iterator = imagenet_samples(root, Path(args.labels))
    records = []
    generator = torch.Generator().manual_seed(int(config["seed"]) + 11000)
    for sample_index, (identifier, image_path, mask_path, box) in enumerate(iterator):
        if sample_index >= args.samples: break
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            original_shape = (image.height, image.width)
            inputs = image_tensor(image, args.size, device)
        foreground = None
        if mask_path is not None:
            with Image.open(mask_path) as source:
                mask = np.asarray(source)
            foreground = torch.from_numpy(((mask > 0) & (mask < 255)).copy()).to(device)
        per_sample = {"id": identifier, "maps": {}}
        with torch.inference_mode(): calculated = maps(model, calibration, inputs, args.modes)
        for name, low_resolution in calculated.items():
            full = upsample(low_resolution, original_shape)
            per_sample["maps"][name] = localization_metrics(full, foreground, box)
            if name == "spectral": per_sample["maps"][name].update(faithfulness(model, inputs, low_resolution, generator))
        records.append(per_sample)
    summary = {}
    for map_name in sorted({name for record in records for name in record["maps"]}):
        keys = sorted({key for record in records for key in record["maps"][map_name]})
        summary[map_name] = {key: statistics.fmean(record["maps"][map_name][key] for record in records if key in record["maps"][map_name]) for key in keys}
        accuracies = []
        for quantile in range(10, 100, 10):
            key = f"box_iou_q{quantile:02d}"
            values = [float(record["maps"][map_name][key] >= 0.5) for record in records if key in record["maps"][map_name]]
            if values: accuracies.append((statistics.fmean(values), quantile / 100.0))
        if accuracies:
            summary[map_name]["max_box_acc_iou50"], summary[map_name]["max_box_acc_quantile"] = max(accuracies)
    payload = {"dataset": args.dataset, "method": config["experiment"].get("method", "fmca_av"),
               "samples": len(records), "randomize_backbone": args.randomize_backbone,
               "randomize_from_stage": args.randomize_from_stage,
               "runtime_seconds": time.perf_counter() - started,
               "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
               "summary": summary, "records": records}
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "localization.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir:
        with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": "dependence_localization", **{key: value for key, value in payload.items() if key != "records"}}, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
