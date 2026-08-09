#!/usr/bin/env python3
"""Compare direct and recursively composed dependence operators across CNN stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from fmca_av.baselines import BaselineSSL
from fmca_av.config import load_config
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION, inverse_sqrt_covariance
from fmca_av.supervised_cli import SupervisedVision
from fmca_av.vision_module import VisionFMCAAV
from scripts.run_dependence_localization import cub_samples, image_tensor, normalize_map


def stage_maps(backbone, inputs: Tensor) -> list[Tensor]:
    network = getattr(backbone, "network", backbone)
    if all(hasattr(network, name) for name in ("conv1", "layer1", "layer2", "layer3", "layer4")):
        values = network.relu(network.bn1(network.conv1(inputs)))
        if hasattr(network, "maxpool"): values = network.maxpool(values)
        outputs = [values]
        for name in ("layer1", "layer2", "layer3", "layer4"):
            values = getattr(network, name)(values); outputs.append(values)
        return outputs
    if all(hasattr(network, name) for name in ("stem", "layer1", "layer2", "layer3", "layer4")):
        values = network.stem(inputs); outputs = [values]
        for name in ("layer1", "layer2", "layer3", "layer4"):
            values = getattr(network, name)(values); outputs.append(values)
        return outputs
    if hasattr(network, "features"):
        values = inputs; snapshots = []
        for module in network.features:
            values = module(values)
            if values.ndim == 4: snapshots.append(values)
        if len(snapshots) < 5: raise ValueError("feature backbone exposes fewer than five spatial stages")
        indices = [round(index * (len(snapshots) - 1) / 4) for index in range(5)]
        return [snapshots[index] for index in indices]
    if all(hasattr(network, name) for name in ("_process_input", "class_token", "encoder")):
        patches = network._process_input(inputs); batch, count, width = patches.shape
        tokens = torch.cat((network.class_token.expand(batch, -1, -1), patches), dim=1)
        tokens = network.encoder.dropout(tokens + network.encoder.pos_embedding)
        layers = list(network.encoder.layers.children()); snapshots = [tokens[:, 1:]]
        boundaries = {max(1, round(index * len(layers) / 4)) for index in range(1, 5)}
        for index, layer in enumerate(layers, 1):
            tokens = layer(tokens)
            if index in boundaries: snapshots.append(tokens[:, 1:])
        snapshots = snapshots[:4] + [network.encoder.ln(tokens)[:, 1:]]
        side = int(round(count ** 0.5))
        return [value.transpose(1, 2).reshape(batch, width, side, side) for value in snapshots]
    raise ValueError("unsupported backbone for stage composition")


def projection(channels: int, modes: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(channels, modes, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(value, mode="reduced").Q.to(device=device, dtype=torch.float32)


def projected_stages(backbone, inputs: Tensor, projections: list[Tensor] | None, modes: int, seed: int) -> tuple[list[Tensor], list[Tensor]]:
    stages = stage_maps(backbone, inputs)
    if len(stages) != 5: raise ValueError(f"expected five stage maps, received {len(stages)}")
    if projections is None:
        projections = [projection(int(value.shape[1]), modes, seed + index * 101, inputs.device) for index, value in enumerate(stages)]
    values = []
    for stage, basis in zip(stages, projections):
        resized = F.interpolate(stage, size=(14, 14), mode="bilinear", align_corners=False)
        local = resized.permute(0, 2, 3, 1).reshape(-1, resized.shape[1])
        values.append(local @ basis)
    return values, projections


def whitening(values: Tensor, ridge: float) -> tuple[Tensor, Tensor]:
    mean = values.mean(0, keepdim=True)
    centered = values.double() - mean.double()
    covariance = centered.transpose(0, 1) @ centered / max(1, len(values) - 1)
    transform = inverse_sqrt_covariance(covariance, ridge)
    return mean, transform.float()


def rank_correlation(left: Tensor, right: Tensor) -> float:
    left_rank = torch.argsort(torch.argsort(left.flatten())).double()
    right_rank = torch.argsort(torch.argsort(right.flatten())).double()
    left_rank -= left_rank.mean(); right_rank -= right_rank.mean()
    return float((left_rank @ right_rank) / (left_rank.norm() * right_rank.norm()).clamp_min(1e-12))


def comparison(direct: Tensor, recursive: Tensor) -> dict[str, float]:
    direct = normalize_map(direct); recursive = normalize_map(recursive)
    count = max(1, round(0.2 * direct.numel()))
    direct_top = torch.topk(direct.flatten(), count).indices
    recursive_top = torch.topk(recursive.flatten(), count).indices
    direct_mask = torch.zeros(direct.numel(), dtype=torch.bool, device=direct.device); direct_mask[direct_top] = True
    recursive_mask = torch.zeros_like(direct_mask); recursive_mask[recursive_top] = True
    intersection = (direct_mask & recursive_mask).sum(); union = (direct_mask | recursive_mask).sum()
    return {
        "rank_correlation": rank_correlation(direct, recursive),
        "normalized_l2": float((direct - recursive).norm() / direct.norm().clamp_min(1e-12)),
        "top20_iou": float(intersection / union.clamp_min(1)),
    }


def load_backbone(model_type: str, checkpoint: str, config: dict[str, object], device: torch.device):
    if model_type == "fmca": model = VisionFMCAAV.load_from_checkpoint(checkpoint, config=config, map_location=device)
    elif model_type == "baseline": model = BaselineSSL.load_from_checkpoint(checkpoint, config=config, map_location=device)
    else: model = SupervisedVision.load_from_checkpoint(checkpoint, config=config, map_location=device)
    return model.to(device).eval().backbone


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=("fmca", "baseline", "supervised"), default="fmca")
    parser.add_argument("--root", required=True); parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--evaluation-samples", type=int, default=50); parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-3); parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--output", default=""); args = parser.parse_args(); started = time.perf_counter()
    config = load_config(args.config); seed = int(config["seed"]) + 19000; torch.manual_seed(seed)
    device = torch.device("cuda"); backbone = load_backbone(args.model_type, args.checkpoint, config, device)
    torch.cuda.reset_peak_memory_stats()
    samples = list(cub_samples(Path(args.root)))[:args.calibration_samples + args.evaluation_samples]
    if len(samples) < args.calibration_samples + args.evaluation_samples: raise RuntimeError("insufficient CUB samples")
    projections = None; calibration_values: list[list[Tensor]] = [[] for _ in range(5)]
    with torch.inference_mode():
        for _, image_path, _, _ in samples[:args.calibration_samples]:
            from PIL import Image
            with Image.open(image_path) as source: inputs = image_tensor(source.convert("RGB"), args.size, device)
            values, projections = projected_stages(backbone, inputs, projections, args.modes, seed)
            for index, value in enumerate(values): calibration_values[index].append(value.cpu())
    stacked = [torch.cat(values).to(device) for values in calibration_values]
    parameters = [whitening(value, args.ridge) for value in stacked]
    whitened = [(value - mean.to(device)) @ transform.to(device) for value, (mean, transform) in zip(stacked, parameters)]
    transitions = [left.transpose(0, 1) @ right / max(1, len(left) - 1) for left, right in zip(whitened, whitened[1:])]
    direct_operator = whitened[0].transpose(0, 1) @ whitened[-1] / max(1, len(whitened[0]) - 1)
    recursive_operator = transitions[0]
    for transition in transitions[1:]: recursive_operator = recursive_operator @ transition
    records = []
    with torch.inference_mode():
        for identifier, image_path, _, _ in samples[args.calibration_samples:]:
            from PIL import Image
            with Image.open(image_path) as source: inputs = image_tensor(source.convert("RGB"), args.size, device)
            values, _ = projected_stages(backbone, inputs, projections, args.modes, seed)
            first = (values[0] - parameters[0][0].to(device)) @ parameters[0][1].to(device)
            direct_map = (first @ direct_operator).square().sum(1).reshape(14, 14)
            recursive_map = (first @ recursive_operator).square().sum(1).reshape(14, 14)
            records.append({"id": identifier, **comparison(direct_map, recursive_map)})
    summary = {key: statistics.fmean(float(record[key]) for record in records) for key in ("rank_correlation", "normalized_l2", "top20_iou")}
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_type": args.model_type, "backbone": config["model"].get("backbone"), "modes": args.modes,
        "calibration_samples": args.calibration_samples, "evaluation_samples": len(records),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "composition_assumption": "five post-stage outputs; ResNet residual DAG is collapsed only at post-addition stage boundaries",
        "direct_singular_values": torch.linalg.svdvals(direct_operator).cpu().tolist(),
        "recursive_singular_values": torch.linalg.svdvals(recursive_operator).cpu().tolist(),
        "summary": summary, "records": records,
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "cnn_composition_maps.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": "cnn_composition_maps", **summary}, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
