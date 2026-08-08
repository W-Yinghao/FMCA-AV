#!/usr/bin/env python3
"""Frozen-network CIFAR-10 conditional-view score/gradient variance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from fmca_av.config import load_config
from fmca_av.data.cifar import CIFARFiles, CIFARViewTransform
from fmca_av.objectives import fmca_score
from fmca_av.operators import estimate_moments
from fmca_av.vision_module import VisionFMCAAV


def augmented_batch(base: CIFARFiles, indices: list[int], views: int, transform: CIFARViewTransform,
                    seed: int, device: torch.device) -> torch.Tensor:
    parents = []
    for parent_offset, index in enumerate(indices):
        image, _ = base[index]
        generator = torch.Generator().manual_seed(seed + parent_offset * 1009)
        parents.append(torch.stack([transform(image, generator) for _ in range(views)]))
    return torch.stack(parents).to(device)


def gradient(model: VisionFMCAAV, batch: torch.Tensor, parameters: list[torch.nn.Parameter],
             anchor: torch.Tensor | None = None) -> tuple[float, torch.Tensor]:
    if anchor is None:
        f_features, g_features, _ = model.feature_maps(batch)
    else:
        batch_size, views = batch.shape[:2]
        anchor_representation = model.backbone(anchor)
        # Reuse the trained g-to-f parent path, but hold this anchor view fixed
        # across all conditional resamples. Mean/DeepSets accept a singleton set.
        anchor_g = model.g_head(anchor_representation)
        f_features = model.f_head(model.parent_aggregator(anchor_g[:, None, :]))
        conditional_representation = model.backbone(batch.flatten(0, 1)).reshape(batch_size, views, -1)
        g_features = model.g_head(conditional_representation.flatten(0, 1)).reshape(batch_size, views, -1)
    objective = model.config["objective"]
    score = fmca_score(
        estimate_moments(f_features, g_features, centered=bool(model.config["model"].get("centered", True))),
        str(objective["name"]), ridge=float(objective.get("ridge", 1e-3)),
        logdet_margin=float(objective.get("logdet_margin", 1e-6)),
    )
    values = torch.autograd.grad(score, parameters, allow_unused=True)
    flattened = torch.cat([
        (value if value is not None else torch.zeros_like(parameter)).detach().flatten().cpu().float()
        for value, parameter in zip(values, parameters)
    ])
    return float(score.detach()), flattened


def reference_gradient(model: VisionFMCAAV, base: CIFARFiles, indices: list[int], transform: CIFARViewTransform,
                       parameters: list[torch.nn.Parameter], views: int, repetitions: int,
                       seed: int, device: torch.device, anchor: torch.Tensor | None = None) -> tuple[float, torch.Tensor]:
    total_score = 0.0
    total_gradient: torch.Tensor | None = None
    for repetition in range(repetitions):
        batch = augmented_batch(base, indices, views, transform, seed + repetition * 1000003, device)
        score, value = gradient(model, batch, parameters, anchor)
        total_score += score
        total_gradient = value.double() if total_gradient is None else total_gradient + value.double()
    assert total_gradient is not None
    return total_score / repetitions, total_gradient / repetitions


def condition(model: VisionFMCAAV, base: CIFARFiles, indices: list[int], transform: CIFARViewTransform,
              parameters: list[torch.nn.Parameter], views: int, repetitions: int, seed: int,
              reference_score: float, reference: torch.Tensor, device: torch.device,
              anchor: torch.Tensor | None = None) -> dict[str, object]:
    score_sum = score_square_sum = cosine_sum = cosine_square_sum = 0.0
    gradient_sum = torch.zeros_like(reference)
    gradient_square_norm_sum = error_square_norm_sum = 0.0
    for repetition in range(repetitions):
        batch = augmented_batch(base, indices, views, transform, seed + repetition * 1000003, device)
        score, value_float = gradient(model, batch, parameters, anchor)
        value = value_float.double()
        error = value - reference
        cosine = float(F.cosine_similarity(value[None], reference[None]).item())
        score_sum += score; score_square_sum += score * score
        cosine_sum += cosine; cosine_square_sum += cosine * cosine
        gradient_sum += value
        gradient_square_norm_sum += float(value.square().sum())
        error_square_norm_sum += float(error.square().sum())
    mean_gradient = gradient_sum / repetitions
    score_mean = score_sum / repetitions
    score_variance = max(0.0, (score_square_sum - repetitions * score_mean * score_mean) / max(1, repetitions - 1))
    cosine_mean = cosine_sum / repetitions
    cosine_variance = max(0.0, (cosine_square_sum - repetitions * cosine_mean * cosine_mean) / max(1, repetitions - 1))
    return {
        "parents": len(indices), "views": views, "total_views": len(indices) * views,
        "repetitions": repetitions, "score_reference": reference_score, "score_mean": score_mean,
        "score_bias": score_mean - reference_score, "score_variance": score_variance,
        "gradient_bias_l2": float(torch.linalg.vector_norm(mean_gradient - reference)),
        "gradient_variance": max(0.0, gradient_square_norm_sum / repetitions - float(mean_gradient.square().sum())),
        "gradient_mse_to_reference": error_square_norm_sum / repetitions,
        "gradient_cosine_to_reference_mean": cosine_mean,
        "gradient_cosine_to_reference_std": cosine_variance ** 0.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--fixed-parents", type=int, default=64)
    parser.add_argument("--total-view-budget", type=int, default=512)
    parser.add_argument("--reference-parents", type=int, default=512)
    parser.add_argument("--reference-views", type=int, default=16)
    parser.add_argument("--reference-repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20264100)
    parser.add_argument("--fixed-anchor-parent", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda")
    model = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.") and parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no non-backbone parameters are available for the gradient experiment")
    base = CIFARFiles(str(config["data"]["root"]), str(config["data"]["dataset"]), train=True)
    permutation = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.seed)).tolist()
    transform = CIFARViewTransform(config["data"].get("augmentation", {}))
    reference_indices = permutation[:args.reference_parents]
    reference_anchor = None
    if args.fixed_anchor_parent:
        reference_anchor = augmented_batch(
            base, reference_indices, 1, transform, args.seed + 80000000, device,
        )[:, 0]
    reference_score, reference = reference_gradient(
        model, base, reference_indices, transform, parameters, args.reference_views,
        args.reference_repetitions, args.seed + 90000000, device, reference_anchor,
    )
    records = []
    for design in ("fixed_parent", "fixed_total_views"):
        for views in (1, 2, 4, 8, 16):
            parents = args.fixed_parents if design == "fixed_parent" else args.total_view_budget // views
            record = condition(
                model, base, permutation[:parents], transform, parameters, views, args.repetitions,
                args.seed + (0 if design == "fixed_parent" else 10000000) + views * 100000,
                reference_score, reference, device,
                reference_anchor[:parents] if reference_anchor is not None else None,
            )
            record["design"] = design
            records.append(record)
    payload = {
        "dataset": str(config["data"]["dataset"]), "checkpoint": str(Path(args.checkpoint).resolve()),
        "gradient_parameters": sum(parameter.numel() for parameter in parameters),
        "reference": {"parents": args.reference_parents, "views": args.reference_views,
                      "repetitions": args.reference_repetitions, "score": reference_score,
                      "gradient_norm": float(torch.linalg.vector_norm(reference))},
        "conditions": records,
        "parent_protocol": "fixed independent anchor; conditional views resampled" if args.fixed_anchor_parent else "parent derived from conditional view set",
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "cifar_gradient_variance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "e2_cifar_gradient_variance", "conditions": len(records)}) + "\n")
    print(json.dumps({"reference": payload["reference"], "conditions": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
