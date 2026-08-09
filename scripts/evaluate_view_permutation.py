#!/usr/bin/env python3
"""Measure parent-feature invariance under a deterministic view permutation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fmca_av.config import load_config
from fmca_av.data.cifar import CIFARDataModule
from fmca_av.objectives import fmca_score
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION, estimate_moments
from fmca_av.vision_module import VisionFMCAAV


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--overrides-json", required=True)
    parser.add_argument("--batches", type=int, default=8); args = parser.parse_args()
    os.environ["FMCA_CONFIG_OVERRIDES"] = args.overrides_json
    config = load_config(args.config); device = torch.device("cuda")
    model = VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location=device).to(device).eval()
    data = CIFARDataModule(config["data"], int(config["seed"])); data.setup()
    f_changes = []; g_errors = []; score_changes = []
    objective = config["objective"]
    with torch.no_grad():
        for index, batch in enumerate(data.calibration_dataloader()):
            if index >= args.batches: break
            views = batch[0].to(device); parent = batch[3].to(device) if len(batch) > 3 else None
            permutation = torch.arange(views.shape[1] - 1, -1, -1, device=device)
            first_f, first_g, _ = model.feature_maps(views, parent)
            second_f, second_g, _ = model.feature_maps(views[:, permutation], parent)
            f_changes.append(float(torch.linalg.vector_norm(second_f - first_f) /
                                   torch.linalg.vector_norm(first_f).clamp_min(torch.finfo(first_f.dtype).tiny)))
            g_errors.append(float(torch.linalg.vector_norm(second_g - first_g[:, permutation]) /
                                  torch.linalg.vector_norm(first_g).clamp_min(torch.finfo(first_g.dtype).tiny)))
            first_score = fmca_score(estimate_moments(first_f, first_g, centered=True), str(objective["name"]),
                                     ridge=float(objective.get("ridge", 1e-3)),
                                     logdet_margin=float(objective.get("logdet_margin", 1e-6)))
            second_score = fmca_score(estimate_moments(second_f, second_g, centered=True), str(objective["name"]),
                                      ridge=float(objective.get("ridge", 1e-3)),
                                      logdet_margin=float(objective.get("logdet_margin", 1e-6)))
            score_changes.append(float((second_score - first_score).abs() /
                                       first_score.abs().clamp_min(torch.finfo(first_score.dtype).tiny)))
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()), "permutation": "reverse view order",
        "batches": len(f_changes), "f_relative_change_mean": sum(f_changes) / len(f_changes),
        "f_relative_change_max": max(f_changes), "g_equivariance_error_mean": sum(g_errors) / len(g_errors),
        "score_relative_change_mean": sum(score_changes) / len(score_changes),
        "score_relative_change_max": max(score_changes),
    }
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]); output = run_dir / "artifacts" / "view_permutation.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "view_permutation", **payload}) + "\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
