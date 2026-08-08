#!/usr/bin/env python3
"""Small CPU validation for the explicit regular-FMCA parent-image path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from fmca_av.config import load_config
from fmca_av.data.cifar import CIFARDataModule
from fmca_av.vision_module import VisionFMCAAV


def main() -> int:
    os.environ["FMCA_CONFIG_OVERRIDES"] = json.dumps({
        "data": {"num_views": 1, "include_raw_parent": True},
        "model": {"parent_aggregation": "raw"},
        "trainer": {"accelerator": "cpu", "devices": 1},
    }, separators=(",", ":"))
    config = load_config("configs/ssl/cifar10_reference.json")
    data = CIFARDataModule(config["data"], int(config["seed"]))
    data.setup()
    samples = [data.datasets["train"][index] for index in range(2)]
    if any(len(sample) != 4 for sample in samples):
        raise AssertionError("raw-parent dataset items must contain four fields")
    views = torch.stack([sample[0] for sample in samples])
    parents = torch.stack([sample[3] for sample in samples])
    model = VisionFMCAAV(config).eval()
    with torch.inference_mode():
        f_features, g_features, representations = model.feature_maps(views, parents)
    expected = (2, int(config["model"]["feature_dim"]))
    if tuple(f_features.shape) != expected:
        raise AssertionError(f"unexpected parent feature shape: {tuple(f_features.shape)}")
    if tuple(g_features.shape[:2]) != (2, 1):
        raise AssertionError(f"unexpected conditional feature shape: {tuple(g_features.shape)}")
    payload = {
        "status": "PASS",
        "views_shape": list(views.shape),
        "parent_shape": list(parents.shape),
        "f_shape": list(f_features.shape),
        "g_shape": list(g_features.shape),
        "representation_shape": list(representations.shape),
        "encoded_images_per_parent": 2,
    }
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"])
    destination = run_dir / "artifacts" / "raw_parent_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
