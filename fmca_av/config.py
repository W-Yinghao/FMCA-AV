"""Configuration loading and validation."""

import json
import os
from pathlib import Path
from typing import Any, Dict


def _merge(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if os.environ.get("FMCA_SEED_OVERRIDE"):
        config["seed"] = int(os.environ["FMCA_SEED_OVERRIDE"])
    if os.environ.get("FMCA_CONFIG_OVERRIDES"):
        overrides = json.loads(os.environ["FMCA_CONFIG_OVERRIDES"])
        if not isinstance(overrides, dict):
            raise ValueError("FMCA_CONFIG_OVERRIDES must be a JSON object")
        _merge(config, overrides)
    required = {"experiment", "seed", "data", "model", "objective", "optimizer", "trainer"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("missing configuration sections: " + ", ".join(missing))
    family = config["experiment"].get("family")
    if family not in {"gaussian_1d", "finite_channel", "cifar_ssl", "vision_ssl"}:
        raise ValueError(
            "experiment.family must be gaussian_1d, finite_channel, cifar_ssl, or vision_ssl"
        )
    if int(config["data"]["num_views"]) < 1:
        raise ValueError("data.num_views must be positive")
    if (bool(config["data"].get("include_raw_parent", False)) and
            str(config["model"].get("parent_aggregation", "mean")) != "raw"):
        raise ValueError("data.include_raw_parent=true requires model.parent_aggregation=raw")
    if not bool(config["model"].get("centered", True)):
        raise ValueError("reference experiments require centered=true to remove the constant mode")
    if config["objective"]["name"] not in {"trace", "logdet"}:
        raise ValueError("objective.name must be trace or logdet")
    vision_datasets = {
        "cifar10", "cifar100", "stl10", "tinyimagenet200", "imagenet1k", "imagenet100",
        "dsprites", "shapes3d", "smallnorb", "mpi3d_toy", "mpi3d_realistic", "mpi3d_real",
    }
    if family == "cifar_ssl" and config["data"].get("dataset") not in {"cifar10", "cifar100"}:
        raise ValueError("cifar_ssl requires data.dataset cifar10 or cifar100")
    if family == "vision_ssl" and config["data"].get("dataset") not in vision_datasets:
        raise ValueError("vision_ssl data.dataset is unsupported")
    if config["data"].get("dataset") in {"imagenet1k", "imagenet100"} and not config["data"].get("val_labels"):
        raise ValueError("ImageNet configurations require data.val_labels")
    config["_config_path"] = str(config_path)
    return config
