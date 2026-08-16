"""CIFAR data assembly for the structure gate: nested view trees per split.

Splits mirror CIFARDataModule: a calibration split (Stage B), a validation
split, and the SSL training split, all cut from the training set with the
run seed; the test set stays untouched.  Calibration/val/test trees are
deterministic per index so Stage-B coordinates and reported certificates
are reproducible.
"""

from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset

from ..data.cifar import CIFARFiles
from ..data.view_tree import (
    CropLevelSpec,
    MaskLevelSpec,
    NestedViewTreeDataset,
    ViewTreeConfig,
)


def view_tree_config_from_dict(config: Dict[str, Any]) -> ViewTreeConfig:
    root = config.get("root", {})
    edges = []
    for edge in config["edges"]:
        kind = str(edge.get("kind", "crop"))
        if kind == "crop":
            edges.append(
                CropLevelSpec(
                    min_scale=float(edge.get("min_scale", 0.3)),
                    max_scale=float(edge.get("max_scale", 0.8)),
                    size=int(edge.get("size", 32)),
                    noise_std=float(edge.get("noise_std", 0.0)),
                )
            )
        elif kind == "mask":
            edges.append(
                MaskLevelSpec(
                    patch_fraction=float(edge.get("patch_fraction", 0.35)),
                    patches=int(edge.get("patches", 1)),
                    noise_std=float(edge.get("noise_std", 0.0)),
                )
            )
        else:
            raise ValueError(f"unknown edge kind {kind!r}")
    return ViewTreeConfig(
        root_spec=CropLevelSpec(
            min_scale=float(root.get("min_scale", 0.5)),
            max_scale=float(root.get("max_scale", 1.0)),
            size=int(root.get("size", 32)),
            noise_std=float(root.get("noise_std", 0.0)),
        ),
        edge_specs=edges,
        children_per_edge=int(config.get("children_per_edge", 4)),
        endpoint_descendants=int(config.get("endpoint_descendants", 1)),
        flip_probability=float(config.get("flip_probability", 0.5)),
        mode=str(config.get("mode", "nested")),
    )


class GateDataModule:
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, NestedViewTreeDataset] = {}

    def setup(self, stage: str = "") -> None:
        data = self.config
        base_train = CIFARFiles(str(data["root"]), str(data["dataset"]), train=True)
        base_test = CIFARFiles(str(data["root"]), str(data["dataset"]), train=False)
        calibration_size = int(data["n_calibration"])
        validation_size = int(data["n_val"])
        if calibration_size + validation_size >= len(base_train):
            raise ValueError("calibration and validation splits leave no training samples")
        permutation = torch.randperm(
            len(base_train), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        calibration_indices = permutation[:calibration_size]
        validation_indices = permutation[calibration_size:calibration_size + validation_size]
        training_indices = permutation[calibration_size + validation_size:]
        tree_config = view_tree_config_from_dict(data["view_tree"])
        self.datasets = {
            "train": NestedViewTreeDataset(Subset(base_train, training_indices), tree_config),
            "calibration": NestedViewTreeDataset(
                Subset(base_train, calibration_indices), tree_config, self.seed + 100000
            ),
            "val": NestedViewTreeDataset(
                Subset(base_train, validation_indices), tree_config, self.seed + 200000
            ),
            "test": NestedViewTreeDataset(base_test, tree_config, self.seed + 300000),
        }

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.config.get("num_workers", 4))
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.config["batch_size"]),
            shuffle=shuffle,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
            drop_last=split == "train",
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def calibration_dataloader(self) -> DataLoader:
        return self._loader("calibration", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)
