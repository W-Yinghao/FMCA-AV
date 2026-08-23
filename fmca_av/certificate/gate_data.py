"""Data assembly for the structure gate: nested view trees per split.

Splits mirror CIFARDataModule: a calibration split (Stage B), a validation
split, and the SSL training split, all cut from the training set with the
run seed; the test set stays untouched.  Calibration/val/test trees are
deterministic per index so Stage-B coordinates and reported certificates
are reproducible.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.cifar import CIFARFiles
from ..data.small_vision import TinyImageNetFiles
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
                    color_jitter_probability=float(edge.get("color_jitter_probability", 0.0)),
                    color_jitter_strength=float(edge.get("color_jitter_strength", 0.5)),
                    color_jitter_hue=bool(edge.get("color_jitter_hue", False)),
                    grayscale_probability=float(edge.get("grayscale_probability", 0.0)),
                    flip_probability=float(edge.get("flip_probability", 0.0)),
                )
            )
        elif kind == "mask":
            edges.append(
                MaskLevelSpec(
                    patch_fraction=float(edge.get("patch_fraction", 0.35)),
                    patches=int(edge.get("patches", 1)),
                    noise_std=float(edge.get("noise_std", 0.0)),
                    grayscale_probability=float(edge.get("grayscale_probability", 0.0)),
                    blur_probability=float(edge.get("blur_probability", 0.0)),
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


class ArrayImageFiles(Dataset):
    """Adapts a PIL-returning file dataset to CIFARFiles' array contract.

    The view tree wants channels-first uint8 arrays; Tiny ImageNet ships
    PIL images.  Emitting the same layout CIFARFiles emits means the
    tree, the probe transform and the normalization path downstream stay
    byte-for-byte the same code.
    """

    def __init__(self, base: Dataset, size: int) -> None:
        self.base = base
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        from PIL import Image

        image, label = self.base[index]
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.uint8)
        return np.ascontiguousarray(array.transpose(2, 0, 1)), int(label)


def gate_base_datasets(data: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """Train/test base datasets for whichever dataset the config names."""

    dataset, root = str(data["dataset"]), str(data["root"])
    if dataset in {"cifar10", "cifar100"}:
        return CIFARFiles(root, dataset, train=True), CIFARFiles(root, dataset, train=False)
    if dataset == "tinyimagenet200":
        size = int(data.get("image_size", 64))
        return (ArrayImageFiles(TinyImageNetFiles(root, "train"), size),
                ArrayImageFiles(TinyImageNetFiles(root, "val"), size))
    raise ValueError(f"unsupported gate dataset {dataset!r}")


class GateProbeTransform:
    """Probe transform that respects the dataset's native resolution.

    CIFARProbeTransform hard-codes a 32-pixel crop, so it cannot serve a
    64-pixel dataset; this keeps the same recipe (reflect-pad 4, random
    crop back to native size, horizontal flip) at any size.
    """

    def __init__(self, train: bool, mean, std, size: int) -> None:
        self.train = train
        self.size = int(size)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, channels_first: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(channels_first.astype(np.float32)) / 255.0
        if self.train:
            tensor = torch.nn.functional.pad(tensor, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, ()))
            left = int(torch.randint(0, 9, ()))
            tensor = tensor[:, top:top + self.size, left:left + self.size]
            if float(torch.rand(())) < 0.5:
                tensor = tensor.flip(-1)
        return (tensor - self.mean) / self.std


class GateProbeDataModule:
    """Linear-probe splits for any gate dataset (mirrors CIFARProbeDataModule)."""

    def __init__(self, config: Dict[str, Any], probe: Dict[str, Any], seed: int) -> None:
        self.config, self.probe, self.seed = config, probe, seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self) -> None:
        from ..data.cifar import LabeledCIFARDataset

        train_base, test_base = gate_base_datasets(self.config)
        validation_size = int(self.probe.get("n_val", 5000))
        permutation = torch.randperm(
            len(train_base), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        validation_indices = permutation[:validation_size]
        training_indices = permutation[validation_size:]
        fraction = float(self.probe.get("label_fraction", 1.0))
        if not 0 < fraction <= 1:
            raise ValueError("probe.label_fraction must be in (0, 1]")
        training_indices = training_indices[:max(1, int(round(fraction * len(training_indices))))]
        augmentation = self.config.get("augmentation", {})
        mean = augmentation.get("mean", [0.4914, 0.4822, 0.4465])
        std = augmentation.get("std", [0.2470, 0.2435, 0.2616])
        size = int(self.config.get("image_size", augmentation.get("size", 32)))
        self.datasets = {
            "train": LabeledCIFARDataset(Subset(train_base, training_indices),
                                         GateProbeTransform(True, mean, std, size)),
            "val": LabeledCIFARDataset(Subset(train_base, validation_indices),
                                       GateProbeTransform(False, mean, std, size)),
            "test": LabeledCIFARDataset(test_base, GateProbeTransform(False, mean, std, size)),
        }

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.config.get("num_workers", 4))
        return DataLoader(self.datasets[split], batch_size=int(self.probe.get("batch_size", 256)),
                          shuffle=shuffle, num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)


class GateDataModule:
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, NestedViewTreeDataset] = {}

    def setup(self, stage: str = "") -> None:
        data = self.config
        base_train, base_test = gate_base_datasets(data)
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
