"""Finite-alphabet channels for exact estimator validation."""

from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def normalized_joint(config: Dict[str, object]) -> Tensor:
    joint = torch.tensor(config["joint_probability"], dtype=torch.float64)
    if joint.ndim != 2 or torch.any(joint < 0) or float(joint.sum()) <= 0:
        raise ValueError("joint_probability must be a nonnegative matrix with positive mass")
    return joint / joint.sum()


def sample_finite_conditionals(x: Tensor, conditional: Tensor, num_views: int) -> Tensor:
    probabilities = conditional.to(device=x.device, dtype=torch.float32)[x]
    return torch.multinomial(probabilities, num_samples=num_views, replacement=True)


class FiniteDataModule:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, TensorDataset] = {}
        joint = normalized_joint(config)
        self.p_x = joint.sum(dim=1).float()
        self.conditional = (joint / joint.sum(dim=1, keepdim=True)).float()

    def setup(self, stage: str = "") -> None:
        sizes = {
            "train": int(self.config["n_train"]),
            "calibration": int(self.config["n_calibration"]),
            "val": int(self.config["n_val"]),
            "test": int(self.config["n_test"]),
        }
        for offset, (split, size) in enumerate(sizes.items()):
            generator = torch.Generator().manual_seed(self.seed + offset)
            x = torch.multinomial(self.p_x, size, replacement=True, generator=generator)
            self.datasets[split] = TensorDataset(x)

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.config.get("num_workers", 0))
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.config["batch_size"]),
            shuffle=shuffle,
            num_workers=workers,
            persistent_workers=workers > 0,
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

