"""Synthetic one-dimensional Gaussian channel with independent conditionals."""

import math
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def gaussian_eigenvalues(noise_variance: float, count: int) -> Tensor:
    """Nonconstant squared canonical correlations for Y=X+sqrt(sigma)N."""

    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative")
    correlation_squared = 1.0 / (1.0 + noise_variance)
    powers = torch.arange(1, count + 1, dtype=torch.float64)
    return torch.tensor(correlation_squared, dtype=torch.float64).pow(powers)


def gaussian_product_eigenvalues(noise_variance: float, dimension: int, count: int) -> Tensor:
    """Leading nonconstant spectrum for an isotropic multivariate Gaussian channel."""

    if dimension < 1 or count < 1:
        raise ValueError("dimension and count must be positive")
    correlation_squared = 1.0 / (1.0 + noise_variance)
    values = []
    degree = 1
    while len(values) < count:
        # Number of multi-indices of total degree `degree` in `dimension` variables.
        multiplicity = math.comb(degree + dimension - 1, dimension - 1)
        values.extend([correlation_squared ** degree] * multiplicity)
        degree += 1
    return torch.tensor(values[:count], dtype=torch.float64)


def sample_conditionals(x: Tensor, num_views: int, noise_variance: float) -> Tensor:
    if num_views < 1:
        raise ValueError("num_views must be positive")
    noise = torch.randn(x.shape[0], num_views, x.shape[1], device=x.device, dtype=x.dtype)
    return x.unsqueeze(1) + noise_variance ** 0.5 * noise


class GaussianDataModule:
    """Minimal Lightning-compatible data module without importing Lightning."""

    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, TensorDataset] = {}

    def setup(self, stage: str = "") -> None:
        sizes = {
            "train": int(self.config["n_train"]),
            "calibration": int(self.config["n_calibration"]),
            "val": int(self.config["n_val"]),
            "test": int(self.config["n_test"]),
        }
        for offset, (split, size) in enumerate(sizes.items()):
            generator = torch.Generator().manual_seed(self.seed + offset)
            dimension = int(self.config.get("dimension", 1))
            self.datasets[split] = TensorDataset(torch.randn(size, dimension, generator=generator))

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.config["batch_size"]),
            shuffle=shuffle,
            num_workers=int(self.config.get("num_workers", 0)),
            persistent_workers=int(self.config.get("num_workers", 0)) > 0,
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
