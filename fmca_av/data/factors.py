"""Memory-efficient readers and Lightning-style data module for factor datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .imagenet import ImageNetProbeTransform, ImageNetViewTransform, MultiViewImageDataset


FACTOR_CARDINALITIES = {
    "dsprites": (1, 3, 6, 40, 32, 32),
    "shapes3d": (10, 10, 10, 8, 4, 15),
    "mpi3d_toy": (6, 6, 2, 3, 3, 40, 40),
    "mpi3d_realistic": (6, 6, 2, 3, 3, 40, 40),
    "mpi3d_real": (6, 6, 2, 3, 3, 40, 40),
    "smallnorb": (5, 10, 9, 18, 6),
}

FACTOR_NAMES = {
    "dsprites": ("color", "shape", "scale", "orientation", "position_x", "position_y"),
    "shapes3d": ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"),
    "mpi3d_toy": ("object_color", "object_shape", "object_size", "camera_height", "background_color", "horizontal", "vertical"),
    "mpi3d_realistic": ("object_color", "object_shape", "object_size", "camera_height", "background_color", "horizontal", "vertical"),
    "mpi3d_real": ("object_color", "object_shape", "object_size", "camera_height", "background_color", "horizontal", "vertical"),
    "smallnorb": ("category", "instance", "elevation", "azimuth", "lighting"),
}


def _rgb(image: np.ndarray) -> Image.Image:
    value = np.asarray(image, dtype=np.uint8)
    if value.ndim == 2:
        value = np.repeat(value[:, :, None], 3, axis=2)
    return Image.fromarray(value, mode="RGB")


class DSprites(Dataset):
    def __init__(self, root: Path) -> None:
        prepared = root / "prepared" / "dsprites"
        self.images = np.load(prepared / "imgs.npy", mmap_mode="r")
        self.labels = np.load(prepared / "latents_classes.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Tensor]:
        return _rgb(self.images[index] * np.uint8(255)), torch.from_numpy(np.asarray(self.labels[index]).astype(np.int64))


class Shapes3D(Dataset):
    def __init__(self, root: Path) -> None:
        self.path = root / "3dshapes" / "3dshapes.h5"
        prepared = root / "prepared" / "shapes3d"
        complete = prepared / ".complete"
        images = prepared / "images.npy"
        labels = prepared / "labels.npy"
        if complete.is_file() and images.is_file() and labels.is_file():
            # The uncompressed, memory-mapped copy is produced atomically by
            # scripts/prepare_shapes3d_memmap.py.  It preserves the official
            # HDF5 ordering while avoiding very slow random compressed reads.
            self.images = np.load(images, mmap_mode="r")
            self.labels = np.load(labels, mmap_mode="r")
        else:
            self.images = None
            self.labels = None
        self._handle = None

    def _open(self):
        if self._handle is None:
            import h5py
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __len__(self) -> int:
        return len(self.images) if self.images is not None else 480_000

    @staticmethod
    def classes(values: np.ndarray) -> np.ndarray:
        result = np.empty(6, dtype=np.int64)
        result[:3] = np.rint(values[:3] * 10).astype(np.int64)
        result[3] = int(round((float(values[3]) - 0.75) * 14))
        result[4] = int(round(float(values[4])))
        result[5] = int(round((float(values[5]) + 30.0) * 14.0 / 60.0))
        return result

    def __getitem__(self, index: int) -> Tuple[Image.Image, Tensor]:
        if self.images is not None and self.labels is not None:
            image = np.asarray(self.images[index])
            labels = self.classes(np.asarray(self.labels[index]))
        else:
            handle = self._open()
            image = np.asarray(handle["images"][index])
            labels = self.classes(np.asarray(handle["labels"][index]))
        return _rgb(image), torch.from_numpy(labels)


def _unravel(index: int, cardinalities: Sequence[int]) -> np.ndarray:
    labels = np.empty(len(cardinalities), dtype=np.int64)
    remaining = index
    for position in range(len(cardinalities) - 1, -1, -1):
        labels[position] = remaining % cardinalities[position]
        remaining //= cardinalities[position]
    return labels


class MPI3D(Dataset):
    def __init__(self, root: Path, variant: str) -> None:
        self.variant = variant
        self.images = np.load(root / "prepared" / variant / "images.npy", mmap_mode="r")
        expected = int(np.prod(FACTOR_CARDINALITIES[variant]))
        if len(self.images) != expected:
            raise ValueError(f"{variant} has {len(self.images)} images, expected {expected}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Tensor]:
        labels = _unravel(index, FACTOR_CARDINALITIES[self.variant])
        return _rgb(self.images[index]), torch.from_numpy(labels)


def _norb_memmap(path: Path, dtype: np.dtype, shape: Tuple[int, ...]) -> np.memmap:
    header_bytes = 8 + 4 * max(len(shape), 3)
    return np.memmap(path, mode="r", dtype=dtype, offset=header_bytes, shape=shape)


class SmallNORB(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        if split == "train":
            prefix = "smallnorb-5x46789x9x18x6x2x96x96-training"
        elif split == "test":
            prefix = "smallnorb-5x01235x9x18x6x2x96x96-testing"
        else:
            raise ValueError("SmallNORB split must be train or test")
        directory = root / "smallnorb"
        self.images = _norb_memmap(directory / f"{prefix}-dat.mat", np.dtype("u1"), (24_300, 2, 96, 96))
        self.categories = _norb_memmap(directory / f"{prefix}-cat.mat", np.dtype("<i4"), (24_300,))
        self.info = _norb_memmap(directory / f"{prefix}-info.mat", np.dtype("<i4"), (24_300, 4))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Tensor]:
        info = np.asarray(self.info[index], dtype=np.int64).copy()
        info[2] //= 2  # The official azimuth field is stored as 0,2,...,34.
        labels = np.concatenate(([int(self.categories[index])], info))
        return _rgb(self.images[index, 0]), torch.from_numpy(labels)


def factor_dataset(root: str, dataset: str, split: str = "all") -> Dataset:
    path = Path(root)
    if dataset == "dsprites":
        return DSprites(path)
    if dataset == "shapes3d":
        return Shapes3D(path)
    if dataset.startswith("mpi3d_"):
        return MPI3D(path, dataset)
    if dataset == "smallnorb":
        return SmallNORB(path, "test" if split == "test" else "train")
    raise ValueError(f"unsupported factor dataset: {dataset}")


class FactorDataModule:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self, stage: str = "") -> None:
        name = str(self.config["dataset"])
        train_base = factor_dataset(str(self.config["root"]), name, "train")
        test_base = factor_dataset(str(self.config["root"]), name, "test") if name == "smallnorb" else train_base
        calibration_size = int(self.config["n_calibration"])
        validation_size = int(self.config["n_val"])
        test_size = int(self.config.get("n_test", min(10_000, len(test_base))))
        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(len(train_base), generator=generator)
        if name == "smallnorb":
            test_indices = torch.randperm(len(test_base), generator=generator)[:test_size].tolist()
        else:
            test_indices = permutation[:test_size].tolist()
            permutation = permutation[test_size:]
        if calibration_size + validation_size >= len(permutation):
            raise ValueError("factor splits leave no training samples")
        calibration = permutation[:calibration_size].tolist()
        validation = permutation[calibration_size:calibration_size + validation_size].tolist()
        training = permutation[calibration_size + validation_size:].tolist()
        augmentation = self.config.get("augmentation", {})
        transform = ImageNetViewTransform(augmentation)
        parent_transform = (
            ImageNetProbeTransform(False, augmentation)
            if bool(self.config.get("include_raw_parent", False)) else None
        )
        views = int(self.config["num_views"])
        self.datasets = {
            "train": MultiViewImageDataset(
                Subset(train_base, training), transform, views, parent_transform=parent_transform,
            ),
            "calibration": MultiViewImageDataset(
                Subset(train_base, calibration), transform, views, self.seed + 100_000,
                parent_transform,
            ),
            "val": MultiViewImageDataset(
                Subset(train_base, validation), transform, views, self.seed + 200_000,
                parent_transform,
            ),
            "test": MultiViewImageDataset(
                Subset(test_base, test_indices), transform, views, self.seed + 300_000,
                parent_transform,
            ),
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
