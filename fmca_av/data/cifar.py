"""CIFAR readers and multi-view augmentation for FMCA-AV SSL."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


class CIFARFiles(Dataset):
    def __init__(self, root: str, dataset: str, train: bool) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.train = train
        self.images, self.labels = self._read()

    @staticmethod
    def _unpickle(path: Path) -> Dict[object, object]:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1")

    def _read(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.dataset == "cifar10":
            names = [f"data_batch_{index}" for index in range(1, 6)] if self.train else ["test_batch"]
            label_key = "labels"
        elif self.dataset == "cifar100":
            names = ["train"] if self.train else ["test"]
            label_key = "fine_labels"
        else:
            raise ValueError("dataset must be cifar10 or cifar100")
        images: List[np.ndarray] = []
        labels: List[int] = []
        for name in names:
            path = self.root / name
            if not path.is_file():
                raise FileNotFoundError(f"missing extracted CIFAR file: {path}")
            payload = self._unpickle(path)
            data = payload.get("data", payload.get(b"data"))
            batch_labels = payload.get(label_key, payload.get(label_key.encode()))
            if data is None or batch_labels is None:
                raise ValueError(f"unrecognized CIFAR file structure: {path}")
            images.append(np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32))
            labels.extend(int(value) for value in batch_labels)
        return np.concatenate(images), np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.images[index], int(self.labels[index])


def _random_uniform(generator: Optional[torch.Generator], low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


class CIFARViewTransform:
    def __init__(self, config: Dict[str, object]) -> None:
        self.size = int(config.get("size", 32))
        self.min_scale = float(config.get("min_scale", 0.08))
        self.color_jitter_probability = float(config.get("color_jitter_probability", 0.8))
        self.color_jitter_strength = float(config.get("color_jitter_strength", 0.5))
        self.grayscale_probability = float(config.get("grayscale_probability", 0.2))
        self.horizontal_flip_probability = float(config.get("horizontal_flip_probability", 0.5))
        self.rotation_degrees = float(config.get("rotation_degrees", 0.0))
        self.gaussian_blur_probability = float(config.get("gaussian_blur_probability", 0.0))
        self.gaussian_blur_sigma = float(config.get("gaussian_blur_sigma", 1.0))
        self.additive_noise_std = float(config.get("additive_noise_std", 0.0))
        self.mean = torch.tensor(config.get("mean", [0.4914, 0.4822, 0.4465])).view(3, 1, 1)
        self.std = torch.tensor(config.get("std", [0.2470, 0.2435, 0.2616])).view(3, 1, 1)

    def _crop(self, image: Image.Image, generator: Optional[torch.Generator]) -> Image.Image:
        width, height = image.size
        area = width * height
        for _ in range(10):
            target = area * _random_uniform(generator, self.min_scale, 1.0)
            aspect = float(torch.exp(torch.empty(()).uniform_(np.log(0.75), np.log(4 / 3), generator=generator)))
            crop_width = int(round((target * aspect) ** 0.5))
            crop_height = int(round((target / aspect) ** 0.5))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
                top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
                return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                    (self.size, self.size), Image.Resampling.BICUBIC
                )
        return image.resize((self.size, self.size), Image.Resampling.BICUBIC)

    def _color(self, image: Image.Image, generator: Optional[torch.Generator]) -> Image.Image:
        strength = self.color_jitter_strength
        operations = [
            lambda value: ImageEnhance.Brightness(value).enhance(_random_uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)),
            lambda value: ImageEnhance.Contrast(value).enhance(_random_uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)),
            lambda value: ImageEnhance.Color(value).enhance(_random_uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)),
        ]
        for index in torch.randperm(len(operations), generator=generator).tolist():
            image = operations[index](image)
        hue_shift = int(round(_random_uniform(generator, -0.2 * strength, 0.2 * strength) * 255))
        hsv = np.asarray(image.convert("HSV")).copy()
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 256
        return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

    def __call__(self, channels_first: np.ndarray, generator: Optional[torch.Generator] = None) -> Tensor:
        image = Image.fromarray(np.transpose(channels_first, (1, 2, 0)), mode="RGB")
        image = self._crop(image, generator)
        if float(torch.rand((), generator=generator)) < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)
        if self.rotation_degrees > 0:
            angle = _random_uniform(generator, -self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        if float(torch.rand((), generator=generator)) < self.color_jitter_probability:
            image = self._color(image, generator)
        if float(torch.rand((), generator=generator)) < self.grayscale_probability:
            image = ImageOps.grayscale(image).convert("RGB")
        if float(torch.rand((), generator=generator)) < self.gaussian_blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.gaussian_blur_sigma))
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        if self.additive_noise_std > 0:
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            tensor = (tensor + self.additive_noise_std * noise).clamp(0.0, 1.0)
        return (tensor - self.mean) / self.std


class MultiViewDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: CIFARViewTransform,
        num_views: int,
        deterministic_seed: Optional[int] = None,
        parent_transform: Optional["CIFARProbeTransform"] = None,
    ) -> None:
        self.base = base
        self.transform = transform
        self.num_views = num_views
        self.deterministic_seed = deterministic_seed
        self.parent_transform = parent_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        generator = None
        if self.deterministic_seed is not None:
            generator = torch.Generator().manual_seed(self.deterministic_seed + index)
        views = torch.stack([self.transform(image, generator) for _ in range(self.num_views)])
        if self.parent_transform is not None:
            return views, label, index, self.parent_transform(image)
        return views, label, index


class CIFARProbeTransform:
    def __init__(self, train: bool, mean: Sequence[float], std: Sequence[float], size: int = 32) -> None:
        self.train = train
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, channels_first: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(channels_first.astype(np.float32)) / 255.0
        if self.train:
            tensor = torch.nn.functional.pad(tensor, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, ()))
            left = int(torch.randint(0, 9, ()))
            tensor = tensor[:, top:top + 32, left:left + 32]
            if float(torch.rand(())) < 0.5:
                tensor = tensor.flip(-1)
        if self.size != 32:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(self.size, self.size), mode="bicubic", align_corners=False
            ).squeeze(0)
        return (tensor - self.mean) / self.std


class LabeledCIFARDataset(Dataset):
    def __init__(self, base: Dataset, transform: CIFARProbeTransform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, label = self.base[index]
        return self.transform(image), label


class CIFARCorruptionDataset(Dataset):
    def __init__(
        self,
        corruption_file: str,
        labels_file: str,
        severity: int,
        transform: CIFARProbeTransform,
    ) -> None:
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("severity must be in 1..5")
        images = np.load(corruption_file, mmap_mode="r")
        labels = np.load(labels_file, mmap_mode="r")
        start = (severity - 1) * 10000
        end = severity * 10000
        if images.shape[0] < end or labels.shape[0] < end:
            raise ValueError("CIFAR-C arrays do not contain five 10k severity blocks")
        self.images = images[start:end]
        self.labels = labels[start:end]
        self.transform = transform

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image = np.transpose(np.asarray(self.images[index]), (2, 0, 1))
        return self.transform(image), int(self.labels[index])


class CIFARDataModule:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self, stage: str = "") -> None:
        dataset = str(self.config["dataset"])
        base_train = CIFARFiles(str(self.config["root"]), dataset, train=True)
        base_test = CIFARFiles(str(self.config["root"]), dataset, train=False)
        calibration_size = int(self.config["n_calibration"])
        validation_size = int(self.config["n_val"])
        if calibration_size + validation_size >= len(base_train):
            raise ValueError("calibration and validation splits leave no training samples")
        permutation = torch.randperm(len(base_train), generator=torch.Generator().manual_seed(self.seed)).tolist()
        calibration_indices = permutation[:calibration_size]
        validation_indices = permutation[calibration_size:calibration_size + validation_size]
        training_indices = permutation[calibration_size + validation_size:]
        augmentation = self.config.get("augmentation", {})
        transform = CIFARViewTransform(augmentation)
        parent_transform = (
            CIFARProbeTransform(
                False,
                augmentation.get("mean", [0.4914, 0.4822, 0.4465]),
                augmentation.get("std", [0.2470, 0.2435, 0.2616]),
                int(augmentation.get("size", 32)),
            )
            if bool(self.config.get("include_raw_parent", False)) else None
        )
        views = int(self.config["num_views"])
        self.datasets = {
            "train": MultiViewDataset(
                Subset(base_train, training_indices), transform, views,
                parent_transform=parent_transform,
            ),
            "calibration": MultiViewDataset(
                Subset(base_train, calibration_indices), transform, views,
                self.seed + 100000, parent_transform,
            ),
            "val": MultiViewDataset(
                Subset(base_train, validation_indices), transform, views,
                self.seed + 200000, parent_transform,
            ),
            "test": MultiViewDataset(
                base_test, transform, views, self.seed + 300000, parent_transform,
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


class CIFARProbeDataModule:
    def __init__(self, config: Dict[str, object], probe: Dict[str, object], seed: int) -> None:
        self.config = config
        self.probe = probe
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self) -> None:
        dataset = str(self.config["dataset"])
        train_base = CIFARFiles(str(self.config["root"]), dataset, train=True)
        test_base = CIFARFiles(str(self.config["root"]), dataset, train=False)
        validation_size = int(self.probe.get("n_val", 5000))
        permutation = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(self.seed)).tolist()
        validation_indices = permutation[:validation_size]
        training_indices = permutation[validation_size:]
        label_fraction = float(self.probe.get("label_fraction", 1.0))
        if not 0 < label_fraction <= 1:
            raise ValueError("probe.label_fraction must be in (0, 1]")
        training_indices = training_indices[:max(1, int(round(label_fraction * len(training_indices))))]
        augmentation = self.config.get("augmentation", {})
        mean = augmentation.get("mean", [0.4914, 0.4822, 0.4465])
        std = augmentation.get("std", [0.2470, 0.2435, 0.2616])
        size = int(augmentation.get("size", 32))
        train_transform = CIFARProbeTransform(True, mean, std, size)
        test_transform = CIFARProbeTransform(False, mean, std, size)
        self.datasets = {
            "train": LabeledCIFARDataset(Subset(train_base, training_indices), train_transform),
            "val": LabeledCIFARDataset(Subset(train_base, validation_indices), test_transform),
            "test": LabeledCIFARDataset(test_base, test_transform),
        }

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.probe.get("num_workers", self.config.get("num_workers", 4)))
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.probe.get("batch_size", 256)),
            shuffle=shuffle,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)
