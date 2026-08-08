"""ImageNet-1K readers and Lightning data modules for FMCA-AV."""

import csv
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def _uniform(generator: Optional[torch.Generator], low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _image_tensor(image: Image.Image, mean: Tensor, std: Tensor) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    return (tensor - mean) / std


def _resize_short_side(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    scale = target / min(width, height)
    output = (int(round(width * scale)), int(round(height * scale)))
    return image.resize(output, Image.Resampling.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


class ImageNetFiles(Dataset):
    """ILSVRC CLS-LOC train folders and flat validation images."""

    def __init__(
        self,
        root: str,
        split: str,
        val_labels: str = "",
        class_wnids: Optional[Sequence[str]] = None,
        manifest: str = "",
    ) -> None:
        self.root = Path(root)
        data_root = self.root / "Data" / "CLS-LOC"
        if not data_root.is_dir():
            data_root = self.root
        self.data_root = data_root
        train_root = data_root / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(f"ImageNet training directory is unavailable: {train_root}")
        available = sorted(path.name for path in train_root.iterdir() if path.is_dir())
        selected = set(str(item) for item in class_wnids) if class_wnids else set(available)
        missing = sorted(selected - set(available))
        if missing:
            raise ValueError(f"requested ImageNet wnids are unavailable: {missing[:5]}")
        self.wnids = [wnid for wnid in available if wnid in selected]
        self.class_to_index = {wnid: index for index, wnid in enumerate(self.wnids)}
        if not self.wnids:
            raise ValueError("ImageNet class selection is empty")
        manifest_path = Path(manifest) if manifest else None
        if split == "train":
            self.samples = self._training_samples(train_root, manifest_path)
        elif split in {"val", "validation"}:
            self.samples = self._validation_samples(
                data_root / "val", Path(val_labels), manifest_path,
            )
        else:
            raise ValueError("ImageNetFiles split must be train or val")

    def _manifest_samples(self, manifest: Path) -> List[Tuple[Path, int]]:
        if not manifest.is_file():
            raise FileNotFoundError(f"ImageNet manifest is unavailable: {manifest}")
        samples: List[Tuple[Path, int]] = []
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or reader.fieldnames != ["path", "wnid"]:
                raise ValueError(
                    f"ImageNet manifest must have tab-separated path and wnid columns: {manifest}"
                )
            for row in reader:
                wnid = str(row["wnid"])
                if wnid not in self.class_to_index:
                    continue
                relative = Path(str(row["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"ImageNet manifest contains an unsafe path: {relative}")
                samples.append((self.data_root / relative, self.class_to_index[wnid]))
        if not samples:
            raise ValueError(f"ImageNet manifest produced no samples for selected classes: {manifest}")
        return samples

    def _training_samples(
        self, train_root: Path, manifest: Optional[Path],
    ) -> List[Tuple[Path, int]]:
        if manifest is not None:
            return self._manifest_samples(manifest)
        samples: List[Tuple[Path, int]] = []
        for wnid in self.wnids:
            label = self.class_to_index[wnid]
            paths = sorted(
                path for path in (train_root / wnid).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            samples.extend((path, label) for path in paths)
        if not samples:
            raise FileNotFoundError(f"no ImageNet training images found under {train_root}")
        return samples

    def _validation_samples(
        self, val_root: Path, labels_path: Path, manifest: Optional[Path],
    ) -> List[Tuple[Path, int]]:
        if manifest is not None:
            return self._manifest_samples(manifest)
        if not val_root.is_dir():
            raise FileNotFoundError(f"ImageNet validation directory is unavailable: {val_root}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"ImageNet validation label CSV is unavailable: {labels_path}")
        assignments: Dict[str, str] = {}
        with labels_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "ImageId" not in reader.fieldnames or "PredictionString" not in reader.fieldnames:
                raise ValueError("validation CSV must contain ImageId and PredictionString")
            for row in reader:
                prediction = str(row["PredictionString"]).split()
                if prediction:
                    assignments[str(row["ImageId"])] = prediction[0]
        samples = []
        for image_id, wnid in assignments.items():
            if wnid not in self.class_to_index:
                continue
            path = val_root / f"{image_id}.JPEG"
            if not path.is_file():
                raise FileNotFoundError(f"validation image listed in CSV is unavailable: {path}")
            samples.append((path, self.class_to_index[wnid]))
        samples.sort(key=lambda item: item[0].name)
        if not samples:
            raise ValueError("validation CSV produced no samples for the selected classes")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), label


class ImageNetViewTransform:
    def __init__(self, config: Dict[str, object]) -> None:
        self.size = int(config.get("size", 224))
        self.min_scale = float(config.get("min_scale", 0.08))
        self.max_scale = float(config.get("max_scale", 1.0))
        self.color_probability = float(config.get("color_jitter_probability", 0.8))
        self.color_strength = float(config.get("color_jitter_strength", 0.5))
        self.grayscale_probability = float(config.get("grayscale_probability", 0.2))
        self.blur_probability = float(config.get("gaussian_blur_probability", 0.5))
        self.blur_sigma = float(config.get("gaussian_blur_sigma", 0.0))
        self.flip_probability = float(config.get("horizontal_flip_probability", 0.5))
        self.rotation_degrees = float(config.get("rotation_degrees", 0.0))
        self.additive_noise_std = float(config.get("additive_noise_std", 0.0))
        self.mean = torch.tensor(config.get("mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
        self.std = torch.tensor(config.get("std", [0.229, 0.224, 0.225])).view(3, 1, 1)

    def _crop(self, image: Image.Image, generator: Optional[torch.Generator]) -> Image.Image:
        width, height = image.size
        area = width * height
        for _ in range(10):
            target = area * _uniform(generator, self.min_scale, self.max_scale)
            aspect = math.exp(_uniform(generator, math.log(0.75), math.log(4.0 / 3.0)))
            crop_width = int(round(math.sqrt(target * aspect)))
            crop_height = int(round(math.sqrt(target / aspect)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
                top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
                return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                    (self.size, self.size), Image.Resampling.BICUBIC
                )
        return _center_crop(_resize_short_side(image, self.size), self.size)

    def _color(self, image: Image.Image, generator: Optional[torch.Generator]) -> Image.Image:
        strength = self.color_strength
        operations = [
            lambda value: ImageEnhance.Brightness(value).enhance(
                _uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)
            ),
            lambda value: ImageEnhance.Contrast(value).enhance(
                _uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)
            ),
            lambda value: ImageEnhance.Color(value).enhance(
                _uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)
            ),
        ]
        for index in torch.randperm(len(operations), generator=generator).tolist():
            image = operations[index](image)
        return image

    def __call__(self, image: Image.Image, generator: Optional[torch.Generator] = None) -> Tensor:
        image = self._crop(image, generator)
        if float(torch.rand((), generator=generator)) < self.flip_probability:
            image = ImageOps.mirror(image)
        if self.rotation_degrees > 0:
            angle = _uniform(generator, -self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        if float(torch.rand((), generator=generator)) < self.color_probability:
            image = self._color(image, generator)
        if float(torch.rand((), generator=generator)) < self.grayscale_probability:
            image = ImageOps.grayscale(image).convert("RGB")
        if float(torch.rand((), generator=generator)) < self.blur_probability:
            sigma = self.blur_sigma if self.blur_sigma > 0 else _uniform(generator, 0.1, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        result = _image_tensor(image, self.mean, self.std)
        if self.additive_noise_std > 0:
            pixels = result * self.std + self.mean
            noise = torch.randn(pixels.shape, generator=generator, dtype=pixels.dtype)
            pixels = (pixels + self.additive_noise_std * noise).clamp(0.0, 1.0)
            result = (pixels - self.mean) / self.std
        return result


class ImageNetProbeTransform:
    def __init__(self, train: bool, config: Dict[str, object]) -> None:
        self.train = train
        self.size = int(config.get("size", 224))
        self.resize_size = int(config.get("eval_resize", 256))
        self.mean = torch.tensor(config.get("mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
        self.std = torch.tensor(config.get("std", [0.229, 0.224, 0.225])).view(3, 1, 1)
        self.train_transform = ImageNetViewTransform(
            {
                **config,
                "size": self.size,
                "min_scale": float(config.get("probe_min_scale", 0.08)),
                "color_jitter_probability": 0.0,
                "grayscale_probability": 0.0,
                "gaussian_blur_probability": 0.0,
            }
        )

    def __call__(self, image: Image.Image) -> Tensor:
        if self.train:
            return self.train_transform(image)
        image = _center_crop(_resize_short_side(image, self.resize_size), self.size)
        return _image_tensor(image, self.mean, self.std)


class MultiViewImageDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: ImageNetViewTransform,
        num_views: int,
        deterministic_seed: Optional[int] = None,
        parent_transform: Optional[Callable[[Image.Image], Tensor]] = None,
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
        if isinstance(label, (int, np.integer)):
            prepared_label: object = int(label)
        else:
            prepared_label = torch.as_tensor(label)
        if self.parent_transform is not None:
            return views, prepared_label, index, self.parent_transform(image)
        return views, prepared_label, index


class LabeledImageDataset(Dataset):
    def __init__(self, base: Dataset, transform: ImageNetProbeTransform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image, label = self.base[index]
        return self.transform(image), int(label)


def _class_selection(config: Dict[str, object]) -> Optional[List[str]]:
    value = config.get("class_wnids")
    if value is None:
        path = str(config.get("class_wnids_file", ""))
        if not path:
            return None
        value = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError("class_wnids must be a list or class_wnids_file must contain one wnid per line")
    return [str(item) for item in value]


class ImageNetDataModule:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self, stage: str = "") -> None:
        classes = _class_selection(self.config)
        train_base = ImageNetFiles(
            str(self.config["root"]), "train", class_wnids=classes,
            manifest=str(self.config.get("train_manifest", "")),
        )
        val_base = ImageNetFiles(
            str(self.config["root"]), "val", str(self.config["val_labels"]),
            class_wnids=classes, manifest=str(self.config.get("val_manifest", "")),
        )
        calibration_size = int(self.config["n_calibration"])
        validation_size = int(self.config["n_val"])
        if calibration_size + validation_size >= len(train_base):
            raise ValueError("calibration and validation splits leave no ImageNet training samples")
        permutation = torch.randperm(
            len(train_base), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        calibration = permutation[:calibration_size]
        validation = permutation[calibration_size:calibration_size + validation_size]
        training = permutation[calibration_size + validation_size:]
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
                Subset(train_base, calibration), transform, views, self.seed + 10000000,
                parent_transform,
            ),
            "val": MultiViewImageDataset(
                Subset(train_base, validation), transform, views, self.seed + 20000000,
                parent_transform,
            ),
            "test": MultiViewImageDataset(
                val_base, transform, views, self.seed + 30000000, parent_transform,
            ),
        }

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.config.get("num_workers", 8))
        options = {
            "batch_size": int(self.config["batch_size"]),
            "shuffle": shuffle,
            "num_workers": workers,
            "persistent_workers": workers > 0,
            "pin_memory": True,
            "drop_last": split == "train",
        }
        if workers > 0:
            options["prefetch_factor"] = int(self.config.get("prefetch_factor", 2))
        return DataLoader(
            self.datasets[split],
            **options,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def calibration_dataloader(self) -> DataLoader:
        return self._loader("calibration", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)


class ImageNetProbeDataModule:
    def __init__(self, config: Dict[str, object], probe: Dict[str, object], seed: int) -> None:
        self.config = config
        self.probe = probe
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self) -> None:
        classes = _class_selection(self.config)
        train_base = ImageNetFiles(
            str(self.config["root"]), "train", class_wnids=classes,
            manifest=str(self.config.get("train_manifest", "")),
        )
        val_base = ImageNetFiles(
            str(self.config["root"]), "val", str(self.config["val_labels"]),
            class_wnids=classes, manifest=str(self.config.get("val_manifest", "")),
        )
        validation_size = int(self.probe.get("n_val", 50000))
        permutation = torch.randperm(
            len(train_base), generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        validation = permutation[:validation_size]
        training = permutation[validation_size:]
        label_fraction = float(self.probe.get("label_fraction", 1.0))
        if not 0 < label_fraction <= 1:
            raise ValueError("probe.label_fraction must be in (0, 1]")
        training = training[:max(1, int(round(label_fraction * len(training))))]
        augmentation = self.config.get("augmentation", {})
        train_transform = ImageNetProbeTransform(True, augmentation)
        eval_transform = ImageNetProbeTransform(False, augmentation)
        self.datasets = {
            "train": LabeledImageDataset(Subset(train_base, training), train_transform),
            "val": LabeledImageDataset(Subset(train_base, validation), eval_transform),
            "test": LabeledImageDataset(val_base, eval_transform),
        }

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.probe.get("num_workers", self.config.get("num_workers", 8)))
        options = {
            "batch_size": int(self.probe.get("batch_size", 256)),
            "shuffle": shuffle,
            "num_workers": workers,
            "persistent_workers": workers > 0,
            "pin_memory": True,
        }
        if workers > 0:
            options["prefetch_factor"] = int(
                self.probe.get("prefetch_factor", self.config.get("prefetch_factor", 2))
            )
        return DataLoader(
            self.datasets[split],
            **options,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", False)
