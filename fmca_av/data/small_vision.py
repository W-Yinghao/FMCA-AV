"""STL-10 and Tiny ImageNet data modules using the shared image pipeline."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .imagenet import (
    IMAGE_EXTENSIONS,
    ImageNetProbeTransform,
    ImageNetViewTransform,
    LabeledImageDataset,
    MultiViewImageDataset,
)


class STL10Binary(Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.root = Path(root)
        if split not in {"train", "test", "unlabeled"}:
            raise ValueError("STL10 split must be train, test, or unlabeled")
        image_path = self.root / f"{split}_X.bin"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing STL-10 image file: {image_path}")
        raw = np.memmap(image_path, mode="r", dtype=np.uint8)
        pixels = 3 * 96 * 96
        if raw.size % pixels:
            raise ValueError(f"invalid STL-10 image file size: {image_path}")
        self.images = raw.reshape(-1, 3, 96, 96)
        label_path = self.root / f"{split}_y.bin"
        if split == "unlabeled":
            self.labels = np.full(len(self.images), -1, dtype=np.int64)
        else:
            if not label_path.is_file():
                raise FileNotFoundError(f"missing STL-10 label file: {label_path}")
            labels = np.fromfile(label_path, dtype=np.uint8).astype(np.int64) - 1
            if len(labels) != len(self.images):
                raise ValueError(f"STL-10 images and labels differ in length: {split}")
            self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        # The official binary stores image width before height.
        array = np.asarray(self.images[index]).transpose(2, 1, 0).copy()
        return Image.fromarray(array, mode="RGB"), int(self.labels[index])


class TinyImageNetFiles(Dataset):
    def __init__(self, root: str, split: str, manifest: str = "") -> None:
        self.root = Path(root)
        if split not in {"train", "val", "validation"}:
            raise ValueError("TinyImageNetFiles split must be train or val")
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.is_file():
            raise FileNotFoundError(f"missing Tiny ImageNet wnids: {wnids_path}")
        self.wnids = [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_index = {wnid: index for index, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[Path, int]] = []
        manifest_path = Path(manifest) if manifest else None
        if manifest_path is not None:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Tiny ImageNet manifest is unavailable: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames != ["path", "wnid"]:
                    raise ValueError(
                        f"Tiny ImageNet manifest must have tab-separated path and wnid columns: {manifest_path}"
                    )
                for row in reader:
                    relative = Path(str(row["path"]))
                    wnid = str(row["wnid"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Tiny ImageNet manifest contains an unsafe path: {relative}")
                    if wnid in self.class_to_index:
                        self.samples.append((self.root / relative, self.class_to_index[wnid]))
        elif split == "train":
            for wnid in self.wnids:
                image_root = self.root / "train" / wnid / "images"
                if not image_root.is_dir():
                    raise FileNotFoundError(f"missing Tiny ImageNet class directory: {image_root}")
                self.samples.extend(
                    (path, self.class_to_index[wnid])
                    for path in sorted(image_root.iterdir())
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
        elif split in {"val", "validation"}:
            annotation_path = self.root / "val" / "val_annotations.txt"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"missing Tiny ImageNet val annotations: {annotation_path}")
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) >= 2 and fields[1] in self.class_to_index:
                    self.samples.append(
                        (self.root / "val" / "images" / fields[0], self.class_to_index[fields[1]])
                    )
        else:
            raise AssertionError("validated Tiny ImageNet split reached an impossible branch")
        if not self.samples or (manifest_path is None and any(not path.is_file() for path, _ in self.samples)):
            raise FileNotFoundError(f"Tiny ImageNet {split} split is incomplete under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), label


def small_vision_base_datasets(config: Dict[str, object], probe: bool = False) -> Tuple[Dataset, Dataset]:
    dataset = str(config["dataset"])
    root = str(config["root"])
    if dataset == "stl10":
        train = STL10Binary(root, "train")
        if not probe and bool(config.get("include_unlabeled", True)):
            train = ConcatDataset([train, STL10Binary(root, "unlabeled")])
        return train, STL10Binary(root, "test")
    if dataset == "tinyimagenet200":
        return (
            TinyImageNetFiles(root, "train", str(config.get("train_manifest", ""))),
            TinyImageNetFiles(root, "val", str(config.get("val_manifest", ""))),
        )
    raise ValueError("small vision dataset must be stl10 or tinyimagenet200")


class SmallVisionDataModule:
    def __init__(self, config: Dict[str, object], seed: int) -> None:
        self.config = config
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self, stage: str = "") -> None:
        train_base, test_base = small_vision_base_datasets(self.config)
        calibration_size = int(self.config["n_calibration"])
        validation_size = int(self.config["n_val"])
        if calibration_size + validation_size >= len(train_base):
            raise ValueError("calibration and validation splits leave no training samples")
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
                Subset(train_base, calibration), transform, views, self.seed + 1000000,
                parent_transform,
            ),
            "val": MultiViewImageDataset(
                Subset(train_base, validation), transform, views, self.seed + 2000000,
                parent_transform,
            ),
            "test": MultiViewImageDataset(
                test_base, transform, views, self.seed + 3000000, parent_transform,
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


class SmallVisionProbeDataModule:
    def __init__(self, config: Dict[str, object], probe: Dict[str, object], seed: int) -> None:
        self.config = config
        self.probe = probe
        self.seed = seed
        self.datasets: Dict[str, Dataset] = {}

    def setup(self) -> None:
        train_base, test_base = small_vision_base_datasets(self.config, probe=True)
        validation_size = int(self.probe.get("n_val", max(1, len(train_base) // 10)))
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
            "test": LabeledImageDataset(test_base, eval_transform),
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
