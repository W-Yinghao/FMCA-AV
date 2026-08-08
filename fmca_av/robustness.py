"""ImageNet-C/R/A datasets and classification robustness metrics."""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data.imagenet import IMAGE_EXTENSIONS


# Canonical AlexNet top-1 corruption errors from the ImageNet-C benchmark.
# The four later "extra" corruptions are intentionally excluded from mCE.
IMAGENET_C_ALEXNET_ERRORS = {
    "gaussian_noise": 0.886428,
    "shot_noise": 0.894468,
    "impulse_noise": 0.922640,
    "defocus_blur": 0.819880,
    "glass_blur": 0.826268,
    "motion_blur": 0.785948,
    "zoom_blur": 0.798360,
    "snow": 0.866816,
    "frost": 0.826572,
    "fog": 0.819324,
    "brightness": 0.564592,
    "contrast": 0.853204,
    "elastic_transform": 0.646056,
    "pixelate": 0.717840,
    "jpeg_compression": 0.606500,
}
IMAGENET_ALEXNET_CLEAN_ERROR = 0.434500


class WNIDFolderDataset(Dataset):
    def __init__(self, root: str, wnids: Sequence[str], transform: object) -> None:
        self.root = Path(root)
        self.class_to_index = {wnid: index for index, wnid in enumerate(wnids)}
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        if not self.root.is_dir():
            raise FileNotFoundError(f"classification folder is unavailable: {self.root}")
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if directory.name not in self.class_to_index:
                continue
            label = self.class_to_index[directory.name]
            self.samples.extend(
                (path, label)
                for path in sorted(directory.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        if not self.samples:
            raise FileNotFoundError(f"no recognized wnid image folders under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


class ClassificationMetricAccumulator:
    def __init__(self, bins: int = 15) -> None:
        self.bins = bins
        self.correct1 = 0
        self.correct5 = 0
        self.total = 0
        self.bin_count = torch.zeros(bins, dtype=torch.long)
        self.bin_confidence = torch.zeros(bins, dtype=torch.float64)
        self.bin_accuracy = torch.zeros(bins, dtype=torch.float64)

    def update(self, logits: Tensor, labels: Tensor) -> None:
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        correct = prediction.eq(labels)
        self.correct1 += int(correct.sum())
        self.correct5 += int(
            (logits.topk(min(5, logits.shape[1]), dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum()
        )
        self.total += labels.numel()
        indices = torch.clamp((confidence * self.bins).long(), max=self.bins - 1).cpu()
        confidence_cpu = confidence.double().cpu()
        correct_cpu = correct.double().cpu()
        for index in range(self.bins):
            selected = indices == index
            count = int(selected.sum())
            if count:
                self.bin_count[index] += count
                self.bin_confidence[index] += confidence_cpu[selected].sum()
                self.bin_accuracy[index] += correct_cpu[selected].sum()

    def metrics(self) -> Dict[str, float]:
        if not self.total:
            raise ValueError("no samples were accumulated")
        ece = 0.0
        for index in range(self.bins):
            count = int(self.bin_count[index])
            if count:
                confidence = float(self.bin_confidence[index] / count)
                accuracy = float(self.bin_accuracy[index] / count)
                ece += count / self.total * abs(accuracy - confidence)
        return {
            "top1_accuracy": self.correct1 / self.total,
            "top5_accuracy": self.correct5 / self.total,
            "ece_15_bin": ece,
            "samples": float(self.total),
        }
