"""File-backed ImageNet val access for the chain pilots.

Two sources, one pairing:

``ImageNetValFiles`` reads the 50k validation images through the
manifest (path -> wnid), in manifest order, so a (seed, samples) draw
selects the same disjoint block everywhere.

``ImageNetCFiles`` serves the SAME images under a named corruption and
severity: ImageNet-C keeps the original validation filenames, so the
pairing between a clean image and its corrupted versions is by name and
is exact.  Clean images pass through the standard eval transform
(resize 256, center crop 224); ImageNet-C ships at 224x224 already and
is only normalized -- the asymmetry is a property of how the benchmark
was published, and is recorded by the caller rather than hidden.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_CLEAN = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
_PRESIZED = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _read_manifest(manifest: Path) -> List[Tuple[str, str]]:
    with manifest.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [(row["path"], row["wnid"]) for row in reader]
    if not rows:
        raise ValueError(f"empty manifest {manifest}")
    return rows


class ImageNetValFiles:
    """The clean validation set, in manifest order, with wnid-sorted labels."""

    def __init__(self, root: str, manifest: Optional[str] = None) -> None:
        base = Path(root)
        rows = _read_manifest(Path(manifest) if manifest else base / "manifests" / "imagenet1k_val.tsv")
        wnids = sorted({wnid for _, wnid in rows})
        self.class_index = {wnid: index for index, wnid in enumerate(wnids)}
        self.records = [(base / "ILSVRC" / "Data" / "CLS-LOC" / path,
                         self.class_index[wnid], Path(path).name, wnid)
                        for path, wnid in rows]
        missing = next((r for r in self.records[:5] if not r[0].exists()), None)
        if missing is not None:
            raise FileNotFoundError(f"manifest points at {missing[0]}, which does not exist")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def classes(self) -> int:
        return len(self.class_index)

    def load_block(self, low: int, high: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if high > len(self.records):
            raise ValueError(f"block [{low}:{high}] exceeds {len(self.records)} images")
        images, labels = [], []
        for path, label, _, _ in self.records[low:high]:
            with Image.open(path) as img:
                images.append(_CLEAN(img.convert("RGB")))
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


class ImageNetCFiles:
    """One corruption at one severity, paired to the clean set by filename."""

    def __init__(self, root: str, corruption: str, severity: int,
                 clean: ImageNetValFiles) -> None:
        base = Path(root) / corruption / str(int(severity))
        if not base.is_dir():
            raise FileNotFoundError(f"no ImageNet-C directory {base}")
        self.base = base
        self.clean = clean

    def load_block(self, low: int, high: int) -> torch.Tensor:
        images = []
        for _, _, name, wnid in self.clean.records[low:high]:
            path = self.base / wnid / name
            with Image.open(path) as img:
                images.append(_PRESIZED(img.convert("RGB")))
        return torch.stack(images)
