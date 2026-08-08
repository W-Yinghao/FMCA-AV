#!/usr/bin/env python3
"""Build reusable ImageNet train/validation sample lists without file hashes."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
from typing import Optional


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def data_root(root: Path) -> Path:
    candidate = root / "Data" / "CLS-LOC"
    return candidate if candidate.is_dir() else root


def atomic_tsv(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["path", "wnid"])
            for relative, wnid in rows:
                writer.writerow([relative.as_posix(), wnid])
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return count


def train_rows(root: Path):
    train = root / "train"
    if not train.is_dir():
        raise FileNotFoundError(f"ImageNet training directory is unavailable: {train}")
    for class_dir in sorted(path for path in train.iterdir() if path.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                yield image.relative_to(root), class_dir.name


def validation_assignments(labels: Path) -> dict[str, str]:
    if not labels.is_file():
        raise FileNotFoundError(f"ImageNet validation label CSV is unavailable: {labels}")
    assignments: dict[str, str] = {}
    with labels.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ImageId" not in reader.fieldnames or "PredictionString" not in reader.fieldnames:
            raise ValueError("validation CSV must contain ImageId and PredictionString")
        for row in reader:
            prediction = str(row["PredictionString"]).split()
            if prediction:
                assignments[str(row["ImageId"])] = prediction[0]
    return assignments


def val_rows(root: Path, labels: Path):
    val = root / "val"
    if not val.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory is unavailable: {val}")
    for image_id, wnid in sorted(validation_assignments(labels).items()):
        image = val / f"{image_id}.JPEG"
        if not image.is_file():
            raise FileNotFoundError(f"validation image listed in CSV is unavailable: {image}")
        yield image.relative_to(root), wnid


def existing_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["path", "wnid"]:
            return 0
        return sum(1 for row in reader if row.get("path") and row.get("wnid"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = data_root(Path(args.root).resolve())
    output = Path(args.output_dir).resolve()
    train_manifest = output / "imagenet1k_train.tsv"
    val_manifest = output / "imagenet1k_val.tsv"

    train_count = 0 if args.force else existing_count(train_manifest)
    if train_count == 0:
        train_count = atomic_tsv(train_manifest, train_rows(root))
    val_count = 0 if args.force else existing_count(val_manifest)
    if val_count == 0:
        val_count = atomic_tsv(val_manifest, val_rows(root, Path(args.val_labels)))

    if train_count < 1_000_000:
        raise RuntimeError(f"unexpected ImageNet training manifest size: {train_count}")
    if val_count != 50_000:
        raise RuntimeError(f"unexpected ImageNet validation manifest size: {val_count}")
    print(f"train_manifest={train_manifest} samples={train_count}", flush=True)
    print(f"val_manifest={val_manifest} samples={val_count}", flush=True)


if __name__ == "__main__":
    main()
