#!/usr/bin/env python3
"""Build reusable Tiny ImageNet sample lists without file hashes."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
from typing import Iterable, Optional, Tuple


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def atomic_tsv(path: Path, rows: Iterable[Tuple[Path, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    count = 0
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


def wnids(root: Path) -> list[str]:
    path = root / "wnids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing Tiny ImageNet wnids: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_rows(root: Path, classes: list[str]):
    for wnid in classes:
        image_root = root / "train" / wnid / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(f"missing Tiny ImageNet class directory: {image_root}")
        for image in sorted(image_root.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                yield image.relative_to(root), wnid


def val_rows(root: Path, classes: set[str]):
    annotations = root / "val" / "val_annotations.txt"
    if not annotations.is_file():
        raise FileNotFoundError(f"missing Tiny ImageNet val annotations: {annotations}")
    for line in annotations.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or fields[1] not in classes:
            continue
        image = root / "val" / "images" / fields[0]
        if not image.is_file():
            raise FileNotFoundError(f"missing Tiny ImageNet validation image: {image}")
        yield image.relative_to(root), fields[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    classes = wnids(root)
    train = output / "tinyimagenet200_train.tsv"
    val = output / "tinyimagenet200_val.tsv"
    train_count = atomic_tsv(train, train_rows(root, classes))
    val_count = atomic_tsv(val, val_rows(root, set(classes)))
    if train_count != 100_000 or val_count != 10_000:
        raise RuntimeError(
            f"unexpected Tiny ImageNet sample counts: train={train_count}, val={val_count}"
        )
    print(f"train_manifest={train} samples={train_count}", flush=True)
    print(f"val_manifest={val} samples={val_count}", flush=True)


if __name__ == "__main__":
    main()
