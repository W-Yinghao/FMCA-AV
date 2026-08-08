#!/usr/bin/env python3
"""Prepare a restartable uncompressed Shapes3D memmap on a CPU Slurm node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_array(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r")
        return tuple(value.shape) == shape and value.dtype == dtype
    except (OSError, ValueError):
        return False


def validate_samples(
    handle: h5py.File,
    images_path: Path,
    labels_path: Path,
    count: int = 32,
) -> None:
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    indices = np.linspace(0, len(images) - 1, num=min(count, len(images)), dtype=np.int64)
    mismatches = []
    for raw_index in indices:
        index = int(raw_index)
        if not np.array_equal(images[index], handle["images"][index]):
            mismatches.append(f"images[{index}]")
        if not np.array_equal(labels[index], handle["labels"][index]):
            mismatches.append(f"labels[{index}]")
    if mismatches:
        raise RuntimeError("Shapes3D memmap differs from official HDF5: " + ",".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    args = parser.parse_args()
    if not os.environ.get("FMCA_HARNESS_RUN_DIR"):
        raise RuntimeError("Shapes3D preparation must run through the Slurm harness")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    root = Path(args.root).resolve()
    source = root / "3dshapes" / "3dshapes.h5"
    output = root / "prepared" / "shapes3d"
    output.mkdir(parents=True, exist_ok=True)
    images_path = output / "images.npy"
    labels_path = output / "labels.npy"
    complete_path = output / ".complete"
    progress_path = output / "progress.json"
    images_partial = output / "images.npy.partial"
    labels_partial = output / "labels.npy.partial"

    with h5py.File(source, "r") as handle:
        image_shape = tuple(int(value) for value in handle["images"].shape)
        label_shape = tuple(int(value) for value in handle["labels"].shape)
        image_dtype = np.dtype(handle["images"].dtype)
        label_dtype = np.dtype(handle["labels"].dtype)

        if (complete_path.is_file()
                and valid_array(images_path, image_shape, image_dtype)
                and valid_array(labels_path, label_shape, label_dtype)):
            validate_samples(handle, images_path, labels_path)
            print(json.dumps({"state": "already_complete_and_validated", "images": str(images_path), "rows": image_shape[0], "validated_samples": 32}))
            return 0

        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        start = int(progress.get("completed_rows", 0))
        if start and not (valid_array(images_partial, image_shape, image_dtype)
                          and valid_array(labels_partial, label_shape, label_dtype)):
            start = 0
        if start == 0:
            images = np.lib.format.open_memmap(images_partial, mode="w+", dtype=image_dtype, shape=image_shape)
            labels = np.lib.format.open_memmap(labels_partial, mode="w+", dtype=label_dtype, shape=label_shape)
        else:
            images = np.load(images_partial, mmap_mode="r+")
            labels = np.load(labels_partial, mmap_mode="r+")

        for begin in range(start, image_shape[0], args.chunk_size):
            end = min(begin + args.chunk_size, image_shape[0])
            images[begin:end] = handle["images"][begin:end]
            labels[begin:end] = handle["labels"][begin:end]
            images.flush()
            labels.flush()
            atomic_json(progress_path, {
                "completed_rows": end,
                "total_rows": image_shape[0],
                "source": str(source),
            })

    del images, labels
    images_partial.replace(images_path)
    labels_partial.replace(labels_path)
    with h5py.File(source, "r") as handle:
        validate_samples(handle, images_path, labels_path)
    complete_path.touch()
    progress_path.unlink(missing_ok=True)
    result = {"state": "completed", "images": str(images_path), "labels": str(labels_path), "rows": image_shape[0]}
    artifact = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact / "shapes3d_memmap.json", result)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "prepare_shapes3d_memmap", **result}, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
