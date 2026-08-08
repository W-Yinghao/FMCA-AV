#!/usr/bin/env python3
"""Inspect factor-dataset containers without materializing their large arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import zipfile

import numpy as np


def npy_header(source) -> dict[str, object]:
    version = np.lib.format.read_magic(source)
    shape, fortran_order, dtype = np.lib.format._read_array_header(source, version)
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
    }


def inspect_npz(path: Path) -> dict[str, object]:
    members: dict[str, object] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".npy"):
                continue
            with archive.open(info, "r") as source:
                members[info.filename] = {
                    **npy_header(source),
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
    return {"kind": "npz", "bytes": path.stat().st_size, "members": members}


def inspect_h5(path: Path) -> dict[str, object]:
    import h5py

    members: dict[str, object] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, item) -> None:
            if isinstance(item, h5py.Dataset):
                members[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

        handle.visititems(visit)
    return {"kind": "h5", "bytes": path.stat().st_size, "members": members}


def inspect_norb_matrix(path: Path) -> dict[str, object]:
    # The NORB matrix header is little-endian: magic, ndim, then dimensions.
    with path.open("rb") as source:
        magic, ndim = struct.unpack("<II", source.read(8))
        stored_dims = max(ndim, 3)
        dimensions = list(struct.unpack(f"<{stored_dims}I", source.read(4 * stored_dims)))[:ndim]
    return {
        "kind": "norb-matrix",
        "bytes": path.stat().st_size,
        "magic": magic,
        "shape": dimensions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result: dict[str, object] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if path.suffix == ".npz":
            result[relative] = inspect_npz(path)
        elif path.suffix in {".h5", ".hdf5"}:
            result[relative] = inspect_h5(path)
        elif path.suffix == ".mat" and "smallnorb" in path.name:
            result[relative] = inspect_norb_matrix(path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
