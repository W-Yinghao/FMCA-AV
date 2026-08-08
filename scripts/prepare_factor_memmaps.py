#!/usr/bin/env python3
"""Stream NPZ members into reusable NPY files without loading arrays into RAM."""

import argparse
import os
from pathlib import Path
import shutil
import zipfile


def extract(npz_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir() and info.filename.endswith(".npy")]
        if not members:
            raise ValueError(f"NPZ contains no NPY members: {npz_path}")
        for member in members:
            name = Path(member.filename).name
            destination = output_root / name
            if destination.is_file() and destination.stat().st_size == member.file_size:
                print(f"SKIP {destination}")
                continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(member, "r") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if temporary.stat().st_size != member.file_size:
                raise IOError(f"extracted size differs from NPZ metadata: {member.filename}")
            temporary.replace(destination)
            print(f"EXTRACTED {destination}")
    marker = output_root / ".complete"
    marker.touch()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract(Path(args.npz).resolve(), Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
