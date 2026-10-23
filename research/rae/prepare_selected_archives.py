#!/usr/bin/env python3
"""Verify and extract the preregistered ten RoboTwin clean archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(selection: Path, dataset: Path) -> None:
    selected = json.loads(selection.read_text())
    suffix = f"{selected['embodiment']}_{selected['variant']}"
    for task, artifact in selected["tasks"].items():
        parent = dataset / task
        archive = parent / f"{suffix}.zip"
        if archive.stat().st_size != artifact["bytes"] or sha256(archive) != artifact["sha256"]:
            raise ValueError(f"Archive size or SHA256 mismatch: {archive}")
        with ZipFile(archive) as container:
            members = container.infolist()
            for member in members:
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            for member in members:
                if member.filename.startswith("__MACOSX/") or member.filename.endswith(".DS_Store"):
                    continue
                destination = parent / member.filename
                if destination.is_file() and destination.stat().st_size == member.file_size:
                    continue
                container.extract(member, parent)
        episodes = sorted((parent / suffix / "data").glob("episode*.hdf5"))
        if len(episodes) != 50:
            raise ValueError(f"Expected 50 HDF5 episodes for {task}; found {len(episodes)}")
        print(f"OK {task}: archive SHA256 verified, 50 episodes extracted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/root/zifan/openwam-assets/robotwin-training/dataset"))
    args = parser.parse_args()
    prepare(Path(__file__).with_name("selection.json"), args.dataset)
