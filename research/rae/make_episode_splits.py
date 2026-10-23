#!/usr/bin/env python3
"""Freeze disjoint episode-level RoboTwin splits as symlink-only datasets.

OpenWAM's RoboTwin ``split=train`` and ``split=val`` both enumerate the same
episode files. They only change window enumeration. This script creates
separate data roots so validation cannot see training episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

EPISODE = re.compile(r"episode([0-9]+)\.hdf5$")
SPLIT_SIZES = {"train": 40, "dev": 5, "test": 5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_splits(task: str, salt: str) -> dict[str, list[int]]:
    order = sorted(
        range(50),
        key=lambda i: hashlib.sha256(f"{salt}{task}:episode{i}".encode()).hexdigest(),
    )
    return {"train": order[:40], "dev": order[40:45], "test": order[45:50]}


def link_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Wrong existing symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Would overwrite an existing file: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))


def make_splits(selection: Path, source_root: Path, output_root: Path) -> Path:
    selected = json.loads(selection.read_text())
    if len(selected["tasks"]) != 10 or sum(SPLIT_SIZES.values()) != 50:
        raise ValueError("Selection or split sizes changed; register a new study")
    if output_root.resolve() == source_root.resolve() or output_root.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("Output must not be inside the source dataset")

    manifest = {"selection": selected, "split_sizes": SPLIT_SIZES, "episodes": {}}
    for task, artifact in selected["tasks"].items():
        task_root = source_root / task / f"{selected['embodiment']}_{selected['variant']}"
        archive = task_root.parent / f"{selected['embodiment']}_{selected['variant']}.zip"
        if not archive.is_file() or archive.stat().st_size != artifact["bytes"]:
            raise FileNotFoundError(f"Missing or wrong-size archive: {archive}")
        if sha256(archive) != artifact["sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive}")

        data = task_root / "data"
        found = {}
        for path in data.glob("episode*.hdf5"):
            match = EPISODE.fullmatch(path.name)
            if match:
                found[int(match.group(1))] = path
        if set(found) != set(range(50)):
            raise ValueError(f"Expected episodes 0..49 for {task}; got {sorted(found)}")

        assigned = episode_splits(task, selected["selection_salt"])
        manifest["episodes"][task] = {}
        for split, ids in assigned.items():
            entries = []
            for episode in ids:
                source = found[episode]
                instruction = task_root / "instructions" / f"episode{episode}.json"
                if not instruction.is_file():
                    raise FileNotFoundError(instruction)
                dest_root = output_root / split / "dataset" / task / f"{selected['embodiment']}_{selected['variant']}"
                link_once(source, dest_root / "data" / source.name)
                link_once(instruction, dest_root / "instructions" / instruction.name)
                entries.append(
                    {"episode": episode, "hdf5_sha256": sha256(source), "instruction_sha256": sha256(instruction)}
                )
            manifest["episodes"][task][split] = entries

    target = output_root / "split_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text() != serialized:
        raise FileExistsError(f"Existing manifest differs: {target}")
    target.write_text(serialized)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/root/zifan/openwam-assets/robotwin-training/dataset"))
    parser.add_argument("--output", type=Path, default=Path("/root/zifan/openwam-assets/robotwin-splits"))
    args = parser.parse_args()
    plan = Path(__file__).with_name("selection.json")
    print(make_splits(plan, args.source, args.output))
