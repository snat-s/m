"""Expand only a corpus training split while preserving validation and test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .collect import EXTENSION_LABELS, _is_valid
from .data import load_manifest


def expand_training_set(
    base_manifest: Path,
    destination: Path,
    sources: list[tuple[str, Path]],
    *,
    target_bytes: int,
    max_file_bytes: int = 64 * 1024 * 1024,
    seed: int = 7,
) -> Path:
    base_rows = load_manifest(base_manifest)
    labels = {row["label"] for row in base_rows}
    seen_hashes = {row["sha256"] for row in base_rows}
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        base_manifest.parent / "files", destination / "files", dirs_exist_ok=True
    )
    current_bytes = sum(
        (base_manifest.parent / row["path"]).stat().st_size for row in base_rows
    )

    candidates: dict[str, list[tuple[str, Path, str, str, int]]] = defaultdict(list)
    for source_name, source_root in sources:
        for path in source_root.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            label = EXTENSION_LABELS.get(path.suffix.lower())
            size = path.stat().st_size
            if label not in labels or size > max_file_bytes:
                continue
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes or not _is_valid(path, label, data):
                continue
            seen_hashes.add(digest)
            candidates[label].append(
                (source_name, path, str(path.relative_to(source_root)), digest, size)
            )

    rng = random.Random(seed)
    for items in candidates.values():
        rng.shuffle(items)

    rows = [
        (
            row["path"],
            row["label"],
            row["split"],
            row.get("source", "base"),
            row.get("original_path", row["path"]),
            row["sha256"],
        )
        for row in base_rows
    ]
    labels_in_order = sorted(candidates)
    added = True
    while current_bytes < target_bytes and added:
        added = False
        for label in labels_in_order:
            items = candidates[label]
            while items:
                source, original, original_path, digest, size = items.pop()
                if current_bytes + size > target_bytes:
                    continue
                relative = (
                    Path("files") / label / f"{digest[:16]}{original.suffix.lower()}"
                )
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                rows.append(
                    (str(relative), label, "train", source, original_path, digest)
                )
                current_bytes += size
                added = True
                break
            if current_bytes >= target_bytes:
                break

    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label", "split", "source", "original_path", "sha256"))
        writer.writerows(rows)
    print(
        f"collected {len(rows):,} files, {current_bytes / 1024 / 1024:.1f} MiB "
        f"({len(rows) - len(base_rows):,} new training files)"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--target-mib", type=int, default=1024)
    parser.add_argument("--max-file-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sources: list[tuple[str, Path]] = []
    for value in args.source:
        name, separator, path = value.partition("=")
        if not separator or not name or not Path(path).is_dir():
            parser.error(f"invalid source {value!r}; expected NAME=EXISTING_DIRECTORY")
        sources.append((name, Path(path)))
    expand_training_set(
        args.base_manifest,
        args.destination,
        sources,
        target_bytes=args.target_mib * 1024 * 1024,
        max_file_bytes=args.max_file_mib * 1024 * 1024,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
