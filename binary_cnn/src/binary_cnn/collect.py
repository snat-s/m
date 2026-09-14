"""Build a deduplicated real-file corpus from local source repositories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

EXTENSION_LABELS = {
    ".c": "c",
    ".cpp": "cpp",
    ".docx": "docx",
    ".gif": "gif",
    ".go": "go",
    ".htm": "html",
    ".html": "html",
    ".java": "java",
    ".jpeg": "jpeg",
    ".jpg": "jpeg",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".pdf": "pdf",
    ".png": "png",
    ".pptx": "pptx",
    ".py": "python",
    ".rs": "rust",
    ".rtf": "rtf",
    ".toml": "toml",
    ".ts": "typescript",
    ".txt": "text",
    ".xls": "xls",
    ".xlsx": "xlsx",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zip": "zip",
}
TEXT_LABELS = {
    "c",
    "cpp",
    "go",
    "html",
    "java",
    "javascript",
    "json",
    "markdown",
    "python",
    "rust",
    "text",
    "toml",
    "typescript",
    "xml",
    "yaml",
}


def _is_valid(path: Path, label: str, data: bytes) -> bool:
    if len(data) < 16:
        return False
    if label in TEXT_LABELS:
        try:
            data.decode("utf-8")
            return b"\0" not in data
        except UnicodeDecodeError:
            return False
    if label == "pdf":
        return data.lstrip()[:5] == b"%PDF-"
    if label == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if label == "jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if label == "gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if label == "rtf":
        return data.lstrip().startswith(b"{\\rtf")
    if label in {"zip", "docx", "xlsx", "pptx"}:
        if not data.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return False
        if label == "zip":
            return True
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile):
            return False
        expected = {"docx": "word/", "xlsx": "xl/", "pptx": "ppt/"}[label]
        return any(name.startswith(expected) for name in names)
    # Legacy XLS uses the OLE Compound File signature.
    if label == "xls":
        return data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    return False


def collect_dataset(
    destination: Path,
    sources: list[tuple[str, Path]],
    *,
    samples_per_class: int = 100,
    minimum_per_class: int = 20,
    max_file_bytes: int = 16 * 1024 * 1024,
    seed: int = 7,
) -> Path:
    candidates: dict[str, list[tuple[str, Path, str, str]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for source_name, source_root in sources:
        for path in source_root.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            label = EXTENSION_LABELS.get(path.suffix.lower())
            if label is None or path.stat().st_size > max_file_bytes:
                continue
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes or not _is_valid(path, label, data):
                continue
            seen_hashes.add(digest)
            candidates[label].append(
                (source_name, path, str(path.relative_to(source_root)), digest)
            )

    rng = random.Random(seed)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str, str, str, str]] = []
    for label, items in sorted(candidates.items()):
        if len(items) < minimum_per_class:
            continue
        rng.shuffle(items)
        selected = items[:samples_per_class]
        train_end = round(len(selected) * 0.7)
        val_end = round(len(selected) * 0.85)
        for index, (source, original, original_path, digest) in enumerate(selected):
            split = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
            suffix = original.suffix.lower()
            relative = Path("files") / label / f"{digest[:16]}{suffix}"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            rows.append((str(relative), label, split, source, original_path, digest))

    if not rows:
        raise ValueError("no class met the minimum sample count")
    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label", "split", "source", "original_path", "sha256"))
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for each independently sourced repository",
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--max-file-mib", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sources: list[tuple[str, Path]] = []
    for value in args.source:
        name, separator, path = value.partition("=")
        if not separator or not name or not Path(path).is_dir():
            parser.error(f"invalid source {value!r}; expected NAME=EXISTING_DIRECTORY")
        sources.append((name, Path(path)))
    manifest = collect_dataset(
        args.destination,
        sources,
        samples_per_class=args.samples_per_class,
        minimum_per_class=args.minimum_per_class,
        max_file_bytes=args.max_file_mib * 1024 * 1024,
        seed=args.seed,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
