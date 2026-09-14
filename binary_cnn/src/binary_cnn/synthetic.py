"""Create a small synthetic dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from pathlib import Path

LABELS = ("compressed_like", "executable_like", "image_like", "text_like")


def _text_like(rng: random.Random, size: int) -> bytes:
    words = (
        b"the quick brown fox jumps over the lazy dog binary histogram network "
        b"small training data file classification experiment convolution byte "
    )
    output = bytearray()
    while len(output) < size:
        start = rng.randrange(len(words))
        output.extend(words[start:] + b"\n")
    return bytes(output[:size])


def _executable_like(rng: random.Random, size: int) -> bytes:
    # Common x86-like instruction fragments mixed with data sections.
    fragments = [
        b"\x55\x48\x89\xe5",
        b"\x48\x83\xec\x20",
        b"\x48\x8b\x45\xf8",
        b"\xe8\x00\x00\x00\x00",
        b"\x31\xc0\xc3",
        b"\x90\x90\x90\x90",
    ]
    output = bytearray(b"\x7fELF\x02\x01\x01\x00" + bytes(56))
    while len(output) < size:
        if rng.random() < 0.8:
            output.extend(rng.choice(fragments))
        else:
            output.extend(rng.randbytes(rng.randint(2, 12)))
    return bytes(output[:size])


def _image_like(rng: random.Random, size: int) -> bytes:
    output = bytearray(b"BM" + size.to_bytes(4, "little") + bytes(48))
    value = rng.randrange(256)
    while len(output) < size:
        value = max(0, min(255, value + rng.choice((-2, -1, 0, 0, 1, 2))))
        pixel = bytes((value, (value * 3) % 256, 255 - value))
        output.extend(pixel * rng.randint(1, 8))
    return bytes(output[:size])


def _compressed_like(rng: random.Random, size: int) -> bytes:
    payload = rng.randbytes(max(1, size * 2))
    compressed = gzip.compress(payload, compresslevel=1, mtime=0)
    if len(compressed) >= size:
        return compressed[:size]
    return compressed + rng.randbytes(size - len(compressed))


GENERATORS = {
    "compressed_like": _compressed_like,
    "executable_like": _executable_like,
    "image_like": _image_like,
    "text_like": _text_like,
}


def generate_dataset(
    destination: Path,
    *,
    samples_per_class: int = 100,
    min_size: int = 4_096,
    max_size: int = 16_384,
    seed: int = 7,
) -> Path:
    if samples_per_class < 3:
        raise ValueError("samples_per_class must be at least 3")
    if min_size < 64 or max_size < min_size:
        raise ValueError("sizes must satisfy 64 <= min_size <= max_size")

    rng = random.Random(seed)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    for label in LABELS:
        label_dir = destination / "files" / label
        label_dir.mkdir(parents=True, exist_ok=True)
        indices = list(range(samples_per_class))
        rng.shuffle(indices)
        train_end = round(samples_per_class * 0.7)
        val_end = round(samples_per_class * 0.85)
        splits = {
            index: "train" if rank < train_end else "val" if rank < val_end else "test"
            for rank, index in enumerate(indices)
        }
        for index in range(samples_per_class):
            size = rng.randint(min_size, max_size)
            path = label_dir / f"{index:04d}.bin"
            path.write_bytes(GENERATORS[label](rng, size))
            rows.append((str(path.relative_to(destination)), label, splits[index]))

    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label", "split"))
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--min-size", type=int, default=4_096)
    parser.add_argument("--max-size", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = generate_dataset(
        args.destination,
        samples_per_class=args.samples_per_class,
        min_size=args.min_size,
        max_size=args.max_size,
        seed=args.seed,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
