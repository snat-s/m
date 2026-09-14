"""Build two-dimensional histograms of adjacent byte pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def byte_pair_histogram(data: bytes, *, offset: int = 1) -> np.ndarray:
    """Return a normalized 256×256 histogram of ``data[i], data[i+offset]``.

    The output sums to one when at least one pair exists and is all zeros
    otherwise. ``offset=1`` measures adjacent bytes.
    """
    if offset < 1:
        raise ValueError("offset must be at least 1")

    histogram = np.zeros(256 * 256, dtype=np.float32)
    if len(data) <= offset:
        return histogram.reshape(256, 256)

    values = np.frombuffer(data, dtype=np.uint8)
    indices = values[:-offset].astype(np.int32) * 256 + values[offset:]
    counts = np.bincount(indices, minlength=256 * 256).astype(np.float32)
    counts /= counts.sum()
    return counts.reshape(256, 256)


def histogram_file(path: str | Path, *, offset: int = 1) -> np.ndarray:
    return byte_pair_histogram(Path(path).read_bytes(), offset=offset)


def histogram_to_model_input(histogram: np.ndarray) -> np.ndarray:
    """Log-scale sparse frequencies to a useful neural-network input range."""
    return np.log1p(histogram * 10_000).astype(np.float32)


def save_histogram_image(histogram: np.ndarray, destination: str | Path) -> None:
    """Save a log-scaled histogram preview without changing model inputs."""
    peak = float(histogram.max())
    if peak == 0:
        pixels = np.zeros((256, 256), dtype=np.uint8)
    else:
        scaled = histogram_to_model_input(histogram)
        pixels = np.rint(scaled / scaled.max() * 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="binary file to inspect")
    parser.add_argument("output", type=Path, help="output .png or .npy file")
    parser.add_argument(
        "--offset", type=int, default=1, help="distance between paired bytes"
    )
    args = parser.parse_args()

    histogram = histogram_file(args.input, offset=args.offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".npy":
        np.save(args.output, histogram)
    elif args.output.suffix.lower() == ".png":
        save_histogram_image(histogram, args.output)
    else:
        parser.error("output must end in .png or .npy")


if __name__ == "__main__":
    main()
