"""Dataset loading for byte-pair CNN experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .histogram import histogram_file, histogram_to_model_input


def load_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"manifest must contain columns: {', '.join(sorted(required))}"
        )
    return rows


class BinaryHistogramDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest: Path,
        split: str,
        labels: list[str],
        *,
        max_bytes: int | None = None,
    ) -> None:
        self.root = manifest.parent
        self.rows = [row for row in load_manifest(manifest) if row["split"] == split]
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.max_bytes = max_bytes
        unknown = {row["label"] for row in self.rows} - self.label_to_index.keys()
        if unknown:
            raise ValueError(f"unknown labels: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = self.root / row["path"]
        if self.max_bytes is None:
            histogram = histogram_file(path)
        else:
            histogram = histogram_file_bytes(path, self.max_bytes)
        inputs = torch.from_numpy(histogram_to_model_input(histogram)).unsqueeze(0)
        target = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        return inputs, target


def extract_byte_windows(data: bytes, window_size: int = 512) -> torch.Tensor:
    """Extract padded beginning, midpoint, and ending windows as Magika v1 did."""
    padding = 256
    beginning = data.lstrip()[:window_size]
    ending = data.rstrip()[-window_size:]
    midpoint_start = max(0, (len(data) - window_size) // 2)
    midpoint = data[midpoint_start : midpoint_start + window_size]

    beginning_values = list(beginning) + [padding] * (window_size - len(beginning))
    midpoint_padding = window_size - len(midpoint)
    midpoint_values = (
        [padding] * (midpoint_padding // 2)
        + list(midpoint)
        + [padding] * (midpoint_padding - midpoint_padding // 2)
    )
    ending_values = [padding] * (window_size - len(ending)) + list(ending)
    return torch.tensor(
        beginning_values + midpoint_values + ending_values, dtype=torch.long
    )


def extract_byte_windows_file(path: Path, window_size: int = 512) -> torch.Tensor:
    """Read only the three required windows instead of loading the entire file."""
    size = path.stat().st_size
    if size < window_size:
        return extract_byte_windows(path.read_bytes(), window_size)

    with path.open("rb") as handle:
        strip_buffer_size = min(size, window_size * 8)
        beginning = handle.read(strip_buffer_size).lstrip()[:window_size]
        handle.seek((size - window_size) // 2)
        midpoint = handle.read(window_size)
        handle.seek(size - strip_buffer_size)
        ending = handle.read(strip_buffer_size).rstrip()[-window_size:]

    padding = 256
    values = (
        list(beginning)
        + [padding] * (window_size - len(beginning))
        + list(midpoint)
        + [padding] * (window_size - len(midpoint))
        + [padding] * (window_size - len(ending))
        + list(ending)
    )
    return torch.tensor(values, dtype=torch.long)


class ByteWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, manifest: Path, split: str, labels: list[str]) -> None:
        self.root = manifest.parent
        self.rows = [row for row in load_manifest(manifest) if row["split"] == split]
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        inputs = extract_byte_windows_file(self.root / row["path"])
        target = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        return inputs, target


def histogram_file_bytes(path: Path, max_bytes: int) -> np.ndarray:
    from .histogram import byte_pair_histogram

    with path.open("rb") as handle:
        return byte_pair_histogram(handle.read(max_bytes))
