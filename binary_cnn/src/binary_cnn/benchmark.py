"""Evaluate libmagic on the same held-out split as the neural models."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import BinaryHistogramDataset, ByteWindowDataset, load_manifest
from .model import BytePairCNN, ByteWindowNet, HistogramResNet18

MIME_LABELS = {
    "application/json": "json",
    "application/pdf": "pdf",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/zip": "zip",
    "image/png": "png",
    "text/rtf": "rtf",
    "text/x-script.python": "python",
    "text/xml": "xml",
}


def benchmark_libmagic(manifest: Path, *, repeats: int = 5) -> dict[str, object]:
    rows = [row for row in load_manifest(manifest) if row["split"] == "test"]
    paths = [manifest.parent / row["path"] for row in rows]

    def run_once() -> tuple[subprocess.CompletedProcess[str], float]:
        started = time.perf_counter()
        process = subprocess.run(
            ["file", "--mime-type", "-b", *map(str, paths)],
            capture_output=True,
            text=True,
            check=True,
        )
        return process, time.perf_counter() - started

    run_once()
    runs = [run_once() for _ in range(repeats)]
    process = runs[0][0]
    mime_types = process.stdout.splitlines()
    if len(mime_types) != len(rows):
        raise RuntimeError("libmagic returned an unexpected number of results")

    class_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for row, mime_type in zip(rows, mime_types):
        prediction = MIME_LABELS.get(mime_type, "unknown")
        is_correct = prediction == row["label"]
        correct += is_correct
        class_totals[row["label"]][0] += is_correct
        class_totals[row["label"]][1] += 1
    return {
        "name": "libmagic",
        "test_accuracy": correct / len(rows),
        "test_samples": len(rows),
        "milliseconds_per_file": statistics.median(
            elapsed * 1_000 / len(rows) for _, elapsed in runs
        ),
        "per_class": {
            label: {"correct": values[0], "total": values[1]}
            for label, values in sorted(class_totals.items())
        },
    }


@torch.no_grad()
def benchmark_checkpoint(
    manifest: Path, checkpoint_path: Path, *, repeats: int = 5
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels: list[str] = checkpoint["labels"]
    representation = checkpoint.get("representation", "histogram")
    config = checkpoint.get("model_config", {})
    if representation == "histogram":
        architecture = checkpoint.get("architecture", "tiny-cnn")
        if architecture == "resnet18":
            model = HistogramResNet18(len(labels), pretrained=False)
        else:
            width = config.get("width", checkpoint.get("width"))
            model = BytePairCNN(len(labels), width=width, dropout=checkpoint["dropout"])
        dataset = BinaryHistogramDataset(manifest, "test", labels)
    elif representation == "byte-windows":
        model = ByteWindowNet(
            len(labels),
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=checkpoint["dropout"],
        )
        dataset = ByteWindowDataset(manifest, "test", labels)
    else:
        raise ValueError(f"unsupported representation: {representation}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(dataset, batch_size=32)

    def run_once() -> tuple[int, float]:
        correct = 0
        started = time.perf_counter()
        for inputs, targets in loader:
            correct += int((model(inputs).argmax(dim=1) == targets).sum())
        return correct, time.perf_counter() - started

    run_once()  # Warm model weights and filesystem caches.
    runs = [run_once() for _ in range(repeats)]
    correct = runs[0][0]
    return {
        "name": (
            checkpoint.get("architecture", "tiny-cnn")
            if representation == "histogram"
            else representation
        ),
        "test_accuracy": correct / len(dataset),
        "test_samples": len(dataset),
        "milliseconds_per_file": statistics.median(
            elapsed * 1_000 / len(dataset) for _, elapsed in runs
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {
        "test_samples": sum(
            row["split"] == "test" for row in load_manifest(args.manifest)
        ),
        "results": [
            benchmark_libmagic(args.manifest, repeats=args.repeats),
            *(
                benchmark_checkpoint(args.manifest, checkpoint, repeats=args.repeats)
                for checkpoint in args.checkpoint
            ),
        ],
    }
    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
