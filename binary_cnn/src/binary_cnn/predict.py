"""Classify one binary file with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import extract_byte_windows
from .histogram import byte_pair_histogram, histogram_to_model_input
from .model import BytePairCNN, ByteWindowNet, HistogramResNet18
from .train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    labels: list[str] = checkpoint["labels"]
    config = checkpoint.get("model_config", {})
    representation = checkpoint.get("representation", "histogram")
    architecture = checkpoint.get("architecture", "tiny-cnn")
    expected_preprocessing = (
        "magika_v1_3x512" if representation == "byte-windows" else "log1p_10000"
    )
    if checkpoint.get("preprocessing") != expected_preprocessing:
        raise ValueError("checkpoint uses an unsupported preprocessing format")
    if representation == "byte-windows":
        model = ByteWindowNet(
            len(labels),
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=checkpoint["dropout"],
        )
    elif architecture == "resnet18":
        model = HistogramResNet18(len(labels), pretrained=False)
    else:
        width = config.get("width", checkpoint.get("width"))
        model = BytePairCNN(len(labels), width=width, dropout=checkpoint["dropout"])
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data = args.input.read_bytes()
    max_bytes = checkpoint.get("max_bytes")
    if max_bytes is not None:
        data = data[:max_bytes]
    if representation == "byte-windows":
        inputs = extract_byte_windows(data).unsqueeze(0).to(device)
    else:
        histogram = byte_pair_histogram(data)
        inputs = (
            torch.from_numpy(histogram_to_model_input(histogram))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
    with torch.no_grad():
        probabilities = model(inputs).softmax(dim=1)[0].cpu()
    for index in probabilities.argsort(descending=True):
        print(f"{labels[index]}\t{float(probabilities[index]):.4f}")


if __name__ == "__main__":
    main()
