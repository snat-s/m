"""Train and evaluate a binary-file classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import BinaryHistogramDataset, ByteWindowDataset, load_manifest
from .model import BytePairCNN, ByteWindowNet, HistogramResNet18


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for inputs, targets in loader:
        predictions = model(inputs.to(device)).argmax(dim=1).cpu()
        for target, prediction in zip(targets, predictions):
            confusion[target, prediction] += 1
        correct += int((predictions == targets).sum())
        total += len(targets)
    return (correct / total if total else 0.0), confusion.tolist()


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.representation == "byte-windows" and args.architecture != "tiny-cnn":
        raise ValueError("resnet18 supports only histogram inputs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    rows = load_manifest(args.manifest)
    labels = sorted({row["label"] for row in rows})
    if args.representation == "histogram":
        datasets = {
            split: BinaryHistogramDataset(
                args.manifest, split, labels, max_bytes=args.max_bytes
            )
            for split in ("train", "val", "test")
        }
        if args.architecture == "tiny-cnn":
            model: nn.Module = BytePairCNN(
                len(labels), width=args.width, dropout=args.dropout
            )
            model_config = {"width": args.width}
        else:
            model = HistogramResNet18(len(labels), pretrained=not args.no_pretrained)
            model_config = {"pretrained": not args.no_pretrained}
    else:
        datasets = {
            split: ByteWindowDataset(args.manifest, split, labels)
            for split in ("train", "val", "test")
        }
        model = ByteWindowNet(
            len(labels),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        model_config = {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
        }
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        sampler = None
        if split == "train" and args.balanced:
            counts = {
                label: sum(row["label"] == label for row in dataset.rows)
                for label in labels
            }
            weights = [1 / counts[row["label"]] for row in dataset.rows]
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

    model = model.to(device)
    if isinstance(model, HistogramResNet18):
        model.set_backbone_trainable(args.freeze_backbone_epochs == 0)
        head_parameters = list(model.backbone.fc.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        backbone_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.learning_rate * 0.1},
                {"params": head_parameters, "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        if (
            isinstance(model, HistogramResNet18)
            and epoch == args.freeze_backbone_epochs + 1
        ):
            model.set_backbone_trainable(True)
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item() * len(targets)
            seen += len(targets)
        val_accuracy, _ = evaluate(model, loaders["val"], device, len(labels))
        print(
            f"epoch={epoch:02d} loss={running_loss / max(seen, 1):.4f} "
            f"val_accuracy={val_accuracy:.3f}"
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    test_accuracy, confusion = evaluate(model, loaders["test"], device, len(labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "labels": labels,
            "representation": args.representation,
            "architecture": args.architecture,
            "model_config": model_config,
            "dropout": args.dropout,
            "max_bytes": args.max_bytes,
            "preprocessing": (
                "log1p_10000"
                if args.representation == "histogram"
                else "magika_v1_3x512"
            ),
        },
        args.output,
    )
    metrics = {
        "best_val_accuracy": best_val,
        "test_accuracy": test_accuracy,
        "labels": labels,
        "confusion_matrix": confusion,
        "checkpoint": str(args.output),
        "representation": args.representation,
        "architecture": args.architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "balanced_sampling": args.balanced,
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"test_accuracy={test_accuracy:.3f}")
    print(f"checkpoint={args.output} metrics={metrics_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--representation",
        choices=("histogram", "byte-windows"),
        default="histogram",
    )
    parser.add_argument(
        "--architecture", choices=("tiny-cnn", "resnet18"), default="tiny-cnn"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--balanced", action="store_true", help="sample training classes equally"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
