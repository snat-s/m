"""Render representative dataset histograms as a labeled contact sheet."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .data import load_manifest
from .histogram import histogram_file, save_histogram_image


def visualize_dataset(
    manifest: Path, output: Path, *, split: str = "train", samples_per_class: int = 3
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load_manifest(manifest):
        if row["split"] == split and len(grouped[row["label"]]) < samples_per_class:
            grouped[row["label"]].append(row)
    if not grouped:
        raise ValueError(f"manifest contains no {split!r} samples")

    output.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(grouped)
    cell_width, cell_height = 280, 310
    header_height = 56
    sheet = Image.new(
        "RGB",
        (cell_width * samples_per_class, header_height + cell_height * len(labels)),
        "#111827",
    )
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    label_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    detail_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    draw.text(
        (16, 14),
        f"Byte-pair histogram dataset · {split} split",
        fill="white",
        font=title_font,
    )

    for row_index, label in enumerate(labels):
        for column, row in enumerate(grouped[label]):
            source = manifest.parent / row["path"]
            preview_path = output.parent / f".{output.stem}-preview.png"
            save_histogram_image(histogram_file(source), preview_path)
            preview = Image.open(preview_path).convert("RGB")
            preview_path.unlink()
            x = column * cell_width + 12
            y = header_height + row_index * cell_height
            sheet.paste(preview, (x, y))
            draw.text((x, y + 263), label, fill="#f9fafb", font=label_font)
            draw.text(
                (x, y + 285),
                f"{source.name} · {source.stat().st_size:,} bytes",
                fill="#9ca3af",
                font=detail_font,
            )

    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()
    visualize_dataset(
        args.manifest,
        args.output,
        split=args.split,
        samples_per_class=args.samples_per_class,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
