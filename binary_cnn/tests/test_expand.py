import csv
import hashlib
from pathlib import Path

from binary_cnn.expand import expand_training_set


def test_expansion_adds_only_training_rows_and_preserves_holdouts(tmp_path) -> None:
    base = tmp_path / "base"
    (base / "files" / "pdf").mkdir(parents=True)
    rows = []
    for split in ("train", "val", "test"):
        data = f"%PDF-1.7\n{split} payload".encode()
        path = Path("files") / "pdf" / f"{split}.pdf"
        (base / path).write_bytes(data)
        rows.append(
            (
                str(path),
                "pdf",
                split,
                "base",
                str(path),
                hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = base / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label", "split", "source", "original_path", "sha256"))
        writer.writerows(rows)

    source = tmp_path / "source"
    source.mkdir()
    (source / "new.pdf").write_bytes(b"%PDF-1.7\na genuinely different file")
    expanded = expand_training_set(
        manifest, tmp_path / "expanded", [("new-source", source)], target_bytes=1_000
    )
    with expanded.open() as handle:
        expanded_rows = list(csv.DictReader(handle))

    assert [row for row in expanded_rows if row["split"] == "val"] == [
        dict(
            zip(
                ("path", "label", "split", "source", "original_path", "sha256"), rows[1]
            )
        )
    ]
    assert [row for row in expanded_rows if row["split"] == "test"] == [
        dict(
            zip(
                ("path", "label", "split", "source", "original_path", "sha256"), rows[2]
            )
        )
    ]
    assert sum(row["split"] == "train" for row in expanded_rows) == 2
