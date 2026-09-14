import csv

from binary_cnn.synthetic import LABELS, generate_dataset


def test_generation_is_balanced_and_split(tmp_path) -> None:
    manifest = generate_dataset(
        tmp_path / "dataset", samples_per_class=10, min_size=64, max_size=80
    )
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(LABELS) * 10
    for label in LABELS:
        label_rows = [row for row in rows if row["label"] == label]
        assert {row["split"] for row in label_rows} == {"train", "val", "test"}
        assert all((manifest.parent / row["path"]).is_file() for row in label_rows)
