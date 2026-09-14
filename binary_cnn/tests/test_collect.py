import csv
import zipfile

from binary_cnn.collect import collect_dataset


def test_collection_validates_deduplicates_and_records_provenance(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(7):
        (source / f"valid-{index}.pdf").write_bytes(
            b"%PDF-1.7\n" + bytes((index,)) + b" real payload bytes"
        )
    (source / "wrong.pdf").write_bytes(b"this is not actually a PDF")
    (source / "duplicate.pdf").write_bytes((source / "valid-0.pdf").read_bytes())
    with zipfile.ZipFile(source / "document.docx", "w") as archive:
        archive.writestr("word/document.xml", "<document />")

    manifest = collect_dataset(
        tmp_path / "output",
        [("fixtures", source)],
        samples_per_class=7,
        minimum_per_class=7,
    )
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 7
    assert {row["label"] for row in rows} == {"pdf"}
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert {row["source"] for row in rows} == {"fixtures"}
    assert len({row["sha256"] for row in rows}) == 7
