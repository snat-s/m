# Binary CNN

This experiment turns each file into a 256×256 adjacent-byte histogram and
trains a very small CNN to classify it. Cell `(a, b)` is the frequency with
which byte `a` is immediately followed by byte `b`. This representation drops
file position and long-range structure, making it cheap and useful as a
baseline rather than a complete detector.

The included four-class generator is **synthetic** and exists only to validate
the pipeline. Real experiments should use `binary-collect`, which validates
extension-derived labels, removes exact duplicates, and records provenance.

## Quick start

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
uv sync
uv run binary-generate data/synthetic --samples-per-class 100
uv run binary-visualize data/synthetic/manifest.csv runs/dataset.png
uv run binary-train data/synthetic/manifest.csv --output runs/synthetic.pt
uv run binary-predict runs/synthetic.pt /path/to/a/file
```

## Real files (Magika-inspired)

Google's original Magika study trained on 24 million real files from GitHub and
VirusTotal across 113 balanced content types. It used 512 bytes from the start,
middle, and end of each file and evaluated on 1.2 million held-out files. Its
dataset is not publicly downloadable, but the collection principles are
reproducible: collect real files, map extensions to canonical labels, validate
labels without relying on the detectors being benchmarked, deduplicate, balance
classes, and preserve a completely held-out test set.

This collector accepts checked-out public corpora and repositories:

```bash
uv run binary-collect data/real \
  --source tika=/path/to/apache-tika \
  --source file-type=/path/to/file-type \
  --source openpreserve=/path/to/format-corpus \
  --samples-per-class 100 --minimum-per-class 20
uv run binary-expand data/real/manifest.csv data/real-1gb \
  --source poi=/path/to/apache-poi --source pdfjs=/path/to/pdf.js \
  --source codec-corpus=/path/to/codec-corpus --target-mib 1024
uv run binary-visualize data/real/manifest.csv runs/real-dataset.png
uv run binary-train data/real/manifest.csv --output runs/real.pt
uv run binary-train data/real-1gb/manifest.csv --balanced \
  --output runs/real-1gb.pt
uv run binary-train data/real-1gb/manifest.csv --architecture resnet18 \
  --balanced --freeze-backbone-epochs 1 --output runs/real-1gb-resnet18.pt
uv run binary-train data/real/manifest.csv --representation byte-windows \
  --output runs/real-byte-windows.pt
uv run binary-benchmark data/real/manifest.csv \
  --checkpoint runs/real.pt --checkpoint runs/real-byte-windows.pt
```

Useful permissively licensed sources are
[Apache Tika](https://github.com/apache/tika),
[`file-type`](https://github.com/sindresorhus/file-type),
[`filetype.py`](https://github.com/h2non/filetype.py), and the
[OpenPreserve format corpus](https://github.com/openpreserve/format-corpus).
The 1 GiB experiment additionally used Apache POI, Mozilla PDF.js, veraPDF,
LibreOffice regression fixtures, and `imazen/codec-corpus`. Dataset directories
and model checkpoints are gitignored; the repository contains the collectors,
training code, and commands—not the source corpora or generated dataset.
Do not commit or redistribute collected files without checking each source's
license and fixture notices. The manifest retains source, original path, and
SHA-256 for auditability.

### Starter benchmark

On the initial 107-file, 15-class held-out split, both neural models were
trained for 30 epochs with seed 7. Timings are warm, batched CPU measurements
that include file reading and feature extraction:

| Detector | Accuracy | Parameters | Milliseconds/file |
| --- | ---: | ---: | ---: |
| libmagic 5.44 | 71.0% | — | 0.56 |
| Byte-pair histogram CNN, 711-file corpus | 84.1% | 10,631 | 2.81 |
| Byte-pair histogram CNN, 1 GiB corpus | 91.6% | 10,631 | 2.56 |
| Magika-style byte windows | 94.4% | 43,695 | 0.58 |

This test set is far too small for a definitive result. In particular,
libmagic's generic `text/plain` output is counted as unknown rather than mapped
to Markdown, Rust, TypeScript, or YAML. The result is an early comparison, not
a reproduction of Google's large benchmark.

Create a viewable histogram or save the raw normalized histogram array:

```bash
uv run binary-histogram /bin/ls runs/ls.png
uv run binary-histogram /bin/ls runs/ls.npy
```

The default network has roughly 6,400 parameters (the exact count depends on
the class count), automatically uses CUDA when available, and stores labels and
preprocessing settings in its checkpoint. Use `--device cuda`, larger batches,
or a wider model for rented-GPU runs:

```bash
uv run binary-train data/real/manifest.csv \
  --device cuda --batch-size 128 --width 16 --epochs 50 \
  --output runs/real-w16.pt
```

## Real dataset format

Create a CSV whose paths are relative to the manifest:

```csv
path,label,split
files/benign/a.exe,benign,train
files/malware/b.exe,malware,train
files/benign/c.exe,benign,val
files/malware/d.exe,malware,test
```

Keep hashes or families together in one split. Near-duplicate malware in train
and test can produce an impressive but meaningless score. For a serious
malware experiment, also use a time-based held-out test set, report precision,
recall, ROC-AUC/PR-AUC and false-positive rate, and compare against simple
baselines such as file size/entropy, byte n-grams, and a raw-byte model. Only
handle samples from a lawful source in an isolated analysis environment; this
project never executes input files.

## Next experiments

Use a repository-held-out test corpus to measure source leakage, add an unknown
class, and run reproducible sweeps over learning rate, width, byte limit, and
weight decay on a GPU. ResNet-18 support is experimental: ImageNet pretraining
may not transfer to byte-pair maps, and no completed ResNet result is claimed.
