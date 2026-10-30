# Shared 10,000-image analysis manifest

Use one shared set for subsequent interpretability and robustness work: **10,000 existing validation images, exactly 10 from each of the 1,000 classes**. This is an ID manifest, not a second image dataset. The full baseline remains a separate 50,000-image evaluation.

## Files and source

- `data/manifests/analysis_10000_seed42.csv`: the authoritative image selection.
- `configs/analysis_set.json`: seed, counts, sorting rules, source paths, pinned revision, and SHA-256 checksums.
- `src/create_analysis_manifest.py`: standalone creation and verification script; imports no baseline evaluation code and loads no models.
- `results/metrics/baseline_notebook/full/selected_images.csv`: the existing 50,000-row source manifest, shared with the validated baseline results.

The raw validation shards remain in `data/imagenet/data/`. They come from `ILSVRC/imagenet-1k`, pinned to revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`. The configuration records the SHA-256 hashes published by Hugging Face for all 14 shards at that revision. Verification checks the local shard bytes against those hashes and checks every source ID and label against the Parquet columns. It does not download, extract, move, or duplicate images. Teammates need authorized local access to the original dataset; it is not included in Git.

## Sampling method

1. Verify the source manifest checksum and all 50,000 local records. Require 1,000 classes, 50 images per class, and unique image IDs.
2. Sort source records by `image_id`, a stable unique identifier. Preserve each original `dataset_index`, filename, and zero-based `label`.
3. Create one local `random.Random(42)` generator. Visit class labels in ascending order from 0 to 999. Sample 10 images without replacement from each class's ID-sorted records.
4. Sort the selected rows by `image_id` and write the columns `dataset_index,image_id,label` using UTF-8 and LF newlines.
5. Require 10,000 unique existing IDs, exactly 10 images per class, and the pinned output checksum. An existing different manifest is reported, never overwritten.

The selection was generated with Python 3.12.7. Use the committed CSV as the shared selection; do not independently resample per experiment. The seed controls selection only. Each model should still use its own official preprocessing. Source and output checksums normalize CRLF to LF **in memory**, so Git's Windows/Linux line-ending conversion does not change verification. The validated source CSV itself is not rewritten.

## Reproduce and verify

Run from the repository root in the existing environment. Only the Python standard library and the already-required PyArrow package are used.

```powershell
# Create the manifest if absent; otherwise compare without overwriting it.
.\.venv\Scripts\python.exe src/create_analysis_manifest.py

# Regenerate in memory and compare with the local CSV; write nothing.
.\.venv\Scripts\python.exe src/create_analysis_manifest.py --check

# After Commit 2 exists, also compare with the manifest stored in HEAD.
.\.venv\Scripts\python.exe src/create_analysis_manifest.py --check-committed
```

These commands hash about 6.69 GB of local shards to verify the pinned revision, but do not decode images or run inference. The committed comparison uses read-only `git show`; it does not stage, commit, or push. If there is no manifest in HEAD yet, it reports that Commit 2 is needed. Differences report row counts, missing/extra IDs, duplicates, changed labels or indices, and ordering changes; a mismatch exits unsuccessfully.

## Load images using the manifest

The columns and IDs match the existing baseline loader. `dataset_index` is the original global row index across sorted shards, not a new subset index. The manifest is sorted by ID; streaming the Parquet files yields selected images in original dataset order. Use `image_id` to join results, rather than assuming CSV and loader row positions are identical. Do not feed the 10,000-image subset into the unchanged full-baseline evaluator, which intentionally checks for 50,000 predictions.

This small example uses the same decoding and preprocessing as the baseline. It reads Parquet in small batches and decodes only selected images. No model weights or images are downloaded, and no image files are written. Run `--check` above first to verify the local dataset and full selection.

```python
import csv
from io import BytesIO
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq
from torchvision.models import ResNet50_Weights

# Run from the repository root.
manifest_path = Path("data/manifests/analysis_10000_seed42.csv")
selected_labels = {}
with manifest_path.open(newline="", encoding="utf-8") as stream:
    for record in csv.DictReader(stream):
        selected_labels[record["image_id"]] = int(record["label"])


def load_analysis_images(transform):
    files = sorted(Path("data/imagenet/data").glob("validation-*.parquet"))
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                image_id = row["image"]["path"]
                if image_id in selected_labels:
                    label = row["label"]
                    if label != selected_labels[image_id]:
                        raise ValueError("Manifest and dataset labels differ.")
                    with Image.open(BytesIO(row["image"]["bytes"])) as image:
                        image_tensor = transform(image.convert("RGB"))
                    yield image_tensor, label, image_id


# transforms() does not load a model or download weights.
transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
images = load_analysis_images(transform)
image_tensor, label, image_id = next(images)
print(image_id, label, image_tensor.shape)  # (3, 224, 224)
images.close()
```

For ViT use `ViT_B_16_Weights.IMAGENET1K_V1.transforms()` instead. The same image IDs and ground-truth indices apply to both models. Only the manifest, configuration, script, and documentation are shared in Git; raw images, caches, and checkpoints remain excluded.
