# ImageNet-1K validation data

Source: [ILSVRC/imagenet-1k on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Only the validation split is downloaded: 50,000 images and their ground-truth labels across 1,000 classes. No training or test files are included.

## Local files

- `imagenet/data/validation-00000-of-00014.parquet` through `validation-00013-of-00014.parquet`: 14 validation shards, totaling 6,693,093,726 bytes (about 6.23 GiB).
- `imagenet/README.md`: the original dataset card, including the class label mapping.
- `imagenet/.cache/huggingface/`: download metadata maintained by Hugging Face.

The Parquet files retain the supplied image bytes and integer ground-truth labels. Images are not resized, re-encoded, or extracted. Labels use the dataset's zero-based class mapping (0 through 999).

## Download method

The official `huggingface_hub.snapshot_download` tool uses the existing local Hugging Face login. The account must first have access to this gated dataset. No token belongs in this repository.

The download completed on 2026-10-31 using Python 3.12 and the following call:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    allow_patterns=["data/validation-*.parquet", "README.md"],
    local_dir="data/imagenet",
    max_workers=2,
)
```

Run from the repository root. Repeating the call resumes or reuses local files. The filename filter excludes the training and test splits.

## Verification

Verified on 2026-10-31 using PyArrow to read Parquet metadata and the label column:

- Exactly 14 validation shards, totaling 6,693,093,726 bytes; no training or test shards.
- 50,000 image records and 50,000 non-missing ground-truth labels.
- 1,000 distinct labels (0 through 999), with exactly 50 images per class.
- All shards include embedded image bytes and the 1,000-class name mapping in their metadata.
- Every shard's SHA-256 checksum matches the source file at the pinned Hugging Face revision.

Checks used existing local tools; no packages were installed. The dataset card supplies the mapping from class indices to names.

`data/imagenet/` is ignored by Git. This README is tracked separately. Do not commit or redistribute the dataset files; follow the source dataset's access terms. No inference or training is part of this download.
