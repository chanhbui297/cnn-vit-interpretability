# Baseline evaluation

Evaluate only ResNet50 with `ResNet50_Weights.IMAGENET1K_V2` and ViT-B/16 with `ViT_B_16_Weights.IMAGENET1K_V1`. No training or other experiments are included.

## Verified full-validation results

Both models completed evaluation on the same 50,000 ImageNet-1K validation images. The saved prediction CSVs contain 50,000 unique image IDs per model, with ground-truth labels matching the original Parquet dataset. Recounting per-image correctness reproduces every summary metric below.

| Model | Top-1 correct | Top-1 | Top-5 correct | Top-5 |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 V2 | 40,422 | 80.844% | 47,716 | 95.432% |
| ViT-B/16 V1 | 40,533 | 81.066% | 47,659 | 95.318% |

ResNet50 minus ViT Top-1 is **-0.222 percentage points**. Report this measured gap; no matched-accuracy threshold has been defined.

Recorded environment: CUDA on an NVIDIA GeForce RTX 3060 Laptop GPU, batch size 32, PyTorch `2.5.1+cu121`, torchvision `0.20.1+cu121`. Full evaluation uses every validation image, without random sampling or test-time augmentation. Seed 42 for the separate analysis manifest is unrelated to this baseline run. Bit-for-bit reproducibility across devices has not been tested.

Each model uses its official `weights.transforms()`: bilinear resize to 232 for ResNet50 and 256 for ViT, center crop to 224, RGB tensor conversion, and normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Models run one at a time in evaluation mode under `torch.inference_mode()`.

## Dataset and label compatibility

The existing `data/imagenet/data/` folder contains 14 validation Parquet shards with 50,000 images and 50 images per class. Each row has embedded image bytes, an original filename, and an integer ground-truth label. Filenames end in `.JPEG`; a decoded sample is JPEG. Pillow reads the actual embedded format and converts images to RGB in memory without changing dataset files.

Class names are stored in each shard's Hugging Face metadata. Labels are already zero-based (0 through 999); **do not subtract one or remap them**. Both torchvision weight sets have identical output categories. During implementation, all 1,000 label-to-synset associations were checked against the [canonical ImageNet synset order in timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/_info/imagenet_synsets.txt), and each index was compared with the [torchvision category list](https://github.com/pytorch/vision/blob/main/torchvision/models/_meta.py).

Torchvision's category at each index appears among the dataset's comma-separated names, with three verified naming differences:

| Index | Synset in local filenames | Dataset name | Torchvision name |
| --- | --- | --- | --- |
| 134 | n02012849 | crane | crane bird |
| 517 | n03126707 | crane2 | crane |
| 639 | n03710721 | maillot, tank suit | maillot tank suit |

The evaluator checks all shards for consistent metadata, unique image IDs, valid labels, class counts, sorted synset order, and category compatibility. It explicitly checks these three aliases and their synsets. A failed check stops execution before weights are loaded. Evaluation needs no Hugging Face login or network access for the dataset.

## Run full evaluation

Use the existing project environment from the repository root:

```powershell
.\.venv\Scripts\python.exe src/evaluate_baseline.py --check-data
.\.venv\Scripts\python.exe src/evaluate_baseline.py
```

The first command verifies the dataset without inference. The second evaluates all 50,000 validation images for both models. Reduce `--batch-size` from its default of 32 if memory is limited. CUDA is selected automatically when available; otherwise CPU is used. Pretrained weights may download to the standard PyTorch cache; the dataset is never downloaded or changed.

For the self-contained notebook, open `notebooks/01_baseline.ipynb` with the project's `.venv` kernel. Run the setup and data-check cells, then the evaluation and comparison cells. Running the evaluation cell starts the full 50,000-image evaluation directly.

## Outputs

The script saves to `results/metrics/baseline/full/`. The notebook saves to `results/metrics/baseline_notebook/full/`.

- `selected_images.csv`: all 50,000 image IDs, dataset indices, and ground-truth labels.
- `resnet50_predictions.csv` and `vit_b_16_predictions.csv`: per-image ground truth, five predicted class indices, and correctness flags.
- `summary.csv`: completed-model metrics, weights, device, image count, mode, and library versions.
- `comparison.csv` (notebook): signed and absolute Top-1 gap in percentage points.

Each model uses its own `weights.transforms()`, evaluation mode, and `torch.inference_mode()`. Small batches limit memory usage. The first model is released before the second loads. Top-1 and Top-5 accuracy are percentages of correct predictions over all validation images.

Rerunning overwrites that run's CSVs. Interrupted runs may leave incomplete prediction files or a summary for only one completed model. The five validated CSVs under `results/metrics/baseline_notebook/full/` are explicitly allowed in Git; other generated outputs remain ignored. The notebook retains useful tables and the accuracy chart, while embedded validation-image preview output is removed before sharing. Clear that preview output again if you rerun its cell before committing.
