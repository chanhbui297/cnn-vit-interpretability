# Shared experiment protocol (Phase 1)

## Fixed baseline

Use `src.models.load_model`: `resnet50` means torchvision
`ResNet50_Weights.IMAGENET1K_V2` (`resnet50-11ad3fa6.pth`); `vit_b_16` means
`ViT_B_16_Weights.IMAGENET1K_V1` (`vit_b_16-c867db91.pth`). Do not replace these
with timm checkpoints or torchvision DEFAULT aliases. Models start in eval mode;
parameters remain available to autograd. Cached weights are required by default;
network download needs explicit `allow_download=True`.

Reuse each weight's official `weights.transforms()`: RGB, bilinear resize of the
short edge to 232 (ResNet) or 256 (ViT), antialias enabled, center crop 224x224,
conversion to float [0,1], mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225].
Do not apply normalization twice. The different resize sizes mean that even
224x224 model inputs cover different fields of view. Specify whether a change
acts on original RGB images or model-input coordinates, how coordinates map
back to the source image, and how each model's clean reference is defined.
Any additional common-view comparison must be reported separately from baseline.

## Dataset and cohort

The shared selection is `data/manifests/analysis_10000_seed42.csv`: 10,000 IDs,
1,000 classes, 10/class. Use the committed file; do not resample per experiment.
It contains original dataset indices, IDs and zero-based labels. Never subtract
one from labels. Details and checksums are in `configs/analysis_set.json` and
[analysis_set.md](analysis_set.md). Full source/revision verification is available
with `python -B src/create_analysis_manifest.py --check-committed` (no inference).

`load_analysis_manifest(limit=5)` validates the manifest before selecting five
rows for debugging. This first-N subset is not representative research data.
`iter_analysis_images` returns `(PIL_RGB_image, label, image_id)`, checks labels,
and decodes only selected images from the original 14 Parquet shards. Pass
`data_dir=...` to use another machine's dataset root; no images are copied.
Consume the iterator fully so missing-ID validation runs. Its output order is
Parquet order, not manifest order. The lightweight loader does not hash 6.69 GB
on each call; use the manifest verification command when onboarding a dataset.

Always join by `image_id`, checking uniqueness and equal labels. Never zip two
CSVs by row position. Keep the same image cohort across model comparisons.
Subset selection, exclusions and any jointly-correct cohort must be reported.
The old ignored `pilot_seed42` outputs are not this analysis cohort.

## Randomness

Use seed 42 by default and record it. Before stochastic code, seed the sources
actually used: `random.seed(42)`, `numpy.random.seed(42)`,
`torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` when using CUDA.
Record worker seeds if adding parallel data loading. For paired perturbations,
generate one parameter table keyed by image ID and condition, then share it
between models; sequentially consuming one RNG across two model loops does not
give paired perturbations. Record package versions and device. A seed alone
does not guarantee bitwise equality across devices or library versions.

## Inference and gradients

`evaluation.evaluate` batches transformed samples and scopes inference mode only
to forward passes. It expects a model already in eval mode, returns prediction
rows and percentage Top-1/Top-5 metrics, and does not save automatically.

Grad-CAM and ERF must run outside `torch.inference_mode()` and `torch.no_grad()`.
Use the model returned by `load_model` directly; `model.eval()` is compatible
with gradients. Build fresh normal tensors for attribution (not tensors created
inside inference mode); set `requires_grad_(True)` when input gradients are needed.
Do not route gradient methods through `evaluation.evaluate`.

## Metrics and joins

Accuracy is `100 * correct / N`, expressed in percent. For each model and each
condition, compare perturbed predictions with clean predictions on exactly the
same IDs. `accuracy_drop(clean_rows, changed_rows, model_name=...)` selects those
IDs from the full clean baseline, checks labels, and rejects duplicate IDs,
mixed conditions and explicit model mismatches. For old baseline CSVs without
`model_name`, the caller must select the matching model's filename.

- Absolute accuracy: e.g. clean 80%, changed 60%.
- Absolute accuracy drop: `80 - 60 = 20` percentage points (pp).
- Relative accuracy drop: `100 * (80 - 60) / 80 = 25%`.

Do not take `abs()` of the difference: improvement gives a negative drop.
Relative drop is undefined when clean accuracy is zero; the helper returns None.
Report clean accuracy on the cohort, not full-50k accuracy as a subset reference.
Top-1 and Top-5 metrics are available through `accuracy_from_predictions`.

## Deliverables

Use a unique run ID; never point new runs at existing baseline output folders.

```text
results/metrics/experiments/<experiment>/<run_id>/final/
    <model_name>_predictions.csv
    metrics.csv
    metadata.json
results/figures/experiments/<experiment>/<run_id>/final/
    <descriptive_name>.png  # also jpg, svg, pdf
```

Only CSV/JSON metrics and final figure formats in these final folders are
eligible for normal Git tracking. Review with `git status` before handing off.
Intermediate files outside final, dataset, weights, arrays and caches remain
ignored. Do not embed images/features in CSV or JSON to bypass these exclusions.

Prediction schema (`save_predictions`, overwrite disabled by default):
`image_id,model_name,label,pred_top1,pred_top2,pred_top3,pred_top4,pred_top5,top1_correct,top5_correct,experiment,condition,severity`.
Empty severity means not applicable; record units/meaning in metadata.
The unique key is image_id + model_name + experiment + condition + severity.
Filter to one model/condition/severity before computing a cohort accuracy/drop.

Metadata JSON must record experiment/run ID, seed, manifest path and SHA-256
(normalize CRLF to LF), evaluated ID list or cohort manifest/hash, model metadata
from `load_model`, parameters, batch size, package versions, device, and code Git
revision plus whether the tree was dirty. Record exclusions and selection rules.
For methods without class predictions, define a small method-specific metric
schema while preserving model/cohort/condition provenance. Figures need captions
with model, condition, target, and image ID where applicable.

## Notebook example

Run from the repository root. If the kernel starts in `notebooks/`, insert the
repository root in `sys.path` first (e.g. `sys.path.insert(0, str(Path.cwd().parent))`).

```python
import csv
from pathlib import Path
from src.models import load_model
from src.data import load_analysis_manifest, iter_analysis_images
from src.evaluation import evaluate, save_predictions, accuracy_drop

records = load_analysis_manifest(limit=5)
model, transform, metadata = load_model("resnet50")
rows, metrics = evaluate(
    model, transform, iter_analysis_images(records),
    model_name="resnet50", batch_size=2, experiment="example",
)
save_predictions(rows, "results/metrics/experiments/example/run1/final/resnet50_predictions.csv")
with Path("results/metrics/baseline_notebook/full/resnet50_predictions.csv").open(newline="") as f:
    clean = list(csv.DictReader(f))
drop = accuracy_drop(clean, rows, model_name="resnet50")
```

The same API works with `vit_b_16`; use a fresh image iterator for each model.
The example compares clean inputs with clean predictions, so its expected drop
is zero. It does not implement a robustness experiment. This is an example;
the output path is created only if you run it and call `save_predictions`.

## Phase 1 verification record

Verification used the first five manifest records with both cached baseline
models. Exactly five RGB images were decoded. Labels matched the manifest and
saved baseline; transforms matched the official weight transforms; all five
ranked predictions per image matched the full-baseline CSV. Input gradients were
finite and nonzero for both models after inference. CSV round-trip and ID joins
passed even with rows reversed. Missing IDs, duplicate IDs, changed labels,
explicit model mismatches, overwrite protection and drop formulas were checked.
The temporary verification script and generated smoke CSV/JSON files were
removed after verification; no research experiment results were removed.

## Methodology still to agree separately

TV1 and each experiment owner must separately agree occlusion masks/fill/size,
texture transformation and severity, shape/texture stimuli and label mapping,
attention extraction/aggregation, Grad-CAM target layer/class/token reshaping,
ERF response definition and aggregation, adversarial crop search/budget, and CKA
layer/features/pooling/centering/cohort. Phase 1 does not prescribe those choices.
In torchvision ViT, attention weights are not returned by default; extraction
changes must preserve model logits. Core tasks take priority over extensions.
