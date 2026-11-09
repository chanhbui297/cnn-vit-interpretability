# Occlusion Robustness

Compare ResNet-50 and ViT-B/16 under square gray occlusion. The experiment has
already been run on the balanced 10,000-image analysis set.

## Report handoff

TV6 should use [REPORT_HANDOFF.md](REPORT_HANDOFF.md) for verified methodology,
results, references, limitations, a suggested paper figure, and compact text for
the final report. The original full-run outputs remain unchanged.

## Files and responsibilities

- `occlusion.py`: reusable preprocessing split, masking, batched experiment loop,
  and metric aggregation extracted from `notebooks/06_occlusion.ipynb`.
- [06_occlusion.ipynb](../../notebooks/06_occlusion.ipynb): the original self-contained notebook with
  methodology, executed results, visual checks, plots, and discussion. It is
  preserved and does not yet import this module.
- `../../src/`: shared model loading, data loading, inference and CSV helpers.

Importing `occlusion.py` does not load models, run inference, or write files.
It has no command-line entry point. The notebook and module currently contain
equivalent experiment logic; keep them synchronized if methodology changes.

## Method

- Models: ResNet50 `IMAGENET1K_V2`; ViT-B/16 `IMAGENET1K_V1`.
- Dataset: committed `analysis_10000_seed42.csv`, 10 images per ImageNet class.
- Areas: 0%, 10%, 20%, 30%, 40%, 50%; side `round(224 * sqrt(area / 100))`.
- Positions: center or random; RGB fill `(128, 128, 128)`.
- Stage: official model-specific resize/center crop, then mask, then normalize.
- Seed: 42. Random coordinates are keyed by seed, image ID and severity, so the
  models receive the same coordinates independently of iteration order.
- Clean is evaluated once, then reused at 0% for both position curves.

Areas are approximate after pixel rounding. Matching coordinates on model inputs
does not guarantee matching regions of the original image because resize differs.
Random square origins are uniform within valid bounds; pixel coverage is not
uniform, especially for large masks.

## Functions

| Function | Purpose |
| --- | --- |
| `resize_crop(image, transform)` | Official spatial preprocessing, returning a PIL crop |
| `normalize_image(image, transform)` | Tensor conversion and official normalization |
| `mask_box(image_id, severity, position, seed=42)` | Stable square bounds, exclusive right/bottom |
| `apply_occlusion(image, image_id, severity, position, seed=42)` | Return a masked copy of a 224x224 RGB crop |
| `run_experiment(records, models, transforms, batch_size=16, seed=42)` | Run clean plus ten masked conditions; return prediction lists by model |
| `summarize_predictions(predictions)` | Return 24 metric rows; accepts in-memory or CSV prediction records |

## Usage

Run from the repository root, or add that root to `sys.path` in a notebook.
Use the existing environment, local dataset and cached model weights. This
example starts inference only when `run_experiment` is explicitly called.

```python
from src.models import load_model
from src.data import load_analysis_manifest
from src.evaluation import save_predictions
from experiments.occlusion.occlusion import run_experiment, summarize_predictions

records = load_analysis_manifest(limit=100)  # Set to 10000 for an explicit full run.
models, transforms = {}, {}
for name in ("resnet50", "vit_b_16"):
    models[name], transforms[name], metadata = load_model(name, device="cuda")

predictions = run_experiment(records, models, transforms, batch_size=16, seed=42)
metrics = summarize_predictions(predictions)

# Choose a NEW run directory; do not reuse the existing full-run directory.
# save_predictions(predictions["resnet50"], new_run_dir / "resnet50_predictions.csv")
# metrics.to_csv(new_run_dir / "metrics.csv", index=False, mode="x")
```

The module returns results without saving. For a research run, preserve both
models' predictions, selected IDs, mask coordinates and metadata using the
[shared protocol](../../docs/experiment_protocol.md). Figures and the complete
saving workflow remain in the notebook. See its setup for global RNG seeding.

You can also recompute metrics from existing CSVs without inference:

```python
import csv
from pathlib import Path
from experiments.occlusion.occlusion import summarize_predictions

run_dir = Path("results/metrics/experiments/occlusion/full10000_20261105T051900440637Z/final")
predictions = {}
for name in ("resnet50", "vit_b_16"):
    with (run_dir / f"{name}_predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions[name] = list(csv.DictReader(stream))
metrics = summarize_predictions(predictions)
```

## Existing full run

Run ID: `full10000_20261105T051900440637Z`.

| Model | Clean Top-1 | Center 50% Top-1 | Random 50% Top-1 |
| --- | ---: | ---: | ---: |
| ResNet-50 | 80.80% | 56.84% | 60.48% |
| ViT-B/16 | 80.75% | 62.50% | 64.40% |

Metrics, predictions, cohort, masks and metadata are under
`results/metrics/experiments/occlusion/<run_id>/final/`; PNG figures are under
`results/figures/experiments/occlusion/<run_id>/final/`. These paths are relative
to the repository root. Old pilot output files have been removed.

The module was added after this run; the saved run was produced by the notebook.
Do not attribute the old results to a rerun of the extracted module. One random
seed was used; no bootstrap confidence intervals have been calculated. See the
notebook's Discussion for limitations and small ResNet clean prediction
differences relative to the historical baseline.
