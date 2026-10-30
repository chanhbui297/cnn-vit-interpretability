# What Do Vision Transformers See That CNNs Do Not?

A Computer Vision final project comparing ResNet50 (a convolutional neural network, or CNN) and ViT-B/16 (a Vision Transformer). The goal is to explore how they use image information and respond to image changes. The selected weights are IMAGENET1K_V2 for ResNet50 and IMAGENET1K_V1 for ViT-B/16.

The repository includes a baseline evaluator for the locally downloaded ImageNet-1K validation split. See [baseline instructions](docs/baseline.md) for full evaluation on all 50,000 validation images. Interpretability and robustness experiments are not implemented yet.

## Verified Baseline Results

Both models were evaluated on the same 50,000 ImageNet-1K validation images.

| Model | Weights | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| ResNet50 | IMAGENET1K_V2 | 80.844% | 95.432% |
| ViT-B/16 | IMAGENET1K_V1 | 81.066% | 95.318% |

The Top-1 gap is 0.222 percentage points in favor of ViT. These measurements do not by themselves establish a matched-accuracy criterion. Per-image predictions and summary CSVs are shared in [the full baseline results](results/metrics/baseline_notebook/full/).

## Project Objectives

- Compare models with similar accuracy to make the analysis more meaningful.
- Use Grad-CAM and attention maps to explore which image regions influence predictions.
- Study effective receptive fields (ERF): which input regions affect model responses.
- Measure robustness to occlusion (hidden image regions) and texture changes.
- Optionally explore whether models rely more on object shape or texture.

These visualizations are complementary tools; attention maps alone are not proof of what causes a prediction.

## Repository Structure

```text
configs/                         # Sampling and experiment settings
data/                            # Local images; approved ID manifests may be tracked
docs/                            # Project notes and methodology
notebooks/                       # Exploratory notebooks
src/                             # Baseline evaluation and manifest generation
experiments/
  gradcam/                       # Grad-CAM comparisons
  attention/                     # ViT attention map analysis
  erf/                           # Effective receptive field analysis
  occlusion/                     # Occlusion robustness experiments
  texture/                       # Texture robustness experiments
  shape_texture_bias/            # Optional shape-versus-texture analysis
results/
  figures/                       # Generated plots (ignored by Git)
  metrics/                       # Approved baseline CSVs; other outputs ignored
report/                          # Report source files and write-up
requirements.txt                 # Python dependencies
```

Empty folders contain a `.gitkeep` placeholder so Git can track them. Raw datasets, caches, credentials, model checkpoints, and virtual environments stay out of Git. Only explicitly approved result CSVs and ID manifests are shared.

## Getting Started

1. Clone this repository and open its folder in your editor.
2. Select the project Python environment in VS Code. Dependencies are listed in `requirements.txt`.
3. Keep the original validation shards under `data/imagenet/data/`; see [data notes](data/README.md). They are not included in Git.
4. Inspect the saved results first. To check local data without inference, run `.\.venv\Scripts\python.exe src/evaluate_baseline.py --check-data`.

Baseline commands, preprocessing, and output details are documented in [docs/baseline.md](docs/baseline.md). Rerunning evaluation is optional and overwrites its result files; preserve the validated CSVs when preparing commits.
