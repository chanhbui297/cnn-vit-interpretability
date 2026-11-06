# What Do Vision Transformers See That CNNs Do Not?

A Computer Vision final project comparing ResNet50 (a convolutional neural network, or CNN) and ViT-B/16 (a Vision Transformer). The goal is to explore how they use image information and respond to image changes. The selected weights are IMAGENET1K_V2 for ResNet50 and IMAGENET1K_V1 for ViT-B/16.

The repository includes a baseline evaluator for the locally downloaded ImageNet-1K validation split. See [baseline instructions](docs/baseline.md) for full evaluation on all 50,000 validation images. The [ERF tutorial](notebooks/04_erf.ipynb) provides a step-by-step Vietnamese notebook for member 3, with [setup and methodology notes](docs/erf.md). Other interpretability and robustness experiments are not implemented yet.

## Verified Baseline Results

Both models were evaluated on the same 50,000 ImageNet-1K validation images.

| Model | Weights | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| ResNet50 | IMAGENET1K_V2 | 80.844% | 95.432% |
| ViT-B/16 | IMAGENET1K_V1 | 81.066% | 95.318% |

The Top-1 gap is 0.222 percentage points in favor of ViT. These measurements do not by themselves establish a matched-accuracy criterion. Per-image predictions and summary CSVs are shared in [the full baseline results](results/metrics/baseline_notebook/full/).

## Project Objectives

Compare ResNet50 and ViT-B/16 to investigate **what vision transformers see that
CNNs do not**, using three directions: interpretability, robustness, and
representation analysis. Use the verified 50,000-image baseline as a reference
and the shared 10,000-image analysis manifest for subsequent ImageNet experiments.
Report the measured accuracy gap when interpreting differences between models.

**Core tasks**

- **Attention Maps:** visualize ViT attention to explore relationships between image regions.
- **Grad-CAM:** compare class-specific attribution maps for ResNet50 and ViT-B/16.
- **Effective Receptive Field (ERF):** study which input regions affect selected responses in each model.
- **Occlusion Robustness:** measure prediction accuracy changes when image regions are hidden.
- **Texture Robustness:** measure prediction accuracy changes under agreed texture transformations.

**Extensions**

- **Shape-vs-Texture Bias:** investigate reliance on object shape versus texture using appropriate stimuli and labels.
- **Adversarial Cropping:** evaluate robustness to crops selected under an agreed search strategy and budget.
- **Layer-wise Centered Kernel Alignment (CKA):** compare learned representations across layers of the two models.

These are planned experiments; the shared infrastructure is available, but the
experiment implementations are not yet included. Prioritize core tasks before
extensions. Experiment-specific methodology must be agreed separately; see the
[shared experiment protocol](docs/experiment_protocol.md).

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
  adversarial_cropping/          # Optional cropping extension
  cka/                          # Optional representation analysis
results/
  figures/                       # Final experiment figures may be tracked
  metrics/                       # Baseline CSVs and final experiment CSV/JSON
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

## Shared experiment infrastructure — Phase 1

Shared modules provide consistent model loading, image access, and accuracy
calculation for all team members. Experiment algorithms are not implemented yet.

| File | Purpose | Output |
| --- | --- | --- |
| `src/models.py` | `load_model("resnet50")` or `load_model("vit_b_16")` with the fixed baseline weights | Model in eval mode, official transform, and metadata |
| `src/data.py` | `load_analysis_manifest()` reads the image selection; `iter_analysis_images(records)` reads selected images from Parquet | Tuples of `(PIL RGB image, label, image_id)` |
| `src/evaluation.py` | `evaluate()` runs batched inference; `accuracy_drop()` compares against clean predictions on the same IDs; `save_predictions()` writes CSV | Predictions, Top-1/Top-5 accuracy in percent, absolute drop in pp, and relative drop in percent |
| `docs/experiment_protocol.md` | Dataset, seed, ID joins, result storage, and notebook examples | Shared guidelines for the team |

Workflow: **select IDs → read RGB images → load model/transform → implement
the experiment → calculate and save results**. For Grad-CAM/ERF, use the model
directly with autograd; do not use the inference function to calculate gradients.

Importing modules or loading a model does not write result files.
`save_predictions()` writes CSV and refuses overwrites by default. Store final
deliverables in `results/metrics/experiments/<experiment>/<run_id>/final/`, with
the corresponding figure folders under `results/figures/experiments/`.
See the [protocol and examples](docs/experiment_protocol.md).

Verification on the same five images with both models confirmed correct IDs
and labels, baseline preprocessing and Top-5 predictions, working autograd, and
CSV joins by ID. The temporary verification script and generated CSV/JSON files
were removed after checking. These checks are not research experiment results;
the original baseline files remain unchanged.
