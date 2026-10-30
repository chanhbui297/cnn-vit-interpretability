# What Do Vision Transformers See That CNNs Do Not?

A university Computer Vision final project comparing ResNet (a convolutional neural network, or CNN) and Vision Transformer (ViT) models with similar classification accuracy. The goal is to explore how they use image information and respond to image changes.

This repository currently contains only the project structure and basic documentation. Models, datasets, and experiment code will be added later.

## Project Objectives

- Compare models with similar accuracy to make the analysis more meaningful.
- Use Grad-CAM and attention maps to explore which image regions influence predictions.
- Study effective receptive fields (ERF): which input regions affect model responses.
- Measure robustness to occlusion (hidden image regions) and texture changes.
- Optionally explore whether models rely more on object shape or texture.

These visualizations are complementary tools; attention maps alone are not proof of what causes a prediction.

## Repository Structure

```text
configs/                         # Future experiment settings
data/                            # Local datasets (ignored by Git)
docs/                            # Project notes and methodology
notebooks/                       # Exploratory notebooks
src/                             # Future reusable Python code
experiments/
  gradcam/                       # Grad-CAM comparisons
  attention/                     # ViT attention map analysis
  erf/                           # Effective receptive field analysis
  occlusion/                     # Occlusion robustness experiments
  texture/                       # Texture robustness experiments
  shape_texture_bias/            # Optional shape-versus-texture analysis
results/
  figures/                       # Generated plots (ignored by Git)
  metrics/                       # Generated measurements (ignored by Git)
report/                          # Report source files and write-up
requirements.txt                 # Planned Python dependencies
```

Empty folders contain a `.gitkeep` placeholder so Git can track them. Keep datasets, model checkpoints, and generated outputs out of Git; keep source code and documentation tracked.

## Getting Started

1. Clone this repository and open its folder in your editor.
2. Read the objectives and use `docs/` to record the planned comparison and evaluation methods.
3. When implementation begins, create a Python virtual environment and install the packages listed in `requirements.txt`. No installation is needed to browse this scaffold.
4. Later, place datasets in `data/`, experiment work in the matching `experiments/` folder, and generated figures and metrics in `results/`.

There are no runnable experiments yet. No datasets or model weights are included.
