# Occlusion Robustness — Report Handoff for TV6

**Project:** What Do Vision Transformers See That CNNs Do Not?

**Owner:** Team Member 1 (TV1).

**Official run:** `full10000_20261105T051900440637Z`.

**Handoff verification date:** 9 November 2026.

This is the academic handoff for a short section in the team's 4–5 page,
double-column report. Sections 1–11 provide evidence and reproducibility details;
Section 12 contains compact text, a table, and a caption to adapt directly.
All experimental numbers below come from the retained official run, not from
published papers. No inference was rerun to prepare this document.

## 1. Research objective

We ask whether ResNet-50 and ViT-B/16 retain different classification performance
when a controlled portion of their input is hidden. Occlusion robustness measures
prediction accuracy and stability after removing spatial evidence, relative to
the same model's clean predictions on the same images. This provides behavioral
evidence complementary to the project's visualizations and representation
analyses. It compares two specific pretrained model-and-preprocessing pipelines;
it is not an isolated test of architecture alone.

## 2. Scientific background

Residual learning provides the architectural basis for ResNet [1], while the
Vision Transformer processes image patches with a Transformer [2]. These papers
identify the model families; they do not supply the results of our experiment.

Zeiler and Fergus [3, Section 4.2 and Fig. 6] examine classifier sensitivity by
covering image regions and observing output changes. That intervention motivates
using missing image evidence as a diagnostic. Our experiment aggregates accuracy
over fixed center/random masks and area levels; it does not reproduce their
sliding occlusion-sensitivity maps.

Naseer et al. [4] investigate occlusion robustness and related properties of
several Vision Transformer models. Their findings motivate our comparison, but
their models and perturbation protocol differ. Our measurements are independent
and should not be presented as a numerical reproduction of their benchmark.

## 3. Experimental setup

### Data and models

- **Cohort:** the committed ImageNet validation analysis manifest, containing
  10,000 unique images from 1,000 classes, exactly 10 images per class. This is
  the complete analysis set, not the historical 5,000-image pilot.
- **Labels:** original zero-based ImageNet indices, 0–999. Join records by
  `image_id`; CSV and Parquet row order need not agree.
- **Checkpoints:** torchvision `ResNet50_Weights.IMAGENET1K_V2` and
  `ViT_B_16_Weights.IMAGENET1K_V1`, respectively. No fine-tuning, weight updates,
  test-time augmentation, or mask optimization was performed in this experiment.
- **Inference:** evaluation mode, with `torch.inference_mode()` scoped to forward
  passes. Batch size 16, CUDA, NVIDIA GeForce RTX 3060 Laptop GPU.

| Preprocessing | ResNet-50 V2 | ViT-B/16 V1 |
| --- | --- | --- |
| Color | RGB | RGB |
| Resize, short side | 232 pixels | 256 pixels |
| Interpolation / antialias | Bilinear / enabled | Bilinear / enabled |
| Center crop | 224×224 | 224×224 |
| Tensor conversion | float values in [0,1] | float values in [0,1] |
| Normalization mean | [0.485, 0.456, 0.406] | Same |
| Normalization std | [0.229, 0.224, 0.225] | Same |

### Occlusion protocol

The operation order is **official resize → official center crop → gray mask →
tensor conversion → normalization → inference**. Normalization is not applied
twice. The clean branch uses the same preprocessing without a mask.

Square masks have RGB fill `(128,128,128)` and nominal area percentages
`s ∈ {0,10,20,30,40,50}`. On a 224×224 crop, the side length is

```text
side(s) = round(224 × sqrt(s / 100)).
```

Pixel rounding makes actual areas approximately 0.000%, 10.047%, 19.930%,
30.152%, 40.187%, and 49.753%. The coordinate CSV records realized areas.
Center masks use `left = top = (224 - side) // 2`. Random mask origins are
sampled uniformly from integer coordinates 0 through `224 - side`, inclusive;
right/bottom bounds are exclusive. All masks stay inside the crop.

Random placement is deterministic: SHA-256 of the UTF-8 string
`"42|<image_id>|<severity>"` initializes a local `random.Random` generator,
which samples left and top. The key excludes model identity and iteration
order. Consequently, the two models receive matching input-space coordinates.
Different severities use separate keys, so their masks are not nested.

**Fairness boundary:** the two official resize operations produce different
fields of view. Matching mask coordinates on their 224×224 inputs therefore
does not guarantee that identical regions of the original image are hidden.
We preserve each checkpoint's preprocessing rather than silently changing the
baseline to impose a common crop.

Clean predictions are evaluated once per model and reused as the 0% point in
both position curves. There are eleven distinct inference conditions per model:
one clean plus five center and five random conditions. Thus the completed run
contains 110,000 prediction rows per model, or 220,000 inference image passes.
The summary still has 24 rows because both position curves include the clean
point. The mask table has 120,000 rows, including 0% for each position.

## 4. Metrics and formulas

Let `N = 10,000`, `y_i` be the ground-truth class, `T_k(m,p,s,i)` the model's
top-k class set, and `ŷ(m,p,s,i)` its Top-1 prediction. Let `s=0` denote the
model's own clean predictions from this run.

```text
Top-k accuracy A_k(m,p,s) = (100/N) × Σ_i 1[y_i ∈ T_k(m,p,s,i)], k ∈ {1,5}.

Absolute drop D_k(m,p,s) = A_k(m,clean,0) - A_k(m,p,s).

Relative drop R_k(m,p,s) = 100 × D_k(m,p,s) / A_k(m,clean,0).

Prediction consistency C(m,p,s)
    = (100/N) × Σ_i 1[ŷ(m,p,s,i) = ŷ(m,clean,0,i)].
```

- Accuracy and consistency are percentages. Absolute drop is in percentage
  points (pp); relative drop is a percentage of clean accuracy.
- A negative drop means an improvement, not an error. Do not apply `abs()`.
- Relative drop is undefined if clean accuracy is zero; the shared helper
  returns `None` in that case. Neither model has a zero clean accuracy here.
- Consistency is not correctness: retaining the same wrong label is consistent.
- `absolute_drop_pp` and `relative_drop_pct` refer to Top-1. The fields
  `absolute_top5_drop_pp` and `relative_top5_drop_pct` report Top-5 drops.
- All comparisons use matching IDs and labels. The original 50,000-image
  baseline accuracy is never subtracted from analysis-set accuracy.

## 5. Verified results

Source of truth: [official metrics.csv](../../results/metrics/experiments/occlusion/full10000_20261105T051900440637Z/final/metrics.csv).
All 24 summary rows were independently recounted from the saved predictions,
including both Top-1/Top-5 drops and prediction consistency.

### Top-1 accuracy across every severity

All entries are percentages; all conditions use the same 10,000 IDs.

| Nominal area | ResNet center | ResNet random | ViT center | ViT random |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 80.80 | 80.80 | 80.75 | 80.75 |
| 10% | 77.65 | 79.14 | 78.81 | 79.37 |
| 20% | 73.67 | 76.36 | 75.46 | 77.34 |
| 30% | 68.30 | 71.60 | 71.72 | 73.31 |
| 40% | 62.97 | 66.55 | 67.75 | 69.53 |
| 50% | 56.84 | 60.48 | 62.50 | 64.40 |

Clean Top-5 accuracy is 95.40% for ResNet and 95.31% for ViT.

### Detailed comparison at nominal 50% area

| Model / position | Top-1 (%) | Top-5 (%) | Top-1 drop (pp) | Relative Top-1 drop (%) | Consistency (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet / center | 56.84 | 79.42 | 23.96 | 29.65 | 62.21 |
| ViT / center | 62.50 | 83.13 | 18.25 | 22.60 | 69.38 |
| ResNet / random | 60.48 | 81.68 | 20.32 | 25.15 | 65.88 |
| ViT / random | 64.40 | 83.84 | 16.35 | 20.25 | 70.94 |

**Direct observations:**

1. Clean Top-1 accuracies differ by only 0.05 pp, with ResNet slightly higher.
   This measured closeness is not a formal matched-accuracy criterion.
2. Accuracy decreases monotonically along each of the four observed curves.
   Monotonicity is an observation, not a constraint enforced by the code.
3. ViT has higher Top-1 accuracy and smaller Top-1 drops at every evaluated
   nonzero severity for both positions.
4. At 50%, ViT's accuracy advantage is 5.66 pp for center and 3.92 pp for random.
   Its reduction in Top-1 drop relative to ResNet is 5.71 pp and 3.97 pp,
   respectively; these differ from raw accuracy gaps because clean accuracies
   differ slightly.
5. At every nonzero level, center masks produce lower accuracy than random
   masks for each model. This does not establish that object centers were
   necessarily occluded; object locations were not annotated in this analysis.

These are point estimates. No confidence intervals, significance tests,
multiple random seeds, or repeated-run variance estimates were produced.

## 6. Discussion

Under the tested protocol, the ViT checkpoint retains classification accuracy
better than the ResNet checkpoint as spatial evidence is removed, despite nearly
equal clean accuracy. The growing separation at larger masks supports the
project's behavioral comparison: similar clean performance does not imply
identical responses to missing input information.

Greater use of remaining context or different feature dependencies are possible
explanations, not measurements made by this experiment. We did not intervene on
attention heads, match training recipes, or establish that attention causes
robustness. Likewise, poorer center-mask performance is compatible with
center-biased object evidence, but we did not measure object-mask overlap.
Visualizations and other experiments may motivate hypotheses; they cannot be
treated as causal explanations of these accuracy curves without additional tests.

## 7. Limitations and threats to validity

1. **Two checkpoints only.** Architecture, pretraining, and training recipes are
   not independently controlled. Avoid universal CNN-versus-Transformer claims.
2. **Different preprocessing.** Short-side resize 232 versus 256 changes the
   observed content. Paired input coordinates do not remove this confound.
3. **One mask sample.** Each image/severity has one random location using seed
   42. Location sensitivity across seeds has not been quantified. Different
   severities use independent deterministic keys rather than nested masks.
4. **Nonuniform spatial coverage.** Uniform valid top-left coordinates do not
   imply uniform pixel coverage. With a 158-pixel square at 50%, all valid
   origins cover the central pixel of the 224×224 input. Random is not a
   center-excluding control.
5. **Artificial perturbation.** A solid gray square introduces a boundary and
   replaces content; it does not cover all natural occlusions. Large masks may
   remove substantial object evidence without changing the original label.
6. **One fixed dataset subset.** Ten images per class yields a balanced
   descriptive comparison, not evidence for all ImageNet images or other
   datasets. No uncertainty intervals or statistical significance are claimed.
7. **Historical clean-prediction discrepancy.** On these IDs, the original
   baseline CSV gives ResNet Top-1 80.82%, versus 80.80% in this run. Exactly
   two Top-1 predictions change:
   `ILSVRC2012_val_00034713_n04552348.JPEG` and
   `ILSVRC2012_val_00030315_n03259280.JPEG`. Seventeen ResNet Top-5 class sets
   differ, but no Top-5 correctness flags change; Top-5 remains 95.40%.
   ViT Top-1 and Top-5 class sets match all 10,000 baseline-subset records.
   The exact cause of the ResNet differences has not been established; do not
   attribute them to a particular numerical backend without evidence. Drops
   correctly use this run's own clean predictions.
8. **Execution provenance.** The run metadata records revision `fd4a8ae...` and
   `git_dirty=true`: the generating notebook was then uncommitted. The retained
   notebook was subsequently committed, and the current local edits preserve
   code ASTs while changing formatting/execution metadata. Some execution counts
   are null despite retained outputs; this is not an immutable execution log.
   Use the saved CSVs and hashes as the numeric evidence. GPU bitwise
   reproducibility and cached checkpoint-file hashes were not established.

Paired bootstrap intervals could later be computed from saved per-image
predictions without model inference. They would not replace a separate
multi-seed evaluation of random-mask placement. Neither analysis was performed
for the results reported here.

## 8. References

Conference years below distinguish the final venue from earlier arXiv releases.
Author lists, titles, venues, and years were checked against the linked primary
sources; no unverified DOI is supplied.

1. **Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.** “Deep Residual
   Learning for Image Recognition.” *Proceedings of the IEEE Conference on
   Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
   [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
   **Supports:** residual-learning architecture and the identity of the ResNet
   family; not the torchvision V2 checkpoint or our occlusion results.
2. **Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
   Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
   Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.** “An Image is Worth
   16x16 Words: Transformers for Image Recognition at Scale.” *International
   Conference on Learning Representations (ICLR)*, 2021.
   [Conference record](https://openreview.net/forum?id=YicbFdNTTy);
   [author paper, identifying ICLR 2021 on its first page](https://arxiv.org/pdf/2010.11929).
   **Supports:** patch-sequence Transformer image classification. Metadata was
   verified from the paper because the OpenReview page required browser
   verification during this handoff. It does not establish our robustness result.
3. **Matthew D. Zeiler and Rob Fergus.** “Visualizing and Understanding
   Convolutional Networks.” *Computer Vision – ECCV 2014*, LNCS 8689,
   pp. 818–833, Springer, 2014.
   [DOI: 10.1007/978-3-319-10590-1_53](https://doi.org/10.1007/978-3-319-10590-1_53);
   [author-hosted paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf).
   **Supports:** input occlusion as a diagnostic of classifier sensitivity;
   distinguish their spatial sensitivity maps from our aggregate robustness test.
4. **Muhammad Muzammal Naseer, Kanchana Ranasinghe, Salman H. Khan, Munawar
   Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang.** “Intriguing Properties of
   Vision Transformers.” *Advances in Neural Information Processing Systems 34
   (NeurIPS)*, 2021.
   [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c404a5adbf90e09631678b13b05d9d7a-Abstract.html).
   **Supports:** prior empirical investigation of ViT occlusion robustness and
   related properties. Our checkpoints, masks, and measurements remain separate.

## 9. Figure for the paper

**Recommended file, exact repository-relative path:**

```text
results/figures/experiments/occlusion/full10000_20261105T051900440637Z/final/accuracy_drop_vs_occlusion.png
```

[Open the figure](../../results/figures/experiments/occlusion/full10000_20261105T051900440637Z/final/accuracy_drop_vs_occlusion.png).

**Suggested caption:** *Top-1 accuracy degradation under square gray occlusion
on 10,000 class-balanced ImageNet validation images. Left: center masks; right:
random masks with deterministic, model-paired input coordinates (seed 42).
The horizontal axis reports nominal mask area; the vertical axis reports the
drop in percentage points from each model's clean accuracy on the same cohort.
ViT-B/16 exhibits smaller drops at every evaluated nonzero level. Each checkpoint
retains its official preprocessing; no confidence intervals are shown.*

**Main message:** the near-equal clean accuracies conceal increasingly different
responses to missing spatial evidence. Lower values indicate better retention
relative to each model's own clean reference. Use the two-panel figure and one
small table in the paper rather than reproducing every handoff table.

The PNG exists and has readable titles, axes, units, and legends. Legibility at
the final typeset size still needs checking by TV6; no vector export was created.

## 10. Cross-experiment integration

| Experiment | Complementary question and safe integration |
| --- | --- |
| Attention Maps | Describe attention patterns; ask whether visual patterns are compatible with perturbation responses without equating attention weights with causal importance. |
| Grad-CAM | Identify class-associated regions; a proposed correspondence with occlusion needs matched images, targets, and quantitative overlap/intervention evidence. |
| Effective Receptive Field (ERF) | Examine response sensitivity across input space; response/layer definitions must be aligned before relating ERF extent to robustness. |
| Texture Robustness | Change appearance cues rather than remove a spatial region; compare trends only after documenting the separate perturbation protocol and cohort. |
| Shape-vs-Texture Bias | Test cue preference with appropriate labeled stimuli; occlusion accuracy alone does not identify a shape or texture mechanism. |
| Adversarial Cropping | Compare partial-evidence perturbations while noting that cropping can change scale, field of view, and visible object content. |
| Centered Kernel Alignment (CKA) | Compare layer representations; representation similarity alone does not establish a cause of occlusion robustness. |

No outcomes from those experiments are asserted here. TV6 should obtain their
verified results from the respective owners. Do not assume shared datasets:
cross-experiment per-image joins require overlapping IDs and compatible targets,
preprocessing, and selection rules. Otherwise present complementary aggregate
evidence with each experiment's own cohort described explicitly.

## 11. Reproducibility and deliverable map

All paths below are relative to the repository root.

| Component | Exact path |
| --- | --- |
| Original generating code | `notebooks/06_occlusion.ipynb` |
| Later extracted reusable module | `experiments/occlusion/occlusion.py` |
| Usage documentation | `experiments/occlusion/README.md` |
| This handoff | `experiments/occlusion/REPORT_HANDOFF.md` |
| Shared model loader | `src/models.py` |
| Shared dataset loader | `src/data.py` |
| Shared evaluation/CSV helpers | `src/evaluation.py` |
| Shared manifest | `data/manifests/analysis_10000_seed42.csv` |
| Sampling/revision/checksum configuration | `configs/analysis_set.json` |
| Shared experiment protocol | `docs/experiment_protocol.md` |

Official metrics-directory prefix:

```text
results/metrics/experiments/occlusion/full10000_20261105T051900440637Z/final/
```

| Retained file under that prefix | Purpose / verified size in records |
| --- | --- |
| `metrics.csv` | 24 model/position/severity summary rows |
| `metadata.json` | Run parameters, checkpoint metadata, environment, and hashes |
| `selected_images.csv` | 10,000 original ID/label/index records |
| `resnet50_predictions.csv` | 110,000 condition-specific predictions |
| `vit_b_16_predictions.csv` | 110,000 condition-specific predictions |
| `mask_coordinates.csv` | 120,000 model-shared masks, including the two 0% positions |

Official figures-directory prefix:

```text
results/figures/experiments/occlusion/full10000_20261105T051900440637Z/final/
```

Retained files: `accuracy_vs_occlusion.png`, `accuracy_drop_vs_occlusion.png`,
and `occlusion_examples.png`. Example images show 30% masks; the quantitative
charts show every severity from 0% to 50%.

The generating notebook is self-contained and calls shared `src/` functions;
it does not import the extracted `occlusion.py`. The module was added afterward
and is not claimed to have generated the original run. It exposes preprocessing,
masking, the batch loop, and metric aggregation, without an automatic run or
file-writing side effect on import.

The run used Python 3.12.7, torch 2.5.1+cu121, torchvision 0.20.1+cu121,
NumPy 2.5.4, pandas 3.1.0, Pillow 13.0.0, PyArrow 26.0.0, and matplotlib 3.11.2,
as recorded in metadata. These are recorded run versions, not a new environment
installation requirement. The normalized-LF SHA-256 of both the complete analysis
manifest and full cohort CSV is
`eea185ba9e733636eb245c8d4eab3137f91af4be3d9a5f3766a48bf11d7b057f`.

Handoff validation checks saved data only: all IDs, labels, correctness flags,
metric formulas, coordinate regeneration, metadata hashes, figure presence,
and module imports/syntax. It does not rerun pretrained inference, independently
re-establish the model's historical numerical execution, or validate new results
from other experiments. The latest `main` has a backward-compatible Arrow-filtered
dataset loader; the original run used the earlier streaming implementation.
The integration preserves that newer shared module rather than replacing it.

## 12. Paper-ready content

Adapt references to the team's bibliography numbering. The following is a compact
section, not a request to include the complete handoff in the paper.

### Methodology paragraph

We evaluated ResNet-50 and ViT-B/16 on a balanced subset of 10,000 ImageNet
validation images, with ten images per class. Both models used fixed torchvision
ImageNet checkpoints without fine-tuning: ResNet-50 IMAGENET1K_V2 and ViT-B/16
IMAGENET1K_V1. Following each checkpoint's official resize and 224×224 center
crop, we applied square gray masks, RGB (128,128,128), covering approximately
0%, 10%, 20%, 30%, 40%, or 50% of input area, before normalization. Masks were
centered or randomly positioned. Random coordinates were deterministic from
seed 42, image ID, and severity, and matched between models. We measured Top-1
and Top-5 accuracy, absolute and relative accuracy drops, and prediction
consistency against each model's clean predictions on the same cohort.

### Results paragraph

Clean Top-1 accuracy was nearly identical: 80.80% for ResNet-50 and 80.75% for
ViT-B/16. Accuracy declined across increasing mask areas for both models and
positions, while ViT retained higher Top-1 accuracy at every nonzero level.
At 50% centered occlusion, accuracies were 56.84% and 62.50%, corresponding to
drops of 23.96 and 18.25 percentage points, respectively. Under 50% random
occlusion, accuracies were 60.48% and 64.40%, with drops of 20.32 and 16.35
percentage points. The accuracy gaps widened with severity in both conditions.
At 50%, Top-5 accuracy also favored ViT: 83.13% versus 79.42% for center masks
and 83.84% versus 81.68% for random masks. These are descriptive point estimates.

### Discussion and Limitations paragraph

The results indicate better accuracy retention by this ViT checkpoint under the
evaluated occlusion protocol, rather than a universal advantage of Transformers
over CNNs. Architecture, checkpoint training, and preprocessing are not isolated.
In particular, different resize operations mean matched input coordinates can
hide different original-image regions. Only one random location was evaluated
per image and severity; spatial pixel coverage is nonuniform, and gray squares
do not represent every natural occlusion. No confidence intervals or statistical
significance are claimed. A small ResNet clean discrepancy from historical
baseline predictions was retained transparently; all drops use the current
run's own clean predictions.

### Concise results table

Top-1 accuracy (%); parentheses show the drop from the same-run clean accuracy
in percentage points. The 0% reference is shared between position curves.

| Model | Clean | Center 50% | Random 50% |
| --- | ---: | ---: | ---: |
| ResNet-50 | 80.80 | 56.84 (23.96) | 60.48 (20.32) |
| ViT-B/16 | 80.75 | 62.50 (18.25) | 64.40 (16.35) |

### Suggested figure caption

*Top-1 accuracy drop under center (left) and random (right) square gray masks
on 10,000 class-balanced ImageNet validation images. Areas are nominal input
percentages; drops are measured in percentage points from each model's own clean
reference. Random coordinates are model-paired with seed 42. ViT-B/16 retains
accuracy better across the evaluated nonzero severities. Official preprocessing
is retained for each checkpoint; no confidence intervals are shown.*
