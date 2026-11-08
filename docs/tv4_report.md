# Báo cáo TV4 — Shape-vs-Texture Bias và Texture Robustness

## 1. Mục tiêu và câu hỏi nghiên cứu

TV4 khảo sát cách ResNet-50 và ViT-B/16 đáp ứng với hai loại bằng chứng cue trên cùng benchmark Geirhos Texture–Shape Cue Conflict. Experiment A đo phân bố quyết định Shape/Texture/Neither trên ảnh có hai cue xung đột. Experiment B đo độ ổn định của chính quyết định mapped đó khi chi tiết bề mặt bị suy giảm dần.

Hai câu hỏi liên kết là: (i) checkpoint nào có Shape Bias cao hơn trên cue-conflict images; và (ii) bias này có đi cùng stability cao hơn khi texture detail bị suy giảm hay không. Đây là phép đo cue-conflict được trưởng nhóm phê duyệt, **không phải ImageNet clean Top-1 accuracy benchmark**.

## 2. Thiết lập chung

- **Models:** torchvision ResNet-50 `IMAGENET1K_V2` và ViT-B/16 `IMAGENET1K_V1`; không fine-tune hoặc thay checkpoint.
- **Preprocessing:** mỗi model dùng nguyên `weights.transforms()` chính thức (RGB, resize/crop và ImageNet normalization riêng); transformations được áp dụng trên RGB source trước preprocessing.
- **Dataset:** `rgeirhos/texture-shape-cue-conflict`, test split 1.280 ảnh. Loại 80 ảnh có `shape_cue == texture_cue`; cohort cuối gồm **1.200** ID duy nhất và 16 categories, chính xác cùng cohort với notebook Shape.
- **Mapping/decision:** softmax 1.000 ImageNet outputs được average theo official ImageNet-1K → 16 category mapping của Geirhos et al. Nếu mapped category trùng shape cue là Shape, trùng texture cue là Texture, còn lại là Neither.
- **Hardware/run:** CUDA trên NVIDIA RTX 3060, batch 64; all model-condition cohorts có 1.200 records. Mọi cache được kiểm tra theo IDs, cue labels, condition, mapping-source checksum và metadata.

Shape Bias = `Shape / (Shape + Texture)` là metric có điều kiện; cue-decision coverage = `(Shape + Texture) / 1.200`. Neither không nằm trong mẫu số Shape Bias, vì vậy mọi diễn giải luôn báo coverage và Neither kèm theo.

## 3. Experiment A — Shape-vs-Texture Bias

### Baseline cue-conflict

| Model | Shape | Texture | Neither | Coverage | Shape Bias (95% CI) |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 181 | 671 | 348 | 71,00% | 21,24% (18,52–24,09) |
| ViT-B/16 | 326 | 493 | 381 | 68,25% | 39,80% (36,45–43,12) |

ViT-B/16 có Shape Bias cao hơn ResNet-50 **18,56 percentage points** trên cue decisions. Đây là khác biệt về cue decision trên ảnh stylized; không phải bằng chứng trực tiếp về causal feature reliance hay natural-image accuracy.

### Controlled Cue Perturbation đã xác minh

Experiment A cũng có intervention paired trên cùng 1.200 ảnh: Gaussian Blur S1 `(kernel=7, sigma=1,5)`, Blur S2 `(13, 3,0)`, và deterministic Patch Shuffle 48/32 px.

| Condition | ResNet Shape Bias / coverage | ViT Shape Bias / coverage |
|---|---:|---:|
| Clean | 21,24% / 71,00% | 39,80% / 68,25% |
| Blur S1 | 48,86% / 54,75% | 66,76% / 58,17% |
| Blur S2 | 72,05% / 42,33% | 83,88% / 48,08% |
| Shuffle 48 px | 8,81% / 66,25% | 13,90% / 61,17% |
| Shuffle 32 px | 7,86% / 57,25% | 9,90% / 52,17% |

Blur S1 thay đổi Shape Bias +27,61 pp ở ResNet và +26,96 pp ở ViT, nhưng coverage giảm. Patch shuffle làm Shape Bias giảm. Pattern này cho thấy intervention sensitivity, không hợp thức hoá kết luận rằng Shape Bias cao tự nó biểu thị global-shape reliance mạnh hơn.

## 4. Experiment B — Texture Robustness

### Protocol

Texture Robustness dùng 7 conditions/model: clean; Gaussian S1 `(7, 1,5)`, S2 `(13, 3,0)`, S3 `(19, 4,5)`; Bilateral S1 `(d=5, σcolor=25, σspace=25)`, S2 `(9, 50, 50)`, S3 `(13, 100, 100)`. Gaussian S1/S2 có **cùng parameters và cùng paired predictions** với Controlled Cue Perturbation ở Experiment A; chúng được tái sử dụng có kiểm chứng và không được coi là bằng chứng độc lập. Gaussian S3 và ba Bilateral conditions là inference mới.

Kết quả có **2 models × 1.200 images × 7 conditions = 16.800 prediction records**. Prediction consistency là tỷ lệ mapped 16-category category bất biến so với clean. Shape/Texture retention lấy mẫu số lần lượt là số clean Shape/Texture decisions, không phải 1.200 ảnh.

### Kết quả chính

| Model | Condition | Consistency | Shape Bias | Coverage | Neither | Shape retention | Texture retention |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | Gaussian S1 / S2 / S3 | 46,58 / 25,83 / 19,00% | 48,86 / 72,05 / 71,82% | 54,75 / 42,33 / 33,42% | 45,25 / 57,67 / 66,58% | 83,43 / 69,06 / 47,51% | 42,77 / 17,29 / 11,92% |
| ViT-B/16 | Gaussian S1 / S2 / S3 | 51,50 / 31,17 / 24,00% | 66,76 / 83,88 / 89,37% | 58,17 / 48,08 / 42,33% | 41,83 / 51,92 / 57,67% | 85,28 / 73,62 / 58,59% | 40,37 / 13,39 / 7,10% |
| ResNet-50 | Bilateral S1 / S2 / S3 | 79,33 / 64,50 / 50,00% | 22,89 / 27,49 / 35,38% | 66,25 / 61,83 / 54,42% | 33,75 / 38,17 / 45,58% | 85,64 / 81,22 / 73,48% | 83,76 / 68,41 / 50,97% |
| ViT-B/16 | Bilateral S1 / S2 / S3 | 86,83 / 72,92 / 56,67% | 41,77 / 47,92 / 55,21% | 67,83 / 66,08 / 60,83% | 32,17 / 33,92 / 39,17% | 93,87 / 88,34 / 80,06% | 89,66 / 74,04 / 53,55% |

Các figures mới:

1. `texture_transformation_grid.png` — ảnh gốc và 6 transformations trên 4 cue-conflict examples.
2. `texture_consistency_by_severity.png` — consistency theo severity, tách Gaussian/Bilateral.
3. `texture_decision_distributions.png` — Shape/Texture/Neither và coverage.
4. `texture_decision_retention.png` — Shape và Texture retention.
5. `texture_shape_bias_coverage.png` — Shape Bias đọc cùng coverage.

### Transitions và quan sát định lượng

Ở Gaussian S3, ResNet chuyển Texture → Shape trên 144/1.200 ảnh (12,00%) và Texture → Neither trên 447/1.200 (37,25%). ViT có các transition tương ứng 160 (13,33%) và 298 (24,83%). Ở Bilateral S3, các transition Texture → Neither giảm còn 264 (22,00%) với ResNet và 137 (11,42%) với ViT. Điều này phù hợp với consistency S3 cao hơn của Bilateral (50,00%/56,67%) so với Gaussian (19,00%/24,00%).

## 5. Thảo luận tổng hợp

**Observation.** ViT có Shape Bias clean cao hơn, và trong toàn bộ các severity Texture Robustness nó cũng có consistency cao hơn ResNet (ví dụ Gaussian S3: 24,00% so với 19,00%; Bilateral S3: 56,67% so với 50,00%). Cả hai model đều ít ổn định hơn khi severity tăng; Gaussian gây giảm mạnh hơn Bilateral. Texture retention suy giảm nhanh hơn Shape retention, còn Neither tăng, đặc biệt với Gaussian.

**Interpretation.** Các kết quả bổ sung cho nhau nhưng không nói rằng “ViT có Shape Bias cao hơn nên robust vì architecture Transformer”. Khi Gaussian làm mờ mạnh, Shape Bias tăng song coverage giảm và Neither tăng: một phần apparent shift Texture → Shape đi cùng với nhiều quyết định không còn khớp cue nào. Vì vậy Shape Bias tăng không tương đương shape-cue recognition tốt hơn. Bilateral giữ consistency/coverage cao hơn là evidence rằng dạng perturbation (bao gồm tác động lên edges) quan trọng, không chỉ cường độ ordinal.

**Joint conclusion.** Trên cohort này, ViT-B/16 vừa thiên Shape hơn trong clean decision vừa giữ mapped prediction tốt hơn dưới cả hai family transformations. Tuy nhiên, kết luận đáng tin cậy là về *paired cue-decision stability under these interventions*. Không thể quy toàn bộ khác biệt cho CNN versus Transformer architecture, và không thể suy ra ImageNet classification robustness chuẩn.

## 6. Limitations

- Cue-conflict images stylized, không đại diện toàn bộ ảnh tự nhiên hay ImageNet validation.
- Gaussian/Bilateral không cô lập hoàn toàn texture: cả hai có thể tác động edges, contours và colour regions.
- Shape Bias là conditional decision metric, không phải accuracy.
- Hai checkpoint pretrained có thể khác training recipe ngoài kiến trúc.
- Gaussian S1/S2 được dùng chung với Controlled Cue Perturbation, nên không phải hai tập bằng chứng độc lập.

## 7. Reproducibility và artifacts

- Notebook đã execute: `notebooks/texture_robustness.ipynb`.
- Backend: `experiments/texture/cue_conflict_robustness.py`, `experiments/texture/run_cue_conflict_robustness.py`.
- Predictions/metrics: `results/tv4/texture_predictions.csv`, `texture_summary.csv`, `texture_transitions.csv`, `texture_metadata.json`.
- Shape artifacts giữ nguyên: `shape_texture_*` và `controlled_perturbation_*` trong `results/tv4/`.

Để recompute toàn bộ, dùng `PYTHONPATH=.deps .venv/bin/python experiments/texture/run_cue_conflict_robustness.py --force-recompute`; mặc định notebook/runner chỉ dùng cache khi metadata và toàn bộ cohorts hợp lệ.

## References

1. Geirhos, R. et al. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*. ICLR. https://openreview.net/forum?id=Bygh9j09KX
2. Geirhos et al., texture-vs-shape official code/dataset. https://github.com/rgeirhos/texture-vs-shape ; https://huggingface.co/datasets/rgeirhos/texture-shape-cue-conflict
3. Hermann, K. L., Chen, T. & Kornblith, S. (2020). *The Origins and Prevalence of Texture Bias in Convolutional Neural Networks*. NeurIPS. https://proceedings.neurips.cc/paper/2020/hash/db5f9f42a7157abe65bb145000b5871a-Abstract.html
4. Burgert, T., Stoll, O., Rota, P. & Demir, B. (2025). *ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2025/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html
