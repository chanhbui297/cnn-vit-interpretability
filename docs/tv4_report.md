# Báo cáo TV4 — Controlled Cue Perturbation và Shape-vs-Texture Bias

## Methodology

TV4 so sánh ResNet-50 `IMAGENET1K_V2` với ViT-B/16 `IMAGENET1K_V1`, giữ nguyên `weights.transforms()` của torchvision: RGB, ImageNet normalization, crop 224 và resize cạnh ngắn 232/256 tương ứng. Không train lại model.

### Baseline cue-conflict

Benchmark `rgeirhos/texture-shape-cue-conflict` có 1.280 ảnh, 16 categories. Loại 80 ảnh có shape cue trùng texture cue, còn 1.200 ảnh conflict hợp lệ. Mỗi output ImageNet-1K được softmax rồi lấy **trung bình** theo official 1.000-to-16 mapping của Geirhos et al. Category score cao nhất được so với shape/texture cue để cho Shape, Texture hoặc Neither. `Shape Bias = N_shape / (N_shape + N_texture)`; Neither nằm ngoài mẫu số, vì vậy báo thêm coverage.

### Controlled Cue Perturbation

**Research question:** *Does a higher shape bias imply stronger reliance on global shape information?* Cùng 1.200 image IDs đi qua clean, Gaussian Blur và deterministic Patch Shuffle trước official preprocessing. Blur dùng `(kernel, sigma)=(7,1,5)` và `(13,3,0)`. Shuffle dùng tile 48 px/32 px; seed ổn định từ `(image_id, 42)` được chia sẻ giữa hai model. Outcome gồm decisions, coverage, clean→perturbed transition và paired-bootstrap CI (4.000 resamples) cho thay đổi Shape Bias.

Pilot phân tầng 160 ảnh (10/shape category) loại shuffle 16 px vì coverage chỉ còn 15,6% cho ResNet và 25,6% cho ViT. Study không trùng TV1 Occlusion, TV2/TV3 Attention/ERF hay TV5 Adversarial Cropping: nó đo paired cue-decision transitions trên cue-conflict, không đo saliency hay crop robustness.

## Results

### Baseline đã xác minh

| Model | Shape | Texture | Neither | Coverage | Shape Bias (95% CI) |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 181 | 671 | 348 | 71,00% | 21,24% (18,52–24,09) |
| ViT-B/16 | 326 | 493 | 381 | 68,25% | 39,80% (36,45–43,12) |

ViT cao hơn 18,56 pp Shape Bias trên cue decisions.

### Controlled Cue Perturbation

| Condition | ResNet Shape Bias / coverage | ViT Shape Bias / coverage |
|---|---:|---:|
| Clean | 21,24% / 71,00% | 39,80% / 68,25% |
| Blur S1 | 48,86% / 54,75% | 66,76% / 58,17% |
| Blur S2 | 72,05% / 42,33% | 83,88% / 48,08% |
| Shuffle 48 px | 8,81% / 66,25% | 13,90% / 61,17% |
| Shuffle 32 px | 7,86% / 57,25% | 9,90% / 52,17% |

Blur S1 thay đổi Shape Bias so với clean **+27,61 pp** cho ResNet (paired 95% CI +24,33–+30,90) và **+26,96 pp** cho ViT (+23,52–+30,39). Blur S2 tăng +50,80 pp/+44,08 pp nhưng có 692/1.200 và 623/1.200 Neither. Shuffle 32 px giảm **−13,38 pp** cho ResNet (−16,09–−10,58) và **−29,90 pp** cho ViT (−33,65–−26,40). Prediction, summary và transition matrices nằm tại `results/tv4/controlled_perturbation_*.csv`.

## Discussion

Kết quả không hỗ trợ suy luận đơn giản “Shape Bias cao hơn ⇒ reliance mạnh hơn vào global shape”. Blur tăng Shape Bias ở cả hai model, nhưng đồng thời giảm coverage và tăng Neither; patch shuffle cho pattern ngược lại. Vì Blur chỉ attenuate high-frequency detail (có thể làm yếu edges) và Shuffle chỉ phá layout (vẫn giữ local tiles, tạo artificial boundaries), đây là evidence về **response sensitivity under interventions**, không phải phép đo trực tiếp global-shape representation.

Điều này phù hợp với cảnh báo của Hermann et al. rằng apparent bias còn phụ thuộc data/training protocol, và với Burgert et al. rằng forced-choice cue-conflict không đủ để kết luận feature reliance. Không tuyên bố novelty hay cơ chế attention/receptive-field. Raw ImageNet validation pixels chưa có do gated data, nên không báo cáo clean-ImageNet classification degradation; workflow ImageNet độc lập vẫn chờ dữ liệu hợp lệ.

## References

1. Geirhos, R. et al. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*. ICLR. https://openreview.net/forum?id=Bygh9j09KX
2. Official mapping and dataset: https://github.com/rgeirhos/texture-vs-shape ; https://huggingface.co/datasets/rgeirhos/texture-shape-cue-conflict
3. Hermann, K. L., Chen, T. & Kornblith, S. (2020). *The Origins and Prevalence of Texture Bias in Convolutional Neural Networks*. NeurIPS. https://proceedings.neurips.cc/paper/2020/hash/db5f9f42a7157abe65bb145000b5871a-Abstract.html
4. Burgert, T., Stoll, O., Rota, P. & Demir, B. (2025). *ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2025/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html
