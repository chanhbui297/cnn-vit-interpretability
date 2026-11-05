# Báo cáo TV4 — Texture Robustness và Shape-vs-Texture Bias

## Methodology

TV4 so sánh ResNet-50 `IMAGENET1K_V2` với ViT-B/16 `IMAGENET1K_V1`, giữ nguyên
`weights.transforms()` của torchvision. Cả hai dùng RGB, ImageNet normalization
và crop 224; resize cạnh ngắn là 232 (ResNet) và 256 (ViT) theo checkpoint.

### Shape-vs-Texture Bias

Benchmark `rgeirhos/texture-shape-cue-conflict` có 1.280 ảnh, 16 categories,
shape cue và texture cue. 80 ảnh có hai cue trùng nhau được loại, còn 1.200 ảnh
conflict hợp lệ. Với mỗi output ImageNet-1K, softmax probabilities của các class
trong từng category được lấy **trung bình** theo danh sách indices chính thức
của Geirhos et al.; category có score cao nhất là decision. Shape Bias là
`N_shape / (N_shape + N_texture)`; Neither nằm ngoài mẫu số. Khoảng tin cậy
95% dùng 2.000 bootstrap samples, seed 42.

### Texture Robustness

Protocol cố định dùng manifest nhóm 10.000 ImageNet validation images (10/class).
Gaussian blur có `(kernel, sigma)=(5,1),(9,2),(15,4)`; bilateral filter có
`(d, sigmaColor, sigmaSpace)=(5,25,25),(9,50,50),(13,100,100)`. Các phép biến
đổi xảy ra trên RGB gốc, trước preprocessing, và giữ nguyên kích thước/nhãn.
Metrics dự kiến là Top-1, accuracy drop theo percentage points, prediction
consistency và paired bootstrap CI. Đây là sensitivity to **texture-detail
degradation**, không cô lập texture bias thuần túy.

## Results

Shape-vs-Texture Bias đã chạy full trên RTX 3060, batch 64:

| Model | Valid conflict | Shape | Texture | Neither | Coverage | Shape Bias (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 1.200 | 181 | 671 | 348 | 71,00% | 21,24% (18,52–24,09) |
| ViT-B/16 | 1.200 | 326 | 493 | 381 | 68,25% | 39,80% (36,45–43,12) |

ViT-B/16 cao hơn 18,56 percentage points Shape Bias trên các cue decisions.
Số liệu nguồn là `results/tv4/shape_texture_*.csv`; metadata lưu seed,
preprocessing, checkpoint, CUDA và Git revision.

Texture Robustness chưa có metric: 14 ImageNet validation Parquet shards không
có tại `data/imagenet/data/`. Đã tìm các workspace/caches/config liên quan và
thử `snapshot_download` đúng repo/revision ILSVRC; lượt tải không xác thực trả
`401 GatedRepoError`. Không thay thế ImageNet hay suy diễn kết quả từ baseline.

## Notebook-first execution

`notebooks/shape_texture_bias.ipynb` đã được execute từ đầu đến cuối, không có
cell error. Notebook chứa trực tiếp cue-conflict intuition image, dataset grid
và distribution, model setup, single-image inference thực của cả hai model,
minh họa mapping 16 categories, cache/full inference logic, summary, ba biểu đồ
inline và case studies được chọn bằng prediction records. Cache chỉ được dùng
khi schema/model/mapping khớp; ngược lại notebook gọi utility batch inference,
không gọi CLI ngoài notebook.

`notebooks/texture_robustness.ipynb` cũng execute không lỗi trong state hiện tại.
Model/manifest/protocol có output; cells cần raw ImageNet pixels hiển thị blocker
và bỏ qua an toàn. Khi tài khoản đã được chấp thuận license tải 14 shards vào
đúng đường dẫn, Run All notebook sẽ trực tiếp tạo transformation grids,
single-image prediction, cache, summary và figures.

## Discussion và limitations

Trên benchmark này, ViT-B/16 chọn shape cue thường xuyên hơn ResNet-50 và hai
bootstrap interval không chồng nhau trong run hiện tại. Đây là quan sát cho một
cặp pretrained checkpoint và 16 entry-level categories, không chứng minh mọi
ViT đều shape-biased hơn mọi CNN. Neither đáng kể (29,0% ResNet; 31,75% ViT
trong conflict images) nên cần được báo cáo cùng coverage.

Hai experiments sẽ bổ sung nhau khi Texture Robustness hoàn tất, nhưng shape
bias cao hơn không tự động suy ra accuracy giảm ít hơn dưới blur. Style-transfer
stimuli, resize riêng mỗi checkpoint và bootstrap sampling đều giới hạn diễn
giải nhân quả/khái quát hóa.

## References

1. Geirhos, R. et al. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*. ICLR. https://openreview.net/forum?id=Bygh9j09KX
2. Official mapping and materials: https://github.com/rgeirhos/texture-vs-shape
3. Cue-conflict dataset: https://huggingface.co/datasets/rgeirhos/texture-shape-cue-conflict
4. Torchvision model weights: https://pytorch.org/vision/stable/models.html
