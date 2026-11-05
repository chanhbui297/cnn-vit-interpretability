# Bàn giao TV4 cho reviewer

## Trạng thái cuối

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Shape-vs-Texture Bias | Hoàn thành | Full benchmark 1.280 ảnh, 2.560 prediction rows, figures và metadata |
| Texture Robustness | Chưa chạy full | Script/notebook đầy đủ; raw ImageNet 14 shards không hiện diện và source gated trả 401 |
| Notebooks | Đã execute | Shape: 10 code cells, 8 inline figures, không error; Texture: 8 code cells, không error, blocker hiển thị inline |
| Báo cáo tiếng Việt | Hoàn thành có điều kiện | `docs/tv4_report.md`; không đưa ra metric Texture khi chưa có dữ liệu |

## Rework notebook-first

`notebooks/shape_texture_bias.ipynb` là workflow thực nghiệm đầy đủ: cue-conflict
intuition image, dataset grid/distribution, model/preprocessing setup,
single-image inference từ hai model, minh họa mapping ImageNet→16 categories,
cache hoặc full batch inference trực tiếp, Shape Bias/CI, ba chart và case-study
grid được chọn có quy tắc từ prediction records.

`notebooks/texture_robustness.ipynb` cùng có workflow thực thi trực tiếp: protocol
parameters, manifest/model setup, grid transformation, single-image prediction,
cache/full evaluation và failure analysis. Trong workspace hiện tại các cell cần
pixel ImageNet thông báo và bỏ qua an toàn, không sinh prediction hoặc metric giả.

## Kết quả Shape-vs-Texture Bias

- ResNet-50: Shape Bias **21,24%** (95% bootstrap CI 18,52–24,09); 181 Shape / 671 Texture / 348 Neither.
- ViT-B/16: Shape Bias **39,80%** (95% bootstrap CI 36,45–43,12); 326 Shape / 493 Texture / 381 Neither.
- ViT-B/16 cao hơn **18,56 percentage points** trên 1.200 cue-conflict images hợp lệ.

## Cách chạy lại

```bash
# Shape — dataset public/cache. Notebook tự dùng cache prediction hợp lệ nếu có.
HF_TOKEN='' HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH=.deps python3 -m nbconvert --to notebook --execute --inplace notebooks/shape_texture_bias.ipynb

# Texture — chỉ sau khi tài khoản đã được cấp quyền ImageNet và 14 shards xuất hiện.
PYTHONPATH=.deps python3 -m nbconvert --to notebook --execute --inplace notebooks/texture_robustness.ipynb
```

Hoặc mở notebook trong VS Code/Jupyter với Python environment có dependencies
`requirements.txt` (cộng `datasets`, `opencv-python-headless`, `nbformat`,
`nbconvert`) và chọn Run All.

## Blocker Texture có kiểm chứng

Không có `data/imagenet/data/validation-00000-of-00014.parquet` đến shard 13
trong workspace/caches đã kiểm tra. Đã thử `snapshot_download` đúng repo/revision
ILSVRC với token vô hiệu hóa; Hugging Face trả `401 GatedRepoError`. Cần đăng
nhập tài khoản đã được chấp thuận license và tải đúng shards, không thay bằng
dataset khác. Khi dữ liệu hợp lệ có mặt, Run All Texture notebook sẽ tạo demo,
cache predictions, CSV summary và figures từ cùng manifest 10.000 IDs.

## Deliverables và tin nhắn bàn giao

- Notebooks: `notebooks/shape_texture_bias.ipynb`, `notebooks/texture_robustness.ipynb`.
- Code: `experiments/shape_texture_bias/`, `experiments/texture/`.
- Kết quả: `results/tv4/`, figures tại `results/tv4/figures/`.

> TV4 đã rework notebooks theo notebook-first và execute Shape-vs-Texture Bias
> đầy đủ: ResNet-50 Shape Bias 21,24%, ViT-B/16 39,80% (ViT cao hơn 18,56 pp).
> Notebook Shape có dataset/prediction/figures/case studies inline. Texture có
> workflow trực tiếp nhưng chưa chạy full vì ImageNet gated shards thiếu; cần
> đăng nhập tài khoản đã có license và đặt đúng 14 shards tại `data/imagenet/data/`.
