# Bàn giao TV4 cho reviewer

## Trạng thái

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Shape-vs-Texture Bias | Hoàn thành | 1.280 ảnh; 2.560 prediction rows, figures và metadata |
| Texture Robustness | Chưa chạy full | Script hoàn chỉnh và smoke-test đã xác nhận thiếu ImageNet shards |
| Notebooks | Hoàn thành có điều kiện | Shape notebook chứa output full; texture notebook ghi rõ blocker và lệnh chạy lại |
| Báo cáo tiếng Việt | Hoàn thành có điều kiện | `docs/tv4_report.md`, không diễn giải texture khi chưa có dữ liệu |

## Kết quả chính

- ResNet-50: Shape Bias 21,24% (95% bootstrap CI 18,52–24,09), 181 Shape / 671 Texture / 348 Neither.
- ViT-B/16: Shape Bias 39,80% (95% bootstrap CI 36,45–43,12), 326 Shape / 493 Texture / 381 Neither.
- ViT-B/16 cao hơn 18,56 điểm phần trăm trên benchmark Geirhos cue-conflict.

## Deliverables và tái chạy

- Script: `experiments/shape_texture_bias/run_shape_texture_bias.py`, `experiments/texture/run_texture_robustness.py`.
- Kết quả Shape: `results/tv4/shape_texture_predictions.csv`, `shape_texture_summary.csv`, `shape_texture_metadata.json` và `results/tv4/figures/`.
- Notebooks: `notebooks/shape_texture_bias.ipynb`, `notebooks/texture_robustness.ipynb`.

```bash
# Shape (token cũ hết hạn: vô hiệu hóa token nhưng dataset vẫn public)
HF_TOKEN='' HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH=.deps python3 experiments/shape_texture_bias/run_shape_texture_bias.py --allow-download --batch-size 64

# Texture: chỉ chạy sau khi nhóm khôi phục đúng 14 ImageNet validation shards
PYTHONPATH=.deps python3 experiments/texture/run_texture_robustness.py --data-dir data/imagenet/data --allow-download --batch-size 64
```

## Lưu ý cần reviewer xử lý

`data/imagenet/data/validation-00000-of-00014.parquet` đến shard 13 không có
trong workspace. Đây là ImageNet gated dataset và token Hugging Face hiện tại
đã hết hạn; không thể thay bằng data khác mà vẫn giữ protocol nhóm. Sau khi dữ
liệu được đặt đúng đường dẫn, chạy lệnh texture ở trên sẽ tạo CSV và figures.

## Tin nhắn bàn giao ngắn

> TV4 đã hoàn tất Shape-vs-Texture Bias trên 1.280 Geirhos cue-conflict images:
> ResNet-50 đạt Shape Bias 21,24%, ViT-B/16 đạt 39,80% (ViT cao hơn 18,56 pp).
> Code, notebook, CSV, figures và report nằm trong `experiments/`, `notebooks/`,
> `results/tv4/`, `docs/tv4_report.md`. Texture Robustness đã có script đầy đủ
> nhưng chưa thể chạy vì thiếu 14 ImageNet validation shards; khôi phục chúng vào
> `data/imagenet/data/` rồi chạy lệnh trong `docs/tv4_handover.md`.
