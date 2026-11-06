# Bàn giao TV4 cho reviewer

## Trạng thái

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Baseline Shape-vs-Texture | Hoàn thành | 1.200 conflict images, 2.560 clean prediction rows |
| Controlled Cue Perturbation | Hoàn thành | 12.000 rows, paired CIs, transitions, figures, executed notebook |
| Texture Robustness trên clean ImageNet | Chưa chạy full | 14 raw validation shards gated/không có; không tạo metric giả |

## Controlled Cue Perturbation

Research question: **Does a higher shape bias imply stronger reliance on global shape information?** Full study dùng blur S1/S2 và deterministic patch shuffle 48/32 px trên cùng 1.200 cue-conflict IDs. Clean parity tái lập đúng baseline: ResNet 181/671/348 và ViT 326/493/381 cho Shape/Texture/Neither.

- Blur tăng Shape Bias nhưng giảm coverage: ResNet 21,24% → 48,86%/72,05%; ViT 39,80% → 66,76%/83,88%.
- Shuffle giảm Shape Bias: ResNet 8,81%/7,86%; ViT 13,90%/9,90% (48/32 px).
- Kết luận: đây là response sensitivity, **không** là chứng cứ trực tiếp rằng Shape Bias cao hơn đồng nghĩa reliance global shape mạnh hơn.

Pilot 160 ảnh loại shuffle 16 px vì coverage quá thấp. Artifacts:

- Notebook: `notebooks/shape_texture_bias.ipynb` (đã execute hoàn chỉnh).
- Utility: `experiments/shape_texture_bias/controlled_perturbation.py`.
- CSV: `results/tv4/controlled_perturbation_{predictions,summary,transitions}.csv`.
- Metadata: `results/tv4/controlled_perturbation_metadata.json`.
- Figures: `results/tv4/figures/controlled_shape_bias_and_coverage.png`, `controlled_transition_heatmaps.png`.

## Cách review / chạy lại

Mở notebook với environment có `requirements.txt`, `datasets`, `opencv-python-headless`, `nbformat`, `nbclient`; Run All dùng cache đã xác minh. Đổi `FORCE_RECOMPUTE_CONTROLLED=True` để chạy lại 1.200 IDs × 5 conditions × 2 models trực tiếp từ notebook. Mọi transform được xác định bởi image ID + seed 42 và shared giữa model.

## Limitation cần giữ

Blur không xóa hoàn toàn texture và cũng có thể làm yếu shape edges. Patch Shuffle không xóa hoàn toàn shape, giữ local tiles nhưng phá context và tạo tile boundaries. ImageNet clean-data check chưa thể chạy vì raw pixels gated; không thay bằng baseline CSV.
