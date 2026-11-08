# Bàn giao TV4 cho reviewer

## Trạng thái: hoàn thành

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Shape-vs-Texture baseline | Hoàn thành | 1.200 valid cue-conflict images; `shape_texture_predictions.csv` / `shape_texture_summary.csv` |
| Controlled Cue Perturbation | Hoàn thành | 12.000 rows, paired transitions/figures |
| Texture Robustness | Hoàn thành | **16.800** rows = 2 models × 1.200 images × 7 conditions |
| Executed research notebook | Hoàn thành | `notebooks/texture_robustness.ipynb`: 10 executed code cells, 12 inline PNG outputs, no error outputs |
| Report tổng hợp | Hoàn thành | `docs/tv4_report.md` |

## Review nhanh

- Dataset: Geirhos Texture–Shape Cue Conflict test split; 1.280 source images, loại 80 same-cue, còn 1.200 valid IDs. Set ID bằng chính Shape/Controlled notebook (hash trong `texture_metadata.json`).
- Models/checkpoints/preprocessing/mapping: nguyên vẹn `ResNet50_Weights.IMAGENET1K_V2`, `ViT_B_16_Weights.IMAGENET1K_V1`, torchvision transforms và official 1.000→16 Geirhos mapping.
- Conditions: clean; Gaussian `(7,1,5)`, `(13,3,0)`, `(19,4,5)`; Bilateral `(5,25,25)`, `(9,50,50)`, `(13,100,100)`.
- Gaussian S1/S2 là exact parameter match với Controlled Cue Perturbation. 7.200 validated rows (2 models × 1.200 × clean/S1/S2) được reused; 9.600 new rows (Gaussian S3 + Bilateral S1–S3) chạy CUDA. Final output kiểm tra 14 model-condition cohorts × 1.200 unique IDs.

## Kết quả headline

- Clean Shape Bias: ResNet **21,24%**, ViT **39,80%**.
- Gaussian S3 consistency: ResNet **19,00%**, ViT **24,00%**; coverage lần lượt **33,42%**, **42,33%**.
- Bilateral S3 consistency: ResNet **50,00%**, ViT **56,67%**; coverage **54,42%**, **60,83%**.
- Do Shape Bias tăng trong Gaussian đồng thời Neither tăng/coverage giảm, không diễn giải đó là accuracy hoặc shape recognition tốt hơn.

## Artifacts

- `notebooks/shape_texture_bias.ipynb` — retained verified Shape results.
- `notebooks/texture_robustness.ipynb` — executed, cache-aware notebook with image grid, direct single-image inference, inline result figures and six paired qualitative cases.
- `experiments/texture/cue_conflict_robustness.py` — transformations, cohort/cache validation, metrics and transitions.
- `experiments/texture/run_cue_conflict_robustness.py` — resumable full runner and PNG generation.
- `results/tv4/texture_predictions.csv`, `texture_summary.csv`, `texture_transitions.csv`, `texture_metadata.json`.
- `results/tv4/figures/texture_*.png`.
- `docs/tv4_report.md` — integrated academic Vietnamese report.

## Cách tái lập / review

```bash
PYTHONPATH=.deps .venv/bin/python experiments/texture/run_cue_conflict_robustness.py
```

Mặc định chỉ cache-hit khi ID hash, conditions và mapping-source hash hợp lệ. Chạy lại inference từ đầu:

```bash
PYTHONPATH=.deps .venv/bin/python experiments/texture/run_cue_conflict_robustness.py --force-recompute
```

Mở notebook là xem được outputs đã lưu; để recompute qua notebook, đổi `FORCE_RECOMPUTE = True` ở Full Experiment cell. Không commit dataset/cache model/raw images.

## Hạn chế cần giữ trong paper

Cue-conflict là dữ liệu stylized; blur không cô lập texture; Shape Bias không phải classification accuracy; training recipe checkpoints có thể khác; khác biệt không thể quy riêng cho CNN/ViT architecture. Gaussian S1/S2 không được đếm như evidence độc lập với controlled experiment.
