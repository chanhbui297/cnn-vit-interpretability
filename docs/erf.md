# ERF chính: 2.000 ảnh, hai crop và hai kiểm tra nhỏ

Mở [04_erf.ipynb](../notebooks/04_erf.ipynb) để chạy ERF 2.000 ảnh và hai probe trên 200 ảnh.
Notebook gọi code trong `experiments/erf/`. `erf_experiment.py` chạy model và tính ERF,
`erf_statistics.py` ghép cặp và tính thống kê, `erf_workflow.py` quản lý quy trình và xuất report,
còn `audit_sensitivity.py` kiểm chứng kết quả của hai probe.

[Archive](../experiments/erf/archive/) lưu bản tutorial 10 ảnh, pilot 100 ảnh và snapshot
2.000 ảnh đã chạy ngày 03/11/2026. Các bản này giữ output và đường dẫn lịch sử để đối chiếu.
Hướng dẫn hiện hành và quy ước bàn giao tập trung trong tài liệu này.

## Mục đích và giới hạn kết luận

ERF bổ sung bằng chứng về phạm vi/phân bố ảnh hưởng lên **feature trung gian**
của ResNet và ViT. Cần trả lời: ảnh hưởng thay đổi theo độ sâu thế nào, khác biệt
giữa hai model có giữ khi cùng crop không, A90 gần nhau có đồng nghĩa phân bố giống nhau không?
Chưa đo class score; không quy kết nguyên nhân dự đoán, shape bias hoặc robustness từ ERF đơn lẻ.

## 1. Kiểm chứng ba run gốc trước khi chạy lại model

Giữ nguyên các thư mục dưới `results/metrics/experiments/erf/`:

- `main_2000_official_20261103T160959_331099Z/final/`
- `main_2000_shared_vit_crop_20261103T161602_476613Z/final/`
- `paired_2000_official_vs_shared_vit_crop_20261103T162207_281699Z/final/`

CSV/JSON/PNG gốc đã được thu hồi từ Drive. Link nguồn và kích thước có trong receipt
của bản phân tích validated. Không chuyển đổi line endings hoặc mở/lưu lại CSV gốc
bằng Excel: SHA-256 provenance kiểm tra đúng byte.

Từ repo root, dùng Python environment của project:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r experiments/erf/requirements-reproduce.txt
python -m experiments.erf.erf_workflow audit
python -m experiments.erf.erf_workflow analyze --write-report
```

`audit` chỉ đọc. `analyze` tạo run ID mới, không ghi đè run gốc; `--write-report`
sinh `report/erf_generated_tables.md` từ kết quả đã kiểm chứng để đối chiếu số liệu.
`report/erf.md` được viết tay và không bị lệnh này ghi đè. Có thể truyền
`--official`, `--shared`, `--paired` bằng đường dẫn `.../final/` nếu di chuyển dữ liệu.

Kiểm tra: cùng 2.000 ID/label/index, đúng 2 ảnh mỗi lớp, checksum cohort/manifest,
16.000 dòng mỗi crop, gradient hữu hạn/dương, metric hợp lệ, checkpoint/transform/target/vị trí,
ViT không đổi giữa hai crop, SHA-256 metric gốc theo paired metadata, summary mean/SD/N,
metric copies, mọi paired difference, 16 Wilcoxon/Holm/CI và 8 crop-sensitivity rows.

NumPy/pandas/SciPy của run gốc là 2.4.6/2.2.3/1.16.3. pandas mới có thể đọc float CSV
khác rất nhỏ và thay đổi tie rank của Wilcoxon sau round 12 chữ số. Nếu audit báo
khác thống kê, không nới tolerance để bỏ qua; dùng environment tái lập rồi kiểm tra lại.
Torch/torchvision và device gradient gốc được giữ trong source metadata; không cần GPU để audit.

## 2. Phương pháp cố định

- Shared loader: ResNet50 IMAGENET1K_V2, ViT-B/16 IMAGENET1K_V1; eval cho phép autograd.
- Cohort: sample không hoàn lại 2 trong 10 ảnh mỗi lớp từ manifest chung, seed 42,
  duyệt lớp 0–999 và sort ID trong mỗi lớp; 2.000 ảnh, không lọc dự đoán đúng/sai.
- Target: vector sau ResNet layer1/2/3/4 hoặc ViT block3/6/9/12, trước encoder LN cuối;
  không chọn CLS. `row=H//2, col=W//2`; ViT token 106. Stage không tương đương về tính toán.
- Scalar gốc: `mean(f.square())`. ERF là tổng absolute input gradient trên RGB [0,1]
  sau resize/crop, trước normalize; chia gradient theo std bằng chain rule.
- A50/A90 là tỷ lệ số pixel ít nhất chứa 50/90% gradient, có thể rời nhau.
  Dispersion là RMS quanh centroid trên [0,1]. Central mass dùng nửa giữa mỗi chiều (~25% diện tích).
- Official giữ resize riêng 232/256 và crop 224. Shared dùng transform ViT 256 cho cả hai.
  Cùng index lưới không đảm bảo receptive center trùng trong tọa độ ảnh gốc.

`compute_erf(..., target_mode='energy'|'absolute', position='center'|'quarter')`
giữ mặc định cũ. `run_experiment` thêm cùng hai tham số và `checkpoint_every=50`.
Metric CSV và metadata mới ghi target_mode, position, chỉ số lưới/token,
thời gian theo model/tổng thời gian, peak CUDA memory và hash các CSV cuối.
Run lịch sử chưa có cột probe được nhận diện bằng target/center_rule legacy;
audit vẫn kiểm tra định nghĩa và vị trí cụ thể.

## 3. Thống kê

Mỗi phép so sánh có 2.000 cặp ảnh; ghép theo ID/stage, không theo thứ tự hàng.
Hiệu số ViT−ResNet; A90 dùng điểm phần trăm, dispersion dùng RMS chuẩn hóa.
Bootstrap paired-image percentile 95%, 20.000 lần, seed 42 cộng chỉ số nhóm.
Không giữ quota lớp khi resample; CI pointwise, chưa hiệu chỉnh multiplicity.
Wilcoxon hai phía round 12 chữ số, bỏ zero; all-zero trả p=1. Holm chung 16 p-value
(2 crop × 2 metric × 4 stage). Wilcoxon không kiểm định trực tiếp mean.

Crop sensitivity đo `(ViT−ResNet)shared − (ViT−ResNet)official` trên cùng ảnh,
chỉ diễn giải mô tả. Không suy ra tương đương từ CI chứa 0, không viết p=0 khi underflow.

## 4. Hai probe trên 200 ảnh: đã chạy và kiểm chứng

Run Colab nguồn: sensitivity_analysis_200_20261104T135825_603042Z.
Bản kiểm chứng local: validated_sensitivity_200_20261105; cùng 200 ID, 194 lớp,
1.600 dòng mỗi probe, 3.200 dòng ghép cặp. Hash nguồn, checkpoint/loader/crop,
target/vị trí, runtime và toàn bộ 16 hàng mean/CI/model-change đã khớp khi tái tính.

Kết luận giới hạn: khoảng cách ở stage 1–3 giữ hướng dưới hai probe; A90 gap stage 4
ở tâm gần 0 nhưng lên 7,925 điểm phần trăm tại quarter (CI95% mô tả [6,753; 9,062]).
Đổi vị trí làm thay đổi cả geometry/biên ảnh; không suy ra kiến trúc là nguyên nhân duy nhất.
Xem report cho bảng đầy đủ và so sánh trên cùng 200 ảnh, thay vì trộn mean của 2.000 và 200 ảnh.

Các hướng dẫn GPU dưới đây dành cho tái lập; không cần chạy lại để hoàn tất phần số liệu hiện tại.

Subset: `random.Random(42).sample(ID-sorted primary 2000 cohort, 200)`, sau đó sort
dataset index. Không phải 200 dòng đầu, không lấy mẫu lại cho mỗi model/probe.
CSV sensitivity_selected_images.csv trong run validated chốt danh sách này.

Chỉ shared_vit_crop, giữ checkpoint/layers/seed:

- absolute_center: scalar `mean(abs(f))`, vị trí tâm như cũ.
- energy_quarter: scalar `mean(f²)`; row/col `round(0.25*(grid_size-1))`.

Baseline lấy từ shared metrics gốc trên đúng 200 ID. Hai probe mới có run ID riêng,
mỗi probe 1.600 dòng. So sánh thay đổi gap theo từng ảnh, CI mô tả; không thêm kiểm định
vào họ Holm chính. Quarter vẫn là vị trí lưới, không sửa triệt để sự lệch receptive center.

### Tái lập trên Colab

1. Đồng bộ notebook chính, `experiments/erf/`, `src/`, config, manifest và các run cần dùng vào project trên Drive.
   Đặt `PROJECT_ROOT` theo thư mục project và `DATA_DIR` theo vị trí 14 shard ImageNet. Có thể đặt shard trong `/content` để giảm I/O.
2. Mở `notebooks/04_erf.ipynb` và chọn GPU nếu tính lại gradient. Giữ `RERUN_MAIN=False` để dùng kết quả 2.000 ảnh hiện có.
   Ô đầu thiết lập phiên bản NumPy, pandas và SciPy của run gốc. Khởi động lại kernel nếu Colab yêu cầu sau khi cài.
3. Điền `ABSOLUTE_RUN_DIR` và `QUARTER_RUN_DIR` bằng đường dẫn tới hai thư mục `final/` của probe đã chạy để chỉ phân tích.
   Để tính lại gradient cho hai probe, bật `RUN_SENSITIVITY`. Đặt `ALLOW_DOWNLOAD=True` nếu cache thiếu checkpoint chính thức.
   `RUN_BENCHMARK=True` đo thời gian và peak VRAM trên 50 ảnh. Run bị ngắt cần chạy lại với run ID mới.
4. Chạy các ô theo thứ tự. Mỗi probe có 1.600 dòng, còn phân tích độ nhạy xuất bảng 16 so sánh,
   dữ liệu ghép cặp và biểu đồ thay đổi gap.

Các lệnh tương ứng khi chạy từ repo root:

```powershell
python -m experiments.erf.erf_workflow benchmark --data-dir PATH_TO_SHARDS --allow-download
python -m experiments.erf.erf_workflow sensitivity --data-dir PATH_TO_SHARDS --allow-download
python -m experiments.erf.erf_workflow sensitivity-analysis --absolute-run PATH_TO_FINAL --quarter-run PATH_TO_FINAL
```

Khi có kết quả đã kiểm chứng, dùng `write_report(PROJECT_ROOT, analysis_dir, sensitivity_dir=...)`
để sinh `report/erf_generated_tables.md`, rồi cập nhật số liệu trong `report/erf.md` bằng tay.

Tái kiểm chứng run hiện có sau khi chạy lệnh sensitivity-analysis:

```powershell
python -m experiments.erf.audit_sensitivity --source results/metrics/experiments/erf/sensitivity_analysis_200_20261104T135825_603042Z/final --recomputed results/metrics/experiments/erf/validated_sensitivity_200_20261105/final
```

Audit không sửa run Colab nguồn. Nó kiểm tra byte/hash và so bảng tính lại, tạo
audit.json, sensitivity_profiles.csv và figure so sánh ba probe trên đúng 200 ID.
Gói `outputs/erf_colab_update.zip` là snapshot của lần chạy gốc. Audit dùng gói đó
để đối chiếu hash sáu module đã thực thi. Đường dẫn trong gói phản ánh cấu trúc repo khi chạy.

## 5. Đầu ra và bàn giao

Mọi run mới dùng metrics `results/metrics/experiments/erf/<run_id>/final/`,
figures `results/figures/experiments/erf/<run_id>/final/`; checkpoint và maps preview
ở intermediate/. Maps/weights/ảnh gốc không nằm trong final.

Run validated có audit.json, source hashes, CSV metric/tables, distribution summary,
subset 200 ảnh và cross_method_handoff.csv. Figures có mean gap/CI kèm zoom tầng cuối,
ECDF theo ảnh. Hình ERF bốn stage của ba ảnh preview mỗi crop nằm trong run gốc.
Ví dụ cực trị chọn từ metric chỉ là case study; chưa có map nếu không thuộc preview.

[Report](../report/erf.md) trình bày phương pháp, kết quả theo tầng/crop, kiểm tra độ nhạy và giới hạn.
Hai run tổng hợp dùng trong report là `validated_2000_20261104_ready` và
`validated_sensitivity_200_20261105`.

### Phối hợp Attention và Grad-CAM

Dùng bản sao [cross_method_handoff.csv](../results/metrics/experiments/erf/validated_2000_20261104_ready/final/cross_method_handoff.csv).
Mỗi ví dụ có hai hàng model, cùng ID, nhãn và crop. Các vai trò gồm ảnh gần median A90 gap tầng cuối,
hai cực trị và ba preview gốc. Dùng các cực trị làm case study.

- Đọc ảnh qua `src.data.iter_analysis_images`, ghép theo `image_id` và kiểm tra nhãn.
  Dùng transform chính thức của ViT cho cả hai model với RGB, bilinear antialias, resize 256 và crop 224.
  Ghi `crop_sha256` của mảng RGB uint8 HWC contiguous trước normalize và giữ map trong tọa độ crop 224×224.
- Với Grad-CAM, dùng cùng ground-truth class, ghi target layer và cách reshape token ViT.
  Nếu khảo sát predicted class, ghi riêng kết quả theo class đó.
- Với Attention, ghi block, query token, head hoặc cách gộp head, và cách rollout nếu có.
  ERF tâm dùng patch token 106. Các cột `official_pred_top1` và `official_top1_correct` thuộc official crop.
  Tính prediction với shared crop nếu muốn giải thích quyết định của model trên crop chung.
- Điền `attention_observation`, `gradcam_observation`, `crop_sha256`, `integration_status`,
  đường dẫn hình, cấu hình target và checkpoint. Giữ ID, nhãn, model và điều kiện theo handoff gốc.

Đặt các map của cùng ảnh cạnh nhau và mô tả độ phân tán, mức trùng vùng nổi bật và khác biệt do target/layer/query.
Gắn nhãn ví dụ điển hình hoặc ngoại lệ. Chỉ tính tương quan pixel hoặc IoU khi đã thống nhất tọa độ,
độ phân giải, target và chuẩn hóa. Dùng thí nghiệm perturbation để kiểm tra nhận xét về robustness.

Các ID cực trị ngoài preview có thể cần thêm map minh họa. Tạo run riêng bằng `run_experiment`
với `preview_ids` gồm các ID đó, `energy/center`, `shared_vit_crop` và seed 42. Giữ cohort minh họa riêng
với cohort dùng để tính kết quả chính và độ nhạy.
