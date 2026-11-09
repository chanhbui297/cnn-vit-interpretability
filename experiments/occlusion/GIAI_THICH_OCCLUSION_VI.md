# Hiểu thí nghiệm Occlusion: từ một tấm ảnh đến kết quả báo cáo

**Câu hỏi của mình: khi ảnh bị che một phần, ResNet và ViT còn đoán đúng được bao nhiêu?**

Đọc phần 1–4 để hiểu thí nghiệm trước. Phần 5 giải thích code; phần 6–8 giải thích kết quả và sản phẩm. Chưa cần nhớ tên hàm ngay từ đầu.

## 1. Bắt đầu bằng một ví dụ

Giả sử mình có ảnh một chiếc xe buýt. Dataset đã cho biết tên đúng của ảnh là “xe buýt”. Tên đúng này gọi là **label**.

Mình có hai model đã được train sẵn: **ResNet-50** và **ViT-B/16**. Mỗi model nhận ảnh và đoán nó thuộc loại nào trong 1.000 **classes** của ImageNet. Một class là một loại, chẳng hạn một loài động vật hoặc một loại phương tiện.

Mình làm lần lượt:

1. Đưa ảnh chưa che cho mỗi model đoán. Đây là ảnh **clean**.
2. Lấy bản sao của ảnh, che một vùng bằng một hình vuông màu xám. Đây là ảnh **occluded**.
3. Đưa ảnh đã che cho mỗi model đoán lại.
4. Đối chiếu dự đoán với label để biết đúng hay sai.
5. Làm tương tự với nhiều ảnh rồi đếm tỷ lệ đúng.

Ví dụ minh họa, không phải số liệu thực nghiệm:

| Ảnh đưa vào | ResNet đoán | ViT đoán | Label thật |
| --- | --- | --- | --- |
| Xe buýt chưa che | Xe buýt — đúng | Xe buýt — đúng | Xe buýt |
| Cùng ảnh, bị che một phần | Xe tải — sai | Xe buýt — đúng | Xe buýt |

Một ảnh chỉ là ví dụ. Để so sánh model, mình thực hiện trên cùng **10.000 ảnh**.

**Occlusion** là che ảnh. **Robustness** ở đây nói về khả năng model vẫn phân loại được khi input bị che. Mình chỉ kiểm tra model đã có, không train lại hoặc fine-tune.

## 2. Mình che ảnh như thế nào?

### Hình vuông màu xám là gì?

**Mask** là vùng hình vuông được tô đè lên ảnh bằng RGB `(128,128,128)`. Phần ảnh nằm dưới mask bị thay bằng màu xám; các phần khác giữ nguyên.

Hãy tưởng tượng đặt một miếng giấy vuông màu xám lên tấm ảnh. Mình không cắt ảnh thành hai phần và cũng không thu nhỏ cả ảnh lại.

### Center và random là gì?

Đây là **chỗ đặt miếng giấy**, không phải hai loại model.

- **Center:** luôn đặt ở chính giữa tấm ảnh, bất kể đối tượng nằm ở đâu.
- **Random:** chọn một vị trí ngẫu nhiên nhưng miếng giấy phải nằm hoàn toàn trong ảnh.

```text
Center                         Random: một ví dụ
┌─────────────┐                ┌─────────────┐
│             │                │ █████       │
│    █████    │                │ █████       │
│    █████    │                │             │
│             │                │             │
└─────────────┘                └─────────────┘
```

Cùng mức che thì miếng giấy của center và random có cùng kích thước, chỉ khác vị trí. Random cũng có thể rơi gần giữa ảnh.

### 10%, 20%, … là gì?

**Severity** là mức độ che, tính theo **diện tích** ảnh đưa vào model:

- 0%: không che, dùng làm clean reference.
- 10%: che khoảng một phần mười diện tích.
- 50%: che khoảng một nửa diện tích.

Mình thử `0%, 10%, 20%, 30%, 40%, 50%`, cho cả center và random.

Ảnh tại bước che có kích thước 224×224. Với 50%, cạnh hình vuông là khoảng 158 pixels. Không dùng cạnh 112: hình vuông 112×112 chỉ chiếm 25% diện tích ảnh 224×224.

Công thức trong code là:

```python
side = round(224 * math.sqrt(severity / 100))
```

Căn bậc hai xuất hiện vì diện tích hình vuông bằng cạnh nhân cạnh. Cạnh phải là số pixel nguyên nên diện tích thực chỉ xấp xỉ mức yêu cầu. File kết quả có lưu diện tích thực.

### Tại sao random vẫn cần seed?

Nếu ResNet bị che một chỗ và ViT bị che chỗ khác trên cùng ảnh, phép so sánh bị ảnh hưởng bởi vị trí che.

Mình dùng **seed 42**, cùng image ID và severity để tạo tọa độ. Kết quả là cùng ảnh, cùng mức che thì hai model dùng cùng tọa độ mask trên input 224×224. Chạy lại quy tắc đó vẫn ra tọa độ cũ.

Mỗi ảnh và severity chỉ dùng **một random mask**. Random ở 10% và 20% có thể ở hai vị trí khác nhau; mask lớn không bắt buộc bao quanh mask nhỏ.

## 3. Từ ảnh gốc đến input của model

Ảnh trên máy có thể có nhiều kích thước. Mình cần chuẩn bị nó theo đúng cách pretrained model đã được train.

| Bước | Làm gì? | Vì sao cần? |
| --- | --- | --- |
| 1. Đọc RGB | Mở ảnh màu với ba kênh Red, Green, Blue | Có ảnh để xử lý |
| 2. Resize | Thay kích thước, giữ tỷ lệ ảnh; cạnh ngắn thành 232 cho ResNet, 256 cho ViT | Làm đúng preprocessing đi kèm checkpoint |
| 3. Center crop | Cắt ô 224×224 ở giữa ảnh đã resize | Đưa về kích thước input dùng trong thí nghiệm |
| 4. Mask | Tô hình vuông xám, hoặc giữ nguyên nếu severity 0% | Tạo ảnh cần kiểm tra robustness |
| 5. Tensor conversion | Đổi ảnh thành mảng số, pixel từ 0–255 thành số thực 0–1 | Model tính toán trên tensor |
| 6. Normalization | Điều chỉnh các số pixel bằng mean/std cố định | Đưa input về thang giá trị model đã quen lúc train |
| 7. Inference | Chạy model, lấy dự đoán | Biết model đoán class nào |

**Bước 2–3 thay đổi kích thước/vùng ảnh. Bước 5–6 đổi cách biểu diễn bằng số.**

### Tensor là gì trong ví dụ này?

Một ảnh màu có ba bảng số: bảng Red, bảng Green và bảng Blue. Sau crop, mỗi bảng có 224 hàng và 224 cột.

Ba bảng được chứa trong một **tensor** có shape `(3,224,224)`. Tensor này có thể đưa lên GPU. Nó vẫn biểu diễn nội dung ảnh đó, không phải ảnh mới.

Đổi pixel sang `[0,1]` nghĩa là pixel 128 trở thành `128/255 ≈ 0,502`. Khi ghép 16 ảnh thành một batch, input có shape `(16,3,224,224)`.

### Normalization đang điều chỉnh cái gì?

Nó điều chỉnh **giá trị số của từng pixel ở từng kênh màu**. Ví dụ chỉ xét kênh Red của pixel xám:

```text
Pixel ban đầu:                  128
Đưa về [0,1]:                   128 / 255 ≈ 0,502
Normalization của kênh Red:     (0,502 - 0,485) / 0,229 ≈ 0,074
```

Trong đó `0,485` là mean và `0,229` là std cố định cho kênh Red. Green và Blue có các hằng số tương ứng. Không tính lại mean/std từ từng ảnh.

Mình làm vậy vì pretrained model được train với kiểu input này. **Sau normalization, giá trị có thể âm hoặc lớn hơn 1.**

Mask được tô xám **trước normalization** để đúng màu RGB mình muốn. Tô tensor bằng 0 sau normalization không tương đương với màu xám `(128,128,128)`.

### Model trả về cái gì?

Model trả một danh sách 1.000 **scores**, mỗi score ứng với một class. Class có score cao nhất là dự đoán Top-1; năm class cao nhất là Top-5. Các scores này không tự động là xác suất.

## 4. Những gì được giữ cố định khi so sánh

Hai model cùng dùng 10.000 ảnh ImageNet validation, có label thật, đúng 10 ảnh cho mỗi class trong 1.000 classes.

Danh sách ảnh nằm trong `data/manifests/analysis_10000_seed42.csv`. Đây là **manifest**: danh sách ID và label, không phải file chứa ảnh. Ảnh thực nằm trong 14 Parquet shards.

| Model | Pretrained weights | Resize cạnh ngắn |
| --- | --- | ---: |
| ResNet-50 | `ResNet50_Weights.IMAGENET1K_V2` | 232 |
| ViT-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1` | 256 |

**Checkpoint/weights** là bộ tham số model đã học. Mình giữ nguyên bộ tham số đó trong toàn bộ thí nghiệm.

Hai model đều crop thành 224×224 nhưng resize khác nhau, nên có thể nhìn thấy vùng nội dung ảnh gốc hơi khác nhau. Cùng tọa độ mask trên input không bảo đảm che cùng chi tiết trong ảnh gốc. Đây là giới hạn cần nói trong báo cáo.

## 5. Đọc code: đi theo việc đang làm

### 5.1. File nào có nhiệm vụ gì?

| File | Hiểu đơn giản |
| --- | --- |
| [src/models.py](../../src/models.py) | Lấy model đã train và cách chuẩn bị input của model |
| [src/data.py](../../src/data.py) | Tìm đúng ảnh trong danh sách rồi đọc ảnh |
| [src/evaluation.py](../../src/evaluation.py) | Cho model đoán, tính đúng/sai và lưu predictions |
| [occlusion.py](occlusion.py) | Tạo ảnh bị che, chạy các mức che và tổng hợp kết quả |
| [06_occlusion.ipynb](../../notebooks/06_occlusion.ipynb) | Trình bày toàn bộ thí nghiệm, chạy, lưu và vẽ kết quả |

Notebook hiện vẫn tự chứa logic Occlusion, **chưa gọi `occlusion.py`**. Cả hai đều gọi shared functions trong `src/`. Module được tách sau khi full run hoàn thành; kết quả đã lưu được tạo bởi notebook.

### 5.2. Lấy model và lấy ảnh

```python
model, transform, metadata = load_model("resnet50")
records = load_analysis_manifest(limit=10000)
```

- `model`: ResNet đã load weights; gọi model để lấy dự đoán.
- `transform`: cách resize, crop và normalize chính thức của checkpoint.
- `metadata`: thông tin model/weights/preprocessing để lưu cùng kết quả.
- `records`: danh sách 10.000 ảnh cần dùng, mỗi record có ID, label và index gốc.

`load_model()` nằm trong `src/models.py`. Nó dùng weights trong cache theo mặc định và gọi `model.eval()` để model hoạt động ở evaluation mode.

`load_analysis_manifest()` nằm trong `src/data.py`. Nó kiểm tra danh sách đủ ảnh, đúng số classes và không trùng ID. Label dùng index 0–999, không trừ 1.

`iter_analysis_images(records)` sau đó đọc ảnh thật từ Parquet, chỉ decode các ảnh được chọn và trả từng bộ `(image, label, image_id)`.

`image_id` giống mã số của ảnh. Dùng nó để ghép clean với occluded, không giả định hàng thứ 10 trong hai CSV là cùng một ảnh.

### 5.3. Bốn hàm chuẩn bị ảnh trong `occlusion.py`

| Hàm | Nhận gì? | Làm gì? | Trả gì? |
| --- | --- | --- | --- |
| `resize_crop(image, transform)` | Ảnh gốc và official transform | Resize rồi crop ở giữa | PIL image 224×224 |
| `mask_box(image_id, severity, position, seed=42)` | ID, mức che, vị trí, seed | Tính bounds hình vuông | `(left, top, right, bottom)` |
| `apply_occlusion(image, image_id, severity, position, seed=42)` | Ảnh đã crop và thông tin mask | Sao chép ảnh rồi tô xám trong bounds | Ảnh đã che; ảnh clean không đổi |
| `normalize_image(image, transform)` | Ảnh crop clean hoặc đã che | Chuyển thành tensor float và normalize | Tensor `(3,224,224)` |

`mask_box()` chỉ tính **che ở đâu**. `apply_occlusion()` mới thực sự **tô đè lên ảnh**.

Trong `mask_box()`, SHA-256 biến chuỗi `seed|image_id|severity` thành số để khởi tạo local RNG. Nó giúp quy tắc random ổn định; không phải một bước model học thêm.

Trong `apply_occlusion()`, `image.copy()` rất quan trọng: nếu tô lên ảnh clean gốc, các điều kiện chạy sau có thể bị ảnh hưởng bởi mask trước.

### 5.4. `run_experiment()` làm gì?

Hàm này phối hợp các bước trên cho cả hai model. Cấu hình hiện tại xử lý **16 ảnh mỗi batch** thay vì đưa cả 10.000 ảnh lên GPU một lúc.

Đọc vòng lặp theo trình tự:

1. Lấy tối đa 16 ảnh từ `image_stream` bằng `islice`.
2. Với ResNet, resize/crop batch ảnh theo transform của ResNet.
3. Cho model đoán batch clean một lần.
4. Tạo center masks ở 10%, 20%, 30%, 40%, 50%, rồi đoán từng điều kiện.
5. Tạo random masks ở năm mức tương tự, rồi đoán từng điều kiện.
6. Làm tương tự với ViT, dùng transform của ViT.
7. Ghi nhớ predictions của batch và chuyển sang batch 16 ảnh tiếp theo.
8. Khi đọc hết ảnh, kiểm tra ID và số predictions rồi trả kết quả.

`evaluate()` được gọi ở bước dự đoán là hàm trong **`src/evaluation.py`**, không phải viết lại ở module Occlusion.

Trong vòng lặp có:

```python
normalize = lambda image: normalize_image(image, transform)
```

Đây là một hàm ngắn để đưa cho `evaluate()`. Vì ảnh đã resize/crop trước đó, ở bước này chỉ chuyển tensor và normalize, không resize lần nữa.

`evaluate()` ghép batch bằng `torch.stack`, chạy model trong `torch.inference_mode()`, lấy Top-5 và tạo prediction rows. Inference mode tắt việc lưu gradient vì thí nghiệm chỉ cần dự đoán.

`predictions[name].extend(rows)` nghĩa là nối kết quả batch vừa chạy vào danh sách kết quả của model. Kết quả được giữ ở RAM, không giữ tensor output trên GPU.

Hàm `run_experiment()` **không tự load weights, ghi file hoặc vẽ biểu đồ**.

### 5.5. `summarize_predictions()` làm gì?

Sau inference, mình có rất nhiều prediction rows. Hàm này biến chúng thành bảng tổng hợp dễ đọc:

1. Tìm clean predictions của từng model, lập bảng tra theo ID.
2. Chọn predictions của từng position và severity.
3. Gọi `accuracy_drop()` trong `src/evaluation.py` để tính accuracy/drop trên cùng ID.
4. Đếm thêm số ảnh có dự đoán giống clean để tính consistency.
5. Trả `pandas.DataFrame`, mỗi dòng là kết quả của một model/position/severity.

Hàm này không chạy model. Có thể dùng predictions đã đọc từ CSV để tính lại bảng mà không cần GPU.

### 5.6. Notebook thêm những phần gì?

Notebook gồm chín phần. Setup đặt số ảnh, batch size, seeds và GPU. Data & Models load ảnh/model. Occlusion Function định nghĩa logic mask. Visualization Check cho xem ba ảnh trước khi chạy lớn. Experiment chạy inference. Metrics & Results tính và lưu CSV/metadata. Visualization vẽ charts. Introduction và Discussion giải thích mục tiêu và kết luận.

`save_predictions()` là hàm shared để ghi predictions theo schema. Metrics và các bảng khác dùng `to_csv`. CSV dùng mode `x` để không ghi đè file cũ. Notebook tạo `RUN_ID` có timestamp để mỗi lần chạy có thư mục mới.

## 6. Những con số trong bảng kết quả có nghĩa gì?

### Accuracy: model đoán đúng được bao nhiêu?

Ví dụ minh họa: đưa 100 ảnh, model đoán đúng 80 ảnh thì **Top-1 accuracy = 80%**.

**Top-5 accuracy** rộng hơn: chỉ cần label thật nằm trong năm class được xếp cao nhất là tính đúng. Một ảnh sai Top-1 vẫn có thể đúng Top-5.

### Accuracy drop: sau khi che, mất bao nhiêu accuracy?

Ví dụ clean accuracy 80%, sau che còn 60%:

```text
Absolute drop = 80 - 60 = 20 percentage points (pp)
Relative drop = 20 / 80 × 100 = 25%
```

- **20 pp:** hai mức accuracy cách nhau 20 điểm phần trăm.
- **25%:** mất một phần tư mức accuracy clean ban đầu.

Hai số này có ý nghĩa khác nhau. `absolute_drop_pp` và `relative_drop_pct` trong CSV là drop của Top-1; các cột có `top5` là drop của Top-5.

Không lấy `abs()` của hiệu: nếu sau che model đúng nhiều hơn, drop âm. Nếu clean accuracy bằng 0 thì relative drop không xác định.

### Prediction consistency: model có đổi câu trả lời không?

Ví dụ 100 ảnh đã che có 75 ảnh vẫn được đoán cùng class như lúc clean, **consistency = 75%**.

Nó không phải accuracy. Nếu model đoán “xe tải” cho một ảnh xe buýt ở cả hai lần thì câu trả lời vẫn consistent nhưng đều sai.

Ở severity 0%, ảnh không đổi nên consistency = 100%, drop = 0. Accuracy clean vẫn có thể chỉ khoảng 80%.

**Mọi phép so sánh phải dùng đúng cùng tập ảnh.** Không lấy accuracy full baseline 50.000 ảnh để trừ cho accuracy của analysis set 10.000 ảnh.

## 7. Sản phẩm: mình đã thu được những file gì?

Full run đã hoàn thành, tên run là:

```text
full10000_20261105T051900440637Z
```

Các đường dẫn dưới đây tính từ repository root.

### Nhóm 1: file để đọc kết quả và viết báo cáo

Trong `results/metrics/experiments/occlusion/<run_id>/final/`:

| File | Dùng để làm gì? |
| --- | --- |
| `metrics.csv` | Bảng 24 dòng: accuracy, drop và consistency; dùng viết Results |
| `metadata.json` | Ghi model, weights, preprocessing, seed, môi trường và thông tin lần chạy |

Trong `results/figures/experiments/occlusion/<run_id>/final/`:

| File | Dùng để làm gì? |
| --- | --- |
| `occlusion_examples.png` | Nhìn trực tiếp ảnh clean, center và random masks |
| `accuracy_vs_occlusion.png` | Xem accuracy thay đổi khi tăng diện tích che |
| `accuracy_drop_vs_occlusion.png` | Xem model mất bao nhiêu accuracy so với clean |

Charts hiện vẽ Top-1. Top-5 và consistency nằm trong CSV, chưa có charts riêng.

### Nhóm 2: file để kiểm tra và tái lập

Cũng trong thư mục metrics:

| File | Dùng để làm gì? |
| --- | --- |
| `resnet50_predictions.csv` | Dự đoán từng ảnh/điều kiện của ResNet |
| `vit_b_16_predictions.csv` | Dự đoán từng ảnh/điều kiện của ViT |
| `selected_images.csv` | Danh sách chính xác 10.000 ảnh đã chạy |
| `mask_coordinates.csv` | Tọa độ mask dùng cho từng ID/position/severity |

Một prediction row có ID, model, label thật, năm classes dự đoán, cờ đúng/sai, condition và severity. Cùng ảnh xuất hiện nhiều lần ở các conditions là đúng.

Tại sao có nhiều predictions hơn ảnh?

```text
Một ảnh/model:
1 clean + 5 center + 5 random = 11 lần dự đoán

10.000 ảnh × 11 = 110.000 prediction rows/model
Hai model = 220.000 inference image passes
```

Clean 0% chỉ tính một lần nhưng được dùng lại cho cả center và random. Bảng metrics vẫn có `2 models × 2 positions × 6 levels = 24 dòng`.

Mask table có `10.000 × 2 positions × 6 levels = 120.000 dòng`, bao gồm 0%. Hai model dùng chung bảng tọa độ đó.

**Metadata** giống ghi chú đi kèm sản phẩm. **Checksum/hash** của danh sách ảnh giúp kiểm tra người khác có dùng đúng danh sách hay không. **Run ID** phân biệt lần chạy này với lần chạy khác.

## 8. Kết quả thực tế của mình nói gì?

Đây là số liệu thật, đọc từ [metrics.csv](../../results/metrics/experiments/occlusion/full10000_20261105T051900440637Z/final/metrics.csv):

| Top-1 accuracy | ResNet-50 | ViT-B/16 |
| --- | ---: | ---: |
| Clean | 80.80% | 80.75% |
| Center 50% | 56.84% | 62.50% |
| Random 50% | 60.48% | 64.40% |

Ví dụ center 50%: ResNet từ 80.80% xuống 56.84%, mất **23.96 pp**. ViT từ 80.75% xuống 62.50%, mất **18.25 pp**. ViT mất ít accuracy hơn trong điều kiện này.

Ở random 50%, ResNet mất **20.32 pp**, ViT mất **16.35 pp**. Với cả hai vị trí tại 50%, ViT giữ accuracy tốt hơn.

Trong toàn bộ bảng, cả hai model giảm accuracy khi severity tăng; ViT có Top-1 cao hơn ở mọi nonzero severity đã thử. Clean accuracy của hai model gần bằng nhau.

Kết luận có thể dùng: **“ViT-B/16 checkpoint này bền vững hơn ResNet-50 checkpoint này dưới protocol Occlusion đã đánh giá.”**

Cần giới hạn kết luận vì:

- Chỉ có hai checkpoints, không đại diện mọi CNN/ViT.
- Official resize khác nhau nên vùng nội dung bị che có thể khác nhau.
- Chỉ có một random mask cho mỗi ảnh/severity với seed 42.
- Chưa chạy nhiều seeds hoặc tính confidence intervals để đánh giá uncertainty.
- Square gray mask là cách che nhân tạo, không bao quát mọi tình huống ngoài đời.

Full run đã kiểm tra lại đủ ID/label, tọa độ masks và cả 24 metrics từ predictions. Có sai khác nhỏ với baseline cũ: ResNet đổi 2 clean Top-1 predictions, tương ứng clean accuracy lệch 0.02 pp; nguyên nhân chính xác chưa xác minh. Drops luôn dùng clean của chính run này. ViT clean Top-1 và Top-5 class sets khớp trên 10.000 ID.

## 9. Khi mở project, nên làm gì trước?

Muốn **hiểu thí nghiệm**, đọc phần 1–4 của file này và xem `occlusion_examples.png`.

Muốn **đọc code**, mở `occlusion.py`, đi theo thứ tự: `resize_crop` → `mask_box` → `apply_occlusion` → `normalize_image` → `run_experiment` → `summarize_predictions`.

Muốn **viết báo cáo**, mở `metrics.csv` và hai charts; dùng phần 8 để hiểu con số. Không cần chạy lại model.

Muốn **kiểm tra một dự đoán**, dùng `image_id` để tìm cùng ảnh ở clean và condition cần xem trong prediction CSV.

Muốn **dùng module trong code mới**, xem ví dụ ở [README](README.md). Import module không tự chạy; gọi `run_experiment()` mới bắt đầu inference. Notebook hiện có full-run output và workflow saving đầy đủ, không cần chạy lại để xem sản phẩm.

## 10. Một đoạn nói khi trình bày

> Em dùng cùng 10.000 ảnh để kiểm tra hai model đã train sẵn. Đầu tiên em đoán trên ảnh chưa che. Sau đó em che ảnh bằng hình vuông xám, ở giữa hoặc vị trí random, với diện tích tăng từ 10% đến 50%, rồi đoán lại. Em đếm tỷ lệ đúng và so với lúc chưa che. Hai model có clean accuracy gần bằng nhau, nhưng ở center 50%, ResNet còn 56.84% còn ViT còn 62.50%. Vì thế, trong thí nghiệm này ViT giữ khả năng phân loại tốt hơn. Kết luận chỉ áp dụng cho hai checkpoints và cách che đã đánh giá.
