# ERF — hướng dẫn cho thành viên 3

Mở `notebooks/04_erf.ipynb`. Notebook tự chứa code và giải thích tiếng Việt,
chạy trên Colab hoặc Jupyter. Bắt đầu với ô **Bước 0**, sau đó chạy từng ô.

## Nên chạy ở đâu?

Google Colab có GPU thuận tiện cho lần đầu khi máy chưa có môi trường PyTorch.
GPU và thời lượng phiên được cấp tùy tình trạng tài nguyên, không được đảm bảo;
xem [Colab FAQ](https://research.google.com/colaboratory/faq.html).
Nếu dùng được máy đã chạy baseline của nhóm thì máy đó cũng phù hợp vì đã có
môi trường CUDA và dữ liệu. Không cần train lại hai model.

Trên Colab, upload notebook bằng **File → Upload notebook**, chọn GPU trong
**Runtime → Change runtime type**. Chạy bước 0 và kiểm tra `CUDA: True`.
Notebook cài thư viện phụ nếu thiếu, giữ bộ torch/torchvision của runtime.
Môi trường `.venv` được tạo trong phiên phát triển này chỉ có bản PyTorch CPU để
kiểm tra code; không phải môi trường CUDA đã chạy baseline của nhóm.

## Chuẩn bị dữ liệu

Repo không có ảnh ImageNet. Dùng đúng 14 shard validation đã được nhóm chuẩn bị,
không thay bằng ảnh ngẫu nhiên trên internet. Đặt trong `data/imagenet/data/`.
Giữ `configs/analysis_set.json` và manifest được chia sẻ trong repo.

Để chạy trên Colab, có thể đặt thư mục project trong
`MyDrive/cnn-vit-interpretability/`. Mount Drive ở bước 1 và sửa `PROJECT_ROOT`
cho đúng. Có thể đọc shard từ Drive trong lần thử đầu; nếu chậm, copy chúng vào
runtime và đổi `DATA_DIR`. Lưu `OUTPUT_BASE` trên Drive để giữ kết quả sau khi
runtime mất. Không cần giải nén 50.000 file ảnh hoặc upload ảnh vào Git.

## Trình tự thực hiện

1. **Bước 0:** kiểm tra phiên bản và GPU. Chưa cần ImageNet.
2. **Bước 1–2:** kiểm tra đường dẫn, chọn 10 ảnh từ manifest bằng seed 42,
   đọc một ảnh để xác nhận đúng bộ dữ liệu.
3. **Bước 3–4:** đọc cách chọn tầng, vị trí và scalar rồi tính gradient theo RGB.
4. **Bước 5:** hiểu A50/A90, dispersion và center energy bằng mảng minh họa.
5. **Bước 6:** chạy hai model trên một ảnh; cần có 8 dòng metrics và 8 ERF maps.
6. **Bước 7:** chạy 10 ảnh, kiểm tra kết quả, sau đó tăng lên 100–200 ảnh.
7. **Bước 8:** đọc chênh lệch ghép cặp, điền phần phương pháp và kết quả báo cáo.

Không cần chạy lại baseline, không chạy cả 10.000 ảnh ngay lần đầu.

## Định nghĩa phép đo

- ResNet-50 dùng IMAGENET1K_V2; ViT-B/16 dùng IMAGENET1K_V1.
- ResNet lấy center feature sau layer1/2/3/4; ViT lấy spatial token sau
  block3/6/9/12, trước final encoder LayerNorm. Không chọn CLS làm target.
- Target scalar: `z = mean(center_feature_vector ** 2)` cho cả hai kiến trúc.
  Đây là gradient của một scalar cụ thể, không phải chuẩn toàn bộ Jacobian.
- ERF thô: `E = abs(dz/dRGB).sum(RGB channels)`. Đầu vào RGB trong [0,1]
  là crop sau resize, trước normalize; chia gradient theo tensor chuẩn hóa
  cho RGB std để áp dụng chain rule.
- Bản đồ xác suất tương đối: `P = E / E.sum()`.
- A90 là phần nhỏ nhất của **số pixel**, sau khi xếp gradient giảm dần,
  chứa ít nhất 90% gradient tuyệt đối. Không yêu cầu các pixel liên thông.
- Dispersion là khoảng cách RMS tới trọng tâm của P theo tọa độ [0,1].
- Center energy dùng hình vuông giữa ảnh có cạnh bằng nửa chiều ảnh.

Không đổi scalar hay vị trí dựa trên kết quả nhìn có đẹp hay không. Mọi gradient
bằng 0 hoặc không hữu hạn sẽ báo lỗi, không được âm thầm thay bằng bản đồ đều.

## So sánh công bằng và giới hạn

Thiết lập chính `official` giữ preprocessing của nhóm: ResNet resize 232 và
ViT resize 256, cùng crop 224. Vì vậy cùng ID ảnh nhưng crop khác nhau;
khác biệt quan sát phản ánh cả model và preprocessing. Chạy thêm
`shared_vit_crop` để hai model dùng đúng cùng crop 256→224 và báo cáo riêng.
Thiết lập phụ này thay preprocessing chính thức của ResNet.

Các stage là mốc độ sâu tương đối, không tương đương tuyệt đối. Vị trí giữa
lưới cũng không trùng chính xác giữa các kiến trúc. Nên nói rõ đây là ERF
dựa trên độ nhạy của một scalar tại một vị trí. Kết quả phụ thuộc trọng số,
ảnh đầu vào và cách chọn target; không chứng minh model hiểu ảnh tốt hơn.
Gradient không phải bằng chứng nhân quả của quyết định phân loại.

Tập nhỏ lấy mẫu đều theo ảnh từ manifest, không cân bằng đầy đủ 1.000 lớp.
Các model cùng dùng mọi ảnh được chọn; không lọc đúng/sai để tránh thay đổi
quần thể nghiên cứu. Summary tính metric trên từng ảnh rồi tổng hợp.
Độ lệch chuẩn giữa ảnh không phải khoảng tin cậy.

## Kết quả và kiểm tra code

Mỗi run tạo một thư mục mới trong `results/metrics/erf/`, có CSV metrics,
summary, danh sách ảnh, raw maps NPZ, hình PNG và `run.json`.
Nếu `status` chưa là `complete`, không dùng như một run hoàn chỉnh.
Chưa có resume: chạy lại tạo run mới. Generated outputs bị Git ignore;
giữ nguyên quy tắc hiện có. Xóa output chứa ảnh ImageNet trước khi chia sẻ notebook.

Code nguồn được giữ ở `src/erf.py`. Sau khi sửa các helper, tạo lại notebook:

```powershell
python src/make_erf_notebook.py
.\.venv\Scripts\python.exe experiments/erf/check_erf.py
```

Kiểm tra dùng mảng có kết quả biết trước, finite differences theo RGB,
Parquet ảnh giả và mô hình torchvision với **trọng số ngẫu nhiên** để kiểm tra
đúng đường lấy feature và backward. Chúng không tải pretrained weights và
không phải số liệu nghiên cứu. Cần ImageNet để chạy và phân tích thực nghiệm thật.

Tham khảo: [Luo et al., 2016](https://papers.neurips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html),
[Raghu et al., 2021](https://research.google/pubs/do-vision-transformers-act-like-convolutional-neural-networks/),
[torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[torchvision ViT-B/16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html).
