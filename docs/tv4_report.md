# Báo cáo TV4 — Texture Robustness và Shape-vs-Texture Bias

## Methodology

Mục tiêu TV4 là khảo sát (i) độ nhạy với suy giảm chi tiết texture và (ii) quyết
định khi shape cue mâu thuẫn texture cue. Hai model giữ nguyên checkpoint chung:
ResNet-50 `IMAGENET1K_V2` và ViT-B/16 `IMAGENET1K_V1`. Mỗi model dùng đúng
`weights.transforms()` của torchvision: RGB, resize cạnh ngắn 232/256, center
crop 224, và ImageNet normalization.

### Texture Robustness

Protocol dùng manifest nhóm `analysis_10000_seed42.csv` (10.000 ImageNet
validation images, 10/class). Biến đổi đặt trước preprocessing, giữ nguyên
kích thước và nhãn: Gaussian blur `(kernel, sigma) = (5,1), (9,2), (15,4)`;
bilateral filter `(d, sigmaColor, sigmaSpace) = (5,25,25), (9,50,50),
(13,100,100)`. Mỗi condition được đánh giá trên cùng ID; báo cáo Top-1,
absolute drop (pp), prediction consistency và bootstrap paired 95% CI.

Đây là đo **sensitivity to texture-detail degradation**, không phải phép đo cô
lập texture bias: blur đồng thời có thể làm giảm thông tin biên/shape.

### Shape-vs-Texture Bias

Tập rgeirhos/texture-shape-cue-conflict có 1.280 ảnh, 16 categories, với
`shape_cue` và `texture_cue`. 80 ảnh có hai cue giống nhau được loại khỏi phân
tích conflict, còn 1.200 ảnh hợp lệ. Prediction 1.000 ImageNet lớp được quy về
16 category bằng **mean softmax probability** trên danh sách ImageNet indices
từ code chính thức của Geirhos et al.; không dùng trực tiếp ImageNet top-1.

`Shape Bias = N_shape / (N_shape + N_texture)`; Neither không vào mẫu số.
95% CI lấy bootstrap 2.000 mẫu với seed 42.

## Results

Shape-vs-Texture Bias đã chạy full trên GPU RTX 3060, batch 64:

| Model | Valid conflict | Shape | Texture | Neither | Coverage | Shape Bias (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 1.200 | 181 | 671 | 348 | 71,00% | 21,24% (18,52–24,09) |
| ViT-B/16 | 1.200 | 326 | 493 | 381 | 68,25% | 39,80% (36,45–43,12) |

Như vậy ViT-B/16 cao hơn 18,56 pp shape bias trên các cue decisions của
benchmark này. Các số liệu nguồn nằm trong `results/tv4/shape_texture_*.csv`;
metadata ghi version torch/torchvision, model checkpoints, GPU, seed và Git SHA.

Texture Robustness chưa có số liệu: 14 ImageNet Parquet validation shards không
có tại `data/imagenet/data/` (smoke test dừng bằng `FileNotFoundError`). Không
thay bằng dataset khác, cũng không suy ra số liệu từ baseline 50k. Script đã sẵn
sàng chạy đầy đủ khi khôi phục đúng dataset nhóm.

## Discussion

Trên cue-conflict, ViT-B/16 chọn shape cue thường xuyên hơn ResNet-50; khoảng
tin cậy bootstrap của hai estimate không chồng nhau. Tuy nhiên đây là kết luận
về đúng một cặp pretrained checkpoint, benchmark và protocol mapping này; nó
không chứng minh mọi ViT đều shape-biased hơn mọi CNN. Tỷ lệ Neither đáng kể
(29,0% ResNet, 31,75% ViT trong 1.200 ảnh hợp lệ) cho thấy một phần output không
khớp hai cue category và vì thế không được đưa vào Shape Bias theo định nghĩa.

Texture Robustness sẽ bổ sung góc nhìn khác: model có shape bias cao hơn chưa
đương nhiên ít giảm Top-1 hơn khi blur. Chỉ khi chạy trên cùng 10.000 IDs mới có
thể kết luận mối liên hệ giữa hai thí nghiệm.

## Limitations

- ImageNet validation chưa hiện diện trong workspace nên thí nghiệm texture chưa chạy.
- Cue-conflict images được style-transfer và chỉ đại diện 16 entry-level categories.
- ResNet và ViT có resize size khác nhau theo preprocessing chính thức (232 vs 256).
- Bootstrap biểu thị bất định lấy mẫu benchmark, không biến kết quả một checkpoint thành kết luận kiến trúc phổ quát.

## References

1. Geirhos, R. et al. (2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*. ICLR. https://openreview.net/forum?id=Bygh9j09KX
2. Geirhos et al. Official code and 16-class mapping. https://github.com/rgeirhos/texture-vs-shape
3. rgeirhos (2026). *texture-shape-cue-conflict* dataset. https://huggingface.co/datasets/rgeirhos/texture-shape-cue-conflict
4. Torchvision model weights documentation, ResNet50 IMAGENET1K_V2 and ViT-B/16 IMAGENET1K_V1. https://pytorch.org/vision/stable/models.html
