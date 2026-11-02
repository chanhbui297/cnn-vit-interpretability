"""Build the standalone Vietnamese ERF tutorial with only the standard library."""

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/erf.py"


def build_notebook():
    sections = {}
    for part in SOURCE.read_text("utf-8").split("# %% ")[1:]:
        name, code = part.split("\n", 1)
        sections[name.strip()] = code.strip() + "\n"
    cells = []

    def add(kind, content, tags=None):
        cell = {
            "cell_type": kind, "id": f"erf-{len(cells):02d}",
            "metadata": {"tags": tags} if tags else {},
            "source": textwrap.dedent(content).strip().splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(text):
        add("markdown", text)

    def code(text):
        add("code", text)

    def helper(name):
        add("code", sections[name], ["erf-source", name])

    md(r"""
    # 04 — Effective Receptive Field (ERF) · Thành viên 3

    **Câu hỏi:** Khi xét một đặc trưng ở giữa mạng, những pixel nào của ảnh ảnh hưởng
    mạnh đến đặc trưng đó? Phân bố này thay đổi thế nào qua các tầng của ResNet-50 và ViT-B/16?

    Notebook này tự chứa code, chạy được trên Colab hoặc Jupyter. Bạn không cần train model.
    Hãy đọc phần giải thích rồi chạy từng ô, chưa dùng **Run all** trong lần đầu.

    Lộ trình: **GPU → dữ liệu → một ảnh → gradient → metrics → nhiều ảnh → báo cáo**.
    Code nguồn tương ứng: `src/erf.py`; notebook được tạo bằng `src/make_erf_notebook.py`.
    Các ô chưa được chạy sẵn; không có số liệu ERF ImageNet giả lập trong notebook.
    """)
    md("""
    ## Bước 0 — Kiểm tra nơi chạy

    Trên [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook**,
    mở file này, rồi vào **Runtime → Change runtime type → chọn GPU** nếu được cấp.
    Chạy ô dưới. `CUDA: True` nghĩa là PyTorch nhìn thấy GPU.
    `False` vẫn chạy thử được bằng CPU, nhưng hãy kiểm tra lại runtime trước khi chạy nhiều ảnh.

    Colab thường có sẵn torch/torchvision. Nếu thiếu thư viện phụ, ô này cài vào đúng kernel.
    Khi dùng máy cá nhân, cài bộ torch/torchvision phù hợp theo
    [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/) trước.
    """)
    code("""
    import importlib.util
    import subprocess
    import sys

    missing = [package for package in ['numpy', 'pandas', 'matplotlib', 'PIL', 'pyarrow', 'tqdm']
               if importlib.util.find_spec(package) is None]
    if missing:
        packages = ['pillow' if name == 'PIL' else name for name in missing]
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *packages])

    import torch
    import torchvision

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Python:', sys.version.split()[0])
    print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
    print('CUDA:', torch.cuda.is_available())
    print('Thiết bị:', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU')
    """)
    helper("imports")
    md("""
    ## Bước 1 — Kết nối đúng dữ liệu của nhóm

    GitHub chứa **danh sách ảnh**, không chứa ảnh ImageNet. Cần các file sau trong cùng
    thư mục project (lấy bộ validation mà nhóm đã chuẩn bị, không đổi sang bộ ảnh khác):

    ```text
    cnn-vit-interpretability/
      configs/analysis_set.json
      data/manifests/analysis_10000_seed42.csv
      data/imagenet/data/validation-00000-of-00014.parquet
      ... đến validation-00013-of-00014.parquet
    ```

    **Colab:** đặt thư mục này trong Google Drive, chạy ô mount rồi sửa đường dẫn nếu cần.
    Đọc trực tiếp các shard từ Drive thuận tiện cho lần thử đầu. Khi I/O chậm, có thể copy
    các shard vào `/content` và đặt lại `DATA_DIR`; giữ `OUTPUT_BASE` trên Drive để lưu kết quả.
    Runtime Colab là tạm thời; file chỉ nằm trong `/content` có thể mất khi runtime bị thu hồi.

    **Máy cá nhân:** mở notebook từ thư mục repo hoặc `notebooks/`; ô cấu hình tự tìm project.
    Kiểm tra thư mục đúng trước khi chạy tiếp. Bước này không tải model.
    """)
    code("""
    IS_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
    if IS_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
    """)
    code("""
    if IS_COLAB:
        PROJECT_ROOT = Path('/content/drive/MyDrive/cnn-vit-interpretability')
    else:
        PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                             if (p / 'configs/analysis_set.json').is_file()), None)
        if PROJECT_ROOT is None:
            raise FileNotFoundError('Mở notebook từ repo hoặc sửa PROJECT_ROOT bằng đường dẫn đầy đủ.')

    DATA_DIR = PROJECT_ROOT / 'data/imagenet/data'
    OUTPUT_BASE = PROJECT_ROOT / 'results/metrics/erf'
    print('Project:', PROJECT_ROOT)
    print('Số shard validation:', len(list(DATA_DIR.glob('validation-*.parquet'))), '/ 14')
    print('Config:', (PROJECT_ROOT / 'configs/analysis_set.json').is_file())
    print('Manifest:', (PROJECT_ROOT / 'data/manifests/analysis_10000_seed42.csv').is_file())
    """)
    md("""
    ## Bước 2 — Chọn một tập ảnh nhỏ, dùng chung cho hai model

    Ta kiểm tra SHA-256 của manifest gốc rồi lấy ngẫu nhiên 10 ảnh bằng seed 42.
    Không lấy 10 dòng đầu vì manifest sắp theo ID, có thể tạo thiên lệch.
    Đây là lấy mẫu đều theo ảnh, **không phải** tập con cân bằng đủ 1.000 lớp.
    Danh sách được lưu cùng kết quả, để biết chính xác đã dùng ảnh nào.

    Trong lần đầu giữ `NUM_IMAGES = 10`; khi chạy ổn mới đổi thành 100–200.
    Để xác minh đầy đủ revision của các shard, chạy script có sẵn
    `src/create_analysis_manifest.py --check` tại project; phép kiểm tra notebook
    kiểm tra checksum manifest và ID/label/index của các ảnh thực sự được đọc.
    """)
    helper("sampling")
    code("""
    SEED = 42
    NUM_IMAGES = 10
    PREPROCESSING = 'official'
    records = select_records(PROJECT_ROOT, NUM_IMAGES, SEED)
    display(pd.DataFrame(records))
    print('Số lớp trong tập nhỏ:', len({row['label'] for row in records}))
    """)
    code("""
    images = load_images(DATA_DIR, records)
    example_record, example_image = next(images)
    images.close()
    print('Ảnh thử:', example_record)
    plt.figure(figsize=(5, 5))
    plt.imshow(example_image)
    plt.axis('off')
    plt.show()
    """)
    md(r"""
    ## Bước 3 — Chọn đặc trưng cần khảo sát

    | Giai đoạn tương đối | ResNet-50 V2 | ViT-B/16 V1 |
    |---|---|---|
    | 1 | layer1 | block 3 |
    | 2 | layer2 | block 6 |
    | 3 | layer3 | block 9 |
    | 4 | layer4 | block 12 |

    Đây là các mốc theo độ sâu, **không phải** các tầng tương đương về tính toán.
    Với ResNet, lấy vector kênh ở giữa feature map, sau toàn bộ stage.
    Với ViT, lấy vector của patch token ở gần tâm, sau toàn bộ block và trước
    LayerNorm cuối của encoder. CLS vẫn tham gia attention nhưng không được chọn làm đầu ra.

    Quy ước chỉ số bắt đầu từ 0: `row=H//2`, `col=W//2`.
    Với lưới patch 14×14: chọn (7,7), token thứ `1 + 7*14 + 7 = 106` trong chuỗi có CLS.
    Lưới chẵn không có duy nhất một ô tâm; vị trí receptive center giữa hai kiến trúc
    cũng không trùng hoàn toàn. Notebook lưu vị trí cụ thể trong CSV.

    Mỗi vector có nhiều số, cần chọn **một scalar** để backward.
    Ta dùng $z=\frac{1}{C}\sum_k f_k^2$ (năng lượng trung bình của vector),
    cùng quy tắc cho cả hai model. Đây là một lựa chọn thực nghiệm có chủ đích;
    kết quả phụ thuộc lựa chọn scalar và không phải chuẩn toàn bộ Jacobian.
    Không dùng trung bình kênh sau LayerNorm vì có thể gần hằng số và triệt tiêu gradient.
    """)
    md(r"""
    ## Bước 4 — Từ gradient thành bản đồ ERF

    Với $u$ là RGB trong [0,1] **sau resize/crop**, ta đo:

    $$ E_{ij}=\sum_{c=1}^3\left|\frac{\partial z}{\partial u_{cij}}\right|,
       \qquad P_{ij}=E_{ij}/\sum_{ab}E_{ab}. $$

    `requires_grad_(True)` yêu cầu PyTorch theo dõi ảnh đầu vào;
    `autograd.grad(z, x)` tính độ nhạy của scalar với từng pixel.
    `eval()` tắt dropout và dùng BatchNorm đã học, vẫn cho phép gradient.
    Không bọc ERF trong `inference_mode()`.

    Model nhận $x_c=(u_c-\mu_c)/\sigma_c$, nên phải chia gradient theo $x_c$
    cho $\sigma_c$ để đổi về gradient theo RGB. Ta không tính đạo hàm ngược
    qua phép resize/crop về ảnh gốc. Gradient thô được lưu riêng; bản đồ chuẩn hóa
    mô tả phân bố tương đối. Tổng gradient thô không dùng để xếp hạng hai kiến trúc.

    Đây là phép đo độ nhạy cục bộ của biểu diễn, không chứng minh pixel đó gây ra
    quyết định phân loại. Cách scalar hóa và trị tuyệt đối là lựa chọn của notebook,
    không phải sao chép nguyên xi mọi thiết lập của Luo et al. (2016).
    """)
    helper("gradient")
    md("""
    ## Bước 5 — Đo ERF bằng số, không chỉ nhìn màu

    - **A50 / A90:** sắp pixel theo gradient giảm dần, lấy số pixel nhỏ nhất chứa
      ít nhất 50% / 90% tổng gradient rồi chia cho 224×224. Pixel có thể rời nhau;
      đây không phải bán kính, bounding box hay diện tích vùng liên thông.
    - **Dispersion:** khoảng cách RMS tới trọng tâm phân bố gradient, dùng tọa độ
      x,y chuẩn hóa [0,1]. Cao hơn nghĩa là ảnh hưởng phân tán xa trọng tâm hơn.
    - **Center / outer energy:** tỷ lệ gradient trong / ngoài hình vuông giữa ảnh,
      có cạnh bằng nửa chiều ảnh. Tâm ở đây là tâm ảnh, không phải vị trí feature.

    Ví dụ minh họa: gradient đều trên toàn ảnh → A90 gần 90%; gradient chỉ tại
    một pixel → A90 bằng 1/(224×224). Không suy ra model nào tốt hơn từ A90.
    """)
    helper("metrics")
    code("""
    # Chỉ kiểm tra định nghĩa metric bằng mảng giả, không phải kết quả ImageNet.
    uniform = np.ones((10, 10))
    point = np.zeros((10, 10))
    point[5, 5] = 1
    print('Đều trên 100 pixel → A90:', erf_metrics(uniform)['area90_fraction'])
    print('Tập trung 1/100 pixel → A90:', erf_metrics(point)['area90_fraction'])
    """)
    helper("plotting")
    md("""
    ## Bước 6 — Chạy đúng một ảnh trước

    Ô dưới tải lần lượt hai bộ trọng số từ cache hoặc máy chủ PyTorch. Chỉ giữ một
    model trên GPU mỗi lần; batch size bằng 1. Đóng băng trọng số không ngăn gradient
    theo ảnh đầu vào. Lần tải trọng số đầu tiên cần internet.

    Mỗi hàng hiển thị crop model thực sự nhận và 4 ERF maps. Cùng thang màu log10
    cho mọi bản đồ để nhìn được gradient yếu. Màu bị giới hạn để hiển thị;
    A90 tính từ gradient gốc, không từ màu hay bản đồ đã clipping.

    Preprocessing chính giữ thiết lập nhóm: ResNet resize cạnh ngắn 232, ViT 256,
    rồi center crop 224. **Cùng ảnh gốc nhưng hai crop khác nhau.**
    Vì vậy khác biệt ERF phản ánh cả model và preprocessing. Sau thí nghiệm chính,
    đặt `PREPROCESSING='shared_vit_crop'` để chạy thêm kiểm tra với cùng crop 256→224;
    chế độ phụ này thay preprocessing chính thức của ResNet và phải báo cáo riêng.
    """)
    helper("experiment")
    code("""
    from datetime import datetime, timezone

    # Đường dẫn có thời gian để không ghi đè kết quả cũ.
    def new_run_dir(name):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        return OUTPUT_BASE / f'{name}_{PREPROCESSING}_{stamp}'

    smoke_dir = new_run_dir('smoke_1')
    smoke_metrics, smoke_summary, smoke_previews = run_experiment(
        records[:1], DATA_DIR, smoke_dir, DEVICE,
        seed=SEED, preprocessing=PREPROCESSING, num_previews=1,
    )
    display(smoke_metrics[['model', 'layer', 'grid_row', 'grid_col', 'token_index',
                           'area50_fraction', 'area90_fraction', 'dispersion_normalized']])
    fig = plot_image_maps(records[0]['image_id'], smoke_previews[records[0]['image_id']])
    plt.show()
    print('Đã lưu:', smoke_dir)
    """)
    md("""
    **Trước khi chạy nhiều ảnh:** xác nhận có 8 dòng metrics (2 model × 4 tầng),
    gradient hữu hạn và tổng dương, A50 ≤ A90 ≤ 1. Nếu lỗi thiếu shard, quay lại
    bước 1. Nếu GPU hết bộ nhớ, khởi động lại runtime để giải phóng model còn sót
    từ các lần chạy trước; notebook đã dùng từng ảnh và từng model.

    Không kết luận xu hướng từ một ảnh. Ta chuyển sang nhiều ảnh để tính phân bố.
    """)
    code("""
    assert len(smoke_metrics) == 8
    assert np.isfinite(smoke_metrics['raw_gradient_sum']).all()
    assert (smoke_metrics['raw_gradient_sum'] > 0).all()
    assert (smoke_metrics['area50_fraction'] <= smoke_metrics['area90_fraction']).all()
    assert smoke_metrics['area90_fraction'].between(0, 1).all()
    print('Kiểm tra một ảnh đạt. Có thể chuyển sang tập nhỏ.')
    """)
    md("""
    ## Bước 7 — Chạy tập nhỏ, rồi mở rộng

    Ô này chạy `NUM_IMAGES` đã chọn ở bước 2. Sau khi chạy ổn với 10 ảnh,
    đổi thành 100–200, chạy lại ô chọn `records` rồi chạy lại ô dưới.
    Không cần chạy lại baseline hay train model. Ta giữ tất cả ảnh được chọn,
    không lọc theo dự đoán đúng/sai.

    CSV có 8×N dòng; NPZ lưu các bản đồ gradient thô cho từng model/ảnh.
    `summary.csv` tổng hợp metric từng ảnh: **không** tính A90 của heatmap trung bình.
    Thanh sai số trên biểu đồ là độ lệch chuẩn giữa ảnh, không phải khoảng tin cậy.
    Khi một ảnh duy nhất, SD không xác định; biểu đồ hiển thị thanh sai số bằng 0.
    """)
    code("""
    run_dir = new_run_dir(f'main_{len(records)}')
    metrics, summary, previews = run_experiment(
        records, DATA_DIR, run_dir, DEVICE,
        seed=SEED, preprocessing=PREPROCESSING, num_previews=3,
    )
    display(summary)
    fig = plot_summary(metrics)
    plt.show()
    print('Hoàn tất:', run_dir)
    print('Số dòng metrics:', len(metrics), '| Mong đợi:', 8 * len(records))
    """)
    md("""
    ## Bước 8 — Đọc và viết kết quả

    Bảng dưới so sánh trên **cùng từng ảnh**, với chênh lệch A90 = ViT − ResNet.
    Giá trị dương nghĩa là ViT cần nhiều pixel hơn để chứa 90% gradient theo phép đo này.
    Chỉ ghép theo giai đoạn tương đối; không khẳng định layer1 tương đương block3.
    """)
    code("""
    paired = metrics.pivot(index=['image_id', 'stage'], columns='model', values='area90_fraction')
    paired['vit_minus_resnet_pp'] = 100 * (paired['vit_b_16'] - paired['resnet50'])
    paired_summary = paired.groupby('stage')['vit_minus_resnet_pp'].agg(['mean', 'std', 'count'])
    paired_summary.to_csv(run_dir / 'paired_area90_difference.csv')
    display(paired_summary)
    """)
    md("""
    **Mẫu viết báo cáo — điền sau khi có số liệu thật:**

    > Chúng tôi phân tích [N] ảnh lấy từ manifest 10.000 ảnh, seed [seed], dùng
    > ResNet-50 IMAGENET1K_V2 và ViT-B/16 IMAGENET1K_V1. ERF được ước lượng bằng
    > tổng trị tuyệt đối gradient theo RGB của năng lượng vector đặc trưng ở gần tâm.
    > Tại giai đoạn [s], A90 trung bình là [...] ở ResNet và [...] ở ViT;
    > chênh lệch ghép cặp trung bình là [...] điểm phần trăm. Bản đồ [...] cho thấy [...].

    **Discussion cần trả lời:**

    1. A90 và dispersion thay đổi thế nào từ tầng đầu tới cuối? Các ảnh có đồng nhất không?
    2. Khác biệt còn giữ khi dùng cùng crop không? Có trường hợp ngược xu hướng không?
    3. Liên hệ với attention/Grad-CAM/occlusion của nhóm, không suy ra quan hệ nhân quả.
    4. Nêu giới hạn: tập con nhỏ; chọn một vị trí và một scalar; gradient là độ nhạy
       cục bộ; trọng số và cách huấn luyện khác nhau; crop và lưới đặc trưng khác nhau.

    Baseline nhóm: Top-1 80,844% và 81,066%, chênh 0,222 điểm phần trăm.
    Báo cáo độ chênh thực tế; nhóm chưa đặt ngưỡng matched accuracy.
    Không viết “ViT luôn có ERF lớn hơn” hoặc “ERF rộng chứng minh model tốt hơn”.

    **Sản phẩm bàn giao trong thư mục run:**

    - `selected_images.csv`: đúng ảnh/nhãn/index đã dùng.
    - `metrics.csv`, `summary.csv`, `paired_area90_difference.csv`: số liệu.
    - `maps/*.npz`: gradient thô từng tầng, có thể chuẩn hóa lại từ đây.
    - `figures/*.png`: so sánh theo ảnh và biểu đồ theo giai đoạn.
    - `run.json`: trạng thái, phiên bản, trọng số, preprocessing, vị trí và target.

    Khi gián đoạn, `run.json` ở trạng thái running hoặc failed và CSV có thể chưa đủ.
    Chạy lại tạo thư mục mới; chưa có chức năng resume. Generated outputs đang được
    `.gitignore` loại ra. Đừng ép thêm ảnh ImageNet, weights hoặc NPZ vào Git.
    Trước khi chia sẻ notebook, xóa output các ô hiển thị ảnh ImageNet.
    """)
    md("""
    ## Tài liệu tham khảo

    1. Luo et al. (2016), [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://papers.neurips.cc/paper_files/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html).
    2. Raghu et al. (2021), [Do Vision Transformers Act Like Convolutional Neural Networks?](https://research.google/pubs/do-vision-transformers-act-like-convolutional-neural-networks/).
    3. Torchvision: [ResNet-50 weights/transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
       [ViT-B/16 weights/transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html),
       [ViT implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html).
    4. [Colab FAQ: GPU availability, runtimes, and Drive](https://research.google.com/colaboratory/faq.html).
    """)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    path = ROOT / "notebooks/04_erf.ipynb"
    path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Created:", path)
