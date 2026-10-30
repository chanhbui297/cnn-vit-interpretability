"""Evaluate ResNet50 and ViT-B/16 on the local ImageNet validation set.

Run from the repository root; see docs/baseline.md for commands and outputs.
"""

import argparse
from collections import Counter
import csv
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image
import pyarrow.parquet as pq
import torch
import torchvision
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)
from tqdm import tqdm


# Chỉ lưu hàm tạo model; đến lượt đánh giá mới tải model vào bộ nhớ.
MODEL_SPECS = [
    ("resnet50", resnet50, ResNet50_Weights.IMAGENET1K_V2),
    ("vit_b_16", vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
]
REPO_ROOT = Path(__file__).resolve().parents[1]


def check_class_mapping(class_names, label_synsets):
    """Check that dataset labels match the output class order of both models."""
    resnet_categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    vit_categories = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]
    if resnet_categories != vit_categories:
        raise ValueError("The two models have different output class mappings.")

    # Ba lớp này có tên viết khác nhau nhưng đã xác minh là cùng một lớp.
    # Synset phân biệt chim sếu và cần cẩu, đều có tên tiếng Anh là 'crane'.
    aliases = {
        134: ("crane", "crane bird", "n02012849"),
        517: ("crane2", "crane", "n03126707"),
        639: ("maillot, tank suit", "maillot tank suit", "n03710721"),
    }
    for label, dataset_name in enumerate(class_names):
        model_class_name = resnet_categories[label]
        if label in aliases:
            compatible = (dataset_name, model_class_name, label_synsets[label]) == aliases[label]
        else:
            # Một lớp có thể có nhiều tên đồng nghĩa, ngăn cách bởi dấu phẩy.
            compatible = model_class_name in dataset_name.split(", ")
        if not compatible:
            raise ValueError(f"Unclear class mapping at index {label}: {dataset_name!r}, {model_class_name!r}")


def inspect_dataset(data_dir):
    """Read filenames and labels to validate the dataset without decoding images."""
    # Sắp xếp file để thứ tự ảnh nhất quán giữa hai model.
    validation_files = sorted((data_dir / "data").glob("validation-*.parquet"))
    expected_filenames = [f"validation-{i:05d}-of-00014.parquet" for i in range(14)]
    if [p.name for p in validation_files] != expected_filenames:
        raise ValueError("Expected all 14 local validation shards. No data will be downloaded.")

    image_records = []
    class_counts = Counter()
    label_synsets = {}
    class_names = None
    for parquet_path in validation_files:
        # Đọc metadata nằm sẵn trong file trên máy, không kết nối Hugging Face.
        metadata = pq.ParquetFile(parquet_path).schema_arrow.metadata or {}
        if b"huggingface" not in metadata:
            raise ValueError(f"Missing Hugging Face class metadata: {parquet_path.name}")
        # Lấy thông tin các cột; mục label chứa danh sách tên lớp.
        features = json.loads(metadata[b"huggingface"])["info"]["features"]
        shard_class_names = features["label"]["names"]
        # Lưu tên lớp từ file đầu tiên làm mốc đối chiếu các file còn lại.
        if class_names is None:
            class_names = shard_class_names
        if shard_class_names != class_names or len(shard_class_names) != 1000:
            raise ValueError(f"Missing or inconsistent 1,000-class mapping: {parquet_path.name}")

        # Đổi cột thành danh sách để duyệt từng tên ảnh cùng nhãn tương ứng.
        table = pq.read_table(parquet_path, columns=["image.path", "label"])
        image_ids = table.column(0).to_pylist()
        labels = table.column(1).to_pylist()
        for image_id, label in zip(image_ids, labels):
            # Ví dụ: ILSVRC2012_val_00027555_n01824575.JPEG
            match = re.fullmatch(r"ILSVRC2012_val_\d{8}_(n\d{8})\.JPEG", image_id or "")
            if not match or not isinstance(label, int) or not 0 <= label < 1000:
                raise ValueError(f"Unrecognized image ID or label: {image_id}, {label}")
            # Lấy mã lớp từ tên ảnh, ví dụ n01824575.
            synset = match.group(1)
            if label in label_synsets and label_synsets[label] != synset:
                raise ValueError(f"Label {label} refers to multiple synsets.")
            label_synsets[label] = synset
            # Đếm ảnh theo lớp; lưu tên ảnh và nhãn vào danh sách dùng để đánh giá.
            class_counts[label] += 1
            image_records.append({"dataset_index": len(image_records), "image_id": image_id, "label": label})

    # Cần đủ 50.000 ảnh khác nhau để accuracy phản ánh toàn bộ validation.
    if len(image_records) != 50000 or len({r["image_id"] for r in image_records}) != 50000:
        raise ValueError("Expected exactly 50,000 distinct validation images.")
    if set(class_counts) != set(range(1000)) or set(class_counts.values()) != {50}:
        raise ValueError("Expected labels 0..999 with 50 images per class.")
    ordered_synsets = [label_synsets[i] for i in range(1000)]
    if len(set(ordered_synsets)) != 1000 or ordered_synsets != sorted(ordered_synsets):
        raise ValueError("Labels do not follow ImageNet's sorted synset order.")

    check_class_mapping(class_names, label_synsets)
    return validation_files, image_records


def load_batches(validation_files, transform, batch_size):
    """Read every validation image in small batches to limit RAM usage."""
    images, labels, image_ids = [], [], []
    for parquet_path in validation_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=64, columns=["image", "label"]):
            for row in batch.to_pylist():
                # Giải mã byte ảnh trong bộ nhớ và chuyển sang RGB trước khi preprocessing.
                with Image.open(BytesIO(row["image"]["bytes"])) as image:
                    images.append(transform(image.convert("RGB")))
                labels.append(row["label"])
                image_ids.append(row["image"]["path"])
                if len(images) == batch_size:
                    # Ghép ảnh thành tensor batch rồi trả về để model xử lý.
                    yield torch.stack(images), labels, image_ids
                    images, labels, image_ids = [], [], []
    # Không bỏ sót batch cuối dù số ảnh ít hơn batch_size.
    if images:
        # Ghép ảnh thành tensor batch rồi trả về để model xử lý.
        yield torch.stack(images), labels, image_ids


def evaluate_model(model_name, create_model, weights, validation_files, image_records, device, batch_size, output_dir):
    """Load one model, evaluate it, save predictions, and release its memory."""
    transform = weights.transforms()
    batches = load_batches(validation_files, transform, batch_size)
    # Preprocessing phải tương ứng với bộ trọng số đang dùng.
    # Tải đúng trọng số pretrained và đưa model lên GPU hoặc CPU.
    model = create_model(weights=weights).to(device)
    model.eval()  # Dùng thống kê đã học và tắt dropout khi đánh giá.
    total = 0
    top1_correct = 0
    top5_correct = 0
    predictions_path = output_dir / f"{model_name}_predictions.csv"
    num_batches = (len(image_records) + batch_size - 1) // batch_size
    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "image_id", "label", "pred_top1", "pred_top2", "pred_top3",
            "pred_top4", "pred_top5", "top1_correct", "top5_correct",
        ])
        # Chỉ dự đoán, không cập nhật trọng số nên không cần lưu gradient.
        with torch.inference_mode():
            for images, labels, image_ids in tqdm(batches, total=num_batches, desc=model_name):
                class_scores = model(images.to(device))
                if class_scores.shape != (len(labels), 1000):
                    raise ValueError(f"Unexpected model output shape: {class_scores.shape}")
                # Lấy chỉ số năm lớp có điểm cao nhất, theo thứ tự giảm dần.
                top5_indices = class_scores.topk(5, dim=1).indices
                top5_predictions = top5_indices.cpu().tolist()
                del top5_indices  # Không giữ tensor dự đoán trên GPU sau khi đã chuyển kết quả về CPU.
                for image_id, label, predicted_classes in zip(image_ids, labels, top5_predictions):
                    # Top-1: lớp đầu đúng; Top-5: nhãn thật nằm trong năm lớp dự đoán.
                    is_top1_correct = int(predicted_classes[0] == label)
                    is_top5_correct = int(label in predicted_classes)
                    csv_writer.writerow([image_id, label, *predicted_classes, is_top1_correct, is_top5_correct])
                    total += 1
                    top1_correct += is_top1_correct
                    top5_correct += is_top5_correct
                del images, class_scores
    if total != len(image_records):
        raise ValueError(f"Evaluated {total} images; expected {len(image_records)}.")
    # Dọn model và cache GPU trước khi tải model tiếp theo.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Accuracy (%) = số ảnh dự đoán đúng / tổng số ảnh * 100.
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return {
        "model": model_name,
        "weights": str(weights),
        "device": str(device),
        "num_images": total,
        "top1_accuracy_percent": top1_accuracy,
        "top5_accuracy_percent": top5_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/imagenet")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/metrics/baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check-data", action="store_true", help="Validate all 50,000 images and labels without inference or writing files")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    # Bước 1: kiểm tra nhãn của toàn bộ validation trước khi tải model.
    validation_files, image_records = inspect_dataset(args.data_dir)
    print("Verified 50,000 images, 1,000 classes, and torchvision label compatibility.")
    print(f"Full validation images: {len(image_records)}")
    if args.check_data:
        return

    # Bước 2: chọn thiết bị và lưu danh sách ảnh dùng chung cho hai model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    output_dir = args.output_dir / "full"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_images.csv").open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["dataset_index", "image_id", "label"])
        csv_writer.writeheader()
        csv_writer.writerows(image_records)

    # Bước 3: chạy từng model lần lượt và lưu bảng accuracy.
    summary_columns = [
        "mode", "model", "weights", "device", "num_images",
        "top1_accuracy_percent", "top5_accuracy_percent",
        "torch_version", "torchvision_version",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=summary_columns)
        csv_writer.writeheader()
        for model_name, create_model, weights in MODEL_SPECS:
            result = evaluate_model(
                model_name, create_model, weights, validation_files, image_records,
                device, args.batch_size, output_dir,
            )
            result["mode"] = "full"
            result["torch_version"] = torch.__version__
            result["torchvision_version"] = torchvision.__version__
            csv_writer.writerow(result)
            csv_file.flush()  # Lưu kết quả model đầu tiên trước khi chạy model thứ hai.
            print(f"{model_name}: Top-1 {result['top1_accuracy_percent']:.3f}%, Top-5 {result['top5_accuracy_percent']:.3f}%")
    print(f"Saved CSV files to {output_dir}")


if __name__ == "__main__":
    main()
