"""Small inference, accuracy and CSV helpers; no global gradient-mode changes."""

import csv
from pathlib import Path

import torch

# Schema chung: mỗi hàng xác định ảnh, model, điều kiện và năm lớp dự đoán.
PREDICTION_COLUMNS = [
    "image_id", "model_name", "label", "pred_top1", "pred_top2", "pred_top3",
    "pred_top4", "pred_top5", "top1_correct", "top5_correct",
    "experiment", "condition", "severity",
]


def accuracy_from_predictions(rows):
    """Compute percent Top-1/Top-5 from labels and ranked predictions, not flags."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot measure accuracy on an empty cohort")
    # Tính lại từ dự đoán và nhãn thật, không tin sẵn các cờ correctness trong CSV.
    top1 = top5 = 0
    for row in rows:
        label = int(row["label"])
        predictions = [int(row[f"pred_top{i}"]) for i in range(1, 6)]
        if not 0 <= label < 1000 or any(not 0 <= p < 1000 for p in predictions):
            raise ValueError("Expected ImageNet class indices 0..999")
        if len(set(predictions)) != 5:
            raise ValueError("Top-5 predictions must be distinct")
        # Top-1: lớp đứng đầu đúng; Top-5: nhãn thật nằm trong năm lớp đứng đầu.
        top1 += predictions[0] == label
        top5 += label in predictions
    # Accuracy trả về theo phần trăm (0..100), không phải tỷ lệ (0..1).
    return {"num_images": len(rows), "top1_accuracy_percent": 100 * top1 / len(rows),
            "top5_accuracy_percent": 100 * top5 / len(rows)}


def evaluate(model, transform, samples, *, model_name, batch_size=32,
             experiment="clean", condition="clean", severity=""):
    """Return (prediction_rows, metrics) for iterable (RGB, label, ID) samples.

    Apply perturbations in the sample iterator when appropriate; their spatial
    definition must be agreed separately. Model must already be in eval mode.
    Inference mode is scoped to each forward pass, never across iterator yields.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    # Không tự đổi trạng thái model của người gọi; yêu cầu eval() trước khi inference.
    if any(module.training for module in model.modules()):
        raise ValueError("Call model.eval() before inference")
    device = next(model.parameters()).device
    rows, images, labels, ids = [], [], [], []
    seen = set()

    # Xử lý batch đang tích lũy rồi xóa bộ đệm để nhận batch tiếp theo.
    def flush():
        # Chỉ tắt gradient trong lượt dự đoán này; không ảnh hưởng code bên ngoài.
        # Hàm evaluate không dùng để tính Grad-CAM/ERF; các phương pháp đó gọi model trực tiếp.
        with torch.inference_mode():
            scores = model(torch.stack(images).to(device))
            if scores.shape != (len(images), 1000) or not torch.isfinite(scores).all():
                raise ValueError("Expected finite model output of shape (batch, 1000)")
            # Chuyển chỉ số lớp về CPU để lưu kết quả, không giữ tensor output trên GPU.
            predictions = scores.topk(5, dim=1).indices.cpu().tolist()
        for image_id, label, predicted in zip(ids, labels, predictions):
            row = dict(image_id=image_id, model_name=model_name, label=label,
                       experiment=experiment, condition=condition, severity=severity,
                       top1_correct=int(predicted[0] == label),
                       top5_correct=int(label in predicted))
            row.update({f"pred_top{i}": p for i, p in enumerate(predicted, 1)})
            rows.append(row)
        images.clear()
        labels.clear()
        ids.clear()

    # Mỗi lượt evaluate xử lý một model/điều kiện; mỗi image_id chỉ được xuất hiện một lần.
    for image, label, image_id in samples:
        label = int(label)
        if not image_id or image_id in seen or not 0 <= label < 1000:
            raise ValueError(f"Invalid or duplicate sample: {image_id}")
        seen.add(image_id)
        # Áp dụng transform của đúng checkpoint trước khi ghép tensor thành batch.
        images.append(transform(image))
        labels.append(label)
        ids.append(image_id)
        if len(images) == batch_size:
            flush()
    # Không bỏ sót batch cuối nếu số ảnh không chia hết cho batch_size.
    if images:
        flush()
    return rows, accuracy_from_predictions(rows)


def accuracy_drop(clean_rows, perturbed_rows, *, model_name):
    """Compare one model/condition; select clean rows by perturbed IDs.

    Clean rows can be the existing 50k baseline CSV (model inferred by caller
    from filename). Reject duplicate IDs, missing IDs and changed labels.
    """
    # Lập bảng tra theo ID, đồng thời từ chối dữ liệu trùng hoặc trộn nhiều điều kiện.
    def index(rows):
        result = {}
        conditions = set()
        for row in rows:
            # CSV baseline cũ không có model_name: người gọi phải chọn đúng file model.
            if row.get("model_name", model_name) != model_name:
                raise ValueError("Pass rows for exactly the requested model")
            image_id = row["image_id"]
            if not image_id or image_id in result:
                raise ValueError("Each input must have unique nonempty image IDs")
            conditions.add((row.get("experiment", ""), row.get("condition", ""), str(row.get("severity", ""))))
            result[image_id] = row
        if len(conditions) > 1:
            raise ValueError("Compare one experiment/condition/severity at a time")
        return result

    clean, perturbed = index(clean_rows), index(perturbed_rows)
    if not perturbed or not perturbed.keys() <= clean.keys():
        raise ValueError("Every perturbed ID must exist in the clean baseline")
    # Chỉ lấy clean trên đúng tập ID đã đánh giá, không dùng accuracy của cả 50.000 ảnh.
    ids = sorted(perturbed)
    if any(int(clean[i]["label"]) != int(perturbed[i]["label"]) for i in ids):
        raise ValueError("Clean and perturbed ground-truth labels differ")
    clean_metrics = accuracy_from_predictions(clean[i] for i in ids)
    changed_metrics = accuracy_from_predictions(perturbed[i] for i in ids)
    result = {"model_name": model_name, "num_images": len(ids)}
    # Tính drop riêng cho Top-1 và Top-5; giá trị âm nghĩa là accuracy tăng.
    for k in (1, 5):
        before = clean_metrics[f"top{k}_accuracy_percent"]
        after = changed_metrics[f"top{k}_accuracy_percent"]
        # Drop tuyệt đối = clean - changed (điểm phần trăm).
        # Drop tương đối = 100 * (clean - changed) / clean; clean=0 thì trả None.
        result.update({f"clean_top{k}_accuracy_percent": before,
                       f"perturbed_top{k}_accuracy_percent": after,
                       f"absolute_top{k}_drop_pp": before - after,
                       f"relative_top{k}_drop_percent": 100 * (before - after) / before if before else None})
    return result


def save_predictions(rows, path, *, overwrite=False):
    """Write the shared schema; refuse overwrites unless explicitly requested."""
    rows = list(rows)
    if not rows:
        raise ValueError("No predictions to save")
    # Kiểm tra toàn bộ dữ liệu trước khi mở file, tránh ghi CSV với schema sai.
    keys = set()
    for row in rows:
        if set(row) != set(PREDICTION_COLUMNS):
            raise ValueError("Prediction columns do not match the shared schema")
        if any(not row[column] for column in ["image_id", "model_name", "experiment", "condition"]):
            raise ValueError("ID, model, experiment and condition must be nonempty")
        # Cùng ảnh có thể có nhiều điều kiện, nhưng không được trùng toàn bộ khóa này.
        key = tuple(str(row[c]) for c in ["image_id", "model_name", "experiment", "condition", "severity"])
        if key in keys:
            raise ValueError("Duplicate prediction key")
        keys.add(key)
        # Xác nhận nhãn/predictions hợp lệ và cờ đúng/sai khớp với dữ liệu thực tế.
        accuracy_from_predictions([row])
        if (int(row["top1_correct"]) != int(int(row["label"]) == int(row["pred_top1"]))
                or int(row["top5_correct"]) != int(int(row["label"]) in [int(row[f"pred_top{i}"]) for i in range(1, 6)])):
            raise ValueError("Correctness flags do not match predictions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Chế độ x từ chối ghi đè; chỉ dùng w khi người gọi chủ động bật overwrite=True.
    with path.open("w" if overwrite else "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
