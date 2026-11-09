"""Occlusion logic extracted from 06_occlusion.ipynb; importing runs nothing."""

import hashlib
from itertools import islice
import math
import random

import pandas as pd
import torch
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from src.data import iter_analysis_images
from src.evaluation import accuracy_drop, evaluate

MODEL_NAMES = ("resnet50", "vit_b_16")
SEVERITIES = (0, 10, 20, 30, 40, 50)
POSITIONS = ("center", "random")
FILL_RGB = (128, 128, 128)


def resize_crop(image, transform):
    """Apply the checkpoint's official spatial preprocessing to a PIL image."""
    # Tách resize/crop khỏi normalization để che trên ảnh RGB 224x224.
    image = TF.resize(image, transform.resize_size,
                      interpolation=transform.interpolation, antialias=transform.antialias)
    return TF.center_crop(image, transform.crop_size)


def normalize_image(image, transform):
    """Convert a cropped PIL image to a normalized input tensor."""
    # Giữ đúng thứ tự và kiểu float của transform baseline; không resize lần nữa.
    tensor = TF.convert_image_dtype(TF.pil_to_tensor(image), torch.float)
    return TF.normalize(tensor, mean=transform.mean, std=transform.std)


def mask_box(image_id, severity, position, seed=42):
    """Return (left, top, right, bottom); severity is nominal area percent."""
    # Căn bậc hai đổi tỷ lệ diện tích thành cạnh vuông, sau đó làm tròn pixel.
    side = round(224 * math.sqrt(severity / 100))
    if severity == 0:
        return (0, 0, 0, 0)
    if position == "center":
        left = top = (224 - side) // 2
    elif position == "random":
        # Không chứa tên model: cùng ID/severity/seed sẽ có cùng tọa độ cho hai model.
        key = f"{seed}|{image_id}|{severity}".encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
        left, top = rng.randint(0, 224 - side), rng.randint(0, 224 - side)
    else:
        raise ValueError("position must be 'center' or 'random'")
    return (left, top, left + side, top + side)


def apply_occlusion(image, image_id, severity, position, seed=42):
    """Mask an already cropped RGB image, leaving the clean image unchanged."""
    masked = image.copy()
    if severity:
        masked.paste(FILL_RGB, mask_box(image_id, severity, position, seed))
    return masked


def run_experiment(records, models, transforms, batch_size=16, seed=42):
    """Return predictions by model for all fixed occlusion conditions.

    Pass manifest records and dictionaries returned by the shared model loader.
    This function performs inference; it does not load weights or write files.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    assert set(models) == set(transforms) == set(MODEL_NAMES)
    records = list(records)
    expected_labels = {row["image_id"]: int(row["label"]) for row in records}
    assert records and len(expected_labels) == len(records)
    predictions = {name: [] for name in MODEL_NAMES}
    image_stream = iter_analysis_images(records)
    seen_labels = {}

    # Đọc RGB một lần theo batch, không giữ cả 10.000 ảnh trong RAM.
    with tqdm(total=len(records), desc="Occlusion images") as progress:
        while True:
            batch = list(islice(image_stream, batch_size))
            if not batch:
                break
            seen_labels.update({image_id: label for _, label, image_id in batch})
            for name in MODEL_NAMES:
                transform = transforms[name]
                cropped_samples = [(resize_crop(image, transform), label, image_id)
                                   for image, label, image_id in batch]
                normalize = lambda image: normalize_image(image, transform)
                # Clean chỉ tính một lần; severity 0 được dùng lại cho cả hai vị trí.
                rows, _ = evaluate(models[name], normalize, cropped_samples,
                                   model_name=name, batch_size=batch_size,
                                   experiment="occlusion", condition="clean", severity=0)
                predictions[name].extend(rows)
                for position in POSITIONS:
                    for severity in SEVERITIES[1:]:
                        masked_samples = (
                            (apply_occlusion(image, image_id, severity, position, seed), label, image_id)
                            for image, label, image_id in cropped_samples
                        )
                        rows, _ = evaluate(models[name], normalize, masked_samples,
                                           model_name=name, batch_size=batch_size,
                                           experiment="occlusion", condition=position, severity=severity)
                        predictions[name].extend(rows)
            progress.update(len(batch))

    assert seen_labels == expected_labels
    assert all(len(predictions[name]) == len(records) * 11 for name in MODEL_NAMES)
    return predictions


def summarize_predictions(predictions):
    """Return the notebook's 24-row metrics DataFrame, also accepting CSV rows."""
    metric_rows = []
    for name in MODEL_NAMES:
        rows = predictions[name]
        clean_rows = [row for row in rows if int(row["severity"]) == 0]
        clean_by_id = {row["image_id"]: row for row in clean_rows}
        n_images = len(clean_by_id)
        assert n_images > 0 and len(clean_rows) == n_images
        for position in POSITIONS:
            for severity in SEVERITIES:
                current = clean_rows if severity == 0 else [
                    row for row in rows
                    if row["condition"] == position and int(row["severity"]) == severity
                ]
                assert len(current) == n_images and {row["image_id"] for row in current} == set(clean_by_id)
                # Dùng lại shared helper để tính drop trên đúng cùng tập ID/nhãn.
                drops = accuracy_drop(clean_rows, current, model_name=name)
                consistency = 100 * sum(
                    int(row["pred_top1"]) == int(clean_by_id[row["image_id"]]["pred_top1"])
                    for row in current
                ) / n_images
                metric_rows.append({
                    "model_name": name, "position": position, "severity": severity,
                    "n_images": drops["num_images"],
                    "top1_accuracy": drops["perturbed_top1_accuracy_percent"],
                    "top5_accuracy": drops["perturbed_top5_accuracy_percent"],
                    "absolute_drop_pp": drops["absolute_top1_drop_pp"],
                    "relative_drop_pct": drops["relative_top1_drop_percent"],
                    "prediction_consistency": consistency,
                    "absolute_top5_drop_pp": drops["absolute_top5_drop_pp"],
                    "relative_top5_drop_pct": drops["relative_top5_drop_percent"],
                })
    return pd.DataFrame(metric_rows)
