"""Cue-conflict texture-detail robustness utilities for TV4.

This experiment measures mapped Geirhos cue decisions, not ImageNet top-1
accuracy.  All transforms operate on source RGB pixels before each model's
official torchvision preprocessing.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

from experiments.shape_texture_bias.controlled_perturbation import cue_decision
from experiments.shape_texture_bias.official_mapping import decisions_from_probabilities

DATASET_ID = "rgeirhos/texture-shape-cue-conflict"
DATASET_SPLIT = "test"
SEED = 42
MODELS = ("resnet50", "vit_b_16")
CONDITIONS = {
    "clean": {"family": "clean", "severity": 0},
    # These two Gaussian settings intentionally equal the completed controlled
    # cue-perturbation study, so verified predictions can be reused.
    "gaussian_s1": {"family": "gaussian_blur", "severity": 1, "kernel_size": 7, "sigma": 1.5},
    "gaussian_s2": {"family": "gaussian_blur", "severity": 2, "kernel_size": 13, "sigma": 3.0},
    "gaussian_s3": {"family": "gaussian_blur", "severity": 3, "kernel_size": 19, "sigma": 4.5},
    "bilateral_s1": {"family": "bilateral_filter", "severity": 1, "diameter": 5, "sigma_color": 25, "sigma_space": 25},
    "bilateral_s2": {"family": "bilateral_filter", "severity": 2, "diameter": 9, "sigma_color": 50, "sigma_space": 50},
    "bilateral_s3": {"family": "bilateral_filter", "severity": 3, "diameter": 13, "sigma_color": 100, "sigma_space": 100},
}
DECISIONS = ("shape", "texture", "neither")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def id_hash(records):
    return sha256_text("\\n".join(sorted(r["file_id"] for r in records)))


def load_valid_records():
    """Load exactly the 1,200 valid cue-conflict images used by Shape TV4."""
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, token=False)
    raw = [dataset[i] for i in range(len(dataset))]
    if len(raw) != 1280 or len({r["file_id"] for r in raw}) != 1280:
        raise ValueError("Expected 1,280 unique cue-conflict source records")
    valid = [r for r in raw if r["shape_cue"] != r["texture_cue"]]
    if len(valid) != 1200 or len({r["file_id"] for r in valid}) != 1200:
        raise ValueError("Expected exactly 1,200 valid cue-conflict images")
    categories = sorted({r["shape_cue"] for r in valid} | {r["texture_cue"] for r in valid})
    if len(categories) != 16:
        raise ValueError(f"Expected 16 cue categories, found {len(categories)}")
    return valid


def apply_transform(image: Image.Image, condition: str) -> Image.Image:
    """Apply one deterministic registered texture-detail transform."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    params = CONDITIONS[condition]
    rgb = image.convert("RGB")
    if params["family"] == "clean":
        return rgb.copy()
    array = np.asarray(rgb)
    if params["family"] == "gaussian_blur":
        out = cv2.GaussianBlur(array, (params["kernel_size"], params["kernel_size"]), params["sigma"])
    elif params["family"] == "bilateral_filter":
        out = cv2.bilateralFilter(array, params["diameter"], params["sigma_color"], params["sigma_space"])
    else:
        raise ValueError(params["family"])
    result = Image.fromarray(out, mode="RGB")
    if result.size != rgb.size:
        raise RuntimeError("Transformation changed image dimensions")
    return result


def evaluate_condition(model, transform, model_name, records, condition, batch_size=64):
    """Infer one full model × condition cohort and produce mapped decisions."""
    device = next(model.parameters()).device
    rows, tensors, batch = [], [], []

    def flush():
        if not tensors:
            return
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = torch.softmax(model(torch.stack(tensors).to(device)).float(), dim=1).cpu().numpy()
        categories, scores = decisions_from_probabilities(probabilities)
        for record, category, score in zip(batch, categories, scores):
            rows.append({
                "image_id": record["file_id"], "shape_cue": record["shape_cue"], "texture_cue": record["texture_cue"],
                "model": model_name, "condition": condition, "family": CONDITIONS[condition]["family"],
                "severity": CONDITIONS[condition]["severity"], "predicted_category": category,
                "category_confidence": float(score), "decision": cue_decision(category, record),
            })
        tensors.clear(); batch.clear()

    for record in records:
        tensors.append(transform(apply_transform(record["image"], condition)))
        batch.append(record)
        if len(tensors) >= batch_size:
            flush()
    flush()
    if not valid_prediction_rows(rows, records, model_name, condition):
        raise RuntimeError("Inference rows failed completeness/protocol validation")
    return rows


def valid_prediction_rows(rows, records, model_name=None, condition=None):
    expected = {r["file_id"]: (r["shape_cue"], r["texture_cue"]) for r in records}
    rows = list(rows)
    found = {r.get("image_id"): r for r in rows}
    return (
        len(rows) == len(expected) and len(found) == len(expected) and set(found) == set(expected)
        and all(
            row.get("shape_cue") == expected[image_id][0]
            and row.get("texture_cue") == expected[image_id][1]
            and row.get("model") == (model_name or row.get("model"))
            and row.get("condition") == (condition or row.get("condition"))
            and row.get("family") == CONDITIONS[row.get("condition")]["family"]
            and int(row.get("severity")) == CONDITIONS[row.get("condition")]["severity"]
            and row.get("decision") in DECISIONS
            and isinstance(row.get("predicted_category"), str) and bool(row.get("predicted_category"))
            for image_id, row in found.items()
        )
    )


def _pct(n, total):
    return 100 * n / total if total else float("nan")


def summarize_condition(clean_rows, rows):
    """Decision metrics specified for one paired model × condition cohort."""
    clean = {r["image_id"]: r for r in clean_rows}
    changed = {r["image_id"]: r for r in rows}
    if set(clean) != set(changed):
        raise ValueError("Clean and transformed ID sets differ")
    ids = sorted(clean)
    decisions = Counter(changed[i]["decision"] for i in ids)
    cue = decisions["shape"] + decisions["texture"]
    shape_base = [i for i in ids if clean[i]["decision"] == "shape"]
    texture_base = [i for i in ids if clean[i]["decision"] == "texture"]
    return {
        "num_images": len(ids),
        "prediction_consistency_percent": _pct(sum(changed[i]["predicted_category"] == clean[i]["predicted_category"] for i in ids), len(ids)),
        "shape_decisions": decisions["shape"], "texture_decisions": decisions["texture"], "neither_decisions": decisions["neither"],
        "shape_decision_rate_percent": _pct(decisions["shape"], len(ids)),
        "texture_decision_rate_percent": _pct(decisions["texture"], len(ids)),
        "neither_rate_percent": _pct(decisions["neither"], len(ids)),
        "cue_decision_coverage_percent": _pct(cue, len(ids)),
        "shape_bias_percent": _pct(decisions["shape"], cue),
        "shape_retention_percent": _pct(sum(changed[i]["decision"] == "shape" for i in shape_base), len(shape_base)),
        "texture_retention_percent": _pct(sum(changed[i]["decision"] == "texture" for i in texture_base), len(texture_base)),
        "clean_shape_base": len(shape_base), "clean_texture_base": len(texture_base),
    }


def transition_rows(clean_rows, rows):
    clean = {r["image_id"]: r for r in clean_rows}
    changed = {r["image_id"]: r for r in rows}
    if set(clean) != set(changed):
        raise ValueError("Transition cohorts differ")
    n = len(clean)
    return [{
        "from_decision": before, "to_decision": after,
        "count": sum(clean[i]["decision"] == before and changed[i]["decision"] == after for i in clean),
        "percent_of_images": _pct(sum(clean[i]["decision"] == before and changed[i]["decision"] == after for i in clean), n),
    } for before in DECISIONS for after in DECISIONS]


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write at {path}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def controlled_rows_for_texture(records, path):
    """Translate verified clean/blur S1/S2 rows from the Shape intervention."""
    source = load_csv(path)
    aliases = {"clean": "clean", "blur_s1": "gaussian_s1", "blur_s2": "gaussian_s2"}
    output = []
    for row in source:
        if row["condition"] not in aliases:
            continue
        condition = aliases[row["condition"]]
        output.append({
            "image_id": row["image_id"], "shape_cue": row["shape_cue"], "texture_cue": row["texture_cue"],
            "model": row["model"], "condition": condition, "family": CONDITIONS[condition]["family"],
            "severity": CONDITIONS[condition]["severity"], "predicted_category": row["predicted_category"],
            "category_confidence": float(row["category_confidence"]), "decision": row["decision"],
        })
    for model in MODELS:
        for condition in aliases.values():
            subset = [r for r in output if r["model"] == model and r["condition"] == condition]
            if not valid_prediction_rows(subset, records, model, condition):
                raise ValueError(f"Controlled cache is not valid for {model}/{condition}")
    return output
