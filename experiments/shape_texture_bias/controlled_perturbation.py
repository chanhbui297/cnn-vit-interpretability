"""Reusable controlled cue-perturbation utilities for TV4.

Blur is applied to source RGB pixels; patch shuffle permutes fixed-size image
tiles using a stable image-id seed.  Both happen before each checkpoint's
official preprocessing and are therefore shared across ResNet-50 and ViT-B/16.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

from experiments.shape_texture_bias.official_mapping import decisions_from_probabilities


CONDITIONS = {
    "clean": {"family": "clean", "severity": 0},
    "blur_s1": {"family": "gaussian_blur", "severity": 1, "kernel_size": 7, "sigma": 1.5},
    "blur_s2": {"family": "gaussian_blur", "severity": 2, "kernel_size": 13, "sigma": 3.0},
    "shuffle_48": {"family": "patch_shuffle", "severity": 1, "tile_size": 48},
    "shuffle_32": {"family": "patch_shuffle", "severity": 2, "tile_size": 32},
}
DECISIONS = ("shape", "texture", "neither")


def cue_decision(category: str, record: dict) -> str:
    if record["shape_cue"] == record["texture_cue"]:
        return "excluded_same_cue"
    if category == record["shape_cue"]:
        return "shape"
    if category == record["texture_cue"]:
        return "texture"
    return "neither"


def _stable_seed(image_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def patch_shuffle(image: Image.Image, tile_size: int, image_id: str, seed: int) -> Image.Image:
    """Permute complete tiles, retaining partial edge tiles in their positions.

    This deliberately disrupts global layout but does not remove all local
    structure or texture; the deterministic permutation is common to models.
    """
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    rows, cols = height // tile_size, width // tile_size
    if rows < 2 or cols < 2:
        raise ValueError(f"Image {image_id} is too small for {tile_size}px tiles: {width}×{height}")
    shuffled = array.copy()
    positions = [(row, col) for row in range(rows) for col in range(cols)]
    permutation = np.random.default_rng(_stable_seed(image_id, seed)).permutation(len(positions))
    tiles = [array[row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size].copy()
             for row, col in positions]
    for (row, col), source_index in zip(positions, permutation):
        shuffled[row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size] = tiles[source_index]
    return Image.fromarray(shuffled, mode="RGB")


def apply_condition(image: Image.Image, condition: str, image_id: str, seed: int = 42) -> Image.Image:
    """Return an RGB image of the original dimensions for one pre-registered condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose {list(CONDITIONS)}")
    params = CONDITIONS[condition]
    rgb = image.convert("RGB")
    if params["family"] == "clean":
        return rgb.copy()
    if params["family"] == "gaussian_blur":
        array = np.asarray(rgb)
        result = cv2.GaussianBlur(array, (params["kernel_size"], params["kernel_size"]), params["sigma"])
        return Image.fromarray(result, mode="RGB")
    result = patch_shuffle(rgb, params["tile_size"], image_id, seed)
    if result.size != rgb.size:
        raise RuntimeError("Patch shuffle changed image dimensions")
    return result


def stratified_pilot_records(records: list[dict], per_shape: int = 10) -> list[dict]:
    """Deterministic valid-conflict pilot cohort, balanced by shape cue."""
    output = []
    for category in sorted({r["shape_cue"] for r in records}):
        candidates = sorted((r for r in records if r["shape_cue"] == category and r["shape_cue"] != r["texture_cue"]),
                            key=lambda r: r["file_id"])
        if len(candidates) < per_shape:
            raise ValueError(f"Only {len(candidates)} valid records for {category}")
        output.extend(candidates[:per_shape])
    return output


def evaluate_condition(model, transform, model_name: str, records: list[dict], condition: str,
                       batch_size: int = 64, seed: int = 42) -> list[dict]:
    """Run one model/condition directly on records and return cue-decision rows."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    device = next(model.parameters()).device
    output, images, batch_records = [], [], []

    def flush():
        if not images:
            return
        inputs = torch.stack(images).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = torch.softmax(model(inputs).float(), dim=1).cpu().numpy()
        categories, scores = decisions_from_probabilities(probabilities)
        for record, category, score in zip(batch_records, categories, scores):
            output.append({
                "image_id": record["file_id"], "shape_cue": record["shape_cue"], "texture_cue": record["texture_cue"],
                "model": model_name, "condition": condition, "family": CONDITIONS[condition]["family"],
                "severity": CONDITIONS[condition]["severity"], "predicted_category": category,
                "category_confidence": float(score), "decision": cue_decision(category, record),
            })
        images.clear(); batch_records.clear()

    for record in records:
        altered = apply_condition(record["image"], condition, record["file_id"], seed)
        images.append(transform(altered)); batch_records.append(record)
        if len(images) >= batch_size:
            flush()
    flush()
    if len(output) != len(records) or len({r["image_id"] for r in output}) != len(records):
        raise RuntimeError("Condition evaluation did not return one unique prediction per image")
    return output


def summarize_rows(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["decision"] != "excluded_same_cue"]
    counts = Counter(row["decision"] for row in valid)
    denominator = counts["shape"] + counts["texture"]
    return {
        "num_valid": len(valid), "shape_decisions": counts["shape"], "texture_decisions": counts["texture"],
        "neither_decisions": counts["neither"], "cue_decisions": denominator,
        "cue_decision_coverage": denominator / len(valid) if valid else np.nan,
        "shape_bias": counts["shape"] / denominator if denominator else np.nan,
    }


def paired_bootstrap(clean_rows: list[dict], changed_rows: list[dict], seed: int = 42, n_bootstrap: int = 4000) -> dict:
    """Paired CI for changed-minus-clean Shape Bias and cue-decision coverage."""
    clean = {r["image_id"]: r for r in clean_rows if r["decision"] != "excluded_same_cue"}
    changed = {r["image_id"]: r for r in changed_rows if r["decision"] != "excluded_same_cue"}
    if clean.keys() != changed.keys():
        raise ValueError("Paired bootstrap requires identical valid image IDs")
    ids = np.array(sorted(clean))
    before = np.array([clean[i]["decision"] for i in ids], dtype=object)
    after = np.array([changed[i]["decision"] for i in ids], dtype=object)
    rng = np.random.default_rng(seed)
    bias_deltas, coverage_deltas = [], []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(ids), len(ids))
        def metrics(decisions):
            shape, texture = (decisions == "shape").sum(), (decisions == "texture").sum()
            cue = shape + texture
            return shape / cue if cue else np.nan, cue / len(decisions)
        clean_bias, clean_coverage = metrics(before[sample])
        changed_bias, changed_coverage = metrics(after[sample])
        if np.isfinite(clean_bias) and np.isfinite(changed_bias):
            bias_deltas.append(changed_bias - clean_bias)
        coverage_deltas.append(changed_coverage - clean_coverage)
    return {
        "shape_bias_delta": summarize_rows(changed_rows)["shape_bias"] - summarize_rows(clean_rows)["shape_bias"],
        "shape_bias_delta_ci95_low": float(np.quantile(bias_deltas, .025)),
        "shape_bias_delta_ci95_high": float(np.quantile(bias_deltas, .975)),
        "coverage_delta": summarize_rows(changed_rows)["cue_decision_coverage"] - summarize_rows(clean_rows)["cue_decision_coverage"],
        "coverage_delta_ci95_low": float(np.quantile(coverage_deltas, .025)),
        "coverage_delta_ci95_high": float(np.quantile(coverage_deltas, .975)),
    }


def transition_counts(clean_rows: list[dict], changed_rows: list[dict]) -> list[dict]:
    clean = {row["image_id"]: row for row in clean_rows if row["decision"] != "excluded_same_cue"}
    changed = {row["image_id"]: row for row in changed_rows if row["decision"] != "excluded_same_cue"}
    if clean.keys() != changed.keys():
        raise ValueError("Transitions require equal valid image IDs")
    total = len(clean)
    return [{
        "from_decision": before, "to_decision": after,
        "count": sum(clean[i]["decision"] == before and changed[i]["decision"] == after for i in clean),
        "percent_of_images": 100 * sum(clean[i]["decision"] == before and changed[i]["decision"] == after for i in clean) / total,
    } for before in DECISIONS for after in DECISIONS]
