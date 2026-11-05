#!/usr/bin/env python3
"""Đánh giá sensitivity to texture-detail degradation trên manifest 10,000 ảnh.

Chạy từ root (sau khi có dependencies và data/imagenet/data):
PYTHONPATH=.deps python3 experiments/texture/run_texture_robustness.py --allow-download

Gaussian blur và bilateral filtering được tác động lên RGB gốc, trước official
preprocessing riêng của ResNet-50 và ViT-B/16. Các thông số là fixed và được
ghi vào metadata; không thay đổi kích thước ảnh hay nhãn ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import iter_analysis_images, load_analysis_manifest  # noqa: E402
from src.models import load_model  # noqa: E402


RUN_ID = "tv4_texture_10000_v1"
CONDITIONS = {
    "gaussian_blur": {
        1: {"kernel_size": 5, "sigma": 1.0},
        2: {"kernel_size": 9, "sigma": 2.0},
        3: {"kernel_size": 15, "sigma": 4.0},
    },
    "bilateral_filter": {
        1: {"diameter": 5, "sigma_color": 25, "sigma_space": 25},
        2: {"diameter": 9, "sigma_color": 50, "sigma_space": 50},
        3: {"diameter": 13, "sigma_color": 100, "sigma_space": 100},
    },
}


def apply_transform(image, transformation: str, severity: int):
    """Biến đổi PIL RGB giữ nguyên W×H; parameters lấy từ CONDITIONS."""
    array = np.asarray(image.convert("RGB"))
    params = CONDITIONS[transformation][severity]
    if transformation == "gaussian_blur":
        result = cv2.GaussianBlur(array, (params["kernel_size"], params["kernel_size"]), params["sigma"])
    elif transformation == "bilateral_filter":
        result = cv2.bilateralFilter(array, params["diameter"], params["sigma_color"], params["sigma_space"])
    else:
        raise ValueError(f"Unknown transformation: {transformation}")
    from PIL import Image
    return Image.fromarray(result, mode="RGB")


def sha256_lf(path: Path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def baseline_rows(model_name: str, selected):
    path = ROOT / "results/metrics/baseline_notebook/full" / f"{model_name}_predictions.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        source = {row["image_id"]: row for row in csv.DictReader(stream)}
    selected_ids = {r["image_id"] for r in selected}
    if not selected_ids <= source.keys():
        raise ValueError(f"Baseline thiếu {len(selected_ids - source.keys())} selected IDs")
    rows = []
    for record in selected:
        item = source[record["image_id"]]
        if int(item["label"]) != record["label"]: raise ValueError("Baseline/manifest label mismatch")
        rows.append({
            "image_id": record["image_id"], "true_label": record["label"], "model": model_name,
            "transformation": "clean", "severity": 0, "predicted_label": int(item["pred_top1"]),
            "confidence": "", "correct": int(item["top1_correct"]),
        })
    return rows


def evaluate_condition(model, transform, model_name, transformation, severity, records, data_dir, batch_size):
    """Stream từng ảnh được chọn, biến đổi trước preprocessing, trả confidence top-1."""
    device = next(model.parameters()).device
    rows, tensors, labels, ids = [], [], [], []
    def flush():
        if not tensors: return
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(torch.stack(tensors).to(device))
            probabilities = torch.softmax(logits.float(), dim=1)
            confidence, predicted = probabilities.max(dim=1)
        for image_id, label, pred, conf in zip(ids, labels, predicted.cpu().tolist(), confidence.cpu().tolist()):
            rows.append({"image_id": image_id, "true_label": label, "model": model_name,
                         "transformation": transformation, "severity": severity,
                         "predicted_label": pred, "confidence": conf, "correct": int(pred == label)})
        tensors.clear(); labels.clear(); ids.clear()
    for image, label, image_id in iter_analysis_images(records, data_dir=data_dir):
        altered = apply_transform(image, transformation, severity)
        if altered.size != image.size: raise RuntimeError("Transformation changed dimensions")
        tensors.append(transform(altered)); labels.append(label); ids.append(image_id)
        if len(tensors) == batch_size: flush()
    flush()
    if len(rows) != len(records) or len({r["image_id"] for r in rows}) != len(rows):
        raise RuntimeError("Thiếu hoặc trùng prediction sau inference")
    return rows


def summarise(clean, changed, seed):
    clean_by_id, changed_by_id = ({r["image_id"]: r for r in clean}, {r["image_id"]: r for r in changed})
    if clean_by_id.keys() != changed_by_id.keys(): raise ValueError("Cohort IDs không đồng nhất")
    ids = sorted(clean_by_id)
    before = np.array([clean_by_id[i]["correct"] for i in ids], dtype=float)
    after = np.array([changed_by_id[i]["correct"] for i in ids], dtype=float)
    agreement = np.array([clean_by_id[i]["predicted_label"] == changed_by_id[i]["predicted_label"] for i in ids])
    rng = np.random.default_rng(seed)
    paired_drops = []
    for _ in range(2000):
        index = rng.integers(0, len(ids), len(ids))
        paired_drops.append(100 * (before[index].mean() - after[index].mean()))
    return {
        "num_images": len(ids), "clean_top1_accuracy_percent": 100 * before.mean(),
        "top1_accuracy_percent": 100 * after.mean(),
        "absolute_accuracy_drop_pp": 100 * (before.mean() - after.mean()),
        "accuracy_drop_ci95_low_pp": float(np.quantile(paired_drops, .025)),
        "accuracy_drop_ci95_high_pp": float(np.quantile(paired_drops, .975)),
        "prediction_consistency_percent": 100 * agreement.mean(),
    }


def save_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def make_figures(summary_rows, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
    colors = {"resnet50": "#377eb8", "vit_b_16": "#e41a1c"}
    for metric, title, ylabel, output in [
        ("top1_accuracy_percent", "Top-1 accuracy theo severity", "Top-1 accuracy (%)", "texture_accuracy_by_severity.png"),
        ("absolute_accuracy_drop_pp", "Accuracy drop theo severity", "Accuracy drop (pp)", "texture_accuracy_drop.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, transformation in zip(axes, CONDITIONS):
            for model in names:
                subset = [r for r in summary_rows if r["transformation"] == transformation and r["model"] == model]
                subset.sort(key=lambda row: row["severity"])
                ax.plot([r["severity"] for r in subset], [r[metric] for r in subset], marker="o", label=names[model], color=colors[model])
            ax.set_title(transformation.replace("_", " ").title()); ax.set_xlabel("Severity level"); ax.set_xticks([1, 2, 3]); ax.grid(alpha=.25)
        axes[0].set_ylabel(ylabel); axes[1].legend(); fig.suptitle(title); fig.tight_layout()
        fig.savefig(figure_dir / output, dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3); width = .18
    for offset, (transformation, model) in enumerate((t, m) for t in CONDITIONS for m in names):
        subset = sorted([r for r in summary_rows if r["transformation"] == transformation and r["model"] == model], key=lambda row: row["severity"])
        label = f"{names[model]} – {transformation.replace('_', ' ')}"
        ax.bar(x + (offset - 1.5) * width, [r["absolute_accuracy_drop_pp"] for r in subset], width, label=label)
    ax.set_xticks(x, ["Severity 1", "Severity 2", "Severity 3"]); ax.set_ylabel("Accuracy drop (pp)")
    ax.set_title("So sánh accuracy drop theo phép biến đổi"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figure_dir / "texture_accuracy_drop_bar.png", dpi=300); plt.close(fig)


def make_image_grid(records, data_dir, figure_dir):
    source = list(iter_analysis_images(records[:4], data_dir=data_dir))
    fig, axes = plt.subplots(len(source), 7, figsize=(16, 3 * len(source)))
    columns = [("clean", 0)] + [(t, s) for t in CONDITIONS for s in (1, 2, 3)]
    for row, (image, _, image_id) in zip(axes, source):
        for ax, (transformation, severity) in zip(row, columns):
            display = image if transformation == "clean" else apply_transform(image, transformation, severity)
            ax.imshow(display); ax.axis("off")
            if ax is row[0]: ax.set_ylabel(image_id[:18], fontsize=7)
            if row is axes[0]: ax.set_title("Original" if transformation == "clean" else f"{transformation}\nS{severity}", fontsize=8)
    fig.suptitle("Minh họa texture-detail degradation trước preprocessing", y=1.01); fig.tight_layout()
    fig.savefig(figure_dir / "texture_transformation_grid.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet/data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; không dùng báo cáo.")
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("--batch-size phải dương")
    seed_everything(args.seed)
    records = load_analysis_manifest(limit=args.limit)
    if args.limit is None and len(records) != 10000: raise RuntimeError("Cohort phải có 10,000 ảnh")
    # Kiểm tra loader sớm để báo lỗi dataset rõ ràng trước khi tải model.
    expected = [f"validation-{i:05d}-of-00014.parquet" for i in range(14)]
    if [p.name for p in sorted(args.data_dir.glob("validation-*.parquet"))] != expected:
        raise FileNotFoundError(f"Không đủ 14 ImageNet validation shards tại {args.data_dir}")

    out = ROOT / "results/tv4"; figures = out / "figures"
    all_rows, summary_rows, model_meta = [], [], {}
    for model_name in ("resnet50", "vit_b_16"):
        clean = baseline_rows(model_name, records)
        all_rows.extend(clean)
        model, transform, metadata = load_model(model_name, allow_download=args.allow_download)
        model_meta[model_name] = metadata
        for transformation in CONDITIONS:
            for severity in (1, 2, 3):
                changed = evaluate_condition(model, transform, model_name, transformation, severity, records, args.data_dir, args.batch_size)
                all_rows.extend(changed)
                summary_rows.append({"model": model_name, "transformation": transformation, "severity": severity,
                                     **summarise(clean, changed, args.seed + severity)})
        device = next(model.parameters()).device; del model
        if device.type == "cuda": torch.cuda.empty_cache()
    save_csv(out / "texture_predictions.csv", all_rows)
    save_csv(out / "texture_summary.csv", summary_rows)
    make_figures(summary_rows, figures); make_image_grid(records, args.data_dir, figures)
    (out / "texture_metadata.json").write_text(json.dumps({
        "run_id": RUN_ID, "seed": args.seed, "manifest": "data/manifests/analysis_10000_seed42.csv",
        "manifest_sha256_lf": sha256_lf(ROOT / "data/manifests/analysis_10000_seed42.csv"), "num_images": len(records),
        "data_dir": str(args.data_dir), "parameters": CONDITIONS, "batch_size": args.batch_size,
        "models": model_meta, "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
        "git_revision": git_revision(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Blur đánh giá sensitivity to texture-detail degradation, không cô lập texture bias thuần túy.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
