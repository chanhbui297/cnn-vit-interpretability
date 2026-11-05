#!/usr/bin/env python3
"""Chạy đầy đủ Shape-vs-Texture Bias với cue-conflict của Geirhos et al.

Ví dụ (từ root):
PYTHONPATH=.deps python3 experiments/shape_texture_bias/run_shape_texture_bias.py

Tập dữ liệu 1,280 ảnh được tải từ Hub chính thức nếu chưa có trong cache. Không
ghi dataset hoặc checkpoint vào Git. Kết quả có thể theo dõi ở ``results/tv4``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models import load_model  # noqa: E402
from experiments.shape_texture_bias.official_mapping import decisions_from_probabilities  # noqa: E402


DATASET_ID = "rgeirhos/texture-shape-cue-conflict"
RUN_ID = "tv4_geirhos_1280_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision() -> tuple[str, bool]:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return revision, dirty
    except Exception:
        return "unavailable", True


def run_model(records, model_name: str, batch_size: int, allow_download: bool):
    model, transform, model_metadata = load_model(model_name, allow_download=allow_download)
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        inputs = torch.stack([transform(record["image"].convert("RGB")) for record in batch]).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            top1 = logits.argmax(dim=1).cpu().tolist()
        categories, category_confidences = decisions_from_probabilities(probabilities)
        for record, index, category, confidence in zip(batch, top1, categories, category_confidences):
            shape, texture = record["shape_cue"], record["texture_cue"]
            # Trường hợp shape == texture không mang xung đột cue; loại khỏi bias.
            if shape == texture:
                decision = "excluded_same_cue"
            elif category == shape:
                decision = "shape"
            elif category == texture:
                decision = "texture"
            else:
                decision = "neither"
            rows.append({
                "image_id": record["file_id"], "shape_cue": shape, "texture_cue": texture,
                "model": model_name, "predicted_imagenet_label": index,
                "predicted_category": category, "category_confidence": float(confidence),
                "decision": decision,
            })
        del inputs, logits
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, model_metadata


def summarize(rows, seed: int, n_bootstrap: int = 2000):
    valid = [row for row in rows if row["decision"] != "excluded_same_cue"]
    counts = Counter(row["decision"] for row in valid)
    denominator = counts["shape"] + counts["texture"]
    shape_bias = counts["shape"] / denominator if denominator else None
    coverage = denominator / len(valid) if valid else None
    ci_low = ci_high = None
    if denominator:
        rng = np.random.default_rng(seed)
        labels = np.array([row["decision"] for row in valid], dtype=object)
        samples = []
        for _ in range(n_bootstrap):
            sampled = labels[rng.integers(0, len(labels), len(labels))]
            n_shape, n_texture = (sampled == "shape").sum(), (sampled == "texture").sum()
            if n_shape + n_texture:
                samples.append(n_shape / (n_shape + n_texture))
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975]).tolist()
    return {
        "num_all_images": len(rows), "num_valid_cue_conflict": len(valid),
        "shape_decisions": counts["shape"], "texture_decisions": counts["texture"],
        "neither_decisions": counts["neither"], "cue_decision_coverage": coverage,
        "shape_bias": shape_bias, "shape_bias_ci95_low": ci_low, "shape_bias_ci95_high": ci_high,
    }


def category_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["decision"] != "excluded_same_cue":
            grouped[row["shape_cue"]].append(row)
    output = []
    for category, items in sorted(grouped.items()):
        counts = Counter(row["decision"] for row in items)
        denom = counts["shape"] + counts["texture"]
        output.append({"shape_cue": category, "num_images": len(items), **dict(counts),
                       "shape_bias": counts["shape"] / denom if denom else None})
    return output


def save_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Không có dữ liệu để ghi: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_figures(all_rows, summaries, figure_dir: Path):
    figure_dir.mkdir(parents=True, exist_ok=True)
    models = ["resnet50", "vit_b_16"]
    labels = ["ResNet-50", "ViT-B/16"]
    values = [100 * summaries[m]["shape_bias"] if summaries[m]["shape_bias"] is not None else 0 for m in models]
    errors = [[100 * (summaries[m]["shape_bias"] - summaries[m]["shape_bias_ci95_low"]) if summaries[m]["shape_bias"] is not None else 0 for m in models],
              [100 * (summaries[m]["shape_bias_ci95_high"] - summaries[m]["shape_bias"]) if summaries[m]["shape_bias"] is not None else 0 for m in models]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, yerr=errors, capsize=5, color=["#377eb8", "#e41a1c"])
    ax.set_ylim(0, 100); ax.set_ylabel("Shape Bias (%)"); ax.set_title("Shape Bias trên Geirhos cue-conflict (95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(figure_dir / "shape_bias_comparison.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(2)
    colors = {"shape": "#4daf4a", "texture": "#ff7f00", "neither": "#999999"}
    for decision in ("shape", "texture", "neither"):
        values = np.array([summaries[m][f"{decision}_decisions"] for m in models])
        ax.bar(labels, values, bottom=bottom, label={"shape": "Shape", "texture": "Texture", "neither": "Neither"}[decision], color=colors[decision])
        bottom += values
    ax.set_ylabel("Số ảnh cue-conflict hợp lệ"); ax.set_title("Phân bố quyết định theo model"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "decision_stacked_bar.png", dpi=300); plt.close(fig)

    by_model = {m: category_summary([r for r in all_rows if r["model"] == m]) for m in models}
    categories = sorted({r["shape_cue"] for rows in by_model.values() for r in rows})
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(categories)); width = 0.38
    for offset, model, label, color in [(-width / 2, models[0], labels[0], "#377eb8"), (width / 2, models[1], labels[1], "#e41a1c")]:
        values = {r["shape_cue"]: r["shape_bias"] for r in by_model[model]}
        ax.bar(x + offset, [100 * values.get(c, 0) if values.get(c) is not None else 0 for c in categories], width, label=label, color=color)
    ax.set_xticks(x, categories, rotation=45, ha="right"); ax.set_ylim(0, 100)
    ax.set_ylabel("Shape Bias (%)"); ax.set_title("Shape Bias theo shape cue category"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "category_shape_bias.png", dpi=300); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true", help="Cho phép torchvision tải checkpoint ViT nếu chưa cache.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ smoke test; không dùng cho kết quả nghiên cứu.")
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("--batch-size phải dương")
    seed_everything(args.seed)
    dataset = load_dataset(DATASET_ID, split="test")
    records = [dataset[i] for i in range(len(dataset))]
    if len(records) != 1280: raise ValueError(f"Expected 1,280 images, got {len(records)}")
    if args.limit is not None: records = records[:args.limit]
    if any(not r["file_id"] or not r["shape_cue"] or not r["texture_cue"] for r in records):
        raise ValueError("Dataset thiếu file_id/shape_cue/texture_cue")
    if len({r["file_id"] for r in records}) != len(records): raise ValueError("file_id không duy nhất")

    all_rows, metadata = [], {}
    for model_name in ("resnet50", "vit_b_16"):
        rows, model_meta = run_model(records, model_name, args.batch_size, args.allow_download)
        all_rows.extend(rows); metadata[model_name] = model_meta
    if len(all_rows) != 2 * len(records): raise RuntimeError("Thiếu prediction rows")
    if len({(r["image_id"], r["model"]) for r in all_rows}) != len(all_rows): raise RuntimeError("Trùng prediction key")

    out = ROOT / "results" / "tv4"; figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    save_csv(out / "shape_texture_predictions.csv", all_rows)
    summaries = {m: summarize([r for r in all_rows if r["model"] == m], args.seed) for m in ("resnet50", "vit_b_16")}
    summary_rows = [{"model": m, **summaries[m]} for m in summaries]
    save_csv(out / "shape_texture_summary.csv", summary_rows)
    for model in summaries:
        save_csv(out / f"shape_texture_category_summary_{model}.csv", category_summary([r for r in all_rows if r["model"] == model]))
    make_figures(all_rows, summaries, figures)
    revision, dirty = git_revision()
    (out / "shape_texture_metadata.json").write_text(json.dumps({
        "run_id": RUN_ID, "dataset": DATASET_ID, "dataset_split": "test", "expected_images": 1280,
        "evaluated_images": len(records), "seed": args.seed, "batch_size": args.batch_size,
        "mapping": "Official Geirhos 16-category mean-softmax mapping", "models": metadata,
        "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
        "git_revision": revision, "git_tree_dirty": dirty, "created_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
