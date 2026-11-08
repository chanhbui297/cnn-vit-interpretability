#!/usr/bin/env python3
"""Run TV4 cue-conflict Texture Robustness on the approved 1,200 images."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models import load_model
from experiments.texture.cue_conflict_robustness import (
    CONDITIONS, MODELS, SEED, controlled_rows_for_texture, evaluate_condition,
    id_hash, load_csv, load_valid_records, summarize_condition, transition_rows,
    valid_prediction_rows, write_csv,
)

RUN_ID = "tv4_cue_conflict_texture_robustness_v1"
OUT = ROOT / "results" / "tv4"
PREDICTIONS_PATH = OUT / "texture_predictions.csv"
SUMMARY_PATH = OUT / "texture_summary.csv"
TRANSITIONS_PATH = OUT / "texture_transitions.csv"
METADATA_PATH = OUT / "texture_metadata.json"
PARTIAL_PATH = OUT / "texture_predictions_partial.csv"
CONTROLLED_PATH = OUT / "controlled_perturbation_predictions.csv"


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def mapping_hash():
    return hashlib.sha256((ROOT / "experiments/shape_texture_bias/official_mapping.py").read_bytes()).hexdigest()


def normalized_rows(rows):
    return sorted(rows, key=lambda r: (r["model"], r["condition"], r["image_id"]))


def full_cache_valid(records):
    if not (PREDICTIONS_PATH.is_file() and METADATA_PATH.is_file()):
        return False
    try:
        rows = load_csv(PREDICTIONS_PATH)
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if meta.get("run_id") != RUN_ID or meta.get("dataset_id") != "rgeirhos/texture-shape-cue-conflict": return False
        if meta.get("valid_image_ids_sha256") != id_hash(records): return False
        if meta.get("mapping_source_sha256") != mapping_hash() or meta.get("conditions") != CONDITIONS: return False
        for model in MODELS:
            for condition in CONDITIONS:
                subset = [r for r in rows if r.get("model") == model and r.get("condition") == condition]
                if not valid_prediction_rows(subset, records, model, condition): return False
        return len(rows) == len(MODELS) * len(CONDITIONS) * len(records)
    except Exception:
        return False


def _load_partial(records):
    if not PARTIAL_PATH.is_file():
        return []
    try:
        rows = load_csv(PARTIAL_PATH)
        good = []
        for model in MODELS:
            for condition in CONDITIONS:
                subset = [r for r in rows if r.get("model") == model and r.get("condition") == condition]
                if valid_prediction_rows(subset, records, model, condition): good.extend(subset)
        return good
    except Exception:
        return []


def _save_partial(rows):
    if rows: write_csv(PARTIAL_PATH, normalized_rows(rows))


def build_summaries(rows):
    summary, transitions = [], []
    for model in MODELS:
        clean = [r for r in rows if r["model"] == model and r["condition"] == "clean"]
        for condition, params in CONDITIONS.items():
            changed = [r for r in rows if r["model"] == model and r["condition"] == condition]
            metrics = summarize_condition(clean, changed)
            summary.append({"model": model, "condition": condition, **params, **metrics})
            for transition in transition_rows(clean, changed):
                transitions.append({"model": model, "condition": condition, **transition})
    return summary, transitions


def make_figures(summary_rows, figure_dir=OUT / "figures"):
    """Save the five review figures; notebook separately displays them inline."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
    colors = {"resnet50": "#377eb8", "vit_b_16": "#e41a1c"}
    families = [("gaussian_blur", "Gaussian Blur"), ("bilateral_filter", "Bilateral Filtering")]
    rows = list(summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (family, label) in zip(axes, families):
        for model in MODELS:
            subset = sorted([r for r in rows if r["family"] == family and r["model"] == model], key=lambda r: r["severity"])
            ax.plot([r["severity"] for r in subset], [r["prediction_consistency_percent"] for r in subset], marker="o", lw=2, color=colors[model], label=names[model])
        ax.set(title=label, xlabel="Severity", xticks=[1, 2, 3]); ax.grid(alpha=.25)
    axes[0].set_ylabel("Prediction consistency (%)"); axes[1].legend(); fig.suptitle("Prediction consistency under texture-detail degradation")
    fig.tight_layout(); fig.savefig(figure_dir / "texture_consistency_by_severity.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    metrics = [("shape_decision_rate_percent", "Shape"), ("texture_decision_rate_percent", "Texture"), ("neither_rate_percent", "Neither"), ("cue_decision_coverage_percent", "Cue-decision coverage")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for model in MODELS:
            for family, label in families:
                subset = sorted([r for r in rows if r["model"] == model and r["family"] == family], key=lambda r: r["severity"])
                ax.plot([r["severity"] for r in subset], [r[metric] for r in subset], marker="o", color=colors[model], ls="-" if family == "gaussian_blur" else "--", label=f"{names[model]} — {label}" if metric == metrics[0][0] else None)
        ax.set(title=title, xlabel="Severity", ylabel="Rate (%)", xticks=[1,2,3]); ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8); fig.suptitle("Mapped cue-decision distributions")
    fig.tight_layout(); fig.savefig(figure_dir / "texture_decision_distributions.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ["shape_retention_percent", "texture_retention_percent"], ["Shape-decision retention", "Texture-decision retention"]):
        for model in MODELS:
            for family, label in families:
                subset = sorted([r for r in rows if r["model"] == model and r["family"] == family], key=lambda r: r["severity"])
                ax.plot([r["severity"] for r in subset], [r[metric] for r in subset], marker="o", color=colors[model], ls="-" if family == "gaussian_blur" else "--", label=f"{names[model]} — {label}")
        ax.set(title=title, xlabel="Severity", xticks=[1,2,3]); ax.grid(alpha=.25)
    axes[0].set_ylabel("Retention within clean decision group (%)"); axes[1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(figure_dir / "texture_decision_retention.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, metric, title in zip(axes, ["shape_bias_percent", "cue_decision_coverage_percent"], ["Shape Bias (conditional)", "Cue-decision coverage"]):
        for model in MODELS:
            for family, label in families:
                subset = sorted([r for r in rows if r["model"] == model and r["family"] == family], key=lambda r: r["severity"])
                ax.plot([r["severity"] for r in subset], [r[metric] for r in subset], marker="o", color=colors[model], ls="-" if family == "gaussian_blur" else "--", label=f"{names[model]} — {label}")
        ax.set(title=title, xlabel="Severity", ylabel="Percent (%)", xticks=[1,2,3]); ax.grid(alpha=.25)
    axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(figure_dir / "texture_shape_bias_coverage.png", dpi=300); plt.close(fig)


def run_experiment(batch_size=64, force_recompute=False, allow_download=False):
    """Run/reuse all 16,800 records and return (records, rows, summary, metadata)."""
    seed_everything()
    records = load_valid_records()
    OUT.mkdir(parents=True, exist_ok=True)
    if not force_recompute and full_cache_valid(records):
        rows = load_csv(PREDICTIONS_PATH); summary = load_csv(SUMMARY_PATH)
        return records, rows, summary, json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    rows = [] if force_recompute else _load_partial(records)
    if not rows:
        rows = controlled_rows_for_texture(records, CONTROLLED_PATH)
        _save_partial(rows)
    metadata_models = {}
    for model_name in MODELS:
        missing = [condition for condition in CONDITIONS if not valid_prediction_rows([r for r in rows if r["model"] == model_name and r["condition"] == condition], records, model_name, condition)]
        if not missing:
            continue
        model, transform, model_meta = load_model(model_name, allow_download=allow_download)
        metadata_models[model_name] = model_meta
        for condition in missing:
            new_rows = evaluate_condition(model, transform, model_name, records, condition, batch_size=batch_size)
            rows = [r for r in rows if not (r["model"] == model_name and r["condition"] == condition)] + new_rows
            _save_partial(rows)
        device = next(model.parameters()).device
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    rows = normalized_rows(rows)
    for model in MODELS:
        for condition in CONDITIONS:
            if not valid_prediction_rows([r for r in rows if r["model"] == model and r["condition"] == condition], records, model, condition):
                raise RuntimeError(f"Invalid final cache: {model}/{condition}")
    summary, transitions = build_summaries(rows)
    write_csv(PREDICTIONS_PATH, rows); write_csv(SUMMARY_PATH, summary); write_csv(TRANSITIONS_PATH, transitions)
    make_figures(summary)
    metadata = {
        "run_id": RUN_ID, "dataset_id": "rgeirhos/texture-shape-cue-conflict", "dataset_split": "test",
        "num_source_images": 1280, "num_valid_cue_conflict_images": len(records), "excluded_same_cue": 80,
        "valid_image_ids_sha256": id_hash(records), "mapping_source": "experiments/shape_texture_bias/official_mapping.py",
        "mapping_source_sha256": mapping_hash(), "conditions": CONDITIONS, "models": metadata_models,
        "batch_size": batch_size, "seed": SEED, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "reused_shape_controlled_conditions": ["clean", "gaussian_s1", "gaussian_s2"],
        "controlled_source": str(CONTROLLED_PATH.relative_to(ROOT)), "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return records, rows, summary, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    records, rows, summary, _ = run_experiment(args.batch_size, args.force_recompute, args.allow_download)
    print(json.dumps({"records": len(records), "prediction_rows": len(rows), "summary": summary}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
