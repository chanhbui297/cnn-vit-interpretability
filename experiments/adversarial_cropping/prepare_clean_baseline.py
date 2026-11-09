"""Prepare clean cohort predictions for the adversarial cropping experiment."""

import argparse
import csv
from datetime import datetime
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from src.data import iter_analysis_images, load_analysis_manifest
from src.evaluation import accuracy_from_predictions, evaluate, save_predictions
from src.models import load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "adversarial_cropping"
MODEL_NAMES = ("resnet50", "vit_b_16")
BASELINE_FILES = {
    "resnet50": REPO_ROOT / "results/metrics/baseline_notebook/full/resnet50_predictions.csv",
    "vit_b_16": REPO_ROOT / "results/metrics/baseline_notebook/full/vit_b_16_predictions.csv",
}
MODEL_PROTOCOL = {
    "resnet50": {
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "checkpoint": "resnet50-11ad3fa6.pth",
        "resize_size": [232],
    },
    "vit_b_16": {
        "weights": "ViT_B_16_Weights.IMAGENET1K_V1",
        "checkpoint": "vit_b_16-c867db91.pth",
        "resize_size": [256],
    },
}
COMMON_PREPROCESSING = {
    "color_mode": "RGB",
    "crop_size": [224],
    "interpolation": "bilinear",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
BASELINE_COLUMNS = [
    "image_id", "label", "pred_top1", "pred_top2", "pred_top3",
    "pred_top4", "pred_top5", "top1_correct", "top5_correct",
]


def read_baseline_rows(path):
    """Read and validate one legacy 50K baseline prediction CSV."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != BASELINE_COLUMNS:
            raise ValueError(f"Unexpected baseline schema in {path}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    if len(rows) != 50000 or len(set(ids)) != 50000 or any(not image_id for image_id in ids):
        raise ValueError(f"Expected 50,000 unique baseline IDs in {path}")
    return rows


def filter_clean_predictions(records, baseline_rows, model_name):
    """Join a legacy baseline to cohort records by image_id."""
    baseline = {row["image_id"]: row for row in baseline_rows}
    clean_rows = []
    for record in records:
        image_id = record["image_id"]
        if image_id not in baseline:
            raise ValueError(f"Missing baseline prediction for {image_id}")
        source = baseline[image_id]
        if int(source["label"]) != int(record["label"]):
            raise ValueError(f"Manifest/baseline label mismatch for {image_id}")
        clean_rows.append(_shared_prediction_row(source, model_name))
    if len({row["image_id"] for row in clean_rows}) != len(records):
        raise ValueError("Clean prediction IDs must be unique")
    return clean_rows


def _shared_prediction_row(source, model_name):
    row = {
        "image_id": source["image_id"],
        "model_name": model_name,
        "label": int(source["label"]),
        "experiment": EXPERIMENT,
        "condition": "clean",
        "severity": "",
    }
    for rank in range(1, 6):
        row[f"pred_top{rank}"] = int(source[f"pred_top{rank}"])
    label = row["label"]
    row["top1_correct"] = int(row["pred_top1"] == label)
    row["top5_correct"] = int(label in [row[f"pred_top{i}"] for i in range(1, 6)])
    return row


def infer_clean_predictions(records, model_name, batch_size, allow_download, data_dir):
    """Run clean inference through the required shared model and evaluation APIs."""
    model, transform, metadata = load_model(
        model_name, allow_download=allow_download
    )
    samples = iter_analysis_images(records, data_dir=data_dir)
    rows, metrics = evaluate(
        model,
        transform,
        samples,
        model_name=model_name,
        batch_size=batch_size,
        experiment=EXPERIMENT,
        condition="clean",
    )
    return rows, metrics, metadata


def csv_content(rows, columns):
    """Serialize rows with stable UTF-8/LF-compatible CSV content."""
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_text_exclusive(path, content):
    """Write text without replacing an existing artifact."""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)


def normalized_file_sha256(path):
    """Hash text after normalizing line endings to LF."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return sha256(text.encode("utf-8")).hexdigest()


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def protocol_metadata(model_name):
    metadata = dict(MODEL_PROTOCOL[model_name])
    metadata["preprocessing"] = {
        **COMMON_PREPROCESSING,
        "resize_size": metadata.pop("resize_size"),
    }
    return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", choices=("baseline", "inference"), default="baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/imagenet/data")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output_dir = (
        REPO_ROOT / "results/metrics/experiments/adversarial_cropping"
        / args.run_id / "intermediate"
    )
    if output_dir.exists():
        raise FileExistsError(f"Run output already exists: {output_dir}")

    records = load_analysis_manifest()
    model_rows = {}
    model_metrics = {}
    model_metadata = {}
    for model_name in MODEL_NAMES:
        if args.source == "baseline":
            rows = filter_clean_predictions(
                records, read_baseline_rows(BASELINE_FILES[model_name]), model_name
            )
            metrics = accuracy_from_predictions(rows)
            metadata = protocol_metadata(model_name)
            metadata["prediction_source"] = str(BASELINE_FILES[model_name].relative_to(REPO_ROOT))
            metadata["inference_device"] = "source baseline CSV; no inference in this run"
        else:
            rows, metrics, metadata = infer_clean_predictions(
                records, model_name, args.batch_size, args.allow_download, args.data_dir
            )
            metadata["prediction_source"] = "fresh inference"
        model_rows[model_name] = rows
        model_metrics[model_name] = metrics
        model_metadata[model_name] = metadata

    expected_ids = {record["image_id"] for record in records}
    for model_name, rows in model_rows.items():
        if {row["image_id"] for row in rows} != expected_ids:
            raise ValueError(f"Cohort IDs differ for {model_name}")

    cohort_columns = ["dataset_index", "image_id", "label"]
    cohort_text = csv_content(records, cohort_columns)
    cohort_hash = sha256(cohort_text.encode("utf-8")).hexdigest()
    metrics_rows = [
        {
            "model_name": model_name,
            "cohort": "analysis_10000_seed42",
            "num_images": model_metrics[model_name]["num_images"],
            "top1_accuracy_percent": model_metrics[model_name]["top1_accuracy_percent"],
            "top5_accuracy_percent": model_metrics[model_name]["top5_accuracy_percent"],
            "prediction_source": args.source,
        }
        for model_name in MODEL_NAMES
    ]
    metric_columns = list(metrics_rows[0])
    manifest_path = REPO_ROOT / "data/manifests/analysis_10000_seed42.csv"
    metadata = {
        "experiment": EXPERIMENT,
        "phase": "clean_baseline",
        "run_id": args.run_id,
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "prediction_source": args.source,
        "score_collection": False,
        "cohort": {
            "name": "analysis_10000_seed42",
            "num_images": len(records),
            "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256_lf": normalized_file_sha256(manifest_path),
            "cohort_csv_sha256": cohort_hash,
        },
        "models": model_metadata,
        "batch_size": args.batch_size if args.source == "inference" else None,
        "git_revision": git_revision(),
        "git_tree_dirty": git_is_dirty(),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_text_exclusive(output_dir / "cohort.csv", cohort_text)
    for model_name in MODEL_NAMES:
        save_predictions(
            model_rows[model_name], output_dir / f"{model_name}_clean_predictions.csv"
        )
    write_text_exclusive(
        output_dir / "clean_metrics.csv", csv_content(metrics_rows, metric_columns)
    )
    write_text_exclusive(
        output_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n"
    )
    print(output_dir)
    for row in metrics_rows:
        print(row)


if __name__ == "__main__":
    main()
