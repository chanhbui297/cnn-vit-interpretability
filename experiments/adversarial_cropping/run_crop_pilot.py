"""Run the fixed adversarial-cropping grid on a deterministic pilot cohort."""

import argparse
import csv
from datetime import datetime
import gc
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

from experiments.adversarial_cropping.crop_utils import (
    CROP_AREAS,
    CROP_POSITIONS,
    crop_image,
)
from src.data import iter_analysis_images, load_analysis_manifest
from src.evaluation import PREDICTION_COLUMNS, evaluate, save_predictions
from src.models import load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "adversarial_cropping"
MODEL_NAMES = ("resnet50", "vit_b_16")
CROP_PARAMETER_COLUMNS = [
    "image_id", "label", "condition", "severity",
    "original_width", "original_height", "crop_width", "crop_height",
    "left", "top", "right", "bottom", "requested_area_ratio",
    "realized_area_ratio",
]


def select_pilot_records(records):
    """Select one stable hash-ranked image from each ImageNet class."""
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)
    if set(by_label) != set(range(1000)):
        raise ValueError("Pilot selection requires all 1,000 classes")
    selected = [
        min(group, key=lambda row: sha256(row["image_id"].encode("utf-8")).digest())
        for group in by_label.values()
    ]
    return sorted(selected, key=lambda row: row["image_id"])


def condition_specs():
    """Return the fixed crop grid in its documented order."""
    return [
        (area_name, position, f"{area_name}__{position}", f"{ratio:.2f}")
        for area_name, ratio in CROP_AREAS.items()
        for position in CROP_POSITIONS
    ]


def load_clean_rows(path, records, model_name):
    """Filter the Phase 2 clean predictions to exactly the pilot IDs."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError(f"Unexpected clean prediction schema: {path}")
        source = {row["image_id"]: row for row in reader}
    clean_rows = []
    for record in records:
        image_id = record["image_id"]
        if image_id not in source:
            raise ValueError(f"Missing clean prediction for {image_id}")
        row = source[image_id]
        if row["model_name"] != model_name or int(row["label"]) != record["label"]:
            raise ValueError(f"Clean prediction mismatch for {image_id}")
        clean_rows.append(row)
    return clean_rows


def crop_samples(images, area_name, position):
    """Yield one crop condition while preserving labels and image IDs."""
    for image, label, image_id in images:
        result = crop_image(image, area_name, position)
        yield result.image, label, image_id


def crop_parameter_rows(images):
    """Record shared crop geometry once, independently of model inference."""
    rows = []
    for image, label, image_id in images:
        for area_name, position, condition, severity in condition_specs():
            result = crop_image(image, area_name, position)
            left, top, right, bottom = result.box
            rows.append({
                "image_id": image_id,
                "label": label,
                "condition": condition,
                "severity": severity,
                "original_width": result.original_size[0],
                "original_height": result.original_size[1],
                "crop_width": result.crop_size[0],
                "crop_height": result.crop_size[1],
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "requested_area_ratio": severity,
                "realized_area_ratio": result.realized_area_ratio,
            })
    return rows


def write_csv_exclusive(path, rows, columns):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json_exclusive(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def read_prediction_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError(f"Unexpected prediction schema: {path}")
        return list(reader)


def read_crop_parameters(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CROP_PARAMETER_COLUMNS:
            raise ValueError(f"Unexpected crop parameter schema: {path}")
        return list(reader)


def record_chunks(records, chunk_size):
    for start in range(0, len(records), chunk_size):
        yield records[start:start + chunk_size]


def git_state():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    return revision, bool(status.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clean-run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--memory-note", default="")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/imagenet/data")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    records = select_pilot_records(load_analysis_manifest())
    if args.smoke_limit is not None:
        if not 1 <= args.smoke_limit <= len(records):
            raise ValueError("smoke-limit must be between 1 and 1,000")
        records = records[:args.smoke_limit]

    clean_dir = (
        REPO_ROOT / "results/metrics/experiments/adversarial_cropping"
        / args.clean_run_id / "intermediate"
    )
    output_dir = (
        REPO_ROOT / "results/metrics/experiments/adversarial_cropping"
        / args.run_id / "intermediate"
    )
    work_dir = output_dir.parent / "work"
    if output_dir.exists():
        raise FileExistsError(f"Run output already exists: {output_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    model_metadata = {}
    parameters = None
    parameter_checkpoint = work_dir / "crop_parameters.csv"
    if parameter_checkpoint.is_file():
        parameters = read_crop_parameters(parameter_checkpoint)
    for model_index, model_name in enumerate(MODEL_NAMES):
        prediction_checkpoint = work_dir / f"{model_name}_predictions.csv"
        metadata_checkpoint = work_dir / f"{model_name}_metadata.json"
        if prediction_checkpoint.is_file() and metadata_checkpoint.is_file():
            predictions[model_name] = read_prediction_rows(prediction_checkpoint)
            model_metadata[model_name] = json.loads(
                metadata_checkpoint.read_text(encoding="utf-8")
            )
            print(f"{model_name}: resumed completed model checkpoint", flush=True)
            continue
        model, transform, metadata = load_model(
            model_name, allow_download=args.allow_download
        )
        rows = load_clean_rows(
            clean_dir / f"{model_name}_clean_predictions.csv", records, model_name
        )
        chunks = list(record_chunks(records, args.chunk_size))
        collected_parameters = [] if parameters is None and model_index == 0 else None
        for chunk_index, chunk_records in enumerate(chunks, 1):
            images = list(iter_analysis_images(chunk_records, data_dir=args.data_dir))
            if len(images) != len(chunk_records):
                raise ValueError("Not all pilot images were decoded")
            if collected_parameters is not None:
                collected_parameters.extend(crop_parameter_rows(images))
            for area_name, position, condition, severity in condition_specs():
                changed_rows, _ = evaluate(
                    model,
                    transform,
                    crop_samples(images, area_name, position),
                    model_name=model_name,
                    batch_size=args.batch_size,
                    experiment=EXPERIMENT,
                    condition=condition,
                    severity=severity,
                )
                rows.extend(changed_rows)
            print(
                f"{model_name}: chunk {chunk_index}/{len(chunks)} complete",
                flush=True,
            )
            del images
            gc.collect()
        predictions[model_name] = rows
        model_metadata[model_name] = metadata
        save_predictions(rows, prediction_checkpoint)
        write_json_exclusive(metadata_checkpoint, metadata)
        if collected_parameters is not None:
            parameters = collected_parameters
            write_csv_exclusive(
                parameter_checkpoint, parameters, CROP_PARAMETER_COLUMNS
            )
        del model
        gc.collect()

    if parameters is None:
        raise ValueError("Crop parameters were not generated")
    expected_parameter_count = len(records) * len(condition_specs())
    if len(parameters) != expected_parameter_count:
        raise ValueError("Crop parameter count is incomplete")

    expected_ids = {record["image_id"] for record in records}
    expected_conditions = {"clean", *(spec[2] for spec in condition_specs())}
    for model_name, rows in predictions.items():
        keys = {(row["image_id"], row["condition"], str(row["severity"])) for row in rows}
        if len(keys) != len(rows) or len(rows) != len(records) * 21:
            raise ValueError(f"Duplicate or missing prediction keys for {model_name}")
        if {row["image_id"] for row in rows} != expected_ids:
            raise ValueError(f"Prediction cohort mismatch for {model_name}")
        if {row["condition"] for row in rows} != expected_conditions:
            raise ValueError(f"Condition set mismatch for {model_name}")
        labels = {row["image_id"]: int(row["label"]) for row in rows}
        if labels != {record["image_id"]: record["label"] for record in records}:
            raise ValueError(f"Prediction labels mismatch for {model_name}")

    revision, dirty = git_state()
    metadata = {
        "experiment": EXPERIMENT,
        "phase": "crop_grid_pilot",
        "run_id": args.run_id,
        "created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "cohort": {
            "selection": "one SHA-256-ranked image per class",
            "num_images": len(records),
            "image_ids_sha256": sha256(
                "\n".join(sorted(expected_ids)).encode("utf-8")
            ).hexdigest(),
            "smoke_limit": args.smoke_limit,
        },
        "clean_reference_run_id": args.clean_run_id,
        "conditions": [spec[2] for spec in condition_specs()],
        "num_conditions": len(condition_specs()),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "device": next(iter(model_metadata.values()))["device"],
        "models": model_metadata,
        "memory_issues": [args.memory_note] if args.memory_note else [],
        "git_revision": revision,
        "git_tree_dirty": dirty,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for model_name, rows in predictions.items():
        save_predictions(rows, output_dir / f"{model_name}_predictions.csv")
    write_csv_exclusive(
        output_dir / "crop_parameters.csv", parameters, CROP_PARAMETER_COLUMNS
    )
    write_json_exclusive(output_dir / "metadata.json", metadata)
    shutil.rmtree(work_dir)
    print(output_dir)
    print(f"images={len(records)} conditions={len(condition_specs())}")
    for model_name, rows in predictions.items():
        print(f"{model_name}: predictions={len(rows)}")


if __name__ == "__main__":
    main()
