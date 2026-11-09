"""Run the fixed crop grid on the complete 10,000-image analysis cohort."""

import argparse
import csv
from datetime import datetime
import gc
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from experiments.adversarial_cropping.run_crop_pilot import (
    CROP_PARAMETER_COLUMNS,
    EXPERIMENT,
    MODEL_NAMES,
    REPO_ROOT,
    condition_specs,
    crop_parameter_rows,
    crop_samples,
    git_state,
    load_clean_rows,
    read_crop_parameters,
    read_prediction_rows,
    record_chunks,
    write_csv_exclusive,
    write_json_exclusive,
)
from src.data import iter_analysis_images, load_analysis_manifest
from src.evaluation import evaluate, save_predictions
from src.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clean-run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--image-buffer-size", type=int, default=10)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/imagenet/data")
    return parser.parse_args()


def timestamp():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def write_status(path, data):
    """Atomically update the run status without creating final artifacts."""
    temporary = path.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_status(path, args, num_images):
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            raise FileExistsError(f"Run is already complete: {args.run_id}")
        return status
    status = {
        "run_id": args.run_id,
        "state": "incomplete",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "reason": "Run in progress or interrupted before final validation",
        "num_images": num_images,
        "num_conditions": len(condition_specs()),
    }
    write_status(path, status)
    return status


def validate_chunk_rows(rows, records, model_name):
    expected_ids = {record["image_id"] for record in records}
    expected_conditions = {spec[2] for spec in condition_specs()}
    keys = {(row["image_id"], row["model_name"], row["condition"], str(row["severity"])) for row in rows}
    if len(rows) != len(records) * len(expected_conditions) or len(keys) != len(rows):
        raise ValueError(f"Invalid prediction count or duplicate key for {model_name}")
    if {row["image_id"] for row in rows} != expected_ids:
        raise ValueError(f"Missing chunk image IDs for {model_name}")
    if {row["condition"] for row in rows} != expected_conditions:
        raise ValueError(f"Missing crop conditions for {model_name}")
    labels = {row["image_id"]: int(row["label"]) for row in rows}
    expected_labels = {record["image_id"]: record["label"] for record in records}
    if labels != expected_labels:
        raise ValueError(f"Chunk label mismatch for {model_name}")


def run_model_chunks(args, records, work_dir, model_name):
    chunks = list(record_chunks(records, args.chunk_size))
    prediction_dir = work_dir / model_name
    parameter_dir = work_dir / "crop_parameters"
    metadata_path = work_dir / f"{model_name}_metadata.json"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        index for index in range(len(chunks))
        if not (prediction_dir / f"chunk_{index:04d}.csv").is_file()
    ]
    if not missing:
        print(f"{model_name}: all {len(chunks)} chunk checkpoints present", flush=True)
        return

    model, transform, metadata = load_model(
        model_name, allow_download=args.allow_download
    )
    if not metadata_path.exists():
        write_json_exclusive(metadata_path, metadata)
    for index, chunk_records in enumerate(chunks):
        prediction_path = prediction_dir / f"chunk_{index:04d}.csv"
        parameter_path = parameter_dir / f"chunk_{index:04d}.csv"
        if prediction_path.is_file():
            validate_chunk_rows(
                read_prediction_rows(prediction_path), chunk_records, model_name
            )
            continue
        rows = []
        parameter_rows = [] if not parameter_path.exists() else None
        for buffered_records in record_chunks(
                chunk_records, args.image_buffer_size):
            images = list(iter_analysis_images(
                buffered_records,
                data_dir=args.data_dir,
                decode_batch_size=args.decode_batch_size,
            ))
            if len(images) != len(buffered_records):
                raise ValueError("Not all main-cohort images were decoded")
            if parameter_rows is not None:
                parameter_rows.extend(crop_parameter_rows(images))
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
            del images
            gc.collect()
        if parameter_rows is not None:
            write_csv_exclusive(
                parameter_path, parameter_rows, CROP_PARAMETER_COLUMNS
            )
        validate_chunk_rows(rows, chunk_records, model_name)
        save_predictions(rows, prediction_path)
        print(f"{model_name}: chunk {index + 1}/{len(chunks)} complete", flush=True)
        del rows
        gc.collect()
    del model
    gc.collect()


def assemble_predictions(work_dir, clean_dir, records, model_name, chunk_size):
    rows = load_clean_rows(
        clean_dir / f"{model_name}_clean_predictions.csv", records, model_name
    )
    chunks = list(record_chunks(records, chunk_size))
    for index, chunk_records in enumerate(chunks):
        path = work_dir / model_name / f"chunk_{index:04d}.csv"
        changed = read_prediction_rows(path)
        validate_chunk_rows(changed, chunk_records, model_name)
        rows.extend(changed)
    expected = len(records) * (len(condition_specs()) + 1)
    keys = {(row["image_id"], row["model_name"], row["condition"], str(row["severity"])) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(f"Invalid assembled predictions for {model_name}")
    return rows


def assemble_parameters(work_dir, records, chunk_size):
    rows = []
    chunks = list(record_chunks(records, chunk_size))
    for index in range(len(chunks)):
        rows.extend(read_crop_parameters(
            work_dir / "crop_parameters" / f"chunk_{index:04d}.csv"
        ))
    keys = {(row["image_id"], row["condition"]) for row in rows}
    expected = len(records) * len(condition_specs())
    if len(rows) != expected or len(keys) != expected:
        raise ValueError("Invalid assembled crop parameters")
    return rows


def finalize(args, records, run_root, work_dir, clean_dir, status):
    staging = run_root / "final_staging"
    final_dir = run_root / "final"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model_metadata = {}
    for model_name in MODEL_NAMES:
        predictions = assemble_predictions(
            work_dir, clean_dir, records, model_name, args.chunk_size
        )
        save_predictions(predictions, staging / f"{model_name}_predictions.csv")
        model_metadata[model_name] = json.loads(
            (work_dir / f"{model_name}_metadata.json").read_text(encoding="utf-8")
        )
    parameters = assemble_parameters(work_dir, records, args.chunk_size)
    write_csv_exclusive(
        staging / "crop_parameters.csv", parameters, CROP_PARAMETER_COLUMNS
    )
    revision, dirty = git_state()
    image_ids = sorted(record["image_id"] for record in records)
    metadata = {
        "experiment": EXPERIMENT,
        "phase": "main_crop_grid",
        "run_id": args.run_id,
        "state": "complete",
        "created_at": timestamp(),
        "cohort": {
            "name": "analysis_10000_seed42",
            "num_images": len(records),
            "image_ids_sha256": sha256("\n".join(image_ids).encode("utf-8")).hexdigest(),
        },
        "clean_reference_run_id": args.clean_run_id,
        "conditions": [spec[2] for spec in condition_specs()],
        "num_conditions": len(condition_specs()),
        "final_attempt_batch_size": args.batch_size,
        "batch_sizes_used": sorted({
            attempt["batch_size"] for attempt in status.get("execution_attempts", [])
        }),
        "chunk_size": args.chunk_size,
        "decode_batch_size": args.decode_batch_size,
        "image_buffer_size": args.image_buffer_size,
        "execution_attempts": status.get("execution_attempts", []),
        "device": next(iter(model_metadata.values()))["device"],
        "models": model_metadata,
        "raw_images_saved": False,
        "git_revision": revision,
        "git_tree_dirty": dirty,
    }
    write_json_exclusive(staging / "metadata.json", metadata)
    staging.rename(final_dir)
    shutil.rmtree(work_dir)
    return final_dir


def main():
    args = parse_args()
    if (args.batch_size < 1 or args.chunk_size < 1
            or args.decode_batch_size < 1 or args.image_buffer_size < 1):
        raise ValueError(
            "batch-size, chunk-size, decode-batch-size and image-buffer-size "
            "must be positive"
        )
    records = load_analysis_manifest()
    run_root = (
        REPO_ROOT / "results/metrics/experiments/adversarial_cropping" / args.run_id
    )
    final_dir = run_root / "final"
    work_dir = run_root / "work"
    status_path = run_root / "run_status.json"
    clean_dir = (
        REPO_ROOT / "results/metrics/experiments/adversarial_cropping"
        / args.clean_run_id / "intermediate"
    )
    if final_dir.exists():
        raise FileExistsError(f"Final run already exists: {final_dir}")
    status = initialize_status(status_path, args, len(records))
    status.setdefault("execution_attempts", []).append({
        "started_at": timestamp(),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "decode_batch_size": args.decode_batch_size,
        "image_buffer_size": args.image_buffer_size,
    })
    status.update({
        "state": "incomplete",
        "updated_at": timestamp(),
        "reason": "Run resumed and in progress",
    })
    write_status(status_path, status)
    try:
        for model_name in MODEL_NAMES:
            run_model_chunks(args, records, work_dir, model_name)
        result = finalize(
            args, records, run_root, work_dir, clean_dir, status
        )
    except BaseException as error:
        status.update({
            "state": "incomplete",
            "updated_at": timestamp(),
            "reason": f"{type(error).__name__}: {error}",
        })
        write_status(status_path, status)
        raise
    status.update({
        "state": "complete",
        "updated_at": timestamp(),
        "reason": "All artifacts validated and final directory published",
    })
    write_status(status_path, status)
    print(result)


if __name__ == "__main__":
    main()
