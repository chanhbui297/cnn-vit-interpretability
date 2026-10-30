"""Reproduce the one shared 10,000-image analysis manifest from local data.

Run from the repository root:
    python src/create_analysis_manifest.py
    python src/create_analysis_manifest.py --check
    python src/create_analysis_manifest.py --check-committed

"""

import argparse
from collections import Counter
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import random
import subprocess

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["dataset_index", "image_id", "label"]


def compare_manifest(expected_bytes, actual_bytes, description):
    """Compare CSV contents and explain differences without changing either file."""
    actual_bytes = actual_bytes.replace(b"\r\n", b"\n")
    if actual_bytes == expected_bytes:
        print(description + ": identical to the regenerated manifest.")
        return

    expected_rows = list(csv.DictReader(StringIO(expected_bytes.decode("utf-8"))))
    reader = csv.DictReader(StringIO(actual_bytes.decode("utf-8-sig")))
    if reader.fieldnames != COLUMNS:
        raise SystemExit(description + ": column names or column order differ. No files changed.")
    actual_rows = list(reader)
    if any(not row["image_id"] for row in actual_rows):
        raise SystemExit(description + ": missing image IDs. No files changed.")
    expected_by_id = {row["image_id"]: row for row in expected_rows}
    actual_by_id = {row["image_id"]: row for row in actual_rows}
    expected_ids = [row["image_id"] for row in expected_rows]
    actual_ids = [row["image_id"] for row in actual_rows]
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    changed = []
    for image_id in sorted(set(expected_ids) & set(actual_ids)):
        if expected_by_id[image_id] != actual_by_id[image_id]:
            changed.append(image_id)
    duplicates = len(actual_ids) - len(set(actual_ids))
    print(description + ": differences found")
    print(f"Rows: expected {len(expected_rows)}, found {len(actual_rows)}")
    print(f"Missing IDs: {len(missing)}; extra IDs: {len(extra)}; duplicate rows: {duplicates}")
    print(f"Changed labels or dataset indices: {len(changed)}")
    print("Image-ID ordering differs:", actual_ids != expected_ids)
    print("First missing / extra / changed IDs:", missing[:5], extra[:5], changed[:5])
    raise SystemExit("CSV contents or formatting differ. No existing manifest was overwritten.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Regenerate in memory and verify the local manifest; write nothing")
    parser.add_argument("--check-committed", action="store_true", help="Also compare with HEAD using read-only git show; write nothing")
    args = parser.parse_args()
    config = json.loads((REPO_ROOT / "configs/analysis_set.json").read_text(encoding="utf-8"))
    source_path = REPO_ROOT / config["source_manifest"]
    manifest_path = REPO_ROOT / config["manifest"]

    # Chuẩn hóa xuống dòng trong bộ nhớ để checksum giống nhau trên Windows/Linux.
    source_bytes = source_path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source_bytes).hexdigest() != config["source_manifest_sha256"]:
        raise ValueError("The full baseline manifest differs from the approved source.")
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Expected dataset_index,image_id,label columns.")
        image_records = list(reader)
    if len(image_records) != config["source_num_images"]:
        raise ValueError("Expected all 50,000 source validation images.")
    for index, record in enumerate(image_records):
        record["dataset_index"] = int(record["dataset_index"])
        record["label"] = int(record["label"])
        if record["dataset_index"] != index or not 0 <= record["label"] < config["num_classes"]:
            raise ValueError("Unexpected dataset order or class index in the source manifest.")
    if len({record["image_id"] for record in image_records}) != len(image_records):
        raise ValueError("The source contains duplicate image IDs.")

    # Checksum xác nhận đúng các file thuộc revision ImageNet đã ghim trong config.
    dataset_dir = REPO_ROOT / config["dataset_dir"]
    validation_files = sorted(dataset_dir.glob("validation-*.parquet"))
    expected_files = config["dataset_files_sha256"]
    if len(expected_files) != 14 or {path.name for path in validation_files} != set(expected_files):
        raise ValueError("Expected the 14 original local validation shards.")
    dataset_index = 0
    for path in validation_files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != expected_files[path.name]:
            raise ValueError(f"Shard does not match the pinned revision: {path.name}")
        # Chỉ đọc tên ảnh và nhãn, không giải mã hay sao chép ảnh.
        table = pq.read_table(path, columns=["image.path", "label"])
        for image_id, label in zip(table.column(0).to_pylist(), table.column(1).to_pylist()):
            if dataset_index >= len(image_records):
                raise ValueError("The dataset has more rows than the source manifest.")
            record = image_records[dataset_index]
            if record["image_id"] != image_id or record["label"] != label:
                raise ValueError(f"Dataset and source manifest differ at row {dataset_index}.")
            dataset_index += 1
    if dataset_index != len(image_records):
        raise ValueError("The dataset is missing source manifest rows.")
    print("Verified local dataset revision:", config["dataset_revision"])

    # Sắp xếp bằng ID duy nhất trước khi chia lớp để không phụ thuộc thứ tự đọc file.
    sorted_records = sorted(image_records, key=lambda record: record["image_id"])
    class_images = {}
    for label in range(config["num_classes"]):
        class_images[label] = []
    for record in sorted_records:
        class_images[record["label"]].append(record)
    if any(len(images) != 50 for images in class_images.values()):
        raise ValueError("Expected 50 validation images per class.")

    # Lấy ngẫu nhiên đúng 10 ảnh mỗi lớp, duyệt nhãn 0..999 với seed cố định.
    generator = random.Random(config["seed"])
    selected_images = []
    for label in range(config["num_classes"]):
        selected_images.extend(generator.sample(class_images[label], config["images_per_class"]))
    selected_images.sort(key=lambda record: record["image_id"])
    class_counts = Counter(record["label"] for record in selected_images)
    if len(selected_images) != config["num_images"] or len({row["image_id"] for row in selected_images}) != config["num_images"]:
        raise ValueError("Expected exactly 10,000 unique analysis images.")
    if len(class_counts) != config["num_classes"] or set(class_counts.values()) != {config["images_per_class"]}:
        raise ValueError("Expected exactly 10 images per class across all 1,000 classes.")
    # Mỗi hàng được chọn từ nguồn đã đối chiếu Parquet, nên giữ nguyên ID và nhãn thật.
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected_images)
    manifest_bytes = buffer.getvalue().encode("utf-8")
    manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_checksum != config["manifest_sha256"]:
        print("Regenerated SHA-256:", manifest_checksum)
        raise ValueError("Sampling differs from the approved checksum; check config and Python version.")

    if manifest_path.exists():
        compare_manifest(manifest_bytes, manifest_path.read_bytes(), "Local manifest")
    elif args.check or args.check_committed:
        raise FileNotFoundError("No local manifest exists. Run without check flags to create it.")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_bytes)
        print("Created:", manifest_path)

    if args.check_committed:
        # Chỉ đọc manifest từ HEAD; không stage, commit hay sửa lịch sử Git.
        result = subprocess.run(
            ["git", "show", "HEAD:" + config["manifest"]],
            cwd=REPO_ROOT, capture_output=True,
        )
        if result.returncode != 0:
            raise SystemExit("No committed manifest found at HEAD. Commit 2 must exist before this check.")
        compare_manifest(manifest_bytes, result.stdout, "Committed manifest (HEAD)")
    print("Verified: 10,000 unique existing images; 1,000 classes; exactly 10/class; seed 42.")
    print("Manifest SHA-256 (LF):", manifest_checksum)


if __name__ == "__main__":
    main()
