"""Read the shared ID manifest and stream selected RGB images from Parquet."""

from collections import Counter
import csv
from io import BytesIO
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq

# Tính đường dẫn từ vị trí file code, không gắn với thư mục cá nhân của thành viên.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/analysis_10000_seed42.csv"
DEFAULT_DATA_DIR = REPO_ROOT / "data/imagenet/data"


def load_analysis_manifest(path=DEFAULT_MANIFEST, *, limit=None):
    """Validate the full shared manifest, then optionally take its first N IDs.

    A small limit is for debugging, not a representative evaluation cohort.
    Paths supplied by the caller are resolved relative to their working directory.
    """
    # Manifest chỉ chứa danh sách ID/nhãn/chỉ số gốc, không chứa dữ liệu ảnh.
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["dataset_index", "image_id", "label"]:
            raise ValueError("Expected dataset_index,image_id,label columns")
        records = [dict(dataset_index=int(r["dataset_index"]),
                        image_id=r["image_id"], label=int(r["label"])) for r in reader]
    # Kiểm tra toàn manifest trước khi cắt subset: đúng 10.000 ảnh, 10 ảnh mỗi lớp.
    counts = Counter(r["label"] for r in records)
    if (len(records) != 10000 or set(counts) != set(range(1000))
            or set(counts.values()) != {10}):
        raise ValueError("Expected 10,000 images, labels 0..999, exactly 10/class")
    # ID và chỉ số gốc phải duy nhất để tránh đếm lặp hoặc join nhầm kết quả.
    if (len({r["image_id"] for r in records}) != len(records)
            or any(not r["image_id"] for r in records)
            or len({r["dataset_index"] for r in records}) != len(records)
            or any(not 0 <= r["dataset_index"] < 50000 for r in records)):
        raise ValueError("Manifest IDs/indices must be unique and valid")
    # limit lấy N hàng đầu để thử nhanh, không phải mẫu đại diện để báo cáo nghiên cứu.
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= len(records):
            raise ValueError("limit must be an integer between 1 and 10,000")
        records = records[:limit]
    return records


def iter_analysis_images(records=None, *, data_dir=DEFAULT_DATA_DIR):
    """Yield (PIL RGB image, label, image_id) in Parquet order, joining by ID.

    Only selected images are decoded. Read lightweight IDs/labels first to skip
    shards without selected IDs. Consume fully to check that all IDs were found.
    Dataset indices and manifest row order are never used to locate image bytes.
    """
    if records is None:
        records = load_analysis_manifest()
    records = list(records)
    # Tra ảnh bằng image_id; không dùng vị trí hàng hoặc dataset_index để ghép ảnh.
    selected = {r["image_id"]: int(r["label"]) for r in records}
    if (not selected or len(selected) != len(records)
            or any(not key or not 0 <= label < 1000 for key, label in selected.items())):
        raise ValueError("Provide nonempty records with unique IDs and labels 0..999")
    # Yêu cầu đủ 14 shard gốc; data_dir có thể được truyền từ máy của thành viên khác.
    files = sorted(Path(data_dir).glob("validation-*.parquet"))
    expected = [f"validation-{i:05d}-of-00014.parquet" for i in range(14)]
    if [p.name for p in files] != expected:
        raise FileNotFoundError(f"Expected 14 validation shards under {data_dir}")
    # Ghi nhận ID đã đọc để phát hiện ảnh trùng và ảnh còn thiếu khi duyệt xong.
    seen = set()
    for path in files:
        # Đọc cột ID/nhãn trước, bỏ qua shard không chứa ảnh được chọn.
        table = pq.read_table(path, columns=["image.path", "label"])
        shard_ids = table.column(0).to_pylist()
        shard_labels = table.column(1).to_pylist()
        wanted = set(shard_ids) & selected.keys()
        if not wanted:
            continue
        # Nhãn trong manifest phải khớp nhãn thật trong Parquet, không tự remap.
        for image_id, label in zip(shard_ids, shard_labels):
            if image_id in selected and label != selected[image_id]:
                raise ValueError(f"Manifest/dataset label mismatch: {image_id}")
        # Đọc byte ảnh theo batch để giới hạn RAM, nhưng chỉ giải mã ID được chọn.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["image", "label"]):
            for row in batch.to_pylist():
                image_id = row["image"]["path"]
                if image_id not in selected:
                    continue
                if image_id in seen:
                    raise ValueError(f"Duplicate dataset ID: {image_id}")
                if row["label"] != selected[image_id]:
                    raise ValueError(f"Manifest/dataset label mismatch: {image_id}")
                seen.add(image_id)
                # Giải mã trong bộ nhớ và chuyển RGB; không extract hoặc sửa dataset.
                with Image.open(BytesIO(row["image"]["bytes"])) as image:
                    rgb = image.convert("RGB")
                # Trả ảnh RGB chưa preprocessing để từng thí nghiệm có thể xử lý tiếp.
                yield rgb, row["label"], image_id
    # Kiểm tra này chỉ chạy khi người gọi duyệt hết iterator, không dừng bằng break.
    missing = selected.keys() - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} selected IDs: {sorted(missing)[:5]}")
