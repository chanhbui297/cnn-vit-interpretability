"""Data, resumable runs, pilot and result tables for notebook 02.

This module and attention_gradcam.py are the only implementation files. Nothing
installs packages, downloads images, extracts the dataset or runs at import time.
"""

import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import time
import uuid
import warnings

import numpy as np
import pandas as pd

from .attention_gradcam import (
    STAGES, agreement, attention, crop_intersection, entropy, gradcam,
    map_status, preprocess, sample_grid,
)

MODELS = ("resnet50", "vit_b_16")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_selection(root, verify_local=True):
    """Validate all selected IDs/labels/indices and baseline joins, without torch."""
    import pyarrow.parquet as pq
    root = Path(root)
    config = json.loads((root / "configs/analysis_set.json").read_text(encoding="utf-8"))
    for field in ("manifest", "source_manifest"):
        if checksum(root / config[field]) != config[field + "_sha256"]:
            raise ValueError(f"Checksum differs: {field}")
    selected = pd.read_csv(root / config["manifest"])
    source = pd.read_csv(root / config["source_manifest"])
    if (len(selected) != 10000 or selected.image_id.duplicated().any()
            or selected.dataset_index.duplicated().any() or set(selected.label) != set(range(1000))
            or not selected.groupby("label").size().eq(10).all()):
        raise ValueError("Expected 10,000 unique images, ten per class")
    if len(source) != 50000 or source.image_id.duplicated().any():
        raise ValueError("Invalid full baseline manifest")
    match = selected.merge(source, on="image_id", validate="one_to_one", suffixes=("", "_source"))
    if (len(match) != len(selected) or not match.label.eq(match.label_source).all()
            or not match.dataset_index.eq(match.dataset_index_source).all()):
        raise ValueError("Analysis/full manifest mapping differs")
    for name in MODELS:
        predictions = pd.read_csv(root / f"results/metrics/baseline_notebook/full/{name}_predictions.csv")
        joined = source.merge(predictions, on="image_id", validate="one_to_one", suffixes=("", "_pred"))
        if (len(predictions) != 50000 or len(joined) != 50000
                or not joined.label.eq(joined.label_pred).all()
                or not joined.pred_top1.between(0, 999).all()
                or not joined.top1_correct.eq(joined.pred_top1.eq(joined.label).astype(int)).all()):
            raise ValueError(f"Baseline label/prediction mismatch: {name}")
        selected = selected.merge(predictions[["image_id", "pred_top1"]].rename(
            columns={"pred_top1": name + "_prediction"}), on="image_id", validate="one_to_one")
    a = selected.resnet50_prediction.eq(selected.label)
    b = selected.vit_b_16_prediction.eq(selected.label)
    selected["group"] = np.select([a & b, a, b], ["both_correct", "resnet_only", "vit_only"], default="both_wrong")
    if verify_local:
        files = sorted((root / "data/imagenet/data").glob("validation-*.parquet"))
        if [p.name for p in files] != [f"validation-{i:05d}-of-00014.parquet" for i in range(14)]:
            raise ValueError("Expected all 14 local validation shards")
        rows, class_names = [], None
        for path in files:
            parquet = pq.ParquetFile(path)
            metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
            names = metadata["info"]["features"]["label"]["names"]
            if len(names) != 1000 or (class_names is not None and names != class_names):
                raise ValueError("Inconsistent local class metadata")
            class_names = names
            table = pq.read_table(path, columns=["image.path", "label"])
            rows.extend(zip(table.column(0).to_pylist(), table.column(1).to_pylist()))
        actual = pd.DataFrame(rows, columns=["image_id", "label"])
        if len(actual) != 50000 or actual.image_id.duplicated().any():
            raise ValueError("Expected exactly 50,000 unique local validation records")
        actual["dataset_index"] = np.arange(len(actual))
        actual = actual.merge(source, on="image_id", validate="one_to_one", suffixes=("", "_source"))
        if (len(actual) != 50000 or not actual.label.eq(actual.label_source).all()
                or not actual.dataset_index.eq(actual.dataset_index_source).all()):
            raise ValueError("Local data differs from baseline ID/index/label mapping")
    return selected.sort_values("image_id").reset_index(drop=True)


def subsets(selected, seed=42):
    """One image/class, then >=100 pilot images balanced across available groups."""
    if selected.label.nunique() != 1000 or selected.image_id.duplicated().any():
        raise ValueError("Subset selection requires all 1,000 classes and unique IDs")
    rng = random.Random(seed)
    indices = [rng.choice(g.sort_values("image_id").index.tolist())
               for _, g in selected.groupby("label", sort=True)]
    subset = selected.loc[indices].sort_values("image_id").reset_index(drop=True)
    rng = random.Random(seed)
    groups = []
    for _, group in subset.groupby("group", sort=True):
        indices = group.sort_values("image_id").index.tolist()
        rng.shuffle(indices)
        groups.append(indices)
    indices = []
    while len(indices) < 100:
        for group in groups:
            if group and len(indices) < 100:
                indices.append(group.pop())
    return subset, subset.loc[indices].sort_values("image_id").reset_index(drop=True)


def images(root, selected):
    """Bounded-memory streaming; no image files are written to disk."""
    import pyarrow.parquet as pq
    from PIL import Image
    lookup = {int(r["dataset_index"]): r for r in selected.to_dict("records")}
    offset, seen = 0, set()
    for path in sorted((Path(root) / "data/imagenet/data").glob("validation-*.parquet")):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if any(offset <= index < offset + count for index in lookup):
                local_offset = offset
                for batch in parquet.iter_batches(batch_size=64, row_groups=[group], columns=["image", "label"]):
                    for i, row in enumerate(batch.to_pylist()):
                        index = local_offset + i
                        if index not in lookup:
                            continue
                        record = lookup[index]
                        if row["image"]["path"] != record["image_id"] or row["label"] != record["label"]:
                            raise ValueError(f"Local record mismatch: {record['image_id']}")
                        seen.add(index)
                        try:
                            with Image.open(BytesIO(row["image"]["bytes"])) as image:
                                decoded = image.convert("RGB")
                        except (OSError, ValueError) as error:
                            yield record, None, str(error)
                        else:
                            yield record, decoded, None
                    local_offset += batch.num_rows
            offset += count
    if seen != set(lookup):
        raise ValueError(f"Missing {len(set(lookup) - seen)} selected images")


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class RunStore:
    """Small chunk checkpoints; JSON is the commit marker for its map archive.

    Single writer per run directory. Incomplete archives from interruption are
    ignored. IDs in committed chunks are never rerun or appended twice.
    """
    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        path = self.directory / "config.json"
        if config is None:
            self.config = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
                raise ValueError(f"Checkpoint configuration mismatch: {directory}")
            if not path.exists():
                _json(path, config)
            self.config = config
        self.done = set()
        for row, _ in self.items():
            if row["image_id"] in self.done:
                raise ValueError("Duplicate committed image ID")
            self.done.add(row["image_id"])
        if not self.done.issubset(set(self.config["image_ids"])):
            raise ValueError("Checkpoint contains unexpected IDs")

    def append(self, records):
        ids = [row["image_id"] for row, _ in records]
        if len(set(ids)) != len(ids) or set(ids) & self.done:
            raise ValueError("Refusing duplicate checkpoint IDs")
        if not set(ids).issubset(self.config["image_ids"]):
            raise ValueError("Unexpected checkpoint IDs")
        if not records:
            return
        stem = self.directory / ("chunk-" + uuid.uuid4().hex)
        arrays, rows = {}, []
        for index, (record, maps) in enumerate(records):
            row = {**record, "map_keys": {}}
            for name, cam in maps.items():
                key = f"r{index}_{name}"
                arrays[key] = cam
                row["map_keys"][name] = key
            rows.append(row)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        _json(stem.with_suffix(".json"), rows)
        self.done.update(ids)

    def items(self):
        for path in sorted(self.directory.glob("chunk-*.json")):
            rows = json.loads(path.read_text(encoding="utf-8"))
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                for row in rows:
                    yield row, {key: archive[value].copy() for key, value in row["map_keys"].items()}


def _spec(name):
    from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
    if ResNet50_Weights.IMAGENET1K_V2.meta["categories"] != ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]:
        raise ValueError("Model category mappings differ")
    return {"resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
            "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1)}[name]


def _model(name, device):
    import torch
    factory, weights = _spec(name)
    # Do not implicitly download hundreds of MB if a teammate lacks cached weights.
    cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not cached.is_file():
        raise FileNotFoundError(f"Pretrained weights must already be cached: {cached}")
    return factory(weights=weights).to(device).eval().float().requires_grad_(False), weights


def _identity(root, selected, name, experiment, batch_size, device):
    import torch
    import torchvision
    import pyarrow
    from PIL import __version__ as pillow_version
    root = Path(root)
    code = "".join(checksum(Path(__file__).with_name(p)) for p in ("attention_gradcam.py", "attention_experiments.py"))
    baseline = root / "results/metrics/baseline_notebook/full"
    return {"protocol": 1, "code": code, "experiment": experiment, "model": name,
            "weights": str(_spec(name)[1]), "target": "ground_truth_logit", "preprocessing": "official",
            "image_ids": sorted(selected.image_id.tolist()),
            "records": hashlib.sha256(selected.sort_values("image_id").to_csv(index=False).encode()).hexdigest(),
            "manifest": checksum(root / "data/manifests/analysis_10000_seed42.csv"),
            "dataset_config": checksum(root / "configs/analysis_set.json"),
            "baseline": {n: checksum(baseline / f"{n}_predictions.csv") for n in MODELS},
            "batch_size": batch_size, "dtype": "float32", "device": str(device),
            "threads": torch.get_num_threads(),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "pillow": pillow_version, "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "grids": [7, 14], "top_fraction": .2}


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract(model, x, labels, name, experiment):
    return (gradcam(model, x, labels, name, experiment == "E2") if experiment in ("E1", "E2")
            else attention(model, x, rollout=experiment == "E4"))


def _records(batch, result, name, elapsed=0., read_seconds=0.):
    records = []
    for i, (record, _, geometry) in enumerate(batch):
        maps = {key: value[i] for key, value in result["maps"].items()}
        diagnostics = {key: {k: v if k.endswith("shape") else v[i] for k, v in value.items()}
                       for key, value in result.get("diagnostics", {}).items()}
        for diagnostic in diagnostics.values():
            if not np.isfinite(diagnostic["gradient_abs_mean"]):
                diagnostic["gradient_abs_mean"] = None
        statuses = {key: map_status(cam) for key, cam in maps.items()}
        for key, diagnostic in diagnostics.items():
            if not diagnostic["finite"]:
                statuses[key] = "nonfinite_features"
            elif diagnostic["gradient_abs_mean"] == 0:
                statuses[key] = "zero_gradient"
        prediction = int(result["logits"][i].argmax())
        records.append(({**record, "geometry": geometry, "status": "ok", "prediction": prediction,
                         "baseline_match": prediction == int(record[name + "_prediction"]),
                         "target_logit": float(result["logits"][i, int(record["label"])]),
                         "map_status": statuses, "diagnostics": diagnostics,
                         "native_shapes": {key: list(cam.shape) for key, cam in maps.items()},
                         "compute_seconds": elapsed / len(batch), "read_seconds": read_seconds / len(batch)}, maps))
    return records


def run(root, selected, name, experiment, output, batch_size=1, device="cpu", chunk_size=64):
    """Run/resume one model. OOM and mapping errors stop, never shrink the dataset."""
    import torch
    if (experiment not in ("E1", "E2", "E3", "E4") or batch_size < 1 or selected.empty
            or selected.image_id.duplicated().any() or chunk_size < 1):
        raise ValueError("Invalid experiment, selection or batch size")
    if name not in MODELS or (experiment in ("E3", "E4") and name != "vit_b_16"):
        raise ValueError("Attention experiments require ViT")
    device = torch.device(device)
    store = RunStore(Path(output) / experiment / name,
                     _identity(root, selected, name, experiment, batch_size, device))
    remaining = selected[~selected.image_id.isin(store.done)]
    if remaining.empty:
        return store
    model, weights = _model(name, device)
    buffer, batch = [], []

    def flush(force=False):
        if len(buffer) >= chunk_size or (force and buffer):
            store.append(buffer)
            buffer.clear()

    def process(batch, read_seconds):
        x = torch.stack([item[1] for item in batch]).to(device)
        labels = torch.tensor([item[0]["label"] for item in batch], device=device)
        _sync(device)
        start = time.perf_counter()
        try:
            result = _extract(model, x, labels, name, experiment)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as error:
            if len(batch) > 1:
                for item in batch:
                    process([item], read_seconds / len(batch))
            else:
                buffer.append(({**batch[0][0], "status": "extraction_error", "error": str(error)}, {}))
            flush()
            return
        _sync(device)
        buffer.extend(_records(batch, result, name, time.perf_counter() - start, read_seconds))
        flush()

    try:
        start = time.perf_counter()
        for record, image, error in images(root, remaining):
            if error:
                buffer.append(({**record, "status": "decode_error", "error": error}, {}))
                flush()
                continue
            try:
                tensor, geometry = preprocess(image, weights)
            except (OSError, ValueError) as error:
                buffer.append(({**record, "status": "decode_error", "error": "preprocess: " + str(error)}, {}))
                flush()
                continue
            batch.append((record, tensor, geometry))
            if len(batch) == batch_size:
                process(batch, time.perf_counter() - start)
                batch = []
                start = time.perf_counter()
        if batch:
            process(batch, time.perf_counter() - start)
        flush(True)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if store.done != set(selected.image_id):
        raise ValueError("Incomplete experiment")
    return store


def pilot(root, selected, output, device="cpu"):
    """Benchmark >=100 fixed images at batches 1/2/4; gate long runs on parity.

    Preprocessing is cached only in RAM (about 60 MB/model for 100 images).
    Timings exclude weight loading and warm-up; read and checkpoint write time
    are measured separately. No model is kept when moving to the next model.
    """
    import torch
    if len(selected) < 100:
        raise ValueError("Pilot requires at least 100 images")
    device, output = torch.device(device), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity(root, selected, "vit_b_16", "pilot", 1, device)
    path = output / "pilot.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("Pilot changed; use a new output directory")
        if previous.get("passed"):
            return previous
    selected.to_csv(output / "pilot_ids.csv", index=False)
    report = {"identity": identity, "n_images": len(selected), "trials": [], "batch_sizes": {},
              "baseline_mismatches": [], "passed": False}
    for name in MODELS:
        print(f"Pilot {name}: {len(selected)} images", flush=True)
        model, weights = _model(name, device)
        prepared, reference = [], {}
        start = time.perf_counter()
        for record, image, error in images(root, selected):
            if error:
                raise ValueError(f"Pilot decode failed: {record['image_id']}: {error}")
            tensor, geometry = preprocess(image, weights)
            prepared.append((record, tensor, geometry))
        read_seconds = time.perf_counter() - start
        reference_store = RunStore(output / "reference/E1" / name,
                                   _identity(root, selected, name, "E1", 1, device))
        try:
            for size in (1, 2, 4):
                equal, baseline_equal, seconds, samples, reference_rows = True, True, 0., [], []
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    warm = torch.stack([item[1] for item in prepared[:size]]).to(device)
                    labels = torch.tensor([item[0]["label"] for item in prepared[:size]], device=device)
                    _extract(model, warm, labels, name, "E1")
                    del warm, labels
                    for offset in range(0, len(prepared), size):
                        batch = prepared[offset:offset + size]
                        x = torch.stack([item[1] for item in batch]).to(device)
                        labels = torch.tensor([item[0]["label"] for item in batch], device=device)
                        _sync(device)
                        start = time.perf_counter()
                        result = _extract(model, x, labels, name, "E1")
                        _sync(device)
                        elapsed = time.perf_counter() - start
                        seconds += elapsed
                        samples.extend([elapsed / len(batch)] * len(batch))
                        expected_shape = (7, 7) if name == "resnet50" else (14, 14)
                        if result["maps"]["near_head"].shape[1:] != expected_shape:
                            raise ValueError("Unexpected native near-head CAM shape")
                        for i, (record, _, _) in enumerate(batch):
                            logits = result["logits"][i]
                            cam = result["maps"]["near_head"][i]
                            baseline_equal &= int(logits.argmax()) == int(record[name + "_prediction"])
                            if size == 1:
                                reference[record["image_id"]] = (logits.clone(), cam.copy())
                                if int(logits.argmax()) != int(record[name + "_prediction"]):
                                    prediction, baseline = int(logits.argmax()), int(record[name + "_prediction"])
                                    top2 = logits.topk(2).values
                                    report["baseline_mismatches"].append({
                                        "image_id": record["image_id"], "model": name, "label": int(record["label"]),
                                        "prediction": prediction, "baseline_prediction": baseline,
                                        "top1_margin": float(top2[0] - top2[1]),
                                        "baseline_logit_gap": float(logits[prediction] - logits[baseline])})
                            else:
                                old_logits, old_cam = reference[record["image_id"]]
                                equal &= bool(torch.allclose(old_logits, logits, rtol=1e-4, atol=1e-5)
                                              and np.allclose(old_cam, cam, rtol=1e-4, atol=1e-5))
                        if size == 1:
                            reference_rows.extend(_records(batch, result, name, elapsed))
                        del x, labels, result
                    write_seconds = 0.
                    if size == 1:
                        start = time.perf_counter()
                        reference_store.append([item for item in reference_rows if item[0]["image_id"] not in reference_store.done])
                        write_seconds = time.perf_counter() - start
                    trial = {"model": name, "batch_size": size, "status": "ok", "parity": bool(equal),
                             "baseline_match": bool(baseline_equal), "seconds_per_image": seconds / len(prepared),
                             "images_per_second": len(prepared) / seconds, "p95_seconds_per_image": float(np.percentile(samples, 95)),
                             "read_seconds": read_seconds, "reference_write_seconds": write_seconds,
                             "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                    trial["estimated_e1_hours"] = (seconds + read_seconds + write_seconds) * 10000 / len(prepared) / 3600
                    print(f"  batch={size}: {trial['seconds_per_image']:.3f}s/image; parity={equal}; baseline={baseline_equal}", flush=True)
                    if equal:
                        report["batch_sizes"][name] = size
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise
                    x = labels = result = warm = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    trial = {"model": name, "batch_size": size, "status": "oom"}
                report["trials"].append(trial)
                _json(path, report)
            if name not in report["batch_sizes"]:
                raise ValueError(f"No verified batch size for {name}")
            if name == "vit_b_16":
                attention_report = {"raw_parity": True, "rollout_parity": True, "top1_match": True,
                                    "row_sum_error": 0., "e3_seconds": 0., "e4_seconds": 0.}
                size = report["batch_sizes"][name]
                for offset in range(0, len(prepared), size):
                    batch = prepared[offset:offset + size]
                    x = torch.stack([item[1] for item in batch]).to(device)
                    expected = torch.stack([reference[item[0]["image_id"]][0] for item in batch])
                    for experiment, key in (("E3", "raw_parity"), ("E4", "rollout_parity")):
                        _sync(device)
                        start = time.perf_counter()
                        result = attention(model, x, rollout=experiment == "E4")
                        _sync(device)
                        attention_report[experiment.lower() + "_seconds"] += time.perf_counter() - start
                        attention_report[key] &= bool(torch.allclose(expected, result["logits"], rtol=1e-4, atol=1e-5))
                        attention_report["top1_match"] &= bool(torch.equal(expected.argmax(1), result["logits"].argmax(1)))
                        attention_report["row_sum_error"] = max(attention_report["row_sum_error"], result["row_sum_error"])
                        attention_report["shapes"] = result["shapes"]
                report["attention"] = attention_report
                if not all(attention_report[key] for key in ("raw_parity", "rollout_parity", "top1_match")):
                    raise ValueError("Attention adapter changed logits or predictions")
        finally:
            del model, prepared
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Reaching here means batch parity, shapes and attention checks succeeded.
    # Baseline prediction differences are recorded, but do not invalidate CAM.
    report["passed"] = True
    if report["baseline_mismatches"]:
        warnings.warn(
            f"{len(report['baseline_mismatches'])} image/model predictions differ from baseline; "
            "technical checks passed, so experiments may continue. See baseline_mismatches in pilot.json.",
            stacklevel=2,
        )
    report["estimated_e1_hours"] = sum(next(t["estimated_e1_hours"] for t in report["trials"]
        if t["model"] == name and t["batch_size"] == report["batch_sizes"][name]) for name in MODELS)
    _json(path, report)
    pd.DataFrame(report["trials"]).to_csv(output / "benchmark.csv", index=False)
    return report


def require_pilot(root, selected, report, device):
    """Reject stale code/data/device evidence before a long notebook run."""
    import torch
    if (not report.get("passed") or report.get("n_images", 0) < 100
            or report["identity"] != _identity(root, selected, "vit_b_16", "pilot", 1, torch.device(device))):
        raise ValueError("Run a successful current >=100-image pilot first")
    return report["batch_sizes"]


def _sample(row, maps, name, grid, region=None):
    if name not in maps or row.get("map_status", {}).get(name) != "ok":
        return None
    value = sample_grid(maps[name], row["geometry"], grid, region)
    return value if map_status(value) == "ok" else None


def analyze(first, output, second=None):
    """E1: first=ResNet, second=ViT; E4: first=attention, second=ViT E1.

    E2/E3: one store. Return per-image metrics, summary and coverage tables.
    Invalid pairs remain rows with NaN metrics, never artificial zero scores.
    """
    experiment = first.config["experiment"]
    if first.done != set(first.config["image_ids"]):
        raise ValueError("Cannot summarize an incomplete run")
    if experiment in ("E1", "E4") and second is None:
        raise ValueError("A paired run is required")
    if second is not None:
        if second.done != set(second.config["image_ids"]):
            raise ValueError("Paired run is incomplete")
        if (second.config["experiment"] != "E1" or second.config["model"] != "vit_b_16"
                or (experiment == "E1" and first.config["model"] != "resnet50")):
            raise ValueError("Expected ResNet E1/ViT E1 or ViT E4/ViT E1")
        for key in ("code", "manifest", "baseline", "dataset_config", "grids", "target", "preprocessing", "dtype", "torch", "torchvision"):
            if first.config[key] != second.config[key]:
                raise ValueError(f"Paired run mismatch: {key}")
        if not set(first.config["image_ids"]).issubset(second.done):
            raise ValueError("Paired run does not contain all required IDs")
    partner = {row["image_id"]: (row, maps) for row, maps in second.items()} if second else {}
    rows, coverage = [], []
    stores = [first, second] if experiment == "E1" else [first]
    for store in stores:
        names = (list(STAGES) if store.config["experiment"] == "E2" else ["near_head"]
                 if store.config["experiment"] == "E1" else ["block_3", "block_6", "block_9", "block_12"]
                 + (["rollout"] if store.config["experiment"] == "E4" else []))
        for row, maps in store.items():
            for name in names:
                status = row.get("map_status", {}).get(name, row["status"])
                coverage.append({"image_id": row["image_id"], "group": row["group"],
                                 "model": store.config["model"], "stage": name, "status": status,
                                 "baseline_match": row.get("baseline_match", np.nan),
                                 "degenerate": status not in ("ok", "decode_error", "extraction_error"),
                                 "failed": status in ("decode_error", "extraction_error")})
    for row, maps in first.items():
        other, other_maps = partner.get(row["image_id"], ({}, {}))
        for grid in (7, 14):
            base = {"image_id": row["image_id"], "group": row["group"], "grid": grid,
                    "model": first.config["model"]}
            if experiment == "E1":
                a = b = None
                if "geometry" in row and "geometry" in other:
                    region = crop_intersection(row["geometry"], other["geometry"])
                    a = _sample(row, maps, "near_head", grid, region)
                    b = _sample(other, other_maps, "near_head", grid, region)
                entry = {**base, "comparison": "resnet_vs_vit", "entropy_resnet": np.nan,
                         "entropy_vit": np.nan, "entropy_delta_vit_minus_resnet": np.nan,
                         "spearman": np.nan, "top20_iou": np.nan}
                if a is not None and b is not None:
                    entry.update(agreement(a, b), entropy_resnet=entropy(a), entropy_vit=entropy(b),
                                 entropy_delta_vit_minus_resnet=entropy(b) - entropy(a))
                rows.append(entry)
            elif experiment == "E4":
                for name in ("block_12", "rollout"):
                    if "geometry" in row and "geometry" in other and row["geometry"] != other["geometry"]:
                        raise ValueError("ViT CAM and attention crop geometry differs")
                    a, b = _sample(row, maps, name, grid), _sample(other, other_maps, "near_head", grid)
                    rows.append({**base, "comparison": name + "_vs_gradcam", **(agreement(a, b)
                        if a is not None and b is not None else {"spearman": np.nan, "top20_iou": np.nan})})
            else:
                names = list(STAGES) if experiment == "E2" else ["block_3", "block_6", "block_9", "block_12"]
                previous = None
                for name in names:
                    cam = _sample(row, maps, name, grid)
                    rows.append({**base, "stage": name, "entropy": entropy(cam) if cam is not None else np.nan,
                                 "native_shape": str(row.get("native_shapes", {}).get(name)),
                                 **(agreement(previous, cam) if experiment == "E2" and previous is not None
                                    and cam is not None else {"spearman": np.nan, "top20_iou": np.nan})})
                    previous = cam
    table, coverage = pd.DataFrame(rows), pd.DataFrame(coverage)
    summaries, rates = [], []
    keys = [key for key in ("grid", "model", "stage", "comparison") if key in table]
    metrics = [key for key in table if key.startswith("entropy") or key in ("spearman", "top20_iou")]
    for group in ["all", "both_correct", "resnet_only", "vit_only", "both_wrong"]:
        frame = table if group == "all" else table[table.group.eq(group)]
        for identity, part in frame.groupby(keys, dropna=False):
            identity = identity if isinstance(identity, tuple) else (identity,)
            for metric in metrics:
                valid = part[metric].dropna()
                summaries.append({**dict(zip(keys, identity)), "group": group, "metric": metric,
                                  "n_total": len(part), "n_valid": len(valid),
                                  "mean": valid.mean() if len(valid) else np.nan,
                                  "median": valid.median() if len(valid) else np.nan})
        frame = coverage if group == "all" else coverage[coverage.group.eq(group)]
        for (model, stage), part in frame.groupby(["model", "stage"]):
            rates.append({"group": group, "model": model, "stage": stage, "n_images": len(part),
                          "degenerate_rate": part.degenerate.mean(), "failure_rate": part.failed.mean(),
                          "n_predictions": int(part.baseline_match.notna().sum()),
                          "baseline_mismatch_rate": 1 - part.baseline_match.dropna().mean()
                          if part.baseline_match.notna().any() else np.nan})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"per_image": table, "summary": pd.DataFrame(summaries), "coverage": coverage,
              "coverage_summary": pd.DataFrame(rates)}
    for name, frame in result.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    return result


def plot_examples(root, store, selected, destination, limit=8):
    """Fixed IDs per correctness group, plus the first degenerate map if present."""
    import matplotlib.pyplot as plt
    from PIL import Image
    ids = selected.sort_values("image_id").groupby("group", sort=True).head(2).image_id.tolist()[:limit]
    chosen, degenerate = {}, None
    for row, maps in store.items():
        if row["image_id"] in ids:
            chosen[row["image_id"]] = (row, maps)
        elif any(s != "ok" for s in row.get("map_status", {}).values()):
            if degenerate is None or row["image_id"] < degenerate[0]["image_id"]:
                degenerate = (row, maps)
    if degenerate is not None:
        chosen[degenerate[0]["image_id"]] = degenerate
    chosen = dict(sorted(chosen.items()))
    if not chosen:
        return None
    records = pd.DataFrame([row for row, _ in chosen.values()])
    originals = {row["image_id"]: image for row, image, _ in images(root, records)}
    experiment = store.config["experiment"]
    names = (list(STAGES) if experiment == "E2" else ["near_head"] if experiment == "E1"
             else ["block_3", "block_6", "block_9", "block_12"] + (["rollout"] if experiment == "E4" else []))
    fig, axes = plt.subplots(len(chosen), len(names) + 1, squeeze=False,
                             figsize=(3.2 * (len(names) + 1), 3 * len(chosen)))
    for axes_row, (row, maps) in zip(axes, chosen.values()):
        image = originals[row["image_id"]]
        for ax in axes_row:
            ax.axis("off")
        if image is None or "geometry" not in row:
            axes_row[0].text(0, .5, row["status"])
            continue
        geometry = row["geometry"]
        resized = image.resize(tuple(geometry["resized_size"]), Image.Resampling.BILINEAR)
        x, y = geometry["crop_offset"]
        w, h = geometry["crop_size"]
        crop = resized.crop((x, y, x + w, y + h))
        axes_row[0].imshow(crop)
        axes_row[0].set_title(f"{row['image_id']}\nGT={row['label']}, pred={row.get('prediction')}\n{row['group']}", fontsize=7)
        for ax, name in zip(axes_row[1:], names):
            ax.imshow(crop)
            if name in maps:
                heatmap = sample_grid(maps[name], geometry, 224)
                if np.isfinite(heatmap).all():
                    ax.imshow(heatmap, cmap="magma", vmin=0, vmax=1, alpha=.45)
                ax.set_title(f"{name} {maps[name].shape}\n{row['map_status'][name]}")
            else:
                ax.set_title(row["status"])
    fig.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    return fig
