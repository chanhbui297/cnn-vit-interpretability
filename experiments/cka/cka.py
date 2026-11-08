"""Pooled layer features and exact, whole-cohort linear CKA. No import side effects."""

import csv
import json
from pathlib import Path

import numpy as np


RESNET_LAYERS = [f"layer{i}" for i in range(1, 5)]
VIT_LAYERS = [f"encoder.layers.encoder_layer_{i}" for i in range(12)]
LAYER_DIMS = {
    "resnet50": dict(zip(RESNET_LAYERS, [256, 512, 1024, 2048])),
    "vit_b_16": dict.fromkeys(VIT_LAYERS, 768),
}
POOLING = {"resnet50": "spatial mean: output.mean(dim=(2, 3))",
           "vit_b_16": "patch-token mean, excluding CLS: output[:, 1:, :].mean(dim=1)"}


def alignment_order(ids, labels, records):
    """Return row indices in manifest order; reject duplicates, ID/label mismatch."""
    ids, labels, records = list(ids), list(labels), list(records)
    expected = {r["image_id"]: int(r["label"]) for r in records}
    if (not expected or len(expected) != len(records) or len(ids) != len(labels)
            or len(set(ids)) != len(ids) or any(not i for i in ids)):
        raise ValueError("Expected unique, nonempty IDs and matching label counts")
    if set(ids) != set(expected):
        raise ValueError("Feature IDs and manifest cohort differ")
    positions = {image_id: i for i, image_id in enumerate(ids)}
    if any(int(labels[positions[i]]) != label for i, label in expected.items()):
        raise ValueError("Feature labels and manifest labels differ")
    return np.array([positions[r["image_id"]] for r in records], dtype=np.int64)


def extract_features(model, transform, samples, model_name, records, *, batch_size=16):
    """Hook stage/block outputs, pool immediately, store float32 on CPU by image ID.

    A fresh fully consumed src.data.iter_analysis_images iterator is required for
    each model. Only pooled activations survive a forward pass. Hooks return None
    and are removed even when decoding, validation or inference raises an error.
    """
    import torch

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if any(m.training for m in model.modules()):
        raise ValueError("Call model.eval() before extracting features")
    records = list(records)
    ids = [r["image_id"] for r in records]
    labels = [int(r["label"]) for r in records]
    alignment_order(ids, labels, records)
    positions = {image_id: i for i, image_id in enumerate(ids)}
    dimensions = LAYER_DIMS[model_name]
    features = {name: np.empty((len(ids), dim), dtype=np.float32)
                for name, dim in dimensions.items()}
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    pooled, handles, images, batch_rows, seen = {}, [], [], [], set()

    def make_hook(name):
        def hook(module, inputs, output):
            if model_name == "resnet50":
                if output.ndim != 4:
                    raise ValueError(f"Unexpected stage shape at {name}")
                value = output.mean(dim=(2, 3))
            else:
                if output.ndim != 3 or output.shape[1] != 197:
                    raise ValueError(f"Expected CLS + 196 patch tokens at {name}")
                value = output[:, 1:, :].mean(dim=1)
            # Keep only pooled vectors; transfer all layers together once per batch.
            pooled[name] = value.detach()
        return hook

    def flush():
        pooled.clear()
        with torch.inference_mode():
            logits = model(torch.stack(images).to(device))
            if logits.shape != (len(images), 1000) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite ImageNet logits")
        if set(pooled) != set(dimensions):
            raise ValueError("Not all feature hooks ran")
        combined = torch.cat([pooled[name] for name in dimensions], dim=1)
        combined = combined.to(device="cpu", dtype=torch.float32).numpy()
        offset = 0
        for name, dim in dimensions.items():
            value = combined[:, offset:offset + dim]
            if value.shape != (len(images), dim) or not np.isfinite(value).all():
                raise ValueError(f"Invalid pooled features at {name}")
            features[name][batch_rows] = value
            offset += dim
        images.clear()
        batch_rows.clear()
        pooled.clear()

    try:
        for name in dimensions:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        from tqdm.auto import tqdm

        for image, label, image_id in tqdm(samples, total=len(ids), desc=model_name, unit="image"):
            if image_id not in positions or image_id in seen:
                raise ValueError(f"Unexpected or duplicate image ID: {image_id}")
            row = positions[image_id]
            if int(label) != labels[row]:
                raise ValueError(f"Label mismatch: {image_id}")
            seen.add(image_id)
            images.append(transform(image))
            batch_rows.append(row)
            if len(images) == batch_size:
                flush()
        if images:
            flush()
        if seen != set(ids):
            raise ValueError("Incomplete feature cohort")
    finally:
        for handle in handles:
            handle.remove()
    return features, ids, labels


def load_or_extract_features(model, transform, samples, model_name, records,
                             cache_path, fingerprint, *, batch_size=16,
                             reuse=False, save=True):
    """Read a compatible complete cache, or extract and save one cache per model.

    Returns (features in manifest order, cache_used). Partial files are never read.
    """
    cache_path = Path(cache_path)
    dimensions = LAYER_DIMS[model_name]
    cache_used = reuse and cache_path.is_file()
    if cache_used:
        with np.load(cache_path, allow_pickle=False) as cached:
            if set(cached.files) != {*dimensions, "ids", "labels", "metadata"}:
                raise ValueError("Incomplete cache; disable REUSE_FEATURES to regenerate")
            if json.loads(str(cached["metadata"].item())) != fingerprint:
                raise ValueError("Incompatible cache; disable REUSE_FEATURES to regenerate")
            ids, labels = cached["ids"].tolist(), cached["labels"].tolist()
            features = {name: cached[name] for name in dimensions}
    else:
        features, ids, labels = extract_features(
            model, transform, samples, model_name, records, batch_size=batch_size)

    order = alignment_order(ids, labels, records)
    for name, dim in dimensions.items():
        value = features[name]
        if (value.shape != (len(records), dim) or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError(f"Invalid feature/cache shape or dtype: {name}")
    if save and not cache_used:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_path.with_suffix(".npz.partial")
        with partial.open("wb") as stream:
            np.savez(stream, **features, ids=np.array(ids), labels=np.array(labels),
                     metadata=np.array(json.dumps(fingerprint, sort_keys=True)))
        partial.replace(cache_path)
    if not np.array_equal(order, np.arange(len(records))):
        features = {name: value[order] for name, value in features.items()}
    return features, bool(cache_used)


def _validate_features(x):
    x = np.asanyarray(x)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("Features must have shape (N >= 2, D >= 1)")
    if x.dtype.kind not in "fi" or not np.isfinite(x).all():
        raise ValueError("Features must be finite real numbers")
    if not np.any(x != x[0]):
        raise ValueError("Undefined CKA: all feature rows are identical")
    return x


def _centered_cross(x, y, mean_x, mean_y, block_size):
    # Center with GLOBAL means, then add cross-products; never center per batch.
    cross = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    for start in range(0, len(x), block_size):
        a = np.asarray(x[start:start + block_size], dtype=np.float64) - mean_x
        b = np.asarray(y[start:start + block_size], dtype=np.float64) - mean_y
        cross += a.T @ b
    return cross


def cka_matrix(left, right, *, block_size=512):
    """Return exact biased linear CKA for already ID-aligned feature dictionaries.

    Global float64 means and block sums reproduce Xc.T @ Yc, up to roundoff.
    Memory: pooled features + O(block_size * (Dx + Dy) + max(D)**2), no NxN
    Gram matrices. Self norms are computed once per layer, reused across pairs.
    Undefined (constant/zero-variance) representations raise ValueError.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not left or not right:
        raise ValueError("Both layer dictionaries must be nonempty")
    left = {k: _validate_features(v) for k, v in left.items()}
    right = {k: _validate_features(v) for k, v in right.items()}
    if len({len(v) for v in [*left.values(), *right.values()]}) != 1:
        raise ValueError("All layers must have the same aligned image count")
    statistics = []
    for layers in (left, right):
        stats = {}
        for name, x in layers.items():
            mean = x.mean(axis=0, dtype=np.float64)
            gram = _centered_cross(x, x, mean, mean, block_size)
            norm = np.linalg.norm(gram, "fro")
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"Undefined CKA: zero or invalid variance at {name}")
            stats[name] = (mean, norm)
        statistics.append(stats)
    matrix = np.empty((len(left), len(right)), dtype=np.float64)
    for i, (a, x) in enumerate(left.items()):
        mean_x, norm_x = statistics[0][a]
        for j, (b, y) in enumerate(right.items()):
            mean_y, norm_y = statistics[1][b]
            cross = _centered_cross(x, y, mean_x, mean_y, block_size)
            # Divide before squaring to reduce overflow for large feature scales.
            score = (np.linalg.norm(cross, "fro") / np.sqrt(norm_x) / np.sqrt(norm_y)) ** 2
            if not np.isfinite(score) or not -1e-10 <= score <= 1 + 1e-10:
                raise ValueError("Numerically invalid CKA; inspect feature scales")
            matrix[i, j] = np.clip(score, 0, 1)  # roundoff only
    return matrix


def linear_cka(x, y, *, block_size=512):
    """Single-pair version of cka_matrix, useful for small mathematical checks."""
    return float(cka_matrix({"X": x}, {"Y": y}, block_size=block_size)[0, 0])


def save_results(matrix, metadata, root, run_id):
    """Write only labeled CSV, metadata JSON and one heatmap; refuse overwrites."""
    import matplotlib.pyplot as plt

    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id must contain only letters, numbers, '_' or '-'")
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 12) or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Expected a finite 4 x 12 CKA matrix in [0, 1]")
    metrics = Path(root) / "results/metrics/experiments/cka" / run_id / "final"
    figures = Path(root) / "results/figures/experiments/cka" / run_id / "final"
    if any(p.exists() and any(p.iterdir()) for p in (metrics, figures)):
        raise FileExistsError("Run folder is not empty; choose a new run_id")
    metadata_text = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (metrics / "cka_matrix.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["resnet50_layer / vit_b_16_block", *VIT_LAYERS])
        writer.writerows([name, *row] for name, row in zip(RESNET_LAYERS, matrix))
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(12), [f"block {i + 1}" for i in range(12)])
    ax.set_yticks(range(4), RESNET_LAYERS)
    ax.set_xlabel("ViT-B/16 V1 encoder block output (patch-token mean; CLS excluded)")
    ax.set_ylabel("ResNet50 V2 stage output\n(spatial mean)")
    status = "SMOKE TEST ONLY" if metadata["mode"] == "smoke" else "Full analysis cohort"
    ax.set_title(f"Linear CKA | clean inputs | {metadata['num_images']:,} images | {status}")
    for i in range(4):
        for j in range(12):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="black" if matrix[i, j] > 0.55 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Linear CKA (0–1)")
    fig.savefig(figures / "cka_heatmap.png", dpi=180)
    # Written last: a run without metadata is incomplete and should not be cited.
    (metrics / "metadata.json").write_text(metadata_text + "\n", encoding="utf-8")
    return fig
