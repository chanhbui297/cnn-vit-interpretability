"""Gradient-based ERF helpers for notebooks/04_erf.ipynb.

The notebook embeds these sections so it can also run by itself on Colab.
Regenerate it with: python src/make_erf_notebook.py
"""

# %% imports
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
import torch
import torchvision
from torchvision.models import (
    ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16,
)
from tqdm.auto import tqdm


MODEL_SPECS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
}
LAYERS = {
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "vit_b_16": ["block3", "block6", "block9", "block12"],
}


# %% sampling
def select_records(project_root, num_images=10, seed=42):
    """Sample uniformly from the approved manifest, never from its ID prefix."""
    root = Path(project_root)
    config = json.loads((root / "configs/analysis_set.json").read_text("utf-8"))
    manifest = root / config["manifest"]
    manifest_bytes = manifest.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(manifest_bytes).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Analysis manifest checksum differs from the shared config.")
    frame = pd.read_csv(manifest)
    counts = Counter(frame.label)
    if (len(frame) != config["num_images"] or not frame.image_id.is_unique
            or not frame.dataset_index.is_unique
            or set(counts) != set(range(config["num_classes"]))
            or set(counts.values()) != {config["images_per_class"]}):
        raise ValueError("Expected 10,000 unique images, exactly 10 per class.")
    if not 1 <= num_images <= len(frame):
        raise ValueError("num_images must be between 1 and 10,000.")
    records = frame.sort_values("image_id").to_dict("records")
    chosen = random.Random(seed).sample(records, num_images)
    return sorted(chosen, key=lambda row: row["dataset_index"])


def load_images(data_dir, records):
    """Stream Parquet; decode only requested IDs and verify their labels/indices."""
    files = sorted(Path(data_dir).glob("validation-*.parquet"))
    if len(files) != 14:
        raise FileNotFoundError(
            f"Expected 14 ImageNet validation shards in {data_dir}. "
            "The repository contains IDs, not the image bytes."
        )
    wanted = {row["image_id"]: row for row in records}
    if len(wanted) != len(records):
        raise ValueError("Duplicate selected image IDs.")
    found = set()
    offset = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        # Skip entire shards that cannot contain any requested global index.
        end = offset + parquet.metadata.num_rows
        if not any(offset <= row["dataset_index"] < end for row in records):
            offset = end
            continue
        for batch in parquet.iter_batches(batch_size=32, columns=["image", "label"]):
            for row in batch.to_pylist():
                index = offset
                offset += 1
                image_id = row["image"]["path"]
                if image_id not in wanted:
                    continue
                expected = wanted[image_id]
                if (image_id in found or row["label"] != expected["label"]
                        or index != expected["dataset_index"]):
                    raise ValueError(f"Dataset/selection mismatch: {image_id}")
                with Image.open(BytesIO(row["image"]["bytes"])) as image:
                    rgb = image.convert("RGB")
                found.add(image_id)
                yield expected, rgb
        if len(found) == len(wanted):
            break
    if found != set(wanted):
        raise ValueError(f"Missing selected images: {sorted(set(wanted) - found)[:5]}")


# %% gradient
def center_features(model, model_name, x):
    """Return center vectors AFTER each complete residual stage/block."""
    features = {}
    positions = {}
    if model_name == "resnet50":
        h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        for name in LAYERS[model_name]:
            h = getattr(model, name)(h)
            row, col = h.shape[-2] // 2, h.shape[-1] // 2
            features[name] = h[0, :, row, col]
            positions[name] = {"grid_row": row, "grid_col": col,
                               "grid_height": h.shape[-2], "grid_width": h.shape[-1],
                               "token_index": -1}
    elif model_name == "vit_b_16":
        # Match torchvision's patch embedding -> CLS -> position -> encoder path.
        # Stop before encoder.ln: channel averaging after LayerNorm is degenerate.
        h = model._process_input(x)
        grid = model.image_size // model.patch_size
        row, col = grid // 2, grid // 2
        token_index = 1 + row * grid + col  # +1 skips the CLS token.
        h = torch.cat([model.class_token.expand(x.shape[0], -1, -1), h], dim=1)
        h = model.encoder.dropout(h + model.encoder.pos_embedding)
        for depth, block in enumerate(model.encoder.layers, start=1):
            h = block(h)
            name = f"block{depth}"
            if name in LAYERS[model_name]:
                features[name] = h[0, token_index, :]
                positions[name] = {"grid_row": row, "grid_col": col,
                                   "grid_height": grid, "grid_width": grid,
                                   "token_index": token_index}
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return features, positions


@torch.enable_grad()
def compute_erf(model, model_name, normalized_image, std):
    """Differentiate center feature energy with respect to RGB crop in [0,1]."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Run ERF outside torch.inference_mode().")
    if normalized_image.shape != (1, 3, 224, 224):
        raise ValueError("Expected one normalized RGB image: [1, 3, 224, 224].")
    if len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Expected three positive RGB normalization std values.")
    model.eval()  # Disable dropout; use saved BatchNorm statistics.
    x = normalized_image.detach().clone().requires_grad_(True)
    features, positions = center_features(model, model_name, x)
    scale = x.new_tensor(std).view(1, 3, 1, 1)
    maps = {}
    for stage, (name, vector) in enumerate(features.items(), start=1):
        # One explicit scalar for both architectures. This is a sensitivity probe,
        # not an exact all-channel Jacobian norm or a class attribution.
        z = vector.square().mean()
        gradient = torch.autograd.grad(
            z, x, retain_graph=stage < len(features), create_graph=False,
        )[0]
        # x_norm=(RGB-mean)/std, so dz/dRGB = (dz/dx_norm)/std.
        raw = (gradient / scale).abs().sum(dim=1)[0].detach().cpu().numpy()
        raw = raw.astype(np.float32)
        if not np.isfinite(raw).all() or float(raw.sum()) <= 0:
            raise ValueError(f"Zero or non-finite gradient for {model_name}/{name}.")
        maps[name] = {"raw": raw, "stage": stage, "target_value": z.item(),
                      **positions[name]}
    return maps


# %% metrics
def erf_metrics(raw):
    """Area = smallest NUMBER of pixels reaching a fraction of absolute gradient."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw < 0).any() or raw.sum() <= 0:
        raise ValueError("ERF must be a finite, nonnegative 2D map with positive sum.")
    p = raw / raw.sum()
    ranked = np.sort(p.ravel())[::-1]
    cumulative = np.cumsum(ranked)

    def area(q):
        count = min(np.searchsorted(cumulative, q, side="left") + 1, p.size)
        return float(count / p.size)

    yy, xx = np.indices(p.shape, dtype=np.float64)
    yy /= max(p.shape[0] - 1, 1)
    xx /= max(p.shape[1] - 1, 1)
    cy, cx = (p * yy).sum(), (p * xx).sum()
    dispersion = np.sqrt((p * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum())
    # Central square occupies the middle half along each axis (~25% area).
    center = (np.abs(yy - 0.5) <= 0.25) & (np.abs(xx - 0.5) <= 0.25)
    center_energy = p[center].sum()
    return {
        "area50_fraction": area(0.5), "area90_fraction": area(0.9),
        "dispersion_normalized": float(dispersion),
        "center_energy_fraction": float(center_energy),
        "outer_energy_fraction": float(1 - center_energy),
        "centroid_y": float(cy), "centroid_x": float(cx),
        "raw_gradient_sum": float(raw.sum()),
    }


# %% plotting
def plot_image_maps(image_id, previews, destination=None):
    """Show each model's own crop and four ERF maps; color is log10 probability."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    for row, model_name in enumerate(MODEL_SPECS):
        crop, maps = previews[model_name]
        axes[row, 0].imshow(crop)
        axes[row, 0].set_title(f"{model_name}: RGB crop")
        for col, (layer, result) in enumerate(maps.items(), start=1):
            raw = result["raw"]
            probability = raw / raw.sum()
            heat = axes[row, col].imshow(
                np.log10(probability + 1e-12), cmap="magma", vmin=-8, vmax=-2,
                interpolation="nearest",
            )
            area90 = erf_metrics(raw)["area90_fraction"]
            axes[row, col].set_title(f"{layer}\nA90 = {area90:.1%}")
        for axis in axes[row]:
            axis.axis("off")
    fig.colorbar(heat, ax=axes[:, 1:].ravel().tolist(), shrink=0.7,
                 label="log10(normalized absolute gradient + 1e-12)")
    fig.suptitle(image_id)
    if destination is not None:
        fig.savefig(destination, dpi=160)
    return fig


def plot_summary(metrics, destination=None):
    """Error bars are across-image standard deviations, not confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model_name, group in metrics.groupby("model", sort=False):
        grouped = group.groupby("stage")
        for axis, metric, title in zip(
            axes, ["area90_fraction", "dispersion_normalized"],
            ["Smallest pixel area containing 90% gradient mass", "Spatial dispersion"],
        ):
            mean = grouped[metric].mean()
            sd = grouped[metric].std().fillna(0)
            axis.errorbar(mean.index, mean, yerr=sd, marker="o", capsize=3,
                          label=model_name)
            axis.set(title=title, xlabel="Relative stage (not equivalent layers)",
                     xticks=[1, 2, 3, 4])
            axis.grid(alpha=0.2)
            axis.legend()
    axes[0].set_ylabel("Fraction of input pixels (mean ± SD)")
    axes[1].set_ylabel("Normalized-coordinate RMS distance (mean ± SD)")
    if destination is not None:
        fig.savefig(destination, dpi=160)
    return fig


# %% experiment
def run_experiment(records, data_dir, output_dir, device, seed=42,
                   preprocessing="official", num_previews=3):
    """One model at a time; save raw maps, per-image metrics, and provenance."""
    if preprocessing not in {"official", "shared_vit_crop"}:
        raise ValueError("preprocessing must be official or shared_vit_crop.")
    # Preflight dataset BEFORE creating output files or downloading model weights.
    loader = load_images(data_dir, records)
    first = next(loader)
    loader.close()
    del first
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # Never mix runs or overwrite results.
    (output / "maps").mkdir()
    (output / "figures").mkdir()
    pd.DataFrame(records).to_csv(output / "selected_images.csv", index=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    metadata = {
        "status": "running", "num_images": len(records), "seed": seed,
        "device": str(device), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "preprocessing": preprocessing,
        "target": "mean(square(center_feature_vector)); post-stage/post-block, pre-final-LN",
        "center_rule": "row=grid_height//2, col=grid_width//2; zero-based",
        "gradient_coordinates": "RGB [0,1] AFTER resize/crop, BEFORE normalization",
        "layers": LAYERS, "models": {},
        "selected_images_sha256": hashlib.sha256(
            (output / "selected_images.csv").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else None,
    }
    metadata_path = output / "run.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_metadata()
    rows, previews = [], {}
    preview_ids = {row["image_id"] for row in records[:num_previews]}
    model = None
    try:
        for model_name, (factory, weights) in MODEL_SPECS.items():
            transform_weights = (ViT_B_16_Weights.IMAGENET1K_V1
                                 if preprocessing == "shared_vit_crop" else weights)
            transform = transform_weights.transforms()
            metadata["models"][model_name] = {
                "weights": str(weights), "transform": repr(transform),
            }
            save_metadata()
            model = factory(weights=weights).to(device).eval()
            model.requires_grad_(False)  # Input gradient still works; no training.
            for record, image in tqdm(load_images(data_dir, records),
                                      total=len(records), desc=model_name):
                x = transform(image).unsqueeze(0).to(device)
                maps = compute_erf(model, model_name, x, transform.std)
                image_id = record["image_id"]
                # Every raw map is retained, including maps not selected for preview.
                np.savez_compressed(
                    output / "maps" / f"{model_name}_{record['dataset_index']:05d}.npz",
                    **{name: result["raw"] for name, result in maps.items()},
                )
                for layer, result in maps.items():
                    rows.append({**record, "model": model_name, "layer": layer,
                                 **{k: v for k, v in result.items() if k != "raw"},
                                 **erf_metrics(result["raw"])})
                if image_id in preview_ids:
                    rgb = (x[0].detach().cpu() * torch.tensor(transform.std)[:, None, None]
                           + torch.tensor(transform.mean)[:, None, None])
                    previews.setdefault(image_id, {})[model_name] = (
                        rgb.permute(1, 2, 0).clamp(0, 1).numpy(), maps,
                    )
                # Checkpoint CSV after each image; an interrupted run stays marked running.
                pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                del x, maps
            del model
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics = pd.DataFrame(rows)
        expected_rows = len(records) * sum(map(len, LAYERS.values()))
        if len(metrics) != expected_rows:
            raise ValueError(f"Expected {expected_rows} metrics rows, got {len(metrics)}.")
        summary = metrics.groupby(["model", "stage", "layer"])[[
            "area50_fraction", "area90_fraction", "dispersion_normalized",
            "center_energy_fraction", "outer_energy_fraction",
        ]].agg(["mean", "std", "count"])
        summary.to_csv(output / "summary.csv")
        for index, (image_id, preview) in enumerate(previews.items()):
            fig = plot_image_maps(image_id, preview, output / "figures" / f"image_{index:02d}.png")
            plt.close(fig)
        fig = plot_summary(metrics, output / "figures" / "comparison.png")
        plt.close(fig)
        metadata["status"] = "complete"
        save_metadata()
        return metrics, summary, previews
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        save_metadata()
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
