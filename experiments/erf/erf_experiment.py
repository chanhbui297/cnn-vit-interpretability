"""Canonical ERF probes and runner, shared by the 2,000-image notebook and CLI.

The 10/100-image notebooks are historical tutorials/pilots. Do not regenerate
them from this module or use their embedded helpers for the primary study.
"""

# %% imports
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm.auto import tqdm

LAYERS = {
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "vit_b_16": ["block3", "block6", "block9", "block12"],
}


# %% sampling
# These imports run after PROJECT_ROOT is added to sys.path in the notebook.
from src.models import load_model
from src.data import load_analysis_manifest, iter_analysis_images


def select_records(project_root, num_images=100, seed=42):
    """Validate the shared manifest, then choose a reproducible pilot cohort."""
    root = Path(project_root)
    config = json.loads((root / "configs/analysis_set.json").read_text("utf-8"))
    manifest = root / config["manifest"]
    normalized = manifest.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Analysis manifest checksum differs from the shared config.")
    records = load_analysis_manifest(manifest)  # Validate all 10,000 IDs; no first-N limit.
    if not isinstance(num_images, int) or isinstance(num_images, bool) or not 1 <= num_images <= len(records):
        raise ValueError("num_images must be an integer between 1 and 10,000.")
    chosen = random.Random(seed).sample(sorted(records, key=lambda row: row["image_id"]), num_images)
    return sorted(chosen, key=lambda row: row["dataset_index"])


def select_main_records(project_root, seed=42):
    """Reproduce the historical primary cohort: two images per class, 2,000 IDs."""
    # select_records validates the checksum and the entire manifest.
    full = select_records(project_root, 10000, seed)
    grouped = {label: [] for label in range(1000)}
    for record in full:
        grouped[record["label"]].append(record)
    rng = random.Random(seed)
    chosen = []
    for label in range(1000):
        chosen.extend(rng.sample(sorted(grouped[label], key=lambda r: r["image_id"]), 2))
    chosen.sort(key=lambda row: row["dataset_index"])
    validate_main_cohort(chosen)
    return chosen


def validate_main_cohort(records):
    counts = Counter(row["label"] for row in records)
    if (len(records) != 2000 or counts != Counter({label: 2 for label in range(1000)})
            or len({row["image_id"] for row in records}) != 2000
            or len({row["dataset_index"] for row in records}) != 2000):
        raise ValueError("Expected 2,000 unique images, exactly 2 per class.")
    return counts


def select_sensitivity_records(main_records, seed=42):
    """Fixed 200-image subset; no correctness filtering or per-probe sampling."""
    validate_main_cohort(main_records)
    chosen = random.Random(seed).sample(sorted(main_records, key=lambda r: r["image_id"]), 200)
    return sorted(chosen, key=lambda row: row["dataset_index"])


def load_images(data_dir, records):
    """Adapt the shared RGB iterator to ERF; consume fully to validate missing IDs."""
    records = list(records)
    wanted = {row["image_id"]: row for row in records}
    if not wanted or len(wanted) != len(records):
        raise ValueError("Provide nonempty unique image IDs.")
    for image, label, image_id in iter_analysis_images(records, data_dir=data_dir):
        record = wanted[image_id]
        if label != record["label"]:
            raise ValueError(f"Label mismatch: {image_id}")
        yield record, image


def erf_output_paths(project_root, run_id):
    """Return protocol-compliant metrics, figures, and ignored intermediates."""
    if not run_id or Path(run_id).name != run_id or any(c in run_id for c in '/\\') or run_id in {'.', '..'}:
        raise ValueError("run_id must be a single nonempty directory name.")
    root = Path(project_root)
    run_root = root / "results/metrics/experiments/erf" / run_id
    return (run_root / "final",
            root / "results/figures/experiments/erf" / run_id / "final",
            run_root / "intermediate")


def erf_code_provenance(project_root):
    root = Path(project_root)
    def git_output(*arguments):
        try:
            return subprocess.check_output(["git", "-C", str(root), *arguments],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    revision = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain", "--untracked-files=normal")
    # Drive may contain a file copy without .git; record unknown explicitly.
    files = ["experiments/erf/erf_experiment.py", "experiments/erf/erf_statistics.py",
             "experiments/erf/erf_workflow.py", "notebooks/04_erf.ipynb",
             "src/models.py", "src/data.py", "src/evaluation.py"]
    return {"git_revision": revision, "git_dirty": None if status is None else bool(status),
            "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in files if (root / name).is_file()}}


# %% gradient
def probe_position(height, width, position="center"):
    """Indices are grid coordinates, not exactly aligned input receptive centers."""
    if position == "center":
        return height // 2, width // 2
    if position == "quarter":
        return round(0.25 * (height - 1)), round(0.25 * (width - 1))
    raise ValueError("position must be center or quarter.")


def probe_description(target_mode="energy", position="center"):
    if target_mode not in {"energy", "absolute"}:
        raise ValueError("target_mode must be energy or absolute.")
    probe_position(14, 14, position)
    operation = "square" if target_mode == "energy" else "abs"
    vector = "center_feature_vector" if position == "center" else "quarter_feature_vector"
    return {
        "target_mode": target_mode, "position": position,
        "target": f"mean({operation}({vector})); post-stage/post-block, pre-final-LN",
        "center_rule": ("row=grid_height//2, col=grid_width//2; zero-based" if position == "center"
                        else "row=round(0.25*(grid_height-1)), col=round(0.25*(grid_width-1)); zero-based"),
    }


def center_features(model, model_name, x, position="center"):
    """Return selected vectors AFTER each complete residual stage/block."""
    features = {}
    positions = {}
    if model_name == "resnet50":
        h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        for name in LAYERS[model_name]:
            h = getattr(model, name)(h)
            row, col = probe_position(h.shape[-2], h.shape[-1], position)
            features[name] = h[0, :, row, col]
            positions[name] = {"grid_row": row, "grid_col": col,
                               "grid_height": h.shape[-2], "grid_width": h.shape[-1],
                               "token_index": -1}
    elif model_name == "vit_b_16":
        # Match torchvision's patch embedding -> CLS -> position -> encoder path.
        # Stop before encoder.ln: channel averaging after LayerNorm is degenerate.
        h = model._process_input(x)
        grid = model.image_size // model.patch_size
        row, col = probe_position(grid, grid, position)
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
def compute_erf(model, model_name, normalized_image, std, *,
                target_mode="energy", position="center"):
    """Differentiate an explicit feature scalar with respect to RGB crop [0,1]."""
    probe_description(target_mode, position)
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Run ERF outside torch.inference_mode().")
    if normalized_image.shape != (1, 3, 224, 224):
        raise ValueError("Expected one normalized RGB image: [1, 3, 224, 224].")
    if len(std) != 3 or any(not np.isfinite(value) or value <= 0 for value in std):
        raise ValueError("Expected three positive RGB normalization std values.")
    model.eval()  # Disable dropout; use saved BatchNorm statistics.
    x = normalized_image.detach().clone().requires_grad_(True)
    features, positions = center_features(model, model_name, x, position)
    scale = x.new_tensor(std).view(1, 3, 1, 1)
    maps = {}
    for stage, (name, vector) in enumerate(features.items(), start=1):
        # One explicit scalar for both architectures. This is a sensitivity probe,
        # not an exact all-channel Jacobian norm or a class attribution.
        z = vector.square().mean() if target_mode == "energy" else vector.abs().mean()
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
    center_energy = np.clip(p[center].sum(), 0.0, 1.0)
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
    for row, model_name in enumerate(LAYERS):
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
                   preprocessing="official", num_previews=3, *, project_root,
                   allow_download=False, save_raw_maps=False, checkpoint_every=50,
                   target_mode="energy", position="center", preview_ids=None):
    """Use shared model/data APIs; save ERF CSV/JSON and figures under final/."""
    if preprocessing not in {"official", "shared_vit_crop"}:
        raise ValueError("preprocessing must be official or shared_vit_crop.")
    probe = probe_description(target_mode, position)
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer.")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Run ERF outside torch.inference_mode().")
    records = list(records)
    if not records or len({r["image_id"] for r in records}) != len(records):
        raise ValueError("Provide a nonempty cohort with unique IDs.")
    if len(records) == 2000:
        validate_main_cohort(records)
    if not isinstance(num_previews, int) or not 0 <= num_previews <= len(records):
        raise ValueError("num_previews must be between zero and the cohort size.")
    chosen_previews = (set(preview_ids) if preview_ids is not None else
                       {r["image_id"] for r in random.Random(seed).sample(records, num_previews)})
    if not chosen_previews.issubset({r["image_id"] for r in records}):
        raise ValueError("Preview IDs must belong to the selected cohort.")
    root, output = Path(project_root), Path(output_dir)
    expected, figure_dir, intermediate = erf_output_paths(root, output.parent.name)
    if output.resolve() != expected.resolve():
        raise ValueError(f"Use the protocol metrics directory: {expected}")
    # A lightweight probe; the main loops below always exhaust the shared iterator.
    loader = load_images(data_dir, records[:1])
    try:
        next(loader)
    finally:
        loader.close()
    if output.exists() or figure_dir.exists() or intermediate.exists():
        raise FileExistsError("Run ID already exists; choose a new run_id.")
    output.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    intermediate.mkdir(parents=True, exist_ok=False)
    maps_dir = intermediate / "maps"
    maps_dir.mkdir()
    pd.DataFrame(records).to_csv(output / "selected_images.csv", index=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = json.loads((root / "configs/analysis_set.json").read_text("utf-8"))
    manifest = root / config["manifest"]
    started = time.perf_counter()
    metadata = {
        "experiment": "erf", "run_id": output.parent.name,
        "status": "running", "num_images": len(records), "seed": seed,
        "device": str(device), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "preprocessing": preprocessing, "condition": preprocessing, "severity": "",
        "batch_size": 1, "allow_download": allow_download, "checkpoint_every": checkpoint_every,
        **probe,
        "gradient_coordinates": "RGB [0,1] AFTER resize/crop, BEFORE normalization",
        "layers": LAYERS, "models": {}, "exclusions": [],
        "selection_rule": "caller-provided cohort; IDs in selected_images.csv; no correctness filtering",
        "cohort_kind": "class_balanced_2000_subset" if len(records) == 2000 else "subset",
        "num_classes": len({r["label"] for r in records}),
        "images_per_class": 2 if len(records) == 2000 else None,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "selected_images_sha256": hashlib.sha256((output / "selected_images.csv").read_bytes()).hexdigest(),
        "cohort_sha256_lf": hashlib.sha256((output / "selected_images.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "preview_ids": sorted(chosen_previews), "raw_maps": "all" if save_raw_maps else "previews_only",
        "timings_seconds": {}, "peak_cuda_memory_bytes": {},
        "metrics_dir": str(output), "figures_dir": str(figure_dir),
        "code": erf_code_provenance(root),
        "metric_units": {"area50_fraction": "fraction of input pixels", "area90_fraction": "fraction of input pixels",
                         "dispersion_normalized": "normalized-coordinate RMS distance"},
    }
    metadata_path = output / "metadata.json"
    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_metadata()
    rows, previews, model, vit_transform = [], {}, None, None
    try:
        # ViT first provides its official transform for the common-view sensitivity run.
        for model_name in ("vit_b_16", "resnet50"):
            model, official_transform, model_metadata = load_model(
                model_name, device=device, allow_download=allow_download,
            )
            if model_name == "vit_b_16":
                vit_transform = official_transform
            transform = vit_transform if preprocessing == "shared_vit_crop" else official_transform
            metadata["models"][model_name] = {
                "weights": model_metadata["weights"], "transform": repr(transform),
                "applied_preprocessing": {key: getattr(transform, key) for key in
                    ["resize_size", "crop_size", "mean", "std", "antialias"]},
                "baseline_loader": model_metadata,
            }
            save_metadata()
            model.requires_grad_(False)  # Input gradients remain enabled; no training.
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            model_started = time.perf_counter()
            for image_number, (record, image) in enumerate(
                tqdm(load_images(data_dir, records), total=len(records), desc=model_name), start=1
            ):
                x = transform(image).unsqueeze(0).to(device)
                maps = compute_erf(model, model_name, x, transform.std,
                                   target_mode=target_mode, position=position)
                image_id = record["image_id"]
                if save_raw_maps or image_id in chosen_previews:
                    np.savez_compressed(maps_dir / f"{model_name}_{record['dataset_index']:05d}.npz",
                                        **{name: result["raw"] for name, result in maps.items()})
                for layer, result in maps.items():
                    rows.append({**record, "model": model_name, "model_name": model_name,
                                 "experiment": "erf", "run_id": output.parent.name,
                                 "condition": preprocessing, "severity": "", "layer": layer,
                                 "target_mode": target_mode, "position": position,
                                 **{k: v for k, v in result.items() if k != "raw"},
                                 **erf_metrics(result["raw"])})
                if image_id in chosen_previews:
                    rgb = (x[0].detach().cpu() * torch.tensor(transform.std)[:, None, None]
                           + torch.tensor(transform.mean)[:, None, None])
                    previews.setdefault(image_id, {})[model_name] = (
                        rgb.permute(1, 2, 0).clamp(0, 1).numpy(), maps)
                # Partial metrics stay outside final until both model loops finish.
                if image_number == 1 or image_number % checkpoint_every == 0:
                    pd.DataFrame(rows).to_csv(intermediate / "metrics_checkpoint.csv", index=False)
                del x, maps
            pd.DataFrame(rows).to_csv(intermediate / "metrics_checkpoint.csv", index=False)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
                metadata["peak_cuda_memory_bytes"][model_name] = torch.cuda.max_memory_allocated(device)
            metadata["timings_seconds"][model_name] = time.perf_counter() - model_started
            del model
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics = pd.DataFrame(rows)
        if len(metrics) != len(records) * 8 or metrics.duplicated(["image_id", "model", "stage"]).any():
            raise ValueError("Expected 8 unique image/model/stage rows per selected image.")
        metrics.to_csv(output / "metrics.csv", index=False)
        summary = metrics.groupby(["model", "stage", "layer"])[[
            "area50_fraction", "area90_fraction", "dispersion_normalized",
            "center_energy_fraction", "outer_energy_fraction",
        ]].agg(["mean", "std", "count"])
        summary.to_csv(output / "summary.csv")
        caption = f"{preprocessing}; ResNet50 V2 / ViT-B16 V1; {target_mode}; {position} feature"
        for index, (image_id, preview) in enumerate(previews.items()):
            fig = plot_image_maps(image_id, preview)
            fig.suptitle(f"{image_id}\n{caption}", fontsize=10)
            fig.savefig(figure_dir / f"image_{index:02d}.png", dpi=160)
            plt.close(fig)
        fig = plot_summary(metrics)
        fig.suptitle(caption, fontsize=9)
        fig.savefig(figure_dir / "comparison.png", dpi=160)
        plt.close(fig)
        metadata["status"] = "complete"
        metadata["timings_seconds"]["total"] = time.perf_counter() - started
        metadata["artifact_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                       for name in ["selected_images.csv", "metrics.csv", "summary.csv"]}
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
