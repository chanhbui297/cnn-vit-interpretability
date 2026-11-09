"""Canonical paired inference, extracted from the executed 2,000-image study.

Keep the bootstrap RNG, ordering, and Wilcoxon rounding unchanged so that
historical CSVs can be independently recomputed. New sensitivity comparisons
are descriptive and are kept outside the original 16-test Holm family.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon


ERF_METRICS = {
    "area90_fraction": (100.0, "percentage points"),
    "dispersion_normalized": (1.0, "normalized-coordinate RMS distance"),
}
ERF_STAGE_LAYERS = {
    "resnet50": {1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4"},
    "vit_b_16": {1: "block3", 2: "block6", 3: "block9", 4: "block12"},
}


def validate_erf_metrics(metrics, records):
    """Require every selected image/model/stage exactly once; never drop pairs."""
    required = {"image_id", "dataset_index", "label", "model", "stage", "layer",
                *ERF_METRICS}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Missing metrics columns: {sorted(required - set(metrics.columns))}")
    selected = pd.DataFrame(records)[["image_id", "dataset_index", "label"]]
    if selected.empty or selected.isna().any().any():
        raise ValueError("Selected records must be nonempty and complete.")
    if not selected.image_id.is_unique or not selected.dataset_index.is_unique:
        raise ValueError("Selected image IDs and dataset indices must be unique.")
    keys = ["image_id", "model", "stage"]
    expected = pd.MultiIndex.from_tuples(
        [(row["image_id"], model, stage) for row in records
         for model, layers in ERF_STAGE_LAYERS.items() for stage in layers], names=keys,
    )
    actual = pd.MultiIndex.from_frame(metrics[keys])
    if not actual.is_unique or set(actual) != set(expected):
        raise ValueError("Expected exactly one row per selected image/model/stage (8 x N).")
    joined = metrics.merge(selected, on="image_id", suffixes=("", "_selected"),
                           validate="many_to_one")
    for column in ["dataset_index", "label"]:
        if not joined[column].eq(joined[f"{column}_selected"]).all():
            raise ValueError(f"Metrics {column} differs from selected images.")
    for model, layers in ERF_STAGE_LAYERS.items():
        group = metrics[metrics.model == model]
        if not group.layer.eq(group.stage.map(layers)).all():
            raise ValueError(f"Layer/stage mismatch for {model}.")
    for metric in ERF_METRICS:
        values = metrics[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"Invalid or missing {metric} values.")
    # Legacy statistical fixtures may provide only the two primary metrics.
    for metric in ["area50_fraction", "center_energy_fraction", "outer_energy_fraction",
                   "centroid_y", "centroid_x"]:
        if metric in metrics:
            values = metrics[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
                raise ValueError(f"Invalid {metric} values.")
    if "area50_fraction" in metrics and (metrics.area50_fraction > metrics.area90_fraction + 1e-12).any():
        raise ValueError("A50 must not exceed A90.")
    if "raw_gradient_sum" in metrics:
        values = metrics.raw_gradient_sum.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Gradient sums must be finite and positive.")
    if {"center_energy_fraction", "outer_energy_fraction"}.issubset(metrics):
        if not np.allclose(metrics.center_energy_fraction + metrics.outer_energy_fraction, 1, atol=1e-10):
            raise ValueError("Central and outer gradient mass must sum to one.")
    return metrics


def load_completed_erf_run(directory, records, preprocessing, seed=42, *,
                           target_mode="energy", position="center"):
    """Read a completed run only if its selection, checksum and settings match."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        metadata_path = directory / "run.json"  # Legacy official runs.
    metadata = json.loads(metadata_path.read_text("utf-8"))
    if (metadata.get("status") != "complete"
            or metadata.get("num_images") != len(records)
            or metadata.get("preprocessing") != preprocessing
            or metadata.get("seed") != seed):
        raise ValueError(f"Incomplete run or incompatible settings: {directory}")
    selection_path = directory / "selected_images.csv"
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != metadata.get("selected_images_sha256"):
        raise ValueError("Saved selection checksum does not match metadata.")
    columns = ["image_id", "dataset_index", "label"]
    saved = pd.read_csv(selection_path)[columns].sort_values("image_id").reset_index(drop=True)
    wanted = pd.DataFrame(records)[columns].sort_values("image_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, wanted, check_dtype=False)
    metrics = pd.read_csv(directory / "metrics.csv")
    validate_erf_metrics(metrics, records)
    if metadata.get("target_mode", "energy") != target_mode or metadata.get("position", "center") != position:
        raise ValueError("Run uses a different target or spatial position.")
    for column, expected_value in [("condition", preprocessing), ("target_mode", target_mode),
                                    ("position", position), ("experiment", "erf")]:
        if column in metrics and not metrics[column].eq(expected_value).all():
            raise ValueError(f"Mixed or incompatible {column} in metrics.")
    if "model_name" in metrics and not metrics.model_name.eq(metrics.model).all():
        raise ValueError("model_name and model disagree.")
    for name, checksum in metadata.get("artifact_sha256", {}).items():
        if Path(name).name != name:
            raise ValueError("Artifact checksum entries must be filenames.")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return metrics, metadata


def validate_erf_run_pair(official_metadata, shared_metadata):
    """The only intended experimental change is preprocessing."""
    for key in ["target", "center_rule", "gradient_coordinates", "layers", "seed", "num_images",
                "selection_rule", "cohort_sha256_lf"]:
        if key not in official_metadata or official_metadata[key] != shared_metadata.get(key):
            raise ValueError(f"Runs differ in {key}; cannot attribute changes to preprocessing.")
    for model in ERF_STAGE_LAYERS:
        before = official_metadata["models"][model]
        after = shared_metadata["models"][model]
        if before["weights"] != after["weights"]:
            raise ValueError(f"Model weights differ: {model}")
    shared_models = shared_metadata["models"]
    if shared_models["resnet50"]["transform"] != shared_models["vit_b_16"]["transform"]:
        raise ValueError("shared_vit_crop must use identical transforms for both models.")
    if official_metadata["models"]["vit_b_16"]["transform"] != shared_models["vit_b_16"]["transform"]:
        raise ValueError("ViT preprocessing should be unchanged between the two runs.")


def paired_erf_differences(metrics, records, preprocessing):
    """Align by image ID and stage, independent of CSV/loader row order."""
    validate_erf_metrics(metrics, records)
    frames = []
    for metric, (scale, unit) in ERF_METRICS.items():
        paired = metrics.pivot(index=["image_id", "stage"], columns="model", values=metric)
        paired = paired.sort_index().reset_index()
        paired = paired.merge(pd.DataFrame(records)[["image_id", "dataset_index", "label"]],
                              on="image_id", validate="many_to_one")
        paired["difference"] = (paired["vit_b_16"] - paired["resnet50"]) * scale
        paired["metric"], paired["unit"], paired["preprocessing"] = metric, unit, preprocessing
        frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def bootstrap_mean_ci(differences, seed=42, n_resamples=20000):
    """Resample images/pairs, not the two model columns separately."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires at least two finite paired differences.")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1000:
        raise ValueError("Use at least 1,000 bootstrap resamples.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):
        stop = min(start + 1000, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def holm_adjust(pvalues):
    """Holm family-wise error correction, preserving the original row order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("P-values must be finite numbers in [0, 1].")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def paired_erf_tests(pairs, seed=42, n_resamples=20000):
    """Two-sided signed-rank tests; bootstrap mean CIs are separate estimates."""
    rows = []
    for group_index, ((mode, metric, stage), group) in enumerate(
        pairs.groupby(["preprocessing", "metric", "stage"], sort=True)
    ):
        group = group.sort_values("image_id")
        if not group.image_id.is_unique:
            raise ValueError("Duplicate image in a paired comparison.")
        delta = group.difference.to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(delta, seed + group_index, n_resamples)
        # Round subtraction noise consistently; zeros are excluded by Wilcoxon.
        signed_rank_delta = np.round(delta, 12)
        nonzero = np.count_nonzero(signed_rank_delta)
        if nonzero == 0:
            statistic, pvalue = 0.0, 1.0
        else:
            result = wilcoxon(signed_rank_delta, zero_method="wilcox",
                              alternative="two-sided", method="auto")
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        rows.append({
            "preprocessing": mode, "metric": metric, "stage": int(stage),
            "resnet_layer": ERF_STAGE_LAYERS["resnet50"][stage],
            "vit_layer": ERF_STAGE_LAYERS["vit_b_16"][stage],
            "n_pairs": len(delta), "n_nonzero_wilcoxon": int(nonzero),
            "unit": group.unit.iloc[0],
            "resnet_mean": group.resnet50.mean() * ERF_METRICS[metric][0],
            "vit_mean": group.vit_b_16.mean() * ERF_METRICS[metric][0],
            "mean_difference": delta.mean(), "median_difference": np.median(delta),
            "sd_difference": delta.std(ddof=1),
            "mean_ci95_low": low, "mean_ci95_high": high,
            "wilcoxon_statistic": statistic, "p_value": pvalue,
        })
    tests = pd.DataFrame(rows)
    if tests.empty:
        raise ValueError("No paired comparisons supplied.")
    # One family: 2 preprocessing modes x 2 metrics x 4 relative stages = 16.
    tests["p_holm"] = holm_adjust(tests.p_value)
    tests["reject_holm_0_05"] = tests.p_holm < 0.05
    return tests


def erf_crop_sensitivity(pairs, seed=42, n_resamples=20000):
    """Descriptive CI for change in the ViT-minus-ResNet gap on the same images."""
    wide = pairs.pivot(index=["image_id", "metric", "stage", "unit"],
                       columns="preprocessing", values="difference")
    if set(wide.columns) != {"official", "shared_vit_crop"} or wide.isna().any().any():
        raise ValueError("Both preprocessing modes must contain exactly the same image pairs.")
    wide["gap_change"] = wide.shared_vit_crop - wide.official
    rows = []
    for index, ((metric, stage, unit), group) in enumerate(wide.reset_index().groupby(
        ["metric", "stage", "unit"], sort=True
    )):
        group = group.sort_values("image_id")
        delta = group.gap_change.to_numpy()
        low, high = bootstrap_mean_ci(delta, seed + index, n_resamples)
        rows.append({"metric": metric, "stage": int(stage), "unit": unit,
                     "n_pairs": len(delta), "official_gap_mean": group.official.mean(),
                     "shared_gap_mean": group.shared_vit_crop.mean(),
                     "gap_change_mean": delta.mean(), "mean_ci95_low": low,
                     "mean_ci95_high": high})
    return pd.DataFrame(rows)
