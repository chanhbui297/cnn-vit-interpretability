"""Audit saved primary runs, create report artifacts, and run bounded ERF probes.

Run from the repository root with ``python -m experiments.erf.erf_workflow``.
Analysis never runs inference or edits its input runs. New outputs use new IDs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .erf_experiment import (
    LAYERS, erf_code_provenance, erf_output_paths, probe_description,
    select_main_records, select_sensitivity_records, run_experiment,
)
from .erf_statistics import (
    ERF_METRICS, bootstrap_mean_ci, erf_crop_sensitivity, load_completed_erf_run,
    paired_erf_differences, paired_erf_tests, validate_erf_run_pair,
)

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_RUNS = {
    "official": "main_2000_official_20261103T160959_331099Z",
    "shared_vit_crop": "main_2000_shared_vit_crop_20261103T161602_476613Z",
    "paired": "paired_2000_official_vs_shared_vit_crop_20261103T162207_281699Z",
}
SENSITIVITY_PROBES = {"absolute_center": ("absolute", "center"),
                      "energy_quarter": ("energy", "quarter")}
IDENTITY = ["image_id", "dataset_index", "label"]
GRADIENT_COORDINATES = "RGB [0,1] AFTER resize/crop, BEFORE normalization"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def new_run_id(prefix):
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}"


def default_run_dirs(project_root):
    base = Path(project_root) / "results/metrics/experiments/erf"
    return {name: base / run_id / "final" for name, run_id in PRIMARY_RUNS.items()}


def assert_same_table(actual, expected, keys, name):
    """Compare saved and recomputed rows by identity, including all saved columns."""
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"Duplicate comparison rows: {name}")
    if not set(expected.columns).issubset(actual.columns):
        raise ValueError(f"Missing columns in {name}: {set(expected) - set(actual)}")
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"Different row counts in {name}")
    for column in right:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            atol = 1e-320 if column in {"p_value", "p_holm"} else 1e-10
            if not np.allclose(a, b, rtol=1e-8, atol=atol, equal_nan=True):
                raise ValueError(f"Saved/recomputed values differ: {name}/{column}")
        elif not a.fillna("").astype(str).eq(b.fillna("").astype(str)).all():
            raise ValueError(f"Saved/recomputed values differ: {name}/{column}")


def validate_probe_run(frame, metadata, records, project_root, target_mode="energy", position="center"):
    """Require the fixed checkpoints and a single, fully documented probe."""
    from .erf_experiment import probe_position
    required = {"raw_gradient_sum", "target_value", "area50_fraction", "centroid_y", "centroid_x",
                "center_energy_fraction", "outer_energy_fraction", "grid_row", "grid_col",
                "grid_height", "grid_width", "token_index"}
    if not required.issubset(frame):
        raise ValueError(f"Missing research metric columns: {required - set(frame)}")
    config = read_json(Path(project_root) / "configs/analysis_set.json")
    if metadata.get("manifest_sha256") != config["manifest_sha256"]:
        raise ValueError("Source manifest differs from the shared config.")
    expected_probe = probe_description(target_mode, position)
    for key in ["target", "center_rule"]:
        if metadata.get(key) != expected_probe[key]:
            raise ValueError(f"Different probe definition: {key}")
    if metadata.get("gradient_coordinates") != GRADIENT_COORDINATES or metadata.get("layers") != LAYERS:
        raise ValueError("Gradient coordinates or layers differ from the protocol.")
    if metadata.get("exclusions") != []:
        raise ValueError("This study does not allow unreported/cohort-changing exclusions.")
    for model, weights in {"resnet50": "ResNet50_Weights.IMAGENET1K_V2",
                           "vit_b_16": "ViT_B_16_Weights.IMAGENET1K_V1"}.items():
        model_metadata = metadata["models"][model]
        if model_metadata.get("weights") != weights:
            raise ValueError(f"Wrong checkpoint for {model}")
        expected_resize = 256 if metadata["preprocessing"] == "shared_vit_crop" or model == "vit_b_16" else 232
        transform = model_metadata["transform"]
        if f"resize_size=[{expected_resize}]" not in transform or "crop_size=[224]" not in transform:
            raise ValueError(f"Wrong applied crop for {model}")
        if "interpolation=InterpolationMode.BILINEAR" not in transform:
            raise ValueError("Expected bilinear preprocessing.")
        if "mean=[0.485, 0.456, 0.406]" not in transform or "std=[0.229, 0.224, 0.225]" not in transform:
            raise ValueError("Wrong applied normalization.")
        # Historical shared-crop metadata records the applied transform in repr;
        # baseline_loader describes each checkpoint's own original transform.
        preprocessing = model_metadata.get("baseline_loader", {}).get("preprocessing", {})
        if (preprocessing.get("mean") != [0.485, 0.456, 0.406]
                or preprocessing.get("std") != [0.229, 0.224, 0.225]
                or preprocessing.get("antialias") is not True):
            raise ValueError("Missing/incompatible normalization or antialias metadata.")
        group = frame[frame.model == model]
        grids = ([56, 28, 14, 7] if model == "resnet50" else [14] * 4)
        for stage, grid in enumerate(grids, 1):
            rows = group[group.stage == stage]
            row, col = probe_position(grid, grid, position)
            token = -1 if model == "resnet50" else 1 + row * grid + col
            for column, expected in [("grid_height", grid), ("grid_width", grid),
                                      ("grid_row", row), ("grid_col", col), ("token_index", token)]:
                if not rows[column].eq(expected).all():
                    raise ValueError(f"Wrong feature location: {model}/{stage}/{column}")
    if not np.isfinite(frame.target_value).all():
        raise ValueError("Non-finite feature scalar.")


def plot_paired_results(tests, destination):
    """Full depth profiles plus a dedicated final-stage zoom for both metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, (metric, (_, unit)) in enumerate(ERF_METRICS.items()):
        for row in range(2):
            axis = axes[row, col]
            for mode, offset, color in [("official", -0.08, "tab:blue"),
                                         ("shared_vit_crop", 0.08, "tab:orange")]:
                group = tests[(tests.metric == metric) & (tests.preprocessing == mode)].sort_values("stage")
                if row:
                    group = group[group.stage == 4]
                y = group.stage.to_numpy() + offset
                axis.hlines(y, group.mean_ci95_low, group.mean_ci95_high, color=color, linewidth=2)
                axis.scatter(group.mean_difference, y, color=color, label=mode, zorder=3)
            axis.axvline(0, color="gray", linestyle="--", linewidth=1)
            axis.set(xlabel=f"ViT - ResNet ({unit})", ylabel="Relative stage",
                     yticks=[4] if row else [1, 2, 3, 4],
                     title=f"{metric}" + (" - final-stage zoom" if row else ""))
            axis.grid(alpha=0.2)
            axis.legend()
    fig.suptitle(f"{int(tests.n_pairs.iloc[0]):,} image pairs; pointwise 95% bootstrap CIs of mean gaps")
    fig.savefig(destination, dpi=170)
    return fig


def plot_pair_distributions(pairs, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, mode in enumerate(["official", "shared_vit_crop"]):
        for col, (metric, (_, unit)) in enumerate(ERF_METRICS.items()):
            axis = axes[row, col]
            for stage in range(1, 5):
                values = np.sort(pairs.loc[(pairs.preprocessing == mode) & (pairs.metric == metric)
                                           & (pairs.stage == stage), "difference"].to_numpy())
                axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"Stage {stage}")
            axis.axvline(0, color="gray", linestyle="--", linewidth=1)
            axis.set(title=f"{mode}: {metric}", xlabel=f"ViT - ResNet ({unit})", ylabel="Empirical cumulative fraction")
            axis.grid(alpha=0.2)
            axis.legend()
    fig.suptitle("Across-image variation and exceptions; each curve uses paired images")
    fig.savefig(destination, dpi=170)
    return fig


def describe_pairs(pairs):
    rows = []
    for (mode, metric, stage), group in pairs.groupby(["preprocessing", "metric", "stage"], sort=True):
        delta = group.difference
        rows.append({"preprocessing": mode, "metric": metric, "stage": stage,
                     "n_pairs": len(group), "fraction_positive": (delta > 0).mean(),
                     "fraction_negative": (delta < 0).mean(), "fraction_zero": (delta == 0).mean(),
                     "q05": delta.quantile(.05), "q25": delta.quantile(.25),
                     "median": delta.median(), "q75": delta.quantile(.75), "q95": delta.quantile(.95),
                     "mean": delta.mean(), "unit": group.unit.iloc[0]})
    return pd.DataFrame(rows)


def handoff_cases(shared_metrics, shared_metadata, project_root):
    """Choose typical/extreme examples reproducibly; retain original random previews."""
    wide = shared_metrics[shared_metrics.stage == 4].pivot(index="image_id", columns="model", values="area90_fraction")
    gap = wide.vit_b_16 - wide.resnet50
    roles = [("typical_final_stage", (gap - gap.median()).abs().sort_index().idxmin()),
             ("largest_vit_minus_resnet_gap", gap.sort_index().idxmax()),
             ("smallest_vit_minus_resnet_gap", gap.sort_index().idxmin())]
    roles += [("original_random_preview", image_id) for image_id in shared_metadata.get("preview_ids", [])]
    identity = shared_metrics.drop_duplicates("image_id").set_index("image_id")
    preview_order = [key for key in identity.index if key in shared_metadata.get("preview_ids", [])]
    rows = []
    for role, image_id in roles:
        record = identity.loc[image_id]
        figure = f"image_{preview_order.index(image_id):02d}.png" if image_id in preview_order else ""
        for model in LAYERS:
            rows.append({"image_id": image_id, "dataset_index": int(record.dataset_index), "label": int(record.label),
                         "role": role, "model_name": model, "condition": "shared_vit_crop",
                         "gradcam_target_class": int(record.label), "erf_target": "energy/center (feature, not class score)",
                         "resize_short_edge": 256, "crop_height": 224, "crop_width": 224,
                         "existing_erf_figure": figure, "crop_sha256": "",
                         "erf_figure_run_id": shared_metadata.get("run_id", ""),
                         "attention_observation": "", "gradcam_observation": "",
                         "integration_status": "awaiting_team_results"})
    result = pd.DataFrame(rows)
    for model in LAYERS:
        path = Path(project_root) / f"results/metrics/baseline_notebook/full/{model}_predictions.csv"
        if path.is_file():
            baseline = pd.read_csv(path)
            if not baseline.image_id.is_unique:
                raise ValueError("Duplicate IDs in baseline predictions.")
            subset = result[result.model_name == model][["image_id", "label"]]
            check = subset.merge(baseline[["image_id", "label"]], on="image_id", how="left", suffixes=("", "_baseline"))
            if not check.label.eq(check.label_baseline).all():
                raise ValueError("Handoff labels differ from baseline.")
            predictions = baseline.set_index("image_id")
            mask = result.model_name == model
            # These are OFFICIAL baseline predictions, never mislabeled as shared-crop predictions.
            result.loc[mask, "official_pred_top1"] = result.loc[mask, "image_id"].map(predictions.pred_top1)
            result.loc[mask, "official_top1_correct"] = result.loc[mask, "image_id"].map(predictions.top1_correct)
    for column in ["official_pred_top1", "official_top1_correct"]:
        if column in result:
            result[column] = result[column].astype("Int64")
    return result


def audit_main_runs(project_root, official_dir, shared_dir, paired_dir, *, seed=42, n_resamples=20000):
    """Read-only audit; no output directories are created until every check passes."""
    records = select_main_records(project_root, seed)
    official, om = load_completed_erf_run(official_dir, records, "official", seed)
    shared, sm = load_completed_erf_run(shared_dir, records, "shared_vit_crop", seed)
    for frame, metadata in [(official, om), (shared, sm)]:
        validate_probe_run(frame, metadata, records, project_root)
    validate_erf_run_pair(om, sm)
    # ViT has identical crop/checkpoint/probe in both conditions: a strong negative control.
    numerical = [c for c in official.select_dtypes(include="number") if c in shared]
    assert_same_table(shared[shared.model == "vit_b_16"][["image_id", "layer", *numerical]],
                      official[official.model == "vit_b_16"][["image_id", "layer", *numerical]],
                      ["image_id", "layer"], "ViT unchanged across crops")
    pairs = pd.concat([paired_erf_differences(official, records, "official"),
                       paired_erf_differences(shared, records, "shared_vit_crop")], ignore_index=True)
    tests = paired_erf_tests(pairs, seed, n_resamples)
    crop = erf_crop_sensitivity(pairs, seed, n_resamples)
    result = {"records": records, "official": official, "shared": shared, "pairs": pairs,
              "tests": tests, "crop": crop, "official_metadata": om, "shared_metadata": sm,
              "checks": ["fixed balanced cohort, IDs and labels", "selection checksums",
                         "16,000 rows per condition and valid gradients/metrics", "fixed checkpoints and transforms",
                         "unchanged ViT negative control"]}
    if paired_dir is None:
        return result  # New main runs have no historical paired table to reproduce.
    paired_dir = Path(paired_dir)
    pm = read_json(paired_dir / "metadata.json")
    if (pm.get("status") != "complete" or pm.get("num_images") != 2000 or pm.get("seed") != seed
            or pm.get("bootstrap_resamples") != n_resamples):
        raise ValueError("Historical paired-analysis metadata is incompatible.")
    cohort = pd.DataFrame(records)[IDENTITY]
    assert_same_table(pd.read_csv(paired_dir / "selected_images.csv"), cohort, ["image_id"], "paired cohort")
    normalized = (paired_dir / "selected_images.csv").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() != pm.get("cohort_sha256_lf"):
        raise ValueError("Paired cohort checksum mismatch.")
    for mode, frame, directory in [("official", official, Path(official_dir)),
                                    ("shared_vit_crop", shared, Path(shared_dir))]:
        if pm["runs"][mode]["metrics_sha256"] != sha256(directory / "metrics.csv"):
            raise ValueError(f"Original metric bytes differ from paired provenance: {mode}")
        saved = pd.read_csv(paired_dir / f"{mode}_metrics.csv")
        assert_same_table(saved, frame, ["image_id", "model", "stage"], f"paired copy of {mode}")
        # Recompute each original descriptive summary, including N and SD.
        columns = ["area50_fraction", "area90_fraction", "dispersion_normalized",
                   "center_energy_fraction", "outer_energy_fraction"]
        summary = frame.groupby(["model", "stage", "layer"])[columns].agg(["mean", "std", "count"])
        old_summary = pd.read_csv(directory / "summary.csv", header=[0, 1], index_col=[0, 1, 2])
        pd.testing.assert_frame_equal(old_summary, summary, check_dtype=False, atol=1e-10, rtol=1e-8)
    assert_same_table(pd.read_csv(paired_dir / "paired_differences.csv"), pairs,
                      ["preprocessing", "metric", "stage", "image_id"], "paired differences")
    assert_same_table(pd.read_csv(paired_dir / "paired_tests.csv"), tests,
                      ["preprocessing", "metric", "stage"], "16 paired tests")
    assert_same_table(pd.read_csv(paired_dir / "crop_sensitivity.csv"), crop,
                      ["metric", "stage"], "crop sensitivity")
    result["checks"].extend(["source metrics SHA-256 from historical paired metadata",
                             "original summary mean/SD/N", "paired metric copies and per-image differences",
                             "all 16 Wilcoxon/Holm/bootstrap rows", "8 crop-sensitivity rows"])
    return result


def export_main_analysis(project_root, official_dir, shared_dir, paired_dir, *, run_id=None,
                         seed=42, n_resamples=20000):
    result = audit_main_runs(project_root, official_dir, shared_dir, paired_dir, seed=seed, n_resamples=n_resamples)
    run_id = run_id or new_run_id("validated_2000")
    output, figures, _ = erf_output_paths(project_root, run_id)
    if output.exists() or figures.exists():
        raise FileExistsError("Choose a new analysis run ID; historical results are never overwritten.")
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    metadata = {"experiment": "erf", "run_id": run_id, "status": "running", "seed": seed,
                "num_images": 2000, "bootstrap_resamples": n_resamples,
                "sensitivity_status": "pending_execution", "cross_method_status": "awaiting_team_results",
                "code": erf_code_provenance(project_root),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    write_json(output / "metadata.json", metadata)
    try:
        pd.DataFrame(result["records"]).to_csv(output / "selected_images.csv", index=False)
        sensitivity = select_sensitivity_records(result["records"], seed)
        pd.DataFrame(sensitivity).to_csv(output / "sensitivity_selected_images.csv", index=False)
        for name, key in [("official_metrics", "official"), ("shared_vit_crop_metrics", "shared"),
                           ("paired_differences", "pairs"), ("paired_tests", "tests"), ("crop_sensitivity", "crop")]:
            result[key].to_csv(output / f"{name}.csv", index=False)
        describe_pairs(result["pairs"]).to_csv(output / "paired_distribution_summary.csv", index=False)
        handoff_cases(result["shared"], result["shared_metadata"], project_root).to_csv(output / "cross_method_handoff.csv", index=False)
        for function, data, name in [(plot_paired_results, result["tests"], "paired_mean_ci95.png"),
                                      (plot_pair_distributions, result["pairs"], "paired_distributions.png")]:
            fig = function(data, figures / name)
            plt.close(fig)
        sources = {}
        for mode, directory in [("official", official_dir), ("shared_vit_crop", shared_dir), ("paired", paired_dir)]:
            if directory is None:
                continue
            sources[mode] = {"path": str(directory), "files": {
                path.name: sha256(path) for path in sorted(Path(directory).iterdir()) if path.is_file()}}
        audit = {"status": "passed", "checks": result["checks"], "sources": sources,
                 "limitations": ["Historical run metadata records a dirty working tree; the executed notebook itself was not hashed.",
                                 "Raw maps were saved only for previews; gradients for every image are not independently recomputed.",
                                 "Image-pair bootstrap does not preserve per-class quotas; CIs are pointwise and not multiplicity-adjusted."]}
        write_json(output / "audit.json", audit)
        metadata.update(status="complete", validation="recomputed_from_original_csvs" if paired_dir else "computed_from_new_run_csvs", sources=sources,
                        cohort_sha256_lf=hashlib.sha256((output / "selected_images.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                        figures_dir=str(figures), artifact_sha256={p.name: sha256(p) for p in output.glob("*.csv")})
        write_json(output / "metadata.json", metadata)
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        write_json(output / "metadata.json", metadata)
        raise
    return output, figures, result


def run_sensitivity(project_root, data_dir, shared_dir, *, existing=None, device=None, allow_download=False):
    """Reuse baseline metrics on the same 200 IDs; run only the two new probes."""
    import torch
    main_records = select_main_records(project_root)
    baseline, metadata = load_completed_erf_run(shared_dir, main_records, "shared_vit_crop")
    validate_probe_run(baseline, metadata, main_records, project_root)
    records = select_sensitivity_records(main_records)
    results = {}
    for name, (target_mode, position) in SENSITIVITY_PROBES.items():
        directory = (existing or {}).get(name)
        if directory is None:
            directory = erf_output_paths(project_root, new_run_id(f"sensitivity_200_{name}"))[0]
            run_experiment(records, data_dir, directory,
                           device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           preprocessing="shared_vit_crop", project_root=project_root,
                           allow_download=allow_download, target_mode=target_mode, position=position)
        results[name] = Path(directory)
    return results


def export_sensitivity(project_root, shared_dir, probe_dirs, *, run_id=None, n_resamples=20000):
    records_main = select_main_records(project_root)
    records = select_sensitivity_records(records_main)
    baseline, bm = load_completed_erf_run(shared_dir, records_main, "shared_vit_crop")
    validate_probe_run(baseline, bm, records_main, project_root)
    ids = {row["image_id"] for row in records}
    baseline = baseline[baseline.image_id.isin(ids)].copy()
    base_pairs = paired_erf_differences(baseline, records, "shared_vit_crop")
    baseline_tables = []
    rows, provenance = [], {}
    for index, (name, (target_mode, position)) in enumerate(SENSITIVITY_PROBES.items()):
        changed, cm = load_completed_erf_run(probe_dirs[name], records, "shared_vit_crop",
                                             target_mode=target_mode, position=position)
        validate_probe_run(changed, cm, records, project_root, target_mode, position)
        for key in ["gradient_coordinates", "layers", "manifest_sha256", "seed"]:
            if bm[key] != cm[key]:
                raise ValueError(f"Sensitivity runs differ in {key}")
        for model in LAYERS:
            if bm["models"][model]["weights"] != cm["models"][model]["weights"] or bm["models"][model]["transform"] != cm["models"][model]["transform"]:
                raise ValueError("Sensitivity controls changed weights/preprocessing.")
        pairs = paired_erf_differences(changed, records, "shared_vit_crop")
        joined = base_pairs.merge(pairs, on=["image_id", "metric", "stage", "unit"],
                                   suffixes=("_baseline", "_changed"), validate="one_to_one").sort_values(["metric", "stage", "image_id"])
        joined["probe"] = name
        joined["gap_change"] = joined.difference_changed - joined.difference_baseline
        baseline_tables.append(joined)
        for j, ((metric, stage, unit), group) in enumerate(joined.groupby(["metric", "stage", "unit"], sort=True)):
            low, high = bootstrap_mean_ci(group.gap_change, 42 + 100 * index + j, n_resamples)
            changed_low, changed_high = bootstrap_mean_ci(group.difference_changed, 42 + 100 * index + j, n_resamples)
            rows.append({"probe": name, "metric": metric, "stage": stage, "unit": unit,
                         "n_pairs": len(group), "baseline_gap_mean": group.difference_baseline.mean(),
                         "changed_gap_mean": group.difference_changed.mean(), "gap_change_mean": group.gap_change.mean(),
                         "mean_ci95_low": low, "mean_ci95_high": high,
                         "changed_gap_ci95_low": changed_low, "changed_gap_ci95_high": changed_high,
                         "resnet_metric_change_mean": ((group.resnet50_changed - group.resnet50_baseline) * ERF_METRICS[metric][0]).mean(),
                         "vit_metric_change_mean": ((group.vit_b_16_changed - group.vit_b_16_baseline) * ERF_METRICS[metric][0]).mean(),
                         "mean_sign_changed": bool(np.sign(group.difference_baseline.mean()) != np.sign(group.difference_changed.mean()))})
        provenance[name] = {"path": str(probe_dirs[name]), "metrics_sha256": sha256(Path(probe_dirs[name]) / "metrics.csv")}
    output, figures, _ = erf_output_paths(project_root, run_id or new_run_id("sensitivity_analysis_200"))
    if output.exists() or figures.exists():
        raise FileExistsError("Sensitivity analysis run already exists.")
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    pd.DataFrame(records).to_csv(output / "selected_images.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "sensitivity_summary.csv", index=False)
    pd.concat(baseline_tables, ignore_index=True).to_csv(output / "sensitivity_paired_differences.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (metric, (_, unit)) in zip(axes, ERF_METRICS.items()):
        for name, offset in [("absolute_center", -.08), ("energy_quarter", .08)]:
            g = table[(table.probe == name) & (table.metric == metric)]
            y = g.stage + offset
            axis.hlines(y, g.mean_ci95_low, g.mean_ci95_high)
            axis.scatter(g.gap_change_mean, y, label=name)
        axis.axvline(0, color="gray", linestyle="--")
        axis.set(xlabel=f"Change in ViT - ResNet gap ({unit})", ylabel="Relative stage", yticks=[1, 2, 3, 4])
        axis.legend()
    fig.suptitle("200 fixed images; descriptive pointwise 95% CIs; separate from the primary test family")
    fig.savefig(figures / "sensitivity_gap_change.png", dpi=170)
    plt.close(fig)
    write_json(output / "metadata.json", {"status": "complete", "experiment": "erf_sensitivity", "seed": 42,
               "num_images": 200, "bootstrap_resamples": n_resamples, "sources": provenance,
               "baseline": {"path": str(shared_dir), "metrics_sha256": sha256(Path(shared_dir) / "metrics.csv")},
               "selection": "random.Random(42).sample(ID-sorted primary 2000 cohort, 200); no correctness filtering",
               "inference": "descriptive paired-image bootstrap; no new p-values or changes to the original Holm family",
               "position_limit": "Quarter-grid positions are not exactly aligned in original-input receptive centers.",
               "artifact_sha256": {p.name: sha256(p) for p in output.glob("*.csv")},
               "code": erf_code_provenance(project_root)})
    return output, figures, table


def write_report(project_root, analysis_dir, *, sensitivity_dir=None):
    """Write the reviewable ERF chapter from audited CSVs; pending work stays explicit."""
    root, directory = Path(project_root), Path(analysis_dir)
    metadata = read_json(directory / "metadata.json")
    audit = read_json(directory / "audit.json")
    if metadata.get("status") != "complete" or audit.get("status") != "passed":
        raise ValueError("Report generation requires a completed, audited analysis.")
    for name, checksum in metadata["artifact_sha256"].items():
        if sha256(directory / name) != checksum:
            raise ValueError(f"Analysis artifact changed after audit: {name}")
    tests = pd.read_csv(directory / "paired_tests.csv")
    crop = pd.read_csv(directory / "crop_sensitivity.csv")
    shared = pd.read_csv(directory / "shared_vit_crop_metrics.csv")
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)
    def link(path):
        return os.path.relpath(path, report_dir).replace(os.sep, "/")
    def p_text(value):
        return "<1e-300 (underflow)" if value == 0 else f"{value:.3g}"
    def metric_table(metric):
        lines = ["| Điều kiện | Stage | ResNet | ViT | ViT − ResNet | CI95% của mean gap | p Holm |",
                 "|---|---:|---:|---:|---:|---|---:|"]
        for _, row in tests[tests.metric == metric].iterrows():
            lines.append(f"| {row.preprocessing} | {int(row.stage)} | {row.resnet_mean:.4f} | {row.vit_mean:.4f} | "
                         f"{row.mean_difference:+.4f} | [{row.mean_ci95_low:+.4f}; {row.mean_ci95_high:+.4f}] | {p_text(row.p_holm)} |")
        return "\n".join(lines)
    final = tests[(tests.preprocessing == "shared_vit_crop") & (tests.metric == "area90_fraction") & (tests.stage == 4)].iloc[0]
    dispersion = tests[(tests.preprocessing == "shared_vit_crop") & (tests.metric == "dispersion_normalized") & (tests.stage == 4)].iloc[0]
    crop_final = crop[(crop.metric == "area90_fraction") & (crop.stage == 4)].iloc[0]
    a50 = shared[shared.stage == 4].groupby("model").area50_fraction.mean() * 100
    source_lines = "\n".join(f"- {mode}: `{Path(item['path']).parent.name}`; file SHA-256 trong audit.json."
                             for mode, item in metadata["sources"].items())
    sensitivity = ("**Chưa chạy trên ImageNet.** Hai probe đã có code/notebook và kiểm tra synthetic; "
                   "không có số liệu độ nhạy để kết luận target/vị trí ổn định.\n\n"
                   "Subset 200 ID chọn từ cohort 2.000 bằng random.Random(42), ID-sort trước sample, "
                   "không lọc đúng/sai. Chỉ shared crop. `absolute_center` đổi mean(f²) thành mean(abs(f)); "
                   "`energy_quarter` giữ mean(f²), đổi mỗi chỉ số thành round(0.25×(H−1)). "
                   "Vị trí quarter vẫn là tọa độ lưới, chưa căn chỉnh chính xác receptive center. "
                   "So với baseline trên chính 200 ID bằng thay đổi gap và CI mô tả; không sửa họ 16 kiểm định chính.")
    if sensitivity_dir is not None:
        sd = Path(sensitivity_dir)
        sm = read_json(sd / "metadata.json")
        if sm.get("status") != "complete" or sm.get("num_images") != 200:
            raise ValueError("Sensitivity report requires a completed 200-image analysis.")
        for name, checksum in sm.get("artifact_sha256", {}).items():
            if sha256(sd / name) != checksum:
                raise ValueError(f"Sensitivity artifact changed: {name}")
        table = pd.read_csv(sd / "sensitivity_summary.csv")
        audit_path = sd / "audit.json"
        validated = audit_path.is_file() and read_json(audit_path).get("status") == "passed"
        if validated and sm.get("audit_sha256") != sha256(audit_path):
            raise ValueError("Sensitivity audit checksum mismatch.")
        expected_keys = {(probe, metric, stage) for probe in SENSITIVITY_PROBES
                         for metric in ERF_METRICS for stage in range(1, 5)}
        if (len(table) != 16 or table.duplicated(["probe", "metric", "stage"]).any()
                or set(table[["probe", "metric", "stage"]].itertuples(index=False, name=None)) != expected_keys
                or not table.n_pairs.eq(200).all()):
            raise ValueError("Expected 16 sensitivity comparisons, each with 200 pairs.")
        def sensitivity_row(probe, metric, stage):
            return table[(table.probe == probe) & (table.metric == metric) & (table.stage == stage)].iloc[0]
        absolute3 = sensitivity_row("absolute_center", "area90_fraction", 3)
        absolute4 = sensitivity_row("absolute_center", "area90_fraction", 4)
        quarter4 = sensitivity_row("energy_quarter", "area90_fraction", 4)
        class_count = pd.read_csv(sd / "selected_images.csv").label.nunique()
        early = table[table.stage.le(3)]
        direction_text = ("Cả hai probe giữ mean gap dương ở stage 1–3 cho A90 và dispersion; "
                          "các CI mô tả của changed gap ở những stage này đều nằm trên 0."
                          if early.changed_gap_ci95_low.gt(0).all() else
                          "Không phải mọi probe/metric ở stage 1–3 đều có CI changed gap nằm trên 0; xem bảng từng điều kiện.")
        lines = [f"Đã hoàn tất{' và kiểm chứng độc lập từ CSV nguồn' if validated else ''} `{sd.parent.name}`.",
                 f"Cùng **200 ảnh, {class_count} lớp**, lấy từ cohort 2.000 ảnh bằng seed 42; không lọc theo dự đoán.",
                 "CI mô tả, pointwise, chưa hiệu chỉnh đa so sánh; không thêm p-value vào họ Holm chính.", "",
                 f"[Bảng đủ 16 so sánh]({link(sd / 'sensitivity_summary.csv')}) và "
                 f"[dữ liệu từng cặp ảnh]({link(sd / 'sensitivity_paired_differences.csv')}).", "",
                 "| Probe | Metric | Stage | Baseline gap | Changed gap | Gap change [CI95%] |",
                 "|---|---|---:|---:|---:|---|"]
        for _, row in table.iterrows():
            lines.append(f"| {row.probe} | {row.metric} | {int(row.stage)} | {row.baseline_gap_mean:+.4f} | "
                         f"{row.changed_gap_mean:+.4f} | {row.gap_change_mean:+.4f} [{row.mean_ci95_low:+.4f}; {row.mean_ci95_high:+.4f}] |")
        lines += ["", "### Diễn giải độ nhạy", "", direction_text, "",
                  f"**Đổi scalar tại tâm:** A90 gap stage 3 giảm {absolute3.gap_change_mean:.4f} điểm %, "
                  f"CI95% [{absolute3.mean_ci95_low:.4f}; {absolute3.mean_ci95_high:.4f}], còn "
                  f"{absolute3.changed_gap_mean:.4f} điểm %. Mức thay đổi metric riêng ResNet là "
                  f"{absolute3.resnet_metric_change_mean:+.4f} điểm %, ViT {absolute3.vit_metric_change_mean:+.4f} điểm %. "
                  "Hướng khác biệt sớm giữ trên các lựa chọn đã thử, nhưng độ lớn không bất biến theo scalar.", "",
                  f"Ở stage 4 tại tâm, đổi scalar cho mean A90 gap {absolute4.changed_gap_mean:+.4f} điểm %, "
                  f"CI95% [{absolute4.changed_gap_ci95_low:+.4f}; {absolute4.changed_gap_ci95_high:+.4f}]. "
                  "CI này không chứng minh tương đương giữa hai model.", "",
                  f"**Đổi vị trí sang quarter:** cùng scalar mean(f²), A90 gap stage 4 tăng từ "
                  f"{quarter4.baseline_gap_mean:.4f} lên {quarter4.changed_gap_mean:.4f} điểm %, "
                  f"CI95% của gap mới [{quarter4.changed_gap_ci95_low:.4f}; {quarter4.changed_gap_ci95_high:.4f}]. "
                  f"Thay đổi gap là {quarter4.gap_change_mean:+.4f} điểm %, "
                  f"CI95% [{quarter4.mean_ci95_low:.4f}; {quarter4.mean_ci95_high:.4f}]. "
                  f"A90 riêng ResNet thay đổi {quarter4.resnet_metric_change_mean:+.4f} điểm %, "
                  f"ViT {quarter4.vit_metric_change_mean:+.4f} điểm %.", "",
                  "**Kết luận cần giới hạn theo vị trí:** nhận xét A90 tầng cuối gần nhau ở probe tâm "
                  "không áp dụng chung cho mọi vị trí. Quarter thay đổi cả vị trí feature, quan hệ với biên ảnh "
                  "và receptive center theo từng kiến trúc; chưa tách riêng ảnh hưởng của từng yếu tố đó.", "",
                  f"Baseline của subset 200 ảnh có A90 gap stage 4 {quarter4.baseline_gap_mean:+.4f} điểm %, "
                  f"trong khi mean của toàn cohort 2.000 ảnh là {final.mean_difference:+.4f} điểm %. "
                  "Mọi hiệu ứng target/vị trí ở đây đều dùng baseline trên chính 200 ID, không lấy hiệu giữa hai cỡ mẫu.", "",
                  "mean_sign_changed chỉ mô tả dấu của ước lượng điểm; False không chứng minh độ lớn bất biến "
                  f"hoặc hai thiết lập tương đương. Chỉ khảo sát hai scalar và hai vị trí trên {class_count} lớp, chưa suy luận mọi target/vị trí."]
        sensitivity_figures = Path(sm.get("figures_dir", root / "results/figures/experiments/erf" / sd.parent.name / "final"))
        for filename, caption in [("sensitivity_gap_levels.png", "Cùng 200 ảnh: gap của ba probe, có zoom stage 4"),
                                   ("sensitivity_gap_change.png", "Thay đổi gap so với baseline trên cùng 200 ảnh")]:
            path = sensitivity_figures / filename
            if path.is_file():
                lines += ["", f"![{caption}]({link(path)})"]
        sensitivity = "\n".join(lines)
    figure_dir = Path(metadata["figures_dir"])
    heatmap_sections = []
    handoff = pd.read_csv(directory / "cross_method_handoff.csv").fillna("")
    preview = handoff[handoff.existing_erf_figure == "image_00.png"]
    if not preview.empty:
        image_id = preview.image_id.iloc[0]
        for mode in ["official", "shared_vit_crop"]:
            source_run = Path(metadata["sources"][mode]["path"]).parent.name
            image_path = root / "results/figures/experiments/erf" / source_run / "final/image_00.png"
            if image_path.is_file():
                heatmap_sections.append(f"![{mode}: bốn stage của {image_id}]({link(image_path)})")
        if heatmap_sections:
            heatmap_sections.append(f"Ví dụ `{image_id}` thuộc ba preview ngẫu nhiên gốc, không chọn theo độ đẹp của map. "
                                    "Mỗi heatmap dùng log10 khối lượng absolute gradient đã normalize, cùng color scale. "
                                    "Hình dùng crop riêng của từng điều kiện; pixel ảnh gốc không được căn chỉnh ngược qua resize.")
    heatmap_text = "\n\n".join(heatmap_sections)
    content = f"""# ERF: phạm vi và phân bố ảnh hưởng lên feature trung gian

## Trạng thái bằng chứng và câu hỏi nghiên cứu

Phân tích `{directory.parent.name}` được tạo lại từ CSV gốc và vượt qua audit;
không lấy bảng làm tròn trong notebook làm đầu vào thống kê.
Nguồn kiểm chứng: [audit.json]({link(directory / 'audit.json')}) và
[paired_tests.csv]({link(directory / 'paired_tests.csv')}).

ERF bổ sung bằng chứng cho câu hỏi trung tâm của project: hai checkpoint có accuracy
gần nhau nhưng tích hợp thông tin không gian khác nhau thế nào? Phép đo ở đây là
độ nhạy bậc nhất của feature trung gian; không trực tiếp quy kết nguyên nhân dự đoán.

{source_lines}

## Phương pháp

ResNet-50 IMAGENET1K_V2 và ViT-B/16 IMAGENET1K_V1; shared model/data loaders,
eval mode, không dùng inference/no_grad khi tính input gradient.
Chọn 2 trong 10 ảnh mỗi lớp của manifest chung bằng random.Random(42), duyệt lớp
0–999 và ID-sort trong lớp: 2.000 ảnh, 1.000 lớp, cùng cohort cho hai model/hai crop,
không lọc theo dự đoán. Cohort SHA-256 LF: `{metadata['cohort_sha256_lf']}`.

ResNet lấy vector sau layer1/2/3/4; ViT sau block3/6/9/12, trước encoder LN cuối,
không chọn CLS. Vị trí row=H//2, col=W//2; ViT token 106. Scalar z=mean(f²).
ERF E_ij=sum_c |∂z/∂u_cij|, với u là RGB [0,1] sau resize/crop, trước normalization;
chain rule chia gradient normalized input cho std. Không ánh xạ về ảnh gốc.
Stage là độ sâu tương đối; số ô/vị trí receptive center không tương đương chính xác.

A50/A90 là tỷ lệ số pixel ít nhất chứa 50%/90% tổng absolute gradient; pixel có thể
rời nhau. Dispersion là RMS khoảng cách quanh centroid trong tọa độ [0,1].
Central mass dùng ô vuông nửa giữa mỗi chiều (~25% diện tích).
Official giữ ResNet resize 232, ViT resize 256; cùng crop 224. Shared crop dùng
transform ViT resize 256 cho cả hai. Không thay baseline accuracy chính bằng shared crop.

Cỡ mẫu dùng pilot SD A90 gap 11,127155 điểm % và biên lập kế hoạch ±0,5 điểm %:
N≈(1,96×11,127155/0,5)²≈1.903; chọn 2.000 để cân bằng lớp. Đây không phải power
analysis cho Wilcoxon, không bảo đảm độ chính xác cho mọi metric và không đủ suy luận từng lớp.

Mỗi so sánh có 2.000 cặp ảnh. Báo cáo mean gap ViT−ResNet và paired-image percentile
bootstrap CI95%, 20.000 resample, seed 42 cộng chỉ số nhóm. Resample không giữ quota lớp.
Wilcoxon hai phía dùng hiệu số round 12 chữ số, bỏ zero; Holm hiệu chỉnh chung 16 p-value.
CI mean là pointwise, không hiệu chỉnh đa kiểm định. Signed-rank không kiểm định trực tiếp mean.

## Kết quả chính

### Ví dụ ERF qua độ sâu

{heatmap_text}

### A90

ResNet/ViT biểu diễn bằng % pixel; gap và CI bằng **điểm phần trăm**.

{metric_table('area90_fraction')}

### Dispersion

Đơn vị: khoảng cách RMS trong tọa độ chuẩn hóa [0,1].

{metric_table('dispersion_normalized')}

![Mean gaps và CI, có phóng đại tầng cuối]({link(figure_dir / 'paired_mean_ci95.png')})

![Phân bố hiệu số từng ảnh]({link(figure_dir / 'paired_distributions.png')})

### Trả lời câu hỏi của project

1. **ResNet tích lũy ảnh hưởng theo độ sâu:** A90 tăng mạnh từ stage 1 tới stage 4.
2. **ViT có ảnh hưởng phân tán rộng tại các block sớm được đo:** block 3 đã có A90
   lớn hơn ResNet layer1. Cả ba khoảng cách đầu vẫn lớn trong shared crop. Chưa đo
   block 1 nên không phát biểu đã chứng minh hiện tượng ngay block đầu.
3. **A90 gần nhau không có nghĩa phân bố giống nhau:** ở shared crop tầng cuối,
   mean A90 gap={final.mean_difference:+.4f} điểm %, CI95%
   [{final.mean_ci95_low:+.4f}; {final.mean_ci95_high:+.4f}], trong khi dispersion gap
   ={dispersion.mean_difference:+.4f}, CI95%
   [{dispersion.mean_ci95_low:+.4f}; {dispersion.mean_ci95_high:+.4f}].
   A50 tầng cuối shared: ResNet {a50['resnet50']:.2f}%, ViT {a50['vit_b_16']:.2f}%.
   Các metric này phù hợp với khả năng có phần tập trung mạnh cùng ảnh hưởng xa hơn;
   đó là diễn giải phân bố, chưa phải kết quả fit Gaussian hay bằng chứng nhân quả.
4. **Preprocessing ảnh hưởng rõ hơn tới khoảng cách nhỏ tầng cuối:** đổi sang shared crop
   làm A90 gap tầng cuối thay đổi {crop_final.gap_change_mean:+.4f} điểm %,
   CI95% [{crop_final.mean_ci95_low:+.4f}; {crop_final.mean_ci95_high:+.4f}].
   Đây là thay đổi gap trên cùng ảnh, không phải so sánh hai mẫu độc lập.

Ở shared crop tầng cuối, mean CI A90 chứa 0 nhưng p Holm của Wilcoxon là
{p_text(final.p_holm)}. Hai đại lượng kiểm tra/ước lượng khác nhau; không dùng p này
để kết luận mean chắc chắn dương, và không dùng CI chứa 0 để chứng minh tương đương.
Các p-value hiển thị dưới ngưỡng số máy được ghi là underflow, không phải xác suất bằng 0.
ViT không tăng A90 đơn điệu ở mọi block. ERF lớn hơn không đồng nghĩa model tốt hơn
hoặc phân bố ảnh hưởng đều trên toàn ảnh.

## Kiểm tra độ nhạy target và vị trí

{sensitivity}

## Liên hệ Attention/Grad-CAM

**Đang chờ kết quả của thành viên phụ trách.**
[cross_method_handoff.csv]({link(directory / 'cross_method_handoff.csv')}) chứa ID/label,
crop chung, target Grad-CAM đề xuất (ground-truth class), ví dụ điển hình/cực trị và
preview ngẫu nhiên có sẵn. Crop hash và quan sát được để trống có chủ đích đến khi có dữ liệu thật.
Prediction official đi kèm chỉ là tham khảo, không thay prediction shared crop.

Đối chiếu: vùng nhạy của feature có trùng vùng Grad-CAM cho class không? Attention
phân tán có đi kèm gradient phân tán không? Khi khác nhau, target/layer/query khác
nhau giải thích được bao nhiêu? Không tính map correlation trực tiếp khi target và
hệ tọa độ chưa thống nhất. Không suy ra shape bias hoặc robustness từ ERF đơn lẻ.

## Giới hạn và tái lập

- Hai checkpoint có lịch sử huấn luyện khác nhau; shared crop chưa cô lập tác động kiến trúc.
- Một scalar và một vị trí không đại diện toàn bộ Jacobian hay class prediction.
- Cohort cân bằng nhưng chỉ 2 ảnh/lớp; không suy luận riêng từng lớp.
- Ảnh dùng để minh họa metric extremes là case study, không phải mẫu đại diện.
- Metadata gốc ghi dirty tree; notebook thực thi không được hash. Giữ snapshot output
  lịch sử, không tuyên bố đã chứng minh toàn bộ provenance của code thực thi.
- Raw maps gốc chỉ giữ preview. Audit xác minh CSV/metadata và tái tính thống kê;
  không phải chạy lại mọi input gradient trên pretrained model.
- Các CI phụ thuộc cách resample theo ảnh, không giữ quota lớp và chưa hiệu chỉnh multiplicity.

Từ repo root: `python -m experiments.erf.erf_workflow audit`.
Cài `experiments/erf/requirements-reproduce.txt` để dùng đúng NumPy/pandas/SciPy của
run gốc, vì cách đọc float CSV có thể làm thay đổi tie rank rất nhỏ.
Tạo lại bảng/hình bằng lệnh `analyze`; notebook chính nhập cùng các helper đó.

## Tài liệu

- Luo et al. (2016), [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf).
- Raghu et al. (2021), [Do Vision Transformers See Like Convolutional Neural Networks?](https://arxiv.org/abs/2108.08810).
- `docs/experiment_protocol.md`, `docs/erf.md`; package versions và file hashes trong metadata/audit.
"""
    # report/erf.md is hand-written; this generated version is for cross-checking numbers only.
    path = report_dir / "erf_generated_tables.md"
    path.write_text(content, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "analyze", "sensitivity", "sensitivity-analysis", "benchmark"])
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--shared", type=Path)
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--absolute-run", type=Path)
    parser.add_argument("--quarter-run", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--sensitivity-dir", type=Path, help="Completed sensitivity analysis to include with --write-report")
    args = parser.parse_args()
    directories = default_run_dirs(args.project_root)
    official = args.official or directories["official"]
    shared = args.shared or directories["shared_vit_crop"]
    paired = args.paired or directories["paired"]
    if args.command == "audit":
        result = audit_main_runs(args.project_root, official, shared, paired)
        print(json.dumps({"status": "passed", "checks": result["checks"]}, indent=2))
    elif args.command == "analyze":
        output, figures, _ = export_main_analysis(args.project_root, official, shared, paired, run_id=args.run_id)
        if args.write_report:
            print("Report:", write_report(args.project_root, output, sensitivity_dir=args.sensitivity_dir))
        print("Metrics:", output, "\nFigures:", figures)
    elif args.command == "sensitivity":
        existing = {name: value for name, value in [("absolute_center", args.absolute_run), ("energy_quarter", args.quarter_run)] if value}
        print(run_sensitivity(args.project_root, args.data_dir or args.project_root / "data/imagenet/data",
                              shared, existing=existing, allow_download=args.allow_download))
    elif args.command == "sensitivity-analysis":
        if not args.absolute_run or not args.quarter_run:
            parser.error("Provide --absolute-run and --quarter-run (completed final directories).")
        output, figures, _ = export_sensitivity(args.project_root, shared,
            {"absolute_center": args.absolute_run, "energy_quarter": args.quarter_run}, run_id=args.run_id)
        print("Metrics:", output, "\nFigures:", figures)
    else:
        import torch
        records = select_sensitivity_records(select_main_records(args.project_root))[:50]
        output = erf_output_paths(args.project_root, args.run_id or new_run_id("benchmark_50"))[0]
        run_experiment(records, args.data_dir or args.project_root / "data/imagenet/data", output,
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       preprocessing="shared_vit_crop", project_root=args.project_root,
                       num_previews=0, allow_download=args.allow_download)
        print("Timing and peak VRAM in", output / "metadata.json")


if __name__ == "__main__":
    main()
