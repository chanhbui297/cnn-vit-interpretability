"""Audit a Colab sensitivity run against its independently recomputed local run.

First run erf_workflow sensitivity-analysis, then this module with --source and
--recomputed final directories. The incoming Colab run is never modified.
"""
import argparse
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .erf_experiment import select_main_records, select_sensitivity_records
from .erf_statistics import ERF_METRICS, bootstrap_mean_ci, load_completed_erf_run
from .erf_workflow import (
    ROOT, SENSITIVITY_PROBES, assert_same_table, read_json, sha256,
    validate_probe_run, write_json,
)


def verify_hashes(directory, metadata):
    checksums = metadata.get("artifact_sha256", {})
    required = {"selected_images.csv", "sensitivity_summary.csv", "sensitivity_paired_differences.csv"}
    if not required.issubset(checksums):
        raise ValueError("Missing sensitivity CSV checksums.")
    for name, checksum in checksums.items():
        if Path(name).name != name or sha256(Path(directory) / name) != checksum:
            raise ValueError(f"Sensitivity checksum mismatch: {name}")


def sensitivity_profiles(pairs, summary, n_resamples=20000):
    """Use the same 200 IDs for all profiles; do not mix in 2,000-image means."""
    baseline = pairs[pairs.probe == "absolute_center"]
    rows = []
    for index, ((metric, stage, unit), group) in enumerate(baseline.groupby(["metric", "stage", "unit"], sort=True)):
        group = group.sort_values("image_id")
        if not group.image_id.is_unique or len(group) != 200:
            raise ValueError("Baseline profile requires exactly 200 unique pairs.")
        low, high = bootstrap_mean_ci(group.difference_baseline, 42 + index, n_resamples)
        scale = ERF_METRICS[metric][0]
        rows.append({"probe": "energy_center", "metric": metric, "stage": stage, "unit": unit,
                     "n_pairs": len(group), "mean_gap": group.difference_baseline.mean(),
                     "mean_ci95_low": low, "mean_ci95_high": high,
                     "resnet_mean": group.resnet50_baseline.mean() * scale,
                     "vit_mean": group.vit_b_16_baseline.mean() * scale})
    for (probe, metric, stage, unit), group in pairs.groupby(["probe", "metric", "stage", "unit"], sort=True):
        result = summary[(summary.probe == probe) & (summary.metric == metric) & (summary.stage == stage)]
        if len(result) != 1 or not group.image_id.is_unique or len(group) != 200:
            raise ValueError("Changed profile requires one summary and 200 unique pairs.")
        result = result.iloc[0]
        scale = ERF_METRICS[metric][0]
        rows.append({"probe": probe, "metric": metric, "stage": stage, "unit": unit,
                     "n_pairs": len(group), "mean_gap": result.changed_gap_mean,
                     "mean_ci95_low": result.changed_gap_ci95_low, "mean_ci95_high": result.changed_gap_ci95_high,
                     "resnet_mean": group.resnet50_changed.mean() * scale,
                     "vit_mean": group.vit_b_16_changed.mean() * scale})
    return pd.DataFrame(rows)


def plot_profiles(profiles, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    styles = [("energy_center", -.15, "tab:blue", "mean square / center"),
              ("absolute_center", 0., "tab:orange", "mean absolute / center"),
              ("energy_quarter", .15, "tab:green", "mean square / quarter")]
    for col, (metric, (_, unit)) in enumerate(ERF_METRICS.items()):
        for row in range(2):
            axis = axes[row, col]
            for probe, offset, color, label in styles:
                group = profiles[(profiles.probe == probe) & (profiles.metric == metric)].sort_values("stage")
                if row:
                    group = group[group.stage == 4]
                y = group.stage.to_numpy() + offset
                axis.hlines(y, group.mean_ci95_low, group.mean_ci95_high, color=color, linewidth=2)
                axis.scatter(group.mean_gap, y, color=color, label=label, zorder=3)
            axis.axvline(0, color="gray", linestyle="--", linewidth=1)
            axis.set(xlabel=f"ViT - ResNet ({unit})", ylabel="Relative stage", yticks=[4] if row else [1, 2, 3, 4],
                     title=("A90" if col == 0 else "Dispersion") + (" - final-stage zoom" if row else ""))
            axis.grid(alpha=.2)
            axis.legend()
    fig.suptitle("Same 200 images: mean gaps and descriptive pointwise 95% bootstrap CIs")
    fig.savefig(destination, dpi=170)
    return fig


def plot_gap_changes(summary, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (metric, (_, unit)) in zip(axes, ERF_METRICS.items()):
        for probe, offset, color in [("absolute_center", -.08, "tab:orange"),
                                      ("energy_quarter", .08, "tab:green")]:
            group = summary[(summary.probe == probe) & (summary.metric == metric)].sort_values("stage")
            y = group.stage.to_numpy() + offset
            axis.hlines(y, group.mean_ci95_low, group.mean_ci95_high, color=color, linewidth=2)
            axis.scatter(group.gap_change_mean, y, color=color, label=probe)
        axis.axvline(0, color="gray", linestyle="--")
        axis.set(xlabel=f"Change in ViT - ResNet gap ({unit})", ylabel="Relative stage", yticks=[1, 2, 3, 4])
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle("Same 200 images: descriptive pointwise 95% CIs for paired gap changes")
    fig.savefig(destination, dpi=170)
    return fig


def audit_sensitivity(project_root, source, recomputed, *, execution_bundle=None):
    root, source, recomputed = Path(project_root), Path(source), Path(recomputed)
    original, derived = read_json(source / "metadata.json"), read_json(recomputed / "metadata.json")
    for metadata in [original, derived]:
        if (metadata.get("status") != "complete" or metadata.get("num_images") != 200
                or metadata.get("seed") != 42 or metadata.get("bootstrap_resamples") != 20000):
            raise ValueError("Expected a complete 200-image, seed-42, 20,000-resample analysis.")
    verify_hashes(source, original)
    verify_hashes(recomputed, derived)
    records = select_sensitivity_records(select_main_records(root))
    assert_same_table(pd.read_csv(source / "selected_images.csv"), pd.DataFrame(records), ["image_id"], "fixed 200-image cohort")
    for filename, keys in [("selected_images.csv", ["image_id"]),
                           ("sensitivity_summary.csv", ["probe", "metric", "stage"]),
                           ("sensitivity_paired_differences.csv", ["probe", "metric", "stage", "image_id"])]:
        assert_same_table(pd.read_csv(source / filename), pd.read_csv(recomputed / filename), keys, filename)
    base = root / "results/metrics/experiments/erf"
    baseline_dir = base / Path(original["baseline"]["path"]).parent.name / "final"
    if sha256(baseline_dir / "metrics.csv") != original["baseline"]["metrics_sha256"]:
        raise ValueError("Baseline metric bytes differ from source provenance.")
    if derived["baseline"]["metrics_sha256"] != original["baseline"]["metrics_sha256"]:
        raise ValueError("Recomputation used a different baseline.")
    baseline_meta = read_json(baseline_dir / "metadata.json")
    provenance, timings = {}, {}
    bundle = zipfile.ZipFile(execution_bundle) if execution_bundle and Path(execution_bundle).is_file() else None
    try:
        for name, (target, position) in SENSITIVITY_PROBES.items():
            expected = original["sources"][name]
            directory = base / Path(expected["path"]).parent.name / "final"
            if sha256(directory / "metrics.csv") != expected["metrics_sha256"] or derived["sources"][name]["metrics_sha256"] != expected["metrics_sha256"]:
                raise ValueError(f"Source metric hash mismatch: {name}")
            frame, metadata = load_completed_erf_run(directory, records, "shared_vit_crop", target_mode=target, position=position)
            validate_probe_run(frame, metadata, records, root, target, position)
            metrics = ["area50_fraction", "area90_fraction", "dispersion_normalized", "center_energy_fraction", "outer_energy_fraction"]
            expected_summary = frame.groupby(["model", "stage", "layer"])[metrics].agg(["mean", "std", "count"])
            pd.testing.assert_frame_equal(pd.read_csv(directory / "summary.csv", header=[0, 1], index_col=[0, 1, 2]),
                                          expected_summary, check_dtype=False, atol=1e-10, rtol=1e-8)
            versions = {key: metadata[key] == baseline_meta[key] for key in ["torch", "torchvision", "numpy", "pandas"]}
            if not all(versions.values()):
                raise ValueError(f"Runtime versions differ from cached baseline: {name}")
            method_files = ["experiments/erf/erf_experiment.py", "experiments/erf/erf_statistics.py", "experiments/erf/erf_workflow.py",
                            "src/models.py", "src/data.py", "src/evaluation.py"]
            matches = {}
            if bundle:
                import hashlib
                for filename in method_files:
                    matches[filename] = hashlib.sha256(bundle.read(filename)).hexdigest() == metadata["code"]["file_sha256"].get(filename)
                if not all(matches.values()):
                    raise ValueError(f"Executed method source differs from supplied execution bundle: {name}")
            provenance[name] = {"path": str(directory), "metrics_sha256": sha256(directory / "metrics.csv"),
                                "rows": len(frame), "runtime_matches_baseline": versions, "method_matches_execution_bundle": matches}
            timings[name] = {"seconds": metadata.get("timings_seconds"), "peak_cuda_allocated_bytes": metadata.get("peak_cuda_memory_bytes")}
    finally:
        if bundle:
            bundle.close()
    checks = ["200 fixed IDs/labels/indices; no correctness filtering", "source CSV and provenance SHA-256",
              "1,600 valid image/model/stage rows per probe; checkpoint, crop, target and position",
              "source summary mean/SD/N", "same model-runtime versions as cached baseline",
              "all 3,200 paired-difference rows", "all 16 mean/CI/model-change summaries recomputed"]
    profiles = sensitivity_profiles(pd.read_csv(recomputed / "sensitivity_paired_differences.csv"),
                                    pd.read_csv(recomputed / "sensitivity_summary.csv"))
    profiles.to_csv(recomputed / "sensitivity_profiles.csv", index=False)
    figure_dir = root / "results/figures/experiments/erf" / recomputed.parent.name / "final"
    fig = plot_profiles(profiles, figure_dir / "sensitivity_gap_levels.png")
    plt.close(fig)
    fig = plot_gap_changes(pd.read_csv(recomputed / "sensitivity_summary.csv"), figure_dir / "sensitivity_gap_change.png")
    plt.close(fig)
    audit = {"status": "passed", "source_analysis": str(source), "source_metadata_sha256": sha256(source / "metadata.json"),
             "num_images": 200, "num_classes": len({r["label"] for r in records}), "checks": checks,
             "sources": provenance, "timings": timings,
             "execution_bundle_sha256": sha256(execution_bundle) if execution_bundle and Path(execution_bundle).is_file() else None,
             "auditor_source_sha256": sha256(Path(__file__)),
             "baseline_profile_ci": "20,000 paired-image percentile resamples; seeds 42..49 for metric/stage groups sorted alphabetically",
             "limitations": ["Descriptive pointwise CIs; no new p-values or changes to the main 16-test Holm family.",
                             "Only two scalar choices and two grid positions; no universal position/target invariance claim.",
                             "Quarter-grid positions are not exactly matched in input coordinates; boundary effects can contribute.",
                             "Notebook bytes differ from the execution bundle; the six method/loader source files were checked independently.",
                             "Gradient calculations are checked through saved artifacts, not independently rerun on GPU."]}
    write_json(recomputed / "audit.json", audit)
    derived.update(validation="passed_against_original_colab_artifacts", figures_dir=str(figure_dir),
                   audit_sha256=sha256(recomputed / "audit.json"),
                   artifact_sha256={path.name: sha256(path) for path in recomputed.glob("*.csv")})
    write_json(recomputed / "metadata.json", derived)
    return audit, profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--recomputed", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--execution-bundle", type=Path, default=ROOT / "outputs/erf_colab_update.zip")
    args = parser.parse_args()
    audit, _ = audit_sensitivity(args.project_root, args.source, args.recomputed, execution_bundle=args.execution_bundle)
    print(json.dumps({"status": audit["status"], "num_images": audit["num_images"], "num_classes": audit["num_classes"], "checks": audit["checks"]}, indent=2))
