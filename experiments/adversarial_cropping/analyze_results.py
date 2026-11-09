"""Validate a completed crop-grid run and create metrics and figures."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.adversarial_cropping.crop_utils import CROP_AREAS, CROP_POSITIONS


MODEL_NAMES = ("resnet50", "vit_b_16")
PREDICTION_COLUMNS = {
    "image_id", "model_name", "label", "pred_top1", "pred_top2",
    "pred_top3", "pred_top4", "pred_top5", "top1_correct",
    "top5_correct", "experiment", "condition", "severity",
}
AREA_TIE_ORDER = ("area_050", "area_065", "area_080", "area_095")
CONDITION_TIE_ORDER = tuple(
    f"{area}__{position}"
    for area in AREA_TIE_ORDER
    for position in CROP_POSITIONS
)
EXPECTED_CONDITIONS = tuple(
    f"{area}__{position}"
    for area in CROP_AREAS
    for position in CROP_POSITIONS
)


def load_run_predictions(final_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the two prediction files from one completed run."""
    frames = {}
    for model_name in MODEL_NAMES:
        path = final_dir / f"{model_name}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        frame = pd.read_csv(path, dtype={"image_id": "string", "condition": "string"})
        _validate_prediction_frame(frame, model_name)
        frames[model_name] = frame
    _validate_model_pair(frames)
    return frames


def _validate_prediction_frame(frame: pd.DataFrame, model_name: str) -> None:
    missing = PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{model_name} predictions are missing columns: {sorted(missing)}")
    if frame.empty or set(frame["model_name"].unique()) != {model_name}:
        raise ValueError(f"Expected only model_name={model_name!r}")
    if set(frame["experiment"].unique()) != {"adversarial_cropping"}:
        raise ValueError("Prediction rows must belong to adversarial_cropping")
    if frame.duplicated(["image_id", "condition"]).any():
        raise ValueError(f"Duplicate image/condition keys in {model_name}")

    conditions = set(frame["condition"].unique())
    expected = {"clean", *EXPECTED_CONDITIONS}
    if conditions != expected:
        raise ValueError(
            f"Unexpected condition set for {model_name}: "
            f"missing={sorted(expected - conditions)}, extra={sorted(conditions - expected)}"
        )
    counts = frame.groupby("condition")["image_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Condition sizes differ for {model_name}")
    cohort_size = int(counts.iloc[0])
    if cohort_size != 10_000 or len(frame) != cohort_size * len(expected):
        raise ValueError(f"Expected 10,000 IDs and 21 conditions for {model_name}")
    if frame.groupby("image_id")["label"].nunique().max() != 1:
        raise ValueError(f"Labels change across conditions for {model_name}")

    predictions = frame[[f"pred_top{i}" for i in range(1, 6)]].astype(int)
    if ((predictions < 0) | (predictions > 999)).any().any():
        raise ValueError("Predicted ImageNet labels must be in 0..999")
    if predictions.nunique(axis=1).ne(5).any():
        raise ValueError("Each row must contain five distinct predictions")
    labels = frame["label"].astype(int)
    top1 = predictions["pred_top1"].eq(labels).astype(int)
    top5 = predictions.eq(labels, axis=0).any(axis=1).astype(int)
    if not top1.equals(frame["top1_correct"].astype(int)):
        raise ValueError(f"Incorrect Top-1 flags in {model_name}")
    if not top5.equals(frame["top5_correct"].astype(int)):
        raise ValueError(f"Incorrect Top-5 flags in {model_name}")


def _validate_model_pair(frames: dict[str, pd.DataFrame]) -> None:
    clean = {
        model: frame.loc[frame["condition"] == "clean", ["image_id", "label"]]
        .set_index("image_id")["label"].astype(int).sort_index()
        for model, frame in frames.items()
    }
    if not clean[MODEL_NAMES[0]].equals(clean[MODEL_NAMES[1]]):
        raise ValueError("The two models do not use identical image IDs and labels")


def compute_condition_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute paired accuracy/drop/flip metrics for every crop condition."""
    model_name = str(frame["model_name"].iloc[0])
    clean = frame.loc[frame["condition"] == "clean"].set_index("image_id")
    clean_top1 = 100 * clean["top1_correct"].astype(int).mean()
    clean_top5 = 100 * clean["top5_correct"].astype(int).mean()
    rows = []
    for condition in EXPECTED_CONDITIONS:
        changed = frame.loc[frame["condition"] == condition].set_index("image_id")
        changed = changed.loc[clean.index]
        top1 = 100 * changed["top1_correct"].astype(int).mean()
        top5 = 100 * changed["top5_correct"].astype(int).mean()
        top1_drop = clean_top1 - top1
        top5_drop = clean_top5 - top5
        area_name, position = condition.split("__", 1)
        rows.append({
            "model_name": model_name,
            "condition": condition,
            "severity": CROP_AREAS[area_name],
            "position": position,
            "num_images": len(changed),
            "clean_top1_accuracy_percent": clean_top1,
            "perturbed_top1_accuracy_percent": top1,
            "absolute_top1_drop_pp": top1_drop,
            "relative_top1_drop_percent": 100 * top1_drop / clean_top1,
            "clean_top5_accuracy_percent": clean_top5,
            "perturbed_top5_accuracy_percent": top5,
            "absolute_top5_drop_pp": top5_drop,
            "relative_top5_drop_percent": 100 * top5_drop / clean_top5,
            "prediction_flip_rate_percent": 100 * changed["pred_top1"].astype(int)
            .ne(clean["pred_top1"].astype(int)).mean(),
        })
    return pd.DataFrame(rows)


def aggregate_condition_metrics(
    metrics: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    """Average balanced condition metrics by severity or crop position."""
    metric_columns = [
        "clean_top1_accuracy_percent", "perturbed_top1_accuracy_percent",
        "absolute_top1_drop_pp", "relative_top1_drop_percent",
        "clean_top5_accuracy_percent", "perturbed_top5_accuracy_percent",
        "absolute_top5_drop_pp", "relative_top5_drop_percent",
        "prediction_flip_rate_percent",
    ]
    grouped = metrics.groupby(["model_name", group_column], sort=False)
    result = grouped[metric_columns].mean().reset_index()
    result.insert(2, "num_conditions", grouped.size().to_numpy())
    result.insert(3, "num_images_per_condition", grouped["num_images"].first().to_numpy())
    return result


def select_worst_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select each model's largest Top-1 drop using the documented tie order."""
    ranked = metrics.copy()
    ranked["area_rank"] = ranked["severity"].map(
        {0.50: 0, 0.65: 1, 0.80: 2, 0.95: 3}
    )
    ranked["position_rank"] = ranked["position"].map(
        {position: index for index, position in enumerate(CROP_POSITIONS)}
    )
    ranked = ranked.sort_values(
        ["model_name", "absolute_top1_drop_pp", "area_rank", "position_rank"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    return ranked.groupby("model_name", sort=False).head(1).drop(
        columns=["area_rank", "position_rank"]
    ).reset_index(drop=True)


def build_transfer_metrics(
    metrics: pd.DataFrame, worst_conditions: pd.DataFrame
) -> pd.DataFrame:
    """Apply each source model's selected condition to both target models."""
    rows = []
    for source in worst_conditions.itertuples(index=False):
        selected = metrics.loc[metrics["condition"] == source.condition]
        for target in selected.itertuples(index=False):
            row = target._asdict()
            row = {
                "source_model": source.model_name,
                "selected_condition": source.condition,
                "target_model": target.model_name,
                **{key: value for key, value in row.items() if key not in {"model_name", "condition"}},
            }
            rows.append(row)
    return pd.DataFrame(rows)


def select_per_image_worst(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Select the first deterministic crop that flips each clean prediction."""
    order = {condition: index for index, condition in enumerate(CONDITION_TIE_ORDER)}
    outputs = []
    for model_name, frame in frames.items():
        clean = frame.loc[frame["condition"] == "clean", [
            "image_id", "label", "pred_top1", "top1_correct"
        ]].rename(columns={
            "pred_top1": "clean_pred_top1",
            "top1_correct": "clean_top1_correct",
        })
        crops = frame.loc[frame["condition"] != "clean", [
            "image_id", "condition", "severity", "pred_top1", "top1_correct"
        ]].rename(columns={
            "pred_top1": "crop_pred_top1",
            "top1_correct": "crop_top1_correct",
        })
        paired = crops.merge(clean, on="image_id", validate="many_to_one")
        paired["prediction_flipped"] = paired["crop_pred_top1"].astype(int).ne(
            paired["clean_pred_top1"].astype(int)
        ).astype(int)
        paired["accuracy_lost"] = (
            paired["clean_top1_correct"].astype(int)
            > paired["crop_top1_correct"].astype(int)
        ).astype(int)
        paired["tie_rank"] = paired["condition"].map(order)
        paired = paired.sort_values(
            ["image_id", "prediction_flipped", "tie_rank"],
            ascending=[True, False, True],
            kind="stable",
        ).groupby("image_id", sort=False).head(1)
        paired.insert(1, "model_name", model_name)
        outputs.append(paired.drop(columns="tie_rank"))
    return pd.concat(outputs, ignore_index=True)


def create_figures(
    metrics: pd.DataFrame,
    severity_metrics: pd.DataFrame,
    transfer_metrics: pd.DataFrame,
    figure_dir: Path,
    *,
    overwrite: bool,
) -> list[Path]:
    """Create the final static figures and return their paths."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.append(_plot_severity_accuracy(severity_metrics, figure_dir, overwrite))
    outputs.append(_plot_severity_drop(severity_metrics, figure_dir, overwrite))
    outputs.append(_plot_flip_rate(severity_metrics, figure_dir, overwrite))
    outputs.append(_plot_condition_heatmap(metrics, figure_dir, overwrite))
    outputs.append(_plot_transfer(transfer_metrics, figure_dir, overwrite))
    return outputs


def _save_figure(fig: plt.Figure, path: Path, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_severity_accuracy(data: pd.DataFrame, output: Path, overwrite: bool) -> Path:
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for model_name, group in data.groupby("model_name", sort=False):
        group = group.sort_values("severity", ascending=False)
        axis.plot(group["severity"], group["perturbed_top1_accuracy_percent"], marker="o", label=model_name)
    axis.set(xlabel="Retained area ratio", ylabel="Mean Top-1 accuracy (%)", title="Crop accuracy by retained area")
    axis.invert_xaxis()
    axis.grid(alpha=0.25)
    axis.legend()
    return _save_figure(fig, output / "top1_accuracy_by_severity.png", overwrite)


def _plot_severity_drop(data: pd.DataFrame, output: Path, overwrite: bool) -> Path:
    pivot = data.pivot(index="severity", columns="model_name", values="absolute_top1_drop_pp").sort_index(ascending=False)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    pivot.plot(kind="bar", ax=axis, rot=0)
    axis.set(xlabel="Retained area ratio", ylabel="Top-1 drop (pp)", title="Mean Top-1 accuracy drop by severity")
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output / "top1_drop_by_severity.png", overwrite)


def _plot_flip_rate(data: pd.DataFrame, output: Path, overwrite: bool) -> Path:
    pivot = data.pivot(index="severity", columns="model_name", values="prediction_flip_rate_percent").sort_index(ascending=False)
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    pivot.plot(kind="bar", ax=axis, rot=0)
    axis.set(xlabel="Retained area ratio", ylabel="Prediction flip rate (%)", title="Top-1 prediction instability by severity")
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output / "prediction_flip_rate_by_severity.png", overwrite)


def _plot_condition_heatmap(data: pd.DataFrame, output: Path, overwrite: bool) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    severities = [0.95, 0.80, 0.65, 0.50]
    for axis, model_name in zip(axes, MODEL_NAMES):
        model = data.loc[data["model_name"] == model_name]
        matrix = model.pivot(index="severity", columns="position", values="absolute_top1_drop_pp").reindex(index=severities, columns=CROP_POSITIONS)
        image = axis.imshow(matrix.to_numpy(), cmap="YlOrRd", aspect="auto")
        axis.set_title(model_name)
        axis.set_xticks(range(len(CROP_POSITIONS)), CROP_POSITIONS, rotation=35, ha="right")
        axis.set_yticks(range(len(severities)), [f"{value:.2f}" for value in severities])
        axis.set_xlabel("Crop position")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, label="Top-1 drop (pp)")
    axes[0].set_ylabel("Retained area ratio")
    fig.suptitle("Top-1 accuracy drop across the fixed crop grid")
    return _save_figure(fig, output / "condition_drop_heatmap.png", overwrite)


def _plot_transfer(data: pd.DataFrame, output: Path, overwrite: bool) -> Path:
    labels = [f"{row.source_model} → {row.target_model}" for row in data.itertuples(index=False)]
    values = data["absolute_top1_drop_pp"].to_numpy()
    fig, axis = plt.subplots(figsize=(8, 4.8))
    colors = np.where(data["source_model"].eq(data["target_model"]), "#4C78A8", "#F58518")
    axis.bar(labels, values, color=colors)
    axis.set(ylabel="Top-1 drop (pp)", title="Cross-model transfer of aggregate worst conditions")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output / "cross_model_transfer.png", overwrite)


def _write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite analysis output: {path}")
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_run(repo_root: Path, run_id: str, *, overwrite: bool = False) -> dict:
    """Analyze one completed run and write validated CSV/JSON/PNG outputs."""
    metric_root = repo_root / "results/metrics/experiments/adversarial_cropping" / run_id
    final_dir = metric_root / "final"
    metadata_path = final_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("state") != "complete" or metadata.get("run_id") != run_id:
        raise ValueError("Only matching, completed main runs may be analyzed")

    frames = load_run_predictions(final_dir)
    metrics = pd.concat(
        [compute_condition_metrics(frames[model]) for model in MODEL_NAMES],
        ignore_index=True,
    )
    severity_metrics = aggregate_condition_metrics(metrics, "severity")
    position_metrics = aggregate_condition_metrics(metrics, "position")
    worst_conditions = select_worst_conditions(metrics)
    transfer_metrics = build_transfer_metrics(metrics, worst_conditions)
    worst_cases = select_per_image_worst(frames)

    outputs = {
        "metrics.csv": metrics,
        "severity_metrics.csv": severity_metrics,
        "position_metrics.csv": position_metrics,
        "worst_conditions.csv": worst_conditions,
        "transfer_metrics.csv": transfer_metrics,
        "worst_cases.csv": worst_cases,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, final_dir / filename, overwrite)

    figure_dir = repo_root / "results/figures/experiments/adversarial_cropping" / run_id / "final"
    figures = create_figures(metrics, severity_metrics, transfer_metrics, figure_dir, overwrite=overwrite)
    analysis_metadata = {
        "run_id": run_id,
        "source_state": metadata["state"],
        "num_images": metadata["cohort"]["num_images"],
        "models": list(MODEL_NAMES),
        "num_crop_conditions": len(EXPECTED_CONDITIONS),
        "selected_worst_conditions": dict(zip(worst_conditions["model_name"], worst_conditions["condition"])),
        "source_sha256": {
            f"{model}_predictions.csv": _sha256(final_dir / f"{model}_predictions.csv")
            for model in MODEL_NAMES
        },
        "generated_metrics": list(outputs),
        "generated_figures": [path.name for path in figures],
    }
    analysis_metadata_path = final_dir / "analysis_metadata.json"
    if analysis_metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite analysis output: {analysis_metadata_path}")
    analysis_metadata_path.write_text(json.dumps(analysis_metadata, indent=2), encoding="utf-8")
    return analysis_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_run(args.repo_root.resolve(), args.run_id, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
