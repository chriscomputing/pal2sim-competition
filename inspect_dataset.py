from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.config import Config


LABEL_DESCRIPTIONS = {
    "No loading": "No annotated loading activity is active at that timestamp.",
    "Driving(straight)": "The vehicle is driving forward or backward without a curve.",
    "Driving(curve)": "The vehicle is moving while steering through a curve or turn.",
    "Lifting(raising)": "The forks or load handling system are being raised.",
    "Lifting(lowering)": "The forks or load handling system are being lowered.",
    "Standing": "The vehicle is stationary or waiting.",
    "Docking": "The vehicle is aligning with or approaching a docking/loading position.",
    "Forks(entering or leaving front)": "Fork insertion or removal happens from the front side of the load.",
    "Forks(entering or leaving side)": "Fork insertion or removal happens from the side of the load.",
    "Wrapping": "The turntable wrapping process is actively running.",
    "Wrapping(preparation)": "Preparation steps before the actual wrapping process.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the Pal2Sim CPS dataset and generate a short report with plots."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the pickle dataset. Defaults to data/<dataset_file> from utils.config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_inspection"),
        help="Directory where plots and summary.json will be written.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=4000,
        help="Number of rows to use for the sensor trace plot.",
    )
    return parser.parse_args()


def load_nested_dataset(dataset_path: Path, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_df = pd.read_pickle(dataset_path)

    frames = []
    for _, row in meta_df.iterrows():
        frame = row["data"].copy()
        frame["experiment"] = int(row["experiment"])
        frame["scenario"] = int(row["scenario"])
        frames.append(frame)

    all_df = pd.concat(frames, ignore_index=True)
    expected_cols = ["time", *config.data.sensor_cols, *config.data.label_cols]
    available_expected = [col for col in expected_cols if col in all_df.columns]
    all_df = all_df[[*available_expected, *(c for c in ["No loading", "experiment", "scenario"] if c in all_df.columns)]]
    return meta_df, all_df


def build_superclass_frame(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, list[str]]:
    super_df = df.copy()
    super_cols = sorted(set(config.data.superclass_mapping.values()))

    for super_name in super_cols:
        children = [
            child
            for child, parent in config.data.superclass_mapping.items()
            if parent == super_name and child in super_df.columns
        ]
        super_df[super_name] = super_df[children].max(axis=1) if children else 0

    return super_df, super_cols


def label_summary(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    counts = df[label_cols].sum().sort_values(ascending=False)
    summary = pd.DataFrame({
        "count": counts.astype(int),
        "share_percent": (counts / len(df) * 100).round(2),
        "meaning": [LABEL_DESCRIPTIONS.get(label, "Meaning not documented in repo.") for label in counts.index],
    })
    return summary


def experiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("experiment")
    summary = grouped.agg(
        rows=("time", "size"),
        start_time_s=("time", "min"),
        end_time_s=("time", "max"),
    )
    summary["duration_s"] = summary["end_time_s"] - summary["start_time_s"]
    return summary.reset_index().round(2)


def sensor_summary(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    return df[sensor_cols].describe().T[["mean", "std", "min", "max"]].round(4)


def plot_bar(series: pd.Series, title: str, output_path: Path, color: str) -> None:
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=series.index, y=series.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)

    for patch, value in zip(ax.patches, series.values):
        ax.annotate(
            f"{int(value):,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_experiment_rows(experiment_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=experiment_df, x="experiment", y="rows", color="#4c78a8")
    ax.set_title("Rows per Experiment")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Rows")

    for patch, value in zip(ax.patches, experiment_df["rows"]):
        ax.annotate(
            f"{int(value):,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_sensor_snippet(df: pd.DataFrame, sensor_cols: list[str], sample_points: int, output_path: Path) -> None:
    sample = df[df["experiment"] == df["experiment"].min()][["time", *sensor_cols]].head(sample_points).copy()
    sample = sample.sort_values("time")

    melted = sample.melt(id_vars="time", value_vars=sensor_cols, var_name="sensor", value_name="value")

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=melted, x="time", y="value", hue="sensor", linewidth=1)
    plt.title(f"Sensor Snippet from Experiment {int(df['experiment'].min())} ({len(sample):,} rows)")
    plt.xlabel("Time [s]")
    plt.ylabel("Raw Sensor Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def print_report(
    dataset_path: Path,
    meta_df: pd.DataFrame,
    all_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    raw_label_df: pd.DataFrame,
    super_label_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print(f"Dataset: {dataset_path}")
    print(f"Top-level shape: {meta_df.shape[0]} experiments x {meta_df.shape[1]} metadata columns")
    print(f"Combined rows: {len(all_df):,}")
    print(f"Columns in each nested timeseries frame: {len(all_df.columns)}")
    print()

    print("Experiments")
    print(experiment_df.to_string(index=False))
    print()

    print("Sensors")
    print(", ".join(sensor_df.index.tolist()))
    print(sensor_df.to_string())
    print()

    label_total = int(raw_label_df["count"].sum())
    multi_label_rows = int((all_df[raw_label_df.index].sum(axis=1) > 1).sum())
    zero_label_rows = int((all_df[raw_label_df.index].sum(axis=1) == 0).sum())

    print("Raw Class Labels")
    print(raw_label_df.to_string())
    print()
    print(f"Total positive raw label activations: {label_total:,}")
    print(f"Rows with more than one active raw label: {multi_label_rows:,}")
    print(f"Rows with no active raw label: {zero_label_rows:,}")
    print()

    print("Superclass Labels Used by Training")
    print(super_label_df.to_string())
    print()

    print("Plot files")
    for path in sorted(output_dir.glob("*.png")):
        print(path)


def save_summary_json(
    output_path: Path,
    dataset_path: Path,
    experiment_df: pd.DataFrame,
    raw_label_df: pd.DataFrame,
    super_label_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
) -> None:
    payload = {
        "dataset": str(dataset_path),
        "experiments": experiment_df.to_dict(orient="records"),
        "raw_labels": raw_label_df.reset_index(names="label").to_dict(orient="records"),
        "superclasses": super_label_df.reset_index(names="label").to_dict(orient="records"),
        "sensor_summary": sensor_df.reset_index(names="sensor").to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    config = Config()
    dataset_path = args.dataset or Path("data") / config.data.dataset_file
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_df, all_df = load_nested_dataset(dataset_path, config)
    experiment_df = experiment_summary(all_df)

    raw_label_cols = [col for col in ["No loading", *config.data.label_cols] if col in all_df.columns]
    raw_label_df = label_summary(all_df, raw_label_cols)

    super_df, super_cols = build_superclass_frame(all_df, config)
    super_label_df = label_summary(super_df, super_cols)
    sensor_df = sensor_summary(all_df, config.data.sensor_cols)

    plot_bar(raw_label_df["count"], "Raw Label Counts", output_dir / "raw_label_counts.png", "#72b7b2")
    plot_bar(super_label_df["count"], "Superclass Counts", output_dir / "superclass_counts.png", "#f58518")
    plot_experiment_rows(experiment_df, output_dir / "rows_per_experiment.png")
    plot_sensor_snippet(all_df, config.data.sensor_cols, args.sample_points, output_dir / "sensor_snippet_experiment_1.png")
    save_summary_json(output_dir / "summary.json", dataset_path, experiment_df, raw_label_df, super_label_df, sensor_df)

    print_report(
        dataset_path=dataset_path,
        meta_df=meta_df,
        all_df=all_df,
        experiment_df=experiment_df,
        raw_label_df=raw_label_df,
        super_label_df=super_label_df,
        sensor_df=sensor_df,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
