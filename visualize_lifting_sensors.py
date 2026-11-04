from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import Config


LIFTING_CLASSES = ("Lifting(lowering)", "Lifting(raising)")
SENSOR_PAIRS = (
    ("Acc.x", "Acc.y"),
    ("Acc.x", "Acc.z"),
    ("Acc.y", "Acc.z"),
    ("Gyro.x", "Gyro.y"),
    ("Gyro.x", "Gyro.z"),
    ("Gyro.y", "Gyro.z"),
    ("Baro.x", "Acc.z"),
    ("Baro.x", "Gyro.z"),
)
CATEGORY_STYLES = {
    "Other classes": {"color": "#b7b7b7", "size": 4, "alpha": 0.18, "zorder": 1},
    "Lifting(lowering)": {"color": "#0072B2", "size": 13, "alpha": 0.82, "zorder": 3},
    "Lifting(raising)": {"color": "#D55E00", "size": 13, "alpha": 0.82, "zorder": 4},
    "Both lifting labels": {"color": "#6A3D9A", "size": 16, "alpha": 0.9, "zorder": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create sensor-point visualizations that highlight Lifting(lowering) "
            "and Lifting(raising), with all non-lifting classes in grey."
        )
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
        default=Path("reports/lifting_sensor_visualizations"),
        help="Directory where PNGs and summary CSVs will be written.",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=None,
        help="Optional experiment id to plot. By default, all experiments are used.",
    )
    parser.add_argument(
        "--skipsize",
        type=int,
        default=25,
        help="Keep every Nth row before plotting, matching the common training downsample.",
    )
    parser.add_argument(
        "--max-other",
        type=int,
        default=25000,
        help="Maximum grey non-lifting points to draw.",
    )
    parser.add_argument(
        "--max-per-lifting-class",
        type=int,
        default=12000,
        help="Maximum points to draw for each highlighted lifting class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible point sampling.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path, config: Config, experiment: int | None) -> pd.DataFrame:
    meta_df = pd.read_pickle(dataset_path)
    if experiment is not None:
        meta_df = meta_df[meta_df["experiment"] == experiment]
        if meta_df.empty:
            raise ValueError(f"No rows found for experiment {experiment}.")

    frames = []
    for _, row in meta_df.iterrows():
        frame = row["data"].copy()
        frame["experiment"] = int(row["experiment"])
        if "scenario" in row:
            frame["scenario"] = int(row["scenario"])
        frames.append(frame)

    if not frames:
        raise ValueError("No timeseries frames found in the dataset.")

    df = pd.concat(frames, ignore_index=True)
    sensor_cols = [col for col in config.data.sensor_cols if col in df.columns]
    raw_label_cols = [col for col in config.data.label_cols if col in df.columns]
    return df[[*sensor_cols, *raw_label_cols, "experiment"]].copy()


def add_superclass_labels(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    superclass_cols = sorted(set(config.data.superclass_mapping.values()))
    for superclass in superclass_cols:
        children = [
            child
            for child, parent in config.data.superclass_mapping.items()
            if parent == superclass and child in df.columns
        ]
        df[superclass] = df[children].max(axis=1) if children else 0

    return df, superclass_cols


def add_plot_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lowering = df["Lifting(lowering)"].astype(bool)
    raising = df["Lifting(raising)"].astype(bool)

    df["plot_category"] = "Other classes"
    df.loc[lowering, "plot_category"] = "Lifting(lowering)"
    df.loc[raising, "plot_category"] = "Lifting(raising)"
    df.loc[lowering & raising, "plot_category"] = "Both lifting labels"
    return df


def sample_for_plot(
    df: pd.DataFrame,
    max_other: int,
    max_per_lifting_class: int,
    seed: int,
) -> pd.DataFrame:
    samples = []
    for category in CATEGORY_STYLES:
        category_df = df[df["plot_category"] == category]
        if category_df.empty:
            continue
        max_points = max_other if category == "Other classes" else max_per_lifting_class
        samples.append(
            category_df.sample(n=min(len(category_df), max_points), random_state=seed)
        )

    return pd.concat(samples, ignore_index=True)


def write_summaries(df: pd.DataFrame, plot_df: pd.DataFrame, output_dir: Path) -> None:
    full_counts = (
        df.groupby(["experiment", "plot_category"])
        .size()
        .rename("rows_after_skipsize")
        .reset_index()
    )
    plot_counts = (
        plot_df.groupby(["experiment", "plot_category"])
        .size()
        .rename("plotted_points")
        .reset_index()
    )
    summary = full_counts.merge(plot_counts, how="left", on=["experiment", "plot_category"])
    summary["plotted_points"] = summary["plotted_points"].fillna(0).astype(int)
    summary.to_csv(output_dir / "lifting_sensor_point_summary.csv", index=False)


def scatter_by_category(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for category, style in CATEGORY_STYLES.items():
        subset = df[df["plot_category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            c=style["color"],
            s=style["size"],
            alpha=style["alpha"],
            label=category,
            linewidths=0,
            zorder=style["zorder"],
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.2)


def plot_sensor_pairs(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, (x_col, y_col) in zip(axes.ravel(), SENSOR_PAIRS):
        scatter_by_category(ax, df, x_col, y_col)
        ax.set_title(f"{x_col} vs {y_col}")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Lifting Classes in Sensor Pair Space", y=0.985, fontsize=16)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sensor_3d(df: pd.DataFrame, axes_cols: tuple[str, str, str], title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for category, style in CATEGORY_STYLES.items():
        subset = df[df["plot_category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset[axes_cols[0]],
            subset[axes_cols[1]],
            subset[axes_cols[2]],
            c=style["color"],
            s=style["size"],
            alpha=min(style["alpha"] + 0.08, 0.95),
            label=category,
            depthshade=False,
            linewidths=0,
        )

    ax.set_xlabel(axes_cols[0])
    ax.set_ylabel(axes_cols[1])
    ax.set_zlabel(axes_cols[2])
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sensor_time_strips(df: pd.DataFrame, sensor_cols: list[str], output_path: Path) -> None:
    df = df.sort_values("source_index").copy()

    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, 2.15 * len(sensor_cols)), sharex=True)
    for ax, sensor_col in zip(axes, sensor_cols):
        scatter_by_category(ax, df, "source_index", sensor_col)
        ax.set_ylabel(sensor_col)
        ax.set_xlabel("")
        ax.legend_.remove() if ax.legend_ else None

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    axes[-1].set_xlabel("Row index after downsampling")
    fig.suptitle("Lifting Classes Across Sensor Traces", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = Config()
    dataset_path = args.dataset or Path("data") / config.data.dataset_file
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path, config, args.experiment)
    df, superclass_cols = add_superclass_labels(df, config)
    df = df.iloc[:: args.skipsize].reset_index(drop=True)
    df = df[~(df[superclass_cols] == 0).all(axis=1)].copy()
    df["source_index"] = np.arange(len(df))
    df = add_plot_category(df)

    plot_df = sample_for_plot(
        df=df,
        max_other=args.max_other,
        max_per_lifting_class=args.max_per_lifting_class,
        seed=args.seed,
    )
    write_summaries(df, plot_df, output_dir)

    plot_sensor_pairs(plot_df, output_dir / "lifting_sensor_pairs.png")
    plot_sensor_3d(
        plot_df,
        ("Acc.x", "Acc.y", "Acc.z"),
        "Lifting Classes in 3D Acceleration Space",
        output_dir / "lifting_acceleration_3d.png",
    )
    plot_sensor_3d(
        plot_df,
        ("Gyro.x", "Gyro.y", "Gyro.z"),
        "Lifting Classes in 3D Gyroscope Space",
        output_dir / "lifting_gyroscope_3d.png",
    )
    plot_sensor_time_strips(
        plot_df,
        config.data.sensor_cols,
        output_dir / "lifting_sensor_time_strips.png",
    )

    print(f"Wrote lifting sensor visualizations to {output_dir}")
    for path in sorted(output_dir.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
