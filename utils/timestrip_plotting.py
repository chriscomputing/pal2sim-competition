import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _timestrip_filename(
    fold: int | None,
    plot_name: str,
    split_name: str | None,
    label_kind: str | None,
) -> str:
    parts = [
        f"fold_{fold}" if fold is not None else None,
        plot_name,
        split_name,
        label_kind,
    ]
    return "_".join(_safe_filename(part) for part in parts if part) + ".jpeg"


def _legend_handle(label: str, color: str, size: int, alpha: float) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=max(5, size + 3),
        alpha=alpha,
        label=label,
    )


def _build_class_color_map(target_names: list[str]) -> dict[str, str]:
    fixed_colors = {
        "Driving(curve)": "tab:green",
        "Driving(straight)": "tab:cyan",
        "Lifting(lowering)": "tab:blue",
        "Lifting(raising)": "tab:orange",
        "Stationary processes": "tab:red",
        "Turntable wrapping": "tab:brown",
    }
    fallback_colors = [
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:gray",
        "navy",
        "darkgreen",
        "goldenrod",
        "teal",
    ]
    color_map: dict[str, str] = {}
    fallback_index = 0

    for class_name in target_names:
        if class_name in fixed_colors:
            color_map[class_name] = fixed_colors[class_name]
        else:
            color_map[class_name] = fallback_colors[
                fallback_index % len(fallback_colors)
            ]
            fallback_index += 1

    return color_map


def _combination_label(active_names: tuple[str, ...]) -> str:
    return " + ".join(active_names) if active_names else "No class"


def _combination_counts(
    label_masks: np.ndarray,
    target_names: list[str],
) -> Counter[tuple[str, ...]]:
    combinations: Counter[tuple[str, ...]] = Counter()
    for row in label_masks:
        active_names = tuple(
            class_name
            for class_name, is_active in zip(target_names, row)
            if is_active
        )
        combinations[active_names] += 1
    return combinations


def _plot_label_overlap_diagrams(
    label_masks: np.ndarray,
    target_names: list[str],
    title: str,
    output_path: Path,
    fold: int | None,
    split_name: str | None,
    label_kind: str | None,
    show: bool,
    max_combinations: int,
) -> dict[str, Path]:
    saved_paths: dict[str, Path] = {}
    label_counts = label_masks.sum(axis=1)

    count_values, count_frequencies = np.unique(label_counts, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(count_values.astype(str), count_frequencies, color="tab:blue", alpha=0.85)
    ax.set_title(f"{title}: labels per sample")
    ax.set_xlabel("Number of active classes")
    ax.set_ylabel("Samples")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    for x_index, frequency in enumerate(count_frequencies):
        ax.text(x_index, frequency, str(int(frequency)), ha="center", va="bottom")
    plt.tight_layout()
    label_count_path = output_path / _timestrip_filename(
        fold, "labels_per_sample", split_name, label_kind
    )
    fig.savefig(label_count_path, dpi=180, bbox_inches="tight")
    saved_paths["labels_per_sample"] = label_count_path
    if show:
        plt.show()
    plt.close(fig)

    combinations = _combination_counts(label_masks, target_names)
    top_combinations = combinations.most_common(max_combinations)
    combination_labels = [
        _combination_label(active_names) for active_names, _ in top_combinations
    ]
    combination_values = [count for _, count in top_combinations]

    fig_height = max(5, 0.45 * len(top_combinations))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y_positions = np.arange(len(top_combinations))
    ax.barh(y_positions, combination_values, color="tab:green", alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(combination_labels)
    ax.invert_yaxis()
    ax.set_title(f"{title}: class combinations")
    ax.set_xlabel("Samples")
    ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.3)
    for y_position, value in zip(y_positions, combination_values):
        ax.text(value, y_position, f" {int(value)}", va="center")
    plt.tight_layout()
    combination_path = output_path / _timestrip_filename(
        fold, "class_combinations", split_name, label_kind
    )
    fig.savefig(combination_path, dpi=180, bbox_inches="tight")
    saved_paths["class_combinations"] = combination_path
    if show:
        plt.show()
    plt.close(fig)

    pairwise_counts = label_masks.astype(int).T @ label_masks.astype(int)
    fig_width = max(8, 0.75 * len(target_names))
    fig_height = max(7, 0.65 * len(target_names))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pairwise_counts, cmap="Blues")
    ax.set_title(f"{title}: pairwise class overlap")
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    fig.colorbar(image, ax=ax, label="Samples")

    max_count = pairwise_counts.max(initial=0)
    text_threshold = max_count / 2 if max_count else 0
    for row_index in range(pairwise_counts.shape[0]):
        for col_index in range(pairwise_counts.shape[1]):
            value = int(pairwise_counts[row_index, col_index])
            text_color = "white" if value > text_threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    plt.tight_layout()
    pairwise_path = output_path / _timestrip_filename(
        fold, "pairwise_class_overlap", split_name, label_kind
    )
    fig.savefig(pairwise_path, dpi=180, bbox_inches="tight")
    saved_paths["pairwise_class_overlap"] = pairwise_path
    if show:
        plt.show()
    plt.close(fig)

    return saved_paths


def plot_label_diagnostics(
    data: np.ndarray,
    labels: np.ndarray,
    target_names: list[str],
    sensor_cols: list[str],
    title: str = "Classes Across Sensor Traces",
    output_dir: Path | str | None = None,
    fold: int | None = None,
    split_name: str | None = None,
    label_kind: str | None = None,
    show: bool = False,
    max_combinations: int = 20,
) -> dict[str, Path]:
    """
    data: (n, features, sequence_length)
    labels: (n, classes) - Multi-hot encoded
    """
    n_features = len(sensor_cols)

    if labels.shape[1] != len(target_names):
        raise ValueError(
            f"labels has {labels.shape[1]} columns, but target_names has "
            f"{len(target_names)} entries"
        )

    # Labels are assigned to the final timestep of each sliding window.
    current_data = data[:, :, -1]
    label_masks = labels.astype(bool)
    x = np.arange(len(current_data))
    output_path = Path(output_dir) if output_dir is not None else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    class_colors = _build_class_color_map(target_names)
    saved_paths.update(
        _plot_label_overlap_diagrams(
            label_masks,
            target_names,
            title,
            output_path,
            fold,
            split_name,
            label_kind,
            show,
            max_combinations,
        )
    )

    lowering_index = target_names.index("Lifting(lowering)")
    raising_index = target_names.index("Lifting(raising)")
    lowering = label_masks[:, lowering_index]
    raising = label_masks[:, raising_index]
    label_counts = label_masks.sum(axis=1)
    both_lifting = lowering & raising
    lifting_categories = [
        (
            "Neither lifting label",
            (label_counts <= 1) & ~(lowering | raising),
            "lightgrey",
            1,
            0.3,
        ),
        (
            "Lifting(lowering)",
            lowering & ~raising & (label_counts == 1),
            class_colors["Lifting(lowering)"],
            5,
            0.85,
        ),
        (
            "Lifting(raising)",
            raising & ~lowering & (label_counts == 1),
            class_colors["Lifting(raising)"],
            5,
            0.85,
        ),
        ("Both lifting labels", both_lifting, "tab:purple", 6, 0.9),
        # ("Other multi-label combinations", other_multiclass, "tab:pink", 6, 0.8),
    ]

    fig, axes = plt.subplots(n_features, 1, figsize=(15, 2 * n_features), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{title}: lifting direction", fontsize=16)
    lifting_legend_handles = [
        _legend_handle(
            f"{label} ({int(mask.sum())})",
            color,
            size,
            alpha,
        )
        for label, mask, color, size, alpha in lifting_categories
    ]

    for sensor_index, (ax, feature_name) in enumerate(zip(axes, sensor_cols)):
        feat_values = current_data[:, sensor_index]
        for label, mask, color, size, alpha in lifting_categories:
            ax.scatter(
                x[mask],
                feat_values[mask],
                color=color,
                s=size,
                alpha=alpha,
            )
        ax.set_ylabel(feature_name)
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[-1].set_xlabel("Row index after stepsampling")
    fig.legend(
        handles=lifting_legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        title="Lifting category",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    lifting_path = output_path / _timestrip_filename(
        fold, "lifting_direction", split_name, label_kind
    )
    fig.savefig(lifting_path, dpi=180, bbox_inches="tight")
    saved_paths["lifting_direction"] = lifting_path
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(n_features, 1, figsize=(15, 2 * n_features), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{title}: all classes", fontsize=16)

    no_label_mask = ~label_masks.any(axis=1)
    all_class_styles = [
        (class_name, label_masks[:, class_index], class_colors[class_name], 5, 0.7)
        for class_index, class_name in enumerate(target_names)
    ]
    all_class_legend_handles = (
        [
            _legend_handle(
                f"No class ({int(no_label_mask.sum())})",
                "lightgrey",
                1,
                0.3,
            )
        ]
        if no_label_mask.any()
        else []
    )
    all_class_legend_handles.extend(
        _legend_handle(
            f"{class_name} ({int(class_mask.sum())})",
            color,
            size,
            alpha,
        )
        for class_name, class_mask, color, size, alpha in all_class_styles
    )

    for sensor_index, (ax, feature_name) in enumerate(zip(axes, sensor_cols)):
        feat_values = current_data[:, sensor_index]
        if no_label_mask.any():
            ax.scatter(
                x[no_label_mask],
                feat_values[no_label_mask],
                color="lightgrey",
                s=1,
                alpha=0.3,
            )
        for class_name, class_mask, color, size, alpha in all_class_styles:
            ax.scatter(
                x[class_mask],
                feat_values[class_mask],
                color=color,
                s=size,
                alpha=alpha,
            )
        ax.set_ylabel(feature_name)
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[-1].set_xlabel("Row index after stepsampling")
    fig.legend(
        handles=all_class_legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        frameon=False,
        title="Classes",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    all_classes_path = output_path / _timestrip_filename(
        fold, "all_classes", split_name, label_kind
    )
    fig.savefig(all_classes_path, dpi=180, bbox_inches="tight")
    saved_paths["all_classes"] = all_classes_path
    if show:
        plt.show()
    plt.close(fig)

    return saved_paths
