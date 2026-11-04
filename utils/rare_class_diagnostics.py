import csv

import numpy as np
from sklearn.metrics import matthews_corrcoef


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _class_metrics(y_true, y_pred, class_index):
    true_values = y_true[:, class_index].astype(int)
    pred_values = y_pred[:, class_index].astype(int)

    tp = int(((true_values == 1) & (pred_values == 1)).sum())
    fp = int(((true_values == 0) & (pred_values == 1)).sum())
    fn = int(((true_values == 1) & (pred_values == 0)).sum())
    tn = int(((true_values == 0) & (pred_values == 0)).sum())
    support = int(true_values.sum())
    predicted_positives = int(pred_values.sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    if len(np.unique(true_values)) < 2 and len(np.unique(pred_values)) < 2:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(true_values, pred_values))

    return {
        "support": support,
        "predicted_positives": predicted_positives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def _rare_class_issue(train_prevalence, metrics):
    if train_prevalence > 0.05:
        return ""
    if metrics["support"] == 0:
        return "no positives in test split"
    if metrics["predicted_positives"] == 0:
        return "rare class never predicted"
    if metrics["recall"] < 0.5:
        return "rare class recall below 0.50"
    if metrics["mcc"] <= 0:
        return "rare class MCC <= 0"
    return ""


def write_rare_class_diagnostics(
    run_dir,
    fold,
    class_names,
    train_y,
    val_y,
    test_y,
    train_pred_y,
    test_pred_y,
):
    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)
    test_y = np.asarray(test_y)
    train_pred_y = np.asarray(train_pred_y)
    test_pred_y = np.asarray(test_pred_y)

    csv_path = run_dir / "rare_class_diagnostics.csv"
    fieldnames = [
        "fold",
        "class_name",
        "train_support",
        "train_prevalence",
        "validation_support",
        "validation_prevalence",
        "test_support",
        "test_prevalence",
        "test_predicted_positives",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_mcc",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
        "train_predicted_positives",
        "train_recall",
        "train_mcc",
        "rare_class",
        "rare_class_issue",
    ]

    rows = []
    for class_index, class_name in enumerate(class_names):
        train_support = int(train_y[:, class_index].sum())
        val_support = int(val_y[:, class_index].sum())
        test_metrics = _class_metrics(test_y, test_pred_y, class_index)
        train_metrics = _class_metrics(train_y, train_pred_y, class_index)
        train_prevalence = _safe_divide(train_support, len(train_y))
        rare_class = train_prevalence <= 0.05

        rows.append(
            {
                "fold": fold,
                "class_name": class_name,
                "train_support": train_support,
                "train_prevalence": train_prevalence,
                "validation_support": val_support,
                "validation_prevalence": _safe_divide(val_support, len(val_y)),
                "test_support": test_metrics["support"],
                "test_prevalence": _safe_divide(test_metrics["support"], len(test_y)),
                "test_predicted_positives": test_metrics["predicted_positives"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_mcc": test_metrics["mcc"],
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_tn": test_metrics["tn"],
                "train_predicted_positives": train_metrics["predicted_positives"],
                "train_recall": train_metrics["recall"],
                "train_mcc": train_metrics["mcc"],
                "rare_class": rare_class,
                "rare_class_issue": _rare_class_issue(train_prevalence, test_metrics),
            }
        )

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)

    rare_rows = [row for row in rows if row["rare_class"]]
    issue_rows = [row for row in rare_rows if row["rare_class_issue"]]
    summary_path = run_dir / "rare_class_summary.txt"
    with summary_path.open("a", encoding="utf-8") as file:
        file.write(f"Fold {fold}\n")
        file.write("Rare class threshold: train prevalence <= 5%\n")
        file.write(f"Rare classes: {len(rare_rows)} / {len(rows)}\n")
        if issue_rows:
            file.write("Potential rare-class problems:\n")
            for row in issue_rows:
                file.write(
                    "- {class_name}: train_prevalence={train_prevalence:.4f}, "
                    "test_support={test_support}, predicted={test_predicted_positives}, "
                    "recall={test_recall:.3f}, mcc={test_mcc:.3f} "
                    "({rare_class_issue})\n".format(**row)
                )
        else:
            file.write("No rare-class warning triggered for this fold.\n")
        file.write("\n")

    print(f"Wrote rare-class diagnostics to {csv_path}")
