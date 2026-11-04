import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import matthews_corrcoef, multilabel_confusion_matrix


def plot_class_distribution(labels, label_names):
    counts = labels.sum(axis=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_names, counts, color='skyblue', edgecolor='black')
    plt.title("Class Distribution", fontsize=16)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_per_class_confusion(y_true, y_pred, class_names):
    # Gibt ein Array der Form (N_classes, 2, 2) zurück
    cms = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Anpassen je nach Klassenanzahl (hier 2x3 für 6 Klassen)
    axes = axes.flatten()

    for i, (cm, name) in enumerate(zip(cms, class_names)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        axes[i].set_title(f"Class: {name}")
        axes[i].set_ylabel("True")
        axes[i].set_xlabel("Predicted")

    plt.tight_layout()
    plt.show()

def calculate_mcc_multilabel(y_true, y_pred):
    n_classes = y_true.shape[1]
    mcc_scores = []

    for i in range(n_classes):
        # MCC for each class treated as binary classification
        score = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        mcc_scores.append(score)
    return np.mean(mcc_scores)