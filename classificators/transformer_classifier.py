from copy import deepcopy
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from classificators.mlp_classifier import AsymmetricFocalLoss

class TransformerClassifierTorch:
    def __init__(
            self,
            target_names,
            input_dim=7,
            seq_len=200,
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=0.2,
            learning_rate=1e-3,
            batch_size=512,
            epochs=20,
            early_stopping=True,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4,
            early_stopping_monitor="val_mcc",
            threshold=0.5,
            max_pos_weight=5.0,
            loss_name="weighted_bce",
            focal_gamma_pos=0.0,
            focal_gamma_neg=4.0,
            focal_clip=0.05,
            focal_use_pos_weight=True,
            exclusive_label_groups=None,
            allowed_label_combinations=None,
            log_dir=None,
    ):
        self.target_names = target_names
        self.num_classes = len(target_names)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_monitor = early_stopping_monitor
        self.threshold = threshold
        self.max_pos_weight = max_pos_weight
        self.loss_name = loss_name
        self.focal_gamma_pos = focal_gamma_pos
        self.focal_gamma_neg = focal_gamma_neg
        self.focal_clip = focal_clip
        self.focal_use_pos_weight = focal_use_pos_weight
        self.exclusive_label_groups = exclusive_label_groups or []
        self.exclusive_label_group_indices = self._resolve_exclusive_label_groups()
        self.allowed_label_combinations = allowed_label_combinations
        self.allowed_label_combination_masks = self._resolve_allowed_label_combinations(
            allowed_label_combinations
        )
        self.allowed_label_combination_set = self._make_allowed_label_combination_set(
            self.allowed_label_combination_masks
        )
        self.log_dir = Path(log_dir) if log_dir is not None else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

        # Normalization constants
        self.mean = None
        self.std = None
        self.loss_history = []
        self.best_epoch = None
        self.best_monitor_value = None

    def _build_model(self):
        class TransformerModel(nn.Module):
            def __init__(self, in_dim, sl, out_dim, dm, nh, nl, dr):
                super().__init__()
                self.projector = nn.Linear(in_dim, dm)
                self.pos_embed = nn.Parameter(torch.randn(1, sl, dm))

                layer = nn.TransformerEncoderLayer(
                    d_model=dm, nhead=nh, dim_feedforward=dm*4,
                    dropout=dr, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(layer, num_layers=nl)
                self.output = nn.Linear(dm, out_dim)

            def forward(self, x):
                # x: (Batch, 7, 200) -> (Batch, 200, 7)
                x = x.transpose(1, 2)
                x = self.projector(x) + self.pos_embed
                x = self.transformer(x)
                x = x.mean(dim=1) # Global Average Pooling
                return self.output(x)

        return TransformerModel(
            self.input_dim, self.seq_len, self.num_classes,
            self.d_model, self.nhead, self.num_layers, self.dropout
        )

    def _normalize(self, x, fit=False):
        # x is (N, 7, 200)
        if fit:
            # Calculate across Samples and Time dimensions for each of the 7 features
            self.mean = np.mean(x, axis=(0, 2), keepdims=True)
            self.std = np.std(x, axis=(0, 2), keepdims=True) + 1e-8

        return (x - self.mean) / self.std

    def _get_pos_weight(self, y):
        pos_counts = y.sum(axis=0)
        neg_counts = len(y) - pos_counts
        pos_weight = np.divide(
            neg_counts,
            pos_counts,
            out=np.ones_like(pos_counts, dtype=np.float32),
            where=pos_counts > 0,
        )
        pos_weight = np.clip(pos_weight, 1.0, self.max_pos_weight)
        return torch.tensor(pos_weight, dtype=torch.float32, device=self.device)

    def _make_criterion(self, y_train):
        pos_weight = self._get_pos_weight(y_train)

        if self.loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        if self.loss_name == "weighted_bce":
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if self.loss_name == "asymmetric_focal":
            return AsymmetricFocalLoss(
                gamma_pos=self.focal_gamma_pos,
                gamma_neg=self.focal_gamma_neg,
                clip=self.focal_clip,
                pos_weight=pos_weight if self.focal_use_pos_weight else None,
            )

        raise ValueError(f"Unknown loss_name: {self.loss_name}")

    def _resolve_exclusive_label_groups(self):
        label_to_index = {
            label_name: label_index
            for label_index, label_name in enumerate(self.target_names)
        }
        resolved_groups = []

        for group in self.exclusive_label_groups:
            missing_labels = [
                label_name
                for label_name in group
                if label_name not in label_to_index
            ]
            if missing_labels:
                raise ValueError(
                    "exclusive_label_groups contains unknown labels: "
                    + ", ".join(missing_labels)
                )

            group_indices = [label_to_index[label_name] for label_name in group]
            if len(group_indices) >= 2:
                resolved_groups.append(group_indices)

        return resolved_groups

    def _resolve_allowed_label_combinations(self, allowed_label_combinations):
        if allowed_label_combinations in (None, "auto"):
            return None

        label_to_index = {
            label_name: label_index
            for label_index, label_name in enumerate(self.target_names)
        }
        masks = []

        for combination in allowed_label_combinations:
            mask = np.zeros(self.num_classes, dtype=int)

            if len(combination) == self.num_classes and all(
                value in (0, 1, False, True) for value in combination
            ):
                mask = np.asarray(combination, dtype=int)
            else:
                missing_labels = [
                    label_name
                    for label_name in combination
                    if label_name not in label_to_index
                ]
                if missing_labels:
                    raise ValueError(
                        "allowed_label_combinations contains unknown labels: "
                        + ", ".join(missing_labels)
                    )

                for label_name in combination:
                    mask[label_to_index[label_name]] = 1

            masks.append(mask)

        if not masks:
            raise ValueError("allowed_label_combinations must not be empty.")

        return np.unique(np.vstack(masks), axis=0).astype(int)

    @staticmethod
    def _make_allowed_label_combination_set(combination_masks):
        if combination_masks is None:
            return None
        return {tuple(row.astype(int).tolist()) for row in combination_masks}

    def _fit_allowed_label_combinations_from_labels(self, y):
        y = np.asarray(y).astype(int)
        if y.ndim != 2 or y.shape[1] != self.num_classes:
            raise ValueError(
                f"allowed label combination fitting expected labels with shape "
                f"(n, {self.num_classes}), got {y.shape}"
            )

        self.allowed_label_combination_masks = np.unique(y, axis=0).astype(int)
        self.allowed_label_combination_set = self._make_allowed_label_combination_set(
            self.allowed_label_combination_masks
        )

        combination_names = [
            " + ".join(
                label_name
                for label_name, is_active in zip(self.target_names, combination)
                if is_active
            )
            or "No class"
            for combination in self.allowed_label_combination_masks
        ]
        print(
            "Allowed label combinations learned from training labels: "
            + "; ".join(combination_names)
        )

    def _apply_allowed_label_combinations(self, probabilities, predictions):
        if self.allowed_label_combination_masks is None:
            return predictions

        invalid_rows = np.array(
            [
                tuple(row.astype(int).tolist())
                not in self.allowed_label_combination_set
                for row in predictions
            ],
            dtype=bool,
        )
        if not np.any(invalid_rows):
            return predictions

        resolved_predictions = predictions.copy()
        eps = 1e-8
        clipped_probabilities = np.clip(probabilities[invalid_rows], eps, 1.0 - eps)
        log_probabilities = np.log(clipped_probabilities)
        log_inverse_probabilities = np.log1p(-clipped_probabilities)
        allowed_masks = self.allowed_label_combination_masks

        scores = (
            log_probabilities @ allowed_masks.T
            + log_inverse_probabilities @ (1 - allowed_masks).T
        )
        best_allowed_indices = np.argmax(scores, axis=1)
        invalid_row_indices = np.where(invalid_rows)[0]
        resolved_predictions[invalid_row_indices] = allowed_masks[best_allowed_indices]

        return resolved_predictions

    def _apply_exclusive_label_groups(self, probabilities, predictions):
        if not self.exclusive_label_group_indices:
            return predictions

        resolved_predictions = predictions.copy()
        for group_indices in self.exclusive_label_group_indices:
            group_predictions = resolved_predictions[:, group_indices]
            conflict_rows = group_predictions.sum(axis=1) > 1

            if not np.any(conflict_rows):
                continue

            group_probabilities = probabilities[conflict_rows][:, group_indices]
            winning_offsets = np.argmax(group_probabilities, axis=1)
            conflict_row_indices = np.where(conflict_rows)[0]

            resolved_predictions[np.ix_(conflict_row_indices, group_indices)] = 0
            for row_index, winning_offset in zip(conflict_row_indices, winning_offsets):
                winning_class_index = group_indices[winning_offset]
                resolved_predictions[row_index, winning_class_index] = 1

        return resolved_predictions

    def _resolve_prediction_constraints(self, probabilities, predictions):
        predictions = self._apply_exclusive_label_groups(probabilities, predictions)
        predictions = self._apply_allowed_label_combinations(probabilities, predictions)
        return predictions

    @staticmethod
    def _safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    @staticmethod
    def _append_csv_row(csv_path, fieldnames, row):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists or csv_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    def _predict_probabilities(self, X):
        self.model.eval()
        X_norm = self._normalize(X)

        probabilities = []
        with torch.no_grad():
            for i in range(0, len(X_norm), self.batch_size):
                batch_x = torch.FloatTensor(X_norm[i:i+self.batch_size]).to(self.device)
                logits = self.model(batch_x)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(probabilities, axis=0)

    def _probability_summary(self, values):
        if len(values) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "p05": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p95": 0.0,
            }

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
        }

    def write_probability_diagnostics(
            self,
            X,
            y,
            split_name,
            fold,
            log_dir=None,
            max_conflict_examples=200,
    ):
        output_dir = Path(log_dir) if log_dir is not None else self.log_dir
        if output_dir is None:
            return

        y = np.asarray(y).astype(int)
        probabilities = self._predict_probabilities(X)
        raw_predictions = (probabilities > self.threshold).astype(int)
        resolved_predictions = self._resolve_prediction_constraints(
            probabilities,
            raw_predictions,
        )

        summary_fieldnames = [
            "fold",
            "split",
            "class_name",
            "support",
            "prevalence",
            "raw_predicted_positives",
            "resolved_predicted_positives",
            "raw_precision",
            "raw_recall",
            "resolved_precision",
            "resolved_recall",
            "prob_mean",
            "prob_std",
            "prob_p05",
            "prob_p25",
            "prob_p50",
            "prob_p75",
            "prob_p95",
            "positive_prob_mean",
            "positive_prob_p50",
            "positive_prob_p95",
            "negative_prob_mean",
            "negative_prob_p50",
            "negative_prob_p95",
        ]
        summary_path = output_dir / "transformer_probability_diagnostics.csv"

        for class_index, class_name in enumerate(self.target_names):
            true_values = y[:, class_index]
            raw_values = raw_predictions[:, class_index]
            resolved_values = resolved_predictions[:, class_index]
            class_probabilities = probabilities[:, class_index]

            raw_tp = int(((true_values == 1) & (raw_values == 1)).sum())
            raw_fp = int(((true_values == 0) & (raw_values == 1)).sum())
            raw_fn = int(((true_values == 1) & (raw_values == 0)).sum())
            resolved_tp = int(((true_values == 1) & (resolved_values == 1)).sum())
            resolved_fp = int(((true_values == 0) & (resolved_values == 1)).sum())
            resolved_fn = int(((true_values == 1) & (resolved_values == 0)).sum())

            all_stats = self._probability_summary(class_probabilities)
            positive_stats = self._probability_summary(class_probabilities[true_values == 1])
            negative_stats = self._probability_summary(class_probabilities[true_values == 0])

            self._append_csv_row(
                summary_path,
                summary_fieldnames,
                {
                    "fold": fold,
                    "split": split_name,
                    "class_name": class_name,
                    "support": int(true_values.sum()),
                    "prevalence": self._safe_divide(int(true_values.sum()), len(y)),
                    "raw_predicted_positives": int(raw_values.sum()),
                    "resolved_predicted_positives": int(resolved_values.sum()),
                    "raw_precision": self._safe_divide(raw_tp, raw_tp + raw_fp),
                    "raw_recall": self._safe_divide(raw_tp, raw_tp + raw_fn),
                    "resolved_precision": self._safe_divide(
                        resolved_tp,
                        resolved_tp + resolved_fp,
                    ),
                    "resolved_recall": self._safe_divide(
                        resolved_tp,
                        resolved_tp + resolved_fn,
                    ),
                    "prob_mean": all_stats["mean"],
                    "prob_std": all_stats["std"],
                    "prob_p05": all_stats["p05"],
                    "prob_p25": all_stats["p25"],
                    "prob_p50": all_stats["p50"],
                    "prob_p75": all_stats["p75"],
                    "prob_p95": all_stats["p95"],
                    "positive_prob_mean": positive_stats["mean"],
                    "positive_prob_p50": positive_stats["p50"],
                    "positive_prob_p95": positive_stats["p95"],
                    "negative_prob_mean": negative_stats["mean"],
                    "negative_prob_p50": negative_stats["p50"],
                    "negative_prob_p95": negative_stats["p95"],
                },
            )

        group_fieldnames = [
            "fold",
            "split",
            "group_name",
            "sample_count",
            "raw_any_positive_count",
            "raw_conflict_count",
            "raw_conflict_rate",
            "true_multi_label_count",
            "resolved_changed_count",
        ]
        group_path = output_dir / "transformer_exclusive_group_diagnostics.csv"
        example_path = output_dir / "transformer_conflict_examples.csv"
        example_fieldnames = [
            "fold",
            "split",
            "group_name",
            "row_index",
            "true_labels",
            "raw_predicted_labels",
            "resolved_predicted_labels",
            "winning_label",
            "winning_probability",
            "probabilities",
        ]

        for group_indices in self.exclusive_label_group_indices:
            group_names = [self.target_names[index] for index in group_indices]
            group_name = " | ".join(group_names)
            group_raw = raw_predictions[:, group_indices]
            group_resolved = resolved_predictions[:, group_indices]
            group_true = y[:, group_indices]
            group_probabilities = probabilities[:, group_indices]
            conflict_rows = group_raw.sum(axis=1) > 1
            changed_rows = np.any(group_raw != group_resolved, axis=1)

            self._append_csv_row(
                group_path,
                group_fieldnames,
                {
                    "fold": fold,
                    "split": split_name,
                    "group_name": group_name,
                    "sample_count": len(y),
                    "raw_any_positive_count": int((group_raw.sum(axis=1) > 0).sum()),
                    "raw_conflict_count": int(conflict_rows.sum()),
                    "raw_conflict_rate": self._safe_divide(int(conflict_rows.sum()), len(y)),
                    "true_multi_label_count": int((group_true.sum(axis=1) > 1).sum()),
                    "resolved_changed_count": int(changed_rows.sum()),
                },
            )

            conflict_indices = np.where(conflict_rows)[0]
            if len(conflict_indices) == 0:
                continue

            margins = np.sort(group_probabilities[conflict_indices], axis=1)[:, -1] - np.sort(
                group_probabilities[conflict_indices],
                axis=1,
            )[:, -2]
            example_order = conflict_indices[np.argsort(margins)[:max_conflict_examples]]

            for row_index in example_order:
                row_probabilities = group_probabilities[row_index]
                winning_offset = int(np.argmax(row_probabilities))
                self._append_csv_row(
                    example_path,
                    example_fieldnames,
                    {
                        "fold": fold,
                        "split": split_name,
                        "group_name": group_name,
                        "row_index": int(row_index),
                        "true_labels": ";".join(
                            name
                            for name, value in zip(group_names, group_true[row_index])
                            if value == 1
                        ),
                        "raw_predicted_labels": ";".join(
                            name
                            for name, value in zip(group_names, group_raw[row_index])
                            if value == 1
                        ),
                        "resolved_predicted_labels": ";".join(
                            name
                            for name, value in zip(group_names, group_resolved[row_index])
                            if value == 1
                        ),
                        "winning_label": group_names[winning_offset],
                        "winning_probability": float(row_probabilities[winning_offset]),
                        "probabilities": ";".join(
                            f"{name}={probability:.6f}"
                            for name, probability in zip(group_names, row_probabilities)
                        ),
                    },
                )

        print(
            f"Wrote transformer probability diagnostics for fold {fold} "
            f"{split_name} to {output_dir}"
        )

    def _evaluate_loss(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_x)
                total_samples += len(batch_x)

        return total_loss / total_samples if total_samples else 0.0

    def _evaluate_mcc(self, loader):
        y_true_batches = []
        y_pred_batches = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions = (probabilities > self.threshold).astype(int)
                predictions = self._resolve_prediction_constraints(
                    probabilities,
                    predictions,
                )
                y_pred_batches.append(predictions)
                y_true_batches.append(batch_y.cpu().numpy().astype(int))

        if not y_true_batches:
            return 0.0

        y_true = np.concatenate(y_true_batches, axis=0)
        y_pred = np.concatenate(y_pred_batches, axis=0)
        mcc_scores = [
            matthews_corrcoef(y_true[:, class_index], y_pred[:, class_index])
            for class_index in range(y_true.shape[1])
        ]
        return float(np.mean(mcc_scores)) if mcc_scores else 0.0

    def _get_monitor_value(self, history_entry):
        if self.early_stopping_monitor == "val_mcc":
            return history_entry["val_mcc"]
        if self.early_stopping_monitor == "val_loss":
            return history_entry["val_loss"]
        raise ValueError(
            "early_stopping_monitor must be either 'val_mcc' or 'val_loss'."
        )

    def _is_monitor_improved(self, value, best_value):
        if best_value is None:
            return True
        if self.early_stopping_monitor == "val_loss":
            return value < best_value - self.early_stopping_min_delta
        return value > best_value + self.early_stopping_min_delta

    def train(self, train_data, val_data):
        X_train, y_train = train_data
        X_val, y_val = val_data

        if self.allowed_label_combinations == "auto":
            self._fit_allowed_label_combinations_from_labels(y_train)

        # 1. Preprocessing
        X_train_norm = self._normalize(X_train, fit=True)
        X_val_norm = self._normalize(X_val)

        # 1.5 add a bit of uncertainty to force generalization
        # Scale noise by a small percentage of each feature's standard deviation
        noise_level = 0.0
        # Calculate std per feature across the dataset
        feature_stds = np.std(X_train_norm, axis=(0, 2), keepdims=True)
        # Add jitter
        X_train_norm = X_train_norm + np.random.normal(0, noise_level * feature_stds, X_train_norm.shape)

        # 2. Prepare DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val_norm), torch.FloatTensor(y_val))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # 3. Loss & Optimizer
        criterion = self._make_criterion(y_train)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # 4. Training Loop
        best_state = None
        best_value = None
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            scheduler.step()
            train_loss /= len(train_ds)
            val_loss = self._evaluate_loss(val_loader, criterion)
            train_mcc = self._evaluate_mcc(train_loader)
            val_mcc = self._evaluate_mcc(val_loader)
            history_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mcc": train_mcc,
                "val_mcc": val_mcc,
            }
            self.loss_history.append(history_entry)

            monitor_value = self._get_monitor_value(history_entry)
            improved = self._is_monitor_improved(monitor_value, best_value)
            if improved:
                best_value = monitor_value
                best_epoch = epoch + 1
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                f"train_mcc: {train_mcc:.4f} | val_mcc: {val_mcc:.4f}"
            )

            if (
                self.early_stopping
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch + 1}; best "
                    f"{self.early_stopping_monitor} was {best_value:.6f} "
                    f"at epoch {best_epoch}."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_epoch = best_epoch
            self.best_monitor_value = float(best_value)
            print(
                f"Restored best transformer checkpoint from epoch {best_epoch} "
                f"({self.early_stopping_monitor}={best_value:.6f})."
            )

        # Return predictions on training data for logging (as hard binary)
        return self.predict(X_train)

    def predict(self, X):
        self.model.eval()
        X_norm = self._normalize(X)

        preds = []
        # Process in batches to avoid OOM on 2M samples
        with torch.no_grad():
            for i in range(0, len(X_norm), self.batch_size):
                batch_x = torch.FloatTensor(X_norm[i:i+self.batch_size]).to(self.device)
                logits = self.model(batch_x)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                binary_out = (probabilities > self.threshold).astype(int)
                binary_out = self._resolve_prediction_constraints(
                    probabilities,
                    binary_out,
                )
                preds.append(binary_out)

        return np.concatenate(preds, axis=0)
