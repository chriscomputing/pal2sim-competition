import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from classificators.mlp_classifier import AsymmetricFocalLoss


class CNN1DClassifierTorch:
    def __init__(
        self,
        classes,
        conv_channels=(32, 64, 128),
        kernel_sizes=(7, 5, 3),
        dense_size=128,
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        epochs=20,
        early_stopping=True,
        early_stopping_patience=8,
        early_stopping_min_delta=1e-4,
        early_stopping_monitor="val_mcc",
        threshold=0.5,
        random_state=42,
        max_pos_weight=20.0,
        loss_name="weighted_bce",
        focal_gamma_pos=0.0,
        focal_gamma_neg=4.0,
        focal_clip=0.05,
        focal_use_pos_weight=True,
        device=None,
        log_dir=None,
    ):
        self.classes = classes
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dense_size = dense_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_monitor = early_stopping_monitor
        self.threshold = threshold
        self.random_state = random_state
        self.max_pos_weight = max_pos_weight
        self.loss_name = loss_name
        self.focal_gamma_pos = focal_gamma_pos
        self.focal_gamma_neg = focal_gamma_neg
        self.focal_clip = focal_clip
        self.focal_use_pos_weight = focal_use_pos_weight
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.loss_history = []
        self.training_parameters = {}
        self.threshold_search = {}
        self.best_epoch = None
        self.best_monitor_value = None
        self.pos_weight = None
        self.channels_first = True
        self.mean = None
        self.std = None

        self.nn = torch.nn
        self.optim = torch.optim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_model(self, input_channels, output_size):
        if len(self.kernel_sizes) != len(self.conv_channels):
            raise ValueError("kernel_sizes and conv_channels must have the same length.")

        layers = []
        previous_channels = input_channels
        for out_channels, kernel_size in zip(self.conv_channels, self.kernel_sizes):
            layers.extend(
                [
                    self.nn.Conv1d(
                        previous_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    self.nn.BatchNorm1d(out_channels),
                    self.nn.ReLU(),
                    self.nn.Dropout(self.dropout),
                ]
            )
            previous_channels = out_channels

        layers.extend(
            [
                self.nn.AdaptiveAvgPool1d(1),
                self.nn.Flatten(),
                self.nn.Linear(previous_channels, self.dense_size),
                self.nn.ReLU(),
                self.nn.Dropout(self.dropout),
                self.nn.Linear(self.dense_size, output_size),
            ]
        )
        return self.nn.Sequential(*layers).to(self.device)

    def _infer_input_layout(self, X):
        if X.ndim != 3:
            raise ValueError(
                "CNN1DClassifierTorch expects 3D input shaped as "
                "(samples, channels, sequence_length) or "
                "(samples, sequence_length, channels)."
            )

        # The data handler returns channels-first tensors. Keep a conservative
        # channels-last fallback for long windows where the feature axis is clearly
        # the final, smaller dimension.
        self.channels_first = not (X.shape[1] > 64 and X.shape[2] <= 64)

    def _prepare_X(self, X):
        X_array = np.ascontiguousarray(X, dtype=np.float32)
        if X_array.ndim != 3:
            raise ValueError(
                "CNN1DClassifierTorch expects 3D input shaped as "
                "(samples, channels, sequence_length) or "
                "(samples, sequence_length, channels)."
            )
        if not self.channels_first:
            X_array = np.ascontiguousarray(np.transpose(X_array, (0, 2, 1)))
        return X_array

    def _normalize(self, X, fit=False):
        X_array = self._prepare_X(X)

        if fit:
            # Match the transformer classifier: normalize each channel across
            # samples and time steps, while preserving the 3D tensor layout.
            self.mean = np.mean(X_array, axis=(0, 2), keepdims=True)
            self.std = np.std(X_array, axis=(0, 2), keepdims=True) + 1e-8

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "CNN1DClassifierTorch normalization stats are not initialized."
            )

        return np.ascontiguousarray((X_array - self.mean) / self.std, dtype=np.float32)

    def _make_loader(self, X, y=None, shuffle=False):
        X_tensor = torch.from_numpy(self._normalize(X))

        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            dataset = TensorDataset(X_tensor, y_tensor)

        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def _get_pos_weight(self, y):
        positives = y.sum(axis=0)
        negatives = len(y) - positives
        pos_weight = np.divide(
            negatives,
            positives,
            out=np.ones_like(positives, dtype=np.float32),
            where=positives > 0,
        )
        pos_weight = np.clip(pos_weight, 1.0, self.max_pos_weight)
        self.pos_weight = pos_weight.astype(np.float32)
        return torch.tensor(self.pos_weight, dtype=torch.float32, device=self.device)

    def _make_criterion(self, y_train):
        pos_weight = self._get_pos_weight(y_train)

        if self.loss_name == "weighted_bce":
            return self.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if self.loss_name == "asymmetric_focal":
            return AsymmetricFocalLoss(
                gamma_pos=self.focal_gamma_pos,
                gamma_neg=self.focal_gamma_neg,
                clip=self.focal_clip,
                pos_weight=pos_weight if self.focal_use_pos_weight else None,
            )

        raise ValueError(f"Unknown loss_name: {self.loss_name}")

    def train(self, train_data, val_data):
        X_train, y_train = train_data
        X_val, y_val = val_data

        self._infer_input_layout(X_train)
        prepared_shape = self._normalize(X_train, fit=True).shape
        input_channels = prepared_shape[1]
        sequence_length = prepared_shape[2]
        output_size = len(self.classes)
        self.training_parameters = {
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "input_shape": list(X_train.shape[1:]),
            "prepared_input_shape": list(prepared_shape[1:]),
            "input_channels": input_channels,
            "sequence_length": sequence_length,
            "output_size": output_size,
            "classes": list(self.classes),
            "channels_first": self.channels_first,
        }
        self.write_settings_file()
        self.model = self._build_model(input_channels, output_size)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val)
        criterion = self._make_criterion(y_train)
        optimizer = self.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        print(
            f"Start training 1D CNN on {len(X_train)} samples, "
            f"{input_channels} channels x {sequence_length} steps, "
            f"device={self.device}..."
        )

        best_state = None
        best_value = None
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in tqdm(range(1, self.epochs + 1), desc="Training 1D CNN"):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_X)

            train_loss /= len(X_train)
            train_mcc = self._evaluate_mcc(train_loader)
            val_loss = self._evaluate_loss(val_loader, criterion)
            val_mcc = self._evaluate_mcc(val_loader)
            self.loss_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mcc": train_mcc,
                    "val_mcc": val_mcc,
                }
            )
            self.write_live_training_artifacts()

            monitor_value = self._get_monitor_value(self.loss_history[-1])
            improved = self._is_monitor_improved(monitor_value, best_value)
            if improved:
                best_value = monitor_value
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                self.early_stopping
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}; best "
                    f"{self.early_stopping_monitor} was {best_value:.6f} "
                    f"at epoch {best_epoch}."
                )
                break

        print("Training done.")
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_epoch = best_epoch
            self.best_monitor_value = float(best_value)
            print(
                f"Restored best 1D CNN checkpoint from epoch {best_epoch} "
                f"({self.early_stopping_monitor}={best_value:.6f})."
            )
        self._fit_validation_thresholds(X_val, y_val)
        self.write_settings_file()
        return self.predict(X_train)

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

    def write_live_training_artifacts(self):
        if self.log_dir is None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._write_loss_file()
        self._write_loss_plot()

    def write_settings_file(self):
        if self.log_dir is None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._write_settings_file()

    def _write_loss_file(self):
        loss_path = self.log_dir / "cnn1d_losses.csv"
        temp_path = loss_path.with_suffix(".csv.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            file.write("epoch,train_loss,val_loss,train_mcc,val_mcc\n")
            for entry in self.loss_history:
                file.write(
                    f"{entry['epoch']},{entry['train_loss']:.8f},"
                    f"{entry['val_loss']:.8f},{entry['train_mcc']:.8f},"
                    f"{entry['val_mcc']:.8f}\n"
                )
        temp_path.replace(loss_path)

    def _write_loss_plot(self):
        if not self.loss_history:
            return

        plot_path = self.log_dir / "cnn1d_losses.png"
        temp_path = plot_path.with_suffix(".png.tmp")
        epochs = [entry["epoch"] for entry in self.loss_history]
        train_losses = [entry["train_loss"] for entry in self.loss_history]
        val_losses = [entry["val_loss"] for entry in self.loss_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Training loss")
        plt.plot(epochs, val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("1D CNN training losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(temp_path, dpi=150, format="png")
        plt.close()
        temp_path.replace(plot_path)
        self._write_mcc_plot()

    def _write_mcc_plot(self):
        if not self.loss_history:
            return

        plot_path = self.log_dir / "cnn1d_mccs.png"
        temp_path = plot_path.with_suffix(".png.tmp")
        epochs = [entry["epoch"] for entry in self.loss_history]
        train_mccs = [entry["train_mcc"] for entry in self.loss_history]
        val_mccs = [entry["val_mcc"] for entry in self.loss_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_mccs, label="Training MCC")
        plt.plot(epochs, val_mccs, label="Validation MCC")
        plt.xlabel("Epoch")
        plt.ylabel("Mean per-class MCC")
        plt.title("1D CNN training MCC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(temp_path, dpi=150, format="png")
        plt.close()
        temp_path.replace(plot_path)

    def _write_settings_file(self):
        settings_path = self.log_dir / "cnn1d_settings.json"
        threshold = (
            self.threshold.tolist()
            if isinstance(self.threshold, np.ndarray)
            else self.threshold
        )
        settings = {
            "model": "CNN1DClassifierTorch",
            "conv_channels": list(self.conv_channels),
            "kernel_sizes": list(self.kernel_sizes),
            "dense_size": self.dense_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_monitor": self.early_stopping_monitor,
            "best_epoch": self.best_epoch,
            "best_monitor_value": self.best_monitor_value,
            "loss_name": self.loss_name,
            "max_pos_weight": self.max_pos_weight,
            "pos_weight": self.pos_weight.tolist()
            if self.pos_weight is not None
            else None,
            "focal_gamma_pos": self.focal_gamma_pos,
            "focal_gamma_neg": self.focal_gamma_neg,
            "focal_clip": self.focal_clip,
            "focal_use_pos_weight": self.focal_use_pos_weight,
            "threshold": threshold,
            "random_state": self.random_state,
            "device": str(self.device),
            "training_parameters": self.training_parameters,
            "threshold_search": self.threshold_search,
        }

        temp_path = settings_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(settings, file, indent=2)
        temp_path.replace(settings_path)

    def _evaluate_loss(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_X)
                total_samples += len(batch_X)

        return total_loss / total_samples if total_samples else 0.0

    def _evaluate_mcc(self, loader):
        y_true_batches = []
        y_pred_batches = []
        threshold = np.asarray(self.threshold)
        self.model.eval()

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                y_pred_batches.append((probabilities >= threshold).astype(int))
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

    def _predict_probabilities(self, X):
        if self.model is None:
            raise RuntimeError("CNN1DClassifierTorch must be trained before prediction.")

        loader = self._make_loader(X)
        probabilities = []
        self.model.eval()

        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                batch_probabilities = torch.sigmoid(self.model(batch_X))
                probabilities.append(batch_probabilities.cpu().numpy())

        return np.concatenate(probabilities, axis=0)

    def _fit_validation_thresholds(self, X_val, y_val):
        probabilities = self._predict_probabilities(X_val)
        candidate_thresholds = np.linspace(0.05, 0.95, 91)
        best_thresholds = np.full(probabilities.shape[1], float(self.threshold))
        best_mccs = np.zeros(probabilities.shape[1], dtype=np.float32)

        for class_index in range(probabilities.shape[1]):
            class_scores = probabilities[:, class_index]
            class_true = y_val[:, class_index]
            best_mcc = -1.0
            best_threshold = float(self.threshold)

            for threshold in candidate_thresholds:
                class_pred = (class_scores >= threshold).astype(int)
                mcc = matthews_corrcoef(class_true, class_pred)
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_threshold = threshold

            best_thresholds[class_index] = best_threshold
            best_mccs[class_index] = best_mcc

        self.threshold = best_thresholds
        self.threshold_search = {
            "method": "per_class_validation_mcc_grid_search",
            "grid_min": float(candidate_thresholds[0]),
            "grid_max": float(candidate_thresholds[-1]),
            "grid_step": float(candidate_thresholds[1] - candidate_thresholds[0]),
            "thresholds": best_thresholds.tolist(),
            "validation_mccs": best_mccs.tolist(),
            "mean_validation_mcc": float(best_mccs.mean()) if len(best_mccs) else 0.0,
        }
        print(
            "Validation thresholds: "
            + ", ".join(
                f"{class_name}={threshold:.2f}"
                for class_name, threshold in zip(self.classes, best_thresholds)
            )
        )

    def predict(self, X):
        probabilities = self._predict_probabilities(X)
        predictions = probabilities >= np.asarray(self.threshold)

        return predictions.astype(int)
