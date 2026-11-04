import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import matthews_corrcoef


class AsymmetricFocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        eps=1e-8,
        pos_weight=None,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        positive_probabilities = probabilities
        negative_probabilities = 1.0 - probabilities

        if self.clip is not None and self.clip > 0:
            negative_probabilities = (negative_probabilities + self.clip).clamp(max=1.0)

        positive_loss = targets * torch.log(positive_probabilities.clamp(min=self.eps))
        negative_loss = (1.0 - targets) * torch.log(
            negative_probabilities.clamp(min=self.eps)
        )

        if self.pos_weight is not None:
            positive_loss = positive_loss * self.pos_weight

        positive_focus = torch.pow(1.0 - positive_probabilities, self.gamma_pos)
        negative_focus = torch.pow(1.0 - negative_probabilities, self.gamma_neg)
        focal_weight = targets * positive_focus + (1.0 - targets) * negative_focus

        return -(focal_weight * (positive_loss + negative_loss)).mean()


class MLPClassifierTorch:
    def __init__(
        self,
        classes,
        hidden_sizes=(128, 64),
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=256,
        epochs=20,
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
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
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
        self.pos_weight = None

        self.nn = torch.nn
        self.optim = torch.optim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_model(self, input_size, output_size):
        layers = [self.nn.Flatten()]
        previous_size = input_size

        for hidden_size in self.hidden_sizes:
            layers.extend(
                [
                    self.nn.Linear(previous_size, hidden_size),
                    self.nn.ReLU(),
                    self.nn.Dropout(self.dropout),
                ]
            )
            previous_size = hidden_size

        layers.append(self.nn.Linear(previous_size, output_size))
        return self.nn.Sequential(*layers).to(self.device)

    def _make_loader(self, X, y=None, shuffle=False):
        X_tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))

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
        pos_weight = np.clip(
            pos_weight, 1.0, self.max_pos_weight
        )  # avoid extreme weights that could destabilize training
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

        input_size = int(np.prod(X_train.shape[1:]))
        output_size = len(self.classes)
        self.training_parameters = {
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "input_shape": list(X_train.shape[1:]),
            "input_size": input_size,
            "output_size": output_size,
            "classes": list(self.classes),
        }
        self.write_settings_file()
        self.model = self._build_model(input_size, output_size)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val)
        criterion = self._make_criterion(y_train)
        optimizer = self.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(
            f"Start training MLP on {len(X_train)} samples, "
            f"{input_size} input features, device={self.device}..."
        )

        # for epoch in range(1, self.epochs + 1):
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training MLP"):
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

        print("Training done.")
        self._fit_validation_thresholds(X_val, y_val)
        self.write_settings_file()
        return self.predict(X_train)

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
        loss_path = self.log_dir / "mlp_losses.csv"
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

        plot_path = self.log_dir / "mlp_losses.png"
        temp_path = plot_path.with_suffix(".png.tmp")
        epochs = [entry["epoch"] for entry in self.loss_history]
        train_losses = [entry["train_loss"] for entry in self.loss_history]
        val_losses = [entry["val_loss"] for entry in self.loss_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Training loss")
        plt.plot(epochs, val_losses, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP training losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(temp_path, dpi=150, format="png")
        plt.close()
        temp_path.replace(plot_path)
        self._write_mcc_plot()

    def _write_mcc_plot(self):
        if not self.loss_history:
            return

        plot_path = self.log_dir / "mlp_mccs.png"
        temp_path = plot_path.with_suffix(".png.tmp")
        epochs = [entry["epoch"] for entry in self.loss_history]
        train_mccs = [entry["train_mcc"] for entry in self.loss_history]
        val_mccs = [entry["val_mcc"] for entry in self.loss_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_mccs, label="Training MCC")
        plt.plot(epochs, val_mccs, label="Validation MCC")
        plt.xlabel("Epoch")
        plt.ylabel("Mean per-class MCC")
        plt.title("MLP training MCC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(temp_path, dpi=150, format="png")
        plt.close()
        temp_path.replace(plot_path)

    def _write_settings_file(self):
        settings_path = self.log_dir / "mlp_settings.json"
        threshold = (
            self.threshold.tolist()
            if isinstance(self.threshold, np.ndarray)
            else self.threshold
        )
        settings = {
            "model": "MLPClassifierTorch",
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
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
            raise RuntimeError("MLPClassifierTorch must be trained before prediction.")

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
