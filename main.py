import json
import random
from pathlib import Path

import numpy as np

# from classificators.mini_rocket import MiniRocketClassifier
from classificators.cnn1d_classifier import CNN1DClassifierTorch
from classificators.mlp_classifier import MLPClassifierTorch
from classificators.transformer_classifier import TransformerClassifierTorch
from utils.config import Config
from utils.plotting import plot_gyroscope_3d, plot_acceleration_3d
from utils.timestrip_plotting import plot_label_diagnostics
from data_handler import DataHandler
from classificators.dummy_classifier import DummyClassifier
from classificators.random_forest_classifier import RandomForestClassifierSK
from utils.utils import (
    calculate_mcc_multilabel,
    plot_per_class_confusion,
    plot_class_distribution,
)
from utils.rare_class_diagnostics import write_rare_class_diagnostics


def create_run_log_dir(base_dir="logs"):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    run_numbers = []
    for path in base_path.glob("run_*"):
        if path.is_dir():
            try:
                run_numbers.append(int(path.name.removeprefix("run_")))
            except ValueError:
                pass

    run_dir = base_path / f"run_{max(run_numbers, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def write_score_log(run_dir, model_name, train_mccs, test_mccs):
    train_mccs = [float(score) for score in train_mccs]
    test_mccs = [float(score) for score in test_mccs]
    scores = {
        "model_name": model_name,
        "train_mccs": train_mccs,
        "test_mccs": test_mccs,
        "total_train_mcc": sum(train_mccs) / len(train_mccs) if train_mccs else None,
        "total_test_mcc": sum(test_mccs) / len(test_mccs) if test_mccs else None,
    }

    with (run_dir / "scores.json").open("w", encoding="utf-8") as file:
        json.dump(scores, file, indent=2)

    with (run_dir / "scores.txt").open("w", encoding="utf-8") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Training scores for each run: {train_mccs}\n")
        file.write(f"Test scores for each run: {test_mccs}\n")
        file.write(f"Total training score: {scores['total_train_mcc']}\n")
        file.write(f"Total test score: {scores['total_test_mcc']}\n")


if __name__ == "__main__":
    config = Config()
    model_name = "transformer"  # options: "dummy", "random_forest", "mlp", "cnn1d", "transformer"
    exclusive_label_groups = [
        ["Lifting(lowering)", "Lifting(raising)"],
        ["Driving(straight)", "Driving(curve)"],
    ]
    run_dir = create_run_log_dir()
    print(f"Writing run logs to {run_dir}")

    # Seeding
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    # if you use any other libraries that require seeding, set it here as well (e.g., torch.manual_seed(SEED) for PyTorch)
    # -> your results should be reproducible across runs with the same seed

    val_mccs = []
    test_mccs = []
    train_mccs = []
    lr_histories_by_fold = {}

    # load data
    datahandler = DataHandler(config=config)

    # Leave-one-out: EXPERIMENT_ID = 1..4
    for fold in range(1, 2):
        print(f"\n--- Fold {fold}/4 | EXPERIMENT_ID={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        # validation hat to be different from test
        datahandler.config.data.validation_experiment_id = val_id

        skipsize: int = 25
        # train, val, test, target_vals = datahandler.get_data_loaders(skipisze)
        # train, val, test, target_vals = datahandler.load_dataframes(skipsize)
        train, val, test, target_vals = datahandler.get_minirocket_data(skipsize)
        print(f"data shape: {train[0].shape}")
        print(f"label shape: {train[1].shape}")

        label_diagnostics_dir = run_dir / "label_diagnostics" / f"fold_{fold}"

        # just to get an insight into the data
        plot_label_diagnostics(
            train[0],
            train[1],
            target_vals,
            config.data.sensor_cols,
            title=f"Fold {fold} training true labels",
            output_dir=label_diagnostics_dir,
            fold=fold,
            split_name="training",
            label_kind="true",
        )
        # plot_acceleration_3d(train.sample(n=5000))
        # plot_gyroscope_3d(train.sample(n=5000))
        # plot_class_distribution(train[1], target_vals)
        # plot_class_distribution(val[1], target_vals)
        # plot_class_distribution(test[1], target_vals)

        try:
            if model_name == "dummy":
                model = DummyClassifier(target_vals)
            elif model_name == "random_forest":
                model = RandomForestClassifierSK(target_vals)
            elif model_name == "mlp":
                model = MLPClassifierTorch(
                    target_vals,
                    hidden_sizes=(64, 32),
                    dropout=0.4,
                    learning_rate=5e-3,
                    batch_size=512,
                    epochs=50,
                    max_pos_weight=10.0,
                    loss_name="asymmetric_focal",
                    threshold=0.5,
                    log_dir=run_dir,
                )
            elif model_name == "cnn1d":
                model = CNN1DClassifierTorch(
                    target_vals,
                    conv_channels=(16, 32, 64),
                    kernel_sizes=(7, 5, 3),
                    dense_size=64,
                    dropout=0.5,
                    learning_rate=5e-4,
                    weight_decay=1e-4,
                    batch_size=512,
                    epochs=50,
                    early_stopping=True,
                    early_stopping_patience=8,
                    early_stopping_monitor="val_mcc",
                    max_pos_weight=5.0,
                    loss_name="weighted_bce",
                    threshold=0.5,
                    log_dir=run_dir,
                )
            elif model_name == "transformer":
                model = TransformerClassifierTorch(
                    target_names=target_vals,
                    input_dim=train[0].shape[1],
                    seq_len=train[0].shape[2],
                    d_model=128,  # Increased for your large dataset
                    nhead=8,
                    num_layers=3,
                    batch_size=1024,  # Large batch for speed with 2M samples
                    epochs=20,
                    max_pos_weight=5.0,
                    threshold=0.5,
                    loss_name="weighted_bce",
                    exclusive_label_groups=exclusive_label_groups,
                    allowed_label_combinations="auto",
                    log_dir=run_dir,
                )
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            # Here, we use the validation set to follow good machine learning practice, which is particularly relevant for evaluation during training.
            # However, the validation data can also be incorporated directly into the model training if necessary.
            print("Training model...")
            # Note: Any kind of preprocessing, data augmentation or feature engineering should be done within the model.train() function
            # so it's capsuled within the model class (see RandomForestClassifierSK for an example)
            train_y_hat = model.train(train, val)
            print("Evaluating model...")
            test_y_hat = model.predict(test[0])

            if hasattr(model, "write_probability_diagnostics"):
                model.write_probability_diagnostics(
                    train[0],
                    train[1],
                    split_name="train",
                    fold=fold,
                    log_dir=run_dir,
                )
                model.write_probability_diagnostics(
                    val[0],
                    val[1],
                    split_name="validation",
                    fold=fold,
                    log_dir=run_dir,
                )
                model.write_probability_diagnostics(
                    test[0],
                    test[1],
                    split_name="test",
                    fold=fold,
                    log_dir=run_dir,
                )

            plot_label_diagnostics(
                train[0],
                train_y_hat,
                target_vals,
                config.data.sensor_cols,
                title=f"Fold {fold} training predicted labels",
                output_dir=label_diagnostics_dir,
                fold=fold,
                split_name="training",
                label_kind="predicted",
            )
            plot_label_diagnostics(
                test[0],
                test[1],
                target_vals,
                config.data.sensor_cols,
                title=f"Fold {fold} test true labels",
                output_dir=label_diagnostics_dir,
                fold=fold,
                split_name="test",
                label_kind="true",
            )
            plot_label_diagnostics(
                test[0],
                test_y_hat,
                target_vals,
                config.data.sensor_cols,
                title=f"Fold {fold} test predicted labels",
                output_dir=label_diagnostics_dir,
                fold=fold,
                split_name="test",
                label_kind="predicted",
            )
            # optional, for more insight, plot the per-class-confusion-matrix for the test set
            # plot_per_class_confusion(test[1], predicted_y, target_vals)

            # Note: The MCC might be negative for a fold since its a correlation coefficient -> Range -1 to +1
            # +1 indicates perfect correlation, 0 indicates no correlation, and -1 indicates perfect inverse correlation.
            # Since we average over classes, it can happen that some classes have a
            # negative MCC while others have a positive MCC, resulting that scores balance each other out
            # Check your individual scores to see if they are reasonable

            test_mcc = calculate_mcc_multilabel(test_y_hat, test[1])
            test_mccs.append(test_mcc)
            train_mcc = calculate_mcc_multilabel(train_y_hat, train[1])
            train_mccs.append(train_mcc)
            write_rare_class_diagnostics(
                run_dir=run_dir,
                fold=fold,
                class_names=target_vals,
                train_y=train[1],
                val_y=val[1],
                test_y=test[1],
                train_pred_y=train_y_hat,
                test_pred_y=test_y_hat,
            )
            write_score_log(run_dir, model_name, train_mccs, test_mccs)

        except Exception as e:
            print(f"Fold {fold} failed with error: {e}")
            raise e
        print("--- End of Fold ---")

    avg_mcc = sum(test_mccs) / len(test_mccs)
    print("Training scores for each run: ", train_mccs)
    print("Test scores for each run: ", test_mccs)
    print("\nTotal training score: ", sum(train_mccs) / len(train_mccs))
    print("Total test score:", avg_mcc)
    write_score_log(run_dir, model_name, train_mccs, test_mccs)
