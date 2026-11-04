from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import tqdm
from numpy.lib._stride_tricks_impl import sliding_window_view
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def get_merged_data(meta_subset):
    merged = [row["data"] for _, row in meta_subset.iterrows()]
    return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()


def clean(df):
    cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df.reset_index(drop=True)


def _reduce_train_size(test_df: DataFrame, train_df: DataFrame,
                       validation_df: DataFrame, skipsize: int = 20) -> tuple[Any, Any, Any]:
    # Keep every 20th sample because datas is time series data
    train_df = train_df.iloc[::skipsize, :]
    test_df = test_df.iloc[::skipsize, :]
    validation_df = validation_df.iloc[::skipsize, :]
    return test_df, train_df, validation_df


def get_challenge_data_numpy(df, seq_len, sensor_cols, label_cols):

    data_values = df[sensor_cols].values
    label_values = df[label_cols].values

    # sliding_window_view is a numpy function
    X_view = sliding_window_view(data_values, window_shape=seq_len, axis=0)

    y_start_index = seq_len - 1
    y_view = label_values[y_start_index: y_start_index + len(X_view)]

    min_len = min(len(X_view), len(y_view))
    X_view = X_view[:min_len]
    y_view = y_view[:min_len]

    return X_view, y_view


class DataHandler:
    AVAILABLE_DERIVED_SENSOR_COLS = [
        "Acc.mag",
        "Gyro.mag",
        "Acc.x.diff",
        "Acc.y.diff",
        "Acc.z.diff",
        "Gyro.x.diff",
        "Gyro.y.diff",
        "Gyro.z.diff",
        "Baro.x.diff",
        "Baro.x.slope_5",
        "Acc.mag.rolling_std_5",
        "Gyro.mag.rolling_std_5",
    ]

    def __init__(self, config):
        self.config = config
        self.scaler = None
        filename = Path(self.config.data.dataset_file).name
        self.local_path = Path("data") / filename
        self.data = None
        self._add_derived_sensor_cols_to_config()
        self._load_data_set()

    def _add_derived_sensor_cols_to_config(self):
        derived_sensor_cols = getattr(self.config.data, "derived_sensor_cols", [])
        unknown_cols = sorted(set(derived_sensor_cols) - set(self.AVAILABLE_DERIVED_SENSOR_COLS))
        if unknown_cols:
            raise ValueError(
                "Unknown derived sensor column(s): " + ", ".join(unknown_cols)
            )

        for sensor_col in derived_sensor_cols:
            if sensor_col not in self.config.data.sensor_cols:
                self.config.data.sensor_cols.append(sensor_col)

    @staticmethod
    def _get_merged_data(meta_subset):
        merged = []
        for source_index, (_, row) in enumerate(meta_subset.iterrows()):
            frame = row["data"].copy()
            frame["_source_segment"] = source_index
            merged.append(frame)
        return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

    @staticmethod
    def _clean(df, label_cols):
        cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        df = df[~(df[label_cols] == 0).all(axis=1)]
        return df.reset_index(drop=True)

    def _load_data_set(self):
        """
        Checks if data exists locally. If not, downloads it.
        Then loads the data into memory.
        """
        # ensure the ./data/ directory exists
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

        # check if file exists
        if not self.local_path.exists():
            print(f"Dataset not found at {self.local_path}")
            print(f"Starting download from {self.config.data.download_url} ...")
            self._download_file(self.config.data.download_url, self.local_path)
        else:
            print(f"Local dataset found: {self.local_path}")

        # load dataset
        try:
            print("Loading data into memory...")
            df = pd.read_pickle(self.local_path)
            print(f"Data loaded.")
            self.data = df
        except Exception as e:
            print(f"Error loading file: {e}")
            raise e

    def _download_file(self, url, dest_path):
        """Helper method for download with progress bar (stream)"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise error for 404, 500 etc.

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte

            with open(dest_path, 'wb') as file, tqdm.tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
        except Exception as e:
            print(f"Download failed: {e}")
            # Clean up partially downloaded file
            if dest_path.exists():
                dest_path.unlink()
            raise e

    def _apply_superclass_mapping(self, df, mapping):
        df = df.copy()
        target_superclasses = list(set(mapping.values()))
        for super_name in target_superclasses:
            children = [child for child, parent in mapping.items() if parent == super_name]

            existing_children = [c for c in children if c in df.columns]

            if existing_children:
                df[super_name] = df[existing_children].max(axis=1)
            else:
                df[super_name] = 0

        cols_to_drop = [k for k in mapping.keys() if k in df.columns and k not in target_superclasses]

        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        return df

    @staticmethod
    def _fill_signal_gaps(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _get_source_groups(df: DataFrame) -> pd.Series:
        if "_source_segment" in df.columns:
            return df["_source_segment"]
        if "time" in df.columns:
            return df["time"].diff().lt(0).cumsum()
        return pd.Series(0, index=df.index)

    def _normalize_baro_baseline(self, df: DataFrame) -> DataFrame:
        mode = getattr(self.config.data, "baro_baseline_mode", "none")
        if mode in (None, "none") or "Baro.x" not in df.columns:
            return df

        df = df.copy()
        grouped_baro = df.groupby(self._get_source_groups(df), sort=False)["Baro.x"]

        if mode == "segment_first":
            baseline = grouped_baro.transform("first")
        elif mode == "segment_median":
            baseline = grouped_baro.transform("median")
        elif mode == "segment_mean":
            baseline = grouped_baro.transform("mean")
        else:
            raise ValueError(
                "baro_baseline_mode must be one of: none, segment_first, "
                "segment_median, segment_mean"
            )

        df["Baro.x"] = df["Baro.x"] - baseline
        return df

    def _normalize_baro_baseline_for_frames(
        self, test_df: DataFrame, train_df: DataFrame, validation_df: DataFrame
    ) -> tuple[DataFrame, DataFrame, DataFrame]:
        train_df = self._normalize_baro_baseline(train_df)
        test_df = self._normalize_baro_baseline(test_df)
        validation_df = self._normalize_baro_baseline(validation_df)
        return test_df, train_df, validation_df

    def _add_derived_sensor_features(self, df: DataFrame) -> DataFrame:
        df = df.copy()

        df["Acc.mag"] = np.sqrt(
            df["Acc.x"] ** 2 + df["Acc.y"] ** 2 + df["Acc.z"] ** 2
        )
        df["Gyro.mag"] = np.sqrt(
            df["Gyro.x"] ** 2 + df["Gyro.y"] ** 2 + df["Gyro.z"] ** 2
        )

        source_group = self._get_source_groups(df)

        diff_cols = [
            "Acc.x",
            "Acc.y",
            "Acc.z",
            "Gyro.x",
            "Gyro.y",
            "Gyro.z",
            "Baro.x",
        ]
        grouped = df.groupby(source_group, sort=False)
        for col in diff_cols:
            df[f"{col}.diff"] = self._fill_signal_gaps(grouped[col].diff())

        baro_window = grouped["Baro.x"].transform(
            lambda series: series.diff(5) / 5.0
        )
        df["Baro.x.slope_5"] = self._fill_signal_gaps(baro_window)

        df["Acc.mag.rolling_std_5"] = self._fill_signal_gaps(
            grouped["Acc.mag"].transform(
                lambda series: series.rolling(window=5, min_periods=2).std()
            )
        )
        df["Gyro.mag.rolling_std_5"] = self._fill_signal_gaps(
            grouped["Gyro.mag"].transform(
                lambda series: series.rolling(window=5, min_periods=2).std()
            )
        )

        return df

    def _add_derived_sensor_features_to_frames(
        self, test_df: DataFrame, train_df: DataFrame, validation_df: DataFrame
    ) -> tuple[DataFrame, DataFrame, DataFrame]:
        train_df = self._add_derived_sensor_features(train_df)
        test_df = self._add_derived_sensor_features(test_df)
        validation_df = self._add_derived_sensor_features(validation_df)
        return test_df, train_df, validation_df

    def get_data_loaders(self, skipsize: int = 20):
        train_df, validation_df, test_df, final_target_cols = self.load_dataframes(skipsize)

        # --- CREATE DATASETS ---
        test_x, test_y, train_x, train_y, val_x, val_y = self._make_default_datasets(final_target_cols, test_df,
                                                                                     train_df, validation_df)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def get_minirocket_data(self, skipsize: int = 20):
        seq_length: int = 300

        train_df, validation_df, test_df, final_target_cols = self.load_dataframes(skipsize)
        train_x, train_y = get_challenge_data_numpy(train_df, int(seq_length/skipsize),
                                                    self.config.data.sensor_cols, final_target_cols)
        val_x, val_y = get_challenge_data_numpy(validation_df, int(seq_length/skipsize),
                                                    self.config.data.sensor_cols, final_target_cols)
        test_x, test_y = get_challenge_data_numpy(test_df, int(seq_length/skipsize),
                                                    self.config.data.sensor_cols, final_target_cols)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols


    def load_dataframes(self, skipsize: int = 20) -> tuple[DataFrame, DataFrame, DataFrame, list]:
        """
        Returns: train_df, validation_df, test_df, final_data_columns

        Each dataframe has the following columns:
        'time', 'Acc.x', 'Acc.y', 'Acc.z', 'Gyro.x', 'Gyro.y', 'Gyro.z', 'Baro.x', 'No loading', 'Driving(straight)',
        'Driving(curve)', 'Lifting(raising)', 'Lifting(lowering)', 'Stationary processes', 'Turntable wrapping'

        Time is the timestep. Acc, Gyro and Baro are sensors, while the others are one-hot encoded labels.
        """
        test_df, train_df, validation_df = self._load_to_df()

        # --- CAP DATASET SIZES DURING DEVELOPMENT ---
        test_df, train_df, validation_df = _reduce_train_size(test_df, train_df, validation_df, skipsize)

        test_df, train_df, validation_df = self._clean_dataframes(test_df, train_df, validation_df)
        # test_df, train_df, validation_df = self._normalize_baro_baseline_for_frames(
        #     test_df, train_df, validation_df
        # )
        # test_df, train_df, validation_df = self._add_derived_sensor_features_to_frames(
        #     test_df, train_df, validation_df
        # )

        # --- Apply superclasses ---
        train_df = self._apply_superclass_mapping(train_df, self.config.data.superclass_mapping)
        validation_df = self._apply_superclass_mapping(validation_df, self.config.data.superclass_mapping)
        test_df = self._apply_superclass_mapping(test_df, self.config.data.superclass_mapping)
        unique_superclasses = set(self.config.data.superclass_mapping.values())
        final_target_cols = sorted(list(unique_superclasses))

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)
        # final_target_cols = self.config.data.label_cols

        # --- SCALING ---
        # self._apply_scaling(test_df, train_df, validation_df)
        return train_df, validation_df, test_df, final_target_cols

    def _make_default_datasets(self, final_target_cols: list,
                               test_df: DataFrame, train_df: DataFrame, validation_df):
        train_x, train_y = get_challenge_data_numpy(train_df, self.config.prep.seq_len,
                                                    self.config.data.sensor_cols, final_target_cols)
        val_x, val_y = get_challenge_data_numpy(validation_df, self.config.prep.seq_len,
                                                self.config.data.sensor_cols, final_target_cols)
        test_x, test_y = get_challenge_data_numpy(test_df, self.config.prep.seq_len, self.config.data.sensor_cols,
                                                  final_target_cols)
        return test_x, test_y, train_x, train_y, val_x, val_y

    def _apply_scaling(self, test_df, train_df, validation_df):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df[self.config.data.sensor_cols] = self.scaler.fit_transform(train_df[self.config.data.sensor_cols])
        validation_df[self.config.data.sensor_cols] = self.scaler.transform(validation_df[self.config.data.sensor_cols])
        test_df[self.config.data.sensor_cols] = self.scaler.transform(test_df[self.config.data.sensor_cols])

    def _clean_dataframes(self, test_df: DataFrame, train_df: DataFrame, validation_df) -> tuple[
        DataFrame, DataFrame, DataFrame]:
        train_df: pd.DataFrame = self._clean(train_df, self.config.data.label_cols)
        test_df = self._clean(test_df, self.config.data.label_cols)
        validation_df = self._clean(validation_df, self.config.data.label_cols)
        return test_df, train_df, validation_df

    def _load_to_df(self) -> tuple[DataFrame, DataFrame, DataFrame]:
        print("Starting data preparation...", flush=True)
        # df_final = self.load_data_set()

        test_mask = self.data['experiment'] == self.config.data.test_experiment_id
        validation_mask = self.data['experiment'] == self.config.data.validation_experiment_id
        train_metadata = self.data[~test_mask & ~validation_mask]
        test_metadata = self.data[test_mask]
        validation_metadata = self.data[validation_mask]

        train_df = self._get_merged_data(train_metadata)
        test_df = self._get_merged_data(test_metadata)
        validation_df = self._get_merged_data(validation_metadata)
        return test_df, train_df, validation_df
