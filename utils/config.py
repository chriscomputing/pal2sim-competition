from dataclasses import dataclass, field
from typing import List, Dict


# --- 1. Data Configuration ---
@dataclass
class DataConfig:
    dataset_file: str = "cps_data_multi_label.pkl"
    download_url: str = "REPLACE_HERE"
    sensor_cols: List[str] = field(
        default_factory=lambda: [
            "Acc.x",
            "Acc.y",
            "Acc.z",
            "Gyro.x",
            "Gyro.y",
            "Gyro.z",
            "Baro.x",
        ]
    )
    derived_sensor_cols: List[str] = field(
        # default_factory=lambda: ["Baro.x.diff", "Baro.x.slope_5", "Acc.mag", "Gyro.mag"]
        default_factory=lambda: []
    )
    baro_baseline_mode: str = "segment_median"

    gyro_cols: List[str] = field(default_factory=lambda: ["Gyro.x", "Gyro.y", "Gyro.z"])
    test_experiment_id: int = 1
    validation_experiment_id: int = 2

    # Label Configs
    label_cols: List[str] = field(
        default_factory=lambda: [
            "Driving(straight)",
            "Driving(curve)",
            "Lifting(raising)",
            "Lifting(lowering)",
            "Standing",
            "Docking",
            "Forks(entering or leaving front)",
            "Forks(entering or leaving side)",
            "Wrapping",
            "Wrapping(preparation)",
        ]
    )

    superclass_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "Driving(curve)": "Driving(curve)",
            "Driving(straight)": "Driving(straight)",
            "Lifting(lowering)": "Lifting(lowering)",
            "Lifting(raising)": "Lifting(raising)",
            "Wrapping": "Turntable wrapping",
            "Wrapping(preparation)": "Stationary processes",
            "Docking": "Stationary processes",
            "Forks(entering or leaving front)": "Stationary processes",
            "Forks(entering or leaving side)": "Stationary processes",
            "Standing": "Stationary processes",
        }
    )


# --- 2. Preprocessing Configuration ---
@dataclass
class PreprocessingConfig:
    original_freq: int = 2000
    target_freq: int = 80
    seq_len_multiplier: float = 2
    filter_order: int = 8
    is_without_ds: bool = False

    seq_len: int = field(init=False)
    ds_factor: int = field(init=False)

    def __post_init__(self):
        self.seq_len = int(self.target_freq * self.seq_len_multiplier)
        self.ds_factor = (
            int(self.original_freq / self.target_freq) if self.target_freq > 0 else 1
        )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    prep: PreprocessingConfig = field(default_factory=PreprocessingConfig)
