"""Load, clean, split and scale the ride data and cut it into windows."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DataConfig

DATA_PATH = "./Dataframes/df_ml.csv"
RESAMPLE_INTERVAL = "5s"
MAX_INTERPOLATION_STEPS = 6  # 30 s
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

KNOWN_FEATURES = [
    "power", "cadence", "seconds_since_session_start",
    "time_of_day_sin", "time_of_day_cos",
    "had_warmup", "days_since_first_session",
    "day_of_week_sin", "day_of_week_cos",
    "TT_10", "PP_10", "RF_10",
]
TARGET = "heart_rate"
MODEL_COLUMNS = KNOWN_FEATURES + [TARGET]
WEATHER_COLUMNS = ["TT_10", "PP_10", "RF_10"]
SCALED_COLUMNS = [
    "power", "cadence", "heart_rate",
    "TT_10", "PP_10", "RF_10",
    "seconds_since_session_start", "days_since_first_session",
]


@dataclass
class ScaledSplits:
    """Scaled data splits plus the fitted scaler."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


@dataclass
class WindowedData:
    """Model-ready arrays for all splits."""
    past_train: np.ndarray
    future_train: np.ndarray
    target_train: np.ndarray
    past_validation: np.ndarray
    future_validation: np.ndarray
    target_validation: np.ndarray
    past_test: np.ndarray
    future_test: np.ndarray
    target_test: np.ndarray
    heart_rate_index: int


# ---------- cleaning ----------

def drop_rows_after_cutoff(data: pd.DataFrame, source_file: str, cutoff: str) -> pd.DataFrame:
    """Remove all rows of a source file whose timestamp is later than the cutoff (UTC)."""
    cutoff_timestamp = pd.Timestamp(cutoff, tz="UTC")
    rows_to_drop = (data["source_file"] == source_file) & (data.index > cutoff_timestamp)
    return data.loc[~rows_to_drop]


def drop_source_file(data: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Remove all rows belonging to a source file."""
    return data.loc[data["source_file"] != source_file]


def load_clean_data() -> pd.DataFrame:
    """Load the raw data and apply the manual session fixes."""
    data = pd.read_csv(DATA_PATH, encoding="utf-8",
                       parse_dates=["timestamp"], index_col="timestamp")
    data = drop_rows_after_cutoff(data, "2026-06-03-15-30-13.fit", "2026-06-03 14:57:23")
    data = drop_rows_after_cutoff(data, "2026-06-17-15-49-01.fit", "2026-06-17 15:29:20")
    data = drop_source_file(data, "2026-02-26-20-51-14.fit")
    return data.dropna(subset=[TARGET])


def resample_and_segment(data: pd.DataFrame) -> pd.DataFrame:
    """Resample per session, split at long gaps and interpolate short gaps."""
    resampled = (
        data.groupby("source_file")[MODEL_COLUMNS]
        .resample(RESAMPLE_INTERVAL)
        .mean()
        .reset_index(level="source_file")
    )

    is_missing = resampled["power"].isna()
    block_id = (is_missing != is_missing.shift()).cumsum()
    gap_length = is_missing.groupby(block_id).transform("size")
    is_long_gap = is_missing & (gap_length > MAX_INTERPOLATION_STEPS)

    long_gap_start = is_long_gap & ~is_long_gap.shift(fill_value=False)
    segment_number = long_gap_start.groupby(resampled["source_file"]).cumsum()
    resampled["segment_id"] = resampled["source_file"].astype(str) + "_" + segment_number.astype(str)

    clean = resampled[~is_long_gap].copy()
    clean[MODEL_COLUMNS] = (
        clean.groupby("segment_id")[MODEL_COLUMNS]
        .transform(lambda column: column.interpolate(limit_area="inside"))
    )
    clean[WEATHER_COLUMNS] = (
        clean.groupby("segment_id")[WEATHER_COLUMNS]
        .transform(lambda column: column.ffill().bfill())
    )
    return clean


# ---------- splitting & scaling ----------

def split_by_session(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically by session start."""
    sessions_in_order = (
        data.index.to_series().groupby(data["source_file"]).min().sort_values().index
    )
    number_of_sessions = len(sessions_in_order)
    train_end = int(number_of_sessions * TRAIN_FRACTION)
    validation_end = int(number_of_sessions * (TRAIN_FRACTION + VALIDATION_FRACTION))

    def select(sessions) -> pd.DataFrame:
        return data[data["source_file"].isin(sessions)]

    return (
        select(sessions_in_order[:train_end]),
        select(sessions_in_order[train_end:validation_end]),
        select(sessions_in_order[validation_end:]),
    )


def scale_split(split: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of the split with SCALED_COLUMNS standardized."""
    scaled = split.copy()
    scaled[SCALED_COLUMNS] = scaler.transform(split[SCALED_COLUMNS])
    return scaled


def prepare_scaled_splits() -> ScaledSplits:
    """Run the full preprocessing pipeline up to scaled train/validation/test splits."""
    clean = resample_and_segment(load_clean_data())
    train, validation, test = split_by_session(clean)

    scaler = StandardScaler().fit(train[SCALED_COLUMNS])  # training data only
    return ScaledSplits(
        train=scale_split(train, scaler),
        validation=scale_split(validation, scaler),
        test=scale_split(test, scaler),
        scaler=scaler,
    )


# ---------- windowing ----------

def create_windows(
        split: pd.DataFrame, data_config: DataConfig, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut every segment into (past, future, target) windows."""
    past_steps, future_steps = data_config.past_steps, data_config.future_steps
    window_steps = past_steps + future_steps
    past_windows, future_windows, target_windows = [], [], []

    for _, segment in split.groupby("segment_id"):
        if len(segment) < window_steps:
            continue

        segment = segment.sort_index()
        past_values = segment[list(data_config.past_columns)].to_numpy(dtype=np.float32)
        future_values = segment[list(data_config.future_columns)].to_numpy(dtype=np.float32)
        target_values = segment[data_config.target_column].to_numpy(dtype=np.float32)

        window_starts = np.arange(0, len(segment) - window_steps + 1, stride)
        past_indices = window_starts[:, None] + np.arange(past_steps)
        future_indices = window_starts[:, None] + past_steps + np.arange(future_steps)

        past_windows.append(past_values[past_indices])
        future_windows.append(future_values[future_indices])
        target_windows.append(target_values[future_indices])

    return (
        np.concatenate(past_windows),
        np.concatenate(future_windows),
        np.concatenate(target_windows),
    )


def build_windowed_data(splits: ScaledSplits, data_config: DataConfig) -> WindowedData:
    """Create windows for all splits according to the data configuration."""
    train = create_windows(splits.train, data_config, data_config.train_stride)
    validation = create_windows(splits.validation, data_config, data_config.evaluation_stride)
    test = create_windows(splits.test, data_config, data_config.evaluation_stride)

    return WindowedData(
        *train, *validation, *test,
        heart_rate_index=data_config.past_columns.index("heart_rate"),
    )