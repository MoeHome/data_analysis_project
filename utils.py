from pathlib import Path
import json
import joblib

import numpy as np
from tensorflow import keras


def persistence_forecast(past: np.ndarray, heart_rate_index: int, future_steps: int) -> np.ndarray:
    """Naive baseline: repeat the last observed heart rate for every future step."""
    last_heart_rate = past[:, -1, heart_rate_index]
    return np.repeat(last_heart_rate[:, None], future_steps, axis=1)


def mae_in_bpm(prediction: np.ndarray, target: np.ndarray, heart_rate_std: float) -> np.ndarray:
    """Mean absolute error per forecast step, converted back to bpm."""
    return np.abs(prediction - target).mean(axis=0) * heart_rate_std


def column_scale(scaler, scaled_columns: list[str], column: str) -> float:
    """Return the standard deviation the scaler used for the given column."""
    return float(scaler.scale_[scaled_columns.index(column)])


def to_original_unit(values: np.ndarray, scaler, scaled_columns: list[str], column: str) -> np.ndarray:
    """Invert the standard scaling for a single column (e.g. watts or bpm)."""
    column_index = scaled_columns.index(column)
    return values * scaler.scale_[column_index] + scaler.mean_[column_index]


def heart_rate_scale(scaler, config: dict) -> float:
    """Return the factor that converts scaled heart rate differences back to bpm."""
    return column_scale(scaler, config["scaled_columns"], "heart_rate")


def power_to_watts(scaled_power: np.ndarray, scaler, config: dict) -> np.ndarray:
    """Convert scaled power values back to watts."""
    return to_original_unit(scaled_power, scaler, config["scaled_columns"], "power")


def mae_in_bpm(prediction: np.ndarray, target: np.ndarray, bpm_scale: float) -> np.ndarray:
    """Mean absolute error per forecast step in bpm.

    bpm_scale must be the heart rate scale from heart_rate_scale(), not a column index.
    """
    return np.abs(prediction.reshape(target.shape) - target).mean(axis=0) * bpm_scale


def per_window_mae(prediction: np.ndarray, target: np.ndarray, bpm_scale: float) -> np.ndarray:
    """Mean absolute error per window in bpm, averaged over the forecast horizon."""
    return np.abs(prediction.reshape(target.shape) - target).mean(axis=1) * bpm_scale


def load_run(run_directory: Path) -> tuple:
    """Load the trained model, config, scaler and test data of a run."""
    config = json.loads((run_directory / "config.json").read_text())
    model = keras.models.load_model(run_directory / "best_model.keras")
    scaler = joblib.load(run_directory / "scaler.joblib")
    test_data = np.load(run_directory / "test_data.npz")
    return model, config, scaler, test_data


def step_seconds(config: dict) -> int:
    """Return the duration of one time step in seconds."""
    return config.get("step_seconds", 5)  # fallback for older runs
