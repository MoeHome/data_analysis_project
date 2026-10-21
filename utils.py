import numpy as np


def persistence_forecast(past: np.ndarray, heart_rate_index: int, future_steps: int) -> np.ndarray:
    """Naive baseline: repeat the last observed heart rate for every future step."""
    last_heart_rate = past[:, -1, heart_rate_index]
    return np.repeat(last_heart_rate[:, None], future_steps, axis=1)


def mae_in_bpm(prediction: np.ndarray, target: np.ndarray, heart_rate_std: float) -> np.ndarray:
    """Mean absolute error per forecast step, converted back to bpm."""
    return np.abs(prediction - target).mean(axis=0) * heart_rate_std
