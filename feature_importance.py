"""Permutation feature importance for the heart rate forecasting model."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from utils import heart_rate_scale, load_run, mae_in_bpm

RUN_DIRECTORY = Path("models") / "2026-10-22_11-16_gru64"
OUTPUT_DIRECTORY = Path("models") / "2026-10-22_11-16_gru64"
N_REPEATS = 5
RANDOM_SEED = 42

FEATURE_GROUPS = {
    "heart_rate": ["heart_rate"],
    "power": ["power"],
    "cadence": ["cadence"],
    "TT_10": ["TT_10"],
    "PP_10": ["PP_10"],
    "RF_10": ["RF_10"],
    "seconds_since_session_start": ["seconds_since_session_start"],
    "had_warmup": ["had_warmup"],
    "days_since_first_session": ["days_since_first_session"],
    "day_of_week": ["day_of_week_sin", "day_of_week_cos"],
    "time_of_day": ["time_of_day_sin", "time_of_day_cos"],
}


def mean_absolute_error_bpm(model, past, future, target, bpm_scale: float) -> float:
    """Predict with the model and return the overall MAE in bpm."""
    prediction = model.predict({"past": past, "future": future}, verbose=0)
    return float(mae_in_bpm(prediction, target, bpm_scale).mean())


def column_indices(columns: list[str], feature_names: list[str]) -> list[int]:
    """Return the indices of the given features that exist in the column list."""
    return [columns.index(name) for name in feature_names if name in columns]


def permute_features(data: np.ndarray, indices: list[int], rng: np.random.Generator) -> np.ndarray:
    """Shuffle the given feature channels across samples, keeping each sequence intact."""
    permuted = data.copy()
    if indices:
        order = rng.permutation(len(data))
        permuted[:, :, indices] = data[order][:, :, indices]
    return permuted


def permutation_importance(
        model, config: dict, test_data, bpm_scale: float, permute_past: bool = True,
        permute_future: bool = True,
) -> pd.DataFrame:
    """Measure how much the test error increases when each feature group is shuffled."""
    past, future, target = test_data["past"], test_data["future"], test_data["target"]
    rng = np.random.default_rng(RANDOM_SEED)
    baseline_error = mean_absolute_error_bpm(model, past, future, target, bpm_scale)

    results = []
    for group_name, feature_names in FEATURE_GROUPS.items():
        past_indices = column_indices(config["past_columns"], feature_names) if permute_past else []
        future_indices = column_indices(config["future_columns"], feature_names) if permute_future else []
        if not past_indices and not future_indices:
            continue

        error_increases = []
        for _ in range(N_REPEATS):
            permuted_past = permute_features(past, past_indices, rng)
            permuted_future = permute_features(future, future_indices, rng)
            error = mean_absolute_error_bpm(model, permuted_past, permuted_future, target, bpm_scale)
            error_increases.append(error - baseline_error)

        results.append({
            "feature": group_name,
            "importance_bpm": np.mean(error_increases),
            "std_bpm": np.std(error_increases),
        })

    print(f"Baseline MAE: {baseline_error:.2f} bpm")
    return pd.DataFrame(results).sort_values("importance_bpm")


def plot_importance(importance: pd.DataFrame, title: str):
    """Create a horizontal bar chart of the permutation importance."""
    figure = px.bar(
        importance, x="importance_bpm", y="feature", error_x="std_bpm", orientation="h",
        title=title,
        labels={"importance_bpm": "Increase in MAE when shuffled [bpm]", "feature": ""},
    )
    figure.update_layout(font_size=18, height=700)
    return figure


def main() -> None:
    model, config, scaler, test_data = load_run(RUN_DIRECTORY)
    bpm_scale = heart_rate_scale(scaler, config)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    scenarios = {
        "importance_all_inputs": ("Feature importance (past and future)", True, True),
        "importance_future_only": ("Feature importance (future input only)", False, True),
    }
    for filename, (title, permute_past, permute_future) in scenarios.items():
        importance = permutation_importance(
            model, config, test_data, bpm_scale, permute_past, permute_future
        )
        print(importance.to_string(index=False))
        figure = plot_importance(importance, title)
        figure.show()
        figure.write_html(OUTPUT_DIRECTORY / f"{filename}.html")


if __name__ == "__main__":
    main()
