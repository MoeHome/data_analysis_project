"""Evaluate the model on test windows with large changes in planned power."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import heart_rate_scale, load_run, step_seconds, mae_in_bpm, per_window_mae, persistence_forecast, \
    power_to_watts

RUN_DIRECTORY = Path("models") / "2026-10-22_11-16_gru64"
OUTPUT_DIRECTORY = Path("models") / "2026-10-22_11-16_gru64"
COMPARISON_SECONDS = 60  # length of the power averaging windows
TOP_FRACTION = 0.20
MIN_POWER_WATTS = 150  # at least one of the two averages must exceed this value


def mean_power_watts(past, future, scaler, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean power of the last COMPARISON_SECONDS of the past window and of the forecast horizon."""
    comparison_steps = COMPARISON_SECONDS // step_seconds(config)
    past_power = past[:, -comparison_steps:, config["past_columns"].index("power")].mean(axis=1)
    future_power = future[:, :, config["future_columns"].index("power")].mean(axis=1)
    return power_to_watts(past_power, scaler, config), power_to_watts(future_power, scaler, config)


def build_subsets(past_power: np.ndarray, future_power: np.ndarray) -> dict[str, np.ndarray]:
    """Select the largest power changes among windows where at least one average is high enough."""
    power_change = future_power - past_power
    is_relevant = np.maximum(past_power, future_power) > MIN_POWER_WATTS
    threshold = np.quantile(np.abs(power_change[is_relevant]), 1 - TOP_FRACTION)
    is_top = is_relevant & (np.abs(power_change) >= threshold)

    print(f"relevant windows: {is_relevant.sum()} of {len(power_change)} | "
          f"top threshold: {threshold:.0f} W | top windows: {is_top.sum()}")

    label = f"top {TOP_FRACTION:.0%}"
    return {
        label: is_top,
        f"{label} increase": is_top & (power_change > 0),
        f"{label} decrease": is_top & (power_change < 0),
        "rest": ~is_top,
    }


def evaluate_subsets(prediction, baseline, target, subsets, bpm_scale) -> tuple[pd.DataFrame, dict]:
    """Compare model and baseline errors on every subset."""
    rows, horizon_errors = [], {}
    for name, mask in subsets.items():
        model_errors = mae_in_bpm(prediction[mask], target[mask], bpm_scale)
        baseline_errors = mae_in_bpm(baseline[mask], target[mask], bpm_scale)
        horizon_errors[name] = (model_errors, baseline_errors)
        rows.append({
            "subset": name,
            "windows": int(mask.sum()),
            "model_mae": model_errors.mean(),
            "model_mae_60s": model_errors[-1],
            "baseline_mae": baseline_errors.mean(),
            "baseline_mae_60s": baseline_errors[-1],
            "improvement_%": np.round(100 * (1 - model_errors.mean() / baseline_errors.mean()), 2),
        })
    return pd.DataFrame(rows), horizon_errors


def plot_horizon_errors(horizon_errors: dict, seconds_per_step: int, subset_names: list[str]) -> go.Figure:
    """Plot MAE over the forecast horizon for model and baseline per subset."""
    figure = go.Figure()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#7f7f7f"]
    for name, color in zip(subset_names, colors):
        model_errors, baseline_errors = horizon_errors[name]
        horizon = np.arange(1, len(model_errors) + 1) * seconds_per_step
        figure.add_trace(go.Scatter(x=horizon, y=model_errors, name=f"Model – {name}",
                                    line=dict(color=color, width=3)))
        figure.add_trace(go.Scatter(x=horizon, y=baseline_errors, name=f"Baseline – {name}",
                                    line=dict(color=color, width=2, dash="dash")))
    figure.update_layout(
        title="Forecast error for large power changes vs. rest",
        xaxis_title="Forecast horizon [s]", yaxis_title="MAE [bpm]",
        font_size=18, height=650,
    )
    return figure


def bootstrap_improvement(model_errors: np.ndarray, baseline_errors: np.ndarray,
                          n_resamples: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Return the 95% confidence interval of the relative improvement over the baseline."""
    rng = np.random.default_rng(seed)
    n_windows = len(model_errors)
    improvements = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.integers(0, n_windows, n_windows)
        improvements[i] = 1 - model_errors[sample].mean() / baseline_errors[sample].mean()
    return tuple(np.percentile(improvements, [2.5, 97.5]))


def report_improvement(model, past, future, target, subsets: dict[str, np.ndarray], scaler, config: dict) -> None:
    """Print model vs. baseline error and bootstrap CI of the improvement for each subset."""
    prediction = model.predict({"past": past, "future": future}, verbose=0)
    bpm_scale = heart_rate_scale(scaler, config)
    baseline = persistence_forecast(past, config["heart_rate_index"], config["future_steps"])
    model_mae = per_window_mae(prediction, target, bpm_scale)
    baseline_mae = per_window_mae(baseline, target, bpm_scale)

    for name, mask in subsets.items():
        if mask.sum() == 0:
            continue
        model_error = model_mae[mask].mean()
        baseline_error = baseline_mae[mask].mean()
        improvement = 1 - model_error / baseline_error
        low, high = bootstrap_improvement(model_mae[mask], baseline_mae[mask])
        print(
            f"{name:<20} n={mask.sum():>5} | model: {model_error:.2f} bpm | "
            f"baseline: {baseline_error:.2f} bpm | "
            f"improvement: {improvement:.1%} (95% CI: {low:.1%} – {high:.1%})"
        )


def main() -> None:
    model, config, scaler, test_data = load_run(RUN_DIRECTORY)
    past, future, target = test_data["past"], test_data["future"], test_data["target"]
    bpm_scale = heart_rate_scale(scaler, config)

    past_power, future_power = mean_power_watts(past, future, scaler, config)
    subsets = build_subsets(past_power, future_power)
    report_improvement(model, past, future, target, subsets, scaler, config)

    prediction = model.predict({"past": past, "future": future}, verbose=0, batch_size=1024)
    baseline = persistence_forecast(past, config["heart_rate_index"], config["future_steps"])

    summary, horizon_errors = evaluate_subsets(prediction, baseline, target, subsets, bpm_scale)
    print(summary.round(2).to_string(index=False))

    seconds_per_step = 60 // config["future_steps"]
    figure = plot_horizon_errors(horizon_errors, seconds_per_step,
                                 [f"top {TOP_FRACTION:.0%}", "rest"])
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure.show()
    figure.write_html(OUTPUT_DIRECTORY / "large_power_changes.html")


if __name__ == "__main__":
    main()
