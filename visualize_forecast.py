"""Visualize heart rate forecasts of a saved model on the test split."""

import json
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow import keras

from utils import persistence_forecast, mae_in_bpm, heart_rate_scale, to_original_unit

RUN_DIRECTORY = Path("models") / "2026-10-22_11-16_gru64"
STEP_SECONDS = 5
OUTPUT_DIRECTORY = RUN_DIRECTORY / "figures"


def load_run(run_directory: Path) -> tuple:
    """Load model, scaler, config and test arrays of one training run."""
    model = keras.models.load_model(run_directory / "best_model.keras", compile=False)
    scaler = joblib.load(run_directory / "scaler.joblib")
    with open(run_directory / "config.json", encoding="utf-8") as file:
        config = json.load(file)
    test_data = np.load(run_directory / "test_data.npz")
    return model, scaler, config, test_data


model, scaler, config, test_data = load_run(RUN_DIRECTORY)
past_test, future_test, target_test = test_data["past"], test_data["future"], test_data["target"]
bpm_scale = heart_rate_scale(scaler, config)

SCALED_COLUMNS = config["scaled_columns"]
HEART_RATE_INDEX = config["heart_rate_index"]
PAST_POWER_INDEX = config["past_columns"].index("power")
FUTURE_POWER_INDEX = config["future_columns"].index("power")
FUTURE_STEPS = config["future_steps"]
seconds_ahead = np.arange(1, config["future_steps"] + 1) * STEP_SECONDS


def plot_forecast_window(index: int, history_steps: int = 36, title: str | None = None) -> go.Figure:
    """Plot past heart rate, truth, model and persistence forecast plus power."""
    past_seconds = np.arange(-history_steps + 1, 1) * STEP_SECONDS
    lines = {
        "measured (future)": (target_test[index], dict(color="black")),
        "model": (prediction_test[index], dict(color="crimson", width=3)),
        "persistence": (persistence_test[index], dict(color="gray", dash="dot")),
    }

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.05)
    figure.add_trace(go.Scatter(
        x=past_seconds,
        y=to_original_unit(past_test[index, -history_steps:, HEART_RATE_INDEX], scaler, SCALED_COLUMNS, "heart_rate"),
        name="measured (past)", line_color="gray"), row=1, col=1)
    for name, (values, line) in lines.items():
        figure.add_trace(go.Scatter(x=seconds_ahead, y=to_original_unit(values, scaler, SCALED_COLUMNS, "heart_rate"),
                                    name=name, line=line), row=1, col=1)

    figure.add_trace(go.Scatter(
        x=past_seconds,
        y=to_original_unit(past_test[index, -history_steps:, PAST_POWER_INDEX], scaler, SCALED_COLUMNS, "power"),
        line_color="steelblue", showlegend=False), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=seconds_ahead,
        y=to_original_unit(future_test[index, :, FUTURE_POWER_INDEX], scaler, SCALED_COLUMNS, "power"),
        name="planned power", line=dict(color="steelblue", dash="dash")), row=2, col=1)

    figure.add_vline(x=0, line_dash="dash", line_color="lightgray")
    figure.update_yaxes(title_text="heart rate [bpm]", row=1, col=1)
    figure.update_yaxes(title_text="power [W]", row=2, col=1)
    figure.update_xaxes(title_text="seconds relative to forecast start", row=2, col=1)
    figure.update_layout(template="plotly_white", height=600,
                         title=title or f"Heart rate forecast (test window {index})")
    return figure


def plot_error_by_horizon() -> go.Figure:
    """Compare model and persistence error over the forecast horizon."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=seconds_ahead, y=mae_in_bpm(persistence_test, target_test, bpm_scale),
                                name="persistence", mode="lines+markers",
                                line=dict(color="gray", dash="dot")))
    figure.add_trace(go.Scatter(x=seconds_ahead, y=mae_in_bpm(prediction_test, target_test, bpm_scale),
                                name="GRU model", mode="lines+markers",
                                line=dict(color="crimson", width=3)))
    figure.update_layout(template="plotly_white", title="Forecast error on test data",
                         xaxis_title="seconds ahead", yaxis_title="MAE [bpm]",
                         yaxis_rangemode="tozero")
    return figure


def download_config(filename: str) -> dict:
    """Configure the browser's camera button for presentation-quality PNG export."""
    return {
        "toImageButtonOptions": {
            "format": "png",  # also possible: "svg", "jpeg", "webp"
            "filename": filename,
            "width": 1600,
            "height": 900,  # 16:9 matches PowerPoint slides
            "scale": 2,  # 3200 x 1800 pixels, sharp on a projector
        },
        "displaylogo": False,
    }


if __name__ == "__main__":
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    prediction_test = model.predict({"past": past_test, "future": future_test}, verbose=0)
    persistence_test = persistence_forecast(past_test, HEART_RATE_INDEX, FUTURE_STEPS)

    # Sanity check: must match the numbers printed after training
    print(f"test model: {mae_in_bpm(prediction_test, target_test, bpm_scale).mean():.2f} bpm | "
          f"baseline: {mae_in_bpm(persistence_test, target_test, bpm_scale).mean():.2f} bpm")

    window_errors_bpm = (np.abs(prediction_test - target_test).mean(axis=1)
                         * scaler.scale_[SCALED_COLUMNS.index("heart_rate")])
    power_change = (future_test[:, :, FUTURE_POWER_INDEX].mean(axis=1)
                    - past_test[:, -1, PAST_POWER_INDEX])

    figures = {"power_increase": plot_forecast_window(
        int(np.argmax(power_change)), title="Forecast during strongest power increase")}

    sorted_indices = np.argsort(window_errors_bpm)
    for quantile in [0.25, 0.5, 0.75]:
        index = int(sorted_indices[int(quantile * (len(sorted_indices) - 1))])
        figures[f"typical_p{int(quantile * 100)}"] = plot_forecast_window(
            index, title=f"Typical window ({int(quantile * 100)}th percentile): "
                         f"MAE {window_errors_bpm[index]:.1f} bpm")

    figures["error_by_horizon"] = plot_error_by_horizon()

    for name, figure in figures.items():
        config = download_config(name)
        figure.show(config=config)
        figure.write_html(OUTPUT_DIRECTORY / f"{name}.html", config=config)
