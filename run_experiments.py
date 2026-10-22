"""Define configurations, train each with several seeds and compare results."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from config import ExperimentConfig, ModelConfig, DataConfig
from data_preparation import prepare_scaled_splits, build_windowed_data, WindowedData
from training import run_experiment

from tqdm.auto import tqdm

SEEDS = [0, 1, 2]
RESULTS_FILE = Path("models/experiment_results.csv")

baseline = ExperimentConfig(name="baseline")

EXPERIMENTS = [
    baseline,
    replace(baseline, name="large", model=ModelConfig(gru_units=128, dense_units=128)),
    replace(baseline, name="small", model=ModelConfig(gru_units=32, dense_units=32)),
    replace(baseline, name="past_20min", data=replace(baseline.data, past_steps=240)),
]


def append_result(record: dict) -> None:
    """Append one run to the results CSV so nothing is lost if the script crashes."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(RESULTS_FILE, mode="a", index=False,
                                  header=not RESULTS_FILE.exists())


def plot_comparison(summary: pd.DataFrame) -> go.Figure:
    """Bar chart of mean validation MAE with seed standard deviation as error bars."""
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=summary.index, y=summary["mean"],
        error_y={"type": "data", "array": summary["std"]}, name="model",
    ))
    figure.add_trace(go.Scatter(
        x=summary.index, y=summary["baseline"], mode="markers",
        marker={"symbol": "line-ew", "size": 30, "line": {"width": 3}}, name="persistence",
    ))
    figure.update_layout(title="Validation MAE per configuration (mean ± std over seeds)",
                         yaxis_title="MAE [bpm]")
    return figure


def get_cached_windows(
        window_cache: dict[DataConfig, WindowedData],
        splits,
        data_config: DataConfig,
        run_bar: tqdm,
) -> WindowedData:
    """Build windows for a data config only once and reuse them afterwards."""
    if data_config not in window_cache:
        run_bar.set_description("Experiments: building windows")
        window_cache[data_config] = build_windowed_data(splits, data_config)
    return window_cache[data_config]


def main() -> None:
    tqdm.write("Preparing data (load, clean, split, scale) ...")
    splits = prepare_scaled_splits()  # expensive part, runs only once
    window_cache: dict[DataConfig, WindowedData] = {}
    records = []

    runs = [(config, seed) for config in EXPERIMENTS for seed in SEEDS]

    with tqdm(total=len(runs), desc="Experiments", unit="run", position=0) as run_bar:
        for config, seed in runs:
            data = get_cached_windows(window_cache, splits, config.data, run_bar)

            run_bar.set_description(f"Experiments: {config.name} seed {seed}")
            record = run_experiment(config, data, splits.scaler, seed)
            append_result(record)
            records.append(record)

            tqdm.write(f"{config.name:<12} seed {seed}: "
                       f"{record['validation_model_mae_bpm']:.2f} bpm "
                       f"(epoch {record['best_epoch']})")
            run_bar.set_postfix(last_mae=f"{record['validation_model_mae_bpm']:.2f}")
            run_bar.update(1)

    results = pd.DataFrame(records)
    summary = results.groupby("experiment").agg(
        mean=("validation_model_mae_bpm", "mean"),
        std=("validation_model_mae_bpm", "std"),
        baseline=("validation_baseline_mae_bpm", "mean"),
        train_loss=("train_loss_at_best", "mean"),
        val_loss=("val_loss_best", "mean"),
    ).sort_values("mean")
    print(summary.round(3))
    plot_comparison(summary).show()


if __name__ == "__main__":
    main()
