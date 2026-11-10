from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRIC_COLUMNS = ["epoch", "loss", "val_loss", "learning_rate"]


def load_training_history(csv_path: Path, one_based: bool = True) -> pd.DataFrame:
    """Load and clean the training history; optionally shift epochs to start at 1."""
    history = pd.read_csv(csv_path)
    history[METRIC_COLUMNS] = history[METRIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    history = history.dropna(subset=METRIC_COLUMNS).sort_values("epoch").reset_index(drop=True)
    history["epoch"] = history["epoch"].astype(int)

    if one_based and history["epoch"].min() == 0:
        history["epoch"] += 1
    return history


def find_best_epoch(history: pd.DataFrame) -> pd.Series:
    """Return the row of the epoch with the lowest validation loss."""
    return history.loc[history["val_loss"].idxmin()]


def plot_training_history(history: pd.DataFrame) -> go.Figure:
    """Plot training/validation loss and learning rate, marking the best epoch."""
    epochs = history["epoch"]
    best = find_best_epoch(history)
    best_epoch = int(best["epoch"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("Loss", "Lernrate"),
    )

    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines+markers",
                             name="Training", connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers",
                             name="Validierung", connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["learning_rate"], mode="lines+markers",
                             line=dict(color="gray", shape="hv"), connectgaps=True,
                             showlegend=False), row=2, col=1)

    for row in (1, 2):
        fig.add_vline(x=best_epoch, line_dash="dash", line_color="green", row=row, col=1)
    fig.add_annotation(x=best_epoch, y=best["val_loss"],
                       text=f"Bestes Modell (Epoche {best_epoch})",
                       showarrow=True, arrowhead=2, ax=40, ay=-40, row=1, col=1)

    x_range = [epochs.min() - 0.5, epochs.max() + 0.5]
    fig.update_xaxes(range=x_range, tickmode="array", tickvals=epochs.tolist())
    fig.update_xaxes(title_text="Epoche", row=2, col=1)
    fig.update_yaxes(title_text="MAE (skaliert)", row=1, col=1)
    fig.update_yaxes(title_text="Lernrate", exponentformat="e", row=2, col=1)
    fig.update_layout(template="plotly_white", height=550,
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


if __name__ == "__main__":
    log_path = Path("models") / "2026-10-22_14-13-16_large_seed0" / "training_log.csv"
    history = load_training_history(log_path)
    plot_training_history(history).show()