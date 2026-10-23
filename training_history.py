from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


def load_training_history(csv_path: str) -> pd.DataFrame:
    """Load the training history from a CSV file."""
    return pd.read_csv(csv_path)


def plot_training_history(history: pd.DataFrame):
    """Plot loss/val_loss and learning rate over epochs in two stacked subplots."""
    # Losses in long format so plotly express creates one line per metric
    loss_df = history.melt(
        id_vars="epoch",
        value_vars=["loss", "val_loss"],
        var_name="metric",
        value_name="value",
    )
    loss_fig = px.line(loss_df, x="epoch", y="value", color="metric", markers=True)
    lr_fig = px.line(history, x="epoch", y="learning_rate", markers=True)
    lr_fig.update_traces(name="learning_rate", showlegend=True)

    # Combine both figures into one with a shared x-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Loss und Validation Loss", "Learning Rate"),
    )
    for trace in loss_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in lr_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_layout(
        title="Trainingsverlauf",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


if __name__ == "__main__":
    history = load_training_history(Path("models") / "2026-10-22_14-13-16_large_seed0" / "training_log.csv")
    plot_training_history(history).show()
