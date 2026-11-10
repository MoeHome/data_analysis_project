import pandas as pd
import plotly.express as px

METRICS = ["heart_rate", "power", "cadence"]
METRIC_COLORS = {
    "heart_rate": "red",
    "power": "blue",
    "cadence": "green",
}

# Use the first column (time) as index; adjust if your time column differs
df = pd.read_csv(
    "Dataframes/df_with_all_information.csv",
    index_col=0,
    parse_dates=True,
)


def plot_ride_metrics(
        df: pd.DataFrame,
        ride_id: str,
        metrics: list[str] = METRICS,
        metric_colors: dict[str, str] = METRIC_COLORS,
):
    """Plot heart rate, cadence and power of a single ride over time.

    The time information is expected to be the index of the DataFrame.

    Args:
        df: DataFrame containing multiple rides with second-level data,
            indexed by time.
        ride_id: Value of `source_file` identifying the ride.
        metrics: Columns to plot, one subplot per metric.
        metric_colors: Mapping of metric name to line color.

    Returns:
        plotly.graph_objects.Figure
    """
    ride_df = df.loc[df["source_file"] == ride_id].sort_index()

    if ride_df.empty:
        raise ValueError(f"No data found for source_file '{ride_id}'")

    # Name the index so it can be used as an id column after reset_index
    time_column = ride_df.index.name or "time"
    ride_df = ride_df.rename_axis(time_column).reset_index()

    # Convert to long format: one row per (time, metric)
    long_df = ride_df.melt(
        id_vars=[time_column],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    fig = px.line(
        long_df,
        x=time_column,
        y="value",
        color="metric",
        facet_row="metric",
        color_discrete_map=metric_colors,
        category_orders={"metric": metrics},  # keep subplot order as given
        title=f"Ride: {ride_id}",
        labels={time_column: "Time", "value": ""},
    )

    # Independent y-axes per subplot, clean facet labels
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False, height=700)

    return fig


# Usage
fig = plot_ride_metrics(df, ride_id="2026-08-04-17-09-50.fit")
fig.show()

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class Block:
    """A box in the architecture diagram."""
    key: str
    label: str
    x: float
    y: float
    color: str


BOX_WIDTH = 1.8
BOX_HEIGHT = 0.9

INPUT_COLOR = "#72B7B2"
LAYER_COLOR = "#4C78A8"
RESIDUAL_COLOR = "#F58518"
OUTPUT_COLOR = "#E45756"


def add_block(figure: go.Figure, block: Block) -> None:
    """Draw a single rounded-looking box with a centered label."""
    figure.add_shape(
        type="rect",
        x0=block.x - BOX_WIDTH / 2, x1=block.x + BOX_WIDTH / 2,
        y0=block.y - BOX_HEIGHT / 2, y1=block.y + BOX_HEIGHT / 2,
        fillcolor=block.color, line=dict(color="white", width=2), layer="below",
    )
    figure.add_annotation(
        x=block.x, y=block.y, text=block.label, showarrow=False,
        font=dict(color="white", size=13),
    )


def add_arrow(figure: go.Figure, source: Block, target: Block, color: str = "#555", dashed: bool = False) -> None:
    """Draw an arrow from the right edge of source to the left edge of target."""
    figure.add_annotation(
        x=target.x - BOX_WIDTH / 2, y=target.y,
        ax=source.x + BOX_WIDTH / 2, ay=source.y,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor=color,
    )
    if dashed:  # dashed line drawn underneath to mark the skip connection
        figure.add_shape(
            type="line",
            x0=source.x + BOX_WIDTH / 2, y0=source.y,
            x1=target.x - BOX_WIDTH / 2, y1=target.y,
            line=dict(color=color, width=2, dash="dash"), layer="below",
        )


Point = tuple[float, float]


def add_elbow_arrow(figure: go.Figure, waypoints: list[Point], color: str, dashed: bool = True) -> None:
    """Draw an orthogonal (elbow) arrow through the given waypoints, arrowhead at the last point."""
    path = "M " + " L ".join(f"{x},{y}" for x, y in waypoints)
    figure.add_shape(
        type="path", path=path,
        line=dict(color=color, width=2, dash="dash" if dashed else "solid"),
        layer="below",
    )

    # Short final segment only, so the arrowhead does not hide the dashed line.
    (x_prev, y_prev), (x_end, y_end) = waypoints[-2], waypoints[-1]
    head_length = 0.15
    dx, dy = x_end - x_prev, y_end - y_prev
    length = (dx ** 2 + dy ** 2) ** 0.5
    figure.add_annotation(
        x=x_end, y=y_end,
        ax=x_end - dx / length * head_length, ay=y_end - dy / length * head_length,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor=color,
    )


def plot_architecture(blocks: list[Block], edges: list[tuple[str, str]], title: str) -> go.Figure:
    """Draw the main data flow plus the residual path routed around the layer boxes."""
    positions = {block.key: block for block in blocks}
    figure = go.Figure()

    for block in blocks:
        add_block(figure, block)
    for source_key, target_key in edges:
        add_arrow(figure, positions[source_key], positions[target_key], GREY)

    past, last_hr, add = positions["past"], positions["last_hr"], positions["add"]
    half_w, half_h = BOX_WIDTH / 2, BOX_HEIGHT / 2

    # Past input (top edge) -> up -> right -> Last measured HR (left edge)
    add_elbow_arrow(figure, [
        (past.x, past.y + half_h),
        (past.x, last_hr.y),
        (last_hr.x - half_w, last_hr.y),
    ], RESIDUAL_COLOR)

    # Last measured HR (right edge) -> right -> down -> Add (top edge)
    add_elbow_arrow(figure, [
        (last_hr.x + half_w, last_hr.y),
        (add.x, last_hr.y),
        (add.x, add.y + half_h),
    ], RESIDUAL_COLOR)

    x_values = [block.x for block in blocks]
    y_values = [block.y for block in blocks]
    figure.update_xaxes(visible=False, range=[min(x_values) - 1.3, max(x_values) + 1.3])
    figure.update_yaxes(visible=False, range=[min(y_values) - 1, max(y_values) + 0.8])
    figure.update_layout(
        title=title, plot_bgcolor="white", width=1400, height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


blocks = [
    Block("past", "Past input<br>(time steps × features)", 0, 0.8, INPUT_COLOR),
    Block("future", "Future input<br>planned power", 0, -0.8, INPUT_COLOR),
    Block("gru", "GRU (128)<br>past encoder", 2.6, 0.8, LAYER_COLOR),
    Block("flatten", "Flatten<br>future values", 2.6, -0.8, LAYER_COLOR),
    Block("concat", "Concatenate", 5.2, 0, LAYER_COLOR),
    Block("dense", "Dense (128, ReLU)<br>+ Dropout", 7.8, 0, LAYER_COLOR),
    Block("delta", "Dense (12)<br>HR change Δ", 10.4, 0, LAYER_COLOR),
    Block("last_hr", "Last measured HR", 7.8, 2.2, RESIDUAL_COLOR),
    Block("add", "Add<br>HR forecast<br>12 × 5 s", 13.0, 0, OUTPUT_COLOR),
]

GREY = "#555"
edges = [
    ("past", "gru"),
    ("future", "flatten"),
    ("gru", "concat"),
    ("flatten", "concat"),
    ("concat", "dense"),
    ("dense", "delta"),
    ("delta", "add"),
]

figure = plot_architecture(blocks, edges, "Heart rate forecasting model (residual GRU)")
figure.show()
