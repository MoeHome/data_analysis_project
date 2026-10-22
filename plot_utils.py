import pandas as pd
import plotly.express as px

METRICS = ["heart_rate", "cadence", "power"]
df = pd.read_csv("Dataframes/df_with_all_information.csv")


def plot_ride_metrics(
        df: pd.DataFrame,
        ride_id: str,
        metrics: list[str] = METRICS,
):
    """Plot heart rate, cadence and power of a single ride over time.

    The time information is expected to be the index of the DataFrame.

    Args:
        df: DataFrame containing multiple rides with second-level data,
            indexed by time.
        ride_id: Value of `source_file` identifying the ride.
        metrics: Columns to plot, one subplot per metric.

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
