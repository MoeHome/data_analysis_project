import pandas as pd
import numpy as np
import keras
from keras import layers
from pathlib import Path
from datetime import datetime

import joblib
import json
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

from utils import persistence_forecast, mae_in_bpm

df = pd.read_csv("./Dataframes/df_ml.csv", encoding="utf-8", parse_dates=['timestamp'], index_col='timestamp')
df_session = pd.read_csv("./Dataframes/session_data_cleaned.csv", encoding="utf-8",
                         parse_dates=["start_time", "end_time", "start_time_Berlin", "end_time_Berlin"],
                         index_col='source_file')


def drop_rows_after_cutoff(data: pd.DataFrame, source_file: str, cutoff: str) -> pd.DataFrame:
    """Remove all rows of a source file whose timestamp is later than the cutoff (UTC)."""
    cutoff_timestamp = pd.Timestamp(cutoff, tz="UTC")
    rows_to_drop = (data["source_file"] == source_file) & (data.index > cutoff_timestamp)
    return data.loc[~rows_to_drop]


def drop_source_file(data: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Remove all rows belonging to a source file."""
    return data.loc[data["source_file"] != source_file]


# Some values are missing after the cutoff in this ride
df = drop_rows_after_cutoff(df, "2026-06-03-15-30-13.fit", "2026-06-03 14:57:23")

# All heart rate values are missing after the cutoff in this ride
df = drop_rows_after_cutoff(df, "2026-06-17-15-49-01.fit", "2026-06-17 15:29:20")

# Too many values are missing in this ride
df = drop_source_file(df, "2026-02-26-20-51-14.fit")

# Drop all rows where heart_rate is NaN
df = df.dropna(subset=["heart_rate"])

RESAMPLE_INTERVAL = "5s"

TARGET = "heart_rate"
KNOWN_FEATURES = [
    "power", "cadence", "seconds_since_session_start",
    "time_of_day_sin", "time_of_day_cos",
    "had_warmup", "days_since_first_session",
    "day_of_week_sin", "day_of_week_cos",
    "TT_10", "PP_10", "RF_10",
]
MODEL_COLUMNS = KNOWN_FEATURES + [TARGET]

df_resampled = (
    df.groupby("source_file")[MODEL_COLUMNS]
    .resample(RESAMPLE_INTERVAL)  # works because the index is a DatetimeIndex
    .mean()
    .reset_index(level="source_file")
)

'''
print(df_resampled.shape)
print(df_resampled.isna().sum())
print(df_resampled.groupby("source_file").size().describe())  # steps per session
'''

is_missing = df_resampled["power"].isna()

# A new block starts whenever the missing-status changes
block_id = (is_missing != is_missing.shift()).cumsum()

gap_lengths = (
    df_resampled[is_missing]
    .groupby(["source_file", block_id[is_missing]])
    .size()
)

'''
print(f"Number of gaps: {len(gap_lengths)}")
print(f"Sessions with gaps: {gap_lengths.index.get_level_values('source_file').nunique()}")
print(gap_lengths.describe())
print(gap_lengths.value_counts().sort_index())  # gap length in steps (1 step = 5 s)
'''

MAX_INTERPOLATION_STEPS = 6  # 6 steps * 5 s = 30 s
WEATHER_COLUMNS = ["TT_10", "PP_10", "RF_10"]

is_missing = df_resampled["power"].isna()
block_id = (is_missing != is_missing.shift()).cumsum()

# Length of the missing block each row belongs to
gap_length = is_missing.groupby(block_id).transform("size")
is_long_gap = is_missing & (gap_length > MAX_INTERPOLATION_STEPS)

# A new segment starts after each long gap (counted per session)
long_gap_start = is_long_gap & ~is_long_gap.shift(fill_value=False)
segment_number = long_gap_start.groupby(df_resampled["source_file"]).cumsum()

df_resampled["segment_id"] = (
        df_resampled["source_file"].astype(str) + "_" + segment_number.astype(str)
)

# Drop the rows of long gaps
df_clean = df_resampled[~is_long_gap].copy()

# Interpolate short gaps within each segment only
df_clean[MODEL_COLUMNS] = (
    df_clean.groupby("segment_id")[MODEL_COLUMNS]
    .transform(lambda column: column.interpolate(limit_area="inside"))
)

# Weather changes slowly -> fill remaining gaps within the segment
df_clean[WEATHER_COLUMNS] = (
    df_clean.groupby("segment_id")[WEATHER_COLUMNS]
    .transform(lambda column: column.ffill().bfill())
)

'''
print(df_clean.shape)
print(df_clean[MODEL_COLUMNS].isna().sum().sum())  # should be 0
print(f"Segments: {df_clean['segment_id'].nunique()}")
print(df_clean.groupby("segment_id").size().describe())
'''

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Order sessions chronologically by their first timestamp
session_start = (
    df_clean.index.to_series()
    .groupby(df_clean["source_file"])
    .min()
    .sort_values()
)
sessions_in_order = session_start.index

number_of_sessions = len(sessions_in_order)
train_end = int(number_of_sessions * TRAIN_FRACTION)
validation_end = int(number_of_sessions * (TRAIN_FRACTION + VALIDATION_FRACTION))

train_sessions = sessions_in_order[:train_end]
validation_sessions = sessions_in_order[train_end:validation_end]
test_sessions = sessions_in_order[validation_end:]

df_train = df_clean[df_clean["source_file"].isin(train_sessions)]
df_validation = df_clean[df_clean["source_file"].isin(validation_sessions)]
df_test = df_clean[df_clean["source_file"].isin(test_sessions)]

'''
WINDOW_STEPS = 120 + 12  # 10 min history + 1 min forecast (5 s steps)


for name, split in [("train", df_train), ("validation", df_validation), ("test", df_test)]:
    segment_sizes = split.groupby("segment_id").size()
    usable_rows = segment_sizes[segment_sizes >= WINDOW_STEPS].sum()
    print(
        f"{name:<10} sessions: {split['source_file'].nunique():>3} | "
        f"rows: {len(split):>6} | "
        f"period: {split.index.min():%Y-%m-%d} – {split.index.max():%Y-%m-%d} | "
        f"usable rows: {usable_rows / len(split):.1%}"
    )
'''

SCALED_COLUMNS = [
    "power", "cadence", "heart_rate",
    "TT_10", "PP_10", "RF_10",
    "seconds_since_session_start", "days_since_first_session",
]


def scale_split(split: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of the split with SCALED_COLUMNS standardized by the given scaler."""
    scaled = split.copy()
    scaled[SCALED_COLUMNS] = scaler.transform(split[SCALED_COLUMNS])
    return scaled


scaler = StandardScaler()
scaler.fit(df_train[SCALED_COLUMNS])  # fit on training data only

df_train_scaled = scale_split(df_train, scaler)
df_validation_scaled = scale_split(df_validation, scaler)
df_test_scaled = scale_split(df_test, scaler)

# Persist the scaler: needed later to convert predictions back to watts
Path("./Models").mkdir(exist_ok=True)
joblib.dump(scaler, "./Models/feature_scaler.joblib")

summary = pd.DataFrame({
    "train_mean": df_train_scaled[SCALED_COLUMNS].mean(),
    "validation_mean": df_validation_scaled[SCALED_COLUMNS].mean(),
    "test_mean": df_test_scaled[SCALED_COLUMNS].mean(),
    "train_std": df_train_scaled[SCALED_COLUMNS].std(),
    "validation_std": df_validation_scaled[SCALED_COLUMNS].std(),
    "test_std": df_test_scaled[SCALED_COLUMNS].std(),
}).round(2)

'''
with pd.option_context("display.max_columns", None, "display.width", None):
    print(summary)
'''

df_compare = pd.concat([
    df_train_scaled.assign(split="train"),
    df_validation_scaled.assign(split="validation"),
    df_test_scaled.assign(split="test"),
])

fig = px.histogram(
    df_compare, x="TT_10", color="split",
    barmode="overlay", histnorm="probability density", opacity=0.6,
    title="Temperature distribution per split (standardized)",
)
fig.show()

PAST_STEPS = 120  # 10 min history
FUTURE_STEPS = 12  # 1-min forecast
WINDOW_STEPS = PAST_STEPS + FUTURE_STEPS

PAST_COLUMNS = [
    "power", "cadence", "heart_rate",
    "TT_10", "PP_10", "RF_10",
    "seconds_since_session_start", "had_warmup",
    "day_of_week_sin", "day_of_week_cos",
    "time_of_day_sin", "time_of_day_cos",
]
FUTURE_COLUMNS = [
    "power", "cadence",
    "seconds_since_session_start",
    "time_of_day_sin", "time_of_day_cos",
]
TARGET_COLUMN = "heart_rate"


def create_windows(
        split: pd.DataFrame, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut every segment into (past, future, target) windows.

    Returns arrays of shape
    past:   (samples, PAST_STEPS, len(PAST_COLUMNS))
    future: (samples, FUTURE_STEPS, len(FUTURE_COLUMNS))
    target: (samples, FUTURE_STEPS)
    """
    past_windows, future_windows, target_windows = [], [], []

    for _, segment in split.groupby("segment_id"):
        if len(segment) < WINDOW_STEPS:
            continue  # too short for a single window

        segment = segment.sort_index()
        past_values = segment[PAST_COLUMNS].to_numpy(dtype=np.float32)
        future_values = segment[FUTURE_COLUMNS].to_numpy(dtype=np.float32)
        target_values = segment[TARGET_COLUMN].to_numpy(dtype=np.float32)

        window_starts = np.arange(0, len(segment) - WINDOW_STEPS + 1, stride)
        past_indices = window_starts[:, None] + np.arange(PAST_STEPS)
        future_indices = window_starts[:, None] + PAST_STEPS + np.arange(FUTURE_STEPS)

        past_windows.append(past_values[past_indices])
        future_windows.append(future_values[future_indices])
        target_windows.append(target_values[future_indices])

    return (
        np.concatenate(past_windows),
        np.concatenate(future_windows),
        np.concatenate(target_windows),
    )


TRAIN_STRIDE = 2
EVALUATION_STRIDE = 12

past_train, future_train, target_train = create_windows(df_train_scaled, TRAIN_STRIDE)
past_validation, future_validation, target_validation = create_windows(df_validation_scaled, EVALUATION_STRIDE)
past_test, future_test, target_test = create_windows(df_test_scaled, EVALUATION_STRIDE)

POWER_INDEX = PAST_COLUMNS.index("power")
HEART_RATE_INDEX = PAST_COLUMNS.index("heart_rate")

# 1) Consecutive windows are shifted by exactly the stride
past_is_shifted = np.allclose(past_train[1, :-TRAIN_STRIDE], past_train[0, TRAIN_STRIDE:])

# 2) The future of window 0 continues seamlessly into the past of window 1
future_is_contiguous = np.allclose(
    past_train[1, -TRAIN_STRIDE:, POWER_INDEX],
    future_train[0, :TRAIN_STRIDE, FUTURE_COLUMNS.index("power")],
)

# 3) The target is the heart rate directly following the past window
target_is_contiguous = np.allclose(
    past_train[1, -TRAIN_STRIDE:, HEART_RATE_INDEX],
    target_train[0, :TRAIN_STRIDE],
)

'''
print(past_is_shifted, future_is_contiguous, target_is_contiguous)
'''

FUTURE_STEPS = target_train.shape[1]  # 12 steps of 5 s = 60 s

heart_rate_std = scaler.scale_[SCALED_COLUMNS.index("heart_rate")]
print(f"heart_rate std: {heart_rate_std:.2f} bpm")

for name, past, target in [
    ("validation", past_validation, target_validation),
    ("test", past_test, target_test),
]:
    step_errors = mae_in_bpm(persistence_forecast(past, HEART_RATE_INDEX, FUTURE_STEPS), target, heart_rate_std)
    print(
        f"{name:<10} MAE overall: {step_errors.mean():.2f} bpm | "
        f"after 5 s: {step_errors[0]:.2f} | after 60 s: {step_errors[-1]:.2f}"
    )

PAST_STEPS, PAST_FEATURES = past_train.shape[1:]
_, FUTURE_FEATURES = future_train.shape[1:]


def build_forecast_model(
        gru_units: int = 64,
        dense_units: int = 64,
        dropout_rate: float = 0.2,
) -> keras.Model:
    """GRU encoder over the past session plus the planned future inputs.

    The network predicts the change relative to the last observed heart rate
    (residual formulation), so it starts out close to the persistence baseline.
    """
    past_input = keras.Input(shape=(PAST_STEPS, PAST_FEATURES), name="past")
    future_input = keras.Input(shape=(FUTURE_STEPS, FUTURE_FEATURES), name="future")

    encoded_past = layers.GRU(gru_units, name="past_encoder")(past_input)
    flat_future = layers.Flatten(name="future_flat")(future_input)

    combined = layers.Concatenate(name="combined")([encoded_past, flat_future])
    hidden = layers.Dense(dense_units, activation="relu")(combined)
    hidden = layers.Dropout(dropout_rate)(hidden)
    heart_rate_delta = layers.Dense(FUTURE_STEPS, name="heart_rate_delta")(hidden)

    last_heart_rate = past_input[:, -1, HEART_RATE_INDEX: HEART_RATE_INDEX + 1]
    forecast = layers.Add(name="heart_rate_forecast")([heart_rate_delta, last_heart_rate])

    return keras.Model(inputs=[past_input, future_input], outputs=forecast)


model = build_forecast_model()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
model.summary()

RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_gru64"
RUN_DIRECTORY = Path("models") / RUN_NAME
RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    keras.callbacks.ModelCheckpoint(
        filepath=RUN_DIRECTORY / "best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    keras.callbacks.CSVLogger(RUN_DIRECTORY / "training_log.csv"),
]

history = model.fit(
    x={"past": past_train, "future": future_train},
    y=target_train,
    validation_data=({"past": past_validation, "future": future_validation}, target_validation),
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=2,
)

for name, past, future, target in [
    ("validation", past_validation, future_validation, target_validation),
    ("test", past_test, future_test, target_test),
]:
    prediction = model.predict({"past": past, "future": future}, verbose=0)
    model_errors = mae_in_bpm(prediction, target, heart_rate_std)
    baseline_errors = mae_in_bpm(persistence_forecast(past, HEART_RATE_INDEX, FUTURE_STEPS), target, heart_rate_std)
    print(
        f"{name:<10} model: {model_errors.mean():.2f} bpm "
        f"(5 s: {model_errors[0]:.2f} | 60 s: {model_errors[-1]:.2f}) | "
        f"baseline: {baseline_errors.mean():.2f} bpm"
    )

figure = go.Figure()
figure.add_trace(go.Scatter(y=history.history["loss"], name="train"))
figure.add_trace(go.Scatter(y=history.history["val_loss"], name="validation"))
figure.add_hline(
    y=3.91 / heart_rate_std, line_dash="dash", annotation_text="persistence (validation)"
)
figure.update_layout(
    title="Training curve (MAE, standardized)",
    xaxis_title="epoch",
    yaxis_title="MAE",
)
figure.show()


def evaluate_split(past: np.ndarray, future: np.ndarray, target: np.ndarray) -> dict:
    """Return model and baseline MAE in bpm for one data split."""
    prediction = model.predict({"past": past, "future": future}, verbose=0)
    model_errors = mae_in_bpm(prediction, target, heart_rate_std)
    baseline_errors = mae_in_bpm(persistence_forecast(past, HEART_RATE_INDEX, FUTURE_STEPS), target, heart_rate_std)
    return {
        "model_mae_bpm": round(float(model_errors.mean()), 3),
        "model_mae_60s_bpm": round(float(model_errors[-1]), 3),
        "baseline_mae_bpm": round(float(baseline_errors.mean()), 3),
    }


def save_run_artifacts(run_directory: Path) -> None:
    """Store everything needed to reproduce predictions with the saved model."""
    joblib.dump(scaler, run_directory / "scaler.joblib")

    run_config = {
        "past_columns": PAST_COLUMNS,
        "future_columns": FUTURE_COLUMNS,
        "scaled_columns": SCALED_COLUMNS,
        "past_steps": int(PAST_STEPS),
        "future_steps": int(FUTURE_STEPS),
        "heart_rate_index": HEART_RATE_INDEX,
        "hyperparameters": {"gru_units": 64, "dense_units": 64, "dropout_rate": 0.2},
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "validation": evaluate_split(past_validation, future_validation, target_validation),
    }
    with open(run_directory / "config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    np.savez_compressed(
        run_directory / "test_data.npz",
        past=past_test,
        future=future_test,
        target=target_test,
    )


save_run_artifacts(RUN_DIRECTORY)
print(f"Saved run to {RUN_DIRECTORY}")
