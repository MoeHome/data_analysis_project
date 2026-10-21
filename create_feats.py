import pandas as pd
import numpy as np

df = pd.read_csv("./Dataframes/processed_data.csv", encoding="utf-8",
                 parse_dates=["timestamp", "MESS_DATUM"], index_col='timestamp')
df_session = pd.read_csv("./Dataframes/session_data_cleaned.csv", encoding="utf-8",
                         parse_dates=["start_time", "end_time", "start_time_Berlin", "end_time_Berlin"],
                         index_col='source_file')

df["timestamp_berlin"] = pd.to_datetime(df["timestamp_berlin"], utc=True).dt.tz_convert("Europe/Berlin")

# Look up the start time of the matching session for each row (NaT where no session matches)
session_start_times = df["source_file"].map(df_session["start_time"])

# Time elapsed since the start of the session
time_since_session_start = df.index - session_start_times
df["seconds_since_session_start"] = time_since_session_start.dt.total_seconds()

# Make sure sessions are ordered chronologically
df_session = df_session.sort_values("start_time")

warmup_threshold_in_seconds = pd.Timedelta(minutes=20).total_seconds()
max_gap_in_seconds = pd.Timedelta(weeks=2).total_seconds()

# Seconds between the end of the previous session and the start of the current one
df_session["time_since_last_session"] = (
        df_session["start_time"] - df_session["end_time"].shift(1)
).dt.total_seconds()

df_session["time_since_last_session"] = df_session["time_since_last_session"].fillna(max_gap_in_seconds)

gap = df_session["time_since_last_session"]

# Binary flag: was there a session within the last 60 minutes? (= warmup already done)
df_session["had_warmup"] = (gap < warmup_threshold_in_seconds).astype(int)

df["had_warmup"] = df["source_file"].map(df_session["had_warmup"])

df_session["days_since_first_session"] = (
                                                 df_session["start_time"] - df_session["start_time"].min()
                                         ).dt.total_seconds() / pd.Timedelta(days=1).total_seconds()

df["days_since_first_session"] = df["source_file"].map(df_session["days_since_first_session"])

df.to_csv("./Dataframes/df_with_all_information.csv")

columns = ["source_file", "cadence", "heart_rate", "power", "TT_10", "PP_10", "RF_10", "seconds_since_session_start", "had_warmup",
           "days_since_first_session"]
df_ml = df[columns].copy()

day_number = df.index.dayofweek  # Monday=0, Sunday=6

df_ml["day_of_week_sin"] = np.sin(2 * np.pi * day_number / 7)
df_ml["day_of_week_cos"] = np.cos(2 * np.pi * day_number / 7)

seconds_per_day = 24 * 60 * 60

seconds_since_midnight = (
    df["timestamp_berlin"].dt.hour * 3600
    + df["timestamp_berlin"].dt.minute * 60
    + df["timestamp_berlin"].dt.second
)

df_ml["time_of_day_sin"] = np.sin(2 * np.pi * seconds_since_midnight / seconds_per_day)
df_ml["time_of_day_cos"] = np.cos(2 * np.pi * seconds_since_midnight / seconds_per_day)

df_ml.to_csv("./Dataframes/df_ml.csv")
